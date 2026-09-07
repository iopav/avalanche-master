from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .data import build_dataset_bundle
from .final_hyperparameters import get_final_hyperparameters
from .flops import (
    PhaseFlopProfiler,
    profile_single_forward,
    summarize_auxiliary_nonflop_ops,
    summarize_learning_flops,
)
from .intransigence import fill_from_joint_run
from .lr_search import search_unit_path
from .models import BACKBONES, build_feature_extractor
from .order_seed_registry import ORDERS_BY_DATASET, SEEDS_BY_DATASET
from .output import atomic_write_json
from .registry import DATASETS
from .runner import _evaluate_experience, _sync, set_determinism
from .search_schema import STATUS_COMPLETED, STATUS_FAILED, STATUS_PENDING, timestamp
from .storage import ByteCounter


class JointClassifier(nn.Module):
    """Ordinary single-backbone classifier used only by joint learning."""

    def __init__(
        self,
        backbone: str,
        dataset: str,
        model_input_shape: tuple[int, ...],
        num_classes: int,
    ):
        super().__init__()
        self.feature_extractor = build_feature_extractor(
            backbone, dataset, model_input_shape
        )
        self.classifier = nn.Linear(
            self.feature_extractor.feature_dim, int(num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


def joint_run_path(
    joint_root: Path, dataset: str, method: str, order_id: int, seed: int
) -> Path:
    name = f"joint_{dataset}_{method}__order-{order_id}__seed-{seed:03d}.json"
    return Path(joint_root) / method / f"order{order_id}" / name


def _optimizer(
    model: nn.Module, parameters: dict[str, Any], learning_rate: float
) -> torch.optim.Optimizer:
    shared = {
        "lr": float(learning_rate),
        "weight_decay": float(parameters["weight_decay"]),
        "foreach": bool(parameters["foreach"]),
    }
    if parameters["optimizer"] == "SGD":
        return torch.optim.SGD(
            model.parameters(), momentum=float(parameters["momentum"]), **shared
        )
    if parameters["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), **shared)
    raise ValueError(f"Unsupported joint optimizer: {parameters['optimizer']}")


def _storage(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    parameter_counter = ByteCounter()
    parameter_bytes = sum(parameter_counter.tensor(value) for value in model.parameters())
    auxiliary_counter = ByteCounter()
    buffer_bytes = sum(auxiliary_counter.tensor(value) for value in model.buffers())
    optimizer_bytes = auxiliary_counter.value(optimizer.state, include_scalars=True)
    total = int(parameter_bytes + buffer_bytes + optimizer_bytes)
    return {
        "model_parameter_bytes": int(parameter_bytes),
        "model_buffer_bytes": int(buffer_bytes),
        "optimizer_state_bytes": int(optimizer_bytes),
        "total_bytes": total,
        "total_mib": float(total / (2**20)),
    }


def _identity(
    exp_name: str,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    backbone: str,
    best_lr: float,
) -> dict[str, Any]:
    return {
        "schema": "pp2-joint-learning-v2",
        "exp_name": exp_name,
        "dataset": dataset,
        "method": method,
        "order": int(order_id),
        "seed": int(seed),
        "backbone": backbone,
        "best_lr": float(best_lr),
    }


def _read_completed(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    differences = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if differences:
        raise ValueError(f"Existing joint run differs: {differences}")
    if payload.get("status") == STATUS_COMPLETED:
        validate_joint_result(payload)
        return payload
    return None


def validate_joint_result(payload: dict[str, Any]) -> None:
    required = {
        "schema", "exp_name", "dataset", "method", "order", "seed", "backbone",
        "best_lr", "status", "started_at", "finished_at", "config", "tasks",
        "task_groups", "accuracy_matrix_lower_triangular", "accuracy_by_task",
        "stage_final_epoch_train_loss", "train_samples_by_stage",
        "final_epoch_train_loss",
        "training_runtime", "training_operations", "persistent_storage",
        "inference", "working_memory_diagnostic",
    }
    if set(payload) != required or payload.get("schema") != "pp2-joint-learning-v2":
        raise ValueError("Joint result schema fields differ")
    if payload.get("status") != STATUS_COMPLETED:
        raise RuntimeError("Joint result is incomplete")
    tasks = int(payload["tasks"])
    accuracy = payload["accuracy_by_task"]
    if not isinstance(accuracy, list) or len(accuracy) != tasks or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= float(value) <= 1.0
        for value in accuracy
    ):
        raise ValueError("Joint task accuracies are invalid")
    task_groups = payload["task_groups"]
    if (
        not isinstance(task_groups, list)
        or len(task_groups) != tasks
        or any(not isinstance(group, list) or not group for group in task_groups)
    ):
        raise ValueError("Joint task groups are invalid")
    matrix = payload["accuracy_matrix_lower_triangular"]
    if not isinstance(matrix, list) or len(matrix) != tasks:
        raise ValueError("Joint accuracy matrix task dimension is invalid")
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != tasks:
            raise ValueError("Joint accuracy matrix row dimension is invalid")
        for column_index, value in enumerate(row):
            if column_index > row_index:
                if value is not None:
                    raise ValueError("Joint accuracy matrix upper triangle must be null")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError("Joint accuracy matrix lower triangle is invalid")
        if not math.isclose(float(row[row_index]), float(accuracy[row_index]), abs_tol=1e-12):
            raise ValueError("Joint accuracy_by_task must equal the matrix diagonal")
    stage_losses = payload["stage_final_epoch_train_loss"]
    stage_samples = payload["train_samples_by_stage"]
    if (
        not isinstance(stage_losses, list)
        or len(stage_losses) != tasks
        or not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in stage_losses)
    ):
        raise ValueError("Joint stage losses are invalid")
    if (
        not isinstance(stage_samples, list)
        or len(stage_samples) != tasks
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in stage_samples)
        or any(left >= right for left, right in zip(stage_samples, stage_samples[1:]))
    ):
        raise ValueError("Joint cumulative train sample counts are invalid")
    if not math.isclose(
        float(payload["final_epoch_train_loss"]), float(stage_losses[-1]), abs_tol=1e-12
    ):
        raise ValueError("Joint final loss differs from the final stage loss")
    latency = payload.get("inference", {}).get("final_latency_ms_per_sample")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency <= 0
    ):
        raise ValueError("Joint inference latency is invalid")
    if not math.isfinite(float(payload["final_epoch_train_loss"])):
        raise ValueError("Joint final train loss is invalid")
    operations = payload["training_operations"]
    for key in ("overall_learning_flops", "core_training_flops", "learning_auxiliary_flops", "single_sample_forward_flops"):
        if isinstance(operations.get(key), bool) or not isinstance(operations.get(key), int) or operations[key] < 0:
            raise ValueError(f"Joint {key} is invalid")
    if operations["overall_learning_flops"] != operations["core_training_flops"] + operations["learning_auxiliary_flops"]:
        raise ValueError("Joint learning FLOP components do not sum")
    storage = payload["persistent_storage"]
    if storage.get("total_bytes") != sum(
        storage.get(key, -1)
        for key in ("model_parameter_bytes", "model_buffer_bytes", "optimizer_state_bytes")
    ):
        raise ValueError("Joint storage components do not sum")


def run_joint_unit(
    *,
    dataset_root: Path,
    search_root: Path,
    joint_root: Path,
    exp_name: str,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    device: torch.device,
    backbone: str,
    parameter_overrides: dict[str, Any] | None = None,
    data_role: str = "formal",
) -> Path:
    search_path = search_unit_path(search_root, method, order_id, seed)
    if not search_path.is_file():
        raise FileNotFoundError(search_path)
    search = json.loads(search_path.read_text(encoding="utf-8"))
    if search.get("status") != STATUS_COMPLETED:
        raise RuntimeError(f"Search unit is incomplete: {search_path}")
    expected_search = {
        "exp_name": exp_name,
        "dataset": dataset,
        "method": method,
        "order": int(order_id),
        "seed": int(seed),
        "backbone": backbone,
    }
    differences = {
        key: (search.get(key), value)
        for key, value in expected_search.items()
        if search.get(key) != value
    }
    if differences:
        raise ValueError(f"Search identity differs from joint request: {differences}")

    best_lr = float(search["best_lr"])
    path = joint_run_path(joint_root, dataset, method, order_id, seed)
    identity = _identity(
        exp_name, dataset, method, order_id, seed, backbone, best_lr
    )
    if _read_completed(path, identity) is not None:
        fill_from_joint_run(search_path, path)
        return path

    parameters = get_final_hyperparameters(
        dataset,
        method,
        parameter_overrides,
        require_locked=data_role == "formal",
    )
    config = {
        "source_search_json": str(search_path.resolve()),
        "optimizer": parameters["optimizer"],
        "learning_rate": best_lr,
        "momentum": parameters["momentum"],
        "weight_decay": parameters["weight_decay"],
        "foreach": parameters["foreach"],
        "train_mb_size": parameters["train_mb_size"],
        "eval_mb_size": parameters["eval_mb_size"],
        "num_workers": parameters["num_workers"],
        "epochs": parameters["epochs_per_experience"],
        "training_scope": "stage k uses the cumulative train data from ordered tasks 1..k",
        "strategy": "one from-scratch ordinary backbone and seen-class linear head per stage",
        "reference_accuracy": "lower-triangular matrix; intransigence uses its diagonal",
        "data_role": data_role,
    }
    pending = {
        **identity,
        "status": STATUS_PENDING,
        "started_at": timestamp(),
        "finished_at": None,
        "config": config,
    }
    atomic_write_json(path, pending)

    try:
        set_determinism(seed)
        data = build_dataset_bundle(
            Path(dataset_root), DATASETS[dataset], order_id, data_role=data_role
        )
        gpu_start = gpu_end = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        wall_s = 0.0
        flop_results = []
        stage_losses: list[float] = []
        train_samples_by_stage: list[int] = []
        accuracy_matrix: list[list[float | None]] = []
        final_test_inference_s = 0.0
        final_test_samples = 0
        model: JointClassifier | None = None
        optimizer: torch.optim.Optimizer | None = None
        for stage_index in range(data.tasks):
            profiler = PhaseFlopProfiler()
            try:
                cumulative_train = ConcatDataset(
                    [
                        data.benchmark.train_stream[index].dataset
                        for index in range(stage_index + 1)
                    ]
                )
                seen_classes = sum(
                    len(data.task_groups[index]) for index in range(stage_index + 1)
                )
                model = JointClassifier(
                    backbone, dataset, data.model_input_shape, seen_classes
                ).to(device)
                optimizer = _optimizer(model, parameters, best_lr)
                loader = DataLoader(
                    cumulative_train,
                    batch_size=int(parameters["train_mb_size"]),
                    shuffle=True,
                    num_workers=int(parameters["num_workers"]),
                    pin_memory=device.type == "cuda",
                )
                final_stage_loss = None
                _sync(device)
                wall_start = time.perf_counter()
                profiler.start()
                stage_epochs = int(parameters["epochs_per_experience"])
                for epoch_index in range(stage_epochs):
                    profiler.begin_epoch()
                    loss_sum = 0.0
                    samples = 0
                    model.train()
                    for batch in loader:
                        x, y = batch[0].to(device), batch[1].to(device)
                        profiler.switch("core_training")
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.cross_entropy(model(x), y)
                        loss.backward()
                        optimizer.step()
                        count = int(y.numel())
                        loss_sum += float(loss.detach().item()) * count
                        samples += count
                        profiler.add_processed_samples(count)
                    profiler.end_epoch()
                    if not samples:
                        raise ValueError("Joint cumulative training dataset is empty")
                    final_stage_loss = loss_sum / samples
                    print(
                        "JOINT_TRAIN "
                        f"dataset={dataset} method={method} order={order_id} "
                        f"seed={seed} stage={stage_index + 1}/{data.tasks} "
                        f"epoch={epoch_index + 1}/{stage_epochs} "
                        f"lr={optimizer.param_groups[0]['lr']:.8g} "
                        f"loss={final_stage_loss:.6f}",
                        flush=True,
                    )
                flop_results.append(profiler.stop(strict=False))
                _sync(device)
                wall_s += time.perf_counter() - wall_start
                if final_stage_loss is None:
                    raise RuntimeError("Joint stage completed without a training loss")
                stage_losses.append(float(final_stage_loss))
                train_samples_by_stage.append(len(cumulative_train))
                row: list[float | None] = [None] * data.tasks
                for test_index in range(stage_index + 1):
                    evaluation = _evaluate_experience(
                        model,
                        data.benchmark.test_stream[test_index],
                        device,
                        int(parameters["eval_mb_size"]),
                        int(parameters["num_workers"]),
                        measure_latency=(stage_index == data.tasks - 1),
                    )
                    if isinstance(evaluation, tuple):
                        accuracy, elapsed_s, sample_count = evaluation
                        final_test_inference_s += elapsed_s
                        final_test_samples += sample_count
                    else:
                        accuracy = evaluation
                    row[test_index] = float(accuracy)
                accuracy_matrix.append(row)
                seen_weights = data.test_samples_per_task[: stage_index + 1]
                seen_accuracy = sum(
                    float(row[index]) * int(seen_weights[index])
                    for index in range(stage_index + 1)
                ) / sum(int(value) for value in seen_weights)
                print(
                    "JOINT_STAGE_COMPLETE "
                    f"dataset={dataset} method={method} order={order_id} "
                    f"seed={seed} stage={stage_index + 1}/{data.tasks} "
                    f"lr={optimizer.param_groups[0]['lr']:.8g} "
                    f"loss={final_stage_loss:.6f} acc={seen_accuracy:.6f}",
                    flush=True,
                )
            except BaseException:
                profiler.abort()
                raise
        gpu_ms = None
        peak_mib = None
        if device.type == "cuda":
            gpu_end.record()
            torch.cuda.synchronize(device)
            gpu_ms = float(gpu_start.elapsed_time(gpu_end))
            peak_mib = float(torch.cuda.max_memory_allocated(device) / (2**20))

        if model is None or optimizer is None or not stage_losses:
            raise RuntimeError("Joint learning produced no completed stages")
        model.eval()
        sample = data.test[0][0].unsqueeze(0).to(device)
        single_forward_flops, single_forward_detail = profile_single_forward(model, sample)
        operation_metrics = summarize_learning_flops(
            flop_results, single_forward_flops
        )
        accuracy = [float(accuracy_matrix[index][index]) for index in range(data.tasks)]
        if final_test_samples <= 0:
            raise RuntimeError("Final joint stage produced no latency samples")
        latency = 1000.0 * final_test_inference_s / final_test_samples
        result = {
            **identity,
            "status": STATUS_COMPLETED,
            "started_at": pending["started_at"],
            "finished_at": timestamp(),
            "config": config,
            "tasks": data.tasks,
            "task_groups": [list(group) for group in data.task_groups],
            "accuracy_matrix_lower_triangular": accuracy_matrix,
            "accuracy_by_task": [float(value) for value in accuracy],
            "stage_final_epoch_train_loss": stage_losses,
            "train_samples_by_stage": train_samples_by_stage,
            "final_epoch_train_loss": float(stage_losses[-1]),
            "training_runtime": {
                "total_s": float(wall_s),
                "total_gpu_ms": gpu_ms,
            },
            "training_operations": {
                "task_summed_terminal_flops_per_sample": float(
                    sum(result.terminal_flops_per_sample for result in flop_results)
                ),
                **operation_metrics,
                "auxiliary_nonflop_ops": summarize_auxiliary_nonflop_ops(
                    flop_results
                ),
                "single_forward_detail": single_forward_detail,
            },
            "persistent_storage": _storage(model, optimizer),
            "inference": {"final_latency_ms_per_sample": float(latency)},
            "working_memory_diagnostic": {
                "native_peak_allocated_gpu_memory_mib": peak_mib
            },
        }
        validate_joint_result(result)
        atomic_write_json(path, result)
        fill_from_joint_run(search_path, path)
        return path
    except BaseException as exc:
        atomic_write_json(
            path,
            {
                **pending,
                "status": STATUS_FAILED,
                "finished_at": timestamp(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def run_method_joint(
    *,
    dataset_root: Path,
    search_root: Path,
    joint_root: Path,
    exp_name: str,
    dataset: str,
    method: str,
    device: torch.device,
    backbone: str,
) -> list[Path]:
    return [
        run_joint_unit(
            dataset_root=dataset_root,
            search_root=search_root,
            joint_root=joint_root,
            exp_name=exp_name,
            dataset=dataset,
            method=method,
            order_id=order_id,
            seed=seed,
            device=device,
            backbone=backbone,
        )
        for order_id in sorted(ORDERS_BY_DATASET[dataset])
        for seed in SEEDS_BY_DATASET[dataset]
    ]
