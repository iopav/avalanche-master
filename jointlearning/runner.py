from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import ConcatDataset, DataLoader

from cil_experiments.data import build_dataset_bundle
from cil_experiments.experiment_config import EXPERIMENT_ID, validate_experiment_id
from cil_experiments.final_hyperparameters import get_final_hyperparameters
from cil_experiments.flops import (
    PhaseFlopProfiler,
    profile_single_forward,
    summarize_auxiliary_nonflop_ops,
    summarize_learning_flops,
)
from cil_experiments.metrics import compute_cil_metrics, validate_summary
from cil_experiments.models import BACKBONES, build_feature_extractor
from cil_experiments.output import AtomicRunArtifacts, completed_summary_path
from cil_experiments.registry import DATASETS, DEFAULT_BACKBONES, get_task_groups
from cil_experiments.runner import (
    _evaluate_experience,
    _fvcore_crosscheck,
    _measure_latency,
    set_determinism,
)
from cil_experiments.storage import ByteCounter


class JointClassifier(nn.Module):
    def __init__(
        self,
        dataset_name: str,
        input_shape: tuple[int, ...],
        backbone_id: str,
        num_classes: int,
    ):
        super().__init__()
        self.feature_extractor = build_feature_extractor(
            backbone_id, dataset_name, input_shape
        )
        self.classifier = nn.Linear(self.feature_extractor.feature_dim, num_classes)
        self.feature_dim = self.feature_extractor.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


class TagFexCapacityMatchedJoint(nn.Module):
    """Joint reference with one same-backbone image branch per TagFex stage."""

    def __init__(
        self,
        dataset_name: str,
        input_shape: tuple[int, ...],
        backbone_id: str,
        branches: int,
        num_classes: int,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            build_feature_extractor(backbone_id, dataset_name, input_shape)
            for _ in range(branches)
        )
        self.feature_dim = self.branches[0].feature_dim * branches
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        return self.classifier(torch.cat(features, dim=1))


def _train_stage(
    model: nn.Module,
    dataset,
    parameters: dict[str, Any],
    epochs: int,
    device: torch.device,
) -> tuple[Any, float, float | None, float | None]:
    optimizer = SGD(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        momentum=float(parameters["momentum"]),
        weight_decay=float(parameters["weight_decay"]),
        foreach=bool(parameters.get("foreach", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(parameters["train_mb_size"]),
        shuffle=True,
        num_workers=int(parameters.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    peak_mib = None
    gpu_ms = None
    start_event = end_event = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    profiler = PhaseFlopProfiler()
    profiler.start()
    started = time.perf_counter()
    try:
        for _ in range(epochs):
            profiler.begin_epoch()
            profiler.switch("core_training")
            model.train()
            for x, y, *_ in loader:
                x = x.to(device, non_blocking=device.type == "cuda")
                y = y.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(x), y).backward()
                optimizer.step()
                profiler.add_processed_samples(len(y))
            profiler.end_epoch()
        result = profiler.stop(strict=True)
    except BaseException:
        profiler.abort()
        raise
    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize(device)
        gpu_ms = float(start_event.elapsed_time(end_event))
        peak_mib = float(torch.cuda.max_memory_allocated(device) / (2**20))
    return result, time.perf_counter() - started, gpu_ms, peak_mib


def _storage_summary(model: nn.Module) -> dict[str, Any]:
    parameter_counter = ByteCounter()
    model_bytes = sum(parameter_counter.tensor(value) for value in model.parameters())
    auxiliary_counter = ByteCounter()
    auxiliary = sum(auxiliary_counter.tensor(value) for value in model.buffers())
    total = int(model_bytes + auxiliary)
    return {
        "model_parameter_bytes": int(model_bytes),
        "replay_sample_bytes": 0,
        "replay_label_bytes": 0,
        "auxiliary_bytes": int(auxiliary),
        "total_bytes": total,
        "total_mib": float(total / (2**20)),
        "pulse_encoding": "NA",
        "label_encoding": "NA",
    }


def _write_markdown(path: Path, summary: dict[str, Any], matrix: np.ndarray) -> None:
    rows = [
        "# Joint Learning Reference",
        "",
        f"- method: `{summary['method']}`",
        f"- tasks: `{summary['tasks']}`",
        f"- final average accuracy: `{summary['cil_performance']['final_average_accuracy']:.6f}`",
        "",
        "## Accuracy matrix",
        "",
        "```json",
        json.dumps(matrix.tolist(), ensure_ascii=False),
        "```",
        "",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def run_joint_reference(
    project_root: Path,
    dataset_root: Path,
    result_root: Path,
    dataset_name: str,
    paired_method: str,
    order_id: int,
    seed: int,
    device: torch.device,
    *,
    backbone_id: str | None = None,
    parameters: dict[str, Any] | None = None,
    overwrite: bool = False,
    experiment_id: int = EXPERIMENT_ID,
    resume: bool = False,
) -> Path:
    experiment_id = validate_experiment_id(experiment_id)
    if paired_method not in DEFAULT_BACKBONES:
        raise ValueError(f"Unknown paired method: {paired_method}")
    resolved_backbone_id = backbone_id or DEFAULT_BACKBONES[paired_method]
    if resolved_backbone_id not in BACKBONES:
        raise ValueError(f"Unknown registered backbone: {resolved_backbone_id}")
    if parameters is None:
        parameters = get_final_hyperparameters(
            dataset_name, paired_method, require_locked=True
        )
    parameters = dict(parameters)
    epochs = int(parameters["epochs_per_experience"])
    set_determinism(seed)
    data = build_dataset_bundle(
        dataset_root,
        DATASETS[dataset_name],
        order_id,
        input_view="image",
    )
    method_name = f"joint_{paired_method}"
    config = {
        "schema": "metrics1.docx-compatible-joint-reference-v1",
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "method": method_name,
        "paired_method": paired_method,
        "order_id": int(order_id),
        "seed": int(seed),
        "selected_task_groups": [list(group) for group in get_task_groups(dataset_name, order_id)],
        "backbone": {
            "backbone_id": resolved_backbone_id,
            "feature_dim": (
                BACKBONES[resolved_backbone_id].feature_dim * data.tasks
                if paired_method == "tagfex"
                else BACKBONES[resolved_backbone_id].feature_dim
            ),
            "feature_dim_per_branch": (
                BACKBONES[resolved_backbone_id].feature_dim
                if paired_method == "tagfex"
                else None
            ),
            "capacity_matched_branches_at_final_stage": (
                data.tasks if paired_method == "tagfex" else None
            ),
            "pretrained": False,
        },
        "training": {
            "epochs_per_cumulative_stage": epochs,
            "paired_training_parameters": parameters,
            "restart_from_scratch_each_stage": True,
        },
        "protocol": {"input_view_id": data.input_view_id, "replay": None},
    }
    if resume and not overwrite:
        completed = completed_summary_path(
            result_root,
            dataset_name,
            method_name,
            order_id,
            seed,
            experiment_id,
            expected_config=config,
        )
        if completed is not None:
            print(
                "SKIP completed "
                f"exp={experiment_id} dataset={dataset_name} method={method_name} "
                f"order={order_id} seed={seed}: {completed}"
            )
            return completed
    with AtomicRunArtifacts(
        result_root,
        dataset_name,
        method_name,
        order_id,
        seed,
        config,
        overwrite,
        experiment_id,
    ) as artifacts:
        matrix = np.full((data.tasks, data.tasks), np.nan, dtype=np.float64)
        seen_datasets = []
        task_seconds: list[float] = []
        task_gpu_ms: list[float] = []
        task_peaks: list[float] = []
        task_flops = []
        terminal_values: list[float] = []
        model: nn.Module | None = None
        for task_index, experience in enumerate(data.benchmark.train_stream):
            seen_datasets.append(experience.dataset.train())
            set_determinism(seed)
            seen_classes = sum(len(group) for group in data.task_groups[: task_index + 1])
            model = (
                TagFexCapacityMatchedJoint(
                    dataset_name,
                    data.model_input_shape,
                    resolved_backbone_id,
                    task_index + 1,
                    seen_classes,
                )
                if paired_method == "tagfex"
                else JointClassifier(
                    dataset_name,
                    data.model_input_shape,
                    resolved_backbone_id,
                    seen_classes,
                )
            ).to(device)
            flop_result, wall_s, gpu_ms, peak_mib = _train_stage(
                model,
                ConcatDataset(seen_datasets),
                parameters,
                epochs,
                device,
            )
            task_flops.append(flop_result)
            terminal_values.append(flop_result.terminal_flops_per_sample)
            task_seconds.append(wall_s)
            if gpu_ms is not None:
                task_gpu_ms.append(gpu_ms)
            if peak_mib is not None:
                task_peaks.append(peak_mib)
            for test_index in range(task_index + 1):
                matrix[task_index, test_index] = _evaluate_experience(
                    model,
                    data.benchmark.test_stream[test_index],
                    device,
                    int(parameters["eval_mb_size"]),
                    int(parameters.get("num_workers", 0)),
                )
        if model is None:
            raise RuntimeError("Joint benchmark has no stages")
        model.eval()
        sample = data.test[0][0].unsqueeze(0).to(device)
        single_flops, single_detail = profile_single_forward(model, sample)
        fvcore_detail = _fvcore_crosscheck(model, sample)
        latency = _measure_latency(
            model,
            list(data.benchmark.test_stream),
            device,
            int(parameters["eval_mb_size"]),
            int(parameters.get("num_workers", 0)),
        )
        cil = compute_cil_metrics(matrix, data.test_samples_per_task)
        cil["intransigence"] = [0.0] * data.tasks
        incremental = task_seconds[1:]
        flop_summary = summarize_learning_flops(task_flops, single_flops)
        final_feature_dim = int(getattr(model, "feature_dim"))
        summary = {
            "method": f"Joint({paired_method})",
            "seed": int(seed),
            "tasks": int(data.tasks),
            "cil_performance": cil,
            "network": {
                "final_hidden_neurons": final_feature_dim,
                "total_new_neurons": (
                    final_feature_dim - BACKBONES[resolved_backbone_id].feature_dim
                    if paired_method == "tagfex" else 0
                ),
                "total_reused_neurons": (
                    final_feature_dim - BACKBONES[resolved_backbone_id].feature_dim
                    if paired_method == "tagfex" else None
                ),
            },
            "training_runtime": {
                "total_s": float(sum(task_seconds)),
                "total_gpu_ms": float(sum(task_gpu_ms)) if task_gpu_ms else None,
                "initial_task_s": float(task_seconds[0]),
                "incremental_tasks_total_s": float(sum(incremental)),
                "mean_incremental_task_s": float(statistics.mean(incremental)),
                "median_incremental_task_s": float(statistics.median(incremental)),
            },
            "training_operations": {
                "task_summed_terminal_flops_per_sample": float(sum(terminal_values)),
                **flop_summary,
                "auxiliary_nonflop_ops": summarize_auxiliary_nonflop_ops(task_flops),
            },
            "persistent_storage": _storage_summary(model),
            "inference": {"final_latency_ms_per_sample": float(latency)},
            "working_memory_diagnostic": {
                "native_peak_allocated_gpu_memory_mib": (
                    float(max(task_peaks)) if task_peaks else None
                ),
                "note": "Transient working memory; excluded from persistent storage.",
            },
        }
        validate_summary(summary)
        lower = [
            [float(matrix[row, col]) if col <= row else None for col in range(data.tasks)]
            for row in range(data.tasks)
        ]
        matrix_payload = {
            "dataset": dataset_name,
            "method": method_name,
            "paired_method": paired_method,
            "backbone_id": resolved_backbone_id,
            "input_view_id": data.input_view_id,
            "order_id": int(order_id),
            "seed": int(seed),
            "experiment_id": experiment_id,
            "tasks": int(data.tasks),
            "orientation": "row=cumulative Joint stage; column=evaluated test experience",
            "test_samples_per_task": [int(value) for value in data.test_samples_per_task],
            "accuracy_matrix_lower_triangular": lower,
            "joint_current_task_accuracy_curve": [
                float(matrix[index, index]) for index in range(data.tasks)
            ],
        }
        artifacts.logger.info("single_forward=%s", json.dumps(single_detail, sort_keys=True))
        artifacts.logger.info("fvcore=%s", json.dumps(fvcore_detail, sort_keys=True))
        artifacts.commit(summary, matrix_payload)
        report_path = artifacts.summary_path.with_suffix(".md")
        _write_markdown(report_path, summary, matrix)
        return artifacts.summary_path
