"""Shared no-FLOPs runner for simple manual regularization controls."""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import ConcatDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path(__file__).resolve().parent / "reports"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cil_experiments.data import build_dataset_bundle
from cil_experiments.metrics import compute_cil_metrics
from cil_experiments.models import BackboneClassifier, build_feature_extractor
from cil_experiments.registry import DATASETS, DEFAULT_BACKBONE_ID, get_task_groups


TRAINING_PARAMETER_KEYS = {
    "learning_rate",
    "momentum",
    "weight_decay",
    "train_mb_size",
    "eval_mb_size",
}
METHOD_PARAMETER_KEYS = {
    "naive": set(),
    "ewc": {"ewc_lambda", "mode"},
    "si": {"si_lambda", "eps"},
    "lwf": {"alpha", "temperature"},
    "er": {"memory_size", "batch_size_mem"},
}


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def validate_parameters(method: str, parameters: dict[str, Any]) -> None:
    if method not in METHOD_PARAMETER_KEYS:
        raise ValueError(f"Unsupported regularization-control method: {method}")
    expected = TRAINING_PARAMETER_KEYS | METHOD_PARAMETER_KEYS[method]
    if set(parameters) != expected:
        raise ValueError(
            f"Parameter keys differ for {method}: "
            f"missing={sorted(expected - set(parameters))}, "
            f"unexpected={sorted(set(parameters) - expected)}"
        )
    for key in ("learning_rate", "train_mb_size", "eval_mb_size"):
        if float(parameters[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if method == "er":
        memory_size = int(parameters["memory_size"])
        batch_size_mem = int(parameters["batch_size_mem"])
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if not 0 < batch_size_mem <= memory_size:
            raise ValueError("batch_size_mem must be in [1, memory_size]")


def build_strategy(
    method: str,
    in_channels: int,
    epochs: int,
    device: torch.device,
    parameters: dict[str, Any],
    *,
    backbone_id: str = DEFAULT_BACKBONE_ID,
    dataset_name: str = "uwave",
    model_input_shape: tuple[int, ...] | None = None,
):
    from avalanche.training.plugins import ReplayPlugin
    from avalanche.training.supervised import EWC, LwF, Naive, SynapticIntelligence

    spec = DATASETS[dataset_name]
    default_shape = (
        tuple(spec.image_train_shape[1:])
        if spec.image_layout == "NCHW"
        else (spec.image_train_shape[3], spec.image_train_shape[1], spec.image_train_shape[2])
    )
    shape = model_input_shape or default_shape
    model = BackboneClassifier(backbone_id, dataset_name, shape).to(device)
    optimizer = SGD(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        momentum=float(parameters["momentum"]),
        weight_decay=float(parameters["weight_decay"]),
        foreach=False,
    )
    common = {
        "model": model,
        "optimizer": optimizer,
        "criterion": nn.CrossEntropyLoss(),
        "train_mb_size": int(parameters["train_mb_size"]),
        "train_epochs": int(epochs),
        "eval_mb_size": int(parameters["eval_mb_size"]),
        "device": device,
        "evaluator": None,
        "eval_every": -1,
    }
    if method == "naive":
        return Naive(**common)
    if method == "ewc":
        return EWC(
            ewc_lambda=float(parameters["ewc_lambda"]),
            mode=str(parameters["mode"]),
            **common,
        )
    if method == "si":
        return SynapticIntelligence(
            si_lambda=float(parameters["si_lambda"]),
            eps=float(parameters["eps"]),
            **common,
        )
    if method == "lwf":
        return LwF(
            alpha=float(parameters["alpha"]),
            temperature=float(parameters["temperature"]),
            **common,
        )
    if method == "er":
        replay = ReplayPlugin(
            mem_size=int(parameters["memory_size"]),
            batch_size=int(parameters["train_mb_size"]),
            batch_size_mem=int(parameters["batch_size_mem"]),
        )
        return Naive(plugins=[replay], **common)
    raise AssertionError(method)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    experience,
    device: torch.device,
    batch_size: int,
) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    loader = DataLoader(
        experience.dataset.eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    for batch in loader:
        x = batch[0].to(device, non_blocking=device.type == "cuda")
        y = batch[1].to(device, non_blocking=device.type == "cuda")
        prediction = model(x).argmax(dim=1)
        correct += int((prediction == y).sum().item())
        total += int(y.numel())
    if was_training:
        model.train()
    if total == 0:
        raise RuntimeError("Empty evaluation experience")
    return correct / total


@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> list[float]:
    """Return final test accuracy for every mapped class without affecting training."""
    was_training = model.training
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.int64)
    counts = torch.zeros(num_classes, dtype=torch.int64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        x = batch[0].to(device, non_blocking=device.type == "cuda")
        y = batch[1].to(device, non_blocking=device.type == "cuda")
        prediction = model(x).argmax(dim=1)
        counts += torch.bincount(y.cpu(), minlength=num_classes)
        matched = y[prediction == y].cpu()
        correct += torch.bincount(matched, minlength=num_classes)
    model.train(was_training)
    if torch.any(counts == 0):
        raise RuntimeError(f"Final test set has empty mapped classes: {torch.where(counts == 0)[0].tolist()}")
    return [float(value) for value in (correct.float() / counts.float()).tolist()]


def summarize_run(
    *,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    epochs: int,
    parameters: dict[str, Any],
    matrix: np.ndarray,
    test_counts: list[int],
    task_seconds: list[float],
    per_class_accuracy: list[float],
    inverse_label_map: dict[int, int],
) -> dict[str, Any]:
    metrics = compute_cil_metrics(matrix, test_counts)
    last_internal_class = len(per_class_accuracy) - 1
    previous_task_counts = np.asarray(test_counts[:-1], dtype=np.float64)
    final_previous_tasks_accuracy = float(
        np.average(matrix[-1, :-1], weights=previous_task_counts)
    )
    return {
        "dataset": dataset,
        "method": method,
        "order_id": int(order_id),
        "seed": int(seed),
        "epochs_per_experience": int(epochs),
        "parameters": dict(parameters),
        "all_class_macro_accuracy": float(np.mean(per_class_accuracy)),
        "last_class_internal_id": int(last_internal_class),
        "last_class_raw_label": int(inverse_label_map[last_internal_class]),
        "last_class_accuracy": float(per_class_accuracy[-1]),
        "preceding_classes_macro_accuracy": float(np.mean(per_class_accuracy[:-1])),
        "per_class_accuracy": [float(value) for value in per_class_accuracy],
        "final_last_task_accuracy": float(matrix[-1, -1]),
        "final_previous_tasks_accuracy": final_previous_tasks_accuracy,
        "final_sample_weighted_accuracy": float(metrics["final_average_accuracy"]),
        "average_incremental_accuracy": float(metrics["average_incremental_accuracy"]),
        "average_forgetting": float(metrics["average_forgetting"]),
        "task_seconds": [float(value) for value in task_seconds],
        "total_training_seconds": float(sum(task_seconds)),
    }


def matrix_markdown(matrix: np.ndarray) -> list[str]:
    tasks = matrix.shape[0]
    header = ["训练后"] + [f"测试任务 {index + 1}" for index in range(tasks)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in range(tasks):
        values = [
            f"{100 * matrix[row, col]:.2f}%" if col <= row else "—"
            for col in range(tasks)
        ]
        lines.append("| " + " | ".join([f"任务 {row + 1}", *values]) + " |")
    return lines


def write_report(
    *,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    epochs: int,
    parameters: dict[str, Any],
    matrix: np.ndarray,
    test_counts: list[int],
    task_seconds: list[float],
    extra_summary: dict[str, Any] | None = None,
) -> Path:
    metrics = compute_cil_metrics(matrix, test_counts)
    old_counts = np.asarray(test_counts[:-1], dtype=np.float64)
    final_old = float(np.average(matrix[-1, :-1], weights=old_counts))
    final_new = float(matrix[-1, -1])
    groups = get_task_groups(dataset, order_id)
    lines = [
        f"# {dataset} / {method} 手动对照结果",
        "",
        "## 配置",
        "",
        f"- 数据集：`{dataset}`",
        f"- 方法：`{method}`",
        f"- 原始标签任务分组：`{groups}`",
        f"- order_id / seed / epochs：`{order_id} / {seed} / {epochs}`",
        "- 默认骨干：`resnet18_cifar`，feature_dim=512；实际值以运行参数为准",
        f"- 参数：`{parameters}`",
        "- FLOPs、持久存储、延迟和正式结果 JSON：未计算",
        "",
        "## 准确率矩阵",
        "",
        *matrix_markdown(matrix),
        "",
        "## 汇总",
        "",
        f"- Average Incremental Accuracy：`{metrics['average_incremental_accuracy']:.6f}`",
        f"- Final Average Accuracy：`{metrics['final_average_accuracy']:.6f}`",
        f"- Final Old-task Weighted Accuracy：`{final_old:.6f}`",
        f"- Final New-task Accuracy：`{final_new:.6f}`",
        f"- Average Forgetting：`{metrics['average_forgetting']:.6f}`",
        f"- 训练耗时：`{sum(task_seconds):.3f}s`，分任务 `{[round(v, 3) for v in task_seconds]}`",
        *[
            f"- {key}：`{value}`"
            for key, value in (extra_summary or {}).items()
        ],
        "",
        "## 解释边界",
        "",
        "这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。",
        "",
    ]
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = REPORT_ROOT / (
        f"{dataset}__{method}__order-{order_id:02d}__seed-{seed}__epochs-{epochs}.md"
    )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def run_manual_report(
    *,
    dataset: str,
    method: str,
    parameters: dict[str, Any],
    order_id: int,
    seed: int,
    epochs: int,
    device: str,
    backbone_id: str = DEFAULT_BACKBONE_ID,
    result_name: str | None = None,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    validate_parameters(method, parameters)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_determinism(seed)
    data = build_dataset_bundle(
        PROJECT_ROOT / "dataset", DATASETS[dataset], order_id, input_view="image"
    )
    strategy = build_strategy(
        method,
        data.spec.in_channels,
        epochs,
        resolved_device,
        parameters,
        backbone_id=backbone_id,
        dataset_name=dataset,
        model_input_shape=data.model_input_shape,
    )
    matrix = np.full((data.tasks, data.tasks), np.nan, dtype=np.float64)
    task_seconds: list[float] = []
    for task_index, experience in enumerate(data.benchmark.train_stream):
        started = time.perf_counter()
        strategy.train(
            experience,
            num_workers=0,
            pin_memory=resolved_device.type == "cuda",
        )
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        task_seconds.append(time.perf_counter() - started)
        for test_index in range(task_index + 1):
            matrix[task_index, test_index] = evaluate(
                strategy.model,
                data.benchmark.test_stream[test_index],
                resolved_device,
                int(parameters["eval_mb_size"]),
            )
    per_class_accuracy = evaluate_per_class(
        strategy.model,
        data.test,
        resolved_device,
        int(parameters["eval_mb_size"]),
        data.spec.num_classes,
    )
    reported_method = result_name or method
    summary = summarize_run(
        dataset=dataset,
        method=reported_method,
        order_id=order_id,
        seed=seed,
        epochs=epochs,
        parameters=parameters,
        matrix=matrix,
        test_counts=data.test_samples_per_task,
        task_seconds=task_seconds,
        per_class_accuracy=per_class_accuracy,
        inverse_label_map=data.inverse_label_map,
    )
    extra_summary: dict[str, Any] = {
        "all_class_macro_accuracy": summary["all_class_macro_accuracy"],
        "last_class_raw_label": summary["last_class_raw_label"],
        "last_class_accuracy": summary["last_class_accuracy"],
        "preceding_classes_macro_accuracy": summary["preceding_classes_macro_accuracy"],
    }
    if method == "er":
        replay_plugin = next(
            plugin
            for plugin in strategy.plugins
            if plugin.__class__.__name__ == "ReplayPlugin"
        )
        stored = len(replay_plugin.storage_policy.buffer)
        if stored > int(parameters["memory_size"]):
            raise AssertionError(
                f"ER stored {stored} samples above cap {parameters['memory_size']}"
            )
        extra_summary["最终实际回放样本数"] = stored
    report = write_report(
        dataset=dataset,
        method=reported_method,
        order_id=order_id,
        seed=seed,
        epochs=epochs,
        parameters=parameters,
        matrix=matrix,
        test_counts=data.test_samples_per_task,
        task_seconds=task_seconds,
        extra_summary=extra_summary,
    )
    print(f"report={report}", flush=True)
    return summary


class CumulativeModel(nn.Module):
    """Configured backbone with a classifier for all classes seen so far."""

    def __init__(
        self,
        dataset_name: str,
        model_input_shape: tuple[int, ...],
        num_classes: int,
        backbone_id: str,
    ):
        super().__init__()
        self.feature_extractor = build_feature_extractor(
            backbone_id, dataset_name, model_input_shape
        )
        self.classifier = nn.Linear(self.feature_extractor.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


def run_joint_report(
    *,
    dataset: str,
    parameters: dict[str, Any],
    order_id: int,
    seed: int,
    epochs: int,
    device: str,
    backbone_id: str = DEFAULT_BACKBONE_ID,
    result_name: str = "joint_cumulative",
) -> dict[str, Any]:
    expected = TRAINING_PARAMETER_KEYS
    if set(parameters) != expected:
        raise ValueError(
            "Joint parameter keys must be exactly the shared training keys: "
            f"missing={sorted(expected - set(parameters))}, "
            f"unexpected={sorted(set(parameters) - expected)}"
        )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = build_dataset_bundle(
        PROJECT_ROOT / "dataset", DATASETS[dataset], order_id, input_view="image"
    )
    matrix = np.full((data.tasks, data.tasks), np.nan, dtype=np.float64)
    task_seconds: list[float] = []
    seen_datasets = []
    for task_index, experience in enumerate(data.benchmark.train_stream):
        seen_datasets.append(experience.dataset.train())
        # Each cumulative stage restarts from the same seeded backbone instead
        # of inheriting a favorable solution from the previous stage.
        set_determinism(seed)
        seen_classes = sum(
            len(group) for group in data.task_groups[: task_index + 1]
        )
        model = CumulativeModel(
            dataset,
            data.model_input_shape,
            seen_classes,
            backbone_id,
        ).to(resolved_device)
        optimizer = SGD(
            model.parameters(),
            lr=float(parameters["learning_rate"]),
            momentum=float(parameters["momentum"]),
            weight_decay=float(parameters["weight_decay"]),
            foreach=False,
        )
        loader = DataLoader(
            ConcatDataset(seen_datasets),
            batch_size=int(parameters["train_mb_size"]),
            shuffle=True,
            num_workers=0,
            pin_memory=resolved_device.type == "cuda",
        )
        started = time.perf_counter()
        model.train()
        for _ in range(epochs):
            for batch in loader:
                x = batch[0].to(
                    resolved_device, non_blocking=resolved_device.type == "cuda"
                )
                y = batch[1].to(
                    resolved_device, non_blocking=resolved_device.type == "cuda"
                )
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(x), y).backward()
                optimizer.step()
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        task_seconds.append(time.perf_counter() - started)
        for test_index in range(task_index + 1):
            matrix[task_index, test_index] = evaluate(
                model,
                data.benchmark.test_stream[test_index],
                resolved_device,
                int(parameters["eval_mb_size"]),
            )
    per_class_accuracy = evaluate_per_class(
        model,
        data.test,
        resolved_device,
        int(parameters["eval_mb_size"]),
        data.spec.num_classes,
    )
    summary = summarize_run(
        dataset=dataset,
        method=result_name,
        order_id=order_id,
        seed=seed,
        epochs=epochs,
        parameters=parameters,
        matrix=matrix,
        test_counts=data.test_samples_per_task,
        task_seconds=task_seconds,
        per_class_accuracy=per_class_accuracy,
        inverse_label_map=data.inverse_label_map,
    )
    report = write_report(
        dataset=dataset,
        method=result_name,
        order_id=order_id,
        seed=seed,
        epochs=epochs,
        parameters=parameters,
        matrix=matrix,
        test_counts=data.test_samples_per_task,
        task_seconds=task_seconds,
        extra_summary={
            "final_stage_accessible_training_samples": len(data.train),
            "all_class_macro_accuracy": summary["all_class_macro_accuracy"],
            "last_class_raw_label": summary["last_class_raw_label"],
            "last_class_accuracy": summary["last_class_accuracy"],
            "preceding_classes_macro_accuracy": summary["preceding_classes_macro_accuracy"],
        },
    )
    print(f"report={report}", flush=True)
    return summary
