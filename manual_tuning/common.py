from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cil_experiments.data import build_dataset_bundle
from cil_experiments.output import local_timestamp
from cil_experiments.registry import DATASETS, METHODS, TRAINING_DEFAULTS
from cil_experiments.strategies import (
    build_lwf_tuning_strategy,
    build_si_tuning_strategy,
    build_strategy,
)

TRAINING_KEYS = set(TRAINING_DEFAULTS)
MANUAL_CANDIDATE_METHODS: dict[str, dict[str, Any]] = {
    # Avalanche 的 SI 包装器要求 si_lambda；eps 使用其源码默认值。
    # 该注册表仅供手工调参，不会进入正式五方法入口。
    "si": {"si_lambda": 0.0001, "eps": 0.0000001},
    # LwF 保存上一 Experience 的教师模型，并在当前样本上蒸馏旧类输出。
    "lwf": {"alpha": 1.0, "temperature": 2.0},
}


def _set_determinism(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _evaluate(model, experience, device: torch.device, batch_size: int) -> float:
    loader = DataLoader(
        experience.dataset.eval(), batch_size=batch_size, shuffle=False, num_workers=0
    )
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            non_blocking = device.type == "cuda"
            x = batch[0].to(device, non_blocking=non_blocking)
            y = batch[1].to(device, non_blocking=non_blocking)
            prediction = torch.argmax(model(x), dim=1)
            correct += int((prediction == y).sum().item())
            total += int(y.numel())
    if was_training:
        model.train()
    if total == 0:
        raise ValueError("Empty evaluation experience")
    return correct / total


def _method_parameters(method: str) -> dict[str, Any]:
    if method in METHODS:
        return METHODS[method]
    if method in MANUAL_CANDIDATE_METHODS:
        return MANUAL_CANDIDATE_METHODS[method]
    raise ValueError(f"Unknown method: {method}")


def _apply_parameters(method: str, parameters: dict[str, Any]):
    training_before = copy.deepcopy(TRAINING_DEFAULTS)
    selected_method_parameters = _method_parameters(method)
    method_before = copy.deepcopy(selected_method_parameters)
    for key, value in parameters.items():
        if key in TRAINING_KEYS:
            TRAINING_DEFAULTS[key] = value
        elif key in selected_method_parameters:
            selected_method_parameters[key] = value
        else:
            raise KeyError(f"Unknown parameter for {method}: {key}")
    return training_before, method_before


def _restore_parameters(method: str, snapshots) -> None:
    training_before, method_before = snapshots
    TRAINING_DEFAULTS.clear()
    TRAINING_DEFAULTS.update(training_before)
    selected_method_parameters = _method_parameters(method)
    selected_method_parameters.clear()
    selected_method_parameters.update(method_before)


def _atomic_save(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    staged = Path(name)
    try:
        staged.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with staged.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(staged, destination)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _format_accuracy_matrix(matrix: np.ndarray) -> str:
    """把完整下三角准确率矩阵格式化为一次性终端输出。"""
    task_count = int(matrix.shape[0])
    cell_width = 8
    row_label_width = 10
    separator = "-" * (row_label_width + 3 + cell_width * task_count + 3 + 12)
    lines = [
        "",
        "准确率矩阵（%，行表示完成训练的任务，列表示测试任务）",
        separator,
        f"{'训练后':>{row_label_width}} |"
        + "".join(f"T{index + 1:02d}".rjust(cell_width) for index in range(task_count))
        + " | 已见任务均值",
        separator,
    ]
    for row in range(task_count):
        cells = []
        for column in range(task_count):
            if column <= row:
                cells.append(f"{matrix[row, column] * 100:>{cell_width}.2f}")
            else:
                cells.append("-".rjust(cell_width))
        seen_mean = float(np.mean(matrix[row, : row + 1])) * 100
        lines.append(
            f"T{row + 1:02d}".rjust(row_label_width)
            + " |"
            + "".join(cells)
            + f" | {seen_mean:>10.2f}"
        )
    lines.extend(
        [
            separator,
            f"最终平均准确率（最后一行）: {float(np.mean(matrix[-1])) * 100:.2f}%",
        ]
    )
    return "\n".join(lines)


def run_manual(
    *,
    dataset: str,
    method: str,
    parameters: dict[str, Any],
    order_id: int = 1,
    seed: int = 62,
    epochs: int = 3,
    device: str | None = "cuda",
) -> np.ndarray:
    """运行不含 FLOPs、存储、延迟和汇总指标的手工调参训练与评估路径。"""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if method not in METHODS and method not in MANUAL_CANDIDATE_METHODS:
        raise ValueError(f"Unknown method: {method}")
    resolved_device = torch.device(device or "cuda")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if resolved_device.type == "cuda":
        if resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(resolved_device)
    _set_determinism(seed)
    data = build_dataset_bundle(PROJECT_ROOT / "dataset", DATASETS[dataset], order_id)
    snapshots = _apply_parameters(method, parameters)
    try:
        if method == "si":
            bundle = build_si_tuning_strategy(
                data.spec.in_channels,
                epochs,
                resolved_device,
                MANUAL_CANDIDATE_METHODS[method],
            )
        elif method == "lwf":
            bundle = build_lwf_tuning_strategy(
                data.spec.in_channels,
                epochs,
                resolved_device,
                MANUAL_CANDIDATE_METHODS[method],
            )
        else:
            bundle = build_strategy(
                method,
                data.spec.in_channels,
                epochs,
                resolved_device,
                dataset,
                enable_flop_accounting=False,
            )
        bundle.strategy.model.to(resolved_device)
        parameter_devices = {parameter.device.type for parameter in bundle.strategy.model.parameters()}
        if parameter_devices != {resolved_device.type}:
            raise RuntimeError(
                f"Model parameters are not entirely on {resolved_device}: {sorted(parameter_devices)}"
            )
        device_name = (
            torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda" else "CPU"
        )
        matrix = np.full((data.spec.tasks, data.spec.tasks), np.nan, dtype=np.float64)
        print(
            f"dataset={dataset} method={method} order_id={order_id} seed={seed} "
            f"epochs={epochs} device={resolved_device} device_name={device_name} "
            f"parameters={parameters}",
            flush=True,
        )
        # FLOPs are intentionally disabled for manual tuning.  Do not replace
        # this direct train call with runner._train_experience, which attaches
        # PhaseFlopProfiler and performs the formal FLOP audit.
        for task_index, experience in enumerate(data.benchmark.train_stream):
            bundle.strategy.train(
                experience,
                num_workers=0,
                pin_memory=resolved_device.type == "cuda",
            )
            for test_index in range(task_index + 1):
                matrix[task_index, test_index] = _evaluate(
                    bundle.strategy.model,
                    data.benchmark.test_stream[test_index],
                    resolved_device,
                    int(TRAINING_DEFAULTS["eval_mb_size"]),
                )
        # 全部训练和评估结束后再用一次 stdout 写入打印完整矩阵，避免训练阶段产生的
        # warning 或 stderr 信息插入矩阵各行。真实异常仍按原逻辑输出 traceback。
        sys.stdout.write(_format_accuracy_matrix(matrix) + "\n")
        sys.stdout.flush()

        # 手工调参阶段只打印准确率矩阵，不保存 JSON。若以后需要恢复保存功能，
        # 再取消下面原保存流程的注释；正式实验的 JSON 输出不受这里影响。
        # lower = [
        #     [float(matrix[row, col]) if col <= row else None for col in range(data.spec.tasks)]
        #     for row in range(data.spec.tasks)
        # ]
        # payload = {
        #     "dataset": dataset,
        #     "method": method,
        #     "order_id": int(order_id),
        #     "seed": int(seed),
        #     "epochs_per_experience": int(epochs),
        #     "timestamp": local_timestamp(),
        #     "parameters": copy.deepcopy(parameters),
        #     "accuracy_matrix_lower_triangular": lower,
        # }
        # destination = (
        #     Path(__file__).resolve().parent
        #     / "results"
        #     / dataset
        #     / method
        #     / (
        #         f"{dataset}__{method}__order-{order_id:02d}__seed-{seed:03d}"
        #         f"__timestamp-{payload['timestamp']}__accuracy-matrix.json"
        #     )
        # )
        # _atomic_save(destination, payload)
        # print(f"accuracy_matrix_saved={destination}", flush=True)
        return matrix.copy()
    finally:
        _restore_parameters(method, snapshots)
