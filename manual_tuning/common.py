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
from cil_experiments.strategies import build_strategy

TRAINING_KEYS = set(TRAINING_DEFAULTS)


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


def _apply_parameters(method: str, parameters: dict[str, Any]):
    training_before = copy.deepcopy(TRAINING_DEFAULTS)
    method_before = copy.deepcopy(METHODS[method])
    for key, value in parameters.items():
        if key in TRAINING_KEYS:
            TRAINING_DEFAULTS[key] = value
        elif key in METHODS[method]:
            METHODS[method][key] = value
        else:
            raise KeyError(f"Unknown parameter for {method}: {key}")
    return training_before, method_before


def _restore_parameters(method: str, snapshots) -> None:
    training_before, method_before = snapshots
    TRAINING_DEFAULTS.clear()
    TRAINING_DEFAULTS.update(training_before)
    METHODS[method].clear()
    METHODS[method].update(method_before)


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


def run_manual(
    *,
    dataset: str,
    method: str,
    parameters: dict[str, Any],
    order_id: int = 1,
    seed: int = 62,
    epochs: int = 3,
    device: str | None = "cuda",
) -> Path:
    """Run the formal train/eval path without FLOPs, storage, latency or summary metrics."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if method not in METHODS:
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
    timestamp = local_timestamp()
    try:
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
            values = " ".join(f"{value:.6f}" for value in matrix[task_index, : task_index + 1])
            print(f"after_task_{task_index + 1:02d} {values}", flush=True)
        lower = [
            [float(matrix[row, col]) if col <= row else None for col in range(data.spec.tasks)]
            for row in range(data.spec.tasks)
        ]
        payload = {
            "dataset": dataset,
            "method": method,
            "order_id": int(order_id),
            "seed": int(seed),
            "epochs_per_experience": int(epochs),
            "timestamp": timestamp,
            "parameters": copy.deepcopy(parameters),
            "accuracy_matrix_lower_triangular": lower,
        }
        destination = (
            Path(__file__).resolve().parent
            / "results"
            / dataset
            / method
            / (
                f"{dataset}__{method}__order-{order_id:02d}__seed-{seed:03d}"
                f"__timestamp-{timestamp}__accuracy-matrix.json"
            )
        )
        _atomic_save(destination, payload)
        print(f"accuracy_matrix_saved={destination}", flush=True)
        return destination
    finally:
        _restore_parameters(method, snapshots)
