from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cil_experiments.flops import PhaseFlopProfiler
from cil_experiments.metrics import compute_cil_metrics
from cil_experiments.registry import DATASETS, METHODS, TRAINING_DEFAULTS
from cil_experiments.output import local_timestamp
from cil_experiments.runner import _sync, _train_experience, run_one, set_determinism
from cil_experiments.strategies import build_strategy
from cil_experiments.validation import build_internal_validation_benchmark


VALIDATION_FRACTION = 0.2
SEARCH_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "er_ace": [
        {"learning_rate": 0.10, "memory_size": 200, "batch_size_mem": 10},
        {"learning_rate": 0.05, "memory_size": 200, "batch_size_mem": 10},
        {"learning_rate": 0.01, "memory_size": 200, "batch_size_mem": 10},
    ],
    "ewc": [
        {"learning_rate": 0.10, "ewc_lambda": 0.1},
        {"learning_rate": 0.10, "ewc_lambda": 0.4},
        {"learning_rate": 0.10, "ewc_lambda": 1.0},
    ],
    "cwr_star": [
        {"learning_rate": 0.10},
        {"learning_rate": 0.05},
        {"learning_rate": 0.01},
    ],
    "icarl": [
        {"learning_rate": 0.10, "memory_size": 2000},
        {"learning_rate": 0.05, "memory_size": 2000},
        {"learning_rate": 0.01, "memory_size": 2000},
    ],
    "fecam": [
        {"learning_rate": 0.10, "shrink1": 1.0, "shrink2": 1.0},
        {"learning_rate": 0.10, "shrink1": 0.5, "shrink2": 0.5},
        {"learning_rate": 0.10, "shrink1": 2.0, "shrink2": 2.0},
    ],
}


def _internal_train_validation_benchmark(
    dataset_root: Path, dataset_name: str, validation_seed: int
):
    return build_internal_validation_benchmark(
        dataset_root,
        dataset_name,
        validation_seed,
        validation_fraction=VALIDATION_FRACTION,
        order_id=1,
    )


def _evaluate_validation_with_flops(model, experiences, device: torch.device):
    model.eval()
    scores: list[float] = []
    counts: list[int] = []
    profiler = PhaseFlopProfiler()
    profiler.start()
    profiler.begin_epoch()
    try:
        with torch.no_grad(), profiler.temporary_phase("student_current_forward"):
            for experience in experiences:
                correct = 0
                total = 0
                loader = DataLoader(
                    experience.dataset.eval(),
                    batch_size=TRAINING_DEFAULTS["eval_mb_size"],
                    shuffle=False,
                    num_workers=0,
                )
                for batch in loader:
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    logits = model(x)
                    correct += int((torch.argmax(logits, dim=1) == y).sum().item())
                    total += int(y.numel())
                    profiler.add_processed_samples(len(y))
                if total == 0:
                    raise ValueError("Empty internal-validation experience")
                scores.append(correct / total)
                counts.append(total)
        profiler.end_epoch()
        flop_result = profiler.stop(strict=True)
    except BaseException:
        profiler.abort()
        raise
    return scores, counts, int(flop_result.total_flops)


def _apply_candidate(method: str, candidate: dict[str, Any]):
    training_before = copy.deepcopy(TRAINING_DEFAULTS)
    method_before = copy.deepcopy(METHODS[method])
    TRAINING_DEFAULTS["learning_rate"] = float(candidate["learning_rate"])
    for key, value in candidate.items():
        if key != "learning_rate":
            METHODS[method][key] = value
    return training_before, method_before


def _restore_candidate(method: str, snapshots) -> None:
    training_before, method_before = snapshots
    TRAINING_DEFAULTS.clear()
    TRAINING_DEFAULTS.update(training_before)
    METHODS[method].clear()
    METHODS[method].update(method_before)


def _run_candidate(
    project_root: Path,
    dataset_name: str,
    method: str,
    candidate: dict[str, Any],
    epochs: int,
    device,
    validation_seed: int,
):
    set_determinism(validation_seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    benchmark, split_counts = _internal_train_validation_benchmark(
        project_root / "dataset", dataset_name, validation_seed
    )
    snapshots = _apply_candidate(method, candidate)
    start = time.perf_counter()
    try:
        strategy = build_strategy(method, DATASETS[dataset_name].in_channels, epochs, device, dataset_name)
        task_count = len(benchmark.train_stream)
        matrix = np.full((task_count, task_count), np.nan)
        training_flops = 0
        validation_flops = 0
        training_runtime = 0.0
        peak_mib = 0.0
        for task_index, experience in enumerate(benchmark.train_stream):
            result, wall_s, _, training_peak = _train_experience(strategy, experience, device)
            training_flops += int(result.total_flops)
            training_runtime += float(wall_s)
            validation_scores, _, evaluation_flops = _evaluate_validation_with_flops(
                strategy.strategy.model,
                list(benchmark.test_stream)[: task_index + 1],
                device,
            )
            matrix[task_index, : task_index + 1] = validation_scores
            validation_flops += evaluation_flops
            if training_peak is not None:
                peak_mib = max(peak_mib, float(training_peak))
            if device.type == "cuda":
                peak_mib = max(peak_mib, torch.cuda.max_memory_allocated(device) / (2**20))
        counts = [len(experience.dataset) for experience in benchmark.test_stream]
        validation_metrics = compute_cil_metrics(matrix, counts)
        _sync(device)
        trial_runtime = time.perf_counter() - start
        return {
            "candidate": candidate,
            "validation_average_incremental_accuracy": validation_metrics["average_incremental_accuracy"],
            "validation_final_average_accuracy": validation_metrics["final_average_accuracy"],
            "training_runtime_s": training_runtime,
            "trial_runtime_s": trial_runtime,
            "training_flops": training_flops,
            "validation_flops": validation_flops,
            "total_search_flops": training_flops + validation_flops,
            "peak_allocated_gpu_memory_mib": peak_mib if device.type == "cuda" else None,
            "split_counts_by_internal_class": split_counts,
            "validation_accuracy_matrix_lower_triangular": [
                [float(matrix[row, col]) if col <= row else None for col in range(matrix.shape[1])]
                for row in range(matrix.shape[0])
            ],
        }
    finally:
        _restore_candidate(method, snapshots)


def _atomic_text(destination: Path, text: str) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    system_temp = Path(tempfile.mkdtemp(prefix="avalanche-cil-search-"))
    volume_temp: Path | None = None
    try:
        source = system_temp / "artifact"
        source.write_text(text, encoding="utf-8")
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        volume_temp = Path(name)
        shutil.copyfile(source, volume_temp)
        with volume_temp.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(volume_temp, destination)
        volume_temp = None
    finally:
        if volume_temp is not None:
            volume_temp.unlink(missing_ok=True)
        shutil.rmtree(system_temp, ignore_errors=True)


def _atomic_json(destination: Path, value: dict[str, Any]) -> None:
    _atomic_text(
        destination,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def search_one(
    project_root: Path,
    dataset_name: str,
    method: str,
    epochs: int,
    device,
    validation_seed: int,
    search_timestamp: str,
):
    trials = []
    for candidate_id, candidate in enumerate(SEARCH_CANDIDATES[method], start=1):
        result = _run_candidate(
            project_root, dataset_name, method, candidate, epochs, device, validation_seed
        )
        result["candidate_id"] = candidate_id
        trials.append(result)
    selected = max(
        trials,
        key=lambda item: (
            item["validation_final_average_accuracy"],
            item["validation_average_incremental_accuracy"],
            -item["candidate_id"],
        ),
    )
    payload = {
        "dataset": dataset_name,
        "method": method,
        "validation_seed": validation_seed,
        "timestamp": search_timestamp,
        "order_id": 1,
        "validation_fraction": VALIDATION_FRACTION,
        "selection_rule": "maximize validation_final_average_accuracy; tie-break by validation_average_incremental_accuracy then lower candidate_id",
        "test_set_used_for_selection": False,
        "epochs_per_experience": epochs,
        "trials": trials,
        "selected_candidate_id": selected["candidate_id"],
        "selected_hyperparameters": selected["candidate"],
        "hyperparameter_search_runtime_s": float(sum(item["trial_runtime_s"] for item in trials)),
        "hyperparameter_search_flops": int(sum(item["total_search_flops"] for item in trials)),
    }
    destination = (
        project_root
        / "result"
        / dataset_name
        / method
        / (
            f"{dataset_name}__{method}__validation-seed-{validation_seed:03d}"
            f"__timestamp-{search_timestamp}__hyperparameter-search-cost.json"
        )
    )
    payload["search_cost_path"] = str(destination)
    _atomic_json(destination, payload)
    print(destination)
    return payload, destination


def _validation_csv(payloads: list[dict[str, Any]]) -> str:
    fields = [
        "timestamp", "dataset", "method", "validation_seed", "validation_fraction",
        "candidate_id", "selected", "hyperparameters",
        "validation_final_average_accuracy", "validation_average_incremental_accuracy",
        "training_runtime_s", "trial_runtime_s", "training_flops", "validation_flops",
        "total_search_flops", "peak_allocated_gpu_memory_mib",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for payload in payloads:
        for trial in payload["trials"]:
            writer.writerow(
                {
                    "timestamp": payload["timestamp"],
                    "dataset": payload["dataset"],
                    "method": payload["method"],
                    "validation_seed": payload["validation_seed"],
                    "validation_fraction": payload["validation_fraction"],
                    "candidate_id": trial["candidate_id"],
                    "selected": trial["candidate_id"] == payload["selected_candidate_id"],
                    "hyperparameters": json.dumps(trial["candidate"], sort_keys=True),
                    "validation_final_average_accuracy": trial["validation_final_average_accuracy"],
                    "validation_average_incremental_accuracy": trial["validation_average_incremental_accuracy"],
                    "training_runtime_s": trial["training_runtime_s"],
                    "trial_runtime_s": trial["trial_runtime_s"],
                    "training_flops": trial["training_flops"],
                    "validation_flops": trial["validation_flops"],
                    "total_search_flops": trial["total_search_flops"],
                    "peak_allocated_gpu_memory_mib": trial["peak_allocated_gpu_memory_mib"],
                }
            )
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Three-candidate train-only CIL hyperparameter search")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--validation-seed", type=int, default=62)
    parser.add_argument("--run-selected", action="store_true")
    parser.add_argument("--selected-order-id", type=int, default=1)
    parser.add_argument("--selected-seed", type=int, default=62)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    search_timestamp = local_timestamp()
    payloads: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        for method in args.methods:
            payload, _ = search_one(
                project_root,
                dataset_name,
                method,
                args.epochs,
                device,
                args.validation_seed,
                search_timestamp,
            )
            payloads.append(payload)
    csv_path = (
        project_root
        / "result"
        / (
            f"validation-grid-search__validation-seed-{args.validation_seed:03d}"
            f"__timestamp-{search_timestamp}.csv"
        )
    )
    _atomic_text(csv_path, _validation_csv(payloads))
    print(csv_path)
    if args.run_selected:
        for payload in payloads:
            method = payload["method"]
            snapshots = _apply_candidate(method, payload["selected_hyperparameters"])
            try:
                provenance = {
                    "status": "selected_by_train_only_validation_grid_search",
                    "required_trials": len(payload["trials"]),
                    "validation_seed": args.validation_seed,
                    "validation_fraction": VALIDATION_FRACTION,
                    "selected_candidate_id": payload["selected_candidate_id"],
                    "selected_hyperparameters": payload["selected_hyperparameters"],
                    "search_timestamp": search_timestamp,
                    "search_cost_file": str(payload["search_cost_path"]),
                    "test_set_used_for_selection": False,
                }
                path = run_one(
                    project_root,
                    project_root / "dataset",
                    project_root / "result",
                    payload["dataset"],
                    method,
                    args.selected_order_id,
                    args.selected_seed,
                    args.epochs,
                    device,
                    overwrite=True,
                    search_provenance=provenance,
                    parameter_overrides=payload["selected_hyperparameters"],
                )
                print(path)
            finally:
                _restore_candidate(method, snapshots)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(exit_code)
