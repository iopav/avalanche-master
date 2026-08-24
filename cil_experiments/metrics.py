from __future__ import annotations

import math
from typing import Any

import numpy as np


SUMMARY_KEYS = {
    "method",
    "seed",
    "tasks",
    "cil_performance",
    "network",
    "training_runtime",
    "training_operations",
    "persistent_storage",
    "inference",
    "working_memory_diagnostic",
}

CIL_KEYS = {
    "average_incremental_accuracy",
    "final_average_accuracy",
    "average_forgetting",
    "mean_final_accuracy_loss",
    "backward_transfer",
    "worst_forgotten_task",
    "forgetting_mean",
    "forgetting_median",
    "forgetting_std",
    "forgetting_min",
    "forgetting_q25",
    "forgetting_q75",
    "forgetting_max",
    "forgetting_positive_fraction",
    "task_end_seen_accuracy_curve",
    "forgetting_per_task",
}

FLOP_KEYS = {
    "overall_learning_flops",
    "core_training_flops",
    "learning_auxiliary_flops",
    "hyperparameter_search_flops",
    "single_sample_forward_flops",
}


def compute_cil_metrics(matrix: np.ndarray, test_counts: list[int]) -> dict[str, Any]:
    tasks = matrix.shape[0]
    if matrix.shape != (tasks, tasks) or len(test_counts) != tasks:
        raise ValueError("Accuracy matrix/test-count dimensions do not match")
    seen_curve: list[float] = []
    for t in range(tasks):
        row = matrix[t, : t + 1]
        if not np.isfinite(row).all():
            raise ValueError(f"Non-finite observed accuracy at task {t + 1}")
        weights = np.asarray(test_counts[: t + 1], dtype=np.float64)
        seen_curve.append(float(np.average(row, weights=weights)))
    forgetting = []
    final_loss = []
    for i in range(tasks - 1):
        forgetting.append(float(np.max(matrix[i:, i]) - matrix[-1, i]))
        final_loss.append(float(matrix[i, i] - matrix[-1, i]))
    f = np.asarray(forgetting, dtype=np.float64)
    loss = np.asarray(final_loss, dtype=np.float64)
    result = {
        "average_incremental_accuracy": float(np.mean(seen_curve)),
        "final_average_accuracy": float(seen_curve[-1]),
        "average_forgetting": float(np.mean(f)),
        "mean_final_accuracy_loss": float(np.mean(loss)),
        "backward_transfer": float(-np.mean(loss)),
        "worst_forgotten_task": int(np.argmax(f) + 1),
        "forgetting_mean": float(np.mean(f)),
        "forgetting_median": float(np.median(f)),
        "forgetting_std": float(np.std(f, ddof=0)),
        "forgetting_min": float(np.min(f)),
        "forgetting_q25": float(np.quantile(f, 0.25)),
        "forgetting_q75": float(np.quantile(f, 0.75)),
        "forgetting_max": float(np.max(f)),
        "forgetting_positive_fraction": float(np.mean(f > 0)),
        "task_end_seen_accuracy_curve": seen_curve,
        "forgetting_per_task": forgetting,
    }
    return result


def validate_summary(summary: dict[str, Any]) -> None:
    if set(summary) != SUMMARY_KEYS:
        raise AssertionError(f"summary top-level keys differ: {set(summary) ^ SUMMARY_KEYS}")
    if set(summary["cil_performance"]) != CIL_KEYS:
        raise AssertionError("cil_performance keys differ from metrics1.docx")
    if set(summary["network"]) != {"final_hidden_neurons", "total_new_neurons", "total_reused_neurons"}:
        raise AssertionError("network keys differ from metrics1.docx")
    if set(summary["training_runtime"]) != {
        "total_s", "total_gpu_ms", "initial_task_s", "incremental_tasks_total_s",
        "mean_incremental_task_s", "median_incremental_task_s",
    }:
        raise AssertionError("training_runtime keys differ from metrics1.docx")
    training_operation_keys = {
        "estimated_cumulative_dense_flops",
        "task_summed_terminal_flops_per_sample",
        *FLOP_KEYS,
        "auxiliary_nonflop_ops",
    }
    if set(summary["training_operations"]) != training_operation_keys:
        raise AssertionError("training_operations keys differ from metrics1.docx")
    flops = summary["training_operations"]
    for key in FLOP_KEYS:
        if isinstance(flops[key], bool) or not isinstance(flops[key], int) or flops[key] < 0:
            raise AssertionError(f"flops.{key} must be a non-negative integer")
    formal_training_flops = flops["core_training_flops"] + flops["learning_auxiliary_flops"]
    if summary["training_operations"]["estimated_cumulative_dense_flops"] != formal_training_flops:
        raise AssertionError("formal training FLOPs differ from core plus learning auxiliary FLOPs")
    expected_overall = formal_training_flops + flops["hyperparameter_search_flops"]
    if flops["overall_learning_flops"] != expected_overall:
        raise AssertionError("overall learning FLOPs identity failed")
    nonflop = flops["auxiliary_nonflop_ops"]
    if set(nonflop) != {
        "explicit_nonflop_operator_calls",
        "manual_nonflop_operations",
        "total_recorded_events",
        "policy",
    }:
        raise AssertionError("auxiliary_nonflop_ops keys differ")
    calls = nonflop["explicit_nonflop_operator_calls"]
    if not isinstance(calls, dict) or any(
        not isinstance(name, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in calls.items()
    ):
        raise AssertionError("auxiliary_nonflop_ops call counts are invalid")
    manual_operations = nonflop["manual_nonflop_operations"]
    if not isinstance(manual_operations, list) or any(
        not isinstance(operation, dict)
        or set(operation) != {"name", "calls", "reason", "variables"}
        or not isinstance(operation["name"], str)
        or not operation["name"]
        or isinstance(operation["calls"], bool)
        or not isinstance(operation["calls"], int)
        or operation["calls"] < 0
        or not isinstance(operation["reason"], str)
        or not operation["reason"]
        or not isinstance(operation["variables"], dict)
        for operation in manual_operations
    ):
        raise AssertionError("manual auxiliary non-FLOP records are invalid")
    expected_nonflop_events = sum(calls.values()) + sum(
        operation["calls"] for operation in manual_operations
    )
    if nonflop["total_recorded_events"] != expected_nonflop_events:
        raise AssertionError("auxiliary_nonflop_ops total_recorded_events mismatch")
    if not isinstance(nonflop["policy"], str) or not nonflop["policy"]:
        raise AssertionError("auxiliary_nonflop_ops policy must be a non-empty string")
    if set(summary["persistent_storage"]) != {
        "model_parameter_bytes", "replay_sample_bytes", "replay_label_bytes",
        "auxiliary_bytes", "total_bytes", "total_mib", "pulse_encoding", "label_encoding",
    }:
        raise AssertionError("persistent_storage keys differ from metrics1.docx")
    if set(summary["inference"]) != {"final_latency_ms_per_sample"}:
        raise AssertionError("inference keys differ from metrics1.docx")
    if set(summary["working_memory_diagnostic"]) != {
        "native_peak_allocated_gpu_memory_mib", "note"
    }:
        raise AssertionError("working_memory_diagnostic keys differ from metrics1.docx")
    tasks = int(summary["tasks"])
    cil = summary["cil_performance"]
    if len(cil["task_end_seen_accuracy_curve"]) != tasks:
        raise AssertionError("seen-accuracy curve length mismatch")
    if len(cil["forgetting_per_task"]) != tasks - 1:
        raise AssertionError("forgetting array length mismatch")
    if not math.isclose(
        cil["average_incremental_accuracy"],
        float(np.mean(cil["task_end_seen_accuracy_curve"])),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AssertionError("AIA consistency check failed")
    if not math.isclose(cil["final_average_accuracy"], cil["task_end_seen_accuracy_curve"][-1], abs_tol=1e-9):
        raise AssertionError("final accuracy consistency check failed")
    if not math.isclose(cil["backward_transfer"], -cil["mean_final_accuracy_loss"], abs_tol=1e-9):
        raise AssertionError("BWT identity failed")
    runtime = summary["training_runtime"]
    if not math.isclose(runtime["total_s"], runtime["initial_task_s"] + runtime["incremental_tasks_total_s"], rel_tol=1e-8, abs_tol=1e-8):
        raise AssertionError("runtime decomposition failed")
    storage = summary["persistent_storage"]
    expected_bytes = sum(storage[k] for k in (
        "model_parameter_bytes", "replay_sample_bytes", "replay_label_bytes", "auxiliary_bytes"
    ))
    if storage["total_bytes"] != expected_bytes:
        raise AssertionError("persistent byte sum failed")
    if not math.isclose(storage["total_mib"], storage["total_bytes"] / (2**20), rel_tol=1e-12):
        raise AssertionError("MiB conversion failed")
    if not all(np.isfinite(v) for v in _numeric_leaves(summary)):
        raise AssertionError("summary contains NaN or infinity")


def _numeric_leaves(value: Any):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float, np.number)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numeric_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _numeric_leaves(item)
