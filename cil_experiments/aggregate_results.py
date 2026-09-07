from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .intransigence import _artifact_paths
from .joint_learning import joint_run_path, validate_joint_result
from .lr_search import search_unit_path, validate_search_unit
from .metrics import validate_summary
from .order_seed_registry import SEEDS_BY_DATASET, get_formal_seeds
from .output import atomic_write_text, completed_summary_path, json_text
from .registry import DATASETS, FORMAL_METHODS, ORDERS_BY_DATASET
from .search_schema import STATUS_COMPLETED, search_result_path


SCALAR_METRICS = {
    "average_incremental_accuracy": ("cil_performance", "average_incremental_accuracy"),
    "final_average_accuracy": ("cil_performance", "final_average_accuracy"),
    "average_forgetting": ("cil_performance", "average_forgetting"),
    "overall_learning_flops": ("training_operations", "overall_learning_flops"),
    "training_runtime_s": ("training_runtime", "total_s"),
    "persistent_storage_bytes": ("persistent_storage", "total_bytes"),
    "inference_latency_ms_per_sample": ("inference", "final_latency_ms_per_sample"),
}


def _mean_std(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    result: dict[str, float | int] = _stats(numbers)
    result["n"] = len(numbers)
    return result


def _stats(numbers: list[float]) -> dict[str, float]:
    mean = float(statistics.fmean(numbers))
    if len(numbers) < 2:
        return {"mean": mean, "ci95": 0.0, "std": 0.0}
    std = float(statistics.stdev(numbers))
    critical = {2: 12.706, 3: 4.303, 4: 3.182}.get(len(numbers), 1.96)
    return {
        "mean": mean,
        "ci95": float(critical * std / math.sqrt(len(numbers))),
        "std": std,
    }


def aggregate_results(
    result_root: Path,
    *,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...] | None = None,
    exp_name: str,
) -> dict[str, Any]:
    seeds_by_dataset = {
        dataset: get_formal_seeds(dataset) if seeds is None else tuple(seeds)
        for dataset in datasets
    }
    records: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    observed_dataset_methods: set[tuple[str, str]] = set()
    for summary_path in result_root.glob("*/*/summary/*__summary.json"):
        matrix_path, config_path = _artifact_paths(summary_path)
        if not matrix_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Incomplete artifact set for {summary_path}")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        dataset = matrix.get("dataset")
        method = matrix.get("method")
        order_id = matrix.get("order_id")
        seed = matrix.get("seed")
        if (
            dataset not in datasets
            or method not in methods
            or order_id not in ORDERS_BY_DATASET[dataset]
            or seed not in seeds_by_dataset[dataset]
        ):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_summary(summary, allow_pending_intransigence=method == "joint")
        if matrix.get("exp_name") != exp_name or config.get("exp_name") != exp_name:
            raise ValueError(f"Experiment name mismatch in {summary_path}")
        backbone_id = config.get("backbone", {}).get("backbone_id")
        if not isinstance(backbone_id, str) or not backbone_id:
            raise ValueError(f"Missing backbone ID in {config_path}")
        if matrix.get("backbone_id") != backbone_id:
            raise ValueError(f"Backbone mismatch inside artifact set: {summary_path}")
        key = (dataset, method, backbone_id, int(order_id), int(seed))
        if key in records:
            raise RuntimeError(
                f"Duplicate formal result for dataset/method/backbone/order/seed {key}: "
                f"{records[key]['summary_path']} and {summary_path}"
            )
        records[key] = {
            "summary": summary,
            "summary_path": str(summary_path.resolve()),
        }
        observed_dataset_methods.add((dataset, method))

    required_dataset_methods = {(dataset, method) for dataset in datasets for method in methods}
    missing_methods = required_dataset_methods - observed_dataset_methods
    if missing_methods:
        raise RuntimeError(f"Missing formal dataset-method results: {sorted(missing_methods)}")

    grouped: dict[tuple[str, str, str], list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for (dataset, method, backbone, order_id, seed), record in records.items():
        grouped[(dataset, method, backbone)].append((order_id, seed, record["summary"]))
    output_groups: list[dict[str, Any]] = []
    for (dataset, method, backbone), values in sorted(grouped.items()):
        expected_pairs = {
            (order_id, seed)
            for order_id in ORDERS_BY_DATASET[dataset]
            for seed in seeds_by_dataset[dataset]
        }
        actual_pairs = {(order_id, seed) for order_id, seed, _ in values}
        if actual_pairs != expected_pairs:
            raise RuntimeError(
                f"Incomplete {dataset}/{method}/{backbone}: "
                f"missing={sorted(expected_pairs - actual_pairs)} "
                f"unexpected={sorted(actual_pairs - expected_pairs)}"
            )
        summaries = [summary for _, _, summary in values]
        task_counts = {int(summary["tasks"]) for summary in summaries}
        if len(task_counts) != 1:
            raise ValueError(f"Task-count mismatch in {dataset}/{method}/{backbone}")
        metrics = {
            name: _mean_std(summary[section][field] for summary in summaries)
            for name, (section, field) in SCALAR_METRICS.items()
        }
        stage_count = task_counts.pop()
        intransigence = []
        if method != "joint":
            intransigence = [
                _mean_std(
                    summary["cil_performance"]["intransigence"][stage]
                    for summary in summaries
                )
                for stage in range(stage_count)
            ]
        output_groups.append(
            {
                "dataset": dataset,
                "method": method,
                "backbone_id": backbone,
                "n": len(summaries),
                "metrics": metrics,
                "intransigence_by_stage": intransigence,
            }
        )
    return {
        "schema": "formal-cil-aggregate-v1",
        "exp_name": exp_name,
        "datasets": list(datasets),
        "methods": list(methods),
        "order_ids_by_dataset": {
            dataset: sorted(ORDERS_BY_DATASET[dataset]) for dataset in datasets
        },
        "seeds_by_dataset": {key: list(value) for key, value in seeds_by_dataset.items()},
        "groups": output_groups,
    }


def _csv_text(payload: dict[str, Any]) -> str:
    rows = []
    for group in payload["groups"]:
        row = {
            "exp_name": payload["exp_name"],
            **{key: group[key] for key in ("dataset", "method", "backbone_id", "n")},
        }
        for metric, aggregate in group["metrics"].items():
            row[f"{metric}_mean"] = aggregate["mean"]
            row[f"{metric}_ci95"] = aggregate["ci95"]
            row[f"{metric}_std"] = aggregate["std"]
        row["intransigence_mean_by_stage"] = json.dumps(
            [entry["mean"] for entry in group["intransigence_by_stage"]]
        )
        row["intransigence_std_by_stage"] = json.dumps(
            [entry["std"] for entry in group["intransigence_by_stage"]]
        )
        row["intransigence_ci95_by_stage"] = json.dumps(
            [entry["ci95"] for entry in group["intransigence_by_stage"]]
        )
        rows.append(row)
    return _rows_csv(rows)


def _markdown_text(payload: dict[str, Any]) -> str:
    lines = [
        "# Formal CIL aggregate",
        "",
        f"Experiment: `{payload['exp_name']}`.",
        "",
        f"Orders: `{payload['order_ids_by_dataset']}`; seeds: `{payload['seeds_by_dataset']}`.",
        "",
        "| Dataset | Method | Backbone | n | Final ACC mean ± std | AIA mean ± std | Forgetting mean ± std |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for group in payload["groups"]:
        metrics = group["metrics"]
        final = metrics["final_average_accuracy"]
        aia = metrics["average_incremental_accuracy"]
        forgetting = metrics["average_forgetting"]
        lines.append(
            f"| {group['dataset']} | {group['method']} | {group['backbone_id']} | {group['n']} | "
            f"{final['mean']:.6f} ± {final['std']:.6f} | "
            f"{aia['mean']:.6f} ± {aia['std']:.6f} | "
            f"{forgetting['mean']:.6f} ± {forgetting['std']:.6f} |"
        )
        means = [entry["mean"] for entry in group["intransigence_by_stage"]]
        stds = [entry["std"] for entry in group["intransigence_by_stage"]]
        lines.extend(
            (
                "",
                f"- `{group['dataset']}/{group['method']}/{group['backbone_id']}` "
                f"Intransigence mean by stage: `{means}`; std: `{stds}`.",
            )
        )
    return "\n".join(lines) + "\n"


def _rows_csv(rows: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _flatten_report_values(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_report_values(name, nested))
        return result
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value, ensure_ascii=False, separators=(",", ":"))}
    return {prefix: "" if value is None else value}


SUMMARY_CSV_METRICS = {
    "average_incremental_accuracy": ("cil_performance", "average_incremental_accuracy"),
    "final_average_accuracy": ("cil_performance", "final_average_accuracy"),
    "average_forgetting": ("cil_performance", "average_forgetting"),
    "mean_final_accuracy_loss": ("cil_performance", "mean_final_accuracy_loss"),
    "backward_transfer": ("cil_performance", "backward_transfer"),
    "worst_forgotten_task": ("cil_performance", "worst_forgotten_task"),
    "forgetting_mean": ("cil_performance", "forgetting_mean"),
    "forgetting_median": ("cil_performance", "forgetting_median"),
    "forgetting_std": ("cil_performance", "forgetting_std"),
    "forgetting_min": ("cil_performance", "forgetting_min"),
    "forgetting_q25": ("cil_performance", "forgetting_q25"),
    "forgetting_q75": ("cil_performance", "forgetting_q75"),
    "forgetting_max": ("cil_performance", "forgetting_max"),
    "forgetting_positive_fraction": ("cil_performance", "forgetting_positive_fraction"),
    "task_end_seen_accuracy_curve": ("cil_performance", "task_end_seen_accuracy_curve"),
    "forgetting_per_task": ("cil_performance", "forgetting_per_task"),
    "intransigence": ("cil_performance", "intransigence"),
    "intransigence_mean": ("cil_performance", "intransigence_mean"),
    "final_hidden_neurons": ("network", "final_hidden_neurons"),
    "training_runtime_s": ("training_runtime", "total_s"),
    "training_gpu_ms": ("training_runtime", "total_gpu_ms"),
    "initial_task_s": ("training_runtime", "initial_task_s"),
    "incremental_tasks_total_s": ("training_runtime", "incremental_tasks_total_s"),
    "mean_incremental_task_s": ("training_runtime", "mean_incremental_task_s"),
    "median_incremental_task_s": ("training_runtime", "median_incremental_task_s"),
    "terminal_flops_per_sample": ("training_operations", "task_summed_terminal_flops_per_sample"),
    "overall_learning_flops": ("training_operations", "overall_learning_flops"),
    "core_training_flops": ("training_operations", "core_training_flops"),
    "learning_auxiliary_flops": ("training_operations", "learning_auxiliary_flops"),
    "single_sample_forward_flops": ("training_operations", "single_sample_forward_flops"),
    "model_parameter_bytes": ("persistent_storage", "model_parameter_bytes"),
    "replay_sample_bytes": ("persistent_storage", "replay_sample_bytes"),
    "replay_label_bytes": ("persistent_storage", "replay_label_bytes"),
    "auxiliary_bytes": ("persistent_storage", "auxiliary_bytes"),
    "persistent_storage_bytes": ("persistent_storage", "total_bytes"),
    "persistent_storage_mib": ("persistent_storage", "total_mib"),
    "inference_latency_ms_per_sample": ("inference", "final_latency_ms_per_sample"),
    "peak_allocated_gpu_memory_mib": ("working_memory_diagnostic", "native_peak_allocated_gpu_memory_mib"),
}


JOINT_CSV_METRICS = {
    "final_epoch_train_loss": ("final_epoch_train_loss",),
    "training_runtime_s": ("training_runtime", "total_s"),
    "training_gpu_ms": ("training_runtime", "total_gpu_ms"),
    "terminal_flops_per_sample": ("training_operations", "task_summed_terminal_flops_per_sample"),
    "overall_learning_flops": ("training_operations", "overall_learning_flops"),
    "core_training_flops": ("training_operations", "core_training_flops"),
    "learning_auxiliary_flops": ("training_operations", "learning_auxiliary_flops"),
    "single_sample_forward_flops": ("training_operations", "single_sample_forward_flops"),
    "model_parameter_bytes": ("persistent_storage", "model_parameter_bytes"),
    "model_buffer_bytes": ("persistent_storage", "model_buffer_bytes"),
    "optimizer_state_bytes": ("persistent_storage", "optimizer_state_bytes"),
    "persistent_storage_bytes": ("persistent_storage", "total_bytes"),
    "persistent_storage_mib": ("persistent_storage", "total_mib"),
    "inference_latency_ms_per_sample": ("inference", "final_latency_ms_per_sample"),
    "peak_allocated_gpu_memory_mib": ("working_memory_diagnostic", "native_peak_allocated_gpu_memory_mib"),
}


def _selected_metrics(payload: dict[str, Any], fields: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for name, path in fields.items():
        value: Any = payload
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (list, tuple)):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        selected[name] = "" if value is None else value
    return selected


def _identity_differences(
    payload: dict[str, Any], expected: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    return {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }


def _validate_candidate_artifacts(
    search: dict[str, Any], *, dataset: str, method: str, order_id: int, seed: int
) -> None:
    expected_matrix = {
        "dataset": dataset,
        "method": method,
        "order_id": int(order_id),
        "seed": int(seed),
        "exp_name": search["exp_name"],
        "backbone_id": search["backbone"],
    }
    for learning_rate, record in search["candidates"].items():
        for key in ("summary_file", "accuracy_matrix_file"):
            path = Path(record.get(key, ""))
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {method}/order{order_id}/seed{seed}/lr{learning_rate} {key}: {path}"
                )
        summary = json.loads(Path(record["summary_file"]).read_text(encoding="utf-8"))
        validate_summary(summary, allow_pending_intransigence=True)
        summary_config = summary["config"]
        summary_expected = {
            "exp_name": search["exp_name"],
            "seed": int(seed),
        }
        summary_differences = _identity_differences(summary, summary_expected)
        nested_differences = {
            "dataset": (summary_config.get("dataset", {}).get("name"), dataset),
            "method": (summary_config.get("method"), method),
            "backbone": (
                summary_config.get("backbone", {}).get("backbone_id"),
                search["backbone"],
            ),
            "learning_rate": (
                summary_config.get("final_hyperparameters", {})
                .get("resolved", {})
                .get("learning_rate"),
                float(learning_rate),
            ),
        }
        summary_differences.update(
            {
                key: values
                for key, values in nested_differences.items()
                if values[0] != values[1]
            }
        )
        if summary_differences:
            raise ValueError(f"Candidate summary identity mismatch: {summary_differences}")
        latency = summary.get("inference", {}).get("final_latency_ms_per_sample")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError(f"Candidate inference latency is invalid: {record['summary_file']}")
        matrix = json.loads(
            Path(record["accuracy_matrix_file"]).read_text(encoding="utf-8")
        )
        differences = _identity_differences(matrix, expected_matrix)
        if differences:
            raise ValueError(f"Candidate matrix identity mismatch: {differences}")
    selected = search["candidates"][str(search["best_lr"])]
    if search["best_summary_file"] != selected["summary_file"]:
        raise ValueError("Best summary path differs from the selected candidate")
    if search["best_accuracy_matrix_file"] != selected["accuracy_matrix_file"]:
        raise ValueError("Best accuracy-matrix path differs from the selected candidate")
    if not Path(search["best_checkpoint"]).is_file():
        raise FileNotFoundError(search["best_checkpoint"])


def build_dataset_search_summary(
    search_root: Path,
    *,
    dataset: str,
    methods: tuple[str, ...] = FORMAL_METHODS,
    order_ids: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> str:
    """Return one unaggregated row per learning-rate search candidate."""

    seeds = SEEDS_BY_DATASET[dataset] if seeds is None else tuple(seeds)
    selected_orders = tuple(sorted(ORDERS_BY_DATASET[dataset])) if order_ids is None else tuple(order_ids)
    rows: list[dict[str, Any]] = []
    expected = 0
    for method in methods:
        for order_id in selected_orders:
            for seed in seeds:
                path = search_unit_path(search_root, method, order_id, seed)
                if not path.is_file():
                    raise FileNotFoundError(path)
                search = json.loads(path.read_text(encoding="utf-8"))
                if search.get("status") != STATUS_COMPLETED:
                    raise RuntimeError(f"Search is incomplete: {path}")
                validate_search_unit(search)
                expected_search = {
                    "dataset": dataset,
                    "method": method,
                    "order": int(order_id),
                    "seed": int(seed),
                }
                differences = _identity_differences(search, expected_search)
                if differences:
                    raise ValueError(f"Search path/payload identity mismatch: {differences}")
                _validate_candidate_artifacts(
                    search,
                    dataset=dataset,
                    method=method,
                    order_id=order_id,
                    seed=seed,
                )
                expected += len(search["lr_candidates"])
                for learning_rate in search["lr_candidates"]:
                    candidate = search["candidates"][str(learning_rate)]
                    summary_path = Path(candidate["summary_file"])
                    if not summary_path.is_file():
                        raise FileNotFoundError(summary_path)
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    validate_summary(summary, allow_pending_intransigence=True)
                    row: dict[str, Any] = {
                        "exp_name": search["exp_name"],
                        "method": method,
                        "dataset": dataset,
                        "order": int(order_id),
                        "seed": int(seed),
                        "backbone": search["backbone"],
                        "lr": float(learning_rate),
                        "lr_candidates": json.dumps(
                            search["lr_candidates"], separators=(",", ":")
                        ),
                        "loss_selection": search["loss_selection"],
                        "selection_loss": float(candidate["selection_loss"]),
                        "best_lr": search["best_lr"],
                        "best_loss": search["best_loss"],
                        "total_search_flops": search["total_search_flops"],
                        "storage_bytes": int(candidate["storage_bytes"]),
                        "tasks": summary["tasks"],
                        "selection_loss_eval_flops": int(
                            candidate["selection_loss_eval_flops"]
                        ),
                    }
                    row.update(_selected_metrics(summary, SUMMARY_CSV_METRICS))
                    rows.append(row)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} search rows, found {len(rows)}")
    return _rows_csv(rows)


def build_dataset_joint_learning(
    joint_root: Path,
    *,
    dataset: str,
    methods: tuple[str, ...] = FORMAL_METHODS,
    order_ids: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> str:
    """Return one unaggregated row per from-scratch joint-learning run."""

    seeds = SEEDS_BY_DATASET[dataset] if seeds is None else tuple(seeds)
    selected_orders = tuple(sorted(ORDERS_BY_DATASET[dataset])) if order_ids is None else tuple(order_ids)
    task_count = len(next(iter(ORDERS_BY_DATASET[dataset].values())))
    rows: list[dict[str, Any]] = []
    for method in methods:
        for order_id in selected_orders:
            for seed in seeds:
                path = joint_run_path(joint_root, dataset, method, order_id, seed)
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != STATUS_COMPLETED:
                    raise RuntimeError(f"Joint run is incomplete: {path}")
                validate_joint_result(payload)
                expected_joint = {
                    "dataset": dataset,
                    "method": method,
                    "order": int(order_id),
                    "seed": int(seed),
                }
                differences = _identity_differences(payload, expected_joint)
                if differences:
                    raise ValueError(f"Joint path/payload identity mismatch: {differences}")
                if payload["task_groups"] != [
                    list(group) for group in ORDERS_BY_DATASET[dataset][order_id]
                ]:
                    raise ValueError(f"Joint task groups differ from the order registry: {path}")
                source_search_path = Path(payload["config"].get("source_search_json", ""))
                if not source_search_path.is_file():
                    raise FileNotFoundError(
                        f"Joint source search JSON is missing: {source_search_path}"
                    )
                source_search = json.loads(
                    source_search_path.read_text(encoding="utf-8")
                )
                validate_search_unit(source_search)
                source_expected = {
                    **expected_joint,
                    "exp_name": payload["exp_name"],
                    "backbone": payload["backbone"],
                    "best_lr": payload["best_lr"],
                }
                source_differences = _identity_differences(
                    source_search, source_expected
                )
                if source_differences:
                    raise ValueError(
                        f"Joint/source-search identity mismatch: {source_differences}"
                    )
                accuracy = payload.get("accuracy_by_task")
                if not isinstance(accuracy, list) or len(accuracy) != task_count:
                    raise ValueError(f"Joint task accuracy count is invalid: {path}")
                row: dict[str, Any] = {
                    "exp_name": payload["exp_name"],
                    "method": method,
                    "dataset": dataset,
                    "order": int(order_id),
                    "seed": int(seed),
                    "backbone": payload["backbone"],
                    "best_lr": payload["best_lr"],
                    "tasks": payload["tasks"],
                }
                row.update(_selected_metrics(payload, JOINT_CSV_METRICS))
                for index, value in enumerate(accuracy, start=1):
                    row[f"task_{index}_acc"] = float(value)
                rows.append(row)
    expected = len(methods) * len(selected_orders) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} joint rows, found {len(rows)}")
    return _rows_csv(rows)


def write_dataset_reports(
    search_root: Path,
    joint_root: Path,
    *,
    dataset: str,
    methods: tuple[str, ...] = FORMAL_METHODS,
    order_ids: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> tuple[Path, Path]:
    search_path = (
        Path(search_root) / "aggregate_results" / f"{dataset}_search_summary.csv"
    )
    joint_path = (
        Path(joint_root) / "aggregate_results" / f"{dataset}_joint_learning.csv"
    )
    atomic_write_text(
        search_path,
        build_dataset_search_summary(
            search_root,
            dataset=dataset,
            methods=methods,
            order_ids=order_ids,
            seeds=seeds,
        ),
    )
    atomic_write_text(
        joint_path,
        build_dataset_joint_learning(
            joint_root,
            dataset=dataset,
            methods=methods,
            order_ids=order_ids,
            seeds=seeds,
        ),
    )
    return search_path, joint_path


def _formal_summaries(
    result_root: Path,
    exp_name: str,
    dataset: str,
    method: str,
    order_id: int,
    seeds: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    seeds = get_formal_seeds(dataset) if seeds is None else tuple(seeds)
    summaries = []
    for seed in seeds:
        path = completed_summary_path(
            result_root, dataset, method, order_id, seed, exp_name
        )
        if path is None:
            raise RuntimeError(
                f"Missing formal result: {dataset}/{method}/order{order_id}/seed{seed}"
            )
        summary = json.loads(path.read_text(encoding="utf-8"))
        validate_summary(summary, allow_pending_intransigence=method == "joint")
        summaries.append(summary)
    return summaries


def _raw_summary_row(
    dataset: str, method: str, order_id: int, summary: dict[str, Any], sections: tuple[str, ...]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "dataset": dataset,
        "order": f"order{order_id}",
        "seed": int(summary["seed"]),
    }
    for section in sections:
        row.update(_flatten_report_values(section, summary[section]))
    return row


def build_search_report(search_root: Path, method: str) -> str:
    fields = [
        "method",
        "dataset",
        "order",
        "lr",
        "best_lr",
        "total_search_flops",
        "storage_bytes",
        "search_flops_mean",
        "search_flops_ci95",
        "search_flops_std",
        "storage_mean",
        "storage_ci95",
        "storage_std",
    ]
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = search_result_path(search_root, dataset, method)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["metadata"]["status"] != STATUS_COMPLETED:
            raise RuntimeError(f"Search is incomplete: {path}")
        flops = payload["search_metrics"]["search_flops_each_order"]
        storage = payload["search_metrics"]["search_storage_each_order"]
        lr_candidates = payload["metadata"]["lr_candidates"]
        for index, order_id in enumerate(sorted(ORDERS_BY_DATASET[dataset])):
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "order": f"order{order_id}",
                    "lr": json.dumps(lr_candidates, separators=(",", ":")),
                    "best_lr": payload["best_per_order"][index],
                    "total_search_flops": flops[index],
                    "storage_bytes": storage[index],
                    "search_flops_mean": "",
                    "search_flops_ci95": "",
                    "search_flops_std": "",
                    "storage_mean": "",
                    "storage_ci95": "",
                    "storage_std": "",
                }
            )
        flops_stats = _stats([float(value) for value in flops])
        storage_stats = _stats([float(value) for value in storage])
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "order": "all orders",
                "lr": "",
                "best_lr": "",
                "total_search_flops": sum(flops),
                "storage_bytes": "",
                **{f"search_flops_{key}": value for key, value in flops_stats.items()},
                **{
                    f"storage_{key}": value
                    for key, value in storage_stats.items()
                },
            }
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_train_report(result_root: Path, exp_name: str, method: str) -> str:
    rows = []
    for dataset in DATASETS:
        for order_id in sorted(ORDERS_BY_DATASET[dataset]):
            for summary in _formal_summaries(result_root, exp_name, dataset, method, order_id):
                rows.append(
                    _raw_summary_row(
                        dataset,
                        method,
                        order_id,
                        summary,
                        (
                            "training_runtime",
                            "training_operations",
                            "persistent_storage",
                            "network",
                            "working_memory_diagnostic",
                        ),
                    )
                )
    return _rows_csv(rows)


def build_test_report(result_root: Path, exp_name: str, method: str) -> str:
    rows = []
    for dataset in DATASETS:
        for order_id in sorted(ORDERS_BY_DATASET[dataset]):
            for summary in _formal_summaries(result_root, exp_name, dataset, method, order_id):
                row = {
                    "method": method,
                    "dataset": dataset,
                    "order": f"order{order_id}",
                    "seed": int(summary["seed"]),
                }
                row.update(_flatten_report_values("", summary["cil_performance"]))
                row.update(_flatten_report_values("inference", summary["inference"]))
                rows.append(row)
    return _rows_csv(rows)


def build_joint_test_report(
    result_root: Path, exp_name: str, source_method: str
) -> str:
    rows = []
    for dataset in DATASETS:
        for order_id in sorted(ORDERS_BY_DATASET[dataset]):
            for summary in _formal_summaries(
                result_root, exp_name, dataset, "joint", order_id, SEEDS_BY_DATASET[dataset]
            ):
                row = {
                    "method": "joint",
                    "dataset": dataset,
                    "order": f"order{order_id}",
                    "seed": int(summary["seed"]),
                }
                row.update(_flatten_report_values("", summary["cil_performance"]))
                row.update(_flatten_report_values("inference", summary["inference"]))
                row["checkpoint_source_method"] = source_method
                rows.append(row)
    return _rows_csv(rows)


def write_joint_test_report(
    result_root: Path, exp_name: str, source_method: str
) -> Path:
    path = Path(result_root) / "aggregate_results" / "joint_test_report.csv"
    atomic_write_text(path, build_joint_test_report(result_root, exp_name, source_method))
    return path


def write_method_reports(
    search_root: Path, result_root: Path, exp_name: str, method: str
) -> tuple[Path, Path, Path]:
    output = Path(result_root) / "aggregate_results"
    paths = (
        output / f"{method}_search_report.csv",
        output / f"{method}_train_report.csv",
        output / f"{method}_test_report.csv",
    )
    texts = (
        build_search_report(search_root, method),
        build_train_report(result_root, exp_name, method),
        build_test_report(result_root, exp_name, method),
    )
    for path, text in zip(paths, texts):
        atomic_write_text(path, text)
    return paths


def write_formal_aggregate(
    result_root: Path,
    *,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...] | None = None,
    exp_name: str,
) -> tuple[Path, Path, Path]:
    payload = aggregate_results(
        result_root,
        datasets=datasets,
        methods=methods,
        seeds=seeds,
        exp_name=exp_name,
    )
    output = Path(result_root) / "aggregate_results"
    paths = (
        output / "formal_aggregate.json",
        output / "formal_aggregate.csv",
        output / "formal_aggregate.md",
    )
    for path, text in zip(
        paths, (json_text(payload), _csv_text(payload), _markdown_text(payload))
    ):
        atomic_write_text(path, text)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate complete formal CIL runs from one result_<EXP_NAME> folder"
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=list(FORMAL_METHODS), default=list(FORMAL_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    payload = aggregate_results(
        result_root,
        datasets=tuple(args.datasets),
        methods=tuple(args.methods),
        seeds=None if args.seeds is None else tuple(args.seeds),
        exp_name=args.exp_name,
    )
    prefix = (args.output_prefix or result_root / "formal_aggregate").resolve()
    atomic_write_text(prefix.with_suffix(".json"), json_text(payload))
    atomic_write_text(prefix.with_suffix(".csv"), _csv_text(payload))
    atomic_write_text(prefix.with_suffix(".md"), _markdown_text(payload))
    print(prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
