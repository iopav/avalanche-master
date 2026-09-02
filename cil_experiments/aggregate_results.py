from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .intransigence import _artifact_paths
from .metrics import validate_summary
from .order_seed_registry import SEEDS
from .output import json_text
from .registry import DATASETS, FORMAL_METHODS, ORDER_IDS


SCALAR_METRICS = {
    "average_incremental_accuracy": ("cil_performance", "average_incremental_accuracy"),
    "final_average_accuracy": ("cil_performance", "final_average_accuracy"),
    "average_forgetting": ("cil_performance", "average_forgetting"),
    "overall_learning_flops": ("training_operations", "overall_learning_flops"),
    "training_runtime_s": ("training_runtime", "total_s"),
    "persistent_storage_bytes": ("persistent_storage", "total_bytes"),
    "inference_latency_ms_per_sample": ("inference", "final_latency_ms_per_sample"),
}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    staged = Path(name)
    try:
        staged.write_text(value, encoding="utf-8")
        with staged.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _mean_std(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(numbers)),
        "std": float(statistics.pstdev(numbers)),
        "n": len(numbers),
    }


def aggregate_results(
    result_root: Path,
    *,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    order_ids: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    records: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    observed_dataset_methods: set[tuple[str, str]] = set()
    for summary_path in result_root.glob("*/*/summary/*__timestamp-*__summary.json"):
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
            or order_id not in order_ids
            or seed not in seeds
        ):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_summary(summary)
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
    expected_pairs = {(order_id, seed) for order_id in order_ids for seed in seeds}
    output_groups: list[dict[str, Any]] = []
    for (dataset, method, backbone), values in sorted(grouped.items()):
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
        intransigence = [
            _mean_std(summary["cil_performance"]["intransigence"][stage] for summary in summaries)
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
        "datasets": list(datasets),
        "methods": list(methods),
        "order_ids": list(order_ids),
        "seeds": list(seeds),
        "groups": output_groups,
    }


def _csv_text(payload: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    fields = ["dataset", "method", "backbone_id", "n"]
    for metric in SCALAR_METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std"))
    fields.extend(("intransigence_mean_by_stage", "intransigence_std_by_stage"))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for group in payload["groups"]:
        row = {key: group[key] for key in ("dataset", "method", "backbone_id", "n")}
        for metric, aggregate in group["metrics"].items():
            row[f"{metric}_mean"] = aggregate["mean"]
            row[f"{metric}_std"] = aggregate["std"]
        row["intransigence_mean_by_stage"] = json.dumps(
            [entry["mean"] for entry in group["intransigence_by_stage"]]
        )
        row["intransigence_std_by_stage"] = json.dumps(
            [entry["std"] for entry in group["intransigence_by_stage"]]
        )
        writer.writerow(row)
    return stream.getvalue()


def _markdown_text(payload: dict[str, Any]) -> str:
    lines = [
        "# Formal CIL aggregate",
        "",
        f"Orders: `{payload['order_ids']}`; seeds: `{payload['seeds']}`.",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate complete formal CIL runs")
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--output-prefix", type=Path, default=Path("result/formal_aggregate"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=list(FORMAL_METHODS), default=list(FORMAL_METHODS))
    parser.add_argument("--order-ids", nargs="+", type=int, default=list(ORDER_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    payload = aggregate_results(
        args.result_root.resolve(),
        datasets=tuple(args.datasets),
        methods=tuple(args.methods),
        order_ids=tuple(args.order_ids),
        seeds=tuple(args.seeds),
    )
    prefix = args.output_prefix.resolve()
    _atomic_text(prefix.with_suffix(".json"), json_text(payload))
    _atomic_text(prefix.with_suffix(".csv"), _csv_text(payload))
    _atomic_text(prefix.with_suffix(".md"), _markdown_text(payload))
    print(prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
