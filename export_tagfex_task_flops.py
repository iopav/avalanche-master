"""Export existing TagFex task FLOPs without importing Torch or changing training.

Run after main_exp/uwave-tagfex.py (or on any existing PP2 search results).
Each log result is already task-local, not cumulative. Only completed candidate
summaries are exported; incomplete candidates have no committed public log.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import tempfile


TASK_LINE = re.compile(r"\| INFO \| task=(\d+) .* flops=(\{.*\})\s*$")
FIELDS = (
    "exp_name", "dataset", "method", "order", "seed", "lr", "task",
    "tasks", "task_learning_flops", "core_training_flops",
    "learning_auxiliary_flops", "cumulative_learning_flops",
    "terminal_epoch_flops", "terminal_epoch_samples",
    "terminal_flops_per_sample", "summary_file", "log_file",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer: {value!r}")
    return value


def candidate_rows(summary_path: Path, exp_name: str) -> list[dict]:
    summary = read_json(summary_path)
    stem = summary_path.name.removesuffix("__summary.json")
    matrix = read_json(summary_path.with_name(stem + "__accuracy-matrix.json"))
    log_path = summary_path.parent / "log" / (stem + ".log")
    config = summary["config"]
    dataset = config["dataset"]["name"]
    if summary["exp_name"] != exp_name or config["method"] != "tagfex":
        raise ValueError(f"Experiment/method mismatch: {summary_path}")
    for key, expected in {
        "exp_name": exp_name, "method": "tagfex", "dataset": dataset,
        "seed": summary["seed"], "tasks": summary["tasks"],
    }.items():
        if matrix.get(key) != expected:
            raise ValueError(f"Matrix {key} mismatch: {summary_path}")
    tasks = count(summary["tasks"], "tasks")
    if not tasks:
        raise ValueError(f"Empty task sequence: {summary_path}")
    results = []
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            match = TASK_LINE.search(line)
            if match:
                results.append((int(match[1]), json.loads(match[2])))
    if [task for task, _ in results] != list(range(1, tasks + 1)):
        raise ValueError(f"Missing, duplicate or unordered task FLOPs: {log_path}")
    rows = []
    cumulative = core_sum = auxiliary_sum = 0
    for task, result in results:
        phases = result["phase_flops"]
        if set(phases) - {"core_training", "learning_auxiliary"}:
            raise ValueError(f"Unexpected learning phases: {log_path}: task {task}")
        total = count(result["total_flops"], "total_flops")
        core = count(phases.get("core_training", 0), "core_training")
        auxiliary = count(phases.get("learning_auxiliary", 0), "learning_auxiliary")
        if total != core + auxiliary:
            raise ValueError(f"Task phase sum mismatch: {log_path}: task {task}")
        cumulative += total
        core_sum += core
        auxiliary_sum += auxiliary
        rows.append(dict(
            exp_name=exp_name, dataset=dataset, method="tagfex",
            order=matrix["order_id"], seed=summary["seed"],
            lr=config["final_hyperparameters"]["resolved"]["learning_rate"],
            task=task, tasks=tasks, task_learning_flops=total,
            core_training_flops=core, learning_auxiliary_flops=auxiliary,
            cumulative_learning_flops=cumulative,
            terminal_epoch_flops=result["terminal_epoch_flops"],
            terminal_epoch_samples=result["terminal_epoch_samples"],
            terminal_flops_per_sample=result["terminal_flops_per_sample"],
            summary_file=str(summary_path), log_file=str(log_path),
        ))
    operations = summary["training_operations"]
    for key, actual in {
        "overall_learning_flops": cumulative,
        "core_training_flops": core_sum,
        "learning_auxiliary_flops": auxiliary_sum,
    }.items():
        if count(operations[key], key) != actual:
            raise ValueError(f"Task sum differs from summary {key}: {summary_path}")
    return rows


def export(search_root: Path, exp_name: str, output: Path) -> tuple[int, int]:
    summaries = sorted((search_root / "tagfex").glob("order*/lr*/*__summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No completed TagFex candidate summaries in {search_root}")
    rows = [row for path in summaries for row in candidate_rows(path, exp_name)]
    rows.sort(key=lambda row: (row["dataset"], row["order"], row["seed"], row["lr"], row["task"]))
    identities = {(r["dataset"], r["order"], r["seed"], r["lr"], r["task"]) for r in rows}
    if len(identities) != len(rows):
        raise ValueError("Duplicate dataset/order/seed/lr/task rows")
    # Validate everything before replacing the previous report.
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tagfex-flops-", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, output)
    finally:
        Path(name).unlink(missing_ok=True)
    return len(summaries), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True, nargs="+", help="One or more experiment names; export a separate CSV for each")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--search-root", type=Path, help="Override input directory (single experiment only)")
    parser.add_argument("--output", type=Path, help="Override CSV path (single experiment only)")
    args = parser.parse_args()
    if len(set(args.exp_name)) != len(args.exp_name):
        parser.error("Experiment names must be unique")
    if len(args.exp_name) > 1 and (args.search_root or args.output):
        parser.error("--search-root and --output require exactly one experiment; use --project-root for a batch")
    if args.output and args.output.suffix.lower() != ".csv":
        parser.error("--output must be a .csv file (Excel-readable)")
    failures = []
    for exp_name in args.exp_name:
        root = (args.search_root or args.project_root / f"search_result_{exp_name}").resolve()
        output = (args.output or root / "aggregate_results" / "tagfex_task_flops.csv").resolve()
        try:
            candidates, tasks = export(root, exp_name, output)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(exp_name)
            print(f"TAGFEX_TASK_FLOPS_FAILED exp_name={exp_name}: {exc}", flush=True)
            continue
        print(f"TAGFEX_TASK_FLOPS exp_name={exp_name} candidates={candidates} task_rows={tasks} output={output}", flush=True)
    if failures:
        raise SystemExit(f"Export failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
