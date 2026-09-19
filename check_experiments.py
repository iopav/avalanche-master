"""Read-only PP2 search/joint/artifact audit (Python standard library only).

Run on the original experiment machine: stored absolute paths must still resolve.
Exit 0: complete and successful; 1: missing/invalid/pending; 2: CLI error;
3: files complete but some LR candidates failed. Checkpoints are checked as ZIP
archives (CRC included), never unpickled; model reload correctness is not tested.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import runpy
import re
import zipfile


def schema_keys(root):
    tree = ast.parse((root / "cil_experiments" / "metrics.py").read_text(encoding="utf-8-sig"))
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"SUMMARY_KEYS", "CIL_KEYS"}}


def log_check(path, tasks, operations):
    records = []
    pattern = re.compile(r"\| INFO \| task=(\d+) .* flops=(\{.*\})\s*$")
    with nonempty(path).open(encoding="utf-8") as stream:
        for line in stream:
            match = pattern.search(line)
            if match:
                records.append((int(match[1]), json.loads(match[2])))
    require([task for task, _ in records] == list(range(1, tasks + 1)), "Missing/duplicate/unordered task FLOPs logs")
    for _, record in records:
        require(record["total_flops"] == sum(record["phase_flops"].values()), "Log task FLOPs phase mismatch")
    require(sum(record["total_flops"] for _, record in records) == operations["overall_learning_flops"], "Log/summary FLOPs mismatch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size > 0, f"Missing/empty: {path}")
    return path


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "NaN/Inf in result JSON")
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def read_json(path):
    value = json.loads(nonempty(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    finite_tree(value)
    return value


def identity(value, expected):
    for key, item in expected.items():
        require(value.get(key) == item, f"{key}: expected {item!r}, got {value.get(key)!r}")


def matrix_check(payload, tasks):
    require(payload["tasks"] == tasks, "Task count differs from registry")
    matrix = payload["accuracy_matrix_lower_triangular"]
    require(len(matrix) == tasks, "Missing matrix rows")
    for i, row in enumerate(matrix):
        require(len(row) == tasks, "Missing matrix columns")
        for j, value in enumerate(row):
            require(value is None if j > i else
                    type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    f"Invalid accuracy matrix cell [{i},{j}]")
    return matrix


def operations_check(payload):
    ops = payload["training_operations"]
    for key in ("overall_learning_flops", "core_training_flops", "learning_auxiliary_flops"):
        require(type(ops[key]) is int and ops[key] >= 0, f"Invalid {key}")
    require(ops["overall_learning_flops"] == ops["core_training_flops"] +
            ops["learning_auxiliary_flops"], "FLOPs sum mismatch")
    require(payload["inference"]["final_latency_ms_per_sample"] > 0, "Invalid latency")
    require(payload["training_runtime"]["total_s"] >= 0, "Invalid runtime")
    require(payload["persistent_storage"]["total_bytes"] >= 0, "Invalid storage")


def checkpoint_check(path):
    with zipfile.ZipFile(nonempty(path)) as archive:
        names = archive.namelist()
        require(any(name.endswith("/data.pkl") for name in names), "Checkpoint missing data.pkl")
        require(any(name.endswith("/version") for name in names), "Checkpoint missing version")
        require(archive.testzip() is None, "Checkpoint CRC failure")


def failure_check(record, expected):
    require(record.get("error_type") == "FloatingPointError" and
            str(record.get("error_message", "")).startswith("Non-finite training loss "),
            "Unhandled candidate failure: " + str(record.get("error_message")))
    if expected["dataset"] != "texture" or expected["method"] != "ewc":
        return
    path = Path(record["failure_manifest"])
    manifest = read_json(path)
    identity(manifest["identity"], expected)
    require(set(manifest["files"]) == {"failure.json", "config.json", "run.log"},
            "Failure evidence file list mismatch")
    for name, digest in manifest["files"].items():
        content = nonempty(path.parent / name).read_bytes()
        require(hashlib.sha256(content).hexdigest() == digest, f"Failure hash mismatch: {name}")
    failure = read_json(path.parent / "failure.json")
    identity(failure["identity"], expected)
    identity(failure, {"status": "failed", "metrics_complete": False,
                       "error_type": record["error_type"], "error_message": record["error_message"]})


def candidate_check(folder, stem, record, expected, groups):
    summary_path = folder / (stem + "__summary.json")
    matrix_path = folder / (stem + "__accuracy-matrix.json")
    require(Path(record["summary_file"]).resolve() == summary_path.resolve(), "Summary path mismatch")
    require(Path(record["accuracy_matrix_file"]).resolve() == matrix_path.resolve(), "Matrix path mismatch")
    summary = read_json(summary_path)
    config = read_json(folder / (stem + "__config.json"))
    nonempty(folder / "log" / (stem + ".log"))
    require(summary["config"] == config, "Summary/config mismatch")
    identity(summary, {"exp_name": expected["exp_name"], "method": expected["method"], "seed": expected["seed"], "tasks": len(groups)})
    identity(config, {"exp_name": expected["exp_name"], "method": expected["method"]})
    require(config["dataset"]["name"] == expected["dataset"], "Config dataset mismatch")
    require(config["selected_task_groups"] == groups, "Config task groups mismatch")
    require(config["final_hyperparameters"]["resolved"]["learning_rate"] == expected["lr"], "LR mismatch")
    matrix = read_json(matrix_path)
    identity(matrix, {k: v for k, v in expected.items() if k not in ("lr", "order")})
    identity(matrix, {"order_id": expected["order"]})
    values = matrix_check(matrix, len(groups))
    operations_check(summary)
    log_check(folder / "log" / (stem + ".log"), len(groups), summary["training_operations"])
    require(summary["selection"]["loss"] == record["selection_loss"], "Selection loss mismatch")
    require(type(record["selection_loss"]) in (int, float), "Missing selection loss")
    require(summary["training_operations"]["overall_learning_flops"] == record["overall_learning_flops"],
            "Search/summary FLOPs mismatch")
    require(summary["selection"]["loss_eval_flops"] == record["selection_loss_eval_flops"], "Selection evaluation FLOPs mismatch")
    require(summary["persistent_storage"]["model_parameter_bytes"] == record["storage_bytes"], "Search/model storage mismatch")
    cil = summary["cil_performance"]
    curve = cil["task_end_seen_accuracy_curve"]
    require(math.isclose(cil["average_incremental_accuracy"], sum(curve) / len(groups), abs_tol=1e-8), "Average incremental accuracy mismatch")
    require(math.isclose(cil["final_average_accuracy"], curve[-1], abs_tol=1e-8), "Final accuracy mismatch")
    require(len(cil["task_end_seen_accuracy_curve"]) == len(groups), "Incomplete accuracy curve")
    require(len(cil["forgetting_per_task"]) == len(groups) - 1, "Incomplete forgetting array")
    weights = matrix["test_samples_per_task"]
    require(len(weights) == len(groups) and all(type(w) is int and w > 0 for w in weights), "Invalid test weights")
    for i, row in enumerate(values):
        accuracy = sum(row[j] * weights[j] for j in range(i + 1)) / sum(weights[:i + 1])
        require(math.isclose(accuracy, cil["task_end_seen_accuracy_curve"][i], abs_tol=1e-8),
                "Matrix/accuracy curve mismatch")
    return summary


class Audit:
    def __init__(self):
        self.rows = []

    def check(self, context, kind, path, callback):
        row = dict(context, kind=kind, path=str(path), status="OK", detail="")
        try:
            result = callback()
        except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError,
                AttributeError, zipfile.BadZipFile, RuntimeError) as exc:
            row.update(status="ERROR", detail=f"{type(exc).__name__}: {exc}")
            result = None
        self.rows.append(row)
        return result


def report_check(path, expected, exp, dataset, with_lr):
    with nonempty(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = []
    for row in rows:
        identity(row, {"exp_name": exp, "dataset": dataset})
        key = (row["method"], int(row["order"]), int(row["seed"]))
        if with_lr:
            key += (float(row["lr"]),)
        keys.append(key)
    require(len(keys) == len(set(keys)), "Duplicate CSV rows")
    require(set(keys) == expected, f"CSV coverage mismatch: missing={len(expected - set(keys))}, extra={len(set(keys) - expected)}")


def audit_experiment(audit, root, dataset, exp, registry, methods, lrs, check_reports):
    search_root = root / f"search_result_{exp}"
    joint_root = root / f"joint_result_{exp}"
    schemas = schema_keys(root)
    search_keys, joint_keys = set(), set()
    for method in methods:
        for order, task_groups in registry["ORDERS_BY_DATASET"][dataset].items():
            groups = [list(g) for g in task_groups]
            for seed in registry["SEEDS_BY_DATASET"][dataset]:
                ctx = dict(dataset=dataset, exp_name=exp, method=method, order=order, seed=seed, lr="")
                expected = {k: v for k, v in ctx.items() if k != "lr"}
                directory = search_root / method / f"order{order}"
                path = directory / f"order{order}_seed{seed:03d}_search.json"
                def load_search():
                    payload = read_json(path)
                    identity(payload, expected)
                    require(payload["lr_candidates"] == list(lrs), "LR candidates differ from registry")
                    require(set(payload["candidates"]) == {str(lr) for lr in lrs}, "Missing/extra candidate keys")
                    return payload
                search = audit.check(ctx, "search_json", path, load_search)
                for lr in lrs:
                    search_keys.add((method, order, seed, lr))
                    token = str(lr).replace(".", "")
                    folder = directory / f"lr{token}"
                    stem = f"search_{dataset}_{method}__order-{order}__lr-{token}__seed-{seed:03d}"
                    def check_candidate():
                        require(search is not None, "Search JSON unavailable/invalid")
                        record = search["candidates"][str(lr)]
                        if record["status"] == "failed":
                            failure_check(record, dict(expected, lr=lr))
                            return "FAILED"
                        require(record["status"] == "completed", f"Candidate status={record['status']}")
                        summary = candidate_check(folder, stem, record, dict(expected, lr=lr), groups)
                        require(set(summary) == schemas["SUMMARY_KEYS"], "Summary fields missing/extra")
                        require(set(summary["cil_performance"]) == schemas["CIL_KEYS"], "CIL metric fields missing/extra")
                        return "OK"
                    status = audit.check(dict(ctx, lr=lr), "candidate", folder, check_candidate)
                    if status == "FAILED":
                        audit.rows[-1].update(status="FAILED", detail=search["candidates"][str(lr)]["error_message"])
                def check_selection():
                    require(search is not None, "Search JSON unavailable/invalid")
                    require(search["status"] == "completed", f"Search status={search['status']}")
                    successful = {lr: search["candidates"][str(lr)] for lr in lrs
                                  if search["candidates"][str(lr)]["status"] == "completed"}
                    best = min(successful, key=lambda lr: (successful[lr]["selection_loss"], lrs.index(lr)))
                    require(search["best_lr"] == best and search["best_loss"] == successful[best]["selection_loss"], "Incorrect best LR/loss")
                    for key in ("summary_file", "accuracy_matrix_file"):
                        require(search["best_" + key] == successful[best][key], f"Best {key} mismatch")
                    total = sum(r["overall_learning_flops"] for r in successful.values())
                    if len(successful) == len(lrs):
                        require(search["total_search_flops"] == total, "Search FLOPs mismatch")
                    else:
                        require(search["total_search_flops"] is None and
                                search["search_flops_status"] == "incomplete_failed_candidates" and
                                search["search_flops_lower_bound"] == total, "Failed candidate FLOPs incorrectly marked complete")
                    best_path = directory / "checkpoint" / f"{dataset}__{method}__order-{order}__seed-{seed:03d}__lr-{str(best).replace('.', '')}__best.pt"
                    require(Path(search["best_checkpoint"]).resolve() == best_path.resolve(), "Best checkpoint path mismatch")
                    checkpoint_check(best_path)
                audit.check(ctx, "selection_checkpoint", path, check_selection)
                joint_keys.add((method, order, seed))
                joint_path = joint_root / method / f"order{order}" / f"joint_{dataset}_{method}__order-{order}__seed-{seed:03d}.json"
                def check_joint():
                    payload = read_json(joint_path)
                    identity(payload, dict(expected, status="completed", schema="pp2-joint-learning-v2"))
                    require(search is not None and payload["best_lr"] == search["best_lr"], "Joint/search best LR mismatch")
                    require(payload["backbone"] == search["backbone"], "Joint/search backbone mismatch")
                    require(Path(payload["config"]["source_search_json"]).resolve() == path.resolve(), "Joint source search path mismatch")
                    require(payload["task_groups"] == groups, "Joint task groups mismatch")
                    matrix = matrix_check(payload, len(groups))
                    require(payload["accuracy_by_task"] == [row[i] for i, row in enumerate(matrix)], "Joint diagonal mismatch")
                    for key in ("stage_final_epoch_train_loss", "train_samples_by_stage"):
                        require(len(payload[key]) == len(groups), f"Incomplete {key}")
                    require(payload["final_epoch_train_loss"] == payload["stage_final_epoch_train_loss"][-1], "Joint final loss mismatch")
                    operations_check(payload)
                audit.check(ctx, "joint", joint_path, check_joint)
    if check_reports:
        ctx = dict(dataset=dataset, exp_name=exp, method="", order="", seed="", lr="")
        for path, keys, with_lr in (
            (search_root / "aggregate_results" / f"{dataset}_search_summary.csv", search_keys, True),
            (joint_root / "aggregate_results" / f"{dataset}_joint_learning.csv", joint_keys, False),
        ):
            audit.check(ctx, "aggregate_csv", path, lambda: report_check(path, keys, exp, dataset, with_lr))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--experiments", nargs="+", default=["spike=spike-all", "texture=texture-all", "uwave=uwave-all"], metavar="DATASET=EXP_NAME")
    parser.add_argument("--output", type=Path, default=Path("experiment_audit.csv"))
    parser.add_argument("--skip-reports", action="store_true", help="Audit raw search/joint artifacts only, not aggregate CSVs")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.suffix.lower() != ".csv" or output.exists():
        parser.error("--output must be a new .csv file; choose another name")
    registry = runpy.run_path(str(root / "cil_experiments" / "order_seed_registry.py"))
    config = runpy.run_path(str(root / "cil_experiments" / "search_config.py"))
    experiments = []
    for item in args.experiments:
        dataset, sep, exp = item.partition("=")
        if not sep or dataset not in registry["ORDERS_BY_DATASET"] or not exp or any(c in exp for c in "/\\") or exp in (".", ".."):
            parser.error(f"Invalid DATASET=EXP_NAME: {item}")
        experiments.append((dataset, exp))
    if len(set(experiments)) != len(experiments):
        parser.error("Duplicate experiments")
    if any(output.is_relative_to(root / f"{prefix}_{exp}")
           for _, exp in experiments for prefix in ("search_result", "joint_result")):
        parser.error("Audit output must be outside experiment artifact directories")
    audit = Audit()
    for dataset, exp in experiments:
        start = len(audit.rows)
        audit_experiment(audit, root, dataset, exp, registry, config["PIPELINE_METHODS"],
                         config["LR_CANDIDATES"], not args.skip_reports)
        print(f"{dataset}/{exp}: {dict(Counter(r['status'] for r in audit.rows[start:]))}")
    # Refuse to overwrite input artifacts, including when --output is mistyped.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "exp_name", "method", "order", "seed", "lr", "kind", "status", "path", "detail"])
        writer.writeheader()
        writer.writerows(audit.rows)
    counts = Counter(row["status"] for row in audit.rows)
    verdict = "INCOMPLETE_OR_INVALID" if counts["ERROR"] else "COMPLETE_WITH_FAILED_CANDIDATES" if counts["FAILED"] else "ALL_SUCCESSFUL"
    print(f"{verdict} counts={dict(counts)} report={output}")
    print("Scope: current registry search + joint; aggregate CSV coverage unless skipped. Checkpoint ZIP/CRC only, no model reload.")
    return 1 if counts["ERROR"] else 3 if counts["FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
