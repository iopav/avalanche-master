"""Generate pp2-newtable Markdown tables from existing results; no training.

Python 3.10+, standard library only. Output is a NEW directory (never overwrite).
For each dataset/method: four cost tables, one performance table, and two
matrices per order. Sources are CSV + search JSON + selected matrix/joint JSON.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
import json
import math
from pathlib import Path
import re
import runpy
import statistics


DISPLAY = dict(ewc="EWC", er_ace="ER-ACE", cwr_star="CWRStar", icarl="ICaRL",
               fecam="FeCAM", tagfex="TagFex")
COST = [
    ("Total learning FLOPs (×10¹²)", "overall_learning_flops", 1e-12),
    ("Core training FLOPs (×10¹²)", "core_training_flops", 1e-12),
    ("Learning auxiliary FLOPs (×10¹²)", "learning_auxiliary_flops", 1e-12),
    ("Model-parameter storage (KiB)", "model_parameter_bytes", 1 / 1024),
    ("Replay-sample storage (MiB)", "replay_sample_bytes", 1 / 2**20),
    ("Replay-label storage (MiB)", "replay_label_bytes", 1 / 2**20),
    ("Auxiliary persistent storage (MiB)", "auxiliary_bytes", 1 / 2**20),
    ("Total persistent storage (MiB)", "persistent_storage_bytes", 1 / 2**20),
]
MECHANISMS = ["Pathfinding FLOPs", "Supervisory-index FLOPs", "Pruning/selection FLOPs",
              "Neuron-recruitment FLOPs", "Neuron-sharing FLOPs", "Output-synaptic-update FLOPs",
              "Residual/stopping FLOPs"]
PERFORMANCE = [
    ("Average incremental accuracy (%)", "average_incremental_accuracy"),
    ("Average forgetting per task (pp)", "average_forgetting"),
    ("Final accuracy loss for earlier tasks (pp)", "mean_final_accuracy_loss"),
    ("Backward transfer, BWT (pp)", "backward_transfer"),
]
T_CRITICAL = {2: 12.7062047364, 3: 4.3026527297, 4: 3.1824463053, 5: 2.7764451052}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(not isinstance(value, bool) and value not in (None, ""), "Missing numeric value")
    result = float(value)
    require(math.isfinite(result), f"Nonfinite numeric value: {value}")
    return result


def close(left, right):
    require(math.isclose(number(left), number(right), rel_tol=1e-8, abs_tol=1e-10),
            f"Values disagree: {left} != {right}")


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"Expected object: {path}")
    return value


def identity(payload, expected):
    for key, value in expected.items():
        require(payload.get(key) == value,
                f"Identity {key}: expected {value!r}, got {payload.get(key)!r}")


def array(text, length):
    values = json.loads(text)
    require(isinstance(values, list) and len(values) == length, f"Expected {length} array entries")
    return [number(v) for v in values]


def matrix(payload, tasks):
    require(payload["tasks"] == tasks, "Matrix task count differs from registry")
    rows = payload["accuracy_matrix_lower_triangular"]
    require(len(rows) == tasks, "Matrix row count differs")
    for i, row in enumerate(rows):
        require(len(row) == tasks, "Matrix column count differs")
        for j, value in enumerate(row):
            if j > i:
                require(value is None, "Upper triangle must be null")
            else:
                require(0 <= number(value) <= 1, f"Accuracy outside [0,1]: {value}")
    return rows


def summary(values, mode="none"):
    """Return mean, sample SD and CI bounds. n counts runs, never tasks/LRs implicitly."""
    values = [number(v) for v in values]
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    sd = statistics.stdev(values)
    critical = 1.96 if mode == "z" else T_CRITICAL.get(len(values)) if mode == "t" else None
    if mode == "t":
        require(critical is not None, "t CI supports 2–5 seeds; registry changed")
    half = critical * sd / math.sqrt(len(values)) if critical is not None else None
    return mean, sd, None if half is None else (mean - half, mean + half)


def fmt(value):
    if value is None:
        return "N/A"
    if value != 0 and abs(value) < 0.0005:
        return f"{value:.3e}"
    return f"{value:.3f}"


def formatted(values, mode="none"):
    mean, sd, interval = summary(values, mode)
    if mean is None:
        return "Missing", "Missing"
    center = f"{fmt(mean)} ± {fmt(sd)}"
    ci = f"[{fmt(interval[0])}, {fmt(interval[1])}]" if interval else (
        "待定口径" if mode == "pending" else "N/A")
    return center, ci


def cell(values, mode="t"):
    center, ci = formatted(values, mode)
    return f"{center} [{ci[1:-1]}]" if ci.startswith("[") else center


def table(headers, rows):
    def escape(value):
        return str(value).replace("|", "&#124;").replace("\n", " ")
    lines = ["| " + " | ".join(map(escape, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        require(len(row) == len(headers), "Markdown column count mismatch")
        lines.append("| " + " | ".join(map(escape, row)) + " |")
    return "\n".join(lines) + "\n"


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def load_method(root, dataset, exp, method, orders, seeds, lrs, csv_rows):
    records, failed, matrices, joints = [], [], {}, {}
    expected_keys = {(o, s, lr) for o in orders for s in seeds for lr in lrs}
    indexed = {}
    for raw in csv_rows:
        if raw["method"] != method:
            continue
        identity(raw, {"dataset": dataset, "exp_name": exp})
        key = (int(raw["order"]), int(raw["seed"]), number(raw["lr"]))
        require(key not in indexed, f"Duplicate CSV row: {method}/{key}")
        indexed[key] = raw
    require(set(indexed) == expected_keys,
            f"CSV coverage {method}: missing={sorted(expected_keys-set(indexed))}, extra={sorted(set(indexed)-expected_keys)}")
    for order, groups in orders.items():
        tasks = len(groups)
        for seed in seeds:
            expected = dict(dataset=dataset, method=method, exp_name=exp, order=order, seed=seed)
            directory = root / f"search_result_{exp}" / method / f"order{order}"
            search = read_json(directory / f"order{order}_seed{seed:03d}_search.json")
            identity(search, dict(expected, status="completed"))
            require(search["lr_candidates"] == list(lrs), "Search LR candidates differ from registry")
            best = number(search["best_lr"])
            require(best in lrs, "Unknown best LR")
            require(search["candidates"][str(best)]["status"] == "completed", "Best LR was not successful")
            joint_path = root / f"joint_result_{exp}" / method / f"order{order}" / f"joint_{dataset}_{method}__order-{order}__seed-{seed:03d}.json"
            joint = read_json(joint_path)
            identity(joint, dict(expected, status="completed", best_lr=best))
            require(joint["task_groups"] == [list(g) for g in groups], "Joint task groups differ")
            joints[order, seed] = matrix(joint, tasks)
            selected = None
            for lr in lrs:
                raw = indexed[order, seed, lr]
                close(raw["best_lr"], best)
                source = search["candidates"][str(lr)]
                if source["status"] == "failed":
                    require(source.get("error_type") == "FloatingPointError" and
                            source.get("error_message", "").startswith("Non-finite training loss "),
                            f"Unresolved failure {method}/{order}/{seed}/{lr}")
                    require(raw.get("selection_loss", "") == "", "Failed row still contains performance metrics; refresh CSV")
                    failed.append((order, seed, lr, source["error_message"]))
                    continue
                require(source["status"] == "completed", f"Incomplete candidate {order}/{seed}/{lr}")
                row = dict(order=order, seed=seed, lr=lr, best_lr=best)
                for _, key, _ in COST:
                    row[key] = number(raw[key])
                    require(row[key] >= 0, f"Negative {key}")
                for _, key in PERFORMANCE:
                    row[key] = number(raw[key])
                row["intransigence"] = array(raw["intransigence"], tasks)
                row["intransigence_mean"] = number(raw["intransigence_mean"])
                row["forgetting_per_task"] = array(raw["forgetting_per_task"], tasks - 1)
                close(raw["selection_loss"], source["selection_loss"])
                close(row["overall_learning_flops"], source["overall_learning_flops"])
                close(row["overall_learning_flops"], row["core_training_flops"] + row["learning_auxiliary_flops"])
                close(row["persistent_storage_bytes"], sum(row[k] for k in
                      ("model_parameter_bytes", "replay_sample_bytes", "replay_label_bytes", "auxiliary_bytes")))
                close(row["intransigence_mean"], statistics.fmean(row["intransigence"]))
                close(row["average_forgetting"], statistics.fmean(row["forgetting_per_task"]))
                close(row["backward_transfer"], -row["mean_final_accuracy_loss"])
                close(raw["intransigence_reference_lr"], best)
                records.append(row)
                if lr == best:
                    selected = row
            require(selected is not None, "Missing selected CSV row")
            token = str(best).replace(".", "")
            path = directory / f"lr{token}" / f"search_{dataset}_{method}__order-{order}__lr-{token}__seed-{seed:03d}__accuracy-matrix.json"
            payload = read_json(path)
            matrix_expected = dict(expected)
            matrix_expected["order_id"] = matrix_expected.pop("order")
            identity(payload, matrix_expected)
            matrices[order, seed] = matrix(payload, tasks)
            for task in range(tasks):
                close(selected["intransigence"][task], joints[order, seed][task][task] - matrices[order, seed][task][task])
            for task in range(tasks - 1):
                close(selected["forgetting_per_task"][task],
                      max(matrices[order, seed][k][task] for k in range(task, tasks)) - matrices[order, seed][-1][task])
    return records, failed, matrices, joints


def cost_rows(groups, modes, planned, pooled):
    counts = [f"{len(g)} / {n}" for g, n in zip(groups, planned)]
    rows = [["Valid runs / planned runs", *counts, *( ["—"] if pooled else [])]]
    # Keep requested rows; absent instrumentation is never invented or zero-filled.
    entries = [(label, None, 1) for label in MECHANISMS] + COST[:3] + [
        ("Replay-attributed FLOPs, subset", None, 1),
        ("Replay-attributed fraction (%)", None, 1)] + COST[3:]
    for label, key, scale in entries:
        if key is None:
            rows.append([label] + (["Not measured", "—"] if pooled else ["Not measured"] * len(groups)))
        elif pooled:
            center, ci = formatted([r[key] * scale for r in groups[0]], modes[0])
            rows.append([label, center, ci])
        else:
            rows.append([label] + [cell([r[key] * scale for r in g], mode)
                                  for g, mode in zip(groups, modes)])
    return rows


def task_list(rows):
    if not rows:
        return "Missing (n=0)"
    return "[" + "; ".join(cell([r["intransigence"][i] * 100 for r in rows], "none")
                           for i in range(len(rows[0]["intransigence"]))) + f"] (n={len(rows)})"


def performance_rows(records, orders, seeds, lrs):
    selected = [[r for r in records if r["order"] == o and r["lr"] == r["best_lr"]] for o in orders]
    rows = [["Valid selected runs / planned runs", *[f"{len(g)} / {len(seeds)}" for g in selected]]]
    for label, key in PERFORMANCE:
        rows.append([label, *[cell([r[key] * 100 for r in g]) for g in selected]])
    rows.append(["Forward transfer, FWT", *["Not measured" for _ in orders]])
    rows.append(["Intransigence_bestlr, per task (pp)", *[task_list(g) for g in selected]])
    all_groups = [[[r for r in records if r["order"] == o and r["lr"] == lr] for lr in lrs] for o in orders]
    rows.append(["Intransigence_all, per task (pp), grouped by LR", *[
        "<br>".join(f"LR={lr}: {task_list(g)}" for lr, g in zip(lrs, order_groups)) for order_groups in all_groups]])
    rows.append(["Intransigence_bestlr mean (pp)", *[cell([r["intransigence_mean"] * 100 for r in g]) for g in selected]])
    rows.append(["Intransigence_all mean (pp), grouped by LR", *[
        "<br>".join(f"LR={lr}: {cell([r['intransigence_mean'] * 100 for r in g])} (n={len(g)})"
                   for lr, g in zip(lrs, order_groups)) for order_groups in all_groups]])
    task_means = [[statistics.fmean(r["forgetting_per_task"][i] for r in g) * 100
                   for i in range(len(g[0]["forgetting_per_task"]))] for g in selected]
    rows.append(["Forgetting distribution, median [Q1, Q3] (pp)", *[
        f"{fmt(quantile(v, .5))} [{fmt(quantile(v, .25))}, {fmt(quantile(v, .75))}]" for v in task_means]])
    rows.append(["Forgetting range across tasks, [min, max] (pp)", *[
        f"[{fmt(min(v))}, {fmt(max(v))}]" for v in task_means]])
    return rows


def matrix_table(matrices, order, seeds, tasks):
    rows = []
    for i in range(tasks):
        rows.append([f"After task {i+1}", *[
            cell([matrices[order, s][i][j] * 100 for s in seeds], "none") if j <= i else "—"
            for j in range(tasks)]])
    return table(["Training stage / Test task", *[f"Task {j+1}" for j in range(tasks)]], rows)


def render(dataset, exp, method, records, failed, matrices, joints, orders, seeds, lrs):
    selected = [r for r in records if r["lr"] == r["best_lr"]]
    parts = [f"# {dataset} — {DISPLAY.get(method, method)}\n",
             f"实验：`{exp}`。每个 order、seed 分别选 best LR。有效候选 {len(records)}/{len(orders)*len(seeds)*len(lrs)}，"
             f"已记录失败候选 {len(failed)}，有效选中运行 {len(selected)}/{len(orders)*len(seeds)}。\n"]
    for order, seed, lr, reason in failed:
        parts.append(f"失败记录：order={order}, seed={seed}, LR={lr}；{html.escape(reason)}。\n")
    by_order = [f"Order {o}" for o in orders]
    variants = [
        ("1A", "仅 best LR，汇总所有 order", [selected], ["z"], [len(orders)*len(seeds)], True),
        ("1B", "仅 best LR，分 order", [[r for r in selected if r["order"] == o] for o in orders], ["t"]*len(orders), [len(seeds)]*len(orders), False),
        ("1C", "全部搜索候选，汇总所有 order", [records], ["pending"], [len(orders)*len(seeds)*len(lrs)], True),
        ("1D", "全部搜索候选，分 order", [[r for r in records if r["order"] == o] for o in orders], ["none"]*len(orders), [len(seeds)*len(lrs)]*len(orders), False),
    ]
    for name, title, groups, modes, planned, pooled in variants:
        parts.append(f"## 表 {name}：{title}\n")
        parts.append(table(["Metric", "Mean ± SD", "95% CI"] if pooled else ["Metric", *by_order],
                           cost_rows(groups, modes, planned, pooled)))
    parts.append("成本表：FLOPs 缩放为 ×10¹²，持久存储按 KiB/MiB 换算。SD 为样本 SD；未单独计量的机制项保留为 Not measured。"
                 "全部候选表展示单次候选运行的成本分布，不是搜索总开销，也不包括 joint 训练成本。\n")
    parts.append("CI：表 1A 使用 1.96 × SD/√n，复现老师成本表示例的反推口径；表 1B 和表 2 使用固定 order 下的 Student-t 区间。"
                 "表 1C/1D 的 CI 待定，不将 LR 当作独立重复。仅一个有效观测时 SD/CI 为 N/A。\n")
    parts.append("## 表 2：方法性能\n")
    parts.append(table(["Metric", *by_order], performance_rows(records, orders, seeds, lrs)))
    parts.append("表 2：标量为 Mean ± SD [95% CI]，逐任务列表为 Mean ± SD。百分比和百分点均已乘 100。"
                 "Intransigence 为同 seed 的 joint[t,t] − candidate[t,t]，任务均值包含首任务。"
                 "All 行按固定 LR 分开统计，失败候选不补零；任务遗忘分布先逐任务跨 seed 求均值，再计算分位数和范围。\n")
    for order, groups in orders.items():
        parts.append(f"## 表 3A：增量准确率矩阵，Order {order}\n")
        parts.append(f"任务类别分组：`{json.dumps(groups)}`；seed：`{list(seeds)}`；每格为 Mean ± SD (%)。\n")
        parts.append(matrix_table(matrices, order, seeds, len(groups)))
        parts.append(f"## 表 3B：{DISPLAY.get(method, method)} 对应 joint 参考矩阵，Order {order}\n")
        parts.append(matrix_table(joints, order, seeds, len(groups)))
    parts.append("来源：该实验的 search_summary.csv、各 order/seed 搜索 JSON、选中候选 accuracy-matrix.json 和对应 joint JSON。"
                 "读取路径按当前结果根目录重建，不依赖原记录中的服务器绝对路径。\n")
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent,
                        help="Root containing registries and search_result_*/joint_result_* directories")
    parser.add_argument("--experiments", nargs="+", default=["spike=spike-all", "uwave=uwave-all", "texture=texture-all"])
    parser.add_argument("--output-dir", type=Path, help="NEW output directory; defaults to pp2_tables/TIMESTAMP under project root")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output = (args.output_dir or root / "pp2_tables" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new --output-dir")
    registry = runpy.run_path(str(root / "cil_experiments/order_seed_registry.py"))
    config = runpy.run_path(str(root / "cil_experiments/search_config.py"))
    experiments = []
    for item in args.experiments:
        dataset, sep, exp = item.partition("=")
        if not sep or dataset not in registry["ORDERS_BY_DATASET"] or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", exp):
            parser.error(f"Invalid dataset=experiment: {item}")
        if any(output.is_relative_to(root / f"{prefix}_{exp}") for prefix in ("search_result", "joint_result")):
            parser.error("Output must be outside source result directories")
        experiments.append((dataset, exp))
    if len({exp for _, exp in experiments}) != len(experiments):
        parser.error("Duplicate experiment names")
    documents, errors, warnings = {}, [], []
    for dataset, exp in experiments:
        source = root / f"search_result_{exp}/aggregate_results/{dataset}_search_summary.csv"
        try:
            with source.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            require(rows, f"Empty CSV: {source}")
            require({r["method"] for r in rows} == set(config["PIPELINE_METHODS"]), "CSV method coverage mismatch")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{exp}: {exc}")
            continue
        orders = registry["ORDERS_BY_DATASET"][dataset]
        seeds = registry["SEEDS_BY_DATASET"][dataset]
        for method in config["PIPELINE_METHODS"]:
            try:
                data = load_method(root, dataset, exp, method, orders, seeds, config["LR_CANDIDATES"], rows)
                records, failed, matrices, joints = data
                documents[f"{exp}/{method}.md"] = render(dataset, exp, method, *data, orders, seeds, config["LR_CANDIDATES"])
                if failed:
                    warnings.append(f"{exp}/{method}: {len(failed)} failed candidates (excluded, never zero-filled)")
                print(f"TABLES_VALIDATED {exp}/{method} tables={5+2*len(orders)} successful_candidates={len(records)} failed={len(failed)}", flush=True)
            except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
                errors.append(f"{exp}/{method}: {type(exc).__name__}: {exc}")
    if errors:
        for error in errors:
            print(f"ERROR {error}", flush=True)
        print("No output written. Fix incomplete/stale/mismatched input files and retry.")
        return 1
    total = sum((5 + 2*len(registry["ORDERS_BY_DATASET"][d])) * len(config["PIPELINE_METHODS"]) for d, _ in experiments)
    index = [f"# PP2 tables\n\n{len(documents)} reports, {total} tables.\n",
             "Four cost variants + one performance table + incremental/joint matrices for each order.\n"]
    index.extend(f"- [{name}]({name})" for name in documents)
    index.extend(["\n## Known failed candidates\n", *(warnings or ["None."])])
    index.append("\nSource files were read only. Missing/invalid input aborts output. Cost 1A uses the teacher-example z approximation; by-order selected-run tables use t intervals; all-LR cost CIs remain unspecified.\n")
    documents["index.md"] = "\n".join(index)
    output.mkdir(parents=True, exist_ok=False)
    for relative, content in documents.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for warning in warnings:
        print(f"WARNING {warning}")
    print(f"PP2_TABLES_COMPLETE tables={total} reports={len(documents)-1} index={output / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
