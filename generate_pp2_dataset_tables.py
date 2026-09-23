"""Read-only PP2 reporting: exactly three tables and one Markdown per dataset.

Uses existing validated CSV/search/matrix readers; never invokes training or
changes the original aggregator. Python 3.10+, standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import re
import runpy

import generate_pp2_tables as base


def failure_cost(directory, expected, tasks):
    """Sum independently archived attempts, never infer an unfinished task cost."""
    total = 0
    notes = []
    search_path = directory / f"order{expected['order']}_seed{expected['seed']:03d}_search.json"
    candidates = base.read_json(search_path).get('candidates', {}) if search_path.is_file() else {}
    referenced = {lr for lr, c in candidates.items()
                  if c.get('status') == 'failed' and c.get('failure_manifest')}
    found = set()
    manifests = sorted(directory.glob("lr*/**/failed_runs/*/manifest.json"))
    for path in manifests:
        manifest = base.read_json(path)
        ident = manifest.get("identity", {})
        if ident.get("seed") != expected["seed"]:
            continue
        base.identity(ident, expected)
        lr = str(ident.get('lr'))
        if lr in referenced:
            chosen = str(candidates[lr]['failure_manifest']).replace('\\', '/').split('/')[-2:]
            if list(path.parts[-2:]) != chosen:
                continue
            base.require(lr not in found, 'Duplicate selected failure archive')
            found.add(lr)
        base.require(str(ident.get("lr")).replace(".", "") == path.relative_to(directory).parts[0][2:],
                     f"Failure LR/path mismatch: {path}")
        base.require(set(manifest["files"]) == {"config.json", "failure.json", "run.log"},
                     f"Invalid failure manifest: {path}")
        for name, digest in manifest["files"].items():
            base.require(hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == digest,
                         f"Failure evidence hash mismatch: {path.parent / name}")
        failure = base.read_json(path.parent / "failure.json")
        base.identity(failure["identity"], ident)
        base.require(failure["status"] == "failed", f"Not a failed attempt: {path}")
        completed = []
        cost = 0
        for line in (path.parent / "run.log").read_text(encoding="utf-8").splitlines():
            if " flops=" not in line:
                continue
            match = re.search(r"\btask=(\d+) .* flops=(\{.*\})$", line)
            base.require(match is not None, f"Malformed task FLOPs: {path}")
            import json
            record = json.loads(match[2])
            task = int(match[1])
            base.require(task == len(completed) + 1 and task <= tasks,
                         f"Duplicate/noncontiguous task FLOPs: {path}")
            value = record["total_flops"]
            base.require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                         f"Invalid failed task FLOPs: {path}")
            # PhaseFlopProfiler omits phases whose count is zero.
            phases = record["phase_flops"]
            base.require(isinstance(phases, dict)
                         and set(phases) <= {"core_training", "learning_auxiliary"}
                         and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                                 for v in phases.values())
                         and value == sum(phases.values()),
                         f"Failure phase sum mismatch: {path}")
            completed.append(task)
            cost += value
        total += cost
        notes.append(f"order={expected['order']}, seed={expected['seed']}, LR={ident['lr']}: "
                     f"失败日志任务 {completed}，计入下界 {cost / 1e12:.6f} ×10¹² FLOPs；"
                     f"失败位置 task={failure.get('partial', {}).get('task', '?')}, "
                     f"epoch={failure.get('partial', {}).get('epoch', '?')}。"
                     f"来源 `{path.relative_to(directory).as_posix()}`。")
    base.require(found == referenced, 'Missing selected failure archive')
    return total, notes


def add_search_costs(root, dataset, exp, method, records, orders, seeds, mode):
    selected = [dict(r) for r in records if r["lr"] == r["best_lr"]]
    notes = []
    for row in selected:
        order, seed = row["order"], row["seed"]
        directory = root / f"search_result_{exp}" / method / f"order{order}"
        search = base.read_json(directory / f"order{order}_seed{seed:03d}_search.json")
        candidates = search["candidates"]
        failed = any(c["status"] == "failed" for c in candidates.values())
        known = math.fsum(base.number(c["overall_learning_flops"])
                          for c in candidates.values() if c["status"] == "completed")
        if failed:
            base.require(search.get("total_search_flops") is None and
                         search.get("search_flops_status") == "incomplete_failed_candidates",
                         "Failed candidates require unknown total search FLOPs")
            base.close(search["search_flops_lower_bound"], known)
        else:
            base.close(search["total_search_flops"], known)
        partial, evidence = failure_cost(directory,
            dict(dataset=dataset, exp_name=exp, method=method, order=order, seed=seed), len(orders[order]))
        notes.extend(evidence)
        # A failed attempt remains incomplete even if a later retry succeeded.
        row["search_complete"] = not failed and not evidence
        if any(c.get("previous_failure_file") for c in candidates.values()):
            row["search_complete"] = False
            notes.append(f"order={order}, seed={seed}: 存在历史失败记录；未完整记录的历史消耗未估算。")
        if failed and not evidence:
            notes.append(f"order={order}, seed={seed}: 失败候选缺少可用 FLOPs 日志，搜索成本仅为成功候选之和的下界。")
        row["search_flops"] = known + partial
        if mode == "extra":
            row["search_flops"] -= row["overall_learning_flops"]
        base.require(row["search_flops"] >= 0, "Negative search cost")
        row["total_flops"] = (row["search_flops"] + row["core_training_flops"]
                              + row["learning_auxiliary_flops"])
    base.require(len(selected) == len(orders) * len(seeds), "Selected pair coverage mismatch")
    return selected, notes


def search_cells(group, key, mode, pooled):
    values = [r[key] / 1e12 for r in group]
    if all(r["search_complete"] for r in group):
        return list(base.formatted(values, mode)) if pooled else [base.cell(values, mode)]
    # Do not disguise varying lower bounds as measurements with SD or a CI.
    complete = sum(r["search_complete"] for r in group)
    value = f"≥ {base.fmt(math.fsum(values) / len(values))}（均值下界；完整 {complete}/{len(group)}）"
    return [value, "N/A（成本不完整）"] if pooled else [value]


def costs(groups, orders, seeds, pooled, mode):
    modes = ["z"] if pooled else ["t"] * len(orders)
    planned = [len(orders) * len(seeds)] if pooled else [len(seeds)] * len(orders)
    rows = base.cost_rows(groups, modes, planned, pooled)
    search_label = "Search overhead FLOPs, excluding best LR" if mode == "extra" else "Full search FLOPs, including best LR"
    total_label = "Total FLOPs: search overhead + core + auxiliary" if mode == "extra" else "Budget FLOPs: full search + retraining core + auxiliary"
    additions = []
    for label, key in [(search_label, "search_flops"), (total_label, "total_flops")]:
        cells = []
        for group, ci_mode in zip(groups, modes):
            cells.extend(search_cells(group, key, ci_mode, pooled))
        additions.append([label + " (×10¹²)", *cells])
    # Keep existing learning subtotal, core and auxiliary rows together.
    index = next(i for i, row in enumerate(rows) if row[0].startswith("Learning auxiliary FLOPs")) + 1
    rows[index:index] = additions
    return rows


def render(dataset, exp, methods, orders, seeds, lrs, mode):
    pooled, ordered, performance, notes = [], [], [], []
    for method, records, selected, evidence in methods:
        name = base.DISPLAY.get(method, method)
        pooled.extend([[name, *r] for r in costs([selected], orders, seeds, True, mode)])
        groups = [[r for r in selected if r["order"] == o] for o in orders]
        ordered.extend([[name, *r] for r in costs(groups, orders, seeds, False, mode)])
        performance.extend([[name, *r] for r in base.performance_rows(records, orders, seeds, lrs)])
        notes.extend(f"{name}: {n}" for n in evidence)
    convention = (
        "搜索额外成本 = 所有候选的已知学习成本 + 已保存失败日志成本 − 最佳候选学习成本。"
        "总 FLOPs = 搜索额外成本 + 最佳候选核心训练 + 最佳候选辅助计算；最佳候选只计一次。"
        if mode == "extra" else
        "完整搜索成本包含最佳候选一次。总预算 FLOPs = 完整搜索成本 + 最佳候选核心训练 + 辅助计算，"
        "表示选参后额外重训一次的预算；该重训项取最佳候选成本作为代理，不代表已发生的额外运行。")
    parts = [f"# {dataset}：所有方法汇总\n",
             f"实验 `{exp}`；orders={list(orders)}；seeds={list(seeds)}。每个 order/seed 单独选 best LR。\n",
             convention + " Total learning FLOPs = core + auxiliary，是小计，不得再次加到总计中。"
             "搜索先在同一 order/seed 内对候选求和，再跨运行统计；总计也先逐运行相加再统计，SD/CI 不直接相加。\n",
             "成本使用既有 learning FLOPs 口径，不包括 joint 参考训练、独立评估或未留存的历史尝试。"
             "失败日志只计入已完整保存的任务；中断任务不按轮数猜算。有缺口时明确报告均值下界，不提供伪精确 SD/CI。\n",
             "## 表 1A：最佳候选成本与搜索成本，汇总所有 order\n",
             base.table(["Method", "Metric", "Mean ± SD", "95% CI"], pooled),
             "## 表 1B：最佳候选成本与搜索成本，分 order\n",
             base.table(["Method", "Metric", *[f"Order {o}" for o in orders]], ordered),
             "## 表 2：方法性能\n",
             base.table(["Method", "Metric", *[f"Order {o}" for o in orders]], performance),
             "表 1A 沿用 1.96 × SD/√n；表 1B 和表 2 标量沿用跨 seed 的 Student-t 区间。"
             "SD 为样本标准差。表 2 保留原方法性能指标：bestlr 行对应最终选中结果，all 行为按 LR 分组的候选分析；"
             "失败候选不参加性能统计。逐任务 intransigence 为 Mean ± SD；遗忘分布基于各任务跨 seed 均值。\n",
             "## 搜索成本记录说明\n",
             "\n\n".join(notes) if notes else "已登记候选均完整。没有日志并不证明从未发生未留存的历史重试。",
             f"\n来源：`search_result_{exp}/aggregate_results/{dataset}_search_summary.csv`、各 order/seed 搜索 JSON、"
             "最佳候选准确率矩阵、joint 参考矩阵和可用的失败 manifest/run.log。输入文件只读。\n"]
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project with cil_experiments registries")
    parser.add_argument("--results-root", type=Path,
                        help="Results parent; accepts search_result_EXP directly or EXP/search_result_EXP")
    parser.add_argument("--output-dir", type=Path, help="Default: PROJECT/pp2_dataset_tables")
    parser.add_argument("--experiments", nargs="+", default=["spike=spike-all", "uwave=uwave-all", "texture=texture-all"])
    parser.add_argument("--search-cost-mode", choices=["extra", "full-plus-retrain"], default="extra")
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    results = (args.results_root or project).resolve()
    output = (args.output_dir or project / "pp2_dataset_tables").resolve()
    try:
        registry = runpy.run_path(str(project / "cil_experiments/order_seed_registry.py"))
        config = runpy.run_path(str(project / "cil_experiments/search_config.py"))
        documents = {}
        for spec in args.experiments:
            dataset, sep, exp = spec.partition("=")
            base.require(sep and dataset in registry["ORDERS_BY_DATASET"] and
                         re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", exp), f"Invalid experiment: {spec}")
            base.require(dataset not in documents, f"Duplicate dataset: {dataset}")
            roots = [p for p in (results, results / exp) if (p / f"search_result_{exp}").is_dir()]
            base.require(len(roots) == 1, f"Missing/ambiguous results root for {exp}: {results}")
            root = roots[0]
            for prefix in ("search_result", "joint_result"):
                base.require(not output.is_relative_to(root / f"{prefix}_{exp}"), "Output overlaps source results")
            source = root / f"search_result_{exp}/aggregate_results/{dataset}_search_summary.csv"
            with source.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            base.require({r["method"] for r in rows} == set(config["PIPELINE_METHODS"]), "Method coverage mismatch")
            orders, seeds = registry["ORDERS_BY_DATASET"][dataset], registry["SEEDS_BY_DATASET"][dataset]
            methods = []
            for method in config["PIPELINE_METHODS"]:
                records, _, _, _ = base.load_method(root, dataset, exp, method, orders, seeds, config["LR_CANDIDATES"], rows)
                selected, notes = add_search_costs(root, dataset, exp, method, records, orders, seeds, args.search_cost_mode)
                methods.append((method, records, selected, notes))
            documents[dataset] = render(dataset, exp, methods, orders, seeds, config["LR_CANDIDATES"], args.search_cost_mode)
        # Validate every dataset before writing any report.
        output.mkdir(parents=True, exist_ok=True)
        for dataset, content in documents.items():
            (output / f"{dataset}.md").write_text(content, encoding="utf-8")
        print(f"PP2_DATASET_TABLES_COMPLETE reports={len(documents)} tables={3 * len(documents)} output={output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
