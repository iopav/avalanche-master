from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from common import REPORT_ROOT, run_joint_report, run_manual_report
from spike__ewc import PARAMETERS as SPIKE_EWC
from spike__lwf import PARAMETERS as SPIKE_LWF
from spike__si import PARAMETERS as SPIKE_SI
from texture__ewc import PARAMETERS as TEXTURE_EWC
from texture__lwf import PARAMETERS as TEXTURE_LWF
from texture__si import PARAMETERS as TEXTURE_SI
from uwave__ewc import PARAMETERS as UWAVE_EWC
from uwave__lwf import PARAMETERS as UWAVE_LWF
from uwave__si import PARAMETERS as UWAVE_SI


DATASETS = ("spike", "texture", "uwave")
PAIRED_METHODS = ("ewc", "si", "lwf")
METHODS = tuple(
    name
    for paired_method in PAIRED_METHODS
    for name in (
        paired_method,
        f"naive_{paired_method}",
        f"joint_{paired_method}",
    )
)
SHARED = {
    "learning_rate": 0.01,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "train_mb_size": 32,
    "eval_mb_size": 128,
}
PARAMETERS = {
    ("spike", "ewc"): SPIKE_EWC,
    ("spike", "lwf"): SPIKE_LWF,
    ("spike", "si"): SPIKE_SI,
    ("texture", "ewc"): TEXTURE_EWC,
    ("texture", "lwf"): TEXTURE_LWF,
    ("texture", "si"): TEXTURE_SI,
    ("uwave", "ewc"): UWAVE_EWC,
    ("uwave", "lwf"): UWAVE_LWF,
    ("uwave", "si"): UWAVE_SI,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _result_key(result: dict[str, Any]) -> tuple[str, str]:
    return str(result["dataset"]), str(result["method"])


def _write_markdown(path: Path, results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    by_key = {_result_key(result): result for result in results}
    lines = [
        f"# reg_exp order {config['order_id']} 全部实验结果",
        "",
        "## 运行口径",
        "",
        f"- 数据集：`{', '.join(DATASETS)}`",
        f"- 方法：`{', '.join(METHODS)}`",
        f"- order / seed / 每个 experience 的固定 epoch：`{config['order_id']} / {config['seed']} / {config['epochs']}`",
        "- 未使用验证集和早停；测试集只在每个 experience 结束后观测，不参与反向传播、参数更新或模型选择。",
        "- ‘最后 task ACC’是最终模型在最后一个 task 测试集上的样本准确率。",
        "- ‘前序 tasks ACC’合并最后一个 task 之前的全部测试样本后计算，不包含最后一个 task。",
        "- ‘全部 tasks ACC’合并全部 task 的测试样本后计算；三项均按测试样本数加权。",
        "",
        "## 最终 task 汇总",
        "",
        "| 数据集 | 方法 | 最后 task ACC | 前序 tasks ACC（不含最后 task） | 全部 tasks 平均 ACC | 平均增量 ACC | Epoch | 训练时间 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            result = by_key[(dataset, method)]
            lines.append(
                "| "
                + " | ".join(
                    (
                        dataset,
                        method.upper() if method in {"ewc", "si", "er"} else method,
                        f"{100 * result['final_last_task_accuracy']:.2f}%",
                        f"{100 * result['final_previous_tasks_accuracy']:.2f}%",
                        f"{100 * result['final_sample_weighted_accuracy']:.2f}%",
                        f"{100 * result['average_incremental_accuracy']:.2f}%",
                        str(result["epochs_per_experience"]),
                        f"{result['total_training_seconds']:.1f}s",
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## 三数据集等权平均",
            "",
            "| 方法 | 最后 task ACC | 前序 tasks ACC（不含最后 task） | 全部 tasks 平均 ACC |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        values = [by_key[(dataset, method)] for dataset in DATASETS]
        lines.append(
            f"| {method.upper() if method in {'ewc', 'si', 'er'} else method} | "
            f"{100 * sum(v['final_last_task_accuracy'] for v in values) / len(values):.2f}% | "
            f"{100 * sum(v['final_previous_tasks_accuracy'] for v in values) / len(values):.2f}% | "
            f"{100 * sum(v['final_sample_weighted_accuracy'] for v in values) / len(values):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            f"这是 order {config['order_id']}、seed {config['seed']} 的单次初步实验，只能用于检查方法在当前固定训练预算下的表现；不能据此声称统计显著性或普遍优劣。",
            "",
        ]
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order-id", type=int, default=6)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.order_id <= 0:
        raise ValueError("order-id must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = REPORT_ROOT / f"order-{args.order_id:02d}__all_results.json"
    markdown_path = REPORT_ROOT / f"order-{args.order_id:02d}__all_results.md"
    config = {
        "order_id": args.order_id,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": args.device,
        "validation": None,
        "early_stopping": False,
        "test_usage": "observation_after_each_experience_only",
    }
    results: list[dict[str, Any]] = []
    if state_path.exists() and not args.overwrite:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("config") != config:
            raise RuntimeError(
                f"Existing state has a different configuration: {state_path}. Use --overwrite."
            )
        results = list(previous.get("results", []))
    completed = {_result_key(result) for result in results}

    started = time.perf_counter()
    for dataset in DATASETS:
        for method in METHODS:
            key = (dataset, method)
            if key in completed:
                print(f"skip_completed dataset={dataset} method={method}", flush=True)
                continue
            print(
                f"start dataset={dataset} method={method} order={args.order_id} "
                f"seed={args.seed} epochs={args.epochs}",
                flush=True,
            )
            if method.startswith("joint_"):
                paired_method = method.removeprefix("joint_")
                shared_parameters = {
                    name: value
                    for name, value in PARAMETERS[(dataset, paired_method)].items()
                    if name in {"learning_rate", "momentum", "weight_decay", "train_mb_size", "eval_mb_size"}
                }
                result = run_joint_report(
                    dataset=dataset,
                    parameters=shared_parameters,
                    order_id=args.order_id,
                    seed=args.seed,
                    epochs=args.epochs,
                    device=args.device,
                    result_name=method,
                )
            elif method.startswith("naive_"):
                paired_method = method.removeprefix("naive_")
                shared_parameters = {
                    name: value
                    for name, value in PARAMETERS[(dataset, paired_method)].items()
                    if name in {"learning_rate", "momentum", "weight_decay", "train_mb_size", "eval_mb_size"}
                }
                result = run_manual_report(
                    dataset=dataset,
                    method="naive",
                    parameters=shared_parameters,
                    order_id=args.order_id,
                    seed=args.seed,
                    epochs=args.epochs,
                    device=args.device,
                    result_name=method,
                )
            else:
                parameters = dict(PARAMETERS[key])
                result = run_manual_report(
                    dataset=dataset,
                    method=method,
                    parameters=parameters,
                    order_id=args.order_id,
                    seed=args.seed,
                    epochs=args.epochs,
                    device=args.device,
                )
            results.append(result)
            completed.add(key)
            _atomic_json(state_path, {"config": config, "results": results})
            print(
                f"done dataset={dataset} method={method} "
                f"all_macro={result['all_class_macro_accuracy']:.6f} "
                f"last_task={result['final_last_task_accuracy']:.6f} "
                f"previous_tasks={result['final_previous_tasks_accuracy']:.6f}",
                flush=True,
            )

    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    if completed != expected:
        raise RuntimeError(f"Incomplete aggregate run: missing={sorted(expected - completed)}")
    _write_markdown(markdown_path, results, config)
    print(f"aggregate_json={state_path}", flush=True)
    print(f"aggregate_markdown={markdown_path}", flush=True)
    print(f"aggregate_wall_seconds={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
