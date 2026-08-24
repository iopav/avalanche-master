from __future__ import annotations

import gc
from itertools import product
from pprint import pformat

import numpy as np
import torch

from common import PROJECT_ROOT, _evaluate, _format_accuracy_matrix, _set_determinism

from cil_experiments.final_hyperparameters import get_final_hyperparameters
from cil_experiments.metrics import compute_cil_metrics
from cil_experiments.registry import DATASETS
from cil_experiments.strategies import build_strategy
from cil_experiments.validation import build_internal_validation_benchmark


# 所有候选使用相同的训练集内部验证划分和相同初始化，保证比较公平。
# 这里只寻找当前网格内的最优值；如果最佳学习率落在边界，需向边界外扩展后复查。
VALIDATION_SEED = 62
VALIDATION_FRACTION = 0.2
EPOCHS = 3
DEVICE = "cuda"

# FeCAM 只在第一个 Experience 训练共享骨干，因此学习率主要影响冻结前的特征质量。
LEARNING_RATES = (0.003, 0.01, 0.03)

# tukey 固定为 False：当前共享骨干输出可能为负，分数次幂会产生 NaN。
# shrinkage=False 时 shrink1/shrink2 不参与计算，所以统一写为 0，避免重复候选。
COVARIANCE_CONFIGS = (
    {"shrinkage": False, "shrink1": 0.0, "shrink2": 0.0, "covnorm": False},
    {"shrinkage": False, "shrink1": 0.0, "shrink2": 0.0, "covnorm": True},
    {"shrinkage": True, "shrink1": 0.25, "shrink2": 0.25, "covnorm": False},
    {"shrinkage": True, "shrink1": 0.25, "shrink2": 0.25, "covnorm": True},
    {"shrinkage": True, "shrink1": 1.0, "shrink2": 1.0, "covnorm": False},
    {"shrinkage": True, "shrink1": 1.0, "shrink2": 1.0, "covnorm": True},
    {"shrinkage": True, "shrink1": 2.0, "shrink2": 2.0, "covnorm": False},
    {"shrinkage": True, "shrink1": 2.0, "shrink2": 2.0, "covnorm": True},
)


def _candidate_grid() -> list[dict[str, object]]:
    return [
        {"learning_rate": learning_rate, "tukey": False, **covariance}
        for learning_rate, covariance in product(LEARNING_RATES, COVARIANCE_CONFIGS)
    ]


def _resolve_device() -> torch.device:
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _run_candidate(candidate: dict[str, object], device: torch.device) -> dict[str, object]:
    # 每个候选都重置随机状态并重建 benchmark/model，不能继承前一候选的权重或统计量。
    _set_determinism(VALIDATION_SEED)
    benchmark, _ = build_internal_validation_benchmark(
        PROJECT_ROOT / "dataset",
        "spike",
        VALIDATION_SEED,
        validation_fraction=VALIDATION_FRACTION,
        order_id=1,
    )
    parameters = get_final_hyperparameters(
        "spike", "fecam", {**candidate, "epochs_per_experience": EPOCHS}
    )
    bundle = build_strategy(
        "fecam",
        DATASETS["spike"].in_channels,
        EPOCHS,
        device,
        "spike",
        enable_flop_accounting=False,
        resolved_parameters=parameters,
    )
    task_count = len(benchmark.train_stream)
    matrix = np.full((task_count, task_count), np.nan, dtype=np.float64)
    try:
        for task_index, experience in enumerate(benchmark.train_stream):
            bundle.strategy.train(
                experience,
                num_workers=int(parameters["num_workers"]),
                pin_memory=device.type == "cuda",
            )
            for validation_index in range(task_index + 1):
                matrix[task_index, validation_index] = _evaluate(
                    bundle.strategy.model,
                    benchmark.test_stream[validation_index],
                    device,
                    int(parameters["eval_mb_size"]),
                )
        if not np.isfinite(matrix[np.tril_indices_from(matrix)]).all():
            raise FloatingPointError(f"Non-finite validation accuracy for candidate {candidate}")
        validation_counts = [len(experience.dataset) for experience in benchmark.test_stream]
        metrics = compute_cil_metrics(matrix, validation_counts)
        return {
            "parameters": candidate.copy(),
            "final_average_accuracy": float(metrics["final_average_accuracy"]),
            "average_incremental_accuracy": float(metrics["average_incremental_accuracy"]),
            "matrix": matrix.copy(),
        }
    finally:
        del bundle
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    device = _resolve_device()
    candidates = _candidate_grid()
    print(
        f"Spike FeCAM train-only validation search: candidates={len(candidates)} "
        f"validation_seed={VALIDATION_SEED} validation_fraction={VALIDATION_FRACTION:.0%} "
        f"epochs={EPOCHS} device={device}",
        flush=True,
    )
    results: list[dict[str, object]] = []
    for candidate_id, candidate in enumerate(candidates, start=1):
        result = _run_candidate(candidate, device)
        result["candidate_id"] = candidate_id
        results.append(result)
        print(
            f"[{candidate_id:02d}/{len(candidates):02d}] "
            f"final={result['final_average_accuracy'] * 100:.2f}% "
            f"incremental={result['average_incremental_accuracy'] * 100:.2f}% "
            f"params={candidate}",
            flush=True,
        )

    best = max(
        results,
        key=lambda item: (
            item["final_average_accuracy"],
            item["average_incremental_accuracy"],
            -item["candidate_id"],
        ),
    )
    print("\n网格内最佳参数：")
    print(pformat(best["parameters"], sort_dicts=False))
    print(
        f"验证集最终加权平均准确率: {best['final_average_accuracy'] * 100:.2f}%\n"
        f"验证集平均增量准确率: {best['average_incremental_accuracy'] * 100:.2f}%"
    )
    print(_format_accuracy_matrix(best["matrix"]))
    if best["parameters"]["learning_rate"] in {
        min(LEARNING_RATES),
        max(LEARNING_RATES),
    }:
        print("警告：最佳 learning_rate 位于当前网格边界，正式锁定前应向该方向扩展搜索。")


if __name__ == "__main__":
    main()
