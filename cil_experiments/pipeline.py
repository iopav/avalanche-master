from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from .output import atomic_write_json
from .registry import DATASETS
from .search_schema import STATUS_COMPLETED, search_result_path, timestamp


def _device(gpu: str):
    if gpu.lower() == "cpu":
        import torch

        return torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda:0")


def _search_payload(search_root: Path, dataset: str, method: str):
    path = search_result_path(search_root, dataset, method)
    if not path.is_file():
        raise FileNotFoundError(f"Missing LR search result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("metadata", {}).get("status") != STATUS_COMPLETED:
        raise RuntimeError(f"LR search is incomplete: {path}")
    return payload, path


def _optimum_lrs(
    search_root: Path, dataset: str, method: str, order_ids: tuple[int, ...]
) -> list[float]:
    path = search_root / "optimum_lrs.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing optimum LR file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        values = [float(value) for value in payload[dataset][method]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing optimum LRs for {dataset}/{method} in {path}") from exc
    if len(values) != len(order_ids):
        raise ValueError(f"Optimum LR count differs from registered orders for {dataset}")
    return values


def _write_formal_error(
    result_root: Path,
    exp_name: str,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    selected_lr: float,
    exc: BaseException,
) -> Path:
    path = result_root / "errors" / dataset / method / f"order{order_id}_seed{seed}.json"
    atomic_write_json(
        path,
        {
            "exp_name": exp_name,
            "dataset": dataset,
            "method": method,
            "order": order_id,
            "seed": seed,
            "selected_lr": selected_lr,
            "status": "failed",
            "error": {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_stage": "formal_training_and_test",
                "timestamp": timestamp(),
            },
        },
    )
    return path


def run_method_pipeline(
    *, method: str, exp_name: str, gpu: str, backbone: str
) -> int:
    if method == "joint":
        raise ValueError("Joint has no independent LR search; run a normal method pipeline")
    parser = argparse.ArgumentParser(description=f"运行 {method} 的 PP2 实验流水线")
    parser.add_argument(
        "--stage",
        choices=("all", "search", "joint", "formal", "aggregate"),
        default="all",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--skip-intransigence", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = (args.dataset_root or project_root / "dataset").resolve()
    search_root = project_root / f"search_result_{exp_name}"
    joint_result_root = project_root / "joint_result" / exp_name
    result_root = project_root / f"result_{exp_name}"
    device = _device(gpu) if args.stage != "aggregate" else None

    from .aggregate_results import (
        write_formal_aggregate,
        write_joint_test_report,
        write_method_reports,
    )
    from .lr_search import formal_parameters, run_search, selected_checkpoint
    from .order_seed_registry import FORMAL_SEEDS, ORDERS_BY_DATASET, SEEDS
    from .runner import evaluate_joint_checkpoint, run_experiment

    for dataset in args.datasets:
        order_ids = tuple(sorted(ORDERS_BY_DATASET[dataset]))
        if args.stage in {"all", "search"}:
            print(
                run_search(
                    project_root, dataset_root, search_root, exp_name,
                    dataset, method, device, backbone,
                )
            )
        if args.stage not in {"all", "joint", "formal"}:
            continue
        search, search_path = _search_payload(search_root, dataset, method)
        optimum_lrs = _optimum_lrs(search_root, dataset, method, order_ids)
        if optimum_lrs != [float(value) for value in search["best_per_order"]]:
            raise ValueError(f"Optimum LR file differs from {search_path}")
        for index, order_id in enumerate(order_ids):
            selected_lr = optimum_lrs[index]
            checkpoint = selected_checkpoint(search, order_id)
            for joint_seed in SEEDS:
                print(
                    evaluate_joint_checkpoint(
                        dataset_root=dataset_root,
                        result_root=joint_result_root,
                        dataset_name=dataset,
                        order_id=order_id,
                        seed=joint_seed,
                        device=device,
                        checkpoint_path=checkpoint,
                        exp_name=exp_name,
                        source_method=method,
                    )
                )
            if args.stage == "joint":
                continue
            parameters = formal_parameters(dataset, method, selected_lr)
            provenance = {
                "status": "selected_by_fixed_validation_lr_search",
                "search_result_file": str(search_path.resolve()),
                "search_seed": search["metadata"]["search_seed"],
                "selected_lr": selected_lr,
                "hyperparameter_search_flops": search["search_metrics"][
                    "search_flops_each_order"
                ][index],
                "test_set_used_for_selection": False,
            }
            for seed in FORMAL_SEEDS:
                try:
                    path = run_experiment(
                        project_root=project_root,
                        dataset_root=dataset_root,
                        result_root=result_root,
                        dataset_name=dataset,
                        method=method,
                        order_id=order_id,
                        seed=seed,
                        epochs=None,
                        device=device,
                        parameter_overrides=parameters,
                        search_provenance=provenance,
                        backbone_id=backbone,
                        compute_intransigence_enabled=not args.skip_intransigence,
                        exp_name=exp_name,
                        resume=True,
                        data_role="formal",
                        joint_result_root=joint_result_root,
                        measure_latency=True,
                    )
                    print(path)
                except BaseException as exc:
                    print(
                        _write_formal_error(
                            result_root, exp_name, dataset, method,
                            order_id, seed, selected_lr, exc,
                        )
                    )
                    raise

    if args.stage in {"all", "aggregate"}:
        if set(args.datasets) != set(DATASETS):
            raise ValueError("Aggregate stage requires all registered datasets")
        for path in write_method_reports(search_root, result_root, exp_name, method):
            print(path)
        for path in write_formal_aggregate(
            result_root,
            datasets=tuple(DATASETS),
            methods=(method,),
            seeds=FORMAL_SEEDS,
            exp_name=exp_name,
        ):
            print(path)
        print(write_joint_test_report(joint_result_root, exp_name, method))
    return 0


def run_joint_checkpoint_pipeline(
    *, source_method: str, exp_name: str, gpu: str
) -> int:
    """Evaluate an existing method search checkpoint without training joint."""

    parser = argparse.ArgumentParser(description="测试搜索阶段保存的最佳 checkpoint")
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = (args.dataset_root or project_root / "dataset").resolve()
    search_root = project_root / f"search_result_{exp_name}"
    result_root = project_root / "joint_result" / exp_name
    device = _device(gpu)

    from .lr_search import selected_checkpoint
    from .order_seed_registry import ORDERS_BY_DATASET, SEEDS
    from .runner import evaluate_joint_checkpoint

    for dataset in args.datasets:
        search, _ = _search_payload(search_root, dataset, source_method)
        for order_id in sorted(ORDERS_BY_DATASET[dataset]):
            checkpoint = selected_checkpoint(search, order_id)
            for seed in SEEDS:
                print(
                    evaluate_joint_checkpoint(
                        dataset_root=dataset_root,
                        result_root=result_root,
                        dataset_name=dataset,
                        order_id=order_id,
                        seed=seed,
                        device=device,
                        checkpoint_path=checkpoint,
                        exp_name=exp_name,
                        source_method=source_method,
                    )
                )
    return 0
