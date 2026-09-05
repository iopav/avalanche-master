from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .registry import DATASETS, FORMAL_METHODS


def _device(gpu: str):
    if gpu.lower() == "cpu":
        import torch

        return torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU {gpu} was requested but CUDA is unavailable")
    return torch.device("cuda:0")


def _run_method_worker(
    project_root: str,
    dataset_root: str,
    search_root: str,
    joint_root: str,
    exp_name: str,
    dataset: str,
    method: str,
    gpu: str,
    backbone: str,
    loss_selection: str,
    stage: str,
) -> tuple[str, int, int]:
    device = _device(gpu)
    search_count = 0
    joint_count = 0
    if stage in {"all", "search"}:
        from .lr_search import run_method_search

        search_count = len(
            run_method_search(
                project_root=Path(project_root),
                dataset_root=Path(dataset_root),
                search_root=Path(search_root),
                exp_name=exp_name,
                dataset=dataset,
                method=method,
                device=device,
                backbone=backbone,
                loss_selection=loss_selection,
            )
        )
    if stage in {"all", "joint"}:
        from .joint_learning import run_method_joint

        joint_count = len(
            run_method_joint(
                dataset_root=Path(dataset_root),
                search_root=Path(search_root),
                joint_root=Path(joint_root),
                exp_name=exp_name,
                dataset=dataset,
                method=method,
                device=device,
                backbone=backbone,
            )
        )
    return method, search_count, joint_count


def run_dataset_pipeline(
    *,
    dataset: str,
    exp_name: str,
    backbone: str,
    loss_selection: str,
    method_gpus: dict[str, str],
    argv: list[str] | None = None,
) -> int:
    """Run one dataset's six independent search -> joint pipelines."""

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    expected_methods = set(FORMAL_METHODS)
    if set(method_gpus) != expected_methods:
        raise ValueError(
            f"METHOD_GPUS must contain exactly {sorted(expected_methods)}"
        )

    parser = argparse.ArgumentParser(description=f"Run the PP2 {dataset} experiment")
    parser.add_argument(
        "--stage", choices=("all", "search", "joint", "aggregate"), default="all"
    )
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = (args.dataset_root or project_root / "dataset").resolve()
    search_root = (project_root / f"search_result_{exp_name}").resolve()
    joint_root = (project_root / f"joint_result_{exp_name}").resolve()

    if args.stage != "aggregate":
        with ProcessPoolExecutor(max_workers=len(method_gpus)) as executor:
            futures = {
                executor.submit(
                    _run_method_worker,
                    str(project_root),
                    str(dataset_root),
                    str(search_root),
                    str(joint_root),
                    exp_name,
                    dataset,
                    method,
                    method_gpus[method],
                    backbone,
                    loss_selection,
                    args.stage,
                ): method
                for method in FORMAL_METHODS
            }
            for future in as_completed(futures):
                method, search_count, joint_count = future.result()
                print(f"{dataset}/{method}: search={search_count}, joint={joint_count}")

    if args.stage in {"all", "joint", "aggregate"}:
        from .aggregate_results import write_dataset_reports

        for path in write_dataset_reports(
            search_root,
            joint_root,
            dataset=dataset,
            methods=FORMAL_METHODS,
        ):
            print(path)
    return 0


def run_method_pipeline(**_):
    raise RuntimeError(
        "Method-level PP2 entrypoints are retired; use main_exp/spike.py, "
        "texture.py, or uwave.py"
    )


def run_joint_checkpoint_pipeline(**_):
    raise RuntimeError(
        "Checkpoint evaluation is retired; joint learning now trains from scratch"
    )
