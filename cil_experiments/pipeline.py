from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .registry import DATASETS, FORMAL_METHODS


def _parse_method_gpus(
    value: str,
    expected_methods: tuple[str, ...] = FORMAL_METHODS,
) -> dict[str, str]:
    """Parse ``method=gpu`` pairs supplied by an HPC launch script."""
    result: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item or "=" not in item:
            raise ValueError(
                "--method-gpus must be comma-separated method=gpu pairs"
            )
        method, gpu = (part.strip() for part in item.split("=", 1))
        if not method or not gpu:
            raise ValueError(f"Invalid --method-gpus entry: {item!r}")
        if method in result:
            raise ValueError(f"Duplicate method in --method-gpus: {method}")
        result[method] = gpu
    expected = set(expected_methods)
    if set(result) != expected:
        raise ValueError(
            "--method-gpus must contain exactly "
            f"{sorted(expected)}; got {sorted(result)}"
        )
    return result


def _device(gpu: str):
    if gpu.lower() == "cpu":
        import torch

        return torch.device("cpu")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU {gpu} was requested but CUDA is unavailable")
    try:
        gpu_index = int(gpu)
    except ValueError as exc:
        raise ValueError(f"GPU mapping must use a job-local integer index: {gpu!r}") from exc
    visible_count = torch.cuda.device_count()
    if not 0 <= gpu_index < visible_count:
        raise ValueError(
            f"GPU index {gpu_index} is outside the {visible_count} devices visible "
            "inside the current Slurm allocation"
        )
    torch.cuda.set_device(gpu_index)
    return torch.device("cuda", gpu_index)


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
    """Run one dataset's configured independent search -> joint pipelines."""

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    unknown_methods = set(method_gpus) - set(FORMAL_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown methods in METHOD_GPUS: {sorted(unknown_methods)}")
    methods = tuple(method for method in FORMAL_METHODS if method in method_gpus)
    if not methods:
        raise ValueError("METHOD_GPUS must contain at least one formal method")
    parser = argparse.ArgumentParser(description=f"Run the PP2 {dataset} experiment")
    parser.add_argument(
        "--stage", choices=("all", "search", "joint", "aggregate"), default="all"
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--exp-name", default=exp_name)
    parser.add_argument("--backbone", default=backbone)
    parser.add_argument(
        "--loss-selection",
        choices=("last_epoch_train_mean", "full_train_final_model"),
        default=loss_selection,
    )
    parser.add_argument(
        "--method-gpus",
        help=(
            "Explicit comma-separated method=gpu mapping for exactly the methods "
            f"configured by this entrypoint: {','.join(methods)}. "
            "Overrides the mapping in the dataset entrypoint."
        ),
    )
    args = parser.parse_args(argv)
    exp_name = args.exp_name
    backbone = args.backbone
    loss_selection = args.loss_selection
    method_gpus = (
        _parse_method_gpus(args.method_gpus, methods)
        if args.method_gpus is not None
        else dict(method_gpus)
    )
    expected_methods = set(methods)
    if set(method_gpus) != expected_methods:
        raise ValueError(
            f"METHOD_GPUS must contain exactly {sorted(expected_methods)}"
        )

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
                for method in methods
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
            methods=methods,
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
