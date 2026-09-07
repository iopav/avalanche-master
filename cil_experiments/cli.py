from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

from .models import BACKBONES
from .order_seed_registry import get_formal_seeds
from .registry import DEFAULT_BACKBONE_ID, FORMAL_METHODS, ORDERS_BY_DATASET
from .runner import run_experiment


def run_dataset_cli(dataset_name: str) -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"Run auditable {dataset_name} CIL experiments. Completed small "
            "experiments in the configured result-exp folder are skipped automatically."
        )
    )
    parser.add_argument(
        "--methods", nargs="+", default=list(FORMAL_METHODS), choices=list(FORMAL_METHODS)
    )
    parser.add_argument(
        "--backbone",
        choices=tuple(BACKBONES),
        help=(
            "Optional registered backbone ID for all methods, including TagFex. "
            "Compatibility is validated by the backbone factory."
        ),
    )
    parser.add_argument(
        "--order-ids", nargs="+", type=int,
        default=list(sorted(ORDERS_BY_DATASET[dataset_name])),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(get_formal_seeds(dataset_name)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--exp-name", default=f"legacy_{dataset_name}")
    parser.add_argument(
        "--skip-intransigence",
        action="store_true",
        help=(
            "Run without a matching joint-learning artifact. The summary keeps "
            "cil_performance.intransigence=null and is not eligible for final aggregation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun and replace an existing artifact for the same experiment/order/seed/method.",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    dataset_root = (args.dataset_root or project_root / "dataset").resolve()
    output_root = (
        args.output_root or project_root / f"result_{args.exp_name}"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    try:
        for order_id in args.order_ids:
            for seed in args.seeds:
                for method in args.methods:
                    path = run_experiment(
                        project_root,
                        dataset_root,
                        output_root,
                        dataset_name,
                        method,
                        order_id,
                        seed,
                        None,
                        device,
                        overwrite=args.overwrite,
                        backbone_id=args.backbone or DEFAULT_BACKBONE_ID,
                        compute_intransigence_enabled=not args.skip_intransigence,
                        exp_name=args.exp_name,
                        joint_result_root=project_root / "joint_result" / args.exp_name,
                        resume=not args.overwrite,
                    )
                    print(path)
        return 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1
