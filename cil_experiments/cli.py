from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

from .registry import FORMAL_METHODS
from .runner import run_one


def run_dataset_cli(dataset_name: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run auditable {dataset_name} CIL experiments")
    parser.add_argument(
        "--methods", nargs="+", default=list(FORMAL_METHODS), choices=list(FORMAL_METHODS)
    )
    parser.add_argument(
        "--backbone",
        help=(
            "Optional registered backbone ID for all methods, including TagFex. "
            "Compatibility is validated by the backbone factory."
        ),
    )
    parser.add_argument("--order-ids", nargs="+", type=int, default=[1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[62])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--skip-intransigence",
        action="store_true",
        help=(
            "Run without a matching Joint artifact. The summary keeps "
            "cil_performance.intransigence=null and is not eligible for final aggregation."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    dataset_root = (args.dataset_root or project_root / "dataset").resolve()
    output_root = (args.output_root or project_root / "result").resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    try:
        for order_id in args.order_ids:
            for seed in args.seeds:
                for method in args.methods:
                    path = run_one(
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
                        backbone_id=args.backbone,
                        compute_intransigence_enabled=not args.skip_intransigence,
                    )
                    print(path)
        return 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1
