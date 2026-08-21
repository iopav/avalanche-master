from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

from .registry import METHODS
from .runner import run_one


def run_dataset_cli(dataset_name: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run auditable {dataset_name} CIL experiments")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--order-ids", nargs="+", type=int, default=[1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[62])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
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
                        args.epochs,
                        device,
                        overwrite=args.overwrite,
                    )
                    print(path)
        return 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1
