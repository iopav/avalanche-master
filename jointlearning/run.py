from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cil_experiments.experiment_config import EXPERIMENT_ID, experiment_result_root
from cil_experiments.registry import FORMAL_METHODS
from jointlearning.runner import run_joint_reference


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run cumulative Joint references. A completed matching small experiment "
            "is skipped automatically."
        )
    )
    parser.add_argument("--dataset", required=True, choices=("spike", "texture", "uwave"))
    parser.add_argument("--paired-method", required=True, choices=FORMAL_METHODS)
    parser.add_argument("--order-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone",
        help="Optional registered backbone ID; compatibility is checked by the factory.",
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun and replace the matching Joint artifact.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    result_root = experiment_result_root(root, EXPERIMENT_ID).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    result = run_joint_reference(
        root,
        root / "dataset",
        result_root,
        args.dataset,
        args.paired_method,
        args.order_id,
        args.seed,
        torch.device(args.device),
        backbone_id=args.backbone,
        overwrite=args.overwrite,
        experiment_id=EXPERIMENT_ID,
        resume=not args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
