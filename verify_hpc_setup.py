"""Verify the minimal PP2 HPC checkout and required dataset files."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_DATA_FILES = (
    "dataset/spike/X_train_20_cla_400steps.npy",
    "dataset/spike/Y_train_20_cla_400steps.npy",
    "dataset/spike/X_test_20_cla_400steps.npy",
    "dataset/spike/Y_test_20_cla_400steps.npy",
    "dataset/texture/X_train_texture_img.npy",
    "dataset/texture/Y_train_texture_img.npy",
    "dataset/texture/X_test_texture_img.npy",
    "dataset/texture/Y_test_texture_img.npy",
    "dataset/texture/Y_train_texture.npy",
    "dataset/texture/Y_test_texture.npy",
    "dataset/uwave/X_train_uwave_img.npy",
    "dataset/uwave/Y_train_uwave_img.npy",
    "dataset/uwave/X_test_uwave_img.npy",
    "dataset/uwave/Y_test_uwave_img.npy",
    "dataset/uwave/Y_train_uwave.npy",
    "dataset/uwave/Y_test_uwave.npy",
)


def verify_datasets() -> None:
    if len(REQUIRED_DATA_FILES) != 16:
        raise ValueError(
            f"Expected 16 required dataset paths, found {len(REQUIRED_DATA_FILES)}"
        )
    for relative_name in REQUIRED_DATA_FILES:
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe dataset path: {relative_name}")
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing required dataset file: {relative_path}")
        print(f"DATA OK  {relative_path}")


def verify_runtime_imports() -> None:
    from cil_experiments.final_hyperparameters import (
        validate_final_hyperparameter_registry,
    )
    from cil_experiments.joint_learning import validate_joint_result  # noqa: F401
    from cil_experiments.pipeline import run_dataset_pipeline  # noqa: F401
    from cil_experiments.registry import DATASETS, FORMAL_METHODS

    validate_final_hyperparameter_registry()
    if set(DATASETS) != {"spike", "texture", "uwave"}:
        raise RuntimeError(f"Unexpected dataset registry: {sorted(DATASETS)}")
    if len(FORMAL_METHODS) != 6:
        raise RuntimeError(f"Expected six formal methods, found {len(FORMAL_METHODS)}")
    print("CODE OK  local Avalanche and PP2 runtime imports")
    print("CONFIG OK  3 datasets and 6 formal methods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Check dataset files without importing PyTorch or the PP2 runtime.",
    )
    args = parser.parse_args()
    verify_datasets()
    if not args.data_only:
        verify_runtime_imports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
