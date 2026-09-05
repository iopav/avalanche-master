"""可直接运行的 PP2 tiny smoke：6 方法、2 seed、2 LR、2 order。"""

from __future__ import annotations

import argparse
import copy
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cil_experiments.aggregate_results import write_dataset_reports
from cil_experiments.final_hyperparameters import FINAL_HYPERPARAMETERS
from cil_experiments.joint_learning import run_method_joint
from cil_experiments.lr_search import run_method_search
from cil_experiments.order_seed_registry import ORDERS_BY_DATASET
from cil_experiments.registry import DATASETS


# 修改 EXP_NAME 后可开始一轮全新的 tiny run；相同名称会使用三级断训恢复。
EXP_NAME = "tiny_pp2_six_methods"
DEVICE = "cpu"  # 也可改为 "cuda:0"。
DATASET = "uwave"
BACKBONE = "temporal"
LOSS_SELECTION = "last_epoch_train_mean"
SEEDS = (62, 63)
LR_CANDIDATES = (0.1, 0.01)
ORDERS = {
    1: ((0, 1), (2, 3)),
    2: ((2, 3), (0, 1)),
}
METHODS = ("ewc", "er_ace", "icarl", "fecam", "tagfex", "cwr_star")


def prepare_tiny_dataset(dataset_root: Path):
    """生成四分类、小尺寸、类别均衡的 UWave image-view 假数据。"""

    folder = dataset_root / DATASET
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260905)
    train_y = np.repeat(np.arange(4, dtype=np.int64), 8)
    test_y = np.repeat(np.arange(4, dtype=np.int64), 4)
    train_x = rng.normal(size=(len(train_y), 3, 16, 16)).astype(np.float32)
    test_x = rng.normal(size=(len(test_y), 3, 16, 16)).astype(np.float32)
    for label in range(4):
        train_x[train_y == label, label % 3] += 0.75
        test_x[test_y == label, label % 3] += 0.75
    arrays = {
        "X_train_raw.npy": train_x,
        "Y_train_raw.npy": train_y,
        "X_test_raw.npy": test_x,
        "Y_test_raw.npy": test_y,
        "X_train_img.npy": train_x,
        "Y_train_img.npy": train_y,
        "X_test_img.npy": test_x,
        "Y_test_img.npy": test_y,
    }
    for name, value in arrays.items():
        np.save(folder / name, value)
    return replace(
        DATASETS[DATASET],
        train_x=f"{DATASET}/X_train_raw.npy",
        train_y=f"{DATASET}/Y_train_raw.npy",
        test_x=f"{DATASET}/X_test_raw.npy",
        test_y=f"{DATASET}/Y_test_raw.npy",
        train_shape=tuple(train_x.shape),
        test_shape=tuple(test_x.shape),
        source_x_dtype="float32",
        image_train_x=f"{DATASET}/X_train_img.npy",
        image_train_y=f"{DATASET}/Y_train_img.npy",
        image_test_x=f"{DATASET}/X_test_img.npy",
        image_test_y=f"{DATASET}/Y_test_img.npy",
        image_train_shape=tuple(train_x.shape),
        image_test_shape=tuple(test_x.shape),
        image_x_dtype="float32",
        image_layout="NCHW",
        num_classes=4,
    )


def tiny_hyperparameters(method: str) -> dict:
    parameters = copy.deepcopy(FINAL_HYPERPARAMETERS[DATASET][method])
    parameters.update(
        epochs_per_experience=1,
        train_mb_size=4,
        eval_mb_size=4,
        num_workers=0,
        foreach=False,
    )
    if method == "er_ace":
        parameters.update(memory_size=16, batch_size_mem=4)
    elif method == "icarl":
        parameters.update(memory_size=16)
    elif method == "tagfex":
        parameters.update(
            memory_size=16,
            init_epochs=1,
            inc_epochs=1,
            proj_hidden_dim=64,
            proj_output_dim=32,
        )
    return parameters


def run(stage: str) -> tuple[Path, Path]:
    output_root = Path(__file__).resolve().parent / "tiny_run"
    dataset_root = output_root / "dataset"
    search_root = output_root / f"search_result_{EXP_NAME}"
    joint_root = output_root / f"joint_result_{EXP_NAME}"
    spec = prepare_tiny_dataset(dataset_root)
    device = torch.device(DEVICE)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(DATASETS, {DATASET: spec}, clear=True))
        stack.enter_context(
            patch.dict(ORDERS_BY_DATASET, {DATASET: ORDERS}, clear=True)
        )
        stack.enter_context(
            patch("cil_experiments.lr_search.LR_CANDIDATES", LR_CANDIDATES)
        )
        for module in (
            "cil_experiments.lr_search",
            "cil_experiments.joint_learning",
            "cil_experiments.aggregate_results",
            "cil_experiments.runner",
        ):
            stack.enter_context(patch(f"{module}.SEEDS", SEEDS))
        for method in METHODS:
            stack.enter_context(
                patch.dict(
                    FINAL_HYPERPARAMETERS[DATASET][method],
                    tiny_hyperparameters(method),
                    clear=True,
                )
            )

        for method in METHODS:
            if stage in {"all", "search"}:
                run_method_search(
                    project_root=PROJECT_ROOT,
                    dataset_root=dataset_root,
                    search_root=search_root,
                    exp_name=EXP_NAME,
                    dataset=DATASET,
                    method=method,
                    device=device,
                    backbone=BACKBONE,
                    loss_selection=LOSS_SELECTION,
                )
            if stage in {"all", "joint"}:
                run_method_joint(
                    dataset_root=dataset_root,
                    search_root=search_root,
                    joint_root=joint_root,
                    exp_name=EXP_NAME,
                    dataset=DATASET,
                    method=method,
                    device=device,
                    backbone=BACKBONE,
                )
            if stage != "aggregate":
                print(f"completed: {method} ({stage})")

        if stage in {"all", "joint", "aggregate"}:
            paths = write_dataset_reports(
                search_root,
                joint_root,
                dataset=DATASET,
                methods=METHODS,
            )
            for path in paths:
                print(path)
    return search_root, joint_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the six-method PP2 tiny search and joint-learning smoke"
    )
    parser.add_argument(
        "--stage", choices=("all", "search", "joint", "aggregate"), default="all"
    )
    args = parser.parse_args()
    run(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
