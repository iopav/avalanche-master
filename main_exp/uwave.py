"""UWave 数据集 PP2 主实验入口。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cil_experiments.pipeline import run_dataset_pipeline


# 每次主实验只需在这里修改实验名、骨干、loss 选择方式和 GPU 分配。
EXP_NAME = "uwave_pp2"
BACKBONE = "resnet18_cifar"
LOSS_SELECTION = "last_epoch_train_mean"

METHOD_GPUS = {
    "ewc": "0",
    "er_ace": "0",
    "icarl": "1",
    "fecam": "1",
    "tagfex": "2",
    "cwr_star": "2",
}


if __name__ == "__main__":
    raise SystemExit(
        run_dataset_pipeline(
            dataset="uwave",
            exp_name=EXP_NAME,
            backbone=BACKBONE,
            loss_selection=LOSS_SELECTION,
            method_gpus=METHOD_GPUS,
        )
    )
