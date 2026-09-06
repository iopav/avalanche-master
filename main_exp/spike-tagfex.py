"""Spike PP2 main-experiment entrypoint for TagFex only."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cil_experiments.pipeline import run_dataset_pipeline


EXP_NAME = "spike-tagfex"
BACKBONE = "resnet18_cifar"
LOSS_SELECTION = "last_epoch_train_mean"

METHOD_GPUS = {
    "tagfex": "2",
}


if __name__ == "__main__":
    raise SystemExit(
        run_dataset_pipeline(
            dataset="spike",
            exp_name=EXP_NAME,
            backbone=BACKBONE,
            loss_selection=LOSS_SELECTION,
            method_gpus=METHOD_GPUS,
        )
    )
