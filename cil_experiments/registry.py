from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .order_seed_registry import ORDERS_BY_DATASET, SEEDS_BY_DATASET


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_x: str
    train_y: str
    test_x: str
    test_y: str
    train_shape: tuple[int, int, int]
    test_shape: tuple[int, int, int]
    source_x_dtype: str
    num_classes: int
    timesteps: int
    in_channels: int
    image_train_x: str
    image_train_y: str
    image_test_x: str
    image_test_y: str
    image_train_shape: tuple[int, int, int, int]
    image_test_shape: tuple[int, int, int, int]
    image_x_dtype: str
    image_layout: str


DATASETS = {
    "spike": DatasetSpec(
        name="spike",
        train_x="spike/X_train_20_cla_400steps.npy",
        train_y="spike/Y_train_20_cla_400steps.npy",
        test_x="spike/X_test_20_cla_400steps.npy",
        test_y="spike/Y_test_20_cla_400steps.npy",
        train_shape=(3219, 400, 64),
        test_shape=(781, 400, 64),
        source_x_dtype="float32",
        num_classes=20,
        timesteps=400,
        in_channels=64,
        image_train_x="spike/X_train_20_cla_400steps_img.npy",
        image_train_y="spike/Y_train_20_cla_400steps_img.npy",
        image_test_x="spike/X_test_20_cla_400steps_img.npy",
        image_test_y="spike/Y_test_20_cla_400steps_img.npy",
        image_train_shape=(3219, 3, 160, 160),
        image_test_shape=(781, 3, 160, 160),
        image_x_dtype="uint8",
        image_layout="NCHW",
    ),
    "texture": DatasetSpec(
        name="texture",
        train_x="texture/X_train_texture.npy",
        train_y="texture/Y_train_texture.npy",
        test_x="texture/X_test_texture.npy",
        test_y="texture/Y_test_texture.npy",
        train_shape=(960, 3775, 9),
        test_shape=(240, 3775, 9),
        source_x_dtype="float64",
        num_classes=12,
        timesteps=3775,
        in_channels=9,
        image_train_x="texture/X_train_texture_img.npy",
        image_train_y="texture/Y_train_texture_img.npy",
        image_test_x="texture/X_test_texture_img.npy",
        image_test_y="texture/Y_test_texture_img.npy",
        image_train_shape=(960, 3, 185, 185),
        image_test_shape=(240, 3, 185, 185),
        image_x_dtype="float32",
        image_layout="NCHW",
    ),
    "uwave": DatasetSpec(
        name="uwave",
        train_x="uwave/X_train_uwave.npy",
        train_y="uwave/Y_train_uwave.npy",
        test_x="uwave/X_test_uwave.npy",
        test_y="uwave/Y_test_uwave.npy",
        train_shape=(3582, 315, 3),
        test_shape=(896, 315, 3),
        source_x_dtype="float32",
        num_classes=8,
        timesteps=315,
        in_channels=3,
        image_train_x="uwave/X_train_uwave_img.npy",
        image_train_y="uwave/Y_train_uwave_img.npy",
        image_test_x="uwave/X_test_uwave_img.npy",
        image_test_y="uwave/Y_test_uwave_img.npy",
        image_train_shape=(3582, 32, 32, 3),
        image_test_shape=(896, 32, 32, 3),
        image_x_dtype="float32",
        image_layout="NHWC",
    ),
}


METHODS: dict[str, dict[str, Any]] = {
    "er_ace": {
        "display_name": "ER-ACE",
        "memory_size": 200,
        "batch_size_mem": 10,
        "source": "Avalanche ER_ACE defaults",
    },
    "ewc": {
        "display_name": "EWC",
        "ewc_lambda": 0.4,
        "mode": "separate",
        "source": "Avalanche examples/tests",
    },
    "cwr_star": {
        "display_name": "CWRStar",
        "cwr_layer_name": "classifier.classifier",
        "source": "Avalanche CWRStar defaults",
    },
    "icarl": {
        "display_name": "ICaRL",
        "memory_size": 2000,
        "fixed_memory": True,
        "source": "iCaRL fixed-memory protocol under the 2000-sample cap",
    },
    "fecam": {
        "display_name": "FeCAM",
        "backbone_training": "first_experience_only_then_frozen",
        "later_experience_update": "class_means_and_covariances_only_no_sgd",
        "tukey": False,
        "shrinkage": True,
        "shrink1": 1.0,
        "shrink2": 1.0,
        "covnorm": True,
        "source": "FeCAM classifier-incremental protocol; Avalanche classifier defaults except tukey disabled for signed shared features",
    },
    "tagfex": {
        "display_name": "TagFex",
        "memory_size": 2000,
        "contrast_factor": 1.0,
        "contrast_kd_factor": 2.0,
        "aux_factor": 2.0,
        "trans_cls_factor": 1.0,
        "transfer_factor": 1.0,
        "infonce_temp": 0.2,
        "infonce_kd_temp": 0.2,
        "kd_temp": 2.0,
        "proj_hidden_dim": 2048,
        "proj_output_dim": 1024,
        "interpolation_factor": 0.95,
        "attention_heads": 8,
        # Optional phase-specific overrides used by source-parity/manual runs.
        # None preserves the shared formal training parameters.
        "init_epochs": None,
        "inc_epochs": None,
        "init_lr": None,
        "inc_lr": None,
        "init_weight_decay": None,
        "inc_weight_decay": None,
        "init_milestones": None,
        "inc_milestones": None,
        "gamma": None,
        "source": "CVPR 2025 TagFex image-network structure, losses, expansion, herding and alignment ported to Avalanche",
    },
    "joint": {
        "display_name": "Joint Learning",
        "source": "Legacy registry name; active PP2 joint learning is the from-scratch ordinary classifier in joint_learning.py",
    },
}


FORMAL_METHODS = (
    "ewc", "icarl", "er_ace", "fecam", "tagfex", "cwr_star"
)

DEFAULT_BACKBONE_ID = "resnet18_cifar"
DEFAULT_BACKBONES = {
    "er_ace": DEFAULT_BACKBONE_ID,
    "ewc": DEFAULT_BACKBONE_ID,
    "cwr_star": DEFAULT_BACKBONE_ID,
    "icarl": DEFAULT_BACKBONE_ID,
    "fecam": DEFAULT_BACKBONE_ID,
    "er": DEFAULT_BACKBONE_ID,
    "tagfex": DEFAULT_BACKBONE_ID,
}

TRAINING_DEFAULTS = {
    "optimizer": "SGD",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "foreach": False,
    "train_mb_size": 32,
    "eval_mb_size": 128,
    "num_workers": 0,
}


def get_task_groups(dataset_name: str, order_id: int) -> tuple[tuple[int, ...], ...]:
    try:
        return ORDERS_BY_DATASET[dataset_name][order_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset/order combination: {dataset_name}/order-{order_id:02d}") from exc


def project_order(dataset_name: str, order_id: int) -> tuple[int, ...]:
    return tuple(class_id for group in get_task_groups(dataset_name, order_id) for class_id in group)


def get_task_split(dataset_name: str, order_id: int) -> tuple[int, ...]:
    return tuple(len(group) for group in get_task_groups(dataset_name, order_id))


def validate_order_registry() -> None:
    if set(ORDERS_BY_DATASET) != set(DATASETS):
        raise ValueError("Order-registry datasets differ from DATASETS")
    for dataset_name, spec in DATASETS.items():
        orders = ORDERS_BY_DATASET[dataset_name]
        for order_id, groups in orders.items():
            if not groups or any(not group for group in groups):
                raise ValueError(f"Empty task group in {dataset_name}/order-{order_id:02d}")
            flat = tuple(class_id for group in groups for class_id in group)
            if any(isinstance(class_id, bool) or not isinstance(class_id, int) for class_id in flat):
                raise TypeError(f"Non-integer class ID in {dataset_name}/order-{order_id:02d}")
            if sorted(flat) != list(range(spec.num_classes)):
                raise ValueError(
                    f"{dataset_name}/order-{order_id:02d} is not a permutation of "
                    f"0..{spec.num_classes - 1}"
                )


validate_order_registry()


def dataset_dict(spec: DatasetSpec) -> dict[str, Any]:
    return asdict(spec)
