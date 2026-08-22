from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SEEDS = (62, 63, 64, 65, 66, 67, 68, 69, 70, 71)

ORDERS = {
    1: (16, 19, 0, 9, 8, 17, 1, 12, 11, 5, 10, 4, 18, 13, 15, 7, 14, 6, 3, 2),
    2: (7, 18, 12, 1, 11, 9, 13, 8, 2, 17, 10, 6, 4, 5, 15, 3, 19, 0, 14, 16),
    3: (14, 1, 19, 2, 16, 3, 5, 7, 6, 13, 12, 0, 8, 11, 10, 18, 4, 9, 17, 15),
    4: (6, 13, 15, 17, 11, 1, 2, 19, 5, 7, 4, 10, 3, 8, 0, 16, 9, 18, 14, 12),
    5: (14, 3, 19, 6, 16, 4, 10, 1, 5, 11, 9, 13, 8, 7, 12, 18, 15, 2, 17, 0),
}


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

    @property
    def tasks(self) -> int:
        return self.num_classes - 2


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
}


BACKBONE_CONFIG = {
    "hidden_channels": [64, 128, 128],
    "kernels": [7, 5, 3],
    "strides": [2, 2, 2],
    "paddings": [3, 2, 1],
    "group_norm_groups": 8,
    "feature_dim": 64,
}

TRAINING_DEFAULTS = {
    "optimizer": "SGD",
    "learning_rate": 0.1,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "foreach": False,
    "train_mb_size": 32,
    "eval_mb_size": 128,
    "num_workers": 0,
}


def project_order(order_id: int, num_classes: int) -> tuple[int, ...]:
    if order_id not in ORDERS:
        raise ValueError(f"Unknown order_id={order_id}; expected one of {sorted(ORDERS)}")
    projected = tuple(c for c in ORDERS[order_id] if c < num_classes)
    if sorted(projected) != list(range(num_classes)):
        raise AssertionError(f"Projected order is not a permutation of 0..{num_classes - 1}")
    return projected


def parse_order_seed_file(path: Path) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
    text = path.read_text(encoding="utf-8")
    seed_match = re.search(r"Held-out seeds:\s*\n\s*\[([^]]+)\]", text)
    if seed_match is None:
        raise ValueError(f"Cannot parse held-out seeds from {path}")
    seeds = tuple(int(v.strip()) for v in seed_match.group(1).split(","))
    parsed: dict[int, tuple[int, ...]] = {}
    pattern = re.compile(r"order ID\s+(\d+)\s*\n+(?:\s*\n)?Full order:\s*\n\s*\[([^]]+)\]", re.I)
    for match in pattern.finditer(text):
        parsed[int(match.group(1))] = tuple(int(v.strip()) for v in match.group(2).split(","))
    if not parsed:
        raise ValueError(f"Cannot parse class orders from {path}")
    return seeds, parsed


def validate_order_seed_file(path: Path) -> None:
    seeds, orders = parse_order_seed_file(path)
    if seeds != SEEDS:
        raise AssertionError(f"Seed registry differs from {path}: {seeds} != {SEEDS}")
    if orders != ORDERS:
        raise AssertionError(f"Order registry differs from {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_dict(spec: DatasetSpec) -> dict[str, Any]:
    return asdict(spec)
