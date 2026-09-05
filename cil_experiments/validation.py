from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .output import atomic_write_json
from .registry import DATASETS
from .search_config import VALIDATION_FRACTION, VALIDATION_SPLIT_SEED
from .search_schema import timestamp


def validation_split_path(dataset_root: Path, dataset_name: str) -> Path:
    return Path(dataset_root) / dataset_name / "validation_split.json"


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _source_training_files(dataset_root: Path, dataset_name: str) -> tuple[Path, Path]:
    spec = DATASETS[dataset_name]
    if dataset_name == "spike":
        return dataset_root / spec.train_x, dataset_root / spec.train_y
    return dataset_root / spec.image_train_x, dataset_root / spec.image_train_y


def _split_files(dataset_root: Path, dataset_name: str) -> dict[str, Path]:
    directory = Path(dataset_root) / dataset_name
    return {
        "train_x": directory / "X_train_search.npy",
        "train_y": directory / "Y_train_search.npy",
        "validation_x": directory / "X_validation.npy",
        "validation_y": directory / "Y_validation.npy",
    }


def create_validation_split(
    dataset_root: Path,
    dataset_name: str,
    *,
    seed: int = VALIDATION_SPLIT_SEED,
    validation_fraction: float = VALIDATION_FRACTION,
    overwrite: bool = False,
) -> Path:
    """Materialize one class-stratified split from the complete training view."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    dataset_root = Path(dataset_root)
    spec = DATASETS[dataset_name]
    source_x_path, source_y_path = _source_training_files(dataset_root, dataset_name)
    if not source_x_path.is_file() or not source_y_path.is_file():
        raise FileNotFoundError(f"Missing complete training data: {source_x_path} or {source_y_path}")
    source_x = np.load(source_x_path, mmap_mode="r")
    targets = np.load(source_y_path)
    if len(source_x) != len(targets):
        raise ValueError(f"Training sample/label count mismatch for {dataset_name}")
    if dataset_name != "spike":
        raw_targets = np.load(dataset_root / spec.train_y)
        if not np.array_equal(targets, raw_targets):
            raise ValueError(f"Image and original training label order differ for {dataset_name}")

    path = validation_split_path(dataset_root, dataset_name)
    files = _split_files(dataset_root, dataset_name)
    if path.exists() and not overwrite:
        payload = load_validation_split(dataset_root, dataset_name)
        if int(payload["seed"]) != int(seed) or not np.isclose(
            float(payload["validation_fraction"]), validation_fraction
        ):
            raise ValueError(
                f"Existing validation split uses seed={payload['seed']} and "
                f"fraction={payload['validation_fraction']}; overwrite it to change either"
            )
        missing = [str(file) for file in files.values() if not file.is_file()]
        if missing:
            raise RuntimeError(f"Validation split metadata exists but NPY files are missing: {missing}")
        train_indices = [int(value) for value in payload["train_indices"]]
        validation_indices = [int(value) for value in payload["validation_indices"]]
        if not np.array_equal(np.load(files["train_y"]), targets[train_indices]):
            raise ValueError(f"Saved search labels differ from the original order for {dataset_name}")
        if not np.array_equal(
            np.load(files["validation_y"]), targets[validation_indices]
        ):
            raise ValueError(f"Saved validation labels differ from the original order for {dataset_name}")
        return path

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    class_counts: dict[str, dict[str, int]] = {}
    for class_id in range(spec.num_classes):
        indices = np.flatnonzero(targets == class_id)
        if len(indices) < 2:
            raise ValueError(f"Class {class_id} cannot be split into train and validation")
        shuffled = rng.permutation(indices)
        validation_count = min(
            max(1, int(round(len(indices) * validation_fraction))), len(indices) - 1
        )
        validation_indices.extend(int(value) for value in shuffled[:validation_count])
        train_indices.extend(int(value) for value in shuffled[validation_count:])
        class_counts[str(class_id)] = {
            "train": int(len(indices) - validation_count),
            "validation": int(validation_count),
        }

    train_indices.sort()
    validation_indices.sort()
    _atomic_save(files["train_x"], np.asarray(source_x[train_indices]))
    _atomic_save(files["train_y"], np.asarray(targets[train_indices]))
    _atomic_save(files["validation_x"], np.asarray(source_x[validation_indices]))
    _atomic_save(files["validation_y"], np.asarray(targets[validation_indices]))
    if not np.array_equal(np.load(files["train_y"]), targets[train_indices]):
        raise RuntimeError(f"Saved search labels differ from the original order for {dataset_name}")
    if not np.array_equal(np.load(files["validation_y"]), targets[validation_indices]):
        raise RuntimeError(f"Saved validation labels differ from the original order for {dataset_name}")

    atomic_write_json(
        path,
        {
            "dataset": dataset_name,
            "source_train_x": str(source_x_path.relative_to(dataset_root)),
            "source_train_labels": str(source_y_path.relative_to(dataset_root)),
            "seed": int(seed),
            "validation_fraction": float(validation_fraction),
            "num_classes": spec.num_classes,
            "created_at": timestamp(),
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "class_counts": class_counts,
            "files": {key: str(value.relative_to(dataset_root)) for key, value in files.items()},
        },
    )
    return path


def load_validation_split(dataset_root: Path, dataset_name: str) -> dict[str, Any]:
    path = validation_split_path(dataset_root, dataset_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing fixed validation split: {path}. Run main_exp/prepare_validation_splits.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset_name:
        raise ValueError(f"Validation split dataset mismatch in {path}")
    train = [int(value) for value in payload.get("train_indices", [])]
    validation = [int(value) for value in payload.get("validation_indices", [])]
    _, source_y_path = _source_training_files(Path(dataset_root), dataset_name)
    sample_count = len(np.load(source_y_path, mmap_mode="r"))
    if sorted(train + validation) != list(range(sample_count)):
        raise ValueError(f"Validation split does not partition all training samples: {path}")
    return payload


def build_internal_validation_benchmark(
    dataset_root: Path,
    dataset_name: str,
    validation_seed: int | None = None,
    *,
    validation_fraction: float | None = None,
    order_id: int = 1,
    **_unused,
):
    """Compatibility wrapper for manual tuning; the shared builder owns loading."""
    from .data import build_dataset_bundle

    split = load_validation_split(dataset_root, dataset_name)
    if validation_seed is not None and int(split["seed"]) != int(validation_seed):
        raise ValueError("Saved validation split uses a different seed")
    if validation_fraction is not None and not np.isclose(
        float(split["validation_fraction"]), float(validation_fraction)
    ):
        raise ValueError("Saved validation split uses a different fraction")
    bundle = build_dataset_bundle(
        Path(dataset_root), DATASETS[dataset_name], order_id, data_role="search"
    )
    return bundle.benchmark, split["class_counts"]
