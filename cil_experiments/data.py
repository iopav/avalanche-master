from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import DatasetSpec, get_task_groups, get_task_split, project_order


SEARCH_FILES = {
    "train": ("X_train_search.npy", "Y_train_search.npy"),
    "validation": ("X_validation.npy", "Y_validation.npy"),
}


class NpyTimeSeriesDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, label_map: dict[int, int]):
        self.x_path = x_path
        self.y_path = y_path
        self._x = np.load(x_path, mmap_mode="r")
        raw_targets = np.load(y_path)
        self.targets = [label_map[int(value)] for value in raw_targets.tolist()]
        self.raw_targets = raw_targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        item = np.array(self._x[index], dtype=np.float32, copy=True)
        return torch.from_numpy(item).transpose(0, 1).contiguous(), self.targets[index]


class NpyImageDataset(Dataset):
    def __init__(
        self, x_path: Path, y_path: Path, label_map: dict[int, int], layout: str
    ):
        if layout not in {"NCHW", "NHWC"}:
            raise ValueError(f"Unsupported stored image layout: {layout}")
        self.x_path = x_path
        self.y_path = y_path
        self._x = np.load(x_path, mmap_mode="r")
        raw_targets = np.load(y_path)
        self.targets = [label_map[int(value)] for value in raw_targets.tolist()]
        self.raw_targets = raw_targets
        self.layout = layout

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        item = np.array(self._x[index], dtype=np.float32, copy=True)
        if self.layout == "NHWC":
            item = np.transpose(item, (2, 0, 1)).copy()
        return torch.from_numpy(item).contiguous(), self.targets[index]


@dataclass
class DatasetBundle:
    spec: DatasetSpec
    data_role: str
    train: Dataset
    test: Dataset
    raw_order: tuple[int, ...]
    label_map: dict[int, int]
    inverse_label_map: dict[int, int]
    benchmark: Any
    test_samples_per_task: list[int]
    task_groups: tuple[tuple[int, ...], ...]
    input_view_id: str
    model_input_shape: tuple[int, ...]

    @property
    def tasks(self) -> int:
        return len(self.task_groups)


def _validate_binary_spike(path: Path, chunk_samples: int = 64) -> None:
    array = np.load(path, mmap_mode="r")
    for start in range(0, len(array), chunk_samples):
        chunk = np.asarray(array[start : start + chunk_samples])
        if not np.logical_or(chunk == 0, chunk == 1).all():
            raise ValueError(f"Spike data is not binary in {path}")


def _load_pair(
    x_path: Path,
    y_path: Path,
    *,
    sample_shape: tuple[int, ...],
    x_dtype: str,
    num_classes: int,
) -> None:
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"Missing dataset files: {x_path} or {y_path}")
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    if tuple(x.shape[1:]) != tuple(sample_shape):
        raise ValueError(f"Unexpected shape for {x_path}: {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"Sample/label count mismatch: {x_path}, {y_path}")
    if str(x.dtype) != x_dtype:
        raise ValueError(f"Unexpected dtype for {x_path}: {x.dtype}, expected {x_dtype}")
    labels = sorted(set(int(value) for value in np.asarray(y).tolist()))
    if labels != list(range(num_classes)):
        raise ValueError(
            f"Labels in {y_path} are {labels}, expected 0..{num_classes - 1}"
        )


def validate_source_files(
    dataset_root: Path, spec: DatasetSpec, *, allow_variable_samples: bool = False
) -> None:
    """Validate the complete raw train/test pair used as the source dataset."""
    for x_name, y_name, expected in (
        (spec.train_x, spec.train_y, spec.train_shape),
        (spec.test_x, spec.test_y, spec.test_shape),
    ):
        x_path, y_path = Path(dataset_root) / x_name, Path(dataset_root) / y_name
        _load_pair(
            x_path,
            y_path,
            sample_shape=tuple(expected[1:]),
            x_dtype=spec.source_x_dtype,
            num_classes=spec.num_classes,
        )
        if not allow_variable_samples and len(np.load(x_path, mmap_mode="r")) != expected[0]:
            raise ValueError(f"Unexpected sample count for {x_path}")
    if spec.name == "spike":
        _validate_binary_spike(Path(dataset_root) / spec.train_x)
        _validate_binary_spike(Path(dataset_root) / spec.test_x)


def validate_image_files(
    dataset_root: Path, spec: DatasetSpec, *, allow_variable_samples: bool = False
) -> None:
    for x_name, y_name, expected in (
        (spec.image_train_x, spec.image_train_y, spec.image_train_shape),
        (spec.image_test_x, spec.image_test_y, spec.image_test_shape),
    ):
        x_path, y_path = Path(dataset_root) / x_name, Path(dataset_root) / y_name
        _load_pair(
            x_path,
            y_path,
            sample_shape=tuple(expected[1:]),
            x_dtype=spec.image_x_dtype,
            num_classes=spec.num_classes,
        )
        if not allow_variable_samples and len(np.load(x_path, mmap_mode="r")) != expected[0]:
            raise ValueError(f"Unexpected sample count for {x_path}")


def _paths_for_role(
    dataset_root: Path, spec: DatasetSpec, data_role: str
) -> tuple[Path, Path, Path, Path, bool]:
    if data_role == "search":
        directory = dataset_root / spec.name
        train_x, train_y = SEARCH_FILES["train"]
        eval_x, eval_y = SEARCH_FILES["validation"]
        return directory / train_x, directory / train_y, directory / eval_x, directory / eval_y, spec.name != "spike"
    if data_role not in {"formal", "smoke"}:
        raise ValueError(
            f"Unknown data_role {data_role!r}; expected 'search', 'formal', or 'smoke'"
        )
    if spec.name == "spike":
        return (
            dataset_root / spec.train_x,
            dataset_root / spec.train_y,
            dataset_root / spec.test_x,
            dataset_root / spec.test_y,
            False,
        )
    return (
        dataset_root / spec.image_train_x,
        dataset_root / spec.image_train_y,
        dataset_root / spec.image_test_x,
        dataset_root / spec.image_test_y,
        True,
    )


def build_dataset_bundle(
    dataset_root: Path,
    spec: DatasetSpec,
    order_id: int,
    *,
    data_role: str,
) -> DatasetBundle:
    from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark

    task_groups = get_task_groups(spec.name, order_id)
    class_split = get_task_split(spec.name, order_id)
    raw_order = project_order(spec.name, order_id)
    label_map = {raw: internal for internal, raw in enumerate(raw_order)}
    inverse = {internal: raw for raw, internal in label_map.items()}
    train_x, train_y, eval_x, eval_y, stored_image = _paths_for_role(
        Path(dataset_root), spec, data_role
    )
    if stored_image:
        sample_shape = tuple(spec.image_train_shape[1:])
        _load_pair(
            train_x, train_y, sample_shape=sample_shape,
            x_dtype=spec.image_x_dtype, num_classes=spec.num_classes,
        )
        _load_pair(
            eval_x, eval_y, sample_shape=sample_shape,
            x_dtype=spec.image_x_dtype, num_classes=spec.num_classes,
        )
        if data_role in {"formal", "smoke"}:
            for image_labels, raw_labels in (
                (train_y, Path(dataset_root) / spec.train_y),
                (eval_y, Path(dataset_root) / spec.test_y),
            ):
                if not np.array_equal(np.load(image_labels), np.load(raw_labels)):
                    raise ValueError(
                        f"Image and original label order differ: {image_labels}, {raw_labels}"
                    )
        train = NpyImageDataset(train_x, train_y, label_map, spec.image_layout)
        test = NpyImageDataset(eval_x, eval_y, label_map, spec.image_layout)
        model_input_shape = (
            sample_shape if spec.image_layout == "NCHW"
            else (sample_shape[2], sample_shape[0], sample_shape[1])
        )
        input_view_id = f"{spec.name}_stored_rgb_image_v1"
    else:
        sample_shape = tuple(spec.train_shape[1:])
        _load_pair(
            train_x, train_y, sample_shape=sample_shape,
            x_dtype=spec.source_x_dtype, num_classes=spec.num_classes,
        )
        _load_pair(
            eval_x, eval_y, sample_shape=sample_shape,
            x_dtype=spec.source_x_dtype, num_classes=spec.num_classes,
        )
        if spec.name == "spike":
            _validate_binary_spike(train_x)
            _validate_binary_spike(eval_x)
        train = NpyTimeSeriesDataset(train_x, train_y, label_map)
        test = NpyTimeSeriesDataset(eval_x, eval_y, label_map)
        model_input_shape = (spec.in_channels, spec.timesteps)
        input_view_id = "spike_raw_binary_to_rgb160_runtime_v1"

    benchmark = nc_benchmark(
        train_dataset=train,
        test_dataset=test,
        n_experiences=len(task_groups),
        task_labels=False,
        shuffle=False,
        fixed_class_order=list(range(spec.num_classes)),
        per_exp_classes={index: size for index, size in enumerate(class_split)},
        class_ids_from_zero_from_first_exp=False,
        train_transform=None,
        eval_transform=None,
    )
    if list(benchmark.n_classes_per_exp) != list(class_split):
        raise RuntimeError(f"Unexpected task split: {benchmark.n_classes_per_exp}")
    return DatasetBundle(
        spec=spec,
        data_role=data_role,
        train=train,
        test=test,
        raw_order=raw_order,
        label_map=label_map,
        inverse_label_map=inverse,
        benchmark=benchmark,
        test_samples_per_task=[len(exp.dataset) for exp in benchmark.test_stream],
        task_groups=task_groups,
        input_view_id=input_view_id,
        model_input_shape=model_input_shape,
    )
