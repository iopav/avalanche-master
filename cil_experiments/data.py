from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import DatasetSpec, project_order


class NpyTimeSeriesDataset(Dataset):
    """Read-only NPY-backed time-series dataset with technical dtype/layout conversion."""

    def __init__(self, x_path: Path, y_path: Path, label_map: dict[int, int]):
        self.x_path = x_path
        self.y_path = y_path
        self._x = np.load(x_path, mmap_mode="r")
        raw_targets = np.load(y_path)
        self.targets = [label_map[int(v)] for v in raw_targets.tolist()]
        self.raw_targets = raw_targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        # A real copy avoids writable-view ambiguity from read-only memmaps.
        item = np.array(self._x[index], dtype=np.float32, copy=True)
        x = torch.from_numpy(item).transpose(0, 1).contiguous()  # [T,C] -> [C,T]
        return x, int(self.targets[index])


@dataclass
class DatasetBundle:
    spec: DatasetSpec
    train: NpyTimeSeriesDataset
    test: NpyTimeSeriesDataset
    raw_order: tuple[int, ...]
    label_map: dict[int, int]
    inverse_label_map: dict[int, int]
    benchmark: Any
    test_samples_per_task: list[int]


def _validate_binary_spike(path: Path, chunk_samples: int = 64) -> None:
    array = np.load(path, mmap_mode="r")
    for start in range(0, len(array), chunk_samples):
        chunk = np.asarray(array[start : start + chunk_samples])
        if not np.logical_or(chunk == 0, chunk == 1).all():
            values = np.unique(chunk)
            raise ValueError(f"Spike data is not binary 0/1 in {path}: sample values={values[:10]}")


def validate_source_files(dataset_root: Path, spec: DatasetSpec) -> None:
    train_x = np.load(dataset_root / spec.train_x, mmap_mode="r")
    test_x = np.load(dataset_root / spec.test_x, mmap_mode="r")
    train_y = np.load(dataset_root / spec.train_y, mmap_mode="r")
    test_y = np.load(dataset_root / spec.test_y, mmap_mode="r")
    if tuple(train_x.shape) != spec.train_shape or tuple(test_x.shape) != spec.test_shape:
        raise ValueError(
            f"{spec.name} shape mismatch: train={train_x.shape}, test={test_x.shape}; "
            f"expected {spec.train_shape}, {spec.test_shape}"
        )
    if str(train_x.dtype) != spec.source_x_dtype or str(test_x.dtype) != spec.source_x_dtype:
        raise ValueError(f"{spec.name} source dtype mismatch: {train_x.dtype}, {test_x.dtype}")
    if train_x.shape[1] != spec.timesteps or test_x.shape[1] != spec.timesteps:
        raise ValueError(f"{spec.name} time-step mismatch; expected T={spec.timesteps}")
    if train_x.shape[2] != spec.in_channels or test_x.shape[2] != spec.in_channels:
        raise ValueError(f"{spec.name} channel mismatch; expected C={spec.in_channels}")
    classes = sorted(set(int(v) for v in np.concatenate((train_y, test_y))))
    if classes != list(range(spec.num_classes)):
        raise ValueError(f"{spec.name} labels are {classes}, expected 0..{spec.num_classes - 1}")
    if spec.name == "spike":
        if "400steps" not in spec.train_x or "400steps" not in spec.test_x:
            raise ValueError("Spike must use the explicit 400-step source files")
        _validate_binary_spike(dataset_root / spec.train_x)
        _validate_binary_spike(dataset_root / spec.test_x)


def build_dataset_bundle(dataset_root: Path, spec: DatasetSpec, order_id: int) -> DatasetBundle:
    from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark

    validate_source_files(dataset_root, spec)
    raw_order = project_order(order_id, spec.num_classes)
    label_map = {raw: internal for internal, raw in enumerate(raw_order)}
    inverse = {internal: raw for raw, internal in label_map.items()}
    train = NpyTimeSeriesDataset(dataset_root / spec.train_x, dataset_root / spec.train_y, label_map)
    test = NpyTimeSeriesDataset(dataset_root / spec.test_x, dataset_root / spec.test_y, label_map)
    benchmark = nc_benchmark(
        train_dataset=train,
        test_dataset=test,
        n_experiences=spec.tasks,
        task_labels=False,
        shuffle=False,
        fixed_class_order=list(range(spec.num_classes)),
        per_exp_classes={0: 3},
        class_ids_from_zero_from_first_exp=False,
        train_transform=None,
        eval_transform=None,
    )
    expected_per_exp = [3] + [1] * (spec.tasks - 1)
    if list(benchmark.n_classes_per_exp) != expected_per_exp:
        raise AssertionError(f"Unexpected task split: {benchmark.n_classes_per_exp}")
    test_counts = [len(exp.dataset) for exp in benchmark.test_stream]
    return DatasetBundle(spec, train, test, raw_order, label_map, inverse, benchmark, test_counts)
