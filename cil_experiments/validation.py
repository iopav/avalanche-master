from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .data import NpyTimeSeriesDataset, validate_source_files
from .registry import DATASETS, get_task_groups, get_task_split, project_order


class IndexedDataset(Dataset):
    """A deterministic index view retaining the targets expected by Avalanche."""

    def __init__(self, source: Dataset, indices: list[int]):
        self.source = source
        self.indices = tuple(int(index) for index in indices)
        self.targets = [int(source.targets[index]) for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.source[self.indices[index]]


def build_internal_validation_benchmark(
    dataset_root: Path,
    dataset_name: str,
    validation_seed: int,
    *,
    validation_fraction: float = 0.2,
    order_id: int = 1,
):
    """Split training data per class and expose the held-out part as a validation stream."""
    from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    spec = DATASETS[dataset_name]
    validate_source_files(dataset_root, spec)
    task_groups = get_task_groups(dataset_name, order_id)
    class_split = get_task_split(dataset_name, order_id)
    raw_order = project_order(dataset_name, order_id)
    label_map = {raw: internal for internal, raw in enumerate(raw_order)}
    source = NpyTimeSeriesDataset(dataset_root / spec.train_x, dataset_root / spec.train_y, label_map)
    targets = np.asarray(source.targets, dtype=np.int64)
    rng = np.random.default_rng(validation_seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    split_counts: dict[str, dict[str, int]] = {}
    for class_id in range(spec.num_classes):
        indices = np.flatnonzero(targets == class_id)
        if len(indices) < 2:
            raise ValueError(f"Class {class_id} cannot be split into train and validation")
        indices = rng.permutation(indices)
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        validation_count = min(validation_count, len(indices) - 1)
        validation_indices.extend(int(value) for value in indices[:validation_count])
        train_indices.extend(int(value) for value in indices[validation_count:])
        split_counts[str(class_id)] = {
            "train": int(len(indices) - validation_count),
            "validation": int(validation_count),
        }

    train = IndexedDataset(source, sorted(train_indices))
    validation = IndexedDataset(source, sorted(validation_indices))
    benchmark = nc_benchmark(
        train_dataset=train,
        test_dataset=validation,
        n_experiences=len(task_groups),
        task_labels=False,
        shuffle=False,
        fixed_class_order=list(range(spec.num_classes)),
        per_exp_classes={index: size for index, size in enumerate(class_split)},
        class_ids_from_zero_from_first_exp=False,
        train_transform=None,
        eval_transform=None,
    )
    expected_per_experience = list(class_split)
    if list(benchmark.n_classes_per_exp) != expected_per_experience:
        raise AssertionError(f"Unexpected validation task split: {benchmark.n_classes_per_exp}")
    return benchmark, split_counts
