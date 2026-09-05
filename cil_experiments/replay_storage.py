from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def validate_uint8_one_hot(
    labels: np.ndarray, num_classes: int | None = None
) -> None:
    """Validate one label vector or a batch of labels in persistent format."""
    if labels.dtype != np.uint8 or labels.ndim not in (1, 2):
        raise TypeError("Packed replay labels must be uint8 one-hot vectors or matrices")
    if labels.shape[-1] == 0:
        raise ValueError("Packed replay one-hot labels must have a positive width")
    if num_classes is not None and labels.shape[-1] != int(num_classes):
        raise ValueError("Packed replay one-hot width does not match the dataset class count")
    rows = labels.reshape(-1, labels.shape[-1])
    if not np.logical_or(rows == 0, rows == 1).all() or not np.all(rows.sum(axis=1) == 1):
        raise ValueError("Packed replay label is not one-hot")


@dataclass(frozen=True)
class PackedBinaryExample:
    """Self-contained persistent representation for one binary replay sample."""

    data: np.ndarray
    shape: tuple[int, ...]
    label_one_hot: np.ndarray

    @classmethod
    def from_tensor(
        cls, sample: torch.Tensor, label: int, num_classes: int
    ) -> "PackedBinaryExample":
        array = sample.detach().cpu().numpy()
        if not np.logical_or(array == 0, array == 1).all():
            raise ValueError("Spike replay samples must be exactly binary before 1-bit packing")
        label = int(label)
        if not 0 <= label < int(num_classes):
            raise ValueError(f"Replay label {label} is outside 0..{int(num_classes) - 1}")
        packed = np.packbits(array.reshape(-1).astype(np.uint8), bitorder="little")
        one_hot = np.zeros(int(num_classes), dtype=np.uint8)
        one_hot[label] = 1
        return cls(
            data=np.ascontiguousarray(packed),
            shape=tuple(int(value) for value in array.shape),
            label_one_hot=one_hot,
        )

    @property
    def label(self) -> int:
        validate_uint8_one_hot(self.label_one_hot)
        return int(self.label_one_hot.argmax())

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))

    def unpack(self) -> torch.Tensor:
        unpacked = np.unpackbits(
            self.data, bitorder="little", count=self.numel
        ).reshape(self.shape)
        return torch.from_numpy(unpacked.astype(np.float32, copy=False))

    def validate(self, num_classes: int | None = None) -> None:
        expected = (self.numel + 7) // 8
        if self.data.dtype != np.uint8 or self.data.ndim != 1 or self.data.nbytes != expected:
            raise RuntimeError(
                f"Packed replay sample uses {self.data.nbytes} bytes, expected ceil({self.numel}/8)={expected}"
            )
        validate_uint8_one_hot(self.label_one_hot, num_classes)


@dataclass(frozen=True)
class Float32ReplayExample:
    """Self-contained persistent representation for one floating-point replay sample."""

    data: torch.Tensor
    label_one_hot: np.ndarray

    @classmethod
    def from_tensor(
        cls, sample: torch.Tensor, label: int, num_classes: int
    ) -> "Float32ReplayExample":
        label = int(label)
        if not 0 <= label < int(num_classes):
            raise ValueError(f"Replay label {label} is outside 0..{int(num_classes) - 1}")
        one_hot = np.zeros(int(num_classes), dtype=np.uint8)
        one_hot[label] = 1
        return cls(
            data=sample.detach().cpu().to(torch.float32).contiguous().clone(),
            label_one_hot=one_hot,
        )

    @property
    def label(self) -> int:
        validate_uint8_one_hot(self.label_one_hot)
        return int(self.label_one_hot.argmax())

    @property
    def numel(self) -> int:
        return int(self.data.numel())

    def materialize(self) -> torch.Tensor:
        return self.data

    def validate(self, num_classes: int | None = None) -> None:
        if self.data.device.type != "cpu" or self.data.dtype != torch.float32:
            raise TypeError("Floating replay samples must persist as CPU float32 tensors")
        if not self.data.is_contiguous():
            raise RuntimeError("Floating replay samples must be contiguous")
        validate_uint8_one_hot(self.label_one_hot, num_classes)


ReplayExample = PackedBinaryExample | Float32ReplayExample


class PersistentReplayDataset(Dataset):
    """Decode a persistent replay example only when it is fetched."""

    def __init__(
        self,
        examples: Sequence[ReplayExample],
        on_unpack: Callable[[int], None] | None = None,
    ):
        self.examples = examples
        self.on_unpack = on_unpack

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        if self.on_unpack is not None and isinstance(example, PackedBinaryExample):
            self.on_unpack(example.numel)
        sample = (
            example.unpack()
            if isinstance(example, PackedBinaryExample)
            else example.materialize()
        )
        return sample, example.label, 0


@dataclass
class _ReplayGroup:
    examples: list[ReplayExample]
    weights: torch.Tensor


class ClassBalancedPersistentBuffer:
    """Class-balanced reservoir with materialized samples and uint8 one-hot labels."""

    one_hot_labels = True

    def __init__(self, max_size: int, num_classes: int, *, pack_binary: bool):
        self.max_size = int(max_size)
        self.num_classes = int(num_classes)
        self.packed_binary = bool(pack_binary)
        self.seen_classes: set[int] = set()
        self.buffer_groups: dict[int, _ReplayGroup] = {}
        self.last_persist_calls = 0
        self.last_packed_elements = 0

    @property
    def buffer(self) -> PersistentReplayDataset:
        examples = [
            example
            for class_id in sorted(self.buffer_groups)
            for example in self.buffer_groups[class_id].examples
        ]
        return PersistentReplayDataset(examples)

    def _make_example(self, sample: torch.Tensor, label: int) -> ReplayExample:
        if self.packed_binary:
            return PackedBinaryExample.from_tensor(sample, label, self.num_classes)
        return Float32ReplayExample.from_tensor(sample, label, self.num_classes)

    def _group_lengths(self) -> dict[int, int]:
        classes = sorted(self.seen_classes)
        if not classes:
            return {}
        quotient, remainder = divmod(self.max_size, len(classes))
        return {
            class_id: quotient + (index < remainder)
            for index, class_id in enumerate(classes)
        }

    def post_adapt(self, agent: Any, exp: Any) -> None:
        dataset = exp.dataset
        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise ValueError("Packed ER-ACE replay requires dataset targets")
        indices_by_class: dict[int, list[int]] = {}
        for index, target in enumerate(targets):
            indices_by_class.setdefault(int(target), []).append(index)
        self.seen_classes.update(indices_by_class)
        group_lengths = self._group_lengths()
        self.last_persist_calls = 0
        self.last_packed_elements = 0

        for class_id in sorted(self.seen_classes):
            current_indices = indices_by_class.get(class_id, [])
            previous = self.buffer_groups.get(
                class_id, _ReplayGroup([], torch.empty(0, dtype=torch.float32))
            )
            new_weights = torch.rand(len(current_indices), dtype=torch.float32)
            combined_weights = torch.cat((new_weights, previous.weights.cpu()))
            keep = min(group_lengths[class_id], len(combined_weights))
            selected = torch.argsort(combined_weights, descending=True)[:keep].tolist()
            examples: list[ReplayExample] = []
            retained_weights: list[float] = []
            for selected_index in selected:
                if selected_index < len(current_indices):
                    item = dataset[current_indices[selected_index]]
                    example = self._make_example(item[0], int(item[1]))
                    self.last_persist_calls += 1
                    if self.packed_binary:
                        self.last_packed_elements += example.numel
                else:
                    example = previous.examples[selected_index - len(current_indices)]
                examples.append(example)
                retained_weights.append(float(combined_weights[selected_index]))
            self.buffer_groups[class_id] = _ReplayGroup(
                examples, torch.tensor(retained_weights, dtype=torch.float32)
            )

    def consume_persist_stats(self) -> tuple[int, int]:
        result = self.last_persist_calls, self.last_packed_elements
        self.last_persist_calls = 0
        self.last_packed_elements = 0
        return result

    def persistent_examples(self) -> list[ReplayExample]:
        return [
            example
            for class_id in sorted(self.buffer_groups)
            for example in self.buffer_groups[class_id].examples
        ]

    def auxiliary_values(self) -> list[Any]:
        return [group.weights for group in self.buffer_groups.values()]


class PackedClassBalancedBuffer(ClassBalancedPersistentBuffer):
    def __init__(self, max_size: int, num_classes: int):
        super().__init__(max_size, num_classes, pack_binary=True)


class Float32ClassBalancedBuffer(ClassBalancedPersistentBuffer):
    def __init__(self, max_size: int, num_classes: int):
        super().__init__(max_size, num_classes, pack_binary=False)
