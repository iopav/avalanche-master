from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .replay_storage import (
    Float32ReplayExample,
    PackedBinaryExample,
    validate_uint8_one_hot,
)
from .strategies import StrategyBundle


class ByteCounter:
    def __init__(self):
        self.seen: set[tuple[Any, ...]] = set()

    def tensor(self, tensor: torch.Tensor | None) -> int:
        if tensor is None:
            return 0
        try:
            storage = tensor.untyped_storage()
            key = ("torch", str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
            size = int(storage.nbytes())
        except (RuntimeError, AttributeError):
            key = ("torch-object", id(tensor))
            size = int(tensor.numel() * tensor.element_size())
        if key in self.seen:
            return 0
        self.seen.add(key)
        return size

    def numpy(self, array: np.ndarray) -> int:
        root = array
        while isinstance(root.base, np.ndarray):
            root = root.base
        key = ("numpy", id(root))
        if key in self.seen:
            return 0
        self.seen.add(key)
        return int(root.nbytes)

    def value(self, value: Any, include_scalars: bool = False) -> int:
        if isinstance(value, torch.Tensor):
            return self.tensor(value)
        if isinstance(value, np.ndarray):
            return self.numpy(value)
        if hasattr(value, "data") and isinstance(getattr(value, "data"), torch.Tensor):
            return self.tensor(value.data)
        if isinstance(value, Mapping):
            return sum(self.value(v, include_scalars=include_scalars) for v in value.values())
        if isinstance(value, (list, tuple, set)):
            return sum(self.value(v, include_scalars=include_scalars) for v in value)
        if include_scalars and isinstance(value, (bool, int, float, np.number)):
            return 8
        return 0


def _model_parameter_bytes(model) -> int:
    counter = ByteCounter()
    return sum(counter.tensor(parameter) for parameter in model.parameters())


def _model_buffer_bytes(model, counter: ByteCounter) -> int:
    return sum(counter.tensor(buffer) for buffer in model.buffers())


def _er_ace_bytes(
    bundle: StrategyBundle, counter: ByteCounter
) -> tuple[int, int, int, str, str]:
    storage_policy = bundle.strategy.storage_policy
    if getattr(storage_policy, "one_hot_labels", False):
        examples = storage_policy.persistent_examples()
        if len(examples) > bundle.strategy.mem_size or len(examples) > 2000:
            raise RuntimeError(f"ER-ACE packed buffer exceeds its cap: {len(examples)}")
        sample_bytes = 0
        label_bytes = 0
        shape_metadata_bytes = 0
        for example in examples:
            example.validate(storage_policy.num_classes)
            if isinstance(example, PackedBinaryExample):
                sample_bytes += counter.numpy(example.data)
                shape_metadata_bytes += 8 * len(example.shape)
            elif isinstance(example, Float32ReplayExample):
                sample_bytes += counter.tensor(example.data)
            else:
                raise TypeError(f"Unexpected ER-ACE replay type: {type(example)!r}")
            label_bytes += counter.numpy(example.label_one_hot)
        auxiliary = counter.value(storage_policy.auxiliary_values())
        auxiliary += 8 * len(storage_policy.seen_classes) + shape_metadata_bytes
        encoding = "1-bit packed" if storage_policy.packed_binary else "float32_input_image"
        return sample_bytes, label_bytes, auxiliary, encoding, "uint8_one_hot"

    raise RuntimeError(
        "ER-ACE formal storage must use materialized replay samples with uint8 one-hot labels"
    )


def _icarl_bytes(bundle: StrategyBundle, counter: ByteCounter) -> tuple[int, int, int, str, str]:
    plugin = bundle.method_plugin
    sample_bytes = 0
    if plugin.pack_binary:
        if plugin.x_memory:
            raise RuntimeError("Packed Spike ICaRL must not retain float32 x_memory between tasks")
        for item in plugin.packed_memory:
            data = item["data"]
            shape = tuple(item["shape"])
            expected = (int(np.prod(shape)) + 7) // 8
            if data.dtype != np.uint8 or data.nbytes != expected:
                raise RuntimeError("Spike packed replay byte count does not match ceil(numel/8)")
            sample_bytes += counter.numpy(data)
        encoding = "1-bit packed"
        if len(plugin.packed_memory) != len(plugin.persistent_labels):
            raise RuntimeError("Packed ICaRL sample and label groups are inconsistent")
    else:
        if plugin.y_memory:
            raise RuntimeError("ICaRL must not retain decoded int64 labels between tasks")
        for tensor in plugin.x_memory:
            if tensor.dtype != torch.float32:
                raise TypeError("Non-Spike ICaRL replay samples must be float32")
            sample_bytes += counter.tensor(tensor)
        encoding = "float32_input_image"
    label_bytes = 0
    labels_count = 0
    for labels in plugin.persistent_labels:
        validate_uint8_one_hot(labels, plugin.num_classes)
        label_bytes += counter.numpy(labels)
        labels_count += int(labels.shape[0])
    label_encoding = "uint8_one_hot"
    if labels_count > plugin.memory_size or labels_count > 2000:
        raise RuntimeError(f"ICaRL replay count {labels_count} exceeds cap")
    auxiliary = counter.value(plugin.order)
    if plugin.pack_binary:
        auxiliary += sum(8 * len(item["shape"]) for item in plugin.packed_memory)
    auxiliary += 8 * len(plugin.observed_classes)
    if bundle.criterion_plugin is not None and bundle.criterion_plugin.old_model is not None:
        auxiliary += counter.value(bundle.criterion_plugin.old_model.state_dict())
        auxiliary += 8 * len(bundle.criterion_plugin.old_classes)
    return sample_bytes, label_bytes, auxiliary, encoding, label_encoding


def _tagfex_bytes(
    bundle: StrategyBundle, counter: ByteCounter
) -> tuple[int, int, int, str, str]:
    strategy = bundle.strategy
    sample_bytes = 0
    label_bytes = 0
    stored_count = 0
    for examples in strategy.memory_by_class.values():
        for example in examples:
            if isinstance(example, PackedBinaryExample):
                if not strategy.pack_binary_replay:
                    raise TypeError("Non-Spike TagFex cannot retain packed binary replay")
                example.validate(strategy.num_classes)
                sample_bytes += counter.numpy(example.data)
                label_bytes += counter.numpy(example.label_one_hot)
                stored_count += 1
            elif isinstance(example, Float32ReplayExample):
                if strategy.pack_binary_replay:
                    raise TypeError("Spike TagFex must not retain float32 replay samples between tasks")
                example.validate(strategy.num_classes)
                sample_bytes += counter.tensor(example.data)
                label_bytes += counter.numpy(example.label_one_hot)
                stored_count += 1
            else:
                raise TypeError(f"TagFex replay is not persistent: {type(example)!r}")
    if stored_count > strategy.hparams.memory_size or stored_count > 2000:
        raise RuntimeError(f"TagFex replay count {stored_count} exceeds its cap")
    auxiliary = counter.value(strategy.class_means)
    auxiliary += counter.value(
        None if strategy.last_ta_net is None else strategy.last_ta_net.state_dict()
    )
    auxiliary += counter.value(
        None if strategy.last_projector is None else strategy.last_projector.state_dict()
    )
    auxiliary += 8 * len(strategy.seen_classes)
    if strategy.pack_binary_replay:
        auxiliary += sum(
            8 * len(example.shape)
            for examples in strategy.memory_by_class.values()
            for example in examples
        )
        return sample_bytes, label_bytes, auxiliary, "1-bit packed", "uint8_one_hot"
    return sample_bytes, label_bytes, auxiliary, "float32_input_image", "uint8_one_hot"


def compute_persistent_storage(bundle: StrategyBundle) -> dict[str, Any]:
    model = bundle.strategy.model
    model_bytes = _model_parameter_bytes(model)
    aux_counter = ByteCounter()
    auxiliary = _model_buffer_bytes(model, aux_counter)
    # Optimizer state is required to continue learning; config-reconstructible param_groups are excluded.
    auxiliary += aux_counter.value(bundle.strategy.optimizer.state, include_scalars=True)
    sample_bytes = 0
    label_bytes = 0
    pulse_encoding = "NA"
    label_encoding = "NA"

    if bundle.method == "er_ace":
        sample_bytes, label_bytes, buffer_metadata, pulse_encoding, label_encoding = _er_ace_bytes(
            bundle, aux_counter
        )
        auxiliary += buffer_metadata
    elif bundle.method == "icarl":
        sample_bytes, label_bytes, extra, pulse_encoding, label_encoding = _icarl_bytes(bundle, aux_counter)
        auxiliary += extra
    elif bundle.method == "ewc":
        auxiliary += aux_counter.value(bundle.method_plugin.saved_params)
        auxiliary += aux_counter.value(bundle.method_plugin.importances)
    elif bundle.method == "cwr_star":
        saved = getattr(bundle.strategy.model, "saved_weights", {})
        auxiliary += aux_counter.value(saved)
        auxiliary += 8 * len(getattr(bundle.strategy.model, "past_j", {}))
        auxiliary += 8 * len(getattr(bundle.strategy.model, "cur_j", {}))
    elif bundle.method == "fecam":
        # FeCAM means/covariances are registered model buffers and already counted above.
        pass
    elif bundle.method == "tagfex":
        sample_bytes, label_bytes, extra, pulse_encoding, label_encoding = _tagfex_bytes(
            bundle, aux_counter
        )
        auxiliary += extra
    else:
        raise ValueError(bundle.method)

    total = int(model_bytes + sample_bytes + label_bytes + auxiliary)
    result = {
        "model_parameter_bytes": int(model_bytes),
        "replay_sample_bytes": int(sample_bytes),
        "replay_label_bytes": int(label_bytes),
        "auxiliary_bytes": int(auxiliary),
        "total_bytes": total,
        "total_mib": float(total / (2**20)),
        "pulse_encoding": pulse_encoding,
        "label_encoding": label_encoding,
    }
    if result["total_bytes"] != sum(
        result[key]
        for key in ("model_parameter_bytes", "replay_sample_bytes", "replay_label_bytes", "auxiliary_bytes")
    ):
        raise RuntimeError("Persistent storage components overlap or do not sum")
    return result
