from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

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


def _er_ace_bytes(bundle: StrategyBundle) -> tuple[int, int, int, str, str]:
    buffer = bundle.strategy.storage_policy.buffer
    sample_bytes = 0
    label_bytes = 0
    count = 0
    label_dtypes: set[str] = set()
    for index in range(len(buffer)):
        item = buffer[index]
        x, y = item[0], item[1]
        if not isinstance(x, torch.Tensor) or x.dtype != torch.float32:
            raise AssertionError("ER-ACE buffer samples must use the actual float32 training representation")
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        sample_bytes += int(x.numel() * x.element_size())
        label_bytes += int(y.numel() * y.element_size())
        label_dtypes.add(str(y.dtype).removeprefix("torch."))
        count += 1
    configured = int(bundle.strategy.mem_size)
    if count > configured or count > 2000:
        raise AssertionError(f"ER-ACE buffer exceeds configured/capacity limit: {count}")
    metadata_bytes = len(getattr(bundle.strategy.storage_policy, "buffer_groups", {})) * 16
    if len(label_dtypes) != 1:
        raise AssertionError(f"ER-ACE buffer label dtype is inconsistent: {label_dtypes}")
    return sample_bytes, label_bytes, metadata_bytes, "float32_raw_timeseries", next(iter(label_dtypes))


def _icarl_bytes(bundle: StrategyBundle, counter: ByteCounter) -> tuple[int, int, int, str, str]:
    plugin = bundle.method_plugin
    sample_bytes = 0
    if plugin.pack_binary:
        if plugin.x_memory:
            raise AssertionError("Packed Spike ICaRL must not retain float32 x_memory between tasks")
        for item in plugin.packed_memory:
            data = item["data"]
            shape = tuple(item["shape"])
            expected = (int(np.prod(shape)) + 7) // 8
            if data.dtype != np.uint8 or data.nbytes != expected:
                raise AssertionError("Spike packed replay byte count does not match ceil(numel/8)")
            sample_bytes += counter.numpy(data)
        encoding = "1-bit packed"
    else:
        for tensor in plugin.x_memory:
            if tensor.dtype != torch.float32:
                raise AssertionError("Non-Spike ICaRL replay samples must be float32")
            sample_bytes += counter.tensor(tensor)
        encoding = "float32"
    label_bytes = 0
    labels_count = 0
    for labels in plugin.y_memory:
        labels = np.asarray(labels)
        if labels.dtype != np.int64:
            raise AssertionError(f"ICaRL label storage must be int64, found {labels.dtype}")
        label_bytes += counter.numpy(labels)
        labels_count += len(labels)
    if labels_count > plugin.memory_size or labels_count > 2000:
        raise AssertionError(f"ICaRL replay count {labels_count} exceeds cap")
    auxiliary = counter.value(plugin.order)
    auxiliary += 8 * len(plugin.observed_classes)
    if bundle.criterion_plugin is not None and bundle.criterion_plugin.old_model is not None:
        auxiliary += counter.value(bundle.criterion_plugin.old_model.state_dict())
        auxiliary += 8 * len(bundle.criterion_plugin.old_classes)
    return sample_bytes, label_bytes, auxiliary, encoding, "int64"


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
        sample_bytes, label_bytes, buffer_metadata, pulse_encoding, label_encoding = _er_ace_bytes(bundle)
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
        raise AssertionError("Persistent storage components overlap or do not sum")
    return result
