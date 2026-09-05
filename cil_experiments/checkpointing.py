from __future__ import annotations

import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import dill
import torch

from .strategies import StrategyBundle


CHECKPOINT_SCHEMA = "pp2-search-strategy-checkpoint-v3"
WEIGHT_CHECKPOINT_SCHEMA = "pp2-joint-weight-checkpoint-v1"


class WeightOnlyModel(torch.nn.Module):
    """Inference model containing trainable network state, not method state."""

    def __init__(self, feature_extractor: torch.nn.Module, classifier: torch.nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


def _weight_only_model(model: torch.nn.Module) -> torch.nn.Module:
    """Detach inference weights from strategy-owned statistics and plugins."""

    if hasattr(model, "feature_extractor") and hasattr(model, "train_classifier"):
        # iCaRL and FeCAM use class means/covariances through eval_classifier.
        # Joint deliberately evaluates the learned parametric classifier instead.
        return WeightOnlyModel(
            deepcopy(model.feature_extractor).cpu(),
            deepcopy(model.train_classifier).cpu(),
        )
    result = deepcopy(model).cpu()
    for module in result.modules():
        for name in (
            "class_means",
            "class_means_dict",
            "class_cov_dict",
            "saved_weights",
            "past_j",
            "cur_j",
        ):
            if hasattr(module, name):
                value = getattr(module, name)
                setattr(module, name, {} if isinstance(value, dict) else None)
    return result


def save_search_checkpoint(
    bundle: StrategyBundle, metadata: dict[str, Any], checkpoint_path: Path
) -> Path:
    """Atomically save the complete strategy selected by LR search."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=checkpoint_path.parent
    )
    os.close(fd)
    staged = Path(name)
    try:
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "metadata": dict(metadata),
                "bundle": bundle,
            },
            staged,
            pickle_module=dill,
        )
        with staged.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(staged, checkpoint_path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    weight_path = weight_checkpoint_path(checkpoint_path)
    torch.save(
        {
            "schema": WEIGHT_CHECKPOINT_SCHEMA,
            "metadata": dict(metadata),
            "model": _weight_only_model(bundle.strategy.model),
        },
        weight_path,
        pickle_module=dill,
    )
    return checkpoint_path


def weight_checkpoint_path(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.weights{checkpoint_path.suffix}")


def load_weight_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    path = weight_checkpoint_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(
        path, map_location="cpu", pickle_module=dill, weights_only=False
    )
    if not isinstance(payload, dict) or payload.get("schema") != WEIGHT_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported joint weight checkpoint: {path}")
    if not isinstance(payload.get("model"), torch.nn.Module):
        raise ValueError(f"Joint checkpoint does not contain a model: {path}")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Joint checkpoint metadata is invalid: {path}")
    return payload


def load_search_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        pickle_module=dill,
        weights_only=False,
    )
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported search checkpoint: {checkpoint_path}")
    if not isinstance(payload.get("bundle"), StrategyBundle):
        raise ValueError(f"Checkpoint does not contain a StrategyBundle: {checkpoint_path}")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Checkpoint metadata is invalid: {checkpoint_path}")
    return payload


def promote_search_checkpoint(source: Path, destination: Path) -> Path:
    """Atomically copy one candidate checkpoint to its stable best path."""

    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    staged = Path(name)
    try:
        shutil.copyfile(source, staged)
        with staged.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(staged, destination)
        shutil.copyfile(weight_checkpoint_path(source), weight_checkpoint_path(destination))
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return destination
