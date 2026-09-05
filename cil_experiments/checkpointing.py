from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import dill
import torch

from .strategies import StrategyBundle


CHECKPOINT_SCHEMA = "pp2-search-strategy-checkpoint-v3"


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
    return checkpoint_path


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
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return destination
