from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from cil_experiments.data import validate_image_files
from cil_experiments.registry import DATASETS


def _write_array_atomic(
    source: np.ndarray,
    destination: Path,
    output_shape: tuple[int, ...],
    output_dtype: np.dtype,
    transform,
    *,
    overwrite: bool,
    chunk_size: int = 16,
) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    target = open_memmap(temporary, mode="w+", dtype=output_dtype, shape=output_shape)
    try:
        for start in range(0, len(source), chunk_size):
            stop = min(len(source), start + chunk_size)
            target[start:stop] = transform(np.asarray(source[start:stop]))
        target.flush()
        del target
        os.replace(temporary, destination)
    except BaseException:
        del target
        temporary.unlink(missing_ok=True)
        raise


def _image_transform(dataset: str):
    if dataset == "spike":
        def transform(batch: np.ndarray) -> np.ndarray:
            plane = batch.reshape(len(batch), 160, 160).astype(np.uint8, copy=False)
            return np.repeat((plane * np.uint8(255))[:, None], 3, axis=1)

        return transform, np.dtype(np.uint8)
    if dataset == "texture":
        def transform(batch: np.ndarray) -> np.ndarray:
            flat = batch.reshape(len(batch), -1).astype(np.float32, copy=False)
            padded = np.pad(flat, ((0, 0), (0, 250)), mode="constant")
            return np.repeat(padded.reshape(len(batch), 1, 185, 185), 3, axis=1)

        return transform, np.dtype(np.float32)
    raise ValueError(f"No generated image transform for {dataset}")


def prepare_dataset(dataset_root: Path, dataset: str, *, overwrite: bool) -> None:
    spec = DATASETS[dataset]
    transform, dtype = _image_transform(dataset)
    for source_x, source_y, destination_x, destination_y, shape in (
        (spec.train_x, spec.train_y, spec.image_train_x, spec.image_train_y, spec.image_train_shape),
        (spec.test_x, spec.test_y, spec.image_test_x, spec.image_test_y, spec.image_test_shape),
    ):
        x = np.load(dataset_root / source_x, mmap_mode="r")
        y = np.load(dataset_root / source_y, mmap_mode="r")
        _write_array_atomic(
            x,
            dataset_root / destination_x,
            shape,
            dtype,
            transform,
            overwrite=overwrite,
        )
        _write_array_atomic(
            y,
            dataset_root / destination_y,
            tuple(y.shape),
            y.dtype,
            lambda batch: batch,
            overwrite=overwrite,
            chunk_size=1024,
        )
    validate_image_files(dataset_root, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic image-view NPY files")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).parent / "dataset")
    parser.add_argument("--datasets", nargs="+", choices=("spike", "texture"), default=("spike", "texture"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        prepare_dataset(args.dataset_root.resolve(), dataset, overwrite=args.overwrite)
        print(f"prepared={dataset}", flush=True)


if __name__ == "__main__":
    main()
