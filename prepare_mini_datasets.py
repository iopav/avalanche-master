from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from cil_experiments.data import validate_image_files, validate_source_files
from cil_experiments.registry import DATASETS


def _stratified_indices(labels: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in sorted(int(value) for value in np.unique(labels)):
        candidates = np.flatnonzero(labels == class_id)
        count = max(1, int(round(len(candidates) * ratio)))
        selected.append(rng.choice(candidates, size=count, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def _atomic_subset(source_path: Path, destination: Path, indices: np.ndarray) -> None:
    source = np.load(source_path, mmap_mode="r")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    output = open_memmap(
        temporary,
        mode="w+",
        dtype=source.dtype,
        shape=(len(indices), *source.shape[1:]),
    )
    try:
        for output_start in range(0, len(indices), 32):
            batch_indices = indices[output_start : output_start + 32]
            output[output_start : output_start + len(batch_indices)] = source[batch_indices]
        output.flush()
        del output
        os.replace(temporary, destination)
    except BaseException:
        del output
        temporary.unlink(missing_ok=True)
        raise


def prepare_mini_datasets(
    source_root: Path,
    destination_root: Path,
    *,
    ratio: float = 0.50,
    seed: int = 62,
) -> dict:
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between zero and one")
    manifest = {
        "schema": "cil-stratified-mini-dataset-v1",
        "ratio_requested": float(ratio),
        "seed": int(seed),
        "selection": "independent deterministic stratified sampling per split and class",
        "datasets": {},
    }
    for dataset_name, spec in DATASETS.items():
        validate_source_files(source_root, spec)
        validate_image_files(source_root, spec)
        dataset_record = {}
        for split, raw_x, raw_y, image_x, image_y in (
            ("train", spec.train_x, spec.train_y, spec.image_train_x, spec.image_train_y),
            ("test", spec.test_x, spec.test_y, spec.image_test_x, spec.image_test_y),
        ):
            labels = np.load(source_root / raw_y)
            indices = _stratified_indices(labels, ratio, seed + (0 if split == "train" else 1))
            for relative_path in (raw_x, raw_y, image_x, image_y):
                _atomic_subset(
                    source_root / relative_path,
                    destination_root / relative_path,
                    indices,
                )
            counts = {
                str(class_id): int(np.sum(labels[indices] == class_id))
                for class_id in sorted(int(value) for value in np.unique(labels))
            }
            dataset_record[split] = {
                "source_samples": int(len(labels)),
                "mini_samples": int(len(indices)),
                "actual_ratio": float(len(indices) / len(labels)),
                "class_counts": counts,
                "source_indices": indices.tolist(),
            }
        validate_source_files(
            destination_root, spec, allow_variable_samples=True
        )
        validate_image_files(
            destination_root, spec, allow_variable_samples=True
        )
        manifest["datasets"][dataset_name] = dataset_record
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_root / "mini_dataset_manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified 50% CIL datasets")
    root = Path(__file__).resolve().parent
    parser.add_argument("--source-root", type=Path, default=root / "dataset")
    parser.add_argument("--destination-root", type=Path, default=root / "dataset_mini")
    parser.add_argument("--ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=62)
    args = parser.parse_args()
    manifest = prepare_mini_datasets(
        args.source_root.resolve(),
        args.destination_root.resolve(),
        ratio=args.ratio,
        seed=args.seed,
    )
    print(json.dumps({
        name: {
            split: values[split]["mini_samples"]
            for split in ("train", "test")
        }
        for name, values in manifest["datasets"].items()
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
