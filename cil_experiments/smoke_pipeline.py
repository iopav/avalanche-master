"""Run a small synthetic end-to-end PP2 pipeline on one allocated GPU."""

from __future__ import annotations

import argparse
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .order_seed_registry import ORDERS_BY_DATASET
from .output import atomic_write_json
from .registry import DATASETS, FORMAL_METHODS
from .search_schema import timestamp


SMOKE_DATASETS = ("spike", "texture", "uwave")
SMOKE_METHODS = FORMAL_METHODS
SMOKE_ORDER_IDS = (1, 2)
SMOKE_SEEDS = (62,)
SMOKE_LRS = (0.1, 0.01)
TRAIN_SAMPLES_PER_CLASS = 2
TEST_SAMPLES_PER_CLASS = 1
SMOKE_EPOCHS = 1


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _labels(num_classes: int, samples_per_class: int) -> np.ndarray:
    return np.repeat(
        np.arange(num_classes, dtype=np.int64), int(samples_per_class)
    )


def _raw_array(spec, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shape = (len(labels), *spec.train_shape[1:])
    if spec.name == "spike":
        values = rng.integers(0, 2, size=shape, dtype=np.int8).astype(np.float32)
        for index, label in enumerate(labels):
            values[index, :, int(label) % spec.in_channels] = 1.0
        return values
    values = rng.normal(0.0, 0.25, size=shape).astype(spec.source_x_dtype)
    for index, label in enumerate(labels):
        values[index, :, int(label) % spec.in_channels] += (
            float(label) + 1.0
        ) / spec.num_classes
    return values


def _image_array(spec, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shape = (len(labels), *spec.image_train_shape[1:])
    if spec.image_x_dtype == "uint8":
        values = rng.integers(0, 32, size=shape, dtype=np.uint8)
        for index, label in enumerate(labels):
            values[index].flat[int(label) % values[index].size] = 255
        return values
    values = rng.normal(0.0, 0.25, size=shape).astype(np.float32)
    for index, label in enumerate(labels):
        values[index] += (float(label) + 1.0) / spec.num_classes
    return values


def generate_toy_datasets(root: Path, *, seed: int = 20260906) -> dict[str, Any]:
    """Write complete raw/image mirrors with very few samples per class."""

    root = Path(root)
    rng = np.random.default_rng(seed)
    manifest: dict[str, Any] = {}
    for dataset_name in SMOKE_DATASETS:
        spec = DATASETS[dataset_name]
        train_y = _labels(spec.num_classes, TRAIN_SAMPLES_PER_CLASS)
        test_y = _labels(spec.num_classes, TEST_SAMPLES_PER_CLASS)
        train_permutation = rng.permutation(len(train_y))
        test_permutation = rng.permutation(len(test_y))
        train_y = train_y[train_permutation]
        test_y = test_y[test_permutation]

        train_raw = _raw_array(spec, train_y, rng)
        test_raw = _raw_array(spec, test_y, rng)
        train_image = _image_array(spec, train_y, rng)
        test_image = _image_array(spec, test_y, rng)

        for relative_path, array in (
            (spec.train_x, train_raw),
            (spec.train_y, train_y),
            (spec.test_x, test_raw),
            (spec.test_y, test_y),
            (spec.image_train_x, train_image),
            (spec.image_train_y, train_y),
            (spec.image_test_x, test_image),
            (spec.image_test_y, test_y),
        ):
            _atomic_save(root / relative_path, array)

        manifest[dataset_name] = {
            "classes": spec.num_classes,
            "train_samples": len(train_y),
            "test_samples": len(test_y),
            "raw_sample_shape": list(train_raw.shape[1:]),
            "image_sample_shape": list(train_image.shape[1:]),
        }
    return manifest


def _parameter_overrides(method: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "epochs_per_experience": SMOKE_EPOCHS,
        "train_mb_size": 4,
        "eval_mb_size": 4,
        "num_workers": 0,
    }
    if method == "er_ace":
        overrides.update(memory_size=8, batch_size_mem=4)
    elif method == "icarl":
        overrides.update(memory_size=8)
    elif method == "tagfex":
        overrides.update(
            memory_size=8,
            init_epochs=SMOKE_EPOCHS,
            inc_epochs=SMOKE_EPOCHS,
            proj_hidden_dim=64,
            proj_output_dim=32,
        )
    return overrides


def run_smoke(output_root: Path, *, device_name: str, exp_name: str) -> Path:
    import torch

    from .aggregate_results import write_dataset_reports
    from .joint_learning import run_joint_unit
    from .lr_search import run_search_unit

    output_root = Path(output_root).resolve()
    toy_root = output_root / "toy_data"
    search_root = output_root / "search"
    joint_root = output_root / "joint"
    manifest_path = output_root / "smoke_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The Arrhenius smoke pipeline requires an allocated CUDA GPU")

    manifest: dict[str, Any] = {
        "schema": "pp2-toy-smoke-v1",
        "status": "running",
        "started_at": timestamp(),
        "finished_at": None,
        "exp_name": exp_name,
        "output_root": str(output_root),
        "device": str(device),
        "backbone": "temporal",
        "datasets": list(SMOKE_DATASETS),
        "methods": list(SMOKE_METHODS),
        "order_ids": list(SMOKE_ORDER_IDS),
        "seeds": list(SMOKE_SEEDS),
        "lr_candidates": list(SMOKE_LRS),
        "epochs_per_experience": SMOKE_EPOCHS,
        "toy_data": generate_toy_datasets(toy_root),
        "reports": [],
    }
    atomic_write_json(manifest_path, manifest)

    try:
        for dataset in SMOKE_DATASETS:
            for method in SMOKE_METHODS:
                overrides = _parameter_overrides(method)
                for order_id in SMOKE_ORDER_IDS:
                    if order_id not in ORDERS_BY_DATASET[dataset]:
                        raise KeyError(f"Missing smoke order {dataset}/order-{order_id}")
                    for seed in SMOKE_SEEDS:
                        search_path = run_search_unit(
                            project_root=Path(__file__).resolve().parents[1],
                            dataset_root=toy_root,
                            search_root=search_root,
                            exp_name=exp_name,
                            dataset=dataset,
                            method=method,
                            order_id=order_id,
                            seed=seed,
                            device=device,
                            backbone="temporal",
                            loss_selection="last_epoch_train_mean",
                            lr_candidates=SMOKE_LRS,
                            parameter_overrides=overrides,
                            data_role="smoke",
                        )
                        run_joint_unit(
                            dataset_root=toy_root,
                            search_root=search_root,
                            joint_root=joint_root,
                            exp_name=exp_name,
                            dataset=dataset,
                            method=method,
                            order_id=order_id,
                            seed=seed,
                            device=device,
                            backbone="temporal",
                            parameter_overrides=overrides,
                            data_role="smoke",
                        )
                        print(
                            "SMOKE_UNIT_COMPLETE "
                            f"dataset={dataset} method={method} order={order_id} "
                            f"seed={seed} search={search_path}",
                            flush=True,
                        )

            reports = write_dataset_reports(
                search_root,
                joint_root,
                dataset=dataset,
                methods=SMOKE_METHODS,
                order_ids=SMOKE_ORDER_IDS,
                seeds=SMOKE_SEEDS,
            )
            manifest["reports"].extend(str(path) for path in reports)

        manifest["status"] = "completed"
        manifest["finished_at"] = timestamp()
        atomic_write_json(manifest_path, manifest)
        print(f"SMOKE_COMPLETE manifest={manifest_path}", flush=True)
        return manifest_path
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = timestamp()
        manifest["error_type"] = type(exc).__name__
        manifest["error_message"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        atomic_write_json(manifest_path, manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exp-name", default="arrhenius_toy_smoke")
    args = parser.parse_args()
    run_smoke(args.output_root, device_name=args.device, exp_name=args.exp_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
