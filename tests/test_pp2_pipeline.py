from __future__ import annotations

import gc
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cil_experiments.aggregate_results import (
    write_formal_aggregate,
    write_joint_test_report,
    write_method_reports,
)
from cil_experiments.checkpointing import load_search_checkpoint
from cil_experiments.data import build_dataset_bundle
from cil_experiments.intransigence import _validate_identity, compute_intransigence
from cil_experiments.lr_search import (
    finalize_search,
    formal_parameters,
    run_search,
    select_best,
    selected_checkpoint,
)
from cil_experiments.models import BACKBONES, build_backbone
from cil_experiments.order_seed_registry import FORMAL_SEEDS, ORDERS_BY_DATASET, SEEDS
from cil_experiments.output import run_artifact_paths
from cil_experiments.registry import DATASETS
from cil_experiments.runner import evaluate_joint_checkpoint, run_experiment
from cil_experiments.search_schema import STATUS_COMPLETED, new_search_result
from cil_experiments.validation import create_validation_split


def _run(acc: float, flops: int, total: int):
    return {
        "status": STATUS_COMPLETED,
        "val_avg_acc": acc,
        "val_average_incremental_accuracy": acc,
        "overall_learning_flops": flops,
        "storage_param_bytes": 10,
        "replay_sample_label_bytes": total - 12,
        "auxiliary_bytes": 2,
        "total_bytes": total,
    }


def _make_tiny_dataset(root: Path, dataset_name: str):
    folder = root / dataset_name
    folder.mkdir(parents=True, exist_ok=True)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    test_labels = np.array([0, 0, 1, 1])
    rng = np.random.default_rng({"spike": 5, "texture": 6, "uwave": 7}[dataset_name])
    train_images = rng.normal(0.0, 0.1, (8, 3, 8, 8)).astype(np.float32)
    test_images = rng.normal(0.0, 0.1, (4, 3, 8, 8)).astype(np.float32)
    if dataset_name == "spike":
        train_raw = rng.integers(0, 2, (8, 400, 64)).astype(np.float32)
        test_raw = rng.integers(0, 2, (4, 400, 64)).astype(np.float32)
        raw_shape, raw_test_shape = (8, 400, 64), (4, 400, 64)
        timesteps, in_channels = 400, 64
    else:
        train_raw = np.transpose(train_images, (0, 2, 3, 1))
        test_raw = np.transpose(test_images, (0, 2, 3, 1))
        raw_shape, raw_test_shape = (8, 8, 8, 3), (4, 8, 8, 3)
        timesteps, in_channels = 8, 3
    for name, value in {
        "raw_x.npy": train_raw,
        "raw_y.npy": labels,
        "raw_test_x.npy": test_raw,
        "raw_test_y.npy": test_labels,
        "image_x.npy": train_images,
        "image_y.npy": labels,
        "test_image_x.npy": test_images,
        "test_image_y.npy": test_labels,
    }.items():
        np.save(folder / name, value)
    return replace(
        DATASETS[dataset_name],
        train_x=f"{dataset_name}/raw_x.npy",
        train_y=f"{dataset_name}/raw_y.npy",
        test_x=f"{dataset_name}/raw_test_x.npy",
        test_y=f"{dataset_name}/raw_test_y.npy",
        train_shape=raw_shape,
        test_shape=raw_test_shape,
        source_x_dtype="float32",
        timesteps=timesteps,
        in_channels=in_channels,
        image_train_x=f"{dataset_name}/image_x.npy",
        image_train_y=f"{dataset_name}/image_y.npy",
        image_test_x=f"{dataset_name}/test_image_x.npy",
        image_test_y=f"{dataset_name}/test_image_y.npy",
        image_train_shape=(8, 3, 8, 8),
        image_test_shape=(4, 3, 8, 8),
        image_x_dtype="float32",
        image_layout="NCHW",
        num_classes=2,
    )


class PP2PipelineTests(unittest.TestCase):
    def test_backbone_registry_contains_only_resnet_and_temporal(self):
        self.assertEqual(set(BACKBONES), {"resnet18_cifar", "temporal"})
        model = build_backbone("temporal", (3, 16, 16))
        convolutions = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
        self.assertEqual(
            [(layer.in_channels, layer.out_channels) for layer in convolutions],
            [(3, 8), (8, 16), (16, 32)],
        )
        features, maps = model.forward_with_feature_maps(torch.zeros(2, 3, 16, 16))
        self.assertEqual(tuple(features.shape), (2, 32))
        self.assertEqual(len(maps), 3)

    def test_string_experiment_path_has_no_repeated_suffix(self):
        paths = run_artifact_paths(
            Path("search_result_alpha"), "uwave", "ewc", 1, 62,
            Path("order-01") / "lr-0.1",
        )
        self.assertEqual(
            paths.method_root, Path("search_result_alpha/uwave/ewc/order-01/lr-0.1")
        )
        self.assertNotIn("exp-", paths.summary.name)

    def test_search_schema_and_selection(self):
        payload = new_search_result(
            exp_name="alpha", dataset="uwave", method="ewc",
            validation_split_file=Path("split.json"), search_seed=62,
            order_ids=(1, 4, 9),
        )
        self.assertEqual(set(payload["search_runs"]), {"order1", "order4", "order9"})
        self.assertEqual(sum(map(len, payload["search_runs"].values())), 9)
        runs = {
            "0.1": _run(0.8, 100, 30),
            "0.05": _run(0.797, 80, 40),
            "0.01": _run(0.79, 60, 50),
        }
        self.assertEqual(select_best(runs), 0.05)
        one = new_search_result(
            exp_name="alpha", dataset="uwave", method="ewc",
            validation_split_file=Path("split.json"), search_seed=62,
            order_ids=(1,),
        )
        one["search_runs"]["order1"] = runs
        finalize_search(one, (1,))
        self.assertEqual(one["search_metrics"]["search_flops_each_order"], [240])
        self.assertEqual(one["search_metrics"]["search_storage_each_order"], [10])

    def test_search_and_formal_use_separate_materialized_data(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            spec = _make_tiny_dataset(root, "uwave")
            with patch.dict(DATASETS, {"uwave": spec}), patch.dict(
                ORDERS_BY_DATASET, {"uwave": {1: ((0,), (1,))}}
            ):
                create_validation_split(root, "uwave", seed=62, validation_fraction=1 / 4)
                search = build_dataset_bundle(root, spec, 1, data_role="search")
                formal = build_dataset_bundle(root, spec, 1, data_role="formal")
                self.assertEqual((len(search.train), len(search.test)), (6, 2))
                self.assertEqual((len(formal.train), len(formal.test)), (8, 4))
                del search, formal
                gc.collect()

    def test_tiny_search_checkpoint_joint_and_ewc_formal_runs(self):
        root = Path(__file__).resolve().parent / "pp2_tiny_run"
        specs = {
            dataset: _make_tiny_dataset(root, dataset)
            for dataset in ("spike", "texture", "uwave")
        }
        orders = {dataset: {1: ((0,), (1,))} for dataset in specs}
        device = torch.device("cpu")
        exp_name = "tiny_ewc_weight_only_v4"
        search_root = root / f"search_result_{exp_name}"
        joint_root = root / "joint_result" / exp_name
        formal_root = root / f"result_{exp_name}"
        with patch.dict(DATASETS, specs, clear=True), patch.dict(
            ORDERS_BY_DATASET, orders, clear=True
        ), patch("cil_experiments.lr_search.SEARCH_EPOCHS", 1):
            for dataset in specs:
                create_validation_split(
                    root, dataset, seed=62, validation_fraction=1 / 4, overwrite=True
                )
                search_path = run_search(
                    root, root, search_root, exp_name,
                    dataset, "ewc", device, "temporal",
                )
                search = json.loads(search_path.read_text(encoding="utf-8"))
                checkpoint = selected_checkpoint(search, 1)
                saved = load_search_checkpoint(checkpoint)
                self.assertEqual(saved["metadata"]["seed"], 62)
                self.assertEqual(saved["bundle"].method, "ewc")
                for seed in SEEDS:
                    evaluate_joint_checkpoint(
                        dataset_root=root,
                        result_root=joint_root,
                        dataset_name=dataset,
                        order_id=1,
                        seed=seed,
                        device=device,
                        checkpoint_path=checkpoint,
                        exp_name=exp_name,
                        source_method="ewc",
                        overwrite=True,
                        resume=False,
                    )
                selected_lr = float(search["best_per_order"][0])
                parameters = formal_parameters(dataset, "ewc", selected_lr)
                parameters.update(
                    epochs_per_experience=1, train_mb_size=2, eval_mb_size=2
                )
                for seed in FORMAL_SEEDS:
                    run_experiment(
                        project_root=root,
                        dataset_root=root,
                        result_root=formal_root,
                        dataset_name=dataset,
                        method="ewc",
                        order_id=1,
                        seed=seed,
                        epochs=None,
                        device=device,
                        parameter_overrides=parameters,
                        search_provenance={
                            "search_result_file": str(search_path.resolve()),
                            "search_seed": 62,
                            "selected_lr": selected_lr,
                            "hyperparameter_search_flops": search["search_metrics"][
                                "search_flops_each_order"
                            ][0],
                            "test_set_used_for_selection": False,
                        },
                        backbone_id="temporal",
                        compute_intransigence_enabled=True,
                        joint_result_root=joint_root,
                        exp_name=exp_name,
                        data_role="formal",
                        measure_latency=True,
                        overwrite=True,
                    )

            ewc_reports = write_method_reports(
                search_root, formal_root, exp_name, "ewc"
            )
            formal_aggregate = write_formal_aggregate(
                formal_root,
                datasets=tuple(specs),
                methods=("ewc",),
                seeds=FORMAL_SEEDS,
                exp_name=exp_name,
            )
            joint_report = write_joint_test_report(joint_root, exp_name, "ewc")
            self.assertTrue(all(path.is_file() for path in ewc_reports))
            self.assertTrue(all(path.is_file() for path in formal_aggregate))
            self.assertTrue(joint_report.is_file())
            joint_matrix = json.loads(
                (joint_root / "uwave/joint/summary/uwave__joint__order-01__seed-062__accuracy-matrix.json").read_text(encoding="utf-8")
            )
            self.assertEqual(joint_matrix["checkpoint_source_method"], "ewc")
            self.assertEqual(joint_matrix["checkpoint_source_seed"], 62)
            ewc_summary = json.loads(
                (formal_root / "uwave/ewc/summary/uwave__ewc__order-01__seed-063__summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(ewc_summary["cil_performance"]["intransigence"]), 2)
        gc.collect()

    def test_intransigence_uses_joint_row_and_cil_diagonal(self):
        cil = {
            "tasks": 2,
            "accuracy_matrix_lower_triangular": [[0.7, None], [0.4, 0.6]],
        }
        joint = {
            "tasks": 2,
            "accuracy_by_task": [0.8, 0.9],
        }
        np.testing.assert_allclose(compute_intransigence(cil, joint), [0.1, 0.3])

    def test_joint_reference_allows_same_experiment_name(self):
        cil_matrix = {
            "dataset": "uwave", "method": "ewc", "order_id": 1,
            "seed": 63, "exp_name": "ewc",
        }
        joint_matrix = {**cil_matrix, "method": "joint"}
        common = {
            "backbone": {"backbone_id": "resnet18_cifar"},
            "protocol": {"input_view_id": "uwave_stored_rgb_image_v1"},
            "selected_task_groups": [[0], [1]],
        }
        _validate_identity(cil_matrix, common, joint_matrix, common)


if __name__ == "__main__":
    unittest.main()
