from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cil_experiments.aggregate_results import write_dataset_reports
from cil_experiments.final_hyperparameters import FINAL_HYPERPARAMETERS
from cil_experiments.intransigence import compute_intransigence
from cil_experiments.joint_learning import joint_run_path, run_joint_unit
from cil_experiments.lr_search import (
    _new_payload,
    candidate_paths,
    lr_token,
    run_search_unit,
    search_unit_path,
)
from cil_experiments.order_seed_registry import ORDERS_BY_DATASET
from cil_experiments.registry import DATASETS
from cil_experiments.runner import _LastEpochLossTracker, _selection_experiences
from cil_experiments.runner import _mean_cross_entropy


def _tiny_uwave(root: Path):
    folder = root / "uwave"
    folder.mkdir(parents=True, exist_ok=True)
    train_y = np.array([0, 1] * 6, dtype=np.int64)
    test_y = np.array([0, 1] * 3, dtype=np.int64)
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(12, 3, 8, 8)).astype(np.float32)
    test_x = rng.normal(size=(6, 3, 8, 8)).astype(np.float32)
    for name, value in {
        "raw_train.npy": train_x,
        "raw_train_y.npy": train_y,
        "raw_test.npy": test_x,
        "raw_test_y.npy": test_y,
        "image_train.npy": train_x,
        "image_train_y.npy": train_y,
        "image_test.npy": test_x,
        "image_test_y.npy": test_y,
    }.items():
        np.save(folder / name, value)
    return replace(
        DATASETS["uwave"],
        train_x="uwave/raw_train.npy",
        train_y="uwave/raw_train_y.npy",
        test_x="uwave/raw_test.npy",
        test_y="uwave/raw_test_y.npy",
        train_shape=(12, 3, 8, 8),
        test_shape=(6, 3, 8, 8),
        source_x_dtype="float32",
        image_train_x="uwave/image_train.npy",
        image_train_y="uwave/image_train_y.npy",
        image_test_x="uwave/image_test.npy",
        image_test_y="uwave/image_test_y.npy",
        image_train_shape=(12, 3, 8, 8),
        image_test_shape=(6, 3, 8, 8),
        image_x_dtype="float32",
        image_layout="NCHW",
        num_classes=2,
    )


class PP2PipelineTests(unittest.TestCase):
    def test_paths_schema_lr_token_and_malformed_completed_unit_rejected(self):
        root = Path("search_result_demo")
        self.assertEqual(lr_token(0.1), "01")
        self.assertEqual(lr_token(0.05), "005")
        self.assertEqual(lr_token(0.01), "001")
        self.assertEqual(
            search_unit_path(root, "ewc", 1, 62),
            root / "ewc/order1/order1_seed062_search.json",
        )
        paths = candidate_paths(root, "spike", "ewc", 1, 62, 0.1)
        self.assertEqual(paths.method_root, root / "ewc/order1/lr01")
        self.assertEqual(paths.summary.parent, paths.method_root)
        self.assertNotIn("config", paths.summary.name)
        payload = _new_payload(
            "demo", "spike", "ewc", 1, 62, "temporal", "last_epoch_train_mean"
        )
        self.assertEqual(payload["schema"], "pp2-order-seed-search-v1")
        self.assertEqual(len(payload["candidates"]), 3)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            temp_root = Path(name)
            unit_path = search_unit_path(temp_root, "ewc", 1, 62)
            files = [temp_root / value for value in ("best.pt", "best.json", "matrix.json")]
            for file in files:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("{}", encoding="utf-8")
            candidate = {
                "status": "completed",
                "selection_loss": 1.0,
                "selection_loss_eval_flops": 0,
                "overall_learning_flops": 10,
                "storage_bytes": 20,
                "summary_file": str(files[1]),
                "accuracy_matrix_file": str(files[2]),
            }
            payload.update(
                status="completed",
                candidates={key: dict(candidate) for key in payload["candidates"]},
                best_lr=0.1,
                best_loss=1.0,
                best_checkpoint=str(files[0]),
                best_summary_file=str(files[1]),
                best_accuracy_matrix_file=str(files[2]),
                total_search_flops=30,
                storage_bytes=20,
            )
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_search_unit(
                    project_root=temp_root,
                    dataset_root=temp_root,
                    search_root=temp_root,
                    exp_name="demo",
                    dataset="spike",
                    method="ewc",
                    order_id=1,
                    seed=62,
                    device=torch.device("cpu"),
                    backbone="temporal",
                    loss_selection="last_epoch_train_mean",
                )

    def test_loss_modes_and_fecam_fallback(self):
        tracker = _LastEpochLossTracker()
        strategy = type("Strategy", (), {})()
        tracker.before_training_epoch(strategy)
        strategy.loss = torch.tensor(2.0)
        strategy.mb_y = torch.zeros(2, dtype=torch.long)
        tracker.after_training_iteration(strategy)
        strategy.loss = torch.tensor(1.0)
        strategy.mb_y = torch.zeros(4, dtype=torch.long)
        tracker.after_training_iteration(strategy)
        tracker.after_training_epoch(strategy)
        self.assertAlmostEqual(tracker.last_epoch_mean, 4 / 3)
        stream = [object(), object(), object()]
        self.assertEqual(
            _selection_experiences("full_train_final_model", "ewc", stream), stream
        )
        self.assertEqual(
            _selection_experiences("last_epoch_train_mean", "fecam", stream),
            [stream[-1]],
        )
        self.assertIsNone(
            _selection_experiences("last_epoch_train_mean", "ewc", stream)
        )
        class EvalDataset(torch.utils.data.TensorDataset):
            def eval(self):
                return self

        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        experience = SimpleNamespace(
            dataset=EvalDataset(
                torch.zeros(3, 2), torch.tensor([0, 1, 0], dtype=torch.long)
            )
        )
        loss, flops = _mean_cross_entropy(
            model, [experience], torch.device("cpu"), 2, 0
        )
        self.assertAlmostEqual(loss, float(np.log(2)), places=6)
        self.assertGreater(flops, 0)

    def test_temporal_search_joint_intransigence_and_csv_smoke(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            spec = _tiny_uwave(root)
            search_root = root / "search_result_smoke"
            joint_root = root / "joint_result_smoke"
            parameters = {
                **FINAL_HYPERPARAMETERS["uwave"]["ewc"],
                "epochs_per_experience": 1,
                "train_mb_size": 4,
                "eval_mb_size": 3,
                "num_workers": 0,
            }
            orders = {"uwave": {1: ((0,), (1,))}}
            with patch.dict(DATASETS, {"uwave": spec}, clear=True), patch.dict(
                ORDERS_BY_DATASET, orders, clear=True
            ), patch.dict(
                FINAL_HYPERPARAMETERS["uwave"]["ewc"], parameters, clear=True
            ), patch(
                "cil_experiments.lr_search.SEEDS", (62,)
            ), patch(
                "cil_experiments.joint_learning.SEEDS", (62,)
            ), patch(
                "cil_experiments.aggregate_results.SEEDS", (62,)
            ):
                search_json = run_search_unit(
                    project_root=root,
                    dataset_root=root,
                    search_root=search_root,
                    exp_name="smoke",
                    dataset="uwave",
                    method="ewc",
                    order_id=1,
                    seed=62,
                    device=torch.device("cpu"),
                    backbone="temporal",
                    loss_selection="last_epoch_train_mean",
                )
                joint_json = run_joint_unit(
                    dataset_root=root,
                    search_root=search_root,
                    joint_root=joint_root,
                    exp_name="smoke",
                    dataset="uwave",
                    method="ewc",
                    order_id=1,
                    seed=62,
                    device=torch.device("cpu"),
                    backbone="temporal",
                )
                paths = write_dataset_reports(
                    search_root,
                    joint_root,
                    dataset="uwave",
                    methods=("ewc",),
                )
                search = json.loads(search_json.read_text(encoding="utf-8"))
                summary = json.loads(
                    Path(search["best_summary_file"]).read_text(encoding="utf-8")
                )
                joint = json.loads(joint_json.read_text(encoding="utf-8"))
                self.assertEqual(search["status"], "completed")
                self.assertEqual(len(search["candidates"]), 3)
                self.assertEqual(len(joint["accuracy_by_task"]), 2)
                self.assertEqual(joint["train_samples_by_stage"], [6, 12])
                self.assertEqual(
                    joint["accuracy_by_task"],
                    [
                        joint["accuracy_matrix_lower_triangular"][index][index]
                        for index in range(2)
                    ],
                )
                cil_matrix = json.loads(
                    Path(search["best_accuracy_matrix_file"]).read_text(encoding="utf-8")
                )
                np.testing.assert_allclose(
                    summary["cil_performance"]["intransigence"],
                    compute_intransigence(cil_matrix, joint),
                )
                self.assertEqual(len(summary["cil_performance"]["intransigence"]), 2)
                self.assertTrue(all(path.is_file() for path in paths))
                self.assertEqual(
                    joint_run_path(joint_root, "uwave", "ewc", 1, 62), joint_json
                )


if __name__ == "__main__":
    unittest.main()
