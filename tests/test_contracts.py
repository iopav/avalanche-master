from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cil_experiments.data import validate_source_files
from cil_experiments.flops import adaptive_pool_flop, add_like_flop, profile_single_forward
from cil_experiments.final_hyperparameters import (
    FINAL_HYPERPARAMETERS,
    get_final_hyperparameters,
    validate_final_hyperparameter_registry,
)
from cil_experiments.metrics import compute_cil_metrics
from cil_experiments.models import TemporalBackbone, assert_shared_backbone_contract
from cil_experiments.output import AtomicRunArtifacts
from cil_experiments.registry import DATASETS, METHODS, ORDERS, SEEDS, validate_order_seed_file
from cil_experiments.strategies import (
    build_lwf_tuning_strategy,
    build_si_tuning_strategy,
    build_strategy,
)


PROJECT_ROOT = Path(r"D:\workspace\Avalanche\avalanche-master")


class ProtocolContractTests(unittest.TestCase):
    def test_order_seed_registry_matches_source(self):
        validate_order_seed_file(PROJECT_ROOT.parent / "5order10seeds.txt")
        self.assertEqual(tuple(SEEDS), tuple(range(62, 72)))
        self.assertEqual(len(ORDERS), 5)

    def test_source_array_contracts_and_spike_400(self):
        for spec in DATASETS.values():
            validate_source_files(PROJECT_ROOT / "dataset", spec)
        self.assertEqual(DATASETS["spike"].timesteps, 400)
        self.assertEqual(DATASETS["spike"].train_shape[1], 400)

    def test_backbones_differ_only_at_first_input_channels(self):
        models = {name: TemporalBackbone(spec.in_channels) for name, spec in DATASETS.items()}
        assert_shared_backbone_contract()
        for name, spec in DATASETS.items():
            output = models[name](torch.zeros(2, spec.in_channels, spec.timesteps))
            self.assertEqual(tuple(output.shape), (2, 64))

    def test_metric_formulas_generalize_to_task_count(self):
        matrix = np.array([[0.8, np.nan, np.nan], [0.7, 0.9, np.nan], [0.6, 0.8, 1.0]])
        result = compute_cil_metrics(matrix, [10, 20, 30])
        self.assertEqual(len(result["task_end_seen_accuracy_curve"]), 3)
        self.assertEqual(len(result["forgetting_per_task"]), 2)
        self.assertAlmostEqual(result["backward_transfer"], -result["mean_final_accuracy_loss"])
        self.assertAlmostEqual(result["final_average_accuracy"], (0.6 * 10 + 0.8 * 20 + 1.0 * 30) / 60)

    def test_flop_counter_counts_actual_forward(self):
        model = torch.nn.Linear(8, 4, bias=True).eval()
        total, detail = profile_single_forward(model, torch.zeros(1, 8))
        self.assertEqual(total, 2 * 1 * 4 * 8 + 4)
        self.assertEqual(detail["flops"], total)

    def test_locked_affine_and_adaptive_average_pool_formulas(self):
        self.assertEqual(add_like_flop((10,), out_shape=(10,)), 20)
        self.assertEqual(adaptive_pool_flop((1, 1, 4), out_shape=(1, 1, 1)), 4)

    def test_registry_uses_er_ace_with_isolated_buffer(self):
        self.assertEqual(set(METHODS), {"er_ace", "ewc", "cwr_star", "icarl", "fecam"})
        first = build_strategy("er_ace", 3, 1, torch.device("cpu"), "uwave")
        second = build_strategy("er_ace", 3, 1, torch.device("cpu"), "uwave")
        self.assertIsNot(first.strategy.storage_policy, second.strategy.storage_policy)
        self.assertEqual(first.strategy.mem_size, 200)

    def test_si_is_available_only_through_the_manual_candidate_builder(self):
        self.assertNotIn("si", METHODS)
        bundle = build_si_tuning_strategy(
            3,
            1,
            torch.device("cpu"),
            {"si_lambda": 0.0001, "eps": 0.0000001},
        )
        self.assertEqual(bundle.method, "si")
        self.assertIsNone(bundle.phase_plugin)
        self.assertEqual(bundle.method_plugin.si_lambda, [0.0001])

    def test_lwf_is_available_only_through_the_manual_candidate_builder(self):
        self.assertNotIn("lwf", METHODS)
        bundle = build_lwf_tuning_strategy(
            3,
            1,
            torch.device("cpu"),
            {"alpha": 1.0, "temperature": 2.0},
        )
        self.assertEqual(bundle.method, "lwf")
        self.assertIsNone(bundle.phase_plugin)
        self.assertEqual(bundle.method_plugin.lwf.alpha, 1.0)
        self.assertEqual(bundle.method_plugin.lwf.temperature, 2.0)

    def test_final_hyperparameters_cover_every_dataset_method(self):
        validate_final_hyperparameter_registry()
        self.assertEqual(set(FINAL_HYPERPARAMETERS), set(DATASETS))
        for dataset in DATASETS:
            self.assertEqual(set(FINAL_HYPERPARAMETERS[dataset]), set(METHODS))
            for method in METHODS:
                first = get_final_hyperparameters(dataset, method)
                second = get_final_hyperparameters(dataset, method)
                self.assertIsNot(first, second)
                first["learning_rate"] = 999.0
                self.assertNotEqual(first["learning_rate"], second["learning_rate"])

    def test_strategy_uses_resolved_dataset_method_hyperparameters(self):
        parameters = get_final_hyperparameters(
            "uwave",
            "ewc",
            {
                "learning_rate": 0.0123,
                "train_mb_size": 7,
                "eval_mb_size": 11,
                "ewc_lambda": 0.7,
            },
        )
        bundle = build_strategy(
            "ewc",
            3,
            int(parameters["epochs_per_experience"]),
            torch.device("cpu"),
            "uwave",
            enable_flop_accounting=False,
            resolved_parameters=parameters,
        )
        self.assertEqual(bundle.strategy.optimizer.param_groups[0]["lr"], 0.0123)
        self.assertEqual(bundle.strategy.train_mb_size, 7)
        self.assertEqual(bundle.strategy.eval_mb_size, 11)
        self.assertEqual(bundle.method_plugin.ewc_lambda, 0.7)

    def test_fecam_trains_first_experience_then_freezes_and_disables_sgd(self):
        bundle = build_strategy("fecam", 3, 3, torch.device("cpu"), "uwave")
        plugin = bundle.method_plugin
        strategy = bundle.strategy
        strategy.clock.train_exp_counter = 0
        plugin.before_training_exp(strategy)
        self.assertEqual(strategy.train_epochs, 3)
        self.assertTrue(all(p.requires_grad for p in strategy.model.feature_extractor.parameters()))
        plugin._freeze_feature_extractor(strategy)
        strategy.clock.train_exp_counter = 1
        plugin.before_training_exp(strategy)
        self.assertEqual(strategy.train_epochs, 0)
        self.assertTrue(all(not p.requires_grad for p in strategy.model.feature_extractor.parameters()))
        self.assertIsInstance(strategy.model.train_classifier, torch.nn.Linear)
        self.assertEqual(strategy.model.train_classifier.out_features, 3)

    def test_manual_tuning_has_one_flop_free_entry_per_dataset_method(self):
        manual_root = PROJECT_ROOT / "manual_tuning"
        formal_entries = {
            f"{dataset}__{method}.py" for dataset in DATASETS for method in METHODS
        }
        candidate_entries = {
            f"{dataset}__{method}.py"
            for dataset in DATASETS
            for method in ("si", "lwf")
        }
        actual = {path.name for path in manual_root.glob("*__*.py")}
        self.assertEqual(actual, formal_entries | candidate_entries)
        for path in manual_root.glob("*__*.py"):
            script = path.read_text(encoding="utf-8")
            tree = ast.parse(script)
            device_values = []
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if not node.targets or not isinstance(node.targets[0], ast.Tuple):
                    continue
                names = [item.id for item in node.targets[0].elts if isinstance(item, ast.Name)]
                if "DEVICE" in names and isinstance(node.value, ast.Tuple):
                    device_values.append(node.value.elts[names.index("DEVICE")].value)
            self.assertEqual(device_values, ["cuda"], path.name)
            self.assertIn("#", script)
        common = (manual_root / "common.py").read_text(encoding="utf-8")
        self.assertIn("bundle.strategy.train", common)
        self.assertIn("enable_flop_accounting=False", common)
        self.assertNotIn("from cil_experiments.flops", common)
        self.assertNotIn("from cil_experiments.runner", common)
        self.assertNotIn("compute_persistent_storage", common)
        self.assertNotIn("profile_single_forward", common)

    def test_cross_volume_transaction_and_failure_cleanup(self):
        workspace_temp = Path(r"D:\workspace\PyCIL")
        with tempfile.TemporaryDirectory(prefix="cil-atomic-test-", dir=workspace_temp) as name:
            root = Path(name) / "result"
            config = {"x": 1}
            timestamp = "20260821T120000+0800"
            matrix_payload = {"accuracy_matrix_lower_triangular": [[0.5, None], [0.4, 0.6]]}
            with AtomicRunArtifacts(root, "d", "m", 1, 62, config, timestamp=timestamp) as artifacts:
                artifacts.logger.info("complete")
                artifacts.commit({"ok": True}, matrix_payload)
            stem = f"d__m__order-01__seed-062__timestamp-{timestamp}"
            config_path = root / "d" / "m" / f"{stem}__config.json"
            self.assertTrue(config_path.is_file())
            self.assertTrue((root / "d" / "m" / "log" / f"{stem}.log").is_file())
            summary = root / "d" / "m" / "summary" / f"{stem}__summary.json"
            matrix = root / "d" / "m" / "summary" / f"{stem}__accuracy-matrix.json"
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(json.loads(matrix.read_text(encoding="utf-8")), matrix_payload)
            with self.assertRaises(FileExistsError):
                with AtomicRunArtifacts(root, "d", "m", 1, 62, config, timestamp=timestamp):
                    pass
            with AtomicRunArtifacts(
                root, "d", "m", 1, 62, {"x": 2}, overwrite=True, timestamp=timestamp
            ) as artifacts:
                artifacts.logger.info("replacement")
                artifacts.commit({"ok": "replacement"}, matrix_payload)
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"ok": "replacement"})
            replaced_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(replaced_config["x"], 2)

            failed_root = Path(name) / "failed"
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with AtomicRunArtifacts(failed_root, "d", "m", 1, 62, config, timestamp=timestamp):
                    raise RuntimeError("intentional")
            self.assertFalse(any(failed_root.rglob("*")) if failed_root.exists() else False)


if __name__ == "__main__":
    unittest.main()
