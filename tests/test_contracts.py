from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cil_experiments.data import validate_source_files
from cil_experiments.flops import (
    adaptive_pool_backward_flop,
    adaptive_pool_flop,
    add_like_flop,
    convolution_backward_flop,
    ExperienceFlopResult,
    mse_loss_flop,
    PhaseFlopProfiler,
    profile_single_forward,
    summarize_auxiliary_nonflop_ops,
    summarize_learning_flops,
    variance_flop,
)
from cil_experiments.final_hyperparameters import (
    FINAL_HYPERPARAMETERS,
    get_final_hyperparameters,
    validate_final_hyperparameter_registry,
)
from cil_experiments.metrics import compute_cil_metrics
from cil_experiments.models import TemporalBackbone, assert_shared_backbone_contract
from cil_experiments.output import AtomicRunArtifacts
from cil_experiments.registry import (
    DATASETS,
    METHODS,
    ORDER_IDS,
    ORDERS_BY_DATASET,
    SEEDS,
    get_task_split,
    project_order,
    validate_order_registry,
)
from cil_experiments.strategies import (
    build_ewc_cosine_tuning_strategy,
    build_lwf_tuning_strategy,
    build_mas_tuning_strategy,
    build_si_tuning_strategy,
    build_strategy,
)


PROJECT_ROOT = Path(r"D:\workspace\Avalanche\avalanche-master")


class ProtocolContractTests(unittest.TestCase):
    def test_order_seed_registry_matches_source(self):
        validate_order_registry()
        self.assertEqual(tuple(SEEDS), tuple(range(62, 72)))
        self.assertEqual(ORDER_IDS, (1, 2, 3, 4, 5))
        self.assertEqual(sum(len(orders) for orders in ORDERS_BY_DATASET.values()), 15)
        for dataset_name, spec in DATASETS.items():
            for order_id in ORDER_IDS:
                self.assertEqual(
                    sorted(project_order(dataset_name, order_id)),
                    list(range(spec.num_classes)),
                )
                self.assertEqual(sum(get_task_split(dataset_name, order_id)), spec.num_classes)

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
        self.assertEqual(add_like_flop((10,), out_shape=(10,)), 10)
        self.assertEqual(add_like_flop((10,), out_shape=(10,), alpha=-0.1), 20)
        self.assertEqual(adaptive_pool_flop((1, 1, 4), out_shape=(1, 1, 1)), 4)
        self.assertEqual(adaptive_pool_backward_flop((1, 1, 1), (1, 1, 4)), 4)

    def test_variance_and_mse_have_explicit_arithmetic_formulas(self):
        self.assertEqual(variance_flop((2, 4), out_shape=()), 32)
        self.assertEqual(mse_loss_flop((2, 4), (2, 4), 0), 16)
        self.assertEqual(mse_loss_flop((2, 4), (2, 4), 1), 24)

    def test_convolution_backward_counts_bias_reduction(self):
        # Conv1d: input [1,2,4], weight [3,2,1], output [1,3,4].
        # grad-input=48, grad-weight=48, grad-bias=3*(4-1)=9 FLOPs.
        total = convolution_backward_flop(
            (1, 3, 4),
            (1, 2, 4),
            (3, 2, 1),
            (3,),
            (1,),
            (0,),
            (1,),
            False,
            (0,),
            1,
            (True, True, True),
            out_shape=((1, 2, 4), (3, 2, 1), (3,)),
        )
        self.assertEqual(total, 105)

    def test_phase_profiler_separates_conv_forward_loss_and_backward(self):
        model = torch.nn.Conv1d(2, 3, kernel_size=1, bias=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        sample = torch.ones(1, 2, 4, requires_grad=True)
        profiler = PhaseFlopProfiler()
        profiler.start()
        profiler.begin_epoch()
        profiler.switch("student_current_forward")
        output = model(sample)
        profiler.switch("loss_and_regularization")
        loss = output.sum()
        profiler.switch("backward")
        loss.backward()
        profiler.switch("optimizer_update")
        optimizer.step()
        profiler.add_processed_samples(1)
        profiler.end_epoch()
        result = profiler.stop(strict=True)
        self.assertEqual(result.phase_flops["student_current_forward"], 60)
        self.assertEqual(result.phase_flops["loss_and_regularization"], 11)
        self.assertEqual(result.phase_flops["backward"], 105)
        self.assertEqual(result.phase_flops["optimizer_update"], 18)
        self.assertEqual(result.total_flops, 194)

    def test_learning_flop_summary_is_an_exact_phase_partition(self):
        result = ExperienceFlopResult(
            total_flops=204,
            phase_flops={
                "student_current_forward": 60,
                "loss_and_regularization": 11,
                "backward": 105,
                "optimizer_update": 18,
                "replay_forward": 7,
                "method_specific": 3,
            },
            terminal_epoch_flops=204,
            terminal_epoch_samples=1,
            terminal_flops_per_sample=204.0,
            operation_calls={},
            zero_flop_operations={},
            custom_formulas=[],
            manual_supplementary_rules=[],
            manual_nonflop_operations=[
                {
                    "name": "numpy.packbits",
                    "calls": 1,
                    "reason": "integer bit packing",
                    "variables": {"samples": 2},
                }
            ],
        )
        summary = summarize_learning_flops([result], 100, 60)
        self.assertEqual(summary["core_training_flops"], 194)
        self.assertEqual(summary["learning_auxiliary_flops"], 10)
        self.assertEqual(summary["hyperparameter_search_flops"], 100)
        self.assertEqual(summary["overall_learning_flops"], 304)
        self.assertEqual(summary["single_sample_forward_flops"], 60)
        result.zero_flop_operations = {"aten.reshape": 4, "aten.copy_": 2}
        nonflop = summarize_auxiliary_nonflop_ops([result])
        self.assertEqual(nonflop["explicit_nonflop_operator_calls"]["aten.reshape"], 4)
        self.assertEqual(nonflop["manual_nonflop_operations"][0]["name"], "numpy.packbits")
        self.assertEqual(nonflop["total_recorded_events"], 7)

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

    def test_ewc_cosine_is_manual_only_and_keeps_classifier_parameter_names(self):
        from types import SimpleNamespace

        self.assertNotIn("ewc_cosine", METHODS)
        bundle = build_ewc_cosine_tuning_strategy(
            3,
            1,
            torch.device("cpu"),
            {"ewc_lambda": 1.0, "mode": "separate"},
        )
        classifier = bundle.strategy.model.classifier
        classifier.adaptation(SimpleNamespace(classes_in_this_experience=[0, 1, 2]))
        first_names = set(dict(bundle.strategy.model.named_parameters()))
        first_output = bundle.strategy.model(torch.zeros(2, 3, 315))
        classifier.adaptation(SimpleNamespace(classes_in_this_experience=[3]))
        second_names = set(dict(bundle.strategy.model.named_parameters()))
        second_output = bundle.strategy.model(torch.zeros(2, 3, 315))
        self.assertEqual(bundle.method, "ewc_cosine")
        self.assertIsNone(bundle.phase_plugin)
        self.assertEqual(bundle.method_plugin.ewc_lambda, 1.0)
        self.assertEqual(tuple(first_output.shape), (2, 3))
        self.assertEqual(tuple(second_output.shape), (2, 4))
        self.assertIn("classifier.fc.weight", first_names)
        self.assertIn("classifier.fc.weight", second_names)
        self.assertNotIn("classifier.fc.fc1.weight", second_names)

    def test_mas_is_available_only_through_the_manual_candidate_builder(self):
        self.assertNotIn("mas", METHODS)
        bundle = build_mas_tuning_strategy(
            3,
            1,
            torch.device("cpu"),
            {"lambda_reg": 1.0, "alpha": 0.5},
        )
        self.assertEqual(bundle.method, "mas")
        self.assertIsNone(bundle.phase_plugin)
        self.assertEqual(bundle.method_plugin._lambda, 1.0)
        self.assertEqual(bundle.method_plugin.alpha, 0.5)

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
        candidate_entries.add("spike__ewc_cosine.py")
        candidate_entries.add("spike__mas.py")
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
