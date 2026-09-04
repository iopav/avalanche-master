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
    log_softmax_flop,
    logsumexp_flop,
    PhaseFlopProfiler,
    profile_single_forward,
    summarize_auxiliary_nonflop_ops,
    summarize_learning_flops,
    softmax_backward_flop,
    softmax_flop,
    variance_flop,
    xlogy_flop,
)
from cil_experiments.final_hyperparameters import (
    FINAL_HYPERPARAMETERS,
    get_final_hyperparameters,
    validate_final_hyperparameter_registry,
)
from cil_experiments.metrics import compute_cil_metrics
from cil_experiments.models import build_backbone, assert_shared_backbone_contract
from cil_experiments.output import AtomicRunArtifacts
from cil_experiments.replay_storage import (
    Float32ClassBalancedBuffer,
    Float32ReplayExample,
    PackedBinaryExample,
    PackedClassBalancedBuffer,
    validate_uint8_one_hot,
)
from cil_experiments.storage import compute_persistent_storage
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
    def test_spike_replay_uses_real_bit_packing_and_uint8_one_hot_labels(self):
        sample = torch.tensor(
            [[[0, 1, 0, 1], [1, 0, 1, 0]]], dtype=torch.float32
        )
        example = PackedBinaryExample.from_tensor(sample, label=2, num_classes=4)
        self.assertEqual(example.data.dtype, np.uint8)
        self.assertEqual(example.data.nbytes, 1)
        self.assertEqual(example.label_one_hot.dtype, np.uint8)
        self.assertEqual(example.label_one_hot.nbytes, 4)
        self.assertEqual(example.label, 2)
        torch.testing.assert_close(example.unpack(), sample)

        validate_uint8_one_hot(np.eye(4, dtype=np.uint8), num_classes=4)
        with self.assertRaises(AssertionError):
            validate_uint8_one_hot(np.asarray([[1, 1, 0, 0]], dtype=np.uint8), 4)

    def test_all_spike_replay_methods_select_packed_persistence(self):
        erace = build_strategy(
            "er_ace", 64, 1, torch.device("cpu"), "spike", enable_flop_accounting=False
        )
        icarl = build_strategy(
            "icarl", 64, 1, torch.device("cpu"), "spike", enable_flop_accounting=False
        )
        tagfex = build_strategy(
            "tagfex", 64, 1, torch.device("cpu"), "spike", enable_flop_accounting=False
        )
        self.assertIsInstance(erace.strategy.storage_policy, PackedClassBalancedBuffer)
        self.assertTrue(icarl.method_plugin.pack_binary)
        self.assertEqual(icarl.method_plugin.num_classes, 20)
        self.assertTrue(tagfex.strategy.pack_binary_replay)
        self.assertEqual(tagfex.strategy.num_classes, 20)

        sample = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.float32)

        class _TinyDataset(torch.utils.data.Dataset):
            targets = [2, 2]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return sample.clone(), self.targets[index], 0

        erace.strategy.storage_policy.post_adapt(None, type("Exp", (), {"dataset": _TinyDataset()})())
        erace_examples = erace.strategy.storage_policy.persistent_examples()
        self.assertEqual(len(erace_examples), 2)
        self.assertTrue(all(isinstance(value, PackedBinaryExample) for value in erace_examples))
        erace_storage = compute_persistent_storage(erace)
        self.assertEqual(erace_storage["replay_sample_bytes"], 2)
        self.assertEqual(erace_storage["replay_label_bytes"], 40)

        icarl.method_plugin.x_memory = [torch.stack((sample, sample))]
        icarl.method_plugin.y_memory = [np.asarray([1, 2], dtype=np.int64)]
        icarl.method_plugin._pack_current_memory()
        self.assertEqual(icarl.method_plugin.x_memory, [])
        self.assertEqual(icarl.method_plugin.y_memory, [])
        self.assertEqual(icarl.method_plugin.persistent_labels[0].dtype, np.uint8)
        self.assertEqual(icarl.method_plugin.persistent_labels[0].shape, (2, 20))
        icarl_storage = compute_persistent_storage(icarl)
        self.assertEqual(icarl_storage["replay_sample_bytes"], 2)
        self.assertEqual(icarl_storage["replay_label_bytes"], 40)

        tagfex.strategy.memory_by_class = {2: [(sample.clone(), 2)]}
        tagfex.strategy._persist_memory()
        self.assertIsInstance(tagfex.strategy.memory_by_class[2][0], PackedBinaryExample)
        tagfex_storage = compute_persistent_storage(tagfex)
        self.assertEqual(tagfex_storage["replay_sample_bytes"], 1)
        self.assertEqual(tagfex_storage["replay_label_bytes"], 20)
        for storage in (erace_storage, icarl_storage, tagfex_storage):
            self.assertEqual(storage["pulse_encoding"], "1-bit packed")
            self.assertEqual(storage["label_encoding"], "uint8_one_hot")

    def test_all_float_replay_methods_persist_uint8_one_hot_labels(self):
        erace = build_strategy(
            "er_ace", 9, 1, torch.device("cpu"), "texture", enable_flop_accounting=False
        )
        icarl = build_strategy(
            "icarl", 9, 1, torch.device("cpu"), "texture", enable_flop_accounting=False
        )
        tagfex = build_strategy(
            "tagfex", 9, 1, torch.device("cpu"), "texture", enable_flop_accounting=False
        )
        self.assertIsInstance(erace.strategy.storage_policy, Float32ClassBalancedBuffer)

        sample = torch.arange(8, dtype=torch.float64).reshape(2, 4)

        class _TinyDataset(torch.utils.data.Dataset):
            targets = [2, 2]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return sample.clone(), self.targets[index], 0

        erace.strategy.storage_policy.post_adapt(None, type("Exp", (), {"dataset": _TinyDataset()})())
        erace_examples = erace.strategy.storage_policy.persistent_examples()
        self.assertTrue(all(isinstance(value, Float32ReplayExample) for value in erace_examples))
        erace_storage = compute_persistent_storage(erace)
        self.assertEqual(erace_storage["replay_sample_bytes"], 64)
        self.assertEqual(erace_storage["replay_label_bytes"], 24)

        icarl.method_plugin.x_memory = [torch.stack((sample, sample))]
        icarl.method_plugin.y_memory = [np.asarray([1, 2], dtype=np.int64)]
        icarl.method_plugin._pack_current_memory()
        self.assertEqual(icarl.method_plugin.y_memory, [])
        self.assertEqual(icarl.method_plugin.persistent_labels[0].shape, (2, 12))
        icarl_storage = compute_persistent_storage(icarl)
        self.assertEqual(icarl_storage["replay_sample_bytes"], 64)
        self.assertEqual(icarl_storage["replay_label_bytes"], 24)

        tagfex.strategy.memory_by_class = {2: [(sample.clone(), 2)]}
        tagfex.strategy._persist_memory()
        self.assertIsInstance(tagfex.strategy.memory_by_class[2][0], Float32ReplayExample)
        tagfex_storage = compute_persistent_storage(tagfex)
        self.assertEqual(tagfex_storage["replay_sample_bytes"], 32)
        self.assertEqual(tagfex_storage["replay_label_bytes"], 12)
        for storage in (erace_storage, icarl_storage, tagfex_storage):
            self.assertEqual(storage["label_encoding"], "uint8_one_hot")
        self.assertEqual(erace_storage["pulse_encoding"], "float32_input_image")
        self.assertEqual(icarl_storage["pulse_encoding"], "float32_input_image")
        self.assertEqual(tagfex_storage["pulse_encoding"], "float32_input_image")

    def test_order_seed_registry_matches_source(self):
        validate_order_registry()
        self.assertEqual(tuple(SEEDS), tuple(range(62, 72)))
        self.assertEqual(ORDER_IDS, (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(sum(len(orders) for orders in ORDERS_BY_DATASET.values()), 21)
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

    def test_registered_resnet18_contract(self):
        assert_shared_backbone_contract()
        model = build_backbone("resnet18_cifar", (3, 32, 32))
        output = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 512))

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

    def test_single_forward_profiles_eval_behavior_and_restores_model_mode(self):
        class _ModeProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 4)
                self.forward_training_mode = None

            def forward(self, value):
                self.forward_training_mode = self.training
                return self.linear(value)

        model = _ModeProbe().train()
        profile_single_forward(model, torch.zeros(1, 8))
        self.assertFalse(model.forward_training_mode)
        self.assertTrue(model.training)

    def test_locked_affine_and_adaptive_average_pool_formulas(self):
        self.assertEqual(add_like_flop((10,), out_shape=(10,)), 10)
        self.assertEqual(add_like_flop((10,), out_shape=(10,), alpha=-0.1), 20)
        self.assertEqual(adaptive_pool_flop((1, 1, 4), out_shape=(1, 1, 1)), 4)
        self.assertEqual(adaptive_pool_backward_flop((1, 1, 1), (1, 1, 4)), 4)

    def test_variance_and_mse_have_explicit_arithmetic_formulas(self):
        self.assertEqual(variance_flop((2, 4), out_shape=()), 32)
        self.assertEqual(mse_loss_flop((2, 4), (2, 4), 0), 16)
        self.assertEqual(mse_loss_flop((2, 4), (2, 4), 1), 24)
        self.assertEqual(logsumexp_flop((2, 4), [-1], out_shape=(2,)), 26)
        self.assertEqual(xlogy_flop((2, 4), (2, 4), out_shape=(2, 4)), 16)

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

    def test_softmax_formulas_remove_one_overcount_per_reduction_group(self):
        shape = (2, 3, 4)
        self.assertEqual(softmax_flop(shape, 1), 4 * 24 - 8)
        self.assertEqual(log_softmax_flop(shape, 1), 4 * 24)
        self.assertEqual(softmax_backward_flop(shape, shape, 1), 4 * 24 - 8)

    def test_phase_profiler_keeps_the_complete_optimizer_iteration_in_core(self):
        model = torch.nn.Conv1d(2, 3, kernel_size=1, bias=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        sample = torch.ones(1, 2, 4, requires_grad=True)
        profiler = PhaseFlopProfiler()
        profiler.start()
        profiler.begin_epoch()
        profiler.switch("core_training")
        output = model(sample)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        profiler.add_processed_samples(1)
        profiler.end_epoch()
        result = profiler.stop(strict=True)
        self.assertEqual(set(result.phase_flops), {"core_training"})
        self.assertEqual(result.phase_flops["core_training"], 194)
        self.assertEqual(result.total_flops, 194)

    def test_learning_flop_summary_is_an_exact_phase_partition(self):
        result = ExperienceFlopResult(
            total_flops=204,
            phase_flops={
                "core_training": 194,
                "learning_auxiliary": 10,
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
        summary = summarize_learning_flops([result], 60)
        self.assertEqual(summary["core_training_flops"], 194)
        self.assertEqual(summary["learning_auxiliary_flops"], 10)
        self.assertEqual(summary["overall_learning_flops"], 204)
        self.assertEqual(summary["single_sample_forward_flops"], 60)
        result.zero_flop_operations = {"aten.reshape": 4, "aten.copy_": 2}
        nonflop = summarize_auxiliary_nonflop_ops([result])
        self.assertEqual(nonflop["explicit_nonflop_operator_calls"]["aten.reshape"], 4)
        self.assertEqual(nonflop["manual_nonflop_operations"][0]["name"], "numpy.packbits")
        self.assertEqual(nonflop["total_recorded_events"], 7)

    def test_registry_uses_er_ace_with_isolated_buffer(self):
        self.assertEqual(
            set(METHODS), {"er_ace", "ewc", "cwr_star", "icarl", "fecam", "tagfex"}
        )
        first = build_strategy("er_ace", 3, 1, torch.device("cpu"), "uwave")
        second = build_strategy("er_ace", 3, 1, torch.device("cpu"), "uwave")
        self.assertIsNot(first.strategy.storage_policy, second.strategy.storage_policy)
        self.assertEqual(first.strategy.mem_size, 200)

    def test_tagfex_is_a_registered_source_structure_image_strategy(self):
        bundle = build_strategy(
            "tagfex", 3, 1, torch.device("cpu"), "uwave", enable_flop_accounting=False
        )
        bundle.strategy.model.update_network(3)
        bundle.strategy.model.eval()
        output = bundle.strategy.model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(bundle.method, "tagfex")
        self.assertEqual(tuple(output["logits"].shape), (2, 3))
        self.assertEqual(tuple(output["ts_features"][0].shape), (2, 512))
        self.assertEqual(
            bundle.network_summary(6),
            {
                "final_hidden_neurons": 512,
                "total_new_neurons": 0,
                "total_reused_neurons": 0,
            },
        )

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
        first_output = bundle.strategy.model(torch.zeros(2, 3, 32, 32))
        classifier.adaptation(SimpleNamespace(classes_in_this_experience=[3]))
        second_names = set(dict(bundle.strategy.model.named_parameters()))
        second_output = bundle.strategy.model(torch.zeros(2, 3, 32, 32))
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
        self.assertEqual(strategy.model.train_classifier.__class__.__name__, "IncrementalClassifier")
        self.assertEqual(strategy.model.train_classifier.classifier.out_features, 1)

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
            experiment_id = 7
            matrix_payload = {"accuracy_matrix_lower_triangular": [[0.5, None], [0.4, 0.6]]}
            with AtomicRunArtifacts(root, "d", "m", 1, 62, config, experiment_id=experiment_id) as artifacts:
                artifacts.logger.info("complete")
                artifacts.commit({"ok": True}, matrix_payload)
            stem = f"d__m__order-01__seed-062__exp-{experiment_id}"
            config_path = root / "d" / "m" / f"{stem}__config.json"
            self.assertTrue(config_path.is_file())
            self.assertTrue((root / "d" / "m" / "log" / f"{stem}.log").is_file())
            summary = root / "d" / "m" / "summary" / f"{stem}__summary.json"
            matrix = root / "d" / "m" / "summary" / f"{stem}__accuracy-matrix.json"
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(json.loads(matrix.read_text(encoding="utf-8")), matrix_payload)
            with self.assertRaises(FileExistsError):
                with AtomicRunArtifacts(root, "d", "m", 1, 62, config, experiment_id=experiment_id):
                    pass
            with AtomicRunArtifacts(
                root, "d", "m", 1, 62, {"x": 2}, overwrite=True, experiment_id=experiment_id
            ) as artifacts:
                artifacts.logger.info("replacement")
                artifacts.commit({"ok": "replacement"}, matrix_payload)
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"ok": "replacement"})
            replaced_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(replaced_config["x"], 2)

            failed_root = Path(name) / "failed"
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with AtomicRunArtifacts(failed_root, "d", "m", 1, 62, config, experiment_id=experiment_id):
                    raise RuntimeError("intentional")
            self.assertFalse(any(failed_root.rglob("*")) if failed_root.exists() else False)


if __name__ == "__main__":
    unittest.main()
