from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cil_experiments.aggregate_results import aggregate_results
from cil_experiments.experiment_config import experiment_result_root
from cil_experiments.flops import PhaseFlopProfiler
from cil_experiments.intransigence import (
    compute_intransigence,
    find_joint_summary,
    validate_joint_artifact_match,
)
from cil_experiments.metrics import compute_cil_metrics
from cil_experiments.models import SpikeImageAdapter, build_backbone
from cil_experiments.output import AtomicRunArtifacts, completed_summary_path
from cil_experiments.registry import DATASETS, DEFAULT_BACKBONES
from cil_experiments.replay_storage import PackedBinaryExample
from cil_experiments.runner import _train_experience
from cil_experiments.strategies import (
    build_lwf_tuning_strategy,
    build_si_tuning_strategy,
    build_strategy,
)
from jointlearning.runner import TagFexCapacityMatchedJoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summary(final_accuracy: float, intransigence: list[float]) -> dict:
    matrix = np.array([[0.8, np.nan], [0.6, final_accuracy]], dtype=np.float64)
    cil = compute_cil_metrics(matrix, [1, 1])
    cil["intransigence"] = intransigence
    return {
        "method": "ER-ACE",
        "seed": 62,
        "tasks": 2,
        "cil_performance": cil,
        "network": {
            "final_hidden_neurons": 512,
            "total_new_neurons": 0,
            "total_reused_neurons": None,
        },
        "training_runtime": {
            "total_s": 1.0,
            "total_gpu_ms": None,
            "initial_task_s": 0.4,
            "incremental_tasks_total_s": 0.6,
            "mean_incremental_task_s": 0.6,
            "median_incremental_task_s": 0.6,
        },
        "training_operations": {
            "task_summed_terminal_flops_per_sample": 10.0,
            "overall_learning_flops": 30,
            "core_training_flops": 20,
            "learning_auxiliary_flops": 10,
            "single_sample_forward_flops": 4,
            "auxiliary_nonflop_ops": {
                "explicit_nonflop_operator_calls": {},
                "manual_nonflop_operations": [],
                "total_recorded_events": 0,
                "policy": "fixture",
            },
        },
        "persistent_storage": {
            "model_parameter_bytes": 1,
            "replay_sample_bytes": 2,
            "replay_label_bytes": 3,
            "auxiliary_bytes": 4,
            "total_bytes": 10,
            "total_mib": 10 / (2**20),
            "pulse_encoding": "float32_input_image",
            "label_encoding": "uint8_one_hot",
        },
        "inference": {"final_latency_ms_per_sample": 0.1},
        "working_memory_diagnostic": {
            "native_peak_allocated_gpu_memory_mib": None,
            "note": "fixture",
        },
    }


class ResNetJointContractTests(unittest.TestCase):
    def test_experiment_root_and_completed_run_detection(self):
        with tempfile.TemporaryDirectory() as name:
            project_root = Path(name)
            result_root = experiment_result_root(project_root, 7)
            self.assertEqual(result_root, project_root / "result-exp7")
            config = {
                "experiment_id": 7,
                "backbone": {"backbone_id": DEFAULT_BACKBONES["er_ace"]},
                "schedule": (1, 2),
            }
            with AtomicRunArtifacts(
                result_root,
                "uwave",
                "er_ace",
                1,
                62,
                config,
                experiment_id=7,
            ) as artifacts:
                artifacts.commit(
                    _summary(0.8, [0.0, 0.1]),
                    {
                        "dataset": "uwave",
                        "method": "er_ace",
                        "backbone_id": DEFAULT_BACKBONES["er_ace"],
                        "order_id": 1,
                        "seed": 62,
                        "experiment_id": 7,
                    },
                )
                expected = artifacts.summary_path
            self.assertEqual(
                completed_summary_path(
                    result_root,
                    "uwave",
                    "er_ace",
                    1,
                    62,
                    7,
                    expected_config=config,
                ),
                expected,
            )
            with self.assertRaisesRegex(ValueError, "config differs"):
                completed_summary_path(
                    result_root,
                    "uwave",
                    "er_ace",
                    1,
                    62,
                    7,
                    expected_config={"experiment_id": 7, "changed": True},
                )

    def test_resnet18_forward_backward_and_strict_flops(self):
        model = build_backbone("resnet18_cifar", (3, 32, 32))
        sample = torch.randn(2, 3, 32, 32)
        profiler = PhaseFlopProfiler()
        profiler.start()
        profiler.begin_epoch()
        profiler.switch("core_training")
        output = model(sample)
        output.square().mean().backward()
        profiler.add_processed_samples(2)
        profiler.end_epoch()
        result = profiler.stop(strict=True)
        self.assertEqual(tuple(output.shape), (2, 512))
        self.assertEqual(model.feature_dim, 512)
        self.assertEqual(set(result.phase_flops), {"core_training"})
        self.assertGreater(result.total_flops, 0)

    def test_three_registered_widths(self):
        expected = {
            "resnet18_cifar_small": (32, (32, 64, 128, 256), 256),
            "resnet18_cifar": (64, (64, 128, 256, 512), 512),
            "resnet18_cifar_large": (96, (96, 192, 384, 768), 768),
        }
        from cil_experiments.models import BACKBONES

        for backbone_id, contract in expected.items():
            spec = BACKBONES[backbone_id]
            self.assertEqual((spec.base_width, spec.stage_channels, spec.feature_dim), contract)
            output = build_backbone(backbone_id, (3, 32, 32))(
                torch.zeros(2, 3, 32, 32)
            )
            self.assertEqual(tuple(output.shape), (2, contract[-1]))

    def test_three_image_views_and_spike_pack_round_trip(self):
        root = PROJECT_ROOT / "dataset"
        spike_raw = np.load(root / DATASETS["spike"].train_x, mmap_mode="r")[0]
        spike_saved = np.load(root / DATASETS["spike"].image_train_x, mmap_mode="r")[0]
        raw_tensor = torch.from_numpy(np.array(spike_raw, copy=True)).transpose(0, 1)
        runtime = SpikeImageAdapter()(raw_tensor.unsqueeze(0))[0].numpy()
        np.testing.assert_array_equal(runtime, spike_saved.astype(np.float32))
        packed = PackedBinaryExample.from_tensor(raw_tensor, 1, DATASETS["spike"].num_classes)
        torch.testing.assert_close(packed.unpack(), raw_tensor.to(torch.float32))

        texture_raw = np.load(root / DATASETS["texture"].train_x, mmap_mode="r")[0]
        texture_saved = np.load(root / DATASETS["texture"].image_train_x, mmap_mode="r")[0]
        expected = np.pad(texture_raw.reshape(-1).astype(np.float32), (0, 250)).reshape(185, 185)
        np.testing.assert_array_equal(texture_saved[0], expected)
        np.testing.assert_array_equal(texture_saved[1], texture_saved[0])

        uwave_raw = np.load(root / DATASETS["uwave"].train_x, mmap_mode="r")[0]
        uwave_saved = np.load(root / DATASETS["uwave"].image_train_x, mmap_mode="r")[0]
        expected_uwave = np.pad(uwave_raw.reshape(-1), (0, 79)).reshape(32, 32)
        np.testing.assert_array_equal(uwave_saved[:, :, 0], expected_uwave)
        np.testing.assert_array_equal(uwave_saved[:, :, 1], uwave_saved[:, :, 0])

    def test_intransigence_keeps_sign(self):
        cil = {
            "tasks": 2,
            "accuracy_matrix_lower_triangular": [[0.8, None], [0.6, 0.9]],
        }
        joint = {"joint_current_task_accuracy_curve": [0.7, 1.0]}
        values = compute_intransigence(cil, joint)
        self.assertAlmostEqual(values[0], -0.1)
        self.assertAlmostEqual(values[1], 0.1)

    def test_joint_missing_and_backbone_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                find_joint_summary(
                    root,
                    dataset="uwave",
                    paired_method="er_ace",
                    order_id=1,
                    seed=62,
                    experiment_id=7,
                )
            joint_config = {
                "experiment_id": 7,
                "backbone": {"backbone_id": "resnet18_cifar"},
                "protocol": {"input_view_id": "view"},
                "selected_task_groups": [[0, 1], [2, 3]],
                "training": {"paired_training_parameters": {"learning_rate": 0.1}},
            }
            with AtomicRunArtifacts(
                root,
                "uwave",
                "joint_er_ace",
                1,
                62,
                joint_config,
                experiment_id=7,
            ) as artifacts:
                artifacts.commit(
                    _summary(0.9, [0.0, 0.0]),
                    {
                        "dataset": "uwave",
                        "method": "joint_er_ace",
                        "paired_method": "er_ace",
                        "order_id": 1,
                        "seed": 62,
                        "experiment_id": 7,
                        "tasks": 2,
                        "joint_current_task_accuracy_curve": [0.8, 0.9],
                    },
                )
                joint_path = artifacts.summary_path
            cil_identity = {
                "dataset": "uwave",
                "method": "er_ace",
                "order_id": 1,
                "seed": 62,
                "experiment_id": 7,
                "tasks": 2,
            }
            cil_config = {
                "experiment_id": 7,
                "backbone": {"backbone_id": "different_registered_backbone"},
                "protocol": {"input_view_id": "view"},
                "selected_task_groups": [[0, 1], [2, 3]],
                "final_hyperparameters": {"resolved": {"learning_rate": 0.1}},
            }
            with self.assertRaisesRegex(ValueError, "backbone_id"):
                validate_joint_artifact_match(cil_identity, cil_config, joint_path)

    def test_seven_resnet_strategies_execute_one_minibatch(self):
        from avalanche.benchmarks import nc_benchmark
        from avalanche.benchmarks.utils import (
            _make_taskaware_tensor_classification_dataset,
        )

        x = torch.randn(8, 3, 32, 32)
        y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        dataset = _make_taskaware_tensor_classification_dataset(x, y)
        benchmark = nc_benchmark(
            dataset,
            dataset,
            n_experiences=2,
            task_labels=False,
            shuffle=False,
            fixed_class_order=[0, 1, 2, 3],
            per_exp_classes={0: 2, 1: 2},
        )
        device = torch.device("cpu")
        formal = []
        for method in ("er_ace", "ewc", "cwr_star", "icarl", "fecam"):
            overrides = {
                "epochs_per_experience": 1,
                "train_mb_size": 4,
                "eval_mb_size": 4,
            }
            if method in {"er_ace", "icarl"}:
                overrides["memory_size"] = 4
            if method == "er_ace":
                overrides["batch_size_mem"] = 2
            formal.append(
                build_strategy(
                    method,
                    3,
                    1,
                    device,
                    "uwave",
                    enable_flop_accounting=False,
                    resolved_parameters=overrides,
                    model_input_shape=(3, 32, 32),
                )
            )
        formal.extend(
            (
                build_si_tuning_strategy(
                    3,
                    1,
                    device,
                    {"si_lambda": 1e-4, "eps": 1e-7},
                    dataset_name="uwave",
                    model_input_shape=(3, 32, 32),
                ),
                build_lwf_tuning_strategy(
                    3,
                    1,
                    device,
                    {"alpha": 1.0, "temperature": 2.0},
                    dataset_name="uwave",
                    model_input_shape=(3, 32, 32),
                ),
            )
        )
        for bundle in formal:
            bundle.strategy.train(benchmark.train_stream[0], num_workers=0)
            self.assertEqual(bundle.backbone_id, "resnet18_cifar")
            self.assertEqual(bundle.strategy.model.feature_extractor.feature_dim, 512)

    def test_tagfex_joint_capacity_matching_uses_selected_image_backbone(self):
        sample = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 0, 1])
        diagonal = []
        for stage in (1, 2):
            model = TagFexCapacityMatchedJoint(
                "uwave", (3, 32, 32), "resnet18_cifar_small", stage, 2 * stage
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            logits = model(sample)
            loss = torch.nn.functional.cross_entropy(logits[:, :2], labels)
            loss.backward()
            optimizer.step()
            diagonal.append(float((logits[:, :2].argmax(1) == labels).float().mean()))
            self.assertEqual(len(model.branches), stage)
            self.assertEqual(model.feature_dim, 256 * stage)
            self.assertEqual(model.branches[0].feature_dim, 256)
        matrix = {
            "tasks": 2,
            "accuracy_matrix_lower_triangular": [
                [diagonal[0], None],
                [0.0, diagonal[1]],
            ],
        }
        joint = {"joint_current_task_accuracy_curve": diagonal}
        self.assertEqual(compute_intransigence(matrix, joint), [0.0, 0.0])

    def test_tagfex_second_stage_passes_strict_flop_accounting(self):
        from avalanche.benchmarks import nc_benchmark
        from avalanche.benchmarks.utils import (
            _make_taskaware_tensor_classification_dataset,
        )

        x = torch.randn(8, 3, 32, 32)
        y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        dataset = _make_taskaware_tensor_classification_dataset(x, y)
        benchmark = nc_benchmark(
            dataset,
            dataset,
            n_experiences=2,
            task_labels=False,
            shuffle=False,
            fixed_class_order=[0, 1, 2, 3],
            per_exp_classes={0: 2, 1: 2},
        )
        bundle = build_strategy(
            "tagfex",
            3,
            1,
            torch.device("cpu"),
            "uwave",
            resolved_parameters={
                "epochs_per_experience": 1,
                "train_mb_size": 4,
                "eval_mb_size": 4,
                "num_workers": 0,
                "memory_size": 4,
                "proj_hidden_dim": 32,
                "proj_output_dim": 16,
            },
            backbone_id="resnet18_cifar_small",
            model_input_shape=(3, 32, 32),
        )
        results = [
            _train_experience(bundle, experience, torch.device("cpu"))[0]
            for experience in benchmark.train_stream
        ]
        self.assertEqual(bundle.strategy.model.__class__.__name__, "ImageTagFexNet")
        self.assertTrue(all(result.phase_flops["core_training"] > 0 for result in results))
        self.assertGreater(results[1].phase_flops["learning_auxiliary"], 0)

    def test_tagfex_accepts_all_three_backbones(self):
        expected_dim = {
            "resnet18_cifar_small": 256,
            "resnet18_cifar": 512,
            "resnet18_cifar_large": 768,
        }
        for backbone_id, feature_dim in expected_dim.items():
            bundle = build_strategy(
                "tagfex",
                3,
                1,
                torch.device("cpu"),
                "uwave",
                enable_flop_accounting=False,
                backbone_id=backbone_id,
                model_input_shape=(3, 32, 32),
            )
            model = bundle.strategy.model
            model.update_network(2)
            model.eval()
            output = model(torch.zeros(2, 3, 32, 32))
            self.assertEqual(tuple(output["ts_features"][0].shape), (2, feature_dim))

    def test_mini_dataset_manifest_preserves_every_class(self):
        manifest = json.loads(
            (PROJECT_ROOT / "dataset_mini" / "mini_dataset_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected_totals = {
            "spike": (1601, 399),
            "texture": (481, 119),
            "uwave": (1790, 449),
        }
        for dataset, (train_total, test_total) in expected_totals.items():
            record = manifest["datasets"][dataset]
            self.assertEqual(record["train"]["mini_samples"], train_total)
            self.assertEqual(record["test"]["mini_samples"], test_total)
            self.assertEqual(len(record["train"]["class_counts"]), DATASETS[dataset].num_classes)
            self.assertTrue(all(value >= 1 for value in record["test"]["class_counts"].values()))

    def test_aggregate_two_orders_by_two_seeds(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for order_id in (1, 2):
                for seed in (62, 63):
                    summary = _summary(0.8 + 0.01 * (seed - 62), [0.0, 0.1])
                    summary["seed"] = seed
                    config = {
                        "experiment_id": 7,
                        "backbone": {"backbone_id": DEFAULT_BACKBONES["er_ace"]}
                    }
                    with AtomicRunArtifacts(
                        root,
                        "uwave",
                        "er_ace",
                        order_id,
                        seed,
                        config,
                        experiment_id=7,
                    ) as artifacts:
                        matrix = {
                            "dataset": "uwave",
                            "method": "er_ace",
                            "backbone_id": DEFAULT_BACKBONES["er_ace"],
                            "order_id": order_id,
                            "seed": seed,
                            "experiment_id": 7,
                        }
                        artifacts.commit(summary, matrix)
            result = aggregate_results(
                root,
                datasets=("uwave",),
                methods=("er_ace",),
                order_ids=(1, 2),
                seeds=(62, 63),
                experiment_id=7,
            )
            self.assertEqual(result["experiment_id"], 7)
            self.assertEqual(len(result["groups"]), 1)
            self.assertEqual(result["groups"][0]["n"], 4)
            self.assertEqual(len(result["groups"][0]["intransigence_by_stage"]), 2)
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                aggregate_results(
                    root,
                    datasets=("uwave",),
                    methods=("er_ace",),
                    order_ids=(1, 2),
                    seeds=(62, 63, 64),
                    experiment_id=7,
                )
            with self.assertRaises(FileExistsError):
                with AtomicRunArtifacts(
                    root,
                    "uwave",
                    "er_ace",
                    1,
                    62,
                    {
                        "experiment_id": 7,
                        "backbone": {"backbone_id": DEFAULT_BACKBONES["er_ace"]},
                    },
                    experiment_id=7,
                ):
                    pass


if __name__ == "__main__":
    unittest.main()
