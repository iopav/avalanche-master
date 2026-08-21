from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cil_experiments.data import validate_source_files
from cil_experiments.flops import profile_single_forward
from cil_experiments.metrics import compute_cil_metrics
from cil_experiments.models import TemporalBackbone, assert_shared_backbone_contract
from cil_experiments.output import AtomicRunArtifacts
from cil_experiments.registry import DATASETS, ORDERS, SEEDS, validate_order_seed_file


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

    def test_cross_volume_transaction_and_failure_cleanup(self):
        workspace_temp = Path(r"D:\workspace\PyCIL")
        with tempfile.TemporaryDirectory(prefix="cil-atomic-test-", dir=workspace_temp) as name:
            root = Path(name) / "result"
            config = {"x": 1}
            with AtomicRunArtifacts(root, "d", "m", 1, 62, config) as artifacts:
                artifacts.logger.info("complete")
                artifacts.commit({"ok": True})
            self.assertTrue((root / "d" / "m" / "config.json").is_file())
            self.assertTrue((root / "d" / "m" / "log" / "d__m__order-01__seed-062.log").is_file())
            summary = root / "d" / "m" / "summary" / "d__m__order-01__seed-062__summary.json"
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"ok": True})
            with self.assertRaises(FileExistsError):
                with AtomicRunArtifacts(root, "d", "m", 1, 62, config):
                    pass
            with AtomicRunArtifacts(root, "d", "m", 1, 62, {"x": 2}, overwrite=True) as artifacts:
                artifacts.logger.info("replacement")
                artifacts.commit({"ok": "replacement"})
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"ok": "replacement"})
            replaced_config = json.loads((root / "d" / "m" / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(replaced_config["x"], 2)

            failed_root = Path(name) / "failed"
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with AtomicRunArtifacts(failed_root, "d", "m", 1, 62, config):
                    raise RuntimeError("intentional")
            self.assertFalse(any(failed_root.rglob("*")) if failed_root.exists() else False)


if __name__ == "__main__":
    unittest.main()
