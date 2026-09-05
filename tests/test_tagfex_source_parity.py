from __future__ import annotations

from copy import deepcopy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "TagFex_CVPR2025" / "TagFex_CVPR2025"


def _load_source():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from methods.tagfex.tagfex import infoNCE_distill_loss, infoNCE_loss
    from methods.tagfex.tagfexnet import TagFexNet

    return TagFexNet, infoNCE_loss, infoNCE_distill_loss


def _network_config() -> dict:
    return {
        "classifier_type": "unified",
        "proj_hidden_dim": 32,
        "proj_output_dim": 16,
        "init_from_last": False,
        "init_from_interpolation": True,
        "init_interpolation_factor": 0.95,
        "attn_num_heads": 8,
        "merge_attn": True,
    }


def _copy_state_by_order(source: torch.nn.Module, target: torch.nn.Module) -> None:
    source_state = list(source.state_dict().items())
    target_state = list(target.state_dict().items())
    if len(source_state) != len(target_state):
        raise AssertionError(
            f"state tensor count differs: source={len(source_state)}, target={len(target_state)}"
        )
    with torch.no_grad():
        for (source_name, source_value), (target_name, target_value) in zip(
            source_state, target_state
        ):
            if source_value.shape != target_value.shape:
                raise AssertionError(
                    f"state shape differs: {source_name}={tuple(source_value.shape)}; "
                    f"{target_name}={tuple(target_value.shape)}"
                )
            target_value.copy_(source_value)


def _full_incremental_loss(
    outputs: dict,
    targets: torch.Tensor,
    old_ta: torch.nn.Module,
    old_projector: torch.nn.Module,
    samples: torch.Tensor,
    info_nce,
    info_nce_distill,
) -> torch.Tensor:
    learned = 2
    cls_loss = F.cross_entropy(outputs["logits"], targets)
    contrast = info_nce(outputs["embedding"], 0.2)
    aux_targets = torch.where(targets >= learned, targets - learned + 1, 0)
    aux_loss = F.cross_entropy(outputs["aux_logits"], aux_targets)
    old_ta_feature = old_ta(samples)["features"]
    kd_loss = info_nce_distill(
        old_projector(outputs["predicted_feature"]),
        old_projector(old_ta_feature),
        0.2,
    )
    current = targets >= learned
    trans_cls_loss = F.cross_entropy(
        outputs["trans_logits"][current], targets[current] - learned
    )
    if trans_cls_loss < cls_loss:
        transfer_loss = F.kl_div(
            (outputs["logits"][current][:, learned:] / 2.0).log_softmax(dim=1),
            (outputs["trans_logits"].detach()[current] / 2.0).softmax(dim=1),
            reduction="batchmean",
        )
    else:
        transfer_loss = torch.zeros((), device=samples.device)
    return (
        cls_loss
        + 2.0 * aux_loss
        + contrast * 0.5
        + 2.0 * kd_loss * 0.5
        + 0.005 * trans_cls_loss
        + transfer_loss
    )


@unittest.skipUnless(SOURCE_ROOT.exists(), "Adjacent original TagFex checkout is absent")
class TagFexSourceParityTests(unittest.TestCase):
    def test_original_cifar_resnet32_is_not_any_cil_resnet18_width(self):
        TagFexNet, _, _ = _load_source()
        from cil_experiments.models import BACKBONES
        from cil_experiments.tagfex_avalanche import ImageTagFexNet

        source = TagFexNet({"name": "resnet32"}, _network_config(), torch.device("cpu"))
        self.assertEqual(source.ta_net.out_dim, 64)
        self.assertEqual(len(source.ta_net.stage_1), 5)
        self.assertEqual(len(source.ta_net.stage_2), 5)
        self.assertEqual(len(source.ta_net.stage_3), 5)
        self.assertEqual(set(BACKBONES), {"resnet18_cifar", "temporal"})

        source.update_network(2)
        self.assertEqual(source.classifier.in_features, 64)
        sample = torch.randn(2, 3, 32, 32)
        self.assertEqual(tuple(source.ta_net(sample)["fmaps"][-1].shape), (2, 64, 8, 8))

        for backbone_id, expected_dim in (("resnet18_cifar", 512),):
            port = ImageTagFexNet(
                backbone_id,
                "uwave",
                (3, 32, 32),
                proj_hidden_dim=32,
                proj_output_dim=16,
                interpolation_factor=0.95,
                attention_heads=8,
            )
            port.update_network(2)
            self.assertEqual(port.classifier.in_features, expected_dim)
            self.assertEqual(
                tuple(port.ta_net(sample)["fmaps"][-1].shape),
                (2, expected_dim, 4, 4),
            )

    def test_original_resnet18_default_stem_is_not_cil_stem(self):
        TagFexNet, _, _ = _load_source()
        from cil_experiments.tagfex_avalanche import ImageTagFexNet

        source = TagFexNet({"name": "resnet18"}, _network_config(), torch.device("cpu"))
        port = ImageTagFexNet(
            "resnet18_cifar",
            "uwave",
            (3, 32, 32),
            proj_hidden_dim=32,
            proj_output_dim=16,
            interpolation_factor=0.95,
            attention_heads=8,
        )
        self.assertEqual(source.ta_net.conv1[0].kernel_size, (7, 7))
        self.assertEqual(source.ta_net.conv1[0].stride, (2, 2))
        self.assertIsInstance(source.ta_net.conv1[3], torch.nn.MaxPool2d)
        self.assertEqual(port.ta_net.backbone.conv1.kernel_size, (3, 3))
        self.assertEqual(port.ta_net.backbone.conv1.stride, (1, 1))
        sample = torch.randn(2, 3, 32, 32)
        self.assertEqual(tuple(source.ta_net(sample)["fmaps"][-1].shape[-2:]), (1, 1))
        self.assertEqual(tuple(port.ta_net(sample)["fmaps"][-1].shape[-2:]), (4, 4))

    def test_aligned_model_loss_gradient_and_one_sgd_step(self):
        TagFexNet, source_info_nce, source_info_nce_distill = _load_source()
        from cil_experiments.tagfex_avalanche import (
            ImageTagFexNet,
            info_nce_distill_loss,
            info_nce_loss,
        )

        torch.manual_seed(62)
        source = TagFexNet(
            {"name": "resnet18", "params": {"dataset_name": "cifar"}},
            _network_config(),
            torch.device("cpu"),
        )
        port = ImageTagFexNet(
            "resnet18_cifar",
            "uwave",
            (3, 32, 32),
            proj_hidden_dim=32,
            proj_output_dim=16,
            interpolation_factor=0.95,
            attention_heads=8,
        )

        source.update_network(2)
        port.update_network(2)
        _copy_state_by_order(source, port)
        source_old_ta = deepcopy(source.ta_net).eval().requires_grad_(False)
        port_old_ta = deepcopy(port.ta_net).eval().requires_grad_(False)
        source_old_projector = deepcopy(source.projector).eval().requires_grad_(False)
        port_old_projector = deepcopy(port.projector).eval().requires_grad_(False)

        source.update_network(2)
        port.update_network(2)
        _copy_state_by_order(source, port)
        source.train()
        port.train()

        first_view = torch.randn(4, 3, 32, 32)
        samples = torch.cat((first_view, first_view.clone()))
        targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        source_outputs = source(samples)
        port_outputs = port(samples)
        for key in (
            "logits",
            "ta_feature",
            "embedding",
            "trans_logits",
            "aux_logits",
            "predicted_feature",
        ):
            torch.testing.assert_close(source_outputs[key], port_outputs[key], rtol=1e-5, atol=1e-6)
        for source_feature, port_feature in zip(
            source_outputs["ts_features"], port_outputs["ts_features"]
        ):
            torch.testing.assert_close(source_feature, port_feature, rtol=1e-5, atol=1e-6)

        source_loss = _full_incremental_loss(
            source_outputs,
            targets,
            source_old_ta,
            source_old_projector,
            samples,
            source_info_nce,
            source_info_nce_distill,
        )
        port_loss = _full_incremental_loss(
            port_outputs,
            targets,
            port_old_ta,
            port_old_projector,
            samples,
            info_nce_loss,
            info_nce_distill_loss,
        )
        torch.testing.assert_close(source_loss, port_loss, rtol=1e-5, atol=1e-6)

        source_optimizer = torch.optim.SGD(source.parameters(), lr=0.01)
        port_optimizer = torch.optim.SGD(port.parameters(), lr=0.01)
        source_optimizer.zero_grad()
        port_optimizer.zero_grad()
        source_loss.backward()
        port_loss.backward()
        source_optimizer.step()
        port_optimizer.step()
        for source_parameter, port_parameter in zip(
            source.parameters(), port.parameters()
        ):
            torch.testing.assert_close(
                source_parameter, port_parameter, rtol=2e-5, atol=2e-6
            )


if __name__ == "__main__":
    unittest.main()
