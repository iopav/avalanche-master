"""TagFex model core and its Avalanche supervised-strategy adapter.

The image model follows the source TagFex expansion, freezing, loss, rehearsal,
and weight-alignment semantics while accepting a registered project backbone.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from avalanche.training.templates import SupervisedTemplate
from .models import build_feature_map_extractor
from .replay_storage import Float32ReplayExample, PackedBinaryExample


def info_nce_loss(feats: torch.Tensor, temperature: float) -> torch.Tensor:
    """Exact tensor formulation used by the original TagFex implementation."""
    cosine = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    self_mask = torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
    cosine = cosine.masked_fill(self_mask, -9e15)
    positive_mask = self_mask.roll(shifts=cosine.shape[0] // 2, dims=0)
    cosine = cosine / temperature
    return (-cosine[positive_mask] + torch.logsumexp(cosine, dim=-1)).mean()


def info_nce_distill_loss(
    predicted: torch.Tensor, target: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Cross-network InfoNCE distillation from the original TagFex loss."""
    cosine = F.cosine_similarity(
        predicted[:, None, :], target[None, :, :], dim=-1
    )
    self_mask = torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
    cosine = cosine.masked_fill(self_mask, -9e15)
    positive_mask = self_mask.roll(shifts=cosine.shape[0] // 2, dims=0)
    cosine = cosine / temperature
    return (-cosine[positive_mask] + torch.logsumexp(cosine, dim=-1)).mean()


class TSAttention(nn.Module):
    """TagFex task-specific attention without a fused, opaque attention op."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("TagFex attention heads must divide the feature dimension")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.norm_ts = nn.LayerNorm(embed_dim)
        self.norm_ta = nn.LayerNorm(embed_dim)
        self.weight_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_k_ts = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_k_ta = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_v_ts = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_v_ta = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (
            self.weight_q,
            self.weight_k_ts,
            self.weight_k_ta,
            self.weight_v_ts,
            self.weight_v_ta,
        ):
            nn.init.xavier_normal_(weight)
        self.norm_ta.reset_parameters()
        self.norm_ts.reset_parameters()

    def forward(self, ta_features: torch.Tensor, ts_features: torch.Tensor) -> torch.Tensor:
        batch, length, channels = ta_features.shape
        head_dim = channels // self.num_heads

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, length, self.num_heads, head_dim).transpose(1, 2)

        ta_features = self.norm_ta(ta_features)
        ts_features = self.norm_ts(ts_features)
        query = split_heads(ts_features @ self.weight_q)
        key = torch.cat(
            (
                split_heads(ta_features @ self.weight_k_ta),
                split_heads(ts_features @ self.weight_k_ts),
            ),
            dim=2,
        )
        value = torch.cat(
            (
                split_heads(ta_features @ self.weight_v_ta),
                split_heads(ts_features @ self.weight_v_ts),
            ),
            dim=2,
        )
        attention = torch.softmax((query @ key.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)
        return (attention @ value).transpose(1, 2).flatten(2)


class OriginalSimpleLinear(nn.Module):
    """Initialization-compatible copy of TagFex's source SimpleLinear."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, nonlinearity="linear")
        nn.init.zeros_(self.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight, self.bias)


class ImageTagFexNet(nn.Module):
    """Source-structure TagFexNet backed by a registered image ResNet18."""

    def __init__(
        self,
        backbone_id: str,
        dataset_name: str,
        model_input_shape: tuple[int, ...],
        *,
        proj_hidden_dim: int,
        proj_output_dim: int,
        interpolation_factor: float,
        attention_heads: int,
    ):
        super().__init__()
        self.backbone_id = backbone_id
        self.dataset_name = dataset_name
        self.model_input_shape = tuple(model_input_shape)
        self.interpolation_factor = float(interpolation_factor)
        self.attention_heads = int(attention_heads)
        self.ta_net = self._new_backbone()
        self.ts_nets = nn.ModuleList()
        self.classifier: OriginalSimpleLinear | None = None
        self.aux_classifier: OriginalSimpleLinear | None = None
        self.trans_classifier: OriginalSimpleLinear | None = None
        self.predictor: OriginalSimpleLinear | None = None
        self.ts_attn: TSAttention | None = None
        self.projector = nn.Sequential(
            OriginalSimpleLinear(self.ta_net.out_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            OriginalSimpleLinear(proj_hidden_dim, proj_output_dim),
        )

    def _new_backbone(self):
        return build_feature_map_extractor(
            self.backbone_id, self.dataset_name, self.model_input_shape
        )

    @property
    def feature_dim(self) -> int:
        return sum(int(network.out_dim) for network in self.ts_nets)

    @property
    def output_dim(self) -> int:
        return 0 if self.classifier is None else int(self.classifier.out_features)

    def update_network(self, num_new_classes: int) -> None:
        device = next(self.parameters()).device
        old_output_dim = self.output_dim
        new_backbone = self._new_backbone().to(device)
        self.ts_nets.append(new_backbone)
        if len(self.ts_nets) > 1:
            gamma = self.interpolation_factor
            with torch.no_grad():
                for ta_value, old_value, new_value in zip(
                    self.ta_net.parameters(),
                    self.ts_nets[-2].parameters(),
                    self.ts_nets[-1].parameters(),
                ):
                    new_value.copy_(gamma * old_value + (1.0 - gamma) * ta_value)

        new_dim = int(new_backbone.out_dim)
        expanded = OriginalSimpleLinear(
            self.feature_dim, old_output_dim + int(num_new_classes)
        ).to(device)
        if self.classifier is not None:
            with torch.no_grad():
                expanded.weight[:old_output_dim, :-new_dim].copy_(self.classifier.weight)
                expanded.weight[:old_output_dim, -new_dim:].zero_()
                expanded.bias[:old_output_dim].copy_(self.classifier.bias)
        self.classifier = expanded

        if len(self.ts_nets) > 1:
            self.aux_classifier = OriginalSimpleLinear(
                new_dim, int(num_new_classes) + 1
            ).to(device)
            if self.predictor is None:
                self.predictor = OriginalSimpleLinear(
                    self.ta_net.out_dim, self.ta_net.out_dim
                ).to(device)
            if self.ts_attn is None:
                self.ts_attn = TSAttention(
                    new_dim, self.attention_heads
                ).to(device)
            else:
                self.ts_attn.reset_parameters()
            self.trans_classifier = OriginalSimpleLinear(
                self.ta_net.out_dim, int(num_new_classes)
            ).to(device)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if self.classifier is None or not self.ts_nets:
            raise RuntimeError("ImageTagFexNet must be expanded before its first forward")
        ts_outputs = [network(x) for network in self.ts_nets]
        ts_features = [output["features"] for output in ts_outputs]
        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "logits": self.classifier(torch.cat(ts_features, dim=-1)),
            "ts_features": ts_features,
        }
        if self.training:
            ta_fmap = self.ta_net(x)["fmaps"][-1]
            ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)
            outputs.update(
                ta_feature=ta_feature,
                embedding=self.projector(ta_feature),
            )
            if self.trans_classifier is not None:
                assert self.ts_attn is not None
                ts_feature = ts_outputs[-1]["fmaps"][-1].flatten(2).permute(0, 2, 1)
                ta_sequence = ta_fmap.flatten(2).permute(0, 2, 1)
                merged = self.ts_attn(ta_sequence.detach(), ts_feature).mean(1)
                outputs["trans_logits"] = self.trans_classifier(merged)
            if self.aux_classifier is not None:
                outputs["aux_logits"] = self.aux_classifier(ts_features[-1])
            if self.predictor is not None:
                outputs["predicted_feature"] = self.predictor(ta_feature)
        return outputs

    def train(self, mode: bool = True):
        super().train(mode)
        for network in self.ts_nets[:-1]:
            network.eval()
        return self

    def get_freezed_copy_ta(self):
        result = deepcopy(self.ta_net).eval()
        result.requires_grad_(False)
        return result

    def get_freezed_copy_projector(self):
        result = deepcopy(self.projector).eval()
        result.requires_grad_(False)
        return result

    def freeze_old_backbones(self) -> None:
        for network in self.ts_nets[:-1]:
            network.requires_grad_(False)
            network.eval()

    @torch.no_grad()
    def weight_align(self, num_new_classes: int) -> None:
        assert self.classifier is not None
        new = self.classifier.weight[-num_new_classes:].norm(dim=-1).mean()
        old = self.classifier.weight[:-num_new_classes].norm(dim=-1).mean()
        self.classifier.weight[-num_new_classes:] *= old / new.clamp_min(1e-12)


def add_original_tagfex_to_path(tagfex_root: Path) -> None:
    """Make the unmodified TagFex Python packages importable."""
    resolved = str(tagfex_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def load_original_tagfex_core(tagfex_root: Path):
    add_original_tagfex_to_path(tagfex_root)
    from methods.tagfex.tagfex import infoNCE_distill_loss, infoNCE_loss
    from methods.tagfex.tagfexnet import TagFexNet

    return TagFexNet, infoNCE_loss, infoNCE_distill_loss


class NpyImageDataset(Dataset):
    """Read-only NPY image dataset used to build an Avalanche benchmark."""

    def __init__(self, x_path: Path, y_path: Path):
        self._x = np.load(x_path, mmap_mode="r")
        self.targets = [int(value) for value in np.load(y_path).tolist()]
        self.classes = sorted(set(self.targets))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        # Copying avoids PyTorch's undefined behavior on read-only memmaps.
        image = np.array(self._x[index], dtype=np.float32, copy=True)
        return torch.from_numpy(image), self.targets[index]


class _TwoViewDataset(Dataset):
    """Match TagFex's num_aug=2 sample layout for transform-free NPY data."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        x, y = item[0], item[1]
        # The source smoke config uses no stochastic transform, so its two
        # separately fetched views are value-identical as well.
        return x, x.clone(), int(y)


class _StoredExamples(Dataset):
    def __init__(
        self,
        examples: Sequence[
            tuple[torch.Tensor, int] | PackedBinaryExample | Float32ReplayExample
        ],
        on_materialize=None,
    ):
        self.examples = examples
        self.on_materialize = on_materialize

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        if isinstance(example, (PackedBinaryExample, Float32ReplayExample)):
            packed = isinstance(example, PackedBinaryExample)
            if self.on_materialize is not None:
                self.on_materialize(example.numel, packed)
            x = example.unpack() if packed else example.materialize()
            y = example.label
        else:
            x, y = example
        return x, y, 0


@dataclass(frozen=True)
class TagFexHyperParameters:
    init_epochs: int = 1
    inc_epochs: int = 1
    train_mb_size: int = 64
    eval_mb_size: int = 128
    memory_size: int = 2000
    init_lr: float = 0.1
    inc_lr: float = 0.1
    momentum: float = 0.9
    init_weight_decay: float = 5e-4
    inc_weight_decay: float = 2e-4
    init_milestones: tuple[int, ...] = (60, 120, 170)
    inc_milestones: tuple[int, ...] = (80, 120, 150)
    gamma: float = 0.1
    contrast_factor: float = 1.0
    contrast_kd_factor: float = 2.0
    aux_factor: float = 2.0
    trans_cls_factor: float = 1.0
    transfer_factor: float = 1.0
    infonce_temp: float = 0.2
    infonce_kd_temp: float = 0.2
    kd_temp: float = 2.0
    proj_hidden_dim: int = 2048
    proj_output_dim: int = 1024
    interpolation_factor: float = 0.95
    attention_heads: int = 8


class AvalancheTagFex(SupervisedTemplate):
    """TagFex expressed through Avalanche's supervised strategy lifecycle."""

    def __init__(
        self,
        *,
        tagfex_root: Path | None = None,
        device: torch.device,
        hparams: TagFexHyperParameters,
        dataset_name: str = "my_npy_dataset",
        num_classes: int | None = None,
        model: nn.Module | None = None,
        info_nce=info_nce_loss,
        info_nce_distill=info_nce_distill_loss,
        plugins: Sequence | None = None,
    ):
        if model is None:
            if tagfex_root is None:
                raise ValueError("tagfex_root is required for the original image model")
            TagFexNet, info_nce, info_nce_distill = load_original_tagfex_core(tagfex_root)
            backbone_configs = {
                "name": "resnet18",
                "params": {"dataset_name": dataset_name, "small_base": False},
            }
            network_configs = {
                "classifier_type": "unified",
                "proj_hidden_dim": hparams.proj_hidden_dim,
                "proj_output_dim": hparams.proj_output_dim,
                "init_from_last": False,
                "init_from_interpolation": True,
                "init_interpolation_factor": hparams.interpolation_factor,
                "attn_num_heads": hparams.attention_heads,
                "merge_attn": True,
            }
            model = TagFexNet(backbone_configs, network_configs, device)
        optimizer = SGD(model.parameters(), lr=hparams.init_lr, foreach=False)
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=hparams.train_mb_size,
            train_epochs=hparams.init_epochs,
            eval_mb_size=hparams.eval_mb_size,
            device=device,
            evaluator=None,
            eval_every=-1,
            plugins=list(plugins or ()),
        )
        self.hparams = hparams
        self.dataset_name = dataset_name
        self.num_classes = None if num_classes is None else int(num_classes)
        self.pack_binary_replay = dataset_name == "spike"
        if self.pack_binary_replay and (self.num_classes is None or self.num_classes <= 0):
            raise ValueError("Packed TagFex replay requires the total dataset class count")
        self._info_nce = info_nce
        self._info_nce_distill = info_nce_distill
        self.last_ta_net = None
        self.last_projector = None
        self.scheduler: MultiStepLR | None = None
        self.seen_classes: list[int] = []
        self.memory_by_class: dict[
            int,
            list[
                tuple[torch.Tensor, int]
                | PackedBinaryExample
                | Float32ReplayExample
            ],
        ] = {}
        self.class_means: dict[int, torch.Tensor] = {}
        self.loss_components: dict[str, float] = {}
        self._cil_phase_plugin = None
        self._replay_unpack_calls = 0
        self._replay_unpacked_elements = 0
        self._replay_label_decode_calls = 0

    def _note_replay_materialize(self, elements: int, packed: bool) -> None:
        self._replay_label_decode_calls += 1
        if packed:
            self._replay_unpack_calls += 1
            self._replay_unpacked_elements += int(elements)

    def _record_nonflop(self, name: str, calls: int, reason: str, variables: dict) -> None:
        profiler = None if self._cil_phase_plugin is None else self._cil_phase_plugin.profiler
        if profiler is not None and calls:
            profiler.add_nonflop(name, calls, reason, variables)

    @property
    def mb_y(self):
        assert self.mbatch is not None
        return self.mbatch[2] if self.is_training else self.mbatch[1]

    def _phase_context(self, phase: str):
        plugin = self._cil_phase_plugin
        profiler = None if plugin is None else plugin.profiler
        return contextlib.nullcontext() if profiler is None else profiler.temporary_phase(phase)

    @property
    def learned_num_classes(self) -> int:
        return len(self.seen_classes) - len(self._current_classes())

    def _current_classes(self) -> list[int]:
        assert self.experience is not None
        return sorted(int(v) for v in self.experience.classes_in_this_experience)

    def model_adaptation(self, model=None):
        # TagFex performs its own expansion once per training experience.
        return self.model if model is None else model

    def check_model_and_optimizer(self, **kwargs):
        current_classes = self._current_classes()
        if self.seen_classes:
            self.last_ta_net = self.model.get_freezed_copy_ta()
            self.last_projector = self.model.get_freezed_copy_projector()

        self.model.update_network(len(current_classes))
        self.model.freeze_old_backbones()
        self.model.to(self.device)
        self.seen_classes.extend(current_classes)

        first = len(self.seen_classes) == len(current_classes)
        self.train_epochs = self.hparams.init_epochs if first else self.hparams.inc_epochs
        lr = self.hparams.init_lr if first else self.hparams.inc_lr
        weight_decay = (
            self.hparams.init_weight_decay if first else self.hparams.inc_weight_decay
        )
        milestones = (
            self.hparams.init_milestones if first else self.hparams.inc_milestones
        )
        # The original code rebuilds both objects for every task.
        self.optimizer = SGD(
            self.model.parameters(),
            lr=lr,
            momentum=self.hparams.momentum,
            weight_decay=weight_decay,
            foreach=False,
        )
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=list(milestones), gamma=self.hparams.gamma
        )

    def make_train_dataloader(
        self,
        num_workers=0,
        shuffle=True,
        pin_memory=None,
        drop_last=False,
        **kwargs,
    ):
        assert self.experience is not None
        datasets: list[Dataset] = [self.experience.dataset.train()]
        replay = [example for values in self.memory_by_class.values() for example in values]
        if replay:
            datasets.append(
                _StoredExamples(
                    replay,
                    self._note_replay_materialize,
                )
            )
        merged: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        self.adapted_dataset = merged
        self.dataloader = DataLoader(
            _TwoViewDataset(merged),
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda" if pin_memory is None else pin_memory,
            drop_last=drop_last,
        )

    def forward(self):
        if self.is_training:
            sample1, sample2 = self.mbatch[0], self.mbatch[1]
            outputs = self.model(torch.cat((sample1, sample2)).contiguous())
            self._tagfex_outputs = outputs
            return outputs["logits"]
        return self.model(self.mb_x.contiguous())["logits"]

    def criterion(self):
        if not self.is_training:
            return F.cross_entropy(self.mb_output, self.mb_y)

        outputs = self._tagfex_outputs
        targets = torch.cat((self.mbatch[2], self.mbatch[2]))
        logits = outputs["logits"]
        cls_loss = F.cross_entropy(logits, targets)
        infonce_loss = self._info_nce(outputs["embedding"], self.hparams.infonce_temp)

        if outputs.get("aux_logits") is None:
            loss = cls_loss + self.hparams.contrast_factor * infonce_loss
            self.loss_components = {
                "cls": float(cls_loss.detach()),
                "contrast": float(infonce_loss.detach()),
            }
            return loss

        learned = self.learned_num_classes
        aux_targets = torch.where(targets >= learned, targets - learned + 1, 0)
        aux_loss = F.cross_entropy(outputs["aux_logits"], aux_targets)

        predicted_feature = outputs["predicted_feature"]
        assert self.last_ta_net is not None and self.last_projector is not None
        samples = torch.cat((self.mbatch[0], self.mbatch[1])).contiguous()
        with self._phase_context("learning_auxiliary"):
            old_ta_feature = self.last_ta_net(samples)["features"]
            kd_loss = self._info_nce_distill(
                self.last_projector(predicted_feature),
                self.last_projector(old_ta_feature),
                self.hparams.infonce_kd_temp,
            )

        current_mask = targets >= learned
        trans_logits = outputs["trans_logits"]
        if current_mask.any():
            trans_cls_loss = F.cross_entropy(
                trans_logits[current_mask], targets[current_mask] - learned
            )
        else:
            # Replay and current samples are shuffled together. The final
            # minibatch is allowed to contain replay samples only.
            trans_cls_loss = torch.tensor(0.0, device=self.device)
        if current_mask.any() and trans_cls_loss < cls_loss:
            temperature = self.hparams.kd_temp
            transfer_loss = F.kl_div(
                (logits[current_mask][:, learned:] / temperature).log_softmax(dim=1),
                (trans_logits.detach()[current_mask] / temperature).softmax(dim=1),
                reduction="batchmean",
            )
        else:
            transfer_loss = torch.tensor(0.0, device=self.device)

        auto_kd_factor = learned / len(self.seen_classes)
        loss = (
            cls_loss
            + self.hparams.aux_factor * aux_loss
            + self.hparams.contrast_factor
            * (
                infonce_loss * (1 - auto_kd_factor)
                + self.hparams.contrast_kd_factor * kd_loss * auto_kd_factor
            )
            + self.hparams.trans_cls_factor * trans_cls_loss
            + self.hparams.transfer_factor * transfer_loss
        )
        self.loss_components = {
            "cls": float(cls_loss.detach()),
            "contrast": float(infonce_loss.detach()),
            "aux": float(aux_loss.detach()),
            "kd": float(kd_loss.detach()),
            "trans_cls": float(trans_cls_loss.detach()),
            "transfer": float(transfer_loss.detach()),
        }
        return loss

    def _after_training_epoch(self, **kwargs):
        if self.scheduler is not None:
            self.scheduler.step()
        super()._after_training_epoch(**kwargs)

    @torch.no_grad()
    def _features(
        self,
        examples: Sequence[
            tuple[torch.Tensor, int] | PackedBinaryExample | Float32ReplayExample
        ],
    ) -> torch.Tensor:
        loader = DataLoader(
            _StoredExamples(
                examples,
                self._note_replay_materialize,
            ),
            batch_size=self.eval_mb_size,
        )
        was_training = self.model.training
        self.model.eval()
        features = []
        for batch in loader:
            x = batch[0].to(self.device, non_blocking=True)
            task_features = self.model(x.contiguous())["ts_features"]
            value = torch.cat(task_features, dim=-1)
            value = value / (value.norm(dim=-1, keepdim=True) + 1e-8)
            features.append(value)
        if was_training:
            self.model.train()
        return torch.cat(features)

    @torch.no_grad()
    def _collect_current_examples(self) -> dict[int, list[tuple[torch.Tensor, int]]]:
        assert self.experience is not None
        result = {class_id: [] for class_id in self._current_classes()}
        for index in range(len(self.experience.dataset)):
            item = self.experience.dataset[index]
            x, y = item[0], int(item[1])
            result[y].append((x.detach().cpu().clone(), y))
        return result

    @torch.no_grad()
    def _update_memory(self) -> None:
        exemplars_per_class = self.hparams.memory_size // len(self.seen_classes)

        # Match reduce_memory: retain the prefix of each old herding list and
        # recompute its mean in the current expanded feature space.
        for class_id in list(self.memory_by_class):
            examples = self.memory_by_class[class_id][:exemplars_per_class]
            self.memory_by_class[class_id] = examples
            features = self._features(examples)
            mean = features.mean(0)
            self.class_means[class_id] = mean / mean.norm()

        for class_id, examples in self._collect_current_examples().items():
            features = self._features(examples)
            num_to_select = min(len(features), exemplars_per_class)
            true_mean = features.mean(dim=0, keepdim=True)
            available_examples = list(examples)
            selected_examples: list[tuple[torch.Tensor, int]] = []
            selected_mean = torch.zeros(self.model.feature_dim, device=self.device)
            for n in range(1, num_to_select + 1):
                candidate_means = ((n - 1) * selected_mean + features) / n
                index = int((candidate_means - true_mean).norm(dim=-1).argmin())
                selected_mean = candidate_means[index]
                selected_examples.append(available_examples.pop(index))
                mask = torch.arange(len(features), device=features.device) != index
                features = features[mask]
            self.memory_by_class[class_id] = selected_examples
            self.class_means[class_id] = selected_mean / selected_mean.norm()

        total = sum(len(values) for values in self.memory_by_class.values())
        if total > self.hparams.memory_size:
            raise AssertionError(f"TagFex replay memory exceeded its cap: {total}")

    def _persist_memory(self) -> None:
        calls = 0
        elements = 0
        for class_id, examples in list(self.memory_by_class.items()):
            persistent_examples = []
            for example in examples:
                if isinstance(example, (PackedBinaryExample, Float32ReplayExample)):
                    persistent = example
                else:
                    sample, label = example
                    example_class = (
                        PackedBinaryExample
                        if self.pack_binary_replay
                        else Float32ReplayExample
                    )
                    persistent = example_class.from_tensor(sample, label, self.num_classes)
                    calls += 1
                    if self.pack_binary_replay:
                        elements += persistent.numel
                persistent_examples.append(persistent)
            self.memory_by_class[class_id] = persistent_examples
        if self.pack_binary_replay:
            self._record_nonflop(
                "numpy.packbits",
                calls,
                "Binary bit packing is persistent-storage preparation, not floating-point arithmetic.",
                {"packed_elements": elements},
            )
        self._record_nonflop(
            "uint8_one_hot_encode",
            calls,
            "One-hot allocation and indexed assignment are non-floating-point storage work.",
            {"classes": self.num_classes},
        )

    def _flush_unpack_record(self) -> None:
        self._record_nonflop(
            "numpy.unpackbits",
            self._replay_unpack_calls,
            "Replay bit unpacking and dtype materialization are non-floating-point work.",
            {"unpacked_elements": self._replay_unpacked_elements},
        )
        self._record_nonflop(
            "uint8_one_hot_decode",
            self._replay_label_decode_calls,
            "One-hot comparison and index recovery are non-floating-point work.",
            {"classes": self.num_classes},
        )
        self._replay_unpack_calls = 0
        self._replay_unpacked_elements = 0
        self._replay_label_decode_calls = 0

    def _after_training_exp(self, **kwargs):
        with self._phase_context("learning_auxiliary"):
            self._update_memory()
            self._persist_memory()
            self._flush_unpack_record()
        current_count = len(self._current_classes())
        if len(self.seen_classes) > current_count:
            with self._phase_context("learning_auxiliary"):
                self.model.weight_align(current_count)
        super()._after_training_exp(**kwargs)

    @torch.no_grad()
    def nme_predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        features = torch.cat(self.model(x.contiguous())["ts_features"], dim=-1)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        means = torch.stack([self.class_means[i] for i in range(len(self.seen_classes))])
        return torch.cdist(features, means).argmin(dim=1)


def build_uwave_image_benchmark(data_root: Path):
    from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark

    train = NpyImageDataset(data_root / "X_train_uwave_img.npy", data_root / "Y_train_uwave_img.npy")
    test = NpyImageDataset(data_root / "X_test_uwave_img.npy", data_root / "Y_test_uwave_img.npy")
    if train.classes != list(range(8)) or test.classes != list(range(8)):
        raise ValueError("The UWave image smoke benchmark requires labels 0..7")
    return nc_benchmark(
        train_dataset=train,
        test_dataset=test,
        n_experiences=6,
        task_labels=False,
        shuffle=False,
        fixed_class_order=list(range(8)),
        per_exp_classes={0: 3, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        class_ids_from_zero_from_first_exp=False,
        train_transform=None,
        eval_transform=None,
    )


@torch.no_grad()
def evaluate_seen_experiences(
    strategy: AvalancheTagFex, experiences: Iterable, task_index: int
) -> tuple[list[float], list[float]]:
    classifier_accuracy = []
    nme_accuracy = []
    strategy.model.eval()
    for experience in list(experiences)[: task_index + 1]:
        loader = DataLoader(experience.dataset.eval(), batch_size=strategy.eval_mb_size)
        classifier_correct = nme_correct = total = 0
        for batch in loader:
            x = batch[0].to(strategy.device)
            y = batch[1].to(strategy.device)
            logits = strategy.model(x.contiguous())["logits"]
            classifier_correct += int((logits.argmax(dim=1) == y).sum())
            nme_correct += int((strategy.nme_predict(x) == y).sum())
            total += int(y.numel())
        classifier_accuracy.append(classifier_correct / total)
        nme_accuracy.append(nme_correct / total)
    return classifier_accuracy, nme_accuracy
