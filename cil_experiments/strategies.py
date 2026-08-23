from __future__ import annotations

import contextlib
import copy
import itertools
from dataclasses import dataclass
from math import floor
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from .models import TemporalBackbone, TemporalClassifier
from .registry import METHODS, TRAINING_DEFAULTS


def _phase_context(phase_plugin, phase: str):
    if phase_plugin is None:
        return contextlib.nullcontext()
    profiler = phase_plugin.profiler
    return profiler.temporary_phase(phase) if profiler is not None else contextlib.nullcontext()


class InstrumentedERACE:
    """Avalanche ER_ACE loop with mutually exclusive current/replay FLOP phases."""

    def _before_training_exp(self, **kwargs):
        with _phase_context(self._cil_phase_plugin, "exemplar_or_sample_selection"):
            return super()._before_training_exp(**kwargs)

    def training_epoch(self, **kwargs):
        from avalanche.models.utils import avalanche_forward

        for self.mbatch in self.dataloader:
            if self._stop_training:
                break
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            if self.replay_loader is not None:
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = next(self.replay_loader)
                self.mb_buffer_x = self.mb_buffer_x.to(self.device)
                self.mb_buffer_y = self.mb_buffer_y.to(self.device)
                self.mb_buffer_tid = self.mb_buffer_tid.to(self.device)
                self._cil_extra_processed_samples = int(len(self.mb_buffer_y))
            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            if self.replay_loader is not None:
                with _phase_context(self._cil_phase_plugin, "replay_forward"):
                    self.mb_buffer_out = avalanche_forward(
                        self.model, self.mb_buffer_x, self.mb_buffer_tid
                    )
            self._after_forward(**kwargs)
            if self.replay_loader is None:
                self.loss += self.criterion()
            else:
                self.loss += self.ace_criterion(
                    self.mb_output, self.mb_y, self.mb_buffer_out, self.mb_buffer_y
                )
            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)
            self._after_training_iteration(**kwargs)


def _make_instrumented_er_ace_class():
    from avalanche.training.supervised import ER_ACE

    class _InstrumentedERACE(InstrumentedERACE, ER_ACE):
        pass

    return _InstrumentedERACE


class FrozenBackboneFeCAMUpdate:
    """Train the backbone on task 1, then only extract FeCAM statistics."""

    def __init__(self, phase_plugin, first_experience_epochs: int):
        from avalanche.core import SupervisedPlugin

        SupervisedPlugin.__init__(self)
        self.phase_plugin = phase_plugin
        self.first_experience_epochs = int(first_experience_epochs)

    @staticmethod
    def _freeze_feature_extractor(strategy) -> None:
        feature_extractor = strategy.model.feature_extractor
        feature_extractor.eval()
        for parameter in feature_extractor.parameters():
            parameter.requires_grad_(False)

    def before_training_exp(self, strategy, **kwargs):
        experience_index = int(strategy.clock.train_exp_counter)
        if experience_index == 0:
            strategy.train_epochs = self.first_experience_epochs
            return
        # FeCAM is classifier-incremental: after the first task there is no
        # gradient training. New classes are incorporated by estimating their
        # means and covariance matrices with the frozen feature extractor.
        self._freeze_feature_extractor(strategy)
        strategy.train_epochs = 0

    def after_training_exp(self, strategy, **kwargs):
        from avalanche.training.plugins.update_fecam import _check_has_fecam, _gather_means_and_cov

        _check_has_fecam(strategy.model)
        experience_index = int(strategy.clock.train_exp_counter)
        profiler = self.phase_plugin.profiler if self.phase_plugin is not None else None
        statistics_only_loop = experience_index > 0 and profiler is not None
        if statistics_only_loop:
            # Later FeCAM experiences have no SGD epoch. Treat the single
            # statistics pass as the terminal executed loop required by the
            # metrics1 denominator.
            profiler.begin_epoch()
        with _phase_context(self.phase_plugin, "prototype_and_class_statistics"):
            means, covs = _gather_means_and_cov(
                strategy.model,
                strategy.experience.dataset,
                strategy.train_mb_size,
                strategy.device,
                **kwargs,
            )
            strategy.model.eval_classifier.update_class_means_dict(means)
            strategy.model.eval_classifier.update_class_cov_dict(covs)
        if statistics_only_loop:
            profiler.add_processed_samples(len(strategy.experience.dataset))
            profiler.end_epoch()
        if experience_index == 0:
            self._freeze_feature_extractor(strategy)


def _make_fecam_plugin(phase_plugin, first_experience_epochs: int):
    from avalanche.core import SupervisedPlugin

    class _Plugin(FrozenBackboneFeCAMUpdate, SupervisedPlugin):
        pass

    return _Plugin(phase_plugin, first_experience_epochs)


def _make_device_safe_fecam_classifier(**parameters):
    from avalanche.models import FeCAMClassifier

    class _DeviceSafeFeCAMClassifier(FeCAMClassifier):
        @torch.no_grad()
        def forward(self, x):
            if self.class_means_dict == {}:
                self.init_missing_classes(range(self.max_class + 1), x.shape[1], x.device)
            if self.class_means_dict == {}:
                raise RuntimeError("FeCAM has no class statistics for inference")
            if self.tukey:
                x = self._tukey_transforms(x)
            # Avalanche 0.6 allocates this tensor on CPU, which returns CPU
            # logits even when features and covariance matrices are on CUDA.
            maha_dist = torch.full(
                (self.max_class + 1, x.shape[0]),
                torch.inf,
                dtype=x.dtype,
                device=x.device,
            )
            for class_id, prototype in self.class_means_dict.items():
                maha_dist[class_id] = self._mahalanobis(
                    x, prototype.to(x.device), self.class_cov_dict[class_id].to(x.device)
                )
            return -maha_dist.T

    return _DeviceSafeFeCAMClassifier(**parameters)


class InstrumentedICaRLLossMixin:
    def __init__(self, phase_plugin):
        super().__init__()
        self.phase_plugin = phase_plugin

    def before_forward(self, strategy, **kwargs):
        if self.old_model is not None:
            with _phase_context(self.phase_plugin, "teacher_distillation_forward"):
                with torch.no_grad():
                    self.old_logits = self.old_model(strategy.mb_x)

    def __call__(self, logits, targets):
        predictions = torch.sigmoid(logits)
        one_hot = torch.zeros(
            targets.shape[0], logits.shape[1], dtype=torch.float, device=logits.device
        )
        one_hot[range(len(targets)), targets.long()] = 1
        if self.old_logits is not None:
            with _phase_context(self.phase_plugin, "teacher_distillation_forward"):
                old_predictions = torch.sigmoid(self.old_logits)
            one_hot[:, self.old_classes] = old_predictions[:, self.old_classes]
            self.old_logits = None
        return self.criterion(predictions, one_hot)

    def after_training_exp(self, strategy, **kwargs):
        # IncrementalClassifier changes shape at every experience.  The stock
        # plugin reuses the previous clone and load_state_dict then fails on
        # the expanded classifier.  Replacing the frozen teacher is the exact
        # state transition required by iCaRL and preserves the just-finished
        # model for distillation in the next experience.
        self.old_model = copy.deepcopy(strategy.model).to(strategy.device)
        # TrainEvalModel selects the differentiable classifier through its
        # training flag. Gradients remain disabled by before_forward.
        self.old_model.train()
        self.old_classes += np.unique(strategy.experience.dataset.targets).tolist()


def _make_icarl_loss(phase_plugin):
    from avalanche.training.losses import ICaRLLossPlugin

    class _Loss(InstrumentedICaRLLossMixin, ICaRLLossPlugin):
        pass

    return _Loss(phase_plugin)


class InstrumentedICaRLPluginMixin:
    """iCaRL plugin with exact <=2000 balancing and real packed Spike persistence."""

    def __init__(self, memory_size, phase_plugin, pack_binary=False):
        super().__init__(memory_size=memory_size, buffer_transform=None, fixed_memory=True)
        self.phase_plugin = phase_plugin
        self.pack_binary = pack_binary
        self.packed_memory: list[dict[str, Any]] = []

    def _budget_for(self, class_id: int) -> int:
        q, r = divmod(self.memory_size, len(self.observed_classes))
        index = self.observed_classes.index(class_id)
        return q + (1 if index < r else 0)

    def _materialize_packed(self) -> list[torch.Tensor]:
        materialized = []
        for item in self.packed_memory:
            shape = tuple(item["shape"])
            count = math_prod(shape)
            unpacked = np.unpackbits(item["data"], bitorder="little", count=count)
            materialized.append(torch.from_numpy(unpacked.reshape(shape).astype(np.float32, copy=False)))
        return materialized

    def _pack_current_memory(self) -> None:
        if not self.pack_binary:
            for idx, tensor in enumerate(self.x_memory):
                self.x_memory[idx] = tensor.detach().cpu().to(torch.float32).contiguous()
            return
        packed = []
        for tensor in self.x_memory:
            array = tensor.detach().cpu().numpy()
            if not np.logical_or(array == 0, array == 1).all():
                raise ValueError("Spike ICaRL exemplar is not exactly binary and cannot be stored as 1-bit packed")
            bits = np.packbits(array.reshape(-1).astype(np.uint8), bitorder="little")
            packed.append({"data": bits, "shape": tuple(int(v) for v in array.shape)})
        self.packed_memory = packed
        self.x_memory = []

    def after_train_dataset_adaptation(self, strategy, **kwargs):
        if self.pack_binary and strategy.clock.train_exp_counter != 0:
            self.x_memory = self._materialize_packed()
        super().after_train_dataset_adaptation(strategy, **kwargs)

    def after_training_exp(self, strategy, **kwargs):
        strategy.model.eval()
        with _phase_context(self.phase_plugin, "exemplar_or_sample_selection"):
            self.construct_exemplar_set(strategy)
            self.reduce_exemplar_set(strategy)
        with _phase_context(self.phase_plugin, "prototype_and_class_statistics"):
            self.compute_class_means(strategy)
        self._pack_current_memory()
        strategy.model.train()
        stored = sum(len(labels) for labels in self.y_memory)
        if stored > self.memory_size:
            raise AssertionError(f"ICaRL stored {stored} samples above cap {self.memory_size}")

    def construct_exemplar_set(self, strategy):
        from avalanche.benchmarks.utils import _taskaware_classification_subset

        tid = strategy.clock.train_exp_counter
        benchmark = strategy.experience.benchmark
        nb_cl = benchmark.n_classes_per_exp[tid]
        previous_seen = sum(benchmark.n_classes_per_exp[:tid])
        new_classes = self.observed_classes[previous_seen : previous_seen + nb_cl]
        dataset = strategy.experience.dataset
        targets = torch.tensor(dataset.targets)
        for class_id in new_classes:
            subset = _taskaware_classification_subset(dataset, torch.where(targets == class_id)[0])
            loader = DataLoader(
                subset.eval(),
                collate_fn=getattr(subset, "collate_fn", None),
                batch_size=strategy.eval_mb_size,
            )
            patterns, features = [], []
            for class_pt, _, _ in loader:
                class_pt = class_pt.to(strategy.device)
                patterns.append(class_pt)
                with torch.no_grad():
                    features.append(strategy.model.feature_extractor(class_pt).detach())
            patterns_t = torch.cat(patterns)
            features_t = torch.cat(features)
            dmat = features_t.T
            dmat = dmat / torch.clamp(torch.norm(dmat, dim=0), min=1e-12)
            mu = torch.mean(dmat, dim=1)
            order = torch.zeros(patterns_t.shape[0], device=dmat.device)
            w_t = mu
            wanted = min(self._budget_for(class_id), patterns_t.shape[0])
            added, selected = 0, set()
            while added < wanted:
                scores = torch.mm(w_t.unsqueeze(0), dmat)
                if selected:
                    scores[0, list(selected)] = -torch.inf
                index = int(torch.argmax(scores).item())
                order[index] = 1 + added
                added += 1
                selected.add(index)
                w_t = w_t + mu - dmat[:, index]
            pick = torch.where((order > 0) & (order <= wanted))[0]
            self.x_memory.append(patterns_t[pick].detach().cpu())
            self.y_memory.append(np.full(len(pick), class_id, dtype=np.int64))
            self.order.append(order[pick].detach().cpu())

    def reduce_exemplar_set(self, strategy):
        for i, class_id in enumerate(self.observed_classes):
            if i >= len(self.x_memory):
                break
            budget = self._budget_for(class_id)
            pick = torch.where(self.order[i] <= budget)[0]
            self.x_memory[i] = self.x_memory[i][pick]
            self.order[i] = self.order[i][pick]
            self.y_memory[i] = np.asarray(self.y_memory[i], dtype=np.int64)[: len(pick)]


def math_prod(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _make_icarl_plugin(memory_size, phase_plugin, pack_binary):
    from avalanche.training.supervised.icarl import _ICaRLPlugin

    class _Plugin(InstrumentedICaRLPluginMixin, _ICaRLPlugin):
        pass

    return _Plugin(memory_size, phase_plugin, pack_binary)


@dataclass
class StrategyBundle:
    method: str
    strategy: Any
    phase_plugin: Any | None
    method_plugin: Any | None = None
    criterion_plugin: Any | None = None

    def add_manual_after_experience(self, profiler: Any, experience) -> None:
        if self.method != "cwr_star":
            return
        classes = len(experience.classes_in_this_experience)
        dim = 64
        # np.average over all current-class weights, then subtract scalar mean per weight.
        flops = 2 * classes * dim
        profiler.add_manual(
            "manual_supplement",
            flops,
            "CWR NumPy mean and mean-shift: (m*d-1 adds + 1 divide) + m*d subtracts = 2*m*d",
            {"m_current_classes": classes, "feature_dim": dim},
        )


def _optimizer(parameters, training_parameters: dict[str, Any]):
    if training_parameters["optimizer"] != "SGD":
        raise ValueError(f"Unsupported optimizer: {training_parameters['optimizer']}")
    return SGD(
        parameters,
        lr=training_parameters["learning_rate"],
        momentum=training_parameters["momentum"],
        weight_decay=training_parameters["weight_decay"],
        foreach=training_parameters["foreach"],
    )


def build_si_tuning_strategy(
    in_channels: int,
    epochs: int,
    device: torch.device,
    method_parameters: dict[str, Any],
) -> StrategyBundle:
    """构建仅供手工候选试验使用、且不启用 FLOPs 统计的 SI 策略。"""
    from avalanche.training.supervised import SynapticIntelligence

    expected = {"si_lambda", "eps"}
    if set(method_parameters) != expected:
        raise KeyError(
            "SI tuning parameters must be exactly "
            f"{sorted(expected)}, received {sorted(method_parameters)}"
        )
    si_lambda = float(method_parameters["si_lambda"])
    eps = float(method_parameters["eps"])
    if si_lambda < 0:
        raise ValueError("SI si_lambda must be non-negative")
    if eps <= 0:
        raise ValueError("SI eps must be positive")

    training_parameters = dict(TRAINING_DEFAULTS)
    model = TemporalClassifier(in_channels)
    strategy = SynapticIntelligence(
        model=model,
        optimizer=_optimizer(model.parameters(), training_parameters),
        criterion=nn.CrossEntropyLoss(),
        si_lambda=si_lambda,
        eps=eps,
        train_mb_size=training_parameters["train_mb_size"],
        train_epochs=epochs,
        eval_mb_size=training_parameters["eval_mb_size"],
        device=device,
        evaluator=None,
        eval_every=-1,
    )
    method_plugin = next(
        plugin
        for plugin in strategy.plugins
        if plugin.__class__.__name__ == "SynapticIntelligencePlugin"
    )
    return StrategyBundle("si", strategy, None, method_plugin)


def build_lwf_tuning_strategy(
    in_channels: int,
    epochs: int,
    device: torch.device,
    method_parameters: dict[str, Any],
) -> StrategyBundle:
    """构建仅供手工候选试验使用、且不启用 FLOPs 统计的 LwF 策略。"""
    from avalanche.training.supervised import LwF

    expected = {"alpha", "temperature"}
    if set(method_parameters) != expected:
        raise KeyError(
            "LwF tuning parameters must be exactly "
            f"{sorted(expected)}, received {sorted(method_parameters)}"
        )
    alpha = float(method_parameters["alpha"])
    temperature = float(method_parameters["temperature"])
    if alpha < 0:
        raise ValueError("LwF alpha must be non-negative")
    if temperature <= 0:
        raise ValueError("LwF temperature must be positive")

    training_parameters = dict(TRAINING_DEFAULTS)
    model = TemporalClassifier(in_channels)
    strategy = LwF(
        model=model,
        optimizer=_optimizer(model.parameters(), training_parameters),
        criterion=nn.CrossEntropyLoss(),
        alpha=alpha,
        temperature=temperature,
        train_mb_size=training_parameters["train_mb_size"],
        train_epochs=epochs,
        eval_mb_size=training_parameters["eval_mb_size"],
        device=device,
        evaluator=None,
        eval_every=-1,
    )
    method_plugin = next(
        plugin for plugin in strategy.plugins if plugin.__class__.__name__ == "LwFPlugin"
    )
    return StrategyBundle("lwf", strategy, None, method_plugin)


def _make_ewc_compatible_cosine_classifier(feature_dim: int):
    """Keep Avalanche cosine semantics while preserving EWC parameter names."""
    from avalanche.models import CosineIncrementalClassifier
    from avalanche.models.cosine_layer import CosineLinear

    class _EWCCosineIncrementalClassifier(CosineIncrementalClassifier):
        def __init__(self, in_features: int):
            # num_classes=0 also avoids the upstream fixed set(range(5)) state.
            super().__init__(in_features, num_classes=0)
            self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

        def adaptation(self, experience):
            new_classes = sorted(
                int(value)
                for value in set(experience.classes_in_this_experience) - self.classes
            )
            if not new_classes:
                return
            self.classes.update(new_classes)
            self.class_order.extend(new_classes)
            expanded = CosineLinear(
                self.feature_dim,
                len(self.class_order),
                sigma=True,
            ).to(self._device_anchor.device)
            if self.fc is not None:
                old_count = int(self.fc.out_features)
                with torch.no_grad():
                    expanded.weight[:old_count].copy_(self.fc.weight)
                    expanded.sigma.copy_(self.fc.sigma)
            self.fc = expanded

        def forward(self, x):
            if self.fc is None or not self.class_order:
                raise RuntimeError("Cosine classifier has not been adapted to any class")
            unmapped_logits = self.fc(x)
            mapped_logits = unmapped_logits.new_full(
                (len(unmapped_logits), max(self.class_order) + 1),
                -1000.0,
            )
            mapped_logits[:, self.class_order] = unmapped_logits
            return mapped_logits

    return _EWCCosineIncrementalClassifier(feature_dim)


def build_ewc_cosine_tuning_strategy(
    in_channels: int,
    epochs: int,
    device: torch.device,
    method_parameters: dict[str, Any],
) -> StrategyBundle:
    """构建仅供手工候选试验使用、且不启用 FLOPs 统计的 EWC+Cosine 策略。"""
    from avalanche.training.supervised import EWC

    expected = {"ewc_lambda", "mode"}
    if set(method_parameters) != expected:
        raise KeyError(
            "EWC+Cosine tuning parameters must be exactly "
            f"{sorted(expected)}, received {sorted(method_parameters)}"
        )
    ewc_lambda = float(method_parameters["ewc_lambda"])
    mode = str(method_parameters["mode"])
    if ewc_lambda < 0:
        raise ValueError("EWC+Cosine ewc_lambda must be non-negative")
    if mode != "separate":
        raise ValueError("EWC+Cosine manual candidate currently requires mode='separate'")

    class _TemporalCosineClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = TemporalBackbone(in_channels)
            self.classifier = _make_ewc_compatible_cosine_classifier(64)

        def forward(self, x):
            return self.classifier(self.feature_extractor(x))

    training_parameters = dict(TRAINING_DEFAULTS)
    model = _TemporalCosineClassifier()
    strategy = EWC(
        model=model,
        optimizer=_optimizer(model.parameters(), training_parameters),
        criterion=nn.CrossEntropyLoss(),
        ewc_lambda=ewc_lambda,
        mode=mode,
        train_mb_size=training_parameters["train_mb_size"],
        train_epochs=epochs,
        eval_mb_size=training_parameters["eval_mb_size"],
        device=device,
        evaluator=None,
        eval_every=-1,
    )
    method_plugin = next(
        plugin for plugin in strategy.plugins if plugin.__class__.__name__ == "EWCPlugin"
    )
    return StrategyBundle("ewc_cosine", strategy, None, method_plugin)


def build_mas_tuning_strategy(
    in_channels: int,
    epochs: int,
    device: torch.device,
    method_parameters: dict[str, Any],
) -> StrategyBundle:
    """构建仅供手工候选试验使用、且不启用 FLOPs 统计的原生 Avalanche MAS。"""
    from avalanche.training.supervised import MAS

    expected = {"lambda_reg", "alpha"}
    if set(method_parameters) != expected:
        raise KeyError(
            "MAS tuning parameters must be exactly "
            f"{sorted(expected)}, received {sorted(method_parameters)}"
        )
    lambda_reg = float(method_parameters["lambda_reg"])
    alpha = float(method_parameters["alpha"])
    if lambda_reg < 0:
        raise ValueError("MAS lambda_reg must be non-negative")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("MAS alpha must be in [0, 1]")

    training_parameters = dict(TRAINING_DEFAULTS)
    model = TemporalClassifier(in_channels)
    strategy = MAS(
        model=model,
        optimizer=_optimizer(model.parameters(), training_parameters),
        criterion=nn.CrossEntropyLoss(),
        lambda_reg=lambda_reg,
        alpha=alpha,
        verbose=False,
        train_mb_size=training_parameters["train_mb_size"],
        train_epochs=epochs,
        eval_mb_size=training_parameters["eval_mb_size"],
        device=device,
        evaluator=None,
        eval_every=-1,
    )
    method_plugin = next(
        plugin for plugin in strategy.plugins if plugin.__class__.__name__ == "MASPlugin"
    )
    return StrategyBundle("mas", strategy, None, method_plugin)


def build_strategy(
    method: str,
    in_channels: int,
    epochs: int,
    device: torch.device,
    dataset_name: str,
    enable_flop_accounting: bool = True,
    resolved_parameters: dict[str, Any] | None = None,
) -> StrategyBundle:
    from avalanche.models import IncrementalClassifier, TrainEvalModel
    from avalanche.training.supervised import CWRStar, EWC, Naive
    from avalanche.training.templates import SupervisedTemplate

    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    training_parameters = dict(TRAINING_DEFAULTS)
    method_parameters = dict(METHODS[method])
    for key, value in (resolved_parameters or {}).items():
        if key in training_parameters:
            training_parameters[key] = value
        elif key in method_parameters:
            method_parameters[key] = value
        elif key != "epochs_per_experience":
            raise KeyError(f"Unknown resolved parameter for {method}: {key}")
    if enable_flop_accounting:
        from .flops import make_flop_phase_plugin

        phase_plugin = make_flop_phase_plugin()
        phase_plugins = [phase_plugin]
    else:
        phase_plugin = None
        phase_plugins = []
    common = dict(
        train_mb_size=training_parameters["train_mb_size"],
        train_epochs=epochs,
        eval_mb_size=training_parameters["eval_mb_size"],
        device=device,
        evaluator=None,
        eval_every=-1,
    )
    if method == "er_ace":
        model = TemporalClassifier(in_channels)
        strategy_cls = _make_instrumented_er_ace_class()
        strategy = strategy_cls(
            model=model,
            optimizer=_optimizer(model.parameters(), training_parameters),
            criterion=nn.CrossEntropyLoss(),
            mem_size=method_parameters["memory_size"],
            batch_size_mem=method_parameters["batch_size_mem"],
            plugins=phase_plugins,
            **common,
        )
        strategy._cil_phase_plugin = phase_plugin
        return StrategyBundle(method, strategy, phase_plugin, strategy.storage_policy)
    if method == "ewc":
        model = TemporalClassifier(in_channels)
        strategy = EWC(
            model=model,
            optimizer=_optimizer(model.parameters(), training_parameters),
            criterion=nn.CrossEntropyLoss(),
            ewc_lambda=method_parameters["ewc_lambda"],
            mode=method_parameters["mode"],
            plugins=phase_plugins,
            **common,
        )
        method_plugin = next(p for p in strategy.plugins if p.__class__.__name__ == "EWCPlugin")
        return StrategyBundle(method, strategy, phase_plugin, method_plugin)
    if method == "cwr_star":
        model = TemporalClassifier(in_channels)
        strategy = CWRStar(
            model=model,
            optimizer=_optimizer(model.parameters(), training_parameters),
            criterion=nn.CrossEntropyLoss(),
            cwr_layer_name=method_parameters["cwr_layer_name"],
            plugins=phase_plugins,
            **common,
        )
        method_plugin = next(p for p in strategy.plugins if p.__class__.__name__ == "CWRStarPlugin")
        return StrategyBundle(method, strategy, phase_plugin, method_plugin)
    if method == "icarl":
        feature_extractor = TemporalBackbone(in_channels)
        classifier = IncrementalClassifier(64, initial_out_features=1)
        eval_classifier = __import__("avalanche.models", fromlist=["NCMClassifier"]).NCMClassifier(normalize=True)
        model = TrainEvalModel(feature_extractor, classifier, eval_classifier)
        loss_plugin = _make_icarl_loss(phase_plugin)
        icarl_plugin = _make_icarl_plugin(
            method_parameters["memory_size"], phase_plugin, pack_binary=dataset_name == "spike"
        )
        strategy = SupervisedTemplate(
            model=model,
            optimizer=_optimizer(
                [*feature_extractor.parameters(), *classifier.parameters()], training_parameters
            ),
            criterion=loss_plugin,
            plugins=[*phase_plugins, icarl_plugin, loss_plugin],
            **common,
        )
        return StrategyBundle(method, strategy, phase_plugin, icarl_plugin, loss_plugin)
    if method == "fecam":
        feature_extractor = TemporalBackbone(in_channels)
        # Only task 1 is optimized and it always contains three remapped
        # classes. A fixed base-session head avoids allocating unused later
        # class weights that FeCAM never consults at inference.
        train_classifier = nn.Linear(64, 3)
        eval_classifier = _make_device_safe_fecam_classifier(
            tukey=method_parameters["tukey"],
            shrinkage=method_parameters["shrinkage"],
            shrink1=method_parameters["shrink1"],
            shrink2=method_parameters["shrink2"],
            covnorm=method_parameters["covnorm"],
        )
        model = TrainEvalModel(feature_extractor, train_classifier, eval_classifier)
        fecam_plugin = _make_fecam_plugin(phase_plugin, epochs)
        strategy = Naive(
            model=model,
            optimizer=_optimizer(model.parameters(), training_parameters),
            criterion=nn.CrossEntropyLoss(),
            plugins=[*phase_plugins, fecam_plugin],
            **common,
        )
        return StrategyBundle(method, strategy, phase_plugin, fecam_plugin)
    raise AssertionError(method)
