from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DatasetBundle, build_dataset_bundle
from .flops import (
    PhaseFlopProfiler,
    profile_single_forward,
    summarize_auxiliary_nonflop_ops,
    summarize_learning_flops,
)
from .final_hyperparameters import get_final_hyperparameters, is_final_hyperparameters_locked
from .experiment_config import EXPERIMENT_ID, validate_experiment_id
from .hyperparameter_search_flops import get_hyperparameter_search_flops
from .intransigence import (
    fill_intransigence,
    find_joint_summary,
    validate_joint_artifact_match,
)
from .metrics import compute_cil_metrics, validate_summary
from .models import BACKBONES
from .output import AtomicRunArtifacts, completed_summary_path
from .registry import (
    DATASETS,
    DEFAULT_BACKBONES,
    METHODS,
    ORDER_IDS,
    ORDERS_BY_DATASET,
    SEEDS,
    TRAINING_DEFAULTS,
    dataset_dict,
    get_task_groups,
    get_task_split,
)
from .storage import compute_persistent_storage
from .strategies import StrategyBundle, build_strategy


def _resolve_hyperparameter_search_flops(
    project_root: Path,
    dataset_name: str,
    method: str,
    search_provenance: dict[str, Any] | None,
) -> int:
    provenance = search_provenance or {}
    if "hyperparameter_search_flops" in provenance:
        value = provenance["hyperparameter_search_flops"]
    elif "search_cost_file" in provenance:
        cost_path = Path(provenance["search_cost_file"])
        if not cost_path.is_absolute():
            cost_path = project_root / cost_path
        payload = json.loads(cost_path.read_text(encoding="utf-8"))
        if payload.get("dataset") != dataset_name or payload.get("method") != method:
            raise ValueError(
                f"Search-cost identity mismatch for {dataset_name}/{method}: {cost_path}"
            )
        value = payload.get("hyperparameter_search_flops")
    else:
        return get_hyperparameter_search_flops(dataset_name, method)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"hyperparameter_search_flops for {dataset_name}/{method} must be a non-negative integer"
        )
    return value


def set_determinism(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_experience(
    model, experience, device: torch.device, batch_size: int, num_workers: int
) -> float:
    loader = DataLoader(
        experience.dataset.eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            output = model(x)
            logits = output["logits"] if isinstance(output, dict) else output
            if logits.ndim != 2:
                raise ValueError(f"Expected 2D logits, received {tuple(logits.shape)}")
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == y).sum().item())
            total += int(y.numel())
    if was_training:
        model.train()
    if total == 0:
        raise ValueError("Empty evaluation experience")
    return correct / total


def _measure_latency(
    model, experiences, device: torch.device, batch_size: int, num_workers: int
) -> float:
    was_training = model.training
    model.eval()
    first_batch = next(
        iter(
            DataLoader(
                experiences[0].dataset.eval(),
                batch_size=min(batch_size, len(experiences[0].dataset)),
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
        )
    )
    warm_x = first_batch[0].to(device)
    with torch.no_grad():
        for _ in range(3):
            model(warm_x)
    _sync(device)
    total_samples = 0
    elapsed = 0.0
    with torch.no_grad():
        for exp in experiences:
            loader = DataLoader(
                exp.dataset.eval(),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
            for batch in loader:
                x = batch[0].to(device)
                _sync(device)
                start = time.perf_counter()
                model(x)
                _sync(device)
                elapsed += time.perf_counter() - start
                total_samples += len(x)
    if was_training:
        model.train()
    if total_samples == 0:
        raise ValueError("No samples for latency measurement")
    return 1000.0 * elapsed / total_samples


def _fvcore_crosscheck(model, sample: torch.Tensor) -> dict[str, Any]:
    from fvcore.nn import FlopCountAnalysis

    analysis = FlopCountAnalysis(model, (sample,))
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    raw = int(analysis.total())
    return {
        "raw_fvcore_fma_as_one_count": raw,
        "unsupported_ops": {str(k): int(v) for k, v in analysis.unsupported_ops().items()},
        "uncalled_modules": sorted(analysis.uncalled_modules()),
        "note": "Cross-check only; formal accounting uses torch.utils.flop_counter.FlopCounterMode.",
    }


def make_config(
    dataset_name: str,
    method: str,
    order_id: int,
    epochs: int,
    resolved_hyperparameters: dict[str, Any],
    hyperparameter_overrides: dict[str, Any] | None = None,
    search_provenance: dict[str, Any] | None = None,
    backbone_id: str | None = None,
    input_view_id: str | None = None,
) -> dict[str, Any]:
    spec = DATASETS[dataset_name]
    task_groups = get_task_groups(dataset_name, order_id)
    class_split = get_task_split(dataset_name, order_id)
    effective_sgd_epochs = [int(epochs)] * len(task_groups)
    if method == "tagfex":
        effective_sgd_epochs = [
            int(resolved_hyperparameters["init_epochs"]),
            *(
                [int(resolved_hyperparameters["inc_epochs"])]
                * (len(task_groups) - 1)
            ),
        ]
    elif method == "fecam":
        effective_sgd_epochs[1:] = [0] * (len(task_groups) - 1)
    resolved_backbone_id = backbone_id or DEFAULT_BACKBONES[method]
    backbone_spec = BACKBONES[resolved_backbone_id]
    backbone = {
        "backbone_id": resolved_backbone_id,
        "feature_dim": backbone_spec.feature_dim,
        "feature_dim_per_branch": (
            backbone_spec.feature_dim if method == "tagfex" else None
        ),
        "base_width": backbone_spec.base_width,
        "stage_channels": list(backbone_spec.stage_channels),
        "pretrained": backbone_spec.pretrained,
        "status": (
            "source_structure_image_tagfex_port"
            if method == "tagfex"
            else "registered_shared_backbone"
        ),
    }
    return {
        "schema": "metrics1.docx-compatible-avalanche-cil-v2",
        "dataset": dataset_dict(spec),
        "method": method,
        "method_parameters": {
            key: resolved_hyperparameters.get(key, value)
            for key, value in METHODS[method].items()
        },
        "backbone": backbone,
        "training": {
            **{
                key: resolved_hyperparameters[key]
                for key in TRAINING_DEFAULTS
            },
            "requested_epochs_per_experience": int(epochs),
            "effective_sgd_epochs_per_experience": effective_sgd_epochs,
            "fecam_later_experience_loop": (
                "one frozen-feature class-statistics pass" if method == "fecam" else None
            ),
        },
        "protocol": {
            "augmentation": "none",
            "normalization": "none",
            "input_view_id": input_view_id,
            "technical_input_conversion": (
                "raw binary [T,C] -> [C,T] persistence view -> runtime RGB 160x160 at model boundary"
                if dataset_name == "spike"
                else "stored RGB float32 image view"
            ),
            "classes_per_experience": list(class_split),
            "replay_budget_cap_samples": 2000,
            "spike_replay_persistence": (
                "selected samples are stored as true 1-bit packed arrays; uint8/float32 tensors exist only while replay is materialized"
                if dataset_name == "spike" and method in {"er_ace", "icarl", "tagfex"}
                else None
            ),
            "spike_replay_label_persistence": (
                "uint8 one-hot using one byte per class entry"
                if dataset_name == "spike" and method in {"er_ace", "icarl", "tagfex"}
                else None
            ),
            "flop_backend": "torch.utils.flop_counter.FlopCounterMode",
            "flop_convention": "1 MAC = 2 FLOPs; no post-hoc doubling of PyTorch 2.11 totals",
            "tagfex_two_view_policy": (
                "two value-identical views because protocol augmentation is none; replay is merged "
                "before the student forward and the inseparable mixed optimization step remains in core_training"
                if method == "tagfex" else None
            ),
        },
        "order_registry_source": "cil_experiments/order_seed_registry.py",
        "selected_task_groups": [list(group) for group in task_groups],
        "dataset_orders": {
            str(key): [list(group) for group in groups]
            for key, groups in ORDERS_BY_DATASET[dataset_name].items()
        },
        "seeds": list(SEEDS),
        "final_hyperparameters": {
            "source": "cil_experiments/final_hyperparameters.py",
            "registry_locked": is_final_hyperparameters_locked(method),
            "resolved": dict(resolved_hyperparameters),
            "explicit_diagnostic_overrides": dict(hyperparameter_overrides or {}),
        },
        "hyperparameter_search": {
            "status": (
                "explicit_diagnostic_hyperparameter_override"
                if hyperparameter_overrides
                else "user_locked_final_hyperparameter_registry"
            ),
            "required_trials": 3,
            "test_set_used_for_selection": False,
            **(search_provenance or {}),
        },
        "summary_null_reasons": {
            **(
                {}
                if method == "tagfex"
                else {
                    "network.total_reused_neurons":
                    "No comparable explicit neuron-reuse mechanism."
                }
            ),
            "training_runtime.total_gpu_ms": "null only for CPU execution.",
            "working_memory_diagnostic.native_peak_allocated_gpu_memory_mib": "null only for CPU execution.",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "avalanche": __import__("avalanche").__version__,
        },
    }


def _train_experience(
    bundle: StrategyBundle,
    experience,
    device: torch.device,
    num_workers: int = 0,
) -> tuple[Any, float, float | None, float | None]:
    profiler = PhaseFlopProfiler()
    bundle.phase_plugin.attach(profiler)
    gpu_event_start = gpu_event_end = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_event_start = torch.cuda.Event(enable_timing=True)
        gpu_event_end = torch.cuda.Event(enable_timing=True)
        gpu_event_start.record()
    _sync(device)
    wall_start = time.perf_counter()
    profiler.start()
    try:
        bundle.strategy.train(
            experience,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        bundle.add_manual_after_experience(profiler, experience)
        flop_result = profiler.stop(strict=True)
    except BaseException:
        profiler.abort()
        raise
    finally:
        bundle.phase_plugin.detach()
    _sync(device)
    wall_s = time.perf_counter() - wall_start
    gpu_ms = None
    peak_mib = None
    if device.type == "cuda":
        gpu_event_end.record()
        torch.cuda.synchronize(device)
        gpu_ms = float(gpu_event_start.elapsed_time(gpu_event_end))
        peak_mib = float(torch.cuda.max_memory_allocated(device) / (2**20))
    return flop_result, wall_s, gpu_ms, peak_mib


def run_one(
    project_root: Path,
    dataset_root: Path,
    result_root: Path,
    dataset_name: str,
    method: str,
    order_id: int,
    seed: int,
    epochs: int | None,
    device: torch.device,
    overwrite: bool = False,
    search_provenance: dict[str, Any] | None = None,
    parameter_overrides: dict[str, Any] | None = None,
    backbone_id: str | None = None,
    compute_intransigence_enabled: bool = True,
    joint_summary_path: Path | None = None,
    experiment_id: int = EXPERIMENT_ID,
    resume: bool = False,
) -> Path:
    experiment_id = validate_experiment_id(experiment_id)
    if seed not in SEEDS:
        raise ValueError(f"Seed {seed} is not in the locked seed registry")
    if order_id not in ORDER_IDS:
        raise ValueError(f"Order {order_id} is not in the locked order registry")
    resolved_overrides = dict(parameter_overrides or {})
    if epochs is not None:
        resolved_overrides["epochs_per_experience"] = int(epochs)
    resolved_hyperparameters = get_final_hyperparameters(
        dataset_name,
        method,
        resolved_overrides or None,
        require_locked=not bool(resolved_overrides),
    )
    epochs = int(resolved_hyperparameters["epochs_per_experience"])
    num_workers = int(resolved_hyperparameters["num_workers"])
    set_determinism(seed)
    resolved_backbone_id = backbone_id or DEFAULT_BACKBONES[method]
    if resolved_backbone_id not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {resolved_backbone_id!r}; registered={sorted(BACKBONES)}"
        )
    input_view = "image"
    data = build_dataset_bundle(
        dataset_root, DATASETS[dataset_name], order_id, input_view=input_view
    )
    hyperparameter_search_flops = _resolve_hyperparameter_search_flops(
        project_root, dataset_name, method, search_provenance
    )
    resolved_search_provenance = dict(search_provenance or {})
    resolved_search_provenance["hyperparameter_search_flops"] = hyperparameter_search_flops
    config = make_config(
        dataset_name,
        method,
        order_id,
        epochs,
        resolved_hyperparameters,
        resolved_overrides or None,
        resolved_search_provenance,
        resolved_backbone_id,
        data.input_view_id,
    )
    config["experiment_id"] = experiment_id
    matched_joint_summary = None
    if compute_intransigence_enabled:
        matched_joint_summary = (
            joint_summary_path.resolve()
            if joint_summary_path is not None
            else find_joint_summary(
                result_root,
                dataset=dataset_name,
                paired_method=method,
                order_id=order_id,
                seed=seed,
                experiment_id=experiment_id,
            )
        )
    config["intransigence"] = {
        "automatic_fill_enabled": bool(compute_intransigence_enabled),
        "joint_summary_path": (
            str(matched_joint_summary) if matched_joint_summary is not None else None
        ),
        "pending_results_are_formal_aggregation_eligible": False,
    }
    if not compute_intransigence_enabled:
        config["summary_null_reasons"]["cil_performance.intransigence"] = (
            "Automatic Joint matching was explicitly disabled with --skip-intransigence."
        )
    elif matched_joint_summary is not None:
        validate_joint_artifact_match(
            {
                "dataset": dataset_name,
                "method": method,
                "order_id": int(order_id),
                "seed": int(seed),
                "tasks": int(data.tasks),
                "experiment_id": experiment_id,
            },
            config,
            matched_joint_summary,
        )
    if resume and not overwrite:
        completed = completed_summary_path(
            result_root,
            dataset_name,
            method,
            order_id,
            seed,
            experiment_id,
            expected_config=config,
        )
        if completed is not None:
            print(
                "SKIP completed "
                f"exp={experiment_id} dataset={dataset_name} method={method} "
                f"order={order_id} seed={seed}: {completed}"
            )
            return completed

    with AtomicRunArtifacts(
        result_root,
        dataset_name,
        method,
        order_id,
        seed,
        config,
        overwrite,
        experiment_id,
    ) as artifacts:
        log = artifacts.logger
        assert log is not None
        log.info("run=%s dataset=%s method=%s order_id=%d seed=%d epochs=%d device=%s", artifacts.summary_path.stem, dataset_name, method, order_id, seed, epochs, device)
        log.info("data_contract=%s", json.dumps(dataset_dict(data.spec), ensure_ascii=False, sort_keys=True))
        log.info("raw_order=%s", list(data.raw_order))
        log.info("label_map=%s inverse_label_map=%s", data.label_map, data.inverse_label_map)
        log.info(
            "final_hyperparameters_source=cil_experiments/final_hyperparameters.py resolved=%s",
            json.dumps(resolved_hyperparameters, ensure_ascii=False, sort_keys=True),
        )
        log.info("resolved_config=%s", json.dumps(artifacts.config, ensure_ascii=False, sort_keys=True))
        log.info(
            "flop_accounting_contract=%s",
            json.dumps(
                {
                    "primary_backend": "torch.utils.flop_counter.FlopCounterMode",
                    "secondary_backend": "fvcore.nn.FlopCountAnalysis",
                    "convention": "1 MAC = 2 FLOPs; PyTorch 2.11 convolution and matrix formulas are not doubled again",
                    "scope": "all actually executed train forwards, replay, teacher/distillation, loss/regularization, backward, optimizer updates, selection, prototypes/class statistics, method-specific operations, and manual non-dispatch formulas",
                    "phase_exclusivity": "training emits only core_training and learning_auxiliary; ER-ACE's separate replay forward is auxiliary, while mixed replay optimization remains core",
                    "unknown_policy": "any unregistered computational leaf operator aborts the run before publication",
                },
                sort_keys=True,
            ),
        )
        log.info(
            "explicit_zero_flop_policy=%s",
            json.dumps(
                {
                    "comparison_logic_sort_index": "comparisons, logic, sorting, indexing and scatter/gather are classified as non-floating-point work",
                    "shape_view_copy_transfer": "reshape/view/transpose, allocation, clone/copy and device transfer are data movement or metadata operations",
                    "random_and_profiler_markers": "random-state generation and profiler record markers are not floating-point arithmetic",
                    "activation_policy": "ReLU/threshold and clamp are comparisons under this protocol and therefore zero FLOPs",
                    "bit_unpacking": "NumPy bit packing/unpacking is integer/data-movement work and is excluded from FLOPs but retained in memory semantics",
                },
                sort_keys=True,
            ),
        )
        bundle = build_strategy(
            method,
            data.spec.in_channels,
            epochs,
            device,
            dataset_name,
            resolved_parameters=resolved_hyperparameters,
            backbone_id=resolved_backbone_id,
            model_input_shape=data.model_input_shape,
        )
        matrix = np.full((data.tasks, data.tasks), np.nan, dtype=np.float64)
        task_wall: list[float] = []
        task_gpu: list[float] = []
        task_peaks: list[float] = []
        task_flops = []
        terminal_values: list[float] = []

        for task_index, experience in enumerate(data.benchmark.train_stream):
            result, wall_s, gpu_ms, peak_mib = _train_experience(
                bundle, experience, device, num_workers
            )
            task_wall.append(wall_s)
            if gpu_ms is not None:
                task_gpu.append(gpu_ms)
            if peak_mib is not None:
                task_peaks.append(peak_mib)
            task_flops.append(result)
            terminal_values.append(result.terminal_flops_per_sample)
            for test_index in range(task_index + 1):
                matrix[task_index, test_index] = _evaluate_experience(
                    bundle.strategy.model,
                    data.benchmark.test_stream[test_index],
                    device,
                    int(resolved_hyperparameters["eval_mb_size"]),
                    num_workers,
                )
            if not np.isfinite(matrix[task_index, : task_index + 1]).all():
                raise FloatingPointError(f"Non-finite accuracy after task {task_index + 1}")
            log.info(
                "task=%d wall_s=%.6f gpu_ms=%s peak_mib=%s accuracies=%s flops=%s",
                task_index + 1,
                wall_s,
                gpu_ms,
                peak_mib,
                matrix[task_index, : task_index + 1].tolist(),
                json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True),
            )

        model = bundle.strategy.model
        model.eval()
        sample_x = data.test[0][0].unsqueeze(0).to(device)
        single_forward_flops, single_detail = profile_single_forward(model, sample_x)
        fvcore_check = _fvcore_crosscheck(model, sample_x)
        latency = _measure_latency(
            model,
            list(data.benchmark.test_stream),
            device,
            int(resolved_hyperparameters["eval_mb_size"]),
            num_workers,
        )
        storage = compute_persistent_storage(bundle)
        cil = compute_cil_metrics(matrix, data.test_samples_per_task)
        total_s = float(sum(task_wall))
        incremental = task_wall[1:]
        flop_summary = summarize_learning_flops(
            task_flops,
            single_forward_flops,
        )
        summary = {
            "method": METHODS[method]["display_name"],
            "seed": int(seed),
            "tasks": int(data.tasks),
            "cil_performance": cil,
            "network": bundle.network_summary(data.tasks),
            "training_runtime": {
                "total_s": total_s,
                "total_gpu_ms": float(sum(task_gpu)) if task_gpu else None,
                "initial_task_s": float(task_wall[0]),
                "incremental_tasks_total_s": float(sum(incremental)),
                "mean_incremental_task_s": float(statistics.mean(incremental)),
                "median_incremental_task_s": float(statistics.median(incremental)),
            },
            "training_operations": {
                "task_summed_terminal_flops_per_sample": float(sum(terminal_values)),
                **flop_summary,
                "auxiliary_nonflop_ops": summarize_auxiliary_nonflop_ops(task_flops),
            },
            "persistent_storage": storage,
            "inference": {"final_latency_ms_per_sample": float(latency)},
            "working_memory_diagnostic": {
                "native_peak_allocated_gpu_memory_mib": float(max(task_peaks)) if task_peaks else None,
                "note": "Transient working memory; excluded from persistent storage.",
            },
        }
        validate_summary(summary, allow_pending_intransigence=True)
        log.info("accuracy_matrix=%s", matrix.tolist())
        log.info("single_sample_forward_flops=%d detail=%s", single_forward_flops, json.dumps(single_detail, ensure_ascii=False, sort_keys=True))
        log.info("fvcore_crosscheck=%s", json.dumps(fvcore_check, ensure_ascii=False, sort_keys=True))
        log.info("persistent_storage=%s", json.dumps(storage, ensure_ascii=False, sort_keys=True))
        log.info("summary=%s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
        lower_triangular = [
            [float(matrix[row, col]) if col <= row else None for col in range(data.tasks)]
            for row in range(data.tasks)
        ]
        accuracy_matrix = {
            "dataset": dataset_name,
            "method": method,
            "backbone_id": resolved_backbone_id,
            "input_view_id": data.input_view_id,
            "order_id": int(order_id),
            "seed": int(seed),
            "experiment_id": experiment_id,
            "tasks": int(data.tasks),
            "orientation": "row=train experience end; column=evaluated test experience",
            "upper_triangle": "null because the corresponding class experience had not been learned yet",
            "test_samples_per_task": [int(value) for value in data.test_samples_per_task],
            "accuracy_matrix_lower_triangular": lower_triangular,
        }
        artifacts.commit(summary, accuracy_matrix)
        if matched_joint_summary is not None:
            values = fill_intransigence(artifacts.summary_path, matched_joint_summary)
            log.info(
                "intransigence_joint=%s values=%s",
                matched_joint_summary,
                values,
            )
        return artifacts.summary_path
