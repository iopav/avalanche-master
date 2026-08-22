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
from .flops import PhaseFlopProfiler, profile_single_forward
from .final_hyperparameters import (
    FINAL_HYPERPARAMETERS_LOCKED,
    final_hyperparameter_hash,
    get_final_hyperparameters,
)
from .metrics import compute_cil_metrics, validate_summary
from .output import AtomicRunArtifacts
from .registry import (
    BACKBONE_CONFIG,
    DATASETS,
    METHODS,
    ORDERS,
    SEEDS,
    TRAINING_DEFAULTS,
    dataset_dict,
    sha256_file,
    validate_order_seed_file,
)
from .storage import compute_persistent_storage
from .strategies import StrategyBundle, build_strategy


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
            logits = model(x)
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


def _source_hashes(project_root: Path, method: str) -> dict[str, str]:
    sources = {
        "avalanche_version": project_root / "avalanche" / "__init__.py",
        "er_ace": project_root / "avalanche" / "training" / "supervised" / "er_ace.py",
        "icarl": project_root / "avalanche" / "training" / "supervised" / "icarl.py",
        "strategy_wrappers": project_root / "avalanche" / "training" / "supervised" / "strategy_wrappers.py",
        "fecam": project_root / "avalanche" / "models" / "fecam.py",
        "fecam_update": project_root / "avalanche" / "training" / "plugins" / "update_fecam.py",
    }
    sources["selected_method_marker"] = sources[
        "er_ace" if method == "er_ace" else "icarl" if method == "icarl" else "fecam" if method == "fecam" else "strategy_wrappers"
    ]
    hashes = {key: sha256_file(path) for key, path in sources.items()}
    experiment_package = project_root / "cil_experiments"
    for path in sorted(experiment_package.glob("*.py")):
        hashes[f"cil_experiments/{path.name}"] = sha256_file(path)
    return hashes


def make_config(
    project_root: Path,
    dataset_name: str,
    method: str,
    epochs: int,
    resolved_hyperparameters: dict[str, Any],
    hyperparameter_overrides: dict[str, Any] | None = None,
    search_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = DATASETS[dataset_name]
    effective_sgd_epochs = [int(epochs)] * spec.tasks
    if method == "fecam":
        effective_sgd_epochs[1:] = [0] * (spec.tasks - 1)
    return {
        "schema": "metrics1.docx-compatible-avalanche-cil-v1",
        "dataset": dataset_dict(spec),
        "method": method,
        "method_parameters": {
            key: resolved_hyperparameters.get(key, value)
            for key, value in METHODS[method].items()
        },
        "backbone": {"in_channels": spec.in_channels, **BACKBONE_CONFIG},
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
            "technical_input_conversion": "source dtype -> float32; [T,C] -> [C,T]",
            "first_experience_classes": 3,
            "later_experience_classes": 1,
            "replay_budget_cap_samples": 2000,
            "flop_backend": "torch.utils.flop_counter.FlopCounterMode",
            "flop_convention": "1 MAC = 2 FLOPs; no post-hoc doubling of PyTorch 2.11 totals",
        },
        "orders": {str(k): list(v) for k, v in ORDERS.items()},
        "seeds": list(SEEDS),
        "final_hyperparameters": {
            "source": "cil_experiments/final_hyperparameters.py",
            "registry_locked": FINAL_HYPERPARAMETERS_LOCKED,
            "resolved_entry_hash": final_hyperparameter_hash(resolved_hyperparameters),
            "resolved": dict(resolved_hyperparameters),
            "explicit_diagnostic_overrides": dict(hyperparameter_overrides or {}),
        },
        "hyperparameter_search": search_provenance or {
            "status": (
                "explicit_diagnostic_hyperparameter_override"
                if hyperparameter_overrides
                else "user_locked_final_hyperparameter_registry"
            ),
            "required_trials": 3,
            "test_set_used_for_selection": False,
        },
        "summary_null_reasons": {
            "network.total_reused_neurons": "No comparable explicit neuron-reuse mechanism.",
            "training_runtime.total_gpu_ms": "null only for CPU execution.",
            "working_memory_diagnostic.native_peak_allocated_gpu_memory_mib": "null only for CPU execution.",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "avalanche": __import__("avalanche").__version__,
        },
        "source_sha256": _source_hashes(project_root, method),
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
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Seed {seed} is not in the locked seed registry")
    if order_id not in ORDERS:
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
    order_file = project_root.parent / "5order10seeds.txt"
    validate_order_seed_file(order_file)
    data = build_dataset_bundle(dataset_root, DATASETS[dataset_name], order_id)
    config = make_config(
        project_root,
        dataset_name,
        method,
        epochs,
        resolved_hyperparameters,
        resolved_overrides or None,
        search_provenance,
    )

    with AtomicRunArtifacts(result_root, dataset_name, method, order_id, seed, config, overwrite) as artifacts:
        log = artifacts.logger
        assert log is not None
        log.info("run=%s dataset=%s method=%s order_id=%d seed=%d epochs=%d device=%s", artifacts.summary_path.stem, dataset_name, method, order_id, seed, epochs, device)
        log.info("data_contract=%s", json.dumps(dataset_dict(data.spec), ensure_ascii=False, sort_keys=True))
        log.info("raw_order=%s", list(data.raw_order))
        log.info("label_map=%s inverse_label_map=%s", data.label_map, data.inverse_label_map)
        log.info("config_hash=%s", artifacts.config["config_hash"])
        log.info(
            "final_hyperparameters_source=cil_experiments/final_hyperparameters.py entry_hash=%s resolved=%s",
            final_hyperparameter_hash(resolved_hyperparameters),
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
                    "phase_exclusivity": "one active phase per dispatched operator; manual rules added once outside dispatch",
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
        )
        matrix = np.full((data.spec.tasks, data.spec.tasks), np.nan, dtype=np.float64)
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
        summary = {
            "method": METHODS[method]["display_name"],
            "seed": int(seed),
            "tasks": int(data.spec.tasks),
            "cil_performance": cil,
            "network": {
                "final_hidden_neurons": 64,
                "total_new_neurons": 0,
                "total_reused_neurons": None,
            },
            "training_runtime": {
                "total_s": total_s,
                "total_gpu_ms": float(sum(task_gpu)) if task_gpu else None,
                "initial_task_s": float(task_wall[0]),
                "incremental_tasks_total_s": float(sum(incremental)),
                "mean_incremental_task_s": float(statistics.mean(incremental)),
                "median_incremental_task_s": float(statistics.median(incremental)),
            },
            "training_operations": {
                "estimated_cumulative_dense_flops": int(sum(item.total_flops for item in task_flops)),
                "task_summed_terminal_flops_per_sample": float(sum(terminal_values)),
            },
            "persistent_storage": storage,
            "inference": {"final_latency_ms_per_sample": float(latency)},
            "working_memory_diagnostic": {
                "native_peak_allocated_gpu_memory_mib": float(max(task_peaks)) if task_peaks else None,
                "note": "Transient working memory; excluded from persistent storage.",
            },
        }
        validate_summary(summary)
        log.info("accuracy_matrix=%s", matrix.tolist())
        log.info("single_sample_forward_flops=%d detail=%s", single_forward_flops, json.dumps(single_detail, ensure_ascii=False, sort_keys=True))
        log.info("fvcore_crosscheck=%s", json.dumps(fvcore_check, ensure_ascii=False, sort_keys=True))
        log.info("persistent_storage=%s", json.dumps(storage, ensure_ascii=False, sort_keys=True))
        log.info("summary=%s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
        lower_triangular = [
            [float(matrix[row, col]) if col <= row else None for col in range(data.spec.tasks)]
            for row in range(data.spec.tasks)
        ]
        accuracy_matrix = {
            "dataset": dataset_name,
            "method": method,
            "order_id": int(order_id),
            "seed": int(seed),
            "timestamp": artifacts.timestamp,
            "tasks": int(data.spec.tasks),
            "orientation": "row=train experience end; column=evaluated test experience",
            "upper_triangle": "null because the corresponding class experience had not been learned yet",
            "test_samples_per_task": [int(value) for value in data.test_samples_per_task],
            "accuracy_matrix_lower_triangular": lower_triangular,
        }
        artifacts.commit(summary, accuracy_matrix)
        return artifacts.summary_path
