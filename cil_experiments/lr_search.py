from __future__ import annotations

import json
import math
import statistics
import traceback
from pathlib import Path
from typing import Any

import torch

from .checkpointing import promote_search_checkpoint, weight_checkpoint_path
from .final_hyperparameters import get_final_hyperparameters
from .metrics import validate_summary
from .order_seed_registry import ORDERS_BY_DATASET, SEARCH_SEED
from .output import atomic_write_json, completed_summary_path
from .runner import run_experiment
from .search_config import ACC_TOLERANCE, LR_CANDIDATES, SEARCH_EPOCHS
from .search_schema import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    new_search_result,
    search_result_path,
    timestamp,
)
from .validation import validation_split_path


def optimum_lr_path(search_root: Path) -> Path:
    return Path(search_root) / "optimum_lrs.json"


def _lr_filename_token(lr: float) -> str:
    return str(lr).replace(".", "")


def candidate_checkpoint_path(
    search_root: Path,
    dataset: str,
    method: str,
    order_id: int,
    lr: float,
) -> Path:
    # 不再在每个 order/lr artifact 目录中建立 checkpoint 子目录：
    # / f"order-{order_id:02d}" / f"lr-{lr}" / "checkpoint"
    return (
        Path(search_root) / dataset / method / "checkpoint_candidates"
        / f"{dataset}__{method}__order-{order_id:02d}__seed-{SEARCH_SEED:03d}__lr-{lr}.pt"
    )


def best_checkpoint_path(
    search_root: Path, dataset: str, method: str, order_id: int, lr: float
) -> Path:
    lr_token = _lr_filename_token(lr)
    return (
        Path(search_root) / dataset / method / "checkpoint"
        / f"{dataset}__{method}__order-{order_id:02d}__seed-{SEARCH_SEED:03d}__lr-{lr_token}__best.pt"
    )


def selected_checkpoint(payload: dict[str, Any], order_id: int) -> Path:
    value = payload.get("best_checkpoint_per_order", {}).get(f"order{order_id}")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Search result does not record a best checkpoint for order{order_id}")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing best search checkpoint: {path}")
    if not weight_checkpoint_path(path).is_file():
        raise FileNotFoundError(f"Missing joint weight checkpoint: {weight_checkpoint_path(path)}")
    return path


def candidate_parameters(dataset: str, method: str, lr: float) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "learning_rate": float(lr),
        "epochs_per_experience": SEARCH_EPOCHS,
    }
    if method == "tagfex":
        overrides.update(
            init_lr=float(lr), inc_lr=float(lr),
            init_epochs=SEARCH_EPOCHS, inc_epochs=SEARCH_EPOCHS,
        )
    return get_final_hyperparameters(dataset, method, overrides)


def formal_parameters(dataset: str, method: str, lr: float) -> dict[str, Any]:
    overrides: dict[str, Any] = {"learning_rate": float(lr)}
    if method == "tagfex":
        overrides.update(init_lr=float(lr), inc_lr=float(lr))
    return get_final_hyperparameters(dataset, method, overrides)


def _candidate_record(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_summary(summary, allow_pending_intransigence=True)
    cil = summary["cil_performance"]
    operations = summary["training_operations"]
    storage = summary["persistent_storage"]
    return {
        "status": STATUS_COMPLETED,
        "summary_file": str(summary_path.resolve()),
        "val_avg_acc": float(cil["final_average_accuracy"]),
        "val_average_incremental_accuracy": float(
            cil["average_incremental_accuracy"]
        ),
        "overall_learning_flops": int(operations["overall_learning_flops"]),
        "storage_param_bytes": int(storage["model_parameter_bytes"]),
        "replay_sample_label_bytes": int(
            storage["replay_sample_bytes"] + storage["replay_label_bytes"]
        ),
        "auxiliary_bytes": int(storage["auxiliary_bytes"]),
        "total_bytes": int(storage["total_bytes"]),
    }


def select_best(runs: dict[str, dict[str, Any]]) -> float:
    completed = [
        (float(lr), record)
        for lr, record in runs.items()
        if record.get("status") == STATUS_COMPLETED
    ]
    if len(completed) != len(LR_CANDIDATES):
        raise RuntimeError("Cannot select LR before every candidate has completed")
    highest = max(record["val_avg_acc"] for _, record in completed)
    close = [
        (lr, record) for lr, record in completed
        if highest - record["val_avg_acc"] <= ACC_TOLERANCE
    ]
    return min(
        close,
        key=lambda item: (
            item[1]["overall_learning_flops"], LR_CANDIDATES.index(item[0])
        ),
    )[0]


def _mean_ci95(values: list[int]) -> tuple[float, float]:
    mean = float(statistics.fmean(values))
    if len(values) < 2:
        return mean, 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182}.get(len(values), 1.96)
    return mean, float(critical * statistics.stdev(values) / math.sqrt(len(values)))


def finalize_search(payload: dict[str, Any], order_ids: tuple[int, ...]) -> None:
    best: list[float] = []
    order_flops: list[int] = []
    order_storage: list[int] = []
    for order_id in order_ids:
        runs = payload["search_runs"][f"order{order_id}"]
        lr = select_best(runs)
        best.append(lr)
        order_flops.append(sum(int(run["overall_learning_flops"]) for run in runs.values()))
        selected = runs[str(lr)]
        order_storage.append(int(selected["storage_param_bytes"]))
    flops_mean, flops_ci95 = _mean_ci95(order_flops)
    storage_mean, storage_ci95 = _mean_ci95(order_storage)
    payload["best_per_order"] = best
    payload["search_metrics"] = {
        "search_flops_each_order": order_flops,
        "search_flops_mean": flops_mean,
        "search_flops_ci95": flops_ci95,
        "search_storage_each_order": order_storage,
        "search_storage_mean": storage_mean,
        "search_storage_ci95": storage_ci95,
    }
    payload["metadata"].update(status=STATUS_COMPLETED, finished_at=timestamp())


def _load_or_create(
    path: Path,
    *,
    exp_name: str,
    dataset: str,
    method: str,
    backbone_id: str,
    split_path: Path,
    order_ids: tuple[int, ...],
) -> dict[str, Any]:
    if not path.exists():
        payload = new_search_result(
            exp_name=exp_name,
            dataset=dataset,
            method=method,
            validation_split_file=split_path,
            search_seed=SEARCH_SEED,
            order_ids=order_ids,
        )
        payload["metadata"].update(
            search_epochs=SEARCH_EPOCHS, backbone_id=backbone_id
        )
        atomic_write_json(path, payload)
        return payload
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    expected = {
        "exp_name": exp_name,
        "dataset": dataset,
        "method": method,
        "backbone_id": backbone_id,
        "search_seed": SEARCH_SEED,
        "lr_candidates": list(LR_CANDIDATES),
        "acc_tolerance": ACC_TOLERANCE,
        "search_storage_definition": "selected_model_parameter_bytes",
        "validation_split_file": str(split_path.resolve()),
        "search_epochs": SEARCH_EPOCHS,
    }
    differences = {
        key: (metadata.get(key), value)
        for key, value in expected.items() if metadata.get(key) != value
    }
    if differences:
        raise ValueError(f"Existing search config differs: {differences}")
    if set(payload.get("search_runs", {})) != {
        f"order{order_id}" for order_id in order_ids
    }:
        raise ValueError("Existing search order set differs from the dataset registry")
    return payload


def save_optimum_lrs(
    search_root: Path, dataset: str, method: str, best_per_order: list[float]
) -> Path:
    path = optimum_lr_path(search_root)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.setdefault(dataset, {})[method] = list(best_per_order)
    atomic_write_json(path, payload)
    return path


def _completed_candidate(
    search_root: Path,
    exp_name: str,
    dataset: str,
    method: str,
    order_id: int,
    lr: float,
) -> Path:
    path = completed_summary_path(
        search_root,
        dataset,
        method,
        order_id,
        SEARCH_SEED,
        exp_name,
        run_subdir=Path(f"order-{order_id:02d}") / f"lr-{lr}",
    )
    if path is None:
        raise RuntimeError(
            f"Search JSON marks a missing candidate complete: order{order_id}/lr={lr}"
        )
    return path


def run_search(
    project_root: Path,
    dataset_root: Path,
    search_root: Path,
    exp_name: str,
    dataset: str,
    method: str,
    device: torch.device,
    backbone_id: str,
) -> Path:
    split_path = validation_split_path(dataset_root, dataset)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Missing validation split: {split_path}. Run main_exp/prepare_validation_splits.py first."
        )
    order_ids = tuple(sorted(ORDERS_BY_DATASET[dataset]))
    path = search_result_path(search_root, dataset, method)
    payload = _load_or_create(
        path,
        exp_name=exp_name,
        dataset=dataset,
        method=method,
        backbone_id=backbone_id,
        split_path=split_path,
        order_ids=order_ids,
    )
    if payload["metadata"]["status"] == STATUS_COMPLETED:
        for order_id in order_ids:
            for lr in LR_CANDIDATES:
                _completed_candidate(
                    search_root, exp_name, dataset, method, order_id, lr
                )
            selected_checkpoint(payload, order_id)
        save_optimum_lrs(search_root, dataset, method, payload["best_per_order"])
        return path

    for order_id in order_ids:
        order_runs = payload["search_runs"][f"order{order_id}"]
        for lr in LR_CANDIDATES:
            key = str(lr)
            parameters = candidate_parameters(dataset, method, lr)
            serialized_parameters = json.loads(json.dumps(parameters))
            if order_runs[key].get("status") == STATUS_COMPLETED:
                _completed_candidate(
                    search_root, exp_name, dataset, method, order_id, lr
                )
                if order_runs[key].get("hyperparameters") != serialized_parameters:
                    raise ValueError(
                        f"Completed search parameters differ for order{order_id}/lr={lr}"
                    )
                continue
            try:
                checkpoint = candidate_checkpoint_path(
                    search_root, dataset, method, order_id, lr
                )
                summary_path = run_experiment(
                    project_root=project_root,
                    dataset_root=dataset_root,
                    result_root=search_root,
                    dataset_name=dataset,
                    method=method,
                    order_id=order_id,
                    seed=SEARCH_SEED,
                    epochs=None,
                    device=device,
                    parameter_overrides=parameters,
                    search_provenance={"hyperparameter_search_flops": 0},
                    backbone_id=backbone_id,
                    compute_intransigence_enabled=False,
                    exp_name=exp_name,
                    resume=True,
                    data_role="search",
                    run_subdir=Path(f"order-{order_id:02d}") / f"lr-{key}",
                    measure_latency=False,
                    checkpoint_path=checkpoint,
                )
                record = _candidate_record(summary_path)
                record["checkpoint_file"] = str(checkpoint.resolve())
                record["checkpoint_retained"] = True
                record["hyperparameters"] = serialized_parameters
                record["finished_at"] = timestamp()
                order_runs[key] = record
                atomic_write_json(path, payload)
            except BaseException as exc:
                error_path = (
                    Path(search_root) / "errors" / dataset / method
                    / f"order{order_id}_lr{key}.json"
                )
                atomic_write_json(
                    error_path,
                    {
                        "exp_name": exp_name,
                        "dataset": dataset,
                        "method": method,
                        "order": order_id,
                        "seed": SEARCH_SEED,
                        "learning_rate": lr,
                        "status": STATUS_FAILED,
                        "error": {
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                            "failed_stage": "lr_search",
                            "timestamp": timestamp(),
                        },
                    },
                )
                order_runs[key] = {
                    "status": STATUS_FAILED,
                    "error_file": str(error_path.resolve()),
                }
                atomic_write_json(path, payload)
                raise

    finalize_search(payload, order_ids)
    best_checkpoints: dict[str, str] = {}
    for order_id, best_lr in zip(order_ids, payload["best_per_order"]):
        runs = payload["search_runs"][f"order{order_id}"]
        selected = runs[str(best_lr)]
        source = Path(selected["checkpoint_file"])
        destination = best_checkpoint_path(
            search_root, dataset, method, order_id, best_lr
        )
        promote_search_checkpoint(source, destination)
        best_checkpoints[f"order{order_id}"] = str(destination.resolve())
        for record in runs.values():
            candidate = record.pop("checkpoint_file", None)
            if isinstance(candidate, str):
                candidate_path = Path(candidate)
                candidate_path.unlink(missing_ok=True)
                weight_checkpoint_path(candidate_path).unlink(missing_ok=True)
            record["checkpoint_retained"] = False
        selected["checkpoint_file"] = str(destination.resolve())
        selected["checkpoint_retained"] = True
    payload["best_checkpoint_per_order"] = best_checkpoints
    atomic_write_json(path, payload)
    save_optimum_lrs(search_root, dataset, method, payload["best_per_order"])
    return path
