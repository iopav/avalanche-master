from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

import torch

from .checkpointing import promote_search_checkpoint
from .final_hyperparameters import get_final_hyperparameters
from .metrics import validate_summary
from .order_seed_registry import ORDERS_BY_DATASET, SEEDS
from .output import atomic_write_json, run_artifact_paths
from .runner import run_experiment
from .search_config import LR_CANDIDATES
from .search_schema import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    new_order_seed_search,
    timestamp,
    validate_order_seed_search,
)

LOSS_SELECTIONS = ("last_epoch_train_mean", "full_train_final_model")


def lr_token(lr: float) -> str:
    return str(lr).replace(".", "")


def search_unit_path(search_root: Path, method: str, order_id: int, seed: int) -> Path:
    return Path(search_root) / method / f"order{order_id}" / f"order{order_id}_seed{seed:03d}_search.json"


def candidate_paths(search_root: Path, dataset: str, method: str, order_id: int, seed: int, lr: float):
    stem = f"search_{dataset}_{method}__order-{order_id}__lr-{lr_token(lr)}__seed-{seed:03d}"
    return run_artifact_paths(
        Path(search_root), dataset, method, order_id, seed,
        Path(f"order{order_id}") / f"lr{lr_token(lr)}",
        include_dataset_dir=False, stem_override=stem, flat_summary=True,
    )


def candidate_checkpoint_path(search_root: Path, dataset: str, method: str, order_id: int, seed: int, lr: float) -> Path:
    return Path(search_root) / method / f"order{order_id}" / "checkpoint_candidates" / f"{dataset}__{method}__order-{order_id}__seed-{seed:03d}__lr-{lr_token(lr)}.pt"


def best_checkpoint_path(search_root: Path, dataset: str, method: str, order_id: int, seed: int, lr: float) -> Path:
    return Path(search_root) / method / f"order{order_id}" / "checkpoint" / f"{dataset}__{method}__order-{order_id}__seed-{seed:03d}__lr-{lr_token(lr)}__best.pt"


def _record(summary_path: Path, matrix_path: Path, checkpoint: Path) -> dict[str, Any]:
    if not summary_path.is_file() or not matrix_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Candidate summary, matrix, or checkpoint is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_summary(summary, allow_pending_intransigence=True)
    selection = summary["selection"]
    if selection["loss"] is None:
        raise ValueError(f"Candidate has no selection loss: {summary_path}")
    latency = summary.get("inference", {}).get("final_latency_ms_per_sample")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency <= 0
    ):
        raise ValueError(f"Candidate has no valid inference latency: {summary_path}")
    return {
        "status": STATUS_COMPLETED,
        "selection_loss": float(selection["loss"]),
        "selection_loss_eval_flops": int(selection["loss_eval_flops"]),
        "overall_learning_flops": int(summary["training_operations"]["overall_learning_flops"]),
        "storage_bytes": int(summary["persistent_storage"]["model_parameter_bytes"]),
        "summary_file": str(summary_path.resolve()),
        "accuracy_matrix_file": str(matrix_path.resolve()),
        "checkpoint_file": str(checkpoint.resolve()),
    }


def _new_payload(
    exp_name: str,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    backbone: str,
    loss_selection: str,
    lr_candidates: tuple[float, ...],
) -> dict[str, Any]:
    return new_order_seed_search(
        exp_name=exp_name,
        dataset=dataset,
        method=method,
        order_id=order_id,
        seed=seed,
        backbone=backbone,
        loss_selection=loss_selection,
        lr_candidates=lr_candidates,
    )


def _load_unit(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return expected
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = ("schema", "exp_name", "dataset", "method", "order", "seed", "backbone", "loss_selection", "lr_candidates")
    differences = {key: (payload.get(key), expected.get(key)) for key in identity if payload.get(key) != expected.get(key)}
    if differences:
        raise ValueError(f"Existing order-seed search differs: {differences}")
    return payload


validate_search_unit = validate_order_seed_search


def run_search_unit(
    *,
    project_root: Path,
    dataset_root: Path,
    search_root: Path,
    exp_name: str,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    device: torch.device,
    backbone: str,
    loss_selection: str,
    lr_candidates: tuple[float, ...] = LR_CANDIDATES,
    parameter_overrides: dict[str, Any] | None = None,
    data_role: str = "formal",
) -> Path:
    if loss_selection not in LOSS_SELECTIONS:
        raise ValueError(f"Unknown loss selection: {loss_selection}")
    lr_candidates = tuple(float(value) for value in lr_candidates)
    if not lr_candidates or len(set(lr_candidates)) != len(lr_candidates):
        raise ValueError("LR candidates must be non-empty and unique")
    if any(not math.isfinite(value) or value <= 0 for value in lr_candidates):
        raise ValueError("LR candidates must contain only positive finite values")
    path = search_unit_path(search_root, method, order_id, seed)
    payload = _load_unit(
        path,
        _new_payload(
            exp_name,
            dataset,
            method,
            order_id,
            seed,
            backbone,
            loss_selection,
            lr_candidates,
        ),
    )
    if payload["status"] == STATUS_COMPLETED:
        validate_search_unit(payload)
        for key in ("best_checkpoint", "best_summary_file", "best_accuracy_matrix_file"):
            if not Path(payload[key]).is_file():
                raise FileNotFoundError(payload[key])
        selected_summary = json.loads(
            Path(payload["best_summary_file"]).read_text(encoding="utf-8")
        )
        # Joint learning fills intransigence after LR selection. A completed
        # search must remain resumable while that downstream stage is pending.
        validate_summary(selected_summary, allow_pending_intransigence=True)
        latency = selected_summary.get("inference", {}).get(
            "final_latency_ms_per_sample"
        )
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError(
                "Completed search predates required latency measurement; use a new "
                "EXP_NAME or remove this order-seed search and rerun it"
            )
        return path
    atomic_write_json(path, payload)
    base_parameters = get_final_hyperparameters(
        dataset,
        method,
        parameter_overrides,
        require_locked=data_role == "formal",
    )
    for lr in lr_candidates:
        key = str(lr)
        paths = candidate_paths(search_root, dataset, method, order_id, seed, lr)
        checkpoint = candidate_checkpoint_path(search_root, dataset, method, order_id, seed, lr)
        if payload["candidates"][key].get("status") == STATUS_COMPLETED:
            _record(paths.summary, paths.accuracy_matrix, checkpoint)
            continue
        recoverable = (paths.summary, paths.accuracy_matrix, checkpoint)
        if all(candidate.is_file() for candidate in recoverable):
            payload["candidates"][key] = _record(
                paths.summary, paths.accuracy_matrix, checkpoint
            )
            atomic_write_json(path, payload)
            continue
        if any(candidate.is_file() for candidate in recoverable):
            for incomplete in (
                paths.summary, paths.accuracy_matrix, paths.config, paths.log, checkpoint
            ):
                incomplete.unlink(missing_ok=True)
        try:
            parameters = dict(base_parameters)
            parameters["learning_rate"] = float(lr)
            if method == "tagfex":
                parameters["init_lr"] = float(lr)
                parameters["inc_lr"] = float(lr)
            print(
                "SEARCH_START "
                f"dataset={dataset} method={method} order={order_id} "
                f"seed={seed} lr={lr:.8g}",
                flush=True,
            )
            summary_path = run_experiment(
                project_root=project_root, dataset_root=dataset_root,
                result_root=search_root, dataset_name=dataset, method=method,
                order_id=order_id, seed=seed, epochs=None, device=device,
                parameter_overrides=parameters, backbone_id=backbone,
                search_provenance={
                    "status": "current_order_seed_lr_candidate",
                    "hyperparameter_search_flops": 0,
                    "test_set_used_for_selection": False,
                    "required_trials": len(lr_candidates),
                },
                compute_intransigence_enabled=False, exp_name=exp_name,
                data_role=data_role, run_subdir=Path(f"order{order_id}") / f"lr{lr_token(lr)}",
                measure_latency=True, checkpoint_path=checkpoint,
                loss_selection=loss_selection, include_dataset_dir=False,
                artifact_stem=paths.summary.name.removesuffix("__summary.json"),
                flat_summary=True,
            )
            paths.config.unlink(missing_ok=True)
            payload["candidates"][key] = _record(summary_path, paths.accuracy_matrix, checkpoint)
            atomic_write_json(path, payload)
            print(
                "SEARCH_COMPLETE "
                f"dataset={dataset} method={method} order={order_id} "
                f"seed={seed} lr={lr:.8g} "
                f"loss={payload['candidates'][key]['selection_loss']:.6f}",
                flush=True,
            )
        except BaseException as exc:
            payload["candidates"][key] = {"status": STATUS_FAILED, "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()}
            atomic_write_json(path, payload)
            raise
    best_lr = min(
        lr_candidates,
        key=lambda value: (
            payload["candidates"][str(value)]["selection_loss"],
            lr_candidates.index(value),
        ),
    )
    selected = payload["candidates"][str(best_lr)]
    destination = best_checkpoint_path(search_root, dataset, method, order_id, seed, best_lr)
    promote_search_checkpoint(Path(selected["checkpoint_file"]), destination)
    for record in payload["candidates"].values():
        checkpoint_file = record.pop("checkpoint_file", None)
        if checkpoint_file:
            Path(checkpoint_file).unlink(missing_ok=True)
    candidate_directory = Path(search_root) / method / f"order{order_id}" / "checkpoint_candidates"
    if candidate_directory.is_dir() and not any(candidate_directory.iterdir()):
        candidate_directory.rmdir()
    payload.update(
        best_lr=float(best_lr), best_loss=float(selected["selection_loss"]),
        best_checkpoint=str(destination.resolve()), best_summary_file=selected["summary_file"],
        best_accuracy_matrix_file=selected["accuracy_matrix_file"],
        total_search_flops=sum(int(record["overall_learning_flops"]) for record in payload["candidates"].values()),
        storage_bytes=int(selected["storage_bytes"]), status=STATUS_COMPLETED,
        finished_at=timestamp(),
    )
    atomic_write_json(path, payload)
    validate_search_unit(payload)
    print(
        "SEARCH_SELECTED "
        f"dataset={dataset} method={method} order={order_id} seed={seed} "
        f"lr={best_lr:.8g} loss={float(selected['selection_loss']):.6f}",
        flush=True,
    )
    return path


def run_method_search(*, project_root: Path, dataset_root: Path, search_root: Path, exp_name: str, dataset: str, method: str, device: torch.device, backbone: str, loss_selection: str) -> list[Path]:
    return [
        run_search_unit(
            project_root=project_root, dataset_root=dataset_root, search_root=search_root,
            exp_name=exp_name, dataset=dataset, method=method, order_id=order_id,
            seed=seed, device=device, backbone=backbone, loss_selection=loss_selection,
        )
        for order_id in sorted(ORDERS_BY_DATASET[dataset]) for seed in SEEDS
    ]
