from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any


STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_order_seed_search(
    *,
    exp_name: str,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    backbone: str,
    loss_selection: str,
    lr_candidates: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "schema": "pp2-order-seed-search-v1",
        "exp_name": exp_name,
        "dataset": dataset,
        "method": method,
        "order": int(order_id),
        "seed": int(seed),
        "backbone": backbone,
        "loss_selection": loss_selection,
        "lr_candidates": list(lr_candidates),
        "candidates": {
            str(lr): {"status": STATUS_PENDING} for lr in lr_candidates
        },
        "best_lr": None,
        "best_loss": None,
        "best_checkpoint": None,
        "best_summary_file": None,
        "best_accuracy_matrix_file": None,
        "total_search_flops": None,
        "storage_bytes": None,
        "status": STATUS_PENDING,
        "started_at": timestamp(),
        "finished_at": None,
    }


def validate_order_seed_search(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "pp2-order-seed-search-v1":
        raise ValueError("Unknown order-seed search schema")
    if payload.get("status") != STATUS_COMPLETED:
        raise RuntimeError("Order-seed search is incomplete")
    candidates = payload.get("candidates")
    learning_rates = payload.get("lr_candidates")
    if not isinstance(candidates, dict) or not isinstance(learning_rates, list):
        raise ValueError("Search candidates are invalid")
    if set(candidates) != {str(value) for value in learning_rates}:
        raise ValueError("Search candidate keys differ from LR candidates")
    for record in candidates.values():
        if record.get("status") != STATUS_COMPLETED:
            raise RuntimeError("A candidate is incomplete")
        for key in (
            "overall_learning_flops",
            "selection_loss_eval_flops",
            "storage_bytes",
        ):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Candidate {key} is invalid")
        if not math.isfinite(float(record.get("selection_loss"))):
            raise ValueError("Candidate selection loss is invalid")
    best_key = str(payload.get("best_lr"))
    if best_key not in candidates:
        raise ValueError("Best LR is not a registered candidate")
    selected = candidates[best_key]
    if not math.isclose(
        float(payload["best_loss"]),
        float(selected["selection_loss"]),
        abs_tol=1e-12,
    ):
        raise ValueError("Best loss differs from selected candidate")
    if payload.get("total_search_flops") != sum(
        record["overall_learning_flops"] for record in candidates.values()
    ):
        raise ValueError("Total search FLOPs do not equal the registered candidates")
    if payload.get("storage_bytes") != selected["storage_bytes"]:
        raise ValueError("Search storage does not equal selected model parameter bytes")


def search_result_path(main_exp_root: Path, dataset: str, method: str) -> Path:
    """Legacy path helper retained only for reading old PP2 artifacts."""

    return Path(main_exp_root) / dataset / method / f"search_{dataset}_{method}.json"
