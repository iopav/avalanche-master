from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .search_config import ACC_TOLERANCE, LR_CANDIDATES


STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def search_result_path(main_exp_root: Path, dataset: str, method: str) -> Path:
    return (
        Path(main_exp_root)
        / dataset
        / method
        / f"search_{dataset}_{method}.json"
    )


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_search_result(
    *, exp_name: str, dataset: str, method: str, validation_split_file: Path,
    search_seed: int, order_ids: tuple[int, ...]
) -> dict[str, Any]:
    return {
        "metadata": {
            "exp_name": exp_name,
            "dataset": dataset,
            "method": method,
            "search_seed": search_seed,
            "lr_candidates": list(LR_CANDIDATES),
            "acc_tolerance": ACC_TOLERANCE,
            "search_storage_definition": "selected_model_parameter_bytes",
            "validation_split_file": str(validation_split_file.resolve()),
            "started_at": timestamp(),
            "finished_at": None,
            "status": STATUS_PENDING,
        },
        "search_runs": {
            f"order{order_id}": {
                str(lr): {"status": STATUS_PENDING} for lr in LR_CANDIDATES
            }
            for order_id in order_ids
        },
        "best_per_order": [],
        "best_checkpoint_per_order": {},
        # 九个候选完成后由 lr_search.finalize_search() 统一填充。
        "search_metrics": {},
    }
