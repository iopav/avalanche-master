from __future__ import annotations

from pathlib import Path


# Edit this integer before starting a new formal experiment batch.
EXPERIMENT_ID = 1


def validate_experiment_id(experiment_id: int = EXPERIMENT_ID) -> int:
    if isinstance(experiment_id, bool) or not isinstance(experiment_id, int):
        raise TypeError("EXPERIMENT_ID must be an integer")
    if experiment_id <= 0:
        raise ValueError("EXPERIMENT_ID must be positive")
    return experiment_id


def experiment_result_root(
    project_root: Path, experiment_id: int = EXPERIMENT_ID
) -> Path:
    return project_root / f"result-exp{validate_experiment_id(experiment_id)}"
