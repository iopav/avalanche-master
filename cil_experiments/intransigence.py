from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .metrics import validate_summary
from .output import json_text
from .search_config import JOINT_METHOD


def _artifact_paths(summary_path: Path) -> tuple[Path, Path]:
    suffix = "__summary.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"Not a canonical summary path: {summary_path}")
    stem = summary_path.name[: -len(suffix)]
    return (
        summary_path.with_name(f"{stem}__accuracy-matrix.json"),
        summary_path.parent.parent / f"{stem}__config.json",
    )


def _read_run(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    matrix_path, config_path = _artifact_paths(summary_path)
    for path in (summary_path, matrix_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (summary_path, matrix_path, config_path)
    )  # type: ignore[return-value]


def _backbone_id(config: dict[str, Any]) -> str:
    value = config.get("backbone", {}).get("backbone_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Config does not contain backbone.backbone_id")
    return value


def compute_intransigence(
    cil_matrix: dict[str, Any], joint_matrix: dict[str, Any]
) -> list[float]:
    lower = cil_matrix.get("accuracy_matrix_lower_triangular")
    reference = joint_matrix.get("accuracy_by_task")
    tasks = int(cil_matrix.get("tasks", 0))
    if not isinstance(lower, list) or len(lower) != tasks:
        raise ValueError("CIL accuracy matrix task dimension is invalid")
    if not isinstance(reference, list) or len(reference) != tasks:
        raise ValueError("Joint reference accuracy row task dimension is invalid")
    values: list[float] = []
    for index in range(tasks):
        row = lower[index]
        if not isinstance(row, list) or len(row) != tasks or row[index] is None:
            raise ValueError(f"Missing CIL diagonal accuracy at task {index + 1}")
        if reference[index] is None:
            raise ValueError(f"Missing joint reference accuracy at task {index + 1}")
        values.append(float(reference[index]) - float(row[index]))
    return values


def _validate_identity(
    cil_matrix: dict[str, Any],
    cil_config: dict[str, Any],
    joint_matrix: dict[str, Any],
    joint_config: dict[str, Any],
) -> None:
    identity_keys = ("dataset", "order_id", "seed", "exp_name")
    mismatches = {
        key: (cil_matrix.get(key), joint_matrix.get(key))
        for key in identity_keys
        if cil_matrix.get(key) != joint_matrix.get(key)
    }
    if joint_matrix.get("method") != JOINT_METHOD:
        mismatches["method"] = (joint_matrix.get("method"), JOINT_METHOD)
    if _backbone_id(cil_config) != _backbone_id(joint_config):
        mismatches["backbone_id"] = (
            _backbone_id(cil_config), _backbone_id(joint_config)
        )
    cil_view = cil_config.get("protocol", {}).get("input_view_id")
    joint_view = joint_config.get("protocol", {}).get("input_view_id")
    if cil_view != joint_view:
        mismatches["input_view_id"] = (cil_view, joint_view)
    cil_groups = cil_config.get("selected_task_groups")
    joint_groups = joint_config.get("selected_task_groups")
    if cil_groups != joint_groups:
        mismatches["selected_task_groups"] = (cil_groups, joint_groups)
    if mismatches:
        raise ValueError(f"CIL/joint-learning identity mismatch: {mismatches}")


def validate_joint_artifact_match(
    cil_matrix_identity: dict[str, Any],
    cil_config: dict[str, Any],
    joint_summary_path: Path,
) -> None:
    joint_summary, joint_matrix, joint_config = _read_run(
        joint_summary_path
    )
    validate_summary(joint_summary, allow_pending_intransigence=True)
    _validate_identity(
        cil_matrix_identity, cil_config, joint_matrix, joint_config
    )
    tasks = int(cil_matrix_identity["tasks"])
    if int(joint_summary["tasks"]) != tasks:
        raise ValueError("Joint-learning task count does not match the CIL run")


def fill_intransigence(cil_summary_path: Path, joint_summary_path: Path) -> list[float]:
    cil_summary, cil_matrix, cil_config = _read_run(cil_summary_path)
    joint_summary, joint_matrix, joint_config = _read_run(
        joint_summary_path
    )
    _validate_identity(cil_matrix, cil_config, joint_matrix, joint_config)
    if int(cil_summary["tasks"]) != int(joint_summary["tasks"]):
        raise ValueError("CIL/joint-learning task counts differ")
    values = compute_intransigence(cil_matrix, joint_matrix)
    cil_summary["cil_performance"]["intransigence"] = values
    validate_summary(cil_summary)
    fd, name = tempfile.mkstemp(
        prefix=f".{cil_summary_path.name}.", suffix=".tmp", dir=cil_summary_path.parent
    )
    os.close(fd)
    staged = Path(name)
    try:
        staged.write_text(json_text(cil_summary), encoding="utf-8")
        with staged.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(staged, cil_summary_path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return values


def find_joint_summary(
    joint_result_root: Path,
    *,
    dataset: str,
    order_id: int,
    seed: int,
) -> Path:
    path = (
        Path(joint_result_root)
        / dataset
        / JOINT_METHOD
        / "summary"
        / f"{dataset}__{JOINT_METHOD}__order-{order_id:02d}__seed-{seed:03d}__summary.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing joint-learning reference: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill CIL summary Intransigence values")
    parser.add_argument("--cil-summary", type=Path, required=True)
    parser.add_argument("--joint-summary", type=Path, required=True)
    args = parser.parse_args()
    values = fill_intransigence(
        args.cil_summary.resolve(), args.joint_summary.resolve()
    )
    print(json.dumps({"intransigence": values}, ensure_ascii=False))


if __name__ == "__main__":
    main()
