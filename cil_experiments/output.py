from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    """Write a standalone text artifact atomically on its destination volume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    staged = Path(name)
    try:
        staged.write_text(text, encoding="utf-8")
        with staged.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json_text(value))


def local_timestamp() -> str:
    """Timestamp retained only for hyperparameter-search diagnostics."""
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


@dataclass(frozen=True)
class RunArtifactPaths:
    method_root: Path
    log: Path
    summary: Path
    accuracy_matrix: Path
    config: Path


def run_artifact_paths(
    result_root: Path,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    run_subdir: Path | str | None = None,
) -> RunArtifactPaths:
    method_root = result_root / dataset / method
    if run_subdir is not None:
        method_root /= Path(run_subdir)
    stem = f"{dataset}__{method}__order-{order_id:02d}__seed-{seed:03d}"
    return RunArtifactPaths(
        method_root=method_root,
        log=method_root / "log" / f"{stem}.log",
        summary=method_root / "summary" / f"{stem}__summary.json",
        accuracy_matrix=method_root / "summary" / f"{stem}__accuracy-matrix.json",
        config=method_root / f"{stem}__config.json",
    )


def completed_summary_path(
    result_root: Path,
    dataset: str,
    method: str,
    order_id: int,
    seed: int,
    exp_name: str,
    expected_config: dict[str, Any] | None = None,
    run_subdir: Path | str | None = None,
) -> Path | None:
    """Return a validated completed run, or None when no public artifact exists."""
    paths = run_artifact_paths(
        result_root, dataset, method, order_id, seed, run_subdir
    )
    required = (paths.log, paths.summary, paths.accuracy_matrix, paths.config)
    existing = [path for path in required if path.is_file()]
    if not existing:
        return None
    if len(existing) != len(required):
        missing = [str(path) for path in required if not path.is_file()]
        raise RuntimeError(
            "Incomplete public artifact set; rerun this small experiment with "
            f"--overwrite. Missing: {missing}"
        )
    from .metrics import validate_summary

    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    matrix = json.loads(paths.accuracy_matrix.read_text(encoding="utf-8"))
    config = json.loads(paths.config.read_text(encoding="utf-8"))
    validate_summary(summary, allow_pending_intransigence=True)
    expected_matrix = {
        "dataset": dataset,
        "method": method,
        "order_id": int(order_id),
        "seed": int(seed),
        "exp_name": exp_name,
    }
    mismatches = {
        key: (matrix.get(key), expected)
        for key, expected in expected_matrix.items()
        if matrix.get(key) != expected
    }
    if summary.get("exp_name") != exp_name:
        mismatches["summary.exp_name"] = (summary.get("exp_name"), exp_name)
    if config.get("exp_name") != exp_name:
        mismatches["config.exp_name"] = (config.get("exp_name"), exp_name)
    normalized_expected_config = (
        json.loads(json_text(expected_config)) if expected_config is not None else None
    )
    if normalized_expected_config is not None and config != normalized_expected_config:
        raise ValueError(
            "A completed run exists for this experiment name, but its config differs. "
            "Use a new EXP_NAME or overwrite this exact run."
        )
    if mismatches:
        raise ValueError(f"Completed artifact identity mismatch: {mismatches}")
    return paths.summary


class AtomicRunArtifacts:
    """Stage a run in a temporary directory and publish only after validation."""

    def __init__(
        self,
        result_root: Path,
        dataset: str,
        method: str,
        order_id: int,
        seed: int,
        config: dict[str, Any],
        overwrite: bool = False,
        run_subdir: Path | str | None = None,
    ):
        paths = run_artifact_paths(
            result_root, dataset, method, order_id, seed, run_subdir
        )
        self.method_root = paths.method_root
        self.log_path = paths.log
        self.summary_path = paths.summary
        self.accuracy_matrix_path = paths.accuracy_matrix
        self.config_path = paths.config
        self.overwrite = overwrite
        self.config = dict(config)
        self.temp_dir: Path | None = None
        self.temp_log: Path | None = None
        self.logger: logging.Logger | None = None

    def __enter__(self):
        if not self.overwrite and (
            self.log_path.exists() or self.summary_path.exists() or self.accuracy_matrix_path.exists()
        ):
            raise FileExistsError(f"Refusing to overwrite existing run: {self.summary_path}")
        if self.config_path.exists():
            existing = json.loads(self.config_path.read_text(encoding="utf-8"))
            if existing != self.config and not self.overwrite:
                raise ValueError(f"Existing config differs for {self.method_root}")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="avalanche-cil-run-"))
        self.temp_log = self.temp_dir / "run.log"
        logger_name = f"cil.{os.getpid()}.{id(self)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        file_handler = logging.FileHandler(self.temp_log, encoding="utf-8")
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        return self

    def commit(self, summary: dict[str, Any], accuracy_matrix: dict[str, Any]) -> None:
        if self.temp_dir is None or self.temp_log is None or self.logger is None:
            raise RuntimeError("Artifact transaction is not active")
        temp_summary = self.temp_dir / "summary.json"
        temp_config = self.temp_dir / "config.json"
        temp_matrix = self.temp_dir / "accuracy-matrix.json"
        temp_summary.write_text(json_text(summary), encoding="utf-8")
        temp_config.write_text(json_text(self.config), encoding="utf-8")
        temp_matrix.write_text(json_text(accuracy_matrix), encoding="utf-8")
        for handler in self.logger.handlers:
            handler.flush()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        committed: list[Path] = []
        backups: list[tuple[Path, Path]] = []
        volume_staged: list[Path] = []

        def stage_on_destination_volume(source: Path, destination: Path) -> Path:
            fd, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            os.close(fd)
            staged = Path(name)
            try:
                shutil.copyfile(source, staged)
                with staged.open("r+b") as stream:
                    os.fsync(stream.fileno())
            except BaseException:
                staged.unlink(missing_ok=True)
                raise
            volume_staged.append(staged)
            return staged

        staged_publications: list[tuple[Path, Path]] = []
        try:
            if self.overwrite or not self.config_path.exists():
                staged_publications.append(
                    (stage_on_destination_volume(temp_config, self.config_path), self.config_path)
                )
            staged_publications.extend(
                [
                    (stage_on_destination_volume(self.temp_log, self.log_path), self.log_path),
                    (stage_on_destination_volume(temp_summary, self.summary_path), self.summary_path),
                    (stage_on_destination_volume(temp_matrix, self.accuracy_matrix_path), self.accuracy_matrix_path),
                ]
            )
            if self.overwrite:
                for _, destination in staged_publications:
                    if not destination.exists():
                        continue
                    fd, backup_name = tempfile.mkstemp(
                        prefix=f".{destination.name}.", suffix=".backup", dir=destination.parent
                    )
                    os.close(fd)
                    backup = Path(backup_name)
                    backup.unlink()
                    os.replace(destination, backup)
                    backups.append((destination, backup))
            for staged, destination in staged_publications:
                os.replace(staged, destination)
                volume_staged.remove(staged)
                committed.append(destination)
            for _, backup in backups:
                backup.unlink(missing_ok=True)
            backups.clear()
        except BaseException:
            for path in volume_staged:
                path.unlink(missing_ok=True)
            for path in reversed(committed):
                path.unlink(missing_ok=True)
            for destination, backup in reversed(backups):
                if backup.exists():
                    os.replace(backup, destination)
            raise

    def __exit__(self, exc_type, exc, tb):
        if self.logger is not None:
            handlers = list(self.logger.handlers)
            for handler in handlers:
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if exc_type is not None:
            self._remove_empty_result_dirs()
        return False

    def _remove_empty_result_dirs(self):
        for path in (self.method_root / "log", self.method_root / "summary", self.method_root):
            try:
                path.rmdir()
            except (FileNotFoundError, OSError):
                pass
