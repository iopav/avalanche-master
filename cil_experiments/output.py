from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n"


def local_timestamp() -> str:
    """Windows-safe, timezone-bearing timestamp shared by one artifact transaction."""
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


class AtomicRunArtifacts:
    """Stage a run outside result/ and publish only after every validation passes."""

    def __init__(
        self,
        result_root: Path,
        dataset: str,
        method: str,
        order_id: int,
        seed: int,
        config: dict[str, Any],
        overwrite: bool = False,
        timestamp: str | None = None,
    ):
        self.method_root = result_root / dataset / method
        self.timestamp = timestamp or local_timestamp()
        stem = f"{dataset}__{method}__order-{order_id:02d}__seed-{seed:03d}__timestamp-{self.timestamp}"
        self.log_path = self.method_root / "log" / f"{stem}.log"
        self.summary_path = self.method_root / "summary" / f"{stem}__summary.json"
        self.accuracy_matrix_path = self.method_root / "summary" / f"{stem}__accuracy-matrix.json"
        self.config_path = self.method_root / f"{stem}__config.json"
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
