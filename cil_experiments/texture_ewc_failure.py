"""Failure evidence for every CIL method; legacy module path retained for imports."""
from __future__ import annotations

import hashlib
import json
import math
import traceback
import uuid
import shutil
from pathlib import Path
from avalanche.training.plugins import SupervisedPlugin

from .output import AtomicRunArtifacts, atomic_write_json, atomic_write_text
from .search_schema import is_nonfinite_training_failure


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_complete(record, identity):
    try:
        path = Path(record['failure_manifest'])
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['identity'] != identity:
            return False
        if set(manifest['files']) != {'failure.json', 'config.json', 'run.log'}:
            return False
        for name, digest in manifest['files'].items():
            file = path.parent / name
            if not file.is_file() or file.stat().st_size == 0 or _digest(file) != digest:
                return False
        failure = json.loads((path.parent / 'failure.json').read_text(encoding='utf-8'))
        return is_nonfinite_training_failure(failure) and failure['identity'] == identity
    except (OSError, ValueError, KeyError, TypeError):
        return False


def archive_before_retry(paths, checkpoint, record):
    """Retain the old failure JSON and any partial artifacts before a retry."""
    folder = paths.method_root / 'failed_runs' / uuid.uuid4().hex
    atomic_write_json(folder / 'previous_search_failure.json', record)
    existing = [p for p in (paths.log, paths.summary, paths.accuracy_matrix,
                            paths.config, checkpoint) if p.is_file()]
    for path in existing:
        shutil.copy2(path, folder / path.name)
    # All copies must succeed before removing partial files that block __enter__.
    for path in existing:
        path.unlink()
    return str((folder / 'previous_search_failure.json').resolve())


class FailureEvidenceArtifacts(AtomicRunArtifacts):
    snapshot = None

    def __init__(self, result_root, dataset, method, order_id, seed, config, *args):
        super().__init__(result_root, dataset, method, order_id, seed, config, *args)
        self.identity = dict(exp_name=config['exp_name'], dataset=dataset, method=method,
                             order=order_id, seed=seed,
                             lr=float(config['final_hyperparameters']['resolved']['learning_rate']))

    def __exit__(self, exc_type, exc, tb):
        record = dict(status='failed', error_type=type(exc).__name__, error_message=str(exc))
        try:
            if is_nonfinite_training_failure(record):
                self.logger.error('FAILED_NONFINITE %s', exc)
                for handler in self.logger.handlers:
                    handler.flush()
                folder = self.method_root / 'failed_runs' / uuid.uuid4().hex
                record.update(identity=self.identity,
                              traceback=''.join(traceback.format_exception(exc_type, exc, tb)),
                              partial=self.snapshot() if self.snapshot else {},
                              metrics_complete=False)
                atomic_write_json(folder / 'config.json', self.config)
                atomic_write_json(folder / 'failure.json', record)
                atomic_write_text(folder / 'run.log', self.temp_log.read_text(encoding='utf-8'))
                manifest = folder / 'manifest.json'
                atomic_write_json(manifest, dict(identity=self.identity, files={
                    name: _digest(folder / name) for name in ('config.json', 'failure.json', 'run.log')}))
                exc.failure_manifest = str(manifest.resolve())
        finally:
            super().__exit__(exc_type, exc, tb)
        return False


TextureEWCArtifacts = FailureEvidenceArtifacts  # Backward-compatible import.


class EpochEvidence(SupervisedPlugin):
    """Diagnostic observer; never changes model, loss, optimizer or RNG state."""
    def __init__(self, progress, logger):
        self.progress = progress
        self.logger = logger

    def after_training_epoch(self, strategy, **kwargs):
        # Runs before the tracker's epoch-end finite check.
        p = self.progress
        if p.samples:
            self.logger.info('TRAIN task=%s epoch=%s samples=%s mean_loss=%s',
                             p.task_index, p.epoch, p.samples, p.loss_sum / p.samples)


def snapshot(progress, matrix, task_wall):
    return dict(task=progress.task_index, epoch=progress.epoch,
                loss=str(progress.last_epoch_mean),
                completed_task_wall_s=list(task_wall),
                accuracy_matrix=[[float(v) if math.isfinite(v) else None for v in row]
                                 for row in matrix])
