"""Normalize existing failure archives without training or changing their contents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cil_experiments.output import atomic_write_json


def normalize(search_root: Path, apply: bool = False) -> tuple[int, int]:
    root = search_root.resolve(strict=True)
    plans = []
    missing = 0
    for unit in sorted(root.glob('*/order*/order*_seed*_search.json')):
        payload = json.loads(unit.read_text(encoding='utf-8'))
        edits = []
        for lr, record in payload['candidates'].items():
            if record.get('status') != 'failed':
                continue
            label = f"{payload['method']} order={payload['order']} seed={payload['seed']} lr={lr}"
            ref = record.get('failure_manifest')
            if not ref:
                print(f'MISSING {label}: no failure_manifest')
                missing += 1
                continue
            parts = str(ref).replace('\\', '/').split('/')
            if len(parts) < 3 or parts[-3] != 'failed_runs' or parts[-1] != 'manifest.json':
                raise ValueError(f'Unrecognized archive reference: {ref}')
            attempt = parts[-2]
            if attempt in ('', '.', '..'):
                raise ValueError(f'Unsafe archive reference: {ref}')
            lr_root = unit.parent / f"lr{str(float(lr)).replace('.', '')}"
            target = lr_root / 'failed_runs' / attempt
            matches = list(lr_root.glob(f'**/failed_runs/{attempt}/manifest.json'))
            if len(matches) != 1:
                raise ValueError(f'{label}: expected one archive, found {len(matches)}')
            source = matches[0].parent
            for path in (unit, source, target):
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f'Path escapes search root: {path}')
            manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
            identity = {key: payload[key] for key in ('exp_name', 'dataset', 'method', 'order', 'seed')}
            identity['lr'] = float(lr)
            if manifest['identity'] != identity:
                raise ValueError(f'{label}: identity mismatch')
            if set(manifest['files']) != {'config.json', 'failure.json', 'run.log'}:
                raise ValueError(f'{label}: unexpected evidence schema')
            if {p.name for p in source.iterdir()} != set(manifest['files']) | {'manifest.json'}:
                raise ValueError(f'{label}: unexpected archive contents')
            for name, digest in manifest['files'].items():
                file = source/name
                if file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'{label}: invalid evidence: {name}')
            if source != target and target.exists():
                raise FileExistsError(target)
            destination = str((target/'manifest.json').resolve())
            if source != target or ref != destination:
                edits.append((record, source, target, destination))
                print(f'{"MOVE" if source != target else "REFERENCE"} {label}: {source} -> {target}')
        if edits:
            plans.append((unit, payload, edits))
    # Validate every archive before making any changes. Stop writers before use.
    if apply:
        for unit, payload, edits in plans:
            for record, source, target, destination in edits:
                if source != target:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source.rename(target)
                record['failure_manifest'] = destination
            atomic_write_json(unit, payload)
            for _, source, target, _ in edits:
                if source != target:
                    parent = source.parent
                    while parent != target.parent.parent and parent.is_relative_to(root):
                        try:
                            parent.rmdir()  # Only remove empty legacy directories.
                        except OSError:
                            break
                        parent = parent.parent
    count = sum(len(edits) for _, _, edits in plans)
    print(f'{"APPLIED" if apply else "DRY_RUN"} updated={count} missing={missing}')
    return count, missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Move validated archives and update references')
    args = parser.parse_args()
    _, missing = normalize(args.search_root, args.apply)
    raise SystemExit(1 if missing else 0)
