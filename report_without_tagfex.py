"""Audit PP2 units and export complete five-method reports without training."""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

from cil_experiments.order_seed_registry import ORDERS_BY_DATASET, SEEDS_BY_DATASET

METHODS = ('ewc', 'er_ace', 'icarl', 'fecam', 'cwr_star')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(ORDERS_BY_DATASET), required=True)
    parser.add_argument('--exp-name', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.exp_name):
        parser.error('Invalid experiment name')
    root = Path(__file__).resolve().parent
    search = root / f'search_result_{args.exp_name}'
    joint = root / f'joint_result_{args.exp_name}'
    if not search.is_dir() or not joint.is_dir():
        parser.error(f'Missing result roots: {search}, {joint}. Run on the experiment host.')
    from cil_experiments.aggregate_results import (
        build_dataset_search_summary, build_dataset_joint_learning,
    )
    from cil_experiments.output import atomic_write_text

    valid_search = []
    valid_joint = []
    audit = []
    failed = False
    for method in (*METHODS, 'tagfex'):
        count = 0
        for order in sorted(ORDERS_BY_DATASET[args.dataset]):
            for seed in SEEDS_BY_DATASET[args.dataset]:
                try:
                    unit_path = search / method / f'order{order}' / f'order{order}_seed{seed:03d}_search.json'
                    payload = json.loads(unit_path.read_text(encoding='utf-8'))
                    if payload.get('exp_name') != args.exp_name:
                        raise ValueError(f'Experiment identity mismatch: {unit_path}')
                    kwargs = dict(dataset=args.dataset, methods=(method,),
                                  order_ids=(order,), seeds=(seed,))
                    search_rows = list(csv.DictReader(io.StringIO(
                        build_dataset_search_summary(search, joint_root=joint, **kwargs))))
                    joint_rows = list(csv.DictReader(io.StringIO(
                        build_dataset_joint_learning(joint, **kwargs))))
                    count += 1
                    error = ''
                    if method in METHODS:
                        valid_search.extend(search_rows)
                        valid_joint.extend(joint_rows)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    if method in METHODS:
                        failed = True
                audit.append(dict(method=method, order=order, seed=seed,
                                  search_and_joint_valid=not bool(error), error=error))
        total = len(ORDERS_BY_DATASET[args.dataset]) * len(SEEDS_BY_DATASET[args.dataset])
        print(f'{method}: validated search+joint {count}/{total}', flush=True)
    destination = search / 'aggregate_results'

    def write_csv(path: Path, rows: list[dict]) -> None:
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(path, output.getvalue())

    audit_path = destination / f'{args.dataset}_completion_audit.csv'
    write_csv(audit_path, audit)
    print(audit_path)
    if failed:
        print('FIVE_METHODS_INCOMPLETE: no summary exported; inspect audit errors. '
              'Any previously exported reports are stale and must not be published.')
        return 1
    # Build both tables fully before replacing either report. These canonical
    # names are consumed by the existing publisher; scope is recorded alongside.
    write_csv(destination / f'{args.dataset}_search_summary.csv', valid_search)
    write_csv(joint / 'aggregate_results' / f'{args.dataset}_joint_learning.csv', valid_joint)
    best = [row for row in valid_search if float(row['lr']) == float(row['best_lr'])]
    write_csv(destination / f'{args.dataset}_best_lr_without_tagfex.csv', best)
    atomic_write_text(destination / 'report_scope_without_tagfex.json', json.dumps({
        'exp_name': args.exp_name, 'dataset': args.dataset, 'included_methods': METHODS,
        'excluded_methods': ['tagfex'], 'search_candidate_rows': len(valid_search),
        'selected_rows': len(best), 'joint_rows': len(valid_joint),
        'note': 'Five-method interim report. TagFex raw artifacts may be incomplete. '
                'Completion audit includes TagFex as a point-in-time observation.',
    }, indent=2) + '\n')
    print(f'FIVE_METHODS_COMPLETE search_rows={len(valid_search)} '
          f'best_lr_rows={len(best)} joint_rows={len(valid_joint)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
