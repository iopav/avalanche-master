"""Resume one PP2 TagFex order, using the existing search/joint implementation."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from cil_experiments.order_seed_registry import ORDERS_BY_DATASET, SEEDS_BY_DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(ORDERS_BY_DATASET), required=True)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--order-id', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--backbone', default='resnet18_cifar')
    parser.add_argument('--loss-selection', default='last_epoch_train_mean',
                        choices=['last_epoch_train_mean', 'full_train_final_model'])
    parser.add_argument('--stage', choices=['search', 'joint', 'all'], default='all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.exp_name):
        parser.error('Invalid experiment name')
    if args.order_id not in ORDERS_BY_DATASET[args.dataset] or args.gpu < 0:
        parser.error('Invalid registered order or GPU index')
    root = Path(__file__).resolve().parent
    search_root = root / f'search_result_{args.exp_name}'
    joint_root = root / f'joint_result_{args.exp_name}'
    # Fail before any training if existing search units use different settings.
    for seed in SEEDS_BY_DATASET[args.dataset]:
        path = search_root / 'tagfex' / f'order{args.order_id}' / f'order{args.order_id}_seed{seed:03d}_search.json'
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf-8'))
            expected = dict(exp_name=args.exp_name, dataset=args.dataset, method='tagfex',
                            order=args.order_id, seed=seed, backbone=args.backbone,
                            loss_selection=args.loss_selection)
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(f'{path}: {key}: {payload.get(key)!r} != {value!r}')
    print(f'{args.dataset}/tagfex order={args.order_id} seeds={SEEDS_BY_DATASET[args.dataset]} '
          f'gpu={args.gpu} stage={args.stage} exp={args.exp_name}', flush=True)
    if args.dry_run:
        return
    from cil_experiments.pipeline import _device
    from cil_experiments.lr_search import run_search_unit
    from cil_experiments.joint_learning import run_joint_unit
    device = _device(str(args.gpu))
    common = dict(dataset_root=(args.dataset_root or root / 'dataset').resolve(),
                  search_root=search_root, exp_name=args.exp_name, dataset=args.dataset,
                  method='tagfex', order_id=args.order_id, device=device, backbone=args.backbone)
    # Same stage ordering as the normal pipeline, confined to this order.
    if args.stage in {'search', 'all'}:
        for seed in SEEDS_BY_DATASET[args.dataset]:
            run_search_unit(project_root=root, seed=seed,
                            loss_selection=args.loss_selection, **common)
    if args.stage in {'joint', 'all'}:
        for seed in SEEDS_BY_DATASET[args.dataset]:
            run_joint_unit(joint_root=joint_root, seed=seed, **common)
    print(f'TAGFEX_ORDER_COMPLETE order={args.order_id}', flush=True)


if __name__ == '__main__':
    main()
