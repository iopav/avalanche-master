"""One-epoch TagFex smoke run using an Avalanche benchmark and strategy."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from cil_experiments.tagfex_avalanche import (
    AvalancheTagFex,
    TagFexHyperParameters,
    build_uwave_image_benchmark,
    evaluate_seen_experiences,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_TAGFEX_ROOT = WORKSPACE_ROOT / "TagFex_CVPR2025" / "TagFex_CVPR2025"
DEFAULT_DATA_ROOT = DEFAULT_TAGFEX_ROOT / "data" / "data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--init-epochs", type=int)
    parser.add_argument("--inc-epochs", type=int)
    parser.add_argument("--train-mb-size", type=int, default=64)
    parser.add_argument("--eval-mb-size", type=int, default=128)
    parser.add_argument("--tagfex-root", type=Path, default=DEFAULT_TAGFEX_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    benchmark = build_uwave_image_benchmark(args.data_root)
    init_epochs = args.epochs if args.init_epochs is None else args.init_epochs
    inc_epochs = args.epochs if args.inc_epochs is None else args.inc_epochs
    if min(init_epochs, inc_epochs, args.train_mb_size, args.eval_mb_size) <= 0:
        raise ValueError("Epoch and minibatch values must be positive")
    hparams = TagFexHyperParameters(
        init_epochs=init_epochs,
        inc_epochs=inc_epochs,
        train_mb_size=args.train_mb_size,
        eval_mb_size=args.eval_mb_size,
    )
    strategy = AvalancheTagFex(
        tagfex_root=args.tagfex_root,
        device=device,
        hparams=hparams,
        dataset_name="uwave",
        num_classes=8,
    )
    classifier_matrix = np.full((6, 6), np.nan)
    nme_matrix = np.full((6, 6), np.nan)

    for task_index, experience in enumerate(benchmark.train_stream):
        strategy.train(experience, num_workers=0, shuffle=True)
        classifier, nme = evaluate_seen_experiences(
            strategy, benchmark.test_stream, task_index
        )
        classifier_matrix[task_index, : task_index + 1] = classifier
        nme_matrix[task_index, : task_index + 1] = nme
        memory_size = sum(len(values) for values in strategy.memory_by_class.values())
        print(
            f"task={task_index + 1}/6 classes={strategy.seen_classes} "
            f"memory={memory_size} loss_components={strategy.loss_components} "
            f"classifier_seen={[round(v * 100, 2) for v in classifier]} "
            f"nme_seen={[round(v * 100, 2) for v in nme]}",
            flush=True,
        )

    print("classifier_accuracy_matrix_percent=")
    print(np.round(classifier_matrix * 100, 2))
    print("nme_accuracy_matrix_percent=")
    print(np.round(nme_matrix * 100, 2))


if __name__ == "__main__":
    main()
