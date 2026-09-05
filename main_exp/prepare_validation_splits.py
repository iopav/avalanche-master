"""生成搜索专用训练集和验证集 NPY；不会读取正式测试集。"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cil_experiments.registry import DATASETS
from cil_experiments.validation import create_validation_split


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 PP2 固定验证拆分")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        print(
            create_validation_split(
                args.dataset_root.resolve(), dataset, overwrite=args.overwrite
            )
        )


if __name__ == "__main__":
    main()
