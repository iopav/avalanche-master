"""EWC 的 PP2 实验入口；只需修改下方三个参数。"""

from pathlib import Path
import sys

# 允许从 main_exp 目录直接执行本文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cil_experiments.pipeline import run_method_pipeline


EXP_NAME = "ewc_resnet18"
GPU = "0"  # 使用 CPU 时填写 "cpu"。
BACKBONE = "resnet18_cifar"  # 轻量流程检查可改为 "temporal"。


if __name__ == "__main__":
    # 可追加 --stage search/joint/formal/aggregate 和 --datasets spike texture uwave。
    raise SystemExit(
        run_method_pipeline(
            method="ewc", exp_name=EXP_NAME, gpu=GPU, backbone=BACKBONE
        )
    )
