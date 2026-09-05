"""把某个普通方法在 seed 62 搜索得到的最佳 checkpoint 作为 joint reference 测试。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cil_experiments.pipeline import run_joint_checkpoint_pipeline


# 必须与生成 search_result_<EXP_NAME> 的普通方法实验脚本保持一致。
EXP_NAME = "ewc_resnet18"
SOURCE_METHOD = "ewc"
GPU = "0"


if __name__ == "__main__":
    # 本入口只加载 checkpoint 并测试，不搜索，也不训练。
    raise SystemExit(
        run_joint_checkpoint_pipeline(
            source_method=SOURCE_METHOD,
            exp_name=EXP_NAME,
            gpu=GPU,
        )
    )
