# 待删除的旧 Joint 实现

本目录不在 PP2 活动调用链中，并且仍依赖已经删除的整数 experiment ID 接口，不能作为当前实验入口。当前 joint reference 由普通方法在 seed 62 的 LR search 最佳完整 checkpoint 产生：`cil_experiments.runner.evaluate_joint_checkpoint()` 加载 checkpoint 后只取 `bundle.strategy.model` 测试，不重新训练。实际入口为普通方法脚本的 `--stage joint`，或 `main_exp/joint_learning.py`。

本目录按用户要求暂不删除，后续确认后可整体移除。
