# PP2 Continual Learning HPC Repository

记得配置自己的github。

这是用于超算运行 PP2 主实验的精简仓库，包含本项目修改过的本地 Avalanche 源码、六种 CIL 方法、三个数据集入口、学习率搜索、阶段式 joint learning、指标与 FLOPs 计算、断点恢复和结果汇总。数据集、checkpoint、训练日志和实验结果不进入 Git。

运行前依次阅读：

1. [HPC_RUN.md](HPC_RUN.md)：超算克隆、数据传输、完整性检查和启动步骤。
2. [PP2_QUICKSTART.md](PP2_QUICKSTART.md)：主实验参数与命令。
3. [PP2_MAIN_EXPERIMENT_FLOW.md](PP2_MAIN_EXPERIMENT_FLOW.md)：实际调用链和产物结构。

三个正式入口：

```bash
python main_exp/spike.py --stage all
python main_exp/texture.py --stage all
python main_exp/uwave.py --stage all
```

数据文件需单独放入 `dataset/`，并在启动实验前执行：

```bash
python verify_hpc_setup.py
```

本仓库基于 [ContinualAI Avalanche](https://github.com/ContinualAI/avalanche) 修改，原项目与本仓库源码遵循 [MIT License](LICENSE)。
