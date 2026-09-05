# PP2 超算运行版本

本仓库只跟踪主实验运行所需代码，不跟踪数据集、checkpoint、日志、搜索结果、joint 结果、上游 Avalanche 文档、示例、notebook 和全量测试。超算端克隆仓库后，需要单独把数据文件传入 `dataset/`。

## GitHub 中必须保留的内容

- `avalanche/`：本项目实际使用并修改过的本地 Avalanche 源码，不能改成安装未经核对的 PyPI Avalanche。
- `cil_experiments/`：PP2 数据、方法、训练、指标、FLOPs、搜索、joint learning、恢复和聚合实现。
- `main_exp/spike.py`、`main_exp/texture.py`、`main_exp/uwave.py`：三个正式入口。
- `requirements.txt`：已验证环境的依赖版本。
- `setup.py`、`setup.cfg`、`pyproject.toml`、`extra_dependencies.txt`：本地包安装与工具配置。
- `PP2_QUICKSTART.md`、`PP2_MAIN_EXPERIMENT_FLOW.md`：运行说明和调用链。
- `verify_hpc_setup.py`：超算端必需数据文件与运行模块检查工具。

## 数据传输

GitHub 普通 Git 仓库会阻止超过 100 MiB 的单个文件，本项目多个训练数组超过该限制，因此不把数据塞进代码仓库；若改用 Git LFS，还会引入独立存储和下载流程。限制说明见 [GitHub 官方文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。把本地 `dataset/` 中主实验需要的16个文件单独传到超算，保持目录结构不变；具体路径写在 `verify_hpc_setup.py` 的 `REQUIRED_DATA_FILES` 中。当前主实验最小数据集合约957 MB；本地 `dataset/` 中的 zip、旧版 `*_20_3.npy`、validation split、Spike 图像副本以及 Texture/UWave 原始 X 不参与当前主调用链，不需要上传。

在超算仓库根目录执行：

```bash
python verify_hpc_setup.py --data-only
```

16行都必须显示 `DATA OK`；缺文件或路径不一致时不要启动正式实验。安装依赖后再执行一次不带参数的 `python verify_hpc_setup.py`，同时核验本地 Avalanche、PP2 模块、三个数据集注册和六种正式方法。

## 环境安装

先按超算 CUDA/驱动版本加载 Python、CUDA 和 PyTorch，再安装其余依赖。仓库当前验证版本见 `requirements.txt`：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果集群要求从专用 wheel 或 module 加载 CUDA 版 PyTorch，应先安装同一公开版本的 `torch` 和 `torchvision`，再执行 requirements 安装。不要直接替换成本地未验证的 Avalanche 包；从仓库根目录运行时会优先导入仓库中的 `avalanche/`。

## 启动前配置

依次检查以下文件：

1. `cil_experiments/final_hyperparameters.py`：epoch、batch size、优化器与方法参数。
2. `cil_experiments/search_config.py`：`LR_CANDIDATES`。
3. `cil_experiments/order_seed_registry.py`：orders 与 seeds。
4. 对应的 `main_exp/<dataset>.py`：`EXP_NAME`、`BACKBONE`、`LOSS_SELECTION` 和 `METHOD_GPUS`。

正式运行示例：

```bash
python main_exp/spike.py --stage all
python main_exp/texture.py --stage all
python main_exp/uwave.py --stage all
```

若调度器会杀死登录 shell，应将同一命令写入集群自己的 Slurm/PBS 作业脚本。作业脚本与分区、GPU、CPU、内存和时限强相关，本仓库不写死未经确认的集群资源参数。

## 结果与恢复

运行结果位于 `search_result_<EXP_NAME>/` 和 `joint_result_<EXP_NAME>/`，这些目录被 Git 忽略。将结果从超算下载到独立归档位置，不要提交 checkpoint 或训练日志。中断后使用相同 `EXP_NAME` 和同一配置重新执行相应 stage；身份或 schema 不一致时必须更换实验名，不能混合旧产物。
