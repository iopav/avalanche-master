# PP2 快速上手

## 1. 安装与数据

在项目根目录创建环境并安装依赖。CUDA 对应的 PyTorch 版本应按本机驱动选择；若直接执行下列命令，pip 会使用当前索引可用的构建版本。

```powershell
cd D:\workspace\Avalanche\avalanche-master
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

数据默认放在 `avalanche-master/dataset`，文件名必须与 `cil_experiments/registry.py` 的 `DATASETS` 注册一致。当前协议直接用完整训练集搜索和训练 joint，不需要先生成 validation split；`main_exp/prepare_validation_splits.py` 只为旧协议或其他实验保留。

## 2. 填写入口参数

根据数据集打开 `main_exp/spike.py`、`main_exp/texture.py` 或 `main_exp/uwave.py`，修改：

```python
EXP_NAME = "spike_pp2_trial1"
BACKBONE = "resnet18_cifar"
LOSS_SELECTION = "last_epoch_train_mean"

METHOD_GPUS = {
    "ewc": "0",
    "er_ace": "0",
    "icarl": "1",
    "fecam": "1",
    "tagfex": "2",
    "cwr_star": "2",
}
```

`EXP_NAME` 应在每次独立主实验中更换，防止复用旧 artifact。正式实验使用 `resnet18_cifar`；`temporal` 只用于流程检查。loss 选择可写 `last_epoch_train_mean` 或 `full_train_final_model`。GPU 值是物理编号；无 CUDA 时可写 `"cpu"`，但完整 ResNet18 实验会很慢。正式 epoch、batch size、优化器和方法参数位于 `cil_experiments/final_hyperparameters.py`；候选学习率位于 `cil_experiments/search_config.py`；orders 和 seeds 位于 `cil_experiments/order_seed_registry.py`。

## 3. 运行

```powershell
# 完整执行：六方法各自 search 完成后立即 joint，最后写两个 CSV
python main_exp/spike.py --stage all

# 分阶段执行或断训后续跑
python main_exp/spike.py --stage search
python main_exp/spike.py --stage joint
python main_exp/spike.py --stage aggregate

# 数据不在默认 dataset 目录时
python main_exp/spike.py --stage all --dataset-root D:\data\pp2
```

一个方法在一个数据集上的搜索量是 `3 orders × 5 seeds × 3 LR = 45` 次完整 CIL 训练，随后生成 `3 × 5 = 15` 个 joint JSON。每个 joint JSON 内部按 order 分阶段建立累计 joint 参考：Spike 每个 order 18 个阶段，Texture 10 个阶段，UWave 6 个阶段；第 k 阶段从头训练 task 1..k 的累计数据并写入矩阵第 k 行。六种方法并行进程各自遵循 search → joint；共享同一 GPU 的两个方法也会并发，请按显存容量调整 `METHOD_GPUS`。

## 4. 查看输出

搜索和最佳 CIL 产物位于 `search_result_<EXP_NAME>/<method>/orderN/`。每个 order-seed 有一个 `orderN_seedSSS_search.json`，候选 summary/matrix 位于 `lr01`、`lr005`、`lr001`，最佳完整 checkpoint 位于 `checkpoint/`。Joint 原始结果位于 `joint_result_<EXP_NAME>/<method>/orderN/`。

最终只看以下两个 CSV：

```text
search_result_<EXP_NAME>/aggregate_results/<dataset>_search_summary.csv
joint_result_<EXP_NAME>/aggregate_results/<dataset>_joint_learning.csv
```

每个 CSV 有 90 个数据行。第一个文件合并全部搜索字段和最佳 CIL summary；第二个文件保存 joint 的全部原始指标，并把 joint 矩阵对角线 accuracy 展开为数值列。`--stage aggregate` 会核对 90 个 search JSON、90 个 joint JSON、每个 search 的三个候选 summary/matrix、最佳 checkpoint、有效推理时延及 dataset/method/order/seed/task-groups 身份；任何数量、文件或身份缺失都会终止，不会补空行。此时继续执行 `--stage search` 或 `--stage joint`，新协议下已完成且验证通过的单元会自动跳过。

旧 `tests/pp2_tiny_run/*tiny_ewc_weight_only_v4*` 属于先前的权重测试协议，不能与当前结果合并。
