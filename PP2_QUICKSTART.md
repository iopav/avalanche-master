# PP2 实验从零上手

这份说明面向第一次接触本仓库的人。所有命令都在 `avalanche-master` 目录执行。流水线会先用固定验证拆分搜索学习率并保存最佳完整 checkpoint，再把其中的纯模型作为 joint reference 测试，最后用最佳学习率从头训练普通方法并聚合结果。

## 1. 安装

建议使用 Python 3.10，并先创建独立环境。GPU 版 PyTorch 对 CUDA 版本有要求；若 `pip install -r requirements.txt` 安装到的 PyTorch 与本机 CUDA 不匹配，应先按 PyTorch 官方安装命令安装合适版本，再安装其余依赖。

```powershell
cd D:\workspace\Avalanche\avalanche-master
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

仓库已包含本地 Avalanche 源码，从 `main_exp/*.py` 启动时会自动加入项目路径，不需要另装 `avalanche-lib`。

## 2. 放置数据

默认数据根目录是 `avalanche-master/dataset/`，三个数据集的实际文件名与 shape 定义在 `cil_experiments/registry.py`。正式运行需要完整 train/test 文件。Texture 和 UWave 还需要 image-view NPY；Spike 保留二值时间序列，由模型入口转成 RGB image。数据在其他目录时，运行命令追加：

```powershell
--dataset-root D:\path\to\dataset
```

生成固定搜索训练集和验证集：

```powershell
python main_exp/prepare_validation_splits.py
```

只准备 UWave：

```powershell
python main_exp/prepare_validation_splits.py --datasets uwave
```

输出包含 `validation_split.json`、`X_train_search.npy`、`Y_train_search.npy`、`X_validation.npy` 和 `Y_validation.npy`。已有拆分不会自动覆盖，明确重建时追加 `--overwrite`。

## 3. 填写实验参数

选择 `main_exp/` 下的普通方法脚本，例如 `main_exp/ewc.py`。修改文件顶部三个值：

```python
EXP_NAME = "ewc_resnet18_trial1"
GPU = "0"
BACKBONE = "resnet18_cifar"
```

- `EXP_NAME`：普通字符串，用于隔离本次 search、joint reference 和 formal result。
- `GPU`：`"0"`、`"1"` 等表示指定 GPU，`"cpu"` 表示只用 CPU。
- `BACKBONE`：正式实验填 `"resnet18_cifar"`；小规模流程检查可填 `"temporal"`。

LR candidates、搜索 epoch 和拆分比例在 `cil_experiments/search_config.py`；各方法正式 epoch、batch size 和方法参数在 `cil_experiments/final_hyperparameters.py`；orders 和 seeds 在 `cil_experiments/order_seed_registry.py`。

## 4. 运行实验

一次运行搜索、joint 测试、formal 训练和聚合：

```powershell
python main_exp/ewc.py
```

按阶段运行：

```powershell
python main_exp/ewc.py --stage search
python main_exp/ewc.py --stage joint
python main_exp/ewc.py --stage formal
python main_exp/ewc.py --stage aggregate
```

搜索对每个 order 使用 seed 62 运行三个 LR。每个 candidate 保存完整 StrategyBundle，选出最佳后只保留最佳 checkpoint。`search_flops` 是三次 candidate 学习 FLOPs 之和；`search_storage_bytes` 只等于最佳模型的 parameter bytes。

joint 不做独立搜索，也不重新训练。完整 checkpoint 另外保存与 strategy 脱离的 `weight_only_model`；joint 只使用该模型的网络推理状态，不读取 optimizer、plugin、replay、class mean 或 covariance，在 seeds `62, 63, 64, 65, 66` 的正式测试协议下生成一行 `accuracy_by_task` reference。formal 阶段仍为普通方法重新构造 strategy，用完整 train 从头训练，seeds 为 `63, 64, 65, 66`。运行 `--stage formal` 时会先补齐 joint reference，以便训练结束后立即计算 intransigence。

只调试某个数据集：

```powershell
python main_exp/ewc.py --stage search --datasets uwave
python main_exp/ewc.py --stage formal --datasets uwave
```

聚合要求 Spike、Texture、UWave 都完整，因此单数据集调试后不要运行 `--stage aggregate`。确实不计算 intransigence 的临时 formal 可追加 `--skip-intransigence`，但该结果不能通过完整正式聚合检查。

## 5. 查看结果

```text
search_result_<EXP_NAME>/                 搜索 JSON、三个 candidate artifact、最佳 .pt
joint_result/<EXP_NAME>/                  五个 seed 的 joint reference 和 joint_test_report.csv
result_<EXP_NAME>/                        四个 seed 的普通方法 formal artifact 和聚合 CSV
```

重点文件：

```text
search_result_<EXP_NAME>/optimum_lrs.json
search_result_<EXP_NAME>/<dataset>/<method>/search_<dataset>_<method>.json
search_result_<EXP_NAME>/<dataset>/<method>/checkpoint/*__lr-<无小数点LR>__best.pt
joint_result/<EXP_NAME>/<dataset>/joint/summary/*__accuracy-matrix.json
result_<EXP_NAME>/<dataset>/<method>/summary/*__summary.json
result_<EXP_NAME>/aggregate_results/<method>_search_report.csv
result_<EXP_NAME>/aggregate_results/<method>_train_report.csv
result_<EXP_NAME>/aggregate_results/<method>_test_report.csv
result_<EXP_NAME>/aggregate_results/formal_aggregate.json
result_<EXP_NAME>/aggregate_results/formal_aggregate.csv
```

同一个 checkpoint 在固定测试数据上的 accuracy 通常不会因 seed 改变；latency 可以有小幅变化。若 accuracy 变化，应检查非确定算子、数据顺序或评估模式。

## 6. 调用关系

```text
main_exp/<method>.py
  -> pipeline.run_method_pipeline()
     -> lr_search.run_search()
        -> runner.run_experiment(data_role="search")
           -> _train_experience()
           -> save_search_checkpoint(完整 StrategyBundle)
     -> runner.evaluate_joint_checkpoint()
        -> load_search_checkpoint()
        -> weight_only_model 在 test stream 上测试，不训练，输出 accuracy_by_task
     -> runner.run_experiment(data_role="formal")
        -> _train_experience()
        -> 正式测试与 intransigence
     -> aggregate_results.write_method_reports()
```

`run_experiment()` 管理完整的一次训练与 artifact；`_train_experience()` 只负责单个 experience 的 `strategy.train()` 和训练开销测量。search 与 formal 共用这两个函数，formal 没有从 checkpoint 续训。

## 7. 最小可检查样例

仓库保留了三个假数据集的完整小实验结果：

```text
tests/pp2_tiny_run/search_result_tiny_ewc_jointref_v2/
tests/pp2_tiny_run/joint_result/tiny_ewc_jointref_v2/
tests/pp2_tiny_run/result_tiny_ewc_jointref_v2/
```

重新生成：

```powershell
D:\anaconda3\envs\py310\python.exe -m unittest tests.test_pp2_pipeline.PP2PipelineTests.test_tiny_search_checkpoint_joint_and_ewc_formal_runs
```

该样例使用三层 temporal backbone 和 1 epoch，只用于检查 JSON、checkpoint round-trip、accuracy matrix、intransigence 与 aggregate CSV，不用于评价方法效果。
