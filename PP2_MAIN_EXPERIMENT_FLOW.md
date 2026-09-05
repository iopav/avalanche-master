# PP2 单个主实验完整流程

本文说明一次普通方法主实验从哪里启动、依次调用哪些文件和函数、每个文件负责什么，以及搜索 checkpoint、joint reference、正式训练和聚合之间的数据关系。以下以 `main_exp/ewc.py` 为例，其他方法只替换入口脚本和方法名。

## 1. 入口参数与启动命令

入口位于 `main_exp/ewc.py`。实验前只需要修改文件顶部的 `EXP_NAME`、`GPU` 和 `BACKBONE`：`EXP_NAME` 是普通字符串，同时决定 `search_result_<EXP_NAME>`、`result_<EXP_NAME>` 和 `joint_result/<EXP_NAME>` 三组目录；`GPU="0"` 使用第一张可见显卡，`GPU="cpu"` 使用 CPU；正式 backbone 通常写 `resnet18_cifar`，最小流程检查可写 `temporal`。入口最后调用 `cil_experiments.pipeline.run_method_pipeline(method="ewc", ...)`，自身不包含训练、搜索或聚合实现。

```powershell
# 先生成固定的 search train / validation 文件
python main_exp/prepare_validation_splits.py

# 完整执行：LR 搜索 -> joint 测试 -> 正式训练 -> 聚合
python main_exp/ewc.py --stage all

# 也可以分阶段执行，后续阶段会读取前一阶段的落盘结果
python main_exp/ewc.py --stage search
python main_exp/ewc.py --stage joint
python main_exp/ewc.py --stage formal
python main_exp/ewc.py --stage aggregate
```

`--datasets uwave texture spike` 控制运行的数据集；聚合阶段要求三个注册数据集齐全。`--dataset-root <目录>` 可替换默认的 `avalanche-master/dataset`。`--skip-intransigence` 只用于暂时跳过正式结果中的 intransigence 填充，不能产生完整发布结果。

## 2. 总调用链

```text
main_exp/ewc.py
  -> cil_experiments.pipeline.run_method_pipeline()
       -> lr_search.run_search()
            -> runner.run_experiment(data_role="search") × 每个 order × 3 个 LR
                 -> data.build_dataset_bundle(search)
                 -> strategies.build_strategy()
                 -> runner._train_experience()
                 -> runner._evaluate_experience()
                 -> checkpointing.save_search_checkpoint()
            -> 选择 validation accuracy 最好的 LR 和完整 checkpoint
       -> runner.evaluate_joint_checkpoint() × 每个 order × joint seeds
            -> 只取 checkpoint 中独立保存的 weight_only_model
            -> data.build_dataset_bundle(formal) 只用于 test stream
            -> 输出 accuracy_by_task 一行
       -> runner.run_experiment(data_role="formal") × 每个 order × formal seeds
            -> 从头构造模型并用完整 train stream 训练
            -> 在 test stream 上形成真正的下三角 accuracy matrix
            -> intransigence.fill_intransigence()
                 -> joint accuracy_by_task - CIL matrix diagonal
       -> aggregate_results 写方法报告、正式汇总和 joint 测试报告
```

## 3. LR 搜索阶段

`cil_experiments/pipeline.py` 的 `run_method_pipeline()` 先确定项目目录、数据目录和三个结果根目录，再按 `order_seed_registry.ORDERS_BY_DATASET[dataset]` 遍历 order。它调用 `cil_experiments/lr_search.py` 的 `run_search()`；后者读取固定的三个候选学习率和 seed 62，为每个 order、每个候选构造方法参数，然后调用统一训练入口 `runner.run_experiment(data_role="search")`。因此 search 与 formal 没有两套训练循环，区别只在数据角色、epoch 配置、输出目录以及 search 是否保存 checkpoint。

`runner.run_experiment()` 是一次有训练过程的实验总控函数。它校验方法/backbone，调用 `data.build_dataset_bundle()` 选择数据，调用 `strategies.build_strategy()` 创建模型、优化器、Avalanche strategy 和方法插件，然后按 experience 调用 `_train_experience()`。`_train_experience()` 只负责一个 experience 的实际学习：普通 CIL 方法训练当前 experience；joint 不在这里训练。每个阶段训练结束后，`_evaluate_experience()` 对已经可见的测试或验证任务逐项计算 accuracy，由此形成下三角矩阵。相同入口还收集分阶段 FLOPs、持久化存储、运行时间和日志，并通过 `output.AtomicRunArtifacts` 原子写出 config、summary 和 accuracy matrix。

搜索使用 `X_train_search.npy/Y_train_search.npy` 学习，使用 `X_validation.npy/Y_validation.npy` 评估，不读取 test。每个候选会保存完整 `StrategyBundle`，便于复现或恢复，同时保存一个从 strategy 脱离的 `weight_only_model`。对于 iCaRL 和 FeCAM 的 `TrainEvalModel`，这个独立模型只复制 feature extractor 与可训练 classifier，明确不复制 eval classifier 中的 class mean/covariance；其他方法也会清除 class mean、covariance、CWR 临时统计等方法状态。标准网络 buffer（例如 BatchNorm running statistics）属于网络推理状态并保留，否则 ResNet 的 eval 行为会失真。`lr_search.finalize_search()` 按 validation 指标选择每个 order 的最佳 LR，把对应完整 checkpoint 提升为稳定的 best checkpoint，并删除未选候选 checkpoint；一个 order 的 search FLOPs 是三个候选训练 FLOPs之和，search memory 指标是最终保留模型的 parameter bytes。

## 4. Joint reference 阶段

这里的 joint 表示“固定最佳搜索模型在测试集上的一次整体参考评估”，不执行 joint training，也不从头训练。`pipeline.run_method_pipeline()` 从 search JSON 找到当前 order 的 best checkpoint，对 joint seed 列表调用 `runner.evaluate_joint_checkpoint()`。checkpoint 内仍完整保存原 strategy，但该函数不会取得 `bundle.strategy`、optimizer、plugin、replay buffer、class mean 或 covariance；它只取得保存时已经分离的 `weight_only_model`，在正式数据角色的 test stream 上逐任务评估。

joint artifact 虽因兼容现有文件定位仍命名为 `__accuracy-matrix.json`，内容的核心字段只有一行 `accuracy_by_task`，例如 `[0.81, 0.76, 0.69]`，依次表示同一个固定模型在当前 order 三个测试 experience 上的准确率。它不是训练轨迹，不能复制成三角矩阵。seed 只控制测试 DataLoader 和运行确定性；模型权重始终来自 seed 62 的最佳 search checkpoint，因此不同 seed 的结果理论上应相同，只有底层非确定性等因素可能造成细微差别。

## 5. 正式训练与 intransigence

正式阶段仍调用 `runner.run_experiment(data_role="formal")`，但会重新构造 backbone、strategy 和 optimizer，从随机初始化开始训练，不加载 search checkpoint。它只读取 search 选出的 LR，并用数据集完整 train 文件学习、test 文件评估。第 t 个 experience 学完后评估第 1..t 个测试 experience，所以正式方法 artifact 中的 `accuracy_matrix_lower_triangular[t][k]` 具有明确的阶段含义；`task_end_seen_accuracy_curve[t]` 是该行按各任务测试样本数加权后的 seen-task 总准确率，因此通常不等于行内某一个单元格。

`runner.run_experiment()` 在写出正式 summary 后查找同 dataset、order、seed、backbone、input view 和 task groups 的 joint summary，然后调用 `intransigence.fill_intransigence()`。第 k 项使用下面的公式，reference 直接来自 joint 的一行结果，不读取任何伪造 joint 矩阵：

```text
intransigence[k] = joint_accuracy_by_task[k] - cil_accuracy_matrix[k][k]
```

这里比较的是 CIL 方法刚学完任务 k 时对任务 k 的准确率与固定 joint 模型对同一测试任务的准确率。若 joint seed 仍有一个无法与正式四个 seed 配对，当前代码可以保存该结果，但正式 paired intransigence 与聚合只消费具有同 seed 的 reference；多出的 seed 不能替代缺失配对，也不应混入四 seed 平均值。

## 6. 文件职责索引

- `main_exp/*.py`：七个薄入口，只声明实验名、GPU、backbone 和方法。
- `main_exp/prepare_validation_splits.py`：调用固定验证拆分准备逻辑，生成 search train 与 validation NPY。
- `cil_experiments/pipeline.py`：阶段编排、目录选择、order/seed 遍历、失败记录及聚合触发。
- `cil_experiments/lr_search.py`：三个 LR 候选、候选记录、最佳 LR/checkpoint 选择、search FLOPs 与 parameter bytes 汇总。
- `cil_experiments/runner.py`：统一的单次训练入口、experience 训练/评估、joint 权重测试、summary 与 artifact 生成。
- `cil_experiments/checkpointing.py`：原子保存完整 search strategy，并额外构造不含方法统计量的 joint 推理模型；负责最佳 checkpoint 原子提升。
- `cil_experiments/data.py`：按 `data_role` 选择 search train/validation 或完整 train/test，并建立 Avalanche benchmark。
- `cil_experiments/strategies.py`：根据方法注册配置构造模型、优化器、criterion、Avalanche strategy 和插件。
- `cil_experiments/models.py`：唯一 backbone 注册表及 `resnet18_cifar`、`temporal` 实现。
- `cil_experiments/registry.py`：数据集、方法和训练默认值注册；不保存重复 backbone 表。
- `cil_experiments/order_seed_registry.py`：各数据集 order、search seed 与 formal/joint seed 列表。
- `cil_experiments/final_hyperparameters.py`：正式 epoch、batch size 和方法参数的锁定值。
- `cil_experiments/metrics.py`：由真实 CIL accuracy matrix 计算 ACC、forgetting、BWT 等，并校验 summary 完整性。
- `cil_experiments/intransigence.py`：校验 CIL/joint 身份匹配，用 joint 单行结果和 CIL 对角线填充 intransigence。
- `cil_experiments/flops.py`、`storage.py`：训练/推理 FLOPs 分项和持久化存储统计。
- `cil_experiments/output.py`：统一路径、日志和 config/summary/matrix 的原子提交。
- `cil_experiments/aggregate_results.py`：读取已完成 artifact，输出 search、formal、joint 的 JSON/CSV 报告。

方法级 `train_report` 和 `test_report` 是原始取数表，每个 dataset/order/seed 一行，不计算 mean、CI95 或 std；其中所有 list 指标序列化为一个 JSON 字符串单元格，intransigence 不再按 task 拆成多列。`joint_test_report` 同样逐 seed 保存原始指标。独立的 `formal_aggregate` 仍承担正式统计汇总；search report 则遵循每个 order 原始值加一行 `all orders` 跨 order 统计的专用格式。

## 7. 结果目录与排错顺序

`search_result_<EXP_NAME>/<dataset>/<method>/order-XX/lr-X/` 保存候选 log/config/summary/matrix，不建立 checkpoint 子目录；候选模型暂存在方法级 `checkpoint_candidates/`，选择结束后清理。最佳完整 checkpoint 位于稳定的 `checkpoint/` 目录，文件名包含 `__lr-<无小数点LR>__best.pt`，例如 `0.1` 编码为 `01`、`0.01` 编码为 `001`、`0.05` 编码为 `005`；`joint_result/<EXP_NAME>/<dataset>/joint/` 保存各 seed 的单行测试 reference；`result_<EXP_NAME>/<dataset>/<method>/` 保存正式四 seed 结果及填充后的 intransigence；聚合结果写入正式结果根目录的 `aggregate_results/`。发生错误时先读对应 `errors/` JSON 的 `failed_stage` 和 traceback，再核对同次运行的 config；不要手工拼接不同 exp name、order、seed 或 backbone 的 joint reference，身份校验会拒绝这种组合。
