# Avalanche CIL 实施变更记录与 metrics1 合规审计

## 2026-08-22 覆盖性修订

后续正式超参读取缺口已经修复。新增 `cil_experiments/final_hyperparameters.py`，显式保存 Spike/Texture/UWave × ER-ACE/EWC/CWRStar/iCaRL/FeCAM 的15套完整配置。`run_spike.py`、`run_texture.py` 和 `run_uwave.py` 不再接受 epochs 覆盖，正式 runner 根据 dataset-method 读取该文件，并把完整 resolved entry、条目 SHA-256 和任何 diagnostic override 写入 config/log。初始 `FINAL_HYPERPARAMETERS_LOCKED=False`，15组参数全部确认并改为 True 前，正式入口直接失败；带显式 candidate override 的验证诊断不受该锁影响。手工调参和验证搜索不会自动修改最终注册表；选定参数后必须由用户写入对应条目。下文关于“正式入口只读通用 `TRAINING_DEFAULTS`、没有 locked-config registry”的结论已失效。

本节覆盖下文与本轮代码不一致的旧审计结论。ER-ACE 的 storage 按用户指定继续采用“逻辑 replay payload”口径：报告 replay 样本和标签作为训练载荷 materialize 后的字节数，不解释为 Python `Dataset/subset/index` 对象图或 checkpoint 文件的物理体积。`add_like_flop` 也采用用户指定的统一实验约定，即使表达式可写成 `1*a+b`，仍按一次乘法加一次加法计 2 FLOPs；因此 `2*numel` 不再列为错误。adaptive average pooling 已按每个池 `k-1` 次加法加 1 次除法修正，全层计 `input_numel` FLOPs。

FLOPs 的正式指标来自每个 Experience 外围同一个全局 `FlopCounterMode`：该计数器覆盖实际执行的 current、replay、teacher、loss、backward、optimizer、selection 和 statistics 路径，`estimated_cumulative_dense_flops` 是所有 Experience 全局总量之和，所以 replay 已包含在总数中。阶段标签只是把同一总数按当前执行上下文互斥归类，用于审计漏计和定位错误，不会额外相加或重复计算。iCaRL 把 current/replay 样本混合进同一 dataloader，因而 replay 计算在总量和 terminal denominator 中，但 log 无法把混合 minibatch 精确拆成两个 dispatch 阶段；metrics1 的 summary 不要求这种细分，因此不再把它列为正式总量错误。

FeCAM 已按 classifier-incremental 冻结协议重写：首个 Experience 使用固定三类线性头监督训练 backbone；首任务统计完成后将 feature extractor 全部参数冻结；后续 Experience 设置 `train_epochs=0`，不再执行 SGD，只对当前数据做一次特征提取并更新新类均值和协方差。后续统计 pass 被定义为该 Experience 的 terminal executed loop，以保证 metrics1 的 terminal FLOPs/sample 分母有意义。UWave 两任务轻量检查确认第二任务后 `train_epochs=0`、backbone 全部冻结、4 个类统计已登记，第二任务 FLOPs 全部落在 `prototype_and_class_statistics`。

新生成的正式 config、log、summary、accuracy-matrix JSON 均使用同一个带时区时间戳的 stem；search cost JSON、验证 CSV 和手工 accuracy matrix 原本已经带时间戳。历史固定名或无时间戳文件不做破坏性重命名。遗留 `tests/test_cil_experiments.py` 已迁移到当前双 JSON 提交接口和时间戳命名；`tests.test_contracts` 与 `tests.test_cil_experiments` 当前合计 16 pass、0 error。下文关于 FeCAM 持续训练、固定 `config.json`、普通 add 过计、iCaRL replay 总量缺失和 13 pass/1 error 的陈述均已失效。

审计日期：2026-08-21  
审计对象：`D:\workspace\Avalanche\avalanche-master` 当前工作树  
规范来源：`D:\workspace\Avalanche\metrics1.docx` 正文、表格和 17 条 Word 批注，以及其后在对话中确认的覆盖性要求。

## 1. 结论

当前代码不能表述为“完美实现 metrics1”，也不应启动 750 次正式实验。准确率矩阵、按测试样本数加权的 seen accuracy、AIA、final accuracy、forgetting、BWT、旧版 summary 层级、时间戳输出、数据形状检查、共享骨干和失败事务清理等核心机制已经实现；15 个带时间戳的历史矩阵与对应 summary 重新计算后没有发现 CIL 指标差异，当前 `tests.test_contracts` 的 8 个测试也通过。但当前正式方法注册表已经由 FeatureReplay 改为 ER-ACE，结果目录仍是旧的 FeatureReplay 组合，三个 ER-ACE 数据集均没有正式 search/config/log/summary；已搜索参数没有被普通正式入口读取和锁定；所有既有 `config.json` 的源码哈希相对当前代码均已过期；ER-ACE replay 字节数没有按真实底层持久化对象计算；FLOPs 至少有两个可构造的确定公式错误；完整测试命令因遗留测试文件失败；50 个配对条件、`pair_id` 关联、`controller_summary.csv`、置信区间、配对检验、Holm 校正和效应量尚未实现。

因此，当前状态适合继续做单次调试和修正，不具备可辩护的正式 benchmark 完成状态。优先级最高的阻断项是：锁定并让正式入口加载每个 dataset-method 的搜索结果，重新完成 ER-ACE 三组搜索，修正 replay 持久存储实现及 FLOPs 公式，扩充 validator，然后在当前源码上重新生成全部轻量验证结果和源码哈希。

## 2. 证据边界与 metrics1 内部冲突

本目录没有既有 `.git` 历史，因此“哪些文件是后来新增、哪些上游文件曾被改过”无法从 commit diff 反推。下文的新文件清单依据当前目录隔离方式、文件时间、入口依赖和本轮连续实施记录整理；对 Avalanche 上游源码的结论是“当前未发现直接修改，本项目通过旁路包和子类适配”，不能等同于有历史提交证明。Git 从本次审计后建立，只能追踪此后的变化。

`metrics1.docx` 本身有三处必须显式处理的冲突。第一，正文规定 `tasks` 永远为 18，但后续确认要求按真实类别数推广为 Spike=18、Texture=10、UWave=6；当前代码采用后者。第二，正文字段解释把 search 写进 `estimated_cumulative_dense_flops`，批注又要求 `hyperparameter_search_flops` 与正式训练分开；后续确认再次明确搜索成本不得写入正式 summary，当前代码采用后者。第三，批注建议把 FLOPs 扩成 `core_training_flops`、`learning_auxiliary_flops`、`overall_learning_flops` 和 `auxiliary_nonflop_ops`，后续确认要求 summary 严格保留旧层级且 `training_operations` 只能有两个字段；当前代码把详细分相放到 log。由此，当前实现可以评价为“遵循后续覆盖性协议”，无法同时字面满足 DOCX 正文和所有批注。

## 3. 当前正式项目流程

正式入口是 `run_spike.py`、`run_texture.py`、`run_uwave.py`。三个文件调用 `cil_experiments.cli.run_dataset_cli(dataset_name)`；CLI 解析 methods、order IDs、seeds、epochs、device 和路径，然后按 `order -> seed -> method` 调用 `cil_experiments.runner.run_one()`。`run_one()` 依次校验 order/seed 文件、验证并装载数据、生成配置、建立原子输出事务、构造 Avalanche strategy、逐 Experience 训练和评估、生成下三角 accuracy matrix、统计 FLOPs/runtime/memory/latency、构造并校验 summary，最后同时提交 log、summary、accuracy-matrix JSON 和共享 `config.json`。

超参数入口是 `search_hyperparameters.py`。它固定使用 order 1，在训练集内部按类分层抽取 20% 验证集，对每个 dataset-method 运行 3 个候选，根据 validation final average accuracy 选择，AIA 和 candidate ID 用于平局处理；搜索训练、验证、FLOPs、runtime 和峰值显存写入方法目录，并把所有 trial 汇总到 CSV。`--run-selected` 只在该进程内临时改写注册表并紧接着运行一个 selected run，进程结束后恢复默认值。

手工调参入口位于 `manual_tuning/`。每个 dataset-method 有一个独立脚本，复用正式数据、骨干、strategy、seed、order、batch 和无增强协议，但显式关闭 FLOPs、storage、latency 和 summary，只打印并原子保存下三角 accuracy matrix。

## 4. 新增文件及用途

### 4.1 `cil_experiments` 实验包

| 文件 | 用途 | 影响范围 |
|---|---|---|
| `cil_experiments/__init__.py` | 导出数据集、方法、顺序和种子注册表。 | 为所有入口提供统一公共配置。 |
| `cil_experiments/registry.py` | 固化 5 orders、10 seeds、三个数据集物理形状、五个方法参数、共享骨干和训练默认值，并校验 `5order10seeds.txt`。 | 决定所有正式和手工运行的默认协议；当前普通正式入口直接读这里。 |
| `cil_experiments/data.py` | 以 mmap 读取 NPY，验证 shape/dtype/classes，强制 Spike 为 400 time steps 和 0/1，执行 float32 与 `[T,C] -> [C,T]` 技术转换，构造 Avalanche NC benchmark。 | 杜绝错误 Spike 文件、增强和归一化混入；Texture 的 float64 仅在取样时转 float32。 |
| `cil_experiments/models.py` | 定义共享 1D CNN、64 维 feature 和动态分类头，并检查三个骨干仅第一层输入通道不同。 | 五种方法共享同一特征结构；参数量仅因输入通道和最终类别数变化。 |
| `cil_experiments/strategies.py` | 构造 ER-ACE、EWC、CWRStar、定制 iCaRL 和 FeCAM；插入 FLOPs 阶段；修复 iCaRL 动态分类头 teacher；为 Spike iCaRL 实现 bit pack；修复 FeCAM CUDA 输出设备。 | 改变方法训练路径和持久状态，是方法正确性与可比性的核心文件。 |
| `cil_experiments/flops.py` | 用 `FlopCounterMode` 统计实际 dispatch 路径，定义阶段、公式、零 FLOP 分类、未知算子失败、手工补充和单样本前向。 | 产生 summary 的两个 FLOPs 指标及 log 细节；当前仍存在公式准确性问题，见第 9 节。 |
| `cil_experiments/storage.py` | 分别核算模型参数、replay samples、labels、optimizer/teacher/Fisher/prototype 等辅助状态。 | 产生 persistent storage 字段；ER-ACE 的“实际表示”口径目前不成立。 |
| `cil_experiments/metrics.py` | 从下三角矩阵计算加权 seen curve、AIA、final accuracy、forgetting 和 BWT，并校验 summary 键和值之间的部分恒等式。 | 已生成矩阵的性能指标重算一致，但 validator 未覆盖 DOCX 的全部断言。 |
| `cil_experiments/output.py` | 建立带时区时间戳的 log/summary/matrix 命名和跨卷原子提交、回滚、失败清理。 | 防止失败运行发布部分结果；不同时间戳仍允许同一 pair 重复成功输出。 |
| `cil_experiments/runner.py` | 串联数据、strategy、逐任务训练/评估、FLOPs、runtime、memory、latency、fvcore cross-check、summary 和事务提交。 | 正式实验主控制器；当前不读取已经搜索的 method-folder `config.json`。 |
| `cil_experiments/cli.py` | 提供三个数据集入口共用的命令行循环与异常退出。 | 默认只运行 order 1、seed 62、3 epochs，可显式传入正式矩阵。 |

### 4.2 根目录入口和搜索

| 文件 | 用途 |
|---|---|
| `run_spike.py` | 调用共享 CLI 运行 Spike。 |
| `run_texture.py` | 调用共享 CLI 运行 Texture。 |
| `run_uwave.py` | 调用共享 CLI 运行 UWave。 |
| `search_hyperparameters.py` | 训练集内部验证、3-candidate 搜索、搜索成本 JSON、验证 CSV 和可选 selected rerun。 |

### 4.3 手工调参目录

| 文件 | 用途 |
|---|---|
| `manual_tuning/common.py` | 手工运行的共享训练/评估实现，关闭 FLOPs 等正式核算，只保存 matrix。 |
| `manual_tuning/README.md` | 说明可编辑参数、命令和输出边界。 |
| `manual_tuning/spike__er_ace.py` | Spike × ER-ACE 手工参数入口。 |
| `manual_tuning/spike__ewc.py` | Spike × EWC 手工参数入口。 |
| `manual_tuning/spike__cwr_star.py` | Spike × CWRStar 手工参数入口。 |
| `manual_tuning/spike__icarl.py` | Spike × iCaRL 手工参数入口。 |
| `manual_tuning/spike__fecam.py` | Spike × FeCAM 手工参数入口。 |
| `manual_tuning/texture__er_ace.py` | Texture × ER-ACE 手工参数入口。 |
| `manual_tuning/texture__ewc.py` | Texture × EWC 手工参数入口。 |
| `manual_tuning/texture__cwr_star.py` | Texture × CWRStar 手工参数入口。 |
| `manual_tuning/texture__icarl.py` | Texture × iCaRL 手工参数入口。 |
| `manual_tuning/texture__fecam.py` | Texture × FeCAM 手工参数入口。 |
| `manual_tuning/uwave__er_ace.py` | UWave × ER-ACE 手工参数入口。 |
| `manual_tuning/uwave__ewc.py` | UWave × EWC 手工参数入口。 |
| `manual_tuning/uwave__cwr_star.py` | UWave × CWRStar 手工参数入口。 |
| `manual_tuning/uwave__icarl.py` | UWave × iCaRL 手工参数入口。 |
| `manual_tuning/uwave__fecam.py` | UWave × FeCAM 手工参数入口。 |

`manual_tuning/results/uwave/er_ace/...accuracy-matrix.json` 是一次手工诊断生成物，不是正式 summary，也不能替代 ER-ACE search 或正式运行。

### 4.4 测试文件

| 文件 | 用途与当前状态 |
|---|---|
| `tests/test_contracts.py` | 当前合同测试，覆盖 order/seed、数据形状、Spike 400、共享骨干、基础指标、线性层 FLOPs、ER-ACE 独立 buffer、15 个手工入口和原子输出；8 项通过。 |
| `tests/test_cil_experiments.py` | 早期版本合同测试；未随时间戳 matrix 输出接口更新，当前会失败，属于应删除或迁移的遗留文件。 |

### 4.5 生成物

`result/` 当前有 106 个生成文件：75 个 JSON、30 个 log、1 个 CSV。它们来自旧的 FeatureReplay/EWC/CWRStar/iCaRL/FeCAM 矩阵，而当前方法集合已经改成 ER-ACE/EWC/CWRStar/iCaRL/FeCAM。`__pycache__/`、`*.pyc`、`avalanche_lib.egg-info/` 是运行或安装生成物。`dataset/` 是用户提供的数据，不属于代码修改。Git 忽略规则应排除这些生成物与数据。

## 5. 对 Avalanche 原代码的适配及影响

当前未发现对 `avalanche/` 和 `setup.py` 的直接补丁；适配集中在新增的 `cil_experiments` 旁路包。这样做避免污染上游 Avalanche，但也意味着方法行为由“上游 strategy + 本地 subclass/plugin”共同决定，不能简单声明为未改动的官方实现。

ER-ACE 使用 Avalanche `ER_ACE` 的非 online 版本，本地覆写 `training_epoch()` 只为拆开 current/replay FLOPs 阶段；buffer 在每个 task 开始前纳入新类数据，行为继承上游的 non-online adaptation。EWC 直接使用 Avalanche `EWC`，mode=`separate` 时上游会强制保留每个历史 Experience 的 importances 和 saved parameters。CWRStar 直接使用 wrapper，上游硬编码 `freeze_remaining_model=True`，因此 Task 1 训练全网，从 Task 2 起只训练 CWR 分类层。iCaRL 没有直接使用 wrapper，而是用 `SupervisedTemplate` 加本地 loss/plugin 重组，以处理动态分类头 teacher 和 Spike bit-packed exemplars；这是功能性改动，必须单独做等价性测试。FeCAM 用 `Naive` 训练 feature extractor 和临时线性分类头，task 结束后只用当前 task 数据更新均值/协方差，并用本地 device-safe classifier 推理；它会持续改变 backbone，却不刷新旧类统计，这与依赖稳定或预训练特征的 FeCAM 典型条件存在显著偏差，也是旧类准确率坍塌的直接候选原因。

## 6. 方法参数、搜索空间和默认值

### 6.1 共享训练参数

| 参数 | 当前值 | 是否搜索 | 说明 |
|---|---:|---|---|
| optimizer | SGD | 否 | 注册表固定。 |
| learning_rate | 0.1 | 部分 | ER-ACE/CWRStar/iCaRL 的候选定义为 0.1/0.05/0.01；历史 CSV 只包含旧 FeatureReplay、CWRStar、iCaRL。EWC 和 FeCAM 搜索时固定 0.1。普通正式入口仍使用 0.1。 |
| momentum | 0.0 | 否 | 固定默认。 |
| weight_decay | 0.0 | 否 | 固定默认。 |
| foreach | False | 否 | 固定，便于逐算子 FLOPs。 |
| train_mb_size | 32 | 否 | 固定。 |
| eval_mb_size | 128 | 否 | 固定。 |
| epochs_per_experience | 本地为 3 | 否 | CLI 参数，不在 search candidate 中。 |
| num_workers | 0 | 否 | 固定。 |

### 6.2 方法特有参数

| 方法 | 参数 | 当前/隐式值 | 搜索状态 | 来源与实际影响 |
|---|---|---:|---|---|
| ER-ACE | memory_size / `mem_size` | 200 | 候选中三次都为 200，实际未搜索 | Avalanche constructor 默认；限制 class-balanced buffer。 |
| ER-ACE | batch_size_mem | 10 | 候选中三次都为 10，实际未搜索 | Avalanche constructor 默认；决定每个 current batch 配对的 replay batch。 |
| ER-ACE | adaptive_size | True | 否 | 上游 `ClassBalancedBuffer` 硬编码；在已见类间均分 200。 |
| EWC | ewc_lambda | 0.4（注册表） | 已历史搜索 0.1/0.4/1.0 | 所有三个数据集历史选择 0.1，但普通入口没有加载该选择。 |
| EWC | mode | separate | 否 | 固定；导致每个 task 的 Fisher/importances 和参数快照持续保留。 |
| EWC | decay_factor | None | 否，隐式默认 | separate 模式不用。 |
| EWC | keep_importance_data | 调用时默认 False，但 separate 模式内部强制 True | 否，未写入配置 | 影响 auxiliary bytes 随任务数增长；当前配置没有明确记录最终有效值。 |
| CWRStar | cwr_layer_name | `classifier.classifier` | 否 | 指向动态线性分类层。 |
| CWRStar | freeze_remaining_model | True | 否，wrapper 硬编码 | Task 2 起冻结 backbone。 |
| iCaRL | memory_size | 2000 | 候选中三次都为 2000，实际未搜索 | 使用预算上限并按类精确分配。 |
| iCaRL | fixed_memory | True | 否 | 注册表虽有该字段，本地 plugin 构造器却硬编码 True；改变注册表值不会生效。 |
| iCaRL | buffer_transform | None | 否 | 不做增强。 |
| iCaRL | NCM normalize | True | 否，代码固定 | 特征与类均值按 L2 归一化后分类。 |
| FeCAM | tukey | False | 否 | 上游默认 True；因共享 feature 可为负，本地显式关闭。 |
| FeCAM | tukey1 | 0.5 | 否，隐式上游默认 | tukey=False 时当前不生效，但未写入注册表/config。 |
| FeCAM | shrinkage | True | 否 | 上游默认；当前上游实现会连续调用两次 shrink。 |
| FeCAM | shrink1, shrink2 | 1.0, 1.0（注册表） | 已历史搜索绑定组合 (1,1)/(0.5,0.5)/(2,2) | 三个数据集历史均选择 (1,1)，没有独立搜索两个维度。 |
| FeCAM | covnorm | True | 否 | 上游默认。 |
| FeCAM | statistics update momentum | 0.5 | 否，隐式默认 | 当前每类只在首次出现时写入，通常不触发动量合并；未写入配置。 |

当前搜索定义已经含 ER-ACE，但没有执行记录。2026-08-21 14:49:51 的唯一 CSV 是替换方法之前生成的历史结果，包含 FeatureReplay 而不包含 ER-ACE。历史 selected candidates 如下，仅可作为旧代码下的诊断结果，不能当作当前正式锁定配置：

| 数据集 | FeatureReplay（已废弃） | EWC | CWRStar | iCaRL | FeCAM |
|---|---|---|---|---|---|
| Spike | lr=0.01, mem=200, replay batch=10 | lr=0.1, lambda=0.1 | lr=0.1 | lr=0.01, mem=2000 | lr=0.1, shrink=(1,1) |
| Texture | lr=0.01, mem=200, replay batch=10 | lr=0.1, lambda=0.1 | lr=0.1 | lr=0.01, mem=2000 | lr=0.1, shrink=(1,1) |
| UWave | lr=0.01, mem=200, replay batch=10 | lr=0.1, lambda=0.1 | lr=0.05 | lr=0.1, mem=2000 | lr=0.1, shrink=(1,1) |

## 7. 已验证正确的部分

1. 数据合同：三个 NPY 数据集当前 shape/dtype 检查通过；Spike 明确读取 `(N,400,64)` 文件并逐块验证 0/1，不会静默接受 1600-step 文件。没有 normalization 或 augmentation，只有 float32 与转置。
2. 骨干合同：三个输入分别为 64/9/3 channels，其余 conv channels、kernel、stride、padding、GroupNorm 和 64-d projection 相同；实际数据长度前向测试通过。
3. 性能指标：15 个带时间戳 accuracy matrix 重新计算后，与对应 summary 的所有 `cil_performance` 字段一致。seen accuracy 使用每个 task group 的真实 test sample count 加权，不能解释成简单任务平均。
4. 输出 schema：现存 30 个 summary 均通过当前旧版 schema 键检查；时间戳 selected rerun 额外保存了 15 个下三角 matrix。旧的 15 个无时间戳 summary 没有配套 matrix，无法做同级独立重算。
5. 原子输出：当前测试证明成功时四项共同发布、失败时临时目录清理、同一精确时间戳拒绝覆盖。
6. 当前合同测试：`python -m unittest tests.test_contracts -v` 对应的 8 项在本轮组合执行中全部通过。

## 8. metrics1 要求映射

| 要求 | 状态 | 审计判断 |
|---|---|---|
| 5 orders × 10 seeds 配对 | 未完成 | 注册表存在；750 次按要求未在本机运行。没有完成 50-run/method 结果。 |
| 同 pair 的 order/seed/split/budget/evaluation 一致 | 部分 | order/seed/budget/evaluation 可统一；没有显式 pair_id 注册和跨方法 controller 校验。 |
| Accuracy matrix 与公式 | 已实现 | 当前带 matrix 的 15 组重算一致；按后续要求推广到 18/10/6 tasks。 |
| Summary 旧版层级 | 已实现 | 顶层和子层键符合 DOCX 正文模板。 |
| 每次运行时间戳 log/JSON | 部分 | 新 run 的 log/summary/matrix 有时间戳；固定 `config.json` 无时间戳，旧无时间戳结果仍保留。 |
| FLOPs 实际路径、1 MAC=2 FLOPs | 部分 | 主后端与阶段机制已实现；公式和阶段拆分仍有确定错误/缺口。 |
| 未知算子审计 | 部分 | dispatch 未注册计算算子会失败；已注册但公式返回 0、近似公式或错误公式不会失败。 |
| search 成本与正式训练分离 | 部分 | 文件和字段分离；ER-ACE 未搜索，历史 search 记录缺少源码哈希和完整解析配置。 |
| selected config 锁定 50 runs | 失败 | 普通正式入口不读取 selected config；只在 search 进程内临时应用。 |
| Persistent storage 四项互斥且为实际表示 | 失败 | iCaRL 有实际 tensor/bit-packed 表示检查；ER-ACE 仍是 dataset subset/index 引用，却按 materialized float32 样本估算。跨类别去重 counter 也不统一。 |
| GPU working memory 与 storage 分离 | 已实现 | summary 独立记录 peak allocated memory，不加入 total bytes。 |
| 失败不留疑议结果 | 正式 run 基本实现 | `AtomicRunArtifacts` 可回滚；search 的批次级别不具备全 15 组合事务。 |
| controller_summary.csv 和统计检验 | 未实现 | 无聚合 controller、paired CI/test/Holm/effect size。 |

## 9. 确定问题与计算错误

### P0：正式实验阻断项

1. **搜索结果没有锁定到正式入口。** `run_*.py -> cli.py -> run_one()` 直接读取进程内 `TRAINING_DEFAULTS` 和 `METHODS`，不会读取 `result/<dataset>/<method>/config.json` 或 search cost JSON。反例是 UWave CWRStar 的历史选择 lr=0.05，但正常执行 `run_uwave.py --methods cwr_star` 仍用 lr=0.1。修复方向是生成独立、只读、带 hash 的 locked-config registry，正式 runner 必须要求并校验它，debug 模式才允许 source defaults。
2. **当前五方法矩阵缺三个 ER-ACE 组合。** 结果目录只有 FeatureReplay 历史结果；ER-ACE 只有一次 UWave 手工 matrix 和一次未发布 probe，没有三个数据集的 3-candidate search 或正式 summary。
3. **既有结果相对当前源码全部过期。** 15 个 method-folder 的 config 均至少在 `registry.py`、`runner.py`、`storage.py`、`strategies.py` 和 ER-ACE source marker 上与当前 hash 不同。旧 summary 的数值内部一致，不代表当前代码复现结果。
4. **ER-ACE persistent replay 不是所报告的实际表示。** 上游 `ClassBalancedBuffer -> ReservoirSamplingBuffer` 保存的是 `AvalancheDataset.subset(indices)`，本地 `_er_ace_bytes()` 逐项调用 dataset 后得到临时 float32 tensor，再把这些临时 tensor 的 `numel * element_size` 当持久字节。底层对象实际保留 dataset/subset/index 关系，并没有把 200 个 tensor 独立持久化。当前 `float32_raw_timeseries` 描述和 byte 数不能称为实际 checkpoint 表示。需要实现明确的 tensor/packed buffer serialization，再按序列化对象去重计数并做 save/load 测试。
5. **完整 metrics1 聚合尚不存在。** 没有 50 个 matched pair、pair manifest、`controller_summary.csv`、paired confidence interval、检验、Holm correction 和 effect size，不能完成最终比较。

### P1：确定代码或方法学错误

6. **完整实验测试失败。** 执行 `python -m unittest tests.test_contracts tests.test_cil_experiments -v` 得到 14 项中 13 项通过、1 项 error；错误是遗留 `tests/test_cil_experiments.py` 调用已经改为双参数的 `AtomicRunArtifacts.commit()` 时仍只传 summary。它还断言旧的无时间戳文件名。该文件应删除或同步迁移，CI 应只保留一个权威合同测试集。
7. **summary validator 没有执行 DOCX 声称的全部一致性断言。** `validate_summary()` 不接收 accuracy matrix，无法重新计算 forgetting；也没有核对 median/std/quantiles/worst task/positive fraction 是否来自 `forgetting_per_task`，没有核对 total FLOPs 是否等于逐 task 总和、terminal 指标是否等于逐 task terminal 之和，也没有跨 50 runs 检查 keys/pair_id。当前 runner 恰好用同一函数生成字段，所以本轮重算一致，但 validator 本身不能阻止错误或篡改 summary 发布。
8. **`add_like_flop` 对普通加法确定过计。** 代码无条件返回 `2 * numel`，理由是带 `alpha` 的一般形式需要乘加；但大量 `aten.add/add_` 的 alpha=1，只执行一次浮点加法。N=10 的普通 add 当前记 20 FLOPs，按所采用的浮点算术口径应为 10。应读取实际 alpha：alpha=1 计 N，其他 alpha 计 2N。
9. **`adaptive_pool_flop` 确定过计。** 从 4 个数平均到 1 个输出，需要 3 次加法和 1 次除法，共 4 FLOPs；当前公式 `input_numel + output_numel` 返回 5。一般平均池化应为每个池 `k-1` 次加法加 1 次除法，即总计 input_numel，而不是 input_numel+output_numel。
10. **部分 FLOPs 公式只是启发式且 log 未记录公式正文。** GroupNorm 的 7N/15N、loss、SVD/pinv、exp/sqrt 等没有在 log 中保存公式、假设和误差界；log 只列注册的 operator 名称。`var` 直接复用 mean 公式，也不是方差的算术量。由此只能称为统一估计，不能声称“确保算全算对”。fvcore 仅对最终单样本 inference 做 cross-check，不验证完整训练阶段。
11. **iCaRL replay FLOPs 没有独立阶段。** iCaRL 把 exemplars 直接 concat 到 adapted training dataset，plugin 看到的 minibatch 已混合 current/replay，因此全部记为 `student_current_forward`，`replay_forward` 为零。terminal denominator 仍包含混合 batch 的全部样本，但 method-level replay breakdown 不满足要求。
12. **wall runtime 包含 FLOP instrumentation 开销。** 每个正式 task 的 `total_s` 在 `FlopCounterMode` 和 Python dispatch audit 激活期间测量，包含逐算子审计、signature 收集和 profiler stop 校验的 CPU 开销。它是“带核算器的实验墙钟时间”，不是原生训练 runtime；不同方法触发算子数不同，会产生方法相关偏差。GPU event 时间较少受 Python 审计影响，但 CPU/host 同步仍会影响调度。应将 FLOPs profiling run 与 clean runtime run 分离，并用相同权重/路径验证。
13. **FeCAM 当前实现存在严重表征漂移。** 旧类均值和协方差在旧 backbone 上计算，但后续 task 继续更新 backbone，旧统计不重算也无 replay；最后新特征空间与旧统计不一致。UWave 历史 matrix 出现旧类逐项归零，符合该失败模式。若目标是忠实 FeCAM，需要按原方法条件冻结/预训练特征，或明确采用保持统计一致性的训练协议；当前实现只能命名为 FeCAM classifier adapter，不能无条件等同论文方法。
14. **search artifact 不自包含。** search JSON 保存 candidate delta，没有保存完整 resolved config、source hashes、PyTorch/Avalanche 版本和候选配置 hash。代码改变后无法证明 CSV 对应哪个实现。尤其现有 CSV 是 FeatureReplay 版本，却与当前 ER-ACE search source 同处一个结果根目录。
15. **`fixed_memory` 是无效可配置字段。** 注册表和手工脚本允许修改 iCaRL `fixed_memory`，但本地 plugin 构造器始终 `fixed_memory=True`。这会让配置看似可调而执行路径不变。应删除该可配假象或真正传递字段。

### P2：完整性与可维护性问题

16. **同一 order/seed 可重复发布。** overwrite 检查针对带时间戳的完整文件名；时间戳不同就能为同一 pair 产生多个成功 summary。应建立 run identity index，默认拒绝同 dataset-method-order-seed-config-hash 的重复正式提交。
17. **storage 去重范围不全局。** model parameters 使用一个 `ByteCounter`，model buffers/optimizer/method state 使用另一个；若跨类别共享 storage，仍可能重复计数。当前模型未观察到这种共享，结构上却未满足“互斥”的强保证。
18. **部分 metadata bytes 是约定值而非真实序列化字节。** ER-ACE 每组固定 16 bytes、iCaRL/CWR 的 Python 类 ID/计数固定 8 bytes，没有 checkpoint 编码依据或说明。必须先定义持久化格式，再测量文件内有效 payload 或按明确 schema 解析计算。
19. **没有 checkpoint round-trip。** 代码只统计 live objects，不实际保存并重新加载模型、buffer、teacher/Fisher/prototype 后验证 inference 和继续学习。由此不能证明 auxiliary state 既充分又无冗余。
20. **数据合同只验证 train/test 标签并集。** 某类若只存在于 train 或只存在于 test，并集检查仍通过，后续会产生空 Experience。应分别验证每个 split 的类覆盖和每 task 非空。
21. **测试含绝对路径。** 两个实验测试文件硬编码 `D:\workspace\Avalanche\avalanche-master`，仓库迁移后失效。应从 `__file__` 推导 project root。
22. **无 Triton 计数能力。** 当前环境打印 `triton not found; flop counting will not work for triton kernels`。现有 Conv/Linear 路径没有 Triton kernel，当前运行不因此漏计；若未来启用 `torch.compile` 或 Triton 方法，strict dispatch audit也未必覆盖内核内部 FLOPs，必须禁止或增加后端计数。

## 10. 本轮审计执行记录

- 完整提取并核对 `metrics1.docx` 正文与 17 条批注。
- 当前源文件共 33 个 Python 文件通过 AST 解析。
- `tests.test_contracts` 的 8 个测试通过。
- 与遗留 `tests.test_cil_experiments` 合并运行时总计 14 项，13 pass、1 error；错误位置和原因见第 9.6 节。
- 三个数据集 source shape/dtype、Spike 400/0-1、骨干输出和 order/seed 注册表测试通过。
- 30 个现存 summary 通过当前 schema validator；15 个带时间戳 matrix 的全部 CIL 字段重算一致。
- 当前方法集合应有 15 个 dataset-method 组合，结果树只有 12 个当前组合，缺 Spike/Texture/UWave × ER-ACE；另有 3 个已废弃 FeatureReplay 组合。
- 15 个现存 `config.json` 均检测到当前源码 hash 漂移。
- FLOPs 反例实测：N=10 普通 add 公式返回 20、理论为 10；4-to-1 average pool 公式返回 5、理论为 4。

## 11. 修复顺序

1. 建立 `locked_hyperparameters/<dataset>/<method>.json` 或等价只读注册表，写入完整配置、source/config hash、搜索成本文件 hash；正式 runner 缺失锁定文件时直接失败。
2. 修正 iCaRL `fixed_memory` 传参，明确 ER-ACE 可序列化 replay 表示，做 checkpoint save/load/continue round-trip，再重写 storage 核算为全局去重。
3. 修正 add/average-pool/variance/loss 等 FLOPs 公式，把实际公式、变量、误差假设写入 log；为 iCaRL replay、EWC Fisher 等补充分相测试；分离 clean runtime 与 profiling runtime。
4. 决定并固定 FeCAM 的 feature protocol，再重跑验证。否则低准确率属于实现协议问题，继续扩大 grid 没有解释力。
5. 让 `validate_summary(summary, matrix, per_task_flops)` 真正执行 metrics1 全部单 run 断言；删除或迁移遗留测试，增加各方法 storage/FLOPs/search-lock/end-to-end 测试。
6. 对当前五方法重新执行三个数据集的三候选搜索，锁定配置；随后重跑 15 个 order1/seed62 轻量验证并确认 hash 全新一致。
7. 工作站执行 750 runs 后再生成 pair controller、`controller_summary.csv`、CI、paired tests、Holm correction 和 effect sizes。

## 12. Git 基线策略

从本次审计后初始化 Git。首个基线提交只纳入 Avalanche 源码、`cil_experiments`、入口、搜索器、手工调参脚本、测试和本文档；`.gitignore` 明确排除 `dataset/`、`result/`、`manual_tuning/results/`、NPY/NPZ、cache、coverage、egg-info 和 IDE 文件。由于缺少此前 Git 历史，该基线提交表示“当前已知状态”，不表示每个文件的原始作者或早期修改顺序。
