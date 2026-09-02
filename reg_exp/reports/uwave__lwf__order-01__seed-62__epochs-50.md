# uwave / lwf 手动对照结果

## 配置

- 数据集：`uwave`
- 方法：`lwf`
- 原始标签任务分组：`((0, 1, 5), (4,), (7,), (6,), (3,), (2,))`
- order_id / seed / epochs：`1 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'alpha': 5.0, 'temperature': 2.0}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 | 测试任务 5 | 测试任务 6 |
|---|---|---|---|---|---|---|
| 任务 1 | 97.65% | — | — | — | — | — |
| 任务 2 | 0.00% | 100.00% | — | — | — | — |
| 任务 3 | 0.00% | 0.00% | 100.00% | — | — | — |
| 任务 4 | 0.00% | 0.00% | 0.00% | 100.00% | — | — |
| 任务 5 | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | — |
| 任务 6 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |

## 汇总

- Average Incremental Accuracy：`0.307702`
- Final Average Accuracy：`0.118304`
- Final Old-task Weighted Accuracy：`0.000000`
- Final New-task Accuracy：`1.000000`
- Average Forgetting：`0.995308`
- 训练耗时：`31.894s`，分任务 `[9.233, 4.735, 4.913, 4.042, 4.539, 4.432]`
- all_class_macro_accuracy：`0.125`
- last_class_raw_label：`2`
- last_class_accuracy：`1.0`
- preceding_classes_macro_accuracy：`0.0`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
