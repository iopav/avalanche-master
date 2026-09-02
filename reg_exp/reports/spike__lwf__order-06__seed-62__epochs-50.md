# spike / lwf 手动对照结果

## 配置

- 数据集：`spike`
- 方法：`lwf`
- 原始标签任务分组：`((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19))`
- order_id / seed / epochs：`6 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'alpha': 5.0, 'temperature': 2.0}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 | 测试任务 5 |
|---|---|---|---|---|---|
| 任务 1 | 100.00% | — | — | — | — |
| 任务 2 | 0.00% | 98.72% | — | — | — |
| 任务 3 | 0.00% | 11.54% | 97.47% | — | — |
| 任务 4 | 0.00% | 0.00% | 0.00% | 90.38% | — |
| 任务 5 | 0.00% | 0.00% | 0.00% | 0.00% | 98.08% |

## 汇总

- Average Incremental Accuracy：`0.456683`
- Final Average Accuracy：`0.195903`
- Final Old-task Weighted Accuracy：`0.000000`
- Final New-task Accuracy：`0.980769`
- Average Forgetting：`0.966427`
- 训练耗时：`42.257s`，分任务 `[6.966, 8.959, 8.745, 8.857, 8.73]`
- all_class_macro_accuracy：`0.1961538463830948`
- last_class_raw_label：`19`
- last_class_accuracy：`0.9743589758872986`
- preceding_classes_macro_accuracy：`0.1551956816723472`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
