# spike / er 手动对照结果

## 配置

- 数据集：`spike`
- 方法：`er`
- 原始标签任务分组：`((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19))`
- order_id / seed / epochs：`6 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'memory_size': 2000, 'batch_size_mem': 32}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 | 测试任务 5 |
|---|---|---|---|---|---|
| 任务 1 | 100.00% | — | — | — | — |
| 任务 2 | 87.74% | 62.82% | — | — | — |
| 任务 3 | 91.61% | 94.23% | 90.51% | — | — |
| 任务 4 | 80.65% | 76.28% | 94.30% | 87.18% | — |
| 任务 5 | 86.45% | 92.95% | 93.67% | 71.15% | 80.13% |

## 汇总

- Average Incremental Accuracy：`0.873766`
- Final Average Accuracy：`0.848912`
- Final Old-task Weighted Accuracy：`0.860800`
- Final New-task Accuracy：`0.801282`
- Average Forgetting：`0.078722`
- 训练耗时：`49.523s`，分任务 `[6.96, 10.081, 10.745, 10.735, 11.002]`
- all_class_macro_accuracy：`0.8483788877725601`
- last_class_raw_label：`19`
- last_class_accuracy：`0.5128205418586731`
- preceding_classes_macro_accuracy：`0.8660398533469752`
- 最终实际回放样本数：`2000`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
