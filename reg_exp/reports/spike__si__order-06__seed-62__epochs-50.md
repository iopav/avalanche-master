# spike / si 手动对照结果

## 配置

- 数据集：`spike`
- 方法：`si`
- 原始标签任务分组：`((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19))`
- order_id / seed / epochs：`6 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'si_lambda': 0.001, 'eps': 1e-05}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 | 测试任务 5 |
|---|---|---|---|---|---|
| 任务 1 | 100.00% | — | — | — | — |
| 任务 2 | 0.00% | 99.36% | — | — | — |
| 任务 3 | 0.00% | 0.00% | 98.73% | — | — |
| 任务 4 | 0.00% | 0.00% | 8.86% | 92.95% | — |
| 任务 5 | 0.00% | 0.00% | 0.00% | 0.00% | 98.72% |

## 汇总

- Average Incremental Accuracy：`0.456520`
- Final Average Accuracy：`0.197183`
- Final Old-task Weighted Accuracy：`0.000000`
- Final New-task Accuracy：`0.987179`
- Average Forgetting：`0.977605`
- 训练耗时：`93.976s`，分任务 `[18.957, 20.209, 19.693, 18.466, 16.651]`
- all_class_macro_accuracy：`0.19743589758872987`
- last_class_raw_label：`19`
- last_class_accuracy：`1.0`
- preceding_classes_macro_accuracy：`0.1551956816723472`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
