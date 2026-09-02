# spike / joint_cumulative 手动对照结果

## 配置

- 数据集：`spike`
- 方法：`joint_cumulative`
- 原始标签任务分组：`((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19))`
- order_id / seed / epochs：`6 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 | 测试任务 5 |
|---|---|---|---|---|---|
| 任务 1 | 100.00% | — | — | — | — |
| 任务 2 | 95.48% | 99.36% | — | — | — |
| 任务 3 | 90.97% | 87.18% | 82.91% | — | — |
| 任务 4 | 91.61% | 94.23% | 88.61% | 87.18% | — |
| 任务 5 | 76.77% | 94.87% | 92.41% | 82.05% | 90.38% |

## 汇总

- Average Incremental Accuracy：`0.924290`
- Final Average Accuracy：`0.873239`
- Final Old-task Weighted Accuracy：`0.865600`
- Final New-task Accuracy：`0.903846`
- Average Forgetting：`0.082103`
- 训练耗时：`89.443s`，分任务 `[6.075, 11.97, 17.773, 23.743, 29.882]`
- final_stage_accessible_training_samples：`3219`
- all_class_macro_accuracy：`0.8729419767856598`
- last_class_raw_label：`19`
- last_class_accuracy：`0.8205128312110901`
- preceding_classes_macro_accuracy：`0.8757014055001108`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
