# uwave / joint_cumulative 手动对照结果

## 配置

- 数据集：`uwave`
- 方法：`joint_cumulative`
- 原始标签任务分组：`((0, 1, 2, 3), (4, 5, 6, 7))`
- order_id / seed / epochs：`6 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 |
|---|---|---|
| 任务 1 | 99.55% | — |
| 任务 2 | 97.98% | 98.44% |

## 汇总

- Average Incremental Accuracy：`0.988829`
- Final Average Accuracy：`0.982143`
- Final Old-task Weighted Accuracy：`0.979821`
- Final New-task Accuracy：`0.984444`
- Average Forgetting：`0.015695`
- 训练耗时：`31.381s`，分任务 `[10.594, 20.787]`
- final_stage_accessible_training_samples：`3582`
- all_class_macro_accuracy：`0.9825550392270088`
- last_class_raw_label：`7`
- last_class_accuracy：`1.0`
- preceding_classes_macro_accuracy：`0.9800629019737244`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
