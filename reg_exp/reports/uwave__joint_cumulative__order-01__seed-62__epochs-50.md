# uwave / joint_cumulative 手动对照结果

## 配置

- 数据集：`uwave`
- 方法：`joint_cumulative`
- 原始标签任务分组：`((0, 1, 5), (4,), (7,), (6,), (3,), (2,))`
- order_id / seed / epochs：`1 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 | 测试任务 5 | 测试任务 6 |
|---|---|---|---|---|---|---|
| 任务 1 | 99.41% | — | — | — | — | — |
| 任务 2 | 99.12% | 99.21% | — | — | — | — |
| 任务 3 | 98.53% | 99.21% | 99.00% | — | — | — |
| 任务 4 | 98.24% | 99.21% | 99.00% | 100.00% | — | — |
| 任务 5 | 96.48% | 100.00% | 99.00% | 100.00% | 96.36% | — |
| 任务 6 | 97.07% | 97.64% | 99.00% | 100.00% | 95.45% | 98.11% |

## 汇总

- Average Incremental Accuracy：`0.986090`
- Final Average Accuracy：`0.976562`
- Final Old-task Weighted Accuracy：`0.975949`
- Final New-task Accuracy：`0.981132`
- Average Forgetting：`0.011235`
- 训练耗时：`89.025s`，分任务 `[8.194, 11.42, 12.81, 16.937, 17.525, 22.139]`
- final_stage_accessible_training_samples：`3582`
- all_class_macro_accuracy：`0.9771048948168755`
- last_class_raw_label：`2`
- last_class_accuracy：`0.9811320900917053`
- preceding_classes_macro_accuracy：`0.9765295812061855`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
