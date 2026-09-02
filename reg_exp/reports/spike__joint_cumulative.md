# spike / joint_cumulative 手动对照结果

## 配置

- 数据集：`spike`
- 方法：`joint_cumulative`
- 原始标签任务分组：`((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19))`
- order_id / seed / epochs：`7 / 62 / 10`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 |
|---|---|---|---|---|
| 任务 1 | 88.14% | — | — | — |
| 任务 2 | 67.53% | 34.69% | — | — |
| 任务 3 | 70.62% | 65.82% | 54.59% | — |
| 任务 4 | 48.97% | 65.82% | 61.22% | 47.18% |

## 汇总

- Average Incremental Accuracy：`0.646619`
- Final Average Accuracy：`0.558259`
- Final Old-task Weighted Accuracy：`0.587031`
- Final New-task Accuracy：`0.471795`
- Average Forgetting：`0.130584`
- 训练耗时：`21.103s`，分任务 `[2.066, 4.151, 6.269, 8.618]`
- 最终阶段可访问训练样本数：`3219`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
