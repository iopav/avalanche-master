# uwave / lwf 手动对照结果

## 配置

- 数据集：`uwave`
- 方法：`lwf`
- 原始标签任务分组：`((0, 1, 2, 3, 4), (5, 6, 7))`
- order_id / seed / epochs：`7 / 62 / 10`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'alpha': 5.0, 'temperature': 2.0}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 |
|---|---|---|
| 任务 1 | 87.61% | — |
| 任务 2 | 0.00% | 84.52% |

## 汇总

- Average Incremental Accuracy：`0.590389`
- Final Average Accuracy：`0.304688`
- Final Old-task Weighted Accuracy：`0.000000`
- Final New-task Accuracy：`0.845201`
- Average Forgetting：`0.876091`
- 训练耗时：`6.974s`，分任务 `[3.558, 3.416]`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
