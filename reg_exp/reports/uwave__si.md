# uwave / si 手动对照结果

## 配置

- 数据集：`uwave`
- 方法：`si`
- 原始标签任务分组：`((0, 1, 2, 3, 4), (5, 6, 7))`
- order_id / seed / epochs：`7 / 62 / 10`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'si_lambda': 0.001, 'eps': 1e-05}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 |
|---|---|---|
| 任务 1 | 87.61% | — |
| 任务 2 | 0.00% | 81.73% |

## 汇总

- Average Incremental Accuracy：`0.585367`
- Final Average Accuracy：`0.294643`
- Final Old-task Weighted Accuracy：`0.000000`
- Final New-task Accuracy：`0.817337`
- Average Forgetting：`0.876091`
- 训练耗时：`23.963s`，分任务 `[15.305, 8.658]`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
