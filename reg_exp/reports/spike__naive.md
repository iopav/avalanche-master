# spike / naive 手动对照结果

## 配置

- 数据集：`spike`
- 方法：`naive`
- 原始标签任务分组：`((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19))`
- order_id / seed / epochs：`7 / 62 / 10`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 |
|---|---|---|---|---|
| 任务 1 | 89.69% | — | — | — |
| 任务 2 | 0.00% | 90.82% | — | — |
| 任务 3 | 0.00% | 0.00% | 48.47% | — |
| 任务 4 | 0.00% | 0.00% | 0.51% | 46.67% |

## 汇总

- Average Incremental Accuracy：`0.408308`
- Final Average Accuracy：`0.117798`
- Final Old-task Weighted Accuracy：`0.001706`
- Final New-task Accuracy：`0.466667`
- Average Forgetting：`0.761554`
- 训练耗时：`10.023s`，分任务 `[2.887, 2.447, 2.429, 2.26]`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
