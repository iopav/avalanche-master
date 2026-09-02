# spike / er 手动对照结果

## 配置

- 数据集：`spike`
- 方法：`er`
- 原始标签任务分组：`((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19))`
- order_id / seed / epochs：`7 / 62 / 10`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'memory_size': 200}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 |
|---|---|---|---|---|
| 任务 1 | 89.69% | — | — | — |
| 任务 2 | 86.60% | 63.78% | — | — |
| 任务 3 | 79.38% | 43.88% | 78.57% | — |
| 任务 4 | 55.15% | 22.45% | 52.04% | 83.08% |

## 汇总

- Average Incremental Accuracy：`0.712979`
- Final Average Accuracy：`0.531370`
- Final Old-task Weighted Accuracy：`0.431741`
- Final New-task Accuracy：`0.830769`
- Average Forgetting：`0.341311`
- 训练耗时：`10.864s`，分任务 `[2.85, 2.82, 2.57, 2.623]`
- 最终实际回放样本数：`200`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
