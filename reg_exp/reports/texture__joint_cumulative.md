# texture / joint_cumulative 手动对照结果

## 配置

- 数据集：`texture`
- 方法：`joint_cumulative`
- 原始标签任务分组：`((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11))`
- order_id / seed / epochs：`7 / 62 / 10`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 |
|---|---|---|---|
| 任务 1 | 60.00% | — | — |
| 任务 2 | 46.67% | 17.00% | — |
| 任务 3 | 39.05% | 36.00% | 100.00% |

## 汇总

- Average Incremental Accuracy：`0.462873`
- Final Average Accuracy：`0.466667`
- Final Old-task Weighted Accuracy：`0.375610`
- Final New-task Accuracy：`1.000000`
- Average Forgetting：`0.104762`
- 训练耗时：`7.910s`，分任务 `[1.401, 2.905, 3.604]`
- 最终阶段可访问训练样本数：`960`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
