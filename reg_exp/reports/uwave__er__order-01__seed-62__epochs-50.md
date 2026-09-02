# uwave / er 手动对照结果

## 配置

- 数据集：`uwave`
- 方法：`er`
- 原始标签任务分组：`((0, 1, 5), (4,), (7,), (6,), (3,), (2,))`
- order_id / seed / epochs：`1 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'memory_size': 2000, 'batch_size_mem': 32}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 | 测试任务 3 | 测试任务 4 | 测试任务 5 | 测试任务 6 |
|---|---|---|---|---|---|---|
| 任务 1 | 97.65% | — | — | — | — | — |
| 任务 2 | 98.53% | 90.55% | — | — | — | — |
| 任务 3 | 97.65% | 99.21% | 99.00% | — | — | — |
| 任务 4 | 95.89% | 99.21% | 99.00% | 100.00% | — | — |
| 任务 5 | 92.67% | 98.43% | 100.00% | 100.00% | 94.55% | — |
| 任务 6 | 81.52% | 98.43% | 99.00% | 100.00% | 95.45% | 99.06% |

## 汇总

- Average Incremental Accuracy：`0.962825`
- Final Average Accuracy：`0.919643`
- Final Old-task Weighted Accuracy：`0.910127`
- Final New-task Accuracy：`0.990566`
- Average Forgetting：`0.037592`
- 训练耗时：`29.530s`，分任务 `[9.107, 3.744, 4.018, 4.186, 4.583, 3.893]`
- all_class_macro_accuracy：`0.9227000325918198`
- last_class_raw_label：`2`
- last_class_accuracy：`0.9905660152435303`
- preceding_classes_macro_accuracy：`0.913004892213004`
- 最终实际回放样本数：`2000`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
