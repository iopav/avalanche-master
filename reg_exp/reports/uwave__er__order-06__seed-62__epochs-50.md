# uwave / er 手动对照结果

## 配置

- 数据集：`uwave`
- 方法：`er`
- 原始标签任务分组：`((0, 1, 2, 3), (4, 5, 6, 7))`
- order_id / seed / epochs：`6 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'memory_size': 2000, 'batch_size_mem': 32}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 |
|---|---|---|
| 任务 1 | 99.55% | — |
| 任务 2 | 98.43% | 98.44% |

## 汇总

- Average Incremental Accuracy：`0.989945`
- Final Average Accuracy：`0.984375`
- Final Old-task Weighted Accuracy：`0.984305`
- Final New-task Accuracy：`0.984444`
- Average Forgetting：`0.011211`
- 训练耗时：`26.709s`，分任务 `[12.222, 14.487]`
- all_class_macro_accuracy：`0.9846096634864807`
- last_class_raw_label：`7`
- last_class_accuracy：`1.0`
- preceding_classes_macro_accuracy：`0.9824110439845494`
- 最终实际回放样本数：`2000`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
