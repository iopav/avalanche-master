# uwave / lwf 手动对照结果

## 配置

- 数据集：`uwave`
- 方法：`lwf`
- 原始标签任务分组：`((0, 1, 2, 3), (4, 5, 6, 7))`
- order_id / seed / epochs：`6 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128, 'alpha': 5.0, 'temperature': 2.0}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 |
|---|---|---|
| 任务 1 | 99.55% | — |
| 任务 2 | 0.00% | 98.67% |

## 汇总

- Average Incremental Accuracy：`0.745526`
- Final Average Accuracy：`0.495536`
- Final Old-task Weighted Accuracy：`0.000000`
- Final New-task Accuracy：`0.986667`
- Average Forgetting：`0.995516`
- 训练耗时：`29.936s`，分任务 `[12.244, 17.692]`
- all_class_macro_accuracy：`0.49338512122631073`
- last_class_raw_label：`7`
- last_class_accuracy：`1.0`
- preceding_classes_macro_accuracy：`0.4210115671157837`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
