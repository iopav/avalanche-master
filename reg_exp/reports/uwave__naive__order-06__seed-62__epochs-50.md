# uwave / naive 手动对照结果

## 配置

- 数据集：`uwave`
- 方法：`naive`
- 原始标签任务分组：`((0, 1, 2, 3), (4, 5, 6, 7))`
- order_id / seed / epochs：`6 / 62 / 50`
- 统一骨干：`cil_experiments.models.TemporalBackbone`，feature_dim=64
- 参数：`{'learning_rate': 0.01, 'momentum': 0.0, 'weight_decay': 0.0, 'train_mb_size': 32, 'eval_mb_size': 128}`
- FLOPs、持久存储、延迟和正式结果 JSON：未计算

## 准确率矩阵

| 训练后 | 测试任务 1 | 测试任务 2 |
|---|---|---|
| 任务 1 | 99.55% | — |
| 任务 2 | 0.00% | 98.89% |

## 汇总

- Average Incremental Accuracy：`0.746084`
- Final Average Accuracy：`0.496652`
- Final Old-task Weighted Accuracy：`0.000000`
- Final New-task Accuracy：`0.988889`
- Average Forgetting：`0.995516`
- 训练耗时：`25.307s`，分任务 `[13.036, 12.271]`
- all_class_macro_accuracy：`0.4945112466812134`
- last_class_raw_label：`7`
- last_class_accuracy：`1.0`
- preceding_classes_macro_accuracy：`0.4222985676356724`

## 解释边界

这是单次 seed 的手动对照，只能作描述性比较；不能据此声称统计显著，也不能证明遗忘的唯一原因是无法访问旧样本。
