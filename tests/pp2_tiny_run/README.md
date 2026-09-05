# PP2 tiny run 检查说明

本目录由 `tests.test_pp2_pipeline.PP2PipelineTests.test_tiny_cumulative_then_ewc_complete_formal_runs` 生成并保留，用于人工核对 PP2 search、formal artifact 和聚合格式。`spike/`、`texture/`、`uwave/` 分别保存一个两类假数据集，每个数据集有 8 个训练样本、4 个测试样本和两个 experience；cumulative 和 EWC 都使用三层 `temporal` backbone，search 与 formal 的每个 experience 都训练 1 epoch。Search seed 为 62，formal seed 为 7。

搜索结果分别位于 `search_result_tiny_cumulative/` 和 `search_result_tiny_ewc/`。每个 dataset/method 目录包含 `search_<dataset>_<method>.json`，每个候选目录包含该 LR 的完整 config、log、summary 和 accuracy matrix，根目录的 `optimum_lrs.json` 保存最终选择。Cumulative 的正式 artifact 位于 `cumulative_result/<dataset>/cumulative/`，EWC 的正式 artifact 位于 `result_tiny_ewc/<dataset>/ewc/`。两者的 `aggregate_results/` 均由原有 `write_method_reports()` 生成 `<method>_search_report.csv`、`<method>_train_report.csv` 和 `<method>_test_report.csv`。

EWC config 中的 `intransigence.cumulative_summary_path` 应指向本目录的固定 `cumulative_result/`。Intransigence 第 `k` 项使用两张 accuracy matrix 的对角线差 `cumulative[k][k] - ewc[k][k]`；当前两个单类 experience 的对角线都为 `[1.0, 1.0]`，因此结果为 `[0.0, 0.0]`。第二阶段第一类从 1.0 降为 0.0 计入 forgetting，不计入 intransigence。`task_end_seen_accuracy_curve` 是每一行对已见任务按测试样本数加权的平均值，所以矩阵第二行 `[0.0, 1.0]` 在两个任务样本数相同时对应曲线值 0.5。

在 `avalanche-master` 下重新生成：

```powershell
D:\anaconda3\envs\py310\python.exe -m unittest tests.test_pp2_pipeline.PP2PipelineTests.test_tiny_cumulative_then_ewc_complete_formal_runs
```

每个正式组合只有一个 seed，因此报告中的标准差和 95% 置信区间为 0；这些数值只用于检查格式、字段与调用关系，不能解释为正式实验性能。
