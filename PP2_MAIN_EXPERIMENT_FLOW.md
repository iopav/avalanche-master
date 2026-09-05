# 一个 PP2 主实验的完整调用流程

本文以 `python main_exp/spike.py --stage all` 为例，说明入口、文件、函数和 artifact 之间的实际调用关系。Texture 和 UWave 只更换数据集入口、任务数量与输入文件，六种方法的编排方式相同。

## 1. 数据集入口与并行方法进程

`main_exp/spike.py` 只保存本次实验的四类参数：`EXP_NAME`、`BACKBONE`、`LOSS_SELECTION` 和 `METHOD_GPUS`。脚本调用 `cil_experiments.pipeline.run_dataset_pipeline()`；该函数解析 `--stage` 和 `--dataset-root`，建立 `search_result_<EXP_NAME>` 与 `joint_result_<EXP_NAME>` 两个根路径，然后通过 `ProcessPoolExecutor` 为六个方法分别启动一个进程。每个子进程先设置自己的 `CUDA_VISIBLE_DEVICES`，随后才导入 PyTorch，避免父进程提前固定 CUDA 设备。

在 `--stage all` 下，每个方法子进程内部执行：

```text
lr_search.run_method_search(method)
  -> 当前方法的 15 个 order-seed 搜索单元
joint_learning.run_method_joint(method)
  -> 当前方法的 15 个从头 joint 训练
```

因此先完成 15 个搜索单元的方法会直接进入 joint，其他方法仍可继续搜索。主进程等六个 future 全部成功后，调用 `aggregate_results.write_dataset_reports()`。任一子进程抛错时不会发布部分 CSV。

## 2. 一个 order-seed 搜索单元

`lr_search.run_method_search()` 按 `order_seed_registry.ORDERS_BY_DATASET[dataset]` 和 `SEEDS=(62,63,64,65,66)` 遍历，并调用 `run_search_unit()`。后者对应一个固定的 dataset、method、order 和 seed，由 `search_schema.new_order_seed_search()` 构造状态 JSON，读取 `final_hyperparameters.get_final_hyperparameters()` 得到正式参数，再对 `search_config.LR_CANDIDATES` 中三个 LR 逐个运行。完成状态由 `search_schema.validate_order_seed_search()` 检查。

每个候选调用：

```text
runner.run_experiment(data_role="formal", loss_selection=...)
  -> data.build_dataset_bundle(..., data_role="formal")
  -> strategies.build_strategy(method, ...)
  -> 对 train_stream 中每个 experience：
       runner._train_experience()
         -> strategy.train(experience)
         -> PhaseFlopProfiler 记录本 experience 学习 FLOPs
       runner._evaluate_experience()
         -> 评估当前已见 test experiences
  -> metrics.compute_cil_metrics()
  -> storage.compute_persistent_storage()
  -> checkpointing.save_search_checkpoint()
  -> output.AtomicRunArtifacts.commit()
```

`run_experiment()` 管理一次完整 CIL 运行，包括数据、策略、全部 experience、accuracy matrix、指标和 artifact。`_train_experience()` 的范围更小，只负责调用一次 `strategy.train(experience)` 并测量该 experience 的 FLOPs、墙钟时间、GPU event 时间和峰值显存。两者不能互换：搜索必须调用 `run_experiment()` 才会形成完整矩阵和 summary，`run_experiment()` 内部再复用 `_train_experience()`，不存在搜索专用的第二套训练循环。

每个候选输出 log、summary 和 accuracy matrix。独立 config 文件在提交后删除，因为相同完整配置已嵌入 summary 的 `config` 字段。完整 StrategyBundle 暂存在 order 目录的 `checkpoint_candidates`，用于候选级恢复；三个候选全部完成并选出最佳者后，最佳文件原子复制到 `checkpoint/`，三个临时文件删除，空的临时目录也删除。

## 3. Loss 选择与 search JSON

`runner._LastEpochLossTracker` 在最后一个 CIL task 的训练 strategy 上接收 Avalanche callback，以 mini-batch 样本数对 `strategy.loss` 加权。普通方法的 `last_epoch_train_mean` 直接使用最后 epoch 累计值。FeCAM 后续 task 没有 SGD epoch，`runner._selection_experiences()` 会要求在最终 FeCAM 推理路径上对最后 task 的完整训练样本重算交叉熵。`full_train_final_model` 则对全部 train experiences 重算。重算调用 `_mean_cross_entropy()`，只影响选择 loss；对应 forward FLOPs独立写入 `selection.loss_eval_flops`。

`run_search_unit()` 从三个 summary 读取 `selection.loss`，按最小值选择，平局保持 LR candidates 原顺序。它把三个候选的 `overall_learning_flops` 相加为 `total_search_flops`，把最佳 summary 中的 `model_parameter_bytes` 写为 `storage_bytes`。最终的 `orderN_seedSSS_search.json` 同时保存三个候选记录和最佳 artifact 路径。最佳候选的 summary/matrix 不移动、不复制，仍位于对应 LR 目录，它们就是最终 CIL 结果。

每个候选都启用最终推理时延测量。最后一个 task 学习结束后，本来就要遍历全部已见 test experiences 以填写 accuracy matrix 最后一行；该轮测试会在每个 `model(x)` 前后同步设备并累计前向耗时，不再额外遍历一次 test stream，也不单独执行预热。最后用模型前向总耗时除以该轮测试总样本数，得到 `final_latency_ms_per_sample`。它仍是按 `eval_mb_size` 批量推理折算到单样本的时延；只有 `single_sample_forward_flops` 使用 batch size 1 的单样本输入。

## 4. 一个 joint-run

`joint_learning.run_method_joint()` 同样遍历 3 orders × 5 seeds，并调用 `run_joint_unit()`。该函数只打开对应 `orderN_seedSSS_search.json` 获取 `best_lr` 和核对身份，不打开 `best_checkpoint`。在第 k 个阶段，它按当前 order 取 task 1..k 的累计训练数据，并从头建立一个只覆盖已见类别的 `JointClassifier`；每个阶段的模型、优化器和 shuffle DataLoader 都独立创建。这样第 k 阶段不会看到未来 task 的训练样本或未来类别输出。

Joint 从配对方法的 `final_hyperparameters.py` 读取 optimizer、batch size、momentum、weight decay、foreach、num_workers 和 `epochs_per_experience`。这里只借用通用训练超参数；不会调用 `strategies.build_strategy()`，所以不存在 EWC penalty、ER-ACE replay、iCaRL exemplar、FeCAM mean/covariance、TagFex 双分支或 CWR 临时权重。TagFex 对应的 joint 也是普通单 ResNet18 与线性头。

每个阶段训练结束后，joint 只评估截至该阶段已经出现的 test experiences，形成 `accuracy_matrix_lower_triangular`。`accuracy_by_task` 是该矩阵对角线的冗余副本；此外还记录各阶段最终 epoch loss、累计训练样本数、全部阶段训练 FLOPs 与 runtime，以及最后阶段模型的 storage、latency 和工作内存。全部字段写入一个 `joint_<dataset>_<method>__order-N__seed-SSS.json`。

## 5. Intransigence 回填

Joint JSON 成功落盘后，`intransigence.fill_from_joint_run()` 同时读取 search JSON、最佳 CIL summary、最佳 CIL accuracy matrix 和 joint JSON。它核对 exp name、dataset、method、order、seed、backbone 与 best LR，然后取下三角矩阵对角线：

```python
intransigence[k] = joint_matrix[k][k] - cil_matrix[k][k]
```

列表及其算术平均值写回最佳 CIL summary 原文件。这里使用的是矩阵对角线，不使用 `task_end_seen_accuracy_curve`；后者是第 k 行所有已见任务按测试样本数加权后的准确率，语义不同。

## 6. 恢复行为

候选级恢复依赖同一候选的 summary、matrix 和临时完整 checkpoint 三者齐全。`run_search_unit()` 每完成一个候选就原子更新 search JSON；进程在下一候选中断后，重跑会跳过已完成候选。三个候选结束后，search JSON 标为 `completed`；重跑会核对最佳 checkpoint、summary、matrix 及有效推理时延。Joint 开始前先写 pending JSON，完成后原子替换为 completed JSON；新版 joint schema 为 `pp2-joint-learning-v2`，旧的一次性全数据 joint artifact 会因 schema 不一致而被拒绝。身份不一致意味着 EXP_NAME 被错误复用，应换新实验名，不能强行拼接。

## 7. 两个最终 CSV

`aggregate_results.build_dataset_search_summary()` 遍历六方法、三个 order 和五个 seed，读取每个 search JSON 及最佳 summary，校验 intransigence 已完成，然后把嵌套字段展开成一行。列表保留为 JSON 字符串，因此 `intransigence` 不会随 task 数膨胀成大量列；`intransigence_mean` 保持数值。

`build_dataset_joint_learning()` 读取同样 90 个 joint JSON，把除 `accuracy_by_task` 外的全部原始字段展开，并把 joint 对角线 accuracy 展成 `task_1_acc` 等数值列。两个 builder 都要求行数完整且不做均值、置信区间或标准差计算；写 CSV 前还会核对 90 个路径对应的 dataset、method、order、seed、task groups，检查每个搜索单元的三个候选 summary/matrix、最佳 checkpoint 与推理时延是否齐全，最终由 `write_dataset_reports()` 原子写入：

```text
search_result_<EXP_NAME>/aggregate_results/<dataset>_search_summary.csv
joint_result_<EXP_NAME>/aggregate_results/<dataset>_joint_learning.csv
```

旧 method-level 入口、`result_<EXP_NAME>`、`optimum_lrs.json`、formal aggregate、method train/test report、权重直接测试 joint 和 `tiny_ewc_weight_only_v4` 不属于当前活动调用链。
