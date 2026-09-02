# Manual CIL hyperparameter runs

This directory contains one editable entry point for every formal dataset-method pair, plus manual-only SI, LwF, Spike EWC+Cosine and Spike MAS candidate scripts. Each script uses the same image adapter, class order, deterministic seed, selectable registered ResNet18, Avalanche strategy, minibatch protocol, no-augmentation protocol and post-experience test evaluation as the formal runner. Candidate methods cannot be selected by the formal entrypoints.

Manual runs use the deterministic class-stratified `dataset_mini/` subset by default. It contains approximately 5% of each train/test class and preserves the exact raw/image representations used by the full datasets. Rebuild it with `python prepare_mini_datasets.py`; pass `use_mini=False` to `run_manual` only when a full-data confirmation is required. The selected backbone can be `resnet18_cifar_small`, `resnet18_cifar`, or `resnet18_cifar_large` through the `backbone_id` argument.

Edit only the `PARAMETERS`, `ORDER_ID`, `SEED`, `EPOCHS` and `DEVICE` constants near the top of a script, then run it from the `py310` environment. Every script now explicitly uses `DEVICE = "cuda"`; CUDA unavailability is an error and never silently falls back to CPU. At startup the runner prints the resolved CUDA device name and verifies that every model parameter is on the requested device. Example:

```powershell
conda run -n py310 python manual_tuning\spike__er_ace.py
```

`search_spike_fecam.py` is a separate grid-search helper with its own validation split. It never reads the formal test set for model selection, disables FLOPs accounting, and writes no result file. Edit its grid constants if the selected value lies on a search boundary.

These entries intentionally do not call `PhaseFlopProfiler`, `compute_persistent_storage`, latency measurement or metrics-summary generation. A successful run prints the complete lower-triangular accuracy matrix once, after all training and evaluation have finished. Manual tuning does not save JSON or any other result artifact; a failed run prints the normal traceback and never prints a partial matrix as a completed result.

CPU work is still expected for DataLoader workers, NumPy/JSON processing, Spike bit packing and persistent replay payloads. Model forward, loss, backward, optimizer updates, FeCAM statistics and iCaRL feature extraction use the selected CUDA device.

Manual runs never change formal hyperparameters automatically. After selecting a validated candidate, copy its complete values into the matching `FINAL_HYPERPARAMETERS[dataset][method]` entry in `cil_experiments/final_hyperparameters.py`. After all 18 entries have been reviewed, set `FINAL_HYPERPARAMETERS_LOCKED = True`; formal entrypoints fail while it remains False. The three formal dataset entrypoints load all optimizer, batch, epoch and method-specific values from that file; `registry.py` remains the source of dataset, backbone and protocol definitions only.
