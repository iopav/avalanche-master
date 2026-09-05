# Manual CIL hyperparameter runs

This directory contains one editable entry point for every formal dataset-method pair, plus manual-only SI, LwF, Spike EWC+Cosine and Spike MAS candidate scripts. Each script uses the same image adapter, class order, deterministic seed, selectable registered ResNet18, Avalanche strategy, minibatch protocol, no-augmentation protocol and post-experience test evaluation as the formal runner. Candidate methods cannot be selected by the formal entrypoints.

Manual scripts use `resnet18_cifar` and the deterministic class-stratified `dataset_mini/` subset by default. TagFex uses SGD with momentum `0.9`; every other method currently uses Adam. The current manual starting learning rate is `0.01` and weight decay is zero in all scripts. Each script writes these choices directly in the Python file. The mini dataset contains approximately 50% of each train/test class and preserves the same raw/image representations used by the full datasets. Rebuild it with `python prepare_mini_datasets.py`.

Edit these constants near the bottom of any individual script before running it:

```python
DATA_MODE = "mini"  # "mini" or "full"
BACKBONE_ID = "resnet18_cifar"  # small, standard, or large
```

`DATA_MODE` accepts `mini` or `full`; `BACKBONE_ID` accepts `resnet18_cifar` or `temporal`.

Edit only the `PARAMETERS`, `ORDER_ID`, `SEED`, `EPOCHS` and `DEVICE` constants near the top of a script, then run it from the `py310` environment. Every script now explicitly uses `DEVICE = "cuda"`; CUDA unavailability is an error and never silently falls back to CPU. At startup the runner prints the resolved CUDA device name and verifies that every model parameter is on the requested device. Example:

```powershell
conda run -n py310 python manual_tuning\spike__er_ace.py
```

`search_spike_fecam.py` is a separate grid-search helper with its own validation split. It never reads the formal test set for model selection, disables FLOPs accounting, and writes no result file. Edit its grid constants if the selected value lies on a search boundary.

These entries intentionally do not call `PhaseFlopProfiler`, `compute_persistent_storage`, latency measurement or metrics-summary generation. A successful run prints the complete lower-triangular accuracy matrix once, after all training and evaluation have finished. Manual tuning does not save JSON or any other result artifact; a failed run prints the normal traceback and never prints a partial matrix as a completed result.

CPU work is still expected for DataLoader workers, NumPy/JSON processing, Spike bit packing and persistent replay payloads. Model forward, loss, backward, optimizer updates, FeCAM statistics and iCaRL feature extraction use the selected CUDA device.

Manual runs never change formal hyperparameters automatically. After selecting a validated candidate, copy its complete values into the matching `FINAL_HYPERPARAMETERS[dataset][method]` entry in `cil_experiments/final_hyperparameters.py`. After all 18 entries have been reviewed, set `FINAL_HYPERPARAMETERS_LOCKED = True`; formal entrypoints fail while it remains False. The three formal dataset entrypoints load all optimizer, batch, epoch and method-specific values from that file; `registry.py` remains the source of dataset, backbone and protocol definitions only.
