# Manual CIL hyperparameter runs

This directory contains one editable entry point for every dataset-method pair.  Each script uses the same dataset adapter, class order, deterministic seed, shared temporal backbone, Avalanche strategy, minibatch protocol, no-augmentation protocol and post-experience test evaluation as the formal runner.

Edit only the `PARAMETERS`, `ORDER_ID`, `SEED`, `EPOCHS` and `DEVICE` constants near the top of a script, then run it from the `py310` environment. Every script now explicitly uses `DEVICE = "cuda"`; CUDA unavailability is an error and never silently falls back to CPU. At startup the runner prints the resolved CUDA device name and verifies that every model parameter is on the requested device. Example:

```powershell
conda run -n py310 python manual_tuning\spike__er_ace.py
```

These entries intentionally do not call `PhaseFlopProfiler`, `compute_persistent_storage`, latency measurement or metrics-summary generation. A successful run prints the complete lower-triangular accuracy matrix once, after all training and evaluation have finished. Manual tuning does not save JSON or any other result artifact; a failed run prints the normal traceback and never prints a partial matrix as a completed result.

CPU work is still expected for DataLoader workers, NumPy/JSON processing, Spike bit packing and persistent replay payloads. Model forward, loss, backward, optimizer updates, FeCAM statistics and iCaRL feature extraction use the selected CUDA device.

Manual runs never change formal hyperparameters automatically. After selecting a validated candidate, copy its complete values into the matching `FINAL_HYPERPARAMETERS[dataset][method]` entry in `cil_experiments/final_hyperparameters.py`. After all 15 entries have been reviewed, set `FINAL_HYPERPARAMETERS_LOCKED = True`; formal entrypoints fail while it remains False. The three formal dataset entrypoints load all optimizer, batch, epoch and method-specific values from that file; `registry.py` remains the source of dataset, backbone and protocol definitions only.
