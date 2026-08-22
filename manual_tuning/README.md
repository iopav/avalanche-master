# Manual CIL hyperparameter runs

This directory contains one editable entry point for every dataset-method pair.  Each script uses the same dataset adapter, class order, deterministic seed, shared temporal backbone, Avalanche strategy, minibatch protocol, no-augmentation protocol and post-experience test evaluation as the formal runner.

Edit only the `PARAMETERS`, `ORDER_ID`, `SEED`, `EPOCHS` and `DEVICE` constants near the top of a script, then run it from the `py310` environment. Every script now explicitly uses `DEVICE = "cuda"`; CUDA unavailability is an error and never silently falls back to CPU. At startup the runner prints the resolved CUDA device name and verifies that every model parameter is on the requested device. Example:

```powershell
conda run -n py310 python manual_tuning\spike__er_ace.py
```

These entries intentionally do not call `PhaseFlopProfiler`, `compute_persistent_storage`, latency measurement or metrics-summary generation.  A successful run prints each observed row and atomically saves one timestamped lower-triangular accuracy-matrix JSON under `manual_tuning/results/<dataset>/<method>/`.  A failed run does not save a matrix.

CPU work is still expected for DataLoader workers, NumPy/JSON processing, Spike bit packing and persistent replay payloads. Model forward, loss, backward, optimizer updates, FeCAM statistics and iCaRL feature extraction use the selected CUDA device.
