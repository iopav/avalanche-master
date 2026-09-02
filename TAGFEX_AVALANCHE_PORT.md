# TagFex Avalanche adapter

`run_tagfex_uwave.py` is a one-epoch end-to-end smoke run for the local UWave
image arrays. It uses an Avalanche `nc_benchmark` and `SupervisedTemplate`, but
loads the original `TagFexNet`, `infoNCE_loss` and `infoNCE_distill_loss` from
the adjacent TagFex checkout. This avoids maintaining a silently diverging
copy of the model core.

```powershell
D:\anaconda3\envs\py310\python.exe run_tagfex_uwave.py --device cuda:0 --epochs 1
```

The adapter preserves the 3+1 class order, two-view batches, per-task backbone
expansion, old-backbone freezing, optimizer/scheduler reset, all TagFex loss
terms, a 2000-example herding memory, post-task weight alignment, classifier
evaluation and NME evaluation.

Exact numerical identity is not a valid expectation across the two launchers.
Avalanche wraps samples in experience datasets and constructs a different
dataloader object, which changes RNG consumption and shuffled minibatch order.
The model and objective are shared, while the resulting floating-point update
sequence is not. The adapter also stores selected replay tensors instead of
the original dataset indices; selected examples and the memory cap are the
same, but checkpoint representation differs.

Both paths guard the original transfer-classification expression against a
replay-only minibatch. Without that guard, cross entropy receives an empty
current-task slice and returns `NaN`.

## Unified raw-time-series strategy

`tagfex` is also registered in `cil_experiments.registry.METHODS` and built by
`cil_experiments.strategies.build_strategy`. This formal path accepts the same
raw `[B,C,T]` tensors as ER-ACE, EWC, CWRStar, iCaRL and FeCAM. The ResNet2D
backbone is replaced by the repository's corresponding 1D temporal feature
stack, while the TagFex task-agnostic/task-specific branches, 64-neuron
per-experience expansion, interpolation initialization, contrastive losses,
transfer attention, herding memory and weight alignment are retained.

The protocol disables augmentation, so the two contrastive views are
value-identical. This is recorded in each formal config and limits the useful
augmentation signal; changing it would change the shared experiment protocol.

The native helper based on a single `fvcore` forward is an inference
cross-check only. Formal TagFex training uses the phase profiler and includes
both views, merged replay, teacher/projector distillation, loss arithmetic,
backward, SGD updates, exemplar feature extraction, class means and weight
alignment. Replay is merged before the student forward, so those FLOPs are
included in `student_current_forward` rather than reported as a separate
`replay_forward` phase. Strict audit aborts on every unclassified executed
computational operator.
