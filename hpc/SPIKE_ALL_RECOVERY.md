# spike-all: interim reports and TagFex order workers

These scripts preserve the existing training implementation and hyperparameters.
Run them in the HPC checkout that owns `search_result_spike-all` and
`joint_result_spike-all`; JSON artifacts contain absolute source paths.
Do not change batch size, epochs, backbone, LR candidates or the order/seed
registry when resuming the same experiment.

## 1. Stop the original writer

Inspect `squeue -u "$USER"`, identify the original spike-all job, and stop that
specific job before publishing or launching replacements. Do not cancel all
your jobs. Wait until its processes have exited. These new locks cannot protect
against the old launcher, which does not participate in locking.

## 2. Validate and publish the interim result

Activate the same `avalanche` environment used for the original run. From the
`avalanche-master` directory:

```bash
python report_without_tagfex.py --dataset spike --exp-name spike-all &&
RESULT_REPO_URL=git@github.com:iopav/result-pp2.git \
  bash hpc/arrhenius_publish_results.sh --publish spike-all spike
```

The `&&` is essential: incomplete five-method results must not be published.
Expected success is `FIVE_METHODS_COMPLETE search_rows=225 best_lr_rows=75
joint_rows=75`, followed by `RESULT_PUBLISH_COMPLETE`. The audit includes all
six methods. A valid unit means both search and joint passed artifact checks;
it does not imply its checkpoint is still available for training resume.

Reports:

- `search_result_spike-all/aggregate_results/spike_completion_audit.csv`: all 90
  method/order/seed units, including detailed errors and TagFex progress.
- `search_result_spike-all/aggregate_results/spike_search_summary.csv`: all 225
  candidate rows for the five methods, excluding TagFex.
- `search_result_spike-all/aggregate_results/spike_best_lr_without_tagfex.csv`:
  75 selected-LR rows, one per method/order/seed. These are not averaged scores.
- `joint_result_spike-all/aggregate_results/spike_joint_learning.csv`: 75 joint rows.
- `report_scope_without_tagfex.json` next to the search report: explicit scope.

The existing publisher copies all non-model artifacts, including incomplete
TagFex raw results. It creates a new orphan branch `spike-all`, refuses an
existing local/remote branch, and requires a clean sibling `result-pp2` checkout.
If it refuses, inspect its exact error and `git -C ../result-pp2 status
--porcelain --untracked-files=all --ignored=matching`; do not delete results or
force push. Publishing is a snapshot; later TagFex results are not uploaded
automatically, and the publisher cannot update an existing branch.

## 3. Resume TagFex in a three-GPU allocation

Use the same site allocation command and environment as the original run,
requesting three visible GPUs. The launcher does not allocate resources or
activate Conda. After entering the allocation and activating `avalanche`:

```bash
OLD_WRITER_STOPPED=1 DATASET=spike EXP_NAME=spike-all \
  bash hpc/arrhenius_tagfex_orders.sh
```

GPU 0 runs order 1; GPU 1 runs order 2; GPU 2 runs order 3. Each worker runs seeds
63, 65, 67, 69, 71 serially, first search, then joint. This is independent-run
parallelism, not multi-GPU training of one model. It keeps the original output
directories and resumes via the original search/joint functions. Completed
searches still require their best checkpoints; an interrupted candidate can
be restarted from its beginning, not from its last epoch. Original recovery
logic can remove that incomplete candidate's artifacts before restarting it.
Preserve any failed-candidate logs you need before launching.

Logs go to `logs/tagfex-spike-all-<timestamp>-<job>-<pid>/order{1,2,3}.log`.
The launcher waits for every worker and exits nonzero if any worker fails.
It neither aggregates shared reports nor publishes from individual workers.
For a subsequent joint-only attempt, set `STAGE=joint`.

## 4. Full report after all workers succeed

```bash
python main_exp/spike.py --stage aggregate --exp-name spike-all
```

This validates and replaces the canonical reports with all six methods. The
separate `*_without_tagfex*` files remain as the earlier five-method snapshot.
An update to an already published branch needs a separate normal commit/push
from its verified checkout; do not rerun the orphan-branch publisher expecting
it to update `spike-all`.
