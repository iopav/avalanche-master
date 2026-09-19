# Read-only experiment audit

Run in the original experiment checkout (Python 3.10+, no GPU or third-party
packages needed):

```bash
python3 check_experiments.py \
  --experiments spike=spike-all texture=texture-all uwave=uwave-all
```

From another directory, use the absolute script path. The project root defaults
to the script directory; `--project-root` overrides it. All check results, paths,
errors and final counts are printed to the terminal. No output files are written.
Existing reports and experiment artifacts are never modified. Experiment names are
explicit: there is no prefix-based dataset inference.

The expected methods, LRs, seeds and orders come from the checkout's
`cil_experiments/search_config.py` and `order_seed_registry.py`. This checks the
current PP2 search + joint pipeline, not legacy formal/manual/smoke experiments.
An entirely missing seed/order/method is reported, not silently skipped.

Checks include candidate status, summary/config/matrix identity and schema
fields, finite JSON numbers, matrix dimensions/range and weighted accuracy,
task log sequence and FLOPs totals, selected LR, selected checkpoint ZIP structure
and CRC, joint task coverage and reference identity, and aggregate CSV row
coverage/duplicates. Texture/EWC nonfinite failures require their hashed failure
evidence. Failed candidates do not need successful summaries or checkpoints.
Only the selected checkpoint is required: the pipeline deletes other checkpoints.
Successful candidate configuration is read from the summary's `config` field.
The standalone `__config.json` is not required: `lr_search.py` explicitly deletes
it after training. Texture/EWC failure evidence still requires its separate
`failed_runs/<attempt>/config.json`, as listed in its manifest.

The default audit requires `<dataset>_search_summary.csv` and
`<dataset>_joint_learning.csv`. `--skip-reports` restricts the audit to raw results;
it does not certify aggregate reports. Optional exports such as
`tagfex_task_flops.csv` are not required. CSV coverage checks do not recalculate
every exported metric; rerun the normal aggregate command to refresh metrics.

Exit codes / final verdict:

| Code | Verdict | Meaning |
|---|---|---|
| 0 | ALL_SUCCESSFUL | All expected audited artifacts passed, no failed candidates |
| 1 | INCOMPLETE_OR_INVALID | Missing, pending, inconsistent or corrupt artifacts |
| 2 | CLI error | Invalid arguments |
| 3 | COMPLETE_WITH_FAILED_CANDIDATES | Files passed, but some LR candidates failed |

Look for `[ERROR]` and `[FAILED]` in terminal output. Each check prints the
dataset, method, order, seed, LR, path and failure reason when applicable.
There can be several checks per run; counts are check counts, not experiment
counts. An all-failed search is
incomplete, because it has no selected model or usable downstream joint result.

This is an artifact consistency audit, not proof of benchmark quality or model
reload correctness. Checkpoints are never unpickled. CRC checking reads their
contents and may take time. Run after writers stop for a stable snapshot; pending
and in-progress artifacts cannot be distinguished from interrupted runs here.
Stored absolute result paths must resolve on the original machine: relocated
copies will be reported as path mismatches rather than silently accepted.

Tests:

```bash
python3 -m unittest discover -s tests -p test_check_experiments.py -v
```
