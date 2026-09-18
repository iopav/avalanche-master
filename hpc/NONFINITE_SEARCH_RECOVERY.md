# Non-finite training-loss recovery

Keep the original experiment name, LR candidates and hyperparameters. Stop older
writers before resuming. Do not manually edit or delete the search JSON.

The search continues past the runner's `FloatingPointError: Non-finite training
loss` and selects only successful candidates. All failed means an explicit
failed search, with no best model. Unrelated exceptions still abort.

For Texture/EWC only, a failed candidate is skipped only when its failure
manifest, failure JSON, config JSON and original run log exist and pass identity
and content checks. Missing evidence triggers a rerun. Old failure records and
partial files are archived before retrying; successful candidates are preserved.
Failure evidence is saved under the candidate's `failed_runs/<attempt>/` folder.
Its location is recorded as `failure_manifest` in the search JSON. The failure
JSON includes task/epoch, traceback and partial accuracy/runtime data, not a
completed-run summary. No model checkpoint is required for a failed candidate.

If a search already selected a best LR, missing failure evidence is repaired in
an isolated `failure_recovery/<attempt>/` run. If that replay unexpectedly
succeeds, the code preserves its result and stops for inspection rather than
silently replacing the previous best selection.

CSV columns remain unchanged. Failed candidates keep their row with missing
metrics blank. Total search FLOPs are blank where failed-attempt cost is unknown;
the known lower bound remains in search JSON only.

Synchronize `lr_search.py`, `search_schema.py`, `aggregate_results.py`,
`runner.py` and the new `texture_ewc_failure.py` under `cil_experiments` together.
Then resume the original experiment command. EWC training code and hyperparameter
configuration are unchanged.
