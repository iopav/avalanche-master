# Update result branches

`arrhenius_publish_results.sh` is the original new-branch-only publisher.
Use `arrhenius_update_results.sh` for existing or new branches:

```bash
bash hpc/arrhenius_update_results.sh --check-only
bash hpc/arrhenius_update_results.sh --publish
```

Defaults, processed sequentially:

| Dataset | Experiment and branch |
|---|---|
| spike | spike-all |
| texture | texture-all |
| uwave | uwave-all |

Explicit selection:

```bash
bash hpc/arrhenius_update_results.sh --publish \
  spike=spike-all texture=texture-all uwave=uwave-all
```

Run after aggregation and TagFex task FLOPs export, with result writers stopped.
Python 3.10+ (standard library), Git and flock are required. `PYTHON_BIN` selects
the interpreter. No GPU computation occurs, but the interpreter must run on the
current node. Original absolute artifact paths must still resolve for auditing.

The default target is the existing Git repository `../result-pp2`, not the
training project's ignored `result/` directory. Set `RESULT_REPO_DIR` to another
existing result repository and `RESULT_REPO_URL` to its expected origin URL if
needed. SSH and HTTPS forms of the default GitHub URL are treated as equivalent.
No repository is cloned automatically.

Destination layout on each experiment-named branch:

```text
<result repository>/<experiment>/search_result_<experiment>/...
<result repository>/<experiment>/joint_result_<experiment>/...
```

Both result trees are copied, including aggregate reports, task FLOPs CSVs,
JSON, logs and failure evidence. Model files (`.pt`, `.pth`, `.ckpt`) and temporary
`.tmp`, `.backup`, `.lock` files are excluded. Source/destination symlinks and
files above 99 MiB are rejected. Copying replaces same-path files; destination
files absent from the source are preserved, not pruned. This is an update, not
an exact deletion-synchronized mirror.

The script audits all requested experiments before publishing any. Audit code 3
(complete files with failed LR candidates) is explicitly accepted and reported;
missing/inconsistent files block publishing. Both aggregate reports and the
TagFex task FLOPs CSV are required. The latter must be exported beforehand.

Existing local or remote branches are reused; remote changes are incorporated
only by fast-forward. Divergence stops the script, with no reset or force-push.
Local commits already on the selected branch are included in its push. New
branches are orphan branches, consistent with the original publisher. The
working tree must be clean, including untracked/ignored files, before switching.
Only copied paths are staged; unrelated branch content and .gitignore survive.
There is no automatic stash, clean, reset, or deletion of source results.

Each push is followed by a remote commit hash check. Unchanged results skip the
commit but still push/verify the branch. Look for `RESULT_UPDATE_COMPLETE` for
each experiment and `ALL_RESULT_UPDATES_COMPLETE` at the end. A later failure
does not roll back earlier completed pushes. Copy/commit failures can leave
local changes for inspection; resolve those before retrying.

`--check-only` audits files, checks the current target checkout and remote access,
and lists copy counts. It does not switch branches, fetch, copy results, commit
or push; it creates/opens only a lock file in Git's metadata directory. Branch
fast-forward compatibility is checked during publishing.

Tests use temporary repositories and a local bare remote, never GitHub:

```bash
python -m unittest discover -s tests -p test_update_results_script.py -v
```

On Windows the integration test stubs flock because Git Bash does not ship it;
the actual Linux lock was not tested there. It also substitutes a fixture audit
result; artifact validation has separate `test_check_experiments.py` tests.
