#!/usr/bin/env bash
# Update existing result branches, or create isolated branches for new experiments.
# Invoke with bash. Does not submit jobs, train, regenerate reports or force-push.
set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 1; }
usage() {
    echo "Usage: bash $0 --check-only|--publish [DATASET=EXP_NAME ...]"
    echo "Default: spike=spike-all texture=texture-all uwave=uwave-all"
    echo "Environment: RESULT_REPO_DIR, RESULT_REPO_URL, PYTHON_BIN"
    exit 2
}
[[ $# -ge 1 ]] || usage
MODE="$1"; shift
case "$MODE" in --check-only|--publish) ;; *) usage ;; esac
if [[ $# -eq 0 ]]; then
    set -- spike=spike-all texture=texture-all uwave=uwave-all
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
RESULT_REPO_DIR="${RESULT_REPO_DIR:-$PROJECT_ROOT/../result-pp2}"
RESULT_REPO_URL="${RESULT_REPO_URL:-git@github.com:iopav/result-pp2.git}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export GIT_TERMINAL_PROMPT="${GIT_TERMINAL_PROMPT:-0}"
command -v git >/dev/null || fail 'git is required'
command -v "$PYTHON_BIN" >/dev/null || fail "Python is unavailable: $PYTHON_BIN"
command -v flock >/dev/null || fail 'flock is required'
[[ -d "$RESULT_REPO_DIR" ]] || fail "Result repository does not exist: $RESULT_REPO_DIR (no automatic clone)"
RESULT_REPO_DIR="$(cd -- "$RESULT_REPO_DIR" && pwd -P)"
REPO_TOP="$(git -C "$RESULT_REPO_DIR" rev-parse --show-toplevel)"
REPO_TOP="$(cd -- "$REPO_TOP" && pwd -P)"
[[ "$REPO_TOP" == "$RESULT_REPO_DIR" ]] || fail 'RESULT_REPO_DIR must be the repository root'
[[ "$RESULT_REPO_DIR" != "$PROJECT_ROOT" ]] || fail 'Result repository must differ from training repository'
COMMON_DIR="$(git -C "$RESULT_REPO_DIR" rev-parse --git-common-dir)"
[[ "$COMMON_DIR" == /* ]] || COMMON_DIR="$RESULT_REPO_DIR/$COMMON_DIR"
exec 9>"$COMMON_DIR/pp2-results-update.lock"
flock -n 9 || fail 'Another result update is running'
normalize_url() {
    local value="${1%/}"
    value="${value%.git}"
    value="${value/#git@github.com:/https:\/\/github.com/}"
    printf '%s' "$value"
}
ORIGIN="$(git -C "$RESULT_REPO_DIR" remote get-url origin)"
[[ "$(normalize_url "$ORIGIN")" == "$(normalize_url "$RESULT_REPO_URL")" ]] \
    || fail "Unexpected origin: $ORIGIN; set RESULT_REPO_URL explicitly for another result repository"
PUSH_ORIGIN="$(git -C "$RESULT_REPO_DIR" remote get-url --push origin)"
[[ "$(normalize_url "$PUSH_ORIGIN")" == "$(normalize_url "$RESULT_REPO_URL")" ]] \
    || fail "Unexpected push URL: $PUSH_ORIGIN"
clean_repo() {
    local state
    state="$(git -C "$RESULT_REPO_DIR" status --porcelain --untracked-files=all --ignored=matching)"
    [[ -z "$state" ]] || { printf '%s\n' "$state" >&2; fail 'Result repository has local changes; preserved without reset/clean/stash'; }
}
clean_repo
git -C "$RESULT_REPO_DIR" var GIT_AUTHOR_IDENT >/dev/null
git -C "$RESULT_REPO_DIR" var GIT_COMMITTER_IDENT >/dev/null
git -C "$RESULT_REPO_DIR" ls-remote origin >/dev/null

# Python copy helper checks symlinks/size before copying. Existing unrelated files
# are preserved; only explicitly copied paths are staged, even with '*' ignores.
copy_results() {
    "$PYTHON_BIN" -s - "$PROJECT_ROOT" "$RESULT_REPO_DIR" "$1" "$2" "$3" <<'PY'
import os
from pathlib import Path
import shutil
import subprocess
import sys

project, repo = map(Path, sys.argv[1:3])
dataset, exp, mode = sys.argv[3:]
destination = repo / exp
files = []
for prefix in ('search_result', 'joint_result'):
    source = project / f'{prefix}_{exp}'
    if source.is_symlink() or not source.is_dir():
        raise SystemExit(f'Missing directory or symlink: {source}')
    for directory, dirs, names in os.walk(source, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            if path.is_symlink():
                raise SystemExit(f'Symlink in source: {path}')
        for name in names:
            path = Path(directory) / name
            if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.tmp', '.backup', '.lock'}:
                continue
            if path.stat().st_size > 99 * 1024 * 1024:
                raise SystemExit(f'File exceeds 99 MiB: {path}')
            target = destination / source.name / path.relative_to(source)
            for parent in (target, *target.parents):
                if parent == repo:
                    break
                if parent.is_symlink():
                    raise SystemExit(f'Symlink in destination: {parent}')
            files.append((path, target))
for required in (
    project / f'search_result_{exp}/aggregate_results/{dataset}_search_summary.csv',
    project / f'joint_result_{exp}/aggregate_results/{dataset}_joint_learning.csv',
    project / f'search_result_{exp}/aggregate_results/tagfex_task_flops.csv',
):
    if not required.is_file() or required.stat().st_size == 0:
        raise SystemExit(f'Missing/empty report: {required}')
if not files:
    raise SystemExit(f'No publishable files: {exp}')
print(f'RESULT_COPY_{mode.upper()} exp={exp} files={len(files)} destination={destination}', flush=True)
if mode == 'copy':
    for source, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Literal NUL-delimited pathspecs protect filenames and avoid command limits.
    paths = b''.join(str(target.relative_to(repo)).replace('\\', '/').encode('utf-8') + b'\0'
                     for _, target in files)
    subprocess.run(['git', '--literal-pathspecs', '-C', str(repo), 'add', '-f',
                    '--pathspec-from-file=-', '--pathspec-file-nul'], input=paths, check=True)
PY
}

declare -A SEEN=()
for item in "$@"; do
    [[ "$item" == *=* ]] || usage
    dataset="${item%%=*}"; exp="${item#*=}"
    case "$dataset" in spike|texture|uwave) ;; *) fail "Unknown dataset: $dataset" ;; esac
    [[ "$exp" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "Invalid experiment: $exp"
    git check-ref-format --branch "$exp" >/dev/null
    [[ -z "${SEEN[$exp]:-}" ]] || fail "Duplicate experiment branch: $exp"
    SEEN[$exp]=1
    copy_results "$dataset" "$exp" check
    # Exit 3 denotes complete evidence with known failed LR candidates (e.g. EWC).
    status=0
    "$PYTHON_BIN" -s "$PROJECT_ROOT/check_experiments.py" \
        --project-root "$PROJECT_ROOT" --experiments "$item" || status=$?
    [[ "$status" == 0 || "$status" == 3 ]] || fail "Artifact audit failed for $exp (exit $status)"
    [[ "$status" != 3 ]] || echo "RESULT_HAS_FAILED_CANDIDATES exp=$exp (failure evidence will be preserved)"
done
if [[ "$MODE" == --check-only ]]; then
    echo 'RESULT_UPDATE_CHECK_OK (no branch switch, copy, commit or push)'
    exit 0
fi

for item in "$@"; do
    dataset="${item%%=*}"; exp="${item#*=}"
    clean_repo
    # Fetch this branch explicitly, without relying on remote fetch refspecs.
    remote_head="$(git -C "$RESULT_REPO_DIR" ls-remote --heads origin "refs/heads/$exp")"
    if [[ -n "$remote_head" ]]; then
        git -C "$RESULT_REPO_DIR" fetch origin "refs/heads/$exp:refs/remotes/origin/$exp"
        if git -C "$RESULT_REPO_DIR" show-ref --verify --quiet "refs/heads/$exp"; then
            git -C "$RESULT_REPO_DIR" switch "$exp"
            git -C "$RESULT_REPO_DIR" merge --ff-only "refs/remotes/origin/$exp"
        else
            git -C "$RESULT_REPO_DIR" switch -c "$exp" --track "origin/$exp"
        fi
    elif git -C "$RESULT_REPO_DIR" show-ref --verify --quiet "refs/heads/$exp"; then
        git -C "$RESULT_REPO_DIR" switch "$exp"
    else
        git -C "$RESULT_REPO_DIR" switch --orphan "$exp"
    fi
    copy_results "$dataset" "$exp" copy
    git -C "$RESULT_REPO_DIR" diff --cached --stat
    if git -C "$RESULT_REPO_DIR" diff --cached --quiet; then
        echo "RESULT_UNCHANGED exp=$exp"
    else
        git -C "$RESULT_REPO_DIR" commit -m "results: update $exp including TagFex task FLOPs"
    fi
    git -C "$RESULT_REPO_DIR" push --set-upstream origin "HEAD:refs/heads/$exp"
    local_commit="$(git -C "$RESULT_REPO_DIR" rev-parse HEAD)"
    remote_commit="$(git -C "$RESULT_REPO_DIR" ls-remote --heads origin "refs/heads/$exp" | awk 'NR == 1 {print $1}')"
    [[ "$local_commit" == "$remote_commit" ]] || fail "Remote commit verification failed: $exp"
    echo "RESULT_UPDATE_COMPLETE exp=$exp branch=$exp commit=$local_commit"
done
echo 'ALL_RESULT_UPDATES_COMPLETE'
