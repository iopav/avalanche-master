#!/bin/bash -l

# Publish one completed formal experiment to its own orphan branch in
# https://github.com/iopav/result-pp2.git. All result artifacts are copied
# except model/checkpoint files (*.pt, *.pth, *.ckpt). This script never
# deletes source results or overwrites an existing branch.
set -euo pipefail

usage() {
    echo "Usage: $0 --check-only|--publish EXP_NAME DATASET" >&2
    exit 2
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ $# -eq 3 ]] || usage
MODE="$1"
EXP_NAME="$2"
DATASET="$3"

case "${MODE}" in
    --check-only|--publish) ;;
    *) usage ;;
esac
case "${DATASET}" in
    spike|texture|uwave) ;;
    *) fail "DATASET must be spike, texture, or uwave." ;;
esac
if ! [[ "${EXP_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    fail "EXP_NAME may contain only letters, digits, dot, underscore, and hyphen, and must start with a letter or digit."
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
AVALANCHE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd -P)"
RESULT_REPO_DIR="${AVALANCHE_ROOT}/result-pp2"
RESULT_REPO_URL="${RESULT_REPO_URL:-https://github.com/iopav/result-pp2.git}"
RESULT_GIT_USER_NAME="${RESULT_GIT_USER_NAME:-}"
RESULT_GIT_USER_EMAIL="${RESULT_GIT_USER_EMAIL:-}"

command -v git >/dev/null 2>&1 || fail "git is not available."
git check-ref-format --branch "${EXP_NAME}" >/dev/null 2>&1 \
    || fail "EXP_NAME is not a valid Git branch name: ${EXP_NAME}"

# A scheduled job must not wait indefinitely for a username/password prompt.
# Configure a credential helper or SSH/HTTPS credentials before the run. Set
# GIT_TERMINAL_PROMPT=1 only when deliberately running this script interactively.
export GIT_TERMINAL_PROMPT="${GIT_TERMINAL_PROMPT:-0}"

if ! git ls-remote "${RESULT_REPO_URL}" >/dev/null 2>&1; then
    fail "cannot authenticate to or read ${RESULT_REPO_URL}; configure GitHub credentials before starting the formal experiment."
fi
if git ls-remote --exit-code --heads "${RESULT_REPO_URL}" \
    "refs/heads/${EXP_NAME}" >/dev/null 2>&1; then
    fail "remote orphan branch already exists: ${EXP_NAME}; use a new EXP_NAME."
fi

if [[ -e "${RESULT_REPO_DIR}" && ! -d "${RESULT_REPO_DIR}" ]]; then
    fail "${RESULT_REPO_DIR} exists but is not a directory."
fi
if [[ ! -e "${RESULT_REPO_DIR}" ]]; then
    git clone "${RESULT_REPO_URL}" "${RESULT_REPO_DIR}"
fi
if ! git -C "${RESULT_REPO_DIR}" rev-parse --is-inside-work-tree \
    >/dev/null 2>&1; then
    fail "${RESULT_REPO_DIR} exists but is not a Git working tree; it was left unchanged."
fi

REPO_TOP="$(git -C "${RESULT_REPO_DIR}" rev-parse --show-toplevel)"
REPO_TOP="$(cd -- "${REPO_TOP}" && pwd -P)"
[[ "${REPO_TOP}" == "${RESULT_REPO_DIR}" ]] \
    || fail "Git top level ${REPO_TOP} does not match ${RESULT_REPO_DIR}."

ORIGIN_URL="$(git -C "${RESULT_REPO_DIR}" remote get-url origin 2>/dev/null || true)"
normalize_url() {
    local value="${1%/}"
    value="${value%.git}"
    printf '%s' "${value}"
}
if [[ "$(normalize_url "${ORIGIN_URL}")" != "$(normalize_url "${RESULT_REPO_URL}")" ]]; then
    fail "origin does not point to the configured result-pp2 repository; it was left unchanged."
fi

REPO_STATE="$(git -C "${RESULT_REPO_DIR}" status --porcelain \
    --untracked-files=all --ignored=matching)"
if [[ -n "${REPO_STATE}" ]]; then
    printf '%s\n' "${REPO_STATE}" >&2
    fail "result-pp2 contains modified, untracked, or ignored files; clean it manually before publishing."
fi

git -C "${RESULT_REPO_DIR}" fetch --prune origin
if git -C "${RESULT_REPO_DIR}" show-ref --verify --quiet \
    "refs/heads/${EXP_NAME}"; then
    fail "local branch already exists: ${EXP_NAME}; it was not overwritten."
fi

if [[ -n "${RESULT_GIT_USER_NAME}" ]]; then
    git -C "${RESULT_REPO_DIR}" config user.name "${RESULT_GIT_USER_NAME}"
fi
if [[ -n "${RESULT_GIT_USER_EMAIL}" ]]; then
    git -C "${RESULT_REPO_DIR}" config user.email "${RESULT_GIT_USER_EMAIL}"
fi
[[ -n "$(git -C "${RESULT_REPO_DIR}" config --get user.name || true)" ]] \
    || fail "Git user.name is unset; configure it or set RESULT_GIT_USER_NAME."
[[ -n "$(git -C "${RESULT_REPO_DIR}" config --get user.email || true)" ]] \
    || fail "Git user.email is unset; configure it or set RESULT_GIT_USER_EMAIL."

if [[ "${MODE}" == "--check-only" ]]; then
    echo "RESULT_PUBLISH_CHECK_OK exp_name=${EXP_NAME} repo=${RESULT_REPO_DIR}"
    exit 0
fi

SEARCH_SOURCE="${PROJECT_ROOT}/search_result_${EXP_NAME}"
JOINT_SOURCE="${PROJECT_ROOT}/joint_result_${EXP_NAME}"
SEARCH_REPORT="${SEARCH_SOURCE}/aggregate_results/${DATASET}_search_summary.csv"
JOINT_REPORT="${JOINT_SOURCE}/aggregate_results/${DATASET}_joint_learning.csv"

[[ -d "${SEARCH_SOURCE}" ]] \
    || fail "missing formal search directory: ${SEARCH_SOURCE}"
[[ -d "${JOINT_SOURCE}" ]] \
    || fail "missing formal joint directory: ${JOINT_SOURCE}"
[[ -s "${SEARCH_REPORT}" ]] \
    || fail "missing or empty final search report: ${SEARCH_REPORT}"
[[ -s "${JOINT_REPORT}" ]] \
    || fail "missing or empty final joint report: ${JOINT_REPORT}"
if [[ -n "$(find "${SEARCH_SOURCE}" "${JOINT_SOURCE}" -type l -print -quit)" ]]; then
    fail "result directories contain a symbolic link; refusing to copy a path that may escape the result roots."
fi

# git switch --orphan creates an unborn branch and removes all tracked files.
# The clean-state check above guarantees that no untracked/ignored files are
# silently carried into the new branch.
git -C "${RESULT_REPO_DIR}" switch --orphan "${EXP_NAME}"
LEFTOVER="$(find "${RESULT_REPO_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name .git -print -quit)"
[[ -z "${LEFTOVER}" ]] \
    || fail "unexpected file remained after orphan switch: ${LEFTOVER}"

DESTINATION="${RESULT_REPO_DIR}/${EXP_NAME}"
mkdir -p "${DESTINATION}"

copy_non_model_files() {
    local source_root="$1"
    local destination_root="$2"
    local source_file relative_path destination_file
    local copied=0

    mkdir -p "${destination_root}"
    while IFS= read -r -d '' source_file; do
        relative_path="${source_file#"${source_root}/"}"
        destination_file="${destination_root}/${relative_path}"
        mkdir -p "$(dirname -- "${destination_file}")"
        cp -p "${source_file}" "${destination_file}"
        copied=1
    done < <(find "${source_root}" -type f \
        ! -iname '*.pt' ! -iname '*.pth' ! -iname '*.ckpt' -print0)
    [[ "${copied}" == "1" ]] \
        || fail "no non-model result artifacts found under ${source_root}"
}

copy_non_model_files "${SEARCH_SOURCE}" \
    "${DESTINATION}/$(basename -- "${SEARCH_SOURCE}")"
copy_non_model_files "${JOINT_SOURCE}" \
    "${DESTINATION}/$(basename -- "${JOINT_SOURCE}")"

MODEL_FILE="$(find "${DESTINATION}" -type f \
    \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' \) -print -quit)"
[[ -z "${MODEL_FILE}" ]] \
    || fail "model file entered the publication folder: ${MODEL_FILE}"
OVERSIZED_FILE="$(find "${DESTINATION}" -type f -size +99M -print -quit)"
[[ -z "${OVERSIZED_FILE}" ]] \
    || fail "non-model result exceeds 99 MiB and cannot be pushed as regular Git: ${OVERSIZED_FILE}"

# Only .gitignore and this experiment folder may be tracked on the branch.
printf '%s\n' \
    '*' \
    '!/.gitignore' \
    "!/${EXP_NAME}/" \
    "!/${EXP_NAME}/**" \
    > "${RESULT_REPO_DIR}/.gitignore"

git -C "${RESULT_REPO_DIR}" add --all
while IFS= read -r -d '' tracked_file; do
    case "${tracked_file}" in
        .gitignore|"${EXP_NAME}"/*) ;;
        *) fail "unexpected tracked path on orphan branch: ${tracked_file}" ;;
    esac
done < <(git -C "${RESULT_REPO_DIR}" ls-files -z)

if git -C "${RESULT_REPO_DIR}" diff --cached --quiet; then
    fail "nothing was staged for ${EXP_NAME}."
fi
git -C "${RESULT_REPO_DIR}" commit -m "results: ${EXP_NAME}"
git -C "${RESULT_REPO_DIR}" push --set-upstream origin \
    "HEAD:refs/heads/${EXP_NAME}"

LOCAL_COMMIT="$(git -C "${RESULT_REPO_DIR}" rev-parse HEAD)"
REMOTE_COMMIT="$(git ls-remote --heads "${RESULT_REPO_URL}" \
    "refs/heads/${EXP_NAME}" | awk 'NR == 1 {print $1}')"
[[ "${LOCAL_COMMIT}" == "${REMOTE_COMMIT}" ]] \
    || fail "push returned successfully but remote commit verification failed."

echo "RESULT_PUBLISH_COMPLETE exp_name=${EXP_NAME} branch=${EXP_NAME} commit=${LOCAL_COMMIT}"
