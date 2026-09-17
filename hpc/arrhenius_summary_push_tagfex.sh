#!/bin/bash -l
# One Slurm job: validate/export five methods -> publish -> resume TagFex
# orders 1/2/3 on three GPUs. Does not submit or cancel other jobs.
set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 1; }
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
[[ "${OLD_WRITER_STOPPED:-0}" == 1 ]] \
    || fail 'Stop the original spike-all writer first, then set OLD_WRITER_STOPPED=1.'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "$PROJECT_ROOT"
export DATASET="${DATASET:-spike}"
export EXP_NAME="${EXP_NAME:-spike-all}"
export BACKBONE="${BACKBONE:-resnet18_cifar}"
export LOSS_SELECTION="${LOSS_SELECTION:-last_epoch_train_mean}"
export THREADS_PER_METHOD="${THREADS_PER_METHOD:-2}"
export RESULT_REPO_URL="${RESULT_REPO_URL:-git@github.com:iopav/result-pp2.git}"
export OLD_WRITER_STOPPED
case "$DATASET" in spike|texture|uwave) ;; *) fail 'Invalid DATASET';; esac
[[ "$EXP_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail 'Invalid EXP_NAME'
[[ "$THREADS_PER_METHOD" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid THREADS_PER_METHOD'
for required in report_without_tagfex.py run_tagfex_order.py \
    hpc/arrhenius_publish_results.sh hpc/arrhenius_tagfex_orders.sh; do
    [[ -f "$required" ]] || fail "Missing script: $required"
done
command -v flock >/dev/null || fail 'flock is required'
exec 8>".summary-push-tagfex-${EXP_NAME}.lock"
flock -n 8 || fail 'Another combined job for this experiment is active'

# Use the same modules and Conda environment as arrhenius_run.sh.
CONDA_ROOT="${AVALANCHE_CONDA_ROOT:-}"
if [[ -z "$CONDA_ROOT" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
fi
CONDA_SCRIPT=""
for candidate in "$CONDA_ROOT" "$PROJECT_ROOT/.miniforge-arrhenius" \
    "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [[ -n "$candidate" && -f "$candidate/etc/profile.d/conda.sh" ]]; then
        CONDA_SCRIPT="$candidate/etc/profile.d/conda.sh"
        break
    fi
done
[[ -n "$CONDA_SCRIPT" ]] || fail 'No Conda installation found; set AVALANCHE_CONDA_ROOT.'
module purge
module load GPU/buildenv-nvhpc/25.9-cu13.0
source "$CONDA_SCRIPT"
conda activate avalanche
export OMP_NUM_THREADS="$THREADS_PER_METHOD"
export MKL_NUM_THREADS="$THREADS_PER_METHOD"
export OPENBLAS_NUM_THREADS="$THREADS_PER_METHOD"
export NUMEXPR_NUM_THREADS="$THREADS_PER_METHOD"
python -c 'import sys, torch; assert sys.version_info[:3] == (3, 10, 20), sys.version; assert torch.version.cuda == "13.0", torch.version.cuda; assert torch.cuda.is_available() and torch.cuda.device_count() >= 3, "Need three visible allocated GPUs"'

echo "job=$SLURM_JOB_ID dataset=$DATASET exp=$EXP_NAME project=$PROJECT_ROOT"
echo 'STEP_1_SUMMARY_BEGIN'
python -u report_without_tagfex.py --dataset "$DATASET" --exp-name "$EXP_NAME"
echo 'STEP_1_SUMMARY_COMPLETE'

# The existing publisher checks repository cleanliness/branch availability,
# pushes, and compares the remote commit with the local commit before returning.
echo 'STEP_2_PUSH_BEGIN'
bash "$SCRIPT_DIR/arrhenius_publish_results.sh" --publish "$EXP_NAME" "$DATASET"
echo 'STEP_2_PUSH_COMPLETE'

echo 'STEP_3_TAGFEX_BEGIN orders=1,2,3 gpus=0,1,2'
# exec keeps Slurm signals directed at the launcher that owns the workers.
# Ignore any inherited STAGE=aggregate/search: this job resumes search + joint.
export STAGE=all
exec bash "$SCRIPT_DIR/arrhenius_tagfex_orders.sh"
