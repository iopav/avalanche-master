#!/bin/bash -l
# Run inside an existing three-GPU Slurm allocation with avalanche activated.
# The original writer must be stopped first. Do not change CUDA_VISIBLE_DEVICES.
set -euo pipefail
: "${SLURM_JOB_ID:?Enter a Slurm GPU allocation first}"
: "${OLD_WRITER_STOPPED:?Set OLD_WRITER_STOPPED=1 after stopping the old experiment job}"
[[ "$OLD_WRITER_STOPPED" == 1 ]] || exit 2
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
DATASET="${DATASET:-spike}"
EXP_NAME="${EXP_NAME:-spike-all}"
STAGE="${STAGE:-all}"
BACKBONE="${BACKBONE:-resnet18_cifar}"
LOSS_SELECTION="${LOSS_SELECTION:-last_epoch_train_mean}"
[[ "$EXP_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 2
export OMP_NUM_THREADS="${THREADS_PER_METHOD:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 3, "Need three allocated visible GPUs"'
# Advisory lock protects this launcher against duplicate launches. Legacy jobs
# do not use it; OLD_WRITER_STOPPED is a required operator assertion.
exec 9>".tagfex-orders-${EXP_NAME}.lock"
flock -n 9 || { echo 'Another order launcher is active' >&2; exit 1; }
LOG_DIR="logs/tagfex-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID}-$$"
mkdir -p "$LOG_DIR"
args=(--dataset "$DATASET" --exp-name "$EXP_NAME" --stage "$STAGE"
      --backbone "$BACKBONE" --loss-selection "$LOSS_SELECTION")
[[ -z "${DATASET_ROOT:-}" ]] || args+=(--dataset-root "$DATASET_ROOT")
# Check all identities before starting any worker.
for order in 1 2 3; do
    python run_tagfex_order.py "${args[@]}" --order-id "$order" --gpu "$((order-1))" --dry-run
done
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait || true
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM HUP
for order in 1 2 3; do
    python -u run_tagfex_order.py "${args[@]}" --order-id "$order" --gpu "$((order-1))" \
        >"$LOG_DIR/order${order}.log" 2>&1 &
    pids+=("$!")
done
echo "TagFex workers=${pids[*]} logs=$ROOT/$LOG_DIR"
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "order=$((index+1)) completed"
    else
        echo "order=$((index+1)) failed; inspect $LOG_DIR/order$((index+1)).log" >&2
        status=1
    fi
done
exit "$status"
