#!/bin/bash -l

# Run only after entering an Arrhenius GPU allocation. This script uses cuda:0
# from the GPUs already assigned to the job; it does not request resources.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="avalanche"
SMOKE_ROOT="${SMOKE_ROOT:-${PROJECT_ROOT}/smoke}"

cd "${PROJECT_ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: no Slurm allocation detected. Apply for resources first." >&2
    exit 1
fi

CONDA_ROOT="${AVALANCHE_CONDA_ROOT:-}"
if [[ -z "${CONDA_ROOT}" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
fi
for candidate in \
    "${CONDA_ROOT}" \
    "${PROJECT_ROOT}/.miniforge-arrhenius" \
    "${HOME}/miniforge3" \
    "${HOME}/miniconda3" \
    "${HOME}/anaconda3"; do
    if [[ -n "${candidate}" && -f "${candidate}/etc/profile.d/conda.sh" ]]; then
        CONDA_ROOT="${candidate}"
        break
    fi
done
if [[ -z "${CONDA_ROOT}" ]]; then
    echo "ERROR: no usable Conda installation found; run arrhenius_setup.sh." >&2
    exit 1
fi

module purge
module load GPU/buildenv-nvhpc/25.9-cu13.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-job_${SLURM_JOB_ID}}"
SMOKE_OUTPUT_ROOT="${SMOKE_ROOT}/${SMOKE_RUN_NAME}"

python -c "import sys, torch; assert sys.version_info[:3] == (3, 10, 20), sys.version; assert torch.version.cuda == '13.0', torch.version.cuda; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 1; print('environment=', 'avalanche'); print('python=', sys.version.split()[0]); print('torch=', torch.__version__); print('visible_gpus=', torch.cuda.device_count())"
python -c "import torch; x=torch.randn(2048,2048,device='cuda:0'); y=x@x; torch.cuda.synchronize(); print('CUDA OK', torch.cuda.get_device_name(0), float(y[0,0]))"

echo "smoke_output_root=${SMOKE_OUTPUT_ROOT}"

# Synthetic end-to-end smoke: first audits one formal ResNet18 train/backward/eval
# step, then runs 3 datasets, 6 methods, 2 orders, 1 seed, 2 learning rates and
# 1 epoch per experience with the lightweight temporal backbone. Runs sequentially
# on cuda:0 and never reads or writes the formal dataset/results.
python -u -m cil_experiments.smoke_pipeline \
    --output-root "${SMOKE_OUTPUT_ROOT}" \
    --device cuda:0 \
    --exp-name "arrhenius_toy_smoke_${SMOKE_RUN_NAME}"
