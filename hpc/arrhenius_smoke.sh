#!/bin/bash -l

# Run only after entering an Arrhenius GPU allocation. This script uses cuda:0
# from the GPUs already assigned to the job; it does not request resources.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="avalanche"

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
export ARRHENIUS_PROJECT_ROOT="${PROJECT_ROOT}"

python -c "import sys, torch; assert sys.version_info[:3] == (3, 10, 20), sys.version; assert torch.version.cuda == '13.0', torch.version.cuda; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 1; print('environment=', 'avalanche'); print('python=', sys.version.split()[0]); print('torch=', torch.__version__); print('visible_gpus=', torch.cuda.device_count())"
python verify_hpc_setup.py
python -c "import torch; x=torch.randn(2048,2048,device='cuda:0'); y=x@x; torch.cuda.synchronize(); print('CUDA OK', torch.cuda.get_device_name(0), float(y[0,0]))"

# One diagnostic epoch using the actual UWave data and formal FLOP path. The
# temporal backbone keeps this test small. Its outputs are diagnostic only.
python -u - <<'PY'
import os
from pathlib import Path

import torch

from cil_experiments.runner import run_experiment

project_root = Path(os.environ["ARRHENIUS_PROJECT_ROOT"])
job_id = os.environ.get("SLURM_JOB_ID", "interactive")
exp_name = f"arrhenius_smoke_{job_id}"
summary = run_experiment(
    project_root=project_root,
    dataset_root=project_root / "dataset",
    result_root=project_root / f"result_{exp_name}",
    dataset_name="uwave",
    method="ewc",
    order_id=1,
    seed=62,
    epochs=1,
    device=torch.device("cuda:0"),
    search_provenance={"hyperparameter_search_flops": 0},
    backbone_id="temporal",
    compute_intransigence_enabled=False,
    exp_name=exp_name,
    measure_latency=False,
)
print(f"SMOKE OK: {summary}")
PY
