#!/bin/bash -l

# Run this script only after entering an Arrhenius GPU allocation. It never
# requests GPUs, CPUs, an account, a partition or walltime.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="avalanche"
FALLBACK_CONDA_ROOT="${PROJECT_ROOT}/.miniforge-arrhenius"
MINIFORGE_VERSION="26.3.2-3"
MINIFORGE_SHA256="2c113a69297e612b01ca0f320c22a3107a11f2ab9b573d79ac868a175945ce29"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-aarch64.sh"

cd "${PROJECT_ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: no Slurm allocation detected. Apply for resources first." >&2
    exit 1
fi

# Remember an already configured Conda installation before changing modules.
CONDA_ROOT="${AVALANCHE_CONDA_ROOT:-}"
if [[ -z "${CONDA_ROOT}" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
fi

module purge
module load GPU/buildenv-nvhpc/25.9-cu13.0

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "architecture=$(uname -m)"
echo "project_root=${PROJECT_ROOT}"
nvidia-smi -L
nvcc --version

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: expected an Arrhenius ARM GPU node, got $(uname -m)." >&2
    exit 1
fi

# Reuse a working user Conda installation when one is already configured.
if [[ -z "${CONDA_ROOT}" || ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    for candidate in \
        "${FALLBACK_CONDA_ROOT}" \
        "${HOME}/miniforge3" \
        "${HOME}/miniconda3" \
        "${HOME}/anaconda3"; do
        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
            CONDA_ROOT="${candidate}"
            break
        fi
    done
fi

# If Conda is genuinely absent, install an ARM64 Miniforge fallback inside the
# project. Existing installations and environments are never deleted.
if [[ -z "${CONDA_ROOT}" ]]; then
    if [[ -e "${FALLBACK_CONDA_ROOT}" ]]; then
        echo "ERROR: incomplete Conda directory: ${FALLBACK_CONDA_ROOT}" >&2
        exit 1
    fi
    DOWNLOAD_DIR="$(mktemp -d)"
    INSTALLER="${DOWNLOAD_DIR}/Miniforge3-Linux-aarch64.sh"
    trap 'rm -f "${INSTALLER}"; rmdir "${DOWNLOAD_DIR}" 2>/dev/null || true' EXIT
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "${INSTALLER}" "${MINIFORGE_URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --output-document="${INSTALLER}" "${MINIFORGE_URL}"
    else
        echo "ERROR: neither curl nor wget is available." >&2
        exit 1
    fi
    echo "${MINIFORGE_SHA256}  ${INSTALLER}" | sha256sum --check --strict
    bash "${INSTALLER}" -b -p "${FALLBACK_CONDA_ROOT}"
    CONDA_ROOT="${FALLBACK_CONDA_ROOT}"
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
CONDA_ENV_DIR="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"

if conda env list | awk 'NF && $1 == "avalanche" {found=1} END {exit !found}'; then
    echo "Reusing existing Conda environment: ${CONDA_ENV_NAME}"
    conda install --name "${CONDA_ENV_NAME}" --yes --strict-channel-priority \
        python=3.10.20 pip setuptools wheel
elif [[ -e "${CONDA_ENV_DIR}" ]]; then
    echo "ERROR: incomplete environment directory: ${CONDA_ENV_DIR}" >&2
    exit 1
else
    conda create --name "${CONDA_ENV_NAME}" --yes --strict-channel-priority \
        python=3.10.20 pip setuptools wheel
fi

conda activate "${CONDA_ENV_NAME}"
python -c "import platform, sys; assert sys.version_info[:3] == (3, 10, 20), sys.version; assert platform.machine() == 'aarch64', platform.machine(); print('python=', platform.python_version()); print('architecture=', platform.machine())"

# These commands are idempotent: already-correct packages remain installed;
# missing or wrong pinned versions are installed or replaced.
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check

python -c "import torch; assert torch.__version__.split('+')[0] == '2.11.0', torch.__version__; assert torch.version.cuda == '13.0', torch.version.cuda; assert torch.cuda.is_available(); print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('visible_gpus=', torch.cuda.device_count()); print('gpu0=', torch.cuda.get_device_name(0))"
python verify_hpc_setup.py

echo "conda_root=${CONDA_ROOT}"
conda env list
echo "SETUP OK: ${CONDA_ENV_NAME}"
