#!/bin/bash -l

# Run only after entering an Arrhenius GPU allocation. This script never
# requests resources; all visible GPUs must already belong to the Slurm job.
set -euo pipefail

# ========================== 用户参数 ==========================
# 数据集：spike、texture 或 uwave。
DATASET="${DATASET:-uwave}"

# 阶段：search、joint、all 或 aggregate。分开执行时先 search，再 joint；
# joint 在产物齐全后也会生成聚合报告。
STAGE="${STAGE:-all}"

# 实验唯一标识。修改模型或协议参数后必须换名，避免新旧产物混合。
EXP_NAME="${EXP_NAME:-${DATASET}_pp2}"

# 已注册骨干。正式图像实验使用 resnet18_cifar；temporal 用于轻量诊断。
BACKBONE="${BACKBONE:-resnet18_cifar}"

# 学习率选择损失：last_epoch_train_mean 或 full_train_final_model。
LOSS_SELECTION="${LOSS_SELECTION:-last_epoch_train_mean}"

# 六种方法到可见GPU编号的显式映射，必须按你已经申请到的GPU填写。编号是
# 当前作业内 torch.cuda.device_count() 所见的局部编号，通常为 0..N-1。
# 六个方法进程会并发启动；映射到同一张卡的方法也会同时占用该卡。
#
# 以下仅演示语法，不代表应申请几张GPU：
# ewc=0,er_ace=0,icarl=1,fecam=1,tagfex=2,cwr_star=2
METHOD_GPUS="${METHOD_GPUS:-}"

# 可选数据集绝对路径；留空时使用 PROJECT_ROOT/dataset。
DATASET_ROOT="${DATASET_ROOT:-}"

# 每个方法进程使用的CPU线程数，应按本次作业申请的 CPU 核数 ÷ 6
THREADS_PER_METHOD="${THREADS_PER_METHOD:-1}"

# 正式实验成功后是否复制并上传结果：1=自动发布，0=不发布。
# 仅 all、joint、aggregate 会发布；search 尚无最终联合结果，因此不发布。
PUBLISH_RESULTS="${PUBLISH_RESULTS:-1}"

# 私有结果仓库。通常不需要修改；如改用 SSH，可在启动命令中覆盖此变量。
RESULT_REPO_URL="${RESULT_REPO_URL:-https://github.com/iopav/result-pp2.git}"

# Git 提交身份。已有全局 user.name/user.email 时留空；否则在启动命令中设置，
# 例如 RESULT_GIT_USER_NAME="Your Name" RESULT_GIT_USER_EMAIL="you@example.com"。
RESULT_GIT_USER_NAME="${RESULT_GIT_USER_NAME:-}"
RESULT_GIT_USER_EMAIL="${RESULT_GIT_USER_EMAIL:-}"
# ======================== 用户参数结束 ========================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="avalanche"

cd "${PROJECT_ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: no Slurm allocation detected. Apply for resources first." >&2
    exit 1
fi
if ! [[ "${THREADS_PER_METHOD}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: THREADS_PER_METHOD must be a positive integer." >&2
    exit 2
fi
if [[ "${PUBLISH_RESULTS}" != "0" && "${PUBLISH_RESULTS}" != "1" ]]; then
    echo "ERROR: PUBLISH_RESULTS must be 0 or 1." >&2
    exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]] \
    && (( THREADS_PER_METHOD * 6 > SLURM_CPUS_PER_TASK )); then
    echo "WARNING: 6 methods x ${THREADS_PER_METHOD} threads exceeds SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}." >&2
fi
if [[ -z "${METHOD_GPUS}" && "${STAGE}" != "aggregate" ]]; then
    echo "ERROR: set METHOD_GPUS for the GPUs in your allocation." >&2
    exit 3
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
export OMP_NUM_THREADS="${THREADS_PER_METHOD}"
export MKL_NUM_THREADS="${THREADS_PER_METHOD}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_METHOD}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_METHOD}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "project_root=${PROJECT_ROOT}"
echo "dataset=${DATASET}"
echo "stage=${STAGE}"
echo "exp_name=${EXP_NAME}"
echo "backbone=${BACKBONE}"
echo "loss_selection=${LOSS_SELECTION}"
echo "method_gpus=${METHOD_GPUS:-not-used-for-aggregate}"
echo "dataset_root=${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
echo "threads_per_method=${THREADS_PER_METHOD}"
echo "publish_results=${PUBLISH_RESULTS}"
echo "result_repo_url=${RESULT_REPO_URL}"
echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
git status --short 2>/dev/null || true
nvidia-smi -L

python -c "import sys, torch; assert sys.version_info[:3] == (3, 10, 20), sys.version; assert torch.version.cuda == '13.0', torch.version.cuda; assert torch.cuda.is_available(); print('python=', sys.version.split()[0]); print('torch=', torch.__version__); print('visible_gpus=', torch.cuda.device_count())"
python verify_hpc_setup.py

SHOULD_PUBLISH=0
if [[ "${PUBLISH_RESULTS}" == "1" ]]; then
    case "${STAGE}" in
        all|joint|aggregate) SHOULD_PUBLISH=1 ;;
    esac
fi
if [[ "${SHOULD_PUBLISH}" == "1" ]]; then
    export RESULT_REPO_URL RESULT_GIT_USER_NAME RESULT_GIT_USER_EMAIL
    bash "${SCRIPT_DIR}/arrhenius_publish_results.sh" \
        --check-only "${EXP_NAME}" "${DATASET}"
fi

RUN_ARGS=(
    --stage "${STAGE}"
    --exp-name "${EXP_NAME}"
    --backbone "${BACKBONE}"
    --loss-selection "${LOSS_SELECTION}"
)
if [[ -n "${METHOD_GPUS}" ]]; then
    RUN_ARGS+=(--method-gpus "${METHOD_GPUS}")
fi
if [[ -n "${DATASET_ROOT}" ]]; then
    RUN_ARGS+=(--dataset-root "${DATASET_ROOT}")
fi

python -u "main_exp/${DATASET}.py" "${RUN_ARGS[@]}"

if [[ "${SHOULD_PUBLISH}" == "1" ]]; then
    bash "${SCRIPT_DIR}/arrhenius_publish_results.sh" \
        --publish "${EXP_NAME}" "${DATASET}"
fi
