#!/bin/bash
#SBATCH --qos=neurips-2026
#SBATCH --job-name=spatiallm-rio10-no-cosine
#SBATCH --output=/group/graphwrit3r/slurm_logs/spatiallm_rio10_no_cosine_%j.out
#SBATCH --error=/group/graphwrit3r/slurm_logs/spatiallm_rio10_no_cosine_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=batch
#SBATCH --gres=gpu:h200:1
#SBATCH --nodelist=sof1-h200-6

set -euo pipefail
set -x
trap 'echo "ERR at line $LINENO"; exit 1' ERR

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Shell: ${SHELL}"

if [[ -x /infra/scripts/module ]]; then
  eval "$(/infra/scripts/module bash-hook)" \
    || echo "warn: /infra/scripts/module hook failed"
fi
if ! type -t module >/dev/null 2>&1; then
  if [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh \
      || echo "warn: /etc/profile.d/modules.sh source failed"
  elif [[ -r /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash \
      || echo "warn: /usr/share/Modules/init/bash source failed"
  fi
fi

type -t module || { echo "FATAL: 'module' is not available"; exit 1; }

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/scratch/luka_milivojevic/micromamba}"
/usr/bin/micromamba --version
eval "$(/usr/bin/micromamba shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")" \
  || { echo "FATAL: micromamba shell hook failed"; exit 1; }

set +u
micromamba activate "${MAMBA_ENV:-unified3d}"
set -u
echo "Activated env: $(which python)"
python -V

REPO_ROOT="${REPO_ROOT:-/group/graphwrit3r/GraphWrit3R}"
CONFIG="${CONFIG:-configs/spatiallm_scene_graph_sft_chorus_coord_matched_rio10_no_cosine.yaml}"

[[ -d "${REPO_ROOT}" ]] || { echo "FATAL: missing repo: ${REPO_ROOT}"; exit 1; }
[[ -f "${REPO_ROOT}/${CONFIG}" ]] || { echo "FATAL: missing config: ${REPO_ROOT}/${CONFIG}"; exit 1; }

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/chorus:${PYTHONPATH:-}"

echo "Config: ${REPO_ROOT}/${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

cd "${REPO_ROOT}"
pwd
python train.py "${CONFIG}"

echo "Training complete."
echo "End: $(date)"
