#!/bin/bash
#SBATCH --qos=neurips-2026
#SBATCH --job-name=graphwrit3r-scannet-val
#SBATCH --output=/group/graphwrit3r/slurm_logs/graphwrit3r_scannet_val_%j.out
#SBATCH --error=/group/graphwrit3r/slurm_logs/graphwrit3r_scannet_val_%j.err
#SBATCH --time=04:00:00
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
MODEL_PATH="${MODEL_PATH:-/group/graphwrit3r/rio10_train_val3dgs/checkpoint-1900}"
INPUT_PCD="${INPUT_PCD:-/group/graphwrit3r/scannet_3dssg_scene_graph_data_subgraphs_val/pcd}"
VAL_JSON="${VAL_JSON:-/group/graphwrit3r/scannet_3dssg_scene_graph_data_subgraphs_val/scene_graph_val.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/group/graphwrit3r/rio10_train_val3dgs/infer_scannet_subgraphs_val_ckpt1900}"
CODE_TEMPLATE="${CODE_TEMPLATE:-${REPO_ROOT}/scene_graph_code_template.txt}"
CHORUS_REPO_ROOT="${CHORUS_REPO_ROOT:-${REPO_ROOT}/third_party/chorus}"
CHORUS_CHECKPOINT="${CHORUS_CHECKPOINT:-/group/graphwrit3r/lang-dino-enc-pretrain-scan-ppv2-mp3d-mcmc.pth}"
CHORUS_ALIGNED_ROOT="${CHORUS_ALIGNED_ROOT:-/group/graphwrit3r/3rscan_subset_chorus_sonata_native_cache_rio10_crop_radius_m0p25}"

require_file() {
  [[ -f "$1" ]] || { echo "FATAL: missing file: $1"; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "FATAL: missing directory: $1"; exit 1; }
}

require_dir "${REPO_ROOT}"
require_dir "${MODEL_PATH}"
require_dir "${INPUT_PCD}"
require_dir "${CHORUS_REPO_ROOT}"
require_dir "${CHORUS_ALIGNED_ROOT}"
require_file "${VAL_JSON}"
require_file "${CODE_TEMPLATE}"
require_file "${CHORUS_CHECKPOINT}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${REPO_ROOT}:${CHORUS_REPO_ROOT}:${PYTHONPATH:-}"

echo "Model:        ${MODEL_PATH}"
echo "Input PCDs:   ${INPUT_PCD}"
echo "Val JSON:     ${VAL_JSON}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Chorus repo:  ${CHORUS_REPO_ROOT}"
echo "Chorus cache: ${CHORUS_ALIGNED_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

cd "${REPO_ROOT}"
pwd

python inference_scene_graph.py \
  --model_path "${MODEL_PATH}" \
  --point_cloud "${INPUT_PCD}" \
  --val_json "${VAL_JSON}" \
  --output "${OUTPUT_DIR}" \
  --skip_existing \
  --code_template_file "${CODE_TEMPLATE}" \
  --chorus_repo_root "${CHORUS_REPO_ROOT}" \
  --chorus_config chorus_3dgs \
  --chorus_checkpoint "${CHORUS_CHECKPOINT}" \
  --chorus_aligned_root "${CHORUS_ALIGNED_ROOT}" \
  --chorus_input_mode prepared \
  --chorus_fusion_mode pcd \
  --chorus_missing_policy pcd \
  --scene_graph_mode full \
  --temperature 0.00001 \
  --top_p 1.0 \
  --top_k 1 \
  --num_beams 1 \
  --max_new_tokens 8192 \
  --inference_dtype bfloat16

echo "Inference complete. Results in: ${OUTPUT_DIR}"
echo "End: $(date)"
