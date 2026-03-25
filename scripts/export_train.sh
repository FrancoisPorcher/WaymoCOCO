#!/usr/bin/env bash
#SBATCH --job-name=waymococo_train
#SBATCH --output=/private/home/francoisporcher/WaymoCOCO/logs/export_train/export_train_%j.out
#SBATCH --error=/private/home/francoisporcher/WaymoCOCO/logs/export_train/export_train_%j.err
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=48:00:00

set -euo pipefail

REPO_ROOT="/private/home/francoisporcher/WaymoCOCO"
LOG_DIR="${REPO_ROOT}/logs/export_train"

mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate waymococo

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TFRECORD_DIR="/private/home/francoisporcher/datasets/waymo_v1_4_3/training"
WORK_DIR="/private/home/francoisporcher/data/waymococo_f0"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Waymo val export on ${SLURM_CPUS_PER_TASK:-8} CPU cores." >&2

srun --cpu-bind=none python convert_waymo_to_coco.py \
  --tfrecord_dir "${TFRECORD_DIR}" \
  --work_dir "${WORK_DIR}" \
  --image_dirname "train2020" \
  --video_dirname "training_video_tensors" \
  --image_filename_prefix "train" \
  --label_filename "instances_train2020.json" \
  --add_waymo_info \
  --write_image \
  "$@"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waymo train export finished." >&2
