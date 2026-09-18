#!/bin/bash
#SBATCH --job-name=hungarian-gpu-test
#SBATCH --clusters=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --account=kkim
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=crc_results/gpu-%j.log
set -euo pipefail
TRACKING_PYTORCH_MODULE="${TRACKING_PYTORCH_MODULE:-python/pytorch_251_311_cu124}"
TRACKING_VENV="${TRACKING_VENV:-/ihome/kkim/xiac/venvs/gpu-ulm-cu124}"
module load "${TRACKING_PYTORCH_MODULE}"
source "${TRACKING_VENV}/bin/activate"
export MAX_JOBS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$PWD/crc_results/torch_extensions"
python -u gpu/validate_gpu.py
python -u gpu/run_gpu_tracking.py fixture_bloc1_track.mat "crc_results/gpu_tracks_${SLURM_JOB_ID}.mat" --reference crc_results/gpu_oracle.mat --batch-size 384
python -u gpu/check_export.py "crc_results/gpu_tracks_${SLURM_JOB_ID}.mat"
