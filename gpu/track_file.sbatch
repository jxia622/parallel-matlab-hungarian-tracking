#!/bin/bash
#SBATCH --job-name=hungarian-track
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=crc_results/track-file-%j.log
# Submit from tracking_acceleration, using absolute data paths:
# sbatch gpu/track_file.sbatch /path/input.mat /path/new_output.mat
# Add --reference /path/matlab_reference.mat for an exact comparison.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo 'Provide input.mat and a new output.mat, plus optional CLI flags.' >&2
  exit 2
fi
if [ -n "${TRACKING_PYTORCH_MODULE:-}" ]; then
  module load "${TRACKING_PYTORCH_MODULE}"
fi
if [ -n "${TRACKING_VENV:-}" ]; then
  source "${TRACKING_VENV}/bin/activate"
fi
export MAX_JOBS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$PWD/crc_results/torch_extensions"
python -u gpu/run_gpu_tracking.py "$@"
