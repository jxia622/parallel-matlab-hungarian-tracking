#!/bin/bash
#SBATCH --job-name=bal-track-check
#SBATCH --clusters=smp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=crc_results/slurm-%j.log
set -euo pipefail
module load matlab/R2025a
matlab -singleCompThread -batch run_crc_validation
