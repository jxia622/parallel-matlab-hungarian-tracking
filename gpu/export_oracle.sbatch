#!/bin/bash
#SBATCH --job-name=track-oracle
#SBATCH --clusters=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=crc_results/oracle-%j.log
set -euo pipefail
module load matlab/R2025a
matlab -singleCompThread -batch "run('gpu/export_gpu_oracle.m')"
