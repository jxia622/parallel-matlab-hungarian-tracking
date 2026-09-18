#!/bin/bash
#SBATCH --job-name=track-export-check
#SBATCH --clusters=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --output=crc_results/export-check-%j.log
set -euo pipefail
module load matlab/R2025a
matlab -singleCompThread -batch "run('gpu/verify_export_matlab.m')"
