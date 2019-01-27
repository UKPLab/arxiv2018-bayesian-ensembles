#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J pico_al_gpu
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e ./pico_al_gpu.err.%j
#SBATCH -o ./pico_al_gpu.out.%j
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=8182
#SBATCH --exclusive

# ----------------------------------

module load python
module load intel

python3 run_pico_active_learning.py
