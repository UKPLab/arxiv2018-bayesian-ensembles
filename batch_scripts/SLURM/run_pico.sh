#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J pico
#SBATCH --mail-user=simpson@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=FAIL
#SBATCH -e ./pico.err.%j
#SBATCH -o ./pico.out.%j
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=8182
#SBATCH --exclusive
#SBATCH -C avx

# ----------------------------------


module load intel python/3.6.8
python -u ./src/run_pico_experiments.py
