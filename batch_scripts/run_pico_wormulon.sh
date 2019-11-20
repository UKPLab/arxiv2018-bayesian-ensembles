#!/bin/bash
#SBATCH -J pico
#SBATCH --mail-user=simpson@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=FAIL
#SBATCH --partition=ukp
#SBATCH --output=/ukp-storage-1/simpson/pico.txt
#SBATCH -e ./pico.err.%j
#SBATCH -o ./pico.out.%j
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=8182


# ----------------------------------


# module load intel python/3.6.8
source /ukp-storage-1/simpson/git/bayesian_annotator_combination/env/bin/activate
python -u /ukp-storage-1/simpson/git/bayesian_annotator_combination/src/run_pico_experiments_wormulon.py
