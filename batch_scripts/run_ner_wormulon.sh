#!/bin/bash
#SBATCH -J ner
#SBATCH --mail-user=simpson@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=FAIL
#SBATCH --partition=ukp
#SBATCH --output=/ukp-storage-1/simpson/ner.txt
#SBATCH -e ./ner.err.%j
#SBATCH -o ./ner.out.%j
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=20000

# ----------------------------------

# module load intel python/3.6.8
source /ukp-storage-1/simpson/git/bayesian_annotator_combination/env/bin/activate
python OMP_NUM_THREADS=8 -u /ukp-storage-1/simpson/git/bayesian_annotator_combination/src/run_ner_experiments_gpu.py
