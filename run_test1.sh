#!/bin/bash
#SBATCH -J ensemble_test1
#SBATCH --mail-user=simpson@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:1

# ----------------------------------

source /ukp-storage-1/simpson/git/bayesian_annotator_combination/env/bin/activate
module purge
module load cuda/10.0
THEANO_FLAGS=mode=FAST_RUN,device=cuda*,floatX=float32,optimizer_including=cudnn python3 /ukp-storage-1/simpson/git/bayesian_annotator_combination/src/test_1.py
