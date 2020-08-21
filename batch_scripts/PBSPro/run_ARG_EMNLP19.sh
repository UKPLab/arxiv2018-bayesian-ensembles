#!/bin/sh

# Job name
#PBS -N ARg_EMNLP19

# Output file
#PBS -o ARG_EMNLP19_output.log

# Error file
#PBS -e ARG_EMNLP19_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=24:mem=16GB
#:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch lang/cuda

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/arxiv2018-bayesian-ensembles
export PYTHONPATH=$PYTHONPATH:"/work/es1595/arxiv2018-bayesian-ensembles/src"

#  run the script
python -u src/experiments/EMNLP2019/run_arg_experiments.py

# To submit: qsub run_NER_EMNLP19.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
