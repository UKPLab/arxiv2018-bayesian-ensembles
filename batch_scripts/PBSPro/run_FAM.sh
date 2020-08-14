#!/bin/sh

# Job name
#PBS -N FAM

# Output file
#PBS -o FAM_output.log

# Error file
#PBS -e FAM_err.log

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
python -u src/experiments/AAAI2020/run_famulus_semisupervised.py
python -u src/experiments/AAAI2020/run_famulus_unsupervised.py

# To submit: qsub run_NER_EMNLP19.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
