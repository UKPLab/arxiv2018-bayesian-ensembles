#!/bin/sh

# Job name
#PBS -N PICO_AAAI20

# Output file
#PBS -o PICO_AAAI20_output.log

# Error file
#PBS -e PICO_AAAI20_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:ngpus=4:mem=32GB
#:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch lang/cuda

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/arxiv2018-bayesian-ensembles
export PYTHONPATH=$PYTHONPATH:"/work/es1595/arxiv2018-bayesian-ensembles/src"

#  run the script
python -u src/experiments/AAAI2020/run_pico_experiments_gpu.py

# To submit: qsub run_NER_EMNLP19.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
