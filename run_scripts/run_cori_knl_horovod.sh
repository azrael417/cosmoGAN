#!/bin/bash
#SBATCH -q regular
#SBATCH -C knl
#SBATCH -t 2:00:00
#SBATCH -J cosmogan_horovod_mpi

#add this to library path:
module load tensorflow/intel-horovod-mpi-1.6
export PYTHONPATH=$(pwd)/../networks  #:${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#run training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u python -u ../networks/run_dcgan.py
