#!/bin/bash
#SBATCH -q regular
#SBATCH -A m1759
#SBATCH -C haswell
#SBATCH -t 1:00:00
#SBATCH -J cosmogan_horovod

#set up python stuff
#module load python
#source activate thorstendl-horovod
#module load gcc/6.3.0

#add this to library path:
module load tensorflow/intel-horovod-mpi-head
#modulebase=$(dirname $(module show tensorflow/intel-head 2>&1 | grep PATH |awk '{print $3}'))
export PYTHONPATH=$(pwd)/../networks  #:${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#run training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 64 -u python -u ../networks/run_dcgan.py
