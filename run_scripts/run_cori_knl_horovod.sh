#!/bin/bash
#SBATCH -q premium
#SBATCH -A nstaff
#SBATCH -C knl
#SBATCH -t 4:00:00
#SBATCH -J cosmogan_horovod

#set up python stuff
#module load python
#source activate thorstendl-horovod
#module load gcc/6.3.0

#add this to library path:
module load tensorflow/intel-horovod-mpi-head
#modulebase=$(dirname $(module show tensorflow/intel-head 2>&1 | grep PATH |awk '{print $3}'))
export PYTHONPATH=$(pwd)/../networks  #:${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#clean cp
rm -r checkpoints/*

#run training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u python -u ../networks/run_ot_gan.py
#srun -N 1 -n 1 -c 272 -u python ../networks/run_cramer_dcgan.py
