#!/bin/bash
#SBATCH -q regular
#SBATCH -A nstaff
#SBATCH -C ivybridge
#SBATCH -t 4:00:00
#SBATCH -J cosmogan_horovod


#add this to library path:
module load gcc/7.1.0
module load python/3.6-anaconda-4.4
source activate thorstendl-edison-2.7-debug
export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_VERSION_DISPLAY=1

#module load tensorflow/intel-horovod-mpi-head
#modulebase=$(dirname $(module show tensorflow/intel-head 2>&1 | grep PATH |awk '{print $3}'))
export PYTHONPATH=$(pwd)/../networks  #:${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#clean cp
rm -r checkpoints/*

#run training
#module load allineatools
#srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 48 -u python -u ../networks/run_ot_gan.py
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 48 -u ./run_ot_gan.sh
#srun -N 1 -n 1 -c 272 -u python ../networks/run_cramer_dcgan.py


