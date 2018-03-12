#!/bin/bash
#SBATCH -q regular
#SBATCH -C knl
#SBATCH -t 2:00:00
#SBATCH -J cosmowgan_horovod

#add this to library path:
module load tensorflow/intel-horovod-mpi-1.6
export PYTHONPATH=$(pwd)/../networks:${PYTHONPATH}:/global/homes/t/tkurth/.conda/envs/helper-env-py2/lib/python2.7/site-packages

#MLSL and OpenMP stuff
export OMP_NUM_THREADS=66

#run training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 python ../networks/run_dcgan.py
