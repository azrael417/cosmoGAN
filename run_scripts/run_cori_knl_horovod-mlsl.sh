#!/bin/bash
#SBATCH -q regular
#SBATCH -C knl
#SBATCH -t 2:00:00
#SBATCH -J cosmowgan_horovod_mlsl

#add this to library path:
module load tensorflow/intel-horovod-mlsl-1.6
export PYTHONPATH=$(pwd)/../networks:${PYTHONPATH}:/global/homes/t/tkurth/.conda/envs/helper-env-py2/lib/python2.7/site-packages

#MLSL and OpenMP stuff
export MLSL_NUM_SERVERS=0
export OMP_NUM_THREADS=66

#better binding
bindstring="numactl -C 1-65,69-133,137-201,205-269"

#run training
srun -l -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 264 ${bindstring} python ../networks/run_dcgan.py
