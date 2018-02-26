#!/bin/bash
#SBATCH -q regular
#SBATCH -C ivybridge
#SBATCH -t 2:00:00
#SBATCH -J cosmogan_horovod

#set up python stuff
module load python
source activate thorstendl-edison-2.7

#set library path
export PYTHONPATH=$(pwd)/../networks

#run training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 48 -u python -u ../networks/run_dcgan_edison.py
