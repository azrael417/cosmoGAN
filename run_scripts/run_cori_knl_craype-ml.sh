#!/bin/bash
#SBATCH -q premium
#SBATCH -A nstaff
#SBATCH -C knl
#SBATCH -t 1:00:00
#SBATCH -J cosmogan_horovod

#use custom craype-ml installation
module use /global/homes/t/tkurth/custom_rpm

#set up python stuff
module load tensorflow/intel-head
module use /global/homes/t/tkurth/custom_rpm/modulefiles
module load craype-ml-plugin-py2/1.1.0

#add this to library path:
export PYTHONPATH=$(pwd)/../networks:${PYTHONPATH}

#run training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u python -u ../networks/run_dcgan.py
#srun -N 1 -n 1 -c 272 -u python ../networks/run_dcgan.py
