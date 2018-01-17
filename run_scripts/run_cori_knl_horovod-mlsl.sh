#!/bin/bash
#SBATCH -q regular
#SBATCH -A nstaff
#SBATCH -C knl
#SBATCH -t 1:00:00
#SBATCH -J cosmogan_horovod-mlsl

#set up python stuff
module load python
source activate thorstendl-horovod-mlsl
module load gcc/6.3.0

#load mlsl
source ${HOME}/lib/mlsl/intel64/bin/mlslvars.sh

#add this to library path:
modulebase=$(dirname $(module show tensorflow/intel-head 2>&1 | grep PATH |awk '{print $3}'))
export PYTHONPATH=$(pwd)/../networks:${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#important, otherwise crash because MPI_Comm_Spawn is not available
export MLSL_NUM_SERVERS=0

#run training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u python -u ../networks/run_dcgan.py
