#!/bin/bash
#SBATCH -q regular
#SBATCH -C knl
#SBATCH -t 0:59:00
#SBATCH --gres=craynetwork:2
#SBATCH -J cosmogan_horovod_mlsl

#add this to library path:
module load tensorflow/intel-horovod-mlsl-1.6
export PYTHONPATH=$(pwd)/../networks  #:${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#MLSL stuff
export MLSL_NUM_SERVERS=2
export EPLIB_MAX_EP_PER_TASK=${MLSL_NUM_SERVERS}
export EPLIB_UUID="00FF00FF-0000-0000-0000-00FF00FF00FF"
export EPLIB_DYNAMIC_SERVER="disable"
export EPLIB_SERVER_AFFINITY=67,66
#export MLSL_LOG_LEVEL=5
export EPLIB_SHM_SIZE_GB=20
export MLSL_SHM_SIZE_GB=20
#export TF_MKL_ALLOC_MAX_BYTES=$((16*1024*1024*1024))
export USE_HVD=1
export PYTHONUNBUFFERED=1
export USE_MLSL_ALLOCATOR=1
export EP_PROCESS_NUM=$((${SLURM_NNODES}*${MLSL_NUM_SERVERS}))

#better binding
bindstring="numactl -C 1-65,69-133,137-201,205-269"

#launch MLSL server
srun -N ${SLURM_NNODES} -n ${EP_PROCESS_NUM} -c 4 --mem=37200 --gres=craynetwork:1 ${MLSL_ROOT}/intel64/bin/ep_server &

#run training
srun -l --zonesort=off -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 264 --mem=37200 --gres=craynetwork:0 ${bindstring} python ../networks/run_dcgan.py
wait
