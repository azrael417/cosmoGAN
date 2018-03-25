#!/bin/bash
#SBATCH -J cosmogan_horovod
#SBATCH -A g107
#SBATCH -t 00:40:00
#SBATCH -p normal
#SBATCH -C gpu

module unload PrgEnv-cray
module load PrgEnv-gnu
module load gcc/5.3.0
module load cudatoolkit/8.0.61_2.4.3-6.0.4.0_3.1__gb475d12
source activate tensorflow-hp

#add the models to the pythonpath
export PYTHONPATH=$(pwd)/../networks

export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export CRAY_CUDA_MPS=1
export MPICH_RDMA_ENABLED_CUDA=1

#files per node
fpn=256
numfiles=$(( ${SLURM_NNODES} * ${fpn} ))

#run training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 24 -u python -u ../networks/run_dcgan_daint.py \
        --fs_type global \
        --epoch 2 \
        --trn_sz ${numfiles} \
	--z_dim=100 \
	--gf_dim=64 \
	--gf_dim=64 \
	--LARC_eta=0.002 \
	--nd_layers=4 \
	--ng_layers=4

