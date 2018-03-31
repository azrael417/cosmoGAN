#!/bin/bash
#SBATCH -J cosmogan_cpe
#SBATCH -A g107
#SBATCH -t 01:00:00
#SBATCH -p normal
#SBATCH -C gpu

module use /scratch/snx3000/pjm/tmp_inst/modulefiles/

module unload PrgEnv-cray
module load PrgEnv-gnu
module load gcc/5.3.0
module load cudatoolkit/8.0.61_2.4.3-6.0.4.0_3.1__gb475d12
module load craype-ml-plugin-py2/1.1.0
source activate tensorflow1.5

#add the models to the pythonpath
export PYTHONPATH=$(pwd)/../networks:${PYTHONPATH}

export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export CRAY_CUDA_MPS=1
export MPICH_RDMA_ENABLED_CUDA=1

#run training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 24 -u python -u ../networks/run_dcgan_daint.py

