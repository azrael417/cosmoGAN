#!/bin/bash
#SBATCH -q premium
#SBATCH -A nstaff
#SBATCH -C knl
#SBATCH -t 1:00:00
#SBATCH -J cosmogan_horovod
#SBATCH --perf=vtune

#perfcounter:
#group=MEM
#group=DATA
#group=HBM_CACHE
#group=FLOPS_SP
group=FLOPS_DP

#set up python stuff
module load python
source activate thorstendl-horovod
module load gcc/6.3.0
module load likwid

#add this to library path:
modulebase=$(dirname $(module show tensorflow/intel-head 2>&1 | grep PATH |awk '{print $3}'))
export PYTHONPATH=$(pwd)/../networks:${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#likwid-string:
likwid_command="likwid-mpirun -omp intel -O -nperdomain N:1 -pin N:0-65"

#run training
${likwid_command} -n ${SLURM_NNODES} -g ${group} python -u ../networks/run_dcgan_likwid.py --group ${group} > likwid.${group}.out
#srun -N 1 -n 1 -c 272 -u python ../networks/run_dcgan.py
