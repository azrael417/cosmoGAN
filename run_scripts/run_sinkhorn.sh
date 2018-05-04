#!/bin/bash
#SBATCH -q regular
#SBATCH -A dasrepo
#SBATCH -C knl
#SBATCH -t 00:10:00
#SBATCH -J distributed_sinkhorn

#module loads
module load python/3.6-anaconda-4.4
source activate thorstendl-devel

#run the code
srun -n 4 -c 64 python ../networks/models/distributed_sinkhorn.py --row_comms=2 --col_comms=2 --rank=8
