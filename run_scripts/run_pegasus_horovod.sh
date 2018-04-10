#!/bin/bash

#load module
source ~/.bashrc
module load gcc/7.1
source activate tensorflow-skx

#add the models to the pythonpath
export PYTHONPATH=$(pwd)/../networks

export OMP_NUM_THREADS=18
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

#files per node
datadir=/home/users/tkurth/dl_benchmarks/cosmoGAN/data
scratchdir=/dev/shm/$(whoami)
train_per_node=264
test_per_node=64

./parallel_stagein.sh ${datadir}/train ${scratchdir} ${train_per_node}
./parallel_stagein.sh ${datadir}/test ${scratchdir} ${test_per_node}
python -u ../networks/run_dcgan_daint.py \
        --num_inter_threads 2 \
	--num_intra_threads ${OMP_NUM_THREADS} \
	--prefix "skx_" \
        --datapath ${scratchdir} \
        --fs_type local \
        --epoch 2 \
	--z_dim=100 \
	--gf_dim=64 \
	--gf_dim=64 \
	--LARC_eta=0.002 \
	--nd_layers=4 \
	--ng_layers=4

