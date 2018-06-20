#!/bin/bash

#input data
train_datafile=$3 #'/global/cscratch1/sd/tkurth/gb2018/cosmoGAN/small_set/cosmo_primary_256_200k_train.npy'
test_datafile=$4 #'/global/cscratch1/sd/tkurth/gb2018/cosmoGAN/small_set/cosmo_primary_256_200k_test.npy'

#rank grid
num_row_ranks=$1
num_col_ranks=$2

output_size=256
epoch=300
print_frequency=10
learning_rate=0.001
n_up=5
flip_labels=0.01
batch_size=384
z_dim=64
nd_layers=4
ng_layers=4
gf_dim=32
df_dim=32
sinkhorn_lambda=1.
data_format='NHWC'
solver="RMSProp"
prior_dist="uniform"
nodeid=${SLURM_PROCID}
numnodes=${SLURM_NNODES}

#experiment directory
experiment="cramer_otgan_cosmo_primary_256_200k_batchSize${batch_size}_solver${solver}_learningRate_${learning_rate}_nUp${n_up}_priorDist${prior_dist}_sinkhornLambda${sinkhorn_lambda}_flipLabel${flip_labels}_nd${nd_layers}_ng${ng_layers}_gfdim${gf_dim}_dfdim${df_dim}_zdim${z_dim}_nodes${numnodes}_rank${nodeid}"

command="python -m models.main --model otgan --dataset cosmo --train_datafile ${train_datafile} --test_datafile ${test_datafile}"
command=${command}" --output_size ${output_size} --solver ${solver} --learning_rate ${learning_rate} --n_up ${n_up} --prior_dist ${prior_dist}"
command=${command}" --sinkhorn_lambda ${sinkhorn_lambda} --flip_labels ${flip_labels} --experiment ${experiment}"
command=${command}" --epoch ${epoch} --batch_size ${batch_size} --z_dim ${z_dim}"
command=${command}" --nd_layers ${nd_layers} --ng_layers ${ng_layers} --gf_dim ${gf_dim} --df_dim ${df_dim} --print_frequency ${print_frequency}"
command=${command}" --data_format ${data_format} --transpose_matmul_b --verbose --num_inter_threads 2 --num_intra_threads 16"
command=${command}" --num_row_ranks ${num_row_ranks} --num_col_ranks ${num_col_ranks}"

if [ ! -d 'output' ]; then
    mkdir -p output
fi

#run command
if [ "${nodeid}" == "0" ]; then
    echo ${command}
fi
${command}
