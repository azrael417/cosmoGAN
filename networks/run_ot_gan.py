import os
import subprocess

datafile = '/data0/mustafa/cosmo/data/cosmo_primary_256_200k_train.npy'
#datafile = '/global/cscratch1/sd/tkurth/gb2018/cosmoGAN/small_set/cosmo_primary_256_200k_train.npy'

output_size = 256
epoch = 300
learning_rate = 0.00005
n_up = 5
flip_labels = 0.01
batch_size = 16
z_dim = 64
nd_layers = 4
ng_layers = 4
gf_dim = 32
df_dim = 32
save_every_step = 'False'
data_format = 'NHWC'
transpose_matmul_b = 'True'
verbose = 'True'
nodeid = int(os.environ['SLURM_PROCID'])
numnodes = int(os.environ['SLURM_NNODES'])

experiment = 'cramer_otgan_cosmo_primary_256_200k_batchSize%i_learningRate_%0.6f_nUp%i_flipLabel%0.3f_'\
             'nd%i_ng%i_gfdim%i_dfdim%i_zdim%i_nodes%i_rank%i'%(batch_size, learning_rate, n_up, flip_labels, nd_layers,\
                                                                 ng_layers, gf_dim, df_dim, z_dim, numnodes, nodeid)

command = 'python -m models.main --model otgan --dataset cosmo --datafile %s '\
          '--output_size %i --learning_rate %f --n_up %i --flip_labels %f --experiment %s '\
          '--epoch %i --batch_size %i --z_dim %i '\
          '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i --save_every_step %s '\
          '--data_format %s --transpose_matmul_b %s --verbose %s --num_inter_threads %i --num_intra_threads %i'%(datafile, output_size, learning_rate, n_up, flip_labels, experiment,\
                                                                                                                epoch, batch_size, z_dim,\
                                                                                                                nd_layers, ng_layers, gf_dim, df_dim, save_every_step,\
                                                                                                                data_format, transpose_matmul_b, verbose, 2, 16)

if not os.path.isdir('output'):
    os.mkdir('output')

# print command.split()
# f_out = open('output/'+experiment+'.log', 'w')
subprocess.call(command.split())
# subprocess.call(command.split(), stdout=f_out)
