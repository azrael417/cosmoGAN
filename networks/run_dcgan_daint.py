import os
import subprocess
import shlex

datapath = '/scratch/snx3000/tkurth/data/cosmoGAN/tfrecord/256'
output_size = 256
c_dim = 1
epoch = 50
batch_size = 64
z_dim = 64
nd_layers = 4
ng_layers = 4
gf_dim = 64
df_dim = 64
save_every_step = 'False'
# data_format = 'NCHW'
data_format = 'NHWC'
transpose_matmul_b = False
use_larc = True
verbose = 'True'
nodeid = int(os.environ['SLURM_PROCID'])
numnodes = int(os.environ['SLURM_NNODES'])

if use_larc:
    larc_flag="--use_larc"
    larc_string="larc_"
else:
    larc_flag=""
    larc_string=""

experiment = 'cosmo-new_%i_batchSize%i_'\
             'nd%i_ng%i_gfdim%i_dfdim%i_zdim%i_bs%i_%snodes%i_rank%i'%(output_size, batch_size, nd_layers, ng_layers, gf_dim, df_dim, z_dim, batch_size, larc_string, numnodes, nodeid)

command = 'python -u -m models.main --dataset cosmo --datapath %s '\
          '--output_size %i --c_dim %i --experiment %s '\
          '--epoch %i --batch_size %i %s --z_dim %i '\
          '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i --save_every_step %s '\
          '--data_format %s --transpose_matmul_b %s --verbose %s '\
          '--num_inter_threads %i --num_intra_threads %i'%(datapath, output_size, c_dim, experiment,\
                                                                   epoch, batch_size, larc_flag, z_dim,\
                                                                   nd_layers, ng_layers, gf_dim, df_dim, save_every_step,\
                                                                   data_format, transpose_matmul_b, verbose, 2, 12)

if not os.path.isdir('output'):
    os.mkdir('output')

print(command)
f_out = open('output/'+experiment+'.log', 'w')
subprocess.call(shlex.split(command), stdout=f_out)
