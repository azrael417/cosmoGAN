import os
import subprocess
import shlex

# datapath = '/global/cscratch1/sd/tkurth/gb2018/cosmoGAN/tfrecord/256'
datapath = '/data1/adalbert/Maps10/tfrecords/256/'
output_size = 256
n_up = 5
c_dim = 1
epoch = 50
batch_size = 16
z_dim = 64
nd_layers = 4
ng_layers = 4
gf_dim = 64
df_dim = 64
save_every_step = 'False'
# data_format = 'NCHW'
data_format = 'NHWC'
transpose_matmul_b = False
verbose = 'True'
# nodeid = int(os.environ['SLURM_PROCID'])
# numnodes = int(os.environ['SLURM_NNODES'])

experiment = 'cosmo-new2-LARC_%i_batchSize%i_'\
             'nd%i_ng%i_gfdim%i_dfdim%i_zdim%i_nup%i'%(output_size, batch_size, nd_layers, ng_layers, gf_dim, df_dim, z_dim, n_up)
command = 'python -u -m models.main --dataset cosmo --datapath %s '\
          '--output_size %i --c_dim %i --experiment %s '\
          '--epoch %i --batch_size %i --num_updates %i --z_dim %i '\
          '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i --save_every_step %s '\
          '--data_format %s --transpose_matmul_b %s --verbose %s '\
          '--num_inter_threads %i --num_intra_threads %i'%(datapath, output_size, c_dim, experiment,\
                                                                   epoch, batch_size, n_up, z_dim,\
                                                                   nd_layers, ng_layers, gf_dim, df_dim, save_every_step,\
                                                                   data_format, transpose_matmul_b, verbose, 1, 1)

if not os.path.isdir('output'):
    os.mkdir('output')

print(command)
f_out = open('output/'+experiment+'.log', 'w')
subprocess.call(shlex.split(command), stdout=f_out)
