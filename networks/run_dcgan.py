import os
import subprocess


datapath = '/global/cscratch1/sd/tkurth/gb2018/cosmoGAN/tfrecord/256'
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
verbose = 'True'

experiment = 'cosmo-new_%i_batchSize%i_'\
             'nd%i_ng%i_gfdim%i_dfdim%i_zdim%i'%(output_size, batch_size, nd_layers, ng_layers, gf_dim, df_dim, z_dim)

command = 'python -m models.main --dataset cosmo --datapath %s '\
          '--output_size %i --c_dim %i --experiment %s '\
          '--epoch %i --batch_size %i --z_dim %i '\
          '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i --save_every_step %s '\
          '--data_format %s --transpose_matmul_b %s --verbose %s'%(datapath, output_size, c_dim, experiment,\
                                                                   epoch, batch_size, z_dim,\
                                                                   nd_layers, ng_layers, gf_dim, df_dim, save_every_step,\
                                                                   data_format, transpose_matmul_b, verbose)

if not os.path.isdir('output'):
    os.mkdir('output')

print command.split()
f_out = open('output/'+experiment+'.log', 'w')
subprocess.call(command.split(), stdout=f_out)
