import os
import subprocess

#datafile = 'data/cosmogan_maps_256_8k_1.npy'
# datafile = '/data0/mustafa/cosmo/data/cosmo_primary_256_200k_train.npy'
datafile = '/data0/tkurth/data/celebA/celebA_202599_128x128.npy'
# datafile = '/data0/adalbert/dummy_array_1.2.npy'
output_size = 128
c_dim = 3
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

experiment = 'celeb_%i_batchSize%i_'\
             'nd%i_ng%i_gfdim%i_dfdim%i_zdim%i'%(output_size, batch_size, nd_layers, ng_layers, gf_dim, df_dim, z_dim)

command = 'python -m models.main --dataset cosmo --datafile %s '\
          '--output_size %i --c_dim %i --experiment %s '\
          '--epoch %i --batch_size %i --z_dim %i '\
          '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i --save_every_step %s '\
          '--data_format %s --transpose_matmul_b %s --verbose %s'%(datafile, output_size, c_dim, experiment,\
                                                                   epoch, batch_size, z_dim,\
                                                                   nd_layers, ng_layers, gf_dim, df_dim, save_every_step,\
                                                                   data_format, transpose_matmul_b, verbose)

if not os.path.isdir('output'):
    os.mkdir('output')

print command.split()
f_out = open('output/'+experiment+'.log', 'w')
subprocess.call(command.split(), stdout=f_out)
