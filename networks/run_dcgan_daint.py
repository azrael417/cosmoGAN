import os
import subprocess
import shlex
import argparse

parser = argparse.ArgumentParser()
datapath = '/scratch/snx3000/tkurth/data/cosmoGAN/tfrecord/256'

parser.add_argument('--datapath', help='path to dataset', default='/scratch/snx3000/tkurth/data/cosmoGAN/tfrecord/256')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--output_size', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--z_dim', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--gf_dim', type=int, default=64)
parser.add_argument('--df_dim', type=int, default=64)
parser.add_argument('--ng_layers', type=int, default=4)
parser.add_argument('--nd_layers', type=int, default=4)
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--LARC_eta', type=float, default=0.002, help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='number of epochs to train for')
parser.add_argument('--use_larc', type=bool, default=True, help='number of epochs to train for')

opt = parser.parse_args()

c_dim = 1
batch_size = 2
save_every_step = 'False'
# data_format = 'NCHW'
data_format = 'NHWC'
transpose_matmul_b = False
verbose = 'True'
nodeid = int(os.environ['SLURM_PROCID'])
numnodes = int(os.environ['SLURM_NNODES'])

if opt.use_larc:
    larc_flag="--use_larc"
    larc_string="larc_"
else:
    larc_flag=""
    larc_string=""

experiment = 'cosmo-new_%i_batchSize%i_'\
             'nd%i_ng%i_gfdim%i_dfdim%i_zdim%i_%snodes%i_rank%i'%(opt.output_size, opt.batch_size, opt.nd_layers, opt.ng_layers, opt.gf_dim, opt.df_dim, opt.z_dim, larc_string, numnodes, nodeid)

command = 'python -u -m models.main --dataset cosmo --datapath %s --fs_type global '\
          '--output_size %i --c_dim %i --experiment %s '\
          '--epoch %i --batch_size %i %s --z_dim %i '\
          '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i --save_every_step %s '\
          '--data_format %s --transpose_matmul_b %s --verbose %s '\
          '--num_inter_threads %i --num_intra_threads %i'%(opt.datapath, opt.output_size, c_dim, experiment,\
           opt.epoch, opt.batch_size, larc_flag, opt.z_dim,\
           opt.nd_layers, opt.ng_layers, opt.gf_dim, opt.df_dim, save_every_step,\
           data_format, transpose_matmul_b, verbose, 2, 12)

if not os.path.isdir('output'):
    os.mkdir('output')

print(command)
f_out = open('output/'+experiment+'.log', 'w')
subprocess.call(shlex.split(command), stdout=f_out)
