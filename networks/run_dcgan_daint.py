import os
import subprocess
import shlex
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--prefix', help='prefix for experiment name',default='')
parser.add_argument('--datapath', help='path to dataset', default='/scratch/snx3000/tkurth/data/cosmoGAN/tfrecord/256')
parser.add_argument("--fs_type",default="global",type=str,help="FS type")
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
parser.add_argument('--use_larc', action='store_true', help='number of epochs to train for')
parser.add_argument("--trn_sz",type=int,default=-1,help="How many samples do you want to use for training? A small number can be used to help debug/overfit")
parser.add_argument('--num_inter_threads', type=int, default=1, help='number of concurrent tasks')
parser.add_argument('--num_intra_threads', type=int, default=1, help='number of threads per tasks')
opt = parser.parse_args()
print("options")
print(opt)

c_dim = 1
save_every_step = 'False'
# data_format = 'NCHW'
data_format = 'NCHW'
transpose_matmul_b = False
verbose_flag = ""
nodeid =  int(os.environ.get('SLURM_PROCID','0'))
numnodes =  int(os.environ.get('SLURM_NNODES','1'))

if opt.use_larc:
    larc_flag="--use_larc"
    larc_string="larc_"
else:
    larc_flag=""
    larc_string=""

experiment = '%scosmo-new-3_%i_batchSize%i_'\
             'nd%i_ng%i_gfdim%i_dfdim%i_zdim%i_%snodes%i_rank%i_LARCeta%2.4f_LR%2.4f'%(opt.prefix, opt.output_size, opt.batch_size, opt.nd_layers, opt.ng_layers, opt.gf_dim, opt.df_dim, opt.z_dim, larc_string, numnodes, nodeid, opt.LARC_eta, opt.learning_rate)

command = 'python -u -m models.main --dataset cosmo --datapath %s --fs_type %s '\
          '--output_size %i --c_dim %i --experiment %s '\
          '--epoch %i --trn_sz %i --batch_size %i %s --z_dim %i '\
          '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i --save_every_step %s '\
          '--data_format %s --transpose_matmul_b %s %s '\
          '--num_inter_threads %i --num_intra_threads %i --LARC_eta %f --learning_rate %f'%(opt.datapath, opt.fs_type, opt.output_size, c_dim, experiment,\
           opt.epoch, opt.trn_sz, opt.batch_size, larc_flag, opt.z_dim,\
           opt.nd_layers, opt.ng_layers, opt.gf_dim, opt.df_dim, save_every_step,\
           data_format, transpose_matmul_b, verbose_flag, opt.num_inter_threads, opt.num_intra_threads, opt.LARC_eta, opt.learning_rate)

if not os.path.isdir('output'):
    os.mkdir('output')

print(command)
print(opt)
f_out = open('output/'+experiment+'.log', 'w')
subprocess.call(shlex.split(command), stdout=f_out)
