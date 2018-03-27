import os
import subprocess as sp
import shlex
import argparse

def main():
    AP = argparse.ArgumentParser()
    AP.add_argument("--datapath",type=str,required=True,help="Path to data")
    AP.add_argument("--batch_size",default=64,type=int,help="Number samples in a batch")
    AP.add_argument("--lr",default=1e-4,type=float,help="Learning rate")
    AP.add_argument("--fs_type",default="global",type=str,help="FS type")
    AP.add_argument("--use_larc", action='store_true')
    AP.add_argument("--epochs",type=int,default=50,help="Number of epochs to run")
    AP.add_argument("--trn_sz",type=int,default=-1,help="How many samples do you want to use for training? A small number can be used to help debug/overfit")
    args = AP.parse_args()

    # datapath = '/global/cscratch1/sd/tkurth/gb2018/cosmoGAN/tfrecord/256'
    datapath = args.datapath
    fs_type = args.fs_type
    output_size = 256
    n_up = 5
    c_dim = 1
    epoch = args.epochs
    trn_sz = args.trn_sz
    batch_size = args.batch_size
    learning_rate = args.lr
    z_dim = 64
    nd_layers = 4
    ng_layers = 4
    gf_dim = 64
    df_dim = 64
    # data_format = 'NCHW'
    data_format = 'NHWC'
    transpose_matmul_b = False
    verbose = True
    larc_string=""
    if args.use_larc:
        larc_string="--use_larc"
    nranks = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
    rank = int(os.environ.get('OMPI_COMM_WORLD_NODE_RANK', '0'))

    #copy environment
    my_env = dict(os.environ.copy())

    experiment = 'cosmo-new2-LARC_%i_batchSize%i_'\
                 'nd%i_ng%i_gfdim%i_dfdim%i_zdim%i_nup%i_nranks%i_rank%i'%(output_size, batch_size, nd_layers, ng_layers, gf_dim, df_dim, z_dim, n_up, nranks, rank)
    command = 'python -u -m models.main --dataset cosmo --datapath %s --fs_type %s '\
            '--output_size %i --c_dim %i --experiment %s '\
            '--epoch %i --trn_sz %i --batch_size %i --learning_rate %f --num_updates %i %s --z_dim %i '\
            '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i '\
            '--data_format %s --transpose_matmul_b %s --%sverbose '\
            '--num_inter_threads %i --num_intra_threads %i'%(datapath, fs_type, output_size, c_dim, experiment,\
                                                             epoch, trn_sz, batch_size, learning_rate, n_up, larc_string, z_dim,\
                                                             nd_layers, ng_layers, gf_dim, df_dim,\
                                                             data_format, transpose_matmul_b, ('' if verbose else 'no'), 5, 1)
    
    if not os.path.isdir('output'):
        try:
            os.mkdir('output')
        except:
            print("Rank {}: directory output already exists.".format(rank))

    print(command)
    outfile = 'output/'+experiment+'.log'
    with open(outfile, 'w') as f_out:
        #proc = sp.Popen(shlex.split(command), stdout=f_out, stderr=sp.PIPE, env=my_env)
        #out, err = proc.communicate()
        #print(err)
        sp.call(shlex.split(command), stdout=f_out)


if __name__ == '__main__':
    main()
        
