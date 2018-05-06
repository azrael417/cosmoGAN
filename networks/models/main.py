import tensorflow as tf
use_horovod = True
try:
    import horovod.tensorflow as hvd
except:
    use_horovod = False
import models.train as train
import numpy as np
import pprint
from ops import comm_utils

flags = tf.app.flags
flags.DEFINE_string("model", "dcgan", "dcgan/cramer_dcgan [dcgan]")
flags.DEFINE_string("dataset", "cosmo", "The name of dataset [cosmo]")
flags.DEFINE_string("datafile", "data/cosmo_primary_64_1k_train.npy", "Input data file for cosmo")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("n_up", 1, "Number of discriminator updates per generator update [1]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("flip_labels", 0, "Probability of flipping labels [0]")
flags.DEFINE_integer("z_dim", 100, "Dimension of noise vector z [100]")
flags.DEFINE_integer("d_out_dim", 256, "discriminator output dimension for cramer dcgan [256]")
flags.DEFINE_float("gradient_lambda", 10., "Gradient penalty scale in cramer dcgan [10.]")
flags.DEFINE_integer("nd_layers", 4, "Number of discriminator convolutional layers. [4]")
flags.DEFINE_integer("ng_layers", 4, "Number of generator conv_T layers. [4]")
flags.DEFINE_integer("gf_dim", 64, "Dimension of gen filters in last conv layer. [64]")
flags.DEFINE_integer("df_dim", 64, "Dimension of discrim filters in first conv layer. [64]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images per node [64]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_string("data_format", "NHWC", "data format [NHWC]")
flags.DEFINE_boolean("transpose_matmul_b", False, "Transpose matmul B matrix for performance [False]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("experiment", "run_0", "Tensorboard run directory name [run_0]")
flags.DEFINE_boolean("save_every_step", False, "Save a checkpoint after every step [False]")
flags.DEFINE_boolean("verbose", True, "print loss on every step [False]")
flags.DEFINE_integer("num_inter_threads", 1, "number of concurrent tasks [1]")
flags.DEFINE_integer("num_intra_threads", 4, "number of threads per task [4]")
#some comm hacking
flags.DEFINE_integer("comm_size", 1, "number of ranks [1]")
flags.DEFINE_integer("comm_rank", 0, "mpi rank id [0]")
flags.DEFINE_integer("comm_local_rank", 0, "node-local mpi rank id [0]")
flags.DEFINE_integer("num_row_ranks", 1, "Number of row ranks [1]")
flags.DEFINE_integer("num_col_ranks", 1, "Number of column ranks [1]")
config = flags.FLAGS

def main(_):
    #init horovod
    assert(use_horovod)
    hvd.init()
    config.comm_size = hvd.size()
    config.comm_rank = hvd.rank()
    config.comm_local_rank = hvd.local_rank()
    #create topological comm
    # Make sure MPI is not re-initialized.
    import mpi4py.rc
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    #make sure sizes are correct
    assert hvd.size() == MPI.COMM_WORLD.Get_size()
    
    #do some sanity checking
    #number of total ranks is a product of row and column ranks
    assert( config.comm_size == config.num_row_ranks * config.num_col_ranks )
    #for the moment, assert square topology
    assert( config.num_row_ranks == config.num_col_ranks )
    #create topological communicator
    #size of comm grid
    comm_row_size = config.num_row_ranks
    comm_col_size = config.num_col_ranks
    local_row_size = config.batch_size
    local_col_size = config.batch_size
    
    #get rank and comm size info
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    comm_topo = comm_utils(comm, comm_size, comm_rank, comm_row_size, comm_col_size)
    comm_topo.local_row_size = local_row_size
    comm_topo.local_col_size = local_col_size
    
    pprint.PrettyPrinter().pprint(config.__flags)

    if config.model == 'otgan':
        train.train_otgan(comm_topo, get_data(), config)
    else:
        raise ValueError("Error, only OT-GAN training supported.")

def get_data():
    data = np.load(config.datafile, mmap_mode='r')

    #make sure that each node only works on its chunk of the data
    num_samples = data.shape[0]
    if use_horovod:
        num_ranks = hvd.size()
        comm_rank = hvd.rank()
    else:
        num_ranks = 1
        comm_rank = 0
    num_samples_per_rank = num_samples // num_ranks
    start = num_samples_per_rank*comm_rank
    end = np.min([num_samples_per_rank*(comm_rank+1),num_samples])
    data = data[start:end,:,:]

    if config.data_format == 'NHWC':
        data = np.expand_dims(data, axis=-1)
    else: # 'NCHW'
        data = np.expand_dims(data, axis=1)

    return data

if __name__ == '__main__':
    tf.app.run()
