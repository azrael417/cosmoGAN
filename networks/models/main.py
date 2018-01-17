import tensorflow as tf
#import cray plugin
import ml_comm as mc
import math

import train
import numpy as np
import pprint
import sys

flags = tf.app.flags
flags.DEFINE_string("dataset", "cosmo", "The name of dataset [cosmo]")
flags.DEFINE_string("datafile", "data/cosmo_primary_64_1k_train.npy", "Input data file for cosmo")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("flip_labels", 0, "Probability of flipping labels [0]")
flags.DEFINE_integer("z_dim", 100, "Dimension of noise vector z [100]")
flags.DEFINE_integer("nd_layers", 4, "Number of discriminator convolutional layers. [4]")
flags.DEFINE_integer("ng_layers", 4, "Number of generator conv_T layers. [4]")
flags.DEFINE_integer("gf_dim", 64, "Dimension of gen filters in last conv layer. [64]")
flags.DEFINE_integer("df_dim", 64, "Dimension of discrim filters in first conv layer. [64]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
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
config = flags.FLAGS

def main(_):
    print("Initializing cosmoGAN")
    
    #init MPI
    nthreads_per_team=1
    nteams=2
    mc.init(nthreads_per_team, nteams, 20*1024*1024, "tensorflow")
    
    #call training
    pprint.PrettyPrinter().pprint(config.__flags)
    train.train_dcgan(get_data(), config)

def get_data():
    data = np.load(config.datafile, mmap_mode='r')
    
    #make sure that each node only works on its chunk of the data
    num_samples = data.shape[0]
    num_ranks = mc.get_nranks()
    num_samples_per_rank = num_samples // num_ranks
    start = num_samples_per_rank*mc.get_rank()
    end = np.min([num_samples_per_rank*(mc.get_rank()+1),num_samples])
    data = data[start:end,:,:]

    if config.data_format == 'NHWC':
        data = np.expand_dims(data, axis=-1)
    else: # 'NCHW'
        data = np.expand_dims(data, axis=1)

    return data

if __name__ == '__main__':
    tf.app.run()
