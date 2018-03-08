import tensorflow as tf
import horovod.tensorflow as hvd
import train
import numpy as np
import pprint
import glob

flags = tf.app.flags
flags.DEFINE_string("dataset", "cosmo", "The name of dataset [cosmo]")
flags.DEFINE_string("datapath", "data/tfrecords", "Path to input data files")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("LARS_eta", 0.001, "LARS eta-parameter [0.001]")
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
    #init horovod
    hvd.init()

    trn_datafiles, tst_datafiles, n_records = get_data_files()
    config.num_records_total = n_records
    pprint.PrettyPrinter().pprint(config.__flags)
    train.train_dcgan((trn_datafiles, tst_datafiles), config)


def get_data_files(n_records=None):
    train_data_files = glob.glob(config.datapath + "/*train*.tfrecords")
    valid_data_files = glob.glob(config.datapath + "/*test*.tfrecords")
    if n_records is None:
        n_records = 0
        for fn in train_data_files:
          for record in tf.python_io.tf_record_iterator(fn):
             n_records += 1
        print "# records =", n_records
    else:
        n_records = len(data_files) * 20

    return train_data_files, valid_data_files, n_records


if __name__ == '__main__':
    tf.app.run()
