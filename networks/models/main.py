import tensorflow as tf
import horovod.tensorflow as hvd
import train
import numpy as np
import pprint
import os
import glob

flags = tf.app.flags
flags.DEFINE_string("dataset", "cosmo", "The name of dataset [cosmo]")
flags.DEFINE_string("datapath", "data/tfrecords", "Path to input data files")
flags.DEFINE_integer("num_records_total", None, "Number of total records. Inferred if not specified (can take time though).")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("LARC_eta", 0.002, "LARC eta-parameter [0.002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("flip_labels", 0, "Probability of flipping labels [0]")
flags.DEFINE_integer("z_dim", 100, "Dimension of noise vector z [100]")
flags.DEFINE_integer("nd_layers", 4, "Number of discriminator convolutional layers. [4]")
flags.DEFINE_integer("ng_layers", 4, "Number of generator conv_T layers. [4]")
flags.DEFINE_integer("gf_dim", 64, "Dimension of gen filters in last conv layer. [64]")
flags.DEFINE_integer("df_dim", 64, "Dimension of discrim filters in first conv layer. [64]")
flags.DEFINE_boolean("normalize_batch", True, "The size of batch images [64]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("num_updates", 5, "Number of critic updates per generator update [5]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_string("data_format", "NHWC", "data format [NHWC]")
flags.DEFINE_boolean("transpose_matmul_b", False, "Transpose matmul B matrix for performance [False]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("plots_dir", "plots", "Directory name to save the plots [plots]")
flags.DEFINE_string("experiment", "run_0", "Tensorboard run directory name [run_0]")
flags.DEFINE_boolean("save_every_step", False, "Save a checkpoint after every step [False]")
flags.DEFINE_boolean("verbose", True, "print loss on every step [False]")
flags.DEFINE_integer("num_inter_threads", 1, "number of concurrent tasks [1]")
flags.DEFINE_integer("num_intra_threads", 4, "number of threads per task [4]")

config = flags.FLAGS

def get_data_files(compute_stat=True):
    train_data_files = glob.glob(config.datapath + "/*train*.tfrecords")
    valid_data_files = glob.glob(config.datapath + "/*test*.tfrecords")

    #load stats file
    statsfile = config.datapath + '/stats_train.npz'
    if os.path.isfile(statsfile):
        f = np.load(statsfile)
        n_records = int(f["nrecords"])
        pix_min = f['min']
        pix_max = f['max']
    else:
        n_records = 0
        pix_min = np.inf
        pix_max = -np.inf
        for fn in train_data_files:
            for record in tf.python_io.tf_record_iterator(fn):
                n_records += 1
        np.savez(statsfile, 
            {"min":pix_min, "max":pix_max, "nrecords":n_records})

    print "# records = %d; min = %2.3f; max = %2.3f"%(n_records, pix_min, pix_max)
    
    return train_data_files, valid_data_files, n_records, pix_min, pix_max


def main(_):

    #init horovod
    hvd.init()

    if hvd.rank() == 0:
        print("Loading data")
    trn_datafiles, tst_datafiles, n_records, pix_min, pix_max=get_data_files()
    if hvd.rank() == 0:
        print("done.")
        
    if not config.num_records_total:
        config.num_records_total = n_records
        config.pix_min = pix_min
        config.pix_max = pix_max
    pprint.PrettyPrinter().pprint(config.__flags)
    train.train_dcgan((trn_datafiles, tst_datafiles), config)


if __name__ == '__main__':
    tf.app.run()
