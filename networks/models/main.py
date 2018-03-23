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
flags.DEFINE_integer("num_records_total", 180000, "Number of total records. Inferred if not specified (can take time though).")
flags.DEFINE_integer("num_files_total", 4500, "Number of total files. Inferred if not specified (can take time though).")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("pix_min", 0.0, "Minimum pixel value for normalization [0.]")
flags.DEFINE_float("pix_max", 1.0, "Maximum pixel value for normalization [1.]")
flags.DEFINE_boolean("use_larc", False, "Decide whether to use LARC or not [False]")
flags.DEFINE_float("LARC_eta", 0.002, "LARC eta-parameter [0.002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
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
flags.DEFINE_string("fs_type", "global", "file system type, global or local [global]")
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

    print("Loading data")
    trn_datafiles, tst_datafiles, config.num_records_total, pix_min, pix_max=get_data_files()
    print("done.")
        
    #adjust the number of files depending on whether a global or local FS is used:
    #cut filenames:
    if config.fs_type == "global":
        #files get distributed across all nodes
        trn_datafiles = trn_datafiles[:(len(trn_datafiles) // hvd.size() * hvd.size())]
        config.num_records_total = config.num_records_total / config.num_files_total * len(trn_datafiles)
        config.num_files_total = len(trn_datafiles)
    else:
        #files are local to each node
        trn_datafiles = trn_datafiles[:(len(trn_datafiles) // hvd.local_size() * hvd.local_size())]
        config.num_records_total = config.num_records_total / config.num_files_total * len(trn_datafiles) * hvd.size() / hvd.local_size()
        config.num_files_total = len(trn_datafiles) * hvd.size() / hvd.local_size()

    if hvd.rank() == 0:
        print("Working on {} files with a total of {} samples".format(config.num_files_total,config.num_records_total))

    #update min and max values
    config.pix_min = pix_min
    config.pix_max = pix_max
    pprint.PrettyPrinter().pprint(config.__flags)
    train.train_dcgan((trn_datafiles, tst_datafiles), config)


if __name__ == '__main__':
    tf.app.run()
