import os
import time
import numpy as np
import tensorflow as tf
import wgan_dcgan as dcgan
import horovod.tensorflow as hvd
from utils import save_checkpoint, load_checkpoint
import matplotlib
matplotlib.use('Agg')

from validation import *

from scipy import stats


class decoder(object):
  
  def __init__(self, output_shape, normalization=None):
    self.output_shape = output_shape
    if normalization:
      self.minval, self.maxval = normalization
    else:
      self.minval, self.maxval = (0.,1.)
      
      
  def decode(self, x):
    parsed_example = tf.parse_single_example(x,
                                             features = {
                                               "data_raw": tf.FixedLenFeature([],tf.string)
                                             })

    example = tf.decode_raw(parsed_example['data_raw'],tf.float32)
    example = 2* (tf.reshape(example,self.output_shape) - self.minval) / (self.maxval - self.minval) -1.
    return example


def generate_samples(sess, dcgan, n_batches=20):
    z_sample = np.random.normal(size=(dcgan.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.G, feed_dict={dcgan.z: z_sample})

    for i in range(0, n_batches-1):
        z_sample = np.random.normal(size=(dcgan.batch_size, dcgan.z_dim))
        samples = np.concatenate((samples, sess.run(dcgan.G, feed_dict={dcgan.z: z_sample})))
        
    return np.squeeze(samples)


def train_dcgan(datafiles, config):
    trn_datafiles, tst_datafiles = datafiles
    num_batches = config.num_records_total // ( config.batch_size * hvd.size() )
    num_steps = config.epoch*num_batches

    # load test data
    test_images = sample_tfrecords_to_numpy(tst_datafiles, config.output_size, normalization=(config.pix_min, config.pix_max))
    
    # prepare plots dir
    plots_dir = config.plots_dir + '/' + config.experiment
    if not os.path.exists(plots_dir):
      os.makedirs(plots_dir)

    training_graph = tf.Graph()

    with training_graph.as_default():

        #setup input pipeline
        dec = decoder(output_shape=[config.output_size, config.output_size, config.c_dim], normalization=(config.pix_min, config.pix_max))
        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(filenames)
        if hvd.size() > 1:
            dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.shuffle(config.batch_size*10)
        dataset = dataset.map(lambda x: dec.decode(x))  # Parse the record into tensors.
        dataset = dataset.repeat(config.epoch)  # Repeat the input indefinitely.
        dataset = dataset.batch(config.batch_size)
        handle = tf.placeholder(tf.string,shape=[],name="iterator-placeholder")
        # iterator = dataset.make_initializable_iterator()
        iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
        next_element = iterator.get_next()
        if config.data_format == 'NCHW':
            next_element = tf.transpose(next_element, [0,3,1,2])
        #train iterator
        trn_iterator = dataset.make_initializable_iterator()
        trn_handle_string = trn_iterator.string_handle()
        trn_init_op = iterator.make_initializer(dataset)

        gan = dcgan.dcgan(output_size=config.output_size,
                          batch_size=config.batch_size,
                          nd_layers=config.nd_layers,
                          ng_layers=config.ng_layers,
                          df_dim=config.df_dim,
                          gf_dim=config.gf_dim,
                          c_dim=config.c_dim,
                          z_dim=config.z_dim,
                          data_format=config.data_format,
                          transpose_b=config.transpose_matmul_b)

        gan.training_graph(next_element)
        gan.sampling_graph()
        
        #determine which optimizer we are going to use
        if config.use_larc:
          if hvd.rank() == 0:
            print("Using LARC optimizer")
          d_update_op, g_update_op = gan.larc_optimizer(config.learning_rate)
        else:
          if hvd.rank() == 0:
            print("Disabling LARC optimizer")
          d_update_op, g_update_op = gan.optimizer(config.learning_rate)

        #session config
        sess_config=tf.ConfigProto(inter_op_parallelism_threads=config.num_inter_threads,
                                   intra_op_parallelism_threads=config.num_intra_threads,
                                   log_device_placement=False,
                                   allow_soft_placement=True)

        #horovod additions
        sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]

        #stop hook
        hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

        checkpoint_dir = os.path.join(config.checkpoint_dir, config.experiment)

        if hvd.rank() == 0:
          checkpoint_save_freq = num_batches * 2
          checkpoint_saver = gan.saver
          hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))


        #variables initializer
        init_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
 
        if hvd.rank() == 0:
            print("Starting session with {} inter- and {} intra-threads".format(config.num_inter_threads, config.num_intra_threads))
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            writer = tf.summary.FileWriter('./logs/'+config.experiment+'/train', sess.graph)

            sess.run([init_op, init_local_op])
  
            #initialize iterator
            print("Initializing Iterator")
            trn_handle = sess.run(trn_handle_string)
            sess.run(trn_init_op, 
              feed_dict={handle: trn_handle, filenames: trn_datafiles})

            load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, step=config.save_every_step)

            epoch = sess.run(gan.increment_epoch)
            start_time = time.time()

            while not sess.should_stop():           
                try:
                    #critic update
                    _, c_sum = sess.run([d_update_op, gan.c_summary], feed_dict={handle: trn_handle})
                    #query global step
                    gstep = sess.run(gan.global_step)
                    writer.add_summary(c_sum, gstep)
                    #generator update if requested
                    if gstep%config.num_updates == 0:
                      _, g_sum = sess.run([g_update_op, gan.g_summary], feed_dict={handle: trn_handle})
                      writer.add_summary(g_sum, gstep)

                    if gstep%200 == 0:
                      # compute GAN evaluation stats
                      g_images = generate_samples(sess, gan)
                      stats = compute_evaluation_stats(g_images, test_images)
                      #KS summary
                      KS_summary = sess.run(gan.KS_summary, feed_dict={gan.KS:stats['KS']})

                      if hvd.rank() == 0:
                        print {k:v for k,v in stats.iteritems()}
                        writer.add_summary(KS_summary, gstep)
                        plot_pixel_histograms(g_images, test_images, dump_path=plots_dir, tag="step%d_epoch%d" % (gstep, gstep/num_batches))
                      

                    #verbose printing
                    if config.verbose:
                        errC, errG = sess.run([gan.c_loss,gan.g_loss], feed_dict={handle: trn_handle})

                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, c_loss: %.8f, g_loss: %.8f" \
                            % (epoch, gstep, num_steps, time.time() - start_time, errC, errG))

                    elif gstep%100 == 0:
                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f"%(epoch, gstep, num_batches, time.time() - start_time))

                    # increment epoch counter
                    if gstep%num_batches == 0:
                      epoch = sess.run(gan.increment_epoch)
                      g_images = generate_samples(sess, gan)
                      stats = compute_evaluation_stats(g_images, test_images)
                      if hvd.rank() == 0:
                        print {k:v for k,v in stats.iteritems()}
                               
                except tf.errors.OutOfRangeError:
                    break
