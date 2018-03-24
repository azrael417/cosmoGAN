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


def sample_tfrecords_to_numpy(tfrecords_filenames, img_size, sess_config, n_samples=1000, normalization=None):

    #init decoder
    dec = decoder((img_size, img_size), normalization=normalization)

    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: dec.decode(x))  # Parse the record into tensors.
    dataset = dataset.repeat(1)  # Repeat the input indefinitely.
    dataset = dataset.batch(n_samples)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Initialize `iterator` with training data.
    with tf.Session(config=sess_config) as sess:
      sess.run(iterator.initializer, 
              feed_dict={filenames: tfrecords_filenames})
      images = sess.run(next_element)
      
    return images


def generate_samples(sess, dcgan, n_batches=20):
    z_sample = np.random.normal(size=(dcgan.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.G, feed_dict={dcgan.z: z_sample})

    for i in range(0, n_batches-1):
        z_sample = np.random.normal(size=(dcgan.batch_size, dcgan.z_dim))
        samples = np.concatenate((samples, sess.run(dcgan.G, feed_dict={dcgan.z: z_sample})))
        
    return np.squeeze(samples)


def train_dcgan(datafiles, config):
    trn_datafiles, tst_datafiles = datafiles
    num_files = len(trn_datafiles)
    #comput enumber of batches and stuff
    num_samples_per_rank = config.num_records_total // hvd.size()
    num_batches_per_rank = num_samples_per_rank // config.batch_size
    num_steps_per_rank = config.epoch * num_batches_per_rank
    num_test_samples = np.min([len(tst_datafiles),1000])

    if hvd.rank() == 0:
      print("Found {} samples per rank. Using batch size {} and max epoch count {}, that gives {} number of total steps.".format(num_samples_per_rank, config.batch_size, config.epoch,num_steps_per_rank))

    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=config.num_inter_threads,
                               intra_op_parallelism_threads=config.num_intra_threads,
                               log_device_placement=False,
                               allow_soft_placement=True)

    #horovod additions
    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
    
    # load test data
    test_images = sample_tfrecords_to_numpy(tst_datafiles, config.output_size, sess_config, n_samples=num_test_samples, normalization=(config.pix_min, config.pix_max))

    # prepare plots dir
    plots_dir = os.path.join(config.plots_dir, config.experiment)
    if not os.path.exists(config.plots_dir):
      try:
        os.makedirs(config.plots_dir)
      except:
        print("Rank {}: path {} does already exist.".format(hvd.rank(),config.plots_dir))
    if not os.path.exists(plots_dir):
      try:
        os.makedirs(plots_dir)
      except:
        print("Rank {}: path {} does already exist.".format(hvd.rank(),plots_dir))


    # load test data
    test_images = sample_tfrecords_to_numpy(tst_datafiles, config.output_size, sess_config, n_samples=1000, normalization=(config.pix_min, config.pix_max))
    dump_samples(test_images, dump_path=plots_dir, tag="real samples")

    training_graph = tf.Graph()

    # save network quality history
    stats_hist = []

    with training_graph.as_default():

        #setup input pipeline
        dec = decoder(output_shape=[config.output_size, config.output_size, config.c_dim], normalization=(config.pix_min, config.pix_max))
        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(filenames)
        if hvd.size() > 1:
            dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.shuffle(config.batch_size*10)
        dataset = dataset.map(lambda x: dec.decode(x))  # Parse the record into tensors.
        # make sure all batches are equal in size
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(config.batch_size))
        dataset = dataset.repeat(config.epoch)  # Repeat the input indefinitely.
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

        #horovod additions
        sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
        hooks = []

        #stop hook
        hooks.append(tf.train.StopAtStepHook(last_step=num_steps_per_rank))

        checkpoint_dir = os.path.join(config.checkpoint_dir, config.experiment)

        if hvd.rank() == 0:
          checkpoint_save_freq = num_batches_per_rank * 2
          checkpoint_saver = gan.saver
          hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))


        #variables initializer
        init_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        init_restore = hvd.broadcast_global_variables(0)

        #some convenience functions
        c_loss_avg = gan.c_loss_average()
        g_loss_avg = gan.g_loss_average()
 
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

            #restore from cp
            if hvd.rank() == 0:
              load_checkpoint(sess, gan.saver, checkpoint_dir, step=config.save_every_step)
            #broadcast
            sess.run(init_restore)

            epoch = sess.run(gan.increment_epoch)
            start_time = time.time()

            while not sess.should_stop():           
                try:
                    #critic update
                    _, c_sum = sess.run([d_update_op, gan.c_summary], feed_dict={handle: trn_handle})
                    #query global step
                    gstep = sess.run(gan.global_step)
                    #writer.add_summary(c_sum, gstep)
                    #generator update if requested
                    if gstep%config.num_updates == 0:
                      _, g_sum = sess.run([g_update_op, gan.g_summary], feed_dict={handle: trn_handle})
                      #writer.add_summary(g_sum, gstep)

                    #if gstep%200 == 0:
                    #  # compute GAN evaluation stats
                    #  g_images = generate_samples(sess, gan)
                    #  stats = compute_evaluation_stats(g_images, test_images)
                    #  #KS summary
                    #  KS_summary = sess.run(gan.KS_summary, feed_dict={gan.KS:stats['KS']})
                    #  stats_hist += [[gstep, epoch, time.time() - start_time, stats['KS']]]

                    #  if hvd.rank() == 0:
                    #    print {k:v for k,v in stats.iteritems()}
                    #    writer.add_summary(KS_summary, gstep)
                    #    plot_pixel_histograms(g_images, test_images, dump_path=plots_dir, tag="step%d_epoch%d" % (gstep, gstep/num_batches_per_rank))
                    #    dump_samples(g_images, dump_path="%s/step%d_epoch%d" % (plots_dir, gstep, gstep/num_batches_per_rank), tag="synthetic")
                    #    np.savez("%s/stats_hist.npz" % plots_dir, np.array(stats_hist))
                    #    np.savetxt("%s/stats_hist.csv" % plots_dir, np.array(stats_hist), fmt='%.4e', delimiter='\t')
                      
                    #verbose printing
                    if config.verbose:
                        errC, errG = sess.run([c_loss_avg,g_loss_avg], feed_dict={handle: trn_handle})

                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, c_loss: %.8f, g_loss: %.8f" \
                            % (epoch, gstep, num_steps_per_rank, time.time() - start_time, errC, errG))

                    elif gstep%10 == 0:
                        errC, errG = sess.run([c_loss_avg,g_loss_avg], feed_dict={handle: trn_handle})
                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, c_loss: %.8f, g_loss: %.8f" \
                              % (epoch, gstep, num_batches_per_rank, time.time() - start_time, errC, errG))

                    # increment epoch counter
                    if gstep%num_batches_per_rank == 0:
                      epoch = sess.run(gan.increment_epoch)
                      g_images = generate_samples(sess, gan)
                      stats = compute_evaluation_stats(g_images, test_images)
                      #KS summary
                      KS_summary = sess.run(gan.KS_summary, feed_dict={gan.KS:stats['KS']})
                      stats_hist += [[gstep, epoch, time.time() - start_time, stats['KS']]]
                      if hvd.rank() == 0:
                        print {k:v for k,v in stats.iteritems()}
                        writer.add_summary(KS_summary, gstep)
                        plot_pixel_histograms(g_images, test_images, dump_path=plots_dir, tag="step%d_epoch%d" % (gstep, gstep/num_batches_per_rank))
                        dump_samples(g_images, dump_path="%s/step%d_epoch%d" % (plots_dir, gstep, gstep/num_batches_per_rank), tag="synthetic")
                        np.savez("%s/stats_hist.npz" % plots_dir, np.array(stats_hist))
                               
                except tf.errors.OutOfRangeError:
                    break
