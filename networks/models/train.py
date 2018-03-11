import os
import time
import numpy as np
import tensorflow as tf
import wgan_dcgan as dcgan
import horovod.tensorflow as hvd
from utils import save_checkpoint, load_checkpoint
from scipy import stats


def sample_tfrecords_to_numpy(tfrecords_filenames, img_size, n_samples=1000):

  def decode_record(x):
    parsed_example = tf.parse_single_example(x,
        features = {
            "data_raw": tf.FixedLenFeature([],tf.string)
        }
    )

    example = tf.decode_raw(parsed_example['data_raw'],tf.float32)
    example = tf.reshape(example,[img_size, img_size])
    return example
         
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(lambda x: decode_record(x))  # Parse the record into tensors.
  dataset = dataset.repeat(1)  # Repeat the input indefinitely.
  dataset = dataset.batch(n_samples)
  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()

  # Initialize `iterator` with training data.
  with tf.Session() as sess:
      sess.run(iterator.initializer, 
              feed_dict={filenames: tfrecords_filenames})
      images = sess.run(next_element)
      sess.close()
  return images
        

def get_hist_bins(data, get_error=False):
    y, x = np.histogram(data, bins=60, range=(-1.1,1.1))
    x = 0.5*(x[1:]+x[:-1])
    if get_error == True:
        y_err = np.sqrt(y)
        return x, y, y_err
    else:
        return x, y


def compute_evaluation_stats(fake, test):
  test_bins, test_hist = get_hist_bins(test)
  fake_bins, fake_hist = get_hist_bins(fake)
  return {"KS":stats.ks_2samp(test_hist, fake_hist)[1]}


def generate_samples(sess, dcgan, n_batches=20):
    z_sample = np.random.normal(size=(dcgan.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.G, feed_dict={dcgan.z: z_sample})

    for i in range(0, n_batches-1):
        z_sample = np.random.normal(size=(dcgan.batch_size, dcgan.z_dim))
        samples = np.concatenate((samples, sess.run(dcgan.G, feed_dict={dcgan.z: z_sample})))
        
    return np.squeeze(samples)


def train_dcgan(datafiles, config):
    trn_datafiles, tst_datafiles = datafiles
    num_batches = config.num_records_total // config.batch_size
    num_steps = config.epoch*num_batches

    # load test data
    test_images = sample_tfrecords_to_numpy(tst_datafiles, config.output_size)
    print test_images.shape

    training_graph = tf.Graph()

    with training_graph.as_default():

        # set up data ingestion pipeline
        def decode_record(x):
            parsed_example = tf.parse_single_example(x,
                features = {
                    "data_raw": tf.FixedLenFeature([],tf.string)
                }
            )
            example = tf.decode_raw(parsed_example['data_raw'],tf.float32)
            example = tf.reshape(example,
              [config.output_size, config.output_size, config.c_dim])
            return example
             
        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(lambda x: decode_record(x))  # Parse the record into tensors.
        dataset = dataset.repeat(config.epoch)  # Repeat the input indefinitely.
        dataset = dataset.batch(config.batch_size)
        handle = tf.placeholder(tf.string,shape=[],name="iterator-placeholder")
        # iterator = dataset.make_initializable_iterator()
        iterator = tf.data.Iterator.from_string_handle(handle, 
          dataset.output_types, dataset.output_shapes)
        next_element = iterator.get_next()
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
        update_op = gan.optimizer(config.learning_rate, config.LARS_eta)

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
                    _, g_sum, c_sum = sess.run([update_op, gan.g_summary, gan.c_summary], feed_dict={handle: trn_handle})
                    gstep = sess.run(gan.global_step)

                    writer.add_summary(g_sum, gstep)
                    writer.add_summary(c_sum, gstep)

                    if gstep%100 == 0:
                      # compute GAN evaluation stats
                      g_images = generate_samples(sess, gan)
                      stats = compute_evaluation_stats(g_images, test_images)
                      print {k:v for k,v in stats.iteritems()}
                      f_summary = lambda txt,v: tf.Summary(value=[tf.Summary.Value(tag=txt, simple_value=v)])
                      stats_tb = [f_summary(k,v) for k,v in stats.iteritems()]
                      # stats_summary = tf.summary.merge(stats_tb)
                      writer.add_summary(stats_tb[0], gstep)

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
                      g_images = generate_images(sess, gan)
                      stats = compute_evaluation_stats(g_images, test_images)
                      stats_tb = [tf.summary.scalar(k,v) for k,v in stats.iteritems()]
                      stats_summary = tf.summary.merge(stats_tb)
                      writer.add_summary(stats_summary, gstep)
         
                               
                except tf.errors.OutOfRangeError:
                    break

