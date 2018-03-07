import os
import time
import numpy as np
import tensorflow as tf
import wgan_dcgan as dcgan
import horovod.tensorflow as hvd
from utils import save_checkpoint, load_checkpoint

def train_dcgan(datafiles, config):
    num_batches = config.num_records_total // config.batch_size
    num_steps = config.epoch*num_batches

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
            example = tf.reshape(example,[config.output_size, config.output_size])
            return example
             
        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(lambda x: decode_record(x))  # Parse the record into tensors.
        dataset = dataset.repeat(config.epoch)  # Repeat the input indefinitely.
        dataset = dataset.batch(config.batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

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
          checkpoint_save_freq = num_batches * 10
          checkpoint_saver = gan.saver
          hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))


       #variables initializer
        init_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
 
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            writer = tf.summary.FileWriter('./logs/'+config.experiment+'/train', sess.graph)

            sess.run([init_op, init_local_op])

            load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, step=config.save_every_step)

            epoch = sess.run(gan.increment_epoch)
            start_time = time.time()

            while not sess.should_stop():           
                try:
                    _, g_sum, c_sum = sess.run([update_op, gan.g_summary, gan.c_summary], feed_dict={handle: trn_handle})
                    gstep = sess.run(gan.global_step)

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
                        
                except tf.errors.OutOfRangeError:
                    break

                # save a checkpoint every epoch
                save_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, epoch)
                sess.run(gan.increment_epoch)
