import os
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import dcgan
from utils import save_checkpoint, load_checkpoint

def train_dcgan(data, config):

    training_graph = tf.Graph()

    with training_graph.as_default():
        
        #some dataset parameters
        num_batches = data.shape[0] // config.batch_size
        num_steps = config.epoch*num_batches
        
        #create dataset feeding:
        trn_placeholder = tf.placeholder(data.dtype, data.shape, name="train-data-placeholder")
        trn_dataset = tf.data.Dataset.from_tensor_slices(trn_placeholder)
        if hvd.size() > 1:
            trn_dataset = trn_dataset.shard(hvd.size(), hvd.rank())
        #trn_dataset = trn_dataset.shuffle(buffer_size=100)
        trn_dataset = trn_dataset.repeat(config.epoch)
        trn_dataset = trn_dataset.batch(config.batch_size)
        #create feedable iterator
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, trn_dataset.output_types, trn_dataset.output_shapes)
        next_element = iterator.get_next()
        #train iterator
        trn_iterator = trn_dataset.make_initializable_iterator()
        trn_handle_string = trn_iterator.string_handle()

        print("Creating GAN")
        gan = dcgan.dcgan(output_size=config.output_size,
                          batch_size=config.batch_size,
                          nd_layers=config.nd_layers,
                          ng_layers=config.ng_layers,
                          df_dim=config.df_dim,
                          gf_dim=config.gf_dim,
                          c_dim=config.c_dim,
                          z_dim=config.z_dim,
                          flip_labels=config.flip_labels,
                          data_format=config.data_format,
                          transpose_b=config.transpose_matmul_b)#,
                          #distributed=True)
        
        #create training graph
        gan.training_graph(images=next_element)
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
        
        #summary hook
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.g_summary))
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.d_summary))
        
        #checkpoint hook for fine grained checkpointing
        #save after every 10 epochs but only on node 0:
        checkpoint_dir = os.path.join(config.checkpoint_dir, config.experiment)
        if hvd.rank() == 0:
          checkpoint_save_freq = num_batches * 10
          checkpoint_saver = gan.saver
          hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))
        
        #variables initializer
        init_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        
        print("Starting Session")
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            
            #init global variables
            print("Initializing Variables")
            sess.run([init_op,init_local_op])
            
            #initialize iterator
            print("Initializing Iterator")
            trn_handle = sess.run(trn_handle_string)
            sess.run(trn_iterator.initializer, feed_dict={handle: trn_handle, trn_placeholder: data})

            #load checkpoint if necessary
            load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, step=config.save_every_step)
            
            #start epoch counter and timings:
            epoch = sess.run(gan.increment_epoch)
            start_time = time.time()
            
            #while loop with epoch counter stop hook
            while not sess.should_stop():
                
                print("Do training loop")
                
                try:
                    _, g_sum, d_sum = sess.run([update_op, gan.g_summary, gan.d_summary], feed_dict={handle: trn_handle})
                    gstep = sess.run(gan.global_step)

                    #verbose printing
                    if config.verbose:
                        errD_fake, errD_real, errG = sess.run([gan.d_loss_fake,gan.d_loss_real,gan.g_loss], feed_dict={handle: trn_handle})

                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                            % (epoch, gstep, num_steps, time.time() - start_time, errD_fake+errD_real, errG))

                    elif gstep%100 == 0:
                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f"%(epoch, gstep, num_batches, time.time() - start_time))

                    # increment epoch counter
                    if gstep%num_batches == 0:
                        epoch = sess.run(gan.increment_epoch)
                        
                except tf.errors.OutOfRangeError:
                    break
