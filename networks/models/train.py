import os
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import dcgan
from utils import save_checkpoint, load_checkpoint

def train_dcgan(data, config):
    
    #first of all, init horovod:
    hvd.init()
    
    training_graph = tf.Graph()

    with training_graph.as_default():

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
                          transpose_b=config.transpose_matmul_b)

        gan.training_graph()
        update_op = gan.optimizer(config.learning_rate, config.beta1)

        checkpoint_dir = os.path.join(config.checkpoint_dir, config.experiment)
        
        #session config
        sess_config=tf.ConfigProto(inter_op_parallelism_threads=config.num_inter_threads,
                                   intra_op_parallelism_threads=config.num_intra_threads,
                                   log_device_placement=False,
                                   allow_soft_placement=True)

        #horovod additions
        sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
        
        #stop hook
        num_batches = data.shape[0] // config.batch_size
        hooks.append(tf.train.StopAtStepHook(last_step=config.epoch*num_batches))
        
        #summary hook
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.g_summary))
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.d_summary))
        
        #variables initializer
        init_op = tf.global_variables_initializer()
        
        print("Starting Session")
        with tf.train.MonitoredTrainingSession(config=sess_config, 
                                               checkpoint_dir=checkpoint_dir if hvd.rank()==0 else None, 
                                               save_checkpoint_secs=300, 
                                               hooks=hooks) as sess:
            
            #init global variables
            sess.run(init_op,feed_dict={gan.images: data[0:config.batch_size,:,:,:]})

            load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, step=config.save_every_step)

            epoch = sess.run(gan.increment_epoch)
            start_time = time.time()
            
            #while loop with epoch counter stop hook
            while not sess.should_stop():
                
                #permute data
                perm = np.random.permutation(data.shape[0])

                #do the epoch                
                for idx in range(0, num_batches):
                    batch_images = data[perm[idx*config.batch_size:(idx+1)*config.batch_size]]

                    _, g_sum, d_sum = sess.run([update_op, gan.g_summary, gan.d_summary], 
                                               feed_dict={gan.images: batch_images})

                    global_step = sess.run(gan.global_step)
                                        
                    #verbose printing
                    if config.verbose:
                        errD_fake, errD_real, errG = sess.run([gan.d_loss_fake,gan.d_loss_real,gan.g_loss], feed_dict={gan.images: batch_images})

                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                                % (epoch, idx, num_batches, time.time() - start_time, errD_fake+errD_real, errG))

                    elif global_step%100 == 0:
                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f"%(epoch, idx, num_batches, time.time() - start_time))

                # save a checkpoint every epoch
                sess.run(gan.increment_epoch)
