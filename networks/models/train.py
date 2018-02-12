import os
import time
import numpy as np
import tensorflow as tf
use_horovod = True
try:
    import horovod.tensorflow as hvd
except:
    use_horovod = False
import dcgan
from cramer_dcgan import cramer_dcgan
from utils import save_checkpoint, load_checkpoint
from tensorflow.python import debug as tf_debug

def train_dcgan(data, config):

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
        hooks = []
        comm_size = 1
        comm_rank = 0
        if use_horovod:
            sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
            hooks.append(hvd.BroadcastGlobalVariablesHook(0))
            comm_size = hvd.size()
            comm_rank = hvd.rank()
        
        #stop hook
        num_batches = data.shape[0] // config.batch_size
        hooks.append(tf.train.StopAtStepHook(last_step=config.epoch*num_batches))
        
        #summary hook
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.g_summary))
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.d_summary))
        
        #checkpoint hook for fine grained checkpointing
        #save after every epoch but only on node 0:
        if comm_rank == 0:
            checkpoint_save_freq = num_batches * 1
            checkpoint_saver = tf.train.Saver(max_to_keep = 1000)
            hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))
        
        #variables initializer
        init_op = tf.global_variables_initializer()
        
        print("Starting Session")
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            
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

                    _, g_sum, d_sum = sess.run([update_op, gan.g_summary, gan.d_summary], feed_dict={gan.images: batch_images})
                    global_step = sess.run(gan.global_step)
                                        
                    #verbose printing
                    if config.verbose:
                        errD_fake, errD_real, errG = sess.run([gan.d_loss_fake,gan.d_loss_real,gan.g_loss], feed_dict={gan.images: batch_images})

                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                                % (epoch, idx, num_batches, time.time() - start_time, errD_fake+errD_real, errG))

                    elif global_step%num_batches == 0:
                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f"%(epoch, idx, num_batches, time.time() - start_time))

                # save a checkpoint every epoch
                epoch = sess.run(gan.increment_epoch)


#Cramer GAN
def train_cramer_dcgan(data, config):

    training_graph = tf.Graph()

    with training_graph.as_default():
        
        print("Creating GAN")
        gan = cramer_dcgan(output_size=config.output_size,
                           batch_size=config.batch_size,
                           nd_layers=config.nd_layers,
                           ng_layers=config.ng_layers,
                           df_dim=config.df_dim,
                           gf_dim=config.gf_dim,
                           c_dim=config.c_dim,
                           z_dim=config.z_dim,
                           d_out_dim=config.d_out_dim,
                           gradient_lambda=config.gradient_lambda,
                           data_format=config.data_format,
                           transpose_b=config.transpose_matmul_b)

        gan.training_graph()
        d_update_op, g_update_op = gan.optimizer(config.learning_rate, config.beta1, clip_param=0.01)
        update_op = tf.group(d_update_op, g_update_op, name="all_optims")

        checkpoint_dir = os.path.join(config.checkpoint_dir, config.experiment)

        #session config
        sess_config=tf.ConfigProto(inter_op_parallelism_threads=config.num_inter_threads,
                                   intra_op_parallelism_threads=config.num_intra_threads,
                                   log_device_placement=False,
                                   allow_soft_placement=True)

        #horovod additions
        hooks = []
        comm_size = 1
        comm_rank = 0
        if use_horovod:
            sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
            hooks.append(hvd.BroadcastGlobalVariablesHook(0))
            comm_size = hvd.size()
            comm_rank = hvd.rank()
        
        #stop hook
        num_batches = data.shape[0] // config.batch_size
        hooks.append(tf.train.StopAtStepHook(last_step=config.epoch*num_batches))
        
        #summary hook
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.g_summary))
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.d_summary))
        
        #checkpoint hook for fine grained checkpointing
        #save after every 10 epochs but only on node 0:
        if comm_rank == 0:
            print("Setting up checkpointing")
            checkpoint_save_freq = num_batches * 1
            checkpoint_saver = tf.train.Saver(max_to_keep = 1000)
            hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))
        
        #variables initializer
        init_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        
        print("Starting Session")
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
        #with tf.Session(config=sess_config) as sess:
            
            #wrap to CLI
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            
            #init global variables
            sess.run([init_op, init_local_op], feed_dict={gan.images: data[0:config.batch_size,:,:,:]})

            #load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, step=config.save_every_step)

            epoch = sess.run(gan.increment_epoch)
            start_time = time.time()
            
            #while loop with epoch counter stop hook
            while not sess.should_stop():
            #while True:
                
                #permute data
                perm = np.random.permutation(data.shape[0])
                
                #do the epoch
                gstep = 0
                for idx in range(0, num_batches):
                    
                    #get new batch
                    batch_images = data[perm[idx*config.batch_size:(idx+1)*config.batch_size]]

                    if gstep%config.n_up==0:
                        #update generator
                        _, g_sum = sess.run([g_update_op, gan.g_summary], feed_dict={gan.images: batch_images})
                    else:
                        #update critic
                        _, d_sum = sess.run([d_update_op, gan.d_summary], feed_dict={gan.images: batch_images})
                    
                    #increase and get step count
                    gstep = sess.run(gan.global_step)
                    
                    #print some stats
                    if config.verbose:
                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, c_loss: %.8f, s_loss: %.8f, g_loss: %.8f" \
                                % (epoch, gstep, num_batches, time.time() - start_time, L_critic, L_surrogate, L_generator))
                    elif gstep%num_batches == 0:
                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f"%(epoch, gstep, num_batches, time.time() - start_time))

                # save a checkpoint every epoch
                epoch = sess.run(gan.increment_epoch)
