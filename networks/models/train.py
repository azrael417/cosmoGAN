import os
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
    
import models.dcgan
from models.cramer_dcgan import cramer_dcgan
from models.ot_gan import ot_gan
from models.utils import save_checkpoint, load_checkpoint
from tensorflow.python import debug as tf_debug
from models.distributed_sinkhorn import distributed_sinkhorn, reduce_loss, allreduce, allgather
from utils import get_data
from validation import *

#generate samples
def generate_samples(sess, gan, n_batches=20):
    z_sample = np.random.normal(size=(gan.batch_size, gan.z_dim))
    samples = sess.run(gan.xg, feed_dict={gan.z: z_sample})

    for i in range(0, n_batches-1):
        z_sample = np.random.normal(size=(gan.batch_size, gan.z_dim))
        samples = np.concatenate((samples, sess.run(gan.xg, feed_dict={gan.z: z_sample})))
        
    return np.squeeze(samples)


#OT GAN
def train_otgan(comm_topo, data_tuple, config):
    
    #disassemble the tuple
    trn_data = data_tuple[0]
    trn_min = data_tuple[1]
    trn_max = data_tuple[2]
    
    training_graph = tf.Graph()

    with training_graph.as_default():
        
        if comm_topo.comm_rank == 0:
            print("Creating GAN")
        gan = ot_gan(comm_topo=comm_topo,
                           output_size=config.output_size,
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
        
        #checkpoint directory
        checkpoint_dir = os.path.join(config.checkpoint_dir, config.experiment)

        #session config
        sess_config=tf.ConfigProto(inter_op_parallelism_threads=config.num_inter_threads,
                                   intra_op_parallelism_threads=config.num_intra_threads,
                                   log_device_placement=False,
                                   allow_soft_placement=True)
        
        # load test data
        test_images, _, _ = get_data(config.test_datafile, config.data_format)
        #preprocess for rescaling to range
        test_images = 2. * (test_images - trn_min) / (trn_max - trn_min) - 1.

        # prepare plots dir
        plots_dir = os.path.join(config.plots_dir, config.experiment)
        if not os.path.exists(config.plots_dir):
            try:
                os.makedirs(config.plots_dir)
            except:
                print("Rank {}: path {} does already exist.".format(hvd.rank(),config.plots_dir))

        #horovod additions
        sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
        comm_size = hvd.size()
        comm_rank = hvd.rank()
        
        #hooks
        hooks = []
        #do not use the bcast hook but instead bcast manually
        #hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        
        #stop hook
        num_batches = trn_data.shape[0] // (2*gan.global_batch_size)
        num_steps = config.epoch*num_batches
        hooks.append(tf.train.StopAtStepHook(last_step=num_steps))
        
        #summary hook
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.g_summary))
        #hooks.append(tf.train.SummarySaverHook(save_steps=num_batches,output_dir='./logs/'+config.experiment+'/train'+str(hvd.rank()),summary_op=gan.d_summary))
        
        #checkpoint hook for fine grained checkpointing
        #save after every 10 epochs but only on node 0:
        if comm_rank == 0:
            print("Setting up checkpointing")
            checkpoint_save_freq = num_batches * 1
            checkpoint_saver = gan.saver
            hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))
        
        #variables initializer
        init_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        init_restore = hvd.broadcast_global_variables(0)
        
        #some math to what fraction this node gets
        local_row_start = gan.comm_topo.comm_row_rank*gan.comm_topo.local_row_size
        local_row_end = (gan.comm_topo.comm_row_rank+1)*gan.comm_topo.local_row_size
        local_col_start = gan.comm_topo.comm_col_rank*gan.comm_topo.local_col_size
        local_col_end = (gan.comm_topo.comm_col_rank+1)*gan.comm_topo.local_col_size
        
        if gan.comm_topo.comm_rank == 0:
            print("Starting Session")
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            
            #wrap to CLI
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            
            #init global variables
            sess.run([init_op, init_local_op])

            #restore from cp
            if hvd.rank() == 0:
                load_checkpoint(sess, gan.saver, checkpoint_dir, step=config.save_every_step)
            #broadcast
            sess.run(init_restore)
            
            #init counters
            epoch = sess.run(gan.increment_epoch)
            start_time = time.time()
            
            #make sure all nodes have the right seed
            trn_shuffle_rng = np.random.RandomState(54321)
            
            #while loop with epoch counter stop hook
            while not sess.should_stop():
            #while True:
                
                #permute data
                perm = trn_shuffle_rng.permutation(trn_data.shape[0])
                
                #do the epoch
                gstep = 0
                idx = 0
                while idx < num_batches:
                    
                    #get node-local fraction of new batches
                    rstart = local_row_start+(idx*gan.global_batch_size)
                    rend = local_row_end+(idx*gan.global_batch_size)
                    cstart = local_col_start+((idx+1)*gan.global_batch_size)
                    cend = local_col_end+((idx+1)*gan.global_batch_size)
                    
                    #now grab the node-local fraction of the new batch
                    xr = 2. * (trn_data[perm[rstart:rend],:] - trn_min) / (trn_max - trn_min) - 1.
                    xrp = 2. * (trn_data[perm[cstart:cend],:] - trn_min) / (trn_max - trn_min) - 1.
                    #do the same with random vectors:
                    z = gan.generate_prior()[local_row_start:local_row_end,:]
                    zp = gan.generate_prior()[local_col_start:local_col_end,:]
                    
                    
                    ##DEBUG
                    #print("Rank {rnk}: rows=({rlow},{rhigh}), cols=({clow},{chigh})".format(rnk=hvd.rank(), rlow=rstart, rhigh=rend, clow=cstart, chigh=cend))
                    ##do xr
                    #xr_exact = 2. * (trn_data[perm[idx*gan.global_batch_size:(idx+1)*gan.global_batch_size],:] - trn_min) / (trn_max - trn_min) - 1.
                    #xr_gather = allgather(gan.comm_topo, xr, "col")
                    #print("Rank {rnk}: |xr_e| = {nrm}, |xr_e-xr_g| = {nrmdiff}".format(rnk=hvd.rank(), nrm=np.sum(np.abs(xr_exact)), nrmdiff=np.sum(np.abs(xr_exact-xr_gather))))
                    ##do xrp
                    #xrp_exact = 2. * (trn_data[perm[(idx+1)*gan.global_batch_size:(idx+2)*gan.global_batch_size],:] - trn_min) / (trn_max - trn_min) - 1.
                    #xrp_gather = allgather(gan.comm_topo, xrp, "row")
                    #print("Rank {rnk}: |xrp_e| = {nrm}, |xrp_e-xrp_g| = {nrmdiff}".format(rnk=hvd.rank(), nrm=np.sum(np.abs(xrp_exact)), nrmdiff=np.sum(np.abs(xrp_exact-xrp_gather))))
                    ##artificial samples
                    ##do z
                    #z_exact = gan.generate_prior()
                    #z = z_exact[local_row_start:local_row_end,:]
                    #z_gather = allgather(gan.comm_topo, z, "col")
                    #print("Rank {rnk}: |z_e| = {nrm}, |z_e-z_g| = {nrmdiff}".format(rnk=hvd.rank(), nrm=np.sum(np.abs(z_exact)), nrmdiff=np.sum(np.abs(z_exact-z_gather))))
                    ##do zp
                    #zp_exact = gan.generate_prior()
                    #zp = zp_exact[local_col_start:local_col_end,:]
                    #zp_gather = allgather(gan.comm_topo, zp, "row")
                    #print("Rank {rnk}: |zp_e| = {nrm}, |zp_e-zp_g| = {nrmdiff}".format(rnk=hvd.rank(), nrm=np.sum(np.abs(zp_exact)), nrmdiff=np.sum(np.abs(zp_exact-zp_gather))))
                    ##DEBUG


                    #generate the distance matrices:
                    c_hxr_hxg, c_hxr_hxgp, c_hxrp_hxg, c_hxrp_hxgp, c_hxr_hxrp, c_hxg_hxgp = sess.run([gan.c_hxr_hxg, 
                                                                                                        gan.c_hxr_hxgp, 
                                                                                                        gan.c_hxrp_hxg,
                                                                                                        gan.c_hxrp_hxgp,
                                                                                                        gan.c_hxr_hxrp,
                                                                                                        gan.c_hxg_hxgp], 
                                                                                                        feed_dict={
                                                                                                            gan.xr: xr,
                                                                                                            gan.xrp: xrp,
                                                                                                            gan.z: z,
                                                                                                            gan.zp: zp
                                                                                                        })
                    
                    #compute distance matrices using sinkhorn:
                    lambd = 30.; tolerance=1.e-6; min_iters=20; max_iters=5000
                    
                    if gan.comm_topo.comm_rank == 0:
                        print("Starting Sinkhorn")
                    map_hxr_hxg = distributed_sinkhorn(gan.comm_topo, c_hxr_hxg, lambd, tolerance, min_iters, max_iters, verbose=False)
                    map_hxr_hxgp = distributed_sinkhorn(gan.comm_topo, c_hxr_hxgp, lambd, tolerance, min_iters, max_iters, verbose=False)
                    map_hxrp_hxg = distributed_sinkhorn(gan.comm_topo, c_hxrp_hxg, lambd, tolerance, min_iters, max_iters, verbose=False)
                    map_hxrp_hxgp = distributed_sinkhorn(gan.comm_topo, c_hxrp_hxgp, lambd, tolerance, min_iters, max_iters, verbose=False)
                    map_hxr_hxrp = distributed_sinkhorn(gan.comm_topo, c_hxr_hxrp, lambd, tolerance, min_iters, max_iters, verbose=False)
                    map_hxg_hxgp = distributed_sinkhorn(gan.comm_topo, c_hxg_hxgp, lambd, tolerance, min_iters, max_iters, verbose=False)
                    if gan.comm_topo.comm_rank == 0:
                        print("Ending Sinkhorn")
                    
                    feed_dict = {gan.xr: xr, gan.xrp: xrp, gan.z: z, gan.zp: zp,
                                gan.map_hxr_hxg: map_hxr_hxg,
                                gan.map_hxr_hxgp: map_hxr_hxgp,
                                gan.map_hxrp_hxg: map_hxrp_hxg,
                                gan.map_hxrp_hxgp: map_hxrp_hxgp,
                                gan.map_hxr_hxrp: map_hxr_hxrp,
                                gan.map_hxg_hxgp: map_hxg_hxgp}
                    
                    if gstep%config.n_up==0:
                        #do combined update
                        _, g_sum, d_sum = sess.run([update_op, gan.g_summary, gan.d_summary], feed_dict=feed_dict)
                    else:
                        #update generator
                        _, g_sum = sess.run([g_update_op, gan.g_summary], feed_dict=feed_dict)
                    
                    #increase and get step count
                    gstep = sess.run(gan.global_step)
                    
                    #print some stats
                    if config.verbose:
                        loss = reduce_loss(gan.comm_topo, sess.run(gan.ot_loss, feed_dict=feed_dict))
                        if gan.comm_topo.comm_rank == 0:
                            print("Rank %2d, Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, loss: %.8f" \
                                    % (comm_rank, epoch, gstep, num_steps, time.time() - start_time, loss))
                    elif gstep%num_batches == 0:
                        if gan.comm_topo.comm_rank == 0:
                            print("Rank %2d, Epoch: [%2d] Step: [%4d/%4d] time: %4.4f"%(comm_rank, epoch, gstep, num_steps, time.time() - start_time))
                    
                    # increment epoch counter
                    if gstep%config.print_frequency == 0:
                        g_images = generate_samples(sess, gan)
                        if hvd.rank() == 0:
                            print("Starting computing pixel statistics")
                            pixel_histogram_deviation(g_images, test_images, dump_path=plots_dir, tag="step%d_epoch%d" % (gstep, epoch))
                            #plot_pixel_histograms(g_images, test_images, dump_path=plots_dir, tag="step%d_epoch%d" % (gstep, epoch))
                            print("Done computing pixel statistics")
                            #dump_samples(g_images, dump_path="%s/step%d_epoch%d" % (plots_dir, gstep, gstep/num_steps_per_rank_per_epoch), tag="synthetic")
                    
                    #update batch counter
                    idx+=2
                    
                # save a checkpoint every epoch
                epoch = sess.run(gan.increment_epoch)
