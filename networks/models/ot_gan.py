import tensorflow as tf
import horovod.tensorflow as hvd
    
from .ops import linear, conv2d, conv2d_transpose, lrelu, compute_cost, ot_distance
import numpy as np


class ot_gan(object):
    def __init__(self, comm_topo, output_size=64, batch_size=64, 
                 nd_layers=4, ng_layers=4, df_dim=128, gf_dim=128, 
                 c_dim=1, z_dim=100, d_out_dim=256, gradient_lambda=10., 
                 data_format="NHWC",
                 gen_prior="normal", solver="ADAM", transpose_b=False, spectral_weight_normalization=True):

        self.comm_topo = comm_topo
        self.output_size = output_size
        self.batch_size = batch_size
        self.global_batch_size = comm_topo.comm_row_size * self.batch_size
        self.nd_layers = nd_layers
        self.ng_layers = ng_layers
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.d_out_dim = d_out_dim
        self.gradient_lambda = gradient_lambda
        self.data_format = data_format
        self.gen_prior = gen_prior
        self.rng = np.random.RandomState(1234567)
        self.transpose_b = transpose_b # transpose weight matrix in linear layers for (possible) better performance when running on HSW/KNL
        self.stride = 2 # this is fixed for this architecture
        self.solver = solver
        self.spectral_weight_normalization = spectral_weight_normalization

        self._check_architecture_consistency()
        
        self.batchnorm_kwargs = {'epsilon' : 1e-5, 'decay': 0.9, 
                                 'updates_collections': None, 'scale': True,
                                 'fused': True, 'data_format': self.data_format}
                                 
    def training_graph(self):
        
        #real input images
        if self.data_format == "NHWC":
            self.xr = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images-1')
            self.xrp = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images-2')
        else:
            self.xr = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images-1')
            self.xrp = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images-2')
            
        #register the mapping matrices as placeholders
        self.map_hxr_hxg = tf.placeholder(tf.float32, [self.batch_size, self.batch_size], name="map_hxr_hxg")
        self.map_hxr_hxgp = tf.placeholder(tf.float32, [self.batch_size, self.batch_size], name="map_hxr_hxgp")
        self.map_hxrp_hxg = tf.placeholder(tf.float32, [self.batch_size, self.batch_size], name="map_hxrp_hxg")
        self.map_hxrp_hxgp = tf.placeholder(tf.float32, [self.batch_size, self.batch_size], name="map_hxrp_hxgp")
        self.map_hxr_hxrp = tf.placeholder(tf.float32, [self.batch_size, self.batch_size], name="map_hxr_hxrp")
        self.map_hxg_hxgp = tf.placeholder(tf.float32, [self.batch_size, self.batch_size], name="map_hxg_hxgp")
        
        #seeds for fake images. Make these placeholders at well because we need to store the random vectors:
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z_prior')
        self.zp = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='zp_prior')
        
        # discriminator
        #with tf.variable_scope("discriminator") as d_scope:
        h = self.discriminator
        
        # pipe through discriminator
        #with tf.variable_scope("generator") as g_scope:
        self.xg = self.generator(self.z, is_training=True)
        self.xgp = self.generator(self.zp, is_training=True)
        
        #apply disc
        self.hxr = h(self.xr, True)
        self.hxrp = h(self.xrp, True)
        self.hxg = h(self.xg, True)
        self.hxgp = h(self.xgp, True)
        
        #generate cost matrices
        self.c_hxr_hxg = compute_cost(self.hxr, self.hxg)
        self.c_hxr_hxgp = compute_cost(self.hxr, self.hxgp)
        self.c_hxrp_hxg = compute_cost(self.hxrp, self.hxg)
        self.c_hxrp_hxgp = compute_cost(self.hxrp, self.hxgp)
        self.c_hxr_hxrp = compute_cost(self.hxr, self.hxrp)
        self.c_hxg_hxgp = compute_cost(self.hxg, self.hxgp)
        
        #compute the individual losses
        w_xr_xg = ot_distance(self.c_hxr_hxg, self.map_hxr_hxg)
        w_xr_xgp = ot_distance(self.c_hxr_hxgp, self.map_hxr_hxgp)
        w_xrp_xg = ot_distance(self.c_hxrp_hxg, self.map_hxrp_hxg)
        w_xrp_xgp = ot_distance(self.c_hxrp_hxgp, self.map_hxrp_hxgp)
        w_xr_xrp = ot_distance(self.c_hxr_hxrp, self.map_hxr_hxrp)
        w_xg_xgp = ot_distance(self.c_hxg_hxgp, self.map_hxg_hxgp)
        
        #defining loss
        with tf.name_scope("loss"):
            self.ot_loss = w_xr_xg + w_xr_xgp + w_xrp_xg + w_xrp_xgp - 2.* ( w_xr_xrp + w_xg_xgp )
            #critic gets - sign
            self.d_loss = - self.ot_loss
            #generator gets + sign
            self.g_loss = self.ot_loss

        self.d_summary = tf.summary.merge([tf.summary.histogram("loss/L_critic", self.d_loss)])

        g_sum = [tf.summary.histogram("loss/L_generator", self.g_loss)]

        if self.data_format == "NHWC": # tf.summary.image is not implemented for NCHW
            g_sum.append(tf.summary.image("G", self.xg, max_outputs=4))
        self.g_summary = tf.summary.merge(g_sum)

        t_vars = tf.trainable_variables()
        
        #create variables
        self.d_vars = [var for var in t_vars if 'discriminator/' in var.name]
        self.g_vars = [var for var in t_vars if 'generator/' in var.name]
        
        #create dictionary for holding the normalizations
        self.d_vars_norm = { var.name: 1. for var in self.d_vars}
        self.g_vars_norm = { var.name: 1. for var in self.g_vars}

        with tf.variable_scope("counters") as counters_scope:
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            self.global_step = tf.train.get_or_create_global_step()
            self.increment_step = tf.assign(self.global_step, self.global_step+1)
        
        self.saver = tf.train.Saver(max_to_keep=8000)

    def inference_graph(self):

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')

        if self.z is not None:
            self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z_prior')
        
        #with tf.variable_scope("discriminator") as d_scope:
        self.D = self.discriminator(self.images, is_training=False)
            
        #with tf.variable_scope("generator") as g_scope:
        self.G = self.generator(self.z, is_training=False)

        with tf.variable_scope("counters") as counters_scope:
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            #do not put that into counters scope
            self.global_step = tf.train.get_or_create_global_step()

        self.saver = tf.train.Saver(max_to_keep=8000)


    def optimizer(self, learning_rate, beta1, clip_param):
        #critic
        d_optim = None
        if self.solver == "ADAM":
            d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        elif self.solver == "RMSProp":
            d_optim = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise ValueError("Solver type {solver} not supported.".format(solver=self.solver))
        d_optim = hvd.DistributedOptimizer(d_optim)
        #d_op = d_optim.minimize(self.d_loss, var_list=self.d_vars)
        
        #normalize weights
        if self.spectral_weight_normalization:
          assign_op = self.normalize_weights(self.d_vars_norm, self.d_vars)
          #compute gradients
          with tf.control_dependencies(assign_op):
            #compute the gradients with respect to the normalized variables
            d_grads_and_vars = d_optim.compute_gradients(self.d_loss, var_list=self.d_vars)
            #create assign op for the weights
            assign_op = [vn.assign(vn*self.d_vars_norm[vn.name]) for (_, vn) in d_grads_and_vars]
        else:
          d_grads_and_vars = d_optim.compute_gradients(self.d_loss, var_list=self.d_vars)
          assign_op = []
        
        #make sure that grads and wars have been computed
        with tf.control_dependencies(assign_op):
          d_op = d_optim.apply_gradients(d_grads_and_vars)
        

        #generator
        g_optim = None
        if self.solver == "ADAM":
            g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        elif self.solver == "RMSProp":
            g_optim = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise ValueError("Solver type {solver} not supported.".format(solver=self.solver))
        g_optim = hvd.DistributedOptimizer(g_optim)
        #g_op = g_optim.minimize(self.g_loss, global_step=self.global_step, var_list=self.g_vars)

        #normalize weights
        if self.spectral_weight_normalization:
          assign_op = self.normalize_weights(self.g_vars_norm, self.g_vars)
          #compute gradients
          with tf.control_dependencies(assign_op):
            #compute the gradients with respect to the normalized variables
            g_grads_and_vars = g_optim.compute_gradients(self.g_loss, var_list=self.g_vars)
            #create assign op for the weights
            assign_op = [vn.assign(vn*self.g_vars_norm[vn.name]) for (_, vn) in g_grads_and_vars]
        else:
          g_grads_and_vars = g_optim.compute_gradients(self.g_loss, var_list=self.g_vars)
          assign_op = []
        
        #make sure that grads and wars have been computed
        with tf.control_dependencies(assign_op):
          g_op = g_optim.apply_gradients(g_grads_and_vars, global_step=self.global_step)

        return d_op, g_op


    def normalize_weights(self, weight_normalization, weightlist):
      #normalize the weights
      assign_op = []
      for idx,w in enumerate(weightlist):
        if w is not None:
          sval = tf.cond(pred = tf.greater(tf.rank(w), 1), 
                          true_fn = lambda: self.get_largest_sv(w),
                          false_fn = lambda: 1.)
          weight_normalization[w.name] = sval
          assign_op.append(weightlist[idx].assign(w/sval))
      return assign_op
    
    
    def get_largest_sv(self, weight):
      shape = weight.get_shape().as_list()
      newshape = (shape[0], int(np.prod(shape[1:])))
      wtilde = tf.reshape(weight, newshape)
      #avoid backpropping through the SVD
      sval = tf.svd(wtilde, compute_uv=False)[0]
      return sval


    #generator helper function
    def generator(self, z, is_training):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as g_scope:

            map_size = self.output_size/int(2**self.ng_layers)
            num_filters = self.gf_dim * int(2**(self.ng_layers -1))

            # h0 = relu(reshape(FC(z)))
            z_ = linear(z, num_filters*map_size*map_size, 'h0_lin', transpose_b=self.transpose_b)
            h0 = tf.reshape(z_, self._tensor_data_format(-1, map_size, map_size, num_filters))
            #bn0 = tf.contrib.layers.batch_norm(h0, is_training=is_training, scope='bn0', **self.batchnorm_kwargs)
            h0 = tf.nn.relu(h0)

            chain = h0
            for h in range(1, self.ng_layers):
                # h1 = relu(conv2d_transpose(h0))
                map_size *= self.stride
                num_filters /= 2
                chain = conv2d_transpose(chain,
                                         self._tensor_data_format(self.batch_size, map_size, map_size, num_filters),
                                         stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%h)
                #chain = tf.contrib.layers.batch_norm(chain, is_training=is_training, scope='bn%i'%h, **self.batchnorm_kwargs)
                chain = tf.nn.relu(chain)

            # h1 = conv2d_transpose(h0)
            map_size *= self.stride
            hn = conv2d_transpose(chain,
                                  self._tensor_data_format(self.batch_size, map_size, map_size, self.c_dim),
                                  stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%(self.ng_layers))

            return tf.nn.tanh(hn)


    def discriminator(self, image, is_training):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as d_scope:

            chain = image
            for h in range(1, self.nd_layers):
                # h1 = lrelu(conv2d(h0))
                num_filters = self.df_dim if h==0 else self.df_dim*2**h
                chain = conv2d(chain, num_filters, self.data_format, name='h%i_conv'%h)
                #chain = tf.contrib.layers.batch_norm(chain, is_training=is_training, scope='bn%i'%h, **self.batchnorm_kwargs)
                chain = lrelu(chain)

            # h1 = linear(reshape(h0))
            hn = linear(tf.reshape(chain, [self.batch_size, -1]), self.d_out_dim, 'h%i_lin'%self.nd_layers, transpose_b=self.transpose_b)
            
            #l2 normalization: important for rescaling between -1,1
            hn = tf.nn.l2_normalize(hn, axis=1, epsilon=1e-12, name="l2-output-normalization")
            
            return hn


    def generate_prior(self, shape=None):
        if self.gen_prior == "normal":
            if not shape:
                return self.rng.normal(size=(self.global_batch_size, self.z_dim)).astype(np.float32)
            else:
                return self.rng.normal(size=shape).astype(np.float32)
        elif self.gen_prior == "uniform":
            if not shape:
                return self.rng.uniform(low=-1., high=1., size=(self.global_batch_size, self.z_dim)).astype(np.float32)
            else:
                return self.rng.uniform(low=-1., high=1., size=shape).astype(np.float32)
        else:
            raise ValueError("Error, only normal distributed random numbers are supported at the moment.")


    def _tensor_data_format(self, N, H, W, C):
        if self.data_format == "NHWC":
            return [int(N), int(H), int(W), int(C)]
        else:
            return [int(N), int(C), int(H), int(W)]

    def _check_architecture_consistency(self):

        if self.output_size/2**self.nd_layers < 1:
            print("Error: Number of discriminator conv. layers are larger than the output_size for this architecture")
            exit(0)

        if self.output_size/2**self.ng_layers < 1:
            print("Error: Number of generator conv_transpose layers are larger than the output_size for this architecture")
            exit(0)
