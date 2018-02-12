import tensorflow as tf
use_horovod=True
try:
    import horovod.tensorflow as hvd
except:
    use_horovod=False
    
from .ops import linear, conv2d, conv2d_transpose, lrelu, debug

class cramer_dcgan(object):
    def __init__(self, output_size=64, batch_size=64, 
                 nd_layers=4, ng_layers=4, df_dim=128, gf_dim=128, 
                 c_dim=1, z_dim=100, d_out_dim=256, gradient_lambda=10., 
                 data_format="NHWC",
                 gen_prior=tf.random_normal, transpose_b=False):

        self.output_size = output_size
        self.batch_size = batch_size
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
        self.transpose_b = transpose_b # transpose weight matrix in linear layers for (possible) better performance when running on HSW/KNL
        self.stride = 2 # this is fixed for this architecture

        self._check_architecture_consistency()
        
        self.batchnorm_kwargs = {'epsilon' : 1e-5, 'decay': 0.9, 
                                 'updates_collections': None, 'scale': True,
                                 'fused': True, 'data_format': self.data_format}
    
    def gendist(self, x0, x1):
        h = self.discriminator
        #apply disc
        hx0 = h(x0)
        hx0 = debug(hx0,[x0,hx0],name="hx0_debug")
        hx1 = h(x1)
        hx1 = debug(hx1,[x1,hx1],name="hx1_debug")
        
        return tf.norm(hx0 - hx1, ord=2, axis=1)

    def critic(self, x, xgp):
        h = self.discriminator
        return self.gendist(x, xgp) - tf.norm(h(x), ord=2, axis=1)

    def training_graph(self):

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')

        x = self.images
        self.z = self.gen_prior(shape=[self.batch_size, self.z_dim], dtype=tf.float32)
        self.zp = self.gen_prior(shape=[self.batch_size, self.z_dim], dtype=tf.float32)
        xg = self.generator(self.z, is_training=True)
        xgp = self.generator(self.zp, is_training=True)

        with tf.name_scope("losses"):
            with tf.name_scope("L_generator"):
                self.L_generator = tf.reduce_mean(self.gendist(x,xg) + self.gendist(x,xgp) - self.gendist(xg,xgp))
                self.L_generator = debug(self.L_generator, [x,xg,xgp,self.gendist(x,xg),self.gendist(x,xgp),self.gendist(xg,xgp)], name="L_g_debug")
            with tf.name_scope("L_surrogate"):
                self.L_surrogate = tf.reduce_mean(self.critic(x, xgp) - self.critic(xg, xgp))
                self.L_surrogate = debug(self.L_surrogate, [self.critic(x, xgp),self.critic(xg, xgp)], name="L_s_debug")
            with tf.name_scope("L_critic"):
                epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1., dtype=tf.float32)
                x_hat = xg + epsilon * (x - xg)
                f_x_hat = self.critic(x_hat, xgp)
                f_x_hat_gradient = tf.gradients(f_x_hat, x_hat)[0]
                gradient_penalty = tf.reduce_mean( self.gradient_lambda * tf.square(tf.norm(f_x_hat_gradient, ord=2, axis=1) - 1.) )
                self.L_critic = -self.L_surrogate + gradient_penalty
                self.L_critic = debug(self.L_critic, [x_hat, f_x_hat, f_x_hat_gradient, gradient_penalty], name="L_c_debug")

        self.d_summary = tf.summary.merge([tf.summary.histogram("loss/L_critic", self.L_critic),
                                           tf.summary.histogram("loss/gradient_penalty", gradient_penalty)])

        g_sum = [tf.summary.histogram("loss/L_generator", self.L_generator), tf.summary.scalar("loss/L_surrogate", self.L_surrogate)]

        if self.data_format == "NHWC": # tf.summary.image is not implemented for NCHW
            g_sum.append(tf.summary.image("G", xg, max_outputs=4))
        self.g_summary = tf.summary.merge(g_sum)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator/' in var.name]
        self.g_vars = [var for var in t_vars if 'generator/' in var.name]

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

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.discriminator(self.images)
        self.G = self.generator(self.z, is_training=False)

        with tf.variable_scope("counters") as counters_scope:
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
        #do not put that into counters scope
        self.global_step = tf.train.get_or_create_global_step()

        self.saver = tf.train.Saver(max_to_keep=8000)

    def clip_critic_weights(self, clip_param=0.01):
        clip_op = [tf.assign(w, tf.clip_by_value(w, -clip_param, clip_param, name="critic_weight_clip")) for w in self.d_vars]
        return clip_op

    def optimizer(self, learning_rate, beta1, clip_param):
        #critic
        d_optim = tf.train.RMSPropOptimizer(learning_rate) #, beta1=beta1)
        #compute grads
        #d_grads_and_vars = d_optim.compute_gradients(self.L_critic)
        if use_horovod:
            #allreduce
            #d_grads_and_vars = [(hvd.allreduce(g[0]),g[1]) for g in d_grads_and_vars]
            d_optim = hvd.DistributedOptimizer(d_optim)
        d_op_apply = d_optim.minimize(self.L_critic, global_step=self.global_step, var_list=self.d_vars)
        #clip grads
        #d_grads_and_vars = [(tf.clip_by_value(g[0],-1,1),g[1]) for g in d_grads_and_vars]
        #apply
        #d_op_apply = d_optim.apply_gradients(d_grads_and_vars, global_step=self.global_step)
        #weight clipping
        with tf.control_dependencies([d_op_apply]):
            clip_op = self.clip_critic_weights(clip_param=clip_param)
        #combine
        d_op = tf.group(d_op_apply, *clip_op)

        #generator
        g_optim = tf.train.RMSPropOptimizer(learning_rate) #, beta1=beta1)
        #compute grads
        #g_grads_and_vars = g_optim.compute_gradients(self.L_surrogate)
        if use_horovod:
            #allreduce
            #g_grads_and_vars = [(hvd.allreduce(g[0]),g[1]) for g in g_grads_and_vars]
            g_optim = hvd.DistributedOptimizer(g_optim)
        g_op = g_optim.minimize(self.L_surrogate, global_step=self.global_step, var_list=self.g_vars)
        #clip grads
        #g_grads_and_vars = [(tf.clip_by_value(g[0],-1,1),g[1]) for g in g_grads_and_vars]
        #apply
        #g_op = g_optim.apply_gradients(g_grads_and_vars)

        return d_op, g_op

                                                   
    def generator(self, z, is_training):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as g_scope:

            map_size = self.output_size/int(2**self.ng_layers)
            num_filters = self.gf_dim * int(2**(self.ng_layers -1))

            # h0 = relu(reshape(FC(z)))
            z_ = linear(z, num_filters*map_size*map_size, 'h0_lin', transpose_b=self.transpose_b)
            h0 = tf.reshape(z_, self._tensor_data_format(-1, map_size, map_size, num_filters))
            bn0 = tf.contrib.layers.batch_norm(h0, is_training=is_training, scope='bn0', **self.batchnorm_kwargs)
            h0 = tf.nn.relu(bn0)

            chain = h0
            for h in range(1, self.ng_layers):
                # h1 = relu(conv2d_transpose(h0))
                map_size *= self.stride
                num_filters /= 2
                chain = conv2d_transpose(chain,
                                         self._tensor_data_format(self.batch_size, map_size, map_size, num_filters),
                                         stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%h)
                chain = tf.contrib.layers.batch_norm(chain, is_training=is_training, scope='bn%i'%h, **self.batchnorm_kwargs)
                chain = tf.nn.relu(chain)

            # h1 = conv2d_transpose(h0)
            map_size *= self.stride
            hn = conv2d_transpose(chain,
                                  self._tensor_data_format(self.batch_size, map_size, map_size, self.c_dim),
                                  stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%(self.ng_layers))

            return tf.nn.tanh(hn)


    def discriminator(self, image):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as d_scope:

            chain = image
            for h in range(1, self.nd_layers):
                # h1 = lrelu(conv2d(h0))
                num_filters = self.df_dim if h==0 else self.df_dim*2**h
                chain = lrelu(conv2d(chain, num_filters, self.data_format, name='h%i_conv'%h))

            # h1 = linear(reshape(h0))
            hn = linear(tf.reshape(chain, [self.batch_size, -1]), self.d_out_dim, 'h%i_lin'%self.nd_layers, transpose_b=self.transpose_b)

            return hn

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
