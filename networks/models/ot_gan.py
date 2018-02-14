import tensorflow as tf
use_horovod = True
try:
    import horovod.tensorflow as hvd
except:
    use_horovd = False
    
from .ops import linear, conv2d, conv2d_transpose, lrelu, ot_distance


class ot_gan(object):
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

    def training_graph(self):
        
        #real input images
        if self.data_format == "NHWC":
            self.images_1 = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images-1')
            self.images_2 = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images-2')
        else:
            self.images_1 = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images-1')
            self.images_2 = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images-2')
        
        #seeds for fake images
        self.z = self.gen_prior(shape=[self.batch_size, self.z_dim], dtype=tf.float32)
        self.zp = self.gen_prior(shape=[self.batch_size, self.z_dim], dtype=tf.float32)
        
        # discriminator
        h = self.discriminator
        
        # pipe through discriminator
        xr = self.images_1
        xrp = self.images_2
        xg = self.generator(self.z, is_training=True)
        xgp = self.generator(self.zp, is_training=True)
        
        #apply disc
        hxr = h(xr)
        hxrp = h(xrp)
        hxg = h(xg)
        hxgp = h(xgp)
        
        #compute the optimal transport metric:
        tolerance = 0.00001
        min_iters = 20
        w_xr_xg = ot_distance(hxr, hxg, tolerance, min_iters)
        w_xr_xgp = ot_distance(hxr, hxgp, tolerance, min_iters)
        w_xrp_xg = ot_distance(hxrp, hxg, tolerance, min_iters)
        w_xrp_xgp = ot_distance(hxrp, hxgp, tolerance, min_iters)
        w_xr_xrp = ot_distance(hxr, hxrp, tolerance, min_iters)
        w_xg_xgp = ot_distance(hxg, hxgp, tolerance, min_iters)
        
        #defining loss
        with tf.name_scope("loss"):
            self.ot_loss = w_xr_xg + w_xr_xgp + w_xrp_xg + w_xrp_xgp - 2.* ( w_xr_xrp + w_xg_xgp )

        self.d_summary = tf.summary.merge([tf.summary.histogram("loss/L_critic", self.ot_loss)])

        g_sum = [tf.summary.histogram("loss/L_generator", self.ot_loss)]

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


    def optimizer(self, learning_rate, beta1, clip_param):
        #critic
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        if use_horovod:
            d_optim = hvd.DistributedOptimizer(d_optim)
        d_op = d_optim.minimize(self.ot_loss, var_list=self.d_vars)

        #generator
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        if use_horovod:
            g_optim = hvd.DistributedOptimizer(g_optim)
        g_op = g_optim.minimize(self.ot_loss, global_step=self.global_step, var_list=self.g_vars)

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
