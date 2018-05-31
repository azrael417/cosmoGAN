import tensorflow as tf
import tensorflow.contrib.keras as tfk


def debug(t, tensorlist, name=None):
    reducelist = [tf.reduce_max(x) for x in tensorlist]
    if name:
        reducelist = [name]+reducelist
    #wrap t
    #t = tf.Print(t, reducelist, name=name)
    #return
    return t

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, transpose_b=False):

    shape = input_.get_shape().as_list()
    if not transpose_b:
        w_shape = [shape[1], output_size]
    else:
        w_shape = [output_size, shape[1]]

    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable('w', w_shape, tf.float32, trainable=True,
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('b', [output_size], trainable=True,
            initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix, transpose_b=transpose_b) + bias


def conv2d(input_, out_channels, data_format, kernel=5, stride=2, stddev=0.02, name="conv2d"):

    if data_format == "NHWC":
        in_channels = input_.get_shape()[-1]
        strides = [1, stride, stride, 1]
    else: # NCHW
        in_channels = input_.get_shape()[1]
        strides = [1, 1, stride, stride]

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, in_channels, out_channels], trainable=True,
                             initializer=tfk.initializers.glorot_uniform())
                            #initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=strides, padding='SAME', data_format=data_format)

        biases = tf.get_variable('biases', [out_channels], trainable=True, initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases, data_format=data_format), conv.get_shape())

        #debug
        conv = debug(conv,[w],name=name+"_debug")
        #debug

        return conv


def conv2d_transpose(input_, output_shape, data_format, kernel=5, stride=2, stddev=0.02,
                     name="conv2d_transpose"):

    if data_format == "NHWC":
        in_channels = input_.get_shape()[-1]
        out_channels = output_shape[-1]
        strides = [1, stride, stride, 1]
    else:
        in_channels = input_.get_shape()[1]
        out_channels = output_shape[1]
        strides = [1, 1, stride, stride]

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, out_channels, in_channels], trainable=True,
                            initializer=tfk.initializers.glorot_uniform())
                            #initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=strides, data_format=data_format)

        biases = tf.get_variable('biases', [out_channels], trainable=True, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases, data_format=data_format), deconv.get_shape())
        
        #debug
        deconv = debug(deconv,[w],name=name+"_debug")
        #debug
        
        return deconv
       

def lrelu(x, alpha=0.2, name="lrelu"):
    with tf.name_scope(name):
      return tf.maximum(x, alpha*x)


#comm utils for distributed sinkhorn
class comm_utils(object):
    
    def __init__(self, comm, comm_size, comm_rank, comm_row_size, comm_col_size):
        if comm_row_size * comm_col_size != comm_size:
            raise ValueError("Error, comm_row_size * comm_col_size has to be comm_size")
        
        #copy input
        self.comm = comm
        self.comm_size = comm_size
        self.comm_rank = comm_rank
        self.comm_row_size = comm_row_size
        self.comm_col_size = comm_col_size
        
        #compute my row and column rank: we set rank = col_rank + num_cols * row_rank
        self.comm_col_rank = self.comm_rank % self.comm_col_size
        self.comm_row_rank = (self.comm_rank - self.comm_col_rank) // self.comm_col_size
        
        #split comms
        self.comm_row = comm.Split(color=self.comm_row_rank, key=self.comm_col_rank)
        self.comm_col = comm.Split(color=self.comm_col_rank, key=self.comm_row_rank)


##Sinkhorn metric: do not back-propagate through here
#def sk_iteration_body(amat, rvec, cvec, rvecp, cvecp, iters, tolerance, min_iters):
#    #backup previous
#    rvecp = rvec
#    cvecp = cvec
#    #update
#    cvec = 1./tf.matmul(tf.matrix_transpose(amat),rvec)
#    rvec = 1./tf.matmul(amat,cvec)
#    iters+=1
#    return amat, rvec, cvec, rvecp, cvecp, iters, tolerance, min_iters
#
#
#def sk_is_converged(amat, rvec, cvec, rvecp, cvecp, iters, tolerance, min_iters):
#    normdiff = tf.norm(rvec-rvecp,ord=2)+tf.norm(cvec-cvecp,ord=2)
#    return tf.logical_and( tf.logical_or( tf.greater_equal(normdiff, tolerance), tf.less_equal(iters,min_iters) ), tf.less_equal(iters,10*min_iters) )
#
#
#def compute_mapping(amat, tolerance, min_iters):
#    #size:
#    size = amat.shape[0]
#    evec = tf.ones((size,1), dtype=tf.float32)
#    iters = tf.zeros((), dtype=tf.int32)
#    
#    #create vectors
#    rvec = evec
#    cvec = evec
#    rvecp = tf.zeros((size,1))
#    cvecp = tf.zeros((size,1))
#    #fixed point iteration
#    _, rvec, cvec, _, _, _, _, _ = tf.while_loop(sk_is_converged, 
#                                                 sk_iteration_body, 
#                                                 loop_vars=[amat, rvec, cvec, rvecp, cvecp, iters, tolerance, min_iters], 
#                                                 parallel_iterations=1,
#                                                 back_prop=False)
#    
#    #squeeze result
#    rvec = tf.squeeze(rvec)
#    cvec = tf.squeeze(cvec)
#    #construct pmat
#    pmat = tf.matmul(tf.matmul(tf.diag(rvec),amat),tf.diag(cvec))
#    
#    return pmat


def compute_cost(batch_x, batch_y):
    "compute distance matrix 1.-x*y/(|x|_2*|y|_2)"
    #compute numerator
    numerator = tf.matmul(batch_x, tf.matrix_transpose(batch_y))
    #denominator pieces: take sqrt individually to improve precision
    denominator_x = tf.expand_dims(tf.norm(batch_x, ord=2, axis=1),axis=1)
    denominator_y = tf.expand_dims(tf.norm(batch_y, ord=2, axis=1),axis=0)
    #denominator is the dyadic product of the two norm vectors
    denominator = tf.matmul(denominator_x, denominator_y)
    #return result
    result = 1.-numerator/denominator
    #clip result
    result = tf.clip_by_value(result, clip_value_min=0., clip_value_max=1.)
    #return
    return result


def ot_distance(c_x_y, map_x_y):
    #compute the local costs:
    local_loss = tf.reduce_sum(map_x_y * c_x_y)
    
    #compute the loss
    return local_loss
