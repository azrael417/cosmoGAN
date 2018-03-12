import tensorflow as tf
import ml_comm as mc
import numpy as np
import argparse

class comm_utils(object):
    
    def __init__(self, comm_size, comm_rank, comm_row_size, comm_col_size):
        if comm_row_size * comm_col_size != comm_size:
            raise ValueError("Error, comm_row_size * comm_col_size has to be comm_size")
            
        self.comm_size = comm_size
        self.comm_rank = comm_rank
        self.comm_row_size = comm_row_size
        self.comm_col_size = comm_col_size
        
        #compute my row and column rank: we set rank = col_rank + num_cols * row_rank
        self.comm_col_rank = self.comm_rank % self.comm_col_size
        self.comm_row_rank = (self.comm_rank - self.comm_col_rank) // self.comm_col_size


#this is the local fraction of the distance matrix
def compute_distance(batch_x, batch_y):
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
    #clip the matrix just in case
    return tf.clip_by_value(result, clip_value_min=0., clip_value_max=1.)


#distributed matmul: lhs_j = sum_k amat_jk * rhs_k
#assumes balanced matrix, i.e.  total_rows = rows(amat)*comm_topo.comm_row_size and the same for columns
#the input and output vectors are supposed to be available on every node
def dist_multiply(comm_topo, amat, rhs):
    #some metadata stuff
    row = comm_topo.comm_row_rank
    col = comm_topo.comm_col_rank
    shapevals = amat.get_shape().as_list()
    num_local_rows = shapevals[0]
    num_local_cols = shapevals[1]
    
    #create lhs
    lhs = tf.Variable(tf.zeros(num_local_rows*comm_topo.comm_row_size), trainable=False)
    
    #slice the input tensor
    loc_rhs = rhs[col*num_local_cols:(col+1)*num_local_cols]
    loc_lhs = tf.matmul(amat, tf.expand_dims(loc_rhs, axis=1))
    with tf.control_dependencies([loc_lhs]):
        lhs_inject_op = tf.assign(lhs[row*num_local_rows:(row+1)*num_local_rows], loc_lhs[:,0])
        with tf.control_dependencies([lhs_inject_op]):
            lhs = mc.gradients([lhs], 0)[0] * comm_topo.comm_size
            return lhs


#main function
def main(config):
    #init ml-comm
    mc.init(1, 1, 5*1024*1024, "tensorflow")
    
    #config the stopping criterion
    mc.config_team(0, 0, 500, 500, 0, 200)
    
    #see if division is even
    if (config.rank % config.row_comms !=0) or (config.rank % config.col_comms !=0):
        raise ValueError("Error, make sure that the matrix rank is integer-divisible by the process grid")
        
    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=2,
                                intra_op_parallelism_threads=33,
                                log_device_placement=False,
                                allow_soft_placement=True)
    
    #size of comm grid
    comm_row_size = config.row_comms
    comm_col_size = config.col_comms
    local_row_size = config.rank // config.row_comms
    local_col_size = config.rank // config.col_comms
    num_features = 10
    
    #get rank and comm size info
    comm_size = mc.get_nranks()
    comm_rank = mc.get_rank()
    comm_topo = comm_utils(comm_size, comm_rank, comm_row_size, comm_col_size)
    
    #local ranges
    local_row_start = comm_topo.comm_row_rank*local_row_size
    local_row_end = (comm_topo.comm_row_rank+1)*local_row_size
    local_col_start = comm_topo.comm_col_rank*local_col_size
    local_col_end = (comm_topo.comm_col_rank+1)*local_col_size
    
    #create two big random vectors
    batch_x = tf.random_uniform([config.rank, num_features], seed=1234)
    batch_y = tf.random_uniform([config.rank, num_features], seed=5678)
    
    #extract local patches for creating the local matrix
    loc_batch_x = batch_x[local_row_start:local_row_end,:]
    loc_batch_y = batch_x[local_col_start:local_col_end,:]
    loc_cmatrix = compute_distance(loc_batch_x, loc_batch_y)
    
    #create global matrix
    cmatrix = tf.Variable(tf.zeros((config.rank, config.rank)), trainable=False)
    with tf.control_dependencies([loc_cmatrix]):
        cmat_inject_op = tf.assign(cmatrix[local_row_start:local_row_end, local_col_start:local_col_end], loc_cmatrix[...])
        with tf.control_dependencies([cmat_inject_op]):
            av_cmatrix = mc.gradients([cmatrix], 0)[0] * comm_topo.comm_size
    
    #create global vector and multiply matrix with the vector
    rhs = tf.random_uniform([config.rank], seed=1234)
    #multiply
    with tf.control_dependencies([rhs, loc_cmatrix]):
        lhs = dist_multiply(comm_topo, loc_cmatrix, rhs)
    
    #execute the code
    with tf.train.MonitoredTrainingSession(config=sess_config) as sess:
        ls, cmat, rs = sess.run([lhs, av_cmatrix, rhs])
        if comm_topo.comm_rank == 0:
            print(ls, np.matmul(cmat, np.expand_dims(rs,axis=1)))
    
    
if __name__ == "__main__":
    AP = argparse.ArgumentParser()
    AP.add_argument("--row_comms",default=2,type=int,help="Number of row communicators")
    AP.add_argument("--col_comms",default=1,type=int,help="Number of column communicators")
    AP.add_argument("--rank",default=8,type=int,help="Rank of matrix (square by definition)")
    config = AP.parse_args()
    
    main(config)