import tensorflow as tf
import ml_comm as mc

class comm_utils(object):
    
    def __init__(self, comm_size, comm_rank, comm_row_size, comm_col_size):
        if num_comm_rows * num_comm_cols != comm_size:
            raise ValueError("Error, comm_row_size * comm_col_size has to be comm_size")
            
        self.comm_size = comm_size
        self.comm-rank = comm_rank
        self.comm_row_size = comm_row_size
        self.comm_col_size = comm_col_size
        
        #compute my row and column rank: we set rank = col_rank + num_cols * row_rank
        self.comm_col_rank = comm_rank % comm_col_size
        self.comm_row_rank = (comm_rank - self.comm_col_rank) // comm_col_size


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
    return 1.-numerator/denominator


#distributed matmul: lhs_j = sum_k amat_jk * rhs_k
#assumes balanced matrix, i.e.  total_rows = rows(amat)*comm_topo.comm_row_size and the same for columns
#the input and output vectors are supposed to be available on every node
def dist_multiply(comm_topo, amat, rhs)
    #some metadata stuff
    row = comm_topo.comm_row_rank
    col = comm_topo.comm_col_rank
    num_local_rows, num_local_cols = amat.shape
    
    #output vector
    lhs = tf.zeros(num_local_rows * comm_topo.comm_row_size)
    
    #slice the input tensor
    loc_rhs = rhs[col*num_local_cols:(col+1)*num_local_cols]
    lhs[row*num_local_rows:(row+1)*num_local_rows] = tf.multiply(amat, loc_rhs)
    ar_op = mc.allreduce(lhs)
    
    with tf.control_dependencies([ar_op]):
        return lhs


#main function
def main():
    #init ml-comm
    mc.init(1, 1, 5*1024*1024, "tensorflow")
    
    #size of comm grid
    comm_row_size = 4
    comm_col_size = 4
    local_row_size = 4
    local_col_size = 4
    num_features = 10
    
    #get rank and comm size info
    comm_size = mc.size()
    comm_rank = mc.rank()
    comm_topo = comm_utils(comm_size, comm_rank, comm_row_size, comm_col_size)
    
    #create two big random vectors
    batch_x = tf.random_uniform([local_row_size * comm_row_size, num_features], seed=1234)
    batch_y = tf.random_normal([local_col_size * comm_col_size, num_features], seed=5678)
    
    #extract local patches for creating the local matrix
    loc_batch_x = batch_x[comm_topo.comm_row_rank*local_row_size:(comm_topo.comm_row_rank+1)*local_row_size,:]
    loc_batch_y = batch_x[comm_topo.comm_col_rank*local_col_size:(comm_topo.comm_col_rank+1)*local_col_size,:]
    amat = compute_distance(loc_batch_x, loc_batch_y)
    
    #print the shape
    print(amat.shape)
    
    
if __name__ == "__main__":
    main()