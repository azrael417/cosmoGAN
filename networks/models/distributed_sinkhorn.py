from mpi4py import MPI
import numpy as np
import argparse

#what datatype shall we use?
dtype = np.float64
mpidtype = MPI.FLOAT if dtype==np.float32 else MPI.DOUBLE

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


#this is the local fraction of the distance matrix
def compute_distance(batch_x, batch_y):
    "compute distance matrix 1.-x*y/(|x|_2*|y|_2)"
    #compute numerator
    numerator = np.matmul(batch_x, np.transpose(batch_y))
    #denominator pieces: take sqrt individually to improve precision
    denominator_x = np.expand_dims(np.linalg.norm(batch_x, ord=2, axis=1),axis=1)
    denominator_y = np.expand_dims(np.linalg.norm(batch_y, ord=2, axis=1),axis=0)
    #denominator is the dyadic product of the two norm vectors
    denominator = np.matmul(denominator_x, denominator_y)
    #return result
    result = 1.-numerator/denominator
    #clip the matrix just in case
    return np.clip(result, a_min=0., a_max=1.)


#distributed matmul: lhs_jn = sum_k amat_jk * rhs_kn
#assumes balanced matrix, i.e.  total_rows = rows(amat)*comm_topo.comm_row_size and the same for columns
#the input and output vectors are supposed to be available on every node
def loc_glob_dist_multiply(comm_topo, amat, rhs, transpose=False):
    #some metadata stuff
    row = comm_topo.comm_row_rank
    col = comm_topo.comm_col_rank
    shapevals = amat.shape
    num_local_rows = shapevals[0]
    num_local_cols = shapevals[1]
    
    #create lhs
    if transpose:
        loc_lhs = np.zeros(num_local_cols, dtype=dtype)
        loc_lhs_tmp = np.zeros(num_local_cols, dtype=dtype)
    else:
        loc_lhs = np.zeros(num_local_rows, dtype=dtype)
        loc_lhs_tmp = np.zeros(num_local_rows, dtype=dtype)
    
    #slice the input tensor
    if transpose:
        loc_rhs = rhs[row*num_local_rows:(row+1)*num_local_rows]
        loc_lhs_tmp = np.matmul(np.transpose(amat), np.expand_dims(loc_rhs, axis=1))
    else:
        loc_rhs = rhs[col*num_local_cols:(col+1)*num_local_cols]
        loc_lhs_tmp = np.matmul(amat, np.expand_dims(loc_rhs, axis=1))
    
    #reduce over all ranks in the row comm
    if transpose:
        comm_topo.comm_col.Allreduce(loc_lhs_tmp, loc_lhs, op=MPI.SUM)
    else:
        comm_topo.comm_row.Allreduce(loc_lhs_tmp, loc_lhs, op=MPI.SUM)
    
    #do alltoall to communicate the columns
    if transpose:
        lhs = np.zeros(num_local_cols*comm_topo.comm_col_size, dtype=dtype)
        comm_topo.comm_row.Allgather([loc_lhs, num_local_cols, mpidtype], [lhs, num_local_cols, mpidtype])
    else:
        lhs = np.zeros(num_local_rows*comm_topo.comm_row_size, dtype=dtype)
        comm_topo.comm_col.Allgather([loc_lhs, num_local_rows, mpidtype], [lhs, num_local_rows, mpidtype])
    
    return lhs


#distributed matmul: lhs_jn = sum_k amat_jk * rhs_kn
#assumes balanced matrix, i.e.  total_rows = rows(amat)*comm_topo.comm_row_size and the same for columns
#the input and output vectors are supposed to be available on every node
def loc_loc_dist_multiply(comm_topo, loc_amat, loc_rhs, transpose=False):
    #some metadata stuff
    row = comm_topo.comm_row_rank
    col = comm_topo.comm_col_rank
    shapevals = loc_amat.shape
    num_local_rows = shapevals[0]
    num_local_cols = shapevals[1]
    
    #create lhs
    if transpose:
        loc_lhs = np.zeros(num_local_cols, dtype=dtype)
        loc_lhs_tmp = np.zeros(num_local_cols, dtype=dtype)
    else:
        loc_lhs = np.zeros(num_local_rows, dtype=dtype)
        loc_lhs_tmp = np.zeros(num_local_rows, dtype=dtype)
    
    #slice the input tensor
    if transpose:
        loc_lhs_tmp = np.matmul(np.transpose(loc_amat), np.expand_dims(loc_rhs, axis=1))
    else:
        loc_lhs_tmp = np.matmul(loc_amat, np.expand_dims(loc_rhs, axis=1))
    
    #reduce over all ranks in the row comm
    if transpose:
        comm_topo.comm_col.Allreduce(loc_lhs_tmp, loc_lhs, op=MPI.SUM)
    else:
        comm_topo.comm_row.Allreduce(loc_lhs_tmp, loc_lhs, op=MPI.SUM)
    
    return loc_lhs
    

#Sinkhorm algorithm to compute optimal mapping: lambda \in [0,\infty[ is regularizer, r is starting guess (could be random vector)
def distributed_sinkhorn(comm_topo, loc_amat, lambd, tolerance, min_iters, max_iters):
    
    #extract size
    loc_row_size = loc_amat.shape[0]
    loc_col_size = loc_amat.shape[1]
    size = comm_topo.comm_row_size*loc_row_size
    assert(comm_topo.comm_col_size*loc_col_size==size)
    
    
    #condition K and create vectors for final results
    loc_kmat = np.exp(-lambd*loc_amat.astype(np.float64)).astype(dtype)
    
    #init uvec and vvec
    loc_uvec = np.ones(loc_row_size, dtype=dtype)
    loc_vvec = np.ones(loc_col_size, dtype=dtype)
    loc_uvecp = np.zeros(loc_row_size, dtype=dtype)
    loc_vvecp = np.zeros(loc_col_size, dtype=dtype)
    
    #difference
    normdiff = 2.*tolerance
    iters=0
    
    #do the loop
    while( ((normdiff>tolerance) or (iters<min_iters)) and (iters<max_iters) ):
        
        #backup old vectors
        loc_uvecp[...] = loc_uvec[...]
        loc_vvecp[...] = loc_vvec[...]
        
        #update
        loc_vvec = 1./loc_loc_dist_multiply(comm_topo, loc_kmat, loc_uvec, transpose=True)        
        loc_uvec = 1./loc_loc_dist_multiply(comm_topo, loc_kmat, loc_vvec, transpose=False)
    
        #compute norm difference locally
        loc_u_normdiff = np.sum(np.square(loc_uvec-loc_uvecp))
        loc_v_normdiff = np.sum(np.square(loc_vvec-loc_vvecp))
        
        #allreduce
        loc_v_normdiff = comm_topo.comm_row.allreduce(loc_v_normdiff, op=MPI.SUM)
        loc_u_normdiff = comm_topo.comm_col.allreduce(loc_u_normdiff, op=MPI.SUM)
        
        #compute global norm
        normdiff = np.sqrt(loc_u_normdiff) + np.sqrt(loc_v_normdiff)
        iters += 1
        
        if comm_topo.comm_rank==0:
            print(iters,normdiff)
            
    #get the interesting fraction of the matrix:
    #print(uvec,vvec)


#main function
def main(config):
    
    #see if division is even
    if (config.rank % config.row_comms !=0) or (config.rank % config.col_comms !=0):
        raise ValueError("Error, make sure that the matrix rank is integer-divisible by the process grid")
    
    #size of comm grid
    comm_row_size = config.row_comms
    comm_col_size = config.col_comms
    local_row_size = config.rank // config.row_comms
    local_col_size = config.rank // config.col_comms
    num_features = 10
    
    #get rank and comm size info
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    comm_topo = comm_utils(comm, comm_size, comm_rank, comm_row_size, comm_col_size)
    
    #local ranges
    local_row_start = comm_topo.comm_row_rank*local_row_size
    local_row_end = (comm_topo.comm_row_rank+1)*local_row_size
    local_col_start = comm_topo.comm_col_rank*local_col_size
    local_col_end = (comm_topo.comm_col_rank+1)*local_col_size
    
    #create two big random vectors
    np.random.seed(1234)
    batch_x = np.random.rand(config.rank, num_features).astype(dtype)
    batch_y = np.random.rand(config.rank, num_features).astype(dtype)
    
    #extract local patches for creating the local matrix
    loc_batch_x = batch_x[local_row_start:local_row_end,:]
    loc_batch_y = batch_x[local_col_start:local_col_end,:]
    loc_cmatrix = compute_distance(loc_batch_x, loc_batch_y)
    
    #create global matrix for checking
    cmatrix = np.zeros((config.rank, config.rank), dtype=dtype)
    cmatrix_tmp = np.zeros((config.rank, config.rank), dtype=dtype)
    cmatrix_tmp[local_row_start:local_row_end, local_col_start:local_col_end]=loc_cmatrix[...]
    comm_topo.comm.Allreduce(cmatrix_tmp, cmatrix, op=MPI.SUM)
    
    #create global vector and multiply matrix with the vector
    #rhs = np.random.rand(config.rank).astype(dtype)
    #multiply
    #lhs = loc_glob_dist_multiply(comm_topo, loc_cmatrix, rhs, transpose=False)
    
    #execute the code
    #if comm_topo.comm_rank == 0:
    #    lhs_singlenode = np.squeeze(np.matmul(cmatrix, np.expand_dims(rhs,axis=1)))
    #    print(lhs,lhs_singlenode)
    #    print(np.linalg.norm(lhs-lhs_singlenode,ord=2))
    
    #run sinkhorn
    distributed_sinkhorn(comm_topo, loc_cmatrix, 1., 0.0001, 20, 100)
    
    
if __name__ == "__main__":
    AP = argparse.ArgumentParser()
    AP.add_argument("--row_comms",default=2,type=int,help="Number of row communicators")
    AP.add_argument("--col_comms",default=1,type=int,help="Number of column communicators")
    AP.add_argument("--rank",default=8,type=int,help="Rank of matrix (square by definition)")
    config = AP.parse_args()
    
    main(config)