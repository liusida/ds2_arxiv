import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from ds2_arxiv.tools.new_algo import loss_gpu
# put matrix in device
# start 10000 threads
# for each thread
#   get two random numbers (Howto: http://numba.pydata.org/numba-doc/0.33.0/cuda/random.html)
#   test if swap will reduce loss
#   if yes, record two numbers in an array, using an atomic increasing index
# on host, check for confliction, remove conflicted ones
# swap all of them on device
@cuda.jit
def _detect_gpu(matrix, vec, rng_states):
    thread_id = cuda.grid(1)
    if thread_id<vec.shape[0]:
        l = matrix.shape[0]
        x = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * l)
        y = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * l)
        ret = 0
        for m in [x, y]:
            m_inv = x + y - m
            for n in range(l):
                if matrix[m, n] > 0 or matrix[m_inv, n] > 0:
                    if m != n and m_inv != n:
                        ret += (abs(m-n)-abs(m_inv-n)) * (matrix[m, n] - matrix[m_inv, n])
        if ret>0:
            vec[thread_id,0] = x
            vec[thread_id,1] = y

@cuda.jit
def _swap_gpu(matrix, indices, vec):
    thread_id = cuda.grid(1)
    if thread_id<vec.shape[0]:
        x = vec[thread_id,0]
        y = vec[thread_id,1]
        l = matrix.shape[0]
        if x<l and y<l:
            for i in range(l):
                _tmp = matrix[x,i]
                matrix[x,i] = matrix[y,i]
                matrix[y,i] = _tmp

                _tmp = matrix[i,x]
                matrix[i,x] = matrix[i,y]
                matrix[i,y] = _tmp

            _tmp = indices[x]
            indices[x] = indices[y]
            indices[y] = _tmp


def detect_and_swap_gpu(matrix, indices, seed):
    threads_per_block = 128
    blocks = 128
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=seed)

    vec = np.zeros([threads_per_block * blocks, 2]).astype(int)
    d_matrix = cuda.to_device(matrix)
    d_vec = cuda.to_device(vec)

    _detect_gpu[blocks, threads_per_block](d_matrix, d_vec, rng_states)
    vec = d_vec.copy_to_host()
    vec = vec[~np.all(vec == 0, axis=1)] # select non-zero rows
    print(vec.shape)
    # remove conflicted rows #TODO: here might have bug.
    visited = {}
    selected = []
    for i in range(vec.shape[0]):
        if vec[i,0] not in visited and vec[i,1] not in visited:
            selected.append(i)
            visited[vec[i,0]] = 1
            visited[vec[i,1]] = 1
    vec = vec[selected, :]
    print(vec.shape)
    if vec.shape[0]>0:
        blocks = ( vec.shape[0] + threads_per_block -1) // threads_per_block
        d_indices = cuda.to_device(indices)
        d_vec = cuda.to_device(vec)
        _swap_gpu[blocks, threads_per_block](d_matrix, d_indices, d_vec)
        matrix = d_matrix.copy_to_host()
        indices = d_indices.copy_to_host()
    return matrix, indices

if __name__=="__main__":
    def unittest():
        m = 50
        indices = np.arange(m)
        matrix = np.random.rand(m,m)
        matrix = (matrix+matrix.T)/2
        old_matrix = matrix.copy()
        print(loss_gpu(matrix))
        for i in range(1):
            matrix, indices = detect_and_swap_gpu(matrix, indices, seed=i)
            print(i)
            print(indices)
            p1 = loss_gpu(matrix)
            print(p1)

        old_matrix = old_matrix[indices, :]
        old_matrix = old_matrix[:, indices]
        p2 = loss_gpu(old_matrix)
        print(p2)
        print(np.sum(old_matrix), "=", np.sum(matrix))
        assert(p1==p2)


    unittest()