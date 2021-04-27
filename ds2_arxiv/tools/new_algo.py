import math
import numpy as np
from numba import prange, njit
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

DISABLE_NUMBA = False

def dummy_wrapper(func):
    """
    use this decorator to disable njit for debug
    """
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
if DISABLE_NUMBA: # disable numba to debug
    njit = dummy_wrapper 


"""
Loss = sum_i(sum_j(e[i,j] * abs(i-j)/m))
"""
def loss(matrix):
    # return loss_gpu(matrix) # use GPU
    return loss_cpu(matrix) # use CPU

@njit
def loss_cpu(elements):
    """ 
    This file directly optimize this loss function.
    calculate the whole matrix
    @return at the scale: Loss / 2
    """
    ret = 0
    l = elements.shape[0]
    _l_1 = 1.0/l
    for i in range(l):
        for j in range(i):
            if elements[i, j] > 0:
                ret += (i-j) * _l_1 * elements[i, j]  # here because j<i, we can safely ommit abs() for speed.
    return ret

@cuda.jit
def _loss_gpu(matrix, ret):
    x, y = cuda.grid(2)
    if x<matrix.shape[0] and y<matrix.shape[1]: # Important: Don't use early return in CUDA functions, will cause Memory Error. Use cuda-memcheck to check memory.
        if y<x: # only compute half of the matrix
            if matrix[x, y] > 0:
                loss = (x-y) * ret[1] * matrix[x, y]
                cuda.atomic.add(ret, 0, loss)
    
def loss_gpu(matrix):
    """ same as loss(), but run on GPU """
    ret = np.array([0.0, 1/matrix.shape[0]], dtype=np.float64)
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(matrix.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(matrix.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    _loss_gpu[blockspergrid, threadsperblock](matrix, ret)
    return ret[0]

@njit
def loss_partial(elements, target_a, target_b):
    """
    compute only for target_a and target_b, and without scale by 1/l
    @return at the scale: Loss * l / 2
    """
    ret = 0
    l = elements.shape[0]
    for i, i_inv in [[target_a, target_b], [target_b, target_a]]:
        for j in range(l):
            if elements[i, j] > 0:
                if i > j:
                    ret += (i-j) * elements[i, j]
                elif i < j:
                    if i_inv!=j:
                        ret += (j - i) * elements[i, j]

    return ret


@njit
def swap_inplace(elements, indices, i, j):
    """ swap the matrix and indices inplace at position i and j """
    _tmp = elements[i, :].copy()
    elements[i, :] = elements[j, :]
    elements[j, :] = _tmp

    _tmp = elements[:, i].copy()
    elements[:, i] = elements[:, j]
    elements[:, j] = _tmp

    # also swap indices to keep track of the direct way from original matrix to final matrix.
    if indices is not None:
        _tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = _tmp


@njit
def loss_gradient_if_swap(elements, target_i, target_j):
    """ 
    This file directly optimize this loss function.
    only calculate i and j-th row and col
    @return at the scale: Loss * l
    """
    ret = 0  # loss gained by swapping
    l = elements.shape[0]

    for m, m_inv in [[target_i, target_j], [target_j, target_i]]:
        for n in range(l):
            if elements[m, n] > 0 or elements[m_inv, n] > 0:
                if m != n and m_inv != n:
                    ret += (abs(m-n)-abs(m_inv-n)) * (elements[m, n] - elements[m_inv, n])
    return ret


"""
detect_and_swap_gpu strategy:

put matrix in device
start 10000 threads
for each thread
  get two random numbers
  test if swap will reduce loss
  if yes, record two numbers in an array, using an atomic increasing index
on host, check for confliction, remove conflicted ones
swap all of them on device
"""
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
def _swap_gpu(matrix, vec):
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

def detect_and_swap_gpu(matrix, seed):
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
    # remove conflicted rows
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
        d_vec = cuda.to_device(vec)
        _swap_gpu[blocks, threads_per_block](d_matrix, d_vec)
        matrix = d_matrix.copy_to_host()
    return matrix


if __name__ == "__main__":
    # unit_test
    def unittest():
        def unittest_1():
            l = 3
            a = np.eye(l)
            a[1, 0] = a[0, 1] = 0.5
            full_loss = loss(a)
            partial_loss = loss_partial(a, 1, 2)
            assert(np.isclose(full_loss, partial_loss/l))

        def unittest_2_5():
            l = 3
            a = np.ones([l, l])
            partial_loss = loss_partial(a, 1, 2)
            assert(partial_loss == 4)

        def unittest_3():
            for _ in range(100):
                l = 3
                a = np.random.random([l, l])
                a = (a+a.T)/2
                full_loss = loss(a)
                partial_loss = loss_partial(a, 1, 2)
                assert(np.isclose(full_loss, partial_loss/l))

        def unittest_5_5():
            a = np.array([[0., 0.5, 0.],
                        [0.5, 0., 0.5],
                        [0., 0.5, 0.]])
            p2 = loss_gradient_if_swap(a, 1, 2)
            assert(np.isclose(-1,p2))
        def unittest_6():
            l = 30
            for _ in range(100):
                a = (np.random.random([l, l]) * 2).astype(int)
                a = (a+a.T)/2
                i, j = (np.random.random([2])*l).astype(int)
                p2 = loss_gradient_if_swap(a, i, j)
                p1_before = loss_partial(a, i, j)
                swap_inplace(a, None, i, j)
                p1_after = loss_partial(a, i, j)
                assert(np.isclose(p2*0.5, p1_before-p1_after))

        def unittest_7(l = 3):
            np.random.seed(0)
            for _ in range(100):
                a = (np.random.random([l, l]) * 2).astype(int)
                a = (a+a.T)/2
                p1 = loss_gpu(a)
                p2 = loss_cpu(a)
                assert(np.isclose(p1,p2))

        def unittest_8():
            matrix = np.random.rand(5000,5000)
            matrix = (matrix+matrix.T)/2
            print(loss_gpu(matrix))
            for i in range(10):
                matrix = detect_and_swap_gpu(matrix, seed=i)
                print(i)
                print(loss_gpu(matrix))

        unittest_1()
        unittest_2_5()
        unittest_3()
        unittest_5_5()
        unittest_6()
        unittest_7(l = 3)
        unittest_7(l = 30)
        unittest_8()

    unittest()
