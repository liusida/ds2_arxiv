import math
import numpy as np
from numba import prange, njit
from numba import cuda

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
    return loss_gpu(matrix) # use GPU

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

        unittest_1()
        unittest_2_5()
        unittest_3()
        unittest_5_5()
        unittest_6()
        unittest_7(l = 3)
        unittest_7(l = 30)

    unittest()
