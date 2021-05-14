import math
import numpy as np
from numba import njit
from numba import cuda

"""
Loss = sum_i(sum_j(A[i,j] * abs(i-j)/(m^2)))
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
    _l_1 = 1.0/(l*l)
    for i in range(l):
        for j in range(i):
            if elements[i, j] > 0:
                ret += (i-j) * _l_1 * elements[i, j]  # here because j<i, we can safely ommit abs() for speed.
    return ret

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

@cuda.jit
def _loss_gpu(matrix, ret):
    """ CUDA kernel function for loss_gpu() """
    x, y = cuda.grid(2)
    if x<matrix.shape[0] and y<matrix.shape[1]: # Important: Don't use early return in CUDA functions, will cause Memory Error. Use cuda-memcheck to check memory.
        if y<x: # only compute half of the matrix
            if matrix[x, y] > 0:
                loss = (x-y) * ret[1] * matrix[x, y]
                cuda.atomic.add(ret, 0, loss)
    
def loss_gpu(matrix):
    """ same as loss(), but run on GPU """
    ret = np.array([0.0, 1/(matrix.shape[0]*matrix.shape[0])], dtype=np.float64)
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(matrix.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(matrix.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    _loss_gpu[blockspergrid, threadsperblock](matrix, ret)
    return ret[0]

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
