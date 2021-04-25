import numpy as np
from numba import prange, njit
import matplotlib.pyplot as plt
import argparse

def dummy_wrapper(func):
    """
    use this decorator to disable njit for debug
    """
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
# njit = dummy_wrapper

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=float, default=1e3, help="time steps at each epoch (10 epoch in total)")
parser.add_argument("--seed", type=int, default=0, help="random seed")
args = parser.parse_args()
args.n = int(args.n)

@njit
def loss(elements):
    """ This file directly optimize this loss function.
    calculate the whole matrix
    """
    ret = 0
    l = elements.shape[0]
    _l_1 = 1.0/l
    for i in range(l):
        for j in range(i):
            if elements[i,j]>0:
                ret += (i-j) * _l_1 * elements[i,j]
    return ret

@njit
def is_good_swap(elements, target_i, target_j, bin_size=1):
    """ This file directly optimize this loss function.
    only calculate i and j-th row and col
    """
    ret = 0
    ret_1 = 0 # after potential swap
    l = elements.shape[0]

    if target_i+bin_size>l or target_j+bin_size>l: # not enough space for a bin
        return False

    for m, m_inv in [[target_i,target_j], [target_j,target_i]]:
        for n in range(l):
            for b in range(bin_size):
                m_b = m+b
                m_inv_b = m_inv+b
                if elements[m_b,n]>0:
                    ret += abs(m_b-n) * elements[m_b,n]
                    ret_1 += abs(m_inv_b-n) * elements[m_b,n]
                if elements[n,m_b]>0:
                    ret += abs(m_b-n) * elements[n,m_b]
                    ret_1 += abs(m_inv_b-n) * elements[n,m_b]
    return ret>ret_1

@njit
def search(elements, indices, bin_size=1, seed=0, total_steps=100):
    current_loss = loss(elements)
    l = elements.shape[0]
    np.random.seed(seed)
    # print(f"loss: {current_loss}")
    for step in prange(total_steps):
        i,j = int(np.random.random() * l), int(np.random.random() * l)
        if i==j:
            continue
        if is_good_swap(elements, i, j, bin_size=bin_size):
            # swap
            _tmp = elements[i:i+bin_size,:].copy()
            elements[i:i+bin_size,:] = elements[j:j+bin_size,:]
            elements[j:j+bin_size,:] = _tmp
            
            _tmp = elements[:,i:i+bin_size].copy()
            elements[:,i:i+bin_size] = elements[:,j:j+bin_size]
            elements[:,j:j+bin_size] = _tmp
            
            # also swap indices to keep track of the direct way from original matrix to final matrix.
            _tmp = indices[i:i+bin_size].copy()
            indices[i:i+bin_size] = indices[j:j+bin_size]
            indices[j:j+bin_size] = _tmp
        
    return elements, indices

def save_pic(elements, title=""):
    plt.figure(figsize=[20,20])
    ret = loss(elements)
    print(f"loss: {ret}")
    plt.title(f"loss: {ret}")
    # plt.figure(figsize=[5,5])
    plt.imshow(elements)
    plt.colorbar()
    plt.savefig(f"tmp/9.1.{title}.png")
    plt.close

random = np.random.default_rng(seed=args.seed)

# small sample dataset:
# elements = np.zeros(shape=[10,10])
# elements[3:5, 3:5] = 0.5
# elements[0:4, 0:4] = 0.5
# for i in range(elements.shape[0]):
#     elements[i,i] = 1

# real dataset:
elements = np.load("shared/author_similarity_matrix.npy")
indices = np.arange(elements.shape[0])
save_pic(elements, "start")
# shuffle
i = random.permutation(np.arange(elements.shape[0]))
elements = elements[i]
elements = elements[:,i]
indices = indices[i]

# print(elements)
save_pic(elements, "shuffled")

# will take about 2 mins
for bin_size in range(10,0,-1): # decrease bin_size
    print(f"current bin size: {bin_size}")
    elements, indices = search(elements, indices, bin_size=bin_size, seed=args.seed, total_steps=args.n)

    save_pic(elements, f"end_n{args.n}_s{args.seed}_b{bin_size}")

np.save(f"tmp/end_n{args.n}_s{args.seed}.npy", indices)
# print(elements)


def test():
    elements = np.load("shared/author_similarity_matrix.npy")
    indices = np.load(f"tmp/end_n{args.n}_s{args.seed}.npy")
    elements = elements[indices, :]
    elements = elements[:, indices]
    save_pic(elements, f"test_n{args.n}_s{args.seed}")

test()