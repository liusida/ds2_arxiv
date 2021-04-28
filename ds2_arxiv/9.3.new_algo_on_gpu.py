import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numpy.core.shape_base import block
from ds2_arxiv.tools.new_algo import loss_gpu, swap_inplace
import wandb
import cv2
import argparse

wandb.init(project="block_diagonal_gpu")

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_epochs", type=float, default=1e3, help="")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--tag", type=str, default="")
args = parser.parse_args()
args.num_epochs = int(args.num_epochs)
wandb.config.update(args)

"""
put matrix in device
start 10000 threads
for each thread
  get two random numbers (Howto: http://numba.pydata.org/numba-doc/0.33.0/cuda/random.html)
  test if swap will reduce loss
  if yes, record two numbers in an array, using an atomic increasing index
on host, check for confliction, remove conflicted ones
swap all of them on device
"""

@cuda.jit
def _detect_gpu(matrix, vec, rng_states):
    grid_id = cuda.grid(1)
    if grid_id<vec.shape[0]:
        l = matrix.shape[0]
        x = int(xoroshiro128p_uniform_float32(rng_states, grid_id) * l)
        y = int(xoroshiro128p_uniform_float32(rng_states, grid_id) * l)
        ret = 0
        for m in [x, y]:
            m_inv = x + y - m
            for n in range(l):
                if matrix[m, n] > 0 or matrix[m_inv, n] > 0:
                    if m != n and m_inv != n:
                        ret += (abs(m-n)-abs(m_inv-n)) * (matrix[m, n] - matrix[m_inv, n])
        if ret>0:
            vec[grid_id,0] = x
            vec[grid_id,1] = y
            vec[grid_id,2] = ret

EPSILON = 1e3
@cuda.jit
def _detect_all_gpu(matrix, index, vec):
    """
    """
    x,y = cuda.grid(2)
    l = matrix.shape[0]
    if x < l and y < l:
        ret = 0
        for m in [x, y]:
            m_inv = x + y - m
            for n in range(l):
                if matrix[m, n] > 0 or matrix[m_inv, n] > 0:
                    if m != n and m_inv != n:
                        ret += (abs(m-n)-abs(m_inv-n)) * (matrix[m, n] - matrix[m_inv, n])
        if ret>EPSILON:
            i = cuda.atomic.add(index, 0, 1)
            if i<vec.shape[0]:
                vec[i,0] = x
                vec[i,1] = y
                vec[i,2] = ret

def detect_and_swap_gpu(matrix, indices, seed, mode='random'):
    threads_per_block, blocks = 128, 128
    # wandb.log({"threads_per_block": threads_per_block, "blocks": blocks})
    if mode=='random':
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=seed)
        vec = np.zeros([threads_per_block * blocks, 3]).astype(int)
    elif mode=='all':
        vec = np.zeros([4096, 3]).astype(int)
        index = np.zeros([1]).astype(int)
        d_index = cuda.to_device(index)
        threadsperblock = (32,32)
        blockspergrid_x = ( matrix.shape[0] + threadsperblock[0] -1) // threadsperblock[0]
        blockspergrid_y = ( matrix.shape[1] + threadsperblock[1] -1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_matrix = cuda.to_device(matrix)
    d_vec = cuda.to_device(vec)

    if mode=='random':
        _detect_gpu[blocks, threads_per_block](d_matrix, d_vec, rng_states)
    elif mode=='all':
        _detect_all_gpu[blockspergrid, threadsperblock](d_matrix, d_index, d_vec)


    vec = d_vec.copy_to_host()
    vec = vec[~np.all(vec == 0, axis=1)] # select non-zero rows
    vec = vec[np.argsort(vec[:, 2])[::-1]] # TODO: greedy?
    # print(vec[:5])
    vec_detected = vec.shape[0]
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
    for i in range(vec.shape[0]):
        swap_inplace(matrix, indices, vec[i,0], vec[i,1])
    vec_swapped = vec.shape[0]
    wandb.log({"detected": vec_detected, "swapped": vec_swapped})
    return matrix, indices, vec_detected

def save_pic(elements, title=""):
    ret = loss_gpu(elements)
    record = {
        "loss": ret,
    }
    wandb.log(record)
    print(f"loss: {ret}")
    im = np.array(elements / elements.max() * 255, dtype = np.uint8)
    im_color = cv2.applyColorMap(im, cv2.COLORMAP_HOT)
    cv2.imwrite(f"tmp/9.3.{title}.png", im_color)

if __name__=="__main__":
    def main():
        matrix = np.load("shared/author_similarity_matrix.npy")
        # np.random.seed(1)
        # matrix = np.random.random([5000,5000]).astype(np.float_)
        # # matrix = np.arange(25).reshape([5,5])
        # matrix = (matrix+matrix.T)/2
        indices = np.arange(matrix.shape[0])
        old_matrix = matrix.copy()
        print(loss_gpu(matrix))
        mode = 'random'
        for i in range(args.num_epochs):
            matrix, indices, vec_detected = detect_and_swap_gpu(matrix, indices, seed=i+args.seed, mode=mode)
            if vec_detected<1 and mode=='all':
                break # finished
            elif vec_detected<20: # start enumerate mode
                mode = 'all'
            # else:
            #     mode = 'random'
            print(i)
            wandb.log({"epoch": i})
            p1 = loss_gpu(matrix)
            print(p1)
            save_pic(matrix, f"step_{i:05}")

        old_matrix = old_matrix[indices, :]
        old_matrix = old_matrix[:, indices]
        p2 = loss_gpu(old_matrix)
        print(p2)
        print(np.sum(old_matrix), "=", np.sum(matrix))
        
        assert(np.isclose(p1, p2)), f"{p1} == {p2}"


    main()