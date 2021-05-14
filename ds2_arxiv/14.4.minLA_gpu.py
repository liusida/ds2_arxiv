# Step 4
# Matrix Reordering using GPU
# push high values to the diagonal line
"""
while True:
    randomly detect swapable pairs (choices) on GPU
    sort choices by the gains, remove the conflicted choices
    swap remain pairs
    if not many pairs found at this step:
        doulbe the search scale up to 8192.
"""

import os, argparse
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from ds2_arxiv.tools.core import loss_gpu, swap_inplace
from ds2_arxiv.tools.images import save_pic
import wandb

#TODO:
# Can detect_and_swap_gpu and loss_gpu use sparse matrix as input? or use indices as input?

target_folder = "arxiv_7636"

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

def detect_and_swap_gpu(matrix, indices, seed, threads_per_block=128, blocks=128):
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=seed)
    vec = np.zeros([threads_per_block * blocks, 3]).astype(int)
    d_matrix = cuda.to_device(matrix)
    d_vec = cuda.to_device(vec)
    _detect_gpu[blocks, threads_per_block](d_matrix, d_vec, rng_states)

    vec = d_vec.copy_to_host()
    vec = vec[~np.all(vec == 0, axis=1)] # select non-zero rows
    vec = vec[np.argsort(vec[:, 2])[::-1]] # TODO: greedy?
    vec_detected = vec.shape[0]
    # remove conflicted rows
    visited = {}
    selected = []
    for i in range(vec.shape[0]):
        if vec[i,0] not in visited and vec[i,1] not in visited:
            selected.append(i)
            visited[vec[i,0]] = 1
            visited[vec[i,1]] = 1
    vec = vec[selected, :]
    for i in range(vec.shape[0]):
        swap_inplace(matrix, indices, vec[i,0], vec[i,1])
    vec_swapped = vec.shape[0]
    return matrix, indices, vec_detected, vec_swapped

def step4(args):
    os.makedirs("tmp/minLA_gpu", exist_ok=True)

    matrix = np.load(f"data/matrix_{target_folder}.npy")
    np.random.seed(args.seed)
    indices = np.arange(matrix.shape[0])
    # random initial state
    np.random.shuffle(indices)
    matrix = matrix[indices, :]
    matrix = matrix[:, indices]
    save_pic(matrix, indices, f"tmp/minLA_gpu/seed_{args.seed}_start")
    # record loss for initial state
    loss_LA = loss_gpu(matrix)
    print(f"After initialization, loss LA = {loss_LA}")

    num_blocks = 256
    threads_per_block = 128
    for step in range(args.num_steps):
        matrix, indices, vec_detected, vec_swapped = detect_and_swap_gpu(matrix, indices, threads_per_block=threads_per_block, blocks=num_blocks, seed=step+args.seed)
        loss_LA = loss_gpu(matrix)
        record= {"step": step, "loss": loss_LA, "detected": vec_detected, "swapped": vec_swapped, "threads_per_block": threads_per_block, "blocks": num_blocks}
        wandb.log(record)
        print(record, flush=True)
        if (step-1)%20==0 or step==args.num_steps-1:
            save_pic(matrix, indices, f"tmp/minLA_gpu/seed_{args.seed}_step_{step:04}")

        if vec_detected<100: # double the search scale
            if num_blocks<8192:
                num_blocks *= 2 
    print("done", flush=True)
    return matrix, indices

def refine_with_Flip3(args, matrix, indices):
    #TODO:
    # loop:
        # pick two random numbers: i and j.
        # Flip3(indices, i, j): rotate i-th item to j-th item.
        # new_loss = loss_gpu(new_matrix)
        # if new_loss<loss:
        #   indices = new_indices
    
    # for step in range(args.num_steps):
    #     i,j = np.random.randint(low=0, high=matrix.shape[0], size=[2])
        
    return  matrix, indices

if __name__=="__main__":

    wandb.init(project=target_folder)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_steps", type=float, default=200, help="")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--tag", type=str, default="gpu")
    parser.add_argument("--exp_name", type=str)
    args = parser.parse_args()
    args.num_steps = int(args.num_steps)
    wandb.config.update(args)

    matrix, indices = step4(args)
    matrix, indices = refine_with_Flip3( matrix, indices )