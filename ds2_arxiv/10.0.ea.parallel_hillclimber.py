import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numpy.core.shape_base import block
from ds2_arxiv.tools.new_algo import loss_gpu, swap_inplace
import cv2
import argparse

import wandb

"""
Based on Josh's psuedo-code, parallel hill climber:
parents = {} x pop_size
while True:
    for each parents i:
        possible swap[i] = get possible swap from parents[i] 
    for each parents i:
        children[i] = parents[i] apply possible_swap[i]
"""


@cuda.jit
def _detect_gpu(matrix, vec, rng_states):
    grid_id = cuda.grid(1)
    if grid_id < vec.shape[0]:
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
        if ret > 0:
            vec[grid_id, 0] = x
            vec[grid_id, 1] = y
            vec[grid_id, 2] = ret


def _detect_possible_swaps(parent, gpu_random_seed, threads_per_block=128, blocks=128):
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=gpu_random_seed)
    # detected_pairs is for storing the detect results
    detected_pairs = np.zeros([threads_per_block * blocks, 3]).astype(int)
    d_parent = cuda.to_device(parent)
    d_detected_pairs = cuda.to_device(detected_pairs)
    _detect_gpu[blocks, threads_per_block](d_parent, d_detected_pairs, rng_states)
    detected_pairs = d_detected_pairs.copy_to_host()
    detected_pairs = detected_pairs[~np.all(detected_pairs == 0, axis=1)]  # select non-zero rows
    return detected_pairs


def _apply_swaps(matrix, indices, detected_pairs, seed, inplace=False):
    """
    @seed: different seed will lead to different outcome, so if you want multiple directions, use multiple seeds.
    """
    if not inplace:
        matrix = matrix.copy()
        indices = indices.copy()

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(detected_pairs, axis=0)
    # remove conflicted rows
    visited = {}
    selected = []
    for i in range(detected_pairs.shape[0]):
        if detected_pairs[i, 0] not in visited and detected_pairs[i, 1] not in visited:
            selected.append(i)
            visited[detected_pairs[i, 0]] = 1
            visited[detected_pairs[i, 1]] = 1
    swap_pairs = detected_pairs[selected, :]
    for i in range(swap_pairs.shape[0]):
        swap_inplace(matrix, indices, swap_pairs[i, 0], swap_pairs[i, 1])
    return matrix, indices, swap_pairs.shape[0]

def save_pic(matrix, indices, title=""):
    im = np.array(matrix / matrix.max() * 255, dtype = np.uint8)
    im = 255-im
    border_width = 2
    a = np.zeros(shape=[im.shape[0], border_width], dtype=np.uint8)
    im = np.concatenate([a,im,a], axis=1)
    b = np.zeros(shape=[border_width, border_width+im.shape[0]+border_width ], dtype=np.uint8)
    im = np.concatenate([b,im,b], axis=0)

    # im_color = cv2.applyColorMap(im, cv2.COLORMAP_HOT)
    cv2.imwrite(f"tmp/{title}.png", im)
    wandb.save(f"tmp/{title}.png")
    np.save(f"tmp/{title}_indicies.npy", indices)
    wandb.save(f"tmp/{title}_indicies.npy")

def load_matrix_from_indices(original_matrix, indices):
    matrix = original_matrix.copy()
    matrix = matrix[indices,:]
    matrix = matrix[:,indices]
    return matrix

def parallel_hill_climber(matrix, pop_size=3, total_steps=100, master_seed=0):
    parent_indices = {}
    print("init...")
    rng = np.random.default_rng(seed=master_seed)
    for i in range(pop_size):
        parent_indices[i] = np.arange(matrix.shape[0])
        # start with random indices
        rng.shuffle(parent_indices[i])

    print("start")
    rng = np.random.default_rng(seed=master_seed)
    for step in range(total_steps):
        possible_swaps = {}
        children_indices = {}
        num_detected = {}
        num_swapped = {}
        LAs = {}
        for i in range(pop_size):
            parent = load_matrix_from_indices(matrix, parent_indices[i]) # lazy instantiate to save memory

            gpu_random_seed = rng.integers(low=0, high=10000000)
            possible_swaps[i] = _detect_possible_swaps(parent, gpu_random_seed=gpu_random_seed)
            num_detected[i] = possible_swaps[i].shape[0]

            swap_random_seed = rng.integers(low=0, high=10000000)
            child, children_indices[i], num_swapped[i] = _apply_swaps(matrix=parent, indices=parent_indices[i],
                                                            detected_pairs=possible_swaps[i], seed=swap_random_seed)

            LAs[i] = loss_gpu(child)

        parent_indices = children_indices

        _f = list(LAs.values())
        _s = list(num_swapped.values())
        _d = list(num_detected.values())
        record = {
            "step": step,
            "LA/all": wandb.Histogram(_f),
            "LA/min": np.min(_f),
            "LA/mean": np.mean(_f),
            "LA/std": np.std(_f),
            "num_swapped/all": wandb.Histogram(_s),
            "num_swapped/mean": np.mean(_s),
            "num_detected/all": wandb.Histogram(_d),
            "num_detected/mean": np.mean(_d),
        }
        wandb.log(record)
        bestsofar = load_matrix_from_indices(matrix, parent_indices[np.argmin(_f)])
        save_pic(bestsofar, parent_indices[np.argmin(_f)], f"10.0/best_step_{step:04}")
        
        print(f"step {step}: min LAs {np.min(_f)}")


if __name__ == "__main__":
    wandb.init(project="block_diagonal_gpu", tags=["ParallelHillClimber"])

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--total_steps", type=float, default=1e3, help="")
    parser.add_argument("-p", "--pop_size", type=int, default=10, help="")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    args.total_steps = int(args.total_steps)
    wandb.config.update(args)

    matrix = np.load("shared/author_similarity_matrix.npy")
    # matrix = matrix[:200, :200]
    # np.random.seed(4)
    # matrix = np.random.random([400,400]) > 0.999
    # matrix = (matrix+matrix.T)/2
    # matrix = np.zeros([6,6])
    # matrix[0:2, 0:2] = 0.5
    # matrix[3:5, 3:5] = 0.5
    # matrix[0,4] = matrix[4,0] = 0.8
    np.fill_diagonal(matrix,1)
    parallel_hill_climber(matrix, pop_size=args.pop_size, total_steps=args.total_steps, master_seed=args.seed)
