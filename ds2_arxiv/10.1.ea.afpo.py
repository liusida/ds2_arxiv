import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numpy.core.shape_base import block
from ds2_arxiv.tools.new_algo import loss_gpu, swap_inplace
import cv2
import argparse

import wandb

"""
AFPO: Age-Fitness Pareto Optimization
https://dl.acm.org/doi/pdf/10.1145/1830483.1830584
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

def save_pic(matrix, title=""):
    im = np.array(matrix / matrix.max() * 255, dtype = np.uint8)
    im_color = cv2.applyColorMap(im, cv2.COLORMAP_HOT)
    cv2.imwrite(f"tmp/{title}.png", im_color)
    wandb.save(f"tmp/{title}.png")

def parallel_hill_climber(matrix, pop_size=3, total_steps=100, master_seed=0):
    parents = {}
    parent_indices = {}
    print("init...")
    rng = np.random.default_rng(seed=master_seed)
    for i in range(pop_size):
        parents[i] = matrix.copy()
        parent_indices[i] = np.arange(matrix.shape[0])
        # start with random indices
        rng.shuffle(parent_indices[i])
        # swap according to the random indices
        parents[i] = parents[i][parent_indices[i],:]
        parents[i] = parents[i][:,parent_indices[i]]

    print("start")
    rng = np.random.default_rng(seed=master_seed)
    for step in range(total_steps):
        possible_swaps = {}
        children = {}
        children_indices = {}
        num_detected = {}
        num_swapped = {}
        fitness = {}
        for i in range(pop_size):
            gpu_random_seed = rng.integers(low=0, high=10000000)
            possible_swaps[i] = _detect_possible_swaps(parents[i], gpu_random_seed=gpu_random_seed)
            num_detected[i] = possible_swaps[i].shape[0]

        for i in range(pop_size):
            swap_random_seed = rng.integers(low=0, high=10000000)
            children[i], children_indices[i], num_swapped[i] = _apply_swaps(matrix=parents[i], indices=parent_indices[i],
                                                            detected_pairs=possible_swaps[i], seed=swap_random_seed)

        for i in range(pop_size):
            fitness[i] = -loss_gpu(children[i])

        parents = children
        parent_indices = children_indices

        _f = list(fitness.values())
        _s = list(num_swapped.values())
        _d = list(num_detected.values())
        record = {
            "step": step,
            "fitness/all": wandb.Histogram(_f),
            "fitness/max": np.max(_f),
            "fitness/mean": np.mean(_f),
            "fitness/std": np.std(_f),
            "num_swapped/all": wandb.Histogram(_s),
            "num_swapped/mean": np.mean(_s),
            "num_detected/all": wandb.Histogram(_d),
            "num_detected/mean": np.mean(_d),
        }
        wandb.log(record)
        save_pic(parents[np.argmax(_f)], f"10.0/best_step_{step:04}")
        print(f"step {step}: max fitness {np.max(_f)}")


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
    parallel_hill_climber(matrix, pop_size=args.pop_size, total_steps=args.total_steps, master_seed=args.seed)