import numpy as np
from numba import prange, njit
import matplotlib.pyplot as plt
import cv2
import argparse

from ds2_arxiv.tools.new_algo import *

import wandb
wandb.init(project="block_diagonal")

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_epochs", type=float, default=1e2, help="")
parser.add_argument("--epoch_steps", type=float, default=1e3, help="")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--max_int_step", type=int, default=1, help="the allowed maximal intermedia steps.")
args = parser.parse_args()
args.num_epochs = int(args.num_epochs)
args.epoch_steps = int(args.epoch_steps)
wandb.config.update(args)

@njit
def _search(elements, indices, max_int_step=1, seed=0, epoch_steps=10000):
    l = elements.shape[0]
    np.random.seed(seed)
    for step in prange(epoch_steps):
        # strategy: 
        # half the time, run for a random number of step, to mediate local optima
        # half of the time, fast single step search, to optimize efficiently.
        i,j = int(np.random.random() * l), int(np.random.random() * l)
        if step%2==0 and max_int_step>1:
            b = int(np.random.random() * max_int_step) + 1
            if i==j:
                continue
            p1 = loss(elements)
            new_elemenets = elements.copy()
            new_indices = indices.copy()
            for _ in range(b): # run for a random number of steps
                swap_inplace(new_elemenets, new_indices, i, j)
            p2 = loss(new_elemenets)
            if p2<p1:
                elements = new_elemenets
                indices  = new_indices
        else:
            if loss_gradient_if_swap(elements, i, j)>0:
                swap_inplace(elements, indices, i, j)

    return elements, indices

def search(elements, indices, max_int_step=1, seed=0, num_epochs=10, epoch_steps=100):
    for epoch in range(num_epochs):
        elements, indices = _search(elements, indices, max_int_step=max_int_step, seed=seed+epoch, epoch_steps=epoch_steps)
        wandb.log({"epoch": epoch, "loss": loss(elements)})
    return elements, indices

def save_pic(elements, title=""):
    ret = loss(elements)
    record = {
        "loss": ret,
    }
    wandb.log(record)
    print(f"loss: {ret}")
    im = np.array(elements / elements.max() * 255, dtype = np.uint8)
    im_color = cv2.applyColorMap(im, cv2.COLORMAP_HOT)
    cv2.imwrite(f"tmp/9.1.{title}.png", im_color)


random = np.random.default_rng(seed=args.seed)

if False:
    # small sample dataset:
    np.random.seed(0)
    elements = np.random.random(size=[10,10])
    elements = np.zeros([10,10])
    elements[3:5, 3:5] = 0.5
    elements[0:4, 0:4] = 0.5
    elements[1,4] = elements[4,1] =0.7
    for i in range(elements.shape[0]):
        elements[i,i] = 1
    elements = (elements+elements.T)/2 # make it a symmetric matrix
else:
    # real dataset:
    elements = np.load("shared/author_similarity_matrix.npy")
    for i in range(elements.shape[0]):
        assert(elements[i,i]==10)

elements_save_for_test = elements.copy()
indices = np.arange(elements.shape[0])
save_pic(elements, "start")
# shuffle
i = random.permutation(np.arange(elements.shape[0]))
elements = elements[i]
elements = elements[:,i]
indices = indices[i]

save_pic(elements, "shuffled")

total_step = 0
elements, indices = search(elements, indices, max_int_step=args.max_int_step, seed=args.seed, num_epochs=args.num_epochs, epoch_steps=args.epoch_steps)
total_step += args.num_epochs
save_pic(elements, f"end_n{args.num_epochs}_s{args.seed}_m{args.max_int_step}")

np.save(f"tmp/end_n{args.num_epochs}_s{args.seed}.npy", indices)
print(indices)
wandb.save(f"tmp/end_n{args.num_epochs}_s{args.seed}.npy")
wandb.save(f"tmp/9.1.end_n{args.num_epochs}_s{args.seed}_m{args.max_int_step}.png")
def test():
    elements = elements_save_for_test.copy()
    indices = np.load(f"tmp/end_n{args.num_epochs}_s{args.seed}.npy")
    elements = elements[indices, :]
    elements = elements[:, indices]
    save_pic(elements, f"test_n{args.num_epochs}_s{args.seed}")

test()