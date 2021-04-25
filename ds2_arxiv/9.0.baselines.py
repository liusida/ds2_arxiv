import numpy as np
from scipy.spatial.distance import pdist
from seriate import seriate
import matplotlib.pyplot as plt
import cv2
from numpy.random import default_rng
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from numba import jit

@jit(nopython=True)
def loss(elements):
    """ what about directly optimize this loss function? """
    ret = 0
    l = elements.shape[0]
    _l_1 = 1.0/l
    for i in range(l):
        for j in range(i):
            if elements[i,j]>0:
                ret += (i-j) * _l_1
    return ret

def save_pic(elements, title=""):
    plt.figure(figsize=[20,20])
    ret = loss(elements)
    plt.title(f"loss: {ret}")
    plt.imshow(elements)
    plt.savefig(f"tmp/{title}.png")
    plt.close

# ShowCase I: tsp is not good for clustering.
# elements = np.zeros([1000, 1000])
# rng = default_rng(seed=0)
# current_pos = 0
# for i in range(1000):
#     elements[i,i] = 10
#     n = rng.integers(low=1, high=50)
#     n = int(1.03**n)
#     if current_pos+n>=elements.shape[0]:
#         break
#     elements[current_pos:current_pos+n, current_pos:current_pos+n] = 10
#     current_pos += n
# elements[450:480, 450:480] = 10
# elements[470:490, 470:490] = 10

# for i in range(100000):
#     n = rng.integers(low=0, high=elements.shape[0])
#     m = rng.integers(low=0, high=elements.shape[0])
#     if 950<(1000-n)+m<1050:
#         elements[n,m] = 10
#         elements[m,n] = 10

# save_pic(elements, "original")

# ShowCase II: tsp is good for recovering images:
elements = cv2.imread("shared/demo/zimo.jpg", flags=cv2.IMREAD_COLOR)
elements = elements[:,:,0]
rng = default_rng(seed=1)
i = rng.permutation(np.arange(elements.shape[0]))
# j = rng.permutation(np.arange(elements.shape[0]))
elements = elements[i]
elements = elements[:,i]


# elements = np.load("author_similarity_matrix.npy")
print(elements.shape)

save_pic(elements, "randomized")



for i in range(1):
    Z = hierarchy.ward(elements)
    indices = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, elements))

    elements = elements[indices].T
    elements = elements[indices].T

    save_pic(elements, f"processed_{i}_olo")


    pca = PCA(n_components=1)
    pca.fit(elements)
    print(pca.components_.shape)
    indices = np.argsort(pca.components_.flatten())

    elements = elements[indices].T
    elements = elements[indices].T

    save_pic(elements, f"processed_{i}_pca")

    perplexity = 128 # for picture of zimo
    # perplexity = 1024
    learning_rate = 50
    verbose = 1
    seed = 0

    ret = TSNE(n_components=1, perplexity=perplexity, learning_rate=learning_rate, n_iter=3000, verbose=verbose, random_state=i).fit(elements)
    indices = np.argsort(ret.embedding_.flatten())
    s_score = 2 * ret.kl_divergence_ + np.log(elements.shape[0]) * perplexity / elements.shape[0]
    print(f"s_score {s_score}")

    elements = elements[indices].T
    elements = elements[indices].T

    save_pic(elements, f"processed_{i}_tsne")

    indices = seriate(pdist(elements), approximation_multiplier=2000, timeout=0)

    elements = elements[indices].T
    elements = elements[indices].T

    save_pic(elements, f"processed_{i}_tsp")


