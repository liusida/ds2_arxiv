import csv, re, pickle, argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

# Read detailed information
with open("shared/cited_100.pickle", "rb") as f:
    data = pickle.load(f)

# Sort from highest cited to lowest
indices = np.argsort(data['cites'])[::-1]

categories = np.array(data['categories'])[indices]
titles = np.array(data['titles'])[indices]
titles = [re.sub(r'\s+', ' ', t) for t in titles]
arxiv_ids = np.array(data['arxiv_ids'])[indices]
topics = np.array(data['topic_id_lists'])[indices]
total_length = len(indices)
print(topics[0])

# to make \in operator faster, convert topics to sets
topics = [set(x) for x in topics]

# Populate similarity matrix
sim = np.zeros(shape=[total_length, total_length], dtype=np.float32)
print(sim.shape)

for i in range(total_length):
    for j in range(i):
        n1 = len(topics[i])     # number of topics in i-th paper
        n2 = len(topics[j])     # number of topics in j-th paper
        s = 0                   # number of same topics in i-th and j-th papers.
        for k in topics[i]:
            if k in topics[j]:
                s +=1
        if n1+n2==0:
            sim[i,j] = 0        # define similarity as 0 if both papers have no topics associated.
        else:
            sim[i,j] = 2*s/(n1+n2)  # similarity score \in [0,1]

sim = sim.T + sim
np.fill_diagonal(sim,1)

plt.imshow(sim)
plt.colorbar()
plt.savefig("shared/topic_similarity_matrix.png")

np.save("shared/topic_similarity_matrix.npy", sim)