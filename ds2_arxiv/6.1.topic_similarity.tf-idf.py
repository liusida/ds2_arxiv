from typing import List
import csv, re, pickle, argparse, copy
from collections import defaultdict
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
all_topic_count = defaultdict(lambda: 0)
all_topic_to_paper = defaultdict(lambda: [])
for i, ts in enumerate(topics):
    for t in ts:
        all_topic_count[t] += 1
        all_topic_to_paper[t].append(i)

topic_to_idx = {}
for i,t in enumerate(all_topic_count):
    topic_to_idx[t] = i

print(all_topic_count['121300'])
print(all_topic_to_paper['121300'])
print(topic_to_idx['121300'])
print(len(topic_to_idx))

# Populate similarity matrix
sim = np.zeros(shape=[total_length, total_length], dtype=np.float32)
print(sim.shape)

for i in range(total_length):
    for j in range(i):
        #try cos_sim of tf-idf
        s = []                   # number of same topics in i-th and j-th papers.
        for k in topics[i]:
            if k in topics[j]:
                s.append(all_topic_count[k])
        s = np.sum(s)
        if s==0:
            sim[i,j] = 0
        else:
            n1 = np.sum([all_topic_count[t] for t in topics[i]])
            n2 = np.sum([all_topic_count[t] for t in topics[j]])
            if n1==0 or n2==0:
                sim[i,j] = 0        # define similarity as 0 if both papers have no topics associated.
            else:
                sim[i,j] = s / (np.sqrt(n1) * np.sqrt(n2))
        


sim = sim.T + sim
np.fill_diagonal(sim,1)

plt.imshow(sim)
plt.colorbar()
plt.savefig("shared/topic_similarity_matrix_tfidf.png")

np.save("shared/topic_similarity_matrix_tfidf.npy", sim)