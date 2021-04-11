import csv, re, pickle, argparse
import numpy as np

top_sim_indices = np.load("shared/topic_similarity_indices.npy")

print(top_sim_indices)



# Read detailed information
with open("shared/cited_100.pickle", "rb") as f:
    data = pickle.load(f)

# Sort from highest cited to lowest
indices = np.argsort(data['cites'])[::-1][top_sim_indices]

categories = np.array(data['categories'])[indices]
titles = np.array(data['titles'])[indices]
titles = [re.sub(r'\s+', ' ', t) for t in titles]
arxiv_ids = np.array(data['arxiv_ids'])[indices]
topics = np.array(data['topic_id_lists'], dtype=object)[indices]
total_length = len(indices)

with open("shared/topic_sorted_list.txt", "w") as f:
    for i in range(total_length):
        print(titles[i], file=f)
        # print(topics[i], file=f)
