import pickle
import numpy as np
with open("shared/top_100.pickle", "rb") as f:
    data = pickle.load(f)

idx = np.argsort(data["cites"])[::-1]
for i in range(10):
    print(f"https://arxiv.org/abs/{data['arxiv_ids'][idx[i]]}")