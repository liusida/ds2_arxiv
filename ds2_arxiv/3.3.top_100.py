import re
import pickle
import numpy as np
with open("shared/top_100.pickle", "rb") as f:
    data = pickle.load(f)

idx = np.argsort(data["cites"])[::-1]
for i in range(100):
    j = idx[i]
    url = f"https://arxiv.org/abs/{data['arxiv_ids'][j]}"
    title = data['titles'][j]
    title = re.sub(r'\s+', r' ', title)

    print(f"{url} \t {title}")