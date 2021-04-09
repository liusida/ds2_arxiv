import re
import pickle
import numpy as np
with open("shared/top_100.pickle", "rb") as f:
    data = pickle.load(f)

idx = np.argsort(data["cites"])[::-1]
with open("shared/top_100.txt", "w") as f:
    for i in range(100):
        j = idx[i]
        url = f"https://arxiv.org/abs/{data['arxiv_ids'][j]}"
        title = data['titles'][j]
        title = re.sub(r'\s+', r' ', title)
        year = data['years'][j]
        cite = data['cites'][j]
        created_date = data['created_dates'][j]
        author = data['authors'][j]
        print(f"{url} \t {title} ({author}, {year})", file=f) 
        print(f"start at: {created_date}, cited by {cite}", file=f)
        print("", file=f)