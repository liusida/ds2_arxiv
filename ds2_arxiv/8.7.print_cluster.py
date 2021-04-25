import numpy as np
import pandas as pd

df = pd.read_pickle("shared/compare_s2_g_citation.pickle")

indices = {}
for seed in range(5):
    indices[seed] = np.load(f"shared/author_similarity_indices_1024_50_{seed}.npy")

# print(indices[0][:10])
polar = {}
for seed in range(1,5):
    # if seed==2:
    #     continue
    print(f"\n== seed = {seed} ==")
    for n in range(500):
        target = indices[0][n]
        pos = np.argwhere(indices[seed]==target)
        print(int(pos[0,0]<2000), end=", ")
        # if not seed in polar:
        #     if pos<2000:
        #         polar[seed] = 0
        #     else:
        #         polar[seed] = 1
        # else:
        #     if pos<2000:
        #         assert polar[seed] == 0
        #     else:
        #         assert polar[seed] == 1

df_head = df.iloc[indices[0][:20]]
print(df_head[['first_author','last_author']])


df_tail = df.iloc[indices[0][-20:]]
print(df_tail[['first_author','last_author']])