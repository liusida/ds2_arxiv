import numpy as np
import pandas as pd

df = pd.read_pickle("shared/compare_s2_g_citation.pickle")
indices = np.load("shared/author_similarity_indices.npy")

df = df.iloc[indices[-10:]]
print(df)