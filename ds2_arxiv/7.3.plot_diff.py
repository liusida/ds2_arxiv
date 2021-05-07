import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle("shared/compare_s2_g_citation.pickle")

df["weight"] = df["g"] + df["s2"]

df = df.sort_values(by="weight", ascending=False)
# print(df.head(100))

x = np.arange(df.shape[0])+1
plt.figure(figsize=[8,4])
plt.scatter(x, df["s2"], alpha=0.3, s=1, label="Semantic Scholar")
plt.scatter(x, df["g"], alpha=0.3, s=1, label="Google Scholar")
plt.plot(x, df["weight"]/2, alpha=0.7, c="green", label="Mean")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Rank by Mean")
plt.ylabel("Number of Citations")
plt.legend()
plt.savefig("shared/citation_number_distribution.pdf")