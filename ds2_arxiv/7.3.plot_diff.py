import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle("shared/compare_s2_g_citation.pickle")

df["weight"] = df["g"] + df["s2"]

df = df.sort_values(by="weight", ascending=False)
# print(df.head(100))

x = np.arange(df.shape[0])
plt.plot(x, df["s2"], alpha=0.3, label="s2")
plt.plot(x, df["g"], alpha=0.3, label="google")
plt.plot(x, df["weight"]/2, alpha=0.7, label="weight")
plt.yscale("log")
plt.legend()
plt.savefig("tmp_weighted.png")