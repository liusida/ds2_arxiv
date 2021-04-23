import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle("shared/compare_s2_g_citation.pickle")


# print(df["first_author"])
df1 = df.groupby(['first_author']).size().sort_values(ascending=False)
print(df1.head(20))

df2 = df.groupby(['last_author']).size().sort_values(ascending=False)
print(df2.head(20))
print(df2['Geoffrey Hinton'])

