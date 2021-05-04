import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle("shared/compare_s2_g_citation.pickle")

df["weight"] = df["g"] + df["s2"]

df = df.sort_values(by="weight", ascending=False)
df = df.head(100)

with open("shared/top_100.html", "w") as f:
    print("<ul>", file=f)
    for i in range(100):
        arxiv_id = df.iloc[i]["arxiv_id"]
        title = df.iloc[i]["title"]
        year_arxiv = int(df.iloc[i]["year_arxiv"])
        year_s2 = int(df.iloc[i]["year_s2"])
        if year_arxiv==year_s2:
            year = f"{year_arxiv}"
        else:
            year = f"{year_arxiv} or {year_s2}" if year_arxiv < year_s2 else f"{year_s2} or {year_arxiv}"
        print(f"<li><a href='https://arxiv.org/abs/{arxiv_id}'>{title}</a> ({year})</li>", file=f)
    print("</ul>", file=f)
    
