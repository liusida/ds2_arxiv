import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=int, default=0, help="0-4")
parser.add_argument("-i", type=int, default=-1, help="Paper index. int. [0,4422)")
parser.add_argument("-a", type=str, default="", help="Search keywords in author. str.")
parser.add_argument("-t", type=str, default="", help="Search keywords in title. str.")

args = parser.parse_args()
seed = 1
df = pd.read_pickle("shared/compare_s2_g_citation.pickle")
indices = np.load(f"shared/author_similarity_indices_1024_50_{seed}.npy")
cos_sim_after = np.load(f"shared/cos_sim_after_1024_50_{seed}.npy")

plt.figure(figsize=[20, 20])

if args.i>=0:
    target = args.i
    query_paper = df.iloc[target]["title"]
    query_author = df.iloc[target]["first_author"]
    year_arxiv = int(df.iloc[target]["year_arxiv"])
    year_s2 = int(df.iloc[target]["year_s2"])
    if year_arxiv==year_s2:
        query_year = f"{year_arxiv}"
    else:
        query_year = f"{year_arxiv} or {year_s2}" if year_arxiv < year_s2 else f"{year_s2} or {year_arxiv}"

    index = np.argwhere(indices==target)

    cos_sim_after[index,:] = cos_sim_after.max()

    plt.title(f"Position of {query_paper}, {query_author} ({query_year})")

if args.a!="":
    author_list = df.index[(df["first_author"].str.find(args.a)!=-1) | ((df["last_author"].str.find(args.a)!=-1)) | ((df["other_authors"].str.find(f":{args.a}:")!=-1))]
    for target in author_list:
        query_paper = df.iloc[target]["title"]
        query_author = f"{df.iloc[target]['first_author']};  {df.iloc[target]['other_authors']}; {df.iloc[target]['last_author']}"
        year_arxiv = int(df.iloc[target]["year_arxiv"])
        year_s2 = int(df.iloc[target]["year_s2"])
        if year_arxiv==year_s2:
            query_year = f"{year_arxiv}"
        else:
            query_year = f"{year_arxiv} or {year_s2}" if year_arxiv < year_s2 else f"{year_s2} or {year_arxiv}"

        index = np.argwhere(indices==target)

        cos_sim_after[index,:] = cos_sim_after.max()
        print(f"hit [{target}] {query_paper} ({query_year})\n by {query_author}\n")
    plt.title(f"Search {args.a} in authors.")

plt.imshow(cos_sim_after)
plt.savefig(f"tmp_position.png")
plt.close