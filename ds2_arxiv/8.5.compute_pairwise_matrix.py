import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle("shared/compare_s2_g_citation.pickle")
matrix = np.zeros([df.shape[0], df.shape[0]])

df_last_author = pd.unique(df['last_author'])
df_first_author = pd.unique(df['first_author'])
df_authors = np.concatenate([ df_first_author,  df_last_author])
df_authors = pd.unique(df_authors)
print(df_authors)

def activate_matrix(matrix, df_authors, df):
    """
    First consideration: main author relationship = 10
    Then consider: main other author relationship = 5
    Last consider: other author relationship = 2
    """
    for author in df_authors:
        df_main_author = df[(df['last_author']==author) | (df['first_author']==author)]
        df_other_author = df[df['other_authors'].str.find(f":{author}:")!=-1]

        for i in df_other_author.index:
            for j in df_other_author.index:
                matrix[i,j] = 2

        for i in df_main_author.index:
            for j in df_other_author.index:
                matrix[i,j] = 5
                matrix[j,i] = 5

        for i in df_main_author.index:
            for j in df_main_author.index:
                matrix[i,j] = 10

    return matrix

matrix = activate_matrix(matrix, df_authors, df)

np.save("shared/author_similarity_matrix.npy", matrix)
