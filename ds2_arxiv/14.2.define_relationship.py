# Step 2
# Define pairwise relationships between papers
# Here we define the relationship in author

import pandas as pd
import numpy as np

target_folder = "arxiv_may"

def step2():
    df = pd.read_pickle(f"data/{target_folder}.pickle")
    matrix = np.zeros([df.shape[0], df.shape[0]])

    df_last_author = pd.unique(df['last_author'])
    df_first_author = pd.unique(df['first_author'])
    
    main_authors = np.concatenate([ df_first_author,  df_last_author])
    print(f"main author examples: {main_authors[:10]}")

    other_authors = {}
    for index, row in df.iterrows():
        _l = row['other_authors'].split(":|:")
        for _a in _l:
            if _a=="":
                continue
            other_authors[_a] = 1
    other_authors = list(other_authors.keys())
    print(f"other author examples: {other_authors[:10]}")
    ar_authors = np.concatenate([main_authors, other_authors])

    ar_authors = pd.unique(ar_authors)
    print(f"There are {ar_authors.shape[0]} unique authors.")

    def activate_matrix(matrix, ar_authors, df):
        """
        First consideration: main author relationship = 1.0
        Then consider: main other author relationship = 0.5
        Last consider: other author relationship = 0.3
        """
        for i, author in enumerate(ar_authors):
            if i%100==0:
                print(f"author {i}: {author}")
            df_main_author = df[(df['last_author']==author) | (df['first_author']==author)]
            df_other_author = df[df['other_authors'].str.find(f":{author}:")!=-1]

            for i in df_other_author.index:
                for j in df_other_author.index:
                    if matrix[i,j]<0.3:
                        matrix[i,j] = 0.3
                        matrix[j,i] = 0.3

            for i in df_main_author.index:
                for j in df_other_author.index:
                    if matrix[i,j]<0.5:
                        matrix[i,j] = 0.5
                        matrix[j,i] = 0.5

            for i in df_main_author.index:
                for j in df_main_author.index:
                    matrix[i,j] = 1
                    matrix[j,i] = 1

        return matrix

    print("Processing...")
    matrix = activate_matrix(matrix, ar_authors, df)

    for i in range(matrix.shape[0]):
        assert(matrix[i,i]==1)
        
    np.save(f"data/matrix_{target_folder}.npy", matrix)
    print("done")
    
if __name__=="__main__":
    step2()
