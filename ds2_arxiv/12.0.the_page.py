import numpy as np
import pandas as pd

filename = "shared/best/seed_51_step_0199"
png_filename = f"{filename}.png"

indices = np.load(f"{filename}_indicies.npy")
matrix = np.load("shared/author_similarity_matrix.npy")
df = pd.read_pickle("shared/compare_s2_g_citation.pickle")
df = df.iloc[indices]
df = df.reset_index(drop=True)
print(df.columns)
with open("shared/the_page.html", "w") as f:
    for index, row in df.iterrows():
        all_authors = authors = row['first_author']
        if row['other_authors']!="":
            authors += ", ..."
            all_authors += ", " + ", ".join(row['other_authors'].split(":|:"))[1:-2]
        if row['first_author']!=row['last_author']:
            authors += ", "+row['last_author']
            all_authors += ", "+row['last_author']

        year = int(row['year_s2'])
        if int(row['year_arxiv'])>year:
            year = f"{year} or {int(row['year_arxiv'])}"
        elif int(row['year_arxiv'])<year:
            year = f"{int(row['year_arxiv'])} or {year}"

        tooltip = f"""<p>{row['title']}</p>
        <p><i>{all_authors}</i></p>
        <p>Year {year}</p>
        <p>Cited by {row['s2']} or {row['g']}</p>
        """
        tooltip = tooltip.replace('\'', '`')
        print(f"<div class='paper' index='{index}' arxiv_id='{row['arxiv_id']}' tooltip='{tooltip}'>{row['title']} ({authors}, {int(row['year_arxiv'])})</div>", file=f)

matrix = matrix[indices,:]
matrix = matrix[:,indices]

import shutil
shutil.copy2(png_filename, f"shared/bg.png")
# print(png_filename)
if False:
    import matplotlib.pyplot as plt
    plt.imshow(matrix)
    plt.show()

