# This is too slow...

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pydot

df = pd.read_pickle("shared/compare_s2_g_citation.pickle")
source, target, weight = [],[],[]

df2 = df.groupby(['last_author']).size().sort_values(ascending=False)
print(df2.shape)
df2 = df2[df2>1]
print(df2.shape)

for author in df2.index:
    df3 = df[df['last_author']==author]
    df4 = df3['arxiv_id']
    # print(df4.shape)
    for index, value in df4.iteritems():
        for index_1, value_1 in df4.iteritems():
            if index<index_1:
                source.append(value)
                source.append(value_1)
                target.append(value_1)
                target.append(value)
                weight.append(1)
                weight.append(1)
print(len(source))
df_edge = pd.DataFrame({'source': source,
                        'target': target,
                        'weight': weight,
                        })
paper_network = nx.from_pandas_edgelist(df_edge, edge_attr=True)
# nx.draw(paper_network)
# plt.savefig("tmp_network.png")
graph = nx.drawing.nx_pydot.to_pydot(paper_network)

graph.write_png('tmp_network.png')