import pickle, json, re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def write_details():
    # Read detailed information
    with open("shared/cited_100.pickle", "rb") as f:
        data = pickle.load(f)

    # Sort from highest cited to lowest
    indices = np.argsort(data['cites'])[::-1]

    categories = np.array(data['categories'])[indices]
    titles = np.array(data['titles'])[indices]
    titles = [re.sub(r'\s+', ' ', t) for t in titles]
    arxiv_ids = np.array(data['arxiv_ids'])[indices]
    topics = np.array(data['topic_id_lists'])[indices]
    topics_str = np.array(data['topic_lists'])[indices]
    total_length = len(indices)
    print(topics[0])
    obj = {
        "titles": titles,
        "topics": topics.tolist(),
        "categories": categories.tolist(),
        "arxiv_ids": arxiv_ids.tolist(),
    }
    topic_counts = defaultdict(lambda: 0)
    for ts in topics_str:
        for t in ts:
            topic_counts[t] += 1
    topic_counts = {k: v for k, v in sorted(topic_counts.items(), key=lambda item: item[1])}

    with open("shared/cited_100.json", "w") as f:
        json.dump(obj, fp=f)
    with open("shared/topic_counts.json", "w") as f:
        json.dump(topic_counts, fp=f)

    return titles, topics_str.tolist()
titles, topics = write_details()

plot = False

colors = np.ones(shape=[4422, 3]) * 0.5
colors[:,0] = np.arange(0,1,step=1/4422)[::-1] # red-ish: top, blue-ish: bottom

width = 16
height = 14
scale = 2
for b in range(60,61):
    with open(f"tmp/tsne_data/{b*50}.pickle", "rb") as f:
        p, kld, grad_norm = pickle.load(f)
    X_embedded = p.reshape(4422, 2)
    if plot:
        plt.figure(figsize=[width, height])
        ax = plt.gca()
        ax.set_xlim([-scale*width,scale*width])
        ax.set_ylim([-scale*height,scale*height])
        ax.scatter(X_embedded[:,0], X_embedded[:,1], s=0.2, c=colors)
        ax.set_title(f"Step: {b*50}, error(KLD): {kld:.04f}, grad: {grad_norm:.04f}")
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"Step: {b*50}, error(KLD): {kld:.04f}, grad: {grad_norm:.04f}")
        plt.savefig(f"tmp/tsne_data/{b*50}.png")
        plt.close()


    # save JSON
    xs = X_embedded[:,0].tolist()
    ys = X_embedded[:,1].tolist()
    nodes = []
    for i in range(4422):
        nodes.append({
            'data': {
                'id': i,
                'title': titles[i],
                'topic': topics[i],
            },
            'position': {
                'x': xs[i],
                'y': ys[i],
            },
            'locked': True,
        })
    # to make \in operator faster, convert topics to sets
    topics = [set(x) for x in topics]
    edges = []
    # for i in range(4422):
    #     for j in range(i):
    #         for t in topics[i]:
    #             if t in topics[j]:
    #                 edges.append({
    #                     'data': {
    #                         'source': i,
    #                         'target': j,
    #                     }
    #                 })
    obj = {
        'nodes': nodes,
        'edges': edges,
    }
    with open(f"tmp/tsne_data/final.json", "w") as f:
        json.dump(obj, fp=f)
