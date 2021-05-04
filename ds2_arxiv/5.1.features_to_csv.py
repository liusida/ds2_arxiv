# only for export csv for js
# not used anymore
import pickle, re
import numpy as np
import torch
import torch.nn.functional as F

features = torch.load("data/features/features_bert_topics.pt") # 4422 x 768
features = features.numpy()
print(f"features.shape {features.shape}")

with open("shared/cited_100.pickle", "rb") as f:
    data = pickle.load(f)

indices = np.argsort(data['cites'])[::-1][:1000]
# print(indices.tolist())
indices = np.append(indices, [3005, 0, 3681])

# indices = np.arange(len(data['cites']))

# indices = np.array([2689,3073,1649,4275,1770,3390,3352,2780,3143,3732,, 3006, , 3682])
categories = np.array(data['categories'])[indices]
titles = np.array(data['titles'])[indices]
titles = [re.sub(r'\s+', ' ', t) for t in titles]
arxiv_ids = np.array(data['arxiv_ids'])[indices]
features = features[indices]

def save_csv(array, name):    
    csv_str = '\n'.join(map(str,array))
    with open(f"shared/{name}.csv", "w") as f:
        print(csv_str, file=f)

save_csv(arxiv_ids, "arxiv_ids")
save_csv(categories, "categories")
save_csv(titles, "titles")

features_csv_str = '\n'.join([','.join(map(str, x)) for x in features])
with open("shared/features.csv", "w") as f:
    print(features_csv_str, file=f)

# debug: check the similarity as well
# exit(0)
import matplotlib.pyplot as plt
features = torch.Tensor(features)
cos_sim = torch.nn.functional.cosine_similarity(features[:,:,None], features.t()[None,:,:])

plt.imshow(cos_sim)
plt.colorbar()
plt.savefig("tmp_cos_sim.png")

cos_sim = cos_sim.numpy()
cos_sim_csv_str = '\n'.join([','.join(map(str, x)) for x in cos_sim])
with open("shared/cos_sim_matrix.csv", "w") as f:
    print(cos_sim_csv_str, file=f)
