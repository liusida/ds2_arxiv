# only read results from js
# not used anymore
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

with open("shared/cos_sim_matrix.csv", "r") as f:
    cos_sim = csv.reader(f, delimiter=",")
    cos_sim = list(cos_sim)
cos_sim = np.array(cos_sim).astype(np.float32)
print(cos_sim.shape)

with open("shared/tsne-sorted.csv", "r") as f:
    content = f.read()
dots = content.split("\n")
indices = np.argsort(dots)

print(len(indices))

plt.imshow(cos_sim)
plt.colorbar()
plt.savefig("tmp_before_sort.png")
plt.close()

cos_sim = cos_sim[indices].T
cos_sim = cos_sim[indices]

plt.imshow(cos_sim)
plt.colorbar()
plt.savefig("tmp_after_sort.png")
plt.close()
