import glob, os, json
from collections import defaultdict
import matplotlib.pyplot as plt

citation_dist = defaultdict(lambda: 0)

s2_filenames = glob.glob("data/citations_s2/*.json")
for s2_filename in s2_filenames:
    if os.stat(s2_filename).st_size<10:
        print(f"Error: empty file. {s2_filename}")
        continue
    with open(s2_filename, "r") as f:
        s2_info = json.load(f)
    num_citations = len(s2_info['citations'])
    citation_dist[num_citations] += 1

x = list(citation_dist.keys())
y = list(citation_dist.values())

plt.figure(figsize=[8,4])
plt.scatter(x,y,s=1)
plt.xscale("log")
plt.yscale("log")
plt.xlim((0.5,1e5))
plt.ylim((0.5,1e5))
plt.xlabel("Number of Citations")
plt.ylabel("Number of Papers")
plt.tight_layout()
plt.savefig("tmp/s2_dist.pdf")
plt.savefig("tmp/s2_dist.png")

with open("tmp/s2_dist.txt", "w") as f:
    print(citation_dist, file=f)