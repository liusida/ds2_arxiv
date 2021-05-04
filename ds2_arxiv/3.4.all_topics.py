import re
import pickle
import numpy as np
with open("shared/cited_100.pickle", "rb") as f:
    data = pickle.load(f)

c=0
all_topics = {}
for i, l in enumerate(data["topic_lists"]):
    abstract = data["abstracts"][i]
    abstract = re.sub(r'\s+', ' ', abstract)
    if abstract.find("$")!=-1:
        print("\n"*2)
        print(abstract)
        abstract = re.sub(r'\$([a-zA-Z0-9])\$', r'\1', abstract)
        abstract = re.sub(r'\$[^\$]+\$', '*', abstract)
        print("="*10)
        print(abstract)
        c += 1
print(c)
    # for topic in l:
    #     if topic not in all_topics:
    #         if topic.find("Bayes")!=-1:
    #             print(topic)
    #             print(abstract)
    #     all_topics[topic] = 1
        
# print(list(all_topics.keys())[:100])