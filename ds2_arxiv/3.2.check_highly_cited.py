import glob
import argparse
import os
import json
import pickle
from typing import OrderedDict
import xmltodict

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=int, default=100)
args = parser.parse_args()

arxiv_ids = []
created_dates = []
cites = []
titles = []
abstracts = []
authors = []
years = []
categories = [] # 0 for none, 1 for LG, 2 for AI, 3 for both (think of a binary number XX)
topic_lists = [] # a list of list
topic_id_lists = [] # a list of list 

filenames = glob.glob(f"data/harvest_LG_AI_{args.threshold}/*.xml")
for filename in filenames:
    with open(filename, "r") as f:
        record_xml = f.read()
    if len(record_xml) < 10:
        # must be bad record
        print(f"Bad record {filename}")
        continue
    record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
    arxiv_id = filename.split("/")[-1].split(".xml")[0]
    arxiv_id_1 = record_dict["id"]
    title = record_dict["title"]
    abstract = record_dict["abstract"]
    created_date = record_dict["created"]
    author_list = record_dict["authors"]["author"]
    category_str = record_dict["categories"]
    category = 0
    if category_str.find("cs.LG"):
        category+=2
    if category_str.find("cs.AI"):
        category+=1

    if isinstance(author_list, OrderedDict):
        first_author = author_list
    else:
        first_author = author_list[0]
    first_author = first_author["keyname"]
    assert arxiv_id == arxiv_id_1, f"arxiv id error: {filename}"

    s2_filename = f"data/citations_s2_{args.threshold}/{arxiv_id}.json"
    if not os.path.exists(s2_filename):
        # print(f"Error: {s2_filename} doesn't exist.")
        continue
    if os.stat(s2_filename).st_size < 10:
        print(f"Error: empty file. {s2_filename}")
        continue
    with open(s2_filename, "r") as f:
        s2_info = json.load(f)
    num_citations = len(s2_info['citations'])
    year = s2_info['year']
    topic_list = []
    topic_id_list = []
    for s in s2_info['topics']:
        topic_list.append(s["topic"])
        topic_id_list.append(s["topicId"])

    # print(f"{title} ({created_date}) Cited by {num_citations}")
    arxiv_ids.append(arxiv_id)
    created_dates.append(created_date)
    cites.append(num_citations)
    titles.append(title)
    abstracts.append(abstract)
    authors.append(first_author)
    years.append(year)
    categories.append(category)
    topic_lists.append(topic_list)
    topic_id_lists.append(topic_id_list)

obj = {
    "arxiv_ids": arxiv_ids, 
    "created_dates": created_dates, 
    "cites": cites,
    "titles": titles,
    "abstracts": abstracts,
    "authors": authors,
    "years": years,
    "categories": categories,
    "topic_lists": topic_lists,
    "topic_id_lists": topic_id_lists,
}
with open(f"shared/cited_{args.threshold}.pickle", "wb") as f:
    pickle.dump(obj, f)
