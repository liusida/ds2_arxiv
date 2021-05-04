import glob, re
import json
import xmltodict
import pandas as pd

df = pd.DataFrame(columns=['arxiv_id', 's2', 'g'])
harvest_LG_AI = glob.glob("data/citation_compare/a/*.xml")
for filename in harvest_LG_AI:
    with open(filename, "r") as f:
        record_xml = f.read()
    record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
    arxiv_id = record_dict['id']
    title = record_dict['title']
    title = re.sub(r'\s+', ' ', title)
    year_arxiv = int(record_dict['created'][:4])
    # print(arxiv_id, end="\t")

    s2_filename = f"data/citation_compare/s2/{arxiv_id}.json"
    with open(s2_filename, "r") as f:
        s2_info = json.load(f)
    s2_citations = len(s2_info['citations'])
    year_s2 = int(s2_info['year'])
    # print("s2: ", s2_citations, end="\t")

    g_filename = f"data/citation_compare/g/{arxiv_id}.txt"
    with open(g_filename, "r") as f:
        g_info = f.read()
    g_citations = int(g_info)
    # print("g:", g_citations)

    record = {
        "arxiv_id": arxiv_id,
        "title": title,
        "year_arxiv": year_arxiv,
        "year_s2": year_s2,
        "s2": s2_citations,
        "g": g_citations
    }
    df = df.append(record, ignore_index=True)
    # break
print(df)
df.to_pickle("shared/compare_s2_g_citation.pickle")