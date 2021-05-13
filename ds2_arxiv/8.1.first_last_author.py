import glob, re
import json
import xmltodict
import pandas as pd

def abbr(name):
    n = name.split(' ')
    return f"{n[0][0]}. {n[-1]}"

df = pd.DataFrame(columns=['arxiv_id', 's2'])
harvest_LG_AI = glob.glob("data/arxiv_7636/a/*.xml")
for filename in harvest_LG_AI:
    with open(filename, "r") as f:
        record_xml = f.read()
    record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
    arxiv_id = record_dict['id']
    title = record_dict['title']
    title = re.sub(r'\s+', ' ', title)
    year_arxiv = int(record_dict['created'][:4])
    other_authors = []
    if isinstance(record_dict['authors']['author'], list):
        first_k, last_k = 0, 0
        if 'forenames' in record_dict['authors']['author'][0]:
            first_author = f"{record_dict['authors']['author'][0]['forenames']} {record_dict['authors']['author'][0]['keyname']}"
        else:
            first_k = 1
            first_author = f"{record_dict['authors']['author'][0]['keyname']} {record_dict['authors']['author'][1]['keyname']}"
        if 'forenames' in record_dict['authors']['author'][-1]:
            last_author = f"{record_dict['authors']['author'][-1]['forenames']} {record_dict['authors']['author'][-1]['keyname']}"
        else:
            last_k = 1
            last_author = f"{record_dict['authors']['author'][-2]['keyname']} {record_dict['authors']['author'][-1]['keyname']}"
        if len(record_dict['authors']['author'])>2+first_k+last_k:
            for i in range(1+first_k, len(record_dict['authors']['author'])-last_k-1):
                if 'forenames' in record_dict['authors']['author'][i]:
                    author = f"{record_dict['authors']['author'][i]['forenames']} {record_dict['authors']['author'][i]['keyname']}"
                else:
                    author = f"{record_dict['authors']['author'][i]['keyname']}"
                other_authors.append(f":{author}:")
            
    else:
        first_author = last_author = f"{record_dict['authors']['author']['forenames']} {record_dict['authors']['author']['keyname']}"
    # print(arxiv_id, end="\t")

    s2_filename = f"data/arxiv_7636/s2/{arxiv_id}.json"
    with open(s2_filename, "r") as f:
        s2_info = json.load(f)
    s2_citations = len(s2_info['citations'])
    year_s2 = int(s2_info['year'])
    # print("s2: ", s2_citations, end="\t")

    # g_filename = f"data/arxiv_7636/g/{arxiv_id}.txt"
    # with open(g_filename, "r") as f:
    #     g_info = f.read()
    # g_citations = int(g_info)
    if (first_author!=s2_info['authors'][0]['name'] and abbr(first_author)!=s2_info['authors'][0]['name']):
        print(f"https://arxiv.org/abs/{arxiv_id} {first_author} v.s. {s2_info['authors'][0]['name']}")
    if (last_author!=s2_info['authors'][-1]['name'] and abbr(last_author)!=s2_info['authors'][-1]['name']):
        print(f"https://arxiv.org/abs/{arxiv_id} {last_author} v.s. {s2_info['authors'][-1]['name']}")

    # print("g:", g_citations)

    record = {
        "arxiv_id": arxiv_id,
        "title": title,
        "year_arxiv": year_arxiv,
        "year_s2": year_s2,
        "s2": s2_citations,
        # "g": g_citations,
        "first_author": first_author,
        "last_author": last_author,
        "other_authors": "|".join(other_authors),
    }
    df = df.append(record, ignore_index=True)
    # break
print(df)
df.to_pickle("shared/7636_citation.pickle")