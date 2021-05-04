import glob, os, shutil
import xmltodict

harvest_LG_AI = glob.glob("data/harvest_LG_AI_100/*")

i=0
for filename in harvest_LG_AI:
    with open(filename, "r") as f:
        record_xml = f.read()
    if len(record_xml)<10:
        # must be bad record
        print(f"Bad record {filename}")
        continue
    record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
    arxiv_id = record_dict['id']
    if not arxiv_id in filename:
        print(filename)
        break
    shutil.copy(filename, f"data/citation_compare/a/{arxiv_id}.xml")
    shutil.copy(f"data/citations_s2/{arxiv_id}.json", f"data/citation_compare/s2/{arxiv_id}.json")
    g = glob.glob(f"data/citations_gscholar/{arxiv_id}:*.txt")
    shutil.copy(g[0], f"data/citation_compare/g/{arxiv_id}.txt")

    if i%10==0:
        print(f"{i} {arxiv_id}")
    i+=1
