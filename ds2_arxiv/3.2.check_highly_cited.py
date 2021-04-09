import glob, argparse, os, json
import xmltodict

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=int, default=100)
args = parser.parse_args()

filenames = glob.glob(f"data/harvest_LG_AI_{args.threshold}/*.xml")
for filename in filenames:
    with open(filename, "r") as f:
        record_xml = f.read()
    if len(record_xml)<10:
        # must be bad record
        print(f"Bad record {filename}")
        continue
    record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
    arxiv_id = filename.split("/")[-1].split(".xml")[0]
    arxiv_id_1 = record_dict["id"]
    title = record_dict["title"]
    created_date  = record_dict["created"]

    assert arxiv_id==arxiv_id_1, f"arxiv id error: {filename}"

    s2_filename = f"data/citations_s2/{arxiv_id}.json"
    if not os.path.exists(s2_filename):
        # print(f"Error: {s2_filename} doesn't exist.")
        continue
    if os.stat(s2_filename).st_size<10:
        print(f"Error: empty file. {s2_filename}")
        continue
    with open(s2_filename, "r") as f:
        s2_info = json.load(f)
    num_citations = len(s2_info['citations'])
    print(f"{title} ({created_date}) Cited by {num_citations}")