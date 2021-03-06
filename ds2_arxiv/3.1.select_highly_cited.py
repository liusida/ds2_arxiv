import os, glob, json, shutil, argparse
import xmltodict
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=int, default=100)
args = parser.parse_args()

os.makedirs(f"data/harvest_LG_AI_{args.threshold}", exist_ok=True)
os.makedirs(f"data/citations_s2_{args.threshold}", exist_ok=True)

filenames = glob.glob("data/harvest_LG_AI/*.xml")
# filenames = sorted(glob.glob("data/harvest_LG_AI/1807.02110.xml"))[::-1]
l = len(filenames)
for i, filename in enumerate(filenames):
    with open(filename, "r") as f:
        record_xml = f.read()
    if len(record_xml)<10:
        # must be bad record
        print(f"Bad record {filename}")
        continue
    record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
    arxiv_id = filename.split("/")[-1].split(".xml")[0]

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
    if num_citations>=args.threshold:
        dest_path = f"data/harvest_LG_AI_{args.threshold}/{arxiv_id}.xml"
        if not os.path.exists(dest_path):
            print(f"copy {arxiv_id}")
            shutil.copy(filename, dest_path)
        s2_dest_path = f"data/citations_s2_{args.threshold}/{arxiv_id}.json"
        if not os.path.exists(s2_dest_path):
            print(f"copy {arxiv_id}")
            shutil.copy(s2_filename, s2_dest_path)