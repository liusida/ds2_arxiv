import os, glob, json, shutil, argparse
import xmltodict
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=int, default=100)
args = parser.parse_args()

source_arxiv_folder = f"data/harvest_202105"
source_s2_folder = f"data/citations_s2_202105"
arxiv_folder = f"data/arxiv_may/0.arxiv"
s2_folder = f"data/arxiv_may/1.s2"

os.makedirs(arxiv_folder, exist_ok=True)
os.makedirs(s2_folder, exist_ok=True)

filenames = glob.glob(f"{source_arxiv_folder}/*.xml")
l = len(filenames)
for i, filename in enumerate(filenames):
    if i%10000==0:
        print(f"iter {i}/{l}")
    with open(filename, "r") as f:
        record_xml = f.read()
    if len(record_xml)<10:
        # must be bad record
        print(f"Bad record {filename}")
        continue
    record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
    arxiv_id = filename.split("/")[-1].split(".xml")[0]

    s2_filename = f"{source_s2_folder}/{arxiv_id}.json"
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
        dest_path = f"{arxiv_folder}/{arxiv_id}.xml"
        if not os.path.exists(dest_path):
            print(f"copy to {dest_path}")
            shutil.copy(filename, dest_path)
        s2_dest_path = f"{s2_folder}/{arxiv_id}.json"
        if not os.path.exists(s2_dest_path):
            print(f"copy to {s2_dest_path}")
            shutil.copy(s2_filename, s2_dest_path)

