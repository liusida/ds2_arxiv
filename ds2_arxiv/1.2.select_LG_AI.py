import glob, os
import xmltodict

filenames = glob.glob("data/harvest/*.xml")
for filename in filenames:
    with open(filename, "r") as f:
        record_xml = f.read()
    if len(record_xml)<10:
        # must be bad record
        print(f"Bad record {filename}")
        continue
    record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
    categories = record_dict['categories']
    if categories.find("cs.LG")==-1 and categories.find("cs.AI")==-1:
        continue
    new_filename = f"./data/harvest_LG_AI/{os.path.basename(filename)}"

    # print(f"move {filename}, {new_filename}")
    os.rename(filename, new_filename)
    break