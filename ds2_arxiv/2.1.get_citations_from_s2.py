import os, time, re, glob, sys
import feedparser
import urllib
from collections import defaultdict
import json
import xmltodict # to convert the raw metadata from xml format to dict

import wandb
import socket
local_debug = socket.gethostname()!="star-lab"

from ds2_arxiv.tools.my_proxy import MyProxy
myproxy = MyProxy(proxy_txt_filename="config/vip.proxy.txt", proxy_disabled=local_debug)

def get_remote_content_through_a_proxy(url, time_sleep=0.1):
    """ 
    Return bytes.
    Return None if something is wrong. 
    """
    r = None
    for retry in range(3):
        try:
            time.sleep(time_sleep)
            r = myproxy.current_proxy().request('GET', url, timeout=3).data
        except Exception as e:
            print(f"HTTPError: {e}")
            myproxy.rotate_proxy(report_bad_proxy=False)    
    return r

def main():
    global g_error, g_count, g_source, g_source_total

    with open(f"shared/s2_bad_citation.txt", "r") as f:
        _c = f.readlines()
    bad_citation = []
    for __c in _c:
        bad_citation.append(__c.strip())

    # filenames = sorted(glob.glob("data/harvest_LG_AI/*.xml"))[::-1]
    filenames = sorted(glob.glob("data/harvest_LG_AI/1807.02110.xml"))[::-1]
    g_source_total = len(filenames)
    g_source = 0

    for filename in filenames:
        g_source += 1
        if g_source%1000==0:
            print(f"{g_source}/{g_source_total} processed. ")
        sys.stdout.flush()
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
        arxiv_id = record_dict['id']
        match = re.search(r'^(.+)v[0-9]+$', arxiv_id)
        if match:
            arxiv_id = match.group(1)
        if arxiv_id.find(".")!=-1:
            get_citation_url = f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"
            path = f"data/citations_s2/{arxiv_id}.json"
            if arxiv_id in bad_citation:
                continue
            if os.path.exists(path):
                if os.stat(path).st_size>100:
                    # already cached
                    continue
            r = get_remote_content_through_a_proxy(get_citation_url, time_sleep=0)
            if r is None:
                # get remote content error
                g_error += 1
                continue
            data = json.loads(r)
            if 'citations' not in data:
                print(f"Error: {r}\n arxiv_id: {arxiv_id}")
                myproxy.rotate_proxy(report_bad_proxy=False)
                if r.decode('utf-8').find("Forbidden")!=-1:
                    time.sleep(10)
                if r.decode('utf-8').find("Paper not found")!=-1:
                    with open(f"shared/s2_bad_citation.txt", "a") as f:
                        print(arxiv_id, file=f)
                    g_error += 1
                continue
            num_citation = len(data['citations'])

            print(arxiv_id, ":", num_citation)
            with open(path, "wb") as f:
                f.write(r)
            g_count += 1
        log()

g_n_calls = 0
def log():
    global g_count, g_n_calls
    g_n_calls += 1
    if g_n_calls%100==0:
        g_count = len(glob.glob("data/citations_s2/*.json"))
    wandb.log({
        "count": g_count,
        "error": g_error,
        "source_processed": g_source,
        "source_total": g_source_total,
    })

if __name__=="__main__":
    wandb.init(project="get_citation")
    
    g_count = len(glob.glob("data/citations_s2/*.json"))
    g_error = 0
    g_source = 0
    g_source_total = 0

    print(f"Start with g_count={g_count}")
    wandb.log({
        "count": g_count,
        "error": g_error,
    })
    while True:
        try:
            main()
        except Exception as e:
            print(f"Error: {e}")
        print(f"Start over again.")
        time.sleep(10)
