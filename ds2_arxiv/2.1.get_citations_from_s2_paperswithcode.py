import os, time, re, glob, sys, shutil, random
import urllib
from collections import defaultdict
import json

# # import wandb
import socket
local_debug = socket.gethostname()!="star-lab"

from ds2_arxiv.tools.my_proxy import MyProxy
myproxy = MyProxy(proxy_disabled=local_debug)

os.makedirs("data/citations_s2_paperswithcode", exist_ok=True)
with open(f"shared/s2_bad_citation_paperswithcode.txt", "a"):
    pass

def get_remote_content_through_a_proxy(url, time_sleep=0.1):
    """ 
    Return bytes.
    Return None if something is wrong. 
    """
    r = None
    for retry in range(1):
        try:
            time.sleep(time_sleep)
            print(f"requesting {url}")
            r = myproxy.current_proxy().request('GET', url, timeout=3).data
        except Exception as e:
            print(f"HTTPError: {e}")
            # myproxy.rotate_proxy(report_bad_proxy=False)    
    return r

def main():
    global g_error, g_count, g_source, g_source_total

    with open(f"shared/s2_bad_citation_paperswithcode.txt", "r") as f:
        _c = f.readlines()
    bad_citation = []
    for __c in _c:
        bad_citation.append(__c.strip())

    with open('data/papers-with-abstracts.json', 'rb') as f:
        papers = json.load(f)

    g_source_total = len(papers)
    g_source = 0

    for paper in papers:
        g_source += 1
        if g_source%1000==0:
            print(f"{g_source}/{g_source_total} processed. ")

        arxiv_id = paper['arxiv_id']
        if arxiv_id is None:
            continue
        # print(f"arxiv_id: {arxiv_id}")
        match = re.search(r'[0-9]+\.[0-9]+', arxiv_id)
        if match:
            arxiv_id = match.group(0)
            get_citation_url = f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"
            old_path = f"data/citations_s2/{arxiv_id}.json"
            path = f"data/citations_s2_paperswithcode/{arxiv_id}.json"
            if arxiv_id in bad_citation:
                continue
            if os.path.exists(path):
                if os.stat(path).st_size>100:
                    # already cached
                    continue
            # copy from April, 2021:
            if os.path.exists(old_path):
                shutil.copy(old_path, path)
                print("copied.")
                continue
            r = get_remote_content_through_a_proxy(get_citation_url, time_sleep=1)
            if r is None:
                # get remote content error
                g_error += 1
                continue
            data = json.loads(r)
            if 'citations' not in data:
                print(f"Error: {r}\n arxiv_id: {arxiv_id}")
                # myproxy.rotate_proxy(report_bad_proxy=False)
                if r.decode('utf-8').find("Forbidden")!=-1:
                    myproxy.current_proxy(reset=True)
                    time.sleep(10)
                if r.decode('utf-8').find("Paper not found")!=-1:
                    with open(f"shared/s2_bad_citation_paperswithcode.txt", "a") as f:
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
        g_count = len(glob.glob("data/citations_s2_paperswithcode/*.json"))
        print(f"g_count: {g_count}")
    # wandb.log({
    #     "count": g_count,
    #     "error": g_error,
    #     "source_processed": g_source,
    #     "source_total": g_source_total,
    # })

if __name__=="__main__":
    # wandb.init(project="get_citation_s2")
    
    g_count = len(glob.glob("data/citations_s2_paperswithcode/*.json"))
    g_error = 0
    g_source = 0
    g_source_total = 0

    print(f"Start with g_count={g_count}")
    # wandb.log({
    #     "count": g_count,
    #     "error": g_error,
    # })
    while True:
        try:
            main()
        except Exception as e:
            print(f"Error: {e}")
        print(f"Start over again.")
        time.sleep(10)
