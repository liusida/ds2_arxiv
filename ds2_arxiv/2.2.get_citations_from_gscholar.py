import os, time, re, glob, sys
import feedparser
import urllib
from collections import defaultdict
import json
import xmltodict # to convert the raw metadata from xml format to dict
import wandb
import socket
local_debug = socket.gethostname()!="star-lab"

from ds2_arxiv.tools.my_firefox import MyFirefox
g_firefox = MyFirefox(proxy_txt_filename="config/vip.proxy.txt", proxy_disabled=local_debug)

def get_citation_count(arxiv_id):
    ret = None
    url = f"https://scholar.google.com/scholar?hl=en&as_sdt=1&q=arxiv+{arxiv_id}"

    # g_firefox.reset()
    html = g_firefox.get(url)
    if html is None:
        return None
    if html.find(arxiv_id)==-1:
        if html.find("your computer or network may be sending automated queries")!=-1:
            print("get_citation_count> Oh, been punished!")
        else:
            print("get_citation_count> Other Error. error html outputted in tmp/error.html")
            with open("tmp/error.html", "w") as f:
                print(html, file=f)
        g_firefox.reset()
        time.sleep(10)
        return None
    if html.find("recaptcha")!=-1:
        print("get_citation_count> Oh, no! I've been caught!")
        g_firefox.reset()
        time.sleep(10)
        return None

    m = re.search(r'Cited by ([0-9]+)</a>', html)
    if m:
        ret = int(m.group(1))
    else:
        ret = 0
    return ret


def main():
    global g_error, g_count, g_source, g_source_total

    filenames = sorted(glob.glob("data/harvest_LG_AI/*.xml"))[::-1]
    g_source_total = len(filenames)
    g_source = 0

    for filename in filenames:
        time.sleep(0.1)
        g_source += 1
        if g_source%10000==0:
            print(f"main> {g_source}/{g_source_total} processed. ")
        sys.stdout.flush()
        with open(filename, "r") as f:
            record_xml = f.read()
        if len(record_xml)<10:
            # must be bad record
            print(f"main> Bad record {filename}")
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
            files =  glob.glob(f"data/citations_gscholar/{arxiv_id}:*.txt")
            if len(files)>0:
                # file exists.
                continue
            count = get_citation_count(arxiv_id)
            print(f"main> {arxiv_id}: {count}")
            if count is None:
                # get remote content error
                g_error += 1
                print("main> Error")
                continue
            path = f"data/citations_gscholar/{arxiv_id}:{count}.txt"
            with open(path, "w") as f:
                f.write(f"{count}")
            g_count += 1
        log()        


g_n_calls = 0
def log():
    global g_count, g_n_calls
    g_n_calls += 1
    if g_n_calls%100==0:
        g_count = len(glob.glob("data/citations_gscholar/*.txt"))
    wandb.log({
        "count": g_count,
        "error": g_error,
        "source_processed": g_source,
        "source_total": g_source_total,
    })

if __name__=="__main__":   
    wandb.init(project="get_citation_google_scholar")
    
    g_count = len(glob.glob("data/citations_gscholar/*.txt"))
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
            print(f"global> Error: {e}")
        print(f"global> Start over again.")
        time.sleep(10)
