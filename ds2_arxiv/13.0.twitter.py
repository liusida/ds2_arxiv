import socket, os, re, time
import pandas as pd
import numpy as np
from ds2_arxiv.tools.my_firefox import MyFirefox
from bs4 import BeautifulSoup
import random

local_debug = socket.gethostname()!="star-lab"

def google_search(author):
    author = author.lower()
    f_author = author.replace(" ", "_")
    filename = f"data/twitter/{f_author}.txt"
    bad_filename = f"data/twitter_bad/{f_author}.html"
    if os.path.exists(filename) or os.path.exists(bad_filename):
        return
    search = author.replace(" ", "+")
    url = f"https://www.google.com/search?q={search}+Twitter+Account&ei={random.getrandbits(32)}"

    g_firefox = MyFirefox(proxy_txt_filename="config/vip.proxy.txt", proxy_disabled=local_debug)
    html = g_firefox.get(url)
    if html is None:
        print("google search> html is None")
        g_firefox.reset()
        time.sleep(10)
        return

    soup = BeautifulSoup(html, 'html.parser')
    search_result = soup.find('div', {"id":'search'})
    if search_result is None:
        print("google search> Oh, no! I've been caught!")
        g_firefox.reset()
        time.sleep(10)
        return
    
    line = search_result.find('h3', text=lambda t: t and 'Twitter' in t and re.search(r'\(@.+\)', t))
    if line:
        with open(filename, 'w') as f:
            print(author, file=f)
            print(line.text, file=f)
            print(f"wrote {filename}")
    else:
        with open(bad_filename, 'w') as f:
            print(html, file=f)
            print(f"bad {bad_filename}")

def main():
    df = pd.read_pickle("shared/arxiv_4422.pickle")
    df = df.sample(frac=1).reset_index(drop=True)

    for index, row in df.iterrows():
        title = row['title']
        first_author = row['first_author']
        last_author = row['last_author']
        other_authors = []
        _list = row['other_authors'].split(":|:")
        _list.append(first_author)
        _list.append(last_author)
        print("\n",title)
        for _author in _list:
            if _author=="":
                continue
            google_search(_author)


if __name__=="__main__":
    main()
    # google_search("tsung-hsien wen")