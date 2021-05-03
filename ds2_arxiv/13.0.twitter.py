import socket, os, re
import pandas as pd
import numpy as np
from ds2_arxiv.tools.my_firefox import MyFirefox
from bs4 import BeautifulSoup

local_debug = socket.gethostname()!="star-lab"

def google_search(author):
    author = author.lower()
    filename = f"data/twitter/{author}.txt"
    if os.path.exists(filename):
        return
    search = author.replace(" ", "+")
    url = f"https://www.google.com/search?q={search}+Twitter+Account"

    g_firefox = MyFirefox(proxy_txt_filename="config/vip.proxy.txt", proxy_disabled=local_debug)
    html = g_firefox.get(url)
    soup = BeautifulSoup(html, 'html.parser')

    line = soup.find('h3', text=lambda t: t and '| Twitter' in t)
    if line:
        with open(filename, 'w') as f:
            print(author, file=f)
            print(line.text, file=f)
            print(f"wrote {filename}")

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