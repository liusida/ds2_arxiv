import os, time
from urllib import request
import pandas as pd

def main():
    os.makedirs("data/pdf_4422/", exist_ok=True)
    df = pd.read_pickle("shared/arxiv_4422.pickle")
    df = df.sample(frac=1).reset_index(drop=True)

    for index, row in df.iterrows():
        arxiv_id = row['arxiv_id']
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        filename = f"data/pdf_4422/{arxiv_id}.pdf"
        if not os.path.exists(filename):
            request.urlretrieve(url, filename)
            print(f"{filename}")
        time.sleep(1)

if __name__=="__main__":
    while True:
        try:
            main()
        finally:
            pass