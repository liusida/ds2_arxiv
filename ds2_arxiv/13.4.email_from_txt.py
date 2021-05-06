import glob, os, re

def main():
    txts = glob.glob("data/pdf_txt_4422/*.txt")
    for txt in txts:
        print(txt)
        with open(txt, "r") as f:
            content = f.read()
        m = re.findall(r'[a-zA-Z0-9\.,{}]+@[a-zA-Z0-9\.]+\.[a-zA-Z0-9]+', content)
        for address in m:
            print(address)
        print()


if __name__=="__main__":
    main()