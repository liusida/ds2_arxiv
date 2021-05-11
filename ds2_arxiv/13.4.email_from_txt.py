import glob, os, re
from collections import defaultdict

def main():
    domains = defaultdict(lambda: 0)
    txts = glob.glob("data/pdf_txt_4422/*.txt")
    for txt in txts:
        with open(txt, "r") as f:
            content = f.read()
        re.sub(r'[\s]+', ' ', content)
        m = re.findall(r'[a-zA-Z0-9\.,{}]+@([a-zA-Z0-9\.]+)', content)
        if m:
            # print(txt)
            for address in m:
                a = address.split(".")
                str_a = ""
                about_to_break = False
                for i in range(len(a)):
                    str_a = a[len(a)-i-1] + "." + str_a
                    if about_to_break:
                        break
                    if a[len(a)-i-1]=="edu":
                        about_to_break = True
                str_a = str_a[:-1]
                domains[str_a] += 1/len(m)
            # print()
    ret = {k: v for k, v in sorted(domains.items(), key=lambda item: item[1])}
    total = sum(ret.values())

    for d, c in ret.items():
        print(f"{d} : {100.*c/total:.1f}%")

if __name__=="__main__":
    main()