import glob, re

accounts = []

filenames = glob.glob("data/twitter/*.txt")
for filename in filenames:
    with open(filename, "r") as f:
        content = f.read()
    m = re.findall(r'@[^\s\)]+', content)
    for _m in m:
        if "@gmail.com" in _m:
            continue
        if not re.search(r'[^@a-zA-Z0-9_]+', _m):
            # print(content, ">>>", _m)
            accounts.append(_m)

with open("shared/researchers_twitter_accounts.txt", "w") as f:
    f.write("\n".join(accounts))

