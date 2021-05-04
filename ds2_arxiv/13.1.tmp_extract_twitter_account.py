import re
from bs4 import BeautifulSoup


with open('tmp/google_search_result.html', 'r') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')
line = soup.find('h3', text=lambda t: t and '| Twitter' in t)
print(line)
matched = re.search(r'\((@\w+)\)', line.text)
if matched:
    print(matched[1])
