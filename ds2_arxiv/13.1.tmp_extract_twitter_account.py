import re
from bs4 import BeautifulSoup


with open('tmp/google_search_result.html', 'r') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')
line = soup.find('h3', text=lambda t: t and '| Twitterf' in t)
print(line)
matched = re.match(r'\(\@(\w+)\)', line.text)
text = "boredbengio (@boredbengio) | Twitter"
matched = re.match(r'\@', text) # question: why @ can't work?
if matched:
    print(matched[1])
