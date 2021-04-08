import socket
local_debug = socket.gethostname()!="star-lab"

from ds2_arxiv.tools.my_proxy import MyProxy
myproxy = MyProxy(proxy_disabled=local_debug)
url = "https://api.semanticscholar.org/v1/paper/arXiv:2103.04727"
r = myproxy.current_proxy().request('GET', url, timeout=3).data
print(r)