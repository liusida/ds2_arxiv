import time
import socket
local_debug = socket.gethostname()!="star-lab"
import wandb
wandb.init(project="get_citation_s2")

from ds2_arxiv.tools.my_proxy import MyProxy
myproxy = MyProxy(proxy_disabled=local_debug)

def get_remote_content_through_a_proxy(url, time_sleep=0.1):
    """ 
    Return bytes.
    Return None if something is wrong. 
    """
    r = None
    for retry in range(1):
        try:
            time.sleep(time_sleep)
            print(f"requesting {url}")
            r = myproxy.current_proxy().request('GET', url, timeout=3).data
        except Exception as e:
            print(f"HTTPError: {e}")
            # myproxy.rotate_proxy(report_bad_proxy=False)    
    return r

url = "https://api.semanticscholar.org/v1/paper/arXiv:2103.04727"
r = get_remote_content_through_a_proxy(url)
print(r)