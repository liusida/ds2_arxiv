import random
import urllib3
from urllib3 import ProxyManager
from urllib3.contrib.socks import SOCKSProxyManager

class MyProxy:

    def __init__(self, proxy_str="p.webshare.io:9999", proxy_disabled=False):
        self.proxy_disabled = proxy_disabled
        self.bad_proxies = []
        self.proxy_string = f"socks5://{proxy_str}"
        self.proxy = None
    def current_proxy(self, reset=False):
        if reset or self.proxy is None:
            if self.proxy_disabled:
                self.proxy = urllib3.PoolManager()
            else:
                print(f"Using proxy: {self.proxy_string}")
                self.proxy = SOCKSProxyManager(self.proxy_string)
        return self.proxy

if __name__ == "__main__":
    import socket
    local_debug = socket.gethostname()!="star-lab"
    myproxy = MyProxy(proxy_disabled=local_debug)
    url = "https://star-lab.ai/"
    url = "https://api.semanticscholar.org/v1/paper/arXiv:2103.04727"
    c = myproxy.current_proxy().request('GET', url, timeout=3).data
    print(c)