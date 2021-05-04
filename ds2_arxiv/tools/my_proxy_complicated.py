import random
import urllib3
from urllib3 import ProxyManager
from urllib3.contrib.socks import SOCKSProxyManager

class MyProxy:

    def __init__(self, proxy_txt_filename="proxies.txt", proxy_disabled=False):
        self.proxy_disabled = proxy_disabled
        self.bad_proxies = []
        self.proxies = []
        self.proxies_cursor = 0
        self.proxy = None
        self.proxy_count = 999 # should rotate the proxy every 90 calls
        self.proxy_txt_filename = proxy_txt_filename
        self.rotate_proxy()

    def rotate_proxy(self, report_bad_proxy=False):
        # dynamically load newest file
        with open(self.proxy_txt_filename, "r") as f:
            self.proxies = f.read().split("\n")
        try:
            with open("config/bad_proxies.txt", "r") as f:
                self.bad_proxies = f.read().split("\n")
        except FileNotFoundError:
            self.bad_proxies = []
        self.proxies_cursor = random.randrange(0, len(self.proxies))

        if report_bad_proxy and self.proxies[self.proxies_cursor]!="p.webshare.io:9999":
            with open("config/bad_proxies.txt", "a") as f:
                f.write(self.proxies[self.proxies_cursor] + "\n")
        while True:
            self.proxies_cursor = random.randrange(0, len(self.proxies))
            if self.proxies[self.proxies_cursor] not in self.bad_proxies:
                break
            print(f"Skipping bad proxy {self.proxies_cursor}: {self.proxies[self.proxies_cursor]}")

        proxy_string = f"socks5://{self.proxies[self.proxies_cursor]}"
        self.custom_proxy(proxy_string)
        return self.proxy

    def current_proxy(self):
        self.proxy_count += 1
        if self.proxy is None or self.proxy_count>90:
            self.rotate_proxy()
        return self.proxy

    def custom_proxy(self, proxy_string):
        if self.proxy_disabled:
            self.proxy = urllib3.PoolManager()
        else:
            print(f"Using proxy: {proxy_string}")
            self.proxy = SOCKSProxyManager(proxy_string)
        self.proxy_count = 0
        return self.proxy

if __name__ == "__main__":
    myproxy = MyProxy()
    url = "https://star-lab.ai/"
    c = myproxy.current_proxy().request('GET', url, timeout=3).data
    print(c)