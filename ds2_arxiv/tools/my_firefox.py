from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.proxy import Proxy, ProxyType
from base64 import b64encode
# star-lab 13.58.77.175
class MyFirefox():
    def __init__(self, proxy_txt_filename="proxies.txt", proxy_disabled=False):
        self.proxy_disabled = proxy_disabled
        self.proxy_txt_filename = proxy_txt_filename
        self.browser = None
        self.reset()

    def reset(self):
        if self.browser is not None:
            self.browser.close()
        opts = FirefoxOptions()
        opts.add_argument("--headless")
        profile = webdriver.FirefoxProfile()
        if not self.proxy_disabled:
            with open(self.proxy_txt_filename, "r") as f:
                proxy = f.read().strip()
            ip, port = proxy.split(':')
            profile.set_preference('network.proxy.type', 1)
            profile.set_preference('network.proxy.socks', ip)
            profile.set_preference('network.proxy.socks_port', int(port))
        self.browser = webdriver.Firefox(profile, options=opts)

    def get(self, url):
        try:
#            print(f"MyFirefox> get> getting {url} using proxy {proxy}...")
            self.browser.get(url)
        except Exception as e:
            print(f"MyFirefox> get> Error: {e}")
            return None
        html = self.browser.page_source
        return html
    def __del__(self):
        self.browser.close()


if __name__=="__main__":
    g_firefox = MyFirefox()

    url = f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C46&q=arxiv+2004.04946&btnG="
    html = g_firefox.get(url)
    print(html)
    if html.find("recaptcha")!=-1:
        print("Oh, no! I've been caught!")
    print("done")
