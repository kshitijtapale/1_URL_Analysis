import re
from urllib.parse import urlparse
from typing import Dict
from app.exceptions import FeatureExtractionError
from app.logger import setup_logger
from googlesearch import search

logger = setup_logger(__name__)


class FeatureExtractor:
    def __init__(self):
        self.suspicious_words = ['PayPal', 'login', 'signin', 'bank', 'account',
                                 'update', 'free', 'lucky', 'service', 'bonus', 'ebayisapi', 'webscr']
        self.shortening_services = ['bit.ly', 'goo.gl', 'shorte.st', 'go2l.ink', 'x.co', 'ow.ly', 't.co', 'tinyurl', 'tr.im', 'is.gd', 'cli.gs', 'yfrog.com', 'migre.me', 'ff.im', 'tiny.cc', 'url4.eu', 'twit.ac', 'su.pr', 'twurl.nl', 'snipurl.com', 'short.to', 'BudURL.com', 'ping.fm', 'post.ly', 'Just.as', 'bkite.com', 'snipr.com', 'fic.kr', 'loopt.us', 'doiop.com', 'short.ie', 'kl.am', 'wp.me', 'rubyurl.com',
                                    'om.ly', 'to.ly', 'bit.do', 't.co', 'lnkd.in', 'db.tt', 'qr.ae', 'adf.ly', 'goo.gl', 'bitly.com', 'cur.lv', 'tinyurl.com', 'ow.ly', 'bit.ly', 'ity.im', 'q.gs', 'is.gd', 'po.st', 'bc.vc', 'twitthis.com', 'u.to', 'j.mp', 'buzurl.com', 'cutt.us', 'u.bb', 'yourls.org', 'x.co', 'prettylinkpro.com', 'scrnch.me', 'filoops.info', 'vzturl.com', 'qr.net', '1url.com', 'tweez.me', 'v.gd', 'tr.im', 'link.zip.net']

    def extract_features(self, url: str) -> Dict[str, int]:
        try:
            return {
                "use_of_ip": self._having_ip_address(url),
                #"google_index": self._google_index(url),
                "abnormal_url": self._abnormal_url(url),
                "count_dot": url.count('.'),
                "count_www": url.count('www'),
                "count_atrate": url.count('@'),
                "count_dir": self._no_of_dir(url),
                "count_embed_domain": self._no_of_embed(url),
                "short_url": self._shortening_service(url),
                "count_percentage": url.count('%'),
                "count_ques": url.count('?'),
                "count_hyphen": url.count('-'),
                "count_equal": url.count('='),
                "url_length": len(url),
                "count_https": url.count('https'),
                "count_http": url.count('http'),
                "hostname_length": self._hostname_length(url),
                "sus_url": self._suspicious_words(url),
                "fd_length": self._fd_length(url),
                "tld_length": self._tld_length(url),
                "count_digits": sum(c.isdigit() for c in url),
                "count_letters": sum(c.isalpha() for c in url),
            }
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            raise FeatureExtractionError(f"Error in feature extraction: {e}")

    # def _google_index(self, url: str) -> int:
    #     try:
    #         site = search(url, num_results=1)
    #         return 1 if next(site, None) else 0
    #     except Exception as e:
    #         logger.warning(
    #             f"Error checking Google index for {url}: {str(e)}")
    #         return 0

    def _having_ip_address(self, url: str) -> int:
        ip_pattern = r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])'
        return 1 if re.search(ip_pattern, url) else 0

    def _abnormal_url(self, url: str) -> int:
        hostname = urlparse(url).hostname
        return 1 if hostname and hostname not in url else 0

    def _no_of_dir(self, url: str) -> int:
        return urlparse(url).path.count('/')

    def _no_of_embed(self, url: str) -> int:
        return url.count('//')

    def _shortening_service(self, url: str) -> int:
        return 1 if any(service in url for service in self.shortening_services) else 0

    def _hostname_length(self, url: str) -> int:
        return len(urlparse(url).netloc)

    def _suspicious_words(self, url: str) -> int:
        return 1 if any(word in url.lower() for word in self.suspicious_words) else 0

    def _fd_length(self, url: str) -> int:
        urlpath = urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0

    def _tld_length(self, url: str) -> int:
        try:
            return len(urlparse(url).netloc.split('.')[-1])
        except:
            return -1
