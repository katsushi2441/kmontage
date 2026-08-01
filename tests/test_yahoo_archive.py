import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from backend.main import fetch_yahoo_news_document, yahoo_news_article_url


ARTICLE_ID = "ebb2a3771bc3481a2c62990a92afdb715b4f743e"
COMMENTS_URL = f"https://news.yahoo.co.jp/articles/{ARTICLE_ID}/comments"
ARTICLE_URL = f"https://news.yahoo.co.jp/articles/{ARTICLE_ID}"


class YahooArchiveTests(unittest.TestCase):
    def test_comments_url_is_canonicalized(self) -> None:
        self.assertEqual(yahoo_news_article_url(COMMENTS_URL), ARTICLE_URL)
        self.assertEqual(yahoo_news_article_url(ARTICLE_URL), ARTICLE_URL)
        self.assertEqual(yahoo_news_article_url("https://example.com/article"), "")

    @patch("backend.main.requests.get")
    def test_expired_article_is_recovered_from_gzipped_wayback_snapshot(self, get: Mock) -> None:
        live = Mock()
        live.raise_for_status.side_effect = requests.HTTPError("404")

        availability = Mock()
        availability.raise_for_status.return_value = None
        availability.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "status": "200",
                    "timestamp": "20260630085243",
                }
            }
        }

        html = """
        <html><head>
          <title>大丸下関店が2027年8月末に営業終了 - Yahoo!ニュース</title>
          <meta property="og:title" content="大丸下関店が2027年8月末に営業終了 - Yahoo!ニュース">
          <meta name="description" content="大丸下関店は近年、減収傾向が続いていました。">
        </head><body><main><h1>大丸下関店が営業終了</h1>
          <p>大丸下関店は2027年8月末に営業を終了します。</p>
          <p>2025年度の売上高は68億6300万円でした。</p>
          <p>1950年の開業以来、76年にわたり地域で親しまれてきました。</p>
          <p>JR下関駅前のシーモール下関で核店舗の役割を担ってきました。</p>
          <p>1959年に下関駅西口へ移り、1977年には大型商業施設の核店舗となりました。</p>
          <p>2020年に大丸松坂屋百貨店の直営店となり、全館改装も実施しました。</p>
          <p>会社は減収傾向と周辺環境の変化を慎重に検討し、営業終了を決めました。</p>
          <p>街のランドマークとして親しまれた百貨店の終了は、地域の大きな節目になります。</p>
          <p>営業終了までの期間は、利用客や周辺店舗への影響も注目されます。</p>
        </main></body></html>
        """
        snapshot = Mock()
        snapshot.raise_for_status.return_value = None
        snapshot.content = gzip.compress(html.encode("utf-8"))
        snapshot.encoding = "utf-8"
        get.side_effect = [live, availability, snapshot]

        with tempfile.TemporaryDirectory() as tmp:
            meta, text = fetch_yahoo_news_document(COMMENTS_URL, Path(tmp))

            self.assertEqual(meta["extractor"], "wayback_yahoo_news")
            self.assertEqual(meta["original_url"], COMMENTS_URL)
            self.assertEqual(meta["archive_timestamp"], "20260630085243")
            self.assertIn("68億6300万円", text)
            self.assertTrue((Path(tmp) / "wayback_source.json").exists())
            self.assertTrue((Path(tmp) / "wayback_snapshot.html").exists())

        self.assertEqual(get.call_args_list[0].args[0], ARTICLE_URL)


if __name__ == "__main__":
    unittest.main()
