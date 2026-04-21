"""크롤러 단위 테스트"""
import pytest

from src.crawler.base_crawler import RawArticle


def test_raw_article_fields():
    article = RawArticle(
        url="https://example.com/news/1",
        title="삼성SDS AI 서비스 확대",
        content="본문 내용",
        published_at=None,
        source_name="naver_news",
        peer_id="samsung_sds",
    )
    assert article.url
    assert article.peer_id == "samsung_sds"
