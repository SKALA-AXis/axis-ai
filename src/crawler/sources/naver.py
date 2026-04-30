"""Naver News API 크롤러 (Tier 1) — REST API + 본문 수집, 1시간 간격."""

import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from src.crawler.base import (
    RETRY_POLICY,
    BaseCrawler,
    DailyLimitGuard,
    RawArticle,
)

log = logging.getLogger(__name__)

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"

# 본문 수집용 — 동시 요청 수 제한 (예의 있는 크롤링)
_BODY_FETCH_CONCURRENCY = 5
_BODY_FETCH_TIMEOUT = 6.0
_BODY_MIN_LEN = 200  # 이 미만이면 description으로 폴백

# 매체별 본문 셀렉터 (n.news.naver.com + 주요 언론사)
_BODY_SELECTORS: list[str] = [
    "#dic_area",  # n.news.naver.com
    "article#dic_area",
    "div#articeBody",  # 일부 네이버 구판
    "div#articleBodyContents",
    "div.newsct_article",  # 네이버 섹션
    "div#article-view-content-div",  # 디지털데일리·바이라인 등
    "div.article_body",
    "div#article_body",
    "div.article-body",
    "div#newsEndContents",
    "div.news_end",
    "section.article_view",
    "div.view_con",
    "div.view_txt",
    "div#content",
]

_BODY_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}


class NaverNewsCrawler(BaseCrawler):
    def __init__(
        self,
        peer_id: str,
        keywords: list[str],
        limit_guard: Optional[DailyLimitGuard] = None,
    ) -> None:
        super().__init__(peer_id, limit_guard)
        self.keywords = keywords
        self.client_id = os.getenv("NAVER_CLIENT_ID", "")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "")

    async def crawl(self) -> list[RawArticle]:
        if not self.client_id:
            log.warning("NAVER_CLIENT_ID 미설정. 크롤링 스킵.")
            return []
        articles: list[RawArticle] = []
        for keyword in self.keywords:
            if not self.limit_guard.allow("naver_news"):
                break
            try:
                articles.extend(await self._fetch(keyword))
            except Exception as e:
                log.error("네이버 크롤링 실패 | keyword=%s error=%s", keyword, e)

        # 본문 수집 — 검색 결과 전체에 대해 병렬 수행
        enriched = await _enrich_with_bodies(articles)
        log.info(
            "네이버 뉴스 수집 | peer=%s raw=%d enriched_count=%d",
            self.peer_id,
            len(articles),
            sum(1 for a in enriched if len(a.content or "") >= _BODY_MIN_LEN),
        )
        return enriched

    async def _fetch(self, keyword: str) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=RETRY_POLICY["timeout"]) as client:
            resp = await client.get(
                NAVER_API_URL,
                params={"query": keyword, "display": 20, "sort": "date"},
                headers={
                    "X-Naver-Client-Id": self.client_id,
                    "X-Naver-Client-Secret": self.client_secret,
                },
            )
            if self._is_blocked(resp.status_code):
                log.warning("네이버 API 접근 차단 | status=%d", resp.status_code)
                return []
            resp.raise_for_status()
            items = resp.json().get("items", [])
            return [
                RawArticle(
                    url=item["link"],
                    title=_strip_html(item["title"]),
                    content=_strip_html(item.get("description", "")),
                    published_at=_parse_naver_date(item.get("pubDate", "")),
                    source_name="naver_news",
                    peer_id=self.peer_id,
                )
                for item in items
            ]


# ── 본문 수집 ────────────────────────────────────────────────────


async def _enrich_with_bodies(articles: list[RawArticle]) -> list[RawArticle]:
    """각 기사의 link에서 본문을 가져와 content를 대체한다.

    실패하면 description(API snippet)을 그대로 유지한다.
    동시 요청 수는 _BODY_FETCH_CONCURRENCY로 제한.
    """
    if not articles:
        return articles

    sem = asyncio.Semaphore(_BODY_FETCH_CONCURRENCY)
    async with httpx.AsyncClient(
        timeout=_BODY_FETCH_TIMEOUT,
        headers=_BODY_FETCH_HEADERS,
        follow_redirects=True,
    ) as client:
        tasks = [_enrich_one(client, sem, a) for a in articles]
        await asyncio.gather(*tasks, return_exceptions=True)
    return articles


async def _enrich_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    article: RawArticle,
) -> None:
    async with sem:
        try:
            resp = await client.get(article.url)
            if resp.status_code != 200:
                return
            body = _extract_body(resp.text, article.url)
            if body and len(body) >= _BODY_MIN_LEN:
                article.content = body
        except Exception as e:
            log.debug("본문 수집 실패 | url=%s error=%s", article.url, e)


def _extract_body(html: str, url: str) -> str:
    """여러 셀렉터 시도 → readability 폴백."""
    soup = BeautifulSoup(html, "html.parser")
    for sel in _BODY_SELECTORS:
        node = soup.select_one(sel)
        if node:
            text = node.get_text(separator="\n", strip=True)
            if len(text) >= _BODY_MIN_LEN:
                return text[:5000]

    # readability-lxml 폴백
    try:
        from readability import Document

        doc = Document(html)
        text = BeautifulSoup(doc.summary(), "html.parser").get_text(separator="\n", strip=True)
        if len(text) >= _BODY_MIN_LEN:
            return text[:5000]
    except Exception:
        pass

    # 최후 폴백: 가장 큰 <p> 블록 묶음
    body = soup.find("body")
    if body:
        text = body.get_text(separator=" ", strip=True)
        return text[:3000]
    return ""


def _strip_html(text: str) -> str:
    """<b> 태그 및 &quot; 등 HTML 엔티티 제거."""
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
    )
    return text.strip()


def _parse_naver_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S +0900")
    except ValueError:
        return None
