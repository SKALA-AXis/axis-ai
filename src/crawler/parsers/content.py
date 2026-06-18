# 작성일: 2026-04-22
# 작성자: 최종민
# 변경이력:
#   2026-04-22 최종민 — Track A/B 크롤러 v4 구축 시 HTML 본문 추출기 추가
"""HTML 기사 본문 추출 — BeautifulSoup 기반."""

import logging

import httpx
from bs4 import BeautifulSoup

from src.crawler.base import RETRY_POLICY

log = logging.getLogger(__name__)

_ARTICLE_SELECTORS = [
    "article",
    '[class*="article-body"]',
    '[class*="news-content"]',
    '[class*="article_body"]',
    '[id*="article-body"]',
    "div.content",
    "div.text",
    "div.article",
]

_HEADERS = {"User-Agent": "AXIS-Crawler/1.0 (research)"}
MIN_PARAGRAPH_LENGTH = 30


async def fetch_full_content(url: str) -> str:
    """URL에서 기사 본문을 추출한다. 실패 시 빈 문자열 반환."""
    try:
        async with httpx.AsyncClient(
            timeout=RETRY_POLICY["timeout"],
            follow_redirects=True,
        ) as client:
            resp = await client.get(url, headers=_HEADERS)
            if resp.status_code != 200:
                return ""
            return _extract_text(resp.text)
    except Exception as e:
        log.debug("본문 추출 실패 | url=%s error=%s", url, e)
        return ""


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for sel in _ARTICLE_SELECTORS:
        node = soup.select_one(sel)
        if node:
            paragraphs = [p.get_text(strip=True) for p in node.find_all("p")]
            text = "\n".join(p for p in paragraphs if len(p) >= MIN_PARAGRAPH_LENGTH)
            if len(text) >= 200:
                return text

    # 폴백: body 전체 텍스트
    body = soup.find("body")
    if body:
        return body.get_text(separator=" ", strip=True)[:2000]
    return ""
