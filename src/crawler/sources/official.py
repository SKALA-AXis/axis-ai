"""삼성SDS·LG CNS 공식 뉴스룸 크롤러 (Tier 1) — Playwright(SDS) / REST API(LG CNS), 6시간 간격."""

import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from src.crawler.base import (
    RETRY_POLICY,
    SOURCE_CREDIBILITY,
    BaseCrawler,
    DailyLimitGuard,
    RawArticle,
)
from src.crawler.playwright_client import PlaywrightClient

log = logging.getLogger(__name__)

_SDS_URL = "https://www.samsungsds.com/kr/news/news.html"
_SDS_BASE = "https://www.samsungsds.com"

# LG CNS: fingerprint → fetch API 패턴
_LGCNS_FP_URL = (
    "https://www.lgcns.com/bin/cf/fetch/fingerprint?cfDirectoryPath=%2Fnewsroom&locale=ko"
)
_LGCNS_FETCH_URL = "https://www.lgcns.com/bin/cf/fetch"
_LGCNS_BASE = "https://www.lgcns.com"
_LGCNS_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.lgcns.com/kr/newsroom/press.html",
}


class OfficialNewsroomCrawler(BaseCrawler):
    """삼성SDS(Playwright) + LG CNS(REST API) 공식 뉴스룸 크롤링."""

    def __init__(
        self,
        peer_id: str,
        limit_guard: Optional[DailyLimitGuard] = None,
    ) -> None:
        super().__init__(peer_id, limit_guard)
        self.pw = PlaywrightClient()

    async def crawl(self) -> list[RawArticle]:
        if self.peer_id not in ("samsung_sds", "lg_cns"):
            return []
        if not self.limit_guard.allow("official"):
            return []
        try:
            if self.peer_id == "samsung_sds":
                return await self._fetch_sds()
            return await self._fetch_lgcns()
        except Exception as e:
            log.error("공식 뉴스룸 크롤링 실패 | peer_id=%s error=%s", self.peer_id, e)
            return []

    async def _fetch_sds(self) -> list[RawArticle]:
        html = await self.pw.fetch_html(_SDS_URL, wait_selector=".board_list, .news_list, article")
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        articles: list[RawArticle] = []
        seen: set[str] = set()
        for a_tag in soup.select("a[href*='/news/']")[:30]:
            href = a_tag.get("href", "")
            if not href or href in seen:
                continue
            seen.add(href)
            if not href.startswith("http"):
                href = urljoin(_SDS_BASE, href)
            title = a_tag.get_text(strip=True).replace("자세히 보기", "").strip()
            if not title:
                parent = a_tag.parent
                title = (
                    parent.get_text(strip=True).replace("자세히 보기", "").strip()[:120]
                    if parent
                    else ""
                )
            if len(title) < 5:
                continue
            articles.append(
                RawArticle(
                    url=href,
                    title=title,
                    content="",
                    source_tier=1,
                    source_name="samsung_sds_newsroom",
                    peer_id="samsung_sds",
                    credibility_score=SOURCE_CREDIBILITY["samsung_sds_newsroom"],
                    published_at=_parse_sds_url_date(href),
                )
            )
        return articles

    async def _fetch_lgcns(self) -> list[RawArticle]:
        async with httpx.AsyncClient(
            timeout=RETRY_POLICY["timeout"], follow_redirects=True
        ) as client:
            # 1단계: fingerprint 획득
            r_fp = await client.get(_LGCNS_FP_URL, headers=_LGCNS_HEADERS)
            fp = r_fp.json().get("fingerprint", "") if r_fp.status_code == 200 else ""
            if not fp:
                log.warning("LG CNS fingerprint 획득 실패")
                return []

            # 2단계: 뉴스 목록 fetch
            r = await client.get(
                _LGCNS_FETCH_URL,
                params={
                    "cfDirectoryPath": "/newsroom",
                    "page": 1,
                    "itemsPerPage": 15,
                    "locale": "ko",
                    "fp": fp,
                    "sortField": "date",
                    "latestCount": 4,
                    "filterFieldName": "category",
                    "filterFieldValue": "press",
                    "isApplyFilterToLatestList": "true",
                },
                headers=_LGCNS_HEADERS,
            )
            if r.status_code != 200:
                log.warning("LG CNS 뉴스 API 실패 | status=%d", r.status_code)
                return []

            items = r.json().get("list", [])
            articles = []
            for item in items:
                title = item.get("jcrTitle", "")
                jcr_name = item.get("jcrName", "")
                link = item.get("link") or (
                    f"{_LGCNS_BASE}/kr/newsroom/press/detail.{jcr_name}" if jcr_name else ""
                )
                if not title or not link:
                    continue
                if not link.startswith("http"):
                    link = urljoin(_LGCNS_BASE, link)
                date_str = item.get("date", "")
                published_at = _parse_iso(date_str)
                articles.append(
                    RawArticle(
                        url=link,
                        title=title,
                        content=item.get("_summary", ""),
                        source_tier=1,
                        source_name="lg_cns_newsroom",
                        peer_id="lg_cns",
                        credibility_score=SOURCE_CREDIBILITY["lg_cns_newsroom"],
                        published_at=published_at,
                    )
                )
            return articles


def _parse_sds_url_date(url: str) -> Optional[datetime]:
    """삼성SDS 뉴스룸 URL 슬러그에서 날짜 추출. 예: kkr-260415.html → 2026-04-15"""
    m = re.search(r"-(\d{6})\.html", url)
    if not m:
        return None
    s = m.group(1)  # e.g. "260415"
    try:
        year = 2000 + int(s[:2])
        return datetime(year, int(s[2:4]), int(s[4:6]))
    except ValueError:
        return None


def _parse_iso(date_str: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str[:26], fmt)
        except ValueError:
            continue
    return None
