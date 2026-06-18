# 작성일: 2026-04-22
# 작성자: 최종민
# 변경이력:
#   2026-04-22 최종민 — Track A/B 크롤러 v4 구축
#   2026-04-30 박지원 — 크롤러 v1 반영
"""공식 뉴스룸 크롤러 (Tier 1) — 4 peer.

- samsung_sds:      Playwright HTML 파싱 (URL 슬러그 패턴)
- lg_cns:           내부 fingerprint REST API
- hyundai_autoever: Playwright + 코퍼레이트 도메인 list 페이지 (generic)
- posco_dx:         Playwright + 코퍼레이트 도메인 list 페이지 (generic)
"""

import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from src.crawler.base import (
    RETRY_POLICY,
    BaseCrawler,
    DailyLimitGuard,
    RawArticle,
)
from src.crawler.playwright_client import PlaywrightClient

log = logging.getLogger(__name__)

_SDS_URL = "https://www.samsungsds.com/kr/news/index.html"
_SDS_BASE = "https://www.samsungsds.com"
# 기사 URL 패턴: /kr/news/<slug>-<YYMMDD>.html
_SDS_ARTICLE_RE = re.compile(r"^/kr/news/[a-z0-9]+-\d{6}\.html$")

# Generic 뉴스룸 (현대오토에버·포스코DX) — 코퍼레이트 도메인 + 베스트-에포트 list 페이지.
# 정확한 셀렉터 검증은 W7 후속작업. 동작 실패 시 graceful skip.
_GENERIC_SOURCES: dict[str, dict[str, str | list[str]]] = {
    "hyundai_autoever": {
        "list_url": "https://www.hyundai-autoever.com/kr/about/news",
        "base": "https://www.hyundai-autoever.com",
        "source_name": "hyundai_autoever_newsroom",
        "list_selectors": [
            ".news-list a",
            ".board-list a",
            "ul.list li a",
            "table tbody tr a",
            "a[href*='/news/']",
            "a[href*='/notice']",
        ],
    },
    "posco_dx": {
        "list_url": "https://www.poscodx.com/kor/news/news.do",
        "base": "https://www.poscodx.com",
        "source_name": "posco_dx_newsroom",
        "list_selectors": [
            ".news-list a",
            ".board-list a",
            "ul.list li a",
            "table tbody tr a",
            "a[href*='news.do']",
            "a[href*='/news/']",
        ],
    },
}

# Generic 뉴스룸 크리덴셜 (코퍼레이트 사이트는 Tier 1)
_SDS_BODY_SELECTORS: list[str] = [
    "div.news-detail__content",
    "div.news-detail",
    "article.news",
    "div.content-detail",
    "div#contents",
    "article",
]

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


def _attr_str(value: object) -> str:
    """BeautifulSoup get()이 돌려주는 Union 값을 안전하게 str로 환원."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return value[0] if value else ""
    return ""


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
        if not self.limit_guard.allow("official"):
            return []
        try:
            if self.peer_id == "samsung_sds":
                return await self._fetch_sds()
            if self.peer_id == "lg_cns":
                return await self._fetch_lgcns()
            if self.peer_id in _GENERIC_SOURCES:
                return await self._fetch_generic(_GENERIC_SOURCES[self.peer_id])
            return []
        except Exception as e:
            log.error("공식 뉴스룸 크롤링 실패 | peer_id=%s error=%s", self.peer_id, e)
            return []

    async def _fetch_sds(self) -> list[RawArticle]:
        html = await self.pw.fetch_html(_SDS_URL, wait_selector="a[href*='/news/']")
        if not html:
            log.warning("Samsung SDS 뉴스룸 HTML 비어있음 | url=%s", _SDS_URL)
            return []
        soup = BeautifulSoup(html, "html.parser")

        # href별로 title 우선순위: "자세히 보기" 제외 + 5자 이상 텍스트
        href_title: dict[str, str] = {}
        for a_tag in soup.select("a[href*='/news/']"):
            href = _attr_str(a_tag.get("href", ""))
            if not _SDS_ARTICLE_RE.match(href):
                continue
            text = a_tag.get_text(strip=True).replace("자세히 보기", "").strip()
            if len(text) < 5:
                continue
            prev = href_title.get(href, "")
            if len(text) > len(prev):
                href_title[href] = text

        articles: list[RawArticle] = []
        for href, title in href_title.items():
            full_url = urljoin(_SDS_BASE, href)
            articles.append(
                RawArticle(
                    url=full_url,
                    title=title,
                    content="",
                    source_name="samsung_sds_newsroom",
                    peer_id="samsung_sds",
                    published_at=_parse_sds_url_date(href),
                )
            )

        # 본문 병렬 수집 (httpx 단순 fetch — SPA 아니므로 렌더링 불필요)
        await _enrich_sds_bodies(articles)

        # 본문 없는 기사 (외부 언론보도 스텁 페이지) 제외
        before = len(articles)
        articles = [a for a in articles if len(a.content or "") >= 200]

        log.info(
            "Samsung SDS 뉴스룸 | html_len=%d raw=%d with_body=%d",
            len(html),
            before,
            len(articles),
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
                        source_name="lg_cns_newsroom",
                        peer_id="lg_cns",
                        published_at=published_at,
                    )
                )
            return articles

    async def _fetch_generic(self, conf: dict[str, str | list[str]]) -> list[RawArticle]:
        """현대오토에버·포스코DX 코퍼레이트 뉴스룸 — Playwright + 다중 셀렉터 시도.

        list_selectors 중 매칭되는 첫번째로 article 링크 추출, readability로 본문 보강.
        URL/셀렉터는 best-effort — 작동 안 하면 빈 list 반환 + warning.
        """
        list_url = str(conf["list_url"])
        base = str(conf["base"])
        source_name = str(conf["source_name"])
        selectors = list(conf["list_selectors"])

        html = ""
        for sel in selectors:
            html = await self.pw.fetch_html(list_url, wait_selector=sel)
            if html:
                break
        if not html:
            log.warning(
                "Generic 뉴스룸 HTML 비어있음 | peer=%s url=%s",
                self.peer_id,
                list_url,
            )
            return []

        soup = BeautifulSoup(html, "html.parser")
        seen: set[str] = set()
        candidates: list[tuple[str, str]] = []
        for sel in selectors:
            for a in soup.select(sel):
                href = _attr_str(a.get("href")).strip()
                title = a.get_text(strip=True)
                if not href or len(title) < 5 or href in seen:
                    continue
                if href.startswith("#") or href.startswith("javascript:"):
                    continue
                full = href if href.startswith("http") else urljoin(base, href)
                if not full.startswith(base):
                    continue
                seen.add(href)
                candidates.append((full, title))

        # 상위 30개로 제한 (코퍼레이트 nav 링크 노이즈 방어)
        candidates = candidates[:30]
        articles: list[RawArticle] = [
            RawArticle(
                url=url,
                title=title,
                content="",
                source_name=source_name,
                peer_id=self.peer_id,
            )
            for url, title in candidates
        ]

        await _enrich_sds_bodies(articles)  # 동일한 readability+selector 폴백 재사용

        # 본문 200자 이상만 통과 (메뉴·footer 링크 노이즈 제거)
        before = len(articles)
        articles = [a for a in articles if len(a.content or "") >= 200]
        log.info(
            "Generic 뉴스룸 | peer=%s html_len=%d candidates=%d with_body=%d",
            self.peer_id,
            len(html),
            before,
            len(articles),
        )
        return articles


async def _enrich_sds_bodies(articles: list[RawArticle]) -> None:
    """Samsung SDS 기사 본문 병렬 수집."""
    import asyncio

    if not articles:
        return
    sem = asyncio.Semaphore(4)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    async with httpx.AsyncClient(timeout=6.0, headers=headers, follow_redirects=True) as client:

        async def fetch_one(art: RawArticle) -> None:
            async with sem:
                try:
                    r = await client.get(art.url)
                    if r.status_code != 200:
                        return
                    soup = BeautifulSoup(r.text, "html.parser")
                    for sel in _SDS_BODY_SELECTORS:
                        node = soup.select_one(sel)
                        if node:
                            text = node.get_text(separator="\n", strip=True)
                            if len(text) >= 200:
                                art.content = text[:5000]
                                return
                    # readability 폴백
                    try:
                        from readability import Document

                        doc = Document(r.text)
                        text = BeautifulSoup(doc.summary(), "html.parser").get_text(
                            separator="\n", strip=True
                        )
                        if len(text) >= 200:
                            art.content = text[:5000]
                    except Exception:
                        pass
                except Exception as e:
                    log.debug("SDS 본문 수집 실패 | url=%s error=%s", art.url, e)

        await asyncio.gather(*[fetch_one(a) for a in articles], return_exceptions=True)


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
