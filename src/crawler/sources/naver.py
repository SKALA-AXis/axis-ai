"""네이버 뉴스 API 크롤러.

DB 저장 없이 로컬에서 단독 실행해 네이버 뉴스 수집 결과를 확인할 수 있다.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.companies import COMPANY_ALIASES  # noqa: E402
from src.config.sectors import SECTOR_KEYWORDS, match_sector_details  # noqa: E402
from src.crawler.base import DailyLimitGuard, RawArticle  # noqa: E402
from src.crawler.base_crawler import BaseCrawler  # noqa: E402
from src.crawler.parsers.article_content import (  # noqa: E402
    extract_body_text,
    extract_image_urls,
    extract_subtitle,
    normalize_title_text,
)

log = logging.getLogger(__name__)

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"
INDUSTRY_NEWS_SOURCE_NAME = "naver_industry_news"
INDUSTRY_TREND_COMPANY = "industry_trend"
NAVER_MAX_DISPLAY = 100
NAVER_MAX_START = 1000
NAVER_CREDENTIAL_RETRY_STATUS_CODES = {401, 403, 429}
PEER_ALIASES = COMPANY_ALIASES
DEFAULT_PEER_IDS = tuple(PEER_ALIASES.keys())
DEFAULT_MAX_RESULTS = DailyLimitGuard.SOURCE_TYPE_LIMITS["news"]
_SEARCH_ALIAS_BLOCKLIST = {
    "sk",
    "skinc.",
    "sk주식회사",
}
_POSCO_DX_GROUP_SEARCH_ALIASES = [
    "포스코 AX",
    "포스코 AI 에이전트",
    "포스코 피지컬 AI",
    "포스코 스마트팩토리",
    "포스코 자동화",
    "포스코 디지털 전환",
]
_POSCO_DX_GROUP_RELEVANCE_RE = re.compile(
    r"포스코.{0,80}(AX|AI\s*에이전트|피지컬\s*AI|스마트\s*팩토리|"
    r"자동화|디지털\s*전환|업무\s*혁신|제조\s*AI|수주|영업|품질)"
    r"|"
    r"(AX|AI\s*에이전트|피지컬\s*AI|스마트\s*팩토리|자동화|"
    r"디지털\s*전환|업무\s*혁신|제조\s*AI).{0,80}포스코",
    re.IGNORECASE,
)
_NON_BUSINESS_PATH_SEGMENTS = {
    "sports",
    "esports",
    "entertainment",
    "celebrity",
    "kpop",
    "movie",
    "music",
    "lifestyle",
}
_NON_BUSINESS_TERMS = (
    "girl group",
    "aespa",
    "karina",
    "faker",
    "esports legend",
    "k-pop",
    "kpop",
    "celebrity",
)

_INDUSTRY_NEWS_QUERY_GROUPS: tuple[dict[str, str], ...] = (
    {"sector": "ax", "query": "AI 경쟁 전력 데이터 인프라"},
    {"sector": "ax", "query": "AI 거버넌스 에이전트 권한 관리"},
    {"sector": "ax", "query": "AX 산업 전환 기업들"},
    {"sector": "ax", "query": "AI 도입 확산 기업 업무 방식"},
    {"sector": "security", "query": "SW 공급망 보안 로드맵"},
    {"sector": "security", "query": "공급망 공격 기업 보안"},
    {"sector": "security", "query": "제로트러스트 공급망 보안"},
    {"sector": "security", "query": "AI 보안 표준 정책"},
    {"sector": "infra", "query": "AI 인프라 전력 데이터 제도"},
    {"sector": "infra", "query": "AI 데이터센터 전력 인프라"},
    {"sector": "infra", "query": "GPU 클라우드 인프라 산업"},
    {"sector": "infra", "query": "AI 인프라 투자 전력망"},
)

_INDUSTRY_DIRECTION_TERMS = (
    "산업",
    "시장",
    "업계",
    "기업들",
    "정부",
    "공공",
    "정책",
    "규제",
    "로드맵",
    "표준",
    "제도",
    "확산",
    "수요",
    "전망",
    "인프라",
    "전력",
    "데이터",
    "공급망",
    "리스크",
    "위협",
    "거버넌스",
    "권한",
    "관리",
    "대응",
    "전환",
)
_INDUSTRY_STRONG_FRAME_RE = re.compile(
    r"(AI\s*경쟁.{0,80}(전력|데이터|인프라|제도)|"
    r"(전력|데이터|인프라|제도).{0,80}AI\s*경쟁|"
    r"공급망.{0,40}(보안|공격|로드맵|SBOM|AIBOM)|"
    r"(로드맵|표준|규제|정책|제도화).{0,40}(보안|AI|AX|공급망)|"
    r"(거버넌스|권한\s*관리).{0,60}(AI\s*에이전트|생성형\s*AI))",
    re.IGNORECASE,
)
_COMPANY_EVENT_TITLE_TERMS = (
    "출범",
    "선정",
    "공급",
    "파트너",
    "협약",
    "수주",
    "대표",
    "회장",
    "출시",
    "상장",
    "매출",
    "신용등급",
    "계약",
    "투자 확대",
    "맡는다",
)
_THIRD_PARTY_TITLE_TERMS = (
    "업스테이지",
    "롯데",
    "삼성",
    "삼성전자",
    "카카오",
    "네이버",
    "KT",
    "SKT",
    "SK텔레콤",
    "LG유플러스",
    "한컴",
    "더존비즈온",
    "솔트룩스",
    "엘리스그룹",
    "NC AI",
    "다올티에스",
    "파이오링크",
    "KX넥스지",
    "핸디소프트",
    "텐센트",
    "넷앱",
    "스플렁크",
    "스패로우",
    "카스퍼스키",
    "에임인텔리전스",
    "NIA",
    "한국지능정보사회진흥원",
    "아케이드 AI",
    "통신 3사",
)
_INDUSTRY_NEWS_TITLE_BLOCK_TERMS = (
    "[GAM]",
    "스테이블코인",
    "크립토",
    "전기차 캐즘",
    "워크숍",
    "세미나",
    "컨퍼런스",
    "에너지밸리포럼",
    "경제발전공유사업",
    "해외진출 지원",
    "KSP",
    "공모",
    "정책 제안 공모전",
    "보험감독국",
    "[데스크가 만났습니다]",
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}


class NaverCredential(NamedTuple):
    client_id: str
    client_secret: str


class NaverApiCredentialExhaustedError(RuntimeError):
    """모든 네이버 API 키가 인증/권한/쿼터 문제로 실패한 경우."""


class NaverNewsCrawler(BaseCrawler):
    def __init__(
        self,
        peer_id: str,
        aliases: list[str],
        search_queries: list[dict[str, str]] | None = None,
        display: int = 100,
        max_results: int = 0,
        cutoff_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        fetch_body: bool = True,
    ):
        super().__init__(peer_id)
        self.aliases = aliases
        self.search_queries = search_queries
        self.display = max(1, min(display, NAVER_MAX_DISPLAY))
        self.max_results = (
            NAVER_MAX_START if max_results <= 0 else min(max_results, NAVER_MAX_START)
        )
        self.cutoff_datetime = cutoff_datetime
        self.end_datetime = end_datetime
        self.fetch_body = fetch_body
        self.credentials = load_naver_credentials()
        self._credential_index = 0

    async def crawl(self) -> list[RawArticle]:
        if not self.credentials:
            log.warning("NAVER_CLIENT_ID 또는 NAVER_CLIENT_IDS 미설정. 크롤링 스킵.")
            return []

        articles = []
        query_specs = self.search_queries or [
            {"query": alias, "sector": ""} for alias in self._search_aliases()
        ]

        for spec in query_specs:
            try:
                articles.extend(
                    await self._fetch(query=spec["query"], sector=spec.get("sector", ""))
                )
            except NaverApiCredentialExhaustedError:
                raise
            except Exception as e:
                log.error("네이버 크롤링 실패 | query=%s error=%s", spec["query"], e)

        return articles

    async def _fetch(self, query: str, sector: str) -> list[RawArticle]:
        async with httpx.AsyncClient(
            timeout=10,
            headers=REQUEST_HEADERS,
            follow_redirects=True,
        ) as client:
            items = []

            for start in range(1, self.max_results + 1, self.display):
                display = min(self.display, self.max_results - len(items))

                if display <= 0:
                    break

                resp = await self._request_api(
                    client,
                    params={
                        "query": query,
                        "display": display,
                        "start": start,
                        "sort": "date",
                    },
                )

                if resp is None:
                    return []

                if self._is_credential_retryable_status(resp.status_code):
                    log.warning(
                        "네이버 API 접근 차단 | status=%d credentials=%d",
                        resp.status_code,
                        len(self.credentials),
                    )
                    raise NaverApiCredentialExhaustedError(
                        f"Naver News API credential exhausted: HTTP {resp.status_code}"
                    )

                resp.raise_for_status()
                page_items = resp.json().get("items", [])

                if not page_items:
                    break

                items.extend(page_items)

                if page_reaches_cutoff(page_items, self.cutoff_datetime):
                    break

                if len(page_items) < display:
                    break

            articles = [
                article
                for article in (
                    self._item_to_article(item, query=query, sector=sector) for item in items
                )
                if article_within_window(article, self.cutoff_datetime, self.end_datetime)
                and article_mentions_target_peer(article, self.peer_id)
                and not is_obvious_non_business_candidate(article)
            ]

            if self.fetch_body:
                await enrich_with_body_text(client, articles)
                before_refilter = len(articles)
                articles = [
                    article
                    for article in articles
                    if article_mentions_target_peer(article, self.peer_id)
                    or not is_body_text_for_relevance(article)
                ]
                dropped = before_refilter - len(articles)
                if dropped:
                    log.info(
                        "네이버 본문 기준 무관 기사 제외 | peer_id=%s query=%s dropped=%d",
                        self.peer_id,
                        query,
                        dropped,
                    )

            return articles

    async def _request_api(
        self,
        client: httpx.AsyncClient,
        params: dict[str, object],
    ) -> httpx.Response | None:
        blocked_response: httpx.Response | None = None

        for _ in range(len(self.credentials)):
            credential_index = self._credential_index
            credential = self.credentials[credential_index]
            self._credential_index = (credential_index + 1) % len(self.credentials)

            resp = await client.get(
                NAVER_API_URL,
                params=params,
                headers={
                    "X-Naver-Client-Id": credential.client_id,
                    "X-Naver-Client-Secret": credential.client_secret,
                },
            )

            if not self._is_credential_retryable_status(resp.status_code):
                return resp

            blocked_response = resp
            log.warning(
                "네이버 API 키 차단/쿼터 초과, 다음 키 재시도 | status=%d key_index=%d/%d",
                resp.status_code,
                credential_index + 1,
                len(self.credentials),
            )

        return blocked_response

    def _is_credential_retryable_status(self, status_code: int) -> bool:
        return status_code in NAVER_CREDENTIAL_RETRY_STATUS_CODES

    def _item_to_article(self, item: dict, query: str, sector: str) -> RawArticle:
        return RawArticle(
            url=item.get("originallink") or item["link"],
            title=strip_html(item["title"]),
            content=strip_html(item.get("description", "")),
            published_at=parse_naver_date(item.get("pubDate")),
            source_name="naver_news",
            source_type="news",
            content_type="html",
            company=[self.peer_id],
            language="ko",
            extra={
                "search_query": query,
                "naver_title": strip_html(item.get("title", "")),
                "naver_description": strip_html(item.get("description", "")),
                **({"sector": sector} if sector else {}),
            },
        )

    def _search_aliases(self) -> list[str]:
        aliases = list(self.aliases)
        if self.peer_id == "posco_dx":
            aliases.extend(_POSCO_DX_GROUP_SEARCH_ALIASES)
        return _dedupe_keep_order(aliases)


class NaverIndustryNewsCrawler(NaverNewsCrawler):
    """산업 방향성/IT 트렌드용 네이버 뉴스 크롤러.

    기존 ``naver_news`` 는 피어사 단독 중심 기사 수집을 유지한다. 이 크롤러는
    섹터 키워드 기반으로 검색하되 특정 타기업/피어사 단독 이벤트는 제외하고,
    산업·시장·정책·보안 리스크·AI 인프라 방향성 기사만 ``industry_trend`` 버킷에 저장한다.
    """

    def __init__(
        self,
        search_queries: list[dict[str, str]] | None = None,
        display: int = 100,
        max_results: int = 60,
        cutoff_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        fetch_body: bool = True,
    ):
        super().__init__(
            peer_id=INDUSTRY_TREND_COMPANY,
            aliases=[INDUSTRY_TREND_COMPANY],
            search_queries=search_queries or list(_INDUSTRY_NEWS_QUERY_GROUPS),
            display=display,
            max_results=max_results,
            cutoff_datetime=cutoff_datetime,
            end_datetime=end_datetime,
            fetch_body=fetch_body,
        )

    async def crawl(self) -> list[RawArticle]:
        if not self.credentials:
            log.warning("NAVER_CLIENT_ID 또는 NAVER_CLIENT_IDS 미설정. 산업 뉴스 크롤링 스킵.")
            return []

        articles: list[RawArticle] = []
        for spec in self.search_queries or []:
            try:
                articles.extend(
                    await self._fetch(query=spec["query"], sector=spec.get("sector", ""))
                )
            except NaverApiCredentialExhaustedError:
                raise
            except Exception as e:
                log.error("산업 뉴스 크롤링 실패 | query=%s error=%s", spec["query"], e)

        return _filter_industry_direction_articles(articles)

    async def _fetch(self, query: str, sector: str) -> list[RawArticle]:
        articles = await self._fetch_without_peer_filter(query=query, sector=sector)
        filtered = _filter_industry_direction_articles(articles)

        if self.fetch_body and filtered:
            async with httpx.AsyncClient(
                timeout=10,
                headers=REQUEST_HEADERS,
                follow_redirects=True,
            ) as client:
                await enrich_with_body_text(client, filtered)
            filtered = _filter_industry_direction_articles(filtered)

        return filtered

    async def _fetch_without_peer_filter(self, query: str, sector: str) -> list[RawArticle]:
        async with httpx.AsyncClient(
            timeout=10,
            headers=REQUEST_HEADERS,
            follow_redirects=True,
        ) as client:
            items = []

            for start in range(1, self.max_results + 1, self.display):
                display = min(self.display, self.max_results - len(items))

                if display <= 0:
                    break

                resp = await self._request_api(
                    client,
                    params={
                        "query": query,
                        "display": display,
                        "start": start,
                        "sort": "date",
                    },
                )

                if resp is None:
                    return []

                if self._is_credential_retryable_status(resp.status_code):
                    log.warning(
                        "네이버 API 접근 차단 | source=%s status=%d credentials=%d",
                        INDUSTRY_NEWS_SOURCE_NAME,
                        resp.status_code,
                        len(self.credentials),
                    )
                    raise NaverApiCredentialExhaustedError(
                        f"Naver News API credential exhausted: HTTP {resp.status_code}"
                    )

                resp.raise_for_status()
                page_items = resp.json().get("items", [])

                if not page_items:
                    break

                items.extend(page_items)

                if page_reaches_cutoff(page_items, self.cutoff_datetime):
                    break

                if len(page_items) < display:
                    break

            return [
                article
                for article in (
                    self._item_to_article(item, query=query, sector=sector) for item in items
                )
                if article_within_window(article, self.cutoff_datetime, self.end_datetime)
                and not is_obvious_non_business_candidate(article)
            ]

    def _item_to_article(self, item: dict, query: str, sector: str) -> RawArticle:
        title = strip_html(item["title"])
        description = strip_html(item.get("description", ""))
        sector_details = _industry_sector_details(title, description, fallback_sector=sector)
        sectors = _dedupe_keep_order(
            [str(detail["sector_id"]) for detail in sector_details if detail.get("sector_id")]
            or ([sector] if sector else [])
        )

        return RawArticle(
            url=item.get("originallink") or item["link"],
            title=title,
            content=description,
            published_at=parse_naver_date(item.get("pubDate")),
            source_name=INDUSTRY_NEWS_SOURCE_NAME,
            source_type="news",
            content_type="html",
            company=[INDUSTRY_TREND_COMPANY],
            language="ko",
            extra={
                "collection_scope": "industry_keyword",
                "topic_scope": "industry_trend",
                "company_scope": "industry",
                "sector": sector,
                "matched_sectors": sectors,
                "matched_sector_details": sector_details,
                "search_query": query,
                "naver_title": title,
                "naver_description": description,
            },
        )


def _filter_industry_direction_articles(articles: list[RawArticle]) -> list[RawArticle]:
    filtered: list[RawArticle] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for article in articles:
        title_key = _industry_title_key(article.title or "")
        url_key = article.url.strip()
        if (url_key and url_key in seen_urls) or (title_key and title_key in seen_titles):
            continue

        decision = classify_industry_news_candidate(article)
        article.extra["industry_relevance"] = decision["decision"]
        article.extra["industry_relevance_reason"] = decision["reason"]
        article.extra["industry_direction_score"] = decision["direction_score"]
        article.extra["mentioned_tracked_peers"] = decision["mentioned_tracked_peers"]
        article.extra["mentioned_title_companies"] = decision["mentioned_title_companies"]

        if decision["decision"] == "pass":
            filtered.append(article)
            if url_key:
                seen_urls.add(url_key)
            if title_key:
                seen_titles.add(title_key)

    return filtered


def classify_industry_news_candidate(article: RawArticle) -> dict[str, object]:
    """산업 방향성/IT 트렌드 카드 후보인지 규칙 기반으로 판정한다."""

    title = article.title or ""
    content = article.content or ""
    text = f"{title} {content}"
    sector_details = _industry_sector_details(title, content)
    sectors = _dedupe_keep_order(
        [str(detail["sector_id"]) for detail in sector_details if detail.get("sector_id")]
    )
    direction_score = _industry_direction_score(text)
    has_strong_frame = bool(_INDUSTRY_STRONG_FRAME_RE.search(text))
    tracked_title_peers = _tracked_peers_in_text(title)
    tracked_text_peers = _tracked_peers_in_text(text)
    third_party_title_companies = [
        company for company in _THIRD_PARTY_TITLE_TERMS if company.lower() in title.lower()
    ]
    event_terms = [term for term in _COMPANY_EVENT_TITLE_TERMS if term in title]
    has_sector_signal = bool(sectors) or bool(str(article.extra.get("sector") or "").strip())

    reason: list[str] = []
    if sectors:
        reason.append(f"sector={','.join(sectors)}")
    if direction_score:
        reason.append(f"direction_score={direction_score}")
    if has_strong_frame:
        reason.append("strong_industry_frame")

    if len(tracked_title_peers) == 1 and len(tracked_text_peers) <= 1:
        return _industry_decision(
            "reject_peer_centered",
            reason + [f"tracked_peer_title={tracked_title_peers[0]}"],
            direction_score,
            tracked_text_peers,
            [*third_party_title_companies, *tracked_title_peers],
        )

    if any(term in title for term in _INDUSTRY_NEWS_TITLE_BLOCK_TERMS):
        return _industry_decision(
            "reject_blocked_title_pattern",
            reason,
            direction_score,
            tracked_text_peers,
            [*third_party_title_companies, *tracked_title_peers],
        )

    if len(tracked_text_peers) >= 2:
        reason.append("multiple_tracked_peers")
        return _industry_decision(
            "pass",
            reason,
            direction_score,
            tracked_text_peers,
            [*third_party_title_companies, *tracked_title_peers],
        )

    if third_party_title_companies:
        if event_terms or not _is_allowed_public_industry_title(title):
            return _industry_decision(
                "reject_third_party_centered",
                reason
                + [f"third_party_title={','.join(third_party_title_companies)}"]
                + ([f"event_terms={','.join(event_terms)}"] if event_terms else []),
                direction_score,
                tracked_text_peers,
                [*third_party_title_companies, *tracked_title_peers],
            )

    if not has_sector_signal:
        return _industry_decision(
            "reject_no_sector_signal",
            reason,
            direction_score,
            tracked_text_peers,
            [*third_party_title_companies, *tracked_title_peers],
        )

    if has_strong_frame and direction_score >= 3:
        return _industry_decision(
            "pass",
            reason,
            direction_score,
            tracked_text_peers,
            [*third_party_title_companies, *tracked_title_peers],
        )

    return _industry_decision(
        "reject_weak_industry_direction",
        reason,
        direction_score,
        tracked_text_peers,
        [*third_party_title_companies, *tracked_title_peers],
    )


def _is_allowed_public_industry_title(title: str) -> bool:
    """회사명이 제목에 있어도 산업·정책 프레임이 명확한 경우만 예외 허용."""

    return any(
        term in title
        for term in (
            "AI 경쟁",
            "공급망 공격",
            "정부",
            "국가",
            "과기정통부",
            "정보보호학회",
            "국제표준",
            "로드맵",
        )
    )


def _industry_decision(
    decision: str,
    reason: list[str],
    direction_score: int,
    tracked_peers: list[str],
    title_companies: list[str],
) -> dict[str, object]:
    return {
        "decision": "pass" if decision == "pass" else "reject",
        "reason": decision if not reason else f"{decision}: {'; '.join(reason)}",
        "direction_score": direction_score,
        "mentioned_tracked_peers": _dedupe_keep_order(tracked_peers),
        "mentioned_title_companies": _dedupe_keep_order(title_companies),
    }


def _industry_sector_details(
    title: str,
    content: str,
    *,
    fallback_sector: str = "",
) -> list[dict[str, str]]:
    details = [dict(item) for item in match_sector_details(f"{title} {content}")]
    if details or not fallback_sector:
        return details

    sector = SECTOR_KEYWORDS.get(fallback_sector)
    if not sector:
        return []

    return [
        {
            "sector_id": fallback_sector,
            "sector_name_ko": str(sector.get("name_ko") or fallback_sector),
            "keyword": str(sector["keywords"][0] if sector.get("keywords") else fallback_sector),
        }
    ]


def _industry_direction_score(text: str) -> int:
    return sum(1 for term in _INDUSTRY_DIRECTION_TERMS if term.lower() in text.lower())


def _industry_title_key(title: str) -> str:
    text = strip_html(title).lower()
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"['\"“”‘’…·,.:;!?~\-\s]+", "", text)
    return text[:120]


def _tracked_peers_in_text(text: str) -> list[str]:
    text_norm = normalize(text)
    peers: list[str] = []

    for peer_id in DEFAULT_PEER_IDS:
        for alias in get_relevance_aliases(peer_id):
            alias_norm = _search_alias_norm(alias)
            if alias_norm and alias_norm in text_norm:
                peers.append(peer_id)
                break

    return _dedupe_keep_order(peers)


def load_naver_credentials() -> list[NaverCredential]:
    credentials: list[NaverCredential] = []

    multi_ids = _split_env_list(os.getenv("NAVER_CLIENT_IDS", ""))
    multi_secrets = _split_env_list(os.getenv("NAVER_CLIENT_SECRETS", ""))

    if multi_ids or multi_secrets:
        if len(multi_ids) != len(multi_secrets):
            log.warning(
                "NAVER_CLIENT_IDS/NAVER_CLIENT_SECRETS 개수가 다릅니다 | ids=%d secrets=%d",
                len(multi_ids),
                len(multi_secrets),
            )

        for client_id, client_secret in zip(multi_ids, multi_secrets, strict=False):
            _append_credential(credentials, client_id, client_secret)

    for suffix in _credential_suffixes_from_env():
        _append_credential(
            credentials,
            os.getenv(f"NAVER_CLIENT_ID_{suffix}", ""),
            os.getenv(f"NAVER_CLIENT_SECRET_{suffix}", ""),
        )

    _append_credential(
        credentials,
        os.getenv("NAVER_CLIENT_ID", ""),
        os.getenv("NAVER_CLIENT_SECRET", ""),
    )

    return credentials


def _split_env_list(value: str) -> list[str]:
    return [
        item.strip().strip('"').strip("'")
        for item in value.split(",")
        if item.strip().strip('"').strip("'")
    ]


def _credential_suffixes_from_env() -> list[str]:
    suffixes = {
        key.removeprefix("NAVER_CLIENT_ID_")
        for key in os.environ
        if key.startswith("NAVER_CLIENT_ID_")
    }
    return sorted(suffixes, key=_credential_suffix_sort_key)


def _credential_suffix_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _append_credential(
    credentials: list[NaverCredential],
    client_id: str,
    client_secret: str,
) -> None:
    client_id = client_id.strip().strip('"').strip("'")
    client_secret = client_secret.strip().strip('"').strip("'")

    if not client_id or not client_secret:
        return

    credential = NaverCredential(client_id, client_secret)

    if credential not in credentials:
        credentials.append(credential)


async def enrich_with_body_text(
    client: httpx.AsyncClient,
    articles: list[RawArticle],
) -> None:
    for article in articles:
        try:
            resp = await client.get(resolve_fetch_url(article.url))

            if resp.status_code != 200:
                article.extra["body_fetch_status"] = f"failed:{resp.status_code}"
                continue

            publisher = extract_publisher(resp.text, str(resp.url))
            if publisher:
                article.publisher = publisher

            article.extra["image_urls"] = extract_image_urls(resp.text, str(resp.url))

            subtitle = extract_subtitle(resp.text)
            if subtitle:
                article.extra["subtitle"] = subtitle

            body = extract_body_text(resp.text)

            if body:
                article.content = body
                article.extra["body_fetch_status"] = "success"
            else:
                article.extra["body_fetch_status"] = "fallback_description"

        except Exception as e:
            article.extra["body_fetch_status"] = "error"
            article.extra["body_fetch_error"] = str(e)


def resolve_fetch_url(url: str) -> str:
    parsed = urlparse(url)

    if parsed.netloc.endswith("thebell.co.kr") and parsed.path.endswith(
        "/free/content/ArticleView.asp"
    ):
        key = parse_qs(parsed.query).get("key", [""])[0]

        if key:
            return f"{parsed.scheme}://{parsed.netloc}/front/newsview.asp?click=F&key={key}"

    return url


def extract_publisher(html: str, url: str) -> str | None:
    publisher = extract_publisher_from_html(html)

    if publisher:
        return publisher

    return None


def extract_publisher_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html or "", "html.parser")

    publisher = extract_json_ld_publisher(soup)

    if publisher:
        return publisher

    meta_selectors = (
        'meta[property="og:site_name"]',
        'meta[name="application-name"]',
        'meta[name="subject"]',
        'meta[name="copyright"]',
        'meta[name="Copyright"]',
        'meta[name="title"]',
        'meta[property="og:title"]',
        'meta[name="twitter:title"]',
    )

    for selector in meta_selectors:
        meta = soup.select_one(selector)
        publisher = clean_publisher_name(
            extract_publisher_candidate(str(meta.get("content", "")) if meta else "")
        )

        if publisher:
            return publisher

    for selector in (
        "header h1 img",
        "#header h1 img",
        ".head_top h1 img",
        "h1 img",
        ".media_end_head_top_logo img",
        ".press_logo img",
        ".journalistcard_summary_press",
        ".byline .press",
    ):
        node = soup.select_one(selector)

        if not node:
            continue

        publisher = clean_publisher_name(str(node.get("alt", "") or node.get_text(" ", strip=True)))

        if publisher:
            return publisher

    return None


def extract_json_ld_publisher(soup: BeautifulSoup) -> str | None:
    for node in soup.select('script[type="application/ld+json"]'):
        raw_json = node.string or node.get_text("", strip=True)

        if not raw_json:
            continue

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            continue

        publisher = publisher_from_structured_data(data)

        if publisher:
            return publisher

    return None


def publisher_from_structured_data(data: object) -> str | None:
    if isinstance(data, list):
        for item in data:
            publisher = publisher_from_structured_data(item)

            if publisher:
                return publisher

        return None

    if not isinstance(data, dict):
        return None

    publisher = data.get("publisher")

    if isinstance(publisher, dict):
        candidate = clean_publisher_name(str(publisher.get("name", "")))

        if candidate:
            return candidate

    if isinstance(publisher, str):
        candidate = clean_publisher_name(publisher)

        if candidate:
            return candidate

    graph = data.get("@graph")

    if isinstance(graph, list):
        return publisher_from_structured_data(graph)

    return None


def extract_publisher_candidate(value: str) -> str:
    text = strip_html(value)

    bracket_match = re.match(r"^\s*\[([^\[\]]{2,30})\]", text)

    if bracket_match:
        return bracket_match.group(1)

    return text


def clean_publisher_name(value: str) -> str | None:
    text = strip_html(value)
    text = re.sub(r"^@", "", text)
    text = re.sub(r"\s*[-|:：]\s*뉴스$", "", text)
    text = text.strip()

    if not text:
        return None

    parsed = urlparse(text)

    if parsed.scheme or parsed.netloc:
        return None

    blocked = {
        "뉴스",
        "네이버 뉴스",
        "네이버뉴스",
        "naver news",
        "news",
    }

    if text.lower() in blocked:
        return None

    return text[:50]


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")

    return (
        text.replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .strip()
    )


def parse_naver_date(date_text: str | None):
    if not date_text:
        return None

    try:
        return parsedate_to_datetime(date_text)
    except (TypeError, ValueError):
        return None


def page_reaches_cutoff(items: list[dict], cutoff: datetime | None) -> bool:
    if cutoff is None:
        return False

    return any(
        published_at is not None and published_at < cutoff
        for published_at in (parse_naver_date(item.get("pubDate")) for item in items)
    )


def article_within_cutoff(article: RawArticle, cutoff: datetime | None) -> bool:
    if cutoff is None:
        return True

    return article.published_at is not None and article.published_at >= cutoff


def article_within_window(
    article: RawArticle,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    if article.published_at is None:
        return False
    if start is not None and article.published_at < _align_tz(start, article.published_at):
        return False
    if end is not None and article.published_at > _align_tz(end, article.published_at):
        return False
    return True


def _align_tz(boundary: datetime, value: datetime) -> datetime:
    if boundary.tzinfo is None and value.tzinfo is not None:
        return boundary.replace(tzinfo=value.tzinfo)
    if boundary.tzinfo is not None and value.tzinfo is None:
        return boundary.replace(tzinfo=None)
    return boundary


def article_mentions_target_peer(article: RawArticle, target_peer_id: str) -> bool:
    text = f"{article.title or ''} {article.content or ''}"
    text_norm = normalize(text)

    return any(
        alias_norm
        for alias_norm in (
            _search_alias_norm(alias) for alias in get_relevance_aliases(target_peer_id)
        )
        if alias_norm and alias_norm in text_norm
    )


def is_obvious_non_business_candidate(article: RawArticle) -> bool:
    """본문 fetch 전에 스포츠/엔터성 소비자 캠페인 후보를 제외한다."""

    parsed = urlparse(article.url or "")
    path_segments = {segment.lower() for segment in parsed.path.split("/") if segment.strip()}
    if path_segments & _NON_BUSINESS_PATH_SEGMENTS:
        return True

    text_norm = f"{article.title or ''} {article.content or ''}".lower()
    return any(term in text_norm for term in _NON_BUSINESS_TERMS)


def annotate_peer_relevance(
    articles: list[RawArticle],
    target_peer_id: str,
    tracked_peer_ids: list[str],
) -> None:
    for article in articles:
        analysis = classify_peer_relevance(article, target_peer_id, tracked_peer_ids)
        company_peer_ids = analysis.pop("_company_peer_ids", [])

        if isinstance(company_peer_ids, list):
            article.company = sorted(set(map(str, company_peer_ids)))

        article.extra.update(analysis)


def classify_peer_relevance(
    article: RawArticle,
    target_peer_id: str,
    tracked_peer_ids: list[str],
) -> dict[str, object]:
    """제목 기준으로 피어사 중심 기사인지 pass/reject 라벨링한다.

    수정 기준:
    - 제목에 추적 대상 회사 5개 중 하나라도 나오면 pass
    - 제목 내 위치 기준은 사용하지 않음
    - 부제/본문 기준은 기존 로직 유지
    """
    title = article.title or ""
    content = article.content or ""
    subtitle = str(article.extra.get("subtitle", ""))

    if _is_peer_filter_noise(title=title, content=content):
        return {
            "matched_aliases_by_peer": {},
            "_company_peer_ids": [],
            "peer_relevance": "reject",
            "peer_relevance_reason": "news_noise_market_or_event_listing",
            "target_peer_mention_count": 0,
        }

    title_norm = normalize(title)
    subtitle_norm = normalize(subtitle)
    content_norm = normalize(content)

    relevance_body_norm = content_norm if is_body_text_for_relevance(article) else ""
    full_norm = f"{title_norm} {subtitle_norm} {content_norm}"
    subbody_norm = f"{subtitle_norm} {relevance_body_norm}"

    matched_peers: list[str] = []
    matched_aliases_by_peer: dict[str, list[str]] = {}
    body_aliases_by_peer: dict[str, list[str]] = {}
    body_mention_counts: dict[str, int] = {}
    subbody_mention_counts: dict[str, int] = {}

    title_peers: list[str] = []
    target_count = 0
    target_subbody_count = 0

    for peer_id in tracked_peer_ids:
        aliases = get_relevance_aliases(peer_id)
        peer_matched = False
        seen_alias_norms: set[str] = set()

        for alias in aliases:
            alias_norm = normalize(alias)

            if not alias_norm:
                continue

            if alias_norm in seen_alias_norms:
                continue

            seen_alias_norms.add(alias_norm)

            count = full_norm.count(alias_norm)

            if count:
                peer_matched = True
                matched_aliases_by_peer.setdefault(peer_id, []).append(alias)

                if peer_id == target_peer_id:
                    target_count += count
                    target_subbody_count += subbody_norm.count(alias_norm)

            subbody_count = subbody_norm.count(alias_norm)

            if subbody_count:
                subbody_mention_counts[peer_id] = (
                    subbody_mention_counts.get(peer_id, 0) + subbody_count
                )

            body_count = relevance_body_norm.count(alias_norm)

            if body_count:
                body_aliases_by_peer.setdefault(peer_id, []).append(alias)
                body_mention_counts[peer_id] = body_mention_counts.get(peer_id, 0) + body_count

            title_pos = title_norm.find(alias_norm)

            if title_pos >= 0:
                if peer_id not in title_peers:
                    title_peers.append(peer_id)

        if peer_matched:
            matched_peers.append(peer_id)

    # 제목 기준:
    # 제목에 추적 대상 회사 5개 중 하나라도 나오면 pass.
    # 제목 내 위치 기준은 사용하지 않는다.
    if title_peers:
        decision = "pass"
        reason = "tracked_peer_in_title"

    elif target_peer_id == "posco_dx" and _matches_posco_dx_group_ax_signal(
        f"{title} {subtitle} {content}"
    ):
        matched_peers.append("posco_dx")
        matched_aliases_by_peer.setdefault("posco_dx", []).append("포스코+AX")
        title_peers.append("posco_dx")
        target_count = max(target_count, 1)
        decision = "pass"
        reason = "posco_group_ax_signal"

    # 부제 + 본문 기준:
    # 제목에는 없지만 부제 + 정제된 본문에서 타깃 피어사명이 2회 이상 반복 등장하면 pass.
    elif target_subbody_count >= 2 and _target_has_core_role_context(
        subbody_norm,
        target_peer_id,
    ):
        decision = "pass"
        reason = "target_peer_core_role_in_body"

    # 본문에 1회 이상 등장했지만 기준 미달인 경우.
    elif target_count >= 1:
        decision = "reject"
        reason = "target_peer_only_in_body"

    else:
        decision = "reject"
        reason = "target_peer_not_in_title"

    return {
        "matched_aliases_by_peer": {
            peer_id: sorted(set(aliases))
            for peer_id, aliases in sorted(matched_aliases_by_peer.items())
        },
        **body_company_mentions_for_output(
            reason=reason,
            body_aliases_by_peer=body_aliases_by_peer,
            body_mention_counts=body_mention_counts,
        ),
        "_company_peer_ids": company_peer_ids_for_article(
            decision=decision,
            reason=reason,
            target_peer_id=target_peer_id,
            title_peers=title_peers,
            subbody_mention_counts=subbody_mention_counts,
        ),
        "peer_relevance": decision,
        "peer_relevance_reason": reason,
        "target_peer_mention_count": target_count,
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


_MARKET_PRICE_RE = re.compile(
    r"(주가|종가|장중|상승\s*마감|하락\s*마감|강세|약세|급등|급락|상한가|하한가|시가총액)"
)
_MARKET_METRIC_RE = re.compile(r"(\d+(?:\.\d+)?\s*%|\d{1,3}(?:,\d{3})+\s*원)")
_LOW_VALUE_NEWS_KEYWORDS = [
    "뉴스브리핑",
    "대학 뉴스브리핑",
    "멘토링",
    "졸업동문",
    "초청 멘토링",
    "박람회 개최",
    "취업 상담",
    "사상 최고 경신",
    "하락종목도",
    "마감시황",
    "증시키워드",
    "기업이슈",
    "레버리지",
    "외국인",
    "기관",
    "순매수",
    "순매도",
]
_EVENT_LISTING_KEYWORDS = [
    "전시회",
    "박람회",
    "컨퍼런스",
    "세미나",
    "포럼",
    "행사",
    "코엑스",
    "개최",
    "참가",
    "총집결",
    "부스",
    "시상식",
]
_STRONG_PEER_NEWS_KEYWORDS = [
    "수주",
    "계약",
    "우선협상",
    "협약",
    "업무협약",
    "제휴",
    "투자유치",
    "지분투자",
    "투자계획",
    "인수",
    "합병",
    "실적",
    "공시",
    "조직개편",
    "채용",
]
_TARGET_CORE_ROLE_KEYWORDS = [
    "수주",
    "계약",
    "구축",
    "공급",
    "선정",
    "우선협상",
    "컨소시엄",
    "협약",
    "업무협약",
    "제휴",
    "출시",
    "공개",
    "개발",
    "투자유치",
    "지분투자",
    "인수",
    "합병",
    "실적",
    "공시",
]
_CONTEXT_WINDOW = 80


def _is_peer_filter_noise(*, title: str, content: str) -> bool:
    if _is_non_korean_title(title):
        return True

    text = f"{title} {content}"
    compact_text = normalize(text)
    title_norm = normalize(title)
    peer_in_title = any(
        normalize(alias) in title_norm for aliases in COMPANY_ALIASES.values() for alias in aliases
    )
    has_strong_keyword = any(
        normalize(keyword) in compact_text for keyword in _STRONG_PEER_NEWS_KEYWORDS
    )

    if any(normalize(keyword) in title_norm for keyword in _LOW_VALUE_NEWS_KEYWORDS):
        return not (peer_in_title and has_strong_keyword)

    if _MARKET_PRICE_RE.search(text) and _MARKET_METRIC_RE.search(text):
        return not (peer_in_title and has_strong_keyword)

    has_event_keyword = any(
        normalize(keyword) in compact_text for keyword in _EVENT_LISTING_KEYWORDS
    )
    if not has_event_keyword:
        return False

    event_in_title = any(normalize(keyword) in title_norm for keyword in _EVENT_LISTING_KEYWORDS)
    if event_in_title and not peer_in_title:
        return True

    return not peer_in_title and not has_strong_keyword


def _is_non_korean_title(title: str) -> bool:
    title = title or ""
    return bool(title.strip()) and re.search(r"[가-힣]", title) is None


def _target_has_core_role_context(text_norm: str, target_peer_id: str) -> bool:
    aliases = [
        alias_norm
        for alias_norm in (
            _search_alias_norm(alias) for alias in get_relevance_aliases(target_peer_id)
        )
        if alias_norm
    ]
    keyword_norms = [normalize(keyword) for keyword in _TARGET_CORE_ROLE_KEYWORDS]

    for alias_norm in aliases:
        start = 0
        while True:
            pos = text_norm.find(alias_norm, start)
            if pos < 0:
                break
            left = max(0, pos - _CONTEXT_WINDOW)
            right = min(len(text_norm), pos + len(alias_norm) + _CONTEXT_WINDOW)
            context = text_norm[left:right]
            if any(keyword in context for keyword in keyword_norms):
                return True
            start = pos + len(alias_norm)

    return False


def _matches_posco_dx_group_ax_signal(text: str) -> bool:
    if not text:
        return False
    if not _POSCO_DX_GROUP_RELEVANCE_RE.search(text):
        return False
    if _MARKET_PRICE_RE.search(text) and _MARKET_METRIC_RE.search(text):
        return False
    return True


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def company_peer_ids_for_article(
    decision: str,
    reason: str,
    target_peer_id: str,
    title_peers: list[str],
    subbody_mention_counts: dict[str, int],
) -> list[str]:
    if decision == "pass" and reason in {
        "tracked_peer_in_title",
        "target_peer_in_title",
        "multiple_tracked_peers_in_title",
        "target_peer_core_role_in_body",
        "posco_group_ax_signal",
    }:
        return sorted(set(title_peers or [target_peer_id]))

    return sorted(peer_id for peer_id, count in subbody_mention_counts.items() if count >= 2)


def body_company_mentions_for_output(
    reason: str,
    body_aliases_by_peer: dict[str, list[str]],
    body_mention_counts: dict[str, int],
) -> dict[str, object]:
    if reason in {
        "tracked_peer_in_title",
        "target_peer_in_title",
        "multiple_tracked_peers_in_title",
    }:
        return {}

    return {
        "body_company_mentions": {
            peer_id: {
                "aliases": sorted(set(body_aliases_by_peer.get(peer_id, []))),
                "count": body_mention_counts[peer_id],
            }
            for peer_id in sorted(body_mention_counts)
        }
    }


def is_body_text_for_relevance(article: RawArticle) -> bool:
    status = str(article.extra.get("body_fetch_status", ""))

    return status not in {"fallback_description", "fallback_rss_summary", "error"} and not (
        status.startswith("failed:")
    )


def get_peer_aliases(peer_id: str) -> tuple[str, ...] | list[str]:
    return PEER_ALIASES.get(peer_id, [peer_id])


def get_relevance_aliases(peer_id: str) -> tuple[str, ...] | list[str]:
    aliases = [alias for alias in get_peer_aliases(peer_id) if _is_relevance_alias(alias)]
    return aliases or [peer_id]


def get_search_aliases(peer_id: str) -> tuple[str, ...] | list[str]:
    aliases = [alias for alias in PEER_ALIASES.get(peer_id, [peer_id]) if _is_search_alias(alias)]
    return aliases or [peer_id]


def _is_search_alias(alias: str) -> bool:
    alias_norm = _search_alias_norm(alias)
    if not alias_norm:
        return False
    if alias_norm in _SEARCH_ALIAS_BLOCKLIST:
        return False
    if len(alias_norm) < 4:
        return False
    return True


def _is_relevance_alias(alias: str) -> bool:
    alias_norm = _search_alias_norm(alias)
    if not alias_norm:
        return False
    return alias_norm not in _SEARCH_ALIAS_BLOCKLIST


def _search_alias_norm(alias: str) -> str:
    return normalize(alias).replace("㈜", "").replace("(주)", "")


def load_local_env() -> None:
    env_path = PROJECT_ROOT / ".env.local"
    load_dotenv(env_path, override=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB 저장 없이 네이버 뉴스 크롤링 로컬 테스트")

    parser.add_argument(
        "--company",
        "--peer-id",
        dest="company",
        action="append",
        default=None,
        help=("특정 회사 id만 테스트한다. 여러 번 지정 가능. --peer-id는 하위 호환 alias."),
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="직접 검색어를 지정한다. 미지정 시 scheduler.PEER_ALIASES를 사용한다.",
    )
    parser.add_argument(
        "--display",
        type=int,
        default=100,
        help="네이버 API 1회 요청당 결과 수. 네이버 제한상 최대 100.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=(
            "검색어별 최대 수집 결과 수. "
            f"기본 base.py news 한도({DEFAULT_MAX_RESULTS}), 네이버 API 제한상 최대 1000."
        ),
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="최근 N시간 이내 기사만 저장한다. 0 이하이면 시간 필터를 끈다. 기본 24.",
    )
    parser.add_argument(
        "--no-body",
        action="store_true",
        help="기사 본문 HTML 추가 수집을 끈다",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "수집 결과를 저장할 JSON 파일 경로. "
            "미지정 시 src/crawler/crawler_results/naver_crawler.json에 저장한다."
        ),
    )

    return parser.parse_args()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    load_local_env()

    args = parse_args()
    peer_ids = args.company or list(DEFAULT_PEER_IDS)

    output_path = Path(args.output) if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cutoff_datetime = hours_cutoff(args.hours)

    articles: list[RawArticle] = []

    for peer_id in peer_ids:
        aliases = args.query or get_search_aliases(peer_id)

        crawler = NaverNewsCrawler(
            peer_id=peer_id,
            aliases=aliases,
            display=args.display,
            max_results=args.max_results,
            cutoff_datetime=cutoff_datetime,
            fetch_body=not args.no_body,
        )

        peer_articles = await crawler.crawl()
        annotate_peer_relevance(peer_articles, peer_id, peer_ids)

        log.info("피어사별 수집 완료 | peer=%s count=%d", peer_id, len(peer_articles))

        articles.extend(peer_articles)

        interim_selected = select_articles_for_output(
            articles,
            hours=args.hours,
        )

        write_json_results(interim_selected, output_path)

        log.info(
            "중간 저장 완료 | path=%s saved=%d",
            output_path,
            len(interim_selected),
        )

    recent_articles = filter_by_hours(articles, args.hours)

    selected_articles = select_articles_for_output(
        articles,
        hours=args.hours,
    )

    payload = write_json_results(selected_articles, output_path)
    image_dir = write_image_url_files(selected_articles, output_path)

    print(f"수집 결과: {len(articles)}건")
    print(f"최근 {args.hours}시간 필터 결과: {len(recent_articles)}건")
    print(f"저장 결과: {len(selected_articles)}건 (peer_relevance=pass)")
    print_relevance_summary(recent_articles)
    print(f"JSON 저장 위치: {output_path}")
    print(f"이미지 URL 저장 폴더: {image_dir}")

    for article in payload[:3]:
        print(json.dumps(article, ensure_ascii=False, indent=2, default=str))


def filter_pass_articles(articles: list[RawArticle]) -> list[RawArticle]:
    return [a for a in articles if a.extra.get("peer_relevance") == "pass"]


def select_articles_for_output(
    articles: list[RawArticle],
    hours: int,
) -> list[RawArticle]:
    merged = merge_articles_by_url(filter_by_hours(articles, hours))
    return filter_pass_articles(merged)


def merge_articles_by_url(articles: list[RawArticle]) -> list[RawArticle]:
    merged: dict[str, RawArticle] = {}

    for article in articles:
        key = article_merge_key(article)
        existing = merged.get(key)

        if not existing:
            merged[key] = article
            continue

        existing.company = sorted(set(existing.company) | set(article.company))
        existing.extra = merge_article_extra(existing.extra, article.extra)

        if len(article.content or "") > len(existing.content or ""):
            existing.content = article.content

        if article.published_at and (
            existing.published_at is None or article.published_at > existing.published_at
        ):
            existing.published_at = article.published_at

    return list(merged.values())


def article_merge_key(article: RawArticle) -> str:
    if article.url:
        return canonicalize_article_url(article.url)

    return f"title:{normalize_title_text(article.title)}"


def canonicalize_article_url(url: str) -> str:
    parsed = urlparse(url)

    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {"sid", "utm_source", "utm_medium", "utm_campaign", "utm_term"}
    ]

    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            urlencode(sorted(query)),
            "",
        )
    )


def merge_article_extra(
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    merged = {**left, **right}

    for key in ("image_urls",):
        values: list[object] = []

        for source in (left, right):
            source_values = source.get(key)

            if isinstance(source_values, list):
                values.extend(source_values)

        if values:
            merged[key] = sorted(set(values), key=str)

    aliases_by_peer: dict[str, list[str]] = {}

    for source in (left, right):
        source_aliases = source.get("matched_aliases_by_peer")

        if not isinstance(source_aliases, dict):
            continue

        for peer_id, aliases in source_aliases.items():
            if isinstance(aliases, list):
                aliases_by_peer.setdefault(str(peer_id), []).extend(map(str, aliases))

    if aliases_by_peer:
        merged["matched_aliases_by_peer"] = {
            peer_id: sorted(set(aliases)) for peer_id, aliases in sorted(aliases_by_peer.items())
        }

    body_mentions: dict[str, dict[str, object]] = {}

    for source in (left, right):
        source_mentions = source.get("body_company_mentions")

        if not isinstance(source_mentions, dict):
            continue

        for peer_id, mention in source_mentions.items():
            if not isinstance(mention, dict):
                continue

            target = body_mentions.setdefault(str(peer_id), {"aliases": [], "count": 0})

            aliases = mention.get("aliases")

            if isinstance(aliases, list):
                target["aliases"] = [*target["aliases"], *map(str, aliases)]

            count = mention.get("count")

            if isinstance(count, int):
                target["count"] = max(int(target["count"]), count)

    if body_mentions:
        merged["body_company_mentions"] = {
            peer_id: {
                "aliases": sorted(set(mention["aliases"])),
                "count": mention["count"],
            }
            for peer_id, mention in sorted(body_mentions.items())
        }

    if left.get("peer_relevance") == "pass" or right.get("peer_relevance") == "pass":
        merged["peer_relevance"] = "pass"

        if left.get("peer_relevance") == "pass":
            merged["peer_relevance_reason"] = left.get("peer_relevance_reason")

        if right.get("peer_relevance") == "pass":
            merged["peer_relevance_reason"] = right.get("peer_relevance_reason")

    return merged


def write_json_results(articles: list[RawArticle], output_path: Path) -> list[dict[str, object]]:
    payload = [article_to_output_dict(article) for article in articles]

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return payload


def article_to_output_dict(article: RawArticle) -> dict[str, object]:
    payload = article.to_common_dict()
    payload["extra"] = clean_output_extra(article.extra)

    return payload


def clean_output_extra(extra: dict[str, object]) -> dict[str, object]:
    keep_keys = (
        "image_urls",
        "subtitle",
        "matched_aliases_by_peer",
        "body_company_mentions",
    )

    cleaned: dict[str, object] = {}

    for key in keep_keys:
        value = extra.get(key)

        if is_empty_extra_value(value):
            continue

        cleaned[key] = value

    reason = extra.get("peer_relevance_reason")

    if reason in {
        "tracked_peer_in_title",
        "target_peer_in_title",
        "multiple_tracked_peers_in_title",
    }:
        cleaned.pop("body_company_mentions", None)

    return cleaned


def is_empty_extra_value(value: object) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, (list, dict)):
        return len(value) == 0

    return False


def filter_by_hours(articles: list[RawArticle], hours: int) -> list[RawArticle]:
    cutoff = hours_cutoff(hours)

    if cutoff is None:
        return articles

    return [
        article
        for article in articles
        if article.published_at is not None and article.published_at >= cutoff
    ]


def hours_cutoff(hours: int) -> datetime | None:
    if hours <= 0:
        return None

    return datetime.now().astimezone() - timedelta(hours=hours)


def print_relevance_summary(articles: list[RawArticle]) -> None:
    summary = {"pass": 0, "reject": 0}

    for article in articles:
        key = str(article.extra.get("peer_relevance", "reject"))

        if key not in summary:
            key = "reject"

        summary[key] += 1

    print(f"관련도 라벨: pass={summary['pass']} / reject={summary['reject']}")


def write_image_url_files(articles: list[RawArticle], output_path: Path) -> Path:
    image_dir = output_path.parent / f"{output_path.stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for stale_file in image_dir.glob("*.json"):
        stale_file.unlink()

    for article in articles:
        image_urls = article.extra.get("image_urls")

        if not isinstance(image_urls, list):
            image_urls = []

        (image_dir / f"{article.id}.json").write_text(
            json.dumps(image_urls, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return image_dir


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "crawler_results" / f"{Path(__file__).stem}.json"


if __name__ == "__main__":
    asyncio.run(main())
