"""KIPRIS 특허·실용신안 출원 정보 크롤러 (Tier 1) — 공공데이터포털 REST API, Track B 주 1회."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree

import httpx

from src.crawler.base import (
    RETRY_POLICY,
    SOURCE_CREDIBILITY,
    DailyLimitGuard,
    RawArticle,
)

log = logging.getLogger(__name__)

KIPRIS_API_BASE = "http://plus.kipris.or.kr/openapi/rest"

# AI 관련 IPC 분류 코드 필터 (§7 설계서)
AI_IPC_CODES = ["G06N", "G06F", "G06V", "G06T", "H04L"]

# 출원인 한글명 (설계서 §7)
_APPLICANTS: dict[str, str] = {
    "samsung_sds": "삼성에스디에스",
    "lg_cns": "엘지씨엔에스",
    "hyundai_autoever": "현대오토에버",
    "posco_dx": "포스코디엑스",
}


class KiprisCrawler:
    """KIPRIS applicantNameSearchInfo API — 특허 출원 동향 수집 (약한 신호 감지용)."""

    def __init__(self, limit_guard: Optional[DailyLimitGuard] = None) -> None:
        self.api_key = os.getenv("KIPRIS_API_KEY", "")
        self.limit_guard = limit_guard or DailyLimitGuard()

    async def crawl(self) -> list[RawArticle]:
        if not self.api_key:
            log.warning("KIPRIS_API_KEY 미설정. 크롤링 스킵.")
            return []
        if not self.limit_guard.allow("kipris"):
            return []

        results = await asyncio.gather(
            *[self._fetch(peer_id, name) for peer_id, name in _APPLICANTS.items()],
            return_exceptions=True,
        )
        articles: list[RawArticle] = []
        for r in results:
            if isinstance(r, list):
                articles.extend(r)
            elif isinstance(r, Exception):
                log.error("KIPRIS 크롤링 오류 | error=%s", r)
        return articles

    async def _fetch(self, peer_id: str, applicant: str) -> list[RawArticle]:
        _timeout = RETRY_POLICY["source_timeout"]["backoff"][0]
        async with httpx.AsyncClient(timeout=_timeout) as client:
            resp = await client.get(
                f"{KIPRIS_API_BASE}/patUtiModInfoSearchSevice/applicantNameSearchInfo",
                params={
                    "applicant": applicant,
                    "sortSpec": "AD",
                    "descSort": "true",
                    "numOfRows": 50,
                    "accessKey": self.api_key,
                },
            )
            if resp.status_code in (403, 429):
                log.warning("KIPRIS API 접근 차단 | status=%d", resp.status_code)
                return []
            resp.raise_for_status()
            return _parse_xml(resp.text, peer_id)


def _parse_xml(xml_text: str, peer_id: str) -> list[RawArticle]:
    articles: list[RawArticle] = []
    try:
        root = ElementTree.fromstring(xml_text)
        for item in root.findall(".//PatentUtilityInfo"):
            ipc = item.findtext("InternationalpatentclassificationNumber", "")
            title = item.findtext("InventionName", "")
            app_no = item.findtext("ApplicationNumber", "")
            app_date = item.findtext("ApplicationDate", "")
            applicant = item.findtext("Applicant", "")
            abstract = item.findtext("Abstract", "")

            is_ai = any(ipc.startswith(code) for code in AI_IPC_CODES)

            articles.append(
                RawArticle(
                    url=(
                        f"https://www.kipris.or.kr/kportal/search/total_search.do"
                        f"?searchQuery={app_no}"
                    ),
                    title=f"[특허] {title}",
                    content=abstract
                    or (f"출원번호: {app_no}, 출원인: {applicant}, 출원일: {app_date}"),
                    source_tier=1,
                    source_name="kipris",
                    credibility_score=SOURCE_CREDIBILITY["kipris"],
                    peer_id=peer_id,
                    published_at=_parse_date(app_date),
                    metadata={
                        "type": "patent",
                        "ipc_code": ipc,
                        "application_no": app_no,
                        "is_ai_related": is_ai,
                        "importance_hint": "high" if is_ai else "normal",
                    },
                )
            )
    except ElementTree.ParseError as e:
        log.error("KIPRIS XML 파싱 실패 | error=%s", e)
    return articles


def _parse_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return None
