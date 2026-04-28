"""DART 금융감독원 공시 크롤러 (Tier 3) — 배치, Track B."""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import httpx

from src.crawler.base import (
    RETRY_POLICY,
    SOURCE_CREDIBILITY,
    BaseCrawler,
    DailyLimitGuard,
    RawArticle,
)

log = logging.getLogger(__name__)

DART_API_BASE = "https://opendart.fss.or.kr/api"

# 주요 공시 유형 (전략 관련성 높은 것만)
DISCLOSURE_TYPES = ["A", "B", "C", "D"]  # 정기·주요사항·발행·기타공시


class DartCrawler(BaseCrawler):
    """DART OpenAPI /api/list.json 기반 공시 수집."""

    def __init__(
        self,
        peer_id: str,
        corp_code: str,
        limit_guard: Optional[DailyLimitGuard] = None,
    ) -> None:
        super().__init__(peer_id, limit_guard)
        self.corp_code = corp_code
        self.api_key = os.getenv("DART_API_KEY", "")

    async def crawl(self) -> list[RawArticle]:
        if not self.api_key:
            log.warning("DART_API_KEY 미설정. 크롤링 스킵.")
            return []
        if not self.corp_code:
            log.warning("corp_code 미설정 | peer_id=%s. 크롤링 스킵.", self.peer_id)
            return []
        if not self.limit_guard.allow("dart"):
            return []
        try:
            return await self._fetch_disclosures()
        except Exception as e:
            log.error("DART 크롤링 실패 | peer_id=%s error=%s", self.peer_id, e)
            return []

    async def _fetch_disclosures(self) -> list[RawArticle]:
        today = datetime.now()
        bgn_de = (today - timedelta(days=1)).strftime("%Y%m%d")
        end_de = today.strftime("%Y%m%d")

        async with httpx.AsyncClient(timeout=RETRY_POLICY["timeout"]) as client:
            resp = await client.get(
                f"{DART_API_BASE}/list.json",
                params={
                    "crtfc_key": self.api_key,
                    "corp_code": self.corp_code,
                    "bgn_de": bgn_de,
                    "end_de": end_de,
                    "sort": "date",
                    "sort_mth": "desc",
                    "page_count": 40,
                },
            )
            if self._is_blocked(resp.status_code):
                log.warning("DART API 접근 차단 | status=%d", resp.status_code)
                return []
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "000":
                log.debug("DART 공시 없음 | status=%s", data.get("status"))
                return []

            articles = []
            for item in data.get("list", []):
                url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item['rcept_no']}"
                articles.append(
                    RawArticle(
                        url=url,
                        title=item.get("report_nm", ""),
                        content=f"{item.get('report_nm', '')} — {item.get('flr_nm', '')}",
                        published_at=_parse_dart_date(item.get("rcept_dt", "")),
                        source_name="dart",
                        peer_id=self.peer_id,
                        credibility_score=SOURCE_CREDIBILITY["dart"],
                        source_tier=3,
                        metadata={
                            "rcept_no": item.get("rcept_no"),
                            "flr_nm": item.get("flr_nm"),
                            "pblntf_ty": item.get("pblntf_ty"),
                        },
                    )
                )
            return articles


def _parse_dart_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return None
