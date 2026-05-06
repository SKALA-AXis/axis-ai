"""채용공고 크롤러 (Tier 5) — Saramin API, 약한 신호 감지용."""

import logging
import os
from datetime import datetime
from typing import Any, Optional

import httpx

from src.config.companies import SARAMIN_COMPANY_NAMES
from src.crawler.base import RETRY_POLICY, BaseCrawler, CrawlWindow, DailyLimitGuard, RawArticle

log = logging.getLogger(__name__)

SARAMIN_API_URL = "https://oapi.saramin.co.kr/job-search"
DEFAULT_LOOKBACK_DAYS = int(os.getenv("JOBS_LOOKBACK_DAYS", "14"))

_COMPANY_NAMES: dict[str, str] = dict(SARAMIN_COMPANY_NAMES)


class JobsCrawler(BaseCrawler):
    """Saramin 채용공고 수집 — 약한 신호(조직 변화, 기술 스택 급증) 감지용."""

    def __init__(
        self,
        peer_id: str,
        keywords: list[str],
        limit_guard: Optional[DailyLimitGuard] = None,
        crawl_window: Optional[CrawlWindow] = None,
    ) -> None:
        super().__init__(peer_id, limit_guard, crawl_window)
        self.keywords = keywords
        self.api_key = os.getenv("SARAMIN_API_KEY", "")

    async def crawl(self) -> list[RawArticle]:
        if not self.api_key:
            log.warning("SARAMIN_API_KEY 미설정. 크롤링 스킵.")
            return []
        if not self.limit_guard.allow("jobs"):
            return []
        peer_id = self.peer_id or ""
        company_name = _COMPANY_NAMES.get(peer_id, "")
        if not company_name:
            return []
        try:
            return await self._fetch(company_name, peer_id)
        except Exception as e:
            log.error("Saramin 크롤링 실패 | peer_id=%s error=%s", peer_id, e)
            return []

    async def _fetch(self, company_name: str, peer_id: str) -> list[RawArticle]:
        window = self.crawl_window or CrawlWindow.last_days(DEFAULT_LOOKBACK_DAYS)
        async with httpx.AsyncClient(timeout=RETRY_POLICY["timeout"]) as client:
            resp = await client.get(
                SARAMIN_API_URL,
                params={
                    "access-key": self.api_key,
                    "company": company_name,
                    "count": 40,
                    "fields": "posting-date,expiration-date,keyword-code,sal-code",
                },
                headers={"Accept": "application/json"},
            )
            if self._is_blocked(resp.status_code):
                log.warning("Saramin API 접근 차단 | status=%d", resp.status_code)
                return []
            resp.raise_for_status()
            data = resp.json()
            jobs = data.get("jobs", {}).get("job", [])
            articles = [_to_article(job, peer_id, window) for job in jobs if isinstance(job, dict)]
            return [article for article in articles if window.contains(article.published_at)]


def _to_article(job: dict[str, Any], peer_id: str, window: CrawlWindow) -> RawArticle:
    position = job.get("position", {})
    title = position.get("title", "")
    company = job.get("company", {}).get("detail", {}).get("name", "")
    url = job.get("url", "")
    industry = position.get("industry", {}).get("name", "")
    job_type = position.get("job-type", {}).get("name", "")
    published_at = _parse_ts(job.get("posting-timestamp"))
    expiration_at = _parse_ts(job.get("expiration-timestamp"))
    return RawArticle(
        url=url,
        title=f"[채용] {company} — {title}",
        content=f"직종: {industry}, 고용형태: {job_type}",
        published_at=published_at,
        source_name="saramin",
        peer_id=peer_id,
        metadata={
            "source_type": "job",
            "content_type": "api",
            "company": company,
            "job_type": job_type,
            "industry": industry,
            "expiration_at": expiration_at.isoformat() if expiration_at else None,
            "crawl_window_start": window.start.date().isoformat(),
            "crawl_window_end": (window.end or datetime.now()).date().isoformat(),
        },
    )


def _parse_ts(ts: Any) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(int(ts))
    except (TypeError, ValueError):
        return None
