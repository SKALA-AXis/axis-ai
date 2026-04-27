"""네이버 금융 리서치 크롤러 (Tier 2) — Playwright, Track B."""

import asyncio
import logging
from typing import Optional

from src.crawler.base import SOURCE_CREDIBILITY, DailyLimitGuard, RawArticle
from src.crawler.playwright_client import PlaywrightClient

log = logging.getLogger(__name__)

# 종목코드 (Naver Finance itemCode)
# - samsung_sds 018260, lg_cns 064400 (이전 코드의 034730 = SK Inc. 오류였음)
# - hyundai_autoever 307950, posco_dx 022100
_COMPANY_ITEM_CODES: dict[str, str] = {
    "samsung_sds": "018260",
    "lg_cns": "064400",
    "hyundai_autoever": "307950",
    "posco_dx": "022100",
}
_COMPANY_URL_TPL = (
    "https://finance.naver.com/research/company_list.naver?searchType=itemCode&itemCode={code}"
)
_COMPANY_URLS: dict[str, str] = {
    peer_id: _COMPANY_URL_TPL.format(code=code) for peer_id, code in _COMPANY_ITEM_CODES.items()
}

# IT서비스 섹터 리포트
_INDUSTRY_URL = (
    "https://finance.naver.com/research/industry_list.naver?searchType=upjongCode&upjongCode=54"
)


class NaverResearchCrawler:
    """네이버 금융 리서치 — 기업·섹터 리포트 수집. 동적 로딩이므로 Playwright 필수."""

    def __init__(self, limit_guard: Optional[DailyLimitGuard] = None) -> None:
        self.pw = PlaywrightClient()
        self.limit_guard = limit_guard or DailyLimitGuard()

    async def crawl(self) -> list[RawArticle]:
        if not self.limit_guard.allow("naver_research"):
            return []
        peer_ids = list(_COMPANY_URLS.keys())
        coros = [self._fetch_company(pid) for pid in peer_ids] + [self._fetch_industry()]
        results = await asyncio.gather(*coros, return_exceptions=True)
        articles: list[RawArticle] = []
        labels = peer_ids + ["industry"]
        for label, r in zip(labels, results):
            if isinstance(r, list):
                log.info("네이버 리서치 수집 | target=%s articles=%d", label, len(r))
                articles.extend(r)
            elif isinstance(r, Exception):
                log.error("네이버 리서치 크롤링 오류 | target=%s error=%s", label, r)
        return articles

    async def _fetch_company(self, peer_id: str) -> list[RawArticle]:
        url = _COMPANY_URLS[peer_id]
        async with self.pw.new_page() as page:
            await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            await page.wait_for_selector("table.type_1", timeout=8_000)
            rows = await page.query_selector_all("table.type_1 tr")
            return await self._parse_rows(rows, peer_id, "company_report")

    async def _fetch_industry(self) -> list[RawArticle]:
        async with self.pw.new_page() as page:
            await page.goto(_INDUSTRY_URL, wait_until="domcontentloaded", timeout=20_000)
            await page.wait_for_selector("table.type_1", timeout=8_000)
            rows = await page.query_selector_all("table.type_1 tr")
            return await self._parse_rows(rows, None, "industry_report")

    async def _parse_rows(
        self,
        rows: list,
        peer_id: Optional[str],
        report_type: str,
    ) -> list[RawArticle]:
        articles = []
        for row in rows:
            tds = await row.query_selector_all("td")
            if len(tds) < 3:
                continue
            try:
                title_el = await tds[0].query_selector("a")
                firm_text = await tds[1].inner_text()
                date_text = await tds[2].inner_text()
                pdf_el = await tds[-1].query_selector("a[href*='.pdf']")

                if not title_el:
                    continue

                title = await title_el.inner_text()
                pdf_url = await pdf_el.get_attribute("href") if pdf_el else None

                articles.append(
                    RawArticle(
                        url=pdf_url or _INDUSTRY_URL,
                        title=f"[{firm_text.strip()}] {title.strip()}",
                        content="",
                        source_tier=2,
                        source_name="naver_research",
                        credibility_score=SOURCE_CREDIBILITY["naver_research"],
                        peer_id=peer_id,
                        metadata={
                            "type": report_type,
                            "firm": firm_text.strip(),
                            "report_date": date_text.strip(),
                            "pdf_url": pdf_url,
                        },
                    )
                )
            except Exception:
                continue
        return articles
