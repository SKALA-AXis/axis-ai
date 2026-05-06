"""네이버 금융 리서치 크롤러 (Tier 2) — Playwright, Track B."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import httpx

from src.config.companies import NAVER_ITEM_CODES
from src.crawler.base import CrawlWindow, DailyLimitGuard, RawArticle
from src.crawler.playwright_client import PlaywrightClient

log = logging.getLogger(__name__)

# 종목코드 (Naver Finance itemCode)
# - samsung_sds 018260, lg_cns 064400 (이전 코드의 034730 = SK Inc. 오류였음)
# - hyundai_autoever 307950, posco_dx 022100
_COMPANY_ITEM_CODES: dict[str, str] = dict(NAVER_ITEM_CODES)
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
DEFAULT_LOOKBACK_DAYS = int(os.getenv("NAVER_RESEARCH_LOOKBACK_DAYS", "30"))
PDF_MAX_TEXT_CHARS = int(os.getenv("RESEARCH_PDF_MAX_TEXT_CHARS", "12000"))
PDF_TIMEOUT = float(os.getenv("RESEARCH_PDF_TIMEOUT", "12"))


class NaverResearchCrawler:
    """네이버 금융 리서치 — 기업·섹터 리포트 수집. 동적 로딩이므로 Playwright 필수."""

    def __init__(
        self,
        limit_guard: Optional[DailyLimitGuard] = None,
        crawl_window: Optional[CrawlWindow] = None,
    ) -> None:
        self.pw = PlaywrightClient()
        self.limit_guard = limit_guard or DailyLimitGuard()
        self.crawl_window = crawl_window or CrawlWindow.last_days(DEFAULT_LOOKBACK_DAYS)

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
                report_date = _parse_report_date(date_text)
                if not self.crawl_window.contains(report_date):
                    continue

                pdf_href = await pdf_el.get_attribute("href") if pdf_el else None
                pdf_url = urljoin(_INDUSTRY_URL, pdf_href) if pdf_href else None
                pdf_text = await _fetch_pdf_text(pdf_url) if pdf_url else ""

                articles.append(
                    RawArticle(
                        url=pdf_url or _INDUSTRY_URL,
                        title=f"[{firm_text.strip()}] {title.strip()}",
                        content=pdf_text,
                        source_name="naver_research",
                        peer_id=peer_id,
                        published_at=report_date,
                        metadata={
                            "source_type": "securities_report",
                            "content_type": "pdf" if pdf_url else "html",
                            "type": report_type,
                            "firm": firm_text.strip(),
                            "report_date": date_text.strip(),
                            "pdf_url": pdf_url,
                            "pdf_text_chars": len(pdf_text),
                            "crawl_window_start": self.crawl_window.start.date().isoformat(),
                            "crawl_window_end": (
                                self.crawl_window.end or datetime.now()
                            ).date().isoformat(),
                        },
                    )
                )
            except Exception:
                continue
        return articles


def _parse_report_date(date_text: str) -> Optional[datetime]:
    text = date_text.strip()
    for fmt in ("%y.%m.%d", "%Y.%m.%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


async def _fetch_pdf_text(pdf_url: str) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=PDF_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0"},
        ) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
        return _extract_pdf_text(resp.content)
    except Exception as e:
        log.debug("네이버 리서치 PDF 추출 실패 | url=%s error=%s", pdf_url, e)
        return ""


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import pymupdf  # type: ignore

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            chunks: list[str] = []
            for page in doc:
                chunks.append(page.get_text("text"))
                if sum(len(chunk) for chunk in chunks) >= PDF_MAX_TEXT_CHARS:
                    break
        return "\n".join(chunks).strip()[:PDF_MAX_TEXT_CHARS]
    except Exception as e:
        log.debug("PDF 텍스트 변환 실패 | error=%s", e)
        return ""
