"""네이버 증권 리서치/PDF 크롤러."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from src.config.companies import NAVER_ITEM_CODES
from src.crawler.article_filter import strip_html
from src.crawler.base import RawArticle
from src.crawler.base_crawler import BaseCrawler

log = logging.getLogger(__name__)

NAVER_RESEARCH_URL = (
    "https://finance.naver.com/research/company_list.naver"
    "?searchType=itemCode&itemCode={item_code}"
)

DEFAULT_LOOKBACK_DAYS = 3
PDF_MAX_TEXT_CHARS = int(os.getenv("RESEARCH_PDF_MAX_TEXT_CHARS", "200000"))

PEER_ITEM_CODES = dict(NAVER_ITEM_CODES)


class NaverResearchCrawler(BaseCrawler):
    """네이버 증권 기업 리포트 PDF를 수집한다."""

    def __init__(
        self,
        peer_id: str,
        item_code: str | None = None,
        lookback_days: int | None = None,
    ):
        super().__init__(peer_id)
        self.item_code = item_code or PEER_ITEM_CODES.get(peer_id, "")
        self.lookback_days = lookback_days or int(
            os.getenv("NAVER_RESEARCH_LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS))
        )

    async def crawl(self) -> list[RawArticle]:
        if not self.item_code:
            log.warning("네이버 증권 itemCode 미설정 | peer_id=%s", self.peer_id)
            return []

        url = NAVER_RESEARCH_URL.format(item_code=self.item_code)
        return await self._fetch(url=url, report_type="company_report")

    async def _fetch(self, url: str, report_type: str) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0"},
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("table.type_1 tr")

            articles: list[RawArticle] = []

            for row in rows:
                article = await self._parse_row(
                    client=client,
                    row=row,
                    report_type=report_type,
                    base_url=url,
                )
                if article:
                    articles.append(article)

            return articles

    async def _parse_row(
        self,
        client: httpx.AsyncClient,
        row,
        report_type: str,
        base_url: str,
    ) -> RawArticle | None:
        cells = row.select("td")

        if len(cells) < 3:
            return None

        title_el = cells[0].select_one("a")
        pdf_el = row.select_one("a[href*='.pdf']")

        if not title_el:
            return None

        firm = strip_html(cells[1].get_text(" ", strip=True))
        report_title = strip_html(title_el.get_text(" ", strip=True))
        published_at = _parse_report_date(cells[2].get_text(" ", strip=True))

        if published_at and published_at < datetime.now() - timedelta(
            days=self.lookback_days
        ):
            return None

        pdf_url = urljoin(base_url, pdf_el.get("href", "")) if pdf_el else ""
        pdf_text = await _fetch_pdf_text(client, pdf_url) if pdf_url else ""

        if pdf_url and not pdf_text:
            log.warning("네이버 기업 리포트 PDF 본문 추출 실패 | url=%s", pdf_url)

        return RawArticle(
            url=pdf_url or base_url,
            title=f"[{firm}] {report_title}",
            content=pdf_text,
            published_at=published_at,
            source_name="naver_research",
            peer_id=self.peer_id,
            source_type="securities_report",
            content_type="pdf" if pdf_url else "html",
            publisher=firm,
            company=[self.peer_id],
            extra={
                "report_type": report_type,
                "firm": firm,
                "pdf_url": pdf_url,
                "pdf_text_chars": len(pdf_text),
                "lookback_days": self.lookback_days,
                "item_code": self.item_code,
            },
        )


async def _fetch_pdf_text(client: httpx.AsyncClient, pdf_url: str) -> str:
    if not pdf_url:
        return ""

    try:
        resp = await client.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0"},
        )
        resp.raise_for_status()
        return _extract_pdf_text(resp.content)

    except Exception as e:
        log.debug("리서치 PDF 추출 실패 | url=%s error=%s", pdf_url, e)
        return ""


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import pymupdf  # type: ignore

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            chunks: list[str] = []
            current_length = 0

            for page in doc:
                page_text = page.get_text("text")
                chunks.append(page_text)
                current_length += len(page_text)

                if current_length >= PDF_MAX_TEXT_CHARS:
                    break

        return "\n".join(chunks).strip()[:PDF_MAX_TEXT_CHARS]

    except Exception as e:
        log.debug("PDF 텍스트 변환 실패 | error=%s", e)
        return ""


def _parse_report_date(date_text: str) -> datetime | None:
    text = strip_html(date_text).strip()

    for fmt in ("%y.%m.%d", "%Y.%m.%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None
