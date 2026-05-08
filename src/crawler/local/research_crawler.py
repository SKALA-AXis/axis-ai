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
from src.crawler.parsers.pdf_payload import extract_pdf_payload

log = logging.getLogger(__name__)

NAVER_RESEARCH_URL = (
    "https://finance.naver.com/research/company_list.naver?searchType=itemCode&itemCode={item_code}"
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

        if published_at and published_at < datetime.now() - timedelta(days=self.lookback_days):
            return None

        pdf_url = urljoin(base_url, pdf_el.get("href", "")) if pdf_el else ""
        pdf_payload = await _fetch_pdf_payload(client, pdf_url) if pdf_url else _empty_pdf_payload()
        pdf_text = str(pdf_payload.get("text") or "")

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
                "pdf_pages": pdf_payload.get("page_count"),
                "pdf_parsed_pages": pdf_payload.get("parsed_page_count"),
                "pdf_page_blocks": pdf_payload.get("pages"),
                "pdf_parse_strategy": pdf_payload.get("pdf_parse_strategy"),
                "contains_images": pdf_payload.get("contains_images"),
                "image_count": pdf_payload.get("image_count"),
                "contains_tables": pdf_payload.get("contains_tables"),
                "table_count": pdf_payload.get("table_count"),
                "tables": pdf_payload.get("tables"),
                "table_parse_strategy": pdf_payload.get("table_parse_strategy"),
                "chart_parse_strategy": pdf_payload.get("chart_parse_strategy"),
                "lookback_days": self.lookback_days,
                "item_code": self.item_code,
            },
        )


async def _fetch_pdf_payload(client: httpx.AsyncClient, pdf_url: str) -> dict:
    if not pdf_url:
        return _empty_pdf_payload()

    try:
        resp = await client.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0"},
        )
        resp.raise_for_status()
        return extract_pdf_payload(resp.content, max_text_chars=PDF_MAX_TEXT_CHARS)

    except Exception as e:
        log.debug("리서치 PDF 추출 실패 | url=%s error=%s", pdf_url, e)
        return _empty_pdf_payload()


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """하위 호환용 텍스트 추출 wrapper."""
    return str(extract_pdf_payload(pdf_bytes, max_text_chars=PDF_MAX_TEXT_CHARS).get("text") or "")


def _empty_pdf_payload() -> dict:
    return {
        "text": "",
        "page_count": None,
        "parsed_page_count": None,
        "image_count": 0,
        "contains_images": False,
        "pages": [],
        "tables": [],
        "table_count": 0,
        "contains_tables": False,
        "pdf_parse_strategy": "not_parsed",
        "table_parse_strategy": "not_parsed",
        "chart_parse_strategy": "not_parsed",
    }


def _parse_report_date(date_text: str) -> datetime | None:
    text = strip_html(date_text).strip()

    for fmt in ("%y.%m.%d", "%Y.%m.%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None
