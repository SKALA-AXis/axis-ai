"""네이버 증권 리서치/PDF 크롤러."""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, time, timedelta
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

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
MAX_PAGES = int(os.getenv("NAVER_RESEARCH_MAX_PAGES", "1000"))

PEER_ITEM_CODES = dict(NAVER_ITEM_CODES)


class NaverResearchCrawler(BaseCrawler):
    """네이버 증권 기업 리포트 PDF를 수집한다."""

    def __init__(
        self,
        peer_id: str,
        item_code: str | None = None,
        lookback_days: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        super().__init__(peer_id)
        self.item_code = item_code or PEER_ITEM_CODES.get(peer_id, "")
        self.lookback_days = lookback_days or int(
            os.getenv("NAVER_RESEARCH_LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS))
        )
        self.start_date = start_date
        self.end_date = end_date

    async def crawl(self) -> list[RawArticle]:
        if not self.item_code:
            log.warning("네이버 증권 itemCode 미설정 | peer_id=%s", self.peer_id)
            return []

        url = NAVER_RESEARCH_URL.format(item_code=self.item_code)
        return await self._fetch(url=url, report_type="company_report")

    async def _fetch(self, url: str, report_type: str) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            articles: list[RawArticle] = []
            seen_urls: set[str] = set()
            seen_page_keys: set[tuple[str, ...]] = set()

            for page_no in range(1, self._max_pages() + 1):
                page_url = _with_query_param(url, "page", str(page_no))
                resp = await client.get(
                    page_url,
                    headers={"User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0"},
                )
                resp.raise_for_status()

                soup = BeautifulSoup(resp.text, "html.parser")
                rows = [row for row in soup.select("table.type_1 tr") if row.select("td")]

                if not rows:
                    break

                page_key = tuple(_row_key(row) for row in rows)
                if page_key in seen_page_keys:
                    log.info(
                        "네이버 기업 리포트 반복 페이지 감지, 수집 중단 | peer_id=%s page=%s",
                        self.peer_id,
                        page_no,
                    )
                    break
                seen_page_keys.add(page_key)

                row_dates = [_row_published_at(row) for row in rows]

                for row in rows:
                    article = await self._parse_row(
                        client=client,
                        row=row,
                        report_type=report_type,
                        base_url=page_url,
                    )
                    if article and article.url not in seen_urls:
                        seen_urls.add(article.url)
                        articles.append(article)

                if self._page_is_older_than_window(row_dates):
                    break

            return articles

    def _max_pages(self) -> int:
        if self.start_date or self.end_date:
            return max(1, MAX_PAGES)
        return 1

    async def _parse_row(
        self,
        client: httpx.AsyncClient,
        row,
        report_type: str,
        base_url: str,
    ) -> RawArticle | None:
        fields = _row_fields(row)
        if not fields:
            return None

        pdf_el = row.select_one("a[href*='.pdf']")
        firm = fields["firm"]
        report_title = fields["title"]
        published_at = fields["published_at"]

        if not self._is_in_collection_window(published_at):
            return None

        pdf_url = urljoin(base_url, pdf_el.get("href", "")) if pdf_el else ""
        pdf_payload = await _fetch_pdf_payload(client, pdf_url) if pdf_url else _empty_pdf_payload()
        pdf_text = str(pdf_payload.get("text") or "")
        pdf_published_at = _extract_pdf_report_date(pdf_text)
        if pdf_published_at is not None:
            published_at = pdf_published_at

        if not self._is_in_collection_window(published_at):
            return None

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
                "list_published_at": fields["published_at_text"],
                "pdf_published_at": pdf_published_at.isoformat() if pdf_published_at else None,
                "lookback_days": self.lookback_days,
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "item_code": self.item_code,
            },
        )

    def _is_in_collection_window(self, published_at: datetime | None) -> bool:
        if published_at is None:
            return False

        if self.start_date or self.end_date:
            start = datetime.combine(self.start_date, time.min) if self.start_date else datetime.min
            end = datetime.combine(self.end_date, time.max) if self.end_date else datetime.max
            return start <= published_at.replace(tzinfo=None) <= end

        return published_at >= datetime.now() - timedelta(days=self.lookback_days)

    def _page_is_older_than_window(self, row_dates: list[datetime | None]) -> bool:
        if not self.start_date:
            return False
        known_dates = [value for value in row_dates if value]
        if not known_dates:
            return False
        return max(value.replace(tzinfo=None) for value in known_dates) < datetime.combine(
            self.start_date,
            time.min,
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


_FULL_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>20\d{2}|19\d{2})[.\-/]\s*(?P<month>\d{1,2})[.\-/]\s*(?P<day>\d{1,2})(?!\d)"
)


def _extract_pdf_report_date(pdf_text: str) -> datetime | None:
    """PDF 표지/첫 페이지 초반에 적힌 실제 리포트 작성일을 추출한다."""
    if not pdf_text:
        return None

    first_page = pdf_text.split("[PAGE 2]", 1)[0][:4000]
    for line in first_page.splitlines()[:40]:
        text = strip_html(line).strip()
        if not text or len(text) > 80:
            continue
        match = _FULL_DATE_RE.search(text)
        if not match:
            continue
        try:
            return datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
        except ValueError:
            continue

    return None


def _row_published_at(row) -> datetime | None:
    fields = _row_fields(row)
    return fields["published_at"] if fields else None


def _row_key(row) -> str:
    fields = _row_fields(row)
    if not fields:
        text_parts = [cell.get_text(" ", strip=True) for cell in row.select("td")[:3]]
    else:
        text_parts = [fields["title"], fields["firm"], fields["published_at_text"]]
    pdf_el = row.select_one("a[href*='.pdf']")
    pdf_href = pdf_el.get("href", "") if pdf_el else ""
    return "|".join([*text_parts, pdf_href])


def _row_fields(row) -> dict[str, object] | None:
    cells = row.select("td")
    if len(cells) >= 5:
        title_el = cells[1].select_one("a")
        firm_cell = cells[2]
        date_cell = cells[4]
    elif len(cells) >= 3:
        title_el = cells[0].select_one("a")
        firm_cell = cells[1]
        date_cell = cells[2]
    else:
        return None

    if not title_el:
        return None

    published_at_text = date_cell.get_text(" ", strip=True)
    return {
        "title": strip_html(title_el.get_text(" ", strip=True)),
        "firm": strip_html(firm_cell.get_text(" ", strip=True)),
        "published_at_text": published_at_text,
        "published_at": _parse_report_date(published_at_text),
    }


def _with_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params[key] = value
    return urlunparse(parsed._replace(query=urlencode(params)))
