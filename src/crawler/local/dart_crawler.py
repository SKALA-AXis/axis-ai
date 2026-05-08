"""DART 공시 크롤러"""

import io
import logging
import os
import re
import zipfile
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag

from src.crawler.article_filter import strip_html
from src.crawler.base import RawArticle
from src.crawler.base_crawler import BaseCrawler

log = logging.getLogger(__name__)

CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
DART_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DART_DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"

_DART_STATUS_OK = "000"
_DART_STATUS_NO_DATA = "013"
_DART_STATUS_BLOCKED = {"010", "011", "012", "020"}

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_PAGE_COUNT = 100
DEFAULT_FETCH_DOCUMENT = True
DEFAULT_MAX_DOCUMENT_LENGTH = 200000
DEFAULT_DISCLOSURE_TYPES = ("A", "B", "F")

_DISCLOSURE_TYPE_LABELS = {
    "A": "regular",
    "B": "material_event",
    "F": "external_audit",
}


class DartCrawler(BaseCrawler):
    def __init__(
        self,
        peer_id: str,
        corp_code: str | None = None,
        corp_names: list[str] | None = None,
    ):
        super().__init__(peer_id)

        env_key = f"DART_CORP_CODE_{peer_id.upper()}"

        self.corp_code = corp_code or os.getenv(env_key, "")
        self.corp_names = corp_names or []
        self.api_key = os.getenv("DART_API_KEY", "")

        self.lookback_days = int(os.getenv("DART_LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS)))
        self.page_count = int(os.getenv("DART_PAGE_COUNT", str(DEFAULT_PAGE_COUNT)))
        self.fetch_document = _env_bool(
            "DART_FETCH_DOCUMENT",
            DEFAULT_FETCH_DOCUMENT,
        )
        self.max_document_length = int(
            os.getenv(
                "DART_MAX_TEXT_CHARS",
                os.getenv("DART_MAX_DOCUMENT_LENGTH", str(DEFAULT_MAX_DOCUMENT_LENGTH)),
            )
        )
        self.disclosure_types = _env_list(
            "DART_DISCLOSURE_TYPES",
            DEFAULT_DISCLOSURE_TYPES,
        )

    async def crawl(self) -> list[RawArticle]:
        if not self.api_key:
            log.warning("DART_API_KEY 미설정. 크롤링 스킵.")
            return []

        corp_codes = [self.corp_code] if self.corp_code else await self._resolve_corp_codes()

        if not corp_codes:
            log.warning(
                "DART corp_code를 찾지 못해 스킵 | peer_id=%s names=%s",
                self.peer_id,
                self.corp_names,
            )
            return []

        articles: list[RawArticle] = []

        for corp_code in corp_codes:
            disclosures = await self._fetch_disclosures(corp_code=corp_code)
            articles.extend(disclosures)

        return articles

    async def _resolve_corp_codes(self) -> list[str]:
        if not self.corp_names:
            log.warning(
                "corp_code, corp_names 모두 미설정 | peer_id=%s",
                self.peer_id,
            )
            return []

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    CORP_CODE_URL,
                    params={"crtfc_key": self.api_key},
                )
                resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
                xml_data = archive.read("CORPCODE.xml")

            root = ElementTree.fromstring(xml_data)

            corp_codes: list[str] = []
            normalized_names = {_normalize_name(name) for name in self.corp_names}

            for item in root.findall("list"):
                corp_name = item.findtext("corp_name", default="")
                normalized_corp_name = _normalize_name(corp_name)

                if normalized_corp_name in normalized_names:
                    corp_code = item.findtext("corp_code", default="")
                    if corp_code:
                        corp_codes.append(corp_code)
                        log.info(
                            "DART corp_code 매칭 | peer_id=%s corp_name=%s corp_code=%s",
                            self.peer_id,
                            corp_name,
                            corp_code,
                        )

            if not corp_codes:
                log.warning(
                    "DART corp_code 매칭 실패 | peer_id=%s names=%s",
                    self.peer_id,
                    self.corp_names,
                )

            return corp_codes

        except Exception as e:
            log.error(
                "DART corp_code 조회 실패 | peer_id=%s error=%s",
                self.peer_id,
                e,
            )
            return []

    async def _fetch_disclosures(self, corp_code: str) -> list[RawArticle]:
        end = datetime.now(tz=timezone.utc)
        begin = end - timedelta(days=self.lookback_days)

        articles: list[RawArticle] = []
        seen_receipts: set[str] = set()

        for disclosure_type in self.disclosure_types:
            disclosures = await self._fetch_disclosures_by_type(
                corp_code=corp_code,
                disclosure_type=disclosure_type,
                begin=begin,
                end=end,
            )

            for article in disclosures:
                receipt_no = article.extra.get("receipt_no", "")
                if receipt_no and receipt_no in seen_receipts:
                    continue

                if receipt_no:
                    seen_receipts.add(receipt_no)

                articles.append(article)

        log.info(
            "DART 수집 완료 | peer_id=%s corp_code=%s types=%s articles=%d",
            self.peer_id,
            corp_code,
            ",".join(self.disclosure_types),
            len(articles),
        )

        return articles

    async def _fetch_disclosures_by_type(
        self,
        corp_code: str,
        disclosure_type: str,
        begin: datetime,
        end: datetime,
    ) -> list[RawArticle]:
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bgn_de": begin.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "pblntf_ty": disclosure_type,
            "page_count": self.page_count,
            "sort": "date",
            "sort_mth": "desc",
        }

        log.info(
            "DART 공시 조회 | peer_id=%s corp_code=%s type=%s bgn_de=%s end_de=%s",
            self.peer_id,
            corp_code,
            disclosure_type,
            params["bgn_de"],
            params["end_de"],
        )

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(DART_LIST_URL, params=params)
                resp.raise_for_status()
        except Exception as e:
            log.error(
                "DART API 요청 실패 | peer_id=%s corp_code=%s error=%s",
                self.peer_id,
                corp_code,
                e,
            )
            return []

        payload = resp.json()
        status = payload.get("status", "")
        message = payload.get("message", "")

        if status == _DART_STATUS_NO_DATA:
            log.info(
                "DART 조회 결과 없음 (정상) | peer_id=%s corp_code=%s type=%s",
                self.peer_id,
                corp_code,
                disclosure_type,
            )
            return []

        if status in _DART_STATUS_BLOCKED:
            log.error(
                "DART API 키 오류 또는 요청 제한 | peer_id=%s status=%s message=%s",
                self.peer_id,
                status,
                message,
            )
            return []

        if status != _DART_STATUS_OK:
            log.warning(
                "DART 알 수 없는 status | peer_id=%s status=%s message=%s",
                self.peer_id,
                status,
                message,
            )
            return []

        items = payload.get("list", [])

        log.info(
            "DART 공시 목록 수신 | peer_id=%s corp_code=%s type=%s count=%d",
            self.peer_id,
            corp_code,
            disclosure_type,
            len(items),
        )

        articles: list[RawArticle] = []

        for item in items:
            receipt_no = item.get("rcept_no", "")
            report_name = strip_html(item.get("report_nm", ""))
            published_at = _parse_dart_date(item.get("rcept_dt", ""))

            if published_at is None:
                log.warning(
                    "DART 날짜 파싱 실패, 스킵 | peer_id=%s receipt_no=%s rcept_dt=%s",
                    self.peer_id,
                    receipt_no,
                    item.get("rcept_dt"),
                )
                continue

            fallback_content = strip_html(_disclosure_content(item))

            document_payload = {
                "text": "",
                "fetched": False,
                "fetch_error": "",
                "raw_text_length": 0,
                "contains_tables": False,
                "table_count": 0,
                "contains_images": False,
                "image_count": 0,
                "file_count": 0,
                "parsed_file_count": 0,
                "parse_strategy": "not_fetched",
            }

            if self.fetch_document and receipt_no:
                try:
                    document_payload = await self._fetch_document_payload(receipt_no)
                except Exception as exc:
                    document_payload["fetch_error"] = str(exc)
                    log.warning(
                        "DART 원문 조회 실패 | peer_id=%s corp_code=%s receipt_no=%s error=%s",
                        self.peer_id,
                        corp_code,
                        receipt_no,
                        exc,
                    )

            document_text = document_payload.get("text", "")
            raw_content = document_text or fallback_content
            raw_content_length = len(raw_content)

            content = raw_content
            content_truncated = False

            if self.max_document_length > 0 and len(content) > self.max_document_length:
                content = content[: self.max_document_length]
                content_truncated = True

            articles.append(
                RawArticle(
                    url=f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt_no}",
                    title=report_name,
                    content=content,
                    published_at=published_at,
                    source_name="dart",
                    peer_id=self.peer_id,
                    source_type="dart",
                    content_type="api",
                    company=[self.peer_id],
                    publisher=item.get("corp_name", ""),
                    extra={
                        "corp_code": corp_code,
                        "receipt_no": receipt_no,
                        "rcept_no": receipt_no,
                        "corp_name": item.get("corp_name", ""),
                        "stock_code": item.get("stock_code", ""),
                        "report_name": report_name,
                        "report_nm": report_name,
                        "rcept_dt": item.get("rcept_dt", ""),
                        "disclosure_type": disclosure_type,
                        "disclosure_type_label": _DISCLOSURE_TYPE_LABELS.get(
                            disclosure_type,
                            disclosure_type,
                        ),
                        "filer_name": item.get("flr_nm", ""),
                        "remark": item.get("rm", ""),
                        "document_fetched": bool(document_payload.get("fetched")),
                        "document_fetch_error": document_payload.get("fetch_error", ""),
                        "document_text_length": len(document_text),
                        "raw_content_chars": raw_content_length,
                        "content_chars": len(content),
                        "content_truncated": content_truncated,
                        "max_document_length": self.max_document_length,
                        "contains_tables": bool(document_payload.get("contains_tables")),
                        "table_count": int(document_payload.get("table_count", 0)),
                        "table_parse_strategy": document_payload.get(
                            "parse_strategy",
                            "text_only",
                        ),
                        "contains_images": bool(document_payload.get("contains_images")),
                        "image_count": int(document_payload.get("image_count", 0)),
                        "chart_parse_strategy": "not_parsed",
                        "document_file_count": int(document_payload.get("file_count", 0)),
                        "document_parsed_file_count": int(
                            document_payload.get("parsed_file_count", 0)
                        ),
                        "collected_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                    },
                )
            )

        return articles

    async def _fetch_document_payload(self, receipt_no: str) -> dict:
        params = {
            "crtfc_key": self.api_key,
            "rcept_no": receipt_no,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(DART_DOCUMENT_URL, params=params)
            resp.raise_for_status()

        content = resp.content

        if not content.startswith(b"PK"):
            decoded = _decode_bytes(content)
            log.debug(
                "DART document.xml 비ZIP 응답 | receipt_no=%s content=%.200s",
                receipt_no,
                decoded,
            )
            return {
                "text": "",
                "fetched": False,
                "fetch_error": "non_zip_response",
                "raw_text_length": 0,
                "contains_tables": False,
                "table_count": 0,
                "contains_images": False,
                "image_count": 0,
                "file_count": 0,
                "parsed_file_count": 0,
                "parse_strategy": "non_zip_response",
            }

        payload = _extract_payload_from_dart_document(
            content=content,
            receipt_no=receipt_no,
        )
        payload["fetched"] = bool(payload.get("text"))
        payload["fetch_error"] = ""

        return payload


def _extract_payload_from_dart_document(content: bytes, receipt_no: str) -> dict:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            texts: list[str] = []
            file_count = len(archive.namelist())
            parsed_file_count = 0
            contains_tables = False
            table_count = 0
            contains_images = False
            image_count = 0

            for filename in archive.namelist():
                lower_filename = filename.lower()

                if not lower_filename.endswith((".xml", ".html", ".htm", ".txt")):
                    continue

                raw_data = archive.read(filename)
                decoded = _decode_bytes(raw_data)

                parsed = _extract_text_payload_from_markup(
                    markup=decoded,
                    filename=filename,
                    receipt_no=receipt_no,
                )

                text = parsed.get("text", "")

                if text:
                    texts.append(text)
                    parsed_file_count += 1

                if parsed.get("contains_tables"):
                    contains_tables = True

                if parsed.get("contains_images"):
                    contains_images = True

                table_count += int(parsed.get("table_count", 0))
                image_count += int(parsed.get("image_count", 0))

            joined = _normalize_document_text("\n\n".join(texts))

            return {
                "text": joined,
                "raw_text_length": len(joined),
                "contains_tables": contains_tables,
                "table_count": table_count,
                "contains_images": contains_images,
                "image_count": image_count,
                "file_count": file_count,
                "parsed_file_count": parsed_file_count,
                "parse_strategy": "table_preserved_text",
            }

    except zipfile.BadZipFile:
        decoded = _decode_bytes(content)
        parsed = _extract_text_payload_from_markup(
            markup=decoded,
            filename="document.xml",
            receipt_no=receipt_no,
        )

        text = parsed.get("text", "")

        return {
            "text": text,
            "raw_text_length": len(text),
            "contains_tables": bool(parsed.get("contains_tables")),
            "table_count": int(parsed.get("table_count", 0)),
            "contains_images": bool(parsed.get("contains_images")),
            "image_count": int(parsed.get("image_count", 0)),
            "file_count": 1,
            "parsed_file_count": 1 if text else 0,
            "parse_strategy": "table_preserved_text_bad_zip_fallback",
        }


def _extract_text_payload_from_markup(
    markup: str,
    filename: str,
    receipt_no: str,
) -> dict:
    cleaned = re.sub(
        r"<(style|script|noscript)\b[^>]*>.*?</\1>",
        " ",
        markup,
        flags=re.IGNORECASE | re.DOTALL,
    )

    parser = _select_markup_parser(cleaned, filename)

    if _should_use_fast_markup_parser(cleaned, filename):
        parsed = _extract_text_payload_fast(cleaned)
        log.info(
            "DART 문서 빠른 파싱 완료 | receipt_no=%s filename=%s text_len=%d tables=%d images=%d",
            receipt_no,
            filename,
            len(parsed["text"]),
            parsed["table_count"],
            parsed["image_count"],
        )
        return parsed

    log.info(
        "DART 문서 파싱 시작 | receipt_no=%s filename=%s parser=%s cleaned_len=%d",
        receipt_no,
        filename,
        parser,
        len(cleaned),
    )

    soup = BeautifulSoup(cleaned, features=parser)

    for node in soup(["style", "script", "noscript"]):
        node.decompose()

    table_count = len(soup.find_all("table"))
    contains_tables = table_count > 0

    image_count = len(soup.find_all("img"))
    contains_images = image_count > 0

    _replace_tables_with_text(soup)

    text = soup.get_text(separator="\n", strip=True)

    if not text:
        text = strip_html(cleaned)

    normalized_text = _normalize_document_text(text)

    log.info(
        "DART 문서 파싱 완료 | receipt_no=%s filename=%s text_len=%d tables=%d images=%d",
        receipt_no,
        filename,
        len(normalized_text),
        table_count,
        image_count,
    )

    return {
        "text": normalized_text,
        "contains_tables": contains_tables,
        "table_count": table_count,
        "contains_images": contains_images,
        "image_count": image_count,
    }


def _should_use_fast_markup_parser(markup: str, filename: str) -> bool:
    return filename.lower().endswith(".xml") or len(markup) >= 300_000


def _extract_text_payload_fast(markup: str) -> dict:
    table_count = len(re.findall(r"<table\b", markup, flags=re.IGNORECASE))
    image_count = len(re.findall(r"<img\b", markup, flags=re.IGNORECASE))

    def table_repl(match: re.Match[str]) -> str:
        table_text = strip_html(match.group(0))
        table_text = re.sub(r"\s+", " ", table_text).strip()
        return f"\n[표]\n{table_text}\n[/표]\n" if table_text else "\n"

    text_source = re.sub(
        r"<table\b[^>]*>.*?</table>",
        table_repl,
        markup,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = strip_html(text_source)
    normalized_text = _normalize_document_text(text)

    return {
        "text": normalized_text,
        "contains_tables": table_count > 0,
        "table_count": table_count,
        "contains_images": image_count > 0,
        "image_count": image_count,
    }


def _replace_tables_with_text(soup: BeautifulSoup) -> None:
    for idx, table in enumerate(soup.find_all("table"), start=1):
        table_text = _table_to_text(table)

        if not table_text:
            continue

        table.replace_with(soup.new_string(f"\n[표 {idx}]\n{table_text}\n[/표 {idx}]\n"))


def _table_to_text(table: Tag) -> str:
    rows: list[str] = []

    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        values = [cell.get_text(" ", strip=True) for cell in cells]
        values = [value for value in values if value]

        if values:
            rows.append(" | ".join(values))

    return "\n".join(rows)


def _select_markup_parser(markup: str, filename: str) -> str:
    lower_filename = (filename or "").lower()
    head = (markup or "")[:1000].lower()

    if lower_filename.endswith(".xml"):
        return "html.parser"

    if lower_filename.endswith((".html", ".htm")):
        return "html.parser"

    if "<html" in head or "<!doctype html" in head:
        return "html.parser"

    return "html.parser"


def _normalize_document_text(text: str) -> str:
    text = re.sub(r"\.[A-Za-z0-9_-][^{]{0,120}\{[^{}]*\}", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"(?im)^\s*[-A-Za-z]+(?:\s*[-A-Za-z]+)*\s*:\s*[^;\n]+;?\s*$",
        " ",
        text,
    )
    text = re.sub(r"(?im)^\s*/\*.*?\*/\s*$", " ", text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not _looks_like_css_line(line)]

    return "\n".join(lines).strip()


def _looks_like_css_line(line: str) -> bool:
    css_markers = (
        "{",
        "}",
        "font-family",
        "font-size",
        "padding",
        "border",
        "line-height",
        "text-align",
        "vertical-align",
        "text-decoration",
        "background",
        "color:",
    )
    lowered = line.lower()
    return any(marker in lowered for marker in css_markers)


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "euc-kr", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="ignore")


def _parse_dart_date(date_str: str) -> datetime | None:
    try:
        return datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _normalize_name(name: str) -> str:
    return (
        name.replace(" ", "").replace("㈜", "").replace("(주)", "").replace("주식회사", "").lower()
    )


def _disclosure_content(item: dict) -> str:
    return f"{item.get('corp_name', '')} {item.get('flr_nm', '')} {item.get('rm', '')}".strip()


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)

    if value is None:
        return default

    return value.lower() in ("1", "true", "yes", "y")


def _env_list(key: str, default: tuple[str, ...]) -> list[str]:
    value = os.getenv(key)

    if not value:
        return list(default)

    items = [item.strip().upper() for item in value.split(",") if item.strip()]

    return items or list(default)
