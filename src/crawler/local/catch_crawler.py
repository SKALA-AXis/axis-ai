"""Catch 기업분석 리포트 크롤러."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from playwright.async_api import async_playwright

from src.crawler.base import RawArticle
from src.crawler.base_crawler import BaseCrawler

log = logging.getLogger(__name__)

CATCH_ANALYSIS_URL = "https://www.catch.co.kr/Comp/AnalysisCompView?ID={catch_id}"
CATCH_SEARCH_URL = "https://www.catch.co.kr/Search/SearchList?Keyword={keyword}"
DEFAULT_SESSION_PATH = Path(__file__).resolve().parents[3] / ".secrets" / "catch_storage_state.json"
LOCKED_MESSAGE = "입사제안 받기 후 확인하세요."
SWOT_LABELS = ("Strength", "Weakness", "Opportunity", "Threat")
SECTION_MARKERS = (
    "핵심 기업정보",
    "쉽게 이해하는 공식자료",
    "최신 사업성과",
    "캐치 분석자료",
    "채용정보",
    "합격꿀팁",
    "직원이 말하는",
    "어떤 기업일까?",
)

REQUEST_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class CatchParsedReport:
    title: str
    text: str
    report_year: int | None
    report_version: str | None
    sections: list[str]
    section_bodies: dict[str, str]
    swot: dict[str, str | None]
    locked_sections: list[str]
    employee_review: dict[str, Any]
    core_company_info_excerpt: str | None


class CatchCompanyAnalysisCrawler(BaseCrawler):
    """로그인 세션을 재사용해 Catch 기업분석 리포트를 수집한다."""

    def __init__(
        self,
        *,
        company: str,
        catch_id: str,
        company_query: str | None = None,
        session_path: str | Path = DEFAULT_SESSION_PATH,
        headless: bool = True,
        discover_latest: bool = False,
    ) -> None:
        super().__init__(company=company)
        self.configured_catch_id = str(catch_id).strip()
        self.catch_id = self.configured_catch_id
        self.company_query = company_query or company
        self.session_path = Path(session_path)
        self.headless = headless
        self.discover_latest = discover_latest

    async def crawl(self) -> list[RawArticle]:
        if self.discover_latest:
            discovered = await self._discover_latest_analysis_id(self.company_query)
            if discovered:
                self.catch_id = discovered["catch_id"]
                log.info(
                    (
                        "Catch 최신 리포트 ID 탐색 완료 | company=%s "
                        "configured=%s resolved=%s summary=%s"
                    ),
                    self.company,
                    self.configured_catch_id,
                    self.catch_id,
                    discovered.get("comp_summary_url"),
                )
            else:
                log.warning(
                    "Catch 최신 리포트 ID 탐색 실패. config ID로 수집합니다 | company=%s id=%s",
                    self.company,
                    self.configured_catch_id,
                )

        if not self.catch_id:
            log.warning("Catch analysis id가 없어 크롤링을 스킵합니다 | company=%s", self.company)
            return []

        url = CATCH_ANALYSIS_URL.format(catch_id=self.catch_id)
        html, text = await self._fetch_report(url)
        parsed = parse_catch_report(text)

        if not parsed.text:
            return [
                RawArticle(
                    url=url,
                    title=f"Catch 기업분석 리포트 {self.catch_id}",
                    content=None,
                    source_name="catch_company_analysis",
                    source_type="company_analysis",
                    content_type="html",
                    publisher="CATCH",
                    company=self.company,
                    crawl_status="failed",
                    error_message="empty_report_text",
                    extra={"catch_id": self.catch_id},
                )
            ]

        content = build_report_content(parsed)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        return [
            RawArticle(
                id=make_catch_article_id(self.catch_id, url),
                url=url,
                title=parsed.title or f"Catch 기업분석 리포트 {self.catch_id}",
                content=content,
                published_at=datetime.now().astimezone(),
                source_name="catch_company_analysis",
                source_type="company_analysis",
                content_type="html",
                publisher="CATCH",
                company=self.company,
                extra={
                    "catch_id": self.catch_id,
                    "configured_catch_id": self.configured_catch_id,
                    "report_title": parsed.title,
                    "report_year": parsed.report_year,
                    "report_version": parsed.report_version,
                    "sections": parsed.sections,
                    "section_bodies": parsed.section_bodies,
                    "swot": parsed.swot,
                    "swot_available": bool(parsed.swot),
                    "swot_unavailable_reason": None
                    if parsed.swot
                    else "catch_report_has_no_swot_section",
                    "locked_sections": parsed.locked_sections,
                    "locked_message_count": parsed.text.count(LOCKED_MESSAGE),
                    "has_locked_content": LOCKED_MESSAGE in parsed.text,
                    "employee_review": parsed.employee_review,
                    "core_company_info_excerpt": parsed.core_company_info_excerpt,
                    "content_hash": content_hash,
                    "html_length": len(html),
                    "session_state_used": self.session_path.exists(),
                    "discovery_enabled": self.discover_latest,
                    "source_url_kind": "catch_analysis_report",
                },
            )
        ]

    async def _discover_latest_analysis_id(self, query: str) -> dict[str, str] | None:
        if not query:
            return None

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ],
            )
            context = await self._new_context(browser)
            page = await context.new_page()
            try:
                search_url = CATCH_SEARCH_URL.format(keyword=quote(query))
                await page.goto(search_url, wait_until="domcontentloaded", timeout=30_000)
                await page.wait_for_selector("body", state="attached", timeout=15_000)
                await page.wait_for_timeout(800)
                comp_summary_url = await _find_best_comp_summary_url(page, query)
                if not comp_summary_url:
                    return None

                await page.goto(comp_summary_url, wait_until="domcontentloaded", timeout=30_000)
                await page.wait_for_selector("body", state="attached", timeout=15_000)
                await page.wait_for_timeout(800)
                analysis_ids = await _extract_analysis_ids_from_page(page)
                if not analysis_ids:
                    return None

                return {
                    "catch_id": analysis_ids[0],
                    "comp_summary_url": comp_summary_url,
                    "search_url": search_url,
                }
            finally:
                await context.close()
                await browser.close()

    async def _fetch_report(self, url: str) -> tuple[str, str]:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ],
            )
            context = await self._new_context(browser)
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_selector("body", state="attached", timeout=15_000)
            await _scroll_to_bottom(page)
            text = await page.locator("body").inner_text(timeout=15_000)
            html = await page.content()
            await context.close()
            await browser.close()

        return html, normalize_text(text)

    async def _new_context(self, browser):
        context_kwargs: dict[str, Any] = {
            "locale": "ko-KR",
            "viewport": {"width": 1280, "height": 900},
            "user_agent": REQUEST_USER_AGENT,
        }
        if self.session_path.exists():
            context_kwargs["storage_state"] = str(self.session_path)
        else:
            log.warning(
                "Catch 세션 파일 없음. 공개 화면 기준으로 수집합니다 | path=%s",
                self.session_path,
            )

        context = await browser.new_context(**context_kwargs)
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )
        return context


async def _scroll_to_bottom(page) -> None:
    last_height = 0
    for _ in range(8):
        height = await page.evaluate("document.body.scrollHeight")
        if height == last_height:
            break
        last_height = height
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(400)


def parse_catch_report(text: str) -> CatchParsedReport:
    lines = clean_lines(text)
    title = find_report_title(lines)
    report_year, report_version = parse_report_version(title)
    sections = find_sections(lines)
    section_bodies = extract_section_bodies(lines)
    swot = extract_swot(lines)
    locked_sections = extract_locked_sections(lines)
    employee_review = extract_employee_review(lines)
    core_excerpt = extract_between(lines, "핵심 기업정보 > 기업개요", "직원이 말하는", limit=1200)

    return CatchParsedReport(
        title=title,
        text="\n".join(lines),
        report_year=report_year,
        report_version=report_version,
        sections=sections,
        section_bodies=section_bodies,
        swot=swot,
        locked_sections=locked_sections,
        employee_review=employee_review,
        core_company_info_excerpt=core_excerpt,
    )


def build_report_content(report: CatchParsedReport) -> str:
    parts = [f"# {report.title or 'Catch 기업분석 리포트'}"]

    if report.core_company_info_excerpt:
        parts.extend(["", "## 핵심 기업정보", report.core_company_info_excerpt])

    if report.swot:
        parts.extend(["", "## SWOT 분석"])
        for label in SWOT_LABELS:
            value = report.swot.get(label)
            parts.append(f"- {label}: {value or LOCKED_MESSAGE}")

    if report.employee_review:
        parts.extend(["", "## 직원 평가"])
        for key, value in report.employee_review.items():
            parts.append(f"- {key}: {value}")

    parts.extend(["", "## 전체 추출 텍스트", report.text])
    return "\n".join(parts).strip()


def clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()


def find_report_title(lines: list[str]) -> str:
    for line in lines:
        if "분석리포트" in line and "ver" in line:
            return line
    for line in lines:
        if "분석리포트" in line:
            return line
    return ""


def parse_report_version(title: str) -> tuple[int | None, str | None]:
    match = re.search(r"(20\d{2})\s*ver\.?", title or "", re.IGNORECASE)
    if not match:
        return None, None
    year = int(match.group(1))
    return year, f"{year} ver."


def find_sections(lines: list[str]) -> list[str]:
    sections: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if any(marker in line for marker in SECTION_MARKERS):
            if line not in seen:
                seen.add(line)
                sections.append(line)
    return sections


def extract_section_bodies(lines: list[str]) -> dict[str, str]:
    section_indices: list[tuple[str, int]] = []

    for index, line in enumerate(lines):
        if is_report_section_heading(line):
            section_indices.append((line, index))

    bodies: dict[str, str] = {}
    for cursor, (heading, start_index) in enumerate(section_indices):
        end_index = (
            section_indices[cursor + 1][1] if cursor + 1 < len(section_indices) else len(lines)
        )
        body = "\n".join(lines[start_index + 1 : end_index]).strip()
        if body:
            bodies[heading] = body

    return bodies


def is_report_section_heading(line: str) -> bool:
    if " > " not in line:
        return False
    parent, child = [part.strip() for part in line.split(" > ", 1)]
    return parent in {
        "핵심 기업정보",
        "쉽게 이해하는 공식자료",
        "최신 사업성과",
        "캐치 분석자료",
        "채용정보",
        "합격꿀팁",
    } and bool(child)


def extract_swot(lines: list[str]) -> dict[str, str | None]:
    swot: dict[str, str | None] = {}
    for label in SWOT_LABELS:
        value = next_value_after(
            lines,
            label,
            stop_labels=(*SWOT_LABELS, *SECTION_MARKERS),
            case_sensitive=False,
            max_values=1,
        )
        if value:
            swot[label] = value
    return swot


def next_value_after(
    lines: list[str],
    label: str,
    *,
    stop_labels: tuple[str, ...],
    case_sensitive: bool = True,
    max_values: int | None = None,
) -> str | None:
    normalized_label = label if case_sensitive else label.lower()
    normalized_stops = {value if case_sensitive else value.lower() for value in stop_labels}

    for index, line in enumerate(lines):
        compare_line = line if case_sensitive else line.lower()
        if compare_line != normalized_label:
            continue
        values: list[str] = []
        for value in lines[index + 1 :]:
            compare_value = value if case_sensitive else value.lower()
            if compare_value in normalized_stops or any(
                value.startswith(marker) for marker in SECTION_MARKERS
            ):
                break
            values.append(value)
            if value == LOCKED_MESSAGE:
                break
            if max_values is not None and len(values) >= max_values:
                break
            if len(" ".join(values)) > 800:
                break
        return " ".join(values).strip() or None
    return None


def extract_locked_sections(lines: list[str]) -> list[str]:
    locked: list[str] = []
    for index, line in enumerate(lines):
        if line != LOCKED_MESSAGE:
            continue
        label = previous_meaningful_label(lines, index)
        if label and label not in locked:
            locked.append(label)
    return locked


def previous_meaningful_label(lines: list[str], index: int) -> str | None:
    for value in reversed(lines[max(0, index - 5) : index]):
        if value == LOCKED_MESSAGE:
            continue
        if len(value) <= 80:
            return value
    return None


def extract_employee_review(lines: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for index, line in enumerate(lines):
        if line.startswith("직원이 말하는"):
            window = lines[index : index + 30]
            date_value = next(
                (value for value in window if re.match(r"\d{4}\.\d{2}\.\d{2}", value)),
                None,
            )
            score_value = next(
                (value for value in window if re.match(r"^\d+(?:\.\d+)?$", value)),
                None,
            )
            if date_value:
                payload["date"] = date_value
            if score_value:
                payload["score"] = score_value
            labels = {
                "급여 및 복지",
                "조직문화",
                "워라벨",
                "커리어 성장",
                "경영진",
                "긍정적평가",
                "부정적평가",
            }
            payload["labels"] = [value for value in window if value in labels]
            break
    return payload


def extract_between(
    lines: list[str],
    start_label: str,
    end_label: str,
    *,
    limit: int,
) -> str | None:
    start_index = next((i for i, value in enumerate(lines) if value == start_label), None)
    if start_index is None:
        return None
    end_index = next(
        (
            i
            for i, value in enumerate(lines[start_index + 1 :], start_index + 1)
            if value.startswith(end_label)
        ),
        len(lines),
    )
    excerpt = "\n".join(lines[start_index:end_index]).strip()
    return excerpt[:limit] if excerpt else None


def make_catch_article_id(catch_id: str, url: str) -> str:
    raw = f"company_analysis|catch_company_analysis|{catch_id}|{url}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


async def _find_best_comp_summary_url(page, query: str) -> str | None:
    links = await page.locator("a").evaluate_all("""
        els => els.map(a => ({ text: a.innerText.trim(), href: a.href }))
            .filter(x => /\\/Comp\\/CompSummary\\//.test(x.href))
    """)
    if not links:
        return None

    normalized_query = normalize_company_name(query)
    for link in links:
        if normalize_company_name(link.get("text", "")) == normalized_query:
            return str(link["href"])

    return str(links[0]["href"])


async def _extract_analysis_ids_from_page(page) -> list[str]:
    hrefs = await page.locator("a").evaluate_all("""
        els => els.map(a => a.href).filter(href => /AnalysisCompView\\?ID=/.test(href))
    """)
    ids: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        parsed = urlparse(str(href))
        catch_id = parse_qs(parsed.query).get("ID", [""])[0]
        if catch_id and catch_id not in seen:
            seen.add(catch_id)
            ids.append(catch_id)
    return ids


def normalize_company_name(value: str) -> str:
    return re.sub(r"[\s㈜()주식회사]+", "", value or "").lower()
