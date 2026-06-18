# 작성일: 2026-05-07
# 작성자: 박지원
# 변경이력:
#   2026-05-07 박지원 — 크롤러 파이프라인 구축, 백필 크롤러·전처리 보완, peer 프로필 스냅샷 파이프라인 추가
"""IR 자료/PDF 본문 크롤러."""

# ruff: noqa: E501

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, time, timedelta
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urljoin

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from src.config.companies import IR_CONFIG
from src.crawler.article_filter import strip_html
from src.crawler.base import RawArticle
from src.crawler.base_crawler import BaseCrawler
from src.crawler.parsers.pdf_payload import extract_pdf_payload
from src.crawler.playwright_client import PlaywrightClient
from src.db.article_store import article_exists_by_url

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 365
PDF_MAX_TEXT_CHARS = int(os.getenv("IR_PDF_MAX_TEXT_CHARS", "200000"))
PDF_OCR_ENABLED = os.getenv("IR_PDF_ENABLE_OCR", "1").lower() not in {"0", "false", "no"}
PDF_OCR_MIN_TEXT_CHARS = int(os.getenv("IR_PDF_OCR_MIN_TEXT_CHARS", "80"))
PDF_OCR_LANGUAGE = os.getenv("IR_PDF_OCR_LANGUAGE", "kor+eng")
PDF_OCR_DPI = int(os.getenv("IR_PDF_OCR_DPI", "200"))
IR_FORCE_REFRESH = os.getenv("IR_FORCE_REFRESH", "0").lower() in {"1", "true", "yes"}


class IRCrawler(BaseCrawler):
    """회사 IR 페이지에서 PDF 링크를 찾고 본문 텍스트를 저장한다."""

    def __init__(
        self,
        peer_id: str,
        ir_pages: list[str] | None = None,
        lookback_days: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        super().__init__(peer_id)

        config = IR_CONFIG.get(peer_id, {})
        env_key = f"IR_PAGES_{peer_id.upper()}"
        env_pages = [url.strip() for url in os.getenv(env_key, "").split(",") if url.strip()]

        self.ir_pages = ir_pages or env_pages or config.get("pages", [])
        self.fetch_strategy = config.get("fetch_strategy", "playwright_then_httpx")
        self.click_fallback = bool(config.get("click_fallback", False))
        self.lookback_days = lookback_days or int(
            os.getenv("IR_LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS))
        )
        self.start_date = start_date
        self.end_date = end_date
        self.pw = PlaywrightClient()

        log.info(
            (
                "IR 설정 확인 | peer_id=%s lookback_days=%s start_date=%s end_date=%s "
                "fetch_strategy=%s click_fallback=%s pages=%s"
            ),
            self.peer_id,
            self.lookback_days,
            self.start_date,
            self.end_date,
            self.fetch_strategy,
            self.click_fallback,
            self.ir_pages,
        )

    async def crawl(self) -> list[RawArticle]:
        if not self.ir_pages:
            log.warning("IR 페이지 미설정 | peer_id=%s", self.peer_id)
            return []

        articles: list[RawArticle] = []

        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
            for page_url in self.ir_pages:
                try:
                    if self.fetch_strategy == "playwright_click_download":
                        page_articles = await _crawl_by_clicking_downloads(
                            peer_id=self.peer_id,
                            page_url=page_url,
                            lookback_days=self.lookback_days,
                            start_date=self.start_date,
                            end_date=self.end_date,
                        )
                    else:
                        html = await self._fetch_page_html(client, page_url)
                        page_articles = await self._parse_ir_page(
                            client=client,
                            page_url=page_url,
                            html=html,
                        )

                        if not page_articles and self.click_fallback:
                            log.info(
                                "IR 정적 파싱 결과 없음. 클릭 다운로드 fallback 실행 | peer_id=%s url=%s",
                                self.peer_id,
                                page_url,
                            )
                            page_articles = await _crawl_by_clicking_downloads(
                                peer_id=self.peer_id,
                                page_url=page_url,
                                lookback_days=self.lookback_days,
                                start_date=self.start_date,
                                end_date=self.end_date,
                            )

                    articles.extend(page_articles)

                except Exception as e:
                    log.warning(
                        "IR 페이지 수집 실패 | peer_id=%s url=%s error=%s",
                        self.peer_id,
                        page_url,
                        e,
                    )

        if self.peer_id == "sk_ax":
            return _dedupe_articles_by_period(articles)

        return articles

    async def _fetch_page_html(self, client: httpx.AsyncClient, page_url: str) -> str:
        if self.fetch_strategy == "httpx_first":
            html = await _fetch_html(client, page_url)
            if html:
                return html

            return await self.pw.fetch_html(page_url)

        html = await self.pw.fetch_html(page_url)
        if html:
            return html

        return await _fetch_html(client, page_url)

    async def _parse_ir_page(
        self,
        client: httpx.AsyncClient,
        page_url: str,
        html: str,
    ) -> list[RawArticle]:
        soup = BeautifulSoup(html, "html.parser")

        direct_pdf_links = _find_static_pdf_links(
            soup=soup,
            base_url=page_url,
            peer_id=self.peer_id,
        )
        detail_links = [] if self.peer_id == "sk_ax" else _find_detail_links(soup, page_url)

        log.info(
            "IR 링크 탐색 | peer_id=%s page_url=%s direct_pdf=%d detail=%d",
            self.peer_id,
            page_url,
            len(direct_pdf_links),
            len(detail_links),
        )

        candidates: list[tuple[str, str, str]] = []

        for pdf_url, label in direct_pdf_links:
            candidates.append((pdf_url, label, page_url))

        for detail_url, label in detail_links:
            detail_pdf_links = await _fetch_detail_pdf_links(
                client=client,
                detail_url=detail_url,
                peer_id=self.peer_id,
            )

            for pdf_url, pdf_label in detail_pdf_links:
                final_label = pdf_label or label
                candidates.append((pdf_url, final_label, detail_url))

        candidates = _dedupe_candidate_triples(candidates)

        log.info(
            "IR 최종 PDF 후보 | peer_id=%s page_url=%s candidates=%d",
            self.peer_id,
            page_url,
            len(candidates),
        )

        articles: list[RawArticle] = []
        seen_urls: set[str] = set()

        for pdf_url, label, detail_url in candidates:
            if pdf_url in seen_urls:
                continue

            seen_urls.add(pdf_url)

            normalized_label = _normalize_ir_label(self.peer_id, label)

            if not _looks_like_ir_performance_label(normalized_label):
                log.info(
                    "IR 성과자료 외 후보 스킵 | peer_id=%s title=%s url=%s",
                    self.peer_id,
                    strip_html(normalized_label),
                    pdf_url,
                )
                continue

            if _is_script_or_replay_pdf(pdf_url):
                log.info(
                    "IR 스크립트/다시듣기 PDF 스킵 | peer_id=%s title=%s url=%s",
                    self.peer_id,
                    strip_html(normalized_label),
                    pdf_url,
                )
                continue

            if _is_non_ir_pdf(pdf_url):
                log.info(
                    "IR 발표자료 외 PDF 스킵 | peer_id=%s title=%s url=%s",
                    self.peer_id,
                    strip_html(normalized_label),
                    pdf_url,
                )
                continue

            if not IR_FORCE_REFRESH and article_exists_by_url(pdf_url):
                log.info(
                    "IR 기존 URL 스킵 | peer_id=%s title=%s url=%s",
                    self.peer_id,
                    strip_html(normalized_label),
                    pdf_url,
                )
                continue

            pdf_bytes = await _fetch_pdf_bytes(client, pdf_url)

            if not pdf_bytes:
                log.warning(
                    "IR PDF 다운로드 실패로 스킵 | peer_id=%s title=%s url=%s",
                    self.peer_id,
                    strip_html(normalized_label),
                    pdf_url,
                )
                continue

            article = _build_ir_article_from_pdf(
                peer_id=self.peer_id,
                pdf_bytes=pdf_bytes,
                label=normalized_label,
                pdf_url=pdf_url,
                source_page=page_url,
                detail_url=detail_url,
                lookback_days=self.lookback_days,
                start_date=self.start_date,
                end_date=self.end_date,
            )

            if article:
                articles.append(article)

        return articles


async def _fetch_html(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    resp.raise_for_status()
    return resp.text


async def _crawl_by_clicking_downloads(
    peer_id: str,
    page_url: str,
    lookback_days: int,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[RawArticle]:
    articles: list[RawArticle] = []
    seen_keys: set[str] = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        try:
            page = await browser.new_page(user_agent="Mozilla/5.0 AXIS-Crawler/1.0")

            await page.goto(page_url, wait_until="domcontentloaded", timeout=30000)

            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass

            for _ in range(10):
                await page.mouse.wheel(0, 1200)
                await page.wait_for_timeout(500)

            await _mark_ir_download_candidates(page)

            handles = await page.query_selector_all("[data-axis-ir-download='true']")

            log.info(
                "IR 다운로드 클릭 후보 탐색 | peer_id=%s page_url=%s candidates=%d",
                peer_id,
                page_url,
                len(handles),
            )

            for index, handle in enumerate(handles):
                try:
                    label = await handle.get_attribute("data-axis-ir-label")
                    label = _normalize_ir_label(peer_id, label or "")

                    if not _looks_like_ir_performance_label(label):
                        continue

                    key = _normalize_key(label)

                    if key in seen_keys:
                        continue

                    seen_keys.add(key)

                    try:
                        async with page.expect_download(timeout=7000) as download_info:
                            await handle.click(force=True)

                        download = await download_info.value
                        path = await download.path()

                        if not path:
                            continue

                        pdf_bytes = Path(path).read_bytes()

                        if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
                            log.warning(
                                "IR 다운로드 파일이 PDF가 아님 | peer_id=%s title=%s",
                                peer_id,
                                strip_html(label),
                            )
                            continue

                        article = _build_ir_article_from_pdf(
                            peer_id=peer_id,
                            pdf_bytes=pdf_bytes,
                            label=label,
                            pdf_url=download.url or page_url,
                            source_page=page_url,
                            detail_url=page_url,
                            lookback_days=lookback_days,
                            start_date=start_date,
                            end_date=end_date,
                        )

                        if article:
                            articles.append(article)

                    except Exception as e:
                        log.debug(
                            "IR 다운로드 클릭 실패 | peer_id=%s idx=%d title=%s error=%s",
                            peer_id,
                            index,
                            strip_html(label),
                            e,
                        )

                except Exception as e:
                    log.debug(
                        "IR 다운로드 후보 처리 실패 | peer_id=%s idx=%d error=%s",
                        peer_id,
                        index,
                        e,
                    )

        finally:
            await browser.close()

    log.info(
        "IR 다운로드 클릭 수집 완료 | peer_id=%s page_url=%s count=%d",
        peer_id,
        page_url,
        len(articles),
    )

    return articles


async def _mark_ir_download_candidates(page) -> None:
    await page.evaluate(
        """
        () => {
            function clean(text) {
                return (text || '').replace(/\\s+/g, ' ').trim();
            }

            function normalize(text) {
                return clean(text).replace(/\\s/g, '');
            }

            function extractTitle(text) {
                const cleaned = clean(text);

                const patterns = [
                    /20\\d{2}\\s*년\\s*(?:[1-4]\\s*분기|4\\s*분기\\s*\\/\\s*연간|연간)\\s*경영실적\\s*발표?/,
                    /20\\d{2}\\s*년\\s*(?:[1-4]\\s*분기|4\\s*분기\\s*\\/\\s*연간|연간)\\s*경영실적/,
                    /20\\d{2}\\s*년\\s*(?:[1-4]\\s*분기|4\\s*분기\\s*\\/\\s*연간|연간).{0,30}(?:경영실적|실적|실적자료|실적발표|발표자료|IR\\s*자료|IR자료)/,
                    /20\\d{2}\\s*(?:[1-4]Q|Q[1-4]|[1-4]\\s*Q).{0,50}?(?:Earnings|Presentation|Results|Financial Results|IR Materials|IR)/i,
                    /20\\d{2}.{0,50}?(?:Earnings|Presentation|Results|Financial Results|IR Materials|IR)/i
                ];

                for (const pattern of patterns) {
                    const match = cleaned.match(pattern);
                    if (match) {
                        return clean(match[0]);
                    }
                }

                return '';
            }

            function isBlocked(text) {
                const value = normalize(text).toLowerCase();

                const blocked = [
                    '이사회',
                    '위원회',
                    '감사',
                    '정관',
                    '지배구조',
                    '규정',
                    'rules',
                    'committee',
                    'director',
                    'audit',
                    'governance'
                ];

                return blocked.some(keyword => value.includes(keyword.toLowerCase()));
            }

            function isDownloadElement(el) {
                const text = clean(
                    el.innerText ||
                    el.getAttribute('aria-label') ||
                    el.getAttribute('title') ||
                    el.getAttribute('href') ||
                    ''
                ).toLowerCase();

                return (
                    text.includes('다운로드') ||
                    text.includes('download') ||
                    text.includes('pdf') ||
                    text.includes('.pdf')
                );
            }

            function findSmallestCard(titleElement, titleText) {
                let current = titleElement.parentElement;
                const titleKey = normalize(titleText);

                for (let i = 0; i < 14 && current; i++) {
                    const text = clean(current.innerText);
                    const textKey = normalize(text);
                    const downloadEls = Array.from(current.querySelectorAll('a, button')).filter(isDownloadElement);
                    const yearMatches = text.match(/20\\d{2}/g) || [];

                    if (
                        textKey.includes(titleKey) &&
                        downloadEls.length >= 1 &&
                        yearMatches.length <= 2 &&
                        !isBlocked(text)
                    ) {
                        return current;
                    }

                    current = current.parentElement;
                }

                return null;
            }

            const textElements = Array.from(
                document.querySelectorAll('h1,h2,h3,h4,h5,h6,p,span,strong,em,dt,dd,li,div')
            );

            const seenTitles = new Set();
            const pairs = [];

            for (const el of textElements) {
                const ownText = clean(el.innerText);

                if (!ownText || ownText.length > 180) {
                    continue;
                }

                const title = extractTitle(ownText);

                if (!title) {
                    continue;
                }

                const titleKey = normalize(title);

                if (seenTitles.has(titleKey)) {
                    continue;
                }

                if (isBlocked(title)) {
                    continue;
                }

                const card = findSmallestCard(el, title);

                if (!card) {
                    continue;
                }

                const downloadEl = Array.from(card.querySelectorAll('a, button')).find(isDownloadElement);

                if (!downloadEl) {
                    continue;
                }

                seenTitles.add(titleKey);
                pairs.push({ title, downloadEl });
            }

            for (const pair of pairs) {
                pair.downloadEl.setAttribute('data-axis-ir-download', 'true');
                pair.downloadEl.setAttribute('data-axis-ir-label', pair.title);
            }
        }
        """
    )


def _find_static_pdf_links(
    soup: BeautifulSoup,
    base_url: str,
    peer_id: str,
) -> list[tuple[str, str]]:
    if peer_id == "lg_cns":
        return _find_lg_cns_pdf_links(soup, base_url)

    return _find_pdf_links(soup, base_url, peer_id)


def _find_lg_cns_pdf_links(
    soup: BeautifulSoup,
    base_url: str,
) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []

    for anchor in soup.select("a[href]"):
        href = _clean_candidate_url(anchor.get("href", ""))
        text = anchor.get_text(" ", strip=True)

        if not href:
            continue

        href_lower = href.lower()
        text_lower = text.lower()

        if (
            ".pdf" not in href_lower
            and "download" not in href_lower
            and "다운로드" not in text
            and "pdf" not in text_lower
        ):
            continue

        url = urljoin(base_url, href)

        if not url.startswith("http"):
            continue

        label = _find_lg_cns_card_label(anchor) or _filename_title(url)
        label = _normalize_lg_cns_label(label)

        if not _looks_like_ir_performance_label(label):
            continue

        links.append((url, label))

    return _dedupe_pairs(links)


def _find_lg_cns_card_label(anchor) -> str:
    current = anchor

    for _ in range(12):
        current = current.parent

        if not current:
            break

        text = current.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        label = _normalize_lg_cns_label(text)

        if _looks_like_ir_performance_label(label):
            return label

    previous = anchor.find_previous(string=re.compile(r"20\d{2}\s*년.*?(?:경영실적|실적발표)"))

    if previous:
        return strip_html(str(previous))

    return ""


def _find_pdf_links(
    soup: BeautifulSoup,
    base_url: str,
    peer_id: str,
) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []

    for anchor in soup.select("a[href]"):
        href = _clean_candidate_url(anchor.get("href", ""))
        text = anchor.get_text(" ", strip=True)

        if not href:
            continue

        if _looks_like_pdf_or_download(href, text):
            url = urljoin(base_url, href)

            if url.startswith("http"):
                if peer_id == "sk_ax" and not _is_sk_ax_presentation_url(url):
                    continue

                label = _nearest_static_label(anchor, peer_id) or text or _filename_title(url)
                label = _normalize_ir_label(peer_id, label)

                if _looks_like_ir_performance_label(label):
                    links.append((url, label))

    for element in soup.select(
        "[onclick], [data-url], [data-href], [data-file], "
        "[data-file-url], [data-download], [data-atch-file-id]"
    ):
        text = element.get_text(" ", strip=True)
        label = _nearest_static_label(element, peer_id) or text

        raw_values = [
            element.get("onclick", ""),
            element.get("data-url", ""),
            element.get("data-href", ""),
            element.get("data-file", ""),
            element.get("data-file-url", ""),
            element.get("data-download", ""),
            element.get("data-atch-file-id", ""),
        ]

        for raw_value in raw_values:
            for url in _extract_urls_from_text(raw_value, base_url):
                if _looks_like_pdf_or_download(url, label):
                    if peer_id == "sk_ax" and not _is_sk_ax_presentation_url(url):
                        continue

                    normalized_label = _normalize_ir_label(peer_id, label)

                    if _looks_like_ir_performance_label(normalized_label):
                        links.append((url, normalized_label))

    return _dedupe_pairs(links)


def _nearest_static_label(element, peer_id: str) -> str:
    year_text = element.find_previous(string=re.compile(r"20\d{2}"))

    if year_text:
        year = strip_html(str(year_text))
        own_text = element.get_text(" ", strip=True)
        parent_text = ""

        current = element.parent
        for _ in range(5):
            if not current:
                break

            text = current.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()

            if "분기" in text or "경영실적" in text or "발표자료" in text:
                parent_text = text
                break

            current = current.parent

        combined = f"{year} {parent_text or own_text}"
        label = _normalize_ir_label(peer_id, combined)

        if _looks_like_ir_performance_label(label):
            return label

    current = element

    for _ in range(10):
        current = current.parent

        if not current:
            break

        text = current.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        if len(text) > 220:
            continue

        label = _normalize_ir_label(peer_id, text)

        if _looks_like_ir_performance_label(label):
            return label

    previous = element.find_previous(
        string=re.compile(
            r"(20\d{2}\s*년.*?(?:경영실적|실적발표|발표자료)|20\d{2}.{0,50}?(?:Earnings|Presentation|IR))",
            re.IGNORECASE,
        )
    )

    if previous:
        return strip_html(str(previous))

    return ""


def _find_detail_links(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []

    for anchor in soup.select("a[href]"):
        href = _clean_candidate_url(anchor.get("href", ""))
        text = anchor.get_text(" ", strip=True)

        if not href:
            continue

        if _looks_like_pdf_or_download(href, text):
            continue

        if not _looks_like_detail_link(href, text):
            continue

        detail_url = urljoin(base_url, href)

        if detail_url.startswith("http"):
            row = anchor.find_parent("tr")
            label = row.get_text(" ", strip=True) if row else text
            links.append((detail_url, label or text))

    for element in soup.select("[onclick], [data-url], [data-href]"):
        text = element.get_text(" ", strip=True)
        row = element.find_parent("tr")
        label = row.get_text(" ", strip=True) if row else text

        raw_values = [
            element.get("onclick", ""),
            element.get("data-url", ""),
            element.get("data-href", ""),
        ]

        for raw_value in raw_values:
            for url in _extract_urls_from_text(raw_value, base_url):
                if _looks_like_detail_link(url, label):
                    links.append((url, label or text))

    return _dedupe_pairs(links)


async def _fetch_detail_pdf_links(
    client: httpx.AsyncClient,
    detail_url: str,
    peer_id: str,
) -> list[tuple[str, str]]:
    try:
        resp = await client.get(
            detail_url,
            headers={
                "User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        links = _find_pdf_links(soup, detail_url, peer_id)

        log.info(
            "IR 상세 페이지 PDF 탐색 | detail_url=%s pdf_count=%d",
            detail_url,
            len(links),
        )

        return _dedupe_pairs(links)

    except Exception as e:
        log.warning(
            "IR 상세 페이지 수집 실패 | detail_url=%s error=%s",
            detail_url,
            e,
        )
        return []


def _normalize_ir_label(peer_id: str, label: str) -> str:
    text = strip_html(label or "")
    text = re.sub(r"\s+", " ", text).strip()

    if peer_id == "hyundai_autoever":
        return _normalize_hyundai_label(text)

    if peer_id == "lg_cns":
        return _normalize_lg_cns_label(text)

    if peer_id == "sk_ax":
        return _normalize_sk_ax_label(text)

    return _normalize_generic_ir_label(text)


def _normalize_hyundai_label(label: str) -> str:
    text = strip_html(label or "")
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"(20\d{2}\s*년\s*(?:[1-4]\s*분기|4\s*분기\s*/\s*연간|연간)\s*경영실적\s*발표?)",
        r"(20\d{2}\s*년\s*(?:[1-4]\s*분기|4\s*분기\s*/\s*연간|연간)\s*경영실적)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()

    return text[:120]


def _normalize_lg_cns_label(label: str) -> str:
    text = strip_html(label or "")
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"(20\d{2}\s*년\s*(?:[1-4]\s*분기|4\s*분기\s*/\s*연간|연간)\s*경영실적\s*발표)",
        r"(20\d{2}\s*년\s*(?:[1-4]\s*분기|4\s*분기\s*/\s*연간|연간).{0,20}실적.{0,10}발표)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()

    return text[:120]


def _normalize_sk_ax_label(label: str) -> str:
    text = strip_html(label or "")
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"(20\d{2})\s*[\.\-_/]?\s*([1-4])\s*Q.{0,80}?(?:Earnings\s*Briefing|Presentation|IR)",
        r"(20\d{2}).{0,80}?(?:Earnings\s*Briefing|Presentation|IR).{0,20}?([1-4])\s*Q",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

        if not matches:
            continue

        match = matches[-1]
        first = match.group(1)
        second = match.group(2)

        if first.startswith("20"):
            year = int(first)
            quarter = int(second)
        else:
            quarter = int(first)
            year = _normalize_year(int(second))

        if 1 <= quarter <= 4:
            return f"{year} {quarter}Q SK Inc. Presentation"

    return _normalize_generic_ir_label(text)


def _normalize_generic_ir_label(label: str) -> str:
    text = strip_html(label or "")
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"(20\d{2}\s*년\s*(?:[1-4]\s*분기|4\s*분기\s*/\s*연간|연간).{0,40}?(?:경영실적|실적발표|발표자료))",
        r"(20\d{2}.{0,60}?(?:Earnings|Presentation|IR))",
        r"(20\d{2}\s+[1-4]\s*/\s*4\s*분기.{0,40}?(?:경영실적|발표자료|스크립트))",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()

    return text[:120]


def _looks_like_ir_performance_label(label: str) -> bool:
    value = strip_html(label or "").lower().replace(" ", "")

    if not value:
        return False

    blocked_keywords = [
        "이사회",
        "위원회",
        "감사",
        "정관",
        "지배구조",
        "규정",
        "rules",
        "committee",
        "director",
        "audit",
        "governance",
    ]

    if any(keyword.lower().replace(" ", "") in value for keyword in blocked_keywords):
        return False

    allowed_keywords = [
        "경영실적",
        "실적발표",
        "실적",
        "실적자료",
        "발표자료",
        "분기",
        "연간",
        "earnings",
        "performance",
        "presentation",
        "businessperformance",
        "results",
        "financialresults",
        "quarterlyresults",
        "irmaterials",
        "ir자료",
    ]

    return any(keyword.lower().replace(" ", "") in value for keyword in allowed_keywords)


def _is_script_or_replay_pdf(pdf_url: str) -> bool:
    value = unquote(pdf_url or "").lower().replace(" ", "")

    return any(
        keyword in value
        for keyword in [
            "script",
            "스크립트",
            "transcript",
            "replay",
            "다시듣기",
        ]
    )


def _is_non_ir_pdf(pdf_url: str) -> bool:
    value = unquote(pdf_url or "").lower().replace(" ", "")

    return any(
        keyword in value
        for keyword in [
            "_upload/busi/",
            "_upload/actl/",
            "resources/download/esg/",
            "sustainability_report",
            "annual_report",
            "businessreports",
            "audited_report",
            "consolidated_audited_report",
            "감사보고서",
            "검토보고서",
        ]
    )


def _is_sk_ax_presentation_url(url: str) -> bool:
    value = unquote(url or "").lower()

    if not value.endswith(".pdf"):
        return False

    return "/pres/" in value or "sk_inc_presentation" in value


def _build_ir_article_from_pdf(
    peer_id: str,
    pdf_bytes: bytes,
    label: str,
    pdf_url: str,
    source_page: str,
    detail_url: str,
    lookback_days: int,
    start_date: date | None = None,
    end_date: date | None = None,
) -> RawArticle | None:
    pdf_payload = _extract_pdf_payload(pdf_bytes)
    pdf_text = pdf_payload.get("text", "")

    if not pdf_text:
        log.warning(
            "IR PDF 본문 추출 실패로 스킵 | peer_id=%s title=%s url=%s",
            peer_id,
            strip_html(label),
            pdf_url,
        )
        return None

    date_info = _resolve_ir_date(
        peer_id=peer_id,
        label=label,
        detail_url=detail_url,
        pdf_url=pdf_url,
        pdf_text=pdf_text,
    )
    published_at = date_info["published_at"]

    if not published_at:
        log.info(
            "IR 날짜/분기 추정 실패로 스킵 | peer_id=%s title=%s url=%s",
            peer_id,
            strip_html(label),
            pdf_url,
        )
        return None

    if not _is_within_ir_window(
        published_at,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
    ):
        log.info(
            (
                "IR 수집 window 제외 | peer_id=%s title=%s 기준일=%s "
                "start_date=%s end_date=%s lookback_days=%s source=%s url=%s"
            ),
            peer_id,
            strip_html(label),
            published_at.isoformat(),
            start_date,
            end_date,
            lookback_days,
            date_info.get("date_source"),
            pdf_url,
        )
        return None

    title = _format_ir_title(
        peer_id=peer_id,
        label=label,
        pdf_url=pdf_url,
        date_info=date_info,
    )

    return RawArticle(
        url=pdf_url,
        title=title,
        content=pdf_text,
        published_at=published_at,
        source_name="ir_pdf",
        peer_id=peer_id,
        source_type="ir",
        content_type="pdf",
        company=[peer_id],
        extra={
            "source_page": source_page,
            "detail_url": detail_url,
            "pdf_url": pdf_url,
            "pdf_text_chars": len(pdf_text),
            "lookback_days": lookback_days,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "published_at": published_at.isoformat(),
            "date_info": _serialize_date_info(date_info),
            "raw_label": label,
            "pdf_parse_strategy": pdf_payload.get("pdf_parse_strategy", "text_only"),
            "pdf_pages": pdf_payload.get("page_count", 0),
            "pdf_parsed_pages": pdf_payload.get("parsed_page_count", 0),
            "contains_images": pdf_payload.get("contains_images", False),
            "image_count": pdf_payload.get("image_count", 0),
            "drawing_count": pdf_payload.get("drawing_count", 0),
            "ocr_applied": pdf_payload.get("ocr_applied", False),
            "ocr_pages": pdf_payload.get("ocr_pages", []),
            "contains_tables": None,
            "table_count": None,
            "table_parse_strategy": pdf_payload.get(
                "table_parse_strategy",
                "pdf_text_blocks",
            ),
            "chart_parse_strategy": pdf_payload.get(
                "chart_parse_strategy",
                "not_parsed",
            ),
            "pdf_page_blocks": pdf_payload.get("pages", []),
            "collected_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _is_within_ir_window(
    published_at: datetime,
    *,
    lookback_days: int,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    if start_date or end_date:
        start = datetime.combine(start_date, time.min) if start_date else datetime.min
        end = datetime.combine(end_date, time.max) if end_date else datetime.max
        value = published_at.replace(tzinfo=None)
        return start <= value <= end

    return published_at >= datetime.now() - timedelta(days=lookback_days)


def _format_ir_title(
    peer_id: str,
    label: str,
    pdf_url: str,
    date_info: dict,
) -> str:
    title = strip_html(label) or _filename_title(pdf_url)

    if peer_id == "hyundai_autoever":
        year = date_info.get("year")
        quarter = date_info.get("quarter")

        if year and quarter:
            suffix = "/연간" if quarter == 4 else ""
            return f"현대오토에버 {year}년 {quarter}분기{suffix} 경영실적"

        return title

    if peer_id == "sk_ax":
        year = date_info.get("year")
        quarter = date_info.get("quarter")

        if year and quarter:
            return f"SK AX {year}년 {quarter}분기 IR Presentation"

        return title

    if peer_id != "samsung_sds":
        return title

    normalized_title = title.replace(" ", "")
    if "다시듣기" not in normalized_title and "발표자료스크립트" not in normalized_title:
        return title

    year = date_info.get("year")
    quarter = date_info.get("quarter")

    if not year or not quarter:
        return title

    source_text = pdf_url.lower()
    material_type = (
        "실적발표 스크립트"
        if "script" in source_text or "스크립트" in source_text
        else "실적발표 자료"
    )

    return f"Samsung SDS {year}년 {quarter}분기 {material_type}"


async def _fetch_pdf_bytes(client: httpx.AsyncClient, pdf_url: str) -> bytes:
    try:
        resp = await client.get(
            pdf_url,
            headers={
                "User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0",
                "Accept": "application/pdf,*/*",
            },
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        content = resp.content

        if "pdf" in content_type or content.startswith(b"%PDF"):
            return content

        log.debug(
            "PDF가 아닌 응답 스킵 | url=%s content_type=%s",
            pdf_url,
            content_type,
        )
        return b""

    except Exception as e:
        log.debug(
            "IR PDF 다운로드 실패 | url=%s error=%s",
            pdf_url,
            e,
        )
        return b""


async def _fetch_pdf_text(client: httpx.AsyncClient, pdf_url: str) -> str:
    pdf_bytes = await _fetch_pdf_bytes(client, pdf_url)

    if not pdf_bytes:
        return ""

    return _extract_pdf_payload(pdf_bytes).get("text", "")


def _extract_pdf_payload(pdf_bytes: bytes) -> dict:
    return extract_pdf_payload(pdf_bytes, max_text_chars=PDF_MAX_TEXT_CHARS)


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"(?<=\dQ\d{2})(?=\dQ\d{2})", " ", text)
    text = re.sub(r"(?<=\d)(?=Q\d{2})", " ", text)
    text = re.sub(r"([A-Za-z])(?=[가-힣])", r"\1 ", text)
    text = re.sub(r"([가-힣])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"([가-힣])(?=\d)", r"\1 ", text)
    text = re.sub(r"(\d)(?=[가-힣])", r"\1 ", text)
    return text.strip()


def _needs_ocr_page(page_text: str, image_count: int) -> bool:
    if not PDF_OCR_ENABLED or image_count <= 0:
        return False

    alpha_numeric_count = len(re.findall(r"[0-9A-Za-z가-힣]", page_text or ""))
    return alpha_numeric_count < PDF_OCR_MIN_TEXT_CHARS


def _extract_pdf_page_ocr_text(page) -> str:
    try:
        textpage = page.get_textpage_ocr(
            flags=0,
            language=PDF_OCR_LANGUAGE,
            dpi=PDF_OCR_DPI,
            full=False,
        )
        return _clean_pdf_text(page.get_text("text", sort=True, textpage=textpage))
    except Exception as exc:
        log.debug("IR PDF OCR fallback 실패 | page=%s error=%s", getattr(page, "number", "?"), exc)
        return ""


def _extract_pdf_page_blocks(page) -> list[dict]:
    blocks: list[dict] = []

    for block in page.get_text("blocks", sort=True):
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]
        text = re.sub(r"\s+", " ", text or "").strip()

        if not text:
            continue

        blocks.append(
            {
                "bbox": [
                    round(float(x0), 2),
                    round(float(y0), 2),
                    round(float(x1), 2),
                    round(float(y1), 2),
                ],
                "text": text,
            }
        )

    return blocks


def _looks_like_pdf_or_download(href: str, text: str) -> bool:
    value = f"{href} {text}".lower()

    return (
        ".pdf" in value
        or "download" in value
        or "downfile" in value
        or "filedown" in value
        or "filedownload" in value
        or "attach" in value
        or "attachment" in value
        or "첨부" in text
        or "다운로드" in text
        or "자료" in text
        or "pdf" in text.lower()
        or "원문보기" in text
    )


def _looks_like_detail_link(href: str, text: str) -> bool:
    value = f"{href} {text}".lower()

    return (
        "view" in value
        or "detail" in value
        or "read" in value
        or "irdata" in value
        or "idx=" in value
        or "seq=" in value
        or "no=" in value
        or "bbs" in value
        or "board" in value
    )


def _extract_urls_from_text(text: str, base_url: str) -> list[str]:
    if not text:
        return []

    candidates: list[str] = []

    candidates.extend(
        re.findall(
            r"""['"]([^'"]+\.(?:pdf|PDF)(?:\?[^'"]*)?)['"]""",
            text,
        )
    )

    candidates.extend(
        re.findall(
            r"""(?:href|src)=["']([^"']+)["']""",
            text,
            flags=re.IGNORECASE,
        )
    )

    candidates.extend(
        re.findall(
            r"""['"]([^'"]*(?:download|downFile|fileDown|fileDownload|attach|attachment)[^'"]*)['"]""",
            text,
            flags=re.IGNORECASE,
        )
    )

    urls: list[str] = []

    for candidate in candidates:
        candidate = _clean_candidate_url(candidate)

        if not candidate:
            continue

        candidate_lower = candidate.lower()

        if not (
            ".pdf" in candidate_lower
            or "download" in candidate_lower
            or "downfile" in candidate_lower
            or "filedown" in candidate_lower
            or "filedownload" in candidate_lower
            or "attach" in candidate_lower
            or "attachment" in candidate_lower
        ):
            continue

        url = urljoin(base_url, candidate)

        if url.startswith("http"):
            urls.append(url)

    return list(dict.fromkeys(urls))


def _clean_candidate_url(candidate: str) -> str:
    value = unescape(candidate or "").strip()
    value = value.replace("\xa0", "").replace("&nbsp;", "")

    if not value:
        return ""

    lower_value = value.lower()

    if (
        value.startswith("#")
        or lower_value.startswith("javascript")
        or lower_value.startswith("mailto:")
        or lower_value.startswith("tel:")
    ):
        return ""

    if any(token in value for token in ["<", ">", "\n", "\r", "\t"]):
        return ""

    if len(value) > 500:
        return ""

    return value


def _dedupe_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    results: list[tuple[str, str]] = []

    for url, label in pairs:
        if url in seen:
            continue

        seen.add(url)
        results.append((url, label))

    return results


def _dedupe_candidate_triples(
    triples: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    seen: set[str] = set()
    results: list[tuple[str, str, str]] = []

    for pdf_url, label, detail_url in triples:
        if pdf_url in seen:
            continue

        seen.add(pdf_url)
        results.append((pdf_url, label, detail_url))

    return results


def _dedupe_articles_by_period(articles: list[RawArticle]) -> list[RawArticle]:
    deduped: dict[tuple[int | None, int | None] | tuple[str, str], RawArticle] = {}

    for article in articles:
        date_info = (article.extra or {}).get("date_info", {})
        year = date_info.get("year")
        quarter = date_info.get("quarter")

        if year and quarter:
            key: tuple[int | None, int | None] | tuple[str, str] = (year, quarter)
        else:
            key = ("url", article.url)

        if key not in deduped:
            deduped[key] = article

    return sorted(
        deduped.values(),
        key=lambda article: article.published_at or datetime.min,
        reverse=True,
    )


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", "", strip_html(text or "").lower())


def _resolve_ir_date(
    peer_id: str,
    label: str,
    detail_url: str,
    pdf_url: str,
    pdf_text: str,
) -> dict:
    if peer_id in {"hyundai_autoever", "sk_ax"}:
        period_from_text = _resolve_ir_period_from_pdf_text(pdf_text)

        if period_from_text:
            return period_from_text

    listing_date = _resolve_ir_period_from_listing(
        label=label,
        pdf_url=pdf_url,
    )

    if not listing_date.get("date_parse_failed"):
        return listing_date

    search_targets = [
        ("label", label),
        ("detail_url", detail_url),
        ("pdf_url", pdf_url),
    ]

    for source, text in search_targets:
        parsed_date = _guess_date(text)

        if parsed_date:
            return {
                "published_at": parsed_date,
                "date_source": source,
                "date_parse_type": "exact_or_month",
                "year": parsed_date.year,
                "quarter": _month_to_quarter(parsed_date.month),
                "date_parse_failed": False,
            }

    period_from_text = _resolve_ir_period_from_pdf_text(pdf_text)

    if period_from_text:
        return period_from_text

    return {
        "published_at": None,
        "date_source": None,
        "date_parse_type": None,
        "year": None,
        "quarter": None,
        "date_parse_failed": True,
    }


def _resolve_ir_period_from_pdf_text(pdf_text: str) -> dict | None:
    quarter_info = _guess_year_quarter(_pdf_lead_title_text(pdf_text))

    if not quarter_info:
        quarter_info = _guess_year_quarter(pdf_text[:3000])

    if not quarter_info:
        return None

    year, quarter = quarter_info
    period_date = _quarter_end_date(year, quarter)

    return {
        "published_at": period_date,
        "date_source": "pdf_text",
        "date_parse_type": "pdf_text_year_quarter",
        "year": year,
        "quarter": quarter,
        "is_period_proxy": True,
        "date_parse_failed": False,
    }


def _pdf_lead_title_text(pdf_text: str) -> str:
    value = strip_html(pdf_text or "")
    first_page = value.split("[PAGE 2]", 1)[0]
    first_page = first_page.replace("[PAGE 1]", " ")
    first_page = re.sub(r"\s+", " ", first_page).strip()

    return first_page[:700]


def _resolve_ir_period_from_listing(
    label: str,
    pdf_url: str = "",
) -> dict:
    targets = [
        ("label", label),
        ("pdf_url", pdf_url),
    ]

    for source, text in targets:
        parsed = _parse_year_quarter_from_text(text)

        if parsed:
            year, quarter = parsed
            period_date = _quarter_end_date(year, quarter)

            return {
                "published_at": period_date,
                "date_source": source,
                "date_parse_type": "listing_year_quarter",
                "year": year,
                "quarter": quarter,
                "is_period_proxy": True,
                "date_parse_failed": False,
            }

    for source, text in targets:
        parsed_year = _parse_year_annual_from_text(text)

        if parsed_year:
            return {
                "published_at": datetime(parsed_year, 12, 31),
                "date_source": source,
                "date_parse_type": "listing_year_annual",
                "year": parsed_year,
                "quarter": 4,
                "is_period_proxy": True,
                "date_parse_failed": False,
            }

    return {
        "published_at": None,
        "date_source": None,
        "date_parse_type": None,
        "year": None,
        "quarter": None,
        "date_parse_failed": True,
    }


def _parse_year_quarter_from_text(text: str) -> tuple[int, int] | None:
    value = strip_html(text or "")
    value = re.sub(r"\s+", " ", value).strip()

    patterns = [
        r"(20\d{2})\s*년\s*([1-4])\s*분기\s*/\s*연간",
        r"(20\d{2})\s*년\s*([1-4])\s*분기",
        r"(20\d{2})\s*[\.\-_/]?\s*([1-4])\s*분기",
        r"(20\d{2})\s*[\.\-_/]?\s*([1-4])\s*Q",
        r"(20\d{2})\s*([1-4])\s*Q",
        r"(20\d{2})\s*Q\s*([1-4])",
        r"([1-4])\s*분기.{0,30}?(20\d{2})\s*년",
        r"([1-4])\s*분기.{0,30}?(20\d{2})(?!\d)",
        r"([1-4])\s*Q\s*(\d{2})(?!\d)",
        r"FY\s*(20\d{2}).{0,10}?([1-4])\s*Q",
    ]

    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)

        if not match:
            continue

        first = match.group(1)
        second = match.group(2)

        if first.startswith("20"):
            year = int(first)
            quarter = int(second)
        else:
            quarter = int(first)
            year = _normalize_year(int(second))

        if 1 <= quarter <= 4:
            return year, quarter

    match = re.search(r"(20\d{2}).{0,80}?([1-4])\s*/\s*4\s*분기", value)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None


def _quarter_end_date(year: int, quarter: int) -> datetime:
    month_day = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31),
    }
    month, day = month_day[quarter]
    return datetime(year, month, day)


def _parse_year_annual_from_text(text: str) -> int | None:
    value = strip_html(text or "")
    value = re.sub(r"\s+", " ", value).strip()

    patterns = [
        r"(20\d{2})\s*년\s*연간",
        r"(20\d{2})\s*연간",
        r"(20\d{2})\s*Annual",
        r"(20\d{2}).{0,20}?Earnings\s*Briefing",
    ]

    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)

        if match:
            return int(match.group(1))

    return None


def _serialize_date_info(date_info: dict) -> dict:
    serialized = dict(date_info)

    if isinstance(serialized.get("published_at"), datetime):
        serialized["published_at"] = serialized["published_at"].isoformat()

    return serialized


def _guess_date(text: str) -> datetime | None:
    value = strip_html(text or "")

    if re.search(r"(20\d{2}).{0,10}?[1-4]\s*(?:Q|분기)", value, flags=re.IGNORECASE):
        return None

    match = re.search(
        r"(20\d{2})[.\-/년\s]+(\d{1,2})(?:[.\-/월\s]+(\d{1,2}))?",
        value,
    )

    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3) or 1)

    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def _guess_year_quarter(text: str) -> tuple[int, int] | None:
    value = strip_html(text or "")

    patterns = [
        r"(?:FY)?\s*(20\d{2}).{0,15}?([1-4])\s*(?:Q|분기)",
        r"(20\d{2})\s*[\.\-_/]?\s*([1-4])\s*Q",
        r"([1-4])\s*(?:Q|분기).{0,15}?(20\d{2})",
        r"(20\d{2}).{0,15}?(?:년)?\s*([1-4])\s*분기",
        r"(20\d{2}).{0,15}?([1-4])Q",
    ]

    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)

        if not match:
            continue

        first = match.group(1)
        second = match.group(2)

        if first.startswith("20"):
            year = _normalize_year(int(first))
            quarter = int(second)
        else:
            quarter = int(first)
            year = _normalize_year(int(second))

        if 1 <= quarter <= 4:
            return year, quarter

    return None


def _month_to_quarter(month: int) -> int:
    if month <= 3:
        return 1

    if month <= 6:
        return 2

    if month <= 9:
        return 3

    return 4


def _normalize_year(year: int) -> int:
    if 0 <= year < 100:
        return 2000 + year

    return year


def _filename_title(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    return strip_html(tail.replace("%20", " ").replace("_", " ").replace("-", " "))
