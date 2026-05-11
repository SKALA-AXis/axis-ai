"""SK AX 공식 사이트 정적/반정적 페이지 크롤러.

뉴스룸처럼 계속 쌓이는 게시글이 아니라 회사 소개, 서비스, 인사이트 등
공식 사이트의 지식 베이스성 페이지를 URL 단위로 수집하고 섹션 구조를 보존한다.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup, Tag

from src.crawler.base import RawArticle
from src.crawler.base_crawler import BaseCrawler
from src.crawler.playwright_client import PlaywrightClient
from src.crawler.result_writer import DEFAULT_RESULTS_DIR

log = logging.getLogger(__name__)

SKAX_BASE_URL = "https://www.skax.co.kr"
SKAX_HOME_URL = f"{SKAX_BASE_URL}/"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

DEFAULT_SEED_URLS = [
    SKAX_HOME_URL,
    f"{SKAX_BASE_URL}/sitemap",
    f"{SKAX_BASE_URL}/axgenticwire",
    f"{SKAX_BASE_URL}/company/about",
    f"{SKAX_BASE_URL}/ax-services/aicc/",
    f"{SKAX_BASE_URL}/ax-services/new-paradigm-operation",
]

EXCLUDED_PATH_PARTS = (
    "/company/news-room",
    "/company/news-rooms",
    "/recruit",
    "/careers",
    "/contact",
    "/privacy",
    "/terms",
)

DISCOVERY_ONLY_PATHS = {"/sitemap"}

INDUSTRY_PATHS = {
    "/manufacturing",
    "/finance",
    "/cloud",
    "/commerce",
    "/telecom",
    "/service",
}

GENERIC_TITLES = {
    "SK AX",
    "Overview",
    "Highlights",
    "Story",
    "Features",
    "Insights",
    "Services",
    "Industries",
    "Experiences",
}

EXCLUDED_EXTENSIONS = (
    ".css",
    ".js",
    ".json",
    ".xml",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
)

REMOVE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "header",
    "footer",
    "nav",
    "form",
    "button",
    "[role='navigation']",
    "[aria-hidden='true']",
)

BLOCK_TEXTS = {
    "본문 바로가기",
    "자세히 알아보기",
    "문의하기",
    "전체보기",
    "모든 Trend",
    "모든 Video",
    "TOP",
}


class SkaxSiteCrawler(BaseCrawler):
    """SK AX 공식 사이트의 회사/서비스/인사이트 페이지를 구조화해 수집한다."""

    def __init__(
        self,
        *,
        seed_urls: list[str] | None = None,
        max_pages: int = 200,
        use_render: bool = True,
        output_path: Path | None = None,
    ) -> None:
        super().__init__(company=["sk_ax"])
        self.seed_urls = seed_urls or DEFAULT_SEED_URLS
        self.max_pages = max_pages
        self.use_render = use_render
        self.output_path = output_path
        self.playwright = PlaywrightClient()

    async def crawl(self) -> list[RawArticle]:
        urls = await self._discover_urls()
        articles: list[RawArticle] = []
        seen_content_hashes: set[str] = set()

        for url in urls[: self.max_pages]:
            article = await self._crawl_page(url)
            if article:
                content_hash = str(article.extra.get("content_hash", ""))
                if content_hash and content_hash in seen_content_hashes:
                    log.info("SK AX site 중복 본문 제외 | url=%s", url)
                    continue
                seen_content_hashes.add(content_hash)
                articles.append(article)

        if self.output_path:
            self._write_output(articles)

        log.info("SK AX site 크롤 완료 | pages=%d articles=%d", len(urls), len(articles))
        return articles

    async def _discover_urls(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        queue: deque[str] = deque()

        for seed_url in self.seed_urls:
            normalized = normalize_skax_url(seed_url)
            if normalized:
                queue.append(normalized)

        async with httpx.AsyncClient(
            headers=REQUEST_HEADERS, timeout=20, follow_redirects=True
        ) as client:
            while queue and len(ordered) < self.max_pages:
                url = queue.popleft()
                if url in seen:
                    continue

                seen.add(url)
                ordered.append(url)

                html = await self._fetch_html(client, url)
                if not html:
                    continue

                for link in extract_internal_links(html, url):
                    if link not in seen and link not in queue:
                        queue.append(link)

        return ordered

    async def _crawl_page(self, url: str) -> RawArticle | None:
        async with httpx.AsyncClient(
            headers=REQUEST_HEADERS, timeout=20, follow_redirects=True
        ) as client:
            html = await self._fetch_html(client, url)

        if not html:
            return None

        if urlparse(url).path in DISCOVERY_ONLY_PATHS:
            return None

        parsed = parse_skax_page(html, url)
        if not parsed["content"]:
            log.info("SK AX site 본문 없음 | url=%s", url)
            return None

        content_hash = hashlib.sha256(parsed["content"].encode("utf-8")).hexdigest()
        article_id = make_id(url)

        return RawArticle(
            id=article_id,
            source_type="company_site",
            source_name="SK AX Site",
            title=parsed["title"],
            content=parsed["content"],
            url=url,
            published_at=None,
            collected_at=datetime.now().astimezone(),
            publisher="SK AX",
            company=["sk_ax"],
            language="ko",
            content_type="html",
            crawl_status="success",
            extra={
                "page_kind": classify_page_kind(url),
                "sections": parsed["sections"],
                "headings": parsed["headings"],
                "content_hash": content_hash,
                "source_family": "sk_ax_site",
            },
        )

    async def _fetch_html(self, client: httpx.AsyncClient, url: str) -> str:
        if self.use_render:
            html = await self.playwright.fetch_html(url)
            if html:
                return html

        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            log.warning("SK AX page fetch 실패 | url=%s error=%s", url, e)
            return ""

    def _write_output(self, articles: list[RawArticle]) -> None:
        if not self.output_path:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [article.to_common_dict() for article in articles]
        self.output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def normalize_skax_url(url: str, base_url: str = SKAX_HOME_URL) -> str | None:
    absolute = urljoin(base_url, url)
    absolute, _fragment = urldefrag(absolute)
    parsed = urlparse(absolute)

    if parsed.scheme not in {"http", "https"}:
        return None

    if parsed.netloc not in {"skax.co.kr", "www.skax.co.kr"}:
        return None

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    lower_path = path.lower()
    if lower_path.endswith(EXCLUDED_EXTENSIONS):
        return None

    if any(part in lower_path for part in EXCLUDED_PATH_PARTS):
        return None

    return urlunparse(("https", "www.skax.co.kr", path, "", parsed.query, ""))


def extract_internal_links(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        normalized = normalize_skax_url(str(href), page_url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)

    return links


def parse_skax_page(html: str, url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    for selector in REMOVE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    root = find_content_root(soup)
    title = extract_title(soup, root, url)
    sections = extract_sections(root)
    content = "\n\n".join(section["text"] for section in sections)
    headings = [section["heading"] for section in sections if section["heading"]]

    if not content:
        content = clean_text(root.get_text("\n", strip=True))

    title = choose_page_title(title, headings, url)

    return {
        "title": title,
        "content": content,
        "sections": sections,
        "headings": headings,
    }


def find_content_root(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", "#contents", "#content", ".contents", ".content"):
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            return node
    body = soup.find("body")
    return body if isinstance(body, Tag) else soup


def extract_title(soup: BeautifulSoup, root: Tag, url: str) -> str:
    candidates: list[str] = []

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title:
        candidates.append(clean_text(str(og_title.get("content", ""))))

    meta_title = soup.find("meta", attrs={"name": "title"})
    if meta_title:
        candidates.append(clean_text(str(meta_title.get("content", ""))))

    if soup.title and soup.title.string:
        candidates.append(clean_text(soup.title.string))

    for selector in ("h1", "h2"):
        node = root.select_one(selector)
        if node:
            candidates.append(clean_text(node.get_text(" ", strip=True)))

    for selector in ("h1", "h2"):
        node = soup.select_one(selector)
        if node:
            candidates.append(clean_text(node.get_text(" ", strip=True)))

    normalized_candidates = [normalize_title(candidate) for candidate in candidates]
    for title in normalized_candidates:
        if title and title not in GENERIC_TITLES:
            return title

    for title in normalized_candidates:
        if title:
            return title

    return urlparse(url).path.strip("/") or "SK AX"


def choose_page_title(raw_title: str, headings: list[str], url: str) -> str:
    path = urlparse(url).path.rstrip("/") or "/"
    if path == "/":
        return "SK AX"

    title = normalize_title(raw_title)
    if title and title not in GENERIC_TITLES:
        return title

    path_title = title_from_path(url)
    if path_title:
        return path_title

    for heading in headings:
        normalized = normalize_title(heading)
        if normalized and normalized not in GENERIC_TITLES:
            return normalized

    return title or "SK AX"


def normalize_title(title: str) -> str:
    title = clean_text(title)
    title = re.sub(r"\s*[-│|]\s*SK AX\s*$", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def title_from_path(url: str) -> str:
    path = urlparse(url).path.rstrip("/") or "/"
    mapping = {
        "/": "SK AX",
        "/axgenticwire": "AXgenticWire",
        "/company/about": "회사소개",
        "/ax-services/aicc": "AICC",
        "/ax-services/new-paradigm-operation": "AIOps Platform",
        "/case-study/usecase": "Experiences",
        "/manufacturing": "제조",
        "/finance": "금융",
        "/cloud": "Cloud",
        "/commerce": "유통/물류",
        "/telecom": "통신",
        "/service": "서비스",
        "/insight/trends": "Trends",
        "/insight/newsletter": "Newsletter",
        "/insight/videos": "Videos",
        "/insight/events": "Events",
        "/insight/resources": "자료실",
    }
    return mapping.get(path, "")


def extract_sections(root: Tag) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_heading = ""
    current_parts: list[str] = []

    for node in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd"], recursive=True):
        text = clean_text(node.get_text(" ", strip=True))
        if should_skip_text(text):
            continue

        if node.name in {"h1", "h2", "h3", "h4"}:
            flush_section(sections, current_heading, current_parts)
            current_heading = text
            current_parts = []
            continue

        current_parts.append(text)

    flush_section(sections, current_heading, current_parts)

    if sections:
        return sections

    fallback = clean_text(root.get_text("\n", strip=True))
    return [{"heading": "", "text": fallback}] if fallback else []


def flush_section(
    sections: list[dict[str, str]],
    heading: str,
    parts: list[str],
) -> None:
    deduped = dedupe_keep_order(parts)
    text = clean_text("\n".join(deduped))
    if not text:
        return
    normalized_heading = clean_text(heading)
    section_text = f"{normalized_heading}\n{text}".strip() if normalized_heading else text
    if any(existing["text"] == section_text for existing in sections):
        return
    sections.append({"heading": normalized_heading, "text": section_text})


def dedupe_keep_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def should_skip_text(text: str) -> bool:
    if not text:
        return True
    if text in BLOCK_TEXTS:
        return True
    if len(text) <= 1:
        return True
    return False


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def classify_page_kind(url: str) -> str:
    path = urlparse(url).path.rstrip("/") or "/"
    if path == "/":
        return "home"
    if path.startswith("/axgenticwire"):
        return "brand_campaign"
    if path in INDUSTRY_PATHS:
        return "industry"
    if path.startswith("/case-study"):
        return "experience"
    if path.startswith("/services"):
        return "service"
    if path.startswith("/industries"):
        return "industry"
    if path.startswith("/experiences"):
        return "experience"
    if path.startswith("/insights"):
        return "insight"
    if path.startswith("/company/about"):
        return "company_about"
    if path.startswith("/company"):
        return "company"
    if path.startswith("/ax-services"):
        return "service"
    if path.startswith("/insight/trend"):
        return "insight_trend"
    if path.startswith("/insight"):
        return "insight"
    return "site_page"


def make_id(url: str) -> str:
    raw = f"company_site|SK AX Site|{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SK AX 공식 사이트 페이지 크롤러")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument(
        "--no-render", action="store_true", help="Playwright 렌더링 없이 httpx만 사용"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_DIR / "skax_site.json"),
        help="결과 JSON 저장 경로",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    crawler = SkaxSiteCrawler(
        max_pages=args.max_pages,
        use_render=not args.no_render,
        output_path=Path(args.output),
    )
    articles = await crawler.crawl()
    print(f"수집 완료: {len(articles)}건")


if __name__ == "__main__":
    asyncio.run(main())
