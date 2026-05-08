"""글로벌 기업 공식 뉴스룸 크롤러.

Google RSS가 아닌 각 회사의 공식 발표/뉴스룸에서 원문 본문을 직접 수집한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base import RawArticle  # noqa: E402
from src.crawler.parsers.article_content import (  # noqa: E402
    extract_body_text,
    extract_image_urls,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT = 20
MIN_CONTENT_LENGTH = 120
MAX_CONTENT_LENGTH = 20000
OFFICIAL_MAX_AGE_DAYS = 30
GLOBAL_BLOCKED_PATH_KEYWORDS = (
    "/_gallery/",
    "/wp-content/",
    "/assets/",
    "/static/",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".pdf",
)


@dataclass(frozen=True)
class NewsroomConfig:
    company: str
    source_name: str
    publisher: str
    sources: tuple["OfficialSource", ...]
    language: str = "en"
    country: str = "US"


@dataclass(frozen=True)
class OfficialSource:
    name: str
    url: str
    allowed_domains: tuple[str, ...]
    article_path_patterns: tuple[str, ...]
    blocked_path_keywords: tuple[str, ...] = ()
    latest_section_heading: str | None = None
    detail_date_probe_limit: int = 0


NEWSROOM_CONFIGS: dict[str, NewsroomConfig] = {
    "nvidia": NewsroomConfig(
        company="nvidia",
        source_name="nvidia_official",
        publisher="NVIDIA",
        sources=(
            OfficialSource(
                name="nvidia_blog_recent_news",
                url="https://blogs.nvidia.com/recent-news/",
                allowed_domains=("blogs.nvidia.com",),
                article_path_patterns=(r"^/blog/[^/]+$",),
            ),
            OfficialSource(
                name="nvidia_developer_recent_posts",
                url="https://developer.nvidia.com/blog/recent-posts/",
                allowed_domains=("developer.nvidia.com",),
                article_path_patterns=(r"^/blog/[^/]+$",),
            ),
        ),
    ),
    "microsoft": NewsroomConfig(
        company="microsoft",
        source_name="microsoft_official",
        publisher="Microsoft",
        sources=(
            OfficialSource(
                name="microsoft_azure_blog",
                url="https://azure.microsoft.com/en-us/blog/",
                allowed_domains=("azure.microsoft.com", "blogs.microsoft.com"),
                article_path_patterns=(
                    r"^/en-us/blog/[^/]+$",
                    r"^/blog/\d{4}/\d{2}/\d{2}/[^/]+$",
                ),
                blocked_path_keywords=(
                    "/content-type/",
                    "/category/",
                    "/product/",
                    "/tag/",
                    "/feed/",
                    "/wp-",
                ),
            ),
            OfficialSource(
                name="microsoft_source_ai",
                url="https://news.microsoft.com/source/topics/ai/",
                allowed_domains=("news.microsoft.com",),
                article_path_patterns=(
                    r"^/source/features/",
                    r"^/source/[a-z-]+/features/",
                    r"^/source/articles/",
                    r"^/signal/articles/",
                    r"^/signalmagazine/issue/",
                ),
                blocked_path_keywords=("/topics/", "/category/", "/tag/", "/feed/"),
                detail_date_probe_limit=20,
            ),
        ),
    ),
    "google": NewsroomConfig(
        company="google",
        source_name="google_official",
        publisher="Google",
        sources=(
            OfficialSource(
                name="google_blog",
                url="https://blog.google/",
                allowed_domains=("blog.google",),
                article_path_patterns=(
                    r"^/innovation-and-ai/[^/]+/[^/]+/[^/]+$",
                    r"^/company-news/[^/]+/[^/]+/[^/]+$",
                    r"^/products-and-platforms/[^/]+/[^/]+/[^/]+$",
                ),
                latest_section_heading="More news",
            ),
            OfficialSource(
                name="google_cloud_blog",
                url="https://cloud.google.com/blog?hl=en",
                allowed_domains=("cloud.google.com",),
                article_path_patterns=(
                    r"^/blog/(products|topics|transform|resources)/[^/]+/[^/]+",
                ),
                detail_date_probe_limit=50,
            ),
        ),
    ),
    "amazon": NewsroomConfig(
        company="amazon",
        source_name="amazon_official",
        publisher="AWS",
        sources=(
            OfficialSource(
                name="aws_news_blog",
                url="https://aws.amazon.com/blogs/aws/",
                allowed_domains=("aws.amazon.com",),
                article_path_patterns=(r"^/blogs/aws/[^/]+$",),
                blocked_path_keywords=(
                    "/author/",
                    "/category/",
                    "/tag/",
                    "/page/",
                    "/comments/",
                ),
            ),
        ),
    ),
    "apple": NewsroomConfig(
        company="apple",
        source_name="apple_newsroom",
        publisher="Apple",
        sources=(
            OfficialSource(
                name="apple_newsroom",
                url="https://www.apple.com/newsroom/archive/",
                allowed_domains=("www.apple.com", "apple.com"),
                article_path_patterns=(r"^/newsroom/\d{4}/\d{2}/[^/]+$",),
                blocked_path_keywords=("/apple-services/", "/apple-stories/"),
                detail_date_probe_limit=20,
            ),
        ),
    ),
    "meta": NewsroomConfig(
        company="meta",
        source_name="meta_official",
        publisher="Meta",
        sources=(
            OfficialSource(
                name="meta_technologies_news",
                url="https://about.fb.com/news/category/technologies/meta/",
                allowed_domains=("about.fb.com",),
                article_path_patterns=(r"^/news/\d{4}/\d{2}/[^/]+$",),
                blocked_path_keywords=("/category/", "/tag/"),
            ),
            OfficialSource(
                name="meta_ai_blog",
                url="https://ai.meta.com/blog/",
                allowed_domains=("ai.meta.com",),
                article_path_patterns=(r"^/blog/[^/]+$",),
            ),
        ),
    ),
}

NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    ".cookie",
    ".modal",
    ".popup",
    ".share",
    ".social",
    ".newsletter",
    ".msx-summary",
    ".msx-summary-more",
    ".table-of-contents-block",
]


def default_output_path() -> Path:
    output_dir = Path(__file__).resolve().parent.parent / "crawler_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "global_newsroom_crawler.json"


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path

    if path != "/":
        path = path.rstrip("/")

    return urlunparse(parsed._replace(path=path, query="", fragment=""))


def same_allowed_domain(url: str, source: OfficialSource) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in source.allowed_domains


def is_blocked_url(url: str, source: OfficialSource) -> bool:
    path = urlparse(url).path.lower()
    blocked_keywords = (*GLOBAL_BLOCKED_PATH_KEYWORDS, *source.blocked_path_keywords)
    return any(keyword.lower() in path for keyword in blocked_keywords)


def is_article_url(url: str, source: OfficialSource) -> bool:
    if not same_allowed_domain(url, source):
        return False

    if is_blocked_url(url, source):
        return False

    path = urlparse(url).path

    if path in {"", "/"}:
        return False

    return any(re.search(pattern, path) for pattern in source.article_path_patterns)


def is_crawlable_url(url: str, source: OfficialSource) -> bool:
    return same_allowed_domain(url, source) and not is_blocked_url(url, source)


def fetch_html(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        response = requests.get(url, headers=HEADERS, timeout=45)

    if response.status_code == 400:
        fallback_headers = {"User-Agent": "Mozilla/5.0"}

        try:
            response = requests.get(url, headers=fallback_headers, timeout=REQUEST_TIMEOUT)
        except requests.Timeout:
            response = requests.get(url, headers=fallback_headers, timeout=45)

    response.raise_for_status()
    return response.content.decode(response.encoding or "utf-8", errors="replace")


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def clean_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_title(soup: BeautifulSoup, fallback_url: str) -> str:
    candidates = [
        soup.select_one("meta[property='og:title']"),
        soup.select_one("meta[name='twitter:title']"),
        soup.select_one("meta[name='title']"),
        soup.select_one("h1"),
        soup.title,
    ]

    for node in candidates:
        if not node:
            continue

        if node.name == "meta":
            title = clean_text(node.get("content"))
        else:
            title = clean_text(node.get_text(" ", strip=True))

        title = re.sub(
            r"\s*[|-]\s*(NVIDIA|Microsoft|Google|AWS|Amazon Web Services|Apple|Meta|Source).*$",
            "",
            title,
        )

        if title:
            return title

    return Path(urlparse(fallback_url).path).name.replace("-", " ")


def remove_noise(soup: BeautifulSoup) -> None:
    for node in soup.select(", ".join(NOISE_SELECTORS)):
        node.decompose()


def extract_content(html: str, url: str | None = None) -> str | None:
    full_body = extract_full_content(html, url)

    if full_body:
        return full_body

    body = extract_body_text(html)

    if len(body) >= MIN_CONTENT_LENGTH:
        return body

    soup = make_soup(html)
    remove_noise(soup)

    best_text = ""

    for selector in ("article", "main", "[role='main']", ".content", ".post", "body"):
        for node in soup.select(selector):
            text = clean_text(node.get_text(" ", strip=True))

            if len(text) > len(best_text):
                best_text = text

    return best_text[:MAX_CONTENT_LENGTH] if len(best_text) >= MIN_CONTENT_LENGTH else None


def extract_full_content(html: str, url: str | None = None) -> str | None:
    soup = make_soup(html)
    remove_noise(soup)

    if url and is_google_blog_url(url):
        text = extract_google_blog_content(soup)

        if text:
            return text

    if url and is_google_cloud_blog_url(url):
        text = extract_google_cloud_content(soup)

        if text:
            return text

    if url and is_microsoft_source_url(url):
        text = extract_microsoft_source_content(soup, url)

        if text:
            return text

    selectors = (
        ".entry-content",
        ".post-content",
        ".article-content",
        ".article_body",
        ".article-body",
        "article [class*='content']",
        "article",
        "main",
    )

    for selector in selectors:
        best_text = ""

        for node in soup.select(selector):
            lines = [
                clean_text(line)
                for line in node.get_text("\n", strip=True).splitlines()
                if len(clean_text(line)) >= 2
            ]
            text = "\n".join(lines)
            text = trim_article_tail(text)

            if len(text) > len(best_text):
                best_text = text

        if len(best_text) >= MIN_CONTENT_LENGTH:
            return best_text[:MAX_CONTENT_LENGTH]

    return None


def is_google_blog_url(url: str) -> bool:
    return (urlparse(url).hostname or "") == "blog.google"


def is_google_cloud_blog_url(url: str) -> bool:
    parsed = urlparse(url)
    return (parsed.hostname or "") == "cloud.google.com" and parsed.path.startswith("/blog/")


def is_microsoft_source_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path

    return host == "news.microsoft.com" and (
        path.startswith("/source/")
        or path.startswith("/signal/")
        or path.startswith("/signalmagazine/")
    )


def normalize_for_compare(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def extract_google_blog_content(soup: BeautifulSoup) -> str | None:
    node = soup.select_one(".article-container__content, .uni-blog-article-container")

    if not node:
        return None

    lines = lines_from_node(node)
    lines = remove_google_blog_intro(lines)

    if not lines:
        return None

    text = trim_article_tail("\n".join(lines))
    return text[:MAX_CONTENT_LENGTH] if len(text) >= MIN_CONTENT_LENGTH else None


def remove_google_blog_intro(lines: list[str]) -> list[str]:
    intro_patterns = (
        r"your browser does not support the audio element",
        r"listen to article",
        r"this content is generated by google ai",
        r"generative ai is experimental",
        r"\[\[duration\]\]\s+minutes",
        r"voice speed",
    )
    cleaned = list(lines)

    while cleaned:
        first = cleaned[0].strip().lower()

        if any(re.search(pattern, first) for pattern in intro_patterns):
            cleaned.pop(0)
            continue

        break

    return cleaned


def extract_google_cloud_content(soup: BeautifulSoup) -> str | None:
    article = soup.select_one("article")

    if not article:
        return None

    collected: list[str] = []
    started = False

    for section in article.select("section"):
        text = clean_text(section.get_text(" ", strip=True))

        if started and re.match(r"^(posted in|related articles)\b", text, re.IGNORECASE):
            break

        if not section.select("p") or len(text) < MIN_CONTENT_LENGTH:
            continue

        lines = lines_from_node(section)
        section_text = trim_article_tail("\n".join(lines))

        if len(section_text) < MIN_CONTENT_LENGTH:
            continue

        collected.append(section_text)
        started = True

    text = "\n".join(collected).strip()
    return text[:MAX_CONTENT_LENGTH] if len(text) >= MIN_CONTENT_LENGTH else None

    return None


def extract_microsoft_source_content(soup: BeautifulSoup, url: str) -> str | None:
    main = soup.select_one("main")

    if not main:
        return None

    lines = lines_from_node(main)
    lines = remove_microsoft_source_header(lines, extract_title(soup, url))

    if not lines:
        return None

    text = trim_after_tags("\n".join(lines))
    return text[:MAX_CONTENT_LENGTH] if len(text) >= MIN_CONTENT_LENGTH else None


def lines_from_node(node) -> list[str]:
    return [
        clean_text(line)
        for line in node.get_text("\n", strip=True).splitlines()
        if len(clean_text(line)) >= 2
    ]


def remove_microsoft_source_header(lines: list[str], title: str | None) -> list[str]:
    cleaned = list(lines)
    normalized_title = normalize_for_compare(title or "")

    while cleaned:
        first = cleaned[0]
        normalized_first = normalize_for_compare(first)

        if normalized_title and normalized_first == normalized_title:
            cleaned.pop(0)
            continue

        if re.match(r"^By\s+\S+", first, re.IGNORECASE):
            cleaned.pop(0)
            continue

        if first.lower() in {"share", "listen", "read next"}:
            cleaned.pop(0)
            continue

        if re.match(r"^(published|updated|posted)\b", first, re.IGNORECASE):
            cleaned.pop(0)
            continue

        break

    return cleaned


def trim_after_tags(text: str) -> str:
    index = text.find("\nTags:\n")
    return text[:index].strip() if index != -1 else text.strip()


def trim_article_tail(text: str) -> str:
    stop_markers = (
        "\nCategories:\n",
        "\nRelated News\n",
        "\nRelated Articles\n",
        "\nRecommended\n",
        "\nTags:\n",
        "\nClick here to load media\n",
        "\nRead more about ",
        "\nThis story was published on ",
    )

    trimmed = text

    for marker in stop_markers:
        index = trimmed.find(marker)

        if index != -1:
            trimmed = trimmed[:index]

    return trimmed.strip()


def extract_published_at(soup: BeautifulSoup) -> datetime | None:
    candidates = [
        ("meta[property='article:published_time']", "content"),
        ("meta[name='published_time']", "content"),
        ("meta[name='track-metadata-page_first_published']", "content"),
        ("meta[name='date']", "content"),
        ("meta[name='pubdate']", "content"),
        ("time[datetime]", "datetime"),
    ]

    for selector, attr in candidates:
        node = soup.select_one(selector)

        if not node:
            continue

        value = clean_text(node.get(attr))

        if not value:
            continue

        parsed = parse_datetime(value)

        if parsed:
            return parsed

        if node.name == "time":
            parsed = parse_date_from_text(clean_text(node.get_text(" ", strip=True)))

            if parsed:
                return parsed

    text = clean_text(soup.get_text(" ", strip=True))
    return parse_published_date_from_text(text)

    return None


def parse_datetime(value: str) -> datetime | None:
    value = value.strip()

    if value.isdigit():
        try:
            return datetime.fromtimestamp(int(value))
        except (OverflowError, OSError, ValueError):
            pass

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass

    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class LinkCandidate:
    url: str
    text: str
    context: str
    published_at: datetime | None
    order: int


def parse_date_from_text(text: str) -> datetime | None:
    patterns = (
        r"\b([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})\b",
        r"\b([A-Za-z]+)\.?\s+(\d{1,2})\b",
        r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b",
    )

    for pattern in patterns:
        match = re.search(pattern, text)

        if not match:
            continue

        groups = match.groups()

        if groups[0].isdigit():
            day = int(groups[0])
            month = MONTHS.get(groups[1].lower().rstrip("."))
            year = int(groups[2])
        else:
            month = MONTHS.get(groups[0].lower().rstrip("."))
            day = int(groups[1])
            year = int(groups[2]) if len(groups) == 3 else datetime.now().year

        if not month:
            continue

        try:
            return datetime(year, month, day)
        except ValueError:
            return None

    return None


def parse_published_date_from_text(text: str) -> datetime | None:
    patterns = (
        r"(?:published|posted|updated)\s+on\s+([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})",
        r"(?:published|posted|updated)\s+([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})",
        r"\b([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})\b",
    )

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if not match:
            continue

        parsed = parse_date_from_text(match.group(1))

        if parsed:
            return parsed

    return None


def parse_date_from_url(url: str) -> datetime | None:
    path = urlparse(url).path
    match = re.search(r"/(20\d{2})/(\d{2})/(\d{2})/", path)

    if match:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    match = re.search(r"/(20\d{2})/(\d{2})/", path)

    if match:
        return datetime(int(match.group(1)), int(match.group(2)), 1)

    return None


def extract_links(soup: BeautifulSoup, page_url: str, source: OfficialSource) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()

    for node in soup.find_all("a", href=True):
        href = str(node.get("href") or "").strip()

        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue

        url = normalize_url(urljoin(page_url, href))

        if url in seen or not is_crawlable_url(url, source):
            continue

        seen.add(url)
        links.append(url)

    return links


def first_article_href(node, page_url: str, source: OfficialSource) -> str | None:
    if node.name == "a" and node.get("href"):
        url = normalize_url(urljoin(page_url, str(node.get("href") or "")))

        if is_article_url(url, source):
            return url

    for link in node.find_all("a", href=True):
        url = normalize_url(urljoin(page_url, str(link.get("href") or "")))

        if is_article_url(url, source):
            return url

    return None


def extract_google_more_news(
    soup: BeautifulSoup,
    page_url: str,
    source: OfficialSource,
) -> list[LinkCandidate]:
    if not source.latest_section_heading:
        return []

    for heading in soup.find_all(["h2", "h3"]):
        heading_text = clean_text(heading.get_text(" ", strip=True)).lower()

        if heading_text != source.latest_section_heading.lower():
            continue

        section = heading.find_parent("section") or heading.find_parent("div")

        if not section:
            continue

        candidates: list[LinkCandidate] = []

        for order, card in enumerate(section.select("li, article, div")):
            url = first_article_href(card, page_url, source)

            if not url:
                continue

            context = clean_text(card.get_text(" ", strip=True))
            heading_node = card.find(["h2", "h3", "h4"])
            text = clean_text(heading_node.get_text(" ", strip=True)) if heading_node else context
            candidates.append(
                LinkCandidate(
                    url=url,
                    text=text,
                    context=context,
                    published_at=parse_date_from_text(context) or parse_date_from_url(url),
                    order=order,
                )
            )

        return unique_candidates(candidates)

    return []


def extract_card_candidates(
    soup: BeautifulSoup,
    page_url: str,
    source: OfficialSource,
) -> list[LinkCandidate]:
    candidates: list[LinkCandidate] = []

    if source.name == "apple_newsroom":
        return extract_apple_archive_candidates(soup, page_url, source)

    nodes = [
        *soup.find_all("article"),
        *soup.select(".wp-block-post"),
        *soup.select("li.nup-card"),
        *soup.select(".mA0uBe"),
        *soup.select(".nRhiJb-kR0ZEf-OWXEXe-GV1x9e-qWD73c"),
        *soup.select(".nRhiJb-kR0ZEf-OWXEXe-GV1x9e-II5mzb"),
        *soup.select(".views-row"),
        *soup.select(".post"),
        *soup.select(".card"),
        *soup.select("li"),
    ]

    for order, node in enumerate(nodes):
        url = first_article_href(node, page_url, source)

        if not url:
            continue

        context = clean_text(node.get_text(" ", strip=True))

        if len(context) < 12:
            continue

        heading = node.find(["h1", "h2", "h3", "h4"])
        text = clean_text(heading.get_text(" ", strip=True)) if heading else context
        candidates.append(
            LinkCandidate(
                url=url,
                text=text,
                context=context,
                published_at=parse_date_from_text(context) or parse_date_from_url(url),
                order=order,
            )
        )

    if candidates:
        return unique_candidates(candidates)

    return extract_anchor_candidates(soup, page_url, source)


def extract_apple_archive_candidates(
    soup: BeautifulSoup,
    page_url: str,
    source: OfficialSource,
) -> list[LinkCandidate]:
    candidates: list[LinkCandidate] = []

    for order, node in enumerate(soup.select("a.result__item")):
        url = first_article_href(node, page_url, source)

        if not url:
            continue

        context = clean_text(node.get_text(" ", strip=True))
        heading = node.find(["h2", "h3", "h4"])
        text = clean_text(heading.get_text(" ", strip=True)) if heading else context
        candidates.append(
            LinkCandidate(
                url=url,
                text=text,
                context=context,
                published_at=parse_date_from_text(context) or parse_date_from_url(url),
                order=order,
            )
        )

    return unique_candidates(candidates)


def extract_anchor_candidates(
    soup: BeautifulSoup,
    page_url: str,
    source: OfficialSource,
) -> list[LinkCandidate]:
    candidates: list[LinkCandidate] = []

    for order, link in enumerate(soup.find_all("a", href=True)):
        url = normalize_url(urljoin(page_url, str(link.get("href") or "")))

        if not is_article_url(url, source):
            continue

        parent = link.find_parent(["article", "li", "div", "section"]) or link
        context = clean_text(parent.get_text(" ", strip=True))
        text = clean_text(link.get_text(" ", strip=True)) or context
        candidates.append(
            LinkCandidate(
                url=url,
                text=text,
                context=context,
                published_at=parse_date_from_text(context) or parse_date_from_url(url),
                order=order,
            )
        )

    return unique_candidates(candidates)


def unique_candidates(candidates: list[LinkCandidate]) -> list[LinkCandidate]:
    unique: list[LinkCandidate] = []
    seen: set[str] = set()

    for candidate in candidates:
        if candidate.url in seen:
            continue

        seen.add(candidate.url)
        unique.append(candidate)

    return unique


def candidate_with_detail_date(candidate: LinkCandidate) -> LinkCandidate:
    if candidate.published_at:
        return candidate

    try:
        html = fetch_html(candidate.url)
        soup = make_soup(html)
        published_at = extract_published_at(soup)
    except Exception:
        published_at = None

    if not published_at:
        return candidate

    return LinkCandidate(
        url=candidate.url,
        text=candidate.text,
        context=candidate.context,
        published_at=published_at,
        order=candidate.order,
    )


def enrich_detail_dates(
    candidates: list[LinkCandidate],
    source: OfficialSource,
) -> list[LinkCandidate]:
    if source.detail_date_probe_limit <= 0:
        return candidates

    enriched: list[LinkCandidate] = []

    for index, candidate in enumerate(candidates):
        if index < source.detail_date_probe_limit:
            enriched.append(candidate_with_detail_date(candidate))
        else:
            enriched.append(candidate)

    return enriched


def latest_candidates(
    soup: BeautifulSoup,
    page_url: str,
    source: OfficialSource,
) -> list[LinkCandidate]:
    candidates = extract_google_more_news(soup, page_url, source) or extract_card_candidates(
        soup,
        page_url,
        source,
    )
    candidates = enrich_detail_dates(candidates, source)

    if any(candidate.published_at for candidate in candidates):
        candidates.sort(
            key=lambda candidate: (
                candidate.published_at is not None,
                datetime_sort_value(candidate.published_at),
                -candidate.order,
            ),
            reverse=True,
        )

    return candidates


def datetime_sort_value(value: datetime | None) -> float:
    if not value:
        return 0.0

    return value.timestamp()


def is_recent_official_article(published_at: datetime | None) -> bool:
    if published_at is None:
        return False

    now = datetime.now(published_at.tzinfo) if published_at.tzinfo else datetime.now()
    age_days = (now - published_at).days
    return age_days <= OFFICIAL_MAX_AGE_DAYS


class _SyncGlobalNewsroomCrawler:
    def __init__(
        self,
        config: NewsroomConfig,
        max_pages: int | None = None,
        max_depth: int = 1,
        debug_links: bool = False,
    ):
        self.config = config
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.debug_links = debug_links

    def crawl(self) -> list[RawArticle]:
        articles: list[RawArticle] = []

        for source in self.config.sources:
            source_articles = self.crawl_source(source)
            articles.extend(source_articles)

        return articles

    def crawl_source(self, source: OfficialSource) -> list[RawArticle]:
        articles: list[RawArticle] = []

        try:
            listing_html = fetch_html(source.url)
            listing_soup = make_soup(listing_html)
            candidates = latest_candidates(listing_soup, source.url, source)
        except Exception as e:
            log.warning(
                "뉴스룸 목록 수집 실패 | company=%s source=%s url=%s error=%s",
                self.config.company,
                source.name,
                source.url,
                e,
            )
            return articles

        if self.debug_links:
            for candidate in candidates[:10]:
                log.info(
                    "후보 기사 링크 | company=%s source=%s date=%s url=%s title=%s",
                    self.config.company,
                    source.name,
                    candidate.published_at,
                    candidate.url,
                    candidate.text,
                )

        for candidate in candidates:
            if self.max_pages is not None and len(articles) >= self.max_pages:
                break

            try:
                html = fetch_html(candidate.url)
                soup = make_soup(html)
                article = self.article_from_page(
                    candidate.url,
                    html,
                    soup,
                    source,
                    candidate,
                )

                if article:
                    if not is_recent_official_article(article.published_at):
                        log.info(
                            "오래된 글로벌 공식 뉴스 제외 | company=%s date=%s title=%s",
                            self.config.company,
                            article.published_at,
                            article.title,
                        )
                        continue
                    articles.append(article)
                    log.info(
                        "뉴스룸 수집 완료 | company=%s source=%s title=%s",
                        self.config.company,
                        source.name,
                        article.title,
                    )

            except Exception as e:
                log.warning(
                    "뉴스룸 수집 실패 | company=%s source=%s url=%s error=%s",
                    self.config.company,
                    source.name,
                    candidate.url,
                    e,
                )

        return articles

    def article_from_page(
        self,
        url: str,
        html: str,
        soup: BeautifulSoup,
        source: OfficialSource,
        candidate: LinkCandidate,
    ) -> RawArticle | None:
        title = extract_title(soup, url)
        content = extract_content(html, url)

        if not content:
            log.warning("본문 부족으로 제외 | company=%s url=%s", self.config.company, url)
            return None

        return RawArticle(
            url=url,
            title=title,
            content=content,
            published_at=extract_published_at(soup) or candidate.published_at,
            source_name=self.config.source_name,
            source_type="official",
            publisher=self.config.publisher,
            company=[self.config.company],
            language=self.config.language,
            content_type="html",
            crawl_status="success",
            extra={
                "image_urls": extract_image_urls(html, url),
                "body_fetch_status": "success",
                "source_url": source.url,
                "source_key": source.name,
            },
        )


class GlobalNewsroomCrawler:
    """공식 글로벌 뉴스룸을 공통 async crawler 인터페이스로 실행한다."""

    def __init__(
        self,
        company: str | None = None,
        max_pages: int | None = None,
        max_depth: int = 1,
        debug_links: bool = False,
    ) -> None:
        self.company = company
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.debug_links = debug_links

    async def crawl(self) -> list[RawArticle]:
        return await asyncio.to_thread(self._crawl_sync)

    def _crawl_sync(self) -> list[RawArticle]:
        company_ids = [self.company] if self.company else list(NEWSROOM_CONFIGS)
        articles: list[RawArticle] = []

        for company_id in company_ids:
            config = NEWSROOM_CONFIGS[company_id]
            crawler = _SyncGlobalNewsroomCrawler(
                config=config,
                max_pages=self.max_pages,
                max_depth=self.max_depth,
                debug_links=self.debug_links,
            )
            articles.extend(crawler.crawl())

        return articles


def save_json(articles: list[RawArticle], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [article.to_common_dict() for article in articles]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="글로벌 기업 공식 뉴스룸 크롤러")
    parser.add_argument(
        "--company",
        action="append",
        choices=sorted(NEWSROOM_CONFIGS),
        default=None,
        help="수집 대상 회사. 여러 번 지정 가능. 생략하면 전체.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="소스별 최대 저장 기사 수. 생략하면 제한하지 않는다.",
    )
    parser.add_argument("--max-depth", type=int, default=1, help="seed 기준 최대 링크 탐색 깊이")
    parser.add_argument("--output", default=None, help="저장 경로")
    parser.add_argument("--debug-links", action="store_true", help="후보 기사 링크 로그 출력")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    company_ids = args.company or list(NEWSROOM_CONFIGS)
    articles: list[RawArticle] = []

    for company_id in company_ids:
        crawler = _SyncGlobalNewsroomCrawler(
            config=NEWSROOM_CONFIGS[company_id],
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            debug_links=args.debug_links,
        )
        articles.extend(crawler.crawl())

    output_path = Path(args.output) if args.output else default_output_path()
    save_json(articles, output_path)

    print(f"수집 결과: {len(articles)}건")
    print(f"JSON 저장 위치: {output_path}")


if __name__ == "__main__":
    main()
