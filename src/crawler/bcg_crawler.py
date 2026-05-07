"""BCG 글로벌 IT/AI 산업동향 크롤러.

사용 예시:
.venv/bin/python src/crawler/bcg_crawler.py --days 7

기능:
1. BCG AI, Publications, Digital/Technology/Data 페이지 수집
2. 상세 페이지 URL 수집
3. 상세 페이지에서 title, published_at, content 추출
4. 최근 N일 이내 글 저장
5. published_at을 못 찾은 글은 일단 저장하되 extra에 표시
6. 키워드 필터링은 하지 않음
7. 본문 이미지는 로컬 저장하지 않고 URL만 저장
8. 목록/상세 실패는 JSON에 저장하지 않고 로그만 남김
9. RawArticle 기반 공통 JSON 스키마로 저장
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base import RawArticle  # noqa: E402
from src.crawler.base_crawler import BaseCrawler  # noqa: E402
from src.crawler.parsers.article_content import extract_image_urls  # noqa: E402
from playwright.async_api import TimeoutError as PlaywrightTimeoutError  # noqa: E402

from src.crawler.playwright_client import PlaywrightClient  # noqa: E402

log = logging.getLogger(__name__)

BCG_SOURCES = {
    "bcg_publications": "https://www.bcg.com/publications",
    "bcg_ai": "https://www.bcg.com/capabilities/artificial-intelligence/insights",
    "bcg_digital_technology_data": "https://www.bcg.com/capabilities/digital-technology-data/insights",
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    "Connection": "keep-alive",
}


def normalize_text(text: str) -> str:
    """공백과 줄바꿈을 정리한다."""
    return re.sub(r"\s+", " ", text or "").strip()


def make_id(source_type: str, source_name: str, title: str, url: str) -> str:
    """같은 글이 다시 수집되어도 같은 id가 나오도록 고정 id를 만든다."""
    raw = f"{source_type}|{source_name}|{title}|{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ensure_aware_utc(dt: datetime | None) -> datetime | None:
    """timezone 없는 datetime이면 UTC timezone을 붙이고, 있으면 UTC로 변환한다."""
    if dt is None:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def default_output_path() -> Path:
    """기본 JSON 저장 경로.

    src/crawler/bcg_crawler.py
    → src/crawler/crawler_results/bcg_crawler.json
    """
    return (
        Path(__file__).resolve().parent
        / "crawler_results"
        / f"{Path(__file__).stem}.json"
    )


def load_local_env() -> None:
    """로컬 실행 시 .env.local을 로드한다."""
    env_path = PROJECT_ROOT / ".env.local"
    if env_path.exists():
        load_dotenv(env_path, override=True)


async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)

    if response.status_code in {403, 429}:
        log.warning(
            "BCG httpx 접근 차단, Playwright fallback 시도 | url=%s status=%d",
            url,
            response.status_code,
        )
        html = await fetch_bcg_html_by_playwright(url)
        if html:
            return html

    response.raise_for_status()
    return response.text


async def fetch_bcg_html_by_playwright(url: str) -> str:
    """BCG 전용 Playwright fallback.

    BCG 페이지는 장시간 열려 있는 네트워크 요청 때문에 networkidle 대기가 자주
    타임아웃된다. DOM 로드 후 짧게 기다린 HTML만으로도 목록/상세 파싱에 충분한
    경우가 많아서 BCG는 더 느슨한 대기 조건을 사용한다.
    """
    try:
        async with PlaywrightClient().new_page() as page:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            except PlaywrightTimeoutError:
                log.warning("BCG Playwright domcontentloaded timeout, 현재 HTML 사용 | url=%s", url)

            try:
                await page.wait_for_load_state("load", timeout=10_000)
            except PlaywrightTimeoutError:
                pass

            await page.wait_for_timeout(2_000)
            html = await page.content()

            if is_likely_bcg_block_page(html):
                log.warning("BCG Playwright 차단 페이지 감지 | url=%s", url)
                return ""

            return html
    except Exception as e:
        log.warning("BCG Playwright fallback 실패 | url=%s error=%s", url, e)
        return ""


def is_likely_bcg_block_page(html: str) -> bool:
    text = normalize_text(BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True))
    lowered = text.lower()

    block_markers = [
        "access denied",
        "you don't have permission to access",
        "request blocked",
        "forbidden",
    ]
    return any(marker in lowered for marker in block_markers)


def is_bcg_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.endswith("bcg.com")


def normalize_url(base_url: str, href: str) -> str | None:
    """상대 URL을 절대 URL로 바꾸고, BCG URL만 남긴다."""
    if not href:
        return None

    href = href.strip()

    if href.startswith("#"):
        return None

    if href.startswith("mailto:") or href.startswith("tel:"):
        return None

    url = urljoin(base_url, href)
    parsed = urlparse(url)

    if not parsed.scheme.startswith("http"):
        return None

    clean_url = parsed._replace(query="", fragment="").geturl()

    if not is_bcg_url(clean_url):
        return None

    return clean_url


def looks_like_article_url(url: str) -> bool:
    """BCG 상세 글 URL 후보만 남긴다."""
    parsed = urlparse(url)
    path = parsed.path.lower().rstrip("/")

    if "/publications/" not in path and "/insights/" not in path:
        return False

    excluded_endings = [
        "/publications",
        "/insights",
        "/capabilities/digital-technology-data/insights",
        "/capabilities/artificial-intelligence/insights",
    ]

    if any(path.endswith(ending) for ending in excluded_endings):
        return False

    if path.endswith((".pdf", ".jpg", ".jpeg", ".png", ".webp", ".gif")):
        return False

    return True


def extract_article_links(list_html: str, source_url: str) -> list[str]:
    """목록 페이지에서 상세 글 URL 후보를 추출한다."""
    soup = BeautifulSoup(list_html, "html.parser")

    urls = []

    for a in soup.find_all("a", href=True):
        url = normalize_url(source_url, a.get("href", ""))

        if not url:
            continue

        if not looks_like_article_url(url):
            continue

        urls.append(url)

    seen = set()
    unique_urls = []

    for url in urls:
        if url in seen:
            continue

        seen.add(url)
        unique_urls.append(url)

    return unique_urls


def extract_json_ld_objects(soup: BeautifulSoup) -> list[dict]:
    """상세 페이지의 JSON-LD를 파싱한다."""
    objects = []

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text(strip=True)

        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        if isinstance(data, dict):
            objects.append(data)
        elif isinstance(data, list):
            objects.extend(item for item in data if isinstance(item, dict))

    return objects


def parse_date_string(date_text: str | None) -> datetime | None:
    """날짜 문자열을 datetime으로 변환한다."""
    if not date_text:
        return None

    date_text = normalize_text(date_text)

    try:
        if "T" in date_text:
            dt = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
            return ensure_aware_utc(dt)
    except Exception:
        pass

    for fmt in [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]:
        try:
            dt = datetime.strptime(date_text, fmt)
            return ensure_aware_utc(dt)
        except Exception:
            continue

    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        date_text,
        flags=re.IGNORECASE,
    )

    if match:
        return parse_date_string(match.group(0))

    return None


def extract_published_at(soup: BeautifulSoup) -> datetime | None:
    """상세 페이지에서 발행일을 추출한다."""
    meta_candidates = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "article:published_time"}),
        ("meta", {"name": "date"}),
        ("meta", {"name": "publishdate"}),
        ("meta", {"name": "publishedDate"}),
        ("meta", {"itemprop": "datePublished"}),
        ("meta", {"property": "og:updated_time"}),
    ]

    for tag_name, attrs in meta_candidates:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get("content"):
            dt = parse_date_string(tag.get("content"))
            if dt:
                return ensure_aware_utc(dt)

    for obj in extract_json_ld_objects(soup):
        for key in ["datePublished", "dateCreated", "dateModified"]:
            value = obj.get(key)
            dt = parse_date_string(value)
            if dt:
                return ensure_aware_utc(dt)

    page_text = soup.get_text(" ", strip=True)
    dt = parse_date_string(page_text)

    return ensure_aware_utc(dt)


def extract_title(soup: BeautifulSoup) -> str:
    """상세 페이지 제목을 추출한다."""
    h1 = soup.find("h1")
    if h1:
        title = normalize_text(h1.get_text(" ", strip=True))
        if title:
            return title

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return normalize_text(og_title["content"])

    title_tag = soup.find("title")
    if title_tag:
        return normalize_text(title_tag.get_text(" ", strip=True))

    return "BCG article"


def extract_description(soup: BeautifulSoup) -> str | None:
    """상세 페이지 요약 설명을 추출한다."""
    for attrs in [
        {"name": "description"},
        {"property": "og:description"},
        {"name": "twitter:description"},
    ]:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return normalize_text(tag["content"])

    return None


def remove_unwanted_tags(soup: BeautifulSoup) -> None:
    """본문 추출 방해 태그 제거."""
    for tag in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "form",
            "button",
            "aside",
        ]
    ):
        tag.decompose()


def extract_body_content(soup: BeautifulSoup) -> str:
    """상세 페이지 HTML 본문 텍스트를 추출한다."""
    soup_copy = BeautifulSoup(str(soup), "html.parser")
    remove_unwanted_tags(soup_copy)

    candidates = []

    for selector in [
        "main",
        "article",
        "[role='main']",
        ".article",
        ".article-body",
        ".content",
        ".body",
        ".rich-text",
    ]:
        node = soup_copy.select_one(selector)
        if node:
            candidates.append(node)

    if candidates:
        main_node = max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))
    else:
        main_node = soup_copy.body or soup_copy

    texts = []

    for tag in main_node.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"]):
        text = normalize_text(tag.get_text(" ", strip=True))

        if not text:
            continue

        if len(text) < 20 and tag.name not in ["h1", "h2", "h3", "h4"]:
            continue

        lower = text.lower()

        if any(
            phrase in lower
            for phrase in [
                "cookie",
                "accept cookies",
                "sign up",
                "subscribe",
                "share",
                "download",
                "print",
                "contact us",
                "manage subscriptions",
            ]
        ) and len(text) < 180:
            continue

        texts.append(text)

    seen = set()
    unique_texts = []

    for text in texts:
        if text in seen:
            continue

        seen.add(text)
        unique_texts.append(text)

    return "\n\n".join(unique_texts)


def get_image_url_from_tag(base_url: str, img_tag) -> str | None:
    """img 태그에서 실제 이미지 URL을 추출한다."""
    candidates = [
        img_tag.get("src"),
        img_tag.get("data-src"),
        img_tag.get("data-original"),
        img_tag.get("data-image-src"),
        img_tag.get("data-lazy-src"),
    ]

    srcset = img_tag.get("srcset") or img_tag.get("data-srcset")
    if srcset:
        parts = [part.strip().split(" ")[0] for part in srcset.split(",") if part.strip()]
        if parts:
            candidates.append(parts[-1])

    for candidate in candidates:
        if not candidate:
            continue

        url = urljoin(base_url, candidate)
        parsed = urlparse(url)

        # 이미지 CDN은 query param을 포함해야 원본이 내려오는 경우가 있어 query는 유지한다.
        if parsed.scheme.startswith("http"):
            return parsed._replace(fragment="").geturl()

    return None


def extract_body_image_urls(soup: BeautifulSoup, article_url: str) -> list[str]:
    """상세 본문 이미지 URL만 추출한다. 로컬 다운로드는 하지 않는다."""
    # 기존 구현은 img 태그만 보고 query를 제거해 404가 나는 케이스가 있었다.
    # 네이버 크롤러의 본문 이미지 추출 로직을 재사용해서:
    # - <picture><source srcset> 등도 포함
    # - 본문 영역 기반 노이즈 필터(logo/profile 등) 적용
    # - query 유지
    try:
        urls = extract_image_urls(str(soup), article_url)

        # BCG는 본문 이미지를 background-image(style)로 두는 경우가 있어 추가로 스캔한다.
        root_candidates = []
        for selector in [
            "main",
            "article",
            "[role='main']",
            ".article",
            ".article-body",
            ".content",
            ".rich-text",
        ]:
            node = soup.select_one(selector)
            if node:
                root_candidates.append(node)
        root = (
            max(root_candidates, key=lambda node: len(node.get_text(" ", strip=True)))
            if root_candidates
            else (soup.body or soup)
        )
        urls.extend(extract_background_image_urls(root, article_url))

        return dedupe_bcg_image_variants(urls)
    except Exception:
        pass

    image_urls = []

    body_candidates = []
    for selector in [
        "main",
        "article",
        "[role='main']",
        ".article",
        ".article-body",
        ".content",
        ".rich-text",
    ]:
        node = soup.select_one(selector)
        if node:
            body_candidates.append(node)

    if body_candidates:
        root = max(body_candidates, key=lambda node: len(node.get_text(" ", strip=True)))
    else:
        root = soup.body or soup

    for img in root.find_all("img"):
        url = get_image_url_from_tag(article_url, img)

        if not url:
            continue

        lower_url = url.lower()

        if any(token in lower_url for token in ["logo", "icon", "favicon", "sprite"]):
            continue

        image_urls.append(url)

    seen = set()
    unique_urls = []

    for url in image_urls:
        if url in seen:
            continue

        seen.add(url)
        unique_urls.append(url)

    return unique_urls


def extract_background_image_urls(root, base_url: str) -> list[str]:
    """본문 root 내부의 style/background-image URL도 이미지 후보로 수집한다."""
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        raw = (raw or "").strip()
        if not raw:
            return
        url = urljoin(base_url, raw)
        if not _looks_like_image_url(url):
            return
        if url in seen:
            return
        seen.add(url)
        found.append(url)

    # style="background-image:url(...)" 혹은 style 내 url(...)
    for node in root.find_all(True):
        style = node.get("style")
        if style:
            for raw in re.findall(r"url\\(['\\\"]?([^'\\\")]+)", str(style)):
                add(raw)

        for attr in ("data-bg", "data-background", "data-background-image", "data-src"):
            value = node.get(attr)
            if value:
                add(str(value))

    return found


def _looks_like_image_url(url: str) -> bool:
    """BCG CDN(dims4) 포함 케이스를 고려한 이미지 URL 판정."""
    if not url:
        return False
    lowered = url.lower()
    if lowered.startswith("data:"):
        return False
    if lowered.endswith((".svg", ".gif", ".ico")):
        return False
    parsed = urlparse(lowered)
    if parsed.path.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        return True
    qs = parse_qs(parsed.query)
    raw = qs.get("url", [""])[0]
    raw = unquote(raw or "").lower()
    if raw.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        return True
    return False


def bcg_image_canonical_key(url: str) -> str:
    """
    BCG 이미지 CDN(dims4) 변형 URL을 같은 원본으로 묶기 위한 키.
    - web-assets.bcg.com/dims4/.../?url=<encoded-original> 패턴은 url= 파라미터를 기준으로 묶는다.
    - 그 외는 netloc+path(+일부 query) 기반으로 묶는다.
    """
    if not url:
        return ""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if parsed.netloc.endswith("bcg.com") and "dims4" in parsed.path and "url" in qs:
        raw = qs.get("url", [""])[0]
        original = unquote(raw or "").strip()
        if original:
            return original
    return f"{parsed.netloc}{parsed.path}"


def bcg_image_variant_score(url: str) -> int:
    """같은 원본의 변형들 중 '가장 큰' 이미지를 고르기 위한 점수."""
    if not url:
        return 0
    # dims4 path에 resize/2880x1784 같은 값이 자주 있다.
    match = re.search(r"/resize/(\\d+)x(\\d+)", url)
    if match:
        w = int(match.group(1))
        h = int(match.group(2))
        return w * 10000 + h
    # crop만 있고 resize가 없는 경우는 낮은 점수
    match = re.search(r"/crop/(\\d+)x(\\d+)", url)
    if match:
        w = int(match.group(1))
        h = int(match.group(2))
        return w * 100 + h
    return 1


def dedupe_bcg_image_variants(urls: list[str]) -> list[str]:
    """BCG CDN이 같은 원본을 여러 사이즈로 제공하는 경우 중복 제거."""
    best_by_key: dict[str, str] = {}
    best_score: dict[str, int] = {}

    for url in urls or []:
        key = bcg_image_canonical_key(url)
        if not key:
            continue
        score = bcg_image_variant_score(url)
        if key not in best_by_key or score > best_score.get(key, -1):
            best_by_key[key] = url
            best_score[key] = score

    # 안정적인 순서를 위해 원래 입력 순서 기준으로 출력
    seen: set[str] = set()
    out: list[str] = []
    for url in urls or []:
        key = bcg_image_canonical_key(url)
        chosen = best_by_key.get(key)
        if not chosen or chosen in seen:
            continue
        if url == chosen:
            seen.add(chosen)
            out.append(chosen)
    # 혹시 chosen이 입력에 없었던 케이스가 생기면(이론상 없음) 보정
    for chosen in best_by_key.values():
        if chosen not in seen:
            seen.add(chosen)
            out.append(chosen)
    return out


def find_pdf_urls(soup: BeautifulSoup, article_url: str) -> list[str]:
    """상세 페이지 안의 PDF 링크를 찾는다."""
    pdf_urls = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = normalize_text(a.get_text(" ", strip=True)).lower()

        if ".pdf" not in href.lower() and "download" not in text and "pdf" not in text:
            continue

        url = normalize_url(article_url, href)

        if not url:
            continue

        pdf_urls.append(url)

    seen = set()
    unique_urls = []

    for url in pdf_urls:
        if url in seen:
            continue

        seen.add(url)
        unique_urls.append(url)

    return unique_urls


def should_keep_by_date(published_at: datetime | None, cutoff: datetime) -> tuple[bool, str]:
    """날짜 기준 저장 여부.

    published_at이 없으면 일단 저장한다.
    timezone 없는 날짜는 UTC 기준으로 보정한다.
    """
    published_at = ensure_aware_utc(published_at)
    cutoff = ensure_aware_utc(cutoff)

    if published_at is None:
        return True, "unknown_date_kept"

    if published_at >= cutoff:
        return True, "within_days"

    return False, "older_than_days"


def _chunk_text(text: str, max_chars: int = 2200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n{2,}|\r\n\r\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        if size + len(p) + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf = []
            size = 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


async def translate_article_to_korean(
    *,
    title: str,
    content: str,
    model: str,
) -> tuple[str | None, str | None, dict]:
    """
    BCG 글(영문)을 한글로 번역해서 저장할 수 있게 한다.

    - title/content는 번역본을 main 필드로 저장
    - 원문은 extra.original_title/original_content로 보존
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        log.warning("OPENAI_API_KEY 미설정: 번역 스킵")
        return None, None, {"translation_status": "no_api_key"}

    try:
        from openai import AsyncOpenAI
    except Exception as e:
        log.warning("openai 패키지 import 실패: 번역 스킵 | error=%s", e)
        return None, None, {"translation_status": "no_openai_pkg", "translation_error": str(e)}

    client = AsyncOpenAI(api_key=api_key)

    async def translate_text(text: str) -> str:
        if not (text or "").strip():
            return ""
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. "
                        "Translate the user's text into natural Korean. "
                        "Preserve names, numbers, and technical terms. "
                        "Do not add commentary."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        title_ko = await translate_text(title)

        chunks = _chunk_text(content, max_chars=2200)
        if not chunks:
            content_ko = ""
        else:
            translated_chunks: list[str] = []
            for chunk in chunks:
                translated_chunks.append(await translate_text(chunk))
            content_ko = "\n\n".join([c for c in translated_chunks if c])

        meta = {
            "translation_status": "success",
            "translation_model": model,
        }
        return title_ko or None, content_ko or None, meta
    except Exception as e:
        log.warning("번역 실패: 원문 유지 | error=%r", e)
        return None, None, {
            "translation_status": "error",
            "translation_model": model,
            "translation_error": str(e),
        }


class BcgCrawler(BaseCrawler):
    """BCG 글로벌 IT/AI 산업동향 크롤러."""

    def __init__(
        self,
        days: int,
        max_articles: int,
        output_path: Path,
        translate_ko: bool = False,
        translate_model: str | None = None,
    ):
        super().__init__(company=[])
        self.days = days
        self.max_articles = max_articles
        self.output_path = output_path
        self.translate_ko = translate_ko
        self.translate_model = translate_model or os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")

    async def crawl(self) -> list[RawArticle]:
        cutoff = ensure_aware_utc(datetime.now(timezone.utc) - timedelta(days=self.days))

        articles: list[RawArticle] = []
        seen_detail_urls: set[str] = set()

        async with httpx.AsyncClient(
            headers=REQUEST_HEADERS,
            timeout=60,
            follow_redirects=True,
            http2=False,
        ) as client:
            detail_targets: list[tuple[str, str, str]] = []

            for source_key, source_url in BCG_SOURCES.items():
                try:
                    log.info("목록 페이지 수집 | source=%s url=%s", source_key, source_url)

                    list_html = await fetch_text(client, source_url)
                    links = extract_article_links(list_html, source_url)

                    log.info("상세 URL 후보 | source=%s count=%d", source_key, len(links))

                    for link in links:
                        if link in seen_detail_urls:
                            continue

                        seen_detail_urls.add(link)
                        detail_targets.append((source_key, source_url, link))

                except Exception as e:
                    # 목록 페이지 실패는 실제 산업동향 글이 아니므로 JSON에 저장하지 않는다.
                    log.exception(
                        "목록 페이지 수집 실패 | source=%s url=%s error_type=%s error=%r",
                        source_key,
                        source_url,
                        type(e).__name__,
                        e,
                    )
                    continue

            success_count = 0

            for index, (source_key, source_url, detail_url) in enumerate(detail_targets, start=1):
                if success_count >= self.max_articles:
                    break

                try:
                    log.info(
                        "상세 페이지 처리 | %d/%d url=%s",
                        index,
                        len(detail_targets),
                        detail_url,
                    )

                    detail_html = await fetch_text(client, detail_url)
                    soup = BeautifulSoup(detail_html, "html.parser")

                    title = extract_title(soup)
                    description = extract_description(soup)
                    published_at = ensure_aware_utc(extract_published_at(soup))
                    keep, date_filter_status = should_keep_by_date(published_at, cutoff)

                    if not keep:
                        log.info(
                            "최근 %d일 밖이라 제외 | date=%s title=%s",
                            self.days,
                            published_at.isoformat() if published_at else None,
                            title,
                        )
                        continue

                    content = extract_body_content(soup)
                    image_urls = extract_body_image_urls(soup, detail_url)
                    pdf_urls = find_pdf_urls(soup, detail_url)

                    if self.translate_ko:
                        title_ko, content_ko, translation_meta = await translate_article_to_korean(
                            title=title,
                            content=content,
                            model=self.translate_model,
                        )
                        if title_ko:
                            # 원문은 extra에 보존하고, main 필드는 한글로 저장
                            translation_meta.update(
                                {
                                    "original_title": title,
                                    "original_content": content,
                                }
                            )
                            title = title_ko
                            content = content_ko or content
                        else:
                            translation_meta = {"translation_status": "skipped"}

                    article_id = make_id(
                        source_type="trend_report",
                        source_name="BCG",
                        title=title,
                        url=detail_url,
                    )

                    article = RawArticle(
                        id=article_id,
                        source_type="trend_report",
                        source_name="BCG",
                        title=title,
                        content=content,
                        url=detail_url,
                        published_at=published_at,
                        publisher="Boston Consulting Group",
                        company=[],
                        language="en",
                        content_type="html",
                        crawl_status="success",
                        error_message=None,
                        extra={
                            "collection": source_key,
                            "description": description,
                            "date_filter_status": date_filter_status,
                            "image_urls": image_urls,
                            "image_count": len(image_urls),
                            "pdf_urls": pdf_urls,
                            "article_index": success_count + 1,
                            **(translation_meta if self.translate_ko else {}),
                        },
                    )

                    articles.append(article)
                    success_count += 1

                except Exception as e:
                    # 상세 페이지 실패도 실제 산업동향 글이 아니므로 JSON에 저장하지 않는다.
                    log.exception("상세 페이지 처리 실패 | url=%s error=%r", detail_url, e)
                    continue

        return articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCG 글로벌 IT/AI 산업동향 크롤러")

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="최근 며칠 이내 글을 수집할지. 기본값: 7",
    )

    parser.add_argument(
        "--max-articles",
        type=int,
        default=100,
        help="최대 저장할 상세 글 수. 기본값: 100",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="결과 JSON 저장 경로. 미지정 시 src/crawler/crawler_results/bcg_crawler.json",
    )
    parser.add_argument(
        "--translate-ko",
        action="store_true",
        help=(
            "title/content를 한글로 번역해서 저장하고, 원문은 extra에 보관한다. "
            "(OPENAI_API_KEY 필요)"
        ),
    )
    parser.add_argument(
        "--translate-model",
        default=None,
        help="번역에 사용할 OpenAI 모델명. 미지정 시 OPENAI_TRANSLATE_MODEL 또는 gpt-4o-mini",
    )

    return parser.parse_args()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args()
    load_local_env()

    output_path = Path(args.output) if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crawler = BcgCrawler(
        days=args.days,
        max_articles=args.max_articles,
        output_path=output_path,
        translate_ko=args.translate_ko,
        translate_model=args.translate_model,
    )

    articles = await crawler.crawl()

    payload = [article.to_common_dict() for article in articles]

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    success_count = sum(1 for article in articles if article.crawl_status == "success")
    failed_count = sum(1 for article in articles if article.crawl_status == "failed")
    image_url_count = sum(int(article.extra.get("image_count", 0)) for article in articles)

    print(f"BCG 최근 {args.days}일 수집")
    print(f"전체 결과: {len(articles)}건")
    print(f"성공: {success_count}건")
    print(f"실패: {failed_count}건")
    print(f"본문 이미지 URL 수: {image_url_count}개")
    print(f"JSON 저장 위치: {output_path}")

    for article in articles[:5]:
        print("-" * 80)
        print(f"title: {article.title}")
        print(f"published_at: {article.published_at.isoformat() if article.published_at else None}")
        print(f"content length: {len(article.content or '')}")
        print(f"image_url_count: {article.extra.get('image_count')}")
        print(f"date_filter_status: {article.extra.get('date_filter_status')}")
        print(f"url: {article.url}")


if __name__ == "__main__":
    asyncio.run(main())
