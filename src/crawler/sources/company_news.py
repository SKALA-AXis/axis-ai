"""
5개 회사 공식 뉴스 통합 크롤러

수집 대상:
- 삼성SDS: 보도자료만 수집, "언론이 본 삼성SDS" 제외
- SK AX: 공식 뉴스룸 수집
- 현대오토에버: 공식 PR 뉴스 수집
- 포스코DX: 공식 NEWS 수집
- LG CNS: 공식 보도자료/Press 수집

저장 규칙:
- 현재 파이썬 파일명이 company_news_crawler.py 라면
- src/crawler/crawler_results/company_news_crawler.json 으로 자동 저장

실행:
cd /Users/shim-yujeong/workspace/final_pj/axis-ai
.venv/bin/python src/crawler/company_news_crawler.py --limit 3

후보 URL 확인:
.venv/bin/python src/crawler/company_news_crawler.py --limit 3 --debug-candidates
"""

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.companies import company_name_ko  # noqa: E402
from src.crawler.base import RawArticle  # noqa: E402
from src.crawler.base_crawler import BaseCrawler  # noqa: E402

# =========================================================
# 0. 로깅 설정
# =========================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# =========================================================
# 1. 기본 설정
# =========================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT = 30
OFFICIAL_MAX_AGE_DAYS = 30


COMPANY_CONFIGS = [
    {
        "company": "samsung_sds",
        "source_name": "Samsung SDS Press Release",
        "source_type": "official",
        "list_url": "https://www.samsungsds.com/kr/news/index.html",
    },
    {
        "company": "sk_ax",
        "source_name": "SK AX Newsroom",
        "source_type": "official",
        "list_url": "https://www.skax.co.kr/company/news-rooms",
    },
    {
        "company": "hyundai_autoever",
        "source_name": "Hyundai AutoEver News",
        "source_type": "official",
        "list_url": "https://www.hyundai-autoever.com/kor/about/pr/news/list.do",
    },
    {
        "company": "posco_dx",
        "source_name": "POSCO DX NewsRoom",
        "source_type": "official",
        "list_url": "https://www.poscodx.com/kor/pr/newsRoom.do",
    },
    {
        "company": "lg_cns",
        "source_name": "LG CNS Press",
        "source_type": "official",
        "list_url": "https://www.lgcns.com/kr/newsroom/press.page_1",
    },
]


BLOCK_WORDS = [
    "개인정보처리방침",
    "이메일무단수집거부",
    "이메일 무단수집거부",
    "사이트맵",
    "문의하기",
    "Family Site",
    "FAMILY SITE",
    "패밀리 사이트",
    "채용",
    "로그인",
    "회원가입",
    "검색",
    "검색버튼",
    "더보기",
    "더 보기",
    "이전",
    "다음",
    "닫기",
    "TOP",
    "HOME",
    "본문 바로가기",
    "본문 바로 가기",
    "뉴스레터",
    "구독",
    "공유하기",
    "자세히 보기",
    "자세히 보기 →",
    "전체보기",
    "목록 보기",
    "Go to List",
    "Prev",
    "Next",
]


DATE_PATTERNS = [
    r"\d{4}[.-]\d{1,2}[.-]\d{1,2}",
    r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일",
]


# =========================================================
# 2. 공통 유틸 함수
# =========================================================


def get_default_output_path(file_format: str = "json") -> str:
    current_file_path = Path(__file__).resolve()
    current_file_name = current_file_path.stem

    output_dir = current_file_path.parent / "crawler_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir / f"{current_file_name}.{file_format}")


def fix_mojibake(text: str | None) -> str:
    if not text:
        return ""

    mojibake_markers = ["ì", "í", "ë", "ê", "â", "ã"]

    if not any(marker in text for marker in mojibake_markers):
        return text

    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def clean_text(text: str | None) -> str:
    if not text:
        return ""

    text = fix_mojibake(text)
    return re.sub(r"\s+", " ", text).strip()


def decode_js_string(text: str) -> str:
    try:
        return json.loads(f'"{text}"')
    except Exception:
        return text.replace(r"\/", "/").replace(r"\"", '"')


def strip_trailing_date(text: str) -> str:
    text = clean_text(text)
    return clean_text(
        re.sub(
            r"\s*(\d{4}[.-]\d{1,2}[.-]\d{1,2}|\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일)\s*$",
            "",
            text,
        )
    )


def parse_date_to_datetime(text: str | None) -> datetime | None:
    if not text:
        return None

    text = clean_text(text)

    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text)

        if not match:
            continue

        raw = match.group(0)

        try:
            raw = raw.replace("년", "-")
            raw = raw.replace("월", "-")
            raw = raw.replace("일", "")
            raw = raw.replace(".", "-")

            return date_parser.parse(raw).astimezone()

        except Exception:
            return None

    return None


def is_bad_title(title: str) -> bool:
    title = clean_text(title)

    if not title:
        return True

    exact_bad_titles = {
        "자세히 보기",
        "자세히 보기 →",
        "더보기",
        "더 보기",
        "전체보기",
        "목록 보기",
        "검색",
        "뉴스",
        "NEWS",
        "PR 센터",
        "INSIGHT",
        "언론보도",
        "보도자료",
    }

    if title in exact_bad_titles:
        return True

    if len(title) < 8:
        return True

    if len(title) > 180:
        return True

    for word in BLOCK_WORDS:
        word = clean_text(word)
        if len(word) >= 4 and word in title:
            return True

    return False


def same_domain(base_url: str, target_url: str) -> bool:
    base_domain = urlparse(base_url).netloc.replace("www.", "")
    target_domain = urlparse(target_url).netloc.replace("www.", "")

    return base_domain == target_domain


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if path != "/":
        path = path.rstrip("/")
    return urlunparse(parsed._replace(path=path, fragment=""))


def fetch_html_by_requests(url: str) -> str:
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    try:
        return response.content.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        encoding = response.apparent_encoding or "utf-8"
        return response.content.decode(encoding, errors="replace")


def fetch_html_by_curl(url: str) -> str:
    """
    포스코DX처럼 requests/Playwright에서 접속이 거부될 수 있는 사이트를 위한 fallback.
    macOS의 curl을 사용하고, -4 옵션으로 IPv4를 강제한다.
    """

    command = [
        "curl",
        "-4",
        "-L",
        "-sS",
        "--connect-timeout",
        "8",
        "--max-time",
        str(REQUEST_TIMEOUT),
        "-A",
        HEADERS["User-Agent"],
        "-H",
        f"Accept-Language: {HEADERS['Accept-Language']}",
        url,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(f"curl fallback 실패: {result.stderr.strip()}")

    html = result.stdout.strip()

    if not html:
        raise RuntimeError("curl fallback 실패: 응답 HTML이 비어 있습니다.")

    return html


def fetch_html_by_playwright(url: str) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        page = browser.new_page(
            user_agent=HEADERS["User-Agent"],
            locale="ko-KR",
        )

        page.goto(
            url,
            wait_until="networkidle",
            timeout=30000,
        )

        time.sleep(2)

        html = page.content()

        browser.close()

    return html


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def remove_noise_tags(soup: BeautifulSoup) -> None:
    noise_selectors = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "button",
        ".header",
        ".footer",
        ".gnb",
        ".lnb",
        ".nav",
        ".breadcrumb",
        ".cookie",
        ".modal",
        ".popup",
        ".popup-wrap",
        ".family",
        ".family-site",
        ".sitemap",
        ".newsletter",
    ]

    for tag in soup.select(", ".join(noise_selectors)):
        tag.decompose()


def get_near_text(tag, depth: int = 5) -> str:
    parent_text = ""
    parent = tag

    for _ in range(depth):
        if parent is None:
            break

        parent_text += " " + clean_text(parent.get_text(" ", strip=True))
        parent = parent.parent

    return clean_text(parent_text)


def pick_title_from_card(a_tag) -> str:
    """
    카드형 목록에서 실제 제목을 찾는다.
    a 태그 텍스트가 '자세히 보기' 같은 버튼이면
    주변 카드 내부의 제목 태그를 다시 찾는다.
    """

    direct_title = clean_text(a_tag.get_text(" ", strip=True))

    if not is_bad_title(direct_title):
        return direct_title

    parent = a_tag

    for _ in range(6):
        if parent is None:
            break

        for selector in [
            "h1",
            "h2",
            "h3",
            "h4",
            "strong",
            ".title",
            ".tit",
            ".subject",
            ".news-title",
            ".card-title",
        ]:
            title_tag = parent.select_one(selector)

            if title_tag:
                title = clean_text(title_tag.get_text(" ", strip=True))

                if not is_bad_title(title):
                    return title

        parent = parent.parent

    return direct_title


def add_candidate(
    candidates: list[dict],
    seen_urls: set[str],
    list_url: str,
    href: str | None,
    title: str,
    published_at: datetime | None = None,
    score: int = 10,
) -> None:
    if not href:
        return

    if href.startswith("#"):
        return

    title = strip_trailing_date(title)

    if is_bad_title(title):
        return

    detail_url = normalize_url(urljoin(list_url, href))

    if normalize_url(list_url) == detail_url:
        return

    if not same_domain(list_url, detail_url):
        return

    if detail_url in seen_urls:
        return

    candidates.append(
        {
            "title": title,
            "url": detail_url,
            "published_at": published_at,
            "score": score,
        }
    )

    seen_urls.add(detail_url)


def sort_candidates(candidates: list[dict]) -> list[dict]:
    candidates.sort(
        key=lambda item: (
            item["published_at"] or datetime(1970, 1, 1).astimezone(),
            item["score"],
        ),
        reverse=True,
    )

    return candidates


# =========================================================
# 3. 회사별 목록 페이지 후보 링크 추출
# =========================================================


def extract_samsung_sds_links(
    list_url: str,
    soup: BeautifulSoup,
) -> list[dict]:
    """
    삼성SDS 전용 후보 추출.

    기준:
    - /kr/news/*.html 상세 글만 수집
    - /kr/news/index.html 제외
    - '언론이 본 삼성SDS' 섹션 제외
    - 보도자료 영역만 수집
    """

    candidates = []
    seen_urls = set()

    exclude_section_keywords = [
        "언론이 본 삼성SDS",
        "SDS seen by the media",
        "SDS_seen_by_the_media",
        "언론보도",
    ]

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        detail_url = urljoin(list_url, href)
        path = urlparse(detail_url).path

        if "/kr/news/" not in path:
            continue

        if path.endswith("/index.html"):
            continue

        if not path.endswith(".html"):
            continue

        near_text = get_near_text(a_tag, depth=5)

        if any(keyword in near_text for keyword in exclude_section_keywords):
            continue

        title = pick_title_from_card(a_tag)
        published_at = parse_date_to_datetime(near_text)

        add_candidate(
            candidates=candidates,
            seen_urls=seen_urls,
            list_url=list_url,
            href=href,
            title=title,
            published_at=published_at,
            score=10,
        )

    return sort_candidates(candidates)


def extract_skax_newsroom_links(
    list_url: str,
    soup: BeautifulSoup,
) -> list[dict]:
    """
    SK AX 뉴스룸 전용 후보 추출.

    기준:
    - /company/news-room/숫자
    - /company/news-rooms/숫자
    - /company/news-room/{slug}
    - /company/news-rooms/{slug}
    """

    candidates = []
    seen_urls = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        detail_url = urljoin(list_url, href)
        path = urlparse(detail_url).path

        is_news_detail = bool(re.search(r"/company/news-rooms?/[^/?#]+", path))

        if not is_news_detail:
            continue

        if path.rstrip("/") == "/company/news-rooms":
            continue

        title = pick_title_from_card(a_tag)
        near_text = get_near_text(a_tag, depth=6)
        published_at = parse_date_to_datetime(near_text)

        add_candidate(
            candidates=candidates,
            seen_urls=seen_urls,
            list_url=list_url,
            href=href,
            title=title,
            published_at=published_at,
            score=10,
        )

    embedded_patterns = [
        (
            r'\\"type\\":\\"newsroom\\".*?'
            r'\\"title\\":\\"([^\\"]+)\\".*?'
            r'\\"permalink\\":\\"(/company/news-room/[^\\"]+)\\".*?'
            r'\\"date\\":\\"([^\\"]+)'
        ),
        (
            r'"type":"newsroom".*?'
            r'"title":"([^"]+)".*?'
            r'"permalink":"(/company/news-room/[^"]+)".*?'
            r'"date":"([^"]+)'
        ),
    ]

    for script in soup.find_all("script"):
        script_text = script.string or script.get_text()
        for pattern in embedded_patterns:
            for match in re.finditer(pattern, script_text):
                title = clean_text(decode_js_string(match.group(1)))
                href = decode_js_string(match.group(2))
                published_at = parse_date_to_datetime(decode_js_string(match.group(3)))

                add_candidate(
                    candidates=candidates,
                    seen_urls=seen_urls,
                    list_url=list_url,
                    href=href,
                    title=title,
                    published_at=published_at,
                    score=20,
                )

    return sort_candidates(candidates)


def extract_hyundai_autoever_links(
    list_url: str,
    soup: BeautifulSoup,
) -> list[dict]:
    """
    현대오토에버 전용 후보 추출.

    기준:
    - 공식 PR 뉴스 상세 URL만 허용
    - business-area, contents.do 같은 사업 소개 페이지 차단
    """

    candidates = []
    seen_urls = set()

    blocked_keywords = [
        "/business-area/",
        "/ir/",
        "/sustainability/",
        "/recruit/",
        "/contents.do",
        "cntnSeq=",
        "blog",
        "newsletter",
        "insight",
    ]

    allowed_patterns = [
        r"/kor/about/pr/news/.*view",
        r"/kor/about/pr/news/.*detail",
        r"/kor/about/pr/news/.*\.do",
        r"/kor/about/pr/newsView",
        r"/kor/about/pr/newsRoom",
    ]

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        detail_url = urljoin(list_url, href)

        if any(keyword in detail_url for keyword in blocked_keywords):
            continue

        if "list.do" in detail_url:
            continue

        if not any(re.search(pattern, detail_url, re.IGNORECASE) for pattern in allowed_patterns):
            continue

        title = pick_title_from_card(a_tag)
        near_text = get_near_text(a_tag, depth=5)
        published_at = parse_date_to_datetime(near_text)

        add_candidate(
            candidates=candidates,
            seen_urls=seen_urls,
            list_url=list_url,
            href=href,
            title=title,
            published_at=published_at,
            score=10,
        )

    return sort_candidates(candidates)


def extract_poscodx_links(
    list_url: str,
    soup: BeautifulSoup,
) -> list[dict]:
    """
    포스코DX 전용 후보 추출.

    기준:
    - newsRoomView.do 상세 글만 수집
    """

    candidates = []
    seen_urls = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        detail_url = urljoin(list_url, href)

        if "newsRoomView.do" not in detail_url:
            continue

        title = pick_title_from_card(a_tag)
        near_text = get_near_text(a_tag, depth=5)
        published_at = parse_date_to_datetime(near_text)

        add_candidate(
            candidates=candidates,
            seen_urls=seen_urls,
            list_url=list_url,
            href=href,
            title=title,
            published_at=published_at,
            score=10,
        )

    return sort_candidates(candidates)


def extract_lgcns_links(
    list_url: str,
    soup: BeautifulSoup,
) -> list[dict]:
    """
    LG CNS 전용 후보 추출.

    기준:
    - /kr/newsroom/press 상세 글만 수집
    - press.page_1 같은 목록 URL 제외
    """

    candidates = []
    seen_urls = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        detail_url = urljoin(list_url, href)
        path = urlparse(detail_url).path

        if "/kr/newsroom/press/detail." not in path:
            continue

        if "press.page_" in path:
            continue

        if path.endswith("/press") or path.endswith("/press/"):
            continue

        title = pick_title_from_card(a_tag)
        near_text = get_near_text(a_tag, depth=5)
        published_at = parse_date_to_datetime(near_text)

        add_candidate(
            candidates=candidates,
            seen_urls=seen_urls,
            list_url=list_url,
            href=href,
            title=title,
            published_at=published_at,
            score=10,
        )

    return sort_candidates(candidates)


def extract_candidate_links(
    list_url: str,
    soup: BeautifulSoup,
    company_name: str,
) -> list[dict]:
    if company_name == "삼성SDS":
        return extract_samsung_sds_links(list_url, soup)

    if company_name == "SK AX":
        return extract_skax_newsroom_links(list_url, soup)

    if company_name == "현대오토에버":
        return extract_hyundai_autoever_links(list_url, soup)

    if company_name == "포스코DX":
        return extract_poscodx_links(list_url, soup)

    if company_name == "LG CNS":
        return extract_lgcns_links(list_url, soup)

    return []


# =========================================================
# 4. 상세 페이지 추출 함수
# =========================================================


def get_company_from_url(url: str) -> str:
    domain = urlparse(url).netloc

    if "samsungsds" in domain:
        return "삼성SDS"

    if "skax" in domain:
        return "SK AX"

    if "hyundai-autoever" in domain:
        return "현대오토에버"

    if "poscodx" in domain:
        return "포스코DX"

    if "lgcns" in domain:
        return "LG CNS"

    return ""


def extract_detail_title(
    soup: BeautifulSoup,
    fallback_title: str,
) -> str:
    fallback_title = strip_trailing_date(fallback_title)

    if not is_bad_title(fallback_title):
        return fallback_title

    title_selectors = [
        "h1",
        "h2",
        ".title",
        ".tit",
        ".subject",
        ".view-title",
        ".news-title",
        ".article-title",
        ".press-title",
    ]

    for selector in title_selectors:
        tag = soup.select_one(selector)

        if not tag:
            continue

        title = strip_trailing_date(tag.get_text(" ", strip=True))

        if not is_bad_title(title):
            return title

    if soup.title:
        title = strip_trailing_date(soup.title.get_text(" ", strip=True))

        if not is_bad_title(title):
            return title

    return fallback_title


def extract_detail_date(
    soup: BeautifulSoup,
    fallback_date: datetime | None,
) -> datetime | None:
    text = clean_text(soup.get_text(" ", strip=True))
    parsed_date = parse_date_to_datetime(text)

    return parsed_date or fallback_date


def dedupe_sentences(text: str) -> str:
    text = clean_text(text)

    if not text:
        return ""

    parts = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+", text)

    if len(parts) <= 1:
        return text

    seen = set()
    cleaned_parts = []

    for part in parts:
        part = clean_text(part)

        if not part:
            continue

        key = part[:120]

        if key in seen:
            continue

        seen.add(key)
        cleaned_parts.append(part)

    return clean_text(" ".join(cleaned_parts))


def extract_generic_content_from_divs(
    soup: BeautifulSoup,
) -> str:
    """
    p/li뿐 아니라 div까지 포함해서 긴 텍스트 후보를 모은다.
    """

    paragraph_texts = []

    for tag in soup.find_all(["p", "li", "div"]):
        text = clean_text(tag.get_text(" ", strip=True))

        if len(text) < 30:
            continue

        if any(word in text for word in BLOCK_WORDS):
            continue

        paragraph_texts.append(text)

    return clean_text(" ".join(paragraph_texts))


def extract_samsung_sds_content(
    soup: BeautifulSoup,
    title: str,
) -> str | None:
    """
    삼성SDS 상세 페이지 전용 본문 추출.

    기존 범용 selector가 제목/날짜/회사명까지만 잡는 문제가 있어서,
    삼성SDS는 body/main/container 단위에서 긴 텍스트를 확보한 뒤
    제목 이후의 실제 본문 시작 지점부터 추출한다.
    """

    remove_noise_tags(soup)

    title = clean_text(title)

    selectors = [
        "main",
        "article",
        "#container",
        ".content",
        ".contents",
        ".news-content",
        ".view-content",
        ".detail-content",
        ".cont",
        ".conts",
        ".board-view",
        ".view",
        ".article",
        ".article-content",
        ".news-view",
        ".view-area",
        "body",
    ]

    best_text = ""

    for selector in selectors:
        for tag in soup.select(selector):
            text = clean_text(tag.get_text(" ", strip=True))

            if title and title in text and len(text) > len(best_text):
                best_text = text

            elif len(text) > len(best_text):
                best_text = text

    if not best_text:
        best_text = clean_text(soup.get_text(" ", strip=True))

    if not best_text:
        return None

    if title and title in best_text:
        best_text = best_text[best_text.find(title) :]

    # 제목 + 날짜 + 회사명까지만 남는 경우를 보완하기 위해 본문 시작 마커를 찾는다.
    search_base = best_text[len(title) :] if title and best_text.startswith(title) else best_text

    start_markers = [
        "□",
        "■",
        "삼성SDS가",
        "삼성SDS는",
        "삼성SDS의",
        "삼성SDS 이",
        "삼성SDS,",
        "삼성SDS 관계자는",
    ]

    start_idx = -1

    for marker in start_markers:
        idx = search_base.find(marker)

        # 제목 자체의 '삼성SDS,'는 제외해야 하므로 너무 앞이면 무시
        if idx != -1 and idx > 5:
            start_idx = idx
            break

        # □, ■는 앞에 나와도 본문 시작일 가능성이 높음
        if marker in {"□", "■"} and idx != -1:
            start_idx = idx
            break

    if start_idx != -1:
        body_part = search_base[start_idx:]
        best_text = f"{title} {body_part}".strip() if title else body_part.strip()

    stop_markers = [
        "Image: like",
        "Image: facebook",
        "Image: twitter",
        "Image: linkedin",
        "<목록 보기",
        "< 목록 보기",
        "목록 보기",
        "공유하기",
        "이전글",
        "다음글",
        "개인정보처리방침",
        "이메일무단수집거부",
        "Family Site",
        "COPYRIGHT",
        "Copyright",
        "©",
    ]

    for marker in stop_markers:
        if marker in best_text:
            best_text = best_text.split(marker)[0].strip()

    best_text = dedupe_sentences(best_text)

    if len(best_text) < 100:
        return None

    return best_text


def trim_content_noise(
    text: str,
    title: str,
    company: str,
) -> str:
    text = clean_text(text)
    title = clean_text(title)

    if title and title in text:
        text = text[text.find(title) :]

    common_stop_markers = [
        "개인정보처리방침",
        "이메일무단수집거부",
        "Family Site",
        "FAMILY SITE",
        "패밀리 사이트",
        "뉴스레터",
        "COPYRIGHT",
        "Copyright",
        "©",
    ]

    company_stop_markers = {
        "삼성SDS": [
            "< 목록 보기",
            "<목록 보기",
            "목록 보기",
            "이전글",
            "다음글",
            "언론이 본 삼성SDS",
        ],
        "SK AX": [
            "개인정보처리방침",
            "이메일 무단수집거부",
            "Family Site",
            "COPYRIGHT",
        ],
        "현대오토에버": [
            "FAMILY SITE",
            "COPYRIGHT",
            "통합검색 서비스",
        ],
        "포스코DX": [
            "Go to List",
            "Prev",
            "Next",
            "D rive to e X cellence",
        ],
        "LG CNS": [
            "LG CNS Cookies",
            "뉴스레터",
            "패밀리 사이트",
            "최상위로 이동",
        ],
    }

    stop_markers = common_stop_markers + company_stop_markers.get(company, [])

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    return dedupe_sentences(text)


def extract_detail_content(
    soup: BeautifulSoup,
    detail_url: str,
    title: str,
) -> str | None:
    company = get_company_from_url(detail_url)

    # 삼성SDS는 상세 페이지에서 제목/날짜만 잡히는 문제가 있어 전용 추출을 먼저 시도한다.
    if company == "삼성SDS":
        samsung_content = extract_samsung_sds_content(
            soup=soup,
            title=title,
        )

        if samsung_content:
            return samsung_content

    remove_noise_tags(soup)

    company_selectors = {
        "삼성SDS": [
            "article",
            "main",
            ".view-content",
            ".news-content",
            ".content",
            ".contents",
            ".detail-content",
            ".cont",
            ".conts",
            ".board-view",
            ".view",
            ".article",
            ".article-content",
            ".news-view",
            ".view-area",
        ],
        "SK AX": [
            "article",
            "main",
            ".content",
            ".contents",
            ".detail",
            ".detail-content",
            ".news-content",
        ],
        "현대오토에버": [
            "article",
            "main",
            ".view-content",
            ".news-content",
            ".content",
            ".contents",
            ".detail-content",
        ],
        "포스코DX": [
            ".detail_con",
            ".detail_con_middle",
            "article",
            "main",
            ".view-content",
            ".board-view",
            ".detail-content",
        ],
        "LG CNS": [
            "article",
            "main",
            ".view-content",
            ".news-content",
            ".content",
            ".contents",
            ".detail-content",
        ],
    }

    default_selectors = [
        "article",
        "main",
        ".content",
        ".contents",
        ".view",
        ".view-content",
        ".news-content",
        ".article",
        ".article-content",
        ".press-content",
        ".board-view",
        ".detail",
        ".detail-content",
    ]

    selectors = company_selectors.get(company, default_selectors)

    best_text = ""

    for selector in selectors:
        for tag in soup.select(selector):
            text = clean_text(tag.get_text(" ", strip=True))

            if len(text) > len(best_text):
                best_text = text

    if len(best_text) < 100:
        best_text = extract_generic_content_from_divs(soup)

    best_text = trim_content_noise(
        text=best_text,
        title=title,
        company=company,
    )

    if len(best_text) < 50:
        return None

    return best_text


def crawl_detail_page(
    detail_url: str,
    fallback_title: str,
    fallback_date: datetime | None,
    use_render: bool,
) -> tuple[str, datetime | None, str | None, bool]:
    used_render = False
    company = get_company_from_url(detail_url)

    try:
        if company == "포스코DX":
            log.info("[포스코DX] 상세 페이지 curl -4 우선 시도: %s", detail_url)
            html = fetch_html_by_curl(detail_url)
            used_render = True
        else:
            html = fetch_html_by_requests(detail_url)

    except Exception:
        if not use_render:
            return fallback_title, fallback_date, None, used_render

        try:
            if company == "포스코DX":
                log.info("[포스코DX] 상세 페이지 requests fallback 재시도: %s", detail_url)
                html = fetch_html_by_requests(detail_url)
            else:
                html = fetch_html_by_playwright(detail_url)
            used_render = True

        except Exception as e:
            log.warning("[%s] 상세 페이지 fallback 실패: %s", company or "unknown", e)
            return fallback_title, fallback_date, None, used_render

    soup = make_soup(html)

    title = extract_detail_title(soup, fallback_title)
    published_at = extract_detail_date(soup, fallback_date)
    content = extract_detail_content(
        soup=soup,
        detail_url=detail_url,
        title=title,
    )

    # requests로 상세 페이지는 열렸지만 본문을 못 찾거나,
    # 삼성SDS처럼 제목/날짜 정도만 잡힌 경우 렌더링해서 재시도한다.
    if (content is None or len(content) < 120) and use_render:
        company = get_company_from_url(detail_url)

        try:
            log.info("[%s] 상세 본문 부족 → 렌더링/대체 방식 재시도: %s", company, detail_url)

            if company == "포스코DX":
                html = fetch_html_by_curl(detail_url)
            else:
                html = fetch_html_by_playwright(detail_url)

            used_render = True
            soup = make_soup(html)

            title = extract_detail_title(soup, title)
            published_at = extract_detail_date(soup, published_at)
            content = extract_detail_content(
                soup=soup,
                detail_url=detail_url,
                title=title,
            )

        except Exception as e:
            log.warning("[%s] 상세 본문 재시도 실패: %s", company or "unknown", e)

    return title, published_at, content, used_render


# =========================================================
# 5. 통합 크롤러 클래스
# =========================================================


class CompanyNewsCrawler(BaseCrawler):
    def __init__(
        self,
        latest_limit: int = 5,
        use_render: bool = False,
        render_fallback: bool = True,
        debug_candidates: bool = False,
    ):
        super().__init__(company=[])

        self.latest_limit = latest_limit
        self.use_render = use_render
        self.render_fallback = render_fallback
        self.debug_candidates = debug_candidates

    async def crawl(self) -> list[RawArticle]:
        all_articles: list[RawArticle] = []

        for config in COMPANY_CONFIGS:
            articles = await asyncio.to_thread(
                self._crawl_one_company,
                config,
            )

            all_articles.extend(articles)

        return all_articles

    def _fetch_list_html_with_fallback(
        self,
        company: str,
        list_url: str,
    ) -> tuple[str, bool]:
        """
        목록 페이지 HTML 수집.
        기본은 requests.
        포스코DX는 requests 실패 시 curl -4를 먼저 시도한다.
        그 외 회사는 Playwright를 시도한다.
        """

        if company == "포스코DX":
            try:
                log.info("[%s] 목록 curl -4 우선 시도", company)
                html = fetch_html_by_curl(list_url)
                return html, True
            except Exception as curl_error:
                log.warning("[%s] curl 우선 시도 실패 → requests 재시도: %s", company, curl_error)

        try:
            html = fetch_html_by_requests(list_url)
            return html, False

        except Exception as first_error:
            log.warning(
                "[%s] requests 목록 수집 실패 → fallback 재시도: %s",
                company,
                first_error,
            )

            if not (self.use_render or self.render_fallback):
                raise first_error

            if company == "포스코DX":
                try:
                    log.info("[%s] curl -4 fallback 재시도", company)
                    html = fetch_html_by_curl(list_url)
                    return html, True

                except Exception as curl_error:
                    log.warning(
                        "[%s] curl fallback 실패 → Playwright 재시도: %s",
                        company,
                        curl_error,
                    )

                    html = fetch_html_by_playwright(list_url)
                    return html, True

            html = fetch_html_by_playwright(list_url)
            return html, True

    def _crawl_one_company(
        self,
        config: dict,
    ) -> list[RawArticle]:
        company_id = config["company"]
        company = company_name_ko(company_id)
        list_url = config["list_url"]

        log.info("[%s] 목록 수집 시작: %s", company, list_url)

        try:
            html, used_list_render = self._fetch_list_html_with_fallback(
                company=company,
                list_url=list_url,
            )

            soup = make_soup(html)

            candidates = extract_candidate_links(
                list_url=list_url,
                soup=soup,
                company_name=company,
            )

            should_try_render = self.use_render or (
                self.render_fallback
                and len(candidates) < self.latest_limit
                and company in {"SK AX", "현대오토에버", "포스코DX", "LG CNS"}
            )

            if should_try_render:
                try:
                    log.info("[%s] 후보 부족/렌더 옵션 → 추가 fallback 재시도", company)

                    if company == "포스코DX":
                        html = fetch_html_by_curl(list_url)
                    else:
                        html = fetch_html_by_playwright(list_url)

                    soup = make_soup(html)

                    rendered_candidates = extract_candidate_links(
                        list_url=list_url,
                        soup=soup,
                        company_name=company,
                    )

                    if len(rendered_candidates) >= len(candidates):
                        candidates = rendered_candidates
                        used_list_render = True

                except Exception as e:
                    log.warning("[%s] 추가 fallback 실패: %s", company, e)

            log.info("[%s] 후보 기사 수: %s", company, len(candidates))

            if self.debug_candidates:
                for idx, candidate in enumerate(candidates, start=1):
                    log.info(
                        "[%s] 후보 %s | date=%s | title=%s | url=%s",
                        company,
                        idx,
                        candidate["published_at"],
                        candidate["title"],
                        candidate["url"],
                    )

            articles: list[RawArticle] = []

            for candidate in candidates[: self.latest_limit]:
                detail_url = candidate["url"]
                fallback_title = candidate["title"]
                fallback_date = candidate["published_at"]

                title, published_at, content, used_detail_render = crawl_detail_page(
                    detail_url=detail_url,
                    fallback_title=fallback_title,
                    fallback_date=fallback_date,
                    use_render=self.use_render or self.render_fallback,
                )

                if not _is_recent_official_article(published_at):
                    log.info(
                        "[%s] 오래된 공식 뉴스 제외 | date=%s title=%s",
                        company,
                        published_at,
                        title,
                    )
                    continue

                article = RawArticle(
                    url=detail_url,
                    title=title,
                    content=content,
                    published_at=published_at,
                    source_name=config["source_name"],
                    source_type=config["source_type"],
                    publisher=company,
                    company=[company_id],
                    language="ko",
                    content_type="html",
                    crawl_status="success",
                    error_message=None,
                    extra={
                        "list_url": list_url,
                        "company_name": company,
                        "candidate_score": candidate["score"],
                        "used_list_render": used_list_render,
                        "used_detail_render": used_detail_render,
                    },
                )

                articles.append(article)

                time.sleep(0.3)

            if not articles:
                return [
                    self._make_failed_article(
                        config=config,
                        error_message=(
                            "후보 기사 URL을 찾지 못했습니다. "
                            "동적 API 또는 selector 확인이 필요합니다."
                        ),
                    )
                ]

            log.info("[%s] 수집 완료: %s건", company, len(articles))

            return articles

        except Exception as e:
            log.exception("[%s] 수집 실패", company)

            return [
                self._make_failed_article(
                    config=config,
                    error_message=str(e),
                )
            ]

    def _make_failed_article(
        self,
        config: dict,
        error_message: str,
    ) -> RawArticle:
        company_id = config["company"]
        company = company_name_ko(company_id)

        return RawArticle(
            url=config["list_url"],
            title=f"{company} 크롤링 실패",
            content=None,
            published_at=None,
            source_name=config["source_name"],
            source_type=config["source_type"],
            publisher=company,
            company=[company_id],
            language="ko",
            content_type="html",
            crawl_status="failed",
            error_message=error_message,
            extra={
                "list_url": config["list_url"],
                "company_name": company,
            },
        )


def _is_recent_official_article(published_at: datetime | None) -> bool:
    if published_at is None:
        return False

    now = datetime.now(published_at.tzinfo) if published_at.tzinfo else datetime.now()
    age_days = (now - published_at).days
    return age_days <= OFFICIAL_MAX_AGE_DAYS


# =========================================================
# 6. 저장 함수
# =========================================================


def save_json(
    articles: list[RawArticle],
    output_path: str,
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = [article.to_common_dict() for article in articles]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# 7. 실행부
# =========================================================


async def main() -> None:
    parser = argparse.ArgumentParser(description="5개 회사 공식 뉴스 통합 크롤러")

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="회사별 최신 수집 개수",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="처음부터 Playwright 렌더링 사용",
    )

    parser.add_argument(
        "--no-render-fallback",
        action="store_true",
        help="requests 후보 부족/실패 시 fallback 사용하지 않기",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["json"],
        default="json",
        help="저장 형식",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="저장 파일명. 지정하지 않으면 crawler_results/현재파일명.json 으로 저장",
    )

    parser.add_argument(
        "--debug-candidates",
        action="store_true",
        help="회사별 후보 URL 목록을 로그로 출력",
    )

    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        output_path = get_default_output_path(args.format)

    crawler = CompanyNewsCrawler(
        latest_limit=args.limit,
        use_render=args.render,
        render_fallback=not args.no_render_fallback,
        debug_candidates=args.debug_candidates,
    )

    articles = await crawler.crawl()

    save_json(articles, output_path)

    success_count = sum(1 for article in articles if article.crawl_status == "success")

    failed_count = sum(1 for article in articles if article.crawl_status == "failed")

    print("\n==============================")
    print("5개 회사 뉴스 크롤링 완료")
    print(f"전체 결과 수: {len(articles)}건")
    print(f"성공: {success_count}건")
    print(f"실패: {failed_count}건")
    print(f"저장 파일: {output_path}")
    print("==============================")


if __name__ == "__main__":
    asyncio.run(main())
