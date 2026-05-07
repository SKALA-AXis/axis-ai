"""
5개 회사 공식 홈페이지 정보 크롤러

현재 구현 대상:
- 삼성SDS: 생성형 AI, 클라우드, 솔루션, 물류 소개 페이지

뉴스/인사이트/고객사례/리포트는 제외하고, 공식 홈페이지의 제품·서비스 소개성
페이지를 섹션별로 따라가며 수집한다.

실행:
cd /Users/shim-yujeong/workspace/final_pj/axis-ai
.venv/bin/python src/crawler/company_homepage_crawler.py --company samsung_sds

링크 확인:
.venv/bin/python src/crawler/company_homepage_crawler.py --company samsung_sds --debug-links
"""

import argparse
import json
import logging
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base import RawArticle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT = 25
MIN_CONTENT_LENGTH = 120


@dataclass(frozen=True)
class SectionSeed:
    section: str
    url: str


@dataclass(frozen=True)
class CompanyHomepageConfig:
    company: str
    source_name: str
    base_url: str
    seeds: tuple[SectionSeed, ...]
    allowed_path_keywords: tuple[str, ...]
    blocked_path_keywords: tuple[str, ...]


SAMSUNG_SDS_CONFIG = CompanyHomepageConfig(
    company="samsung_sds",
    source_name="samsung_sds_homepage",
    base_url="https://www.samsungsds.com/kr/index.html",
    seeds=(
        SectionSeed("generative_ai", "https://www.samsungsds.com/kr/why-sds-ai/why-sds-ai.html"),
        SectionSeed("generative_ai", "https://www.samsungsds.com/kr/chatgpt-enterprise/chatgpt-enterprise.html"),
        SectionSeed("generative_ai", "https://www.samsungsds.com/kr/copilot/brity-copilot.html"),
        SectionSeed("generative_ai", "https://www.samsungsds.com/kr/ai-automation/brity-automation.html"),
        SectionSeed("generative_ai", "https://www.samsungsds.com/kr/ai-fabrix/fabrix.html"),
        SectionSeed("cloud", "https://www.samsungsds.com/kr/samsung-cloud-platform/why-sds-cloud.html"),
        SectionSeed("cloud", "https://www.samsungsds.com/kr/enterprise-cloud/enterprise-cloud.html"),
        SectionSeed("solution", "https://www.samsungsds.com/kr/erp/erp.html"),
        SectionSeed("solution", "https://www.samsungsds.com/kr/collaboration-solution-gov/brity-works-gov.html"),
        SectionSeed("solution", "https://www.samsungsds.com/kr/srm/caidentia.html"),
        SectionSeed("logistics", "https://www.samsungsds.com/kr/logistics/logistics.html"),
    ),
    allowed_path_keywords=(
        "/kr/why-sds-ai/",
        "/kr/chatgpt-enterprise/",
        "/kr/copilot/",
        "/kr/ai-automation/",
        "/kr/ai-fabrix/",
        "/kr/samsung-cloud-platform/",
        "/kr/enterprise-cloud/",
        "/kr/cloud/",
        "/kr/erp/",
        "/kr/srm/",
        "/kr/collaboration-solution",
        "/kr/brity",
        "/kr/logistics/",
        "/kr/cello",
    ),
    blocked_path_keywords=(
        "/kr/news/",
        "/kr/insights/",
        "/kr/case-study/",
        "/kr/vod/",
        "/kr/ar/",
        "/kr/company/",
        "/kr/etc/",
        "/kr/search/",
        "/kr/contact/",
        "/en/",
        "/cn/",
    ),
)


COMPANY_CONFIGS = {
    "samsung_sds": SAMSUNG_SDS_CONFIG,
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
    ".header",
    ".footer",
    ".gnb",
    ".lnb",
    ".nav",
    ".breadcrumb",
    ".cookie",
    ".modal",
    ".popup",
    ".family",
    ".family-site",
    ".sitemap",
]

STOP_MARKERS = [
    "개인정보처리방침",
    "이메일무단수집거부",
    "Family Site",
    "FAMILY SITE",
    "COPYRIGHT",
    "Copyright",
    "©",
]


def get_default_output_path(file_format: str = "json") -> Path:
    output_dir = Path(__file__).resolve().parent / "crawler_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{Path(__file__).stem}.{file_format}"


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if path != "/":
        path = path.rstrip("/")
    return urlunparse(parsed._replace(path=path, fragment="", query=""))


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content.decode("utf-8", errors="replace")


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def remove_noise_tags(soup: BeautifulSoup) -> None:
    for tag in soup.select(", ".join(NOISE_SELECTORS)):
        tag.decompose()


def is_same_domain(base_url: str, target_url: str) -> bool:
    base_domain = urlparse(base_url).netloc.replace("www.", "")
    target_domain = urlparse(target_url).netloc.replace("www.", "")
    return base_domain == target_domain


def is_allowed_url(url: str, config: CompanyHomepageConfig) -> bool:
    parsed = urlparse(url)
    path = parsed.path

    if not is_same_domain(config.base_url, url):
        return False

    if not path.startswith("/kr/"):
        return False

    if any(keyword in path for keyword in config.blocked_path_keywords):
        return False

    if not path.endswith(".html"):
        return False

    return any(keyword in path for keyword in config.allowed_path_keywords)


def extract_title(soup: BeautifulSoup, fallback_url: str) -> str:
    title_candidates = [
        soup.select_one("meta[property='og:title']"),
        soup.select_one("meta[name='title']"),
        soup.select_one("h1"),
        soup.select_one("h2"),
        soup.title,
    ]

    for tag in title_candidates:
        if not tag:
            continue

        if tag.name == "meta":
            title = clean_text(tag.get("content"))
        else:
            title = clean_text(tag.get_text(" ", strip=True))

        title = re.sub(r"\s*\|\s*삼성SDS\s*$", "", title).strip()
        title = re.sub(r"\s*-\s*삼성SDS\s*$", "", title).strip()

        if title and len(title) >= 2:
            return title

    return Path(urlparse(fallback_url).path).stem.replace("-", " ")


def trim_content(text: str, title: str) -> str:
    text = clean_text(text)
    title = clean_text(title)

    if title and title in text:
        text = text[text.find(title):]

    for marker in STOP_MARKERS:
        if marker in text:
            text = text.split(marker)[0].strip()

    return clean_text(text)


def extract_content(soup: BeautifulSoup, title: str) -> str | None:
    remove_noise_tags(soup)

    selectors = [
        "main",
        "#container",
        ".container",
        ".content",
        ".contents",
        ".contents-wrap",
        ".sub-content",
        "body",
    ]

    best_text = ""

    for selector in selectors:
        for tag in soup.select(selector):
            text = clean_text(tag.get_text(" ", strip=True))
            if len(text) > len(best_text):
                best_text = text

    best_text = trim_content(best_text, title)

    if len(best_text) < MIN_CONTENT_LENGTH:
        return None

    return best_text


def extract_image_urls(soup: BeautifulSoup, page_url: str) -> list[str]:
    image_urls = []
    seen = set()

    for tag in soup.select("meta[property='og:image'], img[src], source[srcset]"):
        raw = tag.get("content") or tag.get("src") or tag.get("srcset")
        if not raw:
            continue

        raw = raw.split(",")[0].strip().split(" ")[0]
        image_url = normalize_url(urljoin(page_url, raw))

        if image_url in seen:
            continue

        seen.add(image_url)
        image_urls.append(image_url)

    return image_urls[:20]


def extract_links(soup: BeautifulSoup, page_url: str, config: CompanyHomepageConfig) -> list[str]:
    links = []
    seen = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href")
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue

        next_url = normalize_url(urljoin(page_url, href))

        if next_url in seen:
            continue

        if not is_allowed_url(next_url, config):
            continue

        seen.add(next_url)
        links.append(next_url)

    return links


def classify_section(url: str, fallback_section: str) -> str:
    path = urlparse(url).path

    section_rules = [
        (
            "generative_ai",
            (
                "/why-sds-ai/",
                "/chatgpt-enterprise/",
                "/copilot/",
                "/ai-automation/",
                "/ai-fabrix/",
            ),
        ),
        ("cloud", ("/samsung-cloud-platform/", "/enterprise-cloud/", "/cloud/")),
        ("solution", ("/erp/", "/srm/", "/collaboration-solution", "/brity")),
        ("logistics", ("/logistics/", "/cello")),
    ]

    for section, keywords in section_rules:
        if any(keyword in path for keyword in keywords):
            return section

    return fallback_section


class CompanyHomepageCrawler:
    def __init__(
        self,
        config: CompanyHomepageConfig,
        max_pages: int,
        max_depth: int,
        debug_links: bool = False,
    ):
        self.config = config
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.debug_links = debug_links

    def crawl(self) -> list[RawArticle]:
        queue = deque(
            (normalize_url(seed.url), seed.section, 0, "seed")
            for seed in self.config.seeds
        )
        visited = set()
        articles = []

        while queue and len(articles) < self.max_pages:
            url, section, depth, discovered_from = queue.popleft()

            if url in visited:
                continue

            visited.add(url)

            try:
                html = fetch_html(url)
                soup = make_soup(html)
                title = extract_title(soup, url)
                content = extract_content(soup, title)
                image_urls = extract_image_urls(soup, url)
                page_section = classify_section(url, section)

                if content:
                    articles.append(
                        RawArticle(
                            url=url,
                            title=title,
                            content=content,
                            published_at=None,
                            source_name=self.config.source_name,
                            source_type="official",
                            publisher=self.config.company,
                            company=[self.config.company],
                            language="ko",
                            content_type="html",
                            crawl_status="success",
                            extra={
                                "section": page_section,
                                "depth": depth,
                                "image_urls": image_urls,
                            },
                        )
                    )
                    log.info("수집 완료 | section=%s depth=%s title=%s", page_section, depth, title)
                else:
                    log.warning("본문 부족으로 제외 | url=%s", url)

                if depth >= self.max_depth:
                    continue

                for next_url in extract_links(soup, url, self.config):
                    if next_url in visited:
                        continue
                    next_section = classify_section(next_url, page_section)
                    queue.append((next_url, next_section, depth + 1, url))
                    if self.debug_links:
                        log.info(
                            "후보 링크 | section=%s depth=%s url=%s",
                            next_section,
                            depth + 1,
                            next_url,
                        )

            except Exception as e:
                log.warning("수집 실패 | url=%s error=%s", url, e)

        return articles


def save_json(articles: list[RawArticle], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [article.to_common_dict() for article in articles]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="회사 공식 홈페이지 정보 크롤러")
    parser.add_argument(
        "--company",
        choices=sorted(COMPANY_CONFIGS.keys()),
        default="samsung_sds",
        help="수집 대상 회사",
    )
    parser.add_argument("--max-pages", type=int, default=40, help="최대 저장 페이지 수")
    parser.add_argument("--max-depth", type=int, default=2, help="seed 기준 최대 링크 탐색 깊이")
    parser.add_argument("--output", default=None, help="저장 경로")
    parser.add_argument("--debug-links", action="store_true", help="후보 링크 로그 출력")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = COMPANY_CONFIGS[args.company]
    output_path = Path(args.output) if args.output else get_default_output_path("json")

    crawler = CompanyHomepageCrawler(
        config=config,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        debug_links=args.debug_links,
    )
    articles = crawler.crawl()
    save_json(articles, output_path)

    section_counts: dict[str, int] = {}
    for article in articles:
        section = article.extra.get("section", "unknown")
        section_counts[section] = section_counts.get(section, 0) + 1

    print(f"수집 결과: {len(articles)}건")
    print(f"섹션별 결과: {section_counts}")
    print(f"JSON 저장 위치: {output_path}")


if __name__ == "__main__":
    main()
