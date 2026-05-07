"""RSS 피드 크롤러.

DB 저장 없이 로컬에서 단독 실행해 RSS 수집 결과를 확인할 수 있다.

relevance 판정은 후속 파이프라인에서 수행하므로 이 크롤러는 원천 RSS 수집만 담당한다.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import httpx
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.companies import COMPANY_ALIASES  # noqa: E402
from src.crawler.base import DailyLimitGuard, RawArticle  # noqa: E402
from src.crawler.base_crawler import BaseCrawler  # noqa: E402
from src.crawler.naver_crawler import (  # noqa: E402
    REQUEST_HEADERS,
    clean_publisher_name,
    extract_publisher,
    get_search_aliases,
    merge_articles_by_url,
    strip_html,
)
from src.crawler.parsers.article_content import (  # noqa: E402
    extract_body_text,
    extract_clean_body_text,
    extract_image_urls,
    extract_subtitle,
)

log = logging.getLogger(__name__)

PEER_ALIASES = COMPANY_ALIASES
DEFAULT_PEER_IDS = tuple(PEER_ALIASES.keys())
DEFAULT_MAX_ENTRIES = DailyLimitGuard.SOURCE_TYPE_LIMITS["news"]
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"

RSS_HEADERS = {
    **REQUEST_HEADERS,
    "Accept": "application/rss+xml,application/xml;q=0.9,text/xml;q=0.8,*/*;q=0.5",
}


class RssCrawler(BaseCrawler):
    def __init__(
        self,
        peer_id: str,
        aliases: list[str],
        search_queries: list[dict[str, str]] | None = None,
        max_entries_per_source: int = 50,
        fetch_body: bool = True,
        recent_hours: int = 24,
    ):
        super().__init__(peer_id)
        self.aliases = aliases
        self.search_queries = search_queries
        self.max_entries_per_source = max_entries_per_source
        self.fetch_body = fetch_body
        self.recent_hours = recent_hours

    async def crawl(self) -> list[RawArticle]:
        articles: list[RawArticle] = []

        async with httpx.AsyncClient(
            timeout=10,
            headers=RSS_HEADERS,
            follow_redirects=True,
        ) as client:
            for source_name, source in self._source_urls().items():
                try:
                    feed_text = await self._fetch_feed_text(client, source["url"])

                    if not feed_text:
                        continue

                    feed = feedparser.parse(feed_text)
                    entries = feed.entries[: self.max_entries_per_source]

                    before = len(articles)

                    articles.extend(
                        self._entry_to_article(source_name, source, entry)
                        for entry in entries
                    )

                    log.info(
                        "RSS 수집 완료 | peer=%s source=%s entries=%d",
                        self.peer_id,
                        source_name,
                        len(articles) - before,
                    )

                except Exception as e:
                    log.error("RSS 크롤링 실패 | source=%s error=%s", source_name, e)

            articles = filter_by_hours(articles, self.recent_hours)

            if self.fetch_body:
                await enrich_with_body_text_and_images(
                    client,
                    self._body_fetch_candidates(articles),
                )

        return articles

    def _body_fetch_candidates(self, articles: list[RawArticle]) -> list[RawArticle]:
        return articles

    async def _fetch_feed_text(self, client: httpx.AsyncClient, url: str) -> str:
        resp = await client.get(url)

        if self._is_blocked(resp.status_code):
            log.warning(
                "RSS 접근 차단 | peer=%s url=%s status=%d",
                self.peer_id,
                url,
                resp.status_code,
            )
            return ""

        if resp.status_code != 200:
            log.warning(
                "RSS fetch 실패 | peer=%s url=%s status=%d",
                self.peer_id,
                url,
                resp.status_code,
            )
            return ""

        return resp.text

    def _entry_to_article(
        self,
        source_name: str,
        source: dict[str, str],
        entry,
    ) -> RawArticle:
        title, publisher = split_title_and_publisher(
            strip_html(entry.get("title", "")),
            source_name=source_name,
            url=entry.get("link", ""),
        )
        content = first_text(
            entry.get("summary", ""),
            entry.get("description", ""),
            extract_content_value(entry),
        )

        return RawArticle(
            url=entry.get("link", ""),
            title=title,
            content=extract_rss_entry_text(content),
            published_at=parse_entry_date(entry),
            source_name=source_name,
            source_type="news",
            publisher=publisher,
            company=[self.peer_id],
            language="ko",
            content_type="rss",
            extra={
                "search_query": source.get("query", ""),
                **({"sector": source["sector"]} if source.get("sector") else {}),
                **({"feed_url": source["url"]} if source.get("url") else {}),
            },
        )

    def _source_urls(self) -> dict[str, dict[str, str]]:
        urls: dict[str, dict[str, str]] = {}
        query_specs = self.search_queries or [
            {"query": f'"{alias}"', "sector": ""} for alias in self.aliases
        ]

        for spec in query_specs:
            query_text = spec["query"]
            query = quote_plus(query_text)

            urls[f"google_news:{query_text}"] = {
                "url": GOOGLE_NEWS_RSS_URL.format(query=query),
                "sector": spec.get("sector", ""),
                "query": query_text,
            }

        return urls


def extract_content_value(entry) -> str:
    content = entry.get("content")

    if isinstance(content, list) and content:
        value = content[0].get("value", "")
        return str(value)

    return ""


def extract_rss_entry_text(content: str) -> str:
    if not content:
        return ""

    text = extract_clean_body_text(BeautifulSoup(str(content), "html.parser"))

    return text or strip_html(str(content))


def first_text(*values: str) -> str:
    for value in values:
        if value:
            return str(value)

    return ""


def split_title_and_publisher(
    title: str,
    source_name: str = "",
    url: str = "",
) -> tuple[str, str | None]:
    title = normalize_whitespace(title)
    publisher: str | None = None

    if is_google_news_source(source_name):
        title, publisher = split_google_news_title(title)

    if not publisher:
        publisher = publisher_from_url_or_source(url, source_name)

    return title, publisher


def split_google_news_title(title: str) -> tuple[str, str | None]:
    if " - " not in title:
        return title, None

    article_title, suffix = title.rsplit(" - ", 1)
    publisher = clean_publisher_name(suffix)

    if not article_title.strip() or not is_plausible_publisher(publisher):
        return title, None

    return normalize_whitespace(article_title), publisher


def publisher_from_url_or_source(url: str, source_name: str) -> str | None:
    _ = url

    if not is_google_news_source(source_name):
        return clean_publisher_name(source_name.split(":", 1)[0])

    return None


def is_google_news_source(source_name: str) -> bool:
    return source_name.startswith("google_news:")


def is_plausible_publisher(publisher: str | None) -> bool:
    if not publisher:
        return False

    if len(publisher) > 30:
        return False

    return not any(char in publisher for char in "<>{}[]")


def is_generic_feed_publisher(publisher: str | None) -> bool:
    if not publisher:
        return True

    return publisher.strip().lower() in {
        "google",
        "google news",
        "google 뉴스",
        "구글",
        "구글 뉴스",
        "google.com",
    }


def _extract_google_news_target_url(html: str) -> str:
    """
    Google News 중간 페이지에서 실제 원문 URL을 뽑아낸다.
    실패하면 빈 문자열을 반환한다.
    """
    soup = BeautifulSoup(html, "html.parser")

    for selector, attr in (
        ('meta[property="og:url"]', "content"),
        ('meta[name="og:url"]', "content"),
        ('link[rel="canonical"]', "href"),
    ):
        node = soup.select_one(selector)

        if node:
            value = str(node.get(attr) or "").strip()

            if value and "news.google.com" not in value:
                return value

    for a in soup.find_all("a", href=True):
        href = str(a.get("href") or "").strip()

        if not href:
            continue

        if href.startswith("http") and "news.google.com" not in href:
            return href

    return ""


async def _resolve_fetch_url_for_rss(client: httpx.AsyncClient, url: str) -> str:
    """
    RSS 원문 페이지를 fetch하기 위한 URL로 정규화한다.

    - Google News RSS 링크는 실제 언론사 링크로 해석한다.
    """
    url = url or ""

    if "news.google.com" not in url:
        return url

    try:
        token_match = re.search(r"/(?:rss/)?articles/([A-Za-z0-9_-]+)", url)

        if token_match:
            token = token_match.group(1)
            return f"https://news.google.com/articles/{token}?hl=ko&gl=KR&ceid=KR:ko"

        resp = await client.get(url)

        if resp.status_code != 200:
            return str(resp.url)

        target = _extract_google_news_target_url(resp.text)

        return target or str(resp.url)

    except Exception:
        return url


async def enrich_with_body_text_and_images(
    client: httpx.AsyncClient,
    articles: list[RawArticle],
) -> None:
    """
    RSS 기사 본문/부제/이미지 URL을 원문 페이지 기준으로 채운다.

    네이버 크롤러와 동일하게:
    - 본문 텍스트는 extract_body_text로 추출
    - 본문 안 이미지 URL만 extract_image_urls로 추출
    """
    for article in articles:
        url = article.url or ""

        if not url:
            continue

        try:
            fetch_url = await _resolve_fetch_url_for_rss(client, url)

            if fetch_url and fetch_url != url:
                article.extra["resolved_url"] = fetch_url

            resp = await client.get(fetch_url or url)

            if resp.status_code != 200:
                article.extra["body_fetch_status"] = f"failed:{resp.status_code}"
                continue

            html = resp.text
            base_url = str(resp.url)

            publisher = extract_publisher(html, base_url)
            if publisher and not is_generic_feed_publisher(publisher):
                article.publisher = publisher

            article.extra["image_urls"] = extract_image_urls(html, base_url)

            subtitle = extract_subtitle(html)
            if subtitle:
                article.extra["subtitle"] = subtitle

            body = extract_body_text(html)
            if body:
                article.content = body
                article.extra["body_fetch_status"] = "success"
            else:
                article.extra["body_fetch_status"] = "fallback_rss_summary"

        except Exception as e:
            article.extra["body_fetch_status"] = "error"
            article.extra["body_fetch_error"] = str(e)


def parse_entry_date(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")

    if parsed is not None:
        return datetime(*parsed[:6], tzinfo=timezone.utc)

    for key in ("published", "updated", "created"):
        value = entry.get(key)

        if not value:
            continue

        try:
            return parsedate_to_datetime(value)
        except (TypeError, ValueError):
            continue

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB 저장 없이 RSS 뉴스 크롤링 로컬 테스트")

    parser.add_argument(
        "--company",
        "--peer-id",
        dest="company",
        action="append",
        default=None,
        help="특정 회사 id만 테스트한다. 여러 번 지정 가능. --peer-id는 하위 호환 alias.",
    )

    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="직접 Google News RSS 검색어를 지정한다. 미지정 시 peer alias를 사용한다.",
    )

    parser.add_argument(
        "--max-entries",
        type=int,
        default=DEFAULT_MAX_ENTRIES,
        help=(
            "Google News RSS 검색어별 최대 entry 수. "
            f"기본 base.py news 한도({DEFAULT_MAX_ENTRIES})."
        ),
    )

    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="최근 N시간 이내 기사만 저장한다. 0 이하이면 시간 필터를 끈다. 기본 24.",
    )

    parser.add_argument(
        "--no-body",
        action="store_true",
        help="기사 본문 HTML 추가 수집을 끈다",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "수집 결과 JSON 경로. "
            "미지정 시 src/crawler/crawler_results/rss_crawler.json에 저장한다."
        ),
    )

    return parser.parse_args()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    peer_ids = args.company or list(DEFAULT_PEER_IDS)
    articles: list[RawArticle] = []

    for peer_id in peer_ids:
        aliases = args.query or list(get_search_aliases(peer_id))

        crawler = RssCrawler(
            peer_id=peer_id,
            aliases=aliases,
            max_entries_per_source=args.max_entries,
            fetch_body=not args.no_body,
            recent_hours=args.hours,
        )

        peer_articles = await crawler.crawl()

        log.info("피어사별 RSS 수집 완료 | peer=%s count=%d", peer_id, len(peer_articles))

        articles.extend(peer_articles)

    recent_articles = filter_by_hours(articles, args.hours)
    selected_articles = merge_articles_by_url(recent_articles)

    payload = [article.to_common_dict() for article in selected_articles]

    output_path = Path(args.output) if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    image_dir = write_image_url_files(selected_articles, output_path)

    print(f"수집 결과: {len(articles)}건")
    print(f"최근 {args.hours}시간 필터 결과: {len(recent_articles)}건")
    print(f"저장 결과: {len(selected_articles)}건")
    print(f"JSON 저장 위치: {output_path}")
    print(f"이미지 URL 저장 폴더: {image_dir}")

    for article in payload[:3]:
        print(json.dumps(article, ensure_ascii=False, indent=2, default=str))


def filter_by_hours(articles: list[RawArticle], hours: int) -> list[RawArticle]:
    if hours <= 0:
        return articles

    cutoff = datetime.now().astimezone() - timedelta(hours=hours)

    return [
        article
        for article in articles
        if article.published_at is not None and article.published_at >= cutoff
    ]


def write_image_url_files(articles: list[RawArticle], output_path: Path) -> Path:
    image_dir = output_path.parent / f"{output_path.stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for stale_file in image_dir.glob("*.json"):
        stale_file.unlink()

    for article in articles:
        image_urls = article.extra.get("image_urls")

        if not isinstance(image_urls, list):
            image_urls = []

        (image_dir / f"{article.id}.json").write_text(
            json.dumps(image_urls, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return image_dir


def prune_extra_fields(article: RawArticle) -> None:
    """JSON 저장 시 extra를 가볍게 정리한다.

    현재는 article_to_output_dict가 naver_crawler.py의 clean_output_extra를 사용하므로,
    이 함수는 필요 시 별도 정리용으로 사용할 수 있다.
    """
    extra = article.extra

    if not isinstance(extra, dict):
        return

    resolved = extra.get("resolved_url")

    if not resolved or str(resolved) == str(article.url or ""):
        extra.pop("resolved_url", None)

    extra.pop("feed_url", None)

    for key in ("search_query", "sector", "body_fetch_error"):
        value = extra.get(key)

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            extra.pop(key, None)

    for key in ("image_urls", "matched_aliases"):
        value = extra.get(key)

        if isinstance(value, list) and len(value) == 0:
            extra.pop(key, None)


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "crawler_results" / f"{Path(__file__).stem}.json"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


if __name__ == "__main__":
    asyncio.run(main())
