"""단독 또는 전체 크롤러 실행 후 src/crawler/crawler_results 에 JSON 저장.

전체 수집
uv run python run_local_crawler_once.py

수집 데이터 유형 지정
uv run python run_local_crawler_once.py --source ir --company samsung_sds
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from src.config.companies import (
    CATCH_ANALYSIS_IDS,
    COMPANY_ALIASES,
    CORP_CODES,
    company_name_ko,
)
from src.config.env_loader import load_profile
from src.config.global_companies import GLOBAL_COMPANY_ALIASES, GLOBAL_COMPANY_IDS
from src.crawler.base import DailyLimitGuard
from src.crawler.local.bcg_crawler import BcgCrawler
from src.crawler.local.catch_crawler import CatchCompanyAnalysisCrawler
from src.crawler.local.company_news_crawler import CompanyNewsCrawler
from src.crawler.local.dart_crawler import DartCrawler
from src.crawler.local.ir_crawler import IRCrawler
from src.crawler.local.job_crawler import JobCrawler
from src.crawler.local.keyword_crawler import KeywordCrawler, save_trend_chart
from src.crawler.local.naver_crawler import NaverNewsCrawler, annotate_peer_relevance
from src.crawler.local.research_crawler import NaverResearchCrawler
from src.crawler.local.spri_crawler import SpriCrawler
from src.crawler.local.stock_crawler import StockCrawler
from src.crawler.parsers.link_check import LinkChecker
from src.crawler.result_writer import DEFAULT_RESULTS_DIR, save_crawler_results
from src.crawler.sources.global_newsroom import GlobalNewsroomCrawler
from src.db.article_store import save_articles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_local_crawler")

DOMESTIC_COMPANY_SEARCH_ALIASES = dict(COMPANY_ALIASES)
GLOBAL_COMPANY_SEARCH_ALIASES = dict(GLOBAL_COMPANY_ALIASES)
COMPANY_SEARCH_ALIASES = {
    **DOMESTIC_COMPANY_SEARCH_ALIASES,
    **GLOBAL_COMPANY_SEARCH_ALIASES,
}

COMPANY_SOURCES = [
    "naver_news",
    "dart",
    "jobs",
    "ir",
    "naver_research",
    "stock",
]
OPTIONAL_COMPANY_SOURCES = ["catch"]
INDUSTRY_SOURCES = [
    "naver_datalab",
    "company_news",
    "global_newsroom",
    "bcg",
    "spri",
]

ALL_SOURCES = COMPANY_SOURCES + OPTIONAL_COMPANY_SOURCES + INDUSTRY_SOURCES
SOURCE_ALIASES = {
    "official": "company_news",
    "job": "jobs",
    "rss": "global_newsroom",
}
MERGED_OUTPUT_SOURCES = {
    "dart",
    "ir",
    "jobs",
    "naver_news",
    "naver_datalab",
    "naver_research",
    "global_newsroom",
    "stock",
}
RETRY_ON_EMPTY_SOURCES = {"dart", "ir", "naver_datalab", "naver_research"}
MAX_CRAWL_ATTEMPTS = 2
DEFAULT_NEWS_LIMIT = DailyLimitGuard.SOURCE_TYPE_LIMITS["news"]


def _previous_month_label() -> str:
    today = datetime.now().date()
    year = today.year
    month = today.month - 1

    if month == 0:
        year -= 1
        month = 12

    return f"{year:04d}-{month:02d}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="개별 또는 전체 크롤러 로컬 실행")
    parser.add_argument(
        "--source",
        action="append",
        nargs="+",
        default=None,
        help=(
            "실행할 크롤러. 쉼표 구분/공백 포함 쉼표 구분/반복 지정 가능. "
            "예: --source naver_news,global_newsroom,naver_research 또는 "
            "--source naver_news, global_newsroom"
        ),
    )
    parser.add_argument(
        "--company",
        "--peer",
        dest="company",
        choices=list(COMPANY_SEARCH_ALIASES),
        default=None,
        help="수집할 회사 id. 생략하면 전체 회사 실행. --peer는 하위 호환 alias.",
    )
    parser.add_argument(
        "--company-tier",
        "--company_tier",
        dest="company_tier",
        choices=["domestic", "overseas", "self"],
        default=None,
        help=(
            "수집할 회사 tier. overseas는 global_newsroom, domestic은 국내 peer source를 실행한다. "
            "예: --source official --company-tier overseas"
        ),
    )
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument(
        "--month",
        default=None,
        help="SPRi 월호. 예: 2026-04. 생략하면 직전 월을 사용한다.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="BCG 최근 N일 수집 범위. 기본 7.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=100,
        help="BCG 최대 상세 글 수. 기본 100.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_NEWS_LIMIT,
        help=(
            f"네이버 뉴스 검색어별 최대 수집 결과 수. 기본 base.py news 한도({DEFAULT_NEWS_LIMIT})."
        ),
    )
    parser.add_argument(
        "--max-rss-entries",
        type=int,
        default=DEFAULT_NEWS_LIMIT,
        help=("하위 호환용 인자. rss 크롤러 제거 후 global_newsroom에서는 사용하지 않음."),
    )
    parser.add_argument(
        "--rss-delay",
        type=float,
        default=1.0,
        help="하위 호환용 인자. rss 크롤러 제거 후 global_newsroom에서는 사용하지 않음.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="naver_news 최근 N시간 수집 범위. 0 이하이면 시간 필터를 끈다. 기본 24.",
    )
    parser.add_argument(
        "--latest-limit",
        type=int,
        default=5,
        help="회사 공식 뉴스 회사별 최신 수집 개수. 기본 5.",
    )
    parser.add_argument(
        "--stock-days",
        type=int,
        default=30,
        help="주가 OHLCV 최근 N일 수집 범위. 기본 30.",
    )
    parser.add_argument(
        "--no-stock-realtime",
        action="store_true",
        help="주가 크롤링에서 실시간 quote polling을 끈다.",
    )
    parser.add_argument(
        "--no-body",
        action="store_true",
        help="naver_news 상세 본문 추가 수집을 끈다",
    )
    parser.add_argument(
        "--link-check-concurrency",
        type=int,
        default=6,
        help="저장 전 URL 접근성 검사 동시성. 429가 뜨면 낮춘다. 기본 6.",
    )
    parser.add_argument(
        "--catch-id",
        default=None,
        help="Catch 기업분석 ID. 예: 삼성SDS 리포트 URL의 ID=3393",
    )
    parser.add_argument(
        "--catch-session",
        default=None,
        help="Catch storage_state JSON 경로. 기본 axis-ai/.secrets/catch_storage_state.json",
    )
    parser.add_argument(
        "--catch-headed",
        action="store_true",
        help="Catch 크롤링을 headed 브라우저로 실행한다.",
    )
    parser.add_argument(
        "--discover-latest",
        action="store_true",
        help="Catch 검색/기업요약 페이지에서 최신 분석리포트 ID를 먼저 탐색한다.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="로컬 JSON 저장에 더해 raw_articles DB에도 저장한다.",
    )
    return parser.parse_args()


def _build_crawler(source: str, company: str | None, args: argparse.Namespace) -> Any:
    if source == "naver_datalab":
        return KeywordCrawler()

    if source == "company_news":
        return CompanyNewsCrawler(latest_limit=args.latest_limit)

    if source == "global_newsroom":
        return GlobalNewsroomCrawler(
            company=company if company in GLOBAL_COMPANY_IDS else None,
            max_pages=args.latest_limit,
        )

    if source == "bcg":
        return BcgCrawler(
            days=args.days,
            max_articles=args.max_articles,
            output_path=DEFAULT_RESULTS_DIR / "bcg_crawler.json",
        )

    if source == "spri":
        return SpriCrawler(
            month=args.month or _previous_month_label(),
            output_path=DEFAULT_RESULTS_DIR / "spri_crawler.json",
        )

    if source == "naver_news":
        if company is None:
            raise ValueError("naver_news 크롤러는 company가 필요합니다.")
        cutoff_datetime = (
            datetime.now().astimezone() - timedelta(hours=args.hours) if args.hours > 0 else None
        )
        return NaverNewsCrawler(
            peer_id=company,
            aliases=DOMESTIC_COMPANY_SEARCH_ALIASES[company],
            max_results=args.max_results,
            cutoff_datetime=cutoff_datetime,
            fetch_body=not args.no_body,
        )

    if source == "dart":
        if company is None:
            raise ValueError("dart 크롤러는 company가 필요합니다.")
        return DartCrawler(
            peer_id=company,
            corp_code=CORP_CODES.get(company),
            corp_names=DOMESTIC_COMPANY_SEARCH_ALIASES[company],
        )

    if source == "jobs":
        if company is None:
            raise ValueError("jobs 크롤러는 company가 필요합니다.")
        return JobCrawler(peer_id=company)

    if source == "ir":
        if company is None:
            raise ValueError("ir 크롤러는 company가 필요합니다.")
        return IRCrawler(peer_id=company)

    if source == "naver_research":
        if company is None:
            raise ValueError("naver_research 크롤러는 company가 필요합니다.")
        return NaverResearchCrawler(peer_id=company)

    if source == "stock":
        if company is None:
            raise ValueError("stock 크롤러는 company가 필요합니다.")
        return StockCrawler(
            peer_id=company,
            lookback_days=args.stock_days,
            include_realtime=not args.no_stock_realtime,
        )

    if source == "catch":
        if company is None:
            raise ValueError("catch 크롤러는 company가 필요합니다.")
        catch_id = args.catch_id or CATCH_ANALYSIS_IDS.get(company)
        if not catch_id:
            raise ValueError(
                f"Catch analysis id가 없습니다: company={company}. --catch-id를 지정하세요."
            )
        return CatchCompanyAnalysisCrawler(
            company=company,
            catch_id=catch_id,
            company_query=company_name_ko(company),
            **({"session_path": args.catch_session} if args.catch_session else {}),
            headless=not args.catch_headed,
            discover_latest=args.discover_latest,
        )

    raise ValueError(f"지원하지 않는 source입니다: {source}")


def _make_output_source_name(source: str, company: str | None) -> str:
    if company is not None:
        return f"{source}_{company}"
    return source


def _make_output_group(source: str, company: str) -> tuple[str, str | None]:
    if source in MERGED_OUTPUT_SOURCES:
        return source, None
    return source, company


async def _run_one(
    source: str,
    company: str | None,
    args: argparse.Namespace,
    limit_guard: DailyLimitGuard,
) -> tuple[list[Any], int]:
    crawler = _build_crawler(source, company, args)
    articles = await crawler.crawl()
    articles = _filter_peer_news_articles(source, company, articles)
    articles, limited_count = _apply_daily_limit(articles, limit_guard)
    articles, rejected = await LinkChecker(
        concurrency=max(1, args.link_check_concurrency),
    ).filter_accessible(articles)

    log.info(
        "수집 완료 | source=%s company=%s valid=%d rejected=%d limited=%d",
        source,
        company or "industry",
        len(articles),
        len(rejected),
        limited_count,
    )
    return articles, len(rejected)


def _filter_peer_news_articles(
    source: str,
    company: str | None,
    articles: list[Any],
) -> list[Any]:
    if source not in {"naver_news"} or company is None:
        return articles

    before = len(articles)
    annotate_peer_relevance(
        articles,
        target_peer_id=company,
        tracked_peer_ids=list(DOMESTIC_COMPANY_SEARCH_ALIASES),
    )
    filtered = [
        article
        for article in articles
        if getattr(article, "extra", {}).get("peer_relevance") == "pass"
        and getattr(article, "company", [])
    ]

    log.info(
        "피어 관련 뉴스 필터 완료 | source=%s company=%s before=%d after=%d",
        source,
        company,
        before,
        len(filtered),
    )
    return filtered


async def _run_one_with_retry(
    source: str,
    company: str | None,
    args: argparse.Namespace,
    limit_guard: DailyLimitGuard,
) -> tuple[list[Any], int]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_CRAWL_ATTEMPTS + 1):
        try:
            articles, rejected_count = await _run_one(source, company, args, limit_guard)
        except Exception as e:
            last_error = e
            if attempt < MAX_CRAWL_ATTEMPTS:
                log.warning(
                    "크롤러 실행 실패, 재시도 예정 | source=%s company=%s attempt=%d/%d error=%s",
                    source,
                    company,
                    attempt,
                    MAX_CRAWL_ATTEMPTS,
                    e,
                )
            continue

        if articles or source not in RETRY_ON_EMPTY_SOURCES:
            return articles, rejected_count

        if attempt < MAX_CRAWL_ATTEMPTS:
            log.warning(
                "수집 결과 0건, 재시도 예정 | source=%s company=%s attempt=%d/%d",
                source,
                company,
                attempt,
                MAX_CRAWL_ATTEMPTS,
            )

    if last_error is not None:
        raise last_error

    return [], 0


def _article_company_ids(articles: list[Any]) -> set[str]:
    company_ids: set[str] = set()

    for article in articles:
        companies = getattr(article, "company", None)
        if companies:
            company_ids.update(str(company) for company in companies)
            continue

        peer_id = getattr(article, "peer_id", None)
        if peer_id:
            company_ids.add(str(peer_id))

    return company_ids


def _datalab_rows(articles: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for article in articles:
        extra = getattr(article, "extra", None)
        if isinstance(extra, dict):
            rows.append(extra)

    return rows


def _apply_daily_limit(
    articles: list[Any],
    limit_guard: DailyLimitGuard,
) -> tuple[list[Any], int]:
    limited_articles: list[Any] = []
    limited_count = 0

    for article in articles:
        source_type = str(getattr(article, "source_type", "news") or "news")

        if limit_guard.allow(source_type):
            limited_articles.append(article)
        else:
            limited_count += 1

    return limited_articles, limited_count


def _parse_sources(source_values: list[list[str]] | None) -> list[str] | None:
    if not source_values:
        return None

    sources: list[str] = []
    invalid: list[str] = []

    for source_group in source_values:
        for value in source_group:
            value = value.strip()
            if value == ",":
                continue
            for source in value.split(","):
                source = source.strip()
                if not source:
                    continue
                source = SOURCE_ALIASES.get(source, source)
                if source not in ALL_SOURCES:
                    invalid.append(source)
                    continue
                if source not in sources:
                    sources.append(source)

    if invalid:
        raise SystemExit(
            "알 수 없는 source: "
            + ", ".join(sorted(set(invalid)))
            + f" | available={', '.join(ALL_SOURCES)}"
        )

    return sources or None


def _resolve_sources_for_company_tier(
    sources: list[str] | None,
    company_tier: str | None,
) -> list[str] | None:
    if company_tier != "overseas" or not sources:
        return sources

    resolved: list[str] = []
    for source in sources:
        # official은 국내 company_news와 해외 global_newsroom을 모두 뜻할 수 있다.
        # overseas tier가 명시되면 해외 공식 뉴스룸으로 해석한다.
        source = "global_newsroom" if source == "company_news" else source
        if source not in resolved:
            resolved.append(source)
    return resolved


def _build_run_plan(
    sources: list[str] | None,
    company: str | None,
    company_tier: str | None = None,
) -> list[tuple[str, str | None]]:
    run_plan: list[tuple[str, str | None]] = []
    domestic_company_ids = list(DOMESTIC_COMPANY_SEARCH_ALIASES)
    global_company_ids = list(GLOBAL_COMPANY_IDS)

    if company_tier and company:
        valid_for_tier = (
            company in DOMESTIC_COMPANY_SEARCH_ALIASES
            if company_tier == "domestic"
            else company in GLOBAL_COMPANY_IDS
            if company_tier == "overseas"
            else company == "sk_ax"
        )
        if not valid_for_tier:
            raise SystemExit(
                f"company와 company-tier가 맞지 않습니다: company={company} tier={company_tier}"
            )

    if sources:
        for source in sources:
            if source == "global_newsroom":
                if company in GLOBAL_COMPANY_IDS:
                    run_plan.append((source, company))
                elif company_tier == "overseas":
                    run_plan.extend((source, company_id) for company_id in global_company_ids)
                elif company is None:
                    run_plan.append((source, None))
            elif source in INDUSTRY_SOURCES:
                if company_tier == "overseas" and source == "company_news":
                    run_plan.extend(
                        ("global_newsroom", company_id) for company_id in global_company_ids
                    )
                    continue
                if company_tier == "overseas":
                    continue
                run_plan.append((source, None))
            elif source == "catch":
                if company:
                    run_plan.append((source, company))
                else:
                    run_plan.extend((source, company_id) for company_id in CATCH_ANALYSIS_IDS)
            elif company:
                if company in DOMESTIC_COMPANY_SEARCH_ALIASES:
                    run_plan.append((source, company))
            elif company_tier == "domestic" or company_tier is None:
                run_plan.extend((source, company_id) for company_id in domestic_company_ids)
            else:
                continue

        return run_plan

    if company:
        if company in DOMESTIC_COMPANY_SEARCH_ALIASES:
            for company_source in COMPANY_SOURCES:
                run_plan.append((company_source, company))
        if company in GLOBAL_COMPANY_IDS:
            run_plan.append(("global_newsroom", company))

        return run_plan

    if company_tier == "overseas":
        run_plan.extend(("global_newsroom", company_id) for company_id in global_company_ids)
        return run_plan

    if company_tier == "domestic":
        for company_source in COMPANY_SOURCES:
            for company_id in domestic_company_ids:
                run_plan.append((company_source, company_id))
        run_plan.append(("company_news", None))
        return run_plan

    if company_tier == "self":
        log.warning("self tier는 현재 로컬 크롤러 실행 대상이 없습니다 | company=sk_ax")
        return run_plan

    for company_source in COMPANY_SOURCES:
        for company_id in domestic_company_ids:
            run_plan.append((company_source, company_id))

    for industry_source in INDUSTRY_SOURCES:
        run_plan.append((industry_source, None))

    return run_plan


async def _run() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    log.info("실행 프로파일: %s", profile)

    sources = _resolve_sources_for_company_tier(
        _parse_sources(args.source),
        args.company_tier,
    )
    run_plan = _build_run_plan(sources, args.company, args.company_tier)

    log.info("실행 대상 수: %d", len(run_plan))

    articles_by_output: dict[tuple[str, str | None], list[Any]] = {}
    companies_by_output: dict[tuple[str, str | None], set[str]] = {}
    rejected_by_output: dict[tuple[str, str | None], int] = {}
    limit_guard = DailyLimitGuard()

    for source, company in run_plan:
        if company is None and source not in INDUSTRY_SOURCES:
            continue
        try:
            articles, rejected_count = await _run_one_with_retry(
                source,
                company,
                args,
                limit_guard,
            )
            output_group = _make_output_group(source, company or "all")
            articles_by_output.setdefault(output_group, []).extend(articles)
            if company is not None:
                companies_by_output.setdefault(output_group, set()).add(company)
            else:
                companies_by_output.setdefault(output_group, set())
            rejected_by_output[output_group] = (
                rejected_by_output.get(output_group, 0) + rejected_count
            )
        except Exception as e:
            log.error(
                "크롤러 실행 실패 | source=%s company=%s error=%s",
                source,
                company,
                e,
            )

    for (source, output_company), articles in articles_by_output.items():
        output_source_name = _make_output_source_name(source, output_company)
        missing_companies = companies_by_output[(source, output_company)] - _article_company_ids(
            articles
        )
        if missing_companies and source in RETRY_ON_EMPTY_SOURCES:
            log.error(
                "저장 결과에 일부 회사 데이터가 없습니다 | source=%s missing_companies=%s",
                source,
                sorted(missing_companies),
            )

        output_path = save_crawler_results(
            articles,
            source_name=output_source_name,
        )

        log.info(
            "저장 완료 | source=%s companies=%d valid=%d rejected=%d output=%s",
            source,
            len(companies_by_output[(source, output_company)]),
            len(articles),
            rejected_by_output.get((source, output_company), 0),
            output_path,
        )

        if args.persist:
            inserted = save_articles(articles)
            log.info("DB 저장 완료 | source=%s inserted=%d", source, inserted)

        if source == "naver_datalab":
            chart_path = save_trend_chart(_datalab_rows(articles))
            if chart_path is not None:
                log.info("차트 저장 완료 | source=%s output=%s", source, chart_path)


if __name__ == "__main__":
    asyncio.run(_run())
