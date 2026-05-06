""" 단독 또는 전체 크롤러 실행 후 src/crawler/crawler_results 에 JSONL 저장.

    전체 수집
    uv run python run_local_crawler_once.py

    수집 데이터 유형 지정
    uv run python run_local_crawler_once.py --source ir --peer samsung_sds
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from src.config.companies import COMPANY_ALIASES, CORP_CODES
from src.config.env_loader import load_profile
from src.crawler.dart_crawler import DartCrawler
from src.crawler.ir_crawler import IRCrawler
from src.crawler.job_crawler import JobCrawler
from src.crawler.keyword_crawler import KeywordCrawler, save_trend_chart
from src.crawler.parsers.link_check import LinkChecker
from src.crawler.research_crawler import NaverResearchCrawler
from src.crawler.result_writer import save_crawler_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_local_crawler")

PEER_ALIASES = dict(COMPANY_ALIASES)

PEER_SOURCES = [
    "dart",
    "jobs",
    "ir",
    "naver_research",
]
INDUSTRY_SOURCES = [
    "naver_datalab",
]

ALL_SOURCES = PEER_SOURCES + INDUSTRY_SOURCES
MERGED_OUTPUT_SOURCES = {"dart", "ir", "jobs", "naver_datalab", "naver_research"}
RETRY_ON_EMPTY_SOURCES = {"dart", "ir", "naver_datalab", "naver_research"}
MAX_CRAWL_ATTEMPTS = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="개별 또는 전체 크롤러 로컬 실행")
    parser.add_argument(
        "--source",
        choices=ALL_SOURCES,
        default=None,
        help="실행할 크롤러. 생략하면 전체 source 실행",
    )
    parser.add_argument(
        "--peer",
        choices=list(PEER_ALIASES),
        default=None,
        help="수집할 Peer사. 생략하면 전체 Peer사 실행",
    )
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    return parser.parse_args()


def _build_crawler(source: str, peer: str | None) -> Any:
    if source == "naver_datalab":
        return KeywordCrawler()

    if source == "dart":
        if peer is None:
            raise ValueError("dart 크롤러는 peer가 필요합니다.")
        return DartCrawler(
            peer_id=peer,
            corp_code=CORP_CODES.get(peer),
            corp_names=PEER_ALIASES[peer],
        )

    if source == "jobs":
        if peer is None:
            raise ValueError("jobs 크롤러는 peer가 필요합니다.")
        return JobCrawler(peer_id=peer)

    if source == "ir":
        if peer is None:
            raise ValueError("ir 크롤러는 peer가 필요합니다.")
        return IRCrawler(peer_id=peer)

    if source == "naver_research":
        if peer is None:
            raise ValueError("naver_research 크롤러는 peer가 필요합니다.")
        return NaverResearchCrawler(peer_id=peer)

    raise ValueError(f"지원하지 않는 source입니다: {source}")


def _make_output_source_name(source: str, peer: str | None) -> str:
    if peer is not None:
        return f"{source}_{peer}"
    return source


def _make_output_group(source: str, peer: str) -> tuple[str, str | None]:
    if source in MERGED_OUTPUT_SOURCES:
        return source, None
    return source, peer


async def _run_one(source: str, peer: str | None) -> tuple[list[Any], int]:
    crawler = _build_crawler(source, peer)
    articles = await crawler.crawl()
    articles, rejected = await LinkChecker().filter_accessible(articles)

    log.info(
        "수집 완료 | source=%s peer=%s valid=%d rejected=%d",
        source,
        peer or "industry",
        len(articles),
        len(rejected),
    )
    return articles, len(rejected)


async def _run_one_with_retry(source: str, peer: str | None) -> tuple[list[Any], int]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_CRAWL_ATTEMPTS + 1):
        try:
            articles, rejected_count = await _run_one(source, peer)
        except Exception as e:
            last_error = e
            if attempt < MAX_CRAWL_ATTEMPTS:
                log.warning(
                    "크롤러 실행 실패, 재시도 예정 | source=%s peer=%s attempt=%d/%d error=%s",
                    source,
                    peer,
                    attempt,
                    MAX_CRAWL_ATTEMPTS,
                    e,
                )
            continue

        if articles or source not in RETRY_ON_EMPTY_SOURCES:
            return articles, rejected_count

        if attempt < MAX_CRAWL_ATTEMPTS:
            log.warning(
                "수집 결과 0건, 재시도 예정 | source=%s peer=%s attempt=%d/%d",
                source,
                peer,
                attempt,
                MAX_CRAWL_ATTEMPTS,
            )

    if last_error is not None:
        raise last_error

    return [], 0


def _article_peer_ids(articles: list[Any]) -> set[str]:
    peer_ids: set[str] = set()

    for article in articles:
        companies = getattr(article, "company", None)
        if companies:
            peer_ids.update(str(company) for company in companies)
            continue

        peer_id = getattr(article, "peer_id", None)
        if peer_id:
            peer_ids.add(str(peer_id))

    return peer_ids


def _datalab_rows(articles: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for article in articles:
        extra = getattr(article, "extra", None)
        if isinstance(extra, dict):
            rows.append(extra)

    return rows


def _build_run_plan(source: str | None, peer: str | None) -> list[tuple[str, str | None]]:
    run_plan: list[tuple[str, str | None]] = []

    if source in INDUSTRY_SOURCES:
        return [(source, None)]

    if source and peer:
        return [(source, peer)]

    if source:
        return [(source, peer_id) for peer_id in PEER_ALIASES]

    if peer:
        for peer_source in PEER_SOURCES:
            run_plan.append((peer_source, peer))

        return run_plan

    for peer_id in PEER_ALIASES:
        for peer_source in PEER_SOURCES:
            run_plan.append((peer_source, peer_id))

    for industry_source in INDUSTRY_SOURCES:
        run_plan.append((industry_source, None))

    return run_plan


async def _run() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    log.info("실행 프로파일: %s", profile)

    run_plan = _build_run_plan(args.source, args.peer)

    log.info("실행 대상 수: %d", len(run_plan))

    articles_by_output: dict[tuple[str, str | None], list[Any]] = {}
    peers_by_output: dict[tuple[str, str | None], set[str]] = {}
    rejected_by_output: dict[tuple[str, str | None], int] = {}

    for source, peer in run_plan:
        if peer is None and source not in INDUSTRY_SOURCES:
            continue
        try:
            articles, rejected_count = await _run_one_with_retry(source, peer)
            output_group = _make_output_group(source, peer or "all")
            articles_by_output.setdefault(output_group, []).extend(articles)
            if peer is not None:
                peers_by_output.setdefault(output_group, set()).add(peer)
            else:
                peers_by_output.setdefault(output_group, set())
            rejected_by_output[output_group] = (
                rejected_by_output.get(output_group, 0) + rejected_count
            )
        except Exception as e:
            log.error(
                "크롤러 실행 실패 | source=%s peer=%s error=%s",
                source,
                peer,
                e,
            )

    for (source, output_peer), articles in articles_by_output.items():
        output_source_name = _make_output_source_name(source, output_peer)
        missing_peers = peers_by_output[(source, output_peer)] - _article_peer_ids(articles)
        if missing_peers and source in RETRY_ON_EMPTY_SOURCES:
            log.error(
                "저장 결과에 일부 peer 데이터가 없습니다 | source=%s missing_peers=%s",
                source,
                sorted(missing_peers),
            )

        output_path = save_crawler_results(
            articles,
            source_name=output_source_name,
            peer_aliases=PEER_ALIASES,
        )

        log.info(
            "저장 완료 | source=%s peers=%d valid=%d rejected=%d output=%s",
            source,
            len(peers_by_output[(source, output_peer)]),
            len(articles),
            rejected_by_output.get((source, output_peer), 0),
            output_path,
        )

        if source == "naver_datalab":
            chart_path = save_trend_chart(_datalab_rows(articles))
            if chart_path is not None:
                log.info("차트 저장 완료 | source=%s output=%s", source, chart_path)


if __name__ == "__main__":
    asyncio.run(_run())
