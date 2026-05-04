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

from src.config.env_loader import load_profile
from src.crawler.dart_crawler import DartCrawler
from src.crawler.ir_crawler import IRCrawler
from src.crawler.job_crawler import JobCrawler
from src.crawler.parsers.link_check import LinkChecker
from src.crawler.research_crawler import NaverResearchCrawler
from src.crawler.result_writer import save_crawler_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_local_crawler")

PEER_ALIASES = {
    "samsung_sds": ["삼성SDS", "Samsung SDS", "삼성에스디에스"],
    "lg_cns": ["LG CNS", "엘지씨엔에스", "LGCNS"],
    "hyundai_autoever": ["현대오토에버", "Hyundai AutoEver", "현대오토에버시스템"],
    "posco_dx": ["포스코DX", "포스코디엑스", "POSCO DX"],
    "sk_ax": [
        "SK AX",
        "SK C&C",
        "SK주식회사 C&C",
        "에스케이씨앤씨",
        "SK Inc.",
        "SK주식회사",
        "SK",
    ],
}

CORP_CODES = {
    "sk_ax": "00111722",
    "samsung_sds": "00126186",
    "lg_cns": "00139834",
    "hyundai_autoever": "00362441",
    "posco_dx": "00155212",
}

PEER_SOURCES = [
    "dart",
    "jobs",
    "ir",
    "naver_research",
]

ALL_SOURCES = PEER_SOURCES


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


def _make_source_name(source: str, peer: str) -> str:
    return f"{source}_{peer}"


async def _run_one(source: str, peer: str) -> None:
    crawler = _build_crawler(source, peer)
    articles = await crawler.crawl()
    articles, rejected = await LinkChecker().filter_accessible(articles)

    output_path = save_crawler_results(
        articles,
        source_name=_make_source_name(source, peer),
        peer_aliases=PEER_ALIASES,
    )

    log.info(
        "수집 완료 | source=%s peer=%s valid=%d rejected=%d output=%s",
        source,
        peer or "industry",
        len(articles),
        len(rejected),
        output_path,
    )


def _build_run_plan(source: str | None, peer: str | None) -> list[tuple[str, str | None]]:
    run_plan: list[tuple[str, str | None]] = []

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

    return run_plan


async def _run() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    log.info("실행 프로파일: %s", profile)

    run_plan = _build_run_plan(args.source, args.peer)

    log.info("실행 대상 수: %d", len(run_plan))

    for source, peer in run_plan:
        if peer is None:
            continue
        try:
            await _run_one(source, peer)
        except Exception as e:
            log.error(
                "크롤러 실행 실패 | source=%s peer=%s error=%s",
                source,
                peer,
                e,
            )


if __name__ == "__main__":
    asyncio.run(_run())
