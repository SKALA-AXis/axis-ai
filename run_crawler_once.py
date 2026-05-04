"""크롤러 1회 실행 스크립트 — Track A / B 결과를 콘솔에 출력.

사용법:
  uv run python run_crawler_once.py                       # Track A, 프로세스 env
  uv run python run_crawler_once.py --track a
  uv run python run_crawler_once.py --track b
  uv run python run_crawler_once.py --track all
  uv run python run_crawler_once.py --env local           # .env.local 로드 (로컬 DB)
  uv run python run_crawler_once.py --env cloud           # .env.cloud 로드
"""

import argparse
import asyncio
import json
import logging
from collections import Counter
from datetime import datetime, time, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_crawler")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AXIS 크롤러 1회 실행")
    parser.add_argument(
        "--track",
        choices=["a", "b", "all"],
        default="a",
        help="실행할 트랙 (기본: a)",
    )
    parser.add_argument(
        "--env",
        choices=["local", "cloud"],
        default=None,
        help="DB 프로파일. .env.{profile} 파일이 있으면 로드, 없으면 프로세스 env 사용.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="DB 저장을 건너뛰고 크롤링/중복제거 결과만 반환한다.",
    )
    parser.add_argument(
        "--local-output",
        default=None,
        help="크롤 결과 JSONL을 저장할 디렉터리. 예: data/crawl_outputs",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Track B 수집 기간. --start-date가 없을 때 오늘 기준 최근 N일.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Track B 수집 시작일 YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Track B 수집 종료일 YYYY-MM-DD. 미지정 시 현재 시각.",
    )
    return parser.parse_args()


_args = _parse_args()

from src.config.env_loader import load_profile  # noqa: E402

_profile = load_profile(_args.env)
log.info("실행 프로파일: %s", _profile)

from src.crawler.base import CrawlWindow, RawArticle  # noqa: E402
from src.crawler.batch_processor import BatchProcessor  # noqa: E402
from src.crawler.parsers.quality import attach_quality  # noqa: E402
from src.crawler.scheduler import PEER_ALIASES  # noqa: E402


def _summarize(label: str, articles: list[RawArticle]) -> None:
    attach_quality(articles, PEER_ALIASES)
    print("\n" + "=" * 78)
    print(f"📡 {label} — 수집 요약")
    print("=" * 78)
    print(f"  총 신규 저장:    {len(articles)}건")
    if not articles:
        return

    by_peer: Counter[str] = Counter(a.peer_id or "unknown" for a in articles)
    by_source: Counter[str] = Counter(a.source_name for a in articles)
    usable_count = sum(1 for a in articles if a.metadata.get("quality", {}).get("is_usable"))
    issue_counts: Counter[str] = Counter(
        issue
        for a in articles
        for issue in a.metadata.get("quality", {}).get("quality_issues", [])
    )

    print("\n  ── Peer별 ──")
    for peer, n in by_peer.most_common():
        print(f"    {peer:18s} {n}건")

    print("\n  ── 소스별 ──")
    for source, n in by_source.most_common():
        print(f"    {source:24s} {n}건")

    print("\n  ── 품질 KPI ──")
    print(f"    is_usable              {usable_count}/{len(articles)}건")
    for issue, n in issue_counts.most_common(5):
        print(f"    {issue:24s} {n}건")


def _build_window() -> CrawlWindow | None:
    if not (_args.start_date or _args.end_date or _args.lookback_days):
        return None

    if _args.start_date:
        start = datetime.combine(datetime.strptime(_args.start_date, "%Y-%m-%d").date(), time.min)
    else:
        days = _args.lookback_days or 1
        start = datetime.now().replace(microsecond=0) - timedelta(days=days)

    if _args.end_date:
        end = datetime.combine(datetime.strptime(_args.end_date, "%Y-%m-%d").date(), time.max)
    else:
        end = datetime.now().replace(microsecond=0)
    return CrawlWindow(start=start, end=end)


def _save_local(label: str, articles: list[RawArticle]) -> Path | None:
    if not _args.local_output:
        return None
    attach_quality(articles, PEER_ALIASES)
    output_dir = Path(_args.local_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{label.lower()}_{ts}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(_article_to_dict(article), ensure_ascii=False) + "\n")
    print(f"\n  로컬 저장: {output_path}")
    return output_path


def _article_to_dict(article: RawArticle) -> dict:
    return {
        "url": article.url,
        "title": article.title,
        "content": article.content,
        "source_name": article.source_name,
        "peer_id": article.peer_id,
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "collected_at": article.collected_at.isoformat(),
        "url_hash": article.url_hash,
        "quality": article.metadata.get("quality", {}),
        "metadata": article.metadata,
    }


async def _run(track: str) -> None:
    processor = BatchProcessor()
    persist = not _args.skip_db
    crawl_window = _build_window()
    if track in ("a", "all"):
        log.info("Track A 시작 | peers=%s", list(PEER_ALIASES))
        articles = await processor.run_track_a(PEER_ALIASES, persist=persist)
        _summarize("Track A", articles)
        _save_local("track_a", articles)

    if track in ("b", "all"):
        log.info("Track B 시작 | peers=%s", list(PEER_ALIASES))
        articles = await processor.run_track_b(
            PEER_ALIASES,
            persist=persist,
            crawl_window=crawl_window,
        )
        _summarize("Track B", articles)
        _save_local("track_b", articles)


def main() -> None:
    asyncio.run(_run(_args.track))
    print("\n" + "=" * 78)
    print("✅ 크롤 완료. 다음 단계: uv run python run_pipeline_once.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
