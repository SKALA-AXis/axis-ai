"""크롤러 1회 실행 스크립트 — Track A / B 결과를 콘솔에 출력.

사용법:
  uv run python run_crawler_once.py                       # Track A, 프로세스 env
  uv run python run_crawler_once.py --track a
  uv run python run_crawler_once.py --track b
  uv run python run_crawler_once.py --track all
  uv run python run_crawler_once.py --env local           # .env.local 로드 (로컬 DB)
  uv run python run_crawler_once.py --env cloud           # .env.cloud 로드 (Supabase + Qdrant Cloud)
"""

import argparse
import asyncio
import logging
from collections import Counter

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
    return parser.parse_args()


_args = _parse_args()

from src.config.env_loader import load_profile  # noqa: E402

_profile = load_profile(_args.env)
log.info("실행 프로파일: %s", _profile)

from src.crawler.base import RawArticle  # noqa: E402
from src.crawler.batch_processor import BatchProcessor  # noqa: E402
from src.crawler.scheduler import PEER_KEYWORDS  # noqa: E402


def _summarize(label: str, articles: list[RawArticle]) -> None:
    print("\n" + "=" * 78)
    print(f"📡 {label} — 수집 요약")
    print("=" * 78)
    print(f"  총 신규 저장:    {len(articles)}건")
    if not articles:
        return

    by_peer: Counter[str] = Counter(a.peer_id or "unknown" for a in articles)
    by_source: Counter[str] = Counter(a.source_name for a in articles)

    print("\n  ── Peer별 ──")
    for peer, n in by_peer.most_common():
        print(f"    {peer:18s} {n}건")

    print("\n  ── 소스별 ──")
    for source, n in by_source.most_common():
        print(f"    {source:24s} {n}건")


async def _run(track: str) -> None:
    processor = BatchProcessor()
    if track in ("a", "all"):
        log.info("Track A 시작 | peers=%s", list(PEER_KEYWORDS))
        articles = await processor.run_track_a(PEER_KEYWORDS)
        _summarize("Track A", articles)

    if track in ("b", "all"):
        log.info("Track B 시작 | peers=%s", list(PEER_KEYWORDS))
        articles = await processor.run_track_b(PEER_KEYWORDS)
        _summarize("Track B", articles)


def main() -> None:
    asyncio.run(_run(_args.track))
    print("\n" + "=" * 78)
    print("✅ 크롤 완료. 다음 단계: uv run python run_pipeline_once.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
