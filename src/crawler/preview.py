"""로컬 크롤링 결과를 JSON 파일로 저장하는 미리보기 실행기."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.agents.crawler_agent import PEER_PROFILES, CrawlerAgent
from src.crawler.article_filter import is_duplicate_title, similarity_key
from src.crawler.base_crawler import RawArticle

DEFAULT_OUTPUT_DIR = Path("crawl_results")
PREVIEW_MODES = {"raw", "agent", "both"}


def article_to_dict(article: RawArticle) -> dict[str, str | None]:
    data = asdict(article)
    published_at = data["published_at"]
    data["published_at"] = published_at.isoformat() if published_at else None
    return data


async def run_crawl_preview(
    peer_id: str,
    topics: list[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    corp_code: str | None = None,
    recent_days: int = 1,
    mode: str = "raw",
) -> list[Path]:
    if mode not in PREVIEW_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(PREVIEW_MODES))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _delete_previous_results(output_dir=output_dir, peer_id=peer_id)

    agent = CrawlerAgent(
        peer_id=peer_id,
        topics=topics,
        corp_code=corp_code,
        recent_days=recent_days,
    )
    raw_articles = await agent.collect()
    raw_articles = _deduplicate_near_identical_titles(raw_articles)
    filtered_articles = agent._filter_articles(raw_articles)

    created_at = datetime.now().astimezone()
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    output_paths = []

    if mode in ("raw", "both"):
        output_paths.append(
            _write_payload(
                output_dir=output_dir,
                peer_id=peer_id,
                timestamp=timestamp,
                result_type="raw",
                agent=agent,
                articles=raw_articles,
                created_at=created_at,
            )
        )

    if mode in ("agent", "both"):
        output_paths.append(
            _write_payload(
                output_dir=output_dir,
                peer_id=peer_id,
                timestamp=timestamp,
                result_type="agent",
                agent=agent,
                articles=filtered_articles,
                created_at=created_at,
            )
        )

    return output_paths


def _write_payload(
    output_dir: Path,
    peer_id: str,
    timestamp: str,
    result_type: str,
    agent: CrawlerAgent,
    articles: list[RawArticle],
    created_at: datetime,
) -> Path:
    payload = {
        "peer_id": peer_id,
        "result_type": result_type,
        "aliases": agent.aliases,
        "topics": agent.topics,
        "recent_days": agent.recent_days,
        "created_at": created_at.isoformat(),
        "count": len(articles),
        "articles": [article_to_dict(article) for article in articles],
    }

    suffix = "" if result_type == "raw" else "_agent"
    output_path = output_dir / f"{peer_id}{suffix}_{timestamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _deduplicate_near_identical_titles(articles: list[RawArticle]) -> list[RawArticle]:
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    unique: list[RawArticle] = []

    for article in articles:
        if article.url in seen_urls:
            continue
        if is_duplicate_title(article.title, seen_titles):
            continue
        seen_urls.add(article.url)
        seen_titles.append(similarity_key(article.title, ""))
        unique.append(article)

    return unique


def _delete_previous_results(output_dir: Path, peer_id: str) -> None:
    for path in output_dir.glob(f"{peer_id}_*.json"):
        path.unlink()


async def run_all_crawl_previews(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    recent_days: int = 1,
    mode: str = "raw",
) -> list[Path]:
    output_paths = []
    for peer_id, profile in PEER_PROFILES.items():
        output_paths.extend(
            await run_crawl_preview(
                peer_id=peer_id,
                topics=profile["topics"],
                output_dir=output_dir,
                recent_days=recent_days,
                mode=mode,
            )
        )
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="현재 크롤러 결과를 JSON 파일로 저장합니다.")
    parser.add_argument(
        "--peer-id",
        default=None,
        help="크롤링 대상 peer ID. 생략하면 기본 피어사 전체 실행",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="관심 주제. 여러 번 지정 가능",
    )
    parser.add_argument(
        "--corp-code",
        default=None,
        help="DART corp_code. 없으면 DART는 실행하지 않음",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="결과 저장 폴더")
    parser.add_argument("--days", type=int, default=1, help="최근 며칠 기사까지 저장할지")
    parser.add_argument(
        "--mode",
        choices=sorted(PREVIEW_MODES),
        default="raw",
        help="raw=크롤링 원천 결과, agent=agent 선별 결과, both=둘 다 저장",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()
    if args.peer_id:
        output_paths = asyncio.run(
            run_crawl_preview(
                peer_id=args.peer_id,
                topics=args.topics,
                output_dir=Path(args.output_dir),
                corp_code=args.corp_code,
                recent_days=args.days,
                mode=args.mode,
            )
        )
    else:
        output_paths = asyncio.run(
            run_all_crawl_previews(
                output_dir=Path(args.output_dir),
                recent_days=args.days,
                mode=args.mode,
            )
        )

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
