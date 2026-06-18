# 작성일: 2026-06-12
# 작성자: 안가은
# 변경이력:
#   2026-06-12 안가은 — 대시보드 키워드 트렌드 파이프라인 갱신과 함께 추가
"""Generate and save one Today's Insight report.

Usage:
  uv run python scripts/generate_today_insight_once.py --env local --anchor-date 2026-06-12

  # If DATABASE_URL is already exported, keep it and only load other env values.
  DATABASE_URL=postgresql://axuser:axpass@localhost:15432/axis \
    uv run python scripts/generate_today_insight_once.py --env cloud --anchor-date 2026-06-12
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("generate_today_insight_once")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Today's Insight report and save it.")
    parser.add_argument("--env", choices=["local", "cloud"], default="local")
    parser.add_argument(
        "--anchor-date",
        type=date.fromisoformat,
        default=None,
        help="기준일(YYYY-MM-DD). 비우면 Asia/Seoul 오늘.",
    )
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--max-issues", type=int, default=8)
    parser.add_argument("--max-cards", type=int, default=12)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="같은 기준일의 기존 저장 리포트를 삭제한 뒤 새로 저장.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="DB 저장 없이 생성 결과만 확인.",
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="기존 저장 결과가 있으면 재사용. 기본은 캐시 무시 후 새 생성.",
    )
    return parser.parse_args()


def _load_env(profile: str) -> None:
    from src.config.env_loader import load_profile

    preserved_database_url = os.environ.get("DATABASE_URL")
    load_profile(profile)
    if preserved_database_url:
        os.environ["DATABASE_URL"] = preserved_database_url


def _delete_existing(anchor_date: date) -> int:
    from sqlalchemy import text

    from src.db.postgres import SessionLocal

    with SessionLocal() as db:
        result = db.execute(
            text("DELETE FROM today_insight_reports WHERE report_date = CAST(:d AS date)"),
            {"d": anchor_date.isoformat()},
        )
        db.commit()
        return int(result.rowcount or 0)


async def _run(args: argparse.Namespace) -> dict[str, object]:
    from src.agents.today_insight_agent import TodayInsightAgent
    from src.api.today_insight_schemas import TodayInsightGenerateRequest

    req = TodayInsightGenerateRequest(
        anchor_date=args.anchor_date,
        window_days=args.window_days,
        max_issues=args.max_issues,
        max_cards=args.max_cards,
        use_cached=args.use_cached,
        force_refresh=not args.use_cached,
        cache_only=False,
        save=not args.no_save,
    )
    return await TodayInsightAgent().generate(req)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    _load_env(args.env)

    if args.replace and args.anchor_date:
        removed = _delete_existing(args.anchor_date)
        log.info("기존 리포트 삭제 | report_date=%s removed=%d", args.anchor_date, removed)

    result = asyncio.run(_run(args))
    signals = result.get("signals") if isinstance(result, dict) else []
    sources = result.get("sources") if isinstance(result, dict) else []
    provenance = result.get("provenance") if isinstance(result, dict) else {}
    context_counts = provenance.get("context_counts") if isinstance(provenance, dict) else {}
    source_url_count = sum(1 for source in sources or [] if source.get("url"))
    current_input_policy = (
        provenance.get("current_input_policy") if isinstance(provenance, dict) else ""
    )
    log.info(
        "today insight generated | report_date=%s saved=%s headline=%s",
        result.get("report_date") if isinstance(result, dict) else None,
        not args.no_save,
        (result.get("headline") if isinstance(result, dict) else "") or "",
    )
    log.info("context_counts=%s current_input_policy=%s", context_counts, current_input_policy)
    log.info(
        "signals=%s sources=%d source_urls=%d",
        [s.get("label") for s in signals or []],
        len(sources or []),
        source_url_count,
    )
    for source in (sources or [])[:5]:
        log.info(
            "source | title=%s publisher=%s url=%s",
            (source.get("title") or "")[:100],
            source.get("source_name") or source.get("publisher") or "",
            source.get("url") or "",
        )


if __name__ == "__main__":
    main()
