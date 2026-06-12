"""Backfill weekly/monthly briefing_reports from card news article dates.

The period range is based on raw article ``published_at`` linked to card_news,
not card_news upload time. Existing reports are reused unless ``--force`` is set.

Usage:
    uv run python scripts/backfill_briefings.py --type weekly --type monthly --no-llm
    uv run python scripts/backfill_briefings.py --from-date 2026-05-01 --to-date 2026-06-12
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("backfill_briefings")

BriefingType = Literal["weekly", "monthly"]


_CARD_BOUNDS_SQL = text("""
    SELECT MIN(basis_at)::date AS start_date,
           MAX(basis_at)::date AS end_date
    FROM (
        SELECT COALESCE(src.latest_published_at, cn.created_at) AS basis_at
          FROM card_news cn
          LEFT JOIN LATERAL (
              SELECT MAX(ra.published_at) AS latest_published_at
                FROM raw_articles ra
               WHERE ra.id = cn.primary_raw_article_id
                  OR ra.id = ANY(COALESCE(cn.source_raw_article_ids, '{}'::bigint[]))
          ) src ON TRUE
         WHERE COALESCE(cn.status, 'ACTIVE') = 'ACTIVE'
    ) dated_cards
    WHERE basis_at IS NOT NULL
""")


async def _generate_one(
    *,
    briefing_type: BriefingType,
    anchor_date: date,
    limit: int,
    refine_display_copy: bool,
    reuse_saved: bool,
    dry_run: bool,
) -> dict:
    from src.agents.briefing_generation_agent import BriefingGenerationAgent

    if dry_run:
        return {
            "id": f"dry-run-{briefing_type}-{anchor_date.isoformat()}",
            "briefing_type": briefing_type,
            "anchor_date": anchor_date.isoformat(),
            "status": "dry_run",
            "related_card_ids": [],
        }

    return await BriefingGenerationAgent().generate(
        briefing_type=briefing_type,
        anchor_date=anchor_date,
        limit=limit,
        save=True,
        use_mock=False,
        refine_display_copy=refine_display_copy,
        reuse_saved=reuse_saved,
    )


def _load_card_date_bounds() -> tuple[date, date] | None:
    from src.db.postgres import SessionLocal

    with SessionLocal() as db:
        row = db.execute(_CARD_BOUNDS_SQL).mappings().first()
    if not row or not row.get("start_date") or not row.get("end_date"):
        return None
    return _coerce_date(row["start_date"]), _coerce_date(row["end_date"])


def _coerce_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.fromisoformat(value).date()


def _week_anchors(start: date, end: date) -> list[date]:
    week_start = start - timedelta(days=start.weekday())
    anchors: list[date] = []
    current = week_start
    while current <= end:
        anchors.append(current + timedelta(days=6))
        current += timedelta(days=7)
    return anchors


def _month_anchors(start: date, end: date) -> list[date]:
    anchors: list[date] = []
    current = start.replace(day=1)
    while current <= end:
        anchors.append(current)
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
    return anchors


def _anchors_for(briefing_type: BriefingType, start: date, end: date) -> list[date]:
    if briefing_type == "weekly":
        return _week_anchors(start, end)
    return _month_anchors(start, end)


async def _run(args: argparse.Namespace) -> None:
    from src.config.env_loader import load_profile

    preserve_database_url = os.environ.get("DATABASE_URL")
    load_profile(args.env)
    if preserve_database_url:
        os.environ["DATABASE_URL"] = preserve_database_url

    bounds = _load_card_date_bounds()
    explicit_start = _parse_date(args.from_date)
    explicit_end = _parse_date(args.to_date)
    if bounds is None and (explicit_start is None or explicit_end is None):
        log.warning("활성 카드뉴스 기준 날짜 범위를 찾지 못했습니다.")
        return

    start = explicit_start or bounds[0]  # type: ignore[index]
    end = explicit_end or bounds[1]  # type: ignore[index]
    if start > end:
        raise ValueError(f"invalid date range: {start} > {end}")

    types: list[BriefingType] = args.type or ["weekly", "monthly"]
    total = 0
    for briefing_type in types:
        anchors = _anchors_for(briefing_type, start, end)
        if args.exclude_current:
            today = date.today()
            anchors = [
                anchor
                for anchor in anchors
                if not (
                    (briefing_type == "weekly" and anchor - timedelta(days=6) <= today <= anchor)
                    or (
                        briefing_type == "monthly"
                        and anchor
                        <= today
                        < (anchor.replace(day=28) + timedelta(days=4)).replace(day=1)
                    )
                )
            ]
        log.info("%s backfill 대상 기간=%d | range=%s~%s", briefing_type, len(anchors), start, end)
        for anchor in anchors:
            result = await _generate_one(
                briefing_type=briefing_type,
                anchor_date=anchor,
                limit=args.limit,
                refine_display_copy=not args.no_llm,
                reuse_saved=not args.force,
                dry_run=args.dry_run,
            )
            related_cards = result.get("related_card_ids") or result.get("source_card_ids") or []
            log.info(
                "완료 | type=%s anchor=%s id=%s cards=%d status=%s",
                briefing_type,
                anchor.isoformat(),
                result.get("id"),
                len(related_cards) if isinstance(related_cards, list) else 0,
                result.get("status"),
            )
            total += 1
    log.info("브리핑 backfill 완료 | reports=%d dry_run=%s", total, args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill weekly/monthly briefing_reports.")
    parser.add_argument("--type", action="append", choices=["weekly", "monthly"], default=None)
    parser.add_argument(
        "--from-date",
        default=None,
        help="YYYY-MM-DD. Default: earliest card article date.",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        help="YYYY-MM-DD. Default: latest card article date.",
    )
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--no-llm", action="store_true", help="Skip display-copy LLM refinement.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when a saved report exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target periods without saving.",
    )
    parser.add_argument("--exclude-current", action="store_true", help="Skip current week/month.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
