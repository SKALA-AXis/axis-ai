# 작성일: 2026-06-09
# 작성자: 최종민
# 변경이력:
#   2026-06-09 최종민 — 컨텍스트 레이어 백필 스크립트 추가 (ContextPackAssembler·주간 digest 에이전트 작업)
#   2026-06-09 박지원 — 뉴스 전처리/클러스터링 품질 개선에 따른 반영
"""Backfill context layers from existing DB history (card_news, integrated_issues).

주간 weekly_digest: peer × ISO week 앵커(해당 주 마지막 카드일)로 chronological 재생성.
today_insight: integrated_issues가 있는 기준일만 backfill (LLM 필요).

사용법:
    # 클러스터 Postgres: DATABASE_URL을 먼저 지정하면 --env cloud의 Supabase URL은 무시됨.
    # --env cloud = OPENAI_API_KEY 등 .env.cloud 비DB 설정만 로드.
    DATABASE_URL=postgresql://axuser:axpass@localhost:15432/axis \\
      uv run python scripts/backfill_historical_context.py --env cloud --weekly-digest

    DATABASE_URL=postgresql://axuser:axpass@localhost:15432/axis \\
      uv run python scripts/backfill_historical_context.py --env cloud --today-insight
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("backfill_historical_context")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical context from DB.")
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument(
        "--weekly-digest",
        action="store_true",
        help="peer×주차 card_news → weekly_digest",
    )
    parser.add_argument(
        "--today-insight",
        action="store_true",
        help="card_news·integrated_issues 기준일 → today_insight (LLM)",
    )
    parser.add_argument(
        "--today-insight-min-cards",
        type=int,
        default=3,
        help="card_news 일별 최소 건수 (integrated_issues 날짜는 항상 포함)",
    )
    parser.add_argument(
        "--replace-today-insight",
        action="store_true",
        help="기준일 기존 today_insight_reports 삭제 후 재생성",
    )
    parser.add_argument("--no-llm", action="store_true", help="weekly digest deterministic only")
    parser.add_argument(
        "--llm-from-week",
        default=None,
        help="ISO week 이상만 LLM (예: 2026-W24). 이전 주는 deterministic.",
    )
    parser.add_argument("--min-cards", type=int, default=1, help="주간 digest 최소 카드 수")
    parser.add_argument("--dry-run", action="store_true", help="실행 계획만 출력")
    return parser.parse_args()


def _discover_peer_week_anchors(*, min_cards: int) -> list[tuple[str, str, date, int]]:
    from sqlalchemy import text

    from src.db.postgres import SessionLocal

    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT peer_company_id AS peer_id,
                       to_char(
                           (created_at AT TIME ZONE 'Asia/Seoul')::date,
                           'IYYY-"W"IW'
                       ) AS week_iso,
                       MAX((created_at AT TIME ZONE 'Asia/Seoul')::date) AS anchor_date,
                       COUNT(*)::int AS card_count
                  FROM card_news
                 WHERE peer_company_id IS NOT NULL
                 GROUP BY peer_company_id, week_iso
                HAVING COUNT(*) >= :min_cards
                 ORDER BY peer_company_id, anchor_date
                """
            ),
            {"min_cards": min_cards},
        ).fetchall()
    out: list[tuple[str, str, date, int]] = []
    for row in rows:
        mapping = row._mapping
        anchor = mapping.get("anchor_date")
        if not isinstance(anchor, date):
            continue
        out.append(
            (
                str(mapping.get("peer_id")),
                str(mapping.get("week_iso")),
                anchor,
                int(mapping.get("card_count") or 0),
            )
        )
    return out


def _discover_today_insight_dates(*, min_cards: int) -> list[date]:
    from sqlalchemy import text

    from src.db.postgres import SessionLocal

    found: set[date] = set()
    with SessionLocal() as db:
        for row in db.execute(
            text(
                """
                SELECT DISTINCT (created_at AT TIME ZONE 'Asia/Seoul')::date AS d
                  FROM integrated_issues
                 WHERE created_at IS NOT NULL
                """
            )
        ).fetchall():
            value = row._mapping.get("d")
            if isinstance(value, date):
                found.add(value)

        for row in db.execute(
            text(
                """
                SELECT (created_at AT TIME ZONE 'Asia/Seoul')::date AS d,
                       COUNT(*)::int AS card_count
                  FROM card_news
                 GROUP BY 1
                 HAVING COUNT(*) >= :min_cards
                """
            ),
            {"min_cards": min_cards},
        ).fetchall():
            value = row._mapping.get("d")
            if isinstance(value, date):
                found.add(value)

    return sorted(found)


def _purge_today_insight_for_date(anchor: date) -> int:
    from sqlalchemy import text

    from src.db.postgres import SessionLocal

    with SessionLocal() as db:
        result = db.execute(
            text("DELETE FROM today_insight_reports WHERE report_date = CAST(:d AS date)"),
            {"d": anchor.isoformat()},
        )
        db.commit()
        return int(result.rowcount or 0)


def _use_llm_for_week(week_iso: str, *, no_llm: bool, llm_from_week: str | None) -> bool:
    if no_llm:
        return False
    if llm_from_week:
        return week_iso >= llm_from_week
    return True


def _backfill_weekly_digest(
    *,
    min_cards: int,
    no_llm: bool,
    llm_from_week: str | None,
    dry_run: bool,
) -> None:
    from src.agents.context.weekly_digest_agent import WeeklyDigestAgent

    anchors = _discover_peer_week_anchors(min_cards=min_cards)
    by_peer: dict[str, list[tuple[str, date, int]]] = defaultdict(list)
    for peer_id, week_iso, anchor_date, card_count in anchors:
        by_peer[peer_id].append((week_iso, anchor_date, card_count))

    agent = WeeklyDigestAgent()
    total = sum(len(v) for v in by_peer.values())
    log.info("weekly digest backfill plan | peers=%d jobs=%d", len(by_peer), total)

    for peer_id in sorted(by_peer):
        for week_iso, anchor_date, card_count in sorted(by_peer[peer_id], key=lambda x: x[1]):
            use_llm = _use_llm_for_week(week_iso, no_llm=no_llm, llm_from_week=llm_from_week)
            log.info(
                "weekly | peer=%s week=%s anchor=%s cards=%d llm=%s",
                peer_id,
                week_iso,
                anchor_date.isoformat(),
                card_count,
                use_llm,
            )
            if dry_run:
                continue
            result = agent.run(
                peer_id=peer_id,
                anchor_date=anchor_date,
                use_llm=use_llm,
            )
            agent.persist(peer_id, result)
            if result.get("skipped"):
                log.info(
                    "skip | peer=%s week=%s reason=%s",
                    peer_id,
                    week_iso,
                    result.get("reason"),
                )
            else:
                digest = result.get("digest") or {}
                log.info(
                    "done | peer=%s week=%s cards=%s deltas=%s",
                    peer_id,
                    digest.get("week_iso"),
                    result.get("card_count"),
                    len(digest.get("delta_vs_prev") or []),
                )


async def _backfill_today_insight(
    *,
    dry_run: bool,
    min_cards: int,
    replace: bool,
) -> None:
    from src.agents.today_insight_agent import TodayInsightAgent
    from src.api.today_insight_schemas import TodayInsightGenerateRequest

    dates = _discover_today_insight_dates(min_cards=min_cards)
    log.info(
        "today insight backfill plan | dates=%s min_cards=%d replace=%s",
        [d.isoformat() for d in dates],
        min_cards,
        replace,
    )
    if dry_run:
        return
    agent = TodayInsightAgent()
    for anchor in dates:
        if replace:
            removed = _purge_today_insight_for_date(anchor)
            if removed:
                log.info("today insight purge | date=%s removed=%d", anchor.isoformat(), removed)
        req = TodayInsightGenerateRequest(
            anchor_date=anchor,
            use_cached=False,
            force_refresh=True,
            save=True,
            cache_only=False,
        )
        log.info("today insight | anchor=%s", anchor.isoformat())
        result = await agent.generate(req)
        log.info(
            "today insight done | date=%s headline=%s signals=%s",
            result.get("report_date"),
            (result.get("headline") or "")[:80],
            len(result.get("signals") or []),
        )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.config.env_loader import load_profile

    preserve_database_url = os.environ.get("DATABASE_URL")
    load_profile(args.env)
    if preserve_database_url:
        os.environ["DATABASE_URL"] = preserve_database_url

    if not args.weekly_digest and not args.today_insight:
        log.error("Specify --weekly-digest and/or --today-insight")
        sys.exit(1)

    if args.weekly_digest:
        _backfill_weekly_digest(
            min_cards=args.min_cards,
            no_llm=args.no_llm,
            llm_from_week=args.llm_from_week,
            dry_run=args.dry_run,
        )

    if args.today_insight:
        if args.no_llm:
            log.warning("today insight requires LLM — skipped with --no-llm")
        else:
            asyncio.run(
                _backfill_today_insight(
                    dry_run=args.dry_run,
                    min_cards=args.today_insight_min_cards,
                    replace=args.replace_today_insight,
                )
            )

    log.info("backfill complete")


if __name__ == "__main__":
    main()
