"""Warm up unified context layers (today insight, weekly digest, optional cards).

Usage:
  uv run python scripts/warmup_context_layers.py --env cloud
  uv run python scripts/warmup_context_layers.py --env cloud --skip-llm
  uv run python scripts/warmup_context_layers.py --env cloud --card-clusters 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("warmup_context_layers")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up context layer derived data.")
    parser.add_argument("--env", choices=["local", "cloud"], default="cloud")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Today insight / weekly digest LLM 호출 생략.",
    )
    parser.add_argument(
        "--card-clusters",
        type=int,
        default=0,
        help="카드가 없는 PROCESSED 클러스터 N개에 analysis pipeline 실행 (0=skip).",
    )
    parser.add_argument(
        "--no-today-insight",
        action="store_true",
        help="Today's Insight 생성 생략 (기본은 1회 생성·저장).",
    )
    parser.add_argument(
        "--weekly-digest",
        action="store_true",
        help="Weekly digest 갱신 실행 (card_news 7일 기준).",
    )
    return parser.parse_args()


async def _warmup_today_insight(*, use_llm: bool) -> None:
    from src.agents.today_insight_agent import TodayInsightAgent
    from src.api.today_insight_schemas import TodayInsightGenerateRequest

    if not use_llm:
        log.warning("today insight requires LLM — skipping when --skip-llm")
        return

    req = TodayInsightGenerateRequest(
        use_cached=False,
        force_refresh=True,
        save=True,
        cache_only=False,
    )
    result = await TodayInsightAgent().generate(req)
    log.info(
        "today insight saved | report_date=%s headline=%s signals=%s",
        result.get("report_date"),
        (result.get("headline") or "")[:80],
        len(result.get("signals") or []),
    )


def _warmup_card_clusters(*, limit: int) -> None:
    from sqlalchemy import text

    from src.db.postgres import SessionLocal
    from src.pipeline.analysis_pipeline import AnalysisPipelineRunner

    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT DISTINCT ra.cluster_id::bigint AS cluster_id
                  FROM raw_articles ra
                 WHERE ra.cluster_id IS NOT NULL
                   AND ra.processing_status = 'PROCESSED'
                   AND ra.is_representative = TRUE
                   AND NOT EXISTS (
                       SELECT 1
                         FROM card_news cn
                        WHERE cn.cluster_id = ra.cluster_id
                   )
                 ORDER BY ra.cluster_id DESC
                 LIMIT :limit
                """
            ),
            {"limit": int(limit)},
        ).fetchall()

    cluster_ids = [
        int(row._mapping["cluster_id"]) for row in rows if row._mapping.get("cluster_id")
    ]
    if not cluster_ids:
        log.info("card warmup | no clusters without cards")
        return

    runner = AnalysisPipelineRunner()
    for cluster_id in cluster_ids:
        log.info("card pipeline | cluster_id=%s", cluster_id)
        try:
            result = runner.run_cluster(cluster_id=cluster_id, save_card=True)
            card_id = result.get("card_news_id") or (result.get("card_news") or {}).get("id")
            log.info("card pipeline done | cluster=%s card=%s", cluster_id, card_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("card pipeline failed | cluster=%s error=%s", cluster_id, exc)


def _warmup_weekly_digest(*, use_llm: bool) -> None:
    from src.agents.context.weekly_digest_agent import WeeklyDigestAgent
    from src.config.companies import COMPANY_IDS

    agent = WeeklyDigestAgent()
    for company_id in COMPANY_IDS:
        log.info("weekly digest | peer=%s llm=%s", company_id, use_llm)
        try:
            result = agent.run(peer_id=company_id, use_llm=use_llm)
            agent.persist(company_id, result)
        except Exception as exc:  # noqa: BLE001
            log.exception("weekly digest failed | peer=%s error=%s", company_id, exc)
            continue
        if result.get("skipped"):
            log.warning(
                "weekly digest skipped | peer=%s reason=%s", company_id, result.get("reason")
            )
        else:
            digest = result.get("digest") or {}
            log.info(
                "weekly digest done | peer=%s week=%s cards=%s",
                company_id,
                digest.get("week_iso"),
                result.get("card_count"),
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
    use_llm = not args.skip_llm

    if args.card_clusters > 0:
        log.info("=== card pipeline warmup | clusters=%s ===", args.card_clusters)
        _warmup_card_clusters(limit=args.card_clusters)

    if args.weekly_digest:
        log.info("=== weekly digest warmup (llm=%s) ===", use_llm)
        _warmup_weekly_digest(use_llm=use_llm)

    if not args.no_today_insight:
        log.info("=== today insight warmup ===")
        asyncio.run(_warmup_today_insight(use_llm=use_llm))

    log.info("warmup complete")


if __name__ == "__main__":
    main()
