"""Backfill card_news for historical raw_article clusters.

This runs the current analysis/card-news agent stack for existing processed
news clusters. It is intentionally resumable: clusters that already have an
ACTIVE card are skipped unless --replace-existing is provided.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.env_loader import load_profile  # noqa: E402
from src.db.article_store import save_card_news, sync_card_sources_for_cluster  # noqa: E402
from src.db.postgres import SessionLocal  # noqa: E402
from src.pipeline.analysis_pipeline import AnalysisPipelineRunner  # noqa: E402
from src.preprocessing.preprocessing import PreprocessingService  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("backfill_card_news")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill card_news for existing news clusters")
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--published-since", required=True)
    parser.add_argument("--published-until", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument(
        "--update-existing-in-place",
        action="store_true",
        help="Regenerate with the current agent stack but keep the existing ACTIVE card id.",
    )
    parser.add_argument(
        "--only-card-schema-version",
        default=None,
        help="Limit targets to clusters with an ACTIVE card using this schema version, e.g. v1.",
    )
    parser.add_argument(
        "--only-existing-active",
        action="store_true",
        help="Limit targets to clusters that currently have an ACTIVE card.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    include_existing = bool(
        args.replace_existing or args.update_existing_in_place or args.only_card_schema_version
    )
    targets = _load_targets(
        published_since=args.published_since,
        published_until=args.published_until,
        limit=max(1, args.limit),
        include_existing=include_existing,
        only_card_schema_version=args.only_card_schema_version,
        only_existing_active=args.only_existing_active,
    )
    log.info(
        (
            "card_news backfill 대상 | profile=%s since=%s until=%s targets=%d "
            "replace_existing=%s update_in_place=%s schema=%s"
        ),
        profile,
        args.published_since,
        args.published_until,
        len(targets),
        args.replace_existing,
        args.update_existing_in_place,
        args.only_card_schema_version,
    )
    if args.dry_run:
        for target in targets[:20]:
            print(
                f"cluster_id={target['cluster_id']} "
                f"representative_id={target['representative_id']} "
                f"articles={len(target['article_ids'])} "
                f"existing_card_id={target.get('existing_card_id') or ''}"
            )
        return

    service = PreprocessingService()
    runner = AnalysisPipelineRunner()
    created = 0
    skipped = 0
    errors = 0

    for index, target in enumerate(targets, start=1):
        cluster_id = int(target["cluster_id"])
        article_ids = [int(value) for value in target["article_ids"]]
        representative_id = int(target["representative_id"])
        existing_card_id = str(target.get("existing_card_id") or "")
        try:
            classified = service.classify_clusters(
                representative_ids=[representative_id],
                cluster_map={cluster_id: article_ids},
                requested_companies=[],
            )
            if not classified:
                skipped += 1
                log.info(
                    "card_news backfill skip | cluster_id=%s reason=no_classification", cluster_id
                )
                continue
            if args.replace_existing and not args.update_existing_in_place:
                _mark_existing_cards_deleted(cluster_id)
            result = runner.run_cluster(
                cluster_id=cluster_id,
                representative_id=representative_id,
                cluster_article_ids=article_ids,
                classification=classified[0],
                save_card=True,
            )
            card = result.get("card_news") or {}
            if card:
                if args.update_existing_in_place:
                    if existing_card_id:
                        transient_card_id = str(card.get("id") or "")
                        card["id"] = existing_card_id
                        saved_card_id = save_card_news(card)
                        sync_card_sources_for_cluster(cluster_id)
                        if transient_card_id and transient_card_id != existing_card_id:
                            _mark_card_deleted(transient_card_id)
                        created += 1
                        log.info(
                            (
                                "card_news backfill updated in place | %d/%d "
                                "cluster_id=%s card_id=%s transient_id=%s title=%s"
                            ),
                            index,
                            len(targets),
                            cluster_id,
                            saved_card_id or existing_card_id,
                            transient_card_id,
                            card.get("title"),
                        )
                        continue
                    log.warning(
                        "card_news backfill skip | cluster_id=%s reason=no_existing_card",
                        cluster_id,
                    )
                    skipped += 1
                    continue
                created += 1
                sync_card_sources_for_cluster(cluster_id)
                log.info(
                    "card_news backfill created | %d/%d cluster_id=%s card_id=%s title=%s",
                    index,
                    len(targets),
                    cluster_id,
                    card.get("id"),
                    card.get("title"),
                )
            else:
                if args.update_existing_in_place and existing_card_id:
                    deleted = _mark_card_deleted(existing_card_id)
                    skipped += 1
                    log.info(
                        (
                            "card_news backfill deleted stale existing card | %d/%d "
                            "cluster_id=%s card_id=%s reason=no_card deleted=%d"
                        ),
                        index,
                        len(targets),
                        cluster_id,
                        existing_card_id,
                        deleted,
                    )
                    continue
                skipped += 1
                log.info("card_news backfill skip | cluster_id=%s reason=no_card", cluster_id)
        except Exception as exc:
            errors += 1
            log.exception("card_news backfill error | cluster_id=%s error=%s", cluster_id, exc)

    log.info("card_news backfill 완료 | created=%d skipped=%d errors=%d", created, skipped, errors)


def _load_targets(
    *,
    published_since: str,
    published_until: str,
    limit: int,
    include_existing: bool,
    only_card_schema_version: str | None,
    only_existing_active: bool,
) -> list[dict[str, Any]]:
    existing_filter = (
        ""
        if include_existing
        else """
      AND NOT EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'ACTIVE'
            AND cn.cluster_id = cluster_rows.cluster_id
      )
    """
    )
    schema_filter = (
        """
      AND EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'ACTIVE'
            AND cn.cluster_id = cluster_rows.cluster_id
            AND cn.card_schema_version = :only_card_schema_version
      )
    """
        if only_card_schema_version
        else ""
    )
    existing_active_filter = (
        """
      AND EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'ACTIVE'
            AND cn.cluster_id = cluster_rows.cluster_id
      )
    """
        if only_existing_active
        else ""
    )
    with SessionLocal() as db:
        rows = db.execute(
            text(
                f"""
                WITH cluster_rows AS (
                    SELECT
                        ra.cluster_id,
                        ARRAY_AGG(
                            ra.id
                            ORDER BY
                                ra.published_at DESC NULLS LAST,
                                ra.collected_at DESC NULLS LAST,
                                ra.id DESC
                        ) AS article_ids,
                        (
                            ARRAY_AGG(
                                ra.id
                                ORDER BY
                                    ra.is_representative DESC,
                                    ra.published_at DESC NULLS LAST,
                                    ra.collected_at DESC NULLS LAST,
                                    ra.id DESC
                            )
                        )[1] AS representative_id,
                        MIN(ra.published_at) AS min_published
                    FROM raw_articles ra
                    WHERE ra.source_type = 'news'
                      AND ra.published_at >= CAST(:published_since AS timestamptz)
                      AND ra.published_at < CAST(:published_until AS timestamptz)
                      AND ra.processing_status = 'PROCESSED'
                      AND ra.relevance_label = 'relevant'
                      AND ra.cluster_id IS NOT NULL
                    GROUP BY ra.cluster_id
                )
                SELECT cluster_id, article_ids, representative_id
                     , (
                          SELECT cn.id
                          FROM card_news cn
                          WHERE cn.status = 'ACTIVE'
                            AND cn.cluster_id = cluster_rows.cluster_id
                          ORDER BY cn.created_at DESC
                          LIMIT 1
                       ) AS existing_card_id
                FROM cluster_rows
                WHERE TRUE
                  {existing_filter}
                  {schema_filter}
                  {existing_active_filter}
                ORDER BY min_published, cluster_id
                LIMIT :limit
                """
            ),
            {
                "published_since": published_since,
                "published_until": published_until,
                "limit": limit,
                "only_card_schema_version": only_card_schema_version,
            },
        ).mappings()
        return [dict(row) for row in rows]


def _mark_existing_cards_deleted(cluster_id: int) -> int:
    with SessionLocal() as db:
        result = db.execute(
            text("""
                UPDATE card_news
                SET status = 'DELETED'
                WHERE status = 'ACTIVE'
                  AND cluster_id = :cluster_id
            """),
            {"cluster_id": cluster_id},
        )
        db.commit()
        return int(getattr(result, "rowcount", 0) or 0)


def _mark_card_deleted(card_id: str) -> int:
    with SessionLocal() as db:
        result = db.execute(
            text("""
                UPDATE card_news
                SET status = 'DELETED'
                WHERE id = :card_id
            """),
            {"card_id": card_id},
        )
        db.commit()
        return int(getattr(result, "rowcount", 0) or 0)


if __name__ == "__main__":
    main()
