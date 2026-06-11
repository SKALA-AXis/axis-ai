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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    targets = _load_targets(
        published_since=args.published_since,
        published_until=args.published_until,
        limit=max(1, args.limit),
        replace_existing=bool(args.replace_existing),
    )
    log.info(
        "card_news backfill 대상 | profile=%s since=%s until=%s targets=%d replace_existing=%s",
        profile,
        args.published_since,
        args.published_until,
        len(targets),
        args.replace_existing,
    )
    if args.dry_run:
        for target in targets[:20]:
            print(
                f"cluster_id={target['cluster_id']} representative_id={target['representative_id']} "
                f"articles={len(target['article_ids'])}"
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
        try:
            classified = service.classify_clusters(
                representative_ids=[representative_id],
                cluster_map={cluster_id: article_ids},
                requested_companies=[],
            )
            if not classified:
                skipped += 1
                log.info("card_news backfill skip | cluster_id=%s reason=no_classification", cluster_id)
                continue
            if args.replace_existing:
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
                created += 1
                _sync_card_sources_for_cluster(cluster_id)
                log.info(
                    "card_news backfill created | %d/%d cluster_id=%s card_id=%s title=%s",
                    index,
                    len(targets),
                    cluster_id,
                    card.get("id"),
                    card.get("title"),
                )
            else:
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
    replace_existing: bool,
) -> list[dict[str, Any]]:
    existing_filter = "" if replace_existing else """
      AND NOT EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'ACTIVE'
            AND cn.cluster_id = cluster_rows.cluster_id
      )
    """
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
                FROM cluster_rows
                WHERE TRUE
                  {existing_filter}
                ORDER BY min_published, cluster_id
                LIMIT :limit
                """
            ),
            {
                "published_since": published_since,
                "published_until": published_until,
                "limit": limit,
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


def _sync_card_sources_for_cluster(cluster_id: int) -> int:
    with SessionLocal() as db:
        result = db.execute(
            text(
                """
                WITH ranked_articles AS (
                    SELECT
                        ra.cluster_id,
                        ra.id,
                        ra.title,
                        ra.url,
                        ra.source_name,
                        ra.publisher,
                        ra.published_at,
                        ra.collected_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY ra.cluster_id
                            ORDER BY
                                ra.published_at DESC NULLS LAST,
                                ra.collected_at DESC NULLS LAST,
                                ra.id DESC
                        ) AS rn
                    FROM raw_articles ra
                    WHERE ra.cluster_id = :cluster_id
                      AND ra.processing_status = 'PROCESSED'
                      AND ra.relevance_label = 'relevant'
                ),
                cluster_sources AS (
                    SELECT
                        cluster_id,
                        ARRAY_AGG(
                            id
                            ORDER BY published_at DESC NULLS LAST, collected_at DESC NULLS LAST, id DESC
                        ) AS raw_ids,
                        JSONB_AGG(
                            JSONB_BUILD_OBJECT(
                                'index', rn,
                                'raw_article_id', id,
                                'title', COALESCE(title, ''),
                                'source_name', COALESCE(source_name, publisher, ''),
                                'url', COALESCE(url, ''),
                                'published_at', published_at,
                                'collected_at', collected_at
                            )
                            ORDER BY published_at DESC NULLS LAST, collected_at DESC NULLS LAST, id DESC
                        ) AS sources,
                        JSONB_AGG(
                            JSONB_BUILD_OBJECT(
                                'id', id,
                                'title', COALESCE(title, ''),
                                'url', COALESCE(url, ''),
                                'source_name', COALESCE(source_name, ''),
                                'publisher', COALESCE(publisher, ''),
                                'published_at', published_at,
                                'collected_at', collected_at
                            )
                            ORDER BY published_at DESC NULLS LAST, collected_at DESC NULLS LAST, id DESC
                        ) AS source_articles
                    FROM ranked_articles
                    GROUP BY cluster_id
                )
                UPDATE card_news cn
                SET source_raw_article_ids = cs.raw_ids,
                    sources = cs.sources,
                    source_articles = cs.source_articles
                FROM cluster_sources cs
                WHERE cn.status = 'ACTIVE'
                  AND cn.cluster_id = cs.cluster_id
            """
            ),
            {"cluster_id": cluster_id},
        )
        db.commit()
        return int(getattr(result, "rowcount", 0) or 0)


if __name__ == "__main__":
    main()
