# 작성일: 2026-06-09
# 작성자: 박지원
# 변경이력:
#   2026-06-09 박지원 — 뉴스 클러스터링 품질 점검 스크립트 추가 (전처리/클러스터링 품질 개선 작업)
"""Audit recent news clustering quality without mutating data.

This script inspects recent processed news clusters and reports risk signals:

- too many singleton/small clusters
- small clusters that look mergeable into larger clusters
- multiple small clusters that share a concrete event key
- active card_news rows that now point to the same current raw_articles cluster

It intentionally does not update DB rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.postprocess_singleton_clusters import (  # noqa: E402
    Cluster,
    GroupMergeCandidate,
    MergeCandidate,
    _find_candidates,
    _find_source_group_candidates,
    _load_clusters,
    _split_clusters,
)
from src.config.env_loader import load_profile  # noqa: E402
from src.db.postgres import SessionLocal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("audit_news_cluster_quality")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit recent news cluster quality")
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--lookback-hours", type=int, default=4)
    parser.add_argument(
        "--time-field",
        choices=["published_at", "collected_at"],
        default="published_at",
    )
    parser.add_argument("--source-type", default="news")
    parser.add_argument("--max-source-size", type=int, default=3)
    parser.add_argument("--min-target-size", type=int, default=4)
    parser.add_argument("--min-new-cluster-size", type=int, default=2)
    parser.add_argument("--max-time-gap-hours", type=int, default=72)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--warn-singleton-rate", type=float, default=0.45)
    parser.add_argument("--warn-small-rate", type=float, default=0.65)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    log.info(
        "뉴스 클러스터 품질 점검 시작 | profile=%s lookback_hours=%d time_field=%s",
        profile,
        args.lookback_hours,
        args.time_field,
    )

    with SessionLocal() as db:
        result = run_audit(
            db=db,
            source_type=args.source_type,
            lookback_hours=args.lookback_hours,
            time_field=args.time_field,
            max_source_size=args.max_source_size,
            min_target_size=args.min_target_size,
            min_new_cluster_size=args.min_new_cluster_size,
            max_time_gap_hours=args.max_time_gap_hours,
            min_score=args.min_score,
            warn_singleton_rate=args.warn_singleton_rate,
            warn_small_rate=args.warn_small_rate,
        )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
        return

    _print_text_report(result)


def run_audit(
    *,
    db: Any,
    source_type: str = "news",
    lookback_hours: int = 4,
    time_field: str = "published_at",
    max_source_size: int = 3,
    min_target_size: int = 4,
    min_new_cluster_size: int = 2,
    max_time_gap_hours: int = 72,
    min_score: float = 0.45,
    warn_singleton_rate: float = 0.45,
    warn_small_rate: float = 0.65,
) -> dict[str, Any]:
    clusters = _load_clusters(db, source_type, lookback_hours, time_field)
    sources, targets = _split_clusters(
        clusters,
        max_source_size=max_source_size,
        min_target_size=min_target_size,
    )
    merge_candidates = _find_candidates(
        sources=sources,
        targets=targets,
        max_time_gap_hours=max_time_gap_hours,
        min_score=min_score,
    )
    group_candidates = _find_source_group_candidates(
        sources=sources,
        max_time_gap_hours=max_time_gap_hours,
        min_score=min_score,
        min_new_cluster_size=min_new_cluster_size,
    )
    card_risks = _find_stale_card_risks(db, source_type, lookback_hours, time_field)
    metrics = _cluster_metrics(clusters, max_source_size=max_source_size)
    risk = _risk_level(
        metrics=metrics,
        merge_candidate_count=len(merge_candidates),
        group_candidate_count=len(group_candidates),
        stale_card_count=len(card_risks),
        warn_singleton_rate=warn_singleton_rate,
        warn_small_rate=warn_small_rate,
    )

    return {
        "risk": risk,
        "metrics": metrics,
        "merge_candidates": [_merge_candidate_payload(candidate) for candidate in merge_candidates],
        "group_candidates": [_group_candidate_payload(candidate) for candidate in group_candidates],
        "stale_card_risks": card_risks,
    }


def _cluster_metrics(clusters: list[Cluster], *, max_source_size: int) -> dict[str, Any]:
    cluster_count = len(clusters)
    article_count = sum(cluster.article_count for cluster in clusters)
    singleton_count = sum(1 for cluster in clusters if cluster.article_count == 1)
    small_count = sum(1 for cluster in clusters if cluster.article_count <= max_source_size)
    return {
        "cluster_count": cluster_count,
        "article_count": article_count,
        "singleton_count": singleton_count,
        "small_count": small_count,
        "singleton_rate": _rate(singleton_count, cluster_count),
        "small_rate": _rate(small_count, cluster_count),
    }


def _risk_level(
    *,
    metrics: dict[str, Any],
    merge_candidate_count: int,
    group_candidate_count: int,
    stale_card_count: int,
    warn_singleton_rate: float,
    warn_small_rate: float,
) -> str:
    if stale_card_count > 0:
        return "HIGH"
    if merge_candidate_count >= 3 or group_candidate_count >= 2:
        return "HIGH"
    if merge_candidate_count > 0 or group_candidate_count > 0:
        return "MEDIUM"
    if metrics["cluster_count"] >= 10 and (
        metrics["singleton_rate"] >= warn_singleton_rate or metrics["small_rate"] >= warn_small_rate
    ):
        return "MEDIUM"
    return "LOW"


def _find_stale_card_risks(
    db: Any,
    source_type: str,
    lookback_hours: int,
    time_field: str,
) -> list[dict[str, Any]]:
    if not _table_exists(db, "card_news"):
        return []

    time_column = "published_at" if time_field == "published_at" else "collected_at"
    rows = db.execute(
        text(
            f"""
            WITH recent_cards AS (
                SELECT
                    cn.id,
                    cn.cluster_id,
                    cn.title,
                    cn.source_raw_article_ids,
                    cn.created_at,
                    ARRAY_AGG(DISTINCT ra.cluster_id ORDER BY ra.cluster_id)
                        FILTER (WHERE ra.cluster_id IS NOT NULL) AS source_clusters
                FROM card_news cn
                LEFT JOIN LATERAL unnest(
                    COALESCE(cn.source_raw_article_ids, ARRAY[]::bigint[])
                ) source_id(id) ON TRUE
                LEFT JOIN raw_articles ra ON ra.id = source_id.id
                WHERE cn.status = 'ACTIVE'
                  AND cn.created_at >= now() - (:lookback_hours * interval '1 hour')
                GROUP BY cn.id, cn.cluster_id, cn.title, cn.source_raw_article_ids, cn.created_at
            )
            SELECT
                source_clusters,
                COUNT(*) AS card_count,
                ARRAY_AGG(id ORDER BY created_at DESC) AS card_ids,
                ARRAY_AGG(cluster_id ORDER BY created_at DESC) AS card_cluster_ids,
                ARRAY_AGG(title ORDER BY created_at DESC) AS titles
            FROM recent_cards
            WHERE source_clusters IS NOT NULL
              AND cardinality(source_clusters) = 1
              AND EXISTS (
                  SELECT 1
                  FROM raw_articles recent
                  WHERE recent.cluster_id = source_clusters[1]
                    AND recent.source_type = :source_type
                    AND recent.{time_column} >= now() - (:lookback_hours * interval '1 hour')
              )
            GROUP BY source_clusters
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 20
            """
        ),
        {"source_type": source_type, "lookback_hours": lookback_hours},
    ).mappings()
    return [
        {
            "current_cluster_id": int(row["source_clusters"][0]),
            "card_count": int(row["card_count"]),
            "card_ids": [str(value) for value in row["card_ids"]],
            "card_cluster_ids": [int(value) for value in row["card_cluster_ids"]],
            "titles": [str(value or "") for value in row["titles"][:5]],
        }
        for row in rows
    ]


def _table_exists(db: Any, table_name: str) -> bool:
    return bool(
        db.execute(
            text("SELECT to_regclass(:table_name)"),
            {"table_name": table_name},
        ).scalar()
    )


def _merge_candidate_payload(candidate: MergeCandidate) -> dict[str, Any]:
    return {
        "type": "merge",
        "from_cluster": candidate.source.cluster_id,
        "to_cluster": candidate.target.cluster_id,
        "source_count": candidate.source.article_count,
        "target_count": candidate.target.article_count,
        "article_ids": candidate.source.article_ids,
        "event_key": candidate.event_key,
        "score": round(candidate.score, 3),
        "shared_tokens": sorted(candidate.shared_tokens),
        "source_titles": candidate.source.titles[:3],
        "target_titles": candidate.target.titles[:3],
    }


def _group_candidate_payload(candidate: GroupMergeCandidate) -> dict[str, Any]:
    return {
        "type": "group_merge",
        "to_cluster": candidate.target.cluster_id,
        "from_clusters": [source.cluster_id for source in candidate.sources],
        "article_ids": [
            article_id for source in candidate.sources for article_id in source.article_ids
        ],
        "event_key": candidate.event_key,
        "score": round(candidate.score, 3),
        "shared_tokens": sorted(candidate.shared_tokens),
        "target_titles": candidate.target.titles[:3],
        "source_titles": [
            source.titles[:3]
            for source in candidate.sources
            if source.cluster_id != candidate.target.cluster_id
        ],
    }


def _print_text_report(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print("NEWS_CLUSTER_QUALITY_AUDIT")
    print(
        f"risk={result['risk']} clusters={metrics['cluster_count']} "
        f"articles={metrics['article_count']} singleton_rate={metrics['singleton_rate']:.2f} "
        f"small_rate={metrics['small_rate']:.2f}"
    )
    print(
        f"merge_candidates={len(result['merge_candidates'])} "
        f"group_candidates={len(result['group_candidates'])} "
        f"stale_card_risks={len(result['stale_card_risks'])}"
    )

    for candidate in result["merge_candidates"]:
        print(
            "MERGE_RISK",
            f"from_cluster={candidate['from_cluster']}",
            f"to_cluster={candidate['to_cluster']}",
            f"source_count={candidate['source_count']}",
            f"target_count={candidate['target_count']}",
            f"event_key={candidate['event_key']}",
            f"score={candidate['score']:.3f}",
        )
        print(f"  source_titles={candidate['source_titles']}")
        print(f"  target_titles={candidate['target_titles']}")

    for candidate in result["group_candidates"]:
        print(
            "SPLIT_EVENT_RISK",
            f"from_clusters={candidate['from_clusters']}",
            f"to_cluster={candidate['to_cluster']}",
            f"event_key={candidate['event_key']}",
            f"score={candidate['score']:.3f}",
        )
        print(f"  target_titles={candidate['target_titles']}")
        print(f"  source_titles={candidate['source_titles']}")

    for risk in result["stale_card_risks"]:
        print(
            "STALE_CARD_RISK",
            f"current_cluster={risk['current_cluster_id']}",
            f"card_count={risk['card_count']}",
            f"cards={risk['card_ids']}",
        )
        print(f"  titles={risk['titles']}")


def _rate(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return value / total


if __name__ == "__main__":
    main()
