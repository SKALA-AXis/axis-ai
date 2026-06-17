"""Conservative post-processing for singleton/small news clusters.

The script merges singleton/small clusters into nearby larger clusters, and can
also consolidate multiple singleton/small clusters into one cluster when they
clearly describe the same concrete event. It intentionally never merges by
company or sector alone.

Examples:
  uv run python scripts/postprocess_singleton_clusters.py --env cloud --lookback-hours 24
  uv run python scripts/postprocess_singleton_clusters.py --env cloud --lookback-hours 24 --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.companies import COMPANY_ALIASES  # noqa: E402
from src.config.env_loader import load_profile  # noqa: E402
from src.config.openai_policy import openai_calls_enabled  # noqa: E402
from src.db.postgres import SessionLocal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("postprocess_singleton_clusters")


_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
_STOCK_NOISE_RE = re.compile(
    r"주가|특징주|목표가|목표\s*주가|투자의견|테마주|급등|급락|상한가|하한가|증시|"
    r"지수선물|옵션|시황|강세|약세|반등|폭등|장중|시가총액|거래량"
)
_LIST_LIKE_RE = re.compile(
    r"클라우드\s*월드|ai\s*브리프|it\s*스냅샷|전자\s*it\s*레이더|테크\s*&?\s*나우|tech\s*&?\s*now",
    re.I,
)
_LIST_LIKE_COMPACT_MARKERS = (
    "뉴스브리프",
    "뉴스브리핑",
    "ai브리프",
    "it브리프",
    "it스냅샷",
    "전자it레이더",
    "시큐리티포커스",
    "테크앤나우",
    "technow",
    "클라우드월드",
)
_MOJIBAKE_REPLACEMENT_CHAR = "\ufffd"
_TITLE_LLM_JUDGE_ENABLED = (
    os.getenv("POSTPROCESS_TITLE_LLM_JUDGE_ENABLED", "true").lower() == "true"
)
_TITLE_LLM_MAX_CALLS = int(os.getenv("POSTPROCESS_TITLE_LLM_MAX_CALLS", "80"))
_TITLE_LLM_MODEL = os.getenv("POSTPROCESS_TITLE_LLM_MODEL", "gpt-4o-mini")
_TITLE_LLM_MERGE_SCORE = float(os.getenv("POSTPROCESS_TITLE_LLM_MERGE_SCORE", "0.82"))
_title_llm_calls = 0
_title_llm_cache: dict[tuple[str, str], bool | None] = {}


@dataclass(frozen=True)
class SourceCluster:
    cluster_id: int
    article_ids: list[int]
    titles: list[str]
    title: str
    article_count: int
    event_at: datetime | None
    snippets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Cluster:
    cluster_id: int
    article_count: int
    titles: list[str]
    article_ids: list[int]
    latest_event_at: datetime | None
    snippets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MergeCandidate:
    source: SourceCluster
    target: Cluster
    event_key: str
    shared_tokens: set[str]
    score: float


@dataclass(frozen=True)
class GroupMergeCandidate:
    target: SourceCluster
    sources: list[SourceCluster]
    event_key: str
    shared_tokens: set[str]
    score: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process singleton news clusters")
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument(
        "--time-field",
        choices=["published_at", "collected_at"],
        default="published_at",
        help="Window/time-gap column. Default: published_at.",
    )
    parser.add_argument("--source-type", default="news")
    parser.add_argument(
        "--max-source-size",
        type=int,
        default=0,
        help="Maximum source cluster size. Use 0 to consider every cluster as a merge source.",
    )
    parser.add_argument("--min-target-size", type=int, default=2)
    parser.add_argument(
        "--min-new-cluster-size",
        type=int,
        default=2,
        help="Minimum article count for source-to-source consolidation.",
    )
    parser.add_argument("--max-time-gap-hours", type=int, default=72)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument(
        "--skip-noise", action="store_true", help="Skip stock/list-like processed news rows"
    )
    parser.add_argument("--apply", action="store_true", help="Actually update DB")
    parser.add_argument("--limit", type=int, default=0, help="Limit singleton candidates")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    log.info(
        "singleton 후처리 시작 | profile=%s lookback_hours=%d time_field=%s apply=%s",
        profile,
        args.lookback_hours,
        args.time_field,
        args.apply,
    )

    with SessionLocal() as db:
        result = run_postprocess(
            db=db,
            source_type=args.source_type,
            lookback_hours=args.lookback_hours,
            time_field=args.time_field,
            max_source_size=args.max_source_size,
            min_target_size=args.min_target_size,
            min_new_cluster_size=args.min_new_cluster_size,
            max_time_gap_hours=args.max_time_gap_hours,
            min_score=args.min_score,
            apply=args.apply,
            skip_noise=args.skip_noise,
            limit=args.limit,
        )
        if args.apply:
            db.commit()

        print("SINGLETON_POSTPROCESS_DRY_RUN" if not args.apply else "SINGLETON_POSTPROCESS_APPLY")
        print(
            f"clusters={result['cluster_count']} "
            f"sources={result['source_count']} "
            f"targets={result['target_count']} "
            f"time_field={args.time_field}"
        )
        print(
            f"merge_candidates={len(result['candidates'])} "
            f"group_merge_candidates={len(result['group_candidates'])} "
            f"noise_candidates={len(result['noise_ids'])}"
        )
        candidates = result["candidates"]
        group_candidates = result["group_candidates"]
        for candidate in candidates:
            print(
                "MERGE",
                f"article_ids={candidate.source.article_ids}",
                f"from_cluster={candidate.source.cluster_id}",
                f"source_count={candidate.source.article_count}",
                f"to_cluster={candidate.target.cluster_id}",
                f"target_count={candidate.target.article_count}",
                f"event_key={candidate.event_key}",
                f"score={candidate.score:.3f}",
                f"shared={','.join(sorted(candidate.shared_tokens))}",
            )
            print(f"  source_titles={candidate.source.titles}")
            print(f"  target_titles={candidate.target.titles[:3]}")
        for candidate in group_candidates:
            source_clusters = [source.cluster_id for source in candidate.sources]
            article_ids = [
                article_id for source in candidate.sources for article_id in source.article_ids
            ]
            print(
                "GROUP_MERGE",
                f"article_ids={article_ids}",
                f"from_clusters={source_clusters}",
                f"to_cluster={candidate.target.cluster_id}",
                f"event_key={candidate.event_key}",
                f"score={candidate.score:.3f}",
                f"shared={','.join(sorted(candidate.shared_tokens))}",
            )
            print(f"  target_titles={candidate.target.titles[:3]}")
            for source in candidate.sources:
                if source.cluster_id != candidate.target.cluster_id:
                    print(f"  source_titles={source.titles}")

        print(f"updated={result['updated']}")
        print(f"group_updated={result['group_updated']}")
        print(f"noise_updated={result['noise_updated']}")
        print(f"card_cluster_id_updated={result['card_cluster_id_updated']}")
        print(f"card_source_payload_updated={result['card_source_payload_updated']}")
        print(f"card_stale_deleted={result['card_stale_deleted']}")


def run_postprocess(
    *,
    db: Any,
    source_type: str = "news",
    lookback_hours: int = 2,
    time_field: str = "published_at",
    max_source_size: int = 0,
    min_target_size: int = 4,
    min_new_cluster_size: int = 2,
    max_time_gap_hours: int = 72,
    min_score: float = 0.45,
    apply: bool = False,
    skip_noise: bool = True,
    limit: int = 0,
) -> dict[str, Any]:
    clusters = _load_clusters(db, source_type, lookback_hours, time_field)
    sources, targets = _split_clusters(
        clusters,
        max_source_size=max_source_size,
        min_target_size=min_target_size,
    )
    if limit:
        sources = sources[:limit]

    candidates = _find_candidates(
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
    noise_ids = _find_noise_ids(db, source_type, lookback_hours, time_field) if skip_noise else []

    updated = 0
    group_updated = 0
    noise_updated = 0
    if apply:
        noise_updated = _apply_noise_skips(db, noise_ids)
        updated = _apply_candidates(db, candidates)
        group_updated = _apply_group_candidates(
            db,
            _filter_group_candidates_after_target_merges(group_candidates, candidates),
        )
        card_sync = _sync_card_news_with_current_clusters(db)
    else:
        card_sync = {"cluster_id_updated": 0, "source_payload_updated": 0, "stale_deleted": 0}

    return {
        "cluster_count": len(clusters),
        "source_count": len(sources),
        "target_count": len(targets),
        "candidates": candidates,
        "group_candidates": group_candidates,
        "noise_ids": noise_ids,
        "updated": updated,
        "group_updated": group_updated,
        "noise_updated": noise_updated,
        "card_cluster_id_updated": card_sync["cluster_id_updated"],
        "card_source_payload_updated": card_sync["source_payload_updated"],
        "card_stale_deleted": card_sync["stale_deleted"],
    }


def _load_clusters(
    db: Any, source_type: str, lookback_hours: int, time_field: str
) -> list[Cluster]:
    order_field = "published_at" if time_field == "published_at" else "collected_at"
    aliased_order_clause = f"ra.{order_field} DESC NULLS LAST, ra.collected_at DESC, ra.id DESC"
    rows = db.execute(
        text(
            f"""
            WITH recent_clusters AS (
                SELECT DISTINCT cluster_id
                FROM raw_articles
                WHERE source_type = :source_type
                  AND {order_field} >= now() - (:lookback_hours * interval '1 hour')
                  AND processing_status = 'PROCESSED'
                  AND relevance_label = 'relevant'
                  AND cluster_id IS NOT NULL
            )
            SELECT
                ra.cluster_id,
                COUNT(*) AS article_count,
                ARRAY_AGG(ra.id ORDER BY {aliased_order_clause}) AS article_ids,
                ARRAY_AGG(ra.title ORDER BY {aliased_order_clause}) AS titles,
                ARRAY_AGG(COALESCE(ra.content, '') ORDER BY {aliased_order_clause}) AS contents,
                MAX(ra.{order_field}) AS latest_event_at
            FROM raw_articles ra
            JOIN recent_clusters rc ON rc.cluster_id = ra.cluster_id
            WHERE ra.source_type = :source_type
              AND ra.processing_status = 'PROCESSED'
              AND ra.relevance_label = 'relevant'
              AND ra.cluster_id IS NOT NULL
            GROUP BY ra.cluster_id
            """
        ),
        {"source_type": source_type, "lookback_hours": lookback_hours},
    ).mappings()
    return [
        Cluster(
            cluster_id=int(row["cluster_id"]),
            article_count=int(row["article_count"]),
            article_ids=[int(value) for value in row["article_ids"]],
            titles=[str(value or "") for value in row["titles"]],
            snippets=[_snippet(value) for value in row["contents"]],
            latest_event_at=row["latest_event_at"],
        )
        for row in rows
    ]


def _split_clusters(
    clusters: list[Cluster],
    *,
    max_source_size: int,
    min_target_size: int,
) -> tuple[list[SourceCluster], list[Cluster]]:
    sources: list[SourceCluster] = []
    targets: list[Cluster] = []
    for cluster in clusters:
        if max_source_size <= 0 or cluster.article_count <= max_source_size:
            sources.append(
                SourceCluster(
                    cluster_id=cluster.cluster_id,
                    article_ids=cluster.article_ids,
                    titles=cluster.titles,
                    snippets=cluster.snippets,
                    title=cluster.titles[0],
                    article_count=cluster.article_count,
                    event_at=cluster.latest_event_at,
                )
            )
        if cluster.article_count >= min_target_size:
            targets.append(cluster)
    return sources, targets


def _find_candidates(
    *,
    sources: list[SourceCluster],
    targets: list[Cluster],
    max_time_gap_hours: int,
    min_score: float,
) -> list[MergeCandidate]:
    candidates: list[MergeCandidate] = []
    for source in sources:
        if any(_is_stock_noise(title) for title in source.titles) or any(
            _is_list_like(title) for title in source.titles
        ):
            continue

        scored: list[MergeCandidate] = []
        for target in targets:
            if target.cluster_id == source.cluster_id:
                continue
            if not _within_time_gap(source.event_at, target.latest_event_at, max_time_gap_hours):
                continue
            relation = _cluster_relation(
                source.titles,
                target.titles,
                target.article_count,
                source.snippets,
                target.snippets,
            )
            if relation is None:
                continue
            event_key, shared_tokens, score = relation
            if source.article_count > 1 and not _is_verified_event_key(event_key):
                continue
            if score < min_score:
                continue
            scored.append(
                MergeCandidate(
                    source=source,
                    target=target,
                    event_key=event_key,
                    shared_tokens=shared_tokens,
                    score=score,
                )
            )

        scored.sort(key=lambda item: (item.score, item.target.article_count), reverse=True)
        if not scored:
            continue
        best = scored[0]
        if (
            len(scored) >= 2
            and best.score - scored[1].score < 0.08
            and best.event_key != "title_content_llm"
        ):
            log.info(
                "ambiguous small-cluster merge skipped | cluster_id=%s best=%s second=%s",
                source.cluster_id,
                best.target.cluster_id,
                scored[1].target.cluster_id,
            )
            continue
        candidates.append(best)
    return candidates


def _find_source_group_candidates(
    *,
    sources: list[SourceCluster],
    max_time_gap_hours: int,
    min_score: float,
    min_new_cluster_size: int,
) -> list[GroupMergeCandidate]:
    eligible = [
        source
        for source in sources
        if not any(_is_stock_noise(title) for title in source.titles)
        and not any(_is_list_like(title) for title in source.titles)
    ]
    if len(eligible) < 2:
        return []

    parent = {source.cluster_id: source.cluster_id for source in eligible}
    relation_by_pair: dict[frozenset[int], tuple[str, set[str], float]] = {}

    def find(cluster_id: int) -> int:
        while parent[cluster_id] != cluster_id:
            parent[cluster_id] = parent[parent[cluster_id]]
            cluster_id = parent[cluster_id]
        return cluster_id

    def union(left: int, right: int) -> None:
        parent[find(left)] = find(right)

    for i, left in enumerate(eligible):
        for right in eligible[i + 1 :]:
            if not _within_time_gap(left.event_at, right.event_at, max_time_gap_hours):
                continue
            relation = _cluster_relation(
                left.titles,
                right.titles,
                max(left.article_count, right.article_count),
                left.snippets,
                right.snippets,
            )
            if relation is None:
                continue
            event_key, shared_tokens, score = relation
            if not _is_verified_event_key(event_key):
                continue
            if score < min_score:
                continue
            relation_by_pair[frozenset({left.cluster_id, right.cluster_id})] = (
                event_key,
                shared_tokens,
                score,
            )
            union(left.cluster_id, right.cluster_id)

    groups: dict[int, list[SourceCluster]] = {}
    for source in eligible:
        groups.setdefault(find(source.cluster_id), []).append(source)

    candidates: list[GroupMergeCandidate] = []
    for members in groups.values():
        total_articles = sum(member.article_count for member in members)
        if len(members) < 2 or total_articles < min_new_cluster_size:
            continue
        members.sort(
            key=lambda item: (item.event_at is not None, item.event_at, item.cluster_id),
            reverse=True,
        )
        target = members[0]
        pairs = [
            relation_by_pair[pair]
            for i, left in enumerate(members)
            for right in members[i + 1 :]
            if (pair := frozenset({left.cluster_id, right.cluster_id})) in relation_by_pair
        ]
        if not pairs:
            continue
        event_key = sorted({pair[0] for pair in pairs})[0]
        shared_tokens = set().union(*(pair[1] for pair in pairs))
        score = sum(pair[2] for pair in pairs) / len(pairs)
        candidates.append(
            GroupMergeCandidate(
                target=target,
                sources=members,
                event_key=event_key,
                shared_tokens=shared_tokens,
                score=score,
            )
        )
    return candidates


def _apply_candidates(db: Any, candidates: list[MergeCandidate]) -> int:
    updated = 0
    affected_clusters: set[int] = set()
    for candidate in candidates:
        result = db.execute(
            text(
                """
                UPDATE raw_articles
                SET cluster_id = :target_cluster_id,
                    is_representative = FALSE
                WHERE id = ANY(:article_ids)
                  AND cluster_id = :source_cluster_id
                  AND processing_status = 'PROCESSED'
                  AND relevance_label = 'relevant'
                """
            ),
            {
                "article_ids": candidate.source.article_ids,
                "source_cluster_id": candidate.source.cluster_id,
                "target_cluster_id": candidate.target.cluster_id,
            },
        )
        updated += int(getattr(result, "rowcount", 0) or 0)
        affected_clusters.add(candidate.target.cluster_id)
    _reset_representatives(db, sorted(affected_clusters))
    return updated


def _filter_group_candidates_after_target_merges(
    group_candidates: list[GroupMergeCandidate],
    merge_candidates: list[MergeCandidate],
) -> list[GroupMergeCandidate]:
    target_merged_source_ids = {candidate.source.cluster_id for candidate in merge_candidates}
    if not target_merged_source_ids:
        return group_candidates

    filtered: list[GroupMergeCandidate] = []
    for candidate in group_candidates:
        source_ids = {source.cluster_id for source in candidate.sources}
        if source_ids & target_merged_source_ids:
            log.info(
                "small-cluster group merge skipped after target merge | target=%s sources=%s",
                candidate.target.cluster_id,
                sorted(source_ids),
            )
            continue
        filtered.append(candidate)
    return filtered


def _apply_group_candidates(db: Any, candidates: list[GroupMergeCandidate]) -> int:
    updated = 0
    affected_clusters: set[int] = set()
    for candidate in candidates:
        source_cluster_ids = [
            source.cluster_id
            for source in candidate.sources
            if source.cluster_id != candidate.target.cluster_id
        ]
        if not source_cluster_ids:
            continue
        result = db.execute(
            text(
                """
                UPDATE raw_articles
                SET cluster_id = :target_cluster_id,
                    is_representative = FALSE
                WHERE cluster_id = ANY(:source_cluster_ids)
                  AND processing_status = 'PROCESSED'
                  AND relevance_label = 'relevant'
                """
            ),
            {
                "source_cluster_ids": source_cluster_ids,
                "target_cluster_id": candidate.target.cluster_id,
            },
        )
        updated += int(getattr(result, "rowcount", 0) or 0)
        affected_clusters.add(candidate.target.cluster_id)
    _reset_representatives(db, sorted(affected_clusters))
    return updated


def _find_noise_ids(db: Any, source_type: str, lookback_hours: int, time_field: str) -> list[int]:
    order_field = "published_at" if time_field == "published_at" else "collected_at"
    rows = db.execute(
        text(
            f"""
            SELECT id, title, content
            FROM raw_articles
            WHERE source_type = :source_type
              AND {order_field} >= now() - (:lookback_hours * interval '1 hour')
              AND processing_status = 'PROCESSED'
              AND relevance_label = 'relevant'
            """
        ),
        {
            "source_type": source_type,
            "lookback_hours": lookback_hours,
        },
    ).mappings()
    return [
        int(row["id"])
        for row in rows
        if _is_stock_noise(str(row.get("title") or ""))
        or _is_list_like(str(row.get("title") or ""))
        or _has_mojibake_content(str(row.get("content") or ""))
    ]


def _apply_noise_skips(db: Any, article_ids: list[int]) -> int:
    if not article_ids:
        return 0
    result = db.execute(
        text(
            """
            UPDATE raw_articles
            SET processing_status = 'SKIPPED',
                relevance_label = 'irrelevant',
                relevance_score = LEAST(COALESCE(relevance_score, 0.25), 0.25),
                relevance_reason = :relevance_reason,
                cluster_id = NULL,
                is_representative = FALSE
            WHERE id = ANY(:article_ids)
              AND processing_status = 'PROCESSED'
              AND relevance_label = 'relevant'
            """
        ),
        {
            "article_ids": article_ids,
            "relevance_reason": (
                "주가/특징주/목록형/본문 파싱 품질 문제로 전략 이벤트 통합 재료에서 제외"
            ),
        },
    )
    return int(getattr(result, "rowcount", 0) or 0)


def _sync_card_news_with_current_clusters(db: Any) -> dict[str, int]:
    if not _table_exists(db, "card_news"):
        return {"cluster_id_updated": 0, "source_payload_updated": 0, "stale_deleted": 0}

    cluster_update = db.execute(
        text(
            """
            WITH card_sources AS (
                SELECT
                    cn.id,
                    cn.cluster_id,
                    cardinality(COALESCE(cn.source_raw_article_ids, ARRAY[]::bigint[]))
                        AS source_id_count,
                    COUNT(ra.id) AS resolved_count,
                    COUNT(ra.cluster_id) AS resolved_cluster_count,
                    ARRAY_AGG(DISTINCT ra.cluster_id ORDER BY ra.cluster_id)
                        FILTER (WHERE ra.cluster_id IS NOT NULL) AS source_clusters
                FROM card_news cn
                LEFT JOIN LATERAL unnest(
                    COALESCE(cn.source_raw_article_ids, ARRAY[]::bigint[])
                ) source_id(id) ON TRUE
                LEFT JOIN raw_articles ra ON ra.id = source_id.id
                WHERE cn.status = 'ACTIVE'
                GROUP BY cn.id, cn.cluster_id, cn.source_raw_article_ids
            ),
            fixes AS (
                SELECT id, source_clusters[1] AS current_cluster_id
                FROM card_sources
                WHERE source_clusters IS NOT NULL
                  AND cardinality(source_clusters) = 1
            )
            UPDATE card_news cn
            SET cluster_id = fixes.current_cluster_id
            FROM fixes
            WHERE cn.id = fixes.id
              AND cn.status = 'ACTIVE'
              AND cn.cluster_id <> fixes.current_cluster_id
            """
        )
    )

    stale_update = db.execute(
        text(
            """
            WITH card_sources AS (
                SELECT
                    cn.id,
                    cardinality(COALESCE(cn.source_raw_article_ids, ARRAY[]::bigint[]))
                        AS source_id_count,
                    COUNT(ra.id) AS resolved_count,
                    COUNT(ra.cluster_id) AS resolved_cluster_count,
                    ARRAY_AGG(DISTINCT ra.cluster_id ORDER BY ra.cluster_id)
                        FILTER (WHERE ra.cluster_id IS NOT NULL) AS source_clusters
                FROM card_news cn
                LEFT JOIN LATERAL unnest(
                    COALESCE(cn.source_raw_article_ids, ARRAY[]::bigint[])
                ) source_id(id) ON TRUE
                LEFT JOIN raw_articles ra ON ra.id = source_id.id
                WHERE cn.status = 'ACTIVE'
                GROUP BY cn.id, cn.source_raw_article_ids
            ),
            stale AS (
                SELECT id
                FROM card_sources
                WHERE source_id_count = 0
                   OR cardinality(source_clusters) > 1
                   OR (resolved_count > 0 AND resolved_cluster_count = 0)
            )
            UPDATE card_news cn
            SET status = 'DELETED'
            FROM stale
            WHERE cn.id = stale.id
              AND cn.status = 'ACTIVE'
            """
        )
    )

    source_update = db.execute(
        text(
            """
            WITH active_clusters AS (
                SELECT DISTINCT cluster_id
                FROM card_news
                WHERE status = 'ACTIVE'
            ),
            ranked_articles AS (
                SELECT
                    ra.cluster_id,
                    ra.id,
                    ra.title,
                    ra.url,
                    ra.source_name,
                    ra.publisher,
                    ra.published_at,
                    ra.collected_at,
                    COALESCE(ra.metadata, '{}'::jsonb) AS metadata,
                    ROW_NUMBER() OVER (
                        PARTITION BY ra.cluster_id
                        ORDER BY
                            ra.published_at DESC NULLS LAST,
                            ra.collected_at DESC NULLS LAST,
                            ra.id DESC
                    ) AS rn
                FROM raw_articles ra
                JOIN active_clusters ac ON ac.cluster_id = ra.cluster_id
                WHERE ra.processing_status = 'PROCESSED'
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
                            'collected_at', collected_at,
                            'image_urls',
                                CASE
                                    WHEN jsonb_typeof(metadata -> 'image_urls') = 'array'
                                    THEN metadata -> 'image_urls'
                                    ELSE '[]'::jsonb
                                END
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
                            'collected_at', collected_at,
                            'image_urls',
                                CASE
                                    WHEN jsonb_typeof(metadata -> 'image_urls') = 'array'
                                    THEN metadata -> 'image_urls'
                                    ELSE '[]'::jsonb
                                END
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
              AND (
                  cn.source_raw_article_ids IS DISTINCT FROM cs.raw_ids
                  OR cn.sources IS DISTINCT FROM cs.sources
                  OR cn.source_articles IS DISTINCT FROM cs.source_articles
              )
            """
        )
    )

    return {
        "cluster_id_updated": int(getattr(cluster_update, "rowcount", 0) or 0),
        "source_payload_updated": int(getattr(source_update, "rowcount", 0) or 0),
        "stale_deleted": int(getattr(stale_update, "rowcount", 0) or 0),
    }


def _table_exists(db: Any, table_name: str) -> bool:
    return bool(
        db.execute(text("SELECT to_regclass(:table_name)"), {"table_name": table_name}).scalar()
    )


def _reset_representatives(db: Any, cluster_ids: list[int]) -> None:
    if not cluster_ids:
        return
    db.execute(
        text(
            """
            UPDATE raw_articles
            SET is_representative = FALSE
            WHERE cluster_id = ANY(:cluster_ids)
              AND processing_status = 'PROCESSED'
              AND relevance_label = 'relevant'
            """
        ),
        {"cluster_ids": cluster_ids},
    )
    db.execute(
        text(
            """
            WITH reps AS (
                SELECT DISTINCT ON (cluster_id) id
                FROM raw_articles
                WHERE cluster_id = ANY(:cluster_ids)
                  AND processing_status = 'PROCESSED'
                  AND relevance_label = 'relevant'
                ORDER BY cluster_id, published_at DESC NULLS LAST, collected_at DESC, id DESC
            )
            UPDATE raw_articles r
            SET is_representative = TRUE
            FROM reps
            WHERE r.id = reps.id
            """
        ),
        {"cluster_ids": cluster_ids},
    )


def _cluster_relation(
    left_titles: list[str],
    right_titles: list[str],
    target_size: int,
    left_snippets: list[str] | None = None,
    right_snippets: list[str] | None = None,
) -> tuple[str, set[str], float] | None:
    left_tokens = set().union(*(_event_tokens(title) for title in left_titles))
    right_tokens = set().union(*(_event_tokens(title) for title in right_titles))
    shared_tokens = left_tokens & right_tokens
    left_context_tokens = _cluster_context_tokens(left_titles, left_snippets)
    right_context_tokens = _cluster_context_tokens(right_titles, right_snippets)
    shared_context_tokens = left_context_tokens & right_context_tokens
    left_title_key = _title_event_issue_key(left_titles)
    right_title_key = _title_event_issue_key(right_titles)

    if left_title_key and left_title_key == right_title_key:
        score = max(
            _candidate_score(left_tokens, right_tokens, shared_tokens, target_size),
            0.70,
        )
        return f"title_event:{left_title_key}", shared_tokens, score

    title_llm_decision = _title_llm_same_event(
        left_titles,
        right_titles,
        shared_context_tokens,
        left_snippets,
        right_snippets,
    )
    if title_llm_decision is True:
        score = max(
            _candidate_score(
                left_context_tokens, right_context_tokens, shared_context_tokens, target_size
            ),
            _TITLE_LLM_MERGE_SCORE,
        )
        return "title_content_llm", shared_context_tokens, score

    if not _same_company_family(left_titles, right_titles):
        return None
    if not (_has_event_action(left_titles) and _has_event_action(right_titles)):
        return None

    event_shared_tokens = shared_tokens - _company_anchor_tokens()
    if len(event_shared_tokens) < 2 and not _has_high_signal_single_token(event_shared_tokens):
        return None
    if len(event_shared_tokens) < 3 and not (
        _has_distinctive_shared_tokens(event_shared_tokens)
        or _has_high_signal_single_token(event_shared_tokens)
    ):
        return None

    score = _candidate_score(left_tokens, right_tokens, event_shared_tokens, target_size)
    if _has_high_signal_single_token(event_shared_tokens):
        score = max(score, 0.45)
    event_key = "generic:" + "_".join(sorted(event_shared_tokens)[:4])
    return event_key, event_shared_tokens, score


def _is_verified_event_key(event_key: str) -> bool:
    return event_key.startswith("title_event:") or event_key == "title_content_llm"


def _event_tokens(title: str) -> set[str]:
    tokens = {_normalize_token(token) for token in _TOKEN_RE.findall(title.lower())}
    return {token for token in tokens if _useful_token(token)}


def _same_company_family(left_titles: list[str], right_titles: list[str]) -> bool:
    return bool(_company_families(left_titles) & _company_families(right_titles))


def _company_families(titles: list[str]) -> set[str]:
    compact = " ".join(_compact(title) for title in titles)
    families: set[str] = set()
    for company_id, aliases in COMPANY_ALIASES.items():
        markers = {_compact(company_id), *(_compact(alias) for alias in aliases)}
        if any(marker and marker in compact for marker in markers):
            families.add(company_id)
    return families


def _company_anchor_tokens() -> set[str]:
    tokens: set[str] = set()
    for company_id, aliases in COMPANY_ALIASES.items():
        tokens.add(_normalize_token(company_id))
        tokens.update(_normalize_token(alias) for alias in aliases)
    return {token for token in tokens if token}


def _has_event_action(titles: list[str]) -> bool:
    compact = " ".join(_compact(title) for title in titles)
    return any(
        marker in compact
        for marker in (
            "계약",
            "도입",
            "체결",
            "협력",
            "맞손",
            "수주",
            "선정",
            "출시",
            "공개",
            "공급",
            "구축",
            "투자",
            "인수",
            "확대",
        )
    )


def _has_distinctive_shared_tokens(shared_tokens: set[str]) -> bool:
    if len(shared_tokens) < 2:
        return False
    distinctive = [
        token
        for token in shared_tokens
        if any(char.isascii() and char.isalpha() for char in token) or len(token) >= 5
    ]
    return len(distinctive) >= 2


def _has_high_signal_single_token(shared_tokens: set[str]) -> bool:
    return any(
        len(token) >= 5 or any(char.isascii() and char.isalpha() for char in token)
        for token in shared_tokens
    )


def _candidate_score(
    singleton_tokens: set[str],
    target_tokens: set[str],
    shared_tokens: set[str],
    target_size: int,
) -> float:
    coverage = len(shared_tokens) / max(1, len(singleton_tokens))
    jaccard = len(shared_tokens) / max(1, len(singleton_tokens | target_tokens))
    size_bonus = min(target_size, 10) / 100
    return coverage * 0.7 + jaccard * 0.2 + size_bonus


def _within_time_gap(
    left: datetime | None,
    right: datetime | None,
    max_hours: int,
) -> bool:
    if left is None or right is None:
        return True
    return abs((left - right).total_seconds()) <= max_hours * 3600


def _is_stock_noise(title: str) -> bool:
    return bool(_STOCK_NOISE_RE.search(title or ""))


def _is_list_like(title: str) -> bool:
    compact = _compact(title)
    if any(marker in compact for marker in _LIST_LIKE_COMPACT_MARKERS):
        return True
    if re.match(r"^\[?#?[가-힣a-z0-9]*(?:포커스|레이더|브리프|스냅샷)\]?", compact):
        return True
    return bool(_LIST_LIKE_RE.search(title or ""))


def _has_mojibake_content(content: str) -> bool:
    text = str(content or "")
    if len(text) < 80:
        return False
    replacement_count = text.count(_MOJIBAKE_REPLACEMENT_CHAR)
    if replacement_count < 8:
        return False
    replacement_ratio = replacement_count / max(len(text), 1)
    return replacement_ratio >= 0.01 or _MOJIBAKE_REPLACEMENT_CHAR * 4 in text


def _normalize_token(token: str) -> str:
    compact = _strip_korean_particle(_compact(token))
    aliases = {
        "엘지씨엔에스": "lgcns",
        "lg씨엔에스": "lgcns",
        "삼성에스디에스": "samsungsds",
        "삼성sds": "samsungsds",
        "오픈ai": "openai",
        "챗gpt": "chatgpt",
        "챗지피티": "chatgpt",
        "피지컬": "physical",
        "에이전틱": "agentic",
        "앤트로픽": "anthropic",
        "클로드": "claude",
        "한전": "한국전력",
        "한국전력공사": "한국전력",
    }
    return aliases.get(compact, compact)


def _title_llm_same_event(
    left_titles: list[str],
    right_titles: list[str],
    shared_tokens: set[str],
    left_snippets: list[str] | None = None,
    right_snippets: list[str] | None = None,
) -> bool | None:
    global _title_llm_calls

    if not _should_consult_title_llm(
        left_titles, right_titles, shared_tokens, left_snippets, right_snippets
    ):
        return None

    left_key = " | ".join(sorted(left_titles))
    right_key = " | ".join(sorted(right_titles))
    cache_key: tuple[str, str] = (
        (left_key, right_key) if left_key <= right_key else (right_key, left_key)
    )
    if cache_key in _title_llm_cache:
        return _title_llm_cache[cache_key]
    if _title_llm_calls >= _TITLE_LLM_MAX_CALLS:
        _title_llm_cache[cache_key] = None
        return None

    _title_llm_calls += 1
    payload = {
        "instruction": (
            "Decide whether these Korean news article groups describe the same narrow business "
            "news issue and should be merged into one cluster. Use titles, lead snippets, and "
            "shared keywords; ignore sector labels."
        ),
        "criteria": [
            "same_event=true when titles use different wording for the same underlying "
            "announcement, deal, launch, deployment, or collaboration.",
            "same_event=true when title wording differs but lead snippets and keywords "
            "indicate the same concrete event.",
            "same_event=true when both articles are focused analysis/strategy coverage "
            "of the same company and the same narrow set of anchors such as product "
            "line, market, executive, partner, financial figures, or operational "
            "initiative.",
            "same_event=false when articles share only a broad theme, company, "
            "technology category, earnings season, or industry trend.",
            "same_event=false when the titles have different concrete anchors, even "
            "if snippets share broad AI, cloud, platform, or business terms.",
            "Do not require identical company labels if the titles indicate the same event.",
        ],
        "left_titles": left_titles[:5],
        "right_titles": right_titles[:5],
        "left_lead_snippets": (left_snippets or [])[:5],
        "right_lead_snippets": (right_snippets or [])[:5],
        "shared_keywords": sorted(shared_tokens)[:30],
        "required_json_schema": {
            "same_event": "boolean",
            "confidence": "number between 0 and 1",
            "reason": "short Korean explanation",
        },
    }

    try:
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model=_TITLE_LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict Korean news title clustering judge. Return JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        confidence = float(parsed.get("confidence") or 0)
        decision = bool(parsed.get("same_event")) and confidence >= 0.7
        if decision and not (
            confidence >= 0.85
            or _has_title_anchor_overlap(left_titles, right_titles)
            or _has_context_anchor_overlap(left_titles, right_titles, left_snippets, right_snippets)
        ):
            log.info(
                "title LLM merge vetoed by title anchors | reason=%s",
                parsed.get("reason", ""),
            )
            decision = False
        _title_llm_cache[cache_key] = decision
        log.info(
            "title LLM merge judge | decision=%s confidence=%s reason=%s",
            decision,
            confidence,
            parsed.get("reason", ""),
        )
        return decision
    except Exception as e:
        log.warning("title LLM merge judge failed | error=%s", e)
        _title_llm_cache[cache_key] = None
        return None


def _should_consult_title_llm(
    left_titles: list[str],
    right_titles: list[str],
    shared_tokens: set[str],
    left_snippets: list[str] | None = None,
    right_snippets: list[str] | None = None,
) -> bool:
    if not _TITLE_LLM_JUDGE_ENABLED or not openai_calls_enabled():
        return False
    if _TITLE_LLM_MAX_CALLS <= 0:
        return False
    same_company = _same_company_family(left_titles, right_titles)
    has_event_action = _has_event_action(left_titles) or _has_event_action(right_titles)
    has_context_overlap = _has_context_anchor_overlap(
        left_titles,
        right_titles,
        left_snippets,
        right_snippets,
    )
    if not (has_event_action or same_company and has_context_overlap):
        return False

    left_tokens = _cluster_context_tokens(left_titles, left_snippets)
    right_tokens = _cluster_context_tokens(right_titles, right_snippets)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    if len(shared_tokens) >= 2:
        return True

    jaccard = len(shared_tokens) / max(1, len(left_tokens | right_tokens))
    coverage = len(shared_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return jaccard >= 0.25 or coverage >= 0.40


def _has_context_anchor_overlap(
    left_titles: list[str],
    right_titles: list[str],
    left_snippets: list[str] | None = None,
    right_snippets: list[str] | None = None,
) -> bool:
    left_tokens = _cluster_context_tokens(left_titles, left_snippets) - _company_anchor_tokens()
    right_tokens = _cluster_context_tokens(right_titles, right_snippets) - _company_anchor_tokens()
    shared = left_tokens & right_tokens
    if len(shared) < 3:
        return False
    distinctive = {
        token
        for token in shared
        if len(token) >= 3
        and token not in {"ai", "ax", "dx", "시장", "사업", "기업", "기술", "서비스"}
    }
    return len(distinctive) >= 2


def _cluster_context_tokens(titles: list[str], snippets: list[str] | None = None) -> set[str]:
    text_values = [*titles, *(snippets or [])]
    tokens = set().union(*(_event_tokens(value) for value in text_values if value))
    return {token for token in tokens if _context_token_useful(token)}


def _has_title_anchor_overlap(left_titles: list[str], right_titles: list[str]) -> bool:
    left_tokens = _title_anchor_tokens(left_titles)
    right_tokens = _title_anchor_tokens(right_titles)
    shared = left_tokens & right_tokens
    if any(len(token) >= 4 for token in shared):
        return True
    if _has_high_signal_single_token(shared):
        return True
    if len(shared) >= 2:
        return True

    coverage = len(shared) / max(1, min(len(left_tokens), len(right_tokens)))
    return coverage >= 0.35 and bool(shared)


def _title_anchor_tokens(titles: list[str]) -> set[str]:
    tokens = set().union(*(_event_tokens(title) for title in titles))
    broad = {
        "ai",
        "ax",
        "it",
        "사업",
        "사업자",
        "기업",
        "기술",
        "시장",
        "서비스",
        "솔루션",
        "플랫폼",
        "클라우드",
        "디지털",
        "공공",
        "정부",
        "구축",
        "확대",
        "강화",
        "나선다",
        "본격화",
        "전환",
        "인공지능",
    }
    return {token for token in tokens if token not in broad and len(token) >= 2}


def _context_token_useful(token: str) -> bool:
    if len(token) < 2 or token.isdigit():
        return False
    generic = {
        "기자",
        "뉴스",
        "사진",
        "제공",
        "관련",
        "통해",
        "대한",
        "이번",
        "밝혔다",
        "설명했다",
        "대표",
        "회장",
        "부사장",
        "산업",
        "시장",
        "기업",
        "기술",
        "사업",
    }
    return token not in generic


def _snippet(value: Any, max_chars: int = 360) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def _title_event_issue_key(titles: list[str]) -> str:
    tokens = set().union(*(_event_tokens(title) for title in titles))
    anchors = sorted(token for token in tokens if _is_title_event_key_token(token))
    if len(anchors) >= 3 and _has_event_action(titles):
        return "_".join(anchors[:5])

    return ""


def _is_title_event_key_token(token: str) -> bool:
    if len(token) < 3:
        return False
    return token not in {"기업용", "서비스", "플랫폼", "솔루션", "사업자"}


def _strip_korean_particle(token: str) -> str:
    if len(token) < 4 or not re.fullmatch(r"[가-힣]+", token):
        return token
    for suffix in ("으로", "에게", "에서", "과", "와", "은", "는", "이", "가", "을", "를", "의"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _useful_token(token: str) -> bool:
    if len(token) < 2 or token.isdigit():
        return False
    stopwords = {
        "lg",
        "cns",
        "lgcns",
        "삼성",
        "samsung",
        "sds",
        "samsungsds",
        "sk",
        "ax",
        "ai",
        "단독",
        "종합",
        "속보",
        "브리프",
        "출시",
        "공개",
        "선정",
        "사업",
        "정부",
        "계약",
        "도입",
        "체결",
        "협력",
        "맞손",
        "수주",
        "공급",
        "구축",
        "확대",
        "기반",
        "기업용",
        "기업",
        "전사",
        "그룹",
        "계열사",
        "공략",
        "확대",
    }
    return token not in stopwords


def _compact(text_value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(text_value or "").lower())


if __name__ == "__main__":
    main()
