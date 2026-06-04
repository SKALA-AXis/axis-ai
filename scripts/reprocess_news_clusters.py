"""뉴스 기사 전체 클러스터 재처리 스크립트.

기존 PROCESSED 뉴스 기사까지 다시 RAW로 되돌린 뒤, 전처리 파이프라인을
batch 단위로 끝까지 실행한다. card_news row는 기본적으로 삭제하지 않는다.

사용 예:
  uv run python scripts/reprocess_news_clusters.py --env local --dry-run
  uv run python scripts/reprocess_news_clusters.py --env local
  uv run python scripts/reprocess_news_clusters.py --env cloud --limit 300
  uv run python scripts/reprocess_news_clusters.py --env local --include-skipped
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.env_loader import load_profile  # noqa: E402
from src.db.postgres import SessionLocal  # noqa: E402
from src.preprocessing.classification import ClusterClassifier  # noqa: E402
from src.preprocessing.dedup import ArticleDeduplicator  # noqa: E402
from src.preprocessing.preprocessing import PreprocessingService  # noqa: E402
from src.preprocessing.relevance import RelevanceEvaluator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("reprocess_news_clusters")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="뉴스 클러스터 전체 재처리")
    parser.add_argument(
        "--env",
        choices=["local", "cloud"],
        default=None,
        help="DB 프로파일. .env.{profile} 파일이 있으면 로드한다.",
    )
    parser.add_argument(
        "--company",
        action="append",
        default=None,
        help="특정 company id만 재처리. 여러 번 지정 가능. 생략하면 전체.",
    )
    parser.add_argument(
        "--source-type",
        action="append",
        default=["news"],
        help="재처리할 source_type. 기본 news.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=["RAW", "PROCESSED", "CLASSIFIED"],
        help=(
            "재처리 대상 processing_status. 기본 RAW/PROCESSED/CLASSIFIED. "
            "쉼표 구분/반복 지정 가능."
        ),
    )
    parser.add_argument(
        "--published-since",
        default=None,
        help="published_at 하한 ISO timestamp. 예: 2025-06-02T00:00:00+00:00",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=0,
        help="현재 UTC 기준 최근 N일 기사만 재처리. --published-since보다 우선.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="SKIPPED 기사도 재처리 대상에 포함.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="PreprocessingService 1회 처리 batch 크기. 기본 500.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="최대 batch 수. 0이면 RAW 대상이 없어질 때까지 실행.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="기존 row를 RAW로 되돌리지 않고 현재 RAW 대상만 batch 처리.",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="RAW 되돌리기만 수행하고 전처리는 실행하지 않음.",
    )
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help=(
            "relevance/classification은 다시 돌리지 않고 기존 분석 컬럼을 유지한 채 "
            "cluster_id/is_representative만 재계산."
        ),
    )
    parser.add_argument(
        "--enable-relevance-llm",
        action="store_true",
        help="관련성 판단에서 LLM batch 보조를 켠다. 클러스터링 LLM judge와는 무관.",
    )
    parser.add_argument(
        "--enable-classifier-llm",
        action="store_true",
        help="클러스터 분류에서 LLM 보조를 켠다. 기본은 비활성.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="대상 row 수만 출력하고 DB를 수정하지 않음.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.enable_relevance_llm:
        os.environ.setdefault("ENABLE_RELEVANCE_LLM", "true")
    if args.enable_classifier_llm:
        os.environ.setdefault("ENABLE_OPENAI_CALLS", "true")

    profile = load_profile(args.env)
    source_types = _normalize_source_types(args.source_type)
    statuses = _normalize_statuses(args.status, include_skipped=args.include_skipped)
    companies = list(dict.fromkeys(args.company or []))
    published_since = _resolve_published_since(args)

    before = _count_targets(
        source_types=source_types,
        statuses=statuses,
        companies=companies,
        published_since=published_since,
    )
    raw_before = _count_raw(
        source_types=source_types,
        companies=companies,
        published_since=published_since,
    )
    log.info(
        (
            "재처리 대상 확인 | profile=%s source_types=%s statuses=%s "
            "companies=%s published_since=%s total=%d raw=%d"
        ),
        profile,
        source_types,
        statuses,
        companies or ["*"],
        published_since,
        before,
        raw_before,
    )

    if args.dry_run:
        print("\nDRY RUN")
        print(f"  profile:      {profile}")
        print(f"  source_types: {source_types}")
        print(f"  statuses:     {statuses}")
        print(f"  companies:    {companies or ['*']}")
        print(f"  published_since: {published_since or '*'}")
        print(f"  target rows:  {before}")
        print(f"  current RAW:  {raw_before}")
        return

    if not args.skip_reset:
        updated = _reset_targets(
            source_types=source_types,
            statuses=statuses,
            companies=companies,
            published_since=published_since,
        )
        log.info("재처리 상태 초기화 완료 | updated=%d", updated)

    if args.reset_only:
        return

    if args.cluster_only:
        _run_cluster_only(
            source_types=source_types,
            companies=companies,
            published_since=published_since,
            limit=max(1, args.limit),
            max_batches=max(0, args.max_batches),
        )
        return

    _run_batches(
        source_types=source_types,
        companies=companies,
        published_since=published_since,
        limit=max(1, args.limit),
        max_batches=max(0, args.max_batches),
        enable_relevance_llm=bool(args.enable_relevance_llm),
        enable_classifier_llm=bool(args.enable_classifier_llm),
    )


def _resolve_published_since(args: argparse.Namespace) -> str | None:
    if args.lookback_days and args.lookback_days > 0:
        return (datetime.now(UTC) - timedelta(days=args.lookback_days)).isoformat()
    return args.published_since


def _normalize_source_types(values: list[str]) -> list[str]:
    normalized = [
        source_type.strip().lower()
        for value in values
        for source_type in value.split(",")
        if source_type.strip()
    ]
    return list(dict.fromkeys(normalized)) or ["news"]


def _normalize_statuses(values: list[str], *, include_skipped: bool) -> list[str]:
    statuses = [
        status.strip().upper() for value in values for status in value.split(",") if status.strip()
    ]
    if include_skipped:
        statuses.append("SKIPPED")
    return list(dict.fromkeys(statuses)) or ["RAW", "PROCESSED", "CLASSIFIED"]


def _count_targets(
    *,
    source_types: list[str],
    statuses: list[str],
    companies: list[str],
    published_since: str | None,
) -> int:
    with SessionLocal() as db:
        return int(
            db.execute(
                _count_sql(),
                _params(source_types, companies, statuses, published_since),
            ).scalar()
            or 0
        )


def _count_raw(
    *,
    source_types: list[str],
    companies: list[str],
    published_since: str | None,
) -> int:
    params = _params(source_types, companies, ["RAW"], published_since)
    with SessionLocal() as db:
        return int(
            db.execute(
                text(f"{_target_base_sql()} AND processing_status = 'RAW'"),
                params,
            ).scalar()
            or 0
        )


def _reset_targets(
    *,
    source_types: list[str],
    statuses: list[str],
    companies: list[str],
    published_since: str | None,
) -> int:
    params = _params(source_types, companies, statuses, published_since)
    with SessionLocal() as db:
        result = db.execute(
            text(f"""
                UPDATE raw_articles
                SET processing_status = 'RAW',
                    cluster_id = NULL,
                    is_representative = FALSE,
                    importance_level = NULL,
                    importance_score = NULL,
                    qdrant_vector_id = NULL,
                    error_message = NULL
                WHERE id IN (
                    SELECT id
                    FROM raw_articles
                    WHERE {_target_where_sql()}
                )
            """),
            params,
        )
        db.commit()
        return int(getattr(result, "rowcount", 0) or 0)


def _run_batches(
    *,
    source_types: list[str],
    companies: list[str],
    published_since: str | None,
    limit: int,
    max_batches: int,
    enable_relevance_llm: bool,
    enable_classifier_llm: bool,
) -> None:
    service = PreprocessingService(
        relevance_evaluator=RelevanceEvaluator(enable_llm=enable_relevance_llm),
        classifier=ClusterClassifier(enable_llm=enable_classifier_llm),
    )
    total_raw = 0
    total_relevant = 0
    total_clusters = 0
    total_classified = 0
    batch = 0

    while True:
        if max_batches and batch >= max_batches:
            break
        batch += 1
        result = service.run(
            company=companies,
            source_types=source_types,
            trigger_type="manual:reprocess_news_clusters",
            published_since=published_since,
            limit=limit,
        )
        raw_count = len(result.get("raw_article_ids", []))
        if raw_count == 0:
            batch -= 1
            break

        relevant_count = len(result.get("relevant_ids", []))
        cluster_count = len(result.get("cluster_map", {}))
        classified_count = len(result.get("classified_clusters", []))
        total_raw += raw_count
        total_relevant += relevant_count
        total_clusters += cluster_count
        total_classified += classified_count

        log.info(
            "batch 완료 | batch=%d raw=%d relevant=%d clusters=%d classified=%d",
            batch,
            raw_count,
            relevant_count,
            cluster_count,
            classified_count,
        )

        if raw_count < limit:
            break

    log.info(
        "전체 재처리 완료 | batches=%d raw=%d relevant=%d clusters=%d classified=%d",
        batch,
        total_raw,
        total_relevant,
        total_clusters,
        total_classified,
    )


def _run_cluster_only(
    *,
    source_types: list[str],
    companies: list[str],
    published_since: str | None,
    limit: int,
    max_batches: int,
) -> None:
    deduplicator = ArticleDeduplicator()
    total_articles = 0
    total_clusters = 0
    total_representatives = 0
    batch = 0

    while True:
        if max_batches and batch >= max_batches:
            break

        article_ids = _list_cluster_only_article_ids(
            source_types=source_types,
            companies=companies,
            published_since=published_since,
            limit=limit,
        )
        if not article_ids:
            break

        batch += 1
        log.info(
            "cluster-only batch 시작 | batch=%d articles=%d",
            batch,
            len(article_ids),
        )
        cluster_map, representative_ids = deduplicator.deduplicate(article_ids)
        total_articles += len(article_ids)
        total_clusters += len(cluster_map)
        total_representatives += len(representative_ids)
        log.info(
            "cluster-only batch 완료 | batch=%d articles=%d clusters=%d representatives=%d",
            batch,
            len(article_ids),
            len(cluster_map),
            len(representative_ids),
        )

        if len(article_ids) < limit:
            break

    log.info(
        "cluster-only 완료 | batches=%d articles=%d clusters=%d representatives=%d",
        batch,
        total_articles,
        total_clusters,
        total_representatives,
    )


def _list_cluster_only_article_ids(
    *,
    source_types: list[str],
    companies: list[str],
    published_since: str | None,
    limit: int,
) -> list[int]:
    params = _params(source_types, companies, ["RAW"], published_since)
    params["limit"] = limit
    with SessionLocal() as db:
        rows = db.execute(
            text(f"""
                SELECT id
                FROM raw_articles
                WHERE {_target_where_sql()}
                ORDER BY published_at DESC NULLS LAST, collected_at DESC, id DESC
                LIMIT :limit
            """),
            params,
        ).fetchall()
    return [int(row.id) for row in rows]


def _params(
    source_types: list[str],
    companies: list[str],
    statuses: list[str],
    published_since: str | None,
) -> dict[str, Any]:
    return {
        "source_types": source_types,
        "statuses": statuses,
        "companies": companies if companies else [""],
        "no_company_filter": not companies,
        "published_since": published_since,
    }


def _count_sql():
    return text(f"{_target_base_sql()}")


def _target_base_sql() -> str:
    return f"SELECT COUNT(*) FROM raw_articles WHERE {_target_where_sql()}"


def _target_where_sql() -> str:
    return """
        source_type = ANY(:source_types)
        AND processing_status = ANY(:statuses)
        AND crawl_status = 'success'
        AND (:published_since IS NULL OR published_at >= CAST(:published_since AS timestamptz))
        AND (:no_company_filter OR company ?| :companies)
    """


if __name__ == "__main__":
    main()
