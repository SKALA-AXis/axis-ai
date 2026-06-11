"""뉴스 기사 전체 클러스터 재처리 스크립트.

기존 PROCESSED 뉴스 기사까지 다시 RAW로 되돌린 뒤, 전처리 파이프라인을
batch 단위로 끝까지 실행한다. card_news row는 기본적으로 삭제하지 않는다.

사용 예:
  uv run python scripts/reprocess_news_clusters.py --env local --dry-run
  uv run python scripts/reprocess_news_clusters.py --env local
  uv run python scripts/reprocess_news_clusters.py --env cloud --limit 300
  uv run python scripts/reprocess_news_clusters.py --env cloud \
    --published-since 2026-05-31T00:00:00+00:00 \
    --published-until 2026-06-01T00:00:00+00:00
  uv run python scripts/reprocess_news_clusters.py --env local --include-skipped
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
        "--published-until",
        default=None,
        help="published_at 상한 ISO timestamp. 지정 시 published_at < until 범위만 재처리.",
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
        "--title-llm-cluster",
        action="store_true",
        help=(
            "cluster-only에서 pairwise rule/BGE 대신 제목 목록을 LLM으로 한 번에 그룹핑한다. "
            "본문은 보내지 않는다."
        ),
    )
    parser.add_argument(
        "--title-llm-pipeline",
        action="store_true",
        help=(
            "대량 수동 재처리용 경량 경로. relevance를 저장한 뒤 BGE dedup 대신 "
            "title LLM clustering으로 cluster_id/is_representative를 계산한다."
        ),
    )
    parser.add_argument(
        "--title-llm-batch-size",
        type=int,
        default=120,
        help="title LLM clustering 1회 요청 article 수. 기본 120.",
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
    published_until = args.published_until

    before = _count_targets(
        source_types=source_types,
        statuses=statuses,
        companies=companies,
        published_since=published_since,
        published_until=published_until,
    )
    raw_before = _count_raw(
        source_types=source_types,
        companies=companies,
        published_since=published_since,
        published_until=published_until,
    )
    log.info(
        (
            "재처리 대상 확인 | profile=%s source_types=%s statuses=%s "
            "companies=%s published_since=%s published_until=%s total=%d raw=%d"
        ),
        profile,
        source_types,
        statuses,
        companies or ["*"],
        published_since,
        published_until,
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
        print(f"  published_until: {published_until or '*'}")
        print(f"  target rows:  {before}")
        print(f"  current RAW:  {raw_before}")
        return

    if args.cluster_only:
        cluster_statuses = _cluster_only_statuses(statuses)
        if args.title_llm_cluster:
            _run_title_llm_cluster_only(
                source_types=source_types,
                statuses=cluster_statuses,
                companies=companies,
                published_since=published_since,
                published_until=published_until,
                limit=max(1, args.limit),
                batch_size=max(10, args.title_llm_batch_size),
                reset_cluster_fields=not args.skip_reset,
            )
            return
        if not args.skip_reset:
            updated = _reset_cluster_fields(
                source_types=source_types,
                statuses=cluster_statuses,
                companies=companies,
                published_since=published_since,
                published_until=published_until,
            )
            log.info("cluster-only 클러스터 필드 초기화 완료 | updated=%d", updated)
        _run_cluster_only(
            source_types=source_types,
            statuses=cluster_statuses,
            companies=companies,
            published_since=published_since,
            published_until=published_until,
            limit=max(1, args.limit),
            max_batches=max(0, args.max_batches),
        )
        return

    if not args.skip_reset:
        updated = _reset_targets(
            source_types=source_types,
            statuses=statuses,
            companies=companies,
            published_since=published_since,
            published_until=published_until,
        )
        log.info("재처리 상태 초기화 완료 | updated=%d", updated)

    if args.reset_only:
        return

    if args.title_llm_pipeline:
        _run_title_llm_pipeline(
            source_types=source_types,
            companies=companies,
            published_since=published_since,
            published_until=published_until,
            limit=max(1, args.limit),
            max_batches=max(0, args.max_batches),
            batch_size=max(10, args.title_llm_batch_size),
            enable_relevance_llm=bool(args.enable_relevance_llm),
            enable_classifier_llm=bool(args.enable_classifier_llm),
        )
        return

    _run_batches(
        source_types=source_types,
        companies=companies,
        published_since=published_since,
        published_until=published_until,
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
    published_until: str | None,
) -> int:
    with SessionLocal() as db:
        return int(
            db.execute(
                _count_sql(),
                _params(source_types, companies, statuses, published_since, published_until),
            ).scalar()
            or 0
        )


def _count_raw(
    *,
    source_types: list[str],
    companies: list[str],
    published_since: str | None,
    published_until: str | None,
) -> int:
    params = _params(source_types, companies, ["RAW"], published_since, published_until)
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
    published_until: str | None,
) -> int:
    params = _params(source_types, companies, statuses, published_since, published_until)
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


def _reset_cluster_fields(
    *,
    source_types: list[str],
    statuses: list[str],
    companies: list[str],
    published_since: str | None,
    published_until: str | None,
) -> int:
    params = _params(source_types, companies, statuses, published_since, published_until)
    with SessionLocal() as db:
        result = db.execute(
            text(f"""
                UPDATE raw_articles
                SET cluster_id = NULL,
                    is_representative = FALSE
                WHERE id IN (
                    SELECT id
                    FROM raw_articles
                    WHERE {_target_where_sql()}
                      AND relevance_label = 'relevant'
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
    published_until: str | None,
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
            published_until=published_until,
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


def _run_title_llm_pipeline(
    *,
    source_types: list[str],
    companies: list[str],
    published_since: str | None,
    published_until: str | None,
    limit: int,
    max_batches: int,
    batch_size: int,
    enable_relevance_llm: bool,
    enable_classifier_llm: bool,
) -> None:
    service = PreprocessingService(
        relevance_evaluator=RelevanceEvaluator(enable_llm=enable_relevance_llm),
        classifier=ClusterClassifier(enable_llm=enable_classifier_llm),
    )
    total_raw = 0
    total_relevant = 0
    batch = 0

    while True:
        if max_batches and batch >= max_batches:
            break
        batch += 1
        raw_ids = service.load_raw_ids(
            companies,
            source_types=source_types,
            limit=limit,
            published_since=published_since,
            published_until=published_until,
        )
        if not raw_ids:
            batch -= 1
            break

        route_result = service.route_by_source(raw_ids)
        service.analyze_documents(route_result.get("parsed_document_ids", []))
        relevant_ids = route_result.get("relevant_ids", [])
        if relevant_ids:
            _mark_relevant_articles_processed(relevant_ids)

        total_raw += len(raw_ids)
        total_relevant += len(relevant_ids)
        log.info(
            "title-llm-pipeline relevance 완료 | batch=%d raw=%d relevant=%d skipped=%d",
            batch,
            len(raw_ids),
            len(relevant_ids),
            len(route_result.get("skipped_preprocess_ids", [])),
        )

        if len(raw_ids) < limit:
            break

    cluster_map = _run_title_llm_cluster_only(
        source_types=source_types,
        statuses=["PROCESSED", "CLASSIFIED"],
        companies=companies,
        published_since=published_since,
        published_until=published_until,
        limit=max(limit * max(batch, 1), limit),
        batch_size=batch_size,
        reset_cluster_fields=True,
    )
    classified = service.classify_clusters(
        representative_ids=list(cluster_map),
        cluster_map=cluster_map,
        requested_companies=companies,
    )
    log.info(
        ("title-llm-pipeline 완료 | batches=%d raw=%d relevant=%d clusters=%d classified=%d"),
        batch,
        total_raw,
        total_relevant,
        len(cluster_map),
        len(classified),
    )


def _mark_relevant_articles_processed(article_ids: list[int]) -> int:
    if not article_ids:
        return 0
    with SessionLocal() as db:
        result = db.execute(
            text("""
                UPDATE raw_articles
                SET processing_status = 'PROCESSED'
                WHERE id = ANY(:article_ids)
                  AND relevance_label = 'relevant'
            """),
            {"article_ids": article_ids},
        )
        db.commit()
        return int(getattr(result, "rowcount", 0) or 0)


def _run_cluster_only(
    *,
    source_types: list[str],
    statuses: list[str],
    companies: list[str],
    published_since: str | None,
    published_until: str | None,
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
            statuses=statuses,
            companies=companies,
            published_since=published_since,
            published_until=published_until,
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


def _run_title_llm_cluster_only(
    *,
    source_types: list[str],
    statuses: list[str],
    companies: list[str],
    published_since: str | None,
    published_until: str | None,
    limit: int,
    batch_size: int,
    reset_cluster_fields: bool,
) -> dict[int, list[int]]:
    articles = _list_title_llm_cluster_articles(
        source_types=source_types,
        statuses=statuses,
        companies=companies,
        published_since=published_since,
        published_until=published_until,
        limit=limit,
        unclustered_only=not reset_cluster_fields,
    )
    if not articles:
        log.info("title-llm-cluster 대상 없음")
        return {}

    total_clusters = 0
    total_articles = 0
    combined_cluster_map: dict[int, list[int]] = {}
    for batch_index, batch in enumerate(_title_llm_batches(articles, batch_size), start=1):
        groups = _invoke_title_llm_cluster_resilient(batch)
        cluster_map = _post_merge_title_llm_clusters(
            _split_unrelated_title_groups(
                _normalize_title_llm_groups(groups, batch),
                batch,
            ),
            batch,
            groups,
        )
        combined_cluster_map.update(cluster_map)
        total_clusters += len(cluster_map)
        total_articles += sum(len(article_ids) for article_ids in cluster_map.values())
        log.info(
            "title-llm-cluster batch 완료 | batch=%d articles=%d clusters=%d",
            batch_index,
            len(batch),
            len(cluster_map),
        )

    if reset_cluster_fields:
        updated = _replace_title_llm_clusters(
            cluster_map=combined_cluster_map,
            source_types=source_types,
            statuses=statuses,
            companies=companies,
            published_since=published_since,
            published_until=published_until,
        )
        log.info("title-llm-cluster 클러스터 필드 교체 완료 | reset_updated=%d", updated)
    else:
        _persist_title_llm_clusters(combined_cluster_map)

    log.info(
        "title-llm-cluster 완료 | articles=%d clusters=%d",
        total_articles,
        total_clusters,
    )
    return combined_cluster_map


def _list_cluster_only_article_ids(
    *,
    source_types: list[str],
    statuses: list[str],
    companies: list[str],
    published_since: str | None,
    published_until: str | None,
    limit: int,
) -> list[int]:
    params = _params(source_types, companies, statuses, published_since, published_until)
    params["limit"] = limit
    with SessionLocal() as db:
        rows = db.execute(
            text(f"""
                SELECT id
                FROM raw_articles
                WHERE {_target_where_sql()}
                  AND relevance_label = 'relevant'
                  AND cluster_id IS NULL
                ORDER BY published_at DESC NULLS LAST, collected_at DESC, id DESC
                LIMIT :limit
            """),
            params,
        ).fetchall()
    return [int(row.id) for row in rows]


def _list_title_llm_cluster_articles(
    *,
    source_types: list[str],
    statuses: list[str],
    companies: list[str],
    published_since: str | None,
    published_until: str | None,
    limit: int,
    unclustered_only: bool,
) -> list[dict[str, Any]]:
    params = _params(source_types, companies, statuses, published_since, published_until)
    params["limit"] = limit
    cluster_filter = "AND cluster_id IS NULL" if unclustered_only else ""
    with SessionLocal() as db:
        rows = db.execute(
            text(f"""
                SELECT
                    id,
                    title,
                    content,
                    published_at,
                    matched_companies,
                    matched_sectors
                FROM raw_articles
                WHERE {_target_where_sql()}
                  AND relevance_label = 'relevant'
                  {cluster_filter}
                ORDER BY published_at DESC NULLS LAST, collected_at DESC, id DESC
                LIMIT :limit
            """),
            params,
        ).mappings()
        return [
            {
                "id": int(row["id"]),
                "title": str(row["title"] or ""),
                "content_lead": _short_content_hint(row["content"]),
                "published_at": str(row["published_at"] or ""),
                "matched_companies": _jsonish_list(row["matched_companies"]),
                "matched_sectors": _jsonish_list(row["matched_sectors"]),
            }
            for row in rows
        ]


def _title_llm_batches(
    articles: list[dict[str, Any]],
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for article in articles:
        key = "|".join(article.get("matched_companies") or ["unknown"])
        grouped.setdefault(key, []).append(article)

    batches: list[list[dict[str, Any]]] = []
    for group in grouped.values():
        for i in range(0, len(group), batch_size):
            batches.append(group[i : i + batch_size])
    return batches


def _invoke_title_llm_cluster_resilient(
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        return _invoke_title_llm_cluster(articles)
    except (json.JSONDecodeError, ValueError) as exc:
        if len(articles) <= 10:
            raise ValueError(f"title LLM cluster JSON 파싱 실패: articles={len(articles)}") from exc

        mid = len(articles) // 2
        log.warning(
            "title LLM cluster JSON 파싱 실패, batch 분할 재시도 | "
            "articles=%d left=%d right=%d error=%s",
            len(articles),
            mid,
            len(articles) - mid,
            exc,
        )
        return [
            *_invoke_title_llm_cluster_resilient(articles[:mid]),
            *_invoke_title_llm_cluster_resilient(articles[mid:]),
        ]


def _invoke_title_llm_cluster(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from openai import OpenAI

    payload = {
        "instruction": (
            "Group Korean news article titles into same underlying business-event clusters. "
            "Use title as the primary signal. Use content_hint only when the title is too vague. "
            "Do not group articles that are only broad themes, stock mood, or background context. "
            "Return JSON only."
        ),
        "rules": [
            (
                "Same event: duplicate/syndicated titles, same deal, same partnership, "
                "same product launch, same program announcement."
            ),
            (
                "Group title variants even when wording differs, e.g. SKALA vs 스칼라, "
                "예탁원 vs 예탁결제원, OpenAI vs 오픈AI."
            ),
            "Separate different counterparties, different projects, or different sub-events.",
            (
                "Do not create singleton clusters when multiple titles clearly describe "
                "the same named event, product, program, deal, or press announcement."
            ),
            (
                "For example, all titles mentioning LG CNS and 피지컬웍스/RX platform "
                "public release/demonstration belong in one cluster."
            ),
            (
                "Stock reaction titles should be grouped with the underlying concrete event "
                "when they name the same project, e.g. 삼성SDS 온AI/모바일 업무환경 지원."
            ),
            (
                "If a title is vague but content_hint names the same concrete event, group it. "
                "Ignore content_hint when it only provides old background or related examples."
            ),
            "Each input article id must appear exactly once.",
            (
                "For each cluster, create a stable snake_case event_key from the concrete event, "
                "not from wording style. The same event must reuse the same event_key."
            ),
        ],
        "articles": [_title_llm_article_payload(article) for article in articles],
        "required_json_schema": {
            "clusters": [
                {
                    "event_key": "stable_snake_case_event_key",
                    "label": "short Korean event label",
                    "article_ids": ["integer ids in this cluster"],
                }
            ]
        },
    }
    response = OpenAI().chat.completions.create(
        model=os.getenv("TITLE_LLM_CLUSTER_MODEL", "gpt-4o-mini"),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "title_cluster_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "clusters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "event_key": {"type": "string"},
                                    "label": {"type": "string"},
                                    "article_ids": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                },
                                "required": ["event_key", "label", "article_ids"],
                            },
                        },
                    },
                    "required": ["clusters"],
                },
            },
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict Korean news title clustering engine. Return JSON only."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ],
        temperature=0,
    )
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "length":
        raise ValueError("title LLM cluster response truncated")
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    clusters = parsed.get("clusters", [])
    return clusters if isinstance(clusters, list) else []


def _normalize_title_llm_groups(
    groups: list[dict[str, Any]],
    articles: list[dict[str, Any]],
) -> dict[int, list[int]]:
    valid_ids = {int(article["id"]) for article in articles}
    assigned: set[int] = set()
    cluster_map: dict[int, list[int]] = {}

    for group in groups:
        raw_ids = group.get("article_ids") if isinstance(group, dict) else []
        article_ids = [
            int(article_id)
            for article_id in (raw_ids or [])
            if _is_intish(article_id) and int(article_id) in valid_ids
        ]
        article_ids = list(dict.fromkeys(article_ids))
        if not article_ids:
            continue
        representative_id = article_ids[0]
        cluster_map[representative_id] = article_ids
        assigned.update(article_ids)

    for article_id in valid_ids - assigned:
        cluster_map[article_id] = [article_id]

    return cluster_map


def _post_merge_title_llm_clusters(
    cluster_map: dict[int, list[int]],
    articles: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> dict[int, list[int]]:
    del groups
    article_by_id = {int(article["id"]): article for article in articles}
    split_cluster_map = _split_unrelated_title_groups(cluster_map, articles)
    return _merge_title_related_clusters(split_cluster_map, article_by_id)


def _title_llm_event_keys(
    groups: list[dict[str, Any]],
    valid_ids: set[int],
) -> dict[int, str]:
    event_keys: dict[int, str] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        event_key = _normalize_llm_event_key(group.get("event_key"))
        if event_key is None:
            continue
        raw_ids = group.get("article_ids") or []
        for raw_id in raw_ids:
            if _is_intish(raw_id) and int(raw_id) in valid_ids:
                event_keys[int(raw_id)] = event_key
    return event_keys


def _split_unrelated_title_groups(
    cluster_map: dict[int, list[int]],
    articles: list[dict[str, Any]],
) -> dict[int, list[int]]:
    article_by_id = {int(article["id"]): article for article in articles}
    result: dict[int, list[int]] = {}
    for article_ids in cluster_map.values():
        unique_ids = _unique_ints(article_ids)
        for component in _title_related_components(unique_ids, article_by_id):
            representative_id = _representative_title_llm_id(component, article_by_id)
            result[representative_id] = component
    return result


def _title_related_components(
    article_ids: list[int],
    article_by_id: dict[int, dict[str, Any]],
) -> list[list[int]]:
    if len(article_ids) <= 1:
        return [article_ids]

    parent = {article_id: article_id for article_id in article_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left_id in enumerate(article_ids):
        for right_id in article_ids[index + 1 :]:
            if _title_articles_related(article_by_id.get(left_id), article_by_id.get(right_id)):
                union(left_id, right_id)

    components: dict[int, list[int]] = {}
    for article_id in article_ids:
        components.setdefault(find(article_id), []).append(article_id)
    return list(components.values())


def _title_articles_related(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> bool:
    if not left or not right:
        return False
    left_title = str(left.get("title") or "")
    right_title = str(right.get("title") or "")
    left_tokens = _title_merge_tokens(left_title)
    right_tokens = _title_merge_tokens(right_title)
    if _multi_topic_bridge_conflict(left_title, right_title, left_tokens, right_tokens):
        return False
    left_features = _title_cluster_features([int(left["id"])], {int(left["id"]): left})
    right_features = _title_cluster_features([int(right["id"])], {int(right["id"]): right})
    return _title_clusters_related(left_features, right_features)


def _merge_title_related_clusters(
    cluster_map: dict[int, list[int]],
    article_by_id: dict[int, dict[str, Any]],
) -> dict[int, list[int]]:
    representatives = list(cluster_map)
    parent = {representative_id: representative_id for representative_id in representatives}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    cluster_features = {
        representative_id: _title_cluster_features(article_ids, article_by_id)
        for representative_id, article_ids in cluster_map.items()
    }

    for index, left_id in enumerate(representatives):
        for right_id in representatives[index + 1 :]:
            if _title_clusters_related(cluster_features[left_id], cluster_features[right_id]):
                union(left_id, right_id)

    merged: dict[int, list[int]] = {}
    for representative_id, article_ids in cluster_map.items():
        root = find(representative_id)
        merged.setdefault(root, []).extend(article_ids)

    result: dict[int, list[int]] = {}
    for article_ids in merged.values():
        unique_ids = _unique_ints(article_ids)
        result[_representative_title_llm_id(unique_ids, article_by_id)] = unique_ids
    return result


def _title_cluster_features(
    article_ids: list[int],
    article_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    token_counts: dict[str, int] = {}
    companies: set[str] = set()
    dates: set[str] = set()
    list_like_count = 0
    multi_topic_count = 0
    title_count = 0
    for article_id in article_ids:
        article = article_by_id.get(article_id) or {}
        title = str(article.get("title") or "")
        for token in _title_merge_tokens(title):
            token_counts[token] = token_counts.get(token, 0) + 1
        companies |= set(article.get("matched_companies") or [])
        published_at = str(article.get("published_at") or "")
        if published_at:
            dates.add(published_at[:10])
        if title:
            title_count += 1
            if _is_list_like_title(title):
                list_like_count += 1
            if _is_multi_topic_title(title):
                multi_topic_count += 1
    tokens = _cluster_core_title_tokens(token_counts, title_count)
    return {
        "tokens": tokens,
        "anchors": _concrete_title_merge_anchors(tokens),
        "companies": companies,
        "dates": dates,
        "list_like_ratio": list_like_count / title_count if title_count else 0.0,
        "multi_topic_ratio": multi_topic_count / title_count if title_count else 0.0,
        "article_count": len(article_ids),
    }


def _cluster_core_title_tokens(token_counts: dict[str, int], title_count: int) -> set[str]:
    if title_count <= 1:
        return set(token_counts)
    threshold = max(2, (title_count + 1) // 2)
    return {token for token, count in token_counts.items() if count >= threshold}


def _title_clusters_related(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_companies = left["companies"]
    right_companies = right["companies"]
    if not _companies_compatible(left_companies, right_companies):
        return False
    if not _dates_near(left["dates"], right["dates"]):
        return False

    left_tokens = left["tokens"]
    right_tokens = right["tokens"]
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False

    shared = left_tokens & right_tokens
    if len(shared) < 2:
        return False

    if _cluster_multi_topic_bridge_conflict(left, right):
        return False

    if _has_concrete_product_overlap(shared):
        return True

    strict_mode = _requires_strict_title_merge(left, right)
    coverage = len(shared) / min(len(left_tokens), len(right_tokens))
    jaccard = len(shared) / len(left_tokens | right_tokens)
    if strict_mode:
        return len(shared) >= 3 and coverage >= 0.62 and jaccard >= 0.32
    return coverage >= 0.58 or (len(shared) >= 3 and jaccard >= 0.30)


def _cluster_multi_topic_bridge_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["multi_topic_ratio"] <= 0 and right["multi_topic_ratio"] <= 0:
        return False
    return len(left["anchors"] & right["anchors"]) < 2


def _companies_compatible(left_companies: set[str], right_companies: set[str]) -> bool:
    if not left_companies or not right_companies:
        return True
    return bool(left_companies & right_companies)


def _requires_strict_title_merge(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_companies = left["companies"]
    right_companies = right["companies"]
    company_sets_differ = bool(
        left_companies and right_companies and left_companies != right_companies
    )
    list_like_bridge = left["list_like_ratio"] >= 0.5 or right["list_like_ratio"] >= 0.5
    return company_sets_differ or list_like_bridge


def _is_list_like_title(title: str) -> bool:
    compact = _compact_title(title)
    markers = (
        "뉴스브리프",
        "뉴스브리핑",
        "클라우드월드",
        "ai브리프",
        "it브리프",
        "it스냅샷",
        "전자it레이더",
        "시큐리티포커스",
        "it는지금",
        "biznow",
        "기업경쟁력",
        "테크앤나우",
        "tech&now",
    )
    if any(marker in compact for marker in markers):
        return True
    return bool(re.match(r"^\[?#?[가-힣a-z0-9]*(?:포커스|레이더|브리프|스냅샷)\]?", compact))


def _dates_near(left_dates: set[str], right_dates: set[str]) -> bool:
    if not left_dates or not right_dates:
        return True
    for left in left_dates:
        for right in right_dates:
            try:
                left_date = datetime.fromisoformat(left)
                right_date = datetime.fromisoformat(right)
            except ValueError:
                continue
            if abs((left_date - right_date).days) <= 3:
                return True
    return False


def _title_merge_tokens(title: str) -> set[str]:
    tokens = {
        _normalize_title_merge_token(token)
        for token in re.findall(r"[가-힣A-Za-z0-9]+", str(title or "").lower())
        if token.strip()
    }
    return {token for token in tokens if _useful_title_merge_token(token)}


def _normalize_title_merge_token(token: str) -> str:
    compact = _compact_title(token)
    compact = _strip_title_particle(compact)
    aliases = {
        "엘지씨엔에스": "lgcns",
        "lg씨엔에스": "lgcns",
        "삼성에스디에스": "samsungsds",
        "삼성sds": "samsungsds",
        "에스케이": "sk",
        "오픈ai": "openai",
        "챗gpt": "chatgpt",
        "챗지피티": "chatgpt",
        "chatgpt": "chatgpt",
        "스칼라": "skala",
        "예탁원": "예탁결제원",
        "sto": "토큰증권",
        "온에이아이": "온ai",
    }
    return aliases.get(compact, compact)


def _strip_title_particle(token: str) -> str:
    if len(token) < 4:
        return token
    for suffix in ("으로", "에게", "에서", "과", "와", "은", "는", "이", "가", "을", "를", "의"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _has_concrete_product_overlap(shared_tokens: set[str]) -> bool:
    if len(shared_tokens) < 2:
        return False
    return any(
        len(token) >= 4 and any(char.isascii() and char.isalpha() for char in token)
        for token in shared_tokens
    )


def _multi_topic_bridge_conflict(
    left_title: str,
    right_title: str,
    left_tokens: set[str],
    right_tokens: set[str],
) -> bool:
    left_multi = _is_multi_topic_title(left_title)
    right_multi = _is_multi_topic_title(right_title)
    if not left_multi and not right_multi:
        return False

    shared_anchors = _concrete_title_merge_anchors(left_tokens & right_tokens)
    if len(shared_anchors) >= 2:
        return False

    if left_multi and len(_concrete_title_merge_anchors(left_tokens)) >= 2:
        return True
    return bool(right_multi and len(_concrete_title_merge_anchors(right_tokens)) >= 2)


def _is_multi_topic_title(title: str) -> bool:
    raw = str(title or "").lower()
    if any(marker in raw for marker in ("·", "ㆍ", "/", " 및 ", " 이어 ")):
        return True
    compact = _compact_title(title)
    return any(marker in compact for marker in ("및", "이어"))


def _concrete_title_merge_anchors(tokens: set[str]) -> set[str]:
    generic = {
        "가속",
        "계약",
        "계열사",
        "공개",
        "그룹",
        "기업용",
        "기반",
        "도입",
        "사업",
        "전격",
        "전사",
        "체결",
        "출시",
        "혁신",
        "확대",
    }
    return {
        token
        for token in tokens
        if token not in generic
        and (len(token) >= 3 or any(char.isascii() and char.isalpha() for char in token))
    }


def _useful_title_merge_token(token: str) -> bool:
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
        "현장",
        "포토",
        "이슈",
        "특징주",
        "관련주",
        "상승",
        "하락",
        "급등",
        "급락",
        "강세",
        "약세",
        "공개",
        "시연",
        "추진",
        "개발",
        "협력",
        "협업",
        "맞손",
        "체결",
        "공동",
        "나서",
        "한다",
        "위해",
        "기술",
        "시장",
        "사업",
        "플랫폼",
        "솔루션",
    }
    return token not in stopwords


def _llm_event_merge_key(
    article: dict[str, Any] | None,
    event_key: str | None,
) -> tuple[str, str] | None:
    if not article or not event_key:
        return None
    companies = "|".join(article.get("matched_companies") or ["unknown"])
    return companies, event_key


def _normalize_llm_event_key(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    key = "".join(ch if ch.isalnum() else "_" for ch in raw)
    key = "_".join(part for part in key.split("_") if part)
    if not key or key in {"unknown", "misc", "other", "single", "singleton"}:
        return None
    return key[:96]


def _title_llm_article_payload(article: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": article["id"],
        "title": article["title"],
        "published_at": article["published_at"],
        "matched_companies": article["matched_companies"],
        "matched_sectors": article["matched_sectors"],
    }
    if _needs_content_hint(article):
        payload["content_hint"] = article.get("content_lead", "")
    return payload


def _needs_content_hint(article: dict[str, Any]) -> bool:
    title = _compact_title(article.get("title") or "")
    if len(title) < 24:
        return True
    vague_markers = (
        "공개",
        "시연",
        "강화",
        "확대",
        "정조준",
        "공략",
        "기대",
        "수혜",
        "본격화",
    )
    return any(marker in title for marker in vague_markers)


def _representative_title_llm_id(
    article_ids: list[int],
    article_by_id: dict[int, dict[str, Any]],
) -> int:
    return max(
        article_ids,
        key=lambda article_id: (
            str(article_by_id.get(article_id, {}).get("published_at") or ""),
            article_id,
        ),
    )


def _unique_ints(values: list[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _compact_title(value: str) -> str:
    return "".join(str(value or "").lower().split())


def _short_content_hint(value: Any) -> str:
    return " ".join(str(value or "").split())[:220]


def _persist_title_llm_clusters(cluster_map: dict[int, list[int]]) -> None:
    with SessionLocal() as db:
        for representative_id, article_ids in cluster_map.items():
            for article_id in article_ids:
                db.execute(
                    text("""
                        UPDATE raw_articles
                        SET cluster_id = :cluster_id,
                            is_representative = :is_rep
                        WHERE id = :id
                    """),
                    {
                        "cluster_id": representative_id,
                        "is_rep": article_id == representative_id,
                        "id": article_id,
                    },
                )
        db.commit()


def _replace_title_llm_clusters(
    *,
    cluster_map: dict[int, list[int]],
    source_types: list[str],
    statuses: list[str],
    companies: list[str],
    published_since: str | None,
    published_until: str | None,
) -> int:
    params = _params(source_types, companies, statuses, published_since, published_until)
    with SessionLocal() as db:
        result = db.execute(
            text(f"""
                UPDATE raw_articles
                SET cluster_id = NULL,
                    is_representative = FALSE
                WHERE id IN (
                    SELECT id
                    FROM raw_articles
                    WHERE {_target_where_sql()}
                      AND relevance_label = 'relevant'
                )
            """),
            params,
        )
        updated = int(getattr(result, "rowcount", 0) or 0)
        for representative_id, article_ids in cluster_map.items():
            for article_id in article_ids:
                db.execute(
                    text("""
                        UPDATE raw_articles
                        SET cluster_id = :cluster_id,
                            is_representative = :is_rep
                        WHERE id = :id
                    """),
                    {
                        "cluster_id": representative_id,
                        "is_rep": article_id == representative_id,
                        "id": article_id,
                    },
                )
        db.commit()
    return updated


def _jsonish_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        return [str(parsed)] if parsed else []
    return [str(value)]


def _is_intish(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _cluster_only_statuses(statuses: list[str]) -> list[str]:
    cluster_statuses = [status for status in statuses if status in {"PROCESSED", "CLASSIFIED"}]
    return cluster_statuses or ["PROCESSED", "CLASSIFIED"]


def _params(
    source_types: list[str],
    companies: list[str],
    statuses: list[str],
    published_since: str | None,
    published_until: str | None,
) -> dict[str, Any]:
    return {
        "source_types": source_types,
        "statuses": statuses,
        "companies": companies if companies else [""],
        "no_company_filter": not companies,
        "published_since": published_since,
        "published_until": published_until,
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
        AND (:published_until IS NULL OR published_at < CAST(:published_until AS timestamptz))
        AND (:no_company_filter OR company ?| :companies)
    """


if __name__ == "__main__":
    main()
