# 작성일: 2026-04-22
# 작성자: 최종민
# 변경이력:
#   2026-04-22 최종민 — 크롤러·분석 파이프라인 및 DB(Supabase·Qdrant) 전환, card_news 리네임
#   2026-04-30 박지원 — 크롤러·전처리·DB 스키마 V31·article store 및 카드뉴스 생성·클러스터링
#   2026-05-12 심유정 — peer 뉴스 클러스터링·요약 및 카드뉴스 insight grounding·직렬화
#   2026-06-04 박진 — 통합 이슈 기반 mixer·브리핑 흐름 추가 및 포맷 정리
"""raw_articles / card_news 테이블 저장·조회·업데이트 레이어."""

import json
import logging
import threading
from typing import Any, Optional

from sqlalchemy import text

from src.crawler.base import CrawlRunContext, RawArticle
from src.db.article_store_parts._constants import (  # noqa: F401
    _CONTROL_CHAR_RE,
    _DELETE_BUSINESS_SIGNALS_SQL,
    _DELETE_FINANCIAL_METRICS_SQL,
    _ENSURE_MARKET_PRICE_OHLCV_INDEX_SQL,
    _ENSURE_MARKET_PRICE_OHLCV_SQL,
    _GLOBAL_TREND_UPSERT_SQL,
    _INDUSTRY_MARKER_KEYS,
    _INDUSTRY_REPORT_TYPES,
    _INDUSTRY_SOURCE_TYPES,
    _INSERT_CRAWL_RUN_ARTICLE,
    _INSERT_EVIDENCE,
    _INSERT_PIPELINE_LOG,
    _PARSE_RESULT_METADATA_KEYS,
    _SELECT_ARTICLE_ID_BY_URL,
    _SOURCE_METADATA_EXCLUDED_KEYS,
    _UPDATE_COMPANY_ANALYSIS_IF_CHANGED_SQL,
    _UPDATE_DART_CONTENT_IF_BETTER_SQL,
    _UPDATE_IR_CONTENT_IF_BETTER_SQL,
    _UPSERT_BUSINESS_SIGNAL_SQL,
    _UPSERT_FINANCIAL_METRIC_SQL,
    _UPSERT_MARKET_PRICE_OHLCV_SQL,
    DEFAULT_PEER_COMPANY_IDS,
    GLOBAL_NEWSROOM_SOURCE_NAMES,
    GLOBAL_RESEARCH_SOURCE_NAMES,
    INDUSTRY_TREND_COMPANY,
    SK_AX_RAW_SOURCE_NAMES,
)
from src.db.article_store_parts.article_retrieval import (  # noqa: F401
    article_exists_by_url,
)
from src.db.article_store_parts.business_signals import (  # noqa: F401
    _business_signal_params,
    delete_raw_article_business_signals,
    upsert_raw_article_business_signals,
)
from src.db.article_store_parts.cluster_operations import (  # noqa: F401
    list_card_news_cluster_candidates,
    list_existing_news_cluster_candidates,
    update_classification,
    update_cluster,
    update_preprocess_status,
)
from src.db.article_store_parts.dart_documents import (  # noqa: F401
    _dart_document_summary,
    get_dart_document_detail,
    list_dart_documents,
)
from src.db.article_store_parts.evidence_chain import (  # noqa: F401
    save_evidence_chain,
)
from src.db.article_store_parts.financial_metrics import (  # noqa: F401
    _financial_metric_params,
    _market_price_ohlcv_rows,
    _upsert_market_price_ohlcv_for_article,
    delete_raw_article_financial_metrics,
    ensure_market_price_ohlcv_schema,
    upsert_raw_article_financial_metrics,
)
from src.db.article_store_parts.helpers import (  # noqa: F401
    _article_company_ids,
    _article_matches_company,
    _build_evidence_payload,
    _canonical_card_id,
    _card_company_filter_id,
    _card_created_at_param,
    _collection_metadata,
    _company_match_aliases,
    _company_match_text,
    _date_string_or_none,
    _fetch_dart_rows,
    _filter_articles_for_card_company,
    _generate_card_id,
    _ilike_any_clause,
    _integrated_issue_context_lines,
    _integrated_issue_from_card_context,
    _is_industry_trend_article,
    _is_missing_card_news_articles_table,
    _is_missing_column,
    _is_missing_relation,
    _is_self_company_card,
    _is_undefined_column_error,
    _is_valid,
    _iso_or_none,
    _join_issue_lines,
    _json_or_value,
    _labeled_lines_from_main_detail_blocks,
    _looks_like_fragment,
    _main_detail_blocks,
    _main_detail_blocks_from_labeled_lines,
    _market_int,
    _market_number,
    _merge_source_dicts,
    _metadata_dict,
    _metadata_json,
    _normalize_int_list,
    _normalize_peer_company_fk,
    _parse_datetime,
    _parse_market_trade_date,
    _parse_result_payload,
    _resolve_integrated_issue_id,
    _resolve_peer_company_id,
    _sanitize_jsonish,
    _sanitize_text,
    _sentence_text,
    _source_article_dict_from_article,
    _source_articles_payload,
    _source_basis_datetime,
    _source_dict_from_article,
    _source_ids_from_sources,
    _source_metadata_json,
    _string_list,
    _ticker_from_market_data_url,
    _update_company_analysis_if_changed,
    _update_dart_content_if_better,
    _update_ir_content_if_better,
    _upsert_crawl_run_article,
    _upsert_parse_result,
    _upsert_source_metadata,
    fetch_domestic_trend_representative_inputs,
    fetch_global_trend_inputs,
    fetch_peer_cards_for_alignment,
    fetch_previous_trend_context_for_delta,
    fetch_sk_ax_raw_for_alignment,
    get_articles_by_ids,
    upsert_global_industry_trends,
)
from src.db.article_store_parts.pipeline_log import (  # noqa: F401
    save_pipeline_log,
)
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)


_INSERT_SQL = text("""
    INSERT INTO raw_articles (
        source_type, source_name, publisher, title, content, url, url_hash,
        published_at, collected_at, company, language, content_type,
        crawl_status, error_message, processing_status, metadata, crawl_run_id
    ) VALUES (
        :source_type, :source_name, :publisher, :title, :content, :url, :url_hash,
        :published_at, :collected_at, CAST(:company AS jsonb), :language, :content_type,
        :crawl_status, :error_message, 'RAW', CAST(:metadata AS jsonb),
        CAST(:crawl_run_id AS uuid)
    )
    ON CONFLICT (url) DO NOTHING
    RETURNING id
""")

_INSERT_SQL_WITHOUT_CRAWL_RUN_ID = text("""
    INSERT INTO raw_articles (
        source_type, source_name, publisher, title, content, url, url_hash,
        published_at, collected_at, company, language, content_type,
        crawl_status, error_message, processing_status, metadata
    ) VALUES (
        :source_type, :source_name, :publisher, :title, :content, :url, :url_hash,
        :published_at, :collected_at, CAST(:company AS jsonb), :language, :content_type,
        :crawl_status, :error_message, 'RAW', CAST(:metadata AS jsonb)
    )
    ON CONFLICT (url) DO NOTHING
    RETURNING id
""")


def save_articles(
    articles: list[RawArticle],
    run_context: CrawlRunContext | None = None,
) -> int:
    """RawArticle 목록을 raw_articles 테이블에 저장. 중복 URL은 스킵.

    Returns:
        실제 삽입된 건수
    """
    if not articles:
        return 0

    inserted = 0
    with SessionLocal() as db:
        for article in articles:
            storage_company = _company_for_storage(article)
            if not _is_valid(article, storage_company):
                continue
            try:
                sanitized_title = _sanitize_text(article.title)[:500]
                sanitized_content = _sanitize_text(article.content if article.content else "")
                source_metadata = _source_metadata_json(article, storage_company)
                insert_params = {
                    "source_type": article.source_type,
                    "source_name": article.source_name,
                    "publisher": article.publisher,
                    "title": sanitized_title,
                    "content": sanitized_content,
                    "url": article.url,
                    "url_hash": article.url_hash,
                    "published_at": article.published_at or article.collected_at,
                    "collected_at": article.collected_at,
                    "company": json.dumps(storage_company, ensure_ascii=False),
                    "language": article.language,
                    "content_type": article.content_type,
                    "crawl_status": article.crawl_status,
                    "error_message": article.error_message,
                    "metadata": source_metadata,
                    "crawl_run_id": run_context.crawl_run_id if run_context else None,
                }
                try:
                    result = db.execute(_INSERT_SQL, insert_params)
                except Exception as exc:  # noqa: BLE001
                    if not _is_missing_column(exc, "crawl_run_id"):
                        raise
                    db.rollback()
                    log.info(
                        "raw_articles.crawl_run_id 미적용 DB 감지 → legacy INSERT 사용 | url=%s",
                        article.url,
                    )
                    result = db.execute(_INSERT_SQL_WITHOUT_CRAWL_RUN_ID, insert_params)
                row = result.fetchone()
                if row:
                    article_id = row[0]
                    inserted += 1
                    action = "inserted"
                else:
                    existing = db.execute(
                        _SELECT_ARTICLE_ID_BY_URL,
                        {"url": article.url},
                    ).fetchone()
                    article_id = existing[0] if existing else None
                    action = "duplicate"
                if article_id:
                    _update_dart_content_if_better(
                        db,
                        article_id=article_id,
                        article=article,
                        sanitized_content=sanitized_content,
                    )
                    _update_ir_content_if_better(
                        db,
                        article_id=article_id,
                        article=article,
                        sanitized_content=sanitized_content,
                    )
                    _update_company_analysis_if_changed(
                        db,
                        article_id=article_id,
                        article=article,
                        sanitized_title=sanitized_title,
                        sanitized_content=sanitized_content,
                        source_metadata=source_metadata,
                    )
                    _upsert_source_metadata(
                        db,
                        article_id=article_id,
                        source_type=article.source_type,
                        source_metadata=source_metadata,
                    )
                    _upsert_market_price_ohlcv_for_article(db, article_id, article)
                    _upsert_crawl_run_article(
                        db,
                        article_id=article_id,
                        article=article,
                        run_context=run_context,
                        action=action,
                        source_metadata=source_metadata,
                    )
            except Exception as e:
                log.error("raw_articles 저장 실패 | url=%s error=%s", article.url, e)
                db.rollback()
        db.commit()

    log.info(
        "raw_articles 저장 완료 | total=%d inserted=%d skipped=%d",
        len(articles),
        inserted,
        len(articles) - inserted,
    )
    return inserted


# ──────────────────────────────────────────────────────────────
# 조회
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# 업데이트 — 파이프라인 각 단계에서 호출
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# 카드 뉴스 저장 (구 issue_cards → V9 에서 card_news 로 rename)
# ──────────────────────────────────────────────────────────────

# V33 이후 — 모든 v2 컬럼 (`peer_company_id` / `primary_keyword_category` /
# `source_raw_article_ids` / `evidence_payload` / `card_schema_version` /
# `evaluation_payload`) 을 INSERT 시점에 한 번에 채운다.
_INSERT_CARD_NEWS_V2 = text("""
    INSERT INTO card_news (
        id, company, cluster_id, title, summary_lines,
        event_type, importance, importance_score,
        implication, sources, validation_pass, validation_sc_score,
        peer_company_id, primary_keyword_category, integrated_issue_id, source_raw_article_ids,
        keyword_categories, evidence_payload, source_articles,
        card_schema_version, evaluation_payload, created_at
    ) VALUES (
        :id, :company, :cluster_id, :title, :summary_lines,
        :event_type, :importance, :importance_score,
        CAST(:implication AS jsonb), CAST(:sources AS jsonb),
        :validation_pass, :validation_sc_score,
        :peer_company_id, :primary_keyword_category, CAST(:integrated_issue_id AS uuid),
        CAST(:source_raw_article_ids AS bigint[]),
        CAST(:keyword_categories AS jsonb),
        CAST(:evidence_payload AS jsonb),
        CAST(:source_articles AS jsonb),
        :card_schema_version,
        CAST(:evaluation_payload AS jsonb),
        CAST(:created_at AS timestamptz)
    )
    ON CONFLICT (id) DO UPDATE SET
        status = 'ACTIVE',
        title = EXCLUDED.title,
        summary_lines = EXCLUDED.summary_lines,
        event_type = EXCLUDED.event_type,
        importance = EXCLUDED.importance,
        importance_score = EXCLUDED.importance_score,
        implication = CAST(:implication AS jsonb),
        sources = CAST(:sources AS jsonb),
        validation_pass = :validation_pass,
        validation_sc_score = :validation_sc_score,
        peer_company_id = COALESCE(EXCLUDED.peer_company_id, card_news.peer_company_id),
        primary_keyword_category = COALESCE(
            EXCLUDED.primary_keyword_category, card_news.primary_keyword_category
        ),
        integrated_issue_id = COALESCE(EXCLUDED.integrated_issue_id, card_news.integrated_issue_id),
        source_raw_article_ids = COALESCE(
            EXCLUDED.source_raw_article_ids, card_news.source_raw_article_ids
        ),
        keyword_categories = COALESCE(EXCLUDED.keyword_categories, card_news.keyword_categories),
        evidence_payload = COALESCE(EXCLUDED.evidence_payload, card_news.evidence_payload),
        source_articles = COALESCE(EXCLUDED.source_articles, card_news.source_articles),
        card_schema_version = EXCLUDED.card_schema_version,
        evaluation_payload =
            COALESCE(card_news.evaluation_payload, '{}'::jsonb)
            || COALESCE(EXCLUDED.evaluation_payload, '{}'::jsonb),
        created_at = EXCLUDED.created_at
    RETURNING id
""")

_INSERT_CARD_NEWS_V2_WITHOUT_INTEGRATED_ISSUE = text("""
    INSERT INTO card_news (
        id, company, cluster_id, title, summary_lines,
        event_type, importance, importance_score,
        implication, sources, validation_pass, validation_sc_score,
        peer_company_id, primary_keyword_category, source_raw_article_ids,
        keyword_categories, evidence_payload, source_articles,
        card_schema_version, evaluation_payload, created_at
    ) VALUES (
        :id, :company, :cluster_id, :title, :summary_lines,
        :event_type, :importance, :importance_score,
        CAST(:implication AS jsonb), CAST(:sources AS jsonb),
        :validation_pass, :validation_sc_score,
        :peer_company_id, :primary_keyword_category,
        CAST(:source_raw_article_ids AS bigint[]),
        CAST(:keyword_categories AS jsonb),
        CAST(:evidence_payload AS jsonb),
        CAST(:source_articles AS jsonb),
        :card_schema_version,
        CAST(:evaluation_payload AS jsonb),
        CAST(:created_at AS timestamptz)
    )
    ON CONFLICT (id) DO UPDATE SET
        status = 'ACTIVE',
        title = EXCLUDED.title,
        summary_lines = EXCLUDED.summary_lines,
        event_type = EXCLUDED.event_type,
        importance = EXCLUDED.importance,
        importance_score = EXCLUDED.importance_score,
        implication = CAST(:implication AS jsonb),
        sources = CAST(:sources AS jsonb),
        validation_pass = :validation_pass,
        validation_sc_score = :validation_sc_score,
        peer_company_id = COALESCE(EXCLUDED.peer_company_id, card_news.peer_company_id),
        primary_keyword_category = COALESCE(
            EXCLUDED.primary_keyword_category, card_news.primary_keyword_category
        ),
        source_raw_article_ids = COALESCE(
            EXCLUDED.source_raw_article_ids, card_news.source_raw_article_ids
        ),
        keyword_categories = COALESCE(EXCLUDED.keyword_categories, card_news.keyword_categories),
        evidence_payload = COALESCE(EXCLUDED.evidence_payload, card_news.evidence_payload),
        source_articles = COALESCE(EXCLUDED.source_articles, card_news.source_articles),
        card_schema_version = EXCLUDED.card_schema_version,
        evaluation_payload =
            COALESCE(card_news.evaluation_payload, '{}'::jsonb)
            || COALESCE(EXCLUDED.evaluation_payload, '{}'::jsonb),
        created_at = EXCLUDED.created_at
    RETURNING id
""")

_INSERT_CARD_NEWS_ARTICLE = text("""
    INSERT INTO card_news_articles (
        card_news_id,
        raw_article_id,
        article_order,
        relation_source
    ) VALUES (
        :card_news_id,
        :raw_article_id,
        :article_order,
        :relation_source
    )
    ON CONFLICT (card_news_id, raw_article_id) DO UPDATE SET
        article_order = EXCLUDED.article_order,
        relation_source = EXCLUDED.relation_source
""")

# pre-V33 환경 fallback (v2 컬럼 미존재 시). 신규 컬럼 6개 빼고 INSERT.
_INSERT_CARD_NEWS_V1_FALLBACK = text("""
    INSERT INTO card_news (
        id, company, cluster_id, title, summary_lines,
        event_type, importance, importance_score,
        implication, sources, validation_pass, validation_sc_score
    ) VALUES (
        :id, :company, :cluster_id, :title, :summary_lines,
        :event_type, :importance, :importance_score,
        CAST(:implication AS jsonb), CAST(:sources AS jsonb),
        :validation_pass, :validation_sc_score
    )
    ON CONFLICT (id) DO UPDATE SET
        implication = CAST(:implication AS jsonb),
        validation_pass = :validation_pass,
        validation_sc_score = :validation_sc_score
    RETURNING id
""")


def save_card_news(card: dict[str, Any]) -> Optional[str]:
    """카드 뉴스를 card_news 테이블에 저장한다.

    외부 리뷰 R-2 반영 (2026-05-21) — 모든 v2 컬럼을 단일 INSERT 로 채운다.
    더 이상 두 번째 UPDATE 가 필요하지 않으며, FK / sector / provenance / schema /
    evaluation 모두 카드 INSERT 시점에 일관 보장된다.

    호환 fallback — V33 미적용 환경 (`card_schema_version` 컬럼 없음) 에서는
    `UndefinedColumn` 발생 → 기존 v1 컬럼만 사용하는 fallback INSERT 로 자동 재시도.
    """
    try:
        if _is_self_company_card(card):
            log.info(
                "카드 뉴스 저장 제외 | id=%s company=%s reason=self company",
                card.get("id"),
                card.get("company") or card.get("peer_id") or card.get("peer_company_id"),
            )
            return None
        params = _card_news_insert_params(card)
        insert_card_id = str(params["id"])
        try:
            card_id = _execute_v2_insert(card_id=insert_card_id, params=params)
        except Exception as exc:  # noqa: BLE001
            if not _is_undefined_column_error(exc):
                raise
            log.info(
                (
                    "card_news v2 INSERT 실패 (V40 integrated_issue_id 미적용 가능) "
                    "→ legacy v2 재시도 | id=%s"
                ),
                card.get("id"),
            )
            try:
                card_id = _execute_v2_without_integrated_issue(
                    card_id=insert_card_id,
                    params=params,
                )
            except Exception as legacy_exc:  # noqa: BLE001
                if not _is_undefined_column_error(legacy_exc):
                    raise
                log.info(
                    "card_news legacy v2 INSERT 실패 (v33 미적용) → v1 fallback 사용 | id=%s",
                    card.get("id"),
                )
                card_id = _execute_v1_fallback(card_id=insert_card_id, params=params)
        if card_id:
            _sync_card_news_articles(
                card_id=card_id,
                source_raw_article_ids=params["source_raw_article_ids"],
                relation_source="source_raw_article_ids",
            )
            sync_card_sources_for_cluster(card.get("cluster_id"))
        return card_id
    except Exception as e:
        log.error("카드 뉴스 저장 실패 | id=%s error=%s", card.get("id"), e)
    return None


def _execute_v2_insert(*, card_id: str, params: dict[str, Any]) -> Optional[str]:
    with SessionLocal() as db:
        result = db.execute(_INSERT_CARD_NEWS_V2, params)
        row = result.fetchone()
        db.commit()
        if row:
            log.info("카드 뉴스 저장 완료 (v2) | id=%s", card_id)
            return card_id
    return None


def _execute_v2_without_integrated_issue(*, card_id: str, params: dict[str, Any]) -> Optional[str]:
    legacy_params = dict(params)
    legacy_params.pop("integrated_issue_id", None)
    with SessionLocal() as db:
        result = db.execute(_INSERT_CARD_NEWS_V2_WITHOUT_INTEGRATED_ISSUE, legacy_params)
        row = result.fetchone()
        db.commit()
        if row:
            log.info("카드 뉴스 저장 완료 (v2 legacy no integrated_issue_id) | id=%s", card_id)
            return card_id
    return None


def _execute_v1_fallback(*, card_id: str, params: dict[str, Any]) -> Optional[str]:
    legacy_params = {
        key: params[key]
        for key in (
            "id",
            "company",
            "cluster_id",
            "title",
            "summary_lines",
            "event_type",
            "importance",
            "importance_score",
            "implication",
            "sources",
            "validation_pass",
            "validation_sc_score",
        )
    }
    with SessionLocal() as db:
        result = db.execute(_INSERT_CARD_NEWS_V1_FALLBACK, legacy_params)
        row = result.fetchone()
        db.commit()
        if row:
            log.info("카드 뉴스 저장 완료 (v1 fallback) | id=%s", card_id)
            return card_id
    return None


def _card_news_insert_params(card: dict[str, Any]) -> dict[str, Any]:
    source_ids = _normalize_int_list(card.get("source_raw_article_ids"))
    if not source_ids:
        source_ids = _source_ids_from_sources(card.get("sources"))
    evidence_payload = _build_evidence_payload(card)
    merge_card = dict(card)
    merge_card["evidence_payload"] = evidence_payload
    implication_payload = _merge_implication_payload(merge_card)
    evaluation_payload = card.get("evaluation_payload") or {}
    if not isinstance(evaluation_payload, dict):
        evaluation_payload = {}
    keyword_categories = card.get("keyword_categories") or {}
    if not isinstance(keyword_categories, dict):
        keyword_categories = {}
    source_articles = _source_articles_payload(card, source_ids)
    created_at = _card_created_at_param(card, source_articles)
    return {
        "id": _canonical_card_id(card["id"], created_at),
        "company": card.get("company") or card.get("peer_id") or _resolve_peer_company_id(card),
        "cluster_id": card.get("cluster_id"),
        "title": card["title"][:500],
        "summary_lines": card.get("summary_lines", []),
        "event_type": card.get("event_type", "tech"),
        "importance": card.get("importance", "low"),
        "importance_score": card.get("importance_score", 0.0),
        "implication": json.dumps(implication_payload, ensure_ascii=False, default=str),
        "sources": json.dumps(card.get("sources", []), ensure_ascii=False, default=str),
        "validation_pass": card.get("validation", {}).get("pass", False),
        "validation_sc_score": card.get("validation", {}).get("sc_score", 0.0),
        "peer_company_id": _resolve_peer_company_id(card),
        "primary_keyword_category": card.get("primary_keyword_category") or card.get("sector"),
        "integrated_issue_id": _resolve_integrated_issue_id(card, evidence_payload),
        "source_raw_article_ids": source_ids,
        "keyword_categories": json.dumps(keyword_categories, ensure_ascii=False, default=str),
        "evidence_payload": json.dumps(evidence_payload, ensure_ascii=False, default=str),
        "source_articles": json.dumps(source_articles, ensure_ascii=False, default=str),
        "card_schema_version": str(card.get("card_schema_version") or "v2"),
        "evaluation_payload": json.dumps(evaluation_payload, ensure_ascii=False, default=str),
        "created_at": created_at,
    }


def _sync_card_news_articles(
    *,
    card_id: str,
    source_raw_article_ids: list[int],
    relation_source: str,
) -> None:
    """Keep normalized card -> raw article links in step with card_news."""
    ids = _normalize_int_list(source_raw_article_ids)
    if not ids:
        return
    try:
        with SessionLocal() as db:
            db.execute(
                text("""
                    DELETE FROM card_news_articles
                    WHERE card_news_id = :card_news_id
                      AND NOT (raw_article_id = ANY(CAST(:source_raw_article_ids AS bigint[])))
                """),
                {
                    "card_news_id": card_id,
                    "source_raw_article_ids": ids,
                },
            )
            db.execute(
                _INSERT_CARD_NEWS_ARTICLE,
                [
                    {
                        "card_news_id": card_id,
                        "raw_article_id": raw_id,
                        "article_order": index,
                        "relation_source": relation_source,
                    }
                    for index, raw_id in enumerate(ids, start=1)
                ],
            )
            db.commit()
    except Exception as e:  # noqa: BLE001
        if _is_missing_card_news_articles_table(e):
            log.info(
                "card_news_articles table unavailable; skip normalized card/article sync | "
                "card_id=%s article_ids=%s",
                card_id,
                ids,
            )
            return
        log.warning(
            "card_news_articles 동기화 실패 | card_id=%s article_ids=%s error=%s",
            card_id,
            ids,
            e,
        )


def sync_card_sources_for_cluster(cluster_id: Any) -> int:
    """Sync ACTIVE card provenance from all relevant processed raw articles in a cluster."""
    try:
        normalized_cluster_id = int(cluster_id)
    except (TypeError, ValueError):
        return 0
    try:
        with SessionLocal() as db:
            cards = [
                dict(row)
                for row in db.execute(
                    text(
                        """
                        SELECT id, peer_company_id, company
                        FROM card_news
                        WHERE status = 'ACTIVE'
                          AND cluster_id = :cluster_id
                        """
                    ),
                    {"cluster_id": normalized_cluster_id},
                )
                .mappings()
                .all()
            ]
            if not cards:
                return 0

            articles = [
                dict(row)
                for row in db.execute(
                    text(
                        """
                        SELECT id, title, content, url, source_name, publisher, published_at,
                               collected_at, company, matched_companies
                        FROM raw_articles
                        WHERE cluster_id = :cluster_id
                          AND source_type = 'news'
                          AND processing_status = 'PROCESSED'
                          AND relevance_label = 'relevant'
                        ORDER BY published_at DESC NULLS LAST,
                                 collected_at DESC NULLS LAST,
                                 id DESC
                        """
                    ),
                    {"cluster_id": normalized_cluster_id},
                )
                .mappings()
                .all()
            ]
            updated = 0
            for card in cards:
                scoped_articles = _filter_articles_for_card_company(articles, card)
                if not scoped_articles:
                    continue
                raw_ids = [int(article["id"]) for article in scoped_articles]
                sources = [
                    {**_source_dict_from_article(article), "index": idx}
                    for idx, article in enumerate(scoped_articles, start=1)
                ]
                source_articles = [
                    _source_article_dict_from_article(article) for article in scoped_articles
                ]
                result = db.execute(
                    text(
                        """
                        UPDATE card_news
                        SET source_raw_article_ids = CAST(:raw_ids AS bigint[]),
                            sources = CAST(:sources AS jsonb),
                            source_articles = CAST(:source_articles AS jsonb)
                        WHERE id = :id
                        """
                    ),
                    {
                        "id": card["id"],
                        "raw_ids": raw_ids,
                        "sources": json.dumps(sources, ensure_ascii=False, default=str),
                        "source_articles": json.dumps(
                            source_articles, ensure_ascii=False, default=str
                        ),
                    },
                )
                updated += int(getattr(result, "rowcount", 0) or 0)
            db.commit()
            return updated
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "card_news cluster source sync 실패 | cluster_id=%s error=%s",
            cluster_id,
            exc,
        )
        return 0


def _legacy_sync_card_sources_for_cluster(cluster_id: int) -> int:
    """Old cluster-wide sync retained for manual debugging, not used in write paths."""
    try:
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
                          AND ra.source_type = 'news'
                          AND ra.processing_status = 'PROCESSED'
                          AND ra.relevance_label = 'relevant'
                    ),
                    cluster_sources AS (
                        SELECT
                            cluster_id,
                            ARRAY_AGG(
                                id
                                ORDER BY
                                    published_at DESC NULLS LAST,
                                    collected_at DESC NULLS LAST,
                                    id DESC
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
                                ORDER BY
                                    published_at DESC NULLS LAST,
                                    collected_at DESC NULLS LAST,
                                    id DESC
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
                                ORDER BY
                                    published_at DESC NULLS LAST,
                                    collected_at DESC NULLS LAST,
                                    id DESC
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
    except Exception as exc:  # noqa: BLE001
        log.warning("legacy card source sync 실패 | cluster_id=%s error=%s", cluster_id, exc)
        return 0


def _merge_implication_payload(card: dict[str, Any]) -> dict[str, Any]:
    """ImplicationAgent v4.0/v5.0 결과 + 보조 메타데이터를 단일 JSONB 로 합성.

    v2 schema 가 우선. legacy v3_payload (sector / exposure / signals) 는 보조 key 로
    함께 보존하여 frontend / sidecar 가 둘 다 읽을 수 있게 한다.
    """
    payload: dict[str, Any] = {}
    candidates = _implication_payload_candidates(card)
    for candidate in candidates:
        if _has_structured_implication(candidate):
            payload = dict(candidate)
            break
    if not _frontend_has_display_items(payload.get("frontend")):
        payload.pop("frontend", None)
    if not payload.get("frontend_ready"):
        for candidate in candidates:
            ready = candidate.get("frontend_ready")
            if _frontend_implication_from_frontend_ready(ready):
                payload["frontend_ready"] = ready
                break
    for key in ("peer_implication", "skax_implication"):
        if not isinstance(payload.get(key), dict) or not payload.get(key):
            for candidate in candidates:
                value = candidate.get(key)
                if isinstance(value, dict) and value:
                    payload[key] = value
                    break
    ready_frontend = _frontend_implication_from_frontend_ready(payload.get("frontend_ready"))
    if ready_frontend:
        payload["frontend"] = ready_frontend
        if ready_frontend.get("suggested_actions"):
            payload.setdefault("recommended_actions", ready_frontend["suggested_actions"])
            skax = payload.get("skax_implication")
            if isinstance(skax, dict):
                skax.setdefault("recommended_actions", ready_frontend["suggested_actions"])
    if not _frontend_has_display_items(payload.get("frontend")):
        industry_frontend = _frontend_implication_from_industry_frontend_ready(
            payload.get("industry_frontend_ready")
        )
        if industry_frontend:
            payload["frontend"] = industry_frontend
            if industry_frontend.get("suggested_actions"):
                payload.setdefault("recommended_actions", industry_frontend["suggested_actions"])
                skax = payload.get("skax_implication")
                if isinstance(skax, dict):
                    skax.setdefault("recommended_actions", industry_frontend["suggested_actions"])
    if not _frontend_has_display_items(payload.get("frontend")):
        frontend = card.get("frontend_implication")
        if isinstance(frontend, dict) and frontend:
            payload["frontend"] = dict(frontend)
            if frontend.get("suggested_actions"):
                payload.setdefault("recommended_actions", frontend.get("suggested_actions"))
                skax = payload.get("skax_implication")
                if isinstance(skax, dict):
                    skax.setdefault("recommended_actions", frontend.get("suggested_actions"))
    if not _frontend_has_display_items(payload.get("frontend")):
        display_frontend = _frontend_implication_from_display_sections(card.get("display_sections"))
        if display_frontend:
            payload["frontend"] = display_frontend
            if display_frontend.get("suggested_actions"):
                payload.setdefault("recommended_actions", display_frontend["suggested_actions"])
    if not _frontend_has_display_items(payload.get("frontend")):
        structured_frontend = _frontend_implication_from_structured_implication(
            payload,
            context=card,
        )
        if structured_frontend:
            payload["frontend"] = structured_frontend
            if structured_frontend.get("suggested_actions"):
                payload.setdefault("recommended_actions", structured_frontend["suggested_actions"])
    # 보조 메타데이터 (sector / exposure / signals / evidence_chain) 는 별도 namespace.
    payload.setdefault(
        "sector_meta",
        {
            "sector": card.get("sector", "other"),
            "sectors": card.get("sectors", ["other"]),
            "exposure_score": card.get("exposure_score", 0.0),
            "exposure_band": card.get("exposure_band", "low"),
            "signals": card.get("signals", {}),
            "evidence_chain": card.get("evidence_chain", {}),
        },
    )
    return payload


def _frontend_has_display_items(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for key in (
        "key_implication_items",
        "suggested_action_items",
        "key_implication_blocks",
        "response_direction_blocks",
    ):
        if value.get(key):
            return True
    return bool(value.get("key_implications") or value.get("suggested_actions"))


def _frontend_implication_from_structured_implication(
    value: Any,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    peer_value = value.get("peer_implication")
    skax_value = value.get("skax_implication")
    peer: dict[str, Any] = peer_value if isinstance(peer_value, dict) else {}
    skax: dict[str, Any] = skax_value if isinstance(skax_value, dict) else {}
    issue_lines = _integrated_issue_context_lines(context)
    key_main = str(
        peer.get("peer_meaning")
        or peer.get("capability_change")
        or value.get("why_important")
        or ""
    ).strip()
    key_detail = str(peer.get("capability_change") or value.get("potential_impact") or "").strip()
    if _looks_like_fragment(key_detail) and issue_lines:
        key_detail = issue_lines[0]
    actions = _string_list(skax.get("recommended_actions") or value.get("recommended_actions"))
    action_main = actions[0] if actions else str(skax.get("why_important") or "").strip()
    action_detail = str(skax.get("potential_impact") or skax.get("why_important") or "").strip()
    if _looks_like_fragment(action_detail) and len(issue_lines) >= 2:
        action_detail = issue_lines[1]
    key_blocks = [{"main": key_main, "detail": key_detail}] if key_main else []
    action_blocks = [{"main": action_main, "detail": action_detail}] if action_main else []
    if not key_blocks and not action_blocks:
        return {}
    key_items = _labeled_lines_from_main_detail_blocks(key_blocks, prefix="핵심 시사점")
    action_items = _labeled_lines_from_main_detail_blocks(action_blocks, prefix="핵심 대응")
    payload: dict[str, Any] = {
        "source": "structured_implication_fallback",
        "key_implications": key_items,
        "peer_implications": key_items,
        "suggested_actions": action_items,
        "response_directions": action_items,
        "key_implication_blocks": key_blocks,
        "key_implication_items": key_blocks,
        "response_direction_blocks": action_blocks,
        "suggested_action_items": action_blocks,
        "follow_up_questions": [],
    }
    if key_items:
        payload["potential_impact"] = key_items[0]
    return payload


def _implication_payload_candidates(card: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for value in (card.get("implication"), card.get("implication_result")):
        if isinstance(value, dict):
            candidates.append(value)
    analysis_package = card.get("analysis_package")
    if isinstance(analysis_package, dict):
        for value in (
            analysis_package.get("implication"),
            (analysis_package.get("evidence_payload") or {}).get("implication")
            if isinstance(analysis_package.get("evidence_payload"), dict)
            else None,
        ):
            if isinstance(value, dict):
                candidates.append(value)
    evidence_payload = card.get("evidence_payload")
    if isinstance(evidence_payload, dict):
        package = evidence_payload.get("analysis_package")
        if isinstance(package, dict) and isinstance(package.get("implication"), dict):
            candidates.append(package["implication"])
    return candidates


def _has_structured_implication(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "skax_implication",
            "peer_implication",
            "frontend",
            "industry_frontend_ready",
            "recommended_actions",
            "watch_points",
            "follow_up_questions",
        )
    )


def _frontend_implication_from_display_sections(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {}
    sections: dict[str, list[str]] = {}
    structured_sections: dict[str, list[dict[str, str]]] = {}
    for section in value:
        if not isinstance(section, dict):
            continue
        section_type = str(section.get("type") or "").strip()
        items = [
            str(item).strip() for item in (section.get("items") or []) if str(item or "").strip()
        ]
        if section_type and items:
            sections[section_type] = items
        structured_items = _main_detail_blocks(section.get("structured_items"))
        if section_type and structured_items:
            structured_sections[section_type] = structured_items
    insight_items = sections.get("insight") or []
    action_items = sections.get("action") or []
    if not insight_items and not action_items:
        return {}
    insight_blocks = structured_sections.get("insight") or _main_detail_blocks_from_labeled_lines(
        insight_items
    )
    action_blocks = structured_sections.get("action") or _main_detail_blocks_from_labeled_lines(
        action_items
    )
    payload: dict[str, Any] = {
        "key_implications": insight_items,
        "peer_implications": insight_items,
        "suggested_actions": action_items,
        "response_directions": action_items,
        "key_implication_blocks": insight_blocks,
        "key_implication_items": insight_blocks,
        "response_direction_blocks": action_blocks,
        "suggested_action_items": action_blocks,
        "follow_up_questions": [],
    }
    if insight_items:
        payload["potential_impact"] = insight_items[0]
    return payload


def _frontend_implication_from_frontend_ready(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    key_block = value.get("key_implication")
    action_block = value.get("suggested_action")
    key_items = _frontend_ready_display_lines(key_block, prefix="핵심 시사점")
    action_items = _frontend_ready_display_lines(action_block, prefix="핵심 대응")
    if not key_items and not action_items:
        return {}
    key_blocks = _frontend_ready_display_blocks(key_block)
    action_blocks = _frontend_ready_display_blocks(action_block)
    payload: dict[str, Any] = {
        "source": value.get("source") or "frontend_ready",
        "key_implications": key_items,
        "peer_implications": key_items,
        "suggested_actions": action_items,
        "response_directions": action_items,
        "key_implication_blocks": key_blocks,
        "key_implication_items": key_blocks,
        "response_direction_blocks": action_blocks,
        "suggested_action_items": action_blocks,
        "follow_up_questions": [],
    }
    if key_items:
        payload["potential_impact"] = key_items[0]
    return payload


def _frontend_implication_from_industry_frontend_ready(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if str(value.get("source") or "").strip() != "industry_signal_direct":
        return {}
    insight_items: list[str] = []
    action_items: list[str] = []
    insight_blocks: list[dict[str, str]] = []
    action_blocks: list[dict[str, str]] = []
    for item in value.get("items") or []:
        if not isinstance(item, dict):
            continue
        key_block = item.get("key_implication")
        action_block = item.get("suggested_action")
        insight_items.extend(_frontend_ready_display_lines(key_block, prefix="핵심 시사점"))
        action_items.extend(_frontend_ready_display_lines(action_block, prefix="핵심 대응"))
        insight_blocks.extend(_frontend_ready_display_blocks(key_block))
        action_blocks.extend(_frontend_ready_display_blocks(action_block))
    insight_items = insight_items[:2]
    action_items = action_items[:2]
    insight_blocks = insight_blocks[:2]
    action_blocks = action_blocks[:2]
    if not insight_items and not action_items:
        return {}
    payload: dict[str, Any] = {
        "source": "industry_signal_direct",
        "signal_scope": str(value.get("signal_scope") or "industry_signal"),
        "display_policy": str(value.get("display_policy") or "industry_only"),
        "key_implications": insight_items,
        "peer_implications": insight_items,
        "suggested_actions": action_items,
        "response_directions": action_items,
        "key_implication_blocks": insight_blocks,
        "key_implication_items": insight_blocks,
        "response_direction_blocks": action_blocks,
        "suggested_action_items": action_blocks,
        "follow_up_questions": [],
    }
    if insight_items:
        payload["potential_impact"] = insight_items[0]
    if action_items:
        payload["recommended_action"] = action_items[0]
    return payload


def _frontend_ready_display_lines(value: Any, *, prefix: str) -> list[str]:
    if not isinstance(value, dict):
        return []
    sentence = str(value.get("sentence") or "").strip()
    evidence = str(value.get("evidence_sentence") or "").strip()
    if not sentence:
        return []
    lines = [f"{prefix}: {sentence}"]
    if evidence:
        lines.append(f"근거: {evidence}")
    return lines


def _frontend_ready_display_blocks(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        return []
    sentence = str(value.get("sentence") or "").strip()
    evidence = str(value.get("evidence_sentence") or "").strip()
    if not sentence:
        return []
    return [{"main": sentence, "detail": evidence}]


def merge_card_news_sources_for_cluster(
    cluster_id: int,
    raw_article_ids: list[int],
    *,
    importance_score: float | None = None,
) -> Optional[str]:
    """기존 클러스터 카드에 새 원문 기사 출처를 병합한다.

    같은 주제의 새 기사가 기존 cluster_id에 편입될 때 새 card_news를 만들지 않고,
    기존 카드의 sources/source_raw_article_ids/source_articles를 보강한다.
    """
    if not raw_article_ids:
        return None

    articles = get_articles_by_ids(raw_article_ids)
    if not articles:
        return None

    try:
        with SessionLocal() as db:
            row = db.execute(
                text("""
                    SELECT id, sources, source_raw_article_ids, source_articles,
                           importance_score, peer_company_id, company
                    FROM card_news
                    WHERE cluster_id = :cluster_id
                      AND status <> 'DELETED'
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"cluster_id": cluster_id},
            ).fetchone()
            if row is None:
                return None

            current = dict(row._mapping)
            articles = _filter_articles_for_card_company(articles, current)
            if not articles:
                log.info(
                    "기존 카드 출처 병합 skip | card_id=%s cluster_id=%s reason=company_mismatch",
                    current["id"],
                    cluster_id,
                )
                return None
            existing_sources = _json_or_value(current.get("sources"), [])
            existing_source_articles = _json_or_value(current.get("source_articles"), [])
            existing_ids = [
                int(value)
                for value in (current.get("source_raw_article_ids") or [])
                if value is not None
            ]

            merged_sources = _merge_source_dicts(
                existing_sources if isinstance(existing_sources, list) else [],
                [_source_dict_from_article(article) for article in articles],
            )
            merged_source_articles = _merge_source_dicts(
                existing_source_articles if isinstance(existing_source_articles, list) else [],
                [_source_article_dict_from_article(article) for article in articles],
            )
            merged_ids = sorted({*existing_ids, *(int(article["id"]) for article in articles)})
            merged_importance = max(
                float(current.get("importance_score") or 0.0),
                float(importance_score or 0.0),
            )

            db.execute(
                text("""
                    UPDATE card_news
                    SET sources = CAST(:sources AS jsonb),
                        source_raw_article_ids = CAST(:source_raw_article_ids AS bigint[]),
                        source_articles = CAST(:source_articles AS jsonb),
                        importance_score = GREATEST(
                            COALESCE(importance_score, 0),
                            :importance_score
                        )
                    WHERE id = :id
                """),
                {
                    "id": current["id"],
                    "sources": json.dumps(merged_sources, ensure_ascii=False),
                    "source_raw_article_ids": merged_ids,
                    "source_articles": json.dumps(merged_source_articles, ensure_ascii=False),
                    "importance_score": merged_importance,
                },
            )
            db.commit()
            _sync_card_news_articles(
                card_id=str(current["id"]),
                source_raw_article_ids=merged_ids,
                relation_source="source_raw_article_ids",
            )
            log.info(
                "기존 카드 출처 병합 완료 | card_id=%s cluster_id=%s added=%d",
                current["id"],
                cluster_id,
                len(articles),
            )
            return str(current["id"])
    except Exception as e:
        log.warning(
            "기존 카드 출처 병합 실패 | cluster_id=%s article_ids=%s error=%s",
            cluster_id,
            raw_article_ids,
            e,
        )
        return None


# ──────────────────────────────────────────────────────────────
# evidence_chain 저장
# ──────────────────────────────────────────────────────────────


# V9 (2026-05-12): card_news 테이블 rename 후에도 evidence_chain.issue_card_id 컬럼은
# legacy 이름 유지 (deploy race 회피 — 컬럼 rename 은 V10 분리). Python 변수명은
# card_news_id 로 의미 정렬, SQL column ref 만 issue_card_id 유지.


# ──────────────────────────────────────────────────────────────
# pipeline_logs 저장
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────


def _company_for_storage(article: RawArticle) -> list[str]:
    """DB 저장용 company를 반환한다.

    기업이 명시되지 않은 산업 동향 자료는 파이프라인 필터링을 위해
    가상 company bucket으로 저장한다. 일반 무소속 기사는 계속 제외한다.
    """
    if article.company:
        return article.company
    if _is_industry_trend_article(article):
        return [INDUSTRY_TREND_COMPANY]
    return []


# ──────────────────────────────────────────────────────────────────────────
# Global Trends — readers / upsert / trend context cache.
# design: axis-ai/design/30-analysis/global-trends.md §3.2 / §5.2 / §7.
# ──────────────────────────────────────────────────────────────────────────


# trend 는 cronjob 으로 일 1 회만 갱신되므로 60 초 캐시는 정합성 손실 거의 없음.
_TREND_CACHE_TTL_SEC = 60
_trend_cache: dict[int, tuple[float, dict[str, Any]]] = {}
_trend_cache_lock = threading.Lock()


def fetch_latest_trend_context(within_days: int = 7) -> dict[str, Any]:
    """``global_industry_trends`` 의 최근 N 일 row 를 TrendContext shape 로 aggregate.

    StrategicAnalyzer prompt 에 들어갈 글로벌 배경 정보. trend 가 없으면 빈 dict 반환
    (``strategic_analyzer.py`` 가 already-empty-safe). process-level TTL 60 초 캐시.
    """
    import time

    now = time.monotonic()
    with _trend_cache_lock:
        cached = _trend_cache.get(within_days)
        if cached and (now - cached[0]) < _TREND_CACHE_TTL_SEC:
            return cached[1]

    query = text(
        """
                SELECT keyword, summary, payload, trend_date, source_analysis_id
                FROM global_industry_trends
                WHERE trend_date >= CURRENT_DATE - make_interval(days => :days)
                ORDER BY impact_score DESC NULLS LAST, trend_date DESC
                LIMIT 10
                """
    )
    try:
        with SessionLocal() as db:
            rows = db.execute(query, {"days": within_days}).mappings().all()
    except Exception as exc:  # noqa: BLE001
        if not _is_missing_relation(exc, "global_industry_trends"):
            raise
        rows = []

    result: dict[str, Any]
    if not rows:
        result = {}
    else:
        signals: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        trend_lines: list[str] = []
        for row in rows:
            payload = row["payload"] if isinstance(row["payload"], dict) else {}
            summary = row["summary"] or ""
            if summary:
                trend_lines.append(summary)
            domestic_refs = [
                ref
                for ref in (payload.get("domestic_representative_issues") or [])
                if isinstance(ref, dict)
            ]
            domestic_ref_ids = [
                str(ref.get("card_news_id"))
                for ref in domestic_refs
                if str(ref.get("card_news_id") or "").strip()
            ]
            signals.append(
                {
                    "signal": row["keyword"],
                    "intensity": payload.get("intensity"),
                    "leading_companies": payload.get("leading_companies", []),
                    "source_ids": [row["source_analysis_id"]] if row["source_analysis_id"] else [],
                    "domestic_reference_ids": domestic_ref_ids,
                    "domestic_reference_count": len(domestic_refs),
                }
            )
            sources.append(
                {
                    "source_id": row["source_analysis_id"],
                    "title": row["keyword"],
                }
            )
            for ref in domestic_refs[:5]:
                card_id = str(ref.get("card_news_id") or "").strip()
                if not card_id:
                    continue
                sources.append(
                    {
                        "source_id": card_id,
                        "title": ref.get("title"),
                        "source_name": "domestic_card_news",
                        "url": ref.get("url"),
                        "published_at": ref.get("published_at"),
                        "cluster_id": ref.get("cluster_id"),
                    }
                )
        latest_date = rows[0]["trend_date"]
        has_domestic_refs = any(
            source.get("source_name") == "domestic_card_news" for source in sources
        )
        source_groups = ["global_industry_trends"]
        if has_domestic_refs:
            source_groups.append("domestic_card_news")
        result = {
            "period": f"last_{within_days}d",
            "trend_summary": " / ".join(trend_lines[:3]),
            "trend_lines": trend_lines,
            "signals": signals,
            "source_groups": source_groups,
            "sources": sources,
            "reference_issue_ids": [
                str(source.get("source_id"))
                for source in sources
                if source.get("source_name") == "domestic_card_news"
                and str(source.get("source_id") or "").strip()
            ],
            "updated_at": latest_date.isoformat()
            if hasattr(latest_date, "isoformat")
            else str(latest_date),
            "validation": {"pass": True},
            "metadata": {"row_count": len(rows)},
        }

    with _trend_cache_lock:
        _trend_cache[within_days] = (now, result)
    return result


def invalidate_trend_context_cache() -> None:
    """ITTrendAgent 가 cron 으로 새 trend 를 upsert 한 직후 호출하면 즉시 반영."""
    with _trend_cache_lock:
        _trend_cache.clear()
