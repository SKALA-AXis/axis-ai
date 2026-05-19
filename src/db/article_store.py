"""raw_articles / card_news 테이블 저장·조회·업데이트 레이어."""

import json
import logging
import re
import threading
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import text

from src.agents.credibility_agent import compute_credibility_score, credibility_grade
from src.crawler.base import CrawlRunContext, RawArticle
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

INDUSTRY_TREND_COMPANY = "industry_trend"
_INDUSTRY_SOURCE_TYPES = {"trend_report", "search_trend"}
_INDUSTRY_MARKER_KEYS = {"sector", "industry", "upjong_code"}
_INDUSTRY_REPORT_TYPES = {"industry", "industry_report", "sector_report"}

_INSERT_SQL = text("""
    INSERT INTO raw_articles (
        source_type, source_name, publisher, title, content, url, url_hash,
        published_at, collected_at, company, language, content_type,
        crawl_status, error_message, processing_status, crawl_run_id
    ) VALUES (
        :source_type, :source_name, :publisher, :title, :content, :url, :url_hash,
        :published_at, :collected_at, CAST(:company AS jsonb), :language, :content_type,
        :crawl_status, :error_message, 'RAW', CAST(:crawl_run_id AS uuid)
    )
    ON CONFLICT (url) DO NOTHING
    RETURNING id
""")

_SELECT_ARTICLE_ID_BY_URL = text("SELECT id FROM raw_articles WHERE url = :url")

_APPEND_CRAWL_EVENT = text("""
    UPDATE raw_articles
    SET crawl_events = COALESCE(crawl_events, '[]'::jsonb)
        || jsonb_build_array(CAST(:crawl_event AS jsonb))
    WHERE id = :raw_article_id
""")

_SOURCE_METADATA_EXCLUDED_KEYS = {
    "url_hash",
    "source_type",
    "content_type",
    "company_tier",
    "collection_mode",
    "crawl_run_id",
    "crawl_source_name",
    "track",
    "window_start",
    "window_end",
    "matched_companies",
    "matched_sectors",
}


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
                result = db.execute(
                    _INSERT_SQL,
                    {
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
                        "crawl_run_id": run_context.crawl_run_id if run_context else None,
                    },
                )
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
                    _upsert_source_metadata(
                        db,
                        article_id=article_id,
                        source_type=article.source_type,
                        source_name=article.source_name,
                        source_metadata=source_metadata,
                    )
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


def get_articles_by_ids(ids: list[int]) -> list[dict[str, Any]]:
    """raw_articles 테이블에서 ID 목록으로 기사를 조회한다."""
    if not ids:
        return []
    with SessionLocal() as db:
        rows = db.execute(
            text("""
                SELECT raw_articles.id, raw_articles.company, raw_articles.title,
                       raw_articles.content, raw_articles.url,
                       raw_articles.source_type, raw_articles.content_type,
                       raw_articles.publisher, raw_articles.language,
                       raw_articles.relevance_score, raw_articles.relevance_label,
                       raw_articles.relevance_reason,
                       raw_articles.matched_companies, raw_articles.matched_sectors,
                       raw_articles.source_name, raw_articles.published_at,
                       raw_articles.collected_at,
                       COALESCE(mu.metadata, '{}'::jsonb) AS metadata
                FROM raw_articles
                LEFT JOIN raw_article_metadata_unified mu
                    ON mu.raw_article_id = raw_articles.id
                WHERE raw_articles.id = ANY(:ids)
                ORDER BY raw_articles.published_at DESC NULLS LAST,
                         raw_articles.collected_at DESC
            """),
            {"ids": ids},
        ).fetchall()
    return [_with_credibility(dict(row._mapping)) for row in rows]


def list_dart_documents(
    *,
    company: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """프론트 DART 문서 목록에 필요한 파싱 요약을 조회한다."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    company_filter = "AND (:company IS NULL OR r.company @> CAST(:company_json AS jsonb))"
    rows = _fetch_dart_rows(
        select_sql="""
            SELECT r.id, r.company, r.title, r.url, r.published_at, r.collected_at,
                   r.processing_status, COALESCE(mu.metadata, '{}'::jsonb) AS metadata
            FROM raw_articles r
            LEFT JOIN raw_article_metadata_unified mu
                ON mu.raw_article_id = r.id
        """,
        where_sql=company_filter,
        order_limit_sql="""
            ORDER BY published_at DESC NULLS LAST, collected_at DESC
            LIMIT :limit OFFSET :offset
        """,
        params={
            "company": company,
            "company_json": json.dumps([company], ensure_ascii=False) if company else "[]",
            "limit": limit,
            "offset": offset,
        },
    )
    return [_dart_document_summary(dict(row._mapping)) for row in rows]


def get_dart_document_detail(article_id: int) -> dict[str, Any] | None:
    """DART 문서 상세 페이지에 필요한 파싱 결과를 조회한다."""
    rows = _fetch_dart_rows(
        select_sql="""
            SELECT r.id, r.company, r.title, r.content, r.url, r.published_at, r.collected_at,
                   r.processing_status, COALESCE(mu.metadata, '{}'::jsonb) AS metadata
            FROM raw_articles r
            LEFT JOIN raw_article_metadata_unified mu
                ON mu.raw_article_id = r.id
        """,
        where_sql="AND r.id = :id",
        order_limit_sql="LIMIT 1",
        params={"id": article_id},
    )
    if not rows:
        return None

    row = dict(rows[0]._mapping)
    metadata = _metadata_dict(row.get("metadata"))
    parser_result = _metadata_dict(metadata.get("parser_result"))
    document = _metadata_dict(parser_result.get("document"))
    return {
        **_dart_document_summary(row),
        "content": row.get("content"),
        "document": document,
        "sections": metadata.get("dart_sections") or parser_result.get("sections") or {},
        "section_tree": metadata.get("dart_section_tree")
        or parser_result.get("section_tree")
        or [],
        "document_chunks": metadata.get("dart_document_chunks")
        or parser_result.get("document_chunks")
        or [],
        "classified_tables": metadata.get("dart_classified_tables")
        or parser_result.get("classified_tables")
        or [],
        "financial_statements": metadata.get("dart_financial_statements")
        or parser_result.get("financial_statements")
        or [],
        "topic_signals": metadata.get("topic_signals") or parser_result.get("topic_signals") or [],
        "warnings": parser_result.get("warnings") or [],
    }


def _fetch_dart_rows(
    *,
    select_sql: str,
    where_sql: str,
    order_limit_sql: str,
    params: dict[str, Any],
) -> list[Any]:
    query = text(f"""
        {select_sql}
        WHERE r.source_type = 'dart'
          AND r.crawl_status = 'success'
          {where_sql}
        {order_limit_sql}
    """)
    with SessionLocal() as db:
        return list(db.execute(query, params).fetchall())


def _dart_document_summary(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata_dict(row.get("metadata"))
    parser_result = _metadata_dict(metadata.get("parser_result"))
    financial_record = _metadata_dict(
        metadata.get("financial_record") or parser_result.get("financial_record")
    )
    company = row.get("company")
    return {
        "id": row.get("id"),
        "company": company if isinstance(company, list) else _json_or_value(company, []),
        "title": row.get("title"),
        "url": row.get("url"),
        "published_at": _iso_or_none(row.get("published_at")),
        "collected_at": _iso_or_none(row.get("collected_at")),
        "processing_status": row.get("processing_status"),
        "rcept_no": metadata.get("rcept_no") or metadata.get("receipt_no"),
        "corp_code": metadata.get("corp_code"),
        "corp_name": metadata.get("corp_name"),
        "report_name": metadata.get("report_name") or metadata.get("report_nm") or row.get("title"),
        "period": metadata.get("period") or parser_result.get("period"),
        "period_year": metadata.get("period_year") or parser_result.get("period_year"),
        "period_quarter": metadata.get("period_quarter") or parser_result.get("period_quarter"),
        "period_type": metadata.get("period_type") or parser_result.get("period_type"),
        "document_text_length": metadata.get("document_text_length"),
        "table_count": metadata.get("table_count"),
        "structured_table_count": metadata.get("structured_table_count"),
        "section_count": len(metadata.get("dart_sections") or parser_result.get("sections") or {}),
        "financial_statement_count": financial_record.get("financial_statement_count")
        or len(metadata.get("dart_financial_statements") or []),
        "revenue_total_krwbn": financial_record.get("revenue_total_krwbn")
        or parser_result.get("revenue_total_krwbn"),
        "operating_profit_krwbn": financial_record.get("operating_profit_krwbn")
        or parser_result.get("operating_profit_krwbn"),
    }


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_or_value(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value if value is not None else default


def _with_credibility(row: dict[str, Any]) -> dict[str, Any]:
    score = compute_credibility_score(row.get("source_type"))
    row["credibility_score"] = score
    row["credibility_grade"] = credibility_grade(score)
    return row


def _iso_or_none(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


def list_card_news_cluster_candidates(
    limit: int = 10,
    today_only: bool = False,
) -> list[dict[str, Any]]:
    """프론트 카드뉴스 생성을 위한 뉴스 대표 클러스터 후보를 조회한다."""
    limit = max(1, min(limit, 30))
    where_today = "AND r.published_at >= NOW() - INTERVAL '24 hours'" if today_only else ""
    query = text(f"""
        SELECT
            r.cluster_id,
            r.id AS representative_id,
            r.company,
            r.title,
            r.url,
            r.importance_level,
            r.importance_score,
            CASE LOWER(COALESCE(r.source_type, ''))
                WHEN 'dart' THEN 1.00
                WHEN 'ir' THEN 1.00
                WHEN 'official' THEN 0.90
                WHEN 'company_site' THEN 0.90
                WHEN 'securities_report' THEN 0.80
                WHEN 'trend_report' THEN 0.70
                WHEN 'news' THEN 0.70
                WHEN 'market_data' THEN 0.70
                WHEN 'job' THEN 0.60
                WHEN 'search_trend' THEN 0.55
                WHEN 'social' THEN 0.40
                ELSE 0.50
            END AS credibility_score,
            r.published_at,
            r.collected_at,
            COALESCE(
                array_agg(a.id ORDER BY
                    CASE LOWER(COALESCE(a.source_type, ''))
                        WHEN 'dart' THEN 1.00
                        WHEN 'ir' THEN 1.00
                        WHEN 'official' THEN 0.90
                        WHEN 'company_site' THEN 0.90
                        WHEN 'securities_report' THEN 0.80
                        WHEN 'trend_report' THEN 0.70
                        WHEN 'news' THEN 0.70
                        WHEN 'market_data' THEN 0.70
                        WHEN 'job' THEN 0.60
                        WHEN 'search_trend' THEN 0.55
                        WHEN 'social' THEN 0.40
                        ELSE 0.50
                    END DESC,
                    a.published_at DESC
                )
                    FILTER (WHERE a.id IS NOT NULL),
                ARRAY[]::bigint[]
            ) AS article_ids,
            COUNT(a.id) AS cluster_size
        FROM raw_articles r
        LEFT JOIN raw_articles a
            ON a.cluster_id = r.cluster_id
           AND a.source_type = 'news'
           AND a.collected_at::date = r.collected_at::date
           AND a.company = r.company
        WHERE r.source_type = 'news'
          AND r.is_representative = true
          AND r.cluster_id IS NOT NULL
          AND r.processing_status = 'CLASSIFIED'
          {where_today}
        GROUP BY
            r.cluster_id, r.id, r.company, r.title, r.url,
            r.importance_level, r.importance_score, r.source_type,
            r.published_at, r.collected_at
        ORDER BY
            COALESCE(r.importance_score, 0) DESC,
            r.published_at DESC NULLS LAST,
            r.collected_at DESC
        LIMIT :limit
    """)
    with SessionLocal() as db:
        rows = db.execute(query, {"limit": limit}).fetchall()
    return [dict(row._mapping) for row in rows]


def update_preprocess_status(
    article_id: int,
    processing_status: str,
    metadata_patch: Optional[dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    """전처리 라우팅 결과를 raw_articles에 반영한다."""
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET processing_status = :processing_status,
                    error_message = COALESCE(:error_message, error_message)
                WHERE id = :id
            """),
            {
                "processing_status": processing_status,
                "error_message": error_message,
                "id": article_id,
            },
        )
        if metadata_patch:
            source_row = db.execute(
                text("SELECT source_type, source_name FROM raw_articles WHERE id = :id"),
                {"id": article_id},
            ).fetchone()
            if source_row:
                _upsert_source_metadata(
                    db,
                    article_id=article_id,
                    source_type=source_row.source_type,
                    source_name=source_row.source_name,
                    source_metadata=json.dumps(
                        _sanitize_jsonish(metadata_patch),
                        ensure_ascii=False,
                    ),
                )
        db.commit()


# ──────────────────────────────────────────────────────────────
# 업데이트 — 파이프라인 각 단계에서 호출
# ──────────────────────────────────────────────────────────────


def update_cluster(
    article_id: int,
    cluster_id: int,
    is_representative: bool,
) -> None:
    """cluster_id, is_representative, processing_status를 업데이트한다."""
    status = "CLUSTERED_REP" if is_representative else "CLUSTERED_DUPE"
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET cluster_id = :cluster_id,
                    is_representative = :is_rep,
                    processing_status = :status
                WHERE id = :id
            """),
            {
                "cluster_id": cluster_id,
                "is_rep": is_representative,
                "status": status,
                "id": article_id,
            },
        )
        db.commit()


def update_classification(
    article_id: int,
    importance: str,
    importance_score: float,
    qdrant_vector_id: Optional[str] = None,
) -> None:
    """중요도 분류 결과를 raw_articles에 반영한다."""
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET importance_level = :importance,
                    importance_score = :score,
                    processing_status = 'CLASSIFIED',
                    qdrant_vector_id = CAST(:qdrant_id AS uuid)
                WHERE id = :id
            """),
            {
                "importance": importance,
                "score": importance_score,
                "qdrant_id": qdrant_vector_id,
                "id": article_id,
            },
        )
        db.commit()


# ──────────────────────────────────────────────────────────────
# 카드 뉴스 저장 (구 issue_cards → V9 에서 card_news 로 rename)
# ──────────────────────────────────────────────────────────────

_INSERT_CARD_NEWS = text("""
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

    implication JSONB에는 화면 호환용 섹터/노출도 요약을 유지한다.
    검증 체인 본문은 save_evidence_chain()에서 card_news.evidence_payload에 저장한다.

    Returns:
        저장된 card_news ID, 실패 시 None
    """
    try:
        # implication JSONB에 화면 호환 메타데이터를 유지한다.
        v3_payload = {
            "sector": card.get("sector", "other"),
            "sectors": card.get("sectors", ["other"]),
            "exposure_score": card.get("exposure_score", 0.0),
            "exposure_band": card.get("exposure_band", "low"),
            "signals": card.get("signals", {}),
            "evidence_chain": card.get("evidence_chain", {}),
        }

        with SessionLocal() as db:
            result = db.execute(
                _INSERT_CARD_NEWS,
                {
                    "id": card["id"],
                    "company": card.get("company") or card.get("peer_id"),
                    "cluster_id": card.get("cluster_id"),
                    "title": card["title"][:500],
                    "summary_lines": card.get("summary_lines", []),
                    "event_type": card.get("event_type", "tech"),
                    "importance": card.get("importance", "low"),
                    "importance_score": card.get("importance_score", 0.0),
                    "implication": json.dumps(v3_payload, ensure_ascii=False),
                    "sources": json.dumps(card.get("sources", []), ensure_ascii=False),
                    "validation_pass": card.get("validation", {}).get("pass", False),
                    "validation_sc_score": card.get("validation", {}).get("sc_score", 0.0),
                },
            )
            row = result.fetchone()
            db.commit()
            if row:
                log.info("카드 뉴스 저장 완료 | id=%s", card["id"])
                return card["id"]
    except Exception as e:
        log.error("카드 뉴스 저장 실패 | id=%s error=%s", card.get("id"), e)
    return None


# ──────────────────────────────────────────────────────────────
# card_news.evidence_payload 저장
# ──────────────────────────────────────────────────────────────


_UPDATE_CARD_EVIDENCE = text("""
    UPDATE card_news
    SET evidence_payload = CAST(:evidence_payload AS jsonb),
        source_raw_article_ids = CASE
            WHEN cardinality(COALESCE(source_raw_article_ids, '{}')) = 0
             AND :raw_article_ids <> '{}'
            THEN CAST(:raw_article_ids AS bigint[])
            ELSE source_raw_article_ids
        END,
        implication = COALESCE(implication, '{}'::jsonb)
            || jsonb_build_object('evidence_chain', CAST(:evidence_payload AS jsonb))
    WHERE id = :card_news_id
""")


def save_evidence_chain(
    card_news_id: str,
    chain: dict[str, Any],
    passed: bool,
    missing: list[str],
) -> None:
    """검증 체인 4종을 card_news.evidence_payload에 저장한다."""
    if not card_news_id:
        return
    try:
        with SessionLocal() as db:
            if not _column_exists(db, "card_news", "evidence_payload"):
                return
            raw_article_ids = chain.get("provenance", {}).get("raw_article_ids") or []
            evidence_payload = {
                "source_links": chain.get("source_links", []),
                "provenance": chain.get("provenance", {}),
                "financial_refs": chain.get("financial_refs", []),
                "mbb_refs": chain.get("mbb_refs", []),
                "financial_link": chain.get("financial_link", {}),
                "evidence_version": chain.get("provenance", {}).get("evidence_version", "v3.0"),
                "pass": passed,
                "missing": missing,
            }
            db.execute(
                _UPDATE_CARD_EVIDENCE,
                {
                    "card_news_id": card_news_id,
                    "evidence_payload": json.dumps(evidence_payload, ensure_ascii=False),
                    "raw_article_ids": _pg_bigint_array(raw_article_ids),
                },
            )
            db.commit()
    except Exception as e:
        log.error("card_news.evidence_payload 저장 실패 | card_id=%s error=%s", card_news_id, e)


# ──────────────────────────────────────────────────────────────
# pipeline execution archive 저장
# ──────────────────────────────────────────────────────────────


_INSERT_PIPELINE_ARCHIVE = text("""
    INSERT INTO legacy_records (
        source_table, source_pk, owner_table, owner_id, payload
    ) VALUES (
        'pipeline_logs',
        :source_pk,
        'pipeline',
        :step,
        CAST(:payload AS jsonb)
    )
""")


def save_pipeline_log(
    step: str,
    company: Optional[str],
    input_count: int,
    output_count: int,
    elapsed_ms: int,
    llm_tokens_used: int = 0,
    error_msg: Optional[str] = None,
) -> None:
    """파이프라인 단계별 실행 통계를 legacy_records에 보존한다."""
    try:
        with SessionLocal() as db:
            if not _table_exists(db, "legacy_records"):
                return
            db.execute(
                _INSERT_PIPELINE_ARCHIVE,
                {
                    "source_pk": f"{datetime.utcnow().isoformat()}-{uuid4()}",
                    "step": step,
                    "payload": json.dumps(
                        {
                            "pipeline_step": step,
                            "company": company,
                            "input_count": input_count,
                            "output_count": output_count,
                            "elapsed_ms": elapsed_ms,
                            "llm_tokens_used": llm_tokens_used,
                            "error_msg": error_msg,
                            "created_at": datetime.utcnow().isoformat(),
                        },
                        ensure_ascii=False,
                    ),
                },
            )
            db.commit()
    except Exception as e:
        log.error("pipeline archive 저장 실패 | step=%s error=%s", step, e)


# ──────────────────────────────────────────────────────────────


_card_id_state: dict[str, int] = {}  # {date_str: last_seq}
_card_id_lock = threading.Lock()


def _generate_card_id(company: str) -> str:
    """IC-YYYYMMDD-NNN 형식의 이슈카드 ID를 생성한다.

    DB 저장 전에 병렬 호출되므로 Lock으로 중복 방지.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    with _card_id_lock:
        if date_str not in _card_id_state:
            with SessionLocal() as db:
                row = db.execute(
                    text("SELECT COUNT(*) FROM card_news WHERE id LIKE :prefix"),
                    {"prefix": f"IC-{date_str}-%"},
                ).fetchone()
                _card_id_state[date_str] = row[0] if row else 0
        _card_id_state[date_str] += 1
        return f"IC-{date_str}-{_card_id_state[date_str]:03d}"


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


def _is_industry_trend_article(article: RawArticle) -> bool:
    if article.source_type in _INDUSTRY_SOURCE_TYPES:
        return True

    extra = article.extra or {}
    report_type = str(extra.get("type") or extra.get("report_type") or "").strip()
    if report_type in _INDUSTRY_REPORT_TYPES:
        return True

    if any(extra.get(key) for key in _INDUSTRY_MARKER_KEYS):
        return True

    return "industry" in (article.source_name or "").lower()


def _is_valid(article: RawArticle, storage_company: Optional[list[str]] = None) -> bool:
    """Gate 1: 최소 품질 필터."""
    if not article.url or not article.title:
        return False
    if not (storage_company if storage_company is not None else article.company):
        return False
    if len(article.content or "") < 10 and len(article.title) < 5:
        return False
    return True


def _source_metadata_json(
    article: RawArticle,
    storage_company: Optional[list[str]] = None,
) -> str:
    meta = _sanitize_jsonish(dict(article.extra))
    if not article.company and storage_company and INDUSTRY_TREND_COMPANY in storage_company:
        meta["topic_scope"] = "industry_trend"
        meta["company_scope"] = "industry"
        meta["company_fallback"] = INDUSTRY_TREND_COMPANY
    elif _is_industry_trend_article(article):
        meta.setdefault("topic_scope", "industry_trend")
    for key in _SOURCE_METADATA_EXCLUDED_KEYS:
        meta.pop(key, None)
    return json.dumps(meta, ensure_ascii=False)


def _metadata_json(
    article: RawArticle,
    storage_company: Optional[list[str]] = None,
    run_context: CrawlRunContext | None = None,
) -> str:
    """Compatibility shim for older tests/callers.

    Crawl run context is now written to raw_articles.crawl_events, not metadata.
    """
    return _source_metadata_json(article, storage_company)


def _upsert_source_metadata(
    db,
    *,
    article_id: int,
    source_type: str,
    source_name: str | None,
    source_metadata: str,
) -> None:
    db.execute(
        text("""
            INSERT INTO raw_article_source_metadata (
                raw_article_id,
                source_type,
                source_name,
                source_metadata
            )
            VALUES (
                :raw_article_id,
                :source_type,
                :source_name,
                CAST(:source_metadata AS jsonb)
            )
            ON CONFLICT (raw_article_id) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                source_name = COALESCE(EXCLUDED.source_name, raw_article_source_metadata.source_name),
                source_metadata =
                    raw_article_source_metadata.source_metadata || EXCLUDED.source_metadata,
                updated_at = NOW()
        """),
        {
            "raw_article_id": article_id,
            "source_type": source_type,
            "source_name": source_name,
            "source_metadata": source_metadata,
        },
    )


def _upsert_crawl_run_article(
    db,
    *,
    article_id: int,
    article: RawArticle,
    run_context: CrawlRunContext | None,
    action: str,
    source_metadata: str,
) -> None:
    if not run_context or not run_context.crawl_run_id:
        return
    if not _column_exists(db, "raw_articles", "crawl_events"):
        return

    crawl_event = {
        "crawl_run_id": run_context.crawl_run_id,
        "raw_article_id": article_id,
        "url": article.url,
        "url_hash": article.url_hash,
        "discovered_at": _iso_or_none(article.collected_at),
        "action": action,
        "fetch_status": article.crawl_status,
        "error_message": article.error_message,
        "source_name": run_context.source_name or article.source_name,
        "collection_mode": run_context.collection_mode,
        "track": run_context.track,
        "window_start": _iso_or_none(run_context.window_start),
        "window_end": _iso_or_none(run_context.window_end),
        "source_metadata": _json_or_value(source_metadata, {}),
    }
    db.execute(
        _APPEND_CRAWL_EVENT,
        {
            "raw_article_id": article_id,
            "crawl_event": json.dumps(crawl_event, ensure_ascii=False),
        },
    )


def _table_exists(db, table_name: str) -> bool:
    return bool(
        db.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name = :table_name
                )
            """),
            {"table_name": table_name},
        ).scalar()
    )


def _column_exists(db, table_name: str, column_name: str) -> bool:
    return bool(
        db.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = :table_name
                      AND column_name = :column_name
                )
            """),
            {"table_name": table_name, "column_name": column_name},
        ).scalar()
    )


def _pg_bigint_array(values: Any) -> str:
    out: list[str] = []
    source_values = values if isinstance(values, list) else []
    for value in source_values:
        try:
            out.append(str(int(value)))
        except (TypeError, ValueError):
            continue
    return "{" + ",".join(out) + "}"


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _sanitize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value
