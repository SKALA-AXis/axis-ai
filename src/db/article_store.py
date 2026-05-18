"""raw_articles / card_news 테이블 저장·조회·업데이트 레이어."""

import json
import logging
import re
import threading
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text

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
        crawl_status, error_message, processing_status
    ) VALUES (
        :source_type, :source_name, :publisher, :title, :content, :url, :url_hash,
        :published_at, :collected_at, CAST(:company AS jsonb), :language, :content_type,
        :crawl_status, :error_message, 'RAW'
    )
    ON CONFLICT (url) DO NOTHING
    RETURNING id
""")

_SELECT_ARTICLE_ID_BY_URL = text("SELECT id FROM raw_articles WHERE url = :url")

_INSERT_CRAWL_RUN_ARTICLE = text("""
    INSERT INTO crawl_run_articles (
        crawl_run_id, raw_article_id, url, url_hash, discovered_at,
        action, fetch_status, error_message, raw_payload
    ) VALUES (
        CAST(:crawl_run_id AS uuid), :raw_article_id, :url, :url_hash, :discovered_at,
        :action, :fetch_status, :error_message, CAST(:raw_payload AS jsonb)
    )
    ON CONFLICT (crawl_run_id, url_hash) DO UPDATE SET
        raw_article_id = COALESCE(EXCLUDED.raw_article_id, crawl_run_articles.raw_article_id),
        action = EXCLUDED.action,
        fetch_status = EXCLUDED.fetch_status,
        error_message = EXCLUDED.error_message,
        raw_payload = crawl_run_articles.raw_payload || EXCLUDED.raw_payload
""")

_SOURCE_METADATA_TABLES = {
    "news": "raw_article_metadata_news",
    "official": "raw_article_metadata_official",
    "company_site": "raw_article_metadata_company_site",
    "dart": "raw_article_metadata_dart",
    "ir": "raw_article_metadata_ir",
    "securities_report": "raw_article_metadata_securities_report",
    "trend_report": "raw_article_metadata_trend_report",
    "search_trend": "raw_article_metadata_search_trend",
    "job": "raw_article_metadata_job",
    "market_data": "raw_article_metadata_market_data",
    "social": "raw_article_metadata_social",
}

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
                SELECT raw_articles.id, company, title, content, url,
                       source_type, content_type, publisher, language,
                       credibility_score, credibility_grade,
                       relevance_score, relevance_label, relevance_reason,
                       matched_companies, matched_sectors,
                       source_name, published_at, collected_at,
                       COALESCE(mu.metadata, '{}'::jsonb) AS metadata
                FROM raw_articles
                LEFT JOIN raw_article_metadata_unified mu
                    ON mu.raw_article_id = raw_articles.id
                WHERE raw_articles.id = ANY(:ids)
                ORDER BY credibility_score DESC NULLS LAST
            """),
            {"ids": ids},
        ).fetchall()
    return [dict(row._mapping) for row in rows]


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
        return db.execute(query, params).fetchall()


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
            r.credibility_score,
            r.published_at,
            r.collected_at,
            COALESCE(
                array_agg(a.id ORDER BY a.credibility_score DESC NULLS LAST, a.published_at DESC)
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
            r.importance_level, r.importance_score, r.credibility_score,
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
            source_type = db.execute(
                text("SELECT source_type FROM raw_articles WHERE id = :id"),
                {"id": article_id},
            ).scalar_one_or_none()
            if source_type:
                _upsert_source_metadata(
                    db,
                    article_id=article_id,
                    source_type=source_type,
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

    v3: implication JSONB 컬럼은 evidence_chain + sector 메타데이터의 저장소로 재사용.
    Backend에서 evidence_chain·sector 전용 컬럼 분리 후 마이그레이션 예정.

    Returns:
        저장된 card_news ID, 실패 시 None
    """
    try:
        # implication JSONB에 v3 메타데이터(섹터·노출도·검증체인) 통합 저장
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
# evidence_chain 저장
# ──────────────────────────────────────────────────────────────


# V9 (2026-05-12): card_news 테이블 rename 후에도 evidence_chain.issue_card_id 컬럼은
# legacy 이름 유지 (deploy race 회피 — 컬럼 rename 은 V10 분리). Python 변수명은
# card_news_id 로 의미 정렬, SQL column ref 만 issue_card_id 유지.
_INSERT_EVIDENCE = text("""
    INSERT INTO evidence_chain (
        issue_card_id, source_links, provenance, financial_refs,
        mbb_refs, financial_link, evidence_version, pass, missing
    ) VALUES (
        :card_news_id, CAST(:source_links AS jsonb), CAST(:provenance AS jsonb),
        CAST(:financial_refs AS jsonb), CAST(:mbb_refs AS jsonb),
        CAST(:financial_link AS jsonb), :evidence_version, :passed, :missing
    )
    ON CONFLICT (issue_card_id) DO UPDATE SET
        source_links = EXCLUDED.source_links,
        provenance = EXCLUDED.provenance,
        financial_refs = EXCLUDED.financial_refs,
        mbb_refs = EXCLUDED.mbb_refs,
        financial_link = EXCLUDED.financial_link,
        evidence_version = EXCLUDED.evidence_version,
        pass = EXCLUDED.pass,
        missing = EXCLUDED.missing
""")


def save_evidence_chain(
    card_news_id: str,
    chain: dict[str, Any],
    passed: bool,
    missing: list[str],
) -> None:
    """evidence_chain 테이블에 검증 체인 4종을 저장한다."""
    if not card_news_id:
        return
    try:
        with SessionLocal() as db:
            db.execute(
                _INSERT_EVIDENCE,
                {
                    "card_news_id": card_news_id,
                    "source_links": json.dumps(chain.get("source_links", []), ensure_ascii=False),
                    "provenance": json.dumps(chain.get("provenance", {}), ensure_ascii=False),
                    "financial_refs": json.dumps(
                        chain.get("financial_refs", []), ensure_ascii=False
                    ),
                    "mbb_refs": json.dumps(chain.get("mbb_refs", []), ensure_ascii=False),
                    "financial_link": json.dumps(
                        chain.get("financial_link", {}), ensure_ascii=False
                    ),
                    "evidence_version": chain.get("provenance", {}).get("evidence_version", "v3.0"),
                    "passed": passed,
                    "missing": missing,
                },
            )
            db.commit()
    except Exception as e:
        log.error("evidence_chain 저장 실패 | card_id=%s error=%s", card_news_id, e)


# ──────────────────────────────────────────────────────────────
# pipeline_logs 저장
# ──────────────────────────────────────────────────────────────


_INSERT_PIPELINE_LOG = text("""
    INSERT INTO pipeline_logs (
        pipeline_step, company, input_count, output_count,
        elapsed_ms, llm_tokens_used, error_msg
    ) VALUES (
        :step, :company, :input_count, :output_count,
        :elapsed_ms, :llm_tokens_used, :error_msg
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
    """파이프라인 단계별 실행 통계를 pipeline_logs에 기록."""
    try:
        with SessionLocal() as db:
            db.execute(
                _INSERT_PIPELINE_LOG,
                {
                    "step": step,
                    "company": company,
                    "input_count": input_count,
                    "output_count": output_count,
                    "elapsed_ms": elapsed_ms,
                    "llm_tokens_used": llm_tokens_used,
                    "error_msg": error_msg,
                },
            )
            db.commit()
    except Exception as e:
        log.error("pipeline_logs 저장 실패 | step=%s error=%s", step, e)


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

    Crawl run context is now written to crawl_run_articles, not metadata.
    """
    return _source_metadata_json(article, storage_company)


def _upsert_source_metadata(
    db,
    *,
    article_id: int,
    source_type: str,
    source_metadata: str,
) -> None:
    table = _SOURCE_METADATA_TABLES.get(source_type)
    if not table:
        return
    db.execute(
        text(f"""
            INSERT INTO {table} (raw_article_id, source_metadata)
            VALUES (:raw_article_id, CAST(:source_metadata AS jsonb))
            ON CONFLICT (raw_article_id) DO UPDATE SET
                source_metadata = {table}.source_metadata || EXCLUDED.source_metadata,
                updated_at = NOW()
        """),
        {
            "raw_article_id": article_id,
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

    raw_payload = {
        "source_name": run_context.source_name or article.source_name,
        "collection_mode": run_context.collection_mode,
        "track": run_context.track,
        "window_start": _iso_or_none(run_context.window_start),
        "window_end": _iso_or_none(run_context.window_end),
        "source_metadata": _json_or_value(source_metadata, {}),
    }
    db.execute(
        _INSERT_CRAWL_RUN_ARTICLE,
        {
            "crawl_run_id": run_context.crawl_run_id,
            "raw_article_id": article_id,
            "url": article.url,
            "url_hash": article.url_hash,
            "discovered_at": article.collected_at,
            "action": action,
            "fetch_status": article.crawl_status,
            "error_message": article.error_message,
            "raw_payload": json.dumps(raw_payload, ensure_ascii=False),
        },
    )


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
