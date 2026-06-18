# 작성일: 2026-06-04
# 작성자: 박진
# 변경이력:
#   2026-06-04 박진 — 통합 이슈 기반 mixer·briefing 플로우 추가 및 mixer briefing 소스 포맷 정리
"""Persistence helpers for IntegrationAgent canonical output.

The backend V40 schema owns ``integrated_issues``.  axis-ai can run against older
local databases, so this module treats missing V40 tables/columns as a graceful
no-op and lets callers keep the legacy JSONB evidence path.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from src.analysis.models import AnalysisInputBundle
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

_UPSERT_INTEGRATED_ISSUE_SQL = text("""
    INSERT INTO integrated_issues (
        schema_version, issue_key, cluster_id, representative_raw_article_id,
        main_company, event_type, source_family, scope_type,
        is_valid, confidence, headline, one_line_summary,
        analyzed_source_ids, source_ids, sectors, mentioned_peer_companies,
        content_summary, content_detailed_explanation, content_has_content,
        content_compression_method, content_basis_scope, content_source_count,
        content_section_count, issue_brief, analysis_ready_inputs, content_digest,
        issue_frame, sources, evidence, quality, metadata, payload, payload_hash
    ) VALUES (
        :schema_version, :issue_key, :cluster_id, :representative_raw_article_id,
        :main_company, :event_type, :source_family, :scope_type,
        :is_valid, :confidence, :headline, :one_line_summary,
        CAST(:analyzed_source_ids AS bigint[]), CAST(:source_ids AS bigint[]),
        CAST(:sectors AS text[]), CAST(:mentioned_peer_companies AS text[]),
        :content_summary, :content_detailed_explanation, :content_has_content,
        :content_compression_method, :content_basis_scope, :content_source_count,
        :content_section_count, CAST(:issue_brief AS jsonb),
        CAST(:analysis_ready_inputs AS jsonb), CAST(:content_digest AS jsonb),
        CAST(:issue_frame AS jsonb), CAST(:sources AS jsonb), CAST(:evidence AS jsonb),
        CAST(:quality AS jsonb), CAST(:metadata AS jsonb), CAST(:payload AS jsonb),
        :payload_hash
    )
    ON CONFLICT (schema_version, issue_key) WHERE is_current = TRUE DO UPDATE SET
        cluster_id = EXCLUDED.cluster_id,
        representative_raw_article_id = EXCLUDED.representative_raw_article_id,
        main_company = EXCLUDED.main_company,
        event_type = EXCLUDED.event_type,
        source_family = EXCLUDED.source_family,
        scope_type = EXCLUDED.scope_type,
        is_valid = EXCLUDED.is_valid,
        confidence = EXCLUDED.confidence,
        headline = EXCLUDED.headline,
        one_line_summary = EXCLUDED.one_line_summary,
        analyzed_source_ids = EXCLUDED.analyzed_source_ids,
        source_ids = EXCLUDED.source_ids,
        sectors = EXCLUDED.sectors,
        mentioned_peer_companies = EXCLUDED.mentioned_peer_companies,
        content_summary = EXCLUDED.content_summary,
        content_detailed_explanation = EXCLUDED.content_detailed_explanation,
        content_has_content = EXCLUDED.content_has_content,
        content_compression_method = EXCLUDED.content_compression_method,
        content_basis_scope = EXCLUDED.content_basis_scope,
        content_source_count = EXCLUDED.content_source_count,
        content_section_count = EXCLUDED.content_section_count,
        issue_brief = EXCLUDED.issue_brief,
        analysis_ready_inputs = EXCLUDED.analysis_ready_inputs,
        content_digest = EXCLUDED.content_digest,
        issue_frame = EXCLUDED.issue_frame,
        sources = EXCLUDED.sources,
        evidence = EXCLUDED.evidence,
        quality = EXCLUDED.quality,
        metadata = EXCLUDED.metadata,
        payload = EXCLUDED.payload,
        payload_hash = EXCLUDED.payload_hash,
        status = 'active',
        is_current = TRUE,
        updated_at = NOW()
    RETURNING id
""")

_DELETE_SOURCES_SQL = text("""
    DELETE FROM integrated_issue_source_articles WHERE integrated_issue_id = :issue_id
""")

_INSERT_SOURCE_SQL = text("""
    INSERT INTO integrated_issue_source_articles (
        integrated_issue_id, raw_article_id, source_order, is_analyzed_basis,
        title, source_name, source_type, publisher, published_at, url,
        relevance_label, relevance_score, source_payload
    ) VALUES (
        :integrated_issue_id, :raw_article_id, :source_order, :is_analyzed_basis,
        :title, :source_name, :source_type, :publisher, :published_at, :url,
        :relevance_label, :relevance_score, CAST(:source_payload AS jsonb)
    )
    ON CONFLICT (integrated_issue_id, raw_article_id) DO UPDATE SET
        source_order = EXCLUDED.source_order,
        is_analyzed_basis = EXCLUDED.is_analyzed_basis,
        title = EXCLUDED.title,
        source_name = EXCLUDED.source_name,
        source_type = EXCLUDED.source_type,
        publisher = EXCLUDED.publisher,
        published_at = EXCLUDED.published_at,
        url = EXCLUDED.url,
        relevance_label = EXCLUDED.relevance_label,
        relevance_score = EXCLUDED.relevance_score,
        source_payload = EXCLUDED.source_payload
""")

_DELETE_SECTIONS_SQL = text("""
    DELETE FROM integrated_issue_content_sections WHERE integrated_issue_id = :issue_id
""")

_INSERT_SECTION_SQL = text("""
    INSERT INTO integrated_issue_content_sections (
        integrated_issue_id, section_key, section_order, title, summary, details,
        key_points, raw_article_ids, evidence_ref_ids, section_payload
    ) VALUES (
        :integrated_issue_id, :section_key, :section_order, :title, :summary, :details,
        CAST(:key_points AS text[]), CAST(:raw_article_ids AS bigint[]),
        CAST(:evidence_ref_ids AS text[]), CAST(:section_payload AS jsonb)
    )
    ON CONFLICT (integrated_issue_id, section_key) DO UPDATE SET
        section_order = EXCLUDED.section_order,
        title = EXCLUDED.title,
        summary = EXCLUDED.summary,
        details = EXCLUDED.details,
        key_points = EXCLUDED.key_points,
        raw_article_ids = EXCLUDED.raw_article_ids,
        evidence_ref_ids = EXCLUDED.evidence_ref_ids,
        section_payload = EXCLUDED.section_payload
""")

_DELETE_EVIDENCE_SQL = text("""
    DELETE FROM integrated_issue_evidence_references WHERE integrated_issue_id = :issue_id
""")

_INSERT_EVIDENCE_SQL = text("""
    INSERT INTO integrated_issue_evidence_references (
        integrated_issue_id, evidence_ref_id, evidence_text, source_ids, reference_payload
    ) VALUES (
        :integrated_issue_id, :evidence_ref_id, :evidence_text,
        CAST(:source_ids AS bigint[]), CAST(:reference_payload AS jsonb)
    )
    ON CONFLICT (integrated_issue_id, evidence_ref_id) DO UPDATE SET
        evidence_text = EXCLUDED.evidence_text,
        source_ids = EXCLUDED.source_ids,
        reference_payload = EXCLUDED.reference_payload
""")


def save_integrated_issue(
    integrated_issue: dict[str, Any],
    *,
    input_bundle: AnalysisInputBundle | None = None,
) -> str | None:
    """Upsert IntegrationAgent output and return ``integrated_issues.id``.

    Returns ``None`` when V40 storage is unavailable or the payload is empty.
    """

    if not isinstance(integrated_issue, dict) or not integrated_issue:
        return None
    params = _upsert_params(integrated_issue, input_bundle=input_bundle)
    try:
        with SessionLocal() as db:
            row = db.execute(_UPSERT_INTEGRATED_ISSUE_SQL, params).fetchone()
            if not row:
                db.rollback()
                return None
            issue_id = str(row[0])
            _sync_sources(db, issue_id, integrated_issue, input_bundle=input_bundle)
            _sync_sections(db, issue_id, integrated_issue)
            _sync_evidence(db, issue_id, integrated_issue)
            db.commit()
            return issue_id
    except Exception as exc:  # noqa: BLE001
        if _is_missing_integrated_issue_storage(exc):
            log.info("integrated_issues storage unavailable; using JSONB evidence fallback")
            return None
        log.exception(
            "integrated issue 저장 실패 | issue_key=%s error=%s",
            params["issue_key"],
            exc,
        )
        return None


def _upsert_params(
    integrated_issue: dict[str, Any],
    *,
    input_bundle: AnalysisInputBundle | None,
) -> dict[str, Any]:
    source_ids = _int_list(
        integrated_issue.get("source_article_ids")
        or integrated_issue.get("analyzed_article_ids")
        or []
    )
    all_source_ids = _int_list(integrated_issue.get("cluster_article_ids") or source_ids)
    representative_id = _optional_int(
        integrated_issue.get("representative_id")
        or (input_bundle.metadata.get("representative_id") if input_bundle else None)
    )
    content_digest = _content_digest(integrated_issue)
    issue_frame = _issue_frame(integrated_issue)
    evidence = _evidence_payload(integrated_issue)
    quality = {
        "integration_validation": integrated_issue.get("integration_validation") or {},
        "missing_or_uncertain_points": integrated_issue.get("missing_or_uncertain_points") or [],
    }
    metadata = {
        "bundle_id": integrated_issue.get("bundle_id")
        or (input_bundle.bundle_id if input_bundle else None),
        "input_bundle_ref": integrated_issue.get("input_bundle_ref") or {},
        "prompt_version": integrated_issue.get("prompt_version"),
    }
    return {
        "schema_version": str(
            integrated_issue.get("integration_schema_version") or "integrated_issue_v3"
        ),
        "issue_key": _issue_key(integrated_issue, input_bundle=input_bundle),
        "cluster_id": _optional_int(integrated_issue.get("cluster_id")),
        "representative_raw_article_id": representative_id,
        "main_company": _str_or_none(integrated_issue.get("main_company")),
        "event_type": _str_or_none(
            integrated_issue.get("cluster_event_type") or integrated_issue.get("event_type")
        ),
        "source_family": _str_or_none(
            integrated_issue.get("issue_source_type")
            or (input_bundle.source_type if input_bundle else None)
        ),
        "scope_type": _str_or_none(integrated_issue.get("summary_scope")),
        "is_valid": bool(integrated_issue.get("is_valid_summary", True)),
        "confidence": _optional_float(integrated_issue.get("confidence")),
        "headline": _str_or_none(integrated_issue.get("headline")),
        "one_line_summary": _str_or_none(integrated_issue.get("one_line_summary")),
        "analyzed_source_ids": source_ids,
        "source_ids": all_source_ids or source_ids,
        "sectors": _str_list(integrated_issue.get("matched_sectors"))
        or _str_list(integrated_issue.get("sectors"))
        or (input_bundle.sectors if input_bundle else []),
        "mentioned_peer_companies": _str_list(integrated_issue.get("mentioned_peer_companies")),
        "content_summary": _str_or_none(content_digest.get("summary")),
        "content_detailed_explanation": _str_or_none(content_digest.get("detailed_explanation")),
        "content_has_content": bool(content_digest),
        "content_compression_method": _str_or_none(content_digest.get("compression_method")),
        "content_basis_scope": _str_or_none(content_digest.get("basis_scope")),
        "content_source_count": len(all_source_ids or source_ids),
        "content_section_count": len(_sections(integrated_issue)),
        "issue_brief": _json(
            {
                "headline": integrated_issue.get("headline"),
                "one_line_summary": integrated_issue.get("one_line_summary"),
            }
        ),
        "analysis_ready_inputs": _json(integrated_issue.get("analysis_ready_inputs") or {}),
        "content_digest": _json(content_digest),
        "issue_frame": _json(issue_frame),
        "sources": _json(_source_rows(integrated_issue, input_bundle=input_bundle)),
        "evidence": _json(evidence),
        "quality": _json(quality),
        "metadata": _json(metadata),
        "payload": _json(integrated_issue),
        "payload_hash": _str_or_none(integrated_issue.get("payload_hash")),
    }


def _sync_sources(
    db: Any,
    issue_id: str,
    integrated_issue: dict[str, Any],
    *,
    input_bundle: AnalysisInputBundle | None,
) -> None:
    rows = _source_rows(integrated_issue, input_bundle=input_bundle)
    db.execute(_DELETE_SOURCES_SQL, {"issue_id": issue_id})
    for index, source in enumerate(rows, start=1):
        raw_id = _optional_int(source.get("raw_article_id") or source.get("id"))
        if raw_id is None:
            continue
        db.execute(
            _INSERT_SOURCE_SQL,
            {
                "integrated_issue_id": issue_id,
                "raw_article_id": raw_id,
                "source_order": index,
                "is_analyzed_basis": raw_id
                in set(_int_list(integrated_issue.get("source_article_ids"))),
                "title": _str_or_none(source.get("title")),
                "source_name": _str_or_none(source.get("source_name")),
                "source_type": _str_or_none(source.get("source_type")),
                "publisher": _str_or_none(source.get("publisher")),
                "published_at": source.get("published_at"),
                "url": _str_or_none(source.get("url")),
                "relevance_label": _str_or_none(source.get("relevance_label")),
                "relevance_score": _optional_float(source.get("relevance_score")),
                "source_payload": _json(source),
            },
        )


def _sync_sections(db: Any, issue_id: str, integrated_issue: dict[str, Any]) -> None:
    db.execute(_DELETE_SECTIONS_SQL, {"issue_id": issue_id})
    for index, section in enumerate(_sections(integrated_issue), start=1):
        db.execute(
            _INSERT_SECTION_SQL,
            {
                "integrated_issue_id": issue_id,
                "section_key": section["section_key"],
                "section_order": index,
                "title": _str_or_none(section.get("title")),
                "summary": _str_or_none(section.get("summary")),
                "details": _str_or_none(section.get("details")),
                "key_points": _str_list(section.get("key_points")),
                "raw_article_ids": _int_list(section.get("raw_article_ids")),
                "evidence_ref_ids": _str_list(section.get("evidence_ref_ids")),
                "section_payload": _json(section),
            },
        )


def _sync_evidence(db: Any, issue_id: str, integrated_issue: dict[str, Any]) -> None:
    db.execute(_DELETE_EVIDENCE_SQL, {"issue_id": issue_id})
    for index, item in enumerate(_evidence_refs(integrated_issue), start=1):
        evidence_ref_id = _str_or_none(item.get("evidence_ref_id") or item.get("fact_id"))
        if not evidence_ref_id:
            evidence_ref_id = f"evidence_{index}"
        evidence_text = _str_or_none(item.get("evidence_text") or item.get("fact"))
        if not evidence_text:
            continue
        db.execute(
            _INSERT_EVIDENCE_SQL,
            {
                "integrated_issue_id": issue_id,
                "evidence_ref_id": evidence_ref_id,
                "evidence_text": evidence_text,
                "source_ids": _int_list(item.get("source_article_ids") or item.get("source_ids")),
                "reference_payload": _json(item),
            },
        )


def _issue_key(
    integrated_issue: dict[str, Any],
    *,
    input_bundle: AnalysisInputBundle | None,
) -> str:
    explicit = _str_or_none(integrated_issue.get("issue_key"))
    if explicit:
        return explicit
    schema_version = str(integrated_issue.get("integration_schema_version") or "v3")
    cluster_id = _optional_int(integrated_issue.get("cluster_id"))
    if cluster_id is not None:
        return f"cluster:{cluster_id}:{schema_version}"
    source_ids = _int_list(integrated_issue.get("source_article_ids"))
    if source_ids:
        return f"raw:{source_ids[0]}:{schema_version}"
    bundle_id = integrated_issue.get("bundle_id") or (
        input_bundle.bundle_id if input_bundle else ""
    )
    return f"bundle:{bundle_id or 'unknown'}:{schema_version}"


def _source_rows(
    integrated_issue: dict[str, Any],
    *,
    input_bundle: AnalysisInputBundle | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in (
        integrated_issue.get("representative_sources") or integrated_issue.get("sources") or []
    ):
        if isinstance(source, dict):
            rows.append(dict(source))
    if input_bundle is not None:
        for source in input_bundle.sources:
            if isinstance(source, dict):
                rows.append(dict(source))
        raw_by_id = {
            _optional_int(item.get("id")): item
            for item in input_bundle.items
            if isinstance(item, dict) and _optional_int(item.get("id")) is not None
        }
        for raw_id in _int_list(integrated_issue.get("source_article_ids")):
            if raw_id in raw_by_id:
                rows.append(dict(raw_by_id[raw_id]))
    deduped: list[dict[str, Any]] = []
    seen: set[int | str] = set()
    for row in rows:
        key: int | str | None = _optional_int(row.get("raw_article_id") or row.get("id"))
        if key is None:
            key = str(row.get("url") or row.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _content_digest(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    raw = integrated_issue.get("content_digest")
    if isinstance(raw, dict):
        return raw
    article = integrated_issue.get("integrated_article")
    if isinstance(article, dict):
        return {
            "summary": article.get("lead"),
            "detailed_explanation": " ".join(_str_list(article.get("body_summary_lines"))),
            "sections": _sections(integrated_issue),
        }
    return {
        "summary": integrated_issue.get("one_line_summary"),
        "detailed_explanation": integrated_issue.get("integrated_text"),
        "sections": _sections(integrated_issue),
    }


def _issue_frame(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    raw = integrated_issue.get("issue_frame")
    if isinstance(raw, dict):
        return raw
    return {
        "main_issue": integrated_issue.get("main_issue"),
        "main_company": integrated_issue.get("main_company"),
        "event_type": integrated_issue.get("cluster_event_type")
        or integrated_issue.get("event_type"),
        "business_signals": integrated_issue.get("business_signals") or [],
        "key_numbers": integrated_issue.get("key_numbers") or [],
    }


def _evidence_payload(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    raw = integrated_issue.get("evidence")
    if isinstance(raw, dict):
        return raw
    return {
        "fact_basis": integrated_issue.get("fact_basis") or [],
        "references": _evidence_refs(integrated_issue),
    }


def _sections(integrated_issue: dict[str, Any]) -> list[dict[str, Any]]:
    digest = integrated_issue.get("content_digest")
    if isinstance(digest, dict) and isinstance(digest.get("sections"), list):
        return [section for section in digest["sections"] if isinstance(section, dict)]
    article = integrated_issue.get("integrated_article")
    if isinstance(article, dict):
        body_lines = _str_list(article.get("body_summary_lines"))
        return [
            {
                "section_key": "integrated_article",
                "title": article.get("title") or integrated_issue.get("headline"),
                "summary": article.get("lead") or integrated_issue.get("one_line_summary"),
                "details": " ".join(body_lines),
                "key_points": body_lines,
                "raw_article_ids": _int_list(article.get("source_article_ids")),
                "evidence_ref_ids": _fact_ids(integrated_issue.get("fact_basis")),
            }
        ]
    facts = _str_list(integrated_issue.get("fact_summary"))
    if not facts:
        return []
    return [
        {
            "section_key": "fact_summary",
            "title": integrated_issue.get("headline"),
            "summary": integrated_issue.get("one_line_summary"),
            "details": " ".join(facts),
            "key_points": facts,
            "raw_article_ids": _int_list(integrated_issue.get("source_article_ids")),
            "evidence_ref_ids": _fact_ids(integrated_issue.get("fact_basis")),
        }
    ]


def _evidence_refs(integrated_issue: dict[str, Any]) -> list[dict[str, Any]]:
    refs = integrated_issue.get("evidence_references")
    if isinstance(refs, list):
        return [item for item in refs if isinstance(item, dict)]
    fact_basis = integrated_issue.get("fact_basis")
    if isinstance(fact_basis, list):
        return [item for item in fact_basis if isinstance(item, dict)]
    return []


def _fact_ids(value: Any) -> list[str]:
    ids: list[str] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        ids.extend(_str_list(item.get("fact_ids")))
        fact_id = _str_or_none(item.get("fact_id"))
        if fact_id:
            ids.append(fact_id)
    return _dedupe(ids)


def _is_missing_integrated_issue_storage(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "undefinedtable" in message
        or "undefinedcolumn" in message
        or "integrated_issues" in message
        and "does not exist" in message
    )


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _int_list(value: Any) -> list[int]:
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[int] = []
    for item in values:
        parsed = _optional_int(item)
        if parsed is not None and parsed > 0 and parsed not in result:
            result.append(parsed)
    return result


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
