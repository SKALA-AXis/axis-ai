"""data_layer — strategic_insight_agent 에서 분리 (이동만, 동작 불변).

분리 근거: axis-infra docs/structure-tasks/agent-split-design.md (Phase 2 1단계)
"""

import json
import logging
from datetime import UTC
from pathlib import Path
from typing import Any

from sqlalchemy import text  # noqa: E402

from src.agents.briefing.support import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    KST,
    _analysis_package,
    _analysis_package_from_sources,
    _compact_analysis_package,
    _compact_analysis_unit_for_display,
    _company_label,
    _first_from_list,
    _first_int,
    _first_text,
    _int_list,
    _iso_or_none,
    _json_dict,
    _json_list,
    _nested_get,
    _optional_int,
    _parse_datetime,
    _safe_float,
    _str_values,
)
from src.db.postgres import SessionLocal  # noqa: E402
from src.services.analysis_units import (  # noqa: E402
    analysis_units_from_cards,
)

log = logging.getLogger(__name__)


_DEFAULT_LIMIT = 20


_MAX_LIMIT = 50


_SECTOR_FILTER_FETCH_MULTIPLIER = 5


_DEFAULT_MOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "crawler"
    / "crawler_results"
    / "mixer_mock"
    / "mixer_analysis_packages_20260522.json"
)


def _fetch_period_analysis_units(
    *,
    period: dict[str, Any],
    card_ids: list[str] | None,
    integrated_issue_ids: list[str] | None,
    peer_ids: list[str] | None,
    sectors: list[str] | None,
    limit: int,
) -> list[Any]:
    """Load briefing inputs from ``integrated_issues`` as the primary store.

    ``card_news`` is used only to resolve a display/card anchor id and, while
    analysis/implication results are still not normalized into their own table,
    as a backward-compatible source for ``evidence_payload.analysis_package``.
    """

    rows = _fetch_period_integrated_issue_rows(
        period=period,
        card_ids=card_ids,
        integrated_issue_ids=integrated_issue_ids,
        peer_ids=peer_ids,
        sectors=sectors,
        limit=limit,
    )
    cards = [_card_from_integrated_issue_row(row) for row in rows]
    return analysis_units_from_cards(cards)


def _fetch_period_integrated_issue_rows(
    *,
    period: dict[str, Any],
    card_ids: list[str] | None,
    integrated_issue_ids: list[str] | None,
    peer_ids: list[str] | None,
    sectors: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    fetch_limit = max(1, min(int(limit or _DEFAULT_LIMIT), _MAX_LIMIT))
    params: dict[str, Any] = {
        "start_at": period["start_at"],
        "end_at": period["end_exclusive_at"],
        "limit": fetch_limit,
    }
    where = [
        "ii.status = 'active'",
        "ii.is_current = TRUE",
        "anchor.card_id IS NOT NULL",
        "COALESCE(src.latest_published_at, ii.created_at) >= :start_at",
        "COALESCE(src.latest_published_at, ii.created_at) < :end_at",
    ]
    _append_in_filter(where, params, "ii.id::text", "integrated_issue_id", integrated_issue_ids)
    _append_integrated_issue_card_filter(where, params, card_ids)
    _append_integrated_issue_peer_filter(where, params, peer_ids)
    _append_integrated_issue_sector_filter(where, params, sectors)

    sql = f"""
        SELECT
            ii.id::text AS integrated_issue_id,
            ii.issue_key,
            ii.cluster_id,
            ii.representative_raw_article_id,
            ii.main_company,
            ii.event_type,
            ii.source_family,
            ii.scope_type,
            ii.is_valid,
            ii.confidence,
            ii.status,
            ii.is_current,
            ii.headline,
            ii.one_line_summary,
            ii.analyzed_source_ids,
            ii.source_ids,
            ii.sectors,
            ii.mentioned_peer_companies,
            ii.content_summary,
            ii.content_detailed_explanation,
            ii.issue_brief,
            ii.analysis_ready_inputs,
            ii.content_digest,
            ii.issue_frame,
            ii.sources AS issue_sources,
            ii.evidence AS issue_evidence,
            ii.quality AS issue_quality,
            ii.metadata AS issue_metadata,
            ii.payload AS issue_payload,
            ii.created_at AS issue_created_at,
            ii.updated_at AS issue_updated_at,
            COALESCE(src.sources_json, '[]'::jsonb) AS source_links,
            COALESCE(src.source_names, ARRAY[]::text[]) AS source_names,
            src.first_published_at,
            src.latest_published_at,
            COALESCE(ev.evidence_refs_json, '[]'::jsonb) AS evidence_refs,
            COALESCE(sec.sections_json, '[]'::jsonb) AS content_sections,
            anchor.card_id AS anchor_card_id,
            anchor.peer_id AS anchor_peer_id,
            anchor.importance AS anchor_importance,
            anchor.importance_score AS anchor_importance_score,
            anchor.evidence_payload AS anchor_evidence_payload,
            anchor.created_at AS anchor_created_at,
            COALESCE(src.latest_published_at, ii.created_at) AS basis_at
        FROM integrated_issues ii
        LEFT JOIN LATERAL (
            SELECT
                jsonb_agg(
                    jsonb_build_object(
                        'raw_article_id', s.raw_article_id,
                        'article_id', s.raw_article_id,
                        'title', s.title,
                        'source_name', s.source_name,
                        'publisher', s.publisher,
                        'source_type', s.source_type,
                        'published_at', s.published_at,
                        'url', s.url,
                        'relevance_label', s.relevance_label,
                        'relevance_score', s.relevance_score
                    )
                    ORDER BY s.source_order, s.id
                ) AS sources_json,
                array_agg(DISTINCT COALESCE(NULLIF(s.source_name, ''), NULLIF(s.publisher, '')))
                    FILTER (
                        WHERE COALESCE(
                            NULLIF(s.source_name, ''),
                            NULLIF(s.publisher, '')
                        ) IS NOT NULL
                    )
                    AS source_names,
                MIN(s.published_at) AS first_published_at,
                MAX(s.published_at) AS latest_published_at
            FROM integrated_issue_source_articles s
            WHERE s.integrated_issue_id = ii.id
        ) src ON TRUE
        LEFT JOIN LATERAL (
            SELECT
                jsonb_agg(
                    jsonb_build_object(
                        'evidence_ref_id', e.evidence_ref_id,
                        'text', e.evidence_text,
                        'evidence_text', e.evidence_text,
                        'source_ids', e.source_ids,
                        'reference_payload', e.reference_payload
                    )
                    ORDER BY e.id
                ) AS evidence_refs_json
            FROM integrated_issue_evidence_references e
            WHERE e.integrated_issue_id = ii.id
        ) ev ON TRUE
        LEFT JOIN LATERAL (
            SELECT
                jsonb_agg(
                    jsonb_build_object(
                        'section_key', s.section_key,
                        'title', s.title,
                        'summary', s.summary,
                        'details', s.details,
                        'key_points', s.key_points,
                        'raw_article_ids', s.raw_article_ids,
                        'evidence_ref_ids', s.evidence_ref_ids,
                        'section_payload', s.section_payload
                    )
                    ORDER BY s.section_order, s.id
                ) AS sections_json
            FROM integrated_issue_content_sections s
            WHERE s.integrated_issue_id = ii.id
        ) sec ON TRUE
        LEFT JOIN LATERAL (
            SELECT
                cn.id AS card_id,
                COALESCE(cn.peer_company_id, cn.company) AS peer_id,
                cn.importance,
                cn.importance_score,
                cn.evidence_payload,
                cn.created_at
            FROM card_news cn
            WHERE cn.integrated_issue_id = ii.id
              AND COALESCE(cn.status, 'ACTIVE') = 'ACTIVE'
            ORDER BY
                cn.importance_score DESC NULLS LAST,
                cn.created_at DESC
            LIMIT 1
        ) anchor ON TRUE
        WHERE {" AND ".join(where)}
        ORDER BY
            COALESCE(anchor.importance_score, ii.confidence, 0.0) DESC,
            COALESCE(src.latest_published_at, ii.created_at) DESC,
            ii.updated_at DESC
        LIMIT :limit
    """

    try:
        with SessionLocal() as db:
            rows = db.execute(text(sql), params).mappings().all()
    except Exception as exc:  # noqa: BLE001
        if _is_missing_integrated_issue_storage(exc):
            log.info("BriefingGenerationAgent integrated_issues storage unavailable")
            return []
        log.exception("BriefingGenerationAgent integrated_issues 기간 조회 실패 | error=%s", exc)
        return []
    return [dict(row) for row in rows]


def _append_integrated_issue_card_filter(
    where: list[str],
    params: dict[str, Any],
    card_ids: list[str] | None,
) -> None:
    cleaned = [str(value).strip() for value in card_ids or [] if str(value).strip()]
    if not cleaned:
        return
    placeholders = []
    for index, value in enumerate(cleaned):
        key = f"requested_card_id_{index}"
        placeholders.append(f":{key}")
        params[key] = value
    where.append(
        "EXISTS ("
        "SELECT 1 FROM card_news req "
        "WHERE req.integrated_issue_id = ii.id "
        f"AND req.id IN ({', '.join(placeholders)})"
        ")"
    )


def _append_integrated_issue_peer_filter(
    where: list[str],
    params: dict[str, Any],
    peer_ids: list[str] | None,
) -> None:
    cleaned = [str(value).strip() for value in peer_ids or [] if str(value).strip()]
    if not cleaned:
        return
    placeholders = []
    for index, value in enumerate(cleaned):
        key = f"issue_peer_id_{index}"
        placeholders.append(f":{key}")
        params[key] = value
    array_sql = f"ARRAY[{', '.join(placeholders)}]::text[]"
    where.append(
        f"(ii.main_company IN ({', '.join(placeholders)}) "
        f"OR ii.mentioned_peer_companies && {array_sql})"
    )


def _append_integrated_issue_sector_filter(
    where: list[str],
    params: dict[str, Any],
    sectors: list[str] | None,
) -> None:
    cleaned = [str(value).strip() for value in sectors or [] if str(value).strip()]
    if not cleaned:
        return
    placeholders = []
    for index, value in enumerate(cleaned):
        key = f"issue_sector_{index}"
        placeholders.append(f":{key}")
        params[key] = value
    where.append(f"ii.sectors && ARRAY[{', '.join(placeholders)}]::text[]")


def _card_from_integrated_issue_row(row: dict[str, Any]) -> dict[str, Any]:
    issue_id = _first_text(row.get("integrated_issue_id"), row.get("id"))
    integrated_issue = _integrated_issue_from_period_row(row)
    source_links = _json_list(row.get("source_links")) or _json_list(
        integrated_issue.get("representative_sources")
    )
    evidence_refs = _json_list(row.get("evidence_refs")) or _json_list(
        integrated_issue.get("fact_basis")
    )
    package = _analysis_package_from_integrated_issue_row(
        row,
        integrated_issue=integrated_issue,
        source_links=source_links,
        evidence_refs=evidence_refs,
    )
    anchor_card_id = _first_text(row.get("anchor_card_id"))
    anchor_id = anchor_card_id or issue_id
    sector = _first_text(
        _first_from_list(row.get("sectors")),
        _nested_get(package, "classification", "sector"),
        "other",
    )
    company = _first_text(
        integrated_issue.get("main_company"),
        row.get("main_company"),
        row.get("anchor_peer_id"),
    )
    source_raw_article_ids = _int_list(
        integrated_issue.get("source_article_ids")
        or integrated_issue.get("cluster_article_ids")
        or row.get("source_ids")
    )
    evidence_payload = {
        "integrated_issue_id": issue_id,
        "analysis_package": package,
        "integrated_issue": integrated_issue,
        "analysis": _json_dict(package.get("analysis")),
        "implication": _json_dict(package.get("implication")),
        "classification": _json_dict(package.get("classification")),
        "validation": _json_dict(package.get("validation")),
        "source_links": source_links,
        "evidence_refs": evidence_refs,
        "content_sections": _json_list(row.get("content_sections")),
        "analysis_basis_source": "integrated_issues",
    }
    return {
        "id": anchor_id,
        "card_id": anchor_card_id or None,
        "integrated_issue_id": issue_id or None,
        "title": _first_text(
            integrated_issue.get("main_issue"),
            integrated_issue.get("headline"),
            integrated_issue.get("one_line_summary"),
            row.get("headline"),
            issue_id,
        ),
        # Do not use card_news.summary_lines in the integrated issue path.
        "summary_lines": [],
        "display_summary": _json_list(integrated_issue.get("fact_summary")),
        "event_type": _first_text(
            integrated_issue.get("cluster_event_type"),
            row.get("event_type"),
        ),
        "importance": _first_text(row.get("anchor_importance"), "medium"),
        "importance_score": _safe_float(row.get("anchor_importance_score"), default=-1.0),
        "company": company,
        "peer_id": _first_text(row.get("anchor_peer_id"), company),
        "sector": sector,
        "sectors": _str_values(row.get("sectors")) or [sector],
        "sources": source_links,
        "source_raw_article_ids": source_raw_article_ids,
        "primary_raw_article_id": _first_int(source_raw_article_ids),
        "evidence_payload": evidence_payload,
        "analysis_package": package,
        "has_analysis_package": bool(package.get("analysis") or package.get("implication")),
        "evidence_card_ids": [anchor_id] if anchor_id else [],
        "validation_pass": _nested_get(package, "validation", "pass"),
        "validation_sc_score": _nested_get(package, "validation", "sc_score"),
        "created_at": _iso_or_none(row.get("issue_created_at")),
        "basis_at": _iso_or_none(row.get("basis_at") or row.get("latest_published_at")),
        "analysis_basis_source": "integrated_issues",
    }


def _integrated_issue_from_period_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = _json_dict(row.get("issue_payload"))
    issue = _json_dict(payload.get("integrated_issue")) or (
        dict(payload) if payload.get("is_valid_summary") else {}
    )
    content_digest = _json_dict(row.get("content_digest"))
    issue_frame = _json_dict(row.get("issue_frame"))
    source_links = _json_list(row.get("source_links")) or _json_list(row.get("issue_sources"))
    evidence_refs = _json_list(row.get("evidence_refs"))
    sectors = _str_values(row.get("sectors"))
    source_ids = _int_list(row.get("source_ids"))
    analyzed_ids = _int_list(row.get("analyzed_source_ids"))
    issue.update(
        {
            "integrated_issue_id": _first_text(row.get("integrated_issue_id")),
            "bundle_id": issue.get("bundle_id") or row.get("issue_key"),
            "cluster_id": issue.get("cluster_id") or row.get("cluster_id"),
            "representative_id": issue.get("representative_id")
            or row.get("representative_raw_article_id"),
            "source_article_ids": issue.get("source_article_ids") or source_ids,
            "cluster_article_ids": issue.get("cluster_article_ids") or source_ids,
            "analyzed_article_ids": issue.get("analyzed_article_ids") or analyzed_ids,
            "main_company": issue.get("main_company") or row.get("main_company"),
            "mentioned_peer_companies": issue.get("mentioned_peer_companies")
            or _str_values(row.get("mentioned_peer_companies")),
            "cluster_event_type": issue.get("cluster_event_type") or row.get("event_type"),
            "sectors": issue.get("sectors") or sectors,
            "headline": issue.get("headline") or row.get("headline"),
            "main_issue": issue.get("main_issue") or row.get("headline"),
            "one_line_summary": issue.get("one_line_summary") or row.get("one_line_summary"),
            "integrated_text": issue.get("integrated_text")
            or row.get("content_detailed_explanation")
            or row.get("content_summary"),
            "fact_summary": issue.get("fact_summary")
            or _fact_summary_from_issue_brief(row.get("issue_brief"), content_digest),
            "content_digest": issue.get("content_digest") or content_digest,
            "issue_frame": issue.get("issue_frame") or issue_frame,
            "representative_sources": issue.get("representative_sources") or source_links,
            "fact_basis": issue.get("fact_basis")
            or _fact_basis_from_period_evidence(evidence_refs),
            "consolidated_facts": issue.get("consolidated_facts")
            or _consolidated_facts_from_period_evidence(evidence_refs),
            "confidence": issue.get("confidence") or row.get("confidence") or 0.0,
            "is_valid_summary": issue.get("is_valid_summary", row.get("is_valid", True)),
        }
    )
    return {key: value for key, value in issue.items() if value not in (None, "", [], {})}


def _analysis_package_from_integrated_issue_row(
    row: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    source_links: list[Any],
    evidence_refs: list[Any],
) -> dict[str, Any]:
    anchor_evidence = _json_dict(row.get("anchor_evidence_payload"))
    legacy_package = _json_dict(anchor_evidence.get("analysis_package"))
    classification = _json_dict(
        legacy_package.get("classification")
    ) or _classification_from_integrated_issue_row(row, integrated_issue=integrated_issue)
    analysis = _json_dict(legacy_package.get("analysis")) or _analysis_from_integrated_issue(
        row,
        integrated_issue=integrated_issue,
        evidence_refs=evidence_refs,
    )
    implication = _json_dict(
        legacy_package.get("implication")
    ) or _implication_from_integrated_issue(
        integrated_issue=integrated_issue,
        analysis=analysis,
    )
    validation = _json_dict(legacy_package.get("validation")) or {
        "pass": bool(integrated_issue.get("is_valid_summary", row.get("is_valid", True))),
        "sc_score": _safe_float(row.get("confidence"), default=0.0),
    }
    package = dict(legacy_package)
    package.update(
        {
            "integrated_issue_id": _first_text(row.get("integrated_issue_id")),
            "bundle_id": integrated_issue.get("bundle_id") or row.get("issue_key"),
            "integrated_issue": integrated_issue,
            "summary": integrated_issue,
            "analysis": analysis,
            "implication": implication,
            "classification": classification,
            "validation": validation,
            "sources": source_links,
            "evidence_refs": evidence_refs,
            "analysis_basis_source": "integrated_issues",
            "legacy_analysis_package_source": (
                "card_news.evidence_payload.analysis_package" if legacy_package else None
            ),
        }
    )
    return package


def _classification_from_integrated_issue_row(
    row: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    sectors = _str_values(integrated_issue.get("sectors") or row.get("sectors"))
    sector = sectors[0] if sectors else ""
    company = _first_text(integrated_issue.get("main_company"), row.get("main_company"))
    return {
        "sector": sector,
        "sectors": sectors,
        "company": company,
        "companies": [company] if company else [],
        "peer_id": company,
        "event_type": _first_text(
            integrated_issue.get("cluster_event_type"),
            row.get("event_type"),
        ),
        "importance": _first_text(row.get("anchor_importance"), "medium"),
        "importance_score": _safe_float(row.get("anchor_importance_score"), default=0.0),
        "representative_id": integrated_issue.get("representative_id")
        or row.get("representative_raw_article_id"),
    }


def _analysis_from_integrated_issue(
    row: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    evidence_refs: list[Any],
) -> dict[str, Any]:
    facts = _fact_basis_from_period_evidence(evidence_refs)
    summary = _first_text(
        integrated_issue.get("integrated_text"),
        integrated_issue.get("one_line_summary"),
        integrated_issue.get("main_issue"),
        row.get("content_summary"),
    )
    market_signal = _first_text(
        integrated_issue.get("one_line_summary"),
        integrated_issue.get("main_issue"),
        summary,
    )
    strategic_meaning = _dedupe_keep_order(
        [
            _first_text(integrated_issue.get("content_digest", {}).get("detailed_explanation")),
            *[
                str(item.get("text") or item.get("evidence_text") or "").strip()
                for item in facts[:2]
                if isinstance(item, dict)
            ],
        ]
    )
    return {
        "is_valid_analysis": bool(integrated_issue.get("is_valid_summary", True)),
        "analysis_summary": summary,
        "market_signal": market_signal,
        "strategic_meaning": strategic_meaning[:3],
        "impact_reason": summary,
        "risk_or_opportunity": "입력 근거 기반 추가 판단 필요",
        "confidence": _safe_float(row.get("confidence"), default=0.0),
        "analysis_basis_source": "integrated_issues",
    }


def _implication_from_integrated_issue(
    *,
    integrated_issue: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    market_signal = _first_text(analysis.get("market_signal"), integrated_issue.get("main_issue"))
    why = _first_text(
        analysis.get("impact_reason"),
        integrated_issue.get("one_line_summary"),
        "통합 이슈에서 확인된 사실을 SK AX 사업 판단 기준으로 전환해야 합니다.",
    )
    action = _clip_text(
        (
            "SK AX는 입력에서 확인된 고객군과 업무 범위를 기준으로 "
            "오퍼링 우선순위, 책임 조직, 리스크 승인 기준을 명확히 정한다."
        ),
        max_chars=260,
    )
    return {
        "is_valid_implication": bool(integrated_issue.get("is_valid_summary", True)),
        "implication_scope": "peer_and_skax",
        "peer_implication": {
            "peer_meaning": market_signal,
            "capability_change": market_signal,
        },
        "skax_implication": {
            "why_important": why,
            "potential_impact": market_signal,
            "recommended_actions": [action],
        },
        "recommended_actions": [action],
        "confidence": _safe_float(integrated_issue.get("confidence"), default=0.0),
        "analysis_basis_source": "integrated_issues",
    }


def _fact_summary_from_issue_brief(value: Any, content_digest: dict[str, Any]) -> list[str]:
    issue_brief = _json_dict(value)
    return _dedupe_keep_order(
        [
            _first_text(issue_brief.get("headline"), issue_brief.get("one_line_summary")),
            _first_text(content_digest.get("summary"), content_digest.get("detailed_explanation")),
        ]
    )


def _fact_basis_from_period_evidence(value: Any) -> list[dict[str, Any]]:
    refs = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        text_value = _first_text(item.get("evidence_text"), item.get("text"), item.get("fact"))
        if not text_value:
            continue
        refs.append(
            {
                "evidence_ref_id": _first_text(item.get("evidence_ref_id")),
                "text": text_value,
                "source_ids": _int_list(item.get("source_ids")),
            }
        )
    return refs


def _consolidated_facts_from_period_evidence(value: Any) -> list[str]:
    return [
        str(item.get("text") or item.get("evidence_text") or "").strip()
        for item in _json_list(value)
        if isinstance(item, dict)
        and str(item.get("text") or item.get("evidence_text") or "").strip()
    ][:8]


def _is_missing_integrated_issue_storage(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "integrated_issues" in message
        or "integrated_issue_source_articles" in message
        or "integrated_issue_evidence_references" in message
        or "integrated_issue_content_sections" in message
    ) and ("undefined" in message or "does not exist" in message)


def _fetch_period_cards(
    *,
    period: dict[str, Any],
    card_ids: list[str] | None,
    integrated_issue_ids: list[str] | None,
    peer_ids: list[str] | None,
    sectors: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    fetch_limit = max(1, min(int(limit or _DEFAULT_LIMIT), _DEFAULT_LIMIT))
    if sectors:
        fetch_limit *= _SECTOR_FILTER_FETCH_MULTIPLIER
    params: dict[str, Any] = {
        "start_at": period["start_at"],
        "end_at": period["end_exclusive_at"],
        "limit": fetch_limit,
    }
    where = [
        "cn.status = 'ACTIVE'",
        "COALESCE(ra.published_at, cn.created_at) >= :start_at",
        "COALESCE(ra.published_at, cn.created_at) < :end_at",
    ]
    _append_in_filter(where, params, "cn.id", "card_id", card_ids)
    _append_in_filter(
        where,
        params,
        "cn.integrated_issue_id::text",
        "integrated_issue_id",
        integrated_issue_ids,
    )
    _append_in_filter(
        where,
        params,
        "COALESCE(cn.peer_company_id, cn.company)",
        "peer_id",
        peer_ids,
    )

    sql = f"""
        SELECT
            cn.id,
            cn.title,
            cn.summary_lines,
            cn.event_type,
            cn.importance,
            cn.importance_score,
            cn.company,
            COALESCE(cn.peer_company_id, cn.company) AS peer_id,
            cn.integrated_issue_id,
            cn.primary_keyword_category,
            cn.implication,
            cn.sources,
            cn.source_articles,
            cn.source_raw_article_ids,
            cn.primary_raw_article_id,
            cn.evidence_payload,
            cn.validation_pass,
            cn.validation_sc_score,
            cn.created_at,
            COALESCE(ra.published_at, cn.created_at) AS basis_at
        FROM card_news cn
        LEFT JOIN raw_articles ra ON ra.id = cn.primary_raw_article_id
        WHERE {" AND ".join(where)}
        ORDER BY
            cn.importance_score DESC NULLS LAST,
            COALESCE(ra.published_at, cn.created_at) DESC,
            cn.created_at DESC
        LIMIT :limit
    """

    try:
        with SessionLocal() as db:
            rows = db.execute(text(sql), params).mappings().all()
    except Exception as exc:  # noqa: BLE001
        log.exception("BriefingGenerationAgent card_news 기간 조회 실패 | error=%s", exc)
        return []

    cards = [_normalize_card_row(dict(row)) for row in rows]
    final_limit = max(1, min(int(limit or _DEFAULT_LIMIT), _DEFAULT_LIMIT))
    return _filter_by_sectors(cards, sectors)[:final_limit]


def _load_mock_items(
    *,
    mock_path: str | Path | None,
    mock_items: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if mock_items is not None:
        return mock_items
    if mock_path is None:
        mock_path = _DEFAULT_MOCK_PATH
    path = Path(mock_path)
    if not path.exists():
        log.warning("BriefingGenerationAgent mock path 없음 | path=%s", path)
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("selected_items") or payload.get("mixer_analysis_packages") or []
        return [item for item in items if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _fetch_mock_period_cards(
    *,
    period: dict[str, Any],
    items: list[dict[str, Any]],
    card_ids: list[str] | None,
    integrated_issue_ids: list[str] | None,
    peer_ids: list[str] | None,
    sectors: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    requested = set(card_ids or [])
    requested_issue_ids = set(integrated_issue_ids or [])
    wanted_peers = {str(peer_id).strip() for peer_id in peer_ids or [] if str(peer_id).strip()}
    cards = [_normalize_mock_item(item) for item in items]
    filtered: list[dict[str, Any]] = []
    for card in cards:
        if requested and card["id"] not in requested:
            continue
        if requested_issue_ids and card.get("integrated_issue_id") not in requested_issue_ids:
            continue
        if wanted_peers and card.get("peer_id") not in wanted_peers:
            continue
        basis_at = _parse_datetime(card.get("basis_at") or card.get("created_at"))
        if basis_at is None:
            continue
        basis_utc = basis_at.astimezone(UTC)
        if not (period["start_at"] <= basis_utc < period["end_exclusive_at"]):
            continue
        filtered.append(card)

    filtered = _filter_by_sectors(filtered, sectors)
    filtered.sort(
        key=lambda card: (
            float(card.get("importance_score") or 0.0),
            card.get("basis_at") or "",
        ),
        reverse=True,
    )
    return filtered[: max(1, min(int(limit or _DEFAULT_LIMIT), _DEFAULT_LIMIT))]


def _normalize_mock_item(item: dict[str, Any]) -> dict[str, Any]:
    card_id = str(item.get("card_id") or item.get("id") or "").strip()
    evidence_payload = _json_dict(item.get("evidence_payload"))
    top_package = _json_dict(item.get("analysis_package"))
    payload_package = _json_dict(evidence_payload.get("analysis_package"))
    package_warning = None
    if top_package and payload_package:
        top_bundle = top_package.get("bundle_id")
        payload_bundle = payload_package.get("bundle_id")
        if top_bundle and payload_bundle and top_bundle != payload_bundle:
            package_warning = "analysis_package_bundle_mismatch"
    analysis_package = payload_package or top_package
    if analysis_package:
        evidence_payload["analysis_package"] = analysis_package
    integrated_issue_id = _first_text(
        item.get("integrated_issue_id"),
        evidence_payload.get("integrated_issue_id"),
        analysis_package.get("integrated_issue_id"),
        _nested_get(analysis_package, "integrated_issue", "integrated_issue_id"),
    )

    sources = _json_list(item.get("sources"))
    basis_at = _first_source_published_at(sources) or item.get("basis_at") or item.get("created_at")
    implication = _json_dict(item.get("implication"))
    sectors = _json_list(implication.get("sectors")) or _json_list(
        _nested_get(analysis_package, "classification", "sectors")
    )
    sector = _first_text(
        item.get("primary_keyword_category"),
        implication.get("sector"),
        _nested_get(analysis_package, "classification", "sector"),
        item.get("event_type"),
        "other",
    )
    return {
        "id": card_id,
        "card_id": card_id,
        "integrated_issue_id": integrated_issue_id or None,
        "title": str(item.get("title") or ""),
        "summary_lines": _json_list(item.get("summary_lines")),
        "event_type": str(item.get("event_type") or ""),
        "importance": str(item.get("importance") or "medium"),
        "importance_score": float(item.get("importance_score") or 0.0),
        "company": str(item.get("company") or ""),
        "peer_id": str(item.get("peer_id") or item.get("company") or ""),
        "sector": sector,
        "sectors": [str(value) for value in sectors if str(value).strip()] or [sector],
        "sources": sources,
        "source_raw_article_ids": _int_list(item.get("source_raw_article_ids")),
        "primary_raw_article_id": _first_int(item.get("source_raw_article_ids")),
        "evidence_payload": evidence_payload,
        "analysis_package": analysis_package,
        "has_analysis_package": bool(analysis_package),
        "evidence_card_ids": [card_id] if card_id else [],
        "validation_pass": _nested_get(analysis_package, "validation", "pass"),
        "validation_sc_score": _nested_get(analysis_package, "validation", "sc_score"),
        "created_at": _iso_or_none(item.get("created_at")),
        "basis_at": _iso_or_none(_parse_datetime(basis_at)),
        "mock_warning": package_warning,
    }


def _append_in_filter(
    where: list[str],
    params: dict[str, Any],
    column_sql: str,
    prefix: str,
    values: list[str] | None,
) -> None:
    cleaned = [str(value).strip() for value in values or [] if str(value).strip()]
    if not cleaned:
        return
    placeholders = []
    for index, value in enumerate(cleaned):
        key = f"{prefix}_{index}"
        placeholders.append(f":{key}")
        params[key] = value
    where.append(f"{column_sql} IN ({', '.join(placeholders)})")


def _normalize_card_row(row: dict[str, Any]) -> dict[str, Any]:
    implication = _json_dict(row.get("implication"))
    evidence_payload = _json_dict(row.get("evidence_payload"))
    analysis_package = _analysis_package_from_sources(row, evidence_payload)
    integrated_issue_id = _first_text(
        row.get("integrated_issue_id"),
        evidence_payload.get("integrated_issue_id"),
        analysis_package.get("integrated_issue_id"),
        _nested_get(analysis_package, "integrated_issue", "integrated_issue_id"),
    )
    summary_lines = row.get("summary_lines")
    if isinstance(summary_lines, str):
        summary_lines = [summary_lines]
    elif not isinstance(summary_lines, list):
        summary_lines = list(summary_lines or [])
    source_raw_article_ids = _int_list(row.get("source_raw_article_ids"))
    primary_raw_article_id = _optional_int(row.get("primary_raw_article_id"))
    if primary_raw_article_id is not None and primary_raw_article_id not in source_raw_article_ids:
        source_raw_article_ids = [primary_raw_article_id, *source_raw_article_ids]
    sectors = implication.get("sectors")
    if not isinstance(sectors, list):
        sectors = _json_list(_nested_get(analysis_package, "classification", "sectors"))
    sector = (
        row.get("primary_keyword_category")
        or implication.get("sector")
        or _nested_get(analysis_package, "classification", "sector")
        or row.get("event_type")
        or "other"
    )
    card_id = str(row.get("id"))
    return {
        "id": card_id,
        "card_id": card_id,
        "integrated_issue_id": integrated_issue_id or None,
        "title": str(row.get("title") or ""),
        "summary_lines": [str(item) for item in summary_lines if str(item).strip()],
        "event_type": str(row.get("event_type") or ""),
        "importance": str(row.get("importance") or "medium"),
        "importance_score": float(row.get("importance_score") or 0.0),
        "company": str(row.get("company") or ""),
        "peer_id": str(row.get("peer_id") or row.get("company") or ""),
        "sector": str(sector),
        "sectors": [str(item) for item in sectors if str(item).strip()] or [str(sector)],
        "sources": _json_list(row.get("sources")) or _json_list(row.get("source_articles")),
        "source_raw_article_ids": source_raw_article_ids,
        "primary_raw_article_id": primary_raw_article_id,
        "evidence_payload": evidence_payload,
        "analysis_package": analysis_package,
        "has_analysis_package": bool(analysis_package),
        "evidence_card_ids": [card_id],
        "validation_pass": row.get("validation_pass"),
        "validation_sc_score": row.get("validation_sc_score"),
        "created_at": _iso_or_none(row.get("created_at")),
        "basis_at": _iso_or_none(row.get("basis_at")),
    }


def _filter_by_sectors(
    cards: list[dict[str, Any]],
    sectors: list[str] | None,
) -> list[dict[str, Any]]:
    wanted = {str(item).strip().lower() for item in sectors or [] if str(item).strip()}
    if not wanted:
        return cards
    filtered = []
    for card in cards:
        card_sectors = {str(card.get("sector", "")).lower()}
        card_sectors.update(str(item).lower() for item in card.get("sectors", []))
        if card_sectors & wanted:
            filtered.append(card)
    return filtered


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _clip_text(value: str, max_chars: int) -> str:
    text_value = str(value or "").strip()
    if len(text_value) <= max_chars:
        return text_value
    return text_value[: max_chars - 1].rstrip() + "…"


def _first_source_published_at(sources: list[Any]) -> object:
    for source in sources:
        if isinstance(source, dict) and source.get("published_at"):
            return source.get("published_at")
    return None
