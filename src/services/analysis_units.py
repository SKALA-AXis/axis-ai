"""Canonical analysis units for downstream data-usage agents.

Mixer and Briefing should reason over Integration/Analysis/Implication outputs,
not over card display summaries.  ``AnalysisUnit`` keeps the card as a UI anchor
when available, while making ``integrated_issues.id`` the canonical evidence id.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from src.db.postgres import SessionLocal
from src.shared.json_helpers import json_dict as _json_dict

log = logging.getLogger(__name__)

QUALITY_MISSING_INTEGRATED_ISSUE_ID = "missing_integrated_issue_id"
QUALITY_MISSING_ANALYSIS_PACKAGE = "missing_analysis_package"
QUALITY_SUMMARY_ONLY_FALLBACK = "summary_only_fallback"
QUALITY_SUPERSEDED_INTEGRATED_ISSUE = "superseded_integrated_issue"
QUALITY_EVIDENCE_REF_GAP = "evidence_ref_gap"


@dataclass(slots=True)
class AnalysisUnit:
    integrated_issue_id: str | None
    card_id: str | None
    integrated_issue: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    implication: dict[str, Any] = field(default_factory=dict)
    classification: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    source_raw_article_ids: list[int] = field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    source_links: list[dict[str, Any]] = field(default_factory=list)
    display_summary: list[str] = field(default_factory=list)
    card: dict[str, Any] = field(default_factory=dict)
    integrated_issue_row: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)

    @property
    def anchor_id(self) -> str:
        return str(self.card_id or self.integrated_issue_id or "")

    def to_card_like(self) -> dict[str, Any]:
        """Return the legacy card-shaped payload current agents already consume."""

        card = dict(self.card)
        anchor_id = self.anchor_id
        if anchor_id:
            card["id"] = anchor_id
        if self.card_id:
            card["card_id"] = self.card_id
        if self.integrated_issue_id:
            card["integrated_issue_id"] = self.integrated_issue_id
        card.setdefault("title", _unit_title(self))
        card.setdefault("summary_lines", list(self.display_summary))
        card["display_summary"] = list(self.display_summary)
        card["source_raw_article_ids"] = list(self.source_raw_article_ids)
        card["analysis_unit"] = self.to_public_dict()
        card["quality_flags"] = list(self.quality_flags)
        package = {
            "integrated_issue_id": self.integrated_issue_id,
            "bundle_id": self.integrated_issue.get("bundle_id"),
            "integrated_issue": self.integrated_issue,
            "summary": self.integrated_issue,
            "analysis": self.analysis,
            "implication": self.implication,
            "classification": self.classification,
            "validation": self.validation,
        }
        evidence_payload = _json_dict(card.get("evidence_payload"))
        evidence_payload.update(
            {
                "integrated_issue_id": self.integrated_issue_id,
                "analysis_package": package,
                "integrated_issue": self.integrated_issue,
                "analysis": self.analysis,
                "implication": self.implication,
                "classification": self.classification,
                "validation": self.validation,
                "source_links": list(self.source_links),
                "evidence_refs": self.evidence_refs,
                "quality_flags": list(self.quality_flags),
            }
        )
        if self.source_links and not card.get("sources"):
            card["sources"] = list(self.source_links)
        card["evidence_payload"] = evidence_payload
        card["analysis_package"] = package
        return card

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "integrated_issue_id": self.integrated_issue_id,
            "card_id": self.card_id,
            "anchor_id": self.anchor_id,
            "title": _unit_title(self),
            "source_raw_article_ids": list(self.source_raw_article_ids),
            "source_links": list(self.source_links),
            "evidence_refs": list(self.evidence_refs),
            "display_summary": list(self.display_summary),
            "quality_flags": list(self.quality_flags),
            "has_analysis_package": bool(self.analysis or self.implication or self.validation),
        }


def load_analysis_units_by_card_ids(
    card_ids: list[str],
    *,
    user_id: str | None = None,
) -> list[AnalysisUnit]:
    cleaned = _clean_ids(card_ids)
    if not cleaned:
        return []
    rows = _fetch_card_rows(cleaned, user_id=user_id)
    units = [analysis_unit_from_card(row) for row in rows]
    order = {value: index for index, value in enumerate(cleaned)}
    return sorted(
        units,
        key=lambda unit: order.get(str(unit.card_id or unit.anchor_id), len(order)),
    )


def load_analysis_units_by_integrated_issue_ids(
    integrated_issue_ids: list[str],
    *,
    user_id: str | None = None,
) -> list[AnalysisUnit]:
    cleaned = _clean_ids(integrated_issue_ids)
    if not cleaned:
        return []
    rows = _fetch_integrated_issue_rows(cleaned, user_id=user_id)
    units = [analysis_unit_from_card(row) for row in rows]
    order = {value: index for index, value in enumerate(cleaned)}
    return sorted(
        units,
        key=lambda unit: order.get(str(unit.integrated_issue_id or unit.anchor_id), len(order)),
    )


def analysis_units_from_cards(cards: list[dict[str, Any]]) -> list[AnalysisUnit]:
    return [analysis_unit_from_card(card) for card in cards if isinstance(card, dict)]


def card_like_from_units(units: list[AnalysisUnit]) -> list[dict[str, Any]]:
    return [unit.to_card_like() for unit in units]


def source_integrated_issue_ids(units: list[AnalysisUnit]) -> list[str]:
    return _dedupe([str(unit.integrated_issue_id) for unit in units if unit.integrated_issue_id])


def quality_flags_for_units(units: list[AnalysisUnit]) -> list[str]:
    return _dedupe([flag for unit in units for flag in unit.quality_flags])


def confidence_penalty_for_flags(flags: list[str]) -> float:
    penalty = 0.0
    if QUALITY_SUMMARY_ONLY_FALLBACK in flags:
        penalty += 0.25
    if QUALITY_MISSING_ANALYSIS_PACKAGE in flags:
        penalty += 0.2
    if QUALITY_MISSING_INTEGRATED_ISSUE_ID in flags:
        penalty += 0.1
    if QUALITY_EVIDENCE_REF_GAP in flags:
        penalty += 0.1
    if QUALITY_SUPERSEDED_INTEGRATED_ISSUE in flags:
        penalty += 0.15
    return min(penalty, 0.5)


def analysis_unit_from_card(card: dict[str, Any]) -> AnalysisUnit:
    evidence_payload = _json_dict(card.get("evidence_payload"))
    package = _analysis_package_from_sources(card, evidence_payload)
    issue_row = _integrated_issue_row(card)
    integrated_issue = _integrated_issue_from_sources(issue_row, package, evidence_payload, card)
    analysis = _json_dict(package.get("analysis")) or _json_dict(evidence_payload.get("analysis"))
    implication = (
        _json_dict(package.get("implication"))
        or _json_dict(evidence_payload.get("implication"))
        or _json_dict(card.get("implication"))
    )
    classification = (
        _json_dict(package.get("classification"))
        or _json_dict(evidence_payload.get("classification"))
        or _classification_from_card(card)
    )
    validation = (
        _json_dict(package.get("validation"))
        or _json_dict(evidence_payload.get("validation"))
        or _validation_from_card(card)
    )
    integrated_issue_id = _first_text(
        card.get("integrated_issue_id"),
        evidence_payload.get("integrated_issue_id"),
        package.get("integrated_issue_id"),
        issue_row.get("id"),
    )
    card_id = _first_text(card.get("card_id"), card.get("id"))
    display_summary = _summary_lines(card)
    evidence_refs = _evidence_refs(
        integrated_issue_id=integrated_issue_id,
        card_id=card_id,
        integrated_issue=integrated_issue,
        issue_row=issue_row,
        evidence_payload=evidence_payload,
    )
    source_links = (
        _json_list(evidence_payload.get("source_links"))
        or _json_list(card.get("source_links"))
        or _json_list(card.get("sources"))
    )
    source_ids = _int_list(
        card.get("source_raw_article_ids")
        or integrated_issue.get("source_article_ids")
        or issue_row.get("source_ids")
    )
    flags = _quality_flags(
        integrated_issue_id=integrated_issue_id,
        issue_row=issue_row,
        package=package,
        integrated_issue=integrated_issue,
        analysis=analysis,
        implication=implication,
        evidence_refs=evidence_refs,
        display_summary=display_summary,
    )
    return AnalysisUnit(
        integrated_issue_id=integrated_issue_id or None,
        card_id=card_id or None,
        integrated_issue=integrated_issue,
        analysis=analysis,
        implication=implication,
        classification=classification,
        validation=validation,
        source_raw_article_ids=source_ids,
        evidence_refs=evidence_refs,
        source_links=source_links,
        display_summary=display_summary,
        card=card,
        integrated_issue_row=issue_row,
        quality_flags=flags,
    )


def _fetch_card_rows(card_ids: list[str], *, user_id: str | None = None) -> list[dict[str, Any]]:
    placeholders = ",".join(f":id_{index}" for index in range(len(card_ids)))
    params = {f"id_{index}": card_id for index, card_id in enumerate(card_ids)}
    sql_v40 = text(f"""
        SELECT
            cn.id,
            cn.id AS card_id,
            cn.company,
            COALESCE(cn.peer_company_id, cn.company) AS peer_id,
            cn.primary_keyword_category,
            cn.source_raw_article_ids,
            cn.integrated_issue_id,
            cn.title,
            cn.summary_lines,
            cn.event_type,
            cn.importance,
            cn.importance_score,
            cn.implication,
            cn.sources,
            cn.evidence_payload,
            cn.validation_pass,
            cn.validation_sc_score,
            ii.id AS ii_id,
            ii.status AS ii_status,
            ii.is_current AS ii_is_current,
            ii.payload AS ii_payload,
            ii.content_digest AS ii_content_digest,
            ii.evidence AS ii_evidence,
            ii.issue_frame AS ii_issue_frame,
            ii.sources AS ii_sources,
            COALESCE(src.source_links, '[]'::jsonb) AS source_links
        FROM card_news cn
        LEFT JOIN integrated_issues ii ON ii.id = cn.integrated_issue_id
        LEFT JOIN LATERAL (
            SELECT jsonb_agg(
                       jsonb_build_object(
                           'raw_article_id', ra.id,
                           'title', ra.title,
                           'source_name',
                           COALESCE(NULLIF(ra.source_name, ''), NULLIF(ra.publisher, '')),
                           'publisher', ra.publisher,
                           'published_at', ra.published_at,
                           'url', ra.url
                       )
                       ORDER BY ra.published_at DESC NULLS LAST, ra.id DESC
                   ) AS source_links
              FROM raw_articles ra
             WHERE ra.id = cn.primary_raw_article_id
                OR ra.id = ANY(COALESCE(cn.source_raw_article_ids, '{{}}'::bigint[]))
        ) src ON TRUE
        WHERE cn.id IN ({placeholders})
    """)
    sql_legacy = text(f"""
        SELECT
            id,
            id AS card_id,
            company,
            COALESCE(peer_company_id, company) AS peer_id,
            primary_keyword_category,
            source_raw_article_ids,
            title,
            summary_lines,
            event_type,
            importance,
            importance_score,
            implication,
            sources,
            evidence_payload,
            validation_pass,
            validation_sc_score,
            COALESCE(src.source_links, '[]'::jsonb) AS source_links
        FROM card_news
        LEFT JOIN LATERAL (
            SELECT jsonb_agg(
                       jsonb_build_object(
                           'raw_article_id', ra.id,
                           'title', ra.title,
                           'source_name',
                           COALESCE(NULLIF(ra.source_name, ''), NULLIF(ra.publisher, '')),
                           'publisher', ra.publisher,
                           'published_at', ra.published_at,
                           'url', ra.url
                       )
                       ORDER BY ra.published_at DESC NULLS LAST, ra.id DESC
                   ) AS source_links
              FROM raw_articles ra
             WHERE ra.id = card_news.primary_raw_article_id
                OR ra.id = ANY(COALESCE(card_news.source_raw_article_ids, '{{}}'::bigint[]))
        ) src ON TRUE
        WHERE id IN ({placeholders})
    """)
    try:
        with SessionLocal() as db:
            try:
                rows = db.execute(sql_v40, params).mappings().all()
            except Exception as exc:  # noqa: BLE001
                if not _is_missing_v40_storage(exc):
                    raise
                db.rollback()
                rows = db.execute(sql_legacy, params).mappings().all()
    except Exception as exc:  # noqa: BLE001
        log.exception("analysis unit card 조회 실패 | error=%s", exc)
        return []
    normalized_rows = [_normalize_row(dict(row)) for row in rows]
    _apply_user_strategy_projections(normalized_rows, user_id=user_id)
    return _attach_vdb_contexts(normalized_rows)


def _fetch_integrated_issue_rows(
    integrated_issue_ids: list[str],
    *,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    placeholders = ",".join(f":id_{index}" for index in range(len(integrated_issue_ids)))
    params = {f"id_{index}": issue_id for index, issue_id in enumerate(integrated_issue_ids)}
    sql = text(f"""
        SELECT
            cn.id,
            cn.id AS card_id,
            cn.company,
            COALESCE(cn.peer_company_id, cn.company) AS peer_id,
            cn.primary_keyword_category,
            cn.source_raw_article_ids,
            cn.integrated_issue_id,
            cn.title,
            cn.summary_lines,
            cn.event_type,
            cn.importance,
            cn.importance_score,
            cn.implication,
            cn.sources,
            cn.evidence_payload,
            cn.validation_pass,
            cn.validation_sc_score,
            ii.id AS ii_id,
            ii.status AS ii_status,
            ii.is_current AS ii_is_current,
            ii.payload AS ii_payload,
            ii.content_digest AS ii_content_digest,
            ii.evidence AS ii_evidence,
            ii.issue_frame AS ii_issue_frame,
            ii.sources AS ii_sources,
            COALESCE(src.source_links, '[]'::jsonb) AS source_links
        FROM integrated_issues ii
        LEFT JOIN LATERAL (
            SELECT *
            FROM card_news cn
            WHERE cn.integrated_issue_id = ii.id
              AND COALESCE(cn.status, 'ACTIVE') = 'ACTIVE'
            ORDER BY cn.created_at DESC
            LIMIT 1
        ) cn ON TRUE
        LEFT JOIN LATERAL (
            SELECT jsonb_agg(
                       jsonb_build_object(
                           'raw_article_id', ra.id,
                           'title', ra.title,
                           'source_name',
                           COALESCE(NULLIF(ra.source_name, ''), NULLIF(ra.publisher, '')),
                           'publisher', ra.publisher,
                           'published_at', ra.published_at,
                           'url', ra.url
                       )
                       ORDER BY ra.published_at DESC NULLS LAST, ra.id DESC
                   ) AS source_links
              FROM raw_articles ra
             WHERE ra.id = cn.primary_raw_article_id
                OR ra.id = ANY(COALESCE(cn.source_raw_article_ids, '{{}}'::bigint[]))
        ) src ON TRUE
        WHERE ii.id IN ({placeholders})
    """)
    try:
        with SessionLocal() as db:
            rows = db.execute(sql, params).mappings().all()
    except Exception as exc:  # noqa: BLE001
        if _is_missing_v40_storage(exc):
            log.info("integrated_issues unavailable for AnalysisUnit lookup")
            return []
        log.exception("analysis unit integrated issue 조회 실패 | error=%s", exc)
        return []
    normalized_rows = [_normalize_row(dict(row)) for row in rows]
    _apply_user_strategy_projections(normalized_rows, user_id=user_id)
    return _attach_vdb_contexts(normalized_rows)


def _apply_user_strategy_projections(rows: list[dict[str, Any]], *, user_id: str | None) -> None:
    if not rows or not user_id:
        return
    card_ids = _dedupe(
        [
            str(row.get("card_id") or row.get("id") or "").strip()
            for row in rows
            if str(row.get("card_id") or row.get("id") or "").strip()
        ]
    )
    if not card_ids:
        return
    placeholders = ",".join(f":card_id_{index}" for index in range(len(card_ids)))
    params: dict[str, Any] = {"user_id": user_id}
    params.update({f"card_id_{index}": card_id for index, card_id in enumerate(card_ids)})
    sql = text(f"""
        SELECT card_news_id,
               applied_action,
               applied_at
          FROM card_news_strategy_context_projections
         WHERE user_id = CAST(:user_id AS uuid)
           AND is_applied = TRUE
           AND card_news_id IN ({placeholders})
    """)
    try:
        with SessionLocal() as db:
            projection_rows = db.execute(sql, params).mappings().all()
    except Exception as exc:  # noqa: BLE001
        log.info("user strategy projection lookup skipped | error=%s", exc)
        return
    projections = {
        str(row.get("card_news_id")): {
            "applied_action": _json_dict(row.get("applied_action")),
            "applied_at": str(row.get("applied_at") or ""),
        }
        for row in projection_rows
        if row.get("card_news_id")
    }
    for row in rows:
        card_id = str(row.get("card_id") or row.get("id") or "")
        projection = projections.get(card_id)
        if not projection:
            continue
        action = _json_dict(projection.get("applied_action"))
        if not action:
            continue
        evidence_payload = _json_dict(row.get("evidence_payload"))
        package = _json_dict(evidence_payload.get("analysis_package"))
        base_implication = (
            _json_dict(package.get("implication"))
            or _json_dict(evidence_payload.get("implication"))
            or _json_dict(row.get("implication"))
        )
        implication = _implication_with_applied_action(base_implication, action)
        if package:
            package["implication"] = implication
            evidence_payload["analysis_package"] = package
        evidence_payload["implication"] = implication
        evidence_payload["strategy_context_projection"] = {
            "is_applied": True,
            "applied_at": projection.get("applied_at"),
        }
        row["implication"] = implication
        row["evidence_payload"] = evidence_payload


def _implication_with_applied_action(
    base_implication: dict[str, Any],
    applied_action: dict[str, Any],
) -> dict[str, Any]:
    merged = json.loads(json.dumps(base_implication or {}, ensure_ascii=False))
    frontend = _json_dict(merged.get("frontend"))
    if not frontend:
        frontend = dict(merged)
    for key in (
        "suggested_actions",
        "response_directions",
        "skax_checkpoints",
        "response_direction_blocks",
        "suggested_action_items",
        "skax_checkpoint_blocks",
    ):
        if key in applied_action:
            frontend[key] = applied_action[key]
    merged["frontend"] = frontend

    frontend_ready_action = _json_dict(applied_action.get("frontend_ready_suggested_action"))
    if frontend_ready_action:
        frontend_ready = _json_dict(merged.get("frontend_ready"))
        frontend_ready["suggested_action"] = frontend_ready_action
        merged["frontend_ready"] = frontend_ready

    industry_actions = _json_list(applied_action.get("industry_frontend_ready_actions"))
    if industry_actions:
        industry_ready = _json_dict(merged.get("industry_frontend_ready"))
        items = _json_list(industry_ready.get("items"))
        updated_items: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            item_map = _json_dict(item)
            if index < len(industry_actions):
                action_map = _json_dict(industry_actions[index])
                if action_map:
                    item_map["suggested_action"] = action_map
            updated_items.append(item_map)
        industry_ready["items"] = updated_items
        merged["industry_frontend_ready"] = industry_ready
    return merged


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    row["implication"] = _json_dict(row.get("implication"))
    row["sources"] = _json_list(row.get("sources"))
    row["evidence_payload"] = _json_dict(row.get("evidence_payload"))
    row["source_raw_article_ids"] = _int_list(row.get("source_raw_article_ids"))
    row["source_links"] = _json_list(row.get("source_links"))
    row["summary_lines"] = _json_list(row.get("summary_lines"))
    row["integrated_issue_id"] = _first_text(row.get("integrated_issue_id"), row.get("ii_id"))
    return row


def _attach_vdb_contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    try:
        from src.rag.content_index import attach_vdb_contexts_to_rows

        return attach_vdb_contexts_to_rows(rows)
    except Exception as exc:  # noqa: BLE001
        log.debug("analysis unit content VDB hydration skipped | error=%s", exc)
        return rows


def _integrated_issue_row(card: dict[str, Any]) -> dict[str, Any]:
    row = {
        "id": _first_text(card.get("ii_id"), card.get("integrated_issue_id")),
        "status": card.get("ii_status"),
        "is_current": card.get("ii_is_current"),
        "payload": _json_dict(card.get("ii_payload")),
        "content_digest": _json_dict(card.get("ii_content_digest")),
        "evidence": _json_dict(card.get("ii_evidence")),
        "issue_frame": _json_dict(card.get("ii_issue_frame")),
        "sources": _json_list(card.get("ii_sources")),
    }
    return {key: value for key, value in row.items() if value not in (None, "", {}, [])}


def _analysis_package_from_sources(*sources: object) -> dict[str, Any]:
    for source in sources:
        if not isinstance(source, dict):
            continue
        package = source.get("analysis_package")
        if isinstance(package, dict):
            return package
    return {}


def _integrated_issue_from_sources(
    issue_row: dict[str, Any],
    package: dict[str, Any],
    evidence_payload: dict[str, Any],
    card: dict[str, Any],
) -> dict[str, Any]:
    issue = _json_dict(issue_row.get("payload"))
    if issue:
        issue.setdefault("content_digest", issue_row.get("content_digest") or {})
        issue.setdefault("evidence", issue_row.get("evidence") or {})
        return issue
    issue = _json_dict(package.get("integrated_issue") or package.get("summary"))
    if issue:
        return issue
    issue = _json_dict(evidence_payload.get("integrated_issue"))
    if issue:
        return issue
    return {
        "headline": card.get("title"),
        "fact_summary": _summary_lines(card),
        "source_article_ids": _int_list(card.get("source_raw_article_ids")),
        "is_valid_summary": False,
    }


def _classification_from_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "sector": card.get("primary_keyword_category"),
        "event_type": card.get("event_type"),
        "importance": card.get("importance"),
        "importance_score": card.get("importance_score"),
        "company": card.get("company"),
        "peer_id": card.get("peer_id"),
    }


def _validation_from_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass": bool(card.get("validation_pass")),
        "sc_score": _optional_float(card.get("validation_sc_score")) or 0.0,
    }


def _evidence_refs(
    *,
    integrated_issue_id: str,
    card_id: str,
    integrated_issue: dict[str, Any],
    issue_row: dict[str, Any],
    evidence_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    candidates = [
        _json_list(_json_dict(issue_row.get("evidence")).get("references")),
        _json_list(integrated_issue.get("fact_basis")),
        _json_list(_json_dict(integrated_issue.get("evidence")).get("references")),
        _json_list(evidence_payload.get("evidence_refs")),
        _json_list(evidence_payload.get("source_links")),
    ]
    for group in candidates:
        for item in group:
            if not isinstance(item, dict):
                continue
            text_value = _first_text(
                item.get("evidence_text"),
                item.get("fact"),
                item.get("text"),
                item.get("title"),
            )
            if not text_value:
                continue
            refs.append(
                {
                    "integrated_issue_id": integrated_issue_id or None,
                    "card_id": card_id or None,
                    "evidence_ref_id": _first_text(
                        item.get("evidence_ref_id"),
                        item.get("fact_id"),
                        *(_str_list(item.get("fact_ids"))[:1]),
                    )
                    or None,
                    "text": text_value,
                    "source_ids": _int_list(
                        item.get("source_article_ids")
                        or item.get("source_ids")
                        or item.get("raw_article_ids")
                    ),
                }
            )
    return _dedupe_evidence(refs)[:20]


def _quality_flags(
    *,
    integrated_issue_id: str,
    issue_row: dict[str, Any],
    package: dict[str, Any],
    integrated_issue: dict[str, Any],
    analysis: dict[str, Any],
    implication: dict[str, Any],
    evidence_refs: list[dict[str, Any]],
    display_summary: list[str],
) -> list[str]:
    flags: list[str] = []
    if not integrated_issue_id:
        flags.append(QUALITY_MISSING_INTEGRATED_ISSUE_ID)
    if issue_row and (
        str(issue_row.get("status") or "active") != "active" or issue_row.get("is_current") is False
    ):
        flags.append(QUALITY_SUPERSEDED_INTEGRATED_ISSUE)
    if not package or not (analysis or implication):
        flags.append(QUALITY_MISSING_ANALYSIS_PACKAGE)
    if display_summary and (
        not (integrated_issue or analysis or implication)
        or (integrated_issue.get("is_valid_summary") is False and not (analysis or implication))
    ):
        flags.append(QUALITY_SUMMARY_ONLY_FALLBACK)
    if not evidence_refs:
        flags.append(QUALITY_EVIDENCE_REF_GAP)
    return _dedupe(flags)


def _unit_title(unit: AnalysisUnit) -> str:
    return _first_text(
        unit.card.get("title"),
        unit.integrated_issue.get("headline"),
        unit.integrated_issue.get("main_issue"),
        unit.integrated_issue.get("one_line_summary"),
        unit.integrated_issue_id,
    )


def _summary_lines(card: dict[str, Any]) -> list[str]:
    return [
        str(item).strip() for item in _json_list(card.get("summary_lines")) if str(item).strip()
    ]


def _is_missing_v40_storage(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "undefinedcolumn" in message or "undefinedtable" in message or "does not exist" in message
    )


def _json_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
        return parsed if isinstance(parsed, list) else [value] if value.strip() else []
    return []


def _int_list(value: object) -> list[int]:
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[int] = []
    for item in values:
        try:
            parsed = int(item)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in result:
            result.append(parsed)
    return result


def _str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _clean_ids(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _first_text(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            return text_value
    return ""


def _optional_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
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


def _dedupe_evidence(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in values:
        key = (
            str(item.get("integrated_issue_id") or ""),
            str(item.get("evidence_ref_id") or ""),
            str(item.get("text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
