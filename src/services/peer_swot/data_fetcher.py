"""peer_swot data_fetcher — extracted from facade (move-only)."""

# ruff: noqa: E501  — long lines inherited from E501-exempt facade

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from src.services.peer_swot.prompts import (  # noqa: F401
    COMPARISON_LABELS,
    COMPARISON_PROMPT,
    COMPARISON_SIGNAL_TYPES,
    COMPARISON_SOURCE_TYPES,
    COMPETITOR_PEER_IDS,
    COMPETITOR_PEER_NAMES,
    DEFAULT_MODEL,
    FINANCIAL_NUMBER_PATTERN,
    GENERIC_CHANGE_OBJECT_BY_LABEL,
    GENERIC_SWOT_TITLE_BY_LABEL,
    NOISE_PHRASES,
    PROMPT_VERSION,
    SK_AX_ID,
    SWOT_FACTOR_TYPE_BY_LABEL,
    SWOT_LABELS,
    SWOT_METRIC_NAMES,
    SWOT_PROMPT,
    SWOT_SIGNAL_TYPES,
    TARGET_PEER_IDS,
)
from src.services.peer_swot.utils import (  # noqa: F401
    build_metric_source_citation,
    build_overall_check_point,
    build_overall_reasoning_coverage_note,
    build_overall_rules,
    build_provenance,
    build_signal_source_citation,
    canonical_comparison_label,
    canonical_swot_label,
    choose_diagnostic_refs_for_label,
    choose_signal_refs_for_label,
    clamp_confidence,
    clean_display_text,
    collect_evidence_ids_from_result,
    collect_urls_for_refs,
    compact_date,
    compact_text,
    content_overlaps_comparison,
    count_distinct_peers_for_refs,
    diagnostic_for_label,
    extract_numeric_ids,
    extract_similarity_tokens,
    extract_specific_phrase,
    extract_theme_terms,
    fallback_check_point,
    fallback_swot_title,
    flatten_limited,
    has_negative_metric_signal,
    has_risk_term,
    has_tech_term,
    hash_payload,
    human_source_name,
    infer_change_object,
    is_generic_change_object,
    is_overall_pack,
    is_semantically_close,
    is_usable_signal_text,
    is_weak_info_text,
    iso_date,
    iter_evidence_items,
    joined_item_text,
    normalize_peer,
    normalize_refs,
    normalize_similarity_text,
    parse_json_object,
    print_or_write_json,
    ref_matches_label,
    remove_financial_number_sentences,
    rough_token_count,
    sanitize_overall_company_names,
    summarize_profile,
    to_float,
    unique_evidence_items,
)


def annotate_peer_context(items: list[dict[str, Any]], peer: dict[str, Any]) -> None:
    for item in items:
        item["peer_id"] = peer.get("id")
        item["peer_name"] = peer.get("name")


def fetch_peers(db: Any, peer_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT id, name, profile_snapshot
            FROM peer_companies
            WHERE id = ANY(:peer_ids)
            ORDER BY CASE id
                WHEN 'sk_ax' THEN 0
                WHEN 'samsung_sds' THEN 1
                WHEN 'lg_cns' THEN 2
                WHEN 'hyundai_autoever' THEN 3
                WHEN 'posco_dx' THEN 4
                ELSE 99
            END
            """
        ),
        {"peer_ids": list(peer_ids)},
    ).mappings()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "profile_snapshot": row["profile_snapshot"],
        }
        for row in rows
    ]


def fetch_business_signals(
    db: Any,
    peer_id: str,
    *,
    days: int,
    limit: int,
    signal_types: set[str] | None,
    source_types: set[str] | None,
) -> list[dict[str, Any]]:
    filters = [
        "rabs.peer_id = :peer_id",
        "COALESCE(ra.published_at, ra.collected_at, ra.created_at, rabs.created_at) >= "
        "NOW() - (:days || ' days')::interval",
        "COALESCE(rabs.summary, '') <> ''",
    ]
    params: dict[str, Any] = {
        "peer_id": peer_id,
        "days": days,
        "limit": max(limit * 5, limit),
    }
    if signal_types:
        filters.append("rabs.signal_type = ANY(:signal_types)")
        params["signal_types"] = list(signal_types)
    if source_types:
        filters.append("rabs.source_type = ANY(:source_types)")
        params["source_types"] = list(source_types)

    rows = db.execute(
        text(
            f"""
            SELECT
                rabs.id,
                rabs.raw_article_id,
                rabs.source_type,
                rabs.business_area,
                rabs.signal_type,
                rabs.sentiment,
                rabs.summary,
                rabs.evidence_text,
                rabs.confidence,
                rabs.created_at,
                rabs.updated_at,
                ra.title,
                ra.url,
                COALESCE(ra.published_at, ra.collected_at, ra.created_at, rabs.created_at) AS evidence_at,
                ra.source_name
            FROM raw_article_business_signals rabs
            JOIN raw_articles ra ON ra.id = rabs.raw_article_id
            WHERE {" AND ".join(filters)}
            ORDER BY
                rabs.updated_at DESC NULLS LAST,
                rabs.confidence DESC NULLS LAST,
                COALESCE(ra.published_at, ra.collected_at, ra.created_at, rabs.created_at) DESC NULLS LAST,
                rabs.id DESC
            LIMIT :limit
            """
        ),
        params,
    ).mappings()

    signals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        summary = compact_text(row["summary"], 300)
        evidence_text = compact_text(row["evidence_text"], 260)
        if not is_usable_signal_text(summary, evidence_text) or summary in seen:
            continue
        seen.add(summary)
        signals.append(
            {
                "evidence_id": f"signal:{row['id']}",
                "article_id": row["raw_article_id"],
                "source_type": row["source_type"],
                "source_name": row["source_name"],
                "business_area": row["business_area"],
                "signal_type": row["signal_type"],
                "sentiment": row["sentiment"],
                "title": compact_text(row["title"], 120),
                "summary": summary,
                "evidence_text": evidence_text,
                "url": row["url"],
                "date": iso_date(row["evidence_at"]),
                "updated_at": iso_date(row["updated_at"]),
                "confidence": to_float(row["confidence"]),
            }
        )
    return signals[:limit]


def fetch_financial_metrics(
    db: Any,
    peer_id: str,
    *,
    limit: int,
    metric_names: set[str],
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT
                rafm.id,
                rafm.raw_article_id,
                rafm.period,
                rafm.metric_name,
                rafm.metric_label,
                rafm.metric_scope,
                rafm.business_area,
                rafm.value_numeric,
                rafm.value_krwbn,
                rafm.unit,
                rafm.evidence_text,
                rafm.confidence,
                rafm.updated_at,
                ra.title,
                ra.url,
                COALESCE(ra.published_at, ra.collected_at, ra.created_at) AS evidence_at,
                ra.source_name
            FROM raw_article_financial_metrics rafm
            LEFT JOIN raw_articles ra ON ra.id = rafm.raw_article_id
            WHERE rafm.peer_id = :peer_id
              AND rafm.metric_name = ANY(:metric_names)
            ORDER BY
                rafm.updated_at DESC NULLS LAST,
                rafm.period_year DESC NULLS LAST,
                rafm.period_quarter DESC NULLS LAST,
                rafm.confidence DESC NULLS LAST,
                rafm.id DESC
            LIMIT :limit
            """
        ),
        {
            "peer_id": peer_id,
            "metric_names": list(metric_names),
            "limit": max(limit * 4, limit),
        },
    ).mappings()

    metrics: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for row in rows:
        key = (row["period"], row["metric_name"], row["metric_scope"])
        if key in seen:
            continue
        seen.add(key)
        metrics.append(
            {
                "evidence_id": f"metric:{row['id']}",
                "article_id": row["raw_article_id"],
                "period": row["period"],
                "metric_name": row["metric_name"],
                "metric_label": row["metric_label"],
                "metric_scope": row["metric_scope"],
                "business_area": row["business_area"],
                "value_numeric": to_float(row["value_numeric"]),
                "value_krwbn": to_float(row["value_krwbn"]),
                "unit": row["unit"],
                "evidence_text": compact_text(row["evidence_text"], 240),
                "updated_at": iso_date(row["updated_at"]),
                "title": compact_text(row["title"], 120),
                "url": row["url"],
                "date": iso_date(row["evidence_at"]),
                "source_name": row["source_name"],
                "confidence": to_float(row["confidence"]),
            }
        )
    return metrics[:limit]
