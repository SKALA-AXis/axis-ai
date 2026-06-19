"""peer_swot result_persister — extracted from facade (move-only)."""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from src.db.postgres import SessionLocal
from src.services.peer_swot.data_fetcher import (  # noqa: F401
    annotate_peer_context,
    fetch_business_signals,
    fetch_financial_metrics,
    fetch_peers,
)
from src.services.peer_swot.evidence_builder import (  # noqa: F401
    build_company_evidence,
    build_diagnostic_basis_points,
    build_diagnostic_evidence,
    build_evidence_packs,
    build_pack,
    build_peer_coverage,
    top_values,
)
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
from src.services.peer_swot.result_normalizer import (  # noqa: F401
    build_comparison_evidence_summary,
    build_comparison_reasoning_summary,
    build_diagnostic_swot_body,
    build_overall_change_object,
    build_overall_comparison_body,
    build_overall_swot_body,
    build_swot_evidence_summary,
    build_swot_reasoning_summary,
    ensure_overall_multi_peer_signal_refs,
    fallback_comparison_item,
    fallback_swot_item,
    filter_swot_diagnostics_for_comparison,
    normalize_comparison_points,
    normalize_swot_items,
    separate_swot_body_from_comparison,
    signals_for_label,
    swot_text_overlaps_comparison,
    themes_from_diagnostics,
    themes_from_signal_refs,
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

log = logging.getLogger("generate_peer_swot_llm_preview")


def build_analysis_trace(
    comparison_points: list[dict[str, Any]],
    swot_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for item in comparison_points:
        label = str(item.get("label") or "비교 포인트")
        refs = [str(ref) for ref in item.get("evidence_refs") or []]
        reasoning = compact_text(str(item.get("reasoning_summary") or ""), 300)
        evidence = compact_text(str(item.get("evidence_summary") or ""), 280)
        trace.append(
            {
                "step": f"{label} 판단",
                "summary": reasoning,
                "reasoning": reasoning,
                "evidence": evidence,
                "evidence_refs": refs,
            }
        )
    for item in swot_items:
        label = str(item.get("label") or "SWOT")
        refs = [str(ref) for ref in item.get("evidence_refs") or []]
        reasoning = compact_text(str(item.get("reasoning_summary") or ""), 320)
        evidence = compact_text(str(item.get("evidence_summary") or ""), 300)
        trace.append(
            {
                "step": f"{label} 판단",
                "summary": reasoning,
                "reasoning": reasoning,
                "evidence": evidence,
                "evidence_refs": refs,
            }
        )
    return trace


def save_results_to_db(
    evidence_packs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    model: str,
) -> int:
    pack_by_peer_id = {str(pack.get("peer", {}).get("id")): pack for pack in evidence_packs}
    saved_count = 0
    with SessionLocal() as db:
        for result in results:
            peer_id = str(result.get("peer_id") or "")
            evidence_pack = pack_by_peer_id.get(peer_id)
            if not peer_id or evidence_pack is None:
                log.warning("DB 저장 건너뜀 | peer_id=%s evidence_pack 없음", peer_id)
                continue
            save_result_to_db(db, evidence_pack, result, model=model)
            saved_count += 1
        db.commit()
    log.info("LLM 분석 스냅샷 DB 저장 완료 | count=%s", saved_count)
    return saved_count


def save_result_to_db(
    db: Any, evidence_pack: dict[str, Any], result: dict[str, Any], *, model: str
) -> None:
    peer = evidence_pack["peer"]
    peer_id = str(peer["id"])
    comparison_mode = str(
        evidence_pack.get("comparison_mode") or result.get("comparison_mode") or "peer_vs_sk_ax"
    )
    scope = "all" if peer_id == "all" else "company"
    evidence_refs = collect_evidence_ids_from_result(result)
    source_signal_ids = sorted(
        extract_numeric_ids(evidence_refs, "signal:") | collect_signal_ids(evidence_pack)
    )
    source_metric_ids = sorted(
        extract_numeric_ids(evidence_refs, "metric:") | collect_metric_ids(evidence_pack)
    )
    source_raw_article_ids = sorted(collect_raw_article_ids(evidence_pack))
    peer_ids = collect_peer_ids(evidence_pack)
    confidence = average_confidence(result)
    analysis_trace = (
        result.get("analysis_trace") if isinstance(result.get("analysis_trace"), list) else []
    )
    params = {
        "scope": scope,
        "peer_id": peer_id,
        "reference_peer_id": SK_AX_ID,
        "comparison_mode": comparison_mode,
        "prompt_version": result.get("prompt_version") or PROMPT_VERSION,
        "model_name": result.get("model_name") or model,
        "evidence_hash": result.get("evidence_hash") or evidence_pack.get("evidence_hash"),
        "input_snapshot": json.dumps(evidence_pack, ensure_ascii=False, default=str),
        "output_payload": json.dumps(result, ensure_ascii=False, default=str),
        "analysis_trace": json.dumps(analysis_trace, ensure_ascii=False, default=str),
        "provenance": json.dumps(build_provenance(result), ensure_ascii=False, default=str),
        "confidence": confidence,
        "source_raw_article_ids": source_raw_article_ids,
        "source_signal_ids": source_signal_ids,
        "source_metric_ids": source_metric_ids,
        "peer_ids": peer_ids,
    }

    update_result = db.execute(
        text(
            """
            UPDATE peer_llm_analysis_snapshots
            SET
                scope = :scope,
                reference_peer_id = :reference_peer_id,
                schema_version = 'peer_swot_comparison_v1',
                status = 'active',
                evidence_hash = :evidence_hash,
                input_snapshot = CAST(:input_snapshot AS jsonb),
                output_payload = CAST(:output_payload AS jsonb),
                analysis_trace = CAST(:analysis_trace AS jsonb),
                provenance = CAST(:provenance AS jsonb),
                confidence = :confidence,
                source_raw_article_ids = :source_raw_article_ids,
                source_signal_ids = :source_signal_ids,
                source_metric_ids = :source_metric_ids,
                peer_ids = :peer_ids,
                generated_at = NOW(),
                updated_at = NOW()
            WHERE analysis_type = 'peer_swot_comparison'
              AND peer_id = :peer_id
              AND comparison_mode = :comparison_mode
              AND prompt_version = :prompt_version
              AND COALESCE(model_name, '') = COALESCE(:model_name, '')
            """
        ),
        params,
    )
    if update_result.rowcount and update_result.rowcount > 0:
        return

    db.execute(
        text(
            """
            INSERT INTO peer_llm_analysis_snapshots (
                analysis_type,
                scope,
                peer_id,
                reference_peer_id,
                comparison_mode,
                schema_version,
                prompt_version,
                model_name,
                status,
                evidence_hash,
                input_snapshot,
                output_payload,
                analysis_trace,
                provenance,
                confidence,
                source_raw_article_ids,
                source_signal_ids,
                source_metric_ids,
                peer_ids,
                generated_at
            )
            VALUES (
                'peer_swot_comparison',
                :scope,
                :peer_id,
                :reference_peer_id,
                :comparison_mode,
                'peer_swot_comparison_v1',
                :prompt_version,
                :model_name,
                'active',
                :evidence_hash,
                CAST(:input_snapshot AS jsonb),
                CAST(:output_payload AS jsonb),
                CAST(:analysis_trace AS jsonb),
                CAST(:provenance AS jsonb),
                :confidence,
                :source_raw_article_ids,
                :source_signal_ids,
                :source_metric_ids,
                :peer_ids,
                NOW()
            )
            ON CONFLICT (
                analysis_type,
                peer_id,
                comparison_mode,
                evidence_hash,
                prompt_version,
                (COALESCE(model_name, ''))
            )
            DO UPDATE SET
                scope = EXCLUDED.scope,
                reference_peer_id = EXCLUDED.reference_peer_id,
                schema_version = EXCLUDED.schema_version,
                status = 'active',
                input_snapshot = EXCLUDED.input_snapshot,
                output_payload = EXCLUDED.output_payload,
                analysis_trace = EXCLUDED.analysis_trace,
                provenance = EXCLUDED.provenance,
                confidence = EXCLUDED.confidence,
                source_raw_article_ids = EXCLUDED.source_raw_article_ids,
                source_signal_ids = EXCLUDED.source_signal_ids,
                source_metric_ids = EXCLUDED.source_metric_ids,
                peer_ids = EXCLUDED.peer_ids,
                generated_at = NOW(),
                updated_at = NOW()
            """
        ),
        params,
    )


def collect_signal_ids(evidence_pack: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for item in iter_evidence_items(evidence_pack):
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id.startswith("signal:"):
            raw_id = evidence_id.removeprefix("signal:")
            if raw_id.isdigit():
                ids.add(int(raw_id))
        for ref in item.get("source_signal_refs") or []:
            if (
                isinstance(ref, str)
                and ref.startswith("signal:")
                and ref.removeprefix("signal:").isdigit()
            ):
                ids.add(int(ref.removeprefix("signal:")))
    return ids


def collect_metric_ids(evidence_pack: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for item in iter_evidence_items(evidence_pack):
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id.startswith("metric:"):
            raw_id = evidence_id.removeprefix("metric:")
            if raw_id.isdigit():
                ids.add(int(raw_id))
        for ref in item.get("source_metric_refs") or []:
            if (
                isinstance(ref, str)
                and ref.startswith("metric:")
                and ref.removeprefix("metric:").isdigit()
            ):
                ids.add(int(ref.removeprefix("metric:")))
    return ids


def collect_raw_article_ids(evidence_pack: dict[str, Any]) -> set[int]:
    article_ids: set[int] = set()
    for item in iter_evidence_items(evidence_pack):
        article_id = item.get("article_id")
        if isinstance(article_id, int):
            article_ids.add(article_id)
        elif isinstance(article_id, str) and article_id.isdigit():
            article_ids.add(int(article_id))
    return article_ids


def collect_peer_ids(evidence_pack: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for company in evidence_pack.get("companies") or []:
        peer = company.get("peer") if isinstance(company, dict) else None
        peer_id = peer.get("id") if isinstance(peer, dict) else None
        if isinstance(peer_id, str) and peer_id not in ids:
            ids.append(peer_id)
    peer_id = evidence_pack.get("peer", {}).get("id")
    if isinstance(peer_id, str) and peer_id != "all" and peer_id not in ids:
        ids.append(peer_id)
    return ids


def average_confidence(result: dict[str, Any]) -> float | None:
    values: list[float] = []
    for section_name in ("comparison_points", "swot"):
        for item in result.get(section_name) or []:
            value = item.get("confidence") if isinstance(item, dict) else None
            numeric = to_float(value)
            if numeric is not None:
                values.append(numeric)
    if not values:
        return None
    return sum(values) / len(values)
