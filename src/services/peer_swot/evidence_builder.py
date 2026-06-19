"""peer_swot evidence_builder — extracted from facade (move-only)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from src.db.postgres import SessionLocal
from src.services.peer_swot.data_fetcher import (  # noqa: F401
    annotate_peer_context,
    fetch_business_signals,
    fetch_financial_metrics,
    fetch_peers,
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


def build_evidence_packs(
    peer_ids: tuple[str, ...],
    *,
    include_overall: bool,
    include_companies: bool,
    days: int,
    fallback_days: int,
    signal_limit: int,
    metric_limit: int,
) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        requested_ids = tuple(dict.fromkeys((SK_AX_ID, *peer_ids)))
        peers = fetch_peers(db, requested_ids)
        peer_by_id = {peer["id"]: peer for peer in peers}
        if SK_AX_ID not in peer_by_id:
            raise RuntimeError("SK AX peer row is required")

        company_evidence_by_id = {
            peer["id"]: build_company_evidence(
                db,
                peer,
                days=days,
                fallback_days=fallback_days,
                signal_limit=signal_limit,
                metric_limit=metric_limit,
            )
            for peer in peers
        }

    packs: list[dict[str, Any]] = []
    if include_overall:
        competitor_evidence = [
            company_evidence_by_id[peer_id]
            for peer_id in COMPETITOR_PEER_IDS
            if peer_id in company_evidence_by_id
        ]
        packs.append(
            build_pack(
                peer={"id": "all", "name": "전체 경쟁사", "profile_snapshot": None},
                comparison_mode="overall_competitors_vs_sk_ax",
                companies=competitor_evidence,
            )
        )

    if not include_companies:
        return packs

    for peer_id in peer_ids:
        company_evidence = company_evidence_by_id.get(peer_id)
        if company_evidence is None:
            log.warning("요청 peer를 찾지 못했습니다 | peer_id=%s", peer_id)
            continue
        packs.append(
            build_pack(
                peer=company_evidence["peer"],
                comparison_mode="peer_vs_sk_ax",
                companies=[company_evidence],
            )
        )
    return packs


def build_company_evidence(
    db: Any,
    peer: dict[str, Any],
    *,
    days: int,
    fallback_days: int,
    signal_limit: int,
    metric_limit: int,
) -> dict[str, Any]:
    peer_id = str(peer["id"])
    comparison_signals = fetch_business_signals(
        db,
        peer_id,
        days=days,
        limit=signal_limit,
        signal_types=COMPARISON_SIGNAL_TYPES,
        source_types=COMPARISON_SOURCE_TYPES,
    )
    if not comparison_signals and fallback_days > days:
        comparison_signals = fetch_business_signals(
            db,
            peer_id,
            days=fallback_days,
            limit=signal_limit,
            signal_types=COMPARISON_SIGNAL_TYPES,
            source_types=COMPARISON_SOURCE_TYPES,
        )

    swot_signals = fetch_business_signals(
        db,
        peer_id,
        days=fallback_days,
        limit=signal_limit,
        signal_types=SWOT_SIGNAL_TYPES,
        source_types=None,
    )
    metrics = fetch_financial_metrics(
        db,
        peer_id,
        limit=metric_limit,
        metric_names=SWOT_METRIC_NAMES,
    )
    peer_ref = normalize_peer(peer)
    annotate_peer_context(comparison_signals, peer_ref)
    annotate_peer_context(swot_signals, peer_ref)
    annotate_peer_context(metrics, peer_ref)
    diagnostics = build_diagnostic_evidence(peer, swot_signals, metrics)
    return {
        "peer": peer_ref,
        "comparison_signals": comparison_signals,
        "swot_source_signals": swot_signals,
        "financial_metrics": metrics,
        "diagnostic_evidence": diagnostics,
    }


def build_pack(
    *,
    peer: dict[str, Any],
    comparison_mode: str,
    companies: list[dict[str, Any]],
) -> dict[str, Any]:
    comparison_signals = flatten_limited(
        [company["comparison_signals"] for company in companies],
        limit=max(12, len(companies) * 4),
    )
    diagnostic_evidence = flatten_limited(
        [company["diagnostic_evidence"] for company in companies],
        limit=max(16, len(companies) * 4),
    )
    pack: dict[str, Any] = {
        "peer": normalize_peer(peer),
        "reference_peer": {"id": SK_AX_ID, "name": "SK AX"},
        "comparison_mode": comparison_mode,
        "prompt_version": PROMPT_VERSION,
        "comparison_input": {
            "purpose": "recent observed movements only; no financial numbers",
            "allowed_comparison_evidence_refs": [
                item["evidence_id"] for item in comparison_signals
            ],
            "signals": comparison_signals,
            "overall_rules": build_overall_rules(peer, companies),
            "overall_peer_coverage": build_peer_coverage(companies),
        },
        "swot_input": {
            "purpose": "company diagnosis; separate from recent comparison points",
            "allowed_swot_evidence_refs": [item["evidence_id"] for item in diagnostic_evidence],
            "diagnostic_evidence": diagnostic_evidence,
            "overall_rules": build_overall_rules(peer, companies),
            "overall_peer_coverage": build_peer_coverage(companies),
        },
        "companies": companies,
    }
    pack["evidence_hash"] = hash_payload(
        {
            "peer": pack["peer"],
            "comparison_mode": comparison_mode,
            "comparison_refs": pack["comparison_input"]["allowed_comparison_evidence_refs"],
            "swot_refs": pack["swot_input"]["allowed_swot_evidence_refs"],
        }
    )
    return pack


def build_peer_coverage(companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for company in companies:
        peer = company.get("peer") or {}
        signals = company.get("comparison_signals") or []
        coverage.append(
            {
                "peer_id": peer.get("id"),
                "peer_name": peer.get("name"),
                "signal_count": len(signals),
                "top_business_areas": top_values(signals, "business_area", limit=4),
                "top_signal_types": top_values(signals, "signal_type", limit=4),
            }
        )
    return coverage


def top_values(items: list[dict[str, Any]], key: str, *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        value = compact_text(item.get(key), 60)
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return [
        value
        for value, _count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    ]


def build_diagnostic_evidence(
    peer: dict[str, Any],
    signals: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    peer_id = str(peer["id"])
    profile_text = summarize_profile(peer.get("profile_snapshot"))
    business_signals = unique_evidence_items(
        [s for s in signals if str(s.get("signal_type")) not in {"rd", "risk"}]
    )
    tech_signals = unique_evidence_items(
        [s for s in signals if str(s.get("signal_type")) == "rd" or has_tech_term(s)]
    )
    risk_signals = unique_evidence_items(
        [s for s in signals if str(s.get("signal_type")) == "risk" or has_risk_term(s)]
    )
    external_signals = unique_evidence_items(
        [
            s
            for s in signals
            if str(s.get("signal_type")) in {"forecast", "investment", "valuation", "risk"}
            or str(s.get("source_type")) == "securities_report"
        ]
    )
    weakness_metrics = [
        m
        for m in metrics
        if has_negative_metric_signal(m) or str(m.get("metric_name", "")).startswith("operating")
    ]

    specs = [
        (
            "Strength",
            "strength",
            "Internal capability/differentiator",
            "반복적으로 활용 가능한 내부 역량이나 차별화 자산은 무엇인가?",
            "내부 통제 가능 요소이며, 여러 고객·산업·서비스로 재사용될 수 있는 역량인지 판단",
            [*tech_signals[:2], *business_signals[:2]],
            [],
            profile_text,
        ),
        (
            "Weakness",
            "weakness",
            "Internal limitation or improvement area",
            "성과를 제약하는 내부 비용, 실행 부담, 수익성, 집중도 문제는 무엇인가?",
            "회사가 개선하거나 관리해야 하는 내부 제약인지 판단",
            risk_signals[:2],
            weakness_metrics[:2],
            profile_text,
        ),
        (
            "Opportunity",
            "opportunity",
            "External favorable market/customer/technology condition",
            "외부 시장·고객·정책·기술 변화 중 활용 가능한 기회는 무엇인가?",
            "회사가 통제할 수는 없지만 사업 확장에 유리하게 작용할 외부 조건인지 판단",
            external_signals[:3] or business_signals[:2],
            [],
            "",
        ),
        (
            "Threat",
            "threat",
            "External unfavorable competition/regulation/market condition",
            "외부 경쟁, 수요, 규제, 기술 변화 중 성과를 압박할 요인은 무엇인가?",
            "회사가 통제하기 어렵고 사업 속도나 수익성을 낮출 수 있는 외부 조건인지 판단",
            risk_signals[:3] or external_signals[:2],
            [],
            "",
        ),
    ]

    diagnostics: list[dict[str, Any]] = []
    for (
        label,
        slug,
        diagnosis_type,
        diagnostic_question,
        judgment_basis,
        selected_signals,
        selected_metrics,
        profile,
    ) in specs:
        source_refs = [item["evidence_id"] for item in selected_signals + selected_metrics]
        digest = hashlib.sha1(
            json.dumps(
                [peer_id, label, source_refs, profile], ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()[:10]
        basis_points = build_diagnostic_basis_points(
            label, selected_signals, selected_metrics, profile
        )
        source_signal_citations = [build_signal_source_citation(item) for item in selected_signals]
        source_metric_citations = [build_metric_source_citation(item) for item in selected_metrics]
        source_citations = [*source_signal_citations, *source_metric_citations]
        diagnostics.append(
            {
                "evidence_id": f"diagnostic:{peer_id}:{slug}:{digest}",
                "peer_id": peer_id,
                "peer_name": peer.get("name"),
                "label": label,
                "diagnosis_type": diagnosis_type,
                "diagnostic_question": diagnostic_question,
                "judgment_basis": judgment_basis,
                "basis_points": basis_points,
                "source_citations": [citation for citation in source_citations if citation],
                "source_signal_citations": [
                    citation for citation in source_signal_citations if citation
                ],
                "source_metric_citations": [
                    citation for citation in source_metric_citations if citation
                ],
                "factor_type": SWOT_FACTOR_TYPE_BY_LABEL[label],
                "profile_hint": compact_text(profile, 320),
                "source_signal_refs": [item["evidence_id"] for item in selected_signals],
                "source_metric_refs": [item["evidence_id"] for item in selected_metrics],
                "signal_summaries": [
                    compact_text(
                        " / ".join(
                            str(value or "")
                            for value in (
                                item.get("business_area"),
                                item.get("signal_type"),
                                item.get("summary"),
                            )
                        ),
                        260,
                    )
                    for item in selected_signals
                ],
                "metric_summaries": [
                    compact_text(
                        " / ".join(
                            str(value or "")
                            for value in (
                                item.get("period"),
                                item.get("metric_label") or item.get("metric_name"),
                                item.get("business_area"),
                                item.get("evidence_text"),
                            )
                        ),
                        220,
                    )
                    for item in selected_metrics
                ],
            }
        )
    return diagnostics


def build_diagnostic_basis_points(
    label: str,
    signals: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    profile: str,
) -> list[str]:
    points: list[str] = []
    if profile and label in {"Strength", "Weakness"}:
        points.append(f"회사 프로필상 내부 사업·역량 설명: {compact_text(profile, 120)}")
    for signal in signals[:2]:
        area = compact_text(signal.get("business_area"), 40) or "사업 영역 미상"
        signal_type = compact_text(signal.get("signal_type"), 30) or "신호 유형 미상"
        summary = compact_text(signal.get("summary"), 120)
        points.append(f"{area}/{signal_type} 근거: {summary}")
    for metric in metrics[:2]:
        metric_name = compact_text(metric.get("metric_label") or metric.get("metric_name"), 50)
        period = compact_text(metric.get("period"), 20)
        evidence = compact_text(metric.get("evidence_text"), 120)
        points.append(f"{period} {metric_name} 근거: {evidence}")
    if not points:
        points.append("해당 SWOT 항목을 강하게 뒷받침하는 독립 근거가 제한적입니다.")
    return points[:4]
