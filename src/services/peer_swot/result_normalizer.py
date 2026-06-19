"""peer_swot result_normalizer — extracted from facade (move-only)."""

# ruff: noqa: E501  — long lines inherited from E501-exempt facade

from __future__ import annotations

from typing import Any

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


def filter_swot_diagnostics_for_comparison(
    diagnostics: list[dict[str, Any]],
    comparison_points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparison_refs = {
        str(ref)
        for item in comparison_points
        if isinstance(item, dict)
        for ref in item.get("evidence_refs") or []
    }
    comparison_texts = [
        compact_text(
            " / ".join(str(item.get(key) or "") for key in ("change_object", "body")),
            260,
        )
        for item in comparison_points
        if isinstance(item, dict)
    ]
    filtered_diagnostics: list[dict[str, Any]] = []
    for diagnosis in diagnostics:
        filtered = dict(diagnosis)
        source_signal_refs = [str(ref) for ref in diagnosis.get("source_signal_refs") or []]
        signal_summaries = [str(summary) for summary in diagnosis.get("signal_summaries") or []]
        signal_citations = [
            str(citation) for citation in diagnosis.get("source_signal_citations") or []
        ]
        kept_refs: list[str] = []
        kept_summaries: list[str] = []
        kept_signal_citations: list[str] = []
        removed_count = 0
        for index, summary in enumerate(signal_summaries):
            ref = source_signal_refs[index] if index < len(source_signal_refs) else ""
            if ref in comparison_refs or content_overlaps_comparison(summary, comparison_texts):
                removed_count += 1
                continue
            if ref:
                kept_refs.append(ref)
            kept_summaries.append(summary)
            if index < len(signal_citations):
                kept_signal_citations.append(signal_citations[index])

        basis_points = [str(point) for point in diagnosis.get("basis_points") or []]
        kept_basis_points = [
            point
            for point in basis_points
            if not content_overlaps_comparison(point, comparison_texts)
        ]
        removed_count += len(basis_points) - len(kept_basis_points)

        filtered["source_signal_refs"] = kept_refs
        filtered["signal_summaries"] = kept_summaries
        filtered["source_signal_citations"] = kept_signal_citations
        filtered["source_citations"] = [
            *kept_signal_citations,
            *[str(citation) for citation in diagnosis.get("source_metric_citations") or []],
        ]
        filtered["basis_points"] = kept_basis_points
        if not kept_basis_points and not kept_summaries and not filtered.get("source_metric_refs"):
            filtered["basis_points"] = [
                "비교 포인트와 독립적으로 사용할 수 있는 SWOT 진단 근거가 제한적입니다."
            ]
            filtered["insufficient_independent_evidence"] = True
        if removed_count > 0:
            filtered["comparison_content_excluded"] = True
            filtered["diagnosis_guardrail"] = (
                "비교 포인트와 같은 원문 근거·주제는 제외했으며, 남은 근거로만 SWOT을 판단한다."
            )
        filtered_diagnostics.append(filtered)
    return filtered_diagnostics


def ensure_overall_multi_peer_signal_refs(
    label: str,
    refs: list[str],
    signals: list[dict[str, Any]],
) -> list[str]:
    selected = [signal for signal in signals if str(signal.get("evidence_id")) in set(refs)]
    selected_peer_ids = {str(signal.get("peer_id")) for signal in selected if signal.get("peer_id")}
    if len(selected_peer_ids) >= 2:
        return refs[:3]

    candidates = signals_for_label(label, signals)
    expanded_refs = list(refs)
    for signal in candidates:
        peer_id = str(signal.get("peer_id") or "")
        evidence_id = str(signal.get("evidence_id") or "")
        if not evidence_id or evidence_id in expanded_refs:
            continue
        if peer_id in selected_peer_ids and len(selected_peer_ids) >= 1:
            continue
        expanded_refs.append(evidence_id)
        if peer_id:
            selected_peer_ids.add(peer_id)
        if len(selected_peer_ids) >= 2 or len(expanded_refs) >= 3:
            break
    return expanded_refs[:3]


def signals_for_label(label: str, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if label == "기술 신호":
        selected = [
            signal
            for signal in signals
            if str(signal.get("signal_type")) == "rd" or has_tech_term(signal)
        ]
    elif label == "리스크":
        selected = [
            signal
            for signal in signals
            if str(signal.get("signal_type")) == "risk" or has_risk_term(signal)
        ]
    else:
        selected = [
            signal for signal in signals if str(signal.get("signal_type")) not in {"rd", "risk"}
        ]
    return selected or signals


def build_overall_change_object(
    label: str,
    refs: list[str],
    signals: list[dict[str, Any]],
    fallback: str,
) -> str:
    themes = themes_from_signal_refs(refs, signals)
    if themes:
        return "·".join(themes[:2])
    defaults = {
        "사업 신호": "AI·클라우드 사업 흐름",
        "기술 신호": "AI·클라우드 기술 흐름",
        "리스크": "클라우드·AI 경쟁 리스크",
    }
    return defaults.get(label, fallback)


def build_overall_comparison_body(
    label: str,
    refs: list[str],
    signals: list[dict[str, Any]],
    change_object: str,
) -> str:
    themes = themes_from_signal_refs(refs, signals)
    theme_text = "·".join(themes[:3]) if themes else change_object
    bodies = {
        "사업 신호": f"{theme_text}를 중심으로 고객 산업과 서비스 확장 방향이 재편되고 있습니다. 이는 국내 IT서비스 시장의 수요가 단순 SI보다 AI·클라우드 기반 실행 영역으로 이동하고 있음을 보여줍니다.",
        "기술 신호": f"{theme_text} 관련 기술 적용과 플랫폼화가 경쟁 축으로 부상하고 있습니다. AI, 클라우드, 데이터 기반 역량이 산업 공통의 기술 차별화 요소로 이동하는 흐름입니다.",
        "리스크": f"{theme_text} 확대와 함께 실행 부담과 경쟁 압력이 커지고 있습니다. IT서비스 산업의 수요·투자·경쟁 변동성을 함께 관리해야 하는 상황으로 해석됩니다.",
    }
    return bodies[label]


def build_overall_swot_body(
    label: str,
    diagnostics: list[dict[str, Any]],
    evidence_pack: dict[str, Any],
) -> str:
    themes = themes_from_diagnostics(diagnostics)
    theme_text = "·".join(themes[:3]) if themes else "AI·클라우드·산업 DX"
    bodies = {
        "Strength": f"{theme_text} 역량이 고객 산업 전반으로 확장될 수 있는 실행 기반으로 축적되고 있습니다. 이는 국내 IT서비스 시장에서 반복 적용 가능한 기술·운영 역량이 강점으로 작용할 수 있음을 의미합니다.",
        "Weakness": f"{theme_text} 확대 과정에서 비용, 인력, 납기, 수익성 관리 부담이 함께 커질 수 있습니다. 서비스 고도화 속도만큼 내부 운영 효율과 프로젝트 관리 체계를 개선해야 하는 과제로 해석됩니다.",
        "Opportunity": f"{theme_text} 수요가 공공, 금융, 제조 등 여러 고객 산업으로 확장되고 있습니다. 이는 IT서비스 산업이 활용할 수 있는 외부 수요 조건이 넓어지고 있다는 의미입니다.",
        "Threat": f"{theme_text} 시장에서 경쟁 강도와 고객 투자 변동성이 동시에 커지고 있습니다. 성장 속도와 수익성을 압박할 수 있는 외부 조건으로 관리가 필요합니다.",
    }
    return bodies[label]


def themes_from_signal_refs(refs: list[str], signals: list[dict[str, Any]]) -> list[str]:
    by_ref = {str(signal.get("evidence_id")): signal for signal in signals}
    texts = [
        " ".join(str(signal.get(key) or "") for key in ("business_area", "summary", "title"))
        for ref in refs
        for signal in [by_ref.get(ref)]
        if signal
    ]
    return extract_theme_terms(" ".join(texts))


def themes_from_diagnostics(diagnostics: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for diagnosis in diagnostics:
        texts.extend(str(point) for point in diagnosis.get("basis_points") or [])
        texts.extend(str(summary) for summary in diagnosis.get("signal_summaries") or [])
    return extract_theme_terms(" ".join(texts))


def normalize_comparison_points(
    raw: dict[str, Any], evidence_pack: dict[str, Any]
) -> list[dict[str, Any]]:
    raw_items = raw.get("comparison_points") if isinstance(raw, dict) else []
    if not isinstance(raw_items, list):
        raw_items = []

    allowed_refs = set(evidence_pack["comparison_input"]["allowed_comparison_evidence_refs"])
    signals = evidence_pack["comparison_input"]["signals"]
    by_label: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        label = canonical_comparison_label(item.get("label"))
        if label and label not in by_label:
            by_label[label] = item

    normalized: list[dict[str, Any]] = []
    for label in COMPARISON_LABELS:
        item = dict(by_label.get(label) or fallback_comparison_item(label, signals))
        body = clean_display_text(str(item.get("body") or ""))
        body = remove_financial_number_sentences(body)
        if not body:
            body = str(fallback_comparison_item(label, signals)["body"])
        refs = normalize_refs(item.get("evidence_refs"), allowed_refs)
        if not refs:
            refs = choose_signal_refs_for_label(label, signals)
        if is_overall_pack(evidence_pack):
            refs = ensure_overall_multi_peer_signal_refs(label, refs, signals)
        source_urls = collect_urls_for_refs(refs, signals)
        change_object = compact_text(item.get("change_object"), 80) or infer_change_object(
            label, refs, signals
        )
        if is_generic_change_object(change_object):
            change_object = GENERIC_CHANGE_OBJECT_BY_LABEL[label]
        evidence_summary = build_comparison_evidence_summary(label, refs, signals)
        reasoning_summary = build_comparison_reasoning_summary(
            label,
            change_object,
            refs,
            signals,
            evidence_pack=evidence_pack,
        )
        if is_overall_pack(evidence_pack):
            body = build_overall_comparison_body(label, refs, signals, change_object)
            change_object = build_overall_change_object(label, refs, signals, change_object)
            reasoning_summary = sanitize_overall_company_names(reasoning_summary)
            evidence_summary = sanitize_overall_company_names(evidence_summary)
        normalized.append(
            {
                "label": label,
                "change_object": change_object,
                "body": compact_text(body, 420),
                "evidence_refs": refs,
                "source_urls": source_urls,
                "confidence": clamp_confidence(item.get("confidence")),
                "evidence_summary": evidence_summary,
                "reasoning_summary": reasoning_summary,
                "insufficient_evidence": bool(item.get("insufficient_evidence"))
                or not refs
                or change_object == GENERIC_CHANGE_OBJECT_BY_LABEL[label],
            }
        )
    return normalized


def normalize_swot_items(
    raw: dict[str, Any],
    evidence_pack: dict[str, Any],
    *,
    diagnostics_override: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw_items: list[Any] = []
    if isinstance(raw, dict):
        raw_items = raw.get("swot") or raw.get("swot_monitoring_axes") or []
    if not isinstance(raw_items, list):
        raw_items = []

    diagnostics = diagnostics_override or evidence_pack["swot_input"]["diagnostic_evidence"]
    allowed_refs = {str(item.get("evidence_id")) for item in diagnostics if item.get("evidence_id")}
    by_label: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        label = canonical_swot_label(item.get("label"))
        if label is None and index < len(SWOT_LABELS):
            label = SWOT_LABELS[index]
        if label and label not in by_label:
            by_label[label] = item

    normalized: list[dict[str, Any]] = []
    for label in SWOT_LABELS:
        item = dict(by_label.get(label) or fallback_swot_item(label, diagnostics))
        refs = normalize_refs(item.get("evidence_refs"), allowed_refs)
        preferred_refs = choose_diagnostic_refs_for_label(label, diagnostics)
        if not refs or not ref_matches_label(refs[0], label):
            refs = preferred_refs or refs[:1]
        body = clean_display_text(str(item.get("body") or ""))
        if not body:
            body = str(fallback_swot_item(label, diagnostics)["body"])
        title = clean_display_text(item.get("title") or item.get("axis_name") or "")
        if not title:
            title = fallback_swot_title(label, diagnostics)
        if swot_text_overlaps_comparison(title, evidence_pack):
            title = GENERIC_SWOT_TITLE_BY_LABEL[label]
        if is_overall_pack(evidence_pack):
            title = GENERIC_SWOT_TITLE_BY_LABEL[label]
        diagnosis = diagnostic_for_label(label, diagnostics)
        body = separate_swot_body_from_comparison(label, body, title, evidence_pack, diagnostics)
        if is_overall_pack(evidence_pack):
            body = build_overall_swot_body(label, diagnostics, evidence_pack)
        evidence_summary = build_swot_evidence_summary(diagnosis)
        reasoning_summary = build_swot_reasoning_summary(label, title, diagnosis)
        if is_overall_pack(evidence_pack):
            reasoning_summary = sanitize_overall_company_names(reasoning_summary)
        normalized.append(
            {
                "label": label,
                "title": compact_text(title, 80),
                "body": compact_text(body, 430),
                "factor_type": SWOT_FACTOR_TYPE_BY_LABEL[label],
                "check_point": compact_text(clean_display_text(item.get("check_point")), 180)
                or fallback_check_point(label),
                "evidence_refs": refs,
                "source_urls": [],
                "confidence": clamp_confidence(item.get("confidence")),
                "evidence_summary": evidence_summary,
                "reasoning_summary": reasoning_summary,
                "insufficient_evidence": bool(item.get("insufficient_evidence")) or not refs,
            }
        )
    return normalized


def fallback_comparison_item(label: str, signals: list[dict[str, Any]]) -> dict[str, Any]:
    refs = choose_signal_refs_for_label(label, signals)
    object_name = infer_change_object(label, refs, signals)
    lacks_specific_object = object_name in GENERIC_CHANGE_OBJECT_BY_LABEL.values()
    if label == "사업 신호":
        body = (
            f"{object_name} 수준의 사업 움직임만 확인되며, 구체 사업명은 추가 근거 확인이 필요합니다."
            if lacks_specific_object
            else f"{object_name} 관련 사업 움직임이 최근 근거에서 확인됩니다."
        )
    elif label == "기술 신호":
        body = (
            f"{object_name} 수준의 기술 움직임만 확인되며, 구체 플랫폼명은 추가 근거 확인이 필요합니다."
            if lacks_specific_object
            else f"{object_name} 관련 기술 적용 또는 플랫폼화 움직임이 확인됩니다."
        )
    else:
        body = (
            f"{object_name} 수준의 리스크만 확인되며, 구체 원인은 추가 근거 확인이 필요합니다."
            if lacks_specific_object
            else f"{object_name} 관련 실행 부담이나 시장 불확실성을 함께 점검해야 합니다."
        )
    return {
        "label": label,
        "change_object": object_name,
        "body": body,
        "evidence_refs": refs,
        "confidence": 0.5 if refs else 0.2,
        "evidence_summary": "입력 근거에서 확인 가능한 최신 신호를 기준으로 보수적으로 작성했습니다.",
        "reasoning_summary": "핵심 비교 포인트는 최근 관찰된 변화만 요약했습니다.",
        "insufficient_evidence": not refs or lacks_specific_object,
    }


def fallback_swot_item(label: str, diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    refs = choose_diagnostic_refs_for_label(label, diagnostics)
    title = fallback_swot_title(label, diagnostics)
    diagnosis = diagnostic_for_label(label, diagnostics)
    bodies = {
        "Strength": f"{title}은 내부적으로 경쟁사 대비 활용 가능한 역량입니다.",
        "Weakness": f"{title}은 내부 개선 과제로 남아 있어 실행 품질과 수익성 관리가 필요합니다.",
        "Opportunity": f"{title}은 외부 시장 변화가 유리하게 작용할 수 있는 지점입니다.",
        "Threat": f"{title}은 외부 환경 변화가 사업 성과를 압박할 수 있는 지점입니다.",
    }
    return {
        "label": label,
        "title": title,
        "body": bodies[label],
        "factor_type": SWOT_FACTOR_TYPE_BY_LABEL[label],
        "check_point": fallback_check_point(label),
        "evidence_refs": refs,
        "confidence": 0.45 if refs else 0.2,
        "evidence_summary": build_swot_evidence_summary(diagnosis),
        "reasoning_summary": build_swot_reasoning_summary(label, title, diagnosis),
        "insufficient_evidence": not refs,
    }


def build_comparison_evidence_summary(
    label: str, refs: list[str], signals: list[dict[str, Any]]
) -> str:
    by_ref = {item.get("evidence_id"): item for item in signals}
    picked = [by_ref[ref] for ref in refs if ref in by_ref]
    if not picked:
        return "출처: 해당 판단에 직접 연결되는 공개 원문 출처가 부족합니다."
    citations = [build_signal_source_citation(item) for item in picked[:2]]
    citations = [citation for citation in citations if citation]
    if not citations:
        return "출처: 원문 제목·발행일·URL 정보가 제한적입니다."
    return compact_text("출처: " + " | ".join(citations), 260)


def build_comparison_reasoning_summary(
    label: str,
    change_object: str,
    refs: list[str],
    signals: list[dict[str, Any]],
    *,
    evidence_pack: dict[str, Any] | None = None,
) -> str:
    by_ref = {item.get("evidence_id"): item for item in signals}
    picked = [by_ref[ref] for ref in refs if ref in by_ref]
    checked = (
        compact_text(
            " / ".join(
                str(value or "")
                for item in picked[:1]
                for value in (item.get("business_area"), item.get("summary"))
            ),
            110,
        )
        or change_object
    )
    lens = {
        "사업 신호": "사업·고객·서비스 실행 움직임",
        "기술 신호": "기술·플랫폼·제품 적용 움직임",
        "리스크": "실행·시장·경쟁·규제상 최근 부담",
    }[label]
    meaning = {
        "사업 신호": "최근 사업 방향이나 고객 접점이 이동하고 있다는 의미",
        "기술 신호": "기술 적용 영역이나 플랫폼화 방향이 드러난다는 의미",
        "리스크": "현재 실행 과정에서 관리해야 할 불확실성이 드러난다는 의미",
    }[label]
    coverage_note = build_overall_reasoning_coverage_note(evidence_pack, refs)
    return compact_text(
        f"추론 과정: '{checked}' 내용을 확인했고, 이는 {meaning}입니다. {coverage_note}그래서 '{change_object}' 항목을 {lens}으로 판단했습니다.",
        280,
    )


def build_swot_evidence_summary(diagnosis: dict[str, Any]) -> str:
    citations = diagnosis.get("source_citations") if isinstance(diagnosis, dict) else None
    if isinstance(citations, list) and citations:
        return compact_text("출처: " + " | ".join(str(citation) for citation in citations[:2]), 280)
    return "출처: 비교 포인트와 독립적으로 사용할 수 있는 공개 원문 출처가 부족합니다."


def build_swot_reasoning_summary(label: str, title: str, diagnosis: dict[str, Any]) -> str:
    question = compact_text(
        diagnosis.get("diagnostic_question") if isinstance(diagnosis, dict) else "", 90
    )
    basis = compact_text(
        diagnosis.get("judgment_basis") if isinstance(diagnosis, dict) else "", 110
    )
    basis_points = diagnosis.get("basis_points") if isinstance(diagnosis, dict) else None
    checked = (
        compact_text(str(basis_points[0]), 120)
        if isinstance(basis_points, list) and basis_points
        else title
    )
    label_lens = {
        "Strength": "내부 통제 가능한 경쟁 역량",
        "Weakness": "내부에서 개선해야 할 제약",
        "Opportunity": "외부에서 유리하게 열린 조건",
        "Threat": "외부에서 불리하게 작용할 압박",
    }[label]
    meaning = basis or label_lens
    return compact_text(
        f"추론 과정: '{checked}' 내용을 확인했고, 이는 {meaning} 기준에 해당합니다. 그래서 '{question or title}' 관점에서 {label_lens}으로 판단했습니다.",
        300,
    )


def separate_swot_body_from_comparison(
    label: str,
    body: str,
    title: str,
    evidence_pack: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> str:
    comparison_items = evidence_pack.get("_normalized_comparison_points") or []
    if not comparison_items:
        return body
    comparison_texts = [
        " ".join(str(item.get(key) or "") for key in ("label", "change_object", "body"))
        for item in comparison_items
        if isinstance(item, dict)
    ]
    candidate = f"{title} {body}"
    if not any(is_semantically_close(candidate, text_value) for text_value in comparison_texts):
        return body
    diagnosis = diagnostic_for_label(label, diagnostics)
    return build_diagnostic_swot_body(label, title, diagnosis)


def build_diagnostic_swot_body(label: str, title: str, diagnosis: dict[str, Any]) -> str:
    summaries = diagnosis.get("signal_summaries") if isinstance(diagnosis, dict) else None
    basis_points = diagnosis.get("basis_points") if isinstance(diagnosis, dict) else None
    profile_hint = compact_text(
        diagnosis.get("profile_hint") if isinstance(diagnosis, dict) else "", 120
    )
    checked = ""
    if isinstance(summaries, list) and summaries:
        checked = compact_text(str(summaries[0]), 130)
    elif isinstance(basis_points, list) and basis_points:
        checked = compact_text(str(basis_points[0]), 130)
    else:
        checked = profile_hint or title

    object_name = extract_specific_phrase(
        " ".join(str(item) for item in (summaries or [])) or profile_hint or title
    )
    if is_generic_change_object(object_name):
        object_name = title

    bodies = {
        "Strength": (
            f"근거에서는 {checked} 내용이 확인됩니다. "
            f"{object_name} 역량은 서비스·플랫폼 운영에 반복 적용할 수 있는 내부 자산이므로 강점으로 판단합니다."
        ),
        "Weakness": (
            f"근거에서는 {checked} 내용이 확인됩니다. "
            f"{object_name} 이슈는 비용·일정·인력 또는 수익성 측면에서 내부 관리가 필요한 제약이므로 약점으로 판단합니다."
        ),
        "Opportunity": (
            f"근거에서는 {checked} 내용이 확인됩니다. "
            f"{object_name} 흐름은 회사가 직접 만든 변화는 아니지만 시장·고객 투자 확대를 활용할 수 있는 기회로 판단합니다."
        ),
        "Threat": (
            f"근거에서는 {checked} 내용이 확인됩니다. "
            f"{object_name} 압박은 경쟁·규제·수요 변화처럼 직접 통제하기 어려운 외부 조건이므로 위협으로 판단합니다."
        ),
    }
    return compact_text(bodies[label], 430)


def swot_text_overlaps_comparison(value: str, evidence_pack: dict[str, Any]) -> bool:
    comparison_items = evidence_pack.get("_normalized_comparison_points") or []
    comparison_texts = [
        " ".join(str(item.get(key) or "") for key in ("change_object", "body"))
        for item in comparison_items
        if isinstance(item, dict)
    ]
    return content_overlaps_comparison(value, comparison_texts)
