"""mixer radar — extracted from facade (move-only)."""

from __future__ import annotations

import json
import logging

from src.agents.mixer._constants import (  # noqa: F401
    _DEEP_LLM_MODEL,
    _FOLLOW_UP_INVALID_SUBJECTS,
    _INCOMPLETE_KOREAN_ENDINGS,
    _LEGACY_RESULT_GROUP_KEY,
    _LINKED_RESULT_KEYS,
    _LLM_MODEL,
    _MAX_CARDS,
    _MIN_CARDS,
    _MIXER_FINAL_ONE_LINER_MAX,
    _MIXER_IMPLICATION_MAX,
    _MIXER_PROMPT,
    _MIXER_RADAR_AXIS_FILL_PROMPT,
    _MIXER_REPAIR_PROMPT,
    _POLITE_ENDING_REPLACEMENTS,
    _PROGRESS_ORDER,
    _PROGRESS_STAGES,
    _PROGRESS_TOTAL,
    _PROMPT_VERSION,
    _QUICK_LLM_MODEL,
    _RADAR_AXIS_ALIASES,
    _RADAR_AXIS_LABELS,
    _RADAR_AXIS_ORDER,
    _RADAR_AXIS_PROMPTS,
    ProgressFn,
)
from src.agents.mixer.sentence_normalizer import (  # noqa: F401
    _augment_deep_action_details,
    _avg,
    _block_evidence_facts,
    _brief_action_basis,
    _card_one_liner,
    _card_score,
    _card_sector,
    _cards_for_event,
    _cards_for_evidence_refs,
    _cards_for_sector,
    _cards_from_linked_result_items,
    _clean_follow_up_subject,
    _clean_mixer_sentence_text,
    _clip_to_complete_sentence,
    _compact_analysis_result,
    _compact_integrated_issue,
    _compact_json,
    _complete_sentence_parts,
    _component,
    _connections_from_blocks,
    _dedupe_ints,
    _dedupe_keep_order,
    _dict_or_empty,
    _distinctive_tokens,
    _emit_progress,
    _error_response,
    _fallback_action_details_from_result,
    _fetch_cards,
    _findings_from_blocks,
    _format_analysis_units,
    _format_cards,
    _format_ratios,
    _generate_mix_level_implication,
    _get_langfuse_trace_id,
    _get_llm,
    _grounding_text_for_actions,
    _has_hangul_final_consonant,
    _int_list,
    _is_action_grounded,
    _is_complete_display_sentence,
    _is_generic_action_text,
    _json_list,
    _linked_result_event_type,
    _linked_result_title,
    _linked_results_from_card,
    _linked_results_from_item,
    _linked_results_from_sources,
    _llm_max_completion_tokens,
    _looks_incomplete_display_sentence,
    _mix_analysis_result,
    _mix_classification,
    _mix_input_bundle,
    _mix_integrated_issue,
    _mix_profile_context,
    _mixer_result_for_repair,
    _mixer_sentence_quality_issues,
    _model_for_mode,
    _needs_action_detail_fallback,
    _new_mix_id,
    _normalize_action_details,
    _normalize_analysis_mode,
    _normalize_follow_up_subject_terms,
    _normalize_mix_block,
    _normalize_mix_evidence,
    _normalize_mixer_display_sentence,
    _normalize_mixer_result_display_sentences,
    _parse_and_validate,
    _recommended_actions_from_basis,
    _recommended_actions_from_details,
    _recommended_actions_from_implication,
    _repair_mixer_result_quality,
    _score_for_event,
    _score_for_sector,
    _sentence_base,
    _sources_from_cards,
    _subject_with_particle,
    _to_polite_display_sentence,
    _valid_card_refs,
    _valid_connections,
    _valid_cross_card_findings,
    _valid_reasoning_step_refs,
    _valid_reasoning_trail_refs,
    _warning_for,
    clip_final_one_liner,
    clip_implication,
    clip_string,
)
from src.observability.langfuse_client import tracing_config

log = logging.getLogger(__name__)


def _radar_axis(
    *,
    axis: str,
    score: float,
    explanation: str,
    calculation: str,
    matched_cards: list[dict],
    total_count: int,
) -> dict:
    rounded_score = round(score, 3)
    return {
        "axis": axis,
        "score": rounded_score,
        "explanation": explanation,
        "calculation": calculation,
        "meaning": _radar_score_meaning(rounded_score),
        "support_count": len(matched_cards),
        "total_count": total_count,
        "matched_card_ids": [
            str(card.get("id")) for card in matched_cards if str(card.get("id") or "").strip()
        ],
    }


def _radar_score_meaning(score: float) -> str:
    if score >= 0.7:
        return "선택한 카드 묶음에서 강한 판단 신호로 볼 수 있습니다."
    if score >= 0.35:
        return "일부 카드가 해당 축을 지지하므로 보조 판단 신호로 봅니다."
    if score > 0:
        return "근거는 있으나 선택 묶음 전체를 대표할 정도는 아닙니다."
    return "이번 선택 묶음에서는 이 축을 직접 지지하는 카드가 확인되지 않았습니다."


def _peer_diversity_score(cards: list[dict]) -> float:
    peers = {c.get("peer_id") for c in cards if c.get("peer_id")}
    # 4 국내 peer 기준, 다양성 정규화 (1 peer=0.25, 4 peer=1.0).
    return min(len(peers) / 4.0, 1.0)


def _compute_radar(cards: list[dict]) -> list[dict]:
    total_count = len(cards)
    strategic_cards = _cards_for_event(cards, {"ma", "new_biz"})
    tech_cards = _cards_for_sector(cards, {"ax", "ai_tech", "infra"})
    partnership_cards = _cards_for_event(cards, {"partnership"})
    regulation_cards = _cards_for_event(cards, {"regulation"})
    talent_cards = _cards_for_event(cards, {"personnel"})
    peer_cards = [card for card in cards if card.get("peer_id")]
    axes = {
        "peer_strategic_shift": _radar_axis(
            axis="peer_strategic_shift",
            score=_score_for_event(cards, {"ma", "new_biz"}),
            explanation="M&A·신사업처럼 전략 전환 성격으로 분류된 카드의 exposure 평균입니다.",
            calculation="event_type이 ma 또는 new_biz인 카드만 골라 exposure_score를 평균했습니다.",
            matched_cards=strategic_cards,
            total_count=total_count,
        ),
        "tech_investment": _radar_axis(
            axis="tech_investment",
            score=_score_for_sector(cards, {"ax", "ai_tech", "infra"}),
            explanation=(
                "AX·AI 기술·인프라 섹터 카드가 기술 투자 판단을 얼마나 지지하는지 본 값입니다."
            ),
            calculation="sector가 ax, ai_tech, infra인 카드의 exposure_score를 평균했습니다.",
            matched_cards=tech_cards,
            total_count=total_count,
        ),
        "market_position": _radar_axis(
            axis="market_position",
            score=_peer_diversity_score(cards),
            explanation=(
                "선택 묶음이 특정 peer 한 곳의 소식인지, "
                "여러 peer에 걸친 시장 신호인지 보는 값입니다."
            ),
            calculation="서로 다른 peer 수를 국내 주요 peer 4개 기준으로 나눠 정규화했습니다.",
            matched_cards=peer_cards,
            total_count=total_count,
        ),
        "partnership_momentum": _radar_axis(
            axis="partnership_momentum",
            score=_score_for_event(cards, {"partnership"}),
            explanation="제휴·협력 이벤트가 선택 묶음의 핵심 동력인지 보는 값입니다.",
            calculation="event_type이 partnership인 카드의 exposure_score를 평균했습니다.",
            matched_cards=partnership_cards,
            total_count=total_count,
        ),
        "regulatory_risk": _radar_axis(
            axis="regulatory_risk",
            score=_score_for_event(cards, {"regulation"}),
            explanation="규제·정책 이벤트가 의사결정 리스크로 작동하는지 보는 값입니다.",
            calculation="event_type이 regulation인 카드의 exposure_score를 평균했습니다.",
            matched_cards=regulation_cards,
            total_count=total_count,
        ),
        "talent_movement": _radar_axis(
            axis="talent_movement",
            score=_score_for_event(cards, {"personnel"}),
            explanation="조직·인력 변화가 선택 묶음의 실행 역량 신호인지 보는 값입니다.",
            calculation="event_type이 personnel인 카드의 exposure_score를 평균했습니다.",
            matched_cards=talent_cards,
            total_count=total_count,
        ),
    }
    return [axes[axis] for axis in _RADAR_AXIS_ORDER]


def _radar_axis_analysis_prompt(axis: str) -> str:
    return _RADAR_AXIS_PROMPTS.get(axis, "본문 근거가 이 신호 축을 어떻게 지지하거나 약화하는가?")


def _normalize_radar_axis_key(value: object) -> str:
    key = str(value or "").strip()
    if not key:
        return ""
    if key in _RADAR_AXIS_ORDER:
        return key
    normalized = key.lower().replace("-", "_").replace(" ", "_")
    if normalized in _RADAR_AXIS_ORDER:
        return normalized
    return _RADAR_AXIS_ALIASES.get(key) or _RADAR_AXIS_ALIASES.get(normalized) or ""


def _merge_radar_axis_interpretations(radar: list[dict], interpretations: object) -> list[dict]:
    by_axis: dict[str, dict[str, object]] = {}
    for item in _json_list(interpretations):
        if not isinstance(item, dict):
            continue
        axis_key = _normalize_radar_axis_key(item.get("axis"))
        if axis_key:
            by_axis[axis_key] = item

    enriched: list[dict] = []
    for radar_axis in radar:
        axis_id = str(radar_axis.get("axis") or "").strip()
        interpretation = by_axis.get(axis_id, {})
        next_axis = dict(radar_axis)
        next_axis["analysis_prompt"] = clip_string(
            str(
                interpretation.get("analysis_prompt")
                or interpretation.get("prompt")
                or _radar_axis_analysis_prompt(axis_id)
            ).strip(),
            220,
        )
        interpretation_text = str(
            interpretation.get("interpretation")
            or interpretation.get("prompted_interpretation")
            or interpretation.get("answer")
            or ""
        ).strip()
        next_axis["prompted_interpretation"] = clip_string(interpretation_text, 420)
        enriched.append(next_axis)
    return enriched


def _fill_missing_radar_axis_interpretations(
    radar_axes: list[dict],
    cards: list[dict],
    analysis_mode: str,
) -> list[dict]:
    missing_axes = [
        axis for axis in radar_axes if not str(axis.get("prompted_interpretation") or "").strip()
    ]
    if not missing_axes:
        return radar_axes

    prompt = _MIXER_RADAR_AXIS_FILL_PROMPT.replace(
        "{context}", _format_analysis_units(cards)
    ).replace("{missing_axes}", _format_missing_radar_axes(missing_axes, cards))
    try:
        response = _get_llm(analysis_mode).invoke(
            prompt,
            config=tracing_config(
                agent="MixerAnalysisAgent",
                phase="radar_axis_fill",
                prompt_version=f"{_PROMPT_VERSION}-radar-fill",
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:  # noqa: BLE001
        log.warning("Mixer radar axis LLM 보강 실패 | error=%s", exc)
        return radar_axes

    interpretations = _parse_radar_axis_interpretation_response(content)
    if not interpretations:
        log.warning("Mixer radar axis LLM 보강 응답 파싱 실패 | content=%s", content[:200])
        return radar_axes

    filled_axes = _merge_radar_axis_interpretations(missing_axes, interpretations)
    filled_by_axis = {
        str(axis.get("axis") or ""): axis
        for axis in filled_axes
        if str(axis.get("prompted_interpretation") or "").strip()
    }
    return [
        filled_by_axis.get(str(axis.get("axis") or ""), axis)
        if not str(axis.get("prompted_interpretation") or "").strip()
        else axis
        for axis in radar_axes
    ]


def _parse_radar_axis_interpretation_response(content: str) -> list:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
    if isinstance(parsed, dict):
        return _json_list(
            parsed.get("radar_axis_interpretations")
            or parsed.get("interpretations")
            or parsed.get("axes")
        )
    if isinstance(parsed, list):
        return parsed
    return []


def _format_missing_radar_axes(axes: list[dict], cards: list[dict]) -> str:
    card_lookup = {str(card.get("id") or ""): card for card in cards}
    blocks: list[str] = []
    for axis in axes:
        axis_id = str(axis.get("axis") or "")
        label = _RADAR_AXIS_LABELS.get(axis_id, axis_id)
        support_count = int(axis.get("support_count") or 0)
        total_count = int(axis.get("total_count") or 0)
        score = float(axis.get("score") or 0.0)
        matched_ids = [str(card_id) for card_id in _json_list(axis.get("matched_card_ids"))]
        evidence_lines: list[str] = []
        for card_id in matched_ids[:6]:
            card = card_lookup.get(card_id, {})
            title = str(card.get("title") or card_id).strip()
            company = str(card.get("company") or card.get("peer_id") or "").strip()
            prefix = f"{company}: " if company else ""
            evidence_lines.append(f"  - {card_id}: {prefix}{title}")
        if not evidence_lines:
            evidence_lines.append("  - 직접 근거 카드 없음")
        blocks.append(
            "\n".join(
                [
                    f"- axis: {axis_id}",
                    f"  label: {label}",
                    f"  score: {score:.2f}",
                    f"  support_count: {support_count}/{total_count}",
                    f"  analysis_prompt: {_radar_axis_analysis_prompt(axis_id)}",
                    f"  explanation: {axis.get('explanation') or ''}",
                    "  matched_cards:",
                    *evidence_lines,
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_radar(radar: list[dict]) -> str:
    """6축 radar 결과를 LLM 입력 prompt 용 한국어 라인으로 포맷.

    v2: 결정적 산식 결과를 LLM 에 anchor 로 제공. LLM 이 reasoning 에 활용.
    """
    if not radar:
        return "*radar 점수 산출 불가*"
    lines: list[str] = []
    for axis in radar:
        score = float(axis.get("score") or 0.0)
        bar = "▰" * int(score * 10) + "▱" * (10 - int(score * 10))
        label = _RADAR_AXIS_LABELS.get(axis["axis"], axis["axis"])
        explanation = axis.get("explanation", "")
        lines.append(f"- {label}: {bar} {score:.2f} ({explanation})")
    return "\n".join(lines)
