"""mixer sentence_normalizer — extracted from facade (move-only)."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from src.agents.implication_agent import ImplicationAgent
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
from src.llm import LLMSpec, build_chat_llm
from src.observability.langfuse_client import tracing_config
from src.services.agent_output_validation import (
    confidence_in_range,
)
from src.services.analysis_units import (
    card_like_from_units,
    load_analysis_units_by_card_ids,
)
from src.services.llm_env import (
    ensure_llm_env_loaded,
)
from src.shared.json_helpers import json_dict as _json_dict

log = logging.getLogger(__name__)


def clip_string(value: Any, max_length: int, *, suffix: str = "") -> str:
    """Keep mixer text within display bounds without cutting a sentence mid-way."""
    del suffix
    if not isinstance(value, str):
        return ""
    text = _clean_mixer_sentence_text(value)
    if len(text) <= max_length:
        return text
    return _clip_to_complete_sentence(text, max_length)


def clip_final_one_liner(value: Any, *, max_length: int = _MIXER_FINAL_ONE_LINER_MAX) -> str:
    return clip_string(value, max_length)


def clip_implication(value: Any, *, max_length: int = _MIXER_IMPLICATION_MAX) -> str:
    return clip_string(value, max_length)


def _clean_mixer_sentence_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("…", "").replace("...", "").replace("..", ".").replace(" .", ".").strip()


def _sentence_base(value: str) -> str:
    return value.strip().rstrip(".!?。").strip()


def _looks_incomplete_display_sentence(value: str) -> bool:
    base = _sentence_base(value)
    if not base:
        return True
    return any(base.endswith(ending) for ending in _INCOMPLETE_KOREAN_ENDINGS)


def _to_polite_display_sentence(value: str) -> str:
    base = _sentence_base(value)
    if not base:
        return ""
    if _looks_incomplete_display_sentence(base):
        return ""
    for informal, polite in _POLITE_ENDING_REPLACEMENTS:
        if base.endswith(informal):
            base = f"{base[: -len(informal)]}{polite}"
            break
    if not re.search(r"(습니다|합니다|됩니다|입니다|니다|요)$", base):
        return ""
    return f"{base}."


def _is_complete_display_sentence(value: str) -> bool:
    normalized = _to_polite_display_sentence(value)
    if not normalized:
        return False
    base = _sentence_base(normalized)
    return bool(re.search(r"(습니다|합니다|됩니다|입니다|니다|요|다)$", base))


def _complete_sentence_parts(value: str) -> list[str]:
    text = _clean_mixer_sentence_text(value)
    if not text:
        return []
    matches = re.findall(r"[^.!?。]+[.!?。]", text)
    if not matches and _is_complete_display_sentence(text):
        matches = [text]
    return [normalized for part in matches if (normalized := _to_polite_display_sentence(part))]


def _clip_to_complete_sentence(value: str, max_length: int) -> str:
    parts = _complete_sentence_parts(value)
    if not parts:
        return _clean_mixer_sentence_text(value).strip()
    kept: list[str] = []
    for part in parts:
        candidate = " ".join([*kept, part]).strip()
        if len(candidate) > max_length:
            break
        kept.append(part)
    if kept:
        return " ".join(kept).strip()
    return parts[0]


def _normalize_mixer_display_sentence(value: object, max_length: int) -> str:
    text = _clean_mixer_sentence_text(value)
    if not text:
        return ""
    parts = _complete_sentence_parts(text)
    if parts:
        text = " ".join(parts)
    if len(text) > max_length:
        text = _clip_to_complete_sentence(text, max_length)
    return _to_polite_display_sentence(text)


def _emit_progress(progress: "ProgressFn | None", stage: str) -> None:
    """단계 경계에서 progress 콜백 호출 (None 이면 no-op, 예외는 분석을 막지 않음)."""
    if progress is None:
        return
    try:
        index = _PROGRESS_ORDER.index(stage)
    except ValueError:
        index = 0
    try:
        progress(stage, _PROGRESS_STAGES.get(stage, stage), index, _PROGRESS_TOTAL)
    except Exception:
        log.debug("mixer progress 콜백 실패 (무시)", exc_info=True)


def _model_for_mode(analysis_mode: object = "quick") -> str:
    if _normalize_analysis_mode(analysis_mode) == "deep":
        return _DEEP_LLM_MODEL
    return _QUICK_LLM_MODEL


if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

_llms: dict[str, ChatOpenAI] = {}


def _llm_max_completion_tokens(model: str) -> int:
    default = "9000" if str(model).startswith("gpt-5") else "3000"
    return int(os.getenv("MIXER_MAX_COMPLETION_TOKENS", default))


def _get_llm(analysis_mode: object = "quick") -> ChatOpenAI:
    model = _model_for_mode(analysis_mode)
    if model not in _llms:
        ensure_llm_env_loaded()
        # gpt-5 reasoning_effort 분기·json_object 래핑은 공용 팩토리가 처리.
        # 모델별 _llms 캐시는 그대로 유지(quick/deep 분리).
        _llms[model] = build_chat_llm(
            LLMSpec(
                model=model,
                temperature=0.15,
                max_tokens=_llm_max_completion_tokens(model),
                json_object=True,
                reasoning_effort=os.getenv("MIXER_DEEP_REASONING_EFFORT", "medium"),
            )
        )
    return _llms[model]


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _normalize_analysis_mode(value: object) -> str:
    return "deep" if str(value or "").strip().lower() == "deep" else "quick"


def _avg(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def _cards_for_event(cards: list[dict], event_types: set[str]) -> list[dict]:
    return [card for card in cards if (card.get("event_type") or "") in event_types]


def _score_for_event(cards: list[dict], event_types: set[str]) -> float:
    return _avg([_card_score(c) for c in _cards_for_event(cards, event_types)])


def _cards_for_sector(cards: list[dict], sectors: set[str]) -> list[dict]:
    return [card for card in cards if _card_sector(card) in sectors]


def _score_for_sector(cards: list[dict], sectors: set[str]) -> float:
    return _avg([_card_score(c) for c in _cards_for_sector(cards, sectors)])


def _card_score(card: dict) -> float:
    impl = _dict_or_empty(card.get("implication"))
    sector_meta = _dict_or_empty(impl.get("sector_meta"))
    exposure = impl.get("exposure_score") or sector_meta.get("exposure_score")
    if exposure is None:
        exposure = card.get("importance_score")
    try:
        return float(exposure or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _card_sector(card: dict) -> str:
    impl = _dict_or_empty(card.get("implication"))
    sector_meta = _dict_or_empty(impl.get("sector_meta"))
    return (
        impl.get("sector")
        or sector_meta.get("sector")
        or card.get("primary_keyword_category")
        or card.get("sector")
        or "other"
    ).lower()


def _new_mix_id() -> str:
    return f"mix-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _format_ratios(ratios: dict | None) -> str:
    if not ratios:
        return "*비율 미지정 — 기본 균등*"
    parts: list[str] = []
    for key in ("peer", "industry", "keyword"):
        value = ratios.get(key)
        if value:
            parts.append(f"- **{key}**: {value}")
    return "\n".join(parts) if parts else "*비율 미지정 — 기본 균등*"


def _fetch_cards(card_ids: list[str]) -> list[dict]:
    return card_like_from_units(load_analysis_units_by_card_ids(card_ids))


def _cards_from_linked_result_items(items: list[dict]) -> list[dict]:
    """런타임 linked result 입력을 Mixer 가 쓰는 card-like dict 로 정규화한다."""
    cards: list[dict] = []
    for index, item in enumerate(items or [], start=1):
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id") or item.get("id") or "").strip()
        if not card_id:
            log.warning(
                "MixerAnalysisAgent linked result item without card_id skipped | index=%s",
                index,
            )
            continue
        linked_results = _linked_results_from_item(item)
        implication = item.get("implication")
        if not isinstance(implication, dict):
            implication = linked_results.get("implication") if linked_results else {}
        if not isinstance(implication, dict):
            implication = {}
        evidence_payload = _json_dict(item.get("evidence_payload"))
        evidence_payload.update({key: value for key, value in linked_results.items() if value})
        cards.append(
            {
                "id": card_id,
                "card_id": card_id,
                "company": item.get("company") or item.get("main_company") or "",
                "peer_id": (
                    item.get("peer_id") or item.get("company") or item.get("main_company") or ""
                ),
                "primary_keyword_category": item.get("primary_keyword_category"),
                "source_raw_article_ids": _int_list(item.get("source_raw_article_ids")),
                "title": item.get("title") or _linked_result_title(linked_results),
                "summary_lines": item.get("summary_lines") or item.get("fact_summary") or [],
                "event_type": item.get("event_type") or _linked_result_event_type(linked_results),
                "importance": item.get("importance") or item.get("importance_level") or "medium",
                "importance_score": item.get("importance_score") or 0.0,
                "implication": implication,
                "sources": _json_list(item.get("sources")),
                "evidence_payload": evidence_payload,
                "validation_pass": item.get("validation_pass", True),
                "validation_sc_score": item.get("validation_sc_score", 0.0),
            }
        )
    return cards


def _linked_results_from_item(item: dict) -> dict:
    evidence_payload = _json_dict(item.get("evidence_payload"))
    return _linked_results_from_sources(item, evidence_payload)


def _linked_results_from_sources(*sources: dict) -> dict:
    collected: dict[str, object] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        legacy_group = source.get(_LEGACY_RESULT_GROUP_KEY)
        if isinstance(legacy_group, dict):
            for key in _LINKED_RESULT_KEYS:
                if isinstance(legacy_group.get(key), dict) and key not in collected:
                    collected[key] = legacy_group[key]
        for key in _LINKED_RESULT_KEYS:
            if isinstance(source.get(key), dict):
                collected[key] = source[key]
    return collected


def _linked_result_title(linked_results: dict) -> str:
    for key in ("integrated_issue", "analysis"):
        value = linked_results.get(key)
        if isinstance(value, dict):
            title = value.get("title") or value.get("main_issue") or value.get("headline")
            if title:
                return str(title)
    return ""


def _linked_result_event_type(linked_results: dict) -> str:
    for section_name in ("classification", "integrated_issue"):
        section = linked_results.get(section_name)
        if isinstance(section, dict):
            event_type = section.get("event_type") or section.get("cluster_event_type")
            if event_type:
                return str(event_type)
    return ""


def _linked_results_from_card(card: dict) -> dict:
    evidence_payload = _json_dict(card.get("evidence_payload"))
    return _linked_results_from_sources(card, evidence_payload)


def _component(linked_results: dict, key: str) -> dict:
    value = linked_results.get(key)
    return value if isinstance(value, dict) else {}


def _format_analysis_units(cards: list[dict]) -> str:
    """선택된 카드에 연결된 통합/분석/시사점 결과를 LLM 입력으로 정리한다."""
    blocks: list[str] = []
    for c in cards:
        impl = c.get("implication") or {}
        sector = _card_sector(c)
        exposure_band = impl.get("exposure_band") or c.get("importance") or "low"
        summary_lines = c.get("summary_lines") or []
        if isinstance(summary_lines, str):
            try:
                summary_lines = json.loads(summary_lines)
            except json.JSONDecodeError:
                summary_lines = [summary_lines]
        summary = " / ".join(s for s in summary_lines if s)
        sector_meta = impl.get("sector_meta") if isinstance(impl.get("sector_meta"), dict) else {}
        evidence = _json_dict(c.get("evidence_payload"))
        linked_results = _linked_results_from_card(c)
        integrated_issue = _component(linked_results, "integrated_issue")
        analysis_result = _component(linked_results, "analysis")
        implication_result = _component(linked_results, "implication") or impl
        peer_impl = (
            implication_result.get("peer_implication")
            if isinstance(implication_result.get("peer_implication"), dict)
            else {}
        )
        skax_impl = (
            implication_result.get("skax_implication")
            if isinstance(implication_result.get("skax_implication"), dict)
            else {}
        )
        evidence_links = evidence.get("source_links") or c.get("sources") or []
        evidence_refs = evidence.get("evidence_refs") or []
        financial_refs = evidence.get("financial_refs") or []
        raw_ids = c.get("source_raw_article_ids") or []
        integrated_issue_id = (
            c.get("integrated_issue_id")
            or evidence.get("integrated_issue_id")
            or linked_results.get("integrated_issue_id")
        )
        quality_flags = c.get("quality_flags") or evidence.get("quality_flags") or []
        skax_payload = _compact_json(skax_impl) or _compact_json(implication_result)
        block = "\n".join(
            [
                f"[{c['id']}] {c.get('title', '')}",
                f"- Integrated issue id: {integrated_issue_id or '*없음*'}",
                f"- Peer: {c.get('peer_id') or c.get('company') or ''}",
                f"- Sector: {sector}",
                f"- Event type: {c.get('event_type', '')}",
                f"- Exposure: {exposure_band} ({_card_score(c):.2f})",
                f"- Source raw article ids: {raw_ids}",
                f"- Quality flags: {_compact_json(quality_flags) or '*없음*'}",
                f"- 통합 이슈: {_compact_json(_compact_integrated_issue(integrated_issue))}",
                f"- 전략 분석: {_compact_json(_compact_analysis_result(analysis_result))}",
                f"- Peer 분석: {_compact_json(peer_impl)}",
                f"- SK AX 시사점/대응: {skax_payload}",
                f"- Sector/signals: {_compact_json(sector_meta)}",
                f"- Evidence refs: {_compact_json(evidence_refs[:5])}",
                f"- 재무/수치 근거: {_compact_json(financial_refs)}",
                f"- 출처: {_compact_json(evidence_links[:5])}",
                f"- 표시 요약(최하위 보조): {summary}",
            ]
        )
        blocks.append(block)
    return "\n".join(blocks)


def _format_cards(cards: list[dict]) -> str:
    """Backward-compatible alias for older tests/imports."""
    return _format_analysis_units(cards)


def _json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
        return parsed if isinstance(parsed, list) else []
    return []


def _int_list(value: object) -> list[int]:
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[int] = []
    for item in values:
        if not isinstance(item, int | float | str | bytes | bytearray):
            continue
        try:
            number = int(item)
        except ValueError:
            continue
        if number > 0 and number not in result:
            result.append(number)
    return result


def _compact_json(value: object, *, limit: int = 900) -> str:
    if value in ({}, [], None, ""):
        return ""
    text_value = json.dumps(value, ensure_ascii=False, default=str)
    return text_value[:limit] + "..." if len(text_value) > limit else text_value


def _compact_integrated_issue(value: dict) -> dict:
    if not value:
        return {}
    return {
        "main_issue": value.get("main_issue") or value.get("headline"),
        "integrated_text": value.get("integrated_text"),
        "fact_summary": value.get("fact_summary"),
        "consolidated_facts": value.get("consolidated_facts", [])[:5],
        "key_numbers": value.get("key_numbers", [])[:5],
        "business_signals": value.get("business_signals", [])[:5],
        "main_company": value.get("main_company"),
        "event_type": value.get("cluster_event_type"),
    }


def _compact_analysis_result(value: dict) -> dict:
    if not value:
        return {}
    return {
        "analysis_summary": value.get("analysis_summary"),
        "strategic_moves": value.get("strategic_moves", [])[:5],
        "market_signals": value.get("market_signals", [])[:5],
        "competitive_meaning": value.get("competitive_meaning"),
        "risk_factors": value.get("risk_factors", [])[:5],
        "confidence": value.get("confidence"),
    }


def _valid_card_refs(value: object, allowed_card_ids: set[str]) -> list[str]:
    refs = [str(item) for item in _json_list(value) if str(item) in allowed_card_ids]
    result: list[str] = []
    for ref in refs:
        if ref not in result:
            result.append(ref)
    return result


def _valid_connections(value: object, allowed_card_ids: set[str]) -> list[dict]:
    connections: list[dict] = []
    allowed_labels = {"cause", "effect", "similar", "contrast", "reinforce"}
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_card_id") or "")
        target = str(item.get("target_card_id") or "")
        if source not in allowed_card_ids or target not in allowed_card_ids or source == target:
            continue
        connection = dict(item)
        connection["source_card_id"] = source
        connection["target_card_id"] = target
        try:
            weight = float(connection.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        connection["weight"] = max(0.0, min(weight, 1.0))
        if connection.get("label") not in allowed_labels:
            connection["label"] = "similar"
        connections.append(connection)
    return connections[:20]


def _valid_cross_card_findings(value: object, allowed_card_ids: set[str]) -> list[dict]:
    findings: list[dict] = []
    allowed_pattern_types = {
        "convergent_strategy",
        "divergent_strategy",
        "gap_in_market",
        "acceleration_signal",
        "timing_mismatch",
        "market_baseline",
    }
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        refs = _valid_card_refs(item.get("evidence_card_ids"), allowed_card_ids)
        if len(refs) < 2:
            continue
        finding = dict(item)
        finding["evidence_card_ids"] = refs
        finding["finding"] = _normalize_mixer_display_sentence(finding.get("finding", ""), 180)
        if finding.get("pattern_type") not in allowed_pattern_types:
            finding["pattern_type"] = "convergent_strategy"
        if finding["finding"]:
            findings.append(finding)
    return findings[:5]


def _normalize_mix_block(value: object, allowed_card_ids: set[str]) -> dict:
    if not isinstance(value, dict):
        value = {}
    evidence = _normalize_mix_evidence(value.get("evidence"), allowed_card_ids)
    refs = _valid_card_refs(value.get("evidence_card_ids"), allowed_card_ids)
    for item in evidence:
        card_id = item.get("card_id")
        if card_id and card_id not in refs:
            refs.append(card_id)
    return {
        "finding": _normalize_mixer_display_sentence(value.get("finding", ""), 260),
        "rationale": _normalize_mixer_display_sentence(value.get("rationale", ""), 320),
        "subject_terms": _normalize_follow_up_subject_terms(
            value.get("subject_terms") or value.get("subjects")
        ),
        "evidence": evidence[:4],
        "evidence_card_ids": refs[:6],
    }


def _normalize_action_details(value: object, allowed_card_ids: set[str]) -> list[dict]:
    details: list[dict] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        evidence = _normalize_mix_evidence(item.get("evidence"), allowed_card_ids)
        refs = _valid_card_refs(item.get("evidence_card_ids"), allowed_card_ids)
        for evidence_item in evidence:
            card_id = evidence_item.get("card_id")
            if card_id and card_id not in refs:
                refs.append(card_id)
        action = _normalize_mixer_display_sentence(
            item.get("action") or item.get("text") or "",
            420,
        )
        if not action:
            continue
        details.append(
            {
                "action": action,
                "why": _normalize_mixer_display_sentence(
                    item.get("why") or item.get("rationale") or "",
                    360,
                ),
                "use_case": clip_string(item.get("use_case") or "", 80),
                "evidence": evidence[:3],
                "evidence_card_ids": refs[:6],
            }
        )
    return details[:5]


def _recommended_actions_from_details(result: dict, *, limit: int = 3) -> list[str]:
    return _dedupe_keep_order(
        [
            clip_implication(item.get("action"))
            for item in _json_list(result.get("action_details"))
            if isinstance(item, dict) and item.get("action")
        ]
    )[:limit]


def _needs_action_detail_fallback(details: list[dict]) -> bool:
    if len(details) < 2:
        return True
    return any(
        not item.get("why")
        or len(_json_list(item.get("evidence_card_ids"))) < 2
        or _is_generic_action_text(str(item.get("action") or ""))
        for item in details
    )


def _is_generic_action_text(value: str) -> bool:
    text_value = str(value or "").strip()
    if not text_value:
        return True
    generic_patterns = (
        "다음 메시지를 배치",
        "다음 비교 축을 표로 추가",
        "후속 모니터링 항목을 다음 반복 신호",
        "제안서 첫 장",
        "기능 소개",
        "운영 검증표",
        "PoC 설계",
        "후속 모니터링",
        "뉴스 재요약",
        "제안 우선순위",
        "검증표",
        "이 프로그램",
        "대시보드",
        "화면",
        "강화한다",
        "검토한다",
        "모니터링한다",
    )
    if any(pattern in text_value for pattern in generic_patterns):
        return True
    concrete_markers = (
        "사업 라인",
        "고객군",
        "우선순위",
        "오퍼링",
        "파트너십",
        "투자",
        "조직",
        "책임 조직",
        "영업",
        "리스크",
        "거버넌스",
        "상품화",
        "패키지",
        "시장 대응",
        "의사결정",
        "자원 배분",
        "가격",
        "계약",
        "레퍼런스",
        "수익화",
        "규제",
    )
    return not any(marker in text_value for marker in concrete_markers)


def _fallback_action_details_from_result(result: dict) -> list[dict]:
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    action_basis = _json_list(result.get("recommended_action_basis"))
    common_evidence = _json_list(common.get("evidence"))
    comparison_evidence = _json_list(comparison.get("evidence"))
    hidden_evidence = _json_list(hidden.get("evidence"))
    common_refs = _json_list(common.get("evidence_card_ids"))
    comparison_refs = _json_list(comparison.get("evidence_card_ids"))
    hidden_refs = _json_list(hidden.get("evidence_card_ids"))

    return [
        {
            "action": clip_implication(
                "SK AX는 임원 의사결정에서 우선 공략 고객군과 책임 조직을 먼저 정한다. "
                f"{_brief_action_basis(hidden.get('finding') or action_basis[:1])}"
                " 이 판단을 기준으로 금융과 공공/교육 등 입력에서 확인된 고객군별 "
                "사업 우선순위를 나누고, "
                "각 고객군의 오퍼링 책임 조직과 리스크 승인 권한을 지정한다."
            ),
            "why": clip_string(
                hidden.get("rationale")
                or (
                    "여러 이슈를 함께 볼 때 단일 기능보다 "
                    "조합된 판단 기준이 더 중요하게 드러나기 때문이다."
                ),
                260,
            ),
            "use_case": "사업 우선순위",
            "evidence": hidden_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": _dedupe_keep_order(
                [str(item) for item in (hidden_refs or common_refs)]
            )[:6],
        },
        {
            "action": clip_implication(
                "SK AX는 사업 라인별 오퍼링 패키지를 같은 이름의 AX 상품으로 묶지 말고 "
                "고객 의사결정 기준에 맞춰 분리 상품화한다. "
                f"{_brief_action_basis(comparison.get('finding') or action_basis[1:2])}"
                " 이 차이를 기준으로 금융권은 규제·정산·보안 거버넌스 패키지, "
                "공공/교육은 데이터 비학습·권한 통제·운영 레퍼런스 패키지처럼 "
                "영업 우선순위와 가격/계약 조건을 다르게 둔다."
            ),
            "why": clip_string(
                comparison.get("rationale")
                or "같은 흐름 안에서도 이슈마다 앞세우는 적용 장면과 성과 기준이 다르기 때문이다.",
                260,
            ),
            "use_case": "오퍼링/상품화",
            "evidence": comparison_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": _dedupe_keep_order(
                [str(item) for item in (comparison_refs or common_refs)]
            )[:6],
        },
        {
            "action": clip_implication(
                "투자 규모, 수주 전환, 규제 일정, 운영 KPI 중 어떤 지표가 확인될 때 "
                "자원 배분을 바꿀지 의사결정 기준을 먼저 고정한다. "
                f"{_brief_action_basis(common.get('finding') or action_basis[2:3])}"
                " 이 반복 신호와 연결되는 지표가 확인되면 고객군별 전담 인력, "
                "파트너십 후보, 레퍼런스 확보 예산의 우선순위를 조정한다."
            ),
            "why": clip_string(
                common.get("rationale")
                or (
                    "여러 이슈에서 반복되는 움직임은 단순 관찰 대상이 아니라 "
                    "사업 자원 배분 기준으로 반영할 필요가 있기 때문이다."
                ),
                260,
            ),
            "use_case": "파트너십/시장 대응",
            "evidence": common_evidence[:3],
            "evidence_card_ids": _dedupe_keep_order([str(item) for item in common_refs])[:6],
        },
    ]


def _augment_deep_action_details(result: dict) -> list[dict]:
    """정확 분석 전용: LLM 액션을 보존하되 실행 판단 축을 최대 5개로 보강."""
    existing = [
        dict(item) for item in _json_list(result.get("action_details")) if isinstance(item, dict)
    ]
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    action_basis = _json_list(result.get("recommended_action_basis"))
    common_refs = _json_list(common.get("evidence_card_ids"))
    comparison_refs = _json_list(comparison.get("evidence_card_ids"))
    hidden_refs = _json_list(hidden.get("evidence_card_ids"))
    all_refs = _dedupe_keep_order(
        [
            *[str(ref) for ref in common_refs],
            *[str(ref) for ref in comparison_refs],
            *[str(ref) for ref in hidden_refs],
            *[str(ref) for ref in _json_list(result.get("sources_used"))],
        ]
    )

    def refs(*groups: list[object]) -> list[str]:
        flattened: list[str] = []
        for group in groups:
            flattened.extend(str(item) for item in group if str(item))
        return _dedupe_keep_order(flattened or all_refs)[:6]

    common_evidence = _json_list(common.get("evidence"))
    comparison_evidence = _json_list(comparison.get("evidence"))
    hidden_evidence = _json_list(hidden.get("evidence"))
    candidates = [
        {
            "action": clip_implication(
                "SK AX는 입력 근거에서 성과 기준이 가장 분명한 고객군을 우선 공략군으로 정하고 "
                "해당 고객군별 영업 책임 조직과 리스크 승인 권한을 먼저 배정한다. "
                f"{_brief_action_basis(hidden.get('finding') or action_basis[:1])}"
            ),
            "why": clip_string(
                hidden.get("rationale")
                or "여러 이슈를 함께 볼 때 고객군 선택 기준이 대응 방향의 출발점이 되기 때문이다.",
                260,
            ),
            "use_case": "고객군/영업전략",
            "evidence": hidden_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": refs(hidden_refs, common_refs),
        },
        {
            "action": clip_implication(
                "SK AX는 하나의 AX 메시지로 묶기보다 공통 패턴이 가리키는 성과 기준별로 "
                "오퍼링 패키지와 가격·계약 조건을 분리해 상품화한다. "
                f"{_brief_action_basis(common.get('finding') or action_basis[1:2])}"
            ),
            "why": clip_string(
                common.get("rationale")
                or (
                    "반복 신호가 단순 기술 관심이 아니라 구매 판단 기준으로 "
                    "연결될 수 있기 때문이다."
                ),
                260,
            ),
            "use_case": "오퍼링/상품화",
            "evidence": common_evidence[:3],
            "evidence_card_ids": refs(common_refs),
        },
        {
            "action": clip_implication(
                "SK AX는 카드별 접근 차이가 큰 영역은 동일한 투자 우선순위로 보지 말고 "
                "수주 전환 가능성, 운영 KPI, 적용 범위 중 확인된 지표를 자원 배분 게이트로 둔다. "
                f"{_brief_action_basis(comparison.get('finding') or action_basis[2:3])}"
            ),
            "why": clip_string(
                comparison.get("rationale")
                or "같은 흐름 안에서도 실제 적용 장면과 성과 기준이 다르게 나타나기 때문이다.",
                260,
            ),
            "use_case": "자원 배분",
            "evidence": comparison_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": refs(comparison_refs, common_refs),
        },
        {
            "action": clip_implication(
                "SK AX는 자체 역량만으로 레퍼런스 확보가 느린 영역을 따로 분리하고 "
                "적용 현장, 운영 데이터, 고객 접점을 보완할 파트너십 후보를 우선 검증한다. "
                f"{_brief_action_basis(comparison.get('rationale') or common.get('rationale'))}"
            ),
            "why": clip_string(
                "반복 신호가 실제 사업 전환으로 이어지려면 기술 보유보다 적용 현장과 "
                "고객 설득 근거가 먼저 확보되어야 합니다.",
                260,
            ),
            "use_case": "파트너십/시장 대응",
            "evidence": comparison_evidence[:3] or hidden_evidence[:3],
            "evidence_card_ids": refs(comparison_refs, hidden_refs),
        },
        {
            "action": clip_implication(
                "SK AX는 전망·추정·계획 성격의 근거와 확정 사실을 분리해 "
                "확정 투자, 영업 메시지, 리스크 공지를 서로 다른 의사결정 게이트로 관리한다. "
                f"{_brief_action_basis(hidden.get('rationale') or action_basis[3:4])}"
            ),
            "why": clip_string(
                "정확 분석에서는 반복 신호의 방향뿐 아니라 근거의 확정 수준까지 구분해야 "
                "과잉 투자와 과소 대응을 줄일 수 있기 때문입니다.",
                260,
            ),
            "use_case": "리스크/거버넌스",
            "evidence": hidden_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": refs(hidden_refs, common_refs),
        },
    ]
    merged: list[dict] = []
    seen_actions: set[str] = set()
    for item in [*existing, *candidates]:
        action = clip_implication(item.get("action") if isinstance(item, dict) else "")
        if not action or action in seen_actions:
            continue
        seen_actions.add(action)
        merged.append(
            {
                "action": action,
                "why": clip_string(item.get("why") or "", 260),
                "use_case": clip_string(item.get("use_case") or "실행 전략", 80),
                "evidence": _json_list(item.get("evidence"))[:3],
                "evidence_card_ids": refs(_json_list(item.get("evidence_card_ids"))),
            }
        )
    return merged[:5]


def _brief_action_basis(value: object) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if str(item).strip())
    text_value = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text_value:
        return "입력 근거에서 확인된 핵심 변화"
    brief = clip_string(text_value, 90).strip()
    if not brief.endswith((".", "!", "?", "…")):
        brief = f"{brief}."
    return brief


def _normalize_mix_evidence(value: object, allowed_card_ids: set[str]) -> list[dict]:
    evidence: list[dict] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id") or "")
        if card_id not in allowed_card_ids:
            continue
        text_value = clip_string(item.get("text") or item.get("basis") or "", 160)
        if not text_value:
            continue
        evidence.append({"card_id": card_id, "text": text_value})
    return evidence


def _findings_from_blocks(data: dict) -> list[dict]:
    mappings = (
        ("common_pattern", "convergent_strategy"),
        ("comparison_point", "divergent_strategy"),
        ("hidden_conclusion", "acceleration_signal"),
    )
    findings: list[dict] = []
    for key, pattern_type in mappings:
        block = data.get(key)
        if not isinstance(block, dict):
            continue
        finding = str(block.get("finding") or "").strip()
        refs = _json_list(block.get("evidence_card_ids"))
        if not finding or len(refs) < 2:
            continue
        findings.append(
            {
                "finding": finding,
                "evidence_card_ids": refs,
                "pattern_type": pattern_type,
            }
        )
    return findings


def _connections_from_blocks(data: dict) -> list[dict]:
    connections: list[dict] = []
    for block_key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = data.get(block_key)
        if not isinstance(block, dict):
            continue
        refs = [str(ref) for ref in _json_list(block.get("evidence_card_ids"))]
        if len(refs) < 2:
            continue
        connections.append(
            {
                "source_card_id": refs[0],
                "target_card_id": refs[1],
                "label": "similar" if block_key != "comparison_point" else "contrast",
                "weight": 0.8,
                "reason": clip_string(block.get("finding", ""), 60),
            }
        )
    return connections


def _valid_reasoning_trail_refs(value: object, allowed_card_ids: set[str]) -> list[dict]:
    trail: list[dict] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        updated["evidence_refs"] = _valid_card_refs(item.get("evidence_refs"), allowed_card_ids)
        trail.append(updated)
    return trail


def _valid_reasoning_step_refs(value: object, allowed_card_ids: set[str]) -> list[dict]:
    steps: list[dict] = []
    allowed_phases = {"per_card", "cross_card", "synthesis"}
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        updated["inputs_used"] = _valid_card_refs(item.get("inputs_used"), allowed_card_ids)
        if updated.get("phase") not in allowed_phases:
            updated["phase"] = "synthesis"
        steps.append(updated)
    return steps


def _parse_and_validate(
    content: str,
    cards: list[dict],
    card_ids: list[str],
) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("Mixer JSON parse 실패 — content prefix=%s", content[:200])
        return _error_response(
            "JSON parse 실패",
            "LLM 응답이 JSON 이 아님",
            card_ids,
            confidence=0.0,
        )

    if not isinstance(data, dict):
        return _error_response("응답 형식 오류", "JSON object 가 아님", card_ids, confidence=0.0)

    allowed_card_ids = {str(card["id"]) for card in cards}

    data["mix_insight"] = _normalize_mixer_display_sentence(
        data.get("mix_insight") or data.get("insight", ""),
        _MIXER_FINAL_ONE_LINER_MAX,
    )
    data["common_pattern"] = _normalize_mix_block(data.get("common_pattern"), allowed_card_ids)
    data["comparison_point"] = _normalize_mix_block(data.get("comparison_point"), allowed_card_ids)
    data["hidden_conclusion"] = _normalize_mix_block(
        data.get("hidden_conclusion"), allowed_card_ids
    )
    data["recommended_actions"] = [
        action
        for item in _json_list(data.get("recommended_actions"))
        if (action := _normalize_mixer_display_sentence(item, 420))
    ][:3]
    data["action_details"] = _normalize_action_details(data.get("action_details"), allowed_card_ids)
    fallback_actions_applied = False
    if _needs_action_detail_fallback(data["action_details"]):
        data["action_details"] = _fallback_action_details_from_result(data)
        fallback_actions_applied = True
    detail_actions = _recommended_actions_from_details(data)
    if (
        fallback_actions_applied
        or not data["recommended_actions"]
        or any(_is_generic_action_text(action) for action in data["recommended_actions"])
    ):
        data["recommended_actions"] = detail_actions or data["recommended_actions"]
    data["confidence"] = confidence_in_range(data.get("confidence", 0.0))
    data["sources_used"] = _valid_card_refs(
        data.get("sources_used") or [c["id"] for c in cards],
        allowed_card_ids,
    )
    if not data["sources_used"]:
        data["sources_used"] = [c["id"] for c in cards]

    # Backward-compatible fields for the existing API/UI.
    data["insight"] = data["mix_insight"]
    data["final_one_liner"] = _normalize_mixer_display_sentence(
        data["hidden_conclusion"].get("finding") or data["mix_insight"],
        _MIXER_FINAL_ONE_LINER_MAX,
    )
    data["sk_ax_implication"] = _normalize_mixer_display_sentence(
        " ".join(data["recommended_actions"]),
        _MIXER_IMPLICATION_MAX,
    )
    data["bullet_signals"] = [
        finding
        for finding in (
            data["common_pattern"].get("finding"),
            data["comparison_point"].get("finding"),
            data["hidden_conclusion"].get("finding"),
        )
        if finding
    ]
    block_connections = _connections_from_blocks(data)
    data["connections"] = _valid_connections(
        [*block_connections, *_json_list(data.get("connections"))],
        allowed_card_ids,
    )
    block_findings = _findings_from_blocks(data)
    data["cross_card_findings"] = _valid_cross_card_findings(
        [*block_findings, *_json_list(data.get("cross_card_findings"))],
        allowed_card_ids,
    )
    data["reasoning_trail"] = _valid_reasoning_trail_refs(
        data.get("reasoning_trail", []), allowed_card_ids
    )
    data["reasoning_steps"] = _valid_reasoning_step_refs(
        data.get("reasoning_steps", []), allowed_card_ids
    )
    data["follow_up_questions"] = []
    data.setdefault("radar_axes", [])

    peer_set: list[str] = []
    seen: set[str] = set()
    for c in cards:
        pid = c.get("peer_id")
        if pid and pid not in seen:
            seen.add(pid)
            peer_set.append(pid)
    data["peer_ids"] = peer_set

    data = _normalize_mixer_result_display_sentences(data)
    data["langfuse_trace_id"] = _get_langfuse_trace_id()
    data["warning"] = _warning_for(data)
    return data


def _normalize_mixer_result_display_sentences(result: dict) -> dict:
    result["mix_insight"] = _normalize_mixer_display_sentence(
        result.get("mix_insight") or result.get("insight", ""),
        _MIXER_FINAL_ONE_LINER_MAX,
    )
    result["insight"] = result["mix_insight"]
    for block_key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = result.get(block_key)
        if not isinstance(block, dict):
            continue
        block["finding"] = _normalize_mixer_display_sentence(block.get("finding", ""), 260)
        block["rationale"] = _normalize_mixer_display_sentence(
            block.get("rationale", ""),
            320,
        )
    actions = [
        action
        for item in _json_list(result.get("recommended_actions"))
        if (action := _normalize_mixer_display_sentence(item, 420))
    ]
    result["recommended_actions"] = _dedupe_keep_order(actions)
    for detail in _json_list(result.get("action_details")):
        if not isinstance(detail, dict):
            continue
        detail["action"] = _normalize_mixer_display_sentence(detail.get("action", ""), 420)
        detail["why"] = _normalize_mixer_display_sentence(detail.get("why", ""), 360)
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    result["final_one_liner"] = _normalize_mixer_display_sentence(
        result.get("final_one_liner") or hidden.get("finding") or result["mix_insight"],
        _MIXER_FINAL_ONE_LINER_MAX,
    )
    result["sk_ax_implication"] = _normalize_mixer_display_sentence(
        result.get("sk_ax_implication") or " ".join(result["recommended_actions"]),
        _MIXER_IMPLICATION_MAX,
    )
    result["bullet_signals"] = [
        finding
        for finding in (
            _dict_or_empty(result.get("common_pattern")).get("finding"),
            _dict_or_empty(result.get("comparison_point")).get("finding"),
            _dict_or_empty(result.get("hidden_conclusion")).get("finding"),
        )
        if finding
    ]
    return result


def _mixer_sentence_quality_issues(result: dict) -> list[str]:
    checks: list[tuple[str, object]] = [
        ("mix_insight", result.get("mix_insight")),
        ("final_one_liner", result.get("final_one_liner")),
        ("sk_ax_implication", result.get("sk_ax_implication")),
    ]
    for block_key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = result.get(block_key)
        if not isinstance(block, dict):
            checks.append((block_key, ""))
            continue
        checks.append((f"{block_key}.finding", block.get("finding")))
        checks.append((f"{block_key}.rationale", block.get("rationale")))
    for index, detail in enumerate(_json_list(result.get("action_details"))):
        if not isinstance(detail, dict):
            continue
        checks.append((f"action_details.{index}.action", detail.get("action")))
        checks.append((f"action_details.{index}.why", detail.get("why")))

    issues: list[str] = []
    for label, raw_text in checks:
        text = str(raw_text or "").strip()
        if not text:
            issues.append(f"{label}:empty")
            continue
        if "…" in text or "..." in text:
            issues.append(f"{label}:ellipsis")
            continue
        if not _is_complete_display_sentence(text):
            issues.append(f"{label}:incomplete")
    return issues[:12]


def _card_one_liner(card: dict) -> str:
    """카드 1장의 핵심을 한 문장으로 추출 — per_card 추론 단계 입력."""
    linked = _linked_results_from_card(card)
    analysis = _component(linked, "analysis")
    integrated = _component(linked, "integrated_issue")
    candidates: list[object] = [
        analysis.get("strategic_meaning"),
        analysis.get("market_signal"),
        integrated.get("main_issue"),
    ]
    summary_lines = card.get("summary_lines") or []
    if isinstance(summary_lines, str):
        try:
            summary_lines = json.loads(summary_lines)
        except json.JSONDecodeError:
            summary_lines = [summary_lines]
    if isinstance(summary_lines, list):
        candidates.extend(summary_lines)
    candidates.append(card.get("title"))
    for cand in candidates:
        if cand and str(cand).strip():
            return clip_string(str(cand).strip(), 160)
    return ""


def _normalize_follow_up_subject_terms(value: object) -> list[str]:
    terms = [_clean_follow_up_subject(item) for item in _json_list(value)]
    return _dedupe_keep_order([term for term in terms if term])[:3]


def _has_hangul_final_consonant(value: str) -> bool:
    for char in reversed(value.strip()):
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            return (code - 0xAC00) % 28 != 0
    return False


def _cards_for_evidence_refs(cards: list[dict], evidence_refs: object) -> list[dict]:
    refs = [str(ref).strip() for ref in _json_list(evidence_refs) if str(ref).strip()]
    if not refs:
        return []
    by_id = {str(card.get("id") or "").strip(): card for card in cards}
    return [by_id[ref] for ref in refs if ref in by_id]


def _clean_follow_up_subject(value: object) -> str:
    subject = str(value or "").strip().strip("'\"“”‘’")
    subject = re.sub(r"\s+", " ", subject)
    if subject.lower() in _FOLLOW_UP_INVALID_SUBJECTS:
        return ""
    if len(subject) < 2:
        return ""
    return clip_string(subject, 32)


def _subject_with_particle(subject: str) -> str:
    if subject.endswith(("들", "들)")):
        return f"{subject}이"
    return f"{subject}가"


def _repair_mixer_result_quality(
    *,
    result: dict,
    cards: list[dict],
    requested_card_ids: list[str],
) -> dict:
    prompt = _MIXER_REPAIR_PROMPT.replace("{context}", _format_analysis_units(cards)).replace(
        "{draft_json}",
        json.dumps(_mixer_result_for_repair(result), ensure_ascii=False, indent=2),
    )
    try:
        response = _get_llm("deep").invoke(
            prompt,
            config=tracing_config(
                agent="MixerAnalysisAgent",
                phase="repair_quality",
                prompt_version=f"{_PROMPT_VERSION}-repair",
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        repaired = _parse_and_validate(content, cards, requested_card_ids)
        repaired["repair_actions"] = _dedupe_keep_order(
            [*_json_list(result.get("repair_actions")), "mixer_sentence_quality_repaired"]
        )
        return repaired
    except Exception as exc:  # noqa: BLE001
        log.warning("Mixer quality repair 실패, 초안 사용 | error=%s", exc)
        return result


def _mixer_result_for_repair(result: dict) -> dict:
    keep_keys = (
        "mix_insight",
        "common_pattern",
        "comparison_point",
        "hidden_conclusion",
        "recommended_action_basis",
        "action_details",
        "recommended_actions",
        "sources_used",
        "confidence",
    )
    return {key: result.get(key) for key in keep_keys if key in result}


def _generate_mix_level_implication(result: dict, cards: list[dict]) -> dict:
    """믹스 분석 결과를 기존 ImplicationAgent에 넘겨 믹스 단위 대응방향을 생성한다."""
    try:
        return ImplicationAgent().generate(
            input_bundle=_mix_input_bundle(cards),
            integrated_issue=_mix_integrated_issue(result, cards),
            analysis=_mix_analysis_result(result, cards),
            profile_context=_mix_profile_context(cards),
            classification=_mix_classification(cards),
        )
    except Exception as exc:  # noqa: BLE001 - ImplicationAgent fallback 이후 최종 방어.
        log.warning("Mixer mix-level ImplicationAgent 호출 실패 | error=%s", exc)
        return {}


def _mix_integrated_issue(result: dict, cards: list[dict]) -> dict:
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    action_basis = _json_list(result.get("recommended_action_basis"))
    return {
        "is_valid_summary": True,
        "summary_scope": "mixer_integrated_issue",
        "cluster_id": result.get("mix_id") or "mixer",
        "main_company": "multi_peer",
        "source_article_ids": _dedupe_ints(
            [
                article_id
                for card in cards
                for article_id in _int_list(card.get("source_raw_article_ids"))
            ]
        ),
        "main_issue": result.get("mix_insight") or hidden.get("finding") or "",
        "integrated_text": " ".join(
            [
                str(result.get("mix_insight") or ""),
                str(common.get("finding") or ""),
                str(common.get("rationale") or ""),
                str(comparison.get("finding") or ""),
                str(comparison.get("rationale") or ""),
                str(hidden.get("finding") or ""),
                str(hidden.get("rationale") or ""),
                *[str(item) for item in action_basis],
            ]
        ).strip(),
        "consolidated_facts": _block_evidence_facts(common, cards),
        "business_signals": _dedupe_keep_order(
            [
                *[str(item) for item in action_basis],
                str(common.get("finding") or ""),
                str(common.get("rationale") or ""),
                str(comparison.get("finding") or ""),
                str(comparison.get("rationale") or ""),
                str(hidden.get("finding") or ""),
                str(hidden.get("rationale") or ""),
            ]
        ),
        "representative_sources": _sources_from_cards(cards),
    }


def _mix_analysis_result(result: dict, cards: list[dict]) -> dict:
    del cards
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    action_basis = _json_list(result.get("recommended_action_basis"))
    return {
        "is_valid_analysis": True,
        "analysis_scope": "mixed_peer_and_industry",
        "analysis_summary": result.get("mix_insight") or hidden.get("finding") or "",
        "strategic_meaning": _dedupe_keep_order(
            [
                str(common.get("finding") or ""),
                str(common.get("rationale") or ""),
                str(comparison.get("finding") or ""),
                str(comparison.get("rationale") or ""),
                str(hidden.get("finding") or ""),
                str(hidden.get("rationale") or ""),
                *[str(item) for item in action_basis],
            ]
        ),
        "market_signal": common.get("finding") or "",
        "impact_level": "medium",
        "impact_reason": " ".join(
            [
                str(hidden.get("finding") or result.get("mix_insight") or ""),
                str(hidden.get("rationale") or ""),
                " ".join(str(item) for item in action_basis),
            ]
        ).strip(),
        "risk_or_opportunity": "opportunity",
        "confidence": result.get("confidence", 0.0),
        "reason": "MixerAgent가 여러 카드의 연결 결과를 종합해 생성한 mix-level 분석",
    }


def _mix_profile_context(cards: list[dict]) -> dict:
    peer_profiles: dict[str, dict] = {}
    sector_context: dict[str, dict] = {}
    skax_profile: dict = {}
    for card in cards:
        linked_results = _linked_results_from_card(card)
        profile = linked_results.get("profile_context")
        if isinstance(profile, dict):
            if isinstance(profile.get("skax_profile"), dict) and not skax_profile:
                skax_profile = profile["skax_profile"]
            if isinstance(profile.get("peer_profiles"), dict):
                peer_profiles.update(profile["peer_profiles"])
            if isinstance(profile.get("sector_context"), dict):
                sector_context.update(profile["sector_context"])
        peer_id = str(card.get("peer_id") or card.get("company") or "")
        if peer_id and peer_id not in peer_profiles:
            peer_profiles[peer_id] = {"company_id": peer_id}
        sector = _card_sector(card)
        if sector and sector not in sector_context:
            sector_context[sector] = {"sector": sector}
    return {
        "skax_profile": skax_profile,
        "peer_profiles": peer_profiles,
        "sector_context": sector_context,
    }


def _mix_classification(cards: list[dict]) -> dict:
    sectors = _dedupe_keep_order([_card_sector(card) for card in cards if _card_sector(card)])
    event_types = _dedupe_keep_order(
        [str(card.get("event_type") or "") for card in cards if card.get("event_type")]
    )
    return {
        "sector": sectors[0] if sectors else "other",
        "sectors": sectors,
        "event_type": "mixed_issues",
        "source_event_types": event_types,
        "importance": "medium",
        "importance_score": _avg([_card_score(card) for card in cards]),
        "signals": {
            "source_card_count": len(cards),
            "peer_ids": _dedupe_keep_order(
                [str(card.get("peer_id") or card.get("company") or "") for card in cards]
            ),
        },
    }


def _mix_input_bundle(cards: list[dict]) -> dict:
    return {
        "bundle_id": f"mixer:{','.join(str(card.get('id')) for card in cards)}",
        "source_type": "mixer",
        "companies": _dedupe_keep_order(
            [str(card.get("peer_id") or card.get("company") or "") for card in cards]
        ),
        "sectors": _dedupe_keep_order([_card_sector(card) for card in cards if _card_sector(card)]),
        "items": cards,
        "sources": _sources_from_cards(cards),
        "metadata": {"source_card_ids": [card.get("id") for card in cards]},
    }


def _block_evidence_facts(block: dict, cards: list[dict]) -> list[dict]:
    card_ids = {str(card.get("id")) for card in cards}
    facts: list[dict] = []
    for item in _json_list(block.get("evidence")):
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id") or "")
        if card_id not in card_ids:
            continue
        text_value = str(item.get("text") or "").strip()
        if text_value:
            facts.append({"fact": text_value, "source_card_id": card_id})
    return facts


def _sources_from_cards(cards: list[dict]) -> list[dict]:
    sources: list[dict] = []
    for card in cards:
        for source in _json_list(card.get("sources")):
            if isinstance(source, dict):
                item = dict(source)
                item.setdefault("card_id", card.get("id"))
                sources.append(item)
        if not card.get("sources"):
            sources.append({"card_id": card.get("id"), "title": card.get("title") or ""})
    return sources[:20]


def _recommended_actions_from_implication(
    implication: dict, *, result: dict, limit: int = 3
) -> list[str]:
    skax = implication.get("skax_implication")
    candidates: list[str] = []
    if isinstance(skax, dict):
        candidates.extend(_json_list(skax.get("recommended_actions")))
    candidates.extend(_json_list(implication.get("recommended_actions")))
    grounding_text = _grounding_text_for_actions(result)
    grounded = [
        clip_implication(item)
        for item in _dedupe_keep_order(candidates)
        if item and _is_action_grounded(str(item), grounding_text)
    ]
    return grounded[:limit]


def _recommended_actions_from_basis(result: dict, *, limit: int = 3) -> list[str]:
    actions: list[str] = []
    for item in _json_list(result.get("recommended_action_basis")):
        text_value = str(item or "").strip()
        if not text_value:
            continue
        if not text_value.endswith(("다.", "요.", ".")):
            text_value = f"{text_value}."
        actions.append(clip_implication(text_value))
    return _dedupe_keep_order(actions)[:limit]


def _grounding_text_for_actions(result: dict) -> str:
    parts = [
        str(result.get("mix_insight") or ""),
        " ".join(str(item) for item in _json_list(result.get("recommended_action_basis"))),
    ]
    for key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = result.get(key)
        if not isinstance(block, dict):
            continue
        parts.append(str(block.get("finding") or ""))
        parts.append(str(block.get("rationale") or ""))
        for evidence in _json_list(block.get("evidence")):
            if isinstance(evidence, dict):
                parts.append(str(evidence.get("text") or ""))
    return " ".join(parts)


def _is_action_grounded(action: str, grounding_text: str) -> bool:
    action_tokens = _distinctive_tokens(action)
    if not action_tokens:
        return False
    grounding_tokens = _distinctive_tokens(grounding_text)
    if not grounding_tokens:
        return False
    overlap = action_tokens & grounding_tokens
    return len(overlap) >= max(2, min(4, len(action_tokens) // 3))


def _distinctive_tokens(text_value: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", str(text_value or ""))
        if not token.isdigit()
    }
    if len(tokens) <= 2:
        return tokens
    frequency: dict[str, int] = {}
    for token in tokens:
        frequency[token] = str(text_value).lower().count(token)
    return {token for token in tokens if frequency[token] <= 3}


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _get_langfuse_trace_id() -> str | None:
    try:
        from src.observability.langfuse_client import get_langfuse_handler

        handler = get_langfuse_handler()
        if handler is None:
            return None
        return getattr(handler, "last_trace_id", None)
    except Exception:
        return None


def _warning_for(data: dict) -> str | None:
    warnings: list[str] = []
    confidence = float(data.get("confidence") or 0.0)
    provenance_value = data.get("provenance")
    provenance = provenance_value if isinstance(provenance_value, dict) else {}
    quality_flags = _json_list(data.get("quality_flags") or provenance.get("quality_flags"))
    if confidence < 0.6 or quality_flags:
        warnings.append(
            "일부 카드의 통합 분석 연결이 제한되어 확인 가능한 카드 요약과 "
            "연결 근거를 중심으로 산출했습니다."
        )
    weak_blocks: list[str] = []
    for key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = _dict_or_empty(data.get(key))
        if not block.get("finding") or len(_json_list(block.get("evidence_card_ids"))) < 2:
            weak_blocks.append(key)
    if weak_blocks:
        warnings.append(
            "일부 해석 단계는 근거 카드 연결이 적어 결과 화면의 참조 근거를 함께 확인해야 합니다."
        )
    if not data.get("recommended_actions"):
        warnings.append("대응 방향 생성 결과가 비어 있어 원문 근거 확인이 필요합니다.")
    return " ".join(_dedupe_keep_order(warnings)) if warnings else None


def _error_response(
    short_reason: str,
    detail: str,
    card_ids: list[str],
    confidence: float = 0.0,
    integrated_issue_ids: list[str] | None = None,
) -> dict:
    log.warning("Mixer error | %s | detail=%s | ids=%s", short_reason, detail, card_ids)
    issue_ids = integrated_issue_ids or []
    return {
        "mix_id": _new_mix_id(),
        "mix_insight": "",
        "common_pattern": {
            "finding": "",
            "rationale": "",
            "evidence": [],
            "evidence_card_ids": [],
        },
        "comparison_point": {
            "finding": "",
            "rationale": "",
            "evidence": [],
            "evidence_card_ids": [],
        },
        "hidden_conclusion": {
            "finding": "",
            "rationale": "",
            "evidence": [],
            "evidence_card_ids": [],
        },
        "action_details": [],
        "recommended_actions": [],
        "insight": "",
        "final_one_liner": "",
        "sk_ax_implication": "",
        "bullet_signals": [],
        "radar_axes": [],
        "connections": [],
        "reasoning_trail": [],
        "reasoning_steps": [],
        "follow_up_questions": [],
        "confidence": confidence,
        "sources_used": card_ids,
        "source_integrated_issue_ids": issue_ids,
        "peer_ids": [],
        "langfuse_trace_id": None,
        "warning": f"{short_reason} — {detail}",
        "provenance": {
            "llm_model": _LLM_MODEL,
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": card_ids,
            "source_integrated_issue_ids": issue_ids,
            "quality_flags": [],
            "error": short_reason,
        },
    }
