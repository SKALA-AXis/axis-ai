# 작성일: 2026-05-15
# 작성자: 최종민
# 변경이력:
#   2026-05-15 최종민 — 믹서 분석 프로토타입 도입, v2 프롬프트·6축 레이더
#   2026-05-22 박지원 — 믹서 분석 mypy 오류 수정
#   2026-05-22 심유정 — 에이전트 구조·믹서 인사이트 개선
#   2026-06-04 박진 — 통합 이슈 기반 믹서·브리핑 플로우 추가, PDF 내보내기·믹서 자격증명 수정
#   2026-06-12 안가은 — 믹서 결과 문구·문장 정규화 및 키워드 트렌드 파이프라인 갱신
"""MixerAnalysisAgent — linked integrated/analysis/implication result mixer.

design: ``axis-ai/design/30-analysis/mixer-analysis.md``.

본 모듈은 카드 표시 문구를 다시 요약하지 않는다. 카드에 연결된
``integrated_issue`` / ``analysis`` / ``implication`` / ``profile_context`` 여러 건을
입력으로 받아, 단일 이슈로는 보이지 않는 공통 패턴 / 비교 포인트 / 숨은 결론 /
대응방향을 도출한다.

핵심 entry point:

    ``MixerAnalysisAgent().analyze(card_ids, ratios, user_context, analysis_mode)``
    — DB 카드 기반.
    ``MixerAnalysisAgent().analyze_items(items, ratios, user_context, analysis_mode)``
    — 로컬 목업/테스트 기반.

프론트 입력은 card_id 이지만, Mixer 의 실제 분석 재료는 카드 표시용 3줄 요약이 아니라
카드에 연결된 통합 결과, 분석 결과, 시사점 결과, 프로필 context 다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

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
from src.agents.mixer.radar import (  # noqa: F401
    _compute_radar,
    _fill_missing_radar_axis_interpretations,
    _format_missing_radar_axes,
    _format_radar,
    _merge_radar_axis_interpretations,
    _normalize_radar_axis_key,
    _parse_radar_axis_interpretation_response,
    _peer_diversity_score,
    _radar_axis,
    _radar_axis_analysis_prompt,
    _radar_score_meaning,
)
from src.agents.mixer.result_building import (  # noqa: F401
    _build_analysis_depth,
    _build_deep_dive_sections,
    _build_follow_up_checks,
    _build_follow_up_questions,
    _build_reasoning_steps,
    _build_reasoning_trail,
    _follow_up_subject_phrase,
    _follow_up_title_subject,
    _format_follow_up_subjects,
    _join_follow_up_subject_pair,
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
    _llms,
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
from src.middleware.analysis_ledger import with_ledger_writeback
from src.observability.langfuse_client import tracing_config
from src.services.agent_output_validation import (
    confidence_in_range,
)
from src.services.analysis_units import (
    AnalysisUnit,
    analysis_units_from_cards,
    card_like_from_units,
    confidence_penalty_for_flags,
    load_analysis_units_by_card_ids,
    load_analysis_units_by_integrated_issue_ids,
    quality_flags_for_units,
    source_integrated_issue_ids,
)
from src.services.llm_env import (
    is_missing_llm_credentials_error,
    missing_llm_credentials_message,
)

log = logging.getLogger(__name__)

# 믹서 실행 단계 — SSE progress 용. 에이전트가 실제로 넘는 단계 경계만 emit 한다
# (prepare: 카드/이슈 로드, analyze: 메인 LLM, synthesize: 대응방향 LLM, finalize: 추론 정리).


# ──────────────────────────────────────────────────────────────────────────
# Prompt — design/30-analysis/mixer-analysis.md §6.2.
# ──────────────────────────────────────────────────────────────────────────


class MixerAnalysisAgent:
    """3-phase per_card / cross_card / synthesis CoT 카드 분석 agent."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    @with_ledger_writeback("MixerAnalysisAgent")
    async def analyze(
        self,
        card_ids: list[str] | None = None,
        integrated_issue_ids: list[str] | None = None,
        ratios: dict | None = None,
        user_context: str | None = None,
        analysis_mode: str = "quick",
        user_id: str | None = None,
        progress: "ProgressFn | None" = None,
    ) -> dict:
        """N 카드 선택 → 저장된 분석 payload 기반 6축 radar + cross-issue 분석.

        Args:
            card_ids: 프론트에서 선택한 카드 id (호환 입력, 2 ≤ N ≤ 20 권장).
            integrated_issue_ids: canonical integrated_issues.id 입력. card_ids보다 우선.
            ratios: peer / industry / keyword 가중치 (frontend slider 결과).
            user_context: 사용자 자유 입력.
            analysis_mode: "quick" 은 메인 믹스 분석만 실행, "deep" 은 품질 보강과
                mix-level ImplicationAgent 보강까지 실행.

        Returns:
            MixerAnalysisOutput dict — design §5 schema.
        """
        requested_card_ids = _dedupe_keep_order(
            [str(card_id).strip() for card_id in card_ids or [] if str(card_id).strip()]
        )
        requested_integrated_issue_ids = _dedupe_keep_order(
            [
                str(issue_id).strip()
                for issue_id in integrated_issue_ids or []
                if str(issue_id).strip()
            ]
        )
        requested_ids = (
            requested_integrated_issue_ids if requested_integrated_issue_ids else requested_card_ids
        )
        if len(requested_ids) < _MIN_CARDS:
            return _error_response(
                "분석 단위 부족",
                f"mixer 는 최소 {_MIN_CARDS}개 분석 단위 필요 (받음={len(requested_ids)})",
                requested_ids,
                integrated_issue_ids=requested_integrated_issue_ids,
            )

        if len(requested_ids) > _MAX_CARDS:
            log.warning(
                "Mixer | 분석 단위가 너무 많음 — 상위 %d개로 truncate (받음=%d)",
                _MAX_CARDS,
                len(requested_ids),
            )
            if requested_integrated_issue_ids:
                requested_integrated_issue_ids = requested_integrated_issue_ids[:_MAX_CARDS]
            else:
                requested_card_ids = requested_card_ids[:_MAX_CARDS]
            requested_ids = (
                requested_integrated_issue_ids
                if requested_integrated_issue_ids
                else requested_card_ids
            )

        _emit_progress(progress, "prepare")
        if requested_integrated_issue_ids:
            if user_id is not None:
                analysis_units = load_analysis_units_by_integrated_issue_ids(
                    requested_integrated_issue_ids,
                    user_id=user_id,
                )
            else:
                analysis_units = load_analysis_units_by_integrated_issue_ids(
                    requested_integrated_issue_ids
                )
        else:
            if user_id is not None:
                analysis_units = load_analysis_units_by_card_ids(
                    requested_card_ids, user_id=user_id
                )
            else:
                analysis_units = load_analysis_units_by_card_ids(requested_card_ids)
        if len(analysis_units) < _MIN_CARDS:
            return _error_response(
                "분석 단위 조회 실패",
                f"canonical AnalysisUnit 조회 결과 부족 (받음={len(analysis_units)})",
                requested_ids,
                integrated_issue_ids=requested_integrated_issue_ids,
            )

        cards = card_like_from_units(analysis_units)
        source_anchor_ids = [str(card.get("id")) for card in cards if card.get("id")]
        return await self._analyze_cards(
            cards=cards,
            requested_card_ids=source_anchor_ids,
            requested_integrated_issue_ids=requested_integrated_issue_ids,
            analysis_units=analysis_units,
            ratios=ratios,
            user_context=user_context,
            analysis_mode=analysis_mode,
            progress=progress,
        )

    async def analyze_items(
        self,
        items: list[dict],
        ratios: dict | None = None,
        user_context: str | None = None,
        analysis_mode: str = "quick",
        progress: "ProgressFn | None" = None,
    ) -> dict:
        """로컬 목업/테스트용 linked result 묶음 → 믹스 인사이트 생성.

        실제 서비스에서는 ``analyze(card_ids=...)`` 가 DB 에서 카드와
        카드에 연결된 ``integrated_issue`` / ``analysis`` / ``implication`` /
        ``profile_context`` 를 조회한다. 이 메서드는 아직 카드 저장이 안정화되지 않은
        상황에서 동일한 믹서 로직을 목업 데이터로 검증하기 위한 진입점이다.
        """
        cards = _cards_from_linked_result_items(items)
        card_ids = [str(card.get("id")) for card in cards if card.get("id")]
        if len(cards) < _MIN_CARDS:
            return _error_response(
                "분석 단위 부족",
                f"mixer 는 최소 {_MIN_CARDS}개 linked result 필요 (받음={len(cards)})",
                card_ids,
            )
        if len(cards) > _MAX_CARDS:
            cards = cards[:_MAX_CARDS]
            card_ids = [str(card.get("id")) for card in cards if card.get("id")]
        analysis_units = analysis_units_from_cards(cards)
        cards = card_like_from_units(analysis_units)
        card_ids = [str(card.get("id")) for card in cards if card.get("id")]

        return await self._analyze_cards(
            cards=cards,
            requested_card_ids=card_ids,
            requested_integrated_issue_ids=[],
            analysis_units=analysis_units,
            ratios=ratios,
            user_context=user_context,
            analysis_mode=analysis_mode,
            progress=progress,
        )

    async def _analyze_cards(
        self,
        *,
        cards: list[dict],
        requested_card_ids: list[str],
        requested_integrated_issue_ids: list[str],
        analysis_units: list[AnalysisUnit],
        ratios: dict | None,
        user_context: str | None,
        analysis_mode: str,
        progress: "ProgressFn | None" = None,
    ) -> dict:
        normalized_mode = _normalize_analysis_mode(analysis_mode)
        # 6축 radar 미리 계산 — LLM input 으로 anchor 제공 (v2)
        radar = _compute_radar(cards)
        prompt = (
            _MIXER_PROMPT.replace("{context}", _format_analysis_units(cards))
            .replace("{radar_text}", _format_radar(radar))
            .replace("{ratios_text}", _format_ratios(ratios))
            .replace("{user_context}", (user_context or "").strip() or "*없음*")
        )

        _emit_progress(progress, "analyze")
        try:
            response = _get_llm(normalized_mode).invoke(
                prompt,
                config=tracing_config(
                    agent="MixerAnalysisAgent",
                    phase="analyze",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
        except Exception as e:
            log.exception("MixerAnalysisAgent LLM 호출 실패 | error=%s", e)
            detail = (
                missing_llm_credentials_message() if is_missing_llm_credentials_error(e) else str(e)
            )
            return _error_response(
                "LLM 호출 실패",
                detail,
                requested_card_ids,
                integrated_issue_ids=requested_integrated_issue_ids,
            )

        result = _parse_and_validate(content, cards, requested_card_ids)
        sentence_quality_issues = _mixer_sentence_quality_issues(result)
        if normalized_mode == "deep" or sentence_quality_issues:
            if sentence_quality_issues:
                log.info(
                    "Mixer sentence quality repair requested | issues=%s",
                    sentence_quality_issues,
                )
            result = _repair_mixer_result_quality(
                result=result,
                cards=cards,
                requested_card_ids=requested_card_ids,
            )
        source_issue_ids = source_integrated_issue_ids(analysis_units)
        quality_flags = quality_flags_for_units(analysis_units)
        if source_issue_ids:
            result["source_integrated_issue_ids"] = source_issue_ids
        else:
            result.setdefault("source_integrated_issue_ids", [])
        if quality_flags:
            result["quality_flags"] = quality_flags
            penalty = confidence_penalty_for_flags(quality_flags)
            result["confidence"] = round(
                max(0.0, confidence_in_range(result.get("confidence", 0.0)) - penalty),
                2,
            )
        _emit_progress(progress, "synthesize")
        if normalized_mode == "deep":
            result["action_details"] = _augment_deep_action_details(result)
            actions = _recommended_actions_from_details(result, limit=5)
            mix_implication = _generate_mix_level_implication(result=result, cards=cards)
            if mix_implication:
                result["mix_implication"] = mix_implication
                implication_actions = _recommended_actions_from_implication(
                    mix_implication, result=result, limit=5
                )
                actions = _dedupe_keep_order([*actions, *implication_actions])
                if not actions:
                    actions = _recommended_actions_from_basis(result, limit=5)
            if actions:
                result["recommended_actions"] = actions[:5]
                result["sk_ax_implication"] = clip_implication(" ".join(actions[:5]))
        else:
            actions = _recommended_actions_from_details(result) or _recommended_actions_from_basis(
                result
            )
            if actions:
                result["recommended_actions"] = actions[:3]
                result["sk_ax_implication"] = clip_implication(" ".join(actions[:3]))
        result = _normalize_mixer_result_display_sentences(result)
        result["radar_axes"] = _merge_radar_axis_interpretations(
            radar, result.get("radar_axis_interpretations")
        )
        result["radar_axes"] = _fill_missing_radar_axis_interpretations(
            result["radar_axes"], cards, normalized_mode
        )
        result["mix_id"] = _new_mix_id()
        result.setdefault("provenance", {}).update(
            {
                "llm_model": _model_for_mode(normalized_mode),
                "prompt_version": _PROMPT_VERSION,
                "analysis_mode": normalized_mode,
                "analysis_quality": "fast" if normalized_mode == "quick" else "detailed",
                "source_card_ids": [c["id"] for c in cards],
                "requested_integrated_issue_ids": requested_integrated_issue_ids,
                "source_integrated_issue_ids": source_issue_ids,
                "quality_flags": quality_flags,
                "ratios": ratios or {},
                "analysis_basis": (
                    "integrated_issues.id -> integrated_issue+analysis+implication+"
                    "classification+validation"
                ),
            }
        )
        # 추론 흐름(trail) / 단계별 CoT(steps) / 후속 질문 — repair 이후 최종 blocks 기반으로
        # 결정적 구성 (추가 LLM 호출 없음). card_ids / integrated_issue_ids 양 경로 모두 채워짐.
        _emit_progress(progress, "finalize")
        result["analysis_depth"] = _build_analysis_depth(result, cards, normalized_mode)
        result["deep_dive_sections"] = (
            _build_deep_dive_sections(result, cards) if normalized_mode == "deep" else []
        )
        result["reasoning_trail"] = _build_reasoning_trail(result, cards)
        result["reasoning_steps"] = _build_reasoning_steps(result, cards)
        result["follow_up_checks"] = _build_follow_up_checks(result, cards)
        result["follow_up_questions"] = [
            str(item.get("question"))
            for item in _json_list(result["follow_up_checks"])
            if isinstance(item, dict) and str(item.get("question") or "").strip()
        ][:3]
        result["warning"] = _warning_for(result)
        return result


# ──────────────────────────────────────────────────────────────────────────
# Deterministic radar (no LLM) — design §6.1
# ──────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


class MixerAgent(MixerAnalysisAgent):
    """Architecture-facing name for the 2단계 mixer agent."""
