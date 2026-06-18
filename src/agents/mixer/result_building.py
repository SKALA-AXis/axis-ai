"""mixer result_building — extracted from facade (move-only)."""

from __future__ import annotations

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
from src.services.agent_output_validation import (
    confidence_in_range,
)


def _build_reasoning_trail(result: dict, cards: list[dict]) -> list[dict]:
    """blocks → per-step 요약 trail (결정적). LLM 호출 없이 근거 기반으로 구성."""
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    mix_insight = result.get("mix_insight") or result.get("insight") or ""
    all_ids = [str(c["id"]) for c in cards if c.get("id")]
    trail: list[dict] = []

    def add(label: str, one_liner: object, refs: list[str]) -> None:
        text = str(one_liner or "").strip()
        if not text:
            return
        trail.append(
            {
                "seq": len(trail) + 1,
                "label": label,
                "one_liner": clip_string(text, 160),
                "evidence_refs": refs or all_ids,
                "langfuse_observation_id": None,
            }
        )

    add("공통 패턴", common.get("finding"), _json_list(common.get("evidence_card_ids")))
    add("비교 포인트", comparison.get("finding"), _json_list(comparison.get("evidence_card_ids")))
    add("숨은 결론", hidden.get("finding"), _json_list(hidden.get("evidence_card_ids")))
    add("믹스 인사이트", mix_insight, all_ids)
    return trail


def _build_reasoning_steps(result: dict, cards: list[dict]) -> list[dict]:
    """per_card → cross_card → synthesis 3-phase CoT (결정적, 근거 기반)."""
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    mix_insight = result.get("mix_insight") or result.get("insight") or ""
    all_ids = [str(c["id"]) for c in cards if c.get("id")]
    confidence = round(confidence_in_range(result.get("confidence", 0.0)), 2)
    steps: list[dict] = []

    def push(phase: str, question: str, inputs_used: list[str], answer: str, concl: str) -> None:
        if not str(answer or "").strip():
            return
        steps.append(
            {
                "step_idx": len(steps),
                "phase": phase,
                "question": question,
                "inputs_used": inputs_used or all_ids,
                "answer": clip_string(str(answer).strip(), 220),
                "intermediate_conclusion": clip_string(str(concl or "").strip(), 220),
                "confidence": confidence,
                "langfuse_observation_id": None,
            }
        )

    # per_card — 카드별 핵심 관찰
    for c in cards:
        cid = str(c.get("id") or "")
        if not cid:
            continue
        one_liner = _card_one_liner(c)
        if not one_liner:
            continue
        peer = str(c.get("company") or c.get("peer_id") or "").strip()
        push(
            "per_card",
            f"[{peer or cid}] 이 카드는 무엇을 말하는가?",
            [cid],
            one_liner,
            "",
        )

    # cross_card — 공통 흐름 + 차이
    cross_answer = " / ".join(
        str(x) for x in [common.get("finding"), comparison.get("finding")] if str(x or "").strip()
    )
    cross_concl = " ".join(
        str(x)
        for x in [common.get("rationale"), comparison.get("rationale")]
        if str(x or "").strip()
    )
    cross_inputs = _dedupe_keep_order(
        [
            *_json_list(common.get("evidence_card_ids")),
            *_json_list(comparison.get("evidence_card_ids")),
        ]
    )
    push(
        "cross_card",
        "여러 카드를 함께 보면 어떤 공통 흐름과 차이가 보이는가?",
        cross_inputs,
        cross_answer,
        cross_concl,
    )

    # synthesis — 숨은 결론 → 믹스 인사이트
    push(
        "synthesis",
        "함께 봐야 보이는 판단 기준의 변화는 무엇인가?",
        _json_list(hidden.get("evidence_card_ids")),
        hidden.get("finding") or mix_insight,
        hidden.get("rationale") or "",
    )
    return steps


def _build_follow_up_questions(result: dict, cards: list[dict]) -> list[str]:
    """Backward-compatible question list derived from structured follow-up checks."""
    return [
        str(item.get("question"))
        for item in _build_follow_up_checks(result, cards)
        if isinstance(item, dict) and str(item.get("question") or "").strip()
    ][:3]


def _follow_up_subject_phrase(
    cards: list[dict],
    evidence_refs: object = None,
    subject_terms: object = None,
) -> str:
    llm_subjects = _normalize_follow_up_subject_terms(subject_terms)
    if llm_subjects:
        return _format_follow_up_subjects(llm_subjects)

    scoped_cards = _cards_for_evidence_refs(cards, evidence_refs) or cards
    candidates: list[str] = []
    for card in scoped_cards:
        for raw_value in (
            card.get("company"),
            card.get("peer_id"),
            _follow_up_title_subject(card.get("title")),
        ):
            candidate = _clean_follow_up_subject(raw_value)
            if candidate:
                candidates.append(candidate)
                break
    return _format_follow_up_subjects(_dedupe_keep_order(candidates)[:3])


def _format_follow_up_subjects(subjects: list[str]) -> str:
    if len(subjects) == 1:
        return f"{subjects[0]} 관련 이슈들"
    if len(subjects) == 2:
        return _join_follow_up_subject_pair(subjects[0], subjects[1])
    if len(subjects) >= 3:
        return ", ".join(subjects[:3])
    return "선택한 카드들"


def _join_follow_up_subject_pair(left: str, right: str) -> str:
    particle = "과" if _has_hangul_final_consonant(left) else "와"
    return f"{left}{particle} {right}"


def _follow_up_title_subject(value: object) -> str:
    title = str(value or "").strip()
    if not title:
        return ""
    for separator in (",", "·", "|", " - ", " – ", " — "):
        if separator in title:
            title = title.split(separator, 1)[0].strip()
            break
    return clip_string(title, 32)


def _build_follow_up_checks(result: dict, cards: list[dict]) -> list[dict]:
    """다음 분석 질문과 현재 근거만으로 답할 수 있는 초안을 함께 생성."""
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    common = _dict_or_empty(result.get("common_pattern"))
    checks: list[dict] = []
    if str(hidden.get("finding") or "").strip():
        hidden_refs = _json_list(hidden.get("evidence_card_ids"))
        hidden_answer = (
            f"현재 근거만 보면 이 신호는 {len(hidden_refs) or len(cards)}개 카드에서 연결되는 "
            "반복 판단 기준으로 보는 편이 타당합니다. "
            f"{str(hidden.get('rationale') or hidden.get('finding') or '').strip()}"
        )
        checks.append(
            {
                "question": (
                    f"‘{clip_string(str(hidden['finding']).strip(), 70)}’ 신호가 "
                    "일회성 이벤트인지 반복 신호인지 확인할 후속 근거는 무엇인가?"
                ),
                "answer": clip_string(hidden_answer, 260),
                "purpose": (
                    "숨은 결론이 단일 카드 해석이 아니라 반복되는 시장 판단 기준인지 "
                    "검증하기 위한 확인 포인트입니다."
                ),
                "evidence_refs": hidden_refs,
            }
        )
    if str(comparison.get("finding") or "").strip():
        comparison_refs = _json_list(comparison.get("evidence_card_ids"))
        subject_phrase = _follow_up_subject_phrase(
            cards,
            comparison_refs,
            comparison.get("subject_terms"),
        )
        comparison_answer = (
            f"현재 답은 {_subject_with_particle(subject_phrase)} 같은 흐름 안에서도 "
            "서로 다른 성과 기준이나 적용 맥락을 앞세운다는 점입니다. "
            f"{str(comparison.get('rationale') or comparison.get('finding') or '').strip()}"
        )
        checks.append(
            {
                "question": (
                    f"{subject_phrase}의 서로 다른 접근 중 어느 고객군·업무 맥락에 "
                    "먼저 적용할 수 있는 차이인가?"
                ),
                "answer": clip_string(comparison_answer, 260),
                "purpose": (
                    "비교 포인트가 단순 회사별 차이가 아니라 고객군 선택이나 오퍼링 "
                    "우선순위로 이어질 수 있는지 판단하기 위한 질문입니다."
                ),
                "evidence_refs": comparison_refs,
            }
        )
    if _json_list(result.get("recommended_action_basis")) or _json_list(
        result.get("recommended_actions")
    ):
        action_text = "; ".join(
            clip_implication(str(action))
            for action in _json_list(result.get("recommended_actions"))[:2]
            if str(action or "").strip()
        )
        basis_text = "; ".join(
            clip_implication(str(basis))
            for basis in _json_list(result.get("recommended_action_basis"))[:2]
            if str(basis or "").strip()
        )
        action_focus = action_text or basis_text or "추천 액션의 우선순위를 실행 조건별로 나누는 것"
        action_answer = (
            f"현재 답은 {action_focus}입니다. "
            "실행 전에는 고객군, 적용 업무, 수익화 수치, 리스크 게이트를 "
            "분리해 우선순위를 정해야 합니다."
        )
        checks.append(
            {
                "question": (
                    "대응 방향을 실행 판단으로 바꾸려면 어떤 수치·고객·리스크 "
                    "조건이 추가로 필요한가?"
                ),
                "answer": clip_string(action_answer, 280),
                "purpose": (
                    "권고가 선언으로 끝나지 않고 투자, 파트너십, 리스크 게이트 결정으로 "
                    "이어지려면 부족한 근거를 분리해야 합니다."
                ),
                "evidence_refs": _dedupe_keep_order(
                    [
                        *_json_list(common.get("evidence_card_ids")),
                        *_json_list(comparison.get("evidence_card_ids")),
                        *_json_list(hidden.get("evidence_card_ids")),
                    ]
                ),
            }
        )
    seen: set[str] = set()
    result_checks: list[dict] = []
    for check in checks:
        question = str(check.get("question") or "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        result_checks.append(check)
    return result_checks[:3]


def _build_analysis_depth(result: dict, cards: list[dict], analysis_mode: str) -> dict:
    del result
    card_count = len(cards)
    if analysis_mode == "deep":
        return {
            "mode": "deep",
            "label": "정확 분석",
            "summary": (
                f"선택 카드 {card_count}장을 최신 고정밀 모델로 1차 믹스 분석한 뒤 문장 품질 보강, "
                "믹스 단위 시사점 보강, 레이더 축 해석 정교화, "
                "단계별 근거 재구성을 추가로 수행했습니다."
            ),
            "included_steps": [
                f"{_model_for_mode('deep')} 기반 LLM 1차 믹스 분석",
                "공통 패턴·비교 포인트·숨은 결론 품질 보강",
                "믹스 단위 ImplicationAgent 보강",
                "6축 레이더별 본문 해석 질문과 답변 정교화",
                "고객군·오퍼링·자원 배분·파트너십·리스크 게이트별 실행안 보강",
                "근거 카드와 실행 조건 상세 정리",
            ],
            "omitted_steps": [],
        }
    return {
        "mode": "quick",
        "label": "빠른 실행",
        "summary": (
            f"선택 카드 {card_count}장의 핵심 연결만 빠르게 산출했습니다. "
            "정확 분석보다 짧게 끝나도록 고정밀 모델 재검증과 추가 품질 보강 호출은 생략합니다."
        ),
        "included_steps": [
            f"{_model_for_mode('quick')} 기반 LLM 1차 믹스 분석",
            "기본 근거 연결",
            "기본 대응 방향 정리",
        ],
        "omitted_steps": [
            f"{_model_for_mode('deep')} 기반 고정밀 재검증",
            "문장 품질 보강",
            "믹스 단위 ImplicationAgent 보강",
            "6축 레이더별 본문 해석 정교화",
        ],
    }


def _build_deep_dive_sections(result: dict, cards: list[dict]) -> list[dict]:
    card_lookup = {str(card.get("id") or ""): card for card in cards}
    sections: list[dict] = []

    def evidence_refs(block: dict) -> list[str]:
        refs = [str(item) for item in _json_list(block.get("evidence_card_ids")) if str(item)]
        if refs:
            return _dedupe_keep_order(refs)
        return _dedupe_keep_order(
            [
                str(item.get("card_id") or "")
                for item in _json_list(block.get("evidence"))
                if isinstance(item, dict) and str(item.get("card_id") or "")
            ]
        )

    def evidence_summary(refs: list[str]) -> str:
        lines: list[str] = []
        for ref in refs[:5]:
            card = card_lookup.get(ref, {})
            title = str(card.get("title") or ref).strip()
            company = str(card.get("company") or card.get("peer_id") or "").strip()
            prefix = f"{company}: " if company else ""
            lines.append(f"{prefix}{title}")
        return "\n".join(lines)

    for key, title in (
        ("common_pattern", "공통 패턴 상세 검증"),
        ("comparison_point", "비교 포인트 상세 검증"),
        ("hidden_conclusion", "숨은 결론 상세 검증"),
    ):
        block = _dict_or_empty(result.get(key))
        finding = str(block.get("finding") or "").strip()
        rationale = str(block.get("rationale") or "").strip()
        refs = evidence_refs(block)
        if not (finding or rationale or refs):
            continue
        sections.append(
            {
                "title": title,
                "summary": finding,
                "details": [
                    {"label": "핵심 판단", "text": finding, "evidence_refs": refs},
                    {"label": "판단 근거", "text": rationale, "evidence_refs": refs},
                    {"label": "참조 카드", "text": evidence_summary(refs), "evidence_refs": refs},
                ],
            }
        )

    action_details = [
        item for item in _json_list(result.get("action_details")) if isinstance(item, dict)
    ]
    detail_items: list[dict] = []
    for item in action_details[:5]:
        refs = [str(ref) for ref in _json_list(item.get("evidence_card_ids")) if str(ref)]
        action = str(item.get("action") or "").strip()
        why = str(item.get("why") or "").strip()
        use_case = str(item.get("use_case") or "").strip()
        text = "\n".join(
            part
            for part in (
                f"실행안: {action}" if action else "",
                f"판단 이유: {why}" if why else "",
                f"적용 맥락: {use_case}" if use_case else "",
            )
            if part
        )
        if text:
            detail_items.append(
                {"label": use_case or "실행 조건", "text": text, "evidence_refs": refs}
            )
    if detail_items:
        sections.append(
            {
                "title": "대응 방향 상세",
                "summary": "추천 액션을 실행 조건, 적용 맥락, 참조 근거 기준으로 분해했습니다.",
                "details": detail_items,
            }
        )

    return sections[:4]
