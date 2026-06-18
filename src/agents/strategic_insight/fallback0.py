"""strategic_insight fallback0 — extracted from facade (move-only)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from src.agents.strategic_insight.constants import (  # noqa: F401
    _DOMAIN_ALIASES,
    _EVIDENCE_LABELS,
    _FRONTEND_READY_CLAIM_STRENGTHS,
    _FRONTEND_READY_CLAIM_TYPES,
    _FRONTEND_READY_DISPLAY_SOURCES,
    _FRONTEND_READY_EVIDENCE_MODES,
    _FRONTEND_READY_GENERIC_ROLE_TERMS,
    _FRONTEND_READY_SOURCES,
    _FRONTEND_READY_STRONG_CLAIM_TYPES,
    _IMPACT_LEVELS,
    _INTERNAL_CHECKPOINT_GROUPS,
    _NUMERIC_TOKEN_PATTERN,
    _RELATIONSHIP_ACTIVITY_TYPES,
    _RELATIONSHIP_PATTERN,
    _RISK_OR_OPPORTUNITY,
    _SKAX_ACTION_VERB_GROUPS,
    _SUPPLIER_CAPABILITY_PATTERN,
    _UNCERTAINTY_PATTERN,
    _UNSUPPORTED_CLAIM_PATTERNS,
    OVERCLAIM_PATTERNS,
)
from src.agents.strategic_insight.industry_synthesis import (  # noqa: F401
    _industry_action_result_phrase,
    _industry_anchor_market_reading,
    _industry_axis_context_text,
    _industry_criteria_flags,
    _industry_criteria_phrase,
    _industry_decision_criteria_from_issue,
    _industry_evidence_reading_phrase,
    _industry_evidence_sentence,
    _industry_frontend_axes,
    _industry_frontend_item,
    _industry_frontend_item_is_distinct,
    _industry_frontend_items_are_separable,
    _industry_frontend_ready_from_decision,
    _industry_implication_sentence,
    _industry_item_action_result_terms,
    _industry_item_action_terms,
    _industry_item_anchor_norms,
    _industry_item_conclusion_terms,
    _industry_item_decision_criteria,
    _industry_lines_with_anchors,
    _industry_market_reading_phrase,
    _industry_or_market_infra_watch_only_decision,
    _industry_primary_action_evidence_line,
    _industry_role_structure_change_clause,
    _industry_role_structure_subject,
    _industry_sets_are_substantially_different,
    _industry_signal_evidence_summary,
    _industry_signal_scope,
    _industry_source_grounded_action_block,
    _industry_source_grounded_action_evidence,
    _industry_source_grounded_action_sentence,
    _merge_similar_industry_frontend_items,
    _merged_industry_frontend_item,
)
from src.agents.strategic_insight.loaders import (  # noqa: F401
    _classification_from_card_row,
    _classification_from_issue_row,
    _input_bundle_from_analysis_package,
    _load_card_news_analysis_package,
    _load_integrated_issue,
    _load_integrated_issue_analysis_package,
    _load_profile_context_for_issue,
)
from src.agents.strategic_insight.models_config import (  # noqa: F401
    _DEFAULT_FRONTEND_READY_MODEL,
    _DEFAULT_LLM_MODEL,
    _FRONTEND_READY_MODEL,
    _FRONTEND_READY_REPAIR_MODEL,
    _LLM_MAX_COMPLETION_TOKENS,
    _LLM_MODEL,
    _LLM_REQUEST_TIMEOUT_SECONDS,
    _LLM_TEMPERATURE,
    _PROMPT_VERSION,
    _SELF_REVIEW_DISABLED,
    _SELF_REVIEW_DISABLED_VALUES,
    _SELF_REVIEW_MODEL,
    _SELF_REVIEW_MODEL_RAW,
    _llm_model_config_diagnostics,
    _llm_model_for_phase,
)
from src.agents.strategic_insight.parsers import (  # noqa: F401
    _augment_sourced_evidence_ids,
    _normalize_analysis_block,
    _normalize_frontend_basis,
    _normalize_frontend_ready,
    _normalize_frontend_ready_block,
    _normalize_frontend_ready_source,
    _normalize_grounding_summary,
    _normalize_implication_block,
    _normalize_issue_understanding,
    _normalize_llm_profile_linkage,
    _normalize_recommended_actions,
    _normalize_skax_response_linkage,
    _normalize_used_context_layers,
    _parse_and_normalize,
    _parse_review_and_normalize,
)
from src.agents.strategic_insight.profile_linkage import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    _SUPPLY_CONTRACT_PATTERN,
    GENERIC_BUSINESS_CATEGORIES,
    UNCERTAIN_ACTIVITY_MARKERS,
    _activity_candidates_have,
    _activity_types_from_issue,
    _appears_in_structured_issue_fields,
    _appears_repeatedly_in_evidence,
    _build_profile_linkage_evaluation,
    _business_novelty_status,
    _cluster_fact_intelligence_for_prompt,
    _compact_capability_evolution_for_prompt,
    _compact_evidence_value,
    _compact_financial_profile_context,
    _compact_profile_item,
    _compact_value,
    _companies_from_integrated_issue,
    _company_token_variants,
    _content_tokens,
    _evaluate_single_profile_linkage,
    _evidence_repeat_score,
    _expand_profile_relevance_tokens,
    _extract_issue_structured_signals,
    _extract_ranked_terms,
    _fact_like_text,
    _fact_texts,
    _financial_profile_context_for_prompt,
    _grounding_text,
    _has_uncertain_role,
    _implication_mode_from_linkage,
    _integrated_grounding_text,
    _is_generic_business_term,
    _is_low_signal_profile_relevance_token,
    _issue_relevance_tokens,
    _issue_structured_terms,
    _iter_dicts,
    _json_dumps,
    _linkage_level_from_profile_score,
    _looks_like_korean_function_word_or_ending,
    _looks_like_source_noise,
    _main_company_is_customer_or_buyer,
    _main_company_near_role_pattern,
    _matched_profile_item_from_entry,
    _normalize_activity_candidates,
    _normalize_content_token,
    _normalize_entity_token,
    _peer_role_mode_for_linkage,
    _profile_entries_for_linkage,
    _profile_entry_relevance_text,
    _profile_entry_specificity_level,
    _profile_for_prompt,
    _profile_linkage_for_company,
    _profile_linkage_guidance,
    _profile_linkage_issue_context,
    _profile_relevance_hint_text,
    _profile_relevance_score,
    _rank_relevant_profile_items,
    _raw_normalized_terms,
    _relevant_business_areas_for_prompt,
    _role_mode_from_evidence_fallback,
    _role_mode_from_structured_activity,
    _score_profile_entry_against_issue,
    _shrink_profile,
    _source_noise_terms_from_issue,
    _string_values_from_any,
    _structured_activity_matches_active_role,
    _structured_activity_matches_contract_role,
    _structured_activity_matches_partnership_role,
    _structured_activity_matches_selected_role,
    _structured_field_overlap_score,
    _supplier_names_for_target_counterparty,
    _target_name_patterns,
    _term_signal_weight,
    _tokens_semantically_close,
    _weighted_term_overlap_score,
)
from src.agents.strategic_insight.prompt_builder import (  # noqa: F401
    _action_artifact_plan_for_prompt,
    _analysis_context_for_model,
    _analysis_context_for_prompt,
    _analysis_context_to_dict,
    _bundle_for_prompt,
    _bundle_to_dict,
    _classification_for_prompt,
    _context_availability_for_prompt,
    _integrated_issue_for_prompt,
    _profile_to_dict,
    _role_mode_instructions,
    _strategic_evidence_inventory_for_prompt,
    _strategic_evidence_pack_for_prompt,
)
from src.agents.strategic_insight.prompts import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    ACTION_REPAIR_SYSTEM_PROMPT,
    ACTION_REPAIR_USER_PROMPT_TEMPLATE,
    COUNTERPARTY_REPAIR_SYSTEM_PROMPT,
    COUNTERPARTY_REPAIR_USER_PROMPT_TEMPLATE,
    FRONTEND_READY_REPAIR_SYSTEM_PROMPT,
    FRONTEND_READY_REPAIR_USER_PROMPT_TEMPLATE,
    REPAIR_SYSTEM_PROMPT,
    REPAIR_USER_PROMPT_TEMPLATE,
    REPORT_COPY_REPAIR_SYSTEM_PROMPT,
    REPORT_COPY_REPAIR_USER_PROMPT_TEMPLATE,
    REVIEW_SYSTEM_PROMPT,
    REVIEW_USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from src.agents.strategic_insight.si_base0 import (  # noqa: F401
    _action_structure_axis_phrase,
    _anchor_phrase,
    _available_analysis_layers,
    _available_profile_fields,
    _business_context_terms,
    _business_line_candidate_details,
    _business_line_candidates,
    _compact_issue_term,
    _company_identity_terms,
    _company_variants_for_direct_action_match,
    _consolidated_facts_from_evidence_refs,
    _contains_high_risk_unsupported_claim,
    _contract_duration_phrase,
    _contract_scale_phrase,
    _counterparty_capability_overclaim_violation,
    _counterparty_role_action_violation,
    _distinct_anchor_tokens,
    _evidence_label,
    _evidence_snippets_from_issue,
    _extract_issue_subject_from_text,
    _fact_basis_from_evidence_refs,
    _fact_has_summary_role,
    _fact_is_referenced,
    _financial_structure_terms,
    _first_peer_id,
    _first_peer_name,
    _frontend_ready_action_auxiliary_scale_overreach_violation,
    _frontend_ready_evidence_repeats_summary,
    _frontend_ready_internal_copy_term_violation,
    _frontend_ready_malformed_display_sentence_violation,
    _frontend_ready_role_term_norm,
    _frontend_ready_unsupported_business_concept_violation,
    _frontend_ready_unsupported_effect_violation,
    _generalize_peer_structure_in_action_sentence,
    _global_company_alias_pattern,
    _grounded_numeric_keys_for_issue,
    _has_action_evidence_internal_axis,
    _has_action_execution_perspective,
    _has_all_industry_scope,
    _has_attention_growth_claim,
    _has_broad_expansion_claim,
    _has_clean_displayable_frontend_ready_diagnostics,
    _has_clear_business_domain_fact,
    _has_direct_skax_execution_action,
    _has_effect_scope,
    _has_effectiveness_claim,
    _has_execution_or_operation_fact,
    _has_expansion_support,
    _has_global_scope,
    _has_market_infra_signal,
    _has_moderate_action_decision_axis,
    _has_profile_context,
    _has_public_private_scope,
    _has_public_private_scope_support,
    _has_relationship_grounding,
    _has_relevant_peer_profile_context,
    _has_required_output_structure,
    _has_service_advancement_fact,
    _has_sizable_tech_event_signal,
    _has_status_strength_claim,
    _has_status_strength_event_support,
    _has_stock_market_signal,
    _has_strong_actionable_issue_signal,
    _has_unsupported_pattern,
    _hiring_signal_anchors,
    _infer_frontend_claim_type,
    _integrated_text_anchor_lines,
    _is_claim_scope_term,
    _is_follow_up_or_watch_field,
    _is_low_signal_content_token,
    _is_product_or_service_launch_issue,
    _is_substantive_industry_evidence_line,
    _is_technology_event_adoption_axis,
    _issue_counterparty_terms,
    _issue_specific_product_terms_for_action,
)
from src.agents.strategic_insight.si_base1 import (  # noqa: F401
    _action_plan_terms_for_keys,
    _action_text_has_internal_strategy_checkpoint,
    _action_text_has_issue_signal,
    _action_text_has_skax_change,
    _concrete_profile_terms,
    _evidence_scope_terms,
    _evidence_sentence_has_dynamic_grounding,
    _frontend_ready_insight_evidence_action_language_violation,
    _frontend_ready_role_bigrams,
    _frontend_ready_role_terms,
    _has_business_structure_fact,
    _has_direct_business_signal,
    _has_peer_mention,
    _has_relevant_peer_profile_linkage,
    _high_signal_issue_overlap_count,
    _high_signal_tokens_for_repetition,
    _is_financial_or_transaction_frontend_block,
    _issue_evidence_terms_for_action_plan,
    _issue_fact_lines,
    _issue_subject_phrase,
    _known_fact_ids,
    _linkage_level_from_match_count,
    _linkage_rank,
    _looks_like_sentence_slot,
    _main_company_display,
    _main_issue_context_text,
    _matching_fact_ids,
    _matching_profile_fields,
    _mentions_profile_based_peer_claim,
    _mentions_skax_actor,
    _nested_mapping_keys,
    _non_main_event_product_terms,
    _numeric_token_keys,
    _ordered_anchor_matches,
    _peer_only_issue_product_terms,
    _peer_profile_linkage,
    _polish_frontend_ready_screen_copy,
    _polish_repeated_frontend_phrase,
    _prefer_collective_actor_anchors,
    _prepend_issue_fact_to_evidence,
    _primary_actor_type_for_issue,
    _primary_issue_fact,
    _profile_comparison_phrase,
    _profile_entry_is_referenced,
    _profile_entry_text,
    _profile_has_execution_case,
    _quality_checked_texts,
    _quoted_entity_terms,
    _regex_slot_terms,
    _relationship_only_uncertain,
    _relevant_context_items,
    _relevant_profile_linkage_level,
    _relevant_profile_linkage_level_from_evaluation,
    _relevant_profile_named_terms,
    _repair_customer_role_overstatement,
    _role_interpretation_hints,
    _scope_effect_claim,
    _scope_term_supported,
    _semantic_fingerprint_for_text,
    _should_include_financial_profile_context,
    _skax_action_mode_from_profile_linkage,
    _skax_profile_product_terms,
    _strip_frontend_ready_label,
    _strong_signal_is_only_generic_execution_word,
    _supplier_financial_focus_violation,
    _supplier_role_overstatement_violation,
    _technology_event_domain_line_score,
    _technology_event_domain_phrase,
    _technology_event_name_phrase,
    _technology_event_scale_phrase,
    _used_context_layers,
    _user_strategy_issue_scope,
    _workflow_execution_business_terms,
)
from src.agents.strategic_insight.si_base2 import (  # noqa: F401
    _action_plan_issue_terms,
    _checkpoint_hint_from_action_plan,
    _evidence_sentence_has_issue_anchor,
    _frontend_ready_actionable_signal_level,
    _frontend_ready_claim_violations,
    _frontend_ready_evidence_sentence_anchor_violation,
    _frontend_ready_financial_interpretation_overlap_violation,
    _frontend_ready_key_business_depth_violation,
    _frontend_ready_key_sentence_direction_violation,
    _frontend_ready_peer_product_as_skax_basis_violation,
    _frontend_ready_required_violations,
    _frontend_ready_sentence_restates_issue_fact,
    _frontend_ready_skax_action_mode_violation,
    _frontend_ready_with_issue_evidence_anchors,
    _has_concrete_profile_term,
    _has_direct_peer_action_signal,
    _has_displayable_frontend_ready,
    _has_integrated_issue_candidate_anchor_signal,
    _has_integrated_text_candidate_signal,
    _has_moderate_actionable_issue_signal,
    _is_valid_integrated_issue,
    _is_weak_surface_integrated_issue,
    _issue_execution_slot_diagnostics,
    _issue_fact_line_for_frontend_evidence,
    _profile_area_phrase,
    _profile_based_downgrade_diagnostics,
    _relevant_profile_area_names,
    _safe_business_line_mapping,
    _slot_terms,
    _solution_term_supported_by_evidence,
    _specific_event_anchors_for_frontend,
    _stock_line_without_business_link,
    _stock_market_watch_only_issue,
    _strategic_generation_skip_decision,
    _watch_only_evidence_summary,
    _weak_analysis_statement,
    _weak_hiring_watch_only_decision,
)
from src.agents.strategic_insight.text_predicates import (  # noqa: F401
    _anchor_norm,
    _anchor_tokens,
    _clamp_float,
    _dedupe_keep_order,
    _ensure_sentence,
    _has_korean_final_consonant,
    _matches_any_pattern,
    _natural_join,
    _numeric_token_key,
    _optional_str,
    _sentence_count,
    _short_fact_clause,
    _split_sentences,
    _strip_article_style_lead,
    _text_has_anchor_term,
    _with_korean_object_particle,
)
from src.agents.strategic_insight.utils import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    _choice,
    _has_final_consonant,
    _int_list,
    _json_dict,
    _jsonish_list,
    _parse_json_loose,
    _string_list,
    _with_particle,
)


def _ensure_reasoning_debug_fields(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    """Keep reasoning/debug fields stable on LLM, repair, invalid, and fallback paths."""
    out = dict(result or {})
    out["issue_understanding"] = _normalize_issue_understanding(
        out.get("issue_understanding"),
        integrated_issue=integrated_issue,
    )
    out["profile_linkage"] = _normalize_llm_profile_linkage(out.get("profile_linkage"))
    out["skax_response_linkage"] = _normalize_skax_response_linkage(
        out.get("skax_response_linkage")
    )
    out["claim_strength"] = _choice(
        out.get("claim_strength"),
        {"strong", "moderate", "cautious"},
        "cautious",
    )
    out["grounding_summary"] = _normalize_grounding_summary(
        out.get("grounding_summary"),
        integrated_issue=integrated_issue,
    )
    return out


def _empty_strategic_insight(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    cluster_metadata: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    analysis = {
        "is_valid_analysis": False,
        "analysis_scope": "peer_and_industry",
        "analysis_summary": "",
        "strategic_meaning": [],
        "market_signal": "",
        "impact_level": "low",
        "impact_reason": "",
        "risk_or_opportunity": "neutral",
        "confidence": 0.0,
        "reason": reason,
    }
    implication = {
        "is_valid_implication": False,
        "implication_scope": "peer_and_skax",
        "peer_implication": {
            "company_id": str(integrated_issue.get("main_company") or ""),
            "company_name_ko": "",
            "peer_meaning": "",
            "capability_change": "",
            "sourced_evidence_ids": [],
        },
        "skax_implication": {
            "why_important": "",
            "potential_impact": "",
            "opportunities": [],
            "threats": [],
            "recommended_actions": [],
            "business_line_mapping": [],
        },
        "follow_up_questions": [],
        "watch_points": [],
        "confidence": 0.0,
        "evidence_label": "insufficient",
        "provenance": {
            "generator": "StrategicInsightAgent",
            "prompt_version": _PROMPT_VERSION,
            "model": _LLM_MODEL,
            "model_config": _llm_model_config_diagnostics(),
            "used_fact_ids": [],
            "used_context_layers": [],
            "run_at": datetime.now(UTC).isoformat(),
        },
    }
    result = {
        "is_valid_strategic_insight": False,
        "analysis": analysis,
        "implication": implication,
        "sentence_grounding": {
            "schema_version": "sentence-grounding-v1",
            "generator": "StrategicInsightAgent",
            "entries": [],
            "summary": {
                "entry_count": 0,
                "fact_grounded_count": 0,
                "profile_grounded_count": 0,
                "ungrounded_paths": [],
            },
        },
    }
    return _ensure_reasoning_debug_fields(result, integrated_issue=integrated_issue)


def _action_text_has_grounding_axis_signal(text: str, grounding: str) -> bool:
    output_axis_terms = {
        term
        for term in _raw_normalized_terms(text)
        if len(term) >= 2
        and not _is_low_signal_content_token(term)
        and not _is_generic_business_term(term)
    }
    grounding_axis_terms = {
        term
        for term in _raw_normalized_terms(grounding)
        if len(term) >= 2
        and not _is_low_signal_content_token(term)
        and not _is_generic_business_term(term)
    }
    return bool(output_axis_terms & grounding_axis_terms)


def _relationship_grounding_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    integrated_evidence_text: str,
) -> str:
    if not text or not _RELATIONSHIP_PATTERN.search(text):
        return ""
    if _is_follow_up_or_watch_field(label):
        return ""
    if not _has_relationship_grounding(integrated_issue, integrated_evidence_text):
        return "IntegratedIssue 에 없는 협업/파트너십 계열 관계 표현을 사용했습니다."
    if _relationship_only_uncertain(integrated_evidence_text) and not _UNCERTAINTY_PATTERN.search(
        text
    ):
        return "검토/구상/가능성 단계의 관계를 확정 실행처럼 표현했습니다."
    return ""


def _off_topic_application_product_violation(
    text: str,
    *,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not integrated_issue:
        return ""
    main_context = _main_issue_context_text(integrated_issue)
    off_topic_terms = _non_main_event_product_terms(integrated_issue)
    for term in off_topic_terms:
        if len(term) < 2:
            continue
        if not re.search(re.escape(term), text, flags=re.IGNORECASE):
            continue
        if re.search(re.escape(term), main_context, flags=re.IGNORECASE):
            continue
        return (
            "현재 클러스터의 핵심 사건이 아닌 부가 적용 사례의 제품/서비스명을 "
            "대응방향에 사용했습니다. 메인 사건의 대상 사업·시스템 기준으로 낮춰야 합니다."
        )
    return ""


def _evidence_scoped_business_claim_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any] | None,
    profile_context: dict[str, Any] | None,
) -> str:
    if not text or not label.startswith("skax_implication."):
        return ""
    text_value = str(text or "")
    if "솔루션" in text_value and not (
        _action_text_has_internal_strategy_checkpoint(text_value)
        or _high_signal_issue_overlap_count(text_value, integrated_issue or {}) >= 2
    ):
        return (
            "SK AX 영향/대응을 일반 솔루션 표현으로 썼습니다. 현재 사건에서 확인된 "
            "전환 범위, 업무 영향도, 검증 기준 중심으로 낮춰야 합니다."
        )
    solution_scope_violation = _solution_term_scope_violation(
        text_value,
        integrated_issue=integrated_issue,
    )
    if solution_scope_violation:
        return solution_scope_violation
    if re.search(
        r"성공\s*사례|성공\s*레퍼런스|구축\s*경험|운영\s*역량",
        text_value,
    ) and not _profile_has_execution_case(
        profile_context,
        integrated_issue=integrated_issue,
    ):
        return (
            "ProfileContext에 실행/구축 사례 근거가 없는데 성공 사례·구축 경험·운영 역량을 "
            "사용했습니다. SK AX가 내부적으로 점검할 범위, 책임, 검증 기준, 운영 조건 중심으로 "
            "낮춰야 합니다."
        )
    return ""


def _scope_expansion_guard_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> str:
    if not text or _is_follow_up_or_watch_field(label):
        return ""
    value = str(text or "")
    event_text = _integrated_grounding_text(integrated_issue)
    context_text = "\n".join(
        [
            event_text,
            _json_dumps(profile_linkage_evaluation or {}),
        ]
    )
    event_lower = event_text.casefold()
    context_lower = context_text.casefold()

    if _has_global_scope(value):
        if _scope_effect_claim(value) and not _has_global_scope(event_lower):
            return (
                "글로벌/해외 범위의 강화·확장·영향 표현을 현재 사건 효과처럼 사용했습니다. "
                "원문에 글로벌/해외 근거가 없으면 국가 단위, 국내, 해당 사업 범위로 낮춰야 합니다."
            )
        if not (_has_global_scope(event_lower) or _has_global_scope(context_lower)):
            return (
                "글로벌/해외 시장 범위를 사용했지만 IntegratedIssue 또는 "
                "관련 프로필 근거가 없습니다. "
                "현재 사건의 실제 시장 범위로 낮춰야 합니다."
            )

    if _has_public_private_scope(value) and not _has_public_private_scope_support(context_lower):
        return (
            "공공과 민간 양쪽으로 범위를 넓혔지만 양쪽 고객군 근거가 모두 확인되지 않습니다. "
            "확인된 고객군 또는 사업 범위로 낮춰야 합니다."
        )

    if _has_all_industry_scope(value) and not _has_all_industry_scope(context_lower):
        return (
            "전 산업/산업 전반 범위를 사용했지만 현재 사건 또는 프로필 근거가 부족합니다. "
            "확인된 산업/고객군 범위로 낮춰야 합니다."
        )

    if _has_status_strength_claim(value) and not _has_status_strength_event_support(event_lower):
        return (
            "지위 강화나 역량 검증처럼 강한 표현을 썼지만 "
            "선정, 수주, 공식 계약, 실행 근거 등 직접 근거가 부족합니다. "
            "관찰 신호나 연결 사례 수준으로 낮춰야 합니다."
        )

    if (
        label.startswith(("analysis.", "peer_implication."))
        and (
            _has_status_strength_claim(value)
            or _has_broad_expansion_claim(value)
            or _has_effectiveness_claim(value)
        )
        and (
            _relevant_profile_linkage_level_from_evaluation(
                profile_linkage_evaluation,
                scope="peer",
            )
            or _relevant_profile_linkage_level(
                profile_context,
                integrated_issue=integrated_issue,
                scope="peer",
            )
        )
        in {"high", "medium"}
        and not _has_concrete_profile_term(
            value,
            profile_context=profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
    ):
        return (
            "피어 프로필 기반 강한 해석 표현을 사용했지만 문장 안에 현재 사건과 맞는 "
            "구체 프로필 사업영역/역량명이 보이지 않습니다. 피어의 기존 역량과 현재 사건의 "
            "접점을 명시하거나 관찰 신호 수준으로 낮춰야 합니다."
        )

    if label.startswith(("analysis.", "peer_implication.")) and (
        _has_status_strength_claim(value)
        or _has_broad_expansion_claim(value)
        or _has_attention_growth_claim(value)
    ):
        peer_linkage_level = _relevant_profile_linkage_level_from_evaluation(
            profile_linkage_evaluation,
            scope="peer",
        ) or _relevant_profile_linkage_level(
            profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
        has_issue_connection = _high_signal_issue_overlap_count(value, integrated_issue) >= 2
        has_profile_connection = _has_concrete_profile_term(
            value,
            profile_context=profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
        if (
            label == "analysis.impact_reason"
            and has_issue_connection
            and _has_status_strength_event_support(event_lower)
        ):
            return ""
        attention_supported = not _has_attention_growth_claim(value) or re.search(
            r"관심|주목",
            event_lower,
        )
        if not (
            has_issue_connection
            and has_profile_connection
            and peer_linkage_level in {"high", "medium"}
            and attention_supported
        ):
            return (
                "피어 시사점이 입지 강화/영역 확장/관심 반영 같은 강한 표현을 사용했지만 "
                "현재 사건의 구체 사실과 피어 프로필 접점이 함께 보이지 않습니다. "
                "사실-프로필-사업적 의미가 연결되도록 쓰거나 관찰 신호 수준으로 낮춰야 합니다."
            )

    if _has_broad_expansion_claim(value):
        scope = "skax" if label.startswith("skax_implication") else "peer"
        linkage_level = _relevant_profile_linkage_level_from_evaluation(
            profile_linkage_evaluation,
            scope=scope,
        ) or _relevant_profile_linkage_level(
            profile_context,
            integrated_issue=integrated_issue,
            scope=scope,
        )
        if (
            linkage_level not in {"high", "medium"}
            or not _has_expansion_support(event_lower)
            or _high_signal_issue_overlap_count(value, integrated_issue) < 2
        ):
            return (
                "사업영역/서비스 확장 표현을 사용했지만 현재 사건과 관련 프로필의 연결 또는 "
                "범위 확대 근거가 충분하지 않습니다. 연결 사례나 참여 기반처럼 "
                "강도를 낮춰야 합니다."
            )

    if _has_attention_growth_claim(value) and not re.search(r"관심|주목", event_lower):
        return (
            "관심 증가/주목 같은 시장 반응 표현을 원문 근거 없이 사용했습니다. "
            "확인된 사업, 수요 신호, 비교 기준 변화로 낮춰야 합니다."
        )

    if _has_effectiveness_claim(value):
        scope = "skax" if label.startswith("skax_implication") else "peer"
        linkage_level = _relevant_profile_linkage_level_from_evaluation(
            profile_linkage_evaluation,
            scope=scope,
        ) or _relevant_profile_linkage_level(
            profile_context,
            integrated_issue=integrated_issue,
            scope=scope,
        )
        if linkage_level not in {"high", "medium"} or not _has_effect_scope(value):
            return (
                "긍정적 영향·경쟁력 강화·운영 효율성 향상 같은 효과성 표현에 "
                "현재 사건, 관련 프로필 역량, 기대효과 범위가 함께 보이지 않습니다. "
                "관찰 신호나 검증 계기 수준으로 낮춰야 합니다."
            )

    return ""


def _solution_term_scope_violation(
    text: str,
    *,
    integrated_issue: dict[str, Any] | None,
) -> str:
    if not integrated_issue or "솔루션" not in str(text or ""):
        return ""
    evidence_text = _integrated_grounding_text(integrated_issue)
    evidence_terms = _evidence_scope_terms(evidence_text)
    if not evidence_terms:
        return ""
    for match in re.finditer(r"([가-힣A-Za-z0-9&+·/_\s-]{2,56})\s*솔루션", str(text or "")):
        phrase = match.group(1)
        phrase_terms = _evidence_scope_terms(phrase)
        unsupported_terms = [
            term
            for term in phrase_terms
            if _is_claim_scope_term(term)
            and not _scope_term_supported(
                term, evidence_terms=evidence_terms, evidence_text=evidence_text
            )
        ]
        if unsupported_terms:
            return (
                "대응방향의 솔루션명이 현재 IntegratedIssue 근거 범위를 벗어났습니다. "
                f"근거 없는 용어: {', '.join(unsupported_terms[:3])}. "
                "현재 사건의 대상 시스템/전환 범위/검증 기준 중심 표현으로 낮춰야 합니다."
            )
    return ""


def _scrub_failed_output(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _scrub_failed_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [
            cleaned for item in value if (cleaned := _scrub_failed_output(item)) not in ("", [], {})
        ]
    if isinstance(value, str):
        return "" if _contains_high_risk_unsupported_claim(value) else value
    return value


def _event_based_recommended_actions(
    integrated_issue: dict[str, Any],
    *,
    action_artifact_plan: dict[str, Any] | None = None,
) -> list[str]:
    """Deprecated: final action copy must come from LLM repair, not templates."""
    del integrated_issue
    del action_artifact_plan
    return []


def _counterparty_guard_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> bool:
    value_text = str(text or "").strip()
    if not value_text:
        return False
    return bool(
        _supplier_role_overstatement_violation(
            value_text,
            integrated_issue=integrated_issue,
        )
        or _supplier_financial_focus_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        or _counterparty_capability_overclaim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        or _counterparty_role_action_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
    )


def _event_based_analysis_field(key: str, *, integrated_issue: dict[str, Any]) -> str:
    fact = _primary_issue_fact(integrated_issue)
    if key == "analysis_summary":
        return _event_based_analysis_summary(integrated_issue)
    if key == "market_signal":
        return _event_based_market_signal(integrated_issue)
    if key == "impact_reason":
        return _event_based_impact_reason(integrated_issue)
    if key == "reason":
        return (
            f"{fact} 이 사실을 기준으로 해석하되, 계약 상대방의 수행·운영 역할은 "
            "원문에서 확인되는 범위로만 제한했습니다."
        )
    return fact


def _event_based_analysis_summary(integrated_issue: dict[str, Any]) -> str:
    fact = _primary_issue_fact(integrated_issue)
    target = _main_company_display(integrated_issue)
    if _main_company_is_customer_or_buyer(integrated_issue) and target:
        return (
            f"{fact} {target}는 원문상 계약 상대방으로 확인되며, 이 이슈는 계약 "
            "대상 시스템·범위·기간이 구체화된 사건으로 해석하는 것이 안전합니다."
        )
    return fact


def _profile_linked_analysis_summary(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    fact = _primary_issue_fact(integrated_issue)
    subject = _issue_subject_phrase(integrated_issue)
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if subject and profile_phrase:
        return (
            f"{fact} 이 사건은 피어 프로필의 {profile_phrase} 맥락이 "
            f"{_with_particle(subject, '과', '와')} 연결되는 신호로 해석할 수 있습니다."
        )
    return _event_based_analysis_summary(integrated_issue)


def _event_based_market_signal(integrated_issue: dict[str, Any]) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    duration = _contract_duration_phrase(integrated_issue)
    scale = _contract_scale_phrase(integrated_issue)
    details = " ".join(item for item in (scale, duration) if item)
    if subject and details:
        return f"{subject}이 실제 계약 단위에서 확인됐고, {details}이 함께 제시됐습니다."
    if subject:
        return f"{subject}이 실제 계약 단위에서 확인됐습니다."
    return "현재 사건에서 대상 시스템과 계약 범위가 구체화된 신호가 확인됩니다."


def _event_based_impact_reason(integrated_issue: dict[str, Any]) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    target = _main_company_display(integrated_issue)
    if _main_company_is_customer_or_buyer(integrated_issue) and target and subject:
        return (
            f"{target}의 역할은 계약 상대방으로 확인되는 수준이지만, {subject}의 "
            "계약 범위와 기간이 제시되어 유사 사업에서 비교할 전환 범위와 일정 기준을 "
            "관찰할 수 있습니다."
        )
    if subject:
        return (
            f"{subject}의 계약 범위와 기간이 제시되어 유사 사업의 비교 기준을 관찰할 수 있습니다."
        )
    return "현재 근거에서 계약 범위와 대상 시스템이 확인되어 후속 비교 기준을 관찰할 수 있습니다."


def _event_based_strategic_meaning_candidates(integrated_issue: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    fact = _primary_issue_fact(integrated_issue)
    subject = _issue_subject_phrase(integrated_issue)
    target = _main_company_display(integrated_issue)
    if fact:
        candidates.append(fact)
    if subject:
        candidates.append(
            f"{subject}이 기사에서 확인된 만큼, 이 이슈는 대상 시스템의 전환 범위, "
            "업무 영향도, 운영 안정성 기준이 함께 드러난 사건입니다."
        )
        candidates.append(
            f"유사 사업에서는 {subject}의 기능 구현 여부만이 아니라 기존 시스템과의 "
            "연계 방식, 전환 일정, 장애 대응 기준까지 비교 기준으로 제시될 수 있습니다."
        )
    scale = _contract_scale_phrase(integrated_issue)
    duration = _contract_duration_phrase(integrated_issue)
    if scale or duration:
        candidates.append(
            " ".join(
                part
                for part in (
                    scale,
                    duration,
                    (
                        "이 함께 확인되어 단기 개선보다 일정 규모의 업무 시스템 "
                        "전환 과제로 해석할 수 있습니다."
                    ),
                )
                if part
            )
        )
    if _main_company_is_customer_or_buyer(integrated_issue) and target:
        candidates.append(
            f"{target}는 계약 상대방으로 확인되지만, 최종 발주자 여부나 수행 범위는 "
            "기사에서 확인된 계약 범위와 별도 기준으로 분리해 해석할 수 있습니다."
        )
    return [item for item in candidates if item]


def _profile_linked_capability_change(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    subject = _issue_subject_phrase(integrated_issue) or "현재 사건"
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if profile_phrase:
        return (
            f"확인된 변화는 역량 확장 자체가 아니라 {subject}의 구축 범위와 추진 구조가 "
            f"피어 프로필의 {profile_phrase} 맥락과 연결된다는 점입니다. 유사 사업에서는 "
            "구축 범위, 운영 체계, 단계별 일정이 함께 비교될 수 있습니다."
        )
    return _event_based_capability_change(integrated_issue)


def _event_based_peer_meaning(
    *,
    integrated_issue: dict[str, Any],
    peer: dict[str, Any],
) -> str:
    peer_name = str(peer.get("company_name_ko") or peer.get("company_id") or "타깃 피어").strip()
    fact = _primary_issue_fact(integrated_issue)
    if _main_company_is_customer_or_buyer(integrated_issue):
        return (
            f"{fact} {peer_name}는 원문상 계약 상대방으로 확인되지만, 최종 발주자 "
            "여부와 수행 범위는 별도 확인 축으로 남습니다. 따라서 피어사 "
            "관점에서는 역할 확장보다 금융권 핵심 시스템 전환 과제와 연결된 관찰 "
            "신호로 해석하는 것이 적절합니다."
        )
    return (
        f"{fact} 현재 사건과 직접 맞는 피어 프로필 접점이 충분하지 않아, "
        "이 신호는 사건 기반 1차 해석으로 보는 것이 안전합니다."
    )


def _event_based_capability_change(integrated_issue: dict[str, Any]) -> str:
    subject = _issue_subject_phrase(integrated_issue) or "확인된 사업"
    duration = _contract_duration_phrase(integrated_issue)
    duration_text = f" {duration}도 함께 확인됩니다." if duration else ""
    return (
        f"확인된 변화는 피어사의 확정된 역할 변화가 아니라 {subject}의 대상 시스템과 "
        f"계약 범위가 구체화된 점입니다.{duration_text} 유사 사업에서는 전환 범위, "
        "업무 영향도, 일정 기준을 함께 비교해야 한다는 신호로 볼 수 있습니다."
    )


def _hard_quality_violation_for_text(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> bool:
    value_text = str(text or "").strip()
    if not value_text:
        return False
    integrated_evidence_text = _integrated_grounding_text(integrated_issue)
    if any(
        _has_unsupported_pattern(value_text, pattern, evidence_text=integrated_evidence_text)
        for pattern in _UNSUPPORTED_CLAIM_PATTERNS
    ):
        return True
    return bool(
        _relationship_grounding_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            integrated_evidence_text=integrated_evidence_text,
        )
        or _supplier_role_overstatement_violation(
            value_text,
            integrated_issue=integrated_issue,
        )
        or _supplier_financial_focus_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        or _counterparty_capability_overclaim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        or _counterparty_role_action_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
    )


def _remove_ungrounded_numeric_tokens(
    text: str,
    *,
    integrated_issue: dict[str, Any],
) -> str:
    grounded_keys = _grounded_numeric_keys_for_issue(integrated_issue)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0).strip()
        if _numeric_token_key(token) in grounded_keys:
            return token
        return "근거에 언급된 수치"

    out = _NUMERIC_TOKEN_PATTERN.sub(replace, str(text or ""))
    out = re.sub(r"근거에 언급된 수치\s*%?\s*(이상|내외|가량|정도)", "근거에 언급된 규모", out)
    out = re.sub(r"근거에 언급된 수치\s*이상의\s*규모", "근거에 언급된 규모", out)
    out = re.sub(r"근거에 언급된 수치\s*규모", "근거에 언급된 규모", out)
    return out


def _fallback_matched_profile_areas(linkage: dict[str, Any]) -> list[dict[str, Any]]:
    areas = _jsonish_list(linkage.get("matched_business_areas"))
    capabilities = _string_list(linkage.get("matched_capabilities"), max_items=5)
    out: list[dict[str, Any]] = []
    for area in areas[:5]:
        if not isinstance(area, dict):
            continue
        name = str(area.get("name") or area.get("profile_area_name") or "").strip()
        business_line = str(area.get("business_line") or "").strip()
        raw_business_area = str(area.get("business_area") or "").strip()
        raw_specificity = str(area.get("specificity_level") or "").strip()
        business_area = raw_business_area or (name if raw_specificity == "business_area" else "")
        matched_capabilities = (
            _string_list(
                area.get("matched_capabilities") or area.get("capabilities"),
                max_items=8,
            )
            or capabilities[:3]
        )
        matched_products = _string_list(
            area.get("matched_products_or_services") or area.get("products_or_services"),
            max_items=8,
        )
        profile_capability = ", ".join(matched_capabilities[:3])
        reason = _fallback_area_reason(area, linkage)
        source_ref = _fallback_first_source_ref(area, linkage)
        specificity_level = _choice(
            area.get("specificity_level"),
            {
                "product_or_service",
                "core_capability",
                "business_area",
                "business_line",
                "profile_context",
            },
            (
                "product_or_service"
                if matched_products
                else "core_capability"
                if matched_capabilities
                else "business_area"
                if business_area
                else "business_line"
                if business_line
                else "profile_context"
            ),
        )
        if name or profile_capability or reason or source_ref:
            out.append(
                {
                    "business_line": business_line,
                    "business_area": business_area,
                    "profile_area_name": name,
                    "profile_capability": profile_capability,
                    "matched_capabilities": matched_capabilities,
                    "matched_products_or_services": matched_products,
                    "matched_issue_terms": _string_list(
                        area.get("matched_issue_terms") or area.get("matched_terms"),
                        max_items=12,
                    ),
                    "evidence_text": str(area.get("evidence_text") or "").strip(),
                    "why_relevant_to_issue": reason,
                    "profile_source_ref": source_ref,
                    "specificity_level": specificity_level,
                }
            )
    if not out:
        for capability in capabilities[:3]:
            out.append(
                {
                    "business_line": "",
                    "business_area": "",
                    "profile_area_name": capability,
                    "profile_capability": capability,
                    "matched_capabilities": [capability],
                    "matched_products_or_services": [],
                    "matched_issue_terms": _string_list(
                        linkage.get("matched_terms"),
                        max_items=12,
                    ),
                    "evidence_text": "",
                    "why_relevant_to_issue": str(linkage.get("reason") or "").strip(),
                    "profile_source_ref": _fallback_first_source_ref({}, linkage),
                    "specificity_level": "core_capability",
                }
            )
    return out[:5]


def _fallback_area_reason(area: dict[str, Any], linkage: dict[str, Any]) -> str:
    matched_terms = _string_list(area.get("matched_terms"), max_items=6)
    if matched_terms:
        return "현재 이슈의 " + ", ".join(matched_terms[:4]) + " 신호와 연결됩니다."
    return str(linkage.get("reason") or "").strip()


def _fallback_first_source_ref(area: dict[str, Any], linkage: dict[str, Any]) -> str:
    refs = _jsonish_list(area.get("source_refs")) or _jsonish_list(
        linkage.get("matched_source_refs")
    )
    if not refs:
        return ""
    first = refs[0]
    if isinstance(first, dict):
        return str(first.get("source_ref") or first.get("id") or first.get("url") or "").strip()
    return str(first).strip()


def _fallback_internal_checkpoints(
    *,
    focus_terms: list[str],
    action_artifact_plan: dict[str, Any],
) -> list[str]:
    checkpoint = _checkpoint_hint_from_action_plan(action_artifact_plan)
    out = []
    for term in focus_terms[:3]:
        out.append(f"{term} 관련 {checkpoint}")
    if not out:
        out.append(checkpoint)
    return out


def _fallback_recommended_focus(
    *,
    focus_terms: list[str],
    linkage: dict[str, Any],
) -> list[str]:
    level = _choice(linkage.get("linkage_level"), {"high", "medium", "low", "none"}, "none")
    if focus_terms and level in {"high", "medium"}:
        return [f"{term}와 연결된 사업 적용 방향과 협력 필요 조건" for term in focus_terms[:3]]
    if focus_terms:
        return [f"{term} 관련 후속 근거 확인과 보수적 대응 범위 점검" for term in focus_terms[:3]]
    return ["후속 근거 확인과 대응 범위 점검"]


def _fallback_used_profile_refs(profile_linkage_evaluation: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for linkage in [
        *_jsonish_list(profile_linkage_evaluation.get("peer_linkages")),
        profile_linkage_evaluation.get("skax_linkage"),
    ]:
        if not isinstance(linkage, dict):
            continue
        refs.extend(_string_list(linkage.get("matched_source_refs"), max_items=5))
        for area in _jsonish_list(linkage.get("matched_business_areas")):
            if isinstance(area, dict):
                refs.extend(_string_list(area.get("source_refs"), max_items=5))
    return list(dict.fromkeys(refs))[:12]


def _fallback_peer_meaning_without_summary_repeat(
    current: str,
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    has_peer_profile_link: bool,
) -> str:
    subject = _issue_subject_phrase(integrated_issue) or "현재 사건"
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if has_peer_profile_link and profile_phrase:
        return (
            f"피어 프로필에서는 {profile_phrase}가 "
            f"{_with_particle(subject, '과', '와')} 연결되는 배경으로 확인됩니다. "
            "따라서 이 신호는 역할 확장이나 성과를 단정하기보다, 기존 사업 맥락이 "
            "현재 대형 과제와 만나는 관찰 지점으로 해석하는 것이 안전합니다."
        )
    if current and _primary_issue_fact(integrated_issue).rstrip(".") not in current:
        return current
    return (
        f"{subject}은 피어사의 확정된 역할 변화보다 현재 사건의 대상 사업, "
        "추진 범위, 후속 확인 기준이 구체화된 관찰 신호로 보는 것이 안전합니다."
    )


def _fallback_skax_implication(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_level: str,
    action_artifact_plan: dict[str, Any],
) -> dict[str, Any]:
    subject = _issue_subject_phrase(integrated_issue) or _primary_issue_fact(integrated_issue)
    peer_profile = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    skax_profile = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="skax",
    )
    profile_comparison = _profile_comparison_phrase(
        peer_profile=peer_profile,
        skax_profile=skax_profile,
        linkage_level=profile_linkage_level,
    )
    actions = _fallback_internal_actions(
        subject=subject,
        profile_comparison=profile_comparison,
        action_artifact_plan=action_artifact_plan,
    )
    return {
        "why_important": (
            f"{_with_particle(subject, '은', '는')} 유사 고객군/유사 사업에서 "
            "피어 신호가 SK AX의 사업 방향에 어떤 의미를 갖는지 "
            "살펴볼 만한 사건입니다."
        ),
        "potential_impact": (
            f"유사 사업에서는 {subject}에서 확인된 범위와 검증 기준이 "
            f"사업 판단의 참고점이 될 수 있으므로 SK AX는 {profile_comparison}을 "
            "바탕으로 준비 범위를 정교화할 수 있습니다."
        ),
        "opportunities": [],
        "threats": [],
        "recommended_actions": actions,
        "business_line_mapping": _safe_business_line_mapping(
            profile_context,
            integrated_issue=integrated_issue,
        ),
    }


def _fallback_internal_actions(
    *,
    subject: str,
    profile_comparison: str,
    action_artifact_plan: dict[str, Any],
) -> list[str]:
    issue_term = _compact_issue_term(subject)
    checkpoint_hint = _checkpoint_hint_from_action_plan(action_artifact_plan)
    return [
        (
            f"SK AX는 {_with_particle(issue_term, '과', '와')} 유사한 사업에서 "
            f"{profile_comparison}을 참고해 "
            "입력 사건에서 확인된 변화를 후속 준비 범위에 반영할 수 있습니다."
        ),
        (
            f"SK AX는 {issue_term} 대응 시 "
            f"{_with_particle(checkpoint_hint, '을', '를')} 단순 확인 항목이 아니라 "
            "후속 사업 범위를 정교화하는 기준으로 활용할 수 있습니다."
        ),
    ]


def _fallback_watch_points(integrated_issue: dict[str, Any]) -> list[str]:
    subject = _issue_subject_phrase(integrated_issue) or "현재 사업"
    return [
        f"{subject}의 후속 협약, 구축 완료, 서비스 개시 일정이 구체화되는지 확인합니다.",
        "피어사의 수행 범위와 추가 참여 구조가 원문 근거로 확인되는지 모니터링합니다.",
    ]
