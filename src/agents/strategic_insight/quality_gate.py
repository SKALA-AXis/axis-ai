"""strategic_insight quality_gate — extracted from facade (move-only)."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

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
from src.agents.strategic_insight.fallback0 import (  # noqa: F401
    _action_text_has_grounding_axis_signal,
    _counterparty_guard_violation,
    _empty_strategic_insight,
    _ensure_reasoning_debug_fields,
    _event_based_analysis_field,
    _event_based_analysis_summary,
    _event_based_capability_change,
    _event_based_impact_reason,
    _event_based_market_signal,
    _event_based_peer_meaning,
    _event_based_recommended_actions,
    _event_based_strategic_meaning_candidates,
    _evidence_scoped_business_claim_violation,
    _fallback_area_reason,
    _fallback_first_source_ref,
    _fallback_internal_actions,
    _fallback_internal_checkpoints,
    _fallback_matched_profile_areas,
    _fallback_peer_meaning_without_summary_repeat,
    _fallback_recommended_focus,
    _fallback_skax_implication,
    _fallback_used_profile_refs,
    _fallback_watch_points,
    _hard_quality_violation_for_text,
    _off_topic_application_product_violation,
    _profile_linked_analysis_summary,
    _profile_linked_capability_change,
    _relationship_grounding_violation,
    _remove_ungrounded_numeric_tokens,
    _scope_expansion_guard_violation,
    _scrub_failed_output,
    _solution_term_scope_violation,
)
from src.agents.strategic_insight.fallback1 import (  # noqa: F401
    _fallback_matched_skax_areas,
    _fallback_profile_linkage_payload,
    _fallback_quality_repair,
    _fallback_skax_response_linkage_payload,
    _mark_quality_gate_failed,
    _minimal_quality_guard,
    _profile_linked_analysis_field,
    _profile_linked_impact_reason,
    _profile_linked_market_signal,
    _profile_linked_peer_meaning,
    _profile_linked_strategic_meanings,
    _recommended_action_quality_violation,
    _repair_result_numeric_grounding,
    _safe_event_based_strategic_meanings,
    _safe_quality_gate_violation_text,
    _two_section_fact_based_fallback,
    _with_fallback_linkage_payloads,
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


def _quality_gate_violations(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
    action_artifact_plan: dict[str, Any] | None = None,
) -> list[str]:
    profile_linkage_evaluation = profile_linkage_evaluation or _build_profile_linkage_evaluation(
        integrated_issue=integrated_issue,
        classification={},
        profile_context=profile_context,
    )
    action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
        integrated_issue=integrated_issue,
        classification={},
        profile_linkage_evaluation=profile_linkage_evaluation,
    )
    integrated_evidence_text = _integrated_grounding_text(integrated_issue)
    grounded_numeric_keys = _grounded_numeric_keys_for_issue(integrated_issue)
    violations: list[str] = []
    violations.extend(
        _frontend_ready_required_violations(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
    )
    violations.extend(
        _frontend_ready_claim_violations(
            result,
            integrated_issue=integrated_issue,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
    )
    for label, value_text in _quality_checked_texts(result):
        if "quality_gate_failed:" in value_text:
            value_text = value_text.split("| quality_gate_failed:", 1)[0].strip()
        for token in _NUMERIC_TOKEN_PATTERN.findall(value_text):
            token_text = str(token or "").strip()
            if token_text and _numeric_token_key(token_text) not in grounded_numeric_keys:
                violations.append(
                    f"{label}: fact_basis/key_numbers/representative_sources에 없는 "
                    f"수치 `{token_text}`를 사용했습니다."
                )
        if not _is_follow_up_or_watch_field(label):
            for pattern in _UNSUPPORTED_CLAIM_PATTERNS:
                if _has_unsupported_pattern(
                    value_text,
                    pattern,
                    evidence_text=integrated_evidence_text,
                ):
                    violations.append(
                        f"{label}: 입력 근거 없이 `{pattern}` 계열 표현을 사용했습니다."
                    )
        relation_violation = _relationship_grounding_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            integrated_evidence_text=integrated_evidence_text,
        )
        if relation_violation:
            violations.append(f"{label}: {relation_violation}")
        supplier_role_violation = _supplier_role_overstatement_violation(
            value_text,
            integrated_issue=integrated_issue,
        )
        if supplier_role_violation:
            violations.append(f"{label}: {supplier_role_violation}")
        supplier_financial_focus_violation = _supplier_financial_focus_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if supplier_financial_focus_violation:
            violations.append(f"{label}: {supplier_financial_focus_violation}")
        counterparty_overclaim_violation = _counterparty_capability_overclaim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if counterparty_overclaim_violation:
            violations.append(f"{label}: {counterparty_overclaim_violation}")
        counterparty_role_violation = _counterparty_role_action_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if counterparty_role_violation:
            violations.append(f"{label}: {counterparty_role_violation}")
        novelty_violation = _business_novelty_overclaim_violation(
            value_text,
            label=label,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if novelty_violation:
            violations.append(f"{label}: {novelty_violation}")
        artifact_plan_violation = _action_artifact_plan_violation(
            value_text,
            label=label,
            action_artifact_plan=action_artifact_plan,
        )
        if artifact_plan_violation:
            violations.append(f"{label}: {artifact_plan_violation}")
        evidence_scoped_claim_violation = _evidence_scoped_business_claim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if evidence_scoped_claim_violation:
            violations.append(f"{label}: {evidence_scoped_claim_violation}")
        scope_expansion_violation = _scope_expansion_guard_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if scope_expansion_violation:
            violations.append(f"{label}: {scope_expansion_violation}")
        action_quality_violation = _recommended_action_quality_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if action_quality_violation:
            violations.append(f"{label}: {action_quality_violation}")
        unsupported_profile_violation = _unsupported_peer_profile_claim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if unsupported_profile_violation:
            violations.append(f"{label}: {unsupported_profile_violation}")
        unsupported_skax_term_violation = _unsupported_skax_profile_term_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if unsupported_skax_term_violation:
            violations.append(f"{label}: {unsupported_skax_term_violation}")
        unsupported_domain_violation = _unsupported_domain_term_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if unsupported_domain_violation:
            violations.append(f"{label}: {unsupported_domain_violation}")
    return list(dict.fromkeys(violations))


def _frontend_ready_action_mechanical_split_violation(sentence: Any) -> str:
    value = re.sub(r"\s+", " ", str(sentence or "").strip())
    if not value:
        return ""
    mechanical_patterns = (
        r"(?:인지|할지|할지부터|여부).{0,40}나눠\s*(?:비교|점검|확인|검토|설명|정리)",
        r"나눠\s*[,，]",
        r"나눠\s*(?:비교|점검|확인|검토|설명|정리)(?:해야\s*한다|할\s*필요가\s*있다)",
        r"분리해\s*(?:비교|점검|확인|검토|설명|정리)(?:해야\s*한다|할\s*필요가\s*있다)",
        r"나눠\s*볼\s*필요가\s*있다",
        r"우선\s*비교할지\s*나눠",
        r"(?:적용\s*범위|운영\s*책임|수행\s*범위|검증\s*기준|고객\s*제안\s*단위)"
        r".{0,35}(?:나눠|분리해|구분해)\s*(?:비교|점검|확인|검토|설명|정리)",
    )
    if any(re.search(pattern, value) for pattern in mechanical_patterns):
        return (
            "선택지를 나열한 뒤 나눠/분리해 비교·점검으로 끝나는 대응방향은 "
            "화면 문장으로 사용할 수 없습니다."
        )
    return ""


def _frontend_ready_action_weak_review_phrase_violation(sentence: Any) -> str:
    value = re.sub(r"\s+", " ", str(sentence or "").strip())
    if not value:
        return ""
    if re.search(r"자체\s*AI\s*역량만\s*앞세우기보다", value):
        return "SK AX의 현재 접근을 평가절하하는 표현은 대응방향에 쓰지 않습니다."
    weak_patterns = (
        r"검토할\s*때",
        r"(?:검토|점검|확인)해야\s*한다\.?$",
        r"(?:검토|점검|확인)할\s*필요가\s*있다\.?$",
    )
    has_weak_review = any(re.search(pattern, value) for pattern in weak_patterns)
    if not has_weak_review:
        return ""
    action_terms = (
        r"설계|확보|연결|구성|제안|구축|운영|검증\s*환경|현장\s*데이터|"
        r"파트너|사업화|실행\s*조건|고객\s*제안"
    )
    if not re.search(action_terms, value):
        return (
            "대응방향이 검토/점검/확인에 머물렀습니다. "
            "SK AX의 후속 움직임이 어떤 형태로 전개되는지까지 써야 합니다."
        )
    return ""


def _frontend_ready_action_specificity_violation(
    block: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> str:
    sentence = str(block.get("sentence") or "")
    evidence = str(block.get("evidence_sentence") or "")
    text = f"{sentence} {evidence}"
    if not text.strip():
        return ""
    issue_anchors = _specific_event_anchors_for_frontend(integrated_issue)
    has_issue_target = not issue_anchors or _text_has_anchor_term(text, issue_anchors)
    basis_pattern = (
        r"범위|기준|비율|수치|고객|대상|기능|권한|보안|연동|시스템|운영|책임|"
        r"구조|조건|리스크|위험|성과|비교|일정|처리|검증|매출|구성|적용|"
        r"역할|계약|협약|거래|모니터링"
    )
    sentence_has_evaluation_basis = bool(re.search(basis_pattern, sentence))
    has_evaluation_basis = bool(
        re.search(
            basis_pattern,
            text,
        )
    )
    generic_action_only = (
        bool(
            re.search(
                r"(점검|검토|확인|모니터링)(해야|할\s*필요|할\s*수|합니다|한다|하십시오)",
                sentence,
            )
        )
        and not sentence_has_evaluation_basis
    )
    if not has_issue_target:
        return "현재 사건에서 나온 점검 대상이 문장에 연결되지 않았습니다."
    if generic_action_only or not has_evaluation_basis:
        return "점검 대상과 판단 기준이 함께 드러나야 합니다."
    if not _has_action_execution_perspective(text):
        return (
            "SK AX 내부 판단 축이 부족합니다. 대응 대상, 판단 기준, 실행 관점이 "
            "함께 드러나야 합니다."
        )
    return ""


def _frontend_ready_action_depth_violation(
    block: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> str:
    sentence = str(block.get("sentence") or "")
    evidence = str(block.get("evidence_sentence") or "")
    text = f"{sentence} {evidence}"
    if not text.strip():
        return ""
    if _is_financial_or_transaction_frontend_block(block, integrated_issue):
        depth_pattern = (
            r"매출\s*(구성|분류|구조)|거래\s*(비중|구조|의존도)|"
            r"내부\s*거래|내부거래|대외\s*(매출|고객|거래)|외부\s*(매출|고객|거래)|"
            r"비교\s*(가능|기준|항목)|분류\s*기준|산정\s*기준|관리\s*지표|"
            r"추적|설명\s*가능|증명|검증"
        )
    else:
        depth_pattern = (
            r"증명|검증|설명\s*가능|비교\s*가능|분류\s*기준|산정\s*기준|관리\s*지표|"
            r"대외\s*(성과|고객|설명)|고객\s*(제안|레퍼런스|수요|확인)|"
            r"고객\s*적용\s*범위|서비스\s*적용\s*범위|서비스형\s*제안|"
            r"레퍼런스|운영\s*(책임|구간|범위|조건)|책임\s*(구간|범위|분담|구조)|"
            r"자체\s*(수행|제공|담당)|외부\s*(협력|파트너|보완|연계)|파트너십|"
            r"수행\s*(범위|책임)|적용\s*가능성|기존\s*시스템\s*접점|"
            r"시스템\s*(연계|연동)|연동\s*범위|현장\s*시스템|"
            r"처리\s*업무\s*기준|업무\s*처리\s*범위|사업성\s*기준|"
            r"후속\s*확인\s*기준|전환\s*가능성|리스크\s*(관리|기준|부담)|"
            r"시장\s*모니터링|매출\s*기여도|단순\s*구축|지속\s*운영형|"
            r"고객\s*제안\s*범위|내부\s*대응\s*우선순위"
        )
    if re.search(depth_pattern, text, flags=re.IGNORECASE):
        return ""
    if (
        _frontend_ready_actionable_signal_level(integrated_issue) in {"strong", "moderate"}
        and _has_moderate_action_decision_axis(text)
        and bool(re.search(r"구분|나눠|비교|검토|확인|판단|분리", text))
    ):
        return ""
    shallow_action_pattern = r"(점검|검토|정리|확인|모니터링)(해야|할\s*필요|할\s*수|합니다|한다)"
    if re.search(shallow_action_pattern, sentence):
        return (
            "대응방향이 단순 점검/정리에서 멈췄습니다. SK AX가 무엇을 "
            "증명·검증하거나 어떤 기준으로 설명 가능하게 만들어야 하는지까지 "
            "드러나야 합니다."
        )
    return (
        "대응방향에 실행 결과 관점이 부족합니다. 대응 대상과 판단 기준뿐 아니라 "
        "입력 사건에 맞는 실행 관점이 필요합니다."
    )


def _frontend_ready_action_choice_violation(block: dict[str, Any]) -> str:
    sentence = str(block.get("sentence") or "")
    evidence = str(block.get("evidence_sentence") or "")
    text = f"{sentence} {evidence}"
    if not text.strip():
        return ""
    choice_pattern = (
        r"자체\s*(수행|제공|담당|역량)|외부\s*(협력|파트너|보완|연계)|파트너십|"
        r"대외\s*(성과|매출|고객|설명)|비캡티브|레퍼런스|"
        r"고객\s*(군|수요|제안|접점|확인)|제안\s*(단위|구조|범위)|"
        r"매출\s*(구성|분류|구조)|성과\s*(지표|관리|추적)|관리\s*지표|추적|"
        r"운영\s*(책임|구간|범위|조건)|책임\s*(구간|범위|분담|구조)|"
        r"서비스\s*(범위|구조|적용|제안)|고객\s*적용\s*범위|"
        r"기존\s*시스템\s*접점|시스템\s*(연계|연동)|"
        r"수요\s*검증|적용\s*가능성|모니터링|분리|구분|나눠|비교|"
        r"탐지|보완\s*조치|사고\s*대응|보안\s*운영|보안\s*관제|"
        r"공급망\s*(운영|관리|서비스|적용)|물류\s*(운영|서비스|적용)"
    )
    if re.search(choice_pattern, text, flags=re.IGNORECASE):
        return ""
    shallow_end_pattern = (
        r"(점검|검토|정리|확인|모니터링)(?:해야\s*한다|해야\s*합니다|할\s*필요가\s*있다|"
        r"할\s*필요가\s*있습니다|할\s*수\s*있다|할\s*수\s*있습니다)\s*[.!。]?$"
    )
    if re.search(shallow_end_pattern, sentence.strip()):
        return (
            "대응방향이 관찰자 톤의 점검 문장으로 끝났습니다. SK AX가 비교할 선택지"
            "를 입력 근거 안에서 제시해야 합니다."
        )
    return (
        "대응방향에 SK AX의 선택지가 부족합니다. 대응 대상과 판단 기준을 넘어서 "
        "입력 사건에 맞는 선택 축을 포함해야 합니다."
    )


def _frontend_ready_role_separation_violation(frontend_ready: dict[str, Any]) -> str:
    key_block = frontend_ready.get("key_implication") or {}
    action_block = frontend_ready.get("suggested_action") or {}
    if not isinstance(key_block, dict) or not isinstance(action_block, dict):
        return ""
    key_sentence = str(key_block.get("sentence") or "").strip()
    action_sentence = str(action_block.get("sentence") or "").strip()
    if not key_sentence or not action_sentence:
        return ""
    key_norm = _anchor_norm(key_sentence)
    action_norm = _anchor_norm(action_sentence)
    if len(key_norm) >= 18 and (key_norm in action_norm or action_norm in key_norm):
        return "시사점과 대응방향 결론문이 서로의 문장을 거의 그대로 반복합니다."

    key_terms = _frontend_ready_role_terms(key_sentence)
    action_terms = _frontend_ready_role_terms(action_sentence)
    if not key_terms or not action_terms:
        return ""
    shared_terms = key_terms & action_terms
    shared_bigrams = _frontend_ready_role_bigrams(key_sentence) & _frontend_ready_role_bigrams(
        action_sentence
    )
    action_unique_terms = action_terms - key_terms
    jaccard = len(shared_terms) / max(len(key_terms | action_terms), 1)
    if len(shared_bigrams) >= 2 and len(action_unique_terms) < 4:
        return (
            "대응방향이 시사점의 핵심 명사 조합을 반복하고 있어, "
            "SK AX가 볼 점검 대상과 판단 기준을 별도로 드러내야 합니다."
        )
    if jaccard >= 0.58 and len(action_unique_terms) < 4:
        return (
            "대응방향이 시사점을 단순히 바꿔 말한 수준입니다. "
            "피어/시장 의미와 SK AX 대응 범위를 분리해야 합니다."
        )
    evidence_violation = _frontend_ready_evidence_role_separation_violation(
        str(key_block.get("evidence_sentence") or ""),
        str(action_block.get("evidence_sentence") or ""),
    )
    if evidence_violation:
        return evidence_violation
    return ""


def _frontend_ready_evidence_role_separation_violation(
    key_evidence: str,
    action_evidence: str,
) -> str:
    if not key_evidence.strip() or not action_evidence.strip():
        return ""
    key_terms = _frontend_ready_role_terms(key_evidence)
    action_terms = _frontend_ready_role_terms(action_evidence)
    if not key_terms or not action_terms:
        return ""
    shared_bigrams = _frontend_ready_role_bigrams(key_evidence) & _frontend_ready_role_bigrams(
        action_evidence
    )
    shared_ratio = len(key_terms & action_terms) / max(len(key_terms | action_terms), 1)
    repeats_evidence = len(shared_bigrams) >= 2 or shared_ratio >= 0.55
    if repeats_evidence and not _has_action_evidence_internal_axis(action_evidence):
        return (
            "대응방향 근거/설명이 시사점 근거를 반복합니다. "
            "action 근거에는 SK AX가 볼 내부 판단 축을 별도로 설명해야 합니다."
        )
    return ""


def _frontend_ready_specific_anchor_violations(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> list[str]:
    anchors = _specific_event_anchors_for_frontend(integrated_issue)
    if len(anchors) < 2:
        return []
    implication = result.get("implication") or {}
    frontend_ready = implication.get("frontend_ready") or {}
    if not isinstance(frontend_ready, dict):
        return []
    violations: list[str] = []
    for section_key, label in (
        ("key_implication", "시사점"),
        ("suggested_action", "대응방향"),
    ):
        block = frontend_ready.get(section_key) or {}
        if not isinstance(block, dict):
            continue
        text = " ".join(
            [
                str(block.get("sentence") or ""),
                str(block.get("evidence_sentence") or ""),
            ]
        )
        matched = [anchor for anchor in anchors if _text_has_anchor_term(text, [anchor])]
        if len(set(matched)) < 2:
            violations.append(
                f"frontend_ready.{section_key}: {label} 문장에 현재 사건의 구체 anchor가 "
                f"부족합니다. 다음 중 2개 이상을 직접 사용하세요: {', '.join(anchors[:8])}"
            )
    return violations


def _frontend_ready_repair_already_attempted(result: dict[str, Any]) -> bool:
    implication = result.get("implication") or {}
    if not isinstance(implication, dict):
        return False
    diagnostics = implication.get("frontend_ready_diagnostics") or {}
    if not isinstance(diagnostics, dict):
        return False
    return bool(
        diagnostics.get("frontend_ready_after_repair")
        or diagnostics.get("removed_reason")
        or diagnostics.get("required_violations")
        or diagnostics.get("claim_violations")
    )


def _can_attempt_frontend_ready_repair_for_issue(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> bool:
    del result
    if not isinstance(integrated_issue, dict) or not integrated_issue:
        return False
    if integrated_issue.get(
        "is_valid_summary"
    ) is False and not _has_integrated_text_candidate_signal(integrated_issue):
        return False
    return bool(
        integrated_issue.get("integrated_text")
        or integrated_issue.get("fact_summary")
        or integrated_issue.get("consolidated_facts")
        or integrated_issue.get("one_line_summary")
        or integrated_issue.get("headline")
    )


def _frontend_ready_only_violations(violations: list[str]) -> bool:
    normalized = [str(violation or "").strip() for violation in violations if violation]
    return bool(normalized) and all(
        violation.startswith("frontend_ready") for violation in normalized
    )


def _requires_self_review_for_violations(
    violations: Sequence[str],
    *,
    result: dict[str, Any],
) -> bool:
    if _SELF_REVIEW_DISABLED:
        return False
    normalized = " ".join(str(violation or "") for violation in violations if violation)
    if not normalized.strip():
        return False
    if _frontend_ready_only_violations(list(violations)):
        return False
    hard_risk_pattern = (
        r"계약\s*상대방|고객\s*슬롯|공급자|수행사|운영\s*주체|도입\s*주체|"
        r"과대해석|과대|역할|신규\s*사업|사업\s*확장|입지|경쟁력|"
        r"시장\s*점유율|선도|주도|리더십|역량\s*강화|성과|수익성|"
        r"매출\s*(성장|확대)|영업이익|사업\s*기회|확장|넓히"
    )
    if re.search(hard_risk_pattern, normalized):
        return True
    result_text = _json_dumps(result)
    return bool(re.search(hard_risk_pattern, result_text))


def _business_novelty_overclaim_violation(
    text: str,
    *,
    label: str,
    profile_linkage_evaluation: dict[str, Any],
) -> str:
    if not text or _is_follow_up_or_watch_field(label):
        return ""
    if not label.startswith(("analysis.", "peer_implication.")):
        return ""
    peer_linkages = [
        linkage
        for linkage in _jsonish_list(profile_linkage_evaluation.get("peer_linkages"))
        if isinstance(linkage, dict)
    ]
    if not peer_linkages:
        return ""
    cautious_terms = re.compile(r"관찰|신호|가능성|후속\s*확인|단정하기\s*어렵|미포착")
    for linkage in peer_linkages:
        novelty = str(linkage.get("business_novelty_status") or "")
        if novelty == "not_new_business_counterparty_role" and _matches_any_pattern(
            text,
            OVERCLAIM_PATTERNS["counterparty"],
        ):
            return (
                "계약 상대방/고객 슬롯인 피어를 신규 사업·입지 강화·역량 강화처럼 과대해석했습니다."
            )
        if novelty in {
            "new_or_untracked_business_signal",
            "profile_insufficient_cannot_judge_novelty",
        } and _matches_any_pattern(text, OVERCLAIM_PATTERNS["new_signal"]):
            if not cautious_terms.search(text):
                return (
                    "프로필에 강하게 포착되지 않은 사업 신호를 확정 성과나 역량 강화처럼 "
                    "단정했습니다. 관찰 신호/후속 확인 수준으로 낮춰야 합니다."
                )
    return ""


def _action_artifact_plan_violation(
    text: str,
    *,
    label: str,
    action_artifact_plan: dict[str, Any],
) -> str:
    if not text or not label.startswith("skax_implication.recommended_actions"):
        return ""
    external_phrase = _skax_external_customer_facing_violation(text)
    if external_phrase:
        return external_phrase
    issue_terms = _action_plan_issue_terms(action_artifact_plan)
    if issue_terms and not _action_text_has_issue_signal(text, issue_terms):
        return "대응방향에 현재 사건의 대상 사업/시스템/서비스/고객군 신호가 연결되지 않았습니다."
    if not _action_text_has_internal_strategy_checkpoint(text):
        return "대응방향에 SK AX가 내부적으로 점검할 기준이 부족합니다."
    if not _action_text_has_skax_change(text):
        return "대응방향에 SK AX가 보완하거나 점검할 구체 방식이 부족합니다."
    return ""


def _skax_external_customer_facing_violation(text: str) -> str:
    value = str(text or "")
    if re.search(
        r"고객(이|은|에게|한테).{0,24}(확인|비교|평가|판단|볼 수|보여|제시|설명)",
        value,
    ) or re.search(r"고객\s*제안|고객\s*확인\s*기준|고객이\s*확인", value):
        return (
            "대응방향이 외부 고객 제안/확인 문장처럼 작성되었습니다. "
            "유사 고객군/유사 사업 관점은 유지하되, 외부 고객 제안 문장이 아니라 "
            "SK AX 내부에서 경쟁사 사업군과 자사 사업군의 겹침/차이, 대응 가능 범위, "
            "역량 공백, 운영·영업 전략, 후속 모니터링 항목을 점검하는 문장으로 "
            "바꿔야 합니다."
        )
    return ""


def _unsupported_skax_profile_term_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> str:
    if not label.startswith("skax_implication."):
        return ""
    if _is_follow_up_or_watch_field(label):
        return ""
    issue_tokens = _issue_relevance_tokens(integrated_issue)
    output_tokens = _content_tokens(text)
    skax_terms = _concrete_profile_terms(
        profile_context,
        integrated_issue=integrated_issue,
        scope="skax",
    )
    used_skax_terms = sorted(output_tokens & skax_terms)
    if not used_skax_terms:
        return ""
    linkage_level = _relevant_profile_linkage_level_from_evaluation(
        profile_linkage_evaluation,
        scope="skax",
    ) or _relevant_profile_linkage_level(
        profile_context,
        integrated_issue=integrated_issue,
        scope="skax",
    )
    if linkage_level in {"high", "medium"}:
        return ""
    unsupported = sorted(set(used_skax_terms) - issue_tokens)
    if not unsupported:
        return ""
    return (
        "SK AX 프로필 연결이 약한 상태에서 구체 SK AX 사업/역량 용어를 사용했습니다: "
        f"{', '.join(unsupported[:3])}. 대응 방향은 현재 사건의 적용 범위/대상 업무/"
        "검증 기준으로 낮춰야 합니다."
    )


def _unsupported_domain_term_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    del text, label, integrated_issue
    return ""


def _two_section_repetition_violation(result: dict[str, Any]) -> str:
    analysis = result.get("analysis") or {}
    implication = result.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    insight_texts = [
        str(analysis.get("analysis_summary") or ""),
        *[str(item or "") for item in _string_list(analysis.get("strategic_meaning"), max_items=3)],
        str(analysis.get("market_signal") or ""),
        str(peer.get("peer_meaning") or ""),
        str(peer.get("capability_change") or ""),
    ]
    normalized: list[set[str]] = []
    labels: list[str] = []
    for index, insight_text in enumerate(insight_texts):
        tokens = _high_signal_tokens_for_repetition(insight_text)
        if len(tokens) < 4:
            continue
        normalized.append(tokens)
        labels.append(f"시사점 필드 {index + 1}")
    repeated_pairs: list[tuple[str, str]] = []
    for left_index, left_tokens in enumerate(normalized):
        for right_index in range(left_index + 1, len(normalized)):
            right_tokens = normalized[right_index]
            overlap = len(left_tokens & right_tokens)
            smaller = max(1, min(len(left_tokens), len(right_tokens)))
            if overlap / smaller >= 0.75:
                repeated_pairs.append((labels[left_index], labels[right_index]))
    if len(repeated_pairs) >= 2:
        first, second = repeated_pairs[0]
        return (
            f"{first}와 {second} 등 여러 시사점 필드가 같은 의미를 반복합니다. "
            "analysis 와 peer_implication 은 별도 노출 섹션이 아니라 하나의 "
            "시사점 묶음으로 압축해야 합니다."
        )
    direction_terms = ("가속화", "확장", "확대")
    repeated_direction_count = sum(
        1 for text in insight_texts if any(term in text for term in direction_terms)
    )
    if repeated_direction_count >= 3:
        return (
            "시사점 필드 여러 곳에서 가속화/확장/확대 같은 방향성 표현을 반복합니다. "
            "카드 요약을 반복하지 말고 적용 범위, 대상 업무, 검증 기준으로 나눠 써야 합니다."
        )
    return ""


def _restore_valid_flags_if_structurally_safe(result: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    has_clean_frontend_ready = _has_clean_displayable_frontend_ready_diagnostics(out)
    if "quality_gate_failed" in _json_dumps(out):
        if not has_clean_frontend_ready:
            return out
        out = _clear_quality_gate_failed_marker(out)
        has_clean_frontend_ready = _has_clean_displayable_frontend_ready_diagnostics(out)
    analysis = out.get("analysis") or {}
    implication = out.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}

    if (
        analysis.get("analysis_summary")
        and analysis.get("market_signal")
        and _string_list(analysis.get("strategic_meaning"), max_items=3)
    ):
        analysis["is_valid_analysis"] = True
        analysis["confidence"] = max(_clamp_float(analysis.get("confidence"), 0.0), 0.62)

    if (peer.get("peer_meaning") or skax.get("why_important")) and (
        _string_list(skax.get("recommended_actions"), max_items=3)
        or _string_list(skax.get("opportunities"), max_items=3)
        or skax.get("potential_impact")
    ):
        implication["is_valid_implication"] = True
        implication["confidence"] = max(_clamp_float(implication.get("confidence"), 0.0), 0.62)
        if implication.get("evidence_label") == "insufficient":
            implication["evidence_label"] = "moderate"
    elif has_clean_frontend_ready:
        implication["is_valid_implication"] = True
        implication["confidence"] = max(_clamp_float(implication.get("confidence"), 0.0), 0.62)
        if implication.get("evidence_label") == "insufficient":
            implication["evidence_label"] = "moderate"

    out["analysis"] = analysis
    implication["peer_implication"] = peer
    implication["skax_implication"] = skax
    out["implication"] = implication
    return out


def _clear_quality_gate_failed_marker(result: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    analysis = out.get("analysis") or {}
    reason = str(analysis.get("reason") or "").strip()
    if "quality_gate_failed" in reason:
        reason = re.sub(r"\s*\|\s*quality_gate_failed:.*$", "", reason).strip()
        reason = "" if reason.startswith("quality_gate_failed:") else reason
        analysis["reason"] = reason
    out["analysis"] = analysis
    return out


def _attach_sentence_grounding(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any] | None,
) -> dict[str, Any]:
    out = _ensure_reasoning_debug_fields(
        json.loads(json.dumps(result, ensure_ascii=False, default=str)),
        integrated_issue=integrated_issue,
    )
    grounding = _build_sentence_grounding(
        out,
        integrated_issue=integrated_issue,
        profile_context=profile_context or {},
    )
    out["sentence_grounding"] = grounding
    return out


def _critical_ungrounded_paths(grounding: dict[str, Any]) -> list[str]:
    paths = []
    for entry in grounding.get("entries") or []:
        if not isinstance(entry, dict) or not entry.get("needs_review"):
            continue
        path = str(entry.get("path") or "")
        if path.startswith(
            (
                "analysis.",
                "peer_implication.",
                "skax_implication.why_important",
                "skax_implication.potential_impact",
            )
        ):
            paths.append(path)
    return list(dict.fromkeys(paths))


def _build_sentence_grounding(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> dict[str, Any]:
    fact_entries = _fact_texts(integrated_issue)
    profile_entries = _profile_grounding_entries(
        profile_context,
        integrated_issue=integrated_issue,
    )
    entries: list[dict[str, Any]] = []
    for path, target_text, scope in _grounding_target_texts(result):
        entries.extend(
            _grounding_entries_for_text(
                path=path,
                text=target_text,
                scope=scope,
                fact_entries=fact_entries,
                profile_entries=profile_entries,
            )
        )
    ungrounded_paths = [
        item["path"]
        for item in entries
        if item.get("needs_review") and item.get("grounding_type") == "ungrounded"
    ]
    return {
        "schema_version": "sentence-grounding-v1",
        "generator": "StrategicInsightAgent",
        "entries": entries,
        "summary": {
            "entry_count": len(entries),
            "fact_grounded_count": sum(1 for item in entries if item.get("used_fact_ids")),
            "profile_grounded_count": sum(1 for item in entries if item.get("used_profile_fields")),
            "ungrounded_paths": ungrounded_paths,
        },
    }


def _grounding_target_texts(result: dict[str, Any]) -> list[tuple[str, str, str]]:
    analysis = result.get("analysis") or {}
    implication = result.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    targets: list[tuple[str, str, str]] = [
        ("analysis.analysis_summary", str(analysis.get("analysis_summary") or ""), "peer"),
        ("analysis.market_signal", str(analysis.get("market_signal") or ""), "peer"),
        ("analysis.impact_reason", str(analysis.get("impact_reason") or ""), "peer"),
        ("analysis.reason", str(analysis.get("reason") or ""), "peer"),
        ("peer_implication.peer_meaning", str(peer.get("peer_meaning") or ""), "peer"),
        (
            "peer_implication.capability_change",
            str(peer.get("capability_change") or ""),
            "peer",
        ),
        (
            "skax_implication.why_important",
            str(skax.get("why_important") or ""),
            "skax",
        ),
        (
            "skax_implication.potential_impact",
            str(skax.get("potential_impact") or ""),
            "skax",
        ),
    ]
    for index, item in enumerate(_string_list(analysis.get("strategic_meaning"), max_items=3)):
        targets.append((f"analysis.strategic_meaning[{index}]", item, "peer"))
    for field in ("opportunities", "threats", "recommended_actions"):
        for index, item in enumerate(_string_list(skax.get(field), max_items=3)):
            targets.append((f"skax_implication.{field}[{index}]", item, "skax"))
    return [(path, text.strip(), scope) for path, text, scope in targets if text.strip()]


def _grounding_entries_for_text(
    *,
    path: str,
    text: str,
    scope: str,
    fact_entries: list[tuple[str, str]],
    profile_entries: list[dict[str, str]],
) -> list[dict[str, Any]]:
    sentences = _split_sentences(text) or [text]
    output: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        sentence_text = sentence.strip()
        if not sentence_text:
            continue
        used_fact_ids = _matching_fact_ids(sentence_text, fact_entries)
        used_profile_fields = _matching_profile_fields(
            sentence_text,
            profile_entries=profile_entries,
            scope=scope,
        )
        grounding_type = _grounding_type(used_fact_ids, used_profile_fields)
        output.append(
            {
                "path": f"{path}.sentence[{index}]" if len(sentences) > 1 else path,
                "text": sentence_text,
                "used_fact_ids": used_fact_ids,
                "used_profile_fields": used_profile_fields,
                "grounding_type": grounding_type,
                "needs_review": grounding_type == "ungrounded",
            }
        )
    return output


def _grounding_type(fact_ids: list[str], profile_fields: list[str]) -> str:
    if fact_ids and profile_fields:
        return "fact+profile"
    if fact_ids:
        return "fact"
    if profile_fields:
        return "profile"
    return "ungrounded"


def _profile_grounding_entries(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> list[dict[str, str]]:
    prompt_profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue)
    entries: list[dict[str, str]] = []
    skax = prompt_profile.get("skax_profile") or {}
    entries.extend(_flatten_profile_grounding_entries(skax, path="skax_profile", scope="skax"))
    peers = prompt_profile.get("peer_profiles") or {}
    if isinstance(peers, dict):
        for peer_id, payload in peers.items():
            entries.extend(
                _flatten_profile_grounding_entries(
                    payload,
                    path=f"peer_profiles.{peer_id}",
                    scope="peer",
                )
            )
    return entries


def _flatten_profile_grounding_entries(
    value: Any,
    *,
    path: str,
    scope: str,
) -> list[dict[str, str]]:
    if value in ({}, [], "", None):
        return []
    if isinstance(value, dict):
        entries: list[dict[str, str]] = []
        combined = _profile_entry_text(value)
        if combined:
            entries.append({"path": path, "scope": scope, "text": combined})
        for key, child in value.items():
            if key in {"company_id", "peer_id", "company_name", "company_name_ko"}:
                continue
            entries.extend(
                _flatten_profile_grounding_entries(child, path=f"{path}.{key}", scope=scope)
            )
        return entries
    if isinstance(value, list):
        entries = []
        for index, child in enumerate(value[:8]):
            entries.extend(
                _flatten_profile_grounding_entries(child, path=f"{path}[{index}]", scope=scope)
            )
        return entries
    text = str(value or "").strip()
    return [{"path": path, "scope": scope, "text": text}] if text else []


def _ensure_safe_recommended_actions(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any] | None = None,
    action_artifact_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    implication = out.get("implication") or {}
    skax = implication.get("skax_implication") or {}
    safe_actions: list[str] = []
    for index, action in enumerate(
        _string_list(skax.get("recommended_actions"), max_items=3),
        start=1,
    ):
        label = f"skax_implication.recommended_actions[{index}]"
        if _repair_action_violation(
            action,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context or {},
            action_artifact_plan=action_artifact_plan or {},
        ):
            continue
        safe_actions.append(action)

    skax["recommended_actions"] = safe_actions[:3]
    implication["skax_implication"] = skax
    out["implication"] = implication
    return out


def _repair_action_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    action_artifact_plan: dict[str, Any] | None = None,
) -> str:
    violation = _recommended_action_quality_violation(
        text,
        label=label,
        integrated_issue=integrated_issue,
        profile_context=profile_context,
    )
    if violation:
        return violation
    if _hard_quality_violation_for_text(
        text,
        label=label,
        integrated_issue=integrated_issue,
    ):
        return "대응방향이 현재 사건의 근거 범위를 벗어난 표현을 포함했습니다."
    return _counterparty_role_action_violation(
        text,
        label=label,
        integrated_issue=integrated_issue,
    ) or _action_artifact_plan_violation(
        text,
        label=label,
        action_artifact_plan=action_artifact_plan or {},
    )


def _unsupported_peer_profile_claim_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> str | None:
    if not label.startswith(
        ("peer_implication.peer_meaning", "peer_implication.capability_change")
    ):
        return None
    if isinstance(profile_linkage_evaluation, dict):
        if _has_relevant_peer_profile_linkage(
            profile_linkage_evaluation,
            integrated_issue=integrated_issue,
        ):
            return None
    if _has_relevant_peer_profile_context(profile_context, integrated_issue=integrated_issue):
        return None
    if not _mentions_profile_based_peer_claim(text):
        return None
    if re.search(r"부족|확인되지|단정하기\s*어렵|사건\s*기반|낮춰", text or ""):
        return None
    return (
        "현재 사건과 직접 맞는 피어 프로필 접점이 없는데 사업영역/역량 기반 "
        "시사점처럼 썼습니다. 사건 기반 해석으로 낮춰야 합니다."
    )
