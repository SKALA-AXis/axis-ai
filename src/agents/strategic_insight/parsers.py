"""strategic_insight parsers — extracted from facade (move-only)."""

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


def _parse_and_normalize(
    raw: str,
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    cluster_metadata: dict[str, Any],
    profile_context: dict[str, Any],
    analysis_context: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    data = _parse_json_loose(raw)
    if not isinstance(data, dict) or not (
        isinstance(data.get("analysis"), dict) and isinstance(data.get("implication"), dict)
    ):
        raise ValueError("StrategicInsightAgent response missing analysis/implication JSON blocks")

    analysis = _normalize_analysis_block(data.get("analysis") or {})
    implication = _normalize_implication_block(
        data.get("implication") or {},
        integrated_issue=integrated_issue,
        profile_context=profile_context,
        analysis_context=analysis_context,
        model=model,
    )
    is_valid = bool(
        data.get("is_valid_strategic_insight", True)
        and analysis.get("is_valid_analysis")
        and implication.get("is_valid_implication")
    )
    return {
        "is_valid_strategic_insight": is_valid,
        "issue_understanding": _normalize_issue_understanding(
            data.get("issue_understanding"),
            integrated_issue=integrated_issue,
        ),
        "profile_linkage": _normalize_llm_profile_linkage(data.get("profile_linkage")),
        "skax_response_linkage": _normalize_skax_response_linkage(
            data.get("skax_response_linkage")
        ),
        "claim_strength": _choice(
            data.get("claim_strength"),
            {"strong", "moderate", "cautious"},
            "cautious",
        ),
        "grounding_summary": _normalize_grounding_summary(
            data.get("grounding_summary"),
            integrated_issue=integrated_issue,
        ),
        "analysis": analysis,
        "implication": implication,
    }


def _normalize_issue_understanding(
    value: Any,
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    data = _json_dict(value)
    known_fact_ids = _known_fact_ids(integrated_issue)
    return {
        "confirmed_facts": _string_list(data.get("confirmed_facts"), max_items=8),
        "main_actor": str(data.get("main_actor") or integrated_issue.get("main_company") or ""),
        "peer_role_in_issue": _choice(
            data.get("peer_role_in_issue"),
            {
                "provider",
                "builder",
                "operator",
                "selected_party",
                "contract_counterparty",
                "customer_or_buyer",
                "partner",
                "investor",
                "unclear",
            },
            "unclear",
        ),
        "role_confidence": _clamp_float(data.get("role_confidence"), 0.0),
        "activity_nature": _choice(
            data.get("activity_nature"),
            {
                "selection",
                "contract",
                "launch",
                "investment",
                "partnership",
                "operation",
                "system_transition",
                "infrastructure_build",
                "business_update",
                "unclear",
            },
            "unclear",
        ),
        "target_business_or_system": _string_list(
            data.get("target_business_or_system"),
            max_items=8,
        ),
        "customer_or_market_scope": _string_list(
            data.get("customer_or_market_scope"),
            max_items=8,
        ),
        "confirmed_numbers_or_dates": _string_list(
            data.get("confirmed_numbers_or_dates"),
            max_items=8,
        ),
        "uncertain_points": _string_list(data.get("uncertain_points"), max_items=8),
        "evidence_ids": [
            fact_id
            for fact_id in _string_list(data.get("evidence_ids"), max_items=12)
            if fact_id in known_fact_ids
        ],
    }


def _normalize_llm_profile_linkage(value: Any) -> dict[str, Any]:
    data = _json_dict(value)
    return {
        "peer_company": str(data.get("peer_company") or ""),
        "profile_evidence_available": bool(data.get("profile_evidence_available")),
        "matched_profile_areas": [
            {
                "business_line": str(item.get("business_line") or "").strip(),
                "business_area": str(item.get("business_area") or "").strip(),
                "profile_area_name": str(item.get("profile_area_name") or "").strip(),
                "profile_capability": str(item.get("profile_capability") or "").strip(),
                "matched_capabilities": _string_list(
                    item.get("matched_capabilities")
                    or item.get("core_capabilities")
                    or item.get("capabilities"),
                    max_items=8,
                ),
                "matched_products_or_services": _string_list(
                    item.get("matched_products_or_services")
                    or item.get("products_or_services")
                    or item.get("key_products_services"),
                    max_items=8,
                ),
                "matched_issue_terms": _string_list(
                    item.get("matched_issue_terms") or item.get("matched_terms"),
                    max_items=12,
                ),
                "evidence_text": str(item.get("evidence_text") or "").strip(),
                "why_relevant_to_issue": str(item.get("why_relevant_to_issue") or "").strip(),
                "profile_source_ref": str(
                    item.get("profile_source_ref") or item.get("source_ref") or ""
                ).strip(),
                "specificity_level": _choice(
                    item.get("specificity_level"),
                    {
                        "product_or_service",
                        "core_capability",
                        "business_area",
                        "business_line",
                        "profile_context",
                    },
                    "profile_context",
                ),
            }
            for item in _jsonish_list(data.get("matched_profile_areas"))[:5]
            if isinstance(item, dict)
        ],
        "linkage_level": _choice(
            data.get("linkage_level"),
            {"high", "medium", "low", "none"},
            "none",
        ),
        "business_novelty_status": _choice(
            data.get("business_novelty_status"),
            {
                "existing_profile_business_linked",
                "weak_profile_linkage",
                "new_or_untracked_business_signal",
                "role_sensitive_untracked_signal",
                "profile_insufficient_cannot_judge_novelty",
                "uncertain_not_enough_to_call_new_business",
                "not_new_business_counterparty_role",
            },
            "profile_insufficient_cannot_judge_novelty",
        ),
        "allowed_interpretation_strength": _choice(
            data.get("allowed_interpretation_strength"),
            {
                "profile_based",
                "cautious_profile_based",
                "event_based",
                "observation_only",
            },
            "observation_only",
        ),
        "reason": str(data.get("reason") or "").strip(),
    }


def _normalize_skax_response_linkage(value: Any) -> dict[str, Any]:
    data = _json_dict(value)
    return {
        "skax_profile_evidence_available": bool(data.get("skax_profile_evidence_available")),
        "matched_skax_areas": [
            {
                "business_line": str(item.get("business_line") or "").strip(),
                "business_area": str(item.get("business_area") or "").strip(),
                "profile_area_name": str(item.get("profile_area_name") or "").strip(),
                "matched_capabilities": _string_list(
                    item.get("matched_capabilities")
                    or item.get("core_capabilities")
                    or item.get("capabilities"),
                    max_items=8,
                ),
                "matched_products_or_services": _string_list(
                    item.get("matched_products_or_services")
                    or item.get("products_or_services")
                    or item.get("key_products_services"),
                    max_items=8,
                ),
                "matched_issue_terms": _string_list(
                    item.get("matched_issue_terms") or item.get("matched_terms"),
                    max_items=12,
                ),
                "evidence_text": str(item.get("evidence_text") or "").strip(),
                "why_relevant_to_issue": str(item.get("why_relevant_to_issue") or "").strip(),
                "profile_source_ref": str(
                    item.get("profile_source_ref") or item.get("source_ref") or ""
                ).strip(),
                "specificity_level": _choice(
                    item.get("specificity_level"),
                    {
                        "product_or_service",
                        "core_capability",
                        "business_area",
                        "business_line",
                        "profile_context",
                    },
                    "profile_context",
                ),
            }
            for item in _jsonish_list(data.get("matched_skax_areas"))[:5]
            if isinstance(item, dict)
        ],
        "response_mode": _choice(
            data.get("response_mode"),
            {"profile_based_action", "cautious_action", "generic_monitoring_action"},
            "generic_monitoring_action",
        ),
        "response_focus": _string_list(data.get("response_focus"), max_items=8),
        "internal_checkpoints": _string_list(data.get("internal_checkpoints"), max_items=8),
        "recommended_focus": _string_list(data.get("recommended_focus"), max_items=8),
        "monitoring_points": _string_list(data.get("monitoring_points"), max_items=8),
        "reason": str(data.get("reason") or "").strip(),
    }


def _normalize_grounding_summary(
    value: Any,
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    data = _json_dict(value)
    known_fact_ids = _known_fact_ids(integrated_issue)
    return {
        "used_fact_ids": [
            fact_id
            for fact_id in _string_list(data.get("used_fact_ids"), max_items=12)
            if fact_id in known_fact_ids
        ],
        "used_profile_refs": _string_list(data.get("used_profile_refs"), max_items=12),
        "ungrounded_claims_removed": _string_list(
            data.get("ungrounded_claims_removed"),
            max_items=12,
        ),
    }


def _parse_review_and_normalize(
    raw: str,
    *,
    original: dict[str, Any],
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    profile_context: dict[str, Any],
    analysis_context: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    data = _parse_json_loose(raw)
    if not isinstance(data, dict):
        return original

    revised = data.get("revised_result") or data.get("revised")
    if not isinstance(revised, dict):
        if isinstance(data.get("analysis"), dict) and isinstance(data.get("implication"), dict):
            revised = data
        else:
            return original

    normalized = _parse_and_normalize(
        _json_dumps(revised),
        integrated_issue=integrated_issue,
        classification=classification,
        cluster_metadata={},
        profile_context=profile_context,
        analysis_context=analysis_context,
        model=model,
    )
    return normalized


def _normalize_analysis_block(
    data: dict[str, Any],
) -> dict[str, Any]:
    strategic_meaning = _string_list(data.get("strategic_meaning"), max_items=3)
    return {
        "is_valid_analysis": bool(data.get("is_valid_analysis", True))
        and bool(data.get("analysis_summary") or strategic_meaning),
        "analysis_scope": "peer_and_industry",
        "analysis_summary": str(data.get("analysis_summary") or "").strip(),
        "strategic_meaning": strategic_meaning,
        "market_signal": str(data.get("market_signal") or "").strip(),
        "impact_level": _choice(data.get("impact_level"), _IMPACT_LEVELS, "medium"),
        "impact_reason": str(data.get("impact_reason") or "").strip(),
        "risk_or_opportunity": _choice(
            data.get("risk_or_opportunity"), _RISK_OR_OPPORTUNITY, "neutral"
        ),
        "confidence": _clamp_float(data.get("confidence"), 0.0),
        "reason": str(data.get("reason") or "").strip(),
    }


def _normalize_implication_block(
    data: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    analysis_context: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    peer_input = data.get("peer_implication") or {}
    skax_input = data.get("skax_implication") or {}
    known_fact_ids = _known_fact_ids(integrated_issue)
    sourced_evidence_ids = [
        fact_id
        for fact_id in _string_list(peer_input.get("sourced_evidence_ids"), max_items=10)
        if fact_id in known_fact_ids
    ]
    confidence = _clamp_float(data.get("confidence"), 0.0)
    recommended_actions = _normalize_recommended_actions(
        _string_list(skax_input.get("recommended_actions"), max_items=3)
    )
    skax = {
        "why_important": str(skax_input.get("why_important") or "").strip(),
        "potential_impact": str(skax_input.get("potential_impact") or "").strip(),
        "opportunities": _string_list(skax_input.get("opportunities"), max_items=3),
        "threats": _string_list(skax_input.get("threats"), max_items=3),
        "recommended_actions": recommended_actions,
        "business_line_mapping": [
            item
            for item in _string_list(skax_input.get("business_line_mapping"), max_items=3)
            if item in _business_line_candidates(profile_context)
        ],
    }
    peer = {
        "company_id": str(
            peer_input.get("company_id")
            or integrated_issue.get("main_company")
            or _first_peer_id(profile_context)
            or ""
        ),
        "company_name_ko": str(
            peer_input.get("company_name_ko") or _first_peer_name(profile_context) or ""
        ),
        "peer_meaning": str(peer_input.get("peer_meaning") or "").strip(),
        "capability_change": _optional_str(peer_input.get("capability_change")),
        "sourced_evidence_ids": sourced_evidence_ids,
    }
    sourced_evidence_ids = _augment_sourced_evidence_ids(
        sourced_evidence_ids,
        integrated_issue=integrated_issue,
        output_texts=[
            peer["peer_meaning"],
            peer["capability_change"],
            skax["why_important"],
            skax["potential_impact"],
            *skax["opportunities"],
            *skax["threats"],
            *skax["recommended_actions"],
        ],
    )
    peer["sourced_evidence_ids"] = sourced_evidence_ids
    used_layers = _normalize_used_context_layers(
        ((data.get("provenance") or {}) if isinstance(data.get("provenance"), dict) else {}).get(
            "used_context_layers"
        ),
        analysis_context=analysis_context,
    )
    is_valid = bool(
        data.get("is_valid_implication", True)
        and (peer["peer_meaning"] or skax["why_important"])
        and (skax["recommended_actions"] or skax["opportunities"] or skax["potential_impact"])
    )
    implication_out: dict[str, Any] = {
        "is_valid_implication": is_valid,
        "implication_scope": "peer_and_skax",
        "peer_implication": peer,
        "skax_implication": skax,
        "follow_up_questions": _string_list(data.get("follow_up_questions"), max_items=3),
        "watch_points": _string_list(data.get("watch_points"), max_items=3),
        "confidence": confidence,
        "evidence_label": _evidence_label(data.get("evidence_label"), confidence),
        "provenance": {
            "generator": "StrategicInsightAgent",
            "prompt_version": _PROMPT_VERSION,
            "model": model,
            "model_config": _llm_model_config_diagnostics(),
            "used_fact_ids": sourced_evidence_ids,
            "used_context_layers": used_layers,
            "run_at": datetime.now(UTC).isoformat(),
        },
    }
    frontend_ready = _normalize_frontend_ready(data.get("frontend_ready"))
    frontend_ready = _frontend_ready_with_issue_evidence_anchors(
        frontend_ready,
        integrated_issue=integrated_issue,
    )
    if frontend_ready:
        implication_out["frontend_ready"] = frontend_ready
    return implication_out


def _normalize_frontend_ready(value: Any, *, default_source: str = "llm_direct") -> dict[str, Any]:
    data = _json_dict(value)
    source = _normalize_frontend_ready_source(data.get("source"), default_source)
    key_implication = _normalize_frontend_ready_block(
        data.get("key_implication"),
        default_source=source,
        anchor_key="profile_anchor_terms",
    )
    suggested_action = _normalize_frontend_ready_block(
        data.get("suggested_action"),
        default_source=source,
        anchor_key="skax_anchor_terms",
    )
    out: dict[str, Any] = {}
    if source:
        out["source"] = source
    insight_basis = _normalize_frontend_basis(data.get("insight_basis"))
    action_basis = _normalize_frontend_basis(data.get("action_basis"))
    if insight_basis:
        out["insight_basis"] = insight_basis
    if action_basis:
        out["action_basis"] = action_basis
    if key_implication:
        out["key_implication"] = key_implication
    if suggested_action:
        out["suggested_action"] = suggested_action
    if "key_implication" not in out and "suggested_action" not in out:
        return {}
    return out


def _normalize_frontend_basis(value: Any) -> dict[str, Any]:
    data = _json_dict(value)
    if not data:
        return {}
    out: dict[str, Any] = {}
    for key in (
        "event_anchor",
        "check_target",
        "used_fact_ids",
        "used_profile_refs",
    ):
        values = _string_list(data.get(key), max_items=8)
        if values:
            out[key] = values
    for key in (
        "observed_change",
        "comparison_context",
        "strategic_reading",
        "skax_question",
        "response_angle",
        "required_condition",
        "confidence",
    ):
        text = str(data.get(key) or "").strip()
        if text:
            out[key] = text
    return out


def _normalize_frontend_ready_block(
    value: Any,
    *,
    default_source: str,
    anchor_key: str,
) -> dict[str, Any]:
    data = _json_dict(value)
    sentence = _strip_frontend_ready_label(data.get("sentence"))
    evidence_sentence = _strip_frontend_ready_label(data.get("evidence_sentence"))
    if not sentence and not evidence_sentence:
        return {}
    source = _normalize_frontend_ready_source(data.get("source"), default_source)
    return {
        "source": source,
        "frame": str(data.get("frame") or "").strip(),
        "claim_type": _choice(
            data.get("claim_type"),
            _FRONTEND_READY_CLAIM_TYPES,
            "internal_strategy_check"
            if anchor_key == "skax_anchor_terms"
            else "event_based_signal",
        ),
        "claim_strength": _choice(
            data.get("claim_strength"),
            _FRONTEND_READY_CLAIM_STRENGTHS,
            "cautious",
        ),
        "evidence_mode": _choice(
            data.get("evidence_mode"),
            _FRONTEND_READY_EVIDENCE_MODES,
            "generic_monitoring" if anchor_key == "skax_anchor_terms" else "event_based",
        ),
        "event_anchor_terms": _string_list(data.get("event_anchor_terms"), max_items=8),
        anchor_key: _string_list(data.get(anchor_key), max_items=8),
        "unsupported_claims_removed": _string_list(
            data.get("unsupported_claims_removed"),
            max_items=8,
        ),
        "sentence": sentence,
        "evidence_sentence": evidence_sentence,
    }


def _normalize_frontend_ready_source(value: Any, default: str) -> str:
    source = str(value or default or "").strip()
    return source if source in _FRONTEND_READY_SOURCES else default


def _normalize_recommended_actions(actions: list[str]) -> list[str]:
    """Keep LLM-written action meaning; only trim whitespace and exact duplicates."""
    normalized: list[str] = []
    seen: set[str] = set()
    for action in actions:
        text = re.sub(r"\s+", " ", str(action or "")).strip()
        if not text:
            continue
        key = text.rstrip(".。").casefold()
        if key in seen:
            continue
        normalized.append(text)
        seen.add(key)
    return normalized


def _augment_sourced_evidence_ids(
    existing: list[str],
    *,
    integrated_issue: dict[str, Any],
    output_texts: list[Any],
) -> list[str]:
    out = list(dict.fromkeys(existing))
    known = _fact_texts(integrated_issue)
    combined_output = " ".join(str(item or "") for item in output_texts)
    output_tokens = _content_tokens(combined_output)
    for fact_id, fact_text in known:
        if fact_id in out:
            continue
        if _fact_is_referenced(fact_text, combined_output, output_tokens):
            out.append(fact_id)
        if len(out) >= 10:
            break
    return out


def _normalize_used_context_layers(value: Any, *, analysis_context: dict[str, Any]) -> list[str]:
    available = _used_context_layers(analysis_context)
    if not available:
        return []
    requested = _string_list(value, max_items=10)
    if not requested:
        return available
    return [layer for layer in requested if layer in available]
