"""strategic_insight fallback1 — extracted from facade (move-only)."""

from __future__ import annotations

import json
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


def _recommended_action_quality_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any] | None = None,
    profile_context: dict[str, Any] | None = None,
) -> str:
    if not text or not label.startswith("skax_implication.recommended_actions"):
        return ""
    evidence_text = _integrated_grounding_text(integrated_issue or {})
    specific_anchors = set(_specific_event_anchors_for_frontend(integrated_issue or {}))
    if (
        specific_anchors
        and not _action_text_has_issue_signal(text, specific_anchors)
        and not _action_text_has_grounding_axis_signal(text, evidence_text)
    ):
        return (
            "대응방향에 현재 사건의 구체 anchor가 연결되지 않았습니다. "
            "현재 사건의 대상 사업·시스템·서비스·수치·고객군 중 최소 하나를 "
            "직접 기준으로 삼아야 합니다."
        )
    if re.search(r"주가|거래를\s*마쳤|시장\s*반응|투자자\s*반응", text):
        return (
            "대응방향이 주가/시장 반응을 실행 근거로 사용했습니다. 전략 대응은 현재 사건의 "
            "사업 범위, 운영 조건, 검증 기준, 프로필 접점 중심으로 작성해야 합니다."
        )
    if re.search(r"클라우드|AI|인공지능|에이아이", text, flags=re.IGNORECASE):
        evidence_has_tech = re.search(
            r"클라우드|AI|인공지능|에이아이",
            evidence_text,
            flags=re.IGNORECASE,
        )
        profile_support = _has_concrete_profile_term(
            text,
            profile_context=profile_context or {},
            integrated_issue=integrated_issue or {},
            scope="skax",
        )
        if not evidence_has_tech and not profile_support:
            return (
                "현재 사건 근거 또는 SK AX 프로필 접점 없이 기술명을 대응방향에 사용했습니다. "
                "입력 사건에서 확인된 대상과 검증 기준 중심으로 낮춰야 합니다."
            )
    if re.search(r"성공|수주에\s*영향|신뢰성", text) and not _profile_has_execution_case(
        profile_context,
        integrated_issue=integrated_issue,
    ):
        return (
            "ProfileContext에 실행 사례 근거가 없는데 성공/수주 영향/신뢰성을 사용했습니다. "
            "SK AX가 내부적으로 점검할 검증 기준과 운영 조건 중심으로 낮춰야 합니다."
        )
    if "솔루션" in text and not _solution_term_supported_by_evidence(
        text,
        integrated_issue=integrated_issue or {},
        profile_context=profile_context or {},
    ):
        return (
            "대응방향이 근거 없는 솔루션 표현에 머물렀습니다. "
            "현재 사건에서 확인된 기준으로 낮춰야 합니다."
        )
    if re.search(r"성능.{0,12}(강조|입증)|검증된\s*성능", text):
        return (
            "대응방향이 성능 강조 같은 일반 표현에 머물렀습니다. "
            "현재 사건에서 확인된 기준으로 낮춰야 합니다."
        )
    off_topic_product_violation = _off_topic_application_product_violation(
        text,
        integrated_issue=integrated_issue or {},
    )
    if off_topic_product_violation:
        return off_topic_product_violation
    evidence_scoped_violation = _evidence_scoped_business_claim_violation(
        text,
        label=label,
        integrated_issue=integrated_issue,
        profile_context=profile_context,
    )
    if evidence_scoped_violation:
        return evidence_scoped_violation
    return ""


def _mark_quality_gate_failed(
    result: dict[str, Any],
    violations: list[str],
    *,
    preserve_frontend_ready: bool = False,
) -> dict[str, Any]:
    raw_out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    preserved_frontend_ready = None
    if preserve_frontend_ready:
        raw_implication = raw_out.get("implication") or {}
        if isinstance(raw_implication, dict):
            preserved_frontend_ready = raw_implication.get("frontend_ready")
    out = _scrub_failed_output(raw_out)
    out["is_valid_strategic_insight"] = False
    analysis = out.get("analysis") or {}
    implication = out.get("implication") or {}
    analysis["is_valid_analysis"] = False
    analysis["confidence"] = min(_clamp_float(analysis.get("confidence"), 0.0), 0.3)
    base_reason = str(analysis.get("reason") or "").strip()
    safe_violations = [_safe_quality_gate_violation_text(item) for item in violations[:3]]
    violation_text = " / ".join(item for item in safe_violations if item)
    analysis["reason"] = (
        f"{base_reason} | quality_gate_failed: {violation_text}"
        if base_reason
        else f"quality_gate_failed: {violation_text}"
    )
    implication["is_valid_implication"] = False
    implication["confidence"] = min(_clamp_float(implication.get("confidence"), 0.0), 0.3)
    implication["evidence_label"] = "insufficient"
    if not preserve_frontend_ready:
        implication.pop("frontend_ready", None)
    elif isinstance(preserved_frontend_ready, dict):
        implication["frontend_ready"] = preserved_frontend_ready
    out["analysis"] = analysis
    out["implication"] = implication
    return out


def _safe_quality_gate_violation_text(violation: Any) -> str:
    text = re.sub(r"\s+", " ", str(violation or "")).strip()
    if not text:
        return ""
    scrubbed = _scrub_failed_output(text)
    if isinstance(scrubbed, str) and scrubbed:
        return scrubbed
    return "근거 범위를 벗어난 고위험 주장 제거"


def _fallback_quality_repair(
    result: dict[str, Any],
    *,
    violations: list[str],
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible alias for old tests; no longer writes template copy."""
    return _minimal_quality_guard(result, integrated_issue=integrated_issue)


def _minimal_quality_guard(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply only mechanical safety fixes, never generate strategic copy.

    LLM self-review owns content repair. This guard only removes unsupported
    numeric drift and contract-role overstatement that can be detected safely.
    """
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    analysis = out.get("analysis") or {}
    implication = out.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}

    if _main_company_is_customer_or_buyer(integrated_issue):
        for key in ("analysis_summary", "market_signal", "impact_reason", "reason"):
            if analysis.get(key):
                analysis[key] = _repair_customer_role_overstatement(str(analysis[key]))
                if _counterparty_guard_violation(
                    analysis[key],
                    label=f"analysis.{key}",
                    integrated_issue=integrated_issue,
                ) or _hard_quality_violation_for_text(
                    analysis[key],
                    label=f"analysis.{key}",
                    integrated_issue=integrated_issue,
                ):
                    analysis[key] = ""
        analysis["strategic_meaning"] = [
            _repair_customer_role_overstatement(item)
            for item in _string_list(analysis.get("strategic_meaning"), max_items=3)
        ]
        analysis["strategic_meaning"] = [
            item
            for index, item in enumerate(analysis["strategic_meaning"], start=1)
            if not _hard_quality_violation_for_text(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
            )
        ]
        for key in ("peer_meaning", "capability_change"):
            if peer.get(key):
                peer[key] = _repair_customer_role_overstatement(str(peer[key]))
                if _hard_quality_violation_for_text(
                    peer[key],
                    label=f"peer_implication.{key}",
                    integrated_issue=integrated_issue,
                ):
                    peer[key] = ""
        for key in ("why_important", "potential_impact"):
            if skax.get(key):
                skax[key] = _repair_customer_role_overstatement(str(skax[key]))
                if _hard_quality_violation_for_text(
                    skax[key],
                    label=f"skax_implication.{key}",
                    integrated_issue=integrated_issue,
                ) or _evidence_scoped_business_claim_violation(
                    skax[key],
                    label=f"skax_implication.{key}",
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                ):
                    skax[key] = ""
        if profile_context and not _has_relevant_peer_profile_context(
            profile_context,
            integrated_issue=integrated_issue,
        ):
            peer["peer_meaning"] = ""
            peer["capability_change"] = ""

    for key in ("analysis_summary", "market_signal", "impact_reason", "reason"):
        label = f"analysis.{key}"
        if analysis.get(key) and (
            _hard_quality_violation_for_text(
                str(analysis[key]),
                label=label,
                integrated_issue=integrated_issue,
            )
            or _scope_expansion_guard_violation(
                str(analysis[key]),
                label=label,
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
        ):
            analysis[key] = ""

    strategic_items = _string_list(analysis.get("strategic_meaning"), max_items=3)
    has_hard_or_scope_strategic_violation = any(
        _hard_quality_violation_for_text(
            item,
            label=f"analysis.strategic_meaning[{index}]",
            integrated_issue=integrated_issue,
        )
        or _scope_expansion_guard_violation(
            item,
            label=f"analysis.strategic_meaning[{index}]",
            integrated_issue=integrated_issue,
            profile_context=profile_context or {},
        )
        for index, item in enumerate(strategic_items, start=1)
    )
    if has_hard_or_scope_strategic_violation:
        analysis["strategic_meaning"] = [
            item
            for index, item in enumerate(strategic_items, start=1)
            if not _hard_quality_violation_for_text(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
            )
            and not _scope_expansion_guard_violation(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
        ]

    for key in ("peer_meaning", "capability_change"):
        label = f"peer_implication.{key}"
        if peer.get(key) and (
            _hard_quality_violation_for_text(
                str(peer[key]),
                label=label,
                integrated_issue=integrated_issue,
            )
            or _scope_expansion_guard_violation(
                str(peer[key]),
                label=label,
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
        ):
            peer[key] = ""

    for key in ("why_important", "potential_impact"):
        if skax.get(key) and (
            _hard_quality_violation_for_text(
                str(skax[key]),
                label=f"skax_implication.{key}",
                integrated_issue=integrated_issue,
            )
            or _evidence_scoped_business_claim_violation(
                str(skax[key]),
                label=f"skax_implication.{key}",
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            or _scope_expansion_guard_violation(
                str(skax[key]),
                label=f"skax_implication.{key}",
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
        ):
            skax[key] = ""

    _repair_result_numeric_grounding(
        analysis=analysis,
        implication=implication,
        integrated_issue=integrated_issue,
    )
    for field in ("opportunities", "threats"):
        skax[field] = [
            item
            for index, item in enumerate(_string_list(skax.get(field), max_items=3), start=1)
            if not _hard_quality_violation_for_text(
                item,
                label=f"skax_implication.{field}[{index}]",
                integrated_issue=integrated_issue,
            )
            and not _evidence_scoped_business_claim_violation(
                item,
                label=f"skax_implication.{field}[{index}]",
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            and not _scope_expansion_guard_violation(
                item,
                label=f"skax_implication.{field}[{index}]",
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
        ]
    skax["recommended_actions"] = [
        action
        for index, action in enumerate(
            _string_list(skax.get("recommended_actions"), max_items=3), start=1
        )
        if not _recommended_action_quality_violation(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        and not _hard_quality_violation_for_text(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
        )
        and not _counterparty_role_action_violation(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
        )
    ]

    out["analysis"] = analysis
    implication["peer_implication"] = peer
    implication["skax_implication"] = skax
    out["implication"] = implication
    if not _has_required_output_structure(out):
        return _mark_quality_gate_failed(
            out,
            [
                "최소 품질 보정 후 필수 analysis/implication 구조가 남지 않았습니다. "
                "근거 없는 시사점이나 대응방향을 새로 만들지 않고 human_review로 넘깁니다."
            ],
        )
    return out


def _safe_event_based_strategic_meanings(
    values: Any,
    *,
    integrated_issue: dict[str, Any],
) -> list[str]:
    safe_items: list[str] = []
    for index, item in enumerate(_string_list(values, max_items=3), start=1):
        if (
            _counterparty_guard_violation(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
            )
            or _hard_quality_violation_for_text(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
            )
            or _weak_analysis_statement(
                item,
                integrated_issue=integrated_issue,
            )
        ):
            continue
        safe_items.append(item)
    if len(safe_items) >= 2:
        return safe_items[:3]

    for candidate in _event_based_strategic_meaning_candidates(integrated_issue):
        if candidate not in safe_items:
            safe_items.append(candidate)
        if len(safe_items) >= 3:
            break
    return safe_items[:3]


def _profile_linked_analysis_field(
    key: str,
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    if key == "analysis_summary":
        return _profile_linked_analysis_summary(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
    if key == "market_signal":
        return _profile_linked_market_signal(integrated_issue)
    if key == "impact_reason":
        return _profile_linked_impact_reason(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
    if key == "reason":
        profile_phrase = _profile_area_phrase(
            profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
        fact = _primary_issue_fact(integrated_issue)
        if profile_phrase:
            return (
                f"{fact} 이 사실을 기준으로 해석했고, 피어 프로필에서는 "
                f"{profile_phrase} 접점만 현재 사건과 연결했습니다."
            )
        return f"{fact} 이 사실을 기준으로 사건 범위 안에서만 해석했습니다."
    return _event_based_analysis_field(key, integrated_issue=integrated_issue)


def _profile_linked_market_signal(integrated_issue: dict[str, Any]) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    if re.search(
        r"구축|센터|인프라|컴퓨팅|데이터\s*센터|GPU|반도체|서버|SPC|특수목적법인",
        _integrated_grounding_text(integrated_issue),
        flags=re.IGNORECASE,
    ):
        return (
            f"{subject or '현재 사건'}에서 구축 범위, 인프라 구성, 단계별 추진 일정이 "
            "함께 제시되어 대규모 인프라 사업의 비교 기준이 구체화되고 있습니다."
        )
    return _event_based_market_signal(integrated_issue)


def _profile_linked_impact_reason(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if subject and profile_phrase:
        return (
            f"{subject}이 확인되면서 피어 프로필의 {profile_phrase} 역량이 "
            "현재 사건의 구축 범위와 추진 구조에 연결되는지 관찰할 수 있습니다."
        )
    return _event_based_impact_reason(integrated_issue)


def _profile_linked_strategic_meanings(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> list[str]:
    fact = _primary_issue_fact(integrated_issue)
    subject = _issue_subject_phrase(integrated_issue)
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    meanings = [fact]
    if subject and profile_phrase:
        meanings.append(
            f"피어 프로필의 {profile_phrase} 맥락과 연결하면, 이번 사건은 "
            f"{subject}에서 필요한 구축 범위와 운영 구조를 확인하는 신호입니다."
        )
    meanings.append(_profile_linked_market_signal(integrated_issue))
    return _normalize_recommended_actions(meanings)[:3]


def _profile_linked_peer_meaning(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    peer: dict[str, Any],
) -> str:
    peer_name = str(peer.get("company_name_ko") or peer.get("company_id") or "타깃 피어").strip()
    fact = _primary_issue_fact(integrated_issue)
    fact_sentence = fact
    if peer_name and not re.search(re.escape(peer_name), fact_sentence, flags=re.IGNORECASE):
        fact_sentence = f"{peer_name}는 {fact_sentence}"
    subject = _issue_subject_phrase(integrated_issue) or "현재 사건"
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if profile_phrase:
        return (
            f"{fact_sentence} 피어 프로필에서는 {profile_phrase}가 "
            f"{_with_particle(subject, '과', '와')} 연결되는 배경으로 확인됩니다. "
            "따라서 이 신호는 역할 확장이나 "
            "성과를 단정하기보다, 해당 피어의 기존 사업 맥락이 현재 대형 과제와 만나는 "
            "관찰 지점으로 해석하는 것이 안전합니다."
        )
    return _event_based_peer_meaning(integrated_issue=integrated_issue, peer=peer)


def _repair_result_numeric_grounding(
    *,
    analysis: dict[str, Any],
    implication: dict[str, Any],
    integrated_issue: dict[str, Any],
) -> None:
    for key in ("analysis_summary", "market_signal", "impact_reason", "reason"):
        if analysis.get(key):
            analysis[key] = _remove_ungrounded_numeric_tokens(
                str(analysis[key]),
                integrated_issue=integrated_issue,
            )
    analysis["strategic_meaning"] = [
        _remove_ungrounded_numeric_tokens(item, integrated_issue=integrated_issue)
        for item in _string_list(analysis.get("strategic_meaning"), max_items=3)
    ]

    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    for key in ("peer_meaning", "capability_change"):
        if peer.get(key):
            peer[key] = _remove_ungrounded_numeric_tokens(
                str(peer[key]),
                integrated_issue=integrated_issue,
            )
    for key in ("why_important", "potential_impact"):
        if skax.get(key):
            skax[key] = _remove_ungrounded_numeric_tokens(
                str(skax[key]),
                integrated_issue=integrated_issue,
            )
    for key in ("opportunities", "threats", "recommended_actions"):
        skax[key] = [
            _remove_ungrounded_numeric_tokens(item, integrated_issue=integrated_issue)
            for item in _string_list(skax.get(key), max_items=3)
        ]
    implication["follow_up_questions"] = [
        _remove_ungrounded_numeric_tokens(item, integrated_issue=integrated_issue)
        for item in _string_list(implication.get("follow_up_questions"), max_items=3)
    ]
    implication["watch_points"] = [
        _remove_ungrounded_numeric_tokens(item, integrated_issue=integrated_issue)
        for item in _string_list(implication.get("watch_points"), max_items=3)
    ]


def _two_section_fact_based_fallback(
    original: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    model: str,
    profile_linkage_evaluation: dict[str, Any] | None = None,
    action_artifact_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative, evidence-scoped fallback when LLM repair overclaims.

    This is intentionally generic: it uses only the current IntegratedIssue,
    profile linkage level, and action plan signals. It does not encode a specific
    article, company, or sector outcome.
    """

    fact_ids = list(_known_fact_ids(integrated_issue))[:5]
    profile_linkage_evaluation = profile_linkage_evaluation or {}
    action_artifact_plan = action_artifact_plan or {}
    peer_linkage_level = _relevant_profile_linkage_level_from_evaluation(
        profile_linkage_evaluation,
        scope="peer",
    )
    skax_linkage_level = _relevant_profile_linkage_level_from_evaluation(
        profile_linkage_evaluation,
        scope="skax",
    )
    has_peer_profile_link = peer_linkage_level in {"high", "medium"}
    peer = {
        "company_id": str(integrated_issue.get("main_company") or ""),
        "company_name_ko": _main_company_display(integrated_issue),
        "sourced_evidence_ids": fact_ids,
    }
    if has_peer_profile_link:
        analysis_summary = _profile_linked_analysis_summary(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        strategic_meaning = [
            _profile_linked_market_signal(integrated_issue),
            _profile_linked_impact_reason(
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            ),
        ]
        market_signal = _profile_linked_market_signal(integrated_issue)
        impact_reason = _profile_linked_impact_reason(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        reason = _profile_linked_analysis_field(
            "reason",
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        peer_meaning = _profile_linked_peer_meaning(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            peer=peer,
        )
        capability_change = _profile_linked_capability_change(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
    else:
        analysis_summary = _event_based_analysis_summary(integrated_issue)
        primary_fact = _primary_issue_fact(integrated_issue)
        strategic_meaning = [
            item
            for item in _event_based_strategic_meaning_candidates(integrated_issue)
            if item != primary_fact
        ][:2]
        market_signal = _event_based_market_signal(integrated_issue)
        impact_reason = _event_based_impact_reason(integrated_issue)
        reason = _event_based_analysis_field("reason", integrated_issue=integrated_issue)
        peer_meaning = _event_based_peer_meaning(integrated_issue=integrated_issue, peer=peer)
        capability_change = _event_based_capability_change(integrated_issue)

    peer_meaning = _fallback_peer_meaning_without_summary_repeat(
        peer_meaning,
        integrated_issue=integrated_issue,
        profile_context=profile_context,
        has_peer_profile_link=has_peer_profile_link,
    )
    skax = _fallback_skax_implication(
        integrated_issue=integrated_issue,
        profile_context=profile_context,
        profile_linkage_level=skax_linkage_level,
        action_artifact_plan=action_artifact_plan,
    )
    profile_linkage_payload = _fallback_profile_linkage_payload(
        profile_linkage_evaluation,
        integrated_issue=integrated_issue,
    )
    skax_response_linkage_payload = _fallback_skax_response_linkage_payload(
        profile_linkage_evaluation,
        integrated_issue=integrated_issue,
        action_artifact_plan=action_artifact_plan,
    )
    result = {
        "is_valid_strategic_insight": True,
        "profile_linkage": profile_linkage_payload,
        "skax_response_linkage": skax_response_linkage_payload,
        "claim_strength": (
            "moderate"
            if (
                profile_linkage_payload.get("linkage_level") in {"high", "medium"}
                or skax_response_linkage_payload.get("response_mode") == "profile_based_action"
            )
            else "cautious"
        ),
        "grounding_summary": {
            "used_fact_ids": fact_ids,
            "used_profile_refs": _fallback_used_profile_refs(profile_linkage_evaluation),
            "ungrounded_claims_removed": [],
        },
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": analysis_summary,
            "strategic_meaning": _normalize_recommended_actions(strategic_meaning)[:3],
            "market_signal": market_signal,
            "impact_level": (original.get("analysis") or {}).get("impact_level") or "low",
            "impact_reason": impact_reason,
            "risk_or_opportunity": _choice(
                (original.get("analysis") or {}).get("risk_or_opportunity"),
                _RISK_OR_OPPORTUNITY,
                "neutral",
            ),
            "confidence": 0.65 if has_peer_profile_link else 0.55,
            "reason": reason,
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": peer["company_id"],
                "company_name_ko": peer["company_name_ko"],
                "peer_meaning": peer_meaning,
                "capability_change": capability_change,
                "sourced_evidence_ids": fact_ids,
            },
            "skax_implication": skax,
            "follow_up_questions": [],
            "watch_points": _fallback_watch_points(integrated_issue),
            "confidence": 0.62 if skax_linkage_level in {"high", "medium"} else 0.52,
            "evidence_label": "moderate" if has_peer_profile_link else "insufficient",
            "provenance": {
                "generator": "StrategicInsightAgent",
                "prompt_version": _PROMPT_VERSION,
                "model": model,
                "used_fact_ids": fact_ids,
                "used_context_layers": [
                    "integrated_issue_fact_fallback",
                    "profile_linkage_fallback",
                    "action_plan_fallback",
                ],
                "run_at": datetime.now(UTC).isoformat(),
            },
        },
    }
    return result


def _fallback_profile_linkage_payload(
    profile_linkage_evaluation: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    company_id = (_companies_from_integrated_issue(integrated_issue) or [""])[0]
    linkage = _profile_linkage_for_company(
        profile_linkage_evaluation,
        company_id=company_id,
        scope="peer",
    )
    matched_areas = _fallback_matched_profile_areas(linkage)
    linkage_level = _choice(
        linkage.get("linkage_level"),
        {"high", "medium", "low", "none"},
        "none",
    )
    implication_mode = str(linkage.get("implication_mode") or "")
    if implication_mode == "profile_based":
        interpretation_strength = "profile_based"
    elif implication_mode == "new_business_signal":
        interpretation_strength = "event_based"
    elif linkage_level in {"high", "medium"}:
        interpretation_strength = "cautious_profile_based"
    else:
        interpretation_strength = "observation_only"
    return _normalize_llm_profile_linkage(
        {
            "peer_company": company_id or _main_company_display(integrated_issue),
            "profile_evidence_available": bool(matched_areas)
            or linkage_level in {"high", "medium"},
            "matched_profile_areas": matched_areas,
            "linkage_level": linkage_level,
            "business_novelty_status": linkage.get("business_novelty_status"),
            "allowed_interpretation_strength": interpretation_strength,
            "reason": str(linkage.get("reason") or "").strip(),
        }
    )


def _with_fallback_linkage_payloads(
    result: dict[str, Any],
    *,
    profile_linkage_evaluation: dict[str, Any],
    integrated_issue: dict[str, Any],
    action_artifact_plan: dict[str, Any],
) -> dict[str, Any]:
    out = dict(result or {})
    normalized_profile_linkage = _normalize_llm_profile_linkage(out.get("profile_linkage"))
    if not normalized_profile_linkage.get(
        "profile_evidence_available"
    ) and not normalized_profile_linkage.get("matched_profile_areas"):
        normalized_profile_linkage = _fallback_profile_linkage_payload(
            profile_linkage_evaluation,
            integrated_issue=integrated_issue,
        )
    normalized_skax_linkage = _normalize_skax_response_linkage(out.get("skax_response_linkage"))
    if not normalized_skax_linkage.get(
        "skax_profile_evidence_available"
    ) and not normalized_skax_linkage.get("matched_skax_areas"):
        normalized_skax_linkage = _fallback_skax_response_linkage_payload(
            profile_linkage_evaluation,
            integrated_issue=integrated_issue,
            action_artifact_plan=action_artifact_plan,
        )
    out["profile_linkage"] = normalized_profile_linkage
    out["skax_response_linkage"] = normalized_skax_linkage
    if not out.get("claim_strength"):
        out["claim_strength"] = (
            "moderate"
            if (
                normalized_profile_linkage.get("linkage_level") in {"high", "medium"}
                or normalized_skax_linkage.get("response_mode") == "profile_based_action"
            )
            else "cautious"
        )
    grounding_summary = _normalize_grounding_summary(
        out.get("grounding_summary"),
        integrated_issue=integrated_issue,
    )
    if not grounding_summary.get("used_profile_refs"):
        grounding_summary["used_profile_refs"] = _fallback_used_profile_refs(
            profile_linkage_evaluation
        )
    out["grounding_summary"] = grounding_summary
    return out


def _fallback_skax_response_linkage_payload(
    profile_linkage_evaluation: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    action_artifact_plan: dict[str, Any],
) -> dict[str, Any]:
    linkage = _profile_linkage_for_company(
        profile_linkage_evaluation,
        company_id="sk_ax",
        scope="skax",
    )
    matched_areas = _fallback_matched_skax_areas(linkage)
    linkage_level = _choice(
        linkage.get("linkage_level"),
        {"high", "medium", "low", "none"},
        "none",
    )
    response_mode = str(linkage.get("implication_mode") or "")
    if response_mode not in {
        "profile_based_action",
        "cautious_action",
        "generic_monitoring_action",
    }:
        if linkage_level in {"high", "medium"}:
            response_mode = "profile_based_action"
        elif linkage_level == "low":
            response_mode = "cautious_action"
        else:
            response_mode = "generic_monitoring_action"
    if not matched_areas and response_mode == "profile_based_action":
        response_mode = "cautious_action"
    issue_terms = sorted(_action_plan_issue_terms(action_artifact_plan))[:8]
    focus_terms = issue_terms or [_issue_subject_phrase(integrated_issue)]
    focus_terms = [term for term in focus_terms if term]
    return _normalize_skax_response_linkage(
        {
            "skax_profile_evidence_available": bool(matched_areas),
            "matched_skax_areas": matched_areas,
            "response_mode": response_mode,
            "response_focus": focus_terms[:5],
            "internal_checkpoints": _fallback_internal_checkpoints(
                focus_terms=focus_terms,
                action_artifact_plan=action_artifact_plan,
            ),
            "recommended_focus": _fallback_recommended_focus(
                focus_terms=focus_terms,
                linkage=linkage,
            ),
            "monitoring_points": _fallback_watch_points(integrated_issue),
            "reason": str(linkage.get("reason") or "").strip(),
        }
    )


def _fallback_matched_skax_areas(linkage: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "business_line": item["business_line"],
            "business_area": item["business_area"],
            "profile_area_name": item["profile_area_name"],
            "matched_capabilities": item["matched_capabilities"],
            "matched_products_or_services": item["matched_products_or_services"],
            "matched_issue_terms": item["matched_issue_terms"],
            "evidence_text": item["evidence_text"],
            "why_relevant_to_issue": item["why_relevant_to_issue"],
            "profile_source_ref": item["profile_source_ref"],
            "specificity_level": item["specificity_level"],
        }
        for item in _fallback_matched_profile_areas(linkage)
    ]
