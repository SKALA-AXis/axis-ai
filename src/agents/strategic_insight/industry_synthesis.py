"""strategic_insight industry_synthesis — extracted from facade (move-only)."""

from __future__ import annotations

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


def _industry_or_market_infra_watch_only_decision(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding.strip():
        return {}
    if _has_direct_peer_action_signal(integrated_issue):
        return {}
    scope = _industry_signal_scope(
        integrated_issue=integrated_issue,
        classification=classification,
    )
    if scope not in {"industry_signal", "market_infra_signal"}:
        return {}
    primary_actor_type = _primary_actor_type_for_issue(integrated_issue)
    return {
        "decision_type": "watch_only_industry_signal",
        "watch_only": True,
        "signal_scope": scope,
        "direct_peer_action": False,
        "peer_mention_only": _has_peer_mention(integrated_issue),
        "primary_actor_type": primary_actor_type,
        "reason": (
            "피어사의 직접 실행 사실보다 산업 구조/시장 인프라 변화 신호가 중심이라 "
            "peer 카드뉴스용 시사점/대응방향을 생성하지 않았습니다."
        ),
        "evidence": _industry_signal_evidence_summary(
            integrated_issue=integrated_issue,
            classification=classification,
            signal_scope=scope,
            primary_actor_type=primary_actor_type,
        ),
    }


def _industry_signal_scope(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> str:
    grounding = _integrated_grounding_text(integrated_issue)
    event_type = str(
        classification.get("event_type")
        or integrated_issue.get("cluster_event_type")
        or integrated_issue.get("event_type")
        or ""
    ).casefold()
    if _has_market_infra_signal(grounding):
        return "market_infra_signal"
    if event_type in {"industry_trend", "market_trend", "policy", "regulation"}:
        return "industry_signal"
    if _has_sizable_tech_event_signal(integrated_issue):
        return "industry_signal"
    if re.search(r"산업\s*구조|시장\s*구조|경쟁\s*기준|시장\s*전망|업계\s*전망", grounding):
        return "industry_signal"
    return ""


def _industry_signal_evidence_summary(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    signal_scope: str,
    primary_actor_type: str,
) -> dict[str, Any]:
    fact_lines = _issue_fact_lines(integrated_issue)
    return {
        "category": "industry_signal",
        "signal_scope": signal_scope,
        "primary_actor_type": primary_actor_type,
        "event_type": str(
            classification.get("event_type")
            or integrated_issue.get("cluster_event_type")
            or integrated_issue.get("event_type")
            or ""
        ),
        "direct_peer_action": False,
        "peer_mention_only": _has_peer_mention(integrated_issue),
        "market_infra_lines": [line for line in fact_lines if _has_market_infra_signal(line)][:5],
        "business_signal_lines": [line for line in fact_lines if _has_direct_business_signal(line)][
            :5
        ],
        "fact_line_count": len(fact_lines),
    }


def _industry_frontend_ready_from_decision(
    *,
    integrated_issue: dict[str, Any],
    skip_decision: dict[str, Any],
) -> dict[str, Any]:
    axes = _industry_frontend_axes(integrated_issue)
    if not axes:
        return {}
    candidate_items: list[dict[str, Any]] = []
    used_axis_keys: set[str] = set()
    used_anchor_norms: set[str] = set()
    for axis in axes:
        axis_key = str(axis.get("strategic_axis") or "").strip()
        anchors = _string_list(axis.get("event_anchor_terms"), max_items=6)
        anchor_key = "::".join(sorted(_anchor_norm(anchor) for anchor in anchors[:3]))
        if (
            not axis_key
            or not anchors
            or axis_key in used_axis_keys
            or anchor_key in used_anchor_norms
        ):
            continue
        item = _industry_frontend_item(
            axis,
            integrated_issue=integrated_issue,
            skip_decision=skip_decision,
        )
        if item:
            candidate_items.append(item)
            used_axis_keys.add(axis_key)
            used_anchor_norms.add(anchor_key)
        if len(candidate_items) >= 3:
            break
    items = _merge_similar_industry_frontend_items(
        candidate_items,
        integrated_issue=integrated_issue,
        skip_decision=skip_decision,
    )
    if not items:
        return {}
    return {
        "source": "industry_signal_direct",
        "signal_scope": skip_decision.get("signal_scope") or "industry_signal",
        "display_policy": "industry_only",
        "direct_peer_action": False,
        "primary_actor_type": skip_decision.get("primary_actor_type"),
        "items": items,
    }


def _industry_frontend_axes(integrated_issue: dict[str, Any]) -> list[dict[str, Any]]:
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding.strip():
        return []
    fact_lines = _issue_fact_lines(integrated_issue)
    axes: list[dict[str, Any]] = []
    tech_event_anchors = _ordered_anchor_matches(
        grounding,
        (
            r"스마트테크\s*코리아\s*2026",
            r"STK\s*2026",
            r"코엑스",
            r"\d+\s*개국",
            r"\d+\s*개사",
            r"AI",
            r"인공지능",
            r"자동화",
            r"로봇",
            r"스마트\s*제조",
            r"디지털\s*유통·물류",
            r"스마트테크",
            r"산업\s*전\s*과정",
        ),
    )
    if _has_sizable_tech_event_signal(integrated_issue) and len(tech_event_anchors) >= 2:
        anchors = tech_event_anchors[:6]
        evidence_lines = _industry_lines_with_anchors(fact_lines, anchors)
        domain_lines = [
            line
            for line in fact_lines
            if re.search(
                r"AI|인공지능|로봇|스마트\s*제조|디지털\s*유통·물류|물류|자동화",
                line,
                flags=re.IGNORECASE,
            )
        ]
        domain_lines = sorted(
            domain_lines,
            key=_technology_event_domain_line_score,
            reverse=True,
        )
        evidence_lines = _dedupe_keep_order([*domain_lines[:2], *evidence_lines])[:2]
        axes.append(
            {
                "strategic_axis": "technology_event_adoption_signal",
                "event_anchor_terms": anchors,
                "decision_criteria": _industry_decision_criteria_from_issue(
                    _industry_axis_context_text(evidence_lines, grounding=grounding),
                    anchors=anchors,
                    axis_key="technology_event_adoption_signal",
                ),
                "evidence_lines": evidence_lines,
            }
        )
    infra_anchors = _ordered_anchor_matches(
        grounding,
        (
            r"AI\s*인프라",
            r"인공지능\s*인프라",
            r"AI\s*팩토리",
            r"데이터\s*센터",
            r"데이터센터",
            r"GPU",
            r"그래픽처리장치",
            r"AI\s*컴퓨팅",
            r"컴퓨팅\s*센터",
            r"클라우드\s*인프라",
            r"반도체\s*인프라",
        ),
    )
    numeric_or_capacity = [
        anchor
        for anchor in _specific_event_anchors_for_frontend(integrated_issue)
        if re.search(r"[0-9]|조원|억원|장|개|데이터\s*센터|GPU", anchor, flags=re.IGNORECASE)
    ][:4]
    if len(set(infra_anchors + numeric_or_capacity)) >= 2:
        anchors = _dedupe_keep_order([*infra_anchors, *numeric_or_capacity])[:6]
        evidence_lines = _industry_lines_with_anchors(fact_lines, anchors)
        axes.append(
            {
                "strategic_axis": "market_infra_capacity",
                "event_anchor_terms": anchors,
                "decision_criteria": _industry_decision_criteria_from_issue(
                    _industry_axis_context_text(evidence_lines, grounding=grounding),
                    anchors=anchors,
                    axis_key="market_infra_capacity",
                ),
                "evidence_lines": evidence_lines,
            }
        )
    actor_patterns = [
        r"글로벌\s*(?:벤더|기업|빅테크)",
        r"국내\s*(?:주요\s*)?(?:기업|그룹)",
        r"[가-힣A-Za-z0-9&._-]+그룹",
        r"컨소시엄",
        r"협력",
        r"공동",
        r"정부",
    ]
    global_alias_pattern = _global_company_alias_pattern()
    if global_alias_pattern:
        actor_patterns.append(global_alias_pattern)
    actor_anchors = _ordered_anchor_matches(grounding, actor_patterns)
    actor_anchors = _prefer_collective_actor_anchors(actor_anchors)
    relationship_anchors = _ordered_anchor_matches(
        grounding,
        (r"협력", r"공동", r"파트너십", r"투자", r"구축", r"확장"),
    )
    if len(set(actor_anchors)) >= 2 or (actor_anchors and relationship_anchors):
        anchors = _dedupe_keep_order([*actor_anchors, *relationship_anchors])[:6]
        evidence_lines = _industry_lines_with_anchors(fact_lines, anchors)
        axes.append(
            {
                "strategic_axis": "multi_actor_coordination",
                "event_anchor_terms": anchors,
                "decision_criteria": _industry_decision_criteria_from_issue(
                    _industry_axis_context_text(evidence_lines, grounding=grounding),
                    anchors=anchors,
                    axis_key="multi_actor_coordination",
                ),
                "evidence_lines": evidence_lines,
            }
        )
    adoption_anchors = _ordered_anchor_matches(
        grounding,
        (
            r"기업\s*AI",
            r"AI\s*도입",
            r"업무\s*자동화",
            r"AI\s*서비스",
            r"클라우드\s*전환",
            r"운영\s*요구",
        ),
    )
    if len(adoption_anchors) >= 2:
        anchors = adoption_anchors[:6]
        evidence_lines = _industry_lines_with_anchors(fact_lines, anchors)
        axes.append(
            {
                "strategic_axis": "enterprise_ai_adoption",
                "event_anchor_terms": anchors,
                "decision_criteria": _industry_decision_criteria_from_issue(
                    _industry_axis_context_text(evidence_lines, grounding=grounding),
                    anchors=anchors,
                    axis_key="enterprise_ai_adoption",
                ),
                "evidence_lines": evidence_lines,
            }
        )
    return axes


def _industry_frontend_item(
    axis: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    skip_decision: dict[str, Any],
) -> dict[str, Any]:
    anchors = _string_list(axis.get("event_anchor_terms"), max_items=6)
    if len(anchors) < 2:
        return {}
    anchor_phrase = _anchor_phrase(anchors, max_items=3)
    decision_criteria = _string_list(axis.get("decision_criteria"), max_items=4)
    if not decision_criteria:
        decision_criteria = _industry_decision_criteria_from_issue(
            _integrated_grounding_text(integrated_issue),
            anchors=anchors,
            axis_key=str(axis.get("strategic_axis") or ""),
        )
    evidence_lines = _string_list(axis.get("evidence_lines"), max_items=2)
    axis_key = str(axis.get("strategic_axis") or "")
    evidence_text = _industry_evidence_sentence(
        evidence_lines,
        axis_key=axis_key,
        anchors=anchors,
        decision_criteria=decision_criteria,
        primary_actor_type=str(skip_decision.get("primary_actor_type") or ""),
    )
    action_block = _industry_source_grounded_action_block(
        axis_key=axis_key,
        anchors=anchors,
        decision_criteria=decision_criteria,
        evidence_lines=evidence_lines,
    )
    return {
        "strategic_axis": axis.get("strategic_axis"),
        "event_anchor_terms": anchors,
        "decision_criteria": decision_criteria,
        "evidence_lines": evidence_lines,
        "key_implication": {
            "source": "industry_signal_direct",
            "frame": "industry_signal",
            "claim_type": "self_or_market_signal",
            "claim_strength": "cautious",
            "evidence_mode": "event_based",
            "event_anchor_terms": anchors,
            "decision_criteria": decision_criteria,
            "sentence": _industry_implication_sentence(
                str(axis.get("strategic_axis") or ""),
                anchor_phrase,
                decision_criteria=decision_criteria,
                anchors=anchors,
            ),
            "evidence_sentence": evidence_text,
        },
        "suggested_action": action_block,
    }


def _merge_similar_industry_frontend_items(
    candidate_items: Sequence[dict[str, Any]],
    *,
    integrated_issue: dict[str, Any],
    skip_decision: dict[str, Any],
) -> list[dict[str, Any]]:
    if not candidate_items:
        return []
    groups: list[list[dict[str, Any]]] = []
    for item in candidate_items:
        if not isinstance(item, dict) or not item:
            continue
        placed = False
        for group in groups:
            if not all(
                _industry_frontend_items_are_separable(item, existing) for existing in group
            ):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
        if len(groups) >= 3:
            # Keep collecting only inside the first three meaningful groups.
            continue
    merged: list[dict[str, Any]] = []
    for group in groups[:3]:
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged_item = _merged_industry_frontend_item(
                group,
                integrated_issue=integrated_issue,
                skip_decision=skip_decision,
            )
            if merged_item:
                merged.append(merged_item)
    return merged[:3]


def _merged_industry_frontend_item(
    items: Sequence[dict[str, Any]],
    *,
    integrated_issue: dict[str, Any],
    skip_decision: dict[str, Any],
) -> dict[str, Any]:
    anchors: list[str] = []
    criteria: list[str] = []
    evidence_lines: list[str] = []
    axis_keys: list[str] = []
    for item in items:
        axis_keys.append(str(item.get("strategic_axis") or "").strip())
        anchors.extend(_string_list(item.get("event_anchor_terms"), max_items=8))
        criteria.extend(_string_list(item.get("decision_criteria"), max_items=8))
        evidence_lines.extend(_string_list(item.get("evidence_lines"), max_items=3))
    merged_axis = {
        "strategic_axis": "+".join(value for value in _dedupe_keep_order(axis_keys) if value)
        or "industry_signal",
        "event_anchor_terms": _dedupe_keep_order(anchors)[:6],
        "decision_criteria": _dedupe_keep_order(criteria)[:4],
        "evidence_lines": _dedupe_keep_order(evidence_lines)[:2],
    }
    return _industry_frontend_item(
        merged_axis,
        integrated_issue=integrated_issue,
        skip_decision=skip_decision,
    )


def _industry_frontend_items_are_separable(
    item: dict[str, Any],
    existing: dict[str, Any],
) -> bool:
    checks = [
        _industry_sets_are_substantially_different(
            _industry_item_conclusion_terms(item),
            _industry_item_conclusion_terms(existing),
        ),
        _industry_sets_are_substantially_different(
            _industry_item_action_terms(item),
            _industry_item_action_terms(existing),
        ),
        _industry_sets_are_substantially_different(
            _industry_item_anchor_norms(item),
            _industry_item_anchor_norms(existing),
        ),
        _industry_sets_are_substantially_different(
            _industry_item_decision_criteria(item),
            _industry_item_decision_criteria(existing),
        ),
        _industry_sets_are_substantially_different(
            _industry_item_action_result_terms(item),
            _industry_item_action_result_terms(existing),
        ),
    ]
    return sum(1 for value in checks if value) >= 3


def _industry_sets_are_substantially_different(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    shared_ratio = len(left & right) / max(len(left | right), 1)
    return shared_ratio < 0.45 and len(left - right) >= 2 and len(right - left) >= 1


def _industry_frontend_item_is_distinct(
    item: dict[str, Any],
    existing_items: Sequence[dict[str, Any]],
) -> bool:
    if not existing_items:
        return True
    new_axis = str(item.get("strategic_axis") or "").strip()
    new_terms = _industry_item_conclusion_terms(item)
    for existing in existing_items:
        existing_axis = str(existing.get("strategic_axis") or "").strip()
        if new_axis and existing_axis and new_axis == existing_axis:
            return False
        existing_terms = _industry_item_conclusion_terms(existing)
        if not new_terms or not existing_terms:
            continue
        shared_ratio = len(new_terms & existing_terms) / max(len(new_terms | existing_terms), 1)
        unique_terms = new_terms - existing_terms
        if shared_ratio >= 0.62 and len(unique_terms) < 2:
            return False
        new_criteria = _industry_item_decision_criteria(item)
        existing_criteria = _industry_item_decision_criteria(existing)
        if new_criteria and existing_criteria:
            criteria_shared_ratio = len(new_criteria & existing_criteria) / max(
                len(new_criteria | existing_criteria),
                1,
            )
            if criteria_shared_ratio >= 0.55 and len(new_criteria - existing_criteria) < 2:
                return False
    return True


def _industry_item_conclusion_terms(item: dict[str, Any]) -> set[str]:
    key_block = item.get("key_implication") or {}
    if not isinstance(key_block, dict):
        return set()
    sentence = str(key_block.get("sentence") or "")
    for anchor in _string_list(item.get("event_anchor_terms"), max_items=8):
        sentence = sentence.replace(anchor, " ")
    terms = _frontend_ready_role_terms(sentence)
    return {
        term
        for term in terms
        if term
        not in {
            "산업",
            "신호",
            "논의",
            "관련",
            "특정",
            "피어",
            "피어사",
            "실행",
        }
    }


def _industry_item_action_terms(item: dict[str, Any]) -> set[str]:
    action_block = item.get("suggested_action") or {}
    if not isinstance(action_block, dict):
        return set()
    text = " ".join(
        [
            str(action_block.get("sentence") or ""),
            str(action_block.get("evidence_sentence") or ""),
        ]
    )
    for anchor in _string_list(item.get("event_anchor_terms"), max_items=8):
        text = text.replace(anchor, " ")
    terms = _frontend_ready_role_terms(text)
    return {
        term
        for term in terms
        if term
        not in {
            "sk",
            "ax",
            "자사",
            "시장",
            "신호",
            "기준",
            "판단",
            "입력",
            "근거",
            "조건",
        }
    }


def _industry_item_anchor_norms(item: dict[str, Any]) -> set[str]:
    return {
        _anchor_norm(anchor)
        for anchor in _string_list(item.get("event_anchor_terms"), max_items=8)
        if _anchor_norm(anchor)
    }


def _industry_item_decision_criteria(item: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("key_implication", "suggested_action"):
        block = item.get(key) or {}
        if not isinstance(block, dict):
            continue
        result.update(_string_list(block.get("decision_criteria"), max_items=8))
    result.update(_string_list(item.get("decision_criteria"), max_items=8))
    return {_anchor_norm(value) for value in result if _anchor_norm(value)}


def _industry_item_action_result_terms(item: dict[str, Any]) -> set[str]:
    result_phrase = _industry_action_result_phrase(
        _string_list(item.get("decision_criteria"), max_items=8)
    )
    return _frontend_ready_role_terms(result_phrase)


def _industry_criteria_flags(
    criteria: Sequence[str],
    anchors: Sequence[str] | None = None,
) -> dict[str, bool]:
    normalized = {_anchor_norm(item) for item in criteria}
    value = " ".join(str(anchor or "") for anchor in anchors or ())
    has_actor_or_partner = bool(
        {
            _anchor_norm("참여 주체"),
            _anchor_norm("파트너십 필요성"),
        }
        & normalized
    ) or bool(re.search(r"협력|제휴|파트너|컨소시엄|그룹|기업|정부|기관|참여", value, flags=re.I))
    has_infra_or_supply = bool(
        {
            _anchor_norm("기술 공급 구조"),
            _anchor_norm("데이터/인프라 준비 수준"),
        }
        & normalized
    ) or bool(
        re.search(r"데이터\s*센터|데이터센터|GPU|AI\s*팩토리|컴퓨팅|인프라", value, flags=re.I)
    )
    has_customer_or_system = bool(
        {
            _anchor_norm("고객 적용 가능성"),
            _anchor_norm("고객 적용 방식"),
            _anchor_norm("기존 시스템 접점"),
        }
        & normalized
    ) or bool(re.search(r"고객|업무|시스템|ERP|메일|문서|데이터베이스|자동화", value, flags=re.I))
    has_operation = bool({_anchor_norm("실행 조건")} & normalized)
    has_investment = bool(
        {
            _anchor_norm("투자 조건"),
            _anchor_norm("비용 부담"),
            _anchor_norm("후속 사업화 조건"),
        }
        & normalized
    )
    has_policy = bool({_anchor_norm("규제/정책 대응 조건")} & normalized)
    has_financial = bool({_anchor_norm("내부 관리 지표")} & normalized)
    return {
        "actor_or_partner": has_actor_or_partner,
        "infra_or_supply": has_infra_or_supply,
        "customer_or_system": has_customer_or_system,
        "operation": has_operation,
        "investment": has_investment,
        "policy": has_policy,
        "financial": has_financial,
    }


def _industry_role_structure_subject(
    criteria: Sequence[str],
    anchors: Sequence[str] | None = None,
) -> str:
    flags = _industry_criteria_flags(criteria, anchors)
    if flags["actor_or_partner"] and flags["infra_or_supply"]:
        return "참여 주체와 인프라 준비 조건이 함께 드러난 논의"
    if flags["infra_or_supply"] and (flags["customer_or_system"] or flags["operation"]):
        return "인프라 준비 수준과 실행 조건이 함께 드러난 논의"
    if flags["actor_or_partner"] and (flags["customer_or_system"] or flags["operation"]):
        return "여러 참여 주체와 실행 단계가 함께 드러난 논의"
    if flags["customer_or_system"] and flags["operation"]:
        return "시스템 접점과 실행 조건이 함께 드러난 논의"
    if flags["investment"] and flags["infra_or_supply"]:
        return "투자 조건과 기술 준비가 함께 드러난 논의"
    return ""


def _industry_role_structure_change_clause(
    criteria: Sequence[str],
    anchors: Sequence[str] | None = None,
) -> str:
    flags = _industry_criteria_flags(criteria, anchors)
    if (
        flags["actor_or_partner"]
        and flags["infra_or_supply"]
        and (flags["customer_or_system"] or flags["operation"])
    ):
        return (
            "기술 확보 자체보다 참여 구조와 실행 가능성을 함께 보는 "
            "경쟁 기준으로 이어질 수 있음을 보여준다."
        )
    if flags["infra_or_supply"] and (flags["customer_or_system"] or flags["operation"]):
        return "기술 준비 수준을 운영 조건까지 함께 보는 흐름으로 확장될 수 있음을 보여준다."
    if flags["actor_or_partner"] and (flags["customer_or_system"] or flags["operation"]):
        return (
            "협력 여부보다 각 주체의 역량이 실제 적용 단계에서 "
            "어떻게 맞물리는지가 중요해질 수 있음을 보여준다."
        )
    if flags["customer_or_system"] and flags["operation"]:
        return "기능 제공보다 시스템 접점과 실행 조건을 함께 보는 흐름을 보여준다."
    if flags["investment"] and flags["infra_or_supply"]:
        return "투자 규모보다 실제 적용 조건과 후속 운영 가능성을 함께 따지는 흐름을 보여준다."
    return ""


def _industry_implication_sentence(
    axis_key: str,
    anchor_phrase: str,
    *,
    decision_criteria: Sequence[str],
    anchors: Sequence[str] | None = None,
) -> str:
    if _is_technology_event_adoption_axis(axis_key, anchors or ()):
        return (
            "대규모 기술 전시의 경쟁 포인트가 개별 제품 소개보다 "
            "AI·로봇·스마트제조·물류처럼 실제 산업 적용 분야를 함께 보여주는 쪽으로 "
            "넓어지고 있다."
        )
    role_subject = _industry_role_structure_subject(decision_criteria, anchors or ())
    role_change = _industry_role_structure_change_clause(decision_criteria, anchors or ())
    if role_subject and role_change:
        return f"{role_subject}는 {role_change}"
    anchor_reading = _industry_anchor_market_reading(
        anchors or (),
        decision_criteria=decision_criteria,
    )
    if anchor_reading:
        return f"이번 시장 신호는 {anchor_reading}"
    market_reading = _industry_market_reading_phrase(decision_criteria)
    if market_reading:
        return (
            "이번 시장 신호는 시장의 관심이 개별 기술 발표보다 "
            f"{market_reading} 쪽으로 넓어질 수 있음을 보여준다."
        )
    return (
        "이번 시장 신호는 특정 실행 주체보다 참여 구조와 적용 조건을 함께 "
        "읽어야 하는 시장 신호로 볼 수 있다."
    )


def _industry_axis_context_text(lines: Sequence[str], *, grounding: str) -> str:
    selected = _string_list(lines, max_items=3)
    if selected:
        return "\n".join(selected)
    return str(grounding or "")


def _industry_decision_criteria_from_issue(
    text: Any,
    *,
    anchors: Sequence[str],
    axis_key: str,
) -> list[str]:
    del axis_key
    value = "\n".join([str(text or ""), *[str(anchor or "") for anchor in anchors]])
    criteria_patterns: tuple[tuple[str, str], ...] = (
        ("투자 조건", r"투자|예산|사업비|규모|조원|억원|자금|CAPEX"),
        ("비용 부담", r"비용|부담|원가|가격"),
        ("실행 조건", r"운영|관제|책임|관리|유지|서비스\s*개시"),
        ("기술 공급 구조", r"공급|벤더|기술|GPU|그래픽처리장치|반도체|클라우드|플랫폼|모델|장비"),
        ("고객 적용 가능성", r"고객|적용|도입|사용|업무|기업\s*AI"),
        ("기존 시스템 접점", r"기존\s*시스템|시스템\s*연계|연계|ERP|전환|업무\s*시스템"),
        ("파트너십 필요성", r"협력|제휴|공동|파트너|협약|MOU"),
        ("참여 주체", r"참여|주체|그룹|기업|기관|정부|벤더|컨소시엄|총수|CEO"),
        ("후속 사업화 조건", r"후속|추가\s*논의|사업화|상용화|출시|계약|확대"),
        ("규제/정책 대응 조건", r"정부|정책|규제|공공|과기정통부|금융위|공정위"),
        (
            "데이터/인프라 준비 수준",
            r"데이터|인프라|데이터\s*센터|데이터센터|AI\s*팩토리|컴퓨팅|GPU",
        ),
        ("고객 적용 방식", r"제안|고객|서비스|패키지|솔루션"),
        ("내부 관리 지표", r"지표|성과|매출|비중|수익|모니터링"),
    )
    matched: list[str] = []
    for label, pattern in criteria_patterns:
        if re.search(pattern, value, flags=re.IGNORECASE):
            matched.append(label)
    return matched[:4]


def _industry_lines_with_anchors(lines: Sequence[str], anchors: Sequence[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        if _text_has_anchor_term(line, anchors):
            result.append(str(line).strip())
        if len(result) >= 2:
            break
    return result


def _industry_evidence_sentence(
    evidence_lines: Sequence[str],
    *,
    axis_key: str = "",
    anchors: Sequence[str],
    decision_criteria: Sequence[str],
    primary_actor_type: str,
) -> str:
    if _is_technology_event_adoption_axis(axis_key, anchors):
        scale = _technology_event_scale_phrase([*anchors, *evidence_lines])
        domains = _technology_event_domain_phrase([*anchors, *evidence_lines])
        event_name = _technology_event_name_phrase([*anchors, *evidence_lines])
        return (
            f"{event_name}에는 {scale}가 참가했고 {domains}가 함께 소개돼, "
            "전시의 무게가 개최 사실보다 산업별 적용 장면을 확인하는 쪽에 놓여 있다."
        )
    primary = _anchor_phrase(anchors, max_items=2)
    line = str(
        next(
            (item for item in evidence_lines if _is_substantive_industry_evidence_line(item)),
            next((item for item in evidence_lines if str(item).strip()), ""),
        )
    ).strip()
    evidence_reading = _industry_evidence_reading_phrase(decision_criteria)
    actor_context = {
        "global_vendor": "글로벌 벤더 중심의 논의에서",
        "public_sector": "정부·공공 주체가 포함된 흐름에서",
        "multi_actor": "복수 주체가 함께 언급된 흐름에서",
    }.get(primary_actor_type, "이 사건에서")
    if line:
        fact_clause = _short_fact_clause(line)
        if evidence_reading:
            return (
                f"{actor_context} {fact_clause} {evidence_reading}이 함께 드러나 "
                f"{primary} 논의가 실제 적용 조건과 연결된다."
            )
        return (
            f"{actor_context} {fact_clause} {primary} 논의가 단순 발표보다 "
            "참여 구조와 적용 조건을 함께 포함하는 흐름으로 이어진다."
        )
    if evidence_reading:
        return (
            f"{actor_context} {primary}와 {evidence_reading}이 함께 언급되어 "
            "시장 변화가 기술 발표보다 실행 구조와 맞물려 있음을 보여준다."
        )
    return (
        f"{actor_context} {primary}가 반복적으로 제시되어, 해당 논의가 "
        "참여 주체와 적용 조건이 함께 묶이는 흐름임을 보여준다."
    )


def _industry_source_grounded_action_block(
    *,
    axis_key: str,
    anchors: Sequence[str],
    decision_criteria: Sequence[str],
    evidence_lines: Sequence[str],
) -> dict[str, Any]:
    anchor_phrase = _anchor_phrase(anchors, max_items=3)
    fact_line = _industry_primary_action_evidence_line(evidence_lines)
    action_sentence = _industry_source_grounded_action_sentence(
        anchor_phrase=anchor_phrase,
        fact_line=fact_line,
    )
    evidence_sentence = _industry_source_grounded_action_evidence(
        anchor_phrase=anchor_phrase,
        fact_line=fact_line,
    )
    return {
        "source": "industry_signal_direct",
        "frame": "industry_response_check",
        "claim_type": "internal_strategy_check",
        "claim_strength": "cautious",
        "evidence_mode": "source_grounded",
        "event_anchor_terms": list(anchors),
        "decision_criteria": list(decision_criteria),
        "sentence": action_sentence,
        "evidence_sentence": evidence_sentence,
        "strategic_axis": axis_key,
    }


def _industry_source_grounded_action_sentence(
    *,
    anchor_phrase: str,
    fact_line: str,
) -> str:
    if fact_line:
        fact = _short_fact_clause(fact_line, max_chars=96)
        return f"SK AX는 {fact} {anchor_phrase} 관련 대응 필요 여부를 원문 기준으로 확인한다."
    return f"SK AX는 {anchor_phrase} 관련 대응 필요 여부를 원문 기준으로 확인한다."


def _industry_source_grounded_action_evidence(
    *,
    anchor_phrase: str,
    fact_line: str,
) -> str:
    if fact_line:
        fact = _short_fact_clause(fact_line, max_chars=118)
        return f"{fact} 이 문장이 {anchor_phrase} 관련 대응 검토의 근거다."
    return f"{anchor_phrase}가 원문 근거에서 확인된다."


def _industry_primary_action_evidence_line(evidence_lines: Sequence[str]) -> str:
    return str(
        next(
            (item for item in evidence_lines if _is_substantive_industry_evidence_line(item)),
            next((item for item in evidence_lines if str(item).strip()), ""),
        )
    ).strip()


def _industry_criteria_phrase(criteria: Sequence[str], *, max_items: int = 3) -> str:
    return "·".join(_dedupe_keep_order(_string_list(criteria, max_items=max_items)))


def _industry_market_reading_phrase(criteria: Sequence[str]) -> str:
    normalized = {_anchor_norm(item) for item in criteria}
    parts: list[str] = []
    if {_anchor_norm("데이터/인프라 준비 수준"), _anchor_norm("기술 공급 구조")} & normalized:
        parts.append("기술을 실제로 운영할 기반과 공급 구조")
    if {_anchor_norm("참여 주체"), _anchor_norm("파트너십 필요성")} & normalized:
        parts.append("여러 주체가 역할을 나누는 협력 구조")
    if {_anchor_norm("고객 적용 가능성"), _anchor_norm("고객 적용 방식")} & normalized:
        parts.append("업무 시스템과 맞닿는 방식")
    if {_anchor_norm("실행 조건"), _anchor_norm("기존 시스템 접점")} & normalized:
        parts.append("기존 시스템 연결 방식")
    if {_anchor_norm("투자 조건"), _anchor_norm("비용 부담")} & normalized:
        parts.append("투자 부담과 실행 가능성")
    if {_anchor_norm("후속 사업화 조건"), _anchor_norm("규제/정책 대응 조건")} & normalized:
        parts.append("후속 사업화와 정책 조건")
    if {_anchor_norm("내부 관리 지표")} & normalized:
        parts.append("성과를 추적할 관리 기준")
    return _natural_join(_dedupe_keep_order(parts)[:2])


def _industry_evidence_reading_phrase(criteria: Sequence[str]) -> str:
    normalized = {_anchor_norm(item) for item in criteria}
    parts: list[str] = []
    if {_anchor_norm("참여 주체"), _anchor_norm("파트너십 필요성")} & normalized:
        parts.append("참여 주체 간 협력 관계")
    if {_anchor_norm("데이터/인프라 준비 수준"), _anchor_norm("기술 공급 구조")} & normalized:
        parts.append("기술·인프라를 갖추는 방식")
    if {_anchor_norm("실행 조건"), _anchor_norm("기존 시스템 접점")} & normalized:
        parts.append("시스템 연결 조건")
    if {_anchor_norm("고객 적용 가능성"), _anchor_norm("고객 적용 방식")} & normalized:
        parts.append("시스템 접점")
    if {_anchor_norm("투자 조건"), _anchor_norm("비용 부담")} & normalized:
        parts.append("투자와 비용 부담")
    if {_anchor_norm("후속 사업화 조건"), _anchor_norm("규제/정책 대응 조건")} & normalized:
        parts.append("후속 사업화나 정책 조건")
    if {_anchor_norm("내부 관리 지표")} & normalized:
        parts.append("성과 추적 필요성")
    return _natural_join(_dedupe_keep_order(parts)[:2])


def _industry_anchor_market_reading(
    anchors: Sequence[str],
    *,
    decision_criteria: Sequence[str] | None = None,
) -> str:
    value = " ".join(str(anchor or "") for anchor in anchors)
    if not value.strip():
        return ""
    flags = _industry_criteria_flags(decision_criteria or (), anchors)
    if (
        flags["infra_or_supply"]
        and (flags["operation"] or flags["customer_or_system"] or flags["actor_or_partner"])
        and re.search(r"데이터\s*센터|데이터센터|GPU|AI\s*팩토리|컴퓨팅|인프라", value, flags=re.I)
    ):
        return (
            "기술 확보 자체보다 참여 구조와 구축 범위를 나눠 보는 "
            "경쟁 기준으로 이어질 수 있음을 보여준다."
        )
    if flags["customer_or_system"] and re.search(
        r"고객|업무|시스템|ERP|메일|문서|데이터베이스|자동화", value, flags=re.I
    ):
        return (
            "기업 도입 기준이 기능 소개보다 업무 시스템 접점과 "
            "시스템 연결성으로 이동할 수 있음을 보여준다."
        )
    if (flags["operation"] or flags["customer_or_system"]) and re.search(
        r"계약|수주|공급|구축|운영|운용|실증|도입",
        value,
        flags=re.I,
    ):
        return "시장 평가가 단일 발표보다 실행 범위를 함께 보는 방향으로 옮겨갈 수 있음을 보여준다."
    if flags["actor_or_partner"] and re.search(
        r"협력|제휴|파트너|컨소시엄|그룹|기업|정부|기관|참여",
        value,
        flags=re.I,
    ):
        return (
            "개별 기업의 단독 움직임보다 참여 주체 간 역할과 협력 구조가 "
            "더 중요한 판단 축으로 부각될 수 있음을 보여준다."
        )
    if flags["financial"] and re.search(r"매출|비중|수익|거래|성과|지표", value, flags=re.I):
        return (
            "시장 평가가 규모 자체보다 성과와 거래 구조를 설명할 수 있는 "
            "기준으로 이동할 수 있음을 보여준다."
        )
    return ""


def _industry_action_result_phrase(criteria: Sequence[str]) -> str:
    normalized = {_anchor_norm(item) for item in criteria}
    result: list[str] = []
    if {_anchor_norm("참여 주체"), _anchor_norm("파트너십 필요성")} & normalized:
        result.append("협력 필요성")
    if {_anchor_norm("고객 적용 가능성"), _anchor_norm("고객 적용 방식")} & normalized:
        result.append("시스템 접점")
    if {_anchor_norm("실행 조건"), _anchor_norm("기존 시스템 접점")} & normalized:
        result.append("운영·시스템 연계 조건")
    if {
        _anchor_norm("투자 조건"),
        _anchor_norm("비용 부담"),
        _anchor_norm("내부 관리 지표"),
    } & normalized:
        result.append("내부 관리 기준")
    if {_anchor_norm("기술 공급 구조"), _anchor_norm("데이터/인프라 준비 수준")} & normalized:
        result.append("기술·인프라 준비 수준")
    if {_anchor_norm("후속 사업화 조건"), _anchor_norm("규제/정책 대응 조건")} & normalized:
        result.append("후속 확인 조건")
    return "·".join(_dedupe_keep_order(result)[:3])
