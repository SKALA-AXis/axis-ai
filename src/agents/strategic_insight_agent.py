# 작성일: 2026-06-02
# 작성자: 심유정
# 변경이력:
#   2026-06-02 심유정 — 전략 인사이트 에이전트 신규 구현 및 카드뉴스 grounding·한국어 출력 개선
#   2026-06-02 박지원 — 분석 파이프라인 에이전트 재구성·뉴스 통합, 카드뉴스 시사점 grounding
#   2026-06-11 최종민 — lazy-import 적용, RAG 선례 검색 주입
"""StrategicInsightAgent — analysis + implication in one LLM call.

기존 ``StrategicAnalyzer`` 와 ``ImplicationAgent`` 를 하나의 LLM agent 로 통합하되,
저장/후속 처리 호환성을 위해 출력은 ``analysis`` 와 ``implication`` 두 블록으로
분리한다.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from src.agents.implication_agent import ImplicationAgent
from src.agents.strategic_analyzer import StrategicAnalyzer
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
from src.agents.strategic_insight.diagnostics import (  # noqa: F401
    _attach_frontend_ready_diagnostics,
    _attach_generation_phase_diagnostics,
    _attach_strategy_skip_diagnostics,
    _build_analysis_context_for_issue,
    _can_attempt_frontend_ready_repair,
    _cluster_metadata_from_bundle,
    _domain_supported_by_evidence,
    _frontend_ready_diagnostics_snapshot,
    _mark_frontend_ready_source,
    _merge_frontend_ready_payload,
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

# 분리 모듈 re-export — 기존 참조 호환 유지 (agent-split-design.md 1단계)
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
from src.agents.strategic_insight.quality_gate import (  # noqa: F401
    _action_artifact_plan_violation,
    _attach_sentence_grounding,
    _build_sentence_grounding,
    _business_novelty_overclaim_violation,
    _can_attempt_frontend_ready_repair_for_issue,
    _clear_quality_gate_failed_marker,
    _critical_ungrounded_paths,
    _ensure_safe_recommended_actions,
    _flatten_profile_grounding_entries,
    _frontend_ready_action_choice_violation,
    _frontend_ready_action_depth_violation,
    _frontend_ready_action_mechanical_split_violation,
    _frontend_ready_action_specificity_violation,
    _frontend_ready_action_weak_review_phrase_violation,
    _frontend_ready_evidence_role_separation_violation,
    _frontend_ready_only_violations,
    _frontend_ready_repair_already_attempted,
    _frontend_ready_role_separation_violation,
    _frontend_ready_specific_anchor_violations,
    _grounding_entries_for_text,
    _grounding_target_texts,
    _grounding_type,
    _profile_grounding_entries,
    _quality_gate_violations,
    _repair_action_violation,
    _requires_self_review_for_violations,
    _restore_valid_flags_if_structurally_safe,
    _skax_external_customer_facing_violation,
    _two_section_repetition_violation,
    _unsupported_domain_term_violation,
    _unsupported_peer_profile_claim_violation,
    _unsupported_skax_profile_term_violation,
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
from src.analysis.models import AnalysisContext, AnalysisInputBundle, ProfileContext
from src.llm import LLMSpec, build_chat_llm

log = logging.getLogger(__name__)


# 근거 없이 쓰면 사실 왜곡이 큰 고위험 주장만 최소 차단한다.
# 표현 품질은 아래 구조 게이트와 프롬프트가 담당하고, 문구 blacklist 를 늘리지 않는다.


class StrategicInsightAgent:
    """Generate separated analysis/implication blocks from one LLM prompt."""

    prompt_version = _PROMPT_VERSION

    def __init__(
        self,
        *,
        llm: ChatOpenAI | None = None,
        analyzer: StrategicAnalyzer | None = None,
        implication_agent: ImplicationAgent | None = None,
        fallback_analyzer: StrategicAnalyzer | None = None,
        fallback_implication_agent: ImplicationAgent | None = None,
        enable_self_review: bool = True,
    ) -> None:
        self._llm = llm
        self._fallback_analyzer = fallback_analyzer or analyzer or StrategicAnalyzer()
        self._fallback_implication_agent = (
            fallback_implication_agent or implication_agent or ImplicationAgent()
        )
        self.enable_self_review = enable_self_review
        self.model = _LLM_MODEL
        self._llm_cache: dict[str, ChatOpenAI] = {}

    def generate(
        self,
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any] | None = None,
        input_bundle: AnalysisInputBundle | dict[str, Any] | None = None,
        profile_context: ProfileContext | dict[str, Any] | None = None,
        analysis_context: AnalysisContext | dict[str, Any] | None = None,
        cluster_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return ``{"analysis": ..., "implication": ...}`` for downstream pipeline."""
        classification = classification or {}
        bundle_dict = _bundle_to_dict(input_bundle)
        profile_dict = _profile_to_dict(profile_context)
        context_dict = _analysis_context_to_dict(analysis_context)
        cluster_metadata = cluster_metadata or _cluster_metadata_from_bundle(bundle_dict)

        early_skip_decision = _strategic_generation_skip_decision(
            integrated_issue=integrated_issue,
            classification=classification,
        )
        if early_skip_decision and not _is_valid_integrated_issue(integrated_issue):
            skipped = _empty_strategic_insight(
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata=cluster_metadata,
                reason=str(early_skip_decision.get("reason") or ""),
            )
            skipped = _attach_strategy_skip_diagnostics(
                skipped,
                skip_decision=early_skip_decision,
                integrated_issue=integrated_issue,
                profile_linkage_evaluation={},
                action_artifact_plan={},
            )
            return _attach_generation_phase_diagnostics(
                skipped,
                decisions=[
                    str(early_skip_decision.get("decision_type") or "watch_only_precheck"),
                    "llm_skipped",
                    "invalid_summary_preserved_as_watch_only_signal",
                ],
            )
        if not _is_valid_integrated_issue(integrated_issue):
            return _empty_strategic_insight(
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata=cluster_metadata,
                reason="유효한 통합 이슈가 없어 전략 인사이트를 생성하지 않았습니다.",
            )
        profile_relevance_text = _profile_relevance_hint_text(
            integrated_issue=integrated_issue,
            classification=classification,
            bundle=bundle_dict,
        )
        include_financial_profile_context = _should_include_financial_profile_context(
            integrated_issue=integrated_issue,
            classification=classification,
            bundle=bundle_dict,
        )
        profile_linkage_evaluation = _build_profile_linkage_evaluation(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_context=profile_dict,
            relevance_hint_text=profile_relevance_text,
        )
        action_artifact_plan = _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        skip_decision = early_skip_decision
        if skip_decision:
            skipped = _empty_strategic_insight(
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata=cluster_metadata,
                reason=str(skip_decision.get("reason") or ""),
            )
            skipped = _attach_strategy_skip_diagnostics(
                skipped,
                skip_decision=skip_decision,
                integrated_issue=integrated_issue,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            return _attach_generation_phase_diagnostics(
                skipped,
                decisions=[
                    str(skip_decision.get("decision_type") or "watch_only_precheck"),
                    "llm_skipped",
                ],
            )
        context_for_model = _analysis_context_for_model(
            context_dict,
            integrated_issue=integrated_issue,
            relevance_hint_text=profile_relevance_text,
            include_financial_context=include_financial_profile_context,
        )

        prompt = USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            strategic_evidence_json=_json_dumps(
                _strategic_evidence_pack_for_prompt(
                    integrated_issue=integrated_issue,
                    bundle=bundle_dict,
                )
            ),
            classification_json=_json_dumps(_classification_for_prompt(classification)),
            bundle_json=_json_dumps(_bundle_for_prompt(bundle_dict, cluster_metadata)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_dict,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            context_json=_json_dumps(_analysis_context_for_prompt(context_for_model)),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
            context_availability_json=_json_dumps(
                _context_availability_for_prompt(
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                    analysis_context=context_for_model,
                )
            ),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_dict,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
            role_mode_instructions=_role_mode_instructions(integrated_issue),
            prompt_version=_PROMPT_VERSION,
            model=self.model,
        )

        try:
            content = self._invoke_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                bundle_id=str(
                    bundle_dict.get("bundle_id")
                    or integrated_issue.get("bundle_id")
                    or cluster_metadata.get("bundle_id")
                    or ""
                ),
                phase="generate",
            )
            result = _parse_and_normalize(
                content,
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata=cluster_metadata,
                profile_context=profile_dict,
                analysis_context=context_for_model,
                model=self.model,
            )
            result = _with_fallback_linkage_payloads(
                result,
                profile_linkage_evaluation=profile_linkage_evaluation,
                integrated_issue=integrated_issue,
                action_artifact_plan=action_artifact_plan,
            )
            result = _polish_frontend_ready_screen_copy(
                result,
                integrated_issue=integrated_issue,
                profile_linkage_evaluation=profile_linkage_evaluation,
            )
            if not self.enable_self_review:
                return self._finalize_quality_gate(
                    result,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
            initial_violations = _quality_gate_violations(
                result,
                integrated_issue=integrated_issue,
                profile_context=profile_dict,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            if _has_displayable_frontend_ready(
                result,
                integrated_issue=integrated_issue,
                profile_context=profile_dict,
                profile_linkage_evaluation=profile_linkage_evaluation,
            ):
                non_frontend = [
                    item
                    for item in initial_violations
                    if not str(item or "").startswith("frontend_ready")
                ]
                result_for_display = (
                    _mark_quality_gate_failed(
                        result,
                        non_frontend,
                        preserve_frontend_ready=True,
                    )
                    if non_frontend
                    else _restore_valid_flags_if_structurally_safe(result)
                )
                return _attach_generation_phase_diagnostics(
                    _attach_sentence_grounding(
                        result_for_display,
                        integrated_issue=integrated_issue,
                        profile_context=profile_dict,
                    ),
                    decisions=[
                        "generate_result_displayable",
                        "self_review_skipped",
                        "schema_repair_skipped",
                    ],
                )
            if not initial_violations:
                return _attach_generation_phase_diagnostics(
                    _attach_sentence_grounding(
                        _restore_valid_flags_if_structurally_safe(result),
                        integrated_issue=integrated_issue,
                        profile_context=profile_dict,
                    ),
                    decisions=[
                        "generate_result_clean",
                        "self_review_skipped",
                        "schema_repair_skipped",
                    ],
                )
            if _frontend_ready_only_violations(initial_violations):
                repaired = self._repair_quality_violations(
                    result,
                    violations=initial_violations,
                    integrated_issue=integrated_issue,
                    classification=classification,
                    profile_context=profile_dict,
                    analysis_context=context_for_model,
                    bundle_id=str(
                        bundle_dict.get("bundle_id")
                        or integrated_issue.get("bundle_id")
                        or cluster_metadata.get("bundle_id")
                        or ""
                    ),
                    profile_relevance_text=profile_relevance_text,
                    include_financial_profile_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                return _attach_generation_phase_diagnostics(
                    _attach_sentence_grounding(
                        repaired,
                        integrated_issue=integrated_issue,
                        profile_context=profile_dict,
                    ),
                    decisions=[
                        "generate_result_frontend_only_violation",
                        "self_review_skipped",
                        "frontend_ready_repair_attempted",
                    ],
                )
            if not _requires_self_review_for_violations(
                initial_violations,
                result=result,
            ):
                repaired = self._repair_quality_violations(
                    result,
                    violations=initial_violations,
                    integrated_issue=integrated_issue,
                    classification=classification,
                    profile_context=profile_dict,
                    analysis_context=context_for_model,
                    bundle_id=str(
                        bundle_dict.get("bundle_id")
                        or integrated_issue.get("bundle_id")
                        or cluster_metadata.get("bundle_id")
                        or ""
                    ),
                    profile_relevance_text=profile_relevance_text,
                    include_financial_profile_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                return _attach_generation_phase_diagnostics(
                    _attach_sentence_grounding(
                        repaired,
                        integrated_issue=integrated_issue,
                        profile_context=profile_dict,
                    ),
                    decisions=[
                        "generate_result_schema_or_copy_violation",
                        "self_review_skipped",
                        "quality_repair_attempted",
                    ],
                )
            reviewed = self._review_and_revise(
                result,
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_dict,
                analysis_context=context_for_model,
                bundle_id=str(
                    bundle_dict.get("bundle_id")
                    or integrated_issue.get("bundle_id")
                    or cluster_metadata.get("bundle_id")
                    or ""
                ),
                profile_relevance_text=profile_relevance_text,
                include_financial_profile_context=include_financial_profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            reviewed = _with_fallback_linkage_payloads(
                reviewed,
                profile_linkage_evaluation=profile_linkage_evaluation,
                integrated_issue=integrated_issue,
                action_artifact_plan=action_artifact_plan,
            )
            if "quality_gate_failed" in _json_dumps(reviewed):
                if not _has_displayable_frontend_ready(
                    reviewed,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                ):
                    if _frontend_ready_repair_already_attempted(reviewed):
                        return _attach_sentence_grounding(
                            reviewed,
                            integrated_issue=integrated_issue,
                            profile_context=profile_dict,
                        )
                    return self._finalize_quality_gate(
                        reviewed,
                        integrated_issue=integrated_issue,
                        profile_context=profile_dict,
                        profile_linkage_evaluation=profile_linkage_evaluation,
                        action_artifact_plan=action_artifact_plan,
                    )
                return _attach_sentence_grounding(
                    reviewed,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                )
            return self._finalize_quality_gate(
                reviewed,
                integrated_issue=integrated_issue,
                profile_context=profile_dict,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
        except Exception as exc:  # noqa: BLE001 - fallback preserves pipeline availability.
            log.warning(
                "StrategicInsightAgent LLM failure → legacy fallback | bundle=%s error=%s",
                bundle_dict.get("bundle_id") or integrated_issue.get("bundle_id"),
                exc,
            )
            return self._legacy_fallback(
                integrated_issue=integrated_issue,
                classification=classification,
                input_bundle=input_bundle,
                profile_context=profile_context,
                analysis_context=(
                    analysis_context if isinstance(analysis_context, AnalysisContext) else None
                ),
                cluster_metadata=cluster_metadata,
            )

    def generate_from_analysis_package(
        self,
        analysis_package: dict[str, Any],
        *,
        profile_context: ProfileContext | dict[str, Any] | None = None,
        analysis_context: AnalysisContext | dict[str, Any] | None = None,
        user_id: str | None = None,
        strict_profile: bool = False,
        require_skax_profile: bool = False,
    ) -> dict[str, Any]:
        """Generate strategic insight from a stored analysis_package JSON.

        This is the bridge for persisted pipeline output. The package usually
        comes from ``card_news.evidence_payload.analysis_package`` and contains
        the IntegrationAgent output plus classification/source metadata.
        """
        package = _json_dict(analysis_package)
        integrated_issue = _json_dict(package.get("integrated_issue") or package.get("summary"))
        classification = _json_dict(package.get("classification"))
        input_bundle = _input_bundle_from_analysis_package(
            package=package,
            integrated_issue=integrated_issue,
            classification=classification,
        )
        profile_context = profile_context or _load_profile_context_for_issue(
            integrated_issue=integrated_issue,
            classification=classification,
            package=package,
            user_id=user_id,
            strict=strict_profile,
            require_skax_profile=require_skax_profile,
        )
        analysis_context = analysis_context or _build_analysis_context_for_issue(
            input_bundle=input_bundle,
            profile_context=profile_context,
            integrated_issue=integrated_issue,
        )
        return self.generate(
            integrated_issue=integrated_issue,
            classification=classification,
            input_bundle=input_bundle,
            profile_context=profile_context,
            analysis_context=analysis_context,
            cluster_metadata=_cluster_metadata_from_bundle(input_bundle.to_dict()),
        )

    def generate_from_integrated_issue_id(
        self,
        integrated_issue_id: str,
        *,
        save: bool = False,
        user_id: str | None = None,
        strict_profile: bool = False,
        require_skax_profile: bool = False,
    ) -> dict[str, Any]:
        """Load ``integrated_issues`` by id and run the agent.

        This is the forward pipeline entry point. Card news is created after
        strategic insight generation, so callers that already have an
        IntegratedIssue should use this method instead of a card id.
        """
        if save:
            raise ValueError(
                "StrategicInsightAgent persistence is not enabled yet; "
                "run with save=False until the storage table is finalized."
            )
        record = _load_integrated_issue_analysis_package(integrated_issue_id)
        result = self.generate_from_analysis_package(
            record["analysis_package"],
            user_id=user_id,
            strict_profile=strict_profile,
            require_skax_profile=require_skax_profile,
        )
        return result

    def generate_from_card_news(
        self,
        card_news_id: str,
        *,
        save: bool = False,
        user_id: str | None = None,
        strict_profile: bool = False,
        require_skax_profile: bool = False,
    ) -> dict[str, Any]:
        """Load ``card_news`` by id and run the agent for existing-card debug flows."""
        if save:
            raise ValueError(
                "StrategicInsightAgent persistence is not enabled yet; "
                "run with save=False until the storage table is finalized."
            )
        record = _load_card_news_analysis_package(card_news_id)
        result = self.generate_from_analysis_package(
            record["analysis_package"],
            user_id=user_id,
            strict_profile=strict_profile,
            require_skax_profile=require_skax_profile,
        )
        return result

    def _get_llm(self, *, model: str | None = None) -> ChatOpenAI:
        if self._llm is not None:
            return self._llm
        model_name = str(model or _LLM_MODEL or _DEFAULT_LLM_MODEL).strip() or _DEFAULT_LLM_MODEL
        if model_name not in self._llm_cache:
            # 기존 동작 보존: json_object 미사용, gpt-5 라도 reasoning_effort
            # 미전달(reasoning_effort=None), timeout·max_retries 유지.
            self._llm_cache[model_name] = build_chat_llm(
                LLMSpec(
                    model=model_name,
                    temperature=_LLM_TEMPERATURE,
                    max_tokens=_LLM_MAX_COMPLETION_TOKENS,
                    reasoning_effort=None,
                    timeout=_LLM_REQUEST_TIMEOUT_SECONDS,
                    max_retries=1,
                )
            )
        return self._llm_cache[model_name]

    def _invoke_llm(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        bundle_id: str,
        phase: str = "generate",
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        model_name = _llm_model_for_phase(phase)
        try:
            from src.observability import tracing_config

            config = tracing_config(
                agent="StrategicInsightAgent",
                phase=phase,
                prompt_version=_PROMPT_VERSION,
                bundle_id=bundle_id,
                model=model_name,
            )
        except Exception:
            config = None
        response = (
            self._get_llm(model=model_name).invoke(messages, config=config)
            if config
            else self._get_llm(model=model_name).invoke(messages)
        )
        return response.content if isinstance(response.content, str) else str(response.content)

    def _finalize_quality_gate(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        profile_context: dict[str, Any],
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification={},
                profile_context=profile_context,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification={},
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        result = _with_fallback_linkage_payloads(
            result,
            profile_linkage_evaluation=profile_linkage_evaluation,
            integrated_issue=integrated_issue,
            action_artifact_plan=action_artifact_plan,
        )
        violations = _quality_gate_violations(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
            action_artifact_plan=action_artifact_plan,
        )
        if not violations:
            return _attach_sentence_grounding(
                _restore_valid_flags_if_structurally_safe(result),
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
        if _has_displayable_frontend_ready(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
        ):
            polish_violations = _frontend_ready_claim_violations(
                result,
                integrated_issue=integrated_issue,
                profile_linkage_evaluation=profile_linkage_evaluation,
            )
            if polish_violations and _can_attempt_frontend_ready_repair_for_issue(
                result,
                integrated_issue=integrated_issue,
            ):
                try:
                    repaired = self._repair_frontend_ready_result(
                        result,
                        violations=violations + polish_violations,
                        integrated_issue=integrated_issue,
                        classification={},
                        profile_context=profile_context,
                        analysis_context={},
                        bundle_id=str(integrated_issue.get("bundle_id") or ""),
                        profile_linkage_evaluation=profile_linkage_evaluation,
                        action_artifact_plan=action_artifact_plan,
                    )
                    if _has_displayable_frontend_ready(
                        repaired,
                        integrated_issue=integrated_issue,
                        profile_context=profile_context,
                        profile_linkage_evaluation=profile_linkage_evaluation,
                    ):
                        return _attach_sentence_grounding(
                            _mark_quality_gate_failed(
                                repaired,
                                violations,
                                preserve_frontend_ready=True,
                            ),
                            integrated_issue=integrated_issue,
                            profile_context=profile_context,
                        )
                except Exception as exc:  # noqa: BLE001 - display polish is best-effort.
                    log.warning(
                        "StrategicInsightAgent frontend_ready polish skipped | error=%s",
                        exc,
                    )
            return _attach_sentence_grounding(
                _mark_quality_gate_failed(result, violations, preserve_frontend_ready=True),
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
        frontend_ready_before = _frontend_ready_diagnostics_snapshot(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if _can_attempt_frontend_ready_repair_for_issue(
            result,
            integrated_issue=integrated_issue,
        ):
            try:
                repaired = self._repair_frontend_ready_result(
                    result,
                    violations=violations,
                    integrated_issue=integrated_issue,
                    classification={},
                    profile_context=profile_context,
                    analysis_context={},
                    bundle_id=str(integrated_issue.get("bundle_id") or ""),
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                frontend_ready_after = _frontend_ready_diagnostics_snapshot(
                    repaired,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
                repaired = _attach_frontend_ready_diagnostics(
                    repaired,
                    before=frontend_ready_before,
                    after=frontend_ready_after,
                )
                remaining = _quality_gate_violations(
                    repaired,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                if frontend_ready_after.get("displayable"):
                    non_frontend = [
                        item
                        for item in remaining
                        if not str(item or "").startswith("frontend_ready")
                    ]
                    return _attach_sentence_grounding(
                        _mark_quality_gate_failed(
                            repaired,
                            non_frontend or violations,
                            preserve_frontend_ready=True,
                        )
                        if non_frontend
                        else _restore_valid_flags_if_structurally_safe(repaired),
                        integrated_issue=integrated_issue,
                        profile_context=profile_context,
                    )
            except Exception as exc:  # noqa: BLE001 - frontend repair is best-effort.
                log.warning(
                    "StrategicInsightAgent frontend_ready finalize repair failed | error=%s",
                    exc,
                )
        failed = _attach_frontend_ready_diagnostics(
            result,
            before=frontend_ready_before,
            after=_frontend_ready_diagnostics_snapshot(
                result,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
            ),
            removed_reason="frontend_ready 전용 repair가 화면 표시 조건을 충족하지 못했습니다.",
        )
        remaining = _quality_gate_violations(
            failed,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
            action_artifact_plan=action_artifact_plan,
        )
        return _attach_sentence_grounding(
            _mark_quality_gate_failed(failed, remaining or violations),
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )

    def _review_and_revise(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                relevance_hint_text=profile_relevance_text,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        prompt = REVIEW_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            classification_json=_json_dumps(_classification_for_prompt(classification)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            context_json=_json_dumps(_analysis_context_for_prompt(analysis_context)),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
            context_availability_json=_json_dumps(
                _context_availability_for_prompt(
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    analysis_context=analysis_context,
                )
            ),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
            result_json=_json_dumps(result),
        )
        try:
            content = self._invoke_llm(
                system_prompt=REVIEW_SYSTEM_PROMPT,
                user_prompt=prompt,
                bundle_id=bundle_id,
                phase="self_review",
            )
            revised = _parse_review_and_normalize(
                content,
                original=result,
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                analysis_context=analysis_context,
                model=self.model,
            )
            violations = _quality_gate_violations(
                revised,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            if not violations:
                return revised
            return self._repair_quality_violations(
                revised,
                violations=violations,
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                analysis_context=analysis_context,
                bundle_id=bundle_id,
                profile_relevance_text=profile_relevance_text,
                include_financial_profile_context=include_financial_profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
        except Exception as exc:  # noqa: BLE001 - review is quality layer, not availability gate.
            log.warning(
                "StrategicInsightAgent self-review skipped | bundle=%s error=%s",
                bundle_id,
                exc,
            )
            violations = _quality_gate_violations(
                result,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            if violations:
                return self._repair_quality_violations(
                    result,
                    violations=violations,
                    integrated_issue=integrated_issue,
                    classification=classification,
                    profile_context=profile_context,
                    analysis_context=analysis_context,
                    bundle_id=bundle_id,
                    profile_relevance_text=profile_relevance_text,
                    include_financial_profile_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
            return result

    def _repair_quality_violations(
        self,
        result: dict[str, Any],
        *,
        violations: list[str],
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = result
        current_violations = violations
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                relevance_hint_text=profile_relevance_text,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        frontend_ready_before = _frontend_ready_diagnostics_snapshot(
            current,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if frontend_ready_before.get("displayable"):
            polish_violations = _frontend_ready_claim_violations(
                current,
                integrated_issue=integrated_issue,
                profile_linkage_evaluation=profile_linkage_evaluation,
            )
            if polish_violations and _can_attempt_frontend_ready_repair_for_issue(
                current,
                integrated_issue=integrated_issue,
            ):
                current = self._repair_frontend_ready_result(
                    current,
                    violations=current_violations + polish_violations,
                    integrated_issue=integrated_issue,
                    classification=classification,
                    profile_context=profile_context,
                    analysis_context=analysis_context,
                    bundle_id=bundle_id,
                    profile_relevance_text=profile_relevance_text,
                    include_financial_profile_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                frontend_ready_after = _frontend_ready_diagnostics_snapshot(
                    current,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
                if frontend_ready_after.get("displayable"):
                    current = _attach_frontend_ready_diagnostics(
                        current,
                        before=frontend_ready_before,
                        after=frontend_ready_after,
                    )
                    non_frontend = [
                        violation
                        for violation in current_violations
                        if not str(violation or "").startswith("frontend_ready")
                    ]
                    if non_frontend:
                        return _mark_quality_gate_failed(
                            current,
                            non_frontend,
                            preserve_frontend_ready=True,
                        )
                    return _restore_valid_flags_if_structurally_safe(current)
            current = _attach_frontend_ready_diagnostics(
                current,
                before=frontend_ready_before,
                after=frontend_ready_before,
            )
            non_frontend = [
                violation
                for violation in current_violations
                if not str(violation or "").startswith("frontend_ready")
            ]
            if non_frontend:
                return _mark_quality_gate_failed(
                    current,
                    non_frontend,
                    preserve_frontend_ready=True,
                )
            return current
        if (
            not frontend_ready_before.get("displayable")
            and _frontend_ready_only_violations(current_violations)
            and _can_attempt_frontend_ready_repair_for_issue(
                current,
                integrated_issue=integrated_issue,
            )
        ):
            current = self._repair_frontend_ready_result(
                current,
                violations=current_violations,
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                analysis_context=analysis_context,
                bundle_id=bundle_id,
                profile_relevance_text=profile_relevance_text,
                include_financial_profile_context=include_financial_profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            frontend_ready_after = _frontend_ready_diagnostics_snapshot(
                current,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
            )
            current = _attach_frontend_ready_diagnostics(
                current,
                before=frontend_ready_before,
                after=frontend_ready_after,
            )
            current_violations = _quality_gate_violations(
                current,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            if frontend_ready_after.get("displayable"):
                remaining = [
                    violation
                    for violation in current_violations
                    if not str(violation or "").startswith("frontend_ready")
                ]
                if remaining:
                    return _mark_quality_gate_failed(
                        current,
                        remaining,
                        preserve_frontend_ready=True,
                    )
                return _restore_valid_flags_if_structurally_safe(current)
        elif not _can_attempt_frontend_ready_repair_for_issue(
            current,
            integrated_issue=integrated_issue,
        ):
            current = _attach_frontend_ready_diagnostics(
                current,
                before=frontend_ready_before,
                after=frontend_ready_before,
                removed_reason="IntegratedIssue에 카드뉴스용 문장을 만들 사실 근거가 부족합니다.",
            )
            return _mark_quality_gate_failed(current, current_violations)
        try:
            for attempt in range(2):
                prompt = REPAIR_USER_PROMPT_TEMPLATE.format(
                    integrated_issue_json=_json_dumps(
                        _integrated_issue_for_prompt(integrated_issue)
                    ),
                    profile_json=_json_dumps(
                        _profile_for_prompt(
                            profile_context,
                            integrated_issue=integrated_issue,
                            relevance_hint_text=profile_relevance_text,
                            include_financial_context=include_financial_profile_context,
                            profile_linkage_evaluation=profile_linkage_evaluation,
                        )
                    ),
                    profile_linkage_json=_json_dumps(profile_linkage_evaluation),
                    action_artifact_plan_json=_json_dumps(action_artifact_plan),
                    business_lines_json=_json_dumps(
                        _business_line_candidate_details(
                            profile_context,
                            integrated_issue=integrated_issue,
                            relevance_hint_text=profile_relevance_text,
                        )
                    ),
                    violations_json=_json_dumps(current_violations),
                    result_json=_json_dumps(current),
                )
                content = self._invoke_llm(
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    bundle_id=bundle_id,
                    phase=f"quality_repair_{attempt + 1}",
                )
                current = _parse_and_normalize(
                    content,
                    integrated_issue=integrated_issue,
                    classification=classification,
                    cluster_metadata={},
                    profile_context=profile_context,
                    analysis_context=analysis_context,
                    model=self.model,
                )
                current = _mark_frontend_ready_source(current, "schema_repair_direct")
                current_violations = _quality_gate_violations(
                    current,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                if not current_violations:
                    return current
                if _frontend_ready_only_violations(current_violations):
                    break
            if _main_company_is_customer_or_buyer(integrated_issue):
                current = self._repair_counterparty_role_result(
                    current,
                    integrated_issue=integrated_issue,
                    classification=classification,
                    profile_context=profile_context,
                    analysis_context=analysis_context,
                    bundle_id=bundle_id,
                    profile_relevance_text=profile_relevance_text,
                    include_financial_profile_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                current_violations = _quality_gate_violations(
                    current,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                if not current_violations:
                    return current
            if _has_displayable_frontend_ready(
                current,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
            ):
                return _mark_quality_gate_failed(
                    current,
                    current_violations,
                    preserve_frontend_ready=True,
                )
            if not _can_attempt_frontend_ready_repair_for_issue(
                current,
                integrated_issue=integrated_issue,
            ):
                return _mark_quality_gate_failed(current, current_violations)
            current = self._repair_frontend_ready_result(
                current,
                violations=current_violations,
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                analysis_context=analysis_context,
                bundle_id=bundle_id,
                profile_relevance_text=profile_relevance_text,
                include_financial_profile_context=include_financial_profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            frontend_ready_after = _frontend_ready_diagnostics_snapshot(
                current,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
            )
            current = _attach_frontend_ready_diagnostics(
                current,
                before=frontend_ready_before,
                after=frontend_ready_after,
            )
            current_violations = _quality_gate_violations(
                current,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            if not current_violations:
                return current
            if _has_displayable_frontend_ready(
                current,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
            ):
                return _mark_quality_gate_failed(
                    current,
                    current_violations,
                    preserve_frontend_ready=True,
                )
            return _mark_quality_gate_failed(current, current_violations)
        except Exception as exc:  # noqa: BLE001 - fail closed instead of passing risky copy.
            log.warning(
                "StrategicInsightAgent quality repair failed | bundle=%s error=%s",
                bundle_id,
                exc,
            )
            failed = _attach_frontend_ready_diagnostics(
                result,
                before=frontend_ready_before,
                after=_frontend_ready_diagnostics_snapshot(
                    result,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                ),
                removed_reason=str(exc),
            )
            final_remaining = _quality_gate_violations(
                failed,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            if _can_attempt_frontend_ready_repair_for_issue(
                failed,
                integrated_issue=integrated_issue,
            ):
                try:
                    repaired = self._repair_frontend_ready_result(
                        failed,
                        violations=final_remaining or current_violations,
                        integrated_issue=integrated_issue,
                        classification=classification,
                        profile_context=profile_context,
                        analysis_context=analysis_context,
                        bundle_id=bundle_id,
                        profile_relevance_text=profile_relevance_text,
                        include_financial_profile_context=include_financial_profile_context,
                        profile_linkage_evaluation=profile_linkage_evaluation,
                        action_artifact_plan=action_artifact_plan,
                    )
                    frontend_ready_after = _frontend_ready_diagnostics_snapshot(
                        repaired,
                        integrated_issue=integrated_issue,
                        profile_context=profile_context,
                        profile_linkage_evaluation=profile_linkage_evaluation,
                    )
                    repaired = _attach_frontend_ready_diagnostics(
                        repaired,
                        before=frontend_ready_before,
                        after=frontend_ready_after,
                        removed_reason=str(exc),
                    )
                    repaired_remaining = _quality_gate_violations(
                        repaired,
                        integrated_issue=integrated_issue,
                        profile_context=profile_context,
                        profile_linkage_evaluation=profile_linkage_evaluation,
                        action_artifact_plan=action_artifact_plan,
                    )
                    if frontend_ready_after.get("displayable"):
                        non_frontend = [
                            item
                            for item in repaired_remaining
                            if not str(item or "").startswith("frontend_ready")
                        ]
                        return (
                            _mark_quality_gate_failed(
                                repaired,
                                non_frontend,
                                preserve_frontend_ready=True,
                            )
                            if non_frontend
                            else _restore_valid_flags_if_structurally_safe(repaired)
                        )
                except Exception as frontend_exc:  # noqa: BLE001 - keep fail-closed fallback.
                    log.warning(
                        "StrategicInsightAgent frontend_ready repair after quality failure "
                        "failed | bundle=%s error=%s",
                        bundle_id,
                        frontend_exc,
                    )
            if final_remaining:
                return _mark_quality_gate_failed(failed, final_remaining)
            return failed

    def _repair_counterparty_role_result(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                relevance_hint_text=profile_relevance_text,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        prompt = COUNTERPARTY_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
            result_json=_json_dumps(result),
        )
        content = self._invoke_llm(
            system_prompt=COUNTERPARTY_REPAIR_SYSTEM_PROMPT,
            user_prompt=prompt,
            bundle_id=bundle_id,
            phase="quality_repair_counterparty_role",
        )
        return _mark_frontend_ready_source(
            _parse_and_normalize(
                content,
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata={},
                profile_context=profile_context,
                analysis_context=analysis_context,
                model=self.model,
            ),
            "counterparty_repair_direct",
        )

    def _repair_report_copy_result(
        self,
        result: dict[str, Any],
        *,
        violations: list[str],
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                relevance_hint_text=profile_relevance_text,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        prompt = REPORT_COPY_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
            result_json=_json_dumps(result),
            violations_json=_json_dumps(violations),
        )
        content = self._invoke_llm(
            system_prompt=REPORT_COPY_REPAIR_SYSTEM_PROMPT,
            user_prompt=prompt,
            bundle_id=bundle_id,
            phase="quality_repair_report_copy",
        )
        return _mark_frontend_ready_source(
            _parse_and_normalize(
                content,
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata={},
                profile_context=profile_context,
                analysis_context=analysis_context,
                model=self.model,
            ),
            "report_copy_repair_direct",
        )

    def _repair_frontend_ready_result(
        self,
        result: dict[str, Any],
        *,
        violations: list[str],
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del analysis_context
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                relevance_hint_text=profile_relevance_text,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        base_prompt = FRONTEND_READY_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
            issue_execution_slots_json=_json_dumps(
                _issue_execution_slot_diagnostics(integrated_issue)
            ),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
            result_json=_json_dumps(result),
            violations_json=_json_dumps(violations),
        )
        repaired = result
        retry_notes: list[str] = []
        for attempt in range(2):
            prompt = base_prompt
            if retry_notes:
                prompt = (
                    base_prompt
                    + "\n\n## 이전 frontend_ready repair 실패\n"
                    + "\n".join(retry_notes)
                    + "\nfrontend_ready JSON 객체만 다시 출력하세요."
                )
            content = self._invoke_llm(
                system_prompt=FRONTEND_READY_REPAIR_SYSTEM_PROMPT,
                user_prompt=prompt,
                bundle_id=bundle_id,
                phase=f"frontend_ready_repair_{attempt + 1}",
            )
            data = _parse_json_loose(content)
            frontend_ready = _json_dict(
                data.get("frontend_ready") if isinstance(data, dict) else {}
            )
            if (
                not frontend_ready
                and isinstance(data, dict)
                and ("key_implication" in data or "suggested_action" in data)
            ):
                frontend_ready = _json_dict(data)
            if frontend_ready:
                repaired = _merge_frontend_ready_payload(
                    result,
                    frontend_ready=frontend_ready,
                    source="frontend_repair_direct",
                    integrated_issue=integrated_issue,
                )
                repaired = _polish_frontend_ready_screen_copy(
                    repaired,
                    integrated_issue=integrated_issue,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
                normalized_frontend = _normalize_frontend_ready(
                    frontend_ready,
                    default_source="frontend_repair_direct",
                )
                anchor_violations = _frontend_ready_specific_anchor_violations(
                    repaired,
                    integrated_issue=integrated_issue,
                )
                required_violations = _frontend_ready_required_violations(
                    repaired,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
                claim_violations = _frontend_ready_claim_violations(
                    repaired,
                    integrated_issue=integrated_issue,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
                if (
                    normalized_frontend
                    and not required_violations
                    and not anchor_violations
                    and not claim_violations
                ):
                    return repaired
                retry_notes.extend(required_violations + anchor_violations + claim_violations)
            retry_notes.append(
                "응답에서 sentence/evidence_sentence를 포함한 key_implication 및 "
                "suggested_action 구조를 찾지 못했습니다."
            )
        compact_repaired = self._repair_frontend_ready_from_compact_issue_facts(
            result,
            violations=[*violations, *retry_notes],
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
            bundle_id=bundle_id,
        )
        if compact_repaired is not None:
            return compact_repaired
        return repaired

    def _repair_frontend_ready_from_compact_issue_facts(
        self,
        result: dict[str, Any],
        *,
        violations: list[str],
        integrated_issue: dict[str, Any],
        profile_context: dict[str, Any],
        profile_linkage_evaluation: dict[str, Any],
        bundle_id: str,
    ) -> dict[str, Any] | None:
        """Last frontend-only repair path using compact IntegratedIssue facts.

        This path does not generate template copy. It only gives the repair LLM
        a smaller, fact-focused input when the ordinary frontend_ready repair
        failed to return a displayable JSON block.
        """
        if _is_weak_surface_integrated_issue(integrated_issue):
            return None
        if not _has_integrated_issue_candidate_anchor_signal(integrated_issue):
            return None
        fact_lines = _issue_fact_lines(integrated_issue)[:12]
        if not fact_lines:
            return None
        prompt = "\n".join(
            [
                "## Task",
                "frontend_ready JSON 객체 하나만 반환합니다.",
                (
                    "기존 analysis/implication 문장을 복사하지 말고, "
                    "아래 IntegratedIssue fact만 사용합니다."
                ),
                "",
                "## IntegratedIssue core",
                _json_dumps(
                    {
                        "headline": integrated_issue.get("headline", ""),
                        "main_company": integrated_issue.get("main_company", ""),
                        "event_type": integrated_issue.get("event_type")
                        or integrated_issue.get("cluster_event_type"),
                        "fact_lines": fact_lines,
                        "specific_event_anchors": _specific_event_anchors_for_frontend(
                            integrated_issue
                        )[:10],
                    }
                ),
                "",
                "## Profile linkage",
                _json_dumps(profile_linkage_evaluation),
                "",
                "## Previous violations",
                _json_dumps(violations[:12]),
                "",
                "## Required JSON shape",
                _json_dumps(
                    {
                        "frontend_ready": {
                            "source": "frontend_repair_direct",
                            "key_implication": {
                                "source": "frontend_repair_direct",
                                "frame": "string",
                                "claim_type": "event_based_signal",
                                "claim_strength": "cautious",
                                "evidence_mode": "event_based",
                                "event_anchor_terms": ["현재 사건 fact에서 실제 사용한 표현"],
                                "profile_anchor_terms": [],
                                "unsupported_claims_removed": [],
                                "sentence": "짧은 핵심 시사점 1문장",
                                "evidence_sentence": "fact_lines의 구체 표현으로 뒷받침하는 1문장",
                            },
                            "suggested_action": {
                                "source": "frontend_repair_direct",
                                "frame": "string",
                                "claim_type": "internal_strategy_check",
                                "claim_strength": "cautious",
                                "evidence_mode": "generic_monitoring",
                                "event_anchor_terms": ["현재 사건 fact에서 실제 사용한 표현"],
                                "skax_anchor_terms": [],
                                "unsupported_claims_removed": [],
                                "sentence": "SK AX는 ... 기준을 나눠 볼 필요가 있다는 1문장",
                                "evidence_sentence": (
                                    "fact_lines의 구체 표현이 왜 그 판단 기준으로 "
                                    "이어지는지 설명하는 1문장"
                                ),
                            },
                        }
                    }
                ),
                "",
                "## Rules",
                "- source와 block.source는 모두 frontend_repair_direct로 씁니다.",
                "- profile linkage가 low/none이면 profile_based를 쓰지 않습니다.",
                "- SK AX 대응방안은 직접 구축/확보/진출/성과 입증을 단정하지 않습니다.",
                "- sentence에는 결론만 짧게 쓰고, 구체 fact는 evidence_sentence에 둡니다.",
                (
                    "- evidence_sentence에는 fact_lines에 있는 수치, 제품/서비스, 기능, "
                    "적용 대상, 평가 방식 중 하나 이상을 그대로 연결합니다."
                ),
                (
                    "- 피어 제품명/서비스명은 SK AX가 따라야 할 기준처럼 쓰지 말고, "
                    "근거 설명에만 둡니다."
                ),
                "- JSON 외 텍스트는 출력하지 않습니다.",
            ]
        )
        content = self._invoke_llm(
            system_prompt=FRONTEND_READY_REPAIR_SYSTEM_PROMPT,
            user_prompt=prompt,
            bundle_id=bundle_id,
            phase="frontend_ready_repair_compact_facts",
        )
        data = _parse_json_loose(content)
        frontend_ready = _json_dict(data.get("frontend_ready") if isinstance(data, dict) else {})
        if (
            not frontend_ready
            and isinstance(data, dict)
            and ("key_implication" in data or "suggested_action" in data)
        ):
            frontend_ready = _json_dict(data)
        if not frontend_ready:
            return None
        repaired = _merge_frontend_ready_payload(
            result,
            frontend_ready=frontend_ready,
            source="frontend_repair_direct",
            integrated_issue=integrated_issue,
        )
        repaired = _polish_frontend_ready_screen_copy(
            repaired,
            integrated_issue=integrated_issue,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        required_violations = _frontend_ready_required_violations(
            repaired,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        anchor_violations = _frontend_ready_specific_anchor_violations(
            repaired,
            integrated_issue=integrated_issue,
        )
        claim_violations = _frontend_ready_claim_violations(
            repaired,
            integrated_issue=integrated_issue,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if required_violations or anchor_violations or claim_violations:
            return None
        return repaired

    def _repair_missing_recommended_actions(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        profile_context: dict[str, Any],
        bundle_id: str,
        profile_relevance_text: str = "",
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        implication = out.get("implication") or {}
        skax = implication.get("skax_implication") or {}
        profile_linkage_evaluation = _build_profile_linkage_evaluation(
            integrated_issue=integrated_issue,
            classification={},
            profile_context=profile_context,
            relevance_hint_text=profile_relevance_text,
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification={},
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        prompt = ACTION_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
            skax_json=_json_dumps(skax),
        )
        content = self._invoke_llm(
            system_prompt=ACTION_REPAIR_SYSTEM_PROMPT,
            user_prompt=prompt,
            bundle_id=bundle_id,
            phase="quality_repair_actions",
        )
        data = _json_dict(_parse_json_loose(content))
        actions = [
            action
            for index, action in enumerate(
                _string_list(data.get("recommended_actions"), max_items=3), start=1
            )
            if not _repair_action_violation(
                action,
                label=f"skax_implication.recommended_actions[{index}]",
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                action_artifact_plan=action_artifact_plan,
            )
        ]
        if actions:
            skax["recommended_actions"] = actions
            implication["skax_implication"] = skax
            out["implication"] = implication
        return out

    def _legacy_fallback(
        self,
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        input_bundle: AnalysisInputBundle | dict[str, Any] | None,
        profile_context: ProfileContext | dict[str, Any] | None,
        analysis_context: AnalysisContext | None,
        cluster_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        profile_dict = _profile_to_dict(profile_context)
        profile_relevance_text = _profile_relevance_hint_text(
            integrated_issue=integrated_issue,
            classification=classification,
            bundle=_bundle_to_dict(input_bundle),
        )
        profile_linkage_evaluation = _build_profile_linkage_evaluation(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_context=profile_dict,
            relevance_hint_text=profile_relevance_text,
        )
        action_artifact_plan = _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        analysis = self._fallback_analyzer.analyze(
            integrated_issue=integrated_issue,
            classification=classification,
            cluster_metadata=cluster_metadata,
        )
        implication = self._fallback_implication_agent.generate(
            input_bundle=input_bundle,
            integrated_issue=integrated_issue,
            analysis=analysis,
            profile_context=profile_context,
            analysis_context=analysis_context,
            classification=classification,
        )
        result = {
            "is_valid_strategic_insight": bool(
                analysis.get("is_valid_analysis") and implication.get("is_valid_implication")
            ),
            "analysis": _normalize_analysis_block(analysis),
            "implication": _normalize_implication_block(
                implication,
                integrated_issue=integrated_issue,
                profile_context=profile_dict,
                analysis_context={},
                model=self.model,
            ),
        }
        result = _with_fallback_linkage_payloads(
            result,
            profile_linkage_evaluation=profile_linkage_evaluation,
            integrated_issue=integrated_issue,
            action_artifact_plan=action_artifact_plan,
        )
        return _attach_sentence_grounding(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_dict,
        )


__all__ = ["StrategicInsightAgent"]
