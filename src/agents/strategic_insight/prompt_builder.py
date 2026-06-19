"""strategic_insight prompt_builder — extracted from facade (move-only)."""

from __future__ import annotations

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
from src.analysis.models import AnalysisContext, AnalysisInputBundle, ProfileContext


def _integrated_issue_for_prompt(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": integrated_issue.get("bundle_id"),
        "cluster_id": integrated_issue.get("cluster_id"),
        "representative_id": integrated_issue.get("representative_id"),
        "source_article_ids": integrated_issue.get("source_article_ids", []),
        "main_company": integrated_issue.get("main_company", ""),
        "mentioned_peer_companies": integrated_issue.get("mentioned_peer_companies", []),
        "cluster_event_type": integrated_issue.get("cluster_event_type", ""),
        "headline": integrated_issue.get("headline", ""),
        "main_event": integrated_issue.get("main_event", ""),
        "main_issue": integrated_issue.get("main_issue", ""),
        "one_line_summary": integrated_issue.get("one_line_summary", ""),
        "integrated_text": integrated_issue.get("integrated_text", ""),
        "fact_summary": integrated_issue.get("fact_summary", []),
        "consolidated_facts": (integrated_issue.get("consolidated_facts") or [])[:20],
        "key_numbers": integrated_issue.get("key_numbers", []),
        "business_signals": (integrated_issue.get("business_signals") or [])[:8],
        "representative_sources": integrated_issue.get("representative_sources", []),
        "fact_basis": (integrated_issue.get("fact_basis") or [])[:16],
        "strategic_evidence_inventory": _strategic_evidence_inventory_for_prompt(
            integrated_issue.get("strategic_evidence_inventory") or {}
        ),
        "cluster_fact_intelligence": _cluster_fact_intelligence_for_prompt(
            integrated_issue.get("cluster_fact_intelligence") or {}
        ),
        "role_interpretation_hints": _role_interpretation_hints(integrated_issue),
        "missing_or_uncertain_points": integrated_issue.get("missing_or_uncertain_points", []),
        "confidence": integrated_issue.get("confidence", 0.0),
    }


def _strategic_evidence_pack_for_prompt(
    *,
    integrated_issue: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Compact article-derived evidence for strategic implication generation.

    IntegratedIssue remains the only fact source. This pack simply separates the
    article evidence that the integration step already selected from display
    summaries, so the strategic agent does not infer from card copy alone.
    """

    fact_basis = []
    for item in _jsonish_list(integrated_issue.get("fact_basis"))[:16]:
        if not isinstance(item, dict):
            continue
        evidence_texts = [
            str(text or "").strip()
            for text in _jsonish_list(item.get("evidence_texts"))[:3]
            if str(text or "").strip()
        ]
        evidence_text = str(item.get("evidence_text") or "").strip()
        if evidence_text and evidence_text not in evidence_texts:
            evidence_texts.append(evidence_text)
        fact_text = str(item.get("fact") or "").strip()
        fact_basis.append(
            {
                "fact": fact_text,
                "fact_ids": _string_list(item.get("fact_ids"), max_items=5),
                "source_article_ids": _int_list(item.get("source_article_ids"))[:5],
                "evidence_type": str(item.get("evidence_type") or "").strip(),
                "evidence_texts": evidence_texts[:3],
            }
        )

    consolidated_facts = []
    for item in _jsonish_list(integrated_issue.get("consolidated_facts"))[:20]:
        fact_text = _fact_like_text(item)
        if not fact_text:
            continue
        row: dict[str, Any] = {"fact": fact_text}
        if isinstance(item, dict):
            row["fact_id"] = str(item.get("fact_id") or "").strip()
            row["source_article_ids"] = _int_list(item.get("source_article_ids"))[:5]
        consolidated_facts.append(row)

    representative_sources = []
    for item in _jsonish_list(integrated_issue.get("representative_sources"))[:10]:
        if not isinstance(item, dict):
            continue
        representative_sources.append(
            {
                "article_id": item.get("article_id") or item.get("id"),
                "title": str(item.get("title") or "").strip(),
                "publisher": str(item.get("publisher") or "").strip(),
                "source_name": str(item.get("source_name") or "").strip(),
                "published_at": str(item.get("published_at") or "").strip(),
            }
        )

    bundle_evidence_snippets = []
    for item in _jsonish_list(bundle.get("evidence_snippets"))[:12]:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("evidence_text") or "").strip()
            if not text:
                continue
            bundle_evidence_snippets.append(
                {
                    "text": text,
                    "source_article_ids": _int_list(item.get("source_article_ids"))[:5],
                    "fact_ids": _string_list(item.get("fact_ids"), max_items=5),
                }
            )
        else:
            text = str(item or "").strip()
            if text:
                bundle_evidence_snippets.append({"text": text})

    return {
        "purpose": (
            "Use this article-derived pack before display/card summaries when deriving "
            "strategic implications."
        ),
        "current_event": {
            "headline": integrated_issue.get("headline", ""),
            "main_event": integrated_issue.get("main_event", ""),
            "main_issue": integrated_issue.get("main_issue", ""),
            "one_line_summary": integrated_issue.get("one_line_summary", ""),
            "integrated_text": integrated_issue.get("integrated_text", ""),
        },
        "fact_basis": fact_basis,
        "consolidated_facts": consolidated_facts,
        "strategic_evidence_inventory": _strategic_evidence_inventory_for_prompt(
            integrated_issue.get("strategic_evidence_inventory") or {}
        ),
        "representative_sources": representative_sources,
        "bundle_evidence_snippets": bundle_evidence_snippets,
        "cluster_fact_intelligence": _cluster_fact_intelligence_for_prompt(
            integrated_issue.get("cluster_fact_intelligence") or {}
        ),
        "missing_or_uncertain_points": integrated_issue.get("missing_or_uncertain_points", []),
    }


def _strategic_evidence_inventory_for_prompt(value: Any) -> dict[str, Any]:
    data = _json_dict(value)
    if not data:
        return {}
    out: dict[str, Any] = {}
    for key, limit in (
        ("core_event_facts", 6),
        ("product_or_service_facts", 8),
        ("application_scope_facts", 8),
        ("numbers_and_scale_facts", 6),
        ("roadmap_or_plan_facts", 6),
        ("quote_or_position_facts", 4),
        ("all_preserved_facts", 20),
    ):
        values = _string_list(data.get(key), max_items=limit)
        if values:
            out[key] = values
    return out


def _role_mode_instructions(integrated_issue: dict[str, Any]) -> str:
    if not _main_company_is_customer_or_buyer(integrated_issue):
        return "일반 모드: IntegratedIssue 의 관계 수준을 그대로 보존합니다."
    hints = _role_interpretation_hints(integrated_issue)
    suppliers = _supplier_names_for_target_counterparty(integrated_issue)
    products = _string_list(
        (hints.get("products_or_services") if isinstance(hints, dict) else None),
        max_items=5,
    )
    return "\n".join(
        [
            "계약 상대방/고객 슬롯 모드입니다.",
            "- 타깃 피어가 customers_or_industries 슬롯에 있고 "
            "공급사/계약 체결 주체가 따로 보입니다.",
            f"- 추출된 공급사 후보: {', '.join(suppliers) if suppliers else '없음'}",
            f"- 확인된 사업/제품 후보: {', '.join(products) if products else '없음'}",
            "- analysis 와 peer_implication 에서 타깃 피어를 "
            "프로젝트 추진/참여/제공/공급/지원/운영/확장 주체처럼 쓰지 마세요.",
            "- 타깃 피어는 계약 상대방, 사업 범위, 계약 범위/기간이 확인된 피어로만 설명하세요.",
            "- 다만 '연결성 확인'으로 끝내지 말고, 확인된 사업/제품명이 어떤 산업 과제나 "
            "유사 사업의 비교 기준을 드러내는지까지 해석하세요.",
            "- 이 모드의 좋은 해석은 '계약 사실 → 사업명에 드러난 대상 업무/시스템과 "
            "전환·검증·운영 성격 → 피어 프로필 사업영역 접점' 순서입니다.",
            "- 문장 주어는 가능한 '이번 계약', '해당 사업', '확인된 계약 범위'처럼 "
            "사건/사업명으로 두세요. 타깃 피어를 주어로 두고 참여·추진·제공·수행·확장한다고 "
            "쓰지 마세요.",
            "- '중요성', '필요성', '관련이 깊습니다', '기회로 작용합니다' 같은 결론형 "
            "표현으로 끝내지 말고, 어떤 범위·기간·전환 성격이 확인됐는지 씁니다.",
            "- 공급사 매출 비율은 요약의 계약 규모 근거일 뿐, "
            "타깃 피어의 역량/성과/전략 근거가 아닙니다.",
            "- SK AX 대응방향은 유사 고객군/유사 사업 관점을 유지하되, "
            "외부 고객에게 무엇을 제시하라는 문장이 아니라 SK AX가 가져갈 사업 방향으로 쓰세요.",
            "- SK AX 대응방향은 피어 신호와 SK AX의 접점/차이를 참고하되, "
            "입력 사건이 남기는 사업 판단 관점으로 작성하세요.",
            "- 대응방향은 넓은 실행 장면명이 아니라 입력 사건에서 확인된 "
            "변화의 의미를 중심으로 쓰세요.",
        ]
    )


def _classification_for_prompt(classification: dict[str, Any]) -> dict[str, Any]:
    return {
        "sector": classification.get("sector", ""),
        "sectors": classification.get("sectors", []),
        "event_type": classification.get("event_type", ""),
        "importance": classification.get("importance", ""),
        "importance_score": classification.get("importance_score", 0.0),
        "exposure_band": classification.get("exposure_band", ""),
        "exposure_score": classification.get("exposure_score", 0.0),
        "signals": classification.get("signals", {}),
    }


def _bundle_for_prompt(
    bundle: dict[str, Any],
    cluster_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = bundle.get("metadata") or {}
    return {
        "bundle_id": bundle.get("bundle_id") or cluster_metadata.get("bundle_id"),
        "cluster_id": bundle.get("cluster_id") or cluster_metadata.get("cluster_id"),
        "source_type": bundle.get("source_type") or cluster_metadata.get("source_type"),
        "companies": bundle.get("companies") or cluster_metadata.get("companies", []),
        "sectors": bundle.get("sectors") or cluster_metadata.get("sectors", []),
        "event_type": bundle.get("event_type") or cluster_metadata.get("event_type"),
        "source_count": len(bundle.get("sources") or []),
        "metadata": {
            "representative_id": metadata.get("representative_id"),
            "cluster_article_ids": metadata.get("cluster_article_ids", []),
            "created_at": metadata.get("created_at", ""),
            "trend_context": metadata.get("trend_context", {}),
        },
    }


def _analysis_context_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "peer_event_timeline_recent": (context.get("peer_event_timeline_recent") or [])[:8],
        "sector_pulse_recent": (context.get("sector_pulse_recent") or [])[:4],
        "financial_trend": context.get("financial_trend") or {},
        "event_chain_candidates": (context.get("event_chain_candidates") or [])[:5],
        "similar_cards_rag": (context.get("similar_cards_rag") or [])[:5],
        "evidence_density_per_peer": context.get("evidence_density_per_peer") or {},
        "provenance": context.get("provenance") or {},
    }


def _analysis_context_for_model(
    context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    relevance_hint_text: str = "",
    include_financial_context: bool = False,
) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    relevance_tokens = _issue_relevance_tokens(
        integrated_issue,
        extra_text=relevance_hint_text,
    )
    out = dict(context)
    if not include_financial_context:
        out["financial_trend"] = {}
    for key, limit in (
        ("peer_event_timeline_recent", 8),
        ("sector_pulse_recent", 4),
        ("event_chain_candidates", 5),
        ("similar_cards_rag", 5),
    ):
        out[key] = _relevant_context_items(
            out.get(key),
            relevance_tokens=relevance_tokens,
            max_items=limit,
        )
    return out


def _context_availability_for_prompt(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    analysis_context: dict[str, Any],
) -> dict[str, Any]:
    """Expose whether profile/recent context is usable without generating copy."""
    company_ids = _companies_from_integrated_issue(integrated_issue)
    peer_profiles = profile_context.get("peer_profiles") or {}
    matched_peer_profiles: list[dict[str, Any]] = []
    if isinstance(peer_profiles, dict):
        for company_id in company_ids:
            profile = peer_profiles.get(company_id) or {}
            if isinstance(profile, dict):
                linkage = _peer_profile_linkage(
                    profile_context,
                    integrated_issue=integrated_issue,
                    company_id=company_id,
                )
                matched_peer_profiles.append(
                    {
                        "company_id": company_id,
                        "available": _has_profile_context(profile),
                        "profile_fields": _available_profile_fields(profile),
                        "peer_profile_linkage": linkage,
                    }
                )

    skax_profile = profile_context.get("skax_profile") or {}
    recent_layers = _available_analysis_layers(analysis_context)
    return {
        "matched_peer_profiles": matched_peer_profiles,
        "peer_profile_available": any(item["available"] for item in matched_peer_profiles),
        "skax_profile_available": _has_profile_context(skax_profile),
        "skax_profile_fields": _available_profile_fields(skax_profile),
        "recent_context_layers_available": recent_layers,
        "recent_context_available": bool(recent_layers),
        "guidance": [
            (
                "profile_based_implication_requires_current_fact_plus_peer_profile"
                "_plus_recent_context_when_available"
            ),
            (
                "if_relevant_profile_or_recent_context_is_missing_lower_confidence"
                "_instead_of_fabricating_profile_based_claims"
            ),
            (
                "skax_actions_require_current_signal_plus_skax_profile_plus_internal"
                "_strategy_checkpoint_and_follow_up_monitoring"
            ),
        ],
    }


def _action_artifact_plan_for_prompt(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any],
) -> dict[str, Any]:
    signals = _extract_issue_structured_signals(
        integrated_issue=integrated_issue,
        classification=classification,
    )
    return {
        "artifact_generation_mode": "llm_dynamic",
        "current_issue_signals": {
            "event_type": signals.get("event_type"),
            "activity_types": _string_list(signals.get("activity_types"), max_items=20),
            "products_or_services": _string_list(
                signals.get("products_or_services"),
                max_items=12,
            ),
            "customers_or_industries": _string_list(
                signals.get("customers_or_industries"),
                max_items=12,
            ),
            "target_systems": _string_list(signals.get("target_systems"), max_items=12),
            "structured_terms": _string_list(signals.get("structured_terms"), max_items=30),
            "evidence_terms": _issue_evidence_terms_for_action_plan(integrated_issue),
        },
        "skax_implication_mode": (
            (profile_linkage_evaluation.get("skax_linkage") or {}).get("implication_mode")
            if isinstance(profile_linkage_evaluation, dict)
            else ""
        ),
        "artifact_policy": [
            (
                "현재 사건에서 확인된 대상 사업/시스템/서비스/인프라를 기준으로 "
                "내부 점검 항목을 만든다."
            ),
            (
                "유사 고객군/유사 사업에서 피어사 사업군·역량과 SK AX 사업군·역량이 "
                "겹치는 지점과 달라지는 지점을 비교한다."
            ),
            (
                "겹침/차이는 ProfileContext 또는 business_line_mapping 후보에 있는 항목으로만 "
                "작성하고, 없는 사업군/역량명을 새로 만들지 않는다."
            ),
            (
                "SK AX의 대응 가능 범위, 역량 공백, 운영 구조, 영업 전략, "
                "후속 경쟁사 모니터링 항목을 중심으로 구조화한다."
            ),
            (
                "범위, 책임, 일정 조건, 검증 기준, 리스크, 운영 조건, 후속 모니터링 중 "
                "현재 사건에 맞는 항목을 쓴다."
            ),
            (
                "외부 고객 제안 문장이 아니라 SK AX 내부에서 비교하고 보완할 기준을 "
                "문장 안에 포함한다."
            ),
            "현재 사건이나 SK AX 프로필 근거가 없는 기술명/솔루션명/성공 사례는 쓰지 않는다.",
        ],
        "guidance": (
            "이 객체는 고정 taxonomy가 아니라 LLM이 내부 전략 점검 기준을 만들기 위한 정책입니다."
        ),
    }


def _bundle_to_dict(value: AnalysisInputBundle | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, AnalysisInputBundle):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _profile_to_dict(value: ProfileContext | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, ProfileContext):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _analysis_context_to_dict(value: AnalysisContext | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, AnalysisContext):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}
