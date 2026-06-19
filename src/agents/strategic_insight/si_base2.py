"""strategic_insight si_base2 — extracted from facade (move-only)."""

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


def _frontend_ready_with_issue_evidence_anchors(
    frontend_ready: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(frontend_ready, dict) or not frontend_ready:
        return frontend_ready
    out = json.loads(json.dumps(frontend_ready, ensure_ascii=False, default=str))
    for block_key in ("key_implication", "suggested_action"):
        block = out.get(block_key)
        if not isinstance(block, dict):
            continue
        event_terms = _string_list(block.get("event_anchor_terms"), max_items=8)
        evidence_sentence = str(block.get("evidence_sentence") or "").strip()
        if _evidence_sentence_has_issue_anchor(
            evidence_sentence,
            integrated_issue=integrated_issue,
            event_terms=event_terms,
        ):
            continue
        fact_line = _issue_fact_line_for_frontend_evidence(
            block,
            integrated_issue=integrated_issue,
        )
        if not fact_line:
            continue
        block["evidence_sentence"] = _prepend_issue_fact_to_evidence(
            fact_line,
            evidence_sentence,
        )
    return out


def _issue_fact_line_for_frontend_evidence(
    block: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> str:
    fact_lines = _issue_fact_lines(integrated_issue)
    if not fact_lines:
        return ""
    event_terms = _string_list(block.get("event_anchor_terms"), max_items=8)
    event_terms.extend(_specific_event_anchors_for_frontend(integrated_issue)[:8])
    context_text = " ".join(
        [
            str(block.get("sentence") or ""),
            str(block.get("evidence_sentence") or ""),
            " ".join(event_terms),
        ]
    )
    context_tokens = _distinct_anchor_tokens(context_text)
    best_line = ""
    best_score = -1
    for index, line in enumerate(fact_lines):
        line_text = str(line or "").strip()
        if not line_text:
            continue
        line_norm = _anchor_norm(line_text)
        score = 0
        for term in event_terms:
            term_norm = _anchor_norm(term)
            if term_norm and term_norm in line_norm:
                score += 4
        for token in context_tokens:
            token_norm = _anchor_norm(token)
            if token_norm and token_norm in line_norm:
                score += 1
        # Keep the original fact order as a stable tie-breaker.
        score = score * 1000 - index
        if score > best_score:
            best_score = score
            best_line = line_text
    return best_line


def _is_valid_integrated_issue(integrated_issue: dict[str, Any]) -> bool:
    has_source = bool(
        integrated_issue
        and (
            integrated_issue.get("integrated_text")
            or integrated_issue.get("fact_summary")
            or integrated_issue.get("consolidated_facts")
            or integrated_issue.get("one_line_summary")
            or integrated_issue.get("headline")
        )
    )
    if not has_source:
        return False
    if integrated_issue.get("is_valid_summary", True):
        return True
    return _has_integrated_text_candidate_signal(integrated_issue)


def _has_integrated_text_candidate_signal(integrated_issue: dict[str, Any]) -> bool:
    if not isinstance(integrated_issue, dict):
        return False
    if _is_weak_surface_integrated_issue(integrated_issue) and not _has_sizable_tech_event_signal(
        integrated_issue
    ):
        return False
    lines = _integrated_text_anchor_lines(integrated_issue, max_items=8)
    if len(lines) < 2:
        lines = _dedupe_keep_order([*lines, *_issue_fact_lines(integrated_issue)])
    if len(lines) < 2:
        return False
    grounding = " ".join(lines)
    if _has_sizable_tech_event_signal(integrated_issue):
        return True
    anchors = _specific_event_anchors_for_frontend(integrated_issue)
    return bool(
        len(anchors) >= 2
        and re.search(
            r"변화|확대|고도화|전환|도입|출시|구축|운영|매출|비중|계약|협력|"
            r"시스템|플랫폼|서비스|솔루션|기능|적용|처리|분석|대응|관제|자동화|"
            r"평가|지표|데이터|모델|업무|보안|공급망",
            grounding,
            flags=re.IGNORECASE,
        )
    )


def _strategic_generation_skip_decision(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    """Return a watch-only decision for issues that lack strategic action evidence.

    This gate is intentionally evidence-structure based. It does not block a
    company, card id, or fixed output phrase; it checks whether the current issue
    contains enough business facts for StrategicInsightAgent to create
    frontend_ready copy without inventing a strategy angle.
    """
    if _stock_market_watch_only_issue(
        integrated_issue=integrated_issue,
        classification=classification,
    ):
        return {
            "decision_type": "watch_only_stock_market_signal",
            "watch_only": True,
            "reason": (
                "주식 매매·시황성 신호만 확인되어 전략 시사점/대응방향을 생성하지 않았습니다."
            ),
            "evidence": _watch_only_evidence_summary(
                integrated_issue,
                category="stock_market",
            ),
        }
    hiring_decision = _weak_hiring_watch_only_decision(
        integrated_issue=integrated_issue,
        classification=classification,
    )
    if hiring_decision:
        return hiring_decision
    return {}


def _stock_market_watch_only_issue(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> bool:
    del classification
    fact_lines = _issue_fact_lines(integrated_issue)
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding.strip():
        return False
    stock_lines = [line for line in fact_lines if _has_stock_market_signal(line)]
    if not stock_lines and not _has_stock_market_signal(grounding):
        return False
    business_lines = [
        line
        for line in fact_lines
        if _has_direct_business_signal(line) and not _stock_line_without_business_link(line)
    ]
    if business_lines:
        return False
    return bool(stock_lines) or _stock_line_without_business_link(grounding)


def _has_direct_peer_action_signal(integrated_issue: dict[str, Any]) -> bool:
    fact_lines = _issue_fact_lines(integrated_issue)
    peer_variants: set[str] = set()
    for company_id in _companies_from_integrated_issue(integrated_issue):
        peer_variants.update(_company_variants_for_direct_action_match(company_id))
    peer_variants = {variant for variant in peer_variants if _anchor_norm(variant)}
    if not peer_variants:
        return False
    for line in fact_lines:
        if not _has_direct_business_signal(line):
            continue
        if _text_has_anchor_term(line, sorted(peer_variants)):
            return True
    return False


def _stock_line_without_business_link(text: Any) -> bool:
    value = str(text or "")
    return _has_stock_market_signal(value) and not _has_direct_business_signal(value)


def _weak_hiring_watch_only_decision(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    event_type = str(
        classification.get("event_type")
        or integrated_issue.get("cluster_event_type")
        or integrated_issue.get("event_type")
        or ""
    ).casefold()
    grounding = _integrated_grounding_text(integrated_issue)
    is_hiring_or_org = event_type in {"personnel", "hiring", "organization"} or bool(
        re.search(r"채용|공채|인사|임원|조직|전담\s*조직", grounding)
    )
    if not is_hiring_or_org:
        return {}
    anchors = _hiring_signal_anchors(grounding)
    if anchors.get("job_or_tech") or anchors.get("organization_or_business"):
        return {}
    return {
        "decision_type": "watch_only_weak_hiring_signal",
        "watch_only": True,
        "reason": (
            "채용·인사성 이슈이지만 직무군, 기술/사업 영역, 신규 조직/사업 "
            "연결성이 충분하지 않아 전략 시사점/대응방향을 생성하지 않았습니다."
        ),
        "evidence": {
            "event_type": event_type,
            "hiring_or_org_terms": anchors.get("hiring_or_org_terms", []),
            "job_or_tech_terms": anchors.get("job_or_tech", []),
            "organization_or_business_terms": anchors.get("organization_or_business", []),
            "scale_terms": anchors.get("scale", []),
        },
    }


def _watch_only_evidence_summary(
    integrated_issue: dict[str, Any],
    *,
    category: str,
) -> dict[str, Any]:
    fact_lines = _issue_fact_lines(integrated_issue)
    return {
        "category": category,
        "stock_market_lines": [line for line in fact_lines if _has_stock_market_signal(line)][:5],
        "business_signal_lines": [line for line in fact_lines if _has_direct_business_signal(line)][
            :5
        ],
        "fact_line_count": len(fact_lines),
    }


def _frontend_ready_actionable_signal_level(integrated_issue: dict[str, Any]) -> str:
    """Classify whether issue facts can support frontend_ready copy.

    This is intentionally evidence-shape based, not company/card/type based:
    strong signals have direct execution facts; moderate signals have enough
    concrete business/service facts for cautious strategy copy; weak signals
    should remain review/watch-only unless an LLM can ground them cleanly.
    """
    if _has_moderate_actionable_issue_signal(integrated_issue):
        if not _has_strong_actionable_issue_signal(integrated_issue):
            return "moderate"
        if _strong_signal_is_only_generic_execution_word(integrated_issue):
            return "moderate"
    if _has_strong_actionable_issue_signal(integrated_issue):
        return "strong"
    if _has_moderate_actionable_issue_signal(integrated_issue):
        return "moderate"
    if _has_integrated_issue_candidate_anchor_signal(integrated_issue):
        return "moderate"
    return "weak"


def _has_moderate_actionable_issue_signal(integrated_issue: dict[str, Any]) -> bool:
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding.strip():
        return False
    signal_count = sum(
        bool(check(grounding, integrated_issue))
        for check in (
            _has_business_structure_fact,
            _has_service_advancement_fact,
            _has_execution_or_operation_fact,
            _has_clear_business_domain_fact,
        )
    )
    if _has_business_structure_fact(grounding, integrated_issue) and (
        _has_service_advancement_fact(grounding, integrated_issue)
        or _has_execution_or_operation_fact(grounding, integrated_issue)
    ):
        return True
    return signal_count >= 3 and len(_specific_event_anchors_for_frontend(integrated_issue)) >= 2


def _has_integrated_issue_candidate_anchor_signal(integrated_issue: dict[str, Any]) -> bool:
    if not isinstance(integrated_issue, dict):
        return False
    if integrated_issue.get("is_valid_summary") is False:
        return _has_integrated_text_candidate_signal(integrated_issue)
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding.strip() or _is_weak_surface_integrated_issue(integrated_issue):
        return False
    anchors = _specific_event_anchors_for_frontend(integrated_issue)
    if len(anchors) >= 2:
        return True
    fact_lines = _issue_fact_lines(integrated_issue)
    if len(fact_lines) < 2:
        return False
    return bool(
        re.search(
            r"변화|확대|고도화|전환|도입|출시|구축|운영|매출|비중|계약|협력|"
            r"시스템|플랫폼|서비스|솔루션|기능|적용|처리|분석|대응|관제|자동화",
            grounding,
            flags=re.IGNORECASE,
        )
    )


def _is_weak_surface_integrated_issue(integrated_issue: dict[str, Any]) -> bool:
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding.strip():
        return True
    weak_surface_pattern = r"웨비나|세미나|컨퍼런스|포럼|행사|경진대회|공모전|캠페인|홍보"
    if not re.search(weak_surface_pattern, grounding, flags=re.IGNORECASE):
        return False
    non_event_execution_pattern = (
        r"계약\s*체결|수주|업무협약|실시협약|공급\s*계약|고객\s*(적용|도입)|"
        r"제품\s*출시|서비스\s*출시|솔루션\s*출시|플랫폼\s*출시|"
        r"도입해\s*서비스\s*고도화|서비스\s*고도화|자동화|업무\s*처리|"
        r"취약점\s*탐지|보완\s*조치|보안사고\s*대응|운영\s*책임|"
        r"현장\s*적용|특정\s*현장\s*적용"
    )
    return not (
        _has_business_structure_fact(grounding, integrated_issue)
        or _has_sizable_tech_event_signal(integrated_issue)
        or re.search(non_event_execution_pattern, grounding, flags=re.IGNORECASE)
    )


def _frontend_ready_required_violations(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> list[str]:
    implication = result.get("implication") or {}
    frontend_ready = implication.get("frontend_ready") or {}
    if not isinstance(frontend_ready, dict):
        return ["frontend_ready: 카드뉴스용 직접 생성 문장이 없습니다."]
    violations: list[str] = []
    source = str(frontend_ready.get("source") or "").strip()
    if source not in _FRONTEND_READY_DISPLAY_SOURCES:
        violations.append(
            "frontend_ready.source: llm_direct 또는 frontend_repair_direct "
            "결과만 화면에 노출할 수 있습니다."
        )
    if _frontend_ready_actionable_signal_level(
        integrated_issue
    ) == "weak" and _is_weak_surface_integrated_issue(integrated_issue):
        violations.append(
            "frontend_ready: 단순 행사/웨비나/홍보성 이슈는 통합 fact만으로 "
            "SK AX 판단 축을 만들기 어려워 화면 노출하지 않습니다."
        )
    for section_key, label in (
        ("key_implication", "피어사 시사점"),
        ("suggested_action", "SK AX 대응방향"),
    ):
        block = frontend_ready.get(section_key) or {}
        if not isinstance(block, dict):
            block = {}
        block_source = str(block.get("source") or source or "").strip()
        if not str(block.get("sentence") or "").strip():
            violations.append(
                f"frontend_ready.{section_key}.sentence: {label} 결론 문장이 없습니다."
            )
        if not str(block.get("evidence_sentence") or "").strip():
            violations.append(
                f"frontend_ready.{section_key}.evidence_sentence: "
                f"{label} 근거/설명 문장이 없습니다."
            )
        if block_source not in _FRONTEND_READY_DISPLAY_SOURCES:
            violations.append(
                f"frontend_ready.{section_key}.source: 직접 생성된 카드뉴스 문장이 아닙니다."
            )
        event_terms = _string_list(block.get("event_anchor_terms"), max_items=8)
        anchor_key = (
            "skax_anchor_terms" if section_key == "suggested_action" else "profile_anchor_terms"
        )
        profile_terms = _string_list(block.get(anchor_key), max_items=8)
        evidence_mode = str(block.get("evidence_mode") or "").strip()
        requires_profile_anchor = evidence_mode == "profile_based"
        if requires_profile_anchor and not profile_terms:
            violations.append(
                f"frontend_ready.{section_key}.{anchor_key}: 프로필/대응 anchor가 없습니다."
            )

        text = " ".join(
            [
                str(block.get("sentence") or ""),
                str(block.get("evidence_sentence") or ""),
            ]
        )
        malformed_copy_violation = _frontend_ready_malformed_display_sentence_violation(text)
        if malformed_copy_violation:
            violations.append(f"frontend_ready.{section_key}: {malformed_copy_violation}")
        internal_copy_violation = _frontend_ready_internal_copy_term_violation(text)
        if internal_copy_violation:
            violations.append(f"frontend_ready.{section_key}: {internal_copy_violation}")
        if not event_terms and not _evidence_sentence_has_dynamic_grounding(
            text,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        ):
            violations.append(
                f"frontend_ready.{section_key}.event_anchor_terms: 현재 사건 anchor가 없습니다."
            )
        if event_terms and not _text_has_anchor_term(text, event_terms):
            violations.append(
                f"frontend_ready.{section_key}: 문장에 현재 사건 anchor가 연결되지 않았습니다."
            )
        if _frontend_ready_evidence_repeats_summary(
            block.get("evidence_sentence"),
            integrated_issue=integrated_issue,
        ):
            violations.append(
                f"frontend_ready.{section_key}.evidence_sentence: "
                "요약 문장을 해석 없이 반복했습니다."
            )
        evidence_anchor_violation = _frontend_ready_evidence_sentence_anchor_violation(
            block,
            integrated_issue=integrated_issue,
        )
        if evidence_anchor_violation:
            violations.append(
                f"frontend_ready.{section_key}.evidence_sentence: {evidence_anchor_violation}"
            )
        if section_key == "key_implication":
            direction_violation = _frontend_ready_key_sentence_direction_violation(
                block.get("sentence"),
                integrated_issue=integrated_issue,
            )
            if direction_violation:
                violations.append(f"frontend_ready.key_implication.sentence: {direction_violation}")
            business_depth_violation = _frontend_ready_key_business_depth_violation(
                block.get("sentence"),
                integrated_issue=integrated_issue,
            )
            if business_depth_violation:
                violations.append(
                    f"frontend_ready.key_implication.sentence: {business_depth_violation}"
                )
            action_language_violation = _frontend_ready_insight_evidence_action_language_violation(
                block.get("evidence_sentence")
            )
            if action_language_violation:
                violations.append(
                    f"frontend_ready.key_implication.evidence_sentence: {action_language_violation}"
                )
            interpretation_violation = _frontend_ready_financial_interpretation_overlap_violation(
                block,
                integrated_issue=integrated_issue,
            )
            if interpretation_violation:
                violations.append(f"frontend_ready.key_implication: {interpretation_violation}")
        concept_violation = _frontend_ready_unsupported_business_concept_violation(
            block,
            integrated_issue=integrated_issue,
        )
        if concept_violation:
            violations.append(f"frontend_ready.{section_key}: {concept_violation}")
        if (
            requires_profile_anchor
            and profile_terms
            and not _text_has_anchor_term(
                text,
                profile_terms,
            )
        ):
            violations.append(
                f"frontend_ready.{section_key}: 문장에 프로필/대응 anchor가 연결되지 않았습니다."
            )
        if section_key == "key_implication" and _mentions_skax_actor(text):
            violations.append(
                "frontend_ready.key_implication: 시사점에 SK AX 대응 관점이 섞였습니다."
            )
        if section_key == "suggested_action" and not _mentions_skax_actor(
            str(block.get("sentence") or "")
        ):
            violations.append("frontend_ready.suggested_action: SK AX 행동 관점이 없습니다.")
        if section_key == "suggested_action":
            scale_violation = _frontend_ready_action_auxiliary_scale_overreach_violation(block)
            if scale_violation:
                violations.append(f"frontend_ready.suggested_action: {scale_violation}")
            mode_violation = _frontend_ready_skax_action_mode_violation(
                block,
                integrated_issue=integrated_issue,
                profile_linkage_evaluation=profile_linkage_evaluation,
            )
            if mode_violation:
                violations.append(f"frontend_ready.suggested_action: {mode_violation}")
            peer_product_violation = _frontend_ready_peer_product_as_skax_basis_violation(
                block,
                integrated_issue=integrated_issue,
                profile_linkage_evaluation=profile_linkage_evaluation,
            )
            if peer_product_violation:
                violations.append(f"frontend_ready.suggested_action: {peer_product_violation}")
    return violations


def _frontend_ready_key_sentence_direction_violation(
    sentence: Any,
    *,
    integrated_issue: dict[str, Any],
) -> str:
    value = str(sentence or "").strip()
    if not value:
        return ""
    if _mentions_skax_actor(value):
        return ""
    direction_pattern = (
        r"방향|암시|부각|이동|전환|확장|확대|구체화|재편|분화|"
        r"비교\s*기준|평가\s*기준|경쟁\s*(축|기준|방식)|"
        r"고객\s*(요구|수요|기준)|운영\s*(방식|구조|책임)|"
        r"제안\s*(방식|구조|단위)|서비스\s*구조|협력\s*구조|"
        r"매출\s*(구성|구조)|거래\s*(구조|의존도)|대외\s*(매출|고객)"
    )
    has_direction = bool(re.search(direction_pattern, value, flags=re.IGNORECASE))
    generic_signal_pattern = (
        r"(?:실행|관찰|공개|확인|연결|적용|선정|협약|도입|출시)\s*"
        r"(?:된\s*)?(?:흐름|장면|사례|신호)"
    )
    generic_signal_only = bool(re.search(generic_signal_pattern, value))
    if generic_signal_only and not has_direction:
        return (
            "핵심 시사점이 현재 사실을 라벨링하는 수준입니다. "
            "현재 사건이 앞으로 어떤 경쟁 기준, 사업 구조, 운영 방식, 고객 요구를 "
            "암시하는지 상위 해석을 담아야 합니다."
        )
    if not has_direction and _frontend_ready_sentence_restates_issue_fact(
        value,
        integrated_issue=integrated_issue,
    ):
        return (
            "핵심 시사점이 기사 사실 요약에 가깝습니다. "
            "결론 문장은 사실 자체보다 그 사실이 암시하는 방향성이나 평가 기준을 말해야 합니다."
        )
    return ""


def _frontend_ready_key_business_depth_violation(
    sentence: Any,
    *,
    integrated_issue: dict[str, Any],
) -> str:
    value = str(sentence or "").strip()
    if not value or _mentions_skax_actor(value):
        return ""
    business_terms_pattern = (
        r"고객\s*(접점|제안|수요|요구|군|확보)|대외\s*(시장|매출|고객|성과)|"
        r"매출\s*(구성|구조|기반)|거래\s*(구조|의존도|독립성)|"
        r"수익\s*모델|반복\s*매출|플랫폼\s*(주도권|화|운영|구조|기반|활용|제안|구성|역할|책임)|플랫폼화|"
        r"고객\s*락인|락인|레퍼런스|운영\s*(책임|구조|방식|체계)|"
        r"사업\s*(구조|자생력|실익)|제안\s*(단위|구조|방식)|"
        r"파트너십|협력\s*구조|비캡티브|"
        r"업무\s*(시스템|처리|자동화|범위)|대상\s*시스템|"
        r"처리\s*범위|적용\s*(업무|대상|범위)|시스템\s*(접점|연계|처리)|"
        r"학습\s*(데이터|플랫폼|구조|방식)|통합\s*관제|관제\s*(책임|플랫폼|구조|운영)|"
        r"로봇\s*(학습|운영|적용|자동화)|현장\s*(적용|운영|시스템\s*연계)|물류센터\s*(운영|자동화|적용)|"
        r"계약\s*(구조|형태|범위|금액|기간|체결|추가)|공급\s*계약|추가\s*수주|"
        r"운영\s*계약|운용\s*지원|구축[·ㆍ/\\ -]*운용|자원\s*확보|"
        r"참여\s*(기업|주체|구조)|사업\s*참여|인프라\s*(구축|운영|확보|준비)|"
        r"데이터센터|현장\s*실사|기술\s*공급\s*구조"
    )
    if re.search(business_terms_pattern, value, flags=re.IGNORECASE):
        return ""
    generic_direction_pattern = (
        r"경쟁\s*(기준|축|방식)|평가\s*기준|비교\s*기준|"
        r"부각|이동|전환|확장|확대|구체화|암시"
    )
    if re.search(generic_direction_pattern, value, flags=re.IGNORECASE):
        grounding = _integrated_grounding_text(integrated_issue)
        if _is_product_or_service_launch_issue(integrated_issue) and (
            _workflow_execution_business_terms(grounding)
        ):
            return ""
        if _frontend_ready_actionable_signal_level(integrated_issue) == "strong" and (
            _high_signal_issue_overlap_count(value, integrated_issue) >= 1
            or _has_integrated_issue_candidate_anchor_signal(integrated_issue)
        ):
            return ""
        signal_level = _frontend_ready_actionable_signal_level(integrated_issue)
        if signal_level == "moderate" and (
            _high_signal_issue_overlap_count(value, integrated_issue) >= 1
            or _has_moderate_actionable_issue_signal(integrated_issue)
        ):
            return ""
        if _business_context_terms(grounding):
            return (
                "핵심 시사점이 경쟁 기준 변화만 말하고 비즈니스 실익을 충분히 "
                "해석하지 못했습니다. 입력 근거에서 설명 가능한 사업적 판단 축을 "
                "함께 담아야 합니다."
            )
    return ""


def _frontend_ready_sentence_restates_issue_fact(
    sentence: str,
    *,
    integrated_issue: dict[str, Any],
) -> bool:
    sentence_terms = _frontend_ready_role_terms(sentence)
    if len(sentence_terms) < 3:
        return False
    for summary_line in _string_list(integrated_issue.get("fact_summary"), max_items=8):
        fact_terms = _frontend_ready_role_terms(summary_line)
        if len(fact_terms) < 3:
            continue
        overlap = len(sentence_terms & fact_terms) / max(len(sentence_terms | fact_terms), 1)
        if overlap >= 0.62:
            return True
    return False


def _frontend_ready_financial_interpretation_overlap_violation(
    block: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> str:
    if not _is_financial_or_transaction_frontend_block(block, integrated_issue):
        return ""
    sentence = str(block.get("sentence") or "")
    evidence = str(block.get("evidence_sentence") or "")
    if not sentence.strip() or not evidence.strip():
        return ""
    sentence_numbers = _numeric_token_keys(sentence)
    evidence_numbers = _numeric_token_keys(evidence)
    shared_numbers = sentence_numbers & evidence_numbers
    sentence_structure_terms = _financial_structure_terms(sentence)
    evidence_structure_terms = _financial_structure_terms(evidence)
    shared_structure_terms = sentence_structure_terms & evidence_structure_terms
    if shared_numbers and (
        len(sentence_numbers) >= 2 or len(shared_numbers) >= 2 or len(shared_structure_terms) >= 2
    ):
        return (
            "재무/거래구조 시사점 결론이 수치·비교군 근거를 반복합니다. "
            "결론은 매출 구성, 거래 의존도, 평가 기준 같은 상위 해석으로 쓰고 "
            "수치와 비교군은 근거/설명에 배치해야 합니다."
        )
    return ""


def _frontend_ready_skax_action_mode_violation(
    block: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None,
) -> str:
    if _is_financial_or_transaction_frontend_block(block, integrated_issue):
        return ""
    mode = _skax_action_mode_from_profile_linkage(profile_linkage_evaluation)
    if mode == "direct_business_match":
        return ""
    text = " ".join(
        [
            str(block.get("sentence") or ""),
            str(block.get("evidence_sentence") or ""),
        ]
    )
    if not text.strip():
        return ""
    has_direct_action = _has_direct_skax_execution_action(text)
    if mode == "watch_or_monitor":
        if has_direct_action:
            return (
                "SK AX 프로필 연결 근거가 약한데 직접 사업 대응처럼 작성했습니다. "
                "연결 근거가 거의 없으면 피어/산업 동향 모니터링, 수요 검증, "
                "접점 확인 수준으로 낮춰야 합니다."
            )
    if mode == "adjacent_opportunity_probe" and has_direct_action:
        return (
            "SK AX 프로필 연결이 인접 접점 수준인데 직접 도입/확보/구축처럼 작성했습니다. "
            "고객 수요, 적용 가능성, 파트너십 필요성, 기존 시스템 접점 검토로 낮춰야 합니다."
        )
    return ""


def _frontend_ready_peer_product_as_skax_basis_violation(
    block: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None,
) -> str:
    if _is_financial_or_transaction_frontend_block(block, integrated_issue):
        return ""
    peer_only_terms = _peer_only_issue_product_terms(
        integrated_issue,
        profile_linkage_evaluation=profile_linkage_evaluation,
    )
    if not peer_only_terms:
        return ""
    sentence = str(block.get("sentence") or "")
    evidence = str(block.get("evidence_sentence") or "")
    matched_in_sentence = [
        term for term in peer_only_terms if _text_has_anchor_term(sentence, [term])
    ]
    matched_in_evidence = [
        term for term in peer_only_terms if _text_has_anchor_term(evidence, [term])
    ]
    sentence_basis_pattern = (
        r"처럼|같은\s*제품|동일한\s*제품|비교\s*기준|"
        r"내부\s*비교|판단\s*기준|삼아야|직접\s*(기준|비교)"
    )
    evidence_basis_pattern = r"같은\s*제품|동일한\s*제품|내부\s*비교|삼아야|직접\s*(기준|비교)"
    if matched_in_sentence and re.search(sentence_basis_pattern, sentence):
        return (
            "피어사 고유 제품명을 SK AX 대응방향의 직접 기준처럼 사용했습니다. "
            "SK AX 프로필에 같은 제품/역량 근거가 없으면 제품명 대신 해당 제품이 맡는 "
            "기능, 적용 업무, 대상 시스템, 운영 역할, 기존 시스템 접점 같은 "
            "구조 표현으로 낮춰야 합니다."
        )
    if matched_in_evidence and re.search(evidence_basis_pattern, evidence):
        return (
            "피어사 고유 제품명을 SK AX 내부 판단 근거처럼 사용했습니다. "
            "대응방향 근거에서는 피어 제품명보다 현재 사건의 기능·업무 범위와 "
            "SK AX 연결 강도를 기준으로 설명해야 합니다."
        )
    peer_structure_pattern = (
        r"\d+\s*개\s*(?:모듈|라인업|서비스|제품|솔루션)|"
        r"(?:모듈|라인업|제품\s*구조|서비스\s*라인업)\s*(?:기반|형|구조|제안)"
    )
    if re.search(peer_structure_pattern, sentence) and re.search(
        sentence_basis_pattern + r"|기준|비교|구분|나눠",
        sentence,
    ):
        return (
            "피어사의 제품 구조, 모듈 수, 고유 라인업을 SK AX 대응방향의 직접 "
            "비교 기준처럼 사용했습니다. 대응방향 sentence에서는 제품 구조를 "
            "그대로 옮기지 말고 적용 업무, 대상 시스템, 처리 범위, 운영 역할, "
            "기존 시스템 접점 같은 일반 판단 축으로 낮춰야 합니다."
        )
    return ""


def _specific_event_anchors_for_frontend(integrated_issue: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    texts.extend(
        str(item or "") for item in _string_list(integrated_issue.get("fact_summary"), max_items=8)
    )
    texts.extend(_integrated_text_anchor_lines(integrated_issue, max_items=8))
    for _, fact_text in _fact_texts(integrated_issue):
        texts.append(fact_text)
    for key in ("headline", "main_event", "one_line_summary"):
        texts.append(str(integrated_issue.get(key) or ""))
    joined = "\n".join(text for text in texts if text)
    candidates: list[str] = []
    candidates.extend(
        re.findall(
            r"[0-9][0-9.,]*\s*(?:%|억원|조원|조|개|명|년|개월|주|분|장|여\s*개)",
            joined,
        )
    )
    candidates.extend(re.findall(r"[‘'\"“”]([^‘'\"“”]{2,40})[’'\"“”]", joined))
    candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9&+._-]{1,}\b", joined))
    for token in re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·_-]{2,}", joined):
        normalized = re.sub(
            r"(하기로|했습니다|합니다|했다|한다|하려는|하는|으로|에서|에게|과|와|은|는|이|가|을|를|의)$",
            "",
            token.strip(),
        )
        norm = _anchor_norm(normalized)
        if len(norm) >= 4 or (len(norm) >= 3 and not _is_low_signal_content_token(normalized)):
            candidates.append(normalized)
    generic_norms = {
        _anchor_norm(item)
        for item in (
            "이번",
            "해당",
            "시장",
            "경쟁",
            "가능성",
            "기업",
            "업무",
            "서비스",
            "사업",
            "기반",
            "관련",
            "추진",
            "제공",
            "활용",
        )
    }
    for company_id in _companies_from_integrated_issue(integrated_issue):
        generic_norms.add(_anchor_norm(company_id))
        for variant in _company_token_variants(company_id):
            generic_norms.add(_anchor_norm(variant))
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = re.sub(r"\s+", " ", str(candidate or "")).strip(" ,.;:()[]")
        norm = _anchor_norm(value)
        if len(norm) < 3 or norm in generic_norms or norm in seen:
            continue
        if any(norm and norm in _anchor_norm(existing) for existing in result):
            continue
        result.append(value)
        seen.add(norm)
        if len(result) >= 12:
            break
    return result


def _has_displayable_frontend_ready(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> bool:
    required_violations = _frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context=profile_context,
        profile_linkage_evaluation=profile_linkage_evaluation,
    )
    claim_violations = _frontend_ready_claim_violations(
        result,
        integrated_issue=integrated_issue,
        profile_linkage_evaluation=profile_linkage_evaluation or {},
    )
    return not required_violations and not claim_violations


def _issue_execution_slot_diagnostics(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    signals = _extract_issue_structured_signals(
        integrated_issue=integrated_issue,
        classification={},
    )
    schema_keys = set(signals.keys()) | _nested_mapping_keys(integrated_issue)
    grounding = _integrated_grounding_text(integrated_issue)
    counterparties = _issue_counterparty_terms(integrated_issue, grounding)
    target_systems = _slot_terms(
        [
            *signals.get("target_systems", []),
            *_regex_slot_terms(
                grounding,
                r"[가-힣A-Za-z0-9&+·._-]{2,40}(?:시스템|센터|플랫폼|인프라|서비스|사업|공장|물류센터)",
            ),
        ],
        max_items=8,
        exclude_values=schema_keys,
    )
    products_or_services = _slot_terms(
        [
            *signals.get("products_or_services", []),
            *re.findall(r"[‘'\"“”]([^‘'\"“”]{2,40})[’'\"“”]", grounding),
        ],
        max_items=8,
        exclude_values=schema_keys,
        reject_sentence_like=True,
    )
    execution_scope = _slot_terms(
        [
            *signals.get("activity_types", []),
            str(signals.get("event_type") or ""),
            *_regex_slot_terms(
                grounding,
                r"(?:업무협약|실시협약|주주간\s*계약|계약\s*체결|최종\s*선정|"
                r"구축|운영|도입|출시|개시|실증|공급|전환|투자|협력)",
            ),
        ],
        max_items=8,
        exclude_values=schema_keys,
    )
    slots = {
        "counterparty": counterparties,
        "target_system": target_systems,
        "product_or_service": products_or_services,
        "execution_scope": execution_scope,
    }
    return {
        **slots,
        "missing_slots": [key for key, value in slots.items() if not value],
    }


def _slot_terms(
    values: Sequence[Any],
    *,
    max_items: int,
    exclude_values: set[str] | None = None,
    reject_sentence_like: bool = False,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    excluded_norms = {_anchor_norm(value) for value in (exclude_values or set())}
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:()[]")
        norm = _anchor_norm(cleaned)
        if reject_sentence_like and _looks_like_sentence_slot(cleaned):
            continue
        if len(norm) < 2 or norm in seen or norm in excluded_norms:
            continue
        seen.add(norm)
        result.append(cleaned)
        if len(result) >= max_items:
            break
    return result


def _profile_based_downgrade_diagnostics(
    frontend_ready: dict[str, Any],
    *,
    profile_linkage_evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(frontend_ready, dict):
        frontend_ready = {}
    diagnostics: list[dict[str, Any]] = []
    for section_key, scope, anchor_key in (
        ("key_implication", "peer", "profile_anchor_terms"),
        ("suggested_action", "skax", "skax_anchor_terms"),
    ):
        linkage_level = _relevant_profile_linkage_level_from_evaluation(
            profile_linkage_evaluation,
            scope=scope,
        )
        if linkage_level not in {"high", "medium"}:
            continue
        block = frontend_ready.get(section_key) or {}
        if not isinstance(block, dict):
            block = {}
        evidence_mode = str(block.get("evidence_mode") or "").strip() or None
        if evidence_mode == "profile_based":
            continue
        anchors = _string_list(block.get(anchor_key), max_items=8)
        reason = (
            f"{anchor_key}_missing" if not anchors else "writer_selected_non_profile_based_mode"
        )
        diagnostics.append(
            {
                "section": section_key,
                "scope": scope,
                "linkage_level": linkage_level,
                "evidence_mode": evidence_mode,
                "reason": reason,
                "anchor_terms": anchors,
            }
        )
    return diagnostics


def _frontend_ready_claim_violations(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any],
) -> list[str]:
    implication = result.get("implication") or {}
    frontend_ready = implication.get("frontend_ready") or {}
    if not isinstance(frontend_ready, dict):
        return []
    violations: list[str] = []
    for section_key, scope in (
        ("key_implication", "peer"),
        ("suggested_action", "skax"),
    ):
        block = frontend_ready.get(section_key) or {}
        if not isinstance(block, dict):
            continue
        claim_type = str(block.get("claim_type") or "").strip()
        claim_strength = str(block.get("claim_strength") or "").strip()
        evidence_mode = str(block.get("evidence_mode") or "").strip()
        linkage_level = _relevant_profile_linkage_level_from_evaluation(
            profile_linkage_evaluation,
            scope=scope,
        )
        if not claim_type:
            violations.append(f"frontend_ready.{section_key}.claim_type: 주장 유형이 없습니다.")
            continue
        if claim_type not in _FRONTEND_READY_CLAIM_TYPES:
            violations.append(
                f"frontend_ready.{section_key}.claim_type: 허용되지 않은 주장 유형입니다."
            )
        if claim_strength not in _FRONTEND_READY_CLAIM_STRENGTHS:
            violations.append(f"frontend_ready.{section_key}.claim_strength: 주장 강도가 없습니다.")
        if evidence_mode not in _FRONTEND_READY_EVIDENCE_MODES:
            violations.append(f"frontend_ready.{section_key}.evidence_mode: 근거 모드가 없습니다.")

        if linkage_level in {"low", "none", ""}:
            if evidence_mode == "profile_based":
                violations.append(
                    f"frontend_ready.{section_key}: 프로필 연결이 약한데 "
                    "profile_based로 작성했습니다."
                )
            if claim_strength == "strong":
                violations.append(
                    f"frontend_ready.{section_key}: 프로필 연결이 약한데 "
                    "strong claim으로 작성했습니다."
                )

        if claim_type in _FRONTEND_READY_STRONG_CLAIM_TYPES:
            if evidence_mode != "profile_based" or linkage_level not in {"high", "medium"}:
                violations.append(
                    f"frontend_ready.{section_key}: 강한 주장 유형은 "
                    "충분한 프로필 근거가 필요합니다."
                )
            if claim_strength == "strong" and linkage_level != "high":
                violations.append(
                    f"frontend_ready.{section_key}: strong claim은 high linkage에서만 허용합니다."
                )

        inferred = _infer_frontend_claim_type(
            " ".join(
                [
                    str(block.get("sentence") or ""),
                    str(block.get("evidence_sentence") or ""),
                ]
            )
        )
        if inferred in _FRONTEND_READY_STRONG_CLAIM_TYPES and (
            evidence_mode != "profile_based" or linkage_level not in {"high", "medium"}
        ):
            violations.append(
                f"frontend_ready.{section_key}: 문장 표현은 강한 주장에 "
                "가깝지만 근거 모드가 부족합니다."
            )
        effect_violation = _frontend_ready_unsupported_effect_violation(
            " ".join(
                [
                    str(block.get("sentence") or ""),
                    str(block.get("evidence_sentence") or ""),
                ]
            ),
            integrated_issue=integrated_issue,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if effect_violation:
            violations.append(f"frontend_ready.{section_key}: {effect_violation}")
    return violations


def _evidence_sentence_has_issue_anchor(
    evidence_sentence: Any,
    *,
    integrated_issue: dict[str, Any],
    event_terms: Sequence[str] | None = None,
) -> bool:
    evidence = str(evidence_sentence or "").strip()
    if not evidence:
        return False
    candidate_terms = _dedupe_keep_order(
        [
            *_string_list(event_terms or [], max_items=8),
            *_specific_event_anchors_for_frontend(integrated_issue),
        ]
    )
    if candidate_terms and _text_has_anchor_term(evidence, candidate_terms):
        return True

    evidence_tokens = _distinct_anchor_tokens(evidence)
    if not evidence_tokens:
        return False
    grounding_norm = _anchor_norm(_integrated_grounding_text(integrated_issue))
    matched = [token for token in evidence_tokens if _anchor_norm(token) in grounding_norm]
    if any(len(_anchor_norm(token)) >= 5 for token in matched):
        return True
    return len(matched) >= 2


def _frontend_ready_evidence_sentence_anchor_violation(
    block: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> str:
    evidence_sentence = block.get("evidence_sentence")
    event_terms = _string_list(block.get("event_anchor_terms"), max_items=8)
    issue_anchors = _specific_event_anchors_for_frontend(integrated_issue)
    if not event_terms and not issue_anchors:
        return ""
    if _evidence_sentence_has_issue_anchor(
        evidence_sentence,
        integrated_issue=integrated_issue,
        event_terms=event_terms,
    ):
        return ""
    return "근거/설명에 현재 사건의 구체 기사 anchor가 연결되지 않았습니다."


def _action_plan_issue_terms(action_artifact_plan: dict[str, Any]) -> set[str]:
    primary_terms = _action_plan_terms_for_keys(
        action_artifact_plan,
        keys=(
            "products_or_services",
            "target_systems",
            "customers_or_industries",
            "evidence_terms",
        ),
    )
    if primary_terms:
        return primary_terms
    return _action_plan_terms_for_keys(
        action_artifact_plan,
        keys=(
            "structured_terms",
            "activity_types",
            "event_type",
        ),
    )


def _solution_term_supported_by_evidence(
    text: str,
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> bool:
    del text
    evidence_text = _integrated_grounding_text(integrated_issue)
    if "솔루션" in evidence_text:
        return True
    profile_terms = _concrete_profile_terms(
        profile_context,
        integrated_issue=integrated_issue,
        scope="skax",
    ) | _concrete_profile_terms(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    return "솔루션" in profile_terms


def _has_concrete_profile_term(
    text: str,
    *,
    profile_context: dict[str, Any],
    integrated_issue: dict[str, Any],
    scope: str,
) -> bool:
    output_tokens = _content_tokens(str(text or ""))
    profile_terms = _concrete_profile_terms(
        profile_context,
        integrated_issue=integrated_issue,
        scope=scope,
    )
    return bool(output_tokens & profile_terms)


def _weak_analysis_statement(text: str, *, integrated_issue: dict[str, Any]) -> bool:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return True
    primary = _primary_issue_fact(integrated_issue).rstrip(".")
    if value.rstrip(".") == primary:
        return True
    return bool(
        re.fullmatch(r".{0,40}(중요|변화|관찰|시사)(하|되|되고|된다|고 있다).{0,20}", value)
    )


def _profile_area_phrase(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> str:
    names = _relevant_profile_area_names(
        profile_context,
        integrated_issue=integrated_issue,
        scope=scope,
    )
    limit = 1 if scope == "skax" else 2
    return "·".join(names[:limit])


def _relevant_profile_area_names(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> list[str]:
    prompt_profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue)
    profiles: list[dict[str, Any]] = []
    if scope == "skax":
        skax = prompt_profile.get("skax_profile") or {}
        if isinstance(skax, dict):
            profiles.append(skax)
    else:
        peer_profiles = prompt_profile.get("peer_profiles") or {}
        if isinstance(peer_profiles, dict):
            for company_id in _companies_from_integrated_issue(integrated_issue):
                profile = peer_profiles.get(company_id) or {}
                if isinstance(profile, dict):
                    profiles.append(profile)
    relevance_tokens = _issue_relevance_tokens(integrated_issue)
    names: list[str] = []
    for profile in profiles:
        business_areas = profile.get("business_areas") or []
        if not isinstance(business_areas, list):
            continue
        ranked = _rank_relevant_profile_items(
            [item for item in business_areas if isinstance(item, dict)],
            relevance_tokens=relevance_tokens,
            max_items=5,
        )
        for area in ranked:
            name = str(area.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
    if names:
        return names
    return _relevant_profile_named_terms(
        profiles,
        relevance_tokens=relevance_tokens,
        integrated_issue=integrated_issue,
        max_items=3,
    )


def _checkpoint_hint_from_action_plan(action_artifact_plan: dict[str, Any]) -> str:
    issue_terms = _action_plan_issue_terms(action_artifact_plan)
    focused_terms = [
        term
        for term in _dedupe_keep_order(_string_list(issue_terms, max_items=4))
        if _anchor_norm(term)
    ][:3]
    if focused_terms:
        return f"{'·'.join(focused_terms)} 관련 범위, 책임, 일정 조건, 검증 기준"
    return "범위, 책임, 일정 조건, 검증 기준, 리스크"


def _safe_business_line_mapping(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> list[str]:
    issue_tokens = _issue_relevance_tokens(integrated_issue)
    candidates = _business_line_candidate_details(
        profile_context,
        integrated_issue=integrated_issue,
    )
    selected = []
    for item in candidates:
        name = str(item.get("name") or "").strip()
        if name and (_content_tokens(name) & issue_tokens):
            selected.append(name)
    return selected[:2]
