"""strategic_insight si_base1 — extracted from facade (move-only)."""

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
from src.services.peer_id_aliases import expand_peer_aliases


def _user_strategy_issue_scope(
    *,
    package: dict[str, Any],
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    metadata = _json_dict(package.get("metadata"))
    return {
        "card_news_id": metadata.get("card_news_id"),
        "integrated_issue_id": (
            package.get("integrated_issue_id")
            or integrated_issue.get("integrated_issue_id")
            or integrated_issue.get("id")
        ),
        "cluster_id": integrated_issue.get("cluster_id"),
        "headline": integrated_issue.get("headline") or integrated_issue.get("main_issue"),
        "title": integrated_issue.get("title") or integrated_issue.get("headline"),
        "event_type": classification.get("event_type")
        or integrated_issue.get("cluster_event_type"),
        "main_company": integrated_issue.get("main_company"),
        "fact_summary": integrated_issue.get("fact_summary"),
        "consolidated_facts": integrated_issue.get("consolidated_facts"),
    }


def _prepend_issue_fact_to_evidence(fact_line: str, evidence_sentence: str) -> str:
    fact = re.sub(r"\s+", " ", str(fact_line or "")).strip()
    evidence = re.sub(r"\s+", " ", str(evidence_sentence or "")).strip()
    if not fact:
        return evidence
    if evidence and _anchor_norm(fact) in _anchor_norm(evidence):
        return evidence
    if fact[-1] not in ".!?。":
        fact += "."
    if not evidence:
        return fact
    return f"{fact} {evidence}"


def _strip_frontend_ready_label(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^핵심\s*(?:시사점|대응)\s*:\s*", "", text).strip()
    text = re.sub(r"^근거\s*/?\s*설명\s*:\s*", "", text).strip()
    return _strip_article_style_lead(text)


def _has_peer_mention(integrated_issue: dict[str, Any]) -> bool:
    grounding = _integrated_grounding_text(integrated_issue)
    variants: set[str] = set()
    for company_id in _companies_from_integrated_issue(integrated_issue):
        variants.update(_company_variants_for_direct_action_match(company_id))
    return bool(variants and _text_has_anchor_term(grounding, sorted(variants)))


def _primary_actor_type_for_issue(integrated_issue: dict[str, Any]) -> str:
    grounding = _integrated_grounding_text(integrated_issue)
    if re.search(r"정부|과학기술정보통신부|과기정통부|산업부|공정위|금융위", grounding):
        return "public_sector"
    if re.search(r"글로벌\s*(벤더|기업|빅테크)|해외\s*(벤더|기업)", grounding) or (
        _global_company_alias_pattern()
        and re.search(_global_company_alias_pattern(), grounding, flags=re.IGNORECASE)
    ):
        return "global_vendor"
    actor_like_terms = re.findall(
        r"[가-힣A-Za-z0-9&._-]+(?:그룹|컨소시엄|기업|회사|기관|정부|벤더)",
        grounding,
    )
    if len(set(actor_like_terms)) >= 2:
        return "multi_actor"
    return "unknown"


def _has_direct_business_signal(text: Any) -> bool:
    value = str(text or "")
    if not value.strip():
        return False
    if _has_stock_market_signal(value):
        return bool(
            re.search(
                r"계약|수주|공급|협약|선정|출시|구축|운영|도입|매출|실적|"
                r"영업이익|투자\s*(?:유치|집행|결정|계획|확대)|지분\s*취득|"
                r"인수|합병|파트너십|사업자|사업\s*참여|업무협약",
                value,
                flags=re.IGNORECASE,
            )
        )
    return bool(
        re.search(
            r"계약|수주|공급|협약|선정|출시|구축|운영|도입|매출|실적|"
            r"영업이익|투자|인수|합병|파트너십|서비스|플랫폼|제품|고객|"
            r"사업자|사업\s*참여|업무협약",
            value,
            flags=re.IGNORECASE,
        )
    )


def _issue_fact_lines(integrated_issue: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.extend(_string_list(integrated_issue.get("fact_summary"), max_items=12))
    lines.extend(_integrated_text_anchor_lines(integrated_issue, max_items=10))
    for _, fact_text in _fact_texts(integrated_issue):
        if fact_text:
            lines.append(fact_text)
    for key in ("headline", "one_line_summary", "main_event"):
        value = str(integrated_issue.get(key) or "").strip()
        if value:
            lines.append(value)
    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = _anchor_norm(line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(line)
    return result


def _strong_signal_is_only_generic_execution_word(integrated_issue: dict[str, Any]) -> bool:
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding.strip():
        return False
    direct_pattern = (
        r"계약\s*체결|공급\s*계약|추가\s*수주|수주|업무협약|실시협약|주주간\s*계약|"
        r"\bMOU\b|협약|제휴|파트너십|공동\s*추진|출시|공개|실증|PoC|"
        r"현장\s*적용|특정\s*현장\s*적용|고객\s*(적용|도입|확보)|"
        r"운영\s*책임|공급\s*범위|도입\s*범위"
    )
    if re.search(direct_pattern, grounding, flags=re.IGNORECASE):
        return False
    return bool(re.search(r"구축|도입|공급|운영", grounding, flags=re.IGNORECASE))


def _has_business_structure_fact(text: str, integrated_issue: dict[str, Any]) -> bool:
    del integrated_issue
    value = str(text or "")
    has_number = bool(_numeric_token_keys(value))
    structure_pattern = (
        r"매출(?:액|은|이|의|을|를)?|매출\s*(비중|구성|구조)|전체\s*매출|"
        r"사업\s*비중|수익\s*구조|성장률|성장|영업\s*이익|내부\s*거래|내부거래|"
        r"대외\s*(매출|거래|고객)|외부\s*(매출|거래|고객)"
    )
    return has_number and bool(re.search(structure_pattern, value, flags=re.IGNORECASE))


def _technology_event_scale_phrase(values: Sequence[str]) -> str:
    text = " ".join(str(value or "") for value in values)
    country = re.search(r"\d+\s*개국", text)
    company = re.search(r"\d+\s*개사", text)
    if country and company:
        return f"{country.group(0)} {company.group(0)}"
    if company:
        return company.group(0)
    if country:
        return country.group(0)
    return "여러 국가와 기업"


def _technology_event_domain_phrase(values: Sequence[str]) -> str:
    text = " ".join(str(value or "") for value in values)
    domains: list[str] = []
    for label, pattern in (
        ("AI", r"AI|인공지능"),
        ("로봇", r"로봇"),
        ("스마트제조", r"스마트\s*제조"),
        ("디지털 유통·물류", r"디지털\s*유통·물류|물류"),
        ("자동화", r"자동화"),
    ):
        if re.search(pattern, text, flags=re.IGNORECASE):
            domains.append(label)
    domains = _dedupe_keep_order(domains)
    if len(domains) >= 3:
        return "·".join(domains[:4])
    if domains:
        return "·".join(domains)
    return "산업별 기술 적용 분야"


def _technology_event_domain_line_score(value: Any) -> int:
    text = str(value or "")
    score = 0
    for pattern in (
        r"AI|인공지능",
        r"로봇",
        r"스마트\s*제조",
        r"디지털\s*유통·물류|물류",
        r"자동화",
        r"\d+\s*개국",
        r"\d+\s*개사",
    ):
        if re.search(pattern, text, flags=re.IGNORECASE):
            score += 1
    return score


def _technology_event_name_phrase(values: Sequence[str]) -> str:
    text = " ".join(str(value or "") for value in values)
    if re.search(r"스마트테크\s*코리아\s*2026|STK\s*2026", text, flags=re.IGNORECASE):
        return "스마트테크 코리아 2026"
    return "이번 기술 전시"


def _ordered_anchor_matches(text: Any, patterns: Sequence[str]) -> list[str]:
    value = str(text or "")
    matches: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            token = re.sub(r"\s+", " ", match.group(0)).strip()
            if token:
                matches.append(token)
    return _dedupe_keep_order(matches)


def _polish_frontend_ready_screen_copy(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    implication = result.get("implication")
    if not isinstance(implication, dict):
        return result
    frontend_ready = implication.get("frontend_ready")
    if isinstance(frontend_ready, dict):
        for block_key in ("key_implication", "suggested_action"):
            block = frontend_ready.get(block_key)
            if not isinstance(block, dict):
                continue
            sentence = _strip_article_style_lead(block.get("sentence"))
            evidence_sentence = _strip_article_style_lead(block.get("evidence_sentence"))
            if block_key == "suggested_action":
                sentence = _generalize_peer_structure_in_action_sentence(
                    sentence,
                    integrated_issue=integrated_issue,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            sentence = _polish_repeated_frontend_phrase(sentence)
            evidence_sentence = _polish_repeated_frontend_phrase(evidence_sentence)
            block["sentence"] = sentence
            block["evidence_sentence"] = evidence_sentence
    industry_ready = implication.get("industry_frontend_ready")
    if isinstance(industry_ready, dict):
        for item in industry_ready.get("items") or []:
            if not isinstance(item, dict):
                continue
            for block_key in ("key_implication", "suggested_action"):
                block = item.get(block_key)
                if not isinstance(block, dict):
                    continue
                sentence = _strip_article_style_lead(block.get("sentence"))
                evidence_sentence = _strip_article_style_lead(block.get("evidence_sentence"))
                if block_key == "suggested_action":
                    sentence = _generalize_peer_structure_in_action_sentence(
                        sentence,
                        integrated_issue=integrated_issue,
                        profile_linkage_evaluation=profile_linkage_evaluation,
                    )
                sentence = _polish_repeated_frontend_phrase(sentence)
                evidence_sentence = _polish_repeated_frontend_phrase(evidence_sentence)
                block["sentence"] = sentence
                block["evidence_sentence"] = evidence_sentence
    return result


def _polish_repeated_frontend_phrase(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    previous = None
    repeated_pair_pattern = re.compile(
        r"([가-힣A-Za-z0-9]+(?:\s+[가-힣A-Za-z0-9]+)?·"
        r"[가-힣A-Za-z0-9]+(?:\s+[가-힣A-Za-z0-9]+)?)(?:·\1)+"
    )
    while previous != text:
        previous = text
        text = repeated_pair_pattern.sub(r"\1", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text.strip()


def _prefer_collective_actor_anchors(anchors: Sequence[str]) -> list[str]:
    cleaned = _dedupe_keep_order(anchors)
    has_collective = any(
        re.search(r"국내\s*(?:주요\s*)?(?:기업|그룹)|복수\s*기업", anchor) for anchor in cleaned
    )
    if not has_collective:
        return cleaned
    filtered: list[str] = []
    for anchor in cleaned:
        if re.fullmatch(r"[가-힣A-Za-z0-9&._-]+그룹", anchor) and not re.search(
            r"국내",
            anchor,
        ):
            continue
        filtered.append(anchor)
    return filtered


def _role_interpretation_hints(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    main_company = str(integrated_issue.get("main_company") or "").strip()
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    activity_types = _string_list(intelligence.get("activity_types"), max_items=10)
    customers_or_industries = _string_list(
        intelligence.get("customers_or_industries"), max_items=20
    )
    products_or_services = _string_list(intelligence.get("products_or_services"), max_items=20)
    main_tokens = _company_token_variants(main_company)
    target_in_customer_slot = any(
        _normalize_entity_token(item) in main_tokens for item in customers_or_industries
    )
    contract_like = any(
        str(activity or "").strip().casefold() in {"contract", "order", "supply_contract"}
        for activity in activity_types
    ) or bool(re.search(r"계약|수주|공급계약", _integrated_grounding_text(integrated_issue)))

    guidance: list[str] = []
    if target_in_customer_slot and contract_like:
        guidance.append(
            "target_peer_appears_as_contract_counterparty_or_customer; "
            "do_not_treat_supplier_revenue_ratio_as_target_peer_performance"
        )
        guidance.append(
            "if target role is unclear, describe business connection/contract scope rather than "
            "supplier capability or procurement ownership"
        )
    elif contract_like:
        guidance.append(
            "contract_like_event; preserve supplier/counterparty role from evidence and avoid "
            "unstated customer/procurement assumptions"
        )

    return {
        "target_company": main_company,
        "activity_types": activity_types,
        "target_in_customers_or_industries": target_in_customer_slot,
        "customers_or_industries": customers_or_industries,
        "products_or_services": products_or_services,
        "guidance": guidance,
    }


def _relevant_context_items(
    value: Any,
    *,
    relevance_tokens: set[str],
    max_items: int,
) -> list[Any]:
    if not isinstance(value, list):
        return []
    if not relevance_tokens:
        return value[:max_items]
    scored: list[tuple[int, int, Any]] = []
    for index, item in enumerate(value):
        score = _profile_relevance_score(item, relevance_tokens)
        if score > 0:
            scored.append((score, -index, item))
    scored.sort(reverse=True)
    return [item for _, _, item in scored[:max_items]]


def _semantic_fingerprint_for_text(
    text: str,
    issue_context: dict[str, Any] | None = None,
) -> set[str]:
    return {
        str(term.get("normalized") or "")
        for term in _extract_ranked_terms(text, "output_text", issue_context)
        if float(term.get("weight") or 0.0) >= 0.45
        and not _is_generic_business_term(str(term.get("normalized") or ""))
    }


def _issue_evidence_terms_for_action_plan(integrated_issue: dict[str, Any]) -> list[str]:
    if not isinstance(integrated_issue, dict):
        return []
    parts = [
        str(integrated_issue.get("headline") or ""),
        str(integrated_issue.get("main_event") or ""),
        str(integrated_issue.get("main_issue") or ""),
        str(integrated_issue.get("one_line_summary") or ""),
    ]
    parts.extend(str(item or "") for item in integrated_issue.get("fact_summary") or [])
    for _, fact_text in _fact_texts(integrated_issue):
        parts.append(fact_text)
    ranked = _extract_ranked_terms(
        "\n".join(part for part in parts if part),
        "issue_evidence",
        {
            "structured_terms": _issue_structured_terms(
                integrated_issue=integrated_issue,
                classification={},
            )
        },
    )
    terms: list[str] = []
    seen: set[str] = set()
    for item in ranked:
        if str(item.get("term_type") or "") in {"source_noise", "function_word_or_ending"}:
            continue
        normalized = str(item.get("normalized") or item.get("term") or "").strip()
        if not normalized or _is_low_signal_content_token(normalized):
            continue
        if normalized in seen:
            continue
        terms.append(normalized)
        seen.add(normalized)
        if len(terms) >= 40:
            break
    return terms


def _should_include_financial_profile_context(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    bundle: dict[str, Any],
) -> bool:
    """Gate profile-level financial/IR context separately from current-event numbers."""
    source_parts = [
        integrated_issue.get("issue_source_type"),
        integrated_issue.get("source_type"),
        bundle.get("source_type"),
        (bundle.get("metadata") or {}).get("source_type") if isinstance(bundle, dict) else None,
    ]
    source_text = " ".join(str(item or "").strip().lower() for item in source_parts)
    if re.search(r"\b(dart|ir|securities_report|securities|financial_report)\b", source_text):
        return True

    event_parts = [
        integrated_issue.get("cluster_event_type"),
        classification.get("event_type"),
        bundle.get("event_type"),
        _json_dumps(classification.get("event_type_scores") or {}),
    ]
    event_text = " ".join(str(item or "").strip().lower() for item in event_parts)
    if re.search(
        r"(실적|재무|매출\s*변화|재무\s*지표|공급\s*계약|공급계약|투자|지분|"
        r"earnings|financial|revenue_change|financial_metric|supply_contract|investment)",
        event_text,
    ):
        return True
    return False


def _quality_checked_texts(result: dict[str, Any]) -> list[tuple[str, str]]:
    analysis = result.get("analysis") or {}
    implication = result.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    items: list[tuple[str, Any]] = [
        ("analysis.analysis_summary", analysis.get("analysis_summary")),
        ("analysis.market_signal", analysis.get("market_signal")),
        ("analysis.impact_reason", analysis.get("impact_reason")),
        ("analysis.reason", analysis.get("reason")),
        ("peer_implication.peer_meaning", peer.get("peer_meaning")),
        ("peer_implication.capability_change", peer.get("capability_change")),
        ("skax_implication.why_important", skax.get("why_important")),
        ("skax_implication.potential_impact", skax.get("potential_impact")),
    ]
    for index, value in enumerate(_string_list(analysis.get("strategic_meaning"), max_items=3), 1):
        items.append((f"analysis.strategic_meaning[{index}]", value))
    for field in ("opportunities", "threats", "recommended_actions"):
        for index, value in enumerate(_string_list(skax.get(field), max_items=3), 1):
            items.append((f"skax_implication.{field}[{index}]", value))
    for field in ("follow_up_questions", "watch_points"):
        for index, value in enumerate(_string_list(implication.get(field), max_items=3), 1):
            items.append((f"implication.{field}[{index}]", value))
    frontend_ready = implication.get("frontend_ready") or {}
    if isinstance(frontend_ready, dict):
        key_implication = frontend_ready.get("key_implication") or {}
        suggested_action = frontend_ready.get("suggested_action") or {}
        if isinstance(key_implication, dict):
            items.extend(
                [
                    (
                        "frontend_ready.key_implication.sentence",
                        key_implication.get("sentence"),
                    ),
                    (
                        "frontend_ready.key_implication.evidence_sentence",
                        key_implication.get("evidence_sentence"),
                    ),
                ]
            )
        if isinstance(suggested_action, dict):
            items.extend(
                [
                    (
                        "frontend_ready.suggested_action.sentence",
                        suggested_action.get("sentence"),
                    ),
                    (
                        "frontend_ready.suggested_action.evidence_sentence",
                        suggested_action.get("evidence_sentence"),
                    ),
                ]
            )
    return [(label, str(value or "").strip()) for label, value in items if str(value or "").strip()]


def _frontend_ready_insight_evidence_action_language_violation(
    evidence_sentence: Any,
) -> str:
    evidence = str(evidence_sentence or "").strip()
    if not evidence:
        return ""
    if _mentions_skax_actor(evidence):
        return "시사점 근거/설명에 SK AX 대응 관점이 섞였습니다."
    directive_pattern = (
        r"(관찰|점검|검토|대응|모니터링|확인|비교|구분|보완|정리)"
        r"(?:할\s*필요|해야|해야\s*한다|해야\s*합니다|해야\s*함|할\s*수\s*있|"
        r"하는\s*것이\s*필요|필요가\s*있|필요합니다)"
        r"|내부\s*(?:검토|점검|대응|모니터링)[가-힣\s]*(?:필요|해야)"
    )
    if re.search(directive_pattern, evidence):
        return (
            "시사점 근거/설명에 대응방향성 지시문이 섞였습니다. "
            "근거/설명은 기사 사실이 왜 시사점 결론을 뒷받침하는지만 설명해야 합니다."
        )
    return ""


def _workflow_execution_business_terms(text: Any) -> set[str]:
    value = str(text or "")
    terms: set[str] = set()
    for pattern in (
        r"업무\s*(자동화|시스템|처리|범위)",
        r"사내\s*업무\s*시스템",
        r"메일|ERP|데이터베이스|문서",
        r"사용자\s*PC",
        r"자연어\s*명령",
        r"필요한\s*업무를\s*대신\s*처리",
        r"데스크톱\s*에이전틱\s*AI",
        r"처리\s*범위|적용\s*(업무|대상|범위)",
    ):
        if re.search(pattern, value, flags=re.IGNORECASE):
            terms.add(pattern)
    return terms


def _is_financial_or_transaction_frontend_block(
    block: dict[str, Any],
    integrated_issue: dict[str, Any],
) -> bool:
    claim_type = str(block.get("claim_type") or "").strip()
    if claim_type in {"financial_structure_signal", "governance_exposure_signal"}:
        return True
    event_type = str(
        integrated_issue.get("cluster_event_type") or integrated_issue.get("event_type") or ""
    ).casefold()
    if event_type in {"financial_update", "performance", "earnings", "governance"}:
        return True
    grounding = _integrated_grounding_text(integrated_issue)
    return bool(_financial_structure_terms(grounding) and _numeric_token_keys(grounding))


def _numeric_token_keys(text: Any) -> set[str]:
    return {
        key
        for token in re.findall(
            r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|usd|krw|대|개)?",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        if (key := _numeric_token_key(token))
    }


def _skax_action_mode_from_profile_linkage(
    profile_linkage_evaluation: dict[str, Any] | None,
) -> str:
    linkage = _profile_linkage_for_company(
        profile_linkage_evaluation or {},
        company_id="sk_ax",
        scope="skax",
    )
    level = str(linkage.get("linkage_level") or "none")
    matched_areas = _jsonish_list(linkage.get("matched_business_areas"))
    matched_terms = _string_list(linkage.get("matched_terms"), max_items=12)
    matched_capabilities = _string_list(linkage.get("matched_capabilities"), max_items=12)
    specific_levels = {
        str(area.get("specificity_level") or "").strip()
        for area in matched_areas
        if isinstance(area, dict)
    }
    has_specific_area = bool(
        specific_levels & {"product_or_service", "core_capability", "business_area"}
    )
    has_area_evidence = any(
        isinstance(area, dict)
        and (
            _string_list(area.get("matched_issue_terms"), max_items=5)
            or _string_list(area.get("matched_capabilities"), max_items=5)
            or _string_list(area.get("matched_products_or_services"), max_items=5)
            or str(area.get("evidence_text") or "").strip()
            or str(area.get("why_relevant_to_issue") or "").strip()
        )
        for area in matched_areas
    )
    if level in {"high", "medium"} and has_specific_area and has_area_evidence:
        return "direct_business_match"
    if level in {"high", "medium", "low"} and (
        matched_terms or matched_capabilities or matched_areas or str(linkage.get("reason") or "")
    ):
        return "adjacent_opportunity_probe"
    return "watch_or_monitor"


def _peer_only_issue_product_terms(
    integrated_issue: dict[str, Any],
    *,
    profile_linkage_evaluation: dict[str, Any] | None,
) -> list[str]:
    issue_terms = _issue_specific_product_terms_for_action(integrated_issue)
    if not issue_terms:
        return []
    skax_terms = _skax_profile_product_terms(profile_linkage_evaluation)
    out: list[str] = []
    for term in issue_terms:
        if any(_anchor_norm(term) == _anchor_norm(skax_term) for skax_term in skax_terms):
            continue
        out.append(term)
    return out[:8]


def _skax_profile_product_terms(
    profile_linkage_evaluation: dict[str, Any] | None,
) -> list[str]:
    linkage = _profile_linkage_for_company(
        profile_linkage_evaluation or {},
        company_id="sk_ax",
        scope="skax",
    )
    values: list[str] = []
    values.extend(_string_list(linkage.get("matched_terms"), max_items=20))
    values.extend(_string_list(linkage.get("matched_capabilities"), max_items=20))
    for area in _jsonish_list(linkage.get("matched_business_areas")):
        if not isinstance(area, dict):
            continue
        values.extend(_string_list(area.get("matched_products_or_services"), max_items=20))
        values.extend(_string_list(area.get("products_or_services"), max_items=20))
        values.extend(_string_list(area.get("matched_capabilities"), max_items=20))
    return values


def _frontend_ready_role_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in _anchor_tokens(text):
        norm = _frontend_ready_role_term_norm(token)
        if len(norm) < 2 or norm in _FRONTEND_READY_GENERIC_ROLE_TERMS:
            continue
        if re.fullmatch(r"[0-9.,]+", norm):
            continue
        terms.add(norm)
    return terms


def _frontend_ready_role_bigrams(text: str) -> set[str]:
    tokens: list[str] = []
    for token in _anchor_tokens(text):
        norm = _frontend_ready_role_term_norm(token)
        if len(norm) < 2 or norm in _FRONTEND_READY_GENERIC_ROLE_TERMS:
            continue
        tokens.append(norm)
    return {f"{tokens[index]}::{tokens[index + 1]}" for index in range(len(tokens) - 1)}


def _regex_slot_terms(text: str, pattern: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", str(item or "")).strip(" ,.;:()[]")
        for item in re.findall(pattern, text, flags=re.IGNORECASE)
        if str(item or "").strip()
    ]


def _nested_mapping_keys(value: Any, *, max_keys: int = 200) -> set[str]:
    keys: set[str] = set()

    def collect(item: Any) -> None:
        if len(keys) >= max_keys:
            return
        if isinstance(item, dict):
            for key, nested in item.items():
                cleaned = re.sub(r"\s+", " ", str(key or "")).strip()
                if cleaned:
                    keys.add(cleaned)
                collect(nested)
                if len(keys) >= max_keys:
                    return
        elif isinstance(item, list | tuple | set):
            for nested in item:
                collect(nested)
                if len(keys) >= max_keys:
                    return

    collect(value)
    return keys


def _looks_like_sentence_slot(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if len(value) > 45:
        return True
    if len(value.split()) >= 7:
        return True
    return bool(re.search(r"(?:다|했다|한다|된다|있다|예정이다|계획이다)[.!?]?$", value))


def _evidence_sentence_has_dynamic_grounding(
    text: str,
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> bool:
    evidence_tokens = _distinct_anchor_tokens(text)
    if not evidence_tokens:
        return False
    grounding_text = " ".join(
        [
            _integrated_grounding_text(integrated_issue),
            _json_dumps(_profile_for_prompt(profile_context, integrated_issue=integrated_issue)),
        ]
    )
    grounding_norm = _anchor_norm(grounding_text)
    matched = [token for token in evidence_tokens if _anchor_norm(token) in grounding_norm]
    if any(len(_anchor_norm(token)) >= 5 for token in matched):
        return True
    return len(matched) >= 2


def _mentions_skax_actor(text: str) -> bool:
    return bool(re.search(r"SK\s*AX|자사|우리\s*회사", str(text or ""), flags=re.IGNORECASE))


def _action_plan_terms_for_keys(
    action_artifact_plan: dict[str, Any],
    *,
    keys: Sequence[str],
) -> set[str]:
    signals = _json_dict(action_artifact_plan.get("current_issue_signals"))
    terms: set[str] = set()
    for key in keys:
        values = (
            _string_list(signals.get(key), max_items=50)
            if key != "event_type"
            else [str(signals.get(key) or "")]
        )
        for value in values:
            for token in _raw_normalized_terms(value):
                normalized = token.casefold() if token.isascii() else token
                if normalized and not _is_low_signal_content_token(normalized):
                    if _is_generic_business_term(normalized):
                        continue
                    if len(normalized) < 2:
                        continue
                    terms.add(normalized)
    return terms


def _action_text_has_issue_signal(text: str, issue_terms: set[str]) -> bool:
    output_terms = _content_tokens(text)
    if output_terms & issue_terms:
        return True
    output_axis_terms = {
        term
        for term in _raw_normalized_terms(text)
        if len(term) >= 2
        and not _is_low_signal_content_token(term)
        and not _is_generic_business_term(term)
    }
    issue_axis_terms: set[str] = set()
    for issue_term in issue_terms:
        issue_axis_terms.update(
            term
            for term in _raw_normalized_terms(issue_term)
            if len(term) >= 2
            and not _is_low_signal_content_token(term)
            and not _is_generic_business_term(term)
        )
    if output_axis_terms & issue_axis_terms:
        return True
    return any(
        _tokens_semantically_close(output_term, issue_term)
        for output_term in output_terms
        for issue_term in issue_terms
    )


def _action_text_has_internal_strategy_checkpoint(text: str) -> bool:
    value = str(text or "")
    tokens = [
        token
        for token in _content_tokens(value)
        if len(token) >= 2 and not _is_generic_business_term(token)
    ]
    return _mentions_skax_actor(value) and len(set(tokens)) >= 3


def _action_text_has_skax_change(text: str) -> bool:
    value = str(text or "")
    return _mentions_skax_actor(value) and _sentence_count(value) >= 1


def _high_signal_tokens_for_repetition(text: str) -> set[str]:
    return {token for token in _semantic_fingerprint_for_text(text) if len(token) >= 3}


def _relationship_only_uncertain(text: str) -> bool:
    relation_sentences = [
        sentence for sentence in _split_sentences(text) if _RELATIONSHIP_PATTERN.search(sentence)
    ]
    return bool(relation_sentences) and all(
        _UNCERTAINTY_PATTERN.search(sentence) for sentence in relation_sentences
    )


def _supplier_role_overstatement_violation(
    text: str,
    *,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not _SUPPLIER_CAPABILITY_PATTERN.search(text):
        return ""
    if not _main_company_is_customer_or_buyer(integrated_issue):
        return ""
    return "타깃 피어가 계약의 고객/도입/조달 주체로 보이는데 공급자 역량처럼 표현했습니다."


def _supplier_financial_focus_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not _main_company_is_customer_or_buyer(integrated_issue):
        return ""
    if label.startswith(("implication.follow_up_questions", "implication.watch_points")):
        return ""
    supplier_names = _supplier_names_for_target_counterparty(integrated_issue)
    if not supplier_names:
        return ""
    supplier_mentioned = any(
        re.search(re.escape(name), text, flags=re.IGNORECASE) for name in supplier_names
    )
    if not supplier_mentioned:
        return ""
    if not re.search(
        r"성장|시장\s*입지|중요한\s*매출원|성과|시장\s*반응|"
        r"매출\s*(기여|확대|성장|영향)|매출.{0,16}영향",
        text,
    ):
        return ""
    return (
        "타깃 피어가 계약 상대방으로 보이는데 공급사 재무/성장 논리를 "
        "피어 전략 의미처럼 사용했습니다."
    )


def _main_issue_context_text(integrated_issue: dict[str, Any]) -> str:
    parts: list[str] = [
        str(integrated_issue.get(key) or "")
        for key in ("headline", "main_event", "main_issue", "one_line_summary")
    ]
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for item in [
            *(intelligence.get("common_facts") or []),
            *(intelligence.get("unique_facts") or []),
        ]:
            if (
                isinstance(item, dict)
                and _fact_has_summary_role(item, "main_event")
                and not _fact_has_summary_role(item, "application_case")
            ):
                parts.append(str(item.get("fact") or ""))
                parts.extend(
                    str(value or "") for value in _jsonish_list(item.get("products_or_services"))
                )
    parts.extend(_string_list(integrated_issue.get("fact_summary"), max_items=5))
    return re.sub(r"\s+", " ", " ".join(parts))


def _non_main_event_product_terms(integrated_issue: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if not isinstance(intelligence, dict):
        return terms
    for item in [
        *(intelligence.get("common_facts") or []),
        *(intelligence.get("unique_facts") or []),
    ]:
        if not isinstance(item, dict):
            continue
        if _fact_has_summary_role(item, "main_event") and not _fact_has_summary_role(
            item,
            "application_case",
        ):
            continue
        for value in _jsonish_list(item.get("products_or_services")):
            term = re.sub(r"\s+", " ", str(value or "").strip(" ."))
            if term:
                terms.append(term)
        terms.extend(_quoted_entity_terms(str(item.get("fact") or "")))
        for evidence in _jsonish_list(item.get("evidence_texts"))[:3]:
            terms.extend(_quoted_entity_terms(str(evidence or "")))
    return list(dict.fromkeys(terms))


def _quoted_entity_terms(text: str) -> list[str]:
    value = str(text or "")
    if not value:
        return []
    terms: list[str] = []
    for match in re.finditer(r"['‘’\"“”]([^'‘’\"“”]{2,50})['‘’\"“”]", value):
        term = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        if term:
            terms.append(term)
    return terms


def _scope_effect_claim(text: str) -> bool:
    return bool(
        re.search(
            r"강화|확장|확대|영향|기회|성장|진출|입지|레퍼런스|사업\s*영역|서비스",
            str(text or ""),
        )
    )


def _high_signal_issue_overlap_count(text: str, integrated_issue: dict[str, Any]) -> int:
    issue_tokens = _issue_relevance_tokens(integrated_issue)
    text_tokens = _semantic_fingerprint_for_text(str(text or ""))
    overlap = {
        token
        for token in issue_tokens & text_tokens
        if len(token) >= 2 and not _is_generic_business_term(token)
    }
    return len(overlap)


def _concrete_profile_terms(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> set[str]:
    issue_context = _profile_linkage_issue_context(
        integrated_issue=integrated_issue,
        classification={},
    )
    if scope == "skax":
        profiles = [profile_context.get("skax_profile") or {}]
    else:
        peer_profiles = profile_context.get("peer_profiles") or {}
        profiles = []
        if isinstance(peer_profiles, dict):
            for company_id in _companies_from_integrated_issue(integrated_issue):
                profile = peer_profiles.get(company_id) or {}
                if isinstance(profile, dict):
                    profiles.append(profile)
    terms: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        for entry in _profile_entries_for_linkage(profile):
            for term in _extract_ranked_terms(
                str(entry.get("text") or ""),
                "profile_business_area",
                issue_context,
            ):
                normalized = str(term.get("normalized") or "")
                if float(term.get("weight") or 0.0) >= 0.5:
                    terms.add(normalized)
    company_identity_terms = _company_identity_terms(integrated_issue)
    return {
        term
        for term in terms
        if term not in company_identity_terms
        and not _looks_like_source_noise(term, issue_context)
        and not _looks_like_korean_function_word_or_ending(term)
        and len(term) >= 3
        and not re.fullmatch(r"\d+", term)
    }


def _scope_term_supported(term: str, *, evidence_terms: set[str], evidence_text: str) -> bool:
    if term in evidence_terms or term.upper() in evidence_terms:
        return True
    normalized_evidence = str(evidence_text or "").casefold()
    normalized_term = str(term or "").casefold()
    if normalized_term and normalized_term in normalized_evidence:
        return True
    aliases = {
        "금융": ("금융", "금융권", "금융기관"),
        "IT": ("IT", "아이티"),
        "인프라": ("인프라", "시스템"),
        "AI": ("AI", "인공지능", "에이아이"),
    }
    for alias in aliases.get(term.upper(), aliases.get(term, ())):
        if str(alias).casefold() in normalized_evidence:
            return True
    return False


def _evidence_scope_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·/_-]{1,}", str(text or "")):
        cleaned = token.strip(".,;:()[]{}'\"")
        upper = cleaned.upper()
        if upper in {"AI", "IT", "DX", "AX", "UI", "UX", "SI", "MSP", "ERP", "CRM"}:
            terms.add(upper)
            continue
        normalized = _normalize_content_token(cleaned)
        if normalized and not _is_low_signal_content_token(normalized):
            terms.add(normalized.casefold())
    return terms


def _profile_has_execution_case(
    profile_context: dict[str, Any] | None,
    *,
    integrated_issue: dict[str, Any] | None,
) -> bool:
    if not isinstance(profile_context, dict):
        return False
    profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue or {})
    for item in _iter_dicts(profile):
        for key, value in item.items():
            if str(key).casefold() in {"execution_cases", "execution_case", "case_studies"}:
                if value not in ({}, [], "", None):
                    return True
    profile_text = _json_dumps(profile)
    return bool(re.search(r"성공\s*사례|구축\s*사례|레퍼런스\s*사례", profile_text))


def _matching_fact_ids(text: str, fact_entries: list[tuple[str, str]]) -> list[str]:
    tokens = _content_tokens(text)
    matched: list[str] = []
    for fact_id, fact_text in fact_entries:
        if _fact_is_referenced(fact_text, text, tokens):
            matched.append(fact_id)
        if len(matched) >= 5:
            break
    return matched


def _matching_profile_fields(
    text: str,
    *,
    profile_entries: list[dict[str, str]],
    scope: str,
) -> list[str]:
    tokens = _content_tokens(text)
    matched: list[str] = []
    for entry in profile_entries:
        entry_scope = entry.get("scope") or ""
        if scope == "skax" and entry_scope != "skax":
            continue
        if scope == "peer" and entry_scope == "skax":
            continue
        if not _profile_entry_is_referenced(entry.get("text", ""), text, tokens):
            continue
        path = entry.get("path") or ""
        if path and path not in matched:
            matched.append(path)
        if len(matched) >= 5:
            break
    return matched


def _profile_entry_text(value: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "name",
        "business_area",
        "summary",
        "recent_direction",
        "core_capabilities",
        "capabilities",
        "change_type",
        "overall_change",
    ):
        if key not in value:
            continue
        raw = value.get(key)
        if isinstance(raw, list):
            parts.extend(str(item or "") for item in raw)
        elif isinstance(raw, dict):
            parts.append(_json_dumps(raw))
        else:
            parts.append(str(raw or ""))
    return " ".join(part.strip() for part in parts if part and part.strip())


def _profile_entry_is_referenced(
    entry_text: str,
    output_text: str,
    output_tokens: set[str],
) -> bool:
    tokens = _content_tokens(entry_text)
    if len(tokens & output_tokens) >= 2:
        return True
    for token in tokens:
        if len(token) >= 4 and token in output_text:
            return True
    return False


def _has_relevant_peer_profile_linkage(
    profile_linkage_evaluation: dict[str, Any] | None,
    *,
    integrated_issue: dict[str, Any],
) -> bool:
    if not isinstance(profile_linkage_evaluation, dict):
        return False
    for company_id in _companies_from_integrated_issue(integrated_issue):
        linkage = _profile_linkage_for_company(
            profile_linkage_evaluation,
            company_id=company_id,
            scope="peer",
        )
        if not isinstance(linkage, dict):
            continue
        if _linkage_rank(str(linkage.get("linkage_level") or "none")) < _linkage_rank("medium"):
            continue
        if (
            _string_list(linkage.get("matched_terms"), max_items=8)
            or _string_list(linkage.get("matched_capabilities"), max_items=8)
            or _jsonish_list(linkage.get("matched_business_areas"))
            or str(linkage.get("reason") or "").strip()
        ):
            return True
    return False


def _peer_profile_linkage(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    company_id: str,
) -> dict[str, Any]:
    evaluation = _build_profile_linkage_evaluation(
        integrated_issue=integrated_issue,
        classification={},
        profile_context=profile_context,
    )
    linkage = _profile_linkage_for_company(
        evaluation,
        company_id=company_id,
        scope="peer",
    )
    return {
        "company": company_id,
        "matched_profile_terms": _string_list(linkage.get("matched_terms"), max_items=12),
        "linkage_level": str(linkage.get("linkage_level") or "none"),
        "business_novelty_status": str(linkage.get("business_novelty_status") or ""),
        "implication_mode": str(linkage.get("implication_mode") or ""),
        "reason": str(linkage.get("reason") or "현재 이슈와 비교할 피어 프로필 본문이 없습니다."),
    }


def _relevant_profile_linkage_level(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> str:
    evaluation = _build_profile_linkage_evaluation(
        integrated_issue=integrated_issue,
        classification={},
        profile_context=profile_context,
    )
    if scope == "skax":
        linkage = evaluation.get("skax_linkage") if isinstance(evaluation, dict) else {}
        return str((linkage or {}).get("linkage_level") or "none")
    best = "none"
    for linkage in _jsonish_list(evaluation.get("peer_linkages")):
        if not isinstance(linkage, dict):
            continue
        level = str(linkage.get("linkage_level") or "none")
        if _linkage_rank(level) > _linkage_rank(best):
            best = level
    return best


def _relevant_profile_linkage_level_from_evaluation(
    profile_linkage_evaluation: dict[str, Any] | None,
    *,
    scope: str,
) -> str:
    evaluation = profile_linkage_evaluation or {}
    if not isinstance(evaluation, dict):
        return ""
    if scope == "skax":
        linkage = evaluation.get("skax_linkage") or {}
        if not isinstance(linkage, dict):
            return ""
        return str(linkage.get("linkage_level") or "")
    best = ""
    for linkage in _jsonish_list(evaluation.get("peer_linkages")):
        if not isinstance(linkage, dict):
            continue
        level = str(linkage.get("linkage_level") or "")
        if _linkage_rank(level) > _linkage_rank(best):
            best = level
    return best


def _linkage_level_from_match_count(count: int) -> str:
    if count >= 4:
        return "high"
    if count >= 2:
        return "medium"
    if count >= 1:
        return "low"
    return "none"


def _linkage_rank(level: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(str(level), 0)


def _mentions_profile_based_peer_claim(text: str) -> bool:
    return bool(re.search(r"사업\s*영역|사업영역|역량|프로필|제공|수행|운영|지원", text or ""))


def _issue_subject_phrase(integrated_issue: dict[str, Any]) -> str:
    issue_text = " ".join(
        str(integrated_issue.get(key) or "").strip()
        for key in ("main_event", "main_issue", "headline", "one_line_summary")
    )
    if subject := _extract_issue_subject_from_text(issue_text):
        return subject

    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        main_event_facts = [
            item
            for item in [
                *(intelligence.get("common_facts") or []),
                *(intelligence.get("unique_facts") or []),
            ]
            if isinstance(item, dict) and _fact_has_summary_role(item, "main_event")
        ]
        for item in main_event_facts:
            for value in _jsonish_list(item.get("products_or_services")):
                text = re.sub(r"\s+", " ", str(value or "").strip(" ."))
                if text:
                    return text
        for item in main_event_facts:
            if subject := _extract_issue_subject_from_text(str(item.get("fact") or "")):
                return subject
        for value in _jsonish_list(intelligence.get("products_or_services")):
            text = re.sub(r"\s+", " ", str(value or "").strip(" ."))
            if text:
                return text
        for item in intelligence.get("common_facts") or []:
            if not isinstance(item, dict):
                continue
            if subject := _extract_issue_subject_from_text(str(item.get("fact") or "")):
                return subject
        for item in intelligence.get("unique_facts") or []:
            if not isinstance(item, dict):
                continue
            for value in _jsonish_list(item.get("products_or_services")):
                text = re.sub(r"\s+", " ", str(value or "").strip(" ."))
                if text:
                    return text
            if subject := _extract_issue_subject_from_text(str(item.get("fact") or "")):
                return subject

    return ""


def _main_company_display(integrated_issue: dict[str, Any]) -> str:
    main_company = str(integrated_issue.get("main_company") or "").strip()
    if not main_company:
        return ""
    for alias in expand_peer_aliases(main_company):
        text = str(alias or "").strip()
        if text and not re.fullmatch(r"[a-z0-9_]+", text, flags=re.IGNORECASE):
            return text
    return main_company


def _relevant_profile_named_terms(
    profiles: list[dict[str, Any]],
    *,
    relevance_tokens: set[str],
    integrated_issue: dict[str, Any],
    max_items: int,
) -> list[str]:
    company_terms = _company_identity_terms(integrated_issue)
    candidates: list[str] = []
    for profile in profiles:
        for key in (
            "core_capabilities",
            "strategic_focus",
            "priority_initiatives",
            "key_products_services",
            "recent_changes",
        ):
            for value in _jsonish_list(profile.get(key))[:12]:
                if isinstance(value, dict):
                    text = str(value.get("name") or value.get("summary") or "").strip()
                else:
                    text = str(value or "").strip()
                if not text:
                    continue
                tokens = _content_tokens(text)
                if (
                    tokens
                    and tokens - company_terms
                    and (not relevance_tokens or tokens & relevance_tokens)
                ):
                    candidates.append(text)
    return list(dict.fromkeys(candidates))[:max_items]


def _primary_issue_fact(integrated_issue: dict[str, Any]) -> str:
    for key in ("one_line_summary", "integrated_text", "main_event", "main_issue", "headline"):
        value = str(integrated_issue.get(key) or "").strip()
        if value:
            return _ensure_sentence(value)
    for value in _string_list(integrated_issue.get("fact_summary"), max_items=1):
        if value:
            return _ensure_sentence(value)
    for _, fact_text in _fact_texts(integrated_issue):
        if fact_text:
            return _ensure_sentence(fact_text)
    return "현재 사건에서 확인된 사실이 있습니다."


def _repair_customer_role_overstatement(text: str) -> str:
    sentence = str(text or "").strip()
    replacements = (
        (r"([가-힣A-Za-z0-9&·+_\-\s]+?)의\s*공급\s*역량", r"\1의 계약 범위와 사업영역 접점"),
        (r"([가-힣A-Za-z0-9&·+_\-\s]+?)\s*공급\s*역량", r"\1 계약 범위와 사업영역 접점"),
        (r"전략적\s*방향과\s*일치", "프로필상 사업영역과 연결"),
        (r"전략과의\s*일관성", "프로필상 사업영역과의 접점"),
        (
            r"프로젝트[가은]\s*유사한\s*고객군과\s*사업\s*영역에서의\s*기회를\s*제공합니다",
            "계약 신호는 유사 고객군과 사업 영역에서 참고할 사업영역 접점을 보여줍니다",
        ),
        (r"기회를\s*제공하는\s*것", "참고 근거가 되는 것"),
        (r"기회를\s*제공하는\s*것으로", "참고 근거로"),
        (r"기회를\s*제공할\s*수\s*있습니다", "참고 근거가 될 수 있습니다"),
        (r"기회를\s*제공합니다", "참고 근거가 됩니다"),
        (r"프로젝트에\s*참여하여", "프로젝트와 연결되어"),
        (r"프로젝트에\s*참여", "프로젝트와 연결"),
        (r"사업에\s*참여하여", "사업과 연결되어"),
        (r"사업에\s*참여", "사업과 연결"),
        (r"기여하고\s*있습니다", "사업영역 접점을 보여줍니다"),
        (r"공급\s*역량", "계약 범위와 사업영역 접점"),
        (r"납품\s*역량", "계약 범위와 사업영역 접점"),
        (r"도입[·\s-]*조달\s*주체", "계약 상대방"),
        (r"도입[·\s-]*조달", "계약"),
    )
    for pattern, replacement in replacements:
        sentence = re.sub(pattern, replacement, sentence)
    return sentence


def _profile_comparison_phrase(
    *,
    peer_profile: str,
    skax_profile: str,
    linkage_level: str,
) -> str:
    if linkage_level in {"high", "medium"} and peer_profile and skax_profile:
        return f"피어의 {peer_profile} 접점과 SK AX의 {skax_profile} 접점"
    if skax_profile:
        return f"SK AX의 {skax_profile} 접점"
    if peer_profile:
        return f"피어의 {peer_profile} 접점과 SK AX의 유사 사업 대응 범위"
    return "현재 사건에서 확인된 대상 사업과 SK AX의 유사 사업 대응 범위"


def _known_fact_ids(integrated_issue: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in integrated_issue.get("fact_basis") or []:
        if not isinstance(item, dict):
            continue
        for fact_id in item.get("fact_ids") or []:
            text = str(fact_id or "").strip()
            if text:
                ids.add(text)
        fact_id = str(item.get("fact_id") or "").strip()
        if fact_id:
            ids.add(fact_id)
    for item in integrated_issue.get("consolidated_facts") or []:
        if isinstance(item, dict):
            fact_id = str(item.get("fact_id") or "").strip()
            if fact_id:
                ids.add(fact_id)
    return ids


def _used_context_layers(context: dict[str, Any]) -> list[str]:
    layers: list[str] = []
    if context.get("peer_event_timeline_recent"):
        layers.append("peer_event_timeline_recent")
    if context.get("sector_pulse_recent"):
        layers.append("sector_pulse_recent")
    if context.get("financial_trend"):
        layers.append("financial_trend")
    if context.get("event_chain_candidates"):
        layers.append("event_chain_candidates")
    if context.get("similar_cards_rag"):
        layers.append("similar_cards_rag")
    return layers
