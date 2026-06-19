"""strategic_insight models_config — extracted from facade (move-only)."""

from __future__ import annotations

import os
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

_DEFAULT_LLM_MODEL = "gpt-4o"


_DEFAULT_FRONTEND_READY_MODEL = "gpt-5.5"


_LLM_MODEL = os.getenv("STRATEGIC_INSIGHT_MODEL", _DEFAULT_LLM_MODEL)


_FRONTEND_READY_MODEL = os.getenv(
    "FRONTEND_READY_MODEL",
    _DEFAULT_FRONTEND_READY_MODEL,
)


_FRONTEND_READY_REPAIR_MODEL = os.getenv(
    "FRONTEND_READY_REPAIR_MODEL",
    _FRONTEND_READY_MODEL,
)


_SELF_REVIEW_MODEL_RAW = os.getenv("SELF_REVIEW_MODEL", _LLM_MODEL)


_SELF_REVIEW_DISABLED_VALUES = {"", "0", "false", "off", "none", "disabled"}


_SELF_REVIEW_DISABLED = (
    str(_SELF_REVIEW_MODEL_RAW or "").strip().casefold() in _SELF_REVIEW_DISABLED_VALUES
)


_SELF_REVIEW_MODEL = "" if _SELF_REVIEW_DISABLED else _SELF_REVIEW_MODEL_RAW


_PROMPT_VERSION = "strategic-insight-v1.61-llm-structured-reasoning"


_LLM_TEMPERATURE = 0.0


_LLM_MAX_COMPLETION_TOKENS = 5000


_LLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("STRATEGIC_INSIGHT_LLM_TIMEOUT_SECONDS", "120"))


def _llm_model_for_phase(phase: str) -> str:
    phase_name = str(phase or "").strip()
    if phase_name.startswith(("frontend_ready_repair", "quality_repair")):
        return _FRONTEND_READY_REPAIR_MODEL
    if phase_name.startswith("frontend_ready_generate"):
        return _FRONTEND_READY_MODEL
    if phase_name == "self_review" and _SELF_REVIEW_MODEL:
        return _SELF_REVIEW_MODEL
    return _LLM_MODEL


def _llm_model_config_diagnostics() -> dict[str, Any]:
    return {
        "strategic_insight_model": _LLM_MODEL,
        "frontend_ready_model": _FRONTEND_READY_MODEL,
        "frontend_ready_repair_model": _FRONTEND_READY_REPAIR_MODEL,
        "self_review_model": _SELF_REVIEW_MODEL or None,
        "self_review_disabled": _SELF_REVIEW_DISABLED,
    }
