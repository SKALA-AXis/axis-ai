"""strategic_insight constants — extracted from facade (move-only)."""

from __future__ import annotations

import re

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

_IMPACT_LEVELS = {"high", "medium", "low"}


_RISK_OR_OPPORTUNITY = {"risk", "opportunity", "neutral"}


_EVIDENCE_LABELS = {"sufficient", "moderate", "insufficient"}


_UNSUPPORTED_CLAIM_PATTERNS = (
    r"시장\s*점유율\s*확대",
    r"시장\s*점유율[을를\s]*(확보|높|늘)",
    r"점유율[이을가\s]*(확대|상승|증가)",
    r"시장\s*선점",
    r"선점",
    r"기술적\s*우위",
    r"기술적\s*역량[을를\s]*입증",
    r"역량[을를\s]*입증",
    r"성과[가를은\s]*입증",
    r"검증된\s*역량",
    r"격차[가를은\s]*(확대|벌어|커|발생|나타)",
    r"리더십\s*확보",
    r"매출\s*기여",
    r"시장\s*점유율\s*감소",
    r"점유율[이을가\s]*(감소|하락|축소)",
)


_RELATIONSHIP_PATTERN = re.compile(
    r"협업|협력|파트너십|제휴|MOU|얼라이언스|컨소시엄|"
    r"공동\s*(추진|개발|연구|사업|운영|구축|참여|투자|검증)",
    re.IGNORECASE,
)


_RELATIONSHIP_ACTIVITY_TYPES = {"partnership", "collaboration", "alliance", "joint", "mou"}


_UNCERTAINTY_PATTERN = re.compile(r"검토|가능성|구상|계획|예정|모색|논의|추진\s*(중|예정|계획)")


_SUPPLIER_CAPABILITY_PATTERN = re.compile(
    r"(공급|납품)\s*역량|공급\s*계약.{0,30}(제공|수행)\s*역량"
)


_NUMERIC_TOKEN_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|usd|krw)?",
    re.IGNORECASE,
)


_INTERNAL_CHECKPOINT_GROUPS: dict[str, str] = {}


_SKAX_ACTION_VERB_GROUPS: dict[str, str] = {}


_DOMAIN_ALIASES: dict[str, set[str]] = {}


OVERCLAIM_PATTERNS: dict[str, tuple[str, ...]] = {
    "counterparty": (
        r"신규\s*사업",
        r"사업\s*확장",
        r"영역\s*확장",
        r"입지\s*강화",
        r"역량\s*강화",
    ),
    "new_signal": (
        r"성과[가를은\s]*입증",
        r"역량[을를\s]*(강화|입증|검증)",
        r"검증된\s*역량",
        r"경쟁력[을를\s]*강화",
        r"입지\s*강화",
        r"사업\s*확장",
    ),
}


_FRONTEND_READY_SOURCES = {
    "llm_direct",
    "repair_direct",
    "frontend_repair_direct",
    "schema_repair_direct",
    "report_copy_repair_direct",
    "counterparty_repair_direct",
    "action_repair_direct",
    "derived_from_implication",
    "composer_editorial",
    "legacy_fallback",
}


_FRONTEND_READY_DISPLAY_SOURCES = {"llm_direct", "frontend_repair_direct"}


_FRONTEND_READY_CLAIM_TYPES = {
    "event_based_signal",
    "profile_based_signal",
    "financial_structure_signal",
    "governance_exposure_signal",
    "self_or_market_signal",
    "market_adoption_signal",
    "market_leadership",
    "capability_improvement",
    "performance_improvement",
    "operational_shift",
    "workflow_execution_signal",
    "internal_strategy_check",
}


_FRONTEND_READY_CLAIM_STRENGTHS = {"strong", "moderate", "cautious"}


_FRONTEND_READY_EVIDENCE_MODES = {
    "profile_based",
    "event_based",
    "generic_monitoring",
}


_FRONTEND_READY_STRONG_CLAIM_TYPES = {
    "market_leadership",
    "capability_improvement",
    "performance_improvement",
}


_FRONTEND_READY_GENERIC_ROLE_TERMS = {
    "이번",
    "해당",
    "현재",
    "사건",
    "신호",
    "시장",
    "피어",
    "피어사",
    "기업",
    "사업",
    "서비스",
    "기반",
    "관련",
    "흐름",
    "관점",
    "의미",
    "결론",
    "근거",
    "설명",
    "필요",
    "해야",
    "합니다",
    "한다",
    "보여",
    "가능",
    "점검",
    "검토",
    "확인",
    "모니터링",
    "강화",
    "경쟁력",
    "신호입니다",
    "sk",
    "ax",
    "skax",
}
