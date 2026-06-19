"""strategic_insight text_predicates — extracted from facade (move-only)."""

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


def _natural_join(values: Sequence[str]) -> str:
    items = [str(item or "").strip() for item in values if str(item or "").strip()]
    if len(items) <= 1:
        return items[0] if items else ""
    return " 및 ".join(items)


def _short_fact_clause(value: Any, *, max_chars: int = 92) -> str:
    text = _strip_article_style_lead(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    text = re.sub(r"(?:다|요)\.\s*$", "", text)
    if len(text) <= max_chars:
        return text + "는 점에서,"
    shortened = text[:max_chars].rstrip(" ,.;:·ㆍ")
    return shortened + " 등이 제시되며,"


def _strip_article_style_lead(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(
        r"^(?:\d{1,2}일\s*)?(?:업계|회사|관계자|외신|언론|공시|발표|보도)에\s*따르면\s*,?\s*",
        "",
        text,
    )
    text = re.sub(
        r"^(?:[가-힣A-Za-z0-9&._ -]+은|[가-힣A-Za-z0-9&._ -]+는)\s*"
        r"(?:\d{1,2}일\s*)?(?:밝혔다|전했다|설명했다|발표했다)[,.]?\s*",
        "",
        text,
    )
    return text.strip()


def _with_korean_object_particle(phrase: Any) -> str:
    value = str(phrase or "").strip()
    if not value:
        return ""
    return f"{value}{'을' if _has_korean_final_consonant(value) else '를'}"


def _has_korean_final_consonant(value: str) -> bool:
    for char in reversed(str(value or "").strip()):
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            return (code - 0xAC00) % 28 != 0
        if char.isalnum():
            return True
    return False


def _dedupe_keep_order(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        norm = _anchor_norm(text)
        if not text or not norm or norm in seen:
            continue
        result.append(text)
        seen.add(norm)
    return result


def _text_has_anchor_term(text: str, terms: Sequence[str]) -> bool:
    text_norm = _anchor_norm(text)
    if not text_norm:
        return False
    for term in terms:
        term_norm = _anchor_norm(term)
        if len(term_norm) < 2:
            continue
        if term_norm in text_norm:
            return True
        tokens = _anchor_tokens(term)
        if not tokens:
            continue
        matched = [token for token in tokens if _anchor_norm(token) in text_norm]
        if len(tokens) == 1 and matched:
            return True
        if any(len(_anchor_norm(token)) >= 4 for token in matched):
            return True
        if len(matched) >= 2:
            return True
    return False


def _anchor_norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").casefold())


def _anchor_tokens(value: Any) -> list[str]:
    return [
        token
        for token in re.findall(r"[0-9A-Za-z가-힣]+", str(value or ""))
        if len(_anchor_norm(token)) >= 2
    ]


def _matches_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _numeric_token_key(token: str) -> str:
    text = re.sub(r"\s+", "", str(token or "").strip().lower())
    if not text:
        return ""
    unit = ""
    for candidate in ("억원", "억", "조원", "조", "만원", "만", "천만", "백만", "%", "원"):
        if text.endswith(candidate):
            unit = candidate
            text = text[: -len(candidate)]
            break
    if unit == "억원":
        unit = "억"
    elif unit == "조원":
        unit = "조"
    number_text = text.replace(",", "")
    try:
        number = float(number_text)
    except ValueError:
        normalized_number = number_text
    else:
        normalized_number = str(int(number)) if number.is_integer() else f"{number:.6f}".rstrip("0")
    return f"{normalized_number}{unit}"


def _sentence_count(text: str) -> int:
    return len(_split_sentences(text))


def _split_sentences(text: str) -> list[str]:
    return [item for item in re.split(r"[.!?。]\s*", str(text or "").strip()) if item.strip()]


def _ensure_sentence(text: str) -> str:
    sentence = re.sub(r"\s+", " ", str(text or "").strip())
    if not sentence:
        return ""
    return sentence if sentence.endswith((".", "다.", "요.", "임.")) else f"{sentence}."


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(min(max(number, 0.0), 1.0), 3)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
