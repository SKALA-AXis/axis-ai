"""strategic_insight si_base0 — extracted from facade (move-only)."""

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
from src.config.global_companies import GLOBAL_COMPANY_ALIASES
from src.services.peer_id_aliases import expand_peer_aliases


def _evidence_snippets_from_issue(integrated_issue: dict[str, Any]) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    for item in _jsonish_list(integrated_issue.get("fact_basis")):
        if not isinstance(item, dict):
            continue
        evidence_texts = _jsonish_list(item.get("evidence_texts"))
        if not evidence_texts and item.get("evidence_text"):
            evidence_texts = [item.get("evidence_text")]
        for text_value in evidence_texts:
            text_str = str(text_value or "").strip()
            if text_str:
                snippets.append(
                    {
                        "text": text_str,
                        "fact_ids": _jsonish_list(item.get("fact_ids")),
                        "source_article_ids": _jsonish_list(item.get("source_article_ids")),
                    }
                )
    return snippets[:20]


def _fact_basis_from_evidence_refs(rows: Sequence[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        evidence_ref_id = str(item.get("evidence_ref_id") or "").strip()
        evidence_text = str(item.get("evidence_text") or "").strip()
        if not evidence_ref_id or not evidence_text:
            continue
        result.append(
            {
                "summary_line_index": index,
                "source_article_ids": _int_list(item.get("source_ids")),
                "fact_ids": [evidence_ref_id],
                "evidence_text": evidence_text,
                "evidence_texts": [evidence_text],
                "evidence_type": "reported_fact",
            }
        )
    return result


def _consolidated_facts_from_evidence_refs(rows: Sequence[Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        evidence_ref_id = str(item.get("evidence_ref_id") or "").strip()
        evidence_text = str(item.get("evidence_text") or "").strip()
        if not evidence_ref_id or not evidence_text:
            continue
        facts.append(
            {
                "fact_id": evidence_ref_id,
                "fact": evidence_text,
                "source_article_ids": _int_list(item.get("source_ids")),
                "evidence_texts": [evidence_text],
                "source_type": "news",
                "fact_type": "reported_fact",
            }
        )
    return facts


def _company_variants_for_direct_action_match(company_id: Any) -> set[str]:
    value = str(company_id or "").strip()
    if not value:
        return set()
    variants = set(_company_token_variants(value))
    variants.update(str(alias or "").strip() for alias in expand_peer_aliases(value))
    variants.add(value)
    return {variant for variant in variants if variant}


def _has_market_infra_signal(text: Any) -> bool:
    value = str(text or "")
    return bool(
        re.search(
            r"AI\s*인프라|인공지능\s*인프라|AI\s*팩토리|GPU|그래픽처리장치|"
            r"데이터\s*센터|데이터센터|컴퓨팅\s*센터|AI\s*컴퓨팅|"
            r"반도체\s*인프라|클라우드\s*인프라|인프라\s*(확장|투자|구축|확보)",
            value,
            flags=re.IGNORECASE,
        )
    )


def _global_company_alias_pattern() -> str:
    aliases: list[str] = []
    for values in GLOBAL_COMPANY_ALIASES.values():
        aliases.extend(_string_list(values, max_items=20))
    escaped = [
        re.escape(alias).replace(r"\ ", r"\s*")
        for alias in sorted(set(aliases), key=len, reverse=True)
        if alias
    ]
    return "|".join(escaped)


def _has_stock_market_signal(text: Any) -> bool:
    value = str(text or "")
    if not value.strip():
        return False
    strong_pattern = (
        r"주식\s*초고수|순매수|순매도|매수|매도|거래량|수익률|"
        r"주가|증시|시황|코스피|코스닥|장중|종가|전\s*거래일"
    )
    return bool(re.search(strong_pattern, value, flags=re.IGNORECASE))


def _hiring_signal_anchors(text: Any) -> dict[str, list[str]]:
    value = str(text or "")

    def matches(pattern: str) -> list[str]:
        return list(dict.fromkeys(re.findall(pattern, value, flags=re.IGNORECASE)))[:8]

    return {
        "hiring_or_org_terms": matches(r"채용|공채|인사|임원|조직|전담\s*조직"),
        "job_or_tech": matches(
            r"직무|직군|개발자|엔지니어|(?<![A-Za-z])AI(?![A-Za-z])|"
            r"인공지능|클라우드|데이터|보안|로봇|(?<![A-Za-z])SW(?![A-Za-z])|"
            r"소프트웨어|(?<![A-Za-z])ERP(?![A-Za-z])|컨설팅|전략|물류|"
            r"스마트팩토리|제조"
        ),
        "organization_or_business": matches(
            r"사업부|센터|본부|전담|신설|조직\s*개편|연구소|법인|부문|"
            r"신규\s*사업|사업\s*확대"
        ),
        "scale": matches(r"[0-9][0-9,]*\s*(?:명|개|여\s*명|여\s*개)|규모|채용\s*인원"),
    }


def _integrated_text_anchor_lines(
    integrated_issue: dict[str, Any],
    *,
    max_items: int = 8,
) -> list[str]:
    """Expose useful integrated_text facts to gates without trusting it as a template."""
    text = str(integrated_issue.get("integrated_text") or "").strip()
    if not text:
        return []
    normalized = re.sub(r"\s+", " ", text.replace("■", "\n")).strip()
    raw_parts = re.split(r"\n+|(?<=[.!?。！？])\s+", normalized)
    if len(raw_parts) <= 1:
        raw_parts = re.split(r"(?<=다\.)\s+|(?<=다)\s+(?=[가-힣A-Z0-9])", normalized)
    result: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        value = re.sub(r"\s+", " ", str(part or "")).strip(" -•·,.;:")
        if len(value) < 12:
            continue
        if len(value) > 260:
            value = value[:260].rsplit(" ", 1)[0].strip() or value[:260].strip()
        norm = _anchor_norm(value)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(value)
        if len(result) >= max_items:
            break
    return result


def _has_strong_actionable_issue_signal(integrated_issue: dict[str, Any]) -> bool:
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding.strip():
        return False
    strong_patterns = (
        r"계약\s*체결|공급\s*계약|추가\s*수주|수주|업무협약|실시협약|주주간\s*계약|"
        r"\bMOU\b|협약|제휴|파트너십|공동\s*추진|"
        r"출시|공개|실증|PoC|검증|현장\s*적용|특정\s*현장\s*적용|"
        r"고객\s*(적용|도입|확보)|운영\s*책임|공급\s*범위|도입\s*범위|"
        r"(?:구축|도입|공급|운영).{0,18}(?:계약|협약|제휴|파트너십|고객|현장|센터|시스템)|"
        r"(?:물류센터|데이터센터|센터).{0,24}(?:구축|운영|도입|적용)"
    )
    return bool(re.search(strong_patterns, grounding, flags=re.IGNORECASE))


def _has_sizable_tech_event_signal(integrated_issue: dict[str, Any]) -> bool:
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding.strip():
        return False
    has_event = bool(
        re.search(
            r"행사|전시|박람회|컨퍼런스|포럼|엑스포|코엑스|스마트테크\s*코리아|STK",
            grounding,
            flags=re.IGNORECASE,
        )
    )
    has_scale = bool(
        re.search(
            r"\d+\s*개국|\d+\s*개사|\d+\s*개\s*기업|\d+\s*개\s*부스|"
            r"참가(?:사|기업)?|관람객|방문객|참석자",
            grounding,
            flags=re.IGNORECASE,
        )
    )
    has_tech_theme = bool(
        re.search(
            r"AI|인공지능|자동화|로봇|스마트테크|디지털|데이터|클라우드|산업\s*전\s*과정",
            grounding,
            flags=re.IGNORECASE,
        )
    )
    return has_event and has_scale and has_tech_theme


def _has_service_advancement_fact(text: str, integrated_issue: dict[str, Any]) -> bool:
    del integrated_issue
    value = str(text or "")
    service_pattern = r"서비스|플랫폼|솔루션|시스템|제품|라인업|모듈|에이전트|Agent|AI"
    change_pattern = (
        r"고도화|개선|확대|확장|출시|공개|도입|적용|탑재|연동|전환|자동화|"
        r"활용|처리|지원|제공|분석"
    )
    return bool(
        re.search(service_pattern, value, flags=re.IGNORECASE)
        and re.search(change_pattern, value, flags=re.IGNORECASE)
    )


def _has_execution_or_operation_fact(text: str, integrated_issue: dict[str, Any]) -> bool:
    del integrated_issue
    value = str(text or "")
    return bool(
        re.search(
            r"기능|적용\s*(방식|대상|범위|업무)|운영\s*(방식|범위|책임|구조)|"
            r"자동화|데이터\s*(활용|분석|연계)|업무\s*(처리|자동화|시스템)|"
            r"공급망|관제|취약점\s*탐지|탐지|보완\s*조치|"
            r"보안\s*(모니터링|사고|대응|조치|운영)|관리형\s*보안|"
            r"ERP|메일|문서|기존\s*시스템|연동|디지털\s*서비스|서비스형",
            value,
            flags=re.IGNORECASE,
        )
    )


def _has_clear_business_domain_fact(text: str, integrated_issue: dict[str, Any]) -> bool:
    del integrated_issue
    value = str(text or "")
    return bool(
        re.search(
            r"물류|공급망|제조|스마트\s*팩토리|보안|클라우드|AI|AX|DevOps|"
            r"업무\s*자동화|사내\s*업무|금융|공공|데이터센터|인프라|운영",
            value,
            flags=re.IGNORECASE,
        )
    )


def _has_moderate_action_decision_axis(text: Any) -> bool:
    value = str(text or "")
    if not value.strip():
        return False
    return bool(
        re.search(
            r"고객\s*(수요|적용|제안|범위)|서비스\s*(적용|범위|제안|구조)|"
            r"운영\s*(책임|구간|범위|방식)|기존\s*시스템\s*(접점|연계|연동)|"
            r"서비스형\s*제안|자체\s*(수행|제공|담당)|외부\s*(협력|연계|보완)|"
            r"파트너십\s*필요성|매출\s*(기여도|구성|분류|구조)|"
            r"단순\s*구축|지속\s*운영형|고객\s*제안\s*범위|내부\s*대응\s*우선순위|"
            r"관리\s*지표|적용\s*가능성|수요\s*검증|접점\s*확인|모니터링|"
            r"탐지|보완\s*조치|사고\s*대응|보안\s*운영|보안\s*관제|"
            r"공급망\s*(운영|관리|서비스|적용)|물류\s*(운영|서비스|적용)",
            value,
            flags=re.IGNORECASE,
        )
    )


def _is_technology_event_adoption_axis(axis_key: str, anchors: Sequence[str] | None = None) -> bool:
    if str(axis_key or "") == "technology_event_adoption_signal":
        return True
    anchor_text = " ".join(str(anchor or "") for anchor in anchors or ())
    return bool(
        re.search(
            r"스마트테크\s*코리아|STK\s*2026|코엑스|개국|개사",
            anchor_text,
            flags=re.IGNORECASE,
        )
        and re.search(r"AI|인공지능|로봇|스마트\s*제조|물류|자동화", anchor_text, flags=re.I)
    )


def _generalize_peer_structure_in_action_sentence(
    sentence: str,
    *,
    integrated_issue: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> str:
    text = _strip_article_style_lead(sentence)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _action_structure_axis_phrase(integrated_issue: dict[str, Any]) -> str:
    grounding = _integrated_grounding_text(integrated_issue)
    signals = _extract_issue_structured_signals(
        integrated_issue=integrated_issue,
        classification={},
    )
    target_text = " ".join(
        [
            grounding,
            " ".join(_string_list(signals.get("target_systems"), max_items=8)),
            " ".join(_string_list(signals.get("activity_types"), max_items=8)),
        ]
    )
    axes: list[str] = []
    if re.search(r"ERP|메일|문서|데이터베이스|업무\s*시스템|사용자\s*PC|자연어", target_text):
        axes.extend(["업무 자동화 방향", "운영 시스템 연계"])
    if re.search(r"로봇|물류|관제|학습|센터|현장", target_text):
        axes.extend(["적용 업무", "운영 역할"])
    if re.search(r"보안|취약점|탐지|사고|모니터링|대응", target_text):
        axes.extend(["보안 운영 방향", "위험 대응 체계"])
    if re.search(r"GPU|데이터센터|인프라|AI\s*팩토리|컴퓨팅", target_text, flags=re.IGNORECASE):
        axes.extend(["인프라 운영 범위", "운영 지원 범위"])
    if re.search(r"계약|수주|공급|운영|DevOps|장애", target_text, flags=re.IGNORECASE):
        axes.extend(["사업 추진 방향", "운영 전환 관점"])
    if not axes:
        axes.extend(["적용 업무", "운영 방향", "대상 시스템"])
    return "·".join(_dedupe_keep_order(axes)[:2])


def _is_substantive_industry_evidence_line(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) < 18:
        return False
    if re.search(r"\.\.\.|…", text):
        return False
    return bool(re.search(r"발표|강조|구축|투자|협력|참여|운영|도입|확장|필요|진행|논의", text))


def _anchor_phrase(anchors: Sequence[str], *, max_items: int = 3) -> str:
    cleaned = _dedupe_keep_order(_string_list(anchors, max_items=max_items))
    if not cleaned:
        return "이번 산업 신호"
    if len(cleaned) == 1:
        return cleaned[0]
    return "·".join(cleaned[:max_items])


def _has_profile_context(profile: dict[str, Any]) -> bool:
    if not isinstance(profile, dict):
        return False
    return any(
        bool(profile.get(key))
        for key in (
            "one_liner",
            "company_summary",
            "key_products_services",
            "execution_cases",
            "strategic_focus",
            "priority_initiatives",
            "business_areas",
            "core_capabilities",
            "recent_changes",
            "capability_evolution",
            "market_view",
        )
    )


def _available_profile_fields(profile: dict[str, Any]) -> list[str]:
    if not isinstance(profile, dict):
        return []
    keys = (
        "one_liner",
        "company_summary",
        "key_products_services",
        "execution_cases",
        "strategic_focus",
        "priority_initiatives",
        "business_areas",
        "core_capabilities",
        "recent_changes",
        "capability_evolution",
        "market_view",
    )
    return [key for key in keys if profile.get(key)]


def _available_analysis_layers(context: dict[str, Any]) -> list[str]:
    if not isinstance(context, dict):
        return []
    candidate_keys = (
        "peer_event_timeline_recent",
        "sector_pulse_recent",
        "financial_trend",
        "event_chain_candidates",
        "similar_cards_rag",
        "evidence_density_per_peer",
    )
    return [key for key in candidate_keys if bool(context.get(key))]


def _business_line_candidates(profile: dict[str, Any]) -> list[str]:
    skax = profile.get("skax_profile") or {}
    candidates: list[str] = []
    for item in _string_list(skax.get("business_lines"), max_items=20):
        if item not in candidates:
            candidates.append(item)
    business_areas = skax.get("business_areas") or []
    if isinstance(business_areas, list):
        for area in business_areas:
            if not isinstance(area, dict):
                continue
            name = str(area.get("name") or "").strip()
            if name and name not in candidates:
                candidates.append(name)
            if len(candidates) >= 20:
                break
    return candidates


def _business_line_candidate_details(
    profile: dict[str, Any],
    *,
    integrated_issue: dict[str, Any] | None = None,
    relevance_hint_text: str = "",
) -> list[dict[str, Any]]:
    skax = profile.get("skax_profile") or {}
    relevance_tokens = _issue_relevance_tokens(
        integrated_issue or {},
        extra_text=relevance_hint_text,
    )
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for name in _string_list(skax.get("business_lines"), max_items=20):
        if name in seen:
            continue
        candidates.append({"name": name})
        seen.add(name)

    business_areas = skax.get("business_areas") or []
    if relevance_tokens:
        ranked_areas = _rank_relevant_profile_items(
            business_areas,
            relevance_tokens=relevance_tokens,
            max_items=20,
        )
        business_areas = ranked_areas or business_areas
    if isinstance(business_areas, list):
        for area in business_areas:
            if not isinstance(area, dict):
                continue
            name = str(area.get("name") or "").strip()
            if not name or name in seen:
                continue
            candidates.append(
                {
                    "name": name,
                    "summary": str(area.get("summary") or "").strip(),
                    "core_capabilities": _string_list(area.get("core_capabilities"), max_items=5),
                    "recent_direction": str(area.get("recent_direction") or "").strip(),
                    "source_refs": _compact_value(area.get("source_refs") or []),
                }
            )
            seen.add(name)
            if len(candidates) >= 20:
                break
    return candidates


def _frontend_ready_evidence_repeats_summary(
    evidence_sentence: Any,
    *,
    integrated_issue: dict[str, Any],
) -> bool:
    evidence = str(evidence_sentence or "").strip()
    evidence_norm = _anchor_norm(evidence)
    if len(evidence_norm) < 12:
        return False
    for summary_line in _string_list(integrated_issue.get("fact_summary"), max_items=8):
        summary_norm = _anchor_norm(summary_line)
        if len(summary_norm) < 12:
            continue
        if evidence_norm == summary_norm:
            return True
        if len(evidence_norm) >= len(summary_norm) and summary_norm in evidence_norm:
            interpretation_tail = evidence_norm.replace(summary_norm, "", 1)
            if len(interpretation_tail) < 8:
                return True
    return False


def _frontend_ready_internal_copy_term_violation(text: Any) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    internal_patterns: tuple[tuple[str, str], ...] = (
        (r"입력\s*근거", "내부 검증 표현인 '입력 근거'가 화면 문장에 노출됐습니다."),
        (r"\banchor\b|앵커", "내부 검증 표현인 anchor가 화면 문장에 노출됐습니다."),
        (r"이\s*기준이\s*있어야", "내부 검증식 표현이 화면 문장에 노출됐습니다."),
        (r"기사\s*안에서\s*확인됩니다", "메타 설명이 화면 문장에 노출됐습니다."),
        (r"자사\s*관여\s*가능\s*영역", "내부 검토식 표현이 화면 문장에 노출됐습니다."),
        (r"추가\s*검증(?:이\s*필요한)?\s*조건", "내부 검증식 표현이 화면 문장에 노출됐습니다."),
    )
    for pattern, message in internal_patterns:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return message
    return ""


def _frontend_ready_malformed_display_sentence_violation(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return ""
    malformed_patterns: tuple[tuple[str, str], ...] = (
        (
            r"인접\s*수요|현장\s*자동화\s*범위",
            "대응방향이 너무 넓은 표현으로 뭉개졌습니다.",
        ),
        (
            r"제안\s*단위와\s*운영\s*책임\s*구간을\s*나눠\s*설명하게\s*만든다",
            "대응방향 근거가 구체 판단 축 없이 일반 결론으로 끝났습니다.",
        ),
        (
            r"어떤\s*축에서.{0,40}비교할지\s*기준을\s*나눠|"
            r"기준을\s*나눠\s*볼\s*필요|"
            r"하나로\s*묶을지.{0,50}(?:나눌지|분리할지).{0,20}비교|"
            r"통합\s*제안할지\s*분리\s*제안할지|"
            r"(?:나눌지|분리할지)부터\s*비교|"
            r"범위를\s*나눠\s*설명할\s*필요|"
            r"구조를\s*나눠\s*비교|"
            r"사례와.{0,40}구조를\s*나눠\s*비교",
            "대응방향이 내부 메모식 비교 문장으로 작성됐습니다.",
        ),
        (
            r"범위으로|체계으로|구조으로|기준으로으로",
            "조사 오류가 있는 문장은 화면 문장으로 사용할 수 없습니다.",
        ),
        (
            r"(접점|범위|기준|대상|구간|책임)\s+업무\s*범위처럼",
            "내부 판단 축이 비문 형태로 결합됐습니다.",
        ),
        (
            r"범위\s+업무\s*(?:도구|범위|처리)",
            "업무 범위 표현이 비문 형태로 결합됐습니다.",
        ),
        (
            r"([가-힣A-Za-z0-9]+(?:·[가-힣A-Za-z0-9]+)+)"
            r"\s*(?:가|이|은|는|을|를|으로|로)?\s+\1",
            "같은 명사 묶음이 문장 안에서 반복됐습니다.",
        ),
        (
            r"[가-힣A-Za-z0-9]+(?:·[가-힣A-Za-z0-9]+){1,}\s+[가-힣A-Za-z0-9\s]{0,12}처럼",
            "명사 나열을 비유처럼 붙인 문장은 화면 문장으로 사용할 수 없습니다.",
        ),
        (
            r"([가-힣A-Za-z0-9]{2,}(?:\s+[가-힣A-Za-z0-9]{2,}){0,3})\s+\1",
            "같은 표현이 문장 안에서 반복됐습니다.",
        ),
    )
    for pattern, message in malformed_patterns:
        if re.search(pattern, value):
            return message
    return ""


def _frontend_ready_unsupported_business_concept_violation(
    block: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> str:
    sentence = str(block.get("sentence") or "")
    evidence = str(block.get("evidence_sentence") or "")
    text = f"{sentence} {evidence}"
    if not text.strip():
        return ""
    grounding = _integrated_grounding_text(integrated_issue)
    concept_rules: tuple[tuple[str, str, str], ...] = (
        (
            r"밸류에이션|캡티브(?:\s*트랩)?|공시(?:\s*체계)?|오케스트레이션|SaaS",
            r"밸류에이션|캡티브(?:\s*트랩)?|공시(?:\s*체계)?|오케스트레이션|SaaS",
            "입력 근거에 없는 전문 해석 용어를 사용했습니다.",
        ),
        (
            r"고객\s*락인|락인",
            r"락인|전환\s*비용|장기\s*계약|구독|반복\s*사용|플랫폼|관제|"
            r"운영\s*데이터|고객\s*데이터|계정|멤버십",
            "고객 락인 해석은 플랫폼·데이터·장기 이용 구조 근거가 있을 때만 사용할 수 있습니다.",
        ),
        (
            r"반복\s*매출|수익\s*모델|수익모델",
            r"매출|수익|구독|계약|서비스|운영|요금|과금|반복",
            "반복 매출/수익모델 해석은 매출·수익·계약·과금 근거가 있을 때만 사용할 수 있습니다.",
        ),
        (
            r"플랫폼화|플랫폼\s*주도권",
            r"플랫폼|관제|운영|서비스|솔루션|데이터|시스템",
            "플랫폼화/플랫폼 주도권 해석은 플랫폼·운영·데이터 구조 근거가 "
            "있을 때만 사용할 수 있습니다.",
        ),
        (
            r"사업\s*자생력|비캡티브|대외\s*(?:AX|AI|클라우드)\s*사업",
            r"대외|외부|비캡티브|내부\s*거래|내부거래|AX|AI|클라우드|매출",
            "대외 사업 자생력 해석은 대외 매출·내부거래·관련 사업 근거가 "
            "있을 때만 사용할 수 있습니다.",
        ),
        (
            r"선점|장악|입증",
            r"선점|장악|입증",
            "선점·장악·입증 같은 강한 성과 표현은 입력 근거가 있을 때만 사용할 수 있습니다.",
        ),
    )
    for concept_pattern, support_pattern, message in concept_rules:
        if re.search(concept_pattern, text, flags=re.IGNORECASE) and not re.search(
            support_pattern,
            grounding,
            flags=re.IGNORECASE,
        ):
            return message
    return ""


def _business_context_terms(text: Any) -> set[str]:
    value = str(text or "")
    terms: set[str] = set()
    for pattern in (
        r"고객|대외|외부|매출|수익|계약|협약|서비스|플랫폼|솔루션|관제|"
        r"운영|데이터|레퍼런스|적용처|고객군|파트너십|협력|거래|내부거래|"
        r"비중|시장|제안|물류센터|업무\s*시스템|클라우드|AI|AX",
    ):
        if re.search(pattern, value, flags=re.IGNORECASE):
            terms.add(pattern)
    return terms


def _is_product_or_service_launch_issue(integrated_issue: dict[str, Any]) -> bool:
    classification = integrated_issue.get("classification") or {}
    event_text = " ".join(
        str(item or "").strip().casefold()
        for item in (
            integrated_issue.get("cluster_event_type"),
            integrated_issue.get("event_type"),
            classification.get("event_type") if isinstance(classification, dict) else "",
        )
    )
    if re.search(r"tech_release|product_release|service_launch|launch|출시|공개", event_text):
        return True
    grounding = _integrated_grounding_text(integrated_issue)
    return bool(re.search(r"출시|공개|선보였|서비스를\s*시작", grounding))


def _financial_structure_terms(text: Any) -> set[str]:
    value = str(text or "")
    terms: set[str] = set()
    for pattern in (
        r"내부\s*거래",
        r"내부거래",
        r"대외\s*(?:매출|고객|거래)",
        r"외부\s*(?:매출|고객|거래)",
        r"매출\s*(?:구성|구조|비중|분류)",
        r"거래\s*(?:구성|구조|비중|의존도|상대)",
        r"그룹사\s*(?:매출|거래|의존도)",
        r"수익성",
        r"영업\s*이익",
        r"비교군",
        r"업종\s*(?:평균|기준)",
    ):
        if re.search(pattern, value, flags=re.IGNORECASE):
            terms.add(_anchor_norm(pattern.replace("\\s*", "")))
    return terms


def _frontend_ready_action_auxiliary_scale_overreach_violation(block: dict[str, Any]) -> str:
    sentence = str(block.get("sentence") or "")
    evidence = str(block.get("evidence_sentence") or "")
    text = f"{sentence} {evidence}"
    if not text.strip():
        return ""
    execution_structure_pattern = (
        r"적용\s*(업무|대상|범위)|운영\s*(책임|구간|범위|방식|조건)|"
        r"플랫폼\s*(확보|운영|구조|연계)|학습|관제|현장\s*시스템|"
        r"시스템\s*(연계|연동|접점)|로봇\s*(적용|운영|학습)|"
        r"자체\s*(수행|제공|담당)|외부\s*(협력|파트너|연계|보완)|파트너십"
    )
    has_execution_structure = bool(
        re.search(execution_structure_pattern, text, flags=re.IGNORECASE)
    )
    scale_basis_pattern = (
        r"(?:고객|시장|사업|투자|매출|거점|네트워크)\s*(?:규모|범위|수)|"
        r"거점\s*(?:범위|수|규모)|시장\s*규모|사업\s*규모|고객\s*규모|"
        r"네트워크\s*(?:범위|규모)"
    )
    if re.search(scale_basis_pattern, sentence) and not has_execution_structure:
        return (
            "고객 규모·거점 수·시장 규모 같은 보조 정보를 SK AX 대응 결론의 "
            "직접 기준으로 사용했습니다. 대응 결론은 입력 사건의 실행 구조에서 가져와야 합니다."
        )
    if not re.search(scale_basis_pattern, text):
        return ""
    direct_basis_pattern = r"(?:기준|판단|점검|검토|비교|구분|분리|나눠|내부\s*판단|대응\s*기준)"
    for match in re.finditer(scale_basis_pattern, text):
        start = max(match.start() - 18, 0)
        end = min(match.end() + 28, len(text))
        window = text[start:end]
        if re.search(direct_basis_pattern, window) and not has_execution_structure:
            return (
                "고객 규모·거점 수·시장 규모 같은 보조 정보를 SK AX 대응 기준으로 "
                "직접 연결했습니다. 대응 기준은 현재 사건의 실행 구조에서 가져와야 합니다."
            )
    return ""


def _has_direct_skax_execution_action(text: str) -> bool:
    value = str(text or "")
    if not value.strip():
        return False
    direct_execution_pattern = (
        r"(?:도입|확보|구축|운영|제공|수행|책임|확대|강화|차별화|패키징|사업화|선점|묶)"
        r"\s*(?:해야|해야\s*합니다|한다|합니다|할\s*필요|할\s*수|할\s*것|하는\s*방향)"
    )
    strong_packaging_pattern = (
        r"(?:고객\s*제안\s*단위|사업\s*라인|서비스\s*단위)[가-힣\s]*(?:패키징|차별화|확대|강화)"
        r"|대외\s*레퍼런스[가-힣\s]*(?:확보|강화)"
        r"|제안\s*(?:기준|구조|단위)[가-힣\s]*(?:보완|강화|확대)"
        r"|내부\s*비교\s*항목으로\s*삼아야"
        r"|기존\s*사업/역량\s*안에서"
        r"|기존\s*사업\s*안에서"
    )
    return bool(
        re.search(direct_execution_pattern, value) or re.search(strong_packaging_pattern, value)
    )


def _issue_specific_product_terms_for_action(integrated_issue: dict[str, Any]) -> list[str]:
    values: list[str] = []
    grounding = _integrated_grounding_text(integrated_issue)
    values.extend(re.findall(r"[‘'\"“”]([^‘'\"“”]{2,60})[’'\"“”]", grounding))
    out: list[str] = []
    generic_pattern = (
        r"사업|협약|계약|서비스|플랫폼$|시스템$|솔루션$|센터$|물류$|자동화$|"
        r"AI$|AX$|DX$|프로젝트|구축|운영"
    )
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:()[]")
        if len(text) < 3 or len(text) > 60:
            continue
        if re.search(r"[.!?。]\s*", text):
            continue
        if re.search(
            r"컨설팅|구축|계약|협약|투자|분석|검증|업무|사업|프로젝트|"
            r"실증|적용|도입|운영|전환",
            text,
        ):
            continue
        if re.fullmatch(generic_pattern, text, flags=re.IGNORECASE):
            continue
        is_product_like = bool(re.search(r"[A-Z][A-Za-z0-9]+", text) or " " in text)
        if not is_product_like:
            # Keep quoted/English/product-like names; avoid broad Korean category nouns.
            continue
        if any(_anchor_norm(text) == _anchor_norm(existing) for existing in out):
            continue
        out.append(text)
    return out[:10]


def _has_action_execution_perspective(text: str) -> bool:
    value = str(text or "")
    if not value.strip():
        return False
    perspective_groups = (
        r"자체\s*(수행|제공|담당|역량)|내부\s*(수행|점검|검토|기준|역량|책임)|"
        r"외부\s*(협력|파트너|보완|연계)|파트너십|협력\s*(필요|구간|구조|범위)|"
        r"고객\s*(제안|대상|군|요구|확인)|제안\s*(단위|구조|범위)|"
        r"고객\s*적용\s*범위|서비스\s*적용\s*범위|서비스형\s*제안|"
        r"운영\s*(책임|범위|구간|조건|데이터|체계)|책임\s*(구간|범위|분담|구조)|"
        r"연동\s*(범위|구조|방식)|시스템\s*연동|처리\s*업무\s*기준|업무\s*처리\s*범위|"
        r"매출\s*(구성|분류|구조)|거래\s*(비중|구조|의존도)|대외\s*(매출|고객)|"
        r"리스크\s*(관리|기준|부담)|시장\s*모니터링|수행\s*(범위|책임)|"
        r"역할\s*(분담|구조)|검증\s*(기준|항목|범위)|비교\s*(기준|항목|해야)|"
        r"매출\s*기여도|내부\s*대응\s*우선순위|관리\s*지표|"
        r"탐지|보완\s*조치|사고\s*대응|보안\s*(운영|관제|책임)|"
        r"공급망\s*(운영|관리|서비스|적용)|물류\s*(운영|서비스|적용)|"
        r"분리|구분|나눠"
    )
    return bool(re.search(perspective_groups, value, flags=re.IGNORECASE))


def _has_action_evidence_internal_axis(text: str) -> bool:
    value = str(text or "")
    if not value.strip():
        return False
    axis_pattern = (
        r"SK\s*AX|자사|내부\s*(수행|점검|검토|기준|역량|책임|대응|판단|분류)|"
        r"외부\s*(협력|파트너|보완|연계)|파트너십|"
        r"분리|구분|나눠|분류\s*기준|판단\s*기준|비교\s*(기준|항목|해야)|"
        r"매출\s*(구성|분류|구조)|대외\s*(매출|고객)|그룹사\s*기반|의존도|"
        r"수행\s*(범위|책임)|책임\s*(구간|범위|분담|구조)|운영\s*책임|"
        r"리스크\s*(관리|기준|부담)|연동\s*(범위|구조|방식)|시스템\s*연동|"
        r"처리\s*업무\s*기준|업무\s*처리\s*범위|검증\s*(기준|항목|범위)"
    )
    return bool(re.search(axis_pattern, value, flags=re.IGNORECASE))


def _frontend_ready_role_term_norm(value: Any) -> str:
    norm = _anchor_norm(value)
    return re.sub(
        r"(으로써|으로서|에게서|에서는|에게|에서|으로|로서|부터|까지|과의|와의|"
        r"은|는|이|가|을|를|과|와|의)$",
        "",
        norm,
    )


def _issue_counterparty_terms(
    integrated_issue: dict[str, Any],
    grounding: str,
) -> list[str]:
    values: list[str] = []
    for key in (
        "counterparty",
        "counterparties",
        "partner",
        "partners",
        "customers",
        "customer",
        "related_companies",
        "matched_companies",
        "companies_involved",
    ):
        values.extend(_string_values_from_any(integrated_issue.get(key)))
    values.extend(
        re.findall(
            r"([가-힣A-Za-z0-9&+·._ -]{2,40})(?:와|과)\s*"
            r"(?:[^.\n]{0,60})"
            r"(?:업무협약|협약|계약|파트너십|공동|협력)",
            grounding,
        )
    )
    main_norms = {
        _anchor_norm(item)
        for company_id in _companies_from_integrated_issue(integrated_issue)
        for item in [company_id, *_company_token_variants(company_id)]
    }
    filtered: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:()[]")
        norm = _anchor_norm(cleaned)
        if not norm or norm in main_norms:
            continue
        if any(norm == _anchor_norm(existing) for existing in filtered):
            continue
        filtered.append(cleaned)
        if len(filtered) >= 8:
            break
    return filtered


def _infer_frontend_claim_type(text: str) -> str:
    value = str(text or "")
    if re.search(r"주도|선도|리더십|우위|점유율|입지[을를]?\s*강화", value):
        return "market_leadership"
    if re.search(r"역량[을를]?\s*(강화|고도화|개선|높)|경쟁력[을를]?\s*(강화|높)", value):
        return "capability_improvement"
    if re.search(
        r"성과[을를]?\s*(개선|향상|높)|효율성[을를]?\s*(향상|개선|높)|수익성[을를]?\s*(개선|높)",
        value,
    ):
        return "performance_improvement"
    if re.search(r"업무\s*처리|실행형|자동화|연동|운영\s*방식", value):
        return "workflow_execution_signal"
    return "event_based_signal"


def _frontend_ready_unsupported_effect_violation(
    text: str,
    *,
    integrated_issue: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any],
) -> str:
    del profile_linkage_evaluation
    value = str(text or "")
    if not value.strip():
        return ""
    grounding = _integrated_grounding_text(integrated_issue)
    grounding_norm = _anchor_norm(grounding)
    effect_groups = (
        (
            r"외부\s*시장\s*확장|사업\s*다각화|시장\s*확장",
            ("외부시장", "비계열", "고객확대", "시장확장", "사업다각화"),
        ),
        (
            r"경쟁력[을를]?\s*(강화|높)|차별화",
            ("경쟁력강화", "경쟁력을강화", "차별화", "우위확보"),
        ),
        (
            r"역량[을를]?\s*(강화|고도화|개선|높)",
            ("역량강화", "역량을강화", "역량고도화", "역량을고도화"),
        ),
        (
            r"효율성[을를]?\s*(향상|개선|높)|성과[을를]?\s*(개선|향상|높)",
            ("효율성", "성과개선", "향상", "단축", "감소"),
        ),
        (
            r"입지[을를]?\s*강화|리더십|시장\s*주도|시장[의\s]*주목|새로운\s*기준",
            ("입지강화", "입지를강화", "리더십", "시장주도", "새로운기준"),
        ),
    )
    for pattern, support_terms in effect_groups:
        if not re.search(pattern, value):
            continue
        if any(_anchor_norm(term) in grounding_norm for term in support_terms):
            continue
        return (
            "현재 사건/프로필 근거로 직접 뒷받침되지 않는 효과성 표현이 있습니다. "
            "관찰 가능한 지표·적용 방식·관계 구조 중심으로 낮춰야 합니다."
        )
    return ""


def _distinct_anchor_tokens(value: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in _anchor_tokens(value):
        normalized = _anchor_norm(token)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(token)
    return result


def _has_unsupported_pattern(text: str, pattern: str, *, evidence_text: str) -> bool:
    if not re.search(pattern, text):
        return False
    if re.search(r"점검|비교|확인|모니터링|여부|기준", text):
        return False
    return not re.search(pattern, evidence_text)


def _is_follow_up_or_watch_field(label: str) -> bool:
    return label.startswith(("implication.follow_up_questions", "implication.watch_points"))


def _has_relationship_grounding(integrated_issue: dict[str, Any], evidence_text: str) -> bool:
    if _RELATIONSHIP_PATTERN.search(evidence_text):
        return True
    event_type = str(integrated_issue.get("cluster_event_type") or "").strip().casefold()
    if event_type in _RELATIONSHIP_ACTIVITY_TYPES:
        return True
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    activity_types = _string_list(intelligence.get("activity_types"), max_items=20)
    return any(
        str(activity).strip().casefold() in _RELATIONSHIP_ACTIVITY_TYPES
        for activity in activity_types
    )


def _counterparty_capability_overclaim_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not _main_company_is_customer_or_buyer(integrated_issue):
        return ""
    if label.startswith(("implication.follow_up_questions", "implication.watch_points")):
        return ""
    if label.startswith("skax_implication"):
        target_patterns = _target_name_patterns(str(integrated_issue.get("main_company") or ""))
        if not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in target_patterns):
            return ""
    overclaim_pattern = (
        r"역량[이가을를\s]*(강화|확장)|"
        r"경쟁력[이가을를\s]*강화|"
        r"입지[가를\s]*강화|"
        r"역할[이가을를\s]*강화|"
        r"사업\s*영역[이가을를\s]*(확장|확대)|"
        r"사업\s*범위[가를이은을\s]*(확장|확대|넓)|"
        r"제공\s*범위[가를\s]*(확장|확대|넓)|"
        r"운영\s*(안정성|안정화)[이가을를\s]*(확보|강화)|"
        r"시스템\s*전환.{0,20}운영\s*(안정성|안정화)"
    )
    if not re.search(overclaim_pattern, text):
        return ""
    return (
        "타깃 피어가 계약 상대방으로 보이는 계약을 역량 강화/경쟁력 강화 성과처럼 "
        "단정했습니다. 계약 범위/사업영역 접점 수준으로 낮춰야 합니다."
    )


def _counterparty_role_action_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not _main_company_is_customer_or_buyer(integrated_issue):
        return ""
    if label.startswith(("implication.follow_up_questions", "implication.watch_points")):
        return ""

    target_patterns = _target_name_patterns(str(integrated_issue.get("main_company") or ""))
    mentions_target = any(
        re.search(pattern, text, flags=re.IGNORECASE) for pattern in target_patterns
    )
    if not mentions_target:
        return ""

    current_event_terms = r"이번|해당|계약|사업|프로젝트|과제|수주|협약"
    direct_role_patterns = (
        rf"(?:{current_event_terms}).{{0,28}}(?:제공|공급|수행|구축|운영|지원|추진|참여|기여)",
        rf"(?:제공|공급|수행|구축|운영|지원|추진|참여|기여).{{0,28}}(?:{current_event_terms})",
    )
    target_near_direct_role = any(
        re.search(
            rf"({target_pattern}).{{0,32}}({role_pattern})|"
            rf"({role_pattern}).{{0,32}}({target_pattern})",
            text,
            flags=re.IGNORECASE,
        )
        for target_pattern in target_patterns
        for role_pattern in direct_role_patterns
    )
    profile_background_statement = re.search(
        r"(제공하는|보유한)\s*기업|기존\s*(사업영역|역량)|프로필\s*접점|프로필상",
        text,
    ) and not re.search(
        rf"(?:{current_event_terms}).{{0,28}}(?:제공|공급|수행|구축|운영|지원|추진|참여|기여)",
        text,
    )
    if target_near_direct_role and not profile_background_statement:
        return (
            "타깃 피어가 계약 상대방/고객 슬롯에 있는데 피어의 프로젝트 실행이나 "
            "공급자 행동처럼 썼습니다. 계약 범위/사업영역 접점/관찰 지점으로 낮춰야 합니다."
        )

    if label.startswith("peer_implication.capability_change") and re.search(
        r"(프로젝트|사업|계약)[가-힣\s]*(통해|참여|추진|수행|기여|제공|충족|지원)|"
        r"(효율성|안정성)[가-힣\s]*(높|개선|확보)",
        text,
    ):
        return (
            "계약 상대방 피어의 capability_change 를 프로젝트 수행 성과처럼 썼습니다. "
            "확인된 사업 범위/대상 시스템/프로필 접점으로 낮춰야 합니다."
        )

    conservative_role_terms = (
        r"계약\s*상대방|계약\s*범위|계약\s*기간|사업\s*연결|"
        r"과제와\s*연결|연결성|연결|확인|관찰|참고\s*근거"
    )
    if re.search(conservative_role_terms, text):
        return ""

    if label.startswith("skax_implication.recommended_actions") and any(
        re.search(
            pattern + r".{0,24}(에게|대상|상대로|제안|제시|영업)",
            text,
            flags=re.IGNORECASE,
        )
        for pattern in target_patterns
    ):
        if re.search(r"SK\s*AX|유사\s*고객군|유사\s*사업", text, flags=re.IGNORECASE):
            return ""
        return (
            "SK AX 대응을 타깃 피어의 특정 프로젝트에 직접 제안하는 것처럼 썼습니다. "
            "유사 고객군/유사 사업 관점은 유지하되, 외부 고객 제안 문장이 아니라 "
            "SK AX 내부에서 경쟁사 사업군과 자사 사업군의 겹침/차이, 대응 가능 범위, "
            "역량 공백, 운영·영업 전략, 후속 모니터링 항목을 점검하는 문장으로 "
            "바꿔야 합니다."
        )
    return ""


def _has_global_scope(text: str) -> bool:
    return bool(re.search(r"글로벌|해외|국외|수출|global", str(text or ""), flags=re.IGNORECASE))


def _has_public_private_scope(text: str) -> bool:
    value = str(text or "")
    if re.search(r"민관", value):
        return True
    return bool(re.search(r"공공", value) and re.search(r"민간", value))


def _has_public_private_scope_support(text: str) -> bool:
    value = str(text or "")
    if re.search(r"민관", value):
        return True
    return bool(
        re.search(r"공공|정부|국가|공공기관", value) and re.search(r"민간|기업|민간\s*참여", value)
    )


def _has_all_industry_scope(text: str) -> bool:
    return bool(re.search(r"전\s*산업|산업\s*전반|모든\s*산업|전방위", str(text or "")))


def _has_status_strength_claim(text: str) -> bool:
    return bool(
        re.search(
            r"입지[가를은\s]*(강화|확고|확대|확장|높)|"
            r"입지[를을\s]*(강화|확대|확장|높)|"
            r"레퍼런스[가를은\s]*(확보|강화)|"
            r"역량[이가을를\s]*(검증|입증)|"
            r"사업자[로서의\s]*(입지|지위)",
            str(text or ""),
        )
    )


def _has_status_strength_event_support(text: str) -> bool:
    return bool(
        re.search(
            r"최종\s*선정|사업자\s*선정|우선협상|민간\s*참여자|"
            r"공식\s*협약|실시협약|주주간\s*계약|"
            r"대형\s*수주|공급계약\s*체결|계약\s*체결|레퍼런스",
            str(text or ""),
        )
    )


def _has_broad_expansion_claim(text: str) -> bool:
    return bool(
        re.search(
            r"사업\s*(영역|범위)[이가은을를\s]*(확장|확대|넓)|"
            r"영역[이가은을를\s]*(확장|확대)|"
            r"영역.{0,18}(확장|확대)|"
            r"입지[가를은을\s]*(확장|확대|높)|"
            r"서비스[가를은을\s]*(확장|확대)|"
            r"고객군[이가은을를\s]*(확장|확대)|"
            r"부문[이가은을를에\s]*(확장|확대)",
            str(text or ""),
        )
    )


def _has_attention_growth_claim(text: str) -> bool:
    return bool(re.search(r"관심|주목", str(text or "")))


def _company_identity_terms(integrated_issue: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for company_id in _companies_from_integrated_issue(integrated_issue):
        terms.update(_content_tokens(company_id))
        try:
            aliases = expand_peer_aliases(company_id)
        except Exception:
            aliases = []
        for alias in aliases:
            terms.update(_content_tokens(str(alias or "")))
    return terms


def _has_expansion_support(text: str) -> bool:
    return bool(
        re.search(
            r"선정|수주|계약|협약|구축|참여|추진|확대|확장|신규|진출|전환|도입|센터|인프라",
            str(text or ""),
        )
    )


def _has_effectiveness_claim(text: str) -> bool:
    return bool(
        re.search(
            r"긍정적\s*영향|경쟁력[이가을를\s]*(강화|제고|높)|"
            r"운영\s*효율성[이가을를\s]*(향상|개선|높)|"
            r"수익성[이가을를\s]*(개선|향상)|"
            r"매출[이가을를\s]*(성장|확대|증가)|"
            r"성과[가를은\s]*(확대|개선|향상)",
            str(text or ""),
        )
    )


def _has_effect_scope(text: str) -> bool:
    return bool(
        re.search(
            r"현재|이번|선정|수주|계약|협약|구축|인프라|운영|GPU|데이터센터|"
            r"프로필|기존\s*역량|레퍼런스|검증|범위|기준|고객군|대상\s*시스템",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def _is_claim_scope_term(term: str) -> bool:
    if term.upper() in {"AI", "DX", "MSP", "ERP", "CRM"}:
        return True
    if term in {
        "sk",
        "ax",
        "고객",
        "고객군",
        "유사",
        "유사한",
        "사업",
        "프로젝트",
        "제안",
        "대상",
        "관련",
    }:
        return False
    return bool(
        re.search(
            r"클라우드|인공지능|블록체인|보안|로봇|팩토리|물류|ERP|CRM|MSP|AI|DX",
            term,
            flags=re.IGNORECASE,
        )
    )


def _grounded_numeric_keys_for_issue(integrated_issue: dict[str, Any]) -> set[str]:
    chunks: list[str] = []
    for key in ("fact_basis", "key_numbers", "representative_sources"):
        value = integrated_issue.get(key)
        if value:
            chunks.append(_json_dumps(value))
    grounded = " | ".join(chunks)
    return {
        key
        for match in _NUMERIC_TOKEN_PATTERN.finditer(grounded)
        if (key := _numeric_token_key(match.group(0)))
    }


def _contains_high_risk_unsupported_claim(text: str) -> bool:
    return any(re.search(pattern, str(text or "")) for pattern in _UNSUPPORTED_CLAIM_PATTERNS)


def _has_clean_displayable_frontend_ready_diagnostics(result: dict[str, Any]) -> bool:
    implication = result.get("implication") or {}
    if not isinstance(implication, dict):
        return False
    frontend_ready = implication.get("frontend_ready") or {}
    if not isinstance(frontend_ready, dict) or not frontend_ready:
        return False
    source = str(frontend_ready.get("source") or "").strip()
    if source not in _FRONTEND_READY_DISPLAY_SOURCES:
        return False
    diagnostics = implication.get("frontend_ready_diagnostics") or {}
    if not isinstance(diagnostics, dict):
        return False
    return bool(diagnostics.get("displayable")) and not (
        _string_list(diagnostics.get("required_violations"), max_items=20)
        or _string_list(diagnostics.get("claim_violations"), max_items=20)
    )


def _has_required_output_structure(result: dict[str, Any]) -> bool:
    analysis = result.get("analysis") or {}
    implication = result.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    has_analysis = bool(
        analysis.get("analysis_summary")
        and analysis.get("market_signal")
        and _string_list(analysis.get("strategic_meaning"), max_items=3)
    )
    has_implication = bool(
        (peer.get("peer_meaning") or peer.get("capability_change") or skax.get("why_important"))
        and (
            skax.get("potential_impact")
            or _string_list(skax.get("recommended_actions"), max_items=3)
            or _string_list(skax.get("opportunities"), max_items=3)
        )
    )
    return has_analysis and has_implication


def _has_relevant_peer_profile_context(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> bool:
    peer_profiles = profile_context.get("peer_profiles") or {}
    if not isinstance(peer_profiles, dict):
        return False
    for company_id in _companies_from_integrated_issue(integrated_issue):
        peer = peer_profiles.get(company_id) or {}
        if not isinstance(peer, dict):
            continue
        profile_parts = [
            peer.get(key)
            for key in (
                "business_areas",
                "core_capabilities",
                "products_or_services",
                "recent_changes",
                "recent_signals",
                "capability_evolution",
                "one_liner",
            )
            if peer.get(key)
        ]
        if not profile_parts:
            continue
        # Legacy peer_implication text is not shown as card copy anymore.  If a
        # peer profile body exists, keep the old safety behavior without
        # recomputing the expensive prompt-shaped profile.
        return True
    return False


def _fact_has_summary_role(item: dict[str, Any], role_name: str) -> bool:
    roles = {str(role or "") for role in _jsonish_list(item.get("summary_roles"))}
    role = str(item.get("summary_role") or "")
    return role_name in roles or role == role_name


def _extract_issue_subject_from_text(text: str) -> str:
    issue_text = re.sub(r"\s+", " ", str(text or "").strip())
    if not issue_text:
        return ""
    match = re.search(
        r"([가-힣A-Za-z0-9&·+\s'‘’\"“”_-]{2,100}?"
        r"(?:센터|시스템|플랫폼|인프라|단말|솔루션|서비스|사업|계약)"
        r"[가-힣A-Za-z0-9&·+\s'‘’\"“”_-]{0,40}?"
        r"(?:전환|현대화|구축|도입|개편|고도화|선정|확정|계약|사업|센터)?)",
        issue_text,
    )
    if match:
        subject = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        subject = re.sub(
            r"^(?:[가-힣A-Za-z0-9&·+\s'‘’\"“”_-]{1,30}?(?:이|가|은|는|와|과)\s+)",
            "",
            subject,
        ).strip(" .")
        return subject
    return ""


def _contract_scale_phrase(integrated_issue: dict[str, Any]) -> str:
    numbers = integrated_issue.get("key_numbers") or []
    if isinstance(numbers, list):
        phrases: list[str] = []
        for item in numbers:
            if not isinstance(item, dict):
                continue
            label = str(item.get("metric_label") or item.get("metric_name") or "").strip()
            value = item.get("value")
            unit = str(item.get("unit") or "").strip()
            if value in (None, ""):
                continue
            if re.search(r"계약|금액|매출|비율|규모|amount|revenue|ratio|percent", label, re.I):
                phrases.append(f"{label} {value}{unit}".strip())
        if phrases:
            return ", ".join(phrases[:2])

    evidence = _integrated_grounding_text(integrated_issue)
    matches = _NUMERIC_TOKEN_PATTERN.findall(evidence)
    return ", ".join(
        list(dict.fromkeys(str(match).strip() for match in matches if str(match).strip()))[:2]
    )


def _contract_duration_phrase(integrated_issue: dict[str, Any]) -> str:
    evidence = _integrated_grounding_text(integrated_issue)
    date_matches = re.findall(
        r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|\d{4}[-.]\d{1,2}[-.]\d{1,2}",
        evidence,
    )
    unique_dates = list(dict.fromkeys(re.sub(r"\s+", " ", item).strip() for item in date_matches))
    if len(unique_dates) >= 2:
        return f"계약 기간 {unique_dates[0]}~{unique_dates[1]}"
    return ""


def _fact_is_referenced(fact_text: str, output_text: str, output_tokens: set[str]) -> bool:
    tokens = _content_tokens(fact_text)
    if len(tokens & output_tokens) >= 2:
        return True
    for token in tokens:
        if len(token) >= 4 and token in output_text:
            return True
    return False


def _is_low_signal_content_token(token: str) -> bool:
    return _term_signal_weight(token, "output_text") <= 0.0


def _compact_issue_term(subject: str) -> str:
    text = re.sub(r"\s+", " ", str(subject or "").strip(" ."))
    return text if len(text) <= 80 else f"{text[:77].rstrip()}..."


def _first_peer_id(profile: dict[str, Any]) -> str:
    peer_profiles = profile.get("peer_profiles") or {}
    if isinstance(peer_profiles, dict):
        for peer_id in peer_profiles:
            if peer_id:
                return str(peer_id)
    return ""


def _first_peer_name(profile: dict[str, Any]) -> str:
    peer_profiles = profile.get("peer_profiles") or {}
    if isinstance(peer_profiles, dict):
        for payload in peer_profiles.values():
            if isinstance(payload, dict):
                name = payload.get("company_name_ko") or payload.get("company_name")
                if name:
                    return str(name)
    return ""


def _evidence_label(value: Any, confidence: float) -> str:
    raw = str(value or "").strip().lower()
    if raw in _EVIDENCE_LABELS:
        if raw == "sufficient" and confidence < 0.6:
            return "insufficient"
        return raw
    if confidence < 0.6:
        return "insufficient"
    if confidence < 0.8:
        return "moderate"
    return "sufficient"
