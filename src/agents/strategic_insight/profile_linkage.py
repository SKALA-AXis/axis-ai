"""profile_linkage — strategic_insight_agent 에서 분리 (이동만, 동작 불변).

분리 근거: axis-infra docs/structure-tasks/agent-split-design.md
"""

from __future__ import annotations

import json
import re
from typing import Any

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

_SUPPLY_CONTRACT_PATTERN = re.compile(r"공급\s*계약|공급계약|납품|구매|조달|계약\s*체결|계약")


UNCERTAIN_ACTIVITY_MARKERS = (
    "검토",
    "가능성",
    "구상",
    "계획",
    "예정",
    "모색",
    "논의",
)


GENERIC_BUSINESS_CATEGORIES: dict[str, tuple[str, ...]] = {
    "generic_entity": ("기업", "회사", "고객"),
    "generic_event": ("이번", "해당", "관련"),
    "generic_object": (
        "사업",
        "프로젝트",
        "분야",
        "서비스",
        "시스템",
        "플랫폼",
        "솔루션",
        "업무",
        "자동화",
        "기술",
        "시장",
        "service",
        "services",
        "system",
        "systems",
        "platform",
        "platforms",
        "solution",
        "solutions",
        "business",
        "project",
        "summary",
        "case",
        "use",
    ),
    "generic_relation": ("기반", "통해", "중심", "제공", "구축", "운영"),
}


def _companies_from_integrated_issue(integrated_issue: dict[str, Any]) -> list[str]:
    companies: list[str] = []
    for company_id in [
        integrated_issue.get("main_company"),
        *(_jsonish_list(integrated_issue.get("mentioned_peer_companies")) or []),
    ]:
        text = str(company_id or "").strip()
        if text and text not in companies:
            companies.append(text)
    return companies


def _cluster_fact_intelligence_for_prompt(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "activity_types": _string_list(value.get("activity_types"), max_items=10),
        "customers_or_industries": _string_list(value.get("customers_or_industries"), max_items=10),
        "products_or_services": _string_list(value.get("products_or_services"), max_items=10),
        "numbers_and_dates": _string_list(value.get("numbers_and_dates"), max_items=10),
        "unique_facts": [
            {
                "fact": str(item.get("fact") or "").strip(),
                "activity_types": _string_list(item.get("activity_types"), max_items=5),
                "customers_or_industries": _string_list(
                    item.get("customers_or_industries"), max_items=5
                ),
                "products_or_services": _string_list(item.get("products_or_services"), max_items=5),
                "summary_role": str(item.get("summary_role") or "").strip(),
                "fact_type": str(item.get("fact_type") or "").strip(),
            }
            for item in (value.get("unique_facts") or [])[:8]
            if isinstance(item, dict)
        ],
        "uncertain_facts": [
            fact_text
            for item in (value.get("uncertain_facts") or [])[:5]
            if (fact_text := _fact_like_text(item))
        ],
    }


def _fact_like_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("fact") or value.get("summary") or "").strip()
    return str(value or "").strip()


def _profile_for_prompt(
    profile: dict[str, Any],
    *,
    integrated_issue: dict[str, Any] | None = None,
    relevance_hint_text: str = "",
    include_financial_context: bool = False,
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    skax = profile.get("skax_profile") or {}
    peer_profiles = profile.get("peer_profiles") or {}
    relevance_tokens = _issue_relevance_tokens(
        integrated_issue or {},
        extra_text=relevance_hint_text,
    )
    out = {
        "skax_profile": _shrink_profile(
            skax,
            relevance_tokens=relevance_tokens,
            profile_linkage=_profile_linkage_for_company(
                profile_linkage_evaluation,
                company_id="sk_ax",
                scope="skax",
            ),
        ),
        "peer_profiles": {
            str(peer_id): _shrink_profile(
                payload,
                relevance_tokens=relevance_tokens,
                profile_linkage=_profile_linkage_for_company(
                    profile_linkage_evaluation,
                    company_id=str(peer_id),
                    scope="peer",
                ),
            )
            for peer_id, payload in (
                peer_profiles.items() if isinstance(peer_profiles, dict) else []
            )
        },
        "sector_context": profile.get("sector_context") or {},
    }
    if include_financial_context:
        financial_context = _financial_profile_context_for_prompt(
            skax=skax,
            peer_profiles=peer_profiles,
        )
        if financial_context:
            out["financial_profile_context"] = financial_context
    return out


def _profile_linkage_for_company(
    evaluation: dict[str, Any] | None,
    *,
    company_id: str,
    scope: str,
) -> dict[str, Any]:
    if not isinstance(evaluation, dict):
        return {}
    if scope == "skax":
        linkage = evaluation.get("skax_linkage") or {}
        return linkage if isinstance(linkage, dict) else {}
    for linkage in _jsonish_list(evaluation.get("peer_linkages")):
        if not isinstance(linkage, dict):
            continue
        if str(linkage.get("company_id") or "") == str(company_id):
            return linkage
    return {}


def _financial_profile_context_for_prompt(
    *,
    skax: Any,
    peer_profiles: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    skax_financial = _compact_financial_profile_context(skax)
    if skax_financial:
        out["skax_profile"] = skax_financial
    if isinstance(peer_profiles, dict):
        peer_out = {
            str(peer_id): compacted
            for peer_id, payload in peer_profiles.items()
            if (compacted := _compact_financial_profile_context(payload))
        }
        if peer_out:
            out["peer_profiles"] = peer_out
    return out


def _compact_financial_profile_context(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    keys = (
        "company_id",
        "company_name",
        "company_name_ko",
        "financial_summary",
        "operational_highlights",
        "investment_roadmap",
        "market_view",
        "validation",
    )
    out: dict[str, Any] = {}
    for key in keys:
        value = profile.get(key)
        if value in ({}, [], "", None):
            continue
        compacted = _compact_value(value)
        if compacted not in ({}, [], "", None):
            out[key] = compacted
    return out


def _build_profile_linkage_evaluation(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    profile_context: dict[str, Any],
    relevance_hint_text: str = "",
) -> dict[str, Any]:
    issue_context = _profile_linkage_issue_context(
        integrated_issue=integrated_issue,
        classification=classification,
        relevance_hint_text=relevance_hint_text,
    )
    issue_terms = _issue_relevance_tokens(
        integrated_issue,
        extra_text=" ".join(
            item
            for item in (
                relevance_hint_text,
                " ".join(_string_list(issue_context.get("structured_terms"), max_items=40)),
                str(classification.get("sector") or ""),
                " ".join(_string_list(classification.get("sectors"), max_items=8)),
                str(classification.get("event_type") or ""),
            )
            if item
        ),
    )
    role_mode = _peer_role_mode_for_linkage(
        integrated_issue,
        classification=classification,
    )
    peer_profiles = profile_context.get("peer_profiles") or {}
    peer_linkages: list[dict[str, Any]] = []
    for company_id in _companies_from_integrated_issue(integrated_issue):
        profile = peer_profiles.get(company_id) if isinstance(peer_profiles, dict) else {}
        peer_linkages.append(
            _evaluate_single_profile_linkage(
                company_id=company_id,
                profile=profile if isinstance(profile, dict) else {},
                issue_terms=issue_terms,
                issue_context=issue_context,
                integrated_issue=integrated_issue,
                scope="peer",
                role_mode=role_mode,
            )
        )

    skax_linkage = _evaluate_single_profile_linkage(
        company_id="sk_ax",
        profile=profile_context.get("skax_profile") or {},
        issue_terms=issue_terms,
        issue_context=issue_context,
        integrated_issue=integrated_issue,
        scope="skax",
        role_mode="skax_response",
    )
    return {
        "issue_terms": sorted(issue_terms)[:40],
        "peer_role_mode": role_mode,
        "peer_linkages": peer_linkages,
        "skax_linkage": skax_linkage,
        "guidance": _profile_linkage_guidance(peer_linkages, skax_linkage, role_mode),
    }


def _peer_role_mode_for_linkage(
    integrated_issue: dict[str, Any],
    *,
    classification: dict[str, Any] | None = None,
) -> str:
    if _main_company_is_customer_or_buyer(integrated_issue):
        return "counterparty_or_customer"
    structured_role = _role_mode_from_structured_activity(
        _activity_types_from_issue(integrated_issue, classification=classification)
    )
    if structured_role:
        return structured_role
    return _role_mode_from_evidence_fallback(
        _integrated_grounding_text(integrated_issue),
        integrated_issue=integrated_issue,
    )


def _profile_linkage_issue_context(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    relevance_hint_text: str = "",
) -> dict[str, Any]:
    signals = _extract_issue_structured_signals(
        integrated_issue=integrated_issue,
        classification=classification,
    )
    structured_terms: list[str] = []
    for key in (
        "activity_types",
        "products_or_services",
        "customers_or_industries",
        "target_systems",
        "structured_terms",
    ):
        structured_terms.extend(_string_list(signals.get(key), max_items=50))
    structured_terms.extend(_string_list(classification.get("sectors"), max_items=10))
    structured_terms.append(str(classification.get("sector") or ""))
    structured_terms.append(relevance_hint_text)
    evidence_texts: list[str] = []
    evidence_texts.append(_integrated_grounding_text(integrated_issue))
    for fact_id, fact_text in _fact_texts(integrated_issue):
        del fact_id
        evidence_texts.append(fact_text)
    source_noise_terms = _source_noise_terms_from_issue(integrated_issue)
    return {
        "integrated_issue": integrated_issue,
        "classification": classification,
        "activity_types": _string_list(signals.get("activity_types"), max_items=30),
        "products_or_services": _string_list(signals.get("products_or_services"), max_items=30),
        "customers_or_industries": _string_list(
            signals.get("customers_or_industries"),
            max_items=30,
        ),
        "target_systems": _string_list(signals.get("target_systems"), max_items=30),
        "structured_terms": [
            item
            for item in dict.fromkeys(
                re.sub(r"\s+", " ", str(term or "")).strip() for term in structured_terms
            )
            if item
        ][:80],
        "evidence_text": "\n".join(text for text in evidence_texts if text),
        "source_noise_terms": source_noise_terms,
    }


def _source_noise_terms_from_issue(integrated_issue: dict[str, Any]) -> set[str]:
    """Collect source/company tokens that should not drive profile relevance.

    이 값은 출력 문장을 만들기 위한 금지어가 아니라, 프로필 관련성 계산에서
    기사 출처명·회사명·ID 같은 배경 토큰이 사업/역량 근거처럼 점수를 얻는 것을
    막기 위한 노이즈 집합이다.
    """
    noise_terms: set[str] = set()

    def add_from_text(value: Any, *, include_short_ascii: bool = False) -> None:
        for term in _raw_normalized_terms(str(value or "")):
            normalized = term.casefold() if term.isascii() else term
            if not normalized:
                continue
            if len(normalized) > 2 or (include_short_ascii and normalized.isascii()):
                noise_terms.add(normalized)

    def add_from_url(value: Any) -> None:
        url = str(value or "").strip()
        if not url:
            return
        host_match = re.search(r"^(?:https?://)?([^/?#]+)", url, flags=re.IGNORECASE)
        host = (host_match.group(1) if host_match else url).removeprefix("www.")
        for part in re.split(r"[.\-_/]+", host):
            if len(part) > 2:
                noise_terms.add(_normalize_content_token(part).casefold())

    for source in _jsonish_list(integrated_issue.get("representative_sources")):
        if not isinstance(source, dict):
            continue
        for key in ("source_name", "publisher", "provider", "media", "domain"):
            add_from_text(source.get(key))
        add_from_url(source.get("url"))
    for source_ref in _jsonish_list(integrated_issue.get("source_refs")):
        if isinstance(source_ref, dict):
            for key in ("source_name", "publisher", "provider", "media", "domain"):
                add_from_text(source_ref.get(key))
            add_from_url(source_ref.get("url"))
        else:
            add_from_text(source_ref)
    for company_id in _companies_from_integrated_issue(integrated_issue):
        add_from_text(company_id, include_short_ascii=True)
        for alias in expand_peer_aliases(company_id):
            add_from_text(alias, include_short_ascii=True)
    for key in ("main_company_name", "company_name", "main_actor"):
        add_from_text(integrated_issue.get(key), include_short_ascii=True)
    return {term for term in noise_terms if term}


def _raw_normalized_terms(text: str) -> list[str]:
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·_-]{1,}", text or "")
    terms: list[str] = []
    for token in raw_tokens:
        normalized = _normalize_content_token(token)
        if normalized:
            terms.append(normalized)
    return terms


def _extract_ranked_terms(
    text: str,
    source_role: str,
    issue_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    issue_context = issue_context or {}
    ranked_by_term: dict[str, dict[str, Any]] = {}
    for term in _raw_normalized_terms(text):
        normalized = term.casefold() if term.isascii() else term
        if not normalized:
            continue
        weight = _term_signal_weight(normalized, source_role, issue_context)
        if _looks_like_source_noise(normalized, issue_context):
            term_type = "source_noise"
        elif _looks_like_korean_function_word_or_ending(normalized):
            term_type = "function_word_or_ending"
        elif _appears_in_structured_issue_fields(normalized, issue_context):
            term_type = "structured_issue_term"
        elif _appears_repeatedly_in_evidence(normalized, issue_context):
            term_type = "repeated_evidence_term"
        elif _is_generic_business_term(normalized):
            term_type = "generic_business"
        elif re.search(r"[A-Za-z]", normalized) and re.search(r"[0-9&+._-]|[A-Z]", term):
            term_type = "technical_or_named_term"
        else:
            term_type = "domain_term"
        current = ranked_by_term.get(normalized)
        payload = {
            "term": term,
            "normalized": normalized,
            "source_role": source_role,
            "term_type": term_type,
            "weight": round(weight, 3),
        }
        if current is None or float(current.get("weight") or 0.0) < weight:
            ranked_by_term[normalized] = payload
    return sorted(
        ranked_by_term.values(),
        key=lambda item: (float(item.get("weight") or 0.0), len(str(item.get("normalized") or ""))),
        reverse=True,
    )


def _term_signal_weight(
    term: str,
    source_role: str,
    issue_context: dict[str, Any] | None = None,
) -> float:
    issue_context = issue_context or {}
    normalized = str(term or "").strip()
    if not normalized or len(normalized) <= 1:
        return 0.0
    if _looks_like_source_noise(normalized, issue_context):
        return 0.0
    if normalized.isascii() and len(normalized) <= 2:
        return 0.0
    if re.fullmatch(r"\d+(?:[.,]\d+)*", normalized):
        return 0.0
    if _looks_like_korean_function_word_or_ending(normalized):
        return 0.0

    weight = 0.45
    is_generic = _is_generic_business_term(normalized)
    if is_generic:
        return 0.15
    if source_role in {"profile_business_area", "profile_capability", "profile_recent_change"}:
        weight = max(weight, 0.55)
    elif source_role in {"issue_fact", "structured_issue"}:
        weight = max(weight, 0.6)
    elif source_role == "source_metadata":
        weight = 0.0

    if _appears_repeatedly_in_evidence(normalized, issue_context):
        weight = max(weight, 0.8)
    if _appears_in_structured_issue_fields(normalized, issue_context):
        weight = max(weight, 1.2)
    if re.search(r"[A-Za-z]", normalized) and re.search(r"[0-9&+._-]|[A-Z]", term):
        weight = max(weight, 0.7)
    return weight


def _looks_like_source_noise(
    term: str,
    issue_context: dict[str, Any] | None = None,
) -> bool:
    issue_context = issue_context or {}
    normalized = str(term or "").strip().casefold()
    if not normalized:
        return True
    source_noise_terms = {
        str(item or "").casefold()
        for item in _jsonish_list(issue_context.get("source_noise_terms"))
    }
    return normalized in source_noise_terms


def _looks_like_korean_function_word_or_ending(term: str) -> bool:
    value = str(term or "").strip()
    if not value:
        return True
    if re.fullmatch(r"(이번|해당|관련|통해|위해|대한|따라서|그리고|또는)", value):
        return True
    return bool(
        re.search(
            r"(했다|한다|합니다|있다|있습니다|됐다|됩니다|이며|이다|"
            r"라고|다고|하며|하고|되는|된다|되며|으로서)$",
            value,
        )
    )


def _is_generic_business_term(term: str) -> bool:
    normalized = str(term or "").strip()
    if not normalized:
        return True
    generic_terms = {item for terms in GENERIC_BUSINESS_CATEGORIES.values() for item in terms}
    return normalized in generic_terms


def _appears_in_structured_issue_fields(
    term: str,
    issue_context: dict[str, Any] | None = None,
) -> bool:
    issue_context = issue_context or {}
    normalized = str(term or "").strip()
    if not normalized:
        return False
    structured_values: list[str] = []
    for key in (
        "products_or_services",
        "customers_or_industries",
        "activity_types",
        "target_systems",
        "structured_terms",
    ):
        structured_values.extend(_string_list(issue_context.get(key), max_items=80))
    for value in structured_values:
        for candidate in _raw_normalized_terms(value):
            if _tokens_semantically_close(normalized, candidate):
                return True
    return False


def _appears_repeatedly_in_evidence(
    term: str,
    issue_context: dict[str, Any] | None = None,
) -> bool:
    issue_context = issue_context or {}
    normalized = str(term or "").strip()
    evidence_text = str(issue_context.get("evidence_text") or "")
    if not normalized or not evidence_text:
        return False
    if len(normalized) < 3:
        return False
    return len(re.findall(re.escape(normalized), evidence_text, flags=re.IGNORECASE)) >= 2


def _activity_types_from_issue(
    integrated_issue: dict[str, Any],
    *,
    classification: dict[str, Any] | None = None,
) -> list[str]:
    values: list[str] = []
    classification = classification or {}

    def add(value: Any) -> None:
        if isinstance(value, str):
            stripped = re.sub(r"\s+", " ", value).strip()
            if stripped:
                values.append(stripped)
            return
        if isinstance(value, dict):
            for nested_key in (
                "activity_type",
                "activity_types",
                "event_type",
                "signal_type",
                "fact_type",
                "type",
            ):
                if nested_key in value:
                    add(value.get(nested_key))
            return
        if isinstance(value, list | tuple | set):
            for item in value:
                add(item)

    add(integrated_issue.get("cluster_event_type"))
    add(integrated_issue.get("event_type"))
    add(classification.get("event_type"))
    add(classification.get("activity_types"))
    add(classification.get("activities"))

    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        add(intelligence.get("activity_types"))
        add(intelligence.get("event_type"))
        for key in ("common_facts", "unique_facts", "key_facts"):
            for fact in _jsonish_list(intelligence.get(key)):
                add(fact)

    for key in ("consolidated_facts", "business_signals"):
        for item in _jsonish_list(integrated_issue.get(key)):
            add(item)

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized[:40]


def _role_mode_from_structured_activity(activity_types: list[str]) -> str:
    if not activity_types:
        return ""
    candidates = _normalize_activity_candidates(activity_types)
    if not candidates:
        return ""
    text = " ".join(candidates)

    if _structured_activity_matches_selected_role(candidates):
        return "selected_operator_or_builder"
    if _structured_activity_matches_active_role(candidates):
        return "active_provider_or_operator"
    if _structured_activity_matches_contract_role(candidates):
        return "contract_related"
    if _structured_activity_matches_partnership_role(candidates):
        return "partnership_governance"
    if _has_uncertain_role(text):
        return "unclear"
    return ""


def _normalize_activity_candidates(activity_types: list[str]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in activity_types:
        text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
        if not text or text in seen:
            continue
        seen.add(text)
        candidates.append(text)
    return candidates


def _structured_activity_matches_selected_role(candidates: list[str]) -> bool:
    exact = {
        "selection",
        "selected",
        "award",
        "operator_selected",
        "사업자 선정",
        "최종 선정",
        "민간 참여자",
        "참여 기업",
    }
    return _activity_candidates_have(candidates, exact=exact)


def _structured_activity_matches_active_role(candidates: list[str]) -> bool:
    exact = {
        "launch",
        "release",
        "service_open",
        "go_live",
        "operation_start",
        "build_completed",
        "deployment_completed",
        "정식 출시",
        "서비스 오픈",
        "서비스 개시",
        "운영 시작",
        "운영 개시",
        "구축 완료",
        "도입 완료",
        "개발 완료",
    }
    return _activity_candidates_have(candidates, exact=exact)


def _structured_activity_matches_contract_role(candidates: list[str]) -> bool:
    exact = {
        "contract",
        "supply_contract",
        "procurement",
        "purchase",
        "order",
        "계약",
        "계약 체결",
        "수주",
        "공급계약",
        "납품 계약",
        "구매 계약",
        "조달 계약",
    }
    return _activity_candidates_have(candidates, exact=exact)


def _structured_activity_matches_partnership_role(candidates: list[str]) -> bool:
    exact = {
        "partnership",
        "collaboration",
        "alliance",
        "mou",
        "consortium",
        "spc",
        "협약",
        "협력",
        "제휴",
        "컨소시엄",
        "실시협약",
        "주주간계약",
        "주주간 계약",
        "공동 추진",
    }
    return _activity_candidates_have(candidates, exact=exact)


def _activity_candidates_have(candidates: list[str], *, exact: set[str]) -> bool:
    exact_normalized = {item.casefold() for item in exact}
    for candidate in candidates:
        if candidate.casefold() in exact_normalized:
            return True
    return False


def _has_uncertain_role(text: str) -> bool:
    return any(marker in text for marker in UNCERTAIN_ACTIVITY_MARKERS)


def _role_mode_from_evidence_fallback(
    evidence_text: str,
    *,
    integrated_issue: dict[str, Any] | None = None,
) -> str:
    text = str(evidence_text or "")
    selected_pattern = (
        r"최종\s*선정|사업자(?:로)?\s*선정|민간\s*참여자(?:로)?\s*(확정|선정)|"
        r"구축\s*사업자(?:로)?\s*(확정|선정)|운영\s*사업자(?:로)?\s*(확정|선정)|"
        r"우선협상(?:대상자)?(?:로)?\s*선정"
    )
    if re.search(selected_pattern, text) and _main_company_near_role_pattern(
        integrated_issue,
        text,
        selected_pattern,
    ):
        return "selected_operator_or_builder"
    if re.search(
        r"실시협약\s*체결|주주간\s*계약\s*체결|SPC\s*설립|컨소시엄\s*(구성|참여)|"
        r"공동\s*추진\s*협약|MOU\s*체결",
        text,
        flags=re.IGNORECASE,
    ):
        return "partnership_governance"
    active_pattern = (
        r"정식\s*출시|서비스\s*(오픈|개시|출시)|구축\s*완료|운영\s*(시작|개시)|"
        r"도입\s*완료|개발\s*완료|제공\s*시작"
    )
    if re.search(active_pattern, text) and _main_company_near_role_pattern(
        integrated_issue,
        text,
        active_pattern,
    ):
        return "active_provider_or_operator"
    if re.search(
        r"계약\s*체결|공급계약\s*체결|수주(?:했다|했다고|계약)|납품\s*계약|"
        r"구매\s*계약|조달\s*계약",
        text,
        flags=re.IGNORECASE,
    ):
        return "contract_related"
    return "unclear"


def _main_company_near_role_pattern(
    integrated_issue: dict[str, Any] | None,
    text: str,
    role_pattern: str,
    *,
    window: int = 56,
) -> bool:
    issue = integrated_issue or {}
    main_company = str(issue.get("main_company") or "").strip()
    if not main_company:
        return True
    role_matches = list(re.finditer(role_pattern, text, flags=re.IGNORECASE))
    if not role_matches:
        return False
    company_matches: list[re.Match[str]] = []
    for pattern in _target_name_patterns(main_company):
        company_matches.extend(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not company_matches:
        return False
    for role_match in role_matches:
        role_start = role_match.start()
        role_end = role_match.end()
        for company_match in company_matches:
            if (
                abs(company_match.start() - role_start) <= window
                or abs(company_match.end() - role_end) <= window
                or 0 <= role_start - company_match.end() <= window
                or 0 <= company_match.start() - role_end <= window
            ):
                return True
    return False


def _evaluate_single_profile_linkage(
    *,
    company_id: str,
    profile: dict[str, Any],
    issue_terms: set[str],
    issue_context: dict[str, Any],
    integrated_issue: dict[str, Any],
    scope: str,
    role_mode: str,
) -> dict[str, Any]:
    if not isinstance(profile, dict) or not profile:
        novelty_status = _business_novelty_status(
            linkage_level="none",
            role_mode=role_mode,
            profile_available=False,
            scope=scope,
        )
        return {
            "company_id": company_id,
            "scope": scope,
            "matched_business_areas": [],
            "matched_capabilities": [],
            "matched_source_refs": [],
            "matched_terms": [],
            "linkage_level": "none",
            "business_novelty_status": novelty_status,
            "implication_mode": _implication_mode_from_linkage(
                linkage_level="none",
                novelty_status=novelty_status,
                role_mode=role_mode,
                scope=scope,
            ),
            "reason": "비교 가능한 프로필 본문이 없습니다.",
        }

    scored_entries: list[tuple[float, set[str], dict[str, Any]]] = []
    for entry in _profile_entries_for_linkage(profile):
        score, matched_terms = _score_profile_entry_against_issue(
            entry=entry,
            issue_terms=issue_terms,
            issue_context=issue_context,
        )
        if score > 0:
            scored_entries.append((score, matched_terms, entry))
    scored_entries.sort(key=lambda item: item[0], reverse=True)

    matched_business_areas: list[dict[str, Any]] = []
    matched_capabilities: list[str] = []
    matched_source_refs: list[Any] = []
    matched_terms_all: list[str] = []
    for score, matched_terms, entry in scored_entries[:5]:
        profile_item = _matched_profile_item_from_entry(
            entry=entry,
            matched_terms=matched_terms,
            score=score,
        )
        if profile_item:
            matched_business_areas.append(profile_item)
        for cap in _string_list(entry.get("capabilities"), max_items=8):
            if cap not in matched_capabilities:
                matched_capabilities.append(cap)
        if entry.get("entry_type") == "capability" and entry.get("name"):
            cap_name = str(entry.get("name") or "").strip()
            if cap_name and cap_name not in matched_capabilities:
                matched_capabilities.append(cap_name)
        matched_source_refs.extend(entry.get("source_refs") or [])
        matched_terms_all.extend(matched_terms)

    linkage_level = _linkage_level_from_profile_score(scored_entries)
    top_score = scored_entries[0][0] if scored_entries else 0.0
    novelty_status = _business_novelty_status(
        linkage_level=linkage_level,
        role_mode=role_mode,
        profile_available=True,
        scope=scope,
    )
    implication_mode = _implication_mode_from_linkage(
        linkage_level=linkage_level,
        novelty_status=novelty_status,
        role_mode=role_mode,
        scope=scope,
    )
    reason_code = _profile_linkage_reason_code(
        linkage_level=linkage_level,
        novelty_status=novelty_status,
        role_mode=role_mode,
        scope=scope,
    )
    return {
        "company_id": company_id,
        "scope": scope,
        "matched_business_areas": matched_business_areas[:3],
        "matched_capabilities": list(dict.fromkeys(matched_capabilities))[:5],
        "matched_source_refs": _compact_value(matched_source_refs[:5]),
        "matched_terms": sorted(set(matched_terms_all))[:12],
        "linkage_level": linkage_level,
        "confidence": min(1.0, round(top_score / 10, 3)),
        "business_novelty_status": novelty_status,
        "implication_mode": implication_mode,
        "connection_reason": reason_code,
        "reason": reason_code,
        "connection": {
            "matched_issue_terms": sorted(set(matched_terms_all))[:12],
            "matched_profile_terms": _profile_terms_from_matched_areas(
                matched_business_areas,
                matched_capabilities,
            ),
            "specificity_level": _best_profile_specificity_level(matched_business_areas),
            "confidence": min(1.0, round(top_score / 10, 3)),
            "reason_code": reason_code,
        },
    }


def _matched_profile_item_from_entry(
    *,
    entry: dict[str, Any],
    matched_terms: set[str],
    score: float,
) -> dict[str, Any]:
    name = str(entry.get("name") or "").strip()
    business_area = str(entry.get("business_area") or "").strip()
    business_line = str(entry.get("business_line") or "").strip()
    capabilities = _string_list(entry.get("capabilities"), max_items=8)
    products_or_services = _string_list(entry.get("products_or_services"), max_items=8)
    evidence_texts = _string_list(entry.get("evidence_texts"), max_items=3)
    source_refs = entry.get("source_refs") or []
    if not any([name, business_area, business_line, capabilities, products_or_services]):
        return {}
    return {
        "entry_type": str(entry.get("entry_type") or "").strip(),
        "name": name,
        "business_line": business_line,
        "business_area": business_area
        or (name if entry.get("entry_type") == "business_area" else ""),
        "summary": str(entry.get("summary") or "").strip(),
        "recent_direction": str(entry.get("recent_direction") or "").strip(),
        "matched_capabilities": capabilities,
        "matched_products_or_services": products_or_services,
        "matched_terms": sorted(matched_terms),
        "matched_issue_terms": sorted(matched_terms),
        "connection_reason": _profile_item_connection_reason(
            entry=entry,
            matched_terms=matched_terms,
            capabilities=capabilities,
            products_or_services=products_or_services,
        ),
        "connection": {
            "matched_issue_terms": sorted(matched_terms),
            "matched_profile_terms": _profile_item_terms(
                entry=entry,
                capabilities=capabilities,
                products_or_services=products_or_services,
            ),
            "specificity_level": _profile_entry_specificity_level(entry),
            "confidence": min(1.0, round(score / 10, 3)),
            "reason_code": _profile_item_connection_reason(
                entry=entry,
                matched_terms=matched_terms,
                capabilities=capabilities,
                products_or_services=products_or_services,
            ),
        },
        "evidence_text": evidence_texts[0] if evidence_texts else "",
        "evidence_texts": evidence_texts,
        "source_refs": source_refs,
        "source_ref": source_refs[0] if isinstance(source_refs, list) and source_refs else "",
        "specificity_level": _profile_entry_specificity_level(entry),
        "confidence": min(1.0, round(score / 10, 3)),
    }


def _profile_item_connection_reason(
    *,
    entry: dict[str, Any],
    matched_terms: set[str],
    capabilities: list[str],
    products_or_services: list[str],
) -> str:
    terms = [term for term in sorted(matched_terms) if term][:5]
    anchors = _profile_item_terms(
        entry=entry,
        capabilities=capabilities,
        products_or_services=products_or_services,
    )
    if anchors and terms:
        return "issue_terms_overlap_profile_terms"
    if anchors:
        return "profile_terms_available_without_issue_overlap"
    if terms:
        return "issue_terms_overlap_profile_entry"
    return "no_structured_profile_connection"


def _profile_item_terms(
    *,
    entry: dict[str, Any],
    capabilities: list[str],
    products_or_services: list[str],
) -> list[str]:
    terms = [
        item
        for item in [
            *products_or_services[:3],
            *capabilities[:3],
            str(entry.get("name") or "").strip(),
            str(entry.get("business_area") or "").strip(),
        ]
        if item
    ]
    return list(dict.fromkeys(terms))[:8]


def _profile_terms_from_matched_areas(
    matched_business_areas: list[dict[str, Any]],
    matched_capabilities: list[str],
) -> list[str]:
    terms: list[str] = []
    for item in matched_business_areas[:5]:
        if not isinstance(item, dict):
            continue
        for key in ("matched_products_or_services", "matched_capabilities"):
            terms.extend(_string_list(item.get(key), max_items=5))
        for key in ("name", "business_area", "business_line"):
            value = str(item.get(key) or "").strip()
            if value:
                terms.append(value)
    terms.extend(matched_capabilities[:5])
    return list(dict.fromkeys([term for term in terms if term]))[:12]


def _best_profile_specificity_level(matched_business_areas: list[dict[str, Any]]) -> str:
    rank = {
        "profile_context": 0,
        "business_line": 1,
        "business_area": 2,
        "core_capability": 3,
        "product_or_service": 4,
    }
    best = "profile_context"
    for item in matched_business_areas:
        if not isinstance(item, dict):
            continue
        level = str(item.get("specificity_level") or "profile_context")
        if rank.get(level, 0) > rank.get(best, 0):
            best = level
    return best


def _profile_entry_specificity_level(entry: dict[str, Any]) -> str:
    if _string_list(entry.get("products_or_services"), max_items=1):
        return "product_or_service"
    if _string_list(entry.get("capabilities"), max_items=1):
        return "core_capability"
    if entry.get("entry_type") == "business_area" or entry.get("business_area"):
        return "business_area"
    if entry.get("business_line"):
        return "business_line"
    return "profile_context"


def _business_novelty_status(
    *,
    linkage_level: str,
    role_mode: str,
    profile_available: bool,
    scope: str,
) -> str:
    if scope != "peer":
        return "not_applicable"
    if role_mode == "counterparty_or_customer":
        return "not_new_business_counterparty_role"
    if not profile_available:
        return "profile_insufficient_cannot_judge_novelty"
    if linkage_level in {"high", "medium"}:
        return "existing_profile_business_linked"
    if linkage_level == "low":
        return "weak_profile_linkage"
    if role_mode in {"selected_operator_or_builder", "active_provider_or_operator"}:
        return "new_or_untracked_business_signal"
    if role_mode in {"contract_related", "partnership_governance"}:
        return "role_sensitive_untracked_signal"
    return "uncertain_not_enough_to_call_new_business"


def _implication_mode_from_linkage(
    *,
    linkage_level: str,
    novelty_status: str,
    role_mode: str,
    scope: str,
) -> str:
    if scope == "peer":
        if role_mode == "counterparty_or_customer":
            return "conservative_counterparty"
        if novelty_status == "existing_profile_business_linked":
            return "profile_based"
        if novelty_status == "new_or_untracked_business_signal":
            return "new_business_signal"
        if novelty_status in {
            "weak_profile_linkage",
            "role_sensitive_untracked_signal",
            "profile_insufficient_cannot_judge_novelty",
        }:
            return "event_based"
        return "conservative_observation"
    if scope == "skax":
        if linkage_level in {"high", "medium"}:
            return "profile_based_action"
        if linkage_level == "low":
            return "cautious_action"
        return "generic_monitoring_action"
    return "conservative_observation"


def _profile_entries_for_linkage(profile: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for area in _jsonish_list(profile.get("business_areas")):
        if not isinstance(area, dict):
            continue
        name = str(area.get("name") or area.get("business_area") or "").strip()
        summary = str(area.get("summary") or "").strip()
        recent_direction = str(area.get("recent_direction") or "").strip()
        capabilities = _string_list(
            area.get("core_capabilities") or area.get("capabilities"),
            max_items=8,
        )
        products_or_services = _string_list(
            area.get("products_or_services")
            or area.get("key_products_services")
            or area.get("services"),
            max_items=8,
        )
        evidence_texts = _string_list(area.get("evidence_texts"), max_items=3)
        if area.get("evidence_text"):
            evidence_texts.append(str(area.get("evidence_text") or ""))
        source_refs = area.get("source_refs") or area.get("source_ref") or []
        entries.append(
            {
                "entry_type": "business_area",
                "business_line": str(area.get("business_line") or area.get("line") or "").strip(),
                "business_area": name,
                "name": name,
                "summary": summary,
                "recent_direction": recent_direction,
                "capabilities": capabilities,
                "products_or_services": products_or_services,
                "evidence_texts": evidence_texts,
                "source_refs": source_refs,
                "relevance_text": " ".join(
                    [
                        name,
                        summary,
                        recent_direction,
                        " ".join(capabilities),
                        " ".join(products_or_services),
                    ]
                ),
                "text": " ".join(
                    [
                        name,
                        summary,
                        recent_direction,
                        " ".join(capabilities),
                        " ".join(products_or_services),
                        " ".join(evidence_texts),
                    ]
                ),
            }
        )
        for cap in capabilities:
            entries.append(
                {
                    "entry_type": "capability",
                    "business_line": str(
                        area.get("business_line") or area.get("line") or ""
                    ).strip(),
                    "business_area": name,
                    "name": cap,
                    "capabilities": [cap],
                    "products_or_services": products_or_services,
                    "evidence_texts": evidence_texts,
                    "source_refs": source_refs,
                    "relevance_text": " ".join([cap, name, " ".join(products_or_services)]),
                    "text": " ".join([cap, name, " ".join(products_or_services)]),
                }
            )

    for product in _jsonish_list(profile.get("key_products_services")):
        if isinstance(product, dict):
            name = str(
                product.get("name")
                or product.get("title")
                or product.get("summary")
                or product.get("service_name")
                or ""
            ).strip()
            text = _json_dumps(_compact_profile_item(product))
            source_refs = product.get("source_refs") or product.get("source_ref") or []
            evidence_texts = _string_list(product.get("evidence_texts"), max_items=3)
            if product.get("evidence_text"):
                evidence_texts.append(str(product.get("evidence_text") or ""))
        else:
            name = str(product or "").strip()
            text = name
            source_refs = []
            evidence_texts = []
        if name:
            entries.append(
                {
                    "entry_type": "product_or_service",
                    "business_line": "",
                    "business_area": "",
                    "name": name,
                    "capabilities": [],
                    "products_or_services": [name],
                    "evidence_texts": evidence_texts,
                    "source_refs": source_refs,
                    "relevance_text": name,
                    "text": text,
                }
            )

    for cap in _jsonish_list(profile.get("core_capabilities")):
        if isinstance(cap, dict):
            name = str(cap.get("name") or cap.get("summary") or "").strip()
            text = _json_dumps(cap)
            source_refs = cap.get("source_refs") or cap.get("source_ref") or []
            evidence_texts = _string_list(cap.get("evidence_texts"), max_items=3)
            if cap.get("evidence_text"):
                evidence_texts.append(str(cap.get("evidence_text") or ""))
        else:
            name = str(cap or "").strip()
            text = name
            source_refs = []
            evidence_texts = []
        if name:
            entries.append(
                {
                    "entry_type": "capability",
                    "business_line": "",
                    "business_area": "",
                    "name": name,
                    "capabilities": [name],
                    "products_or_services": [],
                    "evidence_texts": evidence_texts,
                    "source_refs": source_refs,
                    "relevance_text": name,
                    "text": text,
                }
            )

    for key in ("capability_evolution", "recent_changes", "execution_cases"):
        for item in _jsonish_list(profile.get(key)):
            if isinstance(item, dict):
                text = _json_dumps(_compact_profile_item(item))
                source_refs = item.get("source_refs") or item.get("source_ref") or []
                name = str(
                    item.get("name") or item.get("summary") or item.get("title") or ""
                ).strip()
                products_or_services = _string_list(
                    item.get("products_or_services")
                    or item.get("key_products_services")
                    or item.get("services"),
                    max_items=8,
                )
                capabilities = _string_list(
                    item.get("core_capabilities") or item.get("capabilities"),
                    max_items=8,
                )
                evidence_texts = _string_list(item.get("evidence_texts"), max_items=3)
                if item.get("evidence_text"):
                    evidence_texts.append(str(item.get("evidence_text") or ""))
            else:
                text = str(item or "")
                source_refs = []
                name = text[:40]
                products_or_services = []
                capabilities = []
                evidence_texts = []
            if text.strip():
                entries.append(
                    {
                        "entry_type": key,
                        "business_line": "",
                        "business_area": "",
                        "name": name,
                        "capabilities": capabilities,
                        "products_or_services": products_or_services,
                        "evidence_texts": evidence_texts,
                        "source_refs": source_refs,
                        "relevance_text": " ".join(
                            [
                                name,
                                " ".join(capabilities),
                                " ".join(products_or_services),
                            ]
                        ),
                        "text": text,
                    }
                )
    return entries


def _score_profile_entry_against_issue(
    *,
    entry: dict[str, Any],
    issue_terms: set[str],
    issue_context: dict[str, Any],
) -> tuple[float, set[str]]:
    structured_score, structured_matches = _structured_field_overlap_score(
        entry,
        issue_context,
    )
    weighted_score, weighted_matches = _weighted_term_overlap_score(
        entry,
        issue_terms=issue_terms,
        issue_context=issue_context,
    )
    evidence_score, evidence_matches = _evidence_repeat_score(entry, issue_context)
    score = structured_score * 0.6 + weighted_score * 0.3 + evidence_score * 0.1
    matched_terms = structured_matches | weighted_matches | evidence_matches
    if (
        entry.get("entry_type") in {"capability_evolution", "recent_changes", "execution_cases"}
        and not _string_list(entry.get("products_or_services"), max_items=1)
        and not _string_list(entry.get("capabilities"), max_items=1)
    ):
        score = min(score, 1.0)
    if entry.get("entry_type") == "business_area" and structured_score >= 3:
        score += 2
    if entry.get("entry_type") == "capability" and structured_score >= 2:
        score += 1
    if entry.get("source_refs") and (
        structured_score > 0
        or any(
            _term_signal_weight(term, "issue_fact", issue_context) >= 0.5 for term in matched_terms
        )
    ):
        score += 0.5
    return score, matched_terms


def _structured_field_overlap_score(
    entry: dict[str, Any],
    issue_context: dict[str, Any],
) -> tuple[float, set[str]]:
    entry_text = _profile_entry_relevance_text(entry)
    entry_terms = {
        term["normalized"]
        for term in _extract_ranked_terms(entry_text, "profile_business_area", issue_context)
        if float(term.get("weight") or 0.0) >= 0.5
    }
    structured_groups = (
        ("products_or_services", 4.0),
        ("customers_or_industries", 3.0),
        ("activity_types", 2.0),
        ("target_systems", 3.0),
    )
    score = 0.0
    matches: set[str] = set()
    for key, weight in structured_groups:
        for raw_term in _string_list(issue_context.get(key), max_items=40):
            for structured_term in _extract_ranked_terms(
                raw_term,
                "issue_fact",
                issue_context,
            ):
                normalized = str(structured_term.get("normalized") or "")
                if not normalized:
                    continue
                if float(structured_term.get("weight") or 0.0) < 0.5:
                    continue
                if any(
                    _tokens_semantically_close(normalized, entry_term) for entry_term in entry_terms
                ):
                    score += weight
                    matches.add(normalized)
                    break
    return score, matches


def _weighted_term_overlap_score(
    entry: dict[str, Any],
    *,
    issue_terms: set[str],
    issue_context: dict[str, Any],
) -> tuple[float, set[str]]:
    entry_terms = _extract_ranked_terms(
        _profile_entry_relevance_text(entry),
        "profile_business_area",
        issue_context,
    )
    issue_ranked_terms = [
        term
        for term in _extract_ranked_terms(
            " ".join(sorted(issue_terms)),
            "issue_fact",
            issue_context,
        )
        if float(term.get("weight") or 0.0) >= 0.5
    ]
    score = 0.0
    matches: set[str] = set()
    for issue_term in issue_ranked_terms:
        issue_normalized = str(issue_term.get("normalized") or "")
        issue_weight = float(issue_term.get("weight") or 0.0)
        if not issue_normalized:
            continue
        for entry_term in entry_terms:
            entry_normalized = str(entry_term.get("normalized") or "")
            entry_weight = float(entry_term.get("weight") or 0.0)
            if not entry_normalized or min(issue_weight, entry_weight) <= 0.0:
                continue
            if _tokens_semantically_close(issue_normalized, entry_normalized):
                score += min(issue_weight, entry_weight)
                matches.add(issue_normalized)
                break
    return score, matches


def _evidence_repeat_score(
    entry: dict[str, Any],
    issue_context: dict[str, Any],
) -> tuple[float, set[str]]:
    entry_terms = {
        str(term.get("normalized") or "")
        for term in _extract_ranked_terms(
            str(entry.get("text") or ""),
            "profile_business_area",
            issue_context,
        )
        if float(term.get("weight") or 0.0) >= 0.5
    }
    repeated = {
        term
        for term in entry_terms
        if term and _appears_repeatedly_in_evidence(term, issue_context)
    }
    return float(len(repeated)), repeated


def _profile_entry_relevance_text(entry: dict[str, Any]) -> str:
    text = str(entry.get("relevance_text") or "").strip()
    return text if text else str(entry.get("text") or "")


def _linkage_level_from_profile_score(
    scored_entries: list[tuple[float, set[str], dict[str, Any]]],
) -> str:
    if not scored_entries:
        return "none"
    top_entries = scored_entries[:5]
    total_score = sum(score for score, _, _ in top_entries)
    matched_terms: set[str] = set()
    for _, terms, _ in top_entries:
        matched_terms.update(terms)
    if total_score >= 8 and len(matched_terms) >= 3:
        return "high"
    if total_score >= 4 and len(matched_terms) >= 2:
        return "medium"
    if total_score >= 1.5:
        return "low"
    return "none"


def _profile_linkage_reason_code(
    *,
    linkage_level: str,
    novelty_status: str,
    role_mode: str,
    scope: str,
) -> str:
    if linkage_level in {"high", "medium"}:
        return "structured_issue_terms_match_profile_terms"
    if scope == "peer" and novelty_status == "not_new_business_counterparty_role":
        return "counterparty_role_not_business_expansion"
    if scope == "peer" and novelty_status == "new_or_untracked_business_signal":
        return "event_signal_not_strong_profile_linkage"
    if scope == "peer" and novelty_status == "profile_insufficient_cannot_judge_novelty":
        return "profile_insufficient_for_business_novelty"
    if role_mode == "unclear":
        return "unclear_peer_role"
    return "weak_profile_linkage_event_based_preferred"


def _profile_linkage_guidance(
    peer_linkages: list[dict[str, Any]],
    skax_linkage: dict[str, Any],
    role_mode: str,
) -> list[str]:
    guidance = [f"peer_role_mode={role_mode}"]
    for linkage in peer_linkages:
        company_id = linkage.get("company_id")
        guidance.append(
            f"{company_id}: implication_mode={linkage.get('implication_mode')}, "
            f"linkage_level={linkage.get('linkage_level')}, "
            f"business_novelty_status={linkage.get('business_novelty_status')}"
        )
    guidance.append(
        "sk_ax: implication_mode="
        f"{skax_linkage.get('implication_mode')}, linkage_level={skax_linkage.get('linkage_level')}"
    )
    return guidance


def _extract_issue_structured_signals(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    activity_types = _activity_types_from_issue(
        integrated_issue,
        classification=classification,
    )
    products_or_services: list[str] = []
    customers_or_industries: list[str] = []
    target_systems: list[str] = []

    def add(target: list[str], value: Any) -> None:
        for item in _string_values_from_any(value):
            if item not in target:
                target.append(item)

    add(products_or_services, classification.get("products_or_services"))
    add(customers_or_industries, classification.get("customers_or_industries"))
    add(target_systems, classification.get("target_systems"))
    for key in ("products_or_services", "customers_or_industries", "target_systems"):
        value = integrated_issue.get(key)
        if key == "products_or_services":
            add(products_or_services, value)
        elif key == "customers_or_industries":
            add(customers_or_industries, value)
        else:
            add(target_systems, value)

    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        add(products_or_services, intelligence.get("products_or_services"))
        add(customers_or_industries, intelligence.get("customers_or_industries"))
        add(target_systems, intelligence.get("target_systems"))

    structured_terms = _issue_structured_terms(
        integrated_issue=integrated_issue,
        classification=classification,
    )
    event_type = str(
        classification.get("event_type") or integrated_issue.get("cluster_event_type") or ""
    ).strip()
    return {
        "event_type": event_type,
        "activity_types": activity_types,
        "products_or_services": products_or_services,
        "customers_or_industries": customers_or_industries,
        "target_systems": target_systems,
        "structured_terms": structured_terms,
        "fallback_text": _integrated_grounding_text(integrated_issue),
    }


def _string_values_from_any(value: Any) -> list[str]:
    values: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, str):
            stripped = re.sub(r"\s+", " ", item).strip()
            if stripped:
                values.append(stripped)
            return
        if isinstance(item, dict):
            for key in (
                "name",
                "value",
                "summary",
                "title",
                "product",
                "service",
                "industry",
                "target",
                "system",
            ):
                if key in item:
                    collect(item.get(key))
            return
        if isinstance(item, list | tuple | set):
            for nested in item:
                collect(nested)

    collect(value)
    return list(dict.fromkeys(values))


def _issue_structured_terms(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> list[str]:
    terms: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            stripped = re.sub(r"\s+", " ", value).strip()
            if stripped:
                terms.append(stripped)
            return
        if isinstance(value, list | tuple | set):
            for item in value:
                add(item)
            return
        if isinstance(value, dict):
            for key in (
                "activity_type",
                "activity_types",
                "event_type",
                "products_or_services",
                "customers_or_industries",
                "target_systems",
                "business_area",
                "sector",
                "sectors",
                "signal_type",
            ):
                if key in value:
                    add(value.get(key))

    add(_activity_types_from_issue(integrated_issue, classification=classification))
    add(classification.get("sector"))
    add(classification.get("sectors"))
    add(classification.get("products_or_services"))
    add(classification.get("customers_or_industries"))
    for key in ("products_or_services", "customers_or_industries", "target_systems"):
        add(integrated_issue.get(key))

    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for key in (
            "products_or_services",
            "customers_or_industries",
            "target_systems",
            "common_facts",
            "unique_facts",
        ):
            add(intelligence.get(key))

    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return deduped


def _supplier_names_for_target_counterparty(integrated_issue: dict[str, Any]) -> list[str]:
    main_company = str(integrated_issue.get("main_company") or "").strip()
    if not main_company:
        return []
    target_patterns = _target_name_patterns(main_company)
    if not target_patterns:
        return []
    evidence_text = _integrated_grounding_text(integrated_issue)
    suppliers: list[str] = []
    for target_pattern in target_patterns:
        patterns = (
            re.compile(
                rf"([A-Za-z가-힣0-9&㈜\.·_-]{{2,30}})(?:가|이|는|은)\s+"
                rf"[^.。!?\n]{{0,50}}?{target_pattern}\s*(?:와|과|하고)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                rf"([A-Za-z가-힣0-9&㈜\.·_-]{{2,30}})\s*(?:와|과)\s*{target_pattern}",
                flags=re.IGNORECASE,
            ),
        )
        for pattern in patterns:
            for match in pattern.finditer(evidence_text):
                supplier = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:·-")
                if supplier and _normalize_entity_token(supplier) not in _company_token_variants(
                    main_company
                ):
                    suppliers.append(supplier)
    return list(dict.fromkeys(suppliers))[:5]


def _target_name_patterns(company_id: str) -> list[str]:
    variants = expand_peer_aliases(company_id)
    normalized_seen: set[str] = set()
    patterns: list[str] = []
    for variant in variants:
        text = str(variant or "").strip()
        normalized = _normalize_entity_token(text)
        if not text or normalized in normalized_seen:
            continue
        normalized_seen.add(normalized)
        escaped = re.escape(text)
        patterns.append(escaped.replace(r"\ ", r"\s*").replace("_", r"[_\s]*"))
    return patterns


def _main_company_is_customer_or_buyer(integrated_issue: dict[str, Any]) -> bool:
    main_company = str(integrated_issue.get("main_company") or "").strip()
    if not main_company:
        return False
    evidence_text = _integrated_grounding_text(integrated_issue)
    if not _SUPPLY_CONTRACT_PATTERN.search(evidence_text):
        return False

    main_tokens = _company_token_variants(main_company)
    for item in _iter_dicts(integrated_issue):
        for key in ("customers_or_industries", "customers", "customer", "clients", "client"):
            if key not in item:
                continue
            for value in _jsonish_list(item.get(key)):
                if _normalize_entity_token(value) in main_tokens:
                    return True
    return False


def _company_token_variants(company: str) -> set[str]:
    raw = str(company or "").strip()
    if not raw:
        return set()
    variants = {
        raw,
        raw.replace("_", " "),
        raw.replace("_", ""),
        raw.upper(),
        raw.replace("_", " ").upper(),
    }
    return {_normalize_entity_token(value) for value in variants if value}


def _normalize_entity_token(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(value or "")).casefold()


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        result.append(value)
        for child in value.values():
            result.extend(_iter_dicts(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_iter_dicts(child))
    return result


def _grounding_text(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    for key in (
        "headline",
        "main_event",
        "main_issue",
        "one_line_summary",
        "integrated_text",
    ):
        value = str(integrated_issue.get(key) or "").strip()
        if value:
            parts.append(value)
    parts.extend(str(item or "") for item in integrated_issue.get("fact_summary") or [])
    for _, fact_text in _fact_texts(integrated_issue):
        parts.append(fact_text)
    intelligence = integrated_issue.get("cluster_fact_intelligence")
    if intelligence:
        parts.append(_json_dumps(_cluster_fact_intelligence_for_prompt(intelligence)))
    if profile_context:
        parts.append(
            _json_dumps(_profile_for_prompt(profile_context, integrated_issue=integrated_issue))
        )
    return "\n".join(parts)


def _integrated_grounding_text(integrated_issue: dict[str, Any]) -> str:
    return _grounding_text(integrated_issue=integrated_issue, profile_context=None)


def _fact_texts(integrated_issue: dict[str, Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for basis in integrated_issue.get("fact_basis") or []:
        if not isinstance(basis, dict):
            continue
        fact_ids = [str(item or "").strip() for item in basis.get("fact_ids") or []]
        texts = [str(basis.get("fact") or ""), str(basis.get("evidence_text") or "")]
        texts.extend(str(item or "") for item in basis.get("evidence_texts") or [])
        combined = " ".join(text for text in texts if text)
        for fact_id in fact_ids:
            if not fact_id or fact_id in seen or not combined:
                continue
            items.append((fact_id, combined))
            seen.add(fact_id)
    for fact in integrated_issue.get("consolidated_facts") or []:
        if not isinstance(fact, dict):
            continue
        fact_id = str(fact.get("fact_id") or "").strip()
        if not fact_id or fact_id in seen:
            continue
        texts = [str(fact.get("fact") or "")]
        texts.extend(str(item or "") for item in fact.get("evidence_texts") or [])
        items.append((fact_id, " ".join(texts)))
        seen.add(fact_id)
    return items


def _content_tokens(text: str) -> set[str]:
    return {
        str(term.get("normalized") or "")
        for term in _extract_ranked_terms(text, "output_text")
        if float(term.get("weight") or 0.0) > 0.0
    }


def _normalize_content_token(token: str) -> str:
    token = token.strip()
    if len(token) <= 2:
        return token
    return re.sub(r"(으로|에서|에게|과|와|은|는|이|가|을|를|의)$", "", token)


def _issue_relevance_tokens(
    integrated_issue: dict[str, Any],
    *,
    extra_text: str = "",
) -> set[str]:
    if not isinstance(integrated_issue, dict):
        return set()
    parts: list[str] = []
    if extra_text:
        parts.append(extra_text)
    for key in (
        "headline",
        "main_event",
        "main_issue",
        "one_line_summary",
        "integrated_text",
        "cluster_event_type",
    ):
        value = str(integrated_issue.get(key) or "").strip()
        if value:
            parts.append(value)
    parts.extend(str(item or "") for item in integrated_issue.get("fact_summary") or [])
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for key in ("products_or_services", "customers_or_industries", "activity_types"):
            parts.extend(str(item or "") for item in _jsonish_list(intelligence.get(key)))
        for item in intelligence.get("unique_facts") or []:
            if not isinstance(item, dict):
                continue
            parts.append(str(item.get("fact") or ""))
            for key in ("products_or_services", "customers_or_industries", "activity_types"):
                parts.extend(str(value or "") for value in _jsonish_list(item.get(key)))
    tokens = _content_tokens("\n".join(parts))
    company_tokens: set[str] = set()
    for company_id in _companies_from_integrated_issue(integrated_issue):
        company_tokens.update(_content_tokens(" ".join(expand_peer_aliases(company_id))))
        company_tokens.update(_company_token_variants(company_id))
    for supplier in _supplier_names_for_target_counterparty(integrated_issue):
        company_tokens.update(_content_tokens(supplier))
        company_tokens.add(_normalize_entity_token(supplier))
    filtered = {
        token
        for token in tokens
        if token not in company_tokens and not _is_low_signal_profile_relevance_token(token)
    }
    return _expand_profile_relevance_tokens(filtered)


def _expand_profile_relevance_tokens(tokens: set[str]) -> set[str]:
    # 프로필 관련성은 IntegratedIssue/ProfileLinkage의 실제 토큰으로 판단한다.
    # 도메인 alias를 코드에서 확장하면 다양한 기사에서 같은 사업명으로 수렴해
    # 카드 문장이 템플릿처럼 보일 수 있으므로, 여기서는 의미를 덧붙이지 않는다.
    return set(tokens)


def _profile_relevance_hint_text(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    bundle: dict[str, Any],
) -> str:
    parts: list[str] = []
    parts.extend(_string_list(classification.get("sectors"), max_items=10))
    parts.extend(_string_list(classification.get("matched_sectors"), max_items=10))
    sector = str(classification.get("sector") or "").strip()
    if sector:
        parts.append(sector)
    for detail in _jsonish_list(classification.get("matched_sector_details")):
        if isinstance(detail, dict):
            parts.append(str(detail.get("sector_name_ko") or ""))
            parts.append(str(detail.get("keyword") or ""))

    metadata = bundle.get("metadata") or {}
    for detail in _jsonish_list(metadata.get("matched_sector_details")):
        if isinstance(detail, dict):
            parts.append(str(detail.get("sector_name_ko") or ""))
            parts.append(str(detail.get("keyword") or ""))

    representative_id = str(
        integrated_issue.get("representative_id")
        or (bundle.get("metadata") or {}).get("representative_id")
        or ""
    ).strip()
    candidate_items = [
        item for item in _jsonish_list(bundle.get("items")) if isinstance(item, dict)
    ]
    if representative_id:
        selected_items = [
            item
            for item in candidate_items
            if str(item.get("id") or "").strip() == representative_id
        ]
    else:
        selected_items = candidate_items[:1]
    for item in selected_items[:1]:
        if not isinstance(item, dict):
            continue
        parts.append(str(item.get("title") or ""))
        metadata_raw = item.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
        parts.append(str(metadata.get("subtitle") or ""))
        parts.append(str(item.get("content") or "")[:700])

    for key in ("main_event", "main_issue", "one_line_summary"):
        parts.append(str(integrated_issue.get(key) or ""))
    return "\n".join(part for part in parts if part)


def _is_low_signal_profile_relevance_token(token: str) -> bool:
    return _term_signal_weight(token, "issue_fact") <= 0.0


def _shrink_profile(
    profile: Any,
    *,
    relevance_tokens: set[str] | None = None,
    profile_linkage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    relevance_tokens = relevance_tokens or set()
    identity_keys = (
        "company_id",
        "peer_id",
        "company_name",
        "company_name_ko",
        "business_lines",
    )
    scalar_keys = (
        # Broad company summaries often contain multiple business areas and can
        # pull the model toward an unrelated profile branch. Use structured
        # business areas and relevant examples instead.
    )
    relevant_item_keys = (
        # Keep the prompt centered on profile structure. Detailed profile
        # examples can overpower the current IntegratedIssue when the profile
        # snapshot is broad or noisy.
    )
    passthrough_keys = (
        "business_areas",
        "core_capabilities",
        "recent_keywords",
        "capability_evolution",
        "cautions",
    )
    out: dict[str, Any] = {}
    for key in identity_keys:
        if key not in profile:
            continue
        compacted = _compact_value(profile[key])
        if compacted not in ({}, [], "", None):
            out[key] = compacted

    linkage = profile_linkage if isinstance(profile_linkage, dict) and profile_linkage else None
    profile_linkage_level = (
        str(linkage.get("linkage_level") or "").strip() if linkage is not None else ""
    )
    include_linked_profile_body = profile_linkage_level in {"high", "medium"}

    if linkage is not None:
        linkage_keys = [
            "linkage_level",
            "business_novelty_status",
            "implication_mode",
        ]
        if include_linked_profile_body:
            linkage_keys.extend(["matched_terms", "connection"])
        linkage_meta = {
            key: linkage.get(key)
            for key in linkage_keys
            if linkage.get(key) not in ({}, [], "", None)
        }
        if linkage_meta:
            out["machine_profile_linkage_hint"] = _compact_value(linkage_meta)
        matched_areas = _jsonish_list(linkage.get("matched_business_areas"))
        if include_linked_profile_body and matched_areas:
            out["machine_matched_business_areas"] = [
                _compact_profile_item(item, include_evidence=False)
                for item in matched_areas[:3]
                if isinstance(item, dict)
            ]
        matched_caps = _string_list(linkage.get("matched_capabilities"), max_items=5)
        if include_linked_profile_body and matched_caps:
            out["machine_matched_capabilities"] = matched_caps
        return out

    for key in scalar_keys:
        if key not in profile:
            continue
        value = str(profile.get(key) or "").strip()
        if not value:
            continue
        if relevance_tokens and _profile_relevance_score(value, relevance_tokens) <= 0:
            continue
        out[key] = _compact_value(value)

    for key in relevant_item_keys:
        if key not in profile:
            continue
        ranked = _rank_relevant_profile_items(
            profile[key],
            relevance_tokens=relevance_tokens,
            max_items=3,
        )
        if ranked:
            out[key] = ranked

    for key in passthrough_keys:
        if key not in profile:
            continue
        if key in out:
            continue
        if key == "business_areas":
            compacted = _relevant_business_areas_for_prompt(
                profile[key],
                relevance_tokens=relevance_tokens,
            )
        elif key == "capability_evolution":
            compacted = _compact_capability_evolution_for_prompt(
                profile[key],
                relevance_tokens=relevance_tokens,
            )
        else:
            compacted = _compact_value(profile[key])
        if compacted not in ({}, [], "", None):
            out[key] = compacted
    return out


def _relevant_business_areas_for_prompt(
    value: Any,
    *,
    relevance_tokens: set[str],
) -> list[Any]:
    if not isinstance(value, list):
        return []
    if not relevance_tokens:
        return [_compact_profile_item(item, include_evidence=False) for item in value[:5]]

    ranked = _rank_relevant_profile_items(
        value,
        relevance_tokens=relevance_tokens,
        max_items=5,
    )
    return ranked


def _rank_relevant_profile_items(
    value: Any,
    *,
    relevance_tokens: set[str],
    max_items: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    if not relevance_tokens:
        return [
            compacted
            for item in value[:max_items]
            if (compacted := _compact_profile_item(item)) not in ({}, [], "", None)
        ]

    scored: list[tuple[int, int, Any]] = []
    for index, item in enumerate(value):
        score = _profile_relevance_score(item, relevance_tokens)
        if score > 0:
            scored.append((score, -index, item))
    scored.sort(reverse=True)
    return [
        compacted
        for _, _, item in scored[:max_items]
        if (compacted := _compact_profile_item(item, include_evidence=False))
        not in ({}, [], "", None)
    ]


def _profile_relevance_score(value: Any, relevance_tokens: set[str]) -> int:
    if not relevance_tokens:
        return 0
    item_text = _json_dumps(value) if isinstance(value, dict | list) else str(value)
    issue_context = {
        "structured_terms": sorted(relevance_tokens),
        "source_noise_terms": set(),
        "evidence_text": "",
    }
    item_terms = _extract_ranked_terms(item_text, "profile_business_area", issue_context)
    score = 0.0
    for issue_token in relevance_tokens:
        issue_weight = _term_signal_weight(issue_token, "issue_fact", issue_context)
        if issue_weight <= 0.0:
            continue
        for item_term in item_terms:
            item_token = str(item_term.get("normalized") or "")
            item_weight = float(item_term.get("weight") or 0.0)
            if item_weight <= 0.0:
                continue
            if _tokens_semantically_close(issue_token, item_token):
                score += min(issue_weight, item_weight)
                break
    return int(round(score * 10))


def _tokens_semantically_close(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) < 3 or len(right) < 3:
        return False
    return left in right or right in left


def _compact_profile_item(value: Any, *, include_evidence: bool = True) -> dict[str, Any] | Any:
    if not isinstance(value, dict):
        return _compact_value(value)
    preferred_keys = (
        "name",
        "business_area",
        "summary",
        "recent_direction",
        "core_capabilities",
        "capabilities",
        "change_type",
        "period",
        "confidence",
        "source_ref",
        "source_refs",
    )
    out: dict[str, Any] = {}
    for key in preferred_keys:
        if key not in value:
            continue
        compacted = _compact_value(value[key])
        if compacted not in ({}, [], "", None):
            out[key] = compacted
    if include_evidence:
        evidence = value.get("evidence_text") or value.get("evidence_texts")
        compacted_evidence = _compact_evidence_value(evidence)
        if compacted_evidence not in ({}, [], "", None):
            out["evidence_hint"] = compacted_evidence
    return out


def _compact_evidence_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:260]
    if isinstance(value, list):
        return [_compact_evidence_value(item) for item in value[:2]]
    if isinstance(value, dict):
        text = str(value.get("text") or value.get("evidence_text") or "").strip()
        return text[:260] if text else _compact_profile_item(value, include_evidence=False)
    return _compact_value(value)


def _compact_capability_evolution_for_prompt(
    value: Any,
    *,
    relevance_tokens: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("period", "watch_points"):
        compacted = _compact_value(value.get(key))
        if compacted not in ({}, [], "", None):
            out[key] = compacted
    changes = _rank_relevant_profile_items(
        value.get("changes"),
        relevance_tokens=relevance_tokens,
        max_items=3,
    )
    if relevance_tokens and not changes:
        return {}
    if changes:
        out["changes"] = changes
    elif not relevance_tokens:
        out["changes"] = _compact_value(value.get("changes") or [])
    overall_change = str(value.get("overall_change") or "").strip()
    if overall_change and (
        not relevance_tokens or _profile_relevance_score(overall_change, relevance_tokens) > 0
    ):
        out["overall_change"] = _compact_value(overall_change)
    return out


def _compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:700]
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:5]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            compacted = _compact_value(nested)
            if compacted not in ({}, [], "", None):
                out[str(key)] = compacted
            if len(out) >= 10:
                break
        return out
    return value


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)
