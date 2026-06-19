"""summarize text_utils — extracted from facade (move-only)."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from src.analysis.summarize.config import (  # noqa: F401
    _ALLOWED_EVIDENCE_TYPES,
    _ARTICLE_CONTENT_CHARS,
    _ARTICLE_FACT_EXTRACTION_PROMPT,
    _ARTICLE_UI_BOILERPLATE_MARKERS,
    _COMPACT_ARTICLE_CONTENT_CHARS,
    _EVENT_TYPE_VALUES,
    _EVENT_TYPES,
    _FACT_EXTRACTION_BATCH_SIZE,
    _FACT_EXTRACTION_MAX_TOKENS,
    _FACT_EXTRACTION_MODE,
    _FACT_ID_SUMMARY_PROMPT,
    _FACT_TYPES,
    _FULL_TEXT_ARTICLE_LIMIT,
    _INDUSTRY_TREND_ALIASES,
    _INDUSTRY_TREND_COMPANY_ID,
    _KNOWN_COMPANY_ALIASES,
    _LLM_MODEL,
    _MAJORITY_THRESHOLD,
    _MAX_ANALYZED_ARTICLES,
    _MIN_ANALYZED_ARTICLES,
    _MIXED_THRESHOLD,
    _NEAR_DUPLICATE_SIMILARITY,
    _NUMBER_TOKEN_PATTERN,
    _PEER_ALIASES,
    _PROMPT_VERSION,
    _SNIPPET_CANDIDATE_SENTENCES,
    _SNIPPET_DEDUP_SIMILARITY,
    _SNIPPETS_PER_ARTICLE,
    _SUMMARY_LINE_MAX,
    _SUMMARY_LINE_MIN,
    _SUMMARY_MAX_TOKENS,
    _SUMMARY_ROLES,
    _SUPPORTING_ARTICLE_CONTENT_CHARS,
    _USE_FACT_EXTRACTION_LLM,
    _VALIDATION_MAX_TOKENS,
    _env_bool,
    _env_float,
    _env_int,
    _get_llm,
    _llm,
)
from src.config.companies import COMPANY_ALIASES
from src.config.company_tiers import company_tier


def _join_warnings(*values: str) -> str:
    return "; ".join(value for value in values if value)


def _is_company_neutral_context_detail(text: str) -> bool:
    value = normalize_korean_spacing(text)
    if re.search(r"업계|시장|경쟁사|타사|제3자|다른\s*회사|별도\s*사례", value):
        return False
    return bool(re.search(r"기능|업무|자동화|고객|서비스|플랫폼|제품|기술|적용|도입|활용", value))


def _ensure_sentence(text: str) -> str:
    sentence = str(text or "").strip()
    if not sentence:
        return ""
    return sentence if sentence.endswith((".", "。", "!", "?", "！", "？")) else f"{sentence}."


def _has_business_scope_terms(text: str) -> bool:
    return bool(
        re.search(
            r"계약|수주|공급|협약|사업|프로젝트|업무|시스템|전환|구축|"
            r"플랫폼|솔루션|서비스|AI|에이전트|자동화|검증|운영|고객|"
            r"ERP|MES|단말|클라우드|데이터센터|모빌리티|소프트웨어|SW",
            str(text or ""),
            re.I,
        )
    )


def _has_detail_preservation_terms(text: str) -> bool:
    return bool(
        re.search(
            r"기능|모듈|라인업|범위|대상|적용|연계|접속|처리|수행|"
            r"지원|관리|운영|보안|통합|고도화|확장|전환|도입|활용|"
            r"실증|검증|시범|상용|출시|공개|제공|개발|협력|계획|예정|향후",
            str(text or ""),
            re.I,
        )
    )


def _has_technology_mechanism_terms(text: str) -> bool:
    return bool(
        re.search(
            r"학습|검증|관제|운영|제어|배치|수집|처리|분석|예측|자동화|"
            r"통합\s*관리|시뮬레이션|오케스트레이션|워크플로|데이터\s*연계|"
            r"플랫폼|솔루션|엔진|모델|에이전트|로봇|AMR|AI|AX|RX",
            str(text or ""),
            re.I,
        )
    )


def _has_effect_or_outcome_terms(text: str) -> bool:
    return bool(
        re.search(
            r"효과|개선|단축|감소|절감|증가|확대|고도화|정확도|수행\s*속도|"
            r"효율|생산성|안정성|가시성|자동화\s*율|처리\s*시간|리드타임|"
            r"성과|적용\s*범위|현장\s*적합성|사업\s*확장|검증",
            str(text or ""),
            re.I,
        )
    )


def _has_risk_or_signal_terms(text: str) -> bool:
    return bool(
        re.search(
            r"리스크|위험|제약|불확실|한계|장애|고장|중단|보안|규제|"
            r"긍정\s*신호|수요|관심|주목|전략\s*거점|테스트베드|"
            r"후속|본사업|상용화|확산|도입\s*검토",
            str(text or ""),
            re.I,
        )
    )


def _is_customer_site_example_without_mechanism(text: str) -> bool:
    value = str(text or "")
    if not re.search(r"고객|공장|센터|현장|매장|사업장|고객사", value):
        return False
    if _has_technology_mechanism_terms(value) or _has_effect_or_outcome_terms(value):
        return False
    return bool(re.search(r"계약|구축|도입|적용|협약|공급", value))


def _looks_like_customer_site_case(text: str) -> bool:
    return bool(
        re.search(r"고객|공장|센터|현장|매장|사업장|고객사|물류센터", str(text or ""))
        and re.search(r"계약|구축|도입|적용|협약|공급|실증", str(text or ""))
    )


def _fact_is_off_topic_for_article(
    text: str,
    *,
    article: dict[str, Any],
    target_companies: list[str] | None = None,
) -> bool:
    title = str(article.get("title") or "").strip()
    if not title:
        return False
    title_tokens = _article_topic_tokens(title)
    if len(title_tokens) < 2:
        return False
    value = str(text or "")
    fact_tokens = _article_topic_tokens(value)
    if title_tokens & fact_tokens:
        return False
    if target_companies and _article_target_company_alias_mentioned(
        value, article, target_companies
    ):
        return False
    if _article_company_alias_mentioned(value, article) and _has_business_scope_terms(value):
        return False
    return True


def _fact_is_peer_related(
    text: str,
    *,
    article: dict[str, Any],
    target_companies: list[str] | None = None,
) -> bool:
    value = str(text or "")
    if target_companies and _article_target_company_alias_mentioned(
        value, article, target_companies
    ):
        return True
    if _article_company_alias_mentioned(value, article):
        return True
    matched_targets = [
        company
        for company in _matched_companies(article)
        if not target_companies or company in set(target_companies)
    ]
    if len(matched_targets) == 1 and (
        _has_peer_owned_asset_terms(value)
        or _has_business_scope_terms(value)
        or _has_technology_mechanism_terms(value)
    ):
        return True
    return False


def _has_peer_owned_asset_terms(text: str) -> bool:
    return bool(
        re.search(
            r"플랫폼|솔루션|서비스|제품|시스템|사업|프로젝트|계약|협약|MOU|"
            r"RX|AX|AI|피지컬웍스|AgenticWorks|Brity|FabriX",
            str(text or ""),
            re.I,
        )
    )


def _looks_like_industry_background_or_third_party(text: str) -> bool:
    value = str(text or "")
    if _has_peer_owned_asset_terms(value):
        return False
    return bool(
        re.search(
            r"업계에\s*따르면|시장에서는|관심이\s*집중|주목받고\s*있|"
            r"젠슨\s*황|엔비디아|삼성전자|SK하이닉스|노조|성과급|"
            r"일반\s*사무|제3자|타사|경쟁사",
            value,
            re.I,
        )
    )


def _article_topic_tokens(text: str) -> set[str]:
    stopwords = {
        "속보",
        "단독",
        "특징주",
        "정부",
        "사업",
        "참여",
        "선정",
        "체결",
        "규모",
        "지원",
        "구축",
        "확보",
        "운용",
        "관련",
        "오늘",
        "이번",
    }
    return {
        token
        for token in _article_similarity_tokens(text)
        if len(token) >= 2 and token not in stopwords and not token.isdigit()
    }


def _article_similarity_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", str(text or ""))
        if len(token) >= 2
    }


def _article_company_alias_mentioned(text: str, article: dict[str, Any]) -> bool:
    companies = [
        *_normalize_string_list(article.get("company")),
        *_normalize_string_list(article.get("matched_companies")),
        *_normalize_string_list(article.get("matched_company")),
    ]
    value = str(text or "")
    for company_id in companies:
        aliases = _PEER_ALIASES.get(company_id) or COMPANY_ALIASES.get(company_id) or []
        if any(
            alias and re.search(re.escape(str(alias)), value, re.IGNORECASE) for alias in aliases
        ):
            return True
    return False


def _article_target_company_alias_mentioned(
    text: str,
    article: dict[str, Any],
    target_companies: list[str] | None,
) -> bool:
    company_ids = _dedupe_keep_order(
        [
            *(_normalize_string_list(target_companies) if target_companies else []),
            *_company_list(article),
            *_matched_companies(article),
        ]
    )
    value = str(text or "")
    for company_id in company_ids:
        aliases = (
            _INDUSTRY_TREND_ALIASES
            if company_id == _INDUSTRY_TREND_COMPANY_ID
            else _PEER_ALIASES.get(company_id) or COMPANY_ALIASES.get(company_id) or []
        )
        if any(
            alias and re.search(re.escape(str(alias)), value, re.IGNORECASE) for alias in aliases
        ):
            return True
    return False


def _date_tokens(text: str) -> list[str]:
    return _dedupe_keep_order(
        [
            match.group(0).strip()
            for match in re.finditer(
                r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{1,2}월\s*\d{1,2}일|\d{4}-\d{2}-\d{2}",
                str(text or ""),
            )
        ]
    )


def _event_verbs_in_text(text: str) -> list[str]:
    del text
    return []


def _has_bad_korean_join(text: str) -> bool:
    value = str(text or "")
    return bool(re.search(r"(를|을|는|은|이|가)로\b", value))


def _dedupe_similar_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if any(_text_similarity(text, existing) >= 0.8 for existing in result):
            continue
        result.append(text)
    return result


def _text_similarity(left: str, right: str) -> float:
    left_compact = _compact(left)
    right_compact = _compact(right)
    if not left_compact or not right_compact:
        return 0.0
    return SequenceMatcher(None, left_compact, right_compact).ratio()


def _split_evidence_sentences(text: str, limit: int = 80) -> list[str]:
    chunks = re.split(
        r"(?<=[.!?。！？])\s+|(?<=[다요죠임음])\.\s*|\n+",
        _strip_article_ui_boilerplate(str(text or "")),
    )
    sentences: list[str] = []
    for chunk in chunks:
        sentence = re.sub(r"\s+", " ", chunk).strip()
        if len(sentence) < 8:
            continue
        if _is_article_ui_boilerplate(sentence):
            continue
        sentences.append(sentence[:500])
        if len(sentences) >= limit:
            break
    return sentences


def _strip_article_ui_boilerplate(text: str) -> str:
    value = str(text or "")
    for marker in _ARTICLE_UI_BOILERPLATE_MARKERS:
        value = value.replace(marker, " ")
    value = re.sub(r"\b[가]?\s*(?:작게|보통|크게|아주\s*크게)\s*[가]?\b", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _is_article_ui_boilerplate(sentence: str) -> bool:
    compact = re.sub(r"\s+", "", str(sentence or ""))
    if not compact:
        return True
    marker_hits = sum(
        1 for marker in _ARTICLE_UI_BOILERPLATE_MARKERS if marker.replace(" ", "") in compact
    )
    if marker_hits >= 2:
        return True
    if marker_hits and len(compact) < 120:
        return True
    share_markers = ("기사공유", "주소복사", "다크모드", "프린트", "채널구독")
    return sum(1 for marker in share_markers if marker in compact) >= 2


def _number_tokens(text: str) -> list[str]:
    return _dedupe_keep_order(
        [
            _normalize_number_token(match.group(0))
            for match in _NUMBER_TOKEN_PATTERN.finditer(str(text or ""))
            if _normalize_number_token(match.group(0))
        ]
    )


def _normalize_number_token(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").replace(",", "")).strip()


def _number_token_covered(number: str, evidence_numbers: list[str]) -> bool:
    normalized = _normalize_number_token(number)
    if not normalized:
        return True
    number_digits = re.sub(r"[^0-9.]", "", normalized)
    for evidence in evidence_numbers:
        evidence_normalized = _normalize_number_token(evidence)
        evidence_digits = re.sub(r"[^0-9.]", "", evidence_normalized)
        if normalized == evidence_normalized:
            return True
        if number_digits and number_digits == evidence_digits:
            return True
    return False


def normalize_korean_spacing(value: Any) -> str:
    """요약 출력에서 자주 붙는 한국어 조사를 보수적으로 교정한다."""
    text = str(value or "")
    text = re.sub(
        r"([가-힣A-Za-z0-9])(['\"‘’“”])\s*(를|을|은|는|이|가|와|과|에|에서|로|으로)",
        r"\1\2\3",
        text,
    )
    text = re.sub(r"(를|을|은|는|이|가|와|과|에|에서|로|으로)(?=[A-Z][A-Za-z])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _candidate_peer_companies(articles: list[dict[str, Any]]) -> list[str]:
    """Return the actual target companies for this cluster.

    Use preprocessing outputs by default. When the cluster itself is an explicit
    peer-comparison issue, preserve the peer aliases in the article body so the
    IntegratedIssue does not collapse to a single representative company.
    """
    if _is_industry_trend_cluster(articles):
        return [_INDUSTRY_TREND_COMPANY_ID]

    candidates: list[str] = []
    for article in articles:
        candidates.extend(_company_list(article))
        candidates.extend(_matched_companies(article))
    if _is_peer_comparison_issue(articles):
        candidates.extend(_body_peer_companies(articles))

    return [
        company_id
        for company_id in _dedupe_keep_order(candidates)
        if company_id in _PEER_ALIASES and company_tier(company_id) != "self"
    ]


def _is_industry_trend_cluster(articles: list[dict[str, Any]]) -> bool:
    for article in articles:
        if _INDUSTRY_TREND_COMPANY_ID in _company_list(article):
            return True
        if str(article.get("source_name") or "").strip() == "naver_industry_news":
            return True
        metadata = _metadata(article)
        if metadata.get("topic_scope") == _INDUSTRY_TREND_COMPANY_ID:
            return True
        if metadata.get("company_scope") == "industry":
            return True
    return False


def _body_peer_companies(articles: list[dict[str, Any]]) -> list[str]:
    text = _articles_text(articles)
    mentioned: list[str] = []
    for company_id, aliases in _PEER_ALIASES.items():
        if company_tier(company_id) == "self":
            continue
        if any(
            alias and re.search(re.escape(str(alias)), text, re.IGNORECASE) for alias in aliases
        ):
            mentioned.append(company_id)
    return mentioned


def _is_peer_comparison_issue(articles: list[dict[str, Any]]) -> bool:
    text = _articles_text(articles)
    if not text:
        return False
    mentioned_count = len(_body_peer_companies(articles))
    if mentioned_count < 2:
        return False
    comparison_signal = re.search(
        r"비교|대조|엇갈|반면|내부거래|의존도|비중|증가|감소|상승|하락",
        text,
    )
    metric_signal = len(_number_tokens(text)) >= 2
    return bool(comparison_signal and metric_signal)


def _articles_text(articles: list[dict[str, Any]]) -> str:
    return " ".join(
        normalize_korean_spacing(f"{article.get('title') or ''}. {article.get('content') or ''}")
        for article in articles
    )


def _company_display_name(company_id: Any) -> str:
    aliases = _PEER_ALIASES.get(str(company_id or ""), [])
    return str(aliases[0] if aliases else company_id or "")


def _company_list(article: dict[str, Any]) -> list[str]:
    return _normalize_string_list(article.get("company"))


def _matched_companies(article: dict[str, Any]) -> list[str]:
    values = _normalize_string_list(article.get("matched_companies"))
    metadata = _metadata(article)
    values.extend(_normalize_string_list(metadata.get("matched_companies")))
    return _dedupe_keep_order(values)


def _metadata(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("metadata") or article.get("extra") or {}
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _summary_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "subtitle",
        "peer_relevance",
        "peer_relevance_reason",
        "matched_aliases_by_peer",
        "body_company_mentions",
    )
    return {key: metadata[key] for key in keep_keys if key in metadata}


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, tuple):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _target_company_aliases(company_ids: list[str]) -> dict[str, list[str]]:
    return {
        company_id: (
            _INDUSTRY_TREND_ALIASES
            if company_id == _INDUSTRY_TREND_COMPANY_ID
            else _PEER_ALIASES.get(company_id, [company_id])
        )
        for company_id in company_ids
    }


def _summary_mentions_company(summary: dict[str, Any], company_id: str) -> bool:
    aliases = (
        _INDUSTRY_TREND_ALIASES
        if company_id == _INDUSTRY_TREND_COMPANY_ID
        else _PEER_ALIASES.get(company_id, [company_id])
    )
    compact_text = _compact(
        " ".join(
            [
                str(summary.get("headline") or ""),
                str(summary.get("one_line_summary") or ""),
                str(summary.get("main_event") or ""),
                " ".join(_normalize_string_list(summary.get("fact_summary"))),
            ]
        )
    )
    return any(_compact(alias) and _compact(alias) in compact_text for alias in aliases)


def _clean_domain_term(term: str) -> str:
    term = re.sub(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$", "", term)
    return re.sub(r"\s+", " ", term).strip()


def _empty_summary(
    cluster_id: int,
    representative_id: int,
    reason: str,
    source_article_ids: list[int] | None = None,
    cluster_article_ids: list[int] | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_ids = source_article_ids or []
    return {
        "cluster_id": cluster_id,
        "representative_id": representative_id,
        "source_article_ids": source_ids,
        "cluster_article_ids": cluster_article_ids or source_ids,
        "analyzed_article_ids": source_ids,
        "summary_scope": "peer_company_fact_only",
        "excluded_company_tiers": ["self"],
        "target_peer_companies": [],
        "is_valid_summary": False,
        "main_company": "",
        "mentioned_peer_companies": [],
        "cluster_event_type": "unknown",
        "headline": "",
        "one_line_summary": "",
        "fact_summary": [],
        "main_event": "",
        "fact_basis": [],
        "cluster_fact_intelligence": {
            "common_facts": [],
            "unique_facts": [],
            "uncertain_facts": [],
            "conflict_notes": [],
            "numbers_and_dates": [],
            "customers_or_industries": [],
            "products_or_services": [],
            "activity_types": [],
        },
        "coverage": coverage
        or _coverage_info(
            cluster_article_count=len(cluster_article_ids or source_ids),
            analyzed_article_count=len(source_ids),
            warning="",
        ),
        "confidence": 0.0,
        "reason": reason,
        "model": _LLM_MODEL,
    }


def _render_prompt(template: str) -> str:
    return template.replace("{event_types}", _EVENT_TYPE_VALUES)


def _article_ids(articles: list[dict[str, Any]]) -> list[int]:
    return [
        article_id
        for article_id in (_article_numeric_id(article) for article in articles)
        if article_id > 0
    ]


def _article_numeric_id(article: dict[str, Any]) -> int:
    for key in ("id", "preprocess_id"):
        value = article.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _coverage_info(
    *,
    cluster_article_count: int,
    analyzed_article_count: int,
    warning: str,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ratio = analyzed_article_count / cluster_article_count if cluster_article_count else 0.0
    coverage_warning = warning
    if cluster_article_count and analyzed_article_count < cluster_article_count:
        coverage_warning = (
            f"{coverage_warning}; " if coverage_warning else ""
        ) + "일부 cluster_article_ids는 metadata로만 보존하고 fact extraction에서 제외"
    info = {
        "cluster_article_count": cluster_article_count,
        "analyzed_article_count": analyzed_article_count,
        "coverage_ratio": round(ratio, 4),
        "coverage_warning": coverage_warning,
    }
    if selection:
        info.update(
            {
                "selection_status": selection.get("status"),
                "analysis_article_limit": selection.get("analysis_article_limit"),
                "majority_ratio": selection.get("majority_ratio"),
                "majority_article_count": len(selection.get("majority_article_ids") or []),
                "outlier_article_count": len(selection.get("outlier_article_ids") or []),
                "excluded_article_count": len(selection.get("excluded_article_ids") or []),
            }
        )
    return info


def _as_int_list(value: Any) -> list[int]:
    return [_safe_int(item) for item in _as_list(value) if _safe_int(item) > 0]


def _normalize_event_type(value: Any) -> str:
    event_type = str(value or "").strip()
    return event_type if event_type in _EVENT_TYPES else "unknown"


def _fact_key(value: str) -> str:
    compacted = _compact(value)
    return compacted[:120]


def _has_unique_fact_importance(text: str) -> bool:
    value = str(text or "")
    if re.search(r"\d", value):
        return True
    return _has_business_scope_terms(value) and _has_detail_preservation_terms(value)


def _detect_conflict_notes(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del facts
    return []


def _has_uncertain_fact_marker(sentence: str) -> bool:
    del sentence
    return False


def _append_reason(current: Any, reason: str) -> str:
    current_text = str(current or "").strip()
    if not current_text:
        return reason
    if reason in current_text:
        return current_text
    return f"{current_text}; {reason}"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _chunked(values: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parse_json(text: str) -> dict[str, Any]:
    parsed, _status = _safe_json_loads(text)
    return parsed


def _safe_json_loads(text: str) -> tuple[dict[str, Any], str]:
    cleaned = _clean_json_response(text)
    candidates = _dedupe_keep_order(
        [
            cleaned,
            _extract_json_object_text(cleaned),
            _escape_json_string_newlines(cleaned),
            _escape_json_string_newlines(_extract_json_object_text(cleaned)),
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return (parsed if isinstance(parsed, dict) else {}), "parsed"

    repaired = _repair_json_text(cleaned)
    if repaired:
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON repair failed: {exc}") from exc
        return (parsed if isinstance(parsed, dict) else {}), "repaired"

    raise ValueError("JSON parse failed")


def _clean_json_response(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        parts = value.split("```")
        if len(parts) >= 3:
            value = parts[1]
            if value.lstrip().startswith("json"):
                value = value.lstrip()[4:]
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)
    return value.strip()


def _extract_json_object_text(text: str) -> str:
    value = str(text or "")
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return value.strip()
    return value[start : end + 1].strip()


def _escape_json_string_newlines(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    for char in str(text or ""):
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\":
            result.append(char)
            escaped = True
            continue
        if char == '"':
            result.append(char)
            in_string = not in_string
            continue
        if in_string and char in {"\n", "\r"}:
            result.append("\\n")
            continue
        result.append(char)
    return "".join(result).strip()


def _repair_json_text(text: str) -> str:
    source = _extract_json_object_text(text)
    try:
        from json_repair import repair_json
    except Exception:
        return _escape_json_string_newlines(source)
    repaired = repair_json(source)
    if isinstance(repaired, dict):
        return json.dumps(repaired, ensure_ascii=False)
    return str(repaired or "").strip()


def _normalize_content(text: str) -> str:
    stripped = " ".join(text.split())
    return stripped


def _compact(text: str) -> str:
    return "".join(str(text or "").lower().split())


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)
