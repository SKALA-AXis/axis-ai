# 작성일: 2026-06-02
# 작성자: 박지원
# 변경이력:
#   2026-06-02 박지원 — 분석 파이프라인 에이전트 재구성·통합 Agent 정리, 레거시 shim 제거 및 러너 리네임
#   2026-06-05 심유정 — strategic insight 및 카드뉴스 frontend-ready 그라운딩 개선
"""AnalysisInputBundle 기반 통합 Agent.

IntegrationAgent는 source_type별 dispatcher가 아니다. 데이터 유형별 차이는
Parser / Extractor / 전처리 단계에서 처리되고, 이 Agent는 정규화된
AnalysisInputBundle 또는 NormalizedDataBundle을 받아 통합 이슈를 만든다.

기존 호출부와의 호환을 위해 cluster_id / representative_id 기반 메서드는
유지하되, 내부적으로 AnalysisInputBundle을 구성한 뒤 같은 이슈 통합 경로를 사용한다.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from src.agents.prompts.integration_v1 import (
    INTEGRATED_ISSUE_REQUIRED_FIELDS,
    INTEGRATED_ISSUE_SCHEMA_VERSION,
    INTEGRATION_PROMPT_VERSION,
)
from src.analysis.models import AnalysisInputBundle, NormalizedDataBundle
from src.analysis.summarizer import SourceSummarizer
from src.db.article_store import fetch_latest_trend_context, get_articles_by_ids

_SUMMARY_LINE_MIN = 3
_SUMMARY_LINE_MAX = 5
_DISPLAY_TRUNCATED_MARKER_PATTERN = re.compile(r"\.\.\.|…")


class IntegrationAgent:
    """원문/클러스터/문서/파싱 결과를 하나의 통합 이슈로 정리하는 Agent."""

    def __init__(self, *, engine: SourceSummarizer | None = None) -> None:
        self.engine = engine or SourceSummarizer()

    def integrate_input_bundle(self, input_bundle: AnalysisInputBundle) -> dict[str, Any]:
        """AnalysisInputBundle을 기반으로 분석 가능한 IntegratedIssue를 생성한다."""
        if _is_news_bundle(input_bundle):
            return self._integrate_news_bundle(input_bundle)
        return _integrate_non_news_bundle(input_bundle)

    def _integrate_news_bundle(self, input_bundle: AnalysisInputBundle) -> dict[str, Any]:
        """뉴스 클러스터는 기사 묶음 전체를 fact 중심으로 통합한다."""
        cluster_id = _safe_int(input_bundle.cluster_id)
        representative_id = _representative_id(input_bundle)
        cluster_article_ids = _item_ids(input_bundle.items)
        integrated_issue = self.engine.summarize_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=input_bundle.items,
            cluster_article_ids=cluster_article_ids,
        )
        return _tag_integrated_issue(
            integrated_issue,
            input_bundle=input_bundle,
            issue_component="IntegrationAgent",
        )

    def summarize_input_bundle(self, input_bundle: AnalysisInputBundle) -> dict[str, Any]:
        """기존 summarize 호출 호환 메서드."""
        return self.integrate_input_bundle(input_bundle)

    def integrate_bundle(self, bundle: NormalizedDataBundle) -> dict[str, Any]:
        """NormalizedDataBundle을 AnalysisInputBundle으로 변환해 통합 이슈를 만든다."""
        return self.integrate_input_bundle(analysis_input_bundle_from_bundle(bundle))

    def summarize_bundle(self, bundle: NormalizedDataBundle) -> dict[str, Any]:
        """기존 summarize 호출 호환 메서드."""
        return self.integrate_bundle(bundle)

    def summarize(
        self,
        cluster_id: int,
        representative_id: int,
        cluster_article_ids: list[int] | None = None,
        max_cluster_articles: int | None = None,
    ) -> dict[str, Any]:
        """기존 DB 기반 호출 호환 메서드."""
        del max_cluster_articles
        article_ids = _build_fetch_ids(representative_id, cluster_article_ids)
        articles = get_articles_by_ids(article_ids)
        input_bundle = analysis_input_bundle_from_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=articles,
            cluster_article_ids=cluster_article_ids,
        )
        return self.integrate_input_bundle(input_bundle)

    def summarize_articles(
        self,
        cluster_id: int,
        representative_id: int,
        articles: list[dict[str, Any]],
        cluster_article_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """이미 로드된 items/articles 기반 호출 호환 메서드."""
        input_bundle = analysis_input_bundle_from_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=articles,
            cluster_article_ids=cluster_article_ids,
        )
        return self.integrate_input_bundle(input_bundle)


def analysis_input_bundle_from_bundle(bundle: NormalizedDataBundle) -> AnalysisInputBundle:
    return AnalysisInputBundle(
        bundle_id=bundle.bundle_id,
        cluster_id=bundle.cluster_id,
        source_type=bundle.source_type,
        companies=bundle.companies,
        sectors=bundle.sectors,
        event_type=bundle.event_type,
        items=bundle.items,
        facts=bundle.facts,
        evidence_snippets=_evidence_snippets(bundle.items, bundle.facts),
        sources=bundle.sources,
        metadata={
            "collected_at": bundle.collected_at,
            "trend_context": _safe_trend_context(),
        },
    )


def analysis_input_bundle_from_articles(
    *,
    cluster_id: int | str | None,
    representative_id: int,
    articles: list[dict[str, Any]],
    cluster_article_ids: list[int] | None = None,
    classification: dict[str, Any] | None = None,
) -> AnalysisInputBundle:
    classification = classification or {}
    source_type = _dominant_source_type(articles)
    companies = _dedupe_strings(
        [
            *_normalize_string_list(classification.get("company")),
            *[
                company
                for article in articles
                for company in _normalize_string_list(article.get("company"))
            ],
        ]
    )
    sectors = _dedupe_strings(
        [
            *_normalize_string_list(classification.get("sector")),
            *_normalize_string_list(classification.get("sectors")),
        ]
    )
    facts = _facts_from_articles(articles)
    sources = _sources_from_articles(articles)
    bundle_id = f"{source_type}:{cluster_id or representative_id}"
    return AnalysisInputBundle(
        bundle_id=bundle_id,
        cluster_id=str(cluster_id) if cluster_id is not None else None,
        source_type=source_type,
        companies=companies,
        sectors=sectors,
        event_type=str(classification.get("event_type") or "") or None,
        items=articles,
        facts=facts,
        evidence_snippets=_evidence_snippets(articles, facts),
        sources=sources,
        metadata={
            "representative_id": representative_id,
            "cluster_article_ids": cluster_article_ids or _item_ids(articles),
            "classification": classification,
            "created_at": datetime.now(UTC).isoformat(),
            "trend_context": _safe_trend_context(),
        },
    )


_trend_context_failure_logged = False


def _safe_trend_context() -> dict[str, Any]:
    """`fetch_latest_trend_context` 의 graceful wrapper.

    DB 실패 / table 미생성 시 빈 dict 반환 — StrategicAnalyzer 가 already-empty-safe.
    첫 실패만 warning 으로 emit (hot path 라 반복 로그 회피).
    design: global-trends.md §5.2.
    """
    global _trend_context_failure_logged
    try:
        return fetch_latest_trend_context(within_days=7) or {}
    except Exception as exc:  # noqa: BLE001
        import logging

        if not _trend_context_failure_logged:
            logging.getLogger(__name__).warning(
                "fetch_latest_trend_context() 실패 — trend_context 비워둔 채로 진행 "
                "(이후 동일 오류는 반복 출력 안 함). design: global-trends.md §5.2. "
                "error=%s",
                exc,
            )
            _trend_context_failure_logged = True
        return {}


def _build_fetch_ids(
    representative_id: int,
    cluster_article_ids: list[int] | None,
) -> list[int]:
    if not cluster_article_ids:
        return [representative_id]
    return _dedupe_ints([representative_id, *cluster_article_ids])


def _is_news_bundle(input_bundle: AnalysisInputBundle) -> bool:
    return input_bundle.source_type in {"news", "news_cluster"}


def _integrate_non_news_bundle(input_bundle: AnalysisInputBundle) -> dict[str, Any]:
    """문서/리포트/구조화 신호를 분석 가능한 통합 이슈로 정리한다."""
    representative_id = _representative_id(input_bundle)
    source_article_ids = _item_ids(input_bundle.items)
    title = _first_non_empty(
        *[str(item.get("title") or "") for item in input_bundle.items[:3]],
        input_bundle.bundle_id,
    )
    consolidated_facts = _consolidated_facts_from_bundle(input_bundle)
    key_numbers = _key_numbers_from_bundle(input_bundle)
    business_signals = _business_signals_from_bundle(input_bundle)
    fact_summary = _fact_summary_lines(
        title=title,
        consolidated_facts=consolidated_facts,
        key_numbers=key_numbers,
        business_signals=business_signals,
    )
    integrated_text = _integrated_text(
        source_type=input_bundle.source_type,
        title=title,
        fact_summary=fact_summary,
        items=input_bundle.items,
    )
    is_valid = bool(integrated_text or consolidated_facts or key_numbers or business_signals)
    return _tag_integrated_issue(
        {
            "cluster_id": _safe_int(input_bundle.cluster_id),
            "representative_id": representative_id,
            "source_article_ids": source_article_ids,
            "cluster_article_ids": source_article_ids,
            "analyzed_article_ids": source_article_ids,
            "summary_scope": "integrated_issue",
            "is_valid_summary": is_valid,
            "main_company": input_bundle.companies[0] if input_bundle.companies else "",
            "mentioned_peer_companies": input_bundle.companies,
            "cluster_event_type": input_bundle.event_type or "general_update",
            "headline": title,
            "one_line_summary": fact_summary[0] if fact_summary else title,
            "main_event": title,
            "main_issue": title,
            "integrated_text": integrated_text,
            "fact_summary": fact_summary,
            "consolidated_facts": consolidated_facts,
            "key_numbers": key_numbers,
            "business_signals": business_signals,
            "representative_sources": input_bundle.sources[:5],
            "fact_basis": _fact_basis_from_facts(consolidated_facts),
            "missing_or_uncertain_points": [],
            "source_count": len(input_bundle.sources),
            "confidence": 0.75 if is_valid else 0.0,
            "reason": "" if is_valid else "통합 이슈를 구성할 근거가 부족합니다.",
        },
        input_bundle=input_bundle,
        issue_component="IntegrationAgent",
    )


def _tag_integrated_issue(
    integrated_issue: dict[str, Any],
    *,
    input_bundle: AnalysisInputBundle,
    issue_component: str,
) -> dict[str, Any]:
    if not isinstance(integrated_issue, dict):
        return integrated_issue
    input_contract_validation = (
        input_bundle.validate_news_contract() if input_bundle.is_news_cluster else {}
    )
    preprocessing_outputs = (
        input_bundle.news_preprocessing_outputs if input_bundle.is_news_cluster else {}
    )
    source_article_ids = _source_article_ids_for_issue(integrated_issue, input_bundle)
    main_issue = (
        integrated_issue.get("main_issue")
        or integrated_issue.get("main_topic")
        or integrated_issue.get("headline")
        or integrated_issue.get("one_line_summary")
        or integrated_issue.get("main_event")
        or ""
    )
    integrated_text = (
        integrated_issue.get("integrated_text")
        or integrated_issue.get("summary")
        or integrated_issue.get("one_line_summary")
        or " ".join(_normalize_string_list(integrated_issue.get("fact_summary")))
    )
    consolidated_facts = integrated_issue.get("consolidated_facts") or integrated_issue.get(
        "key_facts"
    )
    if not consolidated_facts:
        consolidated_facts = [
            {"fact": fact} for fact in _normalize_string_list(integrated_issue.get("fact_summary"))
        ]
    fact_summary = _fact_summary_from_issue(
        integrated_issue,
        consolidated_facts=consolidated_facts,
        integrated_text=str(integrated_text),
    )
    representative_sources = integrated_issue.get("representative_sources")
    if not isinstance(representative_sources, list) or not representative_sources:
        representative_sources = input_bundle.sources[:5]
    main_company = _first_non_empty(
        integrated_issue.get("main_company"),
        input_bundle.companies[0] if input_bundle.companies else "",
    )
    mentioned_peer_companies = _mentioned_peer_companies(
        integrated_issue=integrated_issue,
        input_bundle=input_bundle,
        main_company=main_company,
    )
    cluster_event_type = _first_non_empty(
        integrated_issue.get("cluster_event_type"),
        input_bundle.event_type,
        _classification(input_bundle).get("event_type"),
        "general_update",
    )
    fact_basis = _normalized_fact_basis(
        integrated_issue.get("fact_basis"),
        consolidated_facts=consolidated_facts,
        fact_summary=fact_summary,
        source_article_ids=source_article_ids,
    )
    missing_or_uncertain_points = _missing_or_uncertain_points(
        integrated_issue.get("missing_or_uncertain_points"),
        fact_summary=fact_summary,
        fact_basis=fact_basis,
        source_article_ids=source_article_ids,
    )
    confidence = _integration_confidence(
        integrated_issue,
        fact_summary=fact_summary,
        fact_basis=fact_basis,
        missing_or_uncertain_points=missing_or_uncertain_points,
        source_article_ids=source_article_ids,
    )
    headline = _first_non_empty(integrated_issue.get("headline"), main_issue)
    one_line_summary = _first_non_empty(
        integrated_issue.get("one_line_summary"),
        fact_summary[0] if fact_summary else "",
        main_issue,
    )
    display_headline = _clean_display_truncated_fragment(headline)
    display_one_line_summary = _clean_display_truncated_fragment(one_line_summary)
    display_fact_summary = [
        display_line
        for line in fact_summary
        if (display_line := _clean_display_truncated_fragment(line))
    ]
    integrated_article = _integrated_article_from_issue(
        integrated_issue,
        title=headline,
        lead=one_line_summary,
        body_summary_lines=fact_summary,
        display_title=display_headline,
        display_lead=display_one_line_summary,
        display_body_summary_lines=display_fact_summary,
        consolidated_facts=consolidated_facts,
        fact_basis=fact_basis,
        source_article_ids=source_article_ids,
        representative_sources=representative_sources,
    )
    payload = {
        **integrated_issue,
        "integration_schema_version": INTEGRATED_ISSUE_SCHEMA_VERSION,
        "prompt_version": integrated_issue.get("prompt_version") or INTEGRATION_PROMPT_VERSION,
        "issue_component": issue_component,
        "integration_component": issue_component,
        "integration_input": "analysis_input_bundle",
        "bundle_id": input_bundle.bundle_id,
        "issue_source_type": input_bundle.source_type,
        "main_company": main_company,
        "mentioned_peer_companies": mentioned_peer_companies,
        "cluster_event_type": cluster_event_type,
        "main_issue": str(main_issue),
        "headline": headline,
        "one_line_summary": one_line_summary,
        "display_headline": display_headline,
        "display_one_line_summary": display_one_line_summary,
        "display_fact_summary": display_fact_summary,
        "integrated_article": integrated_article,
        "integrated_text": str(integrated_text),
        "fact_summary": fact_summary,
        "consolidated_facts": consolidated_facts,
        "key_numbers": integrated_issue.get("key_numbers", []),
        "business_signals": integrated_issue.get("business_signals", []),
        "representative_sources": representative_sources,
        "fact_basis": fact_basis,
        "missing_or_uncertain_points": missing_or_uncertain_points,
        "source_article_ids": source_article_ids,
        "cluster_article_ids": _dedupe_ints(
            [
                *source_article_ids,
                *[
                    _safe_int(value)
                    for value in _normalize_string_list(integrated_issue.get("cluster_article_ids"))
                ],
            ]
        )
        or source_article_ids,
        "analyzed_article_ids": source_article_ids,
        "summary_scope": integrated_issue.get("summary_scope") or "integrated_issue",
        "confidence": confidence,
        "input_bundle_ref": {
            "bundle_id": input_bundle.bundle_id,
            "cluster_id": input_bundle.cluster_id,
            "source_type": input_bundle.source_type,
            "source_count": len(input_bundle.sources),
        },
        "input_contract_validation": input_contract_validation,
        "preprocessing_outputs": _compact_preprocessing_outputs(preprocessing_outputs),
        "downstream_usage": {
            "card_news": "title/summary_lines/fact_basis/source_article_ids",
            "mixer": "integrated_issue+analysis+implication in evidence_payload",
            "briefing": "card_news.evidence_payload.analysis_package.integrated_issue",
        },
    }
    payload["integration_validation"] = _integration_validation(payload)
    return payload


def _integrated_article_from_issue(
    integrated_issue: dict[str, Any],
    *,
    title: str,
    lead: str,
    body_summary_lines: list[str],
    display_title: str,
    display_lead: str,
    display_body_summary_lines: list[str],
    consolidated_facts: list[dict[str, Any]],
    fact_basis: list[dict[str, Any]],
    source_article_ids: list[int],
    representative_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    raw = integrated_issue.get("integrated_article")
    existing = raw if isinstance(raw, dict) else {}
    key_facts = existing.get("key_facts")
    if not isinstance(key_facts, list) or not key_facts:
        key_facts = _key_facts_from_fact_basis(fact_basis) or _key_facts_from_consolidated_facts(
            consolidated_facts
        )
    return {
        **existing,
        "title": _first_non_empty(existing.get("title"), title),
        "lead": _first_non_empty(existing.get("lead"), lead),
        "body_summary_lines": _normalize_summary_lines(existing.get("body_summary_lines"))
        or body_summary_lines[:5],
        "display_title": _first_non_empty(existing.get("display_title"), display_title),
        "display_lead": _first_non_empty(existing.get("display_lead"), display_lead),
        "display_body_summary_lines": _normalize_summary_lines(
            existing.get("display_body_summary_lines")
        )
        or display_body_summary_lines[:5],
        "key_facts": key_facts,
        "fact_basis": fact_basis,
        "source_article_ids": source_article_ids,
        "representative_sources": representative_sources[:5],
    }


def _key_facts_from_fact_basis(fact_basis: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for item in fact_basis[:8]:
        fact_text = _first_non_empty(item.get("fact"), item.get("evidence_text"))
        if not fact_text:
            continue
        facts.append(
            {
                "fact": fact_text,
                "source_article_ids": _dedupe_ints(
                    [
                        _safe_int(value)
                        for value in _normalize_string_list(item.get("source_article_ids"))
                    ]
                ),
                "fact_ids": _normalize_string_list(item.get("fact_ids")),
                "evidence_text": _first_non_empty(
                    item.get("evidence_text"),
                    *(_normalize_string_list(item.get("evidence_texts"))[:1]),
                    fact_text,
                ),
                "fact_type": item.get("evidence_type") or "reported_fact",
            }
        )
    return facts


def _key_facts_from_consolidated_facts(
    consolidated_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "fact": str(fact.get("fact") or "").strip(),
            "source_article_ids": _dedupe_ints(
                [
                    _safe_int(value)
                    for value in _normalize_string_list(fact.get("source_article_ids"))
                ]
            ),
            "evidence_text": _first_non_empty(
                fact.get("evidence_text"),
                *(_normalize_string_list(fact.get("evidence_texts"))[:1]),
                fact.get("fact"),
            ),
            "fact_type": fact.get("fact_type") or fact.get("source_type") or "reported_fact",
        }
        for fact in consolidated_facts[:8]
        if str(fact.get("fact") or "").strip()
    ]


def _compact_preprocessing_outputs(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    return {
        "raw_article_ids": value.get("raw_article_ids", []),
        "representative_id": value.get("representative_id", 0),
        "cluster_article_ids": value.get("cluster_article_ids", []),
        "matched_companies": value.get("matched_companies", []),
        "matched_sectors": value.get("matched_sectors", []),
        "classification": value.get("classification", {}),
        "business_signal_count": len(value.get("business_signals") or []),
        "financial_metric_count": len(value.get("financial_metrics") or []),
        "parser_output_count": len(value.get("parser_outputs") or []),
    }


def _source_article_ids_for_issue(
    integrated_issue: dict[str, Any],
    input_bundle: AnalysisInputBundle,
) -> list[int]:
    candidates: list[int] = []
    for key in ("source_article_ids", "analyzed_article_ids"):
        candidates.extend(
            _safe_int(value) for value in _normalize_string_list(integrated_issue.get(key))
        )
    if candidates:
        return _dedupe_ints(candidates)
    candidates.extend(
        _safe_int(value)
        for value in _normalize_string_list(integrated_issue.get("cluster_article_ids"))
    )
    candidates.extend(_item_ids(input_bundle.items))
    candidates.extend(
        _safe_int(source.get("article_id"))
        for source in input_bundle.sources
        if isinstance(source, dict)
    )
    return _dedupe_ints(candidates)


def _classification(input_bundle: AnalysisInputBundle) -> dict[str, Any]:
    raw = input_bundle.metadata.get("classification")
    return raw if isinstance(raw, dict) else {}


def _fact_summary_from_issue(
    integrated_issue: dict[str, Any],
    *,
    consolidated_facts: list[dict[str, Any]],
    integrated_text: str,
) -> list[str]:
    lines = _normalize_summary_lines(integrated_issue.get("fact_summary"))
    if not lines:
        lines = _normalize_summary_lines(integrated_issue.get("summary_lines"))
    if not lines:
        lines = [
            str(fact.get("fact") or "").strip()
            for fact in consolidated_facts[:_SUMMARY_LINE_MAX]
            if str(fact.get("fact") or "").strip()
        ]
    if not lines and integrated_text:
        lines = [integrated_text]
    return _dedupe_strings(lines)[:_SUMMARY_LINE_MAX]


def _clean_display_truncated_fragment(text: Any) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\[[^\]]*(?:\.\.\.|…)[^\]]*$", "", value)
    value = re.sub(r"\([^)]*(?:\.\.\.|…)[^)]*$", "", value)
    value = _DISPLAY_TRUNCATED_MARKER_PATTERN.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"[\s\[\(「『\"'·,;:/\\|-]+$", "", value).strip()
    return value


def _normalize_summary_lines(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        lines: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("summary") or item.get("fact") or "")
            else:
                text = str(item or "")
            text = text.strip()
            if text:
                lines.append(text)
        return lines
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _mentioned_peer_companies(
    *,
    integrated_issue: dict[str, Any],
    input_bundle: AnalysisInputBundle,
    main_company: str,
) -> list[str]:
    values = [
        main_company,
        *_normalize_string_list(integrated_issue.get("mentioned_peer_companies")),
        *_normalize_string_list(integrated_issue.get("target_peer_companies")),
        *input_bundle.companies,
    ]
    return _dedupe_strings(values)


def _normalized_fact_basis(
    raw_basis: Any,
    *,
    consolidated_facts: list[dict[str, Any]],
    fact_summary: list[str],
    source_article_ids: list[int],
) -> list[dict[str, Any]]:
    basis: list[dict[str, Any]] = []
    for item in raw_basis if isinstance(raw_basis, list) else []:
        if not isinstance(item, dict):
            continue
        source_ids = _dedupe_ints(
            [_safe_int(value) for value in _normalize_string_list(item.get("source_article_ids"))]
        )
        fact_ids = _normalize_string_list(item.get("fact_ids"))
        evidence_texts = _normalize_string_list(item.get("evidence_texts"))
        evidence_text = _first_non_empty(
            item.get("evidence_text"),
            evidence_texts[0] if evidence_texts else "",
            item.get("fact"),
        )
        line_index = _safe_int(item.get("summary_line_index") or item.get("summary_sentence_index"))
        basis.append(
            {
                **item,
                "summary_line_index": line_index or None,
                "source_article_ids": source_ids,
                "fact_ids": fact_ids,
                "evidence_text": evidence_text,
                "evidence_texts": evidence_texts or ([evidence_text] if evidence_text else []),
                "evidence_type": item.get("evidence_type") or "reported_fact",
            }
        )

    present_indexes = {
        _safe_int(item.get("summary_line_index"))
        for item in basis
        if _safe_int(item.get("summary_line_index")) > 0
        and _normalize_string_list(item.get("source_article_ids"))
    }
    for index, line in enumerate(fact_summary[:_SUMMARY_LINE_MAX], start=1):
        if index in present_indexes:
            continue
        fact = _matching_fact(line, consolidated_facts, index=index)
        source_ids = _dedupe_ints(
            [
                *[
                    _safe_int(value)
                    for value in _normalize_string_list(fact.get("source_article_ids"))
                ],
                *(source_article_ids[:1] if fact else []),
            ]
        )
        evidence_texts = _normalize_string_list(fact.get("evidence_texts")) if fact else []
        evidence_text = _first_non_empty(
            evidence_texts[0] if evidence_texts else "",
            fact.get("evidence_text") if fact else "",
            fact.get("fact") if fact else "",
            line if source_ids else "",
        )
        if not source_ids or not evidence_text:
            continue
        basis.append(
            {
                "summary_line_index": index,
                "source_article_ids": source_ids,
                "fact_ids": _normalize_string_list(fact.get("fact_id")) if fact else [],
                "evidence_text": evidence_text,
                "evidence_texts": evidence_texts or [evidence_text],
                "evidence_type": fact.get("fact_type") if fact else "reported_fact",
            }
        )
    return _dedupe_fact_basis(basis)


def _matching_fact(
    line: str,
    facts: list[dict[str, Any]],
    *,
    index: int,
) -> dict[str, Any]:
    if not facts:
        return {}
    normalized_line = str(line or "").strip()
    for fact in facts:
        fact_text = str(fact.get("fact") or "").strip()
        if fact_text and (fact_text in normalized_line or normalized_line in fact_text):
            return fact
    return facts[min(index - 1, len(facts) - 1)]


def _dedupe_fact_basis(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[int, ...], str]] = set()
    for item in items:
        line_index = _safe_int(item.get("summary_line_index"))
        source_ids = tuple(
            _safe_int(value) for value in _normalize_string_list(item.get("source_article_ids"))
        )
        evidence_text = str(item.get("evidence_text") or "").strip()
        key = (line_index, source_ids, evidence_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _missing_or_uncertain_points(
    raw_points: Any,
    *,
    fact_summary: list[str],
    fact_basis: list[dict[str, Any]],
    source_article_ids: list[int],
) -> list[dict[str, Any]]:
    points = (
        [item for item in raw_points if isinstance(item, dict)]
        if isinstance(raw_points, list)
        else []
    )
    missing_indexes = _missing_fact_basis_line_indexes(
        fact_summary=fact_summary,
        fact_basis=fact_basis,
    )
    for index in missing_indexes:
        points.append(
            {
                "type": "missing_fact_basis",
                "summary_line_index": index,
                "description": "summary line has no source-backed fact_basis",
            }
        )
    if not source_article_ids:
        points.append(
            {
                "type": "missing_source_article_ids",
                "description": "integrated issue has no raw article source ids",
            }
        )
    return points


def _integration_confidence(
    integrated_issue: dict[str, Any],
    *,
    fact_summary: list[str],
    fact_basis: list[dict[str, Any]],
    missing_or_uncertain_points: list[dict[str, Any]],
    source_article_ids: list[int],
) -> float:
    raw = _safe_float(integrated_issue.get("confidence"), 0.75)
    if not fact_summary:
        raw -= 0.3
    if _missing_fact_basis_line_indexes(fact_summary=fact_summary, fact_basis=fact_basis):
        raw -= 0.2
    if not source_article_ids:
        raw -= 0.2
    if missing_or_uncertain_points:
        raw -= min(0.15, 0.03 * len(missing_or_uncertain_points))
    return round(max(0.0, min(raw, 1.0)), 2)


def _integration_validation(payload: dict[str, Any]) -> dict[str, Any]:
    missing_required = [
        field
        for field in INTEGRATED_ISSUE_REQUIRED_FIELDS
        if _is_missing_required_value(payload.get(field))
    ]
    missing_fact_basis = _missing_fact_basis_line_indexes(
        fact_summary=_normalize_summary_lines(payload.get("fact_summary")),
        fact_basis=payload.get("fact_basis") or [],
    )
    warnings: list[str] = []
    summary_line_count = len(_normalize_summary_lines(payload.get("fact_summary")))
    if not _SUMMARY_LINE_MIN <= summary_line_count <= _SUMMARY_LINE_MAX:
        warnings.append("fact_summary line count must be 3~5")
    if missing_fact_basis:
        warnings.append(
            "fact_basis missing for summary_line_index: "
            + ", ".join(str(index) for index in missing_fact_basis)
        )
    if missing_required:
        warnings.append("required fields missing: " + ", ".join(missing_required))
    return {
        "pass": not missing_required
        and not missing_fact_basis
        and _SUMMARY_LINE_MIN <= summary_line_count <= _SUMMARY_LINE_MAX,
        "schema_version": INTEGRATED_ISSUE_SCHEMA_VERSION,
        "prompt_version": payload.get("prompt_version"),
        "missing_required_fields": missing_required,
        "missing_fact_basis_line_indexes": missing_fact_basis,
        "summary_line_count": summary_line_count,
        "source_article_ids_count": len(_normalize_string_list(payload.get("source_article_ids"))),
        "warnings": warnings,
    }


def _is_missing_required_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not bool(value.strip())
    if isinstance(value, list | tuple | dict | set):
        return not bool(value)
    return False


def _missing_fact_basis_line_indexes(
    *,
    fact_summary: list[str],
    fact_basis: list[dict[str, Any]],
) -> list[int]:
    if not _SUMMARY_LINE_MIN <= len(fact_summary) <= _SUMMARY_LINE_MAX:
        return []
    expected = set(range(1, len(fact_summary) + 1))
    present = {
        _safe_int(item.get("summary_line_index") or item.get("summary_sentence_index"))
        for item in fact_basis
        if isinstance(item, dict)
        and _safe_int(item.get("summary_line_index") or item.get("summary_sentence_index"))
        in expected
        and _normalize_string_list(item.get("source_article_ids"))
    }
    return sorted(expected - present)


def _consolidated_facts_from_bundle(input_bundle: AnalysisInputBundle) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for fact in input_bundle.facts:
        fact_text = str(fact.get("fact") or "").strip()
        evidence = str(fact.get("evidence_text") or fact_text).strip()
        if not fact_text and not evidence:
            continue
        facts.append(
            {
                "fact_id": fact.get("fact_id"),
                "fact": fact_text or evidence,
                "source_article_ids": [_safe_int(fact.get("article_id"))]
                if _safe_int(fact.get("article_id"))
                else [],
                "evidence_texts": [evidence] if evidence else [],
                "source_type": fact.get("source_type") or input_bundle.source_type,
                "fact_type": fact.get("fact_type") or "general_fact",
            }
        )
    if facts:
        return facts[:20]
    return [
        {
            "fact_id": f"snippet:{index}",
            "fact": str(snippet.get("text") or "").strip(),
            "source_article_ids": [_safe_int(snippet.get("article_id"))]
            if _safe_int(snippet.get("article_id"))
            else [],
            "evidence_texts": [str(snippet.get("text") or "").strip()],
            "source_type": snippet.get("source_type") or input_bundle.source_type,
            "fact_type": "evidence_snippet",
        }
        for index, snippet in enumerate(input_bundle.evidence_snippets[:10], start=1)
        if str(snippet.get("text") or "").strip()
    ]


def _key_numbers_from_bundle(input_bundle: AnalysisInputBundle) -> list[dict[str, Any]]:
    numbers: list[dict[str, Any]] = []
    for item in input_bundle.items:
        article_id = _safe_int(item.get("id") or item.get("preprocess_id"))
        for metric in _as_dict_list(item.get("financial_metrics")):
            label = str(metric.get("metric_label") or metric.get("metric_name") or "").strip()
            value = (
                metric.get("value_numeric") or metric.get("value_krwbn") or metric.get("value_krw")
            )
            if not label and value is None:
                continue
            numbers.append(
                {
                    "article_id": article_id,
                    "metric_name": metric.get("metric_name"),
                    "metric_label": label,
                    "value": value,
                    "unit": metric.get("unit"),
                    "period": metric.get("period"),
                    "evidence_text": metric.get("evidence_text"),
                }
            )
    return numbers[:20]


def _business_signals_from_bundle(input_bundle: AnalysisInputBundle) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for item in input_bundle.items:
        article_id = _safe_int(item.get("id") or item.get("preprocess_id"))
        for signal in _as_dict_list(item.get("business_signals")):
            summary = str(signal.get("summary") or "").strip()
            evidence = str(signal.get("evidence_text") or summary).strip()
            if not summary and not evidence:
                continue
            signals.append(
                {
                    "article_id": article_id,
                    "business_area": signal.get("business_area"),
                    "signal_type": signal.get("signal_type"),
                    "sentiment": signal.get("sentiment"),
                    "summary": summary or evidence,
                    "evidence_text": evidence,
                    "confidence": signal.get("confidence"),
                }
            )
    return signals[:20]


def _fact_summary_lines(
    *,
    title: str,
    consolidated_facts: list[dict[str, Any]],
    key_numbers: list[dict[str, Any]],
    business_signals: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    if title:
        lines.append(title)
    for signal in business_signals:
        if signal.get("summary"):
            lines.append(str(signal["summary"]))
            break
    for number in key_numbers:
        label = str(number.get("metric_label") or number.get("metric_name") or "").strip()
        value = number.get("value")
        if label and value is not None:
            lines.append(f"{label} {value}")
            break
    for fact in consolidated_facts:
        if len(lines) >= _SUMMARY_LINE_MAX:
            break
        fact_text = str(fact.get("fact") or "").strip()
        if fact_text and fact_text not in lines:
            lines.append(fact_text)
    return lines[:_SUMMARY_LINE_MAX]


def _integrated_text(
    *,
    source_type: str,
    title: str,
    fact_summary: list[str],
    items: list[dict[str, Any]],
) -> str:
    lines = [line for line in fact_summary if line]
    if len(lines) >= 2:
        return " ".join(lines)
    content = " ".join(str(item.get("content") or "")[:500] for item in items[:2]).strip()
    parts = [part for part in (title, content) if part]
    if parts:
        return " ".join(parts)
    return f"{source_type} 분석 대상"


def _fact_basis_from_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    basis: list[dict[str, Any]] = []
    for index, fact in enumerate(facts[:_SUMMARY_LINE_MAX], start=1):
        basis.append(
            {
                "summary_line_index": index,
                "source_article_ids": fact.get("source_article_ids", []),
                "fact_ids": [fact.get("fact_id")] if fact.get("fact_id") else [],
                "evidence_text": " ".join(_normalize_string_list(fact.get("evidence_texts"))),
                "evidence_texts": _normalize_string_list(fact.get("evidence_texts")),
                "evidence_type": "reported_fact",
            }
        )
    return basis


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _representative_id(input_bundle: AnalysisInputBundle) -> int:
    value = input_bundle.metadata.get("representative_id")
    if (representative_id := _safe_int(value)) > 0:
        return representative_id
    item_ids = _item_ids(input_bundle.items)
    return item_ids[0] if item_ids else 0


def _item_ids(items: list[dict[str, Any]]) -> list[int]:
    return [
        item_id
        for item_id in (_safe_int(item.get("id") or item.get("preprocess_id")) for item in items)
        if item_id > 0
    ]


def _dominant_source_type(items: list[dict[str, Any]]) -> str:
    source_types = _dedupe_strings(
        [str(item.get("source_type") or "").strip() for item in items if item.get("source_type")]
    )
    if not source_types:
        return "unknown"
    counts = {source_type: 0 for source_type in source_types}
    for item in items:
        source_type = str(item.get("source_type") or "").strip()
        if source_type in counts:
            counts[source_type] += 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _facts_from_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for article in articles:
        article_id = _safe_int(article.get("id") or article.get("preprocess_id"))
        title = str(article.get("title") or "").strip()
        if not title:
            continue
        facts.append(
            {
                "fact_id": f"article:{article_id}:title"
                if article_id
                else f"title:{len(facts) + 1}",
                "article_id": article_id,
                "fact": title,
                "evidence_text": title,
                "source_type": article.get("source_type"),
            }
        )
        facts.extend(_facts_from_structured_rows(article, article_id))
    return facts


def _facts_from_structured_rows(article: dict[str, Any], article_id: int) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, metric in enumerate(_as_dict_list(article.get("financial_metrics")), start=1):
        label = str(metric.get("metric_label") or metric.get("metric_name") or "").strip()
        value = metric.get("value_numeric") or metric.get("value_krwbn") or metric.get("value_krw")
        evidence = str(metric.get("evidence_text") or "").strip()
        if not label and value is None and not evidence:
            continue
        fact_text = " ".join(str(part) for part in (label, value) if part not in {None, ""})
        facts.append(
            {
                "fact_id": f"article:{article_id}:metric:{index}",
                "article_id": article_id,
                "fact": fact_text or evidence,
                "evidence_text": evidence or fact_text,
                "source_type": article.get("source_type"),
                "fact_type": "financial_metric",
            }
        )
    for index, signal in enumerate(_as_dict_list(article.get("business_signals")), start=1):
        summary = str(signal.get("summary") or "").strip()
        evidence = str(signal.get("evidence_text") or "").strip()
        if not summary and not evidence:
            continue
        facts.append(
            {
                "fact_id": f"article:{article_id}:business_signal:{index}",
                "article_id": article_id,
                "fact": summary or evidence,
                "evidence_text": evidence or summary,
                "source_type": article.get("source_type"),
                "fact_type": "business_signal",
            }
        )
    return facts


def _evidence_snippets(
    items: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    snippets = [
        {
            "article_id": fact.get("article_id"),
            "text": fact.get("evidence_text") or fact.get("fact"),
            "source_type": fact.get("source_type"),
        }
        for fact in facts
        if fact.get("evidence_text") or fact.get("fact")
    ]
    if snippets:
        return snippets[:20]
    return [
        {
            "article_id": _safe_int(item.get("id") or item.get("preprocess_id")),
            "text": str(item.get("title") or item.get("content") or "")[:500],
            "source_type": item.get("source_type"),
        }
        for item in items[:20]
    ]


def _sources_from_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "article_id": _safe_int(article.get("id") or article.get("preprocess_id")),
            "title": article.get("title"),
            "url": article.get("url"),
            "source_name": article.get("source_name") or article.get("publisher"),
            "source_type": article.get("source_type"),
            "published_at": article.get("published_at"),
            "collected_at": article.get("collected_at"),
        }
        for article in articles
    ]


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    return [stripped] if stripped else []


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "IntegrationAgent",
    "analysis_input_bundle_from_articles",
    "analysis_input_bundle_from_bundle",
]
