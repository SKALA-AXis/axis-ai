"""AnalysisInputBundle 기반 이슈 통합 Agent.

IssueIntegrationAgent는 source_type별 dispatcher가 아니다. 데이터 유형별 차이는
Parser / Extractor / 전처리 단계에서 처리되고, 이 Agent는 정규화된
AnalysisInputBundle 또는 NormalizedDataBundle을 받아 통합 이슈를 만든다.

기존 호출부와의 호환을 위해 cluster_id / representative_id 기반 메서드는
유지하되, 내부적으로 AnalysisInputBundle을 구성한 뒤 같은 이슈 통합 경로를 사용한다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.analysis.models import AnalysisInputBundle, NormalizedDataBundle
from src.analysis.summarizer import SourceSummarizer
from src.db.article_store import get_articles_by_ids


class IssueIntegrationAgent:
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
            issue_component="IssueIntegrationAgent",
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
        metadata={"collected_at": bundle.collected_at},
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
        },
    )


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
        issue_component="IssueIntegrationAgent",
    )


def _tag_integrated_issue(
    integrated_issue: dict[str, Any],
    *,
    input_bundle: AnalysisInputBundle,
    issue_component: str,
) -> dict[str, Any]:
    if not isinstance(integrated_issue, dict):
        return integrated_issue
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
    return {
        **integrated_issue,
        "issue_component": issue_component,
        "integration_component": issue_component,
        "integration_input": "analysis_input_bundle",
        "bundle_id": input_bundle.bundle_id,
        "issue_source_type": input_bundle.source_type,
        "main_issue": str(main_issue),
        "integrated_text": str(integrated_text),
        "consolidated_facts": consolidated_facts,
        "business_signals": integrated_issue.get("business_signals", []),
        "missing_or_uncertain_points": integrated_issue.get("missing_or_uncertain_points", []),
        "input_bundle_ref": {
            "bundle_id": input_bundle.bundle_id,
            "cluster_id": input_bundle.cluster_id,
            "source_type": input_bundle.source_type,
            "source_count": len(input_bundle.sources),
        },
    }


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
        if len(lines) >= 3:
            break
        fact_text = str(fact.get("fact") or "").strip()
        if fact_text and fact_text not in lines:
            lines.append(fact_text)
    return lines[:3]


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
    for index, fact in enumerate(facts[:3], start=1):
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
        facts.extend(_facts_from_parser_result(article, article_id))
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


def _facts_from_parser_result(article: dict[str, Any], article_id: int) -> list[dict[str, Any]]:
    """Parser 결과의 문서 chunk/topic 신호를 analysis facts 로 승격한다.

    DART/IR/리포트는 제목만으로는 분석 밀도가 낮다. 이미 파서가 만든
    document_chunks/topic_signals/sections 를 통합 이슈의 근거 후보로 사용해
    AnalysisAgent 가 세부 내용을 읽을 수 있게 한다.
    """
    parser_result = _parser_result(article)
    if not parser_result:
        return []

    facts: list[dict[str, Any]] = []
    source_type = article.get("source_type")

    for index, signal in enumerate(_as_dict_list(parser_result.get("topic_signals")), start=1):
        summary = str(
            signal.get("summary")
            or signal.get("topic")
            or signal.get("signal")
            or signal.get("text")
            or ""
        ).strip()
        evidence = str(signal.get("evidence_text") or signal.get("evidence") or summary).strip()
        if not summary and not evidence:
            continue
        facts.append(
            {
                "fact_id": f"article:{article_id}:topic_signal:{index}",
                "article_id": article_id,
                "fact": summary or evidence,
                "evidence_text": evidence or summary,
                "source_type": source_type,
                "fact_type": "business_signal",
                "section_key": signal.get("section_key"),
                "source_chunk_uid": signal.get("chunk_id") or signal.get("source_chunk_uid"),
            }
        )

    for index, chunk in enumerate(_as_dict_list(parser_result.get("document_chunks")), start=1):
        text = _compact_text(chunk.get("text"), limit=360)
        if not text:
            continue
        section_key = str(chunk.get("section_key") or "").strip()
        section_title = str(chunk.get("section_title") or "").strip()
        facts.append(
            {
                "fact_id": f"article:{article_id}:chunk:{chunk.get('chunk_id') or index}",
                "article_id": article_id,
                "fact": f"{section_title}: {text}" if section_title else text,
                "evidence_text": text,
                "source_type": source_type,
                "fact_type": _fact_type_for_section(section_key),
                "section_key": section_key,
                "section_title": section_title,
                "source_chunk_uid": chunk.get("chunk_id"),
            }
        )
        if len(facts) >= 18:
            return facts[:18]

    sections = parser_result.get("sections")
    if isinstance(sections, dict):
        for index, (section_key, section_value) in enumerate(sections.items(), start=1):
            text = _compact_text(section_value, limit=320)
            if not text:
                continue
            facts.append(
                {
                    "fact_id": f"article:{article_id}:section:{section_key}",
                    "article_id": article_id,
                    "fact": text,
                    "evidence_text": text,
                    "source_type": source_type,
                    "fact_type": _fact_type_for_section(str(section_key)),
                    "section_key": str(section_key),
                }
            )
            if index >= 6 or len(facts) >= 18:
                break

    return facts[:18]


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
            "content_type": article.get("content_type"),
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


def _parser_result(article: dict[str, Any]) -> dict[str, Any]:
    parser_result = article.get("parser_result")
    if isinstance(parser_result, dict):
        return parser_result
    metadata = article.get("metadata") or {}
    if isinstance(metadata, dict) and isinstance(metadata.get("parser_result"), dict):
        return metadata["parser_result"]
    return {}


def _compact_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].strip()


def _fact_type_for_section(section_key: str) -> str:
    normalized = section_key.lower()
    if any(token in normalized for token in ("risk", "위험", "uncertain")):
        return "risk_fact"
    if any(token in normalized for token in ("financial", "finance", "재무", "실적")):
        return "financial_metric"
    if any(token in normalized for token in ("business", "segment", "사업", "전략")):
        return "business_signal"
    return "general_fact"


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
