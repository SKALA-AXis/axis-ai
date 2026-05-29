"""AnalysisInputBundle 기반 이슈 통합 Agent.

IssueIntegrationAgent는 source_type별 dispatcher가 아니다. 데이터 유형별 차이는
Parser / Extractor / 전처리 단계에서 처리되고, 이 Agent는 정규화된
AnalysisInputBundle 또는 NormalizedDataBundle을 받아 통합 이슈를 만든다.

기존 호출부와의 호환을 위해 cluster_id / representative_id 기반 메서드는
유지하되, 내부적으로 AnalysisInputBundle을 구성한 뒤 같은 이슈 통합 경로를 사용한다.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from src.analysis.models import AnalysisInputBundle, NormalizedDataBundle
from src.analysis.summarizer import SourceSummarizer
from src.db.article_store import get_articles_by_ids
from src.services.issue_integration import (
    IntegratedIssueComposer,
    companies_from_articles,
    dominant_source_type,
    evidence_snippets,
    extract_facts_from_articles,
    sectors_from_articles,
    sources_from_articles,
)


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
        """뉴스 클러스터는 대표 기사를 분석하고 전체 기사는 출처 참조로 유지한다."""
        eligible_bundle = _eligible_news_bundle(input_bundle)
        if not eligible_bundle.items:
            return IntegratedIssueComposer().compose_non_news(input_bundle)
        analysis_bundle = _representative_news_analysis_bundle(eligible_bundle)
        cluster_id = _safe_int(input_bundle.cluster_id or analysis_bundle.cluster_id)
        representative_id = _representative_id(analysis_bundle)
        cluster_article_ids = _cluster_article_ids(input_bundle)
        integrated_issue = self.engine.summarize_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=analysis_bundle.items,
            cluster_article_ids=cluster_article_ids,
        )
        tagged_issue = _tag_integrated_issue(
            integrated_issue,
            input_bundle=analysis_bundle,
            issue_component="IssueIntegrationAgent",
        )
        return _attach_cluster_source_references(
            tagged_issue,
            source_bundle=input_bundle,
            analysis_bundle=analysis_bundle,
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
        evidence_snippets=evidence_snippets(bundle.items, bundle.facts),
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
    source_type = dominant_source_type(articles)
    companies = companies_from_articles(articles, classification)
    sectors = sectors_from_articles(articles, classification)
    facts = extract_facts_from_articles(articles)
    sources = sources_from_articles(articles)
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
        evidence_snippets=evidence_snippets(articles, facts),
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
    return IntegratedIssueComposer().compose_non_news(input_bundle)


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
    tagged = {
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
    return IntegratedIssueComposer().enrich_existing_summary(tagged, input_bundle=input_bundle)


def _eligible_news_bundle(input_bundle: AnalysisInputBundle) -> AnalysisInputBundle:
    if not any("is_analysis_eligible" in source for source in input_bundle.sources or []):
        return input_bundle
    source_by_id = {
        _safe_int(source.get("raw_article_id") or source.get("article_id")): source
        for source in input_bundle.sources or []
    }
    eligible_items = [
        item
        for item in input_bundle.items
        if _source_eligible(source_by_id.get(_item_id(item)))
    ]
    if len(eligible_items) == len(input_bundle.items):
        return input_bundle
    eligible_ids = {_item_id(item) for item in eligible_items if _item_id(item) > 0}
    return replace(
        input_bundle,
        items=eligible_items,
        facts=[
            fact
            for fact in input_bundle.facts
            if _safe_int(fact.get("raw_article_id") or fact.get("article_id")) in eligible_ids
        ],
        evidence_snippets=[
            snippet
            for snippet in input_bundle.evidence_snippets
            if _safe_int(snippet.get("raw_article_id") or snippet.get("article_id"))
            in eligible_ids
        ],
        sources=[
            source
            for source in input_bundle.sources
            if _safe_int(source.get("raw_article_id") or source.get("article_id")) in eligible_ids
        ],
    )


def _representative_news_analysis_bundle(input_bundle: AnalysisInputBundle) -> AnalysisInputBundle:
    representative_id = _representative_article_id(input_bundle)
    selected_items = [
        item for item in input_bundle.items if _item_id(item) == representative_id
    ]
    if not selected_items and input_bundle.items:
        selected_items = [input_bundle.items[0]]
        representative_id = _item_id(selected_items[0])
    selected_ids = {_item_id(item) for item in selected_items if _item_id(item) > 0}
    return replace(
        input_bundle,
        items=selected_items,
        facts=[
            fact
            for fact in input_bundle.facts
            if _safe_int(fact.get("raw_article_id") or fact.get("article_id")) in selected_ids
        ],
        evidence_snippets=[
            snippet
            for snippet in input_bundle.evidence_snippets
            if _safe_int(snippet.get("raw_article_id") or snippet.get("article_id"))
            in selected_ids
        ],
        sources=[
            source for source in input_bundle.sources if _source_id(source) in selected_ids
        ],
        metadata={
            **input_bundle.metadata,
            "representative_id": representative_id,
        },
    )


def _representative_article_id(input_bundle: AnalysisInputBundle) -> int:
    cluster_id = _safe_int(input_bundle.cluster_id)
    item_ids = _item_ids(input_bundle.items)
    if cluster_id > 0 and cluster_id in item_ids:
        return cluster_id
    if (representative_id := _representative_id(input_bundle)) > 0:
        return representative_id
    for item in input_bundle.items:
        if item.get("is_representative"):
            return _item_id(item)
    return item_ids[0] if item_ids else 0


def _cluster_article_ids(input_bundle: AnalysisInputBundle) -> list[int]:
    metadata_ids = _normalize_int_list(input_bundle.metadata.get("cluster_article_ids"))
    return metadata_ids or _item_ids(input_bundle.items)


def _attach_cluster_source_references(
    integrated_issue: dict[str, Any],
    *,
    source_bundle: AnalysisInputBundle,
    analysis_bundle: AnalysisInputBundle,
) -> dict[str, Any]:
    analyzed_ids = _item_ids(analysis_bundle.items)
    source_refs = _cluster_source_references(source_bundle)
    issue_brief = dict(integrated_issue.get("issue_brief") or {})
    issue_brief["analysis_scope"] = {
        "analyzed_source_ids": analyzed_ids,
    }
    metadata = {
        **(integrated_issue.get("metadata") or {}),
        "cluster_id": source_bundle.cluster_id,
        "source_count": len(source_refs),
        "cluster_source_count": len(source_refs),
        "analyzed_source_count": len(analyzed_ids),
    }
    return {
        **integrated_issue,
        "issue_brief": issue_brief,
        "sources": source_refs,
        "metadata": metadata,
    }


def _cluster_source_references(input_bundle: AnalysisInputBundle) -> list[dict[str, Any]]:
    item_by_id = {_item_id(item): item for item in input_bundle.items if _item_id(item) > 0}
    refs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source in input_bundle.sources or []:
        source_id = _source_id(source)
        if source_id <= 0 or source_id in seen:
            continue
        seen.add(source_id)
        item = item_by_id.get(source_id, {})
        refs.append(_source_reference(source=source, item=item, source_id=source_id))
    for item in input_bundle.items or []:
        item_id = _item_id(item)
        if item_id <= 0 or item_id in seen:
            continue
        seen.add(item_id)
        refs.append(_source_reference(source={}, item=item, source_id=item_id))
    return refs


def _source_reference(
    *,
    source: dict[str, Any],
    item: dict[str, Any],
    source_id: int,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "title": _first_non_empty(source.get("title"), item.get("title")),
        "source_name": _first_non_empty(
            source.get("source_name"),
            item.get("source_name"),
            item.get("publisher"),
        ),
        "source_type": _first_non_empty(source.get("source_type"), item.get("source_type")),
        "publisher": _first_non_empty(source.get("publisher"), item.get("publisher")),
        "published_at": _first_non_empty(source.get("published_at"), item.get("published_at")),
        "url": _first_non_empty(source.get("url"), item.get("url")),
        "relevance_label": _first_non_empty(
            source.get("relevance_label"), item.get("relevance_label")
        ),
        "relevance_score": source.get("relevance_score", item.get("relevance_score")),
    }


def _source_eligible(source: dict[str, Any] | None) -> bool:
    if not source:
        return True
    return bool(source.get("is_analysis_eligible", True))


def _representative_id(input_bundle: AnalysisInputBundle) -> int:
    value = input_bundle.metadata.get("representative_id")
    if (representative_id := _safe_int(value)) > 0:
        return representative_id
    item_ids = _item_ids(input_bundle.items)
    return item_ids[0] if item_ids else 0


def _item_ids(items: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for item in items:
        item_id = _item_id(item)
        if item_id > 0 and item_id not in ids:
            ids.append(item_id)
    return ids


def _item_id(item: dict[str, Any]) -> int:
    return _safe_int(
        item.get("raw_article_id")
        or item.get("id")
        or item.get("preprocess_id")
        or item.get("article_id")
    )


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    return [stripped] if stripped else []


def _normalize_int_list(value: Any) -> list[int]:
    values: list[Any]
    if value is None:
        values = []
    elif isinstance(value, list | tuple | set):
        values = list(value)
    else:
        values = [value]
    return _dedupe_ints([_safe_int(item) for item in values])


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _source_id(source: dict[str, Any]) -> int:
    return _safe_int(source.get("raw_article_id") or source.get("article_id") or source.get("id"))


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
