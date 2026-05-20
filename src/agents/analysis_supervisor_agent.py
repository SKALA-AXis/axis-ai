"""1단계 데이터 분석 Supervisor Agent.

DataAnalysisSupervisorAgent는 전체 시스템 최상위 supervisor가 아니라,
런타임 AnalysisInputBundle 하나를 기준으로 이슈 통합, 분석, 프로필 context,
시사점 생성을 조율하고 AnalysisPackage를 만든다.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.analysis_agent import AnalysisAgent
from src.agents.implication_agent import ImplicationAgent
from src.agents.issue_integration_agent import (
    IssueIntegrationAgent,
    analysis_input_bundle_from_articles,
)
from src.agents.profile_agent import ProfileAgent
from src.analysis.models import AnalysisInputBundle, AnalysisPackage
from src.db.article_store import get_articles_by_ids

log = logging.getLogger(__name__)


class DataAnalysisSupervisorAgent:
    """AnalysisInputBundle 기반 1단계 분석 workflow supervisor."""

    def __init__(
        self,
        *,
        issue_integrator: IssueIntegrationAgent | None = None,
        summarizer: IssueIntegrationAgent | None = None,
        analyzer: AnalysisAgent | None = None,
        profile_agent: ProfileAgent | None = None,
        implication_generator: ImplicationAgent | None = None,
    ) -> None:
        self.issue_integrator = issue_integrator or summarizer or IssueIntegrationAgent()
        self.analyzer = analyzer or AnalysisAgent()
        self.profile_agent = profile_agent or ProfileAgent()
        self.implication_generator = implication_generator or ImplicationAgent()

    def analyze_cluster(
        self,
        *,
        cluster_id: int,
        representative_id: int,
        classification: dict[str, Any],
        cluster_article_ids: list[int] | None = None,
        articles: list[dict[str, Any]] | None = None,
        peer_profile_context: dict[str, Any] | None = None,
        skax_profile_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """클러스터를 AnalysisInputBundle으로 묶은 뒤 AnalysisPackage를 만든다."""
        article_ids = _cluster_fetch_ids(representative_id, cluster_article_ids)
        source_articles = articles if articles is not None else get_articles_by_ids(article_ids)
        input_bundle = analysis_input_bundle_from_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=source_articles,
            cluster_article_ids=cluster_article_ids,
            classification=classification,
        )
        return self.analyze_input_bundle(
            input_bundle=input_bundle,
            classification=classification,
            peer_profile_context=peer_profile_context,
            skax_profile_context=skax_profile_context,
        ).to_dict()

    def analyze_input_bundle(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        classification: dict[str, Any] | None = None,
        peer_profile_context: dict[str, Any] | None = None,
        skax_profile_context: dict[str, Any] | None = None,
    ) -> AnalysisPackage:
        """AnalysisInputBundle을 공통 입력으로 이슈 통합, 분석, 시사점을 생성한다."""
        classification = classification or input_bundle.metadata.get("classification", {})
        profile_context = _profile_context(
            classification=classification,
            input_bundle=input_bundle,
            profile_agent=self.profile_agent,
            peer_profile_context=peer_profile_context,
            skax_profile_context=skax_profile_context,
        )
        skax_profile_context = profile_context.get("skax_profile") or skax_profile_context
        peer_profile_context = profile_context.get("peer_profiles") or peer_profile_context
        integrated_issue = self.issue_integrator.integrate_input_bundle(input_bundle)
        cluster_metadata = _cluster_metadata_from_input_bundle(
            input_bundle=input_bundle,
            peer_profile_context=peer_profile_context,
            skax_profile_context=skax_profile_context,
        )
        analysis = self.analyzer.analyze(
            integrated_issue=integrated_issue,
            classification=classification,
            cluster_metadata=cluster_metadata,
        )
        implication = self.implication_generator.generate(
            summary=integrated_issue,
            analysis=analysis,
            classification=classification,
            input_bundle=input_bundle,
            profile_context=profile_context,
            peer_profile_context=peer_profile_context,
            skax_profile_context=skax_profile_context,
        )
        validation = _analysis_package_validation(
            integrated_issue=integrated_issue,
            analysis=analysis,
        )
        package = AnalysisPackage(
            bundle_id=input_bundle.bundle_id,
            input_bundle=input_bundle,
            integrated_issue=integrated_issue,
            analysis=analysis,
            implication=implication,
            sources=input_bundle.sources,
            validation={
                **validation,
                "classification": classification,
                "profile_context_available": {
                    "peer": bool(peer_profile_context),
                    "skax": bool(skax_profile_context),
                },
                "provenance": {
                    "supervisor": "DataAnalysisSupervisorAgent",
                    "issue_component": "IssueIntegrationAgent",
                    "analysis_component": "AnalysisAgent",
                    "implication_component": "ImplicationAgent",
                },
            },
        )
        log.info(
            "분석 supervisor 완료 | cluster=%s valid_integrated_issue=%s valid_analysis=%s",
            input_bundle.cluster_id,
            integrated_issue.get("is_valid_summary"),
            analysis.get("is_valid_analysis"),
        )
        return package


class AnalysisSupervisorAgent(DataAnalysisSupervisorAgent):
    """Backward compatible alias for DataAnalysisSupervisorAgent."""


def _cluster_fetch_ids(
    representative_id: int,
    cluster_article_ids: list[int] | None,
) -> list[int]:
    ids = list(cluster_article_ids or [])
    if representative_id not in ids:
        ids.insert(0, representative_id)
    return ids


def _cluster_metadata_from_input_bundle(
    *,
    input_bundle: AnalysisInputBundle,
    peer_profile_context: dict[str, Any] | None,
    skax_profile_context: dict[str, Any] | None,
) -> dict[str, Any]:
    source_names = sorted(
        {
            str(source.get("source_name"))
            for source in input_bundle.sources
            if source.get("source_name")
        }
    )
    source_types = sorted(
        {
            str(source.get("source_type"))
            for source in input_bundle.sources
            if source.get("source_type")
        }
    )
    return {
        "bundle_id": input_bundle.bundle_id,
        "cluster_id": input_bundle.cluster_id,
        "source_type": input_bundle.source_type,
        "companies": input_bundle.companies,
        "sectors": input_bundle.sectors,
        "event_type": input_bundle.event_type,
        "cluster_size": len(input_bundle.items),
        "source_count": len(source_names),
        "source_names": source_names,
        "source_types": source_types,
        "has_peer_profile_context": bool(peer_profile_context),
        "has_skax_profile_context": bool(skax_profile_context),
    }


def _profile_context(
    *,
    classification: dict[str, Any],
    input_bundle: AnalysisInputBundle,
    profile_agent: ProfileAgent,
    peer_profile_context: dict[str, Any] | None,
    skax_profile_context: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        return profile_agent.build_context(
            companies=input_bundle.companies,
            sectors=_sector_ids(classification) or input_bundle.sectors,
            event_type=input_bundle.event_type or classification.get("event_type"),
            peer_profile_context=peer_profile_context,
            skax_profile_context=skax_profile_context,
        )
    except Exception as exc:
        log.warning("ProfileAgent context 생성 실패 | error=%s", exc)
        return {
            "skax_profile": skax_profile_context or {},
            "peer_profiles": peer_profile_context or {},
            "sector_context": {"selected_sector_ids": _sector_ids(classification)},
        }


def _sector_ids(classification: dict[str, Any]) -> list[str]:
    values: list[str] = []
    sector = classification.get("sector")
    if sector:
        values.append(str(sector))
    sectors = classification.get("sectors")
    if isinstance(sectors, list):
        values.extend(str(item) for item in sectors if item)
    return list(dict.fromkeys(values))


def _analysis_package_validation(
    *,
    integrated_issue: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pass": bool(integrated_issue.get("is_valid_summary", True))
        and bool(analysis.get("is_valid_analysis", True)),
        "integrated_issue_valid": bool(integrated_issue.get("is_valid_summary", True)),
        "analysis_valid": bool(analysis.get("is_valid_analysis", True)),
    }
