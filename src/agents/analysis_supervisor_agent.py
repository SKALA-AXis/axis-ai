"""Analysis supervisor agent.

Coordinates the reusable analysis flow:
source summary -> strategic analysis -> implication.
"""

from __future__ import annotations

import logging
from typing import Any

from src.analysis.analyzer import StrategicAnalyzer
from src.analysis.implication import ImplicationGenerator
from src.analysis.summarizer import SourceSummarizer
from src.db.article_store import get_articles_by_ids

log = logging.getLogger(__name__)


class AnalysisSupervisorAgent:
    """Supervisor for summary, analysis, and implication generation."""

    def __init__(
        self,
        *,
        summarizer: SourceSummarizer | None = None,
        analyzer: StrategicAnalyzer | None = None,
        implication_generator: ImplicationGenerator | None = None,
    ) -> None:
        self.summarizer = summarizer or SourceSummarizer()
        self.analyzer = analyzer or StrategicAnalyzer()
        self.implication_generator = implication_generator or ImplicationGenerator()

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
        """Run the standard analysis flow for one cluster."""
        article_ids = _cluster_fetch_ids(representative_id, cluster_article_ids)
        source_articles = articles if articles is not None else get_articles_by_ids(article_ids)

        summary = self.summarizer.summarize_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=source_articles,
            cluster_article_ids=cluster_article_ids,
        )
        cluster_metadata = _cluster_metadata(
            articles=source_articles,
            cluster_article_ids=cluster_article_ids,
            peer_profile_context=peer_profile_context,
            skax_profile_context=skax_profile_context,
        )
        analysis = self.analyzer.analyze(
            summary=summary,
            classification=classification,
            cluster_metadata=cluster_metadata,
        )
        implication = self.implication_generator.generate(
            summary=summary,
            analysis=analysis,
            classification=classification,
            peer_profile_context=peer_profile_context,
            skax_profile_context=skax_profile_context,
        )

        result = {
            "cluster_id": cluster_id,
            "representative_id": representative_id,
            "summary": summary,
            "analysis": analysis,
            "implication": implication,
            "classification": classification,
            "cluster_metadata": cluster_metadata,
            "provenance": {
                "supervisor": "AnalysisSupervisorAgent",
                "summary_component": "SourceSummarizer",
                "analysis_component": "StrategicAnalyzer",
                "implication_component": "ImplicationGenerator",
            },
        }
        log.info(
            "분석 supervisor 완료 | cluster=%s valid_summary=%s valid_analysis=%s",
            cluster_id,
            summary.get("is_valid_summary"),
            analysis.get("is_valid_analysis"),
        )
        return result


def _cluster_fetch_ids(
    representative_id: int,
    cluster_article_ids: list[int] | None,
) -> list[int]:
    ids = list(cluster_article_ids or [])
    if representative_id not in ids:
        ids.insert(0, representative_id)
    return ids


def _cluster_metadata(
    *,
    articles: list[dict[str, Any]],
    cluster_article_ids: list[int] | None,
    peer_profile_context: dict[str, Any] | None,
    skax_profile_context: dict[str, Any] | None,
) -> dict[str, Any]:
    source_names = sorted(
        {str(article.get("source_name")) for article in articles if article.get("source_name")}
    )
    source_types = sorted(
        {str(article.get("source_type")) for article in articles if article.get("source_type")}
    )
    return {
        "cluster_size": len(cluster_article_ids or articles),
        "source_count": len(source_names),
        "source_names": source_names,
        "source_types": source_types,
        "has_peer_profile_context": bool(peer_profile_context),
        "has_skax_profile_context": bool(skax_profile_context),
    }
