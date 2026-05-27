"""Issue integration services."""

from src.services.issue_integration.fact_extractor import (
    companies_from_articles,
    dominant_source_type,
    evidence_snippets,
    extract_facts_from_articles,
    sectors_from_articles,
    sources_from_articles,
)
from src.services.issue_integration.issue_composer import IntegratedIssueComposer
from src.services.issue_integration.policy import IntegrationPolicy

__all__ = [
    "IntegratedIssueComposer",
    "IntegrationPolicy",
    "companies_from_articles",
    "dominant_source_type",
    "evidence_snippets",
    "extract_facts_from_articles",
    "sectors_from_articles",
    "sources_from_articles",
]
