"""Backward-compat shim for the old IssueIntegrationAgent module name.

New code should import from ``src.agents.integration_agent`` and use
``IntegrationAgent``. The old class name is kept as an alias so existing
callers do not break during the rename.
"""

from __future__ import annotations

from src.agents.integration_agent import (
    IntegrationAgent,
    analysis_input_bundle_from_articles,
    analysis_input_bundle_from_bundle,
)

IssueIntegrationAgent = IntegrationAgent

__all__ = [
    "IntegrationAgent",
    "IssueIntegrationAgent",
    "analysis_input_bundle_from_articles",
    "analysis_input_bundle_from_bundle",
]
