"""Backward-compat shim for the old analysis supervisor module name.

New code should import ``AnalysisGraphRunner`` from
``src.agents.analysis_graph_runner``. The old names remain available so legacy
callers keep working during the rename.
"""

from __future__ import annotations

from src.agents.analysis_graph_runner import (
    AnalysisGraphRunner,
    AnalysisSupervisorAgent,
    DataAnalysisSupervisorAgent,
    ProfileContext,
)

__all__ = [
    "AnalysisGraphRunner",
    "AnalysisSupervisorAgent",
    "DataAnalysisSupervisorAgent",
    "ProfileContext",
]
