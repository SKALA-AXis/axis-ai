"""Backward-compat shim — 옛 모듈명 ``src.pipeline.supervisor_graph``.

실제 구현은 ``src.pipeline.analysis_flow_graph`` 로 이동 (외부 리뷰 2026-05-21
R-rename). 기존 import 는 그대로 동작하도록 모든 public symbol 을 재노출한다.

새 코드는 다음을 사용:

    from src.pipeline.analysis_flow_graph import (
        AnalysisFlowState,
        AnalysisFlowDeps,
        build_analysis_flow_graph,
        run_analysis_flow,
    )
"""

from __future__ import annotations

from src.pipeline.analysis_flow_graph import (  # noqa: F401 — re-export
    AnalysisFlowDeps,
    AnalysisFlowState,
    EvaluationMetrics,
    ImplicationResult,
    SupervisorDeps,
    SupervisorState,
    analysis_flow_run_id,
    build_analysis_flow_graph,
    build_supervisor_graph,
    default_analysis_flow_graph,
    default_supervisor_graph,
    run_analysis_flow,
    run_supervisor,
    supervisor_run_id,
)

__all__ = [
    "AnalysisFlowDeps",
    "AnalysisFlowState",
    "EvaluationMetrics",
    "ImplicationResult",
    "SupervisorDeps",
    "SupervisorState",
    "analysis_flow_run_id",
    "build_analysis_flow_graph",
    "build_supervisor_graph",
    "default_analysis_flow_graph",
    "default_supervisor_graph",
    "run_analysis_flow",
    "run_supervisor",
    "supervisor_run_id",
]
