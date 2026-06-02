"""AnalysisGraphRunner / DataAnalysisSupervisorAgent — Analysis Flow LangGraph 의
얇은 wrapper.

W2-1 이후 본 클래스는 직접 child agent 를 조율하지 않고 LangGraph 상에서 정의된
고정 순서 DAG pipeline (`src.pipeline.supervisor_graph` 의
``build_analysis_flow_graph()``) 을 호출하는 wrapper 다.

명칭 주의 (외부 리뷰 2026-05-21):
* "Supervisor" 이름은 LLM-router 가 worker 를 동적 선택하는 multi-agent supervisor
  pattern 을 의미하지 않는다. 실제는 정적 DAG pipeline.
* 새 코드는 ``AnalysisGraphRunner`` (본 모듈에서 alias 로 노출) 사용을 권장.
* 기존 ``DataAnalysisSupervisorAgent`` / ``AnalysisSupervisorAgent`` 는 backward-compat
  으로 유지.

기존 외부 인터페이스 (``analyze_cluster`` / ``analyze_input_bundle`` → AnalysisPackage)
는 유지하므로 호출부 변경 최소화.
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
from src.analysis.models import (
    AnalysisInputBundle,
    AnalysisPackage,
    ProfileContext,
)
from src.db.article_store import get_articles_by_ids
from src.pipeline.analysis_flow_graph import (
    SupervisorDeps,
    build_supervisor_graph,
    run_supervisor,
)

log = logging.getLogger(__name__)


class DataAnalysisSupervisorAgent:
    """AnalysisInputBundle 기반 1단계 분석 workflow runner (LangGraph DAG wrapper).

    명명 (외부 리뷰 2026-05-21):
    * 정확한 역할 명은 ``AnalysisGraphRunner`` — multi-agent supervisor pattern 이 아닌
      고정 순서 DAG pipeline 의 실행자.
    * 본 클래스명 (``DataAnalysisSupervisorAgent``) 은 backward-compat 으로 유지.
    """

    def __init__(
        self,
        *,
        issue_integrator: IssueIntegrationAgent | None = None,
        summarizer: IssueIntegrationAgent | None = None,
        analyzer: AnalysisAgent | None = None,
        implication_generator: ImplicationAgent | None = None,
    ) -> None:
        deps = SupervisorDeps(
            issue_integrator=issue_integrator or summarizer,
            analyzer=analyzer,
            implication_agent=implication_generator,
        )
        self._deps = deps
        self._graph = build_supervisor_graph(deps)

    # Backwards-compat attributes (legacy callers reach inside).
    @property
    def issue_integrator(self) -> IssueIntegrationAgent:
        return self._deps.issue_integrator

    @property
    def analyzer(self) -> AnalysisAgent:
        return self._deps.analyzer

    @property
    def implication_generator(self) -> ImplicationAgent:
        return self._deps.implication_agent

    # ──────────────────────────────────────────────────────────────────
    # Public API (backwards-compat)
    # ──────────────────────────────────────────────────────────────────
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
        """뉴스 클러스터를 supervisor graph 에 invoke 하여 AnalysisPackage dict 반환."""
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
        """AnalysisInputBundle 1건을 그래프에 invoke."""
        classification_payload = classification or input_bundle.metadata.get("classification") or {}
        # ProfileContext 외부 주입은 supervisor 내부 build 가 우선이지만, 기존 호출자가
        # 외부 context 를 함께 전달했다면 그래프 input 에 함께 실어준다.
        state = run_supervisor(
            input_bundle=input_bundle,
            classification=classification_payload,
            graph=self._graph,
        )
        pkg = state.get("analysis_package")
        if pkg is None:
            log.info(
                "analyze_input_bundle | bundle=%s validation failed → no package",
                input_bundle.bundle_id,
            )
            validation_state = state.get("validation")
            validation_dict: dict[str, Any] = (
                validation_state.to_dict()
                if validation_state is not None
                else {"pass": False, "passed": False}
            )
            return AnalysisPackage(
                bundle_id=input_bundle.bundle_id,
                input_bundle=input_bundle,
                integrated_issue=state.get("integrated_issue") or {},
                analysis=state.get("analysis") or {},
                implication=state.get("implication") or {},
                sources=list(input_bundle.sources or []),
                validation=validation_dict,
                classification=classification_payload,
            )
        return pkg


class AnalysisSupervisorAgent(DataAnalysisSupervisorAgent):
    """Backward compatible alias for DataAnalysisSupervisorAgent."""


class AnalysisGraphRunner(DataAnalysisSupervisorAgent):
    """권장 이름 (외부 리뷰 R-rename 2026-05-21).

    실제 구현은 ``DataAnalysisSupervisorAgent`` 와 동일. 새 코드는 본 클래스명을 사용.
    """


def _cluster_fetch_ids(
    representative_id: int,
    cluster_article_ids: list[int] | None,
) -> list[int]:
    ids = list(cluster_article_ids or [])
    if representative_id not in ids:
        ids.insert(0, representative_id)
    return ids


__all__ = [
    "AnalysisGraphRunner",
    "AnalysisSupervisorAgent",
    "DataAnalysisSupervisorAgent",
    "ProfileContext",
]
