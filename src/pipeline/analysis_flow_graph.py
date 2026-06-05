"""Analysis Flow LangGraph DAG — W2-1 + W2-3 + W4-2 + W4-5 + W5-1.

명칭 (외부 리뷰 2026-05-21 R-rename):
* 이 모듈의 구조는 LLM-router 가 worker 를 동적 선택하는 multi-agent supervisor pattern
  이 아닌 **고정 순서 DAG pipeline** 이다.
* 권장 이름은 ``AnalysisFlow*`` / ``build_analysis_flow_graph`` (본 모듈 하단 alias).
  기존 ``Supervisor*`` 이름은 backward-compat 으로 유지.

design/01-analysis-pipeline-implementation-plan.md §3.1 + 외부 리뷰 (2026-05-21) 반영:

    issue_integrate  →  profile_context  →  build_analysis_context
        →  strategic_insight  →  validate
        → pass → assemble → card_writer → END
        → fail → human_review → END

순서 변경 근거 (외부 리뷰 R-1):
* IntegratedIssue 가 만들어진 후 main_company / mentioned_peer_companies 가 확정되어야
  ProfileContext / AnalysisContext 조회가 정확해진다.
* 잘못 매칭된 cluster (is_valid_summary=False) 가 ① 단계에서 즉시 차단되어 profile /
  context build 의 DB query 비용을 절약.

각 노드는 `_logged_step` 데코레이터로 `pipeline_logs` 에 elapsed_ms 기록.
LangGraph 의 RetryPolicy 정식 도입은 별도 PR (W3-4 trace + retry 묶음).
"""

from __future__ import annotations

import logging
import operator
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Callable, TypedDict, cast

from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

from src.agents.implication_agent import ImplicationAgent
from src.agents.integration_agent import IntegrationAgent
from src.agents.strategic_analyzer import StrategicAnalyzer
from src.agents.strategic_insight_agent import StrategicInsightAgent
from src.analysis.implication import ImplicationGenerator
from src.analysis.models import (
    AnalysisContext,
    AnalysisInputBundle,
    AnalysisPackage,
    AnalysisResult,
    EvaluationMetrics,
    ImplicationResult,
    NumericViolation,
    ProfileContext,
    ValidationReport,
)
from src.composers.card_news_composer import CardNewsComposer
from src.db.article_store import save_card_news, save_pipeline_log
from src.db.integrated_issues import save_integrated_issue
from src.evaluators.evaluator import Evaluator
from src.services.agent_output_validation import confidence_in_range
from src.services.analysis_context_builder import AnalysisContextBuilder
from src.services.profile_context_loader import ProfileContextLoader

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────────────────────────────────────


class NodeError(TypedDict, total=False):
    node: str
    error_type: str
    message: str
    occurred_at: str


class SupervisorState(TypedDict, total=False):
    # inputs
    input_bundle: AnalysisInputBundle
    classification: dict[str, Any]
    # intermediate
    profile_context: ProfileContext | None
    analysis_context: AnalysisContext | None
    integrated_issue: dict[str, Any] | None
    analysis: dict[str, Any] | None
    implication: dict[str, Any] | None
    validation: ValidationReport | None
    analysis_package: AnalysisPackage | None
    card_news_id: str | None
    card_news_payload: dict[str, Any] | None
    rolling_confidence_avg: float | None
    # outputs / observability
    errors: Annotated[list[NodeError], operator.add]
    human_review_flags: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Logging decorator
# ─────────────────────────────────────────────────────────────────────────────


def _logged_step(step_name: str) -> Callable[[Callable], Callable]:
    """노드 함수 wrapper — elapsed_ms 와 error 를 pipeline_logs 에 자동 기록."""

    def decorator(func: Callable) -> Callable:
        def wrapper(state: SupervisorState) -> SupervisorState:
            t0 = time.perf_counter()
            err: str | None = None
            bundle = state.get("input_bundle")
            company_label = ""
            if isinstance(bundle, AnalysisInputBundle) and bundle.companies:
                company_label = bundle.companies[0]
            try:
                new_state = func(state)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                log.exception("supervisor node 실패 | step=%s", step_name)
                new_errors = list(state.get("errors") or [])
                new_errors.append(
                    {
                        "node": step_name,
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:500],
                        "occurred_at": datetime.now(UTC).isoformat(),
                    }
                )
                return cast(SupervisorState, {**state, "errors": new_errors})
            finally:
                elapsed = int((time.perf_counter() - t0) * 1000)
                save_pipeline_log(
                    step=f"supervisor.{step_name}",
                    company=company_label or None,
                    input_count=1,
                    output_count=1,
                    elapsed_ms=elapsed,
                    error_msg=err,
                )
            return new_state

        return wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Default agent dependencies (lazy singletons via build_supervisor_graph)
# ─────────────────────────────────────────────────────────────────────────────


class SupervisorDeps:
    """Supervisor 가 의존하는 fixed-step agents + evaluator + context builders.

    테스트 시 각 의존성을 mock 으로 교체 가능. 본 모듈은 `build_supervisor_graph()`
    가 만든 단일 인스턴스를 재사용.
    """

    def __init__(
        self,
        *,
        integration_agent: IntegrationAgent | None = None,
        issue_integrator: IntegrationAgent | None = None,
        analyzer: StrategicAnalyzer | None = None,
        implication_agent: ImplicationAgent | None = None,
        strategic_insight_agent: StrategicInsightAgent | Any | None = None,
        evaluator: Evaluator | None = None,
        context_builder: AnalysisContextBuilder | None = None,
        profile_context_loader: ProfileContextLoader | None = None,
        card_news_composer: CardNewsComposer | Any | None = None,
        implication_fallback: ImplicationGenerator | None = None,
    ) -> None:
        legacy_agents_supplied = analyzer is not None or implication_agent is not None
        self.integration_agent = integration_agent or issue_integrator or IntegrationAgent()
        self.issue_integrator = self.integration_agent
        self.analyzer = analyzer or StrategicAnalyzer()
        self.implication_agent = implication_agent or ImplicationAgent(
            fallback=implication_fallback
        )
        self.strategic_insight_agent = strategic_insight_agent or (
            _LegacyStrategicInsightAdapter(self.analyzer, self.implication_agent)
            if legacy_agents_supplied
            else StrategicInsightAgent(
                fallback_analyzer=self.analyzer,
                fallback_implication_agent=self.implication_agent,
            )
        )
        self.evaluator = evaluator or Evaluator()
        self.context_builder = context_builder or AnalysisContextBuilder()
        self.profile_context_loader = profile_context_loader or ProfileContextLoader()
        self.card_news_composer = card_news_composer or CardNewsComposer()


class _LegacyStrategicInsightAdapter:
    """Test/backward-compat adapter that preserves separate legacy calls."""

    def __init__(self, analyzer: StrategicAnalyzer, implication_agent: ImplicationAgent) -> None:
        self._analyzer = analyzer
        self._implication_agent = implication_agent

    def generate(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: ProfileContext | None,
        analysis_context: AnalysisContext | None,
        cluster_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        analysis = self._analyzer.analyze(
            integrated_issue=integrated_issue,
            classification=classification,
            cluster_metadata=cluster_metadata,
        )
        implication = self._implication_agent.generate(
            input_bundle=input_bundle,
            integrated_issue=integrated_issue,
            analysis=AnalysisResult.from_dict(analysis).to_dict(),
            profile_context=profile_context,
            analysis_context=analysis_context,
            classification=classification,
        )
        return {
            "is_valid_strategic_insight": bool(
                analysis.get("is_valid_analysis") and implication.get("is_valid_implication")
            ),
            "analysis": analysis,
            "implication": implication,
            "fallback_reason": "legacy_analyzer_implication",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node implementations
# ─────────────────────────────────────────────────────────────────────────────


def _make_nodes(deps: SupervisorDeps) -> dict[str, Callable[[SupervisorState], SupervisorState]]:
    @_logged_step("issue_integrate")
    def issue_integrate_node(state: SupervisorState) -> SupervisorState:
        bundle = state["input_bundle"]
        integrated = deps.issue_integrator.integrate_input_bundle(bundle)
        return cast(SupervisorState, {**state, "integrated_issue": integrated})

    @_logged_step("profile_context")
    def profile_context_node(state: SupervisorState) -> SupervisorState:
        """IntegratedIssue 가 확정한 main_company / mentioned_peer_companies 우선 사용."""
        existing = state.get("profile_context")
        if isinstance(existing, ProfileContext) and (
            existing.peer_profiles or existing.skax_profile
        ):
            return state

        bundle = state["input_bundle"]
        integrated = state.get("integrated_issue") or {}
        sectors = list(bundle.sectors or [])
        companies = _companies_for_context(bundle=bundle, integrated_issue=integrated)
        try:
            ctx = deps.profile_context_loader.load(
                companies=companies,
                sectors=sectors,
                event_type=bundle.event_type,
            )
            return cast(SupervisorState, {**state, "profile_context": ctx})
        except Exception as exc:  # noqa: BLE001
            log.warning("profile_context load 실패, empty fallback | error=%s", exc)
            ctx = ProfileContext(
                skax_profile={},
                peer_profiles={company_id: {"company_id": company_id} for company_id in companies},
                sector_context={"selected_sector_ids": sectors},
            )
            return cast(SupervisorState, {**state, "profile_context": ctx})

    @_logged_step("build_analysis_context")
    def build_analysis_context_node(state: SupervisorState) -> SupervisorState:
        bundle = state["input_bundle"]
        profile_context = state.get("profile_context")
        integrated = state.get("integrated_issue") or {}
        ctx = deps.context_builder.build(
            input_bundle=bundle,
            profile_context=profile_context,
            integrated_issue=integrated,
        )
        return cast(SupervisorState, {**state, "analysis_context": ctx})

    @_logged_step("strategic_insight")
    def strategic_insight_node(state: SupervisorState) -> SupervisorState:
        bundle = state["input_bundle"]
        integrated = state.get("integrated_issue") or {}
        classification = state.get("classification") or bundle.metadata.get("classification") or {}
        insight = deps.strategic_insight_agent.generate(
            input_bundle=bundle,
            integrated_issue=integrated,
            classification=classification,
            profile_context=state.get("profile_context"),
            analysis_context=state.get("analysis_context"),
            cluster_metadata=_cluster_metadata(bundle, state.get("profile_context")),
        )
        return cast(
            SupervisorState,
            {
                **state,
                "analysis": insight.get("analysis") or {},
                "implication": insight.get("implication") or {},
            },
        )

    @_logged_step("validate")
    def validate_node(state: SupervisorState) -> SupervisorState:
        bundle = state["input_bundle"]
        integrated = state.get("integrated_issue") or {}
        analysis = state.get("analysis") or {}
        implication = state.get("implication") or {}
        evidence_payload = _evidence_payload_from_state(state)
        report = _hard_validate(
            integrated_issue=integrated,
            analysis=analysis,
            implication=implication,
            evidence_payload=evidence_payload,
            sources=list(bundle.sources or []),
        )
        if report.passed:
            metrics = deps.evaluator.evaluate(
                implication=implication,
                evidence_payload=evidence_payload,
                analysis_context=state.get("analysis_context"),
                integrated_issue=integrated,
                rolling_confidence_avg=state.get("rolling_confidence_avg"),
            )
            report.metrics = metrics
        thresholds = _load_calibrated_thresholds()
        human_flags = list(state.get("human_review_flags") or [])
        if not report.passed:
            human_flags.append(bundle.bundle_id)
        elif thresholds is not None and report.metrics is not None:
            for metric_name, lower in thresholds.items():
                value = getattr(report.metrics, metric_name, None)
                if value is None:
                    continue
                try:
                    if float(value) < float(lower):
                        human_flags.append(f"low_{metric_name}:{bundle.bundle_id}")
                except (TypeError, ValueError):
                    continue
        return cast(
            SupervisorState,
            {**state, "validation": report, "human_review_flags": human_flags},
        )

    @_logged_step("assemble")
    def assemble_node(state: SupervisorState) -> SupervisorState:
        bundle = state["input_bundle"]
        validation = state.get("validation")
        integrated_issue = dict(state.get("integrated_issue") or {})
        evidence_payload = _evidence_payload_from_state(
            {**state, "integrated_issue": integrated_issue}
        )
        integrated_issue_id = save_integrated_issue(integrated_issue, input_bundle=bundle)
        if integrated_issue_id:
            integrated_issue["integrated_issue_id"] = integrated_issue_id
            evidence_payload["integrated_issue_id"] = integrated_issue_id
            evidence_payload["integrated_issue_storage"] = "available"
            evidence_payload["analysis_package"]["integrated_issue_id"] = integrated_issue_id
            evidence_payload["analysis_package"]["integrated_issue"] = integrated_issue
        else:
            evidence_payload["integrated_issue_storage"] = "unavailable"
            provenance = evidence_payload.setdefault("provenance", {})
            if isinstance(provenance, dict):
                provenance["integrated_issue_storage"] = "unavailable"
        package = AnalysisPackage(
            bundle_id=bundle.bundle_id,
            input_bundle=bundle,
            integrated_issue=integrated_issue,
            analysis=state.get("analysis") or {},
            implication=state.get("implication") or {},
            sources=list(bundle.sources or []),
            validation=validation.to_dict() if validation is not None else {},
            evidence_payload=evidence_payload,
            classification=state.get("classification") or {},
        )
        return cast(
            SupervisorState,
            {**state, "integrated_issue": integrated_issue, "analysis_package": package},
        )

    @_logged_step("card_writer")
    def card_writer_node(state: SupervisorState) -> SupervisorState:
        pkg = state.get("analysis_package")
        if pkg is None:
            return cast(SupervisorState, {**state, "card_news_id": None})
        card = deps.card_news_composer.generate_from_analysis_package(
            pkg,
            classification=state.get("classification") or {},
        )
        if not card:
            return cast(SupervisorState, {**state, "card_news_id": None})

        # 외부 리뷰 R-2 (2026-05-21) 반영 — card_news INSERT 가 모든 v2 컬럼을 직접
        # 채우도록 보강. 카드 생성 단계의 책임을 명확히 한다.
        card.setdefault("card_schema_version", "v2")
        bundle = state["input_bundle"]
        integrated = state.get("integrated_issue") or {}
        # peer_company_id FK — main_company (integrated) > input_bundle.companies[0].
        if not card.get("peer_company_id"):
            main_company = str(integrated.get("main_company") or "").strip()
            companies = list(bundle.companies or [])
            card["peer_company_id"] = main_company or (companies[0] if companies else None)
        # primary_keyword_category — classification.sector 또는 카드의 sector.
        if not card.get("primary_keyword_category"):
            classification = state.get("classification") or {}
            card["primary_keyword_category"] = (
                classification.get("sector") or card.get("sector") or None
            )
        # source_raw_article_ids — bundle.items 의 id 들.
        if not card.get("source_raw_article_ids"):
            ids: list[int] = []
            for item in bundle.items or []:
                raw_id = item.get("id") if isinstance(item, dict) else None
                if raw_id is None:
                    continue
                try:
                    ids.append(int(raw_id))
                except (TypeError, ValueError):
                    continue
            card["source_raw_article_ids"] = ids
        # evidence_payload — supervisor 가 만든 in-memory payload.
        if not card.get("evidence_payload"):
            package_payload = getattr(pkg, "evidence_payload", None)
            card["evidence_payload"] = (
                dict(package_payload)
                if isinstance(package_payload, dict) and package_payload
                else _evidence_payload_from_state(state)
            )
        integrated_issue_id = _nested_get(card.get("evidence_payload"), "integrated_issue_id")
        if not integrated_issue_id:
            integrated_issue_id = _nested_get(state.get("integrated_issue"), "integrated_issue_id")
        if integrated_issue_id and not card.get("integrated_issue_id"):
            card["integrated_issue_id"] = str(integrated_issue_id)
        # W5-1 rule-based metric.
        validation = state.get("validation")
        if validation is not None and validation.metrics is not None:
            existing_eval = card.get("evaluation_payload") or {}
            if not isinstance(existing_eval, dict):
                existing_eval = {}
            existing_eval["rule_based"] = validation.metrics.to_dict()
            card["evaluation_payload"] = existing_eval

        card_id = save_card_news(card)
        return cast(
            SupervisorState,
            {**state, "card_news_id": card_id, "card_news_payload": card},
        )

    @_logged_step("human_review")
    def human_review_node(state: SupervisorState) -> SupervisorState:
        bundle = state["input_bundle"]
        log.info(
            "supervisor human_review route | bundle=%s flags=%s",
            bundle.bundle_id,
            state.get("human_review_flags"),
        )
        return state

    return {
        "profile_context": profile_context_node,
        "build_analysis_context": build_analysis_context_node,
        "issue_integrate": issue_integrate_node,
        "strategic_insight": strategic_insight_node,
        "validate": validate_node,
        "assemble": assemble_node,
        "card_writer": card_writer_node,
        "human_review": human_review_node,
    }


def _route_after_validate(state: SupervisorState) -> str:
    validation = state.get("validation")
    if validation is None:
        return "fail"
    return "pass" if validation.passed else "fail"


# ─────────────────────────────────────────────────────────────────────────────
# Hard validation (W2-3) — numeric / certainty / evidence chain
# ─────────────────────────────────────────────────────────────────────────────


def _hard_validate(
    *,
    integrated_issue: dict[str, Any],
    analysis: dict[str, Any],
    implication: dict[str, Any],
    evidence_payload: dict[str, Any],
    sources: list[dict[str, Any]],
) -> ValidationReport:
    from src.agents.implication_agent import CERTAINTY_PATTERN, NUMERIC_TOKEN_PATTERN

    integrated_valid = bool(integrated_issue.get("is_valid_summary", True))
    analysis_valid = bool(analysis.get("is_valid_analysis", True))
    implication_valid = bool(implication.get("is_valid_implication", True))

    text_blocks = {
        **_analysis_text_blocks(analysis),
        **_implication_text_blocks(implication),
    }

    grounded_corpus = _grounded_corpus(
        evidence_payload=evidence_payload,
        integrated_issue=integrated_issue,
        sources=sources,
    )
    grounded_numeric_keys = _grounded_numeric_keys(
        grounded_corpus,
        numeric_pattern=NUMERIC_TOKEN_PATTERN,
    )
    numeric_violations: list[NumericViolation] = []
    seen_violation_tokens: set[str] = set()
    for location, block in text_blocks.items():
        for match in NUMERIC_TOKEN_PATTERN.finditer(block):
            token = match.group(0).strip()
            if not token or token in seen_violation_tokens:
                continue
            seen_violation_tokens.add(token)
            if token in grounded_corpus:
                continue
            if _numeric_token_key(token) in grounded_numeric_keys:
                continue
            numeric_violations.append(NumericViolation(value=token, location=location))

    certainty_warnings: list[str] = []
    for location, block in text_blocks.items():
        certainty_match = CERTAINTY_PATTERN.search(block)
        if certainty_match is not None:
            certainty_warnings.append(f"{location}: {certainty_match.group(0)}")

    evidence_chain_warnings = _check_evidence_chain(
        evidence_payload=evidence_payload, sources=sources, implication=implication
    )

    confidence = confidence_in_range(implication.get("confidence"))
    confidence_warning = confidence < 0.4

    passed = (
        integrated_valid
        and analysis_valid
        and implication_valid
        and not numeric_violations
        and not evidence_chain_warnings
    )
    reason = ""
    if not integrated_valid:
        reason = "integrated_issue.is_valid_summary=false"
    elif not analysis_valid:
        reason = "analysis.is_valid_analysis=false"
    elif not implication_valid:
        reason = "implication.is_valid_implication=false"
    elif numeric_violations:
        reason = f"numeric_violations={[v.value for v in numeric_violations[:3]]} (출처 없는 수치)"
    elif evidence_chain_warnings:
        reason = evidence_chain_warnings[0]

    sc_score = round(
        max(
            0.0,
            1.0
            - 0.1 * len(numeric_violations)
            - 0.05 * len(certainty_warnings)
            - 0.05 * len(evidence_chain_warnings),
        ),
        3,
    )

    return ValidationReport(
        passed=passed,
        integrated_issue_valid=integrated_valid,
        analysis_valid=analysis_valid,
        implication_valid=implication_valid,
        numeric_violations=numeric_violations,
        certainty_warnings=certainty_warnings,
        evidence_chain_warnings=evidence_chain_warnings,
        implication_confidence_warning=confidence_warning,
        sc_score=sc_score,
        reason=reason,
    )


def _implication_text_blocks(implication: dict[str, Any]) -> dict[str, str]:
    skax = implication.get("skax_implication") or {}
    peer = implication.get("peer_implication") or {}
    blocks: dict[str, str] = {}
    if skax.get("why_important"):
        blocks["skax_implication.why_important"] = str(skax["why_important"])
    if skax.get("potential_impact"):
        blocks["skax_implication.potential_impact"] = str(skax["potential_impact"])
    opps = " ".join(str(item) for item in (skax.get("opportunities") or []) if item)
    if opps:
        blocks["skax_implication.opportunities"] = opps
    threats = " ".join(str(item) for item in (skax.get("threats") or []) if item)
    if threats:
        blocks["skax_implication.threats"] = threats
    actions = " ".join(str(item) for item in (skax.get("recommended_actions") or []) if item)
    if actions:
        blocks["skax_implication.recommended_actions"] = actions
    if peer.get("peer_meaning"):
        blocks["peer_implication.peer_meaning"] = str(peer["peer_meaning"])
    if peer.get("capability_change"):
        blocks["peer_implication.capability_change"] = str(peer["capability_change"])
    return blocks


def _analysis_text_blocks(analysis: dict[str, Any]) -> dict[str, str]:
    blocks: dict[str, str] = {}
    if analysis.get("analysis_summary"):
        blocks["analysis.analysis_summary"] = str(analysis["analysis_summary"])
    strategic = " ".join(str(item) for item in (analysis.get("strategic_meaning") or []) if item)
    if strategic:
        blocks["analysis.strategic_meaning"] = strategic
    if analysis.get("market_signal"):
        blocks["analysis.market_signal"] = str(analysis["market_signal"])
    if analysis.get("impact_reason"):
        blocks["analysis.impact_reason"] = str(analysis["impact_reason"])
    return blocks


def _grounded_corpus(
    *,
    evidence_payload: dict[str, Any],
    integrated_issue: dict[str, Any],
    sources: list[dict[str, Any]],
) -> str:
    chunks: list[str] = []
    for key in ("fact_basis", "key_numbers", "representative_sources"):
        value = integrated_issue.get(key)
        if value:
            chunks.append(_stringify(value))
    # Backward compatibility for older AnalysisPackage evidence payloads.
    # New StrategicInsightAgent outputs must ground numbers/dates in the three
    # IntegrationAgent evidence fields above; these payload refs only keep legacy
    # cards from false positives during transition.
    for key in ("financial_refs", "source_links"):
        value = evidence_payload.get(key)
        if value:
            chunks.append(_stringify(value))
    for source in sources:
        if source.get("published_at") or source.get("article_id"):
            chunks.append(
                _stringify(
                    {
                        "published_at": source.get("published_at"),
                        "article_id": source.get("article_id"),
                    }
                )
            )
    return " | ".join(chunks)


def _grounded_numeric_keys(grounded_corpus: str, *, numeric_pattern: re.Pattern[str]) -> set[str]:
    return {
        key
        for match in numeric_pattern.finditer(grounded_corpus or "")
        if (key := _numeric_token_key(match.group(0)))
    }


def _numeric_token_key(token: str) -> str:
    text = re.sub(r"\s+", "", str(token or "")).strip().lower()
    if not text:
        return ""
    unit = ""
    for candidate in ("억원", "억", "조원", "조", "만원", "만", "천만", "백만", "%", "원"):
        if text.endswith(candidate):
            unit = candidate
            text = text[: -len(candidate)]
            break
    if unit == "억원":
        unit = "억"
    elif unit == "조원":
        unit = "조"
    number_text = text.replace(",", "")
    try:
        number = float(number_text)
    except ValueError:
        normalized_number = number_text
    else:
        normalized_number = str(int(number)) if number.is_integer() else f"{number:.6f}".rstrip("0")
    return f"{normalized_number}{unit}"


def _check_evidence_chain(
    *,
    evidence_payload: dict[str, Any],
    sources: list[dict[str, Any]],
    implication: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    has_sources = bool(sources) or bool(evidence_payload.get("source_links"))
    if not has_sources:
        warnings.append("evidence_chain: 출처가 비어 있음")
    provenance = implication.get("provenance") or {}
    if isinstance(provenance, dict) and not provenance.get("model"):
        warnings.append("evidence_chain: implication.provenance.model 누락")
    return warnings


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple | set):
        return " ".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_stringify(v) for v in value.values())
    return str(value)


def _nested_get(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _evidence_payload_from_state(state: SupervisorState) -> dict[str, Any]:
    bundle = state["input_bundle"]
    metadata = bundle.metadata or {}
    classification = state.get("classification") or metadata.get("classification") or {}
    integrated = state.get("integrated_issue") or {}
    analysis = state.get("analysis") or {}
    implication = state.get("implication") or {}
    evidence_payload: dict[str, Any] = {
        "source_links": [
            {
                "url": source.get("url"),
                "source_name": source.get("source_name"),
                "title": source.get("title"),
            }
            for source in (bundle.sources or [])
            if source.get("url") or source.get("title")
        ],
        "financial_refs": integrated.get("key_numbers", []),
        "mbb_refs": classification.get("mbb_refs", []),
        "analysis_package": {
            "bundle_id": bundle.bundle_id,
            "integrated_issue": integrated,
            "analysis": analysis,
            "implication": implication,
            "classification": classification,
        },
    }
    validation = state.get("validation")
    if validation is not None:
        evidence_payload["analysis_package"]["validation"] = validation.to_dict()
    return evidence_payload


def _companies_for_context(
    *,
    bundle: AnalysisInputBundle,
    integrated_issue: dict[str, Any],
) -> list[str]:
    """ProfileContext / AnalysisContext query 의 peer 입력.

    우선순위:
      1. IntegratedIssue.main_company (LLM 이 확정한 cluster 의 주체)
      2. IntegratedIssue.mentioned_peer_companies (cluster 안에서 언급된 peer)
      3. AnalysisInputBundle.companies (raw matched_companies fallback)
    """
    out: list[str] = []
    main_company = str(integrated_issue.get("main_company") or "").strip()
    if main_company:
        out.append(main_company)
    mentioned = integrated_issue.get("mentioned_peer_companies") or []
    if isinstance(mentioned, list):
        for company in mentioned:
            text = str(company or "").strip()
            if text and text not in out:
                out.append(text)
    for company in bundle.companies or []:
        text = str(company or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _cluster_metadata(
    bundle: AnalysisInputBundle,
    profile_context: ProfileContext | None,
) -> dict[str, Any]:
    trend_context = bundle.metadata.get("trend_context") or {}
    return {
        "bundle_id": bundle.bundle_id,
        "cluster_id": bundle.cluster_id,
        "source_type": bundle.source_type,
        "companies": list(bundle.companies),
        "sectors": list(bundle.sectors),
        "event_type": bundle.event_type,
        "cluster_size": len(bundle.items),
        "source_count": len(bundle.sources),
        "has_peer_profile_context": bool(
            profile_context is not None and profile_context.peer_profiles
        ),
        "has_skax_profile_context": bool(
            profile_context is not None and profile_context.skax_profile
        ),
        "has_trend_context": bool(trend_context),
        "trend_context": trend_context if isinstance(trend_context, dict) else {},
    }


def _load_calibrated_thresholds() -> dict[str, float] | None:
    """W5-1 Phase 2 calibrated threshold loader.

    Phase 1 (운영 7일 미만) 에서는 weekly CronJob 이 산출하지 않으므로 None.
    None 일 때는 어떤 metric 도 `human_review_flags` 추가 X.
    """
    # Phase 2 에서 별도 storage (예: peer_companies 의 settings JSONB 또는 별도 테이블)
    # 가 채워지면 그 값을 dict 로 반환. 현재는 Phase 1 → None.
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────────────────────────────────────


# 외부 리뷰 R-7 — LangGraph RetryPolicy 정식 적용 (2026-05-21).
# 노드별 retry / fallback 정책. LLM 호출 노드만 exponential backoff 적용.
_NODE_RETRY_POLICIES: dict[str, RetryPolicy | None] = {
    "issue_integrate": RetryPolicy(initial_interval=1.0, backoff_factor=2.0, max_attempts=2),
    "profile_context": None,  # DB only, v2→legacy fallback 으로 처리.
    "build_analysis_context": None,  # DB only, layer 별 graceful fallback.
    "strategic_insight": RetryPolicy(initial_interval=1.0, backoff_factor=2.0, max_attempts=2),
    "validate": None,  # rule-based, retry 불필요.
    "assemble": None,
    "card_writer": RetryPolicy(initial_interval=0.5, backoff_factor=2.0, max_attempts=2),
    "human_review": None,
}


def build_supervisor_graph(deps: SupervisorDeps | None = None) -> Any:
    """Compile 된 LangGraph StateGraph 반환.

    외부 리뷰 R-7 (2026-05-21) — LangGraph ``RetryPolicy`` 를 노드별로 적용한다.
    LLM 호출 노드 (issue_integrate / strategic_insight / card_writer) 는
    1초 시작, 2배 backoff, 최대 2 회 재시도. 다른 노드는 retry 없음 (graceful fallback 만).
    """
    deps = deps or SupervisorDeps()
    nodes = _make_nodes(deps)
    g: StateGraph = StateGraph(SupervisorState)
    for name, fn in nodes.items():
        retry_policy = _NODE_RETRY_POLICIES.get(name)
        if retry_policy is not None:
            g.add_node(name, fn, retry_policy=retry_policy)  # type: ignore[arg-type,call-overload]
        else:
            g.add_node(name, fn)  # type: ignore[arg-type,call-overload]
    # 외부 리뷰 R-1 (2026-05-21) — IntegratedIssue 가 main_company 를 확정한 후에
    # ProfileContext / AnalysisContext 를 build 하도록 순서 재배치.
    g.set_entry_point("issue_integrate")
    g.add_edge("issue_integrate", "profile_context")
    g.add_edge("profile_context", "build_analysis_context")
    g.add_edge("build_analysis_context", "strategic_insight")
    g.add_edge("strategic_insight", "validate")
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"pass": "assemble", "fail": "human_review"},
    )
    g.add_edge("assemble", "card_writer")
    g.add_edge("card_writer", END)
    g.add_edge("human_review", END)
    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton for default usage
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DEPS: SupervisorDeps | None = None
_DEFAULT_GRAPH: Any = None


def default_supervisor_graph() -> Any:
    global _DEFAULT_DEPS, _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        _DEFAULT_DEPS = SupervisorDeps()
        _DEFAULT_GRAPH = build_supervisor_graph(_DEFAULT_DEPS)
    return _DEFAULT_GRAPH


# ─────────────────────────────────────────────────────────────────────────────
# Convenience invocation helpers
# ─────────────────────────────────────────────────────────────────────────────


def run_supervisor(
    *,
    input_bundle: AnalysisInputBundle,
    classification: dict[str, Any] | None = None,
    profile_context: ProfileContext | None = None,
    graph: Any | None = None,
) -> SupervisorState:
    """단일 cluster 처리 진입점.

    Returns
    -------
    SupervisorState
        graph 의 최종 state. `card_news_id` / `analysis_package` / `validation` 등을
        호출자가 직접 사용할 수 있다.
    """
    graph = graph or default_supervisor_graph()
    initial: SupervisorState = cast(
        SupervisorState,
        {
            "input_bundle": input_bundle,
            "classification": classification or {},
            "profile_context": profile_context,
            "errors": [],
            "human_review_flags": [],
        },
    )
    result = graph.invoke(initial)
    return cast(SupervisorState, result)


def supervisor_run_id() -> str:
    return uuid.uuid4().hex


# ─────────────────────────────────────────────────────────────────────────────
# 권장 이름 alias (외부 리뷰 2026-05-21 R-rename)
#
# 이 모듈의 구조는 LLM-router (multi-agent supervisor) 가 아닌 **고정 순서 DAG
# pipeline** 이다. 그에 맞춰 외부에서 사용할 권장 이름을 alias 로 노출한다.
# 기존 ``Supervisor*`` 이름은 backward-compat 으로 유지하되, 새 코드는 ``AnalysisFlow*``
# / ``build_analysis_flow_graph`` 사용을 권장.
# ─────────────────────────────────────────────────────────────────────────────

AnalysisFlowState = SupervisorState
AnalysisFlowDeps = SupervisorDeps
build_analysis_flow_graph = build_supervisor_graph
default_analysis_flow_graph = default_supervisor_graph
run_analysis_flow = run_supervisor
analysis_flow_run_id = supervisor_run_id


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
