"""Analysis pipeline internal DTOs.

이 모델들은 DB schema나 API schema가 아니다.
기존 ``src/schemas.py``, DB table 구조, repository/save 로직을 대체하지 않는다.

용도:
- 1단계 분석 파이프라인 내부에서 Agent 간 데이터를 주고받는 DTO/dataclass
- AnalysisInputBundle 중심으로 이슈 통합, 분석, 시사점, 카드뉴스 생성을 연결하는
  임시 런타임 계약

최종 저장 시에는 반드시 기존 DB schema 또는 기존 repository/save 로직에 맞는
dict payload로 변환해서 사용한다.

v4.0 변경 (W1-1/W1-2/W1-3):
- ImplicationResult / PeerImplication / SkaxImplication / PrecedentLink / ImplicationProvenance 신설
- AnalysisInputMetadata typed dataclass (P2-5)
- AnalysisContext (W4) 신설 — 4-Layer Context Model 의 Layer 4 active context
- ValidationReport (W2-3) — 출처 수치 / 단정 표현 / evidence chain quality gate
- EvaluationMetrics (W5-1) — rule-based metric 4종 + drift
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# AnalysisInputBundle + NormalizedDataBundle (기존 유지, metadata 만 typed)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class NormalizedDataBundle:
    bundle_id: str
    source_type: str
    companies: list[str]
    sectors: list[str]
    event_type: str | None
    items: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    collected_at: str
    cluster_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClassificationPayload:
    """classification dict 를 typed 로 표현 (P2-5).

    free-form dict 를 그대로 받되 핵심 필드를 명시화해서 자동완성 / mypy 도움 확보.
    """

    sector: str = ""
    sectors: list[str] = field(default_factory=list)
    event_type: str = ""
    importance: str = ""
    importance_score: float = 0.0
    exposure_band: str = ""
    exposure_score: float = 0.0
    signals: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sector": self.sector,
            "sectors": self.sectors,
            "event_type": self.event_type,
            "importance": self.importance,
            "importance_score": self.importance_score,
            "exposure_band": self.exposure_band,
            "exposure_score": self.exposure_score,
            "signals": self.signals,
        }
        if self.extra:
            payload.update(self.extra)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ClassificationPayload:
        if not data:
            return cls()
        known = {
            "sector",
            "sectors",
            "event_type",
            "importance",
            "importance_score",
            "exposure_band",
            "exposure_score",
            "signals",
        }
        sectors_raw = data.get("sectors") or []
        if isinstance(sectors_raw, str):
            sectors_list: list[str] = [sectors_raw]
        elif isinstance(sectors_raw, list | tuple):
            sectors_list = [str(item) for item in sectors_raw if item]
        else:
            sectors_list = []
        signals_raw = data.get("signals") or {}
        signals = signals_raw if isinstance(signals_raw, dict) else {}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            sector=str(data.get("sector") or ""),
            sectors=sectors_list,
            event_type=str(data.get("event_type") or ""),
            importance=str(data.get("importance") or ""),
            importance_score=_safe_float(data.get("importance_score"), 0.0),
            exposure_band=str(data.get("exposure_band") or ""),
            exposure_score=_safe_float(data.get("exposure_score"), 0.0),
            signals=signals,
            extra=extra,
        )


@dataclass(slots=True)
class AnalysisInputMetadata:
    """AnalysisInputBundle.metadata 의 typed 표현 (P2-5).

    free-form dict 였던 metadata 의 핵심 필드를 dataclass 로 고정. 기존 호출부는
    `to_dict()` 으로 dict 사용 가능 — 호환성 유지.
    """

    representative_id: int
    cluster_article_ids: list[int] = field(default_factory=list)
    classification: ClassificationPayload = field(default_factory=ClassificationPayload)
    created_at: str = ""
    collected_at: str | None = None
    profile_snapshot_version: str | None = None
    source_run_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "representative_id": self.representative_id,
            "cluster_article_ids": list(self.cluster_article_ids),
            "classification": self.classification.to_dict(),
            "created_at": self.created_at,
        }
        if self.collected_at is not None:
            payload["collected_at"] = self.collected_at
        if self.profile_snapshot_version is not None:
            payload["profile_snapshot_version"] = self.profile_snapshot_version
        if self.source_run_id is not None:
            payload["source_run_id"] = self.source_run_id
        if self.extra:
            payload.update(self.extra)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnalysisInputMetadata:
        if not data:
            return cls(representative_id=0)
        known = {
            "representative_id",
            "cluster_article_ids",
            "classification",
            "created_at",
            "collected_at",
            "profile_snapshot_version",
            "source_run_id",
        }
        cluster_ids_raw = data.get("cluster_article_ids") or []
        cluster_ids: list[int] = []
        if isinstance(cluster_ids_raw, list | tuple):
            for item in cluster_ids_raw:
                try:
                    cluster_ids.append(int(item))
                except (TypeError, ValueError):
                    continue
        classification = ClassificationPayload.from_dict(data.get("classification"))
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            representative_id=_safe_int(data.get("representative_id"), 0),
            cluster_article_ids=cluster_ids,
            classification=classification,
            created_at=str(data.get("created_at") or ""),
            collected_at=_optional_str(data.get("collected_at")),
            profile_snapshot_version=_optional_str(data.get("profile_snapshot_version")),
            source_run_id=_optional_str(data.get("source_run_id")),
            extra=extra,
        )


@dataclass(slots=True)
class AnalysisInputBundle:
    """Agent 실행 시점에만 만들어지는 내부 입력 묶음.

    DB table이 아니며, 뉴스의 경우 하나의 ``raw_articles.cluster_id``에 속한
    기사 묶음을, DART/IR/리포트의 경우 문서 또는 문서 내 분석 단위 묶음을
    표현한다.
    """

    bundle_id: str
    cluster_id: str | None
    source_type: str
    companies: list[str]
    sectors: list[str]
    event_type: str | None
    items: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    evidence_snippets: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def typed_metadata(self) -> AnalysisInputMetadata:
        """metadata dict 를 typed view 로 반환 (P2-5)."""
        return AnalysisInputMetadata.from_dict(self.metadata)


# Backward compatibility: 기존 코드의 EvidencePack import는 내부 DTO alias로 유지한다.
EvidencePack = AnalysisInputBundle


# ─────────────────────────────────────────────────────────────────────────────
# IntegratedIssue / LegacySummaryResult (기존 유지)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class IntegratedIssue:
    main_issue: str
    integrated_text: str
    consolidated_facts: list[dict[str, Any]]
    business_signals: list[dict[str, Any]]
    representative_sources: list[dict[str, Any]]
    fact_basis: list[dict[str, Any]]
    missing_or_uncertain_points: list[dict[str, Any]] = field(default_factory=list)
    key_numbers: list[dict[str, Any]] = field(default_factory=list)
    source_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LegacySummaryResult:
    summary: str
    summary_lines: list[str]
    key_facts: list[dict[str, Any]]
    fact_basis: list[dict[str, Any]]
    main_topic: str | None = None
    source_count: int | None = None
    representative_sources: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Backward compatibility: 기존 SummaryResult import는 예전 요약 DTO 형태로 유지한다.
SummaryResult = LegacySummaryResult


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisResult — W1-2 정합화
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AnalysisResult:
    """StrategicAnalyzer 출력의 typed view.

    실 LLM 출력 key 와 1:1. dict 호환은 ``to_dict()``.
    """

    is_valid_analysis: bool = False
    analysis_scope: str = "peer_and_industry"
    analysis_summary: str = ""
    strategic_meaning: list[str] = field(default_factory=list)
    market_signal: str = ""
    impact_level: Literal["high", "medium", "low"] = "low"
    impact_reason: str = ""
    risk_or_opportunity: Literal["risk", "opportunity", "neutral"] = "neutral"
    confidence: float = 0.0
    reason: str = ""
    model: str = ""
    cluster_id: int | str | None = None
    representative_id: int | str | None = None
    main_company: str = ""
    source_article_ids: list[int] = field(default_factory=list)
    basis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnalysisResult:
        if not data:
            return cls()
        impact_level = str(data.get("impact_level") or "low").lower()
        if impact_level not in {"high", "medium", "low"}:
            impact_level = "low"
        rop = str(data.get("risk_or_opportunity") or "neutral").lower()
        if rop not in {"risk", "opportunity", "neutral"}:
            rop = "neutral"
        return cls(
            is_valid_analysis=bool(data.get("is_valid_analysis", False)),
            analysis_scope=str(data.get("analysis_scope") or "peer_and_industry"),
            analysis_summary=str(data.get("analysis_summary") or ""),
            strategic_meaning=[str(item) for item in (data.get("strategic_meaning") or [])],
            market_signal=str(data.get("market_signal") or ""),
            impact_level=impact_level,  # type: ignore[arg-type]
            impact_reason=str(data.get("impact_reason") or ""),
            risk_or_opportunity=rop,  # type: ignore[arg-type]
            confidence=_safe_float(data.get("confidence"), 0.0),
            reason=str(data.get("reason") or ""),
            model=str(data.get("model") or ""),
            cluster_id=data.get("cluster_id"),
            representative_id=data.get("representative_id"),
            main_company=str(data.get("main_company") or ""),
            source_article_ids=[
                _safe_int(item, 0) for item in (data.get("source_article_ids") or [])
            ],
            basis=data.get("basis") or {},
        )


# ─────────────────────────────────────────────────────────────────────────────
# ProfileContext
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ProfileContext:
    skax_profile: dict[str, Any]
    peer_profiles: dict[str, Any]
    sector_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisContext — W4 신설 (Layer 4 active context)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TimelineEntry:
    company_id: str
    event_date: str
    card_id: str
    event_type: str
    sector: str
    headline: str
    importance: str
    importance_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CapabilityWindow:
    period: str
    business_area: str
    narrative: str
    delta_intensity: float
    confidence: float
    evidence_signal_ids: list[str] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SectorPulseRow:
    sector: str
    week_start: str
    event_count: int
    peer_event_count: int
    general_event_count: int
    intensity_avg: float
    event_type_distribution: dict[str, int] = field(default_factory=dict)
    active_peers: list[str] = field(default_factory=list)
    notable_card_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FinancialSeriesPoint:
    period_year: int
    period_quarter: str
    value_numeric: float | None = None
    value_krwbn: float | None = None
    unit: str = ""
    raw_article_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FinancialSeries:
    company_id: str
    metric_name_canonical: str
    points: list[FinancialSeriesPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "metric_name_canonical": self.metric_name_canonical,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass(slots=True)
class PrecedentCandidate:
    card_id: str
    company_id: str
    event_type: str
    days_since: int
    cosine: float
    headline: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedCard:
    card_id: str
    cosine: float
    headline: str = ""
    sector: str = ""
    company_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceDensity:
    signal_count_4q: int = 0
    metric_count_4q: int = 0
    timeline_count_90d: int = 0
    density_label: Literal["rich", "moderate", "sparse"] = "sparse"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContextProvenance:
    used_layers: list[str] = field(default_factory=list)
    timeline_card_ids: list[str] = field(default_factory=list)
    capability_evidence_ids: list[str] = field(default_factory=list)
    financial_article_ids: list[int] = field(default_factory=list)
    precedent_card_ids: list[str] = field(default_factory=list)
    retrieved_card_ids: list[str] = field(default_factory=list)
    sector_pulse_weeks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisContext:
    """4-Layer Context Model 의 Layer 4 active context. token budget ≤ 4,000."""

    peer_event_timeline_recent: list[TimelineEntry] = field(default_factory=list)
    capability_evolution: dict[str, CapabilityWindow] = field(default_factory=dict)
    sector_pulse_recent: list[SectorPulseRow] = field(default_factory=list)
    financial_trend: dict[str, FinancialSeries] = field(default_factory=dict)
    event_chain_candidates: list[PrecedentCandidate] = field(default_factory=list)
    similar_cards_rag: list[RetrievedCard] = field(default_factory=list)
    evidence_density_per_peer: dict[str, EvidenceDensity] = field(default_factory=dict)
    token_budget_used: int = 0
    provenance: ContextProvenance = field(default_factory=ContextProvenance)

    def available_layer_count(self) -> int:
        """비어있지 않은 layer 의 개수. context_hit_ratio 분모 (P5-LOG-2)."""
        count = 0
        if self.peer_event_timeline_recent:
            count += 1
        if self.capability_evolution:
            count += 1
        if self.sector_pulse_recent:
            count += 1
        if self.financial_trend:
            count += 1
        if self.event_chain_candidates:
            count += 1
        if self.similar_cards_rag:
            count += 1
        return count

    def is_empty(self) -> bool:
        return self.available_layer_count() == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_event_timeline_recent": [e.to_dict() for e in self.peer_event_timeline_recent],
            "capability_evolution": {k: v.to_dict() for k, v in self.capability_evolution.items()},
            "sector_pulse_recent": [e.to_dict() for e in self.sector_pulse_recent],
            "financial_trend": {k: v.to_dict() for k, v in self.financial_trend.items()},
            "event_chain_candidates": [e.to_dict() for e in self.event_chain_candidates],
            "similar_cards_rag": [e.to_dict() for e in self.similar_cards_rag],
            "evidence_density_per_peer": {
                k: v.to_dict() for k, v in self.evidence_density_per_peer.items()
            },
            "token_budget_used": self.token_budget_used,
            "provenance": self.provenance.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ImplicationResult — W1-1 신설 (v4.0 schema), W4-5 에서 PrecedentLink 활용
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PrecedentLink:
    card_id: str
    relation: Literal["follow_up", "reaction", "echo", "contradiction"]
    days_since: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PeerImplication:
    company_id: str
    company_name_ko: str
    peer_meaning: str
    capability_change: str | None = None
    precedent_link: PrecedentLink | None = None
    sourced_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "company_id": self.company_id,
            "company_name_ko": self.company_name_ko,
            "peer_meaning": self.peer_meaning,
            "capability_change": self.capability_change,
            "precedent_link": (
                self.precedent_link.to_dict() if self.precedent_link is not None else None
            ),
            "sourced_evidence_ids": list(self.sourced_evidence_ids),
        }
        return payload


@dataclass(slots=True)
class SkaxImplication:
    why_important: str
    potential_impact: str
    opportunities: list[str] = field(default_factory=list)
    threats: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    business_line_mapping: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImplicationProvenance:
    generator: str = "ImplicationAgent"
    prompt_version: str = "implication-v4.0"
    model: str = "gpt-4o"
    bundle_id: str = ""
    used_peer_profile_keys: list[str] = field(default_factory=list)
    used_skax_profile_keys: list[str] = field(default_factory=list)
    used_fact_ids: list[str] = field(default_factory=list)
    used_context_layers: list[str] = field(default_factory=list)
    run_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImplicationResult:
    """W1-1 — v4.0 schema. `is_valid_implication` 가 False 면 카드 skip 대상.

    Backward compat: 기존 dict 접근 케이스 (`implication.get("opportunities")` 등) 를
    위해 `to_dict()` 가 flatten 된 frontend-friendly 키도 함께 노출한다.
    """

    is_valid_implication: bool = False
    implication_scope: Literal["peer_and_skax"] = "peer_and_skax"
    peer_implication: PeerImplication | None = None
    skax_implication: SkaxImplication | None = None
    follow_up_questions: list[str] = field(default_factory=list)
    watch_points: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_label: Literal["sufficient", "moderate", "insufficient"] = "insufficient"
    provenance: ImplicationProvenance = field(default_factory=ImplicationProvenance)

    def to_dict(self) -> dict[str, Any]:
        skax = self.skax_implication.to_dict() if self.skax_implication is not None else {}
        peer = self.peer_implication.to_dict() if self.peer_implication is not None else {}
        return {
            "is_valid_implication": self.is_valid_implication,
            "implication_scope": self.implication_scope,
            "peer_implication": peer,
            "skax_implication": skax,
            "follow_up_questions": list(self.follow_up_questions),
            "watch_points": list(self.watch_points),
            "confidence": self.confidence,
            "evidence_label": self.evidence_label,
            "provenance": self.provenance.to_dict(),
            # Backward-compat flatten — CardNewsAgent / heuristic fallback 호환.
            "opportunities": skax.get("opportunities", []) if isinstance(skax, dict) else [],
            "threats": skax.get("threats", []) if isinstance(skax, dict) else [],
            "recommended_actions": (
                skax.get("recommended_actions", []) if isinstance(skax, dict) else []
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ValidationReport — W2-3 quality gate
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class NumericViolation:
    value: str
    location: str  # e.g. "skax_implication.why_important"
    reason: str = "출처에 없는 수치"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationMetrics:
    """W5-1 rule-based metric. Phase 1 = 4 metric, Phase 2 = +regression_drift."""

    context_hit_ratio: float = 0.0
    evidence_claim_ratio: float = 0.0
    specificity_score: float = 0.0
    actionability_score: float = 0.0
    regression_drift: float | None = None
    evaluator_version: str = "rule-v1.0"
    evaluated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationReport:
    """W2-3 quality gate + W5-1 evaluation metric 통합 보고.

    ``passed`` 는 **hard gate** 결과만 본다 — 외부 리뷰 R-4 (2026-05-21):

    * **Hard (passed=False 사유)** — 카드 생성 차단, ``human_review`` 라우팅.
        ``numeric_violations``: 출처에 없는 수치
        ``evidence_chain_warnings``: 출처 0건 또는 provenance 누락
        ``integrated_issue_valid=False`` / ``analysis_valid=False`` / ``implication_valid=False``

    * **Soft (passed=True 유지, 품질 경고만)** — 카드 생성 진행, ``evaluation_payload``
      또는 ``human_review_flags`` 에 점수만 기록.
        ``certainty_warnings``: 단정 표현 (반드시/확실히/...)
        ``implication_confidence_warning``: confidence < 0.4
        ``metrics`` (W5-1 rule-based score): threshold calibration (Phase 2) 활성 시
            ``actionability_score`` / ``evidence_claim_ratio`` 등이 bottom-10% 인 경우
            ``low_<metric_name>`` flag 만 추가.
    """

    passed: bool = False
    integrated_issue_valid: bool = False
    analysis_valid: bool = False
    implication_valid: bool = False
    numeric_violations: list[NumericViolation] = field(default_factory=list)
    certainty_warnings: list[str] = field(default_factory=list)
    evidence_chain_warnings: list[str] = field(default_factory=list)
    implication_confidence_warning: bool = False
    sc_score: float = 0.0
    reason: str = ""
    metrics: EvaluationMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "pass": self.passed,
            "passed": self.passed,
            "integrated_issue_valid": self.integrated_issue_valid,
            "analysis_valid": self.analysis_valid,
            "implication_valid": self.implication_valid,
            "numeric_violations": [v.to_dict() for v in self.numeric_violations],
            "certainty_warnings": list(self.certainty_warnings),
            "evidence_chain_warnings": list(self.evidence_chain_warnings),
            "implication_confidence_warning": self.implication_confidence_warning,
            "sc_score": self.sc_score,
            "reason": self.reason,
        }
        if self.metrics is not None:
            payload["metrics"] = self.metrics.to_dict()
        return payload


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisPackage — Supervisor 산출물
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AnalysisPackage:
    bundle_id: str
    input_bundle: AnalysisInputBundle
    integrated_issue: dict[str, Any]
    analysis: dict[str, Any]
    implication: dict[str, Any]
    sources: list[dict[str, Any]]
    validation: dict[str, Any]
    evidence_payload: dict[str, Any] = field(default_factory=dict)
    classification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "input_bundle": self.input_bundle.to_dict(),
            "integrated_issue": self.integrated_issue,
            "summary": self.integrated_issue,
            "analysis": self.analysis,
            "implication": self.implication,
            "sources": self.sources,
            "validation": self.validation,
            "evidence_payload": self.evidence_payload,
            "classification": self.classification,
        }


@dataclass(slots=True)
class CardNews:
    """내부 카드뉴스 DTO.

    DB ``card_news`` table schema가 아니며, 저장 시에는 기존 저장 payload로 변환한다.
    """

    card_id: str
    title: str
    summary_lines: list[str]
    key_points: list[str]
    implication: str
    sources: list[dict[str, Any]]
    validation: dict[str, Any]
    company: list[str]
    sector: list[str]
    event_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "AnalysisContext",
    "AnalysisInputBundle",
    "AnalysisInputMetadata",
    "AnalysisPackage",
    "AnalysisResult",
    "CapabilityWindow",
    "CardNews",
    "ClassificationPayload",
    "ContextProvenance",
    "EvaluationMetrics",
    "EvidenceDensity",
    "EvidencePack",
    "FinancialSeries",
    "FinancialSeriesPoint",
    "ImplicationProvenance",
    "ImplicationResult",
    "IntegratedIssue",
    "LegacySummaryResult",
    "NormalizedDataBundle",
    "NumericViolation",
    "PeerImplication",
    "PrecedentCandidate",
    "PrecedentLink",
    "ProfileContext",
    "RetrievedCard",
    "SectorPulseRow",
    "SkaxImplication",
    "SummaryResult",
    "TimelineEntry",
    "ValidationReport",
]
