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

    @property
    def is_news_cluster(self) -> bool:
        """뉴스 MVP 경로 여부."""
        return self.source_type in {"news", "news_cluster"}

    @property
    def representative_id(self) -> int:
        """대표 raw_article id.

        우선 metadata.representative_id 를 사용하고, 없으면 representative 표시가 있는
        item, 그래도 없으면 첫 raw article id 를 사용한다.
        """
        metadata_id = _safe_int(self.metadata.get("representative_id"), 0)
        if metadata_id > 0:
            return metadata_id
        for item in self.items:
            if item.get("is_representative"):
                item_id = _raw_article_id(item)
                if item_id > 0:
                    return item_id
        return self.raw_article_ids[0] if self.raw_article_ids else 0

    @property
    def cluster_article_ids(self) -> list[int]:
        """클러스터에 속한 raw article ids."""
        raw = self.metadata.get("cluster_article_ids") or []
        ids: list[int] = []
        if isinstance(raw, list | tuple | set):
            ids.extend(_safe_int(item, 0) for item in raw)
        ids.extend(self.raw_article_ids)
        return _dedupe_positive_ints(ids)

    @property
    def classification(self) -> dict[str, Any]:
        """전처리/분류 결과 payload."""
        raw = self.metadata.get("classification") or {}
        return raw if isinstance(raw, dict) else {}

    @property
    def raw_article_ids(self) -> list[int]:
        """items 에 포함된 raw article ids."""
        return _dedupe_positive_ints([_raw_article_id(item) for item in self.items])

    @property
    def matched_companies(self) -> list[str]:
        """전처리 relevance output 의 matched companies."""
        values = [*self.companies]
        for item in self.items:
            values.extend(_string_list(item.get("matched_companies")))
            values.extend(_string_list((item.get("metadata") or {}).get("matched_companies")))
        values.extend(_string_list(self.classification.get("company")))
        values.extend(_string_list(self.classification.get("companies")))
        return _dedupe_strings(values)

    @property
    def matched_sectors(self) -> list[str]:
        """전처리 relevance/classification output 의 matched sectors."""
        values = [*self.sectors]
        for item in self.items:
            values.extend(_string_list(item.get("matched_sectors")))
            values.extend(_string_list((item.get("metadata") or {}).get("matched_sectors")))
        values.extend(_string_list(self.classification.get("sector")))
        values.extend(_string_list(self.classification.get("sectors")))
        return _dedupe_strings(values)

    @property
    def business_signal_rows(self) -> list[dict[str, Any]]:
        """raw_article_business_signals rows attached to items."""
        rows: list[dict[str, Any]] = []
        for item in self.items:
            article_id = _raw_article_id(item)
            for signal in _dict_rows(item.get("business_signals")):
                rows.append({"raw_article_id": article_id, **signal})
        return rows

    @property
    def financial_metric_rows(self) -> list[dict[str, Any]]:
        """raw_article_financial_metrics rows attached to items."""
        rows: list[dict[str, Any]] = []
        for item in self.items:
            article_id = _raw_article_id(item)
            for metric in _dict_rows(item.get("financial_metrics")):
                rows.append({"raw_article_id": article_id, **metric})
        return rows

    @property
    def parser_outputs(self) -> list[dict[str, Any]]:
        """raw_article_parse_results-derived payloads attached to items."""
        outputs: list[dict[str, Any]] = []
        for item in self.items:
            article_id = _raw_article_id(item)
            parser_result = item.get("parser_result")
            financial_record = item.get("financial_record")
            parser_warnings = item.get("parser_warnings")
            if parser_result or financial_record or parser_warnings:
                outputs.append(
                    {
                        "raw_article_id": article_id,
                        "parser_result": parser_result if isinstance(parser_result, dict) else {},
                        "financial_record": financial_record
                        if isinstance(financial_record, dict)
                        else {},
                        "parser_warnings": parser_warnings
                        if isinstance(parser_warnings, list)
                        else [],
                    }
                )
        return outputs

    @property
    def news_preprocessing_outputs(self) -> dict[str, Any]:
        """IntegrationAgent가 참고할 뉴스 전처리 output 묶음."""
        return {
            "raw_article_ids": self.raw_article_ids,
            "representative_id": self.representative_id,
            "cluster_article_ids": self.cluster_article_ids,
            "matched_companies": self.matched_companies,
            "matched_sectors": self.matched_sectors,
            "classification": self.classification,
            "business_signals": self.business_signal_rows,
            "financial_metrics": self.financial_metric_rows,
            "parser_outputs": self.parser_outputs,
        }

    def validate_news_contract(self) -> dict[str, Any]:
        """뉴스 MVP용 AnalysisInputBundle contract 검증.

        hard validation 이 아니라 IntegrationAgent가 입력 품질을 이해하기 위한
        structured diagnostic 이다. 뉴스 MVP는 raw_articles 중심이라 문서 분석용
        financial_metrics / business_signals / parser_outputs 를 요구하지 않는다.
        """
        missing_fields: list[str] = []
        warnings: list[str] = []
        if not self.is_news_cluster:
            warnings.append(f"source_type is not news/news_cluster: {self.source_type}")
        if not self.items:
            missing_fields.append("items")
        if not self.raw_article_ids:
            missing_fields.append("items[].id")
        if not self.representative_id:
            missing_fields.append("metadata.representative_id")
        if not self.cluster_article_ids:
            missing_fields.append("metadata.cluster_article_ids")
        if not self.matched_companies:
            warnings.append("matched_companies empty")
        if not self.matched_sectors:
            warnings.append("matched_sectors empty")

        outputs = self.news_preprocessing_outputs
        return {
            "pass": not missing_fields,
            "source_type": self.source_type,
            "item_count": len(self.items),
            "raw_article_ids": outputs["raw_article_ids"],
            "representative_id": outputs["representative_id"],
            "cluster_article_ids": outputs["cluster_article_ids"],
            "matched_companies": outputs["matched_companies"],
            "matched_sectors": outputs["matched_sectors"],
            "business_signal_count": len(outputs["business_signals"]),
            "financial_metric_count": len(outputs["financial_metrics"]),
            "parser_output_count": len(outputs["parser_outputs"]),
            "missing_fields": missing_fields,
            "warnings": warnings,
        }


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
    """StrategicAnalyzer 또는 StrategicInsightAgent.analysis 출력의 typed view.

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
    financial_article_ids: list[int] = field(default_factory=list)
    precedent_card_ids: list[str] = field(default_factory=list)
    retrieved_card_ids: list[str] = field(default_factory=list)
    sector_pulse_weeks: list[str] = field(default_factory=list)
    executive_memory_dates: list[str] = field(default_factory=list)
    analysis_ledger_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisContext:
    """4-Layer Context Model 의 Layer 4 active context. token budget ≤ 4,000."""

    peer_event_timeline_recent: list[TimelineEntry] = field(default_factory=list)
    capability_evolution: dict[str, Any] = field(default_factory=dict)
    sector_pulse_recent: list[SectorPulseRow] = field(default_factory=list)
    financial_trend: dict[str, FinancialSeries] = field(default_factory=dict)
    event_chain_candidates: list[PrecedentCandidate] = field(default_factory=list)
    similar_cards_rag: list[RetrievedCard] = field(default_factory=list)
    evidence_density_per_peer: dict[str, EvidenceDensity] = field(default_factory=dict)
    executive_memory_recent: list[dict[str, Any]] = field(default_factory=list)
    analysis_ledger_by_peer: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    weekly_digest_by_peer: dict[str, Any] = field(default_factory=dict)
    token_budget_used: int = 0
    provenance: ContextProvenance = field(default_factory=ContextProvenance)

    def available_layer_count(self) -> int:
        """비어있지 않은 layer 의 개수. context_hit_ratio 분모 (P5-LOG-2)."""
        count = 0
        if self.peer_event_timeline_recent:
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
            "capability_evolution": dict(self.capability_evolution),
            "sector_pulse_recent": [e.to_dict() for e in self.sector_pulse_recent],
            "financial_trend": {k: v.to_dict() for k, v in self.financial_trend.items()},
            "event_chain_candidates": [e.to_dict() for e in self.event_chain_candidates],
            "similar_cards_rag": [e.to_dict() for e in self.similar_cards_rag],
            "evidence_density_per_peer": {
                k: v.to_dict() for k, v in self.evidence_density_per_peer.items()
            },
            "executive_memory_recent": list(self.executive_memory_recent),
            "analysis_ledger_by_peer": {
                peer_id: list(entries) for peer_id, entries in self.analysis_ledger_by_peer.items()
            },
            "weekly_digest_by_peer": dict(self.weekly_digest_by_peer),
            "token_budget_used": self.token_budget_used,
            "provenance": self.provenance.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TrendContext — ITTrendAgent 산출물 (런타임/저장 payload용 내부 DTO)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TrendContext:
    """SPRi/BCG + 글로벌 뉴스룸 분석 결과 기반 글로벌·산업 흐름 context.

    DB schema가 아니며, ITTrendAgent가 생성/갱신해 StrategicAnalyzer가 참고할 수 있는
    내부 DTO이다. 글로벌 회사별 뉴스룸 원문은 카드뉴스 생성 파이프라인을 먼저 타고,
    그 산출물인 IntegratedIssue / AnalysisResult가 실행 신호 입력으로 들어온다.
    """

    period: str | None = None
    trend_summary: str = ""
    trend_lines: list[str] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)
    source_groups: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    reference_issue_ids: list[str] = field(default_factory=list)
    updated_at: str = ""
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    model: str = "gpt-4o-mini"
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
            # Backward-compat flatten — CardNewsComposer / heuristic fallback 호환.
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
    issue_understanding: dict[str, Any] = field(default_factory=dict)
    profile_linkage: dict[str, Any] = field(default_factory=dict)
    skax_response_linkage: dict[str, Any] = field(default_factory=dict)
    grounding_summary: dict[str, Any] = field(default_factory=dict)
    claim_strength: str | None = None

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
            "issue_understanding": self.issue_understanding,
            "profile_linkage": self.profile_linkage,
            "skax_response_linkage": self.skax_response_linkage,
            "grounding_summary": self.grounding_summary,
            "claim_strength": self.claim_strength,
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


def _raw_article_id(item: dict[str, Any]) -> int:
    return _safe_int(item.get("id") or item.get("raw_article_id") or item.get("preprocess_id"), 0)


def _dedupe_positive_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    return [stripped] if stripped else []


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


__all__ = [
    "AnalysisContext",
    "AnalysisInputBundle",
    "AnalysisInputMetadata",
    "AnalysisPackage",
    "AnalysisResult",
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
    "TrendContext",
    "ValidationReport",
]
