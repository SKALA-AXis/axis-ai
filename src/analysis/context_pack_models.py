# 작성일: 2026-06-09
# 작성자: 최종민
# 변경이력:
#   2026-06-09 최종민 — ContextPackAssembler 읽기 모델·주간 다이제스트 에이전트 신규 추가, ruff/mypy 정리
"""Unified peer context pack — P0 ContextPackAssembler read model.

Writers (CronJobs / agents) remain separate by cadence. Consumers assemble a
single PeerContextPack at read time (LLM-free).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.analysis.models import (
    AnalysisContext,
    ContextProvenance,
    EvidenceDensity,
    FinancialSeries,
    PrecedentCandidate,
    RetrievedCard,
    SectorPulseRow,
    TimelineEntry,
)


@dataclass(slots=True)
class ExecutiveMemoryEntry:
    """Compact slice from ``today_insight_reports`` for day-over-day comparison."""

    report_date: str
    headline: str = ""
    executive_summary: str = ""
    executive_implication: str = ""
    signal_values: list[str] = field(default_factory=list)
    peer_ids: list[str] = field(default_factory=list)
    source_integrated_issue_ids: list[str] = field(default_factory=list)
    source_card_ids: list[str] = field(default_factory=list)
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContextPackCoverage:
    has_timeline: bool = False
    has_capability_evolution: bool = False
    has_sector_pulse: bool = False
    has_financial_trend: bool = False
    has_event_chain: bool = False
    has_similar_cards_rag: bool = False
    has_executive_memory: bool = False
    has_analysis_ledger: bool = False
    has_weekly_digest: bool = False
    cold_start: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PeerContextPack:
    """Single read model for historical context across cadences."""

    peer_ids: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    bundle_id: str | None = None
    peer_event_timeline_recent: list[TimelineEntry] = field(default_factory=list)
    capability_evolution: dict[str, Any] = field(default_factory=dict)
    sector_pulse_recent: list[SectorPulseRow] = field(default_factory=list)
    financial_trend: dict[str, FinancialSeries] = field(default_factory=dict)
    event_chain_candidates: list[PrecedentCandidate] = field(default_factory=list)
    similar_cards_rag: list[RetrievedCard] = field(default_factory=list)
    evidence_density_per_peer: dict[str, EvidenceDensity] = field(default_factory=dict)
    executive_memory_recent: list[ExecutiveMemoryEntry] = field(default_factory=list)
    analysis_ledger_by_peer: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    weekly_digest_by_peer: dict[str, Any] = field(default_factory=dict)
    provenance: ContextProvenance = field(default_factory=ContextProvenance)
    coverage: ContextPackCoverage = field(default_factory=ContextPackCoverage)
    assembled_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_ids": list(self.peer_ids),
            "sectors": list(self.sectors),
            "bundle_id": self.bundle_id,
            "peer_event_timeline_recent": [e.to_dict() for e in self.peer_event_timeline_recent],
            "capability_evolution": dict(self.capability_evolution),
            "sector_pulse_recent": [e.to_dict() for e in self.sector_pulse_recent],
            "financial_trend": {k: v.to_dict() for k, v in self.financial_trend.items()},
            "event_chain_candidates": [e.to_dict() for e in self.event_chain_candidates],
            "similar_cards_rag": [e.to_dict() for e in self.similar_cards_rag],
            "evidence_density_per_peer": {
                k: v.to_dict() for k, v in self.evidence_density_per_peer.items()
            },
            "executive_memory_recent": [e.to_dict() for e in self.executive_memory_recent],
            "analysis_ledger_by_peer": self.analysis_ledger_by_peer,
            "weekly_digest_by_peer": dict(self.weekly_digest_by_peer),
            "provenance": self.provenance.to_dict(),
            "coverage": self.coverage.to_dict(),
            "assembled_at": self.assembled_at,
        }

    def to_analysis_context(self) -> AnalysisContext:
        """Map pack → legacy AnalysisContext for existing pipeline nodes."""
        return AnalysisContext(
            peer_event_timeline_recent=list(self.peer_event_timeline_recent),
            capability_evolution=dict(self.capability_evolution),
            sector_pulse_recent=list(self.sector_pulse_recent),
            financial_trend=dict(self.financial_trend),
            event_chain_candidates=list(self.event_chain_candidates),
            similar_cards_rag=list(self.similar_cards_rag),
            evidence_density_per_peer=dict(self.evidence_density_per_peer),
            executive_memory_recent=[e.to_dict() for e in self.executive_memory_recent],
            analysis_ledger_by_peer={
                peer_id: list(entries) for peer_id, entries in self.analysis_ledger_by_peer.items()
            },
            weekly_digest_by_peer=dict(self.weekly_digest_by_peer),
            provenance=self.provenance,
        )


__all__ = [
    "ContextPackCoverage",
    "ExecutiveMemoryEntry",
    "PeerContextPack",
]
