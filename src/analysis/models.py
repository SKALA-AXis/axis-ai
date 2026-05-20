"""Analysis pipeline internal DTOs.

이 모델들은 DB schema나 API schema가 아니다.
기존 ``src/schemas.py``, DB table 구조, repository/save 로직을 대체하지 않는다.

용도:
- 1단계 분석 파이프라인 내부에서 Agent 간 데이터를 주고받는 DTO/dataclass
- AnalysisInputBundle 중심으로 이슈 통합, 분석, 시사점, 카드뉴스 생성을 연결하는
  임시 런타임 계약

최종 저장 시에는 반드시 기존 DB schema 또는 기존 repository/save 로직에 맞는
dict payload로 변환해서 사용한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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


# Backward compatibility: 기존 코드의 EvidencePack import는 내부 DTO alias로 유지한다.
EvidencePack = AnalysisInputBundle


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


@dataclass(slots=True)
class AnalysisResult:
    strategic_moves: list[str]
    market_signals: list[str]
    competitive_meaning: str
    risk_factors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProfileContext:
    skax_profile: dict[str, Any]
    peer_profiles: dict[str, Any]
    sector_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImplicationResult:
    implication: str
    opportunities: list[str]
    threats: list[str]
    recommended_actions: list[str]
    follow_up_questions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisPackage:
    bundle_id: str
    input_bundle: AnalysisInputBundle
    integrated_issue: dict[str, Any]
    analysis: dict[str, Any]
    implication: dict[str, Any]
    sources: list[dict[str, Any]]
    validation: dict[str, Any]

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
