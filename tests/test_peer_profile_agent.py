# 작성일: 2026-05-29
# 작성자: 박지원
# 변경이력:
#   2026-05-29 박지원 — peer profile 에이전트 및 스냅샷 파이프라인 추가
#   2026-06-15 최종민 — _FakeLLM.invoke 가 config 인자 수용하도록 시그니처 정합
from __future__ import annotations

from typing import Any

from src.agents.peer_profile_agent import PeerProfileAgent
from src.services.profile_evidence_selector import ProfileEvidenceSelector
from src.services.profile_snapshot_summarizer import ProfileSnapshotSummarizer
from src.services.profile_snapshot_validator import ProfileSnapshotValidator


def _input_pack() -> dict[str, Any]:
    return {
        "company": {"id": "lg_cns", "name": "LG CNS"},
        "period": "2026Q1",
        "business_area_evidence": [
            {
                "business_area": "클라우드&AI",
                "claim": "클라우드 전환, MSP, AI 적용을 포함한 AX 지원 사업",
                "source_ref": {"table": "raw_articles", "id": 14499},
                "confidence": 0.9,
            }
        ],
        "financial_evidence": [
            {
                "business_area": "클라우드&AI",
                "metric": "revenue_total",
                "metric_scope": "segment",
                "value": 7950,
                "unit": "억원",
                "yoy_pct": 6.7,
                "period": "2026Q1",
                "source_ref": {"table": "raw_article_financial_metrics", "id": 48589},
                "confidence": 0.88,
            }
        ],
        "direction_evidence": [
            {
                "business_area": "ai_ax",
                "claim": "AgenticWorks 기반 사업 확장이 본격화되고 있다.",
                "signal_type": "business_update",
                "sentiment": "positive",
                "source_ref": {"table": "raw_article_business_signals", "id": 300465},
                "confidence": 0.78,
            }
        ],
        "execution_evidence": [
            {
                "business_area": "스마트 엔지니어링",
                "claim": "Factova 기반 스마트팩토리 솔루션을 확대하고 있다.",
                "source_ref": {"table": "raw_articles", "id": 36282},
                "confidence": 0.85,
            }
        ],
        "source_coverage": {
            "securities_report": {
                "available": False,
                "status": "unavailable",
                "reason": "최근 3년 내 증권사 리포트 없음",
            }
        },
        "source_index": [],
    }


class _Builder:
    def build(self, peer_id: str) -> dict[str, Any]:
        assert peer_id == "lg_cns"
        return _input_pack()


class _FakeMessage:
    content = """
{
  "company_id": "lg_cns",
  "company_name": "LG CNS",
  "schema_version": "peer-profile-snapshot-v1",
  "generated_at": "2026-06-01T00:00:00+00:00",
  "one_liner": "LG CNS는 클라우드&AI 중심의 IT서비스 기업입니다.",
  "company_summary": "LG CNS는 클라우드&AI와 스마트 엔지니어링을 중심으로 사업을 전개합니다.",
  "business_areas": [
    {
      "name": "클라우드&AI",
      "summary": "클라우드 전환과 AI 적용을 지원합니다.",
      "core_capabilities": ["MSP", "AI"],
      "recent_direction": "AgenticWorks 기반 사업 확대",
      "evidence_texts": [],
      "source_refs": [{"raw_article_id": 14499}]
    }
  ],
  "core_capabilities": ["MSP", "AI"],
  "recent_changes": [],
  "evidence_digest": [],
  "financial_summary": {},
  "market_view": {
    "status": "unavailable",
    "reason": "",
    "positive_points": [],
    "watch_points": [],
    "valuation_notes": [],
    "source_refs": []
  },
  "capability_evolution": {
    "period": {"from": "", "to": ""},
    "overall_change": "",
    "changes": [],
    "watch_points": []
  },
  "cautions": [],
  "source_coverage": {},
  "source_index": []
}
"""


class _FakeLLM:
    def __init__(self) -> None:
        self.called = False

    def invoke(self, prompt: str, config: object | None = None, **_: object) -> _FakeMessage:
        # 실제 ChatOpenAI.invoke 처럼 config(LangChain RunnableConfig·Langfuse tracing) 수용.
        self.called = True
        assert "evidence_pack" in prompt
        return _FakeMessage()


def test_peer_profile_agent_builds_valid_snapshot() -> None:
    agent = PeerProfileAgent(
        input_builder=_Builder(),
        evidence_selector=ProfileEvidenceSelector(),
        summarizer=ProfileSnapshotSummarizer(),
        validator=ProfileSnapshotValidator(),
    )

    snapshot = agent.build_snapshot("lg_cns")

    assert snapshot["company_id"] == "lg_cns"
    assert snapshot["schema_version"] == "peer-profile-snapshot-v1"
    assert snapshot["validation"]["is_valid"] is True
    assert snapshot["business_areas"]
    assert snapshot["financial_summary"] == {"company_total": {}, "segment_revenue": []}
    assert snapshot["market_view"]["status"] == "unavailable"
    assert snapshot["source_coverage"]["securities_report"]["status"] == "unavailable"
    assert any(change["business_area"] == "클라우드&AI" for change in snapshot["recent_changes"])


def test_peer_profile_agent_builds_snapshot_with_llm() -> None:
    llm = _FakeLLM()
    agent = PeerProfileAgent(
        input_builder=_Builder(),
        evidence_selector=ProfileEvidenceSelector(),
        summarizer=ProfileSnapshotSummarizer(),
        validator=ProfileSnapshotValidator(),
        llm=llm,
    )

    snapshot = agent.build_snapshot("lg_cns", use_llm=True)

    assert llm.called is True
    assert snapshot["company_id"] == "lg_cns"
    assert snapshot["company_summary"]
    assert snapshot["validation"]["is_valid"] is True
