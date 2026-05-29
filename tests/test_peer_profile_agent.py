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
        "source_index": [],
    }


class _Builder:
    def build(self, peer_id: str) -> dict[str, Any]:
        assert peer_id == "lg_cns"
        return _input_pack()


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
    assert any(change["business_area"] == "클라우드&AI" for change in snapshot["recent_changes"])
