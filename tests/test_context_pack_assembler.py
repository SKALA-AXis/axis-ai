from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.analysis.context_pack_models import PeerContextPack
from src.analysis.models import AnalysisInputBundle, TimelineEntry
from src.services.context_pack_assembler import (
    ContextPackAssembler,
    resolve_context_peers,
)


def _stub_bundle() -> AnalysisInputBundle:
    return AnalysisInputBundle(
        bundle_id="news:42",
        cluster_id="42",
        source_type="news",
        companies=["samsung_sds"],
        sectors=["ax"],
        event_type="partnership",
        items=[],
        facts=[],
        evidence_snippets=[],
        sources=[],
        metadata={},
    )


def test_resolve_context_peers_prefers_integrated_issue_main_company() -> None:
    peers = resolve_context_peers(
        _stub_bundle(),
        None,
        {"main_company": "lg_cns", "mentioned_peer_companies": ["samsung_sds"]},
    )
    assert peers[0] == "lg_cns"
    assert "samsung_sds" in peers


@patch("src.services.context_pack_assembler.AnalysisLedger.fetch_top_n")
@patch("src.services.context_pack_assembler.SessionLocal")
def test_assembler_builds_pack_with_timeline_and_ledger(
    mock_session_local: MagicMock,
    mock_fetch_ledger: MagicMock,
) -> None:
    mock_db = MagicMock()
    mock_session_local.return_value.__enter__.return_value = mock_db

    timeline_row = MagicMock()
    timeline_row._mapping = {
        "company_id": "samsung_sds",
        "event_date": "2026-06-01",
        "card_id": "CN-001",
        "event_type": "partnership",
        "sector": "ax",
        "headline": "테스트",
        "importance": "high",
        "importance_score": 0.9,
    }

    def execute_side_effect(statement, params=None):
        sql = str(statement)
        result = MagicMock()
        if "peer_event_timeline" in sql:
            result.fetchall.return_value = [timeline_row]
        elif "weekly_digest" in sql:
            weekly_row = MagicMock()
            weekly_row._mapping = {
                "weekly": {
                    "version": "v1",
                    "digests": [
                        {
                            "week_iso": "2026-W23",
                            "period_label": "2026-06-02~2026-06-09",
                            "narrative": "클라우드 파트너십 가속",
                            "delta_vs_prev": ["신규 파트너십"],
                            "strategy_label": "expand",
                            "source_card_ids": ["CN-001"],
                            "confidence": 0.8,
                        }
                    ],
                }
            }
            result.fetchone.return_value = weekly_row
        elif "peer_plus_payload" in sql:
            result.fetchone.return_value = None
        elif "sector_pulse" in sql:
            result.fetchall.return_value = []
        elif "peer_financial_trend" in sql:
            result.fetchall.return_value = []
        elif "raw_article_business_signals" in sql:
            count_row = MagicMock()
            count_row._mapping = {"signal_count": 100, "metric_count": 20}
            result.fetchone.return_value = count_row
        elif "today_insight_reports" in sql:
            result.mappings.return_value.all.return_value = []
        else:
            result.fetchall.return_value = []
            result.fetchone.return_value = None
        return result

    mock_db.execute.side_effect = execute_side_effect
    mock_fetch_ledger.return_value = [
        {
            "id": 7,
            "analysis_type": "insight",
            "conclusion_one_liner": "클라우드 파트너십 가속",
            "confidence": 0.8,
        }
    ]

    pack = ContextPackAssembler().assemble(
        input_bundle=_stub_bundle(),
        integrated_issue={"main_company": "samsung_sds"},
    )

    assert isinstance(pack, PeerContextPack)
    assert pack.peer_ids == ["samsung_sds"]
    assert len(pack.peer_event_timeline_recent) == 1
    assert pack.analysis_ledger_by_peer.get("samsung_sds")
    assert "peer_event_timeline_recent" in pack.provenance.used_layers
    assert "analysis_ledger_by_peer" in pack.provenance.used_layers
    assert pack.weekly_digest_by_peer.get("samsung_sds")
    assert pack.coverage.has_weekly_digest
    assert "weekly_digest_by_peer" in pack.provenance.used_layers


def test_pack_to_analysis_context_maps_timeline() -> None:
    pack = PeerContextPack(
        peer_ids=["samsung_sds"],
        peer_event_timeline_recent=[
            TimelineEntry(
                company_id="samsung_sds",
                event_date="2026-06-01",
                card_id="CN-001",
                event_type="partnership",
                sector="ax",
                headline="테스트",
                importance="high",
                importance_score=0.9,
            )
        ],
        weekly_digest_by_peer={"samsung_sds": {"narrative": "주간 요약"}},
    )
    ctx = pack.to_analysis_context()
    assert len(ctx.peer_event_timeline_recent) == 1
    assert ctx.weekly_digest_by_peer.get("samsung_sds", {}).get("narrative") == "주간 요약"
