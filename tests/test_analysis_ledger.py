# 작성일: 2026-06-09
# 작성자: 최종민
# 변경이력:
#   2026-06-09 최종민 — ContextPackAssembler·주간 다이제스트 도입과 함께 ledger 테스트 추가 및 ledger 읽기 경로 정리
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.middleware.analysis_ledger import AnalysisLedger, _normalize_entry


def test_normalize_entry_skips_empty_conclusion() -> None:
    assert (
        _normalize_entry(
            row_id=1,
            analysis_id="a-1",
            analysis_type="insight",
            conclusion="",
            confidence=0.9,
            source_card_ids=[],
            sk_ax_implication=None,
            created_at="2026-06-01",
            source="insight_reports",
        )
        is None
    )


@patch("src.middleware.analysis_ledger.SessionLocal")
def test_fetch_top_n_merges_insight_and_legacy(mock_session_local: MagicMock) -> None:
    mock_db = MagicMock()
    mock_session_local.return_value.__enter__.return_value = mock_db

    insight_row = {
        "id": "uuid-insight",
        "final_one_liner": "클라우드 파트너십 확대",
        "confidence": 0.85,
        "source_card_ids": ["CN-1"],
        "sk_ax_implication": "SK AX 기회",
        "payload": {"strategy_label": "expand"},
        "created_at": "2026-06-08T00:00:00+00:00",
    }
    legacy_payload = [
        {
            "id": 99,
            "analysis_id": "peer-old",
            "analysis_type": "peer",
            "peer_ids": ["samsung_sds"],
            "conclusion_one_liner": "레거시 판단",
            "confidence": 0.8,
            "source_card_ids": ["CN-0"],
            "created_at": "2026-05-01T00:00:00+00:00",
            "included_in_pack": True,
        }
    ]

    def execute_side_effect(statement, params=None):
        sql = str(statement)
        result = MagicMock()
        if "FROM insight_reports" in sql:
            mappings = MagicMock()
            mappings.all.return_value = [insight_row]
            result.mappings.return_value = mappings
        elif "FROM mixer_results" in sql:
            mappings = MagicMock()
            mappings.all.return_value = []
            result.mappings.return_value = mappings
        elif "legacy_payload" in sql:
            legacy_row = MagicMock()
            legacy_row.__getitem__.return_value = legacy_payload
            result.fetchone.return_value = legacy_row
        else:
            result.fetchall.return_value = []
            result.fetchone.return_value = None
        return result

    mock_db.execute.side_effect = execute_side_effect

    entries = AnalysisLedger.fetch_top_n("samsung_sds", top_n=5, retention_days=90)
    assert len(entries) == 2
    assert entries[0]["conclusion_one_liner"] == "클라우드 파트너십 확대"
    assert entries[0]["source"] == "insight_reports"
    assert any(e["source"] == "legacy_payload" for e in entries)
