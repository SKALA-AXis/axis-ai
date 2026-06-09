from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agents.context.weekly_digest_agent import WeeklyDigestAgent, _deterministic_digest


def test_deterministic_digest_includes_card_ids() -> None:
    cards = [
        {"id": "CN-1", "title": "파트너십 체결"},
        {"id": "CN-2", "title": "클라우드 확장"},
    ]
    body = _deterministic_digest(
        cards=cards,
        week_iso="2026-W23",
        period_label="2026-06-02~2026-06-09",
        prev_digest=None,
    )
    assert body["source_card_ids"]
    assert "2건" in body["narrative"]


@patch("src.agents.context.weekly_digest_agent.SessionLocal")
def test_run_skips_when_no_cards(mock_session_local: MagicMock) -> None:
    mock_db = MagicMock()
    mock_session_local.return_value.__enter__.return_value = mock_db
    mock_db.execute.return_value.fetchall.return_value = []

    agent = WeeklyDigestAgent()
    result = agent.run(peer_id="samsung_sds", use_llm=False)
    assert result["skipped"] is True
