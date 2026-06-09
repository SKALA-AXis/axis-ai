from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agents.context.capability_evolution_agent import (
    CapabilityEvolutionAgent,
    _deterministic_windows,
    _normalize_windows,
)


def _sample_signals() -> list[dict]:
    return [
        {
            "id": "101",
            "business_area": "클라우드&AI",
            "period_year": 2026,
            "period_quarter": "Q1",
            "signal_type": "growth",
            "summary": "클라우드 매출 성장 가속",
            "confidence": 0.9,
        },
        {
            "id": "102",
            "business_area": "클라우드&AI",
            "period_year": 2026,
            "period_quarter": "Q1",
            "signal_type": "strategy",
            "summary": "AI 플랫폼 투자 확대",
            "confidence": 0.8,
        },
        {
            "id": "103",
            "business_area": "물류",
            "period_year": 2026,
            "period_quarter": "Q1",
            "signal_type": "business_update",
            "summary": "스마트물류 사업 확장",
            "confidence": 0.75,
        },
        {
            "id": "104",
            "business_area": "물류",
            "period_year": 2025,
            "period_quarter": "Q4",
            "signal_type": "orders_pipeline",
            "summary": "물류 수주 증가",
            "confidence": 0.7,
        },
        {
            "id": "105",
            "business_area": "IT서비스/SI",
            "period_year": 2025,
            "period_quarter": "Q4",
            "signal_type": "execution",
            "summary": "대형 SI 프로젝트 수행",
            "confidence": 0.65,
        },
    ]


def test_deterministic_windows_groups_by_business_area() -> None:
    windows = _deterministic_windows(_sample_signals(), period_label="2025Q4-2026Q1")
    areas = {w["business_area"] for w in windows}
    assert "클라우드&AI" in areas
    assert all(w["evidence_signal_ids"] for w in windows)


def test_normalize_windows_requires_evidence_ids() -> None:
    signals = _sample_signals()
    windows = _normalize_windows(
        [
            {
                "period": "2026Q1",
                "business_area": "클라우드&AI",
                "narrative": "클라우드·AI 투자가 가속화되고 있습니다.",
                "evidence_signal_ids": [],
                "confidence": 0.8,
            }
        ],
        signals=signals,
        period_label="2026Q1",
    )
    assert len(windows) == 1
    assert windows[0]["evidence_signal_ids"]


@patch("src.agents.context.capability_evolution_agent.SessionLocal")
def test_run_skips_when_insufficient_signals(mock_session_local: MagicMock) -> None:
    mock_db = MagicMock()
    mock_session_local.return_value.__enter__.return_value = mock_db
    mock_db.execute.return_value.fetchall.return_value = []

    agent = CapabilityEvolutionAgent()
    result = agent.run(peer_id="samsung_sds", use_llm=False)
    assert result["skipped"] is True


@patch("src.agents.context.capability_evolution_agent.SessionLocal")
def test_run_no_llm_produces_windows(mock_session_local: MagicMock) -> None:
    mock_db = MagicMock()
    mock_session_local.return_value.__enter__.return_value = mock_db
    rows = []
    for signal in _sample_signals():
        mapping = MagicMock()
        mapping._mapping = {
            "id": int(signal["id"]),
            "business_area": signal["business_area"],
            "period_year": signal["period_year"],
            "pq": signal["period_quarter"],
            "signal_type": signal["signal_type"],
            "sentiment": "positive",
            "summary": signal["summary"],
            "evidence_text": signal["summary"],
            "confidence": signal["confidence"],
            "raw_article_id": 1,
        }
        rows.append(mapping)
    mock_db.execute.return_value.fetchall.return_value = rows

    agent = CapabilityEvolutionAgent()
    result = agent.run(peer_id="samsung_sds", use_llm=False)
    assert result.get("skipped") is not True
    assert len(result.get("windows") or []) >= 1
