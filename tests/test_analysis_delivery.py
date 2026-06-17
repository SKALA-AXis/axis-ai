from src.pipeline import analysis_delivery


def test_document_analysis_targets_excludes_market_data_from_card_generation(monkeypatch):
    monkeypatch.setattr(
        analysis_delivery,
        "get_articles_by_ids",
        lambda ids: [
            {"id": 10, "source_type": "market_data"},
            {"id": 11, "source_type": "job"},
            {"id": 12, "source_type": "search_trend"},
        ],
    )

    targets = analysis_delivery._document_analysis_targets(
        {
            "official_document_ids": [],
            "parsed_document_ids": [],
            "industry_document_ids": [],
            "structured_signal_ids": [10, 11, 10, 12],
        }
    )

    assert targets["structured_signal"] == [11, 12]


def test_document_analysis_targets_preserves_non_structured_document_targets(monkeypatch):
    monkeypatch.setattr(analysis_delivery, "get_articles_by_ids", lambda ids: [])

    targets = analysis_delivery._document_analysis_targets(
        {
            "official_document_ids": [1],
            "parsed_document_ids": [2],
            "industry_document_ids": [3],
            "structured_signal_ids": [],
        }
    )

    assert targets == {
        "official_document": [1],
        "parsed_document": [2],
        "industry_document": [3],
        "structured_signal": [],
    }
