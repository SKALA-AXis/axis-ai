# 작성일: 2026-05-20
# 작성자: 박지원
# 변경이력:
#   2026-05-20 박지원 — 스케줄러 크롤러 수정 작업의 일부로 추가
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.analysis import document_analysis_materializer as materializer


def test_materialize_document_analysis_dispatches_and_upserts(monkeypatch):
    deleted: list[tuple[str, tuple[int, ...]]] = []
    upserted: dict[str, list[dict[str, Any]]] = {"metrics": [], "signals": []}

    monkeypatch.setattr(
        materializer,
        "get_articles_by_ids",
        lambda ids: [
            {
                "id": 1,
                "source_type": "dart",
                "source_name": "dart",
                "company": ["sk_ax"],
                "metadata": {},
                "parser_result": {"ok": True, "period": "2026Q1"},
            },
            {
                "id": 2,
                "source_type": "ir",
                "source_name": "ir",
                "company": ["sk_ax"],
                "metadata": {},
                "parser_result": {"ok": True, "period": "2026Q1"},
            },
            {
                "id": 3,
                "source_type": "news",
                "source_name": "naver",
                "metadata": {},
            },
        ],
    )
    monkeypatch.setattr(
        materializer,
        "financial_metrics_from_dart",
        lambda article, parser_result: [{"raw_article_id": article["id"], "source_type": "dart"}],
    )
    monkeypatch.setattr(
        materializer,
        "business_signals_from_dart",
        lambda article, parser_result: [{"raw_article_id": article["id"], "source_type": "dart"}],
    )

    fake_ir_module = SimpleNamespace(
        _financial_record=lambda article, parser_result: {"period": parser_result["period"]},
        _metrics_from_parser_result=lambda article, parser_result, financial_record: [
            {"raw_article_id": article["id"], "source_type": "ir"}
        ],
        _business_signals_from_parser_result=lambda article, parser_result, financial_record: [
            {"raw_article_id": article["id"], "source_type": "ir"}
        ],
    )
    monkeypatch.setattr(materializer.importlib, "import_module", lambda name: fake_ir_module)
    monkeypatch.setattr(
        materializer,
        "delete_raw_article_financial_metrics",
        lambda ids, *, source_type=None: deleted.append((source_type, tuple(ids))) or len(ids),
    )
    monkeypatch.setattr(
        materializer,
        "delete_raw_article_business_signals",
        lambda ids, *, source_type=None: deleted.append((source_type, tuple(ids))) or len(ids),
    )
    monkeypatch.setattr(
        materializer,
        "upsert_raw_article_financial_metrics",
        lambda rows: upserted["metrics"].extend(rows) or len(rows),
    )
    monkeypatch.setattr(
        materializer,
        "upsert_raw_article_business_signals",
        lambda rows: upserted["signals"].extend(rows) or len(rows),
    )

    result = materializer.materialize_document_analysis([1, 2, 3])

    assert result["analysis_document_ids"] == [1, 2]
    assert result["analysis_source_counts"] == {"dart": 1, "ir": 1}
    assert result["analysis_metric_count"] == 2
    assert result["analysis_signal_count"] == 2
    assert [row["source_type"] for row in upserted["metrics"]] == ["dart", "ir"]
    assert [row["source_type"] for row in upserted["signals"]] == ["dart", "ir"]
    assert ("dart", (1,)) in deleted
    assert ("ir", (2,)) in deleted
    assert all(source_type != "news" for source_type, _ in deleted)
