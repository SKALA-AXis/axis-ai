from __future__ import annotations

import json

from src.db.article_store import _card_news_insert_params
from src.db.integrated_issues import _evidence_refs, _source_rows, _upsert_params


def test_card_news_insert_params_preserve_integrated_issue_id():
    issue_id = "11111111-1111-1111-1111-111111111111"
    params = _card_news_insert_params(
        {
            "id": "CN-1",
            "company": "samsung_sds",
            "title": "삼성SDS, AI 협력 확대",
            "summary_lines": ["요약"],
            "source_raw_article_ids": [1],
            "sources": [{"title": "기사", "url": "https://example.com/a"}],
            "evidence_payload": {
                "integrated_issue_id": issue_id,
                "analysis_package": {
                    "integrated_issue_id": issue_id,
                    "integrated_issue": {"main_issue": "AI 협력"},
                    "analysis": {"analysis_summary": "전략 분석"},
                    "implication": {"skax_implication": {"recommended_actions": ["대응"]}},
                },
            },
        }
    )

    assert params["integrated_issue_id"] == issue_id
    payload = json.loads(params["evidence_payload"])
    assert payload["integrated_issue_id"] == issue_id
    assert payload["analysis_package"]["integrated_issue_id"] == issue_id


def test_integrated_issue_upsert_params_are_stable_by_issue_key():
    integrated_issue = {
        "issue_key": "cluster:42:integrated_issue_v3",
        "integration_schema_version": "integrated_issue_v3",
        "cluster_id": 42,
        "main_company": "samsung_sds",
        "headline": "삼성SDS, AI 협력 확대",
        "one_line_summary": "AI 협력 확대",
        "source_article_ids": [1, 2],
        "fact_basis": [
            {
                "fact_id": "fact-1",
                "evidence_text": "협력 발표",
                "source_article_ids": [1],
            }
        ],
        "representative_sources": [
            {"raw_article_id": 1, "title": "기사 1"},
            {"raw_article_id": 1, "title": "기사 1 중복"},
            {"raw_article_id": 2, "title": "기사 2"},
        ],
    }

    first = _upsert_params(integrated_issue, input_bundle=None)
    second = _upsert_params(dict(integrated_issue), input_bundle=None)

    assert first["issue_key"] == second["issue_key"]
    assert first["source_ids"] == [1, 2]
    assert len(_source_rows(integrated_issue, input_bundle=None)) == 2
    assert _evidence_refs(integrated_issue)[0]["fact_id"] == "fact-1"
