from __future__ import annotations

from src.services.analysis_units import (
    QUALITY_MISSING_ANALYSIS_PACKAGE,
    QUALITY_MISSING_INTEGRATED_ISSUE_ID,
    QUALITY_SUMMARY_ONLY_FALLBACK,
    analysis_unit_from_card,
    confidence_penalty_for_flags,
)


def test_analysis_unit_prefers_integrated_issue_and_detailed_package():
    issue_id = "11111111-1111-1111-1111-111111111111"
    unit = analysis_unit_from_card(
        {
            "id": "CN-1",
            "integrated_issue_id": issue_id,
            "title": "카드 제목",
            "summary_lines": ["표시 요약"],
            "source_raw_article_ids": [10],
            "evidence_payload": {
                "integrated_issue_id": issue_id,
                "evidence_refs": [{"evidence_ref_id": "fact-1", "text": "상세 근거"}],
                "analysis_package": {
                    "integrated_issue_id": issue_id,
                    "integrated_issue": {
                        "main_issue": "통합 이슈",
                        "source_article_ids": [10],
                    },
                    "analysis": {"analysis_summary": "전략 분석"},
                    "implication": {
                        "skax_implication": {"recommended_actions": ["대응방향"]}
                    },
                    "classification": {"sector": "ax"},
                    "validation": {"sc_score": 0.8},
                },
            },
        }
    )

    assert unit.integrated_issue_id == issue_id
    assert unit.card_id == "CN-1"
    assert unit.analysis["analysis_summary"] == "전략 분석"
    assert unit.implication["skax_implication"]["recommended_actions"] == ["대응방향"]
    assert unit.evidence_refs[0]["integrated_issue_id"] == issue_id
    assert QUALITY_MISSING_ANALYSIS_PACKAGE not in unit.quality_flags
    assert QUALITY_MISSING_INTEGRATED_ISSUE_ID not in unit.quality_flags


def test_summary_only_card_is_flagged_and_penalized():
    unit = analysis_unit_from_card(
        {
            "id": "CN-2",
            "title": "요약뿐인 카드",
            "summary_lines": ["요약 1", "요약 2"],
            "evidence_payload": {},
        }
    )

    assert QUALITY_MISSING_INTEGRATED_ISSUE_ID in unit.quality_flags
    assert QUALITY_MISSING_ANALYSIS_PACKAGE in unit.quality_flags
    assert QUALITY_SUMMARY_ONLY_FALLBACK in unit.quality_flags
    assert confidence_penalty_for_flags(unit.quality_flags) >= 0.45
