"""ITTrendAgent (PR #68, 5-phase pipeline) deterministic helper 단위 테스트.

외부 LLM 리뷰의 R-2 (reference id 우선순위) 및 R-5 (테스트 없음) 항목 보강.
DB / LLM 호출은 하지 않는다 — _phase3~5 / upsert 경로는 별도 E2E
(scripts/validate_global_trends.py) 로 검증한다.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

import pytest

from src.agents.it_trend_agent import (
    ITTrendInput,
    _classify_intensity,
    _dedupe_items_by_id,
    _empty_result,
    _extract_keywords_from_rows,
    _is_global_newsroom_row,
    _is_research_row,
    _make_source_analysis_id,
    _reference_issue_ids,
    _slugify,
)

# ──────────────────────────────────────────────────────────────────────────
# _slugify / _make_source_analysis_id  (design §7 — VARCHAR(100) hard cap)
# ──────────────────────────────────────────────────────────────────────────


def test_slugify_normalizes_korean_english_digits() -> None:
    assert _slugify("Agentic AI") == "agentic-ai"
    assert _slugify("에이전트 AI") == "에이전트-ai"
    assert _slugify("  spaced   keyword  ") == "spaced-keyword"
    assert _slugify("with_underscore!@#") == "withunderscore"


def test_slugify_empty_or_only_punct_returns_unknown() -> None:
    assert _slugify("") == "unknown"
    assert _slugify("###") == "unknown"
    assert _slugify("   ") == "unknown"


def test_slugify_respects_max_len_cap() -> None:
    long_kw = "a" * 200
    assert len(_slugify(long_kw, max_len=40)) == 40


def test_make_source_analysis_id_normal_under_100_chars() -> None:
    sid = _make_source_analysis_id("global-20260526-023012", idx=1, keyword="cloud")
    assert sid == "global-20260526-023012-001-cloud"
    assert len(sid) <= 100


def test_make_source_analysis_id_pads_idx_to_three_digits() -> None:
    sid = _make_source_analysis_id("global-20260526-023012", idx=7, keyword="ai")
    assert "-007-" in sid


def test_make_source_analysis_id_falls_back_to_sha1_when_too_long() -> None:
    """비정상적으로 긴 keyword 가 들어와도 100 자 hard cap 을 넘지 않는다."""
    huge_keyword = "에이전트" * 50  # slug 200 자 정도 — batch_id 와 합치면 100 초과
    sid = _make_source_analysis_id("global-20260526-023012", idx=12, keyword=huge_keyword)
    assert len(sid) <= 100
    # sha1 fallback 형식: <batch>-<idx>-<8charhash>
    assert "-012-" in sid


def test_make_source_analysis_id_unique_when_slug_collides() -> None:
    """동일 slug 두 번 만들어도 idx 가 다르면 충돌 안 함 (V29 unique 보호 R-3)."""
    a = _make_source_analysis_id("global-20260526-023012", idx=1, keyword="generative ai")
    b = _make_source_analysis_id("global-20260526-023012", idx=2, keyword="generative ai")
    assert a != b
    assert a.endswith("-001-generative-ai")
    assert b.endswith("-002-generative-ai")


# ──────────────────────────────────────────────────────────────────────────
# row 분류 — _is_global_newsroom_row / _is_research_row
# ──────────────────────────────────────────────────────────────────────────


def test_is_global_newsroom_row_accepts_normalized_source_types() -> None:
    """fetch_global_trend_inputs 가 정규화한 source_type 만 newsroom 으로 인정."""
    assert _is_global_newsroom_row({"source_type": "global_newsroom"})
    assert _is_global_newsroom_row({"source_type": "company_newsroom"})
    assert not _is_global_newsroom_row({"source_type": "trend_report"})
    assert not _is_global_newsroom_row({"source_type": "news"})


def test_is_research_row_requires_trend_report_and_spri_or_bcg() -> None:
    """크롤러 source_type=trend_report 와 source_name in {spri, bcg} 모두 만족 시만 인정."""
    assert _is_research_row({"source_type": "trend_report", "source_name": "spri"})
    assert _is_research_row({"source_type": "trend_report", "source_name": "bcg"})
    assert _is_research_row({"source_type": "trend_report", "publisher": "SPRi"})
    # source_type 만 맞고 source_name 다르면 reject
    assert not _is_research_row({"source_type": "trend_report", "source_name": "naver"})
    # source_name 만 맞고 source_type 이 다르면 reject
    assert not _is_research_row({"source_type": "news", "source_name": "spri"})


def test_is_research_row_is_case_insensitive_for_source_name() -> None:
    assert _is_research_row({"source_type": "trend_report", "source_name": "SPRi"})
    assert _is_research_row({"source_type": "TREND_REPORT", "source_name": "BCG"})


# ──────────────────────────────────────────────────────────────────────────
# _reference_issue_ids — R-2 priority swap (nested bundle_id > wrapper source_id)
# ──────────────────────────────────────────────────────────────────────────


def test_reference_issue_ids_prefers_integrated_bundle_id_over_wrapper_source_id() -> None:
    """nested ``integrated_issue.bundle_id`` 가 wrapper ``source_id`` 보다 우선."""
    items = [
        {
            "source_id": "raw-12345",
            "integrated_issue": {"bundle_id": "bundle-cloud-001"},
            "analysis_result": {"cluster_id": "cl-99"},
        }
    ]
    assert _reference_issue_ids(items) == ["bundle-cloud-001"]


def test_reference_issue_ids_falls_back_to_cluster_id_if_no_bundle_id() -> None:
    items = [
        {
            "source_id": "raw-12345",
            "analysis_result": {"cluster_id": "cl-99"},
        }
    ]
    assert _reference_issue_ids(items) == ["cl-99"]


def test_reference_issue_ids_uses_wrapper_only_when_nested_missing() -> None:
    items = [{"source_id": "raw-67890"}]
    assert _reference_issue_ids(items) == ["raw-67890"]


def test_reference_issue_ids_dedupes_and_handles_non_dict() -> None:
    items = [
        {"integrated_issue": {"bundle_id": "b-1"}},
        {"integrated_issue": {"bundle_id": "b-1"}},
        "not-a-dict",  # type: ignore[list-item]
        {"integrated_issue": {"bundle_id": "b-2"}},
    ]
    assert _reference_issue_ids(items) == ["b-1", "b-2"]


def test_reference_issue_ids_empty_input_returns_empty_list() -> None:
    assert _reference_issue_ids([]) == []


# ──────────────────────────────────────────────────────────────────────────
# _extract_keywords_from_rows — 사전 keyword 후보로 substring match
# ──────────────────────────────────────────────────────────────────────────


def test_extract_keywords_counts_known_themes_in_title_and_content() -> None:
    rows = [
        {"title": "NVIDIA unveils new GPU cloud platform", "content": "AI agent demo"},
        {"title": "Microsoft Copilot expands", "content": "Generative AI rollout"},
        {"title": "Apple announces partnership with TSMC", "content": ""},
    ]
    counts = _extract_keywords_from_rows(rows)
    assert isinstance(counts, Counter)
    assert counts["gpu"] >= 1
    assert counts["cloud"] >= 1
    assert counts["copilot"] >= 1
    assert counts["generative ai"] >= 1
    assert counts["partnership"] >= 1


def test_extract_keywords_is_case_insensitive() -> None:
    rows = [{"title": "GENERATIVE AI", "content": ""}]
    counts = _extract_keywords_from_rows(rows)
    assert counts["generative ai"] >= 1


def test_extract_keywords_empty_rows_returns_empty_counter() -> None:
    counts = _extract_keywords_from_rows([])
    assert counts == Counter()


# ──────────────────────────────────────────────────────────────────────────
# _classify_intensity
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mention_count,expected",
    [
        (0, "weak"),
        (1, "weak"),
        (4, "weak"),
        (5, "moderate"),
        (14, "moderate"),
        (15, "strong"),
        (200, "strong"),
    ],
)
def test_classify_intensity(mention_count: int, expected: str) -> None:
    assert _classify_intensity(mention_count) == expected


# ──────────────────────────────────────────────────────────────────────────
# _dedupe_items_by_id
# ──────────────────────────────────────────────────────────────────────────


def test_dedupe_items_by_id_drops_repeated_ids() -> None:
    items = [
        {"id": 1, "title": "a"},
        {"id": 2, "title": "b"},
        {"id": 1, "title": "a-duplicate"},
        {"id": 3, "title": "c"},
    ]
    deduped = _dedupe_items_by_id(items)
    assert [r["title"] for r in deduped] == ["a", "b", "c"]


def test_dedupe_items_by_id_keeps_rows_without_id() -> None:
    items = [{"title": "x"}, {"title": "y"}, {"id": 1, "title": "z"}]
    deduped = _dedupe_items_by_id(items)
    assert len(deduped) == 3


# ──────────────────────────────────────────────────────────────────────────
# _empty_result — 빈 입력 graceful 처리
# ──────────────────────────────────────────────────────────────────────────


def test_empty_result_returns_safe_dict_with_validation_false() -> None:
    trend_input = ITTrendInput(trend_items=[], period="2026-05-26")
    result = _empty_result(
        batch_id="global-20260526-023012",
        generated_at=datetime(2026, 5, 26, 2, 30, 12, tzinfo=UTC),
        trend_input=trend_input,
        snapshots=[],
        reasoning_steps=[],
        warning="Phase 1 snapshot empty",
    )
    assert result["agent"] == "ITTrendAgent"
    assert result["analysis_id"] == "global-20260526-023012"
    assert result["persisted_row_count"] == 0
    assert result["validation"]["pass"] is False
    assert result["validation"]["reason"] == "Phase 1 snapshot empty"
    assert result["trend_detections"] == []
    assert result["rows"] == []
    assert result["final_one_liner"] == ""

    context = result["trend_context"]
    assert context["trend_summary"] == ""
    assert context["trend_lines"] == []
    assert context["signals"] == []
    assert context["validation"]["pass"] is False
