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
    _phase2_trends,
    _reference_issue_ids,
    _resolve_previous_trend_context,
    _slugify,
    _trend_confidence_for_keyword,
    alignment_match_terms,
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
# _trend_confidence_for_keyword — keyword 별 evidence 기반 confidence
# ──────────────────────────────────────────────────────────────────────────


def test_trend_confidence_rewards_keyword_level_evidence() -> None:
    strong = _trend_confidence_for_keyword(
        det={
            "theme": "cloud",
            "mention_count": 50,
            "leading_companies": ["microsoft", "google", "amazon"],
        },
        peers=[
            {"alignment_type": "aligned", "evidence_card_ids": ["1", "2"]},
            {"alignment_type": "aligned", "evidence_card_ids": ["3"]},
        ],
        evidence_raw_ids=[1, 2, 3, 4, 5],
        has_llm_title=True,
        has_llm_summary=True,
        llm_batch_confidence=0.7,
    )
    weak = _trend_confidence_for_keyword(
        det={"theme": "quantum", "mention_count": 3, "leading_companies": []},
        peers=[
            {"alignment_type": "missing", "evidence_card_ids": []},
            {"alignment_type": "missing", "evidence_card_ids": []},
        ],
        evidence_raw_ids=[],
        has_llm_title=False,
        has_llm_summary=False,
        llm_batch_confidence=0.7,
    )

    assert strong > weak
    assert strong != 0.7
    assert 0.0 <= weak <= 1.0


def test_trend_confidence_clips_llm_batch_confidence() -> None:
    score = _trend_confidence_for_keyword(
        det={"theme": "gpu", "mention_count": 20, "leading_companies": ["nvidia"]},
        peers=[],
        evidence_raw_ids=[1],
        has_llm_title=True,
        has_llm_summary=True,
        llm_batch_confidence=9.9,
    )
    assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────
# _phase2_trends — frequency_delta_pct (previous batch mention_count 기준)
# ──────────────────────────────────────────────────────────────────────────


def _gpu_newsroom_rows(count: int) -> list[dict[str, str]]:
    return [{"title": "NVIDIA GPU cloud platform update", "content": "GPU"} for _ in range(count)]


def test_phase2_trends_cold_start_delta_is_zero() -> None:
    """previous_trend_context 없으면 design §13: frequency_delta_pct = 0."""
    rows = _gpu_newsroom_rows(6)
    detections = _phase2_trends(
        snapshots=[{"company_id": "nvidia", "card_count": 6, "top_themes": ["gpu"]}],
        global_rows=rows,
        research_rows=[],
        previous_trend_context=None,
        min_mention_count=3,
        max_trend_count=8,
        focus_themes=[],
    )
    gpu = next(d for d in detections if d["theme"] == "gpu")
    assert gpu["frequency_delta_pct"] == 0.0


def test_phase2_trends_uses_previous_mention_count_for_delta() -> None:
    rows = _gpu_newsroom_rows(10)
    detections = _phase2_trends(
        snapshots=[{"company_id": "nvidia", "card_count": 10, "top_themes": ["gpu"]}],
        global_rows=rows,
        research_rows=[],
        previous_trend_context={"keyword_counts": {"gpu": 5}},
        min_mention_count=3,
        max_trend_count=8,
        focus_themes=[],
    )
    gpu = next(d for d in detections if d["theme"] == "gpu")
    assert gpu["frequency_delta_pct"] == 100.0


def test_phase2_trends_new_keyword_with_prior_batch_still_zero_delta() -> None:
    """직전 batch 에 없던 keyword 는 delta 0 (더 이상 +100% hardcode 아님)."""
    rows = _gpu_newsroom_rows(6)
    detections = _phase2_trends(
        snapshots=[{"company_id": "nvidia", "card_count": 6, "top_themes": ["gpu"]}],
        global_rows=rows,
        research_rows=[],
        previous_trend_context={"keyword_counts": {"cloud": 8}},
        min_mention_count=3,
        max_trend_count=8,
        focus_themes=[],
    )
    gpu = next(d for d in detections if d["theme"] == "gpu")
    assert gpu["frequency_delta_pct"] == 0.0


def test_resolve_previous_trend_context_keeps_explicit_payload() -> None:
    explicit = {"keyword_counts": {"gpu": 4}}
    assert _resolve_previous_trend_context(explicit) == explicit


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


# ──────────────────────────────────────────────────────────────────────────
# strategic_note 근거 기반 생성 + 한국어 강제 (사용자 리포트: 영어·막연 서술)
# ──────────────────────────────────────────────────────────────────────────


def test_strategic_note_prompt_injects_evidence_titles_and_korean_directive(monkeypatch):
    import json as _json

    from src.agents import it_trend_agent as mod

    captured: dict[str, str] = {}

    class _FakeLLM:
        def invoke(self, prompt, config=None):  # noqa: ANN001
            captured["prompt"] = prompt

            class _R:
                content = _json.dumps(
                    {
                        "notes": [
                            {
                                "theme": "agentic ai",
                                "peer_id": "samsung_sds",
                                "strategic_note": "삼성SDS는 운영 플랫폼 동향으로 따라가는 중",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )

            return _R()

    monkeypatch.setattr(mod, "_get_llm", lambda: _FakeLLM())

    result = {
        "agentic ai": [
            {
                "peer_id": "samsung_sds",
                "alignment_type": "aligned",
                "peer_mention_count": 3,
                "global_mention_count": 10,
                "_evidence_titles": ["삼성SDS, 생성형 AI 운영 플랫폼 출시"],
                "strategic_note": "",
            }
        ]
    }
    detections = [
        {"theme": "agentic ai", "intensity": "strong", "leading_companies": ["microsoft"]}
    ]

    out = mod._llm_fill_strategic_notes(result, detections)

    # 실제 수집 제목이 프롬프트 근거로 주입됐는가
    assert "삼성SDS, 생성형 AI 운영 플랫폼 출시" in captured["prompt"]
    # 한국어 강제 + generic 영어 금지 지시가 있는가
    assert "반드시 한국어로" in captured["prompt"]
    assert "like Amazon and Microsoft" in captured["prompt"]  # 금지 예시로 포함
    # note 가 채워졌는가
    assert out["agentic ai"][0]["strategic_note"] == "삼성SDS는 운영 플랫폼 동향으로 따라가는 중"


def test_strategic_note_payload_marks_missing_evidence(monkeypatch):
    from src.agents import it_trend_agent as mod

    captured: dict[str, str] = {}

    class _FakeLLM:
        def invoke(self, prompt, config=None):  # noqa: ANN001
            captured["prompt"] = prompt

            class _R:
                content = '{"notes": []}'

            return _R()

    monkeypatch.setattr(mod, "_get_llm", lambda: _FakeLLM())

    result = {
        "quantum": [
            {
                "peer_id": "posco_dx",
                "alignment_type": "missing",
                "peer_mention_count": 0,
                "global_mention_count": 5,
                "_evidence_titles": [],
                "strategic_note": "",
            }
        ]
    }
    detections = [{"theme": "quantum", "intensity": "weak", "leading_companies": ["ibm"]}]

    mod._llm_fill_strategic_notes(result, detections)

    # 근거 없을 때 추측 금지 규칙이 프롬프트에 명시돼 있는가
    assert "관련 공개 동향 미확인" in captured["prompt"]
    assert '"evidence_titles": []' in captured["prompt"]


# ──────────────────────────────────────────────────────────────────────────
# alignment_match_terms  (피어 정합 한·영 별칭 — 영문 theme ↔ 한글 card_news 매칭)
# ──────────────────────────────────────────────────────────────────────────
def test_alignment_match_terms_expands_korean_aliases_for_llm() -> None:
    terms = alignment_match_terms("llm")
    assert "llm" in terms
    assert "대규모 언어 모델" in terms
    # 영문 theme 만으로는 한글 card_news 와 안 맞으므로 한글 변형이 반드시 포함돼야 한다.
    assert any(_is_korean(t) for t in terms)


def test_alignment_match_terms_unknown_keyword_returns_self_only() -> None:
    assert alignment_match_terms("blockchain") == ["blockchain"]


def test_alignment_match_terms_normalizes_and_dedupes() -> None:
    terms = alignment_match_terms("  GPU  ")
    assert terms[0] == "gpu"
    assert len(terms) == len(set(terms))
    assert all(t == t.strip().lower() for t in terms)


def test_alignment_match_terms_empty_keyword_returns_empty() -> None:
    assert alignment_match_terms("") == []
    assert alignment_match_terms("   ") == []


def _is_korean(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)
