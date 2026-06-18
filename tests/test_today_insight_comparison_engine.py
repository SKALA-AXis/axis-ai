# 작성일: 2026-06-09
# 작성자: 최종민
# 변경이력:
#   2026-06-09 최종민 — today-insight 비교 엔진 테스트 신규 작성(dual-lane 비교 사실·executive 요약 후처리·헤드라인/한국어 카피·gpt-5.5 프롬프트)
#   2026-06-11 박지원 — ruff 포매팅 수정 적용 및 develop 브랜치 병합
#   2026-06-11 박진 — 챗봇 분석 워크플로우 개선(#132)
from __future__ import annotations

from datetime import date

from src.services.today_insight_comparison_engine import (
    build_comparison_facts,
    build_executive_summary_from_facts,
    build_primary_headline,
    build_ui_change_summary,
    compute_keyword_trend_facts,
    format_evidence_change_lines,
    is_generic_executive_text,
    polish_executive_output,
    trim_comparison_facts_for_prompt,
)


def test_hidden_gem_ma_security_card_ranks_primary() -> None:
    facts = build_comparison_facts(
        anchor_date=date(2026, 6, 9),
        window_days=60,
        current_issues=[],
        history_issues=[],
        cards=[
            {
                "id": "CN-MA-1",
                "peer_id": "lg_cns",
                "title": "LG CNS, 보안 업체 인수 추진",
                "summary_lines": ["보안 솔루션 인수", "중소 업체 대상"],
                "event_type": "ma",
                "primary_keyword_category": "security",
                "importance_score": 0.82,
                "implication": {
                    "exposure_score": 0.18,
                    "signals": {"cluster_size": 1},
                },
                "sources": [{"source_name": "AXIS News"}],
            },
            {
                "id": "CN-PRODUCT-1",
                "peer_id": "samsung_sds",
                "title": "삼성SDS 생성형 AI 제품 확대",
                "summary_lines": ["제품 고도화"],
                "event_type": "tech_release",
                "primary_keyword_category": "ax",
                "importance_score": 0.55,
                "implication": {"exposure_score": 0.72, "signals": {"cluster_size": 4}},
                "sources": [{}, {}, {}],
            },
        ],
        change_stats={
            "today_issue_count": 0,
            "recent_card_count": 2,
            "window_days": 60,
            "peer_delta_vs_window": [
                {
                    "key": "lg_cns",
                    "today_count": 1,
                    "historical_daily_avg": 0.2,
                    "delta_vs_daily_avg": 0.8,
                }
            ],
            "sector_delta_vs_window": [],
        },
        prior_reports=[],
    )

    gaps = facts["visibility_gaps"]
    assert gaps
    assert gaps[0]["id"] == "CN-MA-1"
    assert gaps[0]["label"] == "low_visibility_definite_event"

    primary = facts["primary_selection"]
    assert primary["must_include_hidden_gem"] is True
    assert primary["ids"][0] == "CN-MA-1"
    assert float(primary["items"][0]["salience_score"]) >= 0.70
    assert float(primary["items"][0]["exposure_score"]) < 0.40
    hint = str(gaps[0].get("narrative_hint") or "")
    assert "사업 영향이" in hint
    assert "큼지만" not in hint
    assert "우선 확인하세요" not in hint


def test_structural_facts_do_not_require_news_numbers() -> None:
    facts = build_comparison_facts(
        anchor_date=date(2026, 6, 9),
        window_days=14,
        current_issues=[
            {
                "id": "issue-1",
                "main_company": "lg_cns",
                "event_type": "partnership",
                "headline": "LG CNS MOU 체결",
                "one_line_summary": "공공 AX 협력",
                "sectors": ["ax"],
                "confidence": 0.8,
                "source_ids": [1],
            }
        ],
        history_issues=[
            {
                "id": "issue-old",
                "main_company": "lg_cns",
                "event_type": "partnership",
                "headline": "old",
            }
        ],
        cards=[],
        change_stats={
            "today_issue_count": 1,
            "recent_card_count": 0,
            "window_days": 14,
            "peer_delta_vs_window": [],
            "sector_delta_vs_window": [],
        },
    )

    structural = facts["structural"]
    assert any(item.get("metric") == "today_detected_count" for item in structural)
    assert any(item.get("metric") == "event_type_mix_shift" for item in structural)
    assert facts["coverage"]["mode"] in {"dual_lane_full", "salience_only", "structural_only"}


def test_dart_issue_gets_high_salience() -> None:
    facts = build_comparison_facts(
        anchor_date=date(2026, 6, 9),
        window_days=30,
        current_issues=[
            {
                "id": "issue-dart",
                "main_company": "samsung_sds",
                "event_type": "financial",
                "source_family": "dart",
                "headline": "단일판매 공급계약 공시",
                "one_line_summary": "공급계약 체결",
                "sectors": ["deal"],
                "confidence": 0.9,
                "source_ids": [1],
            }
        ],
        history_issues=[],
        cards=[],
        change_stats={
            "today_issue_count": 1,
            "recent_card_count": 0,
            "window_days": 30,
            "peer_delta_vs_window": [],
            "sector_delta_vs_window": [],
        },
    )

    candidate = facts["salience_candidates"][0]
    assert candidate["kind"] == "integrated_issue"
    assert float(candidate["salience_score"]) >= 0.80
    assert (
        "source_type:dart" in candidate["salience_triggers"]
        or "event_type:financial" in candidate["salience_triggers"]
    )


def test_compute_keyword_trend_facts_from_db_rows() -> None:
    facts = compute_keyword_trend_facts(
        [
            {"group_name": "AI 에이전트", "period": "2026-06-09", "ratio": 118.5},
            {"group_name": "AI 에이전트", "period": "2026-06-08", "ratio": 100.4},
            {"group_name": "사이버보안", "period": "2026-06-09", "ratio": 92.0},
            {"group_name": "사이버보안", "period": "2026-06-08", "ratio": 90.5},
        ]
    )

    assert len(facts) == 2
    agent = next(item for item in facts if item["group_name"] == "AI 에이전트")
    assert agent["ratio_delta"] == 18.1
    assert agent["metric"] == "keyword_search_ratio_delta"
    assert agent["source"] == "raw_articles.search_trend"


def test_security_sector_pulls_cyber_keyword_groups(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_fetch(*, anchor_date, groups, history_days):  # noqa: ANN001
        captured["groups"] = groups
        return [
            {"group_name": "사이버보안", "period": "2026-06-09", "ratio": 95.0},
            {"group_name": "사이버보안", "period": "2026-06-08", "ratio": 80.0},
        ]

    monkeypatch.setattr(
        "src.services.today_insight_comparison_engine._fetch_keyword_trend_rows",
        fake_fetch,
    )

    facts = build_comparison_facts(
        anchor_date=date(2026, 6, 9),
        window_days=14,
        current_issues=[],
        history_issues=[],
        cards=[
            {
                "id": "CN-SEC",
                "peer_id": "lg_cns",
                "title": "보안 업체 인수",
                "event_type": "ma",
                "primary_keyword_category": "security",
                "implication": {"exposure_score": 0.2},
                "sources": [{}],
            }
        ],
        change_stats={
            "today_issue_count": 0,
            "recent_card_count": 1,
            "window_days": 14,
            "peer_delta_vs_window": [],
            "sector_delta_vs_window": [],
        },
    )

    assert "사이버보안" in captured["groups"]
    assert facts["keyword_trends"]
    assert facts["keyword_trends"][0]["group_name"] == "사이버보안"
    assert any(item.get("metric") == "keyword_search_ratio_delta" for item in facts["structural"])


def test_format_evidence_change_lines_and_ui_chips() -> None:
    comparison = {
        "keyword_trends": [
            {
                "group_name": "AI 에이전트",
                "ratio_delta": 18.1,
                "latest_ratio": 118.5,
            }
        ],
        "visibility_gaps": [
            {
                "peer_label": "LG CNS",
                "title": "보안 업체 인수",
                "exposure_score": 0.18,
                "salience_score": 0.85,
            }
        ],
        "structural": [
            {
                "metric": "peer_activity_delta",
                "peer_label": "LG CNS",
                "today_count": 1,
                "delta_vs_daily_avg": 0.8,
            }
        ],
        "coverage": {"hidden_gem_count": 1},
        "primary_selection": {
            "items": [{"label": "low_visibility_definite_event", "title": "보안 업체 인수"}]
        },
    }

    changes = format_evidence_change_lines(comparison)
    assert any("AI 에이전트 검색지수" in line for line in changes)
    assert any("영향은 크지만" in line for line in changes)
    assert not any("salience" in line.lower() for line in changes)

    chips = build_ui_change_summary(
        default_rows=[
            {"label": "오늘 감지된 변화", "value": "3건"},
            {"label": "비교 기준", "value": "최근 60일"},
            {"label": "핵심 축", "value": "-"},
        ],
        comparison_facts=comparison,
    )
    assert chips[1]["label"] == "검색지수 변화"
    assert "AI 에이전트" in chips[1]["value"]


def test_trim_comparison_facts_for_prompt_keeps_primary_lane() -> None:
    facts = {
        "primary_selection": {
            "items": [{"title": "핵심 이벤트", "label": "high_salience_visible"}]
        },
        "salience_candidates": [
            {
                "title": f"후보 {idx}",
                "salience_triggers": [f"trigger-{idx}", f"extra-{idx}"],
                "card_attached_numbers": [{"label": "a"}, {"label": "b"}, {"label": "c"}],
            }
            for idx in range(12)
        ],
        "keyword_trends": [{"group_name": "AI 에이전트", "ratio_delta": 1.2}],
        "structural": [{"metric": "peer_activity_delta"}],
    }
    trimmed = trim_comparison_facts_for_prompt(facts)
    assert trimmed["primary_selection"]["items"][0]["title"] == "핵심 이벤트"
    assert len(trimmed["salience_candidates"]) == 8
    assert "salience_triggers" not in trimmed["salience_candidates"][0]
    assert len(trimmed["salience_candidates"][0]["card_attached_numbers"]) <= 2


def test_primary_headline_is_title_only_without_editorial_prefix() -> None:
    headline = build_primary_headline(
        {
            "primary_selection": {
                "items": [
                    {
                        "title": "LGCNS, 앤트로픽과 클로드 엔터프라이즈 도입 계약 체결",
                        "label": "low_visibility_definite_event",
                    }
                ]
            }
        }
    )

    assert headline == "LGCNS, 앤트로픽과 클로드 엔터프라이즈 도입 계약 체결"
    assert "보도는 적지만" not in headline
    assert "우선 확인" not in headline


def test_executive_summary_uses_plain_korean_not_internal_scores() -> None:
    comparison = {
        "keyword_trends": [
            {"group_name": "AI 에이전트", "ratio_delta": -43.5, "latest_ratio": 74.2}
        ],
        "primary_selection": {
            "items": [
                {
                    "title": "LGCNS, 앤트로픽과 클로드 엔터프라이즈 도입 계약 체결",
                    "label": "low_visibility_definite_event",
                    "narrative_hint": (
                        "LG CNS 계약은 사업 영향이 큰 편인데, "
                        "보도는 아직 많지 않습니다. 단건 보도라 놓치기 쉽습니다."
                    ),
                }
            ]
        },
    }
    summary = build_executive_summary_from_facts(
        comparison,
        change_stats={
            "window_days": 60,
            "today_issue_count": 20,
            "recent_card_count": 16,
        },
    )

    assert "고객 비교 기준" in summary
    assert "PoC 검증 지표" in summary
    assert "오늘 고객별 대응 기준" in summary
    assert "오늘 36건" not in summary
    assert "최근 60일" not in summary
    assert "재조정" not in summary
    assert "salience" not in summary.lower()
    assert "노출 0." not in summary
    assert "단건 보도라 놓치기 쉽습니다" not in summary


def test_is_generic_executive_text_detects_boilerplate() -> None:
    assert is_generic_executive_text("SK AX의 경쟁 환경에 직접적인 영향을 미칩니다.")
    assert not is_generic_executive_text(
        "LG CNS 클로드 엔터프라이즈 도입 — 제안서 검증 지표 재점검"
    )


def test_polish_executive_output_rewrites_generic_llm_fields() -> None:
    comparison = {
        "keyword_trends": [
            {"group_name": "AI 에이전트", "ratio_delta": 18.1, "latest_ratio": 118.5}
        ],
        "visibility_gaps": [],
        "primary_selection": {
            "items": [
                {
                    "title": "LG CNS 클로드 엔터프라이즈 도입",
                    "label": "high_salience_visible",
                    "narrative_hint": (
                        "LG CNS 계약 소식은 사업 영향과 보도 확산 모두 높아 "
                        "오늘의 핵심 판단 축입니다."
                    ),
                }
            ]
        },
        "coverage": {"hidden_gem_count": 0},
        "structural": [],
    }
    context = {
        "comparison_facts": comparison,
        "change_stats": {
            "default_change_summary": [
                {"label": "오늘 감지된 변화", "value": "12건"},
                {"label": "비교 기준", "value": "최근 60일"},
                {"label": "핵심 축", "value": "-"},
            ],
            "window_days": 60,
            "today_issue_count": 4,
            "recent_card_count": 8,
        },
    }
    polished = polish_executive_output(
        {
            "headline": "LG CNS 계약은 SK AX 경쟁 환경에 중요한 변화를 예고합니다.",
            "executive_summary": "경쟁 환경 변화에 대비한 전략 강화가 필요합니다.",
            "executive_implication": "경쟁 심화에 따른 대응이 필요합니다.",
            "change_summary": [
                {"label": "오늘 감지된 변화", "value": "12건"},
                {"label": "비교 기준", "value": "최근 60일"},
                {"label": "핵심 축", "value": "LG CNS"},
            ],
            "signals": [
                {"id": "s1", "label": "주요 신호", "value": "경쟁 환경에 영향"},
                {"id": "s2", "label": "관찰 포인트", "value": "보도량·sector 비중 맥락 확인"},
                {
                    "id": "s3",
                    "label": "다음 판단",
                    "value": "제안서·PoC·운영모델에서 무엇을 바꿀지 오늘 결정",
                },
            ],
            "provenance": {},
        },
        context=context,
    )

    assert "경쟁 환경" not in polished["headline"]
    assert "salience" not in polished["executive_summary"].lower()
    assert "고객 대응 패키지" in polished["executive_summary"]
    assert "클로드 엔터프라이즈" in polished["headline"]
    assert polished["signals"][0]["value"].startswith("핵심 판단 축")
    assert "SK AX는" in polished["executive_implication"]
    assert polished["change_summary"][1]["label"] == "검색지수 변화"
    assert "AI 에이전트" in polished["change_summary"][1]["value"]
    assert "검색지수" in polished["signals"][1]["value"]
    assert polished["provenance"]["postprocess"] == "comparison_facts_v1"
