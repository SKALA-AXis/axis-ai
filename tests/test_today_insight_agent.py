from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime
from typing import Any

import src.agents.today_insight_agent as today_module
import src.services.today_insight_comparison_engine as comparison_engine
from src.agents.today_insight_agent import TodayInsightAgent
from src.api.today_insight_schemas import TodayInsightGenerateRequest, TodayInsightGenerateResponse


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[Any] = []

    def invoke(self, prompt: Any, config: Any | None = None) -> _FakeResponse:
        del config
        self.prompts.append(prompt)
        return _FakeResponse(
            json.dumps(
                {
                    "headline": "엔터프라이즈 LLM 도입 신호가 운영 책임 기준을 앞당깁니다.",
                    "executive_summary": (
                        "오늘 통합 이슈는 엔터프라이즈 LLM과 운영형 AI가 같은 판단 축으로 묶이는 "
                        "변화를 보여줍니다."
                    ),
                    "executive_implication": (
                        "SK AX는 고객 대응 패키지에서 AI 기능보다 운영 책임, 감사 대응, "
                        "PoC 검증 지표를 먼저 보여줘야 합니다."
                    ),
                    "change_summary": [
                        {"label": "오늘 감지된 변화", "value": "3건"},
                        {"label": "비교 기준", "value": "최근 60일"},
                        {"label": "핵심 축", "value": "엔터프라이즈 LLM"},
                    ],
                    "signals": [
                        {
                            "id": "enterprise-llm",
                            "label": "ignored",
                            "value": "엔터프라이즈 LLM이 운영 책임 기준으로 이동",
                            "reasoning": [
                                {
                                    "stage": "관찰",
                                    "detail": "LG CNS 엔터프라이즈 LLM 이슈가 확인됐습니다.",
                                },
                                {
                                    "stage": "비교",
                                    "detail": "과거보다 보안·운영 책임 언급이 앞섭니다.",
                                },
                                {"stage": "판단", "detail": "고객 대응 평가 항목을 바꿔야 합니다."},
                            ],
                            "evidence": {
                                "grounds": ["LG CNS 엔터프라이즈 LLM 도입 이슈"],
                                "changes": ["보안 책임이 고객 대응 비교 항목으로 올라왔습니다."],
                                "related_keywords": ["엔터프라이즈 LLM", "운영 책임"],
                                "source_ids": ["11111111-1111-1111-1111-111111111111"],
                            },
                        },
                        {
                            "id": "watch",
                            "label": "ignored",
                            "value": "반복 패턴 여부를 관찰해야 함",
                            "reasoning": [
                                {
                                    "stage": "관찰",
                                    "detail": "과거 리포트와 유사 축을 비교했습니다.",
                                },
                            ],
                            "evidence": {
                                "grounds": ["전일 메모리"],
                                "changes": ["같은 sector 반복 여부가 관찰 포인트입니다."],
                                "related_keywords": ["audit"],
                                "source_ids": ["CN-1"],
                            },
                        },
                        {
                            "id": "next",
                            "label": "ignored",
                            "value": "PoC 검증 지표를 오늘 결정",
                            "reasoning": [
                                {"stage": "판단", "detail": "임원 의사결정 항목으로 좁혔습니다."},
                            ],
                            "evidence": {
                                "grounds": ["SK AX 공식 관점"],
                                "changes": ["운영 KPI가 핵심 산출물이 됐습니다."],
                                "related_keywords": ["PoC"],
                                "source_ids": ["CN-1"],
                            },
                        },
                    ],
                    "response_direction": [
                        {
                            "action": (
                                "엔터프라이즈 LLM 고객 대응 체크리스트에 운영 책임과 감사 대응 "
                                "범위를 분리합니다."
                            ),
                            "decision_owner": "사업전략",
                            "time_horizon": "이번 주",
                            "rationale": "고객 평가 기준이 운영 책임으로 이동했기 때문입니다.",
                            "evidence_refs": ["CN-1"],
                        }
                    ],
                    "sources": [
                        {
                            "id": "CN-1",
                            "title": "LG CNS 엔터프라이즈 LLM 도입",
                            "source_name": "card_news",
                            "publisher": "LG CNS",
                            "url": "https://example.com/lg",
                        }
                    ],
                    "confidence": 0.81,
                },
                ensure_ascii=False,
            )
        )


def _patch_context(monkeypatch, fake_llm: _FakeLLM, saved: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(comparison_engine, "_fetch_keyword_trend_rows", lambda **kwargs: [])
    monkeypatch.setattr(
        today_module,
        "_fetch_integrated_issues",
        lambda **kwargs: [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "created_date_kst": "2026-06-05",
                "main_company": "lg_cns",
                "event_type": "contract",
                "sectors": ["ax", "security"],
                "headline": "LG CNS 엔터프라이즈 LLM 도입",
                "one_line_summary": "고객의 운영 자동화와 감사 대응 요구가 함께 확인됐습니다.",
                "content_summary": "엔터프라이즈 LLM 도입과 보안 운영 책임이 연결됩니다.",
                "source_ids": [1],
                "sources": [
                    {
                        "id": "raw-1",
                        "title": "원문",
                        "source_name": "AXIS News",
                        "url": "https://example.com/raw",
                    }
                ],
                "confidence": 0.88,
            }
        ],
    )
    monkeypatch.setattr(
        today_module,
        "_fetch_cards_for_issues",
        lambda issue_ids, **kwargs: [
            {
                "id": "CN-1",
                "integrated_issue_id": issue_ids[0],
                "peer_id": "lg_cns",
                "title": "LG CNS 엔터프라이즈 LLM 도입",
                "summary_lines": ["엔터프라이즈 LLM", "보안 책임", "운영 KPI"],
                "sources": [{"id": "raw-1", "title": "원문", "source_name": "AXIS News"}],
            }
        ],
    )
    monkeypatch.setattr(today_module, "_fetch_recent_cards", lambda **kwargs: [])
    monkeypatch.setattr(
        today_module,
        "_fetch_prior_today_reports",
        lambda **kwargs: [{"report_date": "2026-06-04", "headline": "엔터프라이즈 LLM 관찰"}],
    )
    monkeypatch.setattr(
        today_module,
        "_fetch_analysis_ledger",
        lambda **kwargs: [{"analysis_type": "insight", "conclusion_one_liner": "운영 책임 부상"}],
    )
    monkeypatch.setattr(
        today_module,
        "_load_profile_context",
        lambda **kwargs: {"peer_profiles": {"lg_cns": {"business_areas": ["엔터프라이즈 LLM"]}}},
    )
    monkeypatch.setattr(
        today_module,
        "_load_skax_context",
        lambda **kwargs: {
            "ax": {"summary": "엔터프라이즈 LLM 운영·보안 대응 기회", "documents": []}
        },
    )
    monkeypatch.setattr(today_module, "_get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        today_module, "_save_report", lambda result, input_snapshot: saved.append(result)
    )


def test_today_insight_agent_generates_ui_ready_executive_payload(monkeypatch) -> None:
    fake_llm = _FakeLLM()
    saved: list[dict[str, Any]] = []
    card_lookup_args: list[dict[str, Any]] = []
    _patch_context(monkeypatch, fake_llm, saved)

    def fake_cards(issue_ids: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        card_lookup_args.append(kwargs)
        return [
            {
                "id": "CN-1",
                "integrated_issue_id": issue_ids[0],
                "peer_id": "lg_cns",
                "title": "LG CNS 엔터프라이즈 LLM 도입",
                "summary_lines": ["엔터프라이즈 LLM", "보안 책임", "운영 KPI"],
                "sources": [{"id": "raw-1", "title": "원문", "source_name": "AXIS News"}],
            }
        ]

    monkeypatch.setattr(today_module, "_fetch_cards_for_issues", fake_cards)

    result = asyncio.run(
        TodayInsightAgent().generate(
            TodayInsightGenerateRequest(
                anchor_date=date(2026, 6, 5),
                use_cached=False,
                save=True,
            )
        )
    )

    response = TodayInsightGenerateResponse.model_validate(result)
    assert [signal.label for signal in response.signals] == [
        "주요 신호",
        "관찰 포인트",
        "다음 판단",
    ]
    assert response.response_direction[0].action.startswith("엔터프라이즈 LLM 고객 대응")
    assert response.sources[0].id == "CN-1"
    assert response.source_integrated_issue_ids == ["11111111-1111-1111-1111-111111111111"]
    assert response.source_card_ids == ["CN-1"]
    assert response.insight_sections[0].evidence.grounds
    assert response.insight_sections[0].response_direction
    assert response.source_trace[0].source_card_id == "CN-1"
    assert response.memory_document is not None
    assert response.memory_document.update_window["window_days"] == 60
    assert response.memory_document.observed_facts
    assert response.memory_document.important_memory
    assert response.memory_document.next_analysis_hints
    assert response.provenance["prompt_version"] == "today-insight-v1.4-qualitative-signals"
    assert saved and saved[0]["headline"] == response.headline
    assert card_lookup_args[0]["window_days"] == 60
    assert "prior_today_insight_memory" in fake_llm.prompts[0]
    assert "comparison_facts" in fake_llm.prompts[0]
    assert "comparison_facts" in result
    assert result["comparison_facts"]["coverage"]["mode"] in {
        "dual_lane_full",
        "salience_only",
        "structural_only",
        "sparse",
    }
    assert "엔터프라이즈 LLM 운영·보안 대응 기회" in fake_llm.prompts[0]


def test_generate_returns_no_current_signals_when_only_stale_recent_cards(monkeypatch) -> None:
    """current 신호(현재 이슈/당일 카드)가 없고 과거 recent_cards 폴백만 있으면,
    옛 카드로 active 리포트를 만들지 않고 no_current_signals 플레이스홀더를 반환한다.

    회귀: 6/17 리포트가 current_issues=0 인데도 과거(6/2) 카드 헤드라인을 active 로
    노출한 버그. anchor_date 기준 current 신호가 없으면 '신규 신호 없음' 이어야 한다.
    """
    fake_llm = _FakeLLM()
    saved: list[dict[str, Any]] = []
    stale_card = {
        "id": "CN-20260602-44755",
        "peer_id": "lg_cns",
        "title": "LG CNS 인스웨이브와 코어뱅킹 현대화 웹단말 전환 사업 계약 체결",
        "summary_lines": ["코어뱅킹 현대화", "웹단말 전환", "계약 체결"],
        "sources": [{"id": "raw-1", "title": "원문", "source_name": "AXIS News"}],
    }

    monkeypatch.setattr(comparison_engine, "_fetch_keyword_trend_rows", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_integrated_issues", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_cards_for_issues", lambda issue_ids, **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_anchor_date_cards", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_recent_cards", lambda **kwargs: [stale_card])
    monkeypatch.setattr(today_module, "_fetch_prior_today_reports", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_analysis_ledger", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_load_profile_context", lambda **kwargs: {})
    monkeypatch.setattr(today_module, "_load_skax_context", lambda **kwargs: {})
    monkeypatch.setattr(today_module, "_get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        today_module, "_save_report", lambda result, input_snapshot: saved.append(result)
    )

    result = asyncio.run(
        TodayInsightAgent().generate(
            TodayInsightGenerateRequest(
                anchor_date=date(2026, 6, 17),
                use_cached=False,
                save=True,
            )
        )
    )

    # 과거 카드를 오늘 헤드라인으로 만들지 않는다 — LLM 자체가 호출되지 않아야 한다.
    assert fake_llm.prompts == []
    # '신규 신호 없음' 플레이스홀더 — 프론트가 is_status_placeholder 로 빈 상태를 노출한다.
    assert result["provenance"]["result_kind"] == "no_current_signals"
    assert result["provenance"]["is_status_placeholder"] is True
    assert saved and saved[0]["provenance"]["is_status_placeholder"] is True


def test_build_context_keeps_comparison_facts_within_anchor_current_inputs(monkeypatch) -> None:
    anchor = date(2026, 6, 12)

    current_lg_issues = [
        {
            "id": f"11111111-1111-1111-1111-11111111111{idx}",
            "created_date_kst": anchor.isoformat(),
            "latest_source_date_kst": anchor.isoformat(),
            "has_anchor_source": True,
            "main_company": "lg_cns",
            "event_type": "contract",
            "sectors": ["ax"],
            "headline": f"LG CNS 현재 이슈 {idx}",
            "one_line_summary": "LG CNS의 현재 기준 계약 신호입니다.",
            "content_summary": "anchor-date 소스가 확인된 현재 이슈입니다.",
            "source_ids": [idx],
            "sources": [
                {
                    "id": f"raw-{idx}",
                    "title": f"LG CNS 현재 이슈 {idx}",
                    "source_name": "AXIS News",
                }
            ],
            "confidence": 0.9,
        }
        for idx in range(8)
    ]
    stale_backfill_issue = {
        "id": "22222222-2222-2222-2222-222222222222",
        "created_date_kst": anchor.isoformat(),
        "latest_source_date_kst": "2025-12-23",
        "has_anchor_source": False,
        "main_company": "samsung_sds",
        "event_type": "contract",
        "sectors": ["ax"],
        "headline": "삼성SDS, 오픈AI와 리셀러 파트너 계약 체결",
        "one_line_summary": "과거 소스 기반 이슈가 오늘 재생성됐습니다.",
        "content_summary": "소스 날짜가 anchor-date가 아닙니다.",
        "source_ids": [99],
        "sources": [],
        "confidence": 0.95,
    }
    overflow_issue = {
        **stale_backfill_issue,
        "id": "33333333-3333-3333-3333-333333333333",
        "latest_source_date_kst": anchor.isoformat(),
        "has_anchor_source": True,
        "one_line_summary": "anchor-date 소스가 있지만 max_issues 밖 후보입니다.",
    }

    monkeypatch.setattr(comparison_engine, "_fetch_keyword_trend_rows", lambda **kwargs: [])
    monkeypatch.setattr(
        today_module,
        "_fetch_integrated_issues",
        lambda **kwargs: [*current_lg_issues, stale_backfill_issue, overflow_issue],
    )
    monkeypatch.setattr(today_module, "_fetch_cards_for_issues", lambda issue_ids, **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_anchor_date_cards", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_recent_cards", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_prior_today_reports", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_analysis_ledger", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_load_profile_context", lambda **kwargs: {})
    monkeypatch.setattr(today_module, "_load_skax_context", lambda **kwargs: {})

    context = today_module._build_context(
        TodayInsightGenerateRequest(
            anchor_date=anchor,
            max_issues=8,
            max_cards=12,
        ),
        anchor,
    )

    comparison = context["comparison_facts"]
    salience_titles = json.dumps(
        [
            item.get("title")
            for item in comparison.get("salience_candidates", [])
            if isinstance(item, dict)
        ],
        ensure_ascii=False,
    )
    primary_titles = json.dumps(
        [
            item.get("title")
            for item in comparison.get("primary_selection", {}).get("items", [])
            if isinstance(item, dict)
        ],
        ensure_ascii=False,
    )
    assert len(context["current_issues"]) == 8
    assert "삼성SDS" not in salience_titles
    assert "삼성SDS" not in primary_titles
    assert all(issue["main_company"] == "lg_cns" for issue in context["current_issues"])


def test_build_context_excludes_global_peer_issues_and_cards(monkeypatch) -> None:
    """글로벌 IT peer(google 등)는 헤드라인/신호에서 제외되고 국내 4사만 남는다."""
    anchor = date(2026, 6, 16)

    domestic_issue = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "created_date_kst": anchor.isoformat(),
        "latest_source_date_kst": anchor.isoformat(),
        "has_anchor_source": True,
        "main_company": "lg_cns",
        "event_type": "contract",
        "sectors": ["ax"],
        "headline": "LG CNS 국내 계약 신호",
        "one_line_summary": "LG CNS의 현재 기준 계약 신호입니다.",
        "content_summary": "anchor-date 소스가 확인된 국내 이슈입니다.",
        "source_ids": [1],
        "sources": [{"id": "raw-1", "title": "LG CNS 국내 계약 신호", "source_name": "AXIS News"}],
        "confidence": 0.9,
    }
    global_issue = {
        "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "created_date_kst": anchor.isoformat(),
        "latest_source_date_kst": anchor.isoformat(),
        "has_anchor_source": True,
        "main_company": "google",
        "event_type": "product",
        "sectors": ["security"],
        "headline": "Google AI Threat Defense 운영 교훈 공개",
        "one_line_summary": "글로벌 IT peer 의 보안 업데이트입니다.",
        "content_summary": "해외 peer 단건 보도입니다.",
        "source_ids": [2],
        "sources": [
            {"id": "raw-2", "title": "Google AI Threat Defense", "source_name": "Google Blog"}
        ],
        "confidence": 0.95,
    }
    global_card = {
        "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
        "peer_id": "google",
        "title": "Google AI Threat Defense 카드",
        "summary_lines": ["글로벌 보안 카드"],
        "sources": [],
    }

    monkeypatch.setattr(comparison_engine, "_fetch_keyword_trend_rows", lambda **kwargs: [])
    monkeypatch.setattr(
        today_module,
        "_fetch_integrated_issues",
        lambda **kwargs: [domestic_issue, global_issue],
    )
    monkeypatch.setattr(
        today_module, "_fetch_cards_for_issues", lambda issue_ids, **kwargs: [global_card]
    )
    monkeypatch.setattr(today_module, "_fetch_anchor_date_cards", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_recent_cards", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_prior_today_reports", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_analysis_ledger", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_load_profile_context", lambda **kwargs: {})
    monkeypatch.setattr(today_module, "_load_skax_context", lambda **kwargs: {})

    context = today_module._build_context(
        TodayInsightGenerateRequest(anchor_date=anchor, max_issues=8, max_cards=12),
        anchor,
    )

    companies = {issue["main_company"] for issue in context["current_issues"]}
    assert companies == {"lg_cns"}
    facts_dump = json.dumps(context["comparison_facts"], ensure_ascii=False)
    assert "Google AI Threat Defense" not in facts_dump
    assert "google" not in facts_dump


def test_today_insight_agent_returns_cached_payload_without_regeneration(monkeypatch) -> None:
    cached_payload = {
        "report_date": "2026-06-05",
        "generated_at": "2026-06-05T00:00:00+00:00",
        "headline": "캐시된 Today's Insight",
        "signals": [],
    }
    preload_calls: list[bool] = []

    monkeypatch.setattr(
        today_module,
        "_load_latest_report_record",
        lambda anchor_date: {
            "payload": cached_payload,
            "created_at": datetime(2026, 6, 5, 8, 10, tzinfo=UTC),
        },
    )
    monkeypatch.setattr(today_module, "_get_llm", lambda: preload_calls.append(True))

    result = asyncio.run(
        TodayInsightAgent().generate(
            TodayInsightGenerateRequest(
                anchor_date=date(2026, 6, 5),
                preload_model=True,
                save=False,
            )
        )
    )

    assert result["headline"] == "캐시된 Today's Insight"
    assert preload_calls == [True]


def test_today_insight_agent_cache_only_does_not_generate_when_cache_missing(monkeypatch) -> None:
    fake_llm = _FakeLLM()
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(today_module, "_load_latest_report_record", lambda anchor_date: None)
    monkeypatch.setattr(today_module, "_fetch_prior_today_reports", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        today_module, "_save_report", lambda result, input_snapshot: saved.append(result)
    )

    result = asyncio.run(
        TodayInsightAgent().generate(
            TodayInsightGenerateRequest(
                anchor_date=date(2026, 6, 5),
                cache_only=True,
                save=True,
            )
        )
    )

    assert result["headline"] == "Today's Insight 08:10 업데이트 대기"
    assert result["provenance"]["mode"] == "cache_only"
    assert not fake_llm.prompts
    assert not saved


def test_today_insight_agent_cache_only_returns_latest_saved_before_anchor(monkeypatch) -> None:
    cached_payload = {
        "report_date": "2026-06-09",
        "generated_at": "2026-06-09T23:10:00+09:00",
        "headline": "전날 저장된 Today's Insight",
        "signals": [{"id": "s1", "label": "주요 신호", "value": "저장 리포트"}],
        "provenance": {
            "cache_lookup": "latest_saved_on_or_before_anchor",
            "served_anchor_date": "2026-06-10",
            "cached_report_date": "2026-06-09",
            "latest_fallback": True,
        },
    }
    fake_llm = _FakeLLM()
    monkeypatch.setattr(
        today_module,
        "_load_latest_report_record",
        lambda anchor_date: {
            "payload": cached_payload,
            "created_at": datetime(2026, 6, 9, 8, 10, tzinfo=UTC),
            "report_date": "2026-06-09",
        },
    )
    monkeypatch.setattr(today_module, "_get_llm", lambda: fake_llm)

    result = asyncio.run(
        TodayInsightAgent().generate(
            TodayInsightGenerateRequest(
                anchor_date=date(2026, 6, 10),
                cache_only=True,
                save=True,
            )
        )
    )

    assert result["headline"] == "전날 저장된 Today's Insight"
    assert result["report_date"] == "2026-06-09"
    assert result["provenance"]["latest_fallback"] is True
    assert not fake_llm.prompts


def test_today_insight_agent_returns_fallback_when_no_source_data(monkeypatch) -> None:
    monkeypatch.setattr(comparison_engine, "_fetch_keyword_trend_rows", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_integrated_issues", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_cards_for_issues", lambda issue_ids, **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_recent_cards", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_prior_today_reports", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_analysis_ledger", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_load_profile_context", lambda **kwargs: {})
    monkeypatch.setattr(today_module, "_load_skax_context", lambda **kwargs: {})

    result = asyncio.run(
        TodayInsightAgent().generate(
            TodayInsightGenerateRequest(
                anchor_date=date(2026, 6, 5),
                use_cached=False,
                save=False,
            )
        )
    )

    response = TodayInsightGenerateResponse.model_validate(result)
    assert response.warning == "today insight source data unavailable"
    assert response.provenance["result_kind"] == "no_current_signals"
    assert response.provenance["is_status_placeholder"] is True
    assert response.signals[0].label == "주요 신호"
    assert response.signals[2].label == "다음 판단"


def test_today_insight_agent_uses_anchor_date_cards_when_integrated_issues_empty(
    monkeypatch,
) -> None:
    """통합 이슈가 없어도 anchor_date(오늘) 카드가 있으면 그 카드로 리포트를 생성한다.

    과거 recent_cards 폴백은 더 이상 헤드라인 소스로 쓰지 않는다(그 경우 no_current_signals).
    당일 카드는 current 신호이므로 정상 리포트가 나와야 한다.
    """
    fake_llm = _FakeLLM()
    saved: list[dict[str, Any]] = []
    anchor_lookup_args: list[dict[str, Any]] = []

    monkeypatch.setattr(comparison_engine, "_fetch_keyword_trend_rows", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_integrated_issues", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_cards_for_issues", lambda issue_ids, **kwargs: [])

    def fake_anchor_cards(**kwargs: Any) -> list[dict[str, Any]]:
        anchor_lookup_args.append(kwargs)
        return [
            {
                "id": "CN-RECENT",
                "integrated_issue_id": None,
                "peer_id": "samsung_sds",
                "title": "삼성SDS 생성형 AI 운영 자동화 확대",
                "summary_lines": ["운영 자동화", "생성형 AI", "고객 업무 적용"],
                "event_type": "product",
                "importance": "high",
                "importance_score": 0.91,
                "sources": [
                    {
                        "id": "raw-recent-1",
                        "title": "삼성SDS 생성형 AI 운영 자동화 확대",
                        "source_name": "AXIS News",
                    }
                ],
            }
        ]

    monkeypatch.setattr(today_module, "_fetch_anchor_date_cards", fake_anchor_cards)
    monkeypatch.setattr(today_module, "_fetch_recent_cards", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_prior_today_reports", lambda **kwargs: [])
    monkeypatch.setattr(today_module, "_fetch_analysis_ledger", lambda **kwargs: [])
    monkeypatch.setattr(
        today_module,
        "_load_profile_context",
        lambda **kwargs: {"peer_profiles": {"samsung_sds": {"business_areas": ["AI"]}}},
    )
    monkeypatch.setattr(
        today_module,
        "_load_skax_context",
        lambda **kwargs: {"skax_contexts": [{"title": "SK AX AI Transformation"}]},
    )
    monkeypatch.setattr(today_module, "_get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        today_module, "_save_report", lambda result, input_snapshot: saved.append(result)
    )

    result = asyncio.run(
        TodayInsightAgent().generate(
            TodayInsightGenerateRequest(
                anchor_date=date(2026, 6, 5),
                use_cached=False,
                save=False,
            )
        )
    )

    response = TodayInsightGenerateResponse.model_validate(result)
    # 당일 카드는 current 신호 → 정상 리포트(플레이스홀더 아님)
    assert result["provenance"].get("result_kind") != "no_current_signals"
    assert response.source_integrated_issue_ids == []
    assert response.source_card_ids == ["CN-RECENT"]
    assert response.peer_ids == ["samsung_sds"]
    assert response.change_summary[1].value == "최근 60일"
    assert anchor_lookup_args  # anchor-date 카드 조회가 사용됨
    assert "삼성SDS 생성형 AI 운영 자동화 확대" in fake_llm.prompts[0]


def test_fit_llm_prompt_context_avoids_blind_truncation_marker() -> None:
    context = {
        "report_date": "2026-06-09",
        "window_days": 60,
        "comparison_facts": {
            "primary_selection": {"items": [{"title": "핵심", "label": "high_salience_visible"}]},
            "salience_candidates": [{"title": "후보", "label": "moderate_salience"}],
            "keyword_trends": [],
            "structural": [],
            "coverage": {"mode": "dual_lane_full"},
        },
        "change_stats": {"default_change_summary": [], "window_days": 60},
        "current_issues": [
            today_module._issue_for_prompt(
                {
                    "id": "IC-1",
                    "main_company": "lg_cns",
                    "headline": "LG CNS 계약",
                    "one_line_summary": "요약",
                    "content_summary": "본문" * 200,
                    "sectors": ["ax"],
                    "source_ids": [],
                    "sources": [],
                }
            )
        ],
        "recent_cards": [],
        "history_issues": [],
        "sources": [],
        "prior_today_insight_memory": [{"headline": "prior"} for _ in range(5)],
        "analysis_ledger_context": [{"headline": "ledger"} for _ in range(5)],
        "profile_context": {
            "peer_profiles": {
                "lg_cns": {
                    "company_name_ko": "LG CNS",
                    "strategic_direction": "방향" * 300,
                    "recent_signals": [{"headline": "sig"}],
                }
            },
            "sector_context": {"ax": {"summary": "섹터"}},
        },
        "skax_context": {"ax": {"summary": "skax", "documents": [{"title": "doc"}]}},
    }

    llm_context, meta = today_module._fit_llm_prompt_context(context)
    serialized = today_module._json_dumps(llm_context)

    assert "...TRUNCATED..." not in serialized
    assert meta["serialized_chars"] <= today_module._LLM_CONTEXT_MAX_CHARS
    assert llm_context["comparison_facts"]["primary_selection"]["items"][0]["title"] == "핵심"
    assert "generation_focus" in llm_context
    assert "signals" in llm_context["generation_focus"]["llm_priority_fields"]


def test_today_insight_public_copy_qualifies_internal_scores() -> None:
    text = "영향도 점수 0.7, 노출 점수 0.244로 low_visibility_definite_event로 분류됐습니다."

    sanitized = today_module._sanitize_public_text(text)

    assert "0.7" not in sanitized
    assert "0.244" not in sanitized
    assert "점수" not in sanitized
    assert "low_visibility_definite_event" not in sanitized
    assert "내용 영향은 큰 편" in sanitized
    assert "보도 확산은 아직 낮은 편" in sanitized
    assert "노출은 낮지만 내용이 확인된 이벤트" in sanitized


def test_match_sources_no_arbitrary_fallback_when_unmatched() -> None:
    sources = [
        {"id": "CN-1", "url": "https://a.example/1"},
        {"id": "CN-2", "url": "https://b.example/2"},
    ]
    # 인용한 id 가 어느 출처와도 안 맞으면 임의 출처를 붙이지 않는다(빈 리스트) — 오링크 방지.
    assert today_module._match_sources_by_ids(sources, {"CN-999"}) == []


def test_match_sources_exact_and_raw_prefix_normalization() -> None:
    sources = [
        {"id": "CN-1", "url": "https://a.example/1"},
        {"id": "raw-7", "url": "https://b.example/7"},
    ]
    assert today_module._match_sources_by_ids(sources, {"CN-1"})[0]["id"] == "CN-1"
    # 숫자 raw id 인용이 raw- 표기 출처와 양방향 매칭된다.
    assert today_module._match_sources_by_ids(sources, {"7"})[0]["id"] == "raw-7"
    assert today_module._match_sources_by_ids(sources, {"raw-7"})[0]["id"] == "raw-7"


def test_match_sources_no_citation_returns_doc_sources() -> None:
    sources = [{"id": "CN-1", "url": "https://a.example/1"}]
    # source_ids 가 없으면(섹션이 특정 인용 안 함) 문서 출처를 보여주는 기존 동작 유지.
    assert today_module._match_sources_by_ids(sources, set()) == sources


def test_match_trace_no_arbitrary_fallback_when_unmatched() -> None:
    trace = [
        {"source_card_id": "CN-1", "url": "https://a.example/1"},
        {"source_card_id": "CN-2", "url": "https://b.example/2"},
    ]
    assert today_module._match_trace_by_ids(trace, {"CN-999"}) == []


def test_match_trace_matches_card_and_raw_article_ids() -> None:
    trace = [
        {"source_card_id": "CN-1", "source_raw_article_ids": ["123"], "url": "https://x"},
        {"source_card_id": "CN-2", "source_raw_article_ids": ["456"], "url": "https://y"},
    ]
    assert today_module._match_trace_by_ids(trace, {"CN-1"}) == [trace[0]]
    # LLM 이 raw-123 으로 인용해도 trace 의 숫자 "123" 과 매칭된다.
    assert today_module._match_trace_by_ids(trace, {"raw-123"}) == [trace[0]]


def test_apply_insight_state_today_signal_above_bar(monkeypatch) -> None:
    """오늘 신호 + lead salience ≥ 0.70 → today_signal, signal_date=오늘, 페이로드 보존."""
    monkeypatch.setattr(
        today_module, "_build_coverage_stats", lambda **kw: {"reviewed_last_7d": 12}
    )
    base: dict[str, Any] = {
        "report_date": "2026-06-17",
        "comparison_facts": {"primary_selection": {"items": [{"salience_score": 0.82}]}},
        "insight_sections": [{"id": "s1"}],
        "signals": [{"id": "sig1", "value": "x"}],
        "provenance": {},
    }
    today_module._apply_insight_state(base, {"has_current_signal": True})
    assert base["state"] == "today_signal"
    assert base["signal_date"] == "2026-06-17"
    assert base["week_synthesis"] is None
    assert base["insight_sections"] == [{"id": "s1"}]  # 보존
    assert base["coverage_stats"]["reviewed_last_7d"] == 12


def test_apply_insight_state_today_signal_even_below_prior_salience_bar(monkeypatch) -> None:
    """오늘 신호가 있으면 내부 salience 점수로 숨기지 않고 today_signal 로 노출."""
    monkeypatch.setattr(today_module, "_build_coverage_stats", lambda **kw: {"reviewed_last_7d": 9})
    base: dict[str, Any] = {
        "report_date": "2026-06-17",
        "comparison_facts": {
            "primary_selection": {"items": [{"salience_score": 0.40}]},
            "salience_candidates": [{"peer_label": "포스코DX", "event_type": "contract"}],
        },
        "signals": [{"id": "sig1", "value": "x"}],
        "provenance": {},
    }
    today_module._apply_insight_state(base, {"has_current_signal": True})
    assert base["state"] == "today_signal"
    assert base["signal_date"] == "2026-06-17"
    assert base["week_synthesis"] is None
    assert "depth_gate" not in base["provenance"]


def test_apply_insight_state_quiet_when_no_current_signal(monkeypatch) -> None:
    """오늘 신호 없음(placeholder) → quiet + week_synthesis (빈 화면 대신 종합)."""
    monkeypatch.setattr(today_module, "_build_coverage_stats", lambda **kw: {"reviewed_last_7d": 0})
    base: dict[str, Any] = {
        "report_date": "2026-06-17",
        "comparison_facts": {},
        "provenance": {"is_status_placeholder": True, "result_kind": "no_current_signals"},
    }
    today_module._apply_insight_state(base, {"has_current_signal": False})
    assert base["state"] == "quiet"
    assert base["week_synthesis"]
