"""StrategicInsightAgent unit tests with mock LLM response."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "strategic_insight_agent_input.mock.json"


def _fixture() -> dict[str, Any]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if "input_bundle" in data:
        return data
    return _fixture_from_integration_output(data)


def _fixture_from_integration_output(data: dict[str, Any]) -> dict[str, Any]:
    """Accept raw IntegrationAgent dry-run output as the test fixture."""
    integrated_issue = data.get("integrated_issue") or {}
    classification = data.get("classification") or {}
    article_overview = data.get("article_overview") or []
    source_ids = (
        integrated_issue.get("source_article_ids")
        or integrated_issue.get("cluster_article_ids")
        or data.get("article_ids")
        or []
    )
    items = [
        {
            "id": article.get("id"),
            "raw_article_id": article.get("id"),
            "title": article.get("title"),
            "publisher": article.get("publisher"),
            "source_name": article.get("source_name"),
            "published_at": article.get("published_at"),
            "is_representative": bool(article.get("is_representative")),
            "matched_companies": article.get("matched_companies") or [],
            "matched_sectors": article.get("matched_sectors") or [],
            "relevance_score": article.get("relevance_score"),
        }
        for article in article_overview
        if article.get("id") is not None
    ]
    if not items:
        items = [
            {
                "id": article_id,
                "raw_article_id": article_id,
                "is_representative": article_id == integrated_issue.get("representative_id"),
            }
            for article_id in source_ids
        ]
    evidence_snippets: list[dict[str, Any]] = []
    for basis in integrated_issue.get("fact_basis") or []:
        evidence_texts = basis.get("evidence_texts") or []
        if not evidence_texts and basis.get("evidence_text"):
            evidence_texts = [basis.get("evidence_text")]
        for evidence_text in evidence_texts:
            evidence_snippets.append(
                {
                    "article_id": (basis.get("source_article_ids") or [None])[0],
                    "text": evidence_text,
                    "fact_ids": basis.get("fact_ids") or [],
                    "evidence_type": basis.get("evidence_type"),
                }
            )
    sectors = classification.get("sectors") or [classification.get("sector")]
    companies = (
        data.get("target_companies")
        or classification.get("companies")
        or [integrated_issue.get("main_company")]
    )
    input_bundle = {
        "bundle_id": integrated_issue.get("bundle_id")
        or f"news:{integrated_issue.get('representative_id') or data.get('representative_id')}",
        "cluster_id": str(integrated_issue.get("cluster_id") or data.get("cluster_id") or ""),
        "source_type": integrated_issue.get("issue_source_type") or "news",
        "companies": [company for company in companies if company],
        "sectors": [sector for sector in sectors if sector],
        "event_type": (
            classification.get("event_type") or integrated_issue.get("cluster_event_type")
        ),
        "items": items,
        "facts": integrated_issue.get("consolidated_facts")
        or integrated_issue.get("extracted_facts")
        or [],
        "evidence_snippets": evidence_snippets,
        "sources": integrated_issue.get("representative_sources") or [],
        "metadata": {
            "representative_id": integrated_issue.get("representative_id")
            or data.get("representative_id"),
            "cluster_article_ids": integrated_issue.get("cluster_article_ids") or source_ids,
            "classification": classification,
            "article_count": data.get("article_count"),
            "dry_run_type": data.get("dry_run_type"),
            "dry_run_created_at": data.get("dry_run_created_at"),
        },
    }
    return {
        "input_bundle": input_bundle,
        "classification": classification,
        "integrated_issue": integrated_issue,
        "profile_context": _default_profile_context(integrated_issue, classification),
        "analysis_context": _default_analysis_context(data, integrated_issue, classification),
    }


def _default_profile_context(
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    company_id = integrated_issue.get("main_company") or classification.get("company") or ""
    company_name = _display_name_from_company_id(company_id)
    sectors = [str(item) for item in classification.get("sectors") or [] if item]
    if not sectors and classification.get("sector"):
        sectors = [str(classification["sector"])]
    capabilities = _capability_terms_from_issue(integrated_issue, classification)
    return {
        "skax_profile": {
            "company_id": "sk_ax",
            "company_name_ko": "SK AX",
            "business_lines": sectors,
            "strategic_focus": sectors,
            "business_areas": [{"name": sector, "core_capabilities": []} for sector in sectors],
        },
        "peer_profiles": {
            company_id: {
                "peer_id": company_id,
                "company_id": company_id,
                "company_name_ko": company_name,
                "one_liner": f"{company_name} 관련 통합 이슈 기반 테스트 프로필",
                "core_capabilities": capabilities,
            }
        }
        if company_id
        else {},
        "sector_context": {"selected_sector_ids": classification.get("sectors") or []},
    }


def _display_name_from_company_id(company_id: str) -> str:
    text = str(company_id or "").strip()
    if not text:
        return ""
    return text.replace("_", " ").upper()


def _capability_terms_from_issue(
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> list[str]:
    terms: list[str] = []
    for signal in integrated_issue.get("business_signals") or []:
        for key in ("business_area", "signal_type"):
            value = str(signal.get(key) or "").strip()
            if value:
                terms.append(value)
    terms.extend(str(item) for item in classification.get("sectors") or [] if item)
    if classification.get("event_type"):
        terms.append(str(classification["event_type"]))
    return list(dict.fromkeys(terms))


def _default_analysis_context(
    data: dict[str, Any],
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    source_ids = integrated_issue.get("source_article_ids") or data.get("article_ids") or []
    company_id = integrated_issue.get("main_company") or classification.get("company")
    return {
        "peer_event_timeline_recent": [
            {
                "peer_id": company_id,
                "event_type": classification.get("event_type"),
                "title": integrated_issue.get("headline") or integrated_issue.get("main_issue"),
                "source_article_ids": source_ids[:5],
            }
        ],
        "sector_pulse_recent": [
            {
                "sector": ",".join(classification.get("sectors") or []),
                "summary": integrated_issue.get("one_line_summary")
                or integrated_issue.get("main_issue")
                or "",
            }
        ],
        "financial_trend": {},
        "event_chain_candidates": [],
        "similar_cards_rag": [],
        "evidence_density_per_peer": {
            company_id: {
                "source_count": len(source_ids),
                "fact_basis_count": len(integrated_issue.get("fact_basis") or []),
            }
        }
        if company_id
        else {},
        "provenance": {"builder": "fixture_from_integration_output"},
    }


def _load_env_file_for_integration() -> None:
    """Load local .env without shell expansion.

    DATABASE_URL may contain "$" in the password. `source .env` expands it and
    breaks authentication, so integration tests read the file directly.
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    wanted = {"DATABASE_URL", "OPENAI_API_KEY"}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in wanted:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
    try:
        from src.db import postgres

        postgres.reconfigure_from_env()
    except Exception:
        pass


_load_env_file_for_integration()

import src.agents.strategic_insight_agent as strategic_insight_module  # noqa: E402
from src.agents.strategic_insight_agent import StrategicInsightAgent  # noqa: E402


def _fake_llm_response(payload: dict[str, Any]) -> Any:
    response = MagicMock()
    response.content = json.dumps(payload, ensure_ascii=False)
    return response


def _companies_from_integrated_issue(integrated_issue: dict[str, Any]) -> list[str]:
    companies: list[str] = []
    for company_id in [
        integrated_issue.get("main_company"),
        *(integrated_issue.get("mentioned_peer_companies") or []),
    ]:
        if company_id and company_id not in companies:
            companies.append(company_id)
    return companies


def _assert_strategic_insight_schema(result: dict[str, Any]) -> None:
    assert set(result) == {
        "is_valid_strategic_insight",
        "analysis",
        "implication",
        "sentence_grounding",
    }
    assert set(result["analysis"]) == {
        "is_valid_analysis",
        "analysis_scope",
        "analysis_summary",
        "strategic_meaning",
        "market_signal",
        "impact_level",
        "impact_reason",
        "risk_or_opportunity",
        "confidence",
        "reason",
    }
    assert set(result["implication"]) == {
        "is_valid_implication",
        "implication_scope",
        "peer_implication",
        "skax_implication",
        "follow_up_questions",
        "watch_points",
        "confidence",
        "evidence_label",
        "provenance",
    }
    assert set(result["implication"]["peer_implication"]) == {
        "company_id",
        "company_name_ko",
        "peer_meaning",
        "capability_change",
        "sourced_evidence_ids",
    }
    assert set(result["implication"]["skax_implication"]) == {
        "why_important",
        "potential_impact",
        "opportunities",
        "threats",
        "recommended_actions",
        "business_line_mapping",
    }
    assert set(result["sentence_grounding"]) == {
        "schema_version",
        "generator",
        "entries",
        "summary",
    }


TEST_LINE_A = "profile_line_a"
TEST_LINE_B = "profile_line_b"
TEST_LINE_INVALID = "not_in_profile"


def _profile_context_with_business_lines(
    payload: dict[str, Any],
    business_lines: list[str] | None = None,
    *,
    business_areas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a test profile context without domain-specific business-line names."""
    skax_profile = dict(payload["profile_context"].get("skax_profile") or {})
    if business_lines is not None:
        skax_profile["business_lines"] = business_lines
    if business_areas is not None:
        skax_profile["business_areas"] = business_areas
    return {
        **payload["profile_context"],
        "skax_profile": skax_profile,
    }


def _fixture_company_identity(payload: dict[str, Any]) -> tuple[str, str]:
    company_id = payload["integrated_issue"].get("main_company") or "peer_company"
    peer_profile = (payload["profile_context"].get("peer_profiles") or {}).get(company_id) or {}
    company_name = peer_profile.get("company_name_ko") or _display_name_from_company_id(company_id)
    return company_id, company_name


def _minimal_integrated_issue_with_fact(
    *,
    company_id: str,
    fact_id: str,
    fact: str,
) -> dict[str, Any]:
    return {
        "is_valid_summary": True,
        "bundle_id": "test:relationship-grounding",
        "cluster_id": 999,
        "representative_id": 1001,
        "source_article_ids": [1001],
        "cluster_article_ids": [1001],
        "analyzed_article_ids": [1001],
        "main_company": company_id,
        "mentioned_peer_companies": [company_id],
        "cluster_event_type": "general_update",
        "headline": fact,
        "main_event": fact,
        "main_issue": fact,
        "one_line_summary": fact,
        "integrated_text": fact,
        "fact_summary": [fact],
        "consolidated_facts": [
            {
                "fact_id": fact_id,
                "fact": fact,
                "source_article_ids": [1001],
                "evidence_texts": [fact],
                "source_type": "news",
                "fact_type": "reported_fact",
            }
        ],
        "fact_basis": [
            {
                "summary_line_index": 1,
                "source_article_ids": [1001],
                "fact_ids": [fact_id],
                "evidence_text": fact,
                "evidence_texts": [fact],
                "evidence_type": "reported_fact",
            }
        ],
        "key_numbers": [],
        "business_signals": [],
        "representative_sources": [],
        "missing_or_uncertain_points": [],
        "confidence": 0.8,
    }


def test_strategic_insight_agent_returns_separated_blocks():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 확인된 사업 신호가 고객 적용 범위를 넓히고 있습니다.",
            "strategic_meaning": [
                "통합 이슈의 근거 사실은 피어사의 고객 제안 범위가 넓어졌음을 보여줍니다.",
                "프로필의 기존 역량과 이번 근거가 연결되며 적용 장면이 더 구체화됩니다.",
            ],
            "market_signal": (
                "토큰증권 기능분석 컨설팅과 테스트베드 구축처럼 고객은 적용 범위와 "
                "검증 기준을 함께 봅니다."
            ),
            "impact_level": "high",
            "impact_reason": "통합 이슈의 fact_basis와 프로필 역량이 같은 방향을 가리킵니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.82,
            "reason": "c43682_a43100_f1 근거와 business_signals를 기반으로 해석했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": (
                    "피어사는 확인된 사업명과 검증 단계를 기존 역량과 연결해 "
                    "고객 적용 범위를 설명하고 있습니다."
                ),
                "capability_change": (
                    "프로필의 기존 역량이 이번 통합 이슈의 적용 장면으로 연결됩니다."
                ),
                "sourced_evidence_ids": ["c43682_a43100_f1", "not-in-input"],
            },
            "skax_implication": {
                "why_important": (
                    "SK AX도 고객 제안에서 적용 범위와 검증 기준을 함께 보여줘야 합니다."
                ),
                "potential_impact": (
                    "통합 이슈에서 적용 범위와 검증 기준이 함께 확인되므로 고객은 "
                    "제안 단계에서 실행 범위와 확인 기준을 비교할 수 있습니다. 따라서 "
                    "SK AX는 제안서에서 적용 범위와 PoC 확인 기준을 분리해 설명해야 합니다."
                ),
                "opportunities": ["적용 범위와 검증 기준을 묶은 제안 구성을 만들 수 있습니다."],
                "threats": ["검증 근거가 약하면 고객 비교 단계에서 설득력이 낮아질 수 있습니다."],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 실행 범위와 "
                        "검증 기준을 따로 비교할 수 있습니다. 따라서 SK AX 제안서에서 "
                        "적용 범위와 검증 기준을 별도 항목으로 설명합니다."
                    )
                ],
                "business_line_mapping": [TEST_LINE_A, TEST_LINE_B, TEST_LINE_INVALID],
            },
            "follow_up_questions": ["협력 범위가 실제 고객 사례로 이어졌는가?"],
            "watch_points": ["기업용 AI 전환 수주 사례"],
            "confidence": 0.78,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))

    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A, TEST_LINE_B])
    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is True
    _assert_strategic_insight_schema(result)
    assert result["analysis"]["is_valid_analysis"] is True
    assert result["analysis"]["analysis_scope"] == "peer_and_industry"
    assert result["implication"]["is_valid_implication"] is True
    assert result["implication"]["implication_scope"] == "peer_and_skax"
    assert result["implication"]["peer_implication"]["company_id"] == company_id
    # Only fact_ids present in IntegratedIssue are kept, with referenced facts augmented.
    assert "c43682_a43100_f1" in result["implication"]["peer_implication"]["sourced_evidence_ids"]
    assert "not-in-input" not in result["implication"]["peer_implication"]["sourced_evidence_ids"]
    # Only business lines from the supplied SK AX profile candidates are kept.
    assert result["implication"]["skax_implication"]["business_line_mapping"] == [
        TEST_LINE_A,
        TEST_LINE_B,
    ]
    assert result["implication"]["provenance"]["generator"] == "StrategicInsightAgent"
    assert (
        result["implication"]["provenance"]["prompt_version"]
        == strategic_insight_module._PROMPT_VERSION
    )
    grounding = result["sentence_grounding"]
    assert grounding["schema_version"] == "sentence-grounding-v1"
    assert grounding["entries"]
    assert any(
        "c43682_a43100_f1" in entry.get("used_fact_ids", []) for entry in grounding["entries"]
    )
    sent_messages = llm.invoke.call_args_list[0].args[0]
    full_prompt = "\n".join(message["content"] for message in sent_messages)
    user_prompt = sent_messages[1]["content"]
    assert "피어 프로필 기반 시사점 생성" in user_prompt
    assert "사실 기반 해석" in user_prompt
    assert "최종 시사점으로 끝내지 않습니다" in user_prompt
    assert "피어사의 기존 사업영역/역량" in user_prompt
    assert "피어사가 원래 어떤 역량/사업영역을 갖고 있었는지" in user_prompt
    assert "business_area/core_capability/recent_direction" in user_prompt
    assert "기존 역량이 이번 사건에서 어떤 적용 장면" in user_prompt
    assert "SK AX 프로필 기반 대응 생성" in user_prompt
    assert "profile_context 나 recent context 가 없거나" in user_prompt
    assert "SK AX profile_context 의 관련 사업영역/역량" in user_prompt
    assert (
        "현재 상태 → 왜 바꿔야 하는가 → 무엇을 바꿔야 하는가 → 바꾸면 무엇이 달라지는가"
        in user_prompt
    )
    for vague_claim in ("기술적 우위", "선점", "격차", "경쟁 심화"):
        assert vague_claim in full_prompt
    assert "business_line_mapping" in user_prompt
    for action_context in ("제안서", "PoC", "레퍼런스 비교표"):
        assert action_context in user_prompt
    assert "SK AX 프로필과 연결되지 않은 대응방향" in user_prompt
    review_messages = llm.invoke.call_args_list[1].args[0]
    review_prompt = review_messages[1]["content"]
    assert "전략 QA reviewer" in review_messages[0]["content"]
    assert "1차 결과" in review_prompt
    assert "기존 사업영역/역량 → 현재 사건 접점 → 사업적 의미" in review_prompt
    assert "recommended_actions" in review_prompt


def test_strategic_insight_agent_empty_when_integrated_issue_invalid():
    payload = _fixture()
    result = StrategicInsightAgent(llm=MagicMock()).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue={"is_valid_summary": False},
        classification=payload["classification"],
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is False
    _assert_strategic_insight_schema(result)
    assert result["analysis"]["is_valid_analysis"] is False
    assert result["implication"]["is_valid_implication"] is False


def test_strategic_insight_agent_filters_business_lines_by_profile_candidates():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 통합 이슈는 특정 고객군의 적용 기준 변화를 보여줍니다.",
            "strategic_meaning": ["확인된 사업명과 검증 단계가 기존 역량과 연결됩니다."],
            "market_signal": (
                "토큰증권 컨설팅과 테스트베드 구축처럼 고객은 도입 기능과 "
                "운영 기준을 함께 확인하고 있습니다."
            ),
            "impact_level": "high",
            "impact_reason": "통합 이슈에서 고객 적용과 운영 기준 근거가 함께 확인됩니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 fact_basis를 기반으로 판단했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": (
                    "피어사는 확인된 사업명과 검증 단계를 고객 적용 메시지로 연결하고 있습니다."
                ),
                "capability_change": "기존 역량이 고객 적용 장면으로 연결됩니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX도 적용 범위와 운영 기준을 제안에 반영해야 합니다.",
                "potential_impact": (
                    "고객은 입력 근거가 보여준 적용 범위와 운영 기준을 비교할 수 "
                    "있습니다. 따라서 SK AX는 제안서에서 적용 범위와 운영 책임을 "
                    "분리해 설명해야 합니다."
                ),
                "opportunities": ["운영 기준을 포함한 제안 구성을 만들 수 있습니다."],
                "threats": ["근거가 약한 제안은 고객 비교 단계에서 설득력이 낮아질 수 있습니다."],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 적용 범위와 "
                        "운영 책임을 함께 비교할 수 있습니다. 따라서 SK AX 제안서에서 "
                        "적용 범위와 운영 책임을 별도 항목으로 설명합니다."
                    )
                ],
                "business_line_mapping": [TEST_LINE_INVALID, TEST_LINE_A, TEST_LINE_B],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.78,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A, TEST_LINE_B])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    assert result["implication"]["skax_implication"]["business_line_mapping"] == [
        TEST_LINE_A,
        TEST_LINE_B,
    ]


def test_strategic_insight_agent_empty_business_line_candidates_returns_empty_mapping():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 통합 이슈는 특정 고객군의 적용 기준 변화를 보여줍니다.",
            "strategic_meaning": ["확인된 사업명과 검증 단계가 기존 역량과 연결됩니다."],
            "market_signal": (
                "토큰증권 컨설팅과 테스트베드 구축처럼 고객은 도입 기능과 "
                "운영 기준을 함께 확인하고 있습니다."
            ),
            "impact_level": "high",
            "impact_reason": "통합 이슈에서 고객 적용과 운영 기준 근거가 함께 확인됩니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 fact_basis를 기반으로 판단했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": (
                    "피어사는 확인된 사업명과 검증 단계를 고객 적용 메시지로 연결하고 있습니다."
                ),
                "capability_change": "기존 역량이 고객 적용 장면으로 연결됩니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX도 적용 범위와 운영 기준을 제안에 반영해야 합니다.",
                "potential_impact": (
                    "고객은 입력 근거가 보여준 적용 범위와 운영 기준을 비교할 수 "
                    "있습니다. 따라서 SK AX는 제안서에서 적용 범위와 운영 책임을 "
                    "분리해 설명해야 합니다."
                ),
                "opportunities": ["운영 기준을 포함한 제안 구성을 만들 수 있습니다."],
                "threats": ["근거가 약한 제안은 고객 비교 단계에서 설득력이 낮아질 수 있습니다."],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 적용 범위와 "
                        "운영 책임을 함께 비교할 수 있습니다. 따라서 SK AX 제안서에서 "
                        "적용 범위와 운영 책임을 별도 항목으로 설명합니다."
                    )
                ],
                "business_line_mapping": [TEST_LINE_INVALID, TEST_LINE_A],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.78,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))
    profile_context = {
        **payload["profile_context"],
        "skax_profile": {"company_id": "sk_ax"},
    }

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    assert result["implication"]["skax_implication"]["business_line_mapping"] == []


def test_strategic_insight_agent_marks_invalid_when_abstract_actions_survive_repair():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 통합 이슈는 고객 적용 범위와 검증 기준을 보여줍니다.",
            "strategic_meaning": ["입력 근거에서 고객 적용 장면과 검증 단계가 함께 확인됩니다."],
            "market_signal": "고객은 도입 기능보다 적용 범위와 검증 기준을 함께 확인합니다.",
            "impact_level": "high",
            "impact_reason": "통합 이슈에서 고객 적용과 검증 단계 근거가 함께 확인됩니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 fact_basis를 기반으로 판단했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": (
                    "피어사는 입력 근거를 기존 역량과 연결해 고객 적용 범위를 설명합니다."
                ),
                "capability_change": (
                    "고객 적용 범위와 검증 단계까지 프로필 역량의 연결 범위가 넓어집니다."
                ),
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX 제안에서도 적용 범위와 검증 기준을 보여줘야 합니다.",
                "potential_impact": (
                    "고객은 입력 근거의 적용 범위와 검증 기준을 비교할 수 있습니다. "
                    "따라서 SK AX는 제안서에서 적용 범위와 검증 기준을 분리해 설명해야 합니다."
                ),
                "opportunities": ["적용 범위와 검증 기준을 포함한 제안서 개발"],
                "threats": ["검증 근거가 약하면 고객 비교 단계에서 설득력이 낮아질 수 있습니다."],
                "recommended_actions": [
                    "제안서에 적용 범위와 검증 기준 강조",
                    "제안서에 적용 범위와 검증 기준 강조",
                    "운영 모델에서 적용 범위와 검증 책임을 별도 항목으로 설명합니다.",
                ],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.78,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    actions = result["implication"]["skax_implication"]["recommended_actions"]
    assert result["is_valid_strategic_insight"] is True
    assert result["implication"]["is_valid_implication"] is True
    assert "quality_gate_failed" not in result["analysis"]["reason"]
    assert actions
    assert actions == [
        "제안서에 적용 범위와 검증 기준 강조",
        "운영 모델에서 적용 범위와 검증 책임을 별도 항목으로 설명합니다.",
    ]
    action_text = " ".join(actions)
    assert "적용 범위" in action_text
    assert "검증" in action_text
    assert (
        result["implication"]["peer_implication"]["sourced_evidence_ids"]
        == llm_payload["implication"]["peer_implication"]["sourced_evidence_ids"]
    )


def test_strategic_insight_agent_repairs_overstated_relationship_and_generic_signal():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    fact_id = "fact:relationship-grounding"
    fact = "피어사는 고객사의 지분 2%를 취득하기로 결의했습니다."
    integrated_issue = _minimal_integrated_issue_with_fact(
        company_id=company_id,
        fact_id=fact_id,
        fact=fact,
    )
    classification = {
        **payload["classification"],
        "sectors": ["test_sector"],
        "event_type": "investment",
    }
    input_bundle = {
        **payload["input_bundle"],
        "bundle_id": integrated_issue["bundle_id"],
        "companies": [company_id],
        "sectors": ["test_sector"],
        "event_type": "investment",
    }
    bad_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사는 고객사와의 협업으로 사업 기회를 넓히고 있습니다.",
            "strategic_meaning": [
                "피어사는 고객사와 협업해 적용 범위를 확장하고 있습니다.",
            ],
            "market_signal": "고객 요구가 변화하고 있습니다.",
            "impact_level": "high",
            "impact_reason": "고객사와의 협업이 직접 근거입니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 근거에서 확인된 역량 맥락을 바탕으로 판단했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "피어사는 고객사와 협업해 사업 메시지를 구체화하고 있습니다.",
                "capability_change": "협업 기반 운영 범위가 넓어지고 있습니다.",
                "sourced_evidence_ids": [fact_id],
            },
            "skax_implication": {
                "why_important": "SK AX 제안에서도 협업 메시지를 봐야 합니다.",
                "potential_impact": (
                    "고객 요구가 변화하면서 SK AX 제안에도 영향을 줄 수 있습니다. "
                    "따라서 SK AX는 제안서에서 협업 모델을 재검토해야 합니다."
                ),
                "opportunities": ["협업 메시지를 강조할 기회"],
                "threats": ["협업 메시지 선점"],
                "recommended_actions": ["제안서에서 협업을 강조합니다."],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
    }
    repair_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사는 고객사 지분 취득 결의를 통해 고객 접점을 확보했습니다.",
            "strategic_meaning": [
                "고객사 지분 2% 취득 결의는 고객사와의 사업 접점을 투자 관계로 "
                "확보했다는 신호입니다.",
            ],
            "market_signal": (
                "고객사 지분 취득 결의처럼 투자 관계를 통해 고객 접점을 확보하는 방식이 확인됩니다."
            ),
            "impact_level": "medium",
            "impact_reason": "통합 이슈의 지분 2% 취득 결의가 관계 수준의 직접 근거입니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.75,
            "reason": "고객사 지분 취득 결의가 적용 범위 판단의 근거입니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": (
                    "피어사는 고객사 지분 2% 취득 결의를 통해 고객 접점을 확보했습니다. "
                    "이 관계는 투자 관계이므로, 프로필 역량과 연결할 때도 "
                    "사업 접점 확보 수준으로 해석해야 합니다."
                ),
                "capability_change": (
                    "기존 역량이 실행 계약으로 바뀐 것이 아니라, 투자 관계를 통해 고객 접점과 "
                    "후속 제안 가능성을 확인하는 단계로 넓어졌습니다."
                ),
                "sourced_evidence_ids": [fact_id],
            },
            "skax_implication": {
                "why_important": (
                    "SK AX도 피어사 신호를 비교할 때 고객 접점의 관계 수준과 "
                    "후속 검증 가능성을 분리해 봐야 합니다."
                ),
                "potential_impact": (
                    "고객은 지분 취득처럼 실행 계약 이전 단계의 관계도 후속 사업 접점으로 "
                    "비교할 수 있습니다. 따라서 SK AX는 제안서에서 고객 접점, 관계 수준, "
                    "후속 PoC 전환 기준을 분리해 설명해야 합니다."
                ),
                "opportunities": [
                    "투자 관계와 후속 PoC 전환 기준을 함께 제시하는 제안 구성을 만들 수 있습니다."
                ],
                "threats": [
                    "관계 수준을 구분하지 못하면 실행 계약 근거를 가진 경쟁사와 "
                    "비교될 때 설명력이 낮아질 수 있습니다."
                ],
                "recommended_actions": [
                    "지분 취득 신호 때문에 고객은 실행 계약 이전 단계의 관계 수준과 후속 "
                    "PoC 전환 가능성을 따로 비교할 수 있습니다. 따라서 SK AX 제안서에서 "
                    "고객 접점의 관계 수준과 후속 PoC 전환 기준을 별도 항목으로 설명합니다."
                ],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": ["지분 취득 이후 실제 공동 과제가 열리는지 확인이 필요합니다."],
            "watch_points": ["후속 공시나 고객 사례에서 PoC 전환 기준이 구체화되는지 확인합니다."],
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(
        side_effect=[
            _fake_llm_response(bad_payload),
            _fake_llm_response({"needs_revision": True, "revised_result": bad_payload}),
            _fake_llm_response(repair_payload),
        ]
    )
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=input_bundle,
        integrated_issue=integrated_issue,
        classification=classification,
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    output_text = json.dumps(result, ensure_ascii=False)
    assert result["is_valid_strategic_insight"] is True, json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    assert "협업으로 사업 기회" not in output_text
    assert "고객 요구가 변화" not in output_text
    assert "지분 취득" in output_text
    assert llm.invoke.call_count == 3


def test_strategic_insight_agent_self_review_revises_vague_impact_and_actions():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    first_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 기술적 우위가 강화됩니다.",
            "strategic_meaning": ["기술 융합을 통한 혁신성이 중요해지고 있습니다."],
            "market_signal": "경쟁 심화와 혁신성이 중요해지고 있습니다.",
            "impact_level": "high",
            "impact_reason": "기술적 우위가 강화되기 때문입니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "경쟁력이 높아질 것으로 보입니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "피어사는 확인된 통합 이슈에서 기술적 우위를 강화합니다.",
                "capability_change": "경쟁력이 강화됩니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX의 제안 기준에 영향을 줄 수 있습니다.",
                "potential_impact": "SK AX는 제안 방식을 재검토해야 합니다.",
                "opportunities": ["혁신성을 강조할 기회"],
                "threats": ["경쟁 심화"],
                "recommended_actions": [
                    "제안서에 기술 융합과 혁신성을 강조합니다.",
                    "PoC에서 기술 융합과 혁신성을 강조합니다.",
                ],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
    }
    revised_payload = {
        "needs_revision": True,
        "violations": ["근거 없는 추상 표현과 이유 없는 potential_impact를 수정했습니다."],
        "revised_result": {
            "is_valid_strategic_insight": True,
            "analysis": {
                "is_valid_analysis": True,
                "analysis_scope": "peer_and_industry",
                "analysis_summary": (
                    "피어사는 토큰증권 컨설팅과 테스트베드 구축을 계기로 고객 적용 "
                    "범위를 구체화하고 있습니다."
                ),
                "strategic_meaning": [
                    (
                        "통합 이슈의 컨설팅 범위와 검증 환경 근거는 고객이 실행 범위와 "
                        "확인 기준을 함께 본다는 점을 보여줍니다."
                    )
                ],
                "market_signal": (
                    "토큰증권 컨설팅과 테스트베드 구축처럼 고객 제안에서는 적용 범위와 "
                    "검증 환경을 함께 제시하는 흐름이 확인됩니다."
                ),
                "impact_level": "high",
                "impact_reason": (
                    "토큰증권 컨설팅과 테스트베드 구축 fact_id에서 고객 적용 범위와 "
                    "검증 환경 근거가 확인됩니다."
                ),
                "risk_or_opportunity": "opportunity",
                "confidence": 0.8,
                "reason": "토큰증권 컨설팅 근거와 프로필의 기존 역량을 함께 사용했습니다.",
            },
            "implication": {
                "is_valid_implication": True,
                "implication_scope": "peer_and_skax",
                "peer_implication": {
                    "company_id": company_id,
                    "company_name_ko": company_name,
                    "peer_meaning": (
                        "피어사는 토큰증권 컨설팅과 테스트베드 구축 근거를 기존 프로필 역량과 "
                        "연결하고 있습니다. 이 근거는 기능 설명보다 적용 범위와 검증 "
                        "기준을 함께 제시하는 방향을 보여줍니다."
                    ),
                    "capability_change": (
                        "프로필의 기존 역량이 고객 적용 범위와 검증 환경까지 연결됩니다."
                    ),
                    "sourced_evidence_ids": ["c43682_a43100_f1", "c43682_a43158_f1"],
                },
                "skax_implication": {
                    "why_important": (
                        "SK AX 제안에서도 기능 설명보다 적용 범위와 검증 환경 설명이 중요해집니다."
                    ),
                    "potential_impact": (
                        "통합 이슈에서 적용 범위와 검증 환경이 함께 확인되므로 고객은 "
                        "제안 단계에서 실행 범위와 확인 기준을 비교할 수 있습니다. "
                        "따라서 SK AX는 제안서에서 적용 범위와 PoC 검증 항목을 "
                        "분리해 설명해야 합니다."
                    ),
                    "opportunities": [
                        ("제안서에서 적용 범위와 PoC 검증 항목을 묶어 제시할 수 있습니다.")
                    ],
                    "threats": [
                        (
                            "고객이 검증 환경 근거를 직접 비교하면 "
                            "검증 근거가 약한 제안은 설득력이 낮아질 수 있습니다."
                        )
                    ],
                    "recommended_actions": [
                        (
                            "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 적용 "
                            "범위와 운영 책임을 따로 판단할 수 있습니다. 따라서 SK AX "
                            "토큰증권 제안서의 적용 범위표와 운영 책임 정리를 별도 섹션으로 "
                            "재구성할 필요가 있습니다."
                        ),
                        (
                            "테스트베드 구축 신호 때문에 고객은 PoC 통과 기준을 확인하려 "
                            "합니다. 따라서 SK AX PoC에서는 관련 검증 항목과 고객 확인 "
                            "기준을 함께 검증하는 구조로 재구성할 필요가 있습니다."
                        ),
                    ],
                    "business_line_mapping": [TEST_LINE_A],
                },
                "follow_up_questions": [
                    "토큰증권 컨설팅에서 확인된 고객 적용 범위는 어디까지인가?"
                ],
                "watch_points": ["후속 근거에서 검증 항목이 실제 요구사항으로 구체화되는지"],
                "confidence": 0.75,
                "evidence_label": "moderate",
                "provenance": {
                    "used_fact_ids": ["c43682_a43100_f1", "c43682_a43158_f1"],
                    "used_context_layers": [],
                },
            },
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(
        side_effect=[
            _fake_llm_response(first_payload),
            _fake_llm_response(revised_payload),
        ]
    )
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    skax = result["implication"]["skax_implication"]
    assert "재검토" not in skax["potential_impact"]
    assert "기술 융합과 혁신성" not in " ".join(skax["recommended_actions"])
    assert "c43682_a43158_f1" in result["implication"]["peer_implication"]["sourced_evidence_ids"]
    assert llm.invoke.call_count == 2


def test_strategic_insight_agent_self_review_can_mark_low_quality_result_invalid():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    first_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사 신호가 중요합니다.",
            "strategic_meaning": ["고객 요구가 변화하고 있습니다."],
            "market_signal": "관련 시장에서 중요해지고 있습니다.",
            "impact_level": "high",
            "impact_reason": "중요한 신호이기 때문입니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 근거를 봤습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "피어사에 의미가 있습니다.",
                "capability_change": "역량이 강화됩니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX에 중요합니다.",
                "potential_impact": "SK AX는 대응해야 합니다.",
                "opportunities": ["기회가 있습니다."],
                "threats": [],
                "recommended_actions": ["전략을 강화합니다."],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
    }
    invalid_review_payload = {
        "needs_revision": True,
        "violations": ["근거 사실과 프로필 연결이 부족하고 대응방향의 이유가 설명되지 않습니다."],
        "revised_result": {
            "is_valid_strategic_insight": False,
            "analysis": {
                "is_valid_analysis": False,
                "analysis_scope": "peer_and_industry",
                "analysis_summary": "",
                "strategic_meaning": [],
                "market_signal": "",
                "impact_level": "low",
                "impact_reason": "입력 근거만으로 논리적 분석 문장을 복구하기 어렵습니다.",
                "risk_or_opportunity": "neutral",
                "confidence": 0.2,
                "reason": (
                    "현재 출력은 근거 사실, 피어 프로필, SK AX 대응 산출물의 연결이 부족합니다."
                ),
            },
            "implication": {
                "is_valid_implication": False,
                "implication_scope": "peer_and_skax",
                "peer_implication": {
                    "company_id": company_id,
                    "company_name_ko": company_name,
                    "peer_meaning": "",
                    "capability_change": "",
                    "sourced_evidence_ids": [],
                },
                "skax_implication": {
                    "why_important": "",
                    "potential_impact": "",
                    "opportunities": [],
                    "threats": [],
                    "recommended_actions": [],
                    "business_line_mapping": [],
                },
                "follow_up_questions": [],
                "watch_points": [],
                "confidence": 0.2,
                "evidence_label": "insufficient",
            },
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(
        side_effect=[
            _fake_llm_response(first_payload),
            _fake_llm_response(invalid_review_payload),
        ]
    )
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is False
    assert result["analysis"]["is_valid_analysis"] is False
    assert result["implication"]["is_valid_implication"] is False
    assert result["implication"]["evidence_label"] == "insufficient"
    assert llm.invoke.call_count == 2


def test_quality_gate_flags_customer_contract_role_and_unscoped_proposal_artifact():
    integrated_issue = {
        "is_valid_summary": True,
        "main_company": "lg_cns",
        "cluster_event_type": "contract",
        "headline": "공급계약 체결",
        "integrated_text": "인스웨이브가 LG CNS와 98억 규모의 웹단말 공급계약을 체결했다.",
        "fact_summary": ["인스웨이브가 LG CNS와 98억 규모의 웹단말 공급계약을 체결했다."],
        "cluster_fact_intelligence": {
            "activity_types": ["contract"],
            "customers_or_industries": ["LG CNS"],
            "products_or_services": ["웹단말"],
        },
        "consolidated_facts": [
            {
                "fact_id": "fact:contract",
                "fact": "인스웨이브가 LG CNS와 98억 규모의 웹단말 공급계약을 체결했다.",
                "customers_or_industries": ["LG CNS"],
                "products_or_services": ["웹단말"],
                "activity_types": ["contract"],
                "evidence_texts": ["인스웨이브, LG CNS와 98억 규모 웹단말 공급계약"],
            }
        ],
    }
    result = {
        "analysis": {
            "analysis_summary": "LG CNS의 웹단말 공급 역량이 강화됩니다.",
            "strategic_meaning": ["LG CNS가 금융 IT 사업 범위를 확장합니다."],
            "market_signal": "웹단말 공급계약 수요가 지속적으로 증가하고 있습니다.",
            "impact_reason": "웹단말 공급계약이 확인됐기 때문입니다.",
            "reason": "fact:contract를 사용했습니다.",
        },
        "implication": {
            "peer_implication": {
                "peer_meaning": "LG CNS의 웹단말 공급 역량이 강화됩니다.",
                "capability_change": "LG CNS가 장기적인 시스템 전환 및 운영 안정성을 확보합니다.",
            },
            "skax_implication": {
                "potential_impact": (
                    "고객은 공급계약 근거를 비교할 수 있습니다. "
                    "따라서 SK AX는 제안서에서 안정성을 설명해야 합니다."
                ),
                "recommended_actions": [
                    (
                        "공급계약 신호 때문에 고객은 안정성을 비교할 수 있습니다. "
                        "SK AX는 제안서에서 안정성과 비용 기준을 설명합니다."
                    )
                ],
            },
        },
    }

    violations = strategic_insight_module._quality_gate_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
    )

    assert any("공급자 역량" in violation for violation in violations)
    assert any("역량 강화/경쟁력 강화 성과" in violation for violation in violations)
    assert any("수요" in violation and "입력 근거 없이" in violation for violation in violations)
    assert not any("제안서가 어떤 고객/사업/도입 프로젝트" in violation for violation in violations)


def test_strategic_insight_agent_repairs_when_review_still_has_quality_violations():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    first_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 통합 이슈는 고객 적용 범위 신호입니다.",
            "strategic_meaning": ["입력 근거에서 고객 적용 범위가 확인됩니다."],
            "market_signal": "고객 제안에서 검증 환경 설명이 중요해지고 있습니다.",
            "impact_level": "high",
            "impact_reason": "입력 fact_basis의 고객 적용 근거가 확인됩니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 fact_basis를 기반으로 판단했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "피어사는 입력 근거에서 고객 적용 범위를 확인했습니다.",
                "capability_change": "고객 검증 환경으로 적용 범위가 넓어졌습니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX 제안도 검증 환경 설명이 중요합니다.",
                "potential_impact": "SK AX는 제안 방식을 재검토해야 합니다.",
                "opportunities": ["금융 검증 환경을 제안서에 반영할 수 있습니다."],
                "threats": ["피어사의 시장 점유율 확대"],
                "recommended_actions": ["제안서에서 기술 융합을 강조합니다."],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
    }
    review_payload = {
        "needs_revision": True,
        "violations": ["아직 potential_impact와 threat가 과도합니다."],
        "revised_result": first_payload,
    }
    repair_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": (
                "피어사는 토큰증권 컨설팅과 테스트베드 구축으로 고객 검증 단계의 "
                "적용 사례를 확보했습니다."
            ),
            "strategic_meaning": [
                (
                    "토큰증권 컨설팅 범위와 테스트베드 검증 환경은 고객이 실행 범위와 확인 "
                    "기준을 함께 보는 신호입니다."
                )
            ],
            "market_signal": (
                "토큰증권 컨설팅과 테스트베드 구축처럼 고객 제안에서 컨설팅 범위와 "
                "검증 환경이 함께 제시됩니다."
            ),
            "impact_level": "high",
            "impact_reason": "토큰증권 컨설팅과 테스트베드 구축 fact_id가 직접 근거입니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "토큰증권 컨설팅과 테스트베드 구축 근거를 사용했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": (
                    "피어사는 토큰증권 컨설팅과 테스트베드 구축에서 고객 적용 범위와 "
                    "검증 환경을 함께 "
                    "제시했습니다. 이 근거는 기능 설명보다 실행 범위와 확인 기준을 "
                    "함께 설명하는 방향으로 이어집니다."
                ),
                "capability_change": "컨설팅 범위와 검증 단계까지 적용 범위가 넓어졌습니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX 제안에서도 검증 환경과 운영 책임 설명이 중요합니다.",
                "potential_impact": (
                    "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 제안에서 컨설팅 "
                    "범위와 검증 항목을 함께 비교할 수 있습니다. 따라서 SK AX는 "
                    "제안서에서 적용 업무 범위와 PoC "
                    "검증 항목을 분리해 설명해야 합니다."
                ),
                "opportunities": ["제안서에서 컨설팅 범위와 검증 항목을 함께 제시할 수 있습니다."],
                "threats": [
                    (
                        "테스트베드 구축 레퍼런스 비교에서 검증 근거가 약하면 "
                        "설득력이 낮아질 수 있습니다."
                    )
                ],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 업무 적용 "
                        "범위와 데이터 관리 책임을 따로 비교할 수 있습니다. 따라서 SK AX "
                        "제안서에서 두 항목을 별도 항목으로 설명합니다."
                    ),
                    (
                        "테스트베드 구축 신호 때문에 고객은 테스트베드 통과 여부를 판단하려 "
                        "합니다. 따라서 SK AX PoC에서 검증 항목과 고객 확인 기준을 함께 "
                        "제시합니다."
                    ),
                ],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": ["검증 환경에서 확인할 업무 범위는 어디까지인가?"],
            "watch_points": ["후속 근거에서 검증 환경의 구축 범위가 구체화되는지"],
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(
        side_effect=[
            _fake_llm_response(first_payload),
            _fake_llm_response(review_payload),
            _fake_llm_response(repair_payload),
        ]
    )
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    skax = result["implication"]["skax_implication"]
    assert result["is_valid_strategic_insight"] is True
    assert "시장 점유율" not in " ".join(skax["threats"])
    assert "재검토" not in skax["potential_impact"]
    assert llm.invoke.call_count == 3


def test_strategic_insight_agent_fails_closed_when_overclaim_repair_fails():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    bad_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 고객 적용 근거는 기술적 역량을 입증하는 사례입니다.",
            "strategic_meaning": [
                "피어사는 고객 적용 근거를 통해 기술적 역량을 입증했습니다.",
                (
                    "프로필 역량과 입력 근거를 결합하여 대상 고객군에서의 "
                    "경쟁력을 강화하고 있습니다."
                ),
            ],
            "market_signal": "관련 시장에서 생태계 확장이 주목받고 있습니다.",
            "impact_level": "high",
            "impact_reason": "향후 시장 확장 가능성을 제시합니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "금융권 디지털 인프라 시장에서의 기회를 창출할 수 있습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "피어사는 고객 적용 근거를 통해 기술적 역량을 입증했습니다.",
                "capability_change": (
                    "피어사는 대상 고객군에서 기술적 역량을 입증하며 경쟁력을 강화하고 있습니다."
                ),
                "sourced_evidence_ids": [
                    "c43682_a43100_f1",
                    "c43682_a43158_f1",
                ],
            },
            "skax_implication": {
                "why_important": (
                    "피어사의 고객 적용 근거는 SK AX의 제안 기준에 영향을 줄 수 있습니다."
                ),
                "potential_impact": (
                    "피어사의 고객 적용 근거로 인해 고객은 기술적 역량과 협업 능력을 "
                    "더 중시하게 될 것입니다. SK AX는 제안서와 PoC에서 입력 근거와 "
                    "관련된 혁신을 강조해야 합니다."
                ),
                "opportunities": [
                    (
                        "프로필 역량 기반의 제안 혁신을 통해 고객에게 차별화된 "
                        "가치를 제공할 수 있는 기회"
                    )
                ],
                "threats": ["피어사의 관련 시장 점유율 확대"],
                "recommended_actions": [
                    "제안서에서 프로필 역량 기반 혁신을 강조하는 내용을 추가합니다.",
                    "PoC에서 협업 모델을 구체화하여 고객에게 명확한 가치를 제시합니다.",
                ],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": [
                "피어사의 고객 적용 근거가 시장에 미칠 장기적인 영향은 무엇인가?"
            ],
            "watch_points": ["협업을 통한 기술적 시너지 효과"],
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(
        side_effect=[
            _fake_llm_response(bad_payload),
            _fake_llm_response({"needs_revision": True, "revised_result": bad_payload}),
            _fake_llm_response(bad_payload),
        ]
    )
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is False
    assert result["analysis"]["is_valid_analysis"] is False
    assert result["implication"]["is_valid_implication"] is False
    assert "quality_gate_failed" in result["analysis"]["reason"]
    assert result["implication"]["evidence_label"] == "insufficient"


def test_strategic_insight_agent_fails_closed_when_self_review_errors_with_hard_violation():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    bad_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 고객 적용 근거가 시장 선점으로 이어집니다.",
            "strategic_meaning": ["피어사의 고객 적용 근거가 시장 선점을 보여줍니다."],
            "market_signal": "관련 시장에서 고객 적용 근거가 중요해지고 있습니다.",
            "impact_level": "high",
            "impact_reason": "시장 선점 가능성이 크기 때문입니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 근거를 기반으로 판단했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "피어사의 고객 적용 근거가 시장 선점으로 이어집니다.",
                "capability_change": "피어사의 고객 적용 근거가 경쟁력을 강화합니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX에 중요한 신호입니다.",
                "potential_impact": "고객은 시장 선점 여부를 비교할 수 있습니다.",
                "opportunities": ["시장 선점 흐름에 대응할 기회"],
                "threats": [],
                "recommended_actions": ["고객 적용 근거를 제안 산출물에 반영합니다."],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(
        side_effect=[
            _fake_llm_response(bad_payload),
            RuntimeError("self-review unavailable"),
        ]
    )
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is False
    assert result["analysis"]["is_valid_analysis"] is False
    assert result["implication"]["is_valid_implication"] is False
    assert "quality_gate_failed" in result["analysis"]["reason"]
    assert llm.invoke.call_count == 2


def test_strategic_insight_agent_generates_from_analysis_package():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_summary": "피어사가 확인된 통합 이슈를 사업 메시지로 전환하고 있습니다.",
            "strategic_meaning": [
                "통합 이슈의 핵심 사실을 기반으로 피어사의 적용 범위가 구체화됩니다.",
            ],
            "market_signal": (
                "토큰증권 컨설팅과 테스트베드 구축처럼 고객 평가는 기술 보유보다 "
                "운영 책임과 검증 기준을 함께 봅니다."
            ),
            "impact_level": "medium",
            "impact_reason": "통합 이슈의 fact_id로 확인된 사업 신호가 근거입니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.7,
            "reason": "토큰증권 컨설팅 근거와 classification을 함께 사용했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "피어사는 확인된 사업 신호를 운영 메시지로 연결합니다.",
                "capability_change": "운영 책임 설명 범위가 넓어집니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX 제안 기준에도 운영 책임 설명이 중요해집니다.",
                "potential_impact": (
                    "고객은 기능 보유 여부보다 운영 책임과 검증 기준을 함께 비교할 수 "
                    "있습니다. 따라서 SK AX는 제안서에서 데이터 보관 위치와 운영 책임 "
                    "범위를 분리해 설명해야 합니다."
                ),
                "opportunities": ["운영 책임 기준을 포함한 제안 구성"],
                "threats": ["근거 없는 기능 중심 메시지의 설득력 약화"],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 데이터 보관 "
                        "위치와 운영 책임 범위를 따로 비교할 수 있습니다. 따라서 SK AX "
                        "제안서에 두 판단 기준을 분리해 설명합니다."
                    ),
                ],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": ["실제 고객 적용 범위는 어디까지인가?"],
            "watch_points": ["검증 기준이 고객 제안서에 반영되는지"],
            "confidence": 0.7,
            "evidence_label": "moderate",
            "provenance": {
                "used_fact_ids": ["c43682_a43100_f1"],
                "used_context_layers": [],
            },
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))
    analysis_package = {
        "bundle_id": payload["integrated_issue"]["bundle_id"],
        "integrated_issue": payload["integrated_issue"],
        "classification": payload["classification"],
        "sources": payload["input_bundle"].get("sources") or [],
    }
    profile_context = {
        **payload["profile_context"],
        "skax_profile": {
            **payload["profile_context"].get("skax_profile", {}),
            "business_areas": [{"name": TEST_LINE_A}],
        },
    }

    result = StrategicInsightAgent(llm=llm).generate_from_analysis_package(
        analysis_package,
        profile_context=profile_context,
        analysis_context={},
    )

    _assert_strategic_insight_schema(result)
    assert result["is_valid_strategic_insight"] is True
    assert result["analysis"]["analysis_summary"]
    assert result["implication"]["skax_implication"]["business_line_mapping"] == [TEST_LINE_A]
    first_messages = llm.invoke.call_args_list[0].args[0]
    first_user_prompt = first_messages[1]["content"]
    assert "## StrategicEvidencePack" in first_user_prompt
    assert "## ProfileContext" in first_user_prompt
    assert "fact_basis" in first_user_prompt
    assert "시사점" in first_user_prompt
    assert "SK AX 대응 방향" in first_user_prompt


def test_strategic_insight_agent_generates_from_integrated_issue_id(monkeypatch):
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_summary": "통합 이슈 기준으로 전략 의미를 생성합니다.",
            "strategic_meaning": ["통합 이슈의 근거 사실이 피어사 방향성을 보여줍니다."],
            "market_signal": (
                "토큰증권 컨설팅과 테스트베드 구축처럼 고객 평가는 운영 책임과 "
                "검증 기준을 함께 봅니다."
            ),
            "impact_level": "medium",
            "impact_reason": "통합 이슈의 fact_id가 근거입니다.",
            "risk_or_opportunity": "neutral",
            "confidence": 0.7,
            "reason": "integrated_issue_id 기반 입력을 사용했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "피어사는 확인된 이슈를 사업 메시지로 연결합니다.",
                "capability_change": "운영 책임 설명 범위가 넓어집니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX 제안 기준에도 검증 기준 설명이 중요해집니다.",
                "potential_impact": (
                    "고객은 통합 이슈의 근거 사실을 기준으로 책임 범위와 검증 기준을 "
                    "비교할 수 있습니다. 따라서 SK AX는 제안서에서 운영 책임과 검증 "
                    "항목을 분리해 설명해야 합니다."
                ),
                "opportunities": ["검증 기준을 포함한 제안 구성"],
                "threats": ["근거 없는 기능 중심 메시지의 설득력 약화"],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 데이터 보관 "
                        "위치와 운영 책임 범위를 따로 비교할 수 있습니다. 따라서 SK AX "
                        "제안서에 두 판단 기준을 분리해 설명합니다."
                    ),
                ],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": ["고객 적용 범위는 어디까지인가?"],
            "watch_points": ["검증 기준이 제안서에 반영되는지"],
            "confidence": 0.7,
            "evidence_label": "moderate",
            "provenance": {
                "used_fact_ids": ["c43682_a43100_f1"],
                "used_context_layers": [],
            },
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))
    analysis_package = {
        "bundle_id": payload["integrated_issue"]["bundle_id"],
        "integrated_issue": payload["integrated_issue"],
        "classification": payload["classification"],
        "sources": payload["input_bundle"].get("sources") or [],
    }
    monkeypatch.setattr(
        strategic_insight_module,
        "_load_integrated_issue_analysis_package",
        lambda issue_id: {"integrated_issue_id": issue_id, "analysis_package": analysis_package},
    )
    monkeypatch.setattr(
        strategic_insight_module,
        "_load_profile_context_for_issue",
        lambda **_: payload["profile_context"],
    )
    monkeypatch.setattr(
        strategic_insight_module,
        "_build_analysis_context_for_issue",
        lambda **_: payload["analysis_context"],
    )

    result = StrategicInsightAgent(llm=llm).generate_from_integrated_issue_id(
        "00000000-0000-0000-0000-000000000001"
    )

    _assert_strategic_insight_schema(result)
    assert result["is_valid_strategic_insight"] is True
    assert result["analysis"]["analysis_summary"] == "통합 이슈 기준으로 전략 의미를 생성합니다."


def test_strategic_insight_agent_uses_legacy_fallback_on_malformed_llm_output():
    payload = _fixture()
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response({"unexpected": "shape"}))
    fallback_analyzer = MagicMock()
    fallback_analyzer.analyze.return_value = {
        "is_valid_analysis": True,
        "analysis_summary": "fallback analysis",
    }
    fallback_implication = MagicMock()
    fallback_implication.generate.return_value = {
        "is_valid_implication": True,
        "peer_implication": {"peer_meaning": "fallback peer"},
        "skax_implication": {"why_important": "fallback skax"},
    }

    result = StrategicInsightAgent(
        llm=llm,
        fallback_analyzer=fallback_analyzer,
        fallback_implication_agent=fallback_implication,
    ).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is True
    _assert_strategic_insight_schema(result)
    fallback_analyzer.analyze.assert_called_once()
    fallback_implication.generate.assert_called_once()


def test_strategic_insight_agent_with_mock_issue_and_common_db_profile():
    _load_env_file_for_integration()
    if os.getenv("RUN_STRATEGIC_INSIGHT_INTEGRATION") != "1":
        pytest.skip("Set RUN_STRATEGIC_INSIGHT_INTEGRATION=1 to call common DB and real LLM.")
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL is required for common DB profile lookup.")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for the real StrategicInsightAgent LLM call.")

    from src.services.profile_context_loader import ProfileContextLoader

    payload = _fixture()
    integrated_issue = payload["integrated_issue"]
    classification = payload["classification"]
    sectors = classification.get("sectors") or [classification.get("sector")]

    profile_context = (
        ProfileContextLoader()
        .load(
            companies=_companies_from_integrated_issue(integrated_issue),
            sectors=[sector for sector in sectors if sector],
            event_type=classification.get("event_type"),
            strict=True,
        )
        .to_dict()
    )
    assert profile_context["peer_profiles"]
    peer_profile = profile_context["peer_profiles"].get(integrated_issue["main_company"]) or {}
    assert any(
        peer_profile.get(key)
        for key in (
            "schema_version",
            "profile_snapshot_version",
            "one_liner",
            "company_summary",
            "business_areas",
            "core_capabilities",
        )
    ), "common DB peer profile snapshot was not loaded; fail-soft fallback was returned."
    skax_profile = profile_context.get("skax_profile") or {}
    print(
        "[profile_context]",
        json.dumps(
            {
                "peer_profile_loaded": {
                    "company_id": peer_profile.get("company_id"),
                    "schema_version": peer_profile.get("schema_version"),
                    "profile_snapshot_version": peer_profile.get("profile_snapshot_version"),
                    "has_one_liner": bool(peer_profile.get("one_liner")),
                    "business_area_count": len(peer_profile.get("business_areas") or []),
                },
                "skax_profile_loaded": {
                    "company_id": skax_profile.get("company_id"),
                    "schema_version": skax_profile.get("schema_version"),
                    "profile_snapshot_version": skax_profile.get("profile_snapshot_version"),
                    "has_one_liner": bool(skax_profile.get("one_liner")),
                    "business_lines": skax_profile.get("business_lines"),
                    "business_area_count": len(skax_profile.get("business_areas") or []),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )

    result = StrategicInsightAgent().generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=integrated_issue,
        classification=classification,
        profile_context=profile_context,
        analysis_context=payload.get("analysis_context") or {},
    )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    assert result["is_valid_strategic_insight"] is True, json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    _assert_strategic_insight_schema(result)
    assert result["analysis"]["is_valid_analysis"] is True
    assert result["implication"]["is_valid_implication"] is True
    assert result["implication"]["provenance"]["generator"] == "StrategicInsightAgent"
