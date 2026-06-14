"""StrategicInsightAgent unit tests with mock LLM response."""

from __future__ import annotations

import json
import os
import re
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


def _fake_llm_response(payload: dict[str, Any], *, add_default_frontend_ready: bool = True) -> Any:
    response = MagicMock()
    body = _with_default_frontend_ready(payload) if add_default_frontend_ready else payload
    response.content = json.dumps(body, ensure_ascii=False)
    return response


def _fake_frontend_ready_repair_response(
    *,
    key_sentence: str = (
        "현재 사건은 피어사의 고객 제안 범위와 운영 책임이 함께 비교되는 흐름입니다."
    ),
    key_evidence: str = (
        "현재 사건 fact와 피어 사업 근거가 같은 대상 업무와 실행 범위를 가리킵니다."
    ),
    action_sentence: str = (
        "SK AX는 현재 사건과 유사한 사업에서 수행 범위와 검증 기준을 "
        "고객 제안 단위로 나눠야 합니다."
    ),
    action_evidence: str = (
        "이 구분이 있어야 SK AX가 고객 제안에서 직접 책임질 범위와 "
        "외부 확인이 필요한 범위를 설명할 수 있습니다."
    ),
    key_event_terms: list[str] | None = None,
    key_profile_terms: list[str] | None = None,
    action_event_terms: list[str] | None = None,
    action_skax_terms: list[str] | None = None,
    key_evidence_mode: str = "event_based",
    action_evidence_mode: str = "generic_monitoring",
) -> Any:
    key_sentence = _test_directional_key_sentence(key_sentence)
    payload = {
        "frontend_ready": {
            "source": "frontend_repair_direct",
            "key_implication": {
                "source": "frontend_repair_direct",
                "frame": "피어 사업 흐름",
                "claim_type": "event_based_signal",
                "claim_strength": "cautious",
                "evidence_mode": key_evidence_mode,
                "event_anchor_terms": key_event_terms or _test_anchor_terms(key_sentence),
                "profile_anchor_terms": key_profile_terms or _test_anchor_terms(key_evidence),
                "unsupported_claims_removed": [],
                "sentence": key_sentence,
                "evidence_sentence": key_evidence,
            },
            "suggested_action": {
                "source": "frontend_repair_direct",
                "frame": "SK AX 내부 점검",
                "claim_type": "internal_strategy_check",
                "claim_strength": "cautious",
                "evidence_mode": action_evidence_mode,
                "event_anchor_terms": action_event_terms or _test_anchor_terms(action_sentence),
                "skax_anchor_terms": action_skax_terms or _test_anchor_terms(action_evidence),
                "unsupported_claims_removed": [],
                "sentence": action_sentence,
                "evidence_sentence": action_evidence,
            },
        }
    }
    response = MagicMock()
    response.content = json.dumps(payload, ensure_ascii=False)
    return response


def _with_default_frontend_ready(payload: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(payload, ensure_ascii=False))
    revised = data.get("revised_result")
    if isinstance(revised, dict):
        data["revised_result"] = _with_default_frontend_ready(revised)
        return data
    implication = data.get("implication")
    if not isinstance(implication, dict) or implication.get("frontend_ready"):
        return data
    if implication.get("is_valid_implication") is False:
        return data
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    key_sentence = _test_directional_key_sentence(
        str(peer.get("peer_meaning") or "피어사는 현재 사건을 기존 사업과 연결합니다.")
    )
    key_evidence = str(
        peer.get("capability_change") or "현재 사건의 적용 대상이 기존 사업 범위와 연결됩니다."
    )
    action_sentence = str(
        (skax.get("recommended_actions") or [""])[0]
        or (
            "SK AX는 현재 사건과 유사한 사업에서 수행 범위와 검증 기준을 "
            "고객 제안 단위로 나눠야 합니다."
        )
    )
    action_evidence = str(
        skax.get("potential_impact")
        or skax.get("why_important")
        or (
            "이 구분이 있어야 SK AX가 고객 제안에서 직접 책임질 범위와 "
            "외부 확인이 필요한 범위를 설명할 수 있습니다."
        )
    )
    implication["frontend_ready"] = {
        "source": "llm_direct",
        "key_implication": {
            "source": "llm_direct",
            "frame": "피어 사업 흐름",
            "claim_type": "event_based_signal",
            "claim_strength": "cautious",
            "evidence_mode": "event_based",
            "event_anchor_terms": _test_anchor_terms(key_sentence),
            "profile_anchor_terms": _test_anchor_terms(key_evidence),
            "unsupported_claims_removed": [],
            "sentence": key_sentence,
            "evidence_sentence": key_evidence,
        },
        "suggested_action": {
            "source": "llm_direct",
            "frame": "SK AX 내부 점검",
            "claim_type": "internal_strategy_check",
            "claim_strength": "cautious",
            "evidence_mode": "generic_monitoring",
            "event_anchor_terms": _test_anchor_terms(action_sentence),
            "skax_anchor_terms": _test_anchor_terms(action_evidence) or ["SK AX"],
            "unsupported_claims_removed": [],
            "sentence": action_sentence,
            "evidence_sentence": action_evidence,
        },
    }
    data["implication"] = implication
    return data


def _test_anchor_terms(text: str) -> list[str]:
    terms = [
        token for token in re.findall(r"[A-Za-z0-9가-힣·&/]+", str(text or "")) if len(token) >= 2
    ]
    return terms[:2] or ["현재"]


def _test_directional_key_sentence(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return "현재 사건은 유사 사업의 평가 기준이 실행 범위와 검증 조건으로 이동하는 흐름입니다."
    if re.search(
        r"방향|암시|부각|이동|전환|확장|확대|구체화|비교\s*기준|평가\s*기준|"
        r"경쟁\s*(축|기준|방식)|운영\s*(방식|구조|책임)|매출\s*(구성|구조)|"
        r"고객\s*(제안|접점)|레퍼런스|대외\s*(매출|고객)",
        value,
    ):
        return value
    base = re.sub(r"(입니다|합니다|한다|다)$", "", value.rstrip(".。"))
    return (
        base
        + "는 유사 사업의 평가 기준이 고객 제안 범위와 검증 조건으로 이동하는 흐름입니다."
    )


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
        "issue_understanding",
        "profile_linkage",
        "skax_response_linkage",
        "claim_strength",
        "grounding_summary",
        "analysis",
        "implication",
        "sentence_grounding",
    }
    assert set(result["issue_understanding"]) == {
        "confirmed_facts",
        "main_actor",
        "peer_role_in_issue",
        "role_confidence",
        "activity_nature",
        "target_business_or_system",
        "customer_or_market_scope",
        "confirmed_numbers_or_dates",
        "uncertain_points",
        "evidence_ids",
    }
    assert set(result["profile_linkage"]) == {
        "peer_company",
        "profile_evidence_available",
        "matched_profile_areas",
        "linkage_level",
        "business_novelty_status",
        "allowed_interpretation_strength",
        "reason",
    }
    assert set(result["skax_response_linkage"]) == {
        "skax_profile_evidence_available",
        "matched_skax_areas",
        "response_mode",
        "response_focus",
        "internal_checkpoints",
        "recommended_focus",
        "monitoring_points",
        "reason",
    }
    assert result["claim_strength"] in {"strong", "moderate", "cautious"}
    assert set(result["grounding_summary"]) == {
        "used_fact_ids",
        "used_profile_refs",
        "ungrounded_claims_removed",
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
    assert set(result["implication"]).issuperset(
        {
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
    )
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
                    "SK AX도 내부적으로 적용 범위와 검증 기준을 함께 점검해야 합니다."
                ),
                "potential_impact": (
                    "통합 이슈에서 적용 범위와 검증 기준이 함께 확인되므로 SK AX는 "
                    "관련 사업의 실행 범위와 확인 기준을 내부 기준으로 비교해야 합니다. "
                    "따라서 SK AX는 계약 범위와 사업성 점검 항목에서 적용 범위와 "
                    "검증 기준을 분리해 정리해야 합니다."
                ),
                "opportunities": [
                    "적용 범위와 검증 기준을 묶은 내부 점검 구조를 만들 수 있습니다."
                ],
                "threats": [
                    "검증 근거가 약하면 유사 사업 비교 단계에서 설명력이 낮아질 수 있습니다."
                ],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 SK AX는 실행 범위와 "
                        "검증 기준을 내부적으로 분리해 점검해야 합니다. 따라서 SK AX는 "
                        "계약 범위와 사업성 점검 항목에서 적용 범위, 운영 책임, 검증 기준을 "
                        "별도 항목으로 정리합니다."
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
    # Business line mapping never invents labels outside supplied SK AX candidates.
    assert set(result["implication"]["skax_implication"]["business_line_mapping"]) <= {
        TEST_LINE_A,
        TEST_LINE_B,
    }
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
    assert "현재 사건 이해" in user_prompt
    assert "피어 프로필 연결 판단" in user_prompt
    assert "SK AX 대응 연결 판단" in user_prompt
    assert "Machine linkage hints" in user_prompt
    assert "Response artifact guidance" in user_prompt
    assert "issue_understanding" in user_prompt
    assert "profile_linkage" in user_prompt
    assert "skax_response_linkage" in user_prompt
    for principle in ("관계 수준", "Machine linkage hints", "artifact_generation_mode"):
        assert principle in full_prompt
    assert "frontend_ready" in full_prompt
    assert "business_line_mapping" in user_prompt
    assert "고정 taxonomy가 아니라" in user_prompt
    phase_decisions = result["implication"]["frontend_ready_diagnostics"]["phase_decisions"]
    assert "self_review_skipped" in phase_decisions
    assert "schema_repair_skipped" in phase_decisions
    assert llm.invoke.call_count == 1


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


def test_stock_market_only_signal_is_watch_only_without_llm_call():
    payload = _fixture()
    issue = _minimal_integrated_issue_with_fact(
        company_id="peer_company",
        fact_id="market_f1",
        fact="주식 초고수들은 장중 해당 종목을 순매수했고 주가는 전 거래일 대비 상승했다.",
    )
    llm = MagicMock()

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=issue,
        classification={"event_type": "market_reaction"},
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is False
    diagnostics = result["implication"]["frontend_ready_diagnostics"]
    assert diagnostics["watch_only"] is True
    assert diagnostics["decision_type"] == "watch_only_stock_market_signal"
    assert "llm_skipped" in diagnostics["phase_decisions"]
    assert llm.invoke.call_count == 0


def test_weak_hiring_signal_is_watch_only_without_llm_call():
    payload = _fixture()
    issue = _minimal_integrated_issue_with_fact(
        company_id="peer_company",
        fact_id="hiring_f1",
        fact="주요 그룹은 16개 계열사와 함께 상반기 신입 공채를 진행한다고 밝혔다.",
    )
    issue["cluster_event_type"] = "personnel"
    llm = MagicMock()

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=issue,
        classification={"event_type": "personnel"},
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is False
    diagnostics = result["implication"]["frontend_ready_diagnostics"]
    assert diagnostics["watch_only"] is True
    assert diagnostics["decision_type"] == "watch_only_weak_hiring_signal"
    assert "llm_skipped" in diagnostics["phase_decisions"]
    assert llm.invoke.call_count == 0


def test_market_infra_signal_without_direct_peer_action_is_watch_only():
    payload = _fixture()
    issue = _minimal_integrated_issue_with_fact(
        company_id="lg_cns",
        fact_id="infra_f1",
        fact=(
            "글로벌 AI 반도체 기업 CEO는 한국 기업들이 AI 팩토리와 "
            "데이터센터 인프라를 확장해야 한다고 말했다."
        ),
    )
    issue["fact_summary"].append("국내 주요 그룹과 AI 인프라 구축 협력 가능성도 함께 언급됐다.")
    issue["integrated_text"] = "\n".join(issue["fact_summary"])
    llm = MagicMock()

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=issue,
        classification={"event_type": "industry_trend"},
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is False
    diagnostics = result["implication"]["frontend_ready_diagnostics"]
    assert diagnostics["watch_only"] is True
    assert diagnostics["decision_type"] == "watch_only_industry_signal"
    assert diagnostics["signal_scope"] == "market_infra_signal"
    assert diagnostics["direct_peer_action"] is False
    assert diagnostics["primary_actor_type"] in {"global_vendor", "multi_actor", "unknown"}
    assert "frontend_ready" not in result["implication"]
    assert result["implication"]["industry_frontend_ready"]["display_policy"] == "industry_only"
    assert result["implication"]["industry_frontend_ready"]["signal_scope"] == (
        "market_infra_signal"
    )
    assert result["implication"]["industry_frontend_ready"]["items"]
    assert "llm_skipped" in diagnostics["phase_decisions"]
    assert llm.invoke.call_count == 0


def test_industry_frontend_ready_keeps_only_distinct_strategic_axes():
    issue = _minimal_integrated_issue_with_fact(
        company_id="lg_cns",
        fact_id="infra_axis_f1",
        fact=(
            "글로벌 AI 반도체 기업 CEO는 한국 기업들이 AI 팩토리와 "
            "데이터센터 인프라를 확장해야 한다고 말했다."
        ),
    )
    issue["fact_summary"].extend(
        [
            "국내 주요 그룹과 AI 인프라 구축 협력 가능성도 함께 언급됐다.",
            "AI 팩토리 구축에는 GPU 확보와 데이터센터 운영 조건이 포함된다.",
        ]
    )
    issue["integrated_text"] = "\n".join(issue["fact_summary"])

    frontend_ready = strategic_insight_module._industry_frontend_ready_from_decision(
        integrated_issue=issue,
        skip_decision={
            "signal_scope": "market_infra_signal",
            "primary_actor_type": "global_vendor",
        },
    )

    items = frontend_ready["items"]
    assert len(items) == 1
    sentences = [
        item["key_implication"]["sentence"]
        for item in items
        if isinstance(item.get("key_implication"), dict)
    ]
    evidence_sentences = [
        item["key_implication"]["evidence_sentence"]
        for item in items
        if isinstance(item.get("key_implication"), dict)
    ]
    assert len(sentences) == len(set(sentences))
    assert "AI 팩토리" in evidence_sentences[0] or "AI 인프라" in evidence_sentences[0]


def test_industry_frontend_ready_uses_dynamic_decision_criteria_without_peer_reason():
    issue = _minimal_integrated_issue_with_fact(
        company_id="lg_cns",
        fact_id="industry_dynamic_f1",
        fact=(
            "글로벌 공급사는 제조기업과 스마트팩토리 공동 실증을 논의했고, "
            "기존 MES 연계와 비용 부담을 후속 검토한다고 밝혔다."
        ),
    )
    issue["fact_summary"].append(
        "참여 기업들은 현장 적용 범위와 파트너십 필요성을 추가로 확인할 예정이다."
    )
    issue["integrated_text"] = "\n".join(issue["fact_summary"])

    frontend_ready = strategic_insight_module._industry_frontend_ready_from_decision(
        integrated_issue=issue,
        skip_decision={
            "signal_scope": "industry_signal",
            "primary_actor_type": "multi_actor",
        },
    )

    item = frontend_ready["items"][0]
    action = item["suggested_action"]
    insight = item["key_implication"]
    action_text = f"{action['sentence']} {action['evidence_sentence']}"
    display_text = (
        f"{insight['sentence']} {insight['evidence_sentence']} "
        f"{action['sentence']} {action['evidence_sentence']}"
    )

    assert "피어 직접 실행" not in action_text
    assert "투자 조건, 운영 책임, 고객 적용 가능성" not in action_text
    assert "항목" not in display_text
    assert action["decision_criteria"]
    assert any(
        criterion in action["decision_criteria"]
        for criterion in ("비용 부담", "기존 시스템 접점", "파트너십 필요성", "고객 적용 가능성")
    )
    assert any(
        phrase in action_text
        for phrase in ("시스템", "외부 협력", "파트너", "고객 적용", "투자 부담", "접점")
    )


def test_invalid_market_infra_signal_preserves_industry_signal_diagnostics():
    payload = _fixture()
    issue = _minimal_integrated_issue_with_fact(
        company_id="lg_cns",
        fact_id="infra_invalid_f1",
        fact=(
            "글로벌 AI 반도체 기업 CEO는 국내 여러 그룹과 만나 "
            "AI 팩토리와 데이터센터 인프라 확장 필요성을 강조했다."
        ),
    )
    issue["is_valid_summary"] = False
    llm = MagicMock()

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=issue,
        classification={"event_type": "industry_trend"},
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    diagnostics = result["implication"]["frontend_ready_diagnostics"]
    assert result["is_valid_strategic_insight"] is False
    assert diagnostics["decision_type"] == "watch_only_industry_signal"
    assert diagnostics["signal_scope"] == "market_infra_signal"
    assert diagnostics["direct_peer_action"] is False
    assert result["implication"]["industry_signal"]["signal_scope"] == "market_infra_signal"
    assert "display_label" not in result["implication"]["industry_frontend_ready"]
    assert "invalid_summary_preserved_as_watch_only_signal" in diagnostics["phase_decisions"]
    assert llm.invoke.call_count == 0


def test_market_infra_direct_peer_action_is_not_watch_only():
    issue = _minimal_integrated_issue_with_fact(
        company_id="samsung_sds",
        fact_id="infra_direct_f1",
        fact=(
            "삼성SDS 컨소시엄은 국가 AI 컴퓨팅센터 구축 사업자로 선정되어 "
            "GPU 인프라 구축을 추진한다."
        ),
    )

    decision = strategic_insight_module._strategic_generation_skip_decision(
        integrated_issue=issue,
        classification={"event_type": "selection"},
    )

    assert decision == {}


def test_partnership_diagnostics_extracts_dynamic_execution_slots():
    issue = _minimal_integrated_issue_with_fact(
        company_id="peer_company",
        fact_id="partnership_f1",
        fact=(
            "피어사는 협력사와 차세대 물류센터 시스템 구축 업무협약을 체결하고 "
            "로봇 관제 플랫폼을 적용한다."
        ),
    )
    issue["cluster_event_type"] = "partnership"
    issue["products_or_services"] = ["로봇 관제 플랫폼"]
    issue["target_systems"] = ["차세대 물류센터 시스템"]
    issue["cluster_fact_intelligence"] = {
        "products_or_services": {
            "activity_types": ["협력"],
            "products_or_services": [
                "로봇 관제 플랫폼",
                "피어사는 협력사와 다양한 사업 기회를 모색할 계획이다",
            ],
        },
    }

    diagnostics = strategic_insight_module._issue_execution_slot_diagnostics(issue)

    assert diagnostics["counterparty"]
    assert diagnostics["target_system"]
    assert diagnostics["product_or_service"]
    assert diagnostics["execution_scope"]
    assert "missing_slots" in diagnostics
    assert "activity_types" not in diagnostics["product_or_service"]
    assert "products_or_services" not in diagnostics["product_or_service"]
    assert not any("계획이다" in item for item in diagnostics["product_or_service"])


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
                    "있습니다. 따라서 SK AX는 계약 범위 비교표에서 적용 범위와 운영 책임을 "
                    "분리해 설명해야 합니다."
                ),
                "opportunities": ["운영 기준을 포함한 제안 구성을 만들 수 있습니다."],
                "threats": ["근거가 약한 제안은 고객 비교 단계에서 설득력이 낮아질 수 있습니다."],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 적용 범위와 "
                        "운영 책임을 함께 비교할 수 있습니다. 따라서 SK AX는 계약 범위 "
                        "비교표에서 적용 범위와 운영 책임을 별도 항목으로 설명합니다."
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
    llm.invoke = MagicMock(
        return_value=_fake_llm_response(llm_payload, add_default_frontend_ready=False)
    )
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A, TEST_LINE_B])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    assert set(result["implication"]["skax_implication"]["business_line_mapping"]) <= {
        TEST_LINE_A,
        TEST_LINE_B,
    }


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
                    "있습니다. 따라서 SK AX는 계약 범위 비교표에서 적용 범위와 운영 책임을 "
                    "분리해 설명해야 합니다."
                ),
                "opportunities": ["운영 기준을 포함한 제안 구성을 만들 수 있습니다."],
                "threats": ["근거가 약한 제안은 고객 비교 단계에서 설득력이 낮아질 수 있습니다."],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 적용 범위와 "
                        "운영 책임을 함께 비교할 수 있습니다. 따라서 SK AX는 계약 범위 "
                        "비교표에서 적용 범위와 운영 책임을 별도 항목으로 설명합니다."
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


def test_strategic_insight_agent_replaces_abstract_actions_with_grounded_fallback():
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
                    "따라서 SK AX는 계약 범위 비교표에서 적용 범위와 검증 기준을 "
                    "분리해 설명해야 합니다."
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
    llm.invoke = MagicMock(
        return_value=_fake_llm_response(llm_payload, add_default_frontend_ready=False)
    )
    profile_context = _profile_context_with_business_lines(payload, [TEST_LINE_A])

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    actions = result["implication"]["skax_implication"]["recommended_actions"]
    assert result["is_valid_strategic_insight"] is False
    assert result["implication"]["is_valid_implication"] is False
    assert "제안서에 적용 범위와 검증 기준 강조" in actions
    assert (
        "frontend_ready" not in result["implication"]
        or result["implication"]["frontend_ready"].get("source")
        not in strategic_insight_module._FRONTEND_READY_DISPLAY_SOURCES
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
                    "SK AX도 피어사 신호를 비교할 때 투자 관계 수준과 "
                    "후속 검증 가능성을 내부적으로 분리해 봐야 합니다."
                ),
                "potential_impact": (
                    "지분 취득처럼 실행 계약 이전 단계의 관계는 후속 사업 접점으로 이어질지 "
                    "별도 확인이 필요합니다. 따라서 SK AX는 사업성 점검 항목과 후속 공시 "
                    "확인 목록에서 관계 수준, 후속 확인 기준, 실제 사업 전환 가능성을 "
                    "분리해 정리해야 합니다."
                ),
                "opportunities": [
                    "투자 관계와 후속 전환 기준을 함께 점검하는 내부 검토 구조를 만들 수 있습니다."
                ],
                "threats": [
                    "관계 수준을 구분하지 못하면 실행 계약 근거를 가진 경쟁사와 "
                    "비교될 때 설명력이 낮아질 수 있습니다."
                ],
                "recommended_actions": [
                    "지분 취득 신호 때문에 SK AX는 실행 계약 이전 단계의 관계 수준과 후속 "
                    "확인 가능성을 따로 점검해야 합니다. 따라서 SK AX는 사업성 점검 항목과 "
                    "후속 공시 확인 목록에서 관계 수준, 후속 확인 기준, 실제 사업 전환 가능성을 "
                    "별도 항목으로 정리합니다."
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
            _fake_llm_response(bad_payload, add_default_frontend_ready=False),
            _fake_llm_response(
                {"needs_revision": True, "revised_result": bad_payload},
                add_default_frontend_ready=False,
            ),
            _fake_llm_response(repair_payload),
            _fake_frontend_ready_repair_response(
                key_sentence="지분 2% 취득 결의는 피어사의 고객 접점 확보 신호입니다.",
                key_evidence=(
                    "입력 fact에는 고객사 지분 2% 취득 결의가 직접 제시되어 "
                    "실행 계약이 아니라 투자 관계 수준으로 해석해야 합니다."
                ),
                action_sentence=(
                    "SK AX는 지분 취득 같은 실행 계약 이전 신호를 볼 때 "
                    "관계 수준과 후속 사업 전환 가능성을 나눠 점검해야 합니다."
                ),
                action_evidence=(
                    "이 구분이 있어야 SK AX가 투자 관계를 실행 역량이나 "
                    "확정 수주로 과대해석하지 않고 내부 사업성 기준을 조정할 수 있습니다."
                ),
                key_event_terms=["지분", "취득"],
                key_profile_terms=["투자", "고객"],
                action_event_terms=["지분", "취득"],
                action_skax_terms=["SK AX", "사업성"],
            ),
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
    assert "지분" in output_text and "취득" in output_text
    assert llm.invoke.call_count >= 3


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
    assert llm.invoke.call_count >= 2


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
    assert llm.invoke.call_count >= 2


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
    assert any("외부 고객 제안/확인 문장" in violation for violation in violations)
    assert not any("제안서가 어떤 고객/사업/도입 프로젝트" in violation for violation in violations)


def test_normalize_implication_does_not_auto_generate_frontend_ready():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)

    implication = strategic_insight_module._normalize_implication_block(
        {
            "is_valid_implication": True,
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "피어사는 현재 사건을 기존 사업과 연결합니다.",
                "capability_change": "현재 사건의 적용 대상이 기존 사업 범위와 연결됩니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX도 유사 사업에서 내부 점검이 필요합니다.",
                "potential_impact": "현재 사건 신호가 유사 사업 기준을 바꿀 수 있습니다.",
                "recommended_actions": ["SK AX는 유사 사업 기준을 내부적으로 점검합니다."],
                "business_line_mapping": [TEST_LINE_A],
            },
        },
        integrated_issue=payload["integrated_issue"],
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
        model="test-model",
    )

    assert "frontend_ready" not in implication


def test_frontend_ready_without_anchor_terms_fails_quality_gate():
    payload = _fixture()
    result = {
        "analysis": {
            "is_valid_analysis": True,
            "analysis_summary": "피어사의 사업 신호가 확인됩니다.",
            "strategic_meaning": ["현재 사건은 기존 사업과 연결됩니다."],
            "market_signal": "현재 사건은 유사 사업 비교 기준을 보여줍니다.",
            "impact_reason": "입력 fact_id에 근거합니다.",
            "reason": "입력 근거를 사용했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "peer_implication": {
                "peer_meaning": "피어사는 현재 사건을 기존 사업과 연결합니다.",
                "capability_change": "현재 사건의 적용 대상이 기존 사업 범위와 연결됩니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX도 유사 사업에서 내부 점검이 필요합니다.",
                "potential_impact": "현재 사건 신호가 유사 사업 기준을 바꿀 수 있습니다.",
                "recommended_actions": ["SK AX는 유사 사업 기준을 내부적으로 점검합니다."],
            },
            "frontend_ready": {
                "source": "llm_direct",
                "key_implication": {
                    "source": "llm_direct",
                    "frame": "피어 사업 흐름",
                    "claim_type": "profile_based_signal",
                    "claim_strength": "moderate",
                    "evidence_mode": "profile_based",
                    "unsupported_claims_removed": [],
                    "sentence": "피어사는 현재 사건을 기존 사업과 연결합니다.",
                    "evidence_sentence": "현재 사건의 적용 대상이 기존 사업 범위와 연결됩니다.",
                },
                "suggested_action": {
                    "source": "llm_direct",
                    "frame": "SK AX 내부 점검",
                    "claim_type": "internal_strategy_check",
                    "claim_strength": "cautious",
                    "evidence_mode": "profile_based",
                    "unsupported_claims_removed": [],
                    "sentence": "SK AX는 유사 사업 기준을 내부적으로 점검합니다.",
                    "evidence_sentence": "현재 사건 신호가 유사 사업 기준을 바꿀 수 있습니다.",
                },
            },
        },
    }

    violations = strategic_insight_module._quality_gate_violations(
        result,
        integrated_issue=payload["integrated_issue"],
        profile_context=payload["profile_context"],
    )

    assert not any("event_anchor_terms" in violation for violation in violations)
    assert any("profile_anchor_terms" in violation for violation in violations)
    assert any("skax_anchor_terms" in violation for violation in violations)


def test_event_based_frontend_ready_allows_missing_profile_anchors():
    integrated_issue = {
        "is_valid_summary": True,
        "main_company": "대상기업",
        "headline": "대상기업, 업무 자동화 기능 도입",
        "fact_summary": [
            "대상기업은 에이전틱 AI를 업무 시스템에 적용했다.",
            "에이전틱 AI는 반복 업무 처리와 승인 흐름을 지원한다.",
        ],
        "integrated_text": (
            "대상기업은 에이전틱 AI를 업무 시스템에 적용했다. "
            "에이전틱 AI는 반복 업무 처리와 승인 흐름을 지원한다."
        ),
    }
    result = {
        "implication": {
            "is_valid_implication": True,
            "frontend_ready": {
                "source": "frontend_repair_direct",
                "key_implication": {
                    "source": "frontend_repair_direct",
                    "frame": "사건 기반 신호",
                    "claim_type": "event_based_signal",
                    "claim_strength": "cautious",
                    "evidence_mode": "event_based",
                    "event_anchor_terms": ["에이전틱 AI"],
                    "profile_anchor_terms": [],
                    "unsupported_claims_removed": [],
                    "sentence": "에이전틱 AI 도입은 업무 실행 방식 변화 신호입니다.",
                    "evidence_sentence": (
                        "기사에는 에이전틱 AI가 업무 시스템에 적용된 사실이 제시됩니다."
                    ),
                },
                "suggested_action": {
                    "source": "frontend_repair_direct",
                    "frame": "내부 점검",
                    "claim_type": "internal_strategy_check",
                    "claim_strength": "cautious",
                    "evidence_mode": "generic_monitoring",
                    "event_anchor_terms": ["에이전틱 AI"],
                    "skax_anchor_terms": [],
                    "unsupported_claims_removed": [],
                    "sentence": (
                        "SK AX는 에이전틱 AI 관련 처리 업무 범위와 승인 흐름의 "
                        "검증 기준을 나눠 정리해야 합니다."
                    ),
                    "evidence_sentence": (
                        "반복 업무 처리와 승인 흐름이 함께 제시되어, 유사 업무 자동화 "
                        "흐름도 시스템 연동 범위와 검증 항목을 분리해 판단해야 합니다."
                    ),
                },
            },
        }
    }
    linkage = {
        "peer_linkages": [{"linkage_level": "low"}],
        "skax_linkage": {"linkage_level": "none"},
    }

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation=linkage,
    ) + strategic_insight_module._frontend_ready_claim_violations(
        result,
        integrated_issue=integrated_issue,
        profile_linkage_evaluation=linkage,
    )

    assert violations == []


def _synthetic_integrated_issue(
    *,
    headline: str,
    fact_summary: list[str],
    event_type: str = "general_update",
) -> dict[str, Any]:
    return {
        "is_valid_summary": True,
        "main_company": "대상기업",
        "headline": headline,
        "fact_summary": fact_summary,
        "integrated_text": "\n".join(fact_summary),
        "consolidated_facts": [
            {"fact_id": f"fact:{index}", "fact": fact}
            for index, fact in enumerate(fact_summary, start=1)
        ],
        "classification": {"event_type": event_type},
    }


def _frontend_ready_result(
    *,
    key_sentence: str,
    key_evidence: str,
    action_sentence: str,
    action_evidence: str,
    key_event_terms: list[str] | None = None,
    action_event_terms: list[str] | None = None,
    key_claim_type: str = "event_based_signal",
    key_evidence_mode: str = "event_based",
    action_evidence_mode: str = "generic_monitoring",
) -> dict[str, Any]:
    return {
        "implication": {
            "is_valid_implication": True,
            "frontend_ready": {
                "source": "llm_direct",
                "key_implication": {
                    "source": "llm_direct",
                    "frame": "issue signal",
                    "claim_type": key_claim_type,
                    "claim_strength": "cautious",
                    "evidence_mode": key_evidence_mode,
                    "event_anchor_terms": key_event_terms or [],
                    "profile_anchor_terms": [],
                    "sentence": key_sentence,
                    "evidence_sentence": key_evidence,
                },
                "suggested_action": {
                    "source": "llm_direct",
                    "frame": "internal check",
                    "claim_type": "internal_strategy_check",
                    "claim_strength": "cautious",
                    "evidence_mode": action_evidence_mode,
                    "event_anchor_terms": action_event_terms or [],
                    "skax_anchor_terms": [],
                    "sentence": action_sentence,
                    "evidence_sentence": action_evidence,
                },
            },
        }
    }


def test_event_based_frontend_ready_requires_dynamic_issue_anchor():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 실행형 업무 플랫폼 출시",
        fact_summary=[
            "대상기업은 '워크플로우Z'를 출시했다.",
            "워크플로우Z는 자연어 명령으로 ERP와 문서 시스템의 업무 처리를 지원한다.",
        ],
        event_type="launch",
    )
    result = {
        "implication": {
            "is_valid_implication": True,
            "frontend_ready": {
                "source": "llm_direct",
                "key_implication": {
                    "source": "llm_direct",
                    "frame": "사건 기반 신호",
                    "claim_type": "event_based_signal",
                    "claim_strength": "cautious",
                    "evidence_mode": "event_based",
                    "event_anchor_terms": [],
                    "profile_anchor_terms": [],
                    "sentence": "",
                    "evidence_sentence": "",
                },
                "suggested_action": {
                    "source": "llm_direct",
                    "frame": "내부 점검",
                    "claim_type": "internal_strategy_check",
                    "claim_strength": "cautious",
                    "evidence_mode": "generic_monitoring",
                    "event_anchor_terms": [],
                    "skax_anchor_terms": [],
                    "sentence": "",
                    "evidence_sentence": "",
                },
            },
        }
    }

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("key_implication.sentence" in violation for violation in violations)
    assert any("key_implication.evidence_sentence" in violation for violation in violations)
    assert any("suggested_action.sentence" in violation for violation in violations)
    assert any("suggested_action.evidence_sentence" in violation for violation in violations)


def test_financial_structure_signal_uses_numbers_and_comparison_context():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 거래 구조 지표 변화 공시",
        fact_summary=[
            "대상기업의 내부거래 비중은 47.1%로 전년 52.0%보다 낮아졌다.",
            "같은 업종 비교군의 내부거래 평균은 35.0%로 제시됐다.",
        ],
        event_type="financial_update",
    )
    result = _frontend_ready_result(
        key_claim_type="financial_structure_signal",
        key_sentence="내부거래 변화는 거래 구조를 비교할 신호입니다.",
        key_evidence="내부거래 변화가 제시됐습니다.",
        action_sentence="SK AX는 내부거래 구조를 점검해야 합니다.",
        action_evidence="거래 구조 기준을 확인해야 합니다.",
        key_event_terms=["내부거래"],
        action_event_terms=["내부거래"],
    )

    violations = strategic_insight_module._frontend_ready_specific_anchor_violations(
        result,
        integrated_issue=integrated_issue,
    )

    assert any("key_implication" in violation for violation in violations)
    assert any("suggested_action" in violation for violation in violations)


def test_financial_structure_key_sentence_keeps_numbers_in_evidence_role():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 내부거래 비중 변화 공시",
        fact_summary=[
            "대상기업의 내부거래 비중은 47.1%로 낮아졌다.",
            "비교군의 내부거래 비중은 60%에서 96%대로 제시됐다.",
        ],
        event_type="financial_update",
    )
    result = _frontend_ready_result(
        key_claim_type="financial_structure_signal",
        key_sentence=("내부거래 47.1%와 비교군 60%에서 96%대는 거래 구조 비교 기준입니다."),
        key_evidence=("대상기업의 내부거래 비중 47.1%와 비교군 60%에서 96%대가 함께 제시됐습니다."),
        action_sentence=(
            "SK AX는 내부거래와 외부 거래 매출을 분리해 매출 구성 기준을 비교해야 합니다."
        ),
        action_evidence=(
            "거래 비중 차이가 제시되어 자사 매출 분류와 대외 고객 매출 기준을 "
            "나눠 볼 필요가 있습니다."
        ),
        key_event_terms=["내부거래", "47.1%"],
        action_event_terms=["내부거래", "외부 거래"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("수치·비교군 근거를 반복" in violation for violation in violations)


def test_product_launch_signal_uses_product_function_and_target_system():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 실행형 업무 플랫폼 출시",
        fact_summary=[
            "대상기업은 '워크플로우Z'를 출시했다.",
            "워크플로우Z는 자연어 명령으로 ERP와 문서 시스템의 업무 처리를 지원한다.",
        ],
        event_type="launch",
    )
    result = _frontend_ready_result(
        key_claim_type="workflow_execution_signal",
        key_sentence="워크플로우Z 출시는 업무 플랫폼 적용 범위가 드러난 신호입니다.",
        key_evidence="워크플로우Z가 ERP와 문서 시스템의 업무 처리를 지원한다고 제시됐습니다.",
        action_sentence=(
            "SK AX는 워크플로우Z 같은 실행형 업무 플랫폼의 연동 범위를 점검해야 합니다."
        ),
        action_evidence=(
            "ERP와 문서 시스템이 함께 제시되어 시스템 연동과 처리 업무 기준을 비교해야 합니다."
        ),
        key_event_terms=["워크플로우Z", "ERP"],
        action_event_terms=["워크플로우Z", "ERP"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    ) + strategic_insight_module._frontend_ready_specific_anchor_violations(
        result,
        integrated_issue=integrated_issue,
    )

    assert violations == []


def test_moderate_actionable_signal_with_business_structure_and_service_change_can_display():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 물류 서비스 고도화로 사업 구조 변화",
        fact_summary=[
            "대상기업의 지난해 물류 매출은 7조3864억원으로 전체 매출의 53%를 차지했다.",
            (
                "디지털 물류 서비스 '첼로 스퀘어'에 '에이전틱 AI 공급망'을 도입해 "
                "서비스 고도화에 나서고 있다."
            ),
        ],
        event_type="business_update",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "물류 매출 비중과 디지털 물류 서비스 고도화는 물류 사업 평가가 "
            "매출 규모와 운영형 서비스 변화로 나뉠 수 있음을 보여줍니다."
        ),
        key_evidence=(
            "지난해 물류 매출 7조3864억원과 전체 매출 53%, 첼로 스퀘어의 "
            "에이전틱 AI 공급망 도입이 함께 제시돼 매출 구조와 서비스 고도화가 "
            "동시에 드러납니다."
        ),
        action_sentence=(
            "SK AX는 물류 관련 기회를 볼 때 매출 기여도와 서비스 적용 범위를 "
            "나눠 내부 우선순위를 검토해야 합니다."
        ),
        action_evidence=(
            "첼로 스퀘어에 에이전틱 AI 공급망을 도입한 사실은 단순 구축보다 "
            "운영형 서비스 변화가 매출 구조와 맞물릴 수 있어 고객 제안 범위와 "
            "운영 책임을 구분할 근거가 됩니다."
        ),
        key_event_terms=["물류 매출", "첼로 스퀘어"],
        action_event_terms=["물류", "첼로 스퀘어"],
    )

    assert (
        strategic_insight_module._frontend_ready_actionable_signal_level(integrated_issue)
        == "moderate"
    )
    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    ) + strategic_insight_module._frontend_ready_specific_anchor_violations(
        result,
        integrated_issue=integrated_issue,
    )

    assert violations == []


def test_weak_event_without_service_or_structure_fact_remains_weak_signal():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, AI 웨비나 개최",
        fact_summary=[
            "대상기업은 AI 트렌드 웨비나를 개최했다.",
            "행사에서는 업계 동향과 관심 주제가 소개됐다.",
        ],
        event_type="event",
    )

    assert strategic_insight_module._frontend_ready_actionable_signal_level(integrated_issue) == (
        "weak"
    )


def test_webinar_with_cloud_service_topic_remains_weak_without_execution_fact():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, Cloud Talk 웨비나 시리즈 론칭",
        fact_summary=[
            "대상기업은 클라우드 서비스의 최신 기술과 활용 사례를 조명하는 웨비나를 론칭했다.",
            "1회 차 웨비나는 글로벌 네트워크 운영에 최적화된 클라우드 플랫폼을 주제로 진행된다.",
        ],
        event_type="event",
    )

    assert strategic_insight_module._is_weak_surface_integrated_issue(integrated_issue)
    assert strategic_insight_module._frontend_ready_actionable_signal_level(integrated_issue) == (
        "weak"
    )
    result = _frontend_ready_result(
        key_sentence="클라우드 서비스 경쟁은 활용 사례 설명 방식으로 넓어질 수 있다.",
        key_evidence="웨비나는 클라우드 서비스 최신 기술과 활용 사례를 조명한다고 소개됐다.",
        action_sentence="SK AX는 클라우드 운영 주제별 설명 범위를 점검해야 한다.",
        action_evidence="웨비나 주제가 글로벌 네트워크 운영과 Cloud WAN으로 제시됐다.",
        key_event_terms=["클라우드 서비스", "웨비나"],
        action_event_terms=["Cloud WAN", "웨비나"],
    )
    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("단순 행사/웨비나/홍보성" in violation for violation in violations)


def test_valid_integrated_issue_with_dynamic_anchors_defaults_to_moderate_signal():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 넥서스 리포트 운영 기준 개편",
        fact_summary=[
            "대상기업은 넥서스 리포트와 오로라 태그를 연결해 현장 데이터 분류 기준을 바꿨다.",
            "새 기준은 부서별 요청 흐름과 결과 공유 방식을 나누는 데 쓰인다.",
        ],
        event_type="business_update",
    )

    assert strategic_insight_module._has_integrated_issue_candidate_anchor_signal(
        integrated_issue
    )
    assert strategic_insight_module._frontend_ready_actionable_signal_level(integrated_issue) == (
        "moderate"
    )


def test_invalid_summary_can_still_use_rich_integrated_text_as_candidate_signal():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 업무별 모델 평가 체계 공개",
        fact_summary=["대상기업, 업무별 모델 평가 체계 공개"],
        event_type="service_update",
    )
    integrated_issue["is_valid_summary"] = False
    integrated_issue["integrated_text"] = (
        "대상기업은 서비스에 적합한 모델을 고르기 위해 29가지 평가지표를 개발했다. "
        "약 1,200개 데이터셋으로 지식 추론 능력과 업무 이해력을 평가한다. "
        "업무별 서비스에 적합한 모델 선정 기준을 제공한다."
    )

    assert strategic_insight_module._is_valid_integrated_issue(integrated_issue)
    assert strategic_insight_module._has_integrated_issue_candidate_anchor_signal(
        integrated_issue
    )
    assert strategic_insight_module._frontend_ready_actionable_signal_level(
        integrated_issue
    ) in {"strong", "moderate"}


def test_security_execution_flow_action_axis_can_display_without_customer_contract():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, AI 및 클라우드 보안 운영 협력",
        fact_summary=[
            "대상기업은 AI 기반 취약점 탐지 역량을 확대한다.",
            "고객사 자산의 취약점을 찾아내고 보완 조치까지 지원한다.",
            "관리형 보안 운영 서비스 사업자로서 보안사고 대응 서비스를 제공할 계획이다.",
        ],
        event_type="security_update",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "AI·클라우드 보안 경쟁은 취약점 탐지에서 보완 조치와 사고 대응까지 "
            "이어지는 운영 흐름으로 넓어질 수 있다."
        ),
        key_evidence=(
            "AI 기반 취약점 탐지, 고객사 자산 보완 조치 지원, 관리형 보안 운영 "
            "서비스의 보안사고 대응 계획이 함께 제시됐다."
        ),
        action_sentence=(
            "SK AX는 클라우드 보안 과제에서 탐지, 보완 조치, 보안사고 대응의 책임 구간과 "
            "외부 협력 필요성을 나눠 확인해야 한다."
        ),
        action_evidence=(
            "취약점 탐지와 보완 조치, 보안사고 대응이 한 흐름으로 제시되면 "
            "고객 제안 범위와 운영 책임을 기능별로 구분할 근거가 된다."
        ),
        key_event_terms=["취약점 탐지", "보안사고 대응"],
        action_event_terms=["탐지", "보완 조치", "보안사고 대응"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    ) + strategic_insight_module._frontend_ready_specific_anchor_violations(
        result,
        integrated_issue=integrated_issue,
    )

    assert violations == []


def test_partnership_signal_uses_counterparty_target_system_and_execution_scope():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 현장 자동화 협약 체결",
        fact_summary=[
            "대상기업은 협력대상과 현장 자동화 협약을 체결했다.",
            (
                "협약에는 청라센터에 로봇관제 플랫폼을 적용하고 "
                "운영 데이터를 연계하는 범위가 포함됐다."
            ),
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "현장 자동화 협약은 자동화 경쟁 기준이 단순 장비 도입에서 "
            "협력대상과 청라센터에 적용되는 로봇관제 데이터 연계와 운영 책임으로 "
            "이동할 수 있음을 보여줍니다."
        ),
        key_evidence="청라센터의 로봇관제 플랫폼 적용과 운영 데이터 연계 범위가 제시됐습니다.",
        action_sentence=(
            "SK AX는 유사 현장 자동화 사업에서 로봇관제 플랫폼 확보 방식과 "
            "운영 데이터 책임 범위를 분리해야 합니다."
        ),
        action_evidence=(
            "청라센터에 적용되는 운영 데이터 연계가 함께 제시되어 내부 판단은 "
            "플랫폼 제공 범위와 현장 시스템 연계 책임을 나눠야 합니다."
        ),
        key_event_terms=["협력대상", "청라센터"],
        action_event_terms=["청라센터", "운영 데이터"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    ) + strategic_insight_module._frontend_ready_specific_anchor_violations(
        result,
        integrated_issue=integrated_issue,
    )

    assert violations == []


def test_action_copy_requires_check_target_and_evaluation_basis():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 실행형 업무 플랫폼 출시",
        fact_summary=[
            "대상기업은 '워크플로우Z'를 출시했다.",
            "워크플로우Z는 자연어 명령으로 ERP와 문서 시스템의 업무 처리를 지원한다.",
        ],
        event_type="launch",
    )
    result = _frontend_ready_result(
        key_sentence="워크플로우Z 출시는 업무 플랫폼 적용 범위가 드러난 신호입니다.",
        key_evidence="워크플로우Z가 ERP와 문서 시스템의 업무 처리를 지원한다고 제시됐습니다.",
        action_sentence="SK AX는 워크플로우Z를 점검해야 합니다.",
        action_evidence="후속 검토가 필요합니다.",
        key_event_terms=["워크플로우Z", "ERP"],
        action_event_terms=["워크플로우Z"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("점검 대상과 판단 기준" in violation for violation in violations)


def test_action_copy_requires_internal_judgement_axis():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 클라우드 보안 협력 체결",
        fact_summary=[
            "대상기업은 보안 전문기업과 클라우드 보안 협력을 체결했다.",
            "협력 범위에는 취약점 탐지와 보안 모니터링이 포함됐다.",
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence="클라우드 보안 협력은 보안 기능을 외부 전문기업과 묶은 신호입니다.",
        key_evidence="취약점 탐지와 보안 모니터링이 협력 범위에 포함됐습니다.",
        action_sentence="SK AX는 클라우드 보안 협력의 적용 범위와 기준을 점검해야 합니다.",
        action_evidence="취약점 탐지와 보안 모니터링이 적용 범위의 기준으로 제시됐습니다.",
        key_event_terms=["클라우드 보안", "취약점 탐지"],
        action_event_terms=["클라우드 보안", "취약점 탐지"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("내부 판단 축" in violation for violation in violations)


def test_key_implication_evidence_must_not_use_action_directive_language():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 클라우드 보안 협력 체결",
        fact_summary=[
            "대상기업은 보안 전문기업과 클라우드 보안 협력을 체결했다.",
            "협력 범위에는 취약점 탐지와 보안 모니터링이 포함됐다.",
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence="클라우드 보안 협력은 보안 기능을 외부 전문기업과 묶은 신호입니다.",
        key_evidence=(
            "취약점 탐지와 보안 모니터링이 협력 범위에 포함됐기 때문에 "
            "실행 범위와 협력 역할을 관찰할 필요가 있습니다."
        ),
        action_sentence=(
            "SK AX는 클라우드 보안 과제에서 자체 수행 범위와 외부 협력 기준을 구분해야 합니다."
        ),
        action_evidence=(
            "보안 기능이 협력 범위로 제시되어 운영 책임과 협력 구간을 나눠 볼 기준이 필요합니다."
        ),
        key_event_terms=["클라우드 보안", "취약점 탐지"],
        action_event_terms=["클라우드 보안", "취약점 탐지"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("대응방향성 지시문" in violation for violation in violations)


def test_key_implication_sentence_must_explain_direction_not_only_fact_label():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 현장 자동화 협약 체결",
        fact_summary=[
            "대상기업은 협력대상과 현장 자동화 협약을 체결했다.",
            "협약에는 로봇관제 플랫폼을 적용하고 운영 데이터를 연계하는 범위가 포함됐다.",
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence="현장 자동화 협약은 로봇관제 플랫폼 적용이 공개된 실행 신호입니다.",
        key_evidence=(
            "협약에는 로봇관제 플랫폼 적용과 운영 데이터 연계 범위가 함께 제시되어 "
            "현장 자동화가 장비 도입보다 운영 구조와 연결됩니다."
        ),
        action_sentence=(
            "SK AX는 현장 자동화 사업에서 로봇 적용 업무와 관제 책임 기준을 분리해야 합니다."
        ),
        action_evidence=(
            "로봇관제 플랫폼과 운영 데이터 연계가 함께 제시되어 내부 판단은 "
            "적용 업무, 운영 책임, 플랫폼 확보 방식을 나눠야 합니다."
        ),
        key_event_terms=["현장 자동화", "로봇관제 플랫폼"],
        action_event_terms=["현장 자동화", "로봇관제 플랫폼"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("현재 사실을 라벨링" in violation for violation in violations)


def test_action_copy_must_not_use_customer_scale_as_direct_response_basis():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 스마트물류 협약 체결",
        fact_summary=[
            "대상기업은 물류기업과 스마트물류 협약을 체결했다.",
            "학습 플랫폼과 통합 관제 플랫폼을 활용해 물류센터 로봇 운영을 추진한다.",
            "물류기업은 전 세계 380여 개 거점 네트워크를 보유하고 있다.",
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "스마트물류 협약은 로봇 도입 경쟁이 학습·관제 운영 구조로 "
            "확장될 수 있음을 보여줍니다."
        ),
        key_evidence=(
            "학습 플랫폼과 통합 관제 플랫폼이 물류센터 로봇 운영에 함께 쓰인다는 "
            "사실이 제시됐습니다."
        ),
        action_sentence=(
            "SK AX는 유사 물류 고객 대응에서 고객 거점 범위를 기준으로 "
            "내부 판단 항목을 나눠야 합니다."
        ),
        action_evidence=(
            "전 세계 380여 개 거점 네트워크가 제시되어 고객 거점 범위를 "
            "직접 비교 기준으로 삼아야 합니다."
        ),
        key_event_terms=["스마트물류", "통합 관제 플랫폼"],
        action_event_terms=["스마트물류", "통합 관제 플랫폼"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("고객 규모·거점 수·시장 규모" in violation for violation in violations)


def test_key_implication_requires_business_interpretation_when_business_context_exists():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 운영 플랫폼 협약 체결",
        fact_summary=[
            "대상기업은 고객사와 운영 플랫폼 협약을 체결했다.",
            "협약에는 고객 업무 시스템 적용과 운영 데이터 연계 범위가 포함됐다.",
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence="운영 플랫폼 협약은 경쟁 기준이 더 구체화되는 흐름입니다.",
        key_evidence=(
            "고객 업무 시스템 적용과 운영 데이터 연계 범위가 협약에 포함됐습니다."
        ),
        action_sentence=(
            "SK AX는 유사 운영 플랫폼 사업에서 시스템 연계 범위와 운영 책임을 "
            "고객 제안 단위로 나눠야 합니다."
        ),
        action_evidence=(
            "이 구분이 있어야 SK AX가 직접 책임질 운영 범위와 외부 확인이 필요한 "
            "범위를 설명할 수 있습니다."
        ),
        key_event_terms=["운영 플랫폼", "고객 업무 시스템"],
        action_event_terms=["운영 플랫폼", "고객 업무 시스템"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("비즈니스 실익" in violation for violation in violations)


def test_product_launch_business_mechanism_allows_workflow_execution_scope():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 업무 자동화 서비스 출시",
        fact_summary=[
            "대상기업은 업무 자동화 서비스를 출시했다.",
            "서비스는 메일, ERP, 문서, 사내 업무 시스템을 분석하고 필요한 업무를 처리한다.",
        ],
        event_type="tech_release",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "업무 자동화 서비스 출시는 경쟁 기준이 단순 응답보다 "
            "업무 시스템 처리 범위로 이동할 수 있음을 보여줍니다."
        ),
        key_evidence=(
            "서비스가 메일, ERP, 문서, 사내 업무 시스템을 분석하고 "
            "필요한 업무를 처리한다고 제시됐습니다."
        ),
        action_sentence=(
            "SK AX는 유사 업무 자동화 수요에서 적용 업무, 기존 시스템 접점, "
            "외부 연계 필요성을 고객 수요 검증 기준으로 나눠 봐야 합니다."
        ),
        action_evidence=(
            "이 구분이 있어야 SK AX가 직접 도입을 전제하지 않고 "
            "업무 처리 범위와 시스템 연계 가능성을 설명할 수 있습니다."
        ),
        key_event_terms=["업무 자동화", "사내 업무 시스템"],
        action_event_terms=["업무 자동화", "사내 업무 시스템"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {
                "linkage_level": "low",
                "matched_terms": ["업무", "자동화"],
                "matched_business_areas": [],
            },
        },
    )

    assert not any("비즈니스 실익" in violation for violation in violations)


def test_frontend_ready_rejects_unsupported_business_jargon():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 업무 플랫폼 출시",
        fact_summary=[
            "대상기업은 업무 플랫폼을 출시했다.",
            "업무 플랫폼은 문서 시스템 처리 범위를 지원한다.",
        ],
        event_type="launch",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "업무 플랫폼 출시는 고객 제안 범위가 플랫폼 주도권으로 확장되는 "
            "흐름입니다."
        ),
        key_evidence="문서 시스템 처리 범위를 지원하는 업무 플랫폼 출시가 제시됐습니다.",
        action_sentence=(
            "SK AX는 업무 플랫폼 연동 범위와 처리 업무 기준을 고객 제안 단위로 나눠야 합니다."
        ),
        action_evidence=(
            "이 구분이 있어야 SK AX가 직접 책임질 시스템 연동 범위와 외부 확인 범위를 "
            "설명할 수 있습니다."
        ),
        key_event_terms=["업무 플랫폼", "문서 시스템"],
        action_event_terms=["업무 플랫폼", "문서 시스템"],
    )
    result["implication"]["frontend_ready"]["key_implication"][
        "sentence"
    ] += " 밸류에이션 개선 신호로도 볼 수 있습니다."

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("전문 해석 용어" in violation for violation in violations)


def test_action_copy_must_not_use_peer_product_name_as_skax_basis_without_profile_match():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 스마트물류 협약 체결",
        fact_summary=[
            "대상기업은 물류기업과 스마트물류 협약을 체결했다.",
            (
                "학습에는 자체 로봇 학습 플랫폼 ‘피지컬웍스 포지(PhysicalWorks Forge)’를, "
                "운영에는 통합 관제 플랫폼 ‘피지컬웍스 바통(PhysicalWorks Baton)’을 활용한다."
            ),
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "스마트물류 협약은 로봇 자동화 경쟁 기준이 학습·관제 운영 구조로 "
            "확장될 수 있음을 보여줍니다."
        ),
        key_evidence=(
            "피지컬웍스 포지(PhysicalWorks Forge)는 로봇 학습에, "
            "피지컬웍스 바통(PhysicalWorks Baton)은 통합 관제에 활용된다고 제시됐습니다."
        ),
        action_sentence=(
            "SK AX는 피지컬웍스 포지(PhysicalWorks Forge)와 "
            "피지컬웍스 바통(PhysicalWorks Baton)을 내부 비교 기준으로 삼아야 합니다."
        ),
        action_evidence=(
            "피어 제품명이 제시됐기 때문에 SK AX 내부 판단도 같은 제품 구조를 기준으로 해야 합니다."
        ),
        key_event_terms=["스마트물류", "피지컬웍스 포지"],
        action_event_terms=["스마트물류", "피지컬웍스 포지"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {
                "linkage_level": "low",
                "matched_terms": ["물류", "자동화"],
                "matched_business_areas": [],
            },
        },
    )

    assert any("피어사 고유 제품명" in violation for violation in violations)


def test_adjacent_skax_linkage_must_not_write_direct_adoption_action():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 스마트물류 협약 체결",
        fact_summary=[
            "대상기업은 물류기업과 스마트물류 협약을 체결했다.",
            "협약에는 로봇 적용 업무와 통합 관제 운영 방식이 포함됐다.",
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "스마트물류 협약은 물류 자동화 경쟁 기준이 로봇 적용 업무와 "
            "통합 관제 운영 방식으로 확장될 수 있음을 보여줍니다."
        ),
        key_evidence="로봇 적용 업무와 통합 관제 운영 방식이 협약 범위에 포함됐습니다.",
        action_sentence=(
            "SK AX는 스마트물류 플랫폼을 자체 도입해야 하고 통합 관제 운영을 직접 구축해야 합니다."
        ),
        action_evidence="물류 자동화 접점이 있으므로 직접 도입과 구축이 필요합니다.",
        key_event_terms=["스마트물류", "통합 관제"],
        action_event_terms=["스마트물류", "통합 관제"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {
                "linkage_level": "low",
                "matched_terms": ["물류", "자동화"],
                "matched_business_areas": [],
            },
        },
    )

    assert any("인접 접점 수준" in violation for violation in violations)


def test_adjacent_skax_linkage_allows_probe_language_with_execution_context():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 업무 자동화 서비스 출시",
        fact_summary=[
            "대상기업은 기업 업무 자동화 서비스를 출시했다.",
            "서비스는 메일, ERP, 문서, 사내 업무 시스템을 분석하고 필요한 업무를 처리한다.",
        ],
        event_type="tech_release",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "업무 자동화 서비스 출시는 기업 AI 경쟁 기준이 단일 챗봇보다 "
            "업무 시스템 처리 범위로 넓어질 수 있음을 보여줍니다."
        ),
        key_evidence=(
            "서비스가 메일, ERP, 문서, 사내 업무 시스템을 분석하고 "
            "필요한 업무를 처리한다고 제시됐습니다."
        ),
        action_sentence=(
            "SK AX는 유사 업무 자동화 구축 수요가 나올 때 적용 업무, 기존 시스템 접점, "
            "외부 연계 필요성을 고객 수요 검증 항목으로 나눠 봐야 합니다."
        ),
        action_evidence=(
            "인접 접점 수준에서는 직접 도입을 전제하기보다 업무 처리 범위와 "
            "시스템 연계 기준이 확인되어야 고객 제안 가능성을 설명할 수 있습니다."
        ),
        key_event_terms=["업무 자동화", "사내 업무 시스템"],
        action_event_terms=["업무 자동화", "사내 업무 시스템"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {
                "linkage_level": "low",
                "matched_terms": ["업무", "자동화"],
                "matched_business_areas": [],
            },
        },
    )

    assert not any("인접 접점 수준" in violation for violation in violations)


def test_adjacent_skax_linkage_must_not_use_strong_packaging_action():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 운영 플랫폼 협약 체결",
        fact_summary=[
            "대상기업은 고객사와 운영 플랫폼 협약을 체결했다.",
            "협약에는 고객 업무 시스템 적용과 운영 데이터 연계 범위가 포함됐다.",
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence=(
            "운영 플랫폼 협약은 고객 업무 시스템과 운영 데이터가 고객 제안 범위로 "
            "확장될 수 있음을 보여줍니다."
        ),
        key_evidence="고객 업무 시스템 적용과 운영 데이터 연계 범위가 협약에 포함됐습니다.",
        action_sentence=(
            "SK AX는 운영 플랫폼 연계 범위를 고객 제안 단위로 패키징하고 "
            "대외 레퍼런스를 확보해야 합니다."
        ),
        action_evidence=(
            "이 선택지가 있어야 SK AX가 직접 책임질 운영 범위와 외부 협력 범위를 "
            "설명할 수 있습니다."
        ),
        key_event_terms=["운영 플랫폼", "고객 업무 시스템"],
        action_event_terms=["운영 플랫폼", "고객 업무 시스템"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {
                "linkage_level": "low",
                "matched_terms": ["운영", "시스템"],
                "matched_business_areas": [],
            },
        },
    )

    assert any("인접 접점 수준" in violation for violation in violations)


def test_action_copy_must_not_paraphrase_key_implication_noun_phrases():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, AI 보안 협력 체결",
        fact_summary=[
            "대상기업은 AI 보안 전문기업과 클라우드 보안 협력을 체결했다.",
            "이번 협력은 AI 보안 관제와 클라우드 보안 적용 범위를 포함한다.",
        ],
        event_type="partnership",
    )
    result = _frontend_ready_result(
        key_sentence="AI 보안 협력은 클라우드 보안 서비스 구조가 넓어지는 신호입니다.",
        key_evidence="AI 보안 전문기업과의 협력과 클라우드 보안 적용 범위가 함께 제시됐습니다.",
        action_sentence="SK AX는 AI 보안 협력과 클라우드 보안 모니터링 범위를 점검해야 합니다.",
        action_evidence="AI 보안 관제와 클라우드 보안 적용 범위가 내부 점검 기준과 연결됩니다.",
        key_event_terms=["AI 보안", "클라우드 보안"],
        action_event_terms=["AI 보안", "클라우드 보안"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("role_separation" in violation for violation in violations)


def test_action_evidence_must_not_repeat_insight_evidence_without_internal_axis():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 거래 구조 지표 변화 공시",
        fact_summary=[
            "대상기업의 내부거래 비중은 47.1%로 낮아졌다.",
            "비교군의 내부거래 비중은 60%에서 96%대로 제시됐다.",
        ],
        event_type="financial_update",
    )
    result = _frontend_ready_result(
        key_claim_type="financial_structure_signal",
        key_sentence="내부거래 47.1%는 거래 의존도 차이가 비교 지표가 된 신호입니다.",
        key_evidence="내부거래 비중 47.1%와 비교군 60%에서 96%대가 함께 제시됐습니다.",
        action_sentence=(
            "SK AX는 내부거래 47.1%와 비교군 60%에서 96%대를 기준으로 "
            "내부거래 비중을 점검해야 합니다."
        ),
        action_evidence=("내부거래 비중 47.1%와 비교군 60%에서 96%대가 함께 제시됐습니다."),
        key_event_terms=["내부거래", "47.1%"],
        action_event_terms=["내부거래", "47.1%"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("role_separation" in violation for violation in violations)


def test_action_copy_requires_basis_in_display_sentence_not_only_evidence():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 실행형 업무 플랫폼 출시",
        fact_summary=[
            "대상기업은 '워크플로우Z'를 출시했다.",
            "워크플로우Z는 자연어 명령으로 ERP와 문서 시스템의 업무 처리를 지원한다.",
        ],
        event_type="launch",
    )
    result = _frontend_ready_result(
        key_sentence="워크플로우Z 출시는 업무 플랫폼 적용 범위가 드러난 신호입니다.",
        key_evidence="워크플로우Z가 ERP와 문서 시스템의 업무 처리를 지원한다고 제시됐습니다.",
        action_sentence="SK AX는 워크플로우Z를 점검해야 합니다.",
        action_evidence="워크플로우Z는 ERP 연동 범위와 문서 처리 기준을 함께 봐야 합니다.",
        key_event_terms=["워크플로우Z", "ERP"],
        action_event_terms=["워크플로우Z"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("점검 대상과 판단 기준" in violation for violation in violations)


def test_frontend_ready_evidence_sentence_must_not_repeat_summary_only():
    integrated_issue = _synthetic_integrated_issue(
        headline="대상기업, 실행형 업무 플랫폼 출시",
        fact_summary=[
            "대상기업은 '워크플로우Z'를 출시했다.",
            "워크플로우Z는 자연어 명령으로 ERP와 문서 시스템의 업무 처리를 지원한다.",
        ],
        event_type="launch",
    )
    result = _frontend_ready_result(
        key_sentence="워크플로우Z 출시는 업무 플랫폼 적용 범위가 드러난 신호입니다.",
        key_evidence="워크플로우Z는 자연어 명령으로 ERP와 문서 시스템의 업무 처리를 지원한다.",
        action_sentence="SK AX는 워크플로우Z의 ERP 연동 범위를 점검해야 합니다.",
        action_evidence=(
            "ERP와 문서 시스템이 함께 제시되어 시스템 연동과 처리 업무 기준을 비교해야 합니다."
        ),
        key_event_terms=["워크플로우Z", "ERP"],
        action_event_terms=["워크플로우Z", "ERP"],
    )

    violations = strategic_insight_module._frontend_ready_required_violations(
        result,
        integrated_issue=integrated_issue,
        profile_context={},
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "none"},
        },
    )

    assert any("요약 문장을 해석 없이 반복" in violation for violation in violations)


def test_frontend_only_violation_skips_self_review_and_attempts_frontend_ready_repair():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    generated_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_summary": "통합 이슈의 사업 신호가 확인됩니다.",
            "strategic_meaning": ["통합 이슈의 실행 범위가 확인됩니다."],
            "market_signal": "통합 이슈가 유사 사업의 비교 기준을 보여줍니다.",
            "impact_level": "medium",
            "impact_reason": "입력 fact_id에 근거합니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.7,
            "reason": "입력 근거를 사용했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "통합 이슈는 피어사의 사업 신호로 볼 수 있습니다.",
                "capability_change": "통합 이슈의 실행 범위가 비교 기준이 됩니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX도 유사 사업 기준을 확인해야 합니다.",
                "potential_impact": "통합 이슈 신호가 내부 비교 기준과 연결됩니다.",
                "recommended_actions": [
                    (
                        "SK AX는 토큰증권 기능분석 컨설팅과 테스트베드 플랫폼 구축 "
                        "범위를 기준으로 내부 점검 항목을 나눠야 합니다."
                    )
                ],
            },
            "confidence": 0.7,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(
        side_effect=[
            _fake_llm_response(generated_payload, add_default_frontend_ready=False),
            _fake_frontend_ready_repair_response(
                key_sentence=(
                    "토큰증권 기능분석 컨설팅은 테스트베드 플랫폼 구축 "
                    "범위가 함께 확인된 사업 신호입니다."
                ),
                key_evidence=(
                    "입력에는 토큰증권 기능분석 컨설팅과 테스트베드 플랫폼 구축이 함께 제시됩니다."
                ),
                action_sentence=(
                    "SK AX는 토큰증권 유사 사업에서 수행 범위와 검증 기준을 "
                    "나눠 점검해야 합니다."
                ),
                action_evidence=(
                    "두 과제가 함께 제시되어 유사 사업에서 수행 범위와 "
                    "검증 기준을 분리해 판단해야 합니다."
                ),
                key_event_terms=["토큰증권 기능분석 컨설팅", "테스트베드 플랫폼 구축"],
                action_event_terms=["토큰증권 기능분석 컨설팅", "테스트베드 플랫폼 구축"],
            ),
            _fake_llm_response(generated_payload),
        ]
    )

    result = StrategicInsightAgent(llm=llm).generate(
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        input_bundle=payload["input_bundle"],
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    assert result["implication"]["frontend_ready"]["source"] == "frontend_repair_direct"
    phase_decisions = result["implication"]["frontend_ready_diagnostics"]["phase_decisions"]
    assert "self_review_skipped" in phase_decisions
    assert "frontend_ready_repair_attempted" in phase_decisions
    assert llm.invoke.call_count <= 3


def test_displayable_generate_result_skips_self_review_and_schema_repair():
    payload = _fixture()
    company_id, company_name = _fixture_company_identity(payload)
    generated_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_summary": "토큰증권 기능분석 컨설팅과 테스트베드 플랫폼 구축이 확인됩니다.",
            "strategic_meaning": ["두 과제의 실행 범위가 유사 사업 비교 기준이 됩니다."],
            "market_signal": "토큰증권 기능분석 컨설팅과 테스트베드 플랫폼 구축이 함께 제시됩니다.",
            "impact_level": "medium",
            "impact_reason": "입력 fact_id에 근거합니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.7,
            "reason": "입력 근거를 사용했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "peer_implication": {
                "company_id": company_id,
                "company_name_ko": company_name,
                "peer_meaning": "토큰증권 기능분석 컨설팅은 테스트베드 구축과 함께 확인됩니다.",
                "capability_change": "두 과제의 실행 범위가 비교 기준이 됩니다.",
                "sourced_evidence_ids": ["c43682_a43100_f1"],
            },
            "skax_implication": {
                "why_important": "SK AX도 유사 사업 기준을 확인해야 합니다.",
                "potential_impact": "통합 이슈 신호가 내부 비교 기준과 연결됩니다.",
                "recommended_actions": [
                    (
                        "SK AX는 토큰증권 기능분석 컨설팅과 테스트베드 플랫폼 구축의 "
                        "수행 범위와 검증 기준을 나눠 점검해야 합니다."
                    )
                ],
            },
            "confidence": 0.7,
            "evidence_label": "moderate",
            "frontend_ready": {
                "source": "llm_direct",
                "key_implication": {
                    "source": "llm_direct",
                    "frame": "사업 실행 범위",
                    "claim_type": "event_based_signal",
                    "claim_strength": "cautious",
                    "evidence_mode": "event_based",
                    "event_anchor_terms": ["토큰증권 기능분석 컨설팅", "테스트베드 플랫폼 구축"],
                    "profile_anchor_terms": [],
                    "sentence": (
                        "토큰증권 기능분석 컨설팅은 테스트베드 플랫폼 구축 범위가 "
                        "함께 확인된 사업 신호입니다."
                    ),
                    "evidence_sentence": (
                        "입력에는 토큰증권 기능분석 컨설팅과 테스트베드 플랫폼 구축이 "
                        "함께 제시되어 실행 범위가 비교 기준이 됩니다."
                    ),
                },
                "suggested_action": {
                    "source": "llm_direct",
                    "frame": "내부 기준 점검",
                    "claim_type": "internal_strategy_check",
                    "claim_strength": "cautious",
                    "evidence_mode": "generic_monitoring",
                    "event_anchor_terms": ["토큰증권 기능분석 컨설팅", "테스트베드 플랫폼 구축"],
                    "skax_anchor_terms": [],
                    "sentence": (
                        "SK AX는 토큰증권 기능분석 컨설팅과 테스트베드 플랫폼 구축의 "
                        "수행 범위와 검증 기준을 나눠 점검해야 합니다."
                    ),
                    "evidence_sentence": (
                        "두 과제가 함께 제시되어 유사 사업에서 수행 범위와 "
                        "검증 기준을 분리해 판단해야 합니다."
                    ),
                },
            },
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(generated_payload))

    result = StrategicInsightAgent(llm=llm).generate(
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        input_bundle=payload["input_bundle"],
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    phase_decisions = result["implication"]["frontend_ready_diagnostics"]["phase_decisions"]
    assert result["implication"]["frontend_ready"]["source"] == "llm_direct"
    assert "generate_result_displayable" in phase_decisions
    assert "self_review_skipped" in phase_decisions
    assert "schema_repair_skipped" in phase_decisions
    assert llm.invoke.call_count == 1


def test_strong_frontend_ready_claim_requires_profile_based_linkage():
    payload = _fixture()
    result = {
        "implication": {
            "is_valid_implication": True,
            "frontend_ready": {
                "source": "llm_direct",
                "key_implication": {
                    "source": "llm_direct",
                    "frame": "시장 지위",
                    "claim_type": "market_leadership",
                    "claim_strength": "strong",
                    "evidence_mode": "event_based",
                    "event_anchor_terms": ["에이전틱 AI"],
                    "profile_anchor_terms": [],
                    "unsupported_claims_removed": [],
                    "sentence": "피어사는 에이전틱 AI 시장을 주도하고 있습니다.",
                    "evidence_sentence": "기사에는 에이전틱 AI 서비스 출시 사실만 제시됩니다.",
                },
                "suggested_action": {
                    "source": "llm_direct",
                    "frame": "내부 점검",
                    "claim_type": "internal_strategy_check",
                    "claim_strength": "cautious",
                    "evidence_mode": "generic_monitoring",
                    "event_anchor_terms": ["에이전틱 AI"],
                    "skax_anchor_terms": [],
                    "unsupported_claims_removed": [],
                    "sentence": "SK AX는 에이전틱 AI 관련 내부 점검 기준을 정리해야 합니다.",
                    "evidence_sentence": (
                        "이 기준은 유사 업무 자동화 흐름을 판단하는 데 필요합니다."
                    ),
                },
            },
        }
    }

    violations = strategic_insight_module._frontend_ready_claim_violations(
        result,
        integrated_issue=payload["integrated_issue"],
        profile_linkage_evaluation={
            "peer_linkages": [{"linkage_level": "low"}],
            "skax_linkage": {"linkage_level": "low"},
        },
    )

    assert any("강한 주장 유형" in violation for violation in violations)
    assert any("strong claim" in violation for violation in violations)


def test_profile_linkage_returns_structured_connection_metadata_without_natural_reason():
    evaluation = strategic_insight_module._build_profile_linkage_evaluation(
        integrated_issue={
            "main_company": "lg_cns",
            "mentioned_peer_companies": ["lg_cns"],
            "fact_summary": [
                "LG CNS는 물류센터에서 로봇 학습 플랫폼과 통합 관제 플랫폼을 활용한다."
            ],
            "cluster_fact_intelligence": {
                "products_or_services": ["로봇 학습 플랫폼", "통합 관제 플랫폼"],
                "customers_or_industries": ["물류센터"],
                "activity_types": ["협약"],
            },
        },
        classification={"event_type": "partnership", "sectors": ["물류"]},
        profile_context={
            "peer_profiles": {
                "lg_cns": {
                    "business_areas": [
                        {
                            "business_line": "스마트물류",
                            "name": "로봇 관제",
                            "core_capabilities": ["통합 관제"],
                            "products_or_services": ["로봇 학습 플랫폼"],
                        }
                    ]
                }
            },
            "skax_profile": {},
        },
    )

    linkage = evaluation["peer_linkages"][0]

    assert linkage["connection_reason"] == "structured_issue_terms_match_profile_terms"
    assert linkage["reason"] == "structured_issue_terms_match_profile_terms"
    assert linkage["connection"]["matched_issue_terms"]
    assert linkage["connection"]["matched_profile_terms"]
    assert "연결되어" not in json.dumps(linkage, ensure_ascii=False)


def test_structured_role_mode_does_not_treat_joint_push_as_unclear():
    assert (
        strategic_insight_module._role_mode_from_structured_activity(
            ["공동 추진"],
        )
        == "partnership_governance"
    )
    assert (
        strategic_insight_module._role_mode_from_structured_activity(
            ["사업 추진"],
        )
        == ""
    )


def test_quality_gate_allows_gap_check_but_blocks_gap_expansion_claim():
    assert (
        strategic_insight_module._has_unsupported_pattern(
            "SK AX는 유사 사업에서 경쟁사와의 역량 격차를 점검해야 합니다.",
            r"격차[가를은\s]*(확대|벌어|커|발생|나타)",
            evidence_text="",
        )
        is False
    )
    assert (
        strategic_insight_module._has_unsupported_pattern(
            "경쟁사와의 역량 격차가 확대되고 있습니다.",
            r"격차[가를은\s]*(확대|벌어|커|발생|나타)",
            evidence_text="",
        )
        is True
    )


def test_recommended_action_allows_performance_capacity_validation_criteria():
    violation = strategic_insight_module._recommended_action_quality_violation(
        "SK AX는 유사 사업에서 전환 범위와 성능/용량 검증 기준을 분리해 점검해야 합니다.",
        label="skax_implication.recommended_actions[1]",
        integrated_issue={"integrated_text": "코어뱅킹 웹단말 전환 사업이 확인됐다."},
        profile_context={},
    )

    assert violation == ""
    assert strategic_insight_module._recommended_action_quality_violation(
        "SK AX는 검증된 성능을 강조해야 합니다.",
        label="skax_implication.recommended_actions[1]",
        integrated_issue={"integrated_text": "코어뱅킹 웹단말 전환 사업이 확인됐다."},
        profile_context={},
    )


def test_action_plan_issue_terms_drop_generic_business_terms():
    terms = strategic_insight_module._action_plan_terms_for_keys(
        {
            "current_issue_signals": {
                "products_or_services": ["사업 프로젝트 고객 회사", "코어뱅킹 웹단말 전환"],
                "target_systems": ["금융 업무 시스템"],
            }
        },
        keys=("products_or_services", "target_systems"),
    )

    assert "사업" not in terms
    assert "프로젝트" not in terms
    assert "고객" not in terms
    assert "회사" not in terms
    assert "코어뱅킹" in terms
    assert "웹단말" in terms


def test_business_novelty_overclaim_keeps_factual_confirmation_and_check_terms():
    evaluation = {
        "peer_linkages": [
            {
                "company_id": "company_a",
                "business_novelty_status": "new_or_untracked_business_signal",
            }
        ]
    }

    assert (
        strategic_insight_module._business_novelty_overclaim_violation(
            "이번 계약 체결이 확정된 사실은 후속 확인 기준으로 관리해야 합니다.",
            label="analysis.reason",
            profile_linkage_evaluation=evaluation,
        )
        == ""
    )
    assert (
        strategic_insight_module._business_novelty_overclaim_violation(
            "이번 사업은 피어사의 성과가 입증된 사례입니다.",
            label="analysis.reason",
            profile_linkage_evaluation=evaluation,
        )
        != ""
    )


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
                "why_important": "SK AX 내부에서도 검증 환경과 운영 책임 점검이 중요합니다.",
                "potential_impact": (
                    "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 SK AX는 관련 사업에서 "
                    "컨설팅 범위와 검증 항목을 함께 점검해야 합니다. 따라서 SK AX는 "
                    "계약 범위와 사업성 점검 기준에서 적용 업무 범위와 검증 항목을 "
                    "분리해 정리해야 합니다."
                ),
                "opportunities": [
                    (
                        "계약 범위와 사업성 점검 기준에서 컨설팅 범위와 검증 항목을 "
                        "함께 관리할 수 있습니다."
                    )
                ],
                "threats": [
                    (
                        "테스트베드 구축 레퍼런스 비교에서 검증 근거가 약하면 "
                        "설득력이 낮아질 수 있습니다."
                    )
                ],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 SK AX는 업무 적용 "
                        "범위와 데이터 관리 책임을 따로 점검해야 합니다. 따라서 SK AX는 "
                        "계약 범위 점검 항목에서 두 항목을 별도 항목으로 정리합니다."
                    ),
                    (
                        "테스트베드 구축 신호 때문에 SK AX는 검증 단계와 후속 사업 전환 "
                        "가능성을 함께 모니터링해야 합니다. 따라서 SK AX는 사업성 점검 기준과 "
                        "후속 공시 확인 목록에서 검증 항목과 후속 확인 기준을 함께 관리합니다."
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
            _fake_llm_response(first_payload, add_default_frontend_ready=False),
            _fake_llm_response(review_payload, add_default_frontend_ready=False),
            _fake_llm_response(repair_payload),
            _fake_frontend_ready_repair_response(
                key_sentence=(
                    "토큰증권 컨설팅과 테스트베드 구축은 고객 검증 단계가 "
                    "사업 범위에 포함되는 신호입니다."
                ),
                key_evidence=(
                    "입력 fact에는 토큰증권 컨설팅과 테스트베드 구축이 함께 제시되어 "
                    "기능 설명보다 실행 범위와 검증 기준을 함께 봐야 합니다."
                ),
                action_sentence=(
                    "SK AX는 토큰증권 테스트베드와 유사한 흐름에서 업무 적용 범위와 "
                    "검증 항목을 내부 점검 기준으로 분리해야 합니다."
                ),
                action_evidence=(
                    "이 구분이 있어야 SK AX가 관련 사업에서 컨설팅 범위와 "
                    "검증 단계의 책임을 따로 판단할 수 있습니다."
                ),
                key_event_terms=["토큰증권", "테스트베드"],
                key_profile_terms=["컨설팅", "검증"],
                action_event_terms=["토큰증권", "테스트베드"],
                action_skax_terms=["SK AX", "검증"],
            ),
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
    assert llm.invoke.call_count >= 3


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
            _fake_llm_response(bad_payload, add_default_frontend_ready=False),
            _fake_llm_response(
                {"needs_revision": True, "revised_result": bad_payload},
                add_default_frontend_ready=False,
            ),
            _fake_llm_response(bad_payload, add_default_frontend_ready=False),
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
    output_text = json.dumps(result, ensure_ascii=False)
    assert "시장 점유율 확대" not in output_text
    assert "기술적 역량을 입증" not in output_text


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
    output_text = json.dumps(result, ensure_ascii=False)
    assert "시장 선점" not in output_text
    assert llm.invoke.call_count >= 2


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
                    "있습니다. 따라서 SK AX는 계약 범위 비교표와 사업성 점검표에서 "
                    "데이터 보관 위치와 운영 책임 범위를 분리해 설명해야 합니다."
                ),
                "opportunities": ["운영 책임 기준을 포함한 제안 구성"],
                "threats": ["근거 없는 기능 중심 메시지의 설득력 약화"],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 데이터 보관 "
                        "위치와 운영 책임 범위를 따로 비교할 수 있습니다. 따라서 SK AX는 "
                        "계약 범위 비교표와 사업성 점검표에 두 판단 기준을 분리해 설명합니다."
                    ),
                ],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": ["실제 고객 적용 범위는 어디까지인가?"],
            "watch_points": ["검증 기준이 고객 확인 기준에 반영되는지"],
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
    assert set(result["implication"]["skax_implication"]["business_line_mapping"]) <= {TEST_LINE_A}
    first_messages = llm.invoke.call_args_list[0].args[0]
    first_user_prompt = first_messages[1]["content"]
    assert "## StrategicEvidencePack" in first_user_prompt
    assert "## ProfileContext" in first_user_prompt
    assert "fact_basis" in first_user_prompt
    assert "시사점" in first_user_prompt
    assert "SK AX 대응방향 작성" in first_user_prompt


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
                    "비교할 수 있습니다. 따라서 SK AX는 계약 범위 비교표와 사업성 "
                    "점검표에서 운영 책임과 검증 항목을 분리해 설명해야 합니다."
                ),
                "opportunities": ["검증 기준을 포함한 제안 구성"],
                "threats": ["근거 없는 기능 중심 메시지의 설득력 약화"],
                "recommended_actions": [
                    (
                        "토큰증권 컨설팅과 테스트베드 구축 신호 때문에 고객은 데이터 보관 "
                        "위치와 운영 책임 범위를 따로 비교할 수 있습니다. 따라서 SK AX는 "
                        "계약 범위 비교표와 사업성 점검표에 두 판단 기준을 분리해 설명합니다."
                    ),
                ],
                "business_line_mapping": [TEST_LINE_A],
            },
            "follow_up_questions": ["고객 적용 범위는 어디까지인가?"],
            "watch_points": ["검증 기준이 고객 확인 기준에 반영되는지"],
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
    assert result["analysis"]["analysis_summary"]


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


def test_structured_role_mode_prefers_counterparty_slot_over_broad_verbs():
    issue = {
        "main_company": "lg_cns",
        "cluster_event_type": "contract",
        "integrated_text": "인스웨이브가 LG CNS와 웹단말 전환 사업 계약을 체결했다.",
        "cluster_fact_intelligence": {
            "activity_types": ["contract"],
            "customers_or_industries": ["LG CNS"],
            "products_or_services": ["코어뱅킹 웹단말 전환"],
        },
        "consolidated_facts": [
            {
                "fact_id": "fact:contract",
                "fact": "인스웨이브가 LG CNS와 웹단말 전환 사업 계약을 체결했다.",
                "activity_types": ["contract"],
                "customers_or_industries": ["LG CNS"],
                "products_or_services": ["코어뱅킹 웹단말 전환"],
            }
        ],
    }

    assert (
        strategic_insight_module._peer_role_mode_for_linkage(issue, classification={})
        == "counterparty_or_customer"
    )


def test_profile_linkage_uses_matched_ai_infra_profile_for_selected_operator():
    issue = {
        "main_company": "samsung_sds",
        "cluster_event_type": "selection",
        "integrated_text": "삼성SDS 컨소시엄이 국가 AI컴퓨팅센터 사업자로 최종 선정됐다.",
        "cluster_fact_intelligence": {
            "activity_types": ["selection", "consortium"],
            "products_or_services": ["GPU", "AI 컴퓨팅센터", "데이터센터"],
            "customers_or_industries": ["공공 AI 인프라"],
        },
    }
    profile_context = {
        "peer_profiles": {
            "samsung_sds": {
                "company_id": "samsung_sds",
                "business_areas": [
                    {
                        "name": "AI 인프라",
                        "summary": "GPU 서버와 데이터센터 운영 역량",
                        "core_capabilities": ["GPU", "데이터센터", "AI 인프라"],
                    }
                ],
            }
        },
        "skax_profile": {},
    }

    evaluation = strategic_insight_module._build_profile_linkage_evaluation(
        integrated_issue=issue,
        classification={"event_type": "selection", "sectors": ["AI 인프라"]},
        profile_context=profile_context,
    )
    peer_linkage = evaluation["peer_linkages"][0]

    assert evaluation["peer_role_mode"] == "selected_operator_or_builder"
    assert peer_linkage["linkage_level"] in {"high", "medium"}
    assert peer_linkage["implication_mode"] == "profile_based"


def test_unmatched_profile_is_observation_not_confirmed_new_business():
    issue = {
        "main_company": "company_a",
        "cluster_event_type": "selection",
        "integrated_text": "A사가 신사업 플랫폼 구축 사업자로 최종 선정됐다.",
        "cluster_fact_intelligence": {
            "activity_types": ["selection"],
            "products_or_services": ["신사업 플랫폼"],
        },
    }
    profile_context = {
        "peer_profiles": {
            "company_a": {
                "company_id": "company_a",
                "business_areas": [{"name": "기존 유지보수", "core_capabilities": ["운영"]}],
            }
        },
        "skax_profile": {},
    }

    evaluation = strategic_insight_module._build_profile_linkage_evaluation(
        integrated_issue=issue,
        classification={"event_type": "selection"},
        profile_context=profile_context,
    )
    peer_linkage = evaluation["peer_linkages"][0]

    assert peer_linkage["linkage_level"] in {"low", "none"}
    assert peer_linkage["business_novelty_status"] == "new_or_untracked_business_signal"
    assert peer_linkage["implication_mode"] == "new_business_signal"


def test_profile_linkage_terms_keep_business_words_and_drop_source_noise():
    context = {
        "structured_terms": ["코어뱅킹 웹단말 전환", "공급계약"],
        "products_or_services": ["코어뱅킹 웹단말 전환"],
        "source_noise_terms": {"itdaily", "seungyang"},
        "evidence_text": "코어뱅킹 웹단말 전환 공급계약 코어뱅킹 웹단말 전환",
    }

    ranked = strategic_insight_module._extract_ranked_terms(
        "itdaily seungyang 코어뱅킹 전환 시스템 플랫폼 공급계약",
        "issue_fact",
        context,
    )
    weights = {item["normalized"]: item["weight"] for item in ranked}

    assert weights["itdaily"] == 0.0
    assert weights["seungyang"] == 0.0
    assert weights["코어뱅킹"] > 0.0
    assert weights["전환"] > 0.0
    assert weights["시스템"] > 0.0
    assert weights["플랫폼"] > 0.0
    assert weights["공급계약"] > 0.0


def test_profile_linkage_does_not_match_on_generic_business_terms_only():
    issue = {
        "main_company": "company_a",
        "cluster_event_type": "general_update",
        "integrated_text": "A사는 고객 대상 사업 프로젝트를 진행했다.",
        "cluster_fact_intelligence": {
            "activity_types": ["general_update"],
            "products_or_services": ["사업 프로젝트"],
        },
    }
    profile_context = {
        "peer_profiles": {
            "company_a": {
                "company_id": "company_a",
                "business_areas": [
                    {
                        "name": "일반 사업",
                        "summary": "고객 대상 프로젝트와 서비스 운영",
                        "core_capabilities": ["서비스", "운영"],
                    }
                ],
            }
        },
        "skax_profile": {},
    }

    evaluation = strategic_insight_module._build_profile_linkage_evaluation(
        integrated_issue=issue,
        classification={"event_type": "general_update"},
        profile_context=profile_context,
    )

    assert evaluation["peer_linkages"][0]["linkage_level"] in {"low", "none"}


def test_response_artifact_guidance_requires_specific_shape_not_only_proposal():
    issue = {
        "main_company": "lg_cns",
        "cluster_event_type": "contract",
        "integrated_text": "코어뱅킹 현대화 웹단말 전환 사업 계약이 체결됐다.",
        "cluster_fact_intelligence": {
            "activity_types": ["contract"],
            "products_or_services": ["코어뱅킹 현대화", "웹단말 전환"],
        },
    }
    plan = strategic_insight_module._action_artifact_plan_for_prompt(
        integrated_issue=issue,
        classification={"event_type": "contract"},
        profile_linkage_evaluation={},
    )

    assert plan["artifact_generation_mode"] == "llm_dynamic"
    assert "avoid_default_artifacts" not in plan
    assert "코어뱅킹 현대화" in plan["current_issue_signals"]["products_or_services"]
    violation = strategic_insight_module._action_artifact_plan_violation(
        "SK AX는 대응 방향을 마련합니다.",
        label="skax_implication.recommended_actions[1]",
        action_artifact_plan=plan,
    )
    assert violation
    assert not strategic_insight_module._action_artifact_plan_violation(
        (
            "SK AX는 코어뱅킹 웹단말 전환의 적용 범위와 운영 책임을 분리해 정리하고, "
            "전환 리스크와 검증 기준을 내부 점검 기준으로 구성합니다."
        ),
        label="skax_implication.recommended_actions[1]",
        action_artifact_plan=plan,
    )


def test_response_artifact_guidance_exposes_structured_signals_without_fixed_mode():
    issue = {
        "main_company": "company_a",
        "cluster_event_type": "partnership",
        "integrated_text": "A사는 SPC 설립을 위한 주주간계약을 체결했다.",
        "cluster_fact_intelligence": {
            "activity_types": ["spc", "shareholder_agreement"],
            "products_or_services": ["SPC 설립"],
        },
    }
    plan = strategic_insight_module._action_artifact_plan_for_prompt(
        integrated_issue=issue,
        classification={"event_type": "partnership"},
        profile_linkage_evaluation={},
    )

    assert plan["artifact_generation_mode"] == "llm_dynamic"
    assert plan["current_issue_signals"]["activity_types"] == [
        "partnership",
        "spc",
        "shareholder_agreement",
    ]
    assert "SPC 설립" in plan["current_issue_signals"]["products_or_services"]
    assert any("현재 사건" in item for item in plan["artifact_policy"])


def test_response_artifact_guidance_keeps_issue_signals_for_llm_generation():
    issue = {
        "main_company": "samsung_sds",
        "cluster_event_type": "selection",
        "integrated_text": "삼성SDS 컨소시엄은 GPU 기반 AI 컴퓨팅센터 구축 사업자로 선정됐다.",
        "cluster_fact_intelligence": {
            "activity_types": ["selection", "consortium", "spc"],
            "products_or_services": ["GPU", "AI 컴퓨팅센터"],
            "target_systems": ["AI 컴퓨팅 인프라"],
        },
    }
    plan = strategic_insight_module._action_artifact_plan_for_prompt(
        integrated_issue=issue,
        classification={"event_type": "selection"},
        profile_linkage_evaluation={},
    )

    assert plan["artifact_generation_mode"] == "llm_dynamic"
    assert "selection" in plan["current_issue_signals"]["activity_types"]
    assert "consortium" in plan["current_issue_signals"]["activity_types"]
    assert "GPU" in plan["current_issue_signals"]["products_or_services"]
    assert "AI 컴퓨팅 인프라" in plan["current_issue_signals"]["target_systems"]


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
