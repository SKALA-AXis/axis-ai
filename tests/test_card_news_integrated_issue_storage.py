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


def test_card_news_insert_params_leaves_industry_trend_peer_fk_empty():
    params = _card_news_insert_params(
        {
            "id": "CN-20260616-1",
            "company": "industry_trend",
            "cluster_id": 1,
            "title": "산업 동향 카드",
            "summary_lines": ["1", "2", "3"],
            "event_type": "industry_trend",
            "sector": "security",
        }
    )

    assert params["company"] == "industry_trend"
    assert params["peer_company_id"] is None
    assert params["primary_keyword_category"] == "security"


def test_card_news_insert_params_maps_industry_frontend_ready_to_display_frontend():
    params = _card_news_insert_params(
        {
            "id": "CN-20260616-2",
            "company": "industry_trend",
            "cluster_id": 2,
            "title": "산업 동향 카드",
            "summary_lines": ["요약"],
            "event_type": "industry_trend",
            "sector": "security",
            "analysis_package": {
                "implication": {
                    "is_valid_implication": True,
                    "industry_frontend_ready": {
                        "source": "industry_signal_direct",
                        "display_policy": "industry_only",
                        "signal_scope": "industry_trend",
                        "items": [
                            {
                                "key_implication": {
                                    "sentence": (
                                        "공급망 보안은 개별 기업 이슈보다 "
                                        "산업 운영 기준으로 이동하고 있다."
                                    ),
                                    "evidence_sentence": (
                                        "SW 공급망 보안 로드맵 공개 예고와 "
                                        "공급망 공격 증가가 함께 확인됐다."
                                    ),
                                },
                                "suggested_action": {
                                    "sentence": (
                                        "SK AX는 외부 솔루션과 오픈소스 검증 기준을 "
                                        "제안 단계에서 분리해 점검해야 한다."
                                    ),
                                    "evidence_sentence": (
                                        "로드맵 공개 예고가 SBOM과 공급망 검증 기준을 "
                                        "선제 점검할 근거가 된다."
                                    ),
                                },
                            }
                        ],
                    },
                }
            },
        }
    )

    implication = json.loads(params["implication"])
    frontend = implication["frontend"]
    assert frontend["source"] == "industry_signal_direct"
    assert frontend["display_policy"] == "industry_only"
    assert frontend["signal_scope"] == "industry_trend"
    assert frontend["key_implication_items"] == [
        {
            "main": "공급망 보안은 개별 기업 이슈보다 산업 운영 기준으로 이동하고 있다.",
            "detail": "SW 공급망 보안 로드맵 공개 예고와 공급망 공격 증가가 함께 확인됐다.",
        }
    ]
    assert frontend["suggested_action_items"] == [
        {
            "main": (
                "SK AX는 외부 솔루션과 오픈소스 검증 기준을 제안 단계에서 분리해 점검해야 한다."
            ),
            "detail": "로드맵 공개 예고가 SBOM과 공급망 검증 기준을 선제 점검할 근거가 된다.",
        }
    ]
    assert implication["recommended_actions"] == frontend["suggested_actions"]


def test_card_news_insert_params_maps_frontend_ready_to_display_frontend():
    params = _card_news_insert_params(
        {
            "id": "CN-2",
            "company": "samsung_sds",
            "title": "삼성SDS, 플랫폼 구축 수주",
            "summary_lines": ["요약"],
            "source_raw_article_ids": [2],
            "sources": [{"title": "기사", "url": "https://example.com/b"}],
            "analysis_package": {
                "integrated_issue": {"main_issue": "플랫폼 구축 수주"},
                "analysis": {"analysis_summary": "전략 분석"},
                "implication": {
                    "is_valid_implication": True,
                    "skax_implication": {"recommended_actions": ["기존 대응"]},
                    "frontend_ready": {
                        "source": "llm_direct",
                        "key_implication": {
                            "sentence": "공공 업무 플랫폼의 AI 전환 기준이 구체화되고 있다.",
                            "evidence_sentence": (
                                "183억 원 규모 구축 범위와 사용자가 함께 제시됐다."
                            ),
                        },
                        "suggested_action": {
                            "sentence": "SK AX는 공공 업무 플랫폼 제안 범위를 나눠 점검해야 한다.",
                            "evidence_sentence": "구축 범위와 운영 책임이 제안 기준으로 이어진다.",
                        },
                    },
                },
            },
        }
    )

    implication = json.loads(params["implication"])
    frontend = implication["frontend"]
    assert frontend["key_implication_items"] == [
        {
            "main": "공공 업무 플랫폼의 AI 전환 기준이 구체화되고 있다.",
            "detail": "183억 원 규모 구축 범위와 사용자가 함께 제시됐다.",
        }
    ]
    assert frontend["suggested_action_items"] == [
        {
            "main": "SK AX는 공공 업무 플랫폼 제안 범위를 나눠 점검해야 한다.",
            "detail": "구축 범위와 운영 책임이 제안 기준으로 이어진다.",
        }
    ]
    assert implication["recommended_actions"] == frontend["suggested_actions"]


def test_card_news_insert_params_replaces_empty_frontend_from_frontend_ready():
    params = _card_news_insert_params(
        {
            "id": "CN-3",
            "company": "lg_cns",
            "title": "LG CNS, 서비스 출시",
            "summary_lines": ["요약"],
            "source_raw_article_ids": [3],
            "sources": [{"title": "기사", "url": "https://example.com/c"}],
            "implication": {
                "is_valid_implication": True,
                "frontend": {
                    "key_implication_items": [],
                    "suggested_action_items": [],
                    "key_implications": [],
                    "suggested_actions": [],
                },
            },
            "analysis_package": {
                "implication": {
                    "is_valid_implication": True,
                    "frontend_ready": {
                        "source": "llm_direct",
                        "key_implication": {
                            "sentence": "기업 업무 자동화 기준이 실제 처리 범위로 넓어지고 있다.",
                            "evidence_sentence": "업무 시스템과 문서 처리 기능이 함께 제시됐다.",
                        },
                        "suggested_action": {
                            "sentence": (
                                "SK AX는 업무 자동화 제안에서 처리 대상과 "
                                "시스템 접점을 나눠 봐야 한다."
                            ),
                            "evidence_sentence": (
                                "업무 시스템과 문서 처리 기능이 함께 제시돼 "
                                "적용 범위를 구분할 기준이 된다."
                            ),
                        },
                    },
                }
            },
        }
    )

    implication = json.loads(params["implication"])
    frontend = implication["frontend"]
    assert frontend["key_implication_items"]
    assert frontend["suggested_action_items"]
    assert frontend["key_implication_items"][0]["main"] == (
        "기업 업무 자동화 기준이 실제 처리 범위로 넓어지고 있다."
    )


def test_card_news_insert_params_uses_structured_implication_when_frontend_ready_removed():
    params = _card_news_insert_params(
        {
            "id": "CN-4",
            "company": "samsung_sds",
            "title": "삼성SDS, 플랫폼 구축 수주",
            "summary_lines": ["요약"],
            "source_raw_article_ids": [4],
            "sources": [{"title": "기사", "url": "https://example.com/d"}],
            "implication": {
                "is_valid_implication": False,
                "frontend": {
                    "key_implication_items": [],
                    "suggested_action_items": [],
                },
                "peer_implication": {
                    "peer_meaning": "공공 업무 플랫폼 전환 수요가 구체화됐다.",
                    "capability_change": "183억 원 규모 구축사업이 제시됐다.",
                },
                "skax_implication": {
                    "why_important": "SK AX도 공공 업무 플랫폼 적용 범위를 비교할 수 있다.",
                    "recommended_actions": [
                        "SK AX는 공공 업무 플랫폼 적용 범위와 검증 기준을 비교해야 한다."
                    ],
                },
            },
        }
    )

    implication = json.loads(params["implication"])
    frontend = implication["frontend"]
    assert frontend["source"] == "structured_implication_fallback"
    assert frontend["key_implication_items"][0]["main"] == (
        "공공 업무 플랫폼 전환 수요가 구체화됐다."
    )
    assert frontend["suggested_action_items"][0]["main"] == (
        "SK AX는 공공 업무 플랫폼 적용 범위와 검증 기준을 비교해야 한다."
    )


def test_card_news_structured_fallback_uses_issue_fact_for_fragment_detail():
    params = _card_news_insert_params(
        {
            "id": "CN-4B",
            "company": "samsung_sds",
            "title": "삼성SDS, 플랫폼 구축 수주",
            "summary_lines": ["요약"],
            "source_raw_article_ids": [44],
            "sources": [{"title": "기사", "url": "https://example.com/d2"}],
            "analysis_package": {
                "integrated_issue": {
                    "headline": "삼성SDS, 지능형 업무관리 플랫폼 구축사업 수주",
                    "fact_summary": [
                        (
                            "삼성SDS 컨소시엄은 183억 원 규모의 지능형 업무관리 "
                            "플랫폼 구축사업을 수주했다."
                        ),
                        "이 사업은 온-나라 시스템을 AI 기반 협업 환경으로 전환한다.",
                    ],
                },
                "implication": {
                    "peer_implication": {
                        "peer_meaning": (
                            "삼성SDS의 수주는 공공 부문에서 클라우드 기반 AI 협업 환경 "
                            "전환 사업을 수행하는 사례로 관찰된다."
                        ),
                        "capability_change": (
                            "클라우드 전환 및 AI 기반 협업 환경 구축 관련 공공 사업 사례 확보"
                        ),
                    },
                    "skax_implication": {
                        "recommended_actions": [
                            (
                                "SK AX는 유사 공공 플랫폼 전환 범위와 "
                                "클라우드 구성 조건을 점검해야 한다."
                            )
                        ],
                    },
                },
            },
        }
    )

    implication = json.loads(params["implication"])
    frontend = implication["frontend"]
    assert frontend["source"] == "structured_implication_fallback"
    assert frontend["key_implication_items"][0]["detail"] == (
        "삼성SDS 컨소시엄은 183억 원 규모의 지능형 업무관리 플랫폼 구축사업을 수주했다."
    )


def test_card_news_insert_params_merges_structured_implication_from_analysis_package():
    params = _card_news_insert_params(
        {
            "id": "CN-5",
            "company": "samsung_sds",
            "title": "삼성SDS, 플랫폼 구축 수주",
            "summary_lines": ["요약"],
            "source_raw_article_ids": [5],
            "sources": [{"title": "기사", "url": "https://example.com/e"}],
            "implication": {
                "frontend": {"key_implication_items": [], "suggested_action_items": []},
                "key_implications": ["기존 시사점"],
                "suggested_actions": [],
            },
            "analysis_package": {
                "implication": {
                    "peer_implication": {
                        "peer_meaning": "공공 업무 플랫폼 전환 수요가 구체화됐다.",
                        "capability_change": "구축 범위가 제시됐다.",
                    },
                    "skax_implication": {
                        "recommended_actions": [
                            "SK AX는 공공 업무 플랫폼 적용 범위와 검증 기준을 비교해야 한다."
                        ],
                        "why_important": "공공 업무 전환 기준을 비교할 수 있다.",
                    },
                }
            },
        }
    )

    implication = json.loads(params["implication"])
    frontend = implication["frontend"]
    assert implication["skax_implication"]["recommended_actions"]
    assert frontend["suggested_action_items"][0]["main"] == (
        "SK AX는 공공 업무 플랫폼 적용 범위와 검증 기준을 비교해야 한다."
    )


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
