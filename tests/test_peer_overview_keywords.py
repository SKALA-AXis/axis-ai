from src.services.peer_overview_keywords import normalize_result, validate_result


def test_normalize_result_rebuilds_internal_evidence_line() -> None:
    evidence_pack = {
        "peer": {"id": "samsung_sds", "name": "삼성 SDS"},
        "period": "2026Q1",
        "evidence_hash": "hash",
        "profile_context": {
            "business_areas": [{"name": "클라우드·AI 서비스"}],
            "core_capabilities": ["FabriX", "Brity Copilot"],
        },
        "business_signals": [
            {
                "evidence_id": "signal:101",
                "signal_id": 101,
                "raw_article_id": 501,
                "date": "2026-04-01T00:00:00+09:00",
                "source_name": "삼성SDS 뉴스룸",
                "source_type": "newsroom",
                "title": "삼성SDS, AI 플랫폼 FabriX 적용 확대",
                "summary": (
                    "삼성SDS는 FabriX와 Brity Copilot을 "
                    "기업 업무 자동화에 적용한다고 밝혔다."
                ),
                "url": "https://example.com/news",
            }
        ],
    }
    result = {
        "business_keyword": {
            "label": "기업 업무 자동화",
            "reason": "삼성SDS 뉴스룸에서 기업 업무 자동화 적용 확대가 확인됐다.",
            "reasoning": "signal:101 같은 내부 ID가 아니라 원문 내용으로 판단해야 한다.",
            "confidence": 0.8,
            "evidence_refs": ["signal:101"],
            "source_urls": ["https://example.com/news"],
            "evidence_summary": "raw_article_id 501 근거",
        },
        "technology_keyword": {
            "label": "FabriX",
            "reason": "삼성SDS 뉴스룸에서 FabriX 적용 확대가 확인됐다.",
            "reasoning": "FabriX가 업무 자동화 구현 축으로 제시됐다.",
            "confidence": 0.82,
            "evidence_refs": ["signal:101"],
            "source_urls": ["https://example.com/news"],
            "evidence_summary": (
                "삼성SDS 뉴스룸에서 FabriX와 Brity Copilot을 "
                "기업 업무 자동화에 적용한다고 밝혔다."
            ),
        },
        "top_keyword_evidence": [
            (
                "삼성 SDS 사업 키워드 기준: 기업 업무 자동화. "
                "근거 내용: business_area=ai_ax, signal:101. "
                "판단 이유: raw_article_id 기준으로 선정."
            ),
            (
                "삼성 SDS 기술 키워드 기준: FabriX. 근거 내용: [2026-04-01] "
                "삼성SDS 뉴스룸, 삼성SDS, AI 플랫폼 FabriX 적용 확대 - "
                "FabriX 적용 확대. 판단 이유: 원문에 FabriX가 확인되어 선정."
            ),
        ],
        "analysis_trace": [
            {"step": "근거 확인", "summary": "확인", "reasoning": "확인", "evidence": "확인"},
            {"step": "후보 정제", "summary": "정제", "reasoning": "정제", "evidence": "정제"},
            {"step": "최종 판단", "summary": "판단", "reasoning": "판단", "evidence": "판단"},
        ],
    }

    normalize_result(result, evidence_pack, model="test-model")
    validate_result(result, evidence_pack)

    evidence_text = " ".join(result["top_keyword_evidence"])
    assert "business_area" not in evidence_text
    assert "raw_article_id" not in evidence_text
    assert "signal:101" not in evidence_text
    assert "삼성SDS 뉴스룸" in evidence_text
    assert "FabriX" in evidence_text
    assert "확인됐습니다" in evidence_text
    assert "핵심" in evidence_text
    assert "키워드로 판단했습니다" in evidence_text
    assert "핵심" in evidence_text
    assert "클라우드·AI 서비스" in evidence_text
    assert "FabriX" in evidence_text
    assert "원문 근거에서" not in evidence_text
    assert "원문 내용을 보아" not in evidence_text
    assert "[2026-04-01]" in evidence_text
    assert "T00:00:00" not in evidence_text
    assert "..." not in evidence_text


def test_judgment_reason_does_not_copy_source_sentence() -> None:
    evidence_pack = {
        "peer": {"id": "sk_ax", "name": "SK AX"},
        "period": "2026Q1",
        "evidence_hash": "hash",
        "profile_context": {
            "business_areas": [{"name": "클라우드 기반 AX 전환"}],
            "core_capabilities": ["AI 데이터센터", "클라우드 운영"],
        },
        "business_signals": [
            {
                "evidence_id": "signal:202",
                "signal_id": 202,
                "raw_article_id": 902,
                "date": "2026-05-15T00:00:00+09:00",
                "source_name": "DART",
                "source_type": "dart",
                "title": "분기보고서 (2026.03)",
                "summary": (
                    "데이터센터, 모바일 등 신기술의 확산에 따라 "
                    "반도체 소재 시장은 지속적으로 성장하고 있습니다."
                ),
                "url": "https://dart.fss.or.kr/example",
            }
        ],
    }
    result = {
        "business_keyword": {
            "label": "클라우드 기반 AI 솔루션",
            "reason": (
                "AI와 클라우드 기술을 결합하여 기업의 디지털 전환을 "
                "지원하는 사업을 진행하고 있습니다."
            ),
            "reasoning": "",
            "confidence": 0.8,
            "evidence_refs": ["signal:202"],
            "source_urls": ["https://dart.fss.or.kr/example"],
            "evidence_summary": "",
        },
        "technology_keyword": {
            "label": "AI 데이터센터",
            "reason": "AI 데이터센터 관련 솔루션 방향이 확인됐다.",
            "reasoning": "",
            "confidence": 0.82,
            "evidence_refs": ["signal:202"],
            "source_urls": ["https://dart.fss.or.kr/example"],
            "evidence_summary": "",
        },
        "analysis_trace": [
            {"step": "근거 확인", "summary": "확인", "reasoning": "확인", "evidence": "확인"},
            {"step": "후보 정제", "summary": "정제", "reasoning": "정제", "evidence": "정제"},
            {"step": "최종 판단", "summary": "판단", "reasoning": "판단", "evidence": "판단"},
        ],
    }

    normalize_result(result, evidence_pack, model="test-model")

    business_line = result["top_keyword_evidence"][0]
    judgment_reason = business_line.split("판단 이유:", 1)[1]
    confirmation = business_line.split("근거 내용:", 1)[1].split("판단 이유:", 1)[0]

    assert "[2026-05-15]" in business_line
    assert "T00:00:00" not in business_line
    assert "데이터센터, 모바일 등 신기술의 확산에 따라" not in judgment_reason
    assert "지속적으로 성장하고 있습니다" not in business_line
    assert "진행하고 있습니다 때문에" not in judgment_reason
    assert (
        "AI와 클라우드 기술을 결합해 기업의 디지털 전환을 "
        "지원하는 사업 흐름이 확인됩니다"
    ) in judgment_reason
    assert "SK AX" not in judgment_reason
    assert "기존 사업 축인 클라우드 기반 AX 전환" in judgment_reason
    assert "핵심 사업 키워드로 판단했습니다" in judgment_reason
    assert "반도체 소재 시장 성장 흐름" in confirmation
