from __future__ import annotations

from src.analysis import summarizer


def test_format_articles_uses_rule_based_snippets_and_dedupes_repeated_content(
    monkeypatch,
) -> None:
    monkeypatch.setattr(summarizer, "_SNIPPETS_PER_ARTICLE", 3)
    monkeypatch.setattr(summarizer, "_SNIPPET_DEDUP_SIMILARITY", 0.80)

    duplicate_content = "삼성SDS가 생성형 AI 플랫폼 A를 출시했다. 주요 기능은 업무 자동화다."
    articles = [
        {
            "id": 101,
            "title": "삼성SDS, 생성형 AI 플랫폼 A 출시",
            "content": duplicate_content,
            "source_type": "news",
            "importance_score": 0.9,
            "relevance_score": 0.9,
            "is_representative": True,
            "matched_companies": ["samsung_sds"],
        },
        {
            "id": 102,
            "title": "삼성SDS, 생성형 AI 플랫폼 A 출시",
            "content": duplicate_content,
            "source_type": "news",
            "importance_score": 1.0,
            "relevance_score": 1.0,
            "matched_companies": ["samsung_sds"],
        },
        {
            "id": 103,
            "title": "삼성SDS AI 플랫폼, 금융권 PoC 착수",
            "content": "삼성SDS는 플랫폼 A를 금융권 고객 PoC에 적용한다고 밝혔다.",
            "source_type": "news",
            "importance_score": 0.7,
            "relevance_score": 0.7,
            "matched_companies": ["samsung_sds"],
        },
    ]

    text = summarizer._format_articles(
        articles=articles,
        target_companies=["samsung_sds"],
        representative_id=101,
    )

    article_101 = text.split("[1] article_id: 101", maxsplit=1)[1].split(
        "[2] article_id: 102",
        maxsplit=1,
    )[0]
    article_102 = text.split("[2] article_id: 102", maxsplit=1)[1].split(
        "[3] article_id: 103",
        maxsplit=1,
    )[0]
    article_103 = text.split("[3] article_id: 103", maxsplit=1)[1]

    assert "content_policy: 원문 전체 content는 LLM에 넣지 않습니다" in text
    assert "article_role: evidence_snippets" in article_101
    assert "content:" not in article_101
    assert 'evidence_snippets: ["삼성SDS, 생성형 AI 플랫폼 A 출시"]' in article_101
    assert 'evidence_snippets: ["주요 기능은 업무 자동화다"]' in article_102
    assert "article_role: evidence_snippets" in article_103
    assert "삼성SDS는 플랫폼 A를 금융권 고객 PoC에 적용" in article_103


def test_candidate_peer_companies_uses_preprocessing_targets_not_body_mentions() -> None:
    articles = [
        {
            "id": 101,
            "company": ["samsung_sds"],
            "matched_companies": ["samsung_sds"],
            "title": "삼성SDS 주가 강세",
            "content": "기사 본문에는 LG CNS와 현대오토에버도 비교 대상으로 언급됐다.",
        }
    ]

    assert summarizer._candidate_peer_companies(articles) == ["samsung_sds"]
