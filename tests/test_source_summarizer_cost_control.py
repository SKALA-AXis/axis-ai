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


def test_large_cluster_selects_diverse_articles_from_majority_event(monkeypatch) -> None:
    monkeypatch.setattr(summarizer, "_MIN_ANALYZED_ARTICLES", 8)
    monkeypatch.setattr(summarizer, "_MAX_ANALYZED_ARTICLES", 20)
    monkeypatch.setattr(summarizer, "_MAJORITY_THRESHOLD", 0.70)
    monkeypatch.setattr(summarizer, "_MIXED_THRESHOLD", 0.50)

    majority_titles = [
        "LG CNS, 클로드 엔터프라이즈 도입 계약 체결",
        "LG CNS, 클로드 도입 계약으로 그룹 AX 확산",
        "LG CNS, 클로드 엔터프라이즈 전사 활용 계약",
        "LG CNS, 기업용 클로드 계약 체결",
    ]
    articles = [
        {
            "id": article_id,
            "title": majority_titles[article_id % len(majority_titles)],
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "content": "LG CNS가 클로드 엔터프라이즈 도입 계약을 체결했다.",
            "relevance_score": 0.8,
            "importance_score": 0.8,
            "is_representative": article_id == 1,
        }
        for article_id in range(1, 66)
    ]
    articles.extend(
        {
            "id": article_id,
            "title": f"LG CNS, 다른 주제 플랫폼 출시 {article_id}",
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "content": "별도 플랫폼 출시 기사다.",
            "relevance_score": 0.8,
            "importance_score": 0.8,
        }
        for article_id in range(66, 77)
    )

    selection = summarizer._select_analysis_articles(articles=articles, representative_id=1)

    assert selection["status"] == "sampled_majority_group"
    assert len(selection["selected_article_ids"]) == 20
    assert selection["majority_ratio"] >= 0.70
    assert 1 in selection["selected_article_ids"]
    assert set(selection["selected_article_ids"]).issubset(set(selection["majority_article_ids"]))
    assert selection["outlier_article_ids"]


def test_large_mixed_cluster_without_majority_is_blocked(monkeypatch) -> None:
    monkeypatch.setattr(summarizer, "_MIN_ANALYZED_ARTICLES", 8)
    monkeypatch.setattr(summarizer, "_MAX_ANALYZED_ARTICLES", 20)
    monkeypatch.setattr(summarizer, "_MIXED_THRESHOLD", 0.50)

    groups = [
        ("계약 체결", "계약을 체결했다."),
        ("플랫폼 출시", "플랫폼을 출시했다."),
        ("주가 강세", "주가가 강세를 보였다."),
    ]
    articles = []
    article_id = 1
    for title_suffix, content in groups:
        for _ in range(10):
            articles.append(
                {
                    "id": article_id,
                    "title": f"LG CNS, {title_suffix} {article_id}",
                    "company": ["lg_cns"],
                    "matched_companies": ["lg_cns"],
                    "content": content,
                    "relevance_score": 0.8,
                    "importance_score": 0.8,
                }
            )
            article_id += 1

    selection = summarizer._select_analysis_articles(articles=articles, representative_id=1)

    assert selection["status"] == "mixed_cluster_no_majority"
    assert selection["selected_article_ids"] == []
    assert selection["excluded_article_ids"] == list(range(1, 31))
