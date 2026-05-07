"""크롤러 단위 테스트"""

import json
from types import SimpleNamespace

from src.agents.relevance_agent import _metadata_patch_for_relevance
from src.crawler.base import RawArticle
from src.crawler.naver_crawler import (
    article_mentions_target_peer,
    classify_peer_relevance,
    get_search_aliases,
)
from src.db.article_store import (
    INDUSTRY_TREND_COMPANY,
    _company_for_storage,
    _metadata_json,
)
from src.pipeline.ingestion_graph import _company_for_context


def test_raw_article_fields():
    article = RawArticle(
        url="https://example.com/news/1",
        title="삼성SDS AI 서비스 확대",
        content="본문 내용",
        published_at=None,
        source_name="naver_news",
        company=["samsung_sds"],
    )
    assert article.url
    assert article.company == ["samsung_sds"]


def test_companyless_trend_report_is_stored_as_industry_trend():
    article = RawArticle(
        url="https://example.com/report/1",
        title="AI 산업 동향 리포트",
        content="AI 산업 동향 본문",
        source_name="BCG",
        source_type="trend_report",
        company=[],
    )

    assert _company_for_storage(article) == [INDUSTRY_TREND_COMPANY]
    metadata = json.loads(_metadata_json(article, _company_for_storage(article)))
    assert metadata["topic_scope"] == "industry_trend"
    assert metadata["company_scope"] == "industry"


def test_companyless_industry_news_is_stored_as_industry_trend():
    article = RawArticle(
        url="https://example.com/news/industry-ai",
        title="AI 산업 정책 변화",
        content="AI 산업 정책 변화와 시장 동향을 다룬 기사",
        source_name="rss_industry",
        source_type="news",
        company=[],
        extra={"sector": "AI"},
    )

    assert _company_for_storage(article) == [INDUSTRY_TREND_COMPANY]


def test_companyless_unmarked_news_is_not_stored():
    article = RawArticle(
        url="https://example.com/news/random",
        title="일반 기사",
        content="산업 마커가 없는 일반 기사 본문",
        source_name="rss",
        source_type="news",
        company=[],
    )

    assert _company_for_storage(article) == []


def test_multi_peer_sector_article_keeps_company_and_gets_industry_topic_scope():
    row = SimpleNamespace(
        company=["samsung_sds", "lg_cns"],
        source_type="news",
    )
    result = {
        "matched_companies": ["samsung_sds", "lg_cns"],
        "matched_sectors": ["ai"],
    }

    assert _metadata_patch_for_relevance(row, result, is_relevant=True) == {
        "topic_scope": "industry_trend",
        "matched_companies": ["samsung_sds", "lg_cns"],
        "matched_sectors": ["ai"],
        "primary_company": None,
    }


def test_company_context_prefers_requested_company_for_multi_peer_article():
    article = {"company": ["lg_cns", "samsung_sds"]}

    assert _company_for_context(article, ["samsung_sds"]) == "samsung_sds"


def test_naver_search_aliases_drop_overbroad_sk_terms():
    aliases = get_search_aliases("sk_ax")

    assert "SK" not in aliases
    assert "SK Inc." not in aliases
    assert "SK주식회사" not in aliases
    assert "SK C&C" in aliases


def test_naver_prefilter_requires_specific_peer_alias():
    article = RawArticle(
        url="https://example.com/news/sk-hynix",
        title="SK하이닉스 AI 반도체 투자 확대",
        content="SK하이닉스가 신규 투자를 발표했다.",
        source_name="naver_news",
        source_type="news",
        company=["sk_ax"],
    )

    assert article_mentions_target_peer(article, "sk_ax") is False


def test_naver_relevance_rejects_overbroad_sk_title_match():
    article = RawArticle(
        url="https://example.com/news/sk-hynix",
        title="SK하이닉스 AI 반도체 투자 확대",
        content="SK하이닉스가 신규 투자를 발표했다.",
        source_name="naver_news",
        source_type="news",
        company=["sk_ax"],
    )

    result = classify_peer_relevance(article, "sk_ax", ["sk_ax"])

    assert result["peer_relevance"] == "reject"
    assert result["_company_peer_ids"] == []
