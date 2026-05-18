"""크롤러 단위 테스트"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from src.agents.credibility_agent import compute_credibility_score
from src.agents.relevance_agent import (
    _core_company_role_reject_result,
    _metadata_patch_for_relevance,
    _result,
)
from src.crawler.base import CrawlWindow, RawArticle
from src.crawler.backfill_runner import BackfillRunner
from src.crawler.batch_processor import _window_months
from src.crawler.sources.bcg import match_bcg_core_sectors
from src.crawler.sources.naver import (
    article_mentions_target_peer,
    classify_peer_relevance,
    get_search_aliases,
)
from src.crawler.sources.skax_crawler import (
    classify_page_kind,
    extract_internal_links,
    normalize_skax_url,
    parse_skax_page,
    title_from_path,
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


def test_window_months_includes_all_months_crossing_backfill_window():
    window = CrawlWindow(
        start=datetime(2026, 2, 10, tzinfo=timezone.utc),
        end=datetime(2026, 5, 10, tzinfo=timezone.utc),
    )

    assert _window_months(window) == ["2026-02", "2026-03", "2026-04", "2026-05"]


def test_backfill_runner_includes_global_company_keywords():
    runner = BackfillRunner(persist=False, use_state=False, process_after_window="none")

    assert "nvidia" in runner.keywords
    assert "samsung_sds" in runner.keywords


def test_skax_url_normalization_keeps_site_pages_and_excludes_newsroom():
    assert (
        normalize_skax_url("/ax-services/aicc/", "https://www.skax.co.kr/")
        == "https://www.skax.co.kr/ax-services/aicc"
    )
    assert (
        normalize_skax_url("/sitemap", "https://www.skax.co.kr/")
        == "https://www.skax.co.kr/sitemap"
    )
    assert (
        normalize_skax_url("/axgenticwire", "https://www.skax.co.kr/")
        == "https://www.skax.co.kr/axgenticwire"
    )
    assert (
        normalize_skax_url("/services/ai-workforce", "https://www.skax.co.kr/")
        == "https://www.skax.co.kr/services/ai-workforce"
    )
    assert (
        normalize_skax_url("/industries/manufacturing", "https://www.skax.co.kr/")
        == "https://www.skax.co.kr/industries/manufacturing"
    )
    assert (
        normalize_skax_url("/experiences/case-study", "https://www.skax.co.kr/")
        == "https://www.skax.co.kr/experiences/case-study"
    )
    assert (
        normalize_skax_url("/insights/trend", "https://www.skax.co.kr/")
        == "https://www.skax.co.kr/insights/trend"
    )
    assert (
        normalize_skax_url("/manufacturing", "https://www.skax.co.kr/")
        == "https://www.skax.co.kr/manufacturing"
    )
    assert (
        normalize_skax_url("/finance", "https://www.skax.co.kr/")
        == "https://www.skax.co.kr/finance"
    )
    assert normalize_skax_url("/company/news-room/3284", "https://www.skax.co.kr/") is None


def test_skax_page_parser_extracts_sections():
    html = """
    <html>
      <head><title>AIOps Platform - SK AX</title></head>
      <body>
        <header>메뉴</header>
        <main>
          <h1>AIOps Platform</h1>
          <section>
            <h2>Features</h2>
            <p>Agentic AI 서비스 활용</p>
            <p>데이터 의사결정 지원</p>
          </section>
          <section>
            <h2>Highlights</h2>
            <p>통합 운영 환경을 제공합니다.</p>
          </section>
        </main>
      </body>
    </html>
    """

    parsed = parse_skax_page(html, "https://www.skax.co.kr/ax-services/new-paradigm-operation")

    assert parsed["title"] == "AIOps Platform"
    assert parsed["headings"] == ["Features", "Highlights"]
    assert "Agentic AI 서비스 활용" in parsed["content"]
    assert classify_page_kind("https://www.skax.co.kr/ax-services/aicc") == "service"
    assert classify_page_kind("https://www.skax.co.kr/axgenticwire") == "brand_campaign"
    assert classify_page_kind("https://www.skax.co.kr/manufacturing") == "industry"
    assert classify_page_kind("https://www.skax.co.kr/experiences/case-study") == "experience"
    assert classify_page_kind("https://www.skax.co.kr/case-study/usecase") == "experience"
    assert title_from_path("https://www.skax.co.kr/cloud") == "Cloud"


def test_skax_internal_links_are_deduped_and_filtered():
    html = """
    <a href="/company/about">about</a>
    <a href="/company/about#top">about duplicate</a>
    <a href="/company/news-room/3284">news</a>
    <a href="https://example.com/out">out</a>
    """

    assert extract_internal_links(html, "https://www.skax.co.kr/") == [
        "https://www.skax.co.kr/company/about"
    ]


def test_company_site_source_type_has_official_level_credibility():
    assert compute_credibility_score("company_site") == 0.90


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


def test_relevance_rejects_company_listed_only_as_etf_holding():
    result = _core_company_role_reject_result(
        title="한투운용 ACE K휴머노이드로봇산업TOP2+ ETF, 상장 후 수익률 37%",
        content=(
            "이 ETF는 현대차와 로보티즈를 주요 종목으로 편입하고 있으며 "
            "현대오토에버, 현대모비스 등을 포트폴리오에 담았다."
        ),
        source_type="news",
        matched_companies=["hyundai_autoever"],
        matched_sectors=["ax"],
    )

    assert result is not None
    assert result["relevance_label"] == "irrelevant"


def test_relevance_keeps_company_as_subject_with_business_action():
    result = _core_company_role_reject_result(
        title="현대오토에버, 로봇 플랫폼 사업 확대",
        content="현대오토에버는 휴머노이드 로봇 플랫폼 개발과 운영을 확대한다고 밝혔다.",
        source_type="news",
        matched_companies=["hyundai_autoever"],
        matched_sectors=["ax"],
    )

    assert result is None


def test_relevance_sectors_are_limited_to_config_ids():
    result = _result(
        label="relevant",
        score=0.8,
        companies=["samsung_sds"],
        sectors=["물류", "ax"],
        reason="sector normalization test",
    )

    assert result["matched_sectors"] == ["ax"]


def test_relevance_unknown_sectors_fall_back_to_other():
    result = _result(
        label="relevant",
        score=0.8,
        companies=["samsung_sds"],
        sectors=["물류"],
        reason="sector normalization test",
    )

    assert result["matched_sectors"] == ["other"]


def test_bcg_core_sector_filter_keeps_sector_report():
    sectors = match_bcg_core_sectors(
        title="Responsible AI Needs More Than Good Intentions",
        description="Learn why GenAI, agentic AI, and regulation should raise the bar.",
        url="https://www.bcg.com/publications/2026/responsible-ai-needs-more-than-good-intentions",
    )

    assert "ax" in sectors


def test_bcg_core_sector_filter_drops_incidental_ai_report():
    sectors = match_bcg_core_sectors(
        title="Fashion CFO Agenda 2026",
        description=(
            "Discover how sustainability issues are hitting fashion P&Ls, "
            "from raw material shocks to margin pressure."
        ),
        url="https://www.bcg.com/publications/2026/fashion-cfo-agenda-sustainability-and-resilience",
    )

    assert sectors == []
