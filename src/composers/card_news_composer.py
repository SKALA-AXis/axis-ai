"""카드뉴스 생성 에이전트.

AnalysisPackage를 사용자에게 보여줄 카드뉴스/API 응답 형태로 재가공한다.
기존 raw cluster 기반 생성 메서드는 호환용으로 유지한다.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from src.analysis.models import AnalysisPackage
from src.config.companies import COMPANY_ALIASES, company_name_ko
from src.config.global_companies import GLOBAL_COMPANY_ALIASES, global_company_name_ko
from src.config.sectors import SECTOR_KEYWORDS, match_sectors
from src.db.article_store import get_articles_by_ids
from src.llm import LLMSpec, build_chat_llm

log = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None
_PROMPT_VERSION = "card-news-v1.0"
_CARD_PROMPT_VERSION = "card-news-v1.0"
_DEFAULT_COVER_IMAGE_URL = "/png.png"
_DEFAULT_COVER_IMAGE_ALT = "카드뉴스 대표 이미지"
_SUMMARY_LINE_MIN = 3
_SUMMARY_LINE_MAX = 5
_CARD_DETAIL_MAX = 3
_DISPLAY_ITEM_MIN = 1
_DISPLAY_ITEM_PREFERRED = 3
_DISPLAY_ITEM_MAX = 5

_FRONTEND_SECTOR_IDS = {
    "security",
    "ax",
    "infra",
    "biz_area",
    "other",
}

_FRONTEND_EVENT_TYPES = {
    "launch",
    "partnership",
    "contract",
    "technology_update",
    "earnings",
    "stock_market",
    "analyst_report",
    "risk",
    "general_update",
    "investment",
    "hiring",
    "organization",
    "ma",
    "personnel",
    "tech",
    "regulation",
    "new_biz",
}

_FACT_BASIS_EVIDENCE_TYPES = {
    "core_fact",
    "unique_fact",
    "common_fact",
    "uncertain_fact",
    "reported_fact",
    "numeric_fact",
    "market_reaction_fact",
    "risk_fact",
}
_SUMMARY_ACTION_TOKENS = {
    "선정",
    "확정",
    "수주",
    "계약",
    "체결",
    "협약",
    "출시",
    "공개",
    "구축",
    "운영",
    "도입",
    "전환",
    "투자",
    "참여",
}

_EVENT_TO_FACT_BASIS_TYPE = {
    "launch": "unique_fact",
    "technology_update": "unique_fact",
    "contract": "core_fact",
    "partnership": "core_fact",
    "investment": "core_fact",
    "hiring": "reported_fact",
    "organization": "reported_fact",
    "regulation": "reported_fact",
    "general_update": "reported_fact",
    "earnings": "numeric_fact",
    "analyst_report": "reported_fact",
    "stock_market": "market_reaction_fact",
    "risk": "risk_fact",
    "unknown": "reported_fact",
}

_ISSUE_CARD_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 기사들을 종합하여 이슈 카드를 작성해주세요.
여러 기사가 있을 경우 교차 검증하여 가장 신뢰도 높은 사실만 포함하세요.

## 기사 정보
{articles_text}

## 작성 규칙
- 제목: 핵심 사실을 담은 한 문장 (40자 이내)
- 요약: 최소 3줄, 최대 5줄. 각 줄은 "1.", "2."처럼 순번으로 시작
- 출처 목록: 사용한 기사의 title, source_name, url 포함

## 이벤트 타입 (하나만 선택, 가장 두드러진 성격 기준)
- partnership: 기업간 협력·MOU·공동사업·파운드리 제공계약
- ma: 인수·합병·지분 인수·투자 유치
- personnel: 채용·인사·임원 선임·조직 개편
- tech: 신기술·신제품·플랫폼 출시·기술 실증
- regulation: 법규·가이드라인·정부 정책
- new_biz: 신사업 진출·수주·실적 발표·시장 확장

복수 해당 시 본문이 가장 크게 다루는 측면을 선택.
애매하면 tech 대신 personnel/new_biz/partnership 우선.

다음 JSON 형식으로만 응답하세요 (추가 텍스트 금지):
{{
  "title": "이슈 제목",
  "summary_lines": ["1. ...", "2. ...", "3. ...", "4. ...", "5. ..."],
  "event_type": "tech",
  "sources": [
    {{"index": 1, "title": "...", "source_name": "...", "url": "...", "credibility_score": 0.0}}
  ]
}}"""


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        model_name = os.getenv("CARD_NEWS_COMPOSER_MODEL", "gpt-4o")
        # env 로 모델 지정 가능 → gpt-5 라도 reasoning_effort 미전달(기존 동작) 위해 None.
        _llm = build_chat_llm(
            LLMSpec(model=model_name, temperature=0.3, max_tokens=1024, reasoning_effort=None)
        )
    return _llm


class CardNewsComposer:
    """뉴스 요약 결과를 우선 사용해 card_news 저장/API 스키마를 생성한다."""

    def generate(
        self,
        summary: dict[str, Any],
        analysis: dict[str, Any] | None = None,
        classification: dict[str, Any] | None = None,
        articles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """요약·분석 결과를 card_news 저장/API 스키마 호환 dict로 변환한다.

        DB 저장은 하지 않는다.
        """
        analysis = analysis or {}
        classification = classification or {}
        articles = articles if articles is not None else _load_source_articles(summary)

        cluster_id = _optional_int(summary.get("cluster_id"))
        peer_id = _normalize_peer_id(summary.get("main_company")) or _normalize_peer_id(
            classification.get("company")
        )
        trust_score = _trust_score(articles)
        source_article_ids = _source_article_ids(summary, articles)
        created_at = _now_iso()
        published_date = _published_date(articles, created_at)

        title = _first_non_empty(
            summary.get("headline"),
            summary.get("one_line_summary"),
            analysis.get("analysis_summary"),
            classification.get("title"),
            (articles[0] or {}).get("title") if articles else "",
            "피어사 주요 뉴스",
        )
        summary_lines = _plain_summary_lines(summary, use_llm=True)
        if not summary_lines:
            summary_lines = _article_title_summary_lines(articles)
        elif len(summary_lines) < _SUMMARY_LINE_MIN:
            summary_lines = _merge_summary_lines(
                summary_lines, _article_title_summary_lines(articles)
            )
        card_text = _card_text(title, summary_lines, summary, articles)
        event_type = _infer_event_type(summary, classification, card_text)
        title = _business_context_title(title, summary=summary, event_type=event_type)
        card_text = _card_text(title, summary_lines, summary, articles)
        sectors = _infer_sectors(classification, card_text)
        sector = sectors[0] if sectors else "other"
        exposure_score = _normalized_importance_score(
            classification=classification,
            card_text=card_text,
        )
        exposure_band = _importance_band(exposure_score)
        signals = _signals(
            classification=classification,
            summary=summary,
            source_article_ids=source_article_ids,
            card_text=card_text,
        )
        sources = _rich_sources(articles)
        media_assets = _media_assets(articles)
        cover_image = media_assets[0]["url"] if media_assets else _DEFAULT_COVER_IMAGE_URL
        validation_pass = _validation_pass(summary, analysis)
        validation_sc_score = _validation_sc_score(summary, analysis, trust_score)
        evidence_chain = _evidence_chain(
            summary=summary,
            analysis=analysis,
            sources=sources,
            source_article_ids=source_article_ids,
            cluster_id=cluster_id,
            created_at=created_at,
        )

        db_record = _db_record(
            card_id=_card_news_id(cluster_id, published_date),
            company=peer_id,
            cluster_id=cluster_id,
            title=title,
            summary_lines=summary_lines,
            event_type=event_type,
            importance=exposure_band,
            importance_score=exposure_score,
            sector=sector,
            sectors=sectors,
            signals=signals,
            evidence_chain=evidence_chain,
            sources=sources,
            validation_pass=validation_pass,
            validation_sc_score=validation_sc_score,
        )

        insights = _list_string(analysis.get("strategic_meaning")) or _list_string(
            summary.get("fact_summary")
        )
        display_sections = _display_sections_from_strategy_result(
            summary=summary,
            analysis=analysis,
            implication={},
            strategic_root={"analysis": analysis},
            sentence_grounding={},
            sources=sources,
            summary_lines=summary_lines,
        )
        return {
            **db_record,
            "peer_id": peer_id,
            "subtitle": _subtitle(analysis, classification),
            "category_label": sector.upper() if sector == "ax" else sector,
            "published_date": published_date,
            "exposure_band": exposure_band,
            "exposure_score": exposure_score,
            "implication": _implication(analysis),
            "sources": sources,
            "trust_score": trust_score,
            "source_count": len(sources),
            "financial_context": None,
            "display_sections": display_sections,
            "slides": _slides_from_display_sections(
                title=title,
                display_sections=display_sections,
                sources=sources,
                media_assets=media_assets,
            )
            or _slides(title, summary_lines, insights, sources, media_assets),
            "display": _display_meta(sector, cover_image),
            "is_human_reviewed": False,
            "is_bookmarked": False,
            "bookmark_count": 0,
            "share_count": 0,
            "created_at": created_at,
            "frontend_implication": _frontend_implication(analysis),
            "validation": {
                "pass": validation_pass,
                "sc_score": validation_sc_score,
            },
            "db_record": db_record,
        }

    def generate_from_analysis_package(
        self,
        analysis_package: AnalysisPackage | dict[str, Any],
        classification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """AnalysisPackage 를 사용자용 카드뉴스/API 스키마로 재가공.

        W2-4: implication 이중 처리 제거. ImplicationAgent v4.0 (또는 v5.0) 의
        is_valid_implication=true 결과가 있으면 그것을 단일 출처로 사용하고,
        실패/누락 시에만 analysis 기반 frontend implication 으로 fallback.
        """
        package = (
            analysis_package.to_dict()
            if isinstance(analysis_package, AnalysisPackage)
            else analysis_package
        )
        input_bundle = package.get("input_bundle") or {}
        validation = package.get("validation") or {}
        classification = classification or validation.get("classification") or {}
        integrated_issue = package.get("integrated_issue") or package.get("summary") or {}
        card = self.generate(
            summary=integrated_issue,
            analysis=package.get("analysis") or {},
            classification=classification,
            articles=input_bundle.get("items") or [],
        )
        if not card:
            return {}
        implication_result = package.get("implication") or {}
        sentence_grounding = (
            package.get("sentence_grounding")
            or (package.get("evidence_payload") or {})
            .get("analysis_package", {})
            .get("sentence_grounding")
            or {}
        )
        is_llm_valid = bool(
            isinstance(implication_result, dict)
            and implication_result.get("is_valid_implication")
            and (
                (implication_result.get("skax_implication") or {}).get("why_important")
                or (implication_result.get("peer_implication") or {}).get("peer_meaning")
            )
        )
        if is_llm_valid:
            frontend_implication = _frontend_implication_from_result(
                implication_result,
                fallback=card.get("frontend_implication"),
                analysis=package.get("analysis") or {},
            )
            card["implication_result"] = implication_result
            card["implication"] = _implication_from_result(
                implication_result,
                fallback=card.get("implication"),
                frontend=frontend_implication,
            )
            card["frontend_implication"] = frontend_implication
        else:
            # ImplicationAgent 결과 없음/무효 → analysis 기반 frontend fallback 유지.
            card.setdefault(
                "frontend_implication", _frontend_implication(package.get("analysis") or {})
            )
        literal_summary_lines = _literal_summary_lines(integrated_issue)
        if literal_summary_lines and len(literal_summary_lines) < _SUMMARY_LINE_MIN:
            literal_summary_lines = _merge_summary_lines(
                literal_summary_lines,
                _article_title_summary_lines(input_bundle.get("items") or []),
            )
        if literal_summary_lines:
            card["summary_lines"] = literal_summary_lines
            if isinstance(card.get("db_record"), dict):
                card["db_record"]["summary_lines"] = literal_summary_lines
        display_sections = _display_sections_from_strategy_result(
            summary=integrated_issue,
            analysis=package.get("analysis") or {},
            implication=implication_result,
            strategic_root=package,
            sentence_grounding=sentence_grounding,
            sources=card.get("sources") or [],
            summary_lines=literal_summary_lines or _list_string(card.get("summary_lines")),
        )
        missing_frontend_sections = _display_sections_missing_frontend_ready(display_sections)
        card["needs_review"] = bool(missing_frontend_sections)
        if missing_frontend_sections:
            card["needs_review_reason"] = (
                "frontend_ready 직접 생성 문장이 없거나 품질 기준을 통과하지 못했습니다: "
                + ", ".join(missing_frontend_sections)
            )
        card["frontend_implication"] = _sync_frontend_implication_from_display_sections(
            card.get("frontend_implication"),
            display_sections=display_sections,
        )
        card["implication"] = _sync_implication_frontend_from_display_sections(
            card.get("implication"),
            frontend=card["frontend_implication"],
        )
        card["display_sections"] = display_sections
        card["slides"] = _slides_from_display_sections(
            title=str(card.get("title") or ""),
            display_sections=display_sections,
            sources=card.get("sources") or [],
            media_assets=_media_assets(input_bundle.get("items") or []),
        ) or card.get("slides", [])
        card["analysis_package"] = {
            "bundle_id": package.get("bundle_id"),
            "integrated_issue": integrated_issue,
            "summary": integrated_issue,
            "issue_understanding": package.get("issue_understanding") or {},
            "profile_linkage": package.get("profile_linkage") or {},
            "skax_response_linkage": package.get("skax_response_linkage") or {},
            "grounding_summary": package.get("grounding_summary") or {},
            "claim_strength": package.get("claim_strength"),
            "analysis": package.get("analysis") or {},
            "implication": implication_result,
            "sentence_grounding": sentence_grounding,
            "validation": validation,
        }
        return card

    def generate_from_cluster(
        self,
        cluster_id: int,
        representative_id: int,
        company: str,
        classification: dict[str, Any],
        cluster_article_ids: list[int] | None = None,
        peer_id: str = "",
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """클러스터 정보와 요약 결과로 카드뉴스를 생성한다.

        Args:
            cluster_id: 클러스터 ID.
            representative_id: 대표 기사 ID (가장 신뢰도 높은 기사).
            company: 회사 ID.
            classification: ClusterClassifier 결과.
            cluster_article_ids: 클러스터 내 전체 기사 ID 목록 (없으면 대표 기사만 사용).
            summary: SourceSummarizer가 만든 클러스터 사실 요약.

        Returns:
            card_news dict (저장 전 validation 없는 상태) — DB card_news 테이블 row 와 1:1.
        """
        company = company or peer_id

        # 카드 요약/출처 모두 클러스터 전체 기사를 기준으로 한다.
        article_ids = _build_cluster_fetch_ids(representative_id, cluster_article_ids)
        articles = _order_articles(get_articles_by_ids(article_ids), article_ids)
        if not articles:
            return {}

        summary_card = _card_from_summary(
            summary=summary or {},
            articles=articles,
            company=company,
            cluster_id=cluster_id,
            representative_id=representative_id,
            classification=classification,
        )
        if summary_card:
            log.info(
                "카드뉴스 생성 완료(summary) | id=%s sector=%s band=%s event=%s sources=%d",
                summary_card["id"],
                summary_card["sector"],
                summary_card["exposure_band"],
                summary_card["event_type"],
                len(articles),
            )
            return summary_card

        articles_text = _format_articles(articles)
        prompt = _ISSUE_CARD_PROMPT.replace("{articles_text}", articles_text)
        created_at = _now_iso()
        published_date = _published_date(articles, created_at)

        try:
            from src.observability import tracing_config

            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="CardNewsComposer",
                    prompt_version=_PROMPT_VERSION,
                    company=company,
                    cluster_id=cluster_id,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            card_data = _parse_json(content)

            card = {
                "id": _card_news_id(cluster_id, published_date),
                "company": company,
                "cluster_id": cluster_id,
                "representative_id": representative_id,
                "published_date": published_date,
                "title": card_data.get("title", articles[0]["title"][:100]),
                "summary_lines": card_data.get("summary_lines", []),
                "event_type": card_data.get(
                    "event_type",
                    classification.get("event_type", "tech"),
                ),
                # v3: 분류 결과의 sector·exposure 정보를 카드에 그대로 전파
                "sector": classification.get("sector", "other"),
                "sectors": classification.get("sectors", ["other"]),
                "exposure_score": classification.get("exposure_score", 0.0),
                "exposure_band": classification.get("exposure_band", "low"),
                "signals": classification.get("signals", {}),
                # 등급 자체는 v3에서 폐기되었으나, DB 컬럼 호환을 위해 노출도 밴드를 저장
                "importance": classification.get("importance", "low"),
                "importance_score": classification.get("importance_score", 0.0),
                "sources": _default_sources(articles),
            }
            _attach_card_news_schema_fields(card)

            log.info(
                "카드뉴스 생성 완료 | id=%s sector=%s band=%s event=%s sources=%d",
                card["id"],
                card["sector"],
                card["exposure_band"],
                card["event_type"],
                len(articles),
            )
            return card

        except Exception as e:
            log.error("카드뉴스 생성 실패 | cluster=%d error=%s", cluster_id, e)
            return {}


def _build_cluster_fetch_ids(
    representative_id: int, cluster_article_ids: list[int] | None
) -> list[int]:
    """대표 기사를 앞에 두고 클러스터 전체 ID 목록을 만든다."""
    if not cluster_article_ids:
        return [representative_id]
    others = [aid for aid in cluster_article_ids if aid != representative_id]
    return [representative_id, *others]


def _order_articles(
    articles: list[dict[str, Any]],
    ordered_ids: list[int],
) -> list[dict[str, Any]]:
    """DB 조회 결과를 대표기사 우선 순서로 되돌린다."""
    order = {article_id: index for index, article_id in enumerate(ordered_ids)}
    return sorted(articles, key=lambda article: order.get(int(article.get("id") or 0), len(order)))


def _card_from_summary(
    *,
    summary: dict[str, Any],
    articles: list[dict[str, Any]],
    company: str,
    cluster_id: int,
    representative_id: int,
    classification: dict[str, Any],
) -> dict[str, Any] | None:
    if not summary.get("is_valid_summary"):
        return None

    effective_company = _first_non_empty(summary.get("main_company"), company)
    summary_lines = _plain_summary_lines(summary, use_llm=True)
    if not summary_lines:
        return None
    if len(summary_lines) < _SUMMARY_LINE_MIN:
        summary_lines = _merge_summary_lines(summary_lines, _article_title_summary_lines(articles))

    title = _first_non_empty(
        summary.get("headline"),
        summary.get("one_line_summary"),
        classification.get("title"),
        articles[0].get("title"),
    )
    if _looks_like_sentence_title(title):
        title = _first_non_empty(classification.get("title"), articles[0].get("title"), title)
    title = _business_context_title(
        title,
        summary=summary,
        event_type=str(classification.get("event_type") or summary.get("cluster_event_type") or ""),
    )
    created_at = _now_iso()
    published_date = _published_date(articles, created_at)
    media_assets = _media_assets(articles)

    card = {
        "id": _card_news_id(cluster_id, published_date),
        "company": effective_company,
        "cluster_id": cluster_id,
        "representative_id": representative_id,
        "published_date": published_date,
        "title": title[:100],
        "summary_lines": summary_lines,
        "event_type": classification.get("event_type", "tech"),
        "sector": classification.get("sector", "other"),
        "sectors": classification.get("sectors", ["other"]),
        "exposure_score": classification.get("exposure_score", 0.0),
        "exposure_band": classification.get("exposure_band", "low"),
        "signals": classification.get("signals", {}),
        "importance": classification.get("importance", "low"),
        "importance_score": classification.get("importance_score", 0.0),
        "sources": _default_sources(articles),
        "news_summary": summary,
        "image_assets": media_assets,
    }
    _attach_card_news_schema_fields(card)
    return card


def _attach_card_news_schema_fields(card: dict[str, Any]) -> None:
    """card_news 테이블 저장 스키마에 맞는 필드를 카드 dict에 붙인다."""
    implication = {
        "sector": card.get("sector", "other"),
        "sectors": card.get("sectors", ["other"]),
        "exposure_score": card.get("exposure_score", 0.0),
        "exposure_band": card.get("exposure_band", "low"),
        "signals": card.get("signals", {}),
        "evidence_chain": card.get("evidence_chain", {}),
    }
    raw_validation = card.get("validation")
    validation: dict[str, Any] = raw_validation if isinstance(raw_validation, dict) else {}
    validation_pass = bool(validation.get("pass", card.get("validation_pass", False)))
    validation_sc_score = validation.get("sc_score", card.get("validation_sc_score", 0.0))
    card.update(
        {
            "implication": implication,
            "validation_pass": validation_pass,
            "validation_sc_score": validation_sc_score,
            "validation": {
                "pass": validation_pass,
                "sc_score": validation_sc_score,
            },
        }
    )
    card["db_record"] = {
        "id": card.get("id"),
        "company": card.get("company"),
        "cluster_id": card.get("cluster_id"),
        "published_date": card.get("published_date"),
        "title": card.get("title"),
        "summary_lines": card.get("summary_lines", []),
        "event_type": card.get("event_type", "tech"),
        "importance": card.get("importance", "low"),
        "importance_score": card.get("importance_score", 0.0),
        "implication": implication,
        "sources": card.get("sources", []),
        "validation_pass": validation_pass,
        "validation_sc_score": validation_sc_score,
        "image_assets": card.get("image_assets", []),
    }


def _numbered_summary_lines(value: Any) -> list[str]:
    lines = [str(item).strip() for item in _list_value(value) if str(item).strip()]
    numbered: list[str] = []
    for index, line in enumerate(lines[:_SUMMARY_LINE_MAX], start=1):
        prefix = f"{index}."
        numbered.append(line if line.startswith(prefix) else f"{prefix} {line}")
    return numbered


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _list_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in _list_value(value) if isinstance(item, dict)]


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return "피어사 주요 뉴스"


def _business_context_title(title: str, *, summary: dict[str, Any], event_type: str) -> str:
    value = _clean_card_editorial_text(title)
    if not _is_financial_only_title(value):
        return value
    if str(event_type or summary.get("cluster_event_type") or "").casefold() not in {
        "earnings",
        "analyst_report",
        "stock_market",
    }:
        return value
    focus = _business_focus_from_summary(summary)
    peer_name = company_name_ko(str(summary.get("main_company") or "")) or _first_non_empty(
        summary.get("main_company"),
        "",
    )
    if focus:
        if "전망" in value or "목표" in value:
            return f"{peer_name}, {focus} 성장 전망"
        return f"{peer_name}, {focus} 중심 실적 변화"
    return value


def _looks_like_sentence_title(title: str) -> bool:
    value = re.sub(r"\s+", " ", str(title or "")).strip()
    if len(value) > 60 and value.endswith(("다", "다.", "했다", "했다.", "됐다", "됐다.")):
        return True
    return bool(re.search(r"(했다|공개했다|체결했다|진출했다|선보였다|밝혔다)[.]?$", value))


def _is_financial_only_title(title: str) -> bool:
    text = str(title or "")
    if not re.search(r"매출|영업이익|순이익|실적|목표가|목표주가|주가", text):
        return False
    return not re.search(
        r"클라우드|AI|에이전트|데이터센터|IT서비스|IT 서비스|SI|ITO|"
        r"물류|플랫폼|솔루션|ERP|MSP|CSP|AX|SDV|모빌리티|보안|센터|"
        r"사업|서비스|수주|계약|전환|구축",
        text,
        re.I,
    )


def _business_focus_from_summary(summary: dict[str, Any]) -> str:
    facts = _summary_fact_text(summary)
    focus_rules = (
        (r"클라우드|MSP|CSP|데이터센터", "클라우드·AI 인프라"),
        (r"AI\s*에이전트|생성형\s*AI|챗GPT|브리티|AX", "AI·AX 사업"),
        (r"IT\s*서비스|IT서비스|SI|ITO|시스템통합|아웃소싱", "IT서비스 사업"),
        (r"SDV|차량\s*SW|차량SW|모빌리티|내비게이션", "모빌리티 SW 사업"),
        (r"물류|첼로|Cello", "디지털 물류 사업"),
        (r"ERP|SCM|전환|구축|운영", "엔터프라이즈 IT 사업"),
    )
    matches = [label for pattern, label in focus_rules if re.search(pattern, facts, re.I)]
    return "·".join(dict.fromkeys(matches[:2]))


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        credibility_score = a.get("credibility_score")
        credibility_text = f"{credibility_score:.2f}" if credibility_score is not None else "미계산"
        lines.append(
            f"[{i}] 제목: {a['title']}\n"
            f"    출처: {a['source_name']} (신뢰도: {credibility_text})"
            f" | URL: {a['url']}\n"
            f"    내용: {' '.join((a.get('content') or '').split())}"
        )
    return "\n\n".join(lines)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _default_sources(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": i + 1,
            "title": a["title"],
            "source_name": a["source_name"],
            "url": a["url"],
            "credibility_score": a.get("credibility_score"),
        }
        for i, a in enumerate(articles)
    ]


def _load_source_articles(summary: dict[str, Any]) -> list[dict[str, Any]]:
    article_ids = [
        article_id
        for article_id in (
            _optional_int(raw_id) for raw_id in _list_string(summary.get("source_article_ids"))
        )
        if article_id is not None
    ]
    return get_articles_by_ids(article_ids) if article_ids else []


def _card_news_id(cluster_id: int | None, published_date: str) -> str:
    date_key = published_date[:10].replace("-", "")
    suffix = f"{cluster_id:04d}" if cluster_id is not None else "0000"
    return f"CN-{date_key}-{suffix}"


def _plain_summary_lines(summary: dict[str, Any], *, use_llm: bool = False) -> list[str]:
    # Summary is a factual section: preserve IntegrationAgent copy and avoid
    # display-time rewriting, ranking, or LLM re-summary.
    del use_llm
    return _literal_summary_lines(summary)


def _display_summary_lines(
    lines: list[str],
    summary: dict[str, Any],
    *,
    use_llm: bool = False,
) -> list[str]:
    candidates = _summary_candidate_lines(lines, summary)
    if use_llm:
        refined = _llm_display_summary_lines(summary, candidates)
        if _SUMMARY_LINE_MIN <= len(refined) <= _SUMMARY_LINE_MAX:
            return refined

    key_numbers = _key_number_display_map(summary)
    ranked: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(candidates):
        text = _summary_line_for_display(_strip_number_prefix(line), key_numbers)
        dedupe_key = re.sub(r"\s+", " ", text).casefold()
        if not text or dedupe_key in seen:
            continue
        ranked.append((_summary_line_priority(text, summary), -index, text))
        seen.add(dedupe_key)
    ranked.sort(reverse=True)
    selected: list[str] = []
    skipped: list[str] = []
    selected_roles: set[str] = set()
    selected_keys: set[str] = set()
    for _, _, text in ranked:
        if len(selected) >= _SUMMARY_LINE_MAX:
            break
        role = _summary_line_role(text, summary)
        if role == "market_reaction":
            continue
        if role in selected_roles and role != "context" and len(selected) < _SUMMARY_LINE_MIN:
            skipped.append(text)
            continue
        if any(_summary_lines_too_similar(text, existing) for existing in selected):
            skipped.append(text)
            continue
        key = re.sub(r"\s+", " ", text).casefold()
        selected.append(text)
        selected_keys.add(key)
        selected_roles.add(role)
    for text in skipped:
        if len(selected) >= _SUMMARY_LINE_MIN:
            break
        key = re.sub(r"\s+", " ", text).casefold()
        if key in selected_keys:
            continue
        if any(_summary_lines_too_similar(text, existing) for existing in selected):
            continue
        selected.append(text)
        selected_keys.add(key)
    return selected[:_SUMMARY_LINE_MAX] if len(selected) >= _SUMMARY_LINE_MIN else selected


def _summary_candidate_lines(lines: list[str], summary: dict[str, Any]) -> list[str]:
    candidates = [str(line or "").strip() for line in lines if str(line or "").strip()]
    intelligence = summary.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for group_name in ("common_facts", "unique_facts"):
            for item in _list_dicts(intelligence.get(group_name)):
                fact_text = str(item.get("fact") or "").strip()
                if fact_text:
                    candidates.append(fact_text)
    for fact_item in _list_dicts(summary.get("consolidated_facts")):
        text = str(fact_item.get("fact") or "").strip()
        if text:
            candidates.append(text)
    return candidates


def _llm_display_summary_lines(summary: dict[str, Any], candidates: list[str]) -> list[str]:
    if not os.getenv("OPENAI_API_KEY"):
        return []
    clean_candidates = []
    seen: set[str] = set()
    key_numbers = _key_number_display_map(summary)
    for line in candidates:
        text = _summary_line_for_display(_strip_number_prefix(line), key_numbers)
        key = re.sub(r"\s+", " ", text).casefold()
        if text and key not in seen:
            clean_candidates.append(text)
            seen.add(key)
    if len(clean_candidates) < _SUMMARY_LINE_MIN:
        return []

    context = {
        "headline": summary.get("headline"),
        "main_event": summary.get("main_event"),
        "main_issue": summary.get("main_issue"),
        "one_line_summary": summary.get("one_line_summary"),
        "event_type": summary.get("cluster_event_type") or summary.get("event_type"),
        "candidate_facts": clean_candidates[:20],
    }
    prompt = (
        "You are selecting frontend summary lines for a Korean executive card news item.\n"
        "Choose 3 to 5 lines only from candidate_facts. Do not invent facts.\n"
        "Avoid repeating the same fact in different wording. Prefer lines that together cover "
        "different factual dimensions such as the event, amount/scale, period/schedule, purpose, "
        "execution scope, or source-backed consequence when present.\n"
        "Do not include stock/market reaction unless the event_type itself is stock_market.\n"
        'Return strict JSON only: {"summary_lines": ["..."]}.\n\n'
        f"INPUT:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    try:
        from src.observability import tracing_config

        response = _get_llm().invoke(
            prompt,
            config=tracing_config(
                agent="CardNewsComposer",
                prompt_version="card-news-summary-select-v1.0",
                company=str(summary.get("main_company") or ""),
                cluster_id=_optional_int(summary.get("cluster_id")),
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_json(content)
    except Exception as exc:  # noqa: BLE001 - display fallback should not block card generation.
        log.warning("카드뉴스 표시 요약 LLM 선별 실패 | error=%s", exc)
        return []

    selected: list[str] = []
    candidate_set = {re.sub(r"\s+", " ", item).casefold() for item in clean_candidates}
    for item in _list_string(parsed.get("summary_lines")):
        text = _summary_line_for_display(_strip_number_prefix(item), key_numbers)
        key = re.sub(r"\s+", " ", text).casefold()
        if not text or key not in candidate_set:
            continue
        if any(_summary_lines_too_similar(text, existing) for existing in selected):
            continue
        selected.append(text)
        if len(selected) >= _SUMMARY_LINE_MAX:
            break
    return selected if len(selected) >= _SUMMARY_LINE_MIN else []


def _summary_line_priority(text: str, summary: dict[str, Any]) -> int:
    value = str(text or "")
    score = 0
    event_type = str(
        summary.get("cluster_event_type") or summary.get("event_type") or ""
    ).casefold()
    if _is_market_reaction_summary_line(value) and event_type != "stock_market":
        score -= 100
    shared_focus = _summary_similarity_tokens(value) & _summary_focus_tokens(summary)
    score += min(len(shared_focus), 8) * 9
    if _has_numeric_or_period_signal(value):
        score += 20
    if _has_target_capacity_or_schedule(value):
        score += 45
    if _has_schedule_signal(value) and _has_non_money_quantity(value):
        score += 35
    elif _has_schedule_signal(value):
        score += 15
    if _has_target_capacity_or_schedule(value) and not _has_schedule_signal(value):
        if re.search(r"최종|계약|확정|완료", value):
            score -= 25
    if _contains_key_number_text(value, summary):
        score += 25
    if re.search(r"선정|확정|수주|계약|체결|참여|사업자", value):
        score += 20
    if re.search(r"구축|설립|운영|착공|도입|전환|현대화|협약|계약", value):
        score += 15
    if re.search(r"주가|거래소|거래\s*(중|마쳤)|상승|하락|급등|급락", value):
        score -= 30
    return score


def _summary_line_role(text: str, summary: dict[str, Any]) -> str:
    value = str(text or "")
    event_type = str(
        summary.get("cluster_event_type") or summary.get("event_type") or ""
    ).casefold()
    if _is_market_reaction_summary_line(value) and event_type != "stock_market":
        return "market_reaction"
    if re.search(r"최종\s*선정|민간\s*참여|사업자로\s*선정|사업자에", value):
        return "core_event"
    if re.search(r"협약|주주\s*간|SPC|특수목적법인|출자|설립", value, re.IGNORECASE):
        return "governance"
    if _has_target_capacity_or_schedule(value):
        return "scale_schedule"
    if re.search(r"선정|확정|수주|계약|체결|참여|사업자", value):
        return "core_event"
    if _has_numeric_or_period_signal(value):
        return "scale_schedule"
    return "context"


def _summary_lines_too_similar(left: str, right: str) -> bool:
    left_tokens = _summary_similarity_tokens(left)
    right_tokens = _summary_similarity_tokens(right)
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return False
    overlap = len(left_tokens & right_tokens)
    ratio = overlap / min(len(left_tokens), len(right_tokens))
    if ratio >= 0.72:
        return True
    shared_actions = (left_tokens & right_tokens) & _SUMMARY_ACTION_TOKENS
    return bool(shared_actions) and ratio >= 0.6


def _summary_similarity_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·_-]{1,}", str(text or ""))
    stopwords = {
        "사업",
        "계약",
        "체결",
        "완료",
        "밝혔다",
        "위한",
        "관련",
        "통해",
        "규모",
        "계획",
        "예정",
    }
    normalized_stopwords = {_normalize_summary_similarity_token(token) for token in stopwords}
    normalized_tokens: set[str] = set()
    for token in tokens:
        pieces = [token, *re.split(r"[·/&+_-]+", token)]
        for piece in pieces:
            normalized = _normalize_summary_similarity_token(piece)
            if normalized and normalized not in normalized_stopwords:
                normalized_tokens.add(normalized)
    return normalized_tokens


def _normalize_summary_similarity_token(token: str) -> str:
    value = str(token or "").strip()
    if not value:
        return ""
    if value.isascii():
        return value.casefold()
    value = re.sub(
        r"(으로서|으로써|로서|로써|으로|에게서|에게|에서|부터|까지|보다|처럼|"
        r"은|는|이|가|을|를|의|에|와|과|도|만|로)$",
        "",
        value,
    )
    suffix_replacements = (
        (r"선정(?:됐다|되었다|했다|하였다)$", "선정"),
        (r"체결(?:됐다|되었다|했다|하였다)$", "체결"),
        (r"확정(?:됐다|되었다|했다|하였다)$", "확정"),
        (r"구축(?:됐다|되었다|했다|하였다)$", "구축"),
        (r"운영(?:됐다|되었다|했다|하였다)$", "운영"),
        (r"상승(?:했다|하였다)$", "상승"),
        (r"하락(?:했다|하였다)$", "하락"),
        (r"(?:했다|하였다|됐다|되었다|한다|됩니다|합니다|입니다|이다)$", ""),
    )
    for pattern, replacement in suffix_replacements:
        updated = re.sub(pattern, replacement, value)
        if updated != value:
            value = updated
            break
    return value if len(value) >= 2 else ""


def _summary_focus_tokens(summary: dict[str, Any]) -> set[str]:
    focus_parts = [
        summary.get("headline"),
        summary.get("main_event"),
        summary.get("main_issue"),
        summary.get("one_line_summary"),
    ]
    if not any(str(part or "").strip() for part in focus_parts):
        for line in _list_string(summary.get("fact_summary")):
            if not _is_market_reaction_summary_line(line):
                focus_parts.append(line)
                break
    return _summary_similarity_tokens(" ".join(str(part or "") for part in focus_parts))


def _has_numeric_or_period_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\d|년|월|일|까지|부터|규모|금액|기간|비율|대비|투자|자본금|"
            r"매출|수량|장|대|명|억원|조원|만원|%",
            str(text or ""),
        )
    )


def _has_target_capacity_or_schedule(text: str) -> bool:
    value = str(text or "")
    has_quantity = bool(re.search(r"\d[\d,.\s]*(장|대|개|건|명|곳|식|세트|억원|조원)", value))
    has_schedule = _has_schedule_signal(value)
    has_execution = bool(re.search(r"구축|운영|도입|전환|착공|확보|조성|목표|예정|계획", value))
    return has_execution and (has_quantity or has_schedule)


def _has_schedule_signal(text: str) -> bool:
    return bool(re.search(r"\d{4}\s*년|까지|부터|기간|단계|분기|월|일", str(text or "")))


def _has_non_money_quantity(text: str) -> bool:
    return bool(re.search(r"\d[\d,.\s]*(장|대|개|건|명|곳|식|세트)", str(text or "")))


def _contains_key_number_text(text: str, summary: dict[str, Any]) -> bool:
    value = str(text or "")
    for item in _list_dicts(summary.get("key_numbers")):
        label = str(item.get("metric_label") or item.get("metric_name") or "").strip()
        raw_value = str(item.get("value") or "").strip()
        unit = str(item.get("unit") or "").strip()
        if label and label in value:
            return True
        if raw_value and raw_value in value:
            return True
        if unit and raw_value and f"{raw_value}{unit}" in value:
            return True
    return False


def _is_market_reaction_summary_line(text: str) -> bool:
    return bool(
        re.search(
            r"주가|한국거래소|전\s*거래일|거래\s*(중|마쳤)|"
            r"장\s*(초반|마감)|상승|하락|급등|급락|투자자|시장\s*반응",
            str(text or ""),
        )
    )


def _summary_line_for_display(line: str, key_numbers: dict[str, str]) -> str:
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    if not text:
        return ""
    if text in key_numbers:
        return key_numbers[text]
    metric_match = re.fullmatch(
        r"(?P<label>[가-힣A-Za-z&·/\s]+?)\s+(?P<value>-?\d+(?:\.\d+)?)",
        text,
    )
    if metric_match:
        label = re.sub(r"\s+", " ", metric_match.group("label")).strip()
        value = _format_numeric_text(metric_match.group("value"))
        lookup_key = f"{label} {metric_match.group('value')}"
        if lookup_key in key_numbers:
            return key_numbers[lookup_key]
        if re.search(r"\b(YoY|QoQ)\b|증감|성장률|이익률|마진", label, re.IGNORECASE):
            return f"{label} {value}%"
        return f"{label} {value}"
    return text


def _key_number_display_map(summary: dict[str, Any]) -> dict[str, str]:
    display: dict[str, str] = {}
    for item in _list_dicts(summary.get("key_numbers")):
        label = str(item.get("metric_label") or item.get("metric_name") or "").strip()
        raw_value = str(item.get("value") or "").strip()
        if not label or not raw_value:
            continue
        value = _format_numeric_text(raw_value)
        unit = str(item.get("unit") or "").strip()
        text = f"{label} {value}{unit}" if unit and not value.endswith(unit) else f"{label} {value}"
        display[f"{label} {raw_value}"] = text
        display[f"{label} {value}"] = text
    return display


def _format_numeric_text(value: str) -> str:
    try:
        number = float(str(value).replace(",", ""))
    except ValueError:
        return str(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.1f}".rstrip("0").rstrip(".")


def _strip_number_prefix(value: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", str(value or "")).strip()


def _db_record(
    *,
    card_id: str,
    company: str | None,
    cluster_id: int | None,
    title: str,
    summary_lines: list[str],
    event_type: str,
    importance: str,
    importance_score: float | None,
    sector: str,
    sectors: list[str],
    signals: dict[str, Any],
    evidence_chain: dict[str, Any],
    sources: list[dict[str, Any]],
    validation_pass: bool,
    validation_sc_score: float,
) -> dict[str, Any]:
    return {
        "id": card_id,
        "company": company,
        "cluster_id": cluster_id,
        "title": title[:500],
        "summary_lines": summary_lines,
        "event_type": event_type,
        "importance": importance,
        "importance_score": importance_score or 0.0,
        "implication": {
            "sector": sector,
            "sectors": sectors,
            "exposure_score": importance_score or 0.0,
            "exposure_band": importance,
            "signals": signals,
            "evidence_chain": evidence_chain,
        },
        "sources": sources,
        "validation_pass": validation_pass,
        "validation_sc_score": validation_sc_score,
    }


def mark_near_duplicate_card_candidates(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 회사·핵심 제품/서비스명이 겹치는 단일 기사 카드를 병합 후보로 표시한다."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for card in cards:
        signals = _card_signals(card)
        signature = str(signals.get("issue_signature") or "").strip()
        if not signature:
            card_text = _card_text_from_card(card)
            terms = _issue_terms(card.get("news_summary") or {}, card_text)
            if terms:
                signature = _issue_signature(terms)
                signals["issue_signature"] = signature
                signals["near_duplicate_terms"] = terms
                _set_card_signals(card, signals)
        company = str(card.get("company") or card.get("peer_id") or "").strip()
        if signature and company:
            buckets.setdefault((company, signature), []).append(card)

    for grouped in buckets.values():
        if len(grouped) < 2:
            continue
        cluster_ids = [
            cluster_id
            for cluster_id in (_optional_int(card.get("cluster_id")) for card in grouped)
            if cluster_id is not None
        ]
        card_ids = [str(card.get("id")) for card in grouped if card.get("id")]
        for card in grouped:
            signals = _card_signals(card)
            signals["near_duplicate_card_candidate"] = True
            signals["near_duplicate_reason"] = (
                "같은 회사와 핵심 제품/서비스 키워드가 반복된 단일 기사 카드 후보"
            )
            signals["near_duplicate_card_ids"] = card_ids
            signals["near_duplicate_cluster_ids"] = cluster_ids
            _set_card_signals(card, signals)
    return cards


def _frontend_implication(analysis: dict[str, Any]) -> dict[str, Any]:
    key_implications = _bounded_detail_lines(
        [
            analysis.get("analysis_summary"),
            analysis.get("impact_reason"),
            *(_list_string(analysis.get("strategic_meaning"))),
        ]
    )
    return {
        "why_important": str(analysis.get("analysis_summary") or "").strip(),
        "potential_impact": str(analysis.get("impact_reason") or "").strip(),
        "follow_up_questions": [],
        "key_implications": key_implications,
        "suggested_actions": [],
        "confidence": _optional_float(analysis.get("confidence")),
    }


def _implication(analysis: dict[str, Any]) -> dict[str, Any]:
    return _frontend_implication(analysis)


def _frontend_implication_from_result(
    implication: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """v4.0 schema 인식 — peer_implication / skax_implication dict 의 핵심 필드 추출.

    W2-4: 기존에 peer_implication 전체를 str() 으로 변환하던 버그 정정. CardNewsComposer
    가 frontend 에 보내는 표면 schema 와 일치.
    """
    skax = implication.get("skax_implication") or {}
    peer = implication.get("peer_implication") or {}
    frontend_ready_items = _frontend_ready_display_items(implication)
    industry_metadata: dict[str, Any] = {}
    if not (frontend_ready_items.get("insight") or frontend_ready_items.get("action")):
        frontend_ready_items = _industry_frontend_ready_display_items(implication)
        industry_metadata = frontend_ready_items.get("metadata") or {}

    peer_implications = frontend_ready_items.get("insight") or []
    suggested_actions = frontend_ready_items.get("action") or []
    why_important = peer_implications[0] if peer_implications else ""
    potential_impact = peer_implications[0] if peer_implications else ""
    confidence = _optional_float(implication.get("confidence"))
    payload: dict[str, Any] = {
        "why_important": why_important,
        "potential_impact": potential_impact,
        "key_implications": peer_implications,
        "peer_implications": peer_implications,
        "skax_implications": _bounded_detail_items(why_important, potential_impact),
        "response_directions": suggested_actions,
        "follow_up_questions": [],
        "suggested_actions": suggested_actions,
        "confidence": confidence,
    }
    if industry_metadata:
        payload.update(industry_metadata)
    if isinstance(skax, dict):
        if skax.get("opportunities"):
            payload["opportunities"] = _list_string(skax.get("opportunities"))
        if skax.get("threats"):
            payload["threats"] = _list_string(skax.get("threats"))
        if skax.get("business_line_mapping"):
            payload["business_line_mapping"] = _list_string(skax.get("business_line_mapping"))
    if isinstance(peer, dict):
        if peer.get("company_id"):
            payload["peer_company_id"] = peer.get("company_id")
        if peer.get("company_name_ko"):
            payload["peer_company_name_ko"] = peer.get("company_name_ko")
        if peer.get("capability_change"):
            payload["peer_capability_change"] = peer.get("capability_change")
        if peer.get("precedent_link"):
            payload["precedent_link"] = peer.get("precedent_link")
    if implication.get("evidence_label"):
        payload["evidence_label"] = implication.get("evidence_label")
    return _with_structured_frontend_blocks(payload)


def _implication_from_result(
    implication: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
    frontend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """W2-4: CardNewsComposer 가 DB 에 저장할 implication JSONB 의 단일 출처.

    v2 schema 의 `peer_implication` / `skax_implication` / `follow_up_questions` /
    `watch_points` / `confidence` / `evidence_label` / `provenance` 를 모두 포함하고,
    frontend 호환 핵심 필드도 함께 평면화.
    """
    fallback = fallback or {}
    payload = dict(implication)
    # v2 schema 가 사용하는 필드를 모두 안전 default 로 채운다.
    payload.setdefault("implication_scope", "peer_and_skax")
    payload.setdefault("watch_points", payload.get("watch_points", []))
    # frontend 호환 - flatten.
    payload["frontend"] = frontend or _frontend_implication_from_result(
        implication,
        fallback=fallback,
    )
    return payload


def _frontend_ready_display_items(implication: Any) -> dict[str, Any]:
    if not isinstance(implication, dict):
        return {"insight": [], "action": [], "insight_blocks": [], "action_blocks": []}
    frontend_ready = implication.get("frontend_ready") or {}
    if not isinstance(frontend_ready, dict):
        return {"insight": [], "action": [], "insight_blocks": [], "action_blocks": []}
    source = str(frontend_ready.get("source") or "").strip()
    insight_blocks = _frontend_ready_section_blocks(
        frontend_ready.get("key_implication"),
        source=source,
        anchor_key="profile_anchor_terms",
    )
    action_blocks = _frontend_ready_section_blocks(
        frontend_ready.get("suggested_action"),
        source=source,
        anchor_key="skax_anchor_terms",
    )
    return {
        "insight": _labeled_frontend_ready_items(insight_blocks, heading="핵심 시사점"),
        "action": _labeled_frontend_ready_items(action_blocks, heading="핵심 대응"),
        "insight_blocks": insight_blocks,
        "action_blocks": action_blocks,
    }


def _industry_frontend_ready_display_items(implication: Any) -> dict[str, Any]:
    if not isinstance(implication, dict):
        return {"insight": [], "action": [], "metadata": {}}
    industry_ready = implication.get("industry_frontend_ready") or {}
    if not isinstance(industry_ready, dict):
        return {"insight": [], "action": [], "metadata": {}}
    if str(industry_ready.get("source") or "").strip() != "industry_signal_direct":
        return {"insight": [], "action": [], "metadata": {}}
    metadata = {
        "signal_scope": str(industry_ready.get("signal_scope") or "industry_signal"),
        "display_policy": str(industry_ready.get("display_policy") or "industry_only"),
    }
    insight_items: list[str] = []
    action_items: list[str] = []
    insight_blocks: list[dict[str, str]] = []
    action_blocks: list[dict[str, str]] = []
    for item in industry_ready.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_insight_blocks = _frontend_ready_section_blocks(
            item.get("key_implication"),
            source="industry_signal_direct",
            anchor_key="event_anchor_terms",
            display_sources={"industry_signal_direct"},
        )
        item_action_blocks = _frontend_ready_section_blocks(
            item.get("suggested_action"),
            source="industry_signal_direct",
            anchor_key="event_anchor_terms",
            display_sources={"industry_signal_direct"},
        )
        insight_blocks.extend(item_insight_blocks)
        action_blocks.extend(item_action_blocks)
        insight_items.extend(
            _labeled_frontend_ready_items(item_insight_blocks, heading="핵심 시사점")
        )
        action_items.extend(_labeled_frontend_ready_items(item_action_blocks, heading="핵심 대응"))
    return {
        "insight": insight_items[:2],
        "action": action_items[:2],
        "insight_blocks": insight_blocks[:2],
        "action_blocks": action_blocks[:2],
        "metadata": metadata if insight_items or action_items else {},
    }


def _frontend_ready_section_blocks(
    value: Any,
    *,
    source: str,
    anchor_key: str,
    display_sources: set[str] | None = None,
) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        return []
    block_source = str(value.get("source") or source or "").strip()
    allowed_sources = display_sources or {"llm_direct", "frontend_repair_direct"}
    if block_source not in allowed_sources:
        return []
    evidence_mode = str(value.get("evidence_mode") or "").strip()
    if evidence_mode == "profile_based" and not _list_string(value.get(anchor_key)):
        return []
    sentence = _clean_frontend_ready_text(value.get("sentence"))
    evidence_sentence = _clean_frontend_ready_text(value.get("evidence_sentence"))
    if not sentence or not evidence_sentence:
        return []
    return [
        {
            "main": _ensure_card_sentence(sentence),
            "detail": _ensure_card_sentence(evidence_sentence),
        }
    ]


def _labeled_frontend_ready_items(blocks: list[dict[str, str]], *, heading: str) -> list[str]:
    items: list[str] = []
    for block in blocks:
        main = str(block.get("main") or "").strip()
        detail = str(block.get("detail") or "").strip()
        if not main or not detail:
            continue
        items.append(f"{heading}: {main}\n근거/설명: {detail}")
    return items


def _display_sections_from_strategy_result(
    *,
    summary: dict[str, Any],
    analysis: dict[str, Any],
    implication: dict[str, Any],
    strategic_root: dict[str, Any],
    sentence_grounding: dict[str, Any] | None,
    sources: list[dict[str, Any]],
    summary_lines: list[str] | None = None,
) -> list[dict[str, Any]]:
    del sources
    summary_items = _literal_summary_lines(summary, fallback=summary_lines)
    frontend_ready_items = _frontend_ready_display_items(implication)
    insight_items = frontend_ready_items.get("insight") or []
    action_items = frontend_ready_items.get("action") or []
    insight_blocks = frontend_ready_items.get("insight_blocks") or []
    action_blocks = frontend_ready_items.get("action_blocks") or []
    section_metadata: dict[str, Any] = {}
    if not (insight_items or action_items):
        industry_items = _industry_frontend_ready_display_items(implication)
        insight_items = industry_items.get("insight") or []
        action_items = industry_items.get("action") or []
        insight_blocks = industry_items.get("insight_blocks") or []
        action_blocks = industry_items.get("action_blocks") or []
        section_metadata = industry_items.get("metadata") or {}
    return [
        {"type": "summary", "title": "요약", "items": summary_items},
        {
            "type": "insight",
            "title": "시사점",
            "items": insight_items,
            "structured_items": insight_blocks,
            **section_metadata,
        },
        {
            "type": "action",
            "title": "대응방안",
            "items": action_items,
            "structured_items": action_blocks,
            **section_metadata,
        },
    ]


def _display_sections_missing_frontend_ready(display_sections: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for section_type, label in (("insight", "시사점"), ("action", "대응방안")):
        section = next(
            (
                item
                for item in display_sections
                if isinstance(item, dict) and item.get("type") == section_type
            ),
            {},
        )
        if not _list_string(section.get("items") if isinstance(section, dict) else []):
            missing.append(label)
    return missing


def _sync_frontend_implication_from_display_sections(
    frontend: Any,
    *,
    display_sections: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = dict(frontend) if isinstance(frontend, dict) else {}
    sections = {
        str(section.get("type") or ""): _list_string(section.get("items"))
        for section in display_sections
        if isinstance(section, dict)
    }
    structured_sections = {
        str(section.get("type") or ""): _clean_structured_blocks(
            section.get("structured_items")
            or _structured_blocks_from_labeled_lines(section.get("items"))
        )
        for section in display_sections
        if isinstance(section, dict)
    }
    insight_items = sections.get("insight") or []
    action_items = sections.get("action") or []
    insight_blocks = structured_sections.get("insight") or []
    action_blocks = structured_sections.get("action") or []
    payload["key_implications"] = insight_items
    payload["peer_implications"] = insight_items
    payload["response_directions"] = action_items
    payload["suggested_actions"] = action_items
    payload["key_implication_blocks"] = insight_blocks
    payload["key_implication_items"] = insight_blocks
    payload["response_direction_blocks"] = action_blocks
    payload["suggested_action_items"] = action_blocks
    payload["follow_up_questions"] = []
    for section in display_sections:
        if not isinstance(section, dict):
            continue
        for key in ("signal_scope", "display_policy"):
            if section.get(key):
                payload[key] = section.get(key)
    return _cleanup_public_frontend_implication(payload)


def _sync_implication_frontend_from_display_sections(
    implication: Any,
    *,
    frontend: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(implication) if isinstance(implication, dict) else {}
    payload["frontend"] = frontend
    return payload


def _cleanup_public_frontend_implication(frontend: dict[str, Any]) -> dict[str, Any]:
    """Format frontend copy as conclusion + evidence without changing meaning."""

    cleaned = dict(frontend or {})
    for key in (
        "why_important",
        "potential_impact",
        "peer_capability_change",
    ):
        if key in cleaned:
            cleaned[key] = _public_copy_cleanup(cleaned.get(key))
    for key in (
        "key_implications",
        "peer_implications",
        "skax_implications",
        "suggested_actions",
        "response_directions",
        "skax_checkpoints",
    ):
        if key not in cleaned:
            continue
        section = (
            "action"
            if key in {"suggested_actions", "response_directions", "skax_checkpoints"}
            else "insight"
        )
        cleaned[key] = [
            formatted
            for item in _list_string(cleaned.get(key))
            if (
                formatted := _format_public_frontend_item(
                    item if _is_labeled_public_frontend_item(item) else _public_copy_cleanup(item),
                    section=section,
                )
            )
        ]
    if "follow_up_questions" in cleaned:
        cleaned["follow_up_questions"] = [
            _public_copy_cleanup(item) for item in _list_string(cleaned.get("follow_up_questions"))
        ]
    return _with_structured_frontend_blocks(cleaned)


def _with_structured_frontend_blocks(frontend: dict[str, Any]) -> dict[str, Any]:
    payload = dict(frontend or {})
    if not payload.get("key_implication_blocks"):
        payload["key_implication_blocks"] = _structured_blocks_from_labeled_lines(
            payload.get("key_implications") or payload.get("peer_implications")
        )
    if not payload.get("key_implication_items"):
        payload["key_implication_items"] = payload.get("key_implication_blocks") or []
    if not payload.get("response_direction_blocks"):
        payload["response_direction_blocks"] = _structured_blocks_from_labeled_lines(
            payload.get("response_directions")
            or payload.get("suggested_actions")
            or payload.get("skax_checkpoints")
        )
    if not payload.get("suggested_action_items"):
        payload["suggested_action_items"] = payload.get("response_direction_blocks") or []
    if not payload.get("skax_checkpoint_blocks"):
        payload["skax_checkpoint_blocks"] = payload.get("response_direction_blocks") or []
    for key in (
        "key_implication_blocks",
        "key_implication_items",
        "response_direction_blocks",
        "suggested_action_items",
        "skax_checkpoint_blocks",
    ):
        payload[key] = _clean_structured_blocks(payload.get(key))
    return payload


def _clean_structured_blocks(value: Any) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for item in _list_value(value):
        if not isinstance(item, dict):
            continue
        main = _public_copy_cleanup(item.get("main"))
        detail = _public_copy_cleanup(item.get("detail"))
        if main:
            blocks.append({"main": main, "detail": detail})
    return blocks


def _structured_blocks_from_labeled_lines(lines: Any) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for line in _list_string(lines):
        block = _split_main_detail_block(line)
        if block["main"]:
            blocks.append(block)
    return blocks


def _split_main_detail_block(text: str) -> dict[str, str]:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return {"main": "", "detail": ""}

    value = re.sub(r"^핵심\s*(?:시사점|대응)\s*:\s*", "", value).strip()
    parts = re.split(r"\s*근거\s*/?\s*설명\s*:\s*", value, maxsplit=1)
    main = parts[0].strip() if parts else ""
    detail = parts[1].strip() if len(parts) == 2 else ""

    main = re.sub(r"^핵심\s*(?:시사점|대응)\s*:\s*", "", main).strip()
    detail = re.sub(r"^근거\s*/?\s*설명\s*:\s*", "", detail).strip()
    return {
        "main": _public_copy_cleanup(main),
        "detail": _public_copy_cleanup(detail),
    }


def _format_public_frontend_item(value: Any, *, section: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if _is_labeled_public_frontend_item(text):
        return text
    conclusion, evidence = _split_public_conclusion_evidence(text)
    if not conclusion or not evidence:
        return text
    heading = "핵심 대응" if section == "action" else "핵심 시사점"
    return f"{heading}: {conclusion}\n근거/설명: {evidence}"


def _is_labeled_public_frontend_item(value: Any) -> bool:
    return bool(re.match(r"^핵심\s*(시사점|대응)\s*[:：]", str(value or "").strip()))


def _split_public_conclusion_evidence(value: Any) -> tuple[str, str]:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    sentences = _public_sentences(text)
    if len(sentences) >= 2:
        conclusion_count = (
            2 if _first_sentence_is_too_thin(sentences[0]) and len(sentences) >= 3 else 1
        )
        return (
            " ".join(sentences[:conclusion_count]).strip(),
            " ".join(sentences[conclusion_count:]).strip(),
        )

    # Some generated copy is one long sentence. Split only on reasoning
    # connectors so the original logic is preserved.
    for pattern in (
        r"\s+(기사에서는|근거는|이\s*근거|따라서|다만|그래야|이\s*기준|이\s*정보|후속으로는)\s+",
        r"\s+(때문에|확인되므로|확인되어야)\s+",
    ):
        match = re.search(pattern, text)
        if match and match.start() >= 35:
            return text[: match.start()].strip(), text[match.start() :].strip()
    return "", ""


def _public_sentences(value: Any) -> list[str]:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return []
    normalized = re.sub(r"(다\.|[.!?。])\s+", r"\1\n", text)
    return [part.strip() for part in normalized.splitlines() if part.strip()]


def _first_sentence_is_too_thin(sentence: str) -> bool:
    text = str(sentence or "").strip()
    if len(text) < 45:
        return True
    return bool(re.search(r"^(이번|해당|현재)\s", text)) and not re.search(
        r"때문|근거|확인|보여|의미|따라서|다만|왜|기준",
        text,
    )


def _public_copy_cleanup(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = _strip_public_section_prefixes(text)
    return text.strip()


def _insight_candidates_from_strategy(
    *,
    analysis: dict[str, Any],
    implication: dict[str, Any],
    strategic_root: dict[str, Any],
) -> list[dict[str, Any]]:
    peer = implication.get("peer_implication") or {}
    profile_linkage = strategic_root.get("profile_linkage") or {}
    candidates: list[dict[str, Any]] = [
        {
            "text": analysis.get("analysis_summary"),
            "path": "analysis.analysis_summary",
            "priority": 60,
            "semantic_role": "peer_business_meaning",
        },
        {
            "text": peer.get("peer_meaning") if isinstance(peer, dict) else None,
            "path": "peer_implication.peer_meaning",
            "priority": 58,
            "semantic_role": "peer_business_meaning",
        },
        {
            "text": peer.get("capability_change") if isinstance(peer, dict) else None,
            "path": "peer_implication.capability_change",
            "priority": 56,
            "semantic_role": "competition_standard_change",
        },
        {
            "text": analysis.get("market_signal"),
            "path": "analysis.market_signal",
            "priority": 54,
            "semantic_role": "competition_standard_change",
        },
    ]
    for index, item in enumerate(_list_string(analysis.get("strategic_meaning"))):
        candidates.append(
            {
                "text": item,
                "path": f"analysis.strategic_meaning[{index}]",
                "priority": 57 - index,
                "semantic_role": (
                    "peer_business_meaning" if index == 0 else "competition_standard_change"
                ),
            }
        )
    if isinstance(profile_linkage, dict):
        candidates.append(
            {
                "text": profile_linkage.get("reason"),
                "path": "profile_linkage.reason",
                "priority": 45,
                "allow_watch_point": True,
                "semantic_role": "uncertainty_or_follow_up",
            }
        )
    return candidates


def _action_candidates_from_strategy(
    *,
    implication: dict[str, Any],
    strategic_root: dict[str, Any],
) -> list[dict[str, Any]]:
    skax = implication.get("skax_implication") or {}
    skax_linkage = strategic_root.get("skax_response_linkage") or {}
    candidates: list[dict[str, Any]] = []
    if isinstance(skax, dict):
        for index, item in enumerate(_list_string(skax.get("recommended_actions"))):
            candidates.append(
                {
                    "text": _clean_card_editorial_text(item),
                    "path": f"skax_implication.recommended_actions[{index}]",
                    "priority": 60 - (index * 5),
                    "semantic_role": _action_semantic_role_for_index(index),
                }
            )
    if isinstance(skax_linkage, dict):
        for field, priority, allow_watch in (
            ("internal_checkpoints", 90, False),
            ("recommended_focus", 86, False),
            ("monitoring_points", 84, True),
        ):
            for index, item in enumerate(_list_string(skax_linkage.get(field))):
                candidates.append(
                    {
                        "text": item,
                        "path": f"skax_response_linkage.{field}[{index}]",
                        "priority": priority - index,
                        "allow_watch_point": allow_watch,
                        "semantic_role": _action_semantic_role_for_linkage_field(field),
                    }
                )
    return candidates


def _action_semantic_role_for_index(index: int) -> str:
    if index == 0:
        return "skax_capability_gap_check"
    if index == 1:
        return "skax_operating_response"
    return "monitoring_response"


def _action_semantic_role_for_linkage_field(field: str) -> str:
    if field == "internal_checkpoints":
        return "skax_capability_gap_check"
    if field == "recommended_focus":
        return "skax_operating_response"
    if field == "monitoring_points":
        return "monitoring_response"
    return ""


def _literal_summary_lines(
    summary: dict[str, Any],
    *,
    fallback: list[str] | None = None,
) -> list[str]:
    """Return IntegratedIssue summary copy without display rewriting."""
    lines = _list_string(summary.get("fact_summary"))
    if not lines:
        lines = _list_string(summary.get("summary_lines"))
    if not lines:
        lines = [
            str(fact.get("fact") or "").strip()
            for fact in _list_dicts(summary.get("consolidated_facts"))
            if str(fact.get("fact") or "").strip()
        ]
    if not lines:
        lines = _list_string(summary.get("one_line_summary"))
    if not lines:
        lines = _list_string(summary.get("integrated_text"))
    if not lines:
        lines = fallback or []

    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = str(line or "").strip()
        key = re.sub(r"\s+", " ", text).casefold()
        if not text or key in seen:
            continue
        out.append(text)
        seen.add(key)
        if len(out) >= _SUMMARY_LINE_MAX:
            break
    return out


def _article_title_summary_lines(articles: list[dict[str, Any]]) -> list[str]:
    """Fallback factual summary when IntegrationAgent marks a cluster invalid."""
    out: list[str] = []
    seen: set[str] = set()

    def add_line(value: Any) -> None:
        text = _clean_card_editorial_text(str(value or ""))
        text = re.sub(r"\s+", " ", text).strip(" .")
        if not text or _looks_like_article_boilerplate(text):
            return
        key = re.sub(r"\W+", "", text).casefold()
        if key in seen:
            return
        out.append(text)
        seen.add(key)

    for article in articles:
        title = str(article.get("title") or "").strip()
        add_line(title)
        content = str(article.get("content") or "")
        for sentence in _article_content_sentences(content):
            add_line(sentence)
            if len(out) >= _SUMMARY_LINE_MIN:
                break
        if len(out) >= _SUMMARY_LINE_MAX:
            break
    return out[:_SUMMARY_LINE_MAX]


def _merge_summary_lines(primary: list[str], fallback: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for line in [*primary, *fallback]:
        text = _clean_card_editorial_text(str(line or "")).strip(" .")
        key = re.sub(r"\W+", "", text).casefold()
        if not text or key in seen:
            continue
        merged.append(text)
        seen.add(key)
        if len(merged) >= _SUMMARY_LINE_MAX:
            break
    return merged


def _article_content_sentences(content: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(content or "")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?。！？다])\s+", normalized)
    sentences: list[str] = []
    for part in parts:
        text = part.strip(" -·,")
        if not (24 <= len(text) <= 180):
            continue
        if _looks_like_article_boilerplate(text):
            continue
        sentences.append(text)
        if len(sentences) >= _SUMMARY_LINE_MAX:
            break
    return sentences


def _looks_like_article_boilerplate(text: str) -> bool:
    return bool(
        re.search(
            r"뉴스\s*듣기|글자\s*크기|기사\s*공유|주소복사|다크모드|"
            r"무단전재|재배포\s*금지|저작권|기자\s*=|기자\s*$|"
            r"페이스북|카카오톡|이메일|구독|프린트",
            text,
            re.I,
        )
    )


def _first_list_item(value: Any) -> str:
    items = _list_string(value)
    return items[0] if items else ""


def _display_subject_from_summary(summary: dict[str, Any]) -> str:
    intelligence = summary.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        products = _list_string(intelligence.get("products_or_services"))
        if products:
            product = re.sub(r"\s+", " ", products[0]).strip(" .")
            amount = _first_amount_like_term(_list_string(intelligence.get("numbers_and_dates")))
            if amount and amount not in product:
                return _clean_subject_phrase(f"{amount} 규모의 {product}", summary)
            return _clean_subject_phrase(product, summary)
    subject = _first_text(
        summary.get("main_issue"),
        summary.get("main_event"),
        summary.get("headline"),
        summary.get("one_line_summary"),
        _first_list_item(summary.get("fact_summary")),
        "현재 이슈",
    )
    subject = re.sub(r"\s+", " ", subject).strip(" .")
    subject = re.sub(r"^(이번|해당)\s*", "", subject)
    return _clean_subject_phrase(subject, summary)


def _clean_subject_phrase(subject: str, summary: dict[str, Any]) -> str:
    value = re.sub(r"\s+", " ", str(subject or "")).strip(" .,")
    if not value:
        return "현재 이슈"
    company_names = _subject_company_names(summary)
    for company_name in company_names:
        escaped = re.escape(company_name)
        value = re.sub(
            rf"^{escaped}\s*(?:,|·|와|과|및|이|가|은|는)?\s*",
            "",
            value,
        ).strip(" ,.")
    value = re.sub(r"^(이번|해당)\s*", "", value).strip(" ,.")
    value = re.sub(r"^(?:,|·|와|과|및)\s*", "", value).strip(" ,.")
    value = re.sub(
        r"^(?:[가-힣A-Za-z0-9&._-]{2,30}(?:와|과|및)\s+){1,4}함께\s+",
        "",
        value,
    ).strip(" ,.")
    if not value:
        value = "현재 이슈"
    return value[:90] if len(value) > 90 else value


def _subject_company_names(summary: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for company_id in _list_string(summary.get("mentioned_peer_companies")):
        names.extend(
            [
                company_id,
                company_name_ko(company_id),
                global_company_name_ko(company_id),
            ]
        )
    main_company = str(summary.get("main_company") or "").strip()
    if main_company:
        names.extend(
            [
                main_company,
                company_name_ko(main_company),
                global_company_name_ko(main_company),
            ]
        )
    return [
        name
        for name in _unique_nonempty(names)
        if name and name not in {"피어사", "unknown", "None"}
    ]


def _first_amount_like_term(values: list[str]) -> str:
    for value in values:
        text = str(value or "").strip()
        if re.search(r"\d", text) and re.search(r"조|억|만|천|원", text):
            return text
    return ""


def _issue_execution_detail_phrase(summary: dict[str, Any]) -> str:
    intelligence = summary.get("cluster_fact_intelligence") or {}
    facts = " ".join(_list_string(summary.get("fact_summary")))
    numbers: list[str] = []
    if isinstance(intelligence, dict):
        numbers.extend(_list_string(intelligence.get("numbers_and_dates")))
    numbers.extend(
        re.findall(
            r"(?<![A-Za-z0-9])\d[\d,]*(?:조|억|만|천)?\s*(?:원|장|대|개|년|월|일)(?![A-Za-z0-9])",
            facts,
        )
    )
    cleaned_numbers: list[str] = []
    for value in numbers:
        text = re.sub(r"\s+", "", str(value or "")).strip(" ,.")
        if text and text not in cleaned_numbers:
            cleaned_numbers.append(text)
        if len(cleaned_numbers) >= 3:
            break
    details: list[str] = []
    if cleaned_numbers:
        details.append(", ".join(cleaned_numbers))
    if re.search(r"서비스\s*개시|오픈|운영\s*시작|구축\s*완료", facts):
        details.append("서비스 개시 또는 구축 완료 일정")
    if re.search(r"협약|계약|선정|컨소시엄|SPC|주주간", facts, flags=re.IGNORECASE):
        details.append("추진 구조")
    return ", ".join(details[:3])


def _skax_business_phrase_from_implication(skax: dict[str, Any]) -> str:
    mapped = _list_string(skax.get("business_line_mapping"))
    if mapped:
        return "·".join(mapped[:2])
    text = " ".join(
        [
            str(skax.get("why_important") or ""),
            str(skax.get("potential_impact") or ""),
            " ".join(_list_string(skax.get("recommended_actions"))),
        ]
    )
    patterns = (
        r"SK\s*AX의\s*([가-힣A-Za-z0-9&·/+_\-\s]{2,50}?)(?:\s*접점|과|와)",
        r"SK\s*AX도\s*([가-힣A-Za-z0-9&·/+_\-\s]{2,50}?)(?:\s*사업|역량)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        phrase = re.sub(r"\s+", " ", match.group(1)).strip(" ,.")
        if phrase:
            return phrase
    return ""


def _profile_phrase_from_linkage(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("profile_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_profile_areas")
    return _linkage_priority_phrase(records)


def _profile_reason_from_linkage(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("profile_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_profile_areas")
    reasons = _unique_nonempty(
        _first_text(record.get("why_relevant_to_issue"), record.get("reason")) for record in records
    )
    if reasons:
        return reasons[0]
    return ""


def _profile_linkage_level(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("profile_linkage") or {}
    if isinstance(linkage, dict):
        level = str(linkage.get("linkage_level") or "").strip().casefold()
        if level in {"high", "medium", "low", "none"}:
            return level
    return ""


def _skax_business_phrase_from_linkage(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("skax_response_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_skax_areas")
    return _linkage_priority_phrase(records, max_terms=2)


def _skax_reason_from_linkage(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("skax_response_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_skax_areas")
    reasons = _unique_nonempty(
        _first_text(record.get("why_relevant_to_issue"), record.get("reason")) for record in records
    )
    if reasons:
        return reasons[0]
    return ""


def _has_specific_skax_linkage(strategic_root: dict[str, Any]) -> bool:
    linkage = strategic_root.get("skax_response_linkage") or {}
    if not isinstance(linkage, dict):
        return False
    if _linkage_area_records(linkage, field="matched_skax_areas"):
        return True
    mode = str(linkage.get("response_mode") or "").strip()
    return mode == "profile_based_action"


def _linkage_area_records(linkage: Any, *, field: str) -> list[dict[str, Any]]:
    data = linkage if isinstance(linkage, dict) else {}
    records: list[dict[str, Any]] = []
    for item in _list_value(data.get(field)):
        if isinstance(item, dict):
            records.append(item)
    return records


def _linkage_priority_phrase(records: list[dict[str, Any]], *, max_terms: int = 3) -> str:
    products = _unique_nonempty(
        term
        for record in records
        for term in _linkage_record_terms(
            record,
            "matched_products_or_services",
            "products_or_services",
            "product_or_service",
            "services",
        )
    )
    if products:
        return _join_context_terms(products[:max_terms])

    capabilities = _unique_nonempty(
        term
        for record in records
        for term in _linkage_record_terms(
            record,
            "matched_capabilities",
            "profile_capability",
            "capability",
            "capabilities",
            "core_capabilities",
        )
    )
    if capabilities:
        return _join_context_terms(capabilities[:max_terms])

    business_areas = _unique_nonempty(
        _first_text(
            record.get("business_area"),
            record.get("profile_area_name"),
            record.get("name"),
        )
        for record in records
        if _record_specificity_level(record) == "business_area"
        and _business_area_record_has_issue_overlap(record)
    )
    if business_areas:
        return _join_context_terms(business_areas[:max_terms])

    business_lines = _unique_nonempty(record.get("business_line") for record in records)
    return _join_context_terms(business_lines[:1])


def _linkage_record_terms(record: dict[str, Any], *fields: str) -> list[str]:
    terms: list[str] = []
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and "," in value:
            terms.extend(part.strip() for part in value.split(",") if part.strip())
        else:
            terms.extend(_list_string(value))
    return _unique_nonempty(terms)


def _record_specificity_level(record: dict[str, Any]) -> str:
    level = str(record.get("specificity_level") or "").strip().casefold()
    if level:
        return level
    if _linkage_record_terms(record, "matched_products_or_services", "products_or_services"):
        return "product_or_service"
    if _linkage_record_terms(record, "matched_capabilities", "profile_capability", "capability"):
        return "core_capability"
    if _first_text(
        record.get("business_area"),
        record.get("profile_area_name"),
        record.get("name"),
    ):
        return "business_area"
    if record.get("business_line"):
        return "business_line"
    return "profile_context"


def _business_area_record_has_issue_overlap(record: dict[str, Any]) -> bool:
    area = _first_text(
        record.get("business_area"),
        record.get("profile_area_name"),
        record.get("name"),
    )
    if not area:
        return False
    matched_terms = {
        term
        for raw_term in _linkage_record_terms(record, "matched_issue_terms", "matched_terms")
        for term in _display_token_pieces(raw_term)
        if not _low_specificity_display_match_term(term)
    }
    area_terms = {
        term
        for raw_term in re.findall(
            r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·_-]{1,}",
            area,
        )
        for term in _display_token_pieces(raw_term)
        if not _low_specificity_display_match_term(term)
    }
    return bool(area_terms & matched_terms)


def _display_token_pieces(text: str) -> list[str]:
    pieces: list[str] = []
    for piece in re.split(r"[·/&+_\-\s]+", str(text or "")):
        normalized = _normalize_display_token(piece)
        if normalized and len(normalized) >= 2:
            pieces.append(normalized)
    return pieces


def _low_specificity_display_match_term(term: str) -> bool:
    return term in {
        "서비스",
        "사업",
        "지원",
        "기업",
        "관련",
        "기반",
        "summary",
        "역량",
        "도입",
        "제공",
        "확보",
    }


def _normalize_display_token(text: str) -> str:
    token = re.sub(r"\s+", "", str(text or "")).strip(" ,.")
    if len(token) > 3:
        token = re.sub(r"(으로|에서|에게|과|와|은|는|이|가|을|를|의)$", "", token)
    return token.casefold() if token.isascii() else token


def _linkage_has_concrete_profile_detail(records: list[dict[str, Any]]) -> bool:
    for record in records:
        if _linkage_record_terms(
            record,
            "matched_products_or_services",
            "products_or_services",
            "product_or_service",
        ):
            return True
        if _linkage_record_terms(
            record,
            "matched_capabilities",
            "profile_capability",
            "capability",
            "capabilities",
        ):
            return True
    return False


def _profile_specificity_note(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("profile_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_profile_areas")
    if records and not _linkage_has_concrete_profile_detail(records):
        return (
            " 다만 현재 확인되는 사업 정보는 상위 사업명 수준이므로, 세부 "
            "서비스·역량 또는 수행 범위는 후속 단계에서 확인되어야 합니다."
        )
    return ""


def _skax_specificity_note(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("skax_response_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_skax_areas")
    if records and not _linkage_has_concrete_profile_detail(records):
        return (
            " 다만 현재 확인되는 SK AX의 관련 사업은 상위 사업명 수준이므로, "
            "세부 수행 역량은 별도로 확인해야 합니다."
        )
    return ""


def _unique_nonempty(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" ,.")
        key = text.casefold()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def _join_context_terms(values: list[str]) -> str:
    terms = _unique_nonempty(values)
    return "·".join(terms[:4])


def _summary_fact_text(summary: dict[str, Any]) -> str:
    return " ".join(
        [
            str(summary.get("headline") or ""),
            str(summary.get("main_event") or ""),
            str(summary.get("one_line_summary") or ""),
            " ".join(_list_string(summary.get("fact_summary"))),
            " ".join(_list_string(summary.get("summary_lines"))),
            " ".join(
                _first_text(item.get("fact"))
                for item in _list_value(summary.get("consolidated_facts"))
                if isinstance(item, dict)
            ),
        ]
    )


def _compact_repeated_subject_in_phrase(phrase: str, subject: str) -> str:
    value = re.sub(r"\s+", " ", str(phrase or "")).strip()
    topic = re.sub(r"\s+", " ", str(subject or "")).strip()
    if not value or not topic:
        return value
    if not value.startswith(topic):
        return value
    rest = value[len(topic) :].lstrip()
    if not rest:
        return "해당 사업"
    if rest.startswith("의"):
        return f"해당 사업{rest}"
    if rest[:1] in {"은", "는", "이", "가", "을", "를", "과", "와"}:
        return f"해당 사업{rest}"
    return f"해당 사업 {rest}"


def _with_particle(text: str, consonant_particle: str, vowel_particle: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    return f"{value}{consonant_particle if _has_final_consonant(value[-1]) else vowel_particle}"


def _has_final_consonant(char: str) -> bool:
    value = str(char or "")[:1]
    if not value:
        return False
    code = ord(value)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def _ensure_card_sentence(text: str) -> str:
    sentence = _clean_card_editorial_text(text)
    if not sentence:
        return ""
    return sentence if sentence.endswith((".", "다.", "요.", "임.")) else f"{sentence}."


def _clean_card_editorial_text(text: str) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    out = _normalize_company_surface_names(out)
    out = re.sub(r"[!！]+$", "", out).strip()
    out = re.sub(r"^함께\s+", "", out)
    out = re.sub(r"\s+함께\s+(?=\d+[조억만천]|\d+장|[A-Z0-9]+ 서비스)", " ", out)
    return out.strip()


def _clean_frontend_ready_text(text: Any) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    out = _normalize_company_surface_names(out)
    out = _strip_public_section_prefixes(out)
    out = _polish_frontend_ready_internal_terms(out)
    out = re.sub(r"[!！]+$", "", out).strip()
    return out


def _polish_frontend_ready_internal_terms(text: str) -> str:
    out = str(text or "").strip()
    replacements = (
        (
            r"내부거래와\s*외부거래,\s*자체\s*수행과\s*외부\s*협력의\s*구분\s*기준",
            "내부거래와 외부거래의 구분 기준",
        ),
        (
            r"내부거래와\s*외부거래,\s*자체\s*수행과\s*외부\s*협력",
            "내부거래와 외부거래",
        ),
        (
            r"나눠\s*보며\s*검토\s*범위와\s*더\s*확인할\s*조건을\s*분리해야\s*합니다",
            "기준으로 검토 범위를 구분해야 합니다",
        ),
        (r"자사\s*관여\s*가능\s*영역\s*과", "검토 범위와"),
        (r"자사\s*관여\s*가능\s*영역", "검토 범위"),
        (r"추가\s*검증이\s*필요한\s*조건", "더 확인할 조건"),
        (r"추가\s*검증\s*조건", "더 확인할 조건"),
        (r"입력\s*근거", "기사 근거"),
        (r"\banchor\b|앵커", "근거 표현"),
        (r"이\s*기준이\s*있어야", "이 기준을 정리해야"),
        (r"기사\s*안에서\s*확인됩니다", "기사에서 확인됩니다"),
        (r"점검\s*항목", "점검 기준"),
        (r"비교\s*항목", "비교 기준"),
        (r"검증\s*항목", "검증 기준"),
        (r"관리\s*항목", "관리 기준"),
        (
            r"나눠\s*보며\s*검토\s*범위와\s*더\s*확인할\s*조건을\s*분리해야\s*합니다",
            "기준으로 검토 범위를 구분해야 합니다",
        ),
    )
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _strip_public_section_prefixes(text: str) -> str:
    out = str(text or "").strip()
    out = re.sub(
        r"^(?:핵심\s*(?:시사점|대응)|근거\s*[/／]?\s*설명)\s*[:：]\s*",
        "",
        out,
    ).strip()
    return out


def _normalize_company_surface_names(text: str) -> str:
    normalized = text
    for pattern, replacement in _company_surface_replacements():
        normalized = re.sub(pattern, replacement, normalized, flags=re.I)
    return normalized


def _company_surface_replacements() -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for company_id, aliases in COMPANY_ALIASES.items():
        display = company_name_ko(company_id)
        for alias in aliases:
            pattern = _surface_alias_pattern(alias)
            if pattern:
                replacements.append((pattern, display))
    for company_id, aliases in GLOBAL_COMPANY_ALIASES.items():
        display = global_company_name_ko(company_id)
        for alias in aliases:
            pattern = _surface_alias_pattern(alias)
            if pattern:
                replacements.append((pattern, display))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    return replacements


def _surface_alias_pattern(alias: Any) -> str:
    value = re.sub(r"\s+", " ", str(alias or "").strip())
    if not value:
        return ""
    return re.escape(value).replace(r"\ ", r"\s*")


def _profile_phrase_from_peer_copy(text: str) -> str:
    source = re.sub(r"\s+", " ", str(text or "")).strip()
    patterns = (
        r"프로필(?:상|에서는)?\s*([가-힣A-Za-z0-9&·/+_\-\s]{2,70}?)(?:가|이)\s",
        r"피어의\s*([가-힣A-Za-z0-9&·/+_\-\s]{2,70}?)(?:\s*접점|과|와)",
    )
    for pattern in patterns:
        match = re.search(pattern, source)
        if not match:
            continue
        phrase = re.sub(r"\s+", " ", match.group(1)).strip(" ,.")
        phrase = re.sub(r"^(에서는|상)\s*", "", phrase)
        if phrase and not re.search(r"^(이번|현재|따라서|확인된)$", phrase):
            return phrase
    return ""


def _select_display_items(
    candidates: list[dict[str, Any]],
    *,
    sentence_grounding: dict[str, Any] | None,
    summary: dict[str, Any] | None = None,
    strategic_root: dict[str, Any] | None = None,
    preferred: int = _DISPLAY_ITEM_PREFERRED,
    max_items: int = _DISPLAY_ITEM_MAX,
    min_items: int = _DISPLAY_ITEM_MIN,
    section_type: str = "",
) -> list[str]:
    del preferred
    ranked = sorted(
        candidates,
        key=lambda item: _optional_int(item.get("priority")) or 0,
        reverse=True,
    )
    selected: list[str] = []
    seen: set[str] = set()
    seen_roles: set[str] = set()
    for candidate in ranked:
        text = _clean_display_section_text(
            candidate.get("text"),
            section_type=section_type,
        )
        if not text:
            continue
        if _display_internal_term_violation(text):
            continue
        semantic_role = str(candidate.get("semantic_role") or "").strip()
        path = str(candidate.get("path") or "")
        if section_type == "insight" and _is_summary_repeat_insight(text, summary or {}):
            is_editorial_peer_meaning = (
                semantic_role == "peer_business_meaning" and path.startswith("card_editorial.")
            )
            if not is_editorial_peer_meaning:
                continue
        if not candidate.get("allow_without_grounding") and _is_internal_analysis_copy(text):
            continue
        if semantic_role and semantic_role in seen_roles:
            continue
        if _section_role_violation(section_type, text):
            continue
        if candidate.get(
            "allow_without_grounding"
        ) and not _editorial_candidate_has_minimum_grounding(
            text,
            summary=summary or {},
            strategic_root=strategic_root or {},
            section_type=section_type,
        ):
            continue
        if not candidate.get("allow_without_grounding") and not _is_grounded_display_candidate(
            text,
            path,
            sentence_grounding or {},
            allow_watch_point=bool(candidate.get("allow_watch_point")),
        ):
            continue
        key = _detail_line_key(text)
        if not key or key in seen or _is_near_duplicate_detail(text, selected):
            continue
        selected.append(text)
        seen.add(key)
        if semantic_role:
            seen_roles.add(semantic_role)
        if len(selected) >= max_items:
            break
    return selected if len(selected) >= min_items else []


def _is_internal_analysis_copy(text: str) -> bool:
    return bool(
        re.search(
            r"피어\s*프로필|프로필의|프로필상|맥락이|연결되는\s*신호|"
            r"관찰할\s*수\s*있|해석할\s*수\s*있|변화\s*신호|"
            r"현재\s*사건의\s*구축\s*범위와\s*추진\s*구조",
            str(text or ""),
        )
    )


def _is_summary_repeat_insight(text: str, summary: dict[str, Any]) -> bool:
    line = re.sub(r"\s+", " ", str(text or "")).strip()
    if not line:
        return False
    first_sentence = re.split(r"(?<=[.!?。！？])\s+", line, maxsplit=1)[0]
    summary_lines = _literal_summary_lines(summary)
    for summary_line in summary_lines:
        if _summary_lines_too_similar(first_sentence, summary_line):
            return True
    return False


def _clean_display_section_text(value: Any, *, section_type: str) -> str:
    text = _clean_card_editorial_text(value)
    text = _fix_display_particle_spacing(text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    if section_type == "insight":
        text = re.sub(
            r"\bSK\s*AX\b[^.。!?！？]*(?:점검|보완|대응|모니터링)[^.。!?！？]*[.。!?！？]?",
            "",
            text,
        )
    return _ensure_card_sentence(text.strip())


def _fix_display_particle_spacing(text: str) -> str:
    def replace_confirmed(match: re.Match[str]) -> str:
        term = match.group(1)
        particle = "이" if _has_final_consonant(term[-1]) else "가"
        return f"{term}{particle} 확인"

    return re.sub(
        r"([가-힣A-Za-z0-9&·+_-]{2,})[이가]\s+확인",
        replace_confirmed,
        str(text or ""),
    )


def _section_role_violation(section_type: str, text: str) -> bool:
    value = str(text or "")
    if section_type == "insight":
        return bool(
            re.search(
                r"SK\s*AX|우리\s*회사|자사|내부적으로|점검해야|구분해야|보완해야|"
                r"대응해야|모니터링해야|확인해야|맞춰야",
                value,
            )
        )
    if section_type == "action":
        return not bool(
            re.search(
                r"SK\s*AX|자사|내부적으로|점검|보완|구분|대응|모니터링|비교",
                value,
            )
        )
    return False


def _display_internal_term_violation(text: str) -> bool:
    return bool(
        re.search(
            r"프로필\s*근거|프로필\s*접점|profile|linkage|현재\s*입력에서|matched_|"
            r"business_novelty_status",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def _editorial_candidate_has_minimum_grounding(
    text: str,
    *,
    summary: dict[str, Any],
    strategic_root: dict[str, Any],
    section_type: str,
) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    issue_tokens = _grounding_tokens(_summary_fact_text(summary))
    issue_overlap = not issue_tokens or _has_token_overlap(value, issue_tokens)
    if section_type == "insight":
        profile_tokens = _grounding_tokens(_linkage_text(strategic_root.get("profile_linkage")))
        if not issue_overlap and not _has_token_overlap(value, profile_tokens):
            return False
        if profile_tokens and _has_token_overlap(value, profile_tokens):
            return True
        return bool(re.search(r"관찰\s*신호|변화\s*신호|후속|프로필|기존\s*사업", value))
    if section_type == "action":
        if not re.search(r"SK\s*AX|자사", value):
            return False
        skax_tokens = _grounding_tokens(_linkage_text(strategic_root.get("skax_response_linkage")))
        skax_overlap = bool(skax_tokens and _has_token_overlap(value, skax_tokens))
        if not issue_overlap and not skax_overlap:
            return False
        if skax_overlap:
            return True
        return bool(re.search(r"현재\s*입력|보완|모니터링|내부|점검|구분", value))
    return True


def _has_token_overlap(text: str, tokens: set[str]) -> bool:
    if not tokens:
        return True
    return bool(_grounding_tokens(text) & tokens)


def _grounding_tokens(text: str) -> set[str]:
    tokens = _summary_similarity_tokens(str(text or ""))
    return {token for token in tokens if len(token) >= 2}


def _linkage_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if key in {"matched_profile_areas", "matched_skax_areas"}:
                for record in _list_value(item):
                    if isinstance(record, dict):
                        parts.extend(str(v) for v in record.values() if v)
            elif isinstance(item, (str, int, float)):
                parts.append(str(item))
            elif isinstance(item, list):
                parts.extend(str(v) for v in item if isinstance(v, (str, int, float)))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_linkage_text(item) for item in value)
    return str(value or "")


def _is_grounded_display_candidate(
    text: str,
    path: str,
    sentence_grounding: dict[str, Any],
    *,
    allow_watch_point: bool = False,
) -> bool:
    del text
    entries = [
        entry for entry in _list_value(sentence_grounding.get("entries")) if isinstance(entry, dict)
    ]
    if not entries:
        return True
    matched = [
        entry for entry in entries if _grounding_path_matches(str(entry.get("path") or ""), path)
    ]
    if not matched:
        return allow_watch_point
    if allow_watch_point:
        return True
    return any(
        str(entry.get("grounding_type") or "") in {"fact", "profile", "fact+profile"}
        and not entry.get("needs_review")
        for entry in matched
    )


def _grounding_path_matches(entry_path: str, candidate_path: str) -> bool:
    if not entry_path or not candidate_path:
        return False
    return (
        entry_path == candidate_path
        or entry_path.startswith(f"{candidate_path}.")
        or candidate_path.startswith(f"{entry_path}.")
    )


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if _is_untranslated_english_text(text):
            continue
        if text:
            return text
    return ""


def _bounded_detail_lines(*values: Any) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for value in values:
        for text in _detail_line_candidates(value):
            if _is_untranslated_english_text(text):
                continue
            key = _detail_line_key(text)
            if text and key and key not in seen and not _is_near_duplicate_detail(text, lines):
                lines.append(text)
                seen.add(key)
            if len(lines) >= _CARD_DETAIL_MAX:
                return lines
    return lines[:_CARD_DETAIL_MAX]


def _bounded_detail_items(*values: Any) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for value in values:
        for text in _list_string(value):
            line = re.sub(r"\s+", " ", text).strip()
            if _is_untranslated_english_text(line):
                continue
            key = _detail_line_key(line)
            if line and key and key not in seen and not _is_near_duplicate_detail(line, lines):
                lines.append(line)
                seen.add(key)
            if len(lines) >= _CARD_DETAIL_MAX:
                return lines
    return lines[:_CARD_DETAIL_MAX]


def _detail_line_key(text: str) -> str:
    key = re.sub(r"[\s.。!?！？,，]+", "", str(text or "")).strip()
    key = re.sub(r"(합니다|해야합니다|있습니다|됩니다|입니다)$", "", key)
    return key


def _is_untranslated_english_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    hangul_count = len(re.findall(r"[가-힣]", value))
    english_words = re.findall(r"[A-Za-z]{3,}", value)
    if hangul_count == 0 and len(english_words) >= 5:
        return True
    alpha_count = len(re.findall(r"[A-Za-z]", value))
    return len(value) >= 60 and alpha_count > max(12, hangul_count * 3)


def _is_near_duplicate_detail(text: str, existing_lines: list[str]) -> bool:
    tokens = set(re.findall(r"[가-힣A-Za-z0-9&·+_-]{2,}", text or ""))
    if not tokens:
        return False
    for line in existing_lines:
        other = set(re.findall(r"[가-힣A-Za-z0-9&·+_-]{2,}", line or ""))
        if not other:
            continue
        overlap = len(tokens & other) / max(1, min(len(tokens), len(other)))
        if overlap >= 0.6:
            return True
    return False


def _detail_line_candidates(value: Any) -> list[str]:
    candidates: list[str] = []
    for item in _list_string(value):
        split_items = [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"(?<=[.!?。！？])\s+|(?<=다)\.\s*", item)
            if part.strip()
        ]
        candidates.extend(split_items or [re.sub(r"\s+", " ", item).strip()])
    return candidates


def _actionize_detail_lines(*values: Any) -> list[str]:
    actions: list[str] = []
    for text in _bounded_detail_lines(*values):
        stripped = text.rstrip(".。!?！？ ").strip()
        if not stripped:
            continue
        if _looks_like_action(stripped):
            action = stripped
        elif text.strip().endswith(("?", "？")):
            action = f"{stripped}를 확인합니다"
        else:
            action = _follow_up_action_from_statement(stripped)
        if not action.endswith((".", "。")):
            action += "."
        actions.append(action)
    return actions


def _follow_up_action_from_statement(text: str) -> str:
    subject = _statement_to_check_subject(text)
    if subject.endswith(("는지", "인지", "한지", "할지")):
        return f"후속 검토에서 {subject} 확인합니다"
    return f"후속 검토에서 {subject} 여부를 확인합니다"


def _statement_to_check_subject(text: str) -> str:
    subject = text.rstrip(".。!?！？ ").strip()
    replacements = (
        (r"할\s*수\s*있습니다$", "할 수 있는지"),
        (r"될\s*수\s*있습니다$", "될 수 있는지"),
        (r"가능성이\s*있습니다$", "가능성이 있는지"),
        (r"필요가\s*있습니다$", "필요한지"),
        (r"해야\s*합니다$", "해야 하는지"),
        (r"합니다$", "하는지"),
        (r"있습니다$", "있는지"),
        (r"입니다$", "인지"),
    )
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, subject)
        if updated != subject:
            return updated
    return subject


def _looks_like_action(text: str) -> bool:
    return bool(
        re.search(
            r"점검|검토|확인|비교|분석|모니터링|관리|추적|정리|구조화|구분|보완|반영",
            text,
        )
    )


def _rich_sources(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, article in enumerate(articles, start=1):
        title = str(article.get("title") or "").strip()
        url = str(article.get("url") or "").strip()
        if not title and not url:
            continue
        source = {
            "index": index,
            "article_id": _optional_int(article.get("id")),
            "title": title,
            "url": url,
            "archive_url": article.get("archive_url"),
            "source_name": str(article.get("source_name") or article.get("publisher") or ""),
            "published_at": _string_or_none(article.get("published_at")),
            "link_status": "ok",
        }
        image_urls = _article_image_urls(article)
        if image_urls:
            source["image_urls"] = image_urls
        sources.append(source)
    return sources


def _media_assets(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for article_index, article in enumerate(articles):
        article_id = _optional_int(article.get("id"))
        title = str(article.get("title") or "").strip()
        for image_index, url in enumerate(_article_image_urls(article)):
            if url in seen:
                continue
            seen.add(url)
            asset = {
                "id": f"img-{article_id or len(candidates) + 1}-{len(candidates) + 1}",
                "type": "image",
                "url": url,
                "alt": title or "뉴스 본문 이미지",
            }
            score = _image_asset_score(url=url, article=article, image_index=image_index)
            candidates.append((score, -article_index, asset))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [asset for _, _, asset in candidates]


def _image_asset_score(*, url: str, article: dict[str, Any], image_index: int) -> float:
    text = " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("content") or "")[:300],
            url,
        ]
    )
    compact = text.casefold()
    score = 10.0 - image_index * 0.25
    if image_index == 0:
        score += 1.0
    if re.search(r"현장|행사|간담회|협약|mou|체결|센터|데이터센터|공장|회의|대표|부장", text, re.I):
        score += 3.0
    if re.search(r"ai|ax|클라우드|데이터센터|보안|솔루션|플랫폼|로봇|공장", text, re.I):
        score += 1.5
    if re.search(r"주가|차트|목표가|목표주가|거래량|실적표|종목|증권", text):
        score -= 4.0
    if re.search(r"1x1|spacer|blank|placeholder|transparent|pixel", compact):
        score -= 8.0
    if re.search(r"cdn-cgi/image/fit=cover/?$", compact):
        score -= 6.0
    if re.search(r"[?&](?:w|width)=8[0-9]\\b|[?&](?:h|height)=5[0-9]\\b", compact):
        score -= 2.0
    return score


def _media_assets_for_cluster(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility shim for older callers/tests that name cluster-level media."""
    return _media_assets(articles)


def _article_image_urls(article: dict[str, Any]) -> list[str]:
    metadata = _metadata(article)
    candidates: list[str] = []
    for key in (
        "image_url",
        "thumbnail_url",
        "thumbnail",
        "og_image",
        "main_image",
        "image",
    ):
        value = article.get(key) or metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for key in ("image_urls", "images", "media_assets", "visual_images"):
        candidates.extend(_image_urls_from_value(article.get(key)))
        candidates.extend(_image_urls_from_value(metadata.get(key)))
    return _dedupe_keep_order([url for url in candidates if _is_image_reference(url)])


def _image_urls_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        urls: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
            elif isinstance(item, dict):
                urls.extend(
                    _list_string(
                        item.get("url")
                        or item.get("image_url")
                        or item.get("image_path")
                        or item.get("path")
                    )
                )
        return urls
    if isinstance(value, dict):
        return _list_string(
            value.get("url")
            or value.get("image_url")
            or value.get("image_path")
            or value.get("path")
        )
    return []


def _evidence_chain(
    summary: dict[str, Any],
    analysis: dict[str, Any],
    sources: list[dict[str, Any]],
    source_article_ids: list[int],
    cluster_id: int | None,
    created_at: str,
) -> dict[str, Any]:
    missing = _validation_missing(summary, analysis)
    return {
        "source_links": [
            {
                "article_id": source.get("article_id"),
                "title": source.get("title"),
                "source_name": source.get("source_name"),
                "url": source.get("url"),
            }
            for source in sources
        ],
        "provenance": {
            "raw_article_ids": source_article_ids,
            "cluster_id": cluster_id,
            "llm_model": analysis.get("model") or summary.get("model"),
            "prompt_version": _CARD_PROMPT_VERSION,
            "evidence_version": "v1.0",
            "run_at": created_at,
        },
        "financial_refs": [],
        "mbb_refs": [],
        "evidence_version": "v1.0",
        "fact_basis": _fact_basis(summary),
        "pass": _validation_pass(summary, analysis),
        "missing": missing,
    }


def _slides(
    title: str,
    summary_lines: list[str],
    insights: list[str],
    sources: list[dict[str, Any]],
    media_assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cover_image = media_assets[0]["url"] if media_assets else _DEFAULT_COVER_IMAGE_URL
    cover_alt = media_assets[0]["alt"] if media_assets else _DEFAULT_COVER_IMAGE_ALT
    slides = [
        {
            "order": 1,
            "title": title,
            "body": "\n".join(summary_lines[:_SUMMARY_LINE_MAX]) or None,
            "image_url": cover_image,
            "image_alt": cover_alt,
            "evidence_source_indexes": _source_indexes(sources),
            "layout_type": "summary",
        }
    ]
    if insights:
        slides.append(
            {
                "order": 2,
                "title": "의미 분석",
                "body": "\n".join(insights[:3]),
                "image_url": media_assets[1]["url"] if len(media_assets) > 1 else None,
                "image_alt": media_assets[1]["alt"] if len(media_assets) > 1 else None,
                "evidence_source_indexes": _source_indexes(sources),
                "layout_type": "implication",
            }
        )
    return slides


def _slides_from_display_sections(
    *,
    title: str,
    display_sections: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    media_assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not display_sections:
        return []
    source_indexes = _source_indexes(sources)
    cover_image = media_assets[0]["url"] if media_assets else _DEFAULT_COVER_IMAGE_URL
    cover_alt = media_assets[0]["alt"] if media_assets else _DEFAULT_COVER_IMAGE_ALT
    slides: list[dict[str, Any]] = []
    for index, section in enumerate(display_sections, start=1):
        items = _list_string(section.get("items"))
        if not items:
            continue
        section_type = str(section.get("type") or "").strip() or "section"
        section_title = str(section.get("title") or "").strip() or "카드뉴스"
        image_url = (
            cover_image
            if index == 1
            else (media_assets[index - 1]["url"] if len(media_assets) >= index else None)
        )
        image_alt = (
            cover_alt
            if index == 1
            else (media_assets[index - 1]["alt"] if len(media_assets) >= index else None)
        )
        slides.append(
            {
                "order": len(slides) + 1,
                "title": title if section_type == "summary" else section_title,
                "section_title": section_title,
                "body": "\n".join(items[:_DISPLAY_ITEM_MAX]) or None,
                "image_url": image_url,
                "image_alt": image_alt,
                "evidence_source_indexes": source_indexes,
                "layout_type": section_type,
            }
        )
    return slides


def _display_meta(sector: str, background_asset_url: str) -> dict[str, Any]:
    return {
        "home_carousel": True,
        "carousel_order": None,
        "slide_count": None,
        "visual_style": "editorial",
        "background_asset_url": background_asset_url,
        "accent_color": _accent_color(sector),
    }


def _source_indexes(sources: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index in (_optional_int(source.get("index")) for source in sources)
        if index is not None
    ]


def _source_article_ids(summary: dict[str, Any], articles: list[dict[str, Any]]) -> list[int]:
    raw_ids = _list_string(summary.get("source_article_ids"))
    article_ids = [
        article_id
        for article_id in (_optional_int(raw_id) for raw_id in raw_ids)
        if article_id is not None
    ]
    if article_ids:
        return article_ids
    return [
        article_id
        for article_id in (_optional_int(article.get("id")) for article in articles)
        if article_id is not None
    ]


def _published_date(articles: list[dict[str, Any]], created_at: str) -> str:
    for article in articles:
        value = _string_or_none(article.get("published_at"))
        if value:
            return value[:10]
    return created_at[:10]


def _subtitle(analysis: dict[str, Any], classification: dict[str, Any]) -> str:
    impact = str(analysis.get("impact_level") or "").strip()
    event_type = _normalize_event_type(classification.get("event_type"))
    return impact.upper() if impact else event_type


def _trust_score(articles: list[dict[str, Any]]) -> float | None:
    scores = [
        score
        for score in (_optional_float(article.get("credibility_score")) for article in articles)
        if score is not None
    ]
    return max(scores) if scores else None


def _validation_pass(summary: dict[str, Any], analysis: dict[str, Any]) -> bool:
    return (
        bool(summary.get("is_valid_summary", True))
        and not bool(summary.get("fact_extraction_failed"))
        and _summary_line_count_valid(summary)
        and not _missing_fact_basis_line_indexes(summary)
    ) and bool(analysis.get("is_valid_analysis", True))


def _validation_missing(summary: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    missing_indexes = _missing_fact_basis_line_indexes(summary)
    if missing_indexes:
        missing.append(
            "fact_basis missing for summary_line_index: "
            + ", ".join(str(index) for index in missing_indexes)
        )
    if not _summary_line_count_valid(summary):
        missing.append("summary_lines must contain 3~5 lines")
    if not bool(summary.get("is_valid_summary", True)):
        missing.append("summary is invalid")
    if bool(summary.get("fact_extraction_failed")):
        missing.append("fact extraction failed; rule-based candidates used")
    if not bool(analysis.get("is_valid_analysis", True)):
        missing.append("analysis is invalid")
    return missing


def _missing_fact_basis_line_indexes(summary: dict[str, Any]) -> list[int]:
    lines = _plain_summary_lines(summary)
    if not _SUMMARY_LINE_MIN <= len(lines) <= _SUMMARY_LINE_MAX:
        return []
    expected = set(range(1, len(lines) + 1))
    present: set[int] = set()
    for item in summary.get("fact_basis", []) or []:
        if not isinstance(item, dict):
            continue
        index = _optional_int(item.get("summary_line_index", item.get("summary_sentence_index")))
        if index in expected and _list_string(item.get("source_article_ids")):
            present.add(index)
    return sorted(expected - present)


def _summary_line_count_valid(summary: dict[str, Any]) -> bool:
    return _SUMMARY_LINE_MIN <= len(_plain_summary_lines(summary)) <= _SUMMARY_LINE_MAX


def _validation_sc_score(
    summary: dict[str, Any],
    analysis: dict[str, Any],
    trust_score: float | None,
) -> float:
    for value in (
        analysis.get("validation_sc_score"),
        summary.get("validation_sc_score"),
        analysis.get("confidence"),
        summary.get("confidence"),
        trust_score,
    ):
        score = _optional_float(value)
        if score is not None:
            return score
    return 0.0


def _normalize_peer_id(value: Any) -> str | None:
    peer_id = str(value or "").strip()
    return peer_id or None


def _normalize_sector(value: Any) -> str:
    sector = str(value or "").strip()
    if sector == "deal":
        return "biz_area"
    return sector if sector in _FRONTEND_SECTOR_IDS else "other"


def _normalize_event_type(value: Any) -> str:
    event_type = str(value or "").strip()
    if event_type == "tech_release":
        return "technology_update"
    if event_type == "financial":
        return "earnings"
    if event_type in {"expansion", "company", "new_biz"}:
        return "general_update"
    if event_type == "ma":
        return "investment"
    if event_type == "personnel":
        return "organization"
    if event_type == "tech":
        return "technology_update"
    return event_type if event_type in _FRONTEND_EVENT_TYPES else "general_update"


def _infer_event_type(
    summary: dict[str, Any],
    classification: dict[str, Any],
    card_text: str,
) -> str:
    del card_text
    summary_event = _normalize_event_type(summary.get("cluster_event_type"))
    if summary_event not in {"general_update", "unknown"}:
        return summary_event

    classification_event = _normalize_event_type(classification.get("event_type"))
    if classification_event not in {"general_update", "unknown"}:
        return classification_event
    return "general_update"


def _infer_sectors(classification: dict[str, Any], card_text: str) -> list[str]:
    raw_sectors = _list_string(classification.get("sectors"))
    if not raw_sectors and classification.get("sector"):
        raw_sectors = [str(classification.get("sector"))]
    normalized = [_normalize_sector(sector) for sector in raw_sectors]
    normalized = [sector for sector in normalized if sector != "other"]
    matched = [_normalize_sector(sector) for sector in match_sectors(card_text)]
    matched = [sector for sector in matched if sector != "other"]
    candidates = _dedupe_keep_order([*normalized, *matched])
    return _dedupe_keep_order(_filter_false_positive_sectors(candidates, card_text)) or ["other"]


def _filter_false_positive_sectors(sectors: list[str], card_text: str) -> list[str]:
    return [
        sector
        for sector in sectors
        if sector != "other" and _sector_has_direct_evidence(sector, card_text)
    ]


def _sector_has_direct_evidence(sector: str, card_text: str) -> bool:
    config = SECTOR_KEYWORDS.get(sector if sector != "biz_area" else "deal")
    if not config:
        return False
    lowered = str(card_text or "").lower()
    return any(str(keyword or "").lower() in lowered for keyword in config.get("keywords", []))


def _normalized_importance_score(
    *,
    classification: dict[str, Any],
    card_text: str,
) -> float:
    score = _optional_float(
        classification.get("importance_score", classification.get("exposure_score"))
    )
    if score is None:
        band = _normalize_exposure_band(
            classification.get("importance", classification.get("exposure_band"))
        )
        score = {"high": 0.75, "medium": 0.6, "low": 0.45}.get(band, 0.5)
    score = max(0.0, min(score, 1.0))
    del card_text
    return round(score, 2)


def _importance_band(score: float | None) -> str:
    value = score if score is not None else 0.0
    if value >= 0.75:
        return "high"
    if value >= 0.55:
        return "medium"
    return "low"


def _signals(
    *,
    classification: dict[str, Any],
    summary: dict[str, Any],
    source_article_ids: list[int],
    card_text: str,
) -> dict[str, Any]:
    raw = classification.get("signals")
    signals = dict(raw) if isinstance(raw, dict) else {}
    cluster_size = _optional_int(signals.get("cluster_size")) or len(source_article_ids)
    signals["cluster_size"] = cluster_size
    signals["summary_scope"] = "single_article_summary" if cluster_size <= 1 else "cluster_summary"
    if cluster_size <= 1:
        signals["single_article_summary"] = True
    terms = _issue_terms(summary, card_text)
    if terms:
        signals["issue_signature"] = _issue_signature(terms)
        signals["near_duplicate_terms"] = terms
        if cluster_size <= 1:
            signals["near_duplicate_card_candidate"] = True
            signals["near_duplicate_reason"] = (
                "단일 기사 클러스터이며 핵심 제품/서비스 키워드가 있어 유사 카드 병합 검토 필요"
            )
    if summary.get("representative_id"):
        signals["representative_id"] = _optional_int(summary.get("representative_id"))
    return signals


def _issue_terms(summary: dict[str, Any], card_text: str) -> list[str]:
    del card_text
    terms: list[str] = []
    intelligence = summary.get("cluster_fact_intelligence")
    if isinstance(intelligence, dict):
        terms.extend(_list_string(intelligence.get("products_or_services")))
    cleaned = [_clean_issue_term(term) for term in terms]
    return _dedupe_keep_order([term for term in cleaned if _is_signature_noun_phrase(term)])[:8]


def _clean_issue_term(term: str) -> str:
    text = re.sub(r"\s+", " ", str(term or "").strip())
    text = re.sub(r"^[,.'\"‘’“”\s]+|[,.'\"‘’“”\s]+$", "", text)
    return text.strip(" ,.'\"‘’“”")


def _is_signature_noun_phrase(term: str) -> bool:
    text = str(term or "").strip()
    if len(text) < 2:
        return False
    if len(text) > 40:
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9+\-/]{1,20}", text):
        return True
    if re.fullmatch(r"[가-힣A-Za-z0-9·+\-/]+(?:\s+[가-힣A-Za-z0-9·+\-/]+){1,3}", text):
        return True
    if re.fullmatch(r"[가-힣A-Za-z0-9·+\-/]{2,20}", text):
        return True
    return False


def _issue_signature(terms: list[str]) -> str:
    normalized = sorted({_compact_ascii(term) for term in terms if _compact_ascii(term)})
    return "|".join(normalized[:5])


def _compact_ascii(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(text or "").lower())


def _card_text(
    title: str,
    summary_lines: list[str],
    summary: dict[str, Any],
    articles: list[dict[str, Any]],
) -> str:
    return " ".join(
        [
            title,
            " ".join(summary_lines),
            str(summary.get("one_line_summary") or ""),
            str(summary.get("main_event") or ""),
            " ".join(
                f"{article.get('title') or ''} {article.get('content') or ''}"
                for article in articles
            ),
        ]
    )


def _card_text_from_card(card: dict[str, Any]) -> str:
    return " ".join(
        [
            str(card.get("title") or ""),
            " ".join(_list_string(card.get("summary_lines"))),
            _json_like_text(card.get("news_summary")),
        ]
    )


def _json_like_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def _fact_basis(summary: dict[str, Any]) -> list[dict[str, Any]]:
    basis: list[dict[str, Any]] = []
    for item in summary.get("fact_basis", []) or []:
        if not isinstance(item, dict):
            continue
        raw_source_ids = _list_string(item.get("source_article_ids"))
        source_ids = [
            article_id
            for article_id in (_optional_int(value) for value in raw_source_ids)
            if article_id is not None
        ]
        evidence_texts = _list_string(item.get("evidence_texts"))
        basis_item: dict[str, Any] = {
            "summary_line_index": _optional_int(
                item.get("summary_line_index", item.get("summary_sentence_index"))
            ),
            "source_article_ids": source_ids,
            "evidence_text": (evidence_texts[0] if evidence_texts else str(item.get("fact") or "")),
            "evidence_type": _normalize_fact_basis_evidence_type(item.get("evidence_type")),
        }
        fact_ids = _list_string(item.get("fact_ids"))
        if fact_ids:
            basis_item["fact_ids"] = fact_ids
        basis.append(basis_item)
    return _dedupe_card_fact_basis(basis)


def _dedupe_card_fact_basis(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for item in items:
        line_index = _optional_int(item.get("summary_line_index"))
        evidence_text = str(item.get("evidence_text") or "").strip()
        source_ids = _list_string(item.get("source_article_ids"))
        if not line_index or not evidence_text:
            continue
        duplicate = False
        for existing in deduped:
            if _optional_int(existing.get("summary_line_index")) != line_index:
                continue
            if _list_string(existing.get("source_article_ids")) != source_ids:
                continue
            if str(existing.get("evidence_text") or "").strip() == evidence_text:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)
    return deduped


def _normalize_fact_basis_evidence_type(value: Any) -> str:
    evidence_type = str(value or "").strip()
    if evidence_type in _FACT_BASIS_EVIDENCE_TYPES:
        return evidence_type
    if evidence_type in _EVENT_TO_FACT_BASIS_TYPE:
        return _EVENT_TO_FACT_BASIS_TYPE[evidence_type]
    return "reported_fact"


def _card_signals(card: dict[str, Any]) -> dict[str, Any]:
    implication = card.get("implication")
    if isinstance(implication, dict) and isinstance(implication.get("signals"), dict):
        return dict(implication["signals"])
    if isinstance(card.get("signals"), dict):
        return dict(card["signals"])
    return {}


def _set_card_signals(card: dict[str, Any], signals: dict[str, Any]) -> None:
    if "signals" in card:
        card["signals"] = signals
    implication = card.get("implication")
    if isinstance(implication, dict):
        implication["signals"] = signals
    db_record = card.get("db_record")
    if isinstance(db_record, dict):
        db_implication = db_record.get("implication")
        if isinstance(db_implication, dict):
            db_implication["signals"] = signals


def _normalize_exposure_band(value: Any) -> str:
    band = str(value or "").strip()
    return band if band in {"high", "medium", "low"} else "medium"


def _accent_color(sector: str) -> str:
    return {
        "ax": "coral",
        "security": "blue",
        "infra": "green",
        "biz_area": "orange",
        "other": "gray",
    }.get(sector, "gray")


def _list_string(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _metadata(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("metadata") or article.get("extra") or {}
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _is_image_reference(value: str) -> bool:
    lower = value.lower()
    if lower.startswith(("http://", "https://", "/", "file://")):
        return True
    return lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".avif"))


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    if text:
        try:
            return datetime.fromisoformat(text).isoformat()
        except ValueError:
            pass
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def peer_company_label(company_id: str | None) -> str:
    if not company_id:
        return ""
    local_name = company_name_ko(company_id)
    return local_name if local_name != company_id else global_company_name_ko(company_id)


__all__ = [
    "CardNewsComposer",
    "mark_near_duplicate_card_candidates",
    "peer_company_label",
]
