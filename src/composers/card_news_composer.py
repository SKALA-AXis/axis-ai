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
from src.config.companies import company_name_ko
from src.config.global_companies import global_company_name_ko
from src.config.sectors import SECTOR_KEYWORDS, match_sectors
from src.db.article_store import _generate_card_id, get_articles_by_ids

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

_FRONTEND_PEER_IDS = {
    "samsung_sds",
    "lg_cns",
    "hyundai_autoever",
    "posco_dx",
}

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
    from langchain_openai import ChatOpenAI  # lazy: transformers 체인 회피

    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_completion_tokens=1024)
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
        peer_id = _normalize_peer_id(summary.get("main_company"))
        trust_score = _trust_score(articles)
        source_article_ids = _source_article_ids(summary, articles)
        created_at = _now_iso()
        published_date = _published_date(articles, created_at)

        title = _first_non_empty(
            summary.get("headline"),
            summary.get("one_line_summary"),
            analysis.get("analysis_summary"),
            "피어사 주요 뉴스",
        )
        summary_lines = _plain_summary_lines(summary, use_llm=True)
        card_text = _card_text(title, summary_lines, summary, articles)
        event_type = _infer_event_type(summary, classification, card_text)
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
                "id": _generate_card_id(company),
                "company": company,
                "cluster_id": cluster_id,
                "representative_id": representative_id,
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

    title = _first_non_empty(
        summary.get("headline"),
        summary.get("one_line_summary"),
        classification.get("title"),
        articles[0].get("title"),
    )

    card = {
        "id": _generate_card_id(effective_company),
        "company": effective_company,
        "cluster_id": cluster_id,
        "representative_id": representative_id,
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
        "title": card.get("title"),
        "summary_lines": card.get("summary_lines", []),
        "event_type": card.get("event_type", "tech"),
        "importance": card.get("importance", "low"),
        "importance_score": card.get("importance_score", 0.0),
        "implication": implication,
        "sources": card.get("sources", []),
        "validation_pass": validation_pass,
        "validation_sc_score": validation_sc_score,
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
    fallback = fallback or {}
    analysis = analysis or {}
    skax = implication.get("skax_implication") or {}
    peer = implication.get("peer_implication") or {}

    why_important = _first_text(
        skax.get("why_important") if isinstance(skax, dict) else None,
        peer.get("peer_meaning") if isinstance(peer, dict) else None,
        fallback.get("why_important"),
    )
    potential_impact = _first_text(
        skax.get("potential_impact") if isinstance(skax, dict) else None,
        fallback.get("potential_impact"),
    )
    peer_implications = _bounded_detail_items(
        peer.get("peer_meaning") if isinstance(peer, dict) else None,
        peer.get("capability_change") if isinstance(peer, dict) else None,
        analysis.get("market_signal") if isinstance(analysis, dict) else None,
        analysis.get("strategic_meaning") if isinstance(analysis, dict) else None,
    )
    if not peer_implications:
        peer_implications = _bounded_detail_items(fallback.get("key_implications"))
    response_directions = _bounded_detail_items(
        skax.get("recommended_actions") if isinstance(skax, dict) else None,
        implication.get("recommended_actions"),
        fallback.get("suggested_actions"),
    )
    follow_up = (
        _bounded_detail_lines(implication.get("follow_up_questions"))
        or _bounded_detail_lines(implication.get("watch_points"))
        or _bounded_detail_lines(fallback.get("follow_up_questions"))
    )
    suggested_actions = response_directions or _actionize_detail_lines(
        implication.get("watch_points"), implication.get("follow_up_questions")
    )
    confidence = (
        _optional_float(implication.get("confidence"))
        if implication.get("confidence") is not None
        else fallback.get("confidence")
    )
    payload: dict[str, Any] = {
        "why_important": why_important,
        "potential_impact": potential_impact,
        "key_implications": peer_implications,
        "peer_implications": peer_implications,
        "skax_implications": _bounded_detail_items(why_important, potential_impact),
        "response_directions": suggested_actions,
        "follow_up_questions": follow_up,
        "suggested_actions": suggested_actions,
        "confidence": confidence,
    }
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
    return payload


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
    insight_candidates = [
        *_editorial_insight_candidates_from_strategy(
            summary=summary,
            analysis=analysis,
            implication=implication,
            strategic_root=strategic_root,
        ),
        *_insight_candidates_from_strategy(
            analysis=analysis,
            implication=implication,
            strategic_root=strategic_root,
        ),
    ]
    insight_items = _select_display_items(
        insight_candidates,
        sentence_grounding=sentence_grounding,
        summary=summary,
        strategic_root=strategic_root,
        max_items=2,
        preferred=2,
        section_type="insight",
    )
    action_candidates = [
        *_editorial_action_candidates_from_strategy(
            summary=summary,
            implication=implication,
            strategic_root=strategic_root,
        ),
        *_action_candidates_from_strategy(
            implication=implication,
            strategic_root=strategic_root,
        ),
    ]
    action_items = _select_display_items(
        action_candidates,
        sentence_grounding=sentence_grounding,
        summary=summary,
        strategic_root=strategic_root,
        max_items=2,
        preferred=2,
        section_type="action",
    )
    return [
        {"type": "summary", "title": "요약", "items": summary_items},
        {"type": "insight", "title": "시사점", "items": insight_items},
        {"type": "action", "title": "대응방안", "items": action_items},
    ]


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
    insight_items = sections.get("insight") or []
    action_items = sections.get("action") or []
    if insight_items:
        payload["key_implications"] = insight_items
        payload["peer_implications"] = insight_items
        payload["potential_impact"] = insight_items[0]
        payload["why_important"] = insight_items[1] if len(insight_items) > 1 else insight_items[0]
    if action_items:
        payload["response_directions"] = action_items
        payload["suggested_actions"] = action_items
    return payload


def _sync_implication_frontend_from_display_sections(
    implication: Any,
    *,
    frontend: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(implication) if isinstance(implication, dict) else {}
    payload["frontend"] = frontend
    if frontend.get("suggested_actions"):
        payload["recommended_actions"] = frontend["suggested_actions"]
        skax = payload.get("skax_implication")
        if isinstance(skax, dict):
            skax["recommended_actions"] = frontend["suggested_actions"]
    return payload


def _editorial_insight_candidates_from_strategy(
    *,
    summary: dict[str, Any],
    analysis: dict[str, Any],
    implication: dict[str, Any],
    strategic_root: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recompose strategy output into card-news insight copy.

    StrategicInsightAgent keeps analysis fields separated for traceability. The
    card surface needs fewer, denser lines that show why the implication follows:
    confirmed fact -> peer profile touchpoint -> meaning of the change.
    """

    peer = implication.get("peer_implication") or {}
    if not isinstance(peer, dict):
        peer = {}
    peer_name = _first_text(peer.get("company_name_ko"), peer.get("company_id"), "피어사")
    peer_topic = _with_particle(peer_name, "은", "는")
    subject = _display_subject_from_summary(summary)
    profile_phrase = _profile_phrase_from_linkage(strategic_root) or _profile_phrase_from_peer_copy(
        _first_text(peer.get("peer_meaning"), peer.get("capability_change"))
    )
    profile_specificity_note = _profile_specificity_note(strategic_root)
    profile_level = _profile_linkage_level(strategic_root)
    detail_phrase = _issue_execution_detail_phrase(summary)
    execution_focus = _profile_connection_execution_phrase(summary)
    candidates: list[dict[str, Any]] = []
    if profile_phrase:
        if profile_level in {"low", "none"}:
            text = (
                f"{peer_topic} 이번 {subject}에서 {profile_phrase} 사업과의 접점이 "
                "확인됩니다. 다만 현재 확인되는 사업 정보만으로 피어사의 역할 확대나 수행 "
                "범위를 단정하기는 어렵기 때문에, 기존 사업 맥락과 이번 사건이 어디까지 "
                "연결되는지가 후속 단계의 핵심 변수로 남습니다."
            )
        else:
            detail_sentence = (
                f" 특히 {detail_phrase}까지 함께 확인되므로, 피어사의 기존 사업 기반이 "
                "실제 과제의 수행 조건과 어디까지 맞닿는지 보는 것이 중요합니다."
                if detail_phrase
                else (
                    " 피어사의 기존 사업 기반이 이번 사건의 수행 범위와 후속 역할 공개로 "
                    "어떻게 이어지는지 확인하는 것이 중요합니다."
                )
            )
            text = (
                f"이번 {subject}은 {peer_name}의 {profile_phrase} 사업 기반이 "
                f"{execution_focus}와 연결되는 장면입니다."
                f"{detail_sentence}{profile_specificity_note}"
            )
        candidates.append(
            {
                "text": text,
                "path": "card_editorial.insight.peer_business_state",
                "priority": 130,
                "allow_without_grounding": True,
                "semantic_role": "peer_business_meaning",
            }
        )
    else:
        primary_fact = _first_text(
            _first_list_item(summary.get("fact_summary")),
            summary.get("one_line_summary"),
            summary.get("main_event"),
            summary.get("headline"),
        )
        candidates.append(
            {
                "text": (
                    f"{_ensure_card_sentence(primary_fact)} 현재 확인되는 사업 정보만으로는 "
                    "피어사의 확정적 사업 확장까지 단정하기 어렵기 때문에, 이번 이슈는 "
                    "대상 사업과 후속 역할 공개 여부가 핵심 변수로 남는 관찰 신호입니다."
                ),
                "path": "card_editorial.insight.event_signal",
                "priority": 124,
                "allow_without_grounding": True,
                "semantic_role": "uncertainty_or_follow_up",
            }
        )

    market_signal = _first_text(analysis.get("market_signal"))
    criteria_phrase = _fact_based_competition_criteria_phrase(
        summary
    ) or _criteria_phrase_from_strategy(
        analysis=analysis,
        implication={},
    )
    if criteria_phrase:
        competition_variable = _event_frame_competition_variable(
            summary,
            peer_name=peer_name,
        )
        text = (
            f"유사 사업의 비교 기준도 단순 선정 여부보다 {criteria_phrase} 중심으로 "
            f"좁혀질 수 있습니다. 후속 단계에서 공개되는 {competition_variable}은 "
            "피어사의 실제 실행력을 판단하는 핵심 변수로 남습니다."
        )
    elif market_signal:
        text = _ensure_card_sentence(_clean_card_editorial_text(market_signal))
    else:
        text = ""
    if text:
        candidates.append(
            {
                "text": text,
                "path": "card_editorial.insight.market_signal",
                "priority": 126,
                "allow_without_grounding": True,
                "semantic_role": "competition_standard_change",
            }
        )

    return candidates


def _editorial_action_candidates_from_strategy(
    *,
    summary: dict[str, Any],
    implication: dict[str, Any],
    strategic_root: dict[str, Any],
) -> list[dict[str, Any]]:
    subject = _display_subject_from_summary(summary)
    skax = implication.get("skax_implication") or {}
    if not isinstance(skax, dict):
        skax = {}
    skax_phrase = _skax_business_phrase_from_linkage(strategic_root) or (
        _skax_business_phrase_from_implication(skax)
    )
    skax_specificity_note = _skax_specificity_note(strategic_root)
    has_specific_skax_link = _has_specific_skax_linkage(strategic_root)
    criteria_phrase = _fact_based_competition_criteria_phrase(
        summary
    ) or _criteria_phrase_from_strategy(
        analysis={},
        implication=implication,
    )
    if _generic_criteria_phrase(criteria_phrase):
        criteria_phrase = _default_strategy_criteria_phrase(summary)
    criteria_phrase = criteria_phrase or _default_strategy_criteria_phrase(summary)
    criteria_topic = _with_particle(criteria_phrase, "으로", "로")
    monitoring_phrase = (
        _fact_based_monitoring_phrase(summary)
        or _monitoring_phrase_from_strategic_root(strategic_root)
        or _monitoring_phrase_from_implication(implication)
    )
    monitoring_phrase = _normalize_monitoring_phrase_for_issue(summary, monitoring_phrase)
    monitoring_phrase = _compact_repeated_subject_in_phrase(monitoring_phrase, subject)
    monitoring_scope = _event_frame_action_monitoring_scope(summary)
    monitoring_sentence = ""
    if monitoring_phrase:
        monitoring_sentence = (
            f" 이후 {_with_particle(monitoring_phrase, '을', '를')} 모니터링해야 합니다."
        )
    candidates: list[dict[str, Any]] = []
    if skax_phrase:
        certainty_clause = (
            "자사 관련 사업 기반이 확인되더라도 "
            if has_specific_skax_link
            else (f"현재 SK AX의 관련 접점이 넓은 {skax_phrase} 사업명 수준으로 확인되는 만큼 ")
        )
        first = (
            f"SK AX는 {_with_particle(subject, '과', '와')} 유사한 사업을 검토할 때 "
            f"{_with_particle(skax_phrase, '과', '와')} 연결되는 자사 사업 기반이 "
            f"{criteria_phrase} 중 어디까지 감당할 수 있는지 먼저 구분해야 합니다. "
            f"{certainty_clause}확인 항목이 구체적일수록 관련 역량 보유 여부보다 "
            "수행 범위와 리스크 부담 가능 범위가 중요해집니다."
            f"{skax_specificity_note} 따라서 직접 담당 가능한 범위와 외부 보완이 "
            "필요한 범위를 구분해야 후속 사업에서 운영 책임과 리스크 수준을 "
            "현실적으로 판단할 수 있습니다."
        )
    else:
        first = (
            f"SK AX는 {subject}와 유사한 사업에서 현재 확인되는 사업 정보상 직접 연결되는 자사 "
            "사업 기반이 충분한지 먼저 확인해야 합니다. 확인 항목이 "
            f"{criteria_topic} 구체화되는 상황에서는 일반적인 대응 의지만으로는 "
            "수행 가능한 범위와 보완해야 할 영역이 드러나지 않습니다. 따라서 직접 "
            "담당 가능한 범위와 외부 보완이 필요한 범위를 나눠 내부 대응 범위를 "
            "정리해야 후속 사업에서 리스크와 운영 조건을 판단할 수 있습니다."
        )
    candidates.append(
        {
            "text": first,
            "path": "card_editorial.action.skax_gap_response",
            "priority": 130,
            "allow_without_grounding": True,
            "semantic_role": "skax_capability_gap_check",
        }
    )
    if monitoring_sentence:
        scope_sentence = (
            f" 이때 {monitoring_scope}도 함께 봐야 합니다."
            if monitoring_scope and monitoring_scope not in monitoring_phrase
            else ""
        )
        candidates.append(
            {
                "text": (
                    f"SK AX는{monitoring_sentence}{scope_sentence} 이 정보가 확인되어야 앞서 나눈 "
                    "내부 대응 범위, 외부 보완 필요성, 리스크 기준을 후속 상황에 맞게 "
                    "조정할 수 있습니다."
                ),
                "path": "card_editorial.action.monitoring_response",
                "priority": 126,
                "allow_without_grounding": True,
                "semantic_role": "monitoring_response",
            }
        )
    return candidates


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


def _default_strategy_criteria_phrase(summary: dict[str, Any]) -> str:
    return ", ".join(_event_frame_criteria(summary))


def _generic_criteria_phrase(text: str) -> bool:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" .,")
    if not value:
        return False
    if len(value) <= 4:
        return True
    return value in {
        "리스크",
        "범위",
        "책임",
        "일정",
        "검증",
        "운영 조건",
        "모니터링",
    }


def _fact_based_competition_criteria_phrase(summary: dict[str, Any]) -> str:
    facts = " ".join(
        [
            " ".join(_list_string(summary.get("fact_summary"))),
            " ".join(
                _first_text(item.get("fact"))
                for item in _list_value(summary.get("consolidated_facts"))
                if isinstance(item, dict)
            ),
        ]
    )
    frame = _event_frame(summary)
    criteria = _event_frame_criteria(summary)
    if re.search(r"GPU|반도체|서버|컴퓨팅|데이터센터|용량|\d[\d,]*\s*장", facts, re.I):
        criteria = _merge_front(criteria, ["자원 확보 규모", "구축 범위"])
    if re.search(r"서비스\s*개시|오픈|제공\s*개시|상용화", facts):
        criteria = _merge_front(criteria, ["서비스 개시 일정"])
    if re.search(r"협약|컨소시엄|SPC|주주간", facts, re.I):
        criteria = _merge_front(criteria, ["협약·운영 구조"])
    if re.search(r"기간|20\d{2}년|월\s*\d{1,2}일", facts):
        criteria = _merge_front(criteria, ["일정 조건"])
    if frame in {"contract", "performance"} and re.search(r"규모|금액|매출액|원", facts):
        criteria = _merge_front(criteria, ["사업 규모"])
    return ", ".join(dict.fromkeys(criteria[:5]))


def _event_frame(summary: dict[str, Any]) -> str:
    event_type = str(
        summary.get("cluster_event_type") or summary.get("event_type") or ""
    ).casefold()
    facts = _summary_fact_text(summary)
    if event_type in {"earnings", "stock_market", "analyst_report"}:
        return "performance"
    if event_type in {"hiring", "organization", "personnel"}:
        return "organization"
    if event_type in {"regulation", "risk"}:
        return "regulation_risk"
    if event_type in {"partnership", "mou"} or re.search(
        r"MOU|협력|협약|컨소시엄|SPC|공동\s*추진",
        facts,
        re.I,
    ):
        if re.search(r"선정|사업자|구축|GPU|인프라|센터", facts, re.I):
            return "selection_build"
        return "partnership"
    if event_type in {"contract", "investment"} or re.search(
        r"계약|수주|공급|투자|지분",
        facts,
    ):
        if re.search(r"전환|현대화|시스템|단말|플랫폼|마이그레이션", facts):
            return "contract_transition"
        return "contract"
    if event_type in {"launch", "technology_update", "tech"} or re.search(
        r"출시|공개|서비스\s*개시|오픈|상용화",
        facts,
    ):
        return "launch"
    if re.search(r"선정|사업자|구축|운영|운용|GPU|인프라|센터", facts, re.I):
        return "selection_build"
    return "general"


def _event_frame_criteria(summary: dict[str, Any]) -> list[str]:
    frame = _event_frame(summary)
    if frame == "selection_build":
        return ["자원 확보 규모", "구축 범위", "운영 책임", "일정 조건"]
    if frame == "contract_transition":
        return ["계약 범위", "대상 시스템", "전환 리스크", "운영 안정화"]
    if frame == "contract":
        return ["계약 범위", "상대방 역할", "수행 기간", "사업 규모"]
    if frame == "partnership":
        return ["역할 분담", "공동 추진 범위", "후속 협약", "책임 구조"]
    if frame == "launch":
        return ["적용 대상", "기능 범위", "고객군", "확산 조건"]
    if frame == "performance":
        return ["실적 기여 사업", "수익성", "지속성", "재무 영향 범위"]
    if frame == "organization":
        return ["집중 역량", "실행 준비", "조직 확대", "후속 채용·운영 계획"]
    if frame == "regulation_risk":
        return ["영향받는 사업영역", "대응 책임", "리스크 관리", "후속 규제 변화"]
    return ["적용 범위", "운영 책임", "일정 조건", "검증 기준"]


def _merge_front(base: list[str], additions: list[str]) -> list[str]:
    return list(dict.fromkeys([*additions, *base]))


def _issue_execution_focus_phrase(summary: dict[str, Any]) -> str:
    criteria = _fact_based_competition_criteria_phrase(summary)
    if criteria:
        return criteria
    subject = _display_subject_from_summary(summary)
    return f"{subject}의 실제 수행 조건"


def _profile_connection_execution_phrase(summary: dict[str, Any]) -> str:
    frame = _event_frame(summary)
    if frame == "selection_build":
        return "자원 확보와 구축 일정이 포함된 대형 인프라 과제"
    if frame == "contract_transition":
        return "계약 범위와 대상 시스템 전환 조건이 포함된 실행 과제"
    if frame == "contract":
        return "계약 범위와 수행 기간이 명시된 사업 과제"
    if frame == "partnership":
        return "역할 분담과 공동 추진 범위가 드러나는 협력 과제"
    if frame == "launch":
        return "적용 대상과 기능 범위가 공개되는 서비스 과제"
    if frame == "performance":
        return "실적 기여 사업과 지속성을 확인해야 하는 경영 신호"
    if frame == "organization":
        return "집중 역량과 실행 준비 수준을 확인해야 하는 조직 신호"
    if frame == "regulation_risk":
        return "영향받는 사업영역과 대응 책임을 확인해야 하는 리스크 신호"
    return f"{_display_subject_from_summary(summary)}의 실제 수행 조건"


def _event_frame_followup_focus(summary: dict[str, Any], *, peer_name: str) -> str:
    frame = _event_frame(summary)
    if frame == "selection_build":
        return (
            f"{peer_name}가 선정 이후 어떤 구축·운영 범위와 서비스 책임을 실제로 맡는지 확인하는 데"
        )
    if frame == "contract_transition":
        return (
            f"{peer_name}가 계약 이후 어떤 대상 시스템, 전환 범위, 운영 안정화 "
            "책임을 맡는지 확인하는 데"
        )
    if frame == "contract":
        return (
            f"{peer_name}의 계약상 역할, 수행 기간, 사업 규모가 후속 실행에서 "
            "어떻게 확정되는지 확인하는 데"
        )
    if frame == "partnership":
        return (
            f"{peer_name}의 역할 분담, 공동 추진 범위, 후속 협약 구조가 어떻게 "
            "구체화되는지 확인하는 데"
        )
    if frame == "launch":
        return (
            f"{peer_name}의 적용 대상, 기능 범위, 고객군 확산 조건이 실제로 "
            "어떻게 나타나는지 확인하는 데"
        )
    if frame == "performance":
        return f"{peer_name}의 실적 기여 사업, 수익성, 지속성이 후속 지표에서 확인되는지 보는 데"
    if frame == "organization":
        return (
            f"{peer_name}의 집중 역량, 실행 준비, 조직 확대가 후속 운영으로 이어지는지 확인하는 데"
        )
    if frame == "regulation_risk":
        return (
            f"{peer_name}의 영향받는 사업영역, 대응 책임, 리스크 관리 범위가 "
            "구체화되는지 확인하는 데"
        )
    return f"{peer_name}의 후속 수행 범위와 책임이 어떻게 확정되는지 확인하는 데"


def _event_frame_competition_variable(summary: dict[str, Any], *, peer_name: str) -> str:
    frame = _event_frame(summary)
    if frame == "selection_build":
        return f"{peer_name}의 구축·운영 범위와 서비스 책임"
    if frame == "contract_transition":
        return f"{peer_name}의 대상 시스템, 전환 범위, 운영 안정화 책임"
    if frame == "contract":
        return f"{peer_name}의 계약상 역할, 수행 기간, 사업 범위"
    if frame == "partnership":
        return f"{peer_name}의 역할 분담, 공동 추진 범위, 후속 협약 구조"
    if frame == "launch":
        return f"{peer_name}의 적용 대상, 기능 범위, 고객군 확산 조건"
    if frame == "performance":
        return f"{peer_name}의 실적 기여 사업, 수익성, 지속성"
    if frame == "organization":
        return f"{peer_name}의 집중 역량, 실행 준비, 조직 확대 범위"
    if frame == "regulation_risk":
        return f"{peer_name}의 영향받는 사업영역, 대응 책임, 리스크 관리 범위"
    return f"{peer_name}의 후속 수행 범위와 책임"


def _event_frame_action_monitoring_scope(summary: dict[str, Any]) -> str:
    frame = _event_frame(summary)
    if frame == "selection_build":
        return "사업자별 구축·운영 책임 범위와 추가 협약 구조"
    if frame == "contract_transition":
        return "대상 시스템별 전환 범위, 운영 안정화 책임, 전환 리스크"
    if frame == "contract":
        return "계약상 역할, 수행 범위, 책임 구조"
    if frame == "partnership":
        return "역할 분담, 공동 추진 범위, 후속 협약 구조"
    if frame == "launch":
        return "적용 대상, 기능 범위, 고객군 확산 조건"
    if frame == "performance":
        return "실적 기여 사업, 수익성, 지속성"
    if frame == "organization":
        return "집중 역량, 실행 준비 수준, 조직 확대 범위"
    if frame == "regulation_risk":
        return "영향받는 사업영역, 대응 책임, 리스크 관리 범위"
    return "후속 수행 범위와 책임 구조"


def _monitoring_phrase_from_implication(implication: dict[str, Any]) -> str:
    watch_points = _list_string(implication.get("watch_points"))
    if watch_points:
        cleaned = [phrase for item in watch_points if (phrase := _monitoring_phrase_cleanup(item))]
        if cleaned:
            return ", ".join(cleaned[:2])
    skax = implication.get("skax_implication") or {}
    text = " ".join(_list_string(skax.get("recommended_actions"))) if isinstance(skax, dict) else ""
    candidates: list[str] = []
    for term in (
        "사업자 선정",
        "협약",
        "서비스 개시",
        "구축 완료",
        "운영 책임",
        "추가 계약",
        "후속 일정",
    ):
        if term in text:
            candidates.append(term)
    if candidates:
        return "·".join(dict.fromkeys(candidates[:4])) + " 여부"
    return "후속 역할 공개와 실제 수행 범위"


def _monitoring_phrase_from_strategic_root(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("skax_response_linkage") or {}
    if not isinstance(linkage, dict):
        return ""
    cleaned = [
        phrase
        for item in _list_string(linkage.get("monitoring_points"))
        if (phrase := _monitoring_phrase_cleanup(item))
    ]
    return ", ".join(cleaned[:2])


def _fact_based_monitoring_phrase(summary: dict[str, Any]) -> str:
    facts = _summary_fact_text(summary)
    candidates: list[str] = []
    if re.search(r"GPU|반도체|서버|컴퓨팅", facts, re.I) and re.search(
        r"입고|구축|완료|확보",
        facts,
    ):
        candidates.append("GPU 입고·구축 완료 시점")
    for service_name in _service_start_names(facts):
        candidates.append(f"{service_name} 서비스 개시 일정")
    if not any("서비스 개시" in item for item in candidates) and re.search(
        r"서비스\s*(?:개시|시작|오픈)|상용화",
        facts,
    ):
        candidates.append("서비스 개시 일정")
    if re.search(r"협약|계약|컨소시엄|SPC|주주간", facts, re.I):
        candidates.append("협약·계약 구조")
    if re.search(r"운영|운용|운용지원|운영지원|책임", facts):
        candidates.append("운영 책임 범위")
    return ", ".join(dict.fromkeys(candidates[:4]))


def _service_start_names(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(
        r"([A-Za-z0-9가-힣][A-Za-z0-9가-힣·+_-]{1,30})\s*서비스(?:를|가|는|의)?\s*"
        r"(?:[가-힣A-Za-z0-9\s]{0,30})?(?:개시|시작|오픈|상용화)",
        text,
    ):
        name = match.group(1).strip(" ,.")
        if name and name not in names:
            names.append(name)
    return names[:3]


def _normalize_monitoring_phrase_for_issue(summary: dict[str, Any], phrase: str) -> str:
    value = re.sub(r"\s+", " ", str(phrase or "")).strip(" ,.")
    if not value:
        return value
    facts = _summary_fact_text(summary)
    if re.search(r"최종\s*선정|사업자(?:로)?\s*선정|참여\s*기업(?:으로)?\s*선정", facts):
        value = re.sub(r"후속\s*선정[·,\s/]*", "후속 ", value)
        value = re.sub(r"사업자\s*선정[·,\s/]*", "", value)
    return re.sub(r"\s+", " ", value).strip(" ,.")


def _monitoring_phrase_cleanup(text: str) -> str:
    out = _clean_card_editorial_text(text).rstrip(".。!?！？ ")
    out = re.sub(r"\s*(확인|모니터링)(합니다|해야\s*합니다|한다|할\s*필요가\s*있습니다)$", "", out)
    out = re.sub(r"구체화되는지$", "구체화 여부", out)
    out = re.sub(r"확인되는지$", "확인 여부", out)
    out = re.sub(r"공개되는지$", "공개 여부", out)
    out = re.sub(r"되는지$", " 여부", out)
    out = re.sub(r"(구체화|확인|공개)여부", r"\1 여부", out)
    out = re.sub(r"(.+?)(이|가)\s+구체화 여부$", r"\1 구체화 여부", out)
    out = re.sub(r"(.+?)(이|가)\s+원문 근거로 확인 여부$", r"\1의 원문 근거 확인 여부", out)
    out = re.sub(r"\s+", " ", out).strip(" ,.")
    return out


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
    out = re.sub(r"^함께\s+", "", out)
    out = re.sub(r"\s+함께\s+(?=\d+[조억만천]|\d+장|[A-Z0-9]+ 서비스)", " ", out)
    out = re.sub(r"피어\s*프로필에서는\s*", "현재 확인되는 피어 사업 정보상 ", out)
    out = re.sub(r"SK\s*AX\s*프로필에서는\s*", "현재 확인되는 SK AX 사업 정보상 ", out)
    return out.strip()


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


def _criteria_phrase_from_strategy(
    *,
    analysis: dict[str, Any],
    implication: dict[str, Any],
) -> str:
    text = " ".join(
        [
            str(analysis.get("market_signal") or ""),
            str(analysis.get("impact_reason") or ""),
            " ".join(_list_string(analysis.get("strategic_meaning"))[:5]),
            " ".join(
                _list_string(
                    (implication.get("skax_implication") or {}).get("recommended_actions")
                )[:5]
                if isinstance(implication.get("skax_implication"), dict)
                else []
            ),
        ]
    )
    criteria = []
    for term in (
        "구축 범위",
        "운영 책임",
        "운영 체계",
        "단계별 일정",
        "일정 조건",
        "용량 기준",
        "장애 대응",
        "보안·권한 통제",
        "검증 기준",
        "리스크",
        "후속 모니터링",
    ):
        if term in text and term not in criteria:
            criteria.append(term)
        if len(criteria) >= 4:
            break
    return ", ".join(criteria)


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
    replacements = {
        r"현재\s*입력에서": "현재 확인된 근거상",
        r"현재\s*입력": "현재 확인된 근거",
        r"프로필\s*근거": "현재 확인되는 사업 정보",
        r"프로필\s*접점": "관련 사업 기반",
        r"프로필상": "현재 확인되는 사업 정보상",
        r"피어\s*프로필": "피어 사업 정보",
        r"SK\s*AX\s*프로필": "SK AX 사업 정보",
        r"관찰\s*지점": "후속 확인이 필요한 신호",
        r"해석하는\s*것이\s*안전합니다": "단정하기보다 후속 확인이 필요합니다",
        r"제안서": "구조화 자료",
        r"PoC": "사전 검증",
        r"고객에게\s*제시": "확인 가능하게 정리",
        r"linkage_level|business_novelty_status|profile_based|event_based": "",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
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
        if not issue_overlap and not _has_token_overlap(value, skax_tokens):
            return False
        if skax_tokens and _has_token_overlap(value, skax_tokens):
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
        if text:
            return text
    return ""


def _bounded_detail_lines(*values: Any) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for value in values:
        for text in _detail_line_candidates(value):
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
        sources.append(
            {
                "index": index,
                "article_id": _optional_int(article.get("id")),
                "title": title,
                "url": url,
                "archive_url": article.get("archive_url"),
                "source_name": str(article.get("source_name") or article.get("publisher") or ""),
                "published_at": _string_or_none(article.get("published_at")),
                "link_status": "ok",
            }
        )
    return sources


def _media_assets(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for article in articles:
        article_id = _optional_int(article.get("id"))
        title = str(article.get("title") or "").strip()
        for url in _article_image_urls(article):
            if url in seen:
                continue
            seen.add(url)
            assets.append(
                {
                    "id": f"img-{article_id or len(assets) + 1}-{len(assets) + 1}",
                    "type": "image",
                    "url": url,
                    "alt": title or "뉴스 본문 이미지",
                }
            )
    return assets


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
    return peer_id if peer_id in _FRONTEND_PEER_IDS else None


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
