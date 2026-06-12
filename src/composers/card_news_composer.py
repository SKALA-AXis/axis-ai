"""카드뉴스 생성 에이전트.

AnalysisPackage를 사용자에게 보여줄 카드뉴스/API 응답 형태로 재가공한다.
기존 raw cluster 기반 생성 메서드는 호환용으로 유지한다.
"""

import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

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
            card_id=_card_news_id(cluster_id, created_at),
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
            "slides": _slides(title, summary_lines, insights, sources, media_assets),
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
            frontend_implication = _enhance_frontend_implication_with_linkage(
                package,
                frontend_implication,
            )
            card["implication_result"] = implication_result
            card["implication"] = _implication_from_result(
                implication_result,
                fallback=card.get("implication"),
                frontend=frontend_implication,
            )
            card["frontend_implication"] = frontend_implication
        else:
            # ImplicationAgent 결과 없음/무효 → analysis 기반 frontend fallback 유지하되,
            # 현재 이슈 안에서 확인 가능한 최소 SK AX 관찰 포인트는 잃지 않는다.
            fallback_frontend = _frontend_implication(package.get("analysis") or {})
            checkpoints = _issue_frame_checkpoints_for_frontend(
                integrated_issue=integrated_issue,
                analysis=package.get("analysis") or {},
            )
            if checkpoints:
                fallback_frontend["suggested_actions"] = checkpoints
                fallback_frontend["response_directions"] = checkpoints
                fallback_frontend["skax_checkpoints"] = checkpoints
            fallback_frontend = _enhance_frontend_implication_with_linkage(
                package,
                fallback_frontend,
            )
            card["frontend_implication"] = fallback_frontend
            card["implication"] = {
                **(card.get("implication") or {}),
                "frontend": fallback_frontend,
            }
        card["slides"] = _slides(
            str(card.get("title") or ""),
            _list_string(card.get("summary_lines")),
            _card_insights_from_package(package, card.get("frontend_implication") or {}),
            _list_dicts(card.get("sources")),
            _media_assets_from_slides(card.get("slides")),
        )
        card["analysis_package"] = {
            "bundle_id": package.get("bundle_id"),
            "integrated_issue": integrated_issue,
            "summary": integrated_issue,
            "analysis": package.get("analysis") or {},
            "implication": implication_result,
            "sentence_grounding": package.get("sentence_grounding")
            or (package.get("evidence_payload") or {})
            .get("analysis_package", {})
            .get("sentence_grounding")
            or {},
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


def _card_news_id(cluster_id: int | None, created_at: str) -> str:
    date_key = created_at[:10].replace("-", "")
    suffix = f"{cluster_id:04d}" if cluster_id is not None else "0000"
    return f"CN-{date_key}-{suffix}"


def _plain_summary_lines(summary: dict[str, Any], *, use_llm: bool = False) -> list[str]:
    lines = _list_string(summary.get("fact_summary"))[:_SUMMARY_LINE_MAX]
    if lines:
        return [str(line).strip() for line in lines if str(line).strip()]
    lines = _list_string(summary.get("summary_lines"))[:_SUMMARY_LINE_MAX]
    if lines:
        return [str(line).strip() for line in lines if str(line).strip()]
    facts = [
        str(fact.get("fact") or "").strip()
        for fact in _list_dicts(summary.get("consolidated_facts"))
        if str(fact.get("fact") or "").strip()
    ][:_SUMMARY_LINE_MAX]
    if facts:
        return facts
    integrated_text = str(summary.get("integrated_text") or "").strip()
    if integrated_text:
        split_lines = [
            item.strip()
            for item in re.split(r"(?<=[.!?。！？])\s+|(?<=다)\.\s*", integrated_text)
            if item.strip()
        ][:_SUMMARY_LINE_MAX]
        if split_lines:
            return split_lines
    one_line = str(summary.get("one_line_summary") or "").strip()
    return [one_line] if one_line else []


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
    for _, _, text in ranked:
        if len(selected) >= _SUMMARY_LINE_MAX:
            break
        role = _summary_line_role(text, summary)
        if role in selected_roles and role != "context" and len(selected) < _SUMMARY_LINE_MIN:
            skipped.append(text)
            continue
        if any(_summary_lines_too_similar(text, existing) for existing in selected):
            skipped.append(text)
            continue
        selected.append(text)
        selected_roles.add(role)
    for text in skipped:
        if len(selected) >= _SUMMARY_LINE_MIN:
            break
        if text not in selected:
            selected.append(text)
    return selected[:_SUMMARY_LINE_MIN] if len(selected) >= _SUMMARY_LINE_MIN else selected


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
    return overlap / min(len(left_tokens), len(right_tokens)) >= 0.72


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
    return {token for token in tokens if token not in stopwords}


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


def _issue_frame_checkpoints_for_frontend(
    *,
    integrated_issue: dict[str, Any],
    analysis: dict[str, Any],
) -> list[str]:
    """Minimum SK AX-facing checkpoints when strategic implication is invalid.

    These are intentionally generic in structure but grounded in the current
    issue's own terms. They are not a replacement for StrategicInsightAgent;
    they keep the card usable while avoiding invented capability claims.
    """
    subject = _frontend_issue_subject(integrated_issue)
    issue_terms = _frontend_issue_terms(integrated_issue)
    comparison_terms = _frontend_comparison_terms(integrated_issue, analysis)
    follow_up_terms = _frontend_follow_up_terms(integrated_issue)
    if not subject and not issue_terms:
        return []

    subject_text = subject or "현재 사건"
    issue_text = ", ".join(issue_terms[:3]) if issue_terms else "현재 확인된 사업 범위"
    comparison_text = (
        ", ".join(comparison_terms[:4])
        if comparison_terms
        else ", ".join(issue_terms[:3]) or "현재 사건에서 확인된 비교 기준"
    )
    follow_up_text = ", ".join(follow_up_terms[:4]) if follow_up_terms else "후속 역할과 책임 범위"
    return _bounded_detail_items(
        (
            f"SK AX는 {subject_text}와 유사한 흐름을 볼 때 {issue_text}가 "
            f"{comparison_text}와 어떻게 연결되는지 내부적으로 구분해야 합니다. "
            "이 구분이 있어야 피어 신호를 단순 동향이 아니라 자사 대응 가능 범위와 "
            "보완 필요 영역으로 나눠 볼 수 있습니다."
        ),
        (
            f"후속으로는 {follow_up_text}가 추가로 확인되는지 추적해야 합니다. "
            "이 정보가 있어야 현재 관찰한 비교 기준을 실제 사업 대응 범위와 리스크 기준으로 "
            "조정할 수 있습니다."
        ),
    )


def _frontend_issue_subject(integrated_issue: dict[str, Any]) -> str:
    fact_text = " ".join(_plain_summary_lines(integrated_issue, use_llm=False))
    compact = _compact_issue_subject_from_text(fact_text)
    if compact:
        return compact
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for item in [
            *(intelligence.get("common_facts") or []),
            *(intelligence.get("unique_facts") or []),
        ]:
            if not isinstance(item, dict):
                continue
            for value in _list_string(item.get("products_or_services")):
                text = _compact_issue_subject_from_text(value) or re.sub(r"\s+", " ", value).strip(
                    " ."
                )
                if text:
                    return text
    for key in ("main_event", "main_issue", "one_line_summary", "headline"):
        text = str(integrated_issue.get(key) or "").strip()
        if text:
            return (
                _compact_issue_subject_from_text(text) or re.sub(r"\s+", " ", text).strip(" .")[:60]
            )
    return ""


def _compact_issue_subject_from_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .。"))
    if not text:
        return ""
    quotes = [
        item.strip()
        for item in re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,30})[\"'“”‘’]", text)
        if item.strip()
    ]
    if len(quotes) >= 2:
        return "·".join(quotes[:2])
    if len(quotes) == 1 and len(quotes[0]) >= 4:
        return quotes[0]
    patterns = (
        r"([가-힣A-Za-z0-9&·+_-]+(?:\s+[가-힣A-Za-z0-9&·+_-]+){0,4})을\s*위한\s*업무협약",
        r"([가-힣A-Za-z0-9&·+_-]+(?:\s+[가-힣A-Za-z0-9&·+_-]+){0,4})\s*구축\s*사업",
        r"([가-힣A-Za-z0-9&·+_-]+(?:\s+[가-힣A-Za-z0-9&·+_-]+){0,3})\s*전환\s*사업",
        r"([가-힣A-Za-z0-9&·+_-]+(?:\s+[가-힣A-Za-z0-9&·+_-]+){0,3})\s*플랫폼",
        r"([가-힣A-Za-z0-9&·+_-]+(?:\s+[가-힣A-Za-z0-9&·+_-]+){0,3})\s*서비스",
        r"([가-힣A-Za-z0-9&·+_-]+(?:\s+[가-힣A-Za-z0-9&·+_-]+){0,3})\s*센터",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        subject = re.sub(r"^[^,\n]{2,30},\s*", "", match.group(1))
        subject = re.sub(r"^\d{1,2}일\s*", "", subject)
        subject = re.sub(r"\s*(체결|선정|확정|완료|나선다)$", "", subject).strip(" .")
        if 3 <= len(subject) <= 45:
            return subject
    return ""


def _frontend_issue_terms(integrated_issue: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for key in ("products_or_services", "customers_or_industries", "activity_types"):
            terms.extend(_list_string(intelligence.get(key))[:8])
    for item in _list_dicts(integrated_issue.get("key_numbers")):
        metric = str(item.get("metric") or item.get("metric_label") or "").strip()
        value = str(item.get("value") or "").strip()
        unit = str(item.get("unit") or "").strip()
        if metric or value:
            terms.append(" ".join(part for part in (metric, f"{value}{unit}".strip()) if part))
    return _dedupe_keep_order([term for term in terms if term])[:8]


def _frontend_comparison_terms(
    integrated_issue: dict[str, Any],
    analysis: dict[str, Any],
) -> list[str]:
    terms: list[str] = []
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for key in ("products_or_services", "customers_or_industries", "activity_types"):
            terms.extend(_list_string(intelligence.get(key))[:6])
        for fact_group in ("common_facts", "unique_facts"):
            for item in _list_dicts(intelligence.get(fact_group)):
                terms.extend(_list_string(item.get("products_or_services"))[:4])
                terms.extend(_list_string(item.get("customers_or_industries"))[:4])
                terms.extend(_list_string(item.get("activity_types"))[:3])
    for item in _list_dicts(integrated_issue.get("key_numbers")):
        metric = str(item.get("metric") or item.get("metric_label") or "").strip()
        value = str(item.get("value") or "").strip()
        unit = str(item.get("unit") or "").strip()
        if metric or value:
            terms.append(" ".join(part for part in (metric, f"{value}{unit}".strip()) if part))
    text = " ".join(
        [
            str(analysis.get("market_signal") or ""),
            str(analysis.get("impact_reason") or ""),
            " ".join(_list_string(analysis.get("strategic_meaning"))),
            " ".join(_list_string(integrated_issue.get("fact_summary"))),
        ]
    )
    for match in re.finditer(
        r"[가-힣A-Za-z0-9&·+_-]{2,20}(?:\s+[가-힣A-Za-z0-9&·+_-]{2,20}){0,2}"
        r"(?:범위|책임|일정|규모|수치|역할|구조|기간|기준|기여도|수익성|지속성)",
        text,
        flags=re.IGNORECASE,
    ):
        terms.append(match.group(0))
    return _dedupe_keep_order(
        [term for term in (_public_axis_phrase(term) for term in terms) if term]
    )[:6]


def _frontend_follow_up_terms(integrated_issue: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for text in [
        *_list_string(integrated_issue.get("missing_or_uncertain_points")),
        *_list_string(integrated_issue.get("fact_summary")),
    ]:
        for match in re.finditer(
            r"[가-힣A-Za-z0-9&·+_-]{2,20}(?:\s+[가-힣A-Za-z0-9&·+_-]{2,20}){0,2}"
            r"(?:일정|책임|협약|계약|구축|운영|실적|기여도|수익성|서비스\s*개시|후속)",
            text,
            flags=re.IGNORECASE,
        ):
            terms.append(match.group(0))
    return _dedupe_keep_order(
        [term for term in (_public_axis_phrase(term) for term in terms) if term]
    )[:6]


def _enhance_frontend_implication_with_linkage(
    package: dict[str, Any],
    frontend: dict[str, Any],
) -> dict[str, Any]:
    """Fill missing frontend copy from profile-linkage fallback only.

    StrategicInsightAgent output is the source of truth when valid. Linkage
    fallback is intentionally limited to empty fields so the composer does not
    overwrite stronger strategic prose with template-like sentences.
    """
    integrated_issue = package.get("integrated_issue") or package.get("summary") or {}
    profile_linkage = package.get("profile_linkage") or {}
    skax_linkage = package.get("skax_response_linkage") or {}
    if not isinstance(frontend, dict):
        frontend = {}

    enhanced = dict(frontend)
    if _frontend_items_need_linkage(enhanced.get("key_implications"), section="insight"):
        key_implications = _linkage_key_implications(
            integrated_issue=integrated_issue,
            profile_linkage=profile_linkage,
            fallback=frontend.get("key_implications"),
        )
    else:
        key_implications = []
    if key_implications:
        enhanced["key_implications"] = key_implications
        if not _list_string(enhanced.get("peer_implications")):
            enhanced["peer_implications"] = key_implications
        if not str(enhanced.get("potential_impact") or "").strip():
            enhanced["potential_impact"] = key_implications[0]

    if _frontend_items_need_linkage(enhanced.get("suggested_actions"), section="action"):
        suggested_actions = _linkage_suggested_actions(
            integrated_issue=integrated_issue,
            skax_linkage=skax_linkage,
            profile_linkage=profile_linkage,
            fallback=frontend.get("suggested_actions"),
        )
    else:
        suggested_actions = []
    if suggested_actions:
        enhanced["suggested_actions"] = suggested_actions
        if not _list_string(enhanced.get("response_directions")):
            enhanced["response_directions"] = suggested_actions
        if not _list_string(enhanced.get("skax_checkpoints")):
            enhanced["skax_checkpoints"] = suggested_actions
    return _cleanup_public_frontend_implication(enhanced)


def _frontend_items_need_linkage(value: Any, *, section: str) -> bool:
    items = _list_string(value)
    if not items:
        return True
    text = " ".join(items)
    weak_patterns = (
        r"현재\s*사건과\s*직접\s*맞는.*충분하지",
        r"현재\s*확인\s*가능한\s*SK\s*AX\s*관련\s*사업/역량",
        r"기여할\s*것으로\s*보인다",
        r"새로운\s*(기준|국면|방향)",
        r"참고할\s*수\s*있는\s*모델",
        r"예상된다",
        r"가능성[을를]?\s*탐색",
        r"전략적\s*방향",
        r"후속\s*데이터[를을]?\s*수집",
        r"관련\s*기술\s*개발",
        r"계약\s*범위와\s*기간이\s*제시",
        r"유사\s*사업의\s*비교\s*기준을\s*관찰",
        r"대상\s*범위\s*:",
        r"사업·서비스\s*조건\s*:",
        r"진행\s*단계·관계\s*:",
        r"\bconfirmed\b",
        r"효율성\s*증대",
        r"인력\s*관리\s*개선",
        r"긍정적인\s*변화",
        r"새로운\s*기술\s*적용",
        r"일반\s*표현",
        r"general_update|partnership|contract",
        r"technology_update|unknown",
        r"역량[을를]?\s*보여",
        r"비즈니스\s*혁신\s*역량",
        r"경험[을를]?\s*쌓을\s*수",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in weak_patterns):
        return True
    if section == "insight" and re.search(
        r"봐야\s*합니다|확인해야\s*합니다|모니터링해야\s*합니다|점검해야\s*합니다",
        text,
    ):
        return True
    if section == "action" and not re.search(r"SK\s*AX|자사|내부|후속|점검|비교", text):
        return True
    if section == "insight" and re.search(r"SK\s*AX|자사|내부적으로|점검해야|보완해야", text):
        return True
    return False


def _linkage_key_implications(
    *,
    integrated_issue: dict[str, Any],
    profile_linkage: dict[str, Any],
    fallback: Any,
) -> list[str]:
    areas = _list_dicts(profile_linkage.get("matched_profile_areas"))
    if not _linkage_supports_profile_based_claim(profile_linkage, areas=areas):
        return _fallback_key_implications_without_linkage(
            integrated_issue=integrated_issue,
            fallback=fallback,
        )

    subject = _frontend_issue_subject(integrated_issue) or "현재 사건"
    facts = _plain_summary_lines(integrated_issue, use_llm=False)
    second_fact = _first_text(*facts[1:])
    peer_phrase = _public_profile_meaning_phrase(areas, integrated_issue=integrated_issue)
    linkage_level = str(profile_linkage.get("linkage_level") or "").strip()
    interpretation = str(profile_linkage.get("allowed_interpretation_strength") or "").strip()

    comparison_phrase = _fact_based_comparison_phrase(
        integrated_issue,
        areas,
        include_profile_terms=False,
    )
    business_context = _issue_business_context_sentence(
        integrated_issue,
        subject=subject,
        facts=facts,
    )
    first = (
        f"{_topic_phrase(subject)} {_subject_particle(peer_phrase)} 실제 고객·현장·업무에 "
        "적용되는 방식으로 이어진 사건입니다. "
        f"{business_context} "
        f"그래서 이 이슈는 단순 발표보다 {comparison_phrase}가 실제 사업 의미를 "
        "만드는 근거로 해석됩니다."
    )
    if linkage_level in {"low", "none"} or interpretation in {"event_based", "weak"}:
        first += " 다만 기사만으로 역할 확대나 성과를 단정하기는 어렵습니다."

    second = (
        f"{comparison_phrase}는 이 사건에서 피어사의 움직임을 해석하는 핵심 축입니다. "
    )
    if second_fact:
        second += (
            f'기사에서는 "{second_fact}"도 함께 제시되어, '
            "이 축이 단순 개념이 아니라 실제 실행 방식과 연결됩니다. "
        )
    second += (
        "현재 단계의 시사점은 역할 확대를 단정하는 것이 아니라, "
        "기존 사업 흐름이 구체적인 적용 대상과 운영 방식으로 연결되기 시작했다는 점입니다."
    )
    return _public_copy_items(first, second)


def _linkage_suggested_actions(
    *,
    integrated_issue: dict[str, Any],
    skax_linkage: dict[str, Any],
    profile_linkage: dict[str, Any],
    fallback: Any,
) -> list[str]:
    areas = _list_dicts(skax_linkage.get("matched_skax_areas"))
    if not _linkage_supports_profile_based_claim(skax_linkage, areas=areas):
        return _fallback_suggested_actions_without_linkage(
            integrated_issue=integrated_issue,
            fallback=fallback,
        )

    subject = _frontend_issue_subject(integrated_issue) or "현재 사건"
    facts = _plain_summary_lines(integrated_issue, use_llm=False)
    first_fact = _first_text(*facts)
    skax_phrase = _public_skax_meaning_phrase(areas, integrated_issue=integrated_issue)
    peer_phrase = _public_profile_meaning_phrase(
        _list_dicts(profile_linkage.get("matched_profile_areas")),
        integrated_issue=integrated_issue,
    )
    comparison_phrase = _fact_based_comparison_phrase(
        integrated_issue,
        areas,
        include_profile_terms=True,
    )
    monitoring_points = _bounded_detail_items(skax_linkage.get("monitoring_points"))

    first = (
        f"SK AX는 {subject}와 유사한 흐름을 볼 때 {skax_phrase}을 기준으로 "
        f"{comparison_phrase} 중 직접 판단할 영역과 보완이 필요한 영역을 구분해야 합니다. "
    )
    if peer_phrase:
        first += f"해당 기업은 {peer_phrase}을 실제 적용 장면에 연결하고 있으므로, "
    first += (
        "이 기준을 봐야 SK AX가 직접 맡을 수 있는 범위와 외부 보완이 필요한 범위를 "
        "현실적으로 판단할 수 있습니다."
    )

    second = (
        f"후속으로는 {subject}의 실행 결과를 단순 진행 여부가 아니라 "
        f"{comparison_phrase} 기준으로 확인해야 합니다. "
    )
    if first_fact:
        second += f'기사에서 "{first_fact}"가 확인된 만큼, '
    if monitoring_points:
        point = _strip_sentence_ending(monitoring_points[0])
        second += f"또한 {point}는 별도 확인 항목으로 남겨야 합니다. "
    second += (
        "이 정보가 쌓여야 SK AX가 같은 유형의 사업에서 대응 범위와 "
        "보완 우선순위를 조정할 수 있습니다."
    )
    return _public_copy_items(first, second)


def _linkage_supports_profile_based_claim(
    linkage: dict[str, Any],
    *,
    areas: list[dict[str, Any]],
) -> bool:
    """Return true only when profile linkage is strong enough for public copy.

    A broad business-line match is not enough. The card copy may connect the
    issue with profile context only when the linkage evaluation itself is
    strong/medium or the matched area carries concrete profile evidence.
    """
    if not isinstance(linkage, dict) or not areas:
        return False

    level = str(linkage.get("linkage_level") or "").strip().casefold()
    strength = str(linkage.get("allowed_interpretation_strength") or "").strip().casefold()
    if level in {"none", "low"} or strength in {"none", "weak", "event_based"}:
        return False

    if level in {"high", "medium"} or strength in {
        "profile_based",
        "profile_supported",
        "profile_context",
    }:
        return _linkage_has_specific_profile_evidence(areas)

    # Older payloads may not have linkage_level. In that case, require concrete
    # matched profile fields so a business-line-only match does not force a claim.
    return _linkage_has_specific_profile_evidence(areas)


def _linkage_has_specific_profile_evidence(areas: list[dict[str, Any]]) -> bool:
    for area in areas:
        if str(area.get("specificity_level") or "").strip().casefold() in {"high", "medium"}:
            return True
        if _list_string(area.get("matched_products_or_services")):
            return True
        if _list_string(area.get("matched_capabilities")):
            return True
        if len(str(area.get("evidence_text") or "").strip()) >= 15:
            return True
    return False


def _linkage_area_phrase(areas: list[dict[str, Any]]) -> str:
    names: list[str] = []
    for area in areas[:4]:
        products = _list_string(area.get("matched_products_or_services"))
        capabilities = _list_string(area.get("matched_capabilities"))
        business_area = str(area.get("business_area") or area.get("profile_area_name") or "")
        if products:
            names.extend(products[:2])
        elif capabilities:
            names.extend(capabilities[:2])
        elif business_area:
            names.append(business_area)
    return "·".join(_dedupe_keep_order([name for name in names if name])[:4]) or "관련 사업"


def _linkage_issue_terms(
    areas: list[dict[str, Any]],
    *,
    integrated_issue: dict[str, Any],
) -> list[str]:
    terms: list[str] = []
    for area in areas[:4]:
        terms.extend(_list_string(area.get("matched_issue_terms"))[:6])
    terms.extend(_frontend_issue_terms(integrated_issue))
    cleaned = []
    low_signal_terms = {
        "lg",
        "cns",
        "sk",
        "ax",
        "사업",
        "위한",
        "나선다",
        "등을",
        "추진",
        "혁신",
        "고객",
        "rx",
        "스마트",
        "general_update",
    }
    for term in terms:
        text = re.sub(r"\s+", " ", str(term or "").strip(" ."))
        if len(text) < 2 or text.casefold() in low_signal_terms:
            continue
        cleaned.append(text)
    return _dedupe_keep_order(cleaned)[:8]


def _fact_based_comparison_phrase(
    integrated_issue: dict[str, Any],
    areas: list[dict[str, Any]],
    *,
    include_profile_terms: bool = False,
) -> str:
    candidates: list[str] = []
    if include_profile_terms:
        for area in areas[:4]:
            candidates.extend(_list_string(area.get("matched_issue_terms"))[:4])
            candidates.extend(_list_string(area.get("matched_products_or_services"))[:3])
            candidates.extend(_list_string(area.get("matched_capabilities"))[:3])
    candidates.extend(_frontend_comparison_terms(integrated_issue, {}))
    candidates.extend(_frontend_issue_terms(integrated_issue))
    cleaned = _dedupe_keep_order(
        [term for term in (_public_axis_phrase(term) for term in candidates) if term]
    )
    if not cleaned:
        cleaned = _comparison_terms_from_summary_facts(integrated_issue)
    return ", ".join(cleaned[:4]) or "현재 사건의 대상 업무·서비스와 후속 확인 기준"


def _comparison_terms_from_summary_facts(integrated_issue: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    fact_text = " ".join(_plain_summary_lines(integrated_issue, use_llm=False))
    for quoted in re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,45})[\"'“”‘’]", fact_text):
        terms.append(quoted)
    for match in re.finditer(
        r"[가-힣A-Za-z0-9&·+_-]{2,24}(?:\s+[가-힣A-Za-z0-9&·+_-]{2,24}){0,2}"
        r"(?:플랫폼|서비스|시스템|업무|시간|기간|센터|인프라|자동화|전환)",
        fact_text,
    ):
        terms.append(match.group(0))
    return _dedupe_keep_order(
        [term for term in (_public_axis_phrase(term) for term in terms) if term]
    )[:6]


def _comparison_reason_phrase(integrated_issue: dict[str, Any]) -> str:
    facts = _plain_summary_lines(integrated_issue, use_llm=False)
    first_fact = _first_text(*facts)
    if first_fact:
        return "이 사실이 현재 사건의 실제 적용 장면을 보여주기 때문에"
    terms = _frontend_issue_terms(integrated_issue)
    if terms:
        return f"기사에서 {', '.join(terms[:3])}가 확인됐기"
    return "기사에서 현재 사건의 대상과 후속 확인 항목이 함께 제시됐기"


def _issue_business_context_sentence(
    integrated_issue: dict[str, Any],
    *,
    subject: str,
    facts: list[str],
) -> str:
    usable_facts = [
        _short_public_fact(fact)
        for fact in facts
        if fact and not re.search(r"기념촬영|임직원|전무|부사장|사진|이미지", fact)
    ]
    if not usable_facts:
        return f"{subject}의 대상과 실행 조건이 기사에서 확인됩니다."

    first = usable_facts[0]
    details = usable_facts[1:3]
    if details:
        return (
            f'기사에서는 "{first}"가 확인되고, '
            f'"{details[0]}"도 함께 제시됩니다. '
            "즉 이 사건은 제목 수준의 발표가 아니라 적용 대상, 실행 방식, "
            "운영 조건 중 일부가 기사 안에서 드러난 사업입니다."
        )
    return (
        f'기사에서는 "{first}"가 확인됩니다. '
        "즉 이 사건은 제목 수준의 발표가 아니라 기사 안에서 확인된 사실을 기준으로 "
        "사업 의미를 해석해야 하는 신호입니다."
    )


def _public_profile_meaning_phrase(
    areas: list[dict[str, Any]],
    *,
    integrated_issue: dict[str, Any],
) -> str:
    names = _linkage_area_phrase(areas)
    if names == "관련 사업":
        return "해당 기업의 기존 사업 흐름"
    return f"{names} 관련 기존 사업 흐름"


def _public_skax_meaning_phrase(
    areas: list[dict[str, Any]],
    *,
    integrated_issue: dict[str, Any],
) -> str:
    names = _linkage_area_phrase(areas)
    if names == "관련 사업":
        return "SK AX의 관련 사업"
    return f"SK AX의 {names} 관련 사업"


def _public_axis_phrase(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .。,:;，"))
    if not text:
        return ""
    if text.casefold() in {
        "general_update",
        "reported_fact",
        "core_fact",
        "unique_fact",
        "common_fact",
        "partnership",
        "collaboration",
        "contract",
        "selection",
        "launch",
        "performance",
        "investment",
        "mou",
        "technology_update",
        "business_update",
        "reported_update",
        "unknown",
        "unclear",
    }:
        return ""
    text = re.sub(r"^(이번|해당)\s+", "", text)
    text = re.sub(r"^위한\s+", "", text)
    if text in {
        "협약",
        "계약",
        "업무협약",
        "구축",
        "운영",
        "서비스",
        "사업",
        "위한",
        "나선다",
        "스마트",
        "하반기",
        "혁신",
        "고객",
    }:
        return ""
    if re.fullmatch(r"[A-Za-z]{1,3}", text):
        return ""
    if re.search(r"기념촬영|임직원|전무|부사장|사진|이미지", text):
        return ""
    return text[:80]


def _short_public_fact(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .。"))
    return text if len(text) <= 90 else text[:87].rstrip() + "..."


def _fallback_key_implications_without_linkage(
    *,
    integrated_issue: dict[str, Any],
    fallback: Any,
) -> list[str]:
    subject = _frontend_issue_subject(integrated_issue) or "현재 사건"
    facts = _plain_summary_lines(integrated_issue, use_llm=False)
    second_fact = _first_text(*facts[1:])
    comparison_phrase = _fact_based_comparison_phrase(integrated_issue, [])
    business_context = _issue_business_context_sentence(
        integrated_issue,
        subject=subject,
        facts=facts,
    )
    first = (
        f"{_topic_phrase(subject)} 기사에서 확인된 사건 기반 신호입니다. "
        f"{business_context} "
        "현재 기사 근거만으로는 기존 사업 역량과의 세부 연결을 단정하기 어려우므로, "
        "역량 강화나 사업 확장으로 단정하지 않고 "
        f"{comparison_phrase} 같은 비교 기준이 공개된 관찰 신호로 해석됩니다."
    )
    second = (
        f"이 사건의 사업적 의미는 {comparison_phrase}가 실제 적용 대상과 실행 방식으로 "
        "드러났다는 데 있습니다. "
        "현재 단계에서는 확정 성과보다 기사에서 공개된 사건 범위가 유사 사업의 "
        "비교 기준으로 구체화된 점이 핵심입니다."
    )
    if second_fact:
        second += f' 기사에서는 "{second_fact}"도 함께 제시되어 이 해석을 뒷받침합니다.'
    return _public_copy_items(first, second, fallback)[:2]


def _fallback_suggested_actions_without_linkage(
    *,
    integrated_issue: dict[str, Any],
    fallback: Any,
) -> list[str]:
    subject = _frontend_issue_subject(integrated_issue) or "현재 사건"
    facts = _plain_summary_lines(integrated_issue, use_llm=False)
    first_fact = _first_text(*facts)
    comparison_phrase = _fact_based_comparison_phrase(integrated_issue, [])
    follow_up_terms = _frontend_follow_up_terms(integrated_issue)
    follow_up_text = ", ".join(follow_up_terms[:3]) or comparison_phrase
    first = (
        f"SK AX는 {subject}와 유사한 흐름을 볼 때 특정 사업 역량으로 바로 단정하지 말고 "
        f"다음 항목을 기준으로 자사 관련 사업과 겹치는 지점, 추가 확인이 필요한 지점, "
        f"외부 보완이 필요한 지점을 나눠 봐야 합니다: {comparison_phrase}. "
    )
    if first_fact:
        first += f'기사에서 "{first_fact}"가 확인되므로, '
    first += (
        "이 구분이 있어야 해당 기업의 움직임을 과대해석하지 않고 내부 대응 범위를 정할 수 있습니다."
    )
    second = (
        f"후속으로는 다음 항목을 확인해야 합니다: {follow_up_text}. "
        "이 정보가 쌓여야 SK AX가 같은 유형의 사업에서 직접 점검할 영역과 "
        "외부 확인이 필요한 영역을 후속 상황에 맞게 조정할 수 있습니다."
    )
    return _public_copy_items(first, second, fallback)[:2]


def _public_copy_items(*values: Any) -> list[str]:
    return [_public_copy_cleanup(item) for item in _bounded_detail_items(*values)]


def _cleanup_public_frontend_implication(frontend: dict[str, Any]) -> dict[str, Any]:
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
        if key in cleaned:
            section = (
                "action"
                if key
                in {
                    "suggested_actions",
                    "response_directions",
                    "skax_checkpoints",
                }
                else "insight"
            )
            cleaned[key] = [
                _format_public_frontend_item(_public_copy_cleanup(item), section=section)
                for item in _list_string(cleaned.get(key))
            ]
    if "follow_up_questions" in cleaned:
        cleaned["follow_up_questions"] = [
            _public_copy_cleanup(item) for item in _list_string(cleaned.get("follow_up_questions"))
        ]
    return cleaned


def _format_public_frontend_item(value: Any, *, section: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.match(r"^핵심\s*(시사점|대응)\s*:", text):
        return text
    conclusion, evidence = _split_public_conclusion_evidence(text)
    if not conclusion or not evidence:
        return text
    heading = "핵심 대응" if section == "action" else "핵심 시사점"
    return f"{heading}: {conclusion}\n근거/설명: {evidence}"


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

    # Some LLM/fallback copy arrives as one long sentence. Split only on
    # reasoning connectors so the meaning stays intact.
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
    parts = normalized.splitlines()
    return [part.strip() for part in parts if part.strip()]


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
    replacements = (
        (r"통합\s*결과에서는", "기사에서는"),
        (r"통합\s*결과", "기사"),
        (r"피어\s*프로필", "해당 기업의 기존 사업 흐름"),
        (r"자사\s*프로필", "SK AX의 관련 사업/역량"),
        (r"피어\s*쪽", "해당 기업"),
        (r"피어사", "해당 기업"),
        (r"\blinkage\b", "연결 지점"),
        (r"프로필\s*접점", "이어지는 흐름"),
        (r"접점", "연결 지점"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text.strip()


def _topic_phrase(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return "현재 사건은"
    marker = "은" if _has_final_consonant(text[-1]) else "는"
    return f"{text}{marker}"


def _subject_particle(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    marker = "이" if _has_final_consonant(text[-1]) else "가"
    return f"{text}{marker}"


def _has_final_consonant(char: str) -> bool:
    code = ord(char)
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def _strip_sentence_ending(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .。"))
    text = re.sub(r"되는지\s*확인합니다$", "되는지 여부", text)
    text = re.sub(r"인지\s*확인합니다$", "인지 여부", text)
    text = re.sub(r"\s*확인합니다$", " 확인 여부", text)
    text = re.sub(r"\s*모니터링합니다$", " 모니터링 항목", text)
    text = re.sub(r"\s*추적해야\s*합니다$", " 추적 항목", text)
    text = re.sub(r"\s*점검해야\s*합니다$", " 점검 항목", text)
    return text.strip(" .。")


def _card_insights_from_package(
    package: dict[str, Any],
    frontend_implication: dict[str, Any],
) -> list[str]:
    analysis = package.get("analysis") or {}
    implication = package.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    return _bounded_detail_items(
        frontend_implication.get("key_implications"),
        peer.get("peer_meaning") if isinstance(peer, dict) else None,
        peer.get("capability_change") if isinstance(peer, dict) else None,
        analysis.get("strategic_meaning") if isinstance(analysis, dict) else None,
        analysis.get("market_signal") if isinstance(analysis, dict) else None,
        skax.get("recommended_actions") if isinstance(skax, dict) else None,
        frontend_implication.get("skax_checkpoints"),
    )


def _media_assets_from_slides(value: Any) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for slide in _list_dicts(value):
        url = str(slide.get("image_url") or "").strip()
        if not url:
            continue
        assets.append(
            {
                "url": url,
                "alt": str(slide.get("image_alt") or _DEFAULT_COVER_IMAGE_ALT),
            }
        )
    return assets


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
    )
    if not peer_implications:
        peer_implications = _bounded_detail_items(fallback.get("key_implications"))
    market_signals = _bounded_detail_items(
        analysis.get("market_signal") if isinstance(analysis, dict) else None,
    )
    strategic_meanings = _bounded_detail_items(
        analysis.get("strategic_meaning") if isinstance(analysis, dict) else None,
    )
    skax_checkpoints = _bounded_detail_items(
        skax.get("recommended_actions") if isinstance(skax, dict) else None,
        implication.get("recommended_actions"),
        fallback.get("suggested_actions"),
    )
    follow_up = (
        _bounded_detail_lines(implication.get("follow_up_questions"))
        or _bounded_detail_lines(implication.get("watch_points"))
        or _bounded_detail_lines(fallback.get("follow_up_questions"))
    )
    suggested_actions = skax_checkpoints or _actionize_detail_lines(
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
        "market_signals": market_signals,
        "strategic_meanings": strategic_meanings,
        "skax_implications": _bounded_detail_items(why_important, potential_impact),
        "response_directions": suggested_actions,
        "skax_checkpoints": suggested_actions,
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
            r"제안|검토|확인|설명|분리|검증|제시|반영|정리|설계|작성|구성|관리|추적|비교",
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
