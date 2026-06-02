"""카드뉴스 생성 에이전트.

AnalysisPackage를 사용자에게 보여줄 카드뉴스/API 응답 형태로 재가공한다.
기존 raw cluster 기반 생성 메서드는 호환용으로 유지한다.
"""

import json
import logging
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
- 3줄 요약: 반드시 3줄, 각 줄은 "1.", "2.", "3."으로 시작
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
  "summary_lines": ["1. ...", "2. ...", "3. ..."],
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
        summary_lines = _plain_summary_lines(summary)
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
            card["implication_result"] = implication_result
            card["implication"] = _implication_from_result(
                implication_result,
                fallback=card.get("implication"),
            )
            card["frontend_implication"] = _frontend_implication_from_result(
                implication_result,
                fallback=card.get("frontend_implication"),
            )
        else:
            # ImplicationAgent 결과 없음/무효 → analysis 기반 frontend fallback 유지.
            card.setdefault(
                "frontend_implication", _frontend_implication(package.get("analysis") or {})
            )
        card["analysis_package"] = {
            "bundle_id": package.get("bundle_id"),
            "integrated_issue": integrated_issue,
            "summary": integrated_issue,
            "analysis": package.get("analysis") or {},
            "implication": implication_result,
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
    summary_lines = _numbered_summary_lines(summary.get("fact_summary"))
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
    for index, line in enumerate(lines[:3], start=1):
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


def _plain_summary_lines(summary: dict[str, Any]) -> list[str]:
    lines = _list_string(summary.get("fact_summary"))[:3]
    if lines:
        return [_strip_number_prefix(line) for line in lines]
    lines = _list_string(summary.get("summary_lines"))[:3]
    if lines:
        return [_strip_number_prefix(line) for line in lines]
    facts = [
        str(fact.get("fact") or "").strip()
        for fact in _list_dicts(summary.get("consolidated_facts"))
        if str(fact.get("fact") or "").strip()
    ][:3]
    if facts:
        return facts
    integrated_text = str(summary.get("integrated_text") or "").strip()
    if integrated_text:
        split_lines = [
            item.strip()
            for item in re.split(r"(?<=[.!?。！？])\s+|(?<=다)\.\s*", integrated_text)
            if item.strip()
        ][:3]
        if split_lines:
            return split_lines
    one_line = str(summary.get("one_line_summary") or "").strip()
    return [one_line] if one_line else []


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
    return {
        "why_important": str(analysis.get("analysis_summary") or "").strip(),
        "potential_impact": str(analysis.get("impact_reason") or "").strip(),
        "follow_up_questions": [],
        "suggested_actions": [],
        "confidence": _optional_float(analysis.get("confidence")),
    }


def _implication(analysis: dict[str, Any]) -> dict[str, Any]:
    return _frontend_implication(analysis)


def _frontend_implication_from_result(
    implication: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """v4.0 schema 인식 — peer_implication / skax_implication dict 의 핵심 필드 추출.

    W2-4: 기존에 peer_implication 전체를 str() 으로 변환하던 버그 정정. CardNewsComposer
    가 frontend 에 보내는 표면 schema 와 일치.
    """
    fallback = fallback or {}
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
    follow_up = _list_string(
        implication.get("follow_up_questions") or implication.get("watch_points")
    ) or _list_string(fallback.get("follow_up_questions"))
    suggested_actions = (
        _list_string(skax.get("recommended_actions") if isinstance(skax, dict) else None)
        or _list_string(implication.get("recommended_actions"))
        or _list_string(fallback.get("suggested_actions"))
    )
    confidence = (
        _optional_float(implication.get("confidence"))
        if implication.get("confidence") is not None
        else fallback.get("confidence")
    )
    payload: dict[str, Any] = {
        "why_important": why_important,
        "potential_impact": potential_impact,
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
    frontend = _frontend_implication_from_result(implication, fallback=fallback)
    payload["frontend"] = frontend
    return payload


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


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
            "body": "\n".join(summary_lines[:3]) or None,
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
    if not bool(summary.get("is_valid_summary", True)):
        missing.append("summary is invalid")
    if bool(summary.get("fact_extraction_failed")):
        missing.append("fact extraction failed; rule-based candidates used")
    if not bool(analysis.get("is_valid_analysis", True)):
        missing.append("analysis is invalid")
    return missing


def _missing_fact_basis_line_indexes(summary: dict[str, Any]) -> list[int]:
    lines = _plain_summary_lines(summary)
    if len(lines) != 3:
        return []
    expected = {1, 2, 3}
    present: set[int] = set()
    for item in summary.get("fact_basis", []) or []:
        if not isinstance(item, dict):
            continue
        index = _optional_int(item.get("summary_line_index", item.get("summary_sentence_index")))
        if index in expected and _list_string(item.get("source_article_ids")):
            present.add(index)
    return sorted(expected - present)


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
