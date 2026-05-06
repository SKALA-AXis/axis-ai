"""분류 에이전트 — 섹터 태깅 + 노출도 점수 + event_type 분류."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI

from src.config.companies import company_aliases
from src.config.sectors import SECTOR_IDS, match_sectors, primary_sector
from src.db.article_store import get_articles_by_ids, update_classification

log = logging.getLogger(__name__)

EVENT_TYPES = [
    "partnership",
    "ma",
    "personnel",
    "tech_release",
    "regulation",
    "contract",
    "financial",
    "expansion",
]

EVENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "partnership": [
        "협약",
        "업무협약",
        "MOU",
        "파트너십",
        "제휴",
        "협력",
        "공동 개발",
        "공동사업",
        "컨소시엄",
    ],
    "ma": [
        "인수",
        "합병",
        "인수합병",
        "지분 인수",
        "지분 투자",
        "투자 유치",
        "피인수",
        "매각",
    ],
    "personnel": [
        "채용",
        "인사",
        "임원",
        "대표이사",
        "CEO",
        "선임",
        "영입",
        "조직개편",
        "조직 개편",
    ],
    "tech_release": [
        "출시",
        "공개",
        "개발",
        "고도화",
        "플랫폼",
        "솔루션",
        "서비스",
        "기술",
        "AI",
        "클라우드",
        "보안",
        "SaaS",
    ],
    "regulation": [
        "규제",
        "정책",
        "법안",
        "가이드라인",
        "정부",
        "인증",
        "표준",
        "컴플라이언스",
    ],
    "contract": [
        "수주",
        "계약",
        "선정",
        "공급",
        "구축 사업",
        "사업자",
        "우선협상대상자",
        "프로젝트 수주",
    ],
    "financial": [
        "매출",
        "영업이익",
        "실적",
        "분기",
        "연간",
        "흑자",
        "적자",
        "투자계획",
        "가이던스",
    ],
    "expansion": [
        "해외 진출",
        "글로벌",
        "시장 확대",
        "사업 확대",
        "센터 설립",
        "법인 설립",
        "거점",
        "진출",
    ],
}

_HIGH_THRESHOLD = 0.65
_MEDIUM_THRESHOLD = 0.40
_CLUSTER_SIZE_SATURATION = 5

_llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_completion_tokens=400)

_CLASSIFY_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 기사 클러스터의 이벤트 타입을 판단해주세요.
※ 등급은 시스템이 노출도로 결정하므로 판단하지 마세요.

## 기사 정보
ARTICLES_TEXT_PLACEHOLDER

## 클러스터 신호
- 동일 사건 보도 매체 수: CLUSTER_SIZE_PLACEHOLDER건
- 회사 직접 언급: COMPANY_MENTION_PLACEHOLDER

## 이벤트 타입 판단 기준
- partnership: 기업 간 협력, MOU, 공동사업
- ma: 인수, 합병, 지분 인수, 투자 유치
- personnel: 채용, 인사, 임원, 조직개편
- tech_release: 신기술, 신제품, 플랫폼, 솔루션 출시
- regulation: 법규, 정책, 규제, 인증
- contract: 수주, 고객 계약, 사업자 선정
- financial: 실적, 매출, 영업이익, 투자계획
- expansion: 해외 진출, 센터 설립, 시장 확대

복수 해당 시 본문에서 가장 크게 다루는 측면 하나만 선택하세요.
애매하면 tech_release보다 partnership, contract, financial, expansion을 우선하세요.

JSON으로만 응답:
{"event_type": "EVENT_TYPES_PLACEHOLDER", "reasoning": "1줄 근거"}\
""".replace("EVENT_TYPES_PLACEHOLDER", "|".join(EVENT_TYPES))


def compute_exposure(
    cluster_articles: list[dict[str, Any]],
    company: str,
) -> dict[str, Any]:
    """클러스터 메타데이터에서 노출도 점수와 밴드를 계산한다."""
    if not cluster_articles:
        return _zero_exposure()

    cluster_size = len(cluster_articles)
    cluster_size_score = min(cluster_size / _CLUSTER_SIZE_SATURATION, 1.0)
    credibility_max = max(
        min(max(article.get("credibility_score") or 0.0, 0.0), 1.0)
        for article in cluster_articles
    )

    aliases = company_aliases(company)
    company_mention_count = sum(
        1
        for article in cluster_articles
        if any(
            alias in f"{article.get('title') or ''} {article.get('content') or ''}"
            for alias in aliases
        )
    )
    company_mention_score = min(company_mention_count / max(cluster_size, 1), 1.0)

    score = (
        0.50 * cluster_size_score
        + 0.30 * credibility_max
        + 0.20 * company_mention_score
    )

    if score >= _HIGH_THRESHOLD:
        band = "high"
    elif score >= _MEDIUM_THRESHOLD:
        band = "medium"
    else:
        band = "low"

    return {
        "exposure_score": round(score, 3),
        "exposure_band": band,
        "cluster_size": cluster_size,
        "company_mention_count": company_mention_count,
        "credibility_max": round(credibility_max, 3),
    }


class ClassificationAgent:
    """클러스터를 company, sector, event_type 관점으로 분류한다."""

    def classify(
        self,
        cluster_id: int,
        representative_id: int,
        cluster_article_ids: list[int] | None = None,
        company: str = "",
        peer_id: str = "",
    ) -> dict[str, Any]:
        start = time.time()
        ids_to_load = list({representative_id, *(cluster_article_ids or [])})
        articles = get_articles_by_ids(ids_to_load)

        if not articles:
            return _default_result()

        rep = next(
            (article for article in articles if article["id"] == representative_id),
            articles[0],
        )
        target_company = company or _first_company(rep) or peer_id or rep.get("peer_id", "")

        haystack = f"{rep.get('title', '')} {rep.get('content', '')}"
        matched_sectors = match_sectors(haystack)
        sector = primary_sector(haystack)
        exposure = compute_exposure(articles, target_company)
        event_type, reasoning = self._classify_event_type(rep, exposure)

        result = {
            "company": target_company,
            "sector": sector,
            "sectors": matched_sectors,
            "exposure_score": exposure["exposure_score"],
            "exposure_band": exposure["exposure_band"],
            "signals": {
                "cluster_size": exposure["cluster_size"],
                "company_mention_count": exposure["company_mention_count"],
                "credibility_max": exposure["credibility_max"],
            },
            "event_type": event_type,
            "reasoning": reasoning,
            "importance": exposure["exposure_band"],
            "importance_score": exposure["exposure_score"],
        }

        update_classification(
            representative_id,
            result["importance"],
            result["importance_score"],
        )

        elapsed = int((time.time() - start) * 1000)
        log.info(
            "분류 완료 | cluster=%d company=%s sector=%s band=%s score=%.2f event=%s elapsed=%dms",
            cluster_id,
            target_company,
            sector,
            exposure["exposure_band"],
            exposure["exposure_score"],
            event_type,
            elapsed,
        )
        return result

    def _classify_event_type(
        self,
        rep_article: dict[str, Any],
        exposure: dict[str, Any],
    ) -> tuple[str, str]:
        text = f"{rep_article.get('title', '')} {rep_article.get('content', '')}"
        rule_event_type, rule_reasoning = _classify_event_type_rule_based(text)

        if rule_event_type:
            return rule_event_type, rule_reasoning

        articles_text = _format_articles([rep_article])
        company_mention_text = (
            f"{exposure['company_mention_count']}건 "
            f"(cluster_size={exposure['cluster_size']})"
        )
        prompt = (
            _CLASSIFY_PROMPT.replace("ARTICLES_TEXT_PLACEHOLDER", articles_text)
            .replace("CLUSTER_SIZE_PLACEHOLDER", str(exposure["cluster_size"]))
            .replace("COMPANY_MENTION_PLACEHOLDER", company_mention_text)
        )

        try:
            response = _llm.invoke(prompt)
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            data = _parse_json(content)

            event_type = data.get("event_type", "tech_release")
            if event_type not in EVENT_TYPES:
                event_type = "tech_release"

            return event_type, data.get("reasoning", "")
        except Exception as exc:
            log.warning("event_type 분류 실패, tech_release 기본값 | error=%s", exc)
            return "tech_release", ""


def _classify_event_type_rule_based(text: str) -> tuple[str | None, str]:
    matched: list[tuple[str, int]] = []

    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in text)
        if count > 0:
            matched.append((event_type, count))

    if not matched:
        return None, ""

    matched.sort(key=lambda item: item[1], reverse=True)

    if len(matched) == 1 or matched[0][1] > matched[1][1]:
        return matched[0][0], f"규칙 기반 키워드 매칭: {matched[0][0]}"

    return None, "복수 event_type 후보가 유사하여 LLM 판단 필요"


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines = []

    for index, article in enumerate(articles, 1):
        credibility_score = article.get("credibility_score")
        credibility_text = (
            f"{credibility_score:.2f}" if credibility_score is not None else "미계산"
        )
        lines.append(
            f"[{index}] 제목: {article['title']}\n"
            f"    출처: {article['source_name']} (신뢰도: {credibility_text})\n"
            f"    내용: {(article.get('content') or '')[:300]}"
        )

    return "\n\n".join(lines)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text.strip())


def _first_company(article: dict[str, Any]) -> str:
    company = article.get("company")

    if isinstance(company, list) and company:
        return str(company[0])

    if isinstance(company, str):
        try:
            parsed = json.loads(company)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except json.JSONDecodeError:
            return company

    return ""


def _zero_exposure() -> dict[str, Any]:
    return {
        "exposure_score": 0.0,
        "exposure_band": "low",
        "cluster_size": 0,
        "company_mention_count": 0,
        "credibility_max": 0.0,
    }


def _default_result() -> dict[str, Any]:
    return {
        "company": "",
        "sector": "other",
        "sectors": ["other"],
        "exposure_score": 0.0,
        "exposure_band": "low",
        "signals": {
            "cluster_size": 0,
            "company_mention_count": 0,
            "credibility_max": 0.0,
        },
        "event_type": "tech_release",
        "reasoning": "",
        "importance": "low",
        "importance_score": 0.0,
    }


__all__ = ["ClassificationAgent", "EVENT_TYPES", "SECTOR_IDS", "compute_exposure"]
