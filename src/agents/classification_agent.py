"""분류 에이전트 v4 — 섹터 태깅 + 노출도 점수 + event_type 규칙 기반 분류."""

import json
import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.sector_keywords import SECTOR_IDS, match_sectors, primary_sector
from src.config.global_companies import GLOBAL_COMPANY_ALIASES
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
    "company",
]

EVENT_TYPE_PRIORITY = {
    "ma": 0,
    "contract": 1,
    "financial": 2,
    "partnership": 3,
    "regulation": 4,
    "expansion": 5,
    "personnel": 6,
    "tech_release": 7,
    "company": 8,
}

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
        "선보",
        "런칭",
        "상용화",
        "정식 출시",
        "신제품",
        "신기술",
        "신서비스",
        "업그레이드",
        "적용",
        "실증",
        "PoC",
        "상용화 성과",
        "현장 점검",
    ],
    "regulation": [
        "규제",
        "정책",
        "법안",
        "가이드라인",
        "정부 인증",
        "국제표준",
        "표준 인증",
        "컴플라이언스",
    ],
    "contract": [
        "수주",
        "계약",
        "선정",
        "공급 계약",
        "구축 사업",
        "사업자 선정",
        "우선협상대상자",
        "프로젝트 수주",
    ],
    "financial": [
        "매출",
        "영업이익",
        "실적",
        "잠정실적",
        "분기 실적",
        "연간 실적",
        "흑자",
        "적자",
        "가이던스",
    ],
    "expansion": [
        "해외 진출",
        "글로벌 진출",
        "시장 확대",
        "사업 확대",
        "센터 설립",
        "법인 설립",
        "신시장",
        "현지화",
    ],
    "company": [
        "경영 전략",
        "사업 전략",
        "중장기",
        "비전",
        "기업가치",
        "지배구조",
        "그룹 내",
        "CEO 메시지",
        "주주총회",
    ],
}

HIGH_IMPACT_KEYWORDS = [
    "대규모 수주",
    "대형 수주",
    "메가딜",
    "우선협상대상자",
    "단일판매",
    "공급계약",
    "공시",
    "인수합병",
    "M&A",
    "지분 인수",
    "지분 투자",
    "전략적 제휴",
    "해외 진출",
    "신규 법인",
    "실적 발표",
    "영업이익",
    "흑자전환",
    "적자전환",
]

MEDIUM_IMPACT_KEYWORDS = [
    "수주",
    "계약",
    "협약",
    "MOU",
    "출시",
    "공개",
    "고도화",
    "조직개편",
    "임원",
    "채용",
    "시장 확대",
    "투자계획",
]

_llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_completion_tokens=400)
_PROMPT_VERSION = "classify-v3.0"

_CLASSIFY_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 기사 클러스터의 이벤트 타입을 판단해주세요.
※ 등급은 시스템이 노출도와 기사 내용 중요도로 결정하므로 판단하지 마세요.

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
- company: 회사 전반의 경영 동향, 사업 방향, IR 해설, 재무 상태, 지배구조, 브랜드/비전 변화

복수 해당 시 본문에서 가장 크게 다루는 측면 하나만 선택하세요.
구체적 이벤트가 있으면 company보다 partnership, ma, personnel, tech_release,
regulation, contract, financial, expansion을 우선하세요.
애매하면 tech_release보다 partnership, contract, financial, expansion을 우선하세요.

JSON으로만 응답:
{"event_type": "EVENT_TYPES_PLACEHOLDER", "reasoning": "1줄 근거"}\
""".replace("EVENT_TYPES_PLACEHOLDER", "|".join(EVENT_TYPES))


_HIGH_THRESHOLD = 0.65
_MEDIUM_THRESHOLD = 0.40

_CLUSTER_SIZE_SATURATION = 5


# exposure_score는 기사 노출 강도만 측정한다.
# 단독 기사라도 전략적으로 중요한 내용은 impact_score로 보정한다.
# 최종 importance_score는 max(exposure_score, impact_score)를 사용한다.


def compute_exposure(
    cluster_articles: list[dict[str, Any]],
    company: str,
) -> dict[str, Any]:
    """클러스터 메타데이터에서 노출도 점수와 밴드를 계산한다."""
    if not cluster_articles:
        return _zero_exposure()

    cluster_size = len(cluster_articles)
    cluster_size_score = min(cluster_size / _CLUSTER_SIZE_SATURATION, 1.0)

    company_aliases = _company_aliases(company)
    company_mention_count = sum(
        1
        for a in cluster_articles
        if any(
            alias in f"{a.get('title') or ''} {a.get('content') or ''}" for alias in company_aliases
        )
    )
    company_mention_score = min(company_mention_count / max(cluster_size, 1), 1.0)

    score = 0.70 * cluster_size_score + 0.30 * company_mention_score

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
    }


def compute_article_impact(
    article: dict[str, Any],
    event_type: str,
) -> dict[str, Any]:
    """기사 1건의 내용 자체가 갖는 전략적 중요도를 계산한다."""
    text = f"{article.get('title') or ''} {article.get('content') or ''}"
    source_type = str(article.get("source_type") or "").strip().lower()

    score = 0.20
    signals: list[str] = []

    if source_type in {"dart", "ir"}:
        score = max(score, 0.80)
        signals.append(f"source_type:{source_type}")
    elif source_type in {"official", "company_site", "securities_report"}:
        score = max(score, 0.65)
        signals.append(f"source_type:{source_type}")

    event_score = {
        "ma": 0.80,
        "contract": 0.75,
        "financial": 0.75,
        "expansion": 0.70,
        "partnership": 0.65,
        "regulation": 0.65,
        "tech_release": 0.60,
        "personnel": 0.55,
        "company": 0.45,
    }.get(event_type, 0.30)
    score = max(score, event_score)
    signals.append(f"event_type:{event_type}")

    if any(keyword in text for keyword in HIGH_IMPACT_KEYWORDS):
        score = max(score, 0.85)
        signals.append("high_impact_keyword")
    elif any(keyword in text for keyword in MEDIUM_IMPACT_KEYWORDS):
        score = max(score, 0.65)
        signals.append("medium_impact_keyword")

    score = round(min(score, 1.0), 3)
    return {
        "impact_score": score,
        "impact_band": _to_band(score),
        "impact_signals": signals,
    }


def _company_aliases(company: str) -> list[str]:
    aliases = {
        "삼성SDS": ["삼성SDS", "삼성 SDS", "Samsung SDS", "삼성에스디에스"],
        "samsung_sds": ["삼성SDS", "삼성 SDS", "Samsung SDS", "삼성에스디에스"],
        "LG CNS": ["LG CNS", "LGCNS", "엘지씨엔에스"],
        "lg_cns": ["LG CNS", "LGCNS", "엘지씨엔에스"],
        "현대오토에버": ["현대오토에버", "현대 오토에버", "Hyundai Autoever"],
        "hyundai_autoever": ["현대오토에버", "현대 오토에버", "Hyundai Autoever"],
        "포스코DX": ["포스코DX", "포스코 DX", "POSCO DX"],
        "posco_dx": ["포스코DX", "포스코 DX", "POSCO DX"],
        "SK AX": ["SK AX", "SK에이엑스", "SK C&C", "SK㈜ C&C", "에스케이 씨앤씨"],
        "sk_ax": ["SK AX", "SK에이엑스", "SK C&C", "SK㈜ C&C", "에스케이 씨앤씨"],
        **GLOBAL_COMPANY_ALIASES,
    }

    if not company:
        return []

    return aliases.get(company, [company])


def _matched_sectors(article: dict[str, Any], *, title: str, content: str) -> list[str]:
    stored = _string_list(article.get("matched_sectors"))
    valid_stored = [sector for sector in stored if sector in SECTOR_IDS and sector != "other"]
    if valid_stored:
        return valid_stored

    scores: dict[str, int] = {}
    for sector in match_sectors(title):
        if sector != "other":
            scores[sector] = scores.get(sector, 0) + 3
    for sector in match_sectors(content):
        if sector != "other":
            scores[sector] = scores.get(sector, 0) + 1

    if not scores:
        return ["other"]

    return sorted(scores, key=lambda sector: (-scores[sector], SECTOR_IDS.index(sector)))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    return []


def _zero_exposure() -> dict[str, Any]:
    return {
        "exposure_score": 0.0,
        "exposure_band": "low",
        "cluster_size": 0,
        "company_mention_count": 0,
    }


class ClassificationAgent:
    """클러스터를 섹터, 노출도, 이벤트 타입 기준으로 분류한다."""

    def classify(
        self,
        cluster_id: int,
        representative_id: int,
        cluster_article_ids: list[int] | None = None,
        company: str = "",
    ) -> dict[str, Any]:
        start = time.time()

        ids_to_load = list({representative_id, *(cluster_article_ids or [])})
        articles = get_articles_by_ids(ids_to_load)
        if not articles:
            return _default_result()

        rep = next((a for a in articles if a["id"] == representative_id), articles[0])

        target_company = company or rep.get("company", "")

        title = str(rep.get("title") or "")
        content = str(rep.get("content") or "")
        matched_sectors = _matched_sectors(rep, title=title, content=content)
        sector = matched_sectors[0]

        exposure = compute_exposure(articles, target_company)

        event_type, reasoning = self._classify_event_type(rep, exposure)
        impact = compute_article_impact(rep, event_type)
        importance_score = max(exposure["exposure_score"], impact["impact_score"])
        importance_band = _to_band(importance_score)

        result = {
            "title": rep.get("title") or "",
            "url": rep.get("url") or "",
            "sector": sector,
            "sectors": matched_sectors,
            "exposure_score": exposure["exposure_score"],
            "exposure_band": exposure["exposure_band"],
            "impact_score": impact["impact_score"],
            "impact_band": impact["impact_band"],
            "signals": {
                "cluster_size": exposure["cluster_size"],
                "company_mention_count": exposure["company_mention_count"],
                "impact_signals": impact["impact_signals"],
            },
            "event_type": event_type,
            "reasoning": reasoning,
            "importance": importance_band,
            "importance_score": importance_score,
        }

        update_classification(
            representative_id,
            result["importance"],
            result["importance_score"],
        )

        elapsed = int((time.time() - start) * 1000)
        log.info(
            "분류 완료 | cluster=%d company=%s sector=%s importance=%s "
            "score=%.2f exposure=%.2f impact=%.2f event=%s elapsed=%dms",
            cluster_id,
            target_company,
            sector,
            importance_band,
            importance_score,
            exposure["exposure_score"],
            impact["impact_score"],
            event_type,
            elapsed,
        )

        return result

    def _classify_event_type(
        self,
        rep_article: dict[str, Any],
        exposure: dict[str, Any],
    ) -> tuple[str, str]:
        rule_event_type, rule_reasoning = _classify_event_type_rule_based(
            title=str(rep_article.get("title") or ""),
            content=str(rep_article.get("content") or ""),
        )

        if rule_event_type:
            return rule_event_type, rule_reasoning

        articles_text = _format_articles([rep_article])
        company_mention_text = (
            f"{exposure['company_mention_count']}건 (cluster_size={exposure['cluster_size']})"
        )

        prompt = (
            _CLASSIFY_PROMPT.replace("ARTICLES_TEXT_PLACEHOLDER", articles_text)
            .replace("CLUSTER_SIZE_PLACEHOLDER", str(exposure["cluster_size"]))
            .replace("COMPANY_MENTION_PLACEHOLDER", company_mention_text)
        )

        try:
            from src.observability import tracing_config

            response = _llm.invoke(
                prompt,
                config=tracing_config(
                    agent="ClassificationAgent",
                    prompt_version=_PROMPT_VERSION,
                    cluster_size=exposure["cluster_size"],
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            data = _parse_json(content)

            event_type = data.get("event_type", "company")
            if event_type not in EVENT_TYPES:
                event_type = "company"

            return event_type, data.get("reasoning", "")

        except Exception as e:
            log.warning("event_type 분류 실패, company 기본값 | error=%s", e)
            return "company", ""


def _classify_event_type_rule_based(
    text: str | None = None,
    *,
    title: str = "",
    content: str = "",
) -> tuple[str | None, str]:
    if text is not None:
        title = text
        content = ""

    title_matches = _event_type_matches(title)
    if title_matches:
        event_type, count = title_matches[0]
        return event_type, f"규칙 기반 제목 키워드 매칭: {event_type}({count})"

    content_matches = _event_type_matches(content)
    if content_matches:
        event_type, count = _select_content_event_type(content_matches)
        return event_type, f"규칙 기반 본문 키워드 매칭: {event_type}({count})"

    return None, ""


def _event_type_matches(text: str) -> list[tuple[str, int]]:
    matched: list[tuple[str, int]] = []

    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in text)
        if count > 0:
            matched.append((event_type, count))

    matched.sort(key=lambda x: (EVENT_TYPE_PRIORITY.get(x[0], 99), -x[1]))
    return matched


def _select_content_event_type(matches: list[tuple[str, int]]) -> tuple[str, int]:
    strong_event_types = {"ma", "contract", "financial", "partnership", "regulation", "expansion"}
    strong_matches = [
        (event_type, count)
        for event_type, count in matches
        if event_type in strong_event_types and count >= 2
    ]
    if strong_matches:
        strong_matches.sort(key=lambda x: (EVENT_TYPE_PRIORITY.get(x[0], 99), -x[1]))
        tech_count = next(
            (count for event_type, count in matches if event_type == "tech_release"),
            0,
        )
        if tech_count >= strong_matches[0][1] + 2:
            return "tech_release", tech_count
        return strong_matches[0]

    count_first = sorted(matches, key=lambda x: (-x[1], EVENT_TYPE_PRIORITY.get(x[0], 99)))
    return count_first[0]


def _to_band(score: float) -> str:
    if score >= _HIGH_THRESHOLD:
        return "high"

    if score >= _MEDIUM_THRESHOLD:
        return "medium"

    return "low"


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines = []

    for i, a in enumerate(articles, 1):
        lines.append(
            f"[{i}] 제목: {a['title']}\n"
            f"    출처: {a['source_name']}\n"
            f"    내용: {(a.get('content') or '')[:300]}"
        )

    return "\n\n".join(lines)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text.strip())


def _default_result() -> dict[str, Any]:
    return {
        "sector": "other",
        "sectors": ["other"],
        "exposure_score": 0.0,
        "exposure_band": "low",
        "impact_score": 0.0,
        "impact_band": "low",
        "signals": {
            "cluster_size": 0,
            "company_mention_count": 0,
            "impact_signals": [],
        },
        "event_type": "company",
        "reasoning": "",
        "importance": "low",
        "importance_score": 0.0,
    }


def classify_preprocessed_cluster(
    cluster_id: int,
    representative_id: int,
    cluster_articles: list[dict[str, Any]],
    company: str,
) -> dict[str, Any]:
    """로컬 JSON 전처리 결과의 클러스터를 v4 규칙으로 분류한다."""

    if not cluster_articles:
        return _default_result()

    rep = next(
        (
            article
            for article in cluster_articles
            if int(article.get("preprocess_id") or article.get("id") or 0) == int(representative_id)
        ),
        cluster_articles[0],
    )

    text = f"{rep.get('title') or ''} {rep.get('content') or ''}"
    sectors = match_sectors(text)
    sector = primary_sector(text)
    exposure = compute_exposure(cluster_articles, company)
    event_type, reasoning = ClassificationAgent()._classify_event_type(rep, exposure)
    impact = compute_article_impact(rep, event_type)
    importance_score = max(exposure["exposure_score"], impact["impact_score"])
    importance_band = _to_band(importance_score)

    return {
        "cluster_id": cluster_id,
        "representative_id": representative_id,
        "company": company,
        "sector": sector,
        "sectors": sectors,
        "exposure_score": exposure["exposure_score"],
        "exposure_band": exposure["exposure_band"],
        "impact_score": impact["impact_score"],
        "impact_band": impact["impact_band"],
        "signals": {
            "cluster_size": exposure["cluster_size"],
            "company_mention_count": exposure["company_mention_count"],
            "impact_signals": impact["impact_signals"],
        },
        "event_type": event_type,
        "reasoning": reasoning,
        "importance": importance_band,
        "importance_score": importance_score,
        "title": rep.get("title"),
        "url": rep.get("url"),
    }


__all__ = [
    "ClassificationAgent",
    "EVENT_TYPES",
    "SECTOR_IDS",
    "classify_preprocessed_cluster",
    "compute_article_impact",
    "compute_exposure",
]
