# 작성일: 2026-05-19
# 작성자: 박지원
# 변경이력:
#   2026-05-19 박지원 — 분류 전처리 플로우 구축
#   2026-06-11 최종민 — ChatOpenAI 지연 임포트로 import 체인 차단, gpt-4o 하드코딩 팩토리 이행
"""분류 전처리 v4 — 섹터 태깅 + 노출도 점수 + event_type 규칙 기반 분류."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.config.companies import company_aliases
from src.config.event_types import (
    EVENT_TYPE_KEYWORDS,
    EVENT_TYPE_TIE_BREAK_PRIORITY,
    EVENT_TYPES,
    HIGH_IMPACT_KEYWORDS,
    MEDIUM_IMPACT_KEYWORDS,
)
from src.config.global_companies import global_company_aliases
from src.config.sectors import SECTOR_IDS, match_sectors, primary_sector
from src.db.article_store import get_articles_by_ids, update_classification

log = logging.getLogger(__name__)

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

    aliases = _configured_company_aliases(company)
    company_mention_count = sum(
        1
        for a in cluster_articles
        if any(alias in f"{a.get('title') or ''} {a.get('content') or ''}" for alias in aliases)
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


def _configured_company_aliases(company: str) -> list[str]:
    if not company:
        return []

    aliases = [*company_aliases(company), *global_company_aliases(company)]
    return list(dict.fromkeys(alias for alias in aliases if alias))


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


class ClusterClassifier:
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

        return "company", "규칙 매칭 없음, company 기본값"


def classify_article_text(
    title: str,
    content: str,
    company: str = "",
    *,
    source_type: str = "",
) -> dict[str, Any]:
    """단일 기사 텍스트를 운영과 동일한 로직으로 분류한다 (DB 미접근).

    ``ClusterClassifier.classify`` 의 텍스트 전용 버전 — 데모/외부 단발 분류용.
    rule 기반 event_type 판정, exposure/impact 산식 동일. 단일 기사이므로
    cluster_size=1 (exposure 는 낮고, 큰 사건은 impact 로 보정됨).

    Args:
        title: 기사 제목
        content: 기사 본문
        company: peer id (회사 직접 언급 카운트용, 선택)
        source_type: 출처 유형(dart/ir/official 등, impact 보정용, 선택)

    Returns:
        event_type / sector(s) / exposure·impact·importance score+band / reasoning / signals
    """
    rep: dict[str, Any] = {
        "title": title or "",
        "content": content or "",
        "company": company or "",
        "source_type": source_type or "",
        "source_name": "",
    }
    matched_sectors = _matched_sectors(rep, title=rep["title"], content=rep["content"])
    exposure = compute_exposure([rep], company)
    event_type, reasoning = ClusterClassifier()._classify_event_type(rep, exposure)
    impact = compute_article_impact(rep, event_type)
    importance_score = max(exposure["exposure_score"], impact["impact_score"])

    return {
        "title": rep["title"],
        "sector": matched_sectors[0],
        "sectors": matched_sectors,
        "event_type": event_type,
        "reasoning": reasoning,
        "exposure_score": exposure["exposure_score"],
        "exposure_band": exposure["exposure_band"],
        "impact_score": impact["impact_score"],
        "impact_band": impact["impact_band"],
        "importance_score": importance_score,
        "importance": _to_band(importance_score),
        "signals": {
            "cluster_size": exposure["cluster_size"],
            "company_mention_count": exposure["company_mention_count"],
            "impact_signals": impact["impact_signals"],
        },
    }


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

    matched.sort(key=lambda x: (EVENT_TYPE_TIE_BREAK_PRIORITY.get(x[0], 99), -x[1]))
    return matched


def _select_content_event_type(matches: list[tuple[str, int]]) -> tuple[str, int]:
    strong_event_types = {"ma", "contract", "financial", "partnership", "regulation", "expansion"}
    strong_matches = [
        (event_type, count)
        for event_type, count in matches
        if event_type in strong_event_types and count >= 2
    ]
    if strong_matches:
        strong_matches.sort(key=lambda x: (EVENT_TYPE_TIE_BREAK_PRIORITY.get(x[0], 99), -x[1]))
        tech_count = next(
            (count for event_type, count in matches if event_type == "tech_release"),
            0,
        )
        if tech_count >= strong_matches[0][1] + 2:
            return "tech_release", tech_count
        return strong_matches[0]

    count_first = sorted(
        matches,
        key=lambda x: (-x[1], EVENT_TYPE_TIE_BREAK_PRIORITY.get(x[0], 99)),
    )
    return count_first[0]


def _to_band(score: float) -> str:
    if score >= _HIGH_THRESHOLD:
        return "high"

    if score >= _MEDIUM_THRESHOLD:
        return "medium"

    return "low"


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
    event_type, reasoning = ClusterClassifier()._classify_event_type(rep, exposure)
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
    "ClusterClassifier",
    "EVENT_TYPES",
    "SECTOR_IDS",
    "classify_preprocessed_cluster",
    "compute_article_impact",
    "compute_exposure",
]
