"""분류 에이전트 v3 — 4개 섹터 태깅 + 결정적 노출도 점수.

v3 §1.2 (4개 섹터) + §1.4 (긴급/주목 폐기 → 노출도·건수·신뢰도) 반영.

- 섹터 분류: 키워드 매칭(결정적) + LLM 보조(애매한 경우만)
- 노출도 점수: cluster_size·peer_count·source_count·credibility_max를 결정적 공식으로 계산
- LLM은 event_type 태깅과 노출도가 애매할 때만 호출
"""

import json
import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.sector_keywords import SECTOR_IDS, match_sectors, primary_sector
from src.db.article_store import get_articles_by_ids, update_classification

log = logging.getLogger(__name__)

# 이벤트 타입 taxonomy (DB 컬럼 호환을 위해 유지, v3에서는 부차적 메타데이터)
EVENT_TYPES = ["partnership", "ma", "personnel", "tech", "regulation", "new_biz"]

_llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_completion_tokens=400)

_CLASSIFY_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 기사 클러스터의 이벤트 타입을 판단해주세요.
※ 등급(긴급/주목)은 시스템이 노출도로 결정하므로 판단하지 마세요.

## 기사 정보
ARTICLES_TEXT_PLACEHOLDER

## 클러스터 신호
- 동일 사건 보도 매체 수: CLUSTER_SIZE_PLACEHOLDER건
- Peer사 직접 언급: PEER_MENTION_PLACEHOLDER

## 이벤트 타입 판단 기준 (하나만 선택)
- partnership: 기업간 협력·MOU·공동사업
- ma: 인수·합병·지분 인수·투자 유치
- personnel: 채용·인사·임원·조직
- tech: 신기술·신제품·플랫폼 출시
- regulation: 법규·정책·규제
- new_biz: 신사업·수주·실적·시장 확장

복수 해당 시 본문이 가장 크게 다루는 측면 선택.
애매하면 tech 대신 partnership/new_biz/personnel 우선.

JSON으로만 응답:
{"event_type": "EVENT_TYPES_PLACEHOLDER", "reasoning": "1줄 근거"}\
""".replace("EVENT_TYPES_PLACEHOLDER", "|".join(EVENT_TYPES))


# ── 노출도 결정적 점수식 ──────────────────────────────────────────
# v3 §1.4: "관련 기사 건수 + 신뢰성 있는 출처 노출 빈도" 기반.
# exposure_score ∈ [0, 1]
#   = 0.40 * cluster_size_score          (보도 매체 수)
#   + 0.30 * credibility_max             (가장 신뢰도 높은 출처)
#   + 0.20 * peer_mention_score          (Peer사 직접 언급 강도)
#   + 0.10 * source_diversity_score      (Tier 1 vs Tier 2 다양성)
#
# 노출도 밴드 (DB importance 컬럼 호환)
#   high   : exposure_score ≥ 0.65
#   medium : 0.40 ≤ exposure_score < 0.65
#   low    : exposure_score < 0.40
#
# 등급 ≠ 긴급도. 단순히 "얼마나 많이/신뢰성 있게 보도되었는지"의 노출 강도.

_HIGH_THRESHOLD = 0.65
_MEDIUM_THRESHOLD = 0.40

_CLUSTER_SIZE_SATURATION = 5  # 5건 이상이면 cluster_size_score = 1.0


def compute_exposure(
    cluster_articles: list[dict[str, Any]],
    peer_id: str,
) -> dict[str, Any]:
    """클러스터 메타데이터에서 노출도 점수와 밴드를 결정적으로 계산한다."""
    if not cluster_articles:
        return _zero_exposure()

    cluster_size = len(cluster_articles)
    cluster_size_score = min(cluster_size / _CLUSTER_SIZE_SATURATION, 1.0)

    credibility_max = max((a.get("credibility_score") or 0.0) for a in cluster_articles)

    # peer 이름 직접 언급 카운트
    peer_aliases = _peer_aliases(peer_id)
    peer_mention_count = sum(
        1
        for a in cluster_articles
        if any(alias in (a.get("title") or "") + (a.get("content") or "") for alias in peer_aliases)
    )
    peer_mention_score = min(peer_mention_count / max(cluster_size, 1), 1.0)

    # 출처 tier 다양성 (Tier 1 출처가 1개 이상이면 0.5, 2개 이상 0.8, 3개 이상 1.0)
    tier1_count = sum(1 for a in cluster_articles if (a.get("source_tier") or 5) == 1)
    if tier1_count >= 3:
        diversity_score = 1.0
    elif tier1_count == 2:
        diversity_score = 0.8
    elif tier1_count == 1:
        diversity_score = 0.5
    else:
        diversity_score = 0.2

    score = (
        0.40 * cluster_size_score
        + 0.30 * credibility_max
        + 0.20 * peer_mention_score
        + 0.10 * diversity_score
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
        "peer_mention_count": peer_mention_count,
        "credibility_max": round(credibility_max, 3),
        "tier1_count": tier1_count,
    }


def _peer_aliases(peer_id: str) -> list[str]:
    aliases = {
        "samsung_sds": ["삼성SDS", "삼성 SDS", "Samsung SDS", "삼성에스디에스"],
        "lg_cns": ["LG CNS", "LGCNS", "엘지씨엔에스"],
        "hyundai_autoever": ["현대오토에버", "현대 오토에버", "Hyundai Autoever"],
        "posco_dx": ["포스코DX", "포스코 DX", "POSCO DX"],
    }
    return aliases.get(peer_id, [])


def _zero_exposure() -> dict[str, Any]:
    return {
        "exposure_score": 0.0,
        "exposure_band": "low",
        "cluster_size": 0,
        "peer_mention_count": 0,
        "credibility_max": 0.0,
        "tier1_count": 0,
    }


# ── 분류 에이전트 ─────────────────────────────────────────────────


class ClassificationAgent:
    """클러스터를 4개 섹터로 태깅하고 노출도를 결정적으로 계산한다."""

    def classify(
        self,
        cluster_id: int,
        representative_id: int,
        cluster_article_ids: list[int] | None = None,
        peer_id: str = "",
    ) -> dict[str, Any]:
        """v3 분류: sector + exposure + event_type.

        Args:
            cluster_id: 클러스터 ID.
            representative_id: 대표 기사 ID.
            cluster_article_ids: 클러스터 전체 ID (노출도 계산용).
            peer_id: Peer사 ID.

        Returns:
            {sector, sectors, exposure_score, exposure_band, importance,
             importance_score, event_type, reasoning, signals}
        """
        start = time.time()
        ids_to_load = list({representative_id, *(cluster_article_ids or [])})
        articles = get_articles_by_ids(ids_to_load)
        if not articles:
            return _default_result()

        rep = next((a for a in articles if a["id"] == representative_id), articles[0])

        # 1) 섹터 (결정적 키워드 매칭)
        haystack = f"{rep.get('title', '')} {rep.get('content', '')}"
        matched_sectors = match_sectors(haystack)
        sector = primary_sector(haystack)

        # 2) 노출도 (결정적 공식)
        exposure = compute_exposure(articles, peer_id or rep.get("peer_id", ""))

        # 3) event_type (LLM 호출 — 분포 정상화 목적)
        event_type, reasoning = self._classify_event_type(rep, exposure)

        result = {
            "sector": sector,
            "sectors": matched_sectors,
            "exposure_score": exposure["exposure_score"],
            "exposure_band": exposure["exposure_band"],
            "signals": {
                "cluster_size": exposure["cluster_size"],
                "peer_mention_count": exposure["peer_mention_count"],
                "credibility_max": exposure["credibility_max"],
                "tier1_count": exposure["tier1_count"],
            },
            "event_type": event_type,
            "reasoning": reasoning,
            # DB 호환: importance 컬럼에 노출도 밴드를 저장
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
            "분류 완료 | cluster=%d sector=%s band=%s score=%.2f event=%s elapsed=%dms",
            cluster_id,
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
        articles_text = _format_articles([rep_article])
        peer_mention_text = (
            f"{exposure['peer_mention_count']}건 (cluster_size={exposure['cluster_size']})"
        )
        prompt = (
            _CLASSIFY_PROMPT.replace("ARTICLES_TEXT_PLACEHOLDER", articles_text)
            .replace("CLUSTER_SIZE_PLACEHOLDER", str(exposure["cluster_size"]))
            .replace("PEER_MENTION_PLACEHOLDER", peer_mention_text)
        )
        try:
            response = _llm.invoke(prompt)
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            data = _parse_json(content)
            event_type = data.get("event_type", "tech")
            if event_type not in EVENT_TYPES:
                event_type = "tech"
            return event_type, data.get("reasoning", "")
        except Exception as e:
            log.warning("event_type 분류 실패, tech 기본값 | error=%s", e)
            return "tech", ""


# ── helpers ───────────────────────────────────────────────────────


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        credibility_score = a.get("credibility_score")
        credibility_text = (
            f"{credibility_score:.2f}" if credibility_score is not None else "미계산"
        )
        lines.append(
            f"[{i}] 제목: {a['title']}\n"
            f"    출처: {a['source_name']} (신뢰도: {credibility_text})\n"
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
        "signals": {
            "cluster_size": 0,
            "peer_mention_count": 0,
            "credibility_max": 0.0,
            "tier1_count": 0,
        },
        "event_type": "tech",
        "reasoning": "",
        "importance": "low",
        "importance_score": 0.0,
    }


__all__ = ["ClassificationAgent", "EVENT_TYPES", "SECTOR_IDS", "compute_exposure"]
