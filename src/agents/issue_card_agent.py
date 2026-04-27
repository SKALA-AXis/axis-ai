"""이슈 카드 생성 에이전트 — GPT-4o 호출."""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.db.article_store import _generate_card_id, get_articles_by_ids

log = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_completion_tokens=1024)

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

_MAX_CLUSTER_ARTICLES = 3  # 클러스터 내 참고 기사 최대 수


class IssueCardAgent:
    """클러스터 상위 기사들로 이슈 카드를 생성한다."""

    def generate(
        self,
        cluster_id: int,
        representative_id: int,
        peer_id: str,
        classification: dict[str, Any],
        cluster_article_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """이슈 카드를 생성한다.

        Args:
            cluster_id: 클러스터 ID.
            representative_id: 대표 기사 ID (가장 신뢰도 높은 기사).
            peer_id: 'samsung_sds' | 'lg_cns'.
            classification: ClassificationAgent 결과.
            cluster_article_ids: 클러스터 내 전체 기사 ID 목록 (없으면 대표 기사만 사용).

        Returns:
            IssueCard dict (저장 전 validation 없는 상태).
        """
        # 대표 기사 포함 최대 3건 조회 (credibility_score DESC 정렬)
        ids_to_fetch = _build_fetch_ids(representative_id, cluster_article_ids)
        articles = get_articles_by_ids(ids_to_fetch)
        if not articles:
            return {}

        articles_text = _format_articles(articles)
        prompt = _ISSUE_CARD_PROMPT.replace("{articles_text}", articles_text)

        try:
            response = _llm.invoke(prompt)
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            card_data = _parse_json(content)

            card = {
                "id": _generate_card_id(peer_id),
                "peer_id": peer_id,
                "cluster_id": cluster_id,
                "title": card_data.get("title", articles[0]["title"][:100]),
                "summary_lines": card_data.get("summary_lines", []),
                "event_type": card_data.get("event_type", classification.get("event_type", "tech")),
                # v3: 분류 결과의 sector·exposure 정보를 카드에 그대로 전파
                "sector": classification.get("sector", "other"),
                "sectors": classification.get("sectors", ["other"]),
                "exposure_score": classification.get("exposure_score", 0.0),
                "exposure_band": classification.get("exposure_band", "low"),
                "signals": classification.get("signals", {}),
                # 등급 자체는 v3에서 폐기되었으나, DB 컬럼 호환을 위해 노출도 밴드를 저장
                "importance": classification.get("importance", "low"),
                "importance_score": classification.get("importance_score", 0.0),
                "sources": card_data.get("sources", _default_sources(articles)),
            }

            log.info(
                "이슈카드 생성 완료 | id=%s sector=%s band=%s event=%s sources=%d",
                card["id"],
                card["sector"],
                card["exposure_band"],
                card["event_type"],
                len(articles),
            )
            return card

        except Exception as e:
            log.error("이슈카드 생성 실패 | cluster=%d error=%s", cluster_id, e)
            return {}


def _build_fetch_ids(representative_id: int, cluster_article_ids: list[int] | None) -> list[int]:
    """대표 기사를 앞에 두고 최대 3건의 ID 목록을 만든다."""
    if not cluster_article_ids:
        return [representative_id]
    # 대표 기사 먼저, 나머지 중 대표 제외 후 합치기
    others = [aid for aid in cluster_article_ids if aid != representative_id]
    return [representative_id, *others][:_MAX_CLUSTER_ARTICLES]


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(
            f"[{i}] 제목: {a['title']}\n"
            f"    출처: {a['source_name']} (신뢰도: {a.get('credibility_score', 0):.2f})"
            f" | URL: {a['url']}\n"
            f"    내용: {(a.get('content') or '')[:600]}"
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
            "credibility_score": a.get("credibility_score", 0.5),
        }
        for i, a in enumerate(articles)
    ]
