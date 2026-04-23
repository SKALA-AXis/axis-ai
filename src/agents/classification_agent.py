"""중요도 분류 에이전트 — GPT-4o 5개 축 가중치 기반."""

import json
import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI

from src.db.article_store import get_articles_by_ids, update_classification

log = logging.getLogger(__name__)

# 이벤트 타입 taxonomy (6개 고정)
EVENT_TYPES = ["partnership", "ma", "personnel", "tech", "regulation", "new_biz"]

_llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=512)

_EVENT_TYPES_STR = "|".join(EVENT_TYPES)

_CLASSIFY_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 기사 클러스터를 분석하여 중요도를 판단해주세요.

## 기사 정보
ARTICLES_TEXT_PLACEHOLDER

## 판단 기준 (5개 축 가중치)
1. 출처 신뢰도 (30%): credibility_score 기준 — 0.9 이상이면 가산
2. SK AX 사업 연관성 (25%): 에이전틱AI·제조AX·MSP 키워드 포함 여부
3. 신규성 (20%): 처음 보도되는 내용인지 여부 (최초 보도, 단독 표현 확인)
4. 사업 영향도 (15%): M&A·대형수주·전략적파트너십 여부
5. 경쟁사 연관성 (10%): 삼성SDS·LG CNS 직접 언급 여부

## 중요도 기준
- urgent (긴급): 종합 점수 80점 이상
- notable (주목): 50~79점
- reference (참고): 50점 미만

다음 JSON 형식으로만 응답하세요 (추가 텍스트 금지):
{{
  "importance": "urgent|notable|reference",
  "importance_score": 0.0~1.0,
  "event_type": "EVENT_TYPES_PLACEHOLDER",
  "reasoning": "판단 근거 1~2줄"
}}""".replace("EVENT_TYPES_PLACEHOLDER", _EVENT_TYPES_STR)


class ClassificationAgent:
    """대표 기사 클러스터를 GPT-4o로 분류한다."""

    def classify(self, cluster_id: int, representative_id: int) -> dict[str, Any]:
        """중요도 분류 (긴급/주목/참고) + 이벤트 타입 태깅.

        Args:
            cluster_id: 클러스터 ID.
            representative_id: 대표 기사의 raw_articles ID.

        Returns:
            {importance, importance_score, event_type, reasoning}
        """
        start = time.time()
        articles = get_articles_by_ids([representative_id])
        if not articles:
            return _default_result()

        articles_text = _format_articles(articles)
        prompt = _CLASSIFY_PROMPT.replace("ARTICLES_TEXT_PLACEHOLDER", articles_text)

        try:
            response = _llm.invoke(prompt)
            result = _parse_json(response.content)
            result.setdefault("event_type", "tech")
            result.setdefault("importance", "reference")
            result.setdefault("importance_score", 0.3)

            # DB 업데이트
            update_classification(
                representative_id,
                result["importance"],
                result["importance_score"],
            )

            elapsed = int((time.time() - start) * 1000)
            log.info(
                "중요도 분류 완료 | cluster=%d importance=%s score=%.2f elapsed=%dms",
                cluster_id, result["importance"], result["importance_score"], elapsed,
            )
            return result

        except Exception as e:
            log.error("분류 실패 | cluster=%d error=%s", cluster_id, e)
            return _default_result()


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(
            f"[{i}] 제목: {a['title']}\n"
            f"    출처: {a['source_name']} (신뢰도: {a.get('credibility_score', 0):.2f})\n"
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
        "importance": "reference",
        "importance_score": 0.3,
        "event_type": "tech",
        "reasoning": "",
    }
