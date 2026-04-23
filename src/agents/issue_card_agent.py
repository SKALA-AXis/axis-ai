"""이슈 카드 생성 에이전트 — GPT-4o 호출."""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.db.article_store import _generate_card_id, get_articles_by_ids

log = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=1024)

_ISSUE_CARD_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 기사를 바탕으로 이슈 카드를 작성해주세요.

## 기사 정보
{articles_text}

## 작성 규칙
- 제목: 핵심 사실을 담은 한 문장 (40자 이내)
- 3줄 요약: 반드시 3줄, 각 줄은 "1.", "2.", "3."으로 시작
- 이벤트 타입: partnership/ma/personnel/tech/regulation/new_biz 중 하나
- 출처 목록: 사용한 기사의 title, source_name, url 포함

다음 JSON 형식으로만 응답하세요 (추가 텍스트 금지):
{{
  "title": "이슈 제목",
  "summary_lines": ["1. ...", "2. ...", "3. ..."],
  "event_type": "tech",
  "sources": [
    {{"index": 1, "title": "...", "source_name": "...", "url": "...", "credibility_score": 0.0}}
  ]
}}"""


class IssueCardAgent:
    """클러스터 대표 기사로 이슈 카드를 생성하고 DB에 저장한다."""

    def generate(
        self,
        cluster_id: int,
        representative_id: int,
        peer_id: str,
        classification: dict[str, Any],
    ) -> dict[str, Any]:
        """이슈 카드를 생성한다.

        Args:
            cluster_id: 클러스터 ID.
            representative_id: 대표 기사 ID.
            peer_id: 'samsung_sds' | 'lg_cns'.
            classification: ClassificationAgent 결과.

        Returns:
            IssueCard dict (저장 전 validation 없는 상태).
        """
        articles = get_articles_by_ids([representative_id])
        if not articles:
            return {}

        articles_text = _format_articles(articles)
        prompt = _ISSUE_CARD_PROMPT.replace("{articles_text}", articles_text)

        try:
            response = _llm.invoke(prompt)
            card_data = _parse_json(response.content)

            card = {
                "id": _generate_card_id(peer_id),
                "peer_id": peer_id,
                "cluster_id": cluster_id,
                "title": card_data.get("title", articles[0]["title"][:100]),
                "summary_lines": card_data.get("summary_lines", []),
                "event_type": card_data.get("event_type", classification.get("event_type", "tech")),
                "importance": classification.get("importance", "reference"),
                "importance_score": classification.get("importance_score", 0.3),
                "sources": card_data.get("sources", _default_sources(articles)),
            }

            log.info(
                "이슈카드 생성 완료 | id=%s importance=%s event=%s",
                card["id"], card["importance"], card["event_type"],
            )
            return card

        except Exception as e:
            log.error("이슈카드 생성 실패 | cluster=%d error=%s", cluster_id, e)
            return {}


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(
            f"[{i}] 제목: {a['title']}\n"
            f"    출처: {a['source_name']} | URL: {a['url']}\n"
            f"    내용: {(a.get('content') or '')[:800]}"
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
