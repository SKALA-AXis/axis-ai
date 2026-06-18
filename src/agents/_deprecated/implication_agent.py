# 작성일: 2026-04-27
# 작성자: 최종민
# 변경이력:
#   2026-04-27 최종민 — v3 전환(Evidence Chain)으로 시사점 생성 에이전트 추가 및 ruff 포맷 적용
#   2026-05-18 심유정 — 카드뉴스 에이전트 플로우 개선
#   2026-06-02 박지원 — 분석 파이프라인 재구성 및 뉴스 통합 흐름 반영
"""시사점 생성 에이전트 — GPT-4o, SK AX 관점 4개 항목."""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o", temperature=0.7, max_tokens=1500)

_IMPLICATION_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 이슈 카드를 바탕으로 SK AX 관점의 시사점 초안을 작성해주세요.

## 이슈 카드
제목: {title}
요약:
{summary}
이벤트 타입: {event_type}
중요도: {importance}

## 출처
{sources_text}

## SK AX 핵심 사업 맥락
- 에이전틱AI: 자율 AI 에이전트 기반 엔터프라이즈 솔루션
- 제조AX: 제조업 특화 AI 전환 (스마트팩토리, 공정최적화)
- MSP: 멀티클라우드 관리 서비스

## 작성 규칙
- why_important: 이 이슈가 SK AX에 중요한 이유 (출처 근거 명시 필수)
- potential_impact: SK AX 사업에 미칠 잠재적 영향
- follow_up_questions: 추가 모니터링 필요한 질문 3개
- suggested_actions: SK AX가 취할 수 있는 행동 제안 2~3개
- confidence: 시사점의 근거 충분도 (0.0~1.0)
- sources_used: 근거로 사용한 출처 인덱스 목록

주의: 출처에 없는 수치(금액·%·날짜)를 생성하지 마세요.
confidence < 0.6이면 '근거 불충분'을 명시하세요.

다음 JSON 형식으로만 응답하세요 (추가 텍스트 금지):
{{
  "why_important": "...",
  "potential_impact": "...",
  "follow_up_questions": ["...", "...", "..."],
  "suggested_actions": ["...", "...", "..."],
  "confidence": 0.0,
  "sources_used": [1]
}}"""


class ImplicationAgent:
    """이슈 카드 → GPT-4o → SK AX 시사점 초안 생성."""

    def generate(self, issue_card: dict[str, Any]) -> dict[str, Any]:
        """SK AX 관점의 시사점 초안을 생성한다.

        Args:
            issue_card: CardNewsComposer가 생성한 카드뉴스 dict.

        Returns:
            시사점 dict (why_important, potential_impact, follow_up_questions,
                         suggested_actions, confidence, sources_used).
        """
        if not issue_card:
            return _empty_result()

        sources_text = _format_sources(issue_card.get("sources", []))
        summary = "\n".join(issue_card.get("summary_lines", []))

        prompt = (
            _IMPLICATION_PROMPT.replace("{title}", issue_card.get("title", ""))
            .replace("{summary}", summary)
            .replace("{event_type}", issue_card.get("event_type", "tech"))
            .replace("{importance}", issue_card.get("importance", "reference"))
            .replace("{sources_text}", sources_text)
        )

        try:
            response = _llm.invoke(prompt)
            result = _parse_json(response.content)
            result.setdefault("confidence", 0.5)
            result.setdefault("sources_used", [1])
            result["evidence_label"] = _evidence_label(result["confidence"])

            log.info(
                "시사점 생성 완료 | card_id=%s confidence=%.2f label=%s",
                issue_card.get("id"),
                result["confidence"],
                result["evidence_label"],
            )
            return result

        except Exception as e:
            log.error("시사점 생성 실패 | card_id=%s error=%s", issue_card.get("id"), e)
            return _empty_result()


def _evidence_label(confidence: float) -> str:
    """confidence 임계값 → 근거 레이블 매핑 (CLAUDE.md 스펙)."""
    if confidence < 0.6:
        return "insufficient"
    if confidence < 0.8:
        return "moderate"
    return "sufficient"


def _format_sources(sources: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{s.get('index', i + 1)}] {s.get('title', '')} ({s.get('source_name', '')})"
        for i, s in enumerate(sources)
    )


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _empty_result() -> dict[str, Any]:
    return {
        "why_important": "",
        "potential_impact": "",
        "follow_up_questions": [],
        "suggested_actions": [],
        "confidence": 0.0,
        "sources_used": [],
        "evidence_label": "insufficient",
    }
