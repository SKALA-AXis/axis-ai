"""ChatOrchestratorAgent — intent 분류 + 분석 agent 라우팅 + compose.

design: ``axis-ai/design/40-user-query/chat-orchestrator.md``.

prototype 범위 (Walking Skeleton Phase 2):
- Intent Router (gpt-4o-mini) — 8 intent enum 중 1개로 분류 + entity 추출
- Sub-agent 위임: insight / mixer / peer_compare / global_trends / link_verify
- Compose (gpt-4o-mini) — sub-agent 결과 → 대화체 reply
- typed follow_up_suggestions (5 lens — technical/financial/competitive/regulatory/customer)

deferred (Day 90+):
- deep_dive / alternative_view (parent_sub carry-over)
- search / summary (HybridSearchAgent / AnswerAgent 미구현)
- history 요약 압축 (>10 turn)
- session 영구 저장
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from langchain_openai import ChatOpenAI

from src.observability.langfuse_client import tracing_config

log = logging.getLogger(__name__)

_INTENT_MODEL = "gpt-4o-mini"
_COMPOSE_MODEL = "gpt-4o-mini"
_PROMPT_VERSION = "chat-orch-v1.0"

_LENSES: tuple[str, ...] = ("technical", "financial", "competitive", "regulatory", "customer")

_intent_llm: ChatOpenAI | None = None
_compose_llm: ChatOpenAI | None = None


def _get_intent_llm() -> ChatOpenAI:
    global _intent_llm
    if _intent_llm is None:
        _intent_llm = ChatOpenAI(
            model=_INTENT_MODEL,
            temperature=0.0,
            max_completion_tokens=500,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _intent_llm


def _get_compose_llm() -> ChatOpenAI:
    global _compose_llm
    if _compose_llm is None:
        _compose_llm = ChatOpenAI(
            model=_COMPOSE_MODEL,
            temperature=0.4,
            max_completion_tokens=1500,
        )
    return _compose_llm


_INTENT_PROMPT = """\
# SK AX 사업전략팀 도우미 — Intent Router

당신은 SK AX 사업전략팀의 대화형 도우미입니다.
사용자 메시지의 **intent** 와 **entity** 를 분류합니다.

## 입력
- 메시지: {message}
- 이전 대화 (최근 3 turn): {history_short}

## 작성 규칙
- intent 는 enum (insight / mixer / peer_compare / global_trends / link_verify /
  search / summary / smalltalk) 중 1개만
- entity 의 peer_ids 는 4 국내 (samsung_sds / lg_cns / hyundai_autoever / posco_dx)
  또는 6 글로벌 (nvidia / apple / microsoft / google / amazon / meta) 중 메시지/이전
  대화에 등장한 값만
- sectors 는 enum (ax / security / infra / deal / other) 중 메시지에 등장한 값만
- 메시지에 없는 값을 추가하지 마시오

## intent 가이드
| intent | 예시 |
|---|---|
| insight | "이 카드들로 인사이트", "왜 중요한지 분석" |
| mixer | "카드 N개 묶어서", "이 카드 조합 분석" |
| peer_compare | "삼성SDS vs SK AX", "LG CNS 전략" |
| global_trends | "글로벌 동향", "NVIDIA / Microsoft 영향" |
| link_verify | "출처 확인", "이 카드 링크 유효한지" |
| search | "최근 동향 알려줘" |
| summary | "오늘 핵심 변화" |
| smalltalk | 일반 대화 |

## 출력 형식 (strict JSON)
```json
{{
  "intent": "insight",
  "entities": {{
    "peer_ids": [],
    "sectors": [],
    "card_ids": [],
    "keywords": [],
    "date_range": null
  }},
  "confidence": 0.0
}}
```

JSON 만 출력. 다른 텍스트 추가 금지.
"""


_COMPOSE_PROMPT = """\
# SK AX 사업전략팀 도우미 — 대화 응답 작성

당신은 SK AX 사업전략팀의 대화형 도우미입니다.
sub-agent 가 분석한 결과를 받아서 **대화체 한국어 응답** 을 작성합니다.

## 입력
- 사용자 메시지: {message}
- 분류된 intent: {intent}
- sub-agent 결과 (JSON): {sub_result_json}

## 작성 규칙
- 자연스러운 대화체 한국어 (존댓말, 2~5 문장)
- 핵심 결론을 먼저, 근거는 뒤에 한 줄
- 카드 인용은 `[CN-...]` 또는 `[IC-...]` 형식 그대로 유지
- 정량 수치는 sub_result 에 있는 것만 인용 (환각 금지)
- intent 가 smalltalk 이면 부드럽게 대화
- 응답은 1500 자 이내

응답만 출력. JSON / 코드블록 / 메타설명 추가 금지.
"""


class ChatOrchestratorAgent:
    """intent 분류 + sub-agent 라우팅 + compose."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict:
        """user message → intent → sub-agent → compose → reply.

        Args:
            message: 사용자 메시지.
            session_id: 세션 식별 (frontend localStorage). None 이면 신규 생성.
            history: 최근 turn 들 [{role, content}].

        Returns:
            ChatTurnOutput dict — design §5 schema.
        """
        session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        history = history or []

        intent_result = await self._classify(message, history)
        intent = intent_result.get("intent", "smalltalk")
        entities = intent_result.get("entities") or {}

        sub_result = await self._dispatch(intent, message, entities)
        reply = await self._compose(message, intent, sub_result)
        follow_ups = _generate_follow_ups(intent, sub_result)

        confidence = float(intent_result.get("confidence") or 0.0)
        sub_confidence = float(sub_result.get("confidence") or 0.0) if sub_result else 0.0

        return {
            "reply": reply,
            "intent": intent,
            "entities": _normalize_entities(entities),
            "sources": _extract_sources(sub_result),
            "follow_up_suggestions": follow_ups,
            "final_one_liner": (sub_result or {}).get("final_one_liner"),
            "sk_ax_implication": (sub_result or {}).get("sk_ax_implication"),
            "deep_dive_depth": 1,
            "reasoning_steps": (sub_result or {}).get("reasoning_steps"),
            "confidence": min(c for c in (confidence, sub_confidence) if c > 0)
            if (confidence > 0 or sub_confidence > 0)
            else 0.0,
            "session_id": session_id,
            "provenance": {
                "intent_model": _INTENT_MODEL,
                "compose_model": _COMPOSE_MODEL,
                "prompt_version": _PROMPT_VERSION,
                "intent": intent,
                "entities": entities,
            },
            "warning": _warning_for(intent, sub_result),
        }

    async def _classify(self, message: str, history: list[dict]) -> dict:
        prompt = _INTENT_PROMPT.replace("{message}", message).replace(
            "{history_short}", _format_history(history)
        )
        try:
            resp = _get_intent_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="ChatOrchestratorAgent",
                    phase="intent",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = json.loads(content)
            if isinstance(data, dict) and data.get("intent"):
                return data
        except (json.JSONDecodeError, Exception) as e:
            log.warning("ChatOrch intent 분류 실패 — fallback smalltalk | %s", e)
        return {"intent": "smalltalk", "entities": {}, "confidence": 0.0}

    async def _dispatch(self, intent: str, message: str, entities: dict) -> dict:
        try:
            if intent == "insight":
                return await self._dispatch_insight(entities)
            if intent == "mixer":
                return await self._dispatch_mixer(entities)
            if intent == "peer_compare":
                return await self._dispatch_peer(entities)
            if intent == "global_trends":
                return await self._dispatch_global()
            if intent == "link_verify":
                return await self._dispatch_link(entities)
        except Exception as e:
            log.exception("ChatOrch sub-agent 호출 실패 | intent=%s | %s", intent, e)
            return {"warning": f"sub-agent 호출 실패 ({intent}): {e}"}

        # smalltalk / search / summary — sub-agent 없이 compose 에서 직접 답변
        return {"intent_meta": intent}

    async def _dispatch_insight(self, entities: dict) -> dict:
        from src.agents.insight_cascade_agent import InsightCascadeAgent

        card_ids = _ids_from_entities(entities, "card_ids") or _top_today_card_ids(6)
        if len(card_ids) < 2:
            return {"warning": "Insight 호출 위한 카드 부족 (2건 이상 필요)"}
        return await InsightCascadeAgent().generate(card_ids=card_ids)

    async def _dispatch_mixer(self, entities: dict) -> dict:
        from src.agents.mixer_analysis_agent import MixerAnalysisAgent

        card_ids = _ids_from_entities(entities, "card_ids") or _top_today_card_ids(3)
        if len(card_ids) < 2:
            return {"warning": "Mixer 호출 위한 카드 부족 (2건 이상 필요)"}
        return await MixerAnalysisAgent().analyze(card_ids=card_ids)

    async def _dispatch_peer(self, entities: dict) -> dict:
        from src.agents.peer_comparison_agent import PeerComparisonAgent

        peer_ids = _ids_from_entities(entities, "peer_ids")
        if not peer_ids:
            return {"warning": "peer_id 미식별 — 메시지에 회사명 명시 필요"}
        return await PeerComparisonAgent().compare(peer_id=peer_ids[0])

    async def _dispatch_global(self) -> dict:
        from src.agents.global_trends_agent import GlobalTrendsAgent

        return await GlobalTrendsAgent().run(window_days=30)

    async def _dispatch_link(self, entities: dict) -> dict:
        from src.agents.link_verification_agent import LinkVerificationAgent

        card_ids = _ids_from_entities(entities, "card_ids")
        if not card_ids:
            return {"warning": "card_id 미지정 — 메시지에 카드 ID 명시 필요"}
        return await LinkVerificationAgent().verify(card_id=card_ids[0])

    async def _compose(self, message: str, intent: str, sub_result: dict | None) -> str:
        if intent == "smalltalk":
            return await self._smalltalk(message)
        sub_json = json.dumps(sub_result or {}, ensure_ascii=False, default=str)[:6000]
        prompt = (
            _COMPOSE_PROMPT.replace("{message}", message)
            .replace("{intent}", intent)
            .replace("{sub_result_json}", sub_json)
        )
        try:
            resp = _get_compose_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="ChatOrchestratorAgent",
                    phase="compose",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            return content.strip()
        except Exception as e:
            log.exception("ChatOrch compose 실패 | %s", e)
            return f"응답 생성 중 오류가 발생했습니다 ({e})."

    async def _smalltalk(self, message: str) -> str:
        prompt = (
            "당신은 SK AX 사업전략팀의 대화형 도우미입니다. 친절하고 간결하게 응답하세요.\n\n"
            f"사용자: {message}\n도우미:"
        )
        try:
            resp = _get_compose_llm().invoke(prompt)
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            return content.strip()
        except Exception as e:
            log.exception("smalltalk 실패 | %s", e)
            return "지금은 응답을 생성할 수 없어요. 잠시 후 다시 시도해 주세요."


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(없음)"
    last = history[-3:]
    lines = []
    for turn in last:
        role = turn.get("role", "user")
        content = str(turn.get("content", ""))[:200]
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


def _ids_from_entities(entities: dict, key: str) -> list[str]:
    raw = entities.get(key) if isinstance(entities, dict) else None
    if isinstance(raw, list):
        return [str(x) for x in raw if x and isinstance(x, str)]
    return []


def _top_today_card_ids(limit: int) -> list[str]:
    """오늘의 상위 카드 N개 id 반환 (entities.card_ids 비어 있을 때 fallback)."""
    from sqlalchemy import text

    from src.db.postgres import SessionLocal

    sql = (
        "SELECT id FROM card_news WHERE created_at >= NOW() - INTERVAL '7 days' "
        "ORDER BY importance_score DESC NULLS LAST LIMIT :limit"
    )
    try:
        with SessionLocal() as db:
            rows = db.execute(text(sql), {"limit": limit}).mappings().all()
        return [r["id"] for r in rows]
    except Exception as e:
        log.warning("top_today_card_ids 조회 실패 | %s", e)
        return []


def _normalize_entities(entities: dict) -> dict:
    return {
        "peer_ids": _ids_from_entities(entities, "peer_ids"),
        "sectors": _ids_from_entities(entities, "sectors"),
        "card_ids": _ids_from_entities(entities, "card_ids"),
        "keywords": _ids_from_entities(entities, "keywords"),
        "date_range": entities.get("date_range") if isinstance(entities, dict) else None,
    }


def _extract_sources(sub_result: dict | None) -> list[dict]:
    if not sub_result:
        return []
    sources: list[dict] = []
    for cid in (sub_result.get("sources_used") or [])[:10]:
        sources.append({"card_id": cid})
    return sources


def _generate_follow_ups(intent: str, sub_result: dict | None) -> list[dict]:
    """5 lens 버튼 + sub-agent follow_up_questions 추가 (PDF §6 typed follow-up)."""
    if intent not in ("insight", "mixer", "peer_compare", "global_trends"):
        return []
    suggestions: list[dict] = []
    topic = _topic_anchor(sub_result)
    for lens in _LENSES:
        suggestions.append(
            {
                "label": f"{_lens_korean(lens)} 관점에서 깊이 분석",
                "intent": "insight",  # prototype 은 deep_dive 미구현, insight 재호출로 대체
                "deep_dive": True,
                "topic_anchor": topic,
                "lens": lens,
            }
        )
    for q in (sub_result or {}).get("follow_up_questions", [])[:3]:
        if not isinstance(q, str) or not q.strip():
            continue
        suggestions.append(
            {
                "label": q[:30],
                "intent": "insight",
                "deep_dive": True,
                "topic_anchor": q,
                "lens": None,
            }
        )
    return suggestions[:6]


def _topic_anchor(sub_result: dict | None) -> str | None:
    if not sub_result:
        return None
    one_liner = sub_result.get("final_one_liner")
    if isinstance(one_liner, str) and one_liner.strip():
        return one_liner.strip()[:80]
    return None


def _lens_korean(lens: str) -> str:
    return {
        "technical": "기술",
        "financial": "재무",
        "competitive": "경쟁",
        "regulatory": "규제",
        "customer": "고객",
    }.get(lens, lens)


def _warning_for(intent: str, sub_result: dict | None) -> str | None:
    if sub_result and sub_result.get("warning"):
        return str(sub_result["warning"])
    if intent in ("search", "summary"):
        return "search / summary intent 는 prototype 미지원 — Day 90+ 후속 작업"
    return None
