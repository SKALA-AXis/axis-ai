"""ChatOrchestratorAgent — Supervisor pattern (intent + multi-agent + compose + trace).

design: ``axis-ai/design/40-user-query/chat-orchestrator.md`` + PDF 2026-05-14 §5
("에이전트 간 협업 UI 노출").

prototype 범위 (Walking Skeleton Phase 2):
- Intent Router (gpt-4o-mini) — 8 intent enum 중 1개 + entity 추출
- Sub-agent 라우팅:
  - **summary** intent → Insight + GlobalTrends + PeerCompare(top peer) **3 agent 병렬**
    + Synthesizer compose (PDF §5 multi-agent collaboration 직접 대응)
  - **insight / mixer / peer_compare / global_trends / link_verify** → 단일 agent
  - **smalltalk** → 직접 LLM 응답
- **agent_trace** — orchestration 의 모든 step 을 timeline 으로 emit. 각 step:
  step_idx / agent / phase / status / duration_ms / input/output summary / model /
  parent_step_idx. frontend 가 timeline UI 로 렌더.
- Compose (gpt-4o-mini) — sub-agent 결과 → 대화체 reply
- typed follow_up_suggestions (5 lens)

deferred (Day 90+):
- deep_dive / alternative_view
- search intent (HybridSearchAgent 미구현)
- history 요약 압축
- session 영구 저장 (chat_sessions 테이블)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from langchain_openai import ChatOpenAI

from src.observability.langfuse_client import tracing_config

log = logging.getLogger(__name__)

_INTENT_MODEL = "gpt-4o-mini"
_COMPOSE_MODEL = "gpt-4o-mini"
_PROMPT_VERSION = "chat-orch-v2.0-supervisor"

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


# ──────────────────────────────────────────────────────────────────────────
# TraceBuilder — agent_trace step 누적 헬퍼
# ──────────────────────────────────────────────────────────────────────────


class TraceBuilder:
    """Supervisor pattern timeline 누적. 모든 phase 의 시작/종료를 step 으로 기록."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def step(
        self,
        agent: str,
        phase: str,
        input_summary: str = "",
        output_summary: str = "",
        model: str | None = None,
        status: str = "completed",
        duration_ms: int | None = None,
        parent_step_idx: int | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> int:
        step_idx = len(self.steps)
        self.steps.append(
            {
                "step_idx": step_idx,
                "agent": agent,
                "phase": phase,
                "status": status,
                "started_at": started_at or _now_iso(),
                "ended_at": ended_at,
                "duration_ms": duration_ms,
                "input_summary": input_summary[:120],
                "output_summary": output_summary[:160],
                "model": model,
                "parent_step_idx": parent_step_idx,
            }
        )
        return step_idx


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ──────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────


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
| **summary** | "오늘 핵심 변화", "전체 동향 요약" — multi-agent fan-out |
| search | "최근 동향 알려줘" |
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


_SUMMARY_SYNTH_PROMPT = """\
# SK AX 사업전략팀 도우미 — 멀티 agent 결과 종합

3 agent (InsightCascade / GlobalTrends / PeerComparison) 의 결과를 종합하여
**오늘의 핵심 변화 한국어 종합 응답** 을 작성합니다.

## 입력
- 사용자 메시지: {message}
- Insight 결과: {insight_json}
- GlobalTrends 결과: {global_json}
- PeerCompare 결과: {peer_json}

## 작성 규칙
- 자연스러운 대화체 (존댓말)
- 3 agent 결과를 비교/대조하며 "이 부분은 InsightCascade 가, 저 부분은 PeerCompare 가
  발견했어요" 같이 출처 agent 명시 (PDF §5 — 협업 가시화)
- 핵심 결론 먼저, 근거 카드 인용 `[CN-...]` 유지
- 응답 길이 1500 자 이내

응답만 출력. JSON / 메타설명 금지.
"""


# ──────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────


class ChatOrchestratorAgent:
    """Supervisor — intent + multi-agent + compose + trace emission."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict:
        session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        history = history or []
        trace = TraceBuilder()

        # ── Phase 1: Intent classification
        intent_result = await self._classify(message, history, trace)
        intent = intent_result.get("intent", "smalltalk")
        entities = intent_result.get("entities") or {}

        # ── Phase 2: Sub-agent dispatch (single or multi)
        sub_result = await self._dispatch(intent, message, entities, trace)

        # ── Phase 3: Compose
        reply = await self._compose(message, intent, sub_result, trace)

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
            "agent_trace": trace.steps,
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
                "total_steps": len(trace.steps),
            },
            "warning": _warning_for(intent, sub_result),
        }

    async def _classify(self, message: str, history: list[dict], trace: TraceBuilder) -> dict:
        started_at = _now_iso()
        t0 = time.perf_counter()
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
            intent = data.get("intent", "smalltalk") if isinstance(data, dict) else "smalltalk"
            duration_ms = int((time.perf_counter() - t0) * 1000)
            trace.step(
                agent="IntentRouter",
                phase="classify",
                input_summary=f"message: {message[:60]}",
                output_summary=(
                    f"intent={intent}, entities={_summarize_entities(data.get('entities') or {})}"
                ),
                model=_INTENT_MODEL,
                duration_ms=duration_ms,
                started_at=started_at,
                ended_at=_now_iso(),
            )
            if isinstance(data, dict) and data.get("intent"):
                return data
        except Exception as e:
            log.warning("ChatOrch intent 분류 실패 — fallback smalltalk | %s", e)
            duration_ms = int((time.perf_counter() - t0) * 1000)
            trace.step(
                agent="IntentRouter",
                phase="classify",
                status="failed",
                input_summary=f"message: {message[:60]}",
                output_summary=f"실패 → smalltalk fallback ({e})",
                model=_INTENT_MODEL,
                duration_ms=duration_ms,
                started_at=started_at,
                ended_at=_now_iso(),
            )
        return {"intent": "smalltalk", "entities": {}, "confidence": 0.0}

    async def _dispatch(
        self, intent: str, message: str, entities: dict, trace: TraceBuilder
    ) -> dict:
        """intent → sub-agent 호출. summary 는 multi-agent fan-out, 나머지는 단일."""
        try:
            if intent == "summary":
                return await self._dispatch_summary_fanout(message, entities, trace)
            if intent == "insight":
                return await self._dispatch_insight(entities, trace)
            if intent == "mixer":
                return await self._dispatch_mixer(entities, trace)
            if intent == "peer_compare":
                return await self._dispatch_peer(entities, trace)
            if intent == "global_trends":
                return await self._dispatch_global(trace)
            if intent == "link_verify":
                return await self._dispatch_link(entities, trace)
        except Exception as e:
            log.exception("ChatOrch sub-agent 호출 실패 | intent=%s | %s", intent, e)
            trace.step(
                agent="ChatOrchestrator",
                phase="dispatch",
                status="failed",
                output_summary=f"sub-agent 호출 실패: {e}",
            )
            return {"warning": f"sub-agent 호출 실패 ({intent}): {e}"}

        # smalltalk / search — sub-agent 없이 compose 에서 직접 답변
        trace.step(
            agent="ChatOrchestrator",
            phase="skip_dispatch",
            output_summary=f"intent={intent} — sub-agent 스킵, compose 에서 직접 답변",
        )
        return {"intent_meta": intent}

    # ── Single-agent dispatch helpers ────────────────────────────────────

    async def _dispatch_insight(self, entities: dict, trace: TraceBuilder) -> dict:
        from src.agents.insight_cascade_agent import InsightCascadeAgent

        card_ids = _ids_from_entities(entities, "card_ids") or _top_today_card_ids(6)
        if len(card_ids) < 2:
            trace.step(
                agent="ChatOrchestrator",
                phase="dispatch_insight",
                status="skipped",
                output_summary="card_ids 부족 (<2)",
            )
            return {"warning": "Insight 호출 위한 카드 부족 (2건 이상 필요)"}
        started_at = _now_iso()
        t0 = time.perf_counter()
        result = await InsightCascadeAgent().generate(card_ids=card_ids)
        trace.step(
            agent="InsightCascadeAgent",
            phase="generate",
            input_summary=f"card_ids={len(card_ids)}건",
            output_summary=_summarize_subresult(result),
            model="gpt-4o",
            duration_ms=int((time.perf_counter() - t0) * 1000),
            started_at=started_at,
            ended_at=_now_iso(),
        )
        return result

    async def _dispatch_mixer(self, entities: dict, trace: TraceBuilder) -> dict:
        from src.agents.mixer_analysis_agent import MixerAnalysisAgent

        card_ids = _ids_from_entities(entities, "card_ids") or _top_today_card_ids(3)
        if len(card_ids) < 2:
            trace.step(
                agent="ChatOrchestrator",
                phase="dispatch_mixer",
                status="skipped",
                output_summary="card_ids 부족 (<2)",
            )
            return {"warning": "Mixer 호출 위한 카드 부족 (2건 이상 필요)"}
        started_at = _now_iso()
        t0 = time.perf_counter()
        result = await MixerAnalysisAgent().analyze(card_ids=card_ids)
        trace.step(
            agent="MixerAnalysisAgent",
            phase="analyze",
            input_summary=f"card_ids={len(card_ids)}건",
            output_summary=_summarize_subresult(result),
            model="gpt-4o",
            duration_ms=int((time.perf_counter() - t0) * 1000),
            started_at=started_at,
            ended_at=_now_iso(),
        )
        return result

    async def _dispatch_peer(self, entities: dict, trace: TraceBuilder) -> dict:
        from src.agents.peer_comparison_agent import PeerComparisonAgent

        peer_ids = _ids_from_entities(entities, "peer_ids")
        if not peer_ids:
            trace.step(
                agent="ChatOrchestrator",
                phase="dispatch_peer",
                status="skipped",
                output_summary="peer_id 미식별",
            )
            return {"warning": "peer_id 미식별 — 메시지에 회사명 명시 필요"}
        started_at = _now_iso()
        t0 = time.perf_counter()
        result = await PeerComparisonAgent().compare(peer_id=peer_ids[0])
        trace.step(
            agent="PeerComparisonAgent",
            phase="compare",
            input_summary=f"peer={peer_ids[0]}",
            output_summary=_summarize_subresult(result),
            model="gpt-4o",
            duration_ms=int((time.perf_counter() - t0) * 1000),
            started_at=started_at,
            ended_at=_now_iso(),
        )
        return result

    async def _dispatch_global(self, trace: TraceBuilder) -> dict:
        from src.agents.global_trends_agent import GlobalTrendsAgent

        started_at = _now_iso()
        t0 = time.perf_counter()
        result = await GlobalTrendsAgent().run(window_days=30)
        trace.step(
            agent="GlobalTrendsAgent",
            phase="run",
            input_summary="window=30d, default 6사",
            output_summary=_summarize_subresult(result),
            model="gpt-4o" if not result.get("provenance", {}).get("cold_start") else None,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            started_at=started_at,
            ended_at=_now_iso(),
        )
        return result

    async def _dispatch_link(self, entities: dict, trace: TraceBuilder) -> dict:
        from src.agents.link_verification_agent import LinkVerificationAgent

        card_ids = _ids_from_entities(entities, "card_ids")
        if not card_ids:
            trace.step(
                agent="ChatOrchestrator",
                phase="dispatch_link",
                status="skipped",
                output_summary="card_id 미지정",
            )
            return {"warning": "card_id 미지정 — 메시지에 카드 ID 명시 필요"}
        started_at = _now_iso()
        t0 = time.perf_counter()
        result = await LinkVerificationAgent().verify(card_id=card_ids[0])
        trace.step(
            agent="LinkVerificationAgent",
            phase="verify",
            input_summary=f"card={card_ids[0]}",
            output_summary=(
                f"overall={result.get('overall_status')}, sources={len(result.get('sources', []))}"
            ),
            model=None,  # LLM X
            duration_ms=int((time.perf_counter() - t0) * 1000),
            started_at=started_at,
            ended_at=_now_iso(),
        )
        return result

    # ── Multi-agent fan-out (summary intent) ──────────────────────────────

    async def _dispatch_summary_fanout(
        self, message: str, entities: dict, trace: TraceBuilder
    ) -> dict:
        """summary intent → Insight + GlobalTrends + PeerCompare 병렬 → Synthesizer compose.

        PDF §5 직접 대응: agent 간 협업 가시화. 3 agent 가 parent_step_idx 공유로
        timeline 상 병렬 fan-out 으로 렌더링됨.
        """
        from src.agents.global_trends_agent import GlobalTrendsAgent
        from src.agents.insight_cascade_agent import InsightCascadeAgent
        from src.agents.peer_comparison_agent import PeerComparisonAgent

        # Parent step: fan-out 시작 마커
        parent_idx = trace.step(
            agent="ChatOrchestrator",
            phase="fan_out_start",
            output_summary="3 agent 병렬 호출 시작 (Insight + Global + Peer)",
        )

        card_ids = _ids_from_entities(entities, "card_ids") or _top_today_card_ids(6)
        peer_ids = _ids_from_entities(entities, "peer_ids") or _top_active_peer()

        # 병렬 호출 — asyncio.gather
        started_iso = _now_iso()
        t0 = time.perf_counter()

        async def _run_insight() -> dict:
            if len(card_ids) < 2:
                return {"warning": "Insight 스킵 — 카드 부족"}
            return await InsightCascadeAgent().generate(card_ids=card_ids)

        async def _run_global() -> dict:
            return await GlobalTrendsAgent().run(window_days=30)

        async def _run_peer() -> dict:
            if not peer_ids:
                return {"warning": "Peer 스킵 — peer_id 미식별"}
            return await PeerComparisonAgent().compare(peer_id=peer_ids[0])

        gather_results: list[Any] = await asyncio.gather(
            _run_insight(), _run_global(), _run_peer(), return_exceptions=True
        )
        insight_res, global_res, peer_res = gather_results[0], gather_results[1], gather_results[2]

        # 각 결과를 trace 에 기록 (parent_step_idx=parent_idx 공유 → frontend 병렬 표시)
        ended_iso = _now_iso()
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        insight_dict = _exception_to_dict(insight_res)
        global_dict = _exception_to_dict(global_res)
        peer_dict = _exception_to_dict(peer_res)

        trace.step(
            agent="InsightCascadeAgent",
            phase="generate",
            parent_step_idx=parent_idx,
            input_summary=f"card_ids={len(card_ids)}건",
            output_summary=_summarize_subresult(insight_dict),
            model="gpt-4o",
            duration_ms=elapsed_ms,
            started_at=started_iso,
            ended_at=ended_iso,
            status="failed" if isinstance(insight_res, Exception) else "completed",
        )
        trace.step(
            agent="GlobalTrendsAgent",
            phase="run",
            parent_step_idx=parent_idx,
            input_summary="window=30d, 6사 default",
            output_summary=_summarize_subresult(global_dict),
            model="gpt-4o" if not global_dict.get("provenance", {}).get("cold_start") else None,
            duration_ms=elapsed_ms,
            started_at=started_iso,
            ended_at=ended_iso,
            status="failed" if isinstance(global_res, Exception) else "completed",
        )
        trace.step(
            agent="PeerComparisonAgent",
            phase="compare",
            parent_step_idx=parent_idx,
            input_summary=f"peer={(peer_ids or ['?'])[0]}",
            output_summary=_summarize_subresult(peer_dict),
            model="gpt-4o",
            duration_ms=elapsed_ms,
            started_at=started_iso,
            ended_at=ended_iso,
            status="failed" if isinstance(peer_res, Exception) else "completed",
        )

        # Synthesizer 호출
        synth_started = _now_iso()
        s0 = time.perf_counter()
        synth_reply = await self._summary_synthesize(message, insight_dict, global_dict, peer_dict)
        synth_duration = int((time.perf_counter() - s0) * 1000)
        trace.step(
            agent="ChatOrchestrator",
            phase="synthesize",
            input_summary="3 agent 결과 종합",
            output_summary=f"reply {len(synth_reply)}자",
            model=_COMPOSE_MODEL,
            duration_ms=synth_duration,
            started_at=synth_started,
            ended_at=_now_iso(),
        )

        # synth_reply 를 sub_result.final_one_liner 에 담아서 compose 에서 활용
        # 단, compose 는 synth_reply 를 그대로 통과시키도록 sub_result 에 special field 추가
        return {
            "_synth_reply": synth_reply,
            "intent_meta": "summary",
            "sources_used": list(
                {
                    *(insight_dict.get("sources_used") or []),
                    *(peer_dict.get("sources_used") or []),
                }
            )[:10],
            "final_one_liner": insight_dict.get("final_one_liner")
            or peer_dict.get("final_one_liner"),
            "sk_ax_implication": (
                insight_dict.get("sk_ax_implication") or peer_dict.get("sk_ax_implication")
            ),
            "confidence": _avg_confidence(insight_dict, global_dict, peer_dict),
        }

    async def _summary_synthesize(
        self, message: str, insight: dict, global_: dict, peer: dict
    ) -> str:
        prompt = (
            _SUMMARY_SYNTH_PROMPT.replace("{message}", message)
            .replace("{insight_json}", _compact_json(insight))
            .replace("{global_json}", _compact_json(global_))
            .replace("{peer_json}", _compact_json(peer))
        )
        try:
            resp = _get_compose_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="ChatOrchestratorAgent",
                    phase="summary_synth",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            return content.strip()
        except Exception as e:
            log.exception("summary synth 실패 | %s", e)
            return "3 agent 결과 종합 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    # ── Compose ───────────────────────────────────────────────────────────

    async def _compose(
        self, message: str, intent: str, sub_result: dict | None, trace: TraceBuilder
    ) -> str:
        # summary intent 는 이미 synth_reply 가 있음 — compose 스킵
        if sub_result and sub_result.get("_synth_reply"):
            return str(sub_result["_synth_reply"])

        if intent == "smalltalk":
            started_at = _now_iso()
            t0 = time.perf_counter()
            reply = await self._smalltalk(message)
            trace.step(
                agent="ChatOrchestrator",
                phase="smalltalk_compose",
                input_summary=f"message: {message[:60]}",
                output_summary=f"reply {len(reply)}자",
                model=_COMPOSE_MODEL,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                started_at=started_at,
                ended_at=_now_iso(),
            )
            return reply

        sub_json = _compact_json(sub_result or {})
        prompt = (
            _COMPOSE_PROMPT.replace("{message}", message)
            .replace("{intent}", intent)
            .replace("{sub_result_json}", sub_json)
        )
        started_at = _now_iso()
        t0 = time.perf_counter()
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
            reply = content.strip()
        except Exception as e:
            log.exception("ChatOrch compose 실패 | %s", e)
            reply = f"응답 생성 중 오류가 발생했습니다 ({e})."
        trace.step(
            agent="ChatOrchestrator",
            phase="compose",
            input_summary=f"intent={intent}, sub_result {len(sub_json)}자",
            output_summary=f"reply {len(reply)}자",
            model=_COMPOSE_MODEL,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            started_at=started_at,
            ended_at=_now_iso(),
        )
        return reply

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


def _top_active_peer() -> list[str]:
    """최근 7일 카드 수 가장 많은 peer 1개."""
    from sqlalchemy import text

    from src.db.postgres import SessionLocal

    sql = (
        "SELECT company, COUNT(*) AS cnt FROM card_news "
        "WHERE created_at >= NOW() - INTERVAL '7 days' "
        "GROUP BY company ORDER BY cnt DESC LIMIT 1"
    )
    try:
        with SessionLocal() as db:
            row = db.execute(text(sql)).mappings().first()
        return [row["company"]] if row else []
    except Exception as e:
        log.warning("top_active_peer 조회 실패 | %s", e)
        return []


def _normalize_entities(entities: dict) -> dict:
    return {
        "peer_ids": _ids_from_entities(entities, "peer_ids"),
        "sectors": _ids_from_entities(entities, "sectors"),
        "card_ids": _ids_from_entities(entities, "card_ids"),
        "keywords": _ids_from_entities(entities, "keywords"),
        "date_range": entities.get("date_range") if isinstance(entities, dict) else None,
    }


def _summarize_entities(entities: dict) -> str:
    parts = []
    for key in ("peer_ids", "card_ids", "sectors"):
        val = _ids_from_entities(entities, key)
        if val:
            parts.append(f"{key}={val[:3]}")
    return ", ".join(parts) if parts else "(없음)"


def _summarize_subresult(result: dict | None) -> str:
    if not result:
        return "(empty)"
    if result.get("warning"):
        return f"warning: {result['warning'][:80]}"
    parts = []
    if result.get("final_one_liner"):
        parts.append(f"결론: {str(result['final_one_liner'])[:60]}")
    if result.get("strategy_label"):
        parts.append(f"label={result['strategy_label']}")
    if isinstance(result.get("snapshots"), list):
        parts.append(f"snapshots={len(result['snapshots'])}")
    if isinstance(result.get("sources"), list):
        parts.append(f"sources={len(result['sources'])}")
    if isinstance(result.get("overall_status"), str):
        parts.append(f"overall={result['overall_status']}")
    if result.get("confidence") is not None:
        parts.append(f"conf={float(result['confidence']):.2f}")
    return " | ".join(parts) if parts else "(no summary)"


def _extract_sources(sub_result: dict | None) -> list[dict]:
    if not sub_result:
        return []
    sources: list[dict] = []
    for cid in (sub_result.get("sources_used") or [])[:10]:
        sources.append({"card_id": cid})
    return sources


def _generate_follow_ups(intent: str, sub_result: dict | None) -> list[dict]:
    if intent not in ("insight", "mixer", "peer_compare", "global_trends", "summary"):
        return []
    suggestions: list[dict] = []
    topic = _topic_anchor(sub_result)
    for lens in _LENSES:
        suggestions.append(
            {
                "label": f"{_lens_korean(lens)} 관점에서 깊이 분석",
                "intent": "insight",
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
    if intent == "search":
        return "search intent 는 prototype 미지원 — Day 90+ 후속 작업"
    return None


def _compact_json(obj: dict) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)[:6000]
    except Exception:
        return "{}"


def _exception_to_dict(result: Any) -> dict:
    if isinstance(result, Exception):
        return {"warning": f"실패: {result}"}
    if isinstance(result, dict):
        return result
    return {}


def _avg_confidence(*results: dict) -> float:
    confidences = [float(r.get("confidence") or 0.0) for r in results if isinstance(r, dict)]
    valid = [c for c in confidences if c > 0]
    return sum(valid) / len(valid) if valid else 0.0
