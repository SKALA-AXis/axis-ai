# 작성일: 2026-05-15
# 작성자: 최종민
# 변경이력:
#   2026-05-15 최종민 — InsightCascade 프로토타입(4-phase CoT) 작성, 출력 검증 헬퍼
"""InsightCascadeAgent — 4-phase Cause→Change→Impact→Response + Synthesis CoT.

design: ``axis-ai/design/30-analysis/insight-cascade.md``.

본 모듈은 분석 4 agent 의 **prototype** (Walking Skeleton Phase 2). cold-start
fallback 만 활성 — Phase K3 의 ContextPackBuilder 가 도입되면
``_context_packs`` 가 자동 주입.

핵심 entry point:

    ``InsightCascadeAgent().generate(card_ids)`` — 카드 N개 → 4-phase CoT 분석.

흐름:
    1. ``_build_context`` — DB 의 card_news 조회 + 컨텍스트 텍스트 조립
    2. ``_llm_call`` — gpt-4o 단일 호출 (temperature 0.3, JSON 응답)
    3. ``_parse_and_validate`` — 3-tier observability 필드 + 4-phase 필드 검증
    4. ``link_langfuse_trace`` — Tier 3 trace_id 자동 매핑
    5. ``@with_ledger_writeback`` — analysis_ledger INSERT (자동, 데코레이터)
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.db.postgres import SessionLocal
from src.llm import LLMSpec, build_chat_llm
from src.middleware.analysis_ledger import with_ledger_writeback
from src.observability.langfuse_client import tracing_config
from src.services.agent_output_validation import (
    cap_reasoning_steps,
    cap_reasoning_trail,
    clip_final_one_liner,
    clip_implication,
    confidence_in_range,
    dedup_and_cap,
)

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o-mini"
_PROMPT_VERSION = "insight-v1.0"
_MAX_CARDS = int(os.getenv("INSIGHT_MAX_CARDS", "10"))
_MIN_CARDS = 2

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = build_chat_llm(
            LLMSpec(model=_LLM_MODEL, temperature=0.3, max_tokens=3000, json_object=True)
        )
    return _llm


# ──────────────────────────────────────────────────────────────────────────
# Prompt — design/30-analysis/insight-cascade.md §6.2 의 markdown heading 양식.
# ──────────────────────────────────────────────────────────────────────────

_INSIGHT_PROMPT = """\
# SK AX 인텔리전스 분석가

당신은 SK AX 사업전략팀의 인텔리전스 분석가입니다.
본 task 는 **단순 답 생성이 아닌 *추론 과정의 명시적 노출*** — 사용자가 어떻게 결론에
도달했는지 UI 가 단계별로 보여줍니다.

## 입력 데이터

### 카드 뉴스 (분석 대상)
{context}

### Peer Context Packs (cold-start 모드 — 미적용)
*cold start — pack 없음, recent_cards 만 사용*

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **카드 ID 근거**: 모든 bullet 에 `[CN-...]` (또는 `[IC-...]`) 카드 ID 로 출처 인용
- **환각 금지**: 출처에 없는 수치/이름 추가 시 즉시 `[자체 추정]` prefix
- **정량 보강**: 정성 표현 뒤에 정량 수치 (예: `"급성장 (QoQ +18.4%)"`)

### 일반 규칙 (17 요소 매핑)
1. **(#1 역할)** SK AX 사업전략팀 분석가 관점만
2. **(#2 추적 대상)** 카드 안의 4 국내 + 6 글로벌 + SK AX 자체에 한정
3. **(#7 단순 요약 금지)** event_type / 변화 / 시사점 패턴
4. **(#10 수익화 관점)** Impact bullet 에 `긍정/중립/부정` 명시
5. **(#11 정량 우선)** Change bullet 에 수치 / 날짜 / 제품명 우선
6. **(#13 SK AX 화자)** Impact + Response 의 "SK AX 의 ___ 에 영향" pattern
7. **(#15 우선순위)** Response 3 액션 중 가장 영향 큰 1개를 `priority=1`
8. **(#16 리스크)** `risk_assumptions` 에 1~3개
9. **(#17 반복 추적)** `follow_up_questions` 2~3개

## 추론 단계 (Chain of Thought)

### Phase 1 — Cause
- **자기 질문**: `"이 카드들이 발생한 배경 / 시장 환경 / Peer 의 전략적 motivation 은?"`
- **출력**: cause bullet 3~5

### Phase 2 — Change
- **자기 질문**: `"Peer 가 실제로 어떤 행동/투자/제품을 했는가? (사실 위주)"`
- **출력**: change bullet 3~5 (수치 / 날짜 / 제품명 우선)

### Phase 3 — Impact
- **자기 질문**: `"SK AX 의 사업·고객·경쟁 환경에 어떤 영향?"`
- **출력**: impact bullet 3~5 (각각 prefix `긍정:` / `중립:` / `부정:`)

### Phase 4 — Response
- **자기 질문**: `"SK AX 가 취할 구체적 액션? 3 중 최고 1개?"`
- **출력**: response 3개 (`priority` 1/2/3 부여)

### Phase 5 — Synthesis (final)
- **자기 질문**: `"위 4단계 종합 → 한 줄 결론 + 본 분석이 틀릴 가정?"`
- **출력**: `final_one_liner` + `risk_assumptions`

## 3-Tier Observability 출력

### Tier 1 — reasoning_trail (사용자 default)
정확히 **4~5 step** 으로 압축. label 권장: `"배경 진단"` / `"Peer 행동"` /
`"SK AX 영향"` / `"권장 대응"` / `"결론"`.
각 step: `seq` + `label` (≤ 12자) + `one_liner` (≤ 80자) + `evidence_refs` (card_id 목록).
탐색/시도/hedging 금지.

### Tier 2 — reasoning_steps (상세)
5~8 step. phase=`cause` / `change` / `impact` / `response` / `synthesis` 각 1+ step.

### Tier 3 — langfuse_trace_id
`null` 로 출력. 런타임 미들웨어가 자동 매핑.

## 출력 형식 (strict JSON)

```json
{{
  "cause": ["[CN-...] ...", "..."],
  "change": ["[공식 DART] [CN-...] ...", "..."],
  "impact": ["긍정: [CN-...] ...", "중립: ...", "부정: ..."],
  "response": [
    {{"action": "...", "priority": 1, "rationale": "[CN-...]"}},
    {{"action": "...", "priority": 2, "rationale": "..."}},
    {{"action": "...", "priority": 3, "rationale": "..."}}
  ],
  "final_one_liner": "≤ 100자, SK AX 관점, 모호 X",
  "sk_ax_implication": "1~2 문장. 긍정/중립/부정 명시.",
  "reasoning_trail": [
    {{
      "seq": 1, "label": "배경 진단",
      "one_liner": "...", "evidence_refs": ["CN-..."],
      "langfuse_observation_id": null
    }}
  ],
  "reasoning_steps": [
    {{
      "step_idx": 0, "phase": "cause",
      "question": "...", "inputs_used": ["CN-..."],
      "answer": "...", "intermediate_conclusion": "...",
      "confidence": 0.0, "langfuse_observation_id": null
    }}
  ],
  "follow_up_questions": ["...", "...", "..."],
  "risk_assumptions": ["본 인사이트가 ___ 가정에 의존. 그 가정이 틀리면 ___"],
  "confidence": 0.0,
  "sources_used": ["CN-..."]
}}
```

JSON 만 출력. 다른 텍스트 추가 금지.
"""


class InsightCascadeAgent:
    """4-phase + Synthesis CoT 인사이트 분석 agent."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    @with_ledger_writeback("InsightCascadeAgent")
    async def generate(self, card_ids: list[str], context: dict | None = None) -> dict:
        """N 카드 → 4-phase 인사이트.

        Args:
            card_ids: 분석할 카드 id 목록 (2 ≤ N ≤ 10 권장).
            context: 옵션 — frontend 가 전달하는 추가 컨텍스트 (현재 미사용).

        Returns:
            InsightCascadeOutput dict — design §5 schema.
        """
        if not card_ids or len(card_ids) < _MIN_CARDS:
            return _error_response(
                "card_ids 부족",
                f"insight 는 최소 {_MIN_CARDS}개 카드 필요 (받음={len(card_ids or [])})",
                card_ids or [],
            )

        if len(card_ids) > _MAX_CARDS:
            log.warning(
                "InsightCascade | card_ids 너무 많음 — 상위 %d개로 truncate (받음=%d)",
                _MAX_CARDS,
                len(card_ids),
            )
            card_ids = card_ids[:_MAX_CARDS]

        cards = _fetch_cards(card_ids)
        if not cards:
            return _error_response(
                "카드 조회 실패",
                "DB 에서 card_news row 0건 — id 확인 필요",
                card_ids,
            )

        prompt = _INSIGHT_PROMPT.replace("{context}", _format_cards(cards))

        try:
            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="InsightCascadeAgent",
                    phase="generate",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
        except Exception as e:
            log.exception("InsightCascadeAgent LLM 호출 실패 | error=%s", e)
            return _error_response("LLM 호출 실패", str(e), card_ids)

        result = _parse_and_validate(content, cards, card_ids)
        result.setdefault("provenance", {}).update(
            {
                "llm_model": _LLM_MODEL,
                "prompt_version": _PROMPT_VERSION,
                "source_card_ids": [c["id"] for c in cards],
            }
        )
        return result


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _fetch_cards(card_ids: list[str]) -> list[dict]:
    placeholders = ",".join(f":id_{i}" for i in range(len(card_ids)))
    params = {f"id_{i}": cid for i, cid in enumerate(card_ids)}
    sql = (
        "SELECT id, company AS peer_id, title, summary_lines, event_type, importance, "
        "importance_score, implication "
        f"FROM card_news WHERE id IN ({placeholders})"
    )
    try:
        with SessionLocal() as db:
            rows = db.execute(text(sql), params).mappings().all()
    except Exception as e:
        log.exception("InsightCascade DB query 실패 | %s", e)
        return []

    out: list[dict] = []
    for r in rows:
        item = dict(r)
        impl = item.get("implication")
        if isinstance(impl, str):
            try:
                impl = json.loads(impl)
            except json.JSONDecodeError:
                impl = {}
        item["implication"] = impl if isinstance(impl, dict) else {}
        out.append(item)
    return out


def _format_cards(cards: list[dict]) -> str:
    blocks: list[str] = []
    for c in cards:
        impl = c.get("implication") or {}
        sector = impl.get("sector") or "other"
        exposure_band = impl.get("exposure_band") or c.get("importance") or "low"
        summary_lines = c.get("summary_lines") or []
        if isinstance(summary_lines, str):
            try:
                summary_lines = json.loads(summary_lines)
            except json.JSONDecodeError:
                summary_lines = [summary_lines]
        summary = " / ".join(s for s in summary_lines if s)
        why_important = impl.get("why_important") or ""
        block = (
            f"[{c['id']}] {c.get('title', '')}\n"
            f"- Peer: {c.get('peer_id', '')}\n"
            f"- Sector: {sector}\n"
            f"- Event type: {c.get('event_type', '')}\n"
            f"- Exposure: {exposure_band} ({c.get('importance_score', 0.0):.2f})\n"
            f"- 요약: {summary}\n"
        )
        if why_important:
            block += f"- 시사점: {why_important}\n"
        blocks.append(block)
    return "\n".join(blocks)


def _parse_and_validate(
    content: str,
    cards: list[dict],
    card_ids: list[str],
) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # response_format=json_object 가 강제하지만 안전망
        log.warning("InsightCascade JSON parse 실패 — content prefix=%s", content[:200])
        return _error_response(
            "JSON parse 실패",
            "LLM 응답이 JSON 이 아님",
            card_ids,
            confidence=0.0,
        )

    if not isinstance(data, dict):
        return _error_response("응답 형식 오류", "JSON object 가 아님", card_ids, confidence=0.0)

    # 필수 필드 default 보충 — design schema 그대로
    data.setdefault("cause", [])
    data.setdefault("change", [])
    data.setdefault("impact", [])
    data.setdefault("response", [])
    data.setdefault("follow_up_questions", [])
    data.setdefault("risk_assumptions", [])

    # design 제약 강제 (medium-priority validation gap 보정)
    data["final_one_liner"] = clip_final_one_liner(data.get("final_one_liner", ""))
    data["sk_ax_implication"] = clip_implication(data.get("sk_ax_implication", ""))
    data["reasoning_trail"] = cap_reasoning_trail(data.get("reasoning_trail", []))
    data["reasoning_steps"] = cap_reasoning_steps(data.get("reasoning_steps", []))
    data["confidence"] = confidence_in_range(data.get("confidence", 0.0))
    data["sources_used"] = dedup_and_cap(data.get("sources_used") or [c["id"] for c in cards])

    # peer_ids — input cards 의 unique peer (analysis_ledger 의 carry-over 용)
    peer_set: list[str] = []
    seen: set[str] = set()
    for c in cards:
        pid = c.get("peer_id")
        if pid and pid not in seen:
            seen.add(pid)
            peer_set.append(pid)
    data["peer_ids"] = peer_set

    # 3-tier observability Tier 3 — langfuse trace id 매핑
    data["langfuse_trace_id"] = _get_langfuse_trace_id()
    data["warning"] = _warning_for(data)
    return data


def _get_langfuse_trace_id() -> str | None:
    try:
        from src.observability.langfuse_client import get_langfuse_handler

        handler = get_langfuse_handler()
        if handler is None:
            return None
        # v3+ CallbackHandler 의 last_trace_id (langfuse_client.py 의 expose 패턴)
        return getattr(handler, "last_trace_id", None)
    except Exception:
        return None


def _warning_for(data: dict) -> str | None:
    confidence = float(data.get("confidence") or 0.0)
    if confidence < 0.6:
        return "근거 불충분 — 다른 카드 조합 권장 (confidence < 0.6)"
    if not data.get("cause") or not data.get("impact"):
        return "필수 필드 (cause / impact) 누락 — 카드 본문 분석이 부족할 수 있음"
    return None


def _error_response(
    short_reason: str,
    detail: str,
    card_ids: list[str],
    confidence: float = 0.0,
) -> dict:
    log.warning("InsightCascade error | %s | detail=%s | ids=%s", short_reason, detail, card_ids)
    return {
        "cause": [],
        "change": [],
        "impact": [],
        "response": [],
        "final_one_liner": "",
        "sk_ax_implication": "",
        "reasoning_trail": [],
        "reasoning_steps": [],
        "follow_up_questions": [],
        "risk_assumptions": [],
        "confidence": confidence,
        "sources_used": card_ids,
        "peer_ids": [],
        "langfuse_trace_id": None,
        "warning": f"{short_reason} — {detail}",
        "provenance": {
            "llm_model": _LLM_MODEL,
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": card_ids,
            "error": short_reason,
        },
    }


class InsightAgent(InsightCascadeAgent):
    """Architecture-facing name for the 2단계 insight agent."""
