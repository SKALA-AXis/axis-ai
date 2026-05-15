"""MixerAnalysisAgent — 3-phase per_card / cross_card / synthesis CoT.

design: ``axis-ai/design/30-analysis/mixer-analysis.md``.

본 모듈은 분석 4 agent 중 두 번째 **prototype** (Walking Skeleton Phase 2). 6축 radar
score 는 결정적 산식 (LLM X), reasoning 만 LLM 단일 호출. cold-start 모드 — Phase
K3 ContextPackBuilder 이후 ``_context_packs`` 가 자동 주입될 자리만 마련.

핵심 entry point:

    ``MixerAnalysisAgent().analyze(card_ids, ratios, user_context)`` — N 카드 → 분석 결과.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.db.postgres import SessionLocal
from src.middleware.analysis_ledger import with_ledger_writeback
from src.observability.langfuse_client import tracing_config

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "mixer-v1.0"
_MAX_CARDS = int(os.getenv("MIXER_MAX_CARDS", "20"))
_MIN_CARDS = 2

_RADAR_AXIS_ORDER: tuple[str, ...] = (
    "peer_strategic_shift",
    "tech_investment",
    "market_position",
    "partnership_momentum",
    "regulatory_risk",
    "talent_movement",
)

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.3,
            max_completion_tokens=3000,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm


# ──────────────────────────────────────────────────────────────────────────
# Prompt — design/30-analysis/mixer-analysis.md §6.2.
# ──────────────────────────────────────────────────────────────────────────

_MIXER_PROMPT = """\
# SK AX 멀티 카드 분석 전문가

당신은 SK AX 사업전략팀의 멀티 카드 분석 전문가입니다.
본 task 는 **단일 답 생성이 아닌 *추론 과정 자체의 명시적 노출*** — 분석가가 어떻게
결론에 도달했는지 UI 가 사용자에게 보여줍니다.

## 입력 데이터

### 분석 카드 N건
{context}

### 사용자 분석 비율
{ratios_text}

### 사용자 컨텍스트 (자유 입력)
{user_context}

### Peer Context Packs (cold-start — 미적용)
*cold start — pack 없음, 카드 본문만 사용*

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **출처 prefix**: 모든 정량 수치 앞에 `[DART]` / `[기사 인용]` / `[자체 추정]` 강제
- **화자 고정**: "Peer 가 X 했다" 만 금지 → "Peer X 는 SK AX 의 ___ 에 ___ 영향" pattern
- **환각 금지**: 카드 본문에 없는 수치/이름 추가 시 즉시 `[자체 추정]` 명시

### 일반 규칙 (17 요소 매핑)
1. **(#1 역할)** SK AX 사업전략팀 관점만
2. **(#7 단순 요약 금지)** "카드 N 개 요약" 금지 — event_type / 변화 / 시사점 패턴
3. **(#10 수익화 관점)** sk_ax_implication 에 "SK AX 매출/마진 영향 = 긍정/중립/부정"
4. **(#11 정량 우선)** "성장 추세" 금지 → `"QoQ +12.3%"` 류
5. **(#15 우선순위)** bullet_signals 3개 중 가장 영향 큰 1개 (`priority=1`)
6. **(#17 반복 추적)** follow_up_questions 2~3개

## 추론 단계 (Chain of Thought)

### Phase 1 — per_card (각 카드 1 step)
- **자기 질문**: `"이 카드 (CN-XXX) 의 핵심 신호는?"`
- **입력**: 카드 1건 (`inputs_used = ["CN-XXX"]`)
- **답변**: event_type + 핵심 사실 + 정량 수치 (출처 prefix 강제)
- **intermediate_conclusion**: `"카드 X = {{one line signal}}"`

### Phase 2 — cross_card (의미 있는 pair 별 1 step, 최대 8 step)
- **자기 질문**: `"CN-A 와 CN-B 는 어떻게 연결되는가?"`
- **입력**: 카드 2건 (`inputs_used = ["CN-A", "CN-B"]`)
- **답변**: 두 카드 비교 + 인과 / 유사 / 대조 / 강화 관계 추론
- **intermediate_conclusion**: `"{{label}}: {{evidence}}"`
- **출력 매핑**: 각 cross_card step → `connections[]` 항목 1:1 대응

### Phase 3 — synthesis (반드시 마지막 step)
- **자기 질문**: `"위 단계의 결론을 종합하면 SK AX 가 주목해야 하는 단일 주제는?"`
- **입력**: 모든 card_id
- **답변**: 종합 reasoning (≤ 500자)
- **intermediate_conclusion**: 최종 한 줄 결론 (= `final_one_liner` 와 일치 강제)

## 3-Tier Observability 출력

### Tier 1 — reasoning_trail (사용자 default)
정확히 **3~5 step** 으로 압축. label 권장: `"카드 비교"` / `"패턴 발견"` /
`"재무 검증"` / `"결론"` 류 ≤ 12자.

각 trail step: `seq` + `label` + `one_liner` (≤ 80자, 수치 권장) + `evidence_refs` (card_id).
탐색/시도/hedging 표현 금지.

### Tier 2 — reasoning_steps (상세)
5~8 step. Phase 1~3 별 1+ step. `phase` ∈ {{per_card, cross_card, synthesis}}.

### Tier 3 — langfuse_trace_id
`null` 로 출력. 런타임 미들웨어가 자동 매핑.

## connections[] 작성 규칙

- cross_card 단계의 step 마다 1:1 대응. `source_card_id` / `target_card_id` / `label`
  ∈ {{cause, effect, similar, contrast, reinforce}} / `weight` (0~1).
- 의미 없는 pair (단순 동일 peer 의 무관 카드) 는 생략.

## 출력 형식 (strict JSON)

```json
{{
  "insight": "1문장 종합",
  "final_one_liner": "≤ 100자, SK AX 관점, 모호 X",
  "sk_ax_implication": "1~2 문장. 긍정/중립/부정 명시.",
  "bullet_signals": ["...", "...", "..."],
  "connections": [
    {{"source_card_id": "CN-...", "target_card_id": "CN-...",
      "label": "cause", "weight": 0.7}}
  ],
  "reasoning_trail": [
    {{
      "seq": 1, "label": "카드 비교",
      "one_liner": "...", "evidence_refs": ["CN-..."],
      "langfuse_observation_id": null
    }}
  ],
  "reasoning_steps": [
    {{
      "step_idx": 0, "phase": "per_card",
      "question": "...", "inputs_used": ["CN-..."],
      "answer": "...", "intermediate_conclusion": "...",
      "confidence": 0.0, "langfuse_observation_id": null
    }}
  ],
  "follow_up_questions": ["...", "...", "..."],
  "confidence": 0.0,
  "sources_used": ["CN-..."]
}}
```

JSON 만 출력. 다른 텍스트 추가 금지.
"""


class MixerAnalysisAgent:
    """3-phase per_card / cross_card / synthesis CoT 카드 분석 agent."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    @with_ledger_writeback("MixerAnalysisAgent")
    async def analyze(
        self,
        card_ids: list[str],
        ratios: dict | None = None,
        user_context: str | None = None,
    ) -> dict:
        """N 카드 → 6축 radar + cross-card 분석.

        Args:
            card_ids: 분석할 카드 id (2 ≤ N ≤ 20 권장).
            ratios: peer / industry / keyword 가중치 (frontend slider 결과).
            user_context: 사용자 자유 입력.

        Returns:
            MixerAnalysisOutput dict — design §5 schema.
        """
        if not card_ids or len(card_ids) < _MIN_CARDS:
            return _error_response(
                "card_ids 부족",
                f"mixer 는 최소 {_MIN_CARDS}개 카드 필요 (받음={len(card_ids or [])})",
                card_ids or [],
            )

        if len(card_ids) > _MAX_CARDS:
            log.warning(
                "Mixer | card_ids 너무 많음 — 상위 %d개로 truncate (받음=%d)",
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

        prompt = (
            _MIXER_PROMPT.replace("{context}", _format_cards(cards))
            .replace("{ratios_text}", _format_ratios(ratios))
            .replace("{user_context}", (user_context or "").strip() or "*없음*")
        )

        try:
            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="MixerAnalysisAgent",
                    phase="analyze",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
        except Exception as e:
            log.exception("MixerAnalysisAgent LLM 호출 실패 | error=%s", e)
            return _error_response("LLM 호출 실패", str(e), card_ids)

        result = _parse_and_validate(content, cards, card_ids)
        result["radar_axes"] = _compute_radar(cards)
        result["mix_id"] = _new_mix_id()
        result.setdefault("provenance", {}).update(
            {
                "llm_model": _LLM_MODEL,
                "prompt_version": _PROMPT_VERSION,
                "source_card_ids": [c["id"] for c in cards],
                "ratios": ratios or {},
            }
        )
        return result


# ──────────────────────────────────────────────────────────────────────────
# Deterministic radar (no LLM) — design §6.1
# ──────────────────────────────────────────────────────────────────────────


def _avg(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def _score_for_event(cards: list[dict], event_types: set[str]) -> float:
    return _avg([_card_score(c) for c in cards if (c.get("event_type") or "") in event_types])


def _score_for_sector(cards: list[dict], sectors: set[str]) -> float:
    return _avg([_card_score(c) for c in cards if _card_sector(c) in sectors])


def _card_score(card: dict) -> float:
    impl = card.get("implication") or {}
    exposure = impl.get("exposure_score")
    if exposure is None:
        exposure = card.get("importance_score")
    try:
        return float(exposure or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _card_sector(card: dict) -> str:
    impl = card.get("implication") or {}
    return (impl.get("sector") or card.get("sector") or "other").lower()


def _peer_diversity_score(cards: list[dict]) -> float:
    peers = {c.get("peer_id") for c in cards if c.get("peer_id")}
    # 4 국내 peer 기준, 다양성 정규화 (1 peer=0.25, 4 peer=1.0).
    return min(len(peers) / 4.0, 1.0)


def _compute_radar(cards: list[dict]) -> list[dict]:
    axes: dict[str, tuple[float, str]] = {
        "peer_strategic_shift": (
            _score_for_event(cards, {"ma", "new_biz"}),
            "M&A / 신규사업 카드 평균 exposure",
        ),
        "tech_investment": (
            _score_for_sector(cards, {"ax", "ai_tech", "infra"}),
            "AX / 인프라 섹터 카드 평균 exposure",
        ),
        "market_position": (
            _peer_diversity_score(cards),
            "Peer 다양성 (unique peer / 4)",
        ),
        "partnership_momentum": (
            _score_for_event(cards, {"partnership"}),
            "파트너십 카드 평균 exposure",
        ),
        "regulatory_risk": (
            _score_for_event(cards, {"regulation"}),
            "규제 카드 평균 exposure",
        ),
        "talent_movement": (
            _score_for_event(cards, {"personnel"}),
            "인사 카드 평균 exposure",
        ),
    }
    result: list[dict] = []
    for axis in _RADAR_AXIS_ORDER:
        score, explanation = axes[axis]
        result.append({"axis": axis, "score": round(score, 3), "explanation": explanation})
    return result


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _new_mix_id() -> str:
    return f"mix-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _format_ratios(ratios: dict | None) -> str:
    if not ratios:
        return "*비율 미지정 — 기본 균등*"
    parts: list[str] = []
    for key in ("peer", "industry", "keyword"):
        value = ratios.get(key)
        if value:
            parts.append(f"- **{key}**: {value}")
    return "\n".join(parts) if parts else "*비율 미지정 — 기본 균등*"


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
        log.exception("Mixer DB query 실패 | %s", e)
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
        sector = _card_sector(c)
        exposure_band = impl.get("exposure_band") or c.get("importance") or "low"
        summary_lines = c.get("summary_lines") or []
        if isinstance(summary_lines, str):
            try:
                summary_lines = json.loads(summary_lines)
            except json.JSONDecodeError:
                summary_lines = [summary_lines]
        summary = " / ".join(s for s in summary_lines if s)
        block = (
            f"[{c['id']}] {c.get('title', '')}\n"
            f"- Peer: {c.get('peer_id', '')}\n"
            f"- Sector: {sector}\n"
            f"- Event type: {c.get('event_type', '')}\n"
            f"- Exposure: {exposure_band} ({_card_score(c):.2f})\n"
            f"- 요약: {summary}\n"
        )
        why = impl.get("why_important") or ""
        if why:
            block += f"- 시사점: {why}\n"
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
        log.warning("Mixer JSON parse 실패 — content prefix=%s", content[:200])
        return _error_response(
            "JSON parse 실패",
            "LLM 응답이 JSON 이 아님",
            card_ids,
            confidence=0.0,
        )

    if not isinstance(data, dict):
        return _error_response("응답 형식 오류", "JSON object 가 아님", card_ids, confidence=0.0)

    data.setdefault("insight", "")
    data.setdefault("final_one_liner", "")
    data.setdefault("sk_ax_implication", "")
    data.setdefault("bullet_signals", [])
    data.setdefault("connections", [])
    data.setdefault("reasoning_trail", [])
    data.setdefault("reasoning_steps", [])
    data.setdefault("follow_up_questions", [])
    data.setdefault("confidence", 0.0)
    data.setdefault("sources_used", [c["id"] for c in cards])
    data.setdefault("radar_axes", [])

    peer_set: list[str] = []
    seen: set[str] = set()
    for c in cards:
        pid = c.get("peer_id")
        if pid and pid not in seen:
            seen.add(pid)
            peer_set.append(pid)
    data["peer_ids"] = peer_set

    data["langfuse_trace_id"] = _get_langfuse_trace_id()
    data["warning"] = _warning_for(data)
    return data


def _get_langfuse_trace_id() -> str | None:
    try:
        from src.observability.langfuse_client import get_langfuse_handler

        handler = get_langfuse_handler()
        if handler is None:
            return None
        return getattr(handler, "last_trace_id", None)
    except Exception:
        return None


def _warning_for(data: dict) -> str | None:
    confidence = float(data.get("confidence") or 0.0)
    if confidence < 0.6:
        return "근거 불충분 — 다른 카드 조합 권장 (confidence < 0.6)"
    if not data.get("bullet_signals") or not data.get("reasoning_trail"):
        return "필수 필드 누락 — bullet_signals / reasoning_trail 비어 있음"
    return None


def _error_response(
    short_reason: str,
    detail: str,
    card_ids: list[str],
    confidence: float = 0.0,
) -> dict:
    log.warning("Mixer error | %s | detail=%s | ids=%s", short_reason, detail, card_ids)
    return {
        "mix_id": _new_mix_id(),
        "insight": "",
        "final_one_liner": "",
        "sk_ax_implication": "",
        "bullet_signals": [],
        "radar_axes": [],
        "connections": [],
        "reasoning_trail": [],
        "reasoning_steps": [],
        "follow_up_questions": [],
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
