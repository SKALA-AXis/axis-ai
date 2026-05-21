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
from src.services.agent_output_validation import (
    cap_reasoning_steps,
    cap_reasoning_trail,
    clip_final_one_liner,
    clip_implication,
    clip_string,
    confidence_in_range,
    dedup_and_cap,
)

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "mixer-v2.0-cross-card-discovery"
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
# SK AX 멀티 카드 *cross-card 발견* 전문가

당신은 SK AX 사업전략팀의 분석가입니다. 본 task 는 **per-card 요약이 아니라
*N 개의 카드를 함께 봐야만 보이는* 비명백한 패턴 발견**입니다.

**reject 기준** (다시 작성 강제):
- bullet_signals 3개가 각각 단일 카드 요약 → REJECT
- final_one_liner 가 "Peer X 의 AI 투자는 SK AX 에 긍정적" 식 일반론 → REJECT
- connections 의 label 이 전부 "reinforce" → REJECT (최소 3종 label 사용)
- 한 회사 카드끼리만 connection → REJECT (cross-peer pair 최소 1개)
- reasoning_trail step 3개 미만 → REJECT (정확히 4~5 step)
- 정량 수치 0개 → REJECT (final_one_liner 에 최소 1개)

## 입력 데이터

### 분석 카드 N건
{context}

### 결정적 산식 결과 — 6축 radar (LLM 입력 anchor)
{radar_text}

### 사용자 분석 비율
{ratios_text}

### 사용자 컨텍스트 (자유 입력)
{user_context}

### Peer Context Packs (cold-start — 미적용)
*cold start — pack 없음, 카드 본문만 사용*

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효 — 재생성)
- **출처 prefix**: 정량 수치 앞에 `[DART]` / `[기사 인용]` / `[자체 추정]` 강제
- **화자 고정**: "Peer X 의 ___ 는 SK AX 의 ___ 에 ___ 영향" pattern
- **환각 금지**: 카드 본문에 없는 수치/이름 추가 시 `[자체 추정]` 명시

### Cross-card 발견 강제 규칙
- **bullet_signals 의 각 항목은 카드 ID 최소 2개 cross-reference 필수**
  (예: `"[CN-002] + [CN-041] 의 조합은 ___ 패턴을 시사"`)
- **connections 의 label 다양성**: cause / effect / similar / contrast / reinforce 중
  **최소 3종 이상 사용**. cross-peer pair 최소 1개 (다른 회사 카드끼리)
- **cross_card_findings 필드 신규** — `"이 카드들을 함께 보지 않으면 보이지 않는 것"`
  3개 정확히 (`finding` + `evidence_card_ids` 최소 2개 + `pattern_type`)

### 일반 규칙
1. SK AX 사업전략팀 관점만
2. sk_ax_implication 에 "SK AX 매출/마진 영향 = 긍정/중립/부정" + 구체적 action
3. final_one_liner: 최소 1개 정량 수치 + 1개 회사명 + 1개 동사형 action
4. follow_up_questions 2~3개

## 좋은 예 vs 나쁜 예 (cross-card 발견)

### ❌ BAD (단순 요약 — REJECT)
```
"bullet_signals": [
  "삼성SDS 의 10조원 AI 투자",
  "LG CNS 의 오픈AI 대시보드 도입",
  "현대오토에버 매출 12% 증가"
]
"final_one_liner": "Peer 의 AI 투자는 SK AX 에 긍정적 영향"
```

### ✅ GOOD (cross-card discovery)
```
"bullet_signals": [
  "[CN-002, CN-041] 삼성SDS 10조 + KKR M&A → vertical integration 시도",
  "[CN-022, CN-040] LG CNS 오픈AI 대시보드 + 클라우드 → *운영 효율* 노선",
  "[CN-002 vs CN-022] 같은 AX 영역 *수직 (삼성) vs 수평 (LG)* 패러다임 분화"
]
"final_one_liner": "한국 IT 서비스 vertical/horizontal 양분 — SK AX 는 LG 모델 학습"
"cross_card_findings": [
  {{"finding": "삼성/LG 동시 AI 베팅 — 수직 통합 vs 운영 효율로 *대조*",
    "evidence_card_ids": ["CN-002", "CN-022"],
    "pattern_type": "divergent_strategy"}},
  {{"finding": "현대오토에버 매출 증가는 비-AX 부문 호조 [DART]",
    "evidence_card_ids": ["CN-049"],
    "pattern_type": "market_baseline"}}
]
```

## 추론 단계 (CoT)

### Phase 1 — per_card (각 카드 1 step) — 최소 요약만
짧게 1 step / card. event_type + 핵심 정량 수치만.

### Phase 2 — cross_card (의미 있는 pair, 최소 3 step)
- `inputs_used = [CN-A, CN-B]` 2 카드 짝
- 라벨 분포: cause / effect / similar / contrast / reinforce 중 **최소 3종 사용**
- cross-peer pair 최소 1개 (서로 다른 회사 카드)
- intermediate_conclusion: 왜 이 두 카드가 함께 봐야 의미 있는지 (≤ 100자)

### Phase 3 — synthesis (마지막 1 step)
- 모든 cross_card 결과를 한 줄 결론으로
- final_one_liner 와 일치

## 3-Tier Observability 출력

### Tier 1 — reasoning_trail (사용자 default)
**정확히 4~5 step**. 권장 label sequence:
1. `"카드 종합"` — N건 카드 한 줄 개요
2. `"교차 패턴"` — cross-card 발견 핵심
3. `"라벨 분포"` — connections label 다양성 검증
4. `"SK AX 대응"` — 구체적 action
5. `"결론"` — final_one_liner

각 step: `label` (≤ 12자) + `one_liner` (≤ 80자, **정량 수치 1개 필수**)
+ `evidence_refs` (card_id 최소 2개).

### Tier 2 — reasoning_steps (상세)
5~8 step. Phase 1~3 별 1+ step.

### Tier 3 — langfuse_trace_id
`null` 출력. 런타임 미들웨어 매핑.

## 출력 형식 (strict JSON)

```json
{{
  "insight": "1 문장 종합 (cross-card 발견 중심)",
  "final_one_liner": "≤ 100자. 정량 수치 + 회사명 + action verb 필수.",
  "sk_ax_implication": "1~2 문장. 영향 (긍정/중립/부정) + SK AX 가 취할 구체 action.",
  "bullet_signals": [
    "[CN-A, CN-B] 카드 조합이 시사하는 패턴 (단일 카드 요약 X)",
    "[CN-C, CN-D, CN-E] 또 다른 cross-card 패턴",
    "...3개"
  ],
  "cross_card_findings": [
    {{"finding": "이 카드들을 함께 봐야 보이는 비명백한 발견 (≤ 100자)",
      "evidence_card_ids": ["CN-...", "CN-..."],
      "pattern_type": "divergent_strategy"}}
  ],
  // pattern_type 가능값: convergent_strategy / divergent_strategy / gap_in_market
  //                    / acceleration_signal / timing_mismatch / market_baseline
  "connections": [
    {{"source_card_id": "CN-...", "target_card_id": "CN-...",
      "label": "cause | effect | similar | contrast | reinforce",
      "weight": 0.0,
      "reason": "왜 이 연결이 non-obvious 인지 (≤ 60자)"}}
  ],
  "reasoning_trail": [
    {{"seq": 1, "label": "카드 종합", "one_liner": "정량 수치 1개 포함",
      "evidence_refs": ["CN-...", "CN-..."], "langfuse_observation_id": null}}
  ],
  "reasoning_steps": [
    {{"step_idx": 0, "phase": "per_card", "question": "...", "inputs_used": ["CN-..."],
      "answer": "...", "intermediate_conclusion": "...", "confidence": 0.0,
      "langfuse_observation_id": null}}
  ],
  "follow_up_questions": ["...", "...", "..."],
  "confidence": 0.0,
  "sources_used": ["CN-..."]
}}
```

JSON 만 출력.
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

        # 6축 radar 미리 계산 — LLM input 으로 anchor 제공 (v2)
        radar = _compute_radar(cards)
        prompt = (
            _MIXER_PROMPT.replace("{context}", _format_cards(cards))
            .replace("{radar_text}", _format_radar(radar))
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
        result["radar_axes"] = radar  # 이미 위에서 계산된 값 재사용
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


def _format_radar(radar: list[dict]) -> str:
    """6축 radar 결과를 LLM 입력 prompt 용 한국어 라인으로 포맷.

    v2: 결정적 산식 결과를 LLM 에 anchor 로 제공. LLM 이 reasoning 에 활용.
    """
    if not radar:
        return "*radar 점수 산출 불가*"
    lines: list[str] = []
    label_kr = {
        "peer_strategic_shift": "Peer 전략 전환",
        "tech_investment": "기술 투자",
        "market_position": "시장 포지션",
        "partnership_momentum": "파트너십",
        "regulatory_risk": "규제 리스크",
        "talent_movement": "인재 이동",
    }
    for axis in radar:
        score = float(axis.get("score") or 0.0)
        bar = "▰" * int(score * 10) + "▱" * (10 - int(score * 10))
        label = label_kr.get(axis["axis"], axis["axis"])
        explanation = axis.get("explanation", "")
        lines.append(f"- {label}: {bar} {score:.2f} ({explanation})")
    return "\n".join(lines)


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

    data.setdefault("bullet_signals", [])
    data.setdefault("connections", [])
    data.setdefault("cross_card_findings", [])  # v2
    data.setdefault("follow_up_questions", [])
    data.setdefault("radar_axes", [])

    # design 제약 강제
    data["insight"] = clip_string(data.get("insight", ""), 200)
    data["final_one_liner"] = clip_final_one_liner(data.get("final_one_liner", ""))
    data["sk_ax_implication"] = clip_implication(data.get("sk_ax_implication", ""))
    data["reasoning_trail"] = cap_reasoning_trail(data.get("reasoning_trail", []))
    data["reasoning_steps"] = cap_reasoning_steps(data.get("reasoning_steps", []))
    data["confidence"] = confidence_in_range(data.get("confidence", 0.0))
    data["sources_used"] = dedup_and_cap(data.get("sources_used") or [c["id"] for c in cards])

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


class MixerAgent(MixerAnalysisAgent):
    """Architecture-facing name for the 2단계 mixer agent."""
