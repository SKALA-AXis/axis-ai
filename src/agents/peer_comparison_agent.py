"""PeerComparisonAgent — Phase 1 (Current) + Phase 2 (Trend) + Phase 4 (Strategic).

design: ``axis-ai/design/30-analysis/peer-comparison.md``.

prototype 범위 (Walking Skeleton Phase 2): cold-start, ContextPack 미주입.
Phase 3 (Forecast) 는 Day 90+ deferred — `forecasts` 빈 배열 반환.

핵심 entry point:

    ``PeerComparisonAgent().compare(peer_id, window_days, focus_sector)``.

흐름:
    1. ``_fetch_peer_context`` — peer 의 최근 카드 + financial_history 조회
    2. ``_compute_trend_deltas`` — QoQ / YoY deterministic 계산 (LLM X)
    3. ``_llm_call`` — Phase 1 + 4 single gpt-4o call
    4. ``_parse_and_validate`` — 3-tier 필드 검증 + Phase 3 stub
    5. ``@with_ledger_writeback`` — 분석 read model 저장
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.agents._validation_helpers import (
    cap_reasoning_steps,
    cap_reasoning_trail,
    clip_final_one_liner,
    clip_implication,
    confidence_in_range,
    dedup_and_cap,
    normalize_strategy_label,
)
from src.db.postgres import SessionLocal
from src.middleware.analysis_ledger import with_ledger_writeback
from src.observability.langfuse_client import tracing_config

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "peer-compare-v1.0"
_DEFAULT_WINDOW_DAYS = 30
_MAX_CARDS = int(os.getenv("PEER_COMPARE_MAX_CARDS", "20"))

# PDF §9 변화 감지 임계값 (절대값 기준)
_BAND_NORMAL = 5.0
_BAND_NOTABLE = 10.0
_BAND_DRAMATIC = 30.0

_FIN_METRICS: tuple[tuple[str, str], ...] = (
    ("revenue_total_krwbn", "매출"),
    ("operating_profit_krwbn", "영업이익"),
    ("ai_revenue_share_pct", "AI 비중"),
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
# Prompt — design/30-analysis/peer-comparison.md §6.4.
# Phase 3 (Forecast) section 은 prototype 에서 제외 (Day 90+ deferred).
# ──────────────────────────────────────────────────────────────────────────

_PEER_COMPARE_PROMPT = """\
# SK AX 경쟁사 분석가

당신은 SK AX 사업전략팀의 경쟁사 분석가입니다.
**{peer_id} 의 현재 / 추세** 를 SK AX 관점에서 분석합니다.
추론 과정을 단계별로 명시적으로 노출합니다.

## 입력 데이터

### 분석 기간 (KST)
- **since**: {since}
- **until**: {until}
- **window_days**: {window_days}

### {peer_id} 데이터
- **최근 카드 (N건)**:
{peer_cards_summary}

- **재무 IR (분기)**:
{peer_ir_summary}

- **트렌드 deltas (산식 출력)**:
{trend_deltas_summary}

### SK AX 비교 베이스라인 (cold-start prototype)
*Phase K3 ContextPack 도입 전 — peer cards 분석에 집중. SK AX 자사 데이터는 LLM 추론 보강.*

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **출처 prefix**: 모든 정량 수치 앞에 `[공식 DART]` / `[기사 인용 CN-...]` / `[자체 추정]`
- **화자 고정**: 모든 결론이 `"SK AX 의 ___"` pattern (peer 의 행동을 SK AX 영향으로 환산)
- **환각 금지**: trend_deltas 는 산식 출력 — 임의 수치 추가 금지, 입력 carry only

### 일반 규칙 (17 요소 매핑)
1. **(#1 역할)** SK AX 경쟁사 분석가 관점만
2. **(#5 분석 기간)** analysis_period 절대 기준 (`since ~ until`)
3. **(#8 회사별 비교 기준)** 매출 / 영업이익 / AI 비중 / R&D 등
4. **(#9 변화 감지)** ±5% normal / >10% 유의 / >30% 급변 — band 명시
5. **(#10 수익화 관점)** sk_ax_implication 의 긍정/중립/부정
6. **(#11 정량 우선)** trend_deltas 수치 인용 + 정성 해석
7. **(#13 SK AX 화자)** 위 pattern
8. **(#17 반복 추적)** follow_up_questions 2~3개

## 추론 단계 (Chain of Thought)

### Phase 1 — Current state
- **자기 질문**: `"현재 {peer_id} 의 포지션과 SK AX 와의 차별점은?"`
- **입력**: peer_cards
- **출력**: `strategy_label` (5종 중 1) + `differentiators` + `strengths_of_peer` +
  `weaknesses_of_peer` + `collaboration_potential`

### Phase 2 — Trend interpretation
- **자기 질문**: `"trend_deltas 의 의미는? 어떤 사업 방향 전환?"`
- **입력**: trend_deltas + peer_ir_summary
- **출력**: 정량 (QoQ +X%) 후 정성 해석

### Phase 3 — Forecast (prototype: 생략)
*Day 90+ 활성 예정. 본 응답에서는 비활성.*

### Phase 4 — Strategic implication + final
- **자기 질문**: `"SK AX 는 어떤 행동? 우선순위 1개?"`
- **출력**: `sk_ax_implication` (1~2 문장, 긍정/중립/부정) + `final_one_liner` (≤ 100자)

## 전략 라벨 (5종 중 1)
`"Aggressive Expansion"` / `"Defensive Hold"` / `"Tech Pivot"` /
`"Customer Lock-in"` / `"Cost Leadership"`

## 3-Tier Observability 출력

### Tier 1 — reasoning_trail (사용자 default)
정확히 **3 step** (prototype, forecast 제외). 권장:
1. `"현재 포지션"`
2. `"추세 비교"` (one_liner 예시: `"매출 QoQ +12% / 영업이익률 -2pp [DART]"`)
3. `"SK AX 대응"`

각 step: `seq` + `label` (≤ 12자) + `one_liner` (≤ 80자) + `evidence_refs` (card_id / DART id).

### Tier 2 — reasoning_steps (상세)
phase ∈ {{current, trend, strategic}} 각 1+ step. 총 3~5 step.

### Tier 3 — langfuse_trace_id
`null` 로 출력. 미들웨어가 자동 매핑.

## 출력 형식 (strict JSON)

```json
{{
  "strategy_label": "Aggressive Expansion",
  "differentiators": [
    {{"aspect": "...", "peer_position": "...", "skax_position": "...", "opportunity": "..."}}
  ],
  "strengths_of_peer": ["..."],
  "weaknesses_of_peer": ["..."],
  "collaboration_potential": ["..."],
  "sk_ax_implication": "1~2 문장. 긍정/중립/부정 명시.",
  "final_one_liner": "≤ 100자, SK AX 관점",
  "follow_up_questions": ["...", "..."],
  "reasoning_trail": [
    {{
      "seq": 1, "label": "현재 포지션",
      "one_liner": "...", "evidence_refs": ["CN-..."],
      "langfuse_observation_id": null
    }}
  ],
  "reasoning_steps": [
    {{
      "step_idx": 0, "phase": "current",
      "question": "...", "inputs_used": ["CN-..."],
      "answer": "...", "intermediate_conclusion": "...",
      "confidence": 0.0, "langfuse_observation_id": null
    }}
  ],
  "confidence": 0.0,
  "sources_used": ["CN-...", "DART:..."]
}}
```

JSON 만 출력. 다른 텍스트 추가 금지.
"""


class PeerComparisonAgent:
    """Phase 1 + 2 + 4 prototype agent. Phase 3 Forecast 는 Day 90+ deferred."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    @with_ledger_writeback("PeerComparisonAgent")
    async def compare(
        self,
        peer_id: str,
        window_days: int = _DEFAULT_WINDOW_DAYS,
        focus_sector: str | None = None,
    ) -> dict:
        """단일 peer 분석 → 3-phase CoT (forecast 제외).

        Args:
            peer_id: 비교 대상 peer id.
            window_days: 카드 조회 윈도우 (기본 30일).
            focus_sector: 특정 sector 필터 (옵션).

        Returns:
            PeerComparisonOutput dict — design §5 schema.
        """
        if not peer_id or not peer_id.strip():
            return _error_response(peer_id, "peer_id 필수", "비어 있음", [])

        until = datetime.now(UTC)
        since = until - timedelta(days=window_days)

        cards = _fetch_peer_cards(peer_id, since, focus_sector)
        if not cards:
            log.warning("PeerCompare | %s 카드 0건 (window=%d) — LLM 스킵", peer_id, window_days)
            return _error_response(
                peer_id,
                "카드 조회 실패",
                f"{peer_id} 의 최근 {window_days}일 카드 0건",
                [],
            )

        financials = _fetch_financials(peer_id)
        trend_deltas = _compute_trend_deltas(financials)

        prompt = (
            _PEER_COMPARE_PROMPT.replace("{peer_id}", peer_id)
            .replace("{since}", since.date().isoformat())
            .replace("{until}", until.date().isoformat())
            .replace("{window_days}", str(window_days))
            .replace("{peer_cards_summary}", _format_cards(cards))
            .replace("{peer_ir_summary}", _format_financials(financials))
            .replace("{trend_deltas_summary}", _format_trend_deltas(trend_deltas))
        )

        try:
            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="PeerComparisonAgent",
                    phase="compare",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
        except Exception as e:
            log.exception("PeerComparisonAgent LLM 호출 실패 | error=%s", e)
            return _error_response(peer_id, "LLM 호출 실패", str(e), [c["id"] for c in cards])

        result = _parse_and_validate(content, cards, peer_id)
        result["trend_deltas"] = [delta for delta in trend_deltas]
        result["forecasts"] = []  # Day 90+ deferred
        result["analysis_period"] = {
            "since": since.date().isoformat(),
            "until": until.date().isoformat(),
            "window_days": window_days,
            "focus_sector": focus_sector,
        }
        result.setdefault("provenance", {}).update(
            {
                "llm_model": _LLM_MODEL,
                "prompt_version": _PROMPT_VERSION,
                "source_card_ids": [c["id"] for c in cards],
                "financial_rows": len(financials),
                "forecast_status": "deferred_day90",
            }
        )
        return result


# ──────────────────────────────────────────────────────────────────────────
# Trend delta — deterministic (no LLM). design §6.2.
# ──────────────────────────────────────────────────────────────────────────


def _compute_trend_deltas(financials: list[dict]) -> list[dict]:
    if not financials:
        return []
    sorted_rows = sorted(financials, key=lambda r: r.get("period") or "")
    latest = sorted_rows[-1]
    prev_q = sorted_rows[-2] if len(sorted_rows) >= 2 else None
    prev_y = sorted_rows[-5] if len(sorted_rows) >= 5 else None
    period_label = latest.get("period") or ""

    deltas: list[dict] = []
    for column, label in _FIN_METRICS:
        latest_value = _safe_float(latest.get(column))
        if latest_value is None:
            continue
        qoq = _percent_delta(latest_value, _safe_float(prev_q.get(column)) if prev_q else None)
        yoy = _percent_delta(latest_value, _safe_float(prev_y.get(column)) if prev_y else None)
        band = _band(qoq, yoy)
        direction = _direction(qoq if qoq is not None else yoy)
        deltas.append(
            {
                "metric": column,
                "label": label,
                "qoq_pct": qoq,
                "yoy_pct": yoy,
                "band": band,
                "direction": direction,
                "source": f"[공식 DART {period_label}]" if period_label else "[공식 DART]",
            }
        )
    return deltas


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent_delta(latest: float, previous: float | None) -> float | None:
    if previous is None or previous == 0:
        return None
    return round((latest - previous) / abs(previous) * 100.0, 2)


def _band(qoq: float | None, yoy: float | None) -> str:
    candidates = [v for v in (qoq, yoy) if v is not None]
    if not candidates:
        return "normal"
    magnitude = max(abs(v) for v in candidates)
    if magnitude >= _BAND_DRAMATIC:
        return "급변"
    if magnitude >= _BAND_NOTABLE:
        return "유의"
    if magnitude >= _BAND_NORMAL:
        return "normal"
    return "normal"


def _direction(value: float | None) -> str:
    if value is None or abs(value) < 0.5:
        return "flat"
    return "up" if value > 0 else "down"


# ──────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────


def _fetch_peer_cards(peer_id: str, since: datetime, focus_sector: str | None) -> list[dict]:
    sql = (
        "SELECT id, company AS peer_id, title, summary_lines, event_type, importance, "
        "importance_score, implication, created_at "
        "FROM card_news WHERE company = :peer_id AND created_at >= :since "
        "ORDER BY created_at DESC LIMIT :limit"
    )
    params = {"peer_id": peer_id, "since": since, "limit": _MAX_CARDS}
    try:
        with SessionLocal() as db:
            rows = db.execute(text(sql), params).mappings().all()
    except Exception as e:
        log.exception("PeerCompare card_news 조회 실패 | %s", e)
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
        if focus_sector:
            sector_value = (item["implication"].get("sector") or "").lower()
            if sector_value != focus_sector.lower():
                continue
        out.append(item)
    return out


def _fetch_financials(peer_id: str) -> list[dict]:
    sql = "SELECT financial_history FROM peer_companies WHERE id = :peer_id"
    try:
        with SessionLocal() as db:
            row = db.execute(text(sql), {"peer_id": peer_id}).mappings().first()
    except Exception as e:
        log.exception("PeerCompare peer_companies.financial_history 조회 실패 | %s", e)
        return []
    if not row:
        return []
    history = row.get("financial_history") or []
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except json.JSONDecodeError:
            return []
    if not isinstance(history, list):
        return []
    return [item for item in history if isinstance(item, dict)]


def _format_cards(cards: list[dict]) -> str:
    if not cards:
        return "*카드 없음*"
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
        block = (
            f"[{c['id']}] {c.get('title', '')}\n"
            f"- Sector: {sector} · Event: {c.get('event_type', '')} · "
            f"Exposure: {exposure_band} ({c.get('importance_score', 0.0):.2f})\n"
            f"- 요약: {summary}"
        )
        why = impl.get("why_important") or ""
        if why:
            block += f"\n- 시사점: {why}"
        blocks.append(block)
    return "\n".join(blocks)


def _format_financials(rows: list[dict]) -> str:
    if not rows:
        return "*financial_history 데이터 없음 (cold-start)*"
    lines: list[str] = []
    for r in rows[-5:]:  # 최근 5분기
        period = r.get("period") or ""
        revenue = r.get("revenue_total_krwbn")
        op = r.get("operating_profit_krwbn")
        ai_share = r.get("ai_revenue_share_pct")
        parts: list[str] = [f"[{period}]"]
        if revenue is not None:
            parts.append(f"매출 {revenue:.0f}억")
        if op is not None:
            parts.append(f"영업이익 {op:.0f}억")
        if ai_share is not None:
            parts.append(f"AI 비중 {ai_share:.1f}%")
        lines.append(" · ".join(parts))
    return "\n".join(lines)


def _format_trend_deltas(deltas: list[dict]) -> str:
    if not deltas:
        return "*financial_history 부재 — 정량 추세 산출 불가*"
    lines: list[str] = []
    for d in deltas:
        qoq = f"QoQ {d['qoq_pct']:+.1f}%" if d.get("qoq_pct") is not None else "QoQ N/A"
        yoy = f"YoY {d['yoy_pct']:+.1f}%" if d.get("yoy_pct") is not None else "YoY N/A"
        lines.append(
            f"- {d.get('label') or d['metric']}: {qoq} / {yoy} · band={d['band']} {d['source']}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Parse + validate
# ──────────────────────────────────────────────────────────────────────────


def _parse_and_validate(content: str, cards: list[dict], peer_id: str) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("PeerCompare JSON parse 실패 — prefix=%s", content[:200])
        return _error_response(
            peer_id, "JSON parse 실패", "LLM 응답이 JSON 이 아님", [c["id"] for c in cards]
        )

    if not isinstance(data, dict):
        return _error_response(
            peer_id, "응답 형식 오류", "JSON object 가 아님", [c["id"] for c in cards]
        )

    data.setdefault("differentiators", [])
    data.setdefault("strengths_of_peer", [])
    data.setdefault("weaknesses_of_peer", [])
    data.setdefault("collaboration_potential", [])
    data.setdefault("follow_up_questions", [])

    # design 제약 강제 — strategy_label 5종 enum + 길이 / step / confidence
    data["strategy_label"] = normalize_strategy_label(data.get("strategy_label", ""))
    data["final_one_liner"] = clip_final_one_liner(data.get("final_one_liner", ""))
    data["sk_ax_implication"] = clip_implication(data.get("sk_ax_implication", ""))
    data["reasoning_trail"] = cap_reasoning_trail(data.get("reasoning_trail", []))
    data["reasoning_steps"] = cap_reasoning_steps(data.get("reasoning_steps", []))
    data["confidence"] = confidence_in_range(data.get("confidence", 0.0))
    data["sources_used"] = dedup_and_cap(data.get("sources_used") or [c["id"] for c in cards])
    data["peer_id"] = peer_id
    data["peer_ids"] = [peer_id]

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
        return "근거 불충분 — peer 카드 부족 또는 분석 신뢰도 < 0.6"
    if not data.get("strategy_label") or not data.get("differentiators"):
        return "필수 필드 누락 — strategy_label / differentiators 비어 있음"
    return None


def _error_response(
    peer_id: str,
    short_reason: str,
    detail: str,
    card_ids: list[str],
) -> dict:
    log.warning("PeerCompare error | %s | %s | detail=%s", peer_id, short_reason, detail)
    return {
        "peer_id": peer_id,
        "peer_ids": [peer_id] if peer_id else [],
        "strategy_label": "",
        "differentiators": [],
        "strengths_of_peer": [],
        "weaknesses_of_peer": [],
        "collaboration_potential": [],
        "trend_deltas": [],
        "forecasts": [],
        "sk_ax_implication": "",
        "final_one_liner": "",
        "follow_up_questions": [],
        "reasoning_trail": [],
        "reasoning_steps": [],
        "langfuse_trace_id": None,
        "confidence": 0.0,
        "sources_used": card_ids,
        "analysis_period": {},
        "warning": f"{short_reason} — {detail}",
        "provenance": {
            "llm_model": _LLM_MODEL,
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": card_ids,
            "error": short_reason,
        },
    }
