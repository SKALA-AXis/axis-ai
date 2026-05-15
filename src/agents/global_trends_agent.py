"""GlobalTrendsAgent — Phase 1 (Snapshot) + Phase 2 (Trend Detection) +
Phase 3 (Impact Map) + Phase 4 (Forecast) + Phase 5 (Synthesis).

design: ``axis-ai/design/30-analysis/global-trends.md``.

prototype 범위 (Walking Skeleton Phase 2):
- Phase 1/2 결정적 산식 (LLM X) — global cards frequency + theme delta
- Phase 3/4/5 LLM 단일 호출 — impact_matrix + forecasts + synthesis
- cold-start: 글로벌 카드 부재 시 (현재 ingestion 4 Korean peer 만) graceful 빈 응답

핵심 entry point:

    ``GlobalTrendsAgent().run(company_ids, focus_themes, window_days, sk_ax_business_lines)``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import UTC, datetime, timedelta

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.agents._validation_helpers import (
    cap_reasoning_steps,
    cap_reasoning_trail,
    clip_final_one_liner,
    clip_implication,
    confidence_in_range,
    dedup_and_cap,
)
from src.db.postgres import SessionLocal
from src.middleware.analysis_ledger import with_ledger_writeback
from src.observability.langfuse_client import tracing_config

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "global-trends-v1.0"

_GLOBAL_COMPANIES: tuple[str, ...] = (
    "nvidia",
    "apple",
    "microsoft",
    "google",
    "amazon",
    "meta",
)

_SK_AX_LINES_DEFAULT: tuple[str, ...] = (
    "ai_managed",
    "cloud_msp",
    "security",
    "smart_factory",
    "data_platform",
)

_MAX_CARDS_PER_COMPANY = int(os.getenv("GLOBAL_TRENDS_MAX_CARDS_PER_COMPANY", "20"))
_MAX_HEADLINES = 3

# PDF §9 / checklist 9 — frequency delta intensity bands
_INTENSITY_WEAK = 20.0
_INTENSITY_MODERATE = 30.0
_INTENSITY_STRONG = 50.0

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.3,
            max_completion_tokens=3500,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm


# ──────────────────────────────────────────────────────────────────────────
# Prompt — design/30-analysis/global-trends.md §6.3.
# Phase 1/2 산식 결과를 input 으로 받아 Phase 3/4/5 만 LLM.
# ──────────────────────────────────────────────────────────────────────────

_GLOBAL_TRENDS_PROMPT = """\
# SK AX 글로벌 트렌드 분석 전문가

당신은 SK AX 사업전략팀의 글로벌 트렌드 분석 전문가입니다.
**글로벌 빅테크 6사의 최근 동향이 SK AX 의 국내 IT 서비스 사업
({sk_ax_business_lines}) 에 어떤 영향을 미치는지** 를 4-phase 로 추론합니다.

## 입력 데이터

### 분석 기간 (KST 절대 기준)
- **since**: {since}
- **until**: {until}
- **label**: 최근 {window_days}일 (KST)

### Phase 1 산식 결과 — Global Snapshots
{snapshots_json}

### Phase 2 산식 결과 — Trend Detections
{trend_detections_json}

### Global Context Packs (cold-start — 미적용)
*Phase K3 도입 전 — snapshots / trend_detections 만으로 추론.*

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **출처 prefix**: 정량 수치 앞에 `[공식 keynote]` / `[SEC 10-Q]` / `[Tier2 기사]` / `[자체 추정]`
- **화자 고정**: `"NVIDIA 가 X 했다"` 금지 → `"NVIDIA 의 X 는 SK AX 의 ai_managed 에 ___ 영향"`
- **환각 금지**: 입력 snapshots / trend_detections 에 없는 회사/수치/theme 추가 금지

### 일반 규칙 (17 요소 매핑)
1. **(#1 역할)** SK AX 사업전략팀 관점만
2. **(#2 추적 대상)** 6 글로벌 + SK AX 자체
3. **(#5 분석 기간)** analysis_period 절대 기준
4. **(#7 단순 요약 금지)** trend × impact pattern (Phase 2 결과 활용)
5. **(#9 변화 감지)** Phase 2 의 ±20/30/50% band 활용
6. **(#10 수익화 관점)** impact_matrix.direction (positive/neutral/negative)
7. **(#11 정량 우선)** 정성 표현 뒤 수치 (예: `"AI 인프라 지출 YoY +35%"`)
8. **(#13 SK AX 화자)** 위 pattern 강제
9. **(#15 우선순위)** forecasts 는 baseline 만 기본 — risk_level=high 시만
   optimistic/pessimistic 추가
10. **(#16 리스크)** risk_assumptions 2~3개
11. **(#17 반복 추적)** follow_up_questions 2~3개

## 추론 단계 (Chain of Thought)

### Phase 3 — Impact Mapping
각 (trend × sk_ax_line) cell:
- **direction**: positive / neutral / negative
- **magnitude**: low / medium / high
- **channel**: 영향 경로 (≤ 100자, 구체적)
- **quant_hint**: 가능 시 정량 (예: `"AI GPU 가격 +30%"`)
- **source_marker**: 절대 규칙 prefix

상위 (trend × sk_ax_line) 조합 중 의미 있는 cell 만 출력 (5~10개).

### Phase 4 — Forecast
1Q / 6M / 1Y 각 baseline 1개 (총 3 forecast). risk_level=high 인 경우만 추가 시나리오 생성 가능.

각 forecast:
- **narrative**: ≤ 300자
- **sk_ax_impact**: ≤ 200자, 사업 line 별 영향
- **drivers**: 3~5개 가정
- **risk_level**: low / medium / high
- **recommended_response**: SK AX 권장 대응 ≤ 200자

### Phase 5 — Synthesis
- **final_one_liner**: ≤ 100자, SK AX 관점, 모호 X
- **sk_ax_implication**: 1~2 문장 (긍정/중립/부정 명시)
- **follow_up_questions**: 2~3개
- **risk_assumptions**: 본 분석이 틀릴 가정 2~3개

## 3-Tier Observability 출력

### Tier 1 — reasoning_trail (사용자 default)
정확히 **4~5 step**. 권장 label sequence:
1. `"글로벌 스냅샷"` 2. `"트렌드 감지"` 3. `"SK AX 영향"` 4. `"전망"` 5. `"결론"`

각 step: `seq` + `label` (≤ 12자) + `one_liner` (≤ 80자, 수치 1개 우선) +
`evidence_refs` (global card_id).

### Tier 2 — reasoning_steps (상세)
phase ∈ {{snapshot, trend_detect, impact_map, forecast, synthesis}} 각 1+ step.

### Tier 3 — langfuse_trace_id
`null` 로 출력. 미들웨어 자동 매핑.

## 출력 형식 (strict JSON)

```json
{{
  "impact_matrix": [
    {{
      "trend_theme": "...",
      "sk_ax_line": "ai_managed",
      "direction": "positive",
      "magnitude": "medium",
      "channel": "...",
      "quant_hint": "...",
      "source_marker": "[Tier2 기사]"
    }}
  ],
  "forecasts": [
    {{
      "horizon": "1Q",
      "scenario": "baseline",
      "narrative": "...",
      "sk_ax_impact": "...",
      "drivers": ["..."],
      "risk_level": "medium",
      "recommended_response": "..."
    }}
  ],
  "final_one_liner": "≤ 100자",
  "sk_ax_implication": "1~2 문장. 긍정/중립/부정 명시.",
  "follow_up_questions": ["...", "..."],
  "risk_assumptions": ["...", "..."],
  "reasoning_trail": [
    {{
      "seq": 1, "label": "글로벌 스냅샷",
      "one_liner": "...", "evidence_refs": ["CN-..."],
      "langfuse_observation_id": null
    }}
  ],
  "reasoning_steps": [
    {{
      "step_idx": 0, "phase": "impact_map",
      "question": "...", "inputs_used": [],
      "answer": "...", "intermediate_conclusion": "...",
      "confidence": 0.0, "langfuse_observation_id": null
    }}
  ],
  "confidence": 0.0,
  "sources_used": ["CN-..."]
}}
```

JSON 만 출력. 다른 텍스트 추가 금지.
"""


class GlobalTrendsAgent:
    """Phase 1+2 결정적 산식, Phase 3+4+5 LLM 단일 호출."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    @with_ledger_writeback("GlobalTrendsAgent")
    async def run(
        self,
        company_ids: list[str] | None = None,
        focus_themes: list[str] | None = None,
        window_days: int = 30,
        sk_ax_business_lines: list[str] | None = None,
    ) -> dict:
        """글로벌 트렌드 분석 — design §6 흐름.

        Args:
            company_ids: 분석 대상 글로벌 회사 ids. None 이면 default 6사.
            focus_themes: 특정 theme 필터.
            window_days: 카드 조회 윈도우 (기본 30일).
            sk_ax_business_lines: 영향 매트릭스 컬럼 축. None 이면 default 5종.

        Returns:
            GlobalTrendsOutput dict — design §5 schema.
        """
        companies = list(company_ids) if company_ids else list(_GLOBAL_COMPANIES)
        lines = list(sk_ax_business_lines) if sk_ax_business_lines else list(_SK_AX_LINES_DEFAULT)

        until = datetime.now(UTC)
        since = until - timedelta(days=window_days)
        prev_since = since - timedelta(days=window_days)

        snapshots, all_cards = _build_snapshots(companies, since, until)
        trend_detections = _detect_trends(all_cards, companies, since, prev_since, focus_themes)
        analysis_period = {
            "since": since.date().isoformat(),
            "until": until.date().isoformat(),
            "window_days": window_days,
            "label": f"최근 {window_days}일 (KST)",
        }

        # 글로벌 카드 0건 — LLM 스킵, graceful 빈 응답
        if not all_cards:
            log.info("GlobalTrends | 글로벌 카드 0건 (companies=%s) — LLM 스킵", companies)
            return _empty_response(
                companies,
                analysis_period,
                snapshots,
                "글로벌 카드 데이터 없음 — ingestion 가 글로벌 6사 미커버 (4 Korean peer only)",
            )

        prompt = (
            _GLOBAL_TRENDS_PROMPT.replace("{sk_ax_business_lines}", ", ".join(lines))
            .replace("{since}", since.date().isoformat())
            .replace("{until}", until.date().isoformat())
            .replace("{window_days}", str(window_days))
            .replace("{snapshots_json}", json.dumps([s for s in snapshots], ensure_ascii=False))
            .replace("{trend_detections_json}", json.dumps(trend_detections, ensure_ascii=False))
        )

        try:
            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="GlobalTrendsAgent",
                    phase="run",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
        except Exception as e:
            log.exception("GlobalTrendsAgent LLM 호출 실패 | error=%s", e)
            return _error_response(
                companies,
                analysis_period,
                snapshots,
                trend_detections,
                "LLM 호출 실패",
                str(e),
            )

        result = _parse_and_validate(content, all_cards, companies)
        result["analysis_period"] = analysis_period
        result["snapshots"] = snapshots
        result["trend_detections"] = trend_detections
        result["company_ids"] = companies
        result.setdefault("provenance", {}).update(
            {
                "llm_model": _LLM_MODEL,
                "prompt_version": _PROMPT_VERSION,
                "source_card_ids": [c["id"] for c in all_cards],
                "global_companies": companies,
                "sk_ax_lines": lines,
            }
        )
        return result


# ──────────────────────────────────────────────────────────────────────────
# Phase 1 — Snapshot (deterministic)
# ──────────────────────────────────────────────────────────────────────────


def _iso_or_none(value: object) -> str | None:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else None


def _build_snapshots(
    companies: list[str], since: datetime, until: datetime
) -> tuple[list[dict], list[dict]]:
    snapshots: list[dict] = []
    all_cards: list[dict] = []
    for cid in companies:
        cards = _fetch_company_cards(cid, since, until)
        all_cards.extend(cards)
        themes = _themes_counter(cards)
        top_themes = [t for t, _ in themes.most_common(3)]
        headlines = [
            {
                "title": c.get("title", ""),
                "source": (c.get("implication") or {}).get("source_name") or "",
                "source_tier": (c.get("implication") or {}).get("source_tier") or "Tier2",
                "card_id": c["id"],
                "published_at_kst": _iso_or_none(c.get("created_at")),
            }
            for c in cards[:_MAX_HEADLINES]
        ]
        snapshots.append(
            {
                "company_id": cid,
                "card_count": len(cards),
                "top_themes": top_themes,
                "headline_announcements": headlines,
                "source_marker": "[Tier2 기사]" if cards else "[데이터 없음]",
            }
        )
    return snapshots, all_cards


# ──────────────────────────────────────────────────────────────────────────
# Phase 2 — Trend Detection (deterministic)
# ──────────────────────────────────────────────────────────────────────────


def _detect_trends(
    current_cards: list[dict],
    companies: list[str],
    current_since: datetime,
    prev_since: datetime,
    focus_themes: list[str] | None,
) -> list[dict]:
    if not current_cards:
        return []

    cur = _themes_counter(current_cards)
    # prior window — same companies, but immediately before current
    prev_cards = []
    for cid in companies:
        prev_cards.extend(_fetch_company_cards(cid, prev_since, current_since))
    prev = _themes_counter(prev_cards)

    detections: list[dict] = []
    for theme, cur_n in cur.most_common():
        if focus_themes and theme not in focus_themes:
            continue
        prev_n = max(prev.get(theme, 0), 1)
        delta = (cur_n - prev_n) / prev_n * 100.0
        if abs(delta) < _INTENSITY_WEAK:
            continue
        intensity = _intensity(abs(delta))
        leaders = _leaders_for(theme, current_cards)
        evidence = _evidence_for(theme, current_cards)
        detections.append(
            {
                "theme": theme,
                "frequency_delta_pct": round(delta, 1),
                "intensity": intensity,
                "leading_companies": leaders,
                "evidence_card_ids": evidence,
            }
        )
        if len(detections) >= 10:
            break
    return detections


def _intensity(magnitude: float) -> str:
    if magnitude >= _INTENSITY_STRONG:
        return "strong"
    if magnitude >= _INTENSITY_MODERATE:
        return "moderate"
    return "weak"


def _themes_counter(cards: list[dict]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for c in cards:
        for theme in _card_themes(c):
            counter[theme] += 1
    return counter


def _card_themes(card: dict) -> list[str]:
    impl = card.get("implication") or {}
    raw_themes = impl.get("themes")
    if isinstance(raw_themes, list):
        themes = [str(t).lower() for t in raw_themes if t]
        if themes:
            return themes
    raw_sectors = impl.get("sectors")
    if isinstance(raw_sectors, list):
        sectors = [str(s).lower() for s in raw_sectors if s]
        if sectors:
            return sectors
    sector = impl.get("sector") or card.get("event_type")
    return [str(sector).lower()] if sector else []


def _leaders_for(theme: str, cards: list[dict]) -> list[str]:
    counter: Counter[str] = Counter()
    for c in cards:
        if theme in _card_themes(c):
            peer = c.get("peer_id")
            if peer:
                counter[peer] += 1
    return [peer for peer, _ in counter.most_common(3)]


def _evidence_for(theme: str, cards: list[dict]) -> list[str]:
    return [c["id"] for c in cards if theme in _card_themes(c)][:5]


# ──────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────


def _fetch_company_cards(company_id: str, since: datetime, until: datetime) -> list[dict]:
    sql = (
        "SELECT id, company AS peer_id, title, summary_lines, event_type, importance, "
        "importance_score, implication, created_at "
        "FROM card_news WHERE company = :company AND created_at >= :since "
        "AND created_at < :until ORDER BY created_at DESC LIMIT :limit"
    )
    params = {
        "company": company_id,
        "since": since,
        "until": until,
        "limit": _MAX_CARDS_PER_COMPANY,
    }
    try:
        with SessionLocal() as db:
            rows = db.execute(text(sql), params).mappings().all()
    except Exception as e:
        log.exception("GlobalTrends card_news 조회 실패 | %s", e)
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


# ──────────────────────────────────────────────────────────────────────────
# Parse + validate
# ──────────────────────────────────────────────────────────────────────────


def _parse_and_validate(content: str, all_cards: list[dict], companies: list[str]) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("GlobalTrends JSON parse 실패 — prefix=%s", content[:200])
        return _error_response(
            companies,
            {},
            [],
            [],
            "JSON parse 실패",
            "LLM 응답이 JSON 이 아님",
        )

    if not isinstance(data, dict):
        return _error_response(companies, {}, [], [], "응답 형식 오류", "JSON object 가 아님")

    data.setdefault("impact_matrix", [])
    data.setdefault("forecasts", [])
    data.setdefault("follow_up_questions", [])
    data.setdefault("risk_assumptions", [])

    # design 제약 강제
    data["final_one_liner"] = clip_final_one_liner(data.get("final_one_liner", ""))
    data["sk_ax_implication"] = clip_implication(data.get("sk_ax_implication", ""))
    data["reasoning_trail"] = cap_reasoning_trail(data.get("reasoning_trail", []))
    data["reasoning_steps"] = cap_reasoning_steps(data.get("reasoning_steps", []))
    data["confidence"] = confidence_in_range(data.get("confidence", 0.0))
    data["sources_used"] = dedup_and_cap(data.get("sources_used") or [c["id"] for c in all_cards])

    # peer_ids — for analysis_ledger carry-over
    data["peer_ids"] = companies

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
        return "근거 불충분 — confidence < 0.6"
    if not data.get("impact_matrix"):
        return "impact_matrix 비어 있음 — 트렌드 신호 부족"
    return None


def _empty_response(
    companies: list[str],
    analysis_period: dict,
    snapshots: list[dict],
    warning: str,
) -> dict:
    return {
        "analysis_period": analysis_period,
        "snapshots": snapshots,
        "trend_detections": [],
        "impact_matrix": [],
        "forecasts": [],
        "final_one_liner": "",
        "sk_ax_implication": "",
        "follow_up_questions": [],
        "risk_assumptions": [],
        "reasoning_trail": [],
        "reasoning_steps": [],
        "langfuse_trace_id": None,
        "confidence": 0.0,
        "sources_used": [],
        "company_ids": companies,
        "peer_ids": companies,
        "warning": warning,
        "provenance": {
            "llm_model": _LLM_MODEL,
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": [],
            "global_companies": companies,
            "cold_start": True,
        },
    }


def _error_response(
    companies: list[str],
    analysis_period: dict,
    snapshots: list[dict],
    trend_detections: list[dict],
    short_reason: str,
    detail: str,
) -> dict:
    log.warning("GlobalTrends error | %s | detail=%s", short_reason, detail)
    return {
        "analysis_period": analysis_period,
        "snapshots": snapshots,
        "trend_detections": trend_detections,
        "impact_matrix": [],
        "forecasts": [],
        "final_one_liner": "",
        "sk_ax_implication": "",
        "follow_up_questions": [],
        "risk_assumptions": [],
        "reasoning_trail": [],
        "reasoning_steps": [],
        "langfuse_trace_id": None,
        "confidence": 0.0,
        "sources_used": [],
        "company_ids": companies,
        "peer_ids": companies,
        "warning": f"{short_reason} — {detail}",
        "provenance": {
            "llm_model": _LLM_MODEL,
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": [],
            "global_companies": companies,
            "error": short_reason,
        },
    }
