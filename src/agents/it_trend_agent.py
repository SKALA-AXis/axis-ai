"""IT trend context agent — 5-phase global IT trend extractor + peer alignment.

design: ``axis-ai/design/30-analysis/global-trends.md``.

ITTrendAgent 는 카드뉴스 생성 에이전트가 아니다. 글로벌 6 사 (NVIDIA / Apple /
Microsoft / Google / Amazon / Meta) 뉴스룸 + SPRi / BCG 리서치 자료를 기반으로
글로벌 IT 트렌드를 추출하고, AX / peer 4사의 동향이 그 트렌드와 같은 결로
가는지 alignment 를 계산한다.

핵심 entry point:

    ``ITTrendAgent().generate(trend_input)``

Pipeline (design §6):

    Phase 1 (Snapshot,     deterministic) — 글로벌 6 사별 카드 카운트 + top themes
    Phase 2 (Trend Detect, deterministic) — keyword mention / frequency_delta / intensity
    Phase 3 (Peer Align,   deterministic + LLM batch) — keyword × peer alignment
    Phase 4 (Impact Map,   LLM)           — trend × SK AX business line 매트릭스
    Phase 5 (Forecast/Synth, LLM)         — 1Q/6M/1Y narrative + final_one_liner + sk_ax_implication

LLM 호출 총 3 회 (P3 batch strategic_notes / P4 impact / P5 synthesis).

결과는 ``global_industry_trends`` 에 keyword 별 row 직접 upsert (design §7).
``analysis_ledger`` 는 V30 line 694 에서 DROP 되어 더는 사용하지 않는다.
``@with_ledger_writeback`` 데코레이터는 **적용하지 않는다**.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from langchain_openai import ChatOpenAI

from src.analysis.models import TrendContext
from src.config.global_companies import GLOBAL_COMPANY_IDS
from src.db.article_store import (
    DEFAULT_PEER_COMPANY_IDS,
    fetch_global_trend_inputs,
    fetch_peer_cards_for_alignment,
    invalidate_trend_context_cache,
    upsert_global_industry_trends,
)
from src.observability.langfuse_client import tracing_config

log = logging.getLogger(__name__)

_TREND_SOURCE_NAMES = {"spri", "bcg"}
_GLOBAL_NEWSROOM_SOURCE_TYPES = {"global_newsroom", "company_newsroom"}

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "global-trends-v1.0"

_DEFAULT_SK_AX_BUSINESS_LINES: tuple[str, ...] = (
    "cloud_ax",
    "manufacturing_ax",
    "data_platform",
    "smart_factory",
)

# Walking Skeleton: 사전 정의된 글로벌 IT 트렌드 keyword 후보군.
# raw article 의 title + content 에서 substring match (case-insensitive).
# 추후 LLM keyword extraction 으로 보강 가능.
_TREND_KEYWORD_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    # (keyword, category, regex_pattern — 빈 문자열이면 keyword 그대로 substring)
    ("agentic ai", "ai_tech", r"agentic\s*ai|agent\s*ai|에이전트\s*ai|에이전틱\s*ai"),
    ("generative ai", "ai_tech", r"generative\s*ai|generativeAI|생성형\s*ai|genai"),
    ("multimodal", "ai_tech", r"multi[-\s]?modal|멀티\s*모달"),
    ("rag", "ai_tech", r"\brag\b|retrieval[-\s]?augmented"),
    ("llm", "ai_tech", r"\bllm[s]?\b|large\s*language\s*model"),
    ("gpu", "ai_infra", r"\bgpu[s]?\b|graphics\s*processing"),
    ("inference", "ai_infra", r"\binference\b|추론"),
    ("foundation model", "ai_tech", r"foundation\s*model|기반\s*모델"),
    ("ai infrastructure", "ai_infra", r"ai\s*infra(structure)?|ai\s*인프라"),
    ("cloud", "cloud", r"\bcloud\b|클라우드"),
    ("data center", "ai_infra", r"data\s*center|데이터\s*센터"),
    ("ai agent", "ai_tech", r"ai\s*agent|ai\s*에이전트"),
    ("copilot", "ai_tech", r"copilot|코파일럿"),
    ("security", "security", r"\bsecurity\b|보안|cybersecurity"),
    ("partnership", "deal", r"partnership|파트너십|제휴|joint\s*venture"),
    ("acquisition", "deal", r"acquisition|인수|merger"),
    ("open source", "ai_tech", r"open[-\s]?source|오픈\s*소스"),
    ("edge ai", "ai_tech", r"edge\s*ai|온디바이스|on[-\s]?device"),
    ("quantum", "ai_tech", r"quantum|양자"),
    ("robotics", "ai_tech", r"\brobot(ic|ics)?\b|로봇"),
)


@dataclass
class ITTrendInput:
    """Runtime DTO for IT trend context generation."""

    trend_items: list[dict[str, Any]]
    period: str | None = None
    source_groups: list[str] = field(default_factory=list)
    previous_trend_context: dict[str, Any] | None = None
    reference_issue_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def items(self) -> list[dict[str, Any]]:
        """Backward-compatible alias for callers that still read `.items`."""
        return self.trend_items


_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.15,
            max_completion_tokens=3000,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm


# ──────────────────────────────────────────────────────────────────────────
# source_analysis_id helpers (design §7) — VARCHAR(100) hard cap + slug 충돌 방어.
# ──────────────────────────────────────────────────────────────────────────

_MAX_SOURCE_ANALYSIS_ID = 100  # V29 schema.sql:457 — VARCHAR(100) hard cap


def _slugify(keyword: str, max_len: int = 40) -> str:
    """source_analysis_id 의 human-readable 부분.

    한글/영문/숫자만 남기고 공백을 dash 로. max_len 으로 cap.
    예: "Agentic AI" → "agentic-ai", "에이전트 AI" → "에이전트-ai".
    """
    text = (keyword or "").strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^0-9a-z가-힣\-]+", "", text)
    return (text[:max_len] or "unknown").rstrip("-") or "unknown"


def _make_source_analysis_id(batch_id: str, idx: int, keyword: str) -> str:
    """``global-YYYYMMDD-HHMMSS-NNN-<slug>`` 형식.

    - idx (3 자리 zero-pad) — 같은 batch 내 keyword 순번. slug 충돌 시 unique 보장.
    - short slug — human-readable. max 40 자 cap.
    - hard 100 자 cap — keyword 가 비정상 길면 sha1 8 자로 fallback.
    """
    suffix = f"{idx:03d}-{_slugify(keyword, max_len=40)}"
    candidate = f"{batch_id}-{suffix}"
    if len(candidate) <= _MAX_SOURCE_ANALYSIS_ID:
        return candidate
    short_hash = hashlib.sha1(keyword.encode("utf-8")).hexdigest()[:8]
    return f"{batch_id}-{idx:03d}-{short_hash}"[:_MAX_SOURCE_ANALYSIS_ID]


# ──────────────────────────────────────────────────────────────────────────
# ITTrendAgent
# ──────────────────────────────────────────────────────────────────────────


class ITTrendAgent:
    """Generate global IT TrendContext + peer alignment + impact + forecast.

    @with_ledger_writeback 는 **적용하지 않는다** — analysis_ledger 는 V30 line 694
    에서 DROP. 결과는 global_industry_trends 에 직접 upsert (design §7).
    """

    prompt_version = _PROMPT_VERSION

    def build_input(
        self,
        *,
        items: list[dict[str, Any]],
        period: str | None = None,
        source_groups: list[str] | None = None,
        previous_trend_context: dict[str, Any] | None = None,
        reference_issue_results: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ITTrendInput:
        trend_items, candidate_reference_items, unsupported_items = _split_trend_inputs(items or [])
        reference_results = list(reference_issue_results or [])
        reference_results.extend(_reference_issue_results_from_items(candidate_reference_items))
        normalized_groups = _source_groups(
            explicit=source_groups,
            trend_items=trend_items,
            reference_results=reference_results,
        )
        return ITTrendInput(
            trend_items=trend_items,
            period=period,
            source_groups=normalized_groups,
            previous_trend_context=dict(previous_trend_context or {}),
            reference_issue_results=reference_results,
            metadata={
                **dict(metadata or {}),
                "research_source_count": len(trend_items),
                "global_newsroom_reference_count": len(reference_results),
                "unsupported_input_count": len(unsupported_items),
                "unsupported_input_reason": (
                    "trend_context_accepts_spri_bcg_and_global_newsroom_analysis_results"
                    if unsupported_items
                    else ""
                ),
            },
        )

    def generate(self, trend_input: ITTrendInput) -> dict[str, Any]:
        """5-phase pipeline — design §6.

        Phase 1/2 deterministic, Phase 3/4/5 deterministic + LLM (3 호출).
        결과는 ``global_industry_trends`` 에 keyword 별 row 로 upsert.
        """
        generated_at = datetime.now(UTC)
        batch_id = f"global-{generated_at.strftime('%Y%m%d-%H%M%S')}"
        meta = dict(trend_input.metadata or {})

        window_days = int(meta.get("window_days", 30) or 30)
        peer_company_ids: list[str] = list(
            meta.get("peer_company_ids") or list(DEFAULT_PEER_COMPANY_IDS)
        )
        sk_ax_business_lines: list[str] = list(
            meta.get("sk_ax_business_lines") or list(_DEFAULT_SK_AX_BUSINESS_LINES)
        )
        include_peer_alignment: bool = bool(meta.get("include_peer_alignment", True))
        min_mention_count: int = int(meta.get("min_mention_count", 3) or 3)
        max_trend_count: int = int(meta.get("max_trend_count", 8) or 8)
        focus_themes: list[str] = [t.lower() for t in (meta.get("focus_themes") or [])]

        # 1) 원천 데이터 fetch — global newsroom + research + 기존 input.
        try:
            fetched_global = fetch_global_trend_inputs(window_days=window_days)
        except Exception:
            log.exception("ITTrendAgent | fetch_global_trend_inputs 실패")
            fetched_global = []
        combined_items: list[dict[str, Any]] = []
        combined_items.extend(trend_input.trend_items or [])
        combined_items.extend(fetched_global)
        combined_items = _dedupe_items_by_id(combined_items)

        global_rows = [r for r in combined_items if _is_global_newsroom_row(r)]
        research_rows = [r for r in combined_items if _is_research_row(r)]

        reasoning_steps: list[dict[str, Any]] = []
        warning: str | None = None

        # 2) Phase 1 — Snapshot (deterministic).
        snapshots = _phase1_snapshot(global_rows)
        snapshot_card_total = sum(s["card_count"] for s in snapshots)
        reasoning_steps.append(
            {
                "step_idx": 1,
                "phase": "snapshot",
                "question": "글로벌 6 사 newsroom 카드 분포는?",
                "answer": f"총 {snapshot_card_total} 건, 회사별 분포: "
                + ", ".join(f"{s['company_id']}={s['card_count']}" for s in snapshots),
                "confidence": 0.95 if snapshot_card_total > 0 else 0.0,
            }
        )
        if snapshot_card_total == 0:
            warning = "Phase 1 snapshot empty (global newsroom rows = 0) — global_industry_trends 저장 skip."
            log.warning("ITTrendAgent | %s", warning)
            return _empty_result(
                batch_id=batch_id,
                generated_at=generated_at,
                trend_input=trend_input,
                snapshots=snapshots,
                reasoning_steps=reasoning_steps,
                warning=warning,
            )

        # 3) Phase 2 — Trend Detection (deterministic).
        detections = _phase2_trends(
            snapshots=snapshots,
            global_rows=global_rows,
            research_rows=research_rows,
            previous_trend_context=trend_input.previous_trend_context,
            min_mention_count=min_mention_count,
            max_trend_count=max_trend_count,
            focus_themes=focus_themes,
        )
        reasoning_steps.append(
            {
                "step_idx": 2,
                "phase": "trend_detect",
                "question": "글로벌 6 사 + 리서치에서 반복되는 trend keyword 는?",
                "answer": ", ".join(d["theme"] for d in detections) or "(detections empty)",
                "confidence": 0.85 if detections else 0.0,
            }
        )
        if not detections:
            warning = "Phase 2 detections empty — global_industry_trends 저장 skip."
            log.warning("ITTrendAgent | %s", warning)
            return _empty_result(
                batch_id=batch_id,
                generated_at=generated_at,
                trend_input=trend_input,
                snapshots=snapshots,
                reasoning_steps=reasoning_steps,
                warning=warning,
            )

        # 4) Phase 3 — Peer Alignment (deterministic 점수 + LLM batch strategic_note).
        if include_peer_alignment:
            alignment = _phase3_peer_alignment(
                detections=detections,
                snapshots=snapshots,
                peer_company_ids=peer_company_ids,
                window_days=window_days,
            )
        else:
            alignment = {}
        reasoning_steps.append(
            {
                "step_idx": 3,
                "phase": "peer_alignment",
                "question": "AX 와 peer 4 사의 동향이 글로벌 트렌드와 같은 결로 가는가?",
                "answer": _summarize_alignment(alignment),
                "confidence": 0.7 if alignment else 0.0,
            }
        )

        # 5) Phase 4 — Impact Mapping (LLM).
        impact_matrix = _phase4_impact(
            detections=detections,
            sk_ax_business_lines=sk_ax_business_lines,
        )
        reasoning_steps.append(
            {
                "step_idx": 4,
                "phase": "impact_map",
                "question": "각 trend 가 SK AX 사업라인에 어떤 영향을 주는가?",
                "answer": _summarize_impact(impact_matrix),
                "confidence": 0.65 if impact_matrix else 0.0,
            }
        )

        # 6) Phase 5 — Forecast + Synthesis (LLM).
        synthesis = _phase5_forecast_synthesis(
            detections=detections,
            alignment=alignment,
            impact_matrix=impact_matrix,
        )
        forecasts = synthesis.get("forecasts", [])
        final_one_liner: str = synthesis.get("final_one_liner", "") or ""
        sk_ax_implication: str = synthesis.get("sk_ax_implication", "") or ""
        per_keyword_title: dict[str, str] = synthesis.get("per_keyword_title", {}) or {}
        per_keyword_summary: dict[str, str] = synthesis.get("per_keyword_summary", {}) or {}
        per_keyword_implication: dict[str, str] = synthesis.get("per_keyword_implication", {}) or {}
        confidence: float = float(synthesis.get("confidence", 0.6) or 0.6)
        reasoning_steps.append(
            {
                "step_idx": 5,
                "phase": "synthesis",
                "question": "SK AX 가 다음 1Q / 6M / 1Y 에 어떤 자세를 가져야 하는가?",
                "answer": final_one_liner or "(synthesis empty)",
                "confidence": confidence,
            }
        )

        # 7) Persistence — direct upsert to global_industry_trends (design §7).
        rows = _build_persistence_rows(
            batch_id=batch_id,
            generated_at=generated_at,
            detections=detections,
            alignment=alignment,
            impact_matrix=impact_matrix,
            forecasts=forecasts,
            global_rows=global_rows,
            snapshots=snapshots,
            per_keyword_title=per_keyword_title,
            per_keyword_summary=per_keyword_summary,
            per_keyword_implication=per_keyword_implication,
            confidence=confidence,
            sk_ax_implication=sk_ax_implication,
            final_one_liner=final_one_liner,
        )
        persisted = 0
        try:
            persisted = upsert_global_industry_trends(rows)
        except Exception:
            log.exception("ITTrendAgent | upsert_global_industry_trends 실패")
            warning = "upsert_global_industry_trends 실패 — DB 로그 확인 필요"
        else:
            # design §5.2 — cron path 즉시 반영.
            try:
                invalidate_trend_context_cache()
            except Exception:
                log.exception("ITTrendAgent | invalidate_trend_context_cache 실패 (best-effort)")

        # 8) TrendContext shape — AnalysisAgent 가 직접 참조할 수 있는 DTO.
        trend_context = _build_trend_context(
            period=trend_input.period,
            detections=detections,
            global_rows=global_rows,
            generated_at=generated_at,
            warning=warning,
        )

        return {
            "agent": type(self).__name__,
            "prompt_version": self.prompt_version,
            "analysis_id": batch_id,
            "analysis_type": "global",
            "generated_at": generated_at.isoformat(timespec="seconds"),
            "period": trend_input.period,
            "analysis_period": {
                "window_days": window_days,
                "trend_date": generated_at.date().isoformat(),
            },
            "source_groups": trend_input.source_groups,
            "source_count": len(combined_items),
            "snapshots": snapshots,
            "trend_detections": detections,
            "peer_alignment": alignment,
            "impact_matrix": impact_matrix,
            "forecasts": forecasts,
            "final_one_liner": final_one_liner,
            "sk_ax_implication": sk_ax_implication,
            "reasoning_steps": reasoning_steps,
            "rows": rows,
            "persisted_row_count": persisted,
            "trend_context": trend_context.to_dict(),
            "trend_summary": trend_context.trend_summary,
            "trend_lines": trend_context.trend_lines,
            "signals": trend_context.signals,
            "sources": trend_context.sources,
            "reference_issue_ids": _reference_issue_ids(trend_input.reference_issue_results),
            "confidence": confidence,
            "warning": warning,
            "validation": {
                "pass": persisted > 0,
                "phase_completion": [1, 2, 3, 4, 5],
                "trend_source_policy": (
                    "SPRi/BCG research + global newsroom 6 companies (raw_articles direct fetch)"
                ),
            },
            "metadata": {
                **meta,
                "previous_trend_context_provided": bool(trend_input.previous_trend_context),
                "global_newsroom_row_count": len(global_rows),
                "research_row_count": len(research_rows),
                "peer_company_ids": peer_company_ids,
                "sk_ax_business_lines": sk_ax_business_lines,
            },
        }


# ──────────────────────────────────────────────────────────────────────────
# Phase 1 — Snapshot (deterministic).
# ──────────────────────────────────────────────────────────────────────────


def _phase1_snapshot(global_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """글로벌 6 사별 카드 카운트 + top themes + 최근 headline."""
    by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in global_rows:
        company_id = _global_company_id_from_row(row)
        if company_id:
            by_company[company_id].append(row)

    snapshots: list[dict[str, Any]] = []
    for company_id in GLOBAL_COMPANY_IDS:
        rows = by_company.get(company_id, [])
        keywords = _extract_keywords_from_rows(rows)
        snapshots.append(
            {
                "company_id": company_id,
                "card_count": len(rows),
                "top_themes": [kw for kw, _ in keywords.most_common(5)],
                "headline_announcements": [
                    {
                        "title": (r.get("title") or "")[:200],
                        "url": r.get("url"),
                        "published_at": _iso(r.get("published_at") or r.get("collected_at")),
                    }
                    for r in rows[:3]
                ],
                "source_marker": f"raw_articles WHERE source_name~{company_id}",
            }
        )
    # 6사 외 row 가 남아 있으면 별도 group 으로 모음 (graceful).
    other_rows = [r for r in global_rows if not _global_company_id_from_row(r)]
    if other_rows:
        snapshots.append(
            {
                "company_id": "other",
                "card_count": len(other_rows),
                "top_themes": [
                    kw for kw, _ in _extract_keywords_from_rows(other_rows).most_common(5)
                ],
                "headline_announcements": [
                    {"title": (r.get("title") or "")[:200], "url": r.get("url")}
                    for r in other_rows[:3]
                ],
                "source_marker": "raw_articles unmatched global newsroom",
            }
        )
    return snapshots


# ──────────────────────────────────────────────────────────────────────────
# Phase 2 — Trend Detection (deterministic).
# ──────────────────────────────────────────────────────────────────────────


def _phase2_trends(
    *,
    snapshots: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    research_rows: list[dict[str, Any]],
    previous_trend_context: dict[str, Any] | None,
    min_mention_count: int,
    max_trend_count: int,
    focus_themes: list[str],
) -> list[dict[str, Any]]:
    """반복 keyword 추출 + frequency_delta_pct + intensity 분류 + leading_companies."""
    all_rows = global_rows + research_rows
    keyword_counts = _extract_keywords_from_rows(all_rows)
    # research 는 weight 1.0, newsroom 0.7 — 산식 단순화 위해 raw count 그대로 사용.

    prev_counts: Counter[str] = Counter()
    if previous_trend_context:
        for signal in previous_trend_context.get("signals", []) or []:
            kw = (signal.get("signal") or "").lower()
            intensity = (signal.get("intensity") or "").lower()
            prev_counts[kw] = {"weak": 3, "moderate": 8, "strong": 18}.get(intensity, 0)

    detections: list[dict[str, Any]] = []
    for kw, n in keyword_counts.most_common(30):
        if n < min_mention_count:
            continue
        if focus_themes and kw not in focus_themes:
            continue
        prev_n = prev_counts.get(kw, 0)
        if prev_n > 0:
            delta = ((n - prev_n) / prev_n) * 100.0
        else:
            delta = 100.0 if n >= min_mention_count else 0.0
        intensity = _classify_intensity(n)
        leading = _leading_companies_for_keyword(snapshots, kw)
        category = _keyword_category(kw)
        detections.append(
            {
                "theme": kw,
                "mention_count": n,
                "frequency_delta_pct": round(delta, 2),
                "intensity": intensity,
                "leading_companies": leading,
                "evidence_card_ids": [],
                "keyword_category": category,
            }
        )
    detections.sort(key=lambda d: (d["mention_count"], d["frequency_delta_pct"]), reverse=True)
    return detections[:max_trend_count]


def _classify_intensity(n: int) -> str:
    if n >= 15:
        return "strong"
    if n >= 5:
        return "moderate"
    return "weak"


def _leading_companies_for_keyword(snapshots: list[dict[str, Any]], keyword: str) -> list[str]:
    company_match_counts: dict[str, int] = {}
    for snap in snapshots:
        if snap["company_id"] == "other":
            continue
        if keyword in (t.lower() for t in snap.get("top_themes", [])):
            company_match_counts[snap["company_id"]] = snap["card_count"]
    return [
        cid
        for cid, _ in sorted(company_match_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    ]


# ──────────────────────────────────────────────────────────────────────────
# Phase 3 — Peer Alignment (deterministic + LLM batch).
# ──────────────────────────────────────────────────────────────────────────


def _phase3_peer_alignment(
    *,
    detections: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    peer_company_ids: list[str],
    window_days: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    today = datetime.now(UTC).date()

    for det in detections:
        keyword = det["theme"]
        category = det.get("keyword_category")
        global_mention_count = det["mention_count"]
        # 글로벌 평균 활동 시점 (가장 최근 등장일).
        latest_global = _latest_global_date_for_keyword(snapshots, keyword)
        global_recency = (today - latest_global).days if latest_global else 999

        per_peer: list[dict[str, Any]] = []
        for peer_id in peer_company_ids:
            try:
                cards = fetch_peer_cards_for_alignment(
                    peer_id=peer_id,
                    window_days=window_days,
                    keyword=keyword,
                    keyword_category=category,
                )
            except Exception:
                log.exception("ITTrendAgent | peer alignment fetch 실패 | peer=%s", peer_id)
                cards = []

            peer_n = len(cards)
            score = _alignment_score(peer_n=peer_n, global_n=global_mention_count)
            alignment_type = _classify_alignment(score=score, peer_n=peer_n)

            latest_peer = _latest_date_from_cards(cards)
            recency_gap_days: int | None = None
            if latest_peer:
                peer_age = (today - latest_peer).days
                recency_gap_days = peer_age - global_recency

            per_peer.append(
                {
                    "peer_id": peer_id,
                    "alignment_type": alignment_type,
                    "alignment_score": round(score, 3),
                    "peer_mention_count": peer_n,
                    "global_mention_count": global_mention_count,
                    "recency_gap_days": recency_gap_days,
                    "evidence_card_ids": [str(c.get("id")) for c in cards[:5] if c.get("id")],
                    "strategic_note": "",
                }
            )
        result[keyword] = per_peer

    # LLM batch — 모든 (theme × peer) strategic_note 한 번에 채움.
    if any(p for plist in result.values() for p in plist):
        try:
            result = _llm_fill_strategic_notes(result, detections)
        except Exception:
            log.exception("ITTrendAgent | strategic_note LLM 실패 — 빈 문자열 유지")
    return result


def _alignment_score(*, peer_n: int, global_n: int) -> float:
    """Peer activity 가 글로벌 활동량 대비 얼마나 따라잡는지.

    score = peer_n / max(global_n / 3, 1) — clip [-1, 1].
    global 6 사 평균당 peer 가 비슷한 활동을 보이면 score ≈ 1.
    """
    if global_n <= 0:
        return 0.0
    ratio = peer_n / max(global_n / 3.0, 1.0)
    return max(min(ratio, 1.0), -1.0)


def _classify_alignment(*, score: float, peer_n: int) -> str:
    if peer_n == 0:
        return "missing"
    if score >= 0.6:
        return "aligned"
    if score >= 0.2:
        return "lagging"
    if score >= -0.2:
        return "diverging"
    return "missing"


def _llm_fill_strategic_notes(
    result: dict[str, list[dict[str, Any]]],
    detections: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """LLM 1 회 호출로 모든 (theme × peer) strategic_note 한 줄씩 채움."""
    payload = []
    for det in detections:
        keyword = det["theme"]
        peers = result.get(keyword, [])
        payload.append(
            {
                "theme": keyword,
                "intensity": det.get("intensity"),
                "leading_companies": det.get("leading_companies", []),
                "peers": [
                    {
                        "peer_id": p["peer_id"],
                        "alignment_type": p["alignment_type"],
                        "peer_mention_count": p["peer_mention_count"],
                        "global_mention_count": p["global_mention_count"],
                    }
                    for p in peers
                ],
            }
        )
    if not payload:
        return result

    prompt = (
        "당신은 SK AX 의 글로벌 IT 트렌드 전략가입니다. "
        "각 (theme, peer) 조합에 대해 한 줄짜리 strategic_note 를 작성하세요.\n"
        "- aligned: 어디서 따라가고 있는지\n"
        "- lagging: 무엇이 부족한지\n"
        "- missing: 왜 안 하는지 / 시급한가\n"
        "- diverging: 다른 방향이 맞는지\n\n"
        "응답은 반드시 다음 JSON object:\n"
        '{"notes": [{"theme":"...", "peer_id":"...", "strategic_note":"..."}, ...]}\n\n'
        "입력:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    response = _get_llm().invoke(
        prompt,
        config=tracing_config(
            agent="ITTrendAgent",
            phase="peer_alignment_notes",
            prompt_version=_PROMPT_VERSION,
        ),
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    notes = _safe_json_object(content).get("notes") or []
    by_pair: dict[tuple[str, str], str] = {}
    for item in notes:
        if not isinstance(item, dict):
            continue
        theme = str(item.get("theme") or "").lower()
        peer = str(item.get("peer_id") or "").strip()
        note = str(item.get("strategic_note") or "").strip()
        if theme and peer and note:
            by_pair[(theme, peer)] = note[:200]

    for keyword, peers in result.items():
        for p in peers:
            key = (keyword.lower(), p["peer_id"])
            if key in by_pair:
                p["strategic_note"] = by_pair[key]
    return result


# ──────────────────────────────────────────────────────────────────────────
# Phase 4 — Impact Mapping (LLM).
# ──────────────────────────────────────────────────────────────────────────


def _phase4_impact(
    *,
    detections: list[dict[str, Any]],
    sk_ax_business_lines: list[str],
) -> list[dict[str, Any]]:
    if not detections or not sk_ax_business_lines:
        return []
    trend_payload = [
        {
            "theme": d["theme"],
            "intensity": d.get("intensity"),
            "leading_companies": d.get("leading_companies", []),
            "mention_count": d.get("mention_count"),
        }
        for d in detections
    ]
    prompt = (
        "당신은 SK AX 의 사업 전략 분석가입니다. "
        "다음 글로벌 IT 트렌드들이 SK AX 사업라인에 미치는 영향을 매트릭스로 도출하세요.\n\n"
        f"SK AX 사업라인: {', '.join(sk_ax_business_lines)}\n\n"
        "각 (trend, sk_ax_line) 조합에 대해 direction (positive/neutral/negative), "
        "magnitude (low/medium/high), channel (한 줄), quant_hint (옵션) 을 결정.\n\n"
        "응답은 반드시 다음 JSON object:\n"
        '{"impact_matrix": [{"trend_theme":"...", "sk_ax_line":"...", '
        '"direction":"...", "magnitude":"...", "channel":"...", "quant_hint":"..."}]}\n\n'
        "각 trend 마다 최소 1 개 (가장 관련 깊은) sk_ax_line 매칭. 너무 많이 만들지 말 것.\n\n"
        "입력 trends:\n" + json.dumps(trend_payload, ensure_ascii=False, indent=2)
    )
    try:
        response = _get_llm().invoke(
            prompt,
            config=tracing_config(
                agent="ITTrendAgent",
                phase="impact_map",
                prompt_version=_PROMPT_VERSION,
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        cells = _safe_json_object(content).get("impact_matrix") or []
    except Exception:
        log.exception("ITTrendAgent | _phase4_impact LLM 실패")
        return []

    result: list[dict[str, Any]] = []
    valid_themes = {d["theme"] for d in detections}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        theme = str(cell.get("trend_theme") or "").strip().lower()
        if theme not in valid_themes:
            continue
        result.append(
            {
                "trend_theme": theme,
                "sk_ax_line": str(cell.get("sk_ax_line") or "").strip(),
                "direction": _clip_enum(
                    cell.get("direction"), {"positive", "neutral", "negative"}, "neutral"
                ),
                "magnitude": _clip_enum(cell.get("magnitude"), {"low", "medium", "high"}, "low"),
                "channel": str(cell.get("channel") or "")[:200],
                "quant_hint": str(cell.get("quant_hint") or "")[:200] or None,
                "source_marker": "llm_phase4_impact",
            }
        )
    return result


# ──────────────────────────────────────────────────────────────────────────
# Phase 5 — Forecast + Synthesis (LLM).
# ──────────────────────────────────────────────────────────────────────────


def _phase5_forecast_synthesis(
    *,
    detections: list[dict[str, Any]],
    alignment: dict[str, list[dict[str, Any]]],
    impact_matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    if not detections:
        return {}

    alignment_compact: dict[str, list[dict[str, Any]]] = {}
    for keyword, peers in alignment.items():
        alignment_compact[keyword] = [
            {
                "peer_id": p["peer_id"],
                "alignment_type": p["alignment_type"],
                "alignment_score": p["alignment_score"],
            }
            for p in peers
        ]
    prompt = (
        "당신은 SK AX 의 글로벌 IT 트렌드 시나리오 분석가입니다.\n"
        "다음 입력으로 다음 세 가지를 산출하세요:\n"
        "1. forecasts — 각 horizon (1Q, 6M, 1Y) 별 baseline narrative + sk_ax_impact + drivers + risk_level + recommended_response\n"
        "2. final_one_liner — SK AX 임원 한 명이 5초 안에 이해할 한 줄\n"
        "3. sk_ax_implication — SK AX 가 가져야 할 자세 / 행동 권고 (3 문장 이하)\n"
        "4. per_keyword — 각 trend 별 title (한 줄) + summary (1~2 문장) + implication (한 줄)\n"
        "5. confidence — 전반적 자신감 (0.0~1.0)\n\n"
        "응답 JSON object:\n"
        '{"forecasts": [{"horizon":"1Q","scenario":"baseline","narrative":"...",'
        '"sk_ax_impact":"...","drivers":["..."],"risk_level":"medium","recommended_response":"..."}],'
        '"final_one_liner":"...","sk_ax_implication":"...","confidence":0.7,'
        '"per_keyword":[{"theme":"...","title":"...","summary":"...","implication":"..."}]}\n\n'
        "입력 trends:\n"
        + json.dumps(detections, ensure_ascii=False, indent=2)
        + "\n\n입력 alignment (요약):\n"
        + json.dumps(alignment_compact, ensure_ascii=False, indent=2)
        + "\n\n입력 impact_matrix:\n"
        + json.dumps(impact_matrix, ensure_ascii=False, indent=2)
    )
    try:
        response = _get_llm().invoke(
            prompt,
            config=tracing_config(
                agent="ITTrendAgent",
                phase="forecast_synthesis",
                prompt_version=_PROMPT_VERSION,
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        data = _safe_json_object(content)
    except Exception:
        log.exception("ITTrendAgent | _phase5_forecast_synthesis LLM 실패")
        data = {}

    forecasts_raw = data.get("forecasts") or []
    forecasts: list[dict[str, Any]] = []
    valid_horizons = {"1Q", "6M", "1Y"}
    for f in forecasts_raw:
        if not isinstance(f, dict):
            continue
        horizon = str(f.get("horizon") or "").strip()
        if horizon not in valid_horizons:
            continue
        forecasts.append(
            {
                "horizon": horizon,
                "scenario": _clip_enum(
                    f.get("scenario"), {"optimistic", "baseline", "pessimistic"}, "baseline"
                ),
                "narrative": str(f.get("narrative") or "")[:600],
                "sk_ax_impact": str(f.get("sk_ax_impact") or "")[:400],
                "drivers": [str(d)[:120] for d in (f.get("drivers") or []) if d][:6],
                "risk_level": _clip_enum(f.get("risk_level"), {"low", "medium", "high"}, "medium"),
                "recommended_response": str(f.get("recommended_response") or "")[:400],
            }
        )

    per_keyword_raw = data.get("per_keyword") or []
    per_keyword_title: dict[str, str] = {}
    per_keyword_summary: dict[str, str] = {}
    per_keyword_implication: dict[str, str] = {}
    valid_themes = {d["theme"] for d in detections}
    for item in per_keyword_raw:
        if not isinstance(item, dict):
            continue
        theme = str(item.get("theme") or "").strip().lower()
        if theme not in valid_themes:
            continue
        per_keyword_title[theme] = str(item.get("title") or "")[:200]
        per_keyword_summary[theme] = str(item.get("summary") or "")[:500]
        per_keyword_implication[theme] = str(item.get("implication") or "")[:300]

    return {
        "forecasts": forecasts,
        "final_one_liner": str(data.get("final_one_liner") or "")[:300],
        "sk_ax_implication": str(data.get("sk_ax_implication") or "")[:600],
        "confidence": _confidence_in_range(data.get("confidence", 0.6)),
        "per_keyword_title": per_keyword_title,
        "per_keyword_summary": per_keyword_summary,
        "per_keyword_implication": per_keyword_implication,
    }


# ──────────────────────────────────────────────────────────────────────────
# Persistence — global_industry_trends row 변환.
# ──────────────────────────────────────────────────────────────────────────


def _build_persistence_rows(
    *,
    batch_id: str,
    generated_at: datetime,
    detections: list[dict[str, Any]],
    alignment: dict[str, list[dict[str, Any]]],
    impact_matrix: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    per_keyword_title: dict[str, str],
    per_keyword_summary: dict[str, str],
    per_keyword_implication: dict[str, str],
    confidence: float,
    sk_ax_implication: str,
    final_one_liner: str,
) -> list[dict[str, Any]]:
    trend_date = generated_at.date()
    rows: list[dict[str, Any]] = []
    for idx, det in enumerate(detections, start=1):
        keyword = det["theme"]
        category = det.get("keyword_category") or "other"
        peers = alignment.get(keyword, [])
        aligned_peers = [p["peer_id"] for p in peers if p["alignment_type"] == "aligned"]
        evidence_card_ids = [cid for p in peers for cid in p.get("evidence_card_ids", []) if cid]
        evidence_raw_ids = _evidence_raw_ids(global_rows, keyword)
        impact_score = _impact_score_for_keyword(det, peers)
        rows.append(
            {
                "source_analysis_id": _make_source_analysis_id(batch_id, idx, keyword),
                "trend_date": trend_date,
                "industry": _industry_for_category(category),
                "region": "global",
                "keyword": keyword,
                "keyword_category": category,
                "title": per_keyword_title.get(keyword)
                or _fallback_title(keyword, det)
                or final_one_liner[:200],
                "summary": per_keyword_summary.get(keyword) or _fallback_summary(keyword, det),
                "mention_count": int(det.get("mention_count", 0) or 0),
                "impact_score": impact_score,
                "confidence": confidence,
                "related_peer_ids": aligned_peers,
                "related_card_ids": evidence_card_ids,
                "source_raw_article_ids": evidence_raw_ids,
                "sk_ax_implication": per_keyword_implication.get(keyword) or sk_ax_implication,
                "payload": {
                    "peer_alignment": peers,
                    "impact_matrix": [c for c in impact_matrix if c["trend_theme"] == keyword],
                    "forecasts": forecasts,
                    "leading_companies": det.get("leading_companies", []),
                    "intensity": det.get("intensity"),
                    "frequency_delta_pct": det.get("frequency_delta_pct"),
                    "snapshots_summary": [
                        {"company_id": s["company_id"], "card_count": s["card_count"]}
                        for s in snapshots
                    ],
                    "prompt_version": _PROMPT_VERSION,
                    "batch_id": batch_id,
                },
            }
        )
    return rows


def _evidence_raw_ids(global_rows: list[dict[str, Any]], keyword: str) -> list[int]:
    pattern = _keyword_regex(keyword)
    ids: list[int] = []
    for r in global_rows:
        haystack = ((r.get("title") or "") + " " + (r.get("content") or ""))[:2000]
        if pattern.search(haystack.lower()):
            article_id = r.get("id")
            if article_id is not None:
                try:
                    ids.append(int(article_id))
                except (TypeError, ValueError):
                    continue
        if len(ids) >= 10:
            break
    return ids


def _impact_score_for_keyword(det: dict[str, Any], peers: list[dict[str, Any]]) -> float:
    """결정적 산식 — frequency × intensity × peer_alignment_coverage."""
    mention_count = float(det.get("mention_count", 0) or 0)
    intensity_weight = {"weak": 0.3, "moderate": 0.6, "strong": 1.0}.get(
        det.get("intensity") or "weak", 0.3
    )
    aligned_count = sum(1 for p in peers if p["alignment_type"] == "aligned")
    peer_factor = 0.5 + 0.5 * (aligned_count / max(len(peers), 1)) if peers else 0.5
    raw = min(mention_count, 50.0) * 2 * intensity_weight * peer_factor
    return round(max(0.0, min(raw, 100.0)), 2)


def _industry_for_category(category: str) -> str:
    return {
        "ai_tech": "ai",
        "ai_infra": "ai_infrastructure",
        "cloud": "cloud",
        "security": "security",
        "deal": "m_and_a",
        "other": "general",
    }.get(category, "general")


def _fallback_title(keyword: str, det: dict[str, Any]) -> str:
    intensity = det.get("intensity") or "weak"
    return f"{keyword} — {intensity} 강도 글로벌 트렌드"


def _fallback_summary(keyword: str, det: dict[str, Any]) -> str:
    leading = ", ".join(det.get("leading_companies", []) or []) or "-"
    return (
        f"글로벌 6사 newsroom 에서 '{keyword}' 가 {det.get('mention_count', 0)} 건 등장. "
        f"주도 기업: {leading}. intensity={det.get('intensity')}."
    )


# ──────────────────────────────────────────────────────────────────────────
# TrendContext 빌더.
# ──────────────────────────────────────────────────────────────────────────


def _build_trend_context(
    *,
    period: str | None,
    detections: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    generated_at: datetime,
    warning: str | None,
) -> TrendContext:
    signals = [
        {
            "signal": d["theme"],
            "intensity": d.get("intensity"),
            "leading_companies": d.get("leading_companies", []),
            "mention_count": d.get("mention_count", 0),
        }
        for d in detections
    ]
    sources = [
        {
            "source_id": r.get("id"),
            "source_name": r.get("source_name"),
            "title": r.get("title"),
            "url": r.get("url"),
            "published_at": _iso(r.get("published_at") or r.get("collected_at")),
        }
        for r in global_rows[:20]
    ]
    trend_lines = [
        f"{d['theme']} ({d.get('intensity')}, mentions={d.get('mention_count')})"
        for d in detections
    ]
    return TrendContext(
        period=period,
        trend_summary=" / ".join(trend_lines[:3]),
        trend_lines=trend_lines,
        signals=signals,
        source_groups=sorted(
            {(r.get("source_name") or "") for r in global_rows if r.get("source_name")}
        ),
        sources=sources,
        reference_issue_ids=[],
        updated_at=generated_at.isoformat(timespec="seconds"),
        validation={"pass": warning is None and bool(detections), "reason": warning or ""},
        metadata={"row_count": len(global_rows), "trend_count": len(detections)},
    )


def _empty_result(
    *,
    batch_id: str,
    generated_at: datetime,
    trend_input: ITTrendInput,
    snapshots: list[dict[str, Any]],
    reasoning_steps: list[dict[str, Any]],
    warning: str,
) -> dict[str, Any]:
    context = TrendContext(
        period=trend_input.period,
        trend_summary="",
        trend_lines=[],
        signals=[],
        source_groups=trend_input.source_groups,
        sources=[],
        reference_issue_ids=_reference_issue_ids(trend_input.reference_issue_results),
        updated_at=generated_at.isoformat(timespec="seconds"),
        validation={"pass": False, "reason": warning},
        metadata={},
    )
    return {
        "agent": "ITTrendAgent",
        "prompt_version": _PROMPT_VERSION,
        "analysis_id": batch_id,
        "analysis_type": "global",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "period": trend_input.period,
        "snapshots": snapshots,
        "trend_detections": [],
        "peer_alignment": {},
        "impact_matrix": [],
        "forecasts": [],
        "final_one_liner": "",
        "sk_ax_implication": "",
        "rows": [],
        "persisted_row_count": 0,
        "reasoning_steps": reasoning_steps,
        "trend_context": context.to_dict(),
        "warning": warning,
        "validation": {"pass": False, "reason": warning},
        "confidence": 0.0,
        "metadata": trend_input.metadata,
    }


# ──────────────────────────────────────────────────────────────────────────
# Helpers — keyword extraction / classification / parsing.
# ──────────────────────────────────────────────────────────────────────────


def _extract_keywords_from_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        haystack = ((row.get("title") or "") + " " + (row.get("content") or ""))[:1500].lower()
        for keyword, _category, pattern in _TREND_KEYWORD_CANDIDATES:
            if re.search(pattern, haystack, flags=re.IGNORECASE):
                counter[keyword] += 1
    return counter


def _keyword_regex(keyword: str) -> re.Pattern[str]:
    for kw, _cat, pat in _TREND_KEYWORD_CANDIDATES:
        if kw == keyword:
            return re.compile(pat, flags=re.IGNORECASE)
    return re.compile(re.escape(keyword), flags=re.IGNORECASE)


def _keyword_category(keyword: str) -> str:
    for kw, cat, _pat in _TREND_KEYWORD_CANDIDATES:
        if kw == keyword:
            return cat
    return "other"


def _global_company_id_from_row(row: dict[str, Any]) -> str | None:
    name = (row.get("source_name") or "").strip().lower()
    for company_id in GLOBAL_COMPANY_IDS:
        if company_id in name:
            return company_id
    return None


def _is_global_newsroom_row(row: dict[str, Any]) -> bool:
    source_type = (row.get("source_type") or "").strip().lower()
    return source_type in _GLOBAL_NEWSROOM_SOURCE_TYPES


def _is_research_row(row: dict[str, Any]) -> bool:
    source_type = (row.get("source_type") or "").strip().lower()
    source_name = (row.get("source_name") or "").strip().lower()
    publisher = (row.get("publisher") or "").strip().lower()
    return source_type == "trend_report" and (
        source_name in _TREND_SOURCE_NAMES or publisher in _TREND_SOURCE_NAMES
    )


def _dedupe_items_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        item_id = item.get("id") or item.get("source_id")
        if item_id is None:
            result.append(item)
            continue
        if item_id in seen:
            continue
        seen.add(item_id)
        result.append(item)
    return result


def _latest_global_date_for_keyword(snapshots: list[dict[str, Any]], keyword: str) -> date | None:
    latest: date | None = None
    for snap in snapshots:
        for ann in snap.get("headline_announcements", []) or []:
            published = ann.get("published_at")
            if not published:
                continue
            d = _parse_iso_date(published)
            if d and (latest is None or d > latest):
                latest = d
    return latest


def _latest_date_from_cards(cards: list[dict[str, Any]]) -> date | None:
    latest: date | None = None
    for c in cards:
        value = c.get("created_at") or c.get("published_at") or c.get("collected_at")
        if not value:
            continue
        d = _parse_iso_date(value)
        if d and (latest is None or d > latest):
            latest = d
    return latest


def _parse_iso_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _safe_json_object(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("ITTrendAgent | LLM JSON parse 실패 — prefix=%s", content[:200])
        return {}
    return data if isinstance(data, dict) else {}


def _clip_enum(value: Any, choices: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in choices else default


def _confidence_in_range(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(v, 1.0))


def _summarize_alignment(alignment: dict[str, list[dict[str, Any]]]) -> str:
    if not alignment:
        return "(alignment empty)"
    parts: list[str] = []
    for keyword, peers in list(alignment.items())[:3]:
        counts = Counter(p["alignment_type"] for p in peers)
        parts.append(
            f"{keyword}: aligned={counts.get('aligned', 0)}, lagging={counts.get('lagging', 0)}, "
            f"missing={counts.get('missing', 0)}"
        )
    return " / ".join(parts)


def _summarize_impact(impact_matrix: list[dict[str, Any]]) -> str:
    if not impact_matrix:
        return "(impact empty)"
    counts = Counter(c["direction"] for c in impact_matrix)
    return ", ".join(f"{k}={v}" for k, v in counts.most_common())


# ──────────────────────────────────────────────────────────────────────────
# build_input legacy helpers — 그대로 유지.
# ──────────────────────────────────────────────────────────────────────────


def _split_trend_inputs(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trend_items: list[dict[str, Any]] = []
    candidate_reference_items: list[dict[str, Any]] = []
    unsupported_items: list[dict[str, Any]] = []
    for item in items:
        if _is_spri_bcg_trend_item(item):
            trend_items.append(dict(item))
        elif _is_global_newsroom_reference_candidate(item):
            candidate_reference_items.append(dict(item))
        else:
            unsupported_items.append(dict(item))
    return trend_items, candidate_reference_items, unsupported_items


def _is_spri_bcg_trend_item(item: dict[str, Any]) -> bool:
    source_name = str(item.get("source_name") or item.get("publisher") or "").strip().lower()
    source_type = str(item.get("source_type") or "").strip().lower()
    return source_type == "trend_report" and source_name in _TREND_SOURCE_NAMES


def _is_global_newsroom_reference_candidate(item: dict[str, Any]) -> bool:
    source_type = str(item.get("source_type") or "").strip().lower()
    return source_type in _GLOBAL_NEWSROOM_SOURCE_TYPES


def _reference_issue_results_from_items(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for item in items:
        source_type = str(item.get("source_type") or "").strip().lower()
        if source_type not in _GLOBAL_NEWSROOM_SOURCE_TYPES:
            continue
        integrated_issue = item.get("integrated_issue")
        analysis_result = item.get("analysis") or item.get("analysis_result")
        if not isinstance(integrated_issue, dict) and not isinstance(analysis_result, dict):
            continue
        references.append(
            {
                "source_id": item.get("source_id") or item.get("id"),
                "source_type": source_type,
                "integrated_issue": integrated_issue if isinstance(integrated_issue, dict) else {},
                "analysis_result": analysis_result if isinstance(analysis_result, dict) else {},
            }
        )
    return references


def _source_groups(
    *,
    explicit: list[str] | None,
    trend_items: list[dict[str, Any]],
    reference_results: list[dict[str, Any]],
) -> list[str]:
    groups = [str(item) for item in (explicit or []) if item]
    groups.extend(
        str(item.get("source_name") or item.get("publisher") or "").strip()
        for item in trend_items
        if item.get("source_name") or item.get("publisher")
    )
    if reference_results:
        groups.append("global_newsroom_references")
    return _dedupe_strings(groups)


def _reference_issue_ids(reference_issue_results: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in reference_issue_results:
        integrated = item.get("integrated_issue") if isinstance(item, dict) else {}
        analysis = item.get("analysis_result") if isinstance(item, dict) else {}
        for source in (item, integrated, analysis):
            if not isinstance(source, dict):
                continue
            value = (
                source.get("bundle_id")
                or source.get("cluster_id")
                or source.get("source_id")
                or source.get("id")
            )
            if value:
                values.append(str(value))
                break
    return _dedupe_strings(values)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


__all__ = ["ITTrendAgent", "ITTrendInput"]
