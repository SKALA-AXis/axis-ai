"""AnalysisContextBuilder — W4-2.

4-Layer Context Model 의 Layer 4 active context 를 cluster-time 에 합성한다.
LLM 호출 X. DB query + Qdrant retrieve 만. token budget ≤ 4,000 으로 압축.

설계: design/01-analysis-pipeline-implementation-plan.md §3.4.5.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import text

from src.agents.context._data_quality_checks import (
    signal_density_label,
)
from src.analysis.models import (
    AnalysisContext,
    AnalysisInputBundle,
    ContextProvenance,
    EvidenceDensity,
    FinancialSeries,
    FinancialSeriesPoint,
    PrecedentCandidate,
    ProfileContext,
    RetrievedCard,
    SectorPulseRow,
    TimelineEntry,
)
from src.config.company_tiers import SELF_COMPANY_IDS
from src.db.postgres import SessionLocal
from src.services.peer_id_aliases import expand_peer_aliases

log = logging.getLogger(__name__)


TOKEN_BUDGET_DEFAULT = 4000
# Rough char→token ratio for Korean+ENG mix (gpt-4o tokenizer 실측 평균).
_CHAR_PER_TOKEN = 2.5

# Per-layer compression defaults (§3.4.5 priority).
_TIMELINE_PER_PEER_MAX_DEFAULT = 5
_CAPABILITY_NARRATIVE_CHARS_MAX_DEFAULT = 200
_SECTOR_PULSE_WEEKS_DEFAULT = 4
_FINANCIAL_QUARTERS_DEFAULT = 8
_FINANCIAL_TOP_METRICS_DEFAULT = 3
_PRECEDENT_TOP_K_DEFAULT = 5
_RAG_TOP_K_DEFAULT = 3


class AnalysisContextBuilder:
    """LLM 호출 없이 4-Layer context 를 합성하고 token budget 으로 압축."""

    def __init__(
        self,
        *,
        token_budget: int = TOKEN_BUDGET_DEFAULT,
        qdrant_search: Any | None = None,
    ) -> None:
        self._budget = token_budget
        self._qdrant = qdrant_search  # callable / object, optional (none-safe)

    def build(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        profile_context: ProfileContext | dict[str, Any] | None = None,
        integrated_issue: dict[str, Any] | None = None,
        as_of: date | None = None,
    ) -> AnalysisContext:
        # IntegratedIssue 가 우선 — main_company / mentioned_peer_companies 가
        # 확정되면 그것을 peer 입력으로 쓴다. 없으면 input_bundle.companies fallback.
        peers = _resolve_peers(input_bundle, profile_context, integrated_issue)
        sectors = list(dict.fromkeys(input_bundle.sectors or []))
        ctx = AnalysisContext()
        provenance = ContextProvenance()
        # point-in-time 기준일 — None 이면 오늘(기존 now-relative 동작). 설정 시 모든
        # 과거-맥락 쿼리가 anchor 이하로 클램프되어 백필 시 미래 데이터 누출(look-ahead) 방지.
        anchor = as_of or date.today()

        # Layer 2-A timeline -----------------------------------------------
        timeline = self._query_timeline(
            peers=peers, days=90, limit_per_peer=_TIMELINE_PER_PEER_MAX_DEFAULT, anchor=anchor
        )
        ctx.peer_event_timeline_recent = timeline
        provenance.timeline_card_ids = [e.card_id for e in timeline]
        if timeline:
            provenance.used_layers.append("peer_event_timeline_recent")

        # Layer 2-B sector pulse -------------------------------------------
        sector_pulse = self._query_sector_pulse(
            sectors=sectors, weeks=_SECTOR_PULSE_WEEKS_DEFAULT, anchor=anchor
        )
        ctx.sector_pulse_recent = sector_pulse
        if sector_pulse:
            provenance.used_layers.append("sector_pulse_recent")
        provenance.sector_pulse_weeks = sorted({row.week_start for row in sector_pulse})

        # Layer 2-D financial trend ----------------------------------------
        financial = self._query_financial_trend(
            peers=peers,
            quarters=_FINANCIAL_QUARTERS_DEFAULT,
            top_metrics=_FINANCIAL_TOP_METRICS_DEFAULT,
            anchor=anchor,
        )
        ctx.financial_trend = financial
        if financial:
            provenance.used_layers.append("financial_trend")
        provenance.financial_article_ids = sorted(
            {
                point.raw_article_id
                for series in financial.values()
                for point in series.points
                if point.raw_article_id is not None
            }
        )

        # Layer 2-E event chain candidates ---------------------------------
        precedents = self._find_precedents(
            input_bundle=input_bundle,
            peers=peers,
            sectors=sectors,
            min_days_since=7,
            top_k=_PRECEDENT_TOP_K_DEFAULT,
            anchor=anchor,
            as_of=as_of,
        )
        ctx.event_chain_candidates = precedents
        if precedents:
            provenance.used_layers.append("event_chain_candidates")
        provenance.precedent_card_ids = [p.card_id for p in precedents]

        # Layer 3 RAG -----------------------------------------------------
        rag = self._search_similar_cards(input_bundle, top_k=_RAG_TOP_K_DEFAULT, as_of=as_of)
        ctx.similar_cards_rag = rag
        if rag:
            provenance.used_layers.append("similar_cards_rag")
        provenance.retrieved_card_ids = [r.card_id for r in rag]

        # Density per peer -------------------------------------------------
        density = self._compute_evidence_density(
            peers=peers,
            timeline=timeline,
            financial=financial,
            anchor=anchor,
        )
        ctx.evidence_density_per_peer = density

        ctx.provenance = provenance
        ctx = _compress_to_budget(ctx, budget=self._budget)
        ctx.token_budget_used = _estimate_token_count(ctx)
        return ctx

    # ──────────────────────────────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────────────────────────────
    def _query_timeline(
        self, *, peers: list[str], days: int, limit_per_peer: int, anchor: date
    ) -> list[TimelineEntry]:
        if not peers:
            return []
        out: list[TimelineEntry] = []
        try:
            with SessionLocal() as db:
                for peer_id in peers:
                    rows = db.execute(
                        text(
                            """
                            SELECT company_id, event_date, card_id, event_type,
                                   sector, headline, importance, importance_score
                              FROM peer_event_timeline
                             WHERE company_id = :peer_id
                               AND event_date >= (:anchor::date - :days::int)
                               AND event_date <= :anchor::date
                             ORDER BY event_date DESC, importance_score DESC NULLS LAST
                             LIMIT :limit
                            """
                        ),
                        {
                            "peer_id": peer_id,
                            "days": int(days),
                            "limit": int(limit_per_peer),
                            "anchor": anchor,
                        },
                    ).fetchall()
                    for row in rows:
                        out.append(
                            TimelineEntry(
                                company_id=str(row._mapping.get("company_id") or peer_id),
                                event_date=str(row._mapping.get("event_date") or ""),
                                card_id=str(row._mapping.get("card_id") or ""),
                                event_type=str(row._mapping.get("event_type") or ""),
                                sector=str(row._mapping.get("sector") or ""),
                                headline=str(row._mapping.get("headline") or ""),
                                importance=str(row._mapping.get("importance") or ""),
                                importance_score=_safe_float(
                                    row._mapping.get("importance_score"), 0.0
                                ),
                            )
                        )
        except Exception as exc:  # noqa: BLE001 — VIEW 미존재 fallback (pre-V33).
            log.debug("query_timeline fallback | error=%s", exc)
            return []
        return out

    def _query_sector_pulse(
        self, *, sectors: list[str], weeks: int, anchor: date
    ) -> list[SectorPulseRow]:
        if not sectors:
            return []
        try:
            with SessionLocal() as db:
                rows = db.execute(
                    text(
                        """
                        SELECT sector, week_start, event_count, peer_event_count,
                               general_event_count, intensity_avg,
                               event_type_distribution, active_peers, notable_card_ids
                          FROM sector_pulse
                         WHERE sector = ANY(:sectors)
                           AND week_start >= (
                               DATE_TRUNC('week', :anchor::date)
                               - (:weeks * 7) * INTERVAL '1 day'
                           )::date
                           AND week_start <= :anchor::date
                         ORDER BY week_start DESC
                        """
                    ),
                    {"sectors": sectors, "weeks": int(weeks), "anchor": anchor},
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            log.debug("query_sector_pulse fallback | error=%s", exc)
            return []
        out: list[SectorPulseRow] = []
        for row in rows:
            dist = row._mapping.get("event_type_distribution") or {}
            if not isinstance(dist, dict):
                dist = {}
            active = row._mapping.get("active_peers") or []
            notable = row._mapping.get("notable_card_ids") or []
            out.append(
                SectorPulseRow(
                    sector=str(row._mapping.get("sector") or ""),
                    week_start=str(row._mapping.get("week_start") or ""),
                    event_count=int(row._mapping.get("event_count") or 0),
                    peer_event_count=int(row._mapping.get("peer_event_count") or 0),
                    general_event_count=int(row._mapping.get("general_event_count") or 0),
                    intensity_avg=_safe_float(row._mapping.get("intensity_avg"), 0.0),
                    event_type_distribution={str(k): int(v) for k, v in dist.items()},
                    active_peers=[str(item) for item in active if item],
                    notable_card_ids=[str(item) for item in notable if item],
                )
            )
        return out

    def _query_financial_trend(
        self, *, peers: list[str], quarters: int, top_metrics: int, anchor: date
    ) -> dict[str, FinancialSeries]:
        if not peers:
            return {}
        out: dict[str, FinancialSeries] = {}
        try:
            with SessionLocal() as db:
                for peer_id in peers:
                    rows = db.execute(
                        text(
                            """
                            SELECT metric_name_canonical,
                                   period_year,
                                   period_quarter_safe,
                                   value_numeric,
                                   value_krwbn,
                                   unit,
                                   evidence_article_id
                              FROM peer_financial_trend
                             WHERE company_id = :peer_id
                               AND (
                                   evidence_article_id IS NULL
                                   OR evidence_article_id IN (
                                       SELECT id FROM raw_articles
                                        WHERE published_at < (:anchor::date + INTERVAL '1 day')
                                   )
                               )
                             ORDER BY period_year DESC NULLS LAST,
                                      period_quarter DESC NULLS LAST,
                                      confidence DESC NULLS LAST
                             LIMIT :limit
                            """
                        ),
                        {
                            "peer_id": peer_id,
                            "limit": int(quarters * top_metrics * 2),
                            "anchor": anchor,
                        },
                    ).fetchall()
                    series_map: dict[str, list[FinancialSeriesPoint]] = {}
                    for row in rows:
                        canonical = str(row._mapping.get("metric_name_canonical") or "")
                        if not canonical:
                            continue
                        points = series_map.setdefault(canonical, [])
                        if len(points) >= quarters:
                            continue
                        points.append(
                            FinancialSeriesPoint(
                                period_year=int(row._mapping.get("period_year") or 0),
                                period_quarter=str(
                                    row._mapping.get("period_quarter_safe") or "annual"
                                ),
                                value_numeric=_optional_float(row._mapping.get("value_numeric")),
                                value_krwbn=_optional_float(row._mapping.get("value_krwbn")),
                                unit=str(row._mapping.get("unit") or ""),
                                raw_article_id=_optional_int(
                                    row._mapping.get("evidence_article_id")
                                ),
                            )
                        )
                    selected = list(series_map.items())[:top_metrics]
                    for metric_name, points in selected:
                        if not points:
                            continue
                        out[f"{peer_id}::{metric_name}"] = FinancialSeries(
                            company_id=peer_id,
                            metric_name_canonical=metric_name,
                            points=points,
                        )
        except Exception as exc:  # noqa: BLE001
            log.debug("query_financial_trend fallback | error=%s", exc)
            return {}
        return out

    def _find_precedents(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        peers: list[str],
        sectors: list[str],
        min_days_since: int,
        top_k: int,
        anchor: date,
        as_of: date | None = None,
    ) -> list[PrecedentCandidate]:
        """Qdrant embedding 우선, 없으면 DB-only (peer + event_type + 7일 이상) fallback.

        as_of 가 설정된 point-in-time(백필) 모드에선 Qdrant 경로(발행일 상한 미보장)를
        건너뛰고 anchor 로 클램프된 DB fallback 만 쓴다 — look-ahead 방지.
        """
        if as_of is None and self._qdrant is not None and hasattr(self._qdrant, "find_precedents"):
            try:
                return list(
                    self._qdrant.find_precedents(
                        bundle=input_bundle,
                        peers=peers,
                        min_days_since=min_days_since,
                        top_k=top_k,
                    )
                )[:top_k]
            except Exception as exc:  # noqa: BLE001
                log.debug("qdrant find_precedents fallback | error=%s", exc)
        # DB-only fallback (P3-LOG-3 — 7일 이상 시간차로 duplicate 회피).
        event_type = input_bundle.event_type or ""
        if not event_type or not peers:
            return []
        out: list[PrecedentCandidate] = []
        try:
            with SessionLocal() as db:
                rows = db.execute(
                    text(
                        """
                        SELECT card_id, company_id, event_type, headline, event_date
                          FROM peer_event_timeline
                         WHERE company_id = ANY(:peers)
                           AND event_type = :event_type
                           AND event_date <= (:anchor::date - :min_days::int)
                         ORDER BY event_date DESC
                         LIMIT :limit
                        """
                    ),
                    {
                        "peers": peers,
                        "event_type": event_type,
                        "min_days": int(min_days_since),
                        "limit": int(top_k),
                        "anchor": anchor,
                    },
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            log.debug("precedent fallback DB query failed | error=%s", exc)
            return []
        today = anchor
        for row in rows:
            event_date_text = str(row._mapping.get("event_date") or "")
            days_since = _days_since(today=today, iso_date=event_date_text)
            out.append(
                PrecedentCandidate(
                    card_id=str(row._mapping.get("card_id") or ""),
                    company_id=str(row._mapping.get("company_id") or ""),
                    event_type=str(row._mapping.get("event_type") or ""),
                    days_since=days_since,
                    cosine=0.0,
                    headline=str(row._mapping.get("headline") or ""),
                )
            )
        return out

    def _search_similar_cards(
        self, input_bundle: AnalysisInputBundle, *, top_k: int, as_of: date | None = None
    ) -> list[RetrievedCard]:
        # point-in-time(백필) 모드: Qdrant 가 발행일 상한 필터를 보장하지 못하므로
        # as_of 이후 카드 누출(look-ahead) 방지를 위해 RAG 를 생략한다.
        if as_of is not None:
            return []
        if self._qdrant is not None and hasattr(self._qdrant, "search_by_bundle"):
            try:
                return list(self._qdrant.search_by_bundle(input_bundle, top_k=top_k))[:top_k]
            except Exception as exc:  # noqa: BLE001
                log.debug("qdrant search_by_bundle fallback | error=%s", exc)
        return []

    def _compute_evidence_density(
        self,
        *,
        peers: list[str],
        timeline: list[TimelineEntry],
        financial: dict[str, FinancialSeries],
        anchor: date,
    ) -> dict[str, EvidenceDensity]:
        out: dict[str, EvidenceDensity] = {}
        timeline_by_peer: dict[str, int] = {}
        for entry in timeline:
            timeline_by_peer[entry.company_id] = timeline_by_peer.get(entry.company_id, 0) + 1
        signal_counts: dict[str, int] = {}
        metric_counts: dict[str, int] = {}
        try:
            with SessionLocal() as db:
                for peer_id in peers:
                    aliases = expand_peer_aliases(peer_id)
                    rows = db.execute(
                        text(
                            """
                            SELECT
                                (SELECT COUNT(*) FROM raw_article_business_signals
                                  WHERE peer_id = ANY(:aliases)
                                    AND period_year >= EXTRACT(YEAR FROM :anchor::date) - 1)
                                AS signal_count,
                                (SELECT COUNT(*) FROM raw_article_financial_metrics
                                  WHERE peer_id = ANY(:aliases)
                                    AND period_year >= EXTRACT(YEAR FROM :anchor::date) - 1)
                                AS metric_count
                            """
                        ),
                        {"aliases": aliases, "anchor": anchor},
                    ).fetchone()
                    if rows is None:
                        continue
                    signal_counts[peer_id] = int(rows._mapping.get("signal_count") or 0)
                    metric_counts[peer_id] = int(rows._mapping.get("metric_count") or 0)
        except Exception as exc:  # noqa: BLE001
            log.debug("density count fallback | error=%s", exc)

        for peer_id in peers:
            signal_count = signal_counts.get(peer_id, 0)
            metric_count = metric_counts.get(peer_id, 0)
            density = signal_density_label(
                signal_count_4q=signal_count,
                metric_count_4q=metric_count,
            )
            out[peer_id] = EvidenceDensity(
                signal_count_4q=signal_count,
                metric_count_4q=metric_count,
                timeline_count_90d=timeline_by_peer.get(peer_id, 0),
                density_label=density,  # type: ignore[arg-type]
            )
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_peers(
    input_bundle: AnalysisInputBundle,
    profile_context: ProfileContext | dict[str, Any] | None,
    integrated_issue: dict[str, Any] | None = None,
) -> list[str]:
    """우선순위: integrated_issue.main_company > mentioned_peer_companies >
    profile_context.peer_profiles > input_bundle.companies.

    SELF_COMPANY_IDS (sk_ax 등) 은 peer 후보에서 제외.
    """
    candidates: list[str] = []
    if integrated_issue:
        main_company = str(integrated_issue.get("main_company") or "").strip()
        if main_company and main_company not in SELF_COMPANY_IDS:
            candidates.append(main_company)
        mentioned = integrated_issue.get("mentioned_peer_companies") or []
        if isinstance(mentioned, list):
            for company in mentioned:
                text = str(company or "").strip()
                if text and text not in SELF_COMPANY_IDS and text not in candidates:
                    candidates.append(text)
    if isinstance(profile_context, ProfileContext):
        for peer_id in profile_context.peer_profiles or {}:
            if peer_id and peer_id not in SELF_COMPANY_IDS and peer_id not in candidates:
                candidates.append(peer_id)
    elif isinstance(profile_context, dict):
        peers = profile_context.get("peer_profiles") or {}
        if isinstance(peers, dict):
            for peer_id in peers:
                if peer_id and peer_id not in SELF_COMPANY_IDS and peer_id not in candidates:
                    candidates.append(peer_id)
    for company in input_bundle.companies or []:
        if company and company not in SELF_COMPANY_IDS and company not in candidates:
            candidates.append(company)
    return candidates


def _compress_to_budget(ctx: AnalysisContext, *, budget: int) -> AnalysisContext:
    """§3.4.5 compression 우선순위.

    1) RAG → 2개로 축소
    2) financial → 4 분기로 축소
    3) sector_pulse → 2주로 축소
    4) event_chain_candidates → 3건으로 축소
    5) timeline → peer 당 3건으로 축소
    6) no-op
    """
    actions = [
        lambda c: _shrink_rag(c, max_items=2),
        lambda c: _shrink_financial(c, max_quarters=4),
        lambda c: _shrink_sector_pulse(c, max_weeks=2),
        lambda c: _shrink_event_chain(c, max_items=3),
        lambda c: _shrink_timeline_per_peer(c, max_per_peer=3),
    ]
    for action in actions:
        if _estimate_token_count(ctx) <= budget:
            return ctx
        ctx = action(ctx)
    return ctx


def _shrink_rag(ctx: AnalysisContext, *, max_items: int) -> AnalysisContext:
    ctx.similar_cards_rag = ctx.similar_cards_rag[:max_items]
    return ctx


def _shrink_financial(ctx: AnalysisContext, *, max_quarters: int) -> AnalysisContext:
    for series in ctx.financial_trend.values():
        series.points = series.points[:max_quarters]
    return ctx


def _shrink_sector_pulse(ctx: AnalysisContext, *, max_weeks: int) -> AnalysisContext:
    ctx.sector_pulse_recent = ctx.sector_pulse_recent[:max_weeks]
    return ctx


def _shrink_event_chain(ctx: AnalysisContext, *, max_items: int) -> AnalysisContext:
    ctx.event_chain_candidates = ctx.event_chain_candidates[:max_items]
    return ctx


def _shrink_timeline_per_peer(ctx: AnalysisContext, *, max_per_peer: int) -> AnalysisContext:
    per_peer: dict[str, list[TimelineEntry]] = {}
    for entry in ctx.peer_event_timeline_recent:
        per_peer.setdefault(entry.company_id, []).append(entry)
    compressed: list[TimelineEntry] = []
    for peer_id, entries in per_peer.items():
        compressed.extend(entries[:max_per_peer])
    ctx.peer_event_timeline_recent = compressed
    return ctx


def _estimate_token_count(ctx: AnalysisContext) -> int:
    import json

    try:
        text_payload = json.dumps(ctx.to_dict(), ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return 0
    return int(len(text_payload) / _CHAR_PER_TOKEN)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _days_since(*, today: Any, iso_date: str) -> int:
    from datetime import date

    if not iso_date:
        return 0
    try:
        target = date.fromisoformat(iso_date[:10])
    except ValueError:
        return 0
    delta = today - target if isinstance(today, date) else None
    if delta is None:
        return 0
    return max(delta.days, 0)


__all__ = [
    "AnalysisContextBuilder",
    "TOKEN_BUDGET_DEFAULT",
]
