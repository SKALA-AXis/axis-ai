# 작성일: 2026-06-09
# 작성자: 최종민
# 변경이력:
#   2026-06-09 최종민 — ContextPackAssembler·주간 다이제스트 에이전트
"""ContextPackAssembler — P0 unified historical context read path.

LLM-free assembly of PeerContextPack from DB / Qdrant sources. Writers
(capability evolution, today insight, future weekly digest) stay on their
own cadence; this module is the single reader for analysis consumers.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any, Literal

from sqlalchemy import text

from src.agents.context._data_quality_checks import signal_density_label
from src.analysis.context_pack_models import (
    ContextPackCoverage,
    ExecutiveMemoryEntry,
    PeerContextPack,
)
from src.analysis.models import (
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
from src.middleware.analysis_ledger import AnalysisLedger
from src.services.peer_id_aliases import expand_peer_aliases

log = logging.getLogger(__name__)

_TIMELINE_PER_PEER_MAX_DEFAULT = 5
_CAPABILITY_NARRATIVE_CHARS_MAX_DEFAULT = 200
_SECTOR_PULSE_WEEKS_DEFAULT = 4
_FINANCIAL_QUARTERS_DEFAULT = 8
_FINANCIAL_TOP_METRICS_DEFAULT = 3
_PRECEDENT_TOP_K_DEFAULT = 5
_RAG_TOP_K_DEFAULT = 3
_EXECUTIVE_MEMORY_LIMIT_DEFAULT = 5
_LEDGER_TOP_N_DEFAULT = 5


class ContextPackAssembler:
    """Assemble PeerContextPack from all historical context sources."""

    def __init__(self, *, qdrant_search: Any | None = None) -> None:
        self._qdrant = qdrant_search

    def assemble(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        profile_context: ProfileContext | dict[str, Any] | None = None,
        integrated_issue: dict[str, Any] | None = None,
        include_executive_memory: bool = True,
        include_analysis_ledger: bool = True,
        executive_memory_limit: int = _EXECUTIVE_MEMORY_LIMIT_DEFAULT,
        ledger_top_n: int = _LEDGER_TOP_N_DEFAULT,
    ) -> PeerContextPack:
        peers = resolve_context_peers(
            input_bundle,
            profile_context,
            integrated_issue,
        )
        sectors = list(dict.fromkeys(input_bundle.sectors or []))
        provenance = ContextProvenance()

        timeline = self._query_timeline(
            peers=peers, days=90, limit_per_peer=_TIMELINE_PER_PEER_MAX_DEFAULT
        )
        if timeline:
            provenance.used_layers.append("peer_event_timeline_recent")
            provenance.timeline_card_ids = [e.card_id for e in timeline]

        capability = self._query_capability_evolution(peers=peers)
        if capability:
            provenance.used_layers.append("capability_evolution")

        sector_pulse = self._query_sector_pulse(sectors=sectors, weeks=_SECTOR_PULSE_WEEKS_DEFAULT)
        if sector_pulse:
            provenance.used_layers.append("sector_pulse_recent")
            provenance.sector_pulse_weeks = sorted({row.week_start for row in sector_pulse})

        financial = self._query_financial_trend(
            peers=peers,
            quarters=_FINANCIAL_QUARTERS_DEFAULT,
            top_metrics=_FINANCIAL_TOP_METRICS_DEFAULT,
        )
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

        precedents = self._find_precedents(
            input_bundle=input_bundle,
            peers=peers,
            min_days_since=7,
            top_k=_PRECEDENT_TOP_K_DEFAULT,
        )
        if precedents:
            provenance.used_layers.append("event_chain_candidates")
            provenance.precedent_card_ids = [p.card_id for p in precedents]

        rag = self._search_similar_cards(input_bundle, top_k=_RAG_TOP_K_DEFAULT)
        if rag:
            provenance.used_layers.append("similar_cards_rag")
            provenance.retrieved_card_ids = [r.card_id for r in rag]

        density = self._compute_evidence_density(
            peers=peers,
            timeline=timeline,
            financial=financial,
        )

        executive_memory: list[ExecutiveMemoryEntry] = []
        if include_executive_memory:
            executive_memory = self.fetch_executive_memory(
                peer_ids=peers,
                limit=executive_memory_limit,
            )
            if executive_memory:
                provenance.used_layers.append("executive_memory_recent")
                provenance.executive_memory_dates = [
                    entry.report_date for entry in executive_memory
                ]

        weekly_digest = self._query_weekly_digest(peers=peers)
        if weekly_digest:
            provenance.used_layers.append("weekly_digest_by_peer")

        ledger_by_peer: dict[str, list[dict[str, Any]]] = {}
        if include_analysis_ledger:
            ledger_by_peer = self._query_analysis_ledger(
                peer_ids=peers,
                top_n=ledger_top_n,
            )
            if ledger_by_peer:
                provenance.used_layers.append("analysis_ledger_by_peer")
                ledger_ids: list[int] = []
                for entries in ledger_by_peer.values():
                    for entry in entries:
                        entry_id = entry.get("id")
                        if entry_id is not None:
                            try:
                                ledger_ids.append(int(entry_id))
                            except (TypeError, ValueError):
                                continue
                provenance.analysis_ledger_ids = ledger_ids

        coverage = ContextPackCoverage(
            has_timeline=bool(timeline),
            has_capability_evolution=bool(capability),
            has_sector_pulse=bool(sector_pulse),
            has_financial_trend=bool(financial),
            has_event_chain=bool(precedents),
            has_similar_cards_rag=bool(rag),
            has_executive_memory=bool(executive_memory),
            has_analysis_ledger=bool(ledger_by_peer),
            has_weekly_digest=bool(weekly_digest),
            cold_start=not any(
                (
                    timeline,
                    capability,
                    sector_pulse,
                    financial,
                    precedents,
                    rag,
                    executive_memory,
                    ledger_by_peer,
                    weekly_digest,
                )
            ),
        )

        return PeerContextPack(
            peer_ids=peers,
            sectors=sectors,
            bundle_id=input_bundle.bundle_id,
            peer_event_timeline_recent=timeline,
            capability_evolution=capability,
            sector_pulse_recent=sector_pulse,
            financial_trend=financial,
            event_chain_candidates=precedents,
            similar_cards_rag=rag,
            evidence_density_per_peer=density,
            executive_memory_recent=executive_memory,
            analysis_ledger_by_peer=ledger_by_peer,
            weekly_digest_by_peer=weekly_digest,
            provenance=provenance,
            coverage=coverage,
            assembled_at=datetime.now(UTC).isoformat(),
        )

    def _query_timeline(
        self, *, peers: list[str], days: int, limit_per_peer: int
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
                               AND event_date >= (CURRENT_DATE - :days::int)
                             ORDER BY event_date DESC, importance_score DESC NULLS LAST
                             LIMIT :limit
                            """
                        ),
                        {"peer_id": peer_id, "days": int(days), "limit": int(limit_per_peer)},
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
        except Exception as exc:  # noqa: BLE001
            log.debug("context_pack timeline fallback | error=%s", exc)
        return out

    def _query_capability_evolution(self, *, peers: list[str]) -> dict[str, Any]:
        if not peers:
            return {}
        out: dict[str, Any] = {}
        try:
            with SessionLocal() as db:
                for peer_id in peers:
                    row = db.execute(
                        text(
                            """
                            SELECT peer_plus_payload->'capability_evolution' AS cap
                              FROM peer_companies
                             WHERE id = :peer_id
                            """
                        ),
                        {"peer_id": peer_id},
                    ).fetchone()
                    if row is None:
                        continue
                    cap = row._mapping.get("cap")
                    if not isinstance(cap, dict) or not cap.get("windows"):
                        continue
                    compact = compact_capability_for_context(cap)
                    if compact:
                        out[peer_id] = compact
        except Exception as exc:  # noqa: BLE001
            log.debug("context_pack capability fallback | error=%s", exc)
        return out

    def _query_sector_pulse(self, *, sectors: list[str], weeks: int) -> list[SectorPulseRow]:
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
                               DATE_TRUNC('week', CURRENT_DATE)
                               - (:weeks * 7) * INTERVAL '1 day'
                           )::date
                         ORDER BY week_start DESC
                        """
                    ),
                    {"sectors": sectors, "weeks": int(weeks)},
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            log.debug("context_pack sector_pulse fallback | error=%s", exc)
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
        self, *, peers: list[str], quarters: int, top_metrics: int
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
                             ORDER BY period_year DESC NULLS LAST,
                                      period_quarter DESC NULLS LAST,
                                      confidence DESC NULLS LAST
                             LIMIT :limit
                            """
                        ),
                        {
                            "peer_id": peer_id,
                            "limit": int(quarters * top_metrics * 2),
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
            log.debug("context_pack financial fallback | error=%s", exc)
        return out

    def _find_precedents(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        peers: list[str],
        min_days_since: int,
        top_k: int,
    ) -> list[PrecedentCandidate]:
        if self._qdrant is not None and hasattr(self._qdrant, "find_precedents"):
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
                log.debug("context_pack qdrant precedents fallback | error=%s", exc)
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
                           AND event_date <= (CURRENT_DATE - :min_days::int)
                         ORDER BY event_date DESC
                         LIMIT :limit
                        """
                    ),
                    {
                        "peers": peers,
                        "event_type": event_type,
                        "min_days": int(min_days_since),
                        "limit": int(top_k),
                    },
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            log.debug("context_pack precedent DB fallback | error=%s", exc)
            return []
        today = date.today()
        for row in rows:
            event_date_text = str(row._mapping.get("event_date") or "")
            out.append(
                PrecedentCandidate(
                    card_id=str(row._mapping.get("card_id") or ""),
                    company_id=str(row._mapping.get("company_id") or ""),
                    event_type=str(row._mapping.get("event_type") or ""),
                    days_since=_days_since(today=today, iso_date=event_date_text),
                    cosine=0.0,
                    headline=str(row._mapping.get("headline") or ""),
                )
            )
        return out

    def _search_similar_cards(
        self, input_bundle: AnalysisInputBundle, *, top_k: int
    ) -> list[RetrievedCard]:
        if self._qdrant is not None and hasattr(self._qdrant, "search_by_bundle"):
            try:
                return list(self._qdrant.search_by_bundle(input_bundle, top_k=top_k))[:top_k]
            except Exception as exc:  # noqa: BLE001
                log.debug("context_pack qdrant search fallback | error=%s", exc)
        return []

    def _compute_evidence_density(
        self,
        *,
        peers: list[str],
        timeline: list[TimelineEntry],
        financial: dict[str, FinancialSeries],
    ) -> dict[str, EvidenceDensity]:
        del financial
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
                                    AND period_year >= EXTRACT(YEAR FROM CURRENT_DATE) - 1)
                                AS signal_count,
                                (SELECT COUNT(*) FROM raw_article_financial_metrics
                                  WHERE peer_id = ANY(:aliases)
                                    AND period_year >= EXTRACT(YEAR FROM CURRENT_DATE) - 1)
                                AS metric_count
                            """
                        ),
                        {"aliases": aliases},
                    ).fetchone()
                    if rows is None:
                        continue
                    signal_counts[peer_id] = int(rows._mapping.get("signal_count") or 0)
                    metric_counts[peer_id] = int(rows._mapping.get("metric_count") or 0)
        except Exception as exc:  # noqa: BLE001
            log.debug("context_pack density fallback | error=%s", exc)

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

    def _query_weekly_digest(self, *, peers: list[str]) -> dict[str, Any]:
        if not peers:
            return {}
        out: dict[str, Any] = {}
        try:
            with SessionLocal() as db:
                for peer_id in peers:
                    row = db.execute(
                        text(
                            """
                            SELECT peer_plus_payload->'weekly_digest' AS weekly
                              FROM peer_companies
                             WHERE id = :peer_id
                            """
                        ),
                        {"peer_id": peer_id},
                    ).fetchone()
                    if row is None:
                        continue
                    weekly = row._mapping.get("weekly")
                    if not isinstance(weekly, dict):
                        continue
                    compact = _compact_weekly_digest_for_context(weekly)
                    if compact:
                        out[peer_id] = compact
        except Exception as exc:  # noqa: BLE001
            log.debug("context_pack weekly_digest fallback | error=%s", exc)
        return out

    def fetch_executive_memory(
        self,
        *,
        peer_ids: list[str],
        limit: int = _EXECUTIVE_MEMORY_LIMIT_DEFAULT,
        before_date: date | None = None,
        peer_match: Literal["overlap", "any"] = "overlap",
    ) -> list[ExecutiveMemoryEntry]:
        """Load compact executive memory from ``today_insight_reports``."""
        return self._query_executive_memory(
            peer_ids=peer_ids,
            limit=limit,
            before_date=before_date,
            peer_match=peer_match,
        )

    def fetch_broad_ledger_context(
        self,
        *,
        window_days: int = 60,
        limit: int = 8,
        min_confidence: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Recent analysis conclusions (insight/mixer/legacy) for dashboard agents."""
        return AnalysisLedger.fetch_recent(
            window_days=window_days,
            limit=limit,
            min_confidence=min_confidence,
        )

    def _query_executive_memory(
        self,
        *,
        peer_ids: list[str],
        limit: int,
        before_date: date | None = None,
        peer_match: Literal["overlap", "any"] = "overlap",
    ) -> list[ExecutiveMemoryEntry]:
        peer_set = set(peer_ids)
        params: dict[str, Any] = {"fetch_limit": max(limit * 3, limit)}
        date_filter = ""
        if before_date is not None:
            date_filter = "AND report_date < CAST(:before_date AS date)"
            params["before_date"] = before_date.isoformat()
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            f"""
                        SELECT report_date::text AS report_date,
                               headline,
                               executive_summary,
                               executive_implication,
                               output_payload,
                               peer_ids,
                               source_integrated_issue_ids,
                               source_card_ids,
                               confidence
                          FROM today_insight_reports
                         WHERE status = 'active'
                           {date_filter}
                         ORDER BY report_date DESC, created_at DESC
                         LIMIT :fetch_limit
                        """
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("context_pack executive memory fallback | error=%s", exc)
            return []

        out: list[ExecutiveMemoryEntry] = []
        for row in rows:
            row_peers = [str(item) for item in (row.get("peer_ids") or []) if item]
            if (
                peer_match == "overlap"
                and peer_set
                and row_peers
                and peer_set.isdisjoint(row_peers)
            ):
                continue
            payload = row.get("output_payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            signals = payload_dict.get("signals") or []
            signal_values = [
                str(signal.get("value") or "").strip()
                for signal in signals
                if isinstance(signal, dict) and str(signal.get("value") or "").strip()
            ][:3]
            issue_ids = [
                str(item) for item in (row.get("source_integrated_issue_ids") or []) if item
            ]
            out.append(
                ExecutiveMemoryEntry(
                    report_date=str(row.get("report_date") or ""),
                    headline=str(row.get("headline") or payload_dict.get("headline") or ""),
                    executive_summary=str(
                        row.get("executive_summary") or payload_dict.get("executive_summary") or ""
                    ),
                    executive_implication=str(
                        row.get("executive_implication")
                        or payload_dict.get("executive_implication")
                        or ""
                    ),
                    signal_values=signal_values,
                    peer_ids=row_peers,
                    source_integrated_issue_ids=issue_ids,
                    source_card_ids=[
                        str(item) for item in (row.get("source_card_ids") or []) if item
                    ],
                    confidence=_optional_float(row.get("confidence")),
                )
            )
            if len(out) >= limit:
                break
        return out

    def _query_analysis_ledger(
        self, *, peer_ids: list[str], top_n: int
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for peer_id in peer_ids:
            entries = AnalysisLedger.fetch_top_n(peer_id, top_n=top_n)
            if entries:
                out[peer_id] = entries
        return out


def resolve_context_peers(
    input_bundle: AnalysisInputBundle,
    profile_context: ProfileContext | dict[str, Any] | None,
    integrated_issue: dict[str, Any] | None = None,
) -> list[str]:
    """Resolve peer ids for context assembly."""
    candidates: list[str] = []
    if integrated_issue:
        main_company = str(integrated_issue.get("main_company") or "").strip()
        if main_company and main_company not in SELF_COMPANY_IDS:
            candidates.append(main_company)
        mentioned = integrated_issue.get("mentioned_peer_companies") or []
        if isinstance(mentioned, list):
            for company in mentioned:
                text_value = str(company or "").strip()
                if (
                    text_value
                    and text_value not in SELF_COMPANY_IDS
                    and text_value not in candidates
                ):
                    candidates.append(text_value)
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


def compact_capability_for_context(cap: dict[str, Any]) -> dict[str, Any]:
    windows = cap.get("windows") or []
    if not isinstance(windows, list):
        return {}
    compact_windows: list[dict[str, Any]] = []
    for item in windows[-8:]:
        if not isinstance(item, dict):
            continue
        narrative = str(item.get("narrative") or "")[:_CAPABILITY_NARRATIVE_CHARS_MAX_DEFAULT]
        if not narrative:
            continue
        compact_windows.append(
            {
                "period": item.get("period"),
                "business_area": item.get("business_area"),
                "narrative": narrative,
                "confidence": item.get("confidence"),
            }
        )
    if not compact_windows:
        return {}
    return {
        "version": cap.get("version"),
        "generated_at": cap.get("generated_at"),
        "windows": compact_windows,
    }


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


def _days_since(*, today: date, iso_date: str) -> int:
    if not iso_date:
        return 0
    try:
        target = date.fromisoformat(iso_date[:10])
    except ValueError:
        return 0
    return max((today - target).days, 0)


def _compact_weekly_digest_for_context(weekly: dict[str, Any]) -> dict[str, Any]:
    digests = weekly.get("digests") or []
    if not isinstance(digests, list) or not digests:
        return {}
    latest = digests[-1]
    if not isinstance(latest, dict):
        return {}
    narrative = str(latest.get("narrative") or "")[:500]
    if not narrative:
        return {}
    return {
        "version": weekly.get("version"),
        "week_iso": latest.get("week_iso"),
        "period_label": latest.get("period_label"),
        "narrative": narrative,
        "delta_vs_prev": list(latest.get("delta_vs_prev") or [])[:5],
        "strategy_label": latest.get("strategy_label"),
        "source_card_ids": list(latest.get("source_card_ids") or [])[:8],
        "confidence": latest.get("confidence"),
    }


def executive_memory_to_prior_reports(
    entries: list[ExecutiveMemoryEntry],
) -> list[dict[str, Any]]:
    """Map executive memory entries to TodayInsight prior-memory dicts."""
    return [
        {
            "report_date": entry.report_date,
            "headline": entry.headline,
            "executive_summary": entry.executive_summary,
            "signal_values": list(entry.signal_values),
            "source_integrated_issue_ids": list(entry.source_integrated_issue_ids),
            "source_card_ids": list(entry.source_card_ids),
            "confidence": entry.confidence,
        }
        for entry in entries
    ]


__all__ = [
    "ContextPackAssembler",
    "compact_capability_for_context",
    "executive_memory_to_prior_reports",
    "resolve_context_peers",
]
