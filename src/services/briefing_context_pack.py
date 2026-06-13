"""Historical context assembly for BriefingGenerationAgent."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from src.analysis.models import AnalysisInputBundle
from src.config.company_tiers import SELF_COMPANY_IDS
from src.rag.precedent_search import QdrantPrecedentSearch
from src.services.context_pack_assembler import ContextPackAssembler

BriefingType = Literal["daily", "weekly", "monthly"]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def peers_from_cards(
    selected_cards: list[dict[str, Any]],
    *,
    peer_ids: list[str] | None = None,
) -> list[str]:
    peers: list[str] = []
    if peer_ids:
        peers.extend([p for p in peer_ids if p and p not in SELF_COMPANY_IDS])
    for card in selected_cards:
        for key in ("peer_id", "peer_company_id", "company"):
            value = str(card.get(key) or "").strip()
            if value and value not in SELF_COMPANY_IDS and value not in peers:
                peers.append(value)
    return peers


def sectors_from_cards(
    selected_cards: list[dict[str, Any]],
    *,
    sectors: list[str] | None = None,
) -> list[str]:
    found: list[str] = []
    if sectors:
        found.extend([s for s in sectors if s])
    for card in selected_cards:
        for key in ("primary_keyword_category", "sector", "category"):
            value = str(card.get(key) or "").strip()
            if value and value not in found:
                found.append(value)
    return found


def assemble_briefing_historical_context(
    *,
    selected_cards: list[dict[str, Any]],
    period: dict[str, Any],
    briefing_type: BriefingType,
    peer_ids: list[str] | None = None,
    sectors: list[str] | None = None,
) -> dict[str, Any]:
    """Read PeerContextPack layers for briefing synthesis and comparison."""
    peers = peers_from_cards(selected_cards, peer_ids=peer_ids)
    sector_list = sectors_from_cards(selected_cards, sectors=sectors)
    date_from: date = period["date_from"]
    bundle = AnalysisInputBundle(
        bundle_id=f"briefing:{briefing_type}:{date_from.isoformat()}",
        cluster_id="briefing",
        source_type="briefing",
        companies=peers,
        sectors=sector_list,
        event_type="briefing",
        items=[],
        facts=[],
        evidence_snippets=[],
        sources=[],
        metadata={"briefing_type": briefing_type},
    )
    main_company = peers[0] if peers else None
    pack = ContextPackAssembler(qdrant_search=QdrantPrecedentSearch()).assemble(
        input_bundle=bundle,
        integrated_issue={"main_company": main_company} if main_company else None,
        include_executive_memory=True,
        include_analysis_ledger=True,
        executive_memory_limit=5,
        ledger_top_n=5,
    )
    return _compact_pack_for_briefing(pack, briefing_type=briefing_type)


def _compact_pack_for_briefing(pack: Any, *, briefing_type: BriefingType) -> dict[str, Any]:
    timeline = [
        {
            "company_id": entry.company_id,
            "event_date": entry.event_date,
            "headline": (entry.headline or "")[:120],
            "event_type": entry.event_type,
        }
        for entry in pack.peer_event_timeline_recent[:6]
    ]
    capability: dict[str, Any] = {}
    for peer_id, payload in (pack.capability_evolution or {}).items():
        if not isinstance(payload, dict):
            continue
        narrative = str(payload.get("narrative") or payload.get("summary") or "")[:300]
        if narrative:
            capability[str(peer_id)] = {
                "narrative": narrative,
                "strategy_label": payload.get("strategy_label"),
            }
    executive_memory = list(pack.executive_memory_recent or [])[:5]
    if executive_memory and hasattr(executive_memory[0], "to_dict"):
        executive_memory = [entry.to_dict() for entry in executive_memory]

    ledger_by_peer: dict[str, list[dict[str, Any]]] = {}
    for peer_id, entries in (pack.analysis_ledger_by_peer or {}).items():
        compact_entries: list[dict[str, Any]] = []
        for entry in entries[:3]:
            if not isinstance(entry, dict):
                continue
            line = str(entry.get("conclusion_one_liner") or "").strip()
            if not line:
                continue
            compact_entries.append(
                {
                    "analysis_type": entry.get("analysis_type"),
                    "conclusion_one_liner": line[:200],
                    "confidence": entry.get("confidence"),
                    "created_at": entry.get("created_at"),
                    "source": entry.get("source"),
                }
            )
        if compact_entries:
            ledger_by_peer[str(peer_id)] = compact_entries

    sector_pulse = [row.to_dict() for row in pack.sector_pulse_recent[:4]]

    return {
        "briefing_type": briefing_type,
        "peer_ids": list(pack.peer_ids),
        "sectors": list(pack.sectors),
        "used_layers": list(pack.provenance.used_layers),
        "coverage": pack.coverage.to_dict(),
        "weekly_digest_by_peer": dict(pack.weekly_digest_by_peer or {}),
        "capability_evolution": capability,
        "executive_memory_recent": executive_memory,
        "analysis_ledger_by_peer": ledger_by_peer,
        "peer_event_timeline_recent": timeline,
        "sector_pulse_recent": sector_pulse,
    }


def merge_historical_context_into_basis(
    briefing_basis: dict[str, Any],
    historical: dict[str, Any] | None,
) -> dict[str, Any]:
    """Enrich deterministic basis blocks with stored narratives for LLM + UI."""
    if not historical:
        return briefing_basis

    briefing_basis = dict(briefing_basis)
    briefing_basis["historical_context"] = historical
    comparison_notes: list[str] = []

    for peer_id, digest in (historical.get("weekly_digest_by_peer") or {}).items():
        if not isinstance(digest, dict):
            continue
        deltas = [str(d) for d in (digest.get("delta_vs_prev") or [])[:3] if str(d).strip()]
        if deltas:
            comparison_notes.append(f"{peer_id} 지난주 대비: {', '.join(deltas)}")
        narrative = str(digest.get("narrative") or "").strip()
        if narrative:
            comparison_notes.append(f"{peer_id} 주간 narrative: {narrative[:160]}")

    for entry in historical.get("executive_memory_recent") or []:
        if not isinstance(entry, dict):
            continue
        headline = str(entry.get("headline") or entry.get("summary") or "").strip()
        report_date = entry.get("report_date") or entry.get("created_at")
        if headline:
            comparison_notes.append(f"임원 인사이트({report_date}): {headline[:120]}")

    for peer_id, entries in (historical.get("analysis_ledger_by_peer") or {}).items():
        for entry in entries[:2]:
            if not isinstance(entry, dict):
                continue
            line = str(entry.get("conclusion_one_liner") or "").strip()
            if line:
                comparison_notes.append(f"과거 분석({peer_id}): {line[:120]}")

    for peer_id, cap in (historical.get("capability_evolution") or {}).items():
        if not isinstance(cap, dict):
            continue
        narrative = str(cap.get("narrative") or "").strip()
        if narrative:
            comparison_notes.append(f"역량 추이({peer_id}): {narrative[:120]}")

    comparison_notes = _dedupe(comparison_notes)
    if comparison_notes:
        cp = dict(briefing_basis.get("comparison_point") or {})
        prior_rationale = str(cp.get("rationale") or "").strip()
        merged_rationale = " ".join(comparison_notes[:4])
        if prior_rationale:
            merged_rationale = f"{prior_rationale} {merged_rationale}"
        cp["rationale"] = merged_rationale[:500]
        cp["historical_signals"] = comparison_notes[:6]
        briefing_basis["comparison_point"] = cp

        core = dict(briefing_basis.get("core_change") or {})
        if not core.get("finding"):
            core["finding"] = comparison_notes[0][:220]
        core["rationale"] = cp["rationale"]
        briefing_basis["core_change"] = core

    return briefing_basis


__all__ = [
    "assemble_briefing_historical_context",
    "merge_historical_context_into_basis",
    "peers_from_cards",
    "sectors_from_cards",
]
