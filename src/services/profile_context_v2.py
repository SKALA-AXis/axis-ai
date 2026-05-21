"""ProfileAgent 2-tier (W2-2) — Tier B: cluster-time recent enrichment.

Tier A (`peer_companies.profile_snapshot` JSONB, 주1회 CronJob) 가 build 한 정적
snapshot 위에, cluster-time 에 LLM 호출 없이 DB query 만으로 recent signals 와
financial summary 를 덧붙여 ImplicationAgent 가 받을 ProfileContext 를 만든다.

cluster-time LLM 추가 호출 0건. <50ms 목표 (peer 1명 당 2 SQL).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from src.analysis.models import ProfileContext
from src.config.company_tiers import SELF_COMPANY_IDS
from src.db.postgres import SessionLocal
from src.services.metric_canonical import METRIC_CANONICAL
from src.services.peer_id_aliases import expand_peer_aliases

log = logging.getLogger(__name__)


def build_profile_context_v2(
    *,
    companies: list[str],
    sectors: list[str] | None = None,
    event_type: str | None = None,
    lookback_days: int = 30,
    peer_profile_context: dict[str, Any] | None = None,
    skax_profile_context: dict[str, Any] | None = None,
) -> ProfileContext:
    """Tier A snapshot + Tier B recent enrichment 를 합쳐 ProfileContext 반환.

    Tier A: `peer_companies.profile_snapshot` JSONB (없으면 빈 dict fallback).
    Tier B: 최근 N일 business_signals top-3 + 최근 분기 financial_metrics.
    """
    peers_canonical = [company_id for company_id in companies if company_id not in SELF_COMPANY_IDS]
    skax_profile = dict(skax_profile_context or {}) or _load_snapshot("sk_ax")

    peer_profiles: dict[str, Any] = dict(peer_profile_context or {})
    for peer_id in peers_canonical:
        snapshot = _load_snapshot(peer_id)
        if not snapshot:
            snapshot = {"peer_id": peer_id, "company_id": peer_id}
        profile = dict(snapshot)
        profile.setdefault("peer_id", peer_id)
        profile.setdefault("company_id", peer_id)
        profile["recent_signals"] = _load_recent_business_signals(
            peer_id=peer_id, days=lookback_days, limit=3
        )
        profile["recent_financial"] = _load_latest_financial_metrics(peer_id=peer_id)
        profile["recent_capability_change"] = _summarize_capability_change(profile)
        if event_type:
            profile["event_type_focus"] = event_type
        peer_profiles[peer_id] = profile

    sector_context: dict[str, Any] = {
        "selected_sector_ids": list(dict.fromkeys(sectors or [])),
    }
    return ProfileContext(
        skax_profile=skax_profile,
        peer_profiles=peer_profiles,
        sector_context=sector_context,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DB queries
# ─────────────────────────────────────────────────────────────────────────────


def _load_snapshot(company_id: str) -> dict[str, Any]:
    """`peer_companies.profile_snapshot` JSONB 가 있으면 dict 로 반환, 아니면 빈."""
    if not company_id:
        return {}
    try:
        with SessionLocal() as db:
            # peer_companies 의 profile_snapshot 컬럼이 W2-2 V33 이후에만 존재.
            # 컬럼 없으면 NULL 로 처리되어 try/except 로 흡수.
            row = db.execute(
                text(
                    """
                    SELECT id,
                           name_ko,
                           COALESCE(profile_snapshot, '{}'::jsonb) AS profile_snapshot,
                           profile_snapshot_version,
                           profile_snapshot_generated_at
                      FROM peer_companies
                     WHERE id = :company_id
                    """
                ),
                {"company_id": company_id},
            ).fetchone()
    except Exception as exc:  # noqa: BLE001 — pre-V33 schema fallback.
        log.debug(
            "profile_snapshot 컬럼 없음 또는 조회 실패 | company=%s error=%s",
            company_id,
            exc,
        )
        return {}
    if row is None:
        return {}
    snapshot = row._mapping.get("profile_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    snapshot.setdefault("company_id", company_id)
    snapshot.setdefault("company_name_ko", row._mapping.get("name_ko"))
    snapshot.setdefault("profile_snapshot_version", row._mapping.get("profile_snapshot_version"))
    snapshot.setdefault(
        "profile_snapshot_generated_at",
        _iso(row._mapping.get("profile_snapshot_generated_at")),
    )
    return snapshot


def _load_recent_business_signals(*, peer_id: str, days: int, limit: int) -> list[dict[str, Any]]:
    aliases = expand_peer_aliases(peer_id)
    try:
        with SessionLocal() as db:
            rows = db.execute(
                text(
                    """
                    SELECT id,
                           peer_id,
                           business_area,
                           signal_type,
                           sentiment,
                           summary,
                           evidence_text,
                           confidence,
                           period_year,
                           period_quarter,
                           created_at
                      FROM raw_article_business_signals
                     WHERE peer_id = ANY(:aliases)
                       AND created_at >= NOW() - (:days || ' days')::interval
                     ORDER BY confidence DESC NULLS LAST, created_at DESC
                     LIMIT :limit
                    """
                ),
                {"aliases": aliases, "days": int(days), "limit": int(limit)},
            ).fetchall()
    except Exception as exc:  # noqa: BLE001 — env 미구성 fallback.
        log.debug("_load_recent_business_signals fallback | peer=%s error=%s", peer_id, exc)
        return []
    return [
        {
            "signal_id": str(row._mapping.get("id")),
            "business_area": row._mapping.get("business_area"),
            "signal_type": row._mapping.get("signal_type"),
            "sentiment": row._mapping.get("sentiment"),
            "summary": row._mapping.get("summary"),
            "evidence_text": row._mapping.get("evidence_text"),
            "confidence": _safe_float(row._mapping.get("confidence")),
            "period_year": row._mapping.get("period_year"),
            "period_quarter": row._mapping.get("period_quarter"),
            "created_at": _iso(row._mapping.get("created_at")),
        }
        for row in rows
    ]


def _load_latest_financial_metrics(*, peer_id: str) -> list[dict[str, Any]]:
    aliases = expand_peer_aliases(peer_id)
    canonical_keys = sorted(set(METRIC_CANONICAL.values()))
    try:
        with SessionLocal() as db:
            rows = db.execute(
                text(
                    """
                    SELECT peer_id,
                           metric_name,
                           metric_label,
                           value_numeric,
                           value_krwbn,
                           unit,
                           period_year,
                           period_quarter,
                           period,
                           raw_article_id
                      FROM raw_article_financial_metrics
                     WHERE peer_id = ANY(:aliases)
                       AND period_year IS NOT NULL
                     ORDER BY period_year DESC NULLS LAST,
                              period_quarter DESC NULLS LAST,
                              confidence DESC NULLS LAST
                     LIMIT 12
                    """
                ),
                {"aliases": aliases},
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        log.debug("_load_latest_financial_metrics fallback | peer=%s error=%s", peer_id, exc)
        return []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        canonical = METRIC_CANONICAL.get(
            str(row._mapping.get("metric_name") or ""),
            str(row._mapping.get("metric_name") or ""),
        )
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(
            {
                "metric_name_canonical": canonical,
                "metric_name_raw": row._mapping.get("metric_name"),
                "metric_label": row._mapping.get("metric_label"),
                "value_numeric": _safe_float(row._mapping.get("value_numeric")),
                "value_krwbn": _safe_float(row._mapping.get("value_krwbn")),
                "unit": row._mapping.get("unit"),
                "period_year": row._mapping.get("period_year"),
                "period_quarter": row._mapping.get("period_quarter"),
                "period": row._mapping.get("period"),
                "evidence_article_id": row._mapping.get("raw_article_id"),
            }
        )
        if len(out) >= len(canonical_keys):
            break
    return out


def _summarize_capability_change(profile: dict[str, Any]) -> str | None:
    """capability_evolution JSONB 의 가장 최근 window narrative 반환."""
    capability = profile.get("capability_evolution") or {}
    if not isinstance(capability, dict):
        return None
    windows = capability.get("windows") or []
    if not isinstance(windows, list) or not windows:
        return None
    # 가장 최근 generated_at 우선.
    sorted_windows = sorted(
        (w for w in windows if isinstance(w, dict)),
        key=lambda w: str(w.get("generated_at") or w.get("period") or ""),
        reverse=True,
    )
    if not sorted_windows:
        return None
    narrative = sorted_windows[0].get("narrative")
    if isinstance(narrative, str) and narrative.strip():
        return narrative.strip()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = ["build_profile_context_v2"]
