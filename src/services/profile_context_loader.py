# 작성일: 2026-05-26
# 작성자: 박지원
# 변경이력:
#   2026-05-26 박지원 — PeerProfile Agent 최초 구현 및 수정
#   2026-06-02 심유정 — 전략 시사점 에이전트 구현, 사용자 전략 오버레이 추가 및 안정화
#   2026-06-16 최종민 — 백필 시 snapshot 시점-민감 필드 strip(룩어헤드 누수 차단)
"""ProfileContextLoader — cluster-time profile context enrichment.

Tier A (`peer_companies.profile_snapshot` JSONB, 분기 1회 CronJob) 가 build 한 정적
snapshot 위에, cluster-time 에 LLM 호출 없이 DB query 만으로 recent signals 와
financial summary 를 덧붙여 ImplicationAgent 가 받을 ProfileContext 를 만든다.

cluster-time LLM 추가 호출 0건. <50ms 목표 (peer 1명 당 2 SQL).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date
from typing import Any

from sqlalchemy import text

from src.analysis.models import ProfileContext
from src.config.company_tiers import SELF_COMPANY_IDS
from src.db.postgres import SessionLocal
from src.services.metric_canonical import METRIC_CANONICAL
from src.services.peer_id_aliases import expand_peer_aliases

log = logging.getLogger(__name__)

# 백필 point-in-time 시 snapshot 에서 제거할 시점-민감/미래지향 필드.
# snapshot 은 peer 당 단일 현재본이라 과거 카드엔 look-ahead — 안정 프로필
# (business_areas·core_capabilities·key_products_services·company_summary 등)만 남기고
# 아래는 제거한다. (시점 정합 재무/타임라인은 as_of 클램프된 AnalysisContext 가 제공.)
_SNAPSHOT_TIME_SENSITIVE_KEYS = frozenset(
    {
        "financial_summary",
        "recent_changes",
        "capability_evolution",
        "market_view",
        "priority_initiatives",
        "investment_roadmap",
        "operational_highlights",
    }
)


def _strip_snapshot_time_sensitive(snapshot: dict[str, Any]) -> dict[str, Any]:
    """백필(as_of) 시 snapshot 의 시점-민감 필드 제거 — 안정 프로필만 유지."""
    return {k: v for k, v in snapshot.items() if k not in _SNAPSHOT_TIME_SENSITIVE_KEYS}


def _user_strategy_overlay_limit() -> int:
    try:
        return max(1, int(os.getenv("USER_STRATEGY_OVERLAY_LIMIT", "20")))
    except ValueError:
        return 20


class ProfileContextLoader:
    """Load Tier A snapshot and Tier B enrichment into ``ProfileContext``."""

    def load(
        self,
        *,
        companies: list[str],
        sectors: list[str] | None = None,
        event_type: str | None = None,
        lookback_days: int = 30,
        as_of: date | None = None,
        peer_profile_context: dict[str, Any] | None = None,
        skax_profile_context: dict[str, Any] | None = None,
        user_id: str | None = None,
        user_profile_overlay: dict[str, Any] | None = None,
        user_profile_overlays: list[dict[str, Any]] | None = None,
        issue_scope: dict[str, Any] | None = None,
        strict: bool = False,
        require_skax_profile: bool = False,
    ) -> ProfileContext:
        """Tier A snapshot + Tier B recent enrichment 를 합쳐 ProfileContext 반환.

        Tier A: `peer_companies.profile_snapshot` JSONB (없으면 빈 dict fallback).
        Tier B: 최근 N일 business_signals top-3 + 최근 분기 financial_metrics.

        `strict=True` 는 통합 테스트/운영 점검용이다. DB 연결 실패나 peer snapshot
        누락을 fallback 으로 숨기지 않고 즉시 드러낸다. 일반 runtime 에서는 기존처럼
        fail-soft 로 동작한다.
        """
        peers_canonical = [
            company_id for company_id in companies if company_id not in SELF_COMPANY_IDS
        ]
        skax_profile = dict(skax_profile_context or {}) or _load_snapshot(
            "sk_ax",
            strict=require_skax_profile,
        )
        overlays = _user_strategy_overlays(
            user_id=user_id,
            user_profile_overlay=user_profile_overlay,
            user_profile_overlays=user_profile_overlays,
        )
        if overlays:
            skax_profile = dict(skax_profile)
            skax_profile["user_strategy_overlays"] = overlays
        if require_skax_profile and not _has_snapshot_payload(skax_profile):
            raise RuntimeError(
                "SK AX profile_snapshot was not loaded from peer_companies. "
                "Check peer_companies.id/name aliases for sk_ax and profile_snapshot columns."
            )
        if as_of is not None:
            skax_profile = _strip_snapshot_time_sensitive(skax_profile)

        peer_profiles: dict[str, Any] = dict(peer_profile_context or {})
        for peer_id in peers_canonical:
            snapshot = _load_snapshot(peer_id, strict=strict)
            if not snapshot:
                if strict:
                    raise RuntimeError(
                        "peer profile_snapshot was not loaded from peer_companies "
                        f"| peer_id={peer_id} aliases={expand_peer_aliases(peer_id)}"
                    )
                snapshot = {"peer_id": peer_id, "company_id": peer_id}
            if strict and not _has_snapshot_payload(snapshot):
                raise RuntimeError(
                    "peer profile_snapshot payload is empty or profile columns are missing "
                    f"| peer_id={peer_id} aliases={expand_peer_aliases(peer_id)}"
                )
            profile = dict(snapshot)
            if as_of is not None:
                profile = _strip_snapshot_time_sensitive(profile)
            profile.setdefault("peer_id", peer_id)
            profile.setdefault("company_id", peer_id)
            profile["recent_signals"] = _load_recent_business_signals(
                peer_id=peer_id, days=lookback_days, limit=3, as_of=as_of
            )
            profile["recent_financial"] = _load_latest_financial_metrics(
                peer_id=peer_id, as_of=as_of
            )
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


def _load_snapshot(company_id: str, *, strict: bool = False) -> dict[str, Any]:
    """`peer_companies.profile_snapshot` JSONB 가 있으면 dict 로 반환, 아니면 빈."""
    if not company_id:
        return {}
    aliases = expand_peer_aliases(company_id)
    try:
        with SessionLocal() as db:
            # peer_companies 의 profile_snapshot 컬럼은 환경별 migration 상태에 따라
            # 없을 수 있다. SELECT * 로 row 를 가져온 뒤 존재하는 컬럼만 사용한다.
            row = db.execute(
                text(
                    """
                    SELECT *
                      FROM peer_companies
                     WHERE id = ANY(:aliases)
                        OR name = ANY(:aliases)
                     ORDER BY CASE WHEN id = :company_id THEN 0 ELSE 1 END
                     LIMIT 1
                    """
                ),
                {"company_id": company_id, "aliases": aliases},
            ).fetchone()
    except Exception as exc:  # noqa: BLE001 — pre-V33 schema fallback.
        if strict:
            raise RuntimeError(
                "peer_companies.profile_snapshot lookup failed "
                f"| company_id={company_id} aliases={aliases} error={_safe_error(exc)}"
            ) from None
        log.debug(
            "profile_snapshot 컬럼 없음 또는 조회 실패 | company=%s error=%s",
            company_id,
            exc,
        )
        return {}
    if row is None:
        return {}
    mapping = row._mapping
    snapshot = mapping.get("profile_snapshot") or {}
    if not snapshot:
        peer_plus_payload = mapping.get("peer_plus_payload") or {}
        if isinstance(peer_plus_payload, dict):
            snapshot = peer_plus_payload.get("profile_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    row_company_id = mapping.get("id") or company_id
    snapshot.setdefault("company_id", row_company_id)
    snapshot.setdefault("peer_id", row_company_id)
    company_name = mapping.get("name") or mapping.get("name_ko")
    snapshot.setdefault("company_name", company_name)
    snapshot.setdefault("company_name_ko", company_name)
    snapshot.setdefault("profile_snapshot_version", mapping.get("profile_snapshot_version"))
    snapshot.setdefault(
        "profile_snapshot_generated_at",
        _iso(mapping.get("profile_snapshot_generated_at")),
    )
    return snapshot


def _has_snapshot_payload(snapshot: dict[str, Any]) -> bool:
    """Return True when a loaded profile contains actual Tier A snapshot content."""
    return any(
        snapshot.get(key)
        for key in (
            "schema_version",
            "profile_snapshot_version",
            "one_liner",
            "company_summary",
            "business_areas",
            "core_capabilities",
            "recent_changes",
            "capability_evolution",
        )
    )


def _user_strategy_overlays(
    *,
    user_id: str | None,
    user_profile_overlay: dict[str, Any] | None,
    user_profile_overlays: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    limit = _user_strategy_overlay_limit()
    overlays.extend(
        _load_user_strategy_overlays(
            user_id=user_id,
            limit=limit,
        )
    )
    if isinstance(user_profile_overlay, dict) and user_profile_overlay:
        overlays.append(user_profile_overlay)
    for overlay in user_profile_overlays or []:
        if isinstance(overlay, dict) and overlay:
            overlays.append(overlay)
    return overlays[:limit]


def _load_user_strategy_overlays(
    *,
    user_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    if not user_id:
        return []
    try:
        with SessionLocal() as db:
            rows = db.execute(
                text(
                    """
                    SELECT id, overlay_json, metadata
                     FROM user_strategy_contexts
                     WHERE user_id = CAST(:user_id AS uuid)
                     ORDER BY updated_at DESC
                     LIMIT :limit
                    """
                ),
                {"user_id": user_id, "limit": int(limit)},
            ).fetchall()
    except Exception as exc:  # noqa: BLE001 — pre-migration/local fallback.
        log.debug("user_strategy_contexts 조회 실패 | user_id=%s error=%s", user_id, exc)
        return []
    overlays: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        mapping = row._mapping
        context_id = str(mapping.get("id") or "").strip()
        if context_id and context_id in seen:
            continue
        if context_id:
            seen.add(context_id)
        overlay = mapping.get("overlay_json")
        if isinstance(overlay, dict) and overlay:
            item = dict(overlay)
            item["_strategy_context_id"] = context_id
            metadata = mapping.get("metadata")
            if isinstance(metadata, dict):
                item["_strategy_context_metadata"] = metadata
            overlays.append(item)
    return overlays


def _safe_error(exc: Exception) -> str:
    text = str(exc)
    text = re.sub(r"://([^:/@\s]+):([^@\s]+)@", r"://\1:***@", text)
    text = text.replace("\n", " ")
    return f"{type(exc).__name__}: {text[:300]}"


def _load_recent_business_signals(
    *, peer_id: str, days: int, limit: int, as_of: date | None = None
) -> list[dict[str, Any]]:
    aliases = expand_peer_aliases(peer_id)
    # point-in-time(백필): as_of 시점 이후 신호 차단 + 그 이전 N일 윈도우.
    # as_of=None 이면 기존 now-relative 동작 (라이브 무변경).
    if as_of is None:
        time_filter = "AND created_at >= NOW() - (:days || ' days')::interval"
        params: dict[str, Any] = {"aliases": aliases, "days": int(days), "limit": int(limit)}
    else:
        time_filter = (
            "AND created_at >= (:as_of::date - :days::int) AND created_at < (:as_of::date + 1)"
        )
        params = {"aliases": aliases, "days": int(days), "limit": int(limit), "as_of": as_of}
    try:
        with SessionLocal() as db:
            rows = db.execute(
                text(
                    f"""
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
                       {time_filter}
                     ORDER BY confidence DESC NULLS LAST, created_at DESC
                     LIMIT :limit
                    """
                ),
                params,
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


def _load_latest_financial_metrics(
    *, peer_id: str, as_of: date | None = None
) -> list[dict[str, Any]]:
    aliases = expand_peer_aliases(peer_id)
    canonical_keys = sorted(set(METRIC_CANONICAL.values()))
    # point-in-time(백필): as_of 시점까지 공시된(evidence 기사 발행일 <= as_of) 재무만.
    # as_of=None 이면 기존 동작(최신 전체).
    if as_of is None:
        as_of_filter = ""
        params: dict[str, Any] = {"aliases": aliases}
    else:
        as_of_filter = (
            "AND (raw_article_id IS NULL OR raw_article_id IN ("
            " SELECT id FROM raw_articles WHERE published_at < (:as_of::date + 1)))"
        )
        params = {"aliases": aliases, "as_of": as_of}
    try:
        with SessionLocal() as db:
            rows = db.execute(
                text(
                    f"""
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
                       {as_of_filter}
                     ORDER BY period_year DESC NULLS LAST,
                              period_quarter DESC NULLS LAST,
                              confidence DESC NULLS LAST
                     LIMIT 12
                    """
                ),
                params,
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


__all__ = ["ProfileContextLoader"]
