"""today_insight context_loader — extracted from facade (move-only)."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date
from typing import Any

from sqlalchemy import text

from src.agents.today_insight.text_processing import (  # noqa: F401
    _counter_top,
    _dedupe,
    _delta_rows,
    _json_ready,
    _list,
    _top_axis,
)
from src.contracts.today_insight_schemas import (
    TodayInsightGenerateRequest,
)
from src.db.postgres import SessionLocal
from src.services.profile_context_loader import ProfileContextLoader
from src.services.skax_profile_context_loader import SKAXProfileLoader

log = logging.getLogger(__name__)


def _fetch_integrated_issues(
    *, anchor_date: date, window_days: int, limit: int
) -> list[dict[str, Any]]:
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT ii.id::text AS id,
                           ii.cluster_id,
                           ii.main_company,
                           ii.event_type,
                           ii.source_family,
                           ii.confidence,
                           ii.headline,
                           ii.one_line_summary,
                           ii.analyzed_source_ids,
                           ii.source_ids,
                           ii.sectors,
                           ii.mentioned_peer_companies,
                           ii.content_summary,
                           ii.issue_frame,
                           ii.sources,
                           ii.evidence,
                           ii.quality,
                           ii.payload,
                           ii.created_at,
                           ii.updated_at,
                           ((ii.created_at AT TIME ZONE 'Asia/Seoul')::date)::text
                               AS created_date_kst,
                           (
                               SELECT MAX(source_date)::text
                                 FROM (
                                       SELECT (
                                           COALESCE(
                                               iisa.published_at,
                                               ra.published_at,
                                               ra.created_at
                                           ) AT TIME ZONE 'Asia/Seoul'
                                       )::date AS source_date
                                         FROM integrated_issue_source_articles iisa
                                         LEFT JOIN raw_articles ra ON ra.id = iisa.raw_article_id
                                        WHERE iisa.integrated_issue_id = ii.id
                                       UNION ALL
                                       SELECT (COALESCE(ra.published_at, ra.created_at)
                                           AT TIME ZONE 'Asia/Seoul')::date AS source_date
                                         FROM raw_articles ra
                                        WHERE ra.id = ii.representative_raw_article_id
                                           OR ra.id = ANY(ii.source_ids)
                                           OR ra.id = ANY(ii.analyzed_source_ids)
                                      ) issue_source_dates
                           ) AS latest_source_date_kst,
                           EXISTS (
                               SELECT 1
                                 FROM (
                                       SELECT (
                                           COALESCE(
                                               iisa.published_at,
                                               ra.published_at,
                                               ra.created_at
                                           ) AT TIME ZONE 'Asia/Seoul'
                                       )::date AS source_date
                                         FROM integrated_issue_source_articles iisa
                                         LEFT JOIN raw_articles ra ON ra.id = iisa.raw_article_id
                                        WHERE iisa.integrated_issue_id = ii.id
                                       UNION ALL
                                       SELECT (COALESCE(ra.published_at, ra.created_at)
                                           AT TIME ZONE 'Asia/Seoul')::date AS source_date
                                         FROM raw_articles ra
                                        WHERE ra.id = ii.representative_raw_article_id
                                           OR ra.id = ANY(ii.source_ids)
                                           OR ra.id = ANY(ii.analyzed_source_ids)
                                      ) issue_source_dates
                                WHERE source_date = CAST(:anchor_date AS date)
                           ) AS has_anchor_source
                     FROM integrated_issues ii
                     WHERE ii.is_current = TRUE
                       AND ii.status = 'active'
                       AND ii.is_valid = TRUE
                       AND (
                           NOT EXISTS (
                               SELECT 1
                                 FROM card_news cn_any
                                WHERE cn_any.integrated_issue_id = ii.id
                           )
                           OR EXISTS (
                               SELECT 1
                                 FROM card_news cn_active
                                WHERE cn_active.integrated_issue_id = ii.id
                                  AND LOWER(cn_active.status) = 'active'
                           )
                       )
                       AND (ii.created_at AT TIME ZONE 'Asia/Seoul')::date
                           >= CAST(:anchor_date AS date) - (:window_days * INTERVAL '1 day')
                     ORDER BY
                       CASE
                         WHEN EXISTS (
                               SELECT 1
                                 FROM (
                                       SELECT (
                                           COALESCE(
                                               iisa.published_at,
                                               ra.published_at,
                                               ra.created_at
                                           ) AT TIME ZONE 'Asia/Seoul'
                                       )::date AS source_date
                                         FROM integrated_issue_source_articles iisa
                                         LEFT JOIN raw_articles ra ON ra.id = iisa.raw_article_id
                                        WHERE iisa.integrated_issue_id = ii.id
                                       UNION ALL
                                       SELECT (COALESCE(ra.published_at, ra.created_at)
                                           AT TIME ZONE 'Asia/Seoul')::date AS source_date
                                         FROM raw_articles ra
                                        WHERE ra.id = ii.representative_raw_article_id
                                           OR ra.id = ANY(ii.source_ids)
                                           OR ra.id = ANY(ii.analyzed_source_ids)
                                      ) issue_source_dates
                                WHERE source_date = CAST(:anchor_date AS date)
                           )
                         THEN 0 ELSE 1
                       END,
                       ii.confidence DESC NULLS LAST,
                       ii.created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {
                        "anchor_date": anchor_date.isoformat(),
                        "window_days": int(window_days),
                        "limit": int(limit),
                    },
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("today insight integrated_issues lookup failed | error=%s", exc)
        return []
    return _attach_today_vdb_contexts(
        [_json_ready(dict(row)) for row in rows],
        row_kind="integrated_issue",
    )


def _fetch_cards_for_issues(
    issue_ids: list[str],
    *,
    anchor_date: date,
    window_days: int | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    del window_days
    if not issue_ids:
        return []
    placeholders = ", ".join(f"CAST(:issue_{idx} AS uuid)" for idx in range(len(issue_ids)))
    params: dict[str, Any] = {f"issue_{idx}": issue_id for idx, issue_id in enumerate(issue_ids)}
    params["limit"] = int(limit)
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        f"""
                    SELECT id,
                           title,
                           COALESCE(peer_company_id, company) AS peer_id,
                           summary_lines,
                           event_type,
                           importance,
                           importance_score,
                           implication,
                           primary_keyword_category,
                           evidence_payload,
                           source_raw_article_ids,
                           sources,
                           integrated_issue_id::text AS integrated_issue_id,
                           created_at,
                           COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           )::text AS created_date_kst
                     FROM card_news
                     LEFT JOIN LATERAL (
                         SELECT MIN((
                                    COALESCE(ra.published_at, ra.created_at)
                                    AT TIME ZONE 'Asia/Seoul'
                                )::date) AS earliest_source_date
                           FROM raw_articles ra
                          WHERE ra.id = ANY(COALESCE(source_raw_article_ids, '{{}}'::bigint[]))
                     ) src ON TRUE
                     WHERE integrated_issue_id IN ({placeholders})
                       AND LOWER(status) = 'active'
                       AND COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           ) = CAST(:anchor_date AS date)
                       AND COALESCE(peer_company_id, company, '') <> 'sk_ax'
                       AND (
                           COALESCE(cardinality(source_raw_article_ids), 0) = 0
                           OR EXISTS (
                               SELECT 1
                                 FROM raw_articles ra
                                WHERE ra.id = ANY(source_raw_article_ids)
                                  AND (
                                      COALESCE(ra.published_at, ra.created_at)
                                      AT TIME ZONE 'Asia/Seoul'
                                  )::date = CAST(:anchor_date AS date)
                           )
                       )
                     ORDER BY
                       COALESCE(
                           src.earliest_source_date,
                           (created_at AT TIME ZONE 'Asia/Seoul')::date
                       ) DESC,
                       importance_score DESC NULLS LAST,
                       created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {
                        **params,
                        "anchor_date": anchor_date.isoformat(),
                    },
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("today insight card lookup failed | error=%s", exc)
        return []
    return _attach_today_vdb_contexts([_json_ready(dict(row)) for row in rows], row_kind="card")


def _fetch_anchor_date_cards(
    *,
    anchor_date: date,
    limit: int,
    exclude_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    clean_exclude_ids = _dedupe(
        [str(card_id) for card_id in exclude_ids or [] if str(card_id or "").strip()],
        limit=100,
    )
    exclude_clause = ""
    params: dict[str, Any] = {
        "anchor_date": anchor_date.isoformat(),
        "limit": int(limit),
    }
    if clean_exclude_ids:
        placeholders = ", ".join(f":exclude_{idx}" for idx in range(len(clean_exclude_ids)))
        exclude_clause = f"AND id NOT IN ({placeholders})"
        params.update({f"exclude_{idx}": card_id for idx, card_id in enumerate(clean_exclude_ids)})

    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        f"""
                    SELECT id,
                           title,
                           COALESCE(peer_company_id, company) AS peer_id,
                           summary_lines,
                           event_type,
                           importance,
                           importance_score,
                           implication,
                           primary_keyword_category,
                           evidence_payload,
                           source_raw_article_ids,
                           sources,
                           integrated_issue_id::text AS integrated_issue_id,
                           created_at,
                           COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           )::text AS created_date_kst
                      FROM card_news
                      LEFT JOIN LATERAL (
                          SELECT MIN((
                                     COALESCE(ra.published_at, ra.created_at)
                                     AT TIME ZONE 'Asia/Seoul'
                                 )::date) AS earliest_source_date
                            FROM raw_articles ra
                           WHERE ra.id = ANY(COALESCE(source_raw_article_ids, '{{}}'::bigint[]))
                      ) src ON TRUE
                     WHERE COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           ) = CAST(:anchor_date AS date)
                       AND LOWER(status) = 'active'
                       AND COALESCE(peer_company_id, company, '') <> 'sk_ax'
                       AND (
                           COALESCE(cardinality(source_raw_article_ids), 0) = 0
                           OR EXISTS (
                               SELECT 1
                                 FROM raw_articles ra
                                WHERE ra.id = ANY(source_raw_article_ids)
                                  AND (
                                      COALESCE(ra.published_at, ra.created_at)
                                      AT TIME ZONE 'Asia/Seoul'
                                  )::date = CAST(:anchor_date AS date)
                           )
                       )
                       {exclude_clause}
                     ORDER BY
                       COALESCE(
                           src.earliest_source_date,
                           (created_at AT TIME ZONE 'Asia/Seoul')::date
                       ) DESC,
                       importance_score DESC NULLS LAST,
                       created_at DESC
                     LIMIT :limit
                    """
                    ),
                    params,
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("today insight anchor-date card lookup failed | error=%s", exc)
        return []
    return _attach_today_vdb_contexts([_json_ready(dict(row)) for row in rows], row_kind="card")


def _fetch_recent_cards(
    *,
    anchor_date: date,
    window_days: int,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT id,
                           title,
                           COALESCE(peer_company_id, company) AS peer_id,
                           summary_lines,
                           event_type,
                           importance,
                           importance_score,
                           implication,
                           primary_keyword_category,
                           evidence_payload,
                           source_raw_article_ids,
                           sources,
                           integrated_issue_id::text AS integrated_issue_id,
                           created_at,
                           COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           )::text AS created_date_kst
                      FROM card_news
                      LEFT JOIN LATERAL (
                          SELECT MIN((
                                     COALESCE(ra.published_at, ra.created_at)
                                     AT TIME ZONE 'Asia/Seoul'
                                 )::date) AS earliest_source_date
                            FROM raw_articles ra
                           WHERE ra.id = ANY(COALESCE(source_raw_article_ids, '{}'::bigint[]))
                      ) src ON TRUE
                     WHERE COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           )
                           BETWEEN CAST(:anchor_date AS date)
                               - (:window_days * INTERVAL '1 day')
                               AND CAST(:anchor_date AS date)
                       AND LOWER(status) = 'active'
                       AND COALESCE(peer_company_id, company, '') <> 'sk_ax'
                       AND (
                           COALESCE(cardinality(source_raw_article_ids), 0) = 0
                           OR EXISTS (
                               SELECT 1
                                 FROM raw_articles ra
                                WHERE ra.id = ANY(source_raw_article_ids)
                                  AND (
                                      COALESCE(ra.published_at, ra.created_at)
                                      AT TIME ZONE 'Asia/Seoul'
                                  )::date BETWEEN CAST(:anchor_date AS date)
                                      - (:window_days * INTERVAL '1 day')
                                      AND CAST(:anchor_date AS date)
                           )
                       )
                     ORDER BY
                       COALESCE(
                           src.earliest_source_date,
                           (created_at AT TIME ZONE 'Asia/Seoul')::date
                       ) DESC,
                       importance_score DESC NULLS LAST,
                       created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {
                        "anchor_date": anchor_date.isoformat(),
                        "window_days": int(window_days),
                        "limit": int(limit),
                    },
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("today insight recent card lookup failed | error=%s", exc)
        return []
    return _attach_today_vdb_contexts([_json_ready(dict(row)) for row in rows], row_kind="card")


def _attach_today_vdb_contexts(
    rows: list[dict[str, Any]],
    *,
    row_kind: str,
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    prepared = [dict(row) for row in rows]
    if row_kind == "integrated_issue":
        for row in prepared:
            if row.get("id") and not row.get("integrated_issue_id"):
                row["integrated_issue_id"] = str(row["id"])
    try:
        from src.rag.content_index import attach_vdb_contexts_to_rows

        return attach_vdb_contexts_to_rows(prepared, max_chars=1800)
    except Exception as exc:  # noqa: BLE001
        log.debug("today insight content VDB hydration skipped | error=%s", exc)
        return prepared


def _fetch_prior_today_reports(*, anchor_date: date, limit: int) -> list[dict[str, Any]]:
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT report_date::text AS report_date,
                           headline,
                           executive_summary,
                           output_payload,
                           source_integrated_issue_ids,
                           source_card_ids,
                           confidence,
                           created_at
                      FROM today_insight_reports
                     WHERE report_date < CAST(:anchor_date AS date)
                       AND status = 'active'
                     ORDER BY report_date DESC, created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {"anchor_date": anchor_date.isoformat(), "limit": int(limit)},
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("prior today insight reports unavailable | error=%s", exc)
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        item = _json_ready(dict(row))
        payload = item.get("output_payload") if isinstance(item.get("output_payload"), dict) else {}
        out.append(
            {
                "report_date": item.get("report_date"),
                "headline": item.get("headline") or payload.get("headline"),
                "executive_summary": item.get("executive_summary")
                or payload.get("executive_summary"),
                "signal_values": [
                    signal.get("value")
                    for signal in _list(payload.get("signals"))
                    if isinstance(signal, dict)
                ][:3],
                "source_integrated_issue_ids": _list(item.get("source_integrated_issue_ids")),
                "source_card_ids": _list(item.get("source_card_ids")),
                "confidence": item.get("confidence"),
            }
        )
    return out


def _load_latest_report(anchor_date: date) -> dict[str, Any] | None:
    record = _load_latest_report_record(anchor_date)
    return record["payload"] if record else None


def _load_latest_report_record(anchor_date: date) -> dict[str, Any] | None:
    """Return the newest active report on or before ``anchor_date``.

    Home can show the latest saved report, but marks it as a fallback when it is
    older than the selected anchor date.
    """
    try:
        with SessionLocal() as db:
            row = (
                db.execute(
                    text(
                        """
                    SELECT report_date::text AS report_date,
                           output_payload,
                           created_at
                      FROM today_insight_reports
                     WHERE report_date <= CAST(:anchor_date AS date)
                       AND status = 'active'
                     ORDER BY report_date DESC, created_at DESC
                     LIMIT 1
                    """
                    ),
                    {"anchor_date": anchor_date.isoformat()},
                )
                .mappings()
                .first()
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("today insight cache lookup skipped | error=%s", exc)
        return None
    if not row:
        return None
    payload = row.get("output_payload")
    if not isinstance(payload, dict):
        return None
    report_date_value = str(row.get("report_date") or "")
    payload = _json_ready(payload)
    payload.setdefault("report_date", report_date_value)
    provenance = payload.setdefault("provenance", {})
    if isinstance(provenance, dict):
        provenance["cache_lookup"] = "latest_saved_on_or_before_anchor"
        provenance["served_anchor_date"] = anchor_date.isoformat()
        provenance["cached_report_date"] = report_date_value
        if report_date_value and report_date_value != anchor_date.isoformat():
            provenance["latest_fallback"] = True
    return {
        "payload": payload,
        "created_at": row.get("created_at"),
        "report_date": report_date_value,
    }


def _fetch_analysis_ledger(
    *, peer_ids: list[str], window_days: int, limit: int
) -> list[dict[str, Any]]:
    del peer_ids  # broad recent ledger is useful even when peer aliases are sparse.
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT analysis_type,
                           analysis_id,
                           peer_ids,
                           conclusion_one_liner,
                           confidence,
                           source_card_ids,
                           sk_ax_implication,
                           created_at
                      FROM analysis_ledger
                     WHERE confidence >= 0.6
                       AND superseded_by IS NULL
                       AND included_in_pack = TRUE
                       AND created_at >= now() - (:window_days * INTERVAL '1 day')
                     ORDER BY created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {"window_days": int(window_days), "limit": int(limit)},
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("analysis ledger context unavailable | error=%s", exc)
        return []
    return [_json_ready(dict(row)) for row in rows]


def _load_profile_context(
    *, peer_ids: list[str], sectors: list[str], req: TodayInsightGenerateRequest
) -> dict[str, Any]:
    if not peer_ids:
        return {"skax_profile": {}, "peer_profiles": {}, "sector_context": {}}
    try:
        return (
            ProfileContextLoader()
            .load(
                companies=peer_ids[:6],
                sectors=sectors[:6],
                lookback_days=req.window_days,
            )
            .to_dict()
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("profile context load failed | error=%s", exc)
        return {"skax_profile": {}, "peer_profiles": {}, "sector_context": {}}


def _load_skax_context(*, sectors: list[str]) -> dict[str, Any]:
    try:
        return SKAXProfileLoader().load(
            sectors[:6],
            max_documents=5,
            max_newsroom_documents=5,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("SK AX official context load failed | error=%s", exc)
        return {}


def _build_change_stats(
    *,
    anchor_date: date,
    current_issues: list[dict[str, Any]],
    history_issues: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    window_days: int,
) -> dict[str, Any]:
    current_peer_counts = Counter(
        str(row.get("main_company")) for row in current_issues if row.get("main_company")
    )
    history_peer_counts = Counter(
        str(row.get("main_company")) for row in history_issues if row.get("main_company")
    )
    card_peer_counts = Counter(str(card.get("peer_id")) for card in cards if card.get("peer_id"))
    current_sector_counts = Counter(
        str(sector) for row in current_issues for sector in _list(row.get("sectors")) if sector
    )
    history_sector_counts = Counter(
        str(sector) for row in history_issues for sector in _list(row.get("sectors")) if sector
    )
    current_event_counts = Counter(
        str(row.get("event_type")) for row in current_issues if row.get("event_type")
    )
    card_event_counts = Counter(
        str(card.get("event_type")) for card in cards if card.get("event_type")
    )
    combined_peer_counts = current_peer_counts + card_peer_counts
    combined_event_counts = current_event_counts + card_event_counts
    top_axis = _top_axis(current_sector_counts, combined_peer_counts, combined_event_counts)
    return {
        "anchor_date": anchor_date.isoformat(),
        "today_issue_count": len(current_issues),
        "recent_card_count": len(cards),
        "history_issue_count": len(history_issues),
        "window_days": window_days,
        "top_peer": _counter_top(combined_peer_counts),
        "top_sector": _counter_top(current_sector_counts),
        "top_event_type": _counter_top(combined_event_counts),
        "peer_delta_vs_window": _delta_rows(combined_peer_counts, history_peer_counts, window_days),
        "sector_delta_vs_window": _delta_rows(
            current_sector_counts, history_sector_counts, window_days
        ),
        "default_change_summary": [
            {"label": "오늘 감지된 변화", "value": f"{len(current_issues) + len(cards)}건"},
            {"label": "비교 기준", "value": f"최근 {window_days}일"},
            {"label": "핵심 축", "value": top_axis or "-"},
        ],
    }


# === 홈 3상태(today_signal/recent_signal/quiet) + 모니터링 커버리지 ===
# PR #212 의 has_current_signal/has_anchor_cards 게이트는 그대로 두고, 결과 정규화
# 단계에서 state/signal_date/week_synthesis/coverage_stats 만 얹는다.
# (recent_signal: 최근 2~3 영업일 윈도우는 후속 단계에서 추가.)


def _uuid_array_literal(values: list[Any]) -> str:
    cleaned = [str(value) for value in values if value]
    if not cleaned:
        return "{}"
    return "{" + ",".join(cleaned) + "}"


def _pg_text_array(values: list[Any]) -> str:
    cleaned = [str(value).replace('"', '\\"') for value in values if value]
    if not cleaned:
        return "{}"
    return "{" + ",".join(f'"{value}"' for value in cleaned) + "}"


def _pg_bigint_array(values: list[int]) -> str:
    if not values:
        return "{}"
    return "{" + ",".join(str(int(value)) for value in values) + "}"
