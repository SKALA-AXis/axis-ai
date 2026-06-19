# 작성일: 2026-06-11
# 작성자: 박지원
# 변경이력:
#   2026-06-11 박지원 — 카드뉴스 백필 스크립트 추가 및 품질/provenance 동기화, 통합·요약
#   2026-06-16 최종민 — 카드뉴스 백필을 point-in-time(시간순) 방식으로 변경 (as_of + created_at
"""Backfill card_news for historical raw_article clusters.

This runs the current analysis/card-news agent stack for existing processed
news clusters. It is intentionally resumable: clusters that already have an
ACTIVE card are skipped unless --replace-existing is provided.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.env_loader import load_profile  # noqa: E402
from src.db.article_store import save_card_news, sync_card_sources_for_cluster  # noqa: E402
from src.db.postgres import SessionLocal, reconfigure_from_env  # noqa: E402
from src.pipeline.analysis_pipeline import AnalysisPipelineRunner  # noqa: E402
from src.preprocessing.preprocessing import PreprocessingService  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("backfill_card_news")

_FINANCIAL_LIKE_TITLE_RE = re.compile(
    r"영업이익|순이익|수익성|주가|목표가|투자의견|상한가|하한가|"
    r"주가.{0,12}(상승|하락|급등|급락)|전년\s*동기|전분기|"
    r"분기\s*(매출|영업이익|실적)|배당|주주환원|R&D\s*지출|"
    r"공모|공모가|공모주|수요예측|청약|상장|IPO|시가총액|기업가치|"
    r"경영\s*실적|투자재원|사내이사|이사회|주주총회|사회이사진|법률자문|"
    r"증권|투자수익률|주식\s*초고수|사들인\s*종목|매수\s*종목|"
    r"채용|공채|인재\s*모집"
)
_FINANCIAL_LIKE_EVENTS = {"financial", "earnings", "stock_market", "analyst_report"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill card_news for existing news clusters")
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--published-since", required=True)
    parser.add_argument("--published-until", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument(
        "--update-existing-in-place",
        action="store_true",
        help="Regenerate with the current agent stack but keep the existing ACTIVE card id.",
    )
    parser.add_argument(
        "--only-card-schema-version",
        default=None,
        help="Limit targets to clusters with an ACTIVE card using this schema version, e.g. v1.",
    )
    parser.add_argument(
        "--only-existing-active",
        action="store_true",
        help="Limit targets to clusters that currently have an ACTIVE card.",
    )
    parser.add_argument(
        "--only-needs-refresh",
        action="store_true",
        help=(
            "Limit targets to ACTIVE cards with stale schema/provenance, empty/problematic "
            "actions, or body-like titles."
        ),
    )
    parser.add_argument(
        "--only-deleted-needs-refresh",
        action="store_true",
        help=(
            "Limit targets to clusters with DELETED stale cards and no ACTIVE card, so they can "
            "be regenerated with the current agent stack."
        ),
    )
    parser.add_argument(
        "--include-financial-like",
        action="store_true",
        help=(
            "Also regenerate financial/stock/earnings-like stale cards. "
            "Default is to delete/skip them."
        ),
    )
    parser.add_argument(
        "--cluster-ids",
        default=None,
        help="Comma-separated cluster ids to process after the normal target filters.",
    )
    parser.add_argument(
        "--include-industry-trend",
        action="store_true",
        help="Include industry_trend news clusters in addition to peer-company clusters.",
    )
    parser.add_argument(
        "--industry-trend-only",
        action="store_true",
        help="Only process industry_trend news clusters.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    reconfigure_from_env()
    include_existing = bool(
        args.replace_existing
        or args.update_existing_in_place
        or args.only_card_schema_version
        or args.only_needs_refresh
        or args.only_deleted_needs_refresh
    )
    targets = _load_targets(
        published_since=args.published_since,
        published_until=args.published_until,
        limit=max(1, args.limit),
        include_existing=include_existing,
        only_card_schema_version=args.only_card_schema_version,
        only_existing_active=args.only_existing_active,
        only_needs_refresh=args.only_needs_refresh,
        only_deleted_needs_refresh=args.only_deleted_needs_refresh,
        cluster_ids=_parse_cluster_ids(args.cluster_ids),
        include_industry_trend=args.include_industry_trend or args.industry_trend_only,
        industry_trend_only=args.industry_trend_only,
    )
    log.info(
        (
            "card_news backfill 대상 | profile=%s since=%s until=%s targets=%d "
            "replace_existing=%s update_in_place=%s schema=%s"
        ),
        profile,
        args.published_since,
        args.published_until,
        len(targets),
        args.replace_existing,
        args.update_existing_in_place,
        args.only_card_schema_version,
    )
    if args.dry_run:
        for target in targets:
            print(
                f"cluster_id={target['cluster_id']} "
                f"representative_id={target['representative_id']} "
                f"articles={len(target['article_ids'])} "
                f"existing_card_id={target.get('existing_card_id') or ''}"
            )
        return

    service = PreprocessingService()
    runner = AnalysisPipelineRunner()
    created = 0
    skipped = 0
    errors = 0

    for index, target in enumerate(targets, start=1):
        cluster_id = int(target["cluster_id"])
        article_ids = [int(value) for value in target["article_ids"]]
        representative_id = int(target["representative_id"])
        existing_card_id = str(target.get("existing_card_id") or "")
        # point-in-time 백필: 클러스터 최초 발행일을 기준일로 — 과거맥락 클램프(룩어헤드 차단)
        # + 카드 created_at 을 발행일로 박아 프론트 정렬·타임라인 정합.
        _min_published = target.get("min_published")
        as_of = _min_published.date() if isinstance(_min_published, datetime) else _min_published
        try:
            classified = service.classify_clusters(
                representative_ids=[representative_id],
                cluster_map={cluster_id: article_ids},
                requested_companies=[],
            )
            if not classified:
                skipped += 1
                log.info(
                    "card_news backfill skip | cluster_id=%s reason=no_classification", cluster_id
                )
                continue
            if not args.include_financial_like and _is_financial_like_target(classified[0], target):
                deleted = (
                    _mark_existing_cards_deleted(cluster_id)
                    if args.replace_existing or args.update_existing_in_place
                    else 0
                )
                skipped += 1
                log.info(
                    (
                        "card_news backfill skip financial-like | %d/%d "
                        "cluster_id=%s event_type=%s deleted=%d"
                    ),
                    index,
                    len(targets),
                    cluster_id,
                    classified[0].get("event_type"),
                    deleted,
                )
                continue
            if args.replace_existing and not args.update_existing_in_place:
                _mark_existing_cards_deleted(cluster_id)
            result = runner.run_cluster(
                cluster_id=cluster_id,
                representative_id=representative_id,
                cluster_article_ids=article_ids,
                classification=classified[0],
                as_of=as_of,
                save_card=not args.update_existing_in_place,
            )
            card = result.get("card_news") or {}
            if card:
                validation = (
                    card.get("validation") if isinstance(card.get("validation"), dict) else {}
                )
                validation_pass = bool(card.get("validation_pass", validation.get("pass", False)))
                title = str(card.get("title") or "")
                has_displayable_implication = _has_displayable_implication(card)
                if (
                    not validation_pass
                    and _is_problematic_generated_title(title)
                    and not has_displayable_implication
                ):
                    transient_card_id = str(card.get("id") or "")
                    deleted = _mark_card_deleted(transient_card_id) if transient_card_id else 0
                    skipped += 1
                    log.info(
                        (
                            "card_news backfill deleted invalid generated card | "
                            "%d/%d cluster_id=%s card_id=%s title=%s deleted=%d"
                        ),
                        index,
                        len(targets),
                        cluster_id,
                        transient_card_id,
                        title,
                        deleted,
                    )
                    continue
                if (
                    not args.include_financial_like
                    and _is_problematic_generated_title(title)
                    and not has_displayable_implication
                ):
                    transient_card_id = str(card.get("id") or "")
                    deleted = _mark_card_deleted(transient_card_id) if transient_card_id else 0
                    if args.update_existing_in_place and existing_card_id:
                        deleted += _mark_card_deleted(existing_card_id)
                    skipped += 1
                    log.info(
                        (
                            "card_news backfill deleted generated problematic card | "
                            "%d/%d cluster_id=%s card_id=%s title=%s deleted=%d"
                        ),
                        index,
                        len(targets),
                        cluster_id,
                        transient_card_id or existing_card_id,
                        card.get("title"),
                        deleted,
                    )
                    continue
                saved_card_id = str(result.get("saved_card_id") or "")
                if args.update_existing_in_place:
                    if existing_card_id:
                        if not _has_direct_frontend_ready(card):
                            skipped += 1
                            log.warning(
                                (
                                    "card_news backfill skip in-place update | %d/%d "
                                    "cluster_id=%s card_id=%s reason=no_direct_frontend_ready"
                                ),
                                index,
                                len(targets),
                                cluster_id,
                                existing_card_id,
                            )
                            continue
                        transient_card_id = str(card.get("id") or "")
                        card["id"] = existing_card_id
                        saved_card_id = save_card_news(card)
                        if saved_card_id:
                            sync_card_sources_for_cluster(cluster_id)
                            _mark_other_active_cards_deleted(cluster_id, keep_card_id=saved_card_id)
                        if (
                            transient_card_id
                            and transient_card_id != existing_card_id
                            and transient_card_id != saved_card_id
                        ):
                            _mark_card_deleted(transient_card_id)
                        if saved_card_id:
                            created += 1
                            log.info(
                                (
                                    "card_news backfill updated in place | %d/%d "
                                    "cluster_id=%s card_id=%s transient_id=%s title=%s"
                                ),
                                index,
                                len(targets),
                                cluster_id,
                                saved_card_id,
                                transient_card_id,
                                card.get("title"),
                            )
                        else:
                            errors += 1
                            log.error(
                                "card_news backfill save failed | %d/%d cluster_id=%s card_id=%s",
                                index,
                                len(targets),
                                cluster_id,
                                existing_card_id,
                            )
                        continue
                    log.warning(
                        "card_news backfill skip | cluster_id=%s reason=no_existing_card",
                        cluster_id,
                    )
                    skipped += 1
                    continue
                if saved_card_id:
                    created += 1
                    sync_card_sources_for_cluster(cluster_id)
                    log.info(
                        "card_news backfill created | %d/%d cluster_id=%s card_id=%s title=%s",
                        index,
                        len(targets),
                        cluster_id,
                        saved_card_id,
                        card.get("title"),
                    )
                else:
                    errors += 1
                    log.error(
                        "card_news backfill save failed | %d/%d cluster_id=%s card_id=%s title=%s",
                        index,
                        len(targets),
                        cluster_id,
                        card.get("id"),
                        card.get("title"),
                    )
            else:
                if args.update_existing_in_place and existing_card_id:
                    deleted = _mark_card_deleted(existing_card_id)
                    skipped += 1
                    log.info(
                        (
                            "card_news backfill deleted stale existing card | %d/%d "
                            "cluster_id=%s card_id=%s reason=no_card deleted=%d"
                        ),
                        index,
                        len(targets),
                        cluster_id,
                        existing_card_id,
                        deleted,
                    )
                    continue
                skipped += 1
                log.info("card_news backfill skip | cluster_id=%s reason=no_card", cluster_id)
        except Exception as exc:
            errors += 1
            log.exception("card_news backfill error | cluster_id=%s error=%s", cluster_id, exc)

    log.info("card_news backfill 완료 | created=%d skipped=%d errors=%d", created, skipped, errors)


def _load_targets(
    *,
    published_since: str,
    published_until: str,
    limit: int,
    include_existing: bool,
    only_card_schema_version: str | None,
    only_existing_active: bool,
    only_needs_refresh: bool,
    only_deleted_needs_refresh: bool,
    cluster_ids: list[int],
    include_industry_trend: bool,
    industry_trend_only: bool,
) -> list[dict[str, Any]]:
    existing_filter = (
        ""
        if include_existing
        else """
      AND NOT EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'ACTIVE'
            AND cn.cluster_id = cluster_rows.cluster_id
      )
    """
    )
    schema_filter = (
        """
      AND EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'ACTIVE'
            AND cn.cluster_id = cluster_rows.cluster_id
            AND cn.card_schema_version = :only_card_schema_version
      )
    """
        if only_card_schema_version
        else ""
    )
    existing_active_filter = (
        """
      AND EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'ACTIVE'
            AND cn.cluster_id = cluster_rows.cluster_id
      )
    """
        if only_existing_active
        else ""
    )
    needs_refresh_filter = (
        """
      AND EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'ACTIVE'
            AND cn.cluster_id = cluster_rows.cluster_id
            AND (
                cn.card_schema_version IS DISTINCT FROM 'v2'
                OR COALESCE(
                    cn.evidence_payload->'analysis_package'->'implication'->'provenance'->>'prompt_version',
                    ''
                ) NOT LIKE 'strategic-insight-v1.61%'
                OR cn.implication::text ILIKE '%제안서%'
                OR cn.implication::text ILIKE '%PoC%'
                OR cn.implication::text ILIKE '%검증표%'
                OR cn.implication::text ILIKE '%데이터 없음%'
                OR cardinality(COALESCE(cn.summary_lines, ARRAY[]::text[])) < 3
                OR (
                    jsonb_array_length(
                        COALESCE(cn.implication->'frontend'->'key_implications', '[]'::jsonb)
                    ) = 0
                    AND jsonb_array_length(
                        COALESCE(cn.implication->'key_implications', '[]'::jsonb)
                    ) = 0
                )
                OR (
                    jsonb_array_length(
                        COALESCE(cn.implication->'frontend'->'suggested_actions', '[]'::jsonb)
                    ) = 0
                    AND jsonb_array_length(
                        COALESCE(cn.implication->'recommended_actions', '[]'::jsonb)
                    ) = 0
                    AND jsonb_array_length(
                        COALESCE(
                            cn.implication->'skax_implication'->'recommended_actions',
                            '[]'::jsonb
                        )
                    ) = 0
                )
                OR cn.title ILIKE '%사진=%'
                OR cn.title ILIKE '%전자공시시스템%'
                OR cn.title ILIKE '%따르면%'
                OR length(cn.title) > 80
            )
      )
    """
        if only_needs_refresh
        else ""
    )
    deleted_needs_refresh_filter = (
        """
      AND NOT EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'ACTIVE'
            AND cn.cluster_id = cluster_rows.cluster_id
      )
      AND EXISTS (
          SELECT 1 FROM card_news cn
          WHERE cn.status = 'DELETED'
            AND cn.cluster_id = cluster_rows.cluster_id
            AND (
                cn.card_schema_version IS DISTINCT FROM 'v2'
                OR COALESCE(
                    cn.evidence_payload->'analysis_package'->'implication'->'provenance'->>'prompt_version',
                    ''
                ) NOT LIKE 'strategic-insight-v1.61%'
                OR cn.implication::text ILIKE '%제안서%'
                OR cn.implication::text ILIKE '%PoC%'
                OR cn.implication::text ILIKE '%검증표%'
                OR cn.implication::text ILIKE '%데이터 없음%'
                OR cardinality(COALESCE(cn.summary_lines, ARRAY[]::text[])) < 3
                OR (
                    jsonb_array_length(
                        COALESCE(cn.implication->'frontend'->'key_implications', '[]'::jsonb)
                    ) = 0
                    AND jsonb_array_length(
                        COALESCE(cn.implication->'key_implications', '[]'::jsonb)
                    ) = 0
                )
                OR (
                    jsonb_array_length(
                        COALESCE(cn.implication->'frontend'->'suggested_actions', '[]'::jsonb)
                    ) = 0
                    AND jsonb_array_length(
                        COALESCE(cn.implication->'recommended_actions', '[]'::jsonb)
                    ) = 0
                    AND jsonb_array_length(
                        COALESCE(
                            cn.implication->'skax_implication'->'recommended_actions',
                            '[]'::jsonb
                        )
                    ) = 0
                )
                OR cn.title ILIKE '%사진=%'
                OR cn.title ILIKE '%전자공시시스템%'
                OR cn.title ILIKE '%따르면%'
                OR length(cn.title) > 80
            )
      )
    """
        if only_deleted_needs_refresh
        else ""
    )
    cluster_ids_filter = (
        """
      AND cluster_rows.cluster_id = ANY(:cluster_ids)
    """
        if cluster_ids
        else ""
    )
    company_scope_filter = (
        """
                      AND ra.company ? 'industry_trend'
        """
        if industry_trend_only
        else """
                      AND (
                          EXISTS (
                              SELECT 1
                              FROM peer_companies pc
                              WHERE pc.is_active = TRUE
                                AND pc.id <> 'sk_ax'
                                AND (
                                    ra.company ? pc.id
                                    OR ra.matched_companies ? pc.id
                                )
                          )
                          OR (
                              :include_industry_trend
                              AND ra.company ? 'industry_trend'
                          )
                      )
        """
    )
    with SessionLocal() as db:
        rows = db.execute(
            text(
                f"""
                WITH cluster_rows AS (
                    SELECT
                        ra.cluster_id,
                        ARRAY_AGG(
                            ra.id
                            ORDER BY
                                ra.published_at DESC NULLS LAST,
                                ra.collected_at DESC NULLS LAST,
                                ra.id DESC
                        ) AS article_ids,
                        ARRAY_AGG(
                            COALESCE(ra.title, '')
                            ORDER BY
                                ra.published_at DESC NULLS LAST,
                                ra.collected_at DESC NULLS LAST,
                                ra.id DESC
                        ) AS titles,
                        (
                            ARRAY_AGG(
                                ra.id
                                ORDER BY
                                    ra.is_representative DESC,
                                    ra.published_at DESC NULLS LAST,
                                    ra.collected_at DESC NULLS LAST,
                                    ra.id DESC
                            )
                        )[1] AS representative_id,
                        MIN(ra.published_at) AS min_published
                    FROM raw_articles ra
                    WHERE ra.source_type IN ('news', 'official')
                      AND ra.published_at >= CAST(:published_since AS timestamptz)
                      AND ra.published_at < CAST(:published_until AS timestamptz)
                      AND ra.processing_status = 'PROCESSED'
                      AND ra.relevance_label = 'relevant'
                      AND ra.cluster_id IS NOT NULL
                      AND COALESCE(ra.title, '') !~* (
                          '영업이익|순이익|수익성|주가|목표가|투자의견|상한가|하한가|'
                          '전년\\s*동기|전분기|분기\\s*(매출|영업이익|실적)|'
                          '배당|주주환원|R&D\\s*지출|공모|공모가|공모주|수요예측|'
                          '청약|상장|IPO|시가총액|기업가치|증권신고서|'
                          '경영\\s*실적|투자재원|사내이사|이사회|주주총회|'
                          '사회이사진|법률자문|증권|투자수익률|주식\\s*초고수|'
                          '사들인\\s*종목|매수\\s*종목|채용|공채|인재\\s*모집'
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM raw_articles ra2
                          WHERE ra2.cluster_id = ra.cluster_id
                            AND COALESCE(ra2.title, '') !~* '수상|표창'
                            AND COALESCE(ra2.title, '') ~* (
                                '영업이익|순이익|수익성|주가|목표가|투자의견|상한가|하한가|'
                                '전년\\s*동기|전분기|분기\\s*(매출|영업이익|실적)|'
                                '배당|주주환원|R&D\\s*지출|공모|공모가|공모주|수요예측|'
                                '청약|상장|IPO|시가총액|기업가치|증권신고서|'
                                '경영\\s*실적|투자재원|사내이사|이사회|주주총회|'
                                '사회이사진|법률자문|증권|투자수익률|주식\\s*초고수|'
                                '사들인\\s*종목|매수\\s*종목|채용|공채|인재\\s*모집'
                            )
                      )
                      {company_scope_filter}
                    GROUP BY ra.cluster_id
                )
                SELECT cluster_id, article_ids, titles, representative_id, min_published
                     , (
                          SELECT cn.id
                          FROM card_news cn
                          WHERE cn.status = 'ACTIVE'
                            AND cn.cluster_id = cluster_rows.cluster_id
                          ORDER BY cn.created_at ASC
                          LIMIT 1
                       ) AS existing_card_id
                FROM cluster_rows
                WHERE TRUE
                  {existing_filter}
                  {schema_filter}
                  {existing_active_filter}
                  {needs_refresh_filter}
                  {deleted_needs_refresh_filter}
                  {cluster_ids_filter}
                ORDER BY min_published, cluster_id
                LIMIT :limit
                """
            ),
            {
                "published_since": published_since,
                "published_until": published_until,
                "limit": limit,
                "only_card_schema_version": only_card_schema_version,
                "cluster_ids": cluster_ids,
                "include_industry_trend": include_industry_trend,
            },
        ).mappings()
        return [dict(row) for row in rows]


def _parse_cluster_ids(value: str | None) -> list[int]:
    if not value:
        return []
    ids: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item))
    return ids


def _is_financial_like_target(classification: dict[str, Any], target: dict[str, Any]) -> bool:
    event_type = str(classification.get("event_type") or "").strip().lower()
    if event_type in _FINANCIAL_LIKE_EVENTS:
        return True
    titles = target.get("titles") or []
    if isinstance(titles, str):
        titles = [titles]
    normalized_titles = [str(title or "") for title in titles if str(title or "").strip()]
    if not normalized_titles:
        return False
    financial_title_count = sum(1 for title in normalized_titles if _is_financial_like_text(title))
    return financial_title_count >= max(2, int(len(normalized_titles) * 0.4))


def _is_financial_like_text(text: str) -> bool:
    return bool(_FINANCIAL_LIKE_TITLE_RE.search(text))


def _is_problematic_generated_title(title: str) -> bool:
    stripped = title.strip()
    if not stripped:
        return True
    if len(stripped) > 80:
        return True
    if any(fragment in stripped for fragment in ("사진=", "전자공시시스템", "따르면")):
        return True
    if len(stripped) > 42 and stripped.endswith(
        ("다", "다.", "했다", "했다.", "됐다", "됐다.", "있다", "있다.")
    ):
        return True
    return _is_financial_like_text(stripped)


def _card_id_date_key(card_id: str) -> str:
    match = re.match(r"^CN-(\d{8})-", str(card_id or ""))
    return match.group(1) if match else ""


def _has_displayable_implication(card: dict[str, Any]) -> bool:
    implication = card.get("implication")
    if not isinstance(implication, dict):
        return False
    frontend = implication.get("frontend")
    if not isinstance(frontend, dict):
        return False
    key_items = _display_item_list(frontend.get("key_implication_items")) or _display_item_list(
        frontend.get("key_implication_blocks")
    )
    action_items = _display_item_list(frontend.get("suggested_action_items")) or _display_item_list(
        frontend.get("response_direction_blocks")
    )
    return bool(key_items and action_items)


def _has_direct_frontend_ready(card: dict[str, Any]) -> bool:
    implication = card.get("implication")
    if not isinstance(implication, dict):
        return False
    ready = implication.get("frontend_ready")
    if not isinstance(ready, dict):
        return False
    source = str(ready.get("source") or "").strip()
    if source not in {"llm_direct", "frontend_repair_direct"}:
        return False
    key = ready.get("key_implication")
    action = ready.get("suggested_action")
    if not isinstance(key, dict) or not isinstance(action, dict):
        return False
    return all(
        str(block.get(field) or "").strip()
        for block in (key, action)
        for field in ("sentence", "evidence_sentence")
    )


def _display_item_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        main = str(item.get("main") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if main and main != "데이터 없음" and detail and detail != "데이터 없음":
            result.append(item)
    return result


def _mark_existing_cards_deleted(cluster_id: int) -> int:
    with SessionLocal() as db:
        result = db.execute(
            text("""
                UPDATE card_news
                SET status = 'DELETED'
                WHERE status = 'ACTIVE'
                  AND cluster_id = :cluster_id
            """),
            {"cluster_id": cluster_id},
        )
        db.commit()
        return int(getattr(result, "rowcount", 0) or 0)


def _mark_card_deleted(card_id: str) -> int:
    with SessionLocal() as db:
        result = db.execute(
            text("""
                UPDATE card_news
                SET status = 'DELETED'
                WHERE id = :card_id
            """),
            {"card_id": card_id},
        )
        db.commit()
        return int(getattr(result, "rowcount", 0) or 0)


def _mark_other_active_cards_deleted(cluster_id: int, *, keep_card_id: str) -> int:
    with SessionLocal() as db:
        result = db.execute(
            text("""
                UPDATE card_news
                SET status = 'DELETED'
                WHERE status = 'ACTIVE'
                  AND cluster_id = :cluster_id
                  AND id <> :keep_card_id
            """),
            {"cluster_id": cluster_id, "keep_card_id": keep_card_id},
        )
        db.commit()
        return int(getattr(result, "rowcount", 0) or 0)


if __name__ == "__main__":
    main()
