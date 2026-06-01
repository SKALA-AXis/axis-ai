"""ProfileInputBuilder — DB schema-first evidence pack builder.

PeerProfileAgent 의 첫 단계에서 사용하는 입력 조립기다. DB에 이미 구조화된
raw_articles / raw_article_business_signals / raw_article_financial_metrics 를
그대로 LLM에 넘기지 않고, 짧은 claim + source_ref 형태의 evidence pack 으로 바꾼다.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from sqlalchemy import text

from src.db.postgres import SessionLocal
from src.services.peer_id_aliases import expand_peer_aliases

BUSINESS_AREA_ALIASES: dict[str, str] = {
    "ai_ax": "AI/AX",
    "cloud": "클라우드",
    "클라우드&AI": "클라우드&AI",
    "logistics": "물류",
    "rx": "로봇/RX",
    "automation": "스마트팩토리/자동화",
    "스마트엔지니어링": "스마트 엔지니어링",
    "스마트 엔지니어링": "스마트 엔지니어링",
    "Enterprise IT": "IT서비스/SI",
    "enterprise_it": "IT서비스/SI",
    "it_service": "IT서비스/SI",
    "SI": "IT서비스/SI",
    "ITO": "IT서비스/SI",
    "별도 IT": "IT서비스/SI",
    "Digital BusinessService": "Digital Business Service",
    "Digital Business Service": "Digital Business Service",
    "vehicle_sw": "차량 SW",
    "차량 SW": "차량 SW",
    "스마트팩토리/자동화": "스마트팩토리/자동화",
}

FINANCIAL_METRIC_ALLOWLIST = (
    "revenue_total",
    "operating_profit",
    "operating_margin",
    "net_income",
)
SEGMENT_REVENUE_AREAS = (
    "클라우드&AI",
    "스마트엔지니어링",
    "스마트 엔지니어링",
    "Digital Business Service",
)


class ProfileInputBuilder:
    """Build an agent-friendly profile evidence pack from existing DB tables."""

    def __init__(
        self,
        *,
        session_factory: Any = SessionLocal,
        evolution_quarters: int = 4,
        official_evolution_months: int = 24,
        evolution_ir_limit: int = 32,
        evolution_official_limit: int = 12,
    ) -> None:
        self._session_factory = session_factory
        self._evolution_quarters = evolution_quarters
        self._official_evolution_months = official_evolution_months
        self._evolution_ir_limit = evolution_ir_limit
        self._evolution_official_limit = evolution_official_limit

    def build(self, peer_id: str) -> dict[str, Any]:
        aliases = expand_peer_aliases(peer_id)
        with self._session_factory() as db:
            company = self._load_company(db, peer_id)
            latest_period = self._load_latest_ir_period(db, aliases)
            business_area_evidence = self._load_dart_business_evidence(db, peer_id)
            financial_evidence = self._load_financial_evidence(db, aliases, latest_period)
            operational_evidence = self._load_operational_evidence(db, aliases, latest_period)
            direction_evidence = self._load_direction_evidence(db, aliases, latest_period)
            direction_evidence.extend(self._load_ir_document_evidence(db, peer_id))
            execution_evidence = self._load_execution_evidence(db, peer_id)
            evolution_evidence = self._load_evolution_evidence(
                db,
                aliases,
                peer_id,
                latest_period,
                quarters=self._evolution_quarters,
                ir_limit=self._evolution_ir_limit,
                official_months=self._official_evolution_months,
                official_limit=self._evolution_official_limit,
            )
            market_evidence = self._load_market_evidence(db, aliases)
            market_evidence.extend(self._load_market_document_evidence(db, peer_id))
            source_coverage = self._load_source_coverage(db, peer_id, aliases, latest_period)

        source_index = _build_source_index(
            [
                *business_area_evidence,
                *financial_evidence,
                *operational_evidence,
                *direction_evidence,
                *execution_evidence,
                *evolution_evidence,
                *market_evidence,
            ]
        )
        evidence_digest = _build_evidence_digest(
            business_area_evidence=business_area_evidence,
            financial_evidence=financial_evidence,
            operational_evidence=operational_evidence,
            direction_evidence=direction_evidence,
            execution_evidence=execution_evidence,
            evolution_evidence=evolution_evidence,
            market_evidence=market_evidence,
        )
        return {
            "company": company,
            "period": _period_label(latest_period),
            "evidence_digest": evidence_digest,
            "business_area_evidence": business_area_evidence,
            "financial_evidence": financial_evidence,
            "operational_evidence": operational_evidence,
            "direction_evidence": direction_evidence,
            "execution_evidence": execution_evidence,
            "evolution_evidence": evolution_evidence,
            "market_evidence": market_evidence,
            "source_coverage": source_coverage,
            "source_index": source_index,
        }

    def _load_company(self, db: Any, peer_id: str) -> dict[str, Any]:
        core_keywords_expr = (
            "core_keywords"
            if _column_exists(db, "peer_companies", "core_keywords")
            else "ARRAY[]::text[] AS core_keywords"
        )
        row = db.execute(
            text(
                f"""
                SELECT id, name, keywords, {core_keywords_expr}
                  FROM peer_companies
                 WHERE id = :peer_id
                """
            ),
            {"peer_id": peer_id},
        ).first()
        if row is None:
            return {"id": peer_id, "name": peer_id, "keywords": [], "core_keywords": []}
        data = row._mapping
        return {
            "id": str(data.get("id") or peer_id),
            "name": str(data.get("name") or data.get("id") or peer_id),
            "keywords": _list_or_empty(data.get("keywords")),
            "core_keywords": _list_or_empty(data.get("core_keywords")),
        }

    def _load_latest_ir_period(self, db: Any, aliases: list[str]) -> dict[str, int] | None:
        period_queries: list[str] = []
        if _table_exists(db, "raw_article_financial_metrics"):
            period_queries.append(
                """
                SELECT period_year, period_quarter
                  FROM raw_article_financial_metrics
                 WHERE peer_id = ANY(:aliases)
                   AND source_type = 'ir'
                   AND period_year IS NOT NULL
                   AND period_quarter IS NOT NULL
                """
            )
        if _table_exists(db, "raw_article_business_signals"):
            period_queries.append(
                """
                SELECT period_year, period_quarter
                  FROM raw_article_business_signals
                 WHERE peer_id = ANY(:aliases)
                   AND source_type = 'ir'
                   AND period_year IS NOT NULL
                   AND period_quarter IS NOT NULL
                """
            )
        if not period_queries:
            return None
        row = db.execute(
            text(
                f"""
                SELECT period_year, period_quarter
                  FROM (
                        {" UNION ".join(period_queries)}
                  ) p
                 ORDER BY period_year DESC, period_quarter DESC
                 LIMIT 1
                """
            ),
            {"aliases": aliases},
        ).first()
        if row is None:
            return None
        data = row._mapping
        return {
            "period_year": int(data["period_year"]),
            "period_quarter": int(data["period_quarter"]),
        }

    def _load_dart_business_evidence(self, db: Any, peer_id: str) -> list[dict[str, Any]]:
        row = db.execute(
            text(
                """
                SELECT id, source_type, source_name, title, content, url, published_at
                  FROM raw_articles
                 WHERE source_type = 'dart'
                   AND company @> jsonb_build_array(:peer_id)
                   AND COALESCE(content, '') <> ''
                 ORDER BY published_at DESC NULLS LAST, collected_at DESC NULLS LAST, id DESC
                 LIMIT 1
                """
            ),
            {"peer_id": peer_id},
        ).first()
        if row is None:
            return []
        article = row._mapping
        content = str(article.get("content") or "")
        return [
            {
                "business_area": "DART 공식 사업영역 후보",
                "claim": "DART 사업의 내용에서 공식 사업영역을 추출해야 합니다.",
                "evidence_text": _compact_text(_dart_business_section(content), limit=5000),
                "source_ref": _article_source_ref(article, "raw_articles"),
                "confidence": 0.9,
            }
        ]

    def _load_financial_evidence(
        self,
        db: Any,
        aliases: list[str],
        latest_period: dict[str, int] | None,
    ) -> list[dict[str, Any]]:
        if latest_period is None or not _table_exists(db, "raw_article_financial_metrics"):
            return []
        rows = db.execute(
            text(
                """
                SELECT fm.id, fm.raw_article_id, fm.source_type, fm.source_name,
                       fm.peer_id, fm.period, fm.period_year, fm.period_quarter,
                       fm.metric_name, fm.metric_label, fm.metric_scope, fm.business_area,
                       fm.value_numeric, fm.value_krwbn, fm.value_krw, fm.unit, fm.currency,
                       fm.confidence, fm.evidence_text, ra.url, ra.title, ra.published_at
                  FROM raw_article_financial_metrics fm
                  LEFT JOIN raw_articles ra ON ra.id = fm.raw_article_id
                 WHERE fm.peer_id = ANY(:aliases)
                   AND fm.source_type = 'ir'
                   AND fm.period_year = :period_year
                   AND fm.period_quarter = :period_quarter
                   AND (
                        (
                            fm.metric_scope = 'company_total'
                            AND fm.metric_name = ANY(:metric_allowlist)
                        )
                        OR (
                            fm.metric_scope = 'segment'
                            AND fm.metric_name = 'revenue_total'
                            AND fm.business_area = ANY(:segment_areas)
                        )
                   )
                 ORDER BY CASE fm.metric_scope
                            WHEN 'company_total' THEN 1
                            WHEN 'segment' THEN 2
                            ELSE 3
                          END,
                          fm.metric_name,
                          fm.business_area NULLS FIRST,
                          fm.confidence DESC NULLS LAST
                """
            ),
            {
                "aliases": aliases,
                "period_year": latest_period["period_year"],
                "period_quarter": latest_period["period_quarter"],
                "metric_allowlist": list(FINANCIAL_METRIC_ALLOWLIST),
                "segment_areas": list(SEGMENT_REVENUE_AREAS),
            },
        ).fetchall()
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            data = row._mapping
            if _looks_like_bad_operating_profit(data):
                continue
            metric_name = str(data.get("metric_name") or "")
            scope = str(data.get("metric_scope") or "")
            area = normalize_business_area(data.get("business_area"))
            key = (metric_name, scope, area)
            if key in seen:
                continue
            seen.add(key)
            evidence_text = str(data.get("evidence_text") or "")
            out.append(
                {
                    "business_area": area,
                    "source_business_area": data.get("business_area"),
                    "metric": metric_name,
                    "metric_label": data.get("metric_label") or metric_name,
                    "metric_scope": scope,
                    "value": _safe_number(data.get("value_krwbn"), data.get("value_numeric")),
                    "value_krw": _safe_number(data.get("value_krw")),
                    "unit": data.get("unit"),
                    "currency": data.get("currency"),
                    "yoy_pct": _extract_yoy_pct(evidence_text),
                    "yoy_amount": _extract_yoy_amount(evidence_text),
                    "period": data.get("period") or _period_label(latest_period),
                    "evidence_text": _compact_text(evidence_text),
                    "source_ref": _metric_source_ref(data),
                    "confidence": _safe_float(data.get("confidence")),
                }
            )
        return out

    def _load_direction_evidence(
        self,
        db: Any,
        aliases: list[str],
        latest_period: dict[str, int] | None,
        *,
        limit: int = 32,
    ) -> list[dict[str, Any]]:
        if latest_period is None or not _table_exists(db, "raw_article_business_signals"):
            return []
        rows = db.execute(
            text(
                """
                SELECT bs.id, bs.raw_article_id, bs.source_type, bs.source_name,
                       bs.peer_id, bs.period, bs.period_year, bs.period_quarter,
                       bs.business_area, bs.signal_type, bs.sentiment, bs.summary,
                       bs.evidence_text, bs.confidence, ra.url, ra.title, ra.published_at
                  FROM raw_article_business_signals bs
                  LEFT JOIN raw_articles ra ON ra.id = bs.raw_article_id
                 WHERE bs.peer_id = ANY(:aliases)
                   AND bs.source_type = 'ir'
                   AND bs.period_year = :period_year
                   AND bs.period_quarter = :period_quarter
                   AND COALESCE(bs.confidence, 0) >= 0.68
                   AND COALESCE(bs.summary, '') <> ''
                 ORDER BY bs.confidence DESC NULLS LAST,
                          bs.business_area,
                          bs.signal_type,
                          bs.id
                 LIMIT :limit
                """
            ),
            {
                "aliases": aliases,
                "period_year": latest_period["period_year"],
                "period_quarter": latest_period["period_quarter"],
                "limit": int(limit),
            },
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            data = row._mapping
            summary = str(data.get("summary") or "")
            if _looks_like_fragment(summary):
                continue
            out.append(
                {
                    "business_area": normalize_business_area(data.get("business_area")),
                    "source_business_area": data.get("business_area"),
                    "claim": _normalize_spacing(summary),
                    "signal_type": data.get("signal_type"),
                    "sentiment": data.get("sentiment"),
                    "period": data.get("period") or _period_label(latest_period),
                    "evidence_text": _compact_text(data.get("evidence_text") or summary),
                    "source_ref": _signal_source_ref(data),
                    "confidence": _safe_float(data.get("confidence")),
                }
            )
        return out

    def _load_operational_evidence(
        self,
        db: Any,
        aliases: list[str],
        latest_period: dict[str, int] | None,
        *,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        if latest_period is None or not _table_exists(db, "raw_article_financial_metrics"):
            return []
        rows = db.execute(
            text(
                """
                SELECT fm.id, fm.raw_article_id, fm.source_type, fm.source_name,
                       fm.peer_id, fm.period, fm.period_year, fm.period_quarter,
                       fm.metric_name, fm.metric_label, fm.metric_scope, fm.business_area,
                       fm.value_numeric, fm.unit, fm.currency, fm.confidence,
                       fm.evidence_text, ra.url, ra.title, ra.published_at
                  FROM raw_article_financial_metrics fm
                  LEFT JOIN raw_articles ra ON ra.id = fm.raw_article_id
                 WHERE fm.peer_id = ANY(:aliases)
                   AND fm.source_type = 'ir'
                   AND fm.period_year = :period_year
                   AND fm.period_quarter = :period_quarter
                   AND fm.metric_scope = 'segment'
                   AND (
                        fm.metric_name LIKE 'revenue_total%'
                        OR fm.metric_name LIKE 'operating_margin%'
                        OR fm.metric_name LIKE 'operating_profit%'
                   )
                 ORDER BY fm.business_area,
                          CASE
                            WHEN fm.metric_name = 'revenue_total_yoy' THEN 1
                            WHEN fm.metric_name = 'revenue_total_qoq' THEN 2
                            WHEN fm.metric_name = 'revenue_total' THEN 3
                            ELSE 4
                          END,
                          fm.confidence DESC NULLS LAST,
                          fm.id
                 LIMIT :limit
                """
            ),
            {
                "aliases": aliases,
                "period_year": latest_period["period_year"],
                "period_quarter": latest_period["period_quarter"],
                "limit": int(limit),
            },
        ).fetchall()
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            data = row._mapping
            area = normalize_business_area(data.get("business_area"))
            metric_name = str(data.get("metric_name") or "")
            if _looks_like_bad_segment_area(area):
                continue
            key = (area, metric_name)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "evidence_kind": "ir_operational_metric",
                    "business_area": area,
                    "source_business_area": data.get("business_area"),
                    "metric": metric_name,
                    "metric_label": data.get("metric_label") or metric_name,
                    "value": _safe_number(data.get("value_numeric")),
                    "unit": data.get("unit"),
                    "period": data.get("period") or _period_label(latest_period),
                    "claim": _operational_metric_claim(data),
                    "evidence_text": _compact_text(data.get("evidence_text") or ""),
                    "source_ref": _metric_source_ref(data),
                    "confidence": _safe_float(data.get("confidence")),
                }
            )
        return out

    def _load_execution_evidence(
        self,
        db: Any,
        peer_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = db.execute(
            text(
                """
                SELECT id, source_type, source_name, title, content, url,
                       published_at, matched_companies, matched_sectors
                  FROM raw_articles
                 WHERE source_type = 'official'
                   AND (
                        company @> jsonb_build_array(:peer_id)
                        OR matched_companies @> jsonb_build_array(:peer_id)
                   )
                   AND COALESCE(content, '') <> ''
                 ORDER BY published_at DESC NULLS LAST,
                          importance_score DESC NULLS LAST,
                          collected_at DESC NULLS LAST,
                          id DESC
                 LIMIT :limit
                """
            ),
            {"peer_id": peer_id, "limit": int(limit)},
        ).fetchall()
        candidates: list[dict[str, Any]] = []
        for row in rows:
            data = row._mapping
            content = str(data.get("content") or "")
            title = str(data.get("title") or "")
            if _is_financial_press(title, content):
                continue
            candidates.append(
                {
                    "business_area": _infer_execution_area(f"{title} {content}"),
                    "claim": _official_claim(title, content),
                    "evidence_text": _compact_text(content, limit=1200),
                    "source_ref": _article_source_ref(data, "raw_articles"),
                    "confidence": 0.85,
                }
            )
        return _select_execution_by_area(candidates, max_items=12)

    def _load_evolution_evidence(
        self,
        db: Any,
        aliases: list[str],
        peer_id: str,
        latest_period: dict[str, int] | None,
        *,
        quarters: int = 4,
        ir_limit: int = 32,
        official_months: int = 24,
        official_limit: int = 12,
    ) -> list[dict[str, Any]]:
        periods = self._load_recent_ir_periods(db, aliases, latest_period, limit=quarters)
        evidence: list[dict[str, Any]] = []
        if periods and _table_exists(db, "raw_article_business_signals"):
            rows = db.execute(
                text(
                    """
                    SELECT bs.id, bs.raw_article_id, bs.source_type, bs.source_name,
                           bs.peer_id, bs.period, bs.period_year, bs.period_quarter,
                           bs.business_area, bs.signal_type, bs.sentiment, bs.summary,
                           bs.evidence_text, bs.confidence, ra.url, ra.title, ra.published_at
                      FROM raw_article_business_signals bs
                      LEFT JOIN raw_articles ra ON ra.id = bs.raw_article_id
                     WHERE bs.peer_id = ANY(:aliases)
                       AND bs.source_type = 'ir'
                       AND (bs.period_year, bs.period_quarter) IN (
                           SELECT *
                             FROM unnest(:period_years, :period_quarters)
                       )
                       AND COALESCE(bs.confidence, 0) >= 0.68
                       AND COALESCE(bs.summary, '') <> ''
                     ORDER BY bs.period_year DESC,
                              bs.period_quarter DESC,
                              bs.confidence DESC NULLS LAST,
                              bs.business_area,
                              bs.signal_type,
                              bs.id
                     LIMIT :limit
                    """
                ),
                {
                    "aliases": aliases,
                    "period_years": [period["period_year"] for period in periods],
                    "period_quarters": [period["period_quarter"] for period in periods],
                    "limit": int(ir_limit),
                },
            ).fetchall()
            seen: set[tuple[str, str, str]] = set()
            for row in rows:
                data = row._mapping
                summary = str(data.get("summary") or "")
                if _looks_like_fragment(summary):
                    continue
                area = normalize_business_area(data.get("business_area"))
                key = (str(data.get("period") or ""), area, _normalize_spacing(summary))
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    {
                        "evidence_kind": "ir_signal",
                        "business_area": area,
                        "source_business_area": data.get("business_area"),
                        "claim": _normalize_spacing(summary),
                        "signal_type": data.get("signal_type"),
                        "sentiment": data.get("sentiment"),
                        "period": data.get("period") or _period_label(data),
                        "period_year": data.get("period_year"),
                        "period_quarter": data.get("period_quarter"),
                        "evidence_text": _compact_text(data.get("evidence_text") or summary),
                        "source_ref": _signal_source_ref(data),
                        "confidence": _safe_float(data.get("confidence")),
                    }
                )

        official_rows = db.execute(
            text(
                """
                SELECT id, source_type, source_name, title, content, url,
                       published_at, matched_companies, matched_sectors
                  FROM raw_articles
                 WHERE source_type = 'official'
                   AND (
                        company @> jsonb_build_array(:peer_id)
                        OR matched_companies @> jsonb_build_array(:peer_id)
                   )
                   AND COALESCE(content, '') <> ''
                   AND published_at >= (
                        CURRENT_DATE - (:official_months * INTERVAL '1 month')
                   )
                 ORDER BY published_at DESC NULLS LAST,
                          importance_score DESC NULLS LAST,
                          collected_at DESC NULLS LAST,
                          id DESC
                 LIMIT :limit
                """
            ),
            {
                "peer_id": peer_id,
                "official_months": int(official_months),
                "limit": int(official_limit),
            },
        ).fetchall()
        for row in official_rows:
            data = row._mapping
            content = str(data.get("content") or "")
            title = str(data.get("title") or "")
            if _is_financial_press(title, content):
                continue
            evidence.append(
                {
                    "evidence_kind": "official_execution",
                    "business_area": _infer_execution_area(f"{title} {content}"),
                    "claim": _official_claim(title, content),
                    "period": _published_month(data.get("published_at")),
                    "evidence_text": _compact_text(content, limit=1200),
                    "source_ref": _article_source_ref(data, "raw_articles"),
                    "confidence": 0.82,
                }
            )
        return evidence

    def _load_ir_document_evidence(
        self,
        db: Any,
        peer_id: str,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        rows = db.execute(
            text(
                """
                SELECT id, source_type, source_name, title, content, url, published_at
                  FROM raw_articles
                 WHERE source_type = 'ir'
                   AND company @> jsonb_build_array(:peer_id)
                   AND COALESCE(content, '') <> ''
                 ORDER BY published_at DESC NULLS LAST,
                          collected_at DESC NULLS LAST,
                          id DESC
                 LIMIT :limit
                """
            ),
            {"peer_id": peer_id, "limit": int(limit)},
        ).fetchall()
        evidence: list[dict[str, Any]] = []
        for row in rows:
            data = row._mapping
            content = str(data.get("content") or "")
            title = str(data.get("title") or "")
            evidence.append(
                {
                    "evidence_kind": "ir_document",
                    "business_area": _infer_execution_area(f"{title} {content}"),
                    "claim": _ir_document_claim(title, content),
                    "period": _published_quarter(data.get("published_at")),
                    "evidence_text": _compact_text(content, limit=2400),
                    "source_ref": _article_source_ref(data, "raw_articles"),
                    "confidence": 0.78,
                }
            )
        return evidence

    def _load_recent_ir_periods(
        self,
        db: Any,
        aliases: list[str],
        latest_period: dict[str, int] | None,
        *,
        limit: int,
    ) -> list[dict[str, int]]:
        if latest_period is None or not _table_exists(db, "raw_article_business_signals"):
            return []
        rows = db.execute(
            text(
                """
                SELECT DISTINCT period_year, period_quarter
                  FROM raw_article_business_signals
                 WHERE peer_id = ANY(:aliases)
                   AND source_type = 'ir'
                   AND period_year IS NOT NULL
                   AND period_quarter IS NOT NULL
                   AND (
                        period_year < :latest_year
                        OR (
                            period_year = :latest_year
                            AND period_quarter <= :latest_quarter
                        )
                   )
                 ORDER BY period_year DESC, period_quarter DESC
                 LIMIT :limit
                """
            ),
            {
                "aliases": aliases,
                "latest_year": latest_period["period_year"],
                "latest_quarter": latest_period["period_quarter"],
                "limit": int(limit),
            },
        ).fetchall()
        return [
            {
                "period_year": int(row._mapping["period_year"]),
                "period_quarter": int(row._mapping["period_quarter"]),
            }
            for row in rows
        ]

    def _load_source_coverage(
        self,
        db: Any,
        peer_id: str,
        aliases: list[str],
        latest_period: dict[str, int] | None,
    ) -> dict[str, Any]:
        article_rows = db.execute(
            text(
                """
                SELECT source_type, COUNT(*) AS document_count
                  FROM raw_articles
                 WHERE company @> jsonb_build_array(:peer_id)
                   AND source_type IN ('dart', 'ir', 'official', 'securities_report')
                 GROUP BY source_type
                """
            ),
            {"peer_id": peer_id},
        ).fetchall()
        document_counts = {
            str(row._mapping.get("source_type")): int(row._mapping.get("document_count") or 0)
            for row in article_rows
        }

        ir_business_signal_count = _count_source_rows(
            db, "raw_article_business_signals", aliases, "ir"
        )
        ir_financial_metric_count = _count_source_rows(
            db, "raw_article_financial_metrics", aliases, "ir"
        )

        return {
            "dart": {
                "available": document_counts.get("dart", 0) > 0,
                "document_count": document_counts.get("dart", 0),
            },
            "ir": {
                "available": (ir_business_signal_count > 0 or ir_financial_metric_count > 0),
                "document_count": document_counts.get("ir", 0),
                "business_signal_count": ir_business_signal_count,
                "financial_metric_count": ir_financial_metric_count,
                "latest_period": _period_label(latest_period),
            },
            "official_newsroom": {
                "available": document_counts.get("official", 0) > 0,
                "document_count": document_counts.get("official", 0),
            },
            "securities_report": self._load_securities_report_coverage(db, peer_id, aliases),
        }

    def _load_securities_report_coverage(
        self,
        db: Any,
        peer_id: str,
        aliases: list[str],
    ) -> dict[str, Any]:
        report_row = db.execute(
            text(
                """
                SELECT
                    COUNT(*) AS document_count,
                    COUNT(*) FILTER (
                        WHERE published_at >= (CURRENT_DATE - INTERVAL '3 years')
                    ) AS recent_document_count,
                    MIN(published_at) AS first_published_at,
                    MAX(published_at) AS latest_published_at
                  FROM raw_articles
                 WHERE source_type = 'securities_report'
                   AND company @> jsonb_build_array(:peer_id)
                """
            ),
            {"peer_id": peer_id},
        ).first()
        report_counts = report_row._mapping if report_row is not None else {}

        business_signal_count = _count_source_rows(
            db, "raw_article_business_signals", aliases, "securities_report"
        )
        financial_metric_count = _count_source_rows(
            db, "raw_article_financial_metrics", aliases, "securities_report"
        )

        recent_document_count = int(report_counts.get("recent_document_count") or 0)
        available = recent_document_count > 0
        coverage: dict[str, Any] = {
            "available": available,
            "document_count": int(report_counts.get("document_count") or 0),
            "recent_document_count": recent_document_count,
            "business_signal_count": business_signal_count,
            "financial_metric_count": financial_metric_count,
            "first_published_at": _iso(report_counts.get("first_published_at")),
            "latest_published_at": _iso(report_counts.get("latest_published_at")),
            "lookback_years": 3,
            "status": "available" if available else "unavailable",
        }
        if not available:
            coverage["reason"] = "최근 3년 내 증권사 리포트 없음"
        return coverage

    def _load_market_evidence(
        self,
        db: Any,
        aliases: list[str],
        *,
        signal_limit: int = 24,
        metric_limit: int = 8,
    ) -> list[dict[str, Any]]:
        signal_rows = []
        if _table_exists(db, "raw_article_business_signals"):
            signal_rows = db.execute(
                text(
                    """
                SELECT bs.id, bs.raw_article_id, bs.source_type, bs.source_name,
                       bs.peer_id, bs.period, bs.period_year, bs.period_quarter,
                       bs.business_area, bs.signal_type, bs.sentiment, bs.summary,
                       bs.evidence_text, bs.confidence, ra.url, ra.title, ra.published_at
                  FROM raw_article_business_signals bs
                  LEFT JOIN raw_articles ra ON ra.id = bs.raw_article_id
                 WHERE bs.peer_id = ANY(:aliases)
                   AND bs.source_type = 'securities_report'
                   AND ra.published_at >= (CURRENT_DATE - INTERVAL '3 years')
                   AND COALESCE(bs.confidence, 0) >= 0.68
                   AND COALESCE(bs.summary, '') <> ''
                   AND bs.signal_type IN (
                        'risk',
                        'forecast',
                        'valuation',
                        'orders_pipeline',
                        'growth',
                        'strategy'
                   )
                 ORDER BY bs.period_year DESC NULLS LAST,
                          bs.period_quarter DESC NULLS LAST,
                          CASE bs.signal_type
                            WHEN 'risk' THEN 1
                            WHEN 'valuation' THEN 2
                            WHEN 'forecast' THEN 3
                            WHEN 'orders_pipeline' THEN 4
                            ELSE 5
                          END,
                          bs.confidence DESC NULLS LAST,
                          ra.published_at DESC NULLS LAST,
                          bs.id DESC
                 LIMIT :limit
                """
                ),
                {"aliases": aliases, "limit": int(signal_limit)},
            ).fetchall()

        evidence: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in signal_rows:
            data = row._mapping
            summary = str(data.get("summary") or "")
            if _looks_like_fragment(summary) or _looks_like_market_fragment(summary):
                continue
            key = (
                str(data.get("raw_article_id") or ""),
                str(data.get("signal_type") or ""),
                _normalize_spacing(summary)[:120],
            )
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "evidence_kind": "securities_report_signal",
                    "business_area": normalize_business_area(data.get("business_area")),
                    "source_business_area": data.get("business_area"),
                    "claim": _normalize_spacing(summary),
                    "signal_type": data.get("signal_type"),
                    "sentiment": data.get("sentiment"),
                    "period": data.get("period"),
                    "evidence_text": _compact_text(data.get("evidence_text") or summary),
                    "source_ref": _signal_source_ref(data),
                    "confidence": _safe_float(data.get("confidence")),
                }
            )

        metric_rows = []
        if _table_exists(db, "raw_article_financial_metrics"):
            metric_rows = db.execute(
                text(
                    """
                SELECT fm.id, fm.raw_article_id, fm.source_type, fm.source_name,
                       fm.peer_id, fm.period, fm.metric_name, fm.metric_label,
                       fm.value_numeric, fm.unit, fm.currency, fm.confidence,
                       fm.evidence_text, ra.url, ra.title, ra.published_at
                  FROM raw_article_financial_metrics fm
                  LEFT JOIN raw_articles ra ON ra.id = fm.raw_article_id
                 WHERE fm.peer_id = ANY(:aliases)
                   AND fm.source_type = 'securities_report'
                   AND ra.published_at >= (CURRENT_DATE - INTERVAL '3 years')
                   AND fm.metric_name IN ('target_price', 'current_price', 'upside_pct')
                 ORDER BY ra.published_at DESC NULLS LAST,
                          fm.confidence DESC NULLS LAST,
                          fm.id DESC
                 LIMIT :limit
                """
                ),
                {"aliases": aliases, "limit": int(metric_limit)},
            ).fetchall()
        for row in metric_rows:
            data = row._mapping
            evidence.append(
                {
                    "evidence_kind": "securities_report_metric",
                    "business_area": "company_total",
                    "metric": data.get("metric_name"),
                    "metric_label": data.get("metric_label"),
                    "value": _safe_number(data.get("value_numeric")),
                    "unit": data.get("unit"),
                    "currency": data.get("currency"),
                    "period": data.get("period"),
                    "evidence_text": _compact_text(data.get("evidence_text") or ""),
                    "source_ref": _metric_source_ref(data),
                    "confidence": _safe_float(data.get("confidence")),
                }
            )
        return evidence

    def _load_market_document_evidence(
        self,
        db: Any,
        peer_id: str,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        rows = db.execute(
            text(
                """
                SELECT id, source_type, source_name, title, content, url, published_at
                  FROM raw_articles
                 WHERE source_type = 'securities_report'
                   AND company @> jsonb_build_array(:peer_id)
                   AND COALESCE(content, '') <> ''
                   AND published_at >= (CURRENT_DATE - INTERVAL '3 years')
                 ORDER BY published_at DESC NULLS LAST,
                          collected_at DESC NULLS LAST,
                          id DESC
                 LIMIT :limit
                """
            ),
            {"peer_id": peer_id, "limit": int(limit)},
        ).fetchall()
        evidence: list[dict[str, Any]] = []
        for row in rows:
            data = row._mapping
            content = str(data.get("content") or "")
            title = str(data.get("title") or "")
            evidence.append(
                {
                    "evidence_kind": "securities_report_document",
                    "business_area": _infer_execution_area(f"{title} {content}"),
                    "claim": _market_document_claim(title, content),
                    "period": _published_quarter(data.get("published_at")),
                    "evidence_text": _compact_text(content, limit=1600),
                    "source_ref": _article_source_ref(data, "raw_articles"),
                    "confidence": 0.68,
                }
            )
        return evidence


def normalize_business_area(value: Any) -> str:
    raw = str(value or "").strip()
    return BUSINESS_AREA_ALIASES.get(raw, raw or "company_total")


def _build_evidence_digest(
    *,
    business_area_evidence: list[dict[str, Any]],
    financial_evidence: list[dict[str, Any]],
    operational_evidence: list[dict[str, Any]],
    direction_evidence: list[dict[str, Any]],
    execution_evidence: list[dict[str, Any]],
    evolution_evidence: list[dict[str, Any]],
    market_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _digest_entry(
            source="dart",
            role="공식 사업영역 정의",
            items=business_area_evidence,
            summary=_area_digest_summary(business_area_evidence),
        ),
        _digest_entry(
            source="ir",
            role="회사 발표 전략/역량 변화",
            items=[*direction_evidence, *operational_evidence, *financial_evidence],
            summary=_signal_digest_summary(direction_evidence, source_label="IR"),
        ),
        _digest_entry(
            source="ir_operational_metrics",
            role="최신 IR 부문별 운영/실적 신호",
            items=operational_evidence,
            summary=_operational_digest_summary(operational_evidence),
        ),
        _digest_entry(
            source="official_newsroom",
            role="공식 실행 사례",
            items=execution_evidence,
            summary=_signal_digest_summary(execution_evidence, source_label="공식 뉴스룸"),
        ),
        _digest_entry(
            source="capability_evolution",
            role="최근 2~3년 역량 변화 흐름",
            items=evolution_evidence,
            summary=_evolution_digest_summary(evolution_evidence),
        ),
        _digest_entry(
            source="securities_report",
            role="애널리스트 시장 관점",
            items=market_evidence,
            summary=_market_digest_summary(market_evidence),
        ),
    ]


def _digest_entry(
    *,
    source: str,
    role: str,
    items: list[dict[str, Any]],
    summary: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "role": role,
        "item_count": len(items),
        "period": _digest_period(items),
        "summary": summary,
        "top_refs": _digest_refs(items),
    }


def _area_digest_summary(items: list[dict[str, Any]]) -> str:
    areas = _unique_str(item.get("business_area") for item in items)
    if areas == ["DART 공식 사업영역 후보"]:
        inferred: list[str] = []
        for item in items:
            inferred.extend(
                area
                for area, _ in _business_area_candidates_from_text(
                    str(item.get("evidence_text") or "")
                )
            )
        areas = _unique_str(inferred)
    if not areas:
        return "DART 기반 사업영역 근거가 제한적입니다."
    return f"공식 사업영역은 {', '.join(areas[:5])} 중심으로 확인됩니다."


def _signal_digest_summary(items: list[dict[str, Any]], *, source_label: str) -> str:
    areas = _unique_str(item.get("business_area") for item in items if item.get("business_area"))
    signal_types = _unique_str(item.get("signal_type") for item in items if item.get("signal_type"))
    if not items:
        return f"{source_label} 기반 신호가 제한적입니다."
    area_text = ", ".join(areas[:4]) if areas else "회사 전체"
    type_text = ", ".join(signal_types[:4]) if signal_types else "실행/변화"
    return f"{source_label}에서 {area_text} 영역의 {type_text} 신호가 확인됩니다."


def _evolution_digest_summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return "최근 역량 변화 신호가 제한적입니다."
    periods = _digest_period(items)
    areas = _unique_str(item.get("business_area") for item in items if item.get("business_area"))
    kinds = _unique_str(item.get("evidence_kind") for item in items if item.get("evidence_kind"))
    return (
        f"{periods.get('from')}~{periods.get('to')} 기간에 "
        f"{', '.join(areas[:4])} 영역의 변화가 {', '.join(kinds[:3])} 근거로 확인됩니다."
    )


def _operational_digest_summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return "최신 IR의 부문별 운영/실적 신호가 제한적입니다."
    areas = _unique_str(item.get("business_area") for item in items if item.get("business_area"))
    metrics = _unique_str(item.get("metric") for item in items if item.get("metric"))
    return (
        f"최신 IR에서 {', '.join(areas[:4])} 영역의 "
        f"{', '.join(metrics[:4])} 운영 지표가 확인됩니다."
    )


def _market_digest_summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return "증권사 리포트 기반 시장 관점 근거가 제한적입니다."
    signals = _unique_str(item.get("signal_type") for item in items if item.get("signal_type"))
    metrics = _unique_str(item.get("metric_label") or item.get("metric") for item in items)
    parts = []
    if signals:
        parts.append(f"신호: {', '.join(signals[:4])}")
    if metrics:
        parts.append(f"수치: {', '.join(metrics[:4])}")
    return "증권사 리포트에서 " + "; ".join(parts) + "가 확인됩니다."


def _digest_period(items: list[dict[str, Any]]) -> dict[str, str | None]:
    periods = sorted(
        {str(item.get("period")) for item in items if item.get("period")},
        key=_period_sort_key,
    )
    if not periods:
        return {"from": None, "to": None}
    return {"from": periods[0], "to": periods[-1]}


def _digest_refs(items: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for item in items:
        ref = item.get("source_ref")
        if not isinstance(ref, dict):
            continue
        compact = _compact_source_ref(ref)
        if not compact:
            continue
        key = compact.get("raw_article_id") or (compact.get("table"), compact.get("id"))
        if key in seen:
            continue
        seen.add(key)
        refs.append(compact)
        if len(refs) >= limit:
            break
    return refs


def _compact_source_ref(ref: dict[str, Any]) -> dict[str, Any]:
    table = ref.get("table")
    ref_id = ref.get("id")
    raw_article_id = ref.get("raw_article_id")
    if table == "raw_articles" and ref_id is not None:
        return {"raw_article_id": ref_id}
    if raw_article_id is not None:
        return {"raw_article_id": raw_article_id}
    if table and ref_id is not None:
        return {"table": table, "id": ref_id}
    return {}


def _unique_str(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text_value = str(value or "").strip()
        if not text_value or text_value in out:
            continue
        out.append(text_value)
    return out


def _period_sort_key(period: str) -> tuple[int, int, int]:
    quarter_match = re.fullmatch(r"(\d{4})Q([1-4])", period)
    if quarter_match:
        return (int(quarter_match.group(1)), int(quarter_match.group(2)) * 3, 99)
    month_match = re.fullmatch(r"(\d{4})-(\d{2})", period)
    if month_match:
        return (int(month_match.group(1)), int(month_match.group(2)), 0)
    return (0, 0, 0)


def _business_area_candidates_from_text(content: str) -> list[tuple[str, str]]:
    official_names = _official_business_area_names(content)
    if official_names:
        return [(name, f"{name} 관련 사업") for name in official_names[:5]]

    out: list[tuple[str, str]] = []
    for raw_area, normalized in BUSINESS_AREA_ALIASES.items():
        if normalized == "company_total" or "|" in raw_area:
            continue
        if _area_mentioned(normalized, content) and normalized not in [area for area, _ in out]:
            out.append((normalized, f"{normalized} 관련 사업"))
    return out[:5]


def _official_business_area_names(content: str) -> list[str]:
    compact = re.sub(r"\s+", " ", content or "")
    names = _names_from_count_pattern(compact)
    if names:
        return names
    names = _names_from_classification_pattern(compact)
    if names:
        return names
    names = _names_from_numbered_headings(compact)
    if names:
        return names
    return []


def _names_from_count_pattern(compact: str) -> list[str]:
    match = re.search(r"주된 사업은\s+(.{2,80}?)의\s*\d+개\s*사업부문", compact)
    if not match:
        return []
    return _split_business_area_names(match.group(1))


def _names_from_classification_pattern(compact: str) -> list[str]:
    patterns = (
        r"주된 사업은\s+.{1,40}?(?:로|으로)\s+(.{2,140}?)(?:로|으로)\s*구분",
        r"사업부문은\s+(?:.{1,50}?에 따라\s+)?(.{2,160}?)(?:로|으로)\s*구분",
    )
    match = None
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            break
    if not match:
        return []
    return _split_business_area_names(match.group(1))


def _split_business_area_names(value: str) -> list[str]:
    value = re.sub(r"\([^)]*\)", "", value or "")
    raw_names = re.split(r"\s*(?:,|와|과|및|/)\s*", value)
    return _unique_str(_clean_business_area_name(name) for name in raw_names)


def _names_from_numbered_headings(compact: str) -> list[str]:
    overview = re.split(r"\s*2\.\s*주요\s*제품", compact, maxsplit=1)[0]
    names: list[str] = []
    for match in re.finditer(r"\([0-9]+\)\s*(.{1,30}?(?:부문|사업))(?=\s)", overview):
        names.append(_clean_business_area_name(match.group(1)))
    return _unique_str(names)


def _clean_business_area_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip(" .:;·-")
    cleaned = re.sub(r"\s*부문$", "", cleaned)
    cleaned = re.sub(r"\s*사업$", "", cleaned)
    cleaned = cleaned.replace("Digital BusinessService", "Digital Business Service")
    if cleaned.startswith("Digital Business Service"):
        return "Digital Business Service"
    return cleaned


def _dart_business_section(content: str) -> str:
    compact = re.sub(r"\s+", " ", content or "").strip()
    if not compact:
        return ""
    start_patterns = [
        r"II\.\s*사업의\s*내용",
        r"1\.\s*사업의\s*개요",
        r"가\.\s*주요\s*제품",
    ]
    end_patterns = [
        r"2\.\s*주요\s*제품",
        r"3\.\s*원재료",
        r"4\.\s*매출",
        r"III\.\s*재무",
    ]
    start = 0
    for pattern in start_patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            start = match.start()
            break
    end = min(len(compact), start + 8000)
    for pattern in end_patterns:
        match = re.search(pattern, compact[start + 500 :], re.IGNORECASE)
        if match:
            end = min(end, start + 500 + match.start())
    return compact[start:end].strip()


def _article_source_ref(row: Any, table: str) -> dict[str, Any]:
    return {
        "table": table,
        "id": _safe_int(row.get("id")),
        "source_type": row.get("source_type"),
        "source_name": row.get("source_name"),
        "title": row.get("title"),
        "published_at": _iso(row.get("published_at")),
        "url": row.get("url"),
    }


def _metric_source_ref(row: Any) -> dict[str, Any]:
    return {
        "table": "raw_article_financial_metrics",
        "id": _safe_int(row.get("id")),
        "raw_article_id": _safe_int(row.get("raw_article_id")),
        "source_type": row.get("source_type"),
        "source_name": row.get("source_name"),
        "published_at": _iso(row.get("published_at")),
        "url": row.get("url"),
    }


def _signal_source_ref(row: Any) -> dict[str, Any]:
    return {
        "table": "raw_article_business_signals",
        "id": _safe_int(row.get("id")),
        "raw_article_id": _safe_int(row.get("raw_article_id")),
        "source_type": row.get("source_type"),
        "source_name": row.get("source_name"),
        "published_at": _iso(row.get("published_at")),
        "url": row.get("url"),
    }


def _build_source_index(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int | None], dict[str, Any]] = {}
    for item in evidence_items:
        raw_ref = item.get("source_ref")
        ref: dict[str, Any] = raw_ref if isinstance(raw_ref, dict) else {}
        key = (str(ref.get("table") or ""), _safe_int(ref.get("id")))
        if not key[0]:
            continue
        by_key.setdefault(
            key,
            {
                "table": key[0],
                "id": key[1],
                "raw_article_id": ref.get("raw_article_id"),
                "source_type": ref.get("source_type"),
                "source_name": ref.get("source_name"),
                "title": ref.get("title"),
                "published_at": ref.get("published_at"),
                "url": ref.get("url"),
            },
        )
    return list(by_key.values())


def _area_mentioned(area: str, text_value: str) -> bool:
    aliases = {
        "클라우드&AI": ["클라우드&AI", "클라우드", "AX"],
        "스마트 엔지니어링": [
            "스마트 엔지니어링",
            "스마트엔지니어링",
            "스마트팩토리",
            "스마트물류",
            "RX",
        ],
        "Digital Business Service": [
            "Digital Business Service",
            "Digital BusinessService",
            "SI/SM",
        ],
        "차량 SW": [
            "차량 소프트웨어",
            "차량SW",
            "차량 SW",
            "SDV",
            "내비게이션",
            "모빌리티",
        ],
        "스마트팩토리/자동화": [
            "스마트팩토리",
            "자동화",
            "로봇",
            "제어",
            "산업 DX",
        ],
    }.get(area, [area])
    return any(alias in text_value for alias in aliases)


def _sentence_around(text_value: str, keyword: str) -> str:
    compact = re.sub(r"\s+", " ", text_value or "")
    idx = compact.find(keyword)
    if idx < 0 and keyword == "스마트 엔지니어링":
        idx = compact.find("스마트엔지니어링")
    if idx < 0:
        return ""
    start = max(0, idx - 180)
    end = min(len(compact), idx + 360)
    return compact[start:end].strip()


def _business_area_section(text_value: str, area: str) -> str:
    compact = re.sub(r"\s+", " ", text_value or "")
    starts = {
        "클라우드&AI": [
            r"\(1\)\s*클라우드&AI",
            r"①\s*클라우드\s*서비스",
            r"클라우드&AI\s+기업",
        ],
        "스마트 엔지니어링": [
            r"\(2\)\s*스마트\s*엔지니어링",
            r"스마트엔지니어링은",
            r"스마트\s*엔지니어링\s+당사는",
        ],
        "Digital Business Service": [
            r"\(3\)\s*Digital\s*Business\s*Service",
            r"Digital\s*Business\s*Service는",
            r"Digital\s*Business\s*Service\s*\(SI/SM\)",
        ],
        "차량 SW": [
            r"차량\s*소프트웨어",
            r"차량SW",
            r"SDV",
            r"내비게이션",
        ],
        "스마트팩토리/자동화": [
            r"스마트팩토리",
            r"자동화",
            r"로봇",
        ],
    }.get(area, [re.escape(area)])
    boundaries = [
        r"\(1\)\s*클라우드&AI",
        r"\(2\)\s*스마트\s*엔지니어링",
        r"\(3\)\s*Digital\s*Business\s*Service",
        r"2\.\s*주요\s*제품\s*및\s*서비스",
    ]
    start = -1
    for pattern in starts:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            start = match.start()
            break
    if start < 0:
        return ""

    end = len(compact)
    for pattern in boundaries:
        match = re.search(pattern, compact[start + 20 :], re.IGNORECASE)
        if match:
            end = min(end, start + 20 + match.start())
    return compact[start:end].strip()


def _official_claim(title: str, content: str) -> str:
    text_value = f"{title} {content}"
    if any(token in text_value for token in ["챗GPT", "ChatGPT", "구글 클라우드", "Google Cloud"]):
        return "글로벌 AI·클라우드 파트너십을 기반으로 기업 AX 지원 역량을 확대하고 있다."
    if "휴머노이드" in text_value or "로봇" in text_value or "피지컬웍스" in text_value:
        return "로봇 학습·운영 플랫폼과 물류 로봇 사례를 통해 RX 실행 역량을 확대하고 있다."
    if "한국전력" in text_value or "차세대 ISP" in text_value:
        return (
            "공공·에너지 고객의 차세대 ISP 사업을 통해 엔터프라이즈 IT 수행 역량을 보강하고 있다."
        )
    if "Factova" in text_value or "팩토바" in text_value:
        return "Factova 기반 스마트팩토리 솔루션을 앞세워 제조 AX 실행 사례를 확대하고 있다."
    if "북미" in text_value and "제조" in text_value:
        return "북미 제조 AX 시장 공략을 공식 실행 사례로 제시하고 있다."
    return _compact_text(title or content, limit=160)


def _ir_document_claim(title: str, content: str) -> str:
    text_value = f"{title} {content}"
    if "중점 추진 사업" in text_value or "AI Full Stack" in text_value:
        return "IR 자료에서 AI Full Stack 기반 중점 추진 사업과 성장 전략을 제시하고 있다."
    if "투자" in text_value and ("로드맵" in text_value or "Inorganic" in text_value):
        return "IR 자료에서 투자 로드맵과 Inorganic 성장 방향을 제시하고 있다."
    if "클라우드" in text_value or "AX" in text_value or "AI" in text_value:
        return "IR 자료에서 클라우드·AI·AX 중심의 사업 방향과 실행 신호를 제시하고 있다."
    return _compact_text(title or content, limit=180)


def _market_document_claim(title: str, content: str) -> str:
    text_value = f"{title} {content}"
    if "목표주가" in text_value or "상승여력" in text_value:
        return "증권사 리포트에서 목표주가·상승여력 등 시장 관점의 보조 신호를 제시하고 있다."
    if "수주" in text_value or "전망" in text_value:
        return "증권사 리포트에서 수주, 성장 전망, 리스크 등 외부 관찰 신호를 제시하고 있다."
    return _compact_text(title or content, limit=180)


def _infer_execution_area(text_value: str) -> str:
    if any(token in text_value for token in ["챗GPT", "ChatGPT", "구글 클라우드", "Google Cloud"]):
        return "클라우드&AI"
    if any(token in text_value for token in ["Factova", "팩토바", "스마트팩토리", "제조", "물류"]):
        return "스마트 엔지니어링"
    if any(token in text_value for token in ["로봇", "피지컬웍스", "RX", "모바일 셔틀"]):
        return "스마트 엔지니어링"
    enterprise_tokens = [
        "한국전력",
        "차세대 ISP",
        "영업배전",
        "SI/SM",
        "System Integration",
        "시스템 통합",
        "ITO",
    ]
    if any(token in text_value for token in enterprise_tokens):
        return "Digital Business Service"
    cloud_tokens = [
        "AgenticWorks",
        "클라우드",
        "AX",
        "AI 플랫폼",
        "챗GPT",
        "ChatGPT",
        "구글 클라우드",
        "Google Cloud",
    ]
    if any(token in text_value for token in cloud_tokens):
        return "클라우드&AI"
    if "금융" in text_value:
        return "Digital Business Service"
    return "company_total"


def _is_financial_press(title: str, content: str) -> bool:
    del content
    text_value = title
    financial_tokens = ["매출", "영업이익", "실적", "주가", "목표가"]
    return any(token in text_value for token in financial_tokens)


def _select_execution_by_area(
    items: list[dict[str, Any]],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_area_count: dict[str, int] = {}

    for item in items:
        area = str(item.get("business_area") or "company_total")
        if area == "company_total":
            continue
        if per_area_count.get(area, 0) >= 2:
            continue
        selected.append(item)
        per_area_count[area] = per_area_count.get(area, 0) + 1
        if len(selected) >= max_items:
            return selected

    if len(selected) < min(3, max_items):
        for item in items:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= min(3, max_items):
                break
    return selected


def _extract_yoy_pct(text_value: str) -> float | None:
    match = re.search(r"YoY\s*([+-]?\d+(?:\.\d+)?)\s*%?", text_value or "", re.IGNORECASE)
    if not match:
        match = re.search(r"전년\s*대비[^\d+-]*([+-]?\d+(?:\.\d+)?)\s*%", text_value or "")
    return float(match.group(1)) if match else None


def _extract_yoy_amount(text_value: str) -> float | None:
    match = re.search(r"전년\s*대비\s*매출\s*([0-9,]+)\s*억원\s*증가", text_value or "")
    if not match:
        match = re.search(r"전년대비매출\s*([0-9,]+)\s*억원\s*증가", text_value or "")
    return float(match.group(1).replace(",", "")) if match else None


def _looks_like_bad_operating_profit(row: Any) -> bool:
    if str(row.get("metric_name") or "") != "operating_profit":
        return False
    evidence = str(row.get("evidence_text") or "")
    return "[영업이익] [매출] 당기순이익" in evidence


def _looks_like_fragment(text_value: str) -> bool:
    stripped = str(text_value or "").strip()
    if len(stripped) < 18:
        return True
    if re.match(r"^[0-9,]+\s*억원\s+[0-9,]+\s*억원", stripped):
        return True
    if stripped.startswith("확보/"):
        return True
    return False


def _looks_like_market_fragment(text_value: str) -> bool:
    stripped = _normalize_spacing(text_value)
    if len(stripped) < 24:
        return True
    lowered = stripped.lower()
    if any(
        token in stripped
        for token in (
            "[도표",
            "도표 ",
            "자료:",
            "자료 :",
            "추정재무제표",
            "valuation 지표",
            "확정치는",
        )
    ):
        return True
    if re.match(r"^[()+\-\d.,%\s]+은\s", stripped):
        return True
    if stripped.startswith(("속될", "로젝트", "대 등으로", "멘텀은", "현장에서", "프로젝트를")):
        return True
    if lowered.startswith(("opm ", "per ", "pbr ", "eps ")):
        return True
    return False


def _operational_metric_claim(data: Any) -> str:
    area = normalize_business_area(data.get("business_area"))
    metric = str(data.get("metric_name") or "")
    value = _safe_number(data.get("value_numeric"))
    unit = str(data.get("unit") or "")
    period = str(data.get("period") or "")
    label = str(data.get("metric_label") or metric)
    direction = ""
    if isinstance(value, (int, float)):
        if value > 0:
            direction = "증가"
        elif value < 0:
            direction = "감소"
    if metric.endswith("_yoy"):
        suffix = "" if "YoY" in label else " YoY"
        return f"{period} {area} {label}{suffix} {value}{unit} {direction}".strip()
    if metric.endswith("_qoq"):
        suffix = "" if "QoQ" in label else " QoQ"
        return f"{period} {area} {label}{suffix} {value}{unit} {direction}".strip()
    return f"{period} {area} {label} {value}{unit}".strip()


def _looks_like_bad_segment_area(area: str) -> bool:
    if not area or area == "company_total":
        return True
    if len(area) > 40:
        return True
    return any(
        token in area
        for token in (
            "달성",
            "힘입어",
            "상승",
            "하락",
            "법인세",
            "판매비",
            "관리비",
            "지배주주",
            "순이익",
        )
    )


def _compact_text(value: Any, *, limit: int = 360) -> str:
    compact = _normalize_spacing(str(value or ""))
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _normalize_spacing(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _period_label(period: dict[str, int] | None) -> str | None:
    if not period:
        return None
    return f"{period['period_year']}Q{period['period_quarter']}"


def _published_month(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m")
    return str(value)[:7] if value else None


def _published_quarter(value: Any) -> str | None:
    if not value:
        return None
    if not isinstance(value, datetime):
        text_value = str(value)
        try:
            year = int(text_value[:4])
            month = int(text_value[5:7])
        except (TypeError, ValueError):
            return text_value[:10]
    else:
        year = value.year
        month = value.month
    quarter = ((month - 1) // 3) + 1
    return f"{year}Q{quarter}"


def _list_or_empty(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _safe_number(*values: Any) -> float | int | None:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        return int(number) if number.is_integer() else number
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _column_exists(db: Any, table_name: str, column_name: str) -> bool:
    try:
        result = db.execute(
            text(
                """
                SELECT 1
                  FROM information_schema.columns
                 WHERE table_schema = 'public'
                   AND table_name = :table_name
                   AND column_name = :column_name
                 LIMIT 1
                """
            ),
            {"table_name": table_name, "column_name": column_name},
        )
        if hasattr(result, "scalar_one_or_none"):
            return bool(result.scalar_one_or_none())
        # Lightweight fake sessions in unit tests do not emulate to_regclass.
        # Treat them as schema-complete and let the query stubs decide the rows.
        if not hasattr(result, "scalar_one") and hasattr(result, "fetchall"):
            return True
        first = result.first() if hasattr(result, "first") else None
        return bool(first)
    except Exception:
        return True


def _table_exists(db: Any, table_name: str) -> bool:
    try:
        result = db.execute(
            text("SELECT to_regclass(:table_name)"),
            {"table_name": f"public.{table_name}"},
        )
        if hasattr(result, "scalar_one_or_none"):
            return bool(result.scalar_one_or_none())
        # Lightweight fake sessions in unit tests do not emulate to_regclass.
        # Treat them as schema-complete and let the query stubs decide the rows.
        if not hasattr(result, "scalar_one") and hasattr(result, "fetchall"):
            return True
        first = result.first() if hasattr(result, "first") else None
        if first is None:
            return False
        value = first[0] if not hasattr(first, "_mapping") else next(iter(first._mapping.values()))
        return bool(value)
    except Exception:
        return True


def _count_source_rows(
    db: Any,
    table_name: str,
    aliases: list[str],
    source_type: str,
) -> int:
    if not _table_exists(db, table_name):
        return 0
    result = db.execute(
        text(
            f"""
            SELECT COUNT(*)
              FROM {table_name}
             WHERE peer_id = ANY(:aliases)
               AND source_type = :source_type
            """
        ),
        {"aliases": aliases, "source_type": source_type},
    )
    if hasattr(result, "scalar_one"):
        return int(result.scalar_one() or 0)
    first = result.first() if hasattr(result, "first") else None
    if first is None:
        return 0
    value = first[0] if not hasattr(first, "_mapping") else next(iter(first._mapping.values()))
    return int(value or 0)


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


__all__ = ["ProfileInputBuilder", "normalize_business_area"]
