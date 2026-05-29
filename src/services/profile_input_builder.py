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
    "ai_ax": "클라우드&AI",
    "cloud": "클라우드&AI",
    "클라우드&AI": "클라우드&AI",
    "logistics": "스마트 엔지니어링",
    "rx": "스마트 엔지니어링",
    "automation": "스마트팩토리/자동화",
    "스마트엔지니어링": "스마트 엔지니어링",
    "스마트 엔지니어링": "스마트 엔지니어링",
    "Enterprise IT": "Digital Business Service",
    "enterprise_it": "Digital Business Service",
    "it_service": "Digital Business Service",
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

BUSINESS_AREA_CANDIDATES_BY_PEER: dict[str, list[tuple[str, str]]] = {
    "lg_cns": [
        (
            "클라우드&AI",
            "클라우드 전환, MSP, 애플리케이션 현대화, AI 적용을 포함한 AX 지원 사업",
        ),
        (
            "스마트 엔지니어링",
            "제조, 물류, 시티 영역의 스마트팩토리, 스마트물류, 스마트시티 및 RX 관련 사업",
        ),
        (
            "Digital Business Service",
            "SI/SM 기반의 시스템 통합, 운영, 유지보수 및 AI 융합 디지털 서비스",
        ),
    ],
}


class ProfileInputBuilder:
    """Build an agent-friendly profile evidence pack from existing DB tables."""

    def __init__(self, *, session_factory: Any = SessionLocal) -> None:
        self._session_factory = session_factory

    def build(self, peer_id: str) -> dict[str, Any]:
        aliases = expand_peer_aliases(peer_id)
        with self._session_factory() as db:
            company = self._load_company(db, peer_id)
            latest_period = self._load_latest_ir_period(db, aliases)
            business_area_evidence = self._load_dart_business_evidence(db, peer_id)
            financial_evidence = self._load_financial_evidence(db, aliases, latest_period)
            direction_evidence = self._load_direction_evidence(db, aliases, latest_period)
            execution_evidence = self._load_execution_evidence(db, peer_id)

        source_index = _build_source_index(
            [
                *business_area_evidence,
                *financial_evidence,
                *direction_evidence,
                *execution_evidence,
            ]
        )
        return {
            "company": company,
            "period": _period_label(latest_period),
            "business_area_evidence": business_area_evidence,
            "financial_evidence": financial_evidence,
            "direction_evidence": direction_evidence,
            "execution_evidence": execution_evidence,
            "source_index": source_index,
        }

    def _load_company(self, db: Any, peer_id: str) -> dict[str, Any]:
        row = db.execute(
            text(
                """
                SELECT id, name, keywords, core_keywords
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
        row = db.execute(
            text(
                """
                SELECT period_year, period_quarter
                  FROM (
                        SELECT period_year, period_quarter
                          FROM raw_article_financial_metrics
                         WHERE peer_id = ANY(:aliases)
                           AND source_type = 'ir'
                           AND period_year IS NOT NULL
                           AND period_quarter IS NOT NULL
                        UNION
                        SELECT period_year, period_quarter
                          FROM raw_article_business_signals
                         WHERE peer_id = ANY(:aliases)
                           AND source_type = 'ir'
                           AND period_year IS NOT NULL
                           AND period_quarter IS NOT NULL
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
        candidates = BUSINESS_AREA_CANDIDATES_BY_PEER.get(
            peer_id,
            _business_area_candidates_from_text(content),
        )
        evidence: list[dict[str, Any]] = []
        for area, claim in candidates:
            if _area_mentioned(area, content):
                section = _business_area_section(content, area)
                evidence.append(
                    {
                        "business_area": area,
                        "claim": claim,
                        "evidence_text": _compact_text(section or _sentence_around(content, area)),
                        "source_ref": _article_source_ref(article, "raw_articles"),
                        "confidence": 0.9,
                    }
                )
        return evidence

    def _load_financial_evidence(
        self,
        db: Any,
        aliases: list[str],
        latest_period: dict[str, int] | None,
    ) -> list[dict[str, Any]]:
        if latest_period is None:
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
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        if latest_period is None:
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
                   AND COALESCE(bs.confidence, 0) >= 0.7
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

    def _load_execution_evidence(
        self,
        db: Any,
        peer_id: str,
        *,
        limit: int = 8,
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
                    "evidence_text": _compact_text(content),
                    "source_ref": _article_source_ref(data, "raw_articles"),
                    "confidence": 0.85,
                }
            )
        return _select_execution_by_area(candidates, max_items=5)


def normalize_business_area(value: Any) -> str:
    raw = str(value or "").strip()
    return BUSINESS_AREA_ALIASES.get(raw, raw or "company_total")


def _business_area_candidates_from_text(content: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw_area, normalized in BUSINESS_AREA_ALIASES.items():
        if normalized == "company_total" or "|" in raw_area:
            continue
        if _area_mentioned(normalized, content) and normalized not in [area for area, _ in out]:
            out.append((normalized, f"{normalized} 관련 사업"))
    return out[:5]


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


def _infer_execution_area(text_value: str) -> str:
    if any(token in text_value for token in ["챗GPT", "ChatGPT", "구글 클라우드", "Google Cloud"]):
        return "클라우드&AI"
    enterprise_tokens = ["한국전력", "차세대 ISP", "영업배전", "SI", "SM", "ITO"]
    if any(token in text_value for token in enterprise_tokens):
        return "Digital Business Service"
    if any(token in text_value for token in ["Factova", "팩토바", "스마트팩토리", "제조", "물류"]):
        return "스마트 엔지니어링"
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


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


__all__ = ["ProfileInputBuilder", "normalize_business_area"]
