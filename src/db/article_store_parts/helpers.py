"""article_store_parts helpers — extracted from facade (move-only)."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any, Optional

from sqlalchemy import text

from src.config.companies import COMPANY_ALIASES
from src.config.company_tiers import DOMESTIC_COMPANY_IDS, SELF_COMPANY_IDS, resolve_company_id
from src.config.global_companies import GLOBAL_COMPANY_IDS
from src.crawler.base import CrawlRunContext, RawArticle
from src.db.article_store_parts._constants import (  # noqa: F401
    _CONTROL_CHAR_RE,
    _DELETE_BUSINESS_SIGNALS_SQL,
    _DELETE_FINANCIAL_METRICS_SQL,
    _ENSURE_MARKET_PRICE_OHLCV_INDEX_SQL,
    _ENSURE_MARKET_PRICE_OHLCV_SQL,
    _GLOBAL_TREND_UPSERT_SQL,
    _INDUSTRY_MARKER_KEYS,
    _INDUSTRY_REPORT_TYPES,
    _INDUSTRY_SOURCE_TYPES,
    _INSERT_CRAWL_RUN_ARTICLE,
    _INSERT_EVIDENCE,
    _INSERT_PIPELINE_LOG,
    _PARSE_RESULT_METADATA_KEYS,
    _SELECT_ARTICLE_ID_BY_URL,
    _SOURCE_METADATA_EXCLUDED_KEYS,
    _UPDATE_COMPANY_ANALYSIS_IF_CHANGED_SQL,
    _UPDATE_DART_CONTENT_IF_BETTER_SQL,
    _UPDATE_IR_CONTENT_IF_BETTER_SQL,
    _UPSERT_BUSINESS_SIGNAL_SQL,
    _UPSERT_FINANCIAL_METRIC_SQL,
    _UPSERT_MARKET_PRICE_OHLCV_SQL,
    DEFAULT_PEER_COMPANY_IDS,
    GLOBAL_NEWSROOM_SOURCE_NAMES,
    GLOBAL_RESEARCH_SOURCE_NAMES,
    INDUSTRY_TREND_COMPANY,
    SK_AX_RAW_SOURCE_NAMES,
)
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)


def _update_dart_content_if_better(
    db,
    *,
    article_id: int,
    article: RawArticle,
    sanitized_content: str,
) -> None:
    if article.source_type != "dart":
        return
    db.execute(
        _UPDATE_DART_CONTENT_IF_BETTER_SQL,
        {
            "id": article_id,
            "content": sanitized_content,
            "content_type": article.content_type,
            "error_message": article.error_message,
            "new_content_length": len(sanitized_content),
        },
    )


def _update_ir_content_if_better(
    db,
    *,
    article_id: int,
    article: RawArticle,
    sanitized_content: str,
) -> None:
    if article.source_type != "ir":
        return
    db.execute(
        _UPDATE_IR_CONTENT_IF_BETTER_SQL,
        {
            "id": article_id,
            "content": sanitized_content,
            "content_type": article.content_type,
            "error_message": article.error_message,
            "collected_at": article.collected_at,
            "new_content_length": len(sanitized_content),
        },
    )


def _update_company_analysis_if_changed(
    db,
    *,
    article_id: int,
    article: RawArticle,
    sanitized_title: str,
    sanitized_content: str,
    source_metadata: str,
) -> None:
    if article.source_type != "company_analysis":
        return
    result = db.execute(
        _UPDATE_COMPANY_ANALYSIS_IF_CHANGED_SQL,
        {
            "id": article_id,
            "url": article.url,
            "title": sanitized_title,
            "content": sanitized_content,
            "content_type": article.content_type,
            "published_at": article.published_at or article.collected_at,
            "collected_at": article.collected_at,
            "metadata": source_metadata,
            "error_message": article.error_message,
        },
    )
    log.info(
        "company_analysis 최신 본문 갱신 | url=%s rows=%d",
        article.url,
        result.rowcount,
    )


def get_articles_by_ids(ids: list[int]) -> list[dict[str, Any]]:
    """raw_articles 테이블에서 ID 목록으로 기사를 조회한다."""
    if not ids:
        return []
    query = text("""
        SELECT raw_articles.id, raw_articles.company, raw_articles.title,
               raw_articles.content, raw_articles.url,
               raw_articles.source_type, raw_articles.content_type,
               raw_articles.publisher, raw_articles.language,
               raw_articles.cluster_id, raw_articles.is_representative,
               raw_articles.processing_status,
               raw_articles.importance_score, raw_articles.importance_level,
               raw_articles.relevance_score, raw_articles.relevance_label,
               raw_articles.relevance_reason,
               raw_articles.matched_companies, raw_articles.matched_sectors,
               raw_articles.matched_sector_details,
               raw_articles.source_name, raw_articles.published_at,
               raw_articles.collected_at,
               raw_articles.metadata,
               COALESCE(pr.raw_result, '{}'::jsonb) AS parser_result,
               COALESCE(pr.financial_record, '{}'::jsonb) AS financial_record,
               COALESCE(pr.warnings, '[]'::jsonb) AS parser_warnings
        FROM raw_articles
        LEFT JOIN raw_article_parse_results pr
            ON pr.raw_article_id = raw_articles.id
        WHERE raw_articles.id = ANY(:ids)
        ORDER BY raw_articles.published_at DESC NULLS LAST,
                 raw_articles.collected_at DESC NULLS LAST,
                 raw_articles.id DESC
    """)
    fallback_query = text("""
        SELECT raw_articles.id, raw_articles.company, raw_articles.title,
               raw_articles.content, raw_articles.url,
               raw_articles.source_type, raw_articles.content_type,
               raw_articles.publisher, raw_articles.language,
               raw_articles.cluster_id, raw_articles.is_representative,
               raw_articles.processing_status,
               raw_articles.importance_score, raw_articles.importance_level,
               raw_articles.relevance_score, raw_articles.relevance_label,
               raw_articles.relevance_reason,
               raw_articles.matched_companies, raw_articles.matched_sectors,
               '{}'::jsonb AS matched_sector_details,
               raw_articles.source_name, raw_articles.published_at,
               raw_articles.collected_at,
               raw_articles.metadata,
               COALESCE(pr.raw_result, '{}'::jsonb) AS parser_result,
               COALESCE(pr.financial_record, '{}'::jsonb) AS financial_record,
               COALESCE(pr.warnings, '[]'::jsonb) AS parser_warnings
        FROM raw_articles
        LEFT JOIN raw_article_parse_results pr
            ON pr.raw_article_id = raw_articles.id
        WHERE raw_articles.id = ANY(:ids)
        ORDER BY raw_articles.published_at DESC NULLS LAST,
                 raw_articles.collected_at DESC NULLS LAST,
                 raw_articles.id DESC
    """)
    with SessionLocal() as db:
        try:
            rows = db.execute(query, {"ids": ids}).fetchall()
        except Exception as exc:
            if not _is_missing_column(exc, "matched_sector_details"):
                raise
            db.rollback()
            rows = db.execute(fallback_query, {"ids": ids}).fetchall()
    return [dict(row._mapping) for row in rows]


def _fetch_dart_rows(
    *,
    select_sql: str,
    where_sql: str,
    order_limit_sql: str,
    params: dict[str, Any],
) -> list[Any]:
    query = text(f"""
        {select_sql}
        WHERE r.source_type = 'dart'
          AND r.crawl_status = 'success'
          {where_sql}
        {order_limit_sql}
    """)
    with SessionLocal() as db:
        return list(db.execute(query, params).fetchall())


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_or_value(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value if value is not None else default


def _merge_source_dicts(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in [*existing, *incoming]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("raw_article_id") or item.get("id") or item.get("url") or "")
        if not key:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        clean_item = dict(item)
        clean_item["index"] = len(merged) + 1
        merged.append(clean_item)

    return merged


def _source_dict_from_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": 0,
        "raw_article_id": int(article["id"]),
        "title": article.get("title") or "",
        "source_name": article.get("source_name") or article.get("publisher") or "",
        "url": article.get("url") or "",
        "published_at": _iso_or_none(article.get("published_at")),
        "collected_at": _iso_or_none(article.get("collected_at")),
    }


def _source_article_dict_from_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(article["id"]),
        "title": article.get("title") or "",
        "url": article.get("url") or "",
        "source_name": article.get("source_name") or "",
        "publisher": article.get("publisher") or "",
        "published_at": _iso_or_none(article.get("published_at")),
        "collected_at": _iso_or_none(article.get("collected_at")),
    }


def _iso_or_none(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


def _filter_articles_for_card_company(
    articles: list[dict[str, Any]],
    card: dict[str, Any],
) -> list[dict[str, Any]]:
    company_id = _card_company_filter_id(card)
    if not company_id:
        return articles
    return [article for article in articles if _article_matches_company(article, company_id)]


def _card_company_filter_id(card: dict[str, Any]) -> str:
    for key in ("peer_company_id", "company", "peer_id"):
        company_id = resolve_company_id(str(card.get(key) or ""))
        if company_id and company_id != INDUSTRY_TREND_COMPANY:
            return company_id
    return ""


def _article_matches_company(article: dict[str, Any], company_id: str) -> bool:
    if not company_id:
        return True
    if company_id in _article_company_ids(article):
        return True
    haystack = _company_match_text(
        f"{article.get('title') or ''} {(article.get('content') or '')[:3000]}"
    )
    return any(alias and alias in haystack for alias in _company_match_aliases(company_id))


def _article_company_ids(article: dict[str, Any]) -> set[str]:
    raw_values: list[Any] = []
    for key in ("company", "matched_companies"):
        value = _json_or_value(article.get(key), [])
        if isinstance(value, list):
            raw_values.extend(value)
        elif isinstance(value, dict):
            raw_values.extend(value.keys())
            raw_values.extend(value.values())
        elif value:
            raw_values.append(value)
    return {resolve_company_id(str(value)) for value in raw_values if str(value or "").strip()}


def _company_match_aliases(company_id: str) -> list[str]:
    values = [company_id, *COMPANY_ALIASES.get(company_id, [])]
    aliases: list[str] = []
    for value in values:
        normalized = _company_match_text(value)
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return aliases


def _company_match_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _ticker_from_market_data_url(url: str) -> str | None:
    match = re.search(r"(?:code=|/item/)(\d{6})(?!\d)", url or "")
    return match.group(1) if match else None


def _parse_market_trade_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def _market_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _market_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except ValueError:
        return None


def _is_self_company_card(card: dict[str, Any]) -> bool:
    for key in ("peer_company_id", "company", "peer_id"):
        company_id = resolve_company_id(str(card.get(key) or ""))
        if company_id in SELF_COMPANY_IDS:
            return True
    return False


def _is_undefined_column_error(exc: Exception) -> bool:
    """psycopg / SQLAlchemy 가 던지는 UndefinedColumn 인지 확인."""
    text_repr = str(exc).lower()
    return "undefinedcolumn" in text_repr or "does not exist" in text_repr


def _resolve_integrated_issue_id(
    card: dict[str, Any], evidence_payload: dict[str, Any]
) -> Optional[str]:
    for value in (
        card.get("integrated_issue_id"),
        evidence_payload.get("integrated_issue_id"),
        (evidence_payload.get("analysis_package") or {}).get("integrated_issue_id")
        if isinstance(evidence_payload.get("analysis_package"), dict)
        else None,
    ):
        text_value = str(value or "").strip()
        if text_value:
            return text_value
    return None


def _card_created_at_param(card: dict[str, Any], source_articles: list[dict[str, Any]]) -> str:
    source_value = _source_basis_datetime(source_articles)
    if source_value:
        return source_value
    value = _date_string_or_none(card.get("created_at"))
    if value:
        return value
    return datetime.now(UTC).isoformat()


def _canonical_card_id(value: Any, created_at: str) -> str:
    raw = str(value)
    match = re.match(r"^(CN-)(\d{8})(-.+)$", raw)
    basis_date = _date_string_or_none(created_at)
    if not match or not basis_date:
        return raw
    return f"{match.group(1)}{basis_date[:10].replace('-', '')}{match.group(3)}"


def _source_basis_datetime(source_articles: list[dict[str, Any]]) -> str | None:
    values: list[datetime] = []
    for article in source_articles:
        for key in ("published_at", "collected_at"):
            parsed = _parse_datetime(article.get(key))
            if parsed is not None:
                values.append(parsed)
                break
    if not values:
        return None
    return min(values).astimezone(UTC).isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _date_string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text_value = str(value).strip()
    return text_value or None


def _is_missing_card_news_articles_table(exc: Exception) -> bool:
    text_repr = str(exc).lower()
    return "card_news_articles" in text_repr and (
        "undefinedtable" in text_repr or "does not exist" in text_repr
    )


def _source_ids_from_sources(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for source in value:
        if not isinstance(source, dict):
            continue
        ids.extend(
            _normalize_int_list(
                source.get("raw_article_id") or source.get("article_id") or source.get("id")
            )
        )
    return _normalize_int_list(ids)


def _source_articles_payload(card: dict[str, Any], source_ids: list[int]) -> list[dict[str, Any]]:
    source_articles = card.get("source_articles")
    if isinstance(source_articles, list) and source_articles:
        payload = [item for item in source_articles if isinstance(item, dict)]
        if any(item.get("published_at") or item.get("collected_at") for item in payload):
            return payload

    if source_ids:
        try:
            articles = get_articles_by_ids(source_ids)
        except Exception as exc:  # noqa: BLE001
            log.debug("source article lookup skipped | ids=%s error=%s", source_ids, exc)
            articles = []
        articles_by_id = {int(article["id"]): article for article in articles}
        if articles_by_id:
            return [
                _source_article_dict_from_article(articles_by_id[raw_id])
                for raw_id in source_ids
                if raw_id in articles_by_id
            ]

    source_payload: list[dict[str, Any]] = []
    sources = card.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            raw_id = _normalize_int_list(
                source.get("raw_article_id") or source.get("article_id") or source.get("id")
            )
            if not raw_id:
                continue
            item = {
                "id": raw_id[0],
                "title": source.get("title") or "",
                "url": source.get("url") or "",
                "source_name": source.get("source_name") or source.get("publisher") or "",
                "publisher": source.get("publisher") or "",
                "published_at": source.get("published_at"),
                "collected_at": source.get("collected_at"),
            }
            for image_key in (
                "image_url",
                "thumbnail_url",
                "thumbnail",
                "og_image",
                "main_image",
                "image",
                "image_urls",
                "images",
                "media_assets",
                "visual_images",
            ):
                if source.get(image_key):
                    item[image_key] = source[image_key]
            source_payload.append(item)
    if source_payload:
        return source_payload
    return [{"id": raw_id} for raw_id in source_ids]


def _resolve_peer_company_id(card: dict[str, Any]) -> Optional[str]:
    """카드 INSERT 시 peer_company_id FK 를 직접 확정한다.

    우선순위: card['peer_company_id'] (CardNewsComposer 가 set 했을 수 있음) →
    card['company'] (peer_companies.id 와 동일 표기) → card['peer_id'].
    국내 peer FK 로 안전하게 확인되는 값만 저장한다. 산업 트렌드, 글로벌 기업,
    SK AX 같은 self 회사, 미등록 회사는 보조 컬럼 company 만 사용하고 FK 는 NULL.
    """
    for value in (card.get("peer_company_id"), card.get("company"), card.get("peer_id")):
        peer_id = _normalize_peer_company_fk(value)
        if peer_id:
            return peer_id
    return None


def _normalize_peer_company_fk(value: Any) -> Optional[str]:
    text_value = str(value or "").strip()
    if not text_value:
        return None
    company_id = resolve_company_id(text_value)
    if (
        company_id == INDUSTRY_TREND_COMPANY
        or company_id in SELF_COMPANY_IDS
        or company_id in GLOBAL_COMPANY_IDS
        or company_id not in DOMESTIC_COMPANY_IDS
    ):
        return None
    return company_id


def _normalize_int_list(value: Any) -> list[int]:
    if not value:
        return []
    if isinstance(value, list | tuple | set):
        out: list[int] = []
        seen: set[int] = set()
        for item in value:
            try:
                ivalue = int(item)
            except (TypeError, ValueError):
                continue
            if ivalue <= 0 or ivalue in seen:
                continue
            seen.add(ivalue)
            out.append(ivalue)
        return out
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return []
    return [ivalue] if ivalue > 0 else []


def _build_evidence_payload(card: dict[str, Any]) -> dict[str, Any]:
    """카드 저장 시 `evidence_payload` JSONB 의 단일 출처.

    supervisor 의 메모리 `evidence_payload` 와 CardNewsComposer 의 `evidence_chain` /
    `sources` 를 합쳐서 sidecar (W5-2) 가 단일 컬럼만 보고도 풍부한 근거에 접근 가능.
    """
    payload: dict[str, Any] = {}
    raw_evidence = card.get("evidence_payload")
    if isinstance(raw_evidence, dict):
        payload.update(raw_evidence)
    analysis_package = card.get("analysis_package")
    if isinstance(analysis_package, dict) and analysis_package:
        payload.setdefault("analysis_package", analysis_package)
    evidence_chain = card.get("evidence_chain") or {}
    if isinstance(evidence_chain, dict):
        payload.setdefault("source_links", evidence_chain.get("source_links") or [])
        payload.setdefault("financial_refs", evidence_chain.get("financial_refs") or {})
        payload.setdefault("mbb_refs", evidence_chain.get("mbb_refs") or [])
        payload.setdefault("provenance", evidence_chain.get("provenance") or {})
    sources = card.get("sources") or []
    if isinstance(sources, list) and "source_links" not in payload:
        payload["source_links"] = sources
    return payload


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return [str(value).strip()] if str(value or "").strip() else []


def _integrated_issue_from_card_context(card: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {}
    for value in (
        card.get("integrated_issue"),
        (card.get("analysis_package") or {}).get("integrated_issue")
        if isinstance(card.get("analysis_package"), dict)
        else None,
        (card.get("evidence_payload") or {}).get("analysis_package", {}).get("integrated_issue")
        if isinstance(card.get("evidence_payload"), dict)
        and isinstance((card.get("evidence_payload") or {}).get("analysis_package"), dict)
        else None,
    ):
        if isinstance(value, dict) and value:
            return value
    return {}


def _integrated_issue_context_lines(card: dict[str, Any] | None) -> list[str]:
    issue = _integrated_issue_from_card_context(card)
    if not issue:
        return []
    candidates: list[Any] = []
    for key in ("display_fact_summary", "fact_summary", "summary_lines"):
        value = issue.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    for item in issue.get("fact_basis") or issue.get("consolidated_facts") or []:
        if isinstance(item, dict):
            candidates.append(item.get("fact") or item.get("evidence_text"))
        else:
            candidates.append(item)
    seen: set[str] = set()
    lines: list[str] = []
    for item in candidates:
        text = _sentence_text(str(item or "").strip())
        if not text:
            continue
        key = re.sub(r"\W+", "", text.lower())
        if key in seen:
            continue
        seen.add(key)
        lines.append(text)
        if len(lines) >= 5:
            break
    return lines


def _join_issue_lines(lines: list[str]) -> str:
    cleaned = [_sentence_text(line) for line in lines if str(line or "").strip()]
    return " ".join(cleaned)


def _looks_like_fragment(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if re.search(r"(다|요|니다|됐다|했다|한다|된다|있다|없다|였다|이다)[.!?。]?$", text):
        return False
    if len(text.split()) <= 8 and re.search(r"(확보|구축|전환|검증|구성|사례|범위)$", text):
        return True
    return not bool(re.search(r"[.!?。]$", text))


def _sentence_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    if re.search(r"[.!?。]$", text):
        return text
    if re.search(r"(다|요|니다|됐다|했다|한다|된다|있다|없다|였다|이다)$", text):
        return text + "."
    return text + "다."


def _labeled_lines_from_main_detail_blocks(
    blocks: list[dict[str, str]], *, prefix: str
) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        main = str(block.get("main") or "").strip()
        detail = str(block.get("detail") or "").strip()
        if not main:
            continue
        line = f"{prefix}: {main}"
        if detail:
            line = f"{line}\n근거/설명: {detail}"
        lines.append(line)
    return lines


def _main_detail_blocks(value: Any) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    if not isinstance(value, list):
        return blocks
    for item in value:
        if not isinstance(item, dict):
            continue
        main = str(item.get("main") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if main:
            blocks.append({"main": main, "detail": detail})
    return blocks


def _main_detail_blocks_from_labeled_lines(lines: list[str]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for line in lines:
        text = re.sub(r"\s+", " ", str(line or "").strip())
        if not text:
            continue
        text = re.sub(r"^핵심\s*(?:시사점|대응)\s*:\s*", "", text).strip()
        parts = re.split(r"\s*근거\s*/?\s*설명\s*:\s*", text, maxsplit=1)
        main = parts[0].strip() if parts else ""
        detail = parts[1].strip() if len(parts) == 2 else ""
        if main:
            blocks.append({"main": main, "detail": detail})
    return blocks


def _generate_card_id(company: str) -> str:
    """IC-YYYYMMDD-NNN 형식의 이슈카드 ID를 생성한다.

    DB 저장 전에 병렬 호출되므로 Lock으로 중복 방지.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    with _card_id_lock:
        if date_str not in _card_id_state:
            with SessionLocal() as db:
                row = db.execute(
                    text("SELECT COUNT(*) FROM card_news WHERE id LIKE :prefix"),
                    {"prefix": f"IC-{date_str}-%"},
                ).fetchone()
                _card_id_state[date_str] = row[0] if row else 0
        _card_id_state[date_str] += 1
        return f"IC-{date_str}-{_card_id_state[date_str]:03d}"


def _is_industry_trend_article(article: RawArticle) -> bool:
    if article.source_type in _INDUSTRY_SOURCE_TYPES:
        return True

    extra = article.extra or {}
    report_type = str(extra.get("type") or extra.get("report_type") or "").strip()
    if report_type in _INDUSTRY_REPORT_TYPES:
        return True

    if any(extra.get(key) for key in _INDUSTRY_MARKER_KEYS):
        return True

    return "industry" in (article.source_name or "").lower()


def _is_valid(article: RawArticle, storage_company: Optional[list[str]] = None) -> bool:
    """Gate 1: 최소 품질 필터."""
    if not article.url or not article.title:
        return False
    if not (storage_company if storage_company is not None else article.company):
        return False
    if len(article.content or "") < 10 and len(article.title) < 5:
        return False
    return True


def _source_metadata_json(
    article: RawArticle,
    storage_company: Optional[list[str]] = None,
) -> str:
    meta = _sanitize_jsonish(dict(article.extra))
    if not article.company and storage_company and INDUSTRY_TREND_COMPANY in storage_company:
        meta["topic_scope"] = "industry_trend"
        meta["company_scope"] = "industry"
        meta["company_fallback"] = INDUSTRY_TREND_COMPANY
    elif _is_industry_trend_article(article):
        meta.setdefault("topic_scope", "industry_trend")
    for key in _SOURCE_METADATA_EXCLUDED_KEYS:
        meta.pop(key, None)
    return json.dumps(meta, ensure_ascii=False)


def _metadata_json(
    article: RawArticle,
    storage_company: Optional[list[str]] = None,
    run_context: CrawlRunContext | None = None,
) -> str:
    """Compatibility shim for older tests/callers.

    Crawl run context is now written to crawl_run_articles, not metadata.
    """
    return _source_metadata_json(article, storage_company)


def _upsert_source_metadata(
    db,
    *,
    article_id: int,
    source_type: str,
    source_metadata: str,
) -> None:
    metadata = _metadata_dict(source_metadata)
    parse_payload = _parse_result_payload(metadata)
    collection_metadata = _collection_metadata(metadata)

    if collection_metadata:
        db.execute(
            text("""
                UPDATE raw_articles
                SET metadata = metadata || CAST(:metadata AS jsonb)
                WHERE id = :raw_article_id
            """),
            {
                "raw_article_id": article_id,
                "metadata": json.dumps(collection_metadata, ensure_ascii=False),
            },
        )

    if parse_payload:
        _upsert_parse_result(
            db, article_id=article_id, source_type=source_type, payload=parse_payload
        )


def _collection_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _PARSE_RESULT_METADATA_KEYS and value is not None
    }


def _parse_result_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    parser_result = _metadata_dict(metadata.get("parser_result"))
    if not parser_result and not any(key in metadata for key in _PARSE_RESULT_METADATA_KEYS):
        return {}

    financial_record = (
        metadata.get("financial_record") or parser_result.get("financial_record") or {}
    )
    warnings = parser_result.get("warnings") or metadata.get("warnings") or []
    raw_payload = {
        key: metadata.get(key)
        for key in (
            "parser_quality_score",
            "parser_quality_reason",
            "period",
            "period_year",
            "period_quarter",
            "period_type",
            "topics",
            "topic_signals",
        )
        if metadata.get(key) is not None
    }
    return {
        "parser_version": parser_result.get("parser_version"),
        "parser_ok": bool(parser_result.get("ok")),
        "parser_quality_label": metadata.get("parser_quality_label"),
        "parser_quality_score": metadata.get("parser_quality_score"),
        "parser_quality_reason": metadata.get("parser_quality_reason"),
        "period": metadata.get("period") or parser_result.get("period"),
        "period_year": metadata.get("period_year") or parser_result.get("period_year"),
        "period_quarter": metadata.get("period_quarter") or parser_result.get("period_quarter"),
        "period_type": metadata.get("period_type") or parser_result.get("period_type"),
        "published_at": parser_result.get("published_at"),
        "parser_result": parser_result,
        "financial_record": financial_record if isinstance(financial_record, dict) else {},
        "result_metadata": raw_payload,
        "warnings": warnings if isinstance(warnings, list) else [warnings],
    }


def _upsert_parse_result(
    db,
    *,
    article_id: int,
    source_type: str,
    payload: dict[str, Any],
) -> None:
    db.execute(
        text("""
            INSERT INTO raw_article_parse_results (
                raw_article_id, source_type, parser, parser_ok,
                period, period_year, period_quarter, period_type, published_at,
                parser_quality_score, parser_quality_label, parser_quality_reason,
                financial_record, result_metadata, warnings, raw_result
            ) VALUES (
                :raw_article_id, :source_type, :parser, :parser_ok,
                :period, :period_year, :period_quarter, :period_type, :published_at,
                :parser_quality_score, :parser_quality_label, :parser_quality_reason,
                CAST(:financial_record AS jsonb), CAST(:result_metadata AS jsonb),
                CAST(:warnings AS jsonb), CAST(:raw_result AS jsonb)
            )
            ON CONFLICT (raw_article_id) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                parser = EXCLUDED.parser,
                parser_ok = EXCLUDED.parser_ok,
                period = EXCLUDED.period,
                period_year = EXCLUDED.period_year,
                period_quarter = EXCLUDED.period_quarter,
                period_type = EXCLUDED.period_type,
                published_at = EXCLUDED.published_at,
                parser_quality_score = EXCLUDED.parser_quality_score,
                parser_quality_label = EXCLUDED.parser_quality_label,
                parser_quality_reason = EXCLUDED.parser_quality_reason,
                financial_record = EXCLUDED.financial_record,
                result_metadata = raw_article_parse_results.result_metadata
                    || EXCLUDED.result_metadata,
                warnings = EXCLUDED.warnings,
                raw_result = EXCLUDED.raw_result,
                updated_at = NOW()
        """),
        {
            "raw_article_id": article_id,
            "source_type": source_type,
            "parser": payload.get("parser_version") or f"{source_type}_parser",
            "parser_ok": payload.get("parser_ok"),
            "period": payload.get("period"),
            "period_year": payload.get("period_year"),
            "period_quarter": payload.get("period_quarter"),
            "period_type": payload.get("period_type"),
            "published_at": payload.get("published_at"),
            "parser_quality_score": payload.get("parser_quality_score"),
            "parser_quality_label": payload.get("parser_quality_label"),
            "parser_quality_reason": payload.get("parser_quality_reason"),
            "financial_record": json.dumps(
                _sanitize_jsonish(payload.get("financial_record") or {}),
                ensure_ascii=False,
            ),
            "result_metadata": json.dumps(
                _sanitize_jsonish(payload.get("result_metadata") or {}),
                ensure_ascii=False,
            ),
            "warnings": json.dumps(
                _sanitize_jsonish(payload.get("warnings") or []),
                ensure_ascii=False,
            ),
            "raw_result": json.dumps(
                _sanitize_jsonish(payload.get("parser_result") or {}),
                ensure_ascii=False,
            ),
        },
    )


def _upsert_crawl_run_article(
    db,
    *,
    article_id: int,
    article: RawArticle,
    run_context: CrawlRunContext | None,
    action: str,
    source_metadata: str,
) -> None:
    if not run_context or not run_context.crawl_run_id:
        return

    raw_payload = {
        "source_name": run_context.source_name or article.source_name,
        "collection_mode": run_context.collection_mode,
        "track": run_context.track,
        "window_start": _iso_or_none(run_context.window_start),
        "window_end": _iso_or_none(run_context.window_end),
        "source_metadata": _json_or_value(source_metadata, {}),
    }
    db.execute(
        _INSERT_CRAWL_RUN_ARTICLE,
        {
            "crawl_run_id": run_context.crawl_run_id,
            "raw_article_id": article_id,
            "url": article.url,
            "url_hash": article.url_hash,
            "discovered_at": article.collected_at,
            "action": action,
            "fetch_status": article.crawl_status,
            "error_message": article.error_message,
            "raw_payload": json.dumps(raw_payload, ensure_ascii=False),
        },
    )


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _sanitize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def fetch_global_trend_inputs(window_days: int = 30) -> list[dict[str, Any]]:
    """글로벌 6사 newsroom + SPRi/BCG raw_articles 를 ITTrendInput 호환 dict 로 반환.

    raw_articles.source_type 이 'official' 인 글로벌 6사 row 는 ITTrendAgent 의
    ``_split_trend_inputs()`` 에서 unsupported_items 로 떨어지므로, 여기서 source_type 을
    'global_newsroom' / 'trend_report' 로 정규화한다 (design §4.1 옵션 B 채택).
    """
    names = list(GLOBAL_NEWSROOM_SOURCE_NAMES | GLOBAL_RESEARCH_SOURCE_NAMES)
    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    """
                SELECT id, source_name, source_type, publisher, title, content, url,
                       published_at, collected_at, metadata, company
                FROM raw_articles
                WHERE source_name = ANY(:names)
                  AND COALESCE(published_at, collected_at) >= NOW() - make_interval(days => :days)
                ORDER BY COALESCE(published_at, collected_at) DESC NULLS LAST
                """
                ),
                {"names": names, "days": window_days},
            )
            .mappings()
            .all()
        )

    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["source_id"] = item["id"]
        source_name = (item.get("source_name") or "").strip().lower()
        if source_name in GLOBAL_NEWSROOM_SOURCE_NAMES:
            item["source_type"] = "global_newsroom"
        elif source_name in GLOBAL_RESEARCH_SOURCE_NAMES:
            item["source_type"] = "trend_report"
            item.setdefault("publisher", source_name)
        items.append(item)
    return items


def fetch_domestic_trend_representative_inputs(
    window_days: int = 30,
    limit: int = 120,
) -> list[dict[str, Any]]:
    """ACTIVE card_news 대표 원문을 IT trend 국내 참고 맥락으로 반환.

    글로벌 트렌드 감지는 글로벌 뉴스룸/리서치로 유지하고, 국내 카드뉴스는
    전략/시사점 에이전트가 국내 동향을 참고할 수 있는 별도 evidence layer 로 둔다.
    각 ACTIVE 카드뉴스 row 에서 primary_raw_article_id 또는 source_raw_article_ids 첫 항목
    하나만 가져와 대표 클러스터별 중복 입력을 줄인다.
    """
    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    """
                SELECT
                    cn.id AS card_news_id,
                    cn.cluster_id,
                    cn.title AS card_title,
                    cn.summary_lines,
                    cn.company AS card_company,
                    cn.peer_company_id,
                    cn.primary_keyword_category,
                    cn.keyword_categories,
                    cn.keywords,
                    cn.created_at AS card_created_at,
                    cn.primary_raw_article_id,
                    cn.source_raw_article_ids,
                    ra.id AS raw_article_id,
                    ra.source_name,
                    ra.source_type,
                    ra.publisher,
                    ra.title AS raw_title,
                    ra.content,
                    ra.url,
                    ra.published_at,
                    ra.collected_at,
                    ra.company AS raw_company,
                    ra.metadata
                FROM card_news cn
                LEFT JOIN raw_articles ra
                  ON ra.id = COALESCE(cn.primary_raw_article_id, cn.source_raw_article_ids[1])
                WHERE cn.status = 'ACTIVE'
                  AND cn.created_at >= NOW() - make_interval(days => :days)
                ORDER BY cn.created_at DESC NULLS LAST, cn.id DESC
                LIMIT :limit
                """
                ),
                {"days": window_days, "limit": limit},
            )
            .mappings()
            .all()
        )

    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["id"] = item.get("raw_article_id") or item.get("primary_raw_article_id")
        item["source_id"] = item["id"]
        item["source_type"] = "domestic_card_news"
        item["title"] = item.get("raw_title") or item.get("card_title")
        item["company"] = item.get("card_company") or item.get("raw_company")
        item["published_at"] = item.get("published_at") or item.get("card_created_at")
        items.append(item)
    return items


def _ilike_any_clause(
    col_exprs: Sequence[str],
    terms: Sequence[str],
    prefix: str,
) -> tuple[str, dict[str, str]]:
    """terms 중 하나라도 col_exprs 중 한 곳에 ILIKE 매칭되면 참이 되는 SQL 절을 만든다.

    각 ``col_exprs`` 항목은 ``{k}`` placeholder 를 포함한다 (예: ``"title ILIKE :{k}"``).
    term 1개당 모든 컬럼을 OR 로 묶고, term 들 사이도 OR — 즉 "아무 변형이든 어디서든 걸리면 매칭".
    terms 가 비면 ``("", {})`` 을 돌려 호출부에서 절을 생략하게 한다.

    Args:
        col_exprs: ``{k}`` 를 가진 컬럼 매칭 표현식들.
        terms: ILIKE 로 감쌀 검색어들 (한·영 변형 포함).
        prefix: 바인드 파라미터 이름 접두사 (호출부 간 충돌 방지).

    Returns:
        ``(sql_clause, params)``. ``sql_clause`` 는 바깥을 괄호로 감싼 OR 절.
    """
    if not terms:
        return "", {}
    params: dict[str, str] = {}
    groups: list[str] = []
    for idx, term in enumerate(terms):
        key = f"{prefix}{idx}"
        params[key] = f"%{term}%"
        cols = " OR ".join(expr.format(k=key) for expr in col_exprs)
        groups.append(f"({cols})")
    return "(" + " OR ".join(groups) + ")", params


def fetch_sk_ax_raw_for_alignment(
    *,
    window_days: int,
    match_terms: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """SK AX 자사 raw 를 peer alignment 용으로 fetch.

    ``card_news.peer_company_id='sk_ax'`` 가 0 건이라 card 가 아닌 raw_articles 에서
    직접 가져온다 (design §6 Phase 3 분기). ``match_terms`` 는 키워드의 한·영 변형 목록.
    """
    params: dict[str, Any] = {
        "names": list(SK_AX_RAW_SOURCE_NAMES),
        "days": window_days,
    }
    match_sql, match_params = _ilike_any_clause(
        ["title ILIKE :{k}", "content ILIKE :{k}"], match_terms or [], "kw"
    )
    keyword_clause = f"AND {match_sql}" if match_sql else ""
    params.update(match_params)

    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    f"""
                SELECT id, source_name, source_type, title, content, url,
                       published_at, collected_at, metadata
                FROM raw_articles
                WHERE source_name = ANY(:names)
                  AND collected_at >= NOW() - make_interval(days => :days)
                  {keyword_clause}
                ORDER BY COALESCE(published_at, collected_at) DESC NULLS LAST
                LIMIT 100
                """
                ),
                params,
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def fetch_peer_cards_for_alignment(
    *,
    peer_id: str,
    window_days: int,
    match_terms: Sequence[str] | None = None,
    importance_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Peer 4사 (samsung_sds / lg_cns / posco_dx / hyundai_autoever) 의 card_news fetch.

    SK AX 는 card_news 0 건이므로 ``fetch_sk_ax_raw_for_alignment`` 사용.
    importance_threshold 미만 카드는 naver_news noise 제거 (design §13).

    ``match_terms`` 는 트렌드 키워드의 한·영 변형 목록 — 트렌드 키워드(theme)는 영문
    정규형이고 국내 피어 card_news 는 한글이라, 변형을 OR ILIKE 로 함께 건다.
    과거의 ``primary_keyword_category`` 동일성 필터는 제거됨: card_news 의 sector
    분류(ax/security/infra/deal)와 트렌드 토픽 카테고리(ai_tech/ai_infra/cloud…)가
    서로 다른 분류 체계라, AND 비교가 AI 계열 트렌드를 항상 0건으로 만들었음.
    """
    if peer_id == "sk_ax":
        return fetch_sk_ax_raw_for_alignment(window_days=window_days, match_terms=match_terms)

    params: dict[str, Any] = {
        "peer": peer_id,
        "days": window_days,
        "importance": importance_threshold,
    }
    # global_search_text 는 gin_trgm_ops 인덱스 — ILIKE 빠름.
    # title + global_search_text + keywords[] (text[]) 3 곳에서 매칭.
    match_sql, match_params = _ilike_any_clause(
        [
            "title ILIKE :{k}",
            "global_search_text ILIKE :{k}",
            "EXISTS (SELECT 1 FROM unnest(keywords) kw WHERE kw ILIKE :{k})",
        ],
        match_terms or [],
        "kw",
    )
    params.update(match_params)

    filter_sql = f" AND {match_sql}" if match_sql else ""

    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    f"""
                SELECT id, title, summary_lines, primary_keyword_category,
                       peer_company_id, importance_score, created_at,
                       keywords, keyword_categories
                FROM card_news
                WHERE peer_company_id = :peer
                  AND status = 'ACTIVE'
                  AND created_at >= NOW() - make_interval(days => :days)
                  AND COALESCE(importance_score, 0) >= :importance
                  {filter_sql}
                ORDER BY created_at DESC NULLS LAST
                LIMIT 50
                """
                ),
                params,
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def upsert_global_industry_trends(rows: list[dict[str, Any]]) -> int:
    """V29 ``uq_global_industry_trends_daily_keyword`` 로 idempotent upsert.

    같은 날 cronjob 이 두 번 돌아도 안전. ``source_analysis_id`` 는 첫 INSERT
    시점 보존. ``payload`` 는 dict 면 자동 JSON 직렬화.
    design: global-trends.md §7.
    """
    if not rows:
        return 0
    serialized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        payload = item.get("payload")
        if isinstance(payload, (dict, list)):
            item["payload"] = json.dumps(payload, ensure_ascii=False, default=str)
        elif payload is None:
            item["payload"] = "{}"
        for key in ("related_peer_ids", "related_card_ids", "source_raw_article_ids"):
            item[key] = list(item.get(key) or [])
        serialized.append(item)
    with SessionLocal() as db:
        db.execute(_GLOBAL_TREND_UPSERT_SQL, serialized)
        db.commit()
    return len(serialized)


def fetch_previous_trend_context_for_delta() -> dict[str, Any]:
    """오늘 이전 가장 최근 ``global_industry_trends`` batch → Phase 2 delta 입력.

    design: ``global-trends.md`` §16 — ledger 대신 self-read.
    prior batch 가 없으면 빈 dict (cold start → ``frequency_delta_pct = 0``).
    """
    query = text(
        """
                WITH latest_prior AS (
                    SELECT MAX(trend_date) AS d
                    FROM global_industry_trends
                    WHERE trend_date < CURRENT_DATE
                )
                SELECT g.keyword, g.mention_count, g.payload, g.trend_date
                FROM global_industry_trends g
                INNER JOIN latest_prior lp ON g.trend_date = lp.d
                ORDER BY g.impact_score DESC NULLS LAST, g.keyword ASC
                """
    )
    try:
        with SessionLocal() as db:
            rows = db.execute(query).mappings().all()
    except Exception as exc:  # noqa: BLE001
        if not _is_missing_relation(exc, "global_industry_trends"):
            raise
        rows = []

    if not rows:
        return {}

    keyword_counts: dict[str, int] = {}
    signals: list[dict[str, Any]] = []
    for row in rows:
        kw = str(row["keyword"] or "").strip().lower()
        if not kw:
            continue
        payload = row["payload"] if isinstance(row["payload"], dict) else {}
        mention_count = int(row["mention_count"] or payload.get("mention_count") or 0)
        keyword_counts[kw] = mention_count
        signals.append(
            {
                "signal": kw,
                "mention_count": mention_count,
                "intensity": payload.get("intensity"),
                "leading_companies": payload.get("leading_companies", []),
            }
        )

    latest_date = rows[0]["trend_date"]
    return {
        "period": "prior_batch",
        "keyword_counts": keyword_counts,
        "signals": signals,
        "source_groups": ["global_industry_trends"],
        "updated_at": latest_date.isoformat()
        if hasattr(latest_date, "isoformat")
        else str(latest_date),
        "metadata": {"row_count": len(rows), "prior_trend_date": str(latest_date)},
    }


def _is_missing_column(exc: Exception, column_name: str) -> bool:
    """Return True when a read path hit a missing optional DB column."""
    text_value = str(exc)
    return "UndefinedColumn" in text_value and column_name in text_value


def _is_missing_relation(exc: Exception, relation_name: str) -> bool:
    text_value = str(exc)
    return "UndefinedTable" in text_value and relation_name in text_value


_card_id_lock = threading.Lock()
_card_id_state: dict[str, int] = {}  # {date_str: last_seq}
