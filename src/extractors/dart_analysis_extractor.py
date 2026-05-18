"""DART parser_result를 분석용 metric/signal 레코드로 변환한다."""

from __future__ import annotations

import re
from typing import Any

_DART_METRIC_LABELS = {
    "revenue_total": "매출",
    "cost_of_sales": "매출원가",
    "gross_profit": "매출총이익",
    "operating_profit": "영업이익",
    "profit_before_tax": "법인세비용차감전순이익",
    "net_income": "당기순이익",
    "total_assets": "자산총계",
    "total_liabilities": "부채총계",
    "total_equity": "자본총계",
    "operating_cash_flow": "영업활동현금흐름",
    "investing_cash_flow": "투자활동현금흐름",
    "financing_cash_flow": "재무활동현금흐름",
    "cash_and_cash_equivalents": "현금및현금성자산",
}
_BUSINESS_AREA_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cloud",
        (
            "cloud",
            "클라우드",
            "msp",
            "csp",
            "aws",
            "azure",
            "gcp",
            "데이터센터",
            "data center",
            "gpu",
        ),
    ),
    (
        "ai_ax",
        (
            "ai",
            "ax",
            "생성형",
            "genai",
            "llm",
            "agent",
            "fabrix",
            "brity",
            "erp ai",
            "scm",
            "자동화",
            "인공지능",
        ),
    ),
    ("logistics", ("logistics", "물류", "cello", "scl")),
    ("smart_factory", ("smart factory", "스마트팩토리", "mes", "factory", "제조")),
    ("vehicle_sw", ("vehicle", "차량", "sdv", "내비게이션", "navigation")),
    ("enterprise_it", ("enterprise", "erp", "ito", "si", "그룹사", "it서비스")),
    ("robotics", ("robot", "로봇", "automation")),
)
_SECTION_SIGNAL_HINTS = {
    "business": ("strategy", "company_total"),
    "financial": ("growth", "company_total"),
    "management_discussion": ("strategy", "company_total"),
    "other": ("risk", "company_total"),
    "affiliates": ("strategy", "company_total"),
}
_SIGNAL_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("risk", ("위험", "리스크", "소송", "규제", "불확실", "하락", "감소", "부진")),
    ("orders_pipeline", ("수주", "계약", "backlog", "잔고", "pipeline")),
    ("investment", ("투자", "설비", "capex", "연구개발", "r&d", "개발비")),
    ("growth", ("성장", "확대", "증가", "개선", "매출", "영업이익")),
    ("strategy", ("전략", "추진", "강화", "고도화", "출시", "제휴", "협력")),
    ("efficiency", ("효율", "최적화", "자동화", "비용 절감", "생산성")),
)
_NEGATIVE_TERMS = ("위험", "리스크", "하락", "감소", "둔화", "부진", "소송", "규제")
_POSITIVE_TERMS = ("성장", "확대", "증가", "개선", "강화", "고도화", "수주", "계약")


def financial_metrics_from_dart(
    article: dict[str, Any],
    parser_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """DART 재무제표 구조화 결과를 raw_article_financial_metrics row로 변환한다."""
    article_id = int(article["id"])
    peer_id = _peer_id(article, parser_result)
    period = _period(article, parser_result)
    period_year = parser_result.get("period_year") or article["extra"].get("period_year")
    period_quarter = parser_result.get("period_quarter") or article["extra"].get("period_quarter")
    period_type = parser_result.get("period_type") or article["extra"].get("period_type")

    metrics: list[dict[str, Any]] = []
    statements = parser_result.get("financial_statements")
    if isinstance(statements, list):
        for statement in statements:
            if not isinstance(statement, dict):
                continue
            metrics.extend(
                _metrics_from_statement(
                    article=article,
                    article_id=article_id,
                    peer_id=peer_id,
                    period=period,
                    period_year=period_year,
                    period_quarter=period_quarter,
                    period_type=period_type,
                    statement=statement,
                )
            )

    if metrics:
        return metrics

    return _metrics_from_candidates(
        article=article,
        parser_result=parser_result,
        article_id=article_id,
        peer_id=peer_id,
        period=period,
        period_year=period_year,
        period_quarter=period_quarter,
        period_type=period_type,
    )


def business_signals_from_dart(
    article: dict[str, Any],
    parser_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """DART 문서 청크를 사업/회사별 business signal row로 변환한다."""
    chunks = parser_result.get("document_chunks")
    if not isinstance(chunks, list):
        chunks = article["extra"].get("dart_document_chunks")
    if not isinstance(chunks, list):
        return []

    article_id = int(article["id"])
    peer_id = _peer_id(article, parser_result)
    period = _period(article, parser_result)
    period_year = parser_result.get("period_year") or article["extra"].get("period_year")
    period_quarter = parser_result.get("period_quarter") or article["extra"].get("period_quarter")
    period_type = parser_result.get("period_type") or article["extra"].get("period_type")

    signals: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text_value = str(chunk.get("text") or "").strip()
        if len(text_value) < 80:
            continue

        business_area = _business_area_from_chunk(chunk, text_value)
        signal_type = _signal_type_from_chunk(chunk, text_value)
        if not business_area or not signal_type:
            continue

        evidence_text = _evidence_text(text_value, business_area, signal_type)
        if not evidence_text:
            continue
        dedupe_key = (business_area, signal_type, evidence_text[:180])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        signal_uid = (
            f"dart:{business_area}:{signal_type}:"
            f"{chunk.get('chunk_id') or chunk.get('chunk_index') or len(signals) + 1}"
        )
        signals.append(
            {
                "raw_article_id": article_id,
                "signal_uid": signal_uid,
                "source_type": "dart",
                "source_name": article.get("source_name"),
                "peer_id": peer_id,
                "period": period,
                "period_year": period_year,
                "period_quarter": period_quarter,
                "period_type": period_type,
                "business_area": business_area,
                "signal_type": signal_type,
                "sentiment": _sentiment(evidence_text),
                "summary": _summary(evidence_text),
                "evidence_text": evidence_text,
                "source_page": None,
                "source_chunk_uid": str(chunk.get("chunk_id") or chunk.get("chunk_index") or ""),
                "confidence": _signal_confidence(chunk, business_area, signal_type, text_value),
                "extraction_method": "dart_parser.document_chunks.rule_based",
                "payload": {
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "rcept_no": parser_result.get("rcept_no") or article["extra"].get("rcept_no"),
                    "chunk": {
                        "chunk_id": chunk.get("chunk_id"),
                        "chunk_index": chunk.get("chunk_index"),
                        "section_key": chunk.get("section_key"),
                        "section_title": chunk.get("section_title"),
                        "subsection_title": chunk.get("subsection_title"),
                        "topics": chunk.get("topics"),
                    },
                },
            }
        )

    return signals


def _metrics_from_statement(
    *,
    article: dict[str, Any],
    article_id: int,
    peer_id: str | None,
    period: str | None,
    period_year: Any,
    period_quarter: Any,
    period_type: Any,
    statement: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    rows = statement.get("rows")
    if not isinstance(rows, list):
        return []

    statement_scope = str(statement.get("statement_scope") or "company_total")
    table_index = statement.get("table_index")
    for row in rows:
        if not isinstance(row, dict):
            continue
        metric_name = str(row.get("metric_key") or "")
        value = row.get("current_value_krwbn")
        if metric_name not in _DART_METRIC_LABELS or not isinstance(value, int | float):
            continue

        metrics.append(
            _metric_row(
                article=article,
                article_id=article_id,
                peer_id=peer_id,
                period=period,
                period_year=period_year,
                period_quarter=period_quarter,
                period_type=period_type,
                metric_uid=(
                    f"dart:{metric_name}:{statement_scope}:"
                    f"table{table_index or 'x'}:{period or 'unknown'}"
                ),
                metric_name=metric_name,
                metric_label=_DART_METRIC_LABELS[metric_name],
                metric_scope="company_total",
                value_numeric=float(value),
                source_table_uid=f"dart-table-{table_index}" if table_index is not None else None,
                evidence_text=str(row.get("label") or metric_name),
                confidence=0.94 if statement.get("table_type") != "unclassified" else 0.86,
                payload={
                    "statement": {
                        "table_index": table_index,
                        "title": statement.get("title"),
                        "table_type": statement.get("table_type"),
                        "statement_scope": statement_scope,
                        "unit": statement.get("unit"),
                    },
                    "row": row,
                },
            )
        )

    return metrics


def _metrics_from_candidates(
    *,
    article: dict[str, Any],
    parser_result: dict[str, Any],
    article_id: int,
    peer_id: str | None,
    period: str | None,
    period_year: Any,
    period_quarter: Any,
    period_type: Any,
) -> list[dict[str, Any]]:
    candidates = parser_result.get("candidates")
    if not isinstance(candidates, list):
        return []

    metrics: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        metric_name = str(candidate.get("type") or "")
        value = candidate.get("value_krwbn")
        if metric_name not in _DART_METRIC_LABELS or not isinstance(value, int | float):
            continue
        metrics.append(
            _metric_row(
                article=article,
                article_id=article_id,
                peer_id=peer_id,
                period=period,
                period_year=period_year,
                period_quarter=period_quarter,
                period_type=period_type,
                metric_uid=f"dart:{metric_name}:candidate:{index}:{period or 'unknown'}",
                metric_name=metric_name,
                metric_label=_DART_METRIC_LABELS[metric_name],
                metric_scope="company_total",
                value_numeric=float(value),
                source_table_uid=(
                    f"dart-table-{candidate.get('table_index')}"
                    if candidate.get("table_index") is not None
                    else None
                ),
                evidence_text=str(candidate.get("raw") or metric_name),
                confidence=float(candidate.get("confidence") or 0.82),
                payload={"candidate": candidate},
            )
        )

    return metrics


def _metric_row(
    *,
    article: dict[str, Any],
    article_id: int,
    peer_id: str | None,
    period: str | None,
    period_year: Any,
    period_quarter: Any,
    period_type: Any,
    metric_uid: str,
    metric_name: str,
    metric_label: str,
    metric_scope: str,
    value_numeric: float,
    source_table_uid: str | None,
    evidence_text: str,
    confidence: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "raw_article_id": article_id,
        "metric_uid": metric_uid,
        "source_type": "dart",
        "source_name": article.get("source_name"),
        "peer_id": peer_id,
        "period": period,
        "period_year": period_year,
        "period_quarter": period_quarter,
        "period_type": period_type,
        "metric_name": metric_name,
        "metric_label": metric_label,
        "metric_scope": metric_scope,
        "business_area": None,
        "value_numeric": value_numeric,
        "value_krwbn": value_numeric,
        "value_krw": value_numeric * 100_000_000,
        "unit": "억원",
        "currency": "KRW",
        "source_page": None,
        "source_table_uid": source_table_uid,
        "source_chunk_uid": None,
        "confidence": confidence,
        "extraction_method": "dart_parser.financial_statements",
        "evidence_text": evidence_text,
        "payload": {
            **payload,
            "title": article.get("title"),
            "url": article.get("url"),
        },
    }


def _business_area_from_chunk(chunk: dict[str, Any], text_value: str) -> str | None:
    detected = _detect_business_area(text_value)
    if detected:
        return detected

    section_key = str(chunk.get("section_key") or "")
    hint = _SECTION_SIGNAL_HINTS.get(section_key)
    if hint:
        return hint[1]

    return None


def _signal_type_from_chunk(chunk: dict[str, Any], text_value: str) -> str | None:
    detected = _detect_signal_type(text_value)
    if detected:
        return detected

    section_key = str(chunk.get("section_key") or "")
    hint = _SECTION_SIGNAL_HINTS.get(section_key)
    if hint:
        return hint[0]

    return None


def _detect_business_area(text_value: str) -> str | None:
    lowered = text_value.lower()
    for business_area, terms in _BUSINESS_AREA_RULES:
        if any(term.lower() in lowered for term in terms):
            return business_area
    return None


def _detect_signal_type(text_value: str) -> str | None:
    lowered = text_value.lower()
    for signal_type, terms in _SIGNAL_TYPE_RULES:
        if any(term.lower() in lowered for term in terms):
            return signal_type
    return None


def _evidence_text(text_value: str, business_area: str, signal_type: str) -> str | None:
    terms = _terms_for_evidence(business_area, signal_type)
    sentences = _sentences(text_value)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(term.lower() in lowered for term in terms):
            return sentence[:1000]
    return sentences[0][:1000] if sentences else None


def _terms_for_evidence(business_area: str, signal_type: str) -> tuple[str, ...]:
    business_terms = next(
        (terms for area, terms in _BUSINESS_AREA_RULES if area == business_area),
        (),
    )
    signal_terms = next(
        (terms for kind, terms in _SIGNAL_TYPE_RULES if kind == signal_type),
        (),
    )
    return (*business_terms, *signal_terms)


def _sentences(text_value: str) -> list[str]:
    value = re.sub(r"\s+", " ", text_value).strip()
    if not value:
        return []
    pieces = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+", value)
    sentences = [piece.strip(" -•\t") for piece in pieces if len(piece.strip()) >= 30]
    return sentences or [value]


def _sentiment(text_value: str) -> str:
    lowered = text_value.lower()
    if any(term in lowered for term in _NEGATIVE_TERMS):
        return "negative"
    if any(term in lowered for term in _POSITIVE_TERMS):
        return "positive"
    return "neutral"


def _summary(evidence_text: str) -> str:
    value = re.sub(r"\s+", " ", evidence_text).strip()
    return value if len(value) <= 180 else f"{value[:177]}..."


def _signal_confidence(
    chunk: dict[str, Any],
    business_area: str,
    signal_type: str,
    text_value: str,
) -> float:
    detected_area = _detect_business_area(text_value)
    detected_type = _detect_signal_type(text_value)
    if detected_area == business_area and detected_type == signal_type:
        return 0.82
    if chunk.get("section_key") in _SECTION_SIGNAL_HINTS:
        return 0.72
    return 0.68


def _peer_id(article: dict[str, Any], parser_result: dict[str, Any]) -> str | None:
    value = parser_result.get("peer_id")
    if value:
        return str(value)
    company = article.get("company")
    if isinstance(company, list) and company:
        return str(company[0])
    if isinstance(company, str):
        return company
    return None


def _period(article: dict[str, Any], parser_result: dict[str, Any]) -> str | None:
    value = parser_result.get("period") or article["extra"].get("period")
    return str(value) if value else None
