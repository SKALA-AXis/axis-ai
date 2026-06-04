"""DART parser_result를 분석용 metric/signal 레코드로 변환한다."""

from __future__ import annotations

import re
from collections.abc import Collection
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
_EXCLUDED_SIGNAL_SECTIONS = {
    "auditor",
    "governance",
    "shareholder",
    "executives",
    "major_shareholder",
    "detailed_tables",
}
_SECTION_SIGNAL_POLICY: dict[str, dict[str, object]] = {
    "business": {
        "allowed_signal_types": {
            "strategy",
            "growth",
            "orders_pipeline",
            "investment",
            "efficiency",
        },
        "default_business_area": "company_total",
    },
    "management_discussion": {
        "allowed_signal_types": {
            "strategy",
            "growth",
            "orders_pipeline",
            "investment",
            "efficiency",
            "risk",
        },
        "default_business_area": "company_total",
    },
    "financial": {
        "allowed_signal_types": {"investment"},
        "default_business_area": "company_total",
    },
    "other": {
        "allowed_signal_types": {"risk"},
        "default_business_area": "other",
    },
    "affiliates": {
        "allowed_signal_types": {"strategy", "risk"},
        "default_business_area": "company_total",
    },
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
_SK_AX_STRONG_TERMS = (
    "sk ax",
    "에스케이 ax",
    "sk㈜ c&c",
    "sk주식회사 c&c",
    "sk c&c",
    "sk c & c",
    "씨앤씨",
    "c&c부문",
    "c&c 부문",
    "sk주식회사 사업부문",
    "sk 주식회사 사업부문",
    "sk주식회사는 국내 top-tier it 서비스",
    "it서비스",
    "it 서비스",
    "디지털전환",
    "enterprise it",
)
_SK_AX_BUSINESS_TERMS = (
    "ax",
    "ai",
    "인공지능",
    "생성형",
    "클라우드",
    "cloud",
    "데이터센터",
    "data center",
    "it 컨설팅",
    "시스템 구축",
    "아웃소싱",
    "outsourcing",
    "agentic",
    "delivery 역량",
    "erp",
    "scm",
    "자동화",
)
_SK_GROUP_UNRELATED_TERMS = (
    "sk이노베이션",
    "sk innovation",
    "sk텔레콤",
    "sk telecom",
    "skt",
    "에이닷",
    "sk하이닉스",
    "sk hynix",
    "sk스퀘어",
    "sk square",
    "sk바이오팜",
    "sk biopharmaceuticals",
    "sk e&s",
    "sk온",
    "sk on",
    "sk에코플랜트",
    "sk ecoplant",
    "투자부문",
    "계열회사",
    "관계회사",
    "자회사",
    "포트폴리오",
)
_SIGNAL_TYPE_PRIORITY = {
    "risk": 0,
    "orders_pipeline": 1,
    "growth": 2,
    "strategy": 3,
    "investment": 4,
    "efficiency": 5,
    "business_overview": 6,
    "product_service": 7,
    "rd": 8,
}


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
    period, period_year, period_quarter, period_type = _normalize_period_fields(
        period,
        period_year,
        period_quarter,
        period_type,
    )

    if peer_id == "sk_ax":
        return _metrics_from_candidates(
            article=article,
            parser_result=parser_result,
            article_id=article_id,
            peer_id=peer_id,
            period=period,
            period_year=period_year,
            period_quarter=period_quarter,
            period_type=period_type,
            candidate_filter=_is_sk_ax_financial_metric_candidate,
        )

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
        metrics.extend(
            metric
            for metric in _metrics_from_candidates(
                article=article,
                parser_result=parser_result,
                article_id=article_id,
                peer_id=peer_id,
                period=period,
                period_year=period_year,
                period_quarter=period_quarter,
                period_type=period_type,
            )
            if metric.get("metric_scope") == "segment"
            or metric.get("business_area") not in {None, "company_total"}
        )
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
    period, period_year, period_quarter, period_type = _normalize_period_fields(
        period,
        period_year,
        period_quarter,
        period_type,
    )

    signals: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    signals.extend(
        _llm_business_signals_from_dart(
            article=article,
            parser_result=parser_result,
            article_id=article_id,
            peer_id=peer_id,
            period=period,
            period_year=period_year,
            period_quarter=period_quarter,
            period_type=period_type,
            seen=seen,
        )
    )
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text_value = str(chunk.get("text") or "").strip()
        if len(text_value) < 30:
            continue

        for sentence_index, evidence_text, business_area, signal_type in _signals_from_chunk(
            chunk,
            text_value,
            peer_id=peer_id,
        ):
            business_area = _business_area_from_evidence_override(
                peer_id=peer_id,
                business_area=business_area,
                evidence_text=evidence_text,
            )
            dedupe_key = (business_area, evidence_text[:180])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            signal_uid = (
                f"dart:{business_area}:{signal_type}:"
                f"{chunk.get('chunk_id') or chunk.get('chunk_index') or len(signals) + 1}:"
                f"s{sentence_index}"
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
                    "source_chunk_uid": str(
                        chunk.get("chunk_id") or chunk.get("chunk_index") or ""
                    ),
                    "confidence": _signal_confidence(
                        chunk,
                        business_area,
                        signal_type,
                        evidence_text,
                    ),
                    "extraction_method": "dart_parser.section_sentence.rule_based",
                    "payload": {
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "rcept_no": parser_result.get("rcept_no")
                        or article["extra"].get("rcept_no"),
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


def _llm_business_signals_from_dart(
    *,
    article: dict[str, Any],
    parser_result: dict[str, Any],
    article_id: int,
    peer_id: str | None,
    period: Any,
    period_year: Any,
    period_quarter: Any,
    period_type: Any,
    seen: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    llm_signals = parser_result.get("llm_business_signals")
    if not isinstance(llm_signals, list):
        return []

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for index, signal in enumerate(llm_signals, start=1):
        if not isinstance(signal, dict):
            continue
        business_area = str(signal.get("business_area") or "company_total")
        signal_type = str(signal.get("signal_type") or "")
        evidence_text = str(signal.get("evidence_text") or "").strip()
        if not signal_type or not evidence_text:
            continue
        if peer_id == "sk_ax" and not _is_sk_ax_relevant_sentence(evidence_text):
            continue
        business_area = _business_area_from_evidence_override(
            peer_id=peer_id,
            business_area=business_area,
            evidence_text=evidence_text,
        )
        dedupe_key = (business_area, evidence_text[:180])
        if dedupe_key in seen:
            continue
        row = {
            "raw_article_id": article_id,
            "signal_uid": f"dart-llm:{business_area}:{signal_type}:{index}",
            "source_type": "dart",
            "source_name": article.get("source_name"),
            "peer_id": peer_id,
            "period": period,
            "period_year": period_year,
            "period_quarter": period_quarter,
            "period_type": period_type,
            "business_area": business_area,
            "signal_type": signal_type,
            "sentiment": signal.get("sentiment") or _sentiment(evidence_text),
            "summary": signal.get("summary") or _summary(evidence_text),
            "evidence_text": evidence_text,
            "source_page": None,
            "source_chunk_uid": None,
            "confidence": signal.get("confidence") or 0.78,
            "extraction_method": "dart_llm.analysis",
            "payload": {
                "title": article.get("title"),
                "url": article.get("url"),
                "rcept_no": parser_result.get("rcept_no") or article["extra"].get("rcept_no"),
                "llm_signal": signal,
            },
        }
        existing = rows_by_key.get(dedupe_key)
        if existing and _signal_type_priority(existing["signal_type"]) <= _signal_type_priority(
            signal_type
        ):
            continue
        rows_by_key[dedupe_key] = row
    rows = list(rows_by_key.values())
    for row in rows:
        seen.add((row["business_area"], row["evidence_text"][:180]))
    return rows


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
        if metric_name not in _DART_METRIC_LABELS:
            continue
        row_values = row.get("values")
        if not isinstance(row_values, list):
            row_values = [
                {
                    "column_index": 1,
                    "value_krwbn": row.get("current_value_krwbn"),
                }
            ]

        for value_index, value_info in enumerate(row_values, start=1):
            if not isinstance(value_info, dict):
                continue
            value = value_info.get("value_krwbn")
            if not isinstance(value, int | float):
                continue
            metric_period = value_info.get("period") or period
            metric_period_year = (
                value_info.get("period_year") if value_info.get("period") else period_year
            )
            metric_period_quarter = (
                value_info.get("period_quarter") if value_info.get("period") else period_quarter
            )
            metric_period_type = (
                value_info.get("period_type") if value_info.get("period") else period_type
            )
            (
                metric_period,
                metric_period_year,
                metric_period_quarter,
                metric_period_type,
            ) = _normalize_period_fields(
                metric_period,
                metric_period_year,
                metric_period_quarter,
                metric_period_type,
            )

            metrics.append(
                _metric_row(
                    article=article,
                    article_id=article_id,
                    peer_id=peer_id,
                    period=metric_period,
                    period_year=metric_period_year,
                    period_quarter=metric_period_quarter,
                    period_type=metric_period_type,
                    metric_uid=(
                        f"dart:{metric_name}:{statement_scope}:"
                        f"table{table_index or 'x'}:"
                        f"c{value_info.get('column_index') or value_index}:"
                        f"{metric_period or 'unknown'}"
                    ),
                    metric_name=metric_name,
                    metric_label=_DART_METRIC_LABELS[metric_name],
                    metric_scope="company_total",
                    business_area="company_total",
                    value_numeric=float(value),
                    source_table_uid=(
                        f"dart-table-{table_index}" if table_index is not None else None
                    ),
                    evidence_text=" ".join(
                        str(part)
                        for part in (
                            row.get("label") or metric_name,
                            value_info.get("column_header"),
                            value_info.get("raw"),
                        )
                        if part
                    ),
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
                        "value": value_info,
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
    candidate_filter: Any | None = None,
) -> list[dict[str, Any]]:
    candidates = parser_result.get("candidates")
    if not isinstance(candidates, list):
        return []

    metrics: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        if candidate_filter is not None and not candidate_filter(candidate):
            continue
        metric_name = str(candidate.get("type") or "")
        value = candidate.get("value_krwbn")
        if metric_name not in _DART_METRIC_LABELS or not isinstance(value, int | float):
            continue
        metric_period = candidate.get("period") or period
        metric_period_year = (
            candidate.get("period_year") if candidate.get("period") else period_year
        )
        metric_period_quarter = (
            candidate.get("period_quarter") if candidate.get("period") else period_quarter
        )
        metric_period_type = (
            candidate.get("period_type") if candidate.get("period") else period_type
        )
        metric_period, metric_period_year, metric_period_quarter, metric_period_type = (
            _normalize_period_fields(
                metric_period,
                metric_period_year,
                metric_period_quarter,
                metric_period_type,
            )
        )
        metrics.append(
            _metric_row(
                article=article,
                article_id=article_id,
                peer_id=peer_id,
                period=metric_period,
                period_year=metric_period_year,
                period_quarter=metric_period_quarter,
                period_type=metric_period_type,
                metric_uid=f"dart:{metric_name}:candidate:{index}:{metric_period or 'unknown'}",
                metric_name=metric_name,
                metric_label=_DART_METRIC_LABELS[metric_name],
                metric_scope=_candidate_metric_scope(candidate, peer_id=peer_id),
                business_area=_candidate_business_area(candidate, peer_id=peer_id),
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


def _candidate_metric_scope(candidate: dict[str, Any], *, peer_id: str | None) -> str:
    return str(candidate.get("metric_scope") or "company_total")


def _is_sk_ax_financial_metric_candidate(candidate: dict[str, Any]) -> bool:
    if candidate.get("metric_scope") != "segment":
        return False
    if _is_sk_investment_segment_candidate(candidate):
        return False
    if _is_sk_ax_business_segment_candidate(candidate):
        return True
    if candidate.get("standard_business_area") == "sk_ax":
        return True

    evidence = " ".join(
        str(candidate.get(key) or "")
        for key in ("business_area", "segment_label", "raw", "table_title")
    )
    return _is_sk_ax_relevant_sentence(evidence)


def _is_sk_investment_segment_candidate(candidate: dict[str, Any]) -> bool:
    evidence = _candidate_evidence(candidate)
    return "투자부문" in evidence and not _is_sk_ax_business_segment_text(evidence)


def _is_sk_ax_business_segment_candidate(candidate: dict[str, Any]) -> bool:
    return _is_sk_ax_business_segment_text(_candidate_evidence(candidate))


def _is_sk_ax_business_segment_text(value: str) -> bool:
    compact = re.sub(r"\s+", "", value.lower())
    return any(
        token in compact
        for token in (
            "사업부문",
            "it서비스",
            "itservice",
            "digital기술",
            "ai/digital",
            "aidigital",
        )
    )


def _candidate_evidence(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "")
        for key in ("business_area", "segment_label", "raw", "table_title")
    )


def _candidate_business_area(candidate: dict[str, Any], *, peer_id: str | None) -> str:
    if peer_id == "sk_ax" and candidate.get("standard_business_area") == "sk_ax":
        return "sk_ax"
    return str(candidate.get("business_area") or "company_total")


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
    business_area: str,
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
        "business_area": business_area,
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


def _signals_from_chunk(
    chunk: dict[str, Any],
    text_value: str,
    *,
    peer_id: str | None = None,
) -> list[tuple[int, str, str, str]]:
    section_key = str(chunk.get("section_key") or "")
    if section_key in _EXCLUDED_SIGNAL_SECTIONS:
        return []

    policy = _SECTION_SIGNAL_POLICY.get(section_key)
    if policy is None:
        return []

    allowed_signal_types = policy["allowed_signal_types"]
    if not isinstance(allowed_signal_types, Collection):
        return []
    default_business_area = policy["default_business_area"]
    if default_business_area is not None:
        default_business_area = str(default_business_area)
    signals: list[tuple[int, str, str, str]] = []
    chunk_areas = _detect_business_areas(text_value)

    for index, sentence in enumerate(_sentences(text_value), start=1):
        if peer_id == "sk_ax" and not _is_sk_ax_relevant_sentence(sentence):
            continue

        sentence_signal_types = [
            signal_type
            for signal_type in _detect_signal_types(sentence)
            if signal_type in allowed_signal_types
        ]
        if not sentence_signal_types:
            continue

        business_area = _detect_business_area(sentence)
        if not business_area and len(chunk_areas) > 1:
            continue
        if not business_area and len(chunk_areas) == 1:
            sole_chunk_area = next(iter(chunk_areas))
            if sole_chunk_area == "cloud" and not _has_explicit_cloud_context(sentence.lower()):
                business_area = "other"
            else:
                business_area = sole_chunk_area
        if not business_area and peer_id == "sk_ax" and _is_sk_ax_relevant_sentence(sentence):
            business_area = "sk_ax"
        if not business_area and not chunk_areas:
            business_area = default_business_area
        if not business_area:
            continue

        primary_signal_type = _primary_signal_type(sentence_signal_types)
        signals.append((index, sentence[:1000], business_area, primary_signal_type))

    return signals


def _is_sk_ax_relevant_sentence(sentence: str) -> bool:
    lowered = " ".join(sentence.lower().split())
    has_strong = any(_contains_term(lowered, term) for term in _SK_AX_STRONG_TERMS)
    if has_strong:
        return True

    has_unrelated_group = any(_contains_term(lowered, term) for term in _SK_GROUP_UNRELATED_TERMS)
    if has_unrelated_group:
        return False

    business_hits = sum(1 for term in _SK_AX_BUSINESS_TERMS if _contains_term(lowered, term))
    has_sk_business_context = "sk주식회사" in lowered or "sk 주식회사" in lowered
    return business_hits >= 2 or (has_sk_business_context and business_hits >= 1)


def _detect_business_area(text_value: str) -> str | None:
    lowered = text_value.lower()
    for business_area, terms in _BUSINESS_AREA_RULES:
        if any(_contains_term(lowered, term) for term in terms):
            return business_area
    return None


def _business_area_from_evidence_override(
    *,
    peer_id: str | None,
    business_area: str,
    evidence_text: str,
) -> str:
    if business_area != "cloud":
        return business_area

    lowered = evidence_text.lower()
    explicit_areas = _detect_business_areas(evidence_text)
    explicit_non_cloud_areas = [area for area in explicit_areas if area != "cloud"]
    if explicit_non_cloud_areas and not _has_explicit_cloud_context(lowered):
        if len(explicit_non_cloud_areas) == 1:
            return explicit_non_cloud_areas[0]
        return "other"

    if peer_id == "sk_ax":
        if _has_enterprise_it_context(lowered) and not _has_explicit_cloud_context(lowered):
            return "enterprise_it"
        if not _has_explicit_cloud_context(lowered):
            return "other"

    if not _has_explicit_cloud_context(lowered):
        return "other"

    return business_area


def _primary_signal_type(signal_types: list[str]) -> str:
    ranked = sorted(signal_types, key=_signal_type_priority)
    return ranked[0]


def _signal_type_priority(signal_type: str) -> int:
    return _SIGNAL_TYPE_PRIORITY.get(signal_type, 99)


def _detect_business_areas(text_value: str) -> set[str]:
    lowered = text_value.lower()
    areas: set[str] = set()
    for business_area, terms in _BUSINESS_AREA_RULES:
        if any(_contains_term(lowered, term) for term in terms):
            areas.add(business_area)
    return areas


def _has_explicit_cloud_context(lowered_text: str) -> bool:
    return any(
        _contains_term(lowered_text, term)
        for term in (
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
        )
    )


def _has_enterprise_it_context(lowered_text: str) -> bool:
    return any(
        _contains_term(lowered_text, term)
        for term in (
            "enterprise",
            "erp",
            "ito",
            "si",
            "it서비스",
            "it 서비스",
            "it service",
            "it services",
        )
    )


def _detect_signal_types(text_value: str) -> list[str]:
    lowered = text_value.lower()
    signal_types: list[str] = []
    for signal_type, terms in _SIGNAL_TYPE_RULES:
        if any(_contains_term(lowered, term) for term in terms):
            signal_types.append(signal_type)
    return signal_types


def _contains_term(lowered_text: str, term: str) -> bool:
    lowered_term = term.lower()
    if _is_short_ascii_term(lowered_term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(lowered_term)}(?![a-z0-9])", lowered_text))
    return lowered_term in lowered_text


def _is_short_ascii_term(value: str) -> bool:
    return len(value) <= 3 and bool(re.fullmatch(r"[a-z0-9&]+", value))


def _sentences(text_value: str) -> list[str]:
    value = re.sub(r"\s+", " ", text_value).strip()
    if not value:
        return []
    pieces = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+", value)
    sentences = [piece.strip(" -•\t") for piece in pieces if len(piece.strip()) >= 18]
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
    if detected_area == business_area and signal_type in _detect_signal_types(text_value):
        return 0.82
    if chunk.get("section_key") in _SECTION_SIGNAL_POLICY:
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


def _normalize_period_fields(
    period: Any,
    period_year: Any,
    period_quarter: Any,
    period_type: Any,
) -> tuple[str | None, Any, Any, Any]:
    if period_type in {"annual", "year"} and period_year:
        return str(period_year), period_year, None, period_type
    if period_type == "half":
        return str(period) if period else None, period_year, None, period_type
    return str(period) if period else None, period_year, period_quarter, period_type


def _period(article: dict[str, Any], parser_result: dict[str, Any]) -> str | None:
    value = parser_result.get("period") or article["extra"].get("period")
    return str(value) if value else None
