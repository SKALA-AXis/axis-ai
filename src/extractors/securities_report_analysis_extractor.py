# 작성일: 2026-05-18
# 작성자: 박지원
# 변경이력:
#   2026-05-18 박지원 — 증권리포트 분석 추출기 작성 및 Profile/PeerProfile 에이전트 연동 수정
"""Securities report parser_result를 분석용 metric/signal 레코드로 변환한다."""

from __future__ import annotations

import re
from typing import Any

_PRICE_METRIC_LABELS = {
    "target_price": "목표주가",
    "current_price": "현재주가",
    "upside_pct": "상승여력",
}
_FINANCIAL_METRIC_LABELS = {
    "revenue_total": "매출",
    "operating_profit": "영업이익",
    "operating_margin": "영업이익률",
    "net_income": "순이익",
    "eps": "EPS",
}
_BUSINESS_AREA_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cloud", ("cloud", "클라우드", "msp", "csp", "데이터센터", "gpu")),
    (
        "ai_ax",
        ("ai", "ax", "생성형", "인공지능", "llm", "agent", "fabrix", "brity", "자동화"),
    ),
    ("logistics", ("logistics", "물류", "cello", "scl")),
    ("smart_factory", ("스마트팩토리", "smart factory", "mes", "factory", "제조")),
    ("vehicle_sw", ("차량", "vehicle", "sdv", "내비게이션", "navigation")),
    ("enterprise_it", ("enterprise", "erp", "ito", "si", "it서비스", "그룹사")),
)
_SIGNAL_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("valuation", ("목표주가", "투자의견", "valuation", "per", "pbr", "상승여력")),
    ("forecast", ("전망", "예상", "추정", "forecast", "2026e", "2027e", "guidance")),
    ("risk", ("리스크", "우려", "하회", "둔화", "부진", "감소", "하락", "비용")),
    ("orders_pipeline", ("수주", "계약", "backlog", "pipeline", "잔고")),
    ("investment", ("투자", "capex", "설비", "데이터센터", "gpu", "구축")),
    ("service_launch", ("출시", "서비스", "솔루션", "플랫폼", "상용화")),
    ("growth", ("성장", "확대", "증가", "개선", "회복", "모멘텀")),
    ("strategy", ("전략", "추진", "강화", "고도화", "제휴", "협력")),
    ("efficiency", ("효율", "최적화", "자동화", "비용 절감", "생산성")),
)
_SIGNAL_TYPE_PRIORITY = {
    "risk": 0,
    "orders_pipeline": 1,
    "growth": 2,
    "strategy": 3,
    "forecast": 4,
    "investment": 5,
    "valuation": 6,
    "service_launch": 7,
    "efficiency": 8,
}
_NEGATIVE_TERMS = ("리스크", "우려", "하회", "둔화", "부진", "감소", "하락", "비용")
_POSITIVE_TERMS = ("성장", "확대", "증가", "개선", "회복", "강화", "상승", "수주")
_SIGNAL_EXCLUDE_TERMS = (
    "리서치센터",
    "자료:",
    "자료 :",
    "목표주가 변동내역",
    "투자의견 및 목표주가 변동내역",
    "target per",
    "target pbr",
    "forecasts and valuations",
)
_MONEY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("revenue_total", re.compile(r"(?:매출액|매출)\s*([0-9][0-9,\.]*)\s*(조원|억원|십억원)")),
    ("operating_profit", re.compile(r"영업이익\s*([0-9][0-9,\.]*)\s*(조원|억원|십억원)")),
    ("net_income", re.compile(r"(?:순이익|당기순이익)\s*([0-9][0-9,\.]*)\s*(조원|억원|십억원)")),
)
_RATIO_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("operating_margin", re.compile(r"(?:영업이익률|OPM)\s*([+-]?[0-9][0-9,\.]*)\s*%")),
    ("eps", re.compile(r"EPS\s*([0-9][0-9,]*)\s*원?", re.I)),
)
_SK_AX_INCLUDE_TERMS = (
    "sk ax",
    "sk에이엑스",
    "sk c&c",
    "sk㈜ c&c",
    "sk주식회사 c&c",
    "에스케이씨앤씨",
    "c&c",
    "it서비스",
    "it 서비스",
    "ai transformation",
    "ax",
    "agentic",
    "에이전틱",
    "클라우드",
    "msp",
    "csp",
)
_SK_AX_ENTITY_TERMS = (
    "sk ax",
    "sk에이엑스",
    "sk c&c",
    "sk㈜ c&c",
    "sk주식회사 c&c",
    "에스케이씨앤씨",
    "c&c",
)
_SK_GROUP_EXCLUDE_TERMS = (
    "sk하이닉스",
    "sk hynix",
    "하이닉스",
    "sk텔레콤",
    "skt",
    "sk스퀘어",
    "sk이노베이션",
    "sk온",
    "sk엔무브",
    "sk e&s",
    "sk바이오팜",
    "sk실트론",
    "sk네트웍스",
    "skc",
)
_SK_HOLDING_VALUATION_TERMS = (
    "nav",
    "순자산가치",
    "자회사",
    "상장 계열사",
    "비상장자회사",
    "지분가치",
    "배당수익",
    "목표주가",
    "상승여력",
    "할인율",
)


def financial_metrics_from_securities_report(
    article: dict[str, Any],
    parser_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """증권사 리포트의 투자의견/목표가/핵심 수치를 metric row로 변환한다."""
    article_id = int(article["id"])
    peer_id = _peer_id(article, parser_result)
    period = parser_result.get("period") or article["extra"].get("period")
    metrics: list[dict[str, Any]] = []

    target_price = parser_result.get("target_price_krw")
    current_price = parser_result.get("current_price_krw")
    if peer_id == "sk_ax":
        target_price = None
        current_price = None

    if isinstance(target_price, int | float):
        metrics.append(
            _metric_row(
                article=article,
                article_id=article_id,
                peer_id=peer_id,
                period=period,
                metric_uid=f"securities_report:target_price:{period or 'unknown'}",
                metric_name="target_price",
                metric_label=_PRICE_METRIC_LABELS["target_price"],
                metric_scope="company_total",
                value_numeric=float(target_price),
                unit="원",
                currency="KRW",
                evidence_text=f"목표주가 {int(target_price):,}원",
                confidence=0.9,
                payload={"parser_result": _compact_parser_result(parser_result)},
            )
        )

    if isinstance(current_price, int | float):
        metrics.append(
            _metric_row(
                article=article,
                article_id=article_id,
                peer_id=peer_id,
                period=period,
                metric_uid=f"securities_report:current_price:{period or 'unknown'}",
                metric_name="current_price",
                metric_label=_PRICE_METRIC_LABELS["current_price"],
                metric_scope="company_total",
                value_numeric=float(current_price),
                unit="원",
                currency="KRW",
                evidence_text=f"현재주가 {int(current_price):,}원",
                confidence=0.9,
                payload={"parser_result": _compact_parser_result(parser_result)},
            )
        )

    if (
        isinstance(target_price, int | float)
        and isinstance(current_price, int | float)
        and current_price
    ):
        upside = (float(target_price) / float(current_price) - 1) * 100
        metrics.append(
            _metric_row(
                article=article,
                article_id=article_id,
                peer_id=peer_id,
                period=period,
                metric_uid=f"securities_report:upside_pct:{period or 'unknown'}",
                metric_name="upside_pct",
                metric_label=_PRICE_METRIC_LABELS["upside_pct"],
                metric_scope="company_total",
                value_numeric=round(upside, 2),
                unit="%",
                currency=None,
                evidence_text=f"목표주가 대비 상승여력 {upside:.1f}%",
                confidence=0.86,
                payload={"target_price_krw": target_price, "current_price_krw": current_price},
            )
        )

    metrics.extend(
        _financial_metrics_from_text(article, parser_result, article_id, peer_id, period)
    )
    return metrics


def business_signals_from_securities_report(
    article: dict[str, Any],
    parser_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """증권사 리포트 문장/청크를 사업 전망 signal row로 변환한다."""
    chunks = parser_result.get("document_chunks")
    if not isinstance(chunks, list):
        return []

    article_id = int(article["id"])
    peer_id = _peer_id(article, parser_result)
    period = parser_result.get("period") or article["extra"].get("period")
    signals: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text_value = str(chunk.get("text") or "").strip()
        if peer_id == "sk_ax":
            text_value = _sk_ax_relevant_context(text_value)
        if len(text_value) < 30:
            continue
        for sentence_index, sentence, business_area, signal_type in _signals_from_text(text_value):
            dedupe_key = (business_area, sentence[:180])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            source_chunk_uid = str(chunk.get("chunk_id") or chunk.get("chunk_index") or "")
            signals.append(
                {
                    "raw_article_id": article_id,
                    "signal_uid": (
                        f"securities_report:{business_area}:{signal_type}:"
                        f"{source_chunk_uid or len(signals) + 1}:s{sentence_index}"
                    ),
                    "source_type": "securities_report",
                    "source_name": article.get("source_name"),
                    "peer_id": peer_id,
                    "period": period,
                    "period_year": _period_year(period),
                    "period_quarter": _period_quarter(period),
                    "period_type": _period_type(period),
                    "business_area": business_area,
                    "signal_type": signal_type,
                    "sentiment": _sentiment(sentence),
                    "summary": _summary(sentence),
                    "evidence_text": sentence,
                    "source_page": None,
                    "source_chunk_uid": source_chunk_uid or None,
                    "confidence": _signal_confidence(sentence, business_area, signal_type),
                    "extraction_method": "securities_report_parser.document_chunks.rule_based",
                    "payload": {
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "report_firm": parser_result.get("report_firm"),
                        "chunk": {
                            "chunk_id": chunk.get("chunk_id"),
                            "chunk_index": chunk.get("chunk_index"),
                            "section_key": chunk.get("section_key"),
                            "section_title": chunk.get("section_title"),
                        },
                    },
                }
            )

    return signals


def _financial_metrics_from_text(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    article_id: int,
    peer_id: str | None,
    period: Any,
) -> list[dict[str, Any]]:
    text_value = str(article.get("content") or "")[:12000]
    if peer_id == "sk_ax":
        text_value = _sk_ax_relevant_context(text_value)
    metrics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metric_name, pattern in _MONEY_PATTERNS:
        for index, match in enumerate(pattern.finditer(text_value), start=1):
            value = _money_to_krw_100m(match.group(1), match.group(2))
            if value is None:
                continue
            key = f"{metric_name}:{value}"
            if key in seen:
                continue
            seen.add(key)
            metrics.append(
                _metric_row(
                    article=article,
                    article_id=article_id,
                    peer_id=peer_id,
                    period=period,
                    metric_uid=(
                        f"securities_report:{metric_name}:text:{index}:{period or 'unknown'}"
                    ),
                    metric_name=metric_name,
                    metric_label=_FINANCIAL_METRIC_LABELS[metric_name],
                    metric_scope="company_total",
                    value_numeric=value,
                    unit="억원",
                    currency="KRW",
                    evidence_text=match.group(0),
                    confidence=0.72,
                    payload={"report_firm": parser_result.get("report_firm")},
                )
            )
            break

    for metric_name, pattern in _RATIO_PATTERNS:
        ratio_match = pattern.search(text_value)
        if not ratio_match:
            continue
        value = _number(ratio_match.group(1))
        if value is None:
            continue
        is_price = metric_name == "eps"
        metrics.append(
            _metric_row(
                article=article,
                article_id=article_id,
                peer_id=peer_id,
                period=period,
                metric_uid=f"securities_report:{metric_name}:text:{period or 'unknown'}",
                metric_name=metric_name,
                metric_label=_FINANCIAL_METRIC_LABELS[metric_name],
                metric_scope="company_total",
                value_numeric=value,
                unit=(
                    "원"
                    if is_price
                    else ("%" if metric_name in {"operating_margin", "roe"} else "배")
                ),
                currency="KRW" if is_price else None,
                evidence_text=ratio_match.group(0),
                confidence=0.7,
                payload={"report_firm": parser_result.get("report_firm")},
            )
        )

    return metrics


def _metric_row(
    *,
    article: dict[str, Any],
    article_id: int,
    peer_id: str | None,
    period: Any,
    metric_uid: str,
    metric_name: str,
    metric_label: str,
    metric_scope: str,
    value_numeric: float,
    unit: str,
    currency: str | None,
    evidence_text: str,
    confidence: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    is_krw_100m = unit == "억원"
    value_krw = None
    if is_krw_100m:
        value_krw = value_numeric * 100_000_000
    elif currency == "KRW":
        value_krw = value_numeric

    return {
        "raw_article_id": article_id,
        "metric_uid": metric_uid,
        "source_type": "securities_report",
        "source_name": article.get("source_name"),
        "peer_id": peer_id,
        "period": period,
        "period_year": _period_year(period),
        "period_quarter": _period_quarter(period),
        "period_type": _period_type(period),
        "metric_name": metric_name,
        "metric_label": metric_label,
        "metric_scope": metric_scope,
        "business_area": None,
        "value_numeric": value_numeric,
        "value_krwbn": value_numeric if is_krw_100m else None,
        "value_krw": value_krw,
        "unit": unit,
        "currency": currency,
        "source_page": None,
        "source_table_uid": None,
        "source_chunk_uid": None,
        "confidence": confidence,
        "extraction_method": "securities_report_parser.rule_based",
        "evidence_text": evidence_text[:1000],
        "payload": {**payload, "title": article.get("title"), "url": article.get("url")},
    }


def _signals_from_text(text_value: str) -> list[tuple[int, str, str, str]]:
    signals: list[tuple[int, str, str, str]] = []
    for index, sentence in enumerate(_sentences(text_value), start=1):
        if _is_low_value_signal_sentence(sentence):
            continue
        if not _looks_like_narrative_signal_sentence(sentence):
            continue
        signal_types = _detect_signal_types(sentence)
        if not signal_types:
            continue
        business_area = _detect_business_area(sentence) or "company_total"
        primary_signal_type = _primary_signal_type(signal_types)
        signals.append((index, sentence[:1000], business_area, primary_signal_type))
    return signals


def _sk_ax_relevant_context(text_value: str) -> str:
    """SK Inc. 리포트에서 SK AX/C&C/IT서비스 문맥만 evidence 후보로 남긴다."""

    sentences = _sentences(text_value)
    if not sentences:
        return ""

    selected: list[str] = []
    for index, sentence in enumerate(sentences):
        if not _is_sk_ax_relevant_sentence(sentence):
            continue

        context = [sentence]
        next_sentence = sentences[index + 1] if index + 1 < len(sentences) else ""
        if _is_sk_ax_followup_sentence(next_sentence):
            context.append(next_sentence)
        merged = " ".join(context)
        if merged not in selected:
            selected.append(merged)

    return "\n".join(selected)


def _is_sk_ax_relevant_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if _is_sk_group_or_holding_noise(lowered):
        return False
    return any(term in lowered for term in _SK_AX_INCLUDE_TERMS)


def _is_sk_ax_followup_sentence(sentence: str) -> bool:
    if not sentence:
        return False
    lowered = sentence.lower()
    if _is_sk_group_or_holding_noise(lowered):
        return False
    if any(term in lowered for term in _SK_AX_INCLUDE_TERMS):
        return True
    return bool(
        re.search(
            r"(성장|확대|증가|개선|전망|예상|추정|기여|수익성|매출|영업이익|수주|계약|"
            r"플랫폼|서비스|솔루션|자동화|효율)",
            sentence,
        )
    )


def _is_sk_group_or_holding_noise(lowered_sentence: str) -> bool:
    has_sk_ax_entity = any(term in lowered_sentence for term in _SK_AX_ENTITY_TERMS)
    if has_sk_ax_entity:
        return False
    if any(term in lowered_sentence for term in _SK_GROUP_EXCLUDE_TERMS):
        return True
    return any(term in lowered_sentence for term in _SK_HOLDING_VALUATION_TERMS)


def _detect_business_area(text_value: str) -> str | None:
    lowered = text_value.lower()
    for business_area, terms in _BUSINESS_AREA_RULES:
        if any(_contains_term(lowered, term) for term in terms):
            return business_area
    return None


def _detect_signal_types(text_value: str) -> list[str]:
    lowered = text_value.lower()
    return [
        signal_type
        for signal_type, terms in _SIGNAL_TYPE_RULES
        if any(_contains_term(lowered, term) for term in terms)
    ]


def _primary_signal_type(signal_types: list[str]) -> str:
    ranked = sorted(signal_types, key=_signal_type_priority)
    return ranked[0]


def _signal_type_priority(signal_type: str) -> int:
    return _SIGNAL_TYPE_PRIORITY.get(signal_type, 99)


def _contains_term(lowered_text: str, term: str) -> bool:
    lowered_term = term.lower()
    if lowered_term in {"ai", "ax"}:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(lowered_term)}(?![a-z0-9])", lowered_text))
    return lowered_term in lowered_text


def _sentences(text_value: str) -> list[str]:
    value = text_value.replace("\r\n", "\n")
    value = re.sub(r"[ \t]+", " ", value).strip()
    value = re.sub(
        r"((?:\b(?:1q|2q|3q|4q)\d{2}[pe]?\b\s*){3,}(?:\b20\d{2}e?\b\s*){1,})(?=[가-힣A-Za-z])",
        r"\1\n",
        value,
        flags=re.I,
    )
    pieces = re.split(r"\n+|(?<=[.!?。])\s+", value)
    sentences = [piece.strip(" -•\t") for piece in pieces if len(piece.strip()) >= 30]
    return sentences or ([value] if value else [])


def _is_low_value_signal_sentence(text_value: str) -> bool:
    lowered = text_value.lower()
    return any(term in lowered for term in _SIGNAL_EXCLUDE_TERMS)


def _looks_like_narrative_signal_sentence(text_value: str) -> bool:
    value = re.sub(r"\s+", " ", str(text_value or "")).strip()
    if not value:
        return False
    if _looks_like_table_like_signal_text(value):
        return False

    tokens = re.findall(r"[가-힣A-Za-z]{2,}", value)
    if len(tokens) < 3:
        return False

    return bool(
        re.search(
            r"(전망|예상|추정|우려|둔화|부진|감소|하락|성장|확대|증가|개선|회복|강화|수주|"
            r"추진|고도화|협력|상용화|출시|기여|기록|반영|만회|상승여력)",
            value,
        )
    )


def _looks_like_table_like_signal_text(text_value: str) -> bool:
    lowered = text_value.lower()
    compact = re.sub(r"\s+", " ", text_value).strip()
    if compact.startswith("[표") and ("단위:" in compact or "변동내역" in compact):
        return True
    if "1q24 2q24" in lowered or "2024 2025 2026e" in lowered:
        return True
    if re.search(r"\b(?:1q|2q|3q|4q)\d{2}\b(?:\s+\b(?:1q|2q|3q|4q)\d{2}\b){2,}", lowered):
        return True
    if re.search(r"\b20\d{2}e?\b(?:\s+\b20\d{2}e?\b){2,}", lowered):
        return True
    if re.search(r"\b(?:per|pbr|roe|eps)\b\s+[0-9][0-9.\s,%배원]{8,}", lowered):
        return True

    number_like_tokens = re.findall(r"[0-9][0-9,./%]*", compact)
    word_tokens = re.findall(r"[가-힣A-Za-z]{2,}", compact)
    if len(number_like_tokens) >= 6 and len(word_tokens) <= 8:
        return True
    return False


def _sentiment(text_value: str) -> str:
    lowered = text_value.lower()
    if any(term in lowered for term in _NEGATIVE_TERMS):
        return "negative"
    if any(term in lowered for term in _POSITIVE_TERMS):
        return "positive"
    return "neutral"


def _summary(evidence_text: str) -> str:
    value = re.sub(r"\s+", " ", evidence_text).strip()
    return value if len(value) <= 160 else f"{value[:157]}..."


def _signal_confidence(text_value: str, business_area: str, signal_type: str) -> float:
    has_area = _detect_business_area(text_value) == business_area
    has_type = signal_type in _detect_signal_types(text_value)
    if has_area and has_type:
        return 0.76
    if business_area == "company_total" and has_type:
        return 0.68
    return 0.62


def _money_to_krw_100m(raw_value: str, unit: str) -> float | None:
    value = _number(raw_value)
    if value is None:
        return None
    if unit == "조원":
        return value * 10000
    if unit == "십억원":
        return value * 10
    return value


def _number(raw_value: str) -> float | None:
    try:
        return float(raw_value.replace(",", ""))
    except ValueError:
        return None


def _peer_id(article: dict[str, Any], parser_result: dict[str, Any]) -> str | None:
    peer_id = parser_result.get("peer_id")
    if peer_id:
        return str(peer_id)
    company = article.get("company")
    if isinstance(company, list) and company:
        return str(company[0])
    if isinstance(company, str):
        return company
    return None


def _period_year(period: Any) -> int | None:
    match = re.search(r"(20\d{2})", str(period or ""))
    return int(match.group(1)) if match else None


def _period_quarter(period: Any) -> int | None:
    match = re.search(r"Q([1-4])", str(period or ""), re.I)
    return int(match.group(1)) if match else None


def _period_type(period: Any) -> str | None:
    if _period_quarter(period):
        return "quarter"
    if _period_year(period):
        return "annual_forecast" if "E" in str(period).upper() else "annual"
    return None


def _compact_parser_result(parser_result: dict[str, Any]) -> dict[str, Any]:
    keys = ("report_firm", "investment_opinion", "target_price_krw", "current_price_krw", "period")
    return {key: parser_result.get(key) for key in keys}
