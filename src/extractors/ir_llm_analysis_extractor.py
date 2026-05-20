"""LLM-assisted IR analysis extraction.

Deterministic parsing keeps the first pass cheap and reproducible. This module
adds an optional second pass for IR PDFs whose tables/charts/narratives are too
varied for rules alone.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

_LLM_MODEL = os.getenv("IR_ANALYSIS_LLM_MODEL", "gpt-4o-mini")
_MAX_PAGE_CHARS = int(os.getenv("IR_LLM_PAGE_CHARS", "5000"))
_MAX_PAGES = int(os.getenv("IR_LLM_MAX_PAGES", "12"))

_METRIC_ALIASES = {
    "revenue_total": {"revenue_total", "revenue", "sales", "매출", "매출액"},
    "operating_profit": {"operating_profit", "op", "영업이익"},
    "operating_margin": {"operating_margin", "opm", "영업이익률"},
    "gross_profit": {"gross_profit", "매출총이익"},
    "gross_margin": {"gross_margin", "gpm", "매출총이익률", "매출총이익율"},
    "ebitda": {"ebitda"},
    "backlog": {"backlog", "수주잔고", "잔고"},
    "orders": {"orders", "order", "수주"},
    "capex": {"capex", "설비투자", "투자금액"},
    "net_income": {"net_income", "순이익", "당기순이익"},
}
_ALLOWED_METRICS = set(_METRIC_ALIASES)
_ALLOWED_SIGNAL_TYPES = {
    "growth",
    "strategy",
    "efficiency",
    "risk",
    "outlook",
    "orders_pipeline",
    "investment",
    "profitability",
}
_SK_AX_STRONG_TERMS = (
    "sk ax",
    "sk에이엑스",
    "sk㈜ c&c",
    "sk주식회사 c&c",
    "sk c&c",
    "sk c & c",
    "c&c",
    "씨앤씨",
    "it서비스",
    "it 서비스",
    "enterprise it",
    "si",
    "ito",
    "클라우드",
    "물류",
    "cloud",
    "logistics",
    "ai transformation",
    "fabrix",
    "brity",
)
_SK_AX_PAGE_STRONG_TERMS = (
    "sk ax",
    "sk에이엑스",
    "sk㈜ c&c",
    "sk주식회사 c&c",
    "sk c&c",
    "sk c & c",
    "c&c",
    "씨앤씨",
)
_SK_AX_PORTFOLIO_TERMS = (
    "sk에코플랜트",
    "에코플랜트",
    "sk ecoplant",
    "sk이노베이션",
    "sk innovation",
    "sk텔레콤",
    "sk telecom",
    "skt",
    "sk하이닉스",
    "sk hynix",
    "sk스퀘어",
    "sk square",
    "sk바이오팜",
    "sk biopharmaceuticals",
    "sk e&s",
    "sk실트론",
    "sk siltron",
    "sk온",
    "sk on",
    "sk머티리얼즈",
    "sk materials",
    "에이닷",
)
_SK_AX_PAGE_BUSINESS_TERMS = (
    "it서비스",
    "it 서비스",
    "it services",
    "information technology services",
    "enterprise it",
    "digital service",
    "digital services",
    "si",
    "ito",
    "클라우드",
    "cloud",
    "데이터센터",
    "data center",
    "ai transformation",
    "ax",
)

_llm: ChatOpenAI | None = None


def analyze_ir_with_llm(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    *,
    max_pages: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run optional LLM extraction and return normalized metrics/signals."""
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY 미설정: IR LLM 분석 스킵")
        return {"llm_financial_metrics": [], "llm_business_signals": []}

    prompt = _build_ir_llm_prompt(article, parser_result, max_pages=max_pages)
    if not prompt:
        return {"llm_financial_metrics": [], "llm_business_signals": []}

    response = _get_llm().invoke(
        [
            (
                "system",
                "You extract structured facts from Korean/English IR materials. "
                "Return JSON only. Do not infer numbers that are not explicitly present.",
            ),
            ("user", prompt),
        ]
    )
    parsed = _parse_json_response(getattr(response, "content", response))
    peer_id = _peer_id(article, parser_result)
    return {
        "llm_financial_metrics": _normalize_llm_metrics(
            parsed.get("financial_metrics"),
            peer_id=peer_id,
        ),
        "llm_business_signals": _normalize_llm_signals(
            parsed.get("business_signals"),
            peer_id=peer_id,
        ),
    }


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0,
            max_completion_tokens=3500,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm


def _build_ir_llm_prompt(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    *,
    max_pages: int | None = None,
) -> str:
    peer_id = _peer_id(article, parser_result)
    pages = _page_contexts(
        parser_result,
        max_pages=max_pages or _MAX_PAGES,
        peer_id=peer_id,
    )
    if not pages:
        return ""

    metadata = {
        "raw_article_id": article.get("id"),
        "title": article.get("title") or parser_result.get("title"),
        "peer_id": parser_result.get("peer_id"),
        "report_period": parser_result.get("period"),
    }
    peer_guardrail = _peer_guardrail(metadata.get("peer_id") or peer_id)
    page_text = "\n\n".join(pages)
    return f"""
IR 문서에서 표/차트 기반 재무 지표와 본문 기반 사업 시그널을 분리 추출하세요.

원칙:
- financial_metrics에는 표 또는 차트에 명시된 숫자만 넣으세요.
- 본문/Highlights/bullet의 설명 문장 숫자는 financial_metrics가 아니라 business_signals에 넣으세요.
- 표의 상위 행 흐름을 해석하세요.
  예: '매출액' 아래 'SI/ITO/Enterprise IT' 행은 revenue_total의 segment입니다.
- '매출원가', '총이익', '총이익률'은 revenue_total로 오분류하지 마세요.
- period는 가능하면 2026Q1, 2025Q4, 2025 같은 형식으로 정규화하세요.
- business_area는 표/페이지에 보이는 원문 라벨을 쓰고, 전사/전체이면 company_total을 쓰세요.
- evidence_text는 표 제목 > 상위 metric > 행 라벨 > 열/값 흐름이 보이도록 충분히 길게 쓰세요.
- 확실하지 않은 값은 confidence를 0.6 미만으로 두세요.
{peer_guardrail}

허용 metric_name:
{sorted(_ALLOWED_METRICS)}

허용 signal_type:
{sorted(_ALLOWED_SIGNAL_TYPES)}

출력 JSON 형식:
{{
  "financial_metrics": [
    {{
      "metric_name": "revenue_total",
      "metric_label": "매출",
      "metric_scope": "company_total|segment",
      "business_area": "company_total 또는 원문 사업부문",
      "period": "2026Q1",
      "period_year": 2026,
      "period_quarter": 1,
      "period_type": "quarter|year|half",
      "value_numeric": 7378,
      "unit": "억원|%",
      "source_page": 5,
      "table_title": "부문별 손익현황",
      "row_label": "Enterprise IT",
      "parent_row_label": "매출액",
      "column_label": "26년 1분기",
      "evidence_text": "표 흐름과 전체 행 값",
      "confidence": 0.0
    }}
  ],
  "business_signals": [
    {{
      "business_area": "Enterprise IT",
      "signal_type": "growth",
      "sentiment": "positive|neutral|negative",
      "summary": "한 문장 요약",
      "evidence_text": "원문 근거 문장",
      "source_page": 5,
      "confidence": 0.0
    }}
  ]
}}

문서 메타데이터:
{json.dumps(metadata, ensure_ascii=False)}

페이지/표/본문 컨텍스트:
{page_text}
""".strip()


def _page_contexts(
    parser_result: dict[str, Any],
    *,
    max_pages: int,
    peer_id: str | None = None,
) -> list[str]:
    raw_pages = parser_result.get("raw_text_pages")
    pages: list[str] = []
    if isinstance(raw_pages, list):
        selected_pages = _filter_page_items_for_peer(raw_pages, peer_id=peer_id)
        for page in selected_pages[:max_pages]:
            if not isinstance(page, dict):
                continue
            page_no = page.get("page")
            text = str(page.get("text") or "").strip()
            if text:
                pages.append(f"[PAGE {page_no}]\n{text[:_MAX_PAGE_CHARS]}")

    if pages:
        return pages

    chunks = parser_result.get("document_chunks")
    if isinstance(chunks, list):
        selected_chunks = _filter_page_items_for_peer(chunks, peer_id=peer_id)
        for chunk in selected_chunks[: max_pages * 2]:
            if not isinstance(chunk, dict):
                continue
            page_no = chunk.get("page")
            text = str(chunk.get("text") or "").strip()
            if text:
                pages.append(f"[PAGE {page_no} CHUNK {chunk.get('chunk_id')}]\n{text[:2000]}")
    return pages


def _filter_page_items_for_peer(items: list[Any], *, peer_id: str | None) -> list[Any]:
    if peer_id != "sk_ax":
        return items

    filtered = [
        item
        for item in items
        if isinstance(item, dict) and _is_sk_ax_page_text(str(item.get("text") or ""))
    ]
    if filtered:
        log.info(
            "SK AX LLM 입력 페이지 필터 적용 | before=%d after=%d pages=%s",
            len(items),
            len(filtered),
            [item.get("page") for item in filtered if isinstance(item, dict)],
        )
        return filtered

    log.warning("SK AX LLM 입력 페이지를 찾지 못해 원본 컨텍스트로 fallback | items=%d", len(items))
    return items


def _is_sk_ax_page_text(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return False

    has_strong_ax_term = any(_contains_term(lowered, term) for term in _SK_AX_PAGE_STRONG_TERMS)
    if has_strong_ax_term:
        return True

    has_portfolio_term = any(_contains_term(lowered, term) for term in _SK_AX_PORTFOLIO_TERMS)
    if has_portfolio_term:
        return False

    business_hits = sum(
        1 for term in _SK_AX_PAGE_BUSINESS_TERMS if _contains_term(lowered, term)
    )
    return business_hits >= 2


def _parse_json_response(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_llm_metrics(value: Any, *, peer_id: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    metrics: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        metric_name = _normalize_metric_name(item.get("metric_name") or item.get("metric"))
        if metric_name not in _ALLOWED_METRICS:
            continue
        numeric_value = _number(item.get("value_numeric"))
        if numeric_value is None:
            continue
        confidence = _confidence(item.get("confidence"))
        if confidence < 0.6:
            continue
        if not _is_allowed_peer_item(item, peer_id=peer_id):
            continue
        unit = (
            str(item.get("unit") or "%").strip()
            if metric_name.endswith("margin")
            else str(item.get("unit") or "억원").strip()
        )
        value_kind = (
            "percentage" if unit == "%" or metric_name.endswith("margin") else "amount_krwbn"
        )
        metric: dict[str, Any] = {
            "type": metric_name,
            "metric_label": item.get("metric_label"),
            "metric_scope": _metric_scope(item.get("metric_scope")),
            "business_area": str(item.get("business_area") or "company_total").strip()
            or "company_total",
            "period": item.get("period"),
            "period_year": _int_or_none(item.get("period_year")),
            "period_quarter": _int_or_none(item.get("period_quarter")),
            "period_type": item.get("period_type"),
            "page": _int_or_none(item.get("source_page")),
            "value_kind": value_kind,
            "unit": unit,
            "raw": item.get("evidence_text") or item.get("raw"),
            "evidence_text": item.get("evidence_text") or item.get("raw"),
            "source": "ir_llm_analysis",
            "table_title": item.get("table_title"),
            "row_label": item.get("row_label"),
            "metric_parent_label": item.get("parent_row_label"),
            "column_label": item.get("column_label"),
            "confidence": confidence,
        }
        if value_kind == "percentage":
            metric["value_pct"] = numeric_value
        else:
            metric["value_krwbn"] = numeric_value
            metric["value_krw"] = numeric_value * 100_000_000
        metrics.append(metric)
    return metrics


def _normalize_llm_signals(value: Any, *, peer_id: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    signals: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        signal_type = str(item.get("signal_type") or "").strip()
        if signal_type not in _ALLOWED_SIGNAL_TYPES:
            continue
        evidence = str(item.get("evidence_text") or "").strip()
        summary = str(item.get("summary") or evidence[:160]).strip()
        confidence = _confidence(item.get("confidence"))
        if not evidence or confidence < 0.55:
            continue
        if not _is_allowed_peer_item(item, peer_id=peer_id):
            continue
        signals.append(
            {
                "business_area": str(item.get("business_area") or "company_total").strip()
                or "company_total",
                "signal_type": signal_type,
                "sentiment": _sentiment(item.get("sentiment")),
                "summary": summary[:240],
                "evidence_text": evidence[:1200],
                "source_page": _int_or_none(item.get("source_page")),
                "confidence": confidence,
                "source": "ir_llm_analysis",
            }
        )
    return signals


def _normalize_metric_name(value: Any) -> str:
    cleaned = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    for metric_name, aliases in _METRIC_ALIASES.items():
        if cleaned in aliases:
            return metric_name
    return cleaned


def _peer_id(article: dict[str, Any], parser_result: dict[str, Any]) -> str | None:
    peer_id = parser_result.get("peer_id") or article.get("peer_id")
    if peer_id:
        return str(peer_id)
    company = article.get("company")
    if isinstance(company, list) and company:
        return str(company[0])
    if isinstance(company, str):
        return company
    return None


def _peer_guardrail(peer_id: Any) -> str:
    if str(peer_id or "") != "sk_ax":
        return ""
    excluded = ", ".join(_SK_AX_PORTFOLIO_TERMS[:10])
    return (
        "\nSK AX 특수 규칙:"
        "\n- peer_id가 sk_ax이면 SK주식회사 C&C/SK AX의 IT서비스, Enterprise IT, "
        "SI, ITO, 클라우드, 물류 사업만 추출하세요."
        "\n- SK Inc. IR 안의 투자/포트폴리오 회사 실적은 제외하세요."
        f"\n- 특히 다음 회사/부문은 business_area와 evidence에서 보이면 제외하세요: {excluded}."
    )


def _is_allowed_peer_item(item: dict[str, Any], *, peer_id: str | None) -> bool:
    if peer_id != "sk_ax":
        return True

    combined = " ".join(
        str(item.get(key) or "")
        for key in (
            "business_area",
            "evidence_text",
            "summary",
            "table_title",
            "row_label",
            "parent_row_label",
        )
    ).lower()
    if not combined.strip():
        return True

    has_portfolio_term = any(_contains_term(combined, term) for term in _SK_AX_PORTFOLIO_TERMS)
    if not has_portfolio_term:
        return True

    has_strong_ax_term = any(_contains_term(combined, term) for term in _SK_AX_STRONG_TERMS)
    return has_strong_ax_term


def _contains_term(lowered_text: str, term: str) -> bool:
    lowered_term = term.lower()
    if len(lowered_term) <= 3 and re.fullmatch(r"[a-z0-9&]+", lowered_term):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(lowered_term)}(?![a-z0-9])",
                lowered_text,
            )
        )
    return lowered_term in lowered_text


def _number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _confidence(value: Any) -> float:
    parsed = _number(value)
    if parsed is None:
        return 0.7
    return max(0.0, min(1.0, parsed))


def _metric_scope(value: Any) -> str:
    scope = str(value or "").strip()
    return scope if scope in {"company_total", "segment", "portfolio_company"} else "company_total"


def _sentiment(value: Any) -> str:
    sentiment = str(value or "neutral").strip()
    return sentiment if sentiment in {"positive", "neutral", "negative"} else "neutral"
