"""LLM-assisted DART business analysis extraction."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from src.config.openai_policy import openai_calls_enabled, openai_disabled_reason
from src.llm import LLMSpec, build_chat_llm

log = logging.getLogger(__name__)

_LLM_MODEL = os.getenv("DART_ANALYSIS_LLM_MODEL", "gpt-4o-mini")
_MAX_CHUNKS = int(os.getenv("DART_LLM_MAX_CHUNKS", "10"))
_MAX_CHUNK_CHARS = int(os.getenv("DART_LLM_CHUNK_CHARS", "2600"))
_MAX_TABLES = int(os.getenv("DART_LLM_MAX_TABLES", "8"))
_TARGET_SECTIONS = {"company_overview", "business"}
_FINANCIAL_SUMMARY_SECTIONS = {"financial"}
_ALLOWED_SIGNAL_TYPES = {
    "growth",
    "strategy",
    "efficiency",
    "risk",
    "orders_pipeline",
    "investment",
    "product_service",
    "business_overview",
    "rd",
}
_SK_AX_STRONG_TERMS = (
    "sk ax",
    "sk에이엑스",
    "sk㈜ c&c",
    "sk주식회사 c&c",
    "sk c&c",
    "sk c & c",
    "씨앤씨",
    "c&c",
    "sk주식회사 사업부문",
    "sk 주식회사 사업부문",
    "sk주식회사는 국내 top-tier it 서비스",
)
_SK_AX_BUSINESS_TERMS = (
    "it서비스",
    "it 서비스",
    "it services",
    "it 컨설팅",
    "시스템 구축",
    "아웃소싱",
    "outsourcing",
    "cloud",
    "클라우드",
    "ai",
    "digital transformation",
    "ai/digital transformation",
    "agentic",
    "it 비용절감",
    "delivery 역량",
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

_llm: ChatOpenAI | None = None


def analyze_dart_with_llm(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    *,
    max_chunks: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run optional LLM extraction on selected DART sections."""
    if not openai_calls_enabled():
        log.warning("DART LLM 분석 스킵 | reason=%s", openai_disabled_reason())
        return {"llm_business_signals": []}

    if not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY 미설정: DART LLM 분석 스킵")
        return {"llm_business_signals": []}

    prompt = _build_dart_llm_prompt(article, parser_result, max_chunks=max_chunks)
    if not prompt:
        return {"llm_business_signals": []}

    try:
        from src.observability import tracing_config

        config = tracing_config(
            agent="DARTLLMAnalysisExtractor",
            phase="extract",
            peer_id=_peer_id(article, parser_result),
        )
    except Exception:
        config = None

    response = _get_llm().invoke(
        [
            (
                "system",
                "You extract structured business facts from Korean DART filings. "
                "Return JSON only. Do not infer facts that are not explicitly present.",
            ),
            ("user", prompt),
        ],
        config=config,
    )
    parsed = _parse_json_response(getattr(response, "content", response))
    peer_id = _peer_id(article, parser_result)
    return {
        "llm_business_signals": _normalize_llm_signals(
            parsed.get("business_signals"),
            peer_id=peer_id,
        )
    }


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        # env 로 모델 지정 가능 → gpt-5 라도 reasoning_effort 미전달(기존 동작) 위해 None.
        _llm = build_chat_llm(
            LLMSpec(
                model=_LLM_MODEL,
                temperature=0,
                max_tokens=3000,
                json_object=True,
                reasoning_effort=None,
            )
        )
    return _llm


def _build_dart_llm_prompt(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    *,
    max_chunks: int | None = None,
) -> str:
    peer_id = _peer_id(article, parser_result)
    chunks = _chunk_contexts(
        parser_result,
        peer_id=peer_id,
        max_chunks=max_chunks or _MAX_CHUNKS,
    )
    table_summaries = _financial_table_summaries(parser_result)
    if not chunks and not table_summaries:
        return ""

    metadata = {
        "raw_article_id": article.get("id"),
        "title": article.get("title") or parser_result.get("title"),
        "peer_id": peer_id,
        "report_name": parser_result.get("report_name"),
        "report_period": parser_result.get("period"),
        "rcept_no": parser_result.get("rcept_no"),
    }
    peer_guardrail = _peer_guardrail(peer_id)
    chunk_text = "\n\n".join(chunks)
    table_text = "\n\n".join(table_summaries)
    return f"""
DART 정기보고서에서 회사/사업 구조와 사업 시그널을 추출하세요.

원칙:
- 입력은 주로 '회사의 개요'와 '사업의 내용' 섹션입니다.
- 사업 설명, 사업 개요, 제품 및 서비스, 매출 및 수주상황을 분리하세요.
- 주요계약, 연구개발, 위험 요인도 별도 signal로 분리하세요.
- 재무제표 숫자는 참고용 표 요약으로만 사용하고, 명시되지 않은 수치를 만들지 마세요.
- evidence_text에는 원문 근거 문장을 그대로 충분히 남기세요.
- 확실하지 않은 항목은 confidence를 0.6 미만으로 두세요.
{peer_guardrail}

허용 signal_type:
{sorted(_ALLOWED_SIGNAL_TYPES)}

출력 JSON 형식:
{{
  "business_signals": [
    {{
      "business_area": "company_total|cloud|ai_ax|enterprise_it|orders_pipeline|rd|risk",
      "signal_type": "business_overview|product_service|orders_pipeline|rd|strategy|growth",
      "sentiment": "positive|neutral|negative",
      "summary": "한 문장 요약",
      "evidence_text": "원문 근거 문장",
      "section_key": "business",
      "subsection_title": "사업의 개요",
      "confidence": 0.0
    }}
  ]
}}

문서 메타데이터:
{json.dumps(metadata, ensure_ascii=False)}

선별 본문 컨텍스트:
{chunk_text}

재무/표 요약 컨텍스트:
{table_text}
""".strip()


def _chunk_contexts(
    parser_result: dict[str, Any],
    *,
    peer_id: str | None,
    max_chunks: int,
) -> list[str]:
    chunks = parser_result.get("document_chunks")
    if not isinstance(chunks, list):
        return []

    selected: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        section_key = str(chunk.get("section_key") or "")
        if section_key not in _TARGET_SECTIONS:
            continue
        text = str(chunk.get("text") or "").strip()
        if peer_id == "sk_ax" and not _is_sk_ax_relevant_text(text):
            continue
        selected.append(chunk)

    if peer_id == "sk_ax" and not selected:
        log.warning("SK AX DART LLM 입력 chunk를 찾지 못해 business 섹션으로 fallback")
        selected = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict) and str(chunk.get("section_key") or "") in _TARGET_SECTIONS
        ]

    contexts: list[str] = []
    for chunk in selected[:max_chunks]:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        contexts.append(
            "[{section} / {subsection} / chunk={chunk_id}]\n{text}".format(
                section=chunk.get("section_title") or chunk.get("section_key"),
                subsection=chunk.get("subsection_title") or "전체",
                chunk_id=chunk.get("chunk_id") or chunk.get("chunk_index"),
                text=text[:_MAX_CHUNK_CHARS],
            )
        )
    return contexts


def _financial_table_summaries(parser_result: dict[str, Any]) -> list[str]:
    summaries: list[str] = []
    statements = parser_result.get("financial_statements")
    if isinstance(statements, list):
        for statement in statements[:_MAX_TABLES]:
            if not isinstance(statement, dict):
                continue
            rows = statement.get("rows")
            row_labels = []
            if isinstance(rows, list):
                for row in rows[:12]:
                    if isinstance(row, dict):
                        row_labels.append(str(row.get("label") or row.get("metric_key") or ""))
            summaries.append(
                (
                    "[financial_statement table={table_index} "
                    "type={table_type} unit={unit}]\n{rows}"
                ).format(
                    table_index=statement.get("table_index"),
                    table_type=statement.get("table_type"),
                    unit=statement.get("unit"),
                    rows=", ".join(label for label in row_labels if label),
                )
            )

    chunks = parser_result.get("document_chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            if str(chunk.get("section_key") or "") not in _FINANCIAL_SUMMARY_SECTIONS:
                continue
            text = str(chunk.get("text") or "").strip()
            if text and any(term in text for term in ("재무제표", "매출", "영업이익", "수주")):
                summaries.append(
                    "[financial section / {subsection}]\n{text}".format(
                        subsection=chunk.get("subsection_title") or "전체",
                        text=text[:1200],
                    )
                )
            if len(summaries) >= _MAX_TABLES:
                break
    return summaries[:_MAX_TABLES]


def _normalize_llm_signals(value: Any, *, peer_id: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        signal_type = str(item.get("signal_type") or "").strip()
        if signal_type not in _ALLOWED_SIGNAL_TYPES:
            continue
        evidence = str(item.get("evidence_text") or "").strip()
        confidence = _confidence(item.get("confidence"))
        if not evidence or confidence < 0.6:
            continue
        if peer_id == "sk_ax" and not _is_sk_ax_relevant_text(evidence):
            continue
        rows.append(
            {
                "business_area": str(item.get("business_area") or "company_total").strip()
                or "company_total",
                "signal_type": signal_type,
                "sentiment": _sentiment(item.get("sentiment")),
                "summary": str(item.get("summary") or evidence[:180]).strip()[:240],
                "evidence_text": evidence[:1200],
                "section_key": item.get("section_key"),
                "subsection_title": item.get("subsection_title"),
                "confidence": confidence,
                "source": "dart_llm_analysis",
            }
        )
    return rows


def _peer_guardrail(peer_id: str | None) -> str:
    if peer_id != "sk_ax":
        return ""
    excluded = ", ".join(_SK_GROUP_UNRELATED_TERMS[:10])
    return (
        "\nSK AX 특수 규칙:"
        "\n- peer_id가 sk_ax이면 SK주식회사 사업부문 중 IT서비스/IT컨설팅/시스템 구축/"
        "아웃소싱/Cloud/AI/Digital Transformation 관련 내용만 추출하세요."
        "\n- SK Inc. 투자부문, 지주회사 포트폴리오, 계열회사 실적은 제외하세요."
        f"\n- 특히 다음 용어가 중심이면 제외하세요: {excluded}."
    )


def _is_sk_ax_relevant_text(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return False

    if any(_contains_term(lowered, term) for term in _SK_AX_STRONG_TERMS):
        return True

    if any(_contains_term(lowered, term) for term in _SK_GROUP_UNRELATED_TERMS):
        return False

    business_hits = sum(1 for term in _SK_AX_BUSINESS_TERMS if _contains_term(lowered, term))
    has_sk_business_context = "sk주식회사" in lowered or "sk 주식회사" in lowered
    return business_hits >= 2 or (has_sk_business_context and business_hits >= 1)


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


def _peer_id(article: dict[str, Any], parser_result: dict[str, Any]) -> str | None:
    value = parser_result.get("peer_id") or article.get("peer_id")
    if value:
        return str(value)
    company = article.get("company")
    if isinstance(company, list) and company:
        return str(company[0])
    if isinstance(company, str):
        return company
    return None


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


def _confidence(value: Any) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return 0.7
    return max(0.0, min(1.0, parsed))


def _sentiment(value: Any) -> str:
    sentiment = str(value or "neutral").strip()
    return sentiment if sentiment in {"positive", "neutral", "negative"} else "neutral"
