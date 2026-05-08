"""문서형 소스의 파싱 품질을 점검하는 에이전트.

DART, IR, 증권사 리포트는
수집/파싱 품질을 먼저 확인한 뒤 문서형 입력으로 통과시킨다.
"""

from __future__ import annotations

from typing import Any

from src.agents.parser_agent import ParserAgent

MIN_CONTENT_CHARS_BY_SOURCE_TYPE = {
    "dart": 80,
    "ir": 200,
    "securities_report": 500,
}


def analyze_parser_quality_article(article: dict[str, Any]) -> tuple[dict[str, Any], bool, str | None]:
    """문서형 article에 parser quality 결과를 붙인다."""
    item = dict(article)
    parser_result = ParserAgent().parse_article(item)

    quality = _quality_from_parser_result(item, parser_result)
    item["parser_result"] = parser_result
    item["parser_quality_score"] = quality["score"]
    item["parser_quality_label"] = quality["label"]
    item["parser_quality_reason"] = quality["reason"]

    if not quality["ok"]:
        item["processing_status"] = "SKIPPED_PARSER_QUALITY"
        item["skip_reason"] = quality["reason"]
        return item, False, quality["reason"]

    return item, True, None


def _quality_from_parser_result(
    article: dict[str, Any],
    parser_result: dict[str, Any],
) -> dict[str, Any]:
    source_type = _source_type(article)
    content = str(article.get("content") or "")
    min_chars = MIN_CONTENT_CHARS_BY_SOURCE_TYPE.get(source_type, 200)
    warnings = parser_result.get("warnings") or []

    score = 0.0
    reasons: list[str] = []

    if parser_result.get("ok"):
        score += 0.35
    else:
        reasons.append("parser_result ok=false")

    has_content = len(content) >= min_chars
    has_dart_metadata = source_type == "dart" and _has_dart_metadata(article, parser_result)

    if has_content:
        score += 0.35
    elif has_dart_metadata:
        score += 0.30
        score = max(score, 0.65)
    else:
        reasons.append(f"content too short:{len(content)}<{min_chars}")

    if article.get("title"):
        score += 0.10
    else:
        reasons.append("empty_title")

    if article.get("url"):
        score += 0.10
    else:
        reasons.append("empty_url")

    if _normalize_company(article.get("company")):
        score += 0.10
    elif source_type in {"dart", "ir", "securities_report"}:
        reasons.append("empty_company")

    if source_type in {"dart", "ir"} and parser_result.get("period"):
        score = max(score, 0.75)

    if warnings:
        score = max(score - min(0.20, 0.05 * len(warnings)), 0.0)

    score = round(min(score, 1.0), 3)
    content_or_metadata_ok = has_content or has_dart_metadata
    ok = score >= 0.60 and bool(parser_result.get("ok")) and content_or_metadata_ok

    if ok:
        label = "pass"
        if has_dart_metadata and not has_content:
            reason = "DART 공시 메타데이터 기준 통과"
        else:
            reason = "문서 파싱 품질 기준 통과"
    else:
        label = "fail"
        reason = "; ".join(reasons or [str(warning) for warning in warnings] or ["parser quality fail"])

    return {
        "ok": ok,
        "score": score,
        "label": label,
        "reason": reason,
    }


def _source_type(article: dict[str, Any]) -> str:
    return str(article.get("source_type") or "").strip().lower()


def _has_dart_metadata(article: dict[str, Any], parser_result: dict[str, Any]) -> bool:
    extra = article.get("extra") or article.get("metadata") or {}
    if not isinstance(extra, dict):
        extra = {}

    receipt_no = (
        parser_result.get("rcept_no")
        or parser_result.get("receipt_no")
        or extra.get("rcept_no")
        or extra.get("receipt_no")
    )
    report_name = parser_result.get("report_name") or extra.get("report_name") or article.get("title")
    corp_code = parser_result.get("corp_code") or extra.get("corp_code")

    return bool(receipt_no and report_name and (corp_code or _normalize_company(article.get("company"))))


def _normalize_company(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value if item]

    if isinstance(value, tuple):
        return [str(item) for item in value if item]

    if isinstance(value, str):
        return [value] if value.strip() else []

    return []


__all__ = ["analyze_parser_quality_article"]
