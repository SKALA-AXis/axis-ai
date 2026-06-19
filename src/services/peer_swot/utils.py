"""peer_swot utils — extracted from facade (move-only)."""

# ruff: noqa: E501  — long lines inherited from E501-exempt facade

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.services.peer_swot.prompts import (  # noqa: F401
    COMPARISON_LABELS,
    COMPARISON_PROMPT,
    COMPARISON_SIGNAL_TYPES,
    COMPARISON_SOURCE_TYPES,
    COMPETITOR_PEER_IDS,
    COMPETITOR_PEER_NAMES,
    DEFAULT_MODEL,
    FINANCIAL_NUMBER_PATTERN,
    GENERIC_CHANGE_OBJECT_BY_LABEL,
    GENERIC_SWOT_TITLE_BY_LABEL,
    NOISE_PHRASES,
    PROMPT_VERSION,
    SK_AX_ID,
    SWOT_FACTOR_TYPE_BY_LABEL,
    SWOT_LABELS,
    SWOT_METRIC_NAMES,
    SWOT_PROMPT,
    SWOT_SIGNAL_TYPES,
    TARGET_PEER_IDS,
)

log = logging.getLogger("generate_peer_swot_llm_preview")


def build_overall_rules(peer: dict[str, Any], companies: list[dict[str, Any]]) -> list[str]:
    if peer.get("id") != "all":
        return []
    return [
        "전체 결과는 특정 1개 회사가 아니라 peer 그룹의 IT 서비스 산업 흐름을 요약한다.",
        "가능하면 2개 이상 peer에서 반복되는 AI, 클라우드, 보안, 데이터, 자동화, 산업 DX 흐름을 우선한다.",
        "한 회사 근거만 있는 경우 전체 트렌드로 단정하지 말고 근거 제한을 명시한다.",
    ]


def content_overlaps_comparison(value: str, comparison_texts: list[str]) -> bool:
    if not value:
        return False
    return any(
        is_semantically_close(value, comparison_text) for comparison_text in comparison_texts
    )


def is_overall_pack(evidence_pack: dict[str, Any]) -> bool:
    return evidence_pack.get("peer", {}).get("id") == "all"


def extract_theme_terms(text_value: str) -> list[str]:
    lowered = str(text_value or "").lower()
    candidates = [
        ("AI", ("ai", "생성형", "llm", "agent", "에이전트")),
        ("클라우드", ("cloud", "클라우드", "msp", "gpu", "데이터센터")),
        ("데이터", ("데이터", "data")),
        ("보안", ("보안", "security", "인증", "암호")),
        ("자동화", ("자동화", "automation", "robot", "로봇")),
        ("산업 DX", ("dx", "스마트팩토리", "제조", "산업")),
        ("ERP", ("erp",)),
        ("모빌리티", ("모빌리티", "차량", "ota", "자동차")),
    ]
    themes: list[str] = []
    for label, terms in candidates:
        if any(term in lowered for term in terms):
            themes.append(label)
    return themes[:4]


def count_distinct_peers_for_refs(refs: list[str], signals: list[dict[str, Any]]) -> int:
    by_ref = {str(signal.get("evidence_id")): signal for signal in signals}
    return len(
        {
            str(signal.get("peer_id"))
            for ref in refs
            for signal in [by_ref.get(ref)]
            if signal and signal.get("peer_id")
        }
    )


def sanitize_overall_company_names(value: str) -> str:
    text_value = str(value or "")
    for name in COMPETITOR_PEER_NAMES:
        text_value = text_value.replace(name, "일부 기업")
    return re.sub(r"(일부 기업[,·/ ]*){2,}", "일부 기업 ", text_value).strip()


def diagnostic_for_label(label: str, diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return next((item for item in diagnostics if item.get("label") == label), {})


def build_overall_reasoning_coverage_note(
    evidence_pack: dict[str, Any] | None,
    refs: list[str],
) -> str:
    if not evidence_pack or evidence_pack.get("peer", {}).get("id") != "all":
        return ""
    ref_set = set(refs)
    covered_peer_names: list[str] = []
    for company in evidence_pack.get("companies") or []:
        peer = company.get("peer") if isinstance(company, dict) else None
        peer_name = peer.get("name") if isinstance(peer, dict) else None
        company_refs = {
            str(item.get("evidence_id"))
            for item in company.get("comparison_signals") or []
            if isinstance(item, dict)
        }
        if ref_set & company_refs and peer_name:
            covered_peer_names.append(str(peer_name))
    if len(covered_peer_names) >= 2:
        return "여러 기업 근거가 같은 방향을 가리켜 국내 IT서비스 시장의 공통 흐름으로 보았습니다. "
    if len(covered_peer_names) == 1:
        return "다만 직접 근거가 일부 기업에 집중되어 시장 전체 판단에는 제한이 있습니다. "
    return "시장 공통 흐름으로 볼 직접 근거는 제한적입니다. "


def build_signal_source_citation(item: dict[str, Any]) -> str:
    source_name = human_source_name(item.get("source_name") or item.get("source_type"))
    title = compact_text(item.get("title"), 90) or "제목 미확인 원문"
    date_value = compact_date(item.get("date") or item.get("updated_at"))
    url = compact_text(item.get("url"), 120)
    parts = [source_name, f"'{title}'"]
    if date_value:
        parts.append(date_value)
    citation = " ".join(part for part in parts if part)
    if url:
        citation = f"{citation}, {url}"
    return compact_text(citation, 240)


def build_metric_source_citation(item: dict[str, Any]) -> str:
    source_name = human_source_name(item.get("source_name") or "공시/IR 원문")
    title = compact_text(item.get("title"), 90) or "재무 지표 원문"
    date_value = compact_date(item.get("date") or item.get("updated_at"))
    metric = compact_text(item.get("metric_label") or item.get("metric_name"), 50)
    period = compact_text(item.get("period"), 20)
    url = compact_text(item.get("url"), 120)
    detail = " ".join(value for value in (period, metric) if value)
    citation = " ".join(part for part in (source_name, f"'{title}'", date_value, detail) if part)
    if url:
        citation = f"{citation}, {url}"
    return compact_text(citation, 240)


def human_source_name(value: Any) -> str:
    raw = str(value or "").strip()
    aliases = {
        "dart": "DART 공시",
        "ir": "IR 자료",
        "securities_report": "증권사 리포트",
        "naver_research": "네이버 증권 리서치",
        "공시/IR 원문": "공시/IR 원문",
    }
    return aliases.get(raw, raw or "공개 원문")


def compact_date(value: Any) -> str:
    text_value = str(value or "").strip()
    if not text_value:
        return ""
    match = re.search(r"\d{4}-\d{2}-\d{2}", text_value)
    return match.group(0) if match else compact_text(text_value, 20)


def build_overall_check_point(
    comparison_points: list[dict[str, Any]],
    swot_items: list[dict[str, Any]],
) -> str:
    objects = [item.get("change_object") for item in comparison_points if item.get("change_object")]
    titles = [item.get("title") for item in swot_items if item.get("title")]
    values = [str(value) for value in [*objects, *titles] if value]
    return (
        compact_text(" / ".join(values[:4]), 180)
        or "다음 분기에도 사업, 기술, 리스크, SWOT 진단 근거를 분리해 확인합니다."
    )


def collect_evidence_ids_from_result(result: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for section_name in ("comparison_points", "swot", "analysis_trace"):
        for item in result.get(section_name) or []:
            if not isinstance(item, dict):
                continue
            for ref in item.get("evidence_refs") or []:
                if isinstance(ref, str) and ref.strip():
                    refs.add(ref.strip())
    return refs


def iter_evidence_items(evidence_pack: dict[str, Any]):
    for company in evidence_pack.get("companies") or []:
        if not isinstance(company, dict):
            continue
        for key in (
            "comparison_signals",
            "swot_source_signals",
            "financial_metrics",
            "diagnostic_evidence",
        ):
            for item in company.get(key) or []:
                if isinstance(item, dict):
                    yield item
    for key in ("comparison_input", "swot_input"):
        section = evidence_pack.get(key)
        if not isinstance(section, dict):
            continue
        for nested_key in ("signals", "diagnostic_evidence"):
            for item in section.get(nested_key) or []:
                if isinstance(item, dict):
                    yield item


def build_provenance(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "generator": "src/services/peer_swot_llm_preview.py",
        "prompt_version": result.get("prompt_version") or PROMPT_VERSION,
        "model_name": result.get("model_name"),
        "evidence_hash": result.get("evidence_hash"),
    }


def rough_token_count(text_value: str) -> int:
    return max(1, int(len(text_value) / 2.2))


def print_or_write_json(payload: dict[str, Any], output: str | None) -> None:
    text_value = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_value + "\n", encoding="utf-8")
        log.info("JSON written: %s", path)
    else:
        print(text_value)


def parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def canonical_comparison_label(value: Any) -> str | None:
    label = str(value or "").strip()
    aliases = {
        "사업 변화": "사업 신호",
        "비즈니스 신호": "사업 신호",
        "기술 변화": "기술 신호",
        "테크 신호": "기술 신호",
        "리스크 변화": "리스크",
        "위험": "리스크",
    }
    label = aliases.get(label, label)
    return label if label in COMPARISON_LABELS else None


def canonical_swot_label(value: Any) -> str | None:
    label = str(value or "").strip()
    aliases = {
        "강점": "Strength",
        "Strengths": "Strength",
        "약점": "Weakness",
        "Weaknesses": "Weakness",
        "기회": "Opportunity",
        "Opportunities": "Opportunity",
        "위협": "Threat",
        "Threats": "Threat",
    }
    label = aliases.get(label, label)
    return label if label in SWOT_LABELS else None


def normalize_refs(raw_refs: Any, allowed_refs: set[str]) -> list[str]:
    if not isinstance(raw_refs, list):
        return []
    refs: list[str] = []
    for ref in raw_refs:
        if not isinstance(ref, str):
            continue
        ref = ref.strip()
        if ref in allowed_refs and ref not in refs:
            refs.append(ref)
    return refs[:3]


def choose_signal_refs_for_label(label: str, signals: list[dict[str, Any]]) -> list[str]:
    if label == "기술 신호":
        selected = [s for s in signals if str(s.get("signal_type")) == "rd" or has_tech_term(s)]
    elif label == "리스크":
        selected = [s for s in signals if str(s.get("signal_type")) == "risk" or has_risk_term(s)]
    else:
        selected = [s for s in signals if str(s.get("signal_type")) not in {"rd", "risk"}]
    if not selected:
        selected = signals
    return [str(item["evidence_id"]) for item in selected[:2] if item.get("evidence_id")]


def choose_diagnostic_refs_for_label(label: str, diagnostics: list[dict[str, Any]]) -> list[str]:
    selected = [item for item in diagnostics if item.get("label") == label]
    return [str(item["evidence_id"]) for item in selected[:1] if item.get("evidence_id")]


def ref_matches_label(ref: str, label: str) -> bool:
    lower = ref.lower()
    label_slug = {
        "Strength": ":strength:",
        "Weakness": ":weakness:",
        "Opportunity": ":opportunity:",
        "Threat": ":threat:",
    }[label]
    return label_slug in lower


def collect_urls_for_refs(refs: list[str], signals: list[dict[str, Any]]) -> list[str]:
    by_ref = {item.get("evidence_id"): item for item in signals}
    urls: list[str] = []
    for ref in refs:
        url = by_ref.get(ref, {}).get("url")
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)
    return urls[:3]


def infer_change_object(label: str, refs: list[str], signals: list[dict[str, Any]]) -> str:
    by_ref = {item.get("evidence_id"): item for item in signals}
    candidates = [by_ref[ref] for ref in refs if ref in by_ref] or signals
    for item in candidates:
        for key in ("business_area", "title", "summary"):
            value = compact_text(item.get(key), 80)
            extracted = extract_specific_phrase(value)
            if extracted:
                return extracted
    defaults = {
        "사업 신호": GENERIC_CHANGE_OBJECT_BY_LABEL["사업 신호"],
        "기술 신호": GENERIC_CHANGE_OBJECT_BY_LABEL["기술 신호"],
        "리스크": GENERIC_CHANGE_OBJECT_BY_LABEL["리스크"],
    }
    return defaults[label]


def extract_specific_phrase(text_value: str | None) -> str:
    text_value = compact_text(text_value, 90)
    if not text_value:
        return ""
    quoted = re.findall(
        r"[A-Za-z][A-Za-z0-9+.#/-]{2,}|[가-힣A-Za-z0-9+.#/-]{2,}(?:AI|AX|DX|클라우드|플랫폼|서비스|솔루션|팩토리|자동화)",
        text_value,
    )
    if quoted:
        return compact_text(quoted[0], 60)
    parts = re.split(r"[,/·\s]+", text_value)
    for part in parts:
        part = part.strip()
        if len(part) >= 3 and not part.isdigit():
            return part[:60]
    return text_value[:60]


def is_generic_change_object(value: str) -> bool:
    normalized = re.sub(r"\s+", "", str(value or "").lower())
    generic_values = {
        "",
        "사업",
        "사업활동",
        "비즈니스",
        "서비스",
        "플랫폼",
        "솔루션",
        "기술",
        "고객",
        "시장",
        "리스크",
        "위험",
        "실행리스크",
        "시장리스크",
    }
    return normalized in {re.sub(r"\s+", "", item.lower()) for item in generic_values}


def is_semantically_close(left: str, right: str) -> bool:
    left_norm = normalize_similarity_text(left)
    right_norm = normalize_similarity_text(right)
    if not left_norm or not right_norm:
        return False
    ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    if ratio >= 0.58:
        return True
    left_tokens = set(extract_similarity_tokens(left_norm))
    right_tokens = set(extract_similarity_tokens(right_norm))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.62


def normalize_similarity_text(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣]+", " ", str(value or "").lower())
    stopwords = {
        "관련",
        "최근",
        "확인",
        "확인됩니다",
        "움직임",
        "신호",
        "기준",
        "판단",
        "요인",
        "수준",
        "가능",
        "필요",
        "있습니다",
        "합니다",
    }
    tokens = [token for token in value.split() if token not in stopwords and len(token) > 1]
    return " ".join(tokens)


def extract_similarity_tokens(value: str) -> list[str]:
    return [token for token in value.split() if len(token) >= 2]


def fallback_swot_title(label: str, diagnostics: list[dict[str, Any]]) -> str:
    diagnostic = next((item for item in diagnostics if item.get("label") == label), None)
    summaries = diagnostic.get("signal_summaries") if isinstance(diagnostic, dict) else []
    phrase = ""
    if summaries:
        phrase = extract_specific_phrase(str(summaries[0]))
    defaults = {
        "Strength": GENERIC_SWOT_TITLE_BY_LABEL["Strength"],
        "Weakness": GENERIC_SWOT_TITLE_BY_LABEL["Weakness"],
        "Opportunity": GENERIC_SWOT_TITLE_BY_LABEL["Opportunity"],
        "Threat": GENERIC_SWOT_TITLE_BY_LABEL["Threat"],
    }
    if phrase:
        return f"{phrase} 기반 {defaults[label]}"
    return defaults[label]


def fallback_check_point(label: str) -> str:
    return {
        "Strength": "동일 역량이 다른 고객·산업으로 반복 확장되는지 확인",
        "Weakness": "비용, 인력, 일정 부담이 실적과 납품 품질에 미치는 영향 확인",
        "Opportunity": "외부 수요와 고객 투자 흐름이 실제 계약으로 전환되는지 확인",
        "Threat": "경쟁, 규제, 수요 둔화가 사업 속도를 낮추는지 확인",
    }[label]


def has_tech_term(item: dict[str, Any]) -> bool:
    text_value = joined_item_text(item).lower()
    return any(
        term in text_value
        for term in (
            "ai",
            "ax",
            "dx",
            "cloud",
            "클라우드",
            "플랫폼",
            "데이터",
            "보안",
            "자동화",
            "llm",
            "생성형",
        )
    )


def has_risk_term(item: dict[str, Any]) -> bool:
    text_value = joined_item_text(item).lower()
    return any(
        term in text_value
        for term in (
            "risk",
            "리스크",
            "위험",
            "부담",
            "지연",
            "둔화",
            "경쟁",
            "규제",
            "수익성",
            "비용",
            "하락",
            "감소",
        )
    )


def has_negative_metric_signal(metric: dict[str, Any]) -> bool:
    text_value = joined_item_text(metric).lower()
    value = metric.get("value_numeric")
    return (
        any(
            term in text_value
            for term in ("감소", "하락", "둔화", "적자", "손실", "마진", "수익성")
        )
        or bool(re.search(r"(?i)(yoy|qoq|전년|전분기).{0,20}-\d", text_value))
        or (isinstance(value, int | float) and value < 0)
    )


def joined_item_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "business_area",
            "signal_type",
            "summary",
            "evidence_text",
            "title",
            "metric_name",
            "metric_label",
        )
    )


def summarize_profile(profile_snapshot: Any) -> str:
    if profile_snapshot is None:
        return ""
    if isinstance(profile_snapshot, str):
        try:
            profile_snapshot = json.loads(profile_snapshot)
        except json.JSONDecodeError:
            return compact_text(profile_snapshot, 360)
    if isinstance(profile_snapshot, dict):
        parts: list[str] = []
        for key in (
            "summary",
            "businessOverview",
            "business_overview",
            "description",
            "mainBusiness",
        ):
            value = profile_snapshot.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        for key in ("keywords", "businessKeywords", "technologyKeywords"):
            value = profile_snapshot.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value[:8])
        return compact_text(" / ".join(parts), 420)
    return compact_text(str(profile_snapshot), 360)


def clean_display_text(value: Any) -> str:
    text_value = compact_text(value, 1000)
    for phrase in NOISE_PHRASES:
        text_value = text_value.replace(phrase, "")
    text_value = re.sub(r"\s+", " ", text_value).strip()
    return text_value


def is_weak_info_text(value: str) -> bool:
    text_value = clean_display_text(value)
    if len(text_value) < 28:
        return True
    weak_phrases = (
        "근거를 기준으로",
        "보수적으로 작성",
        "확인 가능한",
        "분류했습니다",
        "판단했습니다",
        "short korean",
        "reasoning",
        "evidence",
    )
    if any(phrase in text_value.lower() for phrase in weak_phrases):
        return len(text_value) < 80
    return False


def remove_financial_number_sentences(text_value: str) -> str:
    sentences = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+", text_value)
    kept = [sentence for sentence in sentences if not FINANCIAL_NUMBER_PATTERN.search(sentence)]
    result = " ".join(kept).strip()
    if result:
        return result
    return FINANCIAL_NUMBER_PATTERN.sub("", text_value).strip()


def normalize_peer(peer: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": peer.get("id"),
        "name": peer.get("name"),
        "profile_summary": summarize_profile(peer.get("profile_snapshot")),
    }


def flatten_limited(groups: list[list[dict[str, Any]]], *, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    index = 0
    while len(items) < limit:
        added = False
        for group in groups:
            if index >= len(group):
                continue
            item = group[index]
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                items.append(item)
                added = True
                if len(items) >= limit:
                    break
        if not added:
            break
        index += 1
    return items


def unique_evidence_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        unique_items.append(item)
    return unique_items


def is_usable_signal_text(summary: str, evidence_text: str) -> bool:
    combined = f"{summary} {evidence_text}".strip()
    if len(combined) < 24:
        return False
    if re.search(r"^(그림|표|table|figure)\s*\d*", summary.strip(), re.IGNORECASE):
        return False
    korean_alpha = len(re.findall(r"[가-힣A-Za-z]", combined))
    noisy_chars = len(re.findall(r"[{}<>『』|=]", combined))
    if korean_alpha < 16:
        return False
    if noisy_chars >= 6 and noisy_chars > korean_alpha * 0.15:
        return False
    return True


def extract_numeric_ids(evidence_refs: set[str], prefix: str) -> set[int]:
    ids: set[int] = set()
    for ref in evidence_refs:
        if not ref.startswith(prefix):
            continue
        raw_id = ref.removeprefix(prefix)
        if raw_id.isdigit():
            ids.add(int(raw_id))
    return ids


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]


def compact_text(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    text_value = re.sub(r"\s+", " ", str(value)).strip()
    if len(text_value) <= limit:
        return text_value
    return text_value[: max(0, limit - 1)].rstrip() + "…"


def iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp_confidence(value: Any) -> float:
    numeric = to_float(value)
    if numeric is None:
        return 0.6
    return max(0.0, min(1.0, numeric))
