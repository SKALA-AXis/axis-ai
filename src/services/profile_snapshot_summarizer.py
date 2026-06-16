"""ProfileSnapshotSummarizer — LLM-ready profile snapshot synthesis."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

PROFILE_SNAPSHOT_SCHEMA_VERSION = "peer-profile-snapshot-v1"


class ProfileSnapshotSummarizer:
    """Summarize a selected evidence pack into a peer profile snapshot.

    ``llm`` is intentionally injectable. In production it can be a ChatOpenAI-like
    runnable; in tests and local dry-runs the deterministic fallback keeps the
    pipeline usable without an API call.
    """

    def __init__(self, *, llm: Any | None = None) -> None:
        self._llm = llm

    def summarize(self, evidence_pack: dict[str, Any]) -> dict[str, Any]:
        if self._llm is None:
            return _fallback_snapshot(evidence_pack)
        from src.observability import tracing_config

        prompt = _build_prompt(evidence_pack)
        result = self._llm.invoke(
            prompt,
            config=tracing_config(
                agent="ProfileSnapshotSummarizer",
                phase="summarize",
                prompt_version=PROFILE_SNAPSHOT_SCHEMA_VERSION,
            ),
        )
        content = getattr(result, "content", result)
        snapshot = _parse_json(str(content))
        return _finalize_snapshot(snapshot, evidence_pack)


def _build_prompt(evidence_pack: dict[str, Any]) -> str:
    return f"""
당신은 PeerProfileAgent의 ProfileSnapshotSummarizer입니다.

입력 evidence_pack에 있는 근거만 사용해 회사별 profile_snapshot JSON을 작성하세요.
없는 제품, 수치, 고객, 전략은 만들지 마세요.
DART business_area_evidence의 evidence_text에서 공식 사업영역을 먼저 추출하세요.
business_area_evidence의 business_area 값이 "DART 공식 사업영역 후보"이면 그대로 쓰지 말고,
evidence_text 안의 공식 사업영역명으로 business_areas를 구성하세요.
IR/뉴스룸/증권사 근거는 DART에서 추출한 공식 사업영역에 매핑하세요.
모든 주장에는 가능한 source_ref를 보존하세요.
source_ref는 URL/title 전체를 반복하지 말고 원문 조인 가능한 compact 형태만 사용하세요.
evidence_texts는 사업영역별 최대 2개만 남기고, 각 문장은 160자 이내로 요약하세요.
recent_changes와 capability_evolution.changes도 항목별 summary는 120자 이내로 작성하세요.
source_index는 source_ref 목록 수준으로만 작성하고 원문/URL/title을 반복하지 마세요.
summary, recent_direction, overall_change 같은 설명 문장 안에 "(raw_articles/123)",
"raw_article_business_signals/123", {{"raw_article_id": 123}} 같은 근거 표기를 섞지 마세요.
근거 ID는 반드시 source_ref 또는 source_index에만 넣으세요.
수치 계산은 하지 말고 입력의 financial_evidence 값만 사용하세요.
source_coverage는 입력 값을 요약 없이 보존하세요.
evidence_digest는 source별로 재구성해도 되지만,
입력의 source/period/summary/top_refs를 우선 보존하세요.
증권사 리포트가 unavailable이면 market_view.status를 "unavailable"로 두고 reason을 보존하세요.
증권사 리포트가 available이면 market_evidence에 있는 애널리스트 관점만 market_view에 반영하세요.
증권사 리포트 근거는 business_areas/core_capabilities가 아니라 market_view에만 반영하세요.
market_view.valuation_notes에는 목표주가/상승여력 같은 valuation metric만 넣으세요.
SK AX 대응방향이나 시사점은 작성하지 마세요.

출력은 JSON만 반환하세요. schema:
{{
  "company_id": "...",
  "company_name": "...",
  "schema_version": "{PROFILE_SNAPSHOT_SCHEMA_VERSION}",
  "generated_at": "ISO-8601",
  "one_liner": "",
  "company_summary": "",
  "key_products_services": [],
  "execution_cases": [],
  "strategic_focus": [],
  "priority_initiatives": [],
  "investment_roadmap": [],
  "operational_highlights": [],
  "business_areas": [
    {{
      "name": "",
      "summary": "",
      "core_capabilities": [],
      "recent_direction": "",
      "evidence_texts": [],
      "source_refs": []
    }}
  ],
  "core_capabilities": [],
  "recent_changes": [],
  "evidence_digest": [],
  "financial_summary": {{}},
  "market_view": {{
    "status": "available|unavailable|not_used",
    "reason": "",
    "positive_points": [],
    "watch_points": [],
    "valuation_notes": [],
    "source_refs": []
  }},
  "capability_evolution": {{
    "period": {{"from": "", "to": ""}},
    "overall_change": "",
    "changes": [],
    "watch_points": []
  }},
  "cautions": [],
  "source_coverage": {{}},
  "source_index": []
}}

source_ref 형식:
- 원문 조인용 raw_article_id만 남기세요: {{"raw_article_id": 123}}

evidence_pack:
{json.dumps(evidence_pack, ensure_ascii=False, indent=2, default=str)}
""".strip()


def _fallback_snapshot(evidence_pack: dict[str, Any]) -> dict[str, Any]:
    company = evidence_pack.get("company") or {}
    company_id = str(company.get("id") or company.get("company_id") or "")
    company_name = str(company.get("name") or company.get("company_name") or company_id)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    business_areas = _business_area_snapshots(evidence_pack)
    financial_summary = _financial_summary(evidence_pack)
    official_areas = [str(area.get("name") or "") for area in business_areas if area.get("name")]
    recent_changes = _recent_changes(evidence_pack, official_areas=official_areas)
    core_capabilities = _unique(
        capability
        for area in business_areas
        for capability in area.get("core_capabilities", [])
        if capability
    )

    return {
        "company_id": company_id,
        "company_name": company_name,
        "schema_version": PROFILE_SNAPSHOT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "one_liner": _one_liner(company_name, business_areas),
        "company_summary": _company_summary(company_name, business_areas, evidence_pack),
        "key_products_services": _key_products_services(evidence_pack, business_areas),
        "execution_cases": _execution_cases(evidence_pack, official_areas),
        "strategic_focus": _strategic_focus(evidence_pack, official_areas),
        "priority_initiatives": _priority_initiatives(evidence_pack, official_areas),
        "investment_roadmap": _investment_roadmap(evidence_pack, official_areas),
        "operational_highlights": _operational_highlights(evidence_pack, official_areas),
        "business_areas": business_areas,
        "core_capabilities": core_capabilities,
        "recent_changes": recent_changes,
        "evidence_digest": evidence_pack.get("evidence_digest") or [],
        "financial_summary": financial_summary,
        "market_view": _market_view(evidence_pack),
        "capability_evolution": _capability_evolution(evidence_pack, recent_changes),
        "cautions": _cautions(evidence_pack),
        "source_coverage": evidence_pack.get("source_coverage") or {},
        "source_index": _source_index(evidence_pack),
    }


def _finalize_snapshot(
    snapshot: dict[str, Any],
    evidence_pack: dict[str, Any],
) -> dict[str, Any]:
    source_coverage = evidence_pack.get("source_coverage") or {}
    snapshot = _clean_inline_refs(snapshot)
    snapshot = _normalize_llm_snapshot_shapes(snapshot)
    if source_coverage:
        snapshot["source_coverage"] = source_coverage
    if evidence_pack.get("evidence_digest"):
        snapshot["evidence_digest"] = evidence_pack.get("evidence_digest")
    if not snapshot.get("company_summary"):
        snapshot["company_summary"] = _company_summary(
            str(snapshot.get("company_name") or ""),
            snapshot.get("business_areas") or [],
            evidence_pack,
        )
    official_areas = [
        str(area.get("name") or "")
        for area in snapshot.get("business_areas") or []
        if isinstance(area, dict) and area.get("name")
    ]
    if not snapshot.get("key_products_services"):
        snapshot["key_products_services"] = _key_products_services(
            evidence_pack, snapshot.get("business_areas") or []
        )
    if not snapshot.get("execution_cases"):
        snapshot["execution_cases"] = _execution_cases(evidence_pack, official_areas)
    if not snapshot.get("strategic_focus"):
        snapshot["strategic_focus"] = _strategic_focus(evidence_pack, official_areas)
    if not snapshot.get("priority_initiatives"):
        snapshot["priority_initiatives"] = _priority_initiatives(evidence_pack, official_areas)
    if not snapshot.get("investment_roadmap"):
        snapshot["investment_roadmap"] = _investment_roadmap(evidence_pack, official_areas)
    if not snapshot.get("operational_highlights"):
        snapshot["operational_highlights"] = _operational_highlights(evidence_pack, official_areas)
    if not snapshot.get("recent_changes"):
        snapshot["recent_changes"] = _recent_changes(evidence_pack, official_areas=official_areas)
    capability_evolution = snapshot.get("capability_evolution")
    if not isinstance(capability_evolution, dict) or not capability_evolution.get("changes"):
        snapshot["capability_evolution"] = _capability_evolution(
            evidence_pack,
            snapshot.get("recent_changes") or [],
        )

    securities_report = source_coverage.get("securities_report") or {}
    if securities_report.get("status") == "unavailable" or "market_view" not in snapshot:
        snapshot["market_view"] = _market_view(evidence_pack)
    elif evidence_pack.get("market_evidence"):
        snapshot["market_view"] = _sanitize_market_view_refs(
            snapshot.get("market_view"),
            evidence_pack,
        )
    return snapshot


def _clean_inline_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_inline_refs(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_clean_inline_refs(nested) for nested in value]
    if isinstance(value, str):
        return _strip_inline_ref_text(value)
    return value


def _normalize_llm_snapshot_shapes(snapshot: dict[str, Any]) -> dict[str, Any]:
    business_areas = snapshot.get("business_areas")
    if isinstance(business_areas, list):
        snapshot["business_areas"] = [
            _normalize_business_area_shape(area)
            for area in business_areas
            if isinstance(area, dict)
        ]

    for key in ("recent_changes", "execution_cases", "strategic_focus"):
        if isinstance(snapshot.get(key), list):
            snapshot[key] = [_normalize_ref_container(item) for item in snapshot[key]]

    market_view = snapshot.get("market_view")
    if isinstance(market_view, dict):
        snapshot["market_view"] = _normalize_ref_container(market_view)

    capability_evolution = snapshot.get("capability_evolution")
    if isinstance(capability_evolution, dict) and isinstance(
        capability_evolution.get("changes"), list
    ):
        capability_evolution["changes"] = [
            _normalize_ref_container(item) for item in capability_evolution["changes"]
        ]

    if isinstance(snapshot.get("source_index"), list):
        snapshot["source_index"] = _unique_refs(snapshot["source_index"])

    return snapshot


def _normalize_business_area_shape(area: dict[str, Any]) -> dict[str, Any]:
    evidence_texts = area.get("evidence_texts")
    if isinstance(evidence_texts, list):
        area["evidence_texts"] = [
            _normalize_evidence_text_item(item)
            for item in evidence_texts[:5]
            if _normalize_evidence_text_item(item)
        ]
    else:
        area["evidence_texts"] = []

    area["source_refs"] = _unique_refs(area.get("source_refs") or [])[:6]
    return area


def _normalize_evidence_text_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        text = _compact_text(
            item.get("text") or item.get("claim") or item.get("summary"), limit=180
        )
        if not text:
            return {}
        normalized: dict[str, Any] = {"text": text}
        refs = _unique_refs(item.get("source_refs") or [item.get("source_ref")])
        if refs:
            normalized["source_refs"] = refs[:2]
        return normalized
    if isinstance(item, str):
        text = _compact_text(item, limit=180)
        return {"text": text} if text else {}
    return {}


def _normalize_ref_container(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "source_refs" in value:
        value["source_refs"] = _unique_refs(value.get("source_refs") or [])
    if "source_ref" in value:
        refs = _unique_refs([value.pop("source_ref")])
        if refs:
            value["source_refs"] = _unique_refs([*(value.get("source_refs") or []), *refs])
    return value


def _strip_inline_ref_text(value: str) -> str:
    cleaned = value
    cleaned = re.sub(
        r"\s*\((?:raw_articles|raw_article_business_signals|raw_article_financial_metrics)/\d+\)",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"\s*(?:raw_articles|raw_article_business_signals|raw_article_financial_metrics)/\d+",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\s*\{[\"']raw_article_id[\"']\s*:\s*\d+\s*\}", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _sanitize_market_view_refs(value: Any, evidence_pack: dict[str, Any]) -> dict[str, Any]:
    market_view = dict(value) if isinstance(value, dict) else {}
    market_view.setdefault("status", "available")
    market_view.setdefault("reason", "증권사 리포트 기반 애널리스트 관점")
    market_view.setdefault("positive_points", [])
    market_view.setdefault("watch_points", [])
    market_view.setdefault("valuation_notes", [])
    market_view["source_refs"] = _source_index(
        {"market_evidence": evidence_pack.get("market_evidence") or []}
    )
    return market_view


def _business_area_snapshots(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    by_area: dict[str, dict[str, Any]] = {}
    for item in evidence_pack.get("business_area_evidence") or []:
        area = str(item.get("business_area") or "")
        if not area:
            continue
        if area == "DART 공식 사업영역 후보":
            for inferred_area, claim in _infer_business_areas_from_dart(
                str(item.get("evidence_text") or "")
            ):
                by_area.setdefault(
                    inferred_area,
                    {
                        "name": inferred_area,
                        "summary": claim,
                        "core_capabilities": _capabilities_for_area(inferred_area, claim),
                        "recent_direction": "",
                        "evidence_texts": [],
                        "source_refs": [],
                    },
                )
                _append_evidence_text(by_area[inferred_area]["evidence_texts"], item)
                _append_ref(by_area[inferred_area]["source_refs"], item.get("source_ref"))
            continue
        by_area.setdefault(
            area,
            {
                "name": area,
                "summary": str(item.get("claim") or ""),
                "core_capabilities": _capabilities_for_area(area, str(item.get("claim") or "")),
                "recent_direction": "",
                "evidence_texts": [],
                "source_refs": [],
            },
        )
        _append_evidence_text(by_area[area]["evidence_texts"], item)
        _append_ref(by_area[area]["source_refs"], item.get("source_ref"))

    official_areas = list(by_area)
    for item in evidence_pack.get("direction_evidence") or []:
        area = _map_to_official_area(str(item.get("business_area") or ""), item, official_areas)
        if not area or area == "company_total":
            continue
        entry = by_area.setdefault(
            area,
            {
                "name": area,
                "summary": "",
                "core_capabilities": _capabilities_for_area(area, str(item.get("claim") or "")),
                "recent_direction": "",
                "evidence_texts": [],
                "source_refs": [],
            },
        )
        if not entry["recent_direction"]:
            entry["recent_direction"] = _profile_claim(item)
        entry["core_capabilities"] = _unique(
            [*entry.get("core_capabilities", []), *_capabilities_for_area(area, item.get("claim"))]
        )
        _append_evidence_text(entry["evidence_texts"], item)
        _append_ref(entry["source_refs"], item.get("source_ref"))

    for item in evidence_pack.get("financial_evidence") or []:
        area = _map_to_official_area(str(item.get("business_area") or ""), item, official_areas)
        if not area or area == "company_total":
            continue
        entry = by_area.setdefault(
            area,
            {
                "name": area,
                "summary": "",
                "core_capabilities": _capabilities_for_area(area, ""),
                "recent_direction": "",
                "evidence_texts": [],
                "source_refs": [],
            },
        )
        _append_ref(entry["source_refs"], item.get("source_ref"))

    for item in evidence_pack.get("execution_evidence") or []:
        area = _map_to_official_area(str(item.get("business_area") or ""), item, official_areas)
        if not area or area == "company_total":
            continue
        entry = by_area.setdefault(
            area,
            {
                "name": area,
                "summary": "",
                "core_capabilities": _capabilities_for_area(area, str(item.get("claim") or "")),
                "recent_direction": "",
                "evidence_texts": [],
                "source_refs": [],
            },
        )
        if entry["recent_direction"]:
            claim = _profile_claim(item)
            if claim not in entry["recent_direction"]:
                entry["recent_direction"] = f"{entry['recent_direction']} {claim}"
        else:
            entry["recent_direction"] = _profile_claim(item)
        entry["core_capabilities"] = _unique(
            [*entry.get("core_capabilities", []), *_capabilities_for_area(area, item.get("claim"))]
        )
        _append_evidence_text(entry["evidence_texts"], item)
        _append_ref(entry["source_refs"], item.get("source_ref"))

    for entry in by_area.values():
        entry["source_refs"] = entry.get("source_refs", [])[:6]
        entry["evidence_texts"] = entry.get("evidence_texts", [])[:5]
    return list(by_area.values())


def _map_to_official_area(
    area: str,
    item: dict[str, Any],
    official_areas: list[str],
) -> str:
    if not area or area == "company_total" or not official_areas:
        return area
    if area in official_areas:
        return area

    text_value = f"{item.get('claim') or ''} {item.get('evidence_text') or ''}"
    if "Digital Business Service" in official_areas and area in {"IT서비스/SI"}:
        return "Digital Business Service"
    if "자동화" in official_areas and area in {
        "AI/AX",
        "스마트 엔지니어링",
        "스마트팩토리/자동화",
        "로봇/RX",
    }:
        if any(
            token in text_value
            for token in ("자동화", "로봇", "철강", "제조", "공장", "산업", "Intelligent Factory")
        ):
            return "자동화"

    has_it_service = "IT서비스" in official_areas or "IT 서비스" in official_areas
    it_service_name = "IT서비스" if "IT서비스" in official_areas else "IT 서비스"
    if has_it_service and area in {
        "AI/AX",
        "클라우드",
        "클라우드&AI",
        "IT서비스/SI",
        "Digital Business Service",
    }:
        return it_service_name
    if "물류" in official_areas and area in {"물류", "logistics"}:
        return "물류"
    if (
        has_it_service
        and "물류" in official_areas
        and area
        in {
            "스마트 엔지니어링",
            "스마트팩토리/자동화",
            "로봇/RX",
        }
    ):
        if any(token in text_value for token in ("물류", "Cello", "첼로", "logistics")):
            return "물류"
        return it_service_name
    if (
        has_it_service
        and "차량용 SW" in official_areas
        and area
        in {
            "스마트 엔지니어링",
            "스마트팩토리/자동화",
            "로봇/RX",
        }
    ):
        if any(token in text_value for token in ("차량", "SDV", "내비게이션", "mobilgene")):
            return "차량용 SW"
        return it_service_name
    if "차량용 SW" in official_areas and area in {"차량 SW"}:
        return "차량용 SW"
    if "기타" in official_areas and area in {"물류"}:
        return "기타"

    if "클라우드&AI" in official_areas and area in {"AI/AX", "클라우드", "IT서비스/SI"}:
        return "클라우드&AI"
    if "스마트 엔지니어링" in official_areas and area in {
        "물류",
        "로봇/RX",
        "스마트팩토리/자동화",
    }:
        return "스마트 엔지니어링"
    for official_area in official_areas:
        if official_area and official_area in text_value:
            return official_area
    return area


def _financial_summary(evidence_pack: dict[str, Any]) -> dict[str, Any]:
    company_total: dict[str, Any] = {}
    segment_revenue: list[dict[str, Any]] = []
    for item in evidence_pack.get("financial_evidence") or []:
        payload = {
            "value": item.get("value"),
            "unit": item.get("unit"),
            "yoy_pct": item.get("yoy_pct"),
            "period": item.get("period"),
            "source_ref": _compact_ref(item.get("source_ref")),
        }
        if item.get("metric_scope") == "company_total":
            company_total[str(item.get("metric") or "")] = payload
        elif item.get("metric") == "revenue_total":
            segment_revenue.append({"business_area": item.get("business_area"), **payload})
    return {"company_total": company_total, "segment_revenue": segment_revenue}


def _key_products_services(
    evidence_pack: dict[str, Any],
    business_areas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    official_areas = [str(area.get("name") or "") for area in business_areas if area.get("name")]

    for area in business_areas:
        name = str(area.get("name") or "")
        texts = " ".join(str(item.get("text") or "") for item in area.get("evidence_texts", [])[:3])
        capabilities = list(area.get("core_capabilities") or [])
        services = _extract_service_terms(f"{area.get('summary') or ''} {texts}")
        if capabilities or services:
            out.append(
                {
                    "business_area": name,
                    "capabilities": _unique([*capabilities, *services])[:8],
                    "source_refs": area.get("source_refs", [])[:3],
                }
            )

    for item in evidence_pack.get("direction_evidence") or []:
        mapped_area = _map_to_official_area(
            str(item.get("business_area") or ""), item, official_areas
        )
        if not mapped_area or mapped_area == "company_total":
            continue
        services = _extract_service_terms(
            f"{item.get('claim') or ''} {item.get('evidence_text') or ''}"
        )
        if not services:
            continue
        _merge_product_service(out, mapped_area, services, item.get("source_ref"))
    return out[:8]


def _execution_cases(
    evidence_pack: dict[str, Any],
    official_areas: list[str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in evidence_pack.get("execution_evidence") or []:
        area = _map_to_official_area(str(item.get("business_area") or ""), item, official_areas)
        if not area or area == "company_total":
            continue
        cases.append(
            {
                "business_area": area,
                "summary": _profile_claim(item),
                "evidence_text": _compact_text(item.get("evidence_text"), limit=520),
                "source_ref": _compact_ref(item.get("source_ref")),
                "confidence": item.get("confidence"),
            }
        )
    return cases[:12]


def _strategic_focus(
    evidence_pack: dict[str, Any],
    official_areas: list[str],
) -> list[dict[str, Any]]:
    focus: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for bucket in ("direction_evidence", "evolution_evidence", "market_evidence"):
        for item in evidence_pack.get(bucket) or []:
            signal_type = str(item.get("signal_type") or "")
            if signal_type not in {"strategy", "growth", "business_update", "orders_pipeline"}:
                continue
            area = _map_to_official_area(str(item.get("business_area") or ""), item, official_areas)
            if not area or area == "company_total":
                continue
            summary = _profile_claim(item)
            key = (area, summary)
            if key in seen:
                continue
            seen.add(key)
            focus.append(
                {
                    "business_area": area,
                    "signal_type": signal_type,
                    "summary": summary,
                    "period": item.get("period"),
                    "source_ref": _compact_ref(item.get("source_ref")),
                    "confidence": item.get("confidence"),
                }
            )
    return focus[:12]


def _priority_initiatives(
    evidence_pack: dict[str, Any],
    official_areas: list[str],
) -> list[dict[str, Any]]:
    initiatives: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for bucket in (
        "direction_evidence",
        "evolution_evidence",
        "execution_evidence",
        "market_evidence",
    ):
        for item in evidence_pack.get(bucket) or []:
            text_value = f"{item.get('claim') or ''} {item.get('evidence_text') or ''}"
            if not _looks_like_initiative(text_value, item):
                continue
            area = _map_to_official_area(str(item.get("business_area") or ""), item, official_areas)
            if not area or area == "company_total":
                continue
            summary = _profile_claim(item)
            key = (area, summary)
            if key in seen:
                continue
            seen.add(key)
            initiatives.append(
                {
                    "business_area": area,
                    "summary": summary,
                    "evidence_text": _compact_text(item.get("evidence_text"), limit=520),
                    "period": item.get("period"),
                    "source_ref": _compact_ref(item.get("source_ref")),
                    "confidence": item.get("confidence"),
                }
            )
    return initiatives[:12]


def _investment_roadmap(
    evidence_pack: dict[str, Any],
    official_areas: list[str],
) -> list[dict[str, Any]]:
    roadmap: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for bucket in ("direction_evidence", "evolution_evidence", "market_evidence"):
        for item in evidence_pack.get(bucket) or []:
            text_value = f"{item.get('claim') or ''} {item.get('evidence_text') or ''}"
            if not _looks_like_roadmap(text_value, item):
                continue
            area = _map_to_official_area(str(item.get("business_area") or ""), item, official_areas)
            if not area or area == "company_total":
                continue
            summary = _profile_claim(item)
            period = str(item.get("period") or "")
            key = (period, area, summary)
            if key in seen:
                continue
            seen.add(key)
            roadmap.append(
                {
                    "period": item.get("period"),
                    "business_area": area,
                    "summary": summary,
                    "source": item.get("evidence_kind") or item.get("source_type"),
                    "source_ref": _compact_ref(item.get("source_ref")),
                    "confidence": item.get("confidence"),
                }
            )
    return roadmap[:12]


def _operational_highlights(
    evidence_pack: dict[str, Any],
    official_areas: list[str],
) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for item in evidence_pack.get("operational_evidence") or []:
        area = _map_to_official_area(str(item.get("business_area") or ""), item, official_areas)
        if not area or area == "company_total":
            continue
        highlights.append(
            {
                "period": item.get("period"),
                "business_area": area,
                "metric": item.get("metric"),
                "metric_label": item.get("metric_label"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "summary": item.get("claim"),
                "evidence_text": _compact_text(item.get("evidence_text"), limit=420),
                "source_ref": _compact_ref(item.get("source_ref")),
                "confidence": item.get("confidence"),
            }
        )
    return highlights[:16]


def _looks_like_initiative(text_value: str, item: dict[str, Any]) -> bool:
    signal_type = str(item.get("signal_type") or "")
    if signal_type in {"strategy", "growth", "business_update", "orders_pipeline", "forecast"}:
        return True
    return any(
        token in text_value
        for token in (
            "중점",
            "추진",
            "확대",
            "강화",
            "본격화",
            "로드맵",
            "투자",
            "수주",
            "진출",
            "전환",
        )
    )


def _looks_like_roadmap(text_value: str, item: dict[str, Any]) -> bool:
    signal_type = str(item.get("signal_type") or "")
    if signal_type in {"forecast", "strategy", "orders_pipeline"}:
        return True
    return any(
        token in text_value
        for token in (
            "로드맵",
            "투자",
            "향후",
            "전망",
            "본격화",
            "단계적",
            "확대",
            "진출",
            "수주",
            "CAPEX",
            "파트너십",
        )
    )


def _merge_product_service(
    out: list[dict[str, Any]],
    area: str,
    services: list[str],
    ref: Any,
) -> None:
    compact_ref = _compact_ref(ref)
    for item in out:
        if item.get("business_area") != area:
            continue
        item["capabilities"] = _unique([*item.get("capabilities", []), *services])[:8]
        if compact_ref:
            item["source_refs"] = _unique_refs([*item.get("source_refs", []), compact_ref])[:3]
        return
    out.append(
        {
            "business_area": area,
            "capabilities": services[:8],
            "source_refs": [compact_ref] if compact_ref else [],
        }
    )


def _extract_service_terms(text_value: str) -> list[str]:
    candidates = [
        "AgenticWorks",
        "Brity Copilot",
        "Brity Works",
        "FabriX",
        "Cello",
        "Cello Square",
        "Factova",
        "피지컬웍스",
        "AXgenticWire",
        "ChatGPT Enterprise",
        "ChatGPT Edu",
        "클라우드 MSP",
        "SI",
        "ITO",
        "ERP",
        "스마트팩토리",
        "스마트물류",
        "차량 SW 플랫폼",
        "내비게이션 SW",
        "P-GPT",
        "Intelligent Factory",
        "로봇 자동화",
        "생성형 AI",
    ]
    return [term for term in candidates if term in text_value]


def _market_view(evidence_pack: dict[str, Any]) -> dict[str, Any]:
    securities_report = (evidence_pack.get("source_coverage") or {}).get("securities_report") or {}
    if securities_report.get("status") == "unavailable":
        return {
            "status": "unavailable",
            "reason": securities_report.get("reason") or "최근 증권사 리포트 없음",
            "positive_points": [],
            "watch_points": [],
            "valuation_notes": [],
            "source_refs": [],
        }
    market_items = [
        item for item in evidence_pack.get("market_evidence") or [] if isinstance(item, dict)
    ]
    if market_items:
        positive_points: list[dict[str, Any]] = []
        watch_points: list[dict[str, Any]] = []
        valuation_notes: list[dict[str, Any]] = []
        for item in market_items:
            payload = {
                "summary": _profile_claim(item),
                "source_ref": _compact_ref(item.get("source_ref")),
                "confidence": item.get("confidence"),
            }
            if item.get("evidence_kind") == "securities_report_metric":
                valuation_notes.append(
                    {
                        "metric": item.get("metric"),
                        "metric_label": item.get("metric_label"),
                        "value": item.get("value"),
                        "unit": item.get("unit"),
                        "source_ref": _compact_ref(item.get("source_ref")),
                    }
                )
            elif item.get("sentiment") == "negative" or item.get("signal_type") == "risk":
                watch_points.append(payload)
            else:
                positive_points.append(payload)
        return {
            "status": "available",
            "reason": "증권사 리포트 기반 애널리스트 관점",
            "positive_points": positive_points[:4],
            "watch_points": watch_points[:3],
            "valuation_notes": valuation_notes[:3],
            "source_refs": _source_index({"market_evidence": market_items}),
        }
    if securities_report.get("available"):
        return {
            "status": "not_used",
            "reason": "MVP 프로필은 DART/IR/공식 뉴스룸 근거만 사용합니다.",
            "positive_points": [],
            "watch_points": [],
            "valuation_notes": [],
            "source_refs": [],
        }
    return {
        "status": "unavailable",
        "reason": "증권사 리포트 커버리지 없음",
        "positive_points": [],
        "watch_points": [],
        "valuation_notes": [],
        "source_refs": [],
    }


def _infer_business_areas_from_dart(text_value: str) -> list[tuple[str, str]]:
    explicit = _explicit_business_areas_from_dart(text_value)
    if explicit:
        return explicit
    candidates = [
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
            "SI/SM 기반 시스템 통합, 운영, 유지보수 및 AI 융합 디지털 서비스",
        ),
        (
            "차량 SW",
            "차량 소프트웨어, SDV, 내비게이션 등 모빌리티 소프트웨어 사업",
        ),
        (
            "스마트팩토리/자동화",
            "스마트팩토리, 산업 자동화, 로봇 및 제어 기반 산업 DX 사업",
        ),
    ]
    out: list[tuple[str, str]] = []
    for area, claim in candidates:
        if _area_name_seen(text_value, area):
            out.append((area, claim))
    return out


def _explicit_business_areas_from_dart(text_value: str) -> list[tuple[str, str]]:
    main_names = _main_business_area_names(text_value)
    if main_names:
        return [(name, _dart_claim_for_area(name, text_value)) for name in main_names[:6]]

    names: list[str] = []
    for match in re.finditer(r"\([0-9]+\)\s*", text_value):
        fragment = text_value[match.end() : match.end() + 90]
        names.append(_clean_area_name(_heading_prefix(fragment)))
    cleaned = _unique(name for name in names if _valid_area_name(name))
    return [(name, _dart_claim_for_area(name, text_value)) for name in cleaned[:6]]


def _main_business_area_names(text_value: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text_value or "")
    count_match = re.search(r"주된 사업은\s+(.{2,80}?)의\s*\d+개\s*사업부문", compact)
    if count_match:
        return _split_area_names(count_match.group(1))
    classification_match = re.search(
        r"주된 사업은\s+.{1,40}?(?:로|으로)\s+(.{2,140}?)(?:로|으로)\s*구분",
        compact,
    )
    if classification_match:
        return _split_area_names(classification_match.group(1))
    segment_match = re.search(
        r"사업부문은\s+(?:.{1,50}?에 따라\s+)?(.{2,160}?)(?:로|으로)\s*구분",
        compact,
    )
    if segment_match:
        return _split_area_names(segment_match.group(1))
    return []


def _split_area_names(value: str) -> list[str]:
    value = re.sub(r"\([^)]*\)", "", value or "")
    raw_names = re.split(r"\s*(?:,|와|과|및|/)\s*", value)
    return _unique(_clean_area_name(name) for name in raw_names if _valid_area_name(name))


def _heading_prefix(fragment: str) -> str:
    heading_match = re.match(r"(.{1,30}?(?:부문|사업))(?=\s|$)", fragment)
    if heading_match:
        return heading_match.group(1)
    stops = (" 당사", " 기업", " 물류사업", " 물류 사업", " [표]", " ①")
    end = len(fragment)
    for stop in stops:
        idx = fragment.find(stop)
        if idx >= 0:
            end = min(end, idx)
    return fragment[:end]


def _clean_area_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip(" .:;·-")
    cleaned = re.sub(r"\s*부문$", "", cleaned)
    cleaned = re.sub(r"\s*사업$", "", cleaned)
    cleaned = cleaned.replace("Digital BusinessService", "Digital Business Service")
    if cleaned.startswith("Digital Business Service"):
        return "Digital Business Service"
    return cleaned.strip()


def _valid_area_name(value: str) -> bool:
    if len(value) < 2 or len(value) > 40:
        return False
    noise = ("당사", "연결종속회사", "영위", "주된 사업", "구성")
    return not any(token in value for token in noise)


def _dart_claim_for_area(area: str, text_value: str) -> str:
    section = _section_around_area(text_value, area)
    if section:
        return _compact_text(section, limit=220)
    return f"DART 사업의 내용에서 확인된 공식 사업영역: {area}"


def _section_around_area(text_value: str, area: str) -> str:
    compact = re.sub(r"\s+", " ", text_value or "")
    heading = re.search(rf"\([0-9]+\)\s*{re.escape(area)}", compact)
    idx = heading.start() if heading else compact.find(area)
    if idx < 0:
        return ""
    return compact[idx : idx + 420]


def _area_name_seen(text_value: str, area: str) -> bool:
    aliases = {
        "클라우드&AI": ("클라우드&AI", "클라우드/AI", "Cloud", "AX"),
        "스마트 엔지니어링": (
            "스마트 엔지니어링",
            "스마트엔지니어링",
            "스마트팩토리",
            "스마트물류",
        ),
        "Digital Business Service": (
            "Digital Business Service",
            "Digital BusinessService",
            "SI/SM",
        ),
        "차량 SW": ("차량 소프트웨어", "차량SW", "차량 SW", "SDV", "내비게이션"),
        "스마트팩토리/자동화": ("스마트팩토리", "자동화", "로봇", "산업 DX"),
    }.get(area, (area,))
    return any(alias in text_value for alias in aliases)


def _recent_changes(
    evidence_pack: dict[str, Any],
    *,
    official_areas: list[str] | None = None,
) -> list[dict[str, Any]]:
    official_areas = official_areas or []
    direction_changes: list[dict[str, Any]] = []
    for item in evidence_pack.get("direction_evidence") or []:
        direction_changes.append(
            {
                "business_area": _map_to_official_area(
                    str(item.get("business_area") or ""),
                    item,
                    official_areas,
                ),
                "change_type": _change_type(item),
                "summary": _profile_claim(item),
                "evidence_text": _compact_text(item.get("evidence_text"), limit=700),
                "sentiment": item.get("sentiment"),
                "source_ref": _compact_ref(item.get("source_ref")),
                "confidence": item.get("confidence"),
            }
        )
    execution_changes: list[dict[str, Any]] = []
    for item in evidence_pack.get("execution_evidence") or []:
        execution_changes.append(
            {
                "business_area": _map_to_official_area(
                    str(item.get("business_area") or ""),
                    item,
                    official_areas,
                ),
                "change_type": "execution",
                "summary": _profile_claim(item),
                "evidence_text": _compact_text(item.get("evidence_text"), limit=700),
                "source_ref": _compact_ref(item.get("source_ref")),
                "confidence": item.get("confidence"),
            }
        )
    return [*direction_changes[:12], *execution_changes[:8]][:20]


def _capability_evolution(
    evidence_pack: dict[str, Any],
    recent_changes: list[dict[str, Any]],
) -> dict[str, Any]:
    evolution_items = [
        item for item in evidence_pack.get("evolution_evidence") or [] if isinstance(item, dict)
    ]
    if not evolution_items:
        return {
            "period": _period_range(recent_changes),
            "overall_change": _overall_change(recent_changes),
            "changes": recent_changes[:12],
            "watch_points": [],
        }

    changes: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in evolution_items:
        area = str(item.get("business_area") or "")
        if not area or area == "company_total":
            continue
        summary = _profile_claim(item)
        key = (str(item.get("period") or ""), area, summary)
        if key in seen:
            continue
        seen.add(key)
        changes.append(
            {
                "period": item.get("period"),
                "business_area": area,
                "change_type": _change_type(item),
                "summary": summary,
                "evidence_text": _compact_text(item.get("evidence_text"), limit=700),
                "source_ref": _compact_ref(item.get("source_ref")),
                "confidence": item.get("confidence"),
            }
        )

    balanced = _balanced_changes(changes, max_items=14)
    return {
        "period": _period_range(evolution_items),
        "overall_change": _evolution_overall_change(balanced),
        "changes": balanced,
        "watch_points": [],
    }


def _change_type(item: dict[str, Any]) -> str:
    if item.get("evidence_kind") == "official_execution":
        return "execution"
    signal_type = str(item.get("signal_type") or "")
    if signal_type in {"growth", "business_update", "orders_pipeline", "strategy"}:
        return "strengthening"
    return signal_type or "observed"


def _profile_claim(item: dict[str, Any]) -> str:
    raw = " ".join(str(item.get("claim") or "").split())
    area = str(item.get("business_area") or "")
    signal_type = str(item.get("signal_type") or "")

    if area == "클라우드&AI":
        if "AgenticWorks" in raw:
            return (
                "AgenticWorks 기반 AI 플랫폼 사업 확장과 글로벌 빅테크 파트너십 강화가 확인됩니다."
            )
        if "AX, RX, VX" in raw:
            return (
                "AX/RX/VX 오퍼링을 단계적으로 확대하고 "
                "스마트물류와 북미 사업 역량을 강화하고 있습니다."
            )
        if "RX Innovation" in raw or "PoC" in raw:
            return (
                "RX PoC 확대와 RX Innovation LAB 출범을 통해 "
                "고객 RX 전략 수립 지원을 강화하고 있습니다."
            )
        if "AI/Data" in raw or "MSP" in raw or "Cloud" in raw:
            return (
                "AI/Data 플랫폼, Agent 개발, Cloud 기반 AI 서비스와 "
                "MSP 사업 확대 신호가 확인됩니다."
            )
        if "AI 인프라" in raw or "데이터센터" in raw:
            return (
                "AI 인프라, 유지보수, 데이터센터 사업이 클라우드&AI 기반 역량을 보강하고 있습니다."
            )
        if "AI Native Software Engineering" in raw:
            return (
                "AI Native Software Engineering 역량을 바탕으로 "
                "국내외 사업 경쟁력을 강화하고 있습니다."
            )
    if area == "스마트 엔지니어링":
        if "Factova" in raw or "팩토바" in raw:
            return "Factova 기반 스마트팩토리 솔루션으로 제조 AX 실행 사례를 확대하고 있습니다."
        if "북미" in raw or "스마트물류" in raw or "RX" in raw:
            return (
                "스마트물류, RX, 북미 현지 사업 역량을 중심으로 "
                "스마트 엔지니어링 영역을 확장하고 있습니다."
            )
        if "방산" in raw or "조선" in raw or "반도체" in raw:
            return "제조 및 산업 현장 중심의 대외 프로젝트 확대 신호가 확인됩니다."
    if area == "Digital Business Service":
        if "금융" in raw or "ITO" in raw:
            return (
                "금융 고객 기반과 ITO 수주를 중심으로 Digital Business Service 성장이 확인됩니다."
            )
        return "SI/SM 기반 디지털 서비스 역량을 유지하며 AI 융합 프로젝트로 확장하고 있습니다."

    if signal_type in {"growth", "orders_pipeline"} and _amount_count(raw) >= 2:
        return _strip_amount_noise(raw)
    return raw


def _amount_count(value: str) -> int:
    return len(re.findall(r"\d[\d,]*\s*억원", value or ""))


def _strip_amount_noise(value: str) -> str:
    stripped = re.sub(r"\d[\d,]*\s*억원", "", value or "")
    stripped = re.sub(r"YoY\s*[+-]?\d+(?:\.\d+)?\s*%?", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or value


def _one_liner(company_name: str, business_areas: list[dict[str, Any]]) -> str:
    names = [str(area.get("name")) for area in business_areas if area.get("name")]
    if not names:
        return f"{company_name}의 피어사 프로필 스냅샷입니다."
    areas = ", ".join(names[:3])
    return f"{company_name}는 {areas}를 중심으로 사업을 전개하는 IT서비스 기업입니다."


def _company_summary(
    company_name: str,
    business_areas: list[dict[str, Any]],
    evidence_pack: dict[str, Any],
) -> str:
    area_names = [str(area.get("name")) for area in business_areas if area.get("name")]
    if not area_names:
        return f"{company_name}의 사업 개요를 요약할 수 있는 공식 근거가 제한적입니다."

    area_text = ", ".join(area_names[:4])
    dart_text = ""
    for item in evidence_pack.get("business_area_evidence") or []:
        dart_text = str(item.get("evidence_text") or "")
        if dart_text:
            break

    if "IT서비스" in area_text and "물류" in area_text:
        return (
            f"{company_name}는 공식 DART 기준 IT서비스와 물류를 주된 축으로 하는 기업입니다. "
            "IT서비스 영역에서는 클라우드, SI/ITO, AI/AX 관련 역량을 제공하고, "
            "물류 영역에서는 IT 기반 물류 플랫폼과 글로벌 물류 운영 역량을 결합합니다."
        )
    if "차량용 SW" in area_text:
        return (
            f"{company_name}는 IT 서비스와 차량용 SW를 중심으로 사업을 전개합니다. "
            "IT 컨설팅, SI/ITO, 클라우드 구축·운영 역량과 SDV 전환에 필요한 차량 SW 플랫폼 및 "
            "내비게이션 SW 역량을 함께 보유합니다."
        )
    if "자동화" in area_text and "IT서비스" in area_text:
        return (
            f"{company_name}는 자동화와 IT서비스를 중심으로 산업 현장의 디지털 전환을 지원합니다. "
            "공장 자동화, 제어·엔지니어링, 산업 AI와 엔터프라이즈 IT서비스를 결합해 "
            "제조·물류 현장 중심의 DX 역량을 제공합니다."
        )
    if "클라우드&AI" in area_text:
        return (
            f"{company_name}는 {area_text}를 중심으로 AX 전환과 IT서비스 사업을 수행합니다. "
            "클라우드 전환, AI 플랫폼, 스마트팩토리·물류 등 확인된 역량을 결합해 기업 고객의 "
            "디지털 전환을 지원합니다."
        )
    if "사업부문" in dart_text:
        return (
            f"{company_name}는 공식 DART에서 확인되는 {area_text} 사업영역을 "
            "중심으로 사업을 전개합니다."
        )
    return f"{company_name}는 {area_text}를 중심으로 사업을 전개합니다."


def _overall_change(recent_changes: list[dict[str, Any]]) -> str:
    areas = _unique(str(item.get("business_area") or "") for item in recent_changes)
    areas = [area for area in areas if area]
    if not areas:
        return "최근 변화 신호가 제한적입니다."
    return f"{', '.join(areas[:3])} 영역에서 강화 또는 실행 신호가 확인됩니다."


def _evolution_overall_change(changes: list[dict[str, Any]]) -> str:
    areas = _unique(str(item.get("business_area") or "") for item in changes)
    areas = [area for area in areas if area]
    if not areas:
        return "최근 여러 기간에 걸친 변화 신호가 제한적입니다."
    return f"최근 기간 동안 {', '.join(areas[:3])} 영역의 강화 및 실행 신호가 반복 확인됩니다."


def _period_range(items: list[dict[str, Any]]) -> dict[str, str | None]:
    periods = sorted(
        {str(item.get("period")) for item in items if item.get("period")},
        key=_period_sort_key,
    )
    if not periods:
        return {"from": None, "to": None}
    return {"from": periods[0], "to": periods[-1]}


def _period_sort_key(period: str) -> tuple[int, int, int]:
    quarter_match = re.fullmatch(r"(\d{4})Q([1-4])", period)
    if quarter_match:
        year = int(quarter_match.group(1))
        quarter = int(quarter_match.group(2))
        return (year, quarter * 3, 99)
    month_match = re.fullmatch(r"(\d{4})-(\d{2})", period)
    if month_match:
        return (int(month_match.group(1)), int(month_match.group(2)), 0)
    return (0, 0, 0)


def _balanced_changes(
    changes: list[dict[str, Any]],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_area_count: dict[str, int] = {}
    for change in changes:
        area = str(change.get("business_area") or "")
        if per_area_count.get(area, 0) >= 3:
            continue
        selected.append(change)
        per_area_count[area] = per_area_count.get(area, 0) + 1
        if len(selected) >= max_items:
            return selected
    for change in changes:
        if change in selected:
            continue
        selected.append(change)
        if len(selected) >= max_items:
            break
    return selected


def _cautions(evidence_pack: dict[str, Any]) -> list[str]:
    cautions: list[str] = []
    if not evidence_pack.get("business_area_evidence"):
        cautions.append("DART 기반 사업 영역 근거가 부족합니다.")
    return cautions


def _capabilities_for_area(area: str, text_value: Any) -> list[str]:
    text = str(text_value or "")
    if area == "클라우드&AI":
        caps = ["클라우드 전환", "MSP", "AI/Data 플랫폼"]
        if "AgenticWorks" in text:
            caps.append("AgenticWorks")
        return caps
    if area == "스마트 엔지니어링":
        caps = ["스마트팩토리", "스마트물류"]
        if "Factova" in text or "팩토바" in text:
            caps.append("Factova")
        if "RX" in text:
            caps.append("RX")
        return caps
    if area == "Digital Business Service":
        return ["SI/SM", "Enterprise IT"]
    return []


def _append_ref(refs: list[dict[str, Any]], ref: Any) -> None:
    compact = _compact_ref(ref)
    if not compact:
        return
    key = compact.get("raw_article_id")
    if any(item.get("raw_article_id") == key for item in refs):
        return
    refs.append(compact)


def _append_evidence_text(texts: list[dict[str, Any]], item: dict[str, Any]) -> None:
    text_value = _compact_text(item.get("evidence_text"), limit=900)
    if not text_value:
        return
    ref = _compact_ref(item.get("source_ref"))
    key = (ref.get("raw_article_id"), text_value)
    if any(
        (entry.get("source_ref", {}).get("raw_article_id"), entry.get("text")) == key
        for entry in texts
    ):
        return
    texts.append({"text": text_value, "source_ref": ref})


def _compact_ref(ref: Any) -> dict[str, Any]:
    if not isinstance(ref, dict):
        return {}
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


def _source_index(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for bucket in (
        "business_area_evidence",
        "financial_evidence",
        "direction_evidence",
        "execution_evidence",
        "evolution_evidence",
        "market_evidence",
        "source_index",
    ):
        for item in evidence_pack.get(bucket) or []:
            ref = item.get("source_ref", item) if isinstance(item, dict) else item
            compact = _compact_ref(ref)
            if not compact:
                continue
            key = compact.get("raw_article_id") or (compact.get("table"), compact.get("id"))
            if any(
                (existing.get("raw_article_id") or (existing.get("table"), existing.get("id")))
                == key
                for existing in refs
            ):
                continue
            refs.append(compact)
    return refs


def _compact_text(value: Any, *, limit: int) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _unique(values: Any) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if not value or value in out:
            continue
        out.append(value)
    return out


def _unique_refs(values: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for value in values:
        if not isinstance(value, dict) or not value:
            continue
        key = value.get("raw_article_id") or (value.get("table"), value.get("id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _parse_json(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    parsed = json.loads(stripped.strip())
    if not isinstance(parsed, dict):
        raise ValueError("ProfileSnapshotSummarizer output must be a JSON object")
    return parsed


__all__ = ["PROFILE_SNAPSHOT_SCHEMA_VERSION", "ProfileSnapshotSummarizer"]
