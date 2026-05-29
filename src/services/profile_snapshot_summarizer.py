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
        prompt = _build_prompt(evidence_pack)
        result = self._llm.invoke(prompt)
        content = getattr(result, "content", result)
        return _parse_json(str(content))


def _build_prompt(evidence_pack: dict[str, Any]) -> str:
    return f"""
당신은 PeerProfileAgent의 ProfileSnapshotSummarizer입니다.

입력 evidence_pack에 있는 근거만 사용해 회사별 profile_snapshot JSON을 작성하세요.
없는 제품, 수치, 고객, 전략은 만들지 마세요.
모든 주장에는 가능한 source_ref를 보존하세요.
source_ref는 URL/title 전체를 반복하지 말고 원문 조인 가능한 compact 형태만 사용하세요.
수치 계산은 하지 말고 입력의 financial_evidence 값만 사용하세요.
SK AX 대응방향이나 시사점은 작성하지 마세요.

출력은 JSON만 반환하세요. schema:
{{
  "company_id": "...",
  "company_name": "...",
  "schema_version": "{PROFILE_SNAPSHOT_SCHEMA_VERSION}",
  "generated_at": "ISO-8601",
  "one_liner": "",
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
  "financial_summary": {{}},
  "capability_evolution": {{
    "overall_change": "",
    "changes": [],
    "watch_points": []
  }},
  "cautions": [],
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
    recent_changes = _recent_changes(evidence_pack)
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
        "business_areas": business_areas,
        "core_capabilities": core_capabilities,
        "recent_changes": recent_changes,
        "financial_summary": financial_summary,
        "capability_evolution": {
            "overall_change": _overall_change(recent_changes),
            "changes": recent_changes[:5],
            "watch_points": [],
        },
        "cautions": _cautions(evidence_pack),
        "source_index": _source_index(evidence_pack),
    }


def _business_area_snapshots(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    by_area: dict[str, dict[str, Any]] = {}
    for item in evidence_pack.get("business_area_evidence") or []:
        area = str(item.get("business_area") or "")
        if not area:
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

    for item in evidence_pack.get("direction_evidence") or []:
        area = str(item.get("business_area") or "")
        if not area:
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
        area = str(item.get("business_area") or "")
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
        area = str(item.get("business_area") or "")
        if not area:
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
        entry["source_refs"] = entry.get("source_refs", [])[:4]
        entry["evidence_texts"] = entry.get("evidence_texts", [])[:3]
    return list(by_area.values())


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


def _recent_changes(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    direction_changes: list[dict[str, Any]] = []
    for item in evidence_pack.get("direction_evidence") or []:
        direction_changes.append(
            {
                "business_area": item.get("business_area"),
                "change_type": _change_type(item),
                "summary": _profile_claim(item),
                "evidence_text": _compact_text(item.get("evidence_text"), limit=420),
                "sentiment": item.get("sentiment"),
                "source_ref": _compact_ref(item.get("source_ref")),
                "confidence": item.get("confidence"),
            }
        )
    execution_changes: list[dict[str, Any]] = []
    for item in evidence_pack.get("execution_evidence") or []:
        execution_changes.append(
            {
                "business_area": item.get("business_area"),
                "change_type": "execution",
                "summary": _profile_claim(item),
                "evidence_text": _compact_text(item.get("evidence_text"), limit=420),
                "source_ref": _compact_ref(item.get("source_ref")),
                "confidence": item.get("confidence"),
            }
        )
    return [*direction_changes[:6], *execution_changes[:4]][:10]


def _change_type(item: dict[str, Any]) -> str:
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


def _overall_change(recent_changes: list[dict[str, Any]]) -> str:
    areas = _unique(str(item.get("business_area") or "") for item in recent_changes)
    areas = [area for area in areas if area]
    if not areas:
        return "최근 변화 신호가 제한적입니다."
    return f"{', '.join(areas[:3])} 영역에서 강화 또는 실행 신호가 확인됩니다."


def _cautions(evidence_pack: dict[str, Any]) -> list[str]:
    cautions: list[str] = []
    if not evidence_pack.get("financial_evidence"):
        cautions.append("최신 IR 재무 지표 근거가 부족합니다.")
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
    text_value = _compact_text(item.get("evidence_text"), limit=520)
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
