# 작성일: 2026-04-21
# 작성자: 최종민
# 변경이력:
#   2026-04-21 최종민 — 시사점 Agent 베이스라인, 분석 파이프라인·Evidence Chain·ChatOpenAI 지연 임포트·LLM 팩토리·JSON 헬퍼 정리
#   2026-05-28 박지원 — Profile Agent 수정 및 분석 러너 에이전트 리네임
"""SK AX 관점 시사점 Agent — v4.0 LLM 기반.

W1-1: deprecated 코드 부활 없이 새 입력 계약 + 출력 schema.
W4-5: AnalysisContext 지원 (signature 확장, prompt v5.0 자동 적용).

설계: design/01-analysis-pipeline-implementation-plan.md §3.3 / §4 W1-1, W4-5.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from src.analysis.implication import ImplicationGenerator
from src.analysis.models import (
    AnalysisContext,
    AnalysisInputBundle,
    AnalysisResult,
    ImplicationProvenance,
    ImplicationResult,
    PeerImplication,
    PrecedentLink,
    ProfileContext,
    SkaxImplication,
)
from src.analysis.prompts.implication_v4 import (
    PROMPT_VERSION as PROMPT_VERSION_V4,
)
from src.analysis.prompts.implication_v4 import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_V4,
)
from src.analysis.prompts.implication_v4 import (
    USER_PROMPT_TEMPLATE as USER_PROMPT_TEMPLATE_V4,
)
from src.analysis.prompts.implication_v5 import (
    PROMPT_VERSION as PROMPT_VERSION_V5,
)
from src.analysis.prompts.implication_v5 import (
    SYSTEM_PROMPT_V5,
    USER_PROMPT_TEMPLATE_V5,
)
from src.llm import LLMSpec, build_chat_llm
from src.shared.json_helpers import json_dumps as _json_dumps

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_LLM_TEMPERATURE = 0.2
_LLM_MAX_COMPLETION_TOKENS = 2200

_BUSINESS_LINES = ("에이전틱AI", "제조AX", "MSP")
_FOLLOW_UP_COUNT = 3
_WATCH_POINTS_MAX = 3
_OPPORTUNITY_MAX = 3
_THREAT_MAX = 3
_RECOMMENDED_ACTIONS_MAX = 3
_RELATION_ENUM = ("follow_up", "reaction", "echo", "contradiction")


class ImplicationAgent:
    """LLM-based SK AX 관점 시사점 Agent (v4.0 / v5.0).

    - generate(): cluster-time 호출. fail 시 ImplicationGenerator (heuristic) fallback.
    - LLM 호출 1회. temperature 0.2, max_completion_tokens 2,200.
    """

    prompt_version_v4 = PROMPT_VERSION_V4
    prompt_version_v5 = PROMPT_VERSION_V5

    def __init__(
        self,
        *,
        fallback: ImplicationGenerator | None = None,
        llm: ChatOpenAI | None = None,
    ) -> None:
        self._fallback = fallback or ImplicationGenerator()
        self._llm = llm
        self.model = _LLM_MODEL

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def generate(
        self,
        *,
        input_bundle: AnalysisInputBundle | dict[str, Any] | None = None,
        integrated_issue: dict[str, Any] | None = None,
        analysis: dict[str, Any] | AnalysisResult | None = None,
        profile_context: ProfileContext | dict[str, Any] | None = None,
        analysis_context: AnalysisContext | None = None,
        # Backward-compat: heuristic generator 호출 시그니처
        summary: dict[str, Any] | None = None,
        classification: dict[str, Any] | None = None,
        peer_profile_context: dict[str, Any] | None = None,
        skax_profile_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """LLM 기반 시사점 생성. dict 반환 (Backward-compat)."""
        # Normalize inputs --------------------------------------------------
        integrated_issue_dict = integrated_issue or summary or {}
        analysis_dict = _ensure_dict(analysis)
        profile_dict = _profile_to_dict(profile_context) or {
            "skax_profile": skax_profile_context or {},
            "peer_profiles": peer_profile_context or {},
            "sector_context": {},
        }
        bundle_dict = _bundle_to_dict(input_bundle)
        bundle_id = (bundle_dict or {}).get("bundle_id") or ""

        if not _has_minimum_inputs(integrated_issue_dict, analysis_dict):
            log.info(
                "ImplicationAgent skip → heuristic | bundle=%s reason=insufficient_inputs",
                bundle_id,
            )
            return self._heuristic_fallback(
                integrated_issue=integrated_issue_dict,
                analysis=analysis_dict,
                classification=classification,
                input_bundle=input_bundle,
                profile_context=profile_dict,
                peer_profile_context=peer_profile_context,
                skax_profile_context=skax_profile_context,
            )

        prompt_version = (
            PROMPT_VERSION_V5
            if analysis_context is not None and not analysis_context.is_empty()
            else PROMPT_VERSION_V4
        )
        system_prompt = (
            SYSTEM_PROMPT_V5 if prompt_version == PROMPT_VERSION_V5 else SYSTEM_PROMPT_V4
        )
        user_template = (
            USER_PROMPT_TEMPLATE_V5
            if prompt_version == PROMPT_VERSION_V5
            else USER_PROMPT_TEMPLATE_V4
        )

        user_prompt = _render_user_prompt(
            template=user_template,
            bundle=bundle_dict,
            integrated_issue=integrated_issue_dict,
            analysis=analysis_dict,
            profile=profile_dict,
            analysis_context=analysis_context,
        )

        try:
            llm_result = self._invoke_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                prompt_version=prompt_version,
                bundle_id=bundle_id,
            )
            result = _parse_and_normalize(
                llm_result,
                integrated_issue=integrated_issue_dict,
                profile_context=profile_dict,
                analysis_context=analysis_context,
                analysis=analysis_dict,
            )
            result.provenance.bundle_id = bundle_id
            result.provenance.prompt_version = prompt_version
            result.provenance.model = self.model
            result.provenance.run_at = datetime.now(UTC).isoformat()
            return result.to_dict()
        except Exception as exc:  # broad except: LLM 호출/파싱 어떤 에러도 fallback.
            log.warning(
                "ImplicationAgent LLM failure → heuristic fallback | bundle=%s error=%s",
                bundle_id,
                exc,
            )
            return self._heuristic_fallback(
                integrated_issue=integrated_issue_dict,
                analysis=analysis_dict,
                classification=classification,
                input_bundle=input_bundle,
                profile_context=profile_dict,
                peer_profile_context=peer_profile_context,
                skax_profile_context=skax_profile_context,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────
    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = build_chat_llm(
                LLMSpec(
                    model=_LLM_MODEL,
                    temperature=_LLM_TEMPERATURE,
                    max_tokens=_LLM_MAX_COMPLETION_TOKENS,
                )
            )
        return self._llm

    def _invoke_llm(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        bundle_id: str,
    ) -> str:
        try:
            from src.observability import tracing_config

            config = tracing_config(
                agent="ImplicationAgent",
                phase="generate",
                prompt_version=prompt_version,
                bundle_id=bundle_id,
            )
        except Exception:  # observability 가 미설치된 환경 (테스트 등) 도 동작.
            config = None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = (
            self._get_llm().invoke(messages, config=config)
            if config
            else self._get_llm().invoke(messages)
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        return content

    def _heuristic_fallback(
        self,
        *,
        integrated_issue: dict[str, Any],
        analysis: dict[str, Any],
        classification: dict[str, Any] | None,
        input_bundle: AnalysisInputBundle | dict[str, Any] | None,
        profile_context: dict[str, Any],
        peer_profile_context: dict[str, Any] | None,
        skax_profile_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self._fallback.generate(
            summary=integrated_issue,
            analysis=analysis,
            classification=classification,
            input_bundle=input_bundle,
            profile_context=profile_context,
            peer_profile_context=peer_profile_context,
            skax_profile_context=skax_profile_context,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _has_minimum_inputs(integrated_issue: dict[str, Any], analysis: dict[str, Any]) -> bool:
    if not integrated_issue.get("is_valid_summary", True):
        return False
    if not analysis.get("is_valid_analysis", True):
        return False
    return bool(
        integrated_issue.get("integrated_text")
        or integrated_issue.get("consolidated_facts")
        or integrated_issue.get("fact_summary")
    )


def _ensure_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, AnalysisResult):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _profile_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, ProfileContext):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _bundle_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, AnalysisInputBundle):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _render_user_prompt(
    *,
    template: str,
    bundle: dict[str, Any],
    integrated_issue: dict[str, Any],
    analysis: dict[str, Any],
    profile: dict[str, Any],
    analysis_context: AnalysisContext | None,
) -> str:
    bundle_for_prompt = _bundle_summary_for_prompt(bundle)
    integrated_for_prompt = _integrated_issue_for_prompt(integrated_issue)
    analysis_for_prompt = _analysis_for_prompt(analysis)
    profile_for_prompt = _profile_for_prompt(profile)
    context_payload: dict[str, Any] = (
        analysis_context.to_dict() if analysis_context is not None else {}
    )
    return template.format(
        bundle_json=_json_dumps(bundle_for_prompt),
        integrated_issue_json=_json_dumps(integrated_for_prompt),
        analysis_json=_json_dumps(analysis_for_prompt),
        profile_json=_json_dumps(profile_for_prompt),
        context_json=_json_dumps(context_payload),
    )


def _bundle_summary_for_prompt(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": bundle.get("bundle_id"),
        "cluster_id": bundle.get("cluster_id"),
        "source_type": bundle.get("source_type"),
        "companies": bundle.get("companies", []),
        "sectors": bundle.get("sectors", []),
        "event_type": bundle.get("event_type"),
        "source_count": len(bundle.get("sources", []) or []),
    }


def _integrated_issue_for_prompt(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "main_company": integrated_issue.get("main_company", ""),
        "main_issue": integrated_issue.get("main_issue", ""),
        "integrated_text": integrated_issue.get("integrated_text", ""),
        "consolidated_facts": (integrated_issue.get("consolidated_facts") or [])[:10],
        "key_numbers": integrated_issue.get("key_numbers", []),
        "business_signals": (integrated_issue.get("business_signals") or [])[:5],
        "fact_basis": (integrated_issue.get("fact_basis") or [])[:10],
        "missing_or_uncertain_points": integrated_issue.get("missing_or_uncertain_points", []),
        "confidence": integrated_issue.get("confidence", 0.0),
    }


def _analysis_for_prompt(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "is_valid_analysis": analysis.get("is_valid_analysis", False),
        "analysis_summary": analysis.get("analysis_summary", ""),
        "strategic_meaning": (analysis.get("strategic_meaning") or [])[:5],
        "market_signal": analysis.get("market_signal", ""),
        "impact_level": analysis.get("impact_level", "low"),
        "impact_reason": analysis.get("impact_reason", ""),
        "risk_or_opportunity": analysis.get("risk_or_opportunity", "neutral"),
        "confidence": analysis.get("confidence", 0.0),
    }


def _profile_for_prompt(profile: dict[str, Any]) -> dict[str, Any]:
    skax = profile.get("skax_profile") or {}
    peer_profiles = profile.get("peer_profiles") or {}
    sector_context = profile.get("sector_context") or {}
    return {
        "skax_profile": _shrink_profile(skax),
        "peer_profiles": {
            peer_id: _shrink_profile(payload)
            for peer_id, payload in (
                peer_profiles.items() if isinstance(peer_profiles, dict) else []
            )
        },
        "sector_context": sector_context,
    }


def _shrink_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    keys = (
        "company_name",
        "company_name_ko",
        "peer_id",
        "company_id",
        "schema_version",
        "one_liner",
        "company_summary",
        "business_lines",
        "business_areas",
        "core_capabilities",
        "recent_keywords",
        "recent_changes",
        "recent_signals",
        "recent_financial",
        "financial_summary",
        "market_view",
        "capability_evolution",
        "cautions",
        "narrative",
    )
    return {key: _compact_profile_value(profile[key]) for key in keys if key in profile}


def _compact_profile_value(value: Any) -> Any:
    """Keep profile prompt payload useful without passing the whole snapshot verbatim."""
    if isinstance(value, list):
        compacted = [_compact_profile_value(item) for item in value[:5]]
        return [item for item in compacted if item not in ({}, [], "", None)]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        allowed_keys = {
            "name",
            "summary",
            "core_capabilities",
            "recent_direction",
            "business_area",
            "claim",
            "signal_type",
            "sentiment",
            "metric",
            "metric_name",
            "metric_name_canonical",
            "value",
            "value_numeric",
            "value_krwbn",
            "unit",
            "period",
            "yoy_pct",
            "company_total",
            "segment_revenue",
            "status",
            "reason",
            "positive_points",
            "watch_points",
            "valuation_notes",
            "overall_change",
            "changes",
        }
        for key, nested in value.items():
            if key not in allowed_keys and len(out) >= 8:
                continue
            compacted = _compact_profile_value(nested)
            if compacted not in ({}, [], "", None):
                out[str(key)] = compacted
        return out
    if isinstance(value, str):
        return value[:700]
    return value


def _parse_and_normalize(
    raw: str,
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    analysis_context: AnalysisContext | None,
    analysis: dict[str, Any],
) -> ImplicationResult:
    data = _parse_json_loose(raw)
    if not isinstance(data, dict):
        data = {}

    peer_input = data.get("peer_implication") or {}
    skax_input = data.get("skax_implication") or {}

    peer = PeerImplication(
        company_id=str(
            peer_input.get("company_id") or _primary_company_id(integrated_issue, profile_context)
        ),
        company_name_ko=str(
            peer_input.get("company_name_ko") or _primary_company_name(profile_context)
        ),
        peer_meaning=str(peer_input.get("peer_meaning") or "").strip(),
        capability_change=_optional_str(peer_input.get("capability_change")),
        precedent_link=_parse_precedent_link(peer_input.get("precedent_link")),
        sourced_evidence_ids=_string_list(peer_input.get("sourced_evidence_ids"), max_items=10),
    )

    skax = SkaxImplication(
        why_important=str(skax_input.get("why_important") or "").strip(),
        potential_impact=str(skax_input.get("potential_impact") or "").strip(),
        opportunities=_string_list(skax_input.get("opportunities"), max_items=_OPPORTUNITY_MAX),
        threats=_string_list(skax_input.get("threats"), max_items=_THREAT_MAX),
        recommended_actions=_string_list(
            skax_input.get("recommended_actions"), max_items=_RECOMMENDED_ACTIONS_MAX
        ),
        business_line_mapping=_business_lines(skax_input.get("business_line_mapping")),
    )

    follow_up_raw = _string_list(data.get("follow_up_questions"), max_items=_FOLLOW_UP_COUNT)
    while len(follow_up_raw) < _FOLLOW_UP_COUNT:
        follow_up_raw.append("")
    follow_up = [item for item in follow_up_raw if item]
    watch_points = _string_list(data.get("watch_points"), max_items=_WATCH_POINTS_MAX)

    confidence = _clamp_float(data.get("confidence"), 0.0)
    evidence_label = _normalize_evidence_label(data.get("evidence_label"), confidence)

    is_valid = bool(
        data.get("is_valid_implication", True)
        and (peer.peer_meaning or skax.why_important)
        and (skax.recommended_actions or skax.opportunities or skax.threats)
    )

    used_layers = _detect_used_layers(analysis_context)
    provenance = ImplicationProvenance(
        generator="ImplicationAgent",
        used_peer_profile_keys=_keys_used(profile_context.get("peer_profiles") or {}),
        used_skax_profile_keys=_keys_used(profile_context.get("skax_profile") or {}),
        used_fact_ids=_string_list(peer.sourced_evidence_ids, max_items=10),
        used_context_layers=used_layers,
    )

    return ImplicationResult(
        is_valid_implication=is_valid,
        peer_implication=peer,
        skax_implication=skax,
        follow_up_questions=follow_up,
        watch_points=watch_points,
        confidence=confidence,
        evidence_label=evidence_label,  # type: ignore[arg-type]
        provenance=provenance,
    )


def _parse_json_loose(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.startswith("json"):
                body = body[4:]
            text = body.strip()
    # 흔히 LLM 이 본문 앞 prose 를 흘리는 경우 첫 '{' 부터 파싱.
    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            text = text[first : last + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _parse_precedent_link(value: Any) -> PrecedentLink | None:
    if not isinstance(value, dict):
        return None
    card_id = str(value.get("card_id") or "").strip()
    if not card_id:
        return None
    relation = str(value.get("relation") or "").strip().lower()
    if relation not in _RELATION_ENUM:
        return None
    try:
        days = int(value.get("days_since") or 0)
    except (TypeError, ValueError):
        days = 0
    return PrecedentLink(
        card_id=card_id,
        relation=relation,  # type: ignore[arg-type]
        days_since=max(days, 0),
        rationale=str(value.get("rationale") or "").strip(),
    )


def _primary_company_id(integrated_issue: dict[str, Any], profile: dict[str, Any]) -> str:
    company = str(integrated_issue.get("main_company") or "").strip()
    if company:
        return company
    peer_profiles = profile.get("peer_profiles") or {}
    if isinstance(peer_profiles, dict):
        for key in peer_profiles:
            if key:
                return str(key)
    return ""


def _primary_company_name(profile: dict[str, Any]) -> str:
    peer_profiles = profile.get("peer_profiles") or {}
    if isinstance(peer_profiles, dict):
        for payload in peer_profiles.values():
            if isinstance(payload, dict):
                name = payload.get("company_name") or payload.get("company_name_ko")
                if name:
                    return str(name)
    return ""


def _string_list(value: Any, *, max_items: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text][:max_items] if text else []
    if isinstance(value, list | tuple | set):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
            if len(out) >= max_items:
                break
        return out
    return []


def _business_lines(value: Any) -> list[str]:
    if value is None:
        return []
    raw = _string_list(value, max_items=len(_BUSINESS_LINES))
    return [line for line in raw if line in _BUSINESS_LINES]


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(min(max(number, 0.0), 1.0), 3)


def _normalize_evidence_label(value: Any, confidence: float) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"sufficient", "moderate", "insufficient"}:
        if raw == "sufficient" and confidence < 0.6:
            return "insufficient"
        return raw
    if confidence < 0.6:
        return "insufficient"
    if confidence < 0.8:
        return "moderate"
    return "sufficient"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _detect_used_layers(analysis_context: AnalysisContext | None) -> list[str]:
    if analysis_context is None:
        return []
    layers: list[str] = []
    if analysis_context.peer_event_timeline_recent:
        layers.append("peer_event_timeline_recent")
    if analysis_context.sector_pulse_recent:
        layers.append("sector_pulse_recent")
    if analysis_context.financial_trend:
        layers.append("financial_trend")
    if analysis_context.event_chain_candidates:
        layers.append("event_chain_candidates")
    if analysis_context.similar_cards_rag:
        layers.append("similar_cards_rag")
    return layers


def _keys_used(profile: Any) -> list[str]:
    if not isinstance(profile, dict):
        return []
    return list(profile.keys())[:20]


# Regex used by validate node (W2-3) — exposed via implication module as shared.
NUMERIC_TOKEN_PATTERN = re.compile(r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|usd|krw)?")
CERTAINTY_PATTERN = re.compile(r"(반드시|확실(?:히|하)|분명(?:히|하)|틀림없)")


__all__ = ["ImplicationAgent"]
