"""전략 의미 분석 컴포넌트.

IntegratedIssue와 분류/컨텍스트 메타데이터를 바탕으로 raw 데이터의 의미를
분석한다. 입력은 뉴스 클러스터뿐 아니라 DART, IR, 증권/산업 리포트, 글로벌
자료까지 포함할 수 있으므로, 본 컴포넌트는 특정 문서 유형 요약기가 아니라
evidence-first document intelligence analyzer 로 동작한다.

책임 경계:
- 사실 통합은 IssueIntegrationAgent 책임이다.
- 본 Agent는 통합된 사실을 출처 유형별 분석 프레임으로 재배열하고 의미를 해석한다.
- SK AX 대응 전략, 권고, 실행 과제는 별도 Implication/Response Agent 책임이다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.services.issue_integration.agent_views import analysis_agent_issue_input

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "analysis-v4.0"
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm

    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.12,
            max_completion_tokens=1500,
        )

    return _llm


_MULTI_SOURCE_CONTENT_ANALYSIS_PROMPT = """\
당신은 SK AX 경쟁/산업 데이터 분석을 위한 content analysis Agent입니다.

목적:
- raw DB에서 온 다양한 자료(뉴스, DART, IR, 증권/산업 리포트, 글로벌 자료)를
  "출처 유형별로 다른 분석 렌즈"로 읽고, 사실 기반 분석 결과를 구조화합니다.
- 단순 요약이 아니라 문서 안의 핵심 주장, 수치, 전략 신호, 리스크, 근거 공백을
  분리해 이후 요약 agent, 시사점 agent, SK AX 대응방향 agent가 재사용할 수 있게
  정리합니다.
- SK AX가 해야 할 일, 대응 전략, 권고, 실행 과제는 작성하지 않습니다.

## 분석 방법: Evidence-first Multi-source Document Intelligence
1. Source profile: source_type과 document family를 먼저 판정하고 분석 렌즈를 선택합니다.
   - news/company_news: 사건, 주체, 고객/제품/계약, 일정, 노출 강도
   - dart: 공시 항목, 재무/사업 세그먼트 변화, 확정/예정 구분, 리스크
   - ir: 경영진 메시지, 실적 동인, 가이던스, 사업 우선순위, 반복 강조점
   - securities_report: 애널리스트 주장, 밸류에이션/실적 전망, 근거와 가정
   - trend/global_report: 산업 구조 변화, 기술/수요 신호, 지역/플랫폼 맥락
2. Claim ledger: 입력 facts에 있는 주장과 수치를 claim 단위로 쪼개되, 없는 사실은 만들지 않습니다.
3. Materiality matrix: 전략, 재무/운영, 기술/제품, 시장/고객, 리스크/규제,
   타이밍 관점으로 중요도를 판단합니다.
4. Evidence map: 각 분석 문장은 fact_id, evidence_text, source_article_ids 중
   가능한 근거와 연결합니다.
5. Uncertainty: 전망/계획/가능성 표현과 근거가 약한 지점을 별도로 표시하고 confidence를 낮춥니다.

## 입력
integrated_issue:
{summary_json}

classification:
{classification_json}

cluster_metadata:
{cluster_metadata_json}

active_context:
{analysis_context_json}

## 작성 원칙
1. integrated_issue에 포함된 사실, 수치, business_signals, fact_basis에 기반한 해석만 작성하세요.
2. "무슨 일이 있었나"를 반복하지 말고 "문서/자료가 보여주는 의미 구조"를 설명하세요.
3. 피어사의 사업 방향, 제품/서비스 전략, 시장 접근 방식, 재무/운영 변화,
   리스크를 중심으로 분석하세요.
4. active_context가 제공되면 배경/연속성 판단에만 사용하고,
   입력 이슈에 없는 사실을 새로 만들지 마세요.
5. SK AX의 대응 필요성, 해야 할 일, 권고 문장은 작성하지 마세요.
6. 수치/고객명/제품명/회사명/일정은 입력 근거에 있는 경우에만 쓰세요.
7. 근거가 약하거나 자료 유형상 전망 성격이면 uncertainty_notes에 표시하고 confidence를 낮추세요.

## impact_level 기준
- high: 피어사의 사업 방향, 재무/운영 성과, 시장 경쟁 구도에 직접적인 의미가 있고 근거가 선명함
- medium: 특정 제품/서비스/사업 움직임으로 의미가 있으나 영향 범위나 근거가 제한적임
- low: 사실은 있으나 전략적 의미가 약하거나 근거가 제한적임

다음 JSON 형식으로만 응답하세요.
{{
  "is_valid_analysis": true,
  "analysis_scope": "peer_and_industry",
  "analysis_mode": "multi_source_document_intelligence",
  "source_profile": {{
    "source_family": "news|filing|ir|research|trend|mixed|unknown",
    "source_type": "입력 source_type",
    "analysis_lens": "선택한 분석 렌즈 1문장",
    "materiality_focus": ["전략", "재무/운영", "리스크"]
  }},
  "content_analysis": {{
    "core_thesis": "자료가 보여주는 핵심 주장/의미",
    "key_developments": ["핵심 전개 1", "핵심 전개 2"],
    "strategic_vectors": ["전략 방향 1", "전략 방향 2"],
    "financial_or_operating_readouts": ["수치/운영 변화 1"],
    "risk_factors": ["리스크 또는 제약 1"],
    "timing_and_commitment": "확정/예정/단기/중기 여부",
    "evidence_gaps": ["근거 공백 1"]
  }},
  "detailed_findings": [
    {{
      "finding_type": "strategy|financial|operation|market|technology|risk|governance",
      "finding": "세부 분석 내용",
      "evidence_refs": ["fact_id 또는 evidence id"],
      "source_article_ids": [1],
      "confidence": 0.0
    }}
  ],
  "evidence_map": [
    {{
      "claim": "분석 claim",
      "evidence": "근거 요약",
      "fact_ids": ["fact_id"],
      "source_article_ids": [1]
    }}
  ],
  "uncertainty_notes": ["전망/계획/근거 부족 지점"],
  "analysis_summary": "핵심 의미 1문장",
  "strategic_meaning": ["의미 1", "의미 2", "의미 3"],
  "market_signal": "산업 또는 시장 흐름 1문장",
  "impact_level": "high|medium|low",
  "impact_reason": "impact_level 판단 근거",
  "risk_or_opportunity": "risk|opportunity|neutral",
  "confidence": 0.0,
  "handoff": {{
    "summary_agent_inputs": ["요약 agent가 우선 반영할 사실"],
    "insight_agent_inputs": ["시사점 agent가 검토할 분석 포인트"],
    "skax_response_agent_inputs": ["대응방향 agent에 넘길 수 있는 관찰점. 권고문은 쓰지 말 것"]
  }},
  "reason": "분석 근거 또는 invalid 사유"
}}"""


class StrategicAnalyzer:
    """IntegratedIssue 기반 multi-source 산업/전략 의미 분석기."""

    def analyze(
        self,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any] | None = None,
        cluster_metadata: dict[str, Any] | None = None,
        analysis_context: Any | None = None,
    ) -> dict[str, Any]:
        """IntegratedIssue와 분류/context 정보를 바탕으로 의미 분석을 생성한다."""
        classification = classification or {}
        cluster_metadata = cluster_metadata or {}

        if not _is_valid_integrated_issue(integrated_issue):
            return _empty_analysis(
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata=cluster_metadata,
                reason="유효한 통합 이슈가 없어 분석을 생성하지 않았습니다.",
            )

        metadata_for_prompt = {
            **_cluster_metadata_for_prompt(cluster_metadata),
            "source_analysis_profile": _source_analysis_profile(
                source_type=str(cluster_metadata.get("source_type") or ""),
                source_types=cluster_metadata.get("source_types"),
            ),
        }
        prompt = (
            _MULTI_SOURCE_CONTENT_ANALYSIS_PROMPT.replace(
                "{summary_json}",
                json.dumps(
                    _integrated_issue_for_prompt(integrated_issue),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            .replace(
                "{classification_json}",
                json.dumps(
                    _classification_for_prompt(classification), ensure_ascii=False, indent=2
                ),
            )
            .replace(
                "{cluster_metadata_json}",
                json.dumps(metadata_for_prompt, ensure_ascii=False, indent=2),
            )
            .replace(
                "{analysis_context_json}",
                json.dumps(
                    _analysis_context_for_prompt(analysis_context),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        )

        try:
            from src.observability import tracing_config

            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="StrategicAnalyzer",
                    phase="analyze",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            result = _normalize_analysis_result(_parse_json(content))
        except Exception as exc:
            log.error(
                "자료 의미 분석 실패 | cluster=%s source_type=%s error=%s",
                _issue_metadata(integrated_issue).get("cluster_id"),
                cluster_metadata.get("source_type"),
                exc,
            )
            return _empty_analysis(
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata=cluster_metadata,
                reason=f"LLM 분석 실패: {type(exc).__name__}",
            )

        analysis = {
            "cluster_id": _issue_metadata(integrated_issue).get("cluster_id"),
            "representative_id": _issue_metadata(integrated_issue).get("representative_id"),
            "main_company": _issue_brief(integrated_issue).get("main_company", ""),
            "source_article_ids": _issue_source_ids(integrated_issue),
            "analysis_scope": "peer_and_industry",
            "analysis_mode": "multi_source_document_intelligence",
            "prompt_version": _PROMPT_VERSION,
            "model": _LLM_MODEL,
            "basis": {
                "issue_scope": _issue_metadata(integrated_issue).get("summary_scope", ""),
                "event_type": classification.get("event_type", ""),
                "sector": classification.get("sector", ""),
                "exposure_band": classification.get("exposure_band", ""),
                "bundle_id": cluster_metadata.get("bundle_id", ""),
                "source_type": cluster_metadata.get("source_type", ""),
                "source_types": cluster_metadata.get("source_types", []),
                "document_source_family": cluster_metadata.get("document_source_family", ""),
                "cluster_size": cluster_metadata.get("cluster_size")
                or classification.get("signals", {}).get("cluster_size", 0),
                "analysis_context_layers": _used_context_layers(analysis_context),
            },
            **result,
        }
        log.info(
            "자료 의미 분석 완료 | cluster=%s valid=%s source_type=%s company=%s impact=%s",
            analysis.get("cluster_id"),
            analysis.get("is_valid_analysis"),
            cluster_metadata.get("source_type"),
            analysis.get("main_company"),
            analysis.get("impact_level"),
        )
        return analysis


def _is_valid_integrated_issue(integrated_issue: dict[str, Any]) -> bool:
    brief = _issue_brief(integrated_issue)
    evidence = (
        integrated_issue.get("evidence")
        if isinstance(integrated_issue.get("evidence"), dict)
        else {}
    )
    has_subject = bool(
        brief.get("main_company")
        or brief.get("headline")
        or brief.get("scope_type") in {"industry", "market", "mixed"}
    )
    return bool(
        integrated_issue
        and brief.get("is_valid", integrated_issue.get("is_valid_summary", True))
        and has_subject
        and (
            brief.get("one_line_summary")
            or integrated_issue.get("analysis_ready_inputs")
            or evidence.get("by_section")
            or integrated_issue.get("content_digest")
        )
    )


def _integrated_issue_for_prompt(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    return analysis_agent_issue_input(integrated_issue)


def _classification_for_prompt(classification: dict[str, Any]) -> dict[str, Any]:
    return {
        "sector": classification.get("sector", ""),
        "sectors": classification.get("sectors", []),
        "event_type": classification.get("event_type", ""),
        "exposure_band": classification.get("exposure_band", ""),
        "exposure_score": classification.get("exposure_score", 0.0),
        "importance": classification.get("importance", ""),
        "importance_score": classification.get("importance_score", 0.0),
        "signals": classification.get("signals", {}),
    }


def _cluster_metadata_for_prompt(cluster_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": cluster_metadata.get("bundle_id", ""),
        "source_type": cluster_metadata.get("source_type", ""),
        "source_types": cluster_metadata.get("source_types", []),
        "content_types": cluster_metadata.get("content_types", []),
        "document_source_family": cluster_metadata.get("document_source_family", ""),
        "cluster_size": cluster_metadata.get("cluster_size", 0),
        "source_count": cluster_metadata.get("source_count", 0),
        "source_names": cluster_metadata.get("source_names", []),
        "published_at_range": cluster_metadata.get("published_at_range", {}),
        "companies": cluster_metadata.get("companies", []),
        "sectors": cluster_metadata.get("sectors", []),
        "event_type": cluster_metadata.get("event_type"),
        "has_structured_metrics": bool(cluster_metadata.get("has_structured_metrics")),
        "has_business_signals": bool(cluster_metadata.get("has_business_signals")),
        "parser_warning_count": cluster_metadata.get("parser_warning_count", 0),
        "trend_context": _trend_context_for_prompt(cluster_metadata.get("trend_context")),
    }


def _trend_context_for_prompt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "trend_summary": value.get("trend_summary", ""),
        "trend_lines": value.get("trend_lines", []),
        "signals": value.get("signals", []),
        "source_groups": value.get("source_groups", []),
        "period": value.get("period"),
    }


def _analysis_context_for_prompt(value: Any) -> dict[str, Any]:
    data = _to_dict(value)
    if not data:
        return {}
    financial = data.get("financial_trend") or {}
    compact_financial: dict[str, Any] = {}
    if isinstance(financial, dict):
        for key, series in list(financial.items())[:3]:
            if not isinstance(series, dict):
                continue
            compact_financial[str(key)] = {
                "company_id": series.get("company_id"),
                "metric_name_canonical": series.get("metric_name_canonical"),
                "points": list(series.get("points") or [])[:4],
            }
    return {
        "available_layers": _used_context_layers(value),
        "peer_event_timeline_recent": list(data.get("peer_event_timeline_recent") or [])[:5],
        "capability_evolution": data.get("capability_evolution") or {},
        "sector_pulse_recent": list(data.get("sector_pulse_recent") or [])[:4],
        "financial_trend": compact_financial,
        "event_chain_candidates": list(data.get("event_chain_candidates") or [])[:3],
        "similar_cards_rag": list(data.get("similar_cards_rag") or [])[:3],
        "evidence_density_per_peer": data.get("evidence_density_per_peer") or {},
        "provenance": data.get("provenance") or {},
    }


def _normalize_analysis_result(data: dict[str, Any]) -> dict[str, Any]:
    detailed_findings = _normalize_detailed_findings(data.get("detailed_findings"))[:6]
    strategic_meaning = _normalize_string_list(data.get("strategic_meaning"))[:3]
    if not strategic_meaning:
        strategic_meaning = [item["finding"] for item in detailed_findings if item.get("finding")][
            :3
        ]

    content_analysis = _normalize_content_analysis(data.get("content_analysis"))
    analysis_summary = str(data.get("analysis_summary") or "").strip()
    if not analysis_summary:
        analysis_summary = str(content_analysis.get("core_thesis") or "").strip()

    evidence_map = _normalize_evidence_map(data.get("evidence_map"))[:8]
    uncertainty_notes = _normalize_string_list(data.get("uncertainty_notes"))[:5]

    return {
        "is_valid_analysis": bool(data.get("is_valid_analysis", True))
        and bool(analysis_summary or strategic_meaning or detailed_findings),
        "analysis_scope": "peer_and_industry",
        "analysis_mode": "multi_source_document_intelligence",
        "source_profile": _normalize_source_profile(data.get("source_profile")),
        "content_analysis": content_analysis,
        "detailed_findings": detailed_findings,
        "evidence_map": evidence_map,
        "uncertainty_notes": uncertainty_notes,
        "analysis_summary": analysis_summary,
        "strategic_meaning": strategic_meaning,
        "market_signal": str(data.get("market_signal") or "").strip(),
        "impact_level": _normalize_choice(
            data.get("impact_level"),
            allowed={"high", "medium", "low"},
            default="medium",
        ),
        "impact_reason": str(data.get("impact_reason") or "").strip(),
        "risk_or_opportunity": _normalize_choice(
            data.get("risk_or_opportunity"),
            allowed={"risk", "opportunity", "neutral"},
            default="neutral",
        ),
        "confidence": _clamp_float(data.get("confidence"), default=0.0),
        "handoff": _normalize_handoff(data.get("handoff")),
        "reason": str(data.get("reason") or "").strip(),
    }


def _empty_analysis(
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    cluster_metadata: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "cluster_id": _issue_metadata(integrated_issue).get("cluster_id"),
        "representative_id": _issue_metadata(integrated_issue).get("representative_id"),
        "main_company": _issue_brief(integrated_issue).get("main_company", ""),
        "source_article_ids": _issue_source_ids(integrated_issue),
        "analysis_scope": "peer_and_industry",
        "analysis_mode": "multi_source_document_intelligence",
        "prompt_version": _PROMPT_VERSION,
        "model": _LLM_MODEL,
        "basis": {
            "issue_scope": _issue_metadata(integrated_issue).get("summary_scope", ""),
            "event_type": classification.get("event_type", ""),
            "sector": classification.get("sector", ""),
            "exposure_band": classification.get("exposure_band", ""),
            "source_type": cluster_metadata.get("source_type", ""),
            "cluster_size": cluster_metadata.get("cluster_size", 0),
        },
        "is_valid_analysis": False,
        "source_profile": _source_analysis_profile(
            source_type=str(cluster_metadata.get("source_type") or ""),
            source_types=cluster_metadata.get("source_types"),
        ),
        "content_analysis": {},
        "detailed_findings": [],
        "evidence_map": [],
        "uncertainty_notes": [],
        "analysis_summary": "",
        "strategic_meaning": [],
        "market_signal": "",
        "impact_level": "low",
        "impact_reason": "",
        "risk_or_opportunity": "neutral",
        "confidence": 0.0,
        "handoff": {
            "summary_agent_inputs": [],
            "insight_agent_inputs": [],
            "skax_response_agent_inputs": [],
        },
        "reason": reason,
    }


def _source_analysis_profile(
    *,
    source_type: str,
    source_types: Any = None,
) -> dict[str, Any]:
    normalized_types = _normalize_string_list(source_types) or _normalize_string_list(source_type)
    family = _source_family(normalized_types)
    profile_by_family = {
        "news": {
            "analysis_lens": "사건의 주체, 활동, 고객/제품/계약, 일정과 시장 노출을 구분한다.",
            "materiality_focus": ["event", "market", "strategy"],
        },
        "filing": {
            "analysis_lens": (
                "공시 항목, 수치, 사업 세그먼트 변화, 확정/예정 표현과 리스크를 분리한다."
            ),
            "materiality_focus": ["financial", "operation", "risk"],
        },
        "ir": {
            "analysis_lens": (
                "경영진 메시지, 실적 동인, 가이던스, 사업 우선순위와 반복 강조점을 본다."
            ),
            "materiality_focus": ["strategy", "financial", "operation"],
        },
        "research": {
            "analysis_lens": "애널리스트 주장, 전망치, 밸류에이션 근거와 가정을 구분한다.",
            "materiality_focus": ["market", "financial", "risk"],
        },
        "trend": {
            "analysis_lens": "산업 구조 변화, 기술/수요 신호, 지역/플랫폼 맥락을 분리한다.",
            "materiality_focus": ["market", "technology", "strategy"],
        },
        "mixed": {
            "analysis_lens": (
                "서로 다른 출처의 주장과 수치를 claim ledger로 나눠 중복과 차이를 비교한다."
            ),
            "materiality_focus": ["strategy", "market", "risk"],
        },
        "unknown": {
            "analysis_lens": "자료에 명시된 사실, 수치, 주장, 불확실성을 먼저 분리한다.",
            "materiality_focus": ["strategy", "market", "risk"],
        },
    }
    base = profile_by_family[family]
    return {
        "source_family": family,
        "source_type": normalized_types[0] if normalized_types else str(source_type or "unknown"),
        "source_types": normalized_types,
        **base,
    }


def _source_family(source_types: list[str]) -> str:
    values = {item.strip().lower() for item in source_types if item}
    if len(values) > 1:
        # 동일 family의 mixed source는 대표 family로 유지하고, 서로 다른 family면 mixed.
        families = {_source_family([value]) for value in values}
        return families.pop() if len(families) == 1 else "mixed"
    value = next(iter(values), "")
    if value in {"news", "news_cluster", "company_news", "global_newsroom", "official"}:
        return "news"
    if value in {"dart", "disclosure", "filing"}:
        return "filing"
    if value in {"ir", "earnings_presentation"}:
        return "ir"
    if value in {"securities_report", "analyst_report", "research_report"}:
        return "research"
    if value in {"trend_report", "industry_report", "global_report", "spri", "bcg"}:
        return "trend"
    return "unknown"


def _normalize_source_profile(value: Any) -> dict[str, Any]:
    data = value if isinstance(value, dict) else {}
    return {
        "source_family": _normalize_choice(
            data.get("source_family"),
            allowed={"news", "filing", "ir", "research", "trend", "mixed", "unknown"},
            default="unknown",
        ),
        "source_type": str(data.get("source_type") or "").strip(),
        "analysis_lens": str(data.get("analysis_lens") or "").strip(),
        "materiality_focus": _normalize_string_list(data.get("materiality_focus"))[:5],
    }


def _normalize_content_analysis(value: Any) -> dict[str, Any]:
    data = value if isinstance(value, dict) else {}
    return {
        "core_thesis": str(data.get("core_thesis") or "").strip(),
        "key_developments": _normalize_string_list(data.get("key_developments"))[:5],
        "strategic_vectors": _normalize_string_list(data.get("strategic_vectors"))[:5],
        "financial_or_operating_readouts": _normalize_string_list(
            data.get("financial_or_operating_readouts")
        )[:5],
        "risk_factors": _normalize_string_list(data.get("risk_factors"))[:5],
        "timing_and_commitment": str(data.get("timing_and_commitment") or "").strip(),
        "evidence_gaps": _normalize_string_list(data.get("evidence_gaps"))[:5],
    }


def _normalize_detailed_findings(value: Any) -> list[dict[str, Any]]:
    allowed_types = {
        "strategy",
        "financial",
        "operation",
        "market",
        "technology",
        "risk",
        "governance",
    }
    items = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        finding = str(item.get("finding") or "").strip()
        if not finding:
            continue
        out.append(
            {
                "finding_type": _normalize_choice(
                    item.get("finding_type"),
                    allowed=allowed_types,
                    default="strategy",
                ),
                "finding": finding,
                "evidence_refs": _normalize_string_list(item.get("evidence_refs"))[:5],
                "source_article_ids": _normalize_int_list(item.get("source_article_ids"))[:5],
                "confidence": _clamp_float(item.get("confidence"), default=0.0),
            }
        )
    return out


def _normalize_evidence_map(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        if not claim and not evidence:
            continue
        out.append(
            {
                "claim": claim,
                "evidence": evidence,
                "fact_ids": _normalize_string_list(item.get("fact_ids"))[:6],
                "source_article_ids": _normalize_int_list(item.get("source_article_ids"))[:6],
            }
        )
    return out


def _normalize_handoff(value: Any) -> dict[str, list[str]]:
    data = value if isinstance(value, dict) else {}
    return {
        "summary_agent_inputs": _normalize_string_list(data.get("summary_agent_inputs"))[:5],
        "insight_agent_inputs": _normalize_string_list(data.get("insight_agent_inputs"))[:5],
        "skax_response_agent_inputs": _normalize_string_list(
            data.get("skax_response_agent_inputs")
        )[:5],
    }


def _used_context_layers(value: Any) -> list[str]:
    data = _to_dict(value)
    if not data:
        return []
    provenance = data.get("provenance")
    if isinstance(provenance, dict):
        used_layers = _normalize_string_list(provenance.get("used_layers"))
        if used_layers:
            return used_layers
    layers: list[str] = []
    for key in (
        "peer_event_timeline_recent",
        "capability_evolution",
        "sector_pulse_recent",
        "financial_trend",
        "event_chain_candidates",
        "similar_cards_rag",
    ):
        if data.get(key):
            layers.append(key)
    return layers


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    return parsed if isinstance(parsed, dict) else {}


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_int_list(value: Any) -> list[int]:
    items = value if isinstance(value, list | tuple | set) else [value]
    out: list[int] = []
    for item in items:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number not in out:
            out.append(number)
    return out


def _issue_source_ids(integrated_issue: dict[str, Any]) -> list[int]:
    brief = _issue_brief(integrated_issue)
    scope = brief.get("analysis_scope") if isinstance(brief.get("analysis_scope"), dict) else {}
    ids = _normalize_int_list(
        scope.get("analyzed_source_ids") or scope.get("analyzed_raw_article_ids")
    )
    if ids:
        return ids
    ids = _normalize_int_list(integrated_issue.get("source_article_ids"))
    if ids:
        return ids
    ids = _normalize_int_list(integrated_issue.get("analyzed_article_ids"))
    if ids:
        return ids
    return _normalize_int_list(
        [
            source.get("id")
            for source in integrated_issue.get("sources", []) or []
            if isinstance(source, dict)
        ]
    )


def _issue_brief(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    value = integrated_issue.get("issue_brief")
    if isinstance(value, dict):
        return value
    return {
        "is_valid": integrated_issue.get("is_valid_summary", True),
        "headline": integrated_issue.get("headline") or integrated_issue.get("main_issue") or "",
        "one_line_summary": integrated_issue.get("one_line_summary")
        or integrated_issue.get("integrated_text")
        or integrated_issue.get("fact_summary")
        or "",
        "main_company": integrated_issue.get("main_company", ""),
        "mentioned_peer_companies": integrated_issue.get("mentioned_peer_companies", []),
        "event_type": integrated_issue.get("cluster_event_type")
        or integrated_issue.get("event_type"),
        "sectors": integrated_issue.get("sectors", []),
        "source_family": integrated_issue.get("source_family"),
        "scope_type": integrated_issue.get("scope_type"),
        "analysis_scope": {
            "analyzed_source_ids": _normalize_int_list(
                integrated_issue.get("source_article_ids")
                or integrated_issue.get("analyzed_article_ids")
            ),
        },
        "confidence": integrated_issue.get("confidence", 0.0),
        "reason": integrated_issue.get("reason", ""),
    }


def _issue_metadata(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    value = integrated_issue.get("metadata")
    if isinstance(value, dict):
        return value
    return {
        "cluster_id": integrated_issue.get("cluster_id"),
        "representative_id": integrated_issue.get("representative_id"),
        "bundle_id": integrated_issue.get("bundle_id"),
        "summary_scope": integrated_issue.get("summary_scope"),
    }


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else default


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)
