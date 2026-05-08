"""피어사 뉴스 의미 분석 에이전트.

PeerNewsSummaryAgent가 만든 사실 요약을 바탕으로 피어사의 전략적 움직임과
산업적 의미를 분석한다. SK AX 관점의 대응 제언은 별도 Agent 책임으로 남긴다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm

    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.15,
            max_completion_tokens=900,
        )

    return _llm


_PEER_NEWS_ANALYSIS_PROMPT = """\
당신은 피어사 뉴스 분석 Agent입니다.

목적:
- 피어사 뉴스 사실 요약을 바탕으로 피어사의 전략적 움직임과 산업적 의미를 분석합니다.
- SK AX 관점의 대응 전략, 권고, 실행 과제는 작성하지 않습니다.
- 기사 요약에 없는 구체 수치, 제품명, 회사명, 고객명은 추가하지 않습니다.

## 입력
summary:
{summary_json}

classification:
{classification_json}

cluster_metadata:
{cluster_metadata_json}

## 작성 원칙
1. fact_summary에 근거한 해석만 작성하세요.
2. "무슨 일이 있었나"를 반복하지 말고 "왜 의미가 있는가"를 설명하세요.
3. 피어사의 사업 방향, 제품/서비스 전략, 시장 접근 방식을 중심으로 분석하세요.
4. 산업 동향은 입력 사실에서 자연스럽게 도출되는 범위 안에서만 작성하세요.
5. SK AX의 대응 필요성, 해야 할 일, 권고 문장은 작성하지 마세요.
6. 과장 표현을 피하고, 근거가 약하면 confidence를 낮추세요.

## impact_level 기준
- high: 피어사의 사업 방향이나 시장 경쟁 구도에 직접적인 의미가 있고 보도 노출도도 높음
- medium: 특정 제품/서비스/사업 움직임으로 의미가 있으나 영향 범위가 제한적임
- low: 사실은 있으나 전략적 의미가 약하거나 근거가 제한적임

다음 JSON 형식으로만 응답하세요.
{{
  "is_valid_analysis": true,
  "analysis_scope": "peer_and_industry",
  "analysis_summary": "핵심 의미 1문장",
  "strategic_meaning": ["의미 1", "의미 2", "의미 3"],
  "market_signal": "산업 또는 시장 흐름 1문장",
  "impact_level": "high|medium|low",
  "impact_reason": "impact_level 판단 근거",
  "risk_or_opportunity": "risk|opportunity|neutral",
  "confidence": 0.0,
  "reason": "분석 근거 또는 invalid 사유"
}}"""


class PeerNewsAnalysisAgent:
    """피어사 뉴스 요약 결과를 산업/전략 의미로 분석한다."""

    def analyze(
        self,
        summary: dict[str, Any],
        classification: dict[str, Any] | None = None,
        cluster_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """요약 결과와 분류 정보를 바탕으로 의미 분석을 생성한다."""
        classification = classification or {}
        cluster_metadata = cluster_metadata or {}

        if not _is_valid_summary(summary):
            return _empty_analysis(
                summary=summary,
                classification=classification,
                cluster_metadata=cluster_metadata,
                reason="유효한 뉴스 요약이 없어 분석을 생성하지 않았습니다.",
            )

        prompt = (
            _PEER_NEWS_ANALYSIS_PROMPT.replace(
                "{summary_json}",
                json.dumps(_summary_for_prompt(summary), ensure_ascii=False, indent=2),
            )
            .replace(
                "{classification_json}",
                json.dumps(
                    _classification_for_prompt(classification), ensure_ascii=False, indent=2
                ),
            )
            .replace(
                "{cluster_metadata_json}",
                json.dumps(
                    _cluster_metadata_for_prompt(cluster_metadata), ensure_ascii=False, indent=2
                ),
            )
        )

        try:
            response = _get_llm().invoke(prompt)
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            result = _normalize_analysis_result(_parse_json(content))
        except Exception as exc:
            log.error(
                "피어사 뉴스 분석 실패 | cluster=%s error=%s",
                summary.get("cluster_id"),
                exc,
            )
            return _empty_analysis(
                summary=summary,
                classification=classification,
                cluster_metadata=cluster_metadata,
                reason=f"LLM 분석 실패: {type(exc).__name__}",
            )

        analysis = {
            "cluster_id": summary.get("cluster_id"),
            "representative_id": summary.get("representative_id"),
            "main_company": summary.get("main_company", ""),
            "source_article_ids": summary.get("source_article_ids", []),
            "analysis_scope": "peer_and_industry",
            "model": _LLM_MODEL,
            "basis": {
                "summary_scope": summary.get("summary_scope", ""),
                "event_type": classification.get("event_type", ""),
                "sector": classification.get("sector", ""),
                "exposure_band": classification.get("exposure_band", ""),
                "cluster_size": cluster_metadata.get("cluster_size")
                or classification.get("signals", {}).get("cluster_size", 0),
            },
            **result,
        }
        log.info(
            "피어사 뉴스 분석 완료 | cluster=%s valid=%s company=%s impact=%s",
            analysis.get("cluster_id"),
            analysis.get("is_valid_analysis"),
            analysis.get("main_company"),
            analysis.get("impact_level"),
        )
        return analysis


def _is_valid_summary(summary: dict[str, Any]) -> bool:
    return bool(
        summary
        and summary.get("is_valid_summary", True)
        and summary.get("main_company")
        and summary.get("fact_summary")
    )


def _summary_for_prompt(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": summary.get("cluster_id"),
        "main_company": summary.get("main_company", ""),
        "mentioned_peer_companies": summary.get("mentioned_peer_companies", []),
        "headline": summary.get("headline", ""),
        "one_line_summary": summary.get("one_line_summary", ""),
        "fact_summary": summary.get("fact_summary", []),
        "main_event": summary.get("main_event", ""),
        "confidence": summary.get("confidence", 0.0),
    }


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
        "cluster_size": cluster_metadata.get("cluster_size", 0),
        "source_count": cluster_metadata.get("source_count", 0),
        "source_names": cluster_metadata.get("source_names", []),
        "published_at_range": cluster_metadata.get("published_at_range", {}),
    }


def _normalize_analysis_result(data: dict[str, Any]) -> dict[str, Any]:
    strategic_meaning = _normalize_string_list(data.get("strategic_meaning"))[:3]
    return {
        "is_valid_analysis": bool(data.get("is_valid_analysis", True)) and bool(strategic_meaning),
        "analysis_summary": str(data.get("analysis_summary") or "").strip(),
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
        "reason": str(data.get("reason") or "").strip(),
    }


def _empty_analysis(
    summary: dict[str, Any],
    classification: dict[str, Any],
    cluster_metadata: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "cluster_id": summary.get("cluster_id"),
        "representative_id": summary.get("representative_id"),
        "main_company": summary.get("main_company", ""),
        "source_article_ids": summary.get("source_article_ids", []),
        "analysis_scope": "peer_and_industry",
        "model": _LLM_MODEL,
        "basis": {
            "summary_scope": summary.get("summary_scope", ""),
            "event_type": classification.get("event_type", ""),
            "sector": classification.get("sector", ""),
            "exposure_band": classification.get("exposure_band", ""),
            "cluster_size": cluster_metadata.get("cluster_size", 0),
        },
        "is_valid_analysis": False,
        "analysis_summary": "",
        "strategic_meaning": [],
        "market_signal": "",
        "impact_level": "low",
        "impact_reason": "",
        "risk_or_opportunity": "neutral",
        "confidence": 0.0,
        "reason": reason,
    }


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
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else default


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)
