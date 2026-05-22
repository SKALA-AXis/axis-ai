"""EvaluatorAgent — W5-1 rule-based 4 metric (+regression_drift Phase 2).

`validate` 노드 안에서 호출. LLM 사용 X. latency 추가 <50ms 목표.
설계: design/01-analysis-pipeline-implementation-plan.md §3.6.1, §3.6.6.

Phase 1 (즉시): context_hit_ratio / evidence_claim_ratio / specificity_score /
                actionability_score
Phase 2 (운영 14일 후): regression_drift (rolling 7d avg confidence delta) +
                       calibrated threshold 활성화

Threshold 정책:
- 운영 첫 1주는 모든 카드 통과 (`human_review_flags` 추가 0건).
- 7일 후 weekly CronJob 이 bottom-10% percentile 을 산출하여 `_load_calibrated_thresholds()`
  의 동작을 활성화. percentile 산출 SQL 은 별도 script 가 책임.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from src.analysis.models import (
    AnalysisContext,
    EvaluationMetrics,
    ImplicationResult,
)

log = logging.getLogger(__name__)

EVALUATOR_VERSION = "rule-v1.0"

# 한국어 verb-suffix 패턴 (P5-LOG-1) — 동사 어휘 사전.
# "디지털 전환을 가속화한다" matches, "디지털 전환 가속화" 는 no-match.
_KO_VERB_SUFFIX_PATTERN = re.compile(
    r"("
    r"한다|하라|할\s*것|할것|"
    r"검토|착수|수립|구축|확보|강화|점검|모니터링|"
    r"도입|평가|분석|추진|개시|확장|확대|"
    r"개발|개선|검증|확인|시행|시작|"
    r"수행|진행|마련|준비|운영|관리"
    r")"
)

# 한국어 수치 / 날짜 / % 매칭. evidence_claim_ratio 계산용.
_NUMERIC_CLAIM_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|USD|KRW|usd|krw|건|명)?"
)
_DATE_CLAIM_PATTERN = re.compile(
    r"\d{4}년|\d{1,2}월|\d{1,2}일|\d{4}-\d{2}-\d{2}|\d{4}\.\d{2}|"
    r"\d{4}Q\d|\d{4}\s*년\s*\d{1,2}\s*월"
)

# Specificity 보조 신호: 회사명 / 시점 / 숫자.
_TIME_HINT_PATTERN = re.compile(
    r"\d{4}년|\d{1,2}월|\d{1,2}일|\d{4}Q\d|"
    r"올해|작년|내년|이번 분기|다음 분기|"
    r"분기|반기|상반기|하반기"
)
_NUMBER_HINT_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")

# SK AX 사업 키워드 — peer_name / 시점 / 숫자 외 보완 구체성 신호.
_BUSINESS_KEYWORD_HINTS = (
    "에이전틱AI",
    "에이전틱",
    "제조AX",
    "MSP",
    "PoC",
    "RFP",
    "RFI",
    "MOU",
)


class EvaluatorAgent:
    """Implication 출력에 대한 rule-based 품질 metric 계산."""

    version = EVALUATOR_VERSION

    def evaluate(
        self,
        *,
        implication: ImplicationResult | dict[str, Any] | None,
        evidence_payload: dict[str, Any] | None = None,
        analysis_context: AnalysisContext | None = None,
        integrated_issue: dict[str, Any] | None = None,
        rolling_confidence_avg: float | None = None,
        available_layer_count: int | None = None,
    ) -> EvaluationMetrics:
        impl = _ensure_impl_dict(implication)
        evidence_payload = evidence_payload or {}
        integrated_issue = integrated_issue or {}

        context_layers_count = (
            available_layer_count
            if available_layer_count is not None
            else (analysis_context.available_layer_count() if analysis_context is not None else 0)
        )

        context_hit = _context_hit_ratio(impl, context_layers_count)
        evidence_claim = _evidence_claim_ratio(impl, evidence_payload, integrated_issue)
        specificity = _specificity_score(impl, integrated_issue)
        actionability = _actionability_score(impl)

        drift: float | None = None
        if rolling_confidence_avg is not None and rolling_confidence_avg > 0:
            confidence = _safe_float(impl.get("confidence"), 0.0)
            drift = round((confidence - rolling_confidence_avg) / rolling_confidence_avg, 3)

        return EvaluationMetrics(
            context_hit_ratio=context_hit,
            evidence_claim_ratio=evidence_claim,
            specificity_score=specificity,
            actionability_score=actionability,
            regression_drift=drift,
            evaluator_version=self.version,
            evaluated_at=datetime.now(UTC).isoformat(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based metric implementations
# ─────────────────────────────────────────────────────────────────────────────


def _context_hit_ratio(impl: dict[str, Any], available_layer_count: int) -> float:
    """provenance.used_context_layers / available_layer_count.

    분모 6 고정 X (v3.2.1 정정). 신규 peer 에서 비어있는 layer 만큼 분모 축소.
    """
    used = (
        ((impl.get("provenance") or {}).get("used_context_layers"))
        if isinstance(impl.get("provenance"), dict)
        else None
    )
    used_count = len(used) if isinstance(used, list) else 0
    denominator = max(1, available_layer_count)
    return round(min(1.0, used_count / denominator), 3)


def _evidence_claim_ratio(
    impl: dict[str, Any],
    evidence_payload: dict[str, Any],
    integrated_issue: dict[str, Any],
) -> float:
    """implication 본문의 수치/날짜 claim 중 evidence 에 grounded 된 비율."""
    text = _collect_implication_text(impl)
    if not text:
        return 1.0  # claim 자체 없음 → 위반 없음.
    claims = set(_extract_claim_strings(text))
    if not claims:
        return 1.0

    grounded_corpus = _build_grounded_corpus(evidence_payload, integrated_issue)
    if not grounded_corpus:
        return 0.0

    grounded_text = " | ".join(grounded_corpus)
    grounded_count = sum(1 for claim in claims if claim and claim in grounded_text)
    return round(grounded_count / len(claims), 3)


def _specificity_score(
    impl: dict[str, Any],
    integrated_issue: dict[str, Any],
) -> float:
    """peer name unique occurrence + sector match."""
    skax = impl.get("skax_implication") or {}
    peer = impl.get("peer_implication") or {}
    peer_name = str(peer.get("company_name_ko") or peer.get("company_id") or "").strip()
    text_blocks = [
        str(skax.get("why_important") or ""),
        str(skax.get("potential_impact") or ""),
        " ".join(str(item) for item in (skax.get("opportunities") or [])),
        " ".join(str(item) for item in (skax.get("threats") or [])),
        " ".join(str(item) for item in (skax.get("recommended_actions") or [])),
        str(peer.get("peer_meaning") or ""),
        str(peer.get("capability_change") or ""),
    ]
    combined = " ".join(block for block in text_blocks if block)
    peer_match = 0.0
    if peer_name:
        occurrences = combined.count(peer_name)
        peer_match = min(1.0, occurrences / 1.0) if occurrences else 0.0
    sectors = integrated_issue.get("mentioned_sectors") or integrated_issue.get("sectors") or []
    if not isinstance(sectors, list):
        sectors = []
    sector_hits = sum(1 for sector in sectors if str(sector) and str(sector) in combined)
    sector_match = sector_hits / len(sectors) if sectors else 0.0
    score = 0.5 * peer_match + 0.5 * sector_match
    return round(min(1.0, score), 3)


def _actionability_score(impl: dict[str, Any]) -> float:
    """recommended_actions 의 verb-suffix + 구체성 (회사명/시점/숫자) 비율 곱.

    각 action 텍스트가 한국어 동사형 어미·어휘 (`-한다 / -할 것 / 검토 / 착수 / ...`) 로
    끝나는지 + 회사명/시점/숫자 중 1개 이상 포함하는지를 함께 본다.
    """
    skax = impl.get("skax_implication") or {}
    actions_raw = skax.get("recommended_actions") or []
    actions: list[str] = [str(item).strip() for item in actions_raw if str(item).strip()]
    if not actions:
        return 0.0
    peer = impl.get("peer_implication") or {}
    peer_name = str(peer.get("company_name_ko") or peer.get("company_id") or "").strip()

    verb_hits = sum(1 for action in actions if _KO_VERB_SUFFIX_PATTERN.search(action))
    concrete_hits = sum(
        1
        for action in actions
        if (peer_name and peer_name in action)
        or _TIME_HINT_PATTERN.search(action)
        or _NUMBER_HINT_PATTERN.search(action)
        or any(keyword in action for keyword in _BUSINESS_KEYWORD_HINTS)
    )
    verb_ratio = verb_hits / len(actions)
    concrete_ratio = concrete_hits / len(actions)
    return round(verb_ratio * concrete_ratio, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_impl_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, ImplicationResult):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _collect_implication_text(impl: dict[str, Any]) -> str:
    skax = impl.get("skax_implication") or {}
    peer = impl.get("peer_implication") or {}
    parts: list[str] = [
        str(skax.get("why_important") or ""),
        str(skax.get("potential_impact") or ""),
        " ".join(str(item) for item in (skax.get("opportunities") or [])),
        " ".join(str(item) for item in (skax.get("threats") or [])),
        " ".join(str(item) for item in (skax.get("recommended_actions") or [])),
        str(peer.get("peer_meaning") or ""),
        str(peer.get("capability_change") or ""),
        " ".join(str(item) for item in (impl.get("watch_points") or [])),
        " ".join(str(item) for item in (impl.get("follow_up_questions") or [])),
    ]
    return "\n".join(part for part in parts if part)


def _extract_claim_strings(text: str) -> list[str]:
    out: list[str] = []
    for match in _NUMERIC_CLAIM_PATTERN.finditer(text):
        token = match.group(0).strip()
        if token and token not in out:
            out.append(token)
    for match in _DATE_CLAIM_PATTERN.finditer(text):
        token = match.group(0).strip()
        if token and token not in out:
            out.append(token)
    return out


def _build_grounded_corpus(
    evidence_payload: dict[str, Any],
    integrated_issue: dict[str, Any],
) -> list[str]:
    blocks: list[str] = []
    # 1) evidence_payload 의 텍스트 가능 필드들.
    for key in ("financial_refs", "source_links", "mbb_refs"):
        value = evidence_payload.get(key)
        if value:
            blocks.append(_stringify(value))
    # 2) integrated_issue 의 fact_basis / key_numbers / consolidated_facts.
    for key in ("fact_basis", "key_numbers", "consolidated_facts"):
        value = integrated_issue.get(key)
        if value:
            blocks.append(_stringify(value))
    return blocks


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple | set):
        return " ".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_stringify(v) for v in value.values())
    return str(value)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["EVALUATOR_VERSION", "EvaluatorAgent"]
