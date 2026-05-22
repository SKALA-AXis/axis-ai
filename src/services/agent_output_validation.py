"""Agent output validation helpers — design 의 prompt 제약 (length / enum / step
count) 을 LLM 응답에 강제하기 위한 공용 유틸.

design 문서들의 prompt 에 "≤ 100자" / "정확히 3~5 step" / "5종 enum 중 1개" 같은
제약이 명시되어 있지만 LLM 이 항상 준수하지는 않음. agent 의 `_parse_and_validate`
에서 본 헬퍼를 호출해서 graceful 보정 (reject 아닌 truncate / normalize) 한다.

audit 결과 (89% 정합) 의 medium-priority 갭 해결:
- final_one_liner 길이 강제 (≤ 100자, 또는 별도 한도)
- reasoning_trail step 수 (3~5)
- strategy_label 5종 enum normalize
- sources_used dedup + cap
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_ONE_LINER_MAX = 100
DEFAULT_IMPLICATION_MAX = 300
DEFAULT_TRAIL_MIN = 3
DEFAULT_TRAIL_MAX = 5
DEFAULT_STEPS_MAX = 10
DEFAULT_SOURCES_MAX = 20

STRATEGY_LABEL_ENUM: tuple[str, ...] = (
    "Aggressive Expansion",
    "Defensive Hold",
    "Tech Pivot",
    "Customer Lock-in",
    "Cost Leadership",
)

INTENT_ENUM: tuple[str, ...] = (
    "insight",
    "mixer",
    "it_trend",
    "link_verify",
    "search",
    "summary",
    "smalltalk",
)


# ──────────────────────────────────────────────────────────────────────────
# String constraints
# ──────────────────────────────────────────────────────────────────────────


def clip_string(value: Any, max_length: int, *, suffix: str = "…") -> str:
    """문자열을 max_length 로 잘라낸다. None / 비문자열은 빈 문자열."""
    if not isinstance(value, str):
        return ""
    if len(value) <= max_length:
        return value
    cut = max(max_length - len(suffix), 0)
    return value[:cut] + suffix


def clip_final_one_liner(value: Any, *, max_length: int = DEFAULT_ONE_LINER_MAX) -> str:
    """final_one_liner ≤ 100자 (design 전체 표준)."""
    return clip_string(value, max_length)


def clip_implication(value: Any, *, max_length: int = DEFAULT_IMPLICATION_MAX) -> str:
    """sk_ax_implication ≤ 300자 (1~2 문장 가정)."""
    return clip_string(value, max_length)


# ──────────────────────────────────────────────────────────────────────────
# Enum normalization
# ──────────────────────────────────────────────────────────────────────────


def normalize_strategy_label(value: Any) -> str:
    """5종 enum 중 가장 가까운 것을 반환, 매칭 실패 시 빈 문자열.

    LLM 이 `"aggressive expansion"` 처럼 lowercase 로 반환하는 경우 까지 흡수.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    cleaned = value.strip()
    cleaned_lower = cleaned.lower()
    for canonical in STRATEGY_LABEL_ENUM:
        if cleaned_lower == canonical.lower():
            return canonical
    # 부분 매칭 (예: "Aggressive Expansion 전략" → "Aggressive Expansion")
    for canonical in STRATEGY_LABEL_ENUM:
        if canonical.lower() in cleaned_lower:
            return canonical
    log.info("strategy_label '%s' 가 5종 enum 매칭 실패 — 공란 반환", cleaned[:60])
    return ""


def normalize_intent(value: Any) -> str:
    """8종 intent enum normalize. 매칭 실패 시 smalltalk fallback."""
    if not isinstance(value, str):
        return "smalltalk"
    cleaned = value.strip().lower()
    if cleaned in INTENT_ENUM:
        return cleaned
    return "smalltalk"


# ──────────────────────────────────────────────────────────────────────────
# List constraints
# ──────────────────────────────────────────────────────────────────────────


def cap_reasoning_trail(
    trail: Any,
    *,
    min_steps: int = DEFAULT_TRAIL_MIN,
    max_steps: int = DEFAULT_TRAIL_MAX,
) -> list[dict]:
    """reasoning_trail Tier 1 — 권장 3~5 step. 초과 시 truncate, 미달 시 그대로 (warning).

    seq 필드를 1-based 로 재부여하고, 각 step 의 label/one_liner 길이 제한.
    """
    if not isinstance(trail, list):
        return []
    cleaned: list[dict] = []
    for raw in trail[:max_steps]:
        if not isinstance(raw, dict):
            continue
        step = dict(raw)
        step["seq"] = len(cleaned) + 1
        step["label"] = clip_string(step.get("label"), 12)
        step["one_liner"] = clip_string(step.get("one_liner"), 80)
        evidence = step.get("evidence_refs")
        step["evidence_refs"] = [
            str(e) for e in (evidence if isinstance(evidence, list) else []) if e
        ][:10]
        cleaned.append(step)
    if len(cleaned) < min_steps:
        log.debug(
            "reasoning_trail step 수 %d < 권장 최소 %d — 그대로 반환",
            len(cleaned),
            min_steps,
        )
    return cleaned


def cap_reasoning_steps(steps: Any, *, max_steps: int = DEFAULT_STEPS_MAX) -> list[dict]:
    """reasoning_steps Tier 2 — 상한만 강제 (truncate)."""
    if not isinstance(steps, list):
        return []
    return [s for s in steps[:max_steps] if isinstance(s, dict)]


def dedup_and_cap(items: Any, *, max_items: int = DEFAULT_SOURCES_MAX) -> list[str]:
    """sources_used / peer_ids 등 중복 제거 + cap."""
    if not isinstance(items, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        key = item.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= max_items:
            break
    return out


def confidence_in_range(value: Any) -> float:
    """confidence 0~1 강제. 범위 밖이면 clamp."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return round(v, 3)


__all__ = [
    "cap_reasoning_steps",
    "cap_reasoning_trail",
    "clip_final_one_liner",
    "clip_implication",
    "clip_string",
    "confidence_in_range",
    "dedup_and_cap",
    "normalize_intent",
    "normalize_strategy_label",
]
