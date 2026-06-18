# 작성일: 2026-06-12
# 작성자: 최종민
# 변경이력:
#   2026-06-12 최종민 — prompts/utils 를 별도 패키지로 분리한 뒤 JSON 헬퍼(_json_dict·_json_dumps) 단일 출처화
"""utils — strategic_insight_agent 에서 분리 (이동만, 동작 불변).

분리 근거: axis-infra docs/structure-tasks/agent-split-design.md (Phase 2 1단계)
"""

import json
from typing import Any

from src.shared.json_helpers import json_dict as _json_dict  # noqa: F401  — 재export 유지(R2)


def _jsonish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def _int_list(value: Any) -> list[int]:
    result: list[int] = []
    for item in _jsonish_list(value):
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in result:
            result.append(parsed)
    return result


def _with_particle(text: str, consonant_particle: str, vowel_particle: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    return f"{value}{consonant_particle if _has_final_consonant(value[-1]) else vowel_particle}"


def _has_final_consonant(char: str) -> bool:
    value = str(char or "")[:1]
    if not value:
        return False
    code = ord(value)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


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
    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            text = text[first : last + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _string_list(value: Any, *, max_items: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
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


def _choice(value: Any, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else default
