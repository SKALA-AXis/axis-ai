"""JSON 강제 변환·직렬화 공용 헬퍼 (중복 정의 단일 출처).

통합 대상은 **구현이 동일한** 것만 모은다:
- json_dict: mixer·briefing/support·strategic_insight/utils·analysis_units 의
  _json_dict 4곳이 동일(빈/공백 문자열도 모두 {} 반환). .strip() 유무는 빈 문자열
  단락 최적화일 뿐 결과 동일.
- json_dumps: implication·strategic_insight/profile_linkage 의 _json_dumps 2곳 동일.

통합하지 않은 것(구현이 갈라져 동작이 다름 — 통합 시 회귀):
- _json_list(3곳: non-list 파싱 결과 처리가 []/[parsed]/[value] 로 상이),
- _parse_json_loose(2곳: any-type vs dict-only),
- today_insight._json_dumps(_json_ready 전처리), it_trend._safe_json_object,
  summarizer._safe_json_loads(고유).
"""

from __future__ import annotations

import json
from typing import Any


def json_dict(value: object) -> dict[str, Any]:
    """value 를 dict 로 강제. dict 면 그대로, JSON 문자열이면 파싱, 그 외 {}."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def json_dumps(value: Any) -> str:
    """들여쓰기 2 + ensure_ascii=False 직렬화. 직렬화 불가 시 str() 폴백."""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)
