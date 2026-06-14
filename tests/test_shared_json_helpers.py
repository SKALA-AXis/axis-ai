"""공용 JSON 헬퍼 동작 잠금 (refactoring-architecture R2).

통합된 json_dict/json_dumps 가 기존 _json_dict/_json_dumps 와 동일 동작인지 검증.
"""

from __future__ import annotations

from src.shared.json_helpers import json_dict, json_dumps


# ---------------------------------------------------------------- json_dict


def test_json_dict_passthrough_dict():
    d = {"a": 1}
    assert json_dict(d) is d


def test_json_dict_parses_json_object_string():
    assert json_dict('{"a": 1}') == {"a": 1}


def test_json_dict_non_object_json_returns_empty():
    # JSON 이 list/scalar 면 {} (기존 4개 정의 공통)
    assert json_dict("[1, 2]") == {}
    assert json_dict('"text"') == {}


def test_json_dict_invalid_json_returns_empty():
    assert json_dict("not json") == {}


def test_json_dict_empty_or_whitespace_returns_empty():
    assert json_dict("") == {}
    assert json_dict("   ") == {}


def test_json_dict_other_types_return_empty():
    assert json_dict(None) == {}
    assert json_dict(123) == {}
    assert json_dict([1, 2]) == {}


# --------------------------------------------------------------- json_dumps


def test_json_dumps_pretty_unicode():
    out = json_dumps({"한글": "값"})
    assert "한글" in out  # ensure_ascii=False
    assert "\n" in out  # indent=2


def test_json_dumps_falls_back_to_str_on_unserializable():
    class Weird:
        def __repr__(self):
            return "WEIRD"

    # default=str 로 직렬화되므로 예외 없이 문자열 반환
    out = json_dumps({"x": Weird()})
    assert "WEIRD" in out
