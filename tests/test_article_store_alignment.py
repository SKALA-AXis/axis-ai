"""peer alignment 매칭용 SQL 절 빌더(_ilike_any_clause) 단위 테스트.

DB 연결 없이 SQL 문자열·바인드 파라미터 형태만 검증한다.
"""

from __future__ import annotations

from src.db.article_store import _ilike_any_clause


def test_ilike_any_clause_empty_terms_returns_no_clause() -> None:
    sql, params = _ilike_any_clause(["title ILIKE :{k}"], [], "kw")
    assert sql == ""
    assert params == {}


def test_ilike_any_clause_single_term_single_column() -> None:
    sql, params = _ilike_any_clause(["title ILIKE :{k}"], ["llm"], "kw")
    assert sql == "((title ILIKE :kw0))"
    assert params == {"kw0": "%llm%"}


def test_ilike_any_clause_ors_columns_within_term_and_across_terms() -> None:
    sql, params = _ilike_any_clause(
        ["title ILIKE :{k}", "content ILIKE :{k}"],
        ["llm", "대규모 언어 모델"],
        "kw",
    )
    # term 1개당 컬럼들은 OR, term 들 사이도 OR.
    assert sql == (
        "((title ILIKE :kw0 OR content ILIKE :kw0) OR (title ILIKE :kw1 OR content ILIKE :kw1))"
    )
    assert params == {"kw0": "%llm%", "kw1": "%대규모 언어 모델%"}


def test_ilike_any_clause_supports_keywords_array_exists_expr() -> None:
    sql, params = _ilike_any_clause(
        ["EXISTS (SELECT 1 FROM unnest(keywords) kw WHERE kw ILIKE :{k})"],
        ["gpu"],
        "kw",
    )
    # {k} placeholder 만 치환되고 테이블 alias kw 는 그대로 유지된다.
    assert sql == "((EXISTS (SELECT 1 FROM unnest(keywords) kw WHERE kw ILIKE :kw0)))"
    assert params == {"kw0": "%gpu%"}


def test_ilike_any_clause_prefix_isolates_param_names() -> None:
    _, params_a = _ilike_any_clause(["title ILIKE :{k}"], ["a"], "kw")
    _, params_b = _ilike_any_clause(["title ILIKE :{k}"], ["b"], "other")
    assert set(params_a) == {"kw0"}
    assert set(params_b) == {"other0"}
