# 작성일: 2026-05-21
# 작성자: 최종민
# 변경이력:
#   2026-05-21 최종민 — Layer B 분석 파이프라인(컨텍스트 엔지니어링·LLM 추론·평가)의 일부로 추가
"""Peer ID alias 정규화 — P3-DATA-3 대응.

`raw_articles.matched_companies` 가 string array (객체 X) 이고 한글/영문/축약 ID 가
혼재하여 cluster-time query 에서 single ID 매칭만 하면 row 가 누락된다.
모든 query 가 `expand_peer_aliases()` 의 결과를 IN 절로 사용하면 일관성 확보.

기본 정책:
- 표준 canonical ID 는 `src.config.companies.COMPANIES` 의 key.
- alias 는 한글/영문/대소문자/공백 변형 모두 흡수.
- `normalize_to_canonical_id()` 가 임의 입력을 canonical ID 로 환원 (없으면 None).
"""

from __future__ import annotations

from typing import Final

from src.config.companies import COMPANIES, company_aliases

# (P3-DATA-3) raw_articles.matched_companies 에서 발견된 표기 변형을 모두 흡수.
# 신규 peer 추가 시 본 dict 도 함께 갱신해야 함 (PR review checklist).
PEER_ID_ALIASES: Final[dict[str, list[str]]] = {
    "samsung_sds": [
        "samsung_sds",
        "삼성SDS",
        "삼성 SDS",
        "삼성에스디에스",
        "samsung sds",
        "Samsung SDS",
        "SamsungSDS",
    ],
    "lg_cns": [
        "lg_cns",
        "LG CNS",
        "LGCNS",
        "LG cns",
        "엘지씨엔에스",
    ],
    "sk_ax": [
        "sk_ax",
        "SK AX",
        "SK ax",
        "에스케이에이엑스",
        "SK에이엑스",
        "SK C&C",
        "SK㈜ C&C",
        "SK주식회사 C&C",
        "에스케이씨앤씨",
        "SK Inc.",
        "SK주식회사",
    ],
    "posco_dx": [
        "posco_dx",
        "포스코DX",
        "포스코 DX",
        "POSCO DX",
        "포스코디엑스",
        "포스코ICT",
        "POSCO ICT",
    ],
    "hyundai_autoever": [
        "hyundai_autoever",
        "현대오토에버",
        "현대 오토에버",
        "Hyundai AutoEver",
        "Hyundai Autoever",
    ],
}


def _normalize_form(value: str) -> str:
    """비교용으로 alias 를 lowercase + 공백 제거."""
    return (value or "").lower().replace(" ", "").strip()


# canonical alias map: normalized alias → canonical_id
_ALIAS_INDEX: Final[dict[str, str]] = {}


def _bootstrap_alias_index() -> None:
    """모듈 로드 시 1회 실행되어 lookup 인덱스를 구축."""
    if _ALIAS_INDEX:
        return
    # 1) 본 모듈의 PEER_ID_ALIASES
    for canonical_id, aliases in PEER_ID_ALIASES.items():
        _ALIAS_INDEX[_normalize_form(canonical_id)] = canonical_id
        for alias in aliases:
            _ALIAS_INDEX[_normalize_form(alias)] = canonical_id
    # 2) companies.py 의 aliases (overlapping but covers extras)
    for company_id in COMPANIES:
        _ALIAS_INDEX[_normalize_form(company_id)] = company_id
        for alias in company_aliases(company_id):
            _ALIAS_INDEX.setdefault(_normalize_form(alias), company_id)


_bootstrap_alias_index()


def expand_peer_aliases(peer_id: str) -> list[str]:
    """canonical peer_id 가 cluster DB 에서 어떻게 표기될 수 있는지 모두 반환.

    SQL `WHERE peer_id = ANY(:aliases)` 또는 `matched_companies && :aliases::text[]`
    에 그대로 전달 가능. canonical 자체를 항상 포함.
    """
    aliases = PEER_ID_ALIASES.get(peer_id)
    if aliases:
        return list(dict.fromkeys(aliases))
    return [peer_id]


def normalize_to_canonical_id(any_id: str | None) -> str | None:
    """임의 입력 (한글/영문/축약/대소문자) 을 canonical peer_id 로 환원.

    매칭 실패 시 None. canonical 자체 입력은 그대로 반환.
    """
    if not any_id:
        return None
    return _ALIAS_INDEX.get(_normalize_form(any_id))


def is_known_peer(peer_id: str | None) -> bool:
    """canonical peer_id 가 PEER_ID_ALIASES 에 등록되어 있는가."""
    if not peer_id:
        return False
    return peer_id in PEER_ID_ALIASES


__all__ = [
    "PEER_ID_ALIASES",
    "expand_peer_aliases",
    "is_known_peer",
    "normalize_to_canonical_id",
]
