# 작성일: 2026-05-07
# 작성자: 박지원
# 변경이력:
#   2026-05-07 박지원 — 크롤러 출력용 회사 tier 라벨 헬퍼 추가
"""Company tier helpers for crawler output.

`companies.py` owns the existing domestic registry. This module only derives
the tier labels that downstream JSON consumers need.
"""

from __future__ import annotations

from typing import Final, Literal

from src.config.companies import COMPANY_ALIASES, COMPANY_IDS

CompanyTier = Literal["self", "domestic", "overseas"]

SELF_COMPANY_IDS: Final[set[str]] = {"sk_ax"}
DOMESTIC_COMPANY_IDS: Final[set[str]] = set(COMPANY_IDS) - SELF_COMPANY_IDS


def _normalize_alias(value: str) -> str:
    return value.lower().replace(" ", "")


_ALIAS_TO_COMPANY_ID: Final[dict[str, str]] = {
    _normalize_alias(alias): company_id
    for company_id, aliases in COMPANY_ALIASES.items()
    for alias in [company_id, *aliases]
}


def company_tier(company: str | None) -> CompanyTier:
    """Return the tier for a company id or known alias.

    Unknown ids are treated as overseas so future global company registries can
    flow through crawler JSON before they are added to the domestic config.
    """

    company_id = resolve_company_id(company)
    if company_id in SELF_COMPANY_IDS:
        return "self"
    if company_id in DOMESTIC_COMPANY_IDS:
        return "domestic"
    return "overseas"


def company_tier_map(companies: list[str]) -> dict[str, CompanyTier]:
    return {company: company_tier(company) for company in companies if company}


def resolve_company_id(company: str | None) -> str:
    value = (company or "").strip()
    if not value:
        return ""
    return _ALIAS_TO_COMPANY_ID.get(_normalize_alias(value), value)
