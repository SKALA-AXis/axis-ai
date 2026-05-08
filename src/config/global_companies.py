"""글로벌 기업 RSS 모니터링 대상 설정."""

from __future__ import annotations

from typing import Final, TypedDict


class GlobalCompanyConfig(TypedDict):
    name_ko: str
    aliases: list[str]


GLOBAL_COMPANIES: Final[dict[str, GlobalCompanyConfig]] = {
    "nvidia": {
        "name_ko": "엔비디아",
        "aliases": [
            "NVIDIA",
            "엔비디아",
        ],
    },
    "apple": {
        "name_ko": "애플",
        "aliases": [
            "Apple",
            "애플",
        ],
    },
    "microsoft": {
        "name_ko": "마이크로소프트",
        "aliases": [
            "Microsoft",
            "마이크로소프트",
        ],
    },
    "google": {
        "name_ko": "구글",
        "aliases": [
            "Google",
            "구글",
        ],
    },
    "amazon": {
        "name_ko": "아마존",
        "aliases": [
            "Amazon",
            "아마존",
        ],
    },
    "meta": {
        "name_ko": "메타",
        "aliases": [
            "Meta",
            "메타",
        ],
    },
}

GLOBAL_COMPANY_IDS: Final[list[str]] = list(GLOBAL_COMPANIES.keys())
GLOBAL_COMPANY_ALIASES: Final[dict[str, list[str]]] = {
    company_id: config["aliases"]
    for company_id, config in GLOBAL_COMPANIES.items()
}


def global_company_aliases(company_id: str) -> list[str]:
    if not company_id:
        return []

    return GLOBAL_COMPANY_ALIASES.get(company_id, [company_id])


def global_company_name_ko(company_id: str) -> str:
    config = GLOBAL_COMPANIES.get(company_id)
    return config["name_ko"] if config else company_id