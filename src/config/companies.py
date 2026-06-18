# 작성일: 2026-05-06
# 작성자: 박지원
# 변경이력:
#   2026-05-06 박지원 — 회사 설정 registry 구현, 크롤러 전처리·catch 크롤러 대상 추가
#   2026-05-06 박진 — stock_crawler 관련 회사 설정 추가
#   2026-05-07 심유정 — 크롤러 publisher 출력 개선
"""모니터링 대상 회사 설정 registry.

코드 전역에서 쓰는 회사 alias, DART corp code, 네이버 item code, 채용/특허/IR 설정을
한 곳에서 관리한다. 내부 식별자는 company id를 사용한다.
"""

from __future__ import annotations

from typing import Final, TypedDict


class CompanyConfig(TypedDict, total=False):
    name_ko: str
    aliases: list[str]
    dart_corp_code: str
    naver_item_code: str
    naver_item_name_ko: str
    saramin_name: str
    kipris_applicant: str
    ir_pages: list[str]
    ir_fetch_strategy: str
    ir_click_fallback: bool
    catch_analysis_id: str


COMPANIES: Final[dict[str, CompanyConfig]] = {
    "samsung_sds": {
        "name_ko": "삼성SDS",
        "aliases": ["삼성SDS", "삼성 SDS", "Samsung SDS", "삼성에스디에스"],
        "dart_corp_code": "00126186",
        "naver_item_code": "018260",
        "naver_item_name_ko": "삼성SDS",
        "saramin_name": "삼성SDS",
        "kipris_applicant": "삼성에스디에스",
        "ir_pages": [
            "https://www.samsungsds.com/kr/investor/ir_events/earnings-release.html",
        ],
        "ir_fetch_strategy": "playwright_then_httpx",
        "ir_click_fallback": True,
        "catch_analysis_id": "3393",
    },
    "lg_cns": {
        "name_ko": "LG CNS",
        "aliases": ["LG CNS", "LGCNS", "엘지씨엔에스"],
        "dart_corp_code": "00139834",
        "naver_item_code": "064400",
        "naver_item_name_ko": "LG CNS",
        "saramin_name": "LG CNS",
        "kipris_applicant": "엘지씨엔에스",
        "ir_pages": [
            "https://www.lgcns.com/kr/company/ir/ir-info#실적발표",
        ],
        "ir_fetch_strategy": "playwright_then_httpx",
        "ir_click_fallback": True,
        "catch_analysis_id": "3303",
    },
    "hyundai_autoever": {
        "name_ko": "현대오토에버",
        "aliases": ["현대오토에버", "현대 오토에버", "Hyundai Autoever", "Hyundai AutoEver"],
        "dart_corp_code": "00362441",
        "naver_item_code": "307950",
        "naver_item_name_ko": "현대오토에버",
        "saramin_name": "현대오토에버",
        "kipris_applicant": "현대오토에버",
        "ir_pages": [
            "https://www.hyundai-autoever.com/kor/ir/ir-information/business-performance/list.do",
        ],
        "ir_fetch_strategy": "playwright_then_httpx",
        "ir_click_fallback": True,
        "catch_analysis_id": "3367",
    },
    "posco_dx": {
        "name_ko": "포스코DX",
        "aliases": ["포스코DX", "포스코 DX", "포스코디엑스", "POSCO DX", "포스코ICT", "POSCO ICT"],
        "dart_corp_code": "00155212",
        "naver_item_code": "022100",
        "naver_item_name_ko": "포스코DX",
        "saramin_name": "포스코DX",
        "kipris_applicant": "포스코디엑스",
        "ir_pages": [
            "https://www.poscodx.com/kor/ir/irData.do",
        ],
        "ir_fetch_strategy": "playwright_then_httpx",
        "ir_click_fallback": True,
        "catch_analysis_id": "3507",
    },
    "sk_ax": {
        "name_ko": "SK AX",
        "aliases": [
            "SK AX",
            "SK에이엑스",
            "SK C&C",
            "SK㈜ C&C",
            "SK주식회사 C&C",
            "SK 주식회사 C&C",
            "에스케이씨앤씨",
            "에스케이 씨앤씨",
            "SK Inc.",
            "SK주식회사",
        ],
        "dart_corp_code": "00181712",
        "naver_item_code": "034730",
        "naver_item_name_ko": "SK",
        "ir_pages": [
            "https://sk-inc.com/kr/ir/irArchive.aspx",
            "https://www.sk-inc.com/en/ir/irArchive.aspx",
        ],
        "ir_fetch_strategy": "playwright_then_httpx",
        "ir_click_fallback": True,
        "catch_analysis_id": "3625",
    },
}

COMPANY_IDS: Final[list[str]] = list(COMPANIES.keys())
COMPANY_ALIASES: Final[dict[str, list[str]]] = {
    company_id: config.get("aliases", [company_id]) for company_id, config in COMPANIES.items()
}
PEER_ALIASES: Final[dict[str, list[str]]] = COMPANY_ALIASES
CORP_CODES: Final[dict[str, str]] = {
    company_id: config["dart_corp_code"]
    for company_id, config in COMPANIES.items()
    if "dart_corp_code" in config
}
NAVER_ITEM_CODES: Final[dict[str, str]] = {
    company_id: config["naver_item_code"]
    for company_id, config in COMPANIES.items()
    if "naver_item_code" in config
}
SARAMIN_COMPANY_NAMES: Final[dict[str, str]] = {
    company_id: config["saramin_name"]
    for company_id, config in COMPANIES.items()
    if "saramin_name" in config
}
KIPRIS_APPLICANTS: Final[dict[str, str]] = {
    company_id: config["kipris_applicant"]
    for company_id, config in COMPANIES.items()
    if "kipris_applicant" in config
}
IR_CONFIG: Final[dict[str, dict[str, object]]] = {
    company_id: {
        "pages": config.get("ir_pages", []),
        "fetch_strategy": config.get("ir_fetch_strategy", "playwright_then_httpx"),
        "click_fallback": config.get("ir_click_fallback", False),
    }
    for company_id, config in COMPANIES.items()
    if "ir_pages" in config
}
CATCH_ANALYSIS_IDS: Final[dict[str, str]] = {
    company_id: config["catch_analysis_id"]
    for company_id, config in COMPANIES.items()
    if "catch_analysis_id" in config
}


def company_aliases(company_id: str) -> list[str]:
    if not company_id:
        return []
    return COMPANY_ALIASES.get(company_id, [company_id])


def company_name_ko(company_id: str) -> str:
    config = COMPANIES.get(company_id)
    return config.get("name_ko", company_id) if config else company_id


def all_company_aliases() -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()

    for values in COMPANY_ALIASES.values():
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            aliases.append(value)

    return aliases
