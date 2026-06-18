# 작성일: 2026-05-12
# 작성자: 박지원
# 변경이력:
#   2026-05-12 박지원 — IR 파서 작성 및 전처리·peer profile·스코프 분석 보강
#   2026-05-12 심유정 — feature 머지분 lint·mypy 정리 및 카드뉴스 에이전트 플로우 정리
"""IR PDF 크롤링 결과를 재무 후보 레코드로 변환하는 deterministic parser.

IRCrawler는 PDF 파일을 직접 저장하지 않고 RawArticle 형태로 본문 텍스트와
PDF 페이지 블록을 담는다. 이 파서는 그 RawArticle 결과를 받아
peer_financials 적재에 사용할 수 있는 핵심 재무 후보를 만든다.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.sectors import SECTOR_KEYWORDS
from src.crawler.parsers.pdf_payload import extract_pdf_payload

log = logging.getLogger(__name__)

_REVENUE_PATTERNS = [
    re.compile(
        r"(?:매출액?|Revenue|총매출|Sales)\s*[:：]?\s*([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)\s*(?:매출|sales|revenue)",
        re.IGNORECASE,
    ),
]
_OPERATING_PROFIT_PATTERNS = [
    re.compile(
        r"(?:영업이익|Operating\s*Profit|OP)\s*[:：]?\s*([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)\s*(?:영업이익|operating\s*profit|OP)",
        re.IGNORECASE,
    ),
]
_NET_INCOME_PATTERNS = [
    re.compile(
        r"(?:당기순이익|순이익|Net\s*Income|Net\s*Profit)\s*[:：]?\s*"
        r"([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)[ \t]*(억원|조원|억|조)[ \t]*"
        r"(?:당기순이익|순이익|net\s*income|net\s*profit)",
        re.IGNORECASE,
    ),
]
_EBITDA_PATTERNS = [
    re.compile(
        r"(?:EBITDA|상각전\s*영업이익)\s*[:：]?\s*([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)[ \t]*(억원|조원|억|조)[ \t]*(?:EBITDA|상각전\s*영업이익)",
        re.IGNORECASE,
    ),
]
_BACKLOG_PATTERNS = [
    re.compile(
        r"(?:수주잔고|수주\s*잔고|Backlog|잔여\s*수주|계약\s*잔액)\s*[:：]?\s*"
        r"([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)[ \t]*(억원|조원|억|조)[ \t]*"
        r"(?:수주잔고|수주\s*잔고|backlog|잔여\s*수주|계약\s*잔액)",
        re.IGNORECASE,
    ),
]
_ORDERS_PATTERNS = [
    re.compile(
        r"(?:연결\s*)?(?:수주|신규\s*수주|Order(?:s)?|Order\s*Intake)\s*[:：]?\s*"
        r"([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)[ \t]*(억원|조원|억|조)[ \t]*"
        r"(?:수주|신규\s*수주|order(?:s)?|order\s*intake)",
        re.IGNORECASE,
    ),
]
_CAPEX_PATTERNS = [
    re.compile(
        r"(?:CAPEX|CapEx|설비투자|투자금액|투자\s*집행|자본적\s*지출)\s*[:：]?\s*"
        r"([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)[ \t]*(억원|조원|억|조)[ \t]*"
        r"(?:CAPEX|CapEx|설비투자|투자금액|투자\s*집행|자본적\s*지출)",
        re.IGNORECASE,
    ),
]
_OPERATING_MARGIN_PATTERNS = [
    re.compile(
        r"(?:영업이익률|OPM|Operating[ \t]*Margin)[ \t]*[:：]?[ \t]*"
        r"([\d,]+(?:\.\d+)?)[ \t]*(%|퍼센트|pct|p)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![-△▲])([\d,]+(?:\.\d+)?)[ \t]*(%|퍼센트|pct|p)[ \t]*"
        r"(?:영업이익률|OPM|operating[ \t]*margin)",
        re.IGNORECASE,
    ),
]
_IR_FINANCIAL_METRIC_RULES: tuple[tuple[str, list[re.Pattern[str]], str], ...] = (
    ("revenue_total", _REVENUE_PATTERNS, "amount_krwbn"),
    ("operating_profit", _OPERATING_PROFIT_PATTERNS, "amount_krwbn"),
    ("net_income", _NET_INCOME_PATTERNS, "amount_krwbn"),
    ("ebitda", _EBITDA_PATTERNS, "amount_krwbn"),
    ("backlog", _BACKLOG_PATTERNS, "amount_krwbn"),
    ("orders", _ORDERS_PATTERNS, "amount_krwbn"),
    ("capex", _CAPEX_PATTERNS, "amount_krwbn"),
    ("operating_margin", _OPERATING_MARGIN_PATTERNS, "percentage"),
)
_IR_TABLE_METRIC_ALIASES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "operating_margin",
        ("영업이익률", "opm", "operating margin", "op margin", "margin"),
        "percentage",
    ),
    ("gross_margin", ("gpm", "gross profit margin", "gross margin"), "percentage"),
    ("gross_profit", ("매출총이익", "gross profit"), "amount_krwbn"),
    ("revenue_total", ("매출액", "매출", "revenue", "sales"), "amount_krwbn"),
    ("operating_profit", ("영업이익", "operating profit", "op"), "amount_krwbn"),
    ("net_income", ("당기순이익", "순이익", "net income", "net profit"), "amount_krwbn"),
    ("ebitda", ("ebitda", "상각전영업이익"), "amount_krwbn"),
    ("backlog", ("수주잔고", "backlog", "잔여수주", "계약잔액"), "amount_krwbn"),
    ("orders", ("수주", "신규수주", "orderintake", "orders"), "amount_krwbn"),
    ("capex", ("capex", "설비투자", "투자금액", "자본적지출"), "amount_krwbn"),
)
_IR_TABLE_NON_METRIC_LABEL_PATTERN = re.compile(
    r"매출\s*원가|원가|판매\s*관리비|판관비|영업\s*외\s*손익|"
    r"법인세\s*차감\s*전|세\s*전\s*이\s*익|cost\s*of\s*sales|이익률|margin\s*rate",
    re.IGNORECASE,
)
_IR_TABLE_PERIOD_PATTERN = re.compile(
    r"20\d{2}\s*년\s*(?:[1-4]\s*분기|상반기|하반기)?|"
    r"\d{2}\s*년\s*(?:[1-4]\s*분기|상반기|하반기)?|"
    r"20\d{2}\s*Q\s*[1-4]|"
    r"[1-4]\s*Q\s*['’]?\s*\d{2}|"
    r"\b20\d{2}\b",
    re.IGNORECASE,
)
_IR_TABLE_COMPARISON_PATTERN = re.compile(
    r"\bQoQ\b|\bYoY\b|전분기\s*대비|전년\s*(?:동기\s*)?대비",
    re.IGNORECASE,
)
_IR_TABLE_NUMBER_PATTERN = re.compile(r"\(?[△▲-]?\d[\d,]*(?:\.\d+)?\)?%?")
_IR_TABLE_NOISE_LABEL_PATTERN = re.compile(
    r"yoy|qoq|증감|증가|감소|상승|하락|개선|악화|영향|사유|원인|요청|스케줄|조정|"
    r"전년|전분기|대비|진행중|고부가|프로세스|자동화|운영효율|통한|기반의|고객|"
    r"△|▲",
    re.IGNORECASE,
)
_PERIOD_PATTERNS = [
    re.compile(r"(20\d{2})\s*년?\s*([1-4])\s*분기"),
    re.compile(r"['’]?\s*(\d{2})\s*년\s*([1-4])\s*Q", re.IGNORECASE),
    re.compile(r"['’]?\s*(\d{2})\s*\.\s*([1-4])\s*Q", re.IGNORECASE),
    re.compile(r"(20\d{2})\s*\.\s*([1-4])\s*Q", re.IGNORECASE),
    re.compile(r"(20\d{2})\s*Q\s*([1-4])", re.IGNORECASE),
    re.compile(r"FY\s*(20\d{2})\s*([1-4])\s*Q", re.IGNORECASE),
    re.compile(r"([1-4])\s*Q\s*['’]?\s*(\d{2})", re.IGNORECASE),
]
_PAGE_SPLIT_PATTERN = re.compile(r"(?:^|\n)\[PAGE\s+(\d+)\]\s*", re.IGNORECASE)
_IR_CHUNK_MIN_CHARS = 120
_IR_CHUNK_TARGET_CHARS = 650
_IR_CHUNK_MAX_CHARS = 950
_IR_CHUNK_OVERLAP_CHARS = 120
_IR_LOW_VALUE_PATTERNS = (
    re.compile(r"\bdisclaimer\b", re.IGNORECASE),
    re.compile(r"forward[-\s]?looking", re.IGNORECASE),
    re.compile(r"본\s*자료는\s*투자자", re.IGNORECASE),
    re.compile(r"무단\s*(복제|배포|전재)", re.IGNORECASE),
    re.compile(r"confidential", re.IGNORECASE),
)
_IR_SIGNAL_TERMS = (
    "매출",
    "영업이익",
    "순이익",
    "이익률",
    "revenue",
    "sales",
    "operating profit",
    "op",
    "net income",
    "cloud",
    "클라우드",
    "ai",
    "생성형",
    "ax",
    "dx",
    "erp",
    "scm",
    "스마트",
    "factory",
    "수주",
    "backlog",
    "투자",
    "capex",
    "전망",
    "strategy",
    "성장",
    "risk",
    "리스크",
)
_IR_SECTION_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "summary",
        "Executive Summary",
        ("summary", "highlights", "overview", "요약", "하이라이트", "주요 내용", "경영실적 요약"),
    ),
    (
        "financial",
        "Financial Results",
        ("financial", "results", "실적", "손익", "매출", "영업이익", "재무", "income statement"),
    ),
    (
        "business",
        "Business Segment",
        ("business", "segment", "사업", "부문", "사업별", "사업부", "division"),
    ),
    (
        "cloud",
        "Cloud",
        ("cloud", "클라우드", "aws", "azure", "gcp", "msp", "managed service"),
    ),
    (
        "ai",
        "AI/Data",
        ("ai", "인공지능", "생성형", "llm", "data", "데이터", "analytics", "agentic"),
    ),
    (
        "digital_transformation",
        "Digital Transformation",
        ("dx", "digital", "transformation", "전환", "자동화", "smart", "스마트", "erp", "mes"),
    ),
    (
        "orders_pipeline",
        "Orders/Pipeline",
        ("order", "backlog", "pipeline", "수주", "잔고", "계약", "프로젝트", "pipeline"),
    ),
    (
        "outlook",
        "Outlook",
        ("outlook", "guidance", "forecast", "전망", "계획", "strategy", "전략", "성장"),
    ),
    (
        "shareholder",
        "Shareholder Return",
        ("dividend", "treasury", "shareholder", "배당", "자사주", "주주환원"),
    ),
)
_IR_COMPANY_SECTION_HINTS: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "samsung_sds": (
        ("cloud", "Cloud/MSP", ("msp", "cloud", "클라우드", "하이브리드", "gpu")),
        ("ai", "AI/Data", ("fabrix", "brity", "생성형", "llm", "ai", "데이터")),
        ("logistics", "Digital Logistics", ("첼로", "cello", "물류", "logistics", "scl")),
    ),
    "lg_cns": (
        ("cloud", "Cloud/AM", ("cloudxper", "클라우드", "am", "aws", "azure", "gcp")),
        ("ai", "AI/Data", ("dap", "ai", "data", "factova", "생성형", "agent")),
        ("smart_factory", "Smart Factory", ("smart factory", "스마트팩토리", "mes", "factory")),
    ),
    "hyundai_autoever": (
        (
            "vehicle_sw",
            "Vehicle SW",
            ("vehicle", "차량", "sw", "software-defined", "sdv", "내비게이션"),
        ),
        ("enterprise_it", "Enterprise IT", ("si", "ito", "enterprise", "erp", "그룹사")),
        ("smart_factory", "Smart Factory", ("smart factory", "스마트팩토리", "mes", "mobis")),
    ),
    "posco_dx": (
        (
            "smart_factory",
            "Smart Factory",
            ("smart factory", "스마트팩토리", "자동화", "제철소", "철강"),
        ),
        (
            "robotics",
            "Robotics/Automation",
            ("robot", "로봇", "automation", "자동화", "물류자동화"),
        ),
        ("industrial_ai", "Industrial AI", ("ai", "산업", "vision", "예지", "품질")),
    ),
    "sk_ax": (
        ("ai", "AI/Data", ("ax", "생성형", "llm", "data", "데이터", "ai transformation")),
        ("cloud", "Cloud", ("cloud", "클라우드", "dc", "data center", "데이터센터")),
        ("enterprise_it", "Enterprise IT", ("c&c", "it서비스", "it 서비스", "si", "ito")),
    ),
}
_IR_COMPANY_TOTAL_TERMS = (
    "financial results",
    "경영실적 종합",
    "경영실적",
    "실적 요약",
    "consolidated",
    "연결",
    "company total",
    "overall",
    "income statement",
    "손익계산서",
    "손익",
    "전사",
    "전체",
)
_IR_PORTFOLIO_TERMS = (
    "portfolio",
    "post-rebalancing",
    "investment",
    "subsidiar",
    "affiliate",
    "sk inc. at a glance",
    "주요비상장자회사",
    "비상장자회사",
    "자회사분기별실적",
    "투자회사",
    "투자 포트폴리오",
    "자회사",
    "관계사",
    "포트폴리오",
    "에스케이이노베이션",
    "sk이노베이션",
    "sk innovation",
    "sk스퀘어",
    "sk square",
    "sk바이오팜",
    "sk biopharmaceuticals",
    "sk텔레콤",
    "sk telecom",
    "sk하이닉스",
    "sk hynix",
    "sk e&s",
)
_IR_SEGMENT_CONTEXT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cloud",
        (
            "cloud",
            "클라우드",
            "msp",
            "csp",
            "aws",
            "azure",
            "gcp",
            "data center",
            "데이터센터",
            "dc",
        ),
    ),
    (
        "ai_ax",
        (
            "ai",
            "생성형",
            "genai",
            "llm",
            "agent",
            "fabrix",
            "brity",
            "enterprise ai",
            "제조 ax",
            "공공 ax",
            "erp ai",
            "scm",
        ),
    ),
    ("logistics", ("logistics", "물류", "cello", "scl")),
    ("smart_factory", ("smart factory", "스마트팩토리", "mes", "factory", "제조")),
    ("vehicle_sw", ("vehicle", "차량", "sdv", "내비게이션", "navigation")),
    ("enterprise_it", ("enterprise", "erp", "ito", "si", "그룹사", "it서비스")),
    ("robotics", ("robot", "로봇", "automation", "자동화")),
)
_SK_AX_PAGE_STRONG_TERMS = (
    "sk ax",
    "sk에이엑스",
    "sk㈜ c&c",
    "sk(주) c&c",
    "sk 주식회사 c&c",
    "sk주식회사 c&c",
    "sk주식회사 사업부문",
    "sk c&c",
    "sk c & c",
    "sk cnc",
    "에스케이씨앤씨",
    "c&c",
    "씨앤씨",
)
_SK_AX_PAGE_BUSINESS_TERMS = (
    "it서비스",
    "it 서비스",
    "it서비스 부문",
    "사업부문",
    "it services",
    "information technology services",
    "enterprise it",
    "digital service",
    "digital services",
    "si",
    "ito",
    "클라우드",
    "cloud",
    "데이터센터",
    "data center",
    "ai transformation",
    "ax",
)
_SK_AX_PAGE_EXCLUDE_TERMS = (
    "sk telecom",
    "sk텔레콤",
    "skt",
    "에이닷",
    "sk hynix",
    "sk하이닉스",
    "sk innovation",
    "sk이노베이션",
    "sk square",
    "sk스퀘어",
    "sk biopharmaceuticals",
    "sk바이오팜",
    "sk e&s",
    "sk온",
    "sk on",
    "sk ecoplant",
    "sk에코플랜트",
)


def _normalize_amount_krwbn(value: str, unit: str) -> float:
    """금액 문자열을 기존 peer_financials 관례인 억원 단위 값으로 변환한다."""
    n = float(value.replace(",", ""))
    if unit in ("조원", "조"):
        return n * 10_000
    return n


def _extract_period(text: str) -> str | None:
    year_match = re.search(r"(20\d{2})\s*년\s*(?:경영\s*)?실적", text or "")
    if year_match:
        return year_match.group(1)

    for pattern in _PERIOD_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue

        if len(match.group(1)) == 1 and len(match.group(2)) == 2:
            return f"20{match.group(2)}Q{match.group(1)}"
        if len(match.group(1)) == 2 and len(match.group(2)) == 1:
            return f"20{match.group(1)}Q{match.group(2)}"

        return f"{match.group(1)}Q{match.group(2)}"
    return None


def _period_parts(period: str | None) -> tuple[int | None, int | None, str | None]:
    if not period:
        return None, None, None

    match = re.match(r"^(20\d{2})Q([1-4])$", period)
    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))
        return year, quarter, "quarter"

    match = re.match(r"^(20\d{2})$", period)
    if match:
        return int(match.group(1)), None, "year"

    return None, None, None


def _first_amount(
    text: str,
    patterns: list[re.Pattern[str]],
) -> tuple[float, str] | tuple[None, None]:
    for pattern in patterns:
        match = pattern.search(text or "")
        if match:
            return _normalize_amount_krwbn(match.group(1), match.group(2)), match.group(0)

    return None, None


def _normalize_percentage(value: str) -> float:
    return float(value.replace(",", ""))


def _metric_values(
    text: str,
    patterns: list[re.Pattern[str]],
    value_kind: str,
) -> list[tuple[float, str]]:
    values: list[tuple[float, str]] = []
    seen: set[tuple[str, float]] = set()
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            if value_kind == "percentage":
                value = _normalize_percentage(match.group(1))
            else:
                value = _normalize_amount_krwbn(match.group(1), match.group(2))
            raw = match.group(0)
            dedupe_key = (raw, value)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            values.append((value, raw))

    return values


def _first_metric_value(
    text: str,
    patterns: list[re.Pattern[str]],
    value_kind: str,
) -> tuple[float, str] | tuple[None, None]:
    values = _metric_values(text, patterns, value_kind)
    if not values:
        return None, None
    return values[0]


def _candidate_value_key(value_kind: str) -> str:
    if value_kind == "percentage":
        return "value_pct"
    return "value_krwbn"


def _signed_percentage(raw_value: str, sign_token: str | None = None) -> float | None:
    try:
        value = float(str(raw_value).replace(",", "").rstrip("%"))
    except (TypeError, ValueError):
        return None
    sign = str(sign_token or "").strip()
    if sign in {"-", "△", "▲"}:
        value *= -1
    return value


def _inline_comparison_candidates(
    *,
    page_text: str,
    page_no: Any,
    metric_type: str,
    raw: str,
    base_candidate: dict[str, Any],
    report_period: str | None,
) -> list[dict[str, Any]]:
    if not raw or not report_period:
        return []

    compact_text = " ".join((page_text or "").split())
    raw_index = compact_text.find(raw)
    if raw_index < 0:
        return []

    window = compact_text[raw_index : raw_index + len(raw) + 80]
    candidates: list[dict[str, Any]] = []
    for comparison_type in ("yoy", "qoq"):
        label_pattern = (
            "YoY|전년\\s*(?:동기\\s*)?대비" if comparison_type == "yoy" else "QoQ|전분기\\s*대비"
        )
        match = re.search(
            rf"(?:{label_pattern})\s*([+\-△▲]?)\s*([\d,]+(?:\.\d+)?)\s*%",
            window,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        value = _signed_percentage(match.group(2), match.group(1))
        if value is None:
            continue
        year, quarter, period_type = _period_parts(report_period)
        candidates.append(
            {
                **base_candidate,
                "page": page_no,
                "type": f"{metric_type}_{comparison_type}",
                "base_metric_type": metric_type,
                "comparison_type": comparison_type,
                "value_kind": "percentage",
                "value_pct": value,
                "unit": "%",
                "raw": f"{raw} {match.group(0)}",
                "source": "ir_inline_comparison",
                "period": report_period,
                "period_year": year,
                "period_quarter": quarter,
                "period_type": period_type,
                "confidence": min(float(base_candidate.get("confidence") or 0.8), 0.86),
                "evidence_text": _metric_context_window(page_text, f"{raw} {match.group(0)}"),
            }
        )
    return candidates


def _has_same_table_metric_candidate(
    candidates: list[dict[str, Any]],
    *,
    page_no: Any,
    metric_type: str,
    value_key: str,
    value: float,
) -> bool:
    for candidate in candidates:
        if candidate.get("source") != "ir_table_matrix":
            continue
        if candidate.get("page") != page_no or candidate.get("type") != metric_type:
            continue
        candidate_value = candidate.get(value_key)
        if (
            isinstance(candidate_value, int | float)
            and abs(float(candidate_value) - value) < 0.0001
        ):
            return True
    return False


def _classify_metric_context(
    page_text: str,
    *,
    peer_id: str | None,
    raw_match: str | None = None,
) -> dict[str, Any]:
    text = " ".join((page_text or "").split()).lower()
    metric_window = _metric_context_window(page_text, raw_match)
    current_line_text = _metric_current_line_text(page_text, raw_match)
    metric_text = _metric_context_text(page_text, raw_match)
    has_total = any(term.lower() in text for term in _IR_COMPANY_TOTAL_TERMS)
    portfolio_entity = (
        _detect_portfolio_entity(metric_window)
        or _detect_portfolio_entity(current_line_text)
        or _detect_portfolio_entity(metric_text)
    )
    metric_segment_area = (
        _detect_business_area(current_line_text)
        or _detect_business_area(metric_window)
        or _detect_business_area(metric_text)
    )
    segment_area = metric_segment_area or _detect_business_area(text)

    if portfolio_entity:
        return {
            "metric_scope": "portfolio_company",
            "business_area": "portfolio",
            "entity_name": portfolio_entity,
            "confidence": 0.78,
            "classification_reason": (
                f"metric 주변 문맥에서 SK 포트폴리오 회사 '{portfolio_entity}'가 확인되어 "
                "SK AX/피어 본체 지표가 아닌 portfolio_company로 분류"
            ),
            "context_evidence": _metric_context_evidence(
                current_line_text=current_line_text,
                metric_window=metric_window,
                metric_text=metric_text,
            ),
        }

    if metric_segment_area:
        return {
            "metric_scope": "segment",
            "business_area": metric_segment_area,
            "entity_name": peer_id,
            "confidence": 0.78,
            "classification_reason": (
                f"metric가 있는 행/주변 문맥에서 '{metric_segment_area}' 사업 키워드가 "
                "직접 확인되어 segment 지표로 분류"
            ),
            "context_evidence": _metric_context_evidence(
                current_line_text=current_line_text,
                metric_window=metric_window,
                metric_text=metric_text,
            ),
        }

    if segment_area and not has_total:
        return {
            "metric_scope": "segment",
            "business_area": segment_area,
            "entity_name": peer_id,
            "confidence": 0.72,
            "classification_reason": (
                f"페이지 문맥에서 '{segment_area}' 사업 키워드가 확인되고 전사/연결/전체 "
                "표현은 없어 segment 지표로 분류"
            ),
            "context_evidence": _metric_context_evidence(
                current_line_text=current_line_text,
                metric_window=metric_window,
                metric_text=metric_text,
            ),
        }

    if has_total:
        return {
            "metric_scope": "company_total",
            "business_area": "company_total",
            "entity_name": peer_id,
            "confidence": 0.85,
            "classification_reason": (
                "페이지 또는 표 문맥에서 연결/전사/전체/경영실적 등 회사 전체를 나타내는 "
                "표현이 확인되어 company_total 지표로 분류"
            ),
            "context_evidence": _metric_context_evidence(
                current_line_text=current_line_text,
                metric_window=metric_window,
                metric_text=metric_text,
            ),
        }

    return {
        "metric_scope": "unknown",
        "business_area": segment_area or "company_total",
        "entity_name": peer_id,
        "confidence": 0.55,
        "classification_reason": (
            "metric 주변에서 전사/사업부문 판단 근거가 충분하지 않아 unknown으로 분류"
        ),
        "context_evidence": _metric_context_evidence(
            current_line_text=current_line_text,
            metric_window=metric_window,
            metric_text=metric_text,
        ),
    }


def _metric_context_evidence(
    *,
    current_line_text: str,
    metric_window: str,
    metric_text: str,
) -> str:
    evidence_parts = [
        f"metric 행: {current_line_text}" if current_line_text else "",
        f"주변 문맥: {metric_text or metric_window}" if metric_text or metric_window else "",
    ]
    return " | ".join(part for part in evidence_parts if part)[:1200]


def _metric_context_text(page_text: str, raw_match: str | None) -> str:
    line_context = _metric_line_context(page_text, raw_match)
    if line_context:
        return line_context
    return _metric_context_window(page_text, raw_match)


def _metric_current_line_text(page_text: str, raw_match: str | None) -> str:
    if not raw_match:
        return ""

    raw_lower = raw_match.lower()
    for line in str(page_text or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if raw_lower in line.lower():
            return line.lower()
    return ""


def _metric_line_context(page_text: str, raw_match: str | None) -> str | None:
    if not raw_match:
        return None

    lines = [re.sub(r"\s+", " ", line).strip() for line in str(page_text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None

    raw_lower = raw_match.lower()
    for index, line in enumerate(lines):
        if raw_lower not in line.lower():
            continue

        start = max(0, index - 3)
        context_lines = lines[start : index + 1]
        return " ".join(context_lines).lower()

    return None


def _metric_context_window(page_text: str, raw_match: str | None) -> str:
    compact_text = " ".join((page_text or "").split())
    if not raw_match:
        return compact_text.lower()

    match_index = compact_text.lower().find(raw_match.lower())
    if match_index < 0:
        return compact_text.lower()

    start = max(0, match_index - 35)
    end = min(len(compact_text), match_index + len(raw_match))
    return compact_text[start:end].lower()


def _detect_business_area(text: str) -> str | None:
    lowered = (text or "").lower()
    for business_area, terms in _IR_SEGMENT_CONTEXT_RULES:
        if any(term.lower() in lowered for term in terms):
            return business_area
    return None


def _detect_portfolio_entity(text: str) -> str | None:
    entity_terms = (
        ("sk_innovation", ("sk이노베이션", "sk innovation", "에스케이이노베이션")),
        ("sk_square", ("sk스퀘어", "sk square")),
        ("sk_biopharmaceuticals", ("sk바이오팜", "sk biopharmaceuticals")),
        ("sk_telecom", ("sk텔레콤", "sk telecom")),
        ("sk_hynix", ("sk하이닉스", "sk hynix")),
        ("sk_e_and_s", ("sk e&s", "에스케이 e&s")),
    )
    for entity_name, terms in entity_terms:
        if any(term.lower() in text for term in terms):
            return entity_name
    return None


def _contains_token(text: str, term: str) -> bool:
    lowered_term = term.lower()
    if len(lowered_term) <= 3 and re.fullmatch(r"[a-z0-9&]+", lowered_term):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(lowered_term)}(?![a-z0-9])",
                text,
            )
        )
    return lowered_term in text


def _is_sk_ax_page(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return False

    has_strong_term = any(_contains_token(lowered, term) for term in _SK_AX_PAGE_STRONG_TERMS)
    if has_strong_term:
        return True

    has_excluded_affiliate = any(
        _contains_token(lowered, term) for term in _SK_AX_PAGE_EXCLUDE_TERMS
    )
    if has_excluded_affiliate:
        return False

    business_hits = sum(1 for term in _SK_AX_PAGE_BUSINESS_TERMS if _contains_token(lowered, term))
    return business_hits >= 2


def _filter_pages_for_peer(
    pages: list[dict[str, Any]], peer_id: str | None
) -> list[dict[str, Any]]:
    if peer_id != "sk_ax":
        return pages

    filtered = [page for page in pages if _is_sk_ax_page(str(page.get("text") or ""))]
    if filtered:
        log.info(
            "SK AX IR 관련 페이지 필터 적용 | before=%d after=%d pages=%s",
            len(pages),
            len(filtered),
            [page.get("page") for page in filtered],
        )
        return filtered

    log.warning(
        "SK AX IR 관련 페이지를 찾지 못해 원본 페이지 전체로 fallback | pages=%d", len(pages)
    )
    return pages


def _article_get(article: Any, key: str, default: Any = None) -> Any:
    if isinstance(article, dict):
        return article.get(key, default)
    return getattr(article, key, default)


def _article_extra(article: Any) -> dict[str, Any]:
    extra = _article_get(article, "extra", {}) or {}
    if not extra:
        extra = _article_get(article, "metadata", {}) or {}
    return extra if isinstance(extra, dict) else {}


def _article_peer_id(article: Any) -> str | None:
    peer_id = _article_get(article, "peer_id")
    if peer_id:
        return str(peer_id)

    company = _article_get(article, "company", []) or []
    if isinstance(company, list) and company:
        return str(company[0])

    return None


def _article_published_at(article: Any, extra: dict[str, Any]) -> str | None:
    published_at = _article_get(article, "published_at")
    if isinstance(published_at, datetime):
        return published_at.isoformat()
    if published_at:
        return str(published_at)

    value = extra.get("published_at")
    return str(value) if value else None


def _period_from_ir_article(article: Any, extra: dict[str, Any], text: str) -> str | None:
    title_period = _extract_period(str(_article_get(article, "title", "") or ""))
    text_period = _extract_period(text[:5000])
    if text_period:
        return text_period
    if title_period:
        return title_period

    date_info = extra.get("date_info") or {}
    if isinstance(date_info, dict):
        year = date_info.get("year")
        quarter = date_info.get("quarter")
        if year and quarter:
            return f"{year}Q{quarter}"

    return _extract_period(
        " ".join(
            [
                str(_article_get(article, "title", "") or ""),
                text[:5000],
            ]
        )
    )


def _pages_from_ir_article(article: Any, extra: dict[str, Any]) -> list[dict[str, Any]]:
    page_blocks = extra.get("pdf_page_blocks") or []
    pages: list[dict[str, Any]] = []

    if isinstance(page_blocks, list):
        for idx, page in enumerate(page_blocks, start=1):
            if not isinstance(page, dict):
                continue

            blocks = page.get("blocks") or []
            text = _page_text_from_pdf_blocks(blocks)
            page_no = int(page.get("page") or idx)

            if text:
                pages.append({"page": page_no, "text": text, "blocks": blocks})

    content_pages = _pages_from_article_content(str(_article_get(article, "content", "") or ""))
    if _should_use_content_pages(content_pages, pages):
        return content_pages

    if pages:
        return pages

    return content_pages


def _pages_from_article_content(text: str) -> list[dict[str, Any]]:
    matches = list(_PAGE_SPLIT_PATTERN.finditer(text))

    if not matches:
        return [{"page": None, "text": text}] if text else []

    pages: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        page_no = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            pages.append({"page": page_no, "text": page_text})

    return pages


def _should_use_content_pages(
    content_pages: list[dict[str, Any]],
    block_pages: list[dict[str, Any]],
) -> bool:
    if not content_pages:
        return False
    if not block_pages:
        return True

    content_text = "\n".join(str(page.get("text") or "") for page in content_pages)
    block_text = "\n".join(str(page.get("text") or "") for page in block_pages)
    if "[OCR]" in content_text:
        return True

    content_chars = len(re.findall(r"[0-9A-Za-z가-힣]", content_text))
    block_chars = len(re.findall(r"[0-9A-Za-z가-힣]", block_text))
    return content_chars > block_chars + 500 and content_chars > block_chars * 1.2


def _page_text_from_pdf_blocks(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return ""

    positioned: list[dict[str, Any]] = []
    fallback_texts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip()
        if not text:
            continue
        fallback_texts.append(text)
        bbox = block.get("bbox")
        if not (
            isinstance(bbox, list | tuple)
            and len(bbox) >= 4
            and all(isinstance(value, int | float) for value in bbox[:4])
        ):
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
        positioned.append(
            {
                "text": text,
                "x0": x0,
                "y_center": (y0 + y1) / 2,
                "height": max(y1 - y0, 1.0),
            }
        )

    if not positioned:
        return "\n".join(fallback_texts).strip()

    positioned.sort(key=lambda item: (item["y_center"], item["x0"]))
    median_height = sorted(item["height"] for item in positioned)[len(positioned) // 2]
    tolerance = max(4.0, min(10.0, median_height * 0.75))

    rows: list[list[dict[str, Any]]] = []
    for item in positioned:
        if not rows:
            rows.append([item])
            continue
        current_row = rows[-1]
        row_center = sum(cell["y_center"] for cell in current_row) / len(current_row)
        if abs(item["y_center"] - row_center) <= tolerance:
            current_row.append(item)
        else:
            rows.append([item])

    lines = [
        " ".join(cell["text"] for cell in sorted(row, key=lambda item: item["x0"])).strip()
        for row in rows
    ]
    return "\n".join(line for line in lines if line).strip()


def _candidate_page(candidates: list[dict[str, Any]], metric_type: str) -> int | None:
    for candidate in candidates:
        if candidate.get("type") == metric_type:
            page = candidate.get("page")
            return int(page) if isinstance(page, int) else None
    return None


def _extract_financial_table_candidates(
    pages: list[dict[str, Any]],
    *,
    report_period: str | None,
    peer_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    for page in pages:
        page_no = page.get("page")
        page_text = str(page.get("text") or "")
        lines = [re.sub(r"\s+", " ", line).strip() for line in page_text.splitlines()]
        lines = [line for line in lines if line]
        lines = _merge_split_operating_margin_lines(lines)
        unit = _table_unit(page_text)
        unit_evidence = _table_unit_evidence(lines)

        for header_index, line in enumerate(lines):
            table_title = _nearby_table_title(lines, header_index)
            table_unit = (
                _table_unit("\n".join(lines[max(0, header_index - 3) : header_index + 2])) or unit
            )
            columns = _table_columns_from_header_lines(
                line,
                lines[header_index + 1] if header_index + 1 < len(lines) else None,
                lines[header_index - 1] if header_index > 0 else None,
                report_period=report_period,
            )
            if header_index + 1 < len(lines):
                columns.extend(_comparison_columns_from_continuation(lines[header_index + 1]))
                columns = _dedupe_table_columns(columns)
            columns = _coerce_quarterly_context_columns(
                columns,
                table_title=table_title,
                report_period=report_period,
            )
            period_columns = [column for column in columns if column.get("period")]
            if len(period_columns) < 2:
                continue
            if _looks_like_parent_year_header_only(
                line,
                lines[header_index + 1] if header_index + 1 < len(lines) else None,
                period_columns,
            ):
                continue
            if _has_future_quarter_column(period_columns, report_period):
                continue
            required_value_count = len(period_columns)

            table_uid = f"ir-p{page_no or 'x'}-t{len(tables) + 1}"
            if _should_skip_financial_table(
                peer_id=peer_id,
                table_title=table_title,
                lines=lines,
                header_index=header_index,
            ):
                continue
            table_context_business_area = _table_context_business_area(lines, header_index)
            table_rows: list[dict[str, Any]] = []
            active_metric: tuple[str, str] | None = None
            active_metric_parent_label: str | None = None
            active_amount_metric: tuple[str, str] | None = None
            active_amount_metric_parent_label: str | None = None
            active_business_area: str | None = None
            miss_count = 0
            miss_limit = 8 if required_value_count >= 10 or len(columns) >= 9 else 3
            for row_index, row_line in enumerate(
                lines[header_index + 1 : header_index + 24], start=1
            ):
                row_metric = _table_metric_from_label(row_line)
                row_label = _table_row_label(row_line)
                explicit_metric = row_metric is not None
                if row_metric is not None:
                    active_metric = row_metric
                    active_metric_parent_label = row_label
                    if row_metric[1] != "percentage":
                        active_amount_metric = row_metric
                        active_amount_metric_parent_label = row_label
                elif _looks_like_segment_value_row(row_line, required_value_count):
                    row_values_for_kind = _table_row_values(row_line, len(columns))
                    if (
                        active_metric
                        and active_metric[1] == "percentage"
                        and _row_values_have_percentage(row_values_for_kind)
                    ):
                        row_metric = active_metric
                    elif _row_values_are_percentage(row_values_for_kind) and active_metric:
                        row_metric = active_metric
                    elif active_amount_metric:
                        row_metric = active_amount_metric
                        active_metric_parent_label = active_amount_metric_parent_label
                    elif active_metric and active_metric[1] != "percentage":
                        row_metric = active_metric
                    elif active_metric and active_metric[1] == "percentage":
                        row_metric = ("revenue_total", "amount_krwbn")
                        active_metric_parent_label = None
                    else:
                        if table_rows:
                            miss_count += 1
                            if miss_count >= miss_limit:
                                break
                        continue
                else:
                    if table_rows:
                        miss_count += 1
                        if miss_count >= miss_limit:
                            break
                    continue

                metric_name, value_kind = row_metric
                row_values = _table_row_values(row_line, len(columns))
                if _row_has_misaligned_leading_comparison_columns(columns, row_values):
                    if table_rows:
                        miss_count += 1
                        if miss_count >= miss_limit:
                            break
                    continue
                if len(row_values) < required_value_count:
                    if table_rows:
                        miss_count += 1
                        if miss_count >= miss_limit:
                            break
                    continue

                if not _is_valid_table_metric_row(row_label, row_metric):
                    if table_rows:
                        miss_count += 1
                        if miss_count >= miss_limit:
                            break
                    continue
                display_unit = _table_display_unit(unit=table_unit, value_kind=value_kind)
                row_unit_evidence = _effective_row_unit_evidence(
                    unit_evidence=unit_evidence,
                    row_label=row_label,
                    metric_parent_label=active_metric_parent_label,
                    value_kind=value_kind,
                )
                row_evidence = _table_row_evidence(
                    table_title=table_title,
                    context_business_area=table_context_business_area,
                    metric_parent_label=active_metric_parent_label if not explicit_metric else None,
                    row_label=row_label,
                    columns=columns,
                    row_values=row_values,
                    unit=display_unit,
                    unit_evidence=row_unit_evidence,
                )
                business_area = _table_business_area(
                    row_label,
                    page_text,
                    peer_id=peer_id,
                    context_business_area=table_context_business_area,
                    explicit_metric=explicit_metric,
                    active_metric_parent_label=active_metric_parent_label,
                )
                if (
                    explicit_metric
                    and value_kind == "percentage"
                    and active_business_area
                    and _is_percentage_metric_label(row_label, metric_name)
                ):
                    business_area = active_business_area
                metric_scope = "company_total" if business_area == "company_total" else "segment"
                classification_reason = _table_classification_reason(
                    business_area=business_area,
                    metric_scope=metric_scope,
                    row_label=row_label,
                    table_title=table_title,
                    context_business_area=table_context_business_area,
                    metric_parent_label=active_metric_parent_label if not explicit_metric else None,
                    explicit_metric=explicit_metric,
                )
                table_cells: list[dict[str, Any]] = []
                primary_period_column = _primary_period_column(columns)
                for column, raw_value in zip(columns, row_values, strict=False):
                    comparison_type = column.get("comparison_type")
                    candidate_metric_name = metric_name
                    candidate_value_kind = value_kind
                    candidate_unit = table_unit
                    period_column = column
                    if comparison_type:
                        if "%" not in str(raw_value):
                            continue
                        candidate_metric_name = _comparison_metric_name(
                            metric_name,
                            str(comparison_type),
                        )
                        candidate_value_kind = "percentage"
                        candidate_unit = "%"
                        period_column = (
                            _comparison_target_column(column) or primary_period_column or column
                        )
                    comparison_base = (
                        _comparison_base_cell(
                            comparison_type=str(comparison_type),
                            columns=columns,
                            row_values=row_values,
                            current_period=period_column.get("period"),
                            unit=table_unit,
                        )
                        if comparison_type
                        else {}
                    )

                    normalized = _table_value(
                        raw_value,
                        unit=candidate_unit,
                        value_kind=candidate_value_kind,
                    )
                    if normalized is None:
                        continue
                    cell_unit = _table_cell_unit(
                        raw_value,
                        table_unit=candidate_unit,
                        value_kind=candidate_value_kind,
                    )
                    period = period_column.get("period")
                    candidate: dict[str, Any] = {
                        "page": page_no,
                        "type": candidate_metric_name,
                        "base_metric_type": metric_name,
                        "comparison_type": comparison_type,
                        "value_kind": candidate_value_kind,
                        _candidate_value_key(candidate_value_kind): normalized,
                        "unit": cell_unit,
                        "unit_evidence": row_unit_evidence,
                        "raw": f"{row_label} {column['label']} {raw_value} ({cell_unit})",
                        "source": "ir_table_matrix",
                        "source_table_uid": table_uid,
                        "table_title": table_title,
                        "metric_parent_label": active_metric_parent_label
                        if not explicit_metric
                        else None,
                        "row_label": row_label,
                        "column_label": column["label"],
                        "row_evidence": row_evidence,
                        "evidence_text": (
                            f"{row_evidence} | 선택 셀: {column['label']}={raw_value} "
                            f"({cell_unit}) | 분류 근거: {classification_reason}"
                        ),
                        "classification_reason": classification_reason,
                        "context_evidence": row_evidence,
                        "period": period,
                        "period_year": period_column.get("period_year"),
                        "period_quarter": period_column.get("period_quarter"),
                        "period_type": period_column.get("period_type"),
                        "is_historical": _is_historical_period(period, report_period),
                        **comparison_base,
                        "metric_scope": metric_scope,
                        "business_area": business_area,
                        "entity_name": peer_id,
                        "confidence": 0.88,
                    }
                    candidates.append(candidate)
                    table_cells.append(candidate)
                if table_cells:
                    miss_count = 0
                    table_rows.append(
                        {
                            "row_index": row_index,
                            "row_label": row_label,
                            "metric_name": metric_name,
                            "business_area": business_area,
                            "cells": [
                                {
                                    "column_label": cell["column_label"],
                                    "period": cell["period"],
                                    "raw": cell["raw"],
                                    "type": cell.get("type"),
                                    "comparison_type": cell.get("comparison_type"),
                                    "value": cell.get(_candidate_value_key(cell["value_kind"])),
                                }
                                for cell in table_cells
                            ],
                        }
                    )
                    if metric_scope == "segment" and business_area:
                        active_business_area = str(business_area)
                    elif explicit_metric and value_kind != "percentage":
                        active_business_area = None

            if table_rows:
                tables.append(
                    {
                        "table_uid": table_uid,
                        "page": page_no,
                        "title": table_title,
                        "context_business_area": table_context_business_area,
                        "unit": table_unit,
                        "unit_evidence": unit_evidence,
                        "columns": columns,
                        "rows": table_rows,
                    }
                )

    return candidates, tables


def _merge_split_operating_margin_lines(lines: list[str]) -> list[str]:
    """Merge OCR rows split as `영업` / percentage row / `이익률`."""
    merged: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        next_next = lines[index + 2] if index + 2 < len(lines) else ""
        if (
            re.fullmatch(r"영업", current or "")
            and _row_values_are_percentage(_table_row_values(next_line, 32))
            and re.fullmatch(r"이익률", next_next or "")
        ):
            merged.append(f"영업이익률 {next_line}")
            index += 3
            continue
        merged.append(current)
        index += 1
    return merged


def _extract_chart_block_candidates(
    pages: list[dict[str, Any]],
    *,
    report_period: str | None,
    peer_id: str | None,
) -> list[dict[str, Any]]:
    year, quarter, period_type = _period_parts(report_period)
    if period_type != "quarter" or not year or not quarter:
        return []

    axis_tokens = {
        f"{quarter}q{str(year)[2:]}",
        f"{quarter} q{str(year)[2:]}",
    }
    candidates: list[dict[str, Any]] = []
    for page in pages:
        blocks = page.get("blocks") or []
        if not isinstance(blocks, list):
            continue

        metric_labels = [
            block
            for block in blocks
            if "영업이익률" in str(block.get("text") or "") and _block_center(block) is not None
        ]
        if not metric_labels:
            continue

        axis_blocks = []
        for block in blocks:
            text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip().lower()
            compact = text.replace(" ", "")
            if any(token.replace(" ", "") in compact for token in axis_tokens):
                axis_blocks.append(block)
        if not axis_blocks:
            continue

        axis_center = _block_center(axis_blocks[-1])
        if axis_center is None:
            continue
        axis_x, axis_y = axis_center

        pct_blocks: list[tuple[float, float, float, str]] = []
        for block in blocks:
            text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip()
            match = re.fullmatch(r"([+\-△▲]?)\s*([\d,]+(?:\.\d+)?)\s*%", text)
            if not match:
                continue
            value = _signed_percentage(match.group(2), match.group(1))
            center = _block_center(block)
            if value is None or center is None:
                continue
            x, y = center
            if y >= axis_y:
                continue
            pct_blocks.append((abs(x - axis_x), x, value, text))

        if not pct_blocks:
            continue

        pct_blocks.sort(key=lambda item: item[0])
        distance, _x, value, raw = pct_blocks[0]
        if distance > 90:
            continue

        metric_context = _classify_metric_context(
            str(page.get("text") or ""),
            peer_id=peer_id,
            raw_match=raw,
        )
        if metric_context.get("metric_scope") == "unknown":
            metric_context = {
                **metric_context,
                "metric_scope": "company_total",
                "business_area": "company_total",
                "entity_name": peer_id,
                "confidence": 0.82,
                "classification_reason": "IR 차트 축의 보고기간 라벨과 영업이익률 값을 좌표로 매칭",
            }
        candidates.append(
            {
                "page": page.get("page"),
                "type": "operating_margin",
                "value_kind": "percentage",
                "value_pct": value,
                "unit": "%",
                "raw": f"영업이익률 {report_period} {raw}",
                "source": "ir_chart_blocks",
                "period": report_period,
                "period_year": year,
                "period_quarter": quarter,
                "period_type": period_type,
                "evidence_text": f"영업이익률 차트 {report_period}={raw}",
                **metric_context,
                "confidence": max(float(metric_context.get("confidence") or 0.0), 0.86),
            }
        )

    return candidates


def _block_center(block: dict[str, Any]) -> tuple[float, float] | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    return (x0 + x1) / 2, (y0 + y1) / 2


def _table_columns_from_header_lines(
    line: str,
    next_line: str | None,
    prev_line: str | None = None,
    *,
    report_period: str | None = None,
) -> list[dict[str, Any]]:
    posco_columns = _posco_dx_appendix_statement_columns(prev_line, line, next_line)
    if posco_columns:
        return posco_columns

    posco_compact_columns = _posco_dx_compact_statement_columns(
        prev_line,
        line,
        next_line,
        report_period=report_period,
    )
    if posco_compact_columns:
        return posco_compact_columns

    hierarchical = _hierarchical_table_columns(line, next_line)
    if hierarchical:
        return hierarchical
    return _table_columns_from_header(line)


def _table_columns_from_header(line: str) -> list[dict[str, Any]]:
    repaired_columns = _ocr_scrambled_same_line_columns(line)
    if repaired_columns:
        return repaired_columns

    columns: list[dict[str, Any]] = []
    for match in _IR_TABLE_PERIOD_PATTERN.finditer(line or ""):
        label = match.group(0).strip()
        period, period_year, period_quarter, period_type = _normalize_table_period(label)
        if not period:
            continue
        columns.append(
            {
                "label": label,
                "period": period,
                "period_year": period_year,
                "period_quarter": period_quarter,
                "period_type": period_type,
                "start": match.start(),
            }
        )
    for match in _IR_TABLE_COMPARISON_PATTERN.finditer(line or ""):
        label = match.group(0).strip()
        columns.append(
            {
                "label": label,
                "comparison_type": _comparison_column_type(label),
                "start": match.start(),
            }
        )
    columns.sort(key=lambda column: int(column.get("start") or 0))
    for column in columns:
        column.pop("start", None)
    return columns


def _hierarchical_table_columns(line: str, next_line: str | None) -> list[dict[str, Any]]:
    if not line or not next_line:
        return []

    year_matches = list(re.finditer(r"(20\d{2}|\d{2})\s*년", line))
    if not year_matches:
        year_matches = list(re.finditer(r"\b(20\d{2})\b", line))
    if not year_matches:
        return []

    subpattern = re.compile(
        r"[1-4]\s*분기|연간|\b[1-4]\s*Q\b|\b[12]\s*H\b|\bQoQ\b|\bYoY\b|"
        r"전분기\s*대비|전년\s*(?:동기\s*)?대비",
        re.IGNORECASE,
    )
    sub_matches = list(subpattern.finditer(next_line))
    if not sub_matches:
        return []
    if all(_IR_TABLE_COMPARISON_PATTERN.fullmatch(match.group(0).strip()) for match in sub_matches):
        return []
    if len(sub_matches) < 2 and _IR_TABLE_NUMBER_PATTERN.search(next_line):
        return []

    line_width = max(len(line), 1)
    sub_width = max(len(next_line), 1)
    years = [
        {
            "year": _header_year_value(match.group(1)),
            "start": match.start(),
        }
        for match in year_matches
    ]
    if not years:
        return []

    repaired_columns = _ocr_scrambled_hierarchical_columns(years, sub_matches)
    if repaired_columns:
        return repaired_columns

    assigned_years = _assigned_hierarchical_years(years, sub_matches)
    columns: list[dict[str, Any]] = []
    last_period_column_by_year: dict[int, dict[str, Any]] = {}
    for match, assigned_year in zip(sub_matches, assigned_years, strict=True):
        label = match.group(0).strip()
        scaled_start = int(match.start() * line_width / sub_width)
        year = assigned_year or _nearest_header_year(years, scaled_start)
        if year is None:
            continue

        comparison_type = None
        if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label):
            comparison_type = _comparison_column_type(label)
        if comparison_type:
            target = last_period_column_by_year.get(year)
            column: dict[str, Any] = {
                "label": label,
                "comparison_type": comparison_type,
                "parent_year": year,
            }
            if target:
                column.update(
                    {
                        "comparison_target_label": target.get("label"),
                        "comparison_target_period": target.get("period"),
                        "comparison_target_period_year": target.get("period_year"),
                        "comparison_target_period_quarter": target.get("period_quarter"),
                        "comparison_target_period_type": target.get("period_type"),
                    }
                )
            columns.append(column)
            continue

        period, period_year, period_quarter, period_type = _normalize_subperiod(label, year)
        if not period:
            continue
        column = {
            "label": f"{year}년 {label}",
            "period": period,
            "period_year": period_year,
            "period_quarter": period_quarter,
            "period_type": period_type,
            "parent_year": year,
        }
        columns.append(column)
        last_period_column_by_year[year] = column

    return columns


def _looks_like_parent_year_header_only(
    line: str,
    next_line: str | None,
    period_columns: list[dict[str, Any]],
) -> bool:
    compact_line = re.sub(r"\s+", "", line or "")
    if not re.fullmatch(r"(?:20\d{2}년?){2,4}", compact_line):
        return False
    next_compact = re.sub(r"\s+", "", next_line or "").lower()
    if "구분" not in next_compact:
        return False
    if len(period_columns) == len(re.findall(r"20\d{2}", line or "")):
        return True
    return bool(
        "yoy" in next_compact
        or re.search(r"\b[1-4]\s*q\b", next_line or "", flags=re.IGNORECASE)
        or len(re.findall(r"20\d{2}", next_line or "")) >= 2
    )


def _has_future_quarter_column(
    period_columns: list[dict[str, Any]],
    report_period: str | None,
) -> bool:
    report_year, report_quarter, report_period_type = _period_parts(report_period)
    if report_period_type != "quarter" or not report_year or not report_quarter:
        return False

    for column in period_columns:
        period_year = column.get("period_year")
        period_quarter = column.get("period_quarter")
        period_type = column.get("period_type")
        if period_type != "quarter":
            continue
        if not isinstance(period_year, int) or not isinstance(period_quarter, int):
            continue
        if period_year > report_year:
            return True
        if period_year == report_year and period_quarter > report_quarter:
            return True
    return False


def _posco_dx_appendix_statement_columns(
    prev_line: str | None,
    line: str,
    next_line: str | None,
) -> list[dict[str, Any]]:
    """Repair POSCO DX appendix headers split across three extracted lines.

    The 1Q26 appendix income statement is extracted as:
      `2025 2026`
      `구 분 2023 2024 YoY`
      `1Q 2Q 3Q 4Q 1Q`

    The intended columns are:
      2023 annual, 2024 annual, 2025 Q1~Q4, 2025 annual, 2026 Q1, YoY.
    Without this repair, the generic hierarchical parser maps the same values
    to impossible periods such as 2026Q2/2026Q3/2026Q4.
    """
    prev_years = [
        _header_year_value(match.group(1))
        for match in re.finditer(r"\b(20\d{2})\b", prev_line or "")
    ]
    base_years = [
        _header_year_value(match.group(1)) for match in re.finditer(r"\b(20\d{2})\b", line or "")
    ]
    quarter_labels = [
        re.sub(r"\s+", "", match.group(0)).upper()
        for match in re.finditer(r"\b[1-4]\s*Q\b", next_line or "", flags=re.IGNORECASE)
    ]
    if len(quarter_labels) not in {5, 8}:
        # Some OCR/text extraction paths collapse the sub-period row into the
        # base header: `구 분 2023 2024 YoY 1Q 2Q 3Q 4Q 1Q`.
        quarter_labels = [
            re.sub(r"\s+", "", match.group(0)).upper()
            for match in re.finditer(r"\b[1-4]\s*Q\b", line or "", flags=re.IGNORECASE)
        ]

    prev_year_values: list[int] = [year for year in prev_years if isinstance(year, int)]
    base_year_values: list[int] = [year for year in base_years if isinstance(year, int)]
    if len(prev_year_values) != 2:
        return []

    first_detail_year, second_detail_year = prev_year_values
    if second_detail_year != first_detail_year + 1:
        return []
    if not re.search(r"\bYoY\b|전년", line or "", flags=re.IGNORECASE):
        return []

    if len(base_year_values) == 1 and len(quarter_labels) == 8:
        return _posco_dx_annual_appendix_statement_columns(
            base_year=base_year_values[0],
            first_detail_year=first_detail_year,
            second_detail_year=second_detail_year,
            quarter_labels=quarter_labels,
        )

    if len(base_year_values) != 2 or len(quarter_labels) != 5:
        return []
    if base_year_values != [first_detail_year - 2, first_detail_year - 1]:
        return []

    columns: list[dict[str, Any]] = []
    for year in base_year_values:
        columns.append(
            {
                "label": f"{year}년 연간",
                "period": str(year),
                "period_year": year,
                "period_quarter": None,
                "period_type": "year",
                "parent_year": year,
            }
        )

    detail_year_quarters: list[dict[str, Any]] = []
    for label in quarter_labels[:4]:
        period, period_year, period_quarter, period_type = _normalize_subperiod(
            label,
            first_detail_year,
        )
        if not period:
            return []
        column = {
            "label": f"{first_detail_year}년 {label}",
            "period": period,
            "period_year": period_year,
            "period_quarter": period_quarter,
            "period_type": period_type,
            "parent_year": first_detail_year,
        }
        columns.append(column)
        detail_year_quarters.append(column)

    columns.append(
        {
            "label": f"{first_detail_year}년 연간",
            "period": str(first_detail_year),
            "period_year": first_detail_year,
            "period_quarter": None,
            "period_type": "year",
            "parent_year": first_detail_year,
        }
    )

    period, period_year, period_quarter, period_type = _normalize_subperiod(
        quarter_labels[4],
        second_detail_year,
    )
    if not period:
        return []
    current_column = {
        "label": f"{second_detail_year}년 {quarter_labels[4]}",
        "period": period,
        "period_year": period_year,
        "period_quarter": period_quarter,
        "period_type": period_type,
        "parent_year": second_detail_year,
    }
    columns.append(current_column)
    columns.append(
        {
            "label": "YoY",
            "comparison_type": "yoy",
            "parent_year": second_detail_year,
            "comparison_target_label": current_column.get("label"),
            "comparison_target_period": current_column.get("period"),
            "comparison_target_period_year": current_column.get("period_year"),
            "comparison_target_period_quarter": current_column.get("period_quarter"),
            "comparison_target_period_type": current_column.get("period_type"),
        }
    )
    return columns


def _posco_dx_annual_appendix_statement_columns(
    *,
    base_year: int,
    first_detail_year: int,
    second_detail_year: int,
    quarter_labels: list[str],
) -> list[dict[str, Any]]:
    """Repair POSCO DX annual appendix statement headers.

    The 2025 annual IR appendix income statement is extracted as:
      `2024 2025`
      `구 분 2023 YoY`
      `1Q 2Q 3Q 4Q 1Q 2Q 3Q 4Q`

    The intended columns are:
      2023 annual,
      2024 Q1~Q4, 2024 annual,
      2025 Q1~Q4, 2025 annual,
      YoY for the 2025 annual column.
    """
    if base_year != first_detail_year - 1:
        return []
    if len(quarter_labels) != 8:
        return []

    columns: list[dict[str, Any]] = [
        {
            "label": f"{base_year}년 연간",
            "period": str(base_year),
            "period_year": base_year,
            "period_quarter": None,
            "period_type": "year",
            "parent_year": base_year,
        }
    ]

    for year, labels in (
        (first_detail_year, quarter_labels[:4]),
        (second_detail_year, quarter_labels[4:]),
    ):
        for label in labels:
            period, period_year, period_quarter, period_type = _normalize_subperiod(label, year)
            if not period:
                return []
            columns.append(
                {
                    "label": f"{year}년 {label}",
                    "period": period,
                    "period_year": period_year,
                    "period_quarter": period_quarter,
                    "period_type": period_type,
                    "parent_year": year,
                }
            )
        columns.append(
            {
                "label": f"{year}년 연간",
                "period": str(year),
                "period_year": year,
                "period_quarter": None,
                "period_type": "year",
                "parent_year": year,
            }
        )

    current_column = columns[-1]
    columns.append(
        {
            "label": "YoY",
            "comparison_type": "yoy",
            "parent_year": second_detail_year,
            "comparison_target_label": current_column.get("label"),
            "comparison_target_period": current_column.get("period"),
            "comparison_target_period_year": current_column.get("period_year"),
            "comparison_target_period_quarter": current_column.get("period_quarter"),
            "comparison_target_period_type": current_column.get("period_type"),
        }
    )
    return columns


def _posco_dx_compact_statement_columns(
    prev_line: str | None,
    line: str,
    next_line: str | None,
    *,
    report_period: str | None,
) -> list[dict[str, Any]]:
    """Repair POSCO DX compact quarterly/appendix tables.

    Several POSCO DX IR PDFs extract the table header as three separate lines:
      `2023 2024 2025`
      `구 분 2022`
      `1Q 1Q 4Q 1Q QoQ YoY`

    The annual columns are visually present in the PDF but often missing from
    the OCR subheader line. This helper reconstructs those hidden annual
    columns so the current quarter values are not shifted into historical
    annual periods.
    """
    if "구" not in (line or "") or "분" not in (line or ""):
        return []

    report_year, report_quarter, report_period_type = _period_parts(report_period)
    if report_period_type not in {"quarter", "year"} or not report_year:
        return []

    header_years = [
        year
        for year in (
            _header_year_value(match.group(1))
            for match in re.finditer(r"\b(20\d{2}|\d{2})\s*년?\b", prev_line or "")
        )
        if isinstance(year, int)
    ]
    base_years = [
        year
        for year in (
            _header_year_value(match.group(1))
            for match in re.finditer(r"\b(20\d{2}|\d{2})\s*년?\b", line or "")
        )
        if isinstance(year, int)
    ]
    if not header_years:
        return []

    label_matches = list(
        re.finditer(
            r"\b[1-4]\s*Q\b|\b[12]\s*H\b|연간|\bQoQ\b|\bYoY\b",
            next_line or "",
            flags=re.IGNORECASE,
        )
    )
    if not label_matches:
        return []

    labels = [re.sub(r"\s+", "", match.group(0)).upper() for match in label_matches]
    period_labels = [label for label in labels if not _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)]
    comparison_labels = [label for label in labels if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)]
    if len(header_years) < 2 or not period_labels:
        return []

    if len(base_years) == 1 and len(header_years) == 3 and len(period_labels) == 4:
        return _posco_dx_q1_with_hidden_annual_columns(
            base_year=base_years[0],
            header_years=header_years,
            quarter_labels=period_labels,
            comparison_labels=comparison_labels,
        )

    if (
        not base_years
        and len(header_years) == 3
        and len(period_labels) == 4
        and report_quarter == 1
    ):
        return _posco_dx_q1_with_hidden_annual_columns(
            base_year=None,
            header_years=header_years,
            quarter_labels=period_labels,
            comparison_labels=comparison_labels,
        )

    if not base_years and len(header_years) == 3 and len(period_labels) == 4:
        current_year = header_years[-1]
        columns = [
            _period_column(header_years[0], period_labels[0]),
            _period_column(header_years[1], period_labels[1]),
            _period_column(current_year, period_labels[2]),
            _period_column(current_year, period_labels[3]),
        ]
        return _append_comparison_columns(columns, comparison_labels)

    if not base_years and "연간" in period_labels:
        columns = _posco_dx_annual_interleaved_columns(
            header_years=header_years,
            period_labels=period_labels,
            comparison_labels=comparison_labels,
        )
        if columns:
            return columns

    if not base_years and len(header_years) == 3 and any("H" in label for label in labels):
        columns = _posco_dx_half_year_columns(
            header_years=header_years,
            labels=labels,
        )
        if columns:
            return columns

    if not base_years and len(header_years) == 3:
        columns = _posco_dx_quarter_history_columns(
            header_years=header_years,
            labels=labels,
        )
        if columns:
            return columns

    return []


def _posco_dx_q1_with_hidden_annual_columns(
    *,
    base_year: int | None,
    header_years: list[int],
    quarter_labels: list[str],
    comparison_labels: list[str],
) -> list[dict[str, Any]]:
    if len(header_years) != 3 or len(quarter_labels) != 4:
        return []

    columns: list[dict[str, Any]] = []
    if base_year is not None:
        columns.append(_annual_column(base_year))
        detail_years = header_years
    else:
        first_year, second_year, current_year = header_years
        columns.extend(
            [
                _period_column(first_year, quarter_labels[0]),
                _annual_column(first_year),
            ]
        )
        detail_years = [second_year, current_year]
        quarter_labels = quarter_labels[1:]

    if len(detail_years) == 3:
        first_year, second_year, current_year = detail_years
        columns.extend(
            [
                _period_column(first_year, quarter_labels[0]),
                _annual_column(first_year),
                _period_column(second_year, quarter_labels[1]),
                _period_column(second_year, quarter_labels[2]),
                _annual_column(second_year),
                _period_column(current_year, quarter_labels[3]),
            ]
        )
    elif len(detail_years) == 2:
        previous_year, current_year = detail_years
        columns.extend(
            [
                _period_column(previous_year, quarter_labels[0]),
                _period_column(previous_year, quarter_labels[1]),
                _annual_column(previous_year),
                _period_column(current_year, quarter_labels[2]),
            ]
        )
    else:
        return []

    return _append_comparison_columns(columns, comparison_labels)


def _posco_dx_half_year_columns(
    *,
    header_years: list[int],
    labels: list[str],
) -> list[dict[str, Any]]:
    first_comparison = _first_comparison_index(labels)
    if first_comparison is None:
        return []

    before = labels[:first_comparison]
    after_comparisons = labels[first_comparison:]
    comparison_labels = [
        label for label in after_comparisons if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)
    ]
    trailing_period_labels = [
        label for label in after_comparisons if not _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)
    ]
    if len(header_years) != 3 or len(before) < 8:
        return []

    columns: list[dict[str, Any]] = []
    chunks = [before[:3], before[3:6], before[6:]]
    for year, chunk in zip(header_years, chunks, strict=True):
        for label in chunk:
            columns.append(_period_column(year, label))
    columns = _append_comparison_columns(columns, comparison_labels)
    current_year = header_years[-1]
    for label in trailing_period_labels:
        columns.append(_period_column(current_year, label))
    return columns


def _posco_dx_annual_interleaved_columns(
    *,
    header_years: list[int],
    period_labels: list[str],
    comparison_labels: list[str],
) -> list[dict[str, Any]]:
    if len(header_years) < 2:
        return []

    columns: list[dict[str, Any]] = []
    year_index = 0
    for label in period_labels:
        if year_index >= len(header_years):
            return []
        year = header_years[year_index]
        columns.append(_period_column(year, label))
        if label == "연간" and year_index < len(header_years) - 1:
            year_index += 1

    return _append_comparison_columns(columns, comparison_labels)


def _posco_dx_quarter_history_columns(
    *,
    header_years: list[int],
    labels: list[str],
) -> list[dict[str, Any]]:
    first_comparison = _first_comparison_index(labels)
    if first_comparison is None:
        period_labels = labels
        comparison_labels: list[str] = []
    else:
        period_labels = labels[:first_comparison]
        comparison_labels = labels[first_comparison:]
    if len(header_years) != 3 or len(period_labels) < 7:
        return []

    columns: list[dict[str, Any]] = []
    year_index = 0
    quarter_count_for_year = 0
    for label in period_labels:
        if year_index >= len(header_years):
            return []
        year = header_years[year_index]
        columns.append(_period_column(year, label))
        quarter_count_for_year += 1
        if quarter_count_for_year == 4 and year_index < len(header_years) - 1:
            columns.append(_annual_column(year))
            year_index += 1
            quarter_count_for_year = 0

    if quarter_count_for_year == 4 and year_index == len(header_years) - 1:
        columns.append(_annual_column(header_years[year_index]))

    return _append_comparison_columns(columns, comparison_labels)


def _first_comparison_index(labels: list[str]) -> int | None:
    for index, label in enumerate(labels):
        if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label):
            return index
    return None


def _annual_column(year: int) -> dict[str, Any]:
    return {
        "label": f"{year}년 연간",
        "period": str(year),
        "period_year": year,
        "period_quarter": None,
        "period_type": "year",
        "parent_year": year,
    }


def _period_column(year: int, label: str) -> dict[str, Any]:
    period, period_year, period_quarter, period_type = _normalize_subperiod(label, year)
    if not period:
        return {}
    return {
        "label": f"{year}년 {label}",
        "period": period,
        "period_year": period_year,
        "period_quarter": period_quarter,
        "period_type": period_type,
        "parent_year": year,
    }


def _append_comparison_columns(
    columns: list[dict[str, Any]],
    comparison_labels: list[str],
) -> list[dict[str, Any]]:
    columns = [column for column in columns if column.get("period")]
    target = next(
        (column for column in reversed(columns) if column.get("period_type") == "quarter"),
        columns[-1] if columns else None,
    )
    for label in comparison_labels:
        comparison_column: dict[str, Any] = {
            "label": label,
            "comparison_type": _comparison_column_type(label),
        }
        if target:
            comparison_column.update(
                {
                    "comparison_target_label": target.get("label"),
                    "comparison_target_period": target.get("period"),
                    "comparison_target_period_year": target.get("period_year"),
                    "comparison_target_period_quarter": target.get("period_quarter"),
                    "comparison_target_period_type": target.get("period_type"),
                }
            )
        columns.append(comparison_column)
    return columns


def _ocr_scrambled_same_line_columns(line: str) -> list[dict[str, Any]]:
    year_matches = list(re.finditer(r"\b(20\d{2})\b", line or ""))
    if len(year_matches) != 2:
        return []
    subpattern = re.compile(
        r"연간|\b[1-4]\s*Q\b|\b[12]\s*H\b|\bQoQ\b|\bYoY\b|"
        r"전분기\s*대비|전년\s*(?:동기\s*)?대비",
        re.IGNORECASE,
    )
    sub_matches = list(subpattern.finditer(line or ""))
    if not sub_matches:
        return []
    return _ocr_scrambled_hierarchical_columns(
        [
            {"year": _header_year_value(match.group(1)), "start": match.start()}
            for match in year_matches
        ],
        sub_matches,
    )


def _ocr_scrambled_hierarchical_columns(
    years: list[dict[str, Any]],
    sub_matches: list[re.Match[str]],
) -> list[dict[str, Any]]:
    """Handle OCR headers like `2024 2025 YoY QoQ 2Q 연간 1Q 2Q 1H`.

    In the original table the comparison columns are usually at the far right, but text
    extraction sometimes pulls `YoY/QoQ` before the period subheaders. Reconstruct the
    intended order as previous-year periods, current-year periods, then comparisons.
    """
    if len(years) != 2:
        return []

    first_year = years[0].get("year")
    second_year = years[1].get("year")
    if not isinstance(first_year, int) or not isinstance(second_year, int):
        return []
    if second_year != first_year + 1:
        return []

    labels = [re.sub(r"\s+", "", match.group(0)).lower() for match in sub_matches]
    comparison_labels = [label for label in labels if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)]
    period_labels = [label for label in labels if not _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)]
    if len(comparison_labels) < 1 or len(period_labels) < 3:
        return []
    first_comparison_index = next(
        (
            index
            for index, label in enumerate(labels)
            if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)
        ),
        None,
    )
    first_period_index = next(
        (
            index
            for index, label in enumerate(labels)
            if not _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)
        ),
        None,
    )
    if first_comparison_index is None or first_period_index is None:
        return []
    if first_comparison_index > first_period_index:
        return []

    first_year_count = 1
    for index, label in enumerate(period_labels):
        if label == "연간":
            first_year_count = index + 1
            break
    if first_year_count >= len(period_labels):
        return []

    columns: list[dict[str, Any]] = []
    last_second_year_period: dict[str, Any] | None = None
    for index, label in enumerate(period_labels):
        year = first_year if index < first_year_count else second_year
        period, period_year, period_quarter, period_type = _normalize_subperiod(label, year)
        if not period:
            continue
        column = {
            "label": f"{year}년 {label.upper() if label.endswith('q') else label}",
            "period": period,
            "period_year": period_year,
            "period_quarter": period_quarter,
            "period_type": period_type,
            "parent_year": year,
        }
        columns.append(column)
        if year == second_year and period_type == "quarter":
            last_second_year_period = column

    if not columns:
        return []

    for label in comparison_labels:
        comparison_type = _comparison_column_type(label)
        comparison_column: dict[str, Any] = {
            "label": label.upper(),
            "comparison_type": comparison_type,
            "parent_year": second_year,
        }
        if last_second_year_period:
            comparison_column.update(
                {
                    "comparison_target_label": last_second_year_period.get("label"),
                    "comparison_target_period": last_second_year_period.get("period"),
                    "comparison_target_period_year": last_second_year_period.get("period_year"),
                    "comparison_target_period_quarter": last_second_year_period.get(
                        "period_quarter"
                    ),
                    "comparison_target_period_type": last_second_year_period.get("period_type"),
                }
            )
        columns.append(comparison_column)

    return columns


def _assigned_hierarchical_years(
    years: list[dict[str, Any]],
    sub_matches: list[re.Match[str]],
) -> list[int | None]:
    labels = [re.sub(r"\s+", "", match.group(0)).lower() for match in sub_matches]
    if len(years) == 2:
        first_year = years[0].get("year")
        second_year = years[1].get("year")
        if (
            isinstance(first_year, int)
            and isinstance(second_year, int)
            and second_year == first_year + 1
            and len(labels) >= 4
            and re.fullmatch(r"[1-4]분기|[1-4]q", labels[0], flags=re.IGNORECASE)
            and labels[1] == "연간"
        ):
            return [first_year if index < 2 else second_year for index in range(len(labels))]
        if (
            isinstance(first_year, int)
            and isinstance(second_year, int)
            and second_year == first_year + 1
            and len(labels) >= 2
            and _is_quarter_subheader_label(labels[0])
            and _is_quarter_subheader_label(labels[1])
        ):
            assigned: list[int | None] = []
            active_year = first_year
            period_seen = 0
            for label in labels:
                if _is_quarter_subheader_label(label) or label == "연간":
                    period_seen += 1
                    active_year = first_year if period_seen == 1 else second_year
                assigned.append(active_year)
            return assigned
    return [None for _ in labels]


def _is_quarter_subheader_label(label: str) -> bool:
    return bool(re.fullmatch(r"[1-4]분기|[1-4]q", label, flags=re.IGNORECASE))


def _header_year_value(value: str) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return 2000 + year if year < 100 else year


def _nearest_header_year(years: list[dict[str, Any]], scaled_start: int) -> int | None:
    eligible = [item for item in years if int(item["start"]) <= scaled_start]
    selected = eligible[-1] if eligible else years[0]
    year = selected.get("year")
    return int(year) if isinstance(year, int) else None


def _normalize_subperiod(
    label: str,
    year: int,
) -> tuple[str | None, int | None, int | None, str | None]:
    value = re.sub(r"\s+", "", label or "").lower()
    match = re.search(r"([1-4])분기", value)
    if not match:
        match = re.search(r"([1-4])q", value, flags=re.IGNORECASE)
    if match:
        quarter = int(match.group(1))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    if value == "연간":
        return str(year), year, None, "year"
    match = re.search(r"([12])h", value, flags=re.IGNORECASE)
    if match:
        half = int(match.group(1))
        return f"{year}H{half}", year, None, "half"
    return None, None, None, None


def _dedupe_table_columns(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for column in columns:
        key = (
            column.get("label"),
            column.get("period"),
            column.get("comparison_type"),
            column.get("comparison_target_period"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(column)
    return deduped


def _coerce_quarterly_context_columns(
    columns: list[dict[str, Any]],
    *,
    table_title: str | None,
    report_period: str | None,
) -> list[dict[str, Any]]:
    report_year, report_quarter, report_period_type = _period_parts(report_period)
    if report_period_type != "quarter" or not report_year or not report_quarter:
        return columns
    if not _looks_like_quarterly_table_context(table_title, report_period):
        return columns
    if any(column.get("period_type") == "quarter" for column in columns):
        return columns

    coerced: list[dict[str, Any]] = []
    for column in columns:
        if column.get("period_type") != "year":
            coerced.append(column)
            continue
        period_year = column.get("period_year")
        if not isinstance(period_year, int):
            coerced.append(column)
            continue
        patched = dict(column)
        patched["original_period"] = column.get("period")
        patched["original_period_type"] = column.get("period_type")
        patched["period"] = f"{period_year}Q{report_quarter}"
        patched["period_quarter"] = report_quarter
        patched["period_type"] = "quarter"
        patched["period_inferred_from_quarterly_context"] = True
        coerced.append(patched)
    return coerced


def _looks_like_quarterly_table_context(
    table_title: str | None,
    report_period: str | None,
) -> bool:
    if not table_title:
        return False
    report_year, report_quarter, _period_type = _period_parts(report_period)
    if not report_year or not report_quarter:
        return False

    compact = re.sub(r"\s+", "", table_title or "").lower()
    short_year = str(report_year)[2:]
    quarter_tokens = {
        f"{report_year}년{report_quarter}분기",
        f"{short_year}년{report_quarter}분기",
        f"{report_year}q{report_quarter}",
        f"{short_year}q{report_quarter}",
        f"{report_quarter}q{short_year}",
    }
    if any(token in compact for token in quarter_tokens):
        return True
    return bool("분기별" in compact or "quarterly" in compact)


def _comparison_columns_from_continuation(line: str) -> list[dict[str, Any]]:
    if _IR_TABLE_NUMBER_PATTERN.search(line or ""):
        return []
    return [
        {
            "label": match.group(0).strip(),
            "comparison_type": _comparison_column_type(match.group(0).strip()),
        }
        for match in _IR_TABLE_COMPARISON_PATTERN.finditer(line or "")
    ]


def _period_columns_from_header(line: str) -> list[dict[str, Any]]:
    return [column for column in _table_columns_from_header(line) if column.get("period")]


def _comparison_column_type(label: str) -> str:
    value = re.sub(r"\s+", "", label or "").lower()
    if value == "qoq" or "전분기" in value:
        return "qoq"
    return "yoy"


def _comparison_metric_name(metric_name: str, comparison_type: str) -> str:
    return f"{metric_name}_{comparison_type}"


def _primary_period_column(columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((column for column in columns if column.get("period")), None)


def _report_period_column(report_period: str | None) -> dict[str, Any] | None:
    year, quarter, period_type = _period_parts(report_period)
    if not report_period or not year:
        return None
    return {
        "label": report_period,
        "period": report_period,
        "period_year": year,
        "period_quarter": quarter,
        "period_type": period_type,
    }


def _comparison_target_column(column: dict[str, Any]) -> dict[str, Any] | None:
    period = column.get("comparison_target_period")
    if not period:
        return None
    return {
        "label": column.get("comparison_target_label") or period,
        "period": period,
        "period_year": column.get("comparison_target_period_year"),
        "period_quarter": column.get("comparison_target_period_quarter"),
        "period_type": column.get("comparison_target_period_type"),
    }


def _comparison_base_cell(
    *,
    comparison_type: str,
    columns: list[dict[str, Any]],
    row_values: list[str],
    current_period: str | None,
    unit: str | None,
) -> dict[str, Any]:
    period_cells = [
        (column, value)
        for column, value in zip(columns, row_values, strict=False)
        if column.get("period")
    ]
    if not period_cells:
        return {}

    current_index = next(
        (
            index
            for index, (column, _value) in enumerate(period_cells)
            if column.get("period") == current_period
        ),
        len(period_cells) - 1,
    )

    base_index: int | None = None
    if comparison_type == "qoq":
        base_index = current_index - 1 if current_index > 0 else None
    elif comparison_type == "yoy":
        current_column = period_cells[current_index][0]
        current_year = current_column.get("period_year")
        current_quarter = current_column.get("period_quarter")
        for index, (column, _value) in enumerate(period_cells):
            if (
                isinstance(current_year, int)
                and column.get("period_year") == current_year - 1
                and column.get("period_quarter") == current_quarter
            ):
                base_index = index
                break
        if base_index is None and current_index > 0:
            base_index = current_index - 1

    if base_index is None or base_index < 0 or base_index >= len(period_cells):
        return {}

    base_column, base_raw_value = period_cells[base_index]
    base_value = _table_value(base_raw_value, unit=unit, value_kind="amount_krwbn")
    if base_value is None:
        base_value = _table_value(base_raw_value, unit="%", value_kind="percentage")
    return {
        "comparison_base_period": base_column.get("period"),
        "comparison_base_period_year": base_column.get("period_year"),
        "comparison_base_period_quarter": base_column.get("period_quarter"),
        "comparison_base_period_type": base_column.get("period_type"),
        "comparison_base_column_label": base_column.get("label"),
        "comparison_base_raw_value": base_raw_value,
        "comparison_base_value": base_value,
    }


def _table_row_evidence(
    *,
    table_title: str | None,
    context_business_area: str | None,
    metric_parent_label: str | None,
    row_label: str,
    columns: list[dict[str, Any]],
    row_values: list[str],
    unit: str,
    unit_evidence: str | None,
) -> str:
    path = [
        part
        for part in (
            table_title,
            context_business_area,
            metric_parent_label,
            row_label,
        )
        if part
    ]
    series = [
        f"{column['label']} {value}" for column, value in zip(columns, row_values, strict=False)
    ]
    prefix = " > ".join(path) if path else row_label
    unit_part = f"단위: {unit}"
    if unit_evidence:
        unit_part = f"{unit_part} | 단위 근거: {unit_evidence}"
    else:
        unit_part = f"{unit_part} | 단위 근거 없음"
    return f"{prefix} | {' | '.join(series)} ({unit_part})"


def _effective_row_unit_evidence(
    *,
    unit_evidence: str | None,
    row_label: str,
    metric_parent_label: str | None,
    value_kind: str,
) -> str | None:
    if value_kind == "percentage":
        if "%" in (row_label or ""):
            return row_label
        if metric_parent_label and "%" in metric_parent_label:
            return metric_parent_label
    return unit_evidence


def _table_classification_reason(
    *,
    business_area: str | None,
    metric_scope: str,
    row_label: str,
    table_title: str | None,
    context_business_area: str | None,
    metric_parent_label: str | None,
    explicit_metric: bool,
) -> str:
    table_part = f"표 제목 '{table_title}'" if table_title else "표 제목 없음"
    if metric_scope == "company_total":
        if context_business_area:
            return (
                f"{table_part}, 상위 문맥 '{context_business_area}' 아래의 전체/합계성 행 "
                f"'{row_label}'로 판단되어 company_total로 분류"
            )
        if _is_company_total_table_label(row_label):
            return (
                f"{table_part}, 행 라벨 '{row_label}'이 전체/합계성 라벨이라 company_total로 분류"
            )
        if explicit_metric:
            return (
                f"{table_part}, 행 라벨 '{row_label}'이 별도 사업부문명이 아닌 metric 라벨이라 "
                "company_total로 분류"
            )
        return f"{table_part}, 사업부문 라벨이 확인되지 않아 company_total로 분류"

    parent_part = f", 상위 metric '{metric_parent_label}'" if metric_parent_label else ""
    context_part = f", 표 문맥 '{context_business_area}'" if context_business_area else ""
    return (
        f"{table_part}{context_part}{parent_part}, 행 라벨 '{row_label}'을 "
        f"사업부문/서비스 라벨로 판단해 business_area='{business_area}' segment로 분류"
    )


def _normalize_table_period(label: str) -> tuple[str | None, int | None, int | None, str | None]:
    value = re.sub(r"\s+", " ", label or "").strip()
    match = re.search(r"(\d{2})\s*년\s*([1-4])\s*분기", value)
    if match:
        year = 2000 + int(match.group(1))
        quarter = int(match.group(2))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    match = re.search(r"(20\d{2})\s*년\s*([1-4])\s*분기", value)
    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    match = re.search(r"(20\d{2})\s*Q\s*([1-4])", value, flags=re.IGNORECASE)
    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    match = re.search(r"([1-4])\s*Q\s*['’]?\s*(\d{2})", value, flags=re.IGNORECASE)
    if match:
        quarter = int(match.group(1))
        year = 2000 + int(match.group(2))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    match = re.search(r"(20\d{2})\s*년\s*(상반기|하반기)", value)
    if match:
        year = int(match.group(1))
        half = 1 if match.group(2) == "상반기" else 2
        return f"{year}H{half}", year, None, "half"
    match = re.search(r"(\d{2})\s*년\s*(상반기|하반기)", value)
    if match:
        year = 2000 + int(match.group(1))
        half = 1 if match.group(2) == "상반기" else 2
        return f"{year}H{half}", year, None, "half"
    match = re.search(r"(\d{2})\s*년$", value)
    if match:
        year = 2000 + int(match.group(1))
        return str(year), year, None, "year"
    match = re.search(r"(20\d{2})\s*년$", value)
    if match:
        year = int(match.group(1))
        return str(year), year, None, "year"
    match = re.search(r"\b(20\d{2})\b", value)
    if match:
        year = int(match.group(1))
        return str(year), year, None, "year"
    return None, None, None, None


def _table_metric_from_label(line: str) -> tuple[str, str] | None:
    label = re.split(r"\(?-?\d", line or "", maxsplit=1)[0]
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label).lower()
    if _looks_like_table_noise_label(label):
        return None
    if _looks_like_non_target_metric_label(label):
        return None
    for metric_name, aliases, value_kind in _IR_TABLE_METRIC_ALIASES:
        if any(_metric_alias_matches_label(alias, compact) for alias in aliases):
            return metric_name, value_kind
    return None


def _looks_like_non_target_metric_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "").lower()
    if not compact:
        return False
    if "ebitda" in compact and ("마진" in compact or "margin" in compact):
        return True
    if "영업이익률" in compact or "operatingmargin" in compact or compact in {"opm", "margin"}:
        return False
    if compact in {"총이익", "총이익률"}:
        return True
    return bool(_IR_TABLE_NON_METRIC_LABEL_PATTERN.search(label or ""))


def _metric_alias_matches_label(alias: str, compact_label: str) -> bool:
    compact_alias = alias.replace(" ", "").lower()
    if not compact_alias:
        return False
    if compact_alias in {"매출", "sales", "op"}:
        return compact_label == compact_alias
    return compact_alias in compact_label


def _is_percentage_metric_label(row_label: str, metric_name: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", row_label or "").lower()
    for candidate_metric, aliases, value_kind in _IR_TABLE_METRIC_ALIASES:
        if candidate_metric != metric_name or value_kind != "percentage":
            continue
        if any(_metric_alias_matches_label(alias, compact) for alias in aliases):
            return True
    return False


def _is_valid_table_metric_row(row_label: str, row_metric: tuple[str, str]) -> bool:
    cleaned_label = re.sub(r"\s+", " ", row_label or "").strip()
    if len(cleaned_label) > 60:
        return False
    if re.search(r"[.!?。]|다$", cleaned_label):
        return False
    if _looks_like_non_target_metric_label(row_label):
        return False

    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", row_label or "").lower()
    if not compact:
        return True
    if _looks_like_table_noise_label(row_label):
        return False

    metric_name, _value_kind = row_metric
    metric_aliases = {
        alias.replace(" ", "").lower()
        for candidate_metric, aliases, _candidate_kind in _IR_TABLE_METRIC_ALIASES
        if candidate_metric == metric_name
        for alias in aliases
    }
    label_without_metric = compact
    for alias in metric_aliases:
        label_without_metric = label_without_metric.replace(alias, "")

    if not label_without_metric:
        return True
    if _is_company_total_table_label(label_without_metric):
        return True
    if len(label_without_metric) > 28 and not _detect_business_area(label_without_metric):
        return False
    return True


def _looks_like_segment_value_row(line: str, expected_count: int) -> bool:
    row_label = _table_row_label(line)
    if _looks_like_non_target_metric_label(row_label) or _looks_like_table_noise_label(row_label):
        return False
    if not _looks_like_business_area_label(row_label):
        return False
    values = _table_row_values(line, expected_count)
    return len(values) >= expected_count


def _looks_like_table_noise_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "")
    if not compact:
        return False
    if re.fullmatch(r"[-+△▲▵▴▽▼()]+", compact):
        return True
    return bool(_IR_TABLE_NOISE_LABEL_PATTERN.search(label or ""))


def _table_row_values(line: str, expected_count: int) -> list[str]:
    values = _IR_TABLE_NUMBER_PATTERN.findall(line or "")
    clean_values = [
        value
        for value in values
        if not re.fullmatch(r"20\d{2}", value.strip())
        and not _looks_like_table_footnote_marker(value)
    ]
    return clean_values[:expected_count]


def _row_has_misaligned_leading_comparison_columns(
    columns: list[dict[str, Any]],
    row_values: list[str],
) -> bool:
    if not columns or not row_values:
        return False
    leading_comparison_count = 0
    for column in columns:
        if column.get("comparison_type"):
            leading_comparison_count += 1
            continue
        break
    if leading_comparison_count == 0:
        return False
    if any(
        not column.get("comparison_target_period") for column in columns[:leading_comparison_count]
    ):
        return True
    leading_values = row_values[:leading_comparison_count]
    return any("%" not in str(value) for value in leading_values)


def _row_values_are_percentage(values: list[str]) -> bool:
    return bool(values) and all(str(value).strip().endswith("%") for value in values)


def _row_values_have_percentage(values: list[str]) -> bool:
    return any(str(value).strip().endswith("%") for value in values)


def _looks_like_table_footnote_marker(value: str) -> bool:
    stripped = value.strip()
    return bool(re.fullmatch(r"\d+\)", stripped))


def _table_row_label(line: str) -> str:
    label = re.split(r"\(?-?\d", line or "", maxsplit=1)[0]
    return re.sub(r"\s+", " ", label).strip(" :-|'\"`‘’“”.,;")


def _table_value(raw_value: str, *, unit: str | None, value_kind: str) -> float | None:
    value = raw_value.strip()
    if not value:
        return None
    sign = -1.0 if value.startswith(("△", "▲")) else 1.0
    value = value.lstrip("△▲")
    has_pct_marker = value.endswith("%")
    if value_kind != "percentage" and has_pct_marker:
        return None
    if value_kind == "percentage" and unit != "%" and not has_pct_marker:
        return None
    if value_kind == "percentage" or has_pct_marker:
        try:
            return sign * float(value.rstrip("%").replace(",", ""))
        except ValueError:
            return None
    if not unit:
        return None
    return sign * _normalize_table_amount_krwbn(value.strip("()"), unit)


def _table_display_unit(*, unit: str | None, value_kind: str) -> str:
    if value_kind == "percentage":
        return "%"
    return unit or "단위 미확인"


def _table_cell_unit(raw_value: str, *, table_unit: str | None, value_kind: str) -> str:
    if value_kind == "percentage" or raw_value.strip().endswith("%"):
        return "%"
    return table_unit or "단위 미확인"


def _normalize_table_amount_krwbn(value: str, unit: str) -> float:
    number = float(value.replace(",", ""))
    if unit in {"조원", "조"}:
        return number * 10_000
    if unit in {"십억원", "KRW bn", "krw bn"}:
        return number * 10
    if unit == "백만원":
        return number / 100
    return number


def _table_unit(text: str) -> str | None:
    value = text or ""
    lowered = value.lower()
    compact = re.sub(r"\s+", "", lowered)
    if (
        "십억원" in compact
        or "십억krw" in compact
        or "krwbn" in compact
        or "krwbillion" in compact
        or "billionkrw" in compact
    ):
        return "십억원"
    if "백만원" in compact or "krwmn" in compact or "krwmillion" in compact:
        return "백만원"
    if "조원" in compact:
        return "조원"
    if "억원" in compact:
        return "억원"
    return None


def _table_unit_evidence(lines: list[str]) -> str | None:
    for line in lines:
        compact = re.sub(r"\s+", "", line.lower())
        if any(
            token in compact
            for token in (
                "단위:",
                "unit:",
                "unitof",
                "억원",
                "조원",
                "백만원",
                "십억원",
                "krwbn",
                "krwmn",
                "krwbillion",
                "krwmillion",
                "billionkrw",
            )
        ):
            return line[:300]
    return None


def _nearby_table_title(lines: list[str], header_index: int) -> str | None:
    for line in reversed(lines[max(0, header_index - 3) : header_index]):
        if _looks_like_unit_line(line):
            continue
        if len(_period_columns_from_header(line)) < 2:
            return line
    return None


def _should_skip_financial_table(
    *,
    peer_id: str | None,
    table_title: str | None,
    lines: list[str],
    header_index: int,
) -> bool:
    if peer_id != "sk_ax":
        return False
    context = " ".join(lines[max(0, header_index - 16) : header_index + 1])
    text = f"{table_title or ''} {context}".lower()
    compact = re.sub(r"\s+", "", text)
    if any(
        token in compact
        for token in (
            "skax",
            "skc&c",
            "sk씨앤씨",
            "it서비스부문",
            "it서비스",
        )
    ):
        return False
    return any(term.lower() in text for term in _IR_PORTFOLIO_TERMS if term != "appendix")


def _looks_like_unit_line(line: str) -> bool:
    compact = re.sub(r"\s+", "", line or "").lower()
    if not compact:
        return False
    return bool(re.fullmatch(r"[\(\[]?단위[:：]?(?:억원|십억원|백만원|조원|%)?[\)\]]?", compact))


def _table_context_business_area(lines: list[str], header_index: int) -> str | None:
    context_lines = lines[max(0, header_index - 6) : header_index]
    for line in reversed(context_lines):
        candidate = _business_area_label_from_context(line)
        if candidate:
            return candidate
    return None


def _business_area_label_from_context(text: str) -> str | None:
    cleaned = _clean_table_context_label(text)
    if not cleaned:
        return None
    if len(cleaned) > 50:
        return None
    if _looks_like_table_noise_label(cleaned):
        return None
    if re.search(r"[.!?。]|다$", cleaned):
        return None
    if _looks_like_financial_table_title(cleaned):
        return None
    detected = _detect_business_area(cleaned)
    if detected:
        return cleaned if _looks_like_business_area_label(cleaned) else detected
    return None


def _clean_table_context_label(text: str) -> str:
    cleaned = re.sub(
        r"\([^)]*(?:단위|unit|krw|억원|십억원|백만원|%).*?\)", "", text or "", flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"\b(?:revenue|sales|operating profit|op|margin|results?|financial)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"매출|영업이익|영업이익률|실적|손익|요약|현황|단위|억원|십억원|백만원", "", cleaned
    )
    return re.sub(r"\s+", " ", cleaned).strip(" :-|")


def _looks_like_financial_table_title(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    return any(
        token in compact
        for token in (
            "financialresults",
            "income",
            "손익",
            "경영실적",
            "실적요약",
            "재무",
            "매출",
            "영업이익",
            "margin",
        )
    )


def _is_historical_period(period: str | None, report_period: str | None) -> bool:
    if not period or not report_period:
        return False
    return period != report_period


def _table_business_area(
    row_label: str,
    page_text: str,
    *,
    peer_id: str | None,
    context_business_area: str | None = None,
    explicit_metric: bool = True,
    active_metric_parent_label: str | None = None,
) -> str | None:
    metric_removed = row_label
    for _metric_name, aliases, _value_kind in _IR_TABLE_METRIC_ALIASES:
        for alias in aliases:
            metric_removed = re.sub(re.escape(alias), "", metric_removed, flags=re.IGNORECASE)
    metric_removed = re.sub(r"\s+", " ", metric_removed).strip(" :-|")
    if _is_non_business_area_label(metric_removed):
        return context_business_area or "company_total"
    if _is_punctuation_only_label(metric_removed):
        return context_business_area or "company_total"
    if not explicit_metric and _looks_like_business_area_label(row_label):
        return row_label
    if _is_company_total_table_label(metric_removed):
        return context_business_area or "company_total"
    if active_metric_parent_label and metric_removed == active_metric_parent_label:
        return context_business_area or "company_total"
    if metric_removed and _detect_business_area(metric_removed):
        return metric_removed
    if metric_removed and _looks_like_business_area_label(metric_removed):
        return metric_removed
    return _detect_business_area(metric_removed) or context_business_area or "company_total"


def _is_company_total_table_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "").lower()
    if not compact or _is_punctuation_only_label(label):
        return True
    return compact in {
        "합계",
        "총계",
        "계",
        "전체",
        "전사",
        "연결",
        "별도",
        "total",
        "subtotal",
        "companytotal",
        "consolidated",
    }


def _looks_like_business_area_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "").lower()
    if not compact or _is_company_total_table_label(label):
        return False
    if _is_non_business_area_label(label):
        return False
    if _is_punctuation_only_label(label):
        return False
    if re.fullmatch(r"[-+%.,\d]+", compact):
        return False
    metric_aliases = {
        alias.replace(" ", "").lower()
        for _metric_name, aliases, _value_kind in _IR_TABLE_METRIC_ALIASES
        for alias in aliases
    }
    return compact not in metric_aliases


def _is_non_business_area_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "").lower()
    return compact in {
        "fy",
        "year",
        "annual",
        "profit",
        "operating",
        "operatingprofit",
        "revenue",
        "sales",
    }


def _is_punctuation_only_label(label: str) -> bool:
    compact = re.sub(r"\s+", "", label or "")
    if not compact:
        return True
    return not bool(re.search(r"[A-Za-z가-힣]", compact))


def _company_section_rules(peer_id: str | None) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return _IR_COMPANY_SECTION_HINTS.get(peer_id or "", ()) + _IR_SECTION_RULES


def _classify_page_section(
    text: str,
    *,
    peer_id: str | None,
) -> tuple[str, str, list[str]]:
    normalized = text.lower()
    matches: list[str] = []

    for section_key, section_title, keywords in _company_section_rules(peer_id):
        hits = [keyword for keyword in keywords if keyword.lower() in normalized]
        if hits:
            return section_key, section_title, hits[:8]

    return "other", "Other IR Content", matches


def _match_topics(text: str) -> tuple[list[str], dict[str, list[str]]]:
    lowered = text.lower()
    topic_signals: dict[str, list[str]] = {}

    for topic, keywords in SECTOR_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword.lower() in lowered]
        if hits:
            topic_signals[topic] = hits[:8]

    return sorted(topic_signals), topic_signals


def _split_text_chunks(text: str, max_chars: int = _IR_CHUNK_MAX_CHARS) -> list[str]:
    value = _normalize_ir_chunk_text(text)
    if not value:
        return []
    if len(value) <= max_chars:
        return [value] if _is_informative_ir_chunk(value) else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in _ir_paragraphs(value):
        if len(paragraph) > max_chars:
            if current:
                _append_ir_chunk(chunks, "\n".join(current))
                current = []
                current_len = 0
            for split in _split_long_ir_paragraph(paragraph, max_chars=max_chars):
                _append_ir_chunk(chunks, split)
            continue

        next_len = current_len + len(paragraph) + (1 if current else 0)
        if current and next_len > max_chars:
            _append_ir_chunk(chunks, "\n".join(current))
            current = []
            current_len = 0

        current.append(paragraph)
        current_len += len(paragraph) + (1 if current_len else 0)

        if current_len >= _IR_CHUNK_TARGET_CHARS:
            _append_ir_chunk(chunks, "\n".join(current))
            current = []
            current_len = 0

    if current:
        _append_ir_chunk(chunks, "\n".join(current))

    return chunks


def _normalize_ir_chunk_text(text: str) -> str:
    lines = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", line):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def _ir_paragraphs(text: str) -> list[str]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        return []

    merged: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for paragraph in paragraphs:
        if buffer and buffer_len + len(paragraph) > _IR_CHUNK_TARGET_CHARS:
            merged.append("\n".join(buffer))
            buffer = []
            buffer_len = 0

        buffer.append(paragraph)
        buffer_len += len(paragraph)

    if buffer:
        merged.append("\n".join(buffer))

    return merged


def _split_long_ir_paragraph(text: str, *, max_chars: int) -> list[str]:
    value = text.strip()
    if not value:
        return []
    if len(value) <= max_chars:
        return [value]

    chunks: list[str] = []
    start = 0
    while start < len(value):
        end = min(start + max_chars, len(value))
        boundary = value.rfind("\n", start, end)
        if boundary <= start + max_chars // 2:
            boundary = value.rfind(". ", start, end)
        if boundary <= start:
            boundary = end

        chunk = value[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        if boundary >= len(value):
            break
        start = max(boundary - _IR_CHUNK_OVERLAP_CHARS, start + 1)

    return chunks


def _append_ir_chunk(chunks: list[str], text: str) -> None:
    value = text.strip()
    if _is_informative_ir_chunk(value):
        chunks.append(value)


def _is_informative_ir_chunk(text: str) -> bool:
    value = text.strip()
    if not value:
        return False

    lowered = value.lower()
    has_signal = _has_ir_signal(value)
    has_metric = _has_financial_metric_text(value)

    if any(pattern.search(value) for pattern in _IR_LOW_VALUE_PATTERNS) and not has_metric:
        return False

    if len(value) < _IR_CHUNK_MIN_CHARS and not has_signal and not has_metric:
        return False

    alpha_numeric_count = len(re.findall(r"[0-9A-Za-z가-힣]", value))
    if alpha_numeric_count < 30 and not has_metric:
        return False

    if lowered in {"disclaimer", "contents", "목차"}:
        return False

    return True


def _has_ir_signal(text: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in _IR_SIGNAL_TERMS)


def _has_financial_metric_text(text: str) -> bool:
    return any(
        pattern.search(text) for pattern in (*_REVENUE_PATTERNS, *_OPERATING_PROFIT_PATTERNS)
    )


def _extract_sections_and_chunks(
    pages: list[dict[str, Any]],
    *,
    peer_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, list[str]]]:
    sections_by_key: dict[str, dict[str, Any]] = {}
    document_chunks: list[dict[str, Any]] = []
    all_topic_signals: dict[str, list[str]] = {}

    for page in pages:
        page_no = page.get("page")
        page_text = str(page.get("text", "") or "").strip()
        if not page_text:
            continue

        section_key, section_title, section_signals = _classify_page_section(
            page_text,
            peer_id=peer_id,
        )
        topics, topic_signals = _match_topics(page_text)

        section = sections_by_key.setdefault(
            section_key,
            {
                "section_key": section_key,
                "section_title": section_title,
                "pages": [],
                "text_chars": 0,
                "signals": [],
                "topics": [],
            },
        )
        if page_no not in section["pages"]:
            section["pages"].append(page_no)
        section["text_chars"] += len(page_text)
        section["signals"] = sorted(set(section["signals"]) | set(section_signals))
        section["topics"] = sorted(set(section["topics"]) | set(topics))

        for topic, hits in topic_signals.items():
            merged = set(all_topic_signals.get(topic, []))
            merged.update(hits)
            all_topic_signals[topic] = sorted(merged)[:12]

        for local_idx, chunk_text in enumerate(_split_text_chunks(page_text), start=1):
            chunk_section_key, chunk_section_title, chunk_section_signals = _classify_page_section(
                chunk_text, peer_id=peer_id
            )
            chunk_topics, chunk_topic_signals = _match_topics(chunk_text)
            document_chunks.append(
                {
                    "chunk_id": f"ir-p{page_no or 'x'}-{local_idx}",
                    "page": page_no,
                    "section_key": chunk_section_key,
                    "section_title": chunk_section_title,
                    "section_signals": chunk_section_signals,
                    "chunk_index": len(document_chunks) + 1,
                    "text_chars": len(chunk_text),
                    "text": chunk_text,
                    "topics": chunk_topics,
                    "topic_signals": chunk_topic_signals,
                }
            )

    sections = list(sections_by_key.values())
    for idx, section in enumerate(sections, start=1):
        section["section_order"] = idx

    return sections, document_chunks, sorted(all_topic_signals), all_topic_signals


def _build_page_index(
    pages: list[dict[str, Any]],
    document_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compact page-level index so agents can target pages before reading chunks."""
    chunks_by_page: dict[Any, list[dict[str, Any]]] = {}
    for chunk in document_chunks:
        chunks_by_page.setdefault(chunk.get("page"), []).append(chunk)

    page_index: list[dict[str, Any]] = []
    for index, page in enumerate(pages, start=1):
        page_no = page.get("page") or index
        text = str(page.get("text") or "")
        page_chunks = chunks_by_page.get(page_no, [])
        section_keys = sorted(
            {str(chunk.get("section_key")) for chunk in page_chunks if chunk.get("section_key")}
        )
        topics = sorted(
            {str(topic) for chunk in page_chunks for topic in (chunk.get("topics") or []) if topic}
        )
        page_index.append(
            {
                "page": page_no,
                "text_chars": len(text),
                "chunk_count": len(page_chunks),
                "section_keys": section_keys,
                "topics": topics,
            }
        )
    return page_index


def _build_financial_record(
    *,
    source: str,
    peer_id: str | None,
    period: str | None,
    title: str,
    url: str,
    published_at: str | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_details = {
        str(candidate["type"]): {
            key: value
            for key, value in candidate.items()
            if key
            in {
                "page",
                "raw",
                "value_krwbn",
                "value_pct",
                "value_kind",
                "metric_scope",
                "business_area",
                "entity_name",
                "confidence",
                "source",
                "source_table_uid",
                "table_title",
                "metric_parent_label",
                "row_label",
                "column_label",
                "row_evidence",
                "evidence_text",
                "classification_reason",
                "context_evidence",
                "base_metric_type",
                "comparison_type",
                "unit",
                "unit_evidence",
                "period",
                "period_year",
                "period_quarter",
                "period_type",
                "is_historical",
                "comparison_base_period",
                "comparison_base_period_year",
                "comparison_base_period_quarter",
                "comparison_base_period_type",
                "comparison_base_column_label",
                "comparison_base_raw_value",
                "comparison_base_value",
            }
        }
        for candidate in candidates
        if candidate.get("type")
    }
    accepted_scopes = {"company_total", "segment"}
    revenue_total = _candidate_value(
        candidates,
        "revenue_total",
        "value_krwbn",
        allowed_scopes=accepted_scopes,
    )
    operating_profit = _candidate_value(
        candidates,
        "operating_profit",
        "value_krwbn",
        allowed_scopes=accepted_scopes,
    )
    return {
        "peer_id": peer_id,
        "period": period,
        "revenue_total_krwbn": revenue_total,
        "operating_profit_krwbn": operating_profit,
        "net_income_krwbn": _candidate_value(
            candidates,
            "net_income",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "ebitda_krwbn": _candidate_value(
            candidates,
            "ebitda",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "backlog_krwbn": _candidate_value(
            candidates,
            "backlog",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "orders_krwbn": _candidate_value(
            candidates,
            "orders",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "capex_krwbn": _candidate_value(
            candidates,
            "capex",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "operating_margin_pct": _candidate_value(
            candidates,
            "operating_margin",
            "value_pct",
            allowed_scopes=accepted_scopes,
        ),
        "metric_details": metric_details,
        "source": source,
        "title": title,
        "url": url,
        "published_at": published_at,
        "ir_page": _candidate_page(candidates, "revenue_total")
        or _candidate_page(candidates, "operating_profit"),
    }


def _candidate_value(
    candidates: list[dict[str, Any]],
    metric_type: str,
    value_key: str,
    *,
    allowed_scopes: set[str] | None = None,
) -> float | None:
    matching = [
        candidate
        for candidate in candidates
        if candidate.get("type") == metric_type
        and (allowed_scopes is None or candidate.get("metric_scope") in allowed_scopes)
    ]
    for candidate in sorted(matching, key=lambda item: bool(item.get("is_historical"))):
        value = candidate.get(value_key)
        if isinstance(value, int | float):
            return float(value)
    return None


class IRParser:
    """IR RawArticle에서 핵심 재무 지표 후보를 추출한다."""

    def parse_article(
        self,
        article: Any,
        *,
        include_raw_pages: bool = False,
    ) -> dict[str, Any]:
        """IRCrawler가 만든 RawArticle 또는 dict 결과를 파싱한다."""
        extra = _article_extra(article)
        text = str(_article_get(article, "content", "") or "")
        pages = _pages_from_ir_article(article, extra)
        peer_id = _article_peer_id(article)
        pages = _filter_pages_for_peer(pages, peer_id)
        title = str(_article_get(article, "title", "") or "")
        url = str(_article_get(article, "url", "") or extra.get("pdf_url", "") or "")
        published_at = _article_published_at(article, extra)

        warnings: list[str] = []
        candidates: list[dict[str, Any]] = []
        period = _period_from_ir_article(article, extra, text)
        period_year, period_quarter, period_type = _period_parts(period)
        sections, document_chunks, topics, topic_signals = _extract_sections_and_chunks(
            pages,
            peer_id=peer_id,
        )
        table_candidates, financial_tables = _extract_financial_table_candidates(
            pages,
            report_period=period,
            peer_id=peer_id,
        )
        candidates.extend(table_candidates)
        candidates.extend(
            _extract_chart_block_candidates(
                pages,
                report_period=period,
                peer_id=peer_id,
            )
        )
        seen_metric_candidates: set[tuple[str, int | None, str]] = set()

        for page in pages:
            page_no = page.get("page")
            page_text = str(page.get("text", "") or "")

            if not period:
                period = _extract_period(page_text)
                period_year, period_quarter, period_type = _period_parts(period)

            for metric_type, patterns, value_kind in _IR_FINANCIAL_METRIC_RULES:
                for value, raw in _metric_values(page_text, patterns, value_kind):
                    dedupe_key = (metric_type, page_no if isinstance(page_no, int) else None, raw)
                    if dedupe_key in seen_metric_candidates:
                        continue
                    seen_metric_candidates.add(dedupe_key)
                    value_key = _candidate_value_key(value_kind)
                    if _has_same_table_metric_candidate(
                        candidates,
                        page_no=page_no,
                        metric_type=metric_type,
                        value_key=value_key,
                        value=value,
                    ):
                        continue
                    metric_context = _classify_metric_context(
                        page_text,
                        peer_id=peer_id,
                        raw_match=raw,
                    )
                    candidate = {
                        "page": page_no,
                        "type": metric_type,
                        "value_kind": value_kind,
                        value_key: value,
                        "raw": raw,
                        **metric_context,
                    }
                    candidates.append(candidate)
                    candidates.extend(
                        _inline_comparison_candidates(
                            page_text=page_text,
                            page_no=page_no,
                            metric_type=metric_type,
                            raw=raw,
                            base_candidate=metric_context,
                            report_period=period,
                        )
                    )

        revenue_total = _candidate_value(
            candidates,
            "revenue_total",
            "value_krwbn",
            allowed_scopes={"company_total", "segment"},
        )
        operating_profit = _candidate_value(
            candidates,
            "operating_profit",
            "value_krwbn",
            allowed_scopes={"company_total", "segment"},
        )

        if not period:
            warnings.append("period 추출 실패")
        if revenue_total is None:
            warnings.append("revenue_total 추출 실패")
        if operating_profit is None:
            warnings.append("operating_profit 추출 실패")

        financial_record = _build_financial_record(
            source="ir",
            peer_id=peer_id,
            period=period,
            title=title,
            url=url,
            published_at=published_at,
            candidates=candidates,
        )

        result = {
            "ok": bool(text),
            "source": "ir",
            "peer_id": peer_id,
            "title": title,
            "url": url,
            "published_at": published_at,
            "period": period,
            "period_year": period_year,
            "period_quarter": period_quarter,
            "period_type": period_type,
            "revenue_total_krwbn": revenue_total,
            "operating_profit_krwbn": operating_profit,
            "net_income_krwbn": financial_record.get("net_income_krwbn"),
            "ebitda_krwbn": financial_record.get("ebitda_krwbn"),
            "backlog_krwbn": financial_record.get("backlog_krwbn"),
            "orders_krwbn": financial_record.get("orders_krwbn"),
            "capex_krwbn": financial_record.get("capex_krwbn"),
            "operating_margin_pct": financial_record.get("operating_margin_pct"),
            "candidates": candidates,
            "financial_tables": financial_tables,
            "sections": sections,
            "page_index": _build_page_index(pages, document_chunks),
            "document_chunks": document_chunks,
            "topics": topics,
            "topic_signals": topic_signals,
            "metadata": {
                "source_page": extra.get("source_page"),
                "detail_url": extra.get("detail_url"),
                "pdf_url": extra.get("pdf_url") or url,
                "pdf_pages": extra.get("pdf_pages"),
                "pdf_parsed_pages": extra.get("pdf_parsed_pages"),
                "pdf_text_chars": extra.get("pdf_text_chars") or len(text),
                "pdf_parse_strategy": extra.get("pdf_parse_strategy"),
                "table_parse_strategy": extra.get("table_parse_strategy"),
                "chart_parse_strategy": extra.get("chart_parse_strategy"),
                "contains_images": extra.get("contains_images"),
                "image_count": extra.get("image_count"),
                "drawing_count": extra.get("drawing_count"),
                "ocr_applied": extra.get("ocr_applied"),
                "ocr_pages": extra.get("ocr_pages"),
            },
            "financial_record": financial_record,
            "warnings": warnings,
        }

        if include_raw_pages:
            result["raw_text_pages"] = pages

        log.info(
            "IR article 파싱 완료 | peer_id=%s period=%s rev=%s op=%s",
            peer_id,
            period,
            revenue_total,
            operating_profit,
        )
        return result

    def parse(
        self,
        pdf_path: str | Path,
        *,
        peer_id: str | None = None,
        max_pages: int = 30,
    ) -> dict[str, Any]:
        """PDF 파일 경로 기반 파싱. 이전 PoC 코드 호환용."""
        path = Path(pdf_path)
        if not path.exists():
            return {"ok": False, "reason": f"파일 없음: {path}", "warnings": []}

        try:
            with path.open("rb") as pdf_file:
                payload = extract_pdf_payload(pdf_file.read())
        except Exception as exc:
            return {"ok": False, "reason": f"PDF 열기 실패: {exc}", "warnings": []}

        pages_payload = payload.get("pages", [])[:max_pages]
        pages = [str(page.get("text") or "") for page in pages_payload if isinstance(page, dict)]

        article = {
            "title": path.name,
            "url": str(path),
            "content": "\n".join(f"[PAGE {idx + 1}]\n{text}" for idx, text in enumerate(pages)),
            "peer_id": peer_id,
            "extra": {
                "pdf_pages": payload.get("page_count"),
                "pdf_parsed_pages": len(pages),
                "pdf_page_blocks": pages_payload,
                "pdf_text_chars": len(payload.get("text") or ""),
                "pdf_parse_strategy": payload.get("pdf_parse_strategy"),
                "table_parse_strategy": payload.get("table_parse_strategy"),
                "chart_parse_strategy": payload.get("chart_parse_strategy"),
                "contains_images": payload.get("contains_images"),
                "image_count": payload.get("image_count"),
                "ocr_applied": payload.get("ocr_applied"),
                "ocr_pages": payload.get("ocr_pages"),
            },
        }
        result = self.parse_article(article, include_raw_pages=True)
        result["page_count"] = len(pages)
        return result


__all__ = [
    "IRParser",
    "_OPERATING_PROFIT_PATTERNS",
    "_REVENUE_PATTERNS",
    "_extract_period",
    "_first_amount",
    "_normalize_amount_krwbn",
]
