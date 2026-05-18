"""IR PDF 크롤링 결과를 재무 후보 레코드로 변환하는 deterministic parser.

IRCrawler는 PDF 파일을 직접 저장하지 않고 RawArticle 형태로 본문 텍스트와
PDF 페이지 블록을 담는다. 이 파서는 그 RawArticle 결과를 받아
FinancialLinkerAgent/peer_financials 적재에 사용할 수 있는 핵심 재무 후보를 만든다.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.sectors import SECTOR_KEYWORDS

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
        r"(?:영업이익률|OPM|Operating\s*Margin)\s*[:：]?\s*"
        r"([\d,]+(?:\.\d+)?)\s*(%|퍼센트|pct|p)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)\s*(%|퍼센트|pct|p)\s*"
        r"(?:영업이익률|OPM|operating\s*margin)",
        re.IGNORECASE,
    ),
]
_IR_FINANCIAL_METRIC_RULES: tuple[tuple[str, list[re.Pattern[str]], str], ...] = (
    ("revenue_total", _REVENUE_PATTERNS, "amount_krwbn"),
    ("operating_profit", _OPERATING_PROFIT_PATTERNS, "amount_krwbn"),
    ("net_income", _NET_INCOME_PATTERNS, "amount_krwbn"),
    ("ebitda", _EBITDA_PATTERNS, "amount_krwbn"),
    ("backlog", _BACKLOG_PATTERNS, "amount_krwbn"),
    ("capex", _CAPEX_PATTERNS, "amount_krwbn"),
    ("operating_margin", _OPERATING_MARGIN_PATTERNS, "percentage"),
)
_PERIOD_PATTERNS = [
    re.compile(r"(20\d{2})\s*년?\s*([1-4])\s*분기"),
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
        ("ai", "AI/Data", ("ai", "에이닷", "sapien", "data", "데이터", "생성형")),
        ("cloud", "Cloud", ("cloud", "클라우드", "dc", "data center", "데이터센터")),
        ("portfolio", "Portfolio", ("portfolio", "investment", "투자", "배당", "주주환원")),
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


def _normalize_amount_krwbn(value: str, unit: str) -> float:
    """금액 문자열을 기존 peer_financials 관례인 억원 단위 값으로 변환한다."""
    n = float(value.replace(",", ""))
    if unit in ("조원", "조"):
        return n * 10_000
    return n


def _extract_period(text: str) -> str | None:
    for pattern in _PERIOD_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue

        if len(match.group(1)) == 1 and len(match.group(2)) == 2:
            return f"20{match.group(2)}Q{match.group(1)}"

        return f"{match.group(1)}Q{match.group(2)}"

    return None


def _period_parts(period: str | None) -> tuple[int | None, int | None, str | None]:
    if not period:
        return None, None, None

    match = re.match(r"^(20\d{2})Q([1-4])$", period)
    if not match:
        return None, None, None

    year = int(match.group(1))
    quarter = int(match.group(2))
    return year, quarter, "quarter"


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
        }

    if metric_segment_area:
        return {
            "metric_scope": "segment",
            "business_area": metric_segment_area,
            "entity_name": peer_id,
            "confidence": 0.78,
        }

    if segment_area and not has_total:
        return {
            "metric_scope": "segment",
            "business_area": segment_area,
            "entity_name": peer_id,
            "confidence": 0.72,
        }

    if has_total:
        return {
            "metric_scope": "company_total",
            "business_area": None,
            "entity_name": peer_id,
            "confidence": 0.85,
        }

    return {
        "metric_scope": "unknown",
        "business_area": segment_area,
        "entity_name": peer_id,
        "confidence": 0.55,
    }


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
    for business_area, terms in _IR_SEGMENT_CONTEXT_RULES:
        if any(term.lower() in text for term in terms):
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
            block_texts = [
                str(block.get("text", ""))
                for block in blocks
                if isinstance(block, dict) and block.get("text")
            ]
            text = "\n".join(block_texts).strip()
            page_no = int(page.get("page") or idx)

            if text:
                pages.append({"page": page_no, "text": text})

    if pages:
        return pages

    text = str(_article_get(article, "content", "") or "")
    matches = list(_PAGE_SPLIT_PATTERN.finditer(text))

    if not matches:
        return [{"page": None, "text": text}] if text else []

    for idx, match in enumerate(matches):
        page_no = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            pages.append({"page": page_no, "text": page_text})

    return pages


def _candidate_page(candidates: list[dict[str, Any]], metric_type: str) -> int | None:
    for candidate in candidates:
        if candidate.get("type") == metric_type:
            page = candidate.get("page")
            return int(page) if isinstance(page, int) else None
    return None


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
    for candidate in candidates:
        if candidate.get("type") != metric_type:
            continue
        if allowed_scopes is not None and candidate.get("metric_scope") not in allowed_scopes:
            continue
        value = candidate.get(value_key)
        return float(value) if isinstance(value, int | float) else None
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
                    metric_context = _classify_metric_context(
                        page_text,
                        peer_id=peer_id,
                        raw_match=raw,
                    )
                    candidates.append(
                        {
                            "page": page_no,
                            "type": metric_type,
                            "value_kind": value_kind,
                            _candidate_value_key(value_kind): value,
                            "raw": raw,
                            **metric_context,
                        }
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
            "capex_krwbn": financial_record.get("capex_krwbn"),
            "operating_margin_pct": financial_record.get("operating_margin_pct"),
            "candidates": candidates,
            "sections": sections,
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
            import pymupdf  # type: ignore
        except ImportError:
            return {"ok": False, "reason": "pymupdf 미설치 - `uv add pymupdf`", "warnings": []}

        try:
            doc = pymupdf.open(path)
        except Exception as exc:
            return {"ok": False, "reason": f"PDF 열기 실패: {exc}", "warnings": []}

        pages: list[str] = []
        try:
            for idx in range(min(len(doc), max_pages)):
                page = doc.load_page(idx)
                pages.append(page.get_text("text") or "")
        finally:
            doc.close()

        article = {
            "title": path.name,
            "url": str(path),
            "content": "\n".join(f"[PAGE {idx + 1}]\n{text}" for idx, text in enumerate(pages)),
            "peer_id": peer_id,
            "extra": {"pdf_pages": len(pages), "pdf_parsed_pages": len(pages)},
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
