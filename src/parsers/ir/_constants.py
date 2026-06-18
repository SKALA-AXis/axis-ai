"""ir _constants — extracted from facade (move-only)."""

from __future__ import annotations

import re

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
