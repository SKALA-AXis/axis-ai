"""이벤트 타입 및 키워드 설정."""

from __future__ import annotations

from typing import Final

EVENT_TYPES: Final[list[str]] = [
    "partnership",
    "ma",
    "personnel",
    "tech_release",
    "regulation",
    "legal",
    "contract",
    "financial",
    "expansion",
    "company",
]

EVENT_TYPE_TIE_BREAK_PRIORITY: Final[dict[str, int]] = {
    "ma": 0,
    "contract": 1,
    "financial": 2,
    "partnership": 3,
    "regulation": 4,
    "legal": 5,
    "expansion": 6,
    "personnel": 7,
    "tech_release": 8,
    "company": 9,
}

EVENT_TYPE_KEYWORDS: Final[dict[str, list[str]]] = {
    "partnership": [
        "협약",
        "업무협약",
        "MOU",
        "파트너십",
        "제휴",
        "협력",
        "공동 개발",
        "공동사업",
        "컨소시엄",
    ],
    "ma": [
        "인수",
        "합병",
        "인수합병",
        "지분 인수",
        "지분 투자",
        "투자 유치",
        "피인수",
        "매각",
    ],
    "personnel": [
        "채용",
        "인사",
        "임원",
        "대표이사",
        "CEO",
        "선임",
        "영입",
        "조직개편",
        "조직 개편",
    ],
    "tech_release": [
        "출시",
        "공개",
        "선보",
        "런칭",
        "상용화",
        "정식 출시",
        "신제품",
        "신기술",
        "신서비스",
        "업그레이드",
        "적용",
        "실증",
        "PoC",
        "상용화 성과",
        "현장 점검",
    ],
    "regulation": [
        "규제",
        "정책",
        "법안",
        "가이드라인",
        "정부 인증",
        "국제표준",
        "표준 인증",
        "컴플라이언스",
    ],
    "legal": [
        "소송",
        "판결",
        "상고",
        "파기환송",
        "대법원",
        "법원",
        "재판부",
        "위자료",
        "재산분할",
        "불법원인급여",
    ],
    "contract": [
        "수주",
        "계약",
        "선정",
        "공급 계약",
        "구축 사업",
        "사업자 선정",
        "우선협상대상자",
        "프로젝트 수주",
    ],
    "financial": [
        "매출",
        "영업이익",
        "실적",
        "잠정실적",
        "분기 실적",
        "연간 실적",
        "흑자",
        "적자",
        "가이던스",
    ],
    "expansion": [
        "해외 진출",
        "글로벌 진출",
        "시장 확대",
        "사업 확대",
        "센터 설립",
        "법인 설립",
        "신시장",
        "현지화",
    ],
    "company": [
        "경영 전략",
        "사업 전략",
        "중장기",
        "비전",
        "기업가치",
        "지배구조",
        "그룹 내",
        "CEO 메시지",
        "주주총회",
    ],
}

# Some event keywords are semantically weak unless a nearby domain context exists.
# For example, "적용" appears in legal and accounting sentences as well as technology
# release sentences, so the frame builder only accepts it for tech_release when a
# technology/product context term is present in the same candidate text.
EVENT_KEYWORDS_REQUIRING_CONTEXT: Final[dict[str, list[str]]] = {
    "tech_release": ["적용", "공개", "현장 점검"],
}

EVENT_TYPE_CONTEXT_KEYWORDS: Final[dict[str, list[str]]] = {
    "tech_release": [
        "AI",
        "AX",
        "클라우드",
        "솔루션",
        "서비스",
        "플랫폼",
        "시스템",
        "소프트웨어",
        "데이터",
        "디지털",
        "자동화",
        "MSP",
        "SaaS",
        "보안",
        "모델",
        "생성형",
    ],
}

HIGH_IMPACT_KEYWORDS: Final[list[str]] = [
    "대규모 수주",
    "대형 수주",
    "메가딜",
    "우선협상대상자",
    "단일판매",
    "공급계약",
    "공시",
    "인수합병",
    "M&A",
    "지분 인수",
    "지분 투자",
    "전략적 제휴",
    "해외 진출",
    "신규 법인",
    "실적 발표",
    "영업이익",
    "흑자전환",
    "적자전환",
]

MEDIUM_IMPACT_KEYWORDS: Final[list[str]] = [
    "수주",
    "계약",
    "협약",
    "MOU",
    "출시",
    "공개",
    "고도화",
    "조직개편",
    "임원",
    "채용",
    "시장 확대",
    "투자계획",
]


def event_type_values() -> str:
    return "|".join(EVENT_TYPES)
