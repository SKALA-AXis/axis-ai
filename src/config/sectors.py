"""전략 모니터링 섹터 키워드 설정."""

from __future__ import annotations

from typing import Final, TypedDict


class SectorInfo(TypedDict):
    name_ko: str
    keywords: list[str]


SECTOR_KEYWORDS: Final[dict[str, SectorInfo]] = {
    "ax": {
        "name_ko": "AX",
        "keywords": [
            "AX",
            "AI Transformation",
            "AI 전환",
            "AI 혁신",
            "디지털 전환",
            "DX",
            "제조AX",
            "제조 AX",
            "엔터프라이즈 AI",
            "에이전틱AI",
            "에이전틱 AI",
            "agentic AI",
            "AI 에이전트",
            "AI agent",
            "생성형 AI",
            "generative AI",
            "LLM",
            "RAG",
            "AI 플랫폼",
            "AI 팩토리",
            "스마트팩토리",
            "smart factory",
            "디지털 트윈",
            "digital twin",
            "업무 자동화",
            "프로세스 최적화",
            "운영 최적화",
        ],
    },
    "security": {
        "name_ko": "보안",
        "keywords": [
            "보안",
            "사이버보안",
            "정보보안",
            "정보보호",
            "제로트러스트",
            "ZTA",
            "EDR",
            "XDR",
            "SOC",
            "관제",
            "취약점",
            "랜섬웨어",
            "침해",
            "해킹",
            "데이터 유출",
            "개인정보",
            "ISMS",
            "ISMS-P",
            "클라우드 보안",
            "AI 보안",
            "AI security",
            "secure AI",
            "프롬프트 인젝션",
        ],
    },
    "infra": {
        "name_ko": "인프라",
        "keywords": [
            "인프라",
            "IT 인프라",
            "클라우드",
            "cloud",
            "클라우드 전환",
            "클라우드 관리",
            "MSP",
            "managed service provider",
            "데이터센터",
            "데이터 센터",
            "IDC",
            "GPU",
            "GPU 클러스터",
            "AI 인프라",
            "서버",
            "네트워크",
            "스토리지",
            "가상화",
            "쿠버네티스",
            "Kubernetes",
            "컨테이너",
            "프라이빗 클라우드",
            "하이브리드 클라우드",
            "망분리",
        ],
    },
    "deal": {
        "name_ko": "수주",
        "keywords": [
            "수주",
            "대형 수주",
            "메가딜",
            "단일 수주",
            "프로젝트 수주",
            "계약",
            "공급 계약",
            "사업자 선정",
            "우선협상대상자",
            "MOU",
            "업무협약",
            "양해각서",
            "협약",
            "정부 협약",
            "정부",
            "공공",
            "공공 사업",
            "조달청",
            "디지털플랫폼정부",
            "인수",
            "합병",
            "인수합병",
            "M&A",
            "지분 인수",
            "지분 투자",
            "투자 유치",
        ],
    },
}

SECTOR_IDS: Final[list[str]] = list(SECTOR_KEYWORDS.keys()) + ["other"]


def match_sectors(text: str) -> list[str]:
    if not text:
        return ["other"]

    lowered = text.lower()
    matched: list[str] = []

    for sector_id, info in SECTOR_KEYWORDS.items():
        for keyword in info["keywords"]:
            if keyword.lower() in lowered:
                matched.append(sector_id)
                break

    return matched if matched else ["other"]


def primary_sector(text: str) -> str:
    return match_sectors(text)[0]


def sector_name_ko(sector_id: str) -> str:
    if sector_id == "other":
        return "기타"

    info = SECTOR_KEYWORDS.get(sector_id)
    return info["name_ko"] if info else sector_id
