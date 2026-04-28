"""v3 §1.2 4개 트렌드 섹터 키워드 사전.

ClassificationAgent가 LLM 보조 + 키워드 매칭으로 섹터를 태깅한다.
"기타(other)"는 4개 섹터 중 어디에도 안 맞을 때 부여.
"""

from typing import Final, TypedDict


class SectorInfo(TypedDict):
    name_ko: str
    keywords: list[str]


# (섹터 ID → 섹터명, 키워드)
SECTOR_KEYWORDS: Final[dict[str, SectorInfo]] = {
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
            "AI 보안",
            "AI security",
            "secure AI",
            "프롬프트 인젝션",
        ],
    },
    "ai_tech": {
        "name_ko": "AI 기술동향",
        "keywords": [
            "에이전틱AI",
            "에이전틱 AI",
            "agentic AI",
            "AI 에이전트",
            "AI agent",
            "GPT",
            "LLM",
            "생성형 AI",
            "generative AI",
            "파운데이션",
            "foundation model",
            "팔란티어",
            "Palantir",
            "OpenAI",
            "Anthropic",
            "Claude",
            "Gemini",
            "RAG",
            "벡터",
            "임베딩",
            "AI 인프라",
            "AI Foundry",
            "GPU 클러스터",
            "LangChain",
            "MCP",
            "A2A",
        ],
    },
    "large_deal": {
        "name_ko": "대규모 수주",
        "keywords": [
            "수주",
            "대형 수주",
            "메가딜",
            "단일 수주",
            "프로젝트 수주",
            "공공",
            "공공 사업",
            "정부",
            "조달청",
            "디지털플랫폼정부",
            "국방",
            "국방부",
            "방산",
            "금융 차세대",
            "차세대 시스템",
            "스마트팩토리 구축",
            "ERP 구축",
            "MES 구축",
            "조 단위",
            "수천억",
            "수백억",
        ],
    },
    "sk_ax_biz": {
        "name_ko": "SK AX 사업영역",
        "keywords": [
            "제조AX",
            "제조 AX",
            "manufacturing AX",
            "에이전틱AI",
            "에이전틱 AI",
            "agentic AI",
            "MSP",
            "managed service provider",
            "클라우드 관리",
            "운영최적화",
            "운영 최적화",
            "프로세스 최적화",
            "AI 팩토리",
            "스마트팩토리",
            "smart factory",
            "디지털 트윈",
            "digital twin",
            "데이터 플랫폼",
        ],
    },
}

# v3 §1.2 4개 섹터 ID + "other"
SECTOR_IDS: Final[list[str]] = list(SECTOR_KEYWORDS.keys()) + ["other"]


def match_sectors(text: str) -> list[str]:
    """텍스트에서 매칭되는 섹터 ID 목록을 반환한다 (다중 매칭 허용).

    Args:
        text: 제목 + 본문 합성 문자열.

    Returns:
        매칭된 섹터 ID 목록. 매칭 0건이면 ["other"].
    """
    if not text:
        return ["other"]
    lowered = text.lower()
    matched: list[str] = []
    for sector_id, info in SECTOR_KEYWORDS.items():
        for kw in info["keywords"]:
            if kw.lower() in lowered:
                matched.append(sector_id)
                break
    return matched if matched else ["other"]


def primary_sector(text: str) -> str:
    """가장 매칭이 강한 섹터 1개를 반환 (다중 매칭 시 첫 매칭 우선)."""
    matched = match_sectors(text)
    return matched[0]


def sector_name_ko(sector_id: str) -> str:
    """섹터 ID → 한국어 이름. other → '기타'."""
    if sector_id == "other":
        return "기타"
    info = SECTOR_KEYWORDS.get(sector_id)
    return info["name_ko"] if info else sector_id
