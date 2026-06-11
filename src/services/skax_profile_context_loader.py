"""SK AX profile context loading service.

SK AX 공식 프로필, 섹터별 관점, 공식 문서/newsroom 문서를 로드해
시사점 도출 단계의 ProfileContext로 넘기는 service 모듈이다.

주의:
- 이 파일은 Agent 폴더에서 분리된 service 진입점이다.
- ``SKAXPerspectiveAgent``는 기존 호환을 위해 함께 이동되어 있지만,
  현재 1단계 supervisor의 표준 시사점 생성 경로는 ``ImplicationAgent``다.
- DB schema나 저장 구조를 만들지 않는다.

섹터 기준은 src.config.sectors를 단일 출처로 사용한다.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.config.sectors import SECTOR_IDS, SECTOR_KEYWORDS, sector_name_ko
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROFILE_DOC_CONTENT_LIMIT = 2400
_VIEWPOINT_PROFILE_DOC_CONTENT_LIMIT = 700
_VIEWPOINT_NEWSROOM_DOC_CONTENT_LIMIT = 1500
_CARD_PROFILE_DOC_CONTENT_LIMIT = 320
_CARD_NEWSROOM_DOC_CONTENT_LIMIT = 600
_SOURCE_INTELLIGENCE_DOC_LIMIT = 24
_SOURCE_INTELLIGENCE_POINT_LIMIT = 5
_llm: ChatOpenAI | None = None
_viewpoint_llm: ChatOpenAI | None = None


class OfficialSource(TypedDict):
    title: str
    url: str
    basis: str


class SKAXDomainContext(TypedDict):
    domain_id: str
    title: str
    sector_ids: list[str]
    perspective: str
    capabilities: list[str]


class SKAXSectorContext(TypedDict):
    id: str
    name_ko: str
    keywords: list[str]


class SKAXOfficialDocument(TypedDict):
    id: int
    title: str
    url: str
    page_kind: str
    headings: list[str]
    content_excerpt: str


class SKAXPerspectiveResult(TypedDict):
    is_relevant_to_skax: bool
    perspective_scope: Literal["sk_ax_official"]
    matched_skax_domains: list[str]
    why_important: str
    potential_impact: str
    follow_up_questions: list[str]
    why_it_matters: str
    competitive_implication: str
    opportunity: str
    risk: str
    suggested_actions: list[str]
    confidence: float
    reason: str
    evidence: list[OfficialSource]


_SKAX_OFFICIAL_PROFILE: Final[dict[str, Any]] = {
    "company": "SK AX",
    "positioning": (
        "AI 기술과 산업 전문성을 바탕으로 제조, 통신/미디어, 반도체, "
        "에너지/화학, 금융, 공공 등 고객의 AX 혁신을 지원하는 회사"
    ),
    "default_viewpoint": (
        "피어사 동향을 SK AX의 AX 사업 기회, 운영 혁신, 클라우드/인프라 역량, "
        "산업별 레퍼런스 확장 관점에서 보수적으로 해석한다."
    ),
}

_SKAX_DOMAIN_CONTEXTS: Final[list[SKAXDomainContext]] = [
    {
        "domain_id": "agentic_ai_operation",
        "title": "AXgenticWire 기반 Agentic AI 운영 혁신",
        "sector_ids": ["ax", "infra"],
        "perspective": (
            "기업 운영 체계 전반을 Agentic 환경에 맞게 바꾸고, 의사결정/자원분배/"
            "운영구조의 최적화를 핵심 가치로 본다."
        ),
        "capabilities": [
            "Agent Builder",
            "Human-in-the-Loop",
            "Multi-LLM",
            "AI-Readable Data & Governance",
            "서버리스/Auto Scaling/보안 인프라",
        ],
    },
    {
        "domain_id": "cloud_ai_infra",
        "title": "Cloud 및 AI 인프라 전환/운영",
        "sector_ids": ["infra", "security", "ax"],
        "perspective": (
            "AI와 Big Data 활용을 위한 클라우드 설계, 전환, 운영, 보안관제, "
            "비용 최적화와 AI Cloud Landing Zone을 중요하게 본다."
        ),
        "capabilities": [
            "Cloud Migration",
            "Hybrid/Multi Cloud",
            "Cloud Operation",
            "Cloud Security Monitoring",
            "MCMP",
            "Application Modernization",
            "Container Platform",
            "AI Cloud Landing Zone",
        ],
    },
    {
        "domain_id": "manufacturing_ax",
        "title": "제조 AX 및 스마트팩토리 지능화",
        "sector_ids": ["ax", "infra"],
        "perspective": (
            "제조 현장의 공정 최적화, 품질 향상, 설비 예지정비, 디지털 트윈, "
            "공급망 리스크 대응처럼 산업 데이터 기반 실행력을 중점적으로 본다."
        ),
        "capabilities": [
            "실시간 공정 분석 및 제어 자동화",
            "전 공정 통합 관제",
            "품질 분석 및 보정 자동화",
            "설비 예지정비",
            "디지털 트윈",
            "수요 예측 자동화",
            "물류/공급망 최적화",
        ],
    },
    {
        "domain_id": "public_finance_deal",
        "title": "공공/금융 AX 사업 기회",
        "sector_ids": ["deal", "ax", "security"],
        "perspective": (
            "공공, 금융, 기업 고객의 업무 혁신과 보안 요구를 충족하는 AI/클라우드/"
            "운영 서비스 확장 가능성을 본다."
        ),
        "capabilities": [
            "Backoffice AX",
            "Market Intelligence",
            "AI 콜센터",
            "공공/금융 시스템 운영",
            "보안 요구 기반 클라우드/AI 적용",
        ],
    },
    {
        "domain_id": "digital_esg",
        "title": "Digital ESG 및 지속가능 운영",
        "sector_ids": ["ax", "infra", "deal"],
        "perspective": (
            "AI와 데이터 기반 ESG 전략 수립, 점검, 평가, 공급망 관리와 제조 현장의 "
            "탄소/에너지 효율화를 경쟁 포인트로 본다."
        ),
        "capabilities": [
            "ESG 경영 진단/관리",
            "ESG 공시 지원",
            "공급망 ESG 관리",
            "탄소 배출량 측정 및 관리",
            "에너지 사용 최적화",
        ],
    },
]

_SKAX_PERSPECTIVE_PROMPT: Final[str] = """\
당신은 SK AX 관점 분석 Agent입니다.

목적:
- 피어사 뉴스 요약과 피어사 분석을 SK AX 공식 사업 관점에 연결합니다.
- 카드뉴스의 시사점/대응 후보를 만들기 위한 내부 관점 초안을 생성합니다.
- 사실 요약에 없는 피어사 정보나 SK AX 공식 사이트에 없는 역량을 만들지 않습니다.

## SK AX 공식 관점
skax_profile:
{skax_profile_json}

selected_sector_config:
{sector_config_json}

skax_official_documents:
{skax_documents_json}

skax_newsroom_documents:
{skax_newsroom_documents_json}

source_intelligence:
{source_intelligence_json}

skax_contexts:
{skax_context_json}

## 피어사 입력
issue_card:
{issue_card_json}

summary:
{summary_json}

analysis:
{analysis_json}

classification:
{classification_json}

## 작성 원칙
1. selected_sector_config는 현재 코드의 src/config/sectors.py에서 온 섹터 정의입니다.
   섹터가 바뀌면 이 입력을 기준으로 판단하세요.
2. skax_official_documents는 DB에 저장된 SK AX 공식 사이트 원문입니다.
   공식 문서가 있으면 이 원문을 우선 근거로 삼고, 없으면 skax_profile과 skax_contexts를 사용하세요.
3. skax_newsroom_documents는 company_news 크롤러가 수집한 SK AX 공식 뉴스룸 문서입니다.
   회사/서비스 프로필보다 최신 실행 사례와 공식 발표 흐름을 확인하는 보조 근거로 사용하세요.
4. skax_contexts에는 SK AX 공식 사이트 기반 전체 관점이 모두 들어 있습니다.
   priority="primary"는 카드 섹터와 직접 연결되는 우선 관점이고,
   priority="supporting"은 보조 관점입니다.
5. primary 관점을 먼저 검토하되, 피어사 뉴스와 더 강하게 연결되는 supporting 관점이 있으면
   함께 사용할 수 있습니다. 단, 연결 근거를 reason에 설명하세요.
6. 공식 근거가 약하면 기회/리스크를 단정하지 말고 모니터링 관점으로 낮추세요.
7. 피어사 뉴스 요약에 없는 고객명, 수치, 제품명, 계약 내용을 만들지 마세요.
8. SK AX가 해야 한다고 단정하지 말고, 검토/확인/모니터링/메시지 정리 수준의 후보 행동으로 쓰세요.
9. 시장 반응 또는 주가 기사라도 피어사 단독 상황이면 관점은 만들 수 있지만,
   사업 실행 신호가 약하면 confidence를 낮추고 대응은 관찰 중심으로 쓰세요.
10. why_it_matters는 SK AX 관점에서 왜 볼 만한지 1~2문장으로 설명하세요.
11. competitive_implication은 경쟁 구도상 의미를 1문장으로 쓰세요.
12. opportunity와 risk는 각각 비어 있지 않게 쓰되, 근거가 약하면 "제한적"이라고 명시하세요.
13. suggested_actions는 2~3개만 작성하세요.

다음 JSON 형식으로만 응답하세요.
{{
  "is_relevant_to_skax": true,
  "matched_skax_domains": ["domain_id"],
  "why_important": "SK AX 관점에서 중요한 이유",
  "potential_impact": "SK AX에 줄 수 있는 영향",
  "follow_up_questions": ["추가 확인 질문"],
  "why_it_matters": "SK AX 관점 의미",
  "competitive_implication": "경쟁 구도상 의미",
  "opportunity": "기회 후보",
  "risk": "리스크 후보",
  "suggested_actions": ["검토 후보 1", "검토 후보 2"],
  "confidence": 0.0,
  "reason": "판단 근거",
  "evidence": [
    {{"title": "공식 근거 제목", "url": "https://...", "basis": "연결 근거"}}
  ]
}}"""

_SKAX_VIEWPOINT_PROMPT: Final[str] = """\
당신은 SK AX 공식 자료를 바탕으로 내부 관점 플레이북을 만드는 Agent입니다.

## P.C.R.O 프레임워크
P Persona:
- 당신은 10년 차 B2B AX/클라우드/엔터프라이즈 AI 전략 컨설턴트입니다.
- 동시에 경쟁사 동향을 우리 회사의 사업 기회와 리스크로 번역하는
  Competitive Intelligence 분석가입니다.

C Context:
- 입력 자료는 SK AX 공식 사이트와 SK AX 뉴스룸 문서입니다.
- 이 결과는 최종 사용자에게 바로 노출되는 문구가 아니라,
  이후 카드뉴스의 시사점/대응을 만드는 다음 AI가 참고할 내부 관점 기준입니다.
- 다음 AI는 원문을 다시 보지 못할 수 있으므로, 이 JSON만 보고도
  SK AX의 현재 역량, 실행 사례, 판단 기준을 이해할 수 있어야 합니다.

R Restriction:
- 공식 자료에서 확인되지 않는 고객명, 수치, 제품명, 성과를 만들지 마세요.
- 모르는 내용은 단정하지 말고 "공식 근거 부족" 또는 "추가 확인 필요"로 낮추세요.
- "효율성을 높인다", "경쟁력을 강화한다" 같은 추상 결론만 쓰지 말고,
  어떤 방식으로 그렇게 되는지 함께 쓰세요.
- 단계적으로 검토하되, 추론 과정은 출력하지 말고 최종 JSON만 출력하세요.

O Output:
- 아래 JSON 형식을 반드시 지키세요.
- 각 문장은 다음 AI가 그대로 재사용할 수 있도록 근거, 작동 방식, 활용 의미를 담으세요.

목적:
- 특정 피어사 카드나 기사 없이, SK AX 공식 사이트/뉴스룸 자료만 보고
  SK AX가 어떤 관점으로 시장 이슈를 해석해야 하는지 정리합니다.
- 이후 카드뉴스의 시사점/대응 생성에 재사용할 수 있는 판단 기준을 만듭니다.
- 공식 자료에 없는 사업, 고객, 성과, 역량은 만들지 않습니다.

## SK AX 공식 관점
skax_profile:
{skax_profile_json}

selected_sector_config:
{sector_config_json}

skax_official_documents:
{skax_documents_json}

skax_newsroom_documents:
{skax_newsroom_documents_json}

source_intelligence:
{source_intelligence_json}

skax_contexts:
{skax_context_json}

## 작성 원칙
1. 특정 경쟁사나 카드에 대한 판단을 하지 마세요.
2. SK AX 공식 사이트/뉴스룸에서 확인되는 전략, 역량, 실행 사례를 기준으로 작성하세요.
   source_intelligence는 공식 문서에서 미리 뽑은 고밀도 근거이므로,
   company_viewpoint와 sector_viewpoints를 작성할 때 우선 참고하세요.
3. current_state에는 SK AX가 현재 무엇을 하고 있는지 구체적으로 적으세요.
   예: 서비스/플랫폼, 공식 뉴스룸의 최근 실행 사례, 인재/운영/산업 확장 신호.
4. strategic_direction에는 SK AX가 어느 방향으로 움직이는지 적으세요.
   예: Agentic AI 운영, 산업별 AX, AI 인재 육성, 클라우드/운영 고도화.
5. 시사점 원칙은 "앞으로 어떤 유형의 외부 이슈를 중요하게 볼 것인가"로 쓰세요.
   단순히 "AI/클라우드/인재가 중요하다"라고 쓰지 말고,
   외부 뉴스에서 어떤 조건이 보이면 SK AX 관점에서 의미가 커지는지 적으세요.
   예: 금융권 운영 자동화 계약, 공공/제조의 Agentic AI 적용, AI 인재 확산,
   클라우드 운영/보안/비용 최적화 요구, 산업별 AX 레퍼런스 확대.
6. 대응 원칙은 "실제 카드에서 대응을 만들 때 어떤 방향으로 제안할 것인가"로 쓰세요.
   단순히 "도입/확장/제안"이라고 쓰지 말고,
   어떤 고객군/서비스/메시지/검증 액션으로 연결할지 적으세요.
   예: AXgenticWire NPO 적용 가능 업종 확인, 금융/공공 보안 요구 메시지 정리,
   SKALA/AI Talent Lab과 연결한 인재 확보 관점, 유사 산업 레퍼런스 패키징.
7. "AI가 중요하다", "클라우드가 중요하다" 같은 일반론만 쓰지 말고,
   SK AX 공식 자료에서 확인되는 구체 역량/실행 사례와 연결하세요.
8. 공식 근거가 강한 내용과 약한 내용을 구분하고, 과장하지 마세요.
9. core_viewpoints는 3~5개로 제한하세요.
10. implication_principles, response_principles, watch_signals는 각각 3~5개로 제한하세요.
11. implication_principles와 response_principles는 각각 20자 이상의 구체 문장으로 쓰세요.
12. watch_signals도 "시장 동향" 같은 말 대신 감지 가능한 신호로 쓰세요.
13. sector_viewpoints에는 선택된 섹터별로 SK AX가 무엇을 중요하게 보고,
    어떤 신호를 기회/리스크로 해석할지 분리해서 적으세요.
14. viewpoint_strength는 공식 홈페이지/뉴스룸 근거를 기준으로 strong/medium/weak 중 하나로 쓰세요.
    근거가 적으면 confidence를 과도하게 높이지 마세요.
15. 뉴스룸 근거는 사업 실행/고객 적용/계약/운영 혁신 사례를 우선 반영하세요.
    인재 육성/교육 뉴스는 AX 확산 기반으로만 보조 반영하고, 핵심 관점으로 과대 반영하지 마세요.
16. company_viewpoint는 모든 카드에 기본으로 깔리는 SK AX 전체 관점입니다.
    섹터와 무관하게 SK AX가 현재 무엇을 하고 있고, 어디로 가는지를 정리하세요.
17. sector_viewpoints는 sectors.py의 selected_sector_config 기준으로 섹터별 판단축을 정리하세요.
    섹터 정의를 새로 만들지 말고 selected_sector_config의 id/name/keywords를 따르세요.
18. cross_sector_signals는 여러 섹터에 동시에 걸치는 신호를 적으세요.
    예: Agentic AI 운영 자동화는 AX+Infra+Security에 걸칠 수 있습니다.
19. selected_sector_config에 포함된 모든 섹터에 대해 sector_viewpoints를 하나씩 작성하세요.
20. 보안 섹터가 포함되면 Cloud Security Monitoring, 금융/공공 보안 요구,
    에이전틱 AI 운영에서의 장애/휴먼에러/운영 리스크 차단 관점을 공식 자료와 연결하세요.
21. company_viewpoint는 가장 중요합니다. 짧은 키워드 나열로 끝내지 말고,
    공식 사이트에서 확인되는 보유 역량, 뉴스룸에서 확인되는 실행 사례,
    그 사례가 보여주는 전략 방향, 카드뉴스 시사점/대응에 쓰는 방식을 구체적으로 쓰세요.
22. company_viewpoint의 각 배열 항목은 가능하면 1문장으로 쓰고,
    "무엇을 한다" + "어떤 근거/사례가 있다" + "그래서 어떤 관점으로 해석한다"가 드러나게 하세요.
23. recent_execution은 skax_newsroom_documents에서 확인되는 최근 공식 발표/실행 사례를
    우선합니다. 단순 홍보 문구보다 고객 적용, 계약, 운영 혁신, 서비스 출시,
    인재 확산처럼 실행이 확인되는 뉴스를 우선하세요.
24. company_viewpoint의 배열 항목은 다음 AI가 별도 원문 없이도 이해할 수 있어야 합니다.
    단순 키워드나 제목만 쓰지 말고, 각 항목에 최소한 "근거", "의미", "활용 관점" 중
    2개 이상을 포함하세요.
25. company_viewpoint.current_focus, strategic_direction, execution_priorities,
    default_implication_lens, default_response_principles, card_usage_guidance의 각 항목은
    50자 이상의 설명형 문장으로 작성하세요. "Agentic AI 운영 혁신"처럼 명사구만 쓰지 마세요.
26. recent_execution은 뉴스 제목만 쓰지 말고, 그 뉴스가 보여주는 실행 내용과
    SK AX 관점에서의 의미를 함께 쓰세요.
27. sector_viewpoints의 opportunity_lens, risk_lens, response_lens도 가능하면
    "어떤 신호가 보이면 / 왜 중요한지 / 어떤 대응으로 연결할지"가 드러나게 쓰세요.
28. "효율성을 높인다", "운영 혁신을 한다", "경쟁력을 강화한다"처럼 추상적인 결론만
    쓰지 마세요. 반드시 어떻게 하는지까지 쓰세요. 예를 들어 어떤 Agent,
    어떤 운영 대상, 어떤 자동화 단계, 어떤 고객군, 어떤 리스크 차단,
    어떤 비용/보안/장애 대응 효과인지 포함하세요.
29. current_focus는 2~4개 항목으로 나누고, 각 항목은
    "집중 영역 → 작동 방식 → 기대 효과/카드 활용 의미" 순서가 드러나게 쓰세요.
30. recent_execution은 2~5개 항목으로 작성하고, 뉴스룸 원문에 있는 고객/서비스/
    적용 방식/기대 효과를 함께 요약하세요.
31. execution_pattern은 최근 사례를 묶어서 "SK AX가 반복해서 취하는 실행 방식"을
    설명하세요. 예: 고객 운영 업무에 AI Agent를 단계적으로 적용하고,
    모니터링·백업·장애/상황관리처럼 운영 리스크가 큰 영역부터 자동화한다.
32. default_implication_lens와 default_response_principles는 이후 카드 생성 AI가 그대로
    참고할 수 있도록, 추상 명사 대신 판단 조건과 대응 방향을 함께 쓰세요.

## 원하는 문장 밀도 예시
나쁜 예:
- "Agentic AI 기반 운영 혁신을 통해 기업의 운영 체계를 최적화한다."

좋은 예:
- "SK AX는 AXgenticWire NPO를 금융 인프라 운영에 적용해 모니터링, 백업,
  장애/상황관리 같은 운영 리스크가 큰 영역부터 AI Agent로 자동화하려 한다.
  이 근거가 있는 경쟁사 뉴스는 단순 AI 도입이 아니라 운영 안정성, 보안 요구,
  비용 최적화 경쟁으로 해석한다."

나쁜 예:
- "클라우드 보안 요구를 기회로 본다."

좋은 예:
- "금융/공공 고객군에서 클라우드 전환, AI 인프라 운영, 보안관제 요구가 함께
  등장하면 SK AX의 Hybrid/Multi Cloud 운영과 Cloud Security Monitoring 역량을
  결합해 장애 예방, 접근 통제, 운영 비용 최적화 메시지로 대응할 수 있는
  신호로 해석한다."

다음 JSON 형식으로만 응답하세요.
{{
  "perspective_scope": "sk_ax_profile",
  "selected_sector_ids": ["sector_id"],
  "company_viewpoint": {{
    "positioning": "SK AX의 전체 포지션",
    "official_basis": ["공식 사이트/뉴스룸에서 확인되는 근거와 해석"],
    "current_focus": ["현재 집중하는 사업/기술과 그 근거"],
    "recent_execution": ["최근 공식 뉴스룸 기반 실행 사례와 의미"],
    "execution_pattern": ["최근 실행 사례들이 공통적으로 보여주는 패턴"],
    "strategic_direction": ["앞으로의 방향과 왜 그렇게 해석하는지"],
    "execution_priorities": ["우선 실행/강조 축과 판단 기준"],
    "default_implication_lens": ["모든 카드에 기본 적용할 시사점 렌즈"],
    "default_response_principles": ["모든 카드에 기본 적용할 대응 원칙"],
    "card_usage_guidance": ["카드뉴스 시사점/대응에 이 관점을 쓰는 방법"],
    "evidence": [
      {{"title": "공식 근거 제목", "url": "https://...", "basis": "연결 근거"}}
    ]
  }},
  "sector_viewpoints": [
    {{
      "sector_id": "sector_id",
      "sector_name": "섹터명",
      "viewpoint_strength": "strong|medium|weak",
      "skax_stance": "이 섹터를 바라보는 SK AX의 입장",
      "what_skax_has": ["이 섹터에서 SK AX가 가진 공식 역량/실행 사례"],
      "opportunity_lens": ["어떤 외부 신호를 기회로 볼지"],
      "risk_lens": ["어떤 외부 신호를 리스크나 확인 필요 신호로 볼지"],
      "response_lens": ["어떤 대응 방향으로 연결할지"]
    }}
  ],
  "cross_sector_signals": [
    {{
      "signal": "여러 섹터에 걸치는 외부 신호",
      "related_sector_ids": ["sector_id"],
      "why_it_matters": "SK AX 관점에서 중요한 이유",
      "response_hint": "대응 방향"
    }}
  ],
  "evidence_summary": {{
    "profile_doc_count": 0,
    "newsroom_doc_count": 0,
    "strongest_sources": ["관점 형성에 가장 강하게 쓰인 근거"]
  }},
  "confidence": 0.0,
  "reason": "판단 근거"
}}"""


def _get_llm() -> ChatOpenAI:
    from langchain_openai import ChatOpenAI  # lazy: transformers 체인 회피

    global _llm

    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.15,
            max_completion_tokens=1000,
        )

    return _llm


def _get_viewpoint_llm() -> ChatOpenAI:
    from langchain_openai import ChatOpenAI  # lazy: transformers 체인 회피

    global _viewpoint_llm

    if _viewpoint_llm is None:
        _viewpoint_llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.1,
            max_completion_tokens=4000,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    return _viewpoint_llm


class SKAXPerspectiveAgent:
    """피어사 카드/요약/분석을 SK AX 공식 관점으로 해석한다."""

    def build_viewpoint(
        self,
        sector_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """카드 없이 SK AX 공식 자료만으로 관점 플레이북을 생성한다."""
        context = SKAXProfileLoader().load(sector_ids)
        prompt = _SKAX_VIEWPOINT_PROMPT.replace(
            "{skax_profile_json}",
            json.dumps(context["skax_profile"], ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{sector_config_json}",
            json.dumps(context["sector_config"], ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{skax_documents_json}",
            json.dumps(
                _documents_for_prompt(
                    context["skax_documents"],
                    content_limit=_VIEWPOINT_PROFILE_DOC_CONTENT_LIMIT,
                ),
                ensure_ascii=False,
                indent=2,
            ),
        )
        prompt = prompt.replace(
            "{skax_newsroom_documents_json}",
            json.dumps(
                _documents_for_prompt(
                    context["skax_newsroom_documents"],
                    content_limit=_VIEWPOINT_NEWSROOM_DOC_CONTENT_LIMIT,
                ),
                ensure_ascii=False,
                indent=2,
            ),
        )
        prompt = prompt.replace(
            "{source_intelligence_json}",
            json.dumps(
                _source_intelligence_for_prompt(context),
                ensure_ascii=False,
                indent=2,
            ),
        )
        prompt = prompt.replace(
            "{skax_context_json}",
            json.dumps(context["skax_contexts"], ensure_ascii=False, indent=2),
        )

        try:
            from src.observability import tracing_config

            vp_config = tracing_config(
                agent="SKAXPerspectiveAgent",
                phase="viewpoint_playbook",
            )
        except Exception:
            vp_config = None

        try:
            response = _get_viewpoint_llm().invoke(prompt, config=vp_config)
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            return _normalize_viewpoint(_parse_json(content), context)
        except Exception as exc:
            log.error("SK AX 관점 플레이북 생성 실패 | error=%s", exc)
            return _fallback_viewpoint(
                context,
                f"LLM 관점 플레이북 생성 실패: {type(exc).__name__}",
            )

    def analyze(
        self,
        issue_card: dict[str, Any],
        summary: dict[str, Any] | None = None,
        analysis: dict[str, Any] | None = None,
        classification: dict[str, Any] | None = None,
    ) -> SKAXPerspectiveResult:
        """SK AX 관점 분석을 생성한다."""
        summary = summary or {}
        analysis = analysis or {}
        classification = classification or {}

        if not issue_card and not summary:
            return _empty_perspective("분석할 카드 또는 요약 입력이 없습니다.")

        sector_ids = _sector_ids_from_inputs(
            issue_card=issue_card,
            summary=summary,
            analysis=analysis,
            classification=classification,
        )
        context = SKAXProfileLoader().load(sector_ids)
        prompt = _SKAX_PERSPECTIVE_PROMPT.replace(
            "{skax_profile_json}",
            json.dumps(context["skax_profile"], ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{sector_config_json}",
            json.dumps(context["sector_config"], ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{skax_documents_json}",
            json.dumps(
                _documents_for_prompt(
                    context["skax_documents"],
                    content_limit=_CARD_PROFILE_DOC_CONTENT_LIMIT,
                ),
                ensure_ascii=False,
                indent=2,
            ),
        )
        prompt = prompt.replace(
            "{skax_newsroom_documents_json}",
            json.dumps(
                _documents_for_prompt(
                    context["skax_newsroom_documents"],
                    content_limit=_CARD_NEWSROOM_DOC_CONTENT_LIMIT,
                ),
                ensure_ascii=False,
                indent=2,
            ),
        )
        prompt = prompt.replace(
            "{source_intelligence_json}",
            json.dumps(
                _source_intelligence_for_prompt(context),
                ensure_ascii=False,
                indent=2,
            ),
        )
        prompt = prompt.replace(
            "{skax_context_json}",
            json.dumps(context["skax_contexts"], ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{issue_card_json}",
            json.dumps(_issue_card_for_prompt(issue_card), ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{summary_json}",
            json.dumps(_summary_for_prompt(summary), ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{analysis_json}",
            json.dumps(_analysis_for_prompt(analysis), ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{classification_json}",
            json.dumps(_classification_for_prompt(classification), ensure_ascii=False, indent=2),
        )

        try:
            from src.observability import tracing_config

            persp_config = tracing_config(
                agent="SKAXPerspectiveAgent",
                phase="analyze",
                card_id=issue_card.get("id"),
                cluster_id=issue_card.get("cluster_id") or summary.get("cluster_id"),
            )
        except Exception:
            persp_config = None

        try:
            response = _get_llm().invoke(prompt, config=persp_config)
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            result = _normalize_perspective(_parse_json(content), context)
        except Exception as exc:
            log.error(
                "SK AX 관점 분석 실패 | card=%s cluster=%s error=%s",
                issue_card.get("id"),
                issue_card.get("cluster_id") or summary.get("cluster_id"),
                exc,
            )
            return _empty_perspective(f"LLM 관점 분석 실패: {type(exc).__name__}")

        log.info(
            "SK AX 관점 분석 완료 | card=%s relevant=%s domains=%s confidence=%.2f",
            issue_card.get("id"),
            result["is_relevant_to_skax"],
            ",".join(result["matched_skax_domains"]),
            result["confidence"],
        )
        return result


class SKAXProfileLoader:
    """SK AX 공식 프로필을 섹터 우선순위와 함께 로드한다."""

    def load(
        self,
        sector_ids: list[str] | None = None,
        *,
        use_db: bool = True,
        max_documents: int | None = None,
        max_newsroom_documents: int | None = None,
    ) -> dict[str, Any]:
        """공식 사이트 원문과 섹터 우선 관점을 반환한다."""
        documents = load_skax_official_documents(limit=max_documents) if use_db else []
        newsroom_documents = (
            load_skax_newsroom_documents(limit=max_newsroom_documents) if use_db else []
        )
        return build_skax_context(
            sector_ids,
            official_documents=documents,
            newsroom_documents=newsroom_documents,
        )


def build_skax_context(
    sector_ids: list[str] | None = None,
    official_documents: list[SKAXOfficialDocument] | None = None,
    newsroom_documents: list[SKAXOfficialDocument] | None = None,
) -> dict[str, Any]:
    """전체 SK AX 공식 관점에 섹터 기반 우선순위를 붙인다."""
    normalized_sector_ids = _normalize_sector_ids(sector_ids)
    sector_config = [_sector_context(sector_id) for sector_id in normalized_sector_ids]
    ranked_contexts = _rank_domain_contexts(normalized_sector_ids)

    return {
        "selected_sector_ids": normalized_sector_ids,
        "sector_config": sector_config,
        "skax_profile": _SKAX_OFFICIAL_PROFILE,
        "skax_documents": official_documents or [],
        "skax_newsroom_documents": newsroom_documents or [],
        "skax_contexts": ranked_contexts,
    }


def load_skax_official_documents(
    limit: int | None = None,
) -> list[SKAXOfficialDocument]:
    """DB에 저장된 SK AX 공식 프로필/서비스 문서를 로드한다."""
    limit_clause = "\n                    LIMIT :limit" if limit is not None else ""
    params: dict[str, Any] = {"limit": limit} if limit is not None else {}
    try:
        with SessionLocal() as db:
            rows = db.execute(
                text(f"""
                    SELECT ra.id, ra.title, ra.content, ra.url,
                           COALESCE(mu.metadata, '{{}}'::jsonb) AS metadata
                    FROM raw_articles ra
                    LEFT JOIN raw_article_metadata_unified mu
                        ON mu.raw_article_id = ra.id
                    WHERE ra.source_type = 'company_site'
                      AND (
                        ra.source_name = 'SK AX Site'
                        OR ra.publisher = 'SK AX'
                        OR mu.metadata ->> 'source_family' = 'sk_ax_site'
                      )
                    ORDER BY
                      CASE mu.metadata ->> 'page_kind'
                        WHEN 'company_about' THEN 0
                        WHEN 'service' THEN 1
                        WHEN 'industry' THEN 2
                        WHEN 'insight' THEN 3
                        ELSE 4
                      END,
                      ra.collected_at DESC NULLS LAST,
                      ra.id DESC
                    {limit_clause}
                """),
                params,
            ).fetchall()
    except Exception as exc:
        log.warning("SK AX 공식 문서 로드 실패, 정적 프로필로 대체 | error=%s", exc)
        return []

    documents: list[SKAXOfficialDocument] = []
    for row in rows:
        item = dict(row._mapping)
        metadata_value = item.get("metadata")
        metadata: dict[str, Any] = metadata_value if isinstance(metadata_value, dict) else {}
        documents.append(
            {
                "id": int(item["id"]),
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "page_kind": str(metadata.get("page_kind") or ""),
                "headings": _normalize_string_list(metadata.get("headings")),
                "content_excerpt": _compact_text(
                    str(item.get("content") or ""), limit=_PROFILE_DOC_CONTENT_LIMIT
                ),
            }
        )

    return documents


def load_skax_newsroom_documents(
    limit: int | None = None,
) -> list[SKAXOfficialDocument]:
    """company_news 크롤러가 저장한 SK AX 공식 뉴스룸 문서를 로드한다."""
    limit_clause = "\n                    LIMIT :limit" if limit is not None else ""
    params: dict[str, Any] = {"limit": limit} if limit is not None else {}
    try:
        with SessionLocal() as db:
            rows = db.execute(
                text(f"""
                    SELECT ra.id, ra.title, ra.content, ra.url,
                           COALESCE(mu.metadata, '{{}}'::jsonb) AS metadata
                    FROM raw_articles ra
                    LEFT JOIN raw_article_metadata_unified mu
                        ON mu.raw_article_id = ra.id
                    WHERE ra.source_type = 'official'
                      AND ra.source_name = 'SK AX Newsroom'
                      AND ra.publisher = 'SK AX'
                    ORDER BY ra.published_at DESC NULLS LAST,
                             ra.collected_at DESC NULLS LAST,
                             ra.id DESC
                    {limit_clause}
                """),
                params,
            ).fetchall()
    except Exception as exc:
        log.warning("SK AX 뉴스룸 문서 로드 실패 | error=%s", exc)
        return []

    documents: list[SKAXOfficialDocument] = []
    for row in rows:
        item = dict(row._mapping)
        metadata_value = item.get("metadata")
        metadata: dict[str, Any] = metadata_value if isinstance(metadata_value, dict) else {}
        documents.append(
            {
                "id": int(item["id"]),
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "page_kind": "official_news",
                "headings": _normalize_string_list(metadata.get("headings")),
                "content_excerpt": _compact_text(
                    str(item.get("content") or ""), limit=_PROFILE_DOC_CONTENT_LIMIT
                ),
            }
        )

    return documents


def _sector_ids_from_inputs(
    issue_card: dict[str, Any],
    summary: dict[str, Any],
    analysis: dict[str, Any],
    classification: dict[str, Any],
) -> list[str]:
    candidates: list[str] = []
    for payload in (classification, issue_card, analysis, summary):
        candidates.extend(_normalize_string_list(payload.get("sectors")))
        sector = payload.get("sector") or payload.get("primary_sector")
        if sector:
            candidates.append(str(sector))
        basis = payload.get("basis")
        if isinstance(basis, dict):
            candidates.extend(_normalize_string_list(basis.get("sectors")))
            if basis.get("sector"):
                candidates.append(str(basis["sector"]))
    return _normalize_sector_ids(candidates)


def _normalize_sector_ids(sector_ids: list[str] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for sector_id in sector_ids or []:
        candidate = str(sector_id or "").strip().lower()
        if candidate and candidate in SECTOR_IDS and candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)

    return normalized or ["other"]


def _sector_context(sector_id: str) -> SKAXSectorContext:
    if sector_id == "other":
        return {"id": "other", "name_ko": sector_name_ko("other"), "keywords": []}

    info = SECTOR_KEYWORDS.get(sector_id)
    return {
        "id": sector_id,
        "name_ko": sector_name_ko(sector_id),
        "keywords": list(info["keywords"]) if info else [],
    }


def _rank_domain_contexts(sector_ids: list[str]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    sector_set = set(sector_ids)
    for context in _SKAX_DOMAIN_CONTEXTS:
        configured_sector_ids = [sector for sector in context["sector_ids"] if sector in SECTOR_IDS]
        matched_sector_ids = sorted(sector_set.intersection(configured_sector_ids))
        ranked.append(
            {
                **context,
                "sector_ids": configured_sector_ids,
                "matched_sector_ids": matched_sector_ids,
                "priority": "primary" if matched_sector_ids else "supporting",
            }
        )

    return sorted(
        ranked,
        key=lambda item: (
            0 if item["priority"] == "primary" else 1,
            item["domain_id"],
        ),
    )


def _issue_card_for_prompt(issue_card: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": issue_card.get("id"),
        "company": issue_card.get("company", ""),
        "title": issue_card.get("title", ""),
        "summary": issue_card.get("summary") or issue_card.get("summary_lines") or [],
        "event_type": issue_card.get("event_type", ""),
        "sector": issue_card.get("sector", ""),
        "sectors": issue_card.get("sectors", []),
        "importance_score": issue_card.get("importance_score"),
        "cluster_id": issue_card.get("cluster_id"),
        "sources": issue_card.get("sources", []),
    }


def _summary_for_prompt(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": summary.get("cluster_id"),
        "main_company": summary.get("main_company", ""),
        "headline": summary.get("headline", ""),
        "one_line_summary": summary.get("one_line_summary", ""),
        "fact_summary": summary.get("fact_summary", []),
        "main_event": summary.get("main_event", ""),
        "source_article_ids": summary.get("source_article_ids", []),
        "confidence": summary.get("confidence", 0.0),
    }


def _analysis_for_prompt(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "analysis_summary": analysis.get("analysis_summary", ""),
        "strategic_meaning": analysis.get("strategic_meaning", []),
        "market_signal": analysis.get("market_signal", ""),
        "impact_level": analysis.get("impact_level", ""),
        "risk_or_opportunity": analysis.get("risk_or_opportunity", ""),
        "confidence": analysis.get("confidence", 0.0),
    }


def _classification_for_prompt(classification: dict[str, Any]) -> dict[str, Any]:
    return {
        "sector": classification.get("sector", ""),
        "sectors": classification.get("sectors", []),
        "event_type": classification.get("event_type", ""),
        "exposure_band": classification.get("exposure_band", ""),
        "exposure_score": classification.get("exposure_score", 0.0),
        "importance": classification.get("importance", ""),
        "importance_score": classification.get("importance_score", 0.0),
        "signals": classification.get("signals", {}),
    }


def _documents_for_prompt(
    documents: list[SKAXOfficialDocument],
    *,
    content_limit: int,
) -> list[SKAXOfficialDocument]:
    """로드된 공식 문서는 모두 쓰되, LLM 입력용 본문만 짧게 압축한다."""
    prompt_documents: list[SKAXOfficialDocument] = []
    for document in documents:
        prompt_documents.append(
            {
                "id": document["id"],
                "title": document["title"],
                "url": document["url"],
                "page_kind": document["page_kind"],
                "headings": document["headings"][:5],
                "content_excerpt": _compact_text(
                    document["content_excerpt"],
                    limit=content_limit,
                ),
            }
        )

    return prompt_documents


def _source_intelligence_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    """공식 자료에서 관점 생성에 바로 쓸 수 있는 고밀도 근거를 추출한다."""
    official_documents = [
        document for document in context.get("skax_documents", []) if isinstance(document, dict)
    ]
    newsroom_documents = [
        document
        for document in context.get("skax_newsroom_documents", [])
        if isinstance(document, dict)
    ]

    return {
        "usage": (
            "아래 points는 공식 사이트/뉴스룸 본문에서 추출한 구체 근거입니다. "
            "company_viewpoint에는 이 points의 작동 방식, 고객군, 기대 효과를 반영하세요."
        ),
        "official_capability_briefs": _document_briefs(
            official_documents,
            limit=_SOURCE_INTELLIGENCE_DOC_LIMIT,
        ),
        "newsroom_execution_briefs": _document_briefs(
            newsroom_documents,
            limit=len(newsroom_documents),
        ),
    }


def _document_briefs(
    documents: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    briefs: list[dict[str, Any]] = []
    for document in documents[:limit]:
        points = _extract_document_points(
            str(document.get("content_excerpt") or ""),
            headings=_normalize_string_list(document.get("headings")),
        )
        if not points:
            points = [_compact_text(str(document.get("content_excerpt") or ""), limit=320)]
        briefs.append(
            {
                "title": str(document.get("title") or "").strip(),
                "page_kind": str(document.get("page_kind") or "").strip(),
                "url": str(document.get("url") or "").strip(),
                "points": points,
            }
        )

    return briefs


def _extract_document_points(text: str, *, headings: list[str]) -> list[str]:
    points = [_compact_text(heading, limit=220) for heading in headings[:3]]
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。])\s+", _compact_text(text, limit=2200))
        if sentence.strip()
    ]
    for sentence in sentences[: _SOURCE_INTELLIGENCE_POINT_LIMIT - len(points)]:
        points.append(_compact_text(sentence, limit=360))

    return _dedupe_strings(points)


def _normalize_perspective(
    data: dict[str, Any],
    context: dict[str, Any],
) -> SKAXPerspectiveResult:
    selected_domain_ids = {
        domain["domain_id"]
        for domain in context.get("skax_contexts", [])
        if domain.get("domain_id")
    }
    requested_domain_ids = _normalize_string_list(data.get("matched_skax_domains"))
    matched_domain_ids = [
        domain_id
        for domain_id in requested_domain_ids
        if not selected_domain_ids or domain_id in selected_domain_ids
    ]
    if not matched_domain_ids and selected_domain_ids:
        matched_domain_ids = sorted(selected_domain_ids)[:2]

    evidence = _normalize_evidence(data.get("evidence"))
    if not evidence:
        evidence = _default_evidence(context, matched_domain_ids)

    why_important = str(data.get("why_important") or data.get("why_it_matters") or "").strip()
    potential_impact = str(data.get("potential_impact") or "").strip()
    if not potential_impact:
        potential_impact = _join_nonempty(
            [
                str(data.get("competitive_implication") or ""),
                str(data.get("opportunity") or ""),
                str(data.get("risk") or ""),
            ]
        )

    return {
        "is_relevant_to_skax": bool(data.get("is_relevant_to_skax", True)),
        "perspective_scope": "sk_ax_official",
        "matched_skax_domains": matched_domain_ids,
        "why_important": why_important,
        "potential_impact": potential_impact,
        "follow_up_questions": _normalize_string_list(data.get("follow_up_questions"))[:3],
        "why_it_matters": str(data.get("why_it_matters") or why_important).strip(),
        "competitive_implication": str(data.get("competitive_implication") or "").strip(),
        "opportunity": str(data.get("opportunity") or "").strip(),
        "risk": str(data.get("risk") or "").strip(),
        "suggested_actions": _normalize_string_list(data.get("suggested_actions"))[:3],
        "confidence": _clamp_float(data.get("confidence"), default=0.0),
        "reason": str(data.get("reason") or "").strip(),
        "evidence": evidence[:4],
    }


def _normalize_viewpoint(data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    selected_sector_ids = _normalize_string_list(context.get("selected_sector_ids"))
    company_viewpoint = _normalize_company_viewpoint(data.get("company_viewpoint"), context)

    return {
        "perspective_scope": "sk_ax_profile",
        "selected_sector_ids": selected_sector_ids,
        "source_basis": {
            "profile_doc_count": len(context.get("skax_documents", [])),
            "newsroom_doc_count": len(context.get("skax_newsroom_documents", [])),
        },
        "company_viewpoint": company_viewpoint,
        "sector_viewpoints": _normalize_sector_viewpoints(
            data.get("sector_viewpoints"),
            context,
        ),
        "cross_sector_signals": _normalize_cross_sector_signals(
            data.get("cross_sector_signals"),
            context,
        ),
        "evidence_summary": _normalize_evidence_summary(data.get("evidence_summary"), context),
        "confidence": _clamp_float(data.get("confidence"), default=0.0),
        "reason": str(data.get("reason") or "").strip(),
    }


def _fallback_viewpoint(context: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "perspective_scope": "sk_ax_profile",
        "selected_sector_ids": _normalize_string_list(context.get("selected_sector_ids")),
        "source_basis": {
            "profile_doc_count": len(context.get("skax_documents", [])),
            "newsroom_doc_count": len(context.get("skax_newsroom_documents", [])),
        },
        "company_viewpoint": _fallback_company_viewpoint(context),
        "sector_viewpoints": _fallback_sector_viewpoints(context),
        "cross_sector_signals": _fallback_cross_sector_signals(context),
        "evidence_summary": _fallback_evidence_summary(context),
        "confidence": 0.55,
        "reason": reason,
    }


def _normalize_company_viewpoint(value: Any, context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _fallback_company_viewpoint(context)

    viewpoint = {
        "positioning": str(value.get("positioning") or "").strip(),
        "official_basis": _normalize_string_list(value.get("official_basis"))[:6],
        "current_focus": _normalize_string_list(value.get("current_focus"))[:6],
        "recent_execution": _normalize_string_list(value.get("recent_execution"))[:6],
        "execution_pattern": _normalize_string_list(value.get("execution_pattern"))[:6],
        "strategic_direction": _normalize_string_list(value.get("strategic_direction"))[:6],
        "execution_priorities": _normalize_string_list(value.get("execution_priorities"))[:6],
        "default_implication_lens": _normalize_string_list(value.get("default_implication_lens"))[
            :6
        ],
        "default_response_principles": _normalize_string_list(
            value.get("default_response_principles")
        )[:6],
        "card_usage_guidance": _normalize_string_list(value.get("card_usage_guidance"))[:6],
        "evidence": _normalize_evidence(value.get("evidence"))[:5],
    }

    if not viewpoint["positioning"]:
        viewpoint["positioning"] = _fallback_company_viewpoint(context)["positioning"]
    if not viewpoint["official_basis"]:
        viewpoint["official_basis"] = _fallback_company_viewpoint(context)["official_basis"]
    if not viewpoint["execution_pattern"]:
        viewpoint["execution_pattern"] = _fallback_company_viewpoint(context)["execution_pattern"]
    if not viewpoint["card_usage_guidance"]:
        viewpoint["card_usage_guidance"] = _fallback_company_viewpoint(context)[
            "card_usage_guidance"
        ]

    return viewpoint


def _fallback_company_viewpoint(context: dict[str, Any]) -> dict[str, Any]:
    profile = context.get("skax_profile", {})
    current_state = _fallback_current_state(context)
    return {
        "positioning": str(profile.get("positioning") or profile.get("default_viewpoint") or ""),
        "official_basis": [
            (
                "공식 사이트 문서는 SK AX가 AI 기술과 산업 전문성을 결합해 제조, 금융, "
                "공공, 클라우드 등 여러 산업의 AX 혁신을 지원한다고 설명합니다."
            ),
            (
                "뉴스룸 문서는 AXgenticWire, 금융 인프라 운영 혁신, SKALA 같은 실행 사례를 "
                "통해 전략이 실제 고객 적용과 인재 확산으로 이어지고 있음을 보여줍니다."
            ),
        ],
        "current_focus": current_state["what_skax_is_doing"],
        "recent_execution": current_state["recent_execution_signals"],
        "execution_pattern": [
            (
                "최근 실행 사례는 Agentic AI를 단순 기능이 아니라 금융 인프라 운영, 장애 예방, "
                "운영 자동화처럼 고객 운영 체계에 적용하는 방향을 보여줍니다."
            ),
            (
                "인재 육성 뉴스는 직접 수주 신호라기보다 AX 확산을 위한 공급 기반과 "
                "생태계 구축 신호로 해석합니다."
            ),
        ],
        "strategic_direction": [
            "Agentic AI 기반 기업 운영 혁신을 핵심 방향으로 본다.",
            "산업별 AX 서비스와 클라우드/운영 역량을 결합해 실행 사례를 확장한다.",
            "AI 인재 육성과 실무형 교육은 AX 확산 기반으로 보조 반영한다.",
        ],
        "execution_priorities": [
            "고객 적용, 계약, 운영 혁신처럼 실제 실행 신호가 있는 이슈를 우선한다.",
            "공식 서비스 포트폴리오와 연결되는 산업별 AX 레퍼런스를 우선한다.",
            "교육/인재 뉴스는 핵심 사업 대응보다 AX 확산 기반으로 해석한다.",
        ],
        "default_implication_lens": [
            (
                "외부 이슈에 운영 자동화, 금융/공공 보안 요구, 산업별 AX 적용 사례가 "
                "있으면 SK AX의 AXgenticWire와 산업 AX 확장 관점으로 해석합니다."
            ),
            (
                "AI 인재, 교육, 조직 전환 신호가 보이면 SKALA와 AI Talent Lab을 "
                "AX 확산 기반으로 연결할 수 있는지 확인합니다."
            ),
            (
                "공식 자료로 확인되는 서비스나 뉴스룸 실행 사례와 연결되지 않는 이슈는 "
                "사업 기회보다 모니터링 신호로 낮춥니다."
            ),
        ],
        "default_response_principles": [
            (
                "금융/공공/제조 고객군에서 유사 운영 문제가 보이면 AXgenticWire NPO, "
                "AIOps, 보안/운영 메시지로 연결해 검토합니다."
            ),
            (
                "산업별 AX 적용 사례는 SK AX의 서비스 포트폴리오와 맞는 레퍼런스 "
                "패키징 또는 제안 메시지로 정리합니다."
            ),
            (
                "주가나 시장 반응 중심 이슈는 실제 계약, 고객명, 적용 기술이 확인될 때만 "
                "대응 후보로 올리고 그 전에는 관찰로 둡니다."
            ),
        ],
        "card_usage_guidance": [
            (
                "경쟁사 뉴스에 고객 적용, 계약, 운영 자동화, 클라우드/보안 요구가 있으면 "
                "SK AX의 AXgenticWire, AIOps, 산업 AX 레퍼런스와 연결해 시사점을 만듭니다."
            ),
            (
                "경쟁사 뉴스가 단순 주가 반응이나 일반 전망이면 공식 역량과 직접 연결되는 "
                "실행 근거가 있는지 먼저 확인하고, 없으면 대응보다 모니터링으로 낮춥니다."
            ),
        ],
        "evidence": _document_evidence(context, limit=5),
    }


def _normalize_current_state(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {
            "what_skax_is_doing": [],
            "recent_execution_signals": [],
            "capability_status": [],
        }

    return {
        "what_skax_is_doing": _normalize_string_list(value.get("what_skax_is_doing"))[:5],
        "recent_execution_signals": _normalize_string_list(value.get("recent_execution_signals"))[
            :5
        ],
        "capability_status": _normalize_string_list(value.get("capability_status"))[:5],
    }


def _fallback_current_state(context: dict[str, Any]) -> dict[str, list[str]]:
    newsroom_titles = [
        str(document.get("title") or "").strip()
        for document in context.get("skax_newsroom_documents", [])
        if isinstance(document, dict) and str(document.get("title") or "").strip()
    ]
    profile_titles = [
        str(document.get("title") or "").strip()
        for document in context.get("skax_documents", [])
        if isinstance(document, dict) and str(document.get("title") or "").strip()
    ]

    return {
        "what_skax_is_doing": [
            (
                "Agentic AI, AI Workforce, AI Software Engineering 등 AX 서비스를 "
                "공식 서비스 축으로 제시한다."
            ),
            "금융/제조/공공 등 산업별 AX 적용과 운영 혁신을 주요 사업 영역으로 설명한다.",
        ],
        "recent_execution_signals": newsroom_titles[:5],
        "capability_status": profile_titles[:5],
    }


def _normalize_sector_viewpoints(value: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return _fallback_sector_viewpoints(context)

    selected_sector_ids = set(_normalize_string_list(context.get("selected_sector_ids")))
    viewpoints: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        sector_id = str(item.get("sector_id") or "").strip().lower()
        if not sector_id or (selected_sector_ids and sector_id not in selected_sector_ids):
            continue
        strength = str(item.get("viewpoint_strength") or "").strip().lower()
        if strength not in {"strong", "medium", "weak"}:
            strength = "medium"
        viewpoints.append(
            {
                "sector_id": sector_id,
                "sector_name": str(item.get("sector_name") or sector_name_ko(sector_id)).strip(),
                "viewpoint_strength": strength,
                "skax_stance": str(item.get("skax_stance") or "").strip(),
                "what_skax_has": _normalize_string_list(item.get("what_skax_has"))[:5],
                "opportunity_lens": _normalize_string_list(item.get("opportunity_lens"))[:5],
                "risk_lens": _normalize_string_list(item.get("risk_lens"))[:5],
                "response_lens": _normalize_string_list(item.get("response_lens"))[:5],
            }
        )

    if not viewpoints:
        return _fallback_sector_viewpoints(context)

    existing_sector_ids = {viewpoint["sector_id"] for viewpoint in viewpoints}
    missing_sector_ids = selected_sector_ids.difference(existing_sector_ids)
    if missing_sector_ids:
        fallback_viewpoints = {
            viewpoint["sector_id"]: viewpoint for viewpoint in _fallback_sector_viewpoints(context)
        }
        for sector_id in sorted(missing_sector_ids):
            fallback = fallback_viewpoints.get(sector_id)
            if fallback:
                viewpoints.append(fallback)

    return viewpoints


def _fallback_sector_viewpoints(context: dict[str, Any]) -> list[dict[str, Any]]:
    selected_sector_ids = _normalize_string_list(context.get("selected_sector_ids")) or ["other"]
    documents = context.get("skax_documents", [])
    newsroom_documents = context.get("skax_newsroom_documents", [])
    doc_titles = [
        str(document.get("title") or "").strip()
        for document in documents
        if isinstance(document, dict) and str(document.get("title") or "").strip()
    ]
    newsroom_titles = [
        str(document.get("title") or "").strip()
        for document in newsroom_documents
        if isinstance(document, dict) and str(document.get("title") or "").strip()
    ]

    viewpoints: list[dict[str, Any]] = []
    for sector_id in selected_sector_ids:
        matching_domains = [
            domain
            for domain in context.get("skax_contexts", [])
            if isinstance(domain, dict)
            and sector_id in _normalize_string_list(domain.get("sector_ids"))
        ]
        capabilities: list[str] = []
        domain_titles: list[str] = []
        domain_perspectives: list[str] = []
        for domain in matching_domains:
            capabilities.extend(_normalize_string_list(domain.get("capabilities")))
            if domain.get("title"):
                domain_titles.append(str(domain["title"]))
            if domain.get("perspective"):
                domain_perspectives.append(str(domain["perspective"]))

        stance = (
            " / ".join(domain_perspectives[:2])
            or "공식 서비스와 뉴스룸 실행 사례를 기준으로 보수적으로 판단합니다."
        )
        opportunity_lens = [
            f"{title}와 연결되는 고객 적용, 계약, 운영 혁신 신호" for title in domain_titles[:3]
        ] or [
            "고객 적용, 계약, 운영 혁신처럼 실제 실행 신호가 확인되는 이슈",
            "공식 서비스 포트폴리오와 연결되는 산업별 AX 레퍼런스 이슈",
        ]

        viewpoints.append(
            {
                "sector_id": sector_id,
                "sector_name": sector_name_ko(sector_id),
                "viewpoint_strength": "medium" if doc_titles or newsroom_titles else "weak",
                "skax_stance": stance,
                "what_skax_has": _dedupe_strings(capabilities + doc_titles + newsroom_titles)[:5],
                "opportunity_lens": opportunity_lens,
                "risk_lens": [
                    "공식 역량과 연결 근거가 약한 단순 시장 반응 또는 주가성 이슈",
                    "피어사 실행 사례가 빠르게 늘어 SK AX 메시지 차별화가 필요한 이슈",
                    "고객명, 적용 기술, 계약 범위가 불명확해 추가 확인이 필요한 이슈",
                ],
                "response_lens": [
                    "유사 고객군에 적용 가능한 SK AX 서비스와 공식 레퍼런스를 연결합니다.",
                    "사업 실행 뉴스는 AXgenticWire/AIOps/산업 AX 메시지로 우선 검토합니다.",
                    "인재 육성 뉴스는 핵심 사업 대응보다 AX 확산 기반으로 보조 반영합니다.",
                ],
            }
        )

    return viewpoints


def _normalize_cross_sector_signals(value: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return _fallback_cross_sector_signals(context)

    valid_sector_ids = set(_normalize_string_list(context.get("selected_sector_ids")))
    signals: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        signal = str(item.get("signal") or "").strip()
        if not signal:
            continue
        related_sector_ids = [
            sector_id
            for sector_id in _normalize_string_list(item.get("related_sector_ids"))
            if not valid_sector_ids or sector_id in valid_sector_ids
        ]
        signals.append(
            {
                "signal": signal,
                "related_sector_ids": related_sector_ids,
                "why_it_matters": str(item.get("why_it_matters") or "").strip(),
                "response_hint": str(item.get("response_hint") or "").strip(),
            }
        )

    return signals[:6] or _fallback_cross_sector_signals(context)


def _fallback_cross_sector_signals(context: dict[str, Any]) -> list[dict[str, Any]]:
    selected_sector_ids = _normalize_string_list(context.get("selected_sector_ids"))
    return [
        {
            "signal": "Agentic AI 기반 운영 자동화 계약 또는 고객 적용 사례",
            "related_sector_ids": [
                sector for sector in ["ax", "infra", "security"] if sector in selected_sector_ids
            ],
            "why_it_matters": "AX 서비스, 운영 인프라, 보안/안정성 메시지가 함께 연결됩니다.",
            "response_hint": (
                "AXgenticWire NPO, AIOps, 클라우드 운영 관점의 적용 가능성을 검토합니다."
            ),
        },
        {
            "signal": "공공/금융/제조 산업의 대형 AX 또는 인프라 전환 사업",
            "related_sector_ids": [
                sector for sector in ["ax", "infra", "deal"] if sector in selected_sector_ids
            ],
            "why_it_matters": "산업별 AX 레퍼런스와 수주/계약 관점이 동시에 걸립니다.",
            "response_hint": "유사 산업 레퍼런스, 제안 메시지, 보안/운영 요구를 함께 정리합니다.",
        },
    ]


def _normalize_evidence_summary(value: Any, context: dict[str, Any]) -> dict[str, Any]:
    fallback = _fallback_evidence_summary(context)
    if not isinstance(value, dict):
        return fallback

    return {
        "profile_doc_count": len(context.get("skax_documents", [])),
        "newsroom_doc_count": len(context.get("skax_newsroom_documents", [])),
        "strongest_sources": _normalize_string_list(value.get("strongest_sources"))[:8]
        or fallback["strongest_sources"],
    }


def _fallback_evidence_summary(context: dict[str, Any]) -> dict[str, Any]:
    strongest_sources: list[str] = []
    for document in context.get("skax_documents", [])[:5]:
        if isinstance(document, dict) and document.get("title"):
            strongest_sources.append(str(document["title"]))
    for document in context.get("skax_newsroom_documents", [])[:5]:
        if isinstance(document, dict) and document.get("title"):
            strongest_sources.append(str(document["title"]))

    return {
        "profile_doc_count": len(context.get("skax_documents", [])),
        "newsroom_doc_count": len(context.get("skax_newsroom_documents", [])),
        "strongest_sources": _dedupe_strings(strongest_sources)[:8],
    }


def _default_evidence(context: dict[str, Any], domain_ids: list[str]) -> list[OfficialSource]:
    return _document_evidence(context, limit=max(len(domain_ids), 4))


def _document_evidence(context: dict[str, Any], *, limit: int) -> list[OfficialSource]:
    evidence: list[OfficialSource] = []
    for source_key, basis in (
        ("skax_newsroom_documents", "DB에 저장된 SK AX 공식 뉴스룸 문서"),
        ("skax_documents", "DB에 저장된 SK AX 공식 사이트 문서"),
    ):
        for document in context.get(source_key, []):
            if not isinstance(document, dict):
                continue
            title = str(document.get("title") or "").strip()
            url = str(document.get("url") or "").strip()
            if title and url:
                evidence.append({"title": title, "url": url, "basis": basis})
            if len(evidence) >= limit:
                return evidence

    return evidence


def _normalize_evidence(value: Any) -> list[OfficialSource]:
    if not isinstance(value, list):
        return []

    evidence: list[OfficialSource] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        basis = str(item.get("basis") or "").strip()
        if title and url:
            evidence.append({"title": title, "url": url, "basis": basis})
    return evidence


def _empty_perspective(reason: str) -> SKAXPerspectiveResult:
    return {
        "is_relevant_to_skax": False,
        "perspective_scope": "sk_ax_official",
        "matched_skax_domains": [],
        "why_important": "",
        "potential_impact": "",
        "follow_up_questions": [],
        "why_it_matters": "",
        "competitive_implication": "",
        "opportunity": "",
        "risk": "",
        "suggested_actions": [],
        "confidence": 0.0,
        "reason": reason,
        "evidence": [],
    }


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    return parsed if isinstance(parsed, dict) else {}


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _compact_text(value: str, limit: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[:limit].rstrip() + "..."


def _join_nonempty(values: list[str]) -> str:
    return " ".join(value.strip() for value in values if value.strip())


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)


__all__ = [
    "SKAXPerspectiveAgent",
    "SKAXProfileLoader",
    "build_skax_context",
    "load_skax_newsroom_documents",
    "load_skax_official_documents",
]
