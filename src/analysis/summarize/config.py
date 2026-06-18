"""summarize config — extracted from facade (move-only)."""

# ruff: noqa: E501  — long prompt-string lines (exempt in original facade)

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from src.config.companies import COMPANY_ALIASES
from src.config.company_tiers import company_tier
from src.config.global_companies import GLOBAL_COMPANY_ALIASES

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        log.warning("정수 환경변수 파싱 실패, 기본값 사용 | name=%s default=%s", name, default)
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        log.warning("실수 환경변수 파싱 실패, 기본값 사용 | name=%s default=%s", name, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


_LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"


_PROMPT_VERSION = "summary-v4.0"


_USE_FACT_EXTRACTION_LLM = _env_bool("NEWS_SUMMARY_USE_FACT_EXTRACTION_LLM", False)


_FACT_EXTRACTION_MODE = (
    os.getenv(
        "NEWS_SUMMARY_FACT_EXTRACTION_MODE",
        "llm" if _USE_FACT_EXTRACTION_LLM else "adaptive",
    )
    .strip()
    .lower()
)


_FACT_EXTRACTION_BATCH_SIZE = 10


_FACT_EXTRACTION_MAX_TOKENS = _env_int("FACT_EXTRACTION_MAX_TOKENS", 3000)


_SUMMARY_MAX_TOKENS = _env_int("SUMMARY_MAX_TOKENS", 1500)


_VALIDATION_MAX_TOKENS = _env_int("VALIDATION_MAX_TOKENS", 1200)


_SUMMARY_LINE_MIN = 3


_SUMMARY_LINE_MAX = 5


_ARTICLE_CONTENT_CHARS = _env_int("NEWS_SUMMARY_ARTICLE_CONTENT_CHARS", 2400)


_COMPACT_ARTICLE_CONTENT_CHARS = _env_int("NEWS_SUMMARY_COMPACT_ARTICLE_CONTENT_CHARS", 1200)


_FULL_TEXT_ARTICLE_LIMIT = _env_int("NEWS_SUMMARY_FULL_TEXT_ARTICLE_LIMIT", 3)


_MIN_ANALYZED_ARTICLES = _env_int("NEWS_SUMMARY_MIN_ANALYZED_ARTICLES", 8)


_MAX_ANALYZED_ARTICLES = _env_int("NEWS_SUMMARY_MAX_ANALYZED_ARTICLES", 10)


_MAJORITY_THRESHOLD = _env_float("NEWS_SUMMARY_MAJORITY_THRESHOLD", 0.70)


_MIXED_THRESHOLD = _env_float("NEWS_SUMMARY_MIXED_THRESHOLD", 0.50)


_SUPPORTING_ARTICLE_CONTENT_CHARS = _env_int("NEWS_SUMMARY_SUPPORTING_ARTICLE_CONTENT_CHARS", 0)


_NEAR_DUPLICATE_SIMILARITY = _env_float("NEWS_SUMMARY_NEAR_DUPLICATE_SIMILARITY", 0.86)


_SNIPPETS_PER_ARTICLE = _env_int("NEWS_SUMMARY_SNIPPETS_PER_ARTICLE", 4)


_SNIPPET_CANDIDATE_SENTENCES = _env_int("NEWS_SUMMARY_SNIPPET_CANDIDATE_SENTENCES", 40)


_SNIPPET_DEDUP_SIMILARITY = _env_float("NEWS_SUMMARY_SNIPPET_DEDUP_SIMILARITY", 0.88)


_EVENT_TYPES = (
    "contract",
    "partnership",
    "launch",
    "earnings",
    "stock_market",
    "analyst_report",
    "investment",
    "hiring",
    "organization",
    "risk",
    "regulation",
    "technology_update",
    "general_update",
    "unknown",
)


_EVENT_TYPE_VALUES = "|".join(_EVENT_TYPES)


_ALLOWED_EVIDENCE_TYPES = {
    "core_fact",
    "unique_fact",
    "common_fact",
    "uncertain_fact",
    "reported_fact",
    "numeric_fact",
    "market_reaction_fact",
    "risk_fact",
}


_FACT_TYPES = {
    "launch_fact",
    "platform_definition_fact",
    "application_fact",
    "numeric_fact",
    "market_fact",
    "risk_fact",
    "uncertain_fact",
    "general_fact",
}


_SUMMARY_ROLES = {
    "main_event",
    "product_definition",
    "service_function",
    "application_case",
    "numeric_effect",
    "market_reaction",
    "risk_detail",
    "uncertainty_detail",
}


_NUMBER_TOKEN_PATTERN = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:억원|조원|만원|원|달러|%|퍼센트|건|명|개|대|년|월|일|분기|개월|주|일|시간|배|곳|개사)?"
)


_ARTICLE_UI_BOILERPLATE_MARKERS = (
    "뉴스 듣기",
    "글자 크기",
    "기사 공유",
    "주소복사",
    "다크모드",
    "프린트",
    "채널구독",
    "네이버 채널구독",
    "다음 채널구독",
    "페이스북",
    "카카오톡",
    "이메일 주소복사",
    "북마크",
)


_PEER_ALIASES = {
    company_id: aliases
    for company_id, aliases in {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}.items()
    if company_tier(company_id) != "self"
}


_INDUSTRY_TREND_COMPANY_ID = "industry_trend"


_INDUSTRY_TREND_ALIASES = ["산업 트렌드", "AI 산업", "AX 산업", "industry_trend"]


_KNOWN_COMPANY_ALIASES = {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}


_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    from langchain_openai import ChatOpenAI  # lazy: transformers 체인 회피

    global _llm

    if _llm is None:
        # 공용 build_chat_llm 미적용(의도적 예외): 여기는 model+temperature 만 가진
        # base LLM 이고, max_completion_tokens·response_format 은 호출마다 .bind() 로
        # 동적 주입한다(L1126·L2340). 팩토리는 생성 시 캡을 강제하므로 base 구성이
        # 달라진다. 통합할 boilerplate(gpt-5 분기·json_object·고정 캡)도 없어 제외.
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.1,
        )

    return _llm


_ARTICLE_FACT_EXTRACTION_PROMPT = """\
당신은 기사별 팩트 추출 Agent입니다.

목적:
- 같은 클러스터의 각 기사를 하나씩 독립적으로 읽고, 피어사 관련 사실만 추출합니다.
- 기사 번호와 입력 순서는 중요도나 대표성을 뜻하지 않습니다.
- 이 단계는 뉴스 요약 전 fact extraction 단계이며, 시사점/대응방향/전략 해석을 만들지 않습니다.

규칙:
1. 각 article_id마다 기사에 명시된 핵심 사실을 뽑으세요.
2. 기사에 명시된 사실만 쓰고, 시사점·평가·대응 방향은 쓰지 마세요.
3. 수주, 계약, 협약, 출시, 실적, 주가, 증권 리포트 평가, 투자, 채용, 조직개편,
   리스크, 규제, 기술 업데이트, 고객 규모, 사업 규모, 일정, 후속 본사업,
   예정·계획 정보처럼 뉴스 사실 요약에 필요한 사실을 우선하세요.
   계약/수주 기사에서는 계약 상대방, 정확한 사업명·프로젝트명, 계약 금액,
   계약 기간, 최근 매출 대비 비율, 전환·구축 대상 시스템/업무 범위를 우선 추출하세요.
   "코어뱅킹 현대화 웹단말 전환 사업"처럼 업무·시스템 범위가 들어간 명칭을
   "웹단말 공급"처럼 단순 납품으로 축약하지 마세요.
   SPC·컨소시엄·민관 합작·센터 구축 기사에서는 주도 기업, 지분율, 대표/운영 주체,
   구축 지역, 착공/기공식 일정, GPU·데이터센터 규모, 정부·파트너 지분 구조를
   서로 분리해 보존하세요.
4. 피어사의 일반적 정체성, 기존 포지셔닝, 누구나 알 수 있는 배경 설명은 핵심 사실로 쓰지 마세요.
   기사에서 새로 확인되는 역할, 사건, 범위, 수치, 일정, 시설, 고객, 적용 업무를 우선하세요.
5. 여러 기사에 반복되는 문장이라도 각 기사에서 확인한 사실로 기록하세요.
6. 본문에 없는 수치, 제품명, 회사명은 만들지 마세요.
7. activity_type은 반드시 아래 값 중 하나로 분류하세요.
   {event_types}
8. fact_type과 summary_role은 기사 속 의미를 기준으로 분류하세요.
   fact_type 허용값:
   launch_fact|platform_definition_fact|application_fact|numeric_fact|market_fact|risk_fact|uncertain_fact|general_fact
   summary_role 허용값:
   main_event|product_definition|service_function|application_case|numeric_effect|market_reaction|risk_detail|uncertainty_detail
9. 날짜가 포함되어 있어도 수치 자체가 핵심이 아니면 numeric_fact로 분류하지 마세요.
10. 출시/공개/선보임, 제품 정의, 적용 사례, 수치 효과, 시장 반응, 리스크, 불확실성을 서로 구분하세요.
11. 제3자 회사·서비스·고객 사례는 피어사와 직접 계약/협약/도입/수주/공급/공동개발 관계로
    연결된 경우에만 핵심 사실로 추출하세요. 기사 배경이나 시장 예시로만 언급된 제3자 사례는
    application_fact/main_event 후보에서 제외하세요.
12. 한 기사 안에 여러 회사가 나오면 fact의 주어를 원문 주체 그대로 유지하세요.
    삼성전자/삼성SDS/SK하이닉스/LG CNS처럼 서로 다른 회사의 도입·검증·계약 사실을
    main_company나 다른 피어사 사실로 바꿔 쓰지 마세요.

기사 클러스터:
{articles_text}

반드시 valid JSON object로만 응답하세요. ```json 코드블록, 설명문, 주석, JSON 바깥 텍스트는 쓰지 마세요.
문자열 값 안에는 실제 줄바꿈을 넣지 말고 공백으로 정리하세요.
다음 JSON 형식으로만 응답하세요.
{{
  "article_facts": [
    {{
      "article_id": 0,
      "core_facts": [
        {{
          "fact": "기사에서 확인한 핵심 사실",
          "evidence_text": "원문 근거 요약",
          "activity_type": "{event_types}",
          "fact_type": "launch_fact|platform_definition_fact|application_fact|numeric_fact|market_fact|risk_fact|uncertain_fact|general_fact",
          "summary_role": "main_event|product_definition|service_function|application_case|numeric_effect|market_reaction|risk_detail|uncertainty_detail",
          "numbers_and_dates": [],
          "customers_or_industries": [],
          "products_or_services": []
        }}
      ],
      "unique_facts": [
        {{
          "fact": "다른 기사에 없는 보강 사실",
          "evidence_text": "원문 근거 요약",
          "fact_type": "launch_fact|platform_definition_fact|application_fact|numeric_fact|market_fact|risk_fact|uncertain_fact|general_fact",
          "summary_role": "main_event|product_definition|service_function|application_case|numeric_effect|market_reaction|risk_detail|uncertainty_detail",
          "importance_reason": "수치/일정/고객/후속 사업 등 중요한 이유"
        }}
      ],
      "uncertain_facts": [
        {{
          "fact": "예정·전망·계획·가능성 표현",
          "evidence_text": "원문 근거 요약",
          "fact_type": "uncertain_fact",
          "summary_role": "uncertainty_detail",
          "caution": "확정 사실로 쓰지 말아야 하는 이유"
        }}
      ]
    }}
  ]
}}"""


_FACT_ID_SUMMARY_PROMPT = """\
당신은 피어사 뉴스 클러스터를 fact_id 기반으로 요약하는 Agent입니다.

목적:
- 요약 문장을 먼저 만들고 나중에 근거를 찾지 않습니다.
- selected_facts_by_line에 제공된 fact_id와 normalized_fact만 사용해 3~5문장 요약을 만듭니다.
- 시사점, 대응방향, 전략 해석, 회사 프로필 참조는 금지합니다.

공통 규칙:
1. summary_lines는 최소 3개, 최대 5개입니다. 입력 근거가 충분할 때만 4~5번째 문장을 추가하세요.
2. 각 문장은 반드시 fact_ids를 1개 이상 포함해야 합니다.
3. fact_ids는 입력 selected_facts_by_line에 있는 값만 사용하세요.
4. summary line은 연결된 fact_ids의 normalized_fact/evidence_text에서 확인되는 사실만 사용하세요.
5. 기사에 없는 제품명, 서비스명, 고객명, 수치, 날짜, 원인은 만들지 마세요.
6. 수치나 날짜를 쓰려면 연결된 fact의 evidence_text에 같은 수치나 날짜가 있어야 합니다.
7. uncertain_fact를 사용하는 문장은 확정 표현을 피하고 "소개됐다", "언급됐다", "제시됐다", "설명됐다"처럼 원문 수위를 유지하세요.
8. 요약 문장은 같은 내용을 반복하지 말고 서로 다른 역할을 가져야 합니다.
9. 특정 기사 키워드를 규칙처럼 추가하지 말고, 제품명/서비스명/플랫폼명/프로젝트명/이벤트명/기술명 같은 정보 유형을 기준으로 작성하세요.
10. 같은 회사명으로 시작하는 문장은 최대 1개만 두세요. 이후 문장은 의미가 분명하면 제품명/플랫폼명/서비스명/해당 기술 등으로 이어가세요.
    "기사에서는", "사실이 확인됐다" 같은 보고서체 표현은 쓰지 마세요.
11. fact에 구체 수치·개수·기간·범위·장소·현장이 있으면 3~5문장에 우선 반영하되, 연결된 fact evidence_text에서 검증되는 경우에만 쓰세요.
12. 주어와 서술어의 의미 관계를 맞추세요. 회사/기관 주어는 행동·발표·공개를, 제품/서비스/플랫폼/기술 주어는 기능·역할·적용 범위를, 기사/보도/자료 주어는 소개·설명·언급처럼 전달 행위를 서술하세요.
13. 계약/수주 요약에서는 정확한 사업명·프로젝트명과 계약 금액을 가능하면 1문장에 보존하세요.
    2문장은 단순 공급 여부보다 고객 업무/시스템 전환 범위를 보존하세요.
    계약 기간, 최근 매출 대비 비율, 후속 단계가 별도 fact로 있으면 3~5문장에 우선 반영하세요.
    민관 합작/SPC/컨소시엄형 인프라 사업은 단순 참여 여부보다 주도 기업, 지분율,
    대표 선임, 구축·운영 역할, 예정 일정·자원 규모를 우선 반영하세요.
    기사에 "핵심 축", "영향력", "재무 안정성", "사업 관리" 같은 해석이 원문 근거로
    제시되면 요약 3~5문장 안에서 해당 역할 변화를 보존하세요.
14. 계약/협력 기사가 기술·플랫폼·AI 서비스 도입을 다루면, "계약했다"와 "적용 가능하다"만 반복하지 마세요.
    내부 업무에서 무엇을 하게 되는지, 파트너 기술이 어떤 기능을 제공하는지,
    향후 외부 고객/사업 확장과 어떻게 연결되는지를 서로 다른 문장으로 나누어 쓰세요.
15. 제3자 회사·서비스·고객 사례는 피어사와 직접 계약/협약/도입/수주/공급/공동개발 관계로 연결된
    fact_id가 있을 때만 summary_lines에 넣으세요. 본문 배경이나 시장 사례로만 나온 제3자 서비스는
    핵심 변화 3줄 요약에 넣지 말고, 피어사의 발표·제품·계약·고객 업무 범위로 문장을 구성하세요.
16. 연결된 fact_id의 normalized_fact/evidence_text에 있는 회사 주체를 바꾸지 마세요.
    다른 회사의 수치·검증 규모·서비스 선정 사실을 main_company 문장으로 귀속시키면 안 됩니다.
    다중 회사 기사에서는 "A사는 …, B사는 …"처럼 각 사실의 주체가 분명하게 드러나야 합니다.

문장별 역할:
- 1문장: 핵심 사건·상태·평가
- 2문장: 연결된 제품·서비스·플랫폼·기술·업무·고객·산업 영역
- 3문장: 시연·적용 사례·수치·범위·일정·후속 단계·시장 반응·불확실성 중 가장 구체적인 사실
- 4문장: 별도 근거가 있을 때만 추가되는 보강 사실·고객/산업 범위·운영 단계
- 5문장: 별도 근거가 있을 때만 추가되는 수치·기간·후속 단계·불확실성

event_type별 fact 선택 의도:
- launch: 출시/공개/선보임, 제품·플랫폼 기능, 시연·적용 사례·수치·후속 단계
- technology_update/general_update: 기술 확장/변화, 연결 업무·산업·운영 구조, 적용 방향·현장 투입·시연·불확실성
- contract: 수주/계약/사업자 선정, 고객/시스템/업무 영역, 규모/기간/후속 단계
- earnings: 실적 변화, 연결 사업/원인, 수치/기간
- stock_market: 주가/시장 반응, 기사에서 제시한 배경, 등락률/거래량/전망
- risk: 리스크 발생, 연결 시스템/고객/업무, 피해 범위/대응/불확실성

입력:
main_company:
{main_company}

target_peer_companies:
{target_companies_json}

cluster_event_type:
{cluster_event_type}

selected_facts_by_line:
{selected_facts_json}

additional_available_facts:
{all_facts_json}

다음 JSON 형식으로만 응답하세요.
{{
  "is_valid_summary": true,
  "main_company": "{main_company}",
  "mentioned_peer_companies": ["{main_company}"],
  "cluster_event_type": "{event_types}",
  "headline": "피어사 사실 중심 한 문장",
  "one_line_summary": "기사 클러스터의 핵심 사실 1문장",
  "summary_lines": [
    {{
      "line_index": 1,
      "text": "1문장",
      "fact_ids": []
    }},
    {{
      "line_index": 2,
      "text": "2문장",
      "fact_ids": []
    }},
    {{
      "line_index": 3,
      "text": "3문장",
      "fact_ids": []
    }},
    {{
      "line_index": 4,
      "text": "4문장",
      "fact_ids": []
    }},
    {{
      "line_index": 5,
      "text": "5문장",
      "fact_ids": []
    }}
  ],
  "main_event": "기사에 명시된 핵심 사건",
  "confidence": 0.0,
  "reason": "fact_id 기반 요약 근거 또는 invalid 사유"
}}"""
