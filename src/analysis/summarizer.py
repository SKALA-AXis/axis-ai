# 작성일: 2026-05-19
# 작성자: 박지원
# 변경이력:
#   2026-05-19 박지원 — 파서·에이전트 수정으로 시작, 카드 뉴스 품질·근거(provenance)
#   2026-06-05 심유정 — strategic insight grounding 및 카드 dry run 개선
#   2026-06-11 최종민 — ChatOpenAI lazy-import 적용, LLM gen-search 이행 및 summarizer 예외 명문화
"""소스 사실 요약 컴포넌트.

클러스터에 묶인 기사들을 바탕으로 분석 가능한 사실 요약을 생성한다.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import logging
import math
import os
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from src.config.companies import COMPANY_ALIASES
from src.config.company_tiers import company_tier
from src.config.global_companies import GLOBAL_COMPANY_ALIASES
from src.db.article_store import get_articles_by_ids

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


_LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
_PROMPT_VERSION = "summary-v4.0"
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
_MAX_ANALYZED_ARTICLES = _env_int("NEWS_SUMMARY_MAX_ANALYZED_ARTICLES", 20)
_MAJORITY_THRESHOLD = _env_float("NEWS_SUMMARY_MAJORITY_THRESHOLD", 0.70)
_MIXED_THRESHOLD = _env_float("NEWS_SUMMARY_MIXED_THRESHOLD", 0.50)
_SUPPORTING_ARTICLE_CONTENT_CHARS = _env_int("NEWS_SUMMARY_SUPPORTING_ARTICLE_CONTENT_CHARS", 0)
_NEAR_DUPLICATE_SIMILARITY = _env_float("NEWS_SUMMARY_NEAR_DUPLICATE_SIMILARITY", 0.86)
_SNIPPETS_PER_ARTICLE = _env_int("NEWS_SUMMARY_SNIPPETS_PER_ARTICLE", 4)
_SNIPPET_CANDIDATE_SENTENCES = _env_int("NEWS_SUMMARY_SNIPPET_CANDIDATE_SENTENCES", 80)
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

all_available_facts:
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


class SourceSummarizer:
    """클러스터 단위로 source 사실 요약을 생성한다."""

    def summarize(
        self,
        cluster_id: int,
        representative_id: int,
        cluster_article_ids: list[int] | None = None,
        max_cluster_articles: int | None = None,
    ) -> dict[str, Any]:
        """DB의 raw_articles를 읽어 클러스터 사실 요약을 생성한다."""
        ids_to_fetch = _build_fetch_ids(
            representative_id=representative_id,
            cluster_article_ids=cluster_article_ids,
            max_cluster_articles=max_cluster_articles,
        )
        articles = get_articles_by_ids(ids_to_fetch)
        return self.summarize_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=articles,
            cluster_article_ids=cluster_article_ids,
        )

    def summarize_articles(
        self,
        cluster_id: int,
        representative_id: int,
        articles: list[dict[str, Any]],
        cluster_article_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """이미 로드된 기사 목록으로 클러스터 사실 요약을 생성한다."""
        if not articles:
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason="요약할 기사가 없습니다.",
            )

        all_source_article_ids = _article_ids(articles)
        requested_cluster_ids = list(cluster_article_ids or [])
        coverage_warning = ""
        if not requested_cluster_ids:
            coverage_warning = "cluster_article_ids 없음"
        cluster_ids_for_summary = requested_cluster_ids or all_source_article_ids
        selection = _select_analysis_articles(
            articles=articles,
            representative_id=representative_id,
        )
        if selection["status"] == "mixed_cluster_no_majority":
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason="mixed_cluster_no_majority",
                source_article_ids=[],
                cluster_article_ids=cluster_ids_for_summary,
                coverage=_coverage_info(
                    cluster_article_count=len(cluster_ids_for_summary),
                    analyzed_article_count=0,
                    warning=selection["warning"],
                    selection=selection,
                ),
            )

        articles_for_analysis = selection["articles"]
        source_article_ids = _article_ids(articles_for_analysis)
        cluster_article_count = len(cluster_ids_for_summary)
        analyzed_article_count = len(source_article_ids)
        coverage = _coverage_info(
            cluster_article_count=cluster_article_count,
            analyzed_article_count=analyzed_article_count,
            warning=_join_warnings(coverage_warning, selection["warning"]),
            selection=selection,
        )

        target_companies = _candidate_peer_companies(articles_for_analysis)
        if not target_companies:
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason="self 회사를 제외한 피어사 후보를 찾지 못했습니다.",
                source_article_ids=source_article_ids,
                cluster_article_ids=cluster_ids_for_summary,
                coverage=coverage,
            )

        article_fact_notes, fact_extraction_warnings, fact_extraction_failed = (
            _extract_article_fact_notes_batch(
                cluster_id=cluster_id,
                articles=articles_for_analysis,
                target_companies=target_companies,
                representative_id=representative_id,
            )
        )
        merged_facts = _merge_article_facts(article_fact_notes)
        cluster_fact_intelligence = _build_cluster_fact_intelligence(merged_facts)
        cluster_event_type = _classify_cluster_event_type(
            cluster_fact_intelligence,
            articles_for_analysis,
        )
        extracted_facts = _build_extracted_facts(
            cluster_id=cluster_id,
            article_fact_notes=article_fact_notes,
            articles=articles_for_analysis,
            cluster_event_type=cluster_event_type,
        )
        selected_fact_ids = _select_fact_ids_for_summary_lines(
            extracted_facts=extracted_facts,
            cluster_event_type=cluster_event_type,
        )

        try:
            result = _summarize_from_fact_ids(
                cluster_event_type=cluster_event_type,
                extracted_facts=extracted_facts,
                selected_fact_ids=selected_fact_ids,
                target_companies=target_companies,
                main_company=target_companies[0],
            )
            result = _validate_fact_id_summary(
                result=result,
                extracted_facts=extracted_facts,
                source_article_ids=source_article_ids,
                main_company=result.get("main_company", ""),
            )
        except Exception as exc:
            log.error("피어사 뉴스 요약 실패 | cluster=%s error=%s", cluster_id, exc)
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason=f"LLM 요약 실패: {type(exc).__name__}",
                source_article_ids=source_article_ids,
                cluster_article_ids=cluster_ids_for_summary,
                coverage=coverage,
            )

        summary = {
            "cluster_id": cluster_id,
            "representative_id": representative_id,
            "source_article_ids": source_article_ids,
            "cluster_article_ids": cluster_ids_for_summary,
            "analyzed_article_ids": source_article_ids,
            "summary_scope": "peer_company_fact_only",
            "excluded_company_tiers": ["self"],
            "target_peer_companies": target_companies,
            "cluster_fact_intelligence": cluster_fact_intelligence,
            "extracted_facts": extracted_facts,
            "selected_fact_ids": selected_fact_ids,
            "coverage": coverage,
            "model": _LLM_MODEL,
            **result,
        }
        summary = _enrich_peer_comparison_issue(
            summary,
            articles=articles_for_analysis,
            target_companies=target_companies,
        )
        if fact_extraction_warnings:
            summary["validation_warnings"] = _dedupe_keep_order(
                [
                    *_normalize_string_list(summary.get("validation_warnings")),
                    *fact_extraction_warnings,
                ]
            )
        if fact_extraction_failed:
            summary["fact_extraction_failed"] = True
        log.info(
            "피어사 뉴스 요약 완료 | cluster=%s valid=%s company=%s sources=%d",
            cluster_id,
            summary.get("is_valid_summary"),
            summary.get("main_company"),
            len(summary["source_article_ids"]),
        )
        return summary


def _select_analysis_articles(
    *,
    articles: list[dict[str, Any]],
    representative_id: int,
) -> dict[str, Any]:
    total = len(articles)
    limit = _analysis_article_limit(total)
    if total <= limit:
        return {
            "status": "full_cluster",
            "articles": articles,
            "selected_article_ids": _article_ids(articles),
            "excluded_article_ids": [],
            "majority_article_ids": _article_ids(articles),
            "outlier_article_ids": [],
            "majority_ratio": 1.0 if total else 0.0,
            "analysis_article_limit": limit,
            "warning": "",
        }

    groups = _same_event_title_groups(articles)
    majority = max(groups, key=len) if groups else articles
    majority_ids = set(_article_ids(majority))
    majority_ratio = len(majority) / total if total else 0.0
    outlier_ids = [
        article_id for article_id in _article_ids(articles) if article_id not in majority_ids
    ]
    if majority_ratio < _MIXED_THRESHOLD:
        return {
            "status": "mixed_cluster_no_majority",
            "articles": [],
            "selected_article_ids": [],
            "excluded_article_ids": _article_ids(articles),
            "majority_article_ids": _article_ids(majority),
            "outlier_article_ids": outlier_ids,
            "majority_ratio": round(majority_ratio, 4),
            "analysis_article_limit": limit,
            "warning": (
                "mixed_cluster_no_majority: "
                f"majority_ratio={majority_ratio:.2f} threshold={_MIXED_THRESHOLD:.2f}"
            ),
        }

    selected = _diverse_articles_from_same_event_group(
        articles=majority,
        representative_id=representative_id,
        limit=limit,
    )
    selected_ids = set(_article_ids(selected))
    excluded_ids = [
        article_id for article_id in _article_ids(articles) if article_id not in selected_ids
    ]
    status = "sampled_majority_group"
    warning = (
        f"large_cluster_sampled: analyzed={len(selected)} total={total} "
        f"majority_ratio={majority_ratio:.2f}"
    )
    if majority_ratio < _MAJORITY_THRESHOLD:
        status = "sampled_mixed_majority_group"
        warning = (
            f"mixed_cluster_warning: majority_ratio={majority_ratio:.2f} "
            f"threshold={_MAJORITY_THRESHOLD:.2f}; {warning}"
        )
    return {
        "status": status,
        "articles": selected,
        "selected_article_ids": _article_ids(selected),
        "excluded_article_ids": excluded_ids,
        "majority_article_ids": _article_ids(majority),
        "outlier_article_ids": outlier_ids,
        "majority_ratio": round(majority_ratio, 4),
        "analysis_article_limit": limit,
        "warning": warning,
    }


def _analysis_article_limit(total: int) -> int:
    if total <= 0:
        return 0
    max_limit = max(1, _MAX_ANALYZED_ARTICLES)
    min_limit = min(max_limit, max(1, _MIN_ANALYZED_ARTICLES))
    dynamic = int(math.ceil(math.sqrt(total) * 2.5))
    return min(total, max_limit, max(min_limit, dynamic))


def _same_event_title_groups(articles: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(articles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    features = [_title_group_features(article) for article in articles]
    for left_index in range(len(articles)):
        for right_index in range(left_index + 1, len(articles)):
            if _same_title_event(features[left_index], features[right_index]):
                union(left_index, right_index)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, article in enumerate(articles):
        groups.setdefault(find(index), []).append(article)
    return sorted(groups.values(), key=len, reverse=True)


def _title_group_features(article: dict[str, Any]) -> dict[str, Any]:
    title = normalize_korean_spacing(article.get("title") or "")
    companies = set(_company_list(article)) | set(_matched_companies(article))
    tokens = _title_event_tokens(title, companies=companies)
    return {
        "tokens": tokens,
        "companies": companies,
        "event_type": _rule_based_event_type([title]),
    }


def _same_title_event(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["event_type"] != right["event_type"]:
        return False
    left_companies = left["companies"]
    right_companies = right["companies"]
    if left_companies and right_companies and not left_companies & right_companies:
        return False
    left_tokens = left["tokens"]
    right_tokens = right["tokens"]
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    shared = left_tokens & right_tokens
    if len(shared) < 2:
        return False
    coverage = len(shared) / min(len(left_tokens), len(right_tokens))
    jaccard = len(shared) / len(left_tokens | right_tokens)
    return coverage >= 0.45 or (len(shared) >= 3 and jaccard >= 0.22)


def _title_event_tokens(title: str, *, companies: set[str]) -> set[str]:
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9]+", str(title or "").lower())
    company_tokens = _company_alias_title_tokens(companies)
    return {
        token
        for token in (_normalize_title_event_token(token) for token in raw_tokens)
        if _useful_title_event_token(token) and token not in company_tokens
    }


def _normalize_title_event_token(token: str) -> str:
    value = re.sub(r"[^가-힣a-z0-9]", "", str(token or "").lower())
    if re.search(r"[가-힣]", value):
        value = re.sub(r"(으로|로|과|와|은|는|이|가|을|를|에|의)$", "", value)
    return value


def _company_alias_title_tokens(companies: set[str]) -> set[str]:
    tokens: set[str] = set()
    for company_id in companies:
        for alias in _PEER_ALIASES.get(company_id, [company_id]):
            tokens.update(
                _normalize_title_event_token(token)
                for token in re.findall(r"[가-힣A-Za-z0-9]+", str(alias or "").lower())
            )
    return {token for token in tokens if token}


def _useful_title_event_token(token: str) -> bool:
    if len(token) < 2:
        return False
    if token in {
        "및",
        "로",
        "으로",
        "에서",
        "기반",
        "사업",
        "기업",
        "그룹",
        "전사",
        "확대",
        "가속",
        "추진",
    }:
        return False
    return True


def _diverse_articles_from_same_event_group(
    *,
    articles: list[dict[str, Any]],
    representative_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        articles,
        key=lambda article: _analysis_article_score(article, representative_id=representative_id),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_titles: list[str] = []
    for article in ranked:
        title = str(article.get("title") or "")
        is_representative = _article_numeric_id(article) == representative_id or bool(
            article.get("is_representative")
        )
        if (
            not is_representative
            and title
            and any(_text_similarity(title, existing) >= 0.82 for existing in selected_titles)
        ):
            continue
        selected.append(article)
        if title:
            selected_titles.append(title)
        if len(selected) >= limit:
            break
    if len(selected) < min(limit, len(ranked)):
        selected_ids = set(_article_ids(selected))
        for article in ranked:
            if _article_numeric_id(article) in selected_ids:
                continue
            selected.append(article)
            if len(selected) >= limit:
                break
    selected_ids_order = set(_article_ids(selected))
    return [article for article in articles if _article_numeric_id(article) in selected_ids_order]


def _analysis_article_score(article: dict[str, Any], *, representative_id: int) -> float:
    title = str(article.get("title") or "")
    score = _article_evidence_score(article, representative_id=representative_id)
    score += min(2.0, 0.5 * len(_number_tokens(title)))
    score += min(1.0, 0.5 * len(_date_tokens(title)))
    score += min(1.5, 0.5 * len(_rule_based_entities([title])))
    if _rule_based_event_type([title]) != "general_update":
        score += 1.0
    return score


def _join_warnings(*values: str) -> str:
    return "; ".join(value for value in values if value)


def _build_fetch_ids(
    representative_id: int,
    cluster_article_ids: list[int] | None,
    max_cluster_articles: int | None,
) -> list[int]:
    if not cluster_article_ids:
        return [representative_id]

    others = [article_id for article_id in cluster_article_ids if article_id != representative_id]
    ids = _dedupe_ints([representative_id, *others])
    if max_cluster_articles and max_cluster_articles > 0:
        return ids[:max_cluster_articles]
    return ids


def _format_articles(
    articles: list[dict[str, Any]],
    target_companies: list[str],
    representative_id: int,
    compact: bool = False,
) -> str:
    del representative_id, compact
    seen_snippets: list[str] = []
    lines = [
        f"cluster_target_peer_companies: {json.dumps(target_companies, ensure_ascii=False)}",
        "cluster_target_peer_aliases: "
        f"{json.dumps(_target_company_aliases(target_companies), ensure_ascii=False)}",
        "content_policy: 원문 전체 content는 LLM에 넣지 않습니다. "
        "각 기사에서 rule-based로 추출한 evidence_snippets만 사용하고, "
        "중복 문장은 LLM 호출 전에 제거합니다.",
    ]

    for index, article in enumerate(articles, start=1):
        article_id = _article_numeric_id(article)
        metadata = _metadata(article)
        snippets = _article_prompt_snippets(
            article=article,
            target_companies=target_companies,
            seen_snippets=seen_snippets,
        )
        seen_snippets.extend(snippets)
        lines.append(
            "\n".join(
                [
                    f"[{index}] article_id: {article_id}",
                    "article_role: evidence_snippets",
                    f"title: {article.get('title') or ''}",
                    f"source_name: {article.get('source_name') or ''}",
                    f"publisher: {article.get('publisher') or ''}",
                    f"published_at: {article.get('published_at') or ''}",
                    f"company: {json.dumps(_company_list(article), ensure_ascii=False)}",
                    "matched_companies: "
                    f"{json.dumps(_matched_companies(article), ensure_ascii=False)}",
                    f"metadata: {json.dumps(_summary_metadata(metadata), ensure_ascii=False)}",
                    f"evidence_snippets: {json.dumps(snippets, ensure_ascii=False)}",
                ]
            )
        )

    return "\n\n".join(lines)


def _article_prompt_snippets(
    *,
    article: dict[str, Any],
    target_companies: list[str],
    seen_snippets: list[str],
) -> list[str]:
    title = normalize_korean_spacing(article.get("title") or "")
    sentences = _dedupe_keep_order(
        [
            title,
            *_split_evidence_sentences(
                article.get("content") or "",
                limit=_SNIPPET_CANDIDATE_SENTENCES,
            ),
        ]
    )
    sentences = [sentence for sentence in sentences if not _is_article_ui_boilerplate(sentence)]
    scored = sorted(
        (
            (_snippet_score(sentence, article=article, target_companies=target_companies), sentence)
            for sentence in sentences
            if sentence
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    selected: list[str] = []
    local_seen: list[str] = []
    for score, sentence in scored:
        if score <= 0 and selected:
            continue
        if _is_near_duplicate_snippet(sentence, [*seen_snippets, *local_seen]):
            continue
        selected.append(sentence)
        local_seen.append(sentence)
        if len(selected) >= _SNIPPETS_PER_ARTICLE:
            break
    if not selected and title and not _is_near_duplicate_snippet(title, seen_snippets):
        selected.append(title)
    return selected


def _snippet_score(
    sentence: str,
    *,
    article: dict[str, Any],
    target_companies: list[str],
) -> float:
    text = str(sentence or "")
    compact_text = _compact(text)
    score = 0.0
    title = normalize_korean_spacing(article.get("title") or "")
    if text == title:
        score += 3.0
    aliases = _target_company_aliases(target_companies)
    if any(_compact(alias) in compact_text for alias in aliases):
        score += 3.0
    if any(_compact(company) in compact_text for company in _matched_companies(article)):
        score += 1.0
    event_type = _rule_based_event_type([text])
    if event_type != "general_update":
        score += 2.0
    score += min(2.0, 0.5 * len(_number_tokens(text)))
    score += min(1.0, 0.5 * len(_date_tokens(text)))
    if _rule_based_entities([text]):
        score += 1.0
    if _has_detail_preservation_terms(text):
        score += 1.5
    if _has_business_scope_terms(text):
        score += 1.0
    return score


def _is_near_duplicate_snippet(text: str, selected_texts: list[str]) -> bool:
    if not text or not selected_texts:
        return False
    return any(
        _text_similarity(text, selected) >= _SNIPPET_DEDUP_SIMILARITY for selected in selected_texts
    )


def _full_text_article_ids(
    *,
    articles: list[dict[str, Any]],
    representative_id: int,
) -> set[int]:
    """Select a small evidence set for expensive content analysis.

    News clusters can contain many long articles. The summarizer should still
    know the whole cluster membership, but only a few high-signal articles
    should contribute full body text to the LLM prompt.
    """
    if _FULL_TEXT_ARTICLE_LIMIT <= 0:
        return set()
    ranked: list[tuple[float, int, int, dict[str, Any]]] = []
    for index, article in enumerate(articles):
        article_id = _article_numeric_id(article)
        if article_id <= 0:
            continue
        score = _article_evidence_score(article, representative_id=representative_id)
        ranked.append((score, -index, article_id, article))
    ranked.sort(reverse=True)
    selected: list[int] = []
    selected_texts: list[str] = []
    for _, _, article_id, article in ranked:
        dedupe_text = _article_dedupe_text(article)
        is_representative = article_id == representative_id or bool(
            article.get("is_representative")
        )
        if (
            not is_representative
            and dedupe_text
            and _is_near_duplicate_article(dedupe_text, selected_texts)
        ):
            continue
        selected.append(article_id)
        if dedupe_text:
            selected_texts.append(dedupe_text)
        if len(selected) >= _FULL_TEXT_ARTICLE_LIMIT:
            break
    return set(selected[:_FULL_TEXT_ARTICLE_LIMIT])


def _article_evidence_score(article: dict[str, Any], *, representative_id: int) -> float:
    article_id = _article_numeric_id(article)
    score = 0.0
    if article_id == representative_id:
        score += 10.0
    if article.get("is_representative"):
        score += 8.0
    score += _clamp_float(article.get("importance_score"), default=0.0) * 3.0
    score += _clamp_float(article.get("relevance_score"), default=0.0) * 2.0
    if _matched_companies(article):
        score += 1.0
    if article.get("content"):
        score += 0.5
    return score


def _article_dedupe_text(article: dict[str, Any]) -> str:
    text = " ".join(
        part
        for part in (
            str(article.get("title") or ""),
            _normalize_content(article.get("content") or "")[:1600],
        )
        if part
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_near_duplicate_article(text: str, selected_texts: list[str]) -> bool:
    if not text or not selected_texts:
        return False
    return any(
        SequenceMatcher(None, text, selected).ratio() >= _NEAR_DUPLICATE_SIMILARITY
        for selected in selected_texts
    )


def _extract_article_fact_notes_batch(
    *,
    cluster_id: int,
    articles: list[dict[str, Any]],
    target_companies: list[str],
    representative_id: int,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    notes: list[dict[str, Any]] = []
    warnings: list[str] = []
    extraction_failed = False
    for batch in _chunked(articles, _FACT_EXTRACTION_BATCH_SIZE):
        batch_article_ids = [_article_numeric_id(article) for article in batch]
        articles_text = _format_articles(
            articles=batch,
            target_companies=target_companies,
            representative_id=representative_id,
        )
        compact_articles_text = _format_articles(
            articles=batch,
            target_companies=target_companies,
            representative_id=representative_id,
            compact=True,
        )
        parsed, statuses = _extract_article_fact_notes(
            articles_text,
            compact_articles_text=compact_articles_text,
            cluster_id=cluster_id,
            article_ids=batch_article_ids,
        )
        warnings.extend(statuses)
        if "fact_extraction_rule_based_fallback" in statuses:
            extraction_failed = True
        values = parsed.get("article_facts", []) if isinstance(parsed, dict) else []
        if not values:
            values = _rule_based_article_fact_notes(batch, reason="empty_fact_extraction_result")
            warnings.append("fact_extraction_rule_based_candidates_created")
            extraction_failed = True
        for index, item in enumerate(values):
            if isinstance(item, dict):
                note = _normalize_article_fact_note(item)
                if note["article_id"] <= 0 and index < len(batch_article_ids):
                    note["article_id"] = batch_article_ids[index]
                notes.append(note)
    return notes, _dedupe_keep_order(warnings), extraction_failed


def _extract_article_fact_notes(
    articles_text: str,
    *,
    compact_articles_text: str,
    cluster_id: int,
    article_ids: list[int],
) -> tuple[dict[str, Any], list[str]]:
    from src.observability import tracing_config

    def build_prompt(text: str, *, compact_retry: bool = False) -> str:
        prompt_text = _render_prompt(_ARTICLE_FACT_EXTRACTION_PROMPT).replace(
            "{articles_text}",
            text,
        )
        if compact_retry:
            prompt_text += (
                "\n\n추가 지시: 이전 응답이 length limit에 걸렸습니다. "
                "각 article_id마다 core_facts는 최대 3개로 제한하고, evidence_text는 한 문장으로 짧게 유지하세요. "
                "그래도 제품/서비스/플랫폼 공개 fact, 정의/기능 fact, 시연/적용/수치 fact는 우선 보존하세요."
            )
        return prompt_text

    try:
        response = _invoke_fact_extraction_llm(
            build_prompt(articles_text),
            config=tracing_config(
                agent="SourceSummarizer",
                phase="extract_facts",
                prompt_version=_PROMPT_VERSION,
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        if _response_hit_length_limit(response):
            usage = _response_token_usage(response)
            log.warning(
                "기사별 팩트 추출 응답 length finish_reason 감지 | cluster_id=%s article_ids=%s prompt_tokens=%s completion_tokens=%s max_completion_tokens=%s finish_reason=%s retry=%s",
                cluster_id,
                article_ids,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                _FACT_EXTRACTION_MAX_TOKENS,
                _response_finish_reason(response),
                False,
            )
            raise _LengthLimitError("fact extraction response reached length limit")
        parsed, status = _safe_json_loads(content)
        return parsed, [] if status == "parsed" else ["fact_extraction_json_repaired"]
    except Exception as exc:
        if _is_length_limit_error(exc):
            log.warning(
                "기사별 팩트 추출 length limit, compact retry 수행 | cluster_id=%s article_ids=%s max_completion_tokens=%s error=%s",
                cluster_id,
                article_ids,
                _FACT_EXTRACTION_MAX_TOKENS,
                exc,
            )
            try:
                response = _invoke_fact_extraction_llm(
                    build_prompt(compact_articles_text, compact_retry=True),
                    config=tracing_config(
                        agent="SourceSummarizer",
                        phase="extract_facts_compact_retry",
                        prompt_version=_PROMPT_VERSION,
                    ),
                )
                content = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )
                if _response_hit_length_limit(response):
                    usage = _response_token_usage(response)
                    log.warning(
                        "기사별 팩트 추출 compact retry length finish_reason 감지 | cluster_id=%s article_ids=%s prompt_tokens=%s completion_tokens=%s max_completion_tokens=%s finish_reason=%s retry=%s",
                        cluster_id,
                        article_ids,
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        _FACT_EXTRACTION_MAX_TOKENS,
                        _response_finish_reason(response),
                        True,
                    )
                    raise _LengthLimitError("compact retry response reached length limit")
                parsed, status = _safe_json_loads(content)
                statuses = ["fact_extraction_length_limit", "fact_extraction_compact_retry"]
                if status != "parsed":
                    statuses.append("fact_extraction_json_repaired")
                return parsed, statuses
            except Exception as retry_exc:
                log.warning(
                    "기사별 팩트 추출 compact retry 실패 | cluster_id=%s article_ids=%s max_completion_tokens=%s error=%s",
                    cluster_id,
                    article_ids,
                    _FACT_EXTRACTION_MAX_TOKENS,
                    retry_exc,
                )
                return {}, [
                    "fact_extraction_length_limit",
                    "fact_extraction_compact_retry_failed",
                    "fact_extraction_rule_based_fallback",
                ]
        log.warning("기사별 팩트 추출 실패 | error=%s", exc)
        return {}, ["fact_extraction_rule_based_fallback"]


def _invoke_fact_extraction_llm(prompt: str, *, config: RunnableConfig | None) -> Any:
    return (
        _get_llm()
        .bind(
            response_format={"type": "json_object"},
            max_completion_tokens=_FACT_EXTRACTION_MAX_TOKENS,
        )
        .invoke(prompt, config=config)
    )


class _LengthLimitError(RuntimeError):
    pass


def _is_length_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "length limit" in text or "finish_reason" in text and "length" in text


def _response_hit_length_limit(response: Any) -> bool:
    return _response_finish_reason(response) == "length"


def _response_finish_reason(response: Any) -> str:
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        reason = metadata.get("finish_reason")
        if reason:
            return str(reason)
        generations = metadata.get("generations")
        if isinstance(generations, list) and generations:
            first = generations[0]
            if isinstance(first, dict) and first.get("finish_reason"):
                return str(first["finish_reason"])
    return ""


def _response_token_usage(response: Any) -> dict[str, Any]:
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        usage = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(usage, dict):
            return usage
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    return {}


def _normalize_article_fact_note(item: dict[str, Any]) -> dict[str, Any]:
    article_id = _safe_int(item.get("article_id"))
    return {
        "article_id": article_id,
        "core_facts": [
            _normalize_fact_object(fact, default_type="general_update")
            for fact in _as_list(item.get("core_facts"))
        ],
        "unique_facts": [
            _normalize_unique_fact(fact) for fact in _as_list(item.get("unique_facts"))
        ],
        "uncertain_facts": [
            _normalize_uncertain_fact(fact) for fact in _as_list(item.get("uncertain_facts"))
        ],
    }


def _normalize_fact_object(value: Any, *, default_type: str) -> dict[str, Any]:
    if isinstance(value, dict):
        fact = str(value.get("fact") or "").strip()
        activity_type = _normalize_event_type(value.get("activity_type") or default_type)
        fact_type = _normalize_fact_type_value(value.get("fact_type"))
        return {
            "fact": fact,
            "evidence_text": str(value.get("evidence_text") or fact).strip(),
            "activity_type": activity_type,
            "fact_type": fact_type,
            "summary_role": _normalize_summary_role(value.get("summary_role"), fact_type=fact_type),
            "numbers_and_dates": _normalize_string_list(value.get("numbers_and_dates")),
            "customers_or_industries": _normalize_string_list(value.get("customers_or_industries")),
            "products_or_services": _normalize_string_list(value.get("products_or_services")),
        }
    fact = str(value or "").strip()
    return {
        "fact": fact,
        "evidence_text": fact,
        "activity_type": default_type,
        "fact_type": "general_fact",
        "summary_role": "main_event",
        "numbers_and_dates": [],
        "customers_or_industries": [],
        "products_or_services": [],
    }


def _normalize_unique_fact(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        fact = str(value.get("fact") or "").strip()
        fact_type = _normalize_fact_type_value(value.get("fact_type"))
        return {
            "fact": fact,
            "evidence_text": str(value.get("evidence_text") or fact).strip(),
            "fact_type": fact_type,
            "summary_role": _normalize_summary_role(value.get("summary_role"), fact_type=fact_type),
            "importance_reason": str(value.get("importance_reason") or "").strip(),
        }
    fact = str(value or "").strip()
    return {
        "fact": fact,
        "evidence_text": fact,
        "fact_type": "general_fact",
        "summary_role": "main_event",
        "importance_reason": "",
    }


def _normalize_uncertain_fact(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        fact = str(value.get("fact") or "").strip()
        return {
            "fact": fact,
            "evidence_text": str(value.get("evidence_text") or fact).strip(),
            "fact_type": "uncertain_fact",
            "summary_role": "uncertainty_detail",
            "caution": str(value.get("caution") or "전망/예정/가능성 표현").strip(),
        }
    fact = str(value or "").strip()
    return {
        "fact": fact,
        "evidence_text": fact,
        "fact_type": "uncertain_fact",
        "summary_role": "uncertainty_detail",
        "caution": "전망/예정/가능성 표현",
    }


def _rule_based_article_fact_notes(
    articles: list[dict[str, Any]],
    *,
    reason: str,
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for article in articles:
        article_id = _article_numeric_id(article)
        title = normalize_korean_spacing(article.get("title") or "")
        sentences = [
            sentence
            for sentence in _dedupe_keep_order(
                [
                    title,
                    *_split_evidence_sentences(article.get("content") or "", limit=8),
                ]
            )
            if sentence and not _is_article_ui_boilerplate(sentence)
        ]
        if not article_id or not sentences:
            continue

        event_type = _rule_based_event_type(sentences)
        entities = _rule_based_entities(sentences)
        selected = _select_rule_based_sentences(sentences)
        core_facts: list[dict[str, Any]] = []
        for index, sentence in enumerate(selected, start=1):
            fact_type, summary_role = _rule_based_fact_type_and_role(index, sentence, event_type)
            core_facts.append(
                {
                    "fact": sentence,
                    "evidence_text": sentence,
                    "activity_type": event_type,
                    "fact_type": fact_type,
                    "summary_role": summary_role,
                    "numbers_and_dates": _dedupe_keep_order(
                        [*_number_tokens(sentence), *_date_tokens(sentence)]
                    ),
                    "customers_or_industries": [],
                    "products_or_services": [
                        entity for entity in entities if _compact(entity) in _compact(sentence)
                    ][:5],
                }
            )
        notes.append(
            {
                "article_id": article_id,
                "core_facts": core_facts,
                "unique_facts": [],
                "uncertain_facts": [],
                "extraction_warning": reason,
            }
        )
    return notes


def _rule_based_event_type(sentences: list[str]) -> str:
    text = " ".join(sentences).lower()
    event_markers = (
        ("contract", ("수주", "계약", "사업자 선정", "우선협상")),
        ("partnership", ("mou", "협약", "제휴", "협력")),
        ("launch", ("출시", "공개", "선보", "론칭")),
        ("earnings", ("매출", "영업이익", "실적", "순이익")),
        ("stock_market", ("주가", "거래량", "시가총액", "목표주가")),
        ("analyst_report", ("증권사", "리포트", "투자의견", "전망")),
        ("risk", ("장애", "소송", "침해", "해킹", "리스크")),
        ("technology_update", ("기술", "플랫폼", "서비스", "솔루션", "시스템")),
    )
    for event_type, markers in event_markers:
        if any(marker in text for marker in markers):
            return event_type
    return "general_update"


def _rule_based_entities(sentences: list[str]) -> list[str]:
    text = " ".join(sentences)
    quoted = re.findall(r"['\"‘’“”]([^'\"‘’“”]{2,40})['\"‘’“”]", text)
    acronym_like = re.findall(r"\b[A-Z][A-Za-z0-9+\-/]{1,20}\b", text)
    return _dedupe_keep_order(
        [_clean_domain_term(term) for term in [*quoted, *acronym_like] if _clean_domain_term(term)]
    )[:12]


def _select_rule_based_sentences(sentences: list[str]) -> list[str]:
    selected: list[str] = []
    for sentence in sentences:
        if sentence and sentence not in selected:
            selected.append(sentence)
        if len(selected) >= 3:
            break
    while selected and len(selected) < 3:
        selected.append(selected[-1])
    return selected[:3]


def _rule_based_fact_type_and_role(
    index: int,
    sentence: str,
    event_type: str,
) -> tuple[str, str]:
    if index == 1:
        return ("launch_fact" if event_type == "launch" else "general_fact", "main_event")
    if index == 2:
        if event_type == "launch":
            return "platform_definition_fact", "product_definition"
        return "general_fact", "service_function"
    if _number_tokens(sentence):
        return "numeric_fact", "numeric_effect"
    return "application_fact", "application_case"


def _merge_article_facts(article_fact_notes: list[dict[str, Any]]) -> dict[str, Any]:
    fact_map: dict[str, dict[str, Any]] = {}
    unique_facts: list[dict[str, Any]] = []
    uncertain_facts: list[dict[str, Any]] = []
    for note in article_fact_notes:
        article_id = _safe_int(note.get("article_id"))
        for fact in note.get("core_facts", []):
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_text:
                continue
            key = _fact_key(fact_text)
            entry = fact_map.setdefault(
                key,
                {
                    "fact": fact_text,
                    "source_article_ids": [],
                    "evidence_texts": [],
                    "activity_types": [],
                    "numbers_and_dates": [],
                    "customers_or_industries": [],
                    "products_or_services": [],
                },
            )
            if article_id and article_id not in entry["source_article_ids"]:
                entry["source_article_ids"].append(article_id)
            if fact.get("evidence_text"):
                entry["evidence_texts"].append(str(fact.get("evidence_text")))
            if fact.get("activity_type"):
                entry["activity_types"].append(str(fact.get("activity_type")))
            if fact.get("fact_type"):
                entry.setdefault("fact_types", []).append(str(fact.get("fact_type")))
            if fact.get("summary_role"):
                entry.setdefault("summary_roles", []).append(str(fact.get("summary_role")))
            entry["numbers_and_dates"].extend(_normalize_string_list(fact.get("numbers_and_dates")))
            entry["customers_or_industries"].extend(
                _normalize_string_list(fact.get("customers_or_industries"))
            )
            entry["products_or_services"].extend(
                _normalize_string_list(fact.get("products_or_services"))
            )

        for fact in note.get("unique_facts", []):
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_text:
                continue
            evidence_text = str(fact.get("evidence_text") or fact_text)
            unique_facts.append(
                {
                    "fact": fact_text,
                    "source_article_ids": [article_id] if article_id else [],
                    "evidence_count": 1,
                    "evidence_texts": [evidence_text],
                    "fact_type": fact.get("fact_type") or "general_fact",
                    "summary_role": fact.get("summary_role") or "main_event",
                    "numbers_and_dates": _dedupe_keep_order(
                        [
                            *_normalize_string_list(fact.get("numbers_and_dates")),
                            *_number_tokens(evidence_text),
                            *_date_tokens(evidence_text),
                        ]
                    ),
                    "customers_or_industries": _normalize_string_list(
                        fact.get("customers_or_industries")
                    ),
                    "products_or_services": _normalize_string_list(
                        fact.get("products_or_services")
                    ),
                    "importance_reason": str(fact.get("importance_reason") or ""),
                }
            )

        for fact in note.get("uncertain_facts", []):
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_text:
                continue
            uncertain_facts.append(
                {
                    "fact": fact_text,
                    "source_article_ids": [article_id] if article_id else [],
                    "evidence_count": 1,
                    "evidence_texts": [str(fact.get("evidence_text") or fact_text)],
                    "fact_type": "uncertain_fact",
                    "summary_role": "uncertainty_detail",
                    "caution": str(fact.get("caution") or "확정 사실로 단정하지 않음"),
                }
            )

    common_facts: list[dict[str, Any]] = []
    single_core_facts: list[dict[str, Any]] = []
    for entry in fact_map.values():
        entry["source_article_ids"] = _dedupe_ints(entry["source_article_ids"])
        entry["evidence_texts"] = _dedupe_keep_order(
            [text for text in entry["evidence_texts"] if text]
        )[:5]
        entry["activity_types"] = _dedupe_keep_order(
            [item for item in entry["activity_types"] if item]
        )
        entry["fact_types"] = _dedupe_keep_order(
            [item for item in entry.get("fact_types", []) if item]
        )
        entry["summary_roles"] = _dedupe_keep_order(
            [item for item in entry.get("summary_roles", []) if item]
        )
        entry["numbers_and_dates"] = _dedupe_keep_order(entry["numbers_and_dates"])
        entry["customers_or_industries"] = _dedupe_keep_order(entry["customers_or_industries"])
        entry["products_or_services"] = _dedupe_keep_order(entry["products_or_services"])
        entry["evidence_count"] = len(entry["source_article_ids"])
        if entry["evidence_count"] >= 2:
            common_facts.append(entry)
        else:
            single_core_facts.append(entry)

    unique_facts.extend(_important_single_core_facts(single_core_facts))
    return {
        "common_facts": common_facts,
        "unique_facts": unique_facts,
        "uncertain_facts": uncertain_facts,
        "conflict_notes": _detect_conflict_notes([*common_facts, *unique_facts, *uncertain_facts]),
    }


def _important_single_core_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    important: list[dict[str, Any]] = []
    for fact in facts:
        text = " ".join(
            [
                str(fact.get("fact") or ""),
                " ".join(fact.get("numbers_and_dates", [])),
                " ".join(fact.get("customers_or_industries", [])),
                " ".join(fact.get("products_or_services", [])),
            ]
        )
        if not _has_unique_fact_importance(text):
            continue
        copied = dict(fact)
        copied["importance_reason"] = (
            "단일 기사에만 있지만 수치/일정/고객/서비스/후속 단계 정보가 포함됨"
        )
        important.append(copied)
    return important


def _build_cluster_fact_intelligence(merged_facts: dict[str, Any]) -> dict[str, Any]:
    all_facts = [
        *merged_facts.get("common_facts", []),
        *merged_facts.get("unique_facts", []),
        *merged_facts.get("uncertain_facts", []),
    ]
    return {
        "common_facts": merged_facts.get("common_facts", []),
        "unique_facts": merged_facts.get("unique_facts", []),
        "uncertain_facts": merged_facts.get("uncertain_facts", []),
        "conflict_notes": merged_facts.get("conflict_notes", []),
        "numbers_and_dates": _dedupe_keep_order(
            [
                value
                for fact in all_facts
                for value in _normalize_string_list(fact.get("numbers_and_dates"))
            ]
        ),
        "customers_or_industries": _dedupe_keep_order(
            [
                value
                for fact in all_facts
                for value in _normalize_string_list(fact.get("customers_or_industries"))
            ]
        ),
        "products_or_services": _dedupe_keep_order(
            [
                value
                for fact in all_facts
                for value in _normalize_string_list(fact.get("products_or_services"))
            ]
        ),
        "activity_types": _dedupe_keep_order(
            [
                value
                for fact in all_facts
                for value in _normalize_string_list(fact.get("activity_types"))
            ]
        ),
    }


def _classify_cluster_event_type(
    cluster_fact_intelligence: dict[str, Any],
    articles: list[dict[str, Any]],
) -> str:
    activity_types = _normalize_string_list(cluster_fact_intelligence.get("activity_types"))
    for event_type in _EVENT_TYPES:
        if event_type in activity_types and event_type != "unknown":
            return event_type
    text = " ".join(
        [
            json.dumps(cluster_fact_intelligence, ensure_ascii=False),
            *[
                f"{article.get('title') or ''} {article.get('content') or ''}"
                for article in articles
            ],
        ]
    )
    return _classify_event_type_from_text(text)


def _classify_event_type_from_text(text: str) -> str:
    return "general_update" if text.strip() else "unknown"


def _build_extracted_facts(
    *,
    cluster_id: int,
    article_fact_notes: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    cluster_event_type: str,
) -> list[dict[str, Any]]:
    """기사 fact note에 안정적인 fact_id를 붙여 요약 가능한 fact 목록으로 변환한다."""
    facts: list[dict[str, Any]] = []
    counters: dict[int, int] = {}
    article_by_id = {
        _article_numeric_id(article): article
        for article in articles
        if _article_numeric_id(article) > 0
    }

    def add_fact(
        *,
        article_id: int,
        raw_fact: str,
        evidence_text: str,
        fact_type: str,
        summary_role: str | None = None,
        numbers: list[str] | None = None,
        entities: list[str] | None = None,
        activity_type: str | None = None,
        confidence: str = "medium",
    ) -> None:
        text = normalize_korean_spacing(raw_fact)
        evidence = normalize_korean_spacing(evidence_text or raw_fact)
        if not text or not evidence:
            return
        if _fact_is_off_topic_for_article(
            f"{text} {evidence}",
            article=article_by_id.get(article_id) or {},
        ):
            return
        activity = activity_type or cluster_event_type
        if _is_market_data_fact(f"{text} {evidence}") and _normalize_event_type(activity) not in {
            "stock_market",
            "analyst_report",
        }:
            return
        if _is_duplicate_extracted_fact(facts, article_id, text, evidence):
            return
        counters[article_id] = counters.get(article_id, 0) + 1
        inferred_type = _normalize_fact_type(
            fact_type=fact_type,
            text=f"{text} {evidence}",
            activity_type=activity,
        )
        normalized_role = _normalize_summary_role(summary_role, fact_type=inferred_type)
        normalized_role = _coerce_summary_role(
            role=normalized_role,
            fact_type=inferred_type,
            text=f"{text} {evidence}",
            activity_type=activity,
        )
        facts.append(
            {
                "fact_id": f"c{cluster_id}_a{article_id}_f{counters[article_id]}",
                "article_id": article_id,
                "fact_type": inferred_type,
                "summary_role": normalized_role,
                "role_priority": _summary_role_priority(normalized_role),
                "evidence_text": evidence,
                "normalized_fact": text,
                "entities": _dedupe_keep_order(_normalize_string_list(entities)),
                "numbers": _dedupe_keep_order(
                    [*_normalize_string_list(numbers), *_number_tokens(evidence)]
                ),
                "dates": _date_tokens(evidence),
                "event_verbs": _event_verbs_in_text(f"{text} {evidence}"),
                "confidence": confidence if confidence in {"high", "medium", "low"} else "medium",
            }
        )

    for note in article_fact_notes:
        article_id = _safe_int(note.get("article_id"))
        if article_id <= 0:
            continue
        for fact in note.get("core_facts", []):
            text = str(fact.get("fact") or "").strip()
            if not text:
                continue
            entities = [
                *_normalize_string_list(fact.get("products_or_services")),
                *_normalize_string_list(fact.get("customers_or_industries")),
            ]
            add_fact(
                article_id=article_id,
                raw_fact=text,
                evidence_text=str(fact.get("evidence_text") or text),
                fact_type="general_fact",
                summary_role=str(fact.get("summary_role") or ""),
                numbers=_normalize_string_list(fact.get("numbers_and_dates")),
                entities=entities,
                activity_type=str(fact.get("activity_type") or ""),
                confidence="high",
            )
        for fact in note.get("unique_facts", []):
            text = str(fact.get("fact") or "").strip()
            if not text:
                continue
            add_fact(
                article_id=article_id,
                raw_fact=text,
                evidence_text=str(fact.get("evidence_text") or text),
                fact_type="general_fact",
                summary_role=str(fact.get("summary_role") or ""),
                activity_type=cluster_event_type,
                confidence="medium",
            )
        for fact in note.get("uncertain_facts", []):
            text = str(fact.get("fact") or "").strip()
            if not text:
                continue
            add_fact(
                article_id=article_id,
                raw_fact=_soften_uncertain_sentence(text),
                evidence_text=str(fact.get("evidence_text") or text),
                fact_type="uncertain_fact",
                summary_role="uncertainty_detail",
                activity_type=cluster_event_type,
                confidence="medium",
            )

    for article in articles:
        article_id = _safe_int(article.get("id") or article.get("raw_article_id"))
        if article_id <= 0:
            continue
        for fact in _contract_detail_facts_from_article(article):
            add_fact(
                article_id=article_id,
                raw_fact=fact["fact"],
                evidence_text=fact["evidence_text"],
                fact_type=fact["fact_type"],
                summary_role=fact["summary_role"],
                numbers=fact.get("numbers"),
                entities=fact.get("entities"),
                activity_type="contract",
                confidence="high",
            )

    if len(facts) < 3:
        _add_article_fallback_facts(
            facts=facts,
            counters=counters,
            cluster_id=cluster_id,
            articles=articles,
            cluster_event_type=cluster_event_type,
        )

    return facts


def _contract_detail_facts_from_article(article: dict[str, Any]) -> list[dict[str, Any]]:
    """계약/수주 기사에서 사업명, 금액, 기간, 매출 대비 비율을 보강 추출한다.

    LLM 추출이 계약 범위를 "공급" 정도로 약화할 때를 막기 위한 일반 보조 규칙이다.
    특정 회사나 사업명을 박지 않고, 기사 문장에 이미 있는 계약 관련 문장만 사용한다.
    """
    title = str(article.get("title") or "").strip()
    body = " ".join(
        str(article.get(key) or "").strip()
        for key in ("content", "body", "summary", "description")
        if str(article.get(key) or "").strip()
    )
    text = normalize_korean_spacing(f"{title}. {body}")
    if not re.search(r"계약|수주|공급\s*계약|공급계약", text):
        return []

    sentences = _contract_candidate_sentences(text)
    facts: list[dict[str, Any]] = []

    main_sentence = _first_sentence_matching(
        sentences,
        include=(r"계약|수주|공급\s*계약|공급계약", r"억|원|규모|사업|프로젝트|전환|구축|공급"),
    )
    if main_sentence:
        facts.append(
            {
                "fact": _contract_fact_sentence(main_sentence),
                "evidence_text": main_sentence,
                "fact_type": "general_fact",
                "summary_role": "main_event",
                "numbers": _number_tokens(main_sentence),
                "entities": _contract_entities(main_sentence),
            }
        )

    scope_sentence = _first_sentence_matching(
        sentences,
        include=(r"사업|프로젝트|전환|구축|공급|시스템|단말|플랫폼|업무",),
        exclude=[main_sentence] if main_sentence else None,
    )
    if scope_sentence:
        facts.append(
            {
                "fact": _scope_fact_sentence(scope_sentence),
                "evidence_text": scope_sentence,
                "fact_type": "application_fact",
                "summary_role": "service_function",
                "numbers": _number_tokens(scope_sentence),
                "entities": _contract_entities(scope_sentence),
            }
        )

    period_sentence = _first_sentence_matching(
        sentences,
        include=(r"계약\s*기간|기간은|20\d{2}년\s*\d{1,2}월\s*\d{1,2}일",),
    )
    if period_sentence:
        facts.append(
            {
                "fact": _ensure_sentence(period_sentence),
                "evidence_text": period_sentence,
                "fact_type": "numeric_fact",
                "summary_role": "numeric_effect",
                "numbers": [*_number_tokens(period_sentence), *_date_tokens(period_sentence)],
                "entities": _contract_entities(period_sentence),
            }
        )

    ratio_sentence = _first_sentence_matching(
        sentences,
        include=(r"최근\s*매출|매출액\s*대비|매출\s*대비|%",),
    )
    if ratio_sentence:
        facts.append(
            {
                "fact": _ensure_sentence(ratio_sentence),
                "evidence_text": ratio_sentence,
                "fact_type": "numeric_fact",
                "summary_role": "numeric_effect",
                "numbers": _number_tokens(ratio_sentence),
                "entities": _contract_entities(ratio_sentence),
            }
        )

    return _dedupe_contract_facts(facts)


def _contract_candidate_sentences(text: str) -> list[str]:
    cleaned = normalize_korean_spacing(text)
    parts = [
        re.sub(r"\s+", " ", item).strip(" -·")
        for item in re.split(r"(?<=[.!?。！？])\s+|(?<=다)\.\s*|(?<=다)\s+", cleaned)
        if re.sub(r"\s+", " ", item).strip(" -·")
    ]
    return [item if item.endswith((".", "다")) else _ensure_sentence(item) for item in parts]


def _first_sentence_matching(
    sentences: list[str],
    *,
    include: tuple[str, ...],
    exclude: list[str | None] | None = None,
) -> str:
    excluded = {re.sub(r"\s+", "", str(item or "")) for item in (exclude or []) if item}
    for sentence in sentences:
        key = re.sub(r"\s+", "", sentence)
        if key in excluded:
            continue
        if all(re.search(pattern, sentence) for pattern in include):
            return sentence
    return ""


def _contract_fact_sentence(sentence: str) -> str:
    return _ensure_sentence(sentence)


def _scope_fact_sentence(sentence: str) -> str:
    return _ensure_sentence(sentence)


def _contract_entities(sentence: str) -> list[str]:
    entities = re.findall(
        r"[가-힣A-Za-z0-9&·+_-]{2,}(?:\s+[가-힣A-Za-z0-9&·+_-]{2,}){0,5}"
        r"(?:사업|프로젝트|계약|시스템|플랫폼|단말|서비스|솔루션|업무|인프라)",
        sentence,
    )
    return _dedupe_keep_order([re.sub(r"\s+", " ", item).strip() for item in entities])


def _dedupe_contract_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in facts:
        key = re.sub(r"[\s.。!?！？,，]+", "", str(fact.get("fact") or ""))
        if key and key not in seen:
            result.append(fact)
            seen.add(key)
    return result


def _ensure_sentence(text: str) -> str:
    sentence = str(text or "").strip()
    if not sentence:
        return ""
    return sentence if sentence.endswith((".", "。", "!", "?", "！", "？")) else f"{sentence}."


def _is_duplicate_extracted_fact(
    facts: list[dict[str, Any]],
    article_id: int,
    normalized_fact: str,
    evidence_text: str,
) -> bool:
    for fact in facts:
        if _safe_int(fact.get("article_id")) != article_id:
            continue
        if _text_similarity(str(fact.get("normalized_fact") or ""), normalized_fact) >= 0.9:
            return True
        if _text_similarity(str(fact.get("evidence_text") or ""), evidence_text) >= 0.9:
            return True
    return False


def _add_article_fallback_facts(
    *,
    facts: list[dict[str, Any]],
    counters: dict[int, int],
    cluster_id: int,
    articles: list[dict[str, Any]],
    cluster_event_type: str,
) -> None:
    for article in articles:
        article_id = _article_numeric_id(article)
        if article_id <= 0:
            continue
        title = normalize_korean_spacing(article.get("title") or "")
        if title and not _is_duplicate_extracted_fact(facts, article_id, title, title):
            counters[article_id] = counters.get(article_id, 0) + 1
            facts.append(
                {
                    "fact_id": f"c{cluster_id}_a{article_id}_f{counters[article_id]}",
                    "article_id": article_id,
                    "fact_type": _normalize_fact_type(
                        fact_type="general_fact",
                        text=title,
                        activity_type=cluster_event_type,
                    ),
                    "summary_role": "main_event",
                    "role_priority": _summary_role_priority("main_event"),
                    "evidence_text": title,
                    "normalized_fact": _title_to_fact_sentence(title),
                    "entities": [],
                    "numbers": _number_tokens(title),
                    "dates": _date_tokens(title),
                    "event_verbs": _event_verbs_in_text(title),
                    "confidence": "medium",
                }
            )
        for sentence in _split_evidence_sentences(article.get("content") or "", limit=8):
            if len(facts) >= 6:
                return
            if _is_duplicate_extracted_fact(facts, article_id, sentence, sentence):
                continue
            counters[article_id] = counters.get(article_id, 0) + 1
            facts.append(
                {
                    "fact_id": f"c{cluster_id}_a{article_id}_f{counters[article_id]}",
                    "article_id": article_id,
                    "fact_type": _normalize_fact_type(
                        fact_type="general_fact",
                        text=sentence,
                        activity_type=cluster_event_type,
                    ),
                    "summary_role": "main_event",
                    "role_priority": _summary_role_priority("main_event"),
                    "evidence_text": sentence,
                    "normalized_fact": normalize_korean_spacing(sentence),
                    "entities": [],
                    "numbers": _number_tokens(sentence),
                    "dates": _date_tokens(sentence),
                    "event_verbs": _event_verbs_in_text(sentence),
                    "confidence": "low",
                }
            )


def _normalize_fact_type(*, fact_type: str, text: str, activity_type: str) -> str:
    normalized = _normalize_fact_type_value(fact_type)
    if normalized == "numeric_fact" and _has_business_scope_terms(text):
        if _normalize_event_type(activity_type) in {"contract", "partnership"}:
            return "application_fact"
        if _normalize_event_type(activity_type) in {
            "launch",
            "technology_update",
            "general_update",
        }:
            return "application_fact"
    return normalized


def _normalize_fact_type_value(value: Any) -> str:
    fact_type = str(value or "").strip()
    return fact_type if fact_type in _FACT_TYPES else "general_fact"


def _normalize_summary_role(value: Any, *, fact_type: str) -> str:
    role = str(value or "").strip()
    if role in _SUMMARY_ROLES:
        return role
    return _default_summary_role(fact_type)


def _coerce_summary_role(*, role: str, fact_type: str, text: str, activity_type: str) -> str:
    event_type = _normalize_event_type(activity_type)
    if role == "numeric_effect" and _has_business_scope_terms(text):
        if event_type in {"contract", "partnership"} and re.search(
            r"계약|수주|공급\s*계약|공급계약|협약|MOU", text
        ):
            return "main_event"
        if re.search(r"업무|시스템|전환|구축|플랫폼|솔루션|서비스|AI|에이전트", text, re.I):
            return "service_function"
        return "application_case"
    if fact_type == "numeric_fact":
        return role
    if role == "main_event" and re.search(r"기능|역할|지원|자동화|분석|검증|운영|적용|연계", text):
        return "service_function"
    return role


def _has_business_scope_terms(text: str) -> bool:
    return bool(
        re.search(
            r"계약|수주|공급|협약|사업|프로젝트|업무|시스템|전환|구축|"
            r"플랫폼|솔루션|서비스|AI|에이전트|자동화|검증|운영|고객|"
            r"ERP|MES|단말|클라우드|데이터센터|모빌리티|소프트웨어|SW",
            str(text or ""),
            re.I,
        )
    )


def _has_detail_preservation_terms(text: str) -> bool:
    return bool(
        re.search(
            r"기능|모듈|라인업|범위|대상|적용|연계|접속|처리|수행|"
            r"지원|관리|운영|보안|통합|고도화|확장|전환|도입|활용|"
            r"실증|검증|시범|상용|출시|공개|제공|개발|협력|계획|예정|향후",
            str(text or ""),
            re.I,
        )
    )


def _is_financial_only_fact(text: str) -> bool:
    value = str(text or "")
    if _has_business_scope_terms(value):
        return False
    return bool(
        re.search(r"매출|영업이익|순이익|주가|시가총액|증가|감소|흑자|적자|억원|조원|%", value)
    )


def _is_market_data_fact(text: str) -> bool:
    value = str(text or "")
    if not value:
        return False
    strong_terms = (
        "주가",
        "현재가",
        "전일대비",
        "등락률",
        "거래량",
        "시가총액",
        "목표주가",
        "투자의견",
    )
    if any(term in value for term in strong_terms):
        return True
    return bool(
        re.search(
            r"\b(?:KOSPI|KOSDAQ)\b|전\s*거래일|장\s*(초반|마감)|"
            r"(?:상승|하락|급등|급락)\s*(?:마감|출발|전환)",
            value,
            re.I,
        )
    )


def _default_summary_role(fact_type: str) -> str:
    if fact_type == "launch_fact":
        return "main_event"
    if fact_type == "platform_definition_fact":
        return "product_definition"
    if fact_type == "application_fact":
        return "application_case"
    if fact_type == "numeric_fact":
        return "numeric_effect"
    if fact_type == "market_fact":
        return "market_reaction"
    if fact_type == "risk_fact":
        return "risk_detail"
    if fact_type == "uncertain_fact":
        return "uncertainty_detail"
    return "main_event"


def _summary_role_priority(role: str) -> int:
    ordered = [
        "main_event",
        "product_definition",
        "service_function",
        "application_case",
        "numeric_effect",
        "market_reaction",
        "risk_detail",
        "uncertainty_detail",
    ]
    try:
        return ordered.index(role) + 1
    except ValueError:
        return len(ordered)


def _select_fact_ids_for_summary_lines(
    *,
    extracted_facts: list[dict[str, Any]],
    cluster_event_type: str,
) -> dict[str, list[str]]:
    available = [fact for fact in extracted_facts if fact.get("fact_id")]
    used: set[str] = set()
    selected_facts: list[dict[str, Any]] = []

    def choose(index: int, preferred_roles: tuple[str, ...]) -> list[str]:
        candidates = [
            fact
            for fact in available
            if fact.get("fact_id") not in used and fact.get("summary_role") in preferred_roles
        ]
        if not candidates:
            candidates = [fact for fact in available if fact.get("fact_id") not in used]
        if not candidates:
            candidates = available
        if not candidates:
            return []
        selected = sorted(
            candidates,
            key=lambda fact: (
                _fact_selection_score(fact) - _similar_selected_fact_penalty(fact, selected_facts)
            ),
            reverse=True,
        )[0]
        fact_id = str(selected.get("fact_id"))
        used.add(fact_id)
        selected_facts.append(selected)
        return [fact_id]

    preferences = _line_summary_role_preferences(cluster_event_type)
    desired_count = min(_SUMMARY_LINE_MAX, max(_SUMMARY_LINE_MIN, len(available)))
    return {
        str(index): choose(index, preferences[min(index - 1, len(preferences) - 1)])
        for index in range(1, desired_count + 1)
    }


def _line_summary_role_preferences(
    event_type: str,
) -> tuple[tuple[str, ...], ...]:
    event_type = _normalize_event_type(event_type)
    if event_type == "launch":
        return (
            ("main_event",),
            ("product_definition", "service_function"),
            ("application_case", "numeric_effect", "uncertainty_detail"),
            ("application_case", "service_function", "numeric_effect"),
            ("uncertainty_detail", "numeric_effect", "market_reaction"),
        )
    if event_type in {"technology_update", "general_update", "unknown"}:
        return (
            ("main_event",),
            ("service_function", "product_definition", "application_case"),
            ("uncertainty_detail", "application_case", "numeric_effect"),
            ("application_case", "service_function", "numeric_effect"),
            ("uncertainty_detail", "risk_detail", "market_reaction"),
        )
    if event_type == "contract":
        return (
            ("main_event",),
            ("service_function", "product_definition", "application_case"),
            ("application_case", "service_function", "product_definition"),
            ("service_function", "application_case", "numeric_effect"),
            ("service_function", "application_case", "uncertainty_detail", "numeric_effect"),
        )
    if event_type == "earnings":
        return (
            ("main_event", "numeric_effect"),
            ("service_function", "product_definition"),
            ("numeric_effect", "uncertainty_detail"),
            ("market_reaction", "numeric_effect"),
            ("uncertainty_detail", "risk_detail"),
        )
    if event_type == "stock_market":
        return (
            ("market_reaction", "main_event"),
            ("service_function", "product_definition"),
            ("numeric_effect", "market_reaction", "uncertainty_detail"),
            ("market_reaction", "risk_detail"),
            ("uncertainty_detail", "numeric_effect"),
        )
    if event_type == "risk":
        return (
            ("risk_detail", "main_event"),
            ("service_function", "product_definition", "application_case"),
            ("risk_detail", "uncertainty_detail", "numeric_effect"),
            ("application_case", "risk_detail"),
            ("uncertainty_detail", "market_reaction"),
        )
    return (
        ("main_event",),
        ("product_definition", "service_function", "application_case"),
        ("numeric_effect", "application_case", "uncertainty_detail"),
        ("application_case", "service_function", "market_reaction"),
        ("uncertainty_detail", "risk_detail", "numeric_effect"),
    )


def _fact_selection_score(fact: dict[str, Any]) -> int:
    score = 0
    role = str(fact.get("summary_role") or "")
    score += {
        "main_event": 8,
        "product_definition": 7,
        "service_function": 7,
        "application_case": 6,
        "uncertainty_detail": 2,
        "risk_detail": 2,
        "numeric_effect": 1,
        "market_reaction": 0,
    }.get(role, 0)
    score += (
        3 if fact.get("confidence") == "high" else 2 if fact.get("confidence") == "medium" else 1
    )
    if fact.get("entities"):
        score += 2
    if fact.get("numbers") or fact.get("dates"):
        score += 2
    if fact.get("event_verbs"):
        score += 1
    text = f"{fact.get('normalized_fact') or ''} {fact.get('evidence_text') or ''}"
    if re.search(r"업무|시스템|고객|서비스|솔루션|플랫폼|에이전트|코딩|협업|문서", text):
        score += 3
    if re.search(r"외부|확대|고도화|제공|지원|활용|적용|연계", text):
        score += 2
    if re.search(r"외부\s*기업|기업\s*고객|사업\s*영역|사업\s*확장|고객으로|고객에게", text):
        score += 5
    if role == "numeric_effect" and _is_financial_only_fact(text):
        score -= 8
    score += min(len(str(fact.get("normalized_fact") or "")) // 30, 3)
    return score


def _similar_selected_fact_penalty(
    fact: dict[str, Any],
    selected_facts: list[dict[str, Any]],
) -> int:
    text = _fact_similarity_text(fact)
    if not text or not selected_facts:
        return 0
    max_similarity = max(
        _text_similarity(text, _fact_similarity_text(selected)) for selected in selected_facts
    )
    if max_similarity >= 0.82:
        return 8
    if max_similarity >= 0.68:
        return 4
    return 0


def _fact_similarity_text(fact: dict[str, Any]) -> str:
    return str(fact.get("normalized_fact") or fact.get("evidence_text") or "").strip()


def _fact_is_off_topic_for_article(text: str, *, article: dict[str, Any]) -> bool:
    title = str(article.get("title") or "").strip()
    if not title:
        return False
    title_tokens = _article_topic_tokens(title)
    if len(title_tokens) < 2:
        return False
    value = str(text or "")
    fact_tokens = _article_topic_tokens(value)
    if title_tokens & fact_tokens:
        return False
    if _article_company_alias_mentioned(value, article):
        return False
    return True


def _article_topic_tokens(text: str) -> set[str]:
    stopwords = {
        "속보",
        "단독",
        "특징주",
        "정부",
        "사업",
        "참여",
        "선정",
        "체결",
        "규모",
        "지원",
        "구축",
        "확보",
        "운용",
        "관련",
        "오늘",
        "이번",
    }
    return {
        token
        for token in _article_similarity_tokens(text)
        if len(token) >= 2 and token not in stopwords and not token.isdigit()
    }


def _article_similarity_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", str(text or ""))
        if len(token) >= 2
    }


def _article_company_alias_mentioned(text: str, article: dict[str, Any]) -> bool:
    companies = [
        *_normalize_string_list(article.get("company")),
        *_normalize_string_list(article.get("matched_companies")),
        *_normalize_string_list(article.get("matched_company")),
    ]
    value = str(text or "")
    for company_id in companies:
        aliases = _PEER_ALIASES.get(company_id) or COMPANY_ALIASES.get(company_id) or []
        if any(
            alias and re.search(re.escape(str(alias)), value, re.IGNORECASE) for alias in aliases
        ):
            return True
    return False


def _summarize_from_fact_ids(
    *,
    cluster_event_type: str,
    extracted_facts: list[dict[str, Any]],
    selected_fact_ids: dict[str, list[str]],
    target_companies: list[str],
    main_company: str,
) -> dict[str, Any]:
    if not extracted_facts:
        return _fact_id_empty_result(
            main_company=main_company,
            target_companies=target_companies,
            cluster_event_type=cluster_event_type,
            reason="추출된 fact가 없어 요약할 수 없음",
        )
    try:
        from src.observability import tracing_config

        prompt = _render_prompt(_FACT_ID_SUMMARY_PROMPT)
        prompt = (
            prompt.replace("{main_company}", main_company)
            .replace("{target_companies_json}", json.dumps(target_companies, ensure_ascii=False))
            .replace("{cluster_event_type}", cluster_event_type)
            .replace(
                "{selected_facts_json}",
                json.dumps(
                    _selected_facts_by_line(selected_fact_ids, extracted_facts),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            .replace(
                "{all_facts_json}",
                json.dumps(
                    _compact_facts_for_prompt(extracted_facts), ensure_ascii=False, indent=2
                ),
            )
        )
        response = (
            _get_llm()
            .bind(max_completion_tokens=_SUMMARY_MAX_TOKENS)
            .invoke(
                prompt,
                config=tracing_config(
                    agent="SourceSummarizer",
                    phase="fact_id_summary",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        result = _normalize_fact_id_summary_result(
            _parse_json(content),
            target_companies=target_companies,
            fallback_company=main_company,
            cluster_event_type=cluster_event_type,
            extracted_facts=extracted_facts,
        )
    except Exception as exc:
        log.warning("fact_id 기반 요약 LLM 실패, fallback 사용 | error=%s", exc)
        result = _fallback_fact_id_summary(
            main_company=main_company,
            target_companies=target_companies,
            cluster_event_type=cluster_event_type,
            extracted_facts=extracted_facts,
            selected_fact_ids=selected_fact_ids,
            reason=f"fact_id 요약 LLM 실패: {type(exc).__name__}",
        )

    checked = _validate_fact_id_summary(
        result=result,
        extracted_facts=extracted_facts,
        source_article_ids=[],
        main_company=result.get("main_company") or main_company,
    )
    if checked.get("is_valid_summary"):
        return checked

    fallback = _fallback_fact_id_summary(
        main_company=main_company,
        target_companies=target_companies,
        cluster_event_type=cluster_event_type,
        extracted_facts=extracted_facts,
        selected_fact_ids=selected_fact_ids,
        reason=_append_reason(
            checked.get("reason"), "LLM 결과 검증 실패 후 fallback template 사용"
        ),
    )
    fallback["repair_actions"] = _dedupe_keep_order(
        [
            *_normalize_string_list(checked.get("repair_actions")),
            "fallback_template_from_selected_fact_ids",
        ]
    )
    return fallback


def _normalize_fact_id_summary_result(
    data: dict[str, Any],
    *,
    target_companies: list[str],
    fallback_company: str,
    cluster_event_type: str,
    extracted_facts: list[dict[str, Any]],
) -> dict[str, Any]:
    mentioned = [
        company_id
        for company_id in _normalize_string_list(data.get("mentioned_peer_companies"))
        if company_id in target_companies
    ]
    main_company = str(data.get("main_company") or fallback_company).strip()
    if main_company not in target_companies:
        main_company = mentioned[0] if mentioned else fallback_company
    line_items = _normalize_summary_line_items(data.get("summary_lines"))
    if not line_items and data.get("fact_summary"):
        line_items = [
            {"line_index": index, "text": text, "fact_ids": []}
            for index, text in enumerate(
                _normalize_string_list(data.get("fact_summary"))[:_SUMMARY_LINE_MAX], start=1
            )
        ]
    line_items = _ensure_fact_summary_lines(line_items, extracted_facts)
    fact_summary = [str(item.get("text") or "").strip() for item in line_items]
    is_valid_summary = (
        bool(data.get("is_valid_summary", True))
        and _SUMMARY_LINE_MIN <= len(fact_summary) <= _SUMMARY_LINE_MAX
        and all(fact_summary)
    )
    result = {
        "is_valid_summary": is_valid_summary,
        "main_company": main_company,
        "mentioned_peer_companies": mentioned or [main_company],
        "cluster_event_type": _normalize_event_type(
            data.get("cluster_event_type") or cluster_event_type
        ),
        "headline": str(data.get("headline") or fact_summary[0] if fact_summary else "").strip(),
        "one_line_summary": str(
            data.get("one_line_summary") or fact_summary[0] if fact_summary else ""
        ).strip(),
        "fact_summary": fact_summary,
        "summary_lines_with_fact_ids": line_items,
        "main_event": str(
            data.get("main_event") or fact_summary[0] if fact_summary else ""
        ).strip(),
        "confidence": _clamp_float(data.get("confidence"), default=0.0),
        "reason": str(data.get("reason") or "").strip(),
    }
    result["fact_basis"] = _fact_basis_from_summary_line_fact_ids(line_items, extracted_facts)
    return result


def _normalize_summary_line_items(value: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for fallback_index, item in enumerate(_as_list(value), start=1):
        if isinstance(item, dict):
            index = _safe_int(item.get("line_index")) or fallback_index
            text = normalize_korean_spacing(item.get("text") or item.get("summary_line") or "")
            fact_ids = [
                str(fact_id) for fact_id in _normalize_string_list(item.get("fact_ids")) if fact_id
            ]
        else:
            index = fallback_index
            text = normalize_korean_spacing(item)
            fact_ids = []
        if 1 <= index <= _SUMMARY_LINE_MAX and text:
            lines.append({"line_index": index, "text": text, "fact_ids": fact_ids})
    by_index: dict[int, dict[str, Any]] = {}
    for item in lines:
        by_index[_safe_int(item.get("line_index"))] = item
    return [by_index[index] for index in range(1, _SUMMARY_LINE_MAX + 1) if index in by_index]


def _ensure_fact_summary_lines(
    line_items: list[dict[str, Any]],
    extracted_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    line_items = [
        item for item in line_items if 1 <= _safe_int(item.get("line_index")) <= _SUMMARY_LINE_MAX
    ]
    desired_count = len(line_items)
    if desired_count <= 0:
        desired_count = min(_SUMMARY_LINE_MAX, max(_SUMMARY_LINE_MIN, len(extracted_facts)))
    result: dict[int, dict[str, Any]] = {
        _safe_int(item.get("line_index")): item
        for item in line_items
        if 1 <= _safe_int(item.get("line_index")) <= _SUMMARY_LINE_MAX
    }
    unused_facts = [fact for fact in extracted_facts if str(fact.get("fact_id"))]
    for index in range(1, desired_count + 1):
        item = result.get(index)
        if item and item.get("fact_ids"):
            item["fact_ids"] = [fact_id for fact_id in item["fact_ids"] if fact_id in fact_by_id]
        if item and item.get("fact_ids"):
            continue
        fact = unused_facts[min(index - 1, len(unused_facts) - 1)] if unused_facts else None
        if fact is None:
            result[index] = {"line_index": index, "text": "", "fact_ids": []}
            continue
        result[index] = {
            "line_index": index,
            "text": _fact_text_for_summary_line(fact),
            "fact_ids": [str(fact.get("fact_id"))],
        }
    return [result[index] for index in range(1, desired_count + 1)]


def _fact_basis_from_summary_line_fact_ids(
    line_items: list[dict[str, Any]],
    extracted_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    basis: list[dict[str, Any]] = []
    for item in line_items:
        index = _safe_int(item.get("line_index"))
        fact_ids = [
            fact_id
            for fact_id in _normalize_string_list(item.get("fact_ids"))
            if fact_id in fact_by_id
        ]
        if not index or not fact_ids:
            continue
        facts = [fact_by_id[fact_id] for fact_id in fact_ids]
        evidence_texts = _dedupe_similar_texts(
            [str(fact.get("evidence_text") or "") for fact in facts]
        )
        basis.append(
            {
                "summary_sentence_index": index,
                "summary_line_index": index,
                "fact": str(item.get("text") or ""),
                "source_article_ids": _dedupe_ints(
                    [_safe_int(fact.get("article_id")) for fact in facts]
                ),
                "fact_ids": fact_ids,
                "evidence_count": len(fact_ids),
                "evidence_type": _combined_evidence_type(facts),
                "evidence_texts": evidence_texts[:3],
            }
        )
    return basis


def _combined_evidence_type(facts: list[dict[str, Any]]) -> str:
    evidence_types = [
        _evidence_type_from_fact_type(str(fact.get("fact_type") or "")) for fact in facts
    ]
    for preferred in (
        "risk_fact",
        "market_reaction_fact",
        "uncertain_fact",
        "core_fact",
        "unique_fact",
        "numeric_fact",
    ):
        if preferred in evidence_types:
            return preferred
    return "reported_fact"


def _evidence_type_from_fact_type(fact_type: str) -> str:
    if fact_type == "market_fact":
        return "market_reaction_fact"
    if fact_type == "risk_fact":
        return "risk_fact"
    if fact_type == "uncertain_fact":
        return "uncertain_fact"
    if fact_type in {"launch_fact", "platform_definition_fact"}:
        return "core_fact"
    if fact_type == "application_fact":
        return "unique_fact"
    if fact_type == "numeric_fact":
        return "numeric_fact"
    return "reported_fact"


def _fallback_fact_id_summary(
    *,
    main_company: str,
    target_companies: list[str],
    cluster_event_type: str,
    extracted_facts: list[dict[str, Any]],
    selected_fact_ids: dict[str, list[str]],
    reason: str,
) -> dict[str, Any]:
    line_items: list[dict[str, Any]] = []
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    selected_fact_ids = _select_fact_ids_for_summary_lines(
        extracted_facts=extracted_facts,
        cluster_event_type=cluster_event_type,
    )
    desired_count = min(_SUMMARY_LINE_MAX, max(_SUMMARY_LINE_MIN, len(extracted_facts)))
    for index in range(1, desired_count + 1):
        ids = [
            fact_id for fact_id in selected_fact_ids.get(str(index), []) if fact_id in fact_by_id
        ]
        if not ids and extracted_facts:
            fallback_fact = extracted_facts[min(index - 1, len(extracted_facts) - 1)]
            ids = [str(fallback_fact.get("fact_id"))]
        facts = [fact_by_id[fact_id] for fact_id in ids if fact_id in fact_by_id]
        text = _compose_fallback_line(index=index, facts=facts)
        line_items.append({"line_index": index, "text": text, "fact_ids": ids})
    fact_summary = [item["text"] for item in line_items]
    result = {
        "is_valid_summary": bool(extracted_facts),
        "main_company": main_company,
        "mentioned_peer_companies": [main_company]
        if main_company in target_companies
        else target_companies[:1],
        "cluster_event_type": _normalize_event_type(cluster_event_type),
        "headline": fact_summary[0] if fact_summary else "",
        "one_line_summary": fact_summary[0] if fact_summary else "",
        "fact_summary": fact_summary,
        "summary_lines_with_fact_ids": line_items,
        "main_event": fact_summary[0] if fact_summary else "",
        "fact_basis": _fact_basis_from_summary_line_fact_ids(line_items, extracted_facts),
        "confidence": 0.65 if extracted_facts else 0.0,
        "reason": reason,
    }
    return result


def _fact_id_empty_result(
    *,
    main_company: str,
    target_companies: list[str],
    cluster_event_type: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "is_valid_summary": False,
        "main_company": main_company,
        "mentioned_peer_companies": [main_company] if main_company in target_companies else [],
        "cluster_event_type": _normalize_event_type(cluster_event_type),
        "headline": "",
        "one_line_summary": "",
        "fact_summary": [],
        "summary_lines_with_fact_ids": [],
        "main_event": "",
        "fact_basis": [],
        "confidence": 0.0,
        "reason": reason,
    }


def _validate_fact_id_summary(
    *,
    result: dict[str, Any],
    extracted_facts: list[dict[str, Any]],
    source_article_ids: list[int],
    main_company: str,
) -> dict[str, Any]:
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    lines = [
        _clean_summary_line(normalize_korean_spacing(line))
        for line in _normalize_string_list(result.get("fact_summary"))[:_SUMMARY_LINE_MAX]
    ]
    result["fact_summary"] = lines
    line_items = _normalize_summary_line_items(result.get("summary_lines_with_fact_ids"))
    if not line_items:
        line_items = [
            {"line_index": index, "text": line, "fact_ids": []}
            for index, line in enumerate(lines, start=1)
        ]
    line_items = _ensure_fact_summary_lines(line_items, extracted_facts)
    for item in line_items:
        index = _safe_int(item.get("line_index"))
        if 1 <= index <= len(lines):
            item["text"] = lines[index - 1]
    result["summary_lines_with_fact_ids"] = line_items
    result["fact_basis"] = _fact_basis_from_summary_line_fact_ids(line_items, extracted_facts)

    warnings: list[str] = []
    actions: list[str] = []
    if not (_SUMMARY_LINE_MIN <= len(lines) <= _SUMMARY_LINE_MAX) or any(
        not line for line in lines
    ):
        warnings.append("summary_lines가 3~5개 범위를 벗어남")
    basis_indexes = {
        _safe_int(item.get("summary_line_index", item.get("summary_sentence_index")))
        for item in result.get("fact_basis", [])
    }
    expected_indexes = [
        _safe_int(item.get("line_index"))
        for item in line_items
        if _safe_int(item.get("line_index")) > 0
    ]
    missing_indexes = [index for index in expected_indexes if index not in basis_indexes]
    if missing_indexes:
        warnings.append(f"fact_basis 누락 summary_line_index: {missing_indexes}")
    for item in line_items:
        index = _safe_int(item.get("line_index"))
        fact_ids = _normalize_string_list(item.get("fact_ids"))
        if not fact_ids:
            warnings.append(f"{index}번 문장 fact_id 없음")
        invalid_ids = [fact_id for fact_id in fact_ids if fact_id not in fact_by_id]
        if invalid_ids:
            warnings.append(f"{index}번 문장에 존재하지 않는 fact_id: {invalid_ids}")
    for index, line in enumerate(lines, start=1):
        related_facts = _facts_for_line(index, line_items, fact_by_id)
        related_evidence = " ".join(
            " ".join(
                [
                    str(fact.get("fact") or ""),
                    str(fact.get("evidence_text") or ""),
                    " ".join(_normalize_string_list(fact.get("evidence_texts"))),
                    " ".join(_normalize_string_list(fact.get("numbers_and_dates"))),
                    " ".join(_normalize_string_list(fact.get("numbers"))),
                ]
            )
            for fact in related_facts
        )
        missing_numbers = [
            number
            for number in _number_tokens(line)
            if not _number_token_covered(number, _number_tokens(related_evidence))
        ]
        if missing_numbers:
            warnings.append(f"{index}번 문장 수치 근거 부족: {', '.join(missing_numbers)}")
        attribution_warning = _summary_line_company_attribution_warning(
            line=line,
            evidence=related_evidence,
            main_company=main_company,
        )
        if attribution_warning:
            warnings.append(f"{index}번 문장 {attribution_warning}")
    cleaned_lines = [normalize_korean_spacing(line) for line in lines]
    if cleaned_lines != lines:
        result["fact_summary"] = cleaned_lines
        actions.append("normalize_korean_spacing")
        line_items = _sync_summary_line_item_texts(line_items, cleaned_lines)

    reduced_lines, repetition_actions = normalize_subject_predicate_consistency(
        lines=_normalize_string_list(result.get("fact_summary"))[:_SUMMARY_LINE_MAX],
        line_items=line_items,
        fact_by_id=fact_by_id,
        main_company=main_company,
    )
    if repetition_actions:
        result["fact_summary"] = reduced_lines
        line_items = _sync_summary_line_item_texts(line_items, reduced_lines)
        result["summary_lines_with_fact_ids"] = line_items
        result["fact_basis"] = _fact_basis_from_summary_line_fact_ids(line_items, extracted_facts)
        actions.extend(repetition_actions)

    company_start_count = _company_name_start_count(
        _normalize_string_list(result.get("fact_summary"))[:_SUMMARY_LINE_MAX],
        main_company,
    )
    if company_start_count >= 2:
        warnings.append(f"company_name_start_count={company_start_count}")
        actions.append("company_name_repetition_detected")
    if lines and company_start_count == len(lines):
        warnings.append("summary_lines 모든 문장이 company_name으로 시작함")

    role_counts = _summary_line_role_counts(line_items, fact_by_id)
    cluster_event_type = _normalize_event_type(result.get("cluster_event_type"))
    if cluster_event_type not in {"earnings", "stock_market", "analyst_report"}:
        business_role_count = sum(
            role_counts.get(role, 0)
            for role in ("main_event", "product_definition", "service_function", "application_case")
        )
        numeric_role_count = role_counts.get("numeric_effect", 0) + role_counts.get(
            "market_reaction", 0
        )
        if numeric_role_count >= 2 and business_role_count < 2:
            warnings.append("비실적 이슈 요약이 수치/시장반응 중심으로 치우침")
            actions.append("numeric_heavy_summary_detected")

    bad_korean = [line for line in result["fact_summary"] if _has_bad_korean_join(line)]
    if bad_korean:
        warnings.append("한국어 조사/띄어쓰기 오류가 남아 있음")
    if (
        main_company
        and main_company != _INDUSTRY_TREND_COMPANY_ID
        and result.get("is_valid_summary")
        and not _summary_mentions_company(result, main_company)
    ):
        warnings.append("요약 문장에 main_company alias가 없음")
    if source_article_ids:
        basis_source_ids = _dedupe_ints(
            [
                article_id
                for item in result.get("fact_basis", [])
                for article_id in _as_int_list(item.get("source_article_ids"))
            ]
        )
        if not basis_source_ids:
            warnings.append("fact_basis source_article_ids가 비어 있음")
    result["is_valid_summary"] = bool(result.get("is_valid_summary", True)) and not warnings
    if warnings:
        result["validation_warnings"] = _dedupe_keep_order(
            [*_normalize_string_list(result.get("validation_warnings")), *warnings]
        )
        result["reason"] = _append_reason(result.get("reason"), "; ".join(warnings))
    if actions:
        result["repair_actions"] = _dedupe_keep_order(
            [*_normalize_string_list(result.get("repair_actions")), *actions]
        )
    return result


def _summary_line_company_attribution_warning(
    *,
    line: str,
    evidence: str,
    main_company: str,
) -> str:
    if not main_company:
        return ""
    line_text = str(line or "")
    evidence_text = str(evidence or "")
    if not line_text.strip() or not evidence_text.strip():
        return ""
    main_aliases = _company_aliases_for_detection(main_company)
    if not _text_mentions_any_alias(line_text, main_aliases):
        return ""
    if _text_mentions_any_alias(evidence_text, main_aliases):
        return ""
    other_hits: list[str] = []
    for company_id, aliases in _KNOWN_COMPANY_ALIASES.items():
        if company_id == main_company:
            continue
        if _text_mentions_any_alias(evidence_text, _company_aliases_for_detection(company_id)):
            other_hits.append(company_id)
    if other_hits or _company_like_mentions(evidence_text):
        return "회사 주체 귀속 불일치: related fact evidence가 다른 피어사를 가리킴"
    return ""


def _company_aliases_for_detection(company_id: str) -> list[str]:
    if company_id == _INDUSTRY_TREND_COMPANY_ID:
        return _INDUSTRY_TREND_ALIASES
    aliases = [
        str(alias) for alias in _KNOWN_COMPANY_ALIASES.get(company_id, []) if str(alias).strip()
    ]
    aliases.append(str(company_id or ""))
    return _dedupe_keep_order(aliases)


def _text_mentions_any_alias(text: str, aliases: list[str]) -> bool:
    compact_text = _compact(text)
    for alias in aliases:
        compact_alias = _compact(alias)
        if compact_alias and compact_alias in compact_text:
            return True
    return False


def _company_like_mentions(text: str) -> list[str]:
    value = re.sub(r"\s+", " ", str(text or ""))
    patterns = (
        r"[가-힣A-Z]+(?:전자|SDS|CNS|하이닉스|클라우드|오토에버|DX|테크윈|엔솔)",
        r"(?:네이버|카카오|포스코|현대|삼성|SK|LG)[가-힣A-Z]*",
    )
    mentions: list[str] = []
    for pattern in patterns:
        mentions.extend(match.group(0) for match in re.finditer(pattern, value))
    return _dedupe_keep_order([mention for mention in mentions if len(mention) >= 2])


def _summary_line_role_counts(
    line_items: list[dict[str, Any]],
    fact_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in line_items:
        for fact_id in _normalize_string_list(item.get("fact_ids")):
            role = str((fact_by_id.get(fact_id) or {}).get("summary_role") or "")
            if role:
                counts[role] = counts.get(role, 0) + 1
    return counts


def _sync_summary_line_item_texts(
    line_items: list[dict[str, Any]],
    lines: list[str],
) -> list[dict[str, Any]]:
    synced: list[dict[str, Any]] = []
    for item in line_items:
        index = _safe_int(item.get("line_index"))
        updated = dict(item)
        if 1 <= index <= len(lines):
            updated["text"] = lines[index - 1]
        synced.append(updated)
    return synced


def normalize_subject_predicate_consistency(
    *,
    lines: list[str],
    line_items: list[dict[str, Any]],
    fact_by_id: dict[str, dict[str, Any]],
    main_company: str,
) -> tuple[list[str], list[str]]:
    aliases = sorted(_PEER_ALIASES.get(main_company, [main_company]), key=len, reverse=True)
    updated = list(lines)
    actions: list[str] = []
    company_started_indexes: list[int] = []

    for line_index, line in enumerate(lines):
        alias = _starting_company_alias(line, aliases)
        if not alias:
            continue
        company_started_indexes.append(line_index)
        if len(company_started_indexes) == 1:
            continue

        facts = _facts_for_line(line_index + 1, line_items, fact_by_id)
        entity = _primary_entity_for_summary_line(line_index + 1, line_items, fact_by_id)
        replacement = _rewrite_repeated_company_sentence(
            updated[line_index],
            alias=alias,
            entity=entity,
            facts=facts,
            line_index=line_index + 1,
        )
        if replacement != updated[line_index]:
            updated[line_index] = replacement
            actions.append(f"summary_line_{line_index + 1}_subject_replaced")
            actions.append("subject_predicate_consistency_normalized")
            actions.append("company_name_repetition_reduced")

    if len(company_started_indexes) >= 2:
        actions.insert(0, "company_name_repetition_detected")
    if actions:
        updated = [normalize_korean_spacing(line) for line in updated]
    return updated, _dedupe_keep_order(actions)


def _starting_company_alias(line: str, aliases: list[str]) -> str:
    for alias in aliases:
        if not alias:
            continue
        if re.match(rf"^\s*{re.escape(alias)}\s*(은|는|이|가)\s+", str(line or "")):
            return alias
    return ""


def _company_name_start_count(lines: list[str], main_company: str) -> int:
    aliases = sorted(_PEER_ALIASES.get(main_company, [main_company]), key=len, reverse=True)
    return sum(1 for line in lines if _starting_company_alias(line, aliases))


def _primary_entity_for_summary_line(
    line_index: int,
    line_items: list[dict[str, Any]],
    fact_by_id: dict[str, dict[str, Any]],
) -> str:
    fact_ids: list[str] = []
    for item in line_items:
        if _safe_int(item.get("line_index")) == line_index:
            fact_ids = _normalize_string_list(item.get("fact_ids"))
            break
    for fact_id in fact_ids:
        fact = fact_by_id.get(fact_id)
        if not fact:
            continue
        for entity in _normalize_string_list(fact.get("entities")):
            if len(_compact(entity)) >= 2:
                return entity
    return ""


def _rewrite_repeated_company_sentence(
    line: str,
    *,
    alias: str,
    entity: str,
    facts: list[dict[str, Any]],
    line_index: int,
) -> str:
    match = re.match(
        rf"^\s*{re.escape(alias)}\s*(?:은|는|이|가)\s+(.+)$",
        str(line or "").strip(),
    )
    if not match:
        return line
    rest = match.group(1).strip()
    subject_type = _summary_subject_type(facts)
    if entity and _compact(entity) in _compact(rest[: max(len(entity) + 12, 24)]):
        if subject_type == "product_context":
            return _entity_context_sentence(entity, rest)
        return rest
    if subject_type == "reported_context":
        return _reported_context_sentence(rest)
    if subject_type == "product_context":
        return f"{_product_context_subject(facts)} {rest}"
    if subject_type == "market_context":
        return f"시장 반응은 {rest}"
    if subject_type == "metric_context":
        return _reported_context_sentence(rest)
    if subject_type == "risk_context":
        return f"해당 이슈는 {rest}"
    if line_index == 3:
        return _reported_context_sentence(rest)
    return line


def _summary_subject_type(facts: list[dict[str, Any]]) -> str:
    roles = {str(fact.get("summary_role") or "") for fact in facts}
    fact_types = {str(fact.get("fact_type") or "") for fact in facts}
    if "market_reaction" in roles or "market_fact" in fact_types:
        return "market_context"
    if "numeric_effect" in roles or "numeric_fact" in fact_types:
        return "metric_context"
    if "risk_detail" in roles or "risk_fact" in fact_types:
        return "risk_context"
    if "uncertainty_detail" in roles or "uncertain_fact" in fact_types:
        return "reported_context"
    if "application_case" in roles or "application_fact" in fact_types:
        return "reported_context"
    if (
        "product_definition" in roles
        or "service_function" in roles
        or "platform_definition_fact" in fact_types
    ):
        return "product_context"
    return "company_context"


def _product_context_subject(facts: list[dict[str, Any]]) -> str:
    roles = {str(fact.get("summary_role") or "") for fact in facts}
    if "service_function" in roles:
        return "해당 서비스는"
    return "해당 플랫폼은"


def _entity_context_sentence(entity: str, rest: str) -> str:
    body = _strip_leading_entity_subject(rest, entity)
    body = _extract_reported_clause(body)
    body = body.rstrip(".")
    if not body:
        return rest
    return normalize_korean_spacing(f"{_topic_subject(entity)} {body}.")


def _strip_leading_entity_subject(rest: str, entity: str) -> str:
    match = re.match(
        rf"^\s*{re.escape(entity)}\s*(?:은|는|이|가)\s+(.+)$",
        str(rest or "").strip(),
    )
    return match.group(1).strip() if match else str(rest or "").strip()


def _extract_reported_clause(text: str) -> str:
    value = str(text or "").strip().rstrip(".")
    match = re.match(r"^(.+?고)\s+\S+다$", value)
    if match:
        clause = match.group(1).strip()
        return clause[:-1].strip() if clause.endswith("고") else clause
    return value


def _topic_subject(entity: str) -> str:
    value = str(entity or "").strip()
    if not value:
        return "해당 항목은"
    last = value[-1]
    code = ord(last)
    if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28:
        return f"{value}은"
    return f"{value}는"


def _reported_context_sentence(rest: str) -> str:
    value = _clean_summary_line(rest)
    return value if value else "관련 사실이 확인됐다."


def _clean_summary_line(line: str) -> str:
    value = normalize_korean_spacing(line).strip()
    value = re.sub(r"^기사에서는\s+", "", value)
    value = re.sub(r"\s*사실이\s+확인됐다\.?$", ".", value)
    value = re.sub(r"\s*사실이\s+확인됐습니다\.?$", ".", value)
    return normalize_korean_spacing(value)


def _nominalize_korean_predicate(text: str) -> str:
    value = str(text or "").strip().rstrip(".")
    suffix_map = (
        ("했다", "한"),
        ("한다", "하는"),
        ("됐다", "된"),
        ("된다", "되는"),
        ("있다", "있는"),
        ("이었다", "이었던"),
        ("이다", "인"),
    )
    for suffix, replacement in suffix_map:
        if value.endswith(suffix):
            return value[: -len(suffix)] + replacement
    return value


def _selected_facts_by_line(
    selected_fact_ids: dict[str, list[str]],
    extracted_facts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    return {
        str(index): [
            _compact_fact_for_prompt(fact_by_id[fact_id])
            for fact_id in selected_fact_ids.get(str(index), [])
            if fact_id in fact_by_id
        ]
        for index in range(1, _SUMMARY_LINE_MAX + 1)
    }


def _compact_facts_for_prompt(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_fact_for_prompt(fact) for fact in facts]


def _compact_fact_for_prompt(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": fact.get("fact_id"),
        "article_id": fact.get("article_id"),
        "fact_type": fact.get("fact_type"),
        "summary_role": fact.get("summary_role"),
        "role_priority": fact.get("role_priority"),
        "evidence_text": fact.get("evidence_text"),
        "normalized_fact": fact.get("normalized_fact"),
        "entities": fact.get("entities", []),
        "numbers": fact.get("numbers", []),
        "dates": fact.get("dates", []),
        "event_verbs": fact.get("event_verbs", []),
        "confidence": fact.get("confidence"),
    }


def _facts_for_line(
    index: int,
    line_items: list[dict[str, Any]],
    fact_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    fact_ids: list[str] = []
    for item in line_items:
        if _safe_int(item.get("line_index")) == index:
            fact_ids = _normalize_string_list(item.get("fact_ids"))
            break
    return [fact_by_id[fact_id] for fact_id in fact_ids if fact_id in fact_by_id]


def _compose_fallback_line(index: int, facts: list[dict[str, Any]]) -> str:
    if not facts:
        return ""
    if len(facts) == 1:
        return _fact_text_for_summary_line(facts[0])
    texts = [_fact_text_for_summary_line(fact).rstrip(".") for fact in facts[:2]]
    if index == 2:
        return normalize_korean_spacing(f"{texts[0]}고, {texts[1]}.")
    return normalize_korean_spacing(f"{texts[0]}; {texts[1]}.")


def _fact_text_for_summary_line(fact: dict[str, Any]) -> str:
    text = str(fact.get("normalized_fact") or fact.get("evidence_text") or "").strip()
    text = normalize_korean_spacing(text)
    if fact.get("fact_type") == "uncertain_fact":
        text = _soften_uncertain_sentence(text)
    if text and not text.endswith((".", "다", "요", "죠")):
        text = f"{text}."
    return text


def _soften_uncertain_sentence(text: str) -> str:
    sentence = normalize_korean_spacing(text).rstrip()
    if _has_uncertain_fact_marker(sentence):
        return sentence
    if sentence.endswith("했다."):
        return sentence[:-3] + "한 것으로 소개됐다."
    if sentence.endswith("한다."):
        return sentence[:-3] + "하는 방향으로 제시됐다."
    if sentence.endswith("있다."):
        return sentence[:-3] + "있는 것으로 설명됐다."
    return sentence


def _title_to_fact_sentence(title: str) -> str:
    text = re.sub(r"^\[[^\]]+\]\s*", "", str(title or "")).strip()
    text = normalize_korean_spacing(text)
    return text if text.endswith((".", "다", "요", "죠")) else f"{text}."


def _date_tokens(text: str) -> list[str]:
    return _dedupe_keep_order(
        [
            match.group(0).strip()
            for match in re.finditer(
                r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{1,2}월\s*\d{1,2}일|\d{4}-\d{2}-\d{2}",
                str(text or ""),
            )
        ]
    )


def _event_verbs_in_text(text: str) -> list[str]:
    del text
    return []


def _has_bad_korean_join(text: str) -> bool:
    value = str(text or "")
    return bool(re.search(r"(를|을|는|은|이|가)로\b", value))


def _dedupe_similar_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if any(_text_similarity(text, existing) >= 0.8 for existing in result):
            continue
        result.append(text)
    return result


def _text_similarity(left: str, right: str) -> float:
    left_compact = _compact(left)
    right_compact = _compact(right)
    if not left_compact or not right_compact:
        return 0.0
    return SequenceMatcher(None, left_compact, right_compact).ratio()


def _split_evidence_sentences(text: str, limit: int = 80) -> list[str]:
    chunks = re.split(
        r"(?<=[.!?。！？])\s+|(?<=[다요죠임음])\.\s*|\n+",
        _strip_article_ui_boilerplate(str(text or "")),
    )
    sentences: list[str] = []
    for chunk in chunks:
        sentence = re.sub(r"\s+", " ", chunk).strip()
        if len(sentence) < 8:
            continue
        if _is_article_ui_boilerplate(sentence):
            continue
        sentences.append(sentence[:500])
        if len(sentences) >= limit:
            break
    return sentences


def _strip_article_ui_boilerplate(text: str) -> str:
    value = str(text or "")
    for marker in _ARTICLE_UI_BOILERPLATE_MARKERS:
        value = value.replace(marker, " ")
    value = re.sub(r"\b[가]?\s*(?:작게|보통|크게|아주\s*크게)\s*[가]?\b", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _is_article_ui_boilerplate(sentence: str) -> bool:
    compact = re.sub(r"\s+", "", str(sentence or ""))
    if not compact:
        return True
    marker_hits = sum(
        1 for marker in _ARTICLE_UI_BOILERPLATE_MARKERS if marker.replace(" ", "") in compact
    )
    if marker_hits >= 2:
        return True
    if marker_hits and len(compact) < 120:
        return True
    share_markers = ("기사공유", "주소복사", "다크모드", "프린트", "채널구독")
    return sum(1 for marker in share_markers if marker in compact) >= 2


def _number_tokens(text: str) -> list[str]:
    return _dedupe_keep_order(
        [
            _normalize_number_token(match.group(0))
            for match in _NUMBER_TOKEN_PATTERN.finditer(str(text or ""))
            if _normalize_number_token(match.group(0))
        ]
    )


def _normalize_number_token(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").replace(",", "")).strip()


def _number_token_covered(number: str, evidence_numbers: list[str]) -> bool:
    normalized = _normalize_number_token(number)
    if not normalized:
        return True
    number_digits = re.sub(r"[^0-9.]", "", normalized)
    for evidence in evidence_numbers:
        evidence_normalized = _normalize_number_token(evidence)
        evidence_digits = re.sub(r"[^0-9.]", "", evidence_normalized)
        if normalized == evidence_normalized:
            return True
        if number_digits and number_digits == evidence_digits:
            return True
    return False


def normalize_korean_spacing(value: Any) -> str:
    """요약 출력에서 자주 붙는 한국어 조사를 보수적으로 교정한다."""
    text = str(value or "")
    text = re.sub(
        r"([가-힣A-Za-z0-9])(['\"‘’“”])\s*(를|을|은|는|이|가|와|과|에|에서|로|으로)",
        r"\1\2\3",
        text,
    )
    text = re.sub(r"(를|을|은|는|이|가|와|과|에|에서|로|으로)(?=[A-Z][A-Za-z])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _candidate_peer_companies(articles: list[dict[str, Any]]) -> list[str]:
    """Return the actual target companies for this cluster.

    Use preprocessing outputs by default. When the cluster itself is an explicit
    peer-comparison issue, preserve the peer aliases in the article body so the
    IntegratedIssue does not collapse to a single representative company.
    """
    if _is_industry_trend_cluster(articles):
        return [_INDUSTRY_TREND_COMPANY_ID]

    candidates: list[str] = []
    for article in articles:
        candidates.extend(_company_list(article))
        candidates.extend(_matched_companies(article))
    if _is_peer_comparison_issue(articles):
        candidates.extend(_body_peer_companies(articles))

    return [
        company_id
        for company_id in _dedupe_keep_order(candidates)
        if company_id in _PEER_ALIASES and company_tier(company_id) != "self"
    ]


def _is_industry_trend_cluster(articles: list[dict[str, Any]]) -> bool:
    for article in articles:
        if _INDUSTRY_TREND_COMPANY_ID in _company_list(article):
            return True
        if str(article.get("source_name") or "").strip() == "naver_industry_news":
            return True
        metadata = _metadata(article)
        if metadata.get("topic_scope") == _INDUSTRY_TREND_COMPANY_ID:
            return True
        if metadata.get("company_scope") == "industry":
            return True
    return False


def _body_peer_companies(articles: list[dict[str, Any]]) -> list[str]:
    text = _articles_text(articles)
    mentioned: list[str] = []
    for company_id, aliases in _PEER_ALIASES.items():
        if company_tier(company_id) == "self":
            continue
        if any(
            alias and re.search(re.escape(str(alias)), text, re.IGNORECASE) for alias in aliases
        ):
            mentioned.append(company_id)
    return mentioned


def _is_peer_comparison_issue(articles: list[dict[str, Any]]) -> bool:
    text = _articles_text(articles)
    if not text:
        return False
    mentioned_count = len(_body_peer_companies(articles))
    if mentioned_count < 2:
        return False
    comparison_signal = re.search(
        r"비교|대조|엇갈|반면|내부거래|의존도|비중|증가|감소|상승|하락",
        text,
    )
    metric_signal = len(_number_tokens(text)) >= 2
    return bool(comparison_signal and metric_signal)


def _articles_text(articles: list[dict[str, Any]]) -> str:
    return " ".join(
        normalize_korean_spacing(f"{article.get('title') or ''}. {article.get('content') or ''}")
        for article in articles
    )


def _enrich_peer_comparison_issue(
    summary: dict[str, Any],
    *,
    articles: list[dict[str, Any]],
    target_companies: list[str],
) -> dict[str, Any]:
    if not _is_peer_comparison_issue(articles):
        return summary
    out = dict(summary)
    mentioned = _dedupe_keep_order([*target_companies, *_body_peer_companies(articles)])
    comparison_facts = _peer_metric_comparison_facts(articles, mentioned)
    risk_facts = _risk_facts_from_articles(articles)
    market_structure_facts = _market_structure_facts_from_articles(articles)
    evidence_inventory = _strategic_evidence_inventory_from_articles(
        articles,
        comparison_facts=comparison_facts,
    )
    if (
        not comparison_facts
        and not risk_facts
        and not market_structure_facts
        and not any(evidence_inventory.values())
    ):
        return out

    out["mentioned_peer_companies"] = mentioned or out.get("mentioned_peer_companies")
    out["target_peer_companies"] = mentioned or out.get("target_peer_companies")
    out["issue_frame"] = {
        "frame_type": "peer_comparison",
        "comparison_axis": _comparison_axis_from_facts(comparison_facts),
        "main_company": out.get("main_company"),
        "mentioned_peer_companies": mentioned,
    }
    out["comparison_facts"] = comparison_facts
    out["risk_facts"] = risk_facts
    out["market_structure_facts"] = market_structure_facts
    out["strategic_evidence_inventory"] = evidence_inventory
    out["supporting_facts"] = evidence_inventory.get("supporting_facts", [])
    out["background_facts"] = evidence_inventory.get("background_facts", [])
    out["cause_or_driver_facts"] = evidence_inventory.get("cause_or_driver_facts", [])
    out["uncertainty_or_limitation_facts"] = evidence_inventory.get(
        "uncertainty_or_limitation_facts",
        [],
    )
    out["strategic_tensions"] = evidence_inventory.get("strategic_tensions", [])
    out["actionable_questions"] = evidence_inventory.get("actionable_questions", [])

    existing_lines = _normalize_string_list(out.get("fact_summary"))
    added_lines = _comparison_summary_lines(
        comparison_facts=comparison_facts,
        risk_facts=risk_facts,
        market_structure_facts=market_structure_facts,
    )
    out["fact_summary"] = _dedupe_keep_order([*existing_lines, *added_lines])[:_SUMMARY_LINE_MAX]
    if comparison_facts:
        out["headline"] = _peer_comparison_headline(comparison_facts) or out.get("headline")
        out["one_line_summary"] = _peer_comparison_one_liner(comparison_facts) or out.get(
            "one_line_summary"
        )
    out["integrated_text"] = " ".join(
        [
            str(out.get("one_line_summary") or ""),
            " ".join(out["fact_summary"]),
            " ".join(risk_facts),
            " ".join(market_structure_facts),
            " ".join(evidence_inventory.get("background_facts", [])),
            " ".join(evidence_inventory.get("cause_or_driver_facts", [])),
            " ".join(evidence_inventory.get("uncertainty_or_limitation_facts", [])),
            " ".join(evidence_inventory.get("strategic_tensions", [])),
        ]
    ).strip()
    out["reason"] = _append_reason(
        out.get("reason"),
        "peer_comparison_enriched: 본문에 등장한 비교 피어와 구조적 리스크를 보존함",
    )
    return out


def _peer_metric_comparison_facts(
    articles: list[dict[str, Any]],
    company_ids: list[str],
) -> list[dict[str, Any]]:
    sentences = _split_evidence_sentences(_articles_text(articles), limit=80)
    facts: list[dict[str, Any]] = []
    for company_id in company_ids:
        aliases = _PEER_ALIASES.get(company_id, [company_id])
        candidates: list[dict[str, Any]] = []
        for sentence_index, sentence in enumerate(sentences):
            matched_alias = next(
                (
                    str(alias)
                    for alias in aliases
                    if alias and re.search(re.escape(str(alias)), sentence, re.IGNORECASE)
                ),
                "",
            )
            if not matched_alias:
                continue
            if not _nearby_percentage(sentence, matched_alias) and sentence_index + 1 < len(
                sentences
            ):
                sentence = f"{sentence} {sentences[sentence_index + 1]}"
            if not re.search(r"내부거래|특수관계자|전체\s*매출|매출", sentence):
                continue
            values = _number_tokens(sentence)
            percentages = [value for value in values if "%" in value or "％" in value]
            nearby_percentage = _nearby_percentage(sentence, matched_alias)
            if not values or (percentages and not nearby_percentage):
                continue
            candidates.append(
                {
                    "company_id": company_id,
                    "metric": (
                        "internal_transaction_ratio"
                        if re.search(r"내부거래|특수관계자", sentence)
                        else "financial_metric"
                    ),
                    "value": nearby_percentage or (percentages[0] if percentages else values[0]),
                    "numbers": values[:6],
                    "evidence_text": sentence,
                    "_score": _peer_metric_sentence_score(sentence, matched_alias),
                }
            )
        if candidates:
            selected = sorted(candidates, key=lambda item: item.get("_score", 0), reverse=True)[0]
            selected.pop("_score", None)
            facts.append(selected)
    return facts


def _nearby_percentage(sentence: str, alias: str) -> str:
    for match in re.finditer(re.escape(alias), sentence, re.IGNORECASE):
        windows = [
            sentence[match.start() : min(len(sentence), match.end() + 55)],
            sentence[max(0, match.start() - 35) : min(len(sentence), match.end() + 45)],
        ]
        for window in windows:
            percentage_match = re.search(r"\d+(?:\.\d+)?\s*[%％]", window)
            if percentage_match:
                return percentage_match.group(0).replace("％", "%").strip()
    return ""


def _peer_metric_sentence_score(sentence: str, alias: str) -> int:
    score = 0
    matches = list(re.finditer(re.escape(alias), sentence, re.IGNORECASE))
    if not matches:
        return score
    after_alias = max((sentence[match.end() :] for match in matches), key=len)
    if re.search(r"내부거래|특수관계자|전체\s*매출|매출", after_alias[:80]):
        score += 4
    if re.search(r"\d+(?:\.\d+)?\s*[%％]", after_alias[:80]):
        score += 4
    if re.search(r"\d+\.\d+\s*[%％]", after_alias[:80]):
        score += 2
    if re.search(r"↑|↓|증가|감소|상승|하락", sentence):
        score += 1
    if re.search(rf"^\s*{re.escape(alias)}\s*(?:은|는|이|가|의|역시)", sentence, re.IGNORECASE):
        score += 6
    if re.search(r"^\s*[^.]{0,8}" + re.escape(alias), sentence, re.IGNORECASE):
        score += 1
    if (
        not re.search(
            rf"^\s*{re.escape(alias)}\s*(?:은|는|이|가|의|역시)",
            sentence,
            re.IGNORECASE,
        )
        and len(_mentioned_peer_ids_in_text(sentence)) >= 2
    ):
        score -= 6
    return score


def _mentioned_peer_ids_in_text(text: str) -> list[str]:
    mentioned: list[str] = []
    for company_id, aliases in _PEER_ALIASES.items():
        if any(
            alias and re.search(re.escape(str(alias)), text, re.IGNORECASE) for alias in aliases
        ):
            mentioned.append(company_id)
    return mentioned


def _risk_facts_from_articles(articles: list[dict[str, Any]]) -> list[str]:
    patterns = (r"공정위|공정거래위원회", r"사법\s*리스크|과징금|규제|감시")
    return _sentences_matching_any(articles, patterns, limit=3)


def _market_structure_facts_from_articles(articles: list[dict[str, Any]]) -> list[str]:
    patterns = (
        r"외부\s*경쟁력|외부\s*고객|대외\s*사업|홀로서기",
        r"클라우드&AI|AI\s*성과|재무제표|분리|검증",
        r"그룹\s*일감|계열사\s*일감|내부거래",
    )
    return _sentences_matching_any(articles, patterns, limit=4)


def _strategic_evidence_inventory_from_articles(
    articles: list[dict[str, Any]],
    *,
    comparison_facts: list[dict[str, Any]],
) -> dict[str, list[str]]:
    background_facts = _sentences_matching_any(
        articles,
        (
            r"업종|시장|업계|구조|태생적|특성상|최근\s*\d+\s*년|평균|상위권",
            r"공정위|공정거래위원회|규제|감시|공시대상기업집단",
        ),
        limit=5,
    )
    cause_or_driver_facts = _sentences_matching_any(
        articles,
        (
            r"주효|영향|결과|풀이|때문|연관|연결|배경|전략|확장|강화|전념|주도|전담",
            r"선정|공공사업|정부|그룹\s*차원|신사업|플랫폼|자동화|인프라",
        ),
        limit=6,
    )
    risk_facts = _sentences_matching_any(
        articles,
        (
            r"리스크|위험|한계|문제|족쇄|악화|제한|비판|규제|과징금|소송|감시",
            r"자유로울\s*수\s*없|검증하기\s*어렵|분리하지\s*않",
        ),
        limit=6,
    )
    uncertainty_or_limitation_facts = _sentences_matching_any(
        articles,
        (
            r"검증하기\s*어렵|확인하기\s*어렵|분리하지\s*않|투명하게\s*분리|한계",
            r"목소리가\s*높|따로\s*떼어내|공시|재무제표",
        ),
        limit=4,
    )
    market_structure_facts = _sentences_matching_any(
        articles,
        (
            r"외부\s*고객|외부\s*시장|대외\s*사업|대외\s*성과|독자적인\s*가치",
            r"계열사\s*일감|그룹\s*일감|모그룹|내부거래|거래\s*구조|고착",
        ),
        limit=6,
    )
    strategic_tensions = _strategic_tension_facts(
        background_facts=background_facts,
        market_structure_facts=market_structure_facts,
        risk_facts=risk_facts,
        uncertainty_facts=uncertainty_or_limitation_facts,
    )
    supporting_facts = _dedupe_keep_order(
        [
            *market_structure_facts,
            *cause_or_driver_facts,
            *risk_facts,
            *uncertainty_or_limitation_facts,
        ]
    )[:8]
    return {
        "core_facts": _comparison_core_fact_lines(comparison_facts),
        "supporting_facts": supporting_facts,
        "background_facts": background_facts,
        "cause_or_driver_facts": cause_or_driver_facts,
        "risk_facts": risk_facts,
        "uncertainty_or_limitation_facts": uncertainty_or_limitation_facts,
        "market_structure_facts": market_structure_facts,
        "strategic_tensions": strategic_tensions,
        "actionable_questions": _actionable_questions_from_inventory(
            comparison_facts=comparison_facts,
            background_facts=background_facts,
            cause_or_driver_facts=cause_or_driver_facts,
            risk_facts=risk_facts,
            uncertainty_facts=uncertainty_or_limitation_facts,
        ),
    }


def _comparison_core_fact_lines(comparison_facts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for fact in comparison_facts[:8]:
        company = _company_display_name(fact.get("company_id"))
        value = str(fact.get("value") or "").strip()
        evidence = str(fact.get("evidence_text") or "").strip()
        if company and value and evidence:
            lines.append(f"{company} {value}: {evidence}")
        elif evidence:
            lines.append(evidence)
    return _dedupe_keep_order(lines)


def _strategic_tension_facts(
    *,
    background_facts: list[str],
    market_structure_facts: list[str],
    risk_facts: list[str],
    uncertainty_facts: list[str],
) -> list[str]:
    tensions: list[str] = []
    if background_facts and market_structure_facts:
        background = background_facts[0]
        market = next(
            (
                fact
                for fact in market_structure_facts
                if _inventory_fact_key(fact) != _inventory_fact_key(background)
            ),
            "",
        )
        tensions.append(f"{background} / {market}" if market else background)
    if risk_facts:
        tensions.append(risk_facts[0])
    if uncertainty_facts:
        tensions.append(uncertainty_facts[0])
    return _dedupe_keep_order(tensions)[:4]


def _inventory_fact_key(value: Any) -> str:
    return re.sub(r"\W+", "", normalize_korean_spacing(value)).lower()


def _actionable_questions_from_inventory(
    *,
    comparison_facts: list[dict[str, Any]],
    background_facts: list[str],
    cause_or_driver_facts: list[str],
    risk_facts: list[str],
    uncertainty_facts: list[str],
) -> list[str]:
    questions: list[str] = []
    if comparison_facts:
        axis = _comparison_axis_from_facts(comparison_facts)
        questions.append(
            f"SK AX는 {axis} 비교축을 내부 관리 지표로 어떻게 분리해 설명할 수 있는가?"
        )
    if cause_or_driver_facts:
        questions.append(
            "피어 간 차이가 사업 수행 방식, 고객 기반, 그룹 과제 중 어디에서 발생했는가?"
        )
    if risk_facts:
        questions.append(
            "해당 차이가 규제, 고객 확보, 재무 안정성, 운영 책임 중 어떤 리스크로 이어지는가?"
        )
    if background_facts or uncertainty_facts:
        questions.append(
            "외부에 설명 가능한 성과와 아직 검증하기 어려운 영역을 어떻게 구분할 것인가?"
        )
    return _dedupe_keep_order(questions)[:4]


def _sentences_matching_any(
    articles: list[dict[str, Any]],
    patterns: tuple[str, ...],
    *,
    limit: int,
) -> list[str]:
    out: list[str] = []
    for sentence in _split_evidence_sentences(_articles_text(articles), limit=100):
        sentence = _clean_inventory_sentence(sentence)
        if not sentence:
            continue
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
            out.append(sentence)
        if len(out) >= limit:
            break
    return _dedupe_keep_order(out)


def _clean_inventory_sentence(sentence: str) -> str:
    value = normalize_korean_spacing(sentence)
    if not value:
        return ""
    if re.search(r"…|\.{3}|VS", value) and re.search(r"\d{1,2}일\s+업계에\s+따르면", value):
        value = re.sub(r"^.*?(?=\d{1,2}일\s+업계에\s+따르면)", "", value).strip()
    if re.search(r"…|\.{3}|↑|↓|VS", value) and len(value) < 120:
        return ""
    return value


def _comparison_axis_from_facts(facts: list[dict[str, Any]]) -> str:
    if any(fact.get("metric") == "internal_transaction_ratio" for fact in facts):
        return "internal_transaction_ratio"
    return "peer_metric_comparison"


def _comparison_summary_lines(
    *,
    comparison_facts: list[dict[str, Any]],
    risk_facts: list[str],
    market_structure_facts: list[str],
) -> list[str]:
    lines: list[str] = []
    if len(comparison_facts) >= 2:
        values = [
            f"{_company_display_name(fact.get('company_id'))} {fact.get('value')}"
            for fact in comparison_facts
            if fact.get("company_id") and fact.get("value")
        ]
        if values:
            lines.append(
                "SI 업계 내부거래 의존도 비교에서 " + ", ".join(values[:4]) + "가 제시됐다."
            )
    if risk_facts:
        lines.append(risk_facts[0])
    if market_structure_facts:
        lines.append(market_structure_facts[-1])
    return lines


def _peer_comparison_headline(comparison_facts: list[dict[str, Any]]) -> str:
    if any(fact.get("metric") == "internal_transaction_ratio" for fact in comparison_facts):
        return "SI업계, 내부거래 의존도 격차 확대"
    return ""


def _peer_comparison_one_liner(comparison_facts: list[dict[str, Any]]) -> str:
    if len(comparison_facts) < 2:
        return ""
    values = [
        f"{_company_display_name(fact.get('company_id'))} {fact.get('value')}"
        for fact in comparison_facts
        if fact.get("company_id") and fact.get("value")
    ]
    if not values:
        return ""
    return "SI 업계에서 " + ", ".join(values[:4]) + " 등 내부거래 의존도 격차가 제시됐다."


def _company_display_name(company_id: Any) -> str:
    aliases = _PEER_ALIASES.get(str(company_id or ""), [])
    return str(aliases[0] if aliases else company_id or "")


def _company_list(article: dict[str, Any]) -> list[str]:
    return _normalize_string_list(article.get("company"))


def _matched_companies(article: dict[str, Any]) -> list[str]:
    values = _normalize_string_list(article.get("matched_companies"))
    metadata = _metadata(article)
    values.extend(_normalize_string_list(metadata.get("matched_companies")))
    return _dedupe_keep_order(values)


def _metadata(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("metadata") or article.get("extra") or {}
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _summary_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "subtitle",
        "peer_relevance",
        "peer_relevance_reason",
        "matched_aliases_by_peer",
        "body_company_mentions",
    )
    return {key: metadata[key] for key in keep_keys if key in metadata}


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, tuple):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _target_company_aliases(company_ids: list[str]) -> dict[str, list[str]]:
    return {
        company_id: (
            _INDUSTRY_TREND_ALIASES
            if company_id == _INDUSTRY_TREND_COMPANY_ID
            else _PEER_ALIASES.get(company_id, [company_id])
        )
        for company_id in company_ids
    }


def _summary_mentions_company(summary: dict[str, Any], company_id: str) -> bool:
    aliases = (
        _INDUSTRY_TREND_ALIASES
        if company_id == _INDUSTRY_TREND_COMPANY_ID
        else _PEER_ALIASES.get(company_id, [company_id])
    )
    compact_text = _compact(
        " ".join(
            [
                str(summary.get("headline") or ""),
                str(summary.get("one_line_summary") or ""),
                str(summary.get("main_event") or ""),
                " ".join(_normalize_string_list(summary.get("fact_summary"))),
            ]
        )
    )
    return any(_compact(alias) and _compact(alias) in compact_text for alias in aliases)


def _clean_domain_term(term: str) -> str:
    term = re.sub(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$", "", term)
    return re.sub(r"\s+", " ", term).strip()


def _empty_summary(
    cluster_id: int,
    representative_id: int,
    reason: str,
    source_article_ids: list[int] | None = None,
    cluster_article_ids: list[int] | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_ids = source_article_ids or []
    return {
        "cluster_id": cluster_id,
        "representative_id": representative_id,
        "source_article_ids": source_ids,
        "cluster_article_ids": cluster_article_ids or source_ids,
        "analyzed_article_ids": source_ids,
        "summary_scope": "peer_company_fact_only",
        "excluded_company_tiers": ["self"],
        "target_peer_companies": [],
        "is_valid_summary": False,
        "main_company": "",
        "mentioned_peer_companies": [],
        "cluster_event_type": "unknown",
        "headline": "",
        "one_line_summary": "",
        "fact_summary": [],
        "main_event": "",
        "fact_basis": [],
        "cluster_fact_intelligence": {
            "common_facts": [],
            "unique_facts": [],
            "uncertain_facts": [],
            "conflict_notes": [],
            "numbers_and_dates": [],
            "customers_or_industries": [],
            "products_or_services": [],
            "activity_types": [],
        },
        "coverage": coverage
        or _coverage_info(
            cluster_article_count=len(cluster_article_ids or source_ids),
            analyzed_article_count=len(source_ids),
            warning="",
        ),
        "confidence": 0.0,
        "reason": reason,
        "model": _LLM_MODEL,
    }


def _render_prompt(template: str) -> str:
    return template.replace("{event_types}", _EVENT_TYPE_VALUES)


def _article_ids(articles: list[dict[str, Any]]) -> list[int]:
    return [
        article_id
        for article_id in (_article_numeric_id(article) for article in articles)
        if article_id > 0
    ]


def _article_numeric_id(article: dict[str, Any]) -> int:
    for key in ("id", "preprocess_id"):
        value = article.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _coverage_info(
    *,
    cluster_article_count: int,
    analyzed_article_count: int,
    warning: str,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ratio = analyzed_article_count / cluster_article_count if cluster_article_count else 0.0
    coverage_warning = warning
    if cluster_article_count and analyzed_article_count < cluster_article_count:
        coverage_warning = (
            f"{coverage_warning}; " if coverage_warning else ""
        ) + "일부 cluster_article_ids는 metadata로만 보존하고 fact extraction에서 제외"
    info = {
        "cluster_article_count": cluster_article_count,
        "analyzed_article_count": analyzed_article_count,
        "coverage_ratio": round(ratio, 4),
        "coverage_warning": coverage_warning,
    }
    if selection:
        info.update(
            {
                "selection_status": selection.get("status"),
                "analysis_article_limit": selection.get("analysis_article_limit"),
                "majority_ratio": selection.get("majority_ratio"),
                "majority_article_count": len(selection.get("majority_article_ids") or []),
                "outlier_article_count": len(selection.get("outlier_article_ids") or []),
                "excluded_article_count": len(selection.get("excluded_article_ids") or []),
            }
        )
    return info


def _as_int_list(value: Any) -> list[int]:
    return [_safe_int(item) for item in _as_list(value) if _safe_int(item) > 0]


def _normalize_event_type(value: Any) -> str:
    event_type = str(value or "").strip()
    return event_type if event_type in _EVENT_TYPES else "unknown"


def _fact_key(value: str) -> str:
    compacted = _compact(value)
    return compacted[:120]


def _has_unique_fact_importance(text: str) -> bool:
    value = str(text or "")
    if re.search(r"\d", value):
        return True
    return _has_business_scope_terms(value) and _has_detail_preservation_terms(value)


def _detect_conflict_notes(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del facts
    return []


def _has_uncertain_fact_marker(sentence: str) -> bool:
    del sentence
    return False


def _append_reason(current: Any, reason: str) -> str:
    current_text = str(current or "").strip()
    if not current_text:
        return reason
    if reason in current_text:
        return current_text
    return f"{current_text}; {reason}"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _chunked(values: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parse_json(text: str) -> dict[str, Any]:
    parsed, _status = _safe_json_loads(text)
    return parsed


def _safe_json_loads(text: str) -> tuple[dict[str, Any], str]:
    cleaned = _clean_json_response(text)
    candidates = _dedupe_keep_order(
        [
            cleaned,
            _extract_json_object_text(cleaned),
            _escape_json_string_newlines(cleaned),
            _escape_json_string_newlines(_extract_json_object_text(cleaned)),
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return (parsed if isinstance(parsed, dict) else {}), "parsed"

    repaired = _repair_json_text(cleaned)
    if repaired:
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON repair failed: {exc}") from exc
        return (parsed if isinstance(parsed, dict) else {}), "repaired"

    raise ValueError("JSON parse failed")


def _clean_json_response(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        parts = value.split("```")
        if len(parts) >= 3:
            value = parts[1]
            if value.lstrip().startswith("json"):
                value = value.lstrip()[4:]
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)
    return value.strip()


def _extract_json_object_text(text: str) -> str:
    value = str(text or "")
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return value.strip()
    return value[start : end + 1].strip()


def _escape_json_string_newlines(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    for char in str(text or ""):
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\":
            result.append(char)
            escaped = True
            continue
        if char == '"':
            result.append(char)
            in_string = not in_string
            continue
        if in_string and char in {"\n", "\r"}:
            result.append("\\n")
            continue
        result.append(char)
    return "".join(result).strip()


def _repair_json_text(text: str) -> str:
    source = _extract_json_object_text(text)
    try:
        from json_repair import repair_json
    except Exception:
        return _escape_json_string_newlines(source)
    repaired = repair_json(source)
    if isinstance(repaired, dict):
        return json.dumps(repaired, ensure_ascii=False)
    return str(repaired or "").strip()


def _normalize_content(text: str) -> str:
    stripped = " ".join(text.split())
    return stripped


def _compact(text: str) -> str:
    return "".join(str(text or "").lower().split())


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)
