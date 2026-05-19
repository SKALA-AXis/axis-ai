"""피어사 뉴스 사실 요약 에이전트.

클러스터에 묶인 기사들을 바탕으로 피어사 관련 본문 내용을 요약한다.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any

from langchain_core.runnables import RunnableConfig
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


_LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
_PROMPT_VERSION = "summary-v4.0"
_FACT_EXTRACTION_BATCH_SIZE = 10
_FACT_EXTRACTION_MAX_TOKENS = _env_int("FACT_EXTRACTION_MAX_TOKENS", 3000)
_SUMMARY_MAX_TOKENS = _env_int("SUMMARY_MAX_TOKENS", 1500)
_VALIDATION_MAX_TOKENS = _env_int("VALIDATION_MAX_TOKENS", 1200)
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
_PEER_ALIASES = {
    company_id: aliases
    for company_id, aliases in {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}.items()
    if company_tier(company_id) != "self"
}

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm

    if _llm is None:
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
- selected_facts_by_line에 제공된 fact_id와 normalized_fact만 사용해 3문장 요약을 만듭니다.
- 시사점, 대응방향, 전략 해석, 회사 프로필 참조는 금지합니다.

공통 규칙:
1. summary_lines는 정확히 3개입니다.
2. 각 문장은 반드시 fact_ids를 1개 이상 포함해야 합니다.
3. fact_ids는 입력 selected_facts_by_line에 있는 값만 사용하세요.
4. summary line은 연결된 fact_ids의 normalized_fact/evidence_text에서 확인되는 사실만 사용하세요.
5. 기사에 없는 제품명, 서비스명, 고객명, 수치, 날짜, 원인은 만들지 마세요.
6. 수치나 날짜를 쓰려면 연결된 fact의 evidence_text에 같은 수치나 날짜가 있어야 합니다.
7. uncertain_fact를 사용하는 문장은 확정 표현을 피하고 "소개됐다", "언급됐다", "제시됐다", "설명됐다"처럼 원문 수위를 유지하세요.
8. 세 문장은 같은 내용을 반복하지 말고 서로 다른 역할을 가져야 합니다.
9. 특정 기사 키워드를 규칙처럼 추가하지 말고, 제품명/서비스명/플랫폼명/프로젝트명/이벤트명/기술명 같은 정보 유형을 기준으로 작성하세요.
10. 같은 회사명으로 시작하는 문장은 최대 1개만 두세요. 2문장과 3문장은 의미가 분명하면 제품명/플랫폼명/서비스명/해당 기술/기사에서는 등으로 이어가세요.
11. fact에 구체 수치·개수·기간·범위·장소·현장이 있으면 특히 3문장에 우선 반영하되, 연결된 fact evidence_text에서 검증되는 경우에만 쓰세요.
12. 주어와 서술어의 의미 관계를 맞추세요. 회사/기관 주어는 행동·발표·공개를, 제품/서비스/플랫폼/기술 주어는 기능·역할·적용 범위를, 기사/보도/자료 주어는 소개·설명·언급처럼 전달 행위를 서술하세요.

문장별 역할:
- 1문장: 핵심 사건·상태·평가
- 2문장: 연결된 제품·서비스·플랫폼·기술·업무·고객·산업 영역
- 3문장: 시연·적용 사례·수치·범위·일정·후속 단계·시장 반응·불확실성 중 가장 구체적인 사실

event_type별 fact 선택 의도:
- launch: 1문장 출시/공개/선보임, 2문장 제품·플랫폼 기능, 3문장 시연·적용 사례·수치·후속 단계
- technology_update/general_update: 1문장 기술 확장/변화, 2문장 연결 업무·산업·운영 구조, 3문장 적용 방향·현장 투입·시연·불확실성
- contract: 1문장 수주/계약/사업자 선정, 2문장 고객/시스템/업무 영역, 3문장 규모/기간/후속 단계
- earnings: 1문장 실적 변화, 2문장 연결 사업/원인, 3문장 수치/기간
- stock_market: 1문장 주가/시장 반응, 2문장 기사에서 제시한 배경, 3문장 등락률/거래량/전망
- risk: 1문장 리스크 발생, 2문장 연결 시스템/고객/업무, 3문장 피해 범위/대응/불확실성

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
    }}
  ],
  "main_event": "기사에 명시된 핵심 사건",
  "confidence": 0.0,
  "reason": "fact_id 기반 요약 근거 또는 invalid 사유"
}}"""


class PeerNewsSummaryAgent:
    """클러스터 단위로 피어사 뉴스의 사실 요약을 생성한다."""

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

        source_article_ids = _article_ids(articles)
        requested_cluster_ids = list(cluster_article_ids or [])
        coverage_warning = ""
        if not requested_cluster_ids:
            coverage_warning = "cluster_article_ids 없음"
        cluster_article_count = len(requested_cluster_ids or source_article_ids)
        analyzed_article_count = len(source_article_ids)
        coverage = _coverage_info(
            cluster_article_count=cluster_article_count,
            analyzed_article_count=analyzed_article_count,
            warning=coverage_warning,
        )

        target_companies = _candidate_peer_companies(articles)
        if not target_companies:
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason="self 회사를 제외한 피어사 후보를 찾지 못했습니다.",
                source_article_ids=source_article_ids,
                cluster_article_ids=requested_cluster_ids or source_article_ids,
                coverage=coverage,
            )

        article_fact_notes, fact_extraction_warnings, fact_extraction_failed = (
            _extract_article_fact_notes_batch(
                cluster_id=cluster_id,
                articles=articles,
                target_companies=target_companies,
                representative_id=representative_id,
            )
        )
        merged_facts = _merge_article_facts(article_fact_notes)
        cluster_fact_intelligence = _build_cluster_fact_intelligence(merged_facts)
        cluster_event_type = _classify_cluster_event_type(cluster_fact_intelligence, articles)
        extracted_facts = _build_extracted_facts(
            cluster_id=cluster_id,
            article_fact_notes=article_fact_notes,
            articles=articles,
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
                cluster_article_ids=requested_cluster_ids or source_article_ids,
                coverage=coverage,
            )

        summary = {
            "cluster_id": cluster_id,
            "representative_id": representative_id,
            "source_article_ids": source_article_ids,
            "cluster_article_ids": requested_cluster_ids or source_article_ids,
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


def _build_fetch_ids(
    representative_id: int,
    cluster_article_ids: list[int] | None,
    max_cluster_articles: int | None,
) -> list[int]:
    del max_cluster_articles
    if not cluster_article_ids:
        return [representative_id]

    others = [article_id for article_id in cluster_article_ids if article_id != representative_id]
    return _dedupe_ints([representative_id, *others])


def _format_articles(
    articles: list[dict[str, Any]],
    target_companies: list[str],
    representative_id: int,
    compact: bool = False,
) -> str:
    lines = [
        f"cluster_target_peer_companies: {json.dumps(target_companies, ensure_ascii=False)}",
        "cluster_target_peer_aliases: "
        f"{json.dumps(_target_company_aliases(target_companies), ensure_ascii=False)}",
    ]

    for index, article in enumerate(articles, start=1):
        article_id = _article_numeric_id(article)
        metadata = _metadata(article)
        content = _normalize_content(article.get("content") or "")
        if compact:
            content = content[:1200]
        lines.append(
            "\n".join(
                [
                    f"[{index}] article_id: {article_id}",
                    f"title: {article.get('title') or ''}",
                    f"source_name: {article.get('source_name') or ''}",
                    f"publisher: {article.get('publisher') or ''}",
                    f"published_at: {article.get('published_at') or ''}",
                    f"company: {json.dumps(_company_list(article), ensure_ascii=False)}",
                    "matched_companies: "
                    f"{json.dumps(_matched_companies(article), ensure_ascii=False)}",
                    f"metadata: {json.dumps(_summary_metadata(metadata), ensure_ascii=False)}",
                    f"content: {content}",
                ]
            )
        )

    return "\n\n".join(lines)


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
                agent="PeerNewsSummaryAgent",
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
                        agent="PeerNewsSummaryAgent",
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
            if sentence
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
            unique_facts.append(
                {
                    "fact": fact_text,
                    "source_article_ids": [article_id] if article_id else [],
                    "evidence_count": 1,
                    "evidence_texts": [str(fact.get("evidence_text") or fact_text)],
                    "fact_type": fact.get("fact_type") or "general_fact",
                    "summary_role": fact.get("summary_role") or "main_event",
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
        if _is_duplicate_extracted_fact(facts, article_id, text, evidence):
            return
        counters[article_id] = counters.get(article_id, 0) + 1
        inferred_type = _normalize_fact_type(
            fact_type=fact_type,
            text=f"{text} {evidence}",
            activity_type=activity_type or cluster_event_type,
        )
        normalized_role = _normalize_summary_role(summary_role, fact_type=inferred_type)
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

    if len(facts) < 3:
        _add_article_fallback_facts(
            facts=facts,
            counters=counters,
            cluster_id=cluster_id,
            articles=articles,
            cluster_event_type=cluster_event_type,
        )

    return facts


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
    del text, activity_type
    return _normalize_fact_type_value(fact_type)


def _normalize_fact_type_value(value: Any) -> str:
    fact_type = str(value or "").strip()
    return fact_type if fact_type in _FACT_TYPES else "general_fact"


def _normalize_summary_role(value: Any, *, fact_type: str) -> str:
    role = str(value or "").strip()
    if role in _SUMMARY_ROLES:
        return role
    return _default_summary_role(fact_type)


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
        selected = sorted(candidates, key=_fact_selection_score, reverse=True)[0]
        fact_id = str(selected.get("fact_id"))
        used.add(fact_id)
        return [fact_id]

    preferences = _line_summary_role_preferences(cluster_event_type)
    return {
        "1": choose(1, preferences[0]),
        "2": choose(2, preferences[1]),
        "3": choose(3, preferences[2]),
    }


def _line_summary_role_preferences(
    event_type: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    event_type = _normalize_event_type(event_type)
    if event_type == "launch":
        return (
            ("main_event",),
            ("product_definition", "service_function"),
            ("application_case", "numeric_effect", "uncertainty_detail"),
        )
    if event_type in {"technology_update", "general_update", "unknown"}:
        return (
            ("main_event",),
            ("service_function", "product_definition", "application_case"),
            ("uncertainty_detail", "application_case", "numeric_effect"),
        )
    if event_type == "contract":
        return (
            ("main_event",),
            ("service_function", "product_definition", "application_case"),
            ("numeric_effect", "uncertainty_detail"),
        )
    if event_type == "earnings":
        return (
            ("main_event", "numeric_effect"),
            ("service_function", "product_definition"),
            ("numeric_effect", "uncertainty_detail"),
        )
    if event_type == "stock_market":
        return (
            ("market_reaction", "main_event"),
            ("service_function", "product_definition"),
            ("numeric_effect", "market_reaction", "uncertainty_detail"),
        )
    if event_type == "risk":
        return (
            ("risk_detail", "main_event"),
            ("service_function", "product_definition", "application_case"),
            ("risk_detail", "uncertainty_detail", "numeric_effect"),
        )
    return (
        ("main_event",),
        ("product_definition", "service_function", "application_case"),
        ("numeric_effect", "application_case", "uncertainty_detail"),
    )


def _fact_selection_score(fact: dict[str, Any]) -> int:
    score = 0
    score += (
        3 if fact.get("confidence") == "high" else 2 if fact.get("confidence") == "medium" else 1
    )
    if fact.get("entities"):
        score += 2
    if fact.get("numbers") or fact.get("dates"):
        score += 2
    if fact.get("event_verbs"):
        score += 1
    score += min(len(str(fact.get("normalized_fact") or "")) // 30, 3)
    return score


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
                    agent="PeerNewsSummaryAgent",
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
                _normalize_string_list(data.get("fact_summary"))[:3], start=1
            )
        ]
    line_items = _ensure_three_fact_summary_lines(line_items, extracted_facts)
    fact_summary = [str(item.get("text") or "").strip() for item in line_items]
    result = {
        "is_valid_summary": bool(data.get("is_valid_summary", True)) and len(fact_summary) == 3,
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
        if 1 <= index <= 3 and text:
            lines.append({"line_index": index, "text": text, "fact_ids": fact_ids})
    by_index: dict[int, dict[str, Any]] = {}
    for item in lines:
        by_index[_safe_int(item.get("line_index"))] = item
    return [by_index[index] for index in (1, 2, 3) if index in by_index]


def _ensure_three_fact_summary_lines(
    line_items: list[dict[str, Any]],
    extracted_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    result: dict[int, dict[str, Any]] = {
        _safe_int(item.get("line_index")): item
        for item in line_items
        if _safe_int(item.get("line_index")) in {1, 2, 3}
    }
    unused_facts = [fact for fact in extracted_facts if str(fact.get("fact_id"))]
    for index in (1, 2, 3):
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
    return [result[index] for index in (1, 2, 3)]


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
    for index in (1, 2, 3):
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
        normalize_korean_spacing(line)
        for line in _normalize_string_list(result.get("fact_summary"))[:3]
    ]
    result["fact_summary"] = lines
    line_items = _normalize_summary_line_items(result.get("summary_lines_with_fact_ids"))
    if not line_items:
        line_items = [
            {"line_index": index, "text": line, "fact_ids": []}
            for index, line in enumerate(lines, start=1)
        ]
    line_items = _ensure_three_fact_summary_lines(line_items, extracted_facts)
    for item in line_items:
        index = _safe_int(item.get("line_index"))
        if 1 <= index <= len(lines):
            item["text"] = lines[index - 1]
    result["summary_lines_with_fact_ids"] = line_items
    result["fact_basis"] = _fact_basis_from_summary_line_fact_ids(line_items, extracted_facts)

    warnings: list[str] = []
    actions: list[str] = []
    if len(lines) != 3 or any(not line for line in lines):
        warnings.append("summary_lines가 정확히 3개가 아님")
    basis_indexes = {
        _safe_int(item.get("summary_line_index", item.get("summary_sentence_index")))
        for item in result.get("fact_basis", [])
    }
    missing_indexes = [index for index in (1, 2, 3) if index not in basis_indexes]
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
        related_evidence = " ".join(str(fact.get("evidence_text") or "") for fact in related_facts)
        missing_numbers = [
            number
            for number in _number_tokens(line)
            if not _number_token_covered(number, _number_tokens(related_evidence))
        ]
        if missing_numbers:
            warnings.append(f"{index}번 문장 수치 근거 부족: {', '.join(missing_numbers)}")
    cleaned_lines = [normalize_korean_spacing(line) for line in lines]
    if cleaned_lines != lines:
        result["fact_summary"] = cleaned_lines
        actions.append("normalize_korean_spacing")
        line_items = _sync_summary_line_item_texts(line_items, cleaned_lines)

    reduced_lines, repetition_actions = normalize_subject_predicate_consistency(
        lines=_normalize_string_list(result.get("fact_summary"))[:3],
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
        _normalize_string_list(result.get("fact_summary"))[:3],
        main_company,
    )
    if company_start_count >= 2:
        warnings.append(f"company_name_start_count={company_start_count}")
        actions.append("company_name_repetition_detected")
    if company_start_count == 3:
        warnings.append("summary_lines 3문장이 모두 company_name으로 시작함")

    bad_korean = [line for line in result["fact_summary"] if _has_bad_korean_join(line)]
    if bad_korean:
        warnings.append("한국어 조사/띄어쓰기 오류가 남아 있음")
    if (
        main_company
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
    nominal = _nominalize_korean_predicate(rest)
    if not nominal:
        return "기사에서는 관련 사실이 확인됐다."
    return normalize_korean_spacing(f"기사에서는 {nominal} 사실이 확인됐다.")


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
        for index in (1, 2, 3)
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
    chunks = re.split(r"(?<=[.!?。！？])\s+|(?<=[다요죠임음])\.\s*|\n+", str(text or ""))
    sentences: list[str] = []
    for chunk in chunks:
        sentence = re.sub(r"\s+", " ", chunk).strip()
        if len(sentence) < 8:
            continue
        sentences.append(sentence[:500])
        if len(sentences) >= limit:
            break
    return sentences


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
    candidates: list[str] = []
    text = " ".join(
        f"{article.get('title') or ''} {article.get('content') or ''}" for article in articles
    )
    compact_text = _compact(text)

    for article in articles:
        candidates.extend(_company_list(article))
        candidates.extend(_matched_companies(article))

    for company_id, aliases in _PEER_ALIASES.items():
        if any(_compact(alias) in compact_text for alias in aliases):
            candidates.append(company_id)

    return [
        company_id
        for company_id in _dedupe_keep_order(candidates)
        if company_id in _PEER_ALIASES and company_tier(company_id) != "self"
    ]


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
    return {company_id: _PEER_ALIASES.get(company_id, [company_id]) for company_id in company_ids}


def _summary_mentions_company(summary: dict[str, Any], company_id: str) -> bool:
    aliases = _PEER_ALIASES.get(company_id, [company_id])
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
) -> dict[str, Any]:
    ratio = analyzed_article_count / cluster_article_count if cluster_article_count else 0.0
    coverage_warning = warning
    if cluster_article_count and analyzed_article_count < cluster_article_count:
        coverage_warning = (
            f"{coverage_warning}; " if coverage_warning else ""
        ) + "일부 cluster_article_ids 기사 조회 실패"
    return {
        "cluster_article_count": cluster_article_count,
        "analyzed_article_count": analyzed_article_count,
        "coverage_ratio": round(ratio, 4),
        "coverage_warning": coverage_warning,
    }


def _as_int_list(value: Any) -> list[int]:
    return [_safe_int(item) for item in _as_list(value) if _safe_int(item) > 0]


def _normalize_event_type(value: Any) -> str:
    event_type = str(value or "").strip()
    return event_type if event_type in _EVENT_TYPES else "unknown"


def _fact_key(value: str) -> str:
    compacted = _compact(value)
    return compacted[:120]


def _has_unique_fact_importance(text: str) -> bool:
    return bool(re.search(r"\d", text))


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
