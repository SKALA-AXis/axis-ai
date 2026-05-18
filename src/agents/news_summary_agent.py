"""피어사 뉴스 사실 요약 에이전트.

클러스터에 묶인 기사들을 바탕으로 피어사 관련 본문 내용을 요약한다.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from src.config.companies import COMPANY_ALIASES
from src.config.company_tiers import company_tier
from src.config.global_companies import GLOBAL_COMPANY_ALIASES
from src.db.article_store import get_articles_by_ids

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "summary-v4.0"
_DEFAULT_MAX_CLUSTER_ARTICLES = 5
_FACT_EXTRACTION_BATCH_SIZE = 10
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
            max_completion_tokens=900,
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
   contract|partnership|launch|earnings|stock_market|analyst_report|investment|hiring|organization|risk|regulation|technology_update|general_update|unknown

기사 클러스터:
{articles_text}

다음 JSON 형식으로만 응답하세요.
{{
  "article_facts": [
    {{
      "article_id": 0,
      "core_facts": [
        {{
          "fact": "기사에서 확인한 핵심 사실",
          "evidence_text": "원문 근거 요약",
          "activity_type": "contract|partnership|launch|earnings|stock_market|analyst_report|investment|hiring|organization|risk|regulation|technology_update|general_update|unknown",
          "numbers_and_dates": [],
          "customers_or_industries": [],
          "products_or_services": []
        }}
      ],
      "unique_facts": [
        {{
          "fact": "다른 기사에 없는 보강 사실",
          "evidence_text": "원문 근거 요약",
          "importance_reason": "수치/일정/고객/후속 사업 등 중요한 이유"
        }}
      ],
      "uncertain_facts": [
        {{
          "fact": "예정·전망·계획·가능성 표현",
          "evidence_text": "원문 근거 요약",
          "caution": "확정 사실로 쓰지 말아야 하는 이유"
        }}
      ]
    }}
  ]
}}"""


_PEER_NEWS_SUMMARY_PROMPT = """\
당신은 피어사 뉴스 사실 요약 Agent입니다.

목적:
- 같은 이슈로 묶인 여러 기사에서 공통으로 확인되는 피어사 관련 사실을 요약합니다.
- 요약 대상은 SK AX를 제외한 peer company입니다.
- 시사점, 대응 방향, 전략 해석은 다른 Agent가 담당하므로 여기서는 쓰지 않습니다.

## 입력
- cluster_id: 클러스터 ID
- target_peer_companies: 요약 대상 후보 피어사 company id 목록
- articles: 같은 이슈로 묶인 클러스터 기사 목록
- article_fact_notes: 각 기사를 독립적으로 읽어 추출한 사실 목록
- cluster_fact_intelligence: 기사별 fact를 병합한 클러스터 단위 사실 구조
- cluster_event_type: 클러스터 뉴스 유형

## 요약 설계
응답에는 아래 사고 과정을 드러내지 말고, 결과 JSON만 작성하세요.
1. 기사 번호와 입력 순서는 중요도나 대표성을 뜻하지 않습니다.
   입력된 기사들을 같은 비중으로 읽고 비교하세요.
2. article_fact_notes를 먼저 비교하고, 여러 기사에서 확인되는 공통 사실을 우선하세요.
   필요한 경우 기사 원문으로 표현과 수치를 검증하세요.
3. 기사 간 표현이 다르면 더 구체적인 표현을 선택하고, 충돌하면 보수적으로 쓰거나 제외하세요.
   대표기사에 없더라도 다른 기사에서만 확인되는 중요한 수치, 범위, 후속 단계는 반영하세요.
4. 피어사 동향을 보여주는 실제 움직임을 우선하세요: 수주, 출시, 제휴, 투자, 인수, 채용, 조직개편 등.
5. 사업/기술/서비스 내용은 기사에 나온 산업명, 업무명, 제품명, 기능명,
   데이터 영역을 포함해 구체적으로 쓰세요.
6. 기사에 명시된 고객 규모, 사업 규모, 적용 범위, 후속 본사업, 예정·계획 정보는
   경쟁사 방향성을 보여주는 사실로 반영하세요.
   단, 예상·전망·향후 표현은 확정처럼 단정하지 말고 원문 수위를 유지하세요.
   시장 반응 기사에서는 단순 등락만 쓰지 말고 거래량, 시가총액, 밸류에이션,
   업종 대비 흐름처럼 기사에 나온 회사 상황 근거를 함께 요약하세요.
   원문에 원인이 없으면 하락·상승 원인을 만들지 마세요.
7. fact_summary는 정확히 3문장으로 작성하고, 뉴스 유형에 맞게 역할을 나누세요.

1번째 문장:
이 클러스터에서 새로 확인되는 핵심 사건·상태·평가를 씁니다.
수주/협약/출시 뉴스라면 피어사가 무엇을 했는지 쓰고,
실적/주가/증권 리포트 뉴스라면 어떤 변화나 평가가 있었는지 쓰며,
리스크/사고 뉴스라면 어떤 이슈가 발생했는지 씁니다.

2번째 문장:
그 사실이 연결된 사업·기술·서비스·고객·업무·산업 영역을 씁니다.
기사에서 확인되는 제품명, 서비스명, 플랫폼명, 고객군, 산업군, 업무명, 데이터 영역이 있으면 포함하세요.

3번째 문장:
기사에 나온 수치·범위·일정·후속 단계·시장 반응·불확실성 중 가장 중요한 사실을 씁니다.
수치가 있으면 단위와 기간을 함께 쓰고, 예정·전망·가능성은 확정처럼 쓰지 마세요.
원문에 원인이 없으면 원인을 만들지 마세요.

8. 3문장 안에 모두 담을 수 없으면 사업 행동, 기술·서비스 내용,
   고객/시장 규모, 후속 사업 단계 순으로 선택하세요.
   참석자, 사진 설명, 일반 배경은 제외하세요.

## 작성 원칙
1. 기사에 명시된 사실만 요약하세요.
2. 본문에 없는 수치, 제품명, 회사명은 만들지 마세요.
3. 피어사를 주어로 작성하세요.
4. headline, one_line_summary, fact_summary 중 최소 1곳에는 main_company의 회사명 또는 alias가
   실제 문장으로 들어가야 합니다.
5. 회사의 일반적 역할·정체성·배경만 말하는 문장은 fact_summary에 쓰지 마세요.
   반드시 기사에서 새로 확인되는 구체 사실을 포함하세요.
6. 여러 기사 내용이 충돌하면 공통으로 확인되는 사실만 쓰세요.
7. 기사 내용이 피어사 핵심 내용이 아니거나, 그룹사/계열사/협력사만 핵심 주체이면
   is_valid_summary=false로 표시하세요.
8. "목표로 한다", "강화할 수 있다" 같은 목적·평가식 표현보다 기사에 근거한 사실 표현을 우선하세요.
9. fact_summary에는 "목표로 한다"를 쓰지 마세요. 필요한 경우 기사에 근거해
   "설계한다", "수립한다", "마련한다", "추진한다", "선행 단계다" 같은 사실형 서술로 바꾸세요.
10. 대표기사에 없는 unique_facts도 수치, 일정, 고객, 후속 사업, 시장 반응처럼 중요하면 반영하세요.
11. article_fact_notes와 cluster_fact_intelligence를 원문보다 우선 사용하세요.
12. summary 문장마다 fact_basis로 근거를 연결할 수 있어야 합니다.
13. fact_basis가 없는 문장은 repair 대상입니다.
14. 일반적인 회사 소개나 기존 포지셔닝은 쓰지 마세요.

## 기사별 팩트 노트와 기사 원문
{articles_text}

다음 JSON 형식으로만 응답하세요.
{{
  "is_valid_summary": true,
  "main_company": "company_id",
  "mentioned_peer_companies": ["company_id"],
  "cluster_event_type": "contract|partnership|launch|earnings|stock_market|analyst_report|investment|hiring|organization|risk|regulation|technology_update|general_update|unknown",
  "headline": "피어사 사실 중심 한 문장",
  "one_line_summary": "기사 클러스터의 핵심 사실 1문장",
  "fact_summary": ["사실 1", "사실 2", "사실 3"],
  "main_event": "기사에 명시된 핵심 사건",
  "fact_basis": [
    {{
      "summary_sentence_index": 1,
      "fact": "요약에 반영된 사실",
      "source_article_ids": [],
      "evidence_count": 0,
      "evidence_type": "common_fact|unique_fact|uncertain_fact",
      "evidence_texts": []
    }}
  ],
  "confidence": 0.0,
  "reason": "요약 근거 또는 invalid 사유"
}}"""

_FACT_SUMMARY_REPAIR_PROMPT = """\
당신은 피어사 뉴스 사실 요약 검수 Agent입니다.

아래 current_summary_json의 fact_summary에 너무 일반적인 표현이나 목적/평가식 표현이 있으면,
기사 클러스터 근거만 사용해 더 구체적인 사실 요약으로 고치세요.

검수 기준:
1. 하드코딩하거나 새로운 사실을 만들지 마세요.
2. 기사에서 확인되는 대상 업무명, 데이터 영역, 제품/서비스명, 적용 산업 중
   최소 1개가 사업/기술 설명 문장의 핵심 명사구에 들어가야 합니다.
3. 대상 업무·데이터·산업·제품명이 빠진 일반 표현을 핵심 설명으로 쓰지 마세요.
4. fact_summary 3문장은 각각 중복되지 않게 역할을 나누세요.
   - 이 클러스터에서 새로 확인되는 핵심 사건·상태·평가
   - 연결된 사업·기술·서비스·고객·업무·산업 영역
   - 수치·범위·일정·후속 단계·시장 반응·불확실성 중 가장 중요한 사실
5. 회사의 일반적 정체성, 기존 포지셔닝, 배경 설명만 담긴 문장은 반드시 기사 속
   구체 사실이 들어간 문장으로 바꾸세요.
6. 시사점, 전망, 대응 방향은 쓰지 말고 기사 속 사실만 쓰세요.
7. 기사에 명시된 예정·계획·후속 본사업 정보는 원문 수위를 유지해 쓸 수 있습니다.
8. "목표로 한다", "기대된다", "가능성이 있다"처럼 목적/평가식 표현보다
   "방안을 마련한다", "로드맵을 수립한다", "선행 단계다"처럼 기사에 근거한 사실 표현을 쓰세요.
9. fact_summary에 "목표로 한다"가 남아 있으면 검수 실패로 보고 반드시 사실형 서술로 고치세요.
10. fact_summary 각 문장에 fact_basis를 연결하세요.
11. fact_basis 없이 쓴 문장은 제거하거나 근거가 있는 사실로 바꾸세요.

domain_terms:
{domain_terms_json}

current_summary_json:
{summary_json}

기사 클러스터:
{articles_text}

다음 JSON 형식으로만 응답하세요.
{{
  "is_valid_summary": true,
  "main_company": "company_id",
  "mentioned_peer_companies": ["company_id"],
  "cluster_event_type": "contract|partnership|launch|earnings|stock_market|analyst_report|investment|hiring|organization|risk|regulation|technology_update|general_update|unknown",
  "headline": "피어사 사실 중심 한 문장",
  "one_line_summary": "기사 클러스터의 핵심 사실 1문장",
  "fact_summary": ["사실 1", "사실 2", "사실 3"],
  "main_event": "기사에 명시된 핵심 사건",
  "fact_basis": [
    {{
      "summary_sentence_index": 1,
      "fact": "요약에 반영된 사실",
      "source_article_ids": [],
      "evidence_count": 0,
      "evidence_type": "common_fact|unique_fact|uncertain_fact",
      "evidence_texts": []
    }}
  ],
  "confidence": 0.0,
  "reason": "요약 근거 또는 invalid 사유"
}}"""


class PeerNewsSummaryAgent:
    """클러스터 단위로 피어사 뉴스의 사실 요약을 생성한다."""

    def summarize(
        self,
        cluster_id: int,
        representative_id: int,
        cluster_article_ids: list[int] | None = None,
        max_cluster_articles: int = _DEFAULT_MAX_CLUSTER_ARTICLES,
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
                reason="SK AX를 제외한 피어사 후보를 찾지 못했습니다.",
                source_article_ids=source_article_ids,
                cluster_article_ids=requested_cluster_ids or source_article_ids,
                coverage=coverage,
            )

        articles_text = _format_articles(
            articles=articles,
            target_companies=target_companies,
            representative_id=representative_id,
        )
        article_fact_notes = _extract_article_fact_notes_batch(
            articles=articles,
            target_companies=target_companies,
            representative_id=representative_id,
        )
        merged_facts = _merge_article_facts(article_fact_notes)
        cluster_fact_intelligence = _build_cluster_fact_intelligence(merged_facts)
        cluster_event_type = _classify_cluster_event_type(cluster_fact_intelligence, articles)
        summary_context = _format_summary_context(
            article_fact_notes=article_fact_notes,
            cluster_fact_intelligence=cluster_fact_intelligence,
            cluster_event_type=cluster_event_type,
            articles_text=articles_text,
        )
        prompt = _PEER_NEWS_SUMMARY_PROMPT.replace("{articles_text}", summary_context)

        try:
            from src.observability import tracing_config

            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="PeerNewsSummaryAgent",
                    phase="summarize",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            result = _normalize_summary_result(_parse_json(content), target_companies)
            result["cluster_event_type"] = _normalize_event_type(
                result.get("cluster_event_type") or cluster_event_type
            )
            result["fact_basis"] = _build_fact_basis(
                result=result,
                cluster_fact_intelligence=cluster_fact_intelligence,
            )
            result = _validate_summary_grounding(
                result=result,
                source_article_ids=source_article_ids,
                cluster_fact_intelligence=cluster_fact_intelligence,
                main_company=result.get("main_company", ""),
            )
            result = _repair_fact_summary_if_needed(
                result=result,
                articles=articles,
                articles_text=summary_context,
                target_companies=target_companies,
                cluster_fact_intelligence=cluster_fact_intelligence,
            )
            result["fact_basis"] = _build_fact_basis(
                result=result,
                cluster_fact_intelligence=cluster_fact_intelligence,
            )
            result = _validate_summary_grounding(
                result=result,
                source_article_ids=source_article_ids,
                cluster_fact_intelligence=cluster_fact_intelligence,
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
            "coverage": coverage,
            "model": _LLM_MODEL,
            **result,
        }
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
    max_cluster_articles: int,
) -> list[int]:
    if not cluster_article_ids:
        return [representative_id]

    others = [article_id for article_id in cluster_article_ids if article_id != representative_id]
    return _dedupe_ints([representative_id, *others])


def _format_articles(
    articles: list[dict[str, Any]],
    target_companies: list[str],
    representative_id: int,
) -> str:
    lines = [
        f"cluster_target_peer_companies: {json.dumps(target_companies, ensure_ascii=False)}",
        "cluster_target_peer_aliases: "
        f"{json.dumps(_target_company_aliases(target_companies), ensure_ascii=False)}",
    ]

    for index, article in enumerate(articles, start=1):
        article_id = _article_numeric_id(article)
        metadata = _metadata(article)
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
                    f"content: {_normalize_content(article.get('content') or '')}",
                ]
            )
        )

    return "\n\n".join(lines)


def _extract_article_fact_notes_batch(
    *,
    articles: list[dict[str, Any]],
    target_companies: list[str],
    representative_id: int,
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for batch in _chunked(articles, _FACT_EXTRACTION_BATCH_SIZE):
        articles_text = _format_articles(
            articles=batch,
            target_companies=target_companies,
            representative_id=representative_id,
        )
        parsed = _extract_article_fact_notes(articles_text)
        values = parsed.get("article_facts", []) if isinstance(parsed, dict) else []
        for item in values:
            if isinstance(item, dict):
                notes.append(_normalize_article_fact_note(item))
    return notes


def _extract_article_fact_notes(articles_text: str) -> dict[str, Any]:
    from src.observability import tracing_config

    prompt = _ARTICLE_FACT_EXTRACTION_PROMPT.replace("{articles_text}", articles_text)
    try:
        response = _get_llm().invoke(
            prompt,
            config=tracing_config(
                agent="PeerNewsSummaryAgent",
                phase="extract_facts",
                prompt_version=_PROMPT_VERSION,
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        return _parse_json(content)
    except Exception as exc:
        log.warning("기사별 팩트 추출 실패 | error=%s", exc)
        return {}


def _format_summary_context(
    *,
    article_fact_notes: list[dict[str, Any]],
    cluster_fact_intelligence: dict[str, Any],
    cluster_event_type: str,
    articles_text: str,
) -> str:
    return "\n\n".join(
        [
            "## article_fact_notes",
            json.dumps({"article_facts": article_fact_notes}, ensure_ascii=False, indent=2),
            "## cluster_event_type",
            cluster_event_type,
            "## cluster_fact_intelligence",
            json.dumps(cluster_fact_intelligence, ensure_ascii=False, indent=2),
            "## original_articles",
            articles_text,
        ]
    )


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
        return {
            "fact": fact,
            "evidence_text": str(value.get("evidence_text") or fact).strip(),
            "activity_type": activity_type,
            "numbers_and_dates": _normalize_string_list(value.get("numbers_and_dates")),
            "customers_or_industries": _normalize_string_list(value.get("customers_or_industries")),
            "products_or_services": _normalize_string_list(value.get("products_or_services")),
        }
    fact = str(value or "").strip()
    return {
        "fact": fact,
        "evidence_text": fact,
        "activity_type": default_type,
        "numbers_and_dates": [],
        "customers_or_industries": [],
        "products_or_services": [],
    }


def _normalize_unique_fact(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        fact = str(value.get("fact") or "").strip()
        return {
            "fact": fact,
            "evidence_text": str(value.get("evidence_text") or fact).strip(),
            "importance_reason": str(value.get("importance_reason") or "").strip(),
        }
    fact = str(value or "").strip()
    return {"fact": fact, "evidence_text": fact, "importance_reason": ""}


def _normalize_uncertain_fact(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        fact = str(value.get("fact") or "").strip()
        return {
            "fact": fact,
            "evidence_text": str(value.get("evidence_text") or fact).strip(),
            "caution": str(value.get("caution") or "전망/예정/가능성 표현").strip(),
        }
    fact = str(value or "").strip()
    return {"fact": fact, "evidence_text": fact, "caution": "전망/예정/가능성 표현"}


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
    lowered = text.lower()
    keyword_map = [
        ("contract", ("수주", "계약", "우선협상대상자", "사업자 선정", "공급계약")),
        ("partnership", ("mou", "협약", "파트너십", "제휴")),
        ("launch", ("출시", "공개", "선보", "론칭")),
        ("earnings", ("매출", "영업이익", "실적", "순이익")),
        ("stock_market", ("주가", "시가총액", "거래량", "시장 반응", "등락")),
        ("analyst_report", ("목표주가", "증권사", "리포트", "투자의견", "전망")),
        ("investment", ("투자", "지분", "인수", "펀드")),
        ("hiring", ("채용", "인력", "채용공고")),
        ("organization", ("조직 개편", "임원 인사", "대표이사")),
        ("risk", ("장애", "보안 사고", "소송", "리스크", "침해", "해킹")),
        ("regulation", ("규제", "정책", "정부", "인증")),
        ("technology_update", ("기술", "업데이트", "특허", "모델", "플랫폼")),
    ]
    for event_type, keywords in keyword_map:
        if any(keyword.lower() in lowered for keyword in keywords):
            return event_type
    return "general_update" if text.strip() else "unknown"


def _build_fact_basis(
    *,
    result: dict[str, Any],
    cluster_fact_intelligence: dict[str, Any],
) -> list[dict[str, Any]]:
    existing = result.get("fact_basis")
    if isinstance(existing, list) and len(existing) >= 3:
        normalized = [
            _normalize_fact_basis_item(item) for item in existing if isinstance(item, dict)
        ]
        if len([item for item in normalized if item.get("source_article_ids")]) >= 3:
            return normalized

    facts = _basis_fact_pool(cluster_fact_intelligence)
    basis: list[dict[str, Any]] = []
    for index, sentence in enumerate(
        _normalize_string_list(result.get("fact_summary"))[:3], start=1
    ):
        best = _best_fact_for_sentence(sentence, facts)
        if best is None:
            basis.append(
                {
                    "summary_sentence_index": index,
                    "fact": sentence,
                    "source_article_ids": [],
                    "evidence_count": 0,
                    "evidence_type": "uncertain_fact",
                    "evidence_texts": [],
                }
            )
            continue
        basis.append(
            {
                "summary_sentence_index": index,
                "fact": best.get("fact") or sentence,
                "source_article_ids": best.get("source_article_ids", []),
                "evidence_count": int(
                    best.get("evidence_count") or len(best.get("source_article_ids", []))
                ),
                "evidence_type": best.get("evidence_type", "common_fact"),
                "evidence_texts": best.get("evidence_texts", [])[:3],
            }
        )
    return basis


def _basis_fact_pool(cluster_fact_intelligence: dict[str, Any]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for key, evidence_type in (
        ("common_facts", "common_fact"),
        ("unique_facts", "unique_fact"),
        ("uncertain_facts", "uncertain_fact"),
    ):
        for fact in cluster_fact_intelligence.get(key, []) or []:
            if isinstance(fact, dict):
                item = dict(fact)
                item["evidence_type"] = evidence_type
                pool.append(item)
    return pool


def _best_fact_for_sentence(sentence: str, facts: list[dict[str, Any]]) -> dict[str, Any] | None:
    sentence_tokens = _token_set(sentence)
    best_score = 0
    best: dict[str, Any] | None = None
    for fact in facts:
        fact_text = str(fact.get("fact") or "")
        score = len(sentence_tokens & _token_set(fact_text))
        if score > best_score:
            best_score = score
            best = fact
    if best is not None and best_score > 0:
        return best
    return facts[0] if facts else None


def _validate_summary_grounding(
    *,
    result: dict[str, Any],
    source_article_ids: list[int],
    cluster_fact_intelligence: dict[str, Any],
    main_company: str,
) -> dict[str, Any]:
    facts = _normalize_string_list(result.get("fact_summary"))
    if len(facts) != 3:
        result["is_valid_summary"] = False
        result["reason"] = _append_reason(result.get("reason"), "fact_summary가 3문장이 아님")
    result["fact_summary"] = facts[:3]
    result["fact_basis"] = _build_fact_basis(
        result=result,
        cluster_fact_intelligence=cluster_fact_intelligence,
    )
    if len(result["fact_basis"]) < len(result["fact_summary"]) or any(
        not item.get("source_article_ids") for item in result["fact_basis"]
    ):
        result["reason"] = _append_reason(result.get("reason"), "fact_basis가 없는 요약 문장 존재")
    if not source_article_ids:
        result["is_valid_summary"] = False
        result["reason"] = _append_reason(result.get("reason"), "source_article_ids가 비어 있음")
    if (
        main_company
        and result.get("is_valid_summary")
        and not _summary_mentions_company(result, main_company)
    ):
        result["is_valid_summary"] = False
        result["reason"] = _append_reason(
            result.get("reason"), "요약 문장에 main_company alias가 없음"
        )
    if _contains_disallowed_interpretation(result):
        result["reason"] = _append_reason(
            result.get("reason"), "시사점/대응방향 표현을 제거해야 함"
        )
    return result


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


def _normalize_summary_result(
    data: dict[str, Any],
    target_companies: list[str],
) -> dict[str, Any]:
    mentioned = [
        company_id
        for company_id in _normalize_string_list(data.get("mentioned_peer_companies"))
        if company_id in target_companies
    ]
    main_company = str(data.get("main_company") or "").strip()
    if main_company not in target_companies:
        main_company = mentioned[0] if mentioned else target_companies[0]

    facts = _normalize_string_list(data.get("fact_summary"))[:3]
    result = {
        "is_valid_summary": bool(data.get("is_valid_summary", True)) and bool(facts),
        "main_company": main_company,
        "mentioned_peer_companies": mentioned or [main_company],
        "cluster_event_type": _normalize_event_type(data.get("cluster_event_type")),
        "headline": str(data.get("headline") or "").strip(),
        "one_line_summary": str(data.get("one_line_summary") or "").strip(),
        "fact_summary": facts,
        "main_event": str(data.get("main_event") or "").strip(),
        "fact_basis": [
            _normalize_fact_basis_item(item)
            for item in _as_list(data.get("fact_basis"))
            if isinstance(item, dict)
        ],
        "confidence": _clamp_float(data.get("confidence"), default=0.0),
        "reason": str(data.get("reason") or "").strip(),
    }
    if result["is_valid_summary"] and not _summary_mentions_company(result, main_company):
        result["is_valid_summary"] = False
        result["reason"] = (
            "요약 문장에 main_company alias가 없어 피어사 핵심 동향 요약으로 인정하지 않음"
        )
    return result


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


def _repair_fact_summary_if_needed(
    *,
    result: dict[str, Any],
    articles: list[dict[str, Any]],
    articles_text: str,
    target_companies: list[str],
    cluster_fact_intelligence: dict[str, Any],
) -> dict[str, Any]:
    prompt = (
        _FACT_SUMMARY_REPAIR_PROMPT.replace(
            "{domain_terms_json}",
            json.dumps(_domain_terms(articles), ensure_ascii=False),
        )
        .replace(
            "{summary_json}",
            json.dumps(
                {
                    **result,
                    "cluster_fact_intelligence": cluster_fact_intelligence,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        .replace("{articles_text}", articles_text)
    )

    try:
        from src.observability import tracing_config

        response = _get_llm().invoke(
            prompt,
            config=tracing_config(
                agent="PeerNewsSummaryAgent",
                phase="repair",
                prompt_version=_PROMPT_VERSION,
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        repaired = _normalize_summary_result(_parse_json(content), target_companies)
        repaired["fact_basis"] = _build_fact_basis(
            result=repaired,
            cluster_fact_intelligence=cluster_fact_intelligence,
        )
        if repaired.get("is_valid_summary"):
            log.info("피어사 뉴스 요약 재검수 반영 | main_company=%s", repaired["main_company"])
            return repaired
    except Exception as exc:
        log.warning("피어사 뉴스 요약 재검수 실패 | error=%s", exc)

    return result


def _domain_terms(articles: list[dict[str, Any]], limit: int = 30) -> list[str]:
    text = " ".join(
        f"{article.get('title') or ''} {_normalize_content(article.get('content') or '')}"
        for article in articles
    )
    patterns = (
        r"[가-힣A-Za-z0-9·/+\-]{2,}(?:시스템|플랫폼|서비스|솔루션|사업|프로젝트|컨설팅)",
        r"[가-힣A-Za-z0-9·/+\-]{2,}(?:업무|데이터|설비|요금|수금|검증|탐지|자동화|재설계|운영|구축)",
    )
    terms: list[str] = []
    for pattern in patterns:
        terms.extend(re.findall(pattern, text))
    cleaned = [_clean_domain_term(term) for term in terms]
    return _dedupe_keep_order([term for term in cleaned if len(term) >= 3])[:limit]


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


def _normalize_fact_basis_item(item: dict[str, Any]) -> dict[str, Any]:
    source_ids = [_safe_int(value) for value in _as_list(item.get("source_article_ids"))]
    source_ids = [value for value in source_ids if value > 0]
    return {
        "summary_sentence_index": _safe_int(item.get("summary_sentence_index")),
        "fact": str(item.get("fact") or "").strip(),
        "source_article_ids": _dedupe_ints(source_ids),
        "evidence_count": _safe_int(item.get("evidence_count")) or len(source_ids),
        "evidence_type": str(item.get("evidence_type") or "unique_fact").strip(),
        "evidence_texts": _normalize_string_list(item.get("evidence_texts"))[:5],
    }


def _normalize_event_type(value: Any) -> str:
    event_type = str(value or "").strip()
    return event_type if event_type in _EVENT_TYPES else "unknown"


def _fact_key(value: str) -> str:
    compacted = _compact(value)
    return compacted[:120]


def _has_unique_fact_importance(text: str) -> bool:
    return bool(
        re.search(r"\d", text)
        or any(
            keyword in text
            for keyword in (
                "고객",
                "기관",
                "금융",
                "공공",
                "제조",
                "기간",
                "예정",
                "후속",
                "본사업",
                "플랫폼",
                "서비스",
                "솔루션",
                "시스템",
                "거래량",
                "목표주가",
            )
        )
    )


def _detect_conflict_notes(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    number_map: dict[str, set[str]] = {}
    for fact in facts:
        text = " ".join([str(fact.get("fact") or ""), *fact.get("evidence_texts", [])])
        for value, unit in re.findall(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(억원|조원|%|건|명)", text):
            number_map.setdefault(unit, set()).add(value)
    notes = []
    for unit, values in number_map.items():
        if len(values) >= 2:
            notes.append(
                {
                    "conflict": f"{unit} 단위 수치가 복수로 확인됨",
                    "values": sorted(values),
                    "caution": "요약에서는 출처별 수치 맥락을 보수적으로 유지",
                }
            )
    return notes


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^0-9A-Za-z가-힣]+", str(text or "").lower())
        if len(token) >= 2
    }


def _contains_disallowed_interpretation(result: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(result.get("headline") or ""),
            str(result.get("one_line_summary") or ""),
            " ".join(_normalize_string_list(result.get("fact_summary"))),
        ]
    )
    return any(keyword in text for keyword in ("시사점", "대응", "전략적으로", "해야 한다"))


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
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    return parsed if isinstance(parsed, dict) else {}


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
