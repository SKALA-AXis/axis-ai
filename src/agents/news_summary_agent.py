"""피어사 뉴스 사실 요약 에이전트.

클러스터에 묶인 기사들을 바탕으로 피어사 관련 본문 내용을 요약한다.
"""

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
_DEFAULT_MAX_CLUSTER_ARTICLES = 5
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

규칙:
1. 각 article_id마다 최소 2개, 최대 5개의 핵심 사실을 뽑으세요.
2. 기사에 명시된 사실만 쓰고, 시사점·평가·대응 방향은 쓰지 마세요.
3. 수주, 출시, 제휴, 투자, 적용 업무, 고객 규모, 사업 규모, 후속 본사업,
   예정·계획 정보처럼 경쟁사 동향에 필요한 사실을 우선하세요.
4. 여러 기사에 반복되는 문장이라도 각 기사에서 확인한 사실로 기록하세요.
5. 본문에 없는 수치, 제품명, 회사명은 만들지 마세요.

기사 클러스터:
{articles_text}

다음 JSON 형식으로만 응답하세요.
{{
  "article_facts": [
    {{
      "article_id": 0,
      "core_facts": ["기사에서 확인한 사실"],
      "unique_facts": ["다른 기사 대비 보강되는 사실"],
      "uncertain_facts": ["예정·전망처럼 원문 수위 유지가 필요한 사실"]
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

## 요약 설계
응답에는 아래 사고 과정을 드러내지 말고, 결과 JSON만 작성하세요.
1. 기사 번호와 입력 순서는 중요도나 대표성을 뜻하지 않습니다.
   입력된 기사들을 같은 비중으로 읽고 비교하세요.
2. article_fact_notes를 먼저 비교하고, 여러 기사에서 확인되는 공통 사실을 우선하세요.
   필요한 경우 기사 원문으로 표현과 수치를 검증하세요.
3. 기사 간 표현이 다르면 더 구체적인 표현을 선택하고, 충돌하면 보수적으로 쓰거나 제외하세요.
4. 피어사 동향을 보여주는 실제 움직임을 우선하세요: 수주, 출시, 제휴, 투자, 인수, 채용, 조직개편 등.
5. 사업/기술/서비스 내용은 기사에 나온 산업명, 업무명, 제품명, 기능명,
   데이터 영역을 포함해 구체적으로 쓰세요.
6. 기사에 명시된 고객 규모, 사업 규모, 적용 범위, 후속 본사업, 예정·계획 정보는
   경쟁사 방향성을 보여주는 사실로 반영하세요.
   단, 예상·전망·향후 표현은 확정처럼 단정하지 말고 원문 수위를 유지하세요.
7. fact_summary는 정확히 3문장으로 작성하고, 역할이 겹치지 않게 나누세요.
   - 1번째 문장: 피어사가 무엇을 했는지.
   - 2번째 문장: 어떤 사업/기술/서비스/업무를 다루는지.
   - 3번째 문장: 고객 규모, 적용 범위, 후속 사업 단계, 예정된 추진 범위 중 남은 중요 사실.
8. 3문장 안에 모두 담을 수 없으면 사업 행동, 기술·서비스 내용,
   고객/시장 규모, 후속 사업 단계 순으로 선택하세요.
   참석자, 사진 설명, 일반 배경은 제외하세요.

## 작성 원칙
1. 기사에 명시된 사실만 요약하세요.
2. 본문에 없는 수치, 제품명, 회사명은 만들지 마세요.
3. 피어사를 주어로 작성하세요.
4. 여러 기사 내용이 충돌하면 공통으로 확인되는 사실만 쓰세요.
5. 기사 내용이 피어사 핵심 내용이 아니면 is_valid_summary=false로 표시하세요.
6. "목표로 한다", "강화할 수 있다" 같은 목적·평가식 표현보다 기사에 근거한 사실 표현을 우선하세요.
7. fact_summary에는 "목표로 한다"를 쓰지 마세요. 필요한 경우 기사에 근거해
   "설계한다", "수립한다", "마련한다", "추진한다", "선행 단계다" 같은 사실형 서술로 바꾸세요.

## 기사별 팩트 노트와 기사 원문
{articles_text}

다음 JSON 형식으로만 응답하세요.
{{
  "is_valid_summary": true,
  "main_company": "company_id",
  "mentioned_peer_companies": ["company_id"],
  "headline": "피어사 사실 중심 한 문장",
  "one_line_summary": "기사 클러스터의 핵심 사실 1문장",
  "fact_summary": ["사실 1", "사실 2", "사실 3"],
  "main_event": "기사에 명시된 핵심 사건",
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
   - 피어사가 무엇을 했는지
   - 어떤 사업/기술/업무를 다루는지
   - 고객 규모, 적용 범위, 후속 사업 단계 중 중요한 사실
5. 시사점, 전망, 대응 방향은 쓰지 말고 기사 속 사실만 쓰세요.
6. 기사에 명시된 예정·계획·후속 본사업 정보는 원문 수위를 유지해 쓸 수 있습니다.
7. "목표로 한다", "기대된다", "가능성이 있다"처럼 목적/평가식 표현보다
   "방안을 마련한다", "로드맵을 수립한다", "선행 단계다"처럼 기사에 근거한 사실 표현을 쓰세요.
8. fact_summary에 "목표로 한다"가 남아 있으면 검수 실패로 보고 반드시 사실형 서술로 고치세요.

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
  "headline": "피어사 사실 중심 한 문장",
  "one_line_summary": "기사 클러스터의 핵심 사실 1문장",
  "fact_summary": ["사실 1", "사실 2", "사실 3"],
  "main_event": "기사에 명시된 핵심 사건",
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

        target_companies = _candidate_peer_companies(articles)
        if not target_companies:
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason="SK AX를 제외한 피어사 후보를 찾지 못했습니다.",
                source_article_ids=_article_ids(articles),
            )

        articles_text = _format_articles(
            articles=articles,
            target_companies=target_companies,
            representative_id=representative_id,
        )
        article_fact_notes = _extract_article_fact_notes(articles_text)
        summary_context = _format_summary_context(article_fact_notes, articles_text)
        prompt = _PEER_NEWS_SUMMARY_PROMPT.replace("{articles_text}", summary_context)

        try:
            response = _get_llm().invoke(prompt)
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            result = _normalize_summary_result(_parse_json(content), target_companies)
            result = _repair_fact_summary_if_needed(
                result=result,
                articles=articles,
                articles_text=summary_context,
                target_companies=target_companies,
            )
        except Exception as exc:
            log.error("피어사 뉴스 요약 실패 | cluster=%s error=%s", cluster_id, exc)
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason=f"LLM 요약 실패: {type(exc).__name__}",
                source_article_ids=_article_ids(articles),
            )

        summary = {
            "cluster_id": cluster_id,
            "representative_id": representative_id,
            "source_article_ids": _article_ids(articles),
            "cluster_article_ids": list(cluster_article_ids or _article_ids(articles)),
            "summary_scope": "peer_company_fact_only",
            "excluded_company_tiers": ["self"],
            "target_peer_companies": target_companies,
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

    limit = max(1, max_cluster_articles)
    others = [article_id for article_id in cluster_article_ids if article_id != representative_id]
    return [representative_id, *others][:limit]


def _format_articles(
    articles: list[dict[str, Any]],
    target_companies: list[str],
    representative_id: int,
) -> str:
    lines = [
        f"cluster_target_peer_companies: {json.dumps(target_companies, ensure_ascii=False)}",
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


def _extract_article_fact_notes(articles_text: str) -> str:
    prompt = _ARTICLE_FACT_EXTRACTION_PROMPT.replace("{articles_text}", articles_text)
    try:
        response = _get_llm().invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_json(content)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception as exc:
        log.warning("기사별 팩트 추출 실패 | error=%s", exc)
        return ""


def _format_summary_context(article_fact_notes: str, articles_text: str) -> str:
    if not article_fact_notes:
        return articles_text
    return "\n\n".join(
        [
            "## article_fact_notes",
            article_fact_notes,
            "## original_articles",
            articles_text,
        ]
    )


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
    return {
        "is_valid_summary": bool(data.get("is_valid_summary", True)) and bool(facts),
        "main_company": main_company,
        "mentioned_peer_companies": mentioned or [main_company],
        "headline": str(data.get("headline") or "").strip(),
        "one_line_summary": str(data.get("one_line_summary") or "").strip(),
        "fact_summary": facts,
        "main_event": str(data.get("main_event") or "").strip(),
        "confidence": _clamp_float(data.get("confidence"), default=0.0),
        "reason": str(data.get("reason") or "").strip(),
    }


def _repair_fact_summary_if_needed(
    *,
    result: dict[str, Any],
    articles: list[dict[str, Any]],
    articles_text: str,
    target_companies: list[str],
) -> dict[str, Any]:
    prompt = (
        _FACT_SUMMARY_REPAIR_PROMPT.replace(
            "{domain_terms_json}",
            json.dumps(_domain_terms(articles), ensure_ascii=False),
        )
        .replace(
            "{summary_json}",
            json.dumps(result, ensure_ascii=False, indent=2),
        )
        .replace("{articles_text}", articles_text)
    )

    try:
        response = _get_llm().invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        repaired = _normalize_summary_result(_parse_json(content), target_companies)
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
) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "representative_id": representative_id,
        "source_article_ids": source_article_ids or [],
        "cluster_article_ids": source_article_ids or [],
        "summary_scope": "peer_company_fact_only",
        "excluded_company_tiers": ["self"],
        "target_peer_companies": [],
        "is_valid_summary": False,
        "main_company": "",
        "mentioned_peer_companies": [],
        "headline": "",
        "one_line_summary": "",
        "fact_summary": [],
        "main_event": "",
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
