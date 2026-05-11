"""피어사 뉴스 사실 요약 에이전트.

클러스터에 묶인 기사들을 바탕으로 피어사 관련 본문 내용을 요약한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.config.companies import COMPANY_ALIASES
from src.config.company_tiers import company_tier
from src.config.global_companies import GLOBAL_COMPANY_ALIASES
from src.db.article_store import get_articles_by_ids

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_MAX_PRIMARY_CONTENT = 1800
_MAX_SUPPORT_CONTENT = 700
_DEFAULT_MAX_SUPPORT_ARTICLES = 4
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


_PEER_NEWS_SUMMARY_PROMPT = """\
당신은 피어사 뉴스 사실 요약 Agent입니다.

목적:
- 기사 클러스터에 포함된 여러 보도에서 공통으로 확인되는 사실을 요약합니다.
- 요약 대상은 SK AX를 제외한 peer company입니다.

## 입력
- cluster_id: 클러스터 ID
- target_peer_companies: 요약 대상 후보 피어사 company id 목록
- articles: 대표 기사와 보조 기사 목록

## 내부 요약 설계
응답에는 아래 사고 과정을 드러내지 말고, 결과 JSON만 작성하세요.
1. 기사별로 who/what/when/where/how에 해당하는 사실을 먼저 분리해서 읽으세요.
2. 대표 기사만 믿지 말고 보조 기사와 겹치는 사실을 우선순위로 두세요.
3. 피어사명, 제품명, 서비스명, 기능, 발표/계약/출시 같은 사건 유형을 구분하세요.
4. 같은 사실을 다른 표현으로 쓴 문장은 하나로 합치고, 더 구체적인 표현을 선택하세요.
5. 기사 간 내용이 충돌하면 더 보수적인 문장으로 쓰거나 해당 세부사항은 제외하세요.
6. 핵심 사건이 기술/제품/서비스라면 "무엇을 하는 것인지"를 한 문장에 포함하세요.
7. 사업 규모, 고객 규모, 후속 본사업, 적용 범위처럼 이슈의 중요도를 설명하는
   정량·맥락 정보가 기사에 있으면 fact_summary 안에 우선 반영하세요.
8. 숫자나 규모가 일부 기사에만 있고 표현이 "예상", "전망", "향후"라면 확정처럼 쓰지 말고
   "향후 대규모 본 구축 사업의 선행 단계"처럼 보수적으로 표현하세요.
9. 피어사/경쟁사 동향을 파악하는 데 중요한 기사 속 사실을 우선 선택하세요.
   - 수주, 출시, 제휴, 투자, 인수, 채용, 조직개편처럼 실제 움직임이 드러나는 사건
   - 기사에 명시된 사업 영역, 기술·제품·서비스, 고객군, 산업, 공공/민간 시장
   - 기사에 명시된 사업 규모, 고객 규모, 적용 범위, 후속 사업 단계
10. 단순 기사 소개보다 중요한 사실을 선별하되, 시사점·대응 방향·전략 해석은 쓰지 마세요.
    그런 내용은 다른 Agent가 담당합니다.
11. 3문장 제한 때문에 모든 사실을 담을 수 없으면 사업 행동, 기술·서비스 내용,
    고객/시장 규모, 후속 사업 단계 순으로 우선 선택하고, 참석자·사진 설명·일반 배경은 제외하세요.
12. 여러 규모 정보가 경쟁하면 내부 사용자 수보다 고객 규모, 시장 규모, 계약/사업 규모,
    후속 사업 단계를 우선하세요.
13. "목표로 한다", "강화할 수 있다"처럼 추상적인 표현보다 "방안을 마련하는 프로젝트다",
    "수행하는 사업이다", "선행 단계다"처럼 기사에 근거한 구체적 사실 표현을 우선하세요.
    특히 fact_summary에는 가능하면 "목표로 한다" 표현을 쓰지 마세요.
14. fact_summary 3문장은 각각 역할이 겹치지 않게 구성하세요.
   - 1번째 문장: 피어사가 무엇을 했는지
   - 2번째 문장: 기사에 명시된 제품/서비스/기술/사업 내용.
     "이 사업은 ... 목표로 한다"보다 "이 사업은 ... 방안을 마련하는 프로젝트다",
     "이 사업은 ... 체계를 설계하는 컨설팅이다"처럼 정리하세요.
   - 3번째 문장: 고객 규모, 적용 범위, 후속 본사업, 시장/산업 맥락 중 기사에 있는 중요 사실
15. 임직원 수처럼 서비스 내부 사용자 규모는 고객 규모·사업 규모·후속 본사업 정보보다
    중요도가 낮습니다. 3문장 안에 모두 담을 수 없으면 내부 사용자 규모는 제외하세요.
16. 3번째 문장은 가능하면 "고객/시장 규모 + 후속 사업 단계"를 한 문장으로 결합하세요.
    예: "해당 시스템은 기사에 명시된 고객군의 핵심 업무를 담당하며,
    이번 사업은 후속 본사업을 위한 선행 단계다."

## 작성 원칙
1. 기사에 명시된 사실만 요약하세요.
2. 추론, 전망, 평가, 경쟁 대응 제언은 쓰지 마세요.
3. 피어사를 주어로 작성하세요
4. 여러 기사 내용이 충돌하면 공통으로 확인되는 사실만 쓰세요.
5. 본문에 없는 수치, 제품명, 회사명은 만들지 마세요.
6. 기사 내용이 피어사 핵심 내용이 아니면 is_valid_summary=false로 표시하세요.
7. 기술·제품·서비스가 핵심이면, 기사에 근거해 "무엇을 하는 기술인지"를
   fact_summary 안에 1문장 포함하세요.
8. 기술 설명은 평가나 전망 없이 기능·구성·용도 중심으로 작성하세요.
9. fact_summary는 경쟁사 동향을 빠르게 파악하기 위한 카드뉴스 본문이므로,
   단순 배경보다 "무슨 사업/기술을 어떤 고객·시장에 적용하는지"를 우선하세요.

## 기사 클러스터
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
        max_support_articles: int = _DEFAULT_MAX_SUPPORT_ARTICLES,
    ) -> dict[str, Any]:
        """DB의 raw_articles를 읽어 클러스터 사실 요약을 생성한다."""
        ids_to_fetch = _build_fetch_ids(
            representative_id=representative_id,
            cluster_article_ids=cluster_article_ids,
            max_support_articles=max_support_articles,
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
        prompt = _PEER_NEWS_SUMMARY_PROMPT.replace("{articles_text}", articles_text)

        try:
            response = _get_llm().invoke(prompt)
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            result = _normalize_summary_result(_parse_json(content), target_companies)
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
    max_support_articles: int,
) -> list[int]:
    if not cluster_article_ids:
        return [representative_id]

    limit = max(1, max_support_articles)
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
        content_limit = (
            _MAX_PRIMARY_CONTENT if article_id == representative_id else _MAX_SUPPORT_CONTENT
        )
        metadata = _metadata(article)
        lines.append(
            "\n".join(
                [
                    f"[{index}] article_id: {article_id}",
                    f"role: {'representative' if article_id == representative_id else 'support'}",
                    f"title: {article.get('title') or ''}",
                    f"source_name: {article.get('source_name') or ''}",
                    f"publisher: {article.get('publisher') or ''}",
                    f"published_at: {article.get('published_at') or ''}",
                    f"company: {json.dumps(_company_list(article), ensure_ascii=False)}",
                    "matched_companies: "
                    f"{json.dumps(_matched_companies(article), ensure_ascii=False)}",
                    f"metadata: {json.dumps(_summary_metadata(metadata), ensure_ascii=False)}",
                    f"content: {_truncate(article.get('content') or '', content_limit)}",
                ]
            )
        )

    return "\n\n".join(lines)


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


def _truncate(text: str, limit: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "..."


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
