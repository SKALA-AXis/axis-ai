"""Gate 2.5 내용 기반 관련성 판단 에이전트.

크롤링 단계에서 키워드 기반 후보 수집은 이미 수행되었다고 가정한다.
이 에이전트는 raw_articles에 저장된 원문을 읽고, 모니터링 대상 company와 sector 관점에서
실제로 분석할 가치가 있는 기사인지 판단한다.

판단 결과는 raw_articles에 업데이트하고, 관련 있는 기사 ID만 다음 Gate로 넘긴다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.config.companies import COMPANY_ALIASES
from src.config.sectors import match_sectors
from src.db.article_store import INDUSTRY_TREND_COMPANY
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.60
LLM_CONTENT_LIMIT = 1800

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm

    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_completion_tokens=500)

    return _llm

_RELEVANCE_PROMPT = """\
당신은 SK AX 전략 모니터링 시스템의 관련성 판단 Agent입니다.

아래 기사가 모니터링 대상 company와 sector 관점에서 실제 분석할 가치가 있는지 판단하세요.

## 판단 기준
relevant:
- 대상 company가 기사 핵심 주체다.
- sector와 실제 사업/기술 맥락상 관련 있다.
- 수주, 제휴, 기술 출시, 플랫폼 고도화, 채용 확대, 공시, IR, 실적, 시장 진출 등 동향 신호가 있다.

irrelevant:
- company가 단순 언급만 됐다.
- sector keyword가 단순 단어로만 등장한다.
- 행사 후기, 단순 인물 인터뷰, 광고성 내용 등 전략 모니터링 가치가 낮다.
- 대상 company가 아니라 그룹사나 다른 기업이 핵심 주체다.

uncertain:
- 관련성은 있으나 본문이 부족하거나 판단이 애매하다.

## 기사
title: {title}
content: {content}

## 수집 시 감지된 company
{company}

## 규칙 기반 감지 결과
matched_company_candidates: {matched_company_candidates}
matched_sector_candidates: {matched_sector_candidates}

JSON으로만 응답:
{{
  "relevance_label": "relevant|irrelevant|uncertain",
  "relevance_score": 0.0,
  "matched_companies": ["string"],
  "matched_sectors": ["string"],
  "reason": "1문장 근거"
}}
"""


class RelevanceAgent:
    """company/sector 관점의 내용 기반 관련성을 판단한다."""

    def filter(self, raw_article_ids: list[int]) -> tuple[list[int], list[int]]:
        """관련성 판단 후 relevant ID와 skipped ID를 반환한다.

        Args:
            raw_article_ids: 처리할 raw_articles ID 목록.

        Returns:
            (relevant_ids, skipped_ids) 튜플.
        """
        if not raw_article_ids:
            return [], []

        relevant_ids: list[int] = []
        skipped_ids: list[int] = []

        with SessionLocal() as db:
            rows = db.execute(
                text("""
                    SELECT
                        id,
                        title,
                        content,
                        company,
                        source_type,
                        crawl_status,
                        metadata
                    FROM raw_articles
                    WHERE id = ANY(:ids)
                """),
                {"ids": raw_article_ids},
            ).fetchall()

            for row in rows:
                result = self._analyze(row)

                is_relevant = _is_relevant(result)
                metadata_patch = _metadata_patch_for_relevance(row, result, is_relevant)

                db.execute(
                    text("""
                        UPDATE raw_articles
                        SET relevance_score = :relevance_score,
                            relevance_label = :relevance_label,
                            relevance_reason = :relevance_reason,
                            matched_companies = CAST(:matched_companies AS jsonb),
                            matched_sectors = CAST(:matched_sectors AS jsonb),
                            metadata = COALESCE(metadata, '{}'::jsonb)
                                || CAST(:metadata_patch AS jsonb),
                            processing_status = CASE
                                WHEN :is_relevant THEN processing_status
                                ELSE 'SKIPPED_RELEVANCE'
                            END
                        WHERE id = :id
                    """),
                    {
                        "relevance_score": result["relevance_score"],
                        "relevance_label": result["relevance_label"],
                        "relevance_reason": result["reason"],
                        "matched_companies": json.dumps(
                            result["matched_companies"],
                            ensure_ascii=False,
                        ),
                        "matched_sectors": json.dumps(
                            result["matched_sectors"],
                            ensure_ascii=False,
                        ),
                        "metadata_patch": json.dumps(metadata_patch, ensure_ascii=False),
                        "is_relevant": is_relevant,
                        "id": row.id,
                    },
                )

                if is_relevant:
                    relevant_ids.append(row.id)
                else:
                    skipped_ids.append(row.id)
                    log.debug(
                        "관련성 제외 | id=%d label=%s score=%.2f reason=%s",
                        row.id,
                        result["relevance_label"],
                        result["relevance_score"],
                        result["reason"],
                    )

            db.commit()

        log.info(
            "관련성 판단 완료 | total=%d relevant=%d skipped=%d",
            len(raw_article_ids),
            len(relevant_ids),
            len(skipped_ids),
        )

        return relevant_ids, skipped_ids

    def _analyze(self, row: Any) -> dict[str, Any]:
        title = row.title or ""
        content = row.content or ""
        company = _normalize_company(row.company)

        if row.crawl_status == "failed":
            return _result(
                label="irrelevant",
                score=0.0,
                companies=[],
                sectors=[],
                reason="수집 실패 상태라 관련성 판단 대상에서 제외",
            )

        if not title and not content:
            return _result(
                label="irrelevant",
                score=0.0,
                companies=[],
                sectors=[],
                reason="제목과 본문이 비어 있어 관련성 판단 불가",
            )

        text_body = f"{title} {content}"
        matched_company_candidates = _match_companies(text_body, company)
        matched_sector_candidates = match_sectors(text_body)

        precheck = _precheck(
            company=company,
            matched_companies=matched_company_candidates,
            matched_sectors=matched_sector_candidates,
            source_type=row.source_type,
            content=content,
        )

        if precheck["decision"] == "reject":
            return _result(
                label="irrelevant",
                score=precheck["score"],
                companies=matched_company_candidates,
                sectors=matched_sector_candidates,
                reason=precheck["reason"],
            )

        if precheck["decision"] == "accept":
            return _result(
                label="relevant",
                score=precheck["score"],
                companies=matched_company_candidates,
                sectors=matched_sector_candidates,
                reason=precheck["reason"],
            )

        return self._analyze_with_llm(
            title=title,
            content=content,
            company=company,
            matched_company_candidates=matched_company_candidates,
            matched_sector_candidates=matched_sector_candidates,
        )

    def _analyze_with_llm(
        self,
        title: str,
        content: str,
        company: list[str],
        matched_company_candidates: list[str],
        matched_sector_candidates: list[str],
    ) -> dict[str, Any]:
        prompt = _RELEVANCE_PROMPT.format(
            title=title,
            content=content[:LLM_CONTENT_LIMIT],
            company=company,
            matched_company_candidates=matched_company_candidates,
            matched_sector_candidates=matched_sector_candidates,
        )

        try:
            response = _get_llm().invoke(prompt)
            response_text = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            return _parse_relevance_json(response_text)
        except Exception as e:
            log.warning("관련성 LLM 판단 실패 | error=%s", e)

            if matched_company_candidates and matched_sector_candidates:
                return _result(
                    label="uncertain",
                    score=0.60,
                    companies=matched_company_candidates,
                    sectors=matched_sector_candidates,
                    reason="LLM 판단 실패, company와 sector 후보가 모두 감지되어 보류 통과",
                )

            return _result(
                label="irrelevant",
                score=0.30,
                companies=matched_company_candidates,
                sectors=matched_sector_candidates,
                reason="LLM 판단 실패, 관련성 근거 부족",
            )


def _precheck(
    company: list[str],
    matched_companies: list[str],
    matched_sectors: list[str],
    source_type: str | None,
    content: str,
) -> dict[str, Any]:
    has_company = bool(matched_companies)
    has_sector = bool(matched_sectors and matched_sectors != ["other"])
    content_len = len(content or "")

    if source_type in {"dart", "ir", "official"} and company:
        return {
            "decision": "accept",
            "score": 0.85,
            "reason": "공식성 source_type이며 수집 시 company가 지정된 자료",
        }

    if not has_company and not has_sector:
        return {
            "decision": "reject",
            "score": 0.10,
            "reason": "대상 company와 sector 후보가 모두 감지되지 않음",
        }

    if not has_company and source_type not in {"search_trend", "trend_report"}:
        return {
            "decision": "reject",
            "score": 0.25,
            "reason": "sector 후보는 있으나 대상 company가 본문에서 확인되지 않음",
        }

    if has_company and has_sector and content_len >= 500:
        return {
            "decision": "analyze",
            "score": 0.70,
            "reason": "company와 sector 후보가 모두 감지되어 내용 분석 필요",
        }

    if has_company and has_sector and content_len < 500:
        return {
            "decision": "accept",
            "score": 0.62,
            "reason": "본문은 짧지만 company와 sector 후보가 모두 감지됨",
        }

    return {
        "decision": "analyze",
        "score": 0.50,
        "reason": "관련성 판단이 애매하여 내용 분석 필요",
    }


def _is_relevant(result: dict[str, Any]) -> bool:
    label = result.get("relevance_label", "irrelevant")
    score = float(result.get("relevance_score", 0.0))

    if label == "relevant" and score >= RELEVANCE_THRESHOLD:
        return True

    if label == "uncertain" and score >= RELEVANCE_THRESHOLD:
        return True

    return False


def _metadata_patch_for_relevance(
    row: Any,
    result: dict[str, Any],
    is_relevant: bool,
) -> dict[str, Any]:
    if not is_relevant:
        return {}

    companies = _normalize_company(row.company)
    matched_companies = result.get("matched_companies") or []
    matched_sectors = result.get("matched_sectors") or []
    has_sector = bool(matched_sectors and matched_sectors != ["other"])

    if not has_sector and row.source_type not in {"search_trend", "trend_report"}:
        return {}

    is_industry_bucket = INDUSTRY_TREND_COMPANY in companies
    is_companyless_sector = has_sector and not matched_companies
    is_multi_peer_sector = has_sector and len(matched_companies) >= 2
    is_trend_source = row.source_type in {"search_trend", "trend_report"}

    if not (
        is_industry_bucket
        or is_companyless_sector
        or is_multi_peer_sector
        or is_trend_source
    ):
        return {}

    patch: dict[str, Any] = {
        "topic_scope": "industry_trend",
        "matched_companies": matched_companies,
        "matched_sectors": matched_sectors,
    }

    if is_multi_peer_sector:
        patch["primary_company"] = None

    return patch


def _match_companies(text_body: str, company: list[str]) -> list[str]:
    matched: list[str] = []

    for company_id in company:
        aliases = COMPANY_ALIASES.get(company_id, [company_id])
        if any(_compact(alias) in _compact(text_body) for alias in aliases):
            matched.append(company_id)

    return _dedupe_keep_order(matched)


def _normalize_company(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v) for v in value if v]

    if isinstance(value, tuple):
        return [str(v) for v in value if v]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except json.JSONDecodeError:
            pass

        stripped = value.strip()
        return [stripped] if stripped else []

    return []


def _parse_relevance_json(text_value: str) -> dict[str, Any]:
    text_value = text_value.strip()

    if text_value.startswith("```"):
        text_value = text_value.split("```")[1]
        if text_value.startswith("json"):
            text_value = text_value[4:]

    data = json.loads(text_value.strip())

    label = data.get("relevance_label", "uncertain")
    if label not in {"relevant", "irrelevant", "uncertain"}:
        label = "uncertain"

    return _result(
        label=label,
        score=float(data.get("relevance_score", 0.60)),
        companies=data.get("matched_companies", []),
        sectors=data.get("matched_sectors", []),
        reason=data.get("reason", ""),
    )


def _result(
    label: str,
    score: float,
    companies: list[str],
    sectors: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "relevance_label": label,
        "relevance_score": round(float(score), 3),
        "matched_companies": _dedupe_keep_order([str(c) for c in companies if c]),
        "matched_sectors": _dedupe_keep_order([str(s) for s in sectors if s]),
        "reason": reason,
    }


def _compact(value: str) -> str:
    return "".join(value.lower().split())


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result
