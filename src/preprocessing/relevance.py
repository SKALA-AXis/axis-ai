"""Gate 2.5 내용 기반 관련성 판단 전처리.

크롤링 단계에서 키워드 기반 후보 수집은 이미 수행되었다고 가정한다.
이 단계는 raw_articles에 저장된 원문을 읽고, 모니터링 대상 company와 sector 관점에서
실제로 분석할 가치가 있는 기사인지 판단한다.

판단 결과는 raw_articles에 업데이트하고, 관련 있는 기사 ID만 다음 Gate로 넘긴다.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.config.companies import COMPANY_ALIASES
from src.config.global_companies import GLOBAL_COMPANY_ALIASES
from src.config.sectors import SECTOR_IDS, match_sectors
from src.db.article_store import INDUSTRY_TREND_COMPANY
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.60
UNCERTAIN_CANDIDATE_THRESHOLD = 0.45
PEER_CONTEXT_LIMIT = 1800

# 본문에서 peer alias 언급 *최소 횟수* — 이 횟수 미만이면 "단순 언급" 으로 drop.
#   default 1 (완화) — 1주 카드 누적량 ↑ 목적.
#   과거 값은 2 였으나, cycle 마다 relevance 통과량이 적어 classify=0 이 지배적.
#   noise 증가 시 env 로 2 또는 3 으로 상향 가능.
_MIN_PEER_MENTIONS = int(os.getenv("RELEVANCE_MIN_PEER_MENTIONS", "1"))
ALL_COMPANY_ALIASES = {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}

_llm: ChatOpenAI | None = None
_PROMPT_VERSION = "relevance-v1.0"


def _get_llm() -> ChatOpenAI:
    global _llm

    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_completion_tokens=500)

    return _llm


_RELEVANCE_PROMPT = """\
당신은 기사 본문의 프로젝트 목적과의 관련성 판단 Agent입니다.

이 Agent의 목적은 크롤링된 후보 기사 중에서 Peer사의 전략 모니터링 대상으로
분석할 가치가 있는 기사만 선별하는 것입니다.
단순 키워드 포함 여부가 아니라, company 관련성, sector 관련성, 동향 신호를 분리해서 판단해야 합니다.

## 입력 정보
- title: 기사 제목
- content: 기사 본문
- source_type: 기사 출처 타입
- target_companies: 수집 단계에서 모니터링 대상으로 지정된 company 목록
- matched_company_candidates: 규칙 기반으로 본문에서 감지된 company 후보
- matched_sector_candidates: 규칙 기반으로 본문에서 감지된 sector 후보
- peer_context: 본문 전체에서 피어사명 주변 문장과 행위 키워드 문장을 추출한 참고 문맥

## 판단 원칙
아래 3가지가 모두 충족되면 relevant로 판단하세요.

1. Company 관련성
- 대상 company가 기사에서 핵심 주체로 등장해야 합니다.
- 핵심 주체란 수주, 계약, 제휴, 투자, 출시, 실적, 공시, IR, 채용 확대,
  조직 개편, 시장 진출, 기술 개발, 고객 확보 등의 행위를 수행하거나
  그 영향을 받는 기업을 의미합니다.
- company 이름이 단순 나열, 광고, 태그, 관련 기사, 행사 후원사,
  배경 설명에만 등장하면 핵심 주체로 보지 마세요.
- 그룹사가 언급되더라도 target_companies에 포함된 회사가 핵심 주체가 아니면 irrelevant로 판단하세요.
- 여러 peer사가 함께 언급된 경우, 경쟁 구도, 비교, 협력,
  시장 변화 관점에서 의미 있게 다뤄지면 relevant로 판단할 수 있습니다.

2. Sector 관련성
- sector는 단순 키워드가 아니라 실제 사업, 기술, 시장 맥락과 연결되어야 합니다.
- 예를 들어 AI, 클라우드, 보안, 인프라, 제조 AX, 물류, ERP,
  데이터센터, 스마트팩토리, 공공 DX 등이 회사의 사업 변화나
  기술 변화와 연결되면 sector 관련성이 있습니다.
- sector 단어가 일반 표현, 배경 설명, 비유, 문장 장식으로만 등장하면 sector 관련성이 낮습니다.
- matched_sector_candidates가 ["other"]이거나 비어 있으면 sector 관련성은 낮게 판단하세요.
  단, 공시, IR, 실적, 대규모 수주처럼 회사 동향 자체가 명확하면
  relevant가 될 수 있습니다.

3. 동향 신호
- 전략 모니터링에 사용할 수 있는 변화 신호가 있어야 합니다.
- 동향 신호에는 신규 수주, 계약, 제휴, 투자, 인수합병, 신제품 출시,
  서비스 출시, 플랫폼 고도화, 기술 개발, 특허, 채용 확대, 조직 개편,
  실적 변화, 공시, IR, 신규 시장 진출, 고객사 확보, 정부 사업 참여,
  정책 변화, 산업 트렌드 변화가 포함됩니다.
- 단순 행사 참석, 단순 수상, 단순 인물 인터뷰, 광고성 기사,
  제품 홍보만 있는 기사는 동향 신호가 약하므로 irrelevant로 판단하세요.
- 전시회/컨퍼런스/박람회/시상식/행사 일정 안내 기사에서 company가
  참가사·후원사·발표 기업 목록에만 등장하면 irrelevant로 판단하세요.
- 주가 등락, 장중 시황, 종목별 상승·하락 마감 기사라도 제목과 본문이
  특정 피어사 1곳을 직접 다루고 있으면 company 상황 신호로 relevant 판단할 수 있습니다.
  단, 여러 종목을 묶은 시황·ETF·테마주·브리핑 기사는 irrelevant로 판단하세요.

## label 정의
relevant:
- company 관련성, sector 관련성, 동향 신호가 모두 명확합니다.
- 또는 공시, IR, 실적, 대규모 수주처럼 company 동향 자체가 명확하여
  sector가 약해도 분석 가치가 높습니다.

irrelevant:
- company가 단순 언급입니다.
- sector keyword가 단순 단어 수준입니다.
- 동향 신호가 없습니다.
- 대상 company가 아니라 다른 회사나 그룹사가 핵심 주체입니다.
- 광고성, 행사성, 인물성, 단순 홍보성 기사입니다.

uncertain:
- 본문이 부족해 핵심 주체나 동향 신호를 확정하기 어렵습니다.
- company와 sector 후보는 있으나 실제 사업 맥락인지 불명확합니다.
- 관련 가능성은 있으나 근거가 약합니다.

## score 기준
- 0.90 ~ 1.00: company, sector, 동향 신호가 모두 매우 명확함
- 0.75 ~ 0.89: 관련성이 높고 분석 가치가 있음
- 0.60 ~ 0.74: 관련 가능성은 있으나 일부 근거가 약함
- 0.40 ~ 0.59: 애매하거나 근거 부족
- 0.00 ~ 0.39: 관련성 낮음

## 출력 규칙
- 반드시 JSON만 출력하세요.
- 마크다운 코드블록을 사용하지 마세요.
- relevance_label은 relevant, irrelevant, uncertain 중 하나만 사용하세요.
- relevance_score는 0.0부터 1.0 사이의 숫자로 작성하세요.
- matched_companies는 실제 기사 맥락상 의미 있게 등장한 company만 포함하세요.
- matched_sectors는 실제 기사 맥락상 의미 있게 연결된 sector만 포함하세요.
- reason은 판단 근거를 1문장으로 작성하세요.
- reason에는 company 관련성, sector 관련성, 동향 신호 중 무엇이 충족되었거나 부족한지 포함하세요.

## 기사
title: {title}
content: {content}
source_type: {source_type}

## 수집 시 감지된 target_companies
{company}

## 규칙 기반 감지 결과
matched_company_candidates: {matched_company_candidates}
matched_sector_candidates: {matched_sector_candidates}
peer_context: {peer_context}

## JSON 출력 형식
{{
  "relevance_label": "relevant|irrelevant|uncertain",
  "relevance_score": 0.0,
  "matched_companies": ["string"],
  "matched_sectors": ["string"],
  "reason": "1문장 근거"
}}
"""


class RelevanceEvaluator:
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

        log.info("Gate 2.5 관련성 전처리 시작 | total=%d", len(raw_article_ids))

        with SessionLocal() as db:
            rows = db.execute(
                text("""
                    SELECT
                        ra.id,
                        ra.title,
                        ra.content,
                        ra.company,
                        ra.source_type,
                        ra.crawl_status,
                        COALESCE(mu.metadata, '{}'::jsonb) AS metadata
                    FROM raw_articles ra
                    LEFT JOIN raw_article_metadata_unified mu
                        ON mu.raw_article_id = ra.id
                    WHERE ra.id = ANY(:ids)
                """),
                {"ids": raw_article_ids},
            ).fetchall()

            for row in rows:
                log.info(
                    "Gate 2.5 기사 전처리 중 | id=%s source_type=%s title=%s",
                    row.id,
                    row.source_type,
                    _shorten(row.title or "", 80),
                )
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
                        "is_relevant": is_relevant,
                        "id": row.id,
                    },
                )
                if metadata_patch:
                    from src.db.article_store import _upsert_source_metadata

                    _upsert_source_metadata(
                        db,
                        article_id=row.id,
                        source_type=row.source_type,
                        source_metadata=json.dumps(metadata_patch, ensure_ascii=False),
                    )

                if is_relevant:
                    relevant_ids.append(row.id)
                    log.info(
                        "Gate 2.5 통과 | id=%s label=%s score=%.2f companies=%s sectors=%s",
                        row.id,
                        result["relevance_label"],
                        result["relevance_score"],
                        result["matched_companies"],
                        result["matched_sectors"],
                    )
                else:
                    skipped_ids.append(row.id)
                    log.info(
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
        metadata = _metadata_dict(_row_value(row, "metadata", {}))
        subtitle = _subtitle_text(metadata)
        analysis_content = _join_text(subtitle, content)
        company = _normalize_company(row.company)

        if row.crawl_status == "failed":
            return _result(
                label="irrelevant",
                score=0.0,
                companies=[],
                sectors=[],
                reason="수집 실패 상태라 관련성 판단 대상에서 제외",
            )

        if not title and not analysis_content:
            return _result(
                label="irrelevant",
                score=0.0,
                companies=[],
                sectors=[],
                reason="제목과 본문이 비어 있어 관련성 판단 불가",
            )

        text_body = _join_text(title, subtitle, content)
        matched_company_candidates = _match_companies(text_body, company)
        matched_sector_candidates = match_sectors(text_body)

        noise_result = _noise_reject_result(
            title=title,
            content=analysis_content,
            source_type=row.source_type,
            matched_companies=matched_company_candidates,
            matched_sectors=matched_sector_candidates,
        )
        if noise_result is not None:
            log.info(
                "Gate 2.5 노이즈 기사 제외 | id=%s reason=%s",
                getattr(row, "id", None),
                noise_result["reason"],
            )
            return noise_result

        precheck = _precheck(
            company=company,
            matched_companies=matched_company_candidates,
            matched_sectors=matched_sector_candidates,
            source_type=row.source_type,
        )

        log.info(
            "Gate 2.5 규칙 전처리 | id=%s decision=%s company_candidates=%s "
            "sector_candidates=%s reason=%s",
            getattr(row, "id", None),
            precheck["decision"],
            matched_company_candidates,
            matched_sector_candidates,
            precheck["reason"],
        )

        if precheck["decision"] == "reject":
            return _result(
                label="irrelevant",
                score=precheck["score"],
                companies=matched_company_candidates,
                sectors=matched_sector_candidates,
                reason=precheck["reason"],
            )

        mention_result = _weak_company_mention_reject_result(
            title=title,
            content=analysis_content,
            source_type=row.source_type,
            matched_companies=matched_company_candidates,
            matched_sectors=matched_sector_candidates,
        )
        if mention_result is not None:
            log.info(
                "Gate 2.5 피어사 언급 횟수 부족 제외 | id=%s reason=%s",
                getattr(row, "id", None),
                mention_result["reason"],
            )
            return mention_result

        role_result = _core_company_role_reject_result(
            title=title,
            content=analysis_content,
            source_type=row.source_type,
            matched_companies=matched_company_candidates,
            matched_sectors=matched_sector_candidates,
        )
        if role_result is not None:
            log.info(
                "Gate 2.5 피어사 핵심성 부족 제외 | id=%s reason=%s",
                getattr(row, "id", None),
                role_result["reason"],
            )
            return role_result

        fast_pass_result = _fast_pass_result(
            title=title,
            content=analysis_content,
            source_type=row.source_type,
            matched_companies=matched_company_candidates,
            matched_sectors=matched_sector_candidates,
        )
        if fast_pass_result is not None:
            log.info(
                "Gate 2.5 fast-pass | id=%s reason=%s",
                getattr(row, "id", None),
                fast_pass_result["reason"],
            )
            return fast_pass_result

        return self._analyze_with_llm(
            title=title,
            content=analysis_content,
            source_type=row.source_type,
            company=company,
            matched_company_candidates=matched_company_candidates,
            matched_sector_candidates=matched_sector_candidates,
            peer_context=_peer_context_snippets(
                title=title,
                content=analysis_content,
                matched_companies=matched_company_candidates,
            ),
        )

    def _analyze_with_llm(
        self,
        title: str,
        content: str,
        source_type: str | None,
        company: list[str],
        matched_company_candidates: list[str],
        matched_sector_candidates: list[str],
        peer_context: str,
    ) -> dict[str, Any]:
        prompt = _RELEVANCE_PROMPT.format(
            title=title,
            content=content,
            source_type=source_type or "",
            company=company,
            matched_company_candidates=matched_company_candidates,
            matched_sector_candidates=matched_sector_candidates,
            peer_context=peer_context[:PEER_CONTEXT_LIMIT],
        )

        try:
            log.info(
                "Gate 2.5 LLM 내용 분석 중 | source_type=%s companies=%s sectors=%s",
                source_type,
                matched_company_candidates,
                matched_sector_candidates,
            )
            from src.observability import tracing_config

            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="RelevanceEvaluator",
                    prompt_version=_PROMPT_VERSION,
                    source_type=source_type,
                ),
            )
            response_text = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            result = _parse_relevance_json(response_text)
            log.info(
                "Gate 2.5 LLM 판단 완료 | label=%s score=%.2f reason=%s",
                result["relevance_label"],
                result["relevance_score"],
                result["reason"],
            )
            return result
        except Exception as e:
            log.warning("관련성 LLM 판단 실패 | error=%s", e)

            if matched_company_candidates and matched_sector_candidates:
                return _result(
                    label="uncertain",
                    score=0.50,
                    companies=matched_company_candidates,
                    sectors=matched_sector_candidates,
                    reason="LLM 판단 실패로 핵심 주체와 동향 신호를 확정하지 못함",
                )

            return _result(
                label="irrelevant",
                score=0.30,
                companies=matched_company_candidates,
                sectors=matched_sector_candidates,
                reason="LLM 판단 실패, 관련성 근거 부족",
            )


def analyze_relevance_article(article: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """JSON article에 Gate 2.5 관련성 결과를 붙인다."""

    row = _DictRow(article)
    result = RelevanceEvaluator()._analyze(row)
    is_relevant = _is_relevant(result)

    item = dict(article)
    item.update(
        {
            "relevance_label": result["relevance_label"],
            "relevance_score": result["relevance_score"],
            "relevance_reason": result["reason"],
            "matched_companies": result["matched_companies"],
            "matched_sectors": result["matched_sectors"],
        }
    )

    if not is_relevant:
        item["processing_status"] = "SKIPPED_RELEVANCE"

    return item, is_relevant


class _DictRow:
    def __init__(self, article: dict[str, Any]) -> None:
        self.id = article.get("id")
        self.title = article.get("title")
        self.content = article.get("content")
        self.company = article.get("company")
        self.source_type = article.get("source_type")
        self.crawl_status = article.get("crawl_status", "success")
        self.metadata = article.get("metadata") or article.get("extra") or {}


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    mapping = getattr(row, "_mapping", None)
    if mapping is not None and key in mapping:
        return mapping[key]
    return getattr(row, key, default)


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _subtitle_text(metadata: dict[str, Any]) -> str:
    for key in ("subtitle", "sub_title", "description"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _join_text(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _precheck(
    company: list[str],
    matched_companies: list[str],
    matched_sectors: list[str],
    source_type: str | None,
) -> dict[str, Any]:
    has_company = bool(matched_companies)
    has_sector = bool(matched_sectors and matched_sectors != ["other"])
    is_company_source = source_type in {"dart", "ir", "official", "company_site"} and bool(company)

    if not has_company and not has_sector and not is_company_source:
        return {
            "decision": "reject",
            "score": 0.10,
            "reason": "대상 company와 sector 후보가 모두 감지되지 않음",
        }

    if not has_company and source_type not in {
        "dart",
        "ir",
        "official",
        "company_site",
        "search_trend",
        "trend_report",
    }:
        return {
            "decision": "reject",
            "score": 0.25,
            "reason": "sector 후보는 있으나 대상 company가 본문에서 확인되지 않음",
        }

    return {
        "decision": "analyze",
        "score": 0.50,
        "reason": "규칙 후보 기반으로 LLM 내용 분석 필요",
    }


_MARKET_PRICE_RE = re.compile(
    r"(주가|종가|장중|상승\s*마감|하락\s*마감|강세|약세|급등|급락|상한가|하한가|시가총액)"
)
_MARKET_METRIC_RE = re.compile(r"(\d+(?:\.\d+)?\s*%|\d{1,3}(?:,\d{3})+\s*원)")
_EVENT_LISTING_KEYWORDS = [
    "전시회",
    "박람회",
    "컨퍼런스",
    "세미나",
    "포럼",
    "행사",
    "코엑스",
    "개최",
    "참가",
    "총집결",
    "부스",
    "시상식",
]
_STRATEGIC_ACTION_KEYWORDS = [
    "수주",
    "계약",
    "구축",
    "선정",
    "우선협상",
    "협약",
    "업무협약",
    "제휴",
    "공동",
    "출시",
    "공개",
    "개발",
    "투자",
    "인수",
    "합병",
    "실적",
    "공시",
    "조직개편",
    "채용",
]
_STRONG_STRATEGIC_ACTION_KEYWORDS = [
    "수주",
    "계약",
    "구축",
    "우선협상",
    "협약",
    "업무협약",
    "제휴",
    "투자",
    "인수",
    "합병",
    "실적",
    "공시",
    "조직개편",
    "채용",
]
_DIRECT_COMPANY_ROLE_KEYWORDS = [
    *_STRONG_STRATEGIC_ACTION_KEYWORDS,
    "선정",
    "참여",
    "맡",
    "적용",
    "실증",
    "운영",
    "공급",
    "제공",
    "등록",
    "할당",
    "도입",
    "확정",
    "추진",
]
_FAST_PASS_ACTION_KEYWORDS = [
    *_STRONG_STRATEGIC_ACTION_KEYWORDS,
    "출시",
    "공개",
    "선정",
    "개발",
    "고도화",
    "플랫폼",
]
_FAST_PASS_SOURCE_TYPES = {"news"}
_ROLE_CONTEXT_WINDOW = 100
_LISTING_CONTEXT_KEYWORDS = [
    "etf",
    "펀드",
    "포트폴리오",
    "편입",
    "구성종목",
    "관련주",
    "테마주",
    "수익률",
    "종목",
    "시황",
]
_MARKET_LISTING_KEYWORDS = [
    *_LISTING_CONTEXT_KEYWORDS,
    "특징주",
    "목표가",
    "투자의견",
    "시가총액",
    "per",
    "주가수익비율",
    "코스피",
    "코스닥",
]
_SUBJECT_MARKERS = ["은", "는", "이", "가"]


def _noise_reject_result(
    *,
    title: str,
    content: str,
    source_type: str | None,
    matched_companies: list[str],
    matched_sectors: list[str],
) -> dict[str, Any] | None:
    text = f"{title} {content}"
    compact_text = _compact(text)

    if _is_non_korean_news_title(title=title, source_type=source_type):
        return _result(
            label="irrelevant",
            score=0.25,
            companies=matched_companies,
            sectors=matched_sectors,
            reason="한국어 뉴스 모니터링 대상에서 제외: 제목에 한글이 없는 외국어 기사",
        )

    has_peer_strategy_signal = _has_peer_strategy_signal(
        title=title,
        content=content,
        matched_companies=matched_companies,
        matched_sectors=matched_sectors,
    )
    has_market_listing_noise = _is_market_listing_noise(title=title, content=content)
    has_sector = bool(matched_sectors and matched_sectors != ["other"])
    is_company_market_signal = _is_company_market_signal(
        title=title,
        content=content,
        matched_companies=matched_companies,
    )

    if has_market_listing_noise and not _company_in_title(
        title=title,
        matched_companies=matched_companies,
    ):
        return _result(
            label="irrelevant",
            score=0.25,
            companies=matched_companies,
            sectors=matched_sectors,
            reason="ETF·테마주·종목 브리핑 성격의 제목에서 피어사가 핵심 주체로 드러나지 않아 제외",
        )

    if _is_market_price_noise(text) and is_company_market_signal:
        return _result(
            label="relevant",
            score=0.62,
            companies=matched_companies,
            sectors=matched_sectors,
            reason=(
                "특정 피어사 1곳의 주가·거래·밸류에이션을 직접 다루는 기사라 "
                "company 상황 신호로 판단"
            ),
        )

    if _is_market_price_noise(text) and (not has_sector or not has_peer_strategy_signal):
        return _result(
            label="irrelevant",
            score=0.20,
            companies=matched_companies,
            sectors=matched_sectors,
            reason=(
                "주가 등락/시황 중심 기사라 뉴스 relevance 대상에서 제외하고 market_data에서 처리"
            ),
        )

    if has_market_listing_noise and not has_peer_strategy_signal:
        return _result(
            label="irrelevant",
            score=0.25,
            companies=matched_companies,
            sectors=matched_sectors,
            reason="ETF·테마주·투자의견·종목 브리핑 중심 기사라 피어사 동향 카드 후보에서 제외",
        )

    if _is_event_listing_noise(
        title=title,
        compact_text=compact_text,
        matched_companies=matched_companies,
    ):
        return _result(
            label="irrelevant",
            score=0.30,
            companies=matched_companies,
            sectors=matched_sectors,
            reason=(
                "전시회·행사 안내에서 company가 참가사/나열 대상으로만 등장해 전략 동향 신호가 약함"
            ),
        )

    return None


def _company_in_title(*, title: str, matched_companies: list[str]) -> bool:
    title_compact = _compact(title)
    return any(
        _compact(alias) and _compact(alias) in title_compact
        for company_id in matched_companies
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id])
    )


def _is_company_market_signal(
    *,
    title: str,
    content: str,
    matched_companies: list[str],
) -> bool:
    if len(set(matched_companies)) != 1:
        return False
    if not _company_in_title(title=title, matched_companies=matched_companies):
        return False
    return _is_market_price_noise(f"{title} {content}")


def _has_peer_strategy_signal(
    *,
    title: str,
    content: str,
    matched_companies: list[str],
    matched_sectors: list[str],
) -> bool:
    """주가 기사라도 피어사 사업 이벤트가 있으면 LLM 판단으로 넘긴다."""
    if not matched_companies:
        return False

    text_compact = _compact(f"{title} {content}")
    has_sector = bool(matched_sectors and matched_sectors != ["other"])
    has_action = any(_compact(keyword) in text_compact for keyword in _STRATEGIC_ACTION_KEYWORDS)
    if not has_action:
        return False

    for company_id in matched_companies:
        aliases = ALL_COMPANY_ALIASES.get(company_id, [company_id])
        for alias in aliases:
            alias_compact = _compact(alias)
            if not alias_compact or alias_compact not in text_compact:
                continue
            if _has_direct_role_keyword_near_alias(text_compact, alias_compact):
                return True
            if (
                has_sector
                and _alias_appears_as_subject(text_compact, alias_compact)
                and not _has_listing_context_near_alias(text_compact, alias_compact)
            ):
                return True

    return False


def _fast_pass_result(
    *,
    title: str,
    content: str,
    source_type: str | None,
    matched_companies: list[str],
    matched_sectors: list[str],
) -> dict[str, Any] | None:
    if str(source_type or "").strip().lower() not in _FAST_PASS_SOURCE_TYPES:
        return None

    if not matched_companies:
        return None

    title_compact = _compact(title)
    text_compact = _compact(f"{title} {content}")

    company_in_title = any(
        _compact(alias) in title_compact
        for company_id in matched_companies
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id])
    )
    if not company_in_title:
        return None

    has_strong_action = any(
        _compact(keyword) in text_compact for keyword in _FAST_PASS_ACTION_KEYWORDS
    )
    if not has_strong_action:
        return None

    has_sector = bool(matched_sectors and matched_sectors != ["other"])
    score = 0.82 if has_sector else 0.72

    return _result(
        label="relevant",
        score=score,
        companies=matched_companies,
        sectors=matched_sectors,
        reason=(
            "fast-pass: 제목에 피어사가 있고 명확한 핵심 이벤트 키워드가 있어 "
            "LLM 없이 관련 기사로 판단"
        ),
    )


def _core_company_role_reject_result(
    *,
    title: str,
    content: str,
    source_type: str | None,
    matched_companies: list[str],
    matched_sectors: list[str],
) -> dict[str, Any] | None:
    if str(source_type or "").strip().lower() != "news":
        return None

    if not matched_companies:
        return None

    title_compact = _compact(title)
    full_compact = _compact(f"{title} {content}")

    for company_id in matched_companies:
        aliases = ALL_COMPANY_ALIASES.get(company_id, [company_id])
        if _company_has_core_role(
            aliases=aliases,
            title_compact=title_compact,
            full_compact=full_compact,
            matched_sectors=matched_sectors,
        ):
            return None

    return _result(
        label="irrelevant",
        score=0.25,
        companies=matched_companies,
        sectors=matched_sectors,
        reason=(
            "피어사가 제목/리드문/행위 문맥의 핵심 주체가 아니라 "
            "단순 언급 또는 목록성 언급으로 판단"
        ),
    )


def _weak_company_mention_reject_result(
    *,
    title: str,
    content: str,
    source_type: str | None,
    matched_companies: list[str],
    matched_sectors: list[str],
) -> dict[str, Any] | None:
    if str(source_type or "").strip().lower() != "news":
        return None

    if not matched_companies:
        return None

    title_compact = _compact(title)
    content_compact = _compact(content)

    for company_id in matched_companies:
        aliases = ALL_COMPANY_ALIASES.get(company_id, [company_id])
        company_in_title = any(
            _compact(alias) and _compact(alias) in title_compact for alias in aliases
        )
        if company_in_title:
            return None

        if _alias_mention_count(content_compact, aliases) >= _MIN_PEER_MENTIONS:
            return None

    return _result(
        label="irrelevant",
        score=0.25,
        companies=matched_companies,
        sectors=matched_sectors,
        reason=(
            f"제목에 피어사가 없고 본문/부제목의 피어사 언급이 "
            f"{_MIN_PEER_MENTIONS}회 미만이라 단순 언급으로 판단"
        ),
    )


def _company_has_core_role(
    *,
    aliases: list[str],
    title_compact: str,
    full_compact: str,
    matched_sectors: list[str],
) -> bool:
    has_sector = bool(matched_sectors and matched_sectors != ["other"])

    for alias in aliases:
        alias_compact = _compact(alias)
        if not alias_compact:
            continue

        if (
            alias_compact in title_compact
            and _has_listing_context_near_alias(title_compact, alias_compact)
            and not _has_direct_role_keyword_near_alias(title_compact, alias_compact)
        ):
            continue

        if alias_compact in title_compact and (
            has_sector or _has_direct_role_keyword_near_alias(title_compact, alias_compact)
        ):
            return True

        if (
            _alias_appears_as_subject(full_compact, alias_compact)
            and (
                _has_direct_role_keyword_near_alias(full_compact, alias_compact)
                or (
                    has_sector
                    and _alias_mention_count(full_compact, aliases) >= 2
                    and not _has_listing_context_near_alias(full_compact, alias_compact)
                )
            )
            and not _has_listing_context_near_alias(full_compact, alias_compact)
        ):
            return True

        if _has_direct_role_keyword_near_alias(full_compact, alias_compact):
            return True

    return False


def _alias_appears_as_subject(text_compact: str, alias_compact: str) -> bool:
    return any(f"{alias_compact}{marker}" in text_compact for marker in _SUBJECT_MARKERS)


def _has_action_keyword_near_alias(text_compact: str, alias_compact: str) -> bool:
    action_keywords = [_compact(keyword) for keyword in _STRATEGIC_ACTION_KEYWORDS]
    return _has_keywords_near_alias(
        text_compact,
        alias_compact,
        action_keywords,
        skip_listing_context=True,
    )


def _has_direct_role_keyword_near_alias(text_compact: str, alias_compact: str) -> bool:
    role_keywords = [_compact(keyword) for keyword in _DIRECT_COMPANY_ROLE_KEYWORDS]
    return _has_keywords_near_alias(
        text_compact,
        alias_compact,
        role_keywords,
        skip_listing_context=False,
    )


def _has_keywords_near_alias(
    text_compact: str,
    alias_compact: str,
    keywords: list[str],
    *,
    skip_listing_context: bool,
) -> bool:
    start = 0
    while True:
        pos = text_compact.find(alias_compact, start)
        if pos < 0:
            return False

        left = max(0, pos - _ROLE_CONTEXT_WINDOW)
        right = min(len(text_compact), pos + len(alias_compact) + _ROLE_CONTEXT_WINDOW)
        context = text_compact[left:right]

        if skip_listing_context and any(
            keyword in context for keyword in _LISTING_CONTEXT_KEYWORDS
        ):
            start = pos + len(alias_compact)
            continue

        if any(keyword in context for keyword in keywords):
            return True

        start = pos + len(alias_compact)


def _alias_mention_count(text_compact: str, aliases: list[str]) -> int:
    positions: set[int] = set()
    for alias in aliases:
        alias_compact = _compact(alias)
        if not alias_compact:
            continue
        start = 0
        while True:
            pos = text_compact.find(alias_compact, start)
            if pos < 0:
                break
            positions.add(pos)
            start = pos + len(alias_compact)
    return len(positions)


def _has_listing_context_near_alias(text_compact: str, alias_compact: str) -> bool:
    listing_keywords = [_compact(keyword) for keyword in _MARKET_LISTING_KEYWORDS]

    start = 0
    while True:
        pos = text_compact.find(alias_compact, start)
        if pos < 0:
            return False

        left = max(0, pos - _ROLE_CONTEXT_WINDOW)
        right = min(len(text_compact), pos + len(alias_compact) + _ROLE_CONTEXT_WINDOW)
        context = text_compact[left:right]
        if any(keyword in context for keyword in listing_keywords):
            return True

        start = pos + len(alias_compact)


def _is_market_price_noise(text: str) -> bool:
    return bool(_MARKET_PRICE_RE.search(text) and _MARKET_METRIC_RE.search(text))


def _is_market_listing_noise(*, title: str, content: str) -> bool:
    compact_text = _compact(f"{title} {content}")
    title_compact = _compact(title)
    listing_keywords = [_compact(keyword) for keyword in _MARKET_LISTING_KEYWORDS]

    if any(keyword in title_compact for keyword in listing_keywords):
        return True

    listing_count = sum(1 for keyword in listing_keywords if keyword in compact_text)
    return listing_count >= 2


def _is_non_korean_news_title(*, title: str, source_type: str | None) -> bool:
    if str(source_type or "").strip().lower() != "news":
        return False

    title = title or ""
    if not title.strip():
        return False

    return re.search(r"[가-힣]", title) is None


def _is_event_listing_noise(
    *,
    title: str,
    compact_text: str,
    matched_companies: list[str],
) -> bool:
    has_event_keyword = any(
        _compact(keyword) in compact_text for keyword in _EVENT_LISTING_KEYWORDS
    )
    if not has_event_keyword:
        return False

    title_compact = _compact(title)
    company_in_title = any(
        _compact(alias) in title_compact
        for company_id in matched_companies
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id])
    )
    event_in_title = any(_compact(keyword) in title_compact for keyword in _EVENT_LISTING_KEYWORDS)
    if event_in_title and not company_in_title:
        return True

    has_strong_action_keyword = any(
        _compact(keyword) in compact_text for keyword in _STRONG_STRATEGIC_ACTION_KEYWORDS
    )
    if not company_in_title and not has_strong_action_keyword:
        return True

    has_action_keyword = any(
        _compact(keyword) in compact_text for keyword in _STRATEGIC_ACTION_KEYWORDS
    )
    return not has_action_keyword


def _is_relevant(result: dict[str, Any]) -> bool:
    label = result.get("relevance_label", "irrelevant")
    score = float(result.get("relevance_score", 0.0))
    matched_companies = result.get("matched_companies") or []
    matched_sectors = result.get("matched_sectors") or []
    has_sector = bool(matched_sectors and matched_sectors != ["other"])

    if label == "relevant" and score >= RELEVANCE_THRESHOLD:
        return True

    if (
        label == "uncertain"
        and score >= UNCERTAIN_CANDIDATE_THRESHOLD
        and matched_companies
        and has_sector
    ):
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

    if not (is_industry_bucket or is_companyless_sector or is_multi_peer_sector or is_trend_source):
        return {}

    patch: dict[str, Any] = {
        "topic_scope": "industry_trend",
        "matched_companies": matched_companies,
        "matched_sectors": matched_sectors,
    }

    if is_multi_peer_sector:
        patch["primary_company"] = None

    return patch


def _peer_context_snippets(
    *,
    title: str,
    content: str,
    matched_companies: list[str],
) -> str:
    if not matched_companies:
        return ""

    aliases = [
        _compact(alias)
        for company_id in matched_companies
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id])
        if alias
    ]
    sentences = _split_sentences(content)
    snippets: list[str] = []

    if _sentence_has_alias(title, aliases):
        snippets.append(_shorten(title, 220))

    for index, sentence in enumerate(sentences):
        if not _sentence_has_alias(sentence, aliases):
            continue

        window = sentences[max(0, index - 1) : index + 2]
        snippet = _shorten(" ".join(window), 420)
        if snippet and snippet not in snippets:
            snippets.append(snippet)

        if len(" ".join(snippets)) >= PEER_CONTEXT_LIMIT:
            break

    return "\n".join(f"- {snippet}" for snippet in snippets)[:PEER_CONTEXT_LIMIT]


def _split_sentences(content: str) -> list[str]:
    normalized = " ".join((content or "").split())
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", normalized)
        if sentence.strip()
    ]


def _sentence_has_alias(
    sentence: str,
    aliases: list[str],
) -> bool:
    compact_sentence = _compact(sentence)
    return any(alias and alias in compact_sentence for alias in aliases)


def _match_companies(text_body: str, company: list[str]) -> list[str]:
    matched: list[str] = []

    for company_id in company:
        aliases = ALL_COMPANY_ALIASES.get(company_id, [company_id])
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
        "matched_sectors": _normalize_sectors(sectors),
        "reason": reason,
    }


def _normalize_sectors(sectors: list[str]) -> list[str]:
    allowed = set(SECTOR_IDS)
    normalized = _dedupe_keep_order([str(sector) for sector in sectors if sector in allowed])
    return normalized or ["other"]


def _compact(value: str) -> str:
    return "".join(value.lower().split())


def _shorten(value: str, limit: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= limit:
        return compacted
    return f"{compacted[: limit - 3]}..."


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
