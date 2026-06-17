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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.config.companies import COMPANY_ALIASES, COMPANY_IDS
from src.config.global_companies import GLOBAL_COMPANY_ALIASES
from src.config.openai_policy import relevance_llm_disabled_reason, relevance_llm_enabled
from src.config.preprocessing import (
    COMPANY_SITE_SOURCE_TYPES,
    INDUSTRY_DOCUMENT_SOURCE_TYPES,
    OFFICIAL_SOURCE_TYPES,
    PARSED_DOCUMENT_SOURCE_TYPES,
    STATUS_REVIEW,
    STRUCTURED_SIGNAL_SOURCE_TYPES,
)
from src.config.relevance_policy import (
    DIRECT_COMPANY_ROLE_KEYWORDS,
    EVENT_LISTING_KEYWORDS,
    FAST_PASS_ACTION_KEYWORDS,
    FAST_PASS_SOURCE_TYPES,
    LISTING_CONTEXT_KEYWORDS,
    LOW_VALUE_NEWS_KEYWORDS,
    MARKET_LISTING_KEYWORDS,
    MARKET_METRIC_PATTERN,
    MARKET_PRICE_PATTERN,
    MIN_PEER_MENTIONS,
    PEER_CONTEXT_LIMIT,
    RELEVANCE_THRESHOLD,
    ROLE_CONTEXT_WINDOW,
    STRATEGIC_ACTION_KEYWORDS,
    STRONG_STRATEGIC_ACTION_KEYWORDS,
    SUBJECT_MARKERS,
)
from src.config.sectors import SECTOR_IDS, SectorMatch, match_sector_details, match_sectors
from src.db.article_store import INDUSTRY_TREND_COMPANY
from src.db.postgres import SessionLocal
from src.llm import LLMSpec, build_chat_llm

log = logging.getLogger(__name__)

ALL_COMPANY_ALIASES = {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}
_FAST_PASS_COMPANY_IDS = set(COMPANY_IDS)
_POSCO_DX_GROUP_AX_RE = re.compile(
    r"포스코.{0,80}(AX|AI\s*에이전트|피지컬\s*AI|스마트\s*팩토리|"
    r"자동화|디지털\s*전환|업무\s*혁신|제조\s*AI|수주|영업|품질)"
    r"|"
    r"(AX|AI\s*에이전트|피지컬\s*AI|스마트\s*팩토리|자동화|"
    r"디지털\s*전환|업무\s*혁신|제조\s*AI).{0,80}포스코",
    re.IGNORECASE,
)
_INDUSTRY_TREND_LOW_VALUE_TITLE_TERMS = (
    "에너지밸리포럼",
    "경제발전공유사업",
    "KSP",
    "해외진출 지원",
    "국립대병원",
    "[정치",
    "구윤철",
    "이준석 대표",
    "교육정보화 콘퍼런스",
)
_INDUSTRY_TREND_LOW_VALUE_EVENT_RE = re.compile(
    r"(ISO\s*27001|인증\s*획득|전시회?\s*참가|계약\s*(?:체결|수주)?|정부\s*사업\s*참여|적용\s*완료|표창\s*수상)",
    re.IGNORECASE,
)
_INDUSTRY_TREND_BROAD_ANCHOR_RE = re.compile(
    r"(산업|시장|기업\s*\d+곳|기업\s*보안|공급망\s*(?:공격|보안|위협|로드맵)|"
    r"국가\s*(?:망|SW|소프트웨어)|국제\s*표준|표준화|정책|제도|"
    r"AI\s*(?:경쟁|거버넌스|에이전트|도입|인프라|데이터센터)|"
    r"생성형\s*AI|전력망|데이터\s*제도|N2SF|제로트러스트)",
    re.IGNORECASE,
)

_llm: ChatOpenAI | None = None
_PROMPT_VERSION = "relevance-v1.0"
_LLM_BATCH_SIZE = int(os.getenv("RELEVANCE_LLM_BATCH_SIZE", "20"))
_LLM_MAX_BATCHES_PER_RUN = int(os.getenv("RELEVANCE_LLM_MAX_BATCHES_PER_RUN", "1"))
_LLM_MAX_COMPLETION_TOKENS = int(os.getenv("RELEVANCE_LLM_MAX_COMPLETION_TOKENS", "2000"))
_LLM_ALLOWED_SOURCE_NAMES = {
    name.strip().lower()
    for name in os.getenv("RELEVANCE_LLM_SOURCE_NAMES", "naver_news").split(",")
    if name.strip()
}


def _get_llm() -> ChatOpenAI:
    global _llm

    if _llm is None:
        _llm = build_chat_llm(
            LLMSpec(model="gpt-4o", temperature=0.1, max_tokens=_LLM_MAX_COMPLETION_TOKENS)
        )

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
- relevance_label은 relevant, irrelevant 중 하나만 사용하세요.
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
  "relevance_label": "relevant|irrelevant",
  "relevance_score": 0.0,
  "matched_companies": ["string"],
  "matched_sectors": ["string"],
  "reason": "1문장 근거"
}}
"""


class RelevanceEvaluator:
    """company/sector 관점의 내용 기반 관련성을 판단한다."""

    def __init__(self, *, enable_llm: bool = True, llm_batch_size: int = _LLM_BATCH_SIZE) -> None:
        self.enable_llm = enable_llm
        self.llm_batch_size = max(1, llm_batch_size)
        self.review_ids: list[int] = []

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
        self.review_ids = []

        log.info("Gate 2.5 관련성 전처리 시작 | total=%d", len(raw_article_ids))

        llm_batches_used = 0
        llm_batch_cap = max(0, _LLM_MAX_BATCHES_PER_RUN)

        with SessionLocal() as db:
            rows = db.execute(
                text("""
                    SELECT
                        ra.id,
                        ra.title,
                        ra.content,
                        ra.company,
                        ra.source_type,
                        ra.source_name,
                        ra.crawl_status,
                        COALESCE(mu.metadata, '{}'::jsonb) AS metadata
                    FROM raw_articles ra
                    LEFT JOIN raw_article_metadata_unified mu
                        ON mu.raw_article_id = ra.id
                    WHERE ra.id = ANY(:ids)
                """),
                {"ids": raw_article_ids},
            ).fetchall()
            rows = sorted(rows, key=lambda row: int(row.id))
            db.commit()

            pending_llm: list[tuple[Any, dict[str, Any]]] = []

            def apply_result(row: Any, result: dict[str, Any]) -> None:
                is_relevant = _is_relevant(result)
                needs_review = bool(result.pop("_needs_review", False))
                result["matched_sector_details"] = _matched_sector_details_for_result(
                    row,
                    result["matched_sectors"],
                )
                metadata_patch = _metadata_patch_for_relevance(row, result, is_relevant)
                if needs_review:
                    metadata_patch.pop("skip_reason", None)
                    metadata_patch.update(
                        {
                            "status_detail": "relevance_review",
                            "review_reason": result["reason"],
                            "decision_code": result.get("decision_code", "needs_llm_review"),
                        }
                    )

                db.execute(
                    text("""
                        UPDATE raw_articles
                        SET relevance_score = :relevance_score,
                            relevance_label = :relevance_label,
                            relevance_reason = :relevance_reason,
                            matched_companies = CAST(:matched_companies AS jsonb),
                            matched_sectors = CAST(:matched_sectors AS jsonb),
                            matched_sector_details = CAST(:matched_sector_details AS jsonb),
                            processing_status = CASE
                                WHEN :is_relevant THEN processing_status
                                WHEN :needs_review THEN :review_status
                                ELSE 'SKIPPED'
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
                        "matched_sector_details": json.dumps(
                            result["matched_sector_details"],
                            ensure_ascii=False,
                        ),
                        "is_relevant": is_relevant,
                        "needs_review": needs_review,
                        "review_status": STATUS_REVIEW,
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
                db.commit()

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
                elif needs_review:
                    self.review_ids.append(row.id)
                    log.info(
                        "관련성 REVIEW 보류 | id=%d label=%s score=%.2f reason=%s",
                        row.id,
                        result["relevance_label"],
                        result["relevance_score"],
                        result["reason"],
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

            def flush_pending() -> None:
                nonlocal llm_batches_used
                if not pending_llm:
                    return
                batch = pending_llm[:]
                pending_llm.clear()
                if llm_batch_cap and llm_batches_used >= llm_batch_cap:
                    log.warning(
                        "Gate 2.5 LLM batch cap 도달 | cap=%d pending=%d",
                        llm_batch_cap,
                        len(batch),
                    )
                    for row, result in self._llm_cap_fallback(batch):
                        apply_result(row, result)
                    return

                llm_batches_used += 1
                for row, result in self._analyze_batch_with_llm(batch):
                    apply_result(row, result)

            for row in rows:
                log.info(
                    "Gate 2.5 기사 전처리 중 | id=%s source_type=%s title=%s",
                    row.id,
                    row.source_type,
                    _shorten(row.title or "", 80),
                )
                result = self._analyze(row)
                if result.pop("_llm_pending", False):
                    pending_llm.append((row, result))
                    if len(pending_llm) >= self.llm_batch_size:
                        flush_pending()
                    continue

                apply_result(row, result)

            flush_pending()

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
        if (
            "posco_dx" in company
            and _has_posco_dx_primary_context(title=title, content=content)
            and _matches_posco_dx_group_ax_signal(text_body)
        ):
            matched_company_candidates = _dedupe_keep_order(
                [*matched_company_candidates, "posco_dx"]
            )
        matched_sector_candidates = match_sectors(text_body)
        if _is_industry_trend_news(row, company):
            noise_reason = _industry_trend_noise_reason(title=title, content=analysis_content)
            if noise_reason:
                return _result(
                    label="irrelevant",
                    score=0.0,
                    companies=[],
                    sectors=[],
                    reason=noise_reason,
                )
            sectors = [sector for sector in matched_sector_candidates if sector != "other"]
            if not sectors:
                sectors = _industry_metadata_sectors(metadata)
            return _result(
                label="relevant",
                score=0.82,
                companies=[],
                sectors=sectors or ["other"],
                reason=(
                    "industry fast-pass: naver_industry_news/industry_trend로 수집된 "
                    "산업 방향성 기사"
                ),
            )

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

        # "피어사 언급 횟수 부족 / 핵심성 부족" 규칙은 명백한 노이즈가 아니라 "약한
        # 관련성 의심"이라 false negative 위험이 크다(예: 키워드 사전에 없는 출시·동맹
        # 표현). LLM 이 켜진 경로에서는 하드 reject 하지 않고 판단을 LLM 으로 위임한다
        # ── 명백한 건 이어지는 fast-pass 로 LLM 없이 통과하고, 애매한 건 LLM batch 로
        # 내려간다. LLM 이 없는 경로(track b/c/d 등)에서만 규칙으로 보수적으로 reject.
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

        if not self.enable_llm:
            mention_result = _weak_company_mention_reject_result(
                title=title,
                content=analysis_content,
                source_type=row.source_type,
                matched_companies=matched_company_candidates,
                matched_sectors=matched_sector_candidates,
            )
            if mention_result is not None:
                log.info(
                    "Gate 2.5 피어사 언급 횟수 부족 제외(LLM 비활성) | id=%s reason=%s",
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
                    "Gate 2.5 피어사 핵심성 부족 제외(LLM 비활성) | id=%s reason=%s",
                    getattr(row, "id", None),
                    role_result["reason"],
                )
                return role_result

            return _review_result(
                label="irrelevant",
                score=0.35,
                companies=matched_company_candidates,
                sectors=matched_sector_candidates,
                reason="LLM 비활성화로 규칙 확정 불가: REVIEW 보류",
                decision_code="llm_disabled_needs_review",
            )

        role_result = _core_company_role_reject_result(
            title=title,
            content=analysis_content,
            source_type=row.source_type,
            matched_companies=matched_company_candidates,
            matched_sectors=matched_sector_candidates,
        )
        if role_result is not None:
            if _should_defer_role_reject_to_llm(
                title=title,
                content=analysis_content,
                source_type=row.source_type,
                matched_companies=matched_company_candidates,
                matched_sectors=matched_sector_candidates,
            ):
                log.info(
                    "Gate 2.5 피어사 핵심성 LLM 위임 | id=%s reason=%s",
                    getattr(row, "id", None),
                    role_result["reason"],
                )
            else:
                log.info(
                    "Gate 2.5 피어사 핵심성 부족 제외 | id=%s reason=%s",
                    getattr(row, "id", None),
                    role_result["reason"],
                )
                return role_result

        source_name = str(_row_value(row, "source_name", "") or "").strip().lower()
        if source_name not in _LLM_ALLOWED_SOURCE_NAMES:
            source_display = source_name or "unknown"
            log.info(
                "Gate 2.5 LLM 제한 소스 REVIEW 보류 | id=%s source=%s",
                getattr(row, "id", None),
                source_display,
            )
            return _review_result(
                label="irrelevant",
                score=0.35,
                companies=matched_company_candidates,
                sectors=matched_sector_candidates,
                reason=(
                    f"LLM relevance 제한 소스지만 규칙 확정 불가: REVIEW 보류 ({source_display})"
                ),
                decision_code="llm_source_not_allowed_needs_review",
            )

        return {
            "_llm_pending": True,
            "relevance_label": "irrelevant",
            "relevance_score": 0.35,
            "matched_companies": matched_company_candidates,
            "matched_sectors": matched_sector_candidates,
            "reason": "LLM batch 판단 대기",
            "_llm_payload": {
                "id": int(row.id),
                "title": title,
                "content": analysis_content[:2200],
                "source_type": row.source_type,
                "source_name": _row_value(row, "source_name", ""),
                "company": company,
                "matched_company_candidates": matched_company_candidates,
                "matched_sector_candidates": matched_sector_candidates,
                "peer_context": _peer_context_snippets(
                    title=title,
                    content=analysis_content,
                    matched_companies=matched_company_candidates,
                )[:PEER_CONTEXT_LIMIT],
            },
        }

    def _analyze_batch_with_llm(
        self,
        pending: list[tuple[Any, dict[str, Any]]],
    ) -> list[tuple[Any, dict[str, Any]]]:
        fallback_results = [
            (
                row,
                _review_result(
                    label="irrelevant",
                    score=0.35,
                    companies=result["matched_companies"],
                    sectors=result["matched_sectors"],
                    reason="LLM batch 판단 실패 또는 비활성화: REVIEW 보류",
                    decision_code="llm_failed_needs_review",
                ),
            )
            for row, result in pending
        ]
        if not pending:
            return []
        if not relevance_llm_enabled():
            log.warning("Gate 2.5 LLM batch 스킵 | reason=%s", relevance_llm_disabled_reason())
            return fallback_results

        payloads = [result["_llm_payload"] for _, result in pending]
        prompt = _build_batch_relevance_prompt(payloads)

        try:
            log.info(
                "Gate 2.5 LLM batch 내용 분석 중 | articles=%d ids=%s",
                len(payloads),
                [item["id"] for item in payloads],
            )
            from src.observability import tracing_config

            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="RelevanceEvaluator",
                    prompt_version=f"{_PROMPT_VERSION}-batch",
                    batch_size=len(payloads),
                ),
            )
            response_text = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            results_by_id = _parse_batch_relevance_json(response_text)
            resolved: list[tuple[Any, dict[str, Any]]] = []
            for row, pending_result in pending:
                parsed = results_by_id.get(int(row.id))
                if parsed is None:
                    resolved.append(
                        (
                            row,
                            _review_result(
                                label="irrelevant",
                                score=0.35,
                                companies=pending_result["matched_companies"],
                                sectors=pending_result["matched_sectors"],
                                reason="LLM batch 응답에 해당 id 없음: REVIEW 보류",
                                decision_code="llm_missing_id_needs_review",
                            ),
                        )
                    )
                else:
                    resolved.append((row, parsed))
            log.info(
                "Gate 2.5 LLM batch 판단 완료 | articles=%d relevant=%d",
                len(resolved),
                sum(1 for _, result in resolved if _is_relevant(result)),
            )
            return resolved
        except Exception as e:
            log.warning("관련성 LLM batch 판단 실패 | error=%s", e)
            return fallback_results

    def _llm_cap_fallback(
        self,
        pending: list[tuple[Any, dict[str, Any]]],
    ) -> list[tuple[Any, dict[str, Any]]]:
        return [
            (
                row,
                _review_result(
                    label="irrelevant",
                    score=0.35,
                    companies=result["matched_companies"],
                    sectors=result["matched_sectors"],
                    reason="LLM relevance batch cap 초과: REVIEW 보류",
                    decision_code="llm_cap_needs_review",
                ),
            )
            for row, result in pending
        ]


def analyze_relevance_article(article: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """JSON article에 Gate 2.5 관련성 결과를 붙인다."""

    row = _DictRow(article)
    result = RelevanceEvaluator(enable_llm=False)._analyze(row)
    if result.pop("_llm_pending", False):
        result = _review_result(
            label="irrelevant",
            score=0.35,
            companies=result["matched_companies"],
            sectors=result["matched_sectors"],
            reason="LLM 판단 대기 후보는 로컬 relevance helper에서 REVIEW 보류",
            decision_code="local_helper_needs_review",
        )
    needs_review = bool(result.pop("_needs_review", False))
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
        if needs_review:
            item["processing_status"] = STATUS_REVIEW
            item["status_detail"] = "relevance_review"
            item["review_reason"] = result["reason"]
            item["decision_code"] = result.get("decision_code", "needs_llm_review")
        else:
            item["processing_status"] = "SKIPPED"
            item["skip_reason"] = result["reason"]

    return item, is_relevant


class _DictRow:
    def __init__(self, article: dict[str, Any]) -> None:
        self.id = article.get("id")
        self.title = article.get("title")
        self.content = article.get("content")
        self.company = article.get("company")
        self.source_type = article.get("source_type")
        self.source_name = article.get("source_name")
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


def _is_industry_trend_news(row: Any, company: list[str]) -> bool:
    if INDUSTRY_TREND_COMPANY in company:
        return True

    source_name = str(_row_value(row, "source_name", "") or "").strip().lower()
    if source_name == "naver_industry_news":
        return True

    metadata = _metadata_dict(_row_value(row, "metadata", {}))
    return metadata.get("topic_scope") == "industry_trend"


def _industry_trend_noise_reason(*, title: str, content: str) -> str:
    text = _join_text(title, content)
    if any(term in title for term in _INDUSTRY_TREND_LOW_VALUE_TITLE_TERMS):
        return "industry trend 제외: 산업 방향성보다 행사/정치/외곽 정책 단신 성격이 강함"
    if re.search(r"ISO\s*27001", title, re.IGNORECASE) and re.search(r"획득", title):
        return "industry trend 제외: 개별 기업 인증 획득 단신 성격이 강함"
    if re.search(
        r"(인증\s*획득|보안인증|전시회?\s*참가|계약\s*(?:체결|수주)?|정부\s*사업\s*참여|적용\s*완료|표창\s*수상)",
        title,
    ):
        return "industry trend 제외: 개별 기업 이벤트 단신 성격이 강함"
    if _INDUSTRY_TREND_LOW_VALUE_EVENT_RE.search(title) and not (
        _INDUSTRY_TREND_BROAD_ANCHOR_RE.search(text)
    ):
        return "industry trend 제외: 산업 anchor가 약한 인증·계약·전시 참가 단신 성격이 강함"
    if "병원" in title and not re.search(r"생성형\s*AI|AI\s*도입|인프라|데이터", text):
        return "industry trend 제외: 의료기관 단신으로 IT 산업 방향성 근거가 약함"
    return ""


def _industry_metadata_sectors(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("matched_sectors") or []
    if isinstance(values, list):
        return [str(value) for value in values if value and str(value) != "other"]

    value = metadata.get("sector")
    return [str(value)] if value and str(value) != "other" else []


def _precheck(
    company: list[str],
    matched_companies: list[str],
    matched_sectors: list[str],
    source_type: str | None,
) -> dict[str, Any]:
    has_company = bool(matched_companies)
    has_sector = bool(matched_sectors and matched_sectors != ["other"])
    source_type_value = str(source_type or "").strip().lower()
    is_company_source = source_type_value in {
        *PARSED_DOCUMENT_SOURCE_TYPES,
        *OFFICIAL_SOURCE_TYPES,
        *COMPANY_SITE_SOURCE_TYPES,
    } and bool(company)

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


_MARKET_PRICE_RE = re.compile(MARKET_PRICE_PATTERN)
_MARKET_METRIC_RE = re.compile(MARKET_METRIC_PATTERN)
_PURE_MARKET_TITLE_RE = re.compile(
    r"(주가|종가|장중).{0,40}(\d+(?:\.\d+)?\s*%|\d{1,3}(?:,\d{3})+\s*원).{0,20}"
    r"(상승|하락|급등|급락|강세|약세|상한가|하한가)"
)
_EVENT_DRIVEN_MARKET_KEYWORDS = [
    "수혜",
    "기대",
    "기대감",
    "협력",
    "협업",
    "계약",
    "수주",
    "공급계약",
    "업무협약",
    "제휴",
    "투자",
    "인수",
    "합병",
    "실적",
    "공시",
    "목표가",
    "리포트",
    "증권사",
    "데이터센터",
    "AI 데이터센터",
    "AI 인프라",
    "공공 AI",
    "AX",
    "피지컬AI",
    "피지컬 AI",
    "로봇",
    "로보틱스",
    "자율용접",
    "두나무",
]
_FINANCIAL_THEME_NOISE_KEYWORDS = [
    "두나무",
    "코인원",
    "코빗",
    "가상자산",
    "디지털자산",
    "거래소",
    "지분 확보",
    "지분 인수",
    "토큰증권",
    "sto",
    "온체인",
    "스테이블코인",
]
_WEAK_PEER_CONTEXT_NOISE_KEYWORDS = [
    "ipo",
    "상장",
    "지배구조",
    "실탄",
    "그룹",
    "주가부양",
    "주가 부양",
    "레퍼런스",
    "rfm",
    "보스턴다이나믹스",
    "보스턴 다이나믹스",
]
_OPERATIONAL_CAMPAIGN_NOISE_KEYWORDS = (
    "차량 5부제",
    "5부제",
    "에너지 절약",
    "에너지절약",
    "에너지 절감",
    "에너지절감",
    "절전",
    "캠페인",
    "동참",
)
_BUSINESS_EVENT_TITLE_KEYWORDS = (
    "수주",
    "계약",
    "공급",
    "협약",
    "제휴",
    "맞손",
    "투자",
    "인수",
    "합병",
    "출시",
    "공개",
    "도입",
    "구축",
    "선정",
    "개발",
)
_EXECUTIVE_ROLE_KEYWORDS = (
    "대표",
    "대표이사",
    "사장",
    "부사장",
    "전무",
    "상무",
    "임원",
    "본부장",
    "센터장",
    "실장",
    "CTO",
    "CIO",
    "CISO",
)
_EXECUTIVE_STRATEGY_SIGNAL_KEYWORDS = (
    "발제",
    "발표",
    "강연",
    "기조연설",
    "밝혔",
    "말했",
    "강조",
    "제언",
    "진단",
    "전략",
    "경쟁",
    "컨퍼런스",
    "세미나",
    "포럼",
    "M.AX",
    "AX",
    "AI",
    "제조AI",
    "제조AX",
    "피지컬AI",
    "데이터",
    "시계열",
    "인프라",
    "클라우드",
    "스마트팩토리",
)
_EXECUTIVE_SOURCE_CONTEXT_BLOCKERS = ("출신", "전직", "전임", "前")


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
    if _is_low_value_news_noise(title=title, content=content) and not has_peer_strategy_signal:
        return _result(
            label="irrelevant",
            score=0.20,
            companies=matched_companies,
            sectors=matched_sectors,
            reason="뉴스브리핑·교육/멘토링·일반 시황성 기사로 피어사 전략 동향 신호가 약해 제외",
        )

    if _is_roundup_news_title(title):
        return _result(
            label="irrelevant",
            score=0.25,
            companies=matched_companies,
            sectors=matched_sectors,
            reason=(
                "여러 기업 소식을 묶은 섹션형/브리핑형 기사라 개별 피어사 전략 이벤트 근거에서 제외"
            ),
        )

    if _is_financial_theme_noise(title=title, content=content, matched_companies=matched_companies):
        return _result(
            label="irrelevant",
            score=0.25,
            companies=matched_companies,
            sectors=matched_sectors,
            reason=(
                "가상자산·금융권 지분 경쟁 테마 기사에서 피어사가 핵심 주체로 드러나지 않아 제외"
            ),
        )

    if _is_weak_peer_context_noise(
        title=title,
        content=content,
        matched_companies=matched_companies,
    ):
        return _result(
            label="irrelevant",
            score=0.25,
            companies=matched_companies,
            sectors=matched_sectors,
            reason="피어사가 제목의 핵심 주체가 아니고 그룹/주가/레퍼런스 맥락에 그쳐 제외",
        )

    if _is_operational_campaign_noise(title=title, matched_companies=matched_companies):
        return _result(
            label="irrelevant",
            score=0.25,
            companies=matched_companies,
            sectors=matched_sectors,
            reason=("에너지 절감·캠페인 등 그룹 운영성 기사라 피어사 사업 이벤트 근거가 약해 제외"),
        )

    has_market_listing_noise = _is_market_listing_noise(title=title, content=content)
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

    if _is_vague_market_reaction_title(title):
        return _result(
            label="irrelevant",
            score=0.25,
            companies=matched_companies,
            sectors=matched_sectors,
            reason="구체적 사업 이벤트 없이 주가 반응만 다루는 기사라 제외",
        )

    if _is_market_price_noise(text) and is_company_market_signal:
        if _has_event_driven_market_signal(title=title, content=content):
            return None
        return _result(
            label="irrelevant",
            score=0.25,
            companies=matched_companies,
            sectors=matched_sectors,
            reason="순수 장중 가격·등락 기사라 전략 이벤트 근거가 약해 제외",
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


def _has_event_driven_market_signal(*, title: str, content: str) -> bool:
    if _is_vague_market_reaction_title(title):
        return False

    compact_text = _compact(f"{title} {content}")
    has_event_keyword = any(
        _compact(keyword) and _compact(keyword) in compact_text
        for keyword in _EVENT_DRIVEN_MARKET_KEYWORDS
    )
    if not has_event_keyword:
        return False

    title_compact = _compact(title)
    if _PURE_MARKET_TITLE_RE.search(title) and not any(
        _compact(keyword) and _compact(keyword) in title_compact
        for keyword in _EVENT_DRIVEN_MARKET_KEYWORDS
    ):
        return False

    return True


def _is_vague_market_reaction_title(title: str) -> bool:
    title_compact = _compact(title)
    return (
        "주가" in title_compact
        and any(keyword in title_compact for keyword in ("급등세", "기세등등", "왜"))
        and not any(keyword in title_compact for keyword in ("계약", "수주", "협약", "제휴"))
    )


def _is_financial_theme_noise(
    *,
    title: str,
    content: str,
    matched_companies: list[str],
) -> bool:
    compact_text = _compact(f"{title} {content}")
    if not any(_compact(keyword) in compact_text for keyword in _FINANCIAL_THEME_NOISE_KEYWORDS):
        return False
    return not _company_in_title(title=title, matched_companies=matched_companies)


def _is_weak_peer_context_noise(
    *,
    title: str,
    content: str,
    matched_companies: list[str],
) -> bool:
    if _company_in_title(title=title, matched_companies=matched_companies):
        return False

    title_compact = _compact(title)
    if not any(_compact(keyword) in title_compact for keyword in _WEAK_PEER_CONTEXT_NOISE_KEYWORDS):
        return False

    compact_text = _compact(f"{title} {content}")
    return not any(
        (
            _alias_appears_as_deal_counterparty(compact_text, _compact(alias))
            or _alias_appears_as_subject(compact_text, _compact(alias))
        )
        and not _has_source_only_context_near_alias(compact_text, _compact(alias))
        for company_id in matched_companies
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id])
        if _compact(alias)
    )


def _is_operational_campaign_noise(*, title: str, matched_companies: list[str]) -> bool:
    if not matched_companies:
        return False

    title_compact = _compact(title)
    if not title_compact:
        return False

    has_campaign_context = any(
        _compact(keyword) and _compact(keyword) in title_compact
        for keyword in _OPERATIONAL_CAMPAIGN_NOISE_KEYWORDS
    )
    if not has_campaign_context:
        return False

    return not any(
        _compact(keyword) and _compact(keyword) in title_compact
        for keyword in _BUSINESS_EVENT_TITLE_KEYWORDS
    )


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
    has_action = any(_compact(keyword) in text_compact for keyword in STRATEGIC_ACTION_KEYWORDS)
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
    if str(source_type or "").strip().lower() not in FAST_PASS_SOURCE_TYPES:
        return None

    if not matched_companies:
        return None

    if not any(company_id in _FAST_PASS_COMPANY_IDS for company_id in matched_companies):
        return None

    if (
        "posco_dx" in matched_companies
        and _has_posco_dx_primary_context(title=title, content=content)
        and _matches_posco_dx_group_ax_signal(_join_text(title, content))
    ):
        return _result(
            label="relevant",
            score=0.78,
            companies=matched_companies,
            sectors=matched_sectors,
            reason=(
                "fast-pass: 포스코그룹 AX/AI 업무혁신 신호가 포스코DX 관찰 축과 "
                "직접 연결되어 관련 기사로 판단"
            ),
        )

    title_compact = _compact(title)
    text_compact = _compact(f"{title} {content}")

    company_in_title = any(
        _compact(alias) in title_compact
        for company_id in matched_companies
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id])
    )
    if not company_in_title:
        return None

    has_sector = bool(matched_sectors and matched_sectors != ["other"])
    if not has_sector:
        return None

    has_executive_strategy_role = any(
        _has_executive_strategy_signal_near_alias(text_compact, _compact(alias), matched_sectors)
        for company_id in matched_companies
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id])
        if _compact(alias)
    )
    if has_executive_strategy_role:
        return _result(
            label="relevant",
            score=0.76,
            companies=matched_companies,
            sectors=matched_sectors,
            reason=(
                "fast-pass: 피어사 임원이 섹터 전략/기술 맥락에서 발언한 기사로 "
                "분석 가치가 있어 관련 기사로 판단"
            ),
        )

    has_strong_action = any(
        _compact(keyword) in title_compact for keyword in FAST_PASS_ACTION_KEYWORDS
    )
    if not has_strong_action:
        return None

    has_direct_role = any(
        _has_direct_role_keyword_near_alias(text_compact, _compact(alias))
        for company_id in matched_companies
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id])
        if _compact(alias)
    )
    if not has_direct_role:
        return None

    return _result(
        label="relevant",
        score=0.82,
        companies=matched_companies,
        sectors=matched_sectors,
        reason=(
            "fast-pass: 제목에 피어사/핵심 이벤트가 있고, sector와 직접 역할 근거가 있어 "
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

    if _has_title_company_sector_candidate(
        title_compact=title_compact,
        matched_companies=matched_companies,
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

        if _alias_mention_count(content_compact, aliases) >= MIN_PEER_MENTIONS:
            return None

    return _result(
        label="irrelevant",
        score=0.25,
        companies=matched_companies,
        sectors=matched_sectors,
        reason=(
            f"제목에 피어사가 없고 본문/부제목의 피어사 언급이 "
            f"{MIN_PEER_MENTIONS}회 미만이라 단순 언급으로 판단"
        ),
    )


def _company_has_core_role(
    *,
    aliases: list[str],
    title_compact: str,
    full_compact: str,
    matched_sectors: list[str],
) -> bool:
    for alias in aliases:
        alias_compact = _compact(alias)
        if not alias_compact:
            continue

        if alias_compact in title_compact and _has_executive_strategy_signal_near_alias(
            full_compact,
            alias_compact,
            matched_sectors,
        ):
            return True

        if (
            alias_compact in title_compact
            and _has_listing_context_near_alias(title_compact, alias_compact)
            and not _has_direct_role_keyword_near_alias(title_compact, alias_compact)
        ):
            continue

        if alias_compact in title_compact and _has_source_only_context_near_alias(
            title_compact, alias_compact
        ):
            continue

        if alias_compact in title_compact and (
            _has_direct_role_keyword_near_alias(title_compact, alias_compact)
            or _has_action_keyword_near_alias(title_compact, alias_compact)
        ):
            return True

        if _alias_appears_as_subject(title_compact, alias_compact) and (
            _has_direct_role_keyword_near_alias(title_compact, alias_compact)
            or _has_action_keyword_near_alias(title_compact, alias_compact)
        ):
            return True

        if _alias_appears_as_deal_counterparty(title_compact, alias_compact):
            return True

        if (
            _alias_appears_as_subject(full_compact, alias_compact)
            and _has_direct_role_keyword_near_alias(full_compact, alias_compact)
            and not _has_listing_context_near_alias(full_compact, alias_compact)
            and not _has_source_only_context_near_alias(full_compact, alias_compact)
        ):
            return True

    return False


def _has_title_company_sector_candidate(
    *,
    title_compact: str,
    matched_companies: list[str],
    matched_sectors: list[str],
) -> bool:
    if not matched_sectors or matched_sectors == ["other"]:
        return False

    for company_id in matched_companies:
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id]):
            alias_compact = _compact(alias)
            if not alias_compact or alias_compact not in title_compact:
                continue
            if _has_listing_context_near_alias(title_compact, alias_compact):
                continue
            if _has_source_only_context_near_alias(title_compact, alias_compact):
                continue
            return True

    return False


def _should_defer_role_reject_to_llm(
    *,
    title: str,
    content: str,
    source_type: str | None,
    matched_companies: list[str],
    matched_sectors: list[str],
) -> bool:
    if str(source_type or "").strip().lower() != "news":
        return False
    if not matched_companies or not matched_sectors or matched_sectors == ["other"]:
        return False

    title_compact = _compact(title)
    full_compact = _compact(f"{title} {content}")
    has_action_signal = any(
        keyword and keyword in full_compact
        for keyword in (
            [_compact(item) for item in STRATEGIC_ACTION_KEYWORDS]
            + [_compact(item) for item in FAST_PASS_ACTION_KEYWORDS]
            + [
                "플랫폼",
                "솔루션",
                "로드맵",
                "전략공개",
                "전환전략",
                "고도화",
                "자동화",
                "상용화",
                "리셀러",
                "판매권",
            ]
        )
    )
    if not has_action_signal:
        return False

    for company_id in matched_companies:
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id]):
            alias_compact = _compact(alias)
            if not alias_compact or alias_compact not in title_compact:
                continue
            if _has_listing_context_near_alias(title_compact, alias_compact):
                continue
            if _has_source_only_context_near_alias(title_compact, alias_compact):
                continue
            return True

    return False


def _alias_appears_as_subject(text_compact: str, alias_compact: str) -> bool:
    return any(f"{alias_compact}{marker}" in text_compact for marker in SUBJECT_MARKERS)


def _alias_appears_as_deal_counterparty(text_compact: str, alias_compact: str) -> bool:
    if not any(
        marker in text_compact
        for marker in (
            f"{alias_compact}와",
            f"{alias_compact}과",
            f"{alias_compact}와의",
            f"{alias_compact}과의",
        )
    ):
        return False

    deal_keywords = (
        "계약",
        "수주",
        "공급계약",
        "사업수주",
        "업무협약",
        "협약",
        "협력",
        "협업",
        "공동개발",
        "맞손",
        "동맹",
        "체결",
    )
    return any(keyword in text_compact for keyword in deal_keywords)


def _has_source_only_context_near_alias(text_compact: str, alias_compact: str) -> bool:
    source_only_keywords = (
        "출신",
        "前",
        "전직",
        "전임",
        "상무",
        "전무",
        "부사장",
        "계열사",
        "관련계열사",
        "언급",
        "대표내정",
        "신임대표",
        "대표이사내정",
    )
    start = 0
    while True:
        pos = text_compact.find(alias_compact, start)
        if pos < 0:
            return False

        left = max(0, pos - ROLE_CONTEXT_WINDOW)
        right = min(len(text_compact), pos + len(alias_compact) + ROLE_CONTEXT_WINDOW)
        context = text_compact[left:right]
        if any(_compact(keyword) in context for keyword in source_only_keywords):
            if _has_executive_strategy_context(context):
                start = pos + len(alias_compact)
                continue
            return True

        start = pos + len(alias_compact)


def _has_executive_strategy_signal_near_alias(
    text_compact: str,
    alias_compact: str,
    matched_sectors: list[str],
) -> bool:
    if not alias_compact or not matched_sectors or matched_sectors == ["other"]:
        return False

    start = 0
    while True:
        pos = text_compact.find(alias_compact, start)
        if pos < 0:
            return False

        left = max(0, pos - ROLE_CONTEXT_WINDOW)
        right = min(len(text_compact), pos + len(alias_compact) + ROLE_CONTEXT_WINDOW)
        if _has_executive_strategy_context(text_compact[left:right]):
            return True

        start = pos + len(alias_compact)


def _has_executive_strategy_context(context_compact: str) -> bool:
    if any(_compact(keyword) in context_compact for keyword in _EXECUTIVE_SOURCE_CONTEXT_BLOCKERS):
        return False

    has_role_keyword = any(
        _compact(keyword) in context_compact for keyword in _EXECUTIVE_ROLE_KEYWORDS
    )
    has_strategy_keyword = any(
        _compact(keyword) in context_compact for keyword in _EXECUTIVE_STRATEGY_SIGNAL_KEYWORDS
    )
    return has_role_keyword and has_strategy_keyword


def _has_action_keyword_near_alias(text_compact: str, alias_compact: str) -> bool:
    action_keywords = [_compact(keyword) for keyword in STRATEGIC_ACTION_KEYWORDS]
    return _has_keywords_near_alias(
        text_compact,
        alias_compact,
        action_keywords,
        skip_listing_context=True,
    )


def _has_direct_role_keyword_near_alias(text_compact: str, alias_compact: str) -> bool:
    role_keywords = [_compact(keyword) for keyword in DIRECT_COMPANY_ROLE_KEYWORDS]
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

        left = max(0, pos - ROLE_CONTEXT_WINDOW)
        right = min(len(text_compact), pos + len(alias_compact) + ROLE_CONTEXT_WINDOW)
        context = text_compact[left:right]

        if skip_listing_context and any(keyword in context for keyword in LISTING_CONTEXT_KEYWORDS):
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
    listing_keywords = [_compact(keyword) for keyword in MARKET_LISTING_KEYWORDS]

    start = 0
    while True:
        pos = text_compact.find(alias_compact, start)
        if pos < 0:
            return False

        left = max(0, pos - ROLE_CONTEXT_WINDOW)
        right = min(len(text_compact), pos + len(alias_compact) + ROLE_CONTEXT_WINDOW)
        context = text_compact[left:right]
        if any(keyword in context for keyword in listing_keywords):
            return True

        start = pos + len(alias_compact)


def _is_market_price_noise(text: str) -> bool:
    return bool(_MARKET_PRICE_RE.search(text) and _MARKET_METRIC_RE.search(text))


def _is_market_listing_noise(*, title: str, content: str) -> bool:
    compact_text = _compact(f"{title} {content}")
    title_compact = _compact(title)
    listing_keywords = [_compact(keyword) for keyword in MARKET_LISTING_KEYWORDS]

    if any(keyword in title_compact for keyword in listing_keywords):
        return True

    listing_count = sum(1 for keyword in listing_keywords if keyword in compact_text)
    return listing_count >= 2


def _is_low_value_news_noise(*, title: str, content: str) -> bool:
    title_compact = _compact(title)
    compact_text = _compact(f"{title} {content}")
    low_value_keywords = [_compact(keyword) for keyword in LOW_VALUE_NEWS_KEYWORDS]
    if any(keyword and keyword in title_compact for keyword in low_value_keywords):
        return True

    market_keyword_count = sum(
        1
        for keyword in (_compact(item) for item in MARKET_LISTING_KEYWORDS)
        if keyword and keyword in compact_text
    )
    if market_keyword_count >= 3:
        return True

    return False


def _is_roundup_news_title(title: str) -> bool:
    compact_title = _compact_for_title_marker(title)
    if not compact_title:
        return False

    markers = (
        "뉴스브리프",
        "뉴스브리핑",
        "ai브리프",
        "it브리프",
        "it스냅샷",
        "전자it레이더",
        "시큐리티포커스",
        "테크앤나우",
        "technow",
        "클라우드월드",
    )
    if any(marker in compact_title for marker in markers):
        return True

    if _is_bracketed_multi_item_listing_title(title):
        return True

    return bool(re.match(r"^\[?#?[가-힣a-z0-9]*(?:포커스|레이더|브리프|스냅샷)\]?", compact_title))


def _compact_for_title_marker(value: str) -> str:
    compacted = _compact(value)
    return re.sub(r"[^0-9a-z가-힣]", "", compacted)


def _is_bracketed_multi_item_listing_title(title: str) -> bool:
    match = re.match(r"^\[[^\]]{1,18}\]\s*(.+)$", title.strip())
    if not match:
        return False

    body = match.group(1).strip()
    if not body:
        return False

    if any(_compact(keyword) in _compact(body) for keyword in STRATEGIC_ACTION_KEYWORDS):
        return False

    items = [item.strip() for item in re.split(r"[·ㆍ,]", body) if item.strip()]
    if len(items) < 3:
        return False

    short_item_count = sum(1 for item in items if len(item) <= 16)
    return short_item_count >= 3


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
    has_event_keyword = any(_compact(keyword) in compact_text for keyword in EVENT_LISTING_KEYWORDS)
    if not has_event_keyword:
        return False

    title_compact = _compact(title)
    company_in_title = any(
        _compact(alias) in title_compact
        for company_id in matched_companies
        for alias in ALL_COMPANY_ALIASES.get(company_id, [company_id])
    )
    event_in_title = any(_compact(keyword) in title_compact for keyword in EVENT_LISTING_KEYWORDS)
    if event_in_title and not company_in_title:
        return True

    has_strong_action_keyword = any(
        _compact(keyword) in compact_text for keyword in STRONG_STRATEGIC_ACTION_KEYWORDS
    )
    if not company_in_title and not has_strong_action_keyword:
        return True

    has_action_keyword = any(
        _compact(keyword) in compact_text for keyword in STRATEGIC_ACTION_KEYWORDS
    )
    return not has_action_keyword


def _is_relevant(result: dict[str, Any]) -> bool:
    label = result.get("relevance_label", "irrelevant")
    score = float(result.get("relevance_score", 0.0))

    if label == "relevant" and score >= RELEVANCE_THRESHOLD:
        return True

    return False


def _metadata_patch_for_relevance(
    row: Any,
    result: dict[str, Any],
    is_relevant: bool,
) -> dict[str, Any]:
    if not is_relevant:
        return {
            "status_detail": "relevance_failed",
            "skip_reason": result.get("reason", ""),
        }

    companies = _normalize_company(row.company)
    matched_companies = result.get("matched_companies") or []
    matched_sectors = result.get("matched_sectors") or []
    has_sector = bool(matched_sectors and matched_sectors != ["other"])

    sector_details = result.get("matched_sector_details") or []

    is_industry_bucket = INDUSTRY_TREND_COMPANY in companies
    is_companyless_sector = has_sector and not matched_companies
    is_multi_peer_sector = has_sector and len(matched_companies) >= 2
    source_type = str(row.source_type or "").strip().lower()
    is_trend_source = source_type in {
        *STRUCTURED_SIGNAL_SOURCE_TYPES,
        *INDUSTRY_DOCUMENT_SOURCE_TYPES,
    }

    if not (
        sector_details
        or is_industry_bucket
        or is_companyless_sector
        or is_multi_peer_sector
        or is_trend_source
    ):
        return {}

    patch: dict[str, Any] = {
        "matched_companies": matched_companies,
        "matched_sectors": matched_sectors,
        "matched_sector_details": sector_details,
        "status_detail": "relevance_passed",
    }

    if is_industry_bucket or is_companyless_sector or is_multi_peer_sector or is_trend_source:
        patch["topic_scope"] = "industry_trend"

    if is_multi_peer_sector:
        patch["primary_company"] = None

    return patch


def _matched_sector_details_for_result(
    row: Any,
    matched_sectors: list[str],
) -> list[SectorMatch]:
    allowed = {sector for sector in matched_sectors if sector != "other"}
    if not allowed:
        return []

    text_value = f"{getattr(row, 'title', '') or ''} {getattr(row, 'content', '') or ''}"
    return [detail for detail in match_sector_details(text_value) if detail["sector_id"] in allowed]


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


def _matches_posco_dx_group_ax_signal(text_body: str) -> bool:
    if not text_body:
        return False
    if not _POSCO_DX_GROUP_AX_RE.search(text_body):
        return False
    text = _join_text(text_body)
    if _MARKET_PRICE_RE.search(text) and _MARKET_METRIC_RE.search(text):
        return False
    return True


def _has_posco_dx_primary_context(*, title: str, content: str) -> bool:
    title_compact = _compact(title)
    posco_aliases = [
        _compact(alias)
        for alias in ALL_COMPANY_ALIASES.get("posco_dx", ["포스코DX", "포스코"])
        if _compact(alias)
    ]
    if any(alias in title_compact for alias in posco_aliases):
        return True

    lead_compact = _compact(str(content or "")[:800])
    if not any(alias in lead_compact for alias in posco_aliases):
        return False
    return any(
        keyword in lead_compact
        for keyword in (
            "포스코그룹",
            "포스코DX",
            "포스코디엑스",
            "포스코가",
            "포스코는",
            "포스코의",
        )
    )


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

    label = _normalize_relevance_label(data.get("relevance_label", "irrelevant"))

    return _result(
        label=label,
        score=float(data.get("relevance_score", 0.60)),
        companies=data.get("matched_companies", []),
        sectors=data.get("matched_sectors", []),
        reason=data.get("reason", ""),
    )


def _build_batch_relevance_prompt(items: list[dict[str, Any]]) -> str:
    compact_items = [
        {
            "id": item["id"],
            "title": item["title"],
            "content": item["content"],
            "source_type": item.get("source_type") or "",
            "source_name": item.get("source_name") or "",
            "target_companies": item.get("company") or [],
            "matched_company_candidates": item.get("matched_company_candidates") or [],
            "matched_sector_candidates": item.get("matched_sector_candidates") or [],
            "peer_context": item.get("peer_context") or "",
        }
        for item in items
    ]
    return f"""\
당신은 기사 관련성 판단 Agent입니다.
아래 JSON 배열의 각 기사에 대해 독립적으로 relevant/irrelevant를 판단하세요.

판단 기준:
- 대상 company가 핵심 주체로 등장해야 합니다.
- sector는 실제 사업/기술/시장 맥락과 연결되어야 합니다.
- 수주, 계약, 제휴, 투자, 출시, 실적, 공시, IR, 채용, 조직개편,
  정책 변화 등 동향 신호가 있어야 합니다.
- 단순 키워드 나열, 광고, 관련 기사 목록, 행사 참가사 목록은 irrelevant입니다.

반드시 JSON 배열만 응답하세요. 입력 id를 그대로 유지하세요.
응답 형식:
[
  {{
    "id": 123,
    "relevance_label": "relevant|irrelevant",
    "relevance_score": 0.0,
    "matched_companies": ["company_id"],
    "matched_sectors": ["sector_id"],
    "reason": "짧은 판단 근거"
  }}
]

입력:
{json.dumps(compact_items, ensure_ascii=False)}
"""


def _parse_batch_relevance_json(text_value: str) -> dict[int, dict[str, Any]]:
    text_value = text_value.strip()
    if text_value.startswith("```"):
        text_value = text_value.split("```")[1]
        if text_value.startswith("json"):
            text_value = text_value[4:]

    data = json.loads(text_value.strip())
    if isinstance(data, dict):
        data = data.get("results") or data.get("items") or []
    if not isinstance(data, list):
        return {}

    results: dict[int, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        try:
            article_id = int(item["id"])
        except (TypeError, ValueError):
            continue
        label = _normalize_relevance_label(item.get("relevance_label", "irrelevant"))
        results[article_id] = _result(
            label=label,
            score=float(item.get("relevance_score", 0.60)),
            companies=item.get("matched_companies", []),
            sectors=item.get("matched_sectors", []),
            reason=item.get("reason", ""),
        )
    return results


def _normalize_relevance_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label == "relevant":
        return "relevant"
    return "irrelevant"


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


def _review_result(
    label: str,
    score: float,
    companies: list[str],
    sectors: list[str],
    reason: str,
    decision_code: str,
) -> dict[str, Any]:
    result = _result(
        label=label,
        score=score,
        companies=companies,
        sectors=sectors,
        reason=reason,
    )
    result["_needs_review"] = True
    result["decision_code"] = decision_code
    return result


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
