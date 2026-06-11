"""DB 전처리 서비스.

RAW 처리 대상 조회부터 source_type별 전처리 라우팅, 문서 분석,
기사 dedup/classification까지 전처리 파이프라인을 한 곳에서 조율한다.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import httpx
from sqlalchemy import text

from src.analysis.document_analysis_materializer import materialize_document_analysis
from src.config.companies import company_aliases, company_name_ko
from src.config.global_companies import global_company_aliases, global_company_name_ko
from src.config.preprocessing import (
    COMPANY_SITE_SOURCE_TYPES,
    DEFAULT_GPT_WORKERS,
    INDUSTRY_DOCUMENT_SOURCE_TYPES,
    METADATA_CHUNK_TEXT_CHARS,
    NEWS_SOURCE_TYPES,
    OFFICIAL_SOURCE_TYPES,
    PARSED_DOCUMENT_SOURCE_TYPES,
    STATUS_PROCESSED,
    STATUS_SKIPPED,
    STRUCTURED_SIGNAL_SOURCE_TYPES,
)
from src.db.article_store import get_articles_by_ids, save_pipeline_log, update_preprocess_status
from src.db.postgres import SessionLocal
from src.parsers.parser_quality import analyze_parser_quality_article
from src.parsers.parser_router import DocumentParserRouter
from src.preprocessing.classification import ClusterClassifier
from src.preprocessing.dedup import ArticleDeduplicator
from src.preprocessing.relevance import RelevanceEvaluator

log = logging.getLogger(__name__)

_LINK_VERIFY_TIMEOUT_SECONDS = float(os.getenv("PREPROCESS_LINK_VERIFY_TIMEOUT_SECONDS", "4.0"))
_LINK_VERIFY_USER_AGENT = os.getenv(
    "PREPROCESS_LINK_VERIFY_USER_AGENT",
    "Mozilla/5.0 (compatible; AxisPreprocessor/1.0)",
)

RELEVANCE_SOURCE_TYPES = NEWS_SOURCE_TYPES
OFFICIAL_RELEVANCE_SOURCE_TYPES: set[str] = set()
OFFICIAL_DOCUMENT_SOURCE_TYPES = OFFICIAL_SOURCE_TYPES
COMPANY_SITE_DOCUMENT_SOURCE_TYPES = COMPANY_SITE_SOURCE_TYPES
COMPANY_FILTER_EXEMPT_SOURCE_TYPES = INDUSTRY_DOCUMENT_SOURCE_TYPES | {"search_trend"}

_LOAD_SQL = text("""
    SELECT ra.id
    FROM raw_articles ra
    WHERE ra.processing_status = 'RAW'
      AND ra.crawl_status = 'success'
      AND (
          :no_filter
          OR ra.source_type = ANY(:company_filter_exempt_source_types)
          OR ra.company ?| :company
      )
      AND (:collected_since IS NULL OR ra.collected_at >= CAST(:collected_since AS timestamptz))
      AND (:published_since IS NULL OR ra.published_at >= CAST(:published_since AS timestamptz))
      AND (:published_until IS NULL OR ra.published_at < CAST(:published_until AS timestamptz))
      AND (
          :crawl_run_id IS NULL
          OR EXISTS (
              SELECT 1
              FROM crawl_run_articles cra
              WHERE cra.raw_article_id = ra.id
                AND cra.crawl_run_id = CAST(:crawl_run_id AS uuid)
          )
      )
      AND (:no_source_filter OR ra.source_type = ANY(:source_types))
    ORDER BY ra.published_at DESC NULLS LAST
    LIMIT :limit
""")


@dataclass(frozen=True)
class LinkCheckResult:
    status: str
    http_code: int | None = None
    final_url: str | None = None
    error: str | None = None

    @property
    def is_dead(self) -> bool:
        return self.status == "dead"


class PreprocessingResult(TypedDict):
    company: list[str]
    trigger_type: str
    collected_since: str | None
    crawl_run_id: str | None
    raw_article_ids: list[int]
    relevant_ids: list[int]
    official_document_ids: list[int]
    parsed_document_ids: list[int]
    industry_document_ids: list[int]
    structured_signal_ids: list[int]
    analysis_document_ids: list[int]
    analysis_source_counts: dict[str, int]
    analysis_metric_count: int
    analysis_signal_count: int
    analysis_errors: list[str]
    skipped_preprocess_ids: list[int]
    cluster_map: dict[int, list[int]]
    representative_ids: list[int]
    classified_clusters: list[dict[str, Any]]
    errors: list[str]
    human_review_flags: list[int]


def _company_filter_values(company_ids: list[str]) -> list[str]:
    """Expand peer ids to aliases used by source-specific crawlers."""
    values: list[str] = []
    for company_id in company_ids:
        normalized = company_id.strip()
        if not normalized:
            continue
        values.append(normalized)
        values.extend(company_aliases(normalized))
        values.append(company_name_ko(normalized))
        values.extend(global_company_aliases(normalized))
        values.append(global_company_name_ko(normalized))
    return list(dict.fromkeys(value for value in values if value))


class PreprocessingService:
    """DB 전처리 전담 서비스.

    실제 크롤링은 ``src.crawler`` / BatchProcessor 계층에서 끝난 뒤 raw_articles에 저장된다.
    이 서비스는 저장된 RAW 대상 조회와 DB 전처리 단계를 담당한다.
    """

    def __init__(
        self,
        *,
        relevance_evaluator: RelevanceEvaluator | None = None,
        deduplicator: ArticleDeduplicator | None = None,
        classifier: ClusterClassifier | None = None,
        max_workers: int = DEFAULT_GPT_WORKERS,
        verify_news_links: bool | None = None,
        link_checker: Callable[[str], LinkCheckResult] | None = None,
    ) -> None:
        self.relevance_evaluator = relevance_evaluator or RelevanceEvaluator(enable_llm=False)
        self.deduplicator = deduplicator or ArticleDeduplicator()
        self.classifier = classifier or ClusterClassifier(enable_llm=False)
        self.max_workers = max_workers
        self.verify_news_links = (
            verify_news_links
            if verify_news_links is not None
            else os.getenv("PREPROCESS_VERIFY_NEWS_LINKS", "true").lower()
            not in {"0", "false", "no"}
        )
        self.link_checker = link_checker or check_article_link

    def run(
        self,
        *,
        company: list[str] | None = None,
        source_types: list[str] | None = None,
        trigger_type: str = "manual",
        collected_since: str | None = None,
        published_since: str | None = None,
        published_until: str | None = None,
        crawl_run_id: str | None = None,
        limit: int = 500,
    ) -> PreprocessingResult:
        """DB 전처리 파이프라인을 순차 실행한다."""
        result: PreprocessingResult = {
            "company": company or [],
            "trigger_type": trigger_type,
            "collected_since": collected_since,
            "crawl_run_id": crawl_run_id,
            "raw_article_ids": [],
            "relevant_ids": [],
            "official_document_ids": [],
            "parsed_document_ids": [],
            "industry_document_ids": [],
            "structured_signal_ids": [],
            "analysis_document_ids": [],
            "analysis_source_counts": {},
            "analysis_metric_count": 0,
            "analysis_signal_count": 0,
            "analysis_errors": [],
            "skipped_preprocess_ids": [],
            "cluster_map": {},
            "representative_ids": [],
            "classified_clusters": [],
            "errors": [],
            "human_review_flags": [],
        }

        result["raw_article_ids"] = self._logged_call(
            "load_raw",
            len(result["company"]),
            lambda: self.load_raw_ids(
                result["company"],
                source_types=source_types,
                limit=limit,
                collected_since=collected_since,
                published_since=published_since,
                published_until=published_until,
                crawl_run_id=crawl_run_id,
            ),
            company=result["company"],
        )

        route_result = self._logged_call(
            "preprocess_route",
            len(result["raw_article_ids"]),
            lambda: self.route_by_source(result["raw_article_ids"]),
            company=result["company"],
        )
        result.update(route_result)

        analysis_result = self._logged_call(
            "document_analysis",
            len(result["parsed_document_ids"]),
            lambda: self.analyze_documents(result["parsed_document_ids"]),
            company=result["company"],
        )
        result.update(analysis_result)

        cluster_map, representative_ids = self._logged_call(
            "dedup",
            len(result["relevant_ids"]),
            lambda: self.deduplicate(result["relevant_ids"]),
            company=result["company"],
        )
        result["cluster_map"] = cluster_map
        result["representative_ids"] = representative_ids

        result["classified_clusters"] = self._logged_call(
            "classify",
            len(result["representative_ids"]),
            lambda: self.classify_clusters(
                representative_ids=result["representative_ids"],
                cluster_map=result["cluster_map"],
                requested_companies=result["company"],
            ),
            company=result["company"],
        )

        return result

    def _logged_call(
        self,
        step: str,
        input_count: int,
        call,
        *,
        company: list[str],
    ):
        started = time.perf_counter()
        error_msg: str | None = None
        output = None
        try:
            output = call()
            return output
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            raise
        finally:
            output_count = _count_value(output)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            company_label = company[0] if len(company) == 1 else None
            save_pipeline_log(
                step=step,
                company=company_label,
                input_count=input_count,
                output_count=output_count,
                elapsed_ms=elapsed_ms,
                error_msg=error_msg,
            )

    def load_raw_ids(
        self,
        company: list[str] | None = None,
        source_types: list[str] | None = None,
        limit: int = 500,
        collected_since: str | None = None,
        published_since: str | None = None,
        published_until: str | None = None,
        crawl_run_id: str | None = None,
    ) -> list[int]:
        """처리 대기 중인 RAW article id를 DB에서 조회한다."""
        company_filter = _company_filter_values(company or [])
        source_type_filter = [
            source_type.strip().lower()
            for source_type in (source_types or [])
            if source_type and source_type.strip()
        ]

        with SessionLocal() as db:
            rows = db.execute(
                _LOAD_SQL,
                {
                    "company": company_filter if company_filter else [""],
                    "no_filter": len(company_filter) == 0,
                    "company_filter_exempt_source_types": list(COMPANY_FILTER_EXEMPT_SOURCE_TYPES),
                    "source_types": source_type_filter if source_type_filter else [""],
                    "no_source_filter": len(source_type_filter) == 0,
                    "collected_since": collected_since,
                    "published_since": published_since,
                    "published_until": published_until,
                    "crawl_run_id": crawl_run_id,
                    "limit": limit,
                },
            ).fetchall()

        ids = [row.id for row in rows]
        log.info(
            (
                "RAW 기사 로드 | company=%s collected_since=%s "
                "published_since=%s published_until=%s crawl_run_id=%s count=%d"
            ),
            company_filter,
            collected_since,
            published_since,
            published_until,
            crawl_run_id,
            len(ids),
        )
        return ids

    def route_by_source(self, raw_article_ids: list[int]) -> dict[str, list[int]]:
        """source_type별 DB 전처리 라우팅을 수행한다."""
        if not raw_article_ids:
            return _empty_route_result()

        articles = get_articles_by_ids(raw_article_ids)
        by_source: dict[str, list[int]] = {}
        for article in articles:
            by_source.setdefault(_source_type(article), []).append(int(article["id"]))

        relevant_ids: list[int] = []
        official_document_ids: list[int] = []
        parsed_document_ids: list[int] = []
        industry_document_ids: list[int] = []
        structured_signal_ids: list[int] = []
        skipped_ids: list[int] = []
        review_ids: list[int] = []
        official_relevant_ids: set[int] = set()
        official_skipped_ids: set[int] = set()

        relevance_ids = [
            article_id
            for source_type in RELEVANCE_SOURCE_TYPES
            for article_id in by_source.get(source_type, [])
        ]
        dead_link_ids = self.skip_dead_news_links(articles, relevance_ids)
        if dead_link_ids:
            dead_link_id_set = set(dead_link_ids)
            skipped_ids.extend(dead_link_ids)
            relevance_ids = [
                article_id for article_id in relevance_ids if article_id not in dead_link_id_set
            ]

        if relevance_ids:
            passed, skipped = self.relevance_evaluator.filter(relevance_ids)
            relevant_ids.extend(passed)
            skipped_ids.extend(skipped)
            result_review_ids = getattr(self.relevance_evaluator, "review_ids", [])
            if result_review_ids:
                review_ids.extend(result_review_ids)
                log.info("Gate 2.5 REVIEW 보류 | count=%d", len(result_review_ids))

        official_relevance_ids = [
            article_id
            for source_type in OFFICIAL_RELEVANCE_SOURCE_TYPES
            for article_id in by_source.get(source_type, [])
        ]
        if official_relevance_ids:
            passed, skipped = self.relevance_evaluator.filter(official_relevance_ids)
            official_relevant_ids.update(passed)
            official_skipped_ids.update(skipped)
            skipped_ids.extend(skipped)
            result_review_ids = getattr(self.relevance_evaluator, "review_ids", [])
            if result_review_ids:
                review_ids.extend(result_review_ids)
                log.info("Gate 2.5 official REVIEW 보류 | count=%d", len(result_review_ids))

        for article in articles:
            article_id = int(article["id"])
            source_type = _source_type(article)

            if source_type in RELEVANCE_SOURCE_TYPES:
                continue

            agent_article = _article_for_agent(article)

            if source_type in OFFICIAL_DOCUMENT_SOURCE_TYPES:
                if article_id in official_skipped_ids:
                    continue
                official_document_ids.append(article_id)
                update_preprocess_status(
                    article_id,
                    STATUS_PROCESSED,
                    {
                        "document_scope": "company_official",
                        "preprocess_kind": "official_signal",
                        "status_detail": "official_document",
                        "preprocess_note": (
                            "official 문서는 relevance/sector 신호를 추출한 뒤 공식 원문으로 보존. "
                            "뉴스 dedup/classification 단계는 생략"
                        ),
                    },
                )
                continue

            if source_type in COMPANY_SITE_DOCUMENT_SOURCE_TYPES:
                official_document_ids.append(article_id)
                update_preprocess_status(
                    article_id,
                    STATUS_PROCESSED,
                    {
                        "document_scope": "company_site",
                        "status_detail": "company_site_document",
                        "preprocess_note": (
                            "company_site 문서는 회사 공식 홈페이지의 정적/반정적 원문으로 보존. "
                            "뉴스룸 relevance/dedup/classification 단계는 생략하고 "
                            "회사 지식베이스/과거 분석에서 사용"
                        ),
                    },
                )
                continue

            if source_type in PARSED_DOCUMENT_SOURCE_TYPES:
                item, ok, reason = analyze_parser_quality_article(agent_article)
                parser_result = item.get("parser_result") or {}
                metadata_patch = _parsed_document_metadata_patch(
                    source_type=source_type,
                    parser_result=parser_result,
                    parser_quality_score=item.get("parser_quality_score"),
                    parser_quality_label=item.get("parser_quality_label"),
                    parser_quality_reason=item.get("parser_quality_reason"),
                )
                if ok:
                    parsed_document_ids.append(article_id)
                    update_preprocess_status(
                        article_id,
                        STATUS_PROCESSED,
                        {
                            **metadata_patch,
                            "document_scope": "company_document",
                            "status_detail": "parsed_document",
                            "preprocess_note": (
                                f"{source_type} 문서는 parser quality check 후 보존. "
                                "기사 relevance/dedup/classification 단계는 생략"
                            ),
                        },
                    )
                else:
                    skipped_ids.append(article_id)
                    update_preprocess_status(
                        article_id,
                        STATUS_SKIPPED,
                        {
                            **metadata_patch,
                            "status_detail": "parser_quality_failed",
                            "skip_reason": reason,
                        },
                        error_message=reason,
                    )
                continue

            if source_type in INDUSTRY_DOCUMENT_SOURCE_TYPES:
                parser_result = DocumentParserRouter().parse_article(agent_article)
                industry_document_ids.append(article_id)
                update_preprocess_status(
                    article_id,
                    STATUS_PROCESSED,
                    {
                        "parser_result": parser_result,
                        "document_scope": "industry_trend",
                        "status_detail": "industry_document",
                        "preprocess_note": (
                            "산업 동향 문서는 기사 relevance/signal 축약 없이 "
                            "추후 본문 분석 대상으로 보존"
                        ),
                    },
                )
                continue

            if source_type in STRUCTURED_SIGNAL_SOURCE_TYPES:
                structured_signal_ids.append(article_id)
                update_preprocess_status(
                    article_id,
                    STATUS_PROCESSED,
                    {
                        "signal_scope": source_type,
                        "status_detail": "structured_signal",
                        "preprocess_note": (
                            f"{source_type} 데이터는 기사/문서가 아닌 구조화 신호로 보존. "
                            "급변/급증 탐지 및 종합 분석 단계에서 사용"
                        ),
                    },
                )
                continue

            skipped_ids.append(article_id)
            update_preprocess_status(
                article_id,
                STATUS_SKIPPED,
                {
                    "status_detail": "unsupported_source",
                    "skip_reason": (
                        f"{source_type or 'unknown'} source_type은 현재 전처리 대상이 아님"
                    ),
                },
            )

        log.info(
            (
                "전처리 라우팅 완료 | relevant=%d official_docs=%d parsed_docs=%d "
                "industry_docs=%d structured=%d skipped=%d"
            ),
            len(relevant_ids),
            len(official_document_ids),
            len(parsed_document_ids),
            len(industry_document_ids),
            len(structured_signal_ids),
            len(skipped_ids),
        )
        return {
            "relevant_ids": relevant_ids,
            "official_document_ids": official_document_ids,
            "parsed_document_ids": parsed_document_ids,
            "industry_document_ids": industry_document_ids,
            "structured_signal_ids": structured_signal_ids,
            "skipped_preprocess_ids": skipped_ids,
            "human_review_flags": list(dict.fromkeys(review_ids)),
        }

    def skip_dead_news_links(
        self,
        articles: list[dict[str, Any]],
        relevance_ids: list[int],
    ) -> list[int]:
        """Exclude news rows whose original URL is confirmed dead before relevance."""
        if not self.verify_news_links or not relevance_ids:
            return []

        relevance_id_set = set(relevance_ids)
        skipped_ids: list[int] = []
        for article in articles:
            article_id = int(article["id"])
            if article_id not in relevance_id_set:
                continue
            url = str(article.get("url") or "").strip()
            result = self.link_checker(url)
            if not result.is_dead:
                continue

            skipped_ids.append(article_id)
            _mark_dead_link_article(article_id, url, result)

        if skipped_ids:
            log.info("뉴스 dead link 전처리 제외 | count=%d ids=%s", len(skipped_ids), skipped_ids)
        return skipped_ids

    def analyze_documents(self, parsed_document_ids: list[int]) -> dict[str, Any]:
        """파싱 완료 문서를 분석 테이블로 정규화한다."""
        return materialize_document_analysis(parsed_document_ids)

    def deduplicate(self, relevant_ids: list[int]) -> tuple[dict[int, list[int]], list[int]]:
        """관련 기사 ID를 클러스터링하고 대표 기사 ID를 반환한다."""
        cluster_map, rep_ids = self.deduplicator.deduplicate(relevant_ids)
        log.info("Gate 3 완료 | clusters=%d reps=%d", len(cluster_map), len(rep_ids))
        return cluster_map, rep_ids

    def classify_clusters(
        self,
        representative_ids: list[int],
        cluster_map: dict[int, list[int]],
        requested_companies: list[str],
    ) -> list[dict[str, Any]]:
        """대표 기사 기준으로 클러스터를 분류한다."""
        rep_articles = {a["id"]: a for a in get_articles_by_ids(representative_ids)}

        def _classify_one(cluster_id: int, article_ids: list[int]) -> dict[str, Any] | None:
            rep_id = next(
                (aid for aid in article_ids if aid in rep_articles),
                article_ids[0] if article_ids else None,
            )
            if rep_id is None:
                return None
            company = _company_for_context(rep_articles.get(rep_id, {}), requested_companies)
            result = self.classifier.classify(
                cluster_id=cluster_id,
                representative_id=rep_id,
                cluster_article_ids=article_ids,
                company=company,
            )
            return {
                "cluster_id": cluster_id,
                "representative_id": rep_id,
                "company": company,
                **result,
            }

        classified: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(_classify_one, cid, aids): cid for cid, aids in cluster_map.items()
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    classified.append(result)

        log.info("v3 분류 완료 | clusters=%d", len(classified))
        return classified


def _empty_route_result() -> dict[str, list[int]]:
    return {
        "relevant_ids": [],
        "official_document_ids": [],
        "parsed_document_ids": [],
        "industry_document_ids": [],
        "structured_signal_ids": [],
        "skipped_preprocess_ids": [],
        "human_review_flags": [],
    }


def _compact_parser_result_for_metadata(parser_result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(parser_result)
    if "document_chunks" in compact:
        compact["document_chunks"] = _compact_document_chunks(
            compact.get("document_chunks"),
        )
    return compact


def _parsed_document_metadata_patch(
    *,
    source_type: str,
    parser_result: dict[str, Any],
    parser_quality_score: Any,
    parser_quality_label: Any,
    parser_quality_reason: Any,
) -> dict[str, Any]:
    compact_parser_result = _compact_parser_result_for_metadata(parser_result)
    metadata_patch: dict[str, Any] = {
        "parser_result": compact_parser_result,
        "parser_quality_score": parser_quality_score,
        "parser_quality_label": parser_quality_label,
        "parser_quality_reason": parser_quality_reason,
    }

    if source_type == "dart":
        metadata_patch.update(
            {
                "period": parser_result.get("period"),
                "period_year": parser_result.get("period_year"),
                "period_quarter": parser_result.get("period_quarter"),
                "period_type": parser_result.get("period_type"),
                "financial_record": parser_result.get("financial_record"),
                "topics": parser_result.get("topics"),
                "topic_signals": parser_result.get("topic_signals"),
                "dart_sections": parser_result.get("sections"),
                "dart_section_index": parser_result.get("section_index"),
                "dart_section_tree": parser_result.get("section_tree"),
                "dart_document_chunks": compact_parser_result.get("document_chunks"),
                "dart_classified_tables": parser_result.get("classified_tables"),
                "dart_financial_statements": parser_result.get("financial_statements"),
            }
        )
    elif source_type == "ir":
        metadata_patch.update(
            {
                "period": parser_result.get("period"),
                "period_year": parser_result.get("period_year"),
                "period_quarter": parser_result.get("period_quarter"),
                "period_type": parser_result.get("period_type"),
                "financial_record": parser_result.get("financial_record"),
                "topics": parser_result.get("topics"),
                "topic_signals": parser_result.get("topic_signals"),
                "ir_sections": parser_result.get("sections"),
                "ir_page_index": parser_result.get("page_index"),
                "ir_document_chunks": compact_parser_result.get("document_chunks"),
            }
        )
    elif source_type == "securities_report":
        metadata_patch.update(
            {
                "period": parser_result.get("period"),
                "report_firm": parser_result.get("report_firm"),
                "investment_opinion": parser_result.get("investment_opinion"),
                "target_price_krw": parser_result.get("target_price_krw"),
                "current_price_krw": parser_result.get("current_price_krw"),
                "topics": parser_result.get("topics"),
                "topic_signals": parser_result.get("topic_signals"),
                "securities_report_sections": parser_result.get("sections"),
                "securities_report_document_chunks": compact_parser_result.get("document_chunks"),
            }
        )

    return metadata_patch


def _compact_document_chunks(chunks: Any) -> list[dict[str, Any]]:
    if not isinstance(chunks, list):
        return []

    compacted: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text") or "")
        compacted.append(
            {key: value for key, value in chunk.items() if key != "text"}
            | {
                "text": text[:METADATA_CHUNK_TEXT_CHARS],
                "text_chars": len(text) or chunk.get("text_chars"),
                "text_is_truncated_for_metadata": len(text) > METADATA_CHUNK_TEXT_CHARS,
            }
        )
    return compacted


def _source_type(article: dict) -> str:
    return str(article.get("source_type") or "").strip().lower()


def _article_for_agent(article: dict) -> dict:
    item = dict(article)
    metadata = item.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    item["metadata"] = metadata if isinstance(metadata, dict) else {}
    item["extra"] = item["metadata"]
    return item


def check_article_link(url: str) -> LinkCheckResult:
    """Return dead only for deterministic 404/410 responses.

    Many publishers block HEAD/automation, so 403, timeout, and transient server errors are
    treated as unknown instead of being excluded from preprocessing.
    """
    if not url:
        return LinkCheckResult(status="unknown", error="empty_url")

    headers = {"User-Agent": _LINK_VERIFY_USER_AGENT}
    try:
        with httpx.Client(
            timeout=_LINK_VERIFY_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = client.head(url)
            if response.status_code in {404, 410}:
                return LinkCheckResult(
                    status="dead",
                    http_code=response.status_code,
                    final_url=str(response.url),
                )
            if 200 <= response.status_code < 400:
                return LinkCheckResult(
                    status="live",
                    http_code=response.status_code,
                    final_url=str(response.url),
                )
            if response.status_code in {405, 501}:
                response = client.get(url)
                if response.status_code in {404, 410}:
                    return LinkCheckResult(
                        status="dead",
                        http_code=response.status_code,
                        final_url=str(response.url),
                    )
                if 200 <= response.status_code < 400:
                    return LinkCheckResult(
                        status="live",
                        http_code=response.status_code,
                        final_url=str(response.url),
                    )
            return LinkCheckResult(
                status="unknown",
                http_code=response.status_code,
                final_url=str(response.url),
            )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        return LinkCheckResult(status="unknown", error=f"{type(exc).__name__}: {exc}")


def _mark_dead_link_article(article_id: int, url: str, result: LinkCheckResult) -> None:
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET crawl_status = 'failed',
                    processing_status = :processing_status,
                    relevance_label = 'irrelevant',
                    relevance_score = 0.0,
                    relevance_reason = :reason,
                    error_message = :reason,
                    cluster_id = NULL,
                    is_representative = FALSE,
                    metadata = COALESCE(metadata, '{}'::jsonb) || CAST(:metadata_patch AS jsonb)
                WHERE id = :id
            """),
            {
                "id": article_id,
                "processing_status": STATUS_SKIPPED,
                "reason": "원문 URL이 404/410 dead link로 확인되어 전처리 제외",
                "metadata_patch": json.dumps(
                    {
                        "status_detail": "dead_link",
                        "skip_reason": "dead_link",
                        "link_check": {
                            "url": url,
                            "status": result.status,
                            "http_code": result.http_code,
                            "final_url": result.final_url,
                            "error": result.error,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        )
        db.commit()


def _company_for_context(article: dict, requested_companies: list[str]) -> str:
    article_companies = _company_list(article)

    for company in requested_companies:
        if company in article_companies:
            return company

    return (
        article_companies[0]
        if article_companies
        else (requested_companies[0] if requested_companies else "")
    )


def _company_list(article: dict) -> list[str]:
    company = article.get("company")

    if isinstance(company, list):
        return [str(value) for value in company if value]

    if isinstance(company, str):
        try:
            parsed = json.loads(company)
            if isinstance(parsed, list):
                return [str(value) for value in parsed if value]
        except json.JSONDecodeError:
            stripped = company.strip()
            return [stripped] if stripped else []

    return []


__all__ = [
    "PreprocessingService",
    "PreprocessingResult",
    "RELEVANCE_SOURCE_TYPES",
    "OFFICIAL_DOCUMENT_SOURCE_TYPES",
    "COMPANY_SITE_DOCUMENT_SOURCE_TYPES",
    "PARSED_DOCUMENT_SOURCE_TYPES",
    "STRUCTURED_SIGNAL_SOURCE_TYPES",
]


def _count_value(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, dict, set)):
        return len(value)
    return 1
