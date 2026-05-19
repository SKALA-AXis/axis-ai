"""DB 전처리 서비스.

RAW 처리 대상 조회부터 source_type별 전처리 라우팅, 기사 dedup/classification까지
ingestion_graph의 전처리 구간을 한 곳에서 조율한다.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from sqlalchemy import text

from src.db.article_store import get_articles_by_ids, update_preprocess_status
from src.db.postgres import SessionLocal
from src.parsers.parser_quality import analyze_parser_quality_article
from src.parsers.parser_router import DocumentParserRouter
from src.preprocessing.classification import ClusterClassifier
from src.preprocessing.dedup import ArticleDeduplicator
from src.preprocessing.relevance import RelevanceEvaluator

log = logging.getLogger(__name__)

RELEVANCE_SOURCE_TYPES = {"news"}
OFFICIAL_DOCUMENT_SOURCE_TYPES = {"official"}
COMPANY_SITE_DOCUMENT_SOURCE_TYPES = {"company_site"}
PARSED_DOCUMENT_SOURCE_TYPES = {"dart", "ir", "securities_report"}
STRUCTURED_SIGNAL_SOURCE_TYPES = {"job", "market_data", "search_trend", "social"}
METADATA_CHUNK_TEXT_CHARS = 1200
DEFAULT_GPT_WORKERS = 5

_LOAD_SQL = text("""
    SELECT ra.id
    FROM raw_articles ra
    WHERE ra.processing_status = 'RAW'
      AND ra.crawl_status = 'success'
      AND (:no_filter OR ra.company ?| :company)
      AND (:collected_since IS NULL OR ra.collected_at >= CAST(:collected_since AS timestamptz))
      AND (
          :crawl_run_id IS NULL
          OR EXISTS (
              SELECT 1
              FROM crawl_run_articles cra
              WHERE cra.raw_article_id = ra.id
                AND cra.crawl_run_id = CAST(:crawl_run_id AS uuid)
          )
      )
    ORDER BY ra.published_at DESC NULLS LAST
    LIMIT :limit
""")


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
    ) -> None:
        self.relevance_evaluator = relevance_evaluator or RelevanceEvaluator()
        self.deduplicator = deduplicator or ArticleDeduplicator()
        self.classifier = classifier or ClusterClassifier()
        self.max_workers = max_workers

    def load_raw_ids(
        self,
        company: list[str] | None = None,
        limit: int = 500,
        collected_since: str | None = None,
        crawl_run_id: str | None = None,
    ) -> list[int]:
        """처리 대기 중인 RAW article id를 DB에서 조회한다."""
        company_filter = company or []

        with SessionLocal() as db:
            rows = db.execute(
                _LOAD_SQL,
                {
                    "company": company_filter if company_filter else [""],
                    "no_filter": len(company_filter) == 0,
                    "collected_since": collected_since,
                    "crawl_run_id": crawl_run_id,
                    "limit": limit,
                },
            ).fetchall()

        ids = [row.id for row in rows]
        log.info(
            "RAW 기사 로드 | company=%s collected_since=%s crawl_run_id=%s count=%d",
            company_filter,
            collected_since,
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

        relevance_ids = [
            article_id
            for source_type in RELEVANCE_SOURCE_TYPES
            for article_id in by_source.get(source_type, [])
        ]
        if relevance_ids:
            passed, skipped = self.relevance_evaluator.filter(relevance_ids)
            relevant_ids.extend(passed)
            skipped_ids.extend(skipped)

        for article in articles:
            article_id = int(article["id"])
            source_type = _source_type(article)

            if source_type in RELEVANCE_SOURCE_TYPES:
                continue

            agent_article = _article_for_agent(article)

            if source_type in OFFICIAL_DOCUMENT_SOURCE_TYPES:
                official_document_ids.append(article_id)
                update_preprocess_status(
                    article_id,
                    "PREPROCESSED_OFFICIAL_DOCUMENT",
                    {
                        "document_scope": "company_official",
                        "preprocess_note": (
                            "official 문서는 회사별 공식 원문으로 보존. "
                            "기사 relevance/dedup/classification 단계는 생략하고 "
                            "추후 동향 분석에서 사용"
                        ),
                    },
                )
                continue

            if source_type in COMPANY_SITE_DOCUMENT_SOURCE_TYPES:
                official_document_ids.append(article_id)
                update_preprocess_status(
                    article_id,
                    "PREPROCESSED_COMPANY_SITE_DOCUMENT",
                    {
                        "document_scope": "company_site",
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
                        "PREPROCESSED_PARSED_DOCUMENT",
                        {
                            **metadata_patch,
                            "document_scope": "company_document",
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
                        "SKIPPED_PARSER_QUALITY",
                        metadata_patch,
                        error_message=reason,
                    )
                continue

            if source_type == "trend_report":
                parser_result = DocumentParserRouter().parse_article(agent_article)
                industry_document_ids.append(article_id)
                update_preprocess_status(
                    article_id,
                    "PREPROCESSED_INDUSTRY_DOCUMENT",
                    {
                        "parser_result": parser_result,
                        "document_scope": "industry_trend",
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
                    "PREPROCESSED_STRUCTURED_SIGNAL",
                    {
                        "signal_scope": source_type,
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
                "SKIPPED_PREPROCESS_UNSUPPORTED_SOURCE",
                {"skip_reason": f"{source_type or 'unknown'} source_type은 현재 전처리 대상이 아님"},
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
        }

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
            futures = {ex.submit(_classify_one, cid, aids): cid for cid, aids in cluster_map.items()}
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
            {
                **chunk,
                "text": text[:METADATA_CHUNK_TEXT_CHARS],
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
    "RELEVANCE_SOURCE_TYPES",
    "OFFICIAL_DOCUMENT_SOURCE_TYPES",
    "COMPANY_SITE_DOCUMENT_SOURCE_TYPES",
    "PARSED_DOCUMENT_SOURCE_TYPES",
    "STRUCTURED_SIGNAL_SOURCE_TYPES",
]
