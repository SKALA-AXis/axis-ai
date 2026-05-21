"""Layer B 분석·카드뉴스·벡터 인덱싱 파이프라인.

develop 의 PreprocessingService (Layer A) 이후 실행된다.
DataAnalysisSupervisorAgent → CardNewsAgent 흐름은 AnalysisPipelineRunner 가 담당하며,
본 모듈은 전처리 결과를 받아 카드 생성과 Qdrant 인덱싱을 조율한다.

As-Is: ingestion_graph.card_news_node + vector_index_node
To-Be: W2-1 에서 card_writer 가 supervisor_graph 로 이관되기 전까지의 중간 배치.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypedDict

from src.config.company_tiers import SELF_COMPANY_IDS
from src.config.preprocessing import DEFAULT_GPT_WORKERS
from src.db.article_store import save_card_news, save_pipeline_log
from src.pipeline.analysis_pipeline import AnalysisPipelineRunner
from src.preprocessing.preprocessing import PreprocessingResult

log = logging.getLogger(__name__)


class AnalysisDeliveryResult(TypedDict):
    card_news: list[dict[str, Any]]
    indexed_vector_ids: list[str]
    errors: list[str]


def run_analysis_delivery(
    preprocess_result: PreprocessingResult,
    *,
    max_workers: int = DEFAULT_GPT_WORKERS,
) -> AnalysisDeliveryResult:
    """전처리 결과를 기준으로 supervisor 분석·카드뉴스·벡터 인덱싱을 실행한다."""
    service = AnalysisDeliveryService(max_workers=max_workers)
    return service.run(preprocess_result)


class AnalysisDeliveryService:
    """Layer B: classified cluster / 문서·신호 대상 → card_news → vector index."""

    def __init__(self, *, max_workers: int = DEFAULT_GPT_WORKERS) -> None:
        self.max_workers = max_workers
        self.runner = AnalysisPipelineRunner()

    def run(self, preprocess_result: PreprocessingResult) -> AnalysisDeliveryResult:
        t0 = time.perf_counter()
        errors: list[str] = []
        cards: list[dict[str, Any]] = []

        try:
            cards = self._generate_cards(preprocess_result)
            indexed = self._index_cards(cards)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            log.exception("analysis delivery 실패")
            indexed = []

        elapsed = int((time.perf_counter() - t0) * 1000)
        save_pipeline_log(
            step="analysis_delivery",
            company=None,
            input_count=len(preprocess_result.get("classified_clusters", [])),
            output_count=len(cards),
            elapsed_ms=elapsed,
            error_msg=errors[0] if errors else None,
        )
        log.info(
            "analysis delivery 완료 | cards=%d indexed=%d errors=%d elapsed_ms=%d",
            len(cards),
            len(indexed),
            len(errors),
            elapsed,
        )
        return {
            "card_news": cards,
            "indexed_vector_ids": indexed,
            "errors": errors,
        }

    def _generate_cards(self, preprocess_result: PreprocessingResult) -> list[dict[str, Any]]:
        cluster_map = preprocess_result.get("cluster_map") or {}
        classified_clusters = preprocess_result.get("classified_clusters") or []
        document_targets = _document_analysis_targets(preprocess_result)
        cards: list[dict[str, Any]] = []

        def _generate_cluster_card(cluster: dict[str, Any]) -> dict[str, Any]:
            if cluster.get("company") in SELF_COMPANY_IDS:
                log.info(
                    "카드뉴스 생성 제외 | cluster=%s company=%s reason=self company",
                    cluster.get("cluster_id"),
                    cluster.get("company"),
                )
                return {}

            cluster_id = cluster["cluster_id"]
            article_ids = cluster_map.get(cluster_id, [])
            result = self.runner.run_cluster(
                cluster_id=cluster_id,
                representative_id=cluster["representative_id"],
                cluster_article_ids=article_ids,
                classification=cluster,
                save_card=True,
            )
            analysis_package = result.get("analysis_package") or {}
            integrated_issue = analysis_package.get("integrated_issue") or analysis_package.get(
                "summary",
                {},
            )
            if not integrated_issue.get("is_valid_summary"):
                log.info(
                    "카드뉴스 생성 제외 | cluster=%s company=%s reason=invalid_summary:%s",
                    cluster_id,
                    cluster.get("company"),
                    integrated_issue.get("reason"),
                )
                return {}
            return result.get("card_news") or {}

        def _generate_raw_article_card(raw_article_id: int, source_group: str) -> dict[str, Any]:
            result = self.runner.run_raw_article(raw_article_id=raw_article_id, save_card=False)
            card = result.get("card_news") or {}
            if not card:
                log.info(
                    "문서/신호 카드뉴스 생성 제외 | raw_article_id=%s group=%s reason=no_card",
                    raw_article_id,
                    source_group,
                )
                return {}
            if card.get("company") in SELF_COMPANY_IDS or card.get("peer_id") in SELF_COMPANY_IDS:
                log.info(
                    "문서/신호 카드뉴스 생성 제외 | raw_article_id=%s group=%s "
                    "company=%s reason=self company",
                    raw_article_id,
                    source_group,
                    card.get("company") or card.get("peer_id"),
                )
                return {}
            save_card_news(card)
            card.setdefault("source_group", source_group)
            return card

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                *[
                    executor.submit(_generate_cluster_card, cluster)
                    for cluster in classified_clusters
                ],
                *[
                    executor.submit(_generate_raw_article_card, raw_article_id, source_group)
                    for source_group, raw_article_ids in document_targets.items()
                    for raw_article_id in raw_article_ids
                ],
            ]
            for future in as_completed(futures):
                card = future.result()
                if card:
                    cards.append(card)

        log.info(
            "카드 뉴스 생성 완료 | news_clusters=%d document_targets=%d cards=%d",
            len(classified_clusters),
            sum(len(ids) for ids in document_targets.values()),
            len(cards),
        )
        return cards

    def _index_cards(self, cards: list[dict[str, Any]]) -> list[str]:
        from src.rag.vector_index import index_card

        targets = [
            card
            for card in cards
            if bool(card.get("validation_pass", card.get("validation", {}).get("pass", True)))
        ]
        indexed: list[str] = []
        for card in targets:
            try:
                point_id = index_card(card)
                if point_id:
                    indexed.append(point_id)
            except Exception as exc:
                log.error("Qdrant 인덱싱 실패 | card=%s error=%s", card.get("id"), exc)

        log.info(
            "Qdrant 인덱싱 완료 | targets=%d indexed=%d skipped=%d",
            len(targets),
            len(indexed),
            len(targets) - len(indexed),
        )
        save_pipeline_log(
            step="vector_index",
            company=None,
            input_count=len(targets),
            output_count=len(indexed),
            elapsed_ms=0,
            error_msg=None,
        )
        return indexed


def _document_analysis_targets(preprocess_result: PreprocessingResult) -> dict[str, list[int]]:
    return {
        "official_document": _dedupe_positive_ints(
            preprocess_result.get("official_document_ids", [])
        ),
        "parsed_document": _dedupe_positive_ints(preprocess_result.get("parsed_document_ids", [])),
        "industry_document": _dedupe_positive_ints(
            preprocess_result.get("industry_document_ids", [])
        ),
        "structured_signal": _dedupe_positive_ints(
            preprocess_result.get("structured_signal_ids", [])
        ),
    }


def _dedupe_positive_ints(values: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


__all__ = [
    "AnalysisDeliveryResult",
    "AnalysisDeliveryService",
    "run_analysis_delivery",
]
