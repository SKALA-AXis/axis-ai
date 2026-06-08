"""1단계 분석·카드뉴스 생성 파이프라인 베이스 러너.

W2-1 이후 본 runner 는 LangGraph analysis flow 그래프를 invoke 하고 결과 (state) 를
호환 dict 로 반환한다. 카드 저장 (`save_card_news`) 은 analysis flow 의 `card_writer`
노드 안에서 일어나므로 runner 의 외부 `save_card` 옵션은 호환을 위해 유지한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from src.agents.analysis_graph_runner import AnalysisGraphRunner
from src.analysis.models import AnalysisInputBundle
from src.composers.card_news_composer import CardNewsComposer
from src.db.article_store import get_articles_by_ids, save_card_news
from src.db.postgres import SessionLocal
from src.pipeline.analysis_flow_graph import run_supervisor

log = logging.getLogger(__name__)


class AnalysisPipelineRunner:
    """Raw / 정제 데이터 저장소에서 1단계 분석 아키텍처를 실행한다."""

    def __init__(
        self,
        *,
        analysis_runner: AnalysisGraphRunner | None = None,
        analysis_supervisor: AnalysisGraphRunner | None = None,
        card_news_composer: CardNewsComposer | None = None,
    ) -> None:
        self.analysis_runner = analysis_runner or analysis_supervisor or AnalysisGraphRunner()
        self.analysis_supervisor = self.analysis_runner
        self.card_news_composer = card_news_composer or CardNewsComposer()

    def run_cluster(
        self,
        *,
        cluster_id: int,
        representative_id: int | None = None,
        cluster_article_ids: list[int] | None = None,
        classification: dict[str, Any] | None = None,
        save_card: bool = False,
    ) -> dict[str, Any]:
        """뉴스 클러스터 하나를 분석하고 카드뉴스 payload를 만든다."""
        article_ids = cluster_article_ids or list_cluster_article_ids(cluster_id)
        if representative_id is not None and representative_id not in article_ids:
            return _empty_pipeline_result(
                reason="representative article not in cluster",
                cluster_id=cluster_id,
                representative_id=representative_id,
            )
        articles = _load_articles(article_ids)
        if not articles:
            return _empty_pipeline_result(
                reason="cluster articles not found",
                cluster_id=cluster_id,
                representative_id=representative_id,
            )
        effective_representative_id = representative_id or _representative_id_from_articles(
            articles
        )
        return self.run_articles(
            cluster_id=cluster_id,
            representative_id=effective_representative_id,
            articles=articles,
            cluster_article_ids=article_ids,
            classification=classification,
            save_card=save_card,
        )

    def run_raw_article(
        self,
        *,
        raw_article_id: int,
        classification: dict[str, Any] | None = None,
        save_card: bool = False,
    ) -> dict[str, Any]:
        """단일 문서/기사 raw_article_id를 분석 대상으로 실행한다."""
        articles = _load_articles([raw_article_id])
        if not articles:
            return _empty_pipeline_result(
                reason="raw article not found",
                cluster_id=None,
                representative_id=raw_article_id,
            )
        article = articles[0]
        cluster_id = _safe_int(article.get("cluster_id")) or raw_article_id
        return self.run_articles(
            cluster_id=cluster_id,
            representative_id=raw_article_id,
            articles=articles,
            cluster_article_ids=[raw_article_id],
            classification=classification,
            save_card=save_card,
        )

    def run_articles(
        self,
        *,
        cluster_id: int,
        representative_id: int,
        articles: list[dict[str, Any]],
        cluster_article_ids: list[int] | None = None,
        classification: dict[str, Any] | None = None,
        save_card: bool = False,
    ) -> dict[str, Any]:
        """W2-1: analysis flow graph 를 invoke 하여 카드까지 한 번에 처리.

        - analysis flow 의 `card_writer` 노드가 이미 `card_news` INSERT 를 수행한다.
        - 따라서 본 runner 의 `save_card=True` 는 backward-compat 만을 위한 옵션이며,
          그래프가 카드를 저장하지 않은 경우 (validation fail 등) 에만 의미가 있다.
        """
        if not articles:
            return _empty_pipeline_result(
                reason="articles empty",
                cluster_id=cluster_id,
                representative_id=representative_id,
            )
        effective_classification = build_classification_from_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=articles,
            overrides=classification,
        )
        from src.agents.integration_agent import analysis_input_bundle_from_articles

        input_bundle = analysis_input_bundle_from_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=articles,
            cluster_article_ids=cluster_article_ids or _article_ids(articles),
            classification=effective_classification,
        )
        state = run_supervisor(
            input_bundle=input_bundle,
            classification=effective_classification,
        )
        analysis_package_obj = state.get("analysis_package")
        analysis_package = (
            analysis_package_obj.to_dict() if analysis_package_obj is not None else {}
        )
        card_news = state.get("card_news_payload") or {}
        saved_card_id = state.get("card_news_id")
        validation = state.get("validation")
        validation_payload = (
            validation.to_dict()
            if validation is not None
            else (analysis_package.get("validation") if analysis_package else {})
        )
        if save_card and card_news and not saved_card_id:
            saved_card_id = save_card_news(card_news)
        return {
            "ok": bool(card_news),
            "cluster_id": cluster_id,
            "representative_id": representative_id,
            "classification": effective_classification,
            "analysis_package": analysis_package,
            "integrated_issue": state.get("integrated_issue") or {},
            "analysis": state.get("analysis") or {},
            "implication": state.get("implication") or {},
            "sentence_grounding": state.get("sentence_grounding") or {},
            "card_news": card_news,
            "saved_card_id": saved_card_id,
            "human_review_flags": list(state.get("human_review_flags") or []),
            "validation": validation_payload,
            "errors": list(state.get("errors") or []),
        }

    def run_input_bundle(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        classification: dict[str, Any] | None = None,
        save_card: bool = False,
    ) -> dict[str, Any]:
        """AnalysisInputBundle 을 직접 받아 analysis flow 그래프를 호출."""
        effective_classification = classification or input_bundle.metadata.get(
            "classification",
            {},
        )
        state = run_supervisor(
            input_bundle=input_bundle,
            classification=effective_classification,
        )
        analysis_package_obj = state.get("analysis_package")
        analysis_package = (
            analysis_package_obj.to_dict() if analysis_package_obj is not None else {}
        )
        card_news = state.get("card_news_payload") or {}
        saved_card_id = state.get("card_news_id")
        validation = state.get("validation")
        validation_payload = (
            validation.to_dict()
            if validation is not None
            else (analysis_package.get("validation") if analysis_package else {})
        )
        if save_card and card_news and not saved_card_id:
            saved_card_id = save_card_news(card_news)
        return {
            "ok": bool(card_news),
            "cluster_id": input_bundle.cluster_id,
            "classification": effective_classification,
            "analysis_package": analysis_package,
            "integrated_issue": state.get("integrated_issue") or {},
            "analysis": state.get("analysis") or {},
            "implication": state.get("implication") or {},
            "card_news": card_news,
            "saved_card_id": saved_card_id,
            "human_review_flags": list(state.get("human_review_flags") or []),
            "validation": validation_payload,
            "errors": list(state.get("errors") or []),
        }

    def build_card_news(
        self,
        *,
        analysis_package: dict[str, Any],
        classification: dict[str, Any],
    ) -> dict[str, Any]:
        """AnalysisPackage를 카드뉴스 payload로 변환한다."""
        integrated_issue = analysis_package.get("integrated_issue") or analysis_package.get(
            "summary",
            {},
        )
        if not integrated_issue.get("is_valid_summary", True):
            log.info(
                "카드뉴스 생성 제외 | cluster=%s reason=invalid_integrated_issue:%s",
                integrated_issue.get("cluster_id"),
                integrated_issue.get("reason"),
            )
            return {}
        return self.card_news_composer.generate_from_analysis_package(
            analysis_package,
            classification=classification,
        )


def list_cluster_article_ids(cluster_id: int) -> list[int]:
    """raw_articles.cluster_id 기준으로 클러스터 전체 기사 ID를 조회한다."""
    if cluster_id is None:
        return []
    with SessionLocal() as db:
        rows = db.execute(
            text("""
                SELECT id
                FROM raw_articles
                WHERE cluster_id = :cluster_id
                ORDER BY
                    is_representative DESC NULLS LAST,
                    published_at DESC NULLS LAST,
                    collected_at DESC NULLS LAST,
                    id DESC
            """),
            {"cluster_id": cluster_id},
        ).fetchall()
    return [int(row._mapping["id"]) for row in rows]


def _load_articles(article_ids: list[int]) -> list[dict[str, Any]]:
    articles = _order_articles(get_articles_by_ids(article_ids), article_ids)
    return _attach_structured_rows(articles)


def _attach_structured_rows(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    article_ids = _article_ids(articles)
    if not article_ids:
        return articles
    metrics = _structured_rows(
        table="raw_article_financial_metrics",
        raw_article_ids=article_ids,
    )
    signals = _structured_rows(
        table="raw_article_business_signals",
        raw_article_ids=article_ids,
    )
    metrics_by_article = _group_by_raw_article_id(metrics)
    signals_by_article = _group_by_raw_article_id(signals)
    enriched: list[dict[str, Any]] = []
    for article in articles:
        article_id = _safe_int(article.get("id"))
        enriched.append(
            {
                **article,
                "financial_metrics": metrics_by_article.get(article_id, []),
                "business_signals": signals_by_article.get(article_id, []),
            }
        )
    return enriched


def _structured_rows(*, table: str, raw_article_ids: list[int]) -> list[dict[str, Any]]:
    if table not in {"raw_article_financial_metrics", "raw_article_business_signals"}:
        return []
    try:
        with SessionLocal() as db:
            rows = db.execute(
                text(f"""
                    SELECT *
                    FROM {table}
                    WHERE raw_article_id = ANY(:raw_article_ids)
                    ORDER BY raw_article_id, id
                """),
                {"raw_article_ids": raw_article_ids},
            ).fetchall()
    except Exception as exc:
        if _is_missing_relation(exc, table):
            log.info("structured rows 테이블 없음, 빈 컨텍스트로 진행 | table=%s", table)
        else:
            log.warning("structured rows 조회 실패 | table=%s error=%s", table, exc)
        return []
    return [dict(row._mapping) for row in rows]


def _group_by_raw_article_id(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_safe_int(row.get("raw_article_id")), []).append(row)
    return grouped


def build_classification_from_articles(
    *,
    cluster_id: int,
    representative_id: int,
    articles: list[dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """기존 classification 결과가 없을 때 raw_articles 메타로 최소 분류 payload를 만든다."""
    overrides = dict(overrides or {})
    representative = _representative_article(articles, representative_id)
    companies = _dedupe_strings(
        [
            *_normalize_string_list(overrides.get("company")),
            *_normalize_string_list(representative.get("company")),
            *[
                company
                for article in articles
                for company in _normalize_string_list(article.get("matched_companies"))
            ],
        ]
    )
    sectors = _dedupe_strings(
        [
            *_normalize_string_list(overrides.get("sectors")),
            *_normalize_string_list(overrides.get("sector")),
            *[
                sector
                for article in articles
                for sector in _normalize_string_list(article.get("matched_sectors"))
            ],
        ]
    )
    importance_score = _first_float(
        overrides.get("importance_score"),
        overrides.get("exposure_score"),
        representative.get("importance_score"),
        representative.get("relevance_score"),
        default=0.5,
    )
    importance = str(
        overrides.get("importance")
        or overrides.get("exposure_band")
        or _importance_band(importance_score)
    )
    company = overrides.get("company") or (companies[0] if companies else "")
    sector = overrides.get("sector") or (sectors[0] if sectors else "other")
    signals_override = overrides.get("signals")
    extra_signals: dict[str, Any] = (
        dict(signals_override) if isinstance(signals_override, dict) else {}
    )
    payload = {
        "cluster_id": cluster_id,
        "representative_id": representative_id,
        "company": company,
        "companies": companies,
        "sector": sector,
        "sectors": sectors or [sector],
        "event_type": overrides.get("event_type") or _event_type_from_articles(articles),
        "importance": importance,
        "importance_score": importance_score,
        "exposure_band": overrides.get("exposure_band") or importance,
        "exposure_score": overrides.get("exposure_score") or importance_score,
        "signals": {
            "cluster_size": len(articles),
            "source": "analysis_pipeline",
            "representative_id": representative_id,
            **extra_signals,
        },
    }
    return {**payload, **overrides}


def _event_type_from_articles(articles: list[dict[str, Any]]) -> str:
    for article in articles:
        metadata = _metadata(article)
        for key in ("event_type", "activity_type", "cluster_event_type"):
            value = article.get(key) or metadata.get(key)
            if value:
                return str(value)
    return "general_update"


def _representative_article(
    articles: list[dict[str, Any]],
    representative_id: int,
) -> dict[str, Any]:
    for article in articles:
        if _safe_int(article.get("id")) == representative_id:
            return article
    return articles[0] if articles else {}


def _representative_id_from_articles(articles: list[dict[str, Any]]) -> int:
    for article in articles:
        if article.get("is_representative"):
            return _safe_int(article.get("id"))
    return _safe_int(articles[0].get("id")) if articles else 0


def _order_articles(
    articles: list[dict[str, Any]],
    ordered_ids: list[int],
) -> list[dict[str, Any]]:
    order = {article_id: index for index, article_id in enumerate(ordered_ids)}
    return sorted(
        articles,
        key=lambda article: order.get(_safe_int(article.get("id")), len(order)),
    )


def _article_ids(articles: list[dict[str, Any]]) -> list[int]:
    return [
        article_id
        for article_id in (_safe_int(article.get("id")) for article in articles)
        if article_id
    ]


def _metadata(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        return [stripped]
    return [str(value)]


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _first_float(*values: Any, default: float) -> float:
    for value in values:
        try:
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _importance_band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_missing_relation(exc: Exception, relation_name: str) -> bool:
    text_value = str(exc)
    return "UndefinedTable" in text_value and relation_name in text_value


def _empty_pipeline_result(
    *,
    reason: str,
    cluster_id: int | None,
    representative_id: int | None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "reason": reason,
        "cluster_id": cluster_id,
        "representative_id": representative_id,
        "analysis_package": {},
        "card_news": {},
        "saved_card_id": None,
    }


__all__ = [
    "AnalysisPipelineRunner",
    "build_classification_from_articles",
    "list_cluster_article_ids",
]
