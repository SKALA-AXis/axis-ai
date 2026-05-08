"""DB 저장 없이 저장된 crawler JSON 결과만 전처리한다.

사용법:
  uv run python run_preprocess_once.py --input src/crawler/crawler_results/ir_20260506.json
  uv run python run_preprocess_once.py --source-type news
  uv run python run_preprocess_once.py

이 runner는 로컬 확인용이다. 크롤링은 실행하지 않고, 이미 저장된 crawler JSON을 읽어
agents 모듈의 credibility, relevance, dedup, classification 결과를 JSON으로 저장한다.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.classification_agent import classify_preprocessed_cluster
from src.agents.credibility_agent import analyze_credibility_article
from src.agents.dedup_agent import deduplicate_articles
from src.agents.parser_agent import ParserAgent
from src.agents.parser_quality_agent import analyze_parser_quality_article
from src.agents.relevance_agent import analyze_relevance_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_preprocess")

DEFAULT_INPUT_DIR = Path("src/crawler/crawler_results")
DEFAULT_OUTPUT_DIR = Path("src/crawler/crawler_results/preprocessed")

RELEVANCE_SOURCE_TYPES = {"news"}
OFFICIAL_DOCUMENT_SOURCE_TYPES = {"official"}
PARSED_DOCUMENT_SOURCE_TYPES = {"dart", "ir", "securities_report"}
INDUSTRY_DOCUMENT_SOURCE_NAMES = {"bcg", "spri"}
STRUCTURED_SIGNAL_SOURCE_TYPES = {"job", "market_data", "search_trend", "social"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB 없이 저장된 crawler JSON 전처리 결과 확인")
    parser.add_argument(
        "--input",
        default=None,
        help="전처리할 crawler JSON 파일. 생략하면 src/crawler/crawler_results/*.json 전체를 읽음",
    )
    parser.add_argument(
        "--source-type",
        action="append",
        default=None,
        help=(
            "처리할 source_type 필터. 여러 번 지정 가능. "
            "예: --source-type news --source-type official"
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _load_articles(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input:
        return _read_json(Path(args.input))

    paths = _discover_input_files(DEFAULT_INPUT_DIR)
    if not paths:
        raise ValueError(f"crawler JSON 파일을 찾지 못했습니다: {DEFAULT_INPUT_DIR}")

    articles: list[dict[str, Any]] = []
    loaded_files = 0
    for path in paths:
        rows = _read_json(path, source_types=args.source_type)
        if args.source_type and not rows:
            continue
        loaded_files += 1
        articles.extend(rows)

    log.info(
        "crawler_results 로드 완료 | dir=%s files=%d rows=%d",
        DEFAULT_INPUT_DIR,
        loaded_files,
        len(articles),
    )
    return articles


def _discover_input_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def _read_json(path: Path, source_types: list[str] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("articles"), list):
        rows = payload["articles"]
    else:
        raise ValueError(
            "crawler JSON은 article 배열이거나 articles 배열을 포함한 객체여야 합니다."
        )

    if source_types:
        rows = _filter_by_source_type(rows, source_types, log_result=False)

    if not source_types or rows:
        log.info("JSON 로드 완료 | path=%s rows=%d", path, len(rows))
    return rows


def _filter_by_source_type(
    articles: list[dict[str, Any]],
    source_types: list[str] | None,
    log_result: bool = True,
) -> list[dict[str, Any]]:
    if not source_types:
        return articles

    allowed = {source_type.strip().lower() for source_type in source_types if source_type.strip()}
    if not allowed:
        return articles

    filtered = [
        article
        for article in articles
        if str(article.get("source_type") or "").strip().lower() in allowed
    ]
    if log_result:
        log.info(
            "source_type 필터 적용 | source_types=%s before=%d after=%d",
            sorted(allowed),
            len(articles),
            len(filtered),
        )
    return filtered


def _preprocess(articles: list[dict[str, Any]]) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    official_documents: list[dict[str, Any]] = []
    parsed_documents: list[dict[str, Any]] = []
    industry_documents: list[dict[str, Any]] = []
    structured_signals: list[dict[str, Any]] = []
    skipped_items: list[dict[str, Any]] = []
    credible_count = 0

    for index, article in enumerate(articles, start=1):
        item = dict(article)
        item["preprocess_id"] = index

        item, is_credible, _reason = analyze_credibility_article(item)
        if not is_credible:
            skipped_items.append(item)
            continue
        credible_count += 1

        route = _preprocess_route(item)

        if route == "relevance":
            item, is_relevant = analyze_relevance_article(item)
        elif route == "official_document":
            official_documents.append(_mark_official_document(item))
            continue
        elif route == "parsed_document":
            item, parse_ok, _parse_reason = analyze_parser_quality_article(item)
            if not parse_ok:
                skipped_items.append(item)
                continue
            parsed_documents.append(_mark_parsed_document(item))
            continue
        elif route == "industry_document":
            industry_documents.append(_mark_industry_document(item))
            continue
        elif route == "structured_signal":
            structured_signals.append(_mark_structured_signal(item))
            continue
        else:
            item = _mark_unsupported_source(item)
            skipped_items.append(item)
            continue

        if is_relevant:
            item["processing_status"] = "PREPROCESSED"
            enriched.append(item)
        else:
            skipped_items.append(item)

    cluster_map, representative_ids = deduplicate_articles(enriched)
    _mark_representatives(enriched, cluster_map, representative_ids)
    classified = _classify(enriched, cluster_map, representative_ids)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "raw": len(articles),
            "valid_after_credibility": credible_count,
            "relevant": len(enriched),
            "official_documents": len(official_documents),
            "parsed_documents": len(parsed_documents),
            "industry_documents": len(industry_documents),
            "structured_signals": len(structured_signals),
            "skipped": len(skipped_items),
            "clusters": len(cluster_map),
            "representatives": len(representative_ids),
        },
        "cluster_map": cluster_map,
        "representative_ids": representative_ids,
        "classified_clusters": classified,
        "articles": enriched,
        "official_documents": official_documents,
        "parsed_documents": parsed_documents,
        "industry_documents": industry_documents,
        "structured_signals": structured_signals,
        "skipped_items": skipped_items,
    }


def _preprocess_route(article: dict[str, Any]) -> str:
    source_type = _source_type(article)
    source_name = _source_name(article)

    if source_type in RELEVANCE_SOURCE_TYPES:
        return "relevance"

    if source_type in OFFICIAL_DOCUMENT_SOURCE_TYPES:
        return "official_document"

    if source_type in PARSED_DOCUMENT_SOURCE_TYPES:
        return "parsed_document"

    if source_type == "trend_report" and source_name in INDUSTRY_DOCUMENT_SOURCE_NAMES:
        return "industry_document"

    if source_type == "trend_report":
        return "industry_document"

    if source_type in STRUCTURED_SIGNAL_SOURCE_TYPES:
        return "structured_signal"

    return "unsupported"


def _mark_official_document(article: dict[str, Any]) -> dict[str, Any]:
    item = dict(article)
    item["processing_status"] = "PREPROCESSED_OFFICIAL_DOCUMENT"
    item["document_scope"] = "company_official"
    item["matched_companies"] = _normalize_company(item.get("company"))
    item["matched_sectors"] = _matched_sectors_from_article(item)
    item["preprocess_note"] = (
        "official 문서는 회사별 공식 원문으로 보존. "
        "기사 relevance/dedup/classification 단계는 생략하고 추후 동향 분석에서 사용"
    )
    return item


def _mark_parsed_document(article: dict[str, Any]) -> dict[str, Any]:
    item = dict(article)
    source_type = _source_type(item)

    item["processing_status"] = "PREPROCESSED_PARSED_DOCUMENT"
    item["document_scope"] = "company_document"
    item["preprocess_note"] = (
        f"{source_type} 문서는 parser quality check 후 보존. "
        "기사 relevance/dedup/classification 단계는 생략"
    )
    item["matched_companies"] = _normalize_company(item.get("company"))
    item["matched_sectors"] = _matched_sectors_from_article(item)
    return item


def _mark_industry_document(article: dict[str, Any]) -> dict[str, Any]:
    item = dict(article)
    parser_result = ParserAgent().parse_article(item)
    item["processing_status"] = "PREPROCESSED_INDUSTRY_DOCUMENT"
    item["document_scope"] = "industry_trend"
    item["parser_result"] = parser_result
    item["matched_sectors"] = _matched_sectors_from_article(item)
    item["preprocess_note"] = (
        "산업 동향 문서는 기사 relevance/signal 축약 없이 추후 본문 분석 대상으로 보존"
    )
    return item


def _mark_structured_signal(article: dict[str, Any]) -> dict[str, Any]:
    item = dict(article)
    source_type = _source_type(item)
    item["processing_status"] = "PREPROCESSED_STRUCTURED_SIGNAL"
    item["signal_scope"] = source_type
    item["matched_companies"] = _normalize_company(item.get("company"))
    item["preprocess_note"] = (
        f"{source_type} 데이터는 기사/문서가 아닌 구조화 신호로 보존. "
        "급변/급증 탐지 및 종합 분석 단계에서 사용"
    )
    return item


def _mark_unsupported_source(article: dict[str, Any]) -> dict[str, Any]:
    item = dict(article)
    source_type = _source_type(item) or "unknown"
    item["processing_status"] = "SKIPPED_PREPROCESS_UNSUPPORTED_SOURCE"
    item["skip_reason"] = (
        f"{source_type} source_type은 현재 전처리 대상이 아님. "
        "급변/급증 탐지 단계에서 별도 처리 예정"
    )
    return item


def _matched_sectors_from_article(article: dict[str, Any]) -> list[str]:
    extra = article.get("extra") or article.get("metadata") or {}
    sectors = extra.get("matched_sectors")
    if isinstance(sectors, list) and sectors:
        return [str(sector) for sector in sectors if sector]

    return []


def _source_type(article: dict[str, Any]) -> str:
    return str(article.get("source_type") or "").strip().lower()


def _source_name(article: dict[str, Any]) -> str:
    return str(article.get("source_name") or "").strip().lower()


def _mark_representatives(
    articles: list[dict[str, Any]],
    cluster_map: dict[int, list[int]],
    representative_ids: list[int],
) -> None:
    representative_set = set(representative_ids)
    by_id = {int(article["preprocess_id"]): article for article in articles}

    for cluster_id, article_ids in cluster_map.items():
        for article_id in article_ids:
            article = by_id.get(int(article_id))
            if article is None:
                continue
            article["cluster_id"] = cluster_id
            article["is_representative"] = int(article_id) in representative_set


def _classify(
    articles: list[dict[str, Any]],
    cluster_map: dict[int, list[int]],
    representative_ids: list[int],
) -> list[dict[str, Any]]:
    by_id = {int(article["preprocess_id"]): article for article in articles}
    representative_set = set(representative_ids)
    classified: list[dict[str, Any]] = []

    for cluster_id, article_ids in cluster_map.items():
        rep_id = next(
            (article_id for article_id in article_ids if article_id in representative_set),
            None,
        )
        if rep_id is None:
            continue

        cluster_articles = [by_id[article_id] for article_id in article_ids if article_id in by_id]
        classified.append(
            classify_preprocessed_cluster(
                cluster_id=cluster_id,
                representative_id=rep_id,
                cluster_articles=cluster_articles,
                company=_first_company(by_id[rep_id]),
            )
        )

    return classified


def _normalize_company(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value if item]

    if isinstance(value, tuple):
        return [str(item) for item in value if item]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass
        return [value] if value else []

    return []


def _first_company(article: dict[str, Any]) -> str:
    company = _normalize_company(article.get("company"))
    return company[0] if company else ""


def _save_result(result: dict[str, Any], output_dir: Path, source_label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"preprocess_{source_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return path


def _source_label(args: argparse.Namespace) -> str:
    if args.input:
        label = Path(args.input).stem
    else:
        label = "crawler_results"

    if args.source_type:
        suffix = "_".join(source_type.strip().lower() for source_type in args.source_type)
        if suffix:
            label = f"{label}_{suffix}"

    return label


def _run() -> None:
    args = _parse_args()

    articles = _load_articles(args)
    if args.input:
        articles = _filter_by_source_type(articles, args.source_type)
    result = _preprocess(articles)
    output_path = _save_result(result, Path(args.output_dir), _source_label(args))

    counts = result["counts"]
    print("\n" + "=" * 78)
    print("전처리 결과")
    print("=" * 78)
    print(f"  raw:             {counts['raw']}건")
    print(f"  relevant:        {counts['relevant']}건")
    print(f"  official_docs:   {counts['official_documents']}건")
    print(f"  parsed_docs:     {counts['parsed_documents']}건")
    print(f"  skipped:         {counts['skipped']}건")
    print(f"  industry_docs:   {counts['industry_documents']}건")
    print(f"  structured:      {counts['structured_signals']}건")
    print(f"  clusters:        {counts['clusters']}개")
    print(f"  representatives: {counts['representatives']}건")
    print(f"  output:          {output_path}")

    if result["classified_clusters"]:
        print("\n  대표 클러스터")
        for cluster in result["classified_clusters"][:10]:
            print(
                f"    [{cluster['cluster_id']}] {cluster['sector']} / "
                f"{cluster['event_type']} / {cluster['exposure_band']} | "
                f"{cluster.get('title', '')[:70]}"
            )


if __name__ == "__main__":
    _run()
