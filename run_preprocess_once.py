"""DB 저장 없이 크롤링/JSONL 결과를 전처리까지 실행한다.

사용법:
  uv run python run_preprocess_once.py --input src/crawler/crawler_results/ir_20260506.jsonl
  uv run python run_preprocess_once.py --source ir --company samsung_sds
  uv run python run_preprocess_once.py --source naver_datalab

이 runner는 로컬 확인용이다. DB 기반 agent를 호출하지 않고 같은 config/규칙을 사용해
credibility, relevance, dedup, classification 결과를 JSON으로 저장한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.credibility_agent import compute_credibility_score
from src.config.companies import COMPANY_ALIASES, CORP_CODES
from src.config.env_loader import load_profile
from src.config.sectors import match_sectors, primary_sector
from src.crawler.dart_crawler import DartCrawler
from src.crawler.ir_crawler import IRCrawler
from src.crawler.job_crawler import JobCrawler
from src.crawler.keyword_crawler import KeywordCrawler
from src.crawler.parsers.link_check import LinkChecker
from src.crawler.research_crawler import NaverResearchCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_preprocess")

PEER_SOURCES = ["dart", "jobs", "ir", "naver_research"]
INDUSTRY_SOURCES = ["naver_datalab"]
ALL_SOURCES = PEER_SOURCES + INDUSTRY_SOURCES
DEFAULT_OUTPUT_DIR = Path("src/crawler/crawler_results/preprocessed")

SOURCE_TYPE_CREDIBILITY_GRADES: list[tuple[float, str]] = [
    (0.85, "High"),
    (0.60, "Medium"),
    (0.40, "Low"),
    (0.00, "Unverified"),
]

EVENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "partnership": ["협약", "업무협약", "MOU", "파트너십", "제휴", "협력", "공동사업"],
    "ma": ["인수", "합병", "인수합병", "M&A", "지분 인수", "지분 투자", "투자 유치"],
    "personnel": ["채용", "인사", "임원", "대표이사", "CEO", "선임", "영입", "조직개편"],
    "tech_release": ["출시", "공개", "개발", "고도화", "플랫폼", "솔루션", "서비스", "기술"],
    "regulation": ["규제", "정책", "법안", "가이드라인", "정부", "인증", "표준"],
    "contract": ["수주", "계약", "선정", "공급", "구축 사업", "사업자", "우선협상대상자"],
    "financial": ["매출", "영업이익", "실적", "분기", "연간", "흑자", "적자", "투자계획"],
    "expansion": ["해외 진출", "글로벌", "시장 확대", "사업 확대", "센터 설립", "법인 설립"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB 없이 전처리 결과 확인")
    parser.add_argument("--input", default=None, help="전처리할 crawler JSONL 파일")
    parser.add_argument("--source", choices=ALL_SOURCES, default=None, help="직접 실행할 크롤러")
    parser.add_argument(
        "--company",
        choices=list(COMPANY_ALIASES),
        default=None,
        help="수집할 회사. naver_datalab은 생략",
    )
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _build_crawler(source: str, company: str | None) -> Any:
    if source == "naver_datalab":
        return KeywordCrawler()

    if company is None:
        raise ValueError(f"{source} 크롤러는 --company가 필요합니다.")

    if source == "dart":
        return DartCrawler(
            peer_id=company,
            corp_code=CORP_CODES.get(company),
            corp_names=COMPANY_ALIASES[company],
        )
    if source == "jobs":
        return JobCrawler(peer_id=company)
    if source == "ir":
        return IRCrawler(peer_id=company)
    if source == "naver_research":
        return NaverResearchCrawler(peer_id=company)

    raise ValueError(f"지원하지 않는 source입니다: {source}")


async def _load_or_crawl(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input:
        return _read_jsonl(Path(args.input))

    if not args.source:
        raise ValueError("--input 또는 --source 중 하나는 필요합니다.")

    crawler = _build_crawler(args.source, args.company)
    articles = await crawler.crawl()
    articles, rejected = await LinkChecker().filter_accessible(articles)
    log.info(
        "크롤 완료 | source=%s valid=%d rejected=%d",
        args.source,
        len(articles),
        len(rejected),
    )
    return [_article_to_dict(article) for article in articles]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    log.info("JSONL 로드 완료 | path=%s rows=%d", path, len(rows))
    return rows


def _article_to_dict(article: Any) -> dict[str, Any]:
    if hasattr(article, "to_common_dict"):
        return article.to_common_dict()

    return {
        "id": getattr(article, "id", None),
        "source_type": getattr(article, "source_type", "news"),
        "source_name": getattr(article, "source_name", None),
        "publisher": getattr(article, "publisher", None),
        "title": getattr(article, "title", ""),
        "content": getattr(article, "content", None),
        "url": getattr(article, "url", ""),
        "url_hash": getattr(article, "url_hash", ""),
        "published_at": _iso(getattr(article, "published_at", None)),
        "collected_at": _iso(getattr(article, "collected_at", None)),
        "company": list(getattr(article, "company", []) or []),
        "language": getattr(article, "language", "ko"),
        "content_type": getattr(article, "content_type", "unknown"),
        "crawl_status": getattr(article, "crawl_status", "success"),
        "error_message": getattr(article, "error_message", None),
        "extra": dict(getattr(article, "extra", {}) or {}),
    }


def _preprocess(articles: list[dict[str, Any]]) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for index, article in enumerate(articles, start=1):
        item = dict(article)
        item["preprocess_id"] = index

        valid, reason = _validate_article(item)
        if not valid:
            item["processing_status"] = "SKIPPED_INVALID_SOURCE"
            item["skip_reason"] = reason
            invalid.append(item)
            continue

        credibility_score = compute_credibility_score(item.get("source_type"))
        item["credibility_score"] = credibility_score
        item["credibility_grade"] = _credibility_grade(credibility_score)

        relevance = _analyze_relevance(item)
        item.update(relevance)

        if _is_relevant(item):
            item["processing_status"] = "PREPROCESSED"
            enriched.append(item)
        else:
            item["processing_status"] = "SKIPPED_RELEVANCE"
            invalid.append(item)

    cluster_map, representative_ids = _deduplicate(enriched)
    classified = _classify(enriched, cluster_map, representative_ids)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "raw": len(articles),
            "valid_after_credibility": len(articles) - len(
                [row for row in invalid if row.get("skip_reason")]
            ),
            "relevant": len(enriched),
            "skipped": len(invalid),
            "clusters": len(cluster_map),
            "representatives": len(representative_ids),
        },
        "cluster_map": cluster_map,
        "representative_ids": representative_ids,
        "classified_clusters": classified,
        "articles": enriched,
        "skipped_articles": invalid,
    }


def _validate_article(article: dict[str, Any]) -> tuple[bool, str | None]:
    if article.get("crawl_status") == "failed":
        return False, "crawl_failed"
    if not article.get("url"):
        return False, "empty_url"
    if not article.get("title"):
        return False, "empty_title"
    return True, None


def _credibility_grade(score: float) -> str:
    for threshold, grade in SOURCE_TYPE_CREDIBILITY_GRADES:
        if score >= threshold:
            return grade
    return "Unverified"


def _analyze_relevance(article: dict[str, Any]) -> dict[str, Any]:
    title = article.get("title") or ""
    content = article.get("content") or ""
    company = _normalize_company(article.get("company"))
    source_type = article.get("source_type")
    text = f"{title} {content}"

    matched_companies = _match_companies(text, company)
    matched_sectors = match_sectors(text)
    has_company = bool(matched_companies)
    has_sector = bool(matched_sectors and matched_sectors != ["other"])

    if source_type in {"dart", "ir", "official"} and company:
        return _relevance_result(
            "relevant",
            0.85,
            matched_companies,
            matched_sectors,
            "공식성 자료",
        )

    if not has_company and not has_sector:
        return _relevance_result(
            "irrelevant",
            0.10,
            matched_companies,
            matched_sectors,
            "대상 company와 sector 후보가 모두 감지되지 않음",
        )

    if not has_company and source_type not in {"search_trend", "trend_report"}:
        return _relevance_result(
            "irrelevant",
            0.25,
            matched_companies,
            matched_sectors,
            "sector 후보는 있으나 대상 company가 본문에서 확인되지 않음",
        )

    if has_company and has_sector:
        return _relevance_result(
            "relevant",
            0.70 if len(content) >= 500 else 0.62,
            matched_companies,
            matched_sectors,
            "company와 sector 후보가 모두 감지됨",
        )

    return _relevance_result(
        "uncertain",
        0.60,
        matched_companies,
        matched_sectors,
        "규칙 기반으로 관련성 보류 통과",
    )


def _relevance_result(
    label: str,
    score: float,
    companies: list[str],
    sectors: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "relevance_label": label,
        "relevance_score": score,
        "matched_companies": companies,
        "matched_sectors": sectors,
        "relevance_reason": reason,
    }


def _is_relevant(article: dict[str, Any]) -> bool:
    label = article.get("relevance_label")
    score = float(article.get("relevance_score") or 0.0)
    return label in {"relevant", "uncertain"} and score >= 0.60


def _deduplicate(articles: list[dict[str, Any]]) -> tuple[dict[int, list[int]], list[int]]:
    groups: dict[str, list[dict[str, Any]]] = {}

    for article in articles:
        key = _dedup_key(article)
        groups.setdefault(key, []).append(article)

    cluster_map: dict[int, list[int]] = {}
    representative_ids: list[int] = []

    for cluster_id, group in enumerate(groups.values()):
        ids = [int(article["preprocess_id"]) for article in group]
        cluster_map[cluster_id] = ids
        representative = max(
            group,
            key=lambda item: (
                float(item.get("relevance_score") or 0.0),
                float(item.get("credibility_score") or 0.0),
                len(item.get("content") or ""),
            ),
        )
        representative_ids.append(int(representative["preprocess_id"]))

        for article in group:
            article["cluster_id"] = cluster_id
            article["is_representative"] = article is representative

    return cluster_map, representative_ids


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

        rep = by_id[rep_id]
        text = f"{rep.get('title') or ''} {rep.get('content') or ''}"
        sector = primary_sector(text)
        sectors = match_sectors(text)
        event_type = _classify_event_type(text)
        exposure = _compute_exposure([by_id[article_id] for article_id in article_ids])

        classified.append(
            {
                "cluster_id": cluster_id,
                "representative_id": rep_id,
                "company": _first_company(rep),
                "sector": sector,
                "sectors": sectors,
                "event_type": event_type,
                **exposure,
                "title": rep.get("title"),
                "url": rep.get("url"),
            }
        )

    return classified


def _compute_exposure(cluster_articles: list[dict[str, Any]]) -> dict[str, Any]:
    cluster_size = len(cluster_articles)
    cluster_size_score = min(cluster_size / 5, 1.0)
    credibility_max = max((article.get("credibility_score") or 0.0) for article in cluster_articles)
    company_mention_count = max(
        len(article.get("matched_companies") or [])
        for article in cluster_articles
    )
    company_mention_score = min(company_mention_count, 1.0)
    score = 0.50 * cluster_size_score + 0.30 * credibility_max + 0.20 * company_mention_score

    if score >= 0.65:
        band = "high"
    elif score >= 0.40:
        band = "medium"
    else:
        band = "low"

    return {
        "exposure_score": round(score, 3),
        "exposure_band": band,
        "signals": {
            "cluster_size": cluster_size,
            "company_mention_count": company_mention_count,
            "credibility_max": round(float(credibility_max), 3),
        },
    }


def _classify_event_type(text: str) -> str:
    matched: list[tuple[str, int]] = []

    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in text)
        if count > 0:
            matched.append((event_type, count))

    if not matched:
        return "tech_release"

    matched.sort(key=lambda item: item[1], reverse=True)
    return matched[0][0]


def _dedup_key(article: dict[str, Any]) -> str:
    url_hash = article.get("url_hash")
    if url_hash:
        return f"url:{url_hash}"

    title = _compact(article.get("title") or "")
    if title:
        return f"title:{title[:80]}"

    return f"id:{article.get('preprocess_id')}"


def _match_companies(text: str, company: list[str]) -> list[str]:
    compact_text = _compact(text)
    matched: list[str] = []

    for company_id in company:
        aliases = COMPANY_ALIASES.get(company_id, [company_id])
        if any(_compact(alias) in compact_text for alias in aliases):
            matched.append(company_id)

    return _dedupe_keep_order(matched)


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


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _save_result(result: dict[str, Any], output_dir: Path, source_label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"preprocess_{source_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return path


def _source_label(args: argparse.Namespace) -> str:
    if args.input:
        return Path(args.input).stem
    if args.company:
        return f"{args.source}_{args.company}"
    return str(args.source)


async def _run() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    log.info("실행 프로파일: %s", profile)

    articles = await _load_or_crawl(args)
    result = _preprocess(articles)
    output_path = _save_result(result, Path(args.output_dir), _source_label(args))

    counts = result["counts"]
    print("\n" + "=" * 78)
    print("전처리 결과")
    print("=" * 78)
    print(f"  raw:             {counts['raw']}건")
    print(f"  relevant:        {counts['relevant']}건")
    print(f"  skipped:         {counts['skipped']}건")
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
    asyncio.run(_run())
