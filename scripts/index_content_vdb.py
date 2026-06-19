from __future__ import annotations

import argparse
import json
from typing import Any


def _split_values(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out or None


def _int_values(values: list[str] | None) -> list[int] | None:
    split = _split_values(values)
    if not split:
        return None
    out: list[int] = []
    for value in split:
        out.append(int(value))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror long RDB content into Qdrant axis_documents for LLM context."
    )
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument(
        "--source",
        action="append",
        choices=["raw_articles", "integrated_issues", "card_news"],
        help="색인할 소스. 반복 지정 가능. 기본값은 전체.",
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="색인 전 동일 source/kind 기존 청크를 삭제. 일반 백필은 기본 upsert만 수행.",
    )
    parser.add_argument("--raw-article-id", action="append", default=None)
    parser.add_argument("--integrated-issue-id", action="append", default=None)
    parser.add_argument("--card-news-id", action="append", default=None)
    parser.add_argument(
        "--raw-source-type",
        action="append",
        default=None,
        help="raw_articles.source_type 필터. 콤마 구분 또는 반복 지정 가능.",
    )
    args = parser.parse_args()

    from src.config.env_loader import load_profile
    from src.db.postgres import reconfigure_from_env
    from src.rag.content_index import (
        index_card_analysis_from_rdb,
        index_integrated_issues_from_rdb,
        index_raw_articles_from_rdb,
    )

    load_profile(args.env)
    reconfigure_from_env()

    sources = args.source or ["raw_articles", "integrated_issues", "card_news"]
    result: dict[str, Any] = {"sources": {}}
    if "raw_articles" in sources:
        result["sources"]["raw_articles"] = index_raw_articles_from_rdb(
            article_ids=_int_values(args.raw_article_id),
            source_types=_split_values(args.raw_source_type),
            limit=args.limit,
            offset=args.offset,
            replace_existing=args.replace_existing,
        )
    if "integrated_issues" in sources:
        result["sources"]["integrated_issues"] = index_integrated_issues_from_rdb(
            integrated_issue_ids=_split_values(args.integrated_issue_id),
            limit=args.limit,
            offset=args.offset,
            replace_existing=args.replace_existing,
        )
    if "card_news" in sources:
        result["sources"]["card_news"] = index_card_analysis_from_rdb(
            card_news_ids=_split_values(args.card_news_id),
            limit=args.limit,
            offset=args.offset,
            replace_existing=args.replace_existing,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
