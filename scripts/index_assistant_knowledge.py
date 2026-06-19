# 작성일: 2026-06-08
# 작성자: 박진
# 변경이력:
#   2026-06-08 박진 — 챗봇 에이전트/어시스턴트 RAG 도입과 함께 지식 색인 스크립트 추가
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Index assistant knowledge records into Qdrant.")
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--limit-per-source", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--source-type",
        action="append",
        default=None,
        help=(
            "색인할 source_type. 콤마 구분 또는 반복 지정 가능. "
            "기본값은 integrated_issue, card_news_analysis, peer_profile 전체."
        ),
    )
    parser.add_argument(
        "--exclude-source-type",
        action="append",
        default=None,
        help="색인에서 제외할 source_type. 예: --exclude-source-type card_news_analysis",
    )
    parser.add_argument(
        "--delete-source-type",
        action="append",
        default=None,
        help=(
            "색인 대신 해당 source_type의 assistant knowledge 벡터만 삭제. "
            "예: --delete-source-type card_news_analysis"
        ),
    )
    parser.add_argument(
        "--purge-existing",
        action="store_true",
        help="색인 전에 선택된 source_type의 기존 assistant knowledge 벡터를 삭제.",
    )
    args = parser.parse_args()

    from src.config.env_loader import load_profile
    from src.db.postgres import reconfigure_from_env
    from src.rag.assistant_knowledge_index import (
        delete_assistant_knowledge,
        index_assistant_knowledge,
    )

    load_profile(args.env)
    reconfigure_from_env()
    if args.delete_source_type:
        result = delete_assistant_knowledge(source_types=args.delete_source_type)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    purge_result = None
    if args.purge_existing:
        purge_result = delete_assistant_knowledge(
            source_types=args.source_type,
            exclude_source_types=args.exclude_source_type,
        )
    result = index_assistant_knowledge(
        limit_per_source=args.limit_per_source,
        batch_size=args.batch_size,
        source_types=args.source_type,
        exclude_source_types=args.exclude_source_type,
    )
    if purge_result is not None:
        result["purge"] = purge_result
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
