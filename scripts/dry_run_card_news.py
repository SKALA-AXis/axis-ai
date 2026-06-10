"""Run one cluster through the analysis/card-news pipeline and save JSON.

By default this script patches DB write functions, so it reads DB data and
calls the agents, but does not insert integrated_issues/card_news. Pass
``--save-db`` to run the same pipeline with real DB writes enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Card news dry run for one raw_articles cluster")
    parser.add_argument(
        "--env",
        choices=["local", "cloud", "current"],
        default="cloud",
        help=(
            "dotenv profile to load. Use 'current' when DATABASE_URL/OPENAI_API_KEY "
            "are already exported and must not be overwritten by .env files."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--cluster-id", type=int)
    target.add_argument(
        "--raw-article-id",
        type=int,
        help="Run one raw_articles.id directly, even when cluster_id is empty.",
    )
    parser.add_argument("--representative-id", type=int, default=None)
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Actually insert/update integrated_issues and card_news. Default is dry-run only.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: dryruns/card_news_dry_run_<cluster>_<timestamp>.json",
    )
    return parser.parse_args()


def _default_output_path(
    *,
    cluster_id: int | None = None,
    raw_article_id: int | None = None,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if raw_article_id is not None:
        return Path("dryruns") / f"card_news_dry_run_article_{raw_article_id}_{stamp}.json"
    return Path("dryruns") / f"card_news_dry_run_cluster_{cluster_id}_{stamp}.json"


def main() -> None:
    from src.config.env_loader import load_profile

    args = _parse_args()
    profile = "current" if args.env == "current" else load_profile(args.env)
    from src.pipeline.analysis_pipeline import AnalysisPipelineRunner, list_cluster_article_ids

    output_path = (
        Path(args.out)
        if args.out
        else _default_output_path(
            cluster_id=args.cluster_id,
            raw_article_id=args.raw_article_id,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.raw_article_id is not None:
        cluster_article_ids = [args.raw_article_id]

        def run_pipeline() -> dict[str, Any]:
            return AnalysisPipelineRunner().run_raw_article(
                raw_article_id=args.raw_article_id,
                save_card=args.save_db,
            )

    else:
        cluster_article_ids = list_cluster_article_ids(args.cluster_id)

        def run_pipeline() -> dict[str, Any]:
            return AnalysisPipelineRunner().run_cluster(
                cluster_id=args.cluster_id,
                representative_id=args.representative_id,
                cluster_article_ids=cluster_article_ids,
                save_card=args.save_db,
            )

    if not cluster_article_ids:
        raise SystemExit(
            f"cluster_id={args.cluster_id} 에 속한 raw_articles가 없습니다. "
            "뉴스 카드 dry-run은 raw_articles.cluster_id가 있는 클러스터만 실행합니다."
        )
    if args.representative_id is not None and args.representative_id not in cluster_article_ids:
        raise SystemExit(
            f"representative_id={args.representative_id} 는 cluster_id={args.cluster_id} "
            f"기사 목록에 없습니다. cluster_article_ids={cluster_article_ids}"
        )

    if args.save_db:
        result = run_pipeline()
    else:
        with (
            patch(
                "src.pipeline.analysis_flow_graph.save_integrated_issue",
                return_value="DRYRUN-INTEGRATED-ISSUE-ID",
            ),
            patch(
                "src.pipeline.analysis_flow_graph.save_card_news",
                return_value="DRYRUN-CARD-NEWS-ID",
            ),
            patch("src.pipeline.analysis_flow_graph.save_pipeline_log"),
        ):
            result = run_pipeline()

    payload = {
        "dry_run": {
            "created_at": datetime.now(UTC).isoformat(),
            "env_profile": profile,
            "cluster_id": args.cluster_id,
            "raw_article_id": args.raw_article_id,
            "representative_id": args.representative_id,
            "cluster_article_ids": cluster_article_ids,
            "db_writes_patched": not args.save_db,
            "save_db": args.save_db,
        },
        "quick_view": _quick_view(result),
        "result": result,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"saved={output_path}")
    print(json.dumps(payload["quick_view"], ensure_ascii=False, indent=2, default=str))


def _quick_view(result: dict[str, Any]) -> dict[str, Any]:
    card = result.get("card_news") or {}
    validation = result.get("validation") or {}
    integrated_issue = result.get("integrated_issue") or {}
    analysis = result.get("analysis") or {}
    implication = result.get("implication") or {}
    return {
        "ok": result.get("ok"),
        "card_id": card.get("id"),
        "card_title": card.get("title"),
        "summary_line_count": len(card.get("summary_lines") or []),
        "integrated_fact_summary_count": len(integrated_issue.get("fact_summary") or []),
        "analysis_valid": analysis.get("is_valid_analysis"),
        "implication_valid": implication.get("is_valid_implication"),
        "validation_passed": validation.get("passed") or validation.get("pass"),
        "validation_reason": validation.get("reason"),
        "human_review_flags": result.get("human_review_flags") or [],
        "errors": result.get("errors") or [],
    }


if __name__ == "__main__":
    main()
