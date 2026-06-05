"""Run one cluster through the analysis/card-news pipeline and save JSON.

This script intentionally patches DB write functions, so it reads common DB
data and calls the agents, but does not insert integrated_issues/card_news.
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
    parser.add_argument("--env", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--representative-id", type=int, default=None)
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: dryruns/card_news_dry_run_<cluster>_<timestamp>.json",
    )
    return parser.parse_args()


def _default_output_path(cluster_id: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("dryruns") / f"card_news_dry_run_cluster_{cluster_id}_{stamp}.json"


def main() -> None:
    from src.config.env_loader import load_profile

    args = _parse_args()
    profile = load_profile(args.env)
    from src.pipeline.analysis_pipeline import AnalysisPipelineRunner, list_cluster_article_ids

    output_path = Path(args.out) if args.out else _default_output_path(args.cluster_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_article_ids = list_cluster_article_ids(args.cluster_id)
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
        result = AnalysisPipelineRunner().run_cluster(
            cluster_id=args.cluster_id,
            representative_id=args.representative_id,
            cluster_article_ids=cluster_article_ids,
            save_card=False,
        )

    payload = {
        "dry_run": {
            "created_at": datetime.now(UTC).isoformat(),
            "env_profile": profile,
            "cluster_id": args.cluster_id,
            "representative_id": args.representative_id,
            "cluster_article_ids": cluster_article_ids,
            "db_writes_patched": True,
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
