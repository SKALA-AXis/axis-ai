"""Catch up RAW articles that were collected but missed preprocessing.

Scheduled ingestion triggers run preprocessing in an axis-ai background task.
If the pod restarts after raw_articles are inserted, those rows can stay RAW.
This script is the durable safety net: it re-runs preprocessing by track/source
type until there is nothing left or the configured batch budget is exhausted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.env_loader import load_profile  # noqa: E402
from src.preprocessing.classification import ClusterClassifier  # noqa: E402
from src.preprocessing.preprocessing import PreprocessingService  # noqa: E402
from src.preprocessing.relevance import RelevanceEvaluator  # noqa: E402

log = logging.getLogger("preprocess_raw_catchup")

TRACK_SOURCE_TYPE_GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "a": [
        ("a:news", ["news"]),
        ("a:official", ["official"]),
    ],
    "b": [
        ("b:market_data", ["market_data"]),
        ("b:securities_report", ["securities_report"]),
    ],
    "c": [
        ("c:job", ["job"]),
        ("c:official", ["official"]),
        ("c:company_site", ["company_site"]),
    ],
    "d": [
        ("d:dart", ["dart"]),
        ("d:ir", ["ir"]),
        ("d:search_trend", ["search_trend"]),
        ("d:trend_report", ["trend_report"]),
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAW 전처리 catch-up")
    parser.add_argument(
        "--env",
        choices=["local", "cloud"],
        default=None,
        help=".env.local 또는 .env.cloud 로드. k8s에서는 생략.",
    )
    parser.add_argument(
        "--track",
        choices=["a", "b", "c", "d", "all"],
        default="all",
        help="전처리할 트랙. all이면 A/B/C/D source_type 전체를 한 번씩 처리.",
    )
    parser.add_argument(
        "--company",
        action="append",
        default=None,
        help="특정 company id만 처리. 생략하면 전체.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="batch당 처리 row 수.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=4,
        help="최대 batch 수. 0이면 RAW가 없어질 때까지 실행.",
    )
    return parser.parse_args()


def _source_type_groups(track: str) -> list[tuple[str, list[str]]]:
    if track != "all":
        return TRACK_SOURCE_TYPE_GROUPS[track]

    groups: list[tuple[str, list[str]]] = []
    for track_groups in TRACK_SOURCE_TYPE_GROUPS.values():
        groups.extend(track_groups)
    return groups


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = _parse_args()
    if args.env:
        load_profile(args.env)

    total_raw = 0
    total_skipped = 0
    total_classified = 0
    total_analysis_metrics = 0
    total_analysis_signals = 0

    for group_name, source_types in _source_type_groups(args.track):
        enable_llm = "news" in source_types
        service = PreprocessingService(
            relevance_evaluator=RelevanceEvaluator(enable_llm=enable_llm),
            classifier=ClusterClassifier(enable_llm=enable_llm),
        )
        batch = 0

        while args.max_batches == 0 or batch < args.max_batches:
            batch += 1
            result = service.run(
                company=args.company or [],
                source_types=source_types,
                trigger_type=f"cron:preprocess-catchup:{group_name}",
                limit=args.limit,
            )
            raw_count = len(result.get("raw_article_ids", []))
            if raw_count == 0:
                log.info(
                    "RAW catch-up group 완료 | group=%s source_types=%s batches=%d",
                    group_name,
                    source_types,
                    batch - 1,
                )
                break

            total_raw += raw_count
            total_skipped += len(result.get("skipped_preprocess_ids", []))
            total_classified += len(result.get("classified_clusters", []))
            total_analysis_metrics += int(result.get("analysis_metric_count", 0))
            total_analysis_signals += int(result.get("analysis_signal_count", 0))
            log.info(
                (
                    "RAW catch-up batch 완료 | group=%s batch=%d raw=%d skipped=%d "
                    "classified=%d analysis_metrics=%d analysis_signals=%d"
                ),
                group_name,
                batch,
                raw_count,
                len(result.get("skipped_preprocess_ids", [])),
                len(result.get("classified_clusters", [])),
                result.get("analysis_metric_count", 0),
                result.get("analysis_signal_count", 0),
            )

    log.info(
        (
            "RAW catch-up 종료 | track=%s total_raw=%d "
            "total_skipped=%d total_classified=%d total_metrics=%d total_signals=%d"
        ),
        args.track,
        total_raw,
        total_skipped,
        total_classified,
        total_analysis_metrics,
        total_analysis_signals,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
