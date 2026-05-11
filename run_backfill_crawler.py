"""Cursor 기반 backfill crawler 실행 스크립트."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_backfill_crawler")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AXIS backfill crawler")
    parser.add_argument(
        "--env",
        choices=["local", "cloud"],
        default=None,
        help="DB 프로파일. .env.{profile} 파일이 있으면 로드, 없으면 프로세스 env 사용.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="실행할 source 이름. 여러 번 지정 가능. 생략하거나 all이면 전체.",
    )
    parser.add_argument(
        "--cursor-date",
        default=None,
        help="상태 테이블에 cursor가 없을 때 사용할 초기 cursor 날짜 YYYY-MM-DD. 기본 오늘.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="source별 max_windows_per_run을 일시적으로 덮어쓴다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB 저장과 crawl_runs/crawl_cursors 갱신 없이 window 실행 흐름만 확인한다.",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="crawl_runs/crawl_cursors를 사용하지 않는다. raw_articles 저장은 유지된다.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="각 backfill window 저장 후 전처리를 실행하지 않는다.",
    )
    parser.add_argument(
        "--full-pipeline-after-window",
        action="store_true",
        help=(
            "각 backfill window 저장 후 issue_card/evidence/Qdrant까지 "
            "전체 파이프라인을 실행한다."
        ),
    )
    parser.add_argument(
        "--init-state-schema",
        action="store_true",
        help="현재 연결 DB에 crawl_runs/crawl_cursors 테이블을 생성한다. 로컬 검증용.",
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="현재 crawl_cursors와 최근 crawl_runs를 출력하고 종료한다.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    process_after_window = "none" if args.skip_preprocess else "preprocess"
    if args.full_pipeline_after_window:
        process_after_window = "full"
    if process_after_window != "none" and (args.dry_run or args.no_state):
        raise SystemExit(
            "window 후처리는 --dry-run/--no-state와 함께 사용할 수 없습니다."
        )

    from src.config.env_loader import load_profile

    profile = load_profile(args.env)
    log.info("실행 프로파일: %s", profile)

    from src.crawler.backfill_config import resolve_backfill_sources
    from src.crawler.backfill_runner import BackfillRunner
    from src.db.crawl_state_store import (
        ensure_backfill_state_schema,
        list_cursors,
        list_recent_runs,
    )

    if args.init_state_schema:
        ensure_backfill_state_schema()
        log.info("backfill 상태 테이블 생성/확인 완료")

    if args.show_state:
        _print_state(list_cursors(), list_recent_runs())
        return

    cursor_date = (
        datetime.strptime(args.cursor_date, "%Y-%m-%d").date()
        if args.cursor_date
        else None
    )
    source_names = args.source
    configs = resolve_backfill_sources(source_names)
    runner = BackfillRunner(
        initial_cursor_date=cursor_date,
        persist=not args.dry_run,
        use_state=not (args.dry_run or args.no_state),
        process_after_window=process_after_window,
    )

    print("\n" + "=" * 78)
    print("AXIS backfill crawler")
    print("=" * 78)
    print(f"profile={profile} sources={[config.source_name for config in configs]}")
    print(f"persist={not args.dry_run} use_state={not (args.dry_run or args.no_state)}")
    print(f"process_after_window={process_after_window}")

    for config in configs:
        summaries = await runner.run_source(config, max_windows_override=args.max_windows)
        for summary in summaries:
            print(
                f"{summary.status:7s} {summary.source_name:16s} "
                f"{summary.window_start}~{summary.window_end} "
                f"inserted={summary.inserted_count} skipped={summary.skipped_count}"
            )
            if summary.error_message:
                print(f"  error={summary.error_message}")


def _print_state(cursors: list[dict], runs: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("crawl_cursors")
    print("=" * 78)
    if not cursors:
        print("(empty)")
    for cursor in cursors:
        print(
            f"{cursor['source_name']:16s} cursor={cursor['cursor_date']} "
            f"until={cursor['until_date']} window={cursor['window_days']} "
            f"max={cursor['max_windows_per_run']} enabled={cursor['enabled']}"
        )

    print("\n" + "=" * 78)
    print("recent crawl_runs")
    print("=" * 78)
    if not runs:
        print("(empty)")
    for run in runs:
        print(
            f"{str(run['id'])[:8]} {run['status']:7s} {run['source_name']:16s} "
            f"{run['window_start']}~{run['window_end']} "
            f"inserted={run['inserted_count']} skipped={run['skipped_count']}"
        )
        if run.get("error_message"):
            print(f"  error={run['error_message']}")


if __name__ == "__main__":
    asyncio.run(_main())
