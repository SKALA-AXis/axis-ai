"""DB 기준 크롤링 + 전처리를 한 번에 순차 실행한다.

사용법:
  uv run python run_all_once.py
  uv run python run_all_once.py --track all
  uv run python run_all_once.py --track c
  uv run python run_all_once.py --env cloud --track all
  uv run python run_all_once.py --company samsung_sds --company nvidia
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AXIS DB 크롤링 후 전처리까지만 1회 순차 실행")
    parser.add_argument(
        "--track",
        choices=["a", "b", "c", "d", "all"],
        default="all",
        help="크롤링할 트랙. 기본은 all.",
    )
    parser.add_argument(
        "--env",
        choices=["local", "cloud"],
        default="local",
        help=(
            "DB 프로파일. 기본은 local이며 "
            "run_crawler_once.py/run_pipeline_once.py에 동일하게 전달한다."
        ),
    )
    parser.add_argument(
        "--company",
        action="append",
        default=None,
        help="처리할 company id. 여러 번 지정 가능. 생략하면 전체 회사.",
    )
    parser.add_argument(
        "--news-hours",
        type=int,
        default=10,
        help="Track A 뉴스 수집 범위. 기본 10시간.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Track B/C 수집 기간. 지정 시 run_crawler_once.py에 전달한다.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Track B/C 수집 시작일 YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Track B/C 수집 종료일 YYYY-MM-DD.",
    )
    parser.add_argument(
        "--local-output",
        default=None,
        help="크롤 결과 JSON도 같이 저장할 디렉터리. run_crawler_once.py에만 전달한다.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="크롤링만 실행하고 DB 전처리는 건너뛴다.",
    )
    return parser.parse_args()


def _crawler_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(ROOT / "run_crawler_once.py"), "--track", args.track]
    _append_shared_args(cmd, args)

    if args.news_hours is not None:
        cmd.extend(["--news-hours", str(args.news_hours)])
    if args.lookback_days is not None:
        cmd.extend(["--lookback-days", str(args.lookback_days)])
    if args.start_date:
        cmd.extend(["--start-date", args.start_date])
    if args.end_date:
        cmd.extend(["--end-date", args.end_date])
    if args.local_output:
        cmd.extend(["--local-output", args.local_output])

    return cmd


def _preprocess_cmd(args: argparse.Namespace, collected_since: str) -> list[str]:
    cmd = [sys.executable, str(ROOT / "run_pipeline_once.py"), "--preprocess-only"]
    _append_shared_args(cmd, args)
    cmd.extend(["--collected-since", collected_since])
    return cmd


def _append_shared_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.env:
        cmd.extend(["--env", args.env])
    for company in args.company or []:
        cmd.extend(["--company", company])


def _run_step(label: str, cmd: list[str]) -> None:
    printable = " ".join(cmd)
    print("\n" + "=" * 78)
    print(f"{label} 실행")
    print("=" * 78)
    print(f"$ {printable}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = _parse_args()
    started_at = datetime.now(timezone.utc)
    collected_since = started_at.isoformat()
    _run_step("1/2 크롤러", _crawler_cmd(args))

    if args.skip_preprocess:
        print("\n--skip-preprocess 지정으로 DB 전처리 실행을 건너뜁니다.")
        return

    _run_step("2/2 DB 전처리", _preprocess_cmd(args, collected_since))


if __name__ == "__main__":
    main()
