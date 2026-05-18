"""SK AX self profile agent."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Literal

from src.agents.peer_profile_agent import (
    PeerProfileAgent,
    build_selected_sector_config,
    emit_profile_result,
    load_runtime_env,
    print_selected_sector_config,
)


class SKAXProfileAgent(PeerProfileAgent):
    """Build the self-company profile used as SK AX judgement criteria."""

    role: Literal["self"] = "self"

    def build_profile(
        self,
        company_id: str = "sk_ax",
        lookback_days: int | None = None,
    ) -> dict[str, Any]:
        return super().build_profile(company_id=company_id, lookback_days=lookback_days)


def run_skax_profile_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m src.agents.skax_profile_agent``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    load_runtime_env()
    args = _parse_skax_cli_args(argv)
    print_selected_sector_config(build_selected_sector_config())
    profile = SKAXProfileAgent().build_profile(lookback_days=args.lookback_days)
    emit_profile_result(company_id="sk_ax", profile=profile, dry_run=args.dry_run)
    return 0


def _parse_skax_cli_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SK AX self profile JSON.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help=(
            "최신성 판단 focus window 수동 override. "
            "생략하면 DART/IR 및 전체 자료 날짜 흐름으로 자동 산정."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 저장 없이 콘솔에 JSON 출력",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_skax_profile_cli())


__all__ = ["SKAXProfileAgent", "run_skax_profile_cli"]
