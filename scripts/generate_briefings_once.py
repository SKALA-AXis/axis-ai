"""Generate and save daily/weekly/monthly briefing reports for one anchor date.

Usage:
  uv run python scripts/generate_briefings_once.py --env local --anchor-date 2026-06-12 --replace
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("generate_briefings_once")

BriefingType = Literal["daily", "weekly", "monthly"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate briefing reports and save them.")
    parser.add_argument("--env", choices=["local", "cloud"], default="local")
    parser.add_argument(
        "--anchor-date",
        type=date.fromisoformat,
        required=True,
        help=(
            "기준일(YYYY-MM-DD). daily는 해당 일, weekly는 주 시작일부터 기준일까지, "
            "monthly는 월초부터 기준일까지 생성."
        ),
    )
    parser.add_argument(
        "--type",
        action="append",
        choices=["daily", "weekly", "monthly"],
        default=None,
        help="생성할 브리핑 타입. 여러 번 지정 가능. 기본은 daily/weekly/monthly 전체.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="기존 저장본을 재사용하지 않고 새로 생성해서 같은 id에 UPSERT.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="DB 저장 없이 생성 결과만 확인.",
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM copy refinement.")
    return parser.parse_args()


def _load_env(profile: str) -> None:
    from src.config.env_loader import load_profile

    preserved_database_url = os.environ.get("DATABASE_URL")
    load_profile(profile)
    if preserved_database_url:
        os.environ["DATABASE_URL"] = preserved_database_url


async def _run(args: argparse.Namespace) -> None:
    from src.agents.briefing_generation_agent import BriefingGenerationAgent

    agent = BriefingGenerationAgent()
    briefing_types: list[BriefingType] = args.type or ["daily", "weekly", "monthly"]
    for briefing_type in briefing_types:
        result = await agent.generate(
            briefing_type=briefing_type,
            anchor_date=args.anchor_date,
            limit=args.limit,
            save=not args.no_save,
            use_mock=False,
            refine_display_copy=not args.no_llm,
            reuse_saved=not args.replace,
        )
        related_cards = result.get("related_card_ids") or result.get("source_card_ids") or []
        log.info(
            "briefing generated | type=%s id=%s title=%s saved=%s cards=%d",
            briefing_type,
            result.get("id"),
            result.get("title"),
            not args.no_save,
            len(related_cards) if isinstance(related_cards, list) else 0,
        )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    _load_env(args.env)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
