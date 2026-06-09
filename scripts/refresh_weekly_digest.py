"""K2 CronJob entrypoint — peer 별 weekly digest 갱신 (card_news 7일).

결과: peer_companies.peer_plus_payload['weekly_digest'] JSONB.

사용법:
    uv run python scripts/refresh_weekly_digest.py [--company samsung_sds] [--no-llm]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("refresh_weekly_digest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh peer weekly digests from card_news.")
    parser.add_argument("--company", action="append", default=None)
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.config.env_loader import load_profile

    preserve_database_url = os.environ.get("DATABASE_URL")
    load_profile(args.env)
    if preserve_database_url:
        os.environ["DATABASE_URL"] = preserve_database_url

    from src.agents.context.weekly_digest_agent import WeeklyDigestAgent
    from src.config.companies import COMPANY_IDS

    targets = args.company or list(COMPANY_IDS)
    agent = WeeklyDigestAgent()

    for company_id in targets:
        log.info("→ %s weekly digest 시작", company_id)
        try:
            result = agent.run(
                peer_id=company_id,
                lookback_days=args.lookback_days,
                use_llm=not args.no_llm,
            )
            agent.persist(company_id, result)
        except Exception as exc:  # noqa: BLE001
            log.exception("weekly digest 실패 | company=%s error=%s", company_id, exc)
            continue
        if result.get("skipped"):
            log.info("skip | company=%s reason=%s cards=%s", company_id, result.get("reason"), result.get("card_count"))
        else:
            digest = result.get("digest") or {}
            log.info(
                "완료 | company=%s week=%s cards=%s",
                company_id,
                digest.get("week_iso"),
                result.get("card_count"),
            )


if __name__ == "__main__":
    main()
