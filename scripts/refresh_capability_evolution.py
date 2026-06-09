"""W4-3 CronJob entrypoint — peer 별 capability_evolution narrative 월 1회 갱신.

결과: peer_companies.peer_plus_payload['capability_evolution'] JSONB (append-only windows).

사용법:
    uv run python scripts/refresh_capability_evolution.py [--company samsung_sds] [--no-llm]
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger("refresh_capability_evolution")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh peer capability evolution narratives.")
    parser.add_argument("--company", action="append", default=None)
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="LLM 없이 deterministic aggregation 으로 windows 를 생성한다.",
    )
    parser.add_argument("--lookback-quarters", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.config.env_loader import load_profile

    load_profile(args.env)

    from src.agents.context.capability_evolution_agent import CapabilityEvolutionAgent
    from src.config.companies import COMPANY_IDS

    targets = args.company or list(COMPANY_IDS)
    agent = CapabilityEvolutionAgent()

    for company_id in targets:
        log.info("→ %s capability evolution 시작", company_id)
        try:
            result = agent.run(
                peer_id=company_id,
                lookback_quarters=args.lookback_quarters,
                use_llm=not args.no_llm,
            )
            agent.persist(company_id, result)
        except Exception as exc:  # noqa: BLE001
            log.exception("capability evolution 실패 | company=%s error=%s", company_id, exc)
            continue
        if result.get("skipped"):
            log.info("skip | company=%s reason=%s", company_id, result.get("reason"))
        else:
            log.info(
                "완료 | company=%s windows=%s signals=%s",
                company_id,
                len(result.get("windows") or []),
                result.get("signal_count"),
            )


if __name__ == "__main__":
    main()
