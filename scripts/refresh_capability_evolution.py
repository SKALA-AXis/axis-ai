"""W4-3 CronJob entrypoint — peer 별 capability evolution narrative 갱신.

매월 1일 03:00 KST 에 `axis-cron-capability-evolution` 가 실행. 모든 peer 회사에
대해 `CapabilityEvolutionAgent` 를 호출하고 결과를
`peer_companies.peer_plus_payload['capability_evolution']` JSONB 에 upsert.

사용법:
    uv run python scripts/refresh_capability_evolution.py [--peer samsung_sds] [--env cloud]
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger("refresh_capability_evolution")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh peer capability evolution narratives.")
    parser.add_argument(
        "--peer",
        action="append",
        default=None,
        help="peer_id. 여러 번 지정 가능. 생략하면 등록된 모든 peer 회사 처리.",
    )
    parser.add_argument(
        "--env",
        choices=["local", "cloud"],
        default=None,
        help="DB 프로파일 (.env.{profile}). 생략하면 프로세스 env 사용.",
    )
    parser.add_argument(
        "--lookback-quarters",
        type=int,
        default=4,
        help="피어 별로 가져올 분기 수 (default 4).",
    )
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
    from src.config.company_tiers import SELF_COMPANY_IDS

    peers: list[str] = args.peer or [
        company_id for company_id in COMPANY_IDS if company_id not in SELF_COMPANY_IDS
    ]
    agent = CapabilityEvolutionAgent()

    for peer_id in peers:
        log.info("→ peer=%s", peer_id)
        try:
            result = agent.run(peer_id=peer_id, lookback_quarters=args.lookback_quarters)
            agent.persist(peer_id=peer_id, result=result)
        except Exception as exc:  # noqa: BLE001
            log.exception("capability evolution run failed | peer=%s error=%s", peer_id, exc)


if __name__ == "__main__":
    main()
