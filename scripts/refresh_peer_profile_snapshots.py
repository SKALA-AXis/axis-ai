"""W2-2 CronJob entrypoint — peer 별 profile_snapshot 주 1회 갱신.

매주 월요일 03:00 KST 에 `axis-cron-profile-refresh` 가 실행. 4 peer + SK AX 의
회사 프로필을 `ProfileAgent.build_profile` 로 합성하여
`peer_companies.profile_snapshot` JSONB 에 저장한다.

사용법:
    uv run python scripts/refresh_peer_profile_snapshots.py [--company samsung_sds] [--env cloud]
"""

from __future__ import annotations

import argparse
import json
import logging

log = logging.getLogger("refresh_peer_profile_snapshots")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh peer profile snapshots.")
    parser.add_argument("--company", action="append", default=None)
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument(
        "--version",
        default="profile-v5",
        help="profile_snapshot_version 컬럼에 기록할 값.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.config.env_loader import load_profile

    load_profile(args.env)

    from sqlalchemy import text

    from src.agents.profile_agent import ProfileAgent
    from src.config.companies import COMPANY_IDS
    from src.db.postgres import SessionLocal

    targets = args.company or list(COMPANY_IDS)
    agent = ProfileAgent()

    for company_id in targets:
        log.info("→ %s 프로필 합성 시작", company_id)
        try:
            profile = agent.build_profile(company_id, lookback_days=args.lookback_days)
        except Exception as exc:  # noqa: BLE001
            log.exception("build_profile 실패 | company=%s error=%s", company_id, exc)
            continue
        try:
            _save_snapshot(
                company_id=company_id,
                payload=profile,
                version=args.version,
                session_factory=SessionLocal,
                text=text,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("snapshot persist 실패 | company=%s error=%s", company_id, exc)


def _save_snapshot(
    *,
    company_id: str,
    payload: dict,
    version: str,
    session_factory,
    text,
) -> None:
    with session_factory() as db:
        db.execute(
            text(
                """
                UPDATE peer_companies
                   SET profile_snapshot = CAST(:payload AS jsonb),
                       profile_snapshot_version = :version,
                       profile_snapshot_generated_at = NOW()
                 WHERE id = :company_id
                """
            ),
            {
                "company_id": company_id,
                "payload": json.dumps(payload, ensure_ascii=False),
                "version": version,
            },
        )
        db.commit()
    log.info("snapshot 저장 완료 | company=%s version=%s", company_id, version)


if __name__ == "__main__":
    main()
