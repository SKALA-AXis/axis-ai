# 작성일: 2026-05-21
# 작성자: 최종민
# 변경이력:
#   2026-05-21 최종민 — Layer B 분석 파이프라인 작업의 일부로 추가
#   2026-05-29 박지원 — peer profile snapshot 갱신 에이전트/파이프라인 작성
"""W2-2 CronJob entrypoint — peer 별 profile_snapshot 분기 1회 갱신.

분기 첫날 03:00 KST 에 `axis-cron-profile-refresh` 가 실행. 4 peer + SK AX 의
회사 프로필을 `PeerProfileAgent.build_and_save` 로 합성하여
`peer_companies.profile_snapshot` JSONB 에 저장한다.

사용법:
    uv run python scripts/refresh_peer_profile_snapshots.py [--company samsung_sds] [--env cloud]
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger("refresh_peer_profile_snapshots")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh peer profile snapshots.")
    parser.add_argument("--company", action="append", default=None)
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="LLM 없이 deterministic fallback snapshot 을 저장한다.",
    )
    parser.add_argument(
        "--version",
        default=None,
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

    from src.agents.peer_profile_agent import PeerProfileAgent
    from src.config.companies import COMPANY_IDS
    from src.services.profile_snapshot_summarizer import PROFILE_SNAPSHOT_SCHEMA_VERSION

    targets = args.company or list(COMPANY_IDS)
    version = args.version or PROFILE_SNAPSHOT_SCHEMA_VERSION
    agent = PeerProfileAgent()

    for company_id in targets:
        log.info("→ %s 프로필 합성 시작", company_id)
        try:
            snapshot = agent.build_and_save(
                company_id,
                use_llm=not args.no_llm,
                version=version,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("profile snapshot refresh 실패 | company=%s error=%s", company_id, exc)
            continue
        log.info(
            "snapshot 저장 완료 | company=%s version=%s business_areas=%s recent_changes=%s",
            company_id,
            version,
            len(snapshot.get("business_areas") or []),
            len(snapshot.get("recent_changes") or []),
        )


if __name__ == "__main__":
    main()
