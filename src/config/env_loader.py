# 작성일: 2026-04-28
# 작성자: 최종민
# 변경이력:
#   2026-04-28 최종민 — 프로파일 기반 .env 로더 추가(.env fallback 2-파일 구조), --env 플래그 및 ruff/lint/mypy 정리
"""프로파일 기반 .env 로더.

run_pipeline_once.py / run_crawler_once.py 같은 단독 실행 스크립트에서 사용.
DB 대상(local 컨테이너 vs Supabase·Qdrant Cloud)을 CLI 플래그로 전환할 수 있게 한다.

우선순위:
  1. 인자로 받은 profile (local | cloud) 에 해당하는 `.env.{profile}` 파일 → override 로 로드
  2. 위 파일이 없으면 기본 `.env` 로 fallback (axis-infra 패턴: `.env` = cloud 기본값)
     Docker compose 가 이미 env 주입한 경우 .env 도 없을 수 있는데 그땐 no-op.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

log = logging.getLogger(__name__)

_VALID_PROFILES = {"local", "cloud"}


def load_profile(profile: Optional[str] = None) -> str:
    """선택한 프로파일에 해당하는 dotenv 파일을 로드한다.

    Args:
        profile: "local" | "cloud" | None. None이면 AXIS_PROFILE env var → 기본 .env 순서로 폴백.

    Returns:
        실제 적용된 프로파일 이름 ("local" | "cloud" | "default").
    """
    name = profile or os.getenv("AXIS_PROFILE")
    if name and name not in _VALID_PROFILES:
        log.warning("알 수 없는 프로파일=%s — 기본 .env 사용", name)
        name = None

    if name:
        env_file = Path(f".env.{name}")
        if env_file.exists():
            load_dotenv(env_file, override=True)
            log.info("env 프로파일 로드 | profile=%s file=%s", name, env_file)
            return name
        log.info("env 프로파일=%s — `%s` 없음, 기본 .env 로 fallback", name, env_file)

    load_dotenv()
    return name or "default"
