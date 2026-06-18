# 작성일: 2026-06-10
# 작성자: 박진
# 변경이력:
#   2026-06-10 박진 — 어시스턴트 PDF 내보내기·믹서 자격증명 수정 과정에서 추가
"""LLM credential loading helpers.

Runtime pods should receive credentials through environment variables. Local
and bootRun flows often rely on ``.env`` files, so agents use this helper before
constructing LangChain OpenAI clients.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CREDENTIAL_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
)


def has_llm_credentials() -> bool:
    return any(os.getenv(key, "").strip() for key in _CREDENTIAL_KEYS)


def ensure_llm_env_loaded() -> None:
    if has_llm_credentials():
        return
    try:
        from dotenv import load_dotenv

        profile = os.getenv("AXIS_PROFILE", "").strip()
        if profile in {"local", "cloud"}:
            load_dotenv(_PROJECT_ROOT / f".env.{profile}", override=True)
        if not has_llm_credentials():
            load_dotenv(_PROJECT_ROOT / ".env")
        if not has_llm_credentials():
            load_dotenv(_PROJECT_ROOT / ".env.local")
    except Exception as exc:  # noqa: BLE001 - never hide the agent's original error.
        log.debug("LLM dotenv load skipped | error=%s", exc)


def llm_credentials_ready() -> bool:
    ensure_llm_env_loaded()
    return has_llm_credentials()


def missing_llm_credentials_message() -> str:
    return (
        "AI 모델 인증 정보가 설정되지 않아 LLM 분석을 실행하지 못했습니다. "
        "AI 서버 실행 환경에 OPENAI_API_KEY 또는 OPENAI_ADMIN_KEY를 설정한 뒤 "
        "다시 시도하세요."
    )


def is_missing_llm_credentials_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "missing credentials" in message
        or "openai_api_key" in message
        or "openai api key" in message
        or "openai_admin_key" in message
    )
