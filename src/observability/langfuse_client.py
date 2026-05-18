"""Langfuse self-host (team13) 통합 — LangChain CallbackHandler factory + helpers.

설계: axis-infra/docs/OBSERVABILITY_LANGFUSE.md §6
배포: axis-infra/k8s/argocd/langfuse-application.yaml (chart 0.8.0, v2)

핵심 책임
---------
1. ``LANGFUSE_*`` env 가 있으면 CallbackHandler 활성 / 없으면 silent no-op
   (개발 / CI / unit test 환경에서 부담 없이 동작).
2. agent / phase / prompt_version metadata 를 호출별로 부착.
3. CardNewsAgent 가 ``evidence_chain.provenance.langfuse_trace_id`` 에 적재할
   수 있도록 handler 의 trace_id 노출.
4. git_sha (Helm chart 가 ``AXIS_GIT_SHA`` env 로 주입) 를 module-level 캐시.

사용 예
-------
::

    from src.observability import tracing_config

    response = _llm.invoke(
        prompt,
        config=tracing_config(
            agent="CardComposerAgent",
            phase="compose",
            prompt_version=PROMPT_VERSION,
            cluster_id=cluster.id,
        ),
    )

3-tier 관측 (axis-infra spec §6.5):
- Langfuse: prompt / response / cost / latency / RAG chunks 본문
- app DB usage_logs (V13): 빌링 / quota / langfuse_trace_id pointer
- Prometheus / Grafana: 시계열 메트릭 (P10+ 보류)
"""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig

# Langfuse Sessions / Users — ChatOrchestrator 등에서 set 하면 같은 contextvar 안의
# 모든 tracing_config() 호출 (sub-agent 포함) 이 자동 pickup. asyncio task 도
# contextvar 를 그대로 전파.
_session_id_var: ContextVar[str | None] = ContextVar("axis_session_id", default=None)
_user_id_var: ContextVar[str | None] = ContextVar("axis_user_id", default=None)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# git_sha — Helm chart 의 image build SHA 가 AXIS_GIT_SHA env 로 주입.
# 없으면 git rev-parse, 그것도 실패하면 "unknown".
# Module-level 1회 cache (lru_cache(maxsize=1)).
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def git_sha() -> str:
    """Return short git SHA (12 chars) or 'unknown'."""
    env_sha = os.environ.get("AXIS_GIT_SHA")
    if env_sha:
        return env_sha[:12]
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return result.decode().strip()
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Langfuse CallbackHandler — module-level singleton.
# ENV 가 없으면 _enabled=False, 모든 함수가 None 반환 (no-op).
# ─────────────────────────────────────────────────────────────────────────────


_handler: Any | None = None
_enabled: bool = False


def _init_handler() -> Any | None:
    """Lazy init Langfuse CallbackHandler (v3+). ENV 부족 시 None.

    langfuse v3+ (>=3.0, 4.x) 는 OTel 기반. v2 API (langfuse.callback) 폐기 →
    `langfuse.langchain.CallbackHandler` 사용. SDK 가 env (LANGFUSE_PUBLIC_KEY /
    SECRET_KEY / HOST) 를 자동 인식하므로 init 매개변수 X.
    """
    global _handler, _enabled
    if _handler is not None:
        return _handler

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")

    if not (public_key and secret_key and host):
        log.info(
            "Langfuse 비활성 (LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST 중 하나 누락) — "
            "trace 송신 안 함, agent 동작은 정상"
        )
        _enabled = False
        return None

    # v3+ SDK 는 LANGFUSE_HOST 가 LANGFUSE_BASE_URL 만 박혀있어도 동작하도록
    # 양쪽 모두 환경변수에 set (둘 중 하나만 .env 에 있을 수 있음).
    os.environ.setdefault("LANGFUSE_HOST", host)
    os.environ.setdefault("LANGFUSE_BASE_URL", host)

    try:
        # v3+ 의 새 import path. v2 의 `langfuse.callback` 은 제거됨.
        from langfuse.langchain import CallbackHandler

        # v3+ init signature: CallbackHandler(*, public_key=None, trace_context=None).
        # public_key / secret_key / host 는 env 에서 자동 인식. (None 전달 시 default
        # client 사용.) flush 설정은 client-level (langfuse.get_client()) 에서 조정.
        _handler = CallbackHandler()
        _enabled = True
        log.info("Langfuse CallbackHandler v3+ initialized | host=%s", host)
    except Exception as e:
        log.warning("Langfuse handler 초기화 실패: %s — trace 송신 비활성", e)
        _enabled = False
        _handler = None

    return _handler


def get_langfuse_handler() -> Any | None:
    """본 module 의 singleton handler 반환 (없으면 None)."""
    return _init_handler()


# ─────────────────────────────────────────────────────────────────────────────
# tracing_config — LangChain Runnable 의 config 매개변수 builder.
# ChatOpenAI.invoke(prompt, config=tracing_config(agent="...", ...)) 패턴.
# ─────────────────────────────────────────────────────────────────────────────


def tracing_config(
    agent: str,
    phase: str | None = None,
    prompt_version: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    **extra_metadata: Any,
) -> RunnableConfig:
    """LangChain config builder — callbacks + metadata + tags.

    Args:
        agent: agent class name (예: "CardComposerAgent")
        phase: multi-phase agent 의 step (예: "summarize" / "analyze" / "compose")
        prompt_version: prompt_version 상수 (각 agent 가 보유)
        request_id: BE 가 X-Request-Id 헤더로 전달한 UUID (있으면)
        session_id: Langfuse Sessions — 같은 session_id 의 trace 들을 한 묶음으로
            grouping (ChatOrchestrator 의 conversation session 등).
        user_id: Langfuse Users — 사용자별 token spend / latency 추적용.
        **extra_metadata: cluster_id / peer_id / card_id 등 자유 metadata

    Returns:
        LangChain runnable config dict.
        ENV 없으면 빈 dict (callbacks 없이 그냥 LLM 호출).
    """
    metadata: dict[str, Any] = {
        "agent": agent,
        "git_sha": git_sha(),
    }
    if phase:
        metadata["phase"] = phase
    if prompt_version:
        metadata["prompt_version"] = prompt_version
    if request_id:
        metadata["request_id"] = request_id
    # contextvar fallback — chat orchestrator 등에서 with_session() 으로 set 했으면
    # 같은 task 안의 sub-agent tracing_config 도 자동으로 같은 session_id 사용
    session_id = session_id or _session_id_var.get()
    user_id = user_id or _user_id_var.get()
    # Langfuse 의 magic prefix — session_id / user_id 는 trace 의 top-level field 로 hoist
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if user_id:
        metadata["langfuse_user_id"] = user_id
    metadata.update(extra_metadata)

    handler = get_langfuse_handler()
    if handler is None:
        # ENV 없음 → 그냥 metadata 만 전달 (LangChain 이 내부 로깅에 사용 가능)
        return {"metadata": metadata}

    tags = [agent.lower()]
    if phase:
        tags.append(f"phase={phase}")

    return {
        "callbacks": [handler],
        "metadata": metadata,
        "tags": tags,
        "run_name": f"{agent}.{phase}" if phase else agent,
    }


def get_current_trace_id() -> str | None:
    """직전 LLM 호출의 OTel trace_id 반환 (CardNewsAgent 가 provenance 적재 시 사용).

    v3+ 의 CallbackHandler 는 `last_trace_id` 속성으로 가장 최근 trace id 노출
    (OTel 16-byte hex). 호출 직후 read 가 안전 — multi-thread / async 동시 호출
    시 caller 가 race 보장해야 함.
    """
    handler = get_langfuse_handler()
    if handler is None:
        return None
    try:
        # v3+ API: handler.last_trace_id (16-byte OTel hex string)
        last = getattr(handler, "last_trace_id", None)
        if last:
            return str(last)
        return None
    except Exception:
        return None


def flush() -> None:
    """Pending traces 강제 flush (pod 종료 / unit test cleanup 용).

    v3+ 는 client-level flush. CallbackHandler 자체 .flush() 없음 → singleton
    client 의 flush 호출.
    """
    handler = get_langfuse_handler()
    if handler is None:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        log.debug("Langfuse flush 실패 (무시): %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Sessions / Users — contextvar 기반 propagation
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def with_session(session_id: str | None, user_id: str | None = None):
    """ChatOrchestrator 등에서 conversation 시작 시 wrap.

    같은 contextvar scope 안의 모든 ``tracing_config()`` 호출 (sub-agent 포함) 이
    자동으로 같은 session_id / user_id 를 Langfuse trace 의 top-level field 로 hoist.

    Example::

        async def chat(self, message, session_id, user_id=None):
            with with_session(session_id, user_id):
                # 여기서부터 IntentRouter / 5 sub-agent / Compose 모든 LLM 호출이
                # 같은 session 으로 grouped
                intent = await self._classify(...)
                sub = await self._dispatch(...)
                reply = await self._compose(...)
    """
    session_token = _session_id_var.set(session_id) if session_id else None
    user_token = _user_id_var.set(user_id) if user_id else None
    try:
        yield
    finally:
        if session_token is not None:
            _session_id_var.reset(session_token)
        if user_token is not None:
            _user_id_var.reset(user_token)
