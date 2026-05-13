"""Langfuse self-host (team13) 통합 — LangChain CallbackHandler factory + helpers.

설계: axis-infra/docs/OBSERVABILITY_LANGFUSE.md §6
배포: axis-infra/k8s/argocd/langfuse-application.yaml (chart 0.8.0, v2)

핵심 책임
---------
1. ``LANGFUSE_*`` env 가 있으면 CallbackHandler 활성 / 없으면 silent no-op
   (개발 / CI / unit test 환경에서 부담 없이 동작).
2. agent / phase / prompt_version metadata 를 호출별로 부착.
3. EvidenceAgent 가 ``evidence_chain.provenance.langfuse_trace_id`` 에 적재할
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
from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig

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
    """Lazy init Langfuse CallbackHandler. ENV 부족 시 None."""
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

    try:
        from langfuse.callback import CallbackHandler

        _handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            flush_at=10,  # 10 span 마다 또는
            flush_interval=2.0,  # 2초 마다 flush
        )
        _enabled = True
        log.info("Langfuse CallbackHandler initialized | host=%s", host)
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
    **extra_metadata: Any,
) -> RunnableConfig:
    """LangChain config builder — callbacks + metadata + tags.

    Args:
        agent: agent class name (예: "CardComposerAgent")
        phase: multi-phase agent 의 step (예: "summarize" / "analyze" / "compose")
        prompt_version: prompt_version 상수 (각 agent 가 보유)
        request_id: BE 가 X-Request-Id 헤더로 전달한 UUID (있으면)
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
    """현재 handler 의 마지막 trace_id 반환 (EvidenceAgent 가 provenance 적재 시 사용).

    Langfuse handler 가 trace_id 를 stateful 하게 보존하므로, 호출 직후 본 함수
    로 read. 단, multi-threaded 또는 async 동시 호출 시 race 가능 — caller 가
    보장해야 함 (보통 한 agent 호출 끝나면 즉시 read).
    """
    handler = get_langfuse_handler()
    if handler is None:
        return None
    # langfuse 2.x 의 CallbackHandler 가 보존하는 trace 객체 접근
    try:
        # langfuse 2.x API: handler.trace 가 가장 최근 trace 객체
        trace = getattr(handler, "trace", None)
        if trace is not None:
            return getattr(trace, "id", None) or getattr(trace, "trace_id", None)
        # fallback — handler.langfuse 의 thread-local 마지막 trace
        return None
    except Exception:
        return None


def flush() -> None:
    """Pending traces 강제 flush (pod 종료 / unit test cleanup 용)."""
    handler = get_langfuse_handler()
    if handler is None:
        return
    try:
        handler.flush()
    except Exception as e:
        log.debug("Langfuse flush 실패 (무시): %s", e)
