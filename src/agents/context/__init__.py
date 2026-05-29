"""Context-layer agents (W4) — Supervisor 와는 다른 batch/CronJob 계층.

* `sector_pulse_aggregator` : 주 1회, MATERIALIZED VIEW REFRESH (LLM X).
* `event_chain_discovery_agent` : (옵션) 매일 Qdrant 기반 precedent 후보 탐색.
* `_data_quality_checks` : Builder 들이 graceful fallback 할 때 사용하는 헬퍼.
"""

from __future__ import annotations

__all__: list[str] = []
