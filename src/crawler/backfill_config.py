"""Source별 backfill 기본 정책."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class BackfillSourceConfig:
    source_name: str
    window_days: int
    max_windows_per_run: int
    until_date: date
    enabled: bool = True


DEFAULT_UNTIL_DATE = date(2025, 1, 1)

BACKFILL_SOURCES: dict[str, BackfillSourceConfig] = {
    "naver_news": BackfillSourceConfig("naver_news", 1, 3, DEFAULT_UNTIL_DATE),
    "company_news": BackfillSourceConfig("company_news", 7, 2, DEFAULT_UNTIL_DATE),
    "global_newsroom": BackfillSourceConfig("global_newsroom", 7, 2, DEFAULT_UNTIL_DATE),
    "naver_research": BackfillSourceConfig("naver_research", 14, 1, DEFAULT_UNTIL_DATE),
    "stock": BackfillSourceConfig("stock", 30, 1, DEFAULT_UNTIL_DATE),
    "dart": BackfillSourceConfig("dart", 14, 1, DEFAULT_UNTIL_DATE),
    "ir": BackfillSourceConfig("ir", 30, 1, DEFAULT_UNTIL_DATE),
    "jobs": BackfillSourceConfig("jobs", 14, 1, DEFAULT_UNTIL_DATE),
    "naver_datalab": BackfillSourceConfig("naver_datalab", 30, 1, DEFAULT_UNTIL_DATE),
    "spri": BackfillSourceConfig("spri", 90, 1, DEFAULT_UNTIL_DATE),
    "bcg": BackfillSourceConfig("bcg", 90, 1, DEFAULT_UNTIL_DATE),
}


def resolve_backfill_sources(source_names: list[str] | None) -> list[BackfillSourceConfig]:
    if not source_names or source_names == ["all"]:
        return [config for config in BACKFILL_SOURCES.values() if config.enabled]

    unknown = sorted(set(source_names) - set(BACKFILL_SOURCES))
    if unknown:
        raise ValueError(f"Unknown backfill source(s): {', '.join(unknown)}")

    return [BACKFILL_SOURCES[source_name] for source_name in source_names]
