# 작성일: 2026-05-11
# 작성자: 박지원
# 변경이력:
#   2026-05-11 박지원 — backfill 크롤러/DART 파서 추가, backfill runner 작성
#   2026-05-12 최종민 — issue_cards → card_news 데이터 레이어 리네임
#   2026-05-18 심유정 — 카드뉴스 에이전트 플로우 개선 및 feature 브랜치 병합
"""Cursor 기반 backfill 실행기."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from src.config.companies import COMPANY_ALIASES
from src.config.global_companies import GLOBAL_COMPANY_ALIASES
from src.crawler.backfill_config import BackfillSourceConfig
from src.crawler.base import CrawlRunContext, CrawlWindow
from src.crawler.batch_processor import BatchProcessor
from src.db.crawl_state_store import (
    CrawlCursor,
    create_crawl_run,
    get_or_create_cursor,
    mark_crawl_run_failed,
    mark_crawl_run_success,
    update_cursor,
)
from src.preprocessing.preprocessing import PreprocessingService

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillRunSummary:
    source_name: str
    window_start: date
    window_end: date
    status: str
    inserted_count: int = 0
    skipped_count: int = 0
    error_message: str | None = None


class BackfillRunner:
    def __init__(
        self,
        *,
        processor: BatchProcessor | None = None,
        initial_cursor_date: date | None = None,
        persist: bool = True,
        use_state: bool = True,
        process_after_window: str = "preprocess",
    ) -> None:
        self.processor = processor or BatchProcessor()
        self.initial_cursor_date = initial_cursor_date or datetime.now().astimezone().date()
        self.persist = persist
        self.use_state = use_state
        self.process_after_window = process_after_window
        self.keywords = {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}

    async def run_source(
        self,
        config: BackfillSourceConfig,
        *,
        max_windows_override: int | None = None,
        run_to_end: bool = False,
    ) -> list[BackfillRunSummary]:
        cursor = self._load_cursor(config)
        if not cursor.enabled:
            log.info("backfill source 비활성화 | source=%s", cursor.source_name)
            return []

        max_windows = self._resolve_max_windows(
            cursor,
            max_windows_override=max_windows_override,
            run_to_end=run_to_end,
        )
        summaries: list[BackfillRunSummary] = []

        for _ in range(max_windows):
            if cursor.cursor_date <= cursor.until_date:
                log.info(
                    "backfill 완료 범위 도달 | source=%s cursor=%s",
                    cursor.source_name,
                    cursor.cursor_date,
                )
                break

            window_start = max(
                cursor.cursor_date - timedelta(days=cursor.window_days),
                cursor.until_date,
            )
            window_end = cursor.cursor_date
            summary = await self._run_window(config.source_name, window_start, window_end)
            summaries.append(summary)

            if summary.status != "success":
                break

            cursor = CrawlCursor(
                source_name=cursor.source_name,
                cursor_date=window_start,
                until_date=cursor.until_date,
                window_days=cursor.window_days,
                max_windows_per_run=cursor.max_windows_per_run,
                enabled=cursor.enabled,
            )
            if self.use_state:
                update_cursor(config.source_name, window_start)

        return summaries

    def _resolve_max_windows(
        self,
        cursor: CrawlCursor,
        *,
        max_windows_override: int | None,
        run_to_end: bool,
    ) -> int:
        if max_windows_override is not None:
            return max_windows_override
        if not run_to_end:
            return cursor.max_windows_per_run
        if cursor.cursor_date <= cursor.until_date:
            return 0

        remaining_days = (cursor.cursor_date - cursor.until_date).days
        return max(1, (remaining_days + cursor.window_days - 1) // cursor.window_days)

    def _load_cursor(self, config: BackfillSourceConfig) -> CrawlCursor:
        if self.use_state:
            return get_or_create_cursor(config, self.initial_cursor_date)

        return CrawlCursor(
            source_name=config.source_name,
            cursor_date=self.initial_cursor_date,
            until_date=config.until_date,
            window_days=config.window_days,
            max_windows_per_run=config.max_windows_per_run,
            enabled=config.enabled,
        )

    async def _run_window(
        self,
        source_name: str,
        window_start: date,
        window_end: date,
    ) -> BackfillRunSummary:
        run_id = create_crawl_run(source_name, window_start, window_end) if self.use_state else None
        crawl_window = CrawlWindow(
            start=datetime.combine(window_start, time.min).astimezone(),
            end=datetime.combine(window_end, time.max).astimezone(),
        )
        run_context = CrawlRunContext(
            collection_mode="backfill",
            crawl_run_id=str(run_id) if run_id else None,
            source_name=source_name,
            window_start=crawl_window.start,
            window_end=crawl_window.end,
        )

        try:
            articles = await self.processor.run_sources(
                [source_name],
                keywords=self.keywords,
                persist=self.persist,
                crawl_window=crawl_window,
                run_context=run_context,
            )
            inserted_count = self.processor.last_inserted_count if self.persist else len(articles)
            skipped_count = max(len(articles) - inserted_count, 0)
            if self.process_after_window != "none":
                if run_id is None:
                    raise RuntimeError("post-window processing requires crawl_runs state")
                self._run_pipeline_for_run(str(run_id))
            if run_id:
                mark_crawl_run_success(run_id, inserted_count, skipped_count)
            return BackfillRunSummary(
                source_name=source_name,
                window_start=window_start,
                window_end=window_end,
                status="success",
                inserted_count=inserted_count,
                skipped_count=skipped_count,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            log.exception(
                "backfill window 실패 | source=%s window=%s~%s",
                source_name,
                window_start,
                window_end,
            )
            if run_id:
                mark_crawl_run_failed(run_id, error_message)
            return BackfillRunSummary(
                source_name=source_name,
                window_start=window_start,
                window_end=window_end,
                status="failed",
                error_message=error_message,
            )

    def _run_pipeline_for_run(self, crawl_run_id: str) -> None:
        result = PreprocessingService().run(
            company=list(self.keywords),
            trigger_type=f"backfill:{self.process_after_window}",
            collected_since=None,
            crawl_run_id=crawl_run_id,
        )

        log.info(
            "backfill 후처리 완료 | mode=%s crawl_run_id=%s raw=%d classified=%d "
            "analysis_metrics=%d analysis_signals=%d",
            self.process_after_window,
            crawl_run_id,
            len(result.get("raw_article_ids", [])),
            len(result.get("classified_clusters", [])),
            result.get("analysis_metric_count", 0),
            result.get("analysis_signal_count", 0),
        )
