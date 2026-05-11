"""crawl_runs / crawl_cursors 저장소."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from uuid import UUID, uuid4

from sqlalchemy import text

from src.crawler.backfill_config import BackfillSourceConfig
from src.db.postgres import SessionLocal


@dataclass(frozen=True)
class CrawlCursor:
    source_name: str
    cursor_date: date
    until_date: date
    window_days: int
    max_windows_per_run: int
    enabled: bool


def ensure_backfill_state_schema() -> None:
    """로컬/개발 DB에 backfill 상태 테이블을 생성한다.

    운영 DB의 정식 schema ownership은 infra/backend migration에 둔다.
    이 함수는 axis-ai 단독 로컬 검증을 위한 보조 경로다.
    """
    with SessionLocal() as db:
        db.execute(
            text("""
                CREATE TABLE IF NOT EXISTS crawl_cursors (
                    source_name VARCHAR(100) PRIMARY KEY,
                    cursor_date DATE NOT NULL,
                    until_date DATE NOT NULL,
                    window_days INT NOT NULL,
                    max_windows_per_run INT NOT NULL DEFAULT 1,
                    enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        )
        db.execute(
            text("""
                CREATE TABLE IF NOT EXISTS crawl_runs (
                    id UUID PRIMARY KEY,
                    run_type VARCHAR(30) NOT NULL,
                    source_name VARCHAR(100) NOT NULL,
                    window_start DATE NOT NULL,
                    window_end DATE NOT NULL,
                    status VARCHAR(30) NOT NULL,
                    inserted_count INT NOT NULL DEFAULT 0,
                    skipped_count INT NOT NULL DEFAULT 0,
                    error_message TEXT,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    finished_at TIMESTAMPTZ
                )
            """)
        )
        db.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_crawl_runs_source_started
                    ON crawl_runs (source_name, started_at DESC)
            """)
        )
        db.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_crawl_runs_status
                    ON crawl_runs (status)
            """)
        )
        db.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_crawl_runs_window
                    ON crawl_runs (window_start, window_end)
            """)
        )
        db.commit()


def get_or_create_cursor(config: BackfillSourceConfig, initial_cursor_date: date) -> CrawlCursor:
    with SessionLocal() as db:
        row = db.execute(
            text("""
                SELECT source_name, cursor_date, until_date, window_days,
                       max_windows_per_run, enabled
                FROM crawl_cursors
                WHERE source_name = :source_name
            """),
            {"source_name": config.source_name},
        ).mappings().first()

        if row:
            return _cursor_from_row(row)

        db.execute(
            text("""
                INSERT INTO crawl_cursors (
                    source_name, cursor_date, until_date, window_days,
                    max_windows_per_run, enabled, updated_at
                ) VALUES (
                    :source_name, :cursor_date, :until_date, :window_days,
                    :max_windows_per_run, :enabled, NOW()
                )
            """),
            {
                "source_name": config.source_name,
                "cursor_date": initial_cursor_date,
                "until_date": config.until_date,
                "window_days": config.window_days,
                "max_windows_per_run": config.max_windows_per_run,
                "enabled": config.enabled,
            },
        )
        db.commit()

    return CrawlCursor(
        source_name=config.source_name,
        cursor_date=initial_cursor_date,
        until_date=config.until_date,
        window_days=config.window_days,
        max_windows_per_run=config.max_windows_per_run,
        enabled=config.enabled,
    )


def create_crawl_run(source_name: str, window_start: date, window_end: date) -> UUID:
    run_id = uuid4()
    with SessionLocal() as db:
        db.execute(
            text("""
                INSERT INTO crawl_runs (
                    id, run_type, source_name, window_start, window_end,
                    status, started_at
                ) VALUES (
                    :id, 'backfill', :source_name, :window_start, :window_end,
                    'running', NOW()
                )
            """),
            {
                "id": str(run_id),
                "source_name": source_name,
                "window_start": window_start,
                "window_end": window_end,
            },
        )
        db.commit()
    return run_id


def mark_crawl_run_success(run_id: UUID, inserted_count: int, skipped_count: int) -> None:
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE crawl_runs
                SET status = 'success',
                    inserted_count = :inserted_count,
                    skipped_count = :skipped_count,
                    finished_at = NOW()
                WHERE id = :id
            """),
            {
                "id": str(run_id),
                "inserted_count": inserted_count,
                "skipped_count": skipped_count,
            },
        )
        db.commit()


def mark_crawl_run_failed(run_id: UUID, error_message: str) -> None:
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE crawl_runs
                SET status = 'failed',
                    error_message = :error_message,
                    finished_at = NOW()
                WHERE id = :id
            """),
            {"id": str(run_id), "error_message": error_message[:2000]},
        )
        db.commit()


def update_cursor(source_name: str, cursor_date: date) -> None:
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE crawl_cursors
                SET cursor_date = :cursor_date,
                    updated_at = NOW()
                WHERE source_name = :source_name
            """),
            {"source_name": source_name, "cursor_date": cursor_date},
        )
        db.commit()


def list_cursors() -> list[dict]:
    with SessionLocal() as db:
        rows = db.execute(
            text("""
                SELECT source_name, cursor_date, until_date, window_days,
                       max_windows_per_run, enabled, updated_at
                FROM crawl_cursors
                ORDER BY source_name
            """)
        ).mappings().all()
    return [dict(row) for row in rows]


def list_recent_runs(limit: int = 10) -> list[dict]:
    with SessionLocal() as db:
        rows = db.execute(
            text("""
                SELECT id, run_type, source_name, window_start, window_end,
                       status, inserted_count, skipped_count, error_message,
                       started_at, finished_at
                FROM crawl_runs
                ORDER BY started_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        ).mappings().all()
    return [dict(row) for row in rows]


def _cursor_from_row(row) -> CrawlCursor:
    return CrawlCursor(
        source_name=row["source_name"],
        cursor_date=row["cursor_date"],
        until_date=row["until_date"],
        window_days=int(row["window_days"]),
        max_windows_per_run=int(row["max_windows_per_run"]),
        enabled=bool(row["enabled"]),
    )
