"""processing_status를 coarse status 모델로 마이그레이션한다.

기본은 dry-run이며, 실제 반영은 --apply를 붙인다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

_COUNT_SQL = text("""
    SELECT processing_status, COUNT(*) AS count
    FROM raw_articles
    GROUP BY processing_status
    ORDER BY processing_status
""")

_UPDATE_SQL = text("""
    UPDATE raw_articles
    SET processing_status = CASE
        WHEN processing_status IN (
            'CLASSIFIED',
            'CLUSTERED_REP',
            'CLUSTERED_DUPE',
            'RELEVANCE_PASSED',
            'CREDIBILITY_SCORED'
        ) THEN 'PROCESSED'
        WHEN processing_status LIKE 'PREPROCESSED_%' THEN 'PROCESSED'
        WHEN processing_status = 'PRESERVED' THEN 'PROCESSED'
        WHEN processing_status LIKE 'SKIPPED_%' THEN 'SKIPPED'
        WHEN processing_status IN ('RAW', 'PROCESSED', 'SKIPPED', 'FAILED')
            THEN processing_status
        ELSE 'FAILED'
    END
    WHERE processing_status NOT IN ('RAW', 'PROCESSED', 'SKIPPED', 'FAILED')
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="실제 DB에 반영")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with SessionLocal() as db:
        before = db.execute(_COUNT_SQL).fetchall()
        log.info("현재 processing_status 분포")
        for row in before:
            log.info("  %s=%d", row.processing_status, row.count)

        if not args.apply:
            target = db.execute(
                text("""
                    SELECT COUNT(*)
                    FROM raw_articles
                    WHERE processing_status NOT IN (
                        'RAW', 'PROCESSED', 'SKIPPED', 'FAILED'
                    )
                """)
            ).scalar_one()
            log.info("dry-run 완료 | 변경 대상 rows=%d. 적용하려면 --apply 사용", target)
            return

        result = db.execute(_UPDATE_SQL)
        db.commit()
        log.info("processing_status 마이그레이션 완료 | updated=%d", result.rowcount or 0)

        after = db.execute(_COUNT_SQL).fetchall()
        log.info("변경 후 processing_status 분포")
        for row in after:
            log.info("  %s=%d", row.processing_status, row.count)


if __name__ == "__main__":
    main()
