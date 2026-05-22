"""기존 raw_articles.matched_sector_details 값을 채운다.

기본은 dry-run이며, 실제 반영은 --apply를 붙인다.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.sectors import match_sector_details
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

_LOAD_SQL = text("""
    SELECT id, title, content, matched_sectors
    FROM raw_articles
    WHERE COALESCE(jsonb_array_length(matched_sectors), 0) > 0
      AND COALESCE(jsonb_array_length(matched_sector_details), 0) = 0
    ORDER BY id
    LIMIT :limit
""")

_COUNT_SQL = text("""
    SELECT COUNT(*)
    FROM raw_articles
    WHERE COALESCE(jsonb_array_length(matched_sectors), 0) > 0
      AND COALESCE(jsonb_array_length(matched_sector_details), 0) = 0
""")

_UPDATE_SQL = text("""
    UPDATE raw_articles
    SET matched_sector_details = CAST(:matched_sector_details AS jsonb)
    WHERE id = :id
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="실제 DB에 반영")
    parser.add_argument("--limit", type=int, default=5000, help="한 번에 처리할 최대 row 수")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with SessionLocal() as db:
        target_count = int(db.execute(_COUNT_SQL).scalar_one() or 0)
        rows = db.execute(_LOAD_SQL, {"limit": args.limit}).fetchall()

        patches = []
        for row in rows:
            matched_sectors = _json_list(row.matched_sectors)
            details = _details_for_row(
                title=row.title,
                content=row.content,
                matched_sectors=matched_sectors,
            )
            if details:
                patches.append((int(row.id), details))

        log.info(
            "matched_sector_details 백필 점검 | targets=%d loaded=%d rows_to_update=%d",
            target_count,
            len(rows),
            len(patches),
        )
        for article_id, details in patches[:10]:
            log.info(
                "변경 대상 샘플 | id=%d details=%s",
                article_id,
                json.dumps(details, ensure_ascii=False),
            )

        if not args.apply:
            log.info("dry-run 완료 | 실제 DB 업데이트 없음. 적용하려면 --apply 사용")
            return

        for article_id, details in patches:
            db.execute(
                _UPDATE_SQL,
                {
                    "id": article_id,
                    "matched_sector_details": json.dumps(details, ensure_ascii=False),
                },
            )
        db.commit()
        log.info("matched_sector_details 백필 완료 | updated=%d", len(patches))


def _details_for_row(
    *,
    title: str | None,
    content: str | None,
    matched_sectors: list[str],
) -> list[dict[str, str]]:
    allowed = {sector for sector in matched_sectors if sector != "other"}
    if not allowed:
        return []

    text_value = f"{title or ''} {content or ''}"
    return [detail for detail in match_sector_details(text_value) if detail["sector_id"] in allowed]


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    return []


if __name__ == "__main__":
    main()
