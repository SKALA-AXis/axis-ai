# 작성일: 2026-06-17
# 작성자: 박지원
# 변경이력:
#   2026-06-17 박지원 — 글로벌 동향 표시 문구 정리 스크립트 추가 (글로벌 기업 동향 기능 작업, 포매팅 적용)
"""Clean display copy in existing global_industry_trends rows.

This is a low-cost repair for rows generated before display-safe fallbacks were
added to ITTrendAgent. It does not rerun the global trends agent or change trend
detections; it only replaces internal/debug-style title and summary text.

Usage:
    uv run python scripts/backfill_global_trend_display_copy.py --days 14
    uv run python scripts/backfill_global_trend_display_copy.py --days 14 --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.postgres import SessionLocal  # noqa: E402

_SELECT_SQL = text(
    """
    SELECT id, trend_date, keyword, title, summary, mention_count, payload
      FROM global_industry_trends
     WHERE trend_date >= CURRENT_DATE - make_interval(days => :days)
     ORDER BY trend_date DESC, impact_score DESC NULLS LAST, keyword ASC
    """
)

_UPDATE_SQL = text(
    """
    UPDATE global_industry_trends
       SET title = :title,
           summary = :summary,
           updated_at = NOW()
     WHERE id = :id
    """
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    updates: list[dict[str, Any]] = []
    with SessionLocal() as db:
        rows = db.execute(_SELECT_SQL, {"days": args.days}).mappings().all()
        for row in rows:
            payload = _payload_dict(row.get("payload"))
            next_title = _clean_title(str(row.get("title") or ""), str(row.get("keyword") or ""))
            next_summary = _clean_summary(
                summary=str(row.get("summary") or ""),
                keyword=str(row.get("keyword") or ""),
                mention_count=int(row.get("mention_count") or 0),
                payload=payload,
            )
            if next_title != (row.get("title") or "") or next_summary != (row.get("summary") or ""):
                updates.append(
                    {
                        "id": row["id"],
                        "trend_date": row["trend_date"],
                        "keyword": row["keyword"],
                        "title": next_title,
                        "summary": next_summary,
                    }
                )

        print(f"scanned={len(rows)} updates={len(updates)} apply={args.apply}")
        for item in updates[:20]:
            print(
                f"- {item['trend_date']} {item['keyword']}: "
                f"title={item['title']!r} summary={item['summary']!r}"
            )

        if args.apply and updates:
            db.execute(_UPDATE_SQL, updates)
            db.commit()
            print(f"applied={len(updates)}")
        elif not args.apply:
            print("dry-run only. Re-run with --apply to update rows.")

    return 0


def _payload_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _clean_title(title: str, keyword: str) -> str:
    text_value = re.sub(r"\s+", " ", title).strip()
    text_value = re.sub(
        r"\s+[—-]\s+(?:strong|moderate|weak)\s+강도\s+글로벌\s+트렌드\s*$",
        "",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(r"\s+[—-]\s+글로벌\s+트렌드\s*$", "", text_value, flags=re.IGNORECASE)
    if not text_value:
        return keyword.replace("_", " ")
    return text_value[:300]


def _clean_summary(
    *,
    summary: str,
    keyword: str,
    mention_count: int,
    payload: dict[str, Any],
) -> str:
    text_value = re.sub(r"\s+", " ", summary).strip()
    if text_value and not _is_internal_summary(text_value):
        return text_value

    evidence_title = _first_evidence_title(payload)
    if evidence_title:
        return evidence_title

    return ""


def _is_internal_summary(value: str) -> bool:
    patterns = (
        r"global\s*6\s*사\s*newsroom",
        r"글로벌\s*6\s*사\s*newsroom",
        r"intensity\s*=",
        r"주도\s*기업\s*:\s*-",
    )
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _first_evidence_title(payload: dict[str, Any]) -> str:
    links = payload.get("evidence_source_links") or []
    if not isinstance(links, list):
        return ""
    for link in links:
        if not isinstance(link, dict):
            continue
        title = re.sub(r"\s+", " ", str(link.get("title") or "")).strip()
        if title:
            return title[:500]
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
