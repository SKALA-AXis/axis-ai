# 작성일: 2026-06-11
# 작성자: 안가은
# 변경이력:
#   2026-06-11 안가은 — peer 키워드·SWOT 프리뷰 파이프라인 추가 시 함께 작성
#   2026-06-14 심유정 — 카드뉴스 frontend-ready grounding 개선 반영
"""CLI entrypoint for Peer+ overview keyword snapshot generation."""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.peer_overview_keywords import main


if __name__ == "__main__":
    main()
