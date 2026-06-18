# 작성일: 2026-04-28
# 작성자: 박지원
# 변경이력:
#   2026-04-28 박지원 — 크롤러 로직 개선 및 데이터 수집 안정성 향상(이후 revert) 후 크롤러 코드 업데이트
"""구형 _crawler 호환용 텍스트 정리 함수."""

from __future__ import annotations

import html
import re


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", html.unescape(text)).strip()
