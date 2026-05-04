"""구형 _crawler 호환용 텍스트 정리 함수."""

from __future__ import annotations

import html
import re


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", html.unescape(text)).strip()
