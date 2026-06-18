# 작성일: 2026-04-21
# 작성자: 최종민
# 변경이력:
#   2026-04-21 최종민 — axis-ai 베이스라인으로 전달 파이프라인 추가, ruff format
#   2026-04-28 박지원 — 크롤러 로직 개선 커밋 및 동일 변경 되돌리기(revert)
"""전달 파이프라인 — 매일 오전 8:30 backend 의 Spring @Scheduled 가 호출.

ADR 0004: 수집 파이프라인과 전달 파이프라인 분리 원칙.
ADR 0008 (v4): axis-backend 가 PostgreSQL 의 issue cards 조회 + axis-ai 의
/pipeline/delivery 에 cards 전달 → axis-ai 가 HTML/text 본문 빌더 후 반환 →
backend 의 SesMailService 가 AWS SES V2 SDK (IRSA) 로 발송.

axis-ai 는 *발송 안 함*. 본문 데이터만 반환.
"""

from __future__ import annotations

import html as html_lib
import logging
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

log = logging.getLogger(__name__)


class DeliveryState(TypedDict):
    cards: list[dict[str, Any]]
    subject: str
    html: str
    text: str
    errors: list[str]


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _importance_label(importance: str | None) -> str:
    return {
        "urgent": "[긴급]",
        "notable": "[주목]",
    }.get(importance or "", "[참고]")


def _importance_color(importance: str | None) -> str:
    return {
        "urgent": "#dc2626",
        "notable": "#d97706",
    }.get(importance or "", "#16a34a")


def build_briefing_node(state: DeliveryState) -> DeliveryState:
    """backend 가 보낸 cards 를 HTML/text 본문으로 렌더링."""
    cards = state.get("cards") or []
    today = _today_str()
    log.info("브리핑 본문 빌더 | cards=%d", len(cards))

    subject = f"[AXIS] 오늘의 동향 브리핑 — {today}"

    # ── text (이메일 클라이언트가 HTML 차단 시 fallback) ─────────────────
    text_lines = [
        f"AXIS 오늘의 동향 브리핑 — {today}",
        "",
        f"오늘 {len(cards)}건의 동향이 정리되었습니다.",
        "",
    ]
    for idx, card in enumerate(cards, start=1):
        label = _importance_label(card.get("importance"))
        title = card.get("title") or "(제목 없음)"
        peer = card.get("peerId") or ""
        event = card.get("eventType") or ""
        meta_parts = [p for p in (peer, event) if p]
        meta = f" ({' · '.join(meta_parts)})" if meta_parts else ""
        text_lines.append(f"{idx}. {label} {title}{meta}")
    text_lines.append("")
    text_lines.append("— SK AX 사업전략팀 AXIS")
    text = "\n".join(text_lines)

    # ── HTML ─────────────────────────────────────────────────────────────
    items_html: list[str] = []
    for card in cards:
        label = _importance_label(card.get("importance"))
        color = _importance_color(card.get("importance"))
        title = html_lib.escape(card.get("title") or "(제목 없음)")
        peer = html_lib.escape(card.get("peerId") or "")
        event = html_lib.escape(card.get("eventType") or "")
        meta_parts = [p for p in (peer, event) if p]
        meta_inner = " · ".join(meta_parts)
        meta_html = (
            f'<div style="color:#6b7280;font-size:13px;margin-top:4px;">{meta_inner}</div>'
            if meta_parts
            else ""
        )
        items_html.append(
            f'<li style="padding:12px 0;border-bottom:1px solid #e5e7eb;">'
            f'<span style="color:{color};font-weight:600;">{label}</span> '
            f'<span style="font-weight:500;">{title}</span>'
            f"{meta_html}"
            f"</li>"
        )

    footer_text = "SK AX 사업전략팀 AXIS · 자동 발송 (응답 X)"
    html_body = (
        '<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">'
        f"<title>{html_lib.escape(subject)}</title></head>"
        "<body style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        'max-width:680px;margin:0 auto;padding:24px;color:#111827;">'
        '<header style="border-bottom:2px solid #111827;padding-bottom:16px;margin-bottom:24px;">'
        '<h1 style="margin:0;font-size:22px;">AXIS 오늘의 동향 브리핑</h1>'
        f'<p style="margin:8px 0 0;color:#6b7280;font-size:14px;">{today} · {len(cards)}건</p>'
        "</header>"
        '<ul style="list-style:none;padding:0;margin:0;">' + "".join(items_html) + "</ul>"
        '<footer style="margin-top:32px;padding-top:16px;border-top:1px solid #e5e7eb;'
        f'color:#6b7280;font-size:12px;">{footer_text}</footer>'
        "</body></html>"
    )

    return {**state, "subject": subject, "html": html_body, "text": text}


def build_delivery_graph() -> StateGraph:
    graph = StateGraph(DeliveryState)
    graph.add_node("build_briefing", build_briefing_node)
    graph.set_entry_point("build_briefing")
    graph.add_edge("build_briefing", END)
    return graph.compile()  # type: ignore[return-value]


delivery_graph = build_delivery_graph()
