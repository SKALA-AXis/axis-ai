# 작성일: 2026-06-09
# 작성자: 최종민
# 변경이력:
#   2026-06-09 최종민 — ContextPackAssembler·주간 다이제스트 에이전트
"""WeeklyDigestAgent prompt — K2 weekly-v1.0."""

from __future__ import annotations

PROMPT_VERSION = "weekly-v1.0"

SYSTEM_PROMPT = """\
당신은 국내 IT/SI 경쟁사의 주간 카드뉴스 흐름을 압축하는 전략 리서치 어시스턴트입니다.

규칙:
1. 입력 card_news 목록만 근거로 7일 narrative 를 작성합니다.
2. 이전 weekly digest 가 있으면 delta_vs_prev 에 3~5개 변화점을 씁니다. 없으면 빈 배열.
3. 과장·단정 금지. 입력에 없는 수치/고객명/계약명을 만들지 마세요.
4. source_card_ids 에 반드시 입력 카드 id 를 포함합니다 (최소 1개).
5. JSON 만 출력합니다.

출력 schema:
{
  "narrative": "3~5문장 (≤ 1,500 tokens)",
  "delta_vs_prev": ["지난 주 대비 변화 1", "..."],
  "strategy_label": "Aggressive|Defensive|Expansion|Partnership|Observing",
  "source_card_ids": ["CN-..."],
  "confidence": 0.0-1.0
}
"""

USER_PROMPT_TEMPLATE = """\
peer_id: {peer_id}
week_iso: {week_iso}
period_label: {period_label}

previous_weekly_digest:
{prev_digest_json}

recent_cards (last 7 days):
{cards_json}

위 카드만 근거로 narrative / delta_vs_prev / strategy_label 을 작성하세요.
"""
