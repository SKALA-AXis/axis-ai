"""ImplicationAgent v5.0 — v4.0 + AnalysisContext block (W4-5).

W4 의 AnalysisContext 가 제공되면 prompt 가 자연스럽게 종방향 (시계열) /
횡방향 (sector pulse) 맥락을 함께 인용한다. v4.0 base 는 그대로 재사용.
"""

from __future__ import annotations

from src.analysis.prompts.implication_v4 import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

PROMPT_VERSION = "implication-v5.0"

# v5.0 추가 rule (SYSTEM_PROMPT 와 합쳐 사용).
SYSTEM_PROMPT_EXTENSION = """\

## v5.0 추가 룰 (AnalysisContext 활용)
- AnalysisContext.event_chain_candidates 에 선행 사건이 있으면
  peer_implication.precedent_link 를 채우고, 다음 4 relation 중 하나를 선택:
  follow_up | reaction | echo | contradiction.
- AnalysisContext.capability_evolution 에 narrative 가 있으면
  peer_implication.capability_change 에 해당 narrative 의 핵심을 1 문장 인용.
- AnalysisContext.sector_pulse_recent 의 intensity_avg 가 직전 4주 대비 +20% 이상이면
  skax_implication.why_important 에 sector momentum 을 반영.
- evidence_density_per_peer[peer_id].density_label == "sparse" 이면 confidence 0.7 이상 부여 금지.
- ImplicationProvenance.used_context_layers 에 실제 인용한 layer 이름 (예:
  "peer_event_timeline_recent", "capability_evolution") 을 명시.
"""

SYSTEM_PROMPT_V5 = SYSTEM_PROMPT + SYSTEM_PROMPT_EXTENSION
USER_PROMPT_TEMPLATE_V5 = USER_PROMPT_TEMPLATE  # v4.0 와 동일, context_json 자리 활성화.

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT_V5",
    "USER_PROMPT_TEMPLATE_V5",
]
