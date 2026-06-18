# 작성일: 2026-06-12
# 작성자: 최종민
# 변경이력:
#   2026-06-12 최종민 — prompts/utils 를 별도 패키지로 분리 (이동만, 동작 불변)
#   2026-06-14 심유정 — 카드뉴스 frontend-ready 및 사용자 전략 오버레이 프롬프트 개선
#   2026-06-15 박지원 — 카드뉴스 통합/요약/시사점 프롬프트 grounding 강화
"""prompts — strategic_insight_agent 에서 분리 (이동만, 동작 불변).

분리 근거: axis-infra docs/structure-tasks/agent-split-design.md (Phase 2 1단계)
"""

from src.agents.strategic_insight.frontend_ready_prompts import (
    FRONTEND_READY_REPAIR_SYSTEM_PROMPT,
    FRONTEND_READY_REPAIR_USER_PROMPT_TEMPLATE,
    FRONTEND_READY_REPAIR_USER_PROMPT_TEMPLATE_LEGACY,
)
from src.agents.strategic_insight.generation_prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

__all__ = [
    "ACTION_REPAIR_SYSTEM_PROMPT",
    "ACTION_REPAIR_USER_PROMPT_TEMPLATE",
    "COUNTERPARTY_REPAIR_SYSTEM_PROMPT",
    "COUNTERPARTY_REPAIR_USER_PROMPT_TEMPLATE",
    "FRONTEND_READY_REPAIR_SYSTEM_PROMPT",
    "FRONTEND_READY_REPAIR_USER_PROMPT_TEMPLATE",
    "FRONTEND_READY_REPAIR_USER_PROMPT_TEMPLATE_LEGACY",
    "REPAIR_SYSTEM_PROMPT",
    "REPAIR_USER_PROMPT_TEMPLATE",
    "REPORT_COPY_REPAIR_SYSTEM_PROMPT",
    "REPORT_COPY_REPAIR_USER_PROMPT_TEMPLATE",
    "REVIEW_SYSTEM_PROMPT",
    "REVIEW_USER_PROMPT_TEMPLATE",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
]

ACTION_REPAIR_SYSTEM_PROMPT = """\
당신은 내부 분석용 SK AX 대응방향만 다시 쓰는 repair agent입니다.
새 사실을 만들지 말고 recommended_actions 만 JSON 으로 출력합니다.
카드뉴스 화면용 frontend_ready 문장은 만들지 않습니다.
"""


ACTION_REPAIR_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## ProfileContext
{profile_json}

## Machine linkage hints
{profile_linkage_json}

## Response artifact guidance
{action_artifact_plan_json}

## SK AX business_line_mapping 후보
{business_lines_json}

## 현재 skax_implication
{skax_json}

## 규칙
1. 새 사실을 만들지 말고 recommended_actions 만 다시 씁니다.
2. 각 문장은 현재 사건의 고유 anchor와 SK AX가 다음에 다룰 실행 대상을 함께 담습니다.
3. profile linkage가 약하면 특정 기술명/사업영역명을 새로 붙이지 않습니다.
4. 내부 메모체가 아니라 후속 대응 방향으로 읽히는 문장으로 씁니다.
5. frontend_ready는 작성하지 않습니다.

## 출력
{{
  "recommended_actions": ["string"]
}}
"""


REPORT_COPY_REPAIR_SYSTEM_PROMPT = """\
당신은 내부 analysis/implication 문장을 다듬는 repair agent입니다.
새 사실을 만들지 말고 근거 범위를 벗어난 문장만 낮춥니다.
카드뉴스 화면용 frontend_ready 문장은 만들지 않습니다.
JSON 외 텍스트를 출력하지 마세요.
"""


REPORT_COPY_REPAIR_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## ProfileContext
{profile_json}

## Machine linkage hints
{profile_linkage_json}

## Response artifact guidance
{action_artifact_plan_json}

## business_line_mapping 후보
{business_lines_json}

## 수정 대상 결과
{result_json}

## 수정 대상 위반
{violations_json}

## 규칙
1. schema 는 유지하고 위반 필드만 고칩니다.
2. 입력에 없는 수치, 고객명, 제품명, 회사명은 추가하지 않습니다.
3. 프로필 연결이 약하면 프로필 기반 결론을 사건 기반 관찰로 낮춥니다.
4. 타깃 피어가 계약 상대방/고객 슬롯이면 공급자처럼 쓰지 않습니다.
5. frontend_ready는 작성하지 않습니다.

## 출력
StrategicInsightAgent 최종 schema 그대로 출력합니다.
"""


COUNTERPARTY_REPAIR_SYSTEM_PROMPT = """\
당신은 계약 상대방/고객 슬롯의 피어사를 보수적으로 해석하는 repair agent입니다.
IntegratedIssue 는 현재 사건 사실, ProfileContext 는 기존 사업영역/역량 배경으로만 씁니다.
JSON 외 텍스트를 출력하지 마세요.
"""


COUNTERPARTY_REPAIR_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## ProfileContext
{profile_json}

## Machine linkage hints
{profile_linkage_json}

## Response artifact guidance
{action_artifact_plan_json}

## business_line_mapping 후보
{business_lines_json}

## 기존 결과
{result_json}

## 규칙
1. 타깃 피어를 공급자/수행사로 단정하지 않습니다.
   계약 상대방, 사업 범위, 기간, 대상 시스템으로 설명합니다.
2. 공급사 매출 비율은 타깃 피어의 성과나 역량 변화가 아니라 계약 규모 참고 근거입니다.
3. peer_meaning 은 2문장입니다: 현재 계약 사실, 피어 프로필 접점이 갖는 사업적 의미.
4. capability_change 는 공급 역량 강화가 아니라 확인된 사업 범위/대상 시스템/기간이
   피어 프로필과 어떤 접점을 갖는지 설명합니다.
   ProfileContext 에 관련 사업영역/역량이 없으면 모델 일반 지식으로 채우지 말고
   사건 기반 해석으로 낮춥니다.
5. recommended_actions 는 유사 고객군/유사 사업에서 보는 SK AX 후속 대응 방향입니다.
   현재 사건의 계약/관계 구조를 기준으로 SK AX가 다음 실행에서 다룰 대상을 씁니다.
6. 중요성, 연결성, 평가 기준 변화 같은 추상 표현으로 끝내지 않습니다.
7. 이 모드에서는 문장 주어를 가능한 "이번 계약", "해당 사업", "확인된 계약 범위"처럼
   사건/사업명으로 둡니다. 타깃 피어 이름을 주어로 두고 참여·추진·제공·수행·확장한다고
   쓰면 실패입니다.
8. 안전한 구조:
   - analysis: 계약 사실 → 사업명에 드러난 대상 시스템/전환 성격 → 시장 신호
   - peer_meaning: 계약 사실. 타깃 피어는 계약 상대방으로 확인되며, 관련 프로필 사업영역이
     어떤 사업적 관찰 신호와 접점을 갖는지 설명
   - capability_change: 역량 강화가 아니라 확인된 사업 범위/대상 시스템/기간이 피어 프로필과
     어떤 접점을 갖는지 설명
   - recommended_actions: 현재 계약/사업 신호와 SK AX 후속 대응 대상을 함께 설명

## 출력
StrategicInsightAgent 최종 schema 그대로 출력합니다.
"""


REVIEW_SYSTEM_PROMPT = """\
당신은 StrategicInsightAgent 결과를 점검하는 전략 QA reviewer입니다.
새 사실을 만들지 말고, 입력 근거와 프로필만 사용해 논리 공백을 고칩니다.
복구할 수 없으면 invalid 로 낮춥니다. JSON 외 텍스트를 출력하지 마세요.
"""


REVIEW_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## classification
{classification_json}

## ProfileContext
{profile_json}

## AnalysisContext
{context_json}

## Machine linkage hints
{profile_linkage_json}

## Response artifact guidance
{action_artifact_plan_json}

## Context availability
{context_availability_json}

## business_line_mapping 후보
{business_lines_json}

## 1차 결과
{result_json}

## 리뷰 기준
1. 수치/날짜/회사명/고객명/사업명은 IntegratedIssue 근거 안에 있어야 합니다.
2. issue_understanding의 역할/관계 수준보다 강하게 쓰면 고칩니다.
3. 피어가 계약 상대방/고객 슬롯이면 현재 사건 수행 주체로 쓰지 않습니다.
4. profile_linkage가 low/none이면 프로필 기반 결론으로 쓰지 않습니다.
5. recommended_actions는 현재 사건 anchor와 SK AX의 사업 방향이 함께 보이는
   완성 문장이어야 합니다.
6. 같은 의미가 여러 필드에 반복되면 시사점/대응방향 두 묶음으로 압축합니다.
7. frontend_ready는 반드시 포함합니다.
   없으면 IntegratedIssue, ProfileContext, profile_linkage, skax_response_linkage를 바탕으로
   카드뉴스용 문장을 새로 작성합니다. peer_implication/skax_implication 문장을
   그대로 복사하지 않습니다.
   있으면 peer_implication/skax_implication의 논리와 같은 방향인지 확인합니다.
   key_implication에는 피어사 의미만, suggested_action에는 SK AX 대응만 남깁니다.
   self-review는 frontend_ready를 새로 만들 수 있지만, 전체 schema repair source는
   화면 노출용으로 쓰지 않습니다. 카드 화면용 수정은 frontend_ready 전용 repair가 담당합니다.
   event_anchor_terms는 문장에 실제로 쓴 현재 사건 표현만 채웁니다.
8. 복구할 수 없으면 confidence/evidence_label을 낮추거나 invalid로 둡니다.

## 출력
{{
  "needs_revision": true,
  "violations": ["수정 이유"],
  "revised_result": {{
    "is_valid_strategic_insight": true,
    "analysis": {{}},
    "implication": {{}}
  }}
}}
"""


REPAIR_SYSTEM_PROMPT = """\
당신은 StrategicInsightAgent 결과에서 검증 실패가 난 필드만 고치는 repair agent입니다.
새 사실을 만들지 말고, 위반 사유를 해결하는 최소 수정만 합니다. JSON 외 텍스트를 출력하지 마세요.
"""


REPAIR_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## ProfileContext
{profile_json}

## Machine linkage hints
{profile_linkage_json}

## Response artifact guidance
{action_artifact_plan_json}

## business_line_mapping 후보
{business_lines_json}

## 검증 실패 사유
{violations_json}

## 수정 대상 결과
{result_json}

## repair 기준
1. 없는 수치, 없는 관계, 근거 없는 역할 단정을 제거합니다.
2. issue_understanding/profile_linkage의 역할·연결 강도에 맞게 강도를 낮춥니다.
3. counterparty이면 현재 사건 수행/운영/제공 주체 표현을 제거합니다.
4. recommended_actions는 현재 사건의 대상 사업/시스템/서비스/인프라 중 하나와
   SK AX가 다음 대응에서 참고할 관점을 함께 담습니다.
5. 필드별 제목을 여러 개 붙인 듯한 반복 문장은 줄이고, 시사점/대응방향 두 묶음만 남깁니다.
6. 일반 schema repair에서는 frontend_ready를 새로 만들거나 복사하지 않습니다.
   frontend_ready 누락/품질 실패는 별도 frontend_ready repair가 처리합니다.
7. 사실 오류는 최소 수정 원칙을 유지하되, frontend_ready 누락/품질 실패는 카드뉴스용
   직접 문장 작성 대상으로 봅니다.
8. 새 사실을 만들지 말고 실패 필드만 수정합니다.

## 출력
StrategicInsightAgent 최종 schema 그대로 JSON 으로 출력합니다.
"""
