# 작성일: 2026-06-12
# 작성자: 최종민
# 변경이력:
#   2026-06-12 최종민 — briefing_generation_agent 분해 1단계로 prompts 분리
#   2026-06-14 안가은 — 브리핑 생성 카피와 기간 표기 개선 (#187)
#   2026-06-18 최종민 — 코드 변경
"""prompts — strategic_insight_agent 에서 분리 (이동만, 동작 불변).

분리 근거: axis-infra docs/structure-tasks/agent-split-design.md (Phase 2 1단계)
"""

import copy
import json
from typing import Any

from src.agents.briefing.support import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    KST,
    _analysis_package,
    _analysis_package_from_sources,
    _compact_analysis_package,
    _compact_analysis_unit_for_display,
    _company_label,
    _dedupe_cards_for_prompt,
    _first_from_list,
    _first_int,
    _first_text,
    _int_list,
    _iso_or_none,
    _json_dict,
    _json_list,
    _nested_get,
    _optional_int,
    _parse_datetime,
    _safe_float,
    _str_values,
)


def _briefing_synthesis_context(
    *,
    briefing_basis: dict[str, Any],
    selected_cards: list[dict[str, Any]],
    period: dict[str, Any],
    user_context: str | None,
) -> dict[str, Any]:
    prompt_cards = _dedupe_cards_for_prompt(selected_cards)
    return {
        "period": {
            "date_from": period["date_from"].isoformat(),
            "date_to": period["date_to"].isoformat(),
            "period_label": period["label"],
        },
        "user_context": str(user_context or "").strip() or None,
        "source_card_ids": [card.get("id") for card in selected_cards if card.get("id")],
        "source_integrated_issue_ids": briefing_basis.get("source_integrated_issue_ids") or [],
        "current_deterministic_basis": _briefing_basis_synthesis_view(briefing_basis),
        "frontend_contract_target": _briefing_contract_schema_hint(),
        "analysis_units": [_compact_analysis_unit_for_display(card) for card in prompt_cards],
    }


def _briefing_basis_synthesis_view(briefing_basis: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "briefing_insight",
        "lead",
        "core_change",
        "common_pattern",
        "comparison_point",
        "hidden_conclusion",
        "strategy_implication",
        "recommended_action_basis",
        "action_details",
        "confidence",
    )
    return {key: copy.deepcopy(briefing_basis.get(key)) for key in keys if key in briefing_basis}


def _briefing_contract_schema_hint() -> dict[str, Any]:
    return {
        "executive_summary": "string",
        "briefing_insight": "string",
        "lead": {
            "finding": "string",
            "rationale": "string",
            "evidence_card_ids": ["CN-..."],
        },
        "common_pattern": {
            "finding": "string",
            "rationale": "string",
            "evidence_card_ids": ["CN-..."],
        },
        "comparison_point": {
            "finding": "string",
            "rationale": "string",
            "evidence_card_ids": ["CN-..."],
        },
        "hidden_conclusion": {
            "finding": "string",
            "rationale": "string",
            "evidence_card_ids": ["CN-..."],
        },
        "strategy_implication": {
            "finding": "string",
            "rationale": "string",
            "evidence_card_ids": ["CN-..."],
        },
        "immediate_trends": [
            {
                "title": "string",
                "peer_id": "string",
                "reason": "string",
                "source_name": "string",
                "published_at": "ISO-8601 datetime string",
                "related_card_id": "CN-...",
            }
        ],
        "watch_trends": [
            {
                "title": "string",
                "peer_id": "string",
                "reason": "string",
                "source_name": "string",
                "published_at": "ISO-8601 datetime string",
                "related_card_id": "CN-...",
            }
        ],
        "sections": [
            {
                "title": "핵심 인사이트 요약",
                "summary": "string",
                "bullets": ["string"],
                "related_card_ids": ["CN-..."],
            }
        ],
        "evidence_summary": ["string"],
        "recommended_action_basis": ["string"],
        "action_details": [
            {
                "action": "string",
                "why": "string",
                "use_case": "사업 우선순위",
                "evidence_card_ids": ["CN-..."],
            }
        ],
        "confidence": 0.0,
    }


def _briefing_synthesis_system_prompt() -> str:
    return "\n".join(
        [
            "# Persona",
            "당신은 SK AX Peer Intelligence 브리핑을 만드는 전략 분석 에디터입니다.",
            "",
            "# Source Boundary",
            "- integrated_issues가 사실 근거의 1차 저장소입니다.",
            "- card_news는 화면 이동과 저장 매핑을 위한 card id anchor로만 사용합니다.",
            "- card_news.summary_lines를 사실 판단 근거로 쓰지 않습니다.",
            "- legacy analysis_package는 analysis/implication 보조 근거로만 사용합니다.",
            "",
            "# Non-Negotiables",
            "- 입력 analysis_units 밖의 회사, 사건, 수치, 인과관계를 만들지 않습니다.",
            "- 원문 기사를 새로 요약하지 않습니다.",
            "- related_card_id와 evidence_card_ids는 입력 source_card_ids 안의 값만 사용합니다.",
            "- frontend 목업의 섹션 역할에 맞게 immediate/watch를 분리합니다.",
            "- 출력은 JSON 객체 하나만 반환합니다.",
            "- 내부 추론, self-check, markdown은 출력하지 않습니다.",
        ]
    )


def _briefing_synthesis_user_prompt(context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Task",
            (
                "아래 integrated issue 기반 analysis_units를 브리핑 계약 payload의 "
                "근거로 재구성하세요."
            ),
            (
                "목표는 화면 문장뿐 아니라 immediate_trends/watch_trends/sections에 "
                "들어갈 분석 판단을 만드는 것입니다."
            ),
            "",
            "# Output Schema",
            json.dumps(_briefing_contract_schema_hint(), ensure_ascii=False, indent=2),
            "",
            "# Contract Mapping",
            (
                "- executive_summary는 프론트 dailySnapshot.summary와 API "
                "executive_summary의 근거가 됩니다."
            ),
            (
                "- immediate_trends는 '오늘 바로 검토할 동향' 또는 "
                "'이번 주 핵심 변화'에 들어갈 항목입니다."
            ),
            "- watch_trends는 '지속 관찰할 동향' 또는 '연속 관찰 포인트'에 들어갈 항목입니다.",
            "- sections는 API BriefingSection 구조입니다.",
            "- evidence_summary는 어떤 출처/근거 체인이 사용됐는지 짧게 설명합니다.",
            "",
            "# Selection Rules",
            "- immediate_trends에는 영향도, 긴급성, 경쟁 구도 변화가 큰 항목을 둡니다.",
            "- watch_trends에는 후속 기사, 수주, 고객 확산, 규제/보안 검증이 필요한 항목을 둡니다.",
            "- 각 trend.reason은 단순 요약이 아니라 왜 immediate 또는 watch인지 설명합니다.",
            "- selected card가 여러 개면 한 카드만 대표 결론으로 과대 반영하지 않습니다.",
            "- action_details는 제안서 작성/화면 표시가 아니라 임원 의사결정 행동으로 씁니다.",
            "",
            "# Input",
            json.dumps(context, ensure_ascii=False, indent=2, default=str),
        ]
    )


def _display_copy_system_prompt() -> str:
    return "\n".join(
        [
            "# Persona Handoff",
            ("당신은 SK AX 임원 브리핑 화면의 수석 에디터이자 근거 검수자입니다."),
            (
                "당신의 책임은 analysis_units에 있는 "
                "통합/분석/시사점/분류/검증 결과만 사용해 화면용 문장을 정제하는 것입니다."
            ),
            "",
            "# Non-Negotiables",
            "- card_news.summary_lines는 판단 근거로 쓰지 않습니다.",
            "- 원문 기사 재요약을 하지 않습니다.",
            "- 근거에 없는 회사 주장, 수치, 사건, 인과관계를 만들지 않습니다.",
            "- 수치는 key_numbers.value와 context 의미가 함께 맞을 때만 사용합니다.",
            "- evidence_card_ids는 입력 source_card_ids 안의 실제 card_id만 사용합니다.",
            "- 출력은 JSON 객체 하나만 반환합니다.",
            "- 내부 추론이나 self-check 내용은 출력하지 않습니다.",
        ]
    )


def _display_copy_user_prompt(context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Task",
            "아래 analysis_units를 근거로 브리핑 화면용 표시문을 작성하세요.",
            (
                "current_display_structure는 필드 구조만 보여주는 레퍼런스입니다. "
                "문장 내용은 analysis_units에서 판단해 새로 작성하세요."
            ),
            "",
            "# Output Schema",
            json.dumps(_display_copy_schema_hint(), ensure_ascii=False, indent=2),
            "",
            "# Prompt Pattern Stack",
            "- Persona Handoff: 수석 브리핑 에디터로서 근거와 화면 문장을 동시에 검수합니다.",
            "- Input Flip: analysis_units의 원문 분석 결과를 화면 문장으로 변환합니다.",
            "- Constraint Box: 아래 Must/Cannot 제약을 지킵니다.",
            "- Few-Shot: 좋은/나쁜 문장 예시를 스타일 기준으로 삼습니다.",
            "- Self Evaluation: 반환 전 내부적으로만 중복, 근거성, 섹션 적합성을 점검합니다.",
            "",
            "# Input Reference Rules",
            (
                "- analysis_units에 있는 integrated_issue, analysis, implication을 "
                "Input Flip 레퍼런스로 사용해 화면 문장으로 변환합니다."
            ),
            (
                "- card_signal_index는 카드별 핵심 신호 체크리스트입니다. "
                "여러 카드가 들어오면 key_change_cards와 interpretation_flow가 "
                "한 카드에 쏠리지 않도록 이 목록을 먼저 확인합니다."
            ),
            (
                "- 시장/산업 신호와 피어사 움직임 사이에 공통 축이 있으면 함께 엮어 "
                "설명합니다. 예: AI 인프라·규제 대응 수요가 커지고, 피어사는 이를 "
                "고객 적용형 AI나 현장 운영형 AI 사업으로 구체화하고 있습니다."
            ),
            (
                "- industry_trend 또는 산업/규제/보안/인프라 성격의 카드가 있으면 "
                "market_signal에서 시장 맥락으로 활용합니다. 단, 관련성이 약하면 "
                "억지로 대표 결론으로 올리지 말고 피어사 실행 신호를 보완하는 배경으로 씁니다."
            ),
            (
                "- 단, 공통 축이 약하면 억지로 하나의 결론으로 묶지 말고 "
                "서로 다른 신호로 분리해 씁니다."
            ),
            (
                "- classification과 validation은 신뢰도와 분류 참고용이며 "
                "화면 문장에 그대로 노출하지 않습니다."
            ),
            "- key_numbers는 값과 context가 모두 맞을 때만 사용합니다.",
            "- current_display_structure의 string 값을 문장 초안으로 간주하지 않습니다.",
            (
                "- 화면 문장에는 내부 시스템 용어를 쓰지 않습니다. "
                "'선택된 카드', '피어 프로필', '프로필', 'analysis_units', "
                "'card_signal_index'를 그대로 쓰지 말고 사용자 언어로 풀어 씁니다."
            ),
            "",
            "# Constraint Box: Must",
            "- briefing_lead는 최대 2문장입니다.",
            (
                "- key_change_cards는 market_signal, competitor_move 두 슬롯만 유지하고 "
                "seq/display_label/insight_type/title/description/why_important만 씁니다."
            ),
            (
                "- selected card가 2개 이상이면 key_change_cards 두 슬롯 모두 "
                "최소 2개 card의 근거를 함께 반영합니다."
            ),
            (
                "- selected card가 2개 이상이면 key_change_cards.description에는 "
                "최소 2개 대표 회사/카드 신호가 실제로 드러나야 합니다."
            ),
            (
                "- competitor_move.title은 특정 기술명 하나가 아니라 경쟁사들이 "
                "무엇을 어떤 경쟁 방식으로 묶는지 보여줘야 합니다."
            ),
            (
                "- briefing_lead와 briefingSummaryLine 역할의 문장은 같은 내용을 반복하지 "
                "않습니다. 하나는 가장 중요한 시장/산업 신호를, 다른 하나는 피어사 "
                "움직임과 공통 방향을 설명하되 연결 가능한 축이 있으면 자연스럽게 엮습니다."
            ),
            (
                "- 특히 weekly/monthly 브리핑에서 여러 피어사가 입력되면 "
                "briefing_lead 또는 briefingSummaryLine 중 최소 하나에는 대표 피어사 "
                "2곳 이상과 각 움직임을 구체적으로 씁니다."
            ),
            (
                "- 피어사 움직임을 '경쟁사들이 강화하고 있습니다'처럼 뭉뚱그리지 말고, "
                "'LG CNS는 금융권 현대화, 포스코DX는 로봇 자율작업'처럼 회사별 실행 축을 "
                "드러냅니다."
            ),
            (
                "- interpretation_flow는 관찰된 변화, 평가축의 이동, 경쟁 구도 영향, "
                "전략 시사 4단계 순서를 유지합니다."
            ),
            ("- 각 interpretation_flow step은 seq/label/items만 포함합니다."),
            ("- step 자체에는 title/description을 쓰지 않습니다. 실제 문장은 items 안에만 씁니다."),
            ("- 각 step.items는 최소 1개, 최대 3개만 작성합니다."),
            (
                "- items는 근거가 충분한 것만 선택합니다. 3개를 채우기 위해 "
                "약한 항목을 만들지 않습니다."
            ),
            ("- title은 해당 섹션에서 사용자가 먼저 볼 핵심 결론 한 줄입니다."),
            "- title도 자연스러운 문장으로 쓰고, 가능한 한 '~합니다' 또는 '~있습니다'로 끝냅니다.",
            (
                "- description은 반드시 '근거 -> 그 근거가 title을 지지하는 이유 -> "
                "읽어야 할 의미' 순서가 드러나야 합니다."
            ),
            (
                "- description은 title을 다시 말하지 말고, title이 왜 그렇게 "
                "도출됐는지 실제 근거와 판단 연결고리를 설명합니다."
            ),
            (
                "- why_important는 '중요합니다'로 끝나는 평가가 아니라 고객 평가, "
                "오퍼링, 책임 조직, 자원 배분, 리스크 게이트 중 무엇을 바꿔야 하는지까지 말합니다."
            ),
            ("- 카드가 2개 이상이면 한 회사나 한 카드의 설명으로 전체 결론을 대체하지 않습니다."),
            (
                "- '경쟁사들', '수요 변화', '운영 성과', '리스크 감소' 같은 넓은 표현을 "
                "쓰면 같은 항목의 description에서 실제 회사/이슈 근거를 설명합니다."
            ),
            (
                "- 동일한 회사명+신호 표현은 처음 등장할 때만 온전하게 씁니다. "
                "이후에는 '이 운영 기반', '이 데이터 통제 신호', '앞선 수요 신호'처럼 "
                "문맥형 표현으로 바꿔 반복을 줄입니다."
            ),
            (
                "- 같은 회사명+신호 표현은 전체 출력에서 2회를 넘겨 반복하지 않습니다. "
                "반복이 필요하면 두 번째부터는 기능, 고객 평가 기준, 경쟁 방식, "
                "전략 실행 관점으로 바꿔 씁니다."
            ),
            (
                "- 같은 description 도입부를 반복하지 않습니다. 각 description은 "
                "확인된 사실, 고객 평가 변화, 경쟁 방식 변화, SK AX 실행 방향 중 "
                "서로 다른 역할로 시작합니다."
            ),
            (
                "- 단, 문맥형 표현 때문에 근거가 불명확해지면 회사명과 이슈를 다시 "
                "명확히 씁니다. 정확성이 반복 회피보다 우선입니다."
            ),
            ("- 모든 기본 노출 문장은 '~합니다' 또는 '~있습니다' 문체로 끝냅니다."),
            (
                "- 모든 기본 노출 문장은 완결된 한국어 문장이어야 하며, "
                "조사·명사구·접속절에서 끊기면 안 됩니다."
            ),
            "- 말줄임표(..., …, ⋯)를 쓰지 않습니다.",
            (
                "- 글자 수가 부담되면 문장을 자르지 말고 같은 의미를 "
                "더 짧은 완결 문장으로 다시 씁니다."
            ),
            "- title과 description은 같은 내용을 반복하지 않습니다.",
            (
                "- 같은 카드 안의 첫 문장과 둘째 문장은 각각 다른 역할이어야 합니다. "
                "첫 문장은 핵심 결론, 둘째 문장은 근거 또는 의미를 설명합니다."
            ),
            "",
            "# Internal Selection-Inference Procedure",
            "- 내부적으로만 아래 절차를 수행하고, 절차 내용은 JSON에 쓰지 마세요.",
            "1. 각 card_id에서 확인된 핵심 신호를 하나씩 뽑습니다.",
            "2. 여러 카드가 공유하는 상위 변화와 카드별로 다른 움직임을 분리합니다.",
            "3. market_signal에는 공유되는 시장 변화만 씁니다.",
            (
                "4. competitor_move에는 각 경쟁사의 움직임이 하나의 경쟁 흐름으로 "
                "묶이는 이유를 씁니다."
            ),
            (
                "5. key_change_cards를 작성한 뒤, 각 카드가 최소 한 번 이상 "
                "핵심 변화 판단에 기여했는지 확인합니다."
            ),
            "6. interpretation_flow 4단계는 각각 다른 질문에 답합니다.",
            "- 관찰된 변화: 실제로 무엇이 확인됐나?",
            "- 평가축의 이동: 고객/시장 판단 기준은 무엇으로 이동하나?",
            "- 경쟁 구도 영향: 경쟁사 메시지와 경쟁 방식은 어떻게 달라지나?",
            "- 전략 시사: SK AX는 고객군, 오퍼링, 책임 조직, 자원 배분을 어떻게 바꿔야 하나?",
            "",
            "# Constraint Box: Cannot",
            "- market_reading 키를 생성하지 않습니다.",
            "- sk_ax_view 키를 생성하지 않습니다.",
            "- core_change.title 또는 core_change.summary를 생성하지 않습니다.",
            "- summary 또는 so_what 필드를 생성하지 않습니다.",
            "- 내부 taxonomy 값(ax, infra, analyst_report 등)을 화면 문장에 그대로 쓰지 않습니다.",
            "- 매출 전망 수치를 프라이빗 AI 수요 금액처럼 바꿔 쓰지 않습니다.",
            "- 단계 title과 같은 문장을 items.title로 반복하지 않습니다.",
            "- 같은 description을 여러 item에서 반복하지 않습니다.",
            "- title을 description에서 같은 표현으로 반복하지 않습니다.",
            (
                "- key_change_cards의 market_signal.description과 "
                "competitor_move.description을 같은 의미로 쓰지 않습니다."
            ),
            "- 모든 description을 같은 도입부로 시작하지 않습니다.",
            (
                "- description을 '움직임은', '사례들은', '이러한 변화는'처럼 "
                "모호한 주어로 시작하지 않습니다."
            ),
            (
                "- '중요성이 커지고 있습니다', '중요한 역할을 합니다', "
                "'중요성을 부각시키고 있습니다', '강조하고 있습니다', "
                "'핵심 요소로 자리잡고 있습니다'처럼 이유 없는 중요도 표현을 "
                "결론으로 쓰지 않습니다."
            ),
            (
                "- '경쟁력을 강화하고 있습니다', '입지를 다지고 있습니다', "
                "'성장을 도모하고 있습니다'처럼 성과를 단정하는 표현은 "
                "근거에 같은 의미가 있을 때만 씁니다."
            ),
            "- 근거가 1개뿐인 항목을 전체 시장 결론처럼 과장하지 않습니다.",
            "- competitor_move의 title을 특정 회사 하나의 움직임으로 쓰지 않습니다.",
            "- why_important를 특정 회사의 이익이나 성장 전망만으로 좁히지 않습니다.",
            (
                "- '두각', '시장 입지 강화', '경쟁 우위 확보'처럼 강한 평가 표현은 "
                "analysis_units에 같은 의미의 근거가 있을 때만 씁니다."
            ),
            (
                "- 산업 범위는 근거에 나온 범위를 넘기지 않습니다. 예를 들어 "
                "제조/AX 근거를 임의로 더 넓은 산업 전체로 확대하지 않습니다."
            ),
            "",
            "# Section Logic",
            (
                "- 관찰된 변화: integrated_issue, business_signals, key_numbers, "
                "analysis.impact_reason에서 실제로 확인된 신호를 씁니다."
            ),
            ("- 평가축의 이동: 확인된 신호 때문에 고객 평가 기준이 어떻게 바뀌는지 씁니다."),
            (
                "- 경쟁 구도 영향: peer_implication, analysis_summary, market_signal을 "
                "사용해 경쟁 메시지나 경쟁 방식 변화를 씁니다."
            ),
            (
                "- 전략 시사: skax_implication.why_important, potential_impact, "
                "recommended_actions를 참고하되, 최종 문장은 임원이 실행할 회사 차원의 "
                "고객군/오퍼링/자원 배분 결정으로 씁니다."
            ),
            "",
            "# Lens Selection",
            (
                "각 interpretation_flow step은 아래 후보 렌즈 중 근거가 가장 강한 "
                "관점을 선택해 title/description에 반영하세요."
            ),
            "- 관찰된 변화 후보: 반복 신호, 새 수요, 숫자/규모 신호, 사업 역할 변화",
            "- 평가축의 이동 후보: 보안/통제, 운영 가능성, 성과 검증, 비용/리스크",
            "- 경쟁 구도 영향 후보: 메시지 재구성, 패키지화, 레퍼런스 경쟁, 성장 논리",
            "- 전략 시사 후보: 고객군 우선순위, 오퍼링 상품화, 책임 조직, 자원 배분, 리스크 게이트",
            (
                "- 전략 시사에는 제안서 작성, PoC 운영, 다음 모니터링 항목, 화면 표시 같은 "
                "프로그램 산출물 중심 행동을 쓰지 않습니다."
            ),
            (
                "후보에 맞지 않는 더 중요한 근거가 있으면 후보 밖 렌즈를 선택해도 됩니다. "
                "단, title과 description은 선택한 렌즈에 정확히 맞아야 합니다."
            ),
            "",
            "# Few-Shot Style Guide",
            "Bad title: 경쟁사들은 수요 변화에 맞춰 움직이고 있습니다.",
            ("Good title: 경쟁사들은 기술 신호를 운영 패키지와 성장 논리로 묶고 있습니다."),
            "Bad description: LG CNS 관련 프라이빗 모델 구축 수요가 확인됩니다.",
            (
                "Good description: 프라이빗 모델 구축 수요는 고객이 AI 기능 자체보다 "
                "데이터 위치, 접근 권한, 운영 책임, 거버넌스를 함께 평가하게 만든다는 "
                "점에서 평가축 이동의 근거가 됩니다."
            ),
            "Bad description: 움직임은 제조 및 자동화 산업에서 데이터 관리가 중요함을 보여줍니다.",
            (
                "Good description: 프라이빗 모델 구축 수요와 로봇 운영 기반 신호가 함께 "
                "나오면서 고객 평가 기준은 기능 보유 여부보다 데이터 통제와 운영 가능성으로 "
                "이동합니다."
            ),
            "Bad why_important: 데이터 관리와 운영 SW 인프라가 핵심 요소로 자리잡고 있습니다.",
            (
                "Good why_important: 고객이 기술 보유 여부보다 도입 후 운영 책임과 "
                "성과 검증 근거를 보게 되므로, 제안에서는 실행 범위와 검증 기준을 "
                "먼저 제시해야 합니다."
            ),
            "Bad description: 현대오토에버의 로봇 운영 소프트웨어 인프라가 확인됩니다.",
            (
                "Good description: 이 운영 기반 신호는 로봇 도입 자체보다 현장 데이터 관리, "
                "제조 SW 연동, 운영 안정화 역량이 제조 AX 판단 기준으로 올라오고 있음을 "
                "보여줍니다."
            ),
            "Bad repeated description: 현대오토에버의 로봇 운영 소프트웨어 인프라가 확인됩니다.",
            (
                "Good rewritten description: 이 신호는 로봇 도입 자체보다 현장 데이터 관리와 "
                "운영 SW 연동 역량이 제조 AX 판단 기준으로 올라오고 있음을 보여줍니다."
            ),
            ("Bad step pattern: 모든 단계 description을 'A와 B가 확인됩니다'로 시작합니다."),
            (
                "Good step pattern: 관찰 단계는 확인된 신호, 평가축 단계는 고객 기준 변화, "
                "경쟁 구도 단계는 메시지 재구성, 전략 시사 단계는 SK AX 실행 방향으로 "
                "각각 다르게 씁니다."
            ),
            (
                "Bad key_change split: market_signal은 A회사만, competitor_move는 B회사만 "
                "설명합니다."
            ),
            (
                "Good key_change split: market_signal은 여러 카드에서 공통으로 감지된 "
                "시장 변화, competitor_move는 각 경쟁사 움직임이 하나의 경쟁 방식 변화로 "
                "묶이는 이유를 설명합니다."
            ),
            (
                "Good 연결 pattern: 시장에서는 AI 인프라와 규제 대응 요구가 커지고, "
                "피어사들은 이를 중소기업 적용 지원이나 현장 운영형 AI 사업으로 "
                "구체화하고 있습니다."
            ),
            (
                "Good industry 활용: AI 보안 위협 증가 같은 산업 카드는 단독 결론으로 "
                "과대 포장하지 말고, 금융권 AX·로봇 자동화 같은 피어사 실행 움직임이 "
                "왜 보안·거버넌스 기준과 함께 읽혀야 하는지 설명합니다."
            ),
            ("Bad 연결 pattern: 관련성이 약한 카드들을 모두 같은 흐름이라고 단정합니다."),
            (
                "Bad key_change description: 한 회사의 역할만 설명하고 전체 시장 신호처럼 "
                "확대합니다."
            ),
            (
                "Good key_change description: 입력된 대표 카드들의 서로 다른 신호를 먼저 "
                "짚고, 그 신호들이 왜 하나의 시장 변화나 경쟁 방식 변화로 묶이는지 "
                "설명합니다."
            ),
            "",
            "# Internal Self Evaluation",
            "- 반환 전에 내부적으로만 확인하세요.",
            "- 각 title은 해당 섹션 역할의 결론인가?",
            "- 각 description은 title의 이유와 근거를 설명하는가?",
            "- 같은 근거를 같은 문장으로 반복하지 않았는가?",
            "- key_change_cards와 interpretation_flow가 같은 내용을 같은 표현으로 반복하지 않는가?",
            "- 같은 화면 카드 안에서 title/description/why_important가 서로 다른 역할을 하는가?",
            "- 문장이 말줄임표 없이 끝까지 완결되어 있는가?",
            "- 모든 기본 노출 문장의 말투가 '~합니다' 또는 '~있습니다'로 통일되어 있는가?",
            (
                "- 화면 문장에 '선택된 카드', '피어 프로필', '프로필' 같은 "
                "내부 용어가 남아 있지 않은가?"
            ),
            "- competitor_move와 전략 시사가 한 회사에만 쏠리지 않는가?",
            "- key_change_cards.description이 입력 카드 중 최소 2개 이상의 대표 신호를 반영하는가?",
            "- 모든 수치와 회사 표현은 analysis_units에 근거가 있는가?",
            "- 이 self evaluation 결과는 JSON에 포함하지 마세요.",
            "",
            "# Input",
            json.dumps(context, ensure_ascii=False, indent=2),
        ]
    )


def _display_copy_revision_prompt(
    context: dict[str, Any],
    draft: dict[str, Any],
    issues: list[str],
) -> str:
    return "\n".join(
        [
            "# Task",
            "아래 draft_display_copy를 다시 정제하세요.",
            "목표는 새 내용을 만드는 것이 아니라, 감지된 품질 이슈만 고치는 것입니다.",
            "",
            "# Detected Issues",
            json.dumps(issues, ensure_ascii=False, indent=2),
            "",
            "# Revision Rules",
            "- Output Schema는 1차 프롬프트와 동일하게 유지합니다.",
            "- 말줄임표(..., …, ⋯)를 쓰지 않습니다.",
            "- 문장이 끊겼으면 마침표만 붙이지 말고 같은 의미의 짧은 완결 문장으로 다시 씁니다.",
            "- 기본 노출 문장은 '~합니다' 또는 '~있습니다' 문체로 통일합니다.",
            (
                "- title/description/why_important가 같은 내용을 반복하면 "
                "각 필드 역할에 맞게 다시 분리합니다."
            ),
            "- market_reading, sk_ax_view, core_change, summary, so_what은 만들지 않습니다.",
            "- 회사명은 근거를 명확히 해야 할 때만 씁니다.",
            ("- 이미 한 번 설명한 회사+신호 조합은 다음 항목에서 문맥형 표현으로 바꿉니다."),
            (
                "- key_change_cards는 회사별 카드가 아닙니다. market_signal은 "
                "여러 카드의 공통 시장 변화, competitor_move는 경쟁 방식 변화를 씁니다."
            ),
            (
                "- selected card가 2개 이상이면 key_change_cards.description에는 "
                "최소 2개 대표 회사/카드 신호가 실제로 드러나야 합니다."
            ),
            (
                "- interpretation_flow의 4단계는 서로 다른 질문에 답해야 합니다: "
                "확인된 변화, 평가 기준 변화, 경쟁 방식 변화, SK AX 실행 방향."
            ),
            "- interpretation_flow.steps 안에는 반드시 items를 생성합니다.",
            "- 각 step.items는 최소 1개, 최대 3개만 작성합니다.",
            "- interpretation_flow.steps[*]에 title 또는 description을 생성하지 않습니다.",
            (
                "- description은 title의 반복이 아니라 근거와 논리입니다. "
                "왜 그 title이 나왔는지 읽고 납득되어야 합니다."
            ),
            (
                "- '두각', '시장 입지 강화', '경쟁 우위 확보'처럼 강한 평가 표현은 "
                "근거에 같은 의미가 있을 때만 씁니다."
            ),
            (
                "- '중요성이 커지고 있습니다', '강조하고 있습니다', "
                "'경쟁력을 강화하고 있습니다' 같은 표현으로 끝내지 말고 "
                "근거가 고객 평가/제안/경쟁 방식에 어떤 변화를 만드는지 쓰세요."
            ),
            "- 근거 범위를 넘어 산업 범위를 넓히지 않습니다.",
            "- 근거에 없는 수치, 사건, 인과관계는 추가하지 않습니다.",
            "- JSON 객체 하나만 반환합니다.",
            "",
            "# Reference Analysis Units",
            json.dumps(context.get("analysis_units") or [], ensure_ascii=False, indent=2),
            "",
            "# Card Signal Index",
            json.dumps(context.get("card_signal_index") or [], ensure_ascii=False, indent=2),
            "",
            "# Draft Display Copy",
            json.dumps(draft, ensure_ascii=False, indent=2),
        ]
    )


def _display_copy_schema_hint() -> dict[str, Any]:
    return {
        "key_summary": "string",
        "briefing_lead": "string",
        "key_change_cards": [
            {
                "seq": 1,
                "display_label": "시장 신호",
                "insight_type": "market_signal",
                "title": "string",
                "description": "string",
                "why_important": "string",
                "evidence_card_ids": ["CN-..."],
            },
            {
                "seq": 2,
                "display_label": "경쟁사 움직임",
                "insight_type": "competitor_move",
                "title": "string",
                "description": "string",
                "why_important": "string",
                "evidence_card_ids": ["CN-..."],
            },
        ],
        "interpretation_flow": {
            "steps": [
                {
                    "seq": 1,
                    "label": "관찰된 변화",
                    "items": [
                        {
                            "seq": 1,
                            "title": "관찰된 변화 포인트 한 줄",
                            "description": "해당 포인트의 근거와 의미 1~3문장",
                            "evidence_card_ids": ["CN-..."],
                        }
                    ],
                    "evidence_card_ids": ["CN-..."],
                },
                {
                    "seq": 2,
                    "label": "평가축의 이동",
                    "items": [
                        {
                            "seq": 1,
                            "title": "평가축 이동 포인트 한 줄",
                            "description": "해당 포인트의 근거와 의미 1~3문장",
                            "evidence_card_ids": ["CN-..."],
                        }
                    ],
                    "evidence_card_ids": ["CN-..."],
                },
                {
                    "seq": 3,
                    "label": "경쟁 구도 영향",
                    "items": [
                        {
                            "seq": 1,
                            "title": "경쟁 구도 영향 포인트 한 줄",
                            "description": "해당 포인트의 근거와 의미 1~3문장",
                            "evidence_card_ids": ["CN-..."],
                        }
                    ],
                    "evidence_card_ids": ["CN-..."],
                },
                {
                    "seq": 4,
                    "label": "전략 시사",
                    "items": [
                        {
                            "seq": 1,
                            "title": "전략 시사 실행 포인트 한 줄",
                            "description": "해당 포인트의 근거와 의미 1~3문장",
                            "evidence_card_ids": ["CN-..."],
                        }
                    ],
                    "evidence_card_ids": ["CN-..."],
                },
            ]
        },
    }
