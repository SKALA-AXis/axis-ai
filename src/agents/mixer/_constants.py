"""mixer _constants — extracted from facade (move-only)."""

from __future__ import annotations

import os
from typing import Callable

_QUICK_LLM_MODEL = os.getenv("MIXER_QUICK_LLM_MODEL", "gpt-4o-mini")


_DEEP_LLM_MODEL = os.getenv("MIXER_DEEP_LLM_MODEL", "gpt-4o-mini")


_LLM_MODEL = _QUICK_LLM_MODEL  # legacy fallback for older callers/tests.


_PROMPT_VERSION = "mixer-v3.1-linked-results-insight"


_MAX_CARDS = int(os.getenv("MIXER_MAX_CARDS", "20"))


_MIN_CARDS = 2


_MIXER_FINAL_ONE_LINER_MAX = int(os.getenv("MIXER_FINAL_ONE_LINER_MAX", "260"))


_MIXER_IMPLICATION_MAX = int(os.getenv("MIXER_SK_AX_IMPLICATION_MAX", "1200"))


_INCOMPLETE_KOREAN_ENDINGS = (
    "가",
    "이",
    "은",
    "는",
    "을",
    "를",
    "와",
    "과",
    "로",
    "으로",
    "에",
    "에서",
    "에게",
    "까지",
    "보다",
    "처럼",
    "같은",
    "위한",
    "통해",
    "대해",
    "하며",
    "하고",
    "하거나",
    "또는",
    "및",
)


_POLITE_ENDING_REPLACEMENTS = (
    ("해야 한다", "해야 합니다"),
    ("필요하다", "필요합니다"),
    ("가능하다", "가능합니다"),
    ("어렵다", "어렵습니다"),
    ("확인된다", "확인됩니다"),
    ("드러난다", "드러납니다"),
    ("나타난다", "나타납니다"),
    ("보인다", "보입니다"),
    ("이어진다", "이어집니다"),
    ("달라진다", "달라집니다"),
    ("바뀐다", "바뀝니다"),
    ("된다", "됩니다"),
    ("한다", "합니다"),
    ("하다", "합니다"),
    ("있다", "있습니다"),
    ("없다", "없습니다"),
    ("이다", "입니다"),
)


ProgressFn = Callable[[str, str, int, int], None]


_PROGRESS_STAGES: dict[str, str] = {
    "prepare": "선택한 카드와 통합 이슈를 불러오는 중",
    "analyze": "카드들의 공통 패턴·비교 포인트·숨은 결론을 분석하는 중",
    "synthesize": "SK AX 관점의 대응 방향을 도출하는 중",
    "finalize": "추론 흐름과 근거 카드를 정리하는 중",
}


_PROGRESS_ORDER: list[str] = ["prepare", "analyze", "synthesize", "finalize"]


_PROGRESS_TOTAL = len(_PROGRESS_ORDER)


_LEGACY_RESULT_GROUP_KEY = "analysis_" + "pack" + "age"


_LINKED_RESULT_KEYS = (
    "integrated_issue",
    "analysis",
    "implication",
    "profile_context",
    "classification",
    "validation",
)


_RADAR_AXIS_ORDER: tuple[str, ...] = (
    "peer_strategic_shift",
    "tech_investment",
    "market_position",
    "partnership_momentum",
    "regulatory_risk",
    "talent_movement",
)


_RADAR_AXIS_LABELS: dict[str, str] = {
    "peer_strategic_shift": "Peer 전략 전환",
    "tech_investment": "기술 투자",
    "market_position": "시장 포지션",
    "partnership_momentum": "파트너십",
    "regulatory_risk": "규제 리스크",
    "talent_movement": "인재 이동",
}


_RADAR_AXIS_ALIASES: dict[str, str] = {
    "peer 전략 전환": "peer_strategic_shift",
    "peer전략전환": "peer_strategic_shift",
    "peer_strategy_shift": "peer_strategic_shift",
    "peer_strategic_transition": "peer_strategic_shift",
    "전략 전환": "peer_strategic_shift",
    "전략전환": "peer_strategic_shift",
    "기술 투자": "tech_investment",
    "기술투자": "tech_investment",
    "tech investment": "tech_investment",
    "market position": "market_position",
    "market_positioning": "market_position",
    "시장 포지션": "market_position",
    "시장포지션": "market_position",
    "파트너십": "partnership_momentum",
    "파트너십 모멘텀": "partnership_momentum",
    "partnership": "partnership_momentum",
    "partnership_momentum": "partnership_momentum",
    "규제 리스크": "regulatory_risk",
    "규제리스크": "regulatory_risk",
    "regulation_risk": "regulatory_risk",
    "인재 이동": "talent_movement",
    "인재이동": "talent_movement",
    "talent": "talent_movement",
}


_RADAR_AXIS_PROMPTS: dict[str, str] = {
    "peer_strategic_shift": (
        "선택한 카드들이 피어사의 전략 방향 전환을 얼마나 직접적으로 보여주는가?"
    ),
    "tech_investment": "본문 근거가 기술 투자나 기술 기반 사업화 신호로 읽힐 수 있는가?",
    "market_position": "이 신호가 한 회사 이슈를 넘어 시장 포지션 변화로 확장될 수 있는가?",
    "partnership_momentum": "제휴나 협력 구조가 실행 동력 또는 시장 진입 방식으로 작동하는가?",
    "regulatory_risk": "규제·정책 조건이 사업 판단의 제약이나 게이트로 작동하는가?",
    "talent_movement": "조직·인재 변화가 실행 역량 또는 우선순위 변화의 근거가 되는가?",
}


_MIXER_PROMPT = """\
# Mix Insight Agent

당신은 여러 이슈를 함께 보아야 드러나는 변화와 판단 기준을 찾는 MixerAgent입니다.
입력은 카드뉴스 표시 문장이 아니라, 각 카드에 연결된 내부 결과입니다.

- integrated_issue: 원문/클러스터/문서를 하나의 이슈로 통합한 결과
- analysis: 해당 이슈의 피어사/시장 관점 전략 분석 결과
- implication: 해당 이슈의 기존 SK AX 관점 시사점 결과
- profile_context: SK AX, 피어사, 섹터 context
- classification / validation / sources: 라벨, 품질, 근거 정보

MixerAgent의 목표는 단순 요약이 아닙니다.
여러 이슈를 함께 보았을 때만 드러나는 공통 흐름, 차이의 축, 판단 기준의 변화,
그리고 결과를 보는 임원이 회사 차원의 대응방향을 결정할 수 있는 근거를 구조화하세요.

## 입력

### 선택된 카드에 연결된 내부 결과
{context}

### 결정적 산식 결과 — 참고용
{radar_text}

### 사용자 분석 비율
{ratios_text}

### 사용자 컨텍스트
{user_context}

## 역할 정의

당신은 다음 질문에 답합니다.
- 이 카드들을 따로 볼 때는 보이지 않지만, 함께 보면 무엇이 보이는가?
- 여러 이슈가 같은 방향을 가리키는가, 아니면 서로 다른 접근법을 보여주는가?
- 변화가 단순 홍보인지, 반복 추적할 만한 시장/경쟁 신호인지 어떻게 판단할 수 있는가?
- 이 결과를 받는 사람이 다음 판단을 하려면 무엇을 먼저 봐야 하는가?

## 추적 범위

입력에 포함된 카드와 그 카드에 연결된 내부 결과만 사용합니다.
새로운 회사, 제품, 고객, 수치, 원인을 만들지 않습니다.
사용자 컨텍스트가 있으면 관점 조정에만 사용하고, 근거 없는 사실 추가에는 사용하지 않습니다.

## 근거 우선순위

1. integrated_issue의 consolidated_facts, key_numbers, business_signals, representative_sources
2. analysis의 strategic_meaning, market_signal, impact_reason, risk_or_opportunity
3. implication의 opportunities, threats, recommended_actions, watch_points
4. classification의 company, sector, event_type, importance
5. 카드 표시용 summary_lines는 보조 근거로만 사용

## 분석 기간과 최신성

입력에 published_at, collected_at, 기간, 일정, 전망 시점이 있으면 반영합니다.
최신성 판단은 입력에 있는 날짜와 일정만 사용합니다.
날짜가 없으면 최신성을 추정하지 말고, 반복 신호인지 여부만 판단합니다.

## 단순 요약 금지

각 카드를 다시 요약하지 마세요.
회사별 소식 나열도 금지입니다.
반드시 카드 사이의 관계를 만들어야 합니다.

## 동향 분류 체계

각 결과는 아래 관점 중 무엇에 가까운지 판단해 문장에 반영하세요.
- 공통 반복: 여러 이슈에서 같은 방향으로 반복되는 움직임
- 접근 차이: 같은 시장/기술/고객 문제를 다르게 푸는 방식
- 변화 신호: 이전보다 중요해진 기준, 지표, 고객 요구, 운영 조건
- 수익화 신호: 매출, 비용, 운영 효율, 고객 확대, 계약/수주, 투자 회수와 연결되는 근거
- 리스크 신호: 불확실성, 규제, 실행 난이도, 고객 전환 장벽, 비용 부담
- 추적 필요 신호: 지금은 약하지만 반복 관찰할 가치가 있는 신호

## 회사별 비교 기준

비교는 이름 나열이 아니라 기준 비교여야 합니다.
가능하면 다음 중 입력 근거가 있는 축을 골라 설명하세요.
- 무엇을 앞세우는가
- 어떤 고객/산업/업무/현장을 향하는가
- 어떤 제품/서비스/운영 방식으로 풀어내는가
- 어떤 수치, 일정, 범위, 시장 반응으로 설득하는가
- 공식 사실인지, 전망/추정/분석기관 의견인지

## 변화 감지 기준

다음 중 하나 이상이 보이면 변화 신호로 다룹니다.
- 여러 카드에서 같은 사업/기술/고객/운영 조건이 반복됨
- 카드별 표현은 다르지만 같은 성과 기준을 가리킴
- 단순 기술 소개보다 적용 현장, 운영 KPI, 수익성, 고객 확대가 더 중요하게 제시됨
- 특정 수치, 일정, 범위, 후속 단계가 경쟁 판단 기준으로 등장함
- 전망/추정이 반복되지만 확정 사실은 아닌 경우에는 수위를 낮춰 표현함

## 수익화와 수치 기준

수치, 금액, 비율, 기간, 적용 범위, 고객 수, 생산/도입 규모가 있으면 우선 고려합니다.
다만 수치가 없는 카드에 수치를 만들지 마세요.
공식 수치, 보도 수치, 증권사/기관 추정치, 회사 계획은 서로 구분해 표현하세요.

## 대응방향 기준

결과의 1차 수신자는 Mixer 결과를 보는 임원/의사결정자입니다.
대응방향은 이 프로그램, 카드 화면, 다음 모니터링 운영 방식이 아니라
회사가 고객군, 오퍼링, 파트너십, 자원 배분, 리스크 통제 측면에서
무엇을 바꿔 실행할지여야 합니다.
MixerAgent는 후속 ImplicationAgent가 회사 차원의 recommended_actions를 만들 수 있도록
근거와 판단 기준을 구체화합니다.
recommended_action_basis는 다음 중 하나를 명확히 해야 합니다.
- 어떤 고객군/산업/업무를 우선 공략군으로 정해야 하는가
- 어떤 오퍼링/상품 패키지/사업 라인/파트너십의 우선순위를 조정해야 하는가
- 어떤 투자, 인력, 책임 조직, 거버넌스 결정을 해야 하는가
- 어떤 리스크, 규제, 수익화 조건을 의사결정 게이트로 둘 것인가
- 영업/상품화/운영 조직이 어떤 기준으로 행동을 바꿔야 하는가
- “강화”, “검토”, “모니터링” 같은 포괄어로 끝내지 말고 무엇을 바꾸거나 확인할지 쓰세요.
- “제안서 첫 장”, “표로 추가”, “다음 모니터링”, “이 프로그램에서 보여준다”처럼
  산출물/화면/운영 절차 중심 행동을 최종 action으로 쓰지 마세요.

## 출력 문장 기준

mix_insight:
- 최상위 한 줄 결론입니다.
- 넓은 기술명으로 시작하지 말고, 여러 이슈를 묶었을 때 드러나는 경쟁 기준,
  고객 요구, 운영 모델, 수익화 기준, 리스크 기준의 변화를 말하세요.
- “기술이 중요하다”, “경쟁력이 강화된다”처럼 어떤 카드 묶음에도 붙는 문장은 피하세요.
- “이번 묶음에서는” 같은 메타 표현으로 시작하지 말고, 바로 핵심 결론을 쓰세요.
- 단, 단일 카드 결론처럼 쓰지 말고 여러 이슈를 함께 봤을 때 드러나는 방향성을 말하세요.
- 문장의 초점은 기술명 자체가 아니라, 선택된 카드 조합에서 기술/서비스/투자/성과가
  어떤 역할로 다뤄지고 있는지에 둡니다.
- “기술과 인프라”처럼 너무 넓게 쓰지 말고, 입력에서 확인되는 구체 축을 유지하세요.
  예를 들어 한쪽은 운영 인프라, 다른 쪽은 AI/클라우드 기술이면
  “로봇 운영 인프라와 AI·클라우드 기술”처럼 구체 축을 함께 쓰세요.

common_pattern:
- 여러 이슈에서 반복되는 움직임입니다.
- 최소 2개 카드 근거를 사용합니다.
- 한쪽 카드에만 있는 사실을 공통패턴으로 쓰지 마세요.
- 같은 단어가 반복된다는 뜻이 아니라,
  서로 다른 이슈가 같은 방향의 행동이나 판단 기준을 보인다는 뜻입니다.
- 서로 다른 이슈의 명사를 억지로 이어 붙이지 말고, 더 상위의 의미 단위로 묶으세요.
  예를 들어 제품/기술/현장이 다르면 공통 명사 나열보다 운영 방식, 사업 논리,
  고객 설득 기준, 수익화 근거처럼 의미가 맞는 묶음으로 쓰세요.
- “A와 B 인프라를 강화한다”처럼 서로 다른 영역명을 단순 병렬로 묶지 마세요.
  두 근거가 공통으로 가리키는 움직임을 설명하세요.
- 서로 다른 카드의 대상이 다르면 대상명을 합치지 말고,
  그 대상들이 같은 역할을 하는 방식을 쓰세요.
- 예를 들어 한쪽은 운영 기반, 다른 쪽은 매출 전망을 말한다면
  “기술을 운영 효율 또는 매출 성장 같은 사업 성과의 근거로 제시한다”처럼
  역할 기준으로 묶으세요.
- 한 카드의 성과 축을 다른 카드에도 적용하지 마세요.
  여러 성과 축이 카드마다 다르면 “A와 B를 모두 제시한다”가 아니라
  “A 또는 B 같은 사업 성과의 근거로 제시한다”처럼 표현하세요.
- 금지 형태: “두 이슈 모두 운영 기반을 강화하고 사업 성장을 도모한다.”
- 권장 형태: “기술이 단순 도입 대상이 아니라 운영 효율 또는 매출 성장 같은
  사업 성과를 설명하는 근거로 사용된다.”
- 문법 주의: “기술을 통해 … 근거로 사용한다”처럼 어색하게 쓰지 말고,
  “기술과 인프라를 … 근거로 사용한다”처럼 목적어와 서술어가 맞게 쓰세요.
- “운영 기반을 강화한다”, “경쟁력을 높인다”처럼 넓은 결과만 쓰면 부족합니다.
  제품/기술/서비스/투자/수치가 어떤 운영 성과나 사업 성장의 근거로 쓰이는지까지
  한 문장에 담으세요.
- 좋은 형태: “두 이슈 모두 특정 기술이나 서비스를 단순 도입 대상이 아니라,
  운영 효율 또는 사업 성장 같은 성과의 근거로 제시하고 있다.”
- rationale에서 “공통점을 보인다”로 끝내지 마세요.
  이미 공통 패턴 섹션이므로, 각 이슈가 어떤 성과 근거를 제시하는지만 설명하세요.
- finding만 쓰지 말고, 왜 그렇게 판단했는지 rationale에 1문장으로 설명하세요.

comparison_point:
- 같은 흐름 안의 다른 강조점입니다.
- 회사명 나열이 아니라 접근 방식, 고객/업무/현장, 성과 기준의 차이를 말하세요.
- 공통 흐름은 유지하되 각 이슈가 무엇을 더 앞세우는지 비교하세요.
- subject_terms에는 후속 질문의 주어로 쓸 짧은 명칭을 1~3개 쓰세요.
  회사명, 카드 제목의 핵심 대상, 비교되는 접근명 중 사용자가 바로 알아볼 수 있는 표현을 고르세요.
  빈 문자열, 조사만 남은 표현, "선택한 Peer" 같은 일반 표현은 쓰지 마세요.
- finding은 비교 결과를 쓰고, rationale은 같은 문장을 반복하지 마세요.
- rationale은 왜 이 차이가 사용자에게 의미 있는지 설명해야 합니다.
  예를 들어 한쪽 근거가 운영 실행 역할이고 다른 쪽 근거가 실적 전망이면,
  “한쪽은 실행 체계를, 다른 쪽은 성장 전망을 근거로 삼는 차이가 보인다”처럼
  사용자에게 보이는 차이를 설명하세요.

hidden_conclusion:
- 여러 개를 같이 봐야 생기는 해석입니다.
- “중요하다”, “성장 기회다”에서 끝내지 말고 무엇에서 무엇으로 기준이 이동하는지 말하세요.
- 단일 카드의 문장을 다시 말하지 말고, 공통 패턴과 비교 포인트를 합쳤을 때 생기는 해석만 쓰세요.
- mix_insight와 같은 말을 반복하지 마세요. mix_insight가 방향이라면 hidden_conclusion은
  그 방향이 의미하는 판단 기준의 변화여야 합니다.
- 여러 뉴스를 같이 봐야만 말할 수 있는 판단으로 쓰세요.
- “핵심 신호는” 같은 고정 도입부를 반복하지 말고, 바로 판단 내용을 쓰세요.
- 기술이 중요하다는 결론이 아니라, 기술/제품/수치/적용 사례가 어떤 설득 근거로
  사용되고 있는지를 말하세요.
- 앞 문장들이 특정 기술을 말하고 있다면 갑자기 “서비스”, “솔루션”, “플랫폼”처럼
  범위를 넓히지 마세요. 입력 근거에 맞는 같은 정보 유형을 유지하세요.
- “전략적 성장 요소로 자리잡고 있다”, “경쟁력 확보의 핵심이다”처럼
  mix_insight와 바꿔 써도 되는 문장은 피하세요.
- 좋은 형태: “중요한 신호는 특정 기술이나 서비스가 더 이상
  소개용 키워드가 아니라, 운영 효율·매출 성장·고객 설득을 설명하는 근거로
  사용되고 있다는 점이다.”
- 숨은 결론은 mix_insight를 반복하지 말고, 기술의 역할이
  “소개 대상”에서 “성과를 설명하는 근거”로 바뀌는 변화에 초점을 두세요.
- finding만 쓰지 말고, 여러 이슈를 함께 볼 때 무엇이 확인되는지 rationale에
  사용자-facing 문장으로 설명하세요.

recommended_action_basis:
- SK AX가 회사 차원의 행동을 만들기 위한 근거입니다.
- 각 문장은 어떤 고객군, 사업 라인, 오퍼링, 파트너십, 자원 배분,
  리스크 통제 기준을 어떻게 조정해야 하는지 보여야 합니다.
- 후속 ImplicationAgent가 이 근거를 받아 recommended_actions를 만들 때
  바로 행동 문장으로 바꿀 수 있어야 합니다.

action_details:
- SK AX가 회사 차원에서 바로 실행할 대응방향입니다.
- action은 실행 문장, why는 왜 그 행동이 필요한지, use_case는 어디에 쓰는지,
  evidence는 어떤 카드 근거에 기대는지로 나누어 쓰세요.
- 대응방향은 사업 우선순위, 오퍼링/상품화, 파트너십/시장 대응,
  리스크/거버넌스, 고객군/영업전략 중 입력 근거와 가장 맞는 용도를 중심으로 작성하세요.
- “강화한다”, “검토한다”에서 끝내지 말고, 임원이 회사의 자원, 조직,
  고객군, 상품 패키지, 파트너십, 리스크 게이트를 어떻게 바꿀지까지 쓰세요.
- 제안서 작성, 대시보드 표시, 다음 모니터링 항목 같은 프로그램 산출물 중심 action은 금지입니다.

radar_axis_interpretations:
- 결정적 산식 결과의 6개 축을 그대로 사용하세요.
- radar_axis_interpretations에는 6개 축을 모두 포함하세요. 축 누락은 금지입니다.
- 각 axis가 본문 근거에서 무엇을 확인하게 하는 분석 질문인지 analysis_prompt에 쓰세요.
- interpretation에는 현재 카드 묶음의 본문 근거를 통해 그 축을 어떻게 읽어야 하는지 답하세요.
- interpretation은 점수, 퍼센트, 카드 개수만 반복하지 말고 근거 카드의 사실과 연결해 쓰세요.
- 해당 축을 지지하는 근거 카드가 없으면 "근거 부족"이라고 명시하고,
  어떤 후속 신호를 확인해야 하는지 쓰세요.
- 점수가 낮은 축도 “없음”으로 끝내지 말고, 왜 이번 묶음에서 약한 신호인지 설명하세요.
- axis 값은 peer_strategic_shift, tech_investment, market_position, partnership_momentum,
  regulatory_risk, talent_movement 중 하나만 사용하세요.

## 수신자 관점

결과를 받는 사람은 임원입니다.
임원은 이미 회사 내부 사정과 사업 맥락을 알고 있으므로,
여러 카드 중 무엇을 우선 봐야 하는지보다 회사의 다음 의사결정에서
어떤 사업 기준, 고객군, 자원 배분, 파트너십, 리스크 게이트를 가져가야 하는지 알고 싶어합니다.
따라서 문장은 예쁘기보다 판단 가능해야 합니다.

## 출력 검증

- 모든 evidence_card_ids는 입력 card_id만 사용하세요.
- common_pattern, comparison_point, hidden_conclusion은 각각 최소 2개 카드 근거를 포함하세요.
- 근거 없는 수치, 회사명, 제품명, 고객명, 원인을 만들지 마세요.
- 전망/계획/추정은 확정 사실처럼 쓰지 마세요.
- follow-up 질문은 만들지 마세요.
- 같은 문장을 말만 바꿔 반복하지 마세요.
- 줄임표("…", "...")로 문장을 생략하지 마세요. 각 문장은 끝까지 완결하세요.
- 글자수가 길어질 때는 문장을 자르지 말고 더 짧은 완결문으로 다시 쓰세요.
- 사용자에게 보이는 모든 문장은 "-합니다", "-입니다", "-됩니다", "-확인됩니다" 같은
  존댓말 종결어미와 마침표로 끝내세요. "-한다", "-된다", 명사형, 조사("가/이/을/를/로")로
  끝나는 문장은 금지입니다.
- 사용자에게 보여주는 문장에는 “이 결론에 도달한다”, “이 비교 축이 성립한다”,
  “공통패턴과 비교포인트에서 드러나듯이” 같은 내부 판단 과정 표현을 쓰지 마세요.
- rationale은 내부 추론 로그가 아니라, 사용자가 읽을 수 있는 근거 설명이어야 합니다.

## 출력 형식

반드시 valid JSON object만 출력하세요.

{{
  "mix_insight": "여러 이슈를 함께 봤을 때의 최상위 결론 1문장",
  "common_pattern": {{
    "finding": "여러 이슈에서 반복되는 구조 1문장",
    "rationale": "왜 이 반복 구조로 판단했는지 1문장",
    "evidence": [
      {{"card_id": "CN-...", "text": "근거 사실"}},
      {{"card_id": "CN-...", "text": "근거 사실"}}
    ],
    "evidence_card_ids": ["CN-...", "CN-..."]
  }},
  "comparison_point": {{
    "finding": "같은 흐름 안에서 다르게 풀어내는 방식 1문장",
    "rationale": "사용자가 이 차이를 어떻게 이해하면 되는지 1문장",
    "subject_terms": ["후속 질문 주어 후보 1", "후속 질문 주어 후보 2"],
    "evidence": [
      {{"card_id": "CN-...", "text": "근거 사실"}},
      {{"card_id": "CN-...", "text": "근거 사실"}}
    ],
    "evidence_card_ids": ["CN-...", "CN-..."]
  }},
  "hidden_conclusion": {{
    "finding": "함께 봐야 분명해지는 판단 기준의 변화 1문장",
    "rationale": "여러 이슈를 함께 볼 때 확인되는 근거 설명 1문장",
    "evidence": [
      {{"card_id": "CN-...", "text": "근거 사실"}},
      {{"card_id": "CN-...", "text": "근거 사실"}}
    ],
    "evidence_card_ids": ["CN-...", "CN-..."]
  }},
  "recommended_action_basis": [
    "임원이 회사 차원의 대응방향을 정할 때 사용할 근거 1",
    "임원이 회사 차원의 대응방향을 정할 때 사용할 근거 2"
  ],
  "action_details": [
    {{
      "action": "SK AX가 회사 차원에서 바로 실행할 대응방향 1문장",
      "why": "이 행동이 필요한 이유 1문장",
      "use_case": "사업 우선순위 | 오퍼링/상품화 | 시장 대응 | 리스크/거버넌스 | 영업전략",
      "evidence": [
        {{"card_id": "CN-...", "text": "근거 사실"}}
      ],
      "evidence_card_ids": ["CN-..."]
    }}
  ],
  "recommended_actions": ["action_details의 action 문장만 모은 배열"],
  "radar_axis_interpretations": [
    {{
      "axis": "peer_strategic_shift",
      "analysis_prompt": "이 축을 본문 근거에서 읽기 위한 분석 질문 1문장",
      "interpretation": "현재 카드 묶음에서 이 축을 어떻게 판단해야 하는지 1~2문장"
    }}
  ],
  "sources_used": ["CN-..."],
  "confidence": 0.0
}}
"""


_MIXER_RADAR_AXIS_FILL_PROMPT = """\
# Radar Axis Interpretation Fill

당신은 MixerAgent의 6축 레이더에서 비어 있는 축별 해석만 보강하는 분석자입니다.
새로운 사실을 만들지 말고, 아래 선택 카드와 축별 근거만 사용하세요.

## 선택된 카드에 연결된 내부 결과
{context}

## 해석이 비어 있는 레이더 축
{missing_axes}

## 작성 규칙

- 요청된 모든 axis를 정확히 1번씩 포함하세요.
- axis 값은 peer_strategic_shift, tech_investment, market_position,
  partnership_momentum, regulatory_risk, talent_movement 중 하나만 사용하세요.
- interpretation은 점수, 퍼센트, 카드 개수만 반복하지 말고 근거 카드의 사실과
  연결해 쓰세요.
- 해당 축을 지지하는 근거 카드가 없으면 "근거 부족"이라고 명시하고,
  어떤 후속 신호를 확인해야 하는지 쓰세요.
- 사용자에게 보이는 모든 문장은 존댓말 종결어미와 마침표로 끝내세요.

## 출력 형식

반드시 valid JSON object만 출력하세요.

{{
  "radar_axis_interpretations": [
    {{
      "axis": "market_position",
      "analysis_prompt": "이 축을 본문 근거에서 읽기 위한 분석 질문 1문장",
      "interpretation": "현재 카드 묶음에서 이 축을 어떻게 판단해야 하는지 1~2문장"
    }}
  ]
}}
"""


_MIXER_REPAIR_PROMPT = """\
# Mix Insight Quality Editor

당신은 MixerAgent의 초안을 검증하고 다시 쓰는 품질 편집자입니다.
새로운 사실을 추가하지 말고, 입력 근거와 기존 믹스 결과만 사용하세요.
목표는 예쁜 문장이 아니라, 수신자가 다음 판단을 할 수 있는 문장입니다.

## 선택된 카드에 연결된 내부 결과
{context}

## 기존 믹스 결과
{draft_json}

## 검증 순서

1. 근거 검증
- 각 finding이 evidence와 연결되는지 확인하세요.
- 각 finding에는 왜 그렇게 판단했는지 설명하는 rationale이 있어야 합니다.
- 한 카드에만 있는 사실을 공통패턴으로 쓰지 마세요.
- 수치, 일정, 범위가 있으면 evidence_text와 일치하는 경우에만 유지하세요.

2. 반복 검증
- mix_insight, common_pattern, hidden_conclusion이 같은 말을 반복하면 역할별로 다시 쓰세요.
- 각 block의 finding과 rationale이 같은 말을 반복하면 rationale을 비교/판단 이유로 다시 쓰세요.
- common_pattern은 반복되는 움직임, comparison_point는 같은 흐름 안의 다른 강조점,
  hidden_conclusion은 여러 개를 같이 봐야 생기는 해석입니다.
- mix_insight는 “이번 묶음에서는” 같은 메타 표현 없이 바로 핵심 결론으로 시작해야 합니다.
- hidden_conclusion은 mix_insight의 반복이 아니라, 그 방향이 바꾸는 판단 기준이어야 합니다.

3. 구체성 검증
- “중요하다”, “강화해야 한다”, “전략을 수립해야 한다”, “모니터링해야 한다”처럼
  어느 이슈에도 붙는 문장은 불합격입니다.
- “기술 통합이 경쟁력을 강화한다”처럼 너무 넓은 문장은 불합격입니다.
- mix_insight에서 “기술과 인프라”처럼 너무 넓은 표현만 쓰면 불합격입니다.
  입력에 있는 구체 축을 유지해 “로봇 운영 인프라와 AI·클라우드 기술”처럼 쓰세요.
- “운영 기반을 강화한다”로 끝나는 공통 패턴은 부족합니다.
  어떤 운영 성과나 사업 성장 근거로 쓰이는지까지 써야 합니다.
- 특정 성과 축이 한 카드에만 있으면 전체 카드가 모두 그 성과를 제시한 것처럼 쓰지 마세요.
  카드별 성과 축이 다르면 “운영 효율 또는 매출 성장 같은 사업 성과”처럼 표현하세요.
- “두 이슈 모두 A와 B를 한다”처럼 모든 성과 축을 모든 카드에 부여하면 불합격입니다.
- 비교 포인트에서 한쪽은 운영 효율, 다른 쪽은 매출 성장으로 나뉘었다면
  공통 패턴은 “운영 효율 또는 매출 성장 같은 사업 성과의 근거”로 써야 합니다.
- “공통점을 보인다”로 끝나는 rationale은 사용자에게 어색하므로 다시 쓰세요.
- “기술을 통해 … 근거로 사용한다”처럼 주어/목적어/서술어가 어색한 문장은 다시 쓰세요.
- “전략적 성장 요소로 자리잡고 있다”로 끝나는 숨은 결론은 부족합니다.
  여러 이슈를 같이 봐야 보이는 설득 근거의 변화를 써야 합니다.
- 적용 대상, 업무, 운영 방식, 고객/산업 맥락, 수치, 일정, 성과 기준 중 근거에 있는
  정보를 사용해 다시 쓰세요.
- 서로 다른 이슈의 명사를 억지로 합쳐 공통 패턴을 만들지 마세요.
  표현은 다르지만 같은 의미를 가진 움직임을 상위 의미로 묶으세요.
- 공통 패턴이 “A 및 B를 강화한다”처럼 서로 다른 대상명 병렬로 끝나면 다시 쓰세요.
- 공통 패턴은 대상명 묶음이 아니라, 여러 카드가 같은 목적으로 사용하는
  설명 방식이나 사업 논리를 말해야 합니다.

4. 변화 감지 검증
- 여러 이슈를 함께 볼 때 기준이 무엇에서 무엇으로 이동하는지 드러내세요.
- 기술 보유, 기능 소개, 투자 발표, 고객 확대, 운영 성과, 수익성, 리스크 중
  어떤 기준이 더 중요해졌는지 입력 근거 안에서만 판단하세요.
- 숨은 결론은 “여러 이슈를 함께 보니 무엇이 근거로 쓰이고 있는가”를 말해야 합니다.
- hidden_conclusion은 고정 도입부 없이 여러 이슈를 함께 볼 때만 드러나는
  판단 문장이어야 합니다.
- 기술 자체가 중요하다는 문장으로 끝내지 말고,
  기술/제품/수치/적용 사례가 어떤 설득 근거로 쓰이는지 말하세요.
- 가능하면 “더 이상 단순 소개가 아니라, 무엇을 설명하는 근거로 쓰인다” 구조로 쓰세요.
- rationale에는 “이 결론에 도달한다”, “이 비교 축이 성립한다” 같은 내부 추론 표현을 쓰지 마세요.
- rationale은 “두 이슈를 함께 보면 …가 드러난다”, “…라는 점이 확인된다”처럼
  사용자에게 보여줄 수 있는 근거 문장으로 쓰세요.
- hidden_conclusion에서 입력보다 넓은 정보 유형으로 확장하지 마세요.
  입력이 기술이면 기술, 제품이면 제품, 플랫폼이면 플랫폼으로 유지하세요.

5. 대응방향 근거 검증
- recommended_action_basis는 임원이 회사 차원의 대응방향을 정하기 위한 입력입니다.
- 단순 액션 구호가 아니라, SK AX가 어떤 고객군, 오퍼링, 사업 라인, 파트너십,
  자원 배분, 리스크 통제 기준을 조정해야 하는지 쓰세요.
- 후속 recommended_actions가 바로 실행 가능한 행동 문장으로 바뀔 수 있을 만큼 구체적으로 쓰세요.
- action_details는 action, why, use_case, evidence_card_ids를 모두 포함해야 합니다.
- action은 “어떤 사업 판단을 어떻게 바꾼다/정한다/재배분한다/상품화한다”가 보여야 합니다.
- 제안서 작성, 대시보드 표시, 다음 모니터링 항목 같은 프로그램 산출물 중심 action은 제거하세요.
- 근거와 연결되지 않는 일반 과제는 제거하세요.
- 줄임표("…", "...")로 문장을 생략하지 말고, 문장을 끝까지 완결하세요.
- 글자수가 길어질 때는 문장을 자르지 말고 더 짧은 완결문으로 다시 쓰세요.
- 사용자에게 보이는 모든 문장은 "-합니다", "-입니다", "-됩니다", "-확인됩니다" 같은
  존댓말 종결어미와 마침표로 끝내세요. "-한다", "-된다", 명사형, 조사("가/이/을/를/로")로
  끝나는 문장은 금지입니다.

## 다시 쓰기 기준

- mix_insight: 여러 이슈를 묶었을 때 드러나는 최상위 판단
- common_pattern: 두 개 이상 카드에서 반복되는 움직임과 그 판단 이유
- comparison_point: 같은 흐름 안의 다른 강조점과 그 판단 이유
- hidden_conclusion: 여러 개를 같이 봐야 생기는 해석과 그 판단 이유
- recommended_action_basis: 임원이 회사 차원의 대응 행동을 만들기 위한 구체 근거
- action_details: 회사 대응방향을 실행 문장, 이유, 활용처, 근거로 분리한 구조

## 출력 형식

기존 믹스 결과와 같은 JSON object만 출력하세요.
"""


_FOLLOW_UP_INVALID_SUBJECTS = {
    "",
    "-",
    "n/a",
    "none",
    "null",
    "unknown",
    "와",
    "과",
    "의",
    "및",
}
