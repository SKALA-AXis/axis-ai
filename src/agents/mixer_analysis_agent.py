"""MixerAnalysisAgent — linked integrated/analysis/implication result mixer.

design: ``axis-ai/design/30-analysis/mixer-analysis.md``.

본 모듈은 카드 표시 문구를 다시 요약하지 않는다. 카드에 연결된
``integrated_issue`` / ``analysis`` / ``implication`` / ``profile_context`` 여러 건을
입력으로 받아, 단일 이슈로는 보이지 않는 공통 패턴 / 비교 포인트 / 숨은 결론 /
대응방향을 도출한다.

핵심 entry point:

    ``MixerAnalysisAgent().analyze(card_ids, ratios, user_context)`` — DB 카드 기반.
    ``MixerAnalysisAgent().analyze_items(items, ratios, user_context)`` — 로컬 목업/테스트 기반.

프론트 입력은 card_id 이지만, Mixer 의 실제 분석 재료는 카드 표시용 3줄 요약이 아니라
카드에 연결된 통합 결과, 분석 결과, 시사점 결과, 프로필 context 다.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.agents.implication_agent import ImplicationAgent
from src.db.postgres import SessionLocal
from src.middleware.analysis_ledger import with_ledger_writeback
from src.observability.langfuse_client import tracing_config
from src.services.agent_output_validation import (
    clip_final_one_liner,
    clip_implication,
    clip_string,
    confidence_in_range,
)

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "mixer-v3.1-linked-results-insight"
_MAX_CARDS = int(os.getenv("MIXER_MAX_CARDS", "20"))
_MIN_CARDS = 2
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

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.15,
            max_completion_tokens=3000,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm


# ──────────────────────────────────────────────────────────────────────────
# Prompt — design/30-analysis/mixer-analysis.md §6.2.
# ──────────────────────────────────────────────────────────────────────────

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
그리고 후속 ImplicationAgent가 대응방향을 만들 수 있는 근거를 구조화하세요.

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

SK AX 관점 대응방향은 후속 ImplicationAgent가 생성합니다.
MixerAgent는 그 전 단계에서 대응방향이 바로 실행 가능한 문장으로 나올 수 있도록
근거와 판단 기준을 구체화합니다.
recommended_action_basis는 다음 중 하나를 명확히 해야 합니다.
- 제안 메시지를 어떤 기준으로 바꿔야 하는가
- 고객 설득에서 어떤 증거를 더 앞세워야 하는가
- 어떤 수치/일정/운영 지표를 계속 추적해야 하는가
- 어떤 리스크나 불확실성을 확인해야 하는가
- “강화”, “검토”, “모니터링” 같은 포괄어로 끝내지 말고 무엇을 바꾸거나 확인할지 쓰세요.

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
- 권장 형태: “각 이슈는 기술을 운영 효율 또는 매출 성장 같은 사업 성과를
  설명하는 근거로 사용한다.”
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
- “핵심 신호는 …라는 점이다”처럼 여러 뉴스를 같이 봐야만 말할 수 있는 판단으로 쓰세요.
- 반드시 “핵심 신호는 …라는 점이다” 또는 이에 준하는 판단 문장으로 쓰세요.
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
- SK AX가 바로 쓸 수 있는 행동을 만들기 위한 근거입니다.
- 각 문장은 무엇을, 어떤 근거 기준으로, 어떻게 조정해야 하는지 보여야 합니다.
- 후속 ImplicationAgent가 이 근거를 받아 recommended_actions를 만들 때
  바로 행동 문장으로 바꿀 수 있어야 합니다.

action_details:
- SK AX가 바로 쓸 수 있는 대응방향입니다.
- action은 실행 문장, why는 왜 그 행동이 필요한지, use_case는 어디에 쓰는지,
  evidence는 어떤 카드 근거에 기대는지로 나누어 쓰세요.
- 대응방향은 제안 전략, 후속 모니터링, 레퍼런스 구성, 시장 대응 중
  입력 근거와 가장 맞는 용도를 중심으로 작성하세요.
- “강화한다”, “검토한다”에서 끝내지 말고, 무엇을 어떤 형식으로 바꾸거나
  어떤 지표를 추적할지까지 쓰세요.

## 수신자 관점

결과를 받는 사람은 여러 카드 중 무엇을 우선 봐야 하는지, 왜 같이 봐야 하는지,
다음 의사결정에서 어떤 기준을 가져가야 하는지 알고 싶어합니다.
따라서 문장은 예쁘기보다 판단 가능해야 합니다.

## 출력 검증

- 모든 evidence_card_ids는 입력 card_id만 사용하세요.
- common_pattern, comparison_point, hidden_conclusion은 각각 최소 2개 카드 근거를 포함하세요.
- 근거 없는 수치, 회사명, 제품명, 고객명, 원인을 만들지 마세요.
- 전망/계획/추정은 확정 사실처럼 쓰지 마세요.
- follow-up 질문은 만들지 마세요.
- 같은 문장을 말만 바꿔 반복하지 마세요.
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
    "후속 시사점 에이전트가 대응방향을 만들 때 사용할 근거 1",
    "후속 시사점 에이전트가 대응방향을 만들 때 사용할 근거 2"
  ],
  "action_details": [
    {{
      "action": "SK AX가 바로 실행할 대응방향 1문장",
      "why": "이 행동이 필요한 이유 1문장",
      "use_case": "제안 전략 | 후속 모니터링 | 레퍼런스 구성 | 시장 대응",
      "evidence": [
        {{"card_id": "CN-...", "text": "근거 사실"}}
      ],
      "evidence_card_ids": ["CN-..."]
    }}
  ],
  "recommended_actions": ["action_details의 action 문장만 모은 배열"],
  "sources_used": ["CN-..."],
  "confidence": 0.0
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
- hidden_conclusion은 반드시 “핵심 신호는 …라는 점이다” 또는 그에 준하는 판단 문장이어야 합니다.
- 기술 자체가 중요하다는 문장으로 끝내지 말고,
  기술/제품/수치/적용 사례가 어떤 설득 근거로 쓰이는지 말하세요.
- 가능하면 “더 이상 단순 소개가 아니라, 무엇을 설명하는 근거로 쓰인다” 구조로 쓰세요.
- rationale에는 “이 결론에 도달한다”, “이 비교 축이 성립한다” 같은 내부 추론 표현을 쓰지 마세요.
- rationale은 “두 이슈를 함께 보면 …가 드러난다”, “…라는 점이 확인된다”처럼
  사용자에게 보여줄 수 있는 근거 문장으로 쓰세요.
- hidden_conclusion에서 입력보다 넓은 정보 유형으로 확장하지 마세요.
  입력이 기술이면 기술, 제품이면 제품, 플랫폼이면 플랫폼으로 유지하세요.

5. 대응방향 근거 검증
- recommended_action_basis는 후속 ImplicationAgent의 입력입니다.
- 단순 액션 구호가 아니라, SK AX가 어떤 제안 메시지, 검증 기준, 모니터링 기준,
  고객 설득 기준을 조정해야 하는지 쓰세요.
- 후속 recommended_actions가 바로 실행 가능한 행동 문장으로 바뀔 수 있을 만큼 구체적으로 쓰세요.
- action_details는 action, why, use_case, evidence_card_ids를 모두 포함해야 합니다.
- action은 “무엇을 어떻게 바꾼다/만든다/추적한다/비교한다”가 보여야 합니다.
- 근거와 연결되지 않는 일반 과제는 제거하세요.

## 다시 쓰기 기준

- mix_insight: 여러 이슈를 묶었을 때 드러나는 최상위 판단
- common_pattern: 두 개 이상 카드에서 반복되는 움직임과 그 판단 이유
- comparison_point: 같은 흐름 안의 다른 강조점과 그 판단 이유
- hidden_conclusion: 여러 개를 같이 봐야 생기는 해석과 그 판단 이유
- recommended_action_basis: SK AX가 바로 쓸 수 있는 행동을 만들기 위한 구체 근거
- action_details: 대응방향을 실행 문장, 이유, 활용처, 근거로 분리한 구조

## 출력 형식

기존 믹스 결과와 같은 JSON object만 출력하세요.
"""


class MixerAnalysisAgent:
    """3-phase per_card / cross_card / synthesis CoT 카드 분석 agent."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    @with_ledger_writeback("MixerAnalysisAgent")
    async def analyze(
        self,
        card_ids: list[str],
        ratios: dict | None = None,
        user_context: str | None = None,
    ) -> dict:
        """N 카드 선택 → 저장된 분석 payload 기반 6축 radar + cross-issue 분석.

        Args:
            card_ids: 프론트에서 선택한 카드 id (2 ≤ N ≤ 20 권장).
            ratios: peer / industry / keyword 가중치 (frontend slider 결과).
            user_context: 사용자 자유 입력.

        Returns:
            MixerAnalysisOutput dict — design §5 schema.
        """
        if not card_ids or len(card_ids) < _MIN_CARDS:
            return _error_response(
                "card_ids 부족",
                f"mixer 는 최소 {_MIN_CARDS}개 카드 필요 (받음={len(card_ids or [])})",
                card_ids or [],
            )

        if len(card_ids) > _MAX_CARDS:
            log.warning(
                "Mixer | card_ids 너무 많음 — 상위 %d개로 truncate (받음=%d)",
                _MAX_CARDS,
                len(card_ids),
            )
            card_ids = card_ids[:_MAX_CARDS]

        cards = _fetch_cards(card_ids)
        if not cards:
            return _error_response(
                "카드 조회 실패",
                "DB 에서 card_news row 0건 — id 확인 필요",
                card_ids,
            )

        return await self._analyze_cards(
            cards=cards,
            requested_card_ids=card_ids,
            ratios=ratios,
            user_context=user_context,
        )

    async def analyze_items(
        self,
        items: list[dict],
        ratios: dict | None = None,
        user_context: str | None = None,
    ) -> dict:
        """로컬 목업/테스트용 linked result 묶음 → 믹스 인사이트 생성.

        실제 서비스에서는 ``analyze(card_ids=...)`` 가 DB 에서 카드와
        카드에 연결된 ``integrated_issue`` / ``analysis`` / ``implication`` /
        ``profile_context`` 를 조회한다. 이 메서드는 아직 카드 저장이 안정화되지 않은
        상황에서 동일한 믹서 로직을 목업 데이터로 검증하기 위한 진입점이다.
        """
        cards = _cards_from_linked_result_items(items)
        card_ids = [str(card.get("id")) for card in cards if card.get("id")]
        if len(cards) < _MIN_CARDS:
            return _error_response(
                "분석 단위 부족",
                f"mixer 는 최소 {_MIN_CARDS}개 linked result 필요 (받음={len(cards)})",
                card_ids,
            )
        if len(cards) > _MAX_CARDS:
            cards = cards[:_MAX_CARDS]
            card_ids = [str(card.get("id")) for card in cards if card.get("id")]

        return await self._analyze_cards(
            cards=cards,
            requested_card_ids=card_ids,
            ratios=ratios,
            user_context=user_context,
        )

    async def _analyze_cards(
        self,
        *,
        cards: list[dict],
        requested_card_ids: list[str],
        ratios: dict | None,
        user_context: str | None,
    ) -> dict:
        # 6축 radar 미리 계산 — LLM input 으로 anchor 제공 (v2)
        radar = _compute_radar(cards)
        prompt = (
            _MIXER_PROMPT.replace("{context}", _format_analysis_units(cards))
            .replace("{radar_text}", _format_radar(radar))
            .replace("{ratios_text}", _format_ratios(ratios))
            .replace("{user_context}", (user_context or "").strip() or "*없음*")
        )

        try:
            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="MixerAnalysisAgent",
                    phase="analyze",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
        except Exception as e:
            log.exception("MixerAnalysisAgent LLM 호출 실패 | error=%s", e)
            return _error_response("LLM 호출 실패", str(e), requested_card_ids)

        result = _parse_and_validate(content, cards, requested_card_ids)
        result = _repair_mixer_result_quality(
            result=result,
            cards=cards,
            requested_card_ids=requested_card_ids,
        )
        mix_implication = _generate_mix_level_implication(result=result, cards=cards)
        if mix_implication:
            result["mix_implication"] = mix_implication
            detail_actions = _recommended_actions_from_details(result)
            actions = _recommended_actions_from_implication(mix_implication, result=result)
            if not actions:
                actions = _recommended_actions_from_basis(result)
            if detail_actions:
                actions = _dedupe_keep_order([*detail_actions, *actions])[:3]
            if actions:
                result["recommended_actions"] = actions
                result["sk_ax_implication"] = clip_implication(" ".join(actions))
                result["warning"] = _warning_for(result)
        result["radar_axes"] = radar  # 이미 위에서 계산된 값 재사용
        result["mix_id"] = _new_mix_id()
        result.setdefault("provenance", {}).update(
            {
                "llm_model": _LLM_MODEL,
                "prompt_version": _PROMPT_VERSION,
                "source_card_ids": [c["id"] for c in cards],
                "ratios": ratios or {},
                "analysis_basis": "integrated_issue+analysis+implication+profile_context",
            }
        )
        return result


# ──────────────────────────────────────────────────────────────────────────
# Deterministic radar (no LLM) — design §6.1
# ──────────────────────────────────────────────────────────────────────────


def _avg(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def _score_for_event(cards: list[dict], event_types: set[str]) -> float:
    return _avg([_card_score(c) for c in cards if (c.get("event_type") or "") in event_types])


def _score_for_sector(cards: list[dict], sectors: set[str]) -> float:
    return _avg([_card_score(c) for c in cards if _card_sector(c) in sectors])


def _card_score(card: dict) -> float:
    impl = card.get("implication") or {}
    sector_meta = impl.get("sector_meta") if isinstance(impl.get("sector_meta"), dict) else {}
    exposure = impl.get("exposure_score") or sector_meta.get("exposure_score")
    if exposure is None:
        exposure = card.get("importance_score")
    try:
        return float(exposure or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _card_sector(card: dict) -> str:
    impl = card.get("implication") or {}
    sector_meta = impl.get("sector_meta") if isinstance(impl.get("sector_meta"), dict) else {}
    return (
        impl.get("sector")
        or sector_meta.get("sector")
        or card.get("primary_keyword_category")
        or card.get("sector")
        or "other"
    ).lower()


def _peer_diversity_score(cards: list[dict]) -> float:
    peers = {c.get("peer_id") for c in cards if c.get("peer_id")}
    # 4 국내 peer 기준, 다양성 정규화 (1 peer=0.25, 4 peer=1.0).
    return min(len(peers) / 4.0, 1.0)


def _compute_radar(cards: list[dict]) -> list[dict]:
    axes: dict[str, tuple[float, str]] = {
        "peer_strategic_shift": (
            _score_for_event(cards, {"ma", "new_biz"}),
            "전략 전환 성격 이벤트의 평균 exposure",
        ),
        "tech_investment": (
            _score_for_sector(cards, {"ax", "ai_tech", "infra"}),
            "기술·플랫폼·인프라 성격 섹터의 평균 exposure",
        ),
        "market_position": (
            _peer_diversity_score(cards),
            "선택 카드의 peer 다양성",
        ),
        "partnership_momentum": (
            _score_for_event(cards, {"partnership"}),
            "협력·제휴 성격 이벤트의 평균 exposure",
        ),
        "regulatory_risk": (
            _score_for_event(cards, {"regulation"}),
            "규제·정책 성격 이벤트의 평균 exposure",
        ),
        "talent_movement": (
            _score_for_event(cards, {"personnel"}),
            "조직·인력 변화 성격 이벤트의 평균 exposure",
        ),
    }
    result: list[dict] = []
    for axis in _RADAR_AXIS_ORDER:
        score, explanation = axes[axis]
        result.append({"axis": axis, "score": round(score, 3), "explanation": explanation})
    return result


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _new_mix_id() -> str:
    return f"mix-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _format_radar(radar: list[dict]) -> str:
    """6축 radar 결과를 LLM 입력 prompt 용 한국어 라인으로 포맷.

    v2: 결정적 산식 결과를 LLM 에 anchor 로 제공. LLM 이 reasoning 에 활용.
    """
    if not radar:
        return "*radar 점수 산출 불가*"
    lines: list[str] = []
    label_kr = {
        "peer_strategic_shift": "Peer 전략 전환",
        "tech_investment": "기술 투자",
        "market_position": "시장 포지션",
        "partnership_momentum": "파트너십",
        "regulatory_risk": "규제 리스크",
        "talent_movement": "인재 이동",
    }
    for axis in radar:
        score = float(axis.get("score") or 0.0)
        bar = "▰" * int(score * 10) + "▱" * (10 - int(score * 10))
        label = label_kr.get(axis["axis"], axis["axis"])
        explanation = axis.get("explanation", "")
        lines.append(f"- {label}: {bar} {score:.2f} ({explanation})")
    return "\n".join(lines)


def _format_ratios(ratios: dict | None) -> str:
    if not ratios:
        return "*비율 미지정 — 기본 균등*"
    parts: list[str] = []
    for key in ("peer", "industry", "keyword"):
        value = ratios.get(key)
        if value:
            parts.append(f"- **{key}**: {value}")
    return "\n".join(parts) if parts else "*비율 미지정 — 기본 균등*"


def _fetch_cards(card_ids: list[str]) -> list[dict]:
    placeholders = ",".join(f":id_{i}" for i in range(len(card_ids)))
    params = {f"id_{i}": cid for i, cid in enumerate(card_ids)}
    sql_v2 = (
        "SELECT id, company, COALESCE(peer_company_id, company) AS peer_id, "
        "primary_keyword_category, source_raw_article_ids, title, summary_lines, "
        "event_type, importance, importance_score, implication, sources, "
        "evidence_payload, validation_pass, validation_sc_score "
        f"FROM card_news WHERE id IN ({placeholders})"
    )
    sql_legacy = (
        "SELECT id, company, company AS peer_id, title, summary_lines, event_type, "
        "importance, importance_score, implication, sources, validation_pass, "
        f"validation_sc_score FROM card_news WHERE id IN ({placeholders})"
    )
    try:
        with SessionLocal() as db:
            try:
                rows = db.execute(text(sql_v2), params).mappings().all()
            except Exception as exc:  # noqa: BLE001
                if not _is_missing_v2_column(exc):
                    raise
                log.info("Mixer v2 컬럼 조회 실패 → legacy card_news 조회 사용")
                db.rollback()
                rows = db.execute(text(sql_legacy), params).mappings().all()
    except Exception as e:
        log.exception("Mixer DB query 실패 | %s", e)
        return []

    out: list[dict] = []
    for r in rows:
        item = dict(r)
        item["implication"] = _json_dict(item.get("implication"))
        item["sources"] = _json_list(item.get("sources"))
        item["evidence_payload"] = _json_dict(item.get("evidence_payload"))
        item["source_raw_article_ids"] = _int_list(item.get("source_raw_article_ids"))
        out.append(item)
    return out


def _cards_from_linked_result_items(items: list[dict]) -> list[dict]:
    """목업/런타임 linked result 입력을 Mixer 가 쓰는 card-like dict 로 정규화한다."""
    cards: list[dict] = []
    for index, item in enumerate(items or [], start=1):
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id") or item.get("id") or f"mock-card-{index}")
        linked_results = _linked_results_from_item(item)
        implication = item.get("implication")
        if not isinstance(implication, dict):
            implication = linked_results.get("implication") if linked_results else {}
        if not isinstance(implication, dict):
            implication = {}
        evidence_payload = _json_dict(item.get("evidence_payload"))
        evidence_payload.update({key: value for key, value in linked_results.items() if value})
        cards.append(
            {
                "id": card_id,
                "card_id": card_id,
                "company": item.get("company") or item.get("main_company") or "",
                "peer_id": (
                    item.get("peer_id") or item.get("company") or item.get("main_company") or ""
                ),
                "primary_keyword_category": item.get("primary_keyword_category"),
                "source_raw_article_ids": _int_list(item.get("source_raw_article_ids")),
                "title": item.get("title") or _linked_result_title(linked_results),
                "summary_lines": item.get("summary_lines") or item.get("fact_summary") or [],
                "event_type": item.get("event_type") or _linked_result_event_type(linked_results),
                "importance": item.get("importance") or item.get("importance_level") or "medium",
                "importance_score": item.get("importance_score") or 0.0,
                "implication": implication,
                "sources": _json_list(item.get("sources")),
                "evidence_payload": evidence_payload,
                "validation_pass": item.get("validation_pass", True),
                "validation_sc_score": item.get("validation_sc_score", 0.0),
            }
        )
    return cards


def _linked_results_from_item(item: dict) -> dict:
    evidence_payload = _json_dict(item.get("evidence_payload"))
    return _linked_results_from_sources(item, evidence_payload)


def _linked_results_from_sources(*sources: dict) -> dict:
    collected: dict[str, object] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        legacy_group = source.get(_LEGACY_RESULT_GROUP_KEY)
        if isinstance(legacy_group, dict):
            for key in _LINKED_RESULT_KEYS:
                if isinstance(legacy_group.get(key), dict) and key not in collected:
                    collected[key] = legacy_group[key]
        for key in _LINKED_RESULT_KEYS:
            if isinstance(source.get(key), dict):
                collected[key] = source[key]
    return collected


def _linked_result_title(linked_results: dict) -> str:
    for key in ("integrated_issue", "analysis"):
        value = linked_results.get(key)
        if isinstance(value, dict):
            title = value.get("title") or value.get("main_issue") or value.get("headline")
            if title:
                return str(title)
    return ""


def _linked_result_event_type(linked_results: dict) -> str:
    for section_name in ("classification", "integrated_issue"):
        section = linked_results.get(section_name)
        if isinstance(section, dict):
            event_type = section.get("event_type") or section.get("cluster_event_type")
            if event_type:
                return str(event_type)
    return ""


def _linked_results_from_card(card: dict) -> dict:
    evidence_payload = _json_dict(card.get("evidence_payload"))
    return _linked_results_from_sources(card, evidence_payload)


def _component(linked_results: dict, key: str) -> dict:
    value = linked_results.get(key)
    return value if isinstance(value, dict) else {}


def _format_analysis_units(cards: list[dict]) -> str:
    """선택된 카드에 연결된 통합/분석/시사점 결과를 LLM 입력으로 정리한다."""
    blocks: list[str] = []
    for c in cards:
        impl = c.get("implication") or {}
        sector = _card_sector(c)
        exposure_band = impl.get("exposure_band") or c.get("importance") or "low"
        summary_lines = c.get("summary_lines") or []
        if isinstance(summary_lines, str):
            try:
                summary_lines = json.loads(summary_lines)
            except json.JSONDecodeError:
                summary_lines = [summary_lines]
        summary = " / ".join(s for s in summary_lines if s)
        sector_meta = impl.get("sector_meta") if isinstance(impl.get("sector_meta"), dict) else {}
        evidence = _json_dict(c.get("evidence_payload"))
        linked_results = _linked_results_from_card(c)
        integrated_issue = _component(linked_results, "integrated_issue")
        analysis_result = _component(linked_results, "analysis")
        implication_result = _component(linked_results, "implication") or impl
        peer_impl = (
            implication_result.get("peer_implication")
            if isinstance(implication_result.get("peer_implication"), dict)
            else {}
        )
        skax_impl = (
            implication_result.get("skax_implication")
            if isinstance(implication_result.get("skax_implication"), dict)
            else {}
        )
        evidence_links = evidence.get("source_links") or c.get("sources") or []
        financial_refs = evidence.get("financial_refs") or []
        raw_ids = c.get("source_raw_article_ids") or []
        skax_payload = _compact_json(skax_impl) or _compact_json(implication_result)
        block = "\n".join(
            [
                f"[{c['id']}] {c.get('title', '')}",
                f"- Peer: {c.get('peer_id') or c.get('company') or ''}",
                f"- Sector: {sector}",
                f"- Event type: {c.get('event_type', '')}",
                f"- Exposure: {exposure_band} ({_card_score(c):.2f})",
                f"- Source raw article ids: {raw_ids}",
                f"- 표시 요약(보조): {summary}",
                f"- 통합 이슈: {_compact_json(_compact_integrated_issue(integrated_issue))}",
                f"- 전략 분석: {_compact_json(_compact_analysis_result(analysis_result))}",
                f"- Peer 분석: {_compact_json(peer_impl)}",
                f"- SK AX 시사점/대응: {skax_payload}",
                f"- Sector/signals: {_compact_json(sector_meta)}",
                f"- 재무/수치 근거: {_compact_json(financial_refs)}",
                f"- 출처: {_compact_json(evidence_links[:5])}",
            ]
        )
        blocks.append(block)
    return "\n".join(blocks)


def _format_cards(cards: list[dict]) -> str:
    """Backward-compatible alias for older tests/imports."""
    return _format_analysis_units(cards)


def _json_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
        return parsed if isinstance(parsed, list) else []
    return []


def _int_list(value: object) -> list[int]:
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[int] = []
    for item in values:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in result:
            result.append(number)
    return result


def _compact_json(value: object, *, limit: int = 900) -> str:
    if value in ({}, [], None, ""):
        return ""
    text_value = json.dumps(value, ensure_ascii=False, default=str)
    return text_value[:limit] + "..." if len(text_value) > limit else text_value


def _compact_integrated_issue(value: dict) -> dict:
    if not value:
        return {}
    return {
        "main_issue": value.get("main_issue") or value.get("headline"),
        "integrated_text": value.get("integrated_text"),
        "fact_summary": value.get("fact_summary"),
        "consolidated_facts": value.get("consolidated_facts", [])[:5],
        "key_numbers": value.get("key_numbers", [])[:5],
        "business_signals": value.get("business_signals", [])[:5],
        "main_company": value.get("main_company"),
        "event_type": value.get("cluster_event_type"),
    }


def _compact_analysis_result(value: dict) -> dict:
    if not value:
        return {}
    return {
        "analysis_summary": value.get("analysis_summary"),
        "strategic_moves": value.get("strategic_moves", [])[:5],
        "market_signals": value.get("market_signals", [])[:5],
        "competitive_meaning": value.get("competitive_meaning"),
        "risk_factors": value.get("risk_factors", [])[:5],
        "confidence": value.get("confidence"),
    }


def _is_missing_v2_column(exc: Exception) -> bool:
    message = str(exc).lower()
    return "undefinedcolumn" in message or "does not exist" in message


def _valid_card_refs(value: object, allowed_card_ids: set[str]) -> list[str]:
    refs = [str(item) for item in _json_list(value) if str(item) in allowed_card_ids]
    result: list[str] = []
    for ref in refs:
        if ref not in result:
            result.append(ref)
    return result


def _valid_connections(value: object, allowed_card_ids: set[str]) -> list[dict]:
    connections: list[dict] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_card_id") or "")
        target = str(item.get("target_card_id") or "")
        if source not in allowed_card_ids or target not in allowed_card_ids or source == target:
            continue
        connection = dict(item)
        connection["source_card_id"] = source
        connection["target_card_id"] = target
        try:
            weight = float(connection.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        connection["weight"] = max(0.0, min(weight, 1.0))
        connections.append(connection)
    return connections[:20]


def _valid_cross_card_findings(value: object, allowed_card_ids: set[str]) -> list[dict]:
    findings: list[dict] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        refs = _valid_card_refs(item.get("evidence_card_ids"), allowed_card_ids)
        if len(refs) < 2:
            continue
        finding = dict(item)
        finding["evidence_card_ids"] = refs
        finding["finding"] = clip_string(finding.get("finding", ""), 120)
        findings.append(finding)
    return findings[:5]


def _normalize_mix_block(value: object, allowed_card_ids: set[str]) -> dict:
    if not isinstance(value, dict):
        value = {}
    evidence = _normalize_mix_evidence(value.get("evidence"), allowed_card_ids)
    refs = _valid_card_refs(value.get("evidence_card_ids"), allowed_card_ids)
    for item in evidence:
        card_id = item.get("card_id")
        if card_id and card_id not in refs:
            refs.append(card_id)
    return {
        "finding": clip_string(value.get("finding", ""), 180),
        "rationale": clip_string(value.get("rationale", ""), 240),
        "evidence": evidence[:4],
        "evidence_card_ids": refs[:6],
    }


def _normalize_action_details(value: object, allowed_card_ids: set[str]) -> list[dict]:
    details: list[dict] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        evidence = _normalize_mix_evidence(item.get("evidence"), allowed_card_ids)
        refs = _valid_card_refs(item.get("evidence_card_ids"), allowed_card_ids)
        for evidence_item in evidence:
            card_id = evidence_item.get("card_id")
            if card_id and card_id not in refs:
                refs.append(card_id)
        action = clip_implication(item.get("action") or item.get("text") or "")
        if not action:
            continue
        details.append(
            {
                "action": action,
                "why": clip_string(item.get("why") or item.get("rationale") or "", 260),
                "use_case": clip_string(item.get("use_case") or "", 80),
                "evidence": evidence[:3],
                "evidence_card_ids": refs[:6],
            }
        )
    return details[:3]


def _recommended_actions_from_details(result: dict) -> list[str]:
    return _dedupe_keep_order(
        [
            clip_implication(item.get("action"))
            for item in _json_list(result.get("action_details"))
            if isinstance(item, dict) and item.get("action")
        ]
    )[:3]


def _needs_action_detail_fallback(details: list[dict]) -> bool:
    if len(details) < 2:
        return True
    return any(
        not item.get("why") or len(_json_list(item.get("evidence_card_ids"))) < 2
        for item in details
    )


def _fallback_action_details_from_result(result: dict) -> list[dict]:
    common = result.get("common_pattern") if isinstance(result.get("common_pattern"), dict) else {}
    comparison = (
        result.get("comparison_point") if isinstance(result.get("comparison_point"), dict) else {}
    )
    hidden = (
        result.get("hidden_conclusion") if isinstance(result.get("hidden_conclusion"), dict) else {}
    )
    common_evidence = _json_list(common.get("evidence"))
    comparison_evidence = _json_list(comparison.get("evidence"))
    hidden_evidence = _json_list(hidden.get("evidence"))
    common_refs = _json_list(common.get("evidence_card_ids"))
    comparison_refs = _json_list(comparison.get("evidence_card_ids"))
    hidden_refs = _json_list(hidden.get("evidence_card_ids"))

    return [
        {
            "action": clip_implication(
                "제안 첫 장에 다음 메시지를 배치한다: "
                f"{hidden.get('finding') or result.get('mix_insight') or '믹스 인사이트'}"
            ),
            "why": clip_string(
                hidden.get("rationale")
                or (
                    "여러 이슈를 함께 볼 때 단일 기능보다 "
                    "조합된 판단 기준이 더 중요하게 드러나기 때문이다."
                ),
                260,
            ),
            "use_case": "제안 전략",
            "evidence": hidden_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": _dedupe_keep_order(
                [str(item) for item in (hidden_refs or common_refs)]
            )[:6],
        },
        {
            "action": clip_implication(
                "고객 설명 자료에 다음 비교 축을 표로 추가한다: "
                f"{comparison.get('finding') or '이슈별 강조점 차이'}"
            ),
            "why": clip_string(
                comparison.get("rationale")
                or "같은 흐름 안에서도 이슈마다 앞세우는 적용 장면과 성과 기준이 다르기 때문이다.",
                260,
            ),
            "use_case": "레퍼런스 구성",
            "evidence": comparison_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": _dedupe_keep_order(
                [str(item) for item in (comparison_refs or common_refs)]
            )[:6],
        },
        {
            "action": clip_implication(
                "후속 모니터링 항목을 다음 반복 신호의 실제 적용 사례와 "
                f"성과 근거로 잡는다: {common.get('finding') or '반복되는 움직임'}"
            ),
            "why": clip_string(
                common.get("rationale")
                or (
                    "여러 이슈에서 반복되는 움직임은 "
                    "다음 카드 조합에서도 계속 확인할 필요가 있기 때문이다."
                ),
                260,
            ),
            "use_case": "후속 모니터링",
            "evidence": common_evidence[:3],
            "evidence_card_ids": _dedupe_keep_order([str(item) for item in common_refs])[:6],
        },
    ]


def _normalize_mix_evidence(value: object, allowed_card_ids: set[str]) -> list[dict]:
    evidence: list[dict] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id") or "")
        if card_id not in allowed_card_ids:
            continue
        text_value = clip_string(item.get("text") or item.get("basis") or "", 160)
        if not text_value:
            continue
        evidence.append({"card_id": card_id, "text": text_value})
    return evidence


def _findings_from_blocks(data: dict) -> list[dict]:
    mappings = (
        ("common_pattern", "convergent_strategy"),
        ("comparison_point", "divergent_strategy"),
        ("hidden_conclusion", "acceleration_signal"),
    )
    findings: list[dict] = []
    for key, pattern_type in mappings:
        block = data.get(key)
        if not isinstance(block, dict):
            continue
        finding = str(block.get("finding") or "").strip()
        refs = _json_list(block.get("evidence_card_ids"))
        if not finding or len(refs) < 2:
            continue
        findings.append(
            {
                "finding": finding,
                "evidence_card_ids": refs,
                "pattern_type": pattern_type,
            }
        )
    return findings


def _connections_from_blocks(data: dict) -> list[dict]:
    connections: list[dict] = []
    for block_key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = data.get(block_key)
        if not isinstance(block, dict):
            continue
        refs = [str(ref) for ref in _json_list(block.get("evidence_card_ids"))]
        if len(refs) < 2:
            continue
        connections.append(
            {
                "source_card_id": refs[0],
                "target_card_id": refs[1],
                "label": "similar" if block_key != "comparison_point" else "contrast",
                "weight": 0.8,
                "reason": clip_string(block.get("finding", ""), 60),
            }
        )
    return connections


def _valid_reasoning_trail_refs(value: object, allowed_card_ids: set[str]) -> list[dict]:
    trail: list[dict] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        updated["evidence_refs"] = _valid_card_refs(item.get("evidence_refs"), allowed_card_ids)
        trail.append(updated)
    return trail


def _valid_reasoning_step_refs(value: object, allowed_card_ids: set[str]) -> list[dict]:
    steps: list[dict] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        updated["inputs_used"] = _valid_card_refs(item.get("inputs_used"), allowed_card_ids)
        steps.append(updated)
    return steps


def _parse_and_validate(
    content: str,
    cards: list[dict],
    card_ids: list[str],
) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("Mixer JSON parse 실패 — content prefix=%s", content[:200])
        return _error_response(
            "JSON parse 실패",
            "LLM 응답이 JSON 이 아님",
            card_ids,
            confidence=0.0,
        )

    if not isinstance(data, dict):
        return _error_response("응답 형식 오류", "JSON object 가 아님", card_ids, confidence=0.0)

    allowed_card_ids = {str(card["id"]) for card in cards}

    data["mix_insight"] = clip_string(data.get("mix_insight") or data.get("insight", ""), 220)
    data["common_pattern"] = _normalize_mix_block(data.get("common_pattern"), allowed_card_ids)
    data["comparison_point"] = _normalize_mix_block(data.get("comparison_point"), allowed_card_ids)
    data["hidden_conclusion"] = _normalize_mix_block(
        data.get("hidden_conclusion"), allowed_card_ids
    )
    data["recommended_actions"] = [
        clip_implication(item) for item in _json_list(data.get("recommended_actions")) if item
    ][:3]
    data["action_details"] = _normalize_action_details(data.get("action_details"), allowed_card_ids)
    if _needs_action_detail_fallback(data["action_details"]):
        data["action_details"] = _fallback_action_details_from_result(data)
    if not data["recommended_actions"]:
        data["recommended_actions"] = _recommended_actions_from_details(data)
    data["confidence"] = confidence_in_range(data.get("confidence", 0.0))
    data["sources_used"] = _valid_card_refs(
        data.get("sources_used") or [c["id"] for c in cards],
        allowed_card_ids,
    )
    if not data["sources_used"]:
        data["sources_used"] = [c["id"] for c in cards]

    # Backward-compatible fields for the existing API/UI.
    data["insight"] = data["mix_insight"]
    data["final_one_liner"] = clip_final_one_liner(
        data["hidden_conclusion"].get("finding") or data["mix_insight"]
    )
    data["sk_ax_implication"] = clip_implication(" ".join(data["recommended_actions"]))
    data["bullet_signals"] = [
        finding
        for finding in (
            data["common_pattern"].get("finding"),
            data["comparison_point"].get("finding"),
            data["hidden_conclusion"].get("finding"),
        )
        if finding
    ]
    block_connections = _connections_from_blocks(data)
    data["connections"] = _valid_connections(
        [*block_connections, *_json_list(data.get("connections"))],
        allowed_card_ids,
    )
    block_findings = _findings_from_blocks(data)
    data["cross_card_findings"] = _valid_cross_card_findings(
        [*block_findings, *_json_list(data.get("cross_card_findings"))],
        allowed_card_ids,
    )
    data["reasoning_trail"] = _valid_reasoning_trail_refs(
        data.get("reasoning_trail", []), allowed_card_ids
    )
    data["reasoning_steps"] = _valid_reasoning_step_refs(
        data.get("reasoning_steps", []), allowed_card_ids
    )
    data["follow_up_questions"] = []
    data.setdefault("radar_axes", [])

    peer_set: list[str] = []
    seen: set[str] = set()
    for c in cards:
        pid = c.get("peer_id")
        if pid and pid not in seen:
            seen.add(pid)
            peer_set.append(pid)
    data["peer_ids"] = peer_set

    data["langfuse_trace_id"] = _get_langfuse_trace_id()
    data["warning"] = _warning_for(data)
    return data


def _repair_mixer_result_quality(
    *,
    result: dict,
    cards: list[dict],
    requested_card_ids: list[str],
) -> dict:
    prompt = _MIXER_REPAIR_PROMPT.replace("{context}", _format_analysis_units(cards)).replace(
        "{draft_json}",
        json.dumps(_mixer_result_for_repair(result), ensure_ascii=False, indent=2),
    )
    try:
        response = _get_llm().invoke(
            prompt,
            config=tracing_config(
                agent="MixerAnalysisAgent",
                phase="repair_quality",
                prompt_version=f"{_PROMPT_VERSION}-repair",
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        repaired = _parse_and_validate(content, cards, requested_card_ids)
        repaired["repair_actions"] = _dedupe_keep_order(
            [*_json_list(result.get("repair_actions")), "mixer_sentence_quality_repaired"]
        )
        return repaired
    except Exception as exc:  # noqa: BLE001
        log.warning("Mixer quality repair 실패, 초안 사용 | error=%s", exc)
        return result


def _mixer_result_for_repair(result: dict) -> dict:
    keep_keys = (
        "mix_insight",
        "common_pattern",
        "comparison_point",
        "hidden_conclusion",
        "recommended_action_basis",
        "action_details",
        "recommended_actions",
        "sources_used",
        "confidence",
    )
    return {key: result.get(key) for key in keep_keys if key in result}


def _generate_mix_level_implication(result: dict, cards: list[dict]) -> dict:
    """믹스 분석 결과를 기존 ImplicationAgent에 넘겨 믹스 단위 대응방향을 생성한다."""
    try:
        return ImplicationAgent().generate(
            input_bundle=_mix_input_bundle(cards),
            integrated_issue=_mix_integrated_issue(result, cards),
            analysis=_mix_analysis_result(result, cards),
            profile_context=_mix_profile_context(cards),
            classification=_mix_classification(cards),
        )
    except Exception as exc:  # noqa: BLE001 - ImplicationAgent fallback 이후 최종 방어.
        log.warning("Mixer mix-level ImplicationAgent 호출 실패 | error=%s", exc)
        return {}


def _mix_integrated_issue(result: dict, cards: list[dict]) -> dict:
    common = result.get("common_pattern") if isinstance(result.get("common_pattern"), dict) else {}
    comparison = (
        result.get("comparison_point") if isinstance(result.get("comparison_point"), dict) else {}
    )
    hidden = (
        result.get("hidden_conclusion") if isinstance(result.get("hidden_conclusion"), dict) else {}
    )
    action_basis = _json_list(result.get("recommended_action_basis"))
    return {
        "is_valid_summary": True,
        "summary_scope": "mixer_integrated_issue",
        "cluster_id": result.get("mix_id") or "mixer",
        "main_company": "multi_peer",
        "source_article_ids": _dedupe_ints(
            [
                article_id
                for card in cards
                for article_id in _int_list(card.get("source_raw_article_ids"))
            ]
        ),
        "main_issue": result.get("mix_insight") or hidden.get("finding") or "",
        "integrated_text": " ".join(
            [
                str(result.get("mix_insight") or ""),
                str(common.get("finding") or ""),
                str(common.get("rationale") or ""),
                str(comparison.get("finding") or ""),
                str(comparison.get("rationale") or ""),
                str(hidden.get("finding") or ""),
                str(hidden.get("rationale") or ""),
                *[str(item) for item in action_basis],
            ]
        ).strip(),
        "consolidated_facts": _block_evidence_facts(common, cards),
        "business_signals": _dedupe_keep_order(
            [
                *[str(item) for item in action_basis],
                str(common.get("finding") or ""),
                str(common.get("rationale") or ""),
                str(comparison.get("finding") or ""),
                str(comparison.get("rationale") or ""),
                str(hidden.get("finding") or ""),
                str(hidden.get("rationale") or ""),
            ]
        ),
        "representative_sources": _sources_from_cards(cards),
    }


def _mix_analysis_result(result: dict, cards: list[dict]) -> dict:
    del cards
    common = result.get("common_pattern") if isinstance(result.get("common_pattern"), dict) else {}
    comparison = (
        result.get("comparison_point") if isinstance(result.get("comparison_point"), dict) else {}
    )
    hidden = (
        result.get("hidden_conclusion") if isinstance(result.get("hidden_conclusion"), dict) else {}
    )
    action_basis = _json_list(result.get("recommended_action_basis"))
    return {
        "is_valid_analysis": True,
        "analysis_scope": "mixed_peer_and_industry",
        "analysis_summary": result.get("mix_insight") or hidden.get("finding") or "",
        "strategic_meaning": _dedupe_keep_order(
            [
                str(common.get("finding") or ""),
                str(common.get("rationale") or ""),
                str(comparison.get("finding") or ""),
                str(comparison.get("rationale") or ""),
                str(hidden.get("finding") or ""),
                str(hidden.get("rationale") or ""),
                *[str(item) for item in action_basis],
            ]
        ),
        "market_signal": common.get("finding") or "",
        "impact_level": "medium",
        "impact_reason": " ".join(
            [
                str(hidden.get("finding") or result.get("mix_insight") or ""),
                str(hidden.get("rationale") or ""),
                " ".join(str(item) for item in action_basis),
            ]
        ).strip(),
        "risk_or_opportunity": "opportunity",
        "confidence": result.get("confidence", 0.0),
        "reason": "MixerAgent가 여러 카드의 연결 결과를 종합해 생성한 mix-level 분석",
    }


def _mix_profile_context(cards: list[dict]) -> dict:
    peer_profiles: dict[str, dict] = {}
    sector_context: dict[str, dict] = {}
    skax_profile: dict = {}
    for card in cards:
        linked_results = _linked_results_from_card(card)
        profile = linked_results.get("profile_context")
        if isinstance(profile, dict):
            if isinstance(profile.get("skax_profile"), dict) and not skax_profile:
                skax_profile = profile["skax_profile"]
            if isinstance(profile.get("peer_profiles"), dict):
                peer_profiles.update(profile["peer_profiles"])
            if isinstance(profile.get("sector_context"), dict):
                sector_context.update(profile["sector_context"])
        peer_id = str(card.get("peer_id") or card.get("company") or "")
        if peer_id and peer_id not in peer_profiles:
            peer_profiles[peer_id] = {"company_id": peer_id}
        sector = _card_sector(card)
        if sector and sector not in sector_context:
            sector_context[sector] = {"sector": sector}
    return {
        "skax_profile": skax_profile,
        "peer_profiles": peer_profiles,
        "sector_context": sector_context,
    }


def _mix_classification(cards: list[dict]) -> dict:
    sectors = _dedupe_keep_order([_card_sector(card) for card in cards if _card_sector(card)])
    event_types = _dedupe_keep_order(
        [str(card.get("event_type") or "") for card in cards if card.get("event_type")]
    )
    return {
        "sector": sectors[0] if sectors else "other",
        "sectors": sectors,
        "event_type": "mixed_issues",
        "source_event_types": event_types,
        "importance": "medium",
        "importance_score": _avg([_card_score(card) for card in cards]),
        "signals": {
            "source_card_count": len(cards),
            "peer_ids": _dedupe_keep_order(
                [str(card.get("peer_id") or card.get("company") or "") for card in cards]
            ),
        },
    }


def _mix_input_bundle(cards: list[dict]) -> dict:
    return {
        "bundle_id": f"mixer:{','.join(str(card.get('id')) for card in cards)}",
        "source_type": "mixer",
        "companies": _dedupe_keep_order(
            [str(card.get("peer_id") or card.get("company") or "") for card in cards]
        ),
        "sectors": _dedupe_keep_order([_card_sector(card) for card in cards if _card_sector(card)]),
        "items": cards,
        "sources": _sources_from_cards(cards),
        "metadata": {"source_card_ids": [card.get("id") for card in cards]},
    }


def _block_evidence_facts(block: dict, cards: list[dict]) -> list[dict]:
    card_ids = {str(card.get("id")) for card in cards}
    facts: list[dict] = []
    for item in _json_list(block.get("evidence")):
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id") or "")
        if card_id not in card_ids:
            continue
        text_value = str(item.get("text") or "").strip()
        if text_value:
            facts.append({"fact": text_value, "source_card_id": card_id})
    return facts


def _sources_from_cards(cards: list[dict]) -> list[dict]:
    sources: list[dict] = []
    for card in cards:
        for source in _json_list(card.get("sources")):
            if isinstance(source, dict):
                item = dict(source)
                item.setdefault("card_id", card.get("id"))
                sources.append(item)
        if not card.get("sources"):
            sources.append({"card_id": card.get("id"), "title": card.get("title") or ""})
    return sources[:20]


def _recommended_actions_from_implication(implication: dict, *, result: dict) -> list[str]:
    skax = implication.get("skax_implication")
    candidates: list[str] = []
    if isinstance(skax, dict):
        candidates.extend(_json_list(skax.get("recommended_actions")))
    candidates.extend(_json_list(implication.get("recommended_actions")))
    grounding_text = _grounding_text_for_actions(result)
    grounded = [
        clip_implication(item)
        for item in _dedupe_keep_order(candidates)
        if item and _is_action_grounded(str(item), grounding_text)
    ]
    return grounded[:3]


def _recommended_actions_from_basis(result: dict) -> list[str]:
    actions: list[str] = []
    for item in _json_list(result.get("recommended_action_basis")):
        text_value = str(item or "").strip()
        if not text_value:
            continue
        if not text_value.endswith(("다.", "요.", ".")):
            text_value = f"{text_value}."
        actions.append(clip_implication(text_value))
    return _dedupe_keep_order(actions)[:3]


def _grounding_text_for_actions(result: dict) -> str:
    parts = [
        str(result.get("mix_insight") or ""),
        " ".join(str(item) for item in _json_list(result.get("recommended_action_basis"))),
    ]
    for key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = result.get(key)
        if not isinstance(block, dict):
            continue
        parts.append(str(block.get("finding") or ""))
        parts.append(str(block.get("rationale") or ""))
        for evidence in _json_list(block.get("evidence")):
            if isinstance(evidence, dict):
                parts.append(str(evidence.get("text") or ""))
    return " ".join(parts)


def _is_action_grounded(action: str, grounding_text: str) -> bool:
    action_tokens = _distinctive_tokens(action)
    if not action_tokens:
        return False
    grounding_tokens = _distinctive_tokens(grounding_text)
    if not grounding_tokens:
        return False
    overlap = action_tokens & grounding_tokens
    return len(overlap) >= max(2, min(4, len(action_tokens) // 3))


def _distinctive_tokens(text_value: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", str(text_value or ""))
        if not token.isdigit()
    }
    if len(tokens) <= 2:
        return tokens
    frequency: dict[str, int] = {}
    for token in tokens:
        frequency[token] = str(text_value).lower().count(token)
    return {token for token in tokens if frequency[token] <= 3}


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _get_langfuse_trace_id() -> str | None:
    try:
        from src.observability.langfuse_client import get_langfuse_handler

        handler = get_langfuse_handler()
        if handler is None:
            return None
        return getattr(handler, "last_trace_id", None)
    except Exception:
        return None


def _warning_for(data: dict) -> str | None:
    warnings: list[str] = []
    confidence = float(data.get("confidence") or 0.0)
    if confidence < 0.6:
        warnings.append("근거 불충분 — 다른 카드 조합 권장 (confidence < 0.6)")
    for key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = data.get(key) if isinstance(data.get(key), dict) else {}
        if not block.get("finding") or len(_json_list(block.get("evidence_card_ids"))) < 2:
            warnings.append(f"{key} 근거 부족")
    if not data.get("recommended_actions"):
        warnings.append("recommended_actions 비어 있음")
    return "; ".join(warnings) if warnings else None


def _error_response(
    short_reason: str,
    detail: str,
    card_ids: list[str],
    confidence: float = 0.0,
) -> dict:
    log.warning("Mixer error | %s | detail=%s | ids=%s", short_reason, detail, card_ids)
    return {
        "mix_id": _new_mix_id(),
        "mix_insight": "",
        "common_pattern": {
            "finding": "",
            "rationale": "",
            "evidence": [],
            "evidence_card_ids": [],
        },
        "comparison_point": {
            "finding": "",
            "rationale": "",
            "evidence": [],
            "evidence_card_ids": [],
        },
        "hidden_conclusion": {
            "finding": "",
            "rationale": "",
            "evidence": [],
            "evidence_card_ids": [],
        },
        "action_details": [],
        "recommended_actions": [],
        "insight": "",
        "final_one_liner": "",
        "sk_ax_implication": "",
        "bullet_signals": [],
        "radar_axes": [],
        "connections": [],
        "reasoning_trail": [],
        "reasoning_steps": [],
        "follow_up_questions": [],
        "confidence": confidence,
        "sources_used": card_ids,
        "peer_ids": [],
        "langfuse_trace_id": None,
        "warning": f"{short_reason} — {detail}",
        "provenance": {
            "llm_model": _LLM_MODEL,
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": card_ids,
            "error": short_reason,
        },
    }


class MixerAgent(MixerAnalysisAgent):
    """Architecture-facing name for the 2단계 mixer agent."""
