"""StrategicInsightAgent — analysis + implication in one LLM call.

기존 ``StrategicAnalyzer`` 와 ``ImplicationAgent`` 를 하나의 LLM agent 로 통합하되,
저장/후속 처리 호환성을 위해 출력은 ``analysis`` 와 ``implication`` 두 블록으로
분리한다.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.agents.implication_agent import ImplicationAgent
from src.agents.strategic_analyzer import StrategicAnalyzer
from src.analysis.models import AnalysisContext, AnalysisInputBundle, ProfileContext
from src.db.postgres import SessionLocal
from src.services.analysis_context_builder import AnalysisContextBuilder
from src.services.profile_context_loader import ProfileContextLoader

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "strategic-insight-v1.18-grounded-claims"
_LLM_TEMPERATURE = 0.15
_LLM_MAX_COMPLETION_TOKENS = 2600

_IMPACT_LEVELS = {"high", "medium", "low"}
_RISK_OR_OPPORTUNITY = {"risk", "opportunity", "neutral"}
_EVIDENCE_LABELS = {"sufficient", "moderate", "insufficient"}
_UNSUPPORTED_CLAIM_PATTERNS = (
    r"시장\s*점유율\s*확대",
    r"점유율[이을가\s]*(확대|상승|증가)",
    r"시장\s*선점",
    r"선점",
    r"기술적\s*우위",
    r"혁신성",
    r"경쟁\s*심화",
    r"격차",
    r"리더십\s*확보",
    r"매출\s*기여",
    r"큰\s*영향",
    r"시장\s*확장\s*가능성",
    r"시장\s*확장",
    r"생태계\s*확장",
    r"차별화된\s*가치",
    r"기회를\s*창출",
    r"기술적\s*시너지",
    r"기술적\s*발전",
    r"수요[가를\s]*(증가|확대)",
    r"더\s*중시",
    r"입증",
    r"맞춤형\s*솔루션",
    r"시장\s*진입",
    r"가능성을\s*열어",
)
_BROAD_RESULT_PATTERNS = (
    r"입지[를\s]*강화",
    r"경쟁력[을\s]*(높|강화)",
    r"기술적\s*역량",
    r"기술적\s*협업",
)
_IMPACT_EVALUATION_PATTERN = re.compile(r"고객|평가|비교|판단|검증|기준|요구")
_IMPACT_ARTIFACT_PATTERN = re.compile(
    r"제안서|PoC|레퍼런스|운영\s*모델|운영\s*책임|책임\s*범위|검증\s*항목"
)
_ACTION_SENTENCE_END_PATTERN = re.compile(r"(다|니다|세요|십시오|함|음)\.?$")
_RELATIONSHIP_PATTERN = re.compile(r"협업|협력|파트너십|제휴|공동\s*(개발|사업|운영|추진)")
_UNCERTAINTY_PATTERN = re.compile(r"검토|가능성|구상|계획|예정|추진|모색|방향")
_EVENT_ANCHOR_PATTERN = re.compile(
    r"수주|선정|투자|지분|취득|계약|PoC|검증|테스트베드|구축|운영|도입|출시|매입|인수|판매|권한|검토"
)
_WEAK_MARKET_SIGNAL_PATTERN = re.compile(
    r"고객\s*요구[가-힣\s]*(변화|증가|확대|중요)|"
    r"(중요|부각)해지고\s*있습니다\.?$"
)
_INTERNAL_OR_GENERIC_PATTERN = re.compile(
    r"입력\s*근거에서\s*확인된\s*(역량\s*맥락|적용\s*범위|변화|신호)|"
    r"관련\s*(시장|분야|영역|적용\s*범위)|"
    r"해당\s*(시장|분야|영역)"
)


SYSTEM_PROMPT = """\
당신은 피어사 전략 분석과 SK AX 시사점을 동시에 작성하는 전략 인사이트 에이전트입니다.

역할:
- IntegratedIssue 로 피어사/시장 관점의 전략적 의미를 분석합니다.
- ProfileContext 와 AnalysisContext 로 SK AX 관점의 시사점과 대응 방향을 작성합니다.
- analysis 와 implication 의 관점을 섞지 않습니다.

핵심 원칙:
1. 현재 사건의 사실 근거는 IntegratedIssue 에서만 가져옵니다.
2. ProfileContext 는 기존 사업영역/역량 배경입니다. 통합 이슈와 직접 연결되지 않는
   프로필 문구를 현재 사건처럼 쓰지 마세요.
3. AnalysisContext 는 흐름 보조 맥락입니다. 새 사실이나 확정 성과의 근거로 쓰지 마세요.
4. 수치, 날짜, 고객명, 제품명, 회사명은 IntegratedIssue 의 fact_basis, key_numbers,
   representative_sources, consolidated_facts 중 하나에 있어야 합니다.
5. 채택, 선정, PoC, 판매 권한 확보 수준의 근거를 시장 선점, 점유율 확대, 매출 기여,
   리더십 확보로 과대해석하지 마세요.
6. 투자, 지분 취득, 매입, 수주, 선정, PoC, 검토처럼 IntegratedIssue 에 적힌
   관계 수준을 그대로 보존합니다. 투자/지분 취득만 있는 사실을 협업/파트너십으로
   바꾸지 말고, 검토/구상/가능성 단계의 관계를 확정 실행처럼 쓰지 마세요.
7. "강화", "입지", "경쟁력" 같은 넓은 표현을 단독으로 쓰지 마세요.
   해당 표현이 필요하면 무엇이 어떻게 바뀌는지까지 구체적으로 씁니다.
8. 비교 근거가 없으면 "기술적 우위", "혁신성", "선점", "격차", "경쟁 심화"처럼
   우열이나 시장 판도를 단정하는 표현을 쓰지 마세요. 대신 IntegratedIssue 에 나온
   고객명, 사업명, 적용 영역, 운영 단계, 검증 대상 같은 구체 명사로 좁혀 씁니다.
9. 짧게 요약하는 것보다 독자가 "왜 이런 분석과 시사점이 나왔는지" 이해할 수 있게
   근거 사실, 해석 기준, 대응 산출물을 함께 씁니다.
10. 추상 표현을 쓰면 같은 문장 또는 바로 다음 문장에서 구체 앵커로 풀어 씁니다.
    구체 앵커는 IntegratedIssue 또는 ProfileContext 에 있는 고객명, 사업명, 제품/서비스명,
    적용 영역, 검증 단계, 관계 수준, 운영 책임, 프로필 역량 중 하나입니다.
11. JSON 외 텍스트를 출력하지 마세요.
"""


USER_PROMPT_TEMPLATE = """\
## 입력 1 — IntegrationAgent 결과: integrated_issue
{integrated_issue_json}

## 입력 2 — classification
{classification_json}

## 입력 3 — input_bundle metadata
{bundle_json}

## 입력 4 — ProfileContext
{profile_json}

## 입력 5 — AnalysisContext
{context_json}

## 입력 6 — SK AX business_line_mapping 후보
{business_lines_json}

## 작성 기준
당신의 작업 방식:
1. 먼저 IntegratedIssue 에서 확인된 핵심 사실을 2~4개 고릅니다.
   이때 고객명, 사업명, 적용 영역, 검증 단계, 관계 유형(투자/수주/선정/검토 등),
   fact_id 를 함께 봅니다.
2. 각 사실이 피어사의 기존 ProfileContext 중 어떤 사업영역/역량과 연결되는지 확인합니다.
3. 그 연결이 고객 평가 기준, 제안서, PoC, 레퍼런스, 운영 모델 중 무엇을 바꾸는지 판단합니다.
4. 마지막으로 analysis 는 피어사/시장 의미만, implication 은 SK AX 대응만 남깁니다.

문장 품질 기준:
- 출력 schema 의 placeholder 는 길이 제한이 아닙니다. 각 string 필드는 필요하면
  2~3문장까지 써도 됩니다. 짧은 문장보다 근거와 판단 연결이 분명한 문장을 우선합니다.
- "역량을 입증", "입지 강화", "경쟁력 강화"처럼 결론만 말하지 마세요.
  무엇을 수주/선정/협업/검토했고, 그 결과 적용 범위·검증 단계·운영 책임 중
  무엇이 구체화되는지까지 씁니다.
- "중요합니다", "영향을 줄 수 있습니다", "재검토해야 합니다"로 끝내지 말고
  왜 그런 판단이 나왔는지 근거와 판단 기준을 함께 씁니다.
- 한 필드 안에서 같은 근거 문장을 반복하지 마세요. 이미 언급한 근거는 다음 문장에서
  "이 신호", "이 계약", "이 검증 단계"처럼 문맥형 표현으로 이어갑니다.
- "입력 근거", "관련 시장", "해당 분야" 같은 내부/범용 표현으로 설명하지 마세요.
  고객명, 사업명, 검증 단계, 운영 대상, 관계 수준 중 하나 이상을 문장에 넣습니다.
- "두 가지 사업", "여러 기술", "이러한 융합", "포괄적 솔루션", "기회", "위협"처럼
  독자가 다시 위 문장을 찾아봐야 이해되는 압축 표현만 쓰지 마세요. 압축 표현이 필요하면
  같은 문장 또는 다음 문장에서 실제 고객명, 사업명, 기술/역량명, 검증 단계, 운영 책임 중
  무엇을 뜻하는지 풀어 씁니다.

1. analysis 는 피어사/시장 분석만 작성합니다.
   - analysis_summary: 현재 사건과 피어사의 전략적 의미를 함께 담습니다. 단순 요약이 아니라
     "무슨 근거가 확인됐고, 그래서 피어사의 어떤 사업/고객/운영 범위가 달라지는지"를
     한 문장 또는 두 문장으로 씁니다.
   - strategic_meaning: 2~3개. 각 항목은 "근거 사실 → 피어사 포지셔닝/역량 변화" 구조로
     씁니다. 필요하면 한 항목을 2문장으로 써도 됩니다.
   - market_signal: classification 라벨을 그대로 쓰지 말고 자연어 시장 흐름으로 씁니다.
     "고객 요구가 변화하고 있습니다" 같은 일반론만 쓰지 말고, 통합 근거에서 보이는
     수주/선정/투자/검증/운영 단계가 어떤 수요나 평가 기준을 보여주는지 씁니다.
     한 사건만으로 시장 전체 증가를 단정하지 말고, 확인된 신호의 범위 안에서 씁니다.
     통합 근거에 "수요 증가/확대"가 직접 없으면 "수요가 증가"가 아니라
     "수요 신호가 확인/구체화"되는 정도로 씁니다.
   - impact_reason: impact_level 을 왜 그렇게 판단했는지 씁니다. 고객/사업 범위,
     검증 단계, 운영 책임, 피어 프로필의 기존 역량과 연결되는 정도 중 무엇이 영향도를
     높이거나 낮췄는지 설명합니다.
   - reason: impact_reason 을 반복하지 마세요. 사용한 fact_id, business_signal,
     profile_context 항목이 무엇인지와 그 근거가 어떤 해석에 쓰였는지를 설명합니다.
   - "역량 강화", "입지 강화", "경쟁력 강화"처럼 넓은 결론을 쓰면,
     판매 접점, 적용 레퍼런스, 운영 지원 범위, 검증 기준, 제안 메시지 중
     무엇이 바뀌는지까지 함께 씁니다.
   - 신뢰성, 경쟁력, 수요, 기회, 위협 같은 추상 표현은 결론이 아니라 해석의 결과입니다.
     해당 표현을 쓰면 어떤 고객/사업/검증 단계/운영 구간을 보고 그렇게 판단했는지
     바로 이어서 설명합니다.

2. implication 은 SK AX 대응 관점만 작성합니다.
   - peer_implication.peer_meaning: 짧은 요약이 아니라 2문장으로 씁니다.
     문장부호로 분리하세요. 첫 문장은 IntegratedIssue 의 핵심 근거 사실을 명시하고,
     둘째 문장은 ProfileContext 의 기존 사업영역/역량과 연결해 피어사의 전략적 의미를
     설명합니다.
   - peer_implication.capability_change: "기술적 우위", "입지 강화" 같은 결론만 쓰지 말고,
     기존 역량이 현재 사건을 통해 어떤 적용 범위, 고객군, 운영 방식, 제안 메시지로
     넓어지는지 씁니다. 가능하면 "기존 역량/배경 → 이번 사건으로 구체화된 적용 장면"의
     흐름으로 1~2문장을 씁니다.
     투자/지분 취득은 투자/지분 취득으로, 수주는 수주로, 검토/구상은 검토/구상으로
     유지합니다. 근거에 없는 협업/파트너십으로 바꾸지 않습니다.
   - why_important: 피어사 신호가 SK AX의 제안 기준, 레퍼런스 구성, 운영 책임,
     사업영역 판단 중 무엇에 영향을 주는지 씁니다. SK AX ProfileContext 의 어떤
     사업영역/역량 후보와 맞닿는지도 함께 설명합니다.
   - potential_impact: SK AX의 매출/수주/점유율 예측이 아니라 고객 평가 기준,
     제안 경쟁 방식, 운영 책임 설명 방식의 변화로 씁니다.
     반드시 2문장으로 작성합니다. 첫 문장은 IntegratedIssue 근거 신호 때문에 고객이
     무엇을 더 보게 되는지 설명하고, 둘째 문장은 그래서 SK AX의 제안서/PoC/레퍼런스/
     운영 모델 중 무엇이 어떻게 달라져야 하는지 설명합니다. "재검토해야 합니다"처럼
     이유 없는 결론으로 끝내지 마세요.
     "기술 융합", "혁신성"처럼 넓은 말로 끝내지 말고, 입력에 있는 사업명/고객군/
     운영 구간/검증 대상을 사용해 고객이 비교할 판단 기준을 구체화합니다.
     통합 근거에 직접 비교/선호 변화가 없으면 "더 중시"라고 단정하지 말고
     "함께 비교/확인할 수 있습니다"처럼 고객의 평가 항목 변화로 씁니다.
   - recommended_actions: "강화", "전략 수립" 같은 일반론 대신 제안서, PoC,
     레퍼런스, 운영 모델, 보안/데이터 거버넌스, 성과 검증 방식 중 실제로 바꿀
     산출물이나 행동을 씁니다.
   - recommended_actions: 각각 완성된 실행 문장으로 씁니다. 항목마다 실행 장면과
     바꿀 판단 기준이 달라야 하며, 같은 꼬리 문장이나 같은 template 을 반복하지
     마세요.
     좋은 실행 권고는 "어느 산출물에서 / 어떤 근거를 / 어떤 판단 기준으로 / 어떻게
     보여줄지"가 드러납니다.
   - recommended_actions 에서는 "기술적 융합/혁신성 강조"처럼 추상 메시지를 쓰지
     않습니다. 입력 근거에서 확인되는 사업/운영 맥락을 기준으로 제안서, PoC,
     레퍼런스, 운영 모델에서 무엇을 분리·설명·검증할지 씁니다.
     단일 action 이 길어져도 괜찮습니다. 짧은 슬로건보다 실제 실행 장면과 판단 기준이
     보이는 문장을 우선합니다.
   - opportunities/threats/recommended_actions 는 "무엇을 해야 하는가"보다
     "어떤 산출물이나 판단 기준을 어떻게 바꿀 것인가"가 드러나야 합니다.

3. business_line_mapping 은 입력 6의 후보 name 중에서만 0~3개 선택합니다.
   후보 설명이 IntegratedIssue 의 sectors, business_signals, fact_summary 와 직접 맞닿을 때만
   선택하고, 관련성이 약하면 [] 로 둡니다.

4. missing_or_uncertain_points 는 확정 사실로 쓰지 말고 follow_up_questions 또는
   watch_points 로 보냅니다. sourced_evidence_ids 와 used_fact_ids 는 입력에 존재하는
   fact_id 만 사용합니다. 출력 문장에서 사용한 핵심 사실의 fact_id 는 가능한 한
   sourced_evidence_ids 에 포함합니다.

5. 출력 schema 의 문구는 형태 안내입니다. 예시 문구를 복사하지 마세요.

## 출력 schema
{{
  "is_valid_strategic_insight": true,
  "analysis": {{
    "is_valid_analysis": true,
    "analysis_scope": "peer_and_industry",
    "analysis_summary": "피어사의 전략적 의미 1문장",
    "strategic_meaning": ["의미 1", "의미 2", "의미 3"],
    "market_signal": "시장/산업 흐름 1문장",
    "impact_level": "high|medium|low",
    "impact_reason": "영향도 판단 근거",
    "risk_or_opportunity": "risk|opportunity|neutral",
    "confidence": 0.8,
    "reason": "분석 근거"
  }},
  "implication": {{
    "is_valid_implication": true,
    "implication_scope": "peer_and_skax",
    "peer_implication": {{
      "company_id": "integrated_issue.main_company",
      "company_name_ko": "피어사명",
      "peer_meaning": "피어사 관점 의미",
      "capability_change": "역량 변화",
      "sourced_evidence_ids": ["입력에 존재하는 fact_id"]
    }},
    "skax_implication": {{
      "why_important": "SK AX에 중요한 이유",
      "potential_impact": "예상 영향",
      "opportunities": ["기회 1", "기회 2"],
      "threats": ["위협 1"],
      "recommended_actions": ["실행 권고 1", "실행 권고 2"],
      "business_line_mapping": ["후보 중 실제 관련 있는 사업라인명"]
    }},
    "follow_up_questions": ["추가 확인 질문 1", "추가 확인 질문 2", "추가 확인 질문 3"],
    "watch_points": ["관찰 포인트 1", "관찰 포인트 2"],
    "confidence": 0.75,
    "evidence_label": "moderate",
    "provenance": {{
      "generator": "StrategicInsightAgent",
      "prompt_version": "strategic-insight-v1.18-grounded-claims",
      "model": "gpt-4o",
      "used_fact_ids": ["입력에 존재하는 fact_id"],
      "used_context_layers": ["실제로 사용한 context layer명"],
      "run_at": "ISO-8601 timestamp"
    }}
  }}
}}
"""


REVIEW_SYSTEM_PROMPT = """\
당신은 StrategicInsightAgent 결과를 근거성, 구체성, 논리성 기준으로 점검하는
전략 QA reviewer 입니다. 입력 근거에 없는 사실을 추가하지 말고, 문제가 있는
필드만 같은 schema 안에서 더 정확하게 고칩니다. JSON 외 텍스트를 출력하지 마세요.
"""


REVIEW_USER_PROMPT_TEMPLATE = """\
## 입력 1 — IntegratedIssue
{integrated_issue_json}

## 입력 2 — classification
{classification_json}

## 입력 3 — ProfileContext
{profile_json}

## 입력 4 — AnalysisContext
{context_json}

## 입력 5 — business_line_mapping 후보
{business_lines_json}

## 입력 6 — 1차 StrategicInsightAgent 결과
{result_json}

## 검토 기준
0. 결과가 지나치게 일반론으로 바뀌면 실패입니다. 원래 결과의 구체 사실을 보존하면서
   과대표현과 논리 공백만 수정하세요.
1. 모든 수치, 날짜, 고객명, 제품명, 회사명, 사업명은 IntegratedIssue 또는
   ProfileContext 에 있는 표현만 사용해야 합니다.
2. IntegratedIssue 가 단순 수주, 선정, PoC, 가능성 검토 수준이면 시장 선점,
   점유율 확대, 매출 기여, 기술적 우위, 혁신성, 경쟁 심화처럼 우열이나 성과를
   단정하지 않습니다.
3. IntegratedIssue 의 관계 수준을 보존합니다. 투자/지분 취득을 협업/파트너십으로
   바꾸지 말고, 검토/구상/가능성 관계는 확정 실행처럼 쓰지 않습니다.
4. market_signal 이 "고객 요구 변화", "중요해지고 있습니다" 같은 일반론이면
   고객명, 사업명, 검증 단계, 운영 대상, 관계 유형을 넣어 다시 작성합니다.
   통합 근거에 수요 증가/확대가 직접 없으면 수요 증가를 단정하지 말고 확인된
   수요 신호나 평가 항목으로 낮춰 씁니다.
5. analysis 는 피어사와 시장 의미만 다룹니다. SK AX 대응 문장은 implication 에만 둡니다.
6. peer_meaning 과 capability_change 는 짧은 결론으로 끝내지 말고,
   "근거 사실 → 피어사의 적용 범위/고객군/운영 방식/제안 메시지 변화"가 보이게 씁니다.
   peer_meaning 은 문장부호로 분리된 2문장 구조를 지킵니다.
7. potential_impact 는 2문장으로 씁니다.
   - 1문장: 입력 근거 때문에 고객이 무엇을 더 비교하거나 평가하게 되는지
   - 2문장: 그래서 SK AX의 제안서, PoC, 레퍼런스, 운영 모델 중 무엇이 어떻게 바뀌어야 하는지
8. impact_reason 과 reason 이 같은 문장이면 실패입니다. impact_reason 은 영향도 판단,
   reason 은 사용한 fact/profile/context 근거와 해석 연결을 설명합니다.
9. opportunities, threats, recommended_actions 는 서로 다른 판단 포인트를 다룹니다.
   같은 꼬리 문장이나 같은 template 을 반복하지 않습니다.
10. recommended_actions 는 "강조/강화/재검토" 같은 추상 동사만으로 끝내지 말고,
   산출물 또는 실행 장면에서 무엇을 설명, 분리, 검증, 비교할지까지 씁니다.
11. 추상 표현이 구체 앵커 없이 남아 있으면 수정합니다. 구체 앵커는 고객명, 사업명,
   제품/서비스명, 적용 영역, 검증 단계, 관계 수준, 운영 책임, 프로필 역량입니다.
12. "두 가지 사업", "여러 기술", "이러한 융합"처럼 앞 문장을 다시 봐야 하는 압축 표현은
   실제 명칭이나 판단 기준으로 풀어 씁니다.
13. 통합 근거에 직접 없는 수요 증가/확대와 선호 비교 표현은 보수적으로 낮춥니다.
   "수요가 증가"는 "수요 신호가 확인", "더 중시"는 "함께 비교/확인" 수준으로 씁니다.
14. business_line_mapping 은 입력 후보 name 중에서만 고릅니다.
15. sourced_evidence_ids 와 used_fact_ids 는 입력에 존재하는 fact_id 만 사용합니다.

## 출력 형식
수정이 필요 없으면 revised_result 는 1차 결과와 동일하게 둡니다.
{{
  "needs_revision": true,
  "violations": ["수정 이유 1", "수정 이유 2"],
  "revised_result": {{
    "is_valid_strategic_insight": true,
    "analysis": {{
      "is_valid_analysis": true,
      "analysis_scope": "peer_and_industry",
      "analysis_summary": "string",
      "strategic_meaning": ["string"],
      "market_signal": "string",
      "impact_level": "high|medium|low",
      "impact_reason": "string",
      "risk_or_opportunity": "risk|opportunity|neutral",
      "confidence": 0.8,
      "reason": "string"
    }},
    "implication": {{
      "is_valid_implication": true,
      "implication_scope": "peer_and_skax",
      "peer_implication": {{
        "company_id": "string",
        "company_name_ko": "string",
        "peer_meaning": "string",
        "capability_change": "string",
        "sourced_evidence_ids": ["입력에 존재하는 fact_id"]
      }},
      "skax_implication": {{
        "why_important": "string",
        "potential_impact": "string",
        "opportunities": ["string"],
        "threats": ["string"],
        "recommended_actions": ["string"],
        "business_line_mapping": ["후보 중 실제 관련 있는 사업라인명"]
      }},
      "follow_up_questions": ["string"],
      "watch_points": ["string"],
      "confidence": 0.75,
      "evidence_label": "moderate",
      "provenance": {{
        "generator": "StrategicInsightAgent",
        "prompt_version": "strategic-insight-v1.18-grounded-claims",
        "model": "gpt-4o",
        "used_fact_ids": ["입력에 존재하는 fact_id"],
        "used_context_layers": ["실제로 사용한 context layer명"],
        "run_at": "ISO-8601 timestamp"
      }}
    }}
  }}
}}
"""


REPAIR_SYSTEM_PROMPT = """\
당신은 StrategicInsightAgent 결과에서 검증 실패가 난 필드만 보수적으로 고치는
전략 인사이트 repair agent 입니다. 새 사실을 만들지 말고, 제공된 위반 사유를
해결하는 데 필요한 최소 수정만 합니다. JSON 외 텍스트를 출력하지 마세요.
"""


REPAIR_USER_PROMPT_TEMPLATE = """\
## 입력 1 — IntegratedIssue
{integrated_issue_json}

## 입력 2 — ProfileContext
{profile_json}

## 입력 3 — business_line_mapping 후보
{business_lines_json}

## 입력 4 — 검증 실패 사유
{violations_json}

## 입력 5 — 수정 대상 결과
{result_json}

## repair 원칙
1. 위반 사유가 있는 필드만 고칩니다. 위반이 없는 구체 문장과 fact_id 는 유지합니다.
2. 시장 점유율, 선점, 기술적 우위, 혁신성, 경쟁 심화, 매출 기여 같은 표현은
   IntegratedIssue 에 직접 근거가 없으면 제거합니다.
3. 투자/지분 취득, 수주, 선정, PoC, 검토 같은 관계 수준을 IntegratedIssue 와
   동일하게 맞춥니다. 근거에 없는 협업/파트너십으로 바꾸지 않고, 검토/구상은
   확정 실행으로 바꾸지 않습니다.
4. market_signal, reason, potential_impact 가 "고객 요구 변화", "입력 근거",
   "관련 시장" 같은 일반론이면 고객명, 사업명, 검증 단계, 운영 대상, 관계 유형을
   사용해 다시 씁니다.
5. "입지 강화", "경쟁력 강화"가 필요하면 고객명, 사업명, 적용 범위, 검증 단계,
   운영 책임 중 무엇이 바뀌는지 함께 씁니다.
6. potential_impact 는 2문장으로 씁니다. 첫 문장은 고객 평가 기준 변화,
   둘째 문장은 SK AX 제안서/PoC/레퍼런스/운영 모델 변화입니다.
7. impact_reason 과 reason 이 같으면 둘 중 하나를 다시 씁니다. impact_reason 은
   영향도 판단 기준을, reason 은 사용한 fact/profile/context 근거를 설명합니다.
8. peer_meaning 이 한 문장으로 뭉쳐 있으면 2문장으로 나눕니다. 첫 문장은 현재 사건,
   둘째 문장은 피어 프로필 역량과의 연결입니다.
9. 추상 표현은 구체 앵커로 풀어 씁니다. 고객명, 사업명, 제품/서비스명, 적용 영역,
   검증 단계, 관계 수준, 운영 책임, 프로필 역량 중 입력에 있는 표현을 사용합니다.
10. "두 가지 사업", "여러 기술", "이러한 융합"처럼 압축된 지시어는 실제 명칭이나
   판단 기준으로 풀어 씁니다.
11. 통합 근거에 직접 없는 수요 증가/확대와 선호 비교 표현은 보수적으로 낮춥니다.
   "수요가 증가"는 "수요 신호가 확인", "더 중시"는 "함께 비교/확인" 수준으로 씁니다.
12. recommended_actions 는 서로 다른 실행 장면을 다룹니다. 같은 꼬리 문장,
   "강조합니다" 반복, 추상 메시지 반복을 피합니다.
13. evidence id 와 business_line_mapping 은 입력 후보 안에서만 유지합니다.
14. 전체 결과를 "입력 근거", "관련 시장", "관련 적용 범위" 같은 일반 문장으로
   덮어쓰지 마세요. 기존 결과에 있던 고객명/사업명/검증 단계/협업 대상을 보존합니다.
15. 문장 전체 뒤에 "이 확인됩니다", "이 직접 근거입니다"를 붙이지 마세요.
   사실 문장은 자연스럽게 명사절로 바꾸거나, 해당 사실이 고객 평가 기준으로
   어떻게 연결되는지 설명합니다.

## 출력
StrategicInsightAgent 최종 schema 그대로 출력합니다.
"""


class StrategicInsightAgent:
    """Generate separated analysis/implication blocks from one LLM prompt."""

    prompt_version = _PROMPT_VERSION

    def __init__(
        self,
        *,
        llm: ChatOpenAI | None = None,
        analyzer: StrategicAnalyzer | None = None,
        implication_agent: ImplicationAgent | None = None,
        fallback_analyzer: StrategicAnalyzer | None = None,
        fallback_implication_agent: ImplicationAgent | None = None,
        enable_self_review: bool = True,
    ) -> None:
        self._llm = llm
        self._fallback_analyzer = fallback_analyzer or analyzer or StrategicAnalyzer()
        self._fallback_implication_agent = (
            fallback_implication_agent or implication_agent or ImplicationAgent()
        )
        self.enable_self_review = enable_self_review
        self.model = _LLM_MODEL

    def generate(
        self,
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any] | None = None,
        input_bundle: AnalysisInputBundle | dict[str, Any] | None = None,
        profile_context: ProfileContext | dict[str, Any] | None = None,
        analysis_context: AnalysisContext | dict[str, Any] | None = None,
        cluster_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return ``{"analysis": ..., "implication": ...}`` for downstream pipeline."""
        classification = classification or {}
        bundle_dict = _bundle_to_dict(input_bundle)
        profile_dict = _profile_to_dict(profile_context)
        context_dict = _analysis_context_to_dict(analysis_context)
        cluster_metadata = cluster_metadata or _cluster_metadata_from_bundle(bundle_dict)

        if not _is_valid_integrated_issue(integrated_issue):
            return _empty_strategic_insight(
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata=cluster_metadata,
                reason="유효한 통합 이슈가 없어 전략 인사이트를 생성하지 않았습니다.",
            )

        prompt = USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            classification_json=_json_dumps(_classification_for_prompt(classification)),
            bundle_json=_json_dumps(_bundle_for_prompt(bundle_dict, cluster_metadata)),
            profile_json=_json_dumps(_profile_for_prompt(profile_dict)),
            context_json=_json_dumps(_analysis_context_for_prompt(context_dict)),
            business_lines_json=_json_dumps(_business_line_candidate_details(profile_dict)),
        )

        try:
            content = self._invoke_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                bundle_id=str(
                    bundle_dict.get("bundle_id")
                    or integrated_issue.get("bundle_id")
                    or cluster_metadata.get("bundle_id")
                    or ""
                ),
                phase="generate",
            )
            result = _parse_and_normalize(
                content,
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata=cluster_metadata,
                profile_context=profile_dict,
                analysis_context=context_dict,
                model=self.model,
            )
            if not self.enable_self_review or not result.get("is_valid_strategic_insight"):
                return result
            return self._review_and_revise(
                result,
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_dict,
                analysis_context=context_dict,
                bundle_id=str(
                    bundle_dict.get("bundle_id")
                    or integrated_issue.get("bundle_id")
                    or cluster_metadata.get("bundle_id")
                    or ""
                ),
            )
        except Exception as exc:  # noqa: BLE001 - fallback preserves pipeline availability.
            log.warning(
                "StrategicInsightAgent LLM failure → legacy fallback | bundle=%s error=%s",
                bundle_dict.get("bundle_id") or integrated_issue.get("bundle_id"),
                exc,
            )
            return self._legacy_fallback(
                integrated_issue=integrated_issue,
                classification=classification,
                input_bundle=input_bundle,
                profile_context=profile_context,
                analysis_context=(
                    analysis_context if isinstance(analysis_context, AnalysisContext) else None
                ),
                cluster_metadata=cluster_metadata,
            )

    def generate_from_analysis_package(
        self,
        analysis_package: dict[str, Any],
        *,
        profile_context: ProfileContext | dict[str, Any] | None = None,
        analysis_context: AnalysisContext | dict[str, Any] | None = None,
        strict_profile: bool = False,
        require_skax_profile: bool = False,
    ) -> dict[str, Any]:
        """Generate strategic insight from a stored analysis_package JSON.

        This is the bridge for persisted pipeline output. The package usually
        comes from ``card_news.evidence_payload.analysis_package`` and contains
        the IntegrationAgent output plus classification/source metadata.
        """
        package = _json_dict(analysis_package)
        integrated_issue = _json_dict(package.get("integrated_issue") or package.get("summary"))
        classification = _json_dict(package.get("classification"))
        input_bundle = _input_bundle_from_analysis_package(
            package=package,
            integrated_issue=integrated_issue,
            classification=classification,
        )
        profile_context = profile_context or _load_profile_context_for_issue(
            integrated_issue=integrated_issue,
            classification=classification,
            strict=strict_profile,
            require_skax_profile=require_skax_profile,
        )
        analysis_context = analysis_context or _build_analysis_context_for_issue(
            input_bundle=input_bundle,
            profile_context=profile_context,
            integrated_issue=integrated_issue,
        )
        return self.generate(
            integrated_issue=integrated_issue,
            classification=classification,
            input_bundle=input_bundle,
            profile_context=profile_context,
            analysis_context=analysis_context,
            cluster_metadata=_cluster_metadata_from_bundle(input_bundle.to_dict()),
        )

    def generate_from_integrated_issue_id(
        self,
        integrated_issue_id: str,
        *,
        save: bool = False,
        strict_profile: bool = False,
        require_skax_profile: bool = False,
    ) -> dict[str, Any]:
        """Load ``integrated_issues`` by id and run the agent.

        This is the forward pipeline entry point. Card news is created after
        strategic insight generation, so callers that already have an
        IntegratedIssue should use this method instead of a card id.
        """
        if save:
            raise ValueError(
                "StrategicInsightAgent persistence is not enabled yet; "
                "run with save=False until the storage table is finalized."
            )
        record = _load_integrated_issue_analysis_package(integrated_issue_id)
        result = self.generate_from_analysis_package(
            record["analysis_package"],
            strict_profile=strict_profile,
            require_skax_profile=require_skax_profile,
        )
        return result

    def generate_from_card_news(
        self,
        card_news_id: str,
        *,
        save: bool = False,
        strict_profile: bool = False,
        require_skax_profile: bool = False,
    ) -> dict[str, Any]:
        """Load ``card_news`` by id and run the agent for existing-card debug flows."""
        if save:
            raise ValueError(
                "StrategicInsightAgent persistence is not enabled yet; "
                "run with save=False until the storage table is finalized."
            )
        record = _load_card_news_analysis_package(card_news_id)
        result = self.generate_from_analysis_package(
            record["analysis_package"],
            strict_profile=strict_profile,
            require_skax_profile=require_skax_profile,
        )
        return result

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=_LLM_MODEL,
                temperature=_LLM_TEMPERATURE,
                max_completion_tokens=_LLM_MAX_COMPLETION_TOKENS,
            )
        return self._llm

    def _invoke_llm(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        bundle_id: str,
        phase: str = "generate",
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            from src.observability import tracing_config

            config = tracing_config(
                agent="StrategicInsightAgent",
                phase=phase,
                prompt_version=_PROMPT_VERSION,
                bundle_id=bundle_id,
            )
        except Exception:
            config = None
        response = (
            self._get_llm().invoke(messages, config=config)
            if config
            else self._get_llm().invoke(messages)
        )
        return response.content if isinstance(response.content, str) else str(response.content)

    def _review_and_revise(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
    ) -> dict[str, Any]:
        prompt = REVIEW_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            classification_json=_json_dumps(_classification_for_prompt(classification)),
            profile_json=_json_dumps(_profile_for_prompt(profile_context)),
            context_json=_json_dumps(_analysis_context_for_prompt(analysis_context)),
            business_lines_json=_json_dumps(_business_line_candidate_details(profile_context)),
            result_json=_json_dumps(result),
        )
        try:
            content = self._invoke_llm(
                system_prompt=REVIEW_SYSTEM_PROMPT,
                user_prompt=prompt,
                bundle_id=bundle_id,
                phase="self_review",
            )
            revised = _parse_review_and_normalize(
                content,
                original=result,
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                analysis_context=analysis_context,
                model=self.model,
            )
            violations = _quality_gate_violations(
                revised,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            if not violations:
                return revised
            return self._repair_quality_violations(
                revised,
                violations=violations,
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                analysis_context=analysis_context,
                bundle_id=bundle_id,
            )
        except Exception as exc:  # noqa: BLE001 - review is quality layer, not availability gate.
            log.warning(
                "StrategicInsightAgent self-review skipped | bundle=%s error=%s",
                bundle_id,
                exc,
            )
            return result

    def _repair_quality_violations(
        self,
        result: dict[str, Any],
        *,
        violations: list[str],
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
    ) -> dict[str, Any]:
        prompt = REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(_profile_for_prompt(profile_context)),
            business_lines_json=_json_dumps(_business_line_candidate_details(profile_context)),
            violations_json=_json_dumps(violations),
            result_json=_json_dumps(result),
        )
        try:
            content = self._invoke_llm(
                system_prompt=REPAIR_SYSTEM_PROMPT,
                user_prompt=prompt,
                bundle_id=bundle_id,
                phase="quality_repair",
            )
            repaired = _parse_and_normalize(
                content,
                integrated_issue=integrated_issue,
                classification=classification,
                cluster_metadata={},
                profile_context=profile_context,
                analysis_context=analysis_context,
                model=self.model,
            )
            remaining = _quality_gate_violations(
                repaired,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            if not remaining:
                return repaired
            guarded = _fallback_quality_repair(
                repaired,
                violations=remaining,
                integrated_issue=integrated_issue,
            )
            final_remaining = _quality_gate_violations(
                guarded,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            return (
                guarded
                if not final_remaining
                else _mark_quality_gate_failed(guarded, final_remaining)
            )
        except Exception as exc:  # noqa: BLE001 - fail closed instead of passing risky copy.
            log.warning(
                "StrategicInsightAgent quality repair failed | bundle=%s error=%s",
                bundle_id,
                exc,
            )
            guarded = _fallback_quality_repair(
                result,
                violations=violations,
                integrated_issue=integrated_issue,
            )
            final_remaining = _quality_gate_violations(
                guarded,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            return (
                guarded
                if not final_remaining
                else _mark_quality_gate_failed(guarded, final_remaining)
            )

    def _legacy_fallback(
        self,
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        input_bundle: AnalysisInputBundle | dict[str, Any] | None,
        profile_context: ProfileContext | dict[str, Any] | None,
        analysis_context: AnalysisContext | None,
        cluster_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        analysis = self._fallback_analyzer.analyze(
            integrated_issue=integrated_issue,
            classification=classification,
            cluster_metadata=cluster_metadata,
        )
        implication = self._fallback_implication_agent.generate(
            input_bundle=input_bundle,
            integrated_issue=integrated_issue,
            analysis=analysis,
            profile_context=profile_context,
            analysis_context=analysis_context,
            classification=classification,
        )
        return {
            "is_valid_strategic_insight": bool(
                analysis.get("is_valid_analysis") and implication.get("is_valid_implication")
            ),
            "analysis": _normalize_analysis_block(analysis),
            "implication": _normalize_implication_block(
                implication,
                integrated_issue=integrated_issue,
                profile_context=_profile_to_dict(profile_context),
                analysis_context={},
                model=self.model,
            ),
        }


def _parse_and_normalize(
    raw: str,
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    cluster_metadata: dict[str, Any],
    profile_context: dict[str, Any],
    analysis_context: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    data = _parse_json_loose(raw)
    if not isinstance(data, dict) or not (
        isinstance(data.get("analysis"), dict) and isinstance(data.get("implication"), dict)
    ):
        raise ValueError("StrategicInsightAgent response missing analysis/implication JSON blocks")

    analysis = _normalize_analysis_block(data.get("analysis") or {})
    implication = _normalize_implication_block(
        data.get("implication") or {},
        integrated_issue=integrated_issue,
        profile_context=profile_context,
        analysis_context=analysis_context,
        model=model,
    )
    is_valid = bool(
        data.get("is_valid_strategic_insight", True)
        and analysis.get("is_valid_analysis")
        and implication.get("is_valid_implication")
    )
    return {
        "is_valid_strategic_insight": is_valid,
        "analysis": analysis,
        "implication": implication,
    }


def _parse_review_and_normalize(
    raw: str,
    *,
    original: dict[str, Any],
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    profile_context: dict[str, Any],
    analysis_context: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    data = _parse_json_loose(raw)
    if not isinstance(data, dict):
        return original

    revised = data.get("revised_result") or data.get("revised")
    if not isinstance(revised, dict):
        if isinstance(data.get("analysis"), dict) and isinstance(data.get("implication"), dict):
            revised = data
        else:
            return original

    normalized = _parse_and_normalize(
        _json_dumps(revised),
        integrated_issue=integrated_issue,
        classification=classification,
        cluster_metadata={},
        profile_context=profile_context,
        analysis_context=analysis_context,
        model=model,
    )
    return normalized if normalized.get("is_valid_strategic_insight") else original


def _load_card_news_analysis_package(card_news_id: str) -> dict[str, Any]:
    card_id = str(card_news_id or "").strip()
    if not card_id:
        raise ValueError("card_news_id is required")
    with SessionLocal() as db:
        row = (
            db.execute(
                text(
                    """
                SELECT id,
                       company,
                       peer_company_id,
                       primary_keyword_category,
                       event_type,
                       importance,
                       importance_score,
                       source_raw_article_ids,
                       primary_raw_article_id,
                       integrated_issue_id,
                       evidence_payload
                  FROM card_news
                 WHERE id = :card_id
                 LIMIT 1
                """
                ),
                {"card_id": card_id},
            )
            .mappings()
            .fetchone()
        )
    if row is None:
        raise ValueError(f"card_news row not found: {card_id}")

    payload = _json_dict(row.get("evidence_payload"))
    package = _json_dict(payload.get("analysis_package"))
    if package:
        return {
            "card_news_id": card_id,
            "integrated_issue_id": row.get("integrated_issue_id"),
            "analysis_package": package,
        }

    issue_id = row.get("integrated_issue_id")
    integrated_issue = _load_integrated_issue(issue_id) if issue_id else {}
    if not integrated_issue:
        raise ValueError(
            "card_news row has no evidence_payload.analysis_package and no loadable "
            f"integrated_issue_id: {card_id}"
        )
    classification = _classification_from_card_row(dict(row), integrated_issue=integrated_issue)
    package = {
        "bundle_id": integrated_issue.get("bundle_id") or f"card:{card_id}",
        "integrated_issue": integrated_issue,
        "classification": classification,
        "sources": integrated_issue.get("representative_sources") or [],
    }
    return {
        "card_news_id": card_id,
        "integrated_issue_id": issue_id,
        "analysis_package": package,
    }


def _load_integrated_issue_analysis_package(integrated_issue_id: str) -> dict[str, Any]:
    issue_id = str(integrated_issue_id or "").strip()
    if not issue_id:
        raise ValueError("integrated_issue_id is required")
    with SessionLocal() as db:
        row = (
            db.execute(
                text(
                    """
                SELECT id,
                       issue_key,
                       event_type,
                       main_company,
                       sectors,
                       representative_raw_article_id,
                       payload
                  FROM integrated_issues
                 WHERE id = :issue_id
                 LIMIT 1
                """
                ),
                {"issue_id": issue_id},
            )
            .mappings()
            .fetchone()
        )
    if row is None:
        raise ValueError(f"integrated_issues row not found: {issue_id}")

    row_dict = dict(row)
    payload = _json_dict(row_dict.get("payload"))
    integrated_issue = _load_integrated_issue(issue_id)
    if not integrated_issue:
        raise ValueError(f"integrated_issues row is not loadable: {issue_id}")
    classification = _json_dict(payload.get("classification")) or _classification_from_issue_row(
        row_dict,
        integrated_issue=integrated_issue,
    )
    package = {
        "bundle_id": integrated_issue.get("bundle_id") or row_dict.get("issue_key"),
        "integrated_issue": integrated_issue,
        "classification": classification,
        "sources": integrated_issue.get("representative_sources") or [],
    }
    return {"integrated_issue_id": issue_id, "analysis_package": package}


def _load_integrated_issue(issue_id: Any) -> dict[str, Any]:
    issue_id_text = str(issue_id or "").strip()
    if not issue_id_text:
        return {}
    with SessionLocal() as db:
        row = (
            db.execute(
                text(
                    """
                SELECT *
                  FROM integrated_issues
                 WHERE id = :issue_id
                 LIMIT 1
                """
                ),
                {"issue_id": issue_id_text},
            )
            .mappings()
            .fetchone()
        )
        if row is None:
            return {}
        sources = (
            db.execute(
                text(
                    """
                SELECT raw_article_id AS article_id,
                       title,
                       url,
                       source_name,
                       publisher,
                       published_at,
                       source_type
                  FROM integrated_issue_source_articles
                 WHERE integrated_issue_id = :issue_id
                 ORDER BY source_order, id
                """
                ),
                {"issue_id": issue_id_text},
            )
            .mappings()
            .all()
        )
        evidence_refs = (
            db.execute(
                text(
                    """
                SELECT evidence_ref_id,
                       evidence_text,
                       source_ids,
                       reference_payload
                  FROM integrated_issue_evidence_references
                 WHERE integrated_issue_id = :issue_id
                 ORDER BY id
                """
                ),
                {"issue_id": issue_id_text},
            )
            .mappings()
            .all()
        )
    row_dict = dict(row)
    payload = _json_dict(row_dict.get("payload"))
    if isinstance(payload.get("integrated_issue"), dict):
        issue = dict(payload["integrated_issue"])
    else:
        issue = dict(payload) if payload.get("is_valid_summary") else {}
    issue.update(
        {
            "bundle_id": issue.get("bundle_id") or row_dict.get("issue_key"),
            "cluster_id": issue.get("cluster_id") or row_dict.get("cluster_id"),
            "representative_id": issue.get("representative_id")
            or row_dict.get("representative_raw_article_id"),
            "source_article_ids": issue.get("source_article_ids")
            or _int_list(row_dict.get("source_ids")),
            "cluster_article_ids": issue.get("cluster_article_ids")
            or _int_list(row_dict.get("source_ids")),
            "analyzed_article_ids": issue.get("analyzed_article_ids")
            or _int_list(row_dict.get("analyzed_source_ids")),
            "main_company": issue.get("main_company") or row_dict.get("main_company"),
            "mentioned_peer_companies": issue.get("mentioned_peer_companies")
            or _jsonish_list(row_dict.get("mentioned_peer_companies")),
            "cluster_event_type": issue.get("cluster_event_type") or row_dict.get("event_type"),
            "headline": issue.get("headline") or row_dict.get("headline"),
            "main_issue": issue.get("main_issue") or row_dict.get("headline"),
            "one_line_summary": issue.get("one_line_summary") or row_dict.get("one_line_summary"),
            "integrated_text": issue.get("integrated_text")
            or row_dict.get("content_detailed_explanation")
            or row_dict.get("content_summary"),
            "fact_summary": issue.get("fact_summary") or _jsonish_list(row_dict.get("issue_brief")),
            "representative_sources": issue.get("representative_sources")
            or [dict(source) for source in sources],
            "fact_basis": issue.get("fact_basis") or _fact_basis_from_evidence_refs(evidence_refs),
            "consolidated_facts": issue.get("consolidated_facts")
            or _consolidated_facts_from_evidence_refs(evidence_refs),
            "confidence": issue.get("confidence") or row_dict.get("confidence") or 0.0,
            "is_valid_summary": issue.get("is_valid_summary", row_dict.get("is_valid", True)),
        }
    )
    return {key: value for key, value in issue.items() if value not in (None, "", [], {})}


def _classification_from_card_row(
    row: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    sector = (
        row.get("primary_keyword_category")
        or integrated_issue.get("sector")
        or (integrated_issue.get("sectors") or [""])[0]
    )
    return {
        "sector": sector,
        "sectors": [sector] if sector else [],
        "company": row.get("peer_company_id") or row.get("company"),
        "companies": [
            value for value in [row.get("peer_company_id") or row.get("company")] if value
        ],
        "event_type": row.get("event_type") or integrated_issue.get("cluster_event_type"),
        "importance": row.get("importance"),
        "importance_score": row.get("importance_score"),
        "representative_id": integrated_issue.get("representative_id")
        or row.get("primary_raw_article_id"),
    }


def _classification_from_issue_row(
    row: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    sectors = _string_list(integrated_issue.get("sectors") or row.get("sectors"), max_items=10)
    sector = sectors[0] if sectors else ""
    company = integrated_issue.get("main_company") or row.get("main_company")
    return {
        "sector": sector,
        "sectors": sectors,
        "company": company,
        "companies": [company] if company else [],
        "event_type": integrated_issue.get("cluster_event_type") or row.get("event_type"),
        "representative_id": integrated_issue.get("representative_id")
        or row.get("representative_raw_article_id"),
    }


def _input_bundle_from_analysis_package(
    *,
    package: dict[str, Any],
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> AnalysisInputBundle:
    source_ids = _int_list(
        integrated_issue.get("source_article_ids")
        or integrated_issue.get("cluster_article_ids")
        or integrated_issue.get("analyzed_article_ids")
    )
    sources = _jsonish_list(package.get("sources")) or _jsonish_list(
        integrated_issue.get("representative_sources")
    )
    companies = _companies_from_integrated_issue(integrated_issue)
    sectors = _string_list(classification.get("sectors"), max_items=10)
    sector = str(classification.get("sector") or "").strip()
    if sector and sector not in sectors:
        sectors.append(sector)
    return AnalysisInputBundle(
        bundle_id=str(
            package.get("bundle_id")
            or integrated_issue.get("bundle_id")
            or f"news:{integrated_issue.get('representative_id') or ''}"
        ),
        cluster_id=(
            str(integrated_issue.get("cluster_id"))
            if integrated_issue.get("cluster_id") is not None
            else None
        ),
        source_type=str(integrated_issue.get("issue_source_type") or "news"),
        companies=companies,
        sectors=sectors,
        event_type=str(
            classification.get("event_type") or integrated_issue.get("cluster_event_type") or ""
        )
        or None,
        items=[
            {
                "id": article_id,
                "raw_article_id": article_id,
                "is_representative": article_id == integrated_issue.get("representative_id"),
            }
            for article_id in source_ids
        ],
        facts=_jsonish_list(integrated_issue.get("consolidated_facts"))
        or _jsonish_list(integrated_issue.get("extracted_facts")),
        evidence_snippets=_evidence_snippets_from_issue(integrated_issue),
        sources=sources,
        metadata={
            "representative_id": integrated_issue.get("representative_id"),
            "cluster_article_ids": integrated_issue.get("cluster_article_ids") or source_ids,
            "classification": classification,
        },
    )


def _load_profile_context_for_issue(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    strict: bool,
    require_skax_profile: bool,
) -> dict[str, Any]:
    sectors = _string_list(classification.get("sectors"), max_items=10)
    sector = str(classification.get("sector") or "").strip()
    if sector and sector not in sectors:
        sectors.append(sector)
    return (
        ProfileContextLoader()
        .load(
            companies=_companies_from_integrated_issue(integrated_issue),
            sectors=sectors,
            event_type=str(
                classification.get("event_type") or integrated_issue.get("cluster_event_type") or ""
            )
            or None,
            strict=strict,
            require_skax_profile=require_skax_profile,
        )
        .to_dict()
    )


def _build_analysis_context_for_issue(
    *,
    input_bundle: AnalysisInputBundle,
    profile_context: ProfileContext | dict[str, Any],
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    try:
        return (
            AnalysisContextBuilder()
            .build(
                input_bundle=input_bundle,
                profile_context=profile_context,
                integrated_issue=integrated_issue,
            )
            .to_dict()
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("StrategicInsightAgent analysis_context load failed | error=%s", exc)
        return {}


def _companies_from_integrated_issue(integrated_issue: dict[str, Any]) -> list[str]:
    companies: list[str] = []
    for company_id in [
        integrated_issue.get("main_company"),
        *(_jsonish_list(integrated_issue.get("mentioned_peer_companies")) or []),
    ]:
        text = str(company_id or "").strip()
        if text and text not in companies:
            companies.append(text)
    return companies


def _evidence_snippets_from_issue(integrated_issue: dict[str, Any]) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    for item in _jsonish_list(integrated_issue.get("fact_basis")):
        if not isinstance(item, dict):
            continue
        evidence_texts = _jsonish_list(item.get("evidence_texts"))
        if not evidence_texts and item.get("evidence_text"):
            evidence_texts = [item.get("evidence_text")]
        for text_value in evidence_texts:
            text_str = str(text_value or "").strip()
            if text_str:
                snippets.append(
                    {
                        "text": text_str,
                        "fact_ids": _jsonish_list(item.get("fact_ids")),
                        "source_article_ids": _jsonish_list(item.get("source_article_ids")),
                    }
                )
    return snippets[:20]


def _fact_basis_from_evidence_refs(rows: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        evidence_ref_id = str(item.get("evidence_ref_id") or "").strip()
        evidence_text = str(item.get("evidence_text") or "").strip()
        if not evidence_ref_id or not evidence_text:
            continue
        result.append(
            {
                "summary_line_index": index,
                "source_article_ids": _int_list(item.get("source_ids")),
                "fact_ids": [evidence_ref_id],
                "evidence_text": evidence_text,
                "evidence_texts": [evidence_text],
                "evidence_type": "reported_fact",
            }
        )
    return result


def _consolidated_facts_from_evidence_refs(rows: list[Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        evidence_ref_id = str(item.get("evidence_ref_id") or "").strip()
        evidence_text = str(item.get("evidence_text") or "").strip()
        if not evidence_ref_id or not evidence_text:
            continue
        facts.append(
            {
                "fact_id": evidence_ref_id,
                "fact": evidence_text,
                "source_article_ids": _int_list(item.get("source_ids")),
                "evidence_texts": [evidence_text],
                "source_type": "news",
                "fact_type": "reported_fact",
            }
        )
    return facts


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _jsonish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def _int_list(value: Any) -> list[int]:
    result: list[int] = []
    for item in _jsonish_list(value):
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in result:
            result.append(parsed)
    return result


def _normalize_analysis_block(
    data: dict[str, Any],
) -> dict[str, Any]:
    strategic_meaning = _string_list(data.get("strategic_meaning"), max_items=3)
    return {
        "is_valid_analysis": bool(data.get("is_valid_analysis", True))
        and bool(data.get("analysis_summary") or strategic_meaning),
        "analysis_scope": "peer_and_industry",
        "analysis_summary": str(data.get("analysis_summary") or "").strip(),
        "strategic_meaning": strategic_meaning,
        "market_signal": str(data.get("market_signal") or "").strip(),
        "impact_level": _choice(data.get("impact_level"), _IMPACT_LEVELS, "medium"),
        "impact_reason": str(data.get("impact_reason") or "").strip(),
        "risk_or_opportunity": _choice(
            data.get("risk_or_opportunity"), _RISK_OR_OPPORTUNITY, "neutral"
        ),
        "confidence": _clamp_float(data.get("confidence"), 0.0),
        "reason": str(data.get("reason") or "").strip(),
    }


def _normalize_implication_block(
    data: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    analysis_context: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    peer_input = data.get("peer_implication") or {}
    skax_input = data.get("skax_implication") or {}
    known_fact_ids = _known_fact_ids(integrated_issue)
    sourced_evidence_ids = [
        fact_id
        for fact_id in _string_list(peer_input.get("sourced_evidence_ids"), max_items=10)
        if fact_id in known_fact_ids
    ]
    confidence = _clamp_float(data.get("confidence"), 0.0)
    recommended_actions = _normalize_recommended_actions(
        _string_list(skax_input.get("recommended_actions"), max_items=3)
    )
    skax = {
        "why_important": str(skax_input.get("why_important") or "").strip(),
        "potential_impact": str(skax_input.get("potential_impact") or "").strip(),
        "opportunities": _string_list(skax_input.get("opportunities"), max_items=3),
        "threats": _string_list(skax_input.get("threats"), max_items=3),
        "recommended_actions": recommended_actions,
        "business_line_mapping": [
            item
            for item in _string_list(skax_input.get("business_line_mapping"), max_items=3)
            if item in _business_line_candidates(profile_context)
        ],
    }
    peer = {
        "company_id": str(
            peer_input.get("company_id")
            or integrated_issue.get("main_company")
            or _first_peer_id(profile_context)
            or ""
        ),
        "company_name_ko": str(
            peer_input.get("company_name_ko") or _first_peer_name(profile_context) or ""
        ),
        "peer_meaning": str(peer_input.get("peer_meaning") or "").strip(),
        "capability_change": _optional_str(peer_input.get("capability_change")),
        "sourced_evidence_ids": sourced_evidence_ids,
    }
    sourced_evidence_ids = _augment_sourced_evidence_ids(
        sourced_evidence_ids,
        integrated_issue=integrated_issue,
        output_texts=[
            peer["peer_meaning"],
            peer["capability_change"],
            skax["why_important"],
            skax["potential_impact"],
            *skax["opportunities"],
            *skax["threats"],
            *skax["recommended_actions"],
        ],
    )
    peer["sourced_evidence_ids"] = sourced_evidence_ids
    used_layers = _normalize_used_context_layers(
        ((data.get("provenance") or {}) if isinstance(data.get("provenance"), dict) else {}).get(
            "used_context_layers"
        ),
        analysis_context=analysis_context,
    )
    is_valid = bool(
        data.get("is_valid_implication", True)
        and (peer["peer_meaning"] or skax["why_important"])
        and (skax["recommended_actions"] or skax["opportunities"] or skax["potential_impact"])
    )
    return {
        "is_valid_implication": is_valid,
        "implication_scope": "peer_and_skax",
        "peer_implication": peer,
        "skax_implication": skax,
        "follow_up_questions": _string_list(data.get("follow_up_questions"), max_items=3),
        "watch_points": _string_list(data.get("watch_points"), max_items=3),
        "confidence": confidence,
        "evidence_label": _evidence_label(data.get("evidence_label"), confidence),
        "provenance": {
            "generator": "StrategicInsightAgent",
            "prompt_version": _PROMPT_VERSION,
            "model": model,
            "used_fact_ids": sourced_evidence_ids,
            "used_context_layers": used_layers,
            "run_at": datetime.now(UTC).isoformat(),
        },
    }


def _empty_strategic_insight(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    cluster_metadata: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    analysis = {
        "is_valid_analysis": False,
        "analysis_scope": "peer_and_industry",
        "analysis_summary": "",
        "strategic_meaning": [],
        "market_signal": "",
        "impact_level": "low",
        "impact_reason": "",
        "risk_or_opportunity": "neutral",
        "confidence": 0.0,
        "reason": reason,
    }
    implication = {
        "is_valid_implication": False,
        "implication_scope": "peer_and_skax",
        "peer_implication": {
            "company_id": str(integrated_issue.get("main_company") or ""),
            "company_name_ko": "",
            "peer_meaning": "",
            "capability_change": "",
            "sourced_evidence_ids": [],
        },
        "skax_implication": {
            "why_important": "",
            "potential_impact": "",
            "opportunities": [],
            "threats": [],
            "recommended_actions": [],
            "business_line_mapping": [],
        },
        "follow_up_questions": [],
        "watch_points": [],
        "confidence": 0.0,
        "evidence_label": "insufficient",
        "provenance": {
            "generator": "StrategicInsightAgent",
            "prompt_version": _PROMPT_VERSION,
            "model": _LLM_MODEL,
            "used_fact_ids": [],
            "used_context_layers": [],
            "run_at": datetime.now(UTC).isoformat(),
        },
    }
    return {
        "is_valid_strategic_insight": False,
        "analysis": analysis,
        "implication": implication,
    }


def _is_valid_integrated_issue(integrated_issue: dict[str, Any]) -> bool:
    return bool(
        integrated_issue
        and integrated_issue.get("is_valid_summary", True)
        and integrated_issue.get("main_company")
        and (
            integrated_issue.get("integrated_text")
            or integrated_issue.get("fact_summary")
            or integrated_issue.get("consolidated_facts")
        )
    )


def _integrated_issue_for_prompt(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": integrated_issue.get("bundle_id"),
        "cluster_id": integrated_issue.get("cluster_id"),
        "representative_id": integrated_issue.get("representative_id"),
        "source_article_ids": integrated_issue.get("source_article_ids", []),
        "main_company": integrated_issue.get("main_company", ""),
        "mentioned_peer_companies": integrated_issue.get("mentioned_peer_companies", []),
        "cluster_event_type": integrated_issue.get("cluster_event_type", ""),
        "headline": integrated_issue.get("headline", ""),
        "main_event": integrated_issue.get("main_event", ""),
        "main_issue": integrated_issue.get("main_issue", ""),
        "one_line_summary": integrated_issue.get("one_line_summary", ""),
        "integrated_text": integrated_issue.get("integrated_text", ""),
        "fact_summary": integrated_issue.get("fact_summary", []),
        "consolidated_facts": (integrated_issue.get("consolidated_facts") or [])[:10],
        "key_numbers": integrated_issue.get("key_numbers", []),
        "business_signals": (integrated_issue.get("business_signals") or [])[:8],
        "representative_sources": integrated_issue.get("representative_sources", []),
        "fact_basis": (integrated_issue.get("fact_basis") or [])[:10],
        "missing_or_uncertain_points": integrated_issue.get("missing_or_uncertain_points", []),
        "confidence": integrated_issue.get("confidence", 0.0),
    }


def _classification_for_prompt(classification: dict[str, Any]) -> dict[str, Any]:
    return {
        "sector": classification.get("sector", ""),
        "sectors": classification.get("sectors", []),
        "event_type": classification.get("event_type", ""),
        "importance": classification.get("importance", ""),
        "importance_score": classification.get("importance_score", 0.0),
        "exposure_band": classification.get("exposure_band", ""),
        "exposure_score": classification.get("exposure_score", 0.0),
        "signals": classification.get("signals", {}),
    }


def _bundle_for_prompt(
    bundle: dict[str, Any],
    cluster_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = bundle.get("metadata") or {}
    return {
        "bundle_id": bundle.get("bundle_id") or cluster_metadata.get("bundle_id"),
        "cluster_id": bundle.get("cluster_id") or cluster_metadata.get("cluster_id"),
        "source_type": bundle.get("source_type") or cluster_metadata.get("source_type"),
        "companies": bundle.get("companies") or cluster_metadata.get("companies", []),
        "sectors": bundle.get("sectors") or cluster_metadata.get("sectors", []),
        "event_type": bundle.get("event_type") or cluster_metadata.get("event_type"),
        "source_count": len(bundle.get("sources") or []),
        "metadata": {
            "representative_id": metadata.get("representative_id"),
            "cluster_article_ids": metadata.get("cluster_article_ids", []),
            "created_at": metadata.get("created_at", ""),
            "trend_context": metadata.get("trend_context", {}),
        },
    }


def _profile_for_prompt(profile: dict[str, Any]) -> dict[str, Any]:
    skax = profile.get("skax_profile") or {}
    peer_profiles = profile.get("peer_profiles") or {}
    return {
        "skax_profile": _shrink_profile(skax),
        "peer_profiles": {
            str(peer_id): _shrink_profile(payload)
            for peer_id, payload in (
                peer_profiles.items() if isinstance(peer_profiles, dict) else []
            )
        },
        "sector_context": profile.get("sector_context") or {},
    }


def _analysis_context_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "peer_event_timeline_recent": (context.get("peer_event_timeline_recent") or [])[:8],
        "sector_pulse_recent": (context.get("sector_pulse_recent") or [])[:4],
        "financial_trend": context.get("financial_trend") or {},
        "event_chain_candidates": (context.get("event_chain_candidates") or [])[:5],
        "similar_cards_rag": (context.get("similar_cards_rag") or [])[:5],
        "evidence_density_per_peer": context.get("evidence_density_per_peer") or {},
        "provenance": context.get("provenance") or {},
    }


def _bundle_to_dict(value: AnalysisInputBundle | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, AnalysisInputBundle):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _profile_to_dict(value: ProfileContext | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, ProfileContext):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _analysis_context_to_dict(value: AnalysisContext | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, AnalysisContext):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _cluster_metadata_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": bundle.get("bundle_id", ""),
        "cluster_id": bundle.get("cluster_id"),
        "source_type": bundle.get("source_type", ""),
        "companies": bundle.get("companies", []),
        "sectors": bundle.get("sectors", []),
        "event_type": bundle.get("event_type"),
        "cluster_size": len(bundle.get("items") or []),
        "source_count": len(bundle.get("sources") or []),
    }


def _business_line_candidates(profile: dict[str, Any]) -> list[str]:
    skax = profile.get("skax_profile") or {}
    candidates: list[str] = []
    for item in _string_list(skax.get("business_lines"), max_items=20):
        if item not in candidates:
            candidates.append(item)
    business_areas = skax.get("business_areas") or []
    if isinstance(business_areas, list):
        for area in business_areas:
            if not isinstance(area, dict):
                continue
            name = str(area.get("name") or "").strip()
            if name and name not in candidates:
                candidates.append(name)
            if len(candidates) >= 20:
                break
    return candidates


def _business_line_candidate_details(profile: dict[str, Any]) -> list[dict[str, Any]]:
    skax = profile.get("skax_profile") or {}
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for name in _string_list(skax.get("business_lines"), max_items=20):
        if name in seen:
            continue
        candidates.append({"name": name})
        seen.add(name)

    business_areas = skax.get("business_areas") or []
    if isinstance(business_areas, list):
        for area in business_areas:
            if not isinstance(area, dict):
                continue
            name = str(area.get("name") or "").strip()
            if not name or name in seen:
                continue
            candidates.append(
                {
                    "name": name,
                    "summary": str(area.get("summary") or "").strip(),
                    "core_capabilities": _string_list(area.get("core_capabilities"), max_items=5),
                    "recent_direction": str(area.get("recent_direction") or "").strip(),
                    "source_refs": _compact_value(area.get("source_refs") or []),
                }
            )
            seen.add(name)
            if len(candidates) >= 20:
                break
    return candidates


def _normalize_recommended_actions(actions: list[str]) -> list[str]:
    """Keep LLM-written action meaning; only trim whitespace and exact duplicates."""
    normalized: list[str] = []
    seen: set[str] = set()
    for action in actions:
        text = re.sub(r"\s+", " ", str(action or "")).strip()
        if not text:
            continue
        key = text.rstrip(".。").casefold()
        if key in seen:
            continue
        normalized.append(text)
        seen.add(key)
    return normalized


def _quality_gate_violations(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> list[str]:
    evidence_text = _grounding_text(
        integrated_issue=integrated_issue,
        profile_context=profile_context,
    )
    integrated_evidence_text = _integrated_grounding_text(integrated_issue)
    violations: list[str] = []
    for label, value_text in _quality_checked_texts(result):
        for pattern in _UNSUPPORTED_CLAIM_PATTERNS:
            if _has_unsupported_pattern(value_text, pattern, evidence_text=evidence_text):
                violations.append(f"{label}: 입력 근거 없이 `{pattern}` 계열 표현을 사용했습니다.")
        for pattern in _BROAD_RESULT_PATTERNS:
            if _has_broad_result_violation(
                value_text,
                pattern=pattern,
                integrated_issue=integrated_issue,
            ):
                violations.append(f"{label}: 넓은 결과 표현을 구체 변화로 풀어써야 합니다.")
        relation_violation = _relationship_grounding_violation(
            value_text,
            integrated_evidence_text=integrated_evidence_text,
        )
        if relation_violation:
            violations.append(f"{label}: {relation_violation}")

    analysis = result.get("analysis") or {}
    market_signal = str(analysis.get("market_signal") or "").strip()
    if market_signal and _is_generic_market_signal(
        market_signal,
        integrated_issue=integrated_issue,
    ):
        violations.append(
            "analysis.market_signal: 통합 근거의 사업명, 관계 유형, 검증 단계, "
            "운영 대상 없이 일반론으로 작성되었습니다."
        )
    for label, reason_text in (
        ("analysis.reason", str(analysis.get("reason") or "").strip()),
        ("analysis.impact_reason", str(analysis.get("impact_reason") or "").strip()),
    ):
        if reason_text and _has_internal_or_generic_phrase(reason_text):
            violations.append(
                f"{label}: 내부 근거 지칭이나 범용 표현 대신 고객명/사업명/검증 단계로 써야 합니다."
            )

    skax = (result.get("implication") or {}).get("skax_implication") or {}
    potential_impact = str(skax.get("potential_impact") or "").strip()
    if potential_impact:
        if _has_internal_or_generic_phrase(potential_impact):
            violations.append(
                "skax_implication.potential_impact: 내부 근거 지칭이나 범용 표현 대신 "
                "고객 평가 기준과 바뀔 산출물을 구체화해야 합니다."
            )
        if _sentence_count(potential_impact) < 2:
            violations.append(
                "skax_implication.potential_impact: "
                "고객 평가 변화와 SK AX 대응을 2문장으로 분리해야 합니다."
            )
        if not _IMPACT_EVALUATION_PATTERN.search(potential_impact):
            violations.append(
                "skax_implication.potential_impact: "
                "고객이 무엇을 평가하거나 비교하는지 빠져 있습니다."
            )
        if not _IMPACT_ARTIFACT_PATTERN.search(potential_impact):
            violations.append(
                "skax_implication.potential_impact: "
                "제안서, PoC, 레퍼런스, 운영 모델 중 바뀔 산출물이 빠져 있습니다."
            )
        if re.search(r"재검토해야\s*합니다|영향을\s*줄\s*수\s*있습니다\.?$", potential_impact):
            violations.append(
                "skax_implication.potential_impact: 이유 없는 결론형 문장으로 끝납니다."
            )

    actions = _string_list(skax.get("recommended_actions"), max_items=3)
    if _has_repeated_action_template(actions):
        violations.append(
            "skax_implication.recommended_actions: 같은 꼬리 문장이나 같은 template 이 반복됩니다."
        )
    for index, action in enumerate(actions, start=1):
        if not _ACTION_SENTENCE_END_PATTERN.search(action.strip()):
            violations.append(
                f"skax_implication.recommended_actions[{index}]: "
                "완성된 실행 문장으로 끝나야 합니다."
            )
        if re.search(r"(강조|강화|재검토)(합니다|해야 합니다)\.?$", action) and not (
            _IMPACT_ARTIFACT_PATTERN.search(action)
        ):
            violations.append(
                f"skax_implication.recommended_actions[{index}]: 실행 장면 없는 추상 액션입니다."
            )
        if "강조" in action and not re.search(
            r"설명|분리|검증|제시|비교|항목|범위|책임|기준", action
        ):
            violations.append(
                f"skax_implication.recommended_actions[{index}]: "
                "무엇을 설명하거나 검증할지 없는 강조 중심 액션입니다."
            )

    return list(dict.fromkeys(violations))


def _quality_checked_texts(result: dict[str, Any]) -> list[tuple[str, str]]:
    analysis = result.get("analysis") or {}
    implication = result.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    items: list[tuple[str, Any]] = [
        ("analysis.analysis_summary", analysis.get("analysis_summary")),
        ("analysis.market_signal", analysis.get("market_signal")),
        ("analysis.impact_reason", analysis.get("impact_reason")),
        ("analysis.reason", analysis.get("reason")),
        ("peer_implication.peer_meaning", peer.get("peer_meaning")),
        ("peer_implication.capability_change", peer.get("capability_change")),
        ("skax_implication.why_important", skax.get("why_important")),
        ("skax_implication.potential_impact", skax.get("potential_impact")),
    ]
    for index, value in enumerate(_string_list(analysis.get("strategic_meaning"), max_items=3), 1):
        items.append((f"analysis.strategic_meaning[{index}]", value))
    for field in ("opportunities", "threats", "recommended_actions"):
        for index, value in enumerate(_string_list(skax.get(field), max_items=3), 1):
            items.append((f"skax_implication.{field}[{index}]", value))
    for field in ("follow_up_questions", "watch_points"):
        for index, value in enumerate(_string_list(implication.get(field), max_items=3), 1):
            items.append((f"implication.{field}[{index}]", value))
    return [(label, str(value or "").strip()) for label, value in items if str(value or "").strip()]


def _has_unsupported_pattern(text: str, pattern: str, *, evidence_text: str) -> bool:
    if not re.search(pattern, text):
        return False
    return not re.search(pattern, evidence_text)


def _has_broad_result_violation(
    text: str,
    *,
    pattern: str,
    integrated_issue: dict[str, Any],
) -> bool:
    """Flag broad result words only when they stand alone without issue detail."""
    if not re.search(pattern, text):
        return False
    fact_tokens = _content_tokens(_integrated_grounding_text(integrated_issue))
    text_tokens = _content_tokens(text)
    overlap_count = len(fact_tokens & text_tokens)
    has_issue_overlap = overlap_count >= 2
    has_strong_issue_overlap = overlap_count >= 4
    has_event_anchor = bool(_EVENT_ANCHOR_PATTERN.search(text))
    return not (has_strong_issue_overlap or (has_issue_overlap and has_event_anchor))


def _relationship_grounding_violation(
    text: str,
    *,
    integrated_evidence_text: str,
) -> str:
    if not text or not _RELATIONSHIP_PATTERN.search(text):
        return ""
    if not _RELATIONSHIP_PATTERN.search(integrated_evidence_text):
        return "IntegratedIssue 에 없는 협업/파트너십 계열 관계 표현을 사용했습니다."
    if _relationship_only_uncertain(integrated_evidence_text) and not _UNCERTAINTY_PATTERN.search(
        text
    ):
        return "검토/구상/가능성 단계의 관계를 확정 실행처럼 표현했습니다."
    return ""


def _relationship_only_uncertain(text: str) -> bool:
    relation_sentences = [
        sentence for sentence in _split_sentences(text) if _RELATIONSHIP_PATTERN.search(sentence)
    ]
    return bool(relation_sentences) and all(
        _UNCERTAINTY_PATTERN.search(sentence) for sentence in relation_sentences
    )


def _is_generic_market_signal(text: str, *, integrated_issue: dict[str, Any]) -> bool:
    if _has_internal_or_generic_phrase(text):
        return True
    fact_tokens = _content_tokens(_integrated_grounding_text(integrated_issue))
    text_tokens = _content_tokens(text)
    if len(fact_tokens & text_tokens) < 2:
        return True
    return bool(_WEAK_MARKET_SIGNAL_PATTERN.search(text)) and not _EVENT_ANCHOR_PATTERN.search(text)


def _has_internal_or_generic_phrase(text: str) -> bool:
    return bool(_INTERNAL_OR_GENERIC_PATTERN.search(text or ""))


def _grounding_text(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    for key in (
        "headline",
        "main_event",
        "main_issue",
        "one_line_summary",
        "integrated_text",
    ):
        value = str(integrated_issue.get(key) or "").strip()
        if value:
            parts.append(value)
    parts.extend(str(item or "") for item in integrated_issue.get("fact_summary") or [])
    for _, fact_text in _fact_texts(integrated_issue):
        parts.append(fact_text)
    if profile_context:
        parts.append(_json_dumps(_profile_for_prompt(profile_context)))
    return "\n".join(parts)


def _integrated_grounding_text(integrated_issue: dict[str, Any]) -> str:
    return _grounding_text(integrated_issue=integrated_issue, profile_context=None)


def _sentence_count(text: str) -> int:
    return len(_split_sentences(text))


def _split_sentences(text: str) -> list[str]:
    return [item for item in re.split(r"[.!?。]\s*", str(text or "").strip()) if item.strip()]


def _split_sentences_with_delimiters(text: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for match in re.finditer(r"([^.!?。]+)([.!?。]?\s*)", str(text or "").strip()):
        sentence = match.group(1).strip()
        if sentence:
            result.append((sentence, match.group(2)))
    return result or [(str(text or "").strip(), "")]


def _has_repeated_action_template(actions: list[str]) -> bool:
    tails: list[str] = []
    for action in actions:
        compact = re.sub(r"\s+", "", action.rstrip(".。"))
        if len(compact) >= 18:
            tails.append(compact[-18:])
    return len(tails) != len(set(tails))


def _mark_quality_gate_failed(result: dict[str, Any], violations: list[str]) -> dict[str, Any]:
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    out["is_valid_strategic_insight"] = False
    analysis = out.get("analysis") or {}
    implication = out.get("implication") or {}
    analysis["is_valid_analysis"] = False
    analysis["confidence"] = min(_clamp_float(analysis.get("confidence"), 0.0), 0.3)
    base_reason = str(analysis.get("reason") or "").strip()
    violation_text = " / ".join(violations[:3])
    analysis["reason"] = (
        f"{base_reason} | quality_gate_failed: {violation_text}"
        if base_reason
        else f"quality_gate_failed: {violation_text}"
    )
    implication["is_valid_implication"] = False
    implication["confidence"] = min(_clamp_float(implication.get("confidence"), 0.0), 0.3)
    implication["evidence_label"] = "insufficient"
    return out


def _fallback_quality_repair(
    result: dict[str, Any],
    *,
    violations: list[str],
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    focus = _issue_focus_sentence(integrated_issue)
    integrated_evidence_text = _integrated_grounding_text(integrated_issue)
    analysis = out.get("analysis") or {}
    implication = out.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}

    for key in ("analysis_summary", "market_signal", "impact_reason", "reason"):
        if analysis.get(key):
            analysis[key] = _fallback_repair_text(
                str(analysis[key]),
                focus=focus,
                integrated_evidence_text=integrated_evidence_text,
            )
    analysis["strategic_meaning"] = [
        _fallback_repair_text(
            item,
            focus=focus,
            integrated_evidence_text=integrated_evidence_text,
        )
        for item in _string_list(analysis.get("strategic_meaning"), max_items=3)
    ]
    for key in ("peer_meaning", "capability_change"):
        if peer.get(key):
            peer[key] = _fallback_repair_text(
                str(peer[key]),
                focus=focus,
                integrated_evidence_text=integrated_evidence_text,
            )
    for key in ("why_important", "potential_impact"):
        if skax.get(key):
            skax[key] = _fallback_repair_text(
                str(skax[key]),
                focus=focus,
                integrated_evidence_text=integrated_evidence_text,
            )
    for key in ("opportunities", "threats", "recommended_actions"):
        skax[key] = [
            _fallback_repair_text(
                item,
                focus=focus,
                integrated_evidence_text=integrated_evidence_text,
            )
            for item in _string_list(skax.get(key), max_items=3)
        ]
    for key in ("follow_up_questions", "watch_points"):
        implication[key] = [
            _fallback_repair_text(
                item,
                focus=focus,
                integrated_evidence_text=integrated_evidence_text,
            )
            for item in _string_list(implication.get(key), max_items=3)
        ]

    out["analysis"] = analysis
    implication["peer_implication"] = peer
    implication["skax_implication"] = skax
    out["implication"] = implication
    return out


def _fallback_repair_text(
    text: str,
    *,
    focus: str,
    integrated_evidence_text: str,
) -> str:
    repaired = _soften_quality_text(text, focus=focus)
    return _preserve_uncertain_relationship_level(
        repaired,
        integrated_evidence_text=integrated_evidence_text,
    )


def _preserve_uncertain_relationship_level(
    text: str,
    *,
    integrated_evidence_text: str,
) -> str:
    if not text or not _RELATIONSHIP_PATTERN.search(text):
        return text
    if not _relationship_only_uncertain(integrated_evidence_text):
        return text

    repaired: list[str] = []
    for sentence, delimiter in _split_sentences_with_delimiters(text):
        if _RELATIONSHIP_PATTERN.search(sentence) and not _UNCERTAINTY_PATTERN.search(sentence):
            sentence = re.sub(r"(협업|협력|파트너십|제휴)", r"\1 가능성 검토", sentence)
            sentence = re.sub(
                r"공동\s*(개발|사업|운영|추진)",
                r"공동 \1 가능성 검토",
                sentence,
            )
            sentence = _normalize_uncertain_relationship_particles(sentence)
        repaired.append(f"{sentence}{delimiter}")
    return "".join(repaired).strip()


def _normalize_uncertain_relationship_particles(text: str) -> str:
    out = str(text or "")
    replacements = (
        (r"가능성 검토이", "가능성 검토가"),
        (r"가능성 검토가가", "가능성 검토가"),
        (r"가능성 검토을", "가능성 검토를"),
        (r"가능성 검토를를", "가능성 검토를"),
        (r"가능성 검토은", "가능성 검토는"),
        (r"가능성 검토는는", "가능성 검토는"),
        (r"가능성 검토와", "가능성 검토와"),
    )
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out)
    return out


def _soften_quality_text(text: str, *, focus: str) -> str:
    replacements = (
        (
            r"기술적\s*역량을\s*입증하는\s*사례",
            "확인된 적용 사례",
        ),
        (
            r"기술적\s*역량을\s*입증했습니다",
            "확인된 적용 사례를 제시했습니다",
        ),
        (
            r"기술적\s*역량을\s*입증하며",
            "확인된 적용 사례를 바탕으로",
        ),
        (
            r"기회를\s*창출할\s*수\s*있습니다",
            "검토할 지점입니다",
        ),
        (r"시장\s*점유율\s*확대", "후속 적용 범위 확인 필요"),
        (r"점유율[이을가\s]*(확대|상승|증가)", "후속 적용 범위 확인 필요"),
        (r"시장\s*선점", "초기 적용 신호"),
        (r"선점", "초기 적용 신호"),
        (r"기술적\s*우위", "확인된 적용 범위"),
        (r"기술적\s*발전", "운영 방식 변화"),
        (r"혁신성", "검증 방식"),
        (r"경쟁\s*심화", "검증 근거 비교 증가"),
        (r"격차", "검증 근거 차이"),
        (r"리더십\s*확보", "확인된 적용 사례"),
        (r"매출\s*기여", "사업 영향 확인 필요"),
        (r"큰\s*영향", "제안 기준 변화"),
        (r"시장\s*확장\s*가능성", "적용 범위 변화 가능성"),
        (r"시장\s*확장", "적용 범위 변화"),
        (r"생태계\s*확장", "적용 범위 변화"),
        (
            r"차별화된\s*가치를\s*제공할\s*수\s*있는\s*기회",
            "검증 가능한 적용 근거를 제시할 수 있는 지점",
        ),
        (r"차별화된\s*가치", "검증 가능한 적용 근거"),
        (r"기회를\s*창출", "검토할 지점이 됩니다"),
        (r"기술적\s*시너지", "협업 방식"),
        (r"기술적\s*역량을\s*입증", "확인된 적용 사례를 제시"),
        (r"기술적\s*역량", "확인된 역량 맥락"),
        (r"기술적\s*협업", "협업 방식"),
        (r"수요가\s*증가하고\s*있습니다", "수요 신호가 확인됩니다"),
        (r"수요가\s*확대되고\s*있습니다", "수요 신호가 확인됩니다"),
        (r"수요를\s*증가시", "수요 신호를 구체화시"),
        (r"수요를\s*확대시", "수요 신호를 구체화시"),
        (r"더\s*중시하게\s*될\s*것입니다", "함께 비교할 수 있습니다"),
        (r"더\s*중시하게\s*됩니다", "함께 비교하게 됩니다"),
        (r"더\s*중시", "함께 비교"),
        (r"맞춤형\s*솔루션", "고객 업무 조건에 맞춘 제안 항목"),
        (r"시장\s*진입", "적용 가능성 검토"),
        (r"가능성을\s*열어\s*줍니다", "검토할 지점을 남깁니다"),
        (r"가능성을\s*열어\s*주기", "검토할 지점을 남기기"),
        (r"입증", "제시"),
        (r"입지[를\s]*강화하고\s*있습니다", "적용 범위를 구체화하고 있습니다"),
        (r"입지[를\s]*강화하는", "적용 범위를 구체화하는"),
        (r"입지[를\s]*강화할", "적용 범위를 구체화할"),
        (r"입지[를\s]*강화", "적용 범위 구체화"),
        (r"경쟁력[을\s]*강화하고\s*있습니다", "제안 근거를 구체화하고 있습니다"),
        (r"경쟁력[을\s]*강화하는", "제안 근거를 구체화하는"),
        (r"경쟁력[을\s]*강화할", "제안 근거를 구체화할"),
        (r"경쟁력[을\s]*높이고\s*있습니다", "제안 근거를 구체화하고 있습니다"),
        (r"경쟁력[을\s]*높이는", "제안 근거를 구체화하는"),
        (r"경쟁력[을\s]*높일", "제안 근거를 구체화할"),
        (r"경쟁력[을\s]*강화", "제안 근거 구체화"),
        (r"경쟁력[을\s]*높", "제안 근거 구체화"),
    )
    out = str(text or "").strip()
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out)
    out = re.sub(r"확대이", "확대가", out)
    out = re.sub(r"적용 범위을", "적용 범위를", out)
    out = re.sub(r"구체화하이", "구체화하", out)
    out = re.sub(r"됩니다할\s*수\s*있습니다", "됩니다", out)
    out = re.sub(r"지점이\s*됩니다할\s*수\s*있습니다", "지점입니다", out)
    if not out:
        return focus
    return out


def _issue_focus_sentence(integrated_issue: dict[str, Any]) -> str:
    candidates = [
        integrated_issue.get("one_line_summary"),
        integrated_issue.get("main_event"),
        integrated_issue.get("main_issue"),
        integrated_issue.get("headline"),
        *((integrated_issue.get("fact_summary") or [])[:2]),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return re.split(r"[.。]\s*", text)[0].strip()
    for _, fact_text in _fact_texts(integrated_issue):
        text = re.split(r"[.。]\s*", fact_text.strip())[0].strip()
        if text:
            return text
    return "입력 근거에서 확인된 신호"


def _augment_sourced_evidence_ids(
    existing: list[str],
    *,
    integrated_issue: dict[str, Any],
    output_texts: list[Any],
) -> list[str]:
    out = list(dict.fromkeys(existing))
    known = _fact_texts(integrated_issue)
    combined_output = " ".join(str(item or "") for item in output_texts)
    output_tokens = _content_tokens(combined_output)
    for fact_id, fact_text in known:
        if fact_id in out:
            continue
        if _fact_is_referenced(fact_text, combined_output, output_tokens):
            out.append(fact_id)
        if len(out) >= 10:
            break
    return out


def _fact_texts(integrated_issue: dict[str, Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for basis in integrated_issue.get("fact_basis") or []:
        if not isinstance(basis, dict):
            continue
        fact_ids = [str(item or "").strip() for item in basis.get("fact_ids") or []]
        texts = [str(basis.get("fact") or ""), str(basis.get("evidence_text") or "")]
        texts.extend(str(item or "") for item in basis.get("evidence_texts") or [])
        combined = " ".join(text for text in texts if text)
        for fact_id in fact_ids:
            if not fact_id or fact_id in seen or not combined:
                continue
            items.append((fact_id, combined))
            seen.add(fact_id)
    for fact in integrated_issue.get("consolidated_facts") or []:
        if not isinstance(fact, dict):
            continue
        fact_id = str(fact.get("fact_id") or "").strip()
        if not fact_id or fact_id in seen:
            continue
        texts = [str(fact.get("fact") or "")]
        texts.extend(str(item or "") for item in fact.get("evidence_texts") or [])
        items.append((fact_id, " ".join(texts)))
        seen.add(fact_id)
    return items


def _fact_is_referenced(fact_text: str, output_text: str, output_tokens: set[str]) -> bool:
    tokens = _content_tokens(fact_text)
    if len(tokens & output_tokens) >= 2:
        return True
    for token in tokens:
        if len(token) >= 4 and token in output_text:
            return True
    return False


def _content_tokens(text: str) -> set[str]:
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·_-]{1,}", text or "")
    return {
        normalized
        for token in raw_tokens
        if (normalized := _normalize_content_token(token))
        and not _is_low_signal_content_token(normalized)
    }


def _is_low_signal_content_token(token: str) -> bool:
    if len(token) <= 1:
        return True
    if token.isascii() and token.isupper() and len(token) <= 2:
        return True
    if re.fullmatch(r"\d+", token):
        return len(token) <= 1
    return False


def _normalize_content_token(token: str) -> str:
    token = token.strip()
    if len(token) <= 3:
        return token
    return re.sub(r"(으로|에서|에게|과|와|은|는|이|가|을|를|의)$", "", token)


def _shrink_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    keys = (
        "company_id",
        "peer_id",
        "company_name",
        "company_name_ko",
        "one_liner",
        "company_summary",
        "business_lines",
        "business_areas",
        "core_capabilities",
        "recent_keywords",
        "recent_changes",
        "recent_signals",
        "recent_financial",
        "financial_summary",
        "market_view",
        "capability_evolution",
        "cautions",
        "narrative",
    )
    return {key: _compact_value(profile[key]) for key in keys if key in profile}


def _compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:700]
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:5]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            compacted = _compact_value(nested)
            if compacted not in ({}, [], "", None):
                out[str(key)] = compacted
            if len(out) >= 10:
                break
        return out
    return value


def _parse_json_loose(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.startswith("json"):
                body = body[4:]
            text = body.strip()
    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            text = text[first : last + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _known_fact_ids(integrated_issue: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in integrated_issue.get("fact_basis") or []:
        if not isinstance(item, dict):
            continue
        for fact_id in item.get("fact_ids") or []:
            text = str(fact_id or "").strip()
            if text:
                ids.add(text)
        fact_id = str(item.get("fact_id") or "").strip()
        if fact_id:
            ids.add(fact_id)
    for item in integrated_issue.get("consolidated_facts") or []:
        if isinstance(item, dict):
            fact_id = str(item.get("fact_id") or "").strip()
            if fact_id:
                ids.add(fact_id)
    return ids


def _used_context_layers(context: dict[str, Any]) -> list[str]:
    layers: list[str] = []
    if context.get("peer_event_timeline_recent"):
        layers.append("peer_event_timeline_recent")
    if context.get("sector_pulse_recent"):
        layers.append("sector_pulse_recent")
    if context.get("financial_trend"):
        layers.append("financial_trend")
    if context.get("event_chain_candidates"):
        layers.append("event_chain_candidates")
    if context.get("similar_cards_rag"):
        layers.append("similar_cards_rag")
    return layers


def _normalize_used_context_layers(value: Any, *, analysis_context: dict[str, Any]) -> list[str]:
    available = _used_context_layers(analysis_context)
    if not available:
        return []
    requested = _string_list(value, max_items=10)
    if not requested:
        return available
    return [layer for layer in requested if layer in available]


def _first_peer_id(profile: dict[str, Any]) -> str:
    peer_profiles = profile.get("peer_profiles") or {}
    if isinstance(peer_profiles, dict):
        for peer_id in peer_profiles:
            if peer_id:
                return str(peer_id)
    return ""


def _first_peer_name(profile: dict[str, Any]) -> str:
    peer_profiles = profile.get("peer_profiles") or {}
    if isinstance(peer_profiles, dict):
        for payload in peer_profiles.values():
            if isinstance(payload, dict):
                name = payload.get("company_name_ko") or payload.get("company_name")
                if name:
                    return str(name)
    return ""


def _string_list(value: Any, *, max_items: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list | tuple | set):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
            if len(out) >= max_items:
                break
        return out
    return []


def _choice(value: Any, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else default


def _evidence_label(value: Any, confidence: float) -> str:
    raw = str(value or "").strip().lower()
    if raw in _EVIDENCE_LABELS:
        if raw == "sufficient" and confidence < 0.6:
            return "insufficient"
        return raw
    if confidence < 0.6:
        return "insufficient"
    if confidence < 0.8:
        return "moderate"
    return "sufficient"


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(min(max(number, 0.0), 1.0), 3)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


__all__ = ["StrategicInsightAgent"]
