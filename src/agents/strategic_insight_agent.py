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
from typing import Any, Sequence

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.agents.implication_agent import ImplicationAgent
from src.agents.strategic_analyzer import StrategicAnalyzer
from src.analysis.models import AnalysisContext, AnalysisInputBundle, ProfileContext
from src.db.postgres import SessionLocal
from src.services.analysis_context_builder import AnalysisContextBuilder
from src.services.peer_id_aliases import expand_peer_aliases
from src.services.profile_context_loader import ProfileContextLoader

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_PROMPT_VERSION = "strategic-insight-v1.39-report-specificity"
_LLM_TEMPERATURE = 0.0
_LLM_MAX_COMPLETION_TOKENS = 2600

_IMPACT_LEVELS = {"high", "medium", "low"}
_RISK_OR_OPPORTUNITY = {"risk", "opportunity", "neutral"}
_EVIDENCE_LABELS = {"sufficient", "moderate", "insufficient"}
# 근거 없이 쓰면 사실 왜곡이 큰 고위험 주장만 최소 차단한다.
# 표현 품질은 아래 구조 게이트와 프롬프트가 담당하고, 문구 blacklist 를 늘리지 않는다.
_UNSUPPORTED_CLAIM_PATTERNS = (
    r"시장\s*점유율\s*확대",
    r"점유율[이을가\s]*(확대|상승|증가)",
    r"시장\s*선점",
    r"선점",
    r"기술적\s*우위",
    r"격차",
    r"리더십\s*확보",
    r"매출\s*기여",
    r"큰\s*영향",
    r"수요[가를\s]*(증가|확대)",
    r"고객.{0,12}기대[가를을\s]*(높|상승)",
    r"적용\s*범위\s*확대",
    r"시장\s*점유율\s*감소",
    r"점유율[이을가\s]*(감소|하락|축소)",
    r"경쟁\s*심화",
    r"속도[가를\s]*(증가|상승|가속)",
    r"가속화",
    r"초기\s*단계",
)
_RELATIONSHIP_PATTERN = re.compile(r"협업|협력|파트너십|제휴|공동")
_RELATIONSHIP_ACTIVITY_TYPES = {"partnership", "collaboration", "alliance", "joint", "mou"}
_UNCERTAINTY_PATTERN = re.compile(r"검토|가능성|구상|계획|예정|추진|모색|방향")
_SUPPLIER_CAPABILITY_PATTERN = re.compile(
    r"(공급|납품)\s*역량|공급\s*계약.{0,30}(제공|수행)\s*역량"
)
_SUPPLY_CONTRACT_PATTERN = re.compile(r"공급\s*계약|공급계약|납품|구매|조달")
_NUMERIC_TOKEN_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|usd|krw)?",
    re.IGNORECASE,
)


SYSTEM_PROMPT = """\
당신은 임원 보고용 전략 인사이트를 작성하는 Agent입니다.

역할:
- IntegratedIssue 로 현재 사건의 사실과 피어사/시장 의미를 분석합니다.
- ProfileContext 는 기존 사업영역/역량 배경으로만 사용합니다.
- AnalysisContext 는 흐름 보조로만 사용합니다.
- analysis, peer_implication, skax_implication 의 관점을 섞지 않습니다.

핵심 원칙:
1. 현재 사건의 사실 근거는 IntegratedIssue 에서만 가져옵니다.
2. 수치, 날짜, 고객명, 제품명, 회사명은 IntegratedIssue 의 fact_basis, key_numbers,
   representative_sources, consolidated_facts 중 하나에 있어야 합니다.
   수치는 반올림하거나 "이상/내외"로 바꾸지 말고 근거 형태 그대로 씁니다.
3. 계약/수주/선정/투자/검토 같은 관계 수준을 보존합니다.
   계약 상대방의 정확한 역할이 불명확하면 고객·원청·도입 주체로 단정하지 말고
   "계약 상대방", "사업과 연결된 피어사", "해당 과제와 연결된 신호"처럼 씁니다.
4. 기술적 우위, 혁신성, 선점, 격차, 경쟁 심화, 시장 점유율 확대처럼
   비교 근거가 필요한 표현은 직접 근거가 없으면 쓰지 않습니다.
5. 시사점에는 기사에 직접 없는 운영 책임·검증 지표를 넣지 않습니다.
   그런 항목은 SK AX 대응방향에서 챙길 기준으로만 사용합니다.
6. 독자가 "왜 이런 분석과 대응이 나왔는지" 바로 납득할 수 있게
   근거 사실 → 해석 기준 → SK AX가 바꿀 산출물/운영 방식 순서로 씁니다.
7. "초기 단계", "경쟁 심화", "고객 평가 기준 변화"처럼 넓은 표현은
   IntegratedIssue 에 직접 근거가 있거나, 바로 뒤에 확인된 사업명·계약 범위·기간·
   적용 대상·검증 대상이 붙는 경우에만 씁니다.
   "수요 증가/확대", "고객 기대 상승", "큰 영향", "적용 범위 확대"도 직접 근거가 없으면
   쓰지 말고 "수요 신호 확인", "비교/확인 기준", "참고 근거" 수준으로 낮춥니다.
8. JSON 외 텍스트를 출력하지 마세요.
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

## 역할 해석 모드
{role_mode_instructions}

## 작성 기준
1. 사실 근거
   - IntegratedIssue 의 고객명, 사업명, 제품/서비스명, 금액, 기간, 비율,
     관계 수준을 우선 사용합니다.
   - 수치와 날짜는 근거에 있는 표현 그대로만 씁니다.
   - cluster_fact_intelligence.products_or_services 에 사업/제품명이 있으면
     "관련 프로젝트", "디지털 전환 흐름" 같은 넓은 말보다 그 명칭을 우선 사용합니다.
   - role_interpretation_hints 가 있으면 반드시 따릅니다. 타깃 피어가 고객/계약 상대방으로
     보이는 경우, 공급사의 매출 비율이나 공급 역량을 타깃 피어의 성과/역량처럼 쓰지 말고
     계약 규모나 사업 연결성의 근거로만 사용합니다.
   - "A사가 타깃 피어 B와 계약했다"는 구조에서는 A사의 매출 비율, 성장 가능성,
     시장 입지를 B사의 전략 의미로 쓰지 않습니다. B사의 의미는 B가 연결된 사업 범위,
     계약 기간, 산업/고객군, B의 프로필 역량과의 접점에서만 해석합니다.
   - 타깃 피어가 customers_or_industries 슬롯에 있고 공급사가 따로 보이면,
     "타깃 피어가 프로젝트를 추진/참여/제공/공급/수행했다"처럼 쓰지 않습니다.
     "타깃 피어가 해당 계약의 상대방으로 확인됐고, 그 사업 범위가 피어 프로필의
     어떤 사업영역과 연결된다"는 수준으로 씁니다.
   - 계약 상대방의 역할이 불명확하면 고객/도입 주체로 단정하지 말고 "계약 상대방",
     "해당 사업과 연결된 피어사"처럼 보수적으로 씁니다.
   - 기술적 우위, 혁신성, 선점, 격차, 경쟁 심화, 점유율 확대는 직접 비교 근거가 없으면 금지합니다.

2. analysis
   - analysis_summary: 핵심 근거와 피어/시장 의미를 함께 씁니다. 1~2문장까지 허용합니다.
   - strategic_meaning: 2~3개. 각 항목은 "근거 사실 → 무엇이 구체화됐는지" 구조입니다.
   - market_signal: "고객 요구 변화" 같은 일반론 금지. 확인된 사업/계약/검증/적용 범위 안에서
     시장 또는 고객 평가 신호를 씁니다. 한 사건만으로 수요 증가를 단정하지 않습니다.
     "변화하고 있습니다", "초기 단계"만 쓰지 말고, 확인된 사업명/계약 범위/기간/적용 대상을
     넣어 무엇이 신호인지 설명합니다.
   - impact_reason: 영향도 판단 근거입니다.
   - reason: 사용한 fact_id/profile/context가 어떤 해석에 쓰였는지 설명합니다.

3. implication
   - peer_implication.peer_meaning: 짧은 요약이 아니라 2문장입니다.
     첫 문장은 IntegratedIssue 핵심 근거, 둘째 문장은 ProfileContext 의
     기존 사업영역/역량과 연결합니다.
   - peer_implication.capability_change: 기사 근거에서 직접 확인되는 적용 범위, 계약 범위,
     운영 기간, 고객군, 사업 연결성만 씁니다. 검증 지표나 운영 책임처럼
     기사에 없는 대응 기준은 쓰지 않습니다.
     타깃 피어가 공급자가 아니라 계약 상대방/고객으로 잡힌 경우 "제공 역량 강화",
     "공급 역량 확대"처럼 공급자 관점으로 쓰지 말고, 어떤 현대화 과제와 연결됐는지,
     어떤 사업 범위가 관찰되는지로 씁니다.
     공급사 매출 비율은 계약 규모 설명에는 쓸 수 있지만, 타깃 피어의 역량 변화나
     시장 입지 변화 근거로 쓰지 않습니다.
   - potential_impact: 2문장입니다. 첫 문장은 IntegratedIssue 근거 신호 때문에 고객이 무엇을
     함께 비교하거나 확인할 수 있는지, 둘째 문장은 SK AX 산출물이 어떻게 달라져야 하는지입니다.
     이유 없는 결론으로 끝내지 마세요. 입력에 있는 사업명/고객군/운영 구간/검증 대상을 사용합니다.
     "고객 평가 기준이 변화"라고만 쓰지 말고, 어떤 유사 고객군이 어떤 사업에서 무엇을
     비교/확인할 수 있는지 씁니다.
     products_or_services 가 있으면 유사 사업명을 그 표현으로 잡습니다.
   - recommended_actions: 대응방향입니다. 제안서, PoC, 레퍼런스, 운영 모델 중
     무엇을 분리·설명·검증할지 구체적으로 씁니다. "제안서" 단독 표현은 금지하고,
     어떤 사업/고객/도입 프로젝트용 산출물인지 붙입니다.
     피어 신호 → 관련 판단 기준 → SK AX가 재구성할 산출물/운영 방식 순서가 보여야 합니다.
     타깃 피어의 특정 프로젝트에 SK AX가 직접 제안서를 준비하라는 식으로 쓰지 말고,
     유사 고객군/유사 사업에서 SK AX가 바꿔야 할 제안 산출물, PoC 기준,
     레퍼런스 구성, 운영 모델로 씁니다.
     문장 패턴은 "이 피어 신호는 [유사 고객군/유사 사업]에서 [관련 판단 기준]이
     비교될 수 있음을 보여줍니다. SK AX는 [산출물/운영 방식]을 [변경 방식]으로
     재구성할 필요가 있습니다."처럼 인과가 보이게 씁니다.
     IntegratedIssue 에 사업 범위, 기간, 금액/규모, 고객군/계약 상대방 중 2개 이상이 있으면
     recommended_actions 는 서로 다른 산출물 기준으로 2개까지 작성합니다. 억지로 3개를
     만들 필요는 없습니다.
     products_or_services 가 있으면 action 안에서도 그 사업명을 사용해 어떤 제안 산출물이나
     PoC 기준을 재구성하는지 명확히 씁니다.
     "준비합니다", "강화합니다", "경쟁력을 높입니다", "차별화된 기능을 강조합니다"로
     끝내지 않습니다. 무엇을 재구성하고, 무엇을 분리해 보여주며, 무엇을 검증 기준으로
     넣을지까지 써야 합니다.
     recommended_actions 안에서는 "강화", "경쟁력", "준비", "차별화" 단어를 쓰지 말고
     재구성, 분리, 검증 기준, 안착 조건, 운영 모델처럼 산출물 변화가 보이는 표현을 씁니다.
   - opportunities/threats는 확정 성과 예측이 아니라, SK AX가 활용하거나
     주의할 판단 기준으로 씁니다.

4. business_line_mapping
   - 입력 6 후보 name 중 실제 관련 있는 항목만 0~3개 선택합니다.

5. evidence
   - sourced_evidence_ids 와 used_fact_ids 는 입력에 존재하는 fact_id 만 사용합니다.
   - 출력 schema 의 예시 문구를 복사하지 마세요.

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
      "prompt_version": "strategic-insight-v1.39-report-specificity",
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
필드만 같은 schema 안에서 더 정확하게 고칩니다.
품질 게이트는 단순 사실 검증이 아니라, 임원 보고 문장으로서 "왜 이런 분석이
나왔는지", "왜 이런 대응이 필요한지"가 근거와 논리로 설명되는지도 판단합니다.
입력 근거만으로 품질을 회복할 수 없으면 그럴듯한 문장을 만들지 말고 invalid 로
낮춥니다. JSON 외 텍스트를 출력하지 마세요.
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
   단, 입력 근거와 ProfileContext 만으로 논리 공백을 메울 수 없으면
   is_valid_strategic_insight=false, analysis.is_valid_analysis=false,
   implication.is_valid_implication=false 로 낮추고 reason 에 부족한 근거를 적습니다.
1. 모든 수치, 날짜, 고객명, 제품명, 회사명, 사업명은 IntegratedIssue 또는
   ProfileContext 에 있는 표현만 사용해야 합니다.
   수치는 fact_basis/key_numbers/representative_sources에 있는 형태 그대로만 유지합니다.
   반올림·추정·요약 수치는 제거하거나 원문 수치로 되돌립니다.
2. IntegratedIssue 가 단순 수주, 선정, PoC, 가능성 검토 수준이면 시장 선점,
   점유율 확대, 매출 기여, 기술적 우위, 혁신성, 경쟁 심화처럼 우열이나 성과를
   단정하지 않습니다.
   단일 계약/수주 근거만으로 디지털 전환 가속화, 높은 신뢰, 장기 고객 관계,
   안정적 수익원처럼 결과를 확정하지 않습니다. 근거 수준에 맞춰 계약 범위,
   사업 연결성, 계약 기간, 검증 대상이 확인됐다고 낮춰 씁니다.
3. IntegratedIssue 의 관계 수준을 보존합니다. 투자/지분 취득을 협업/파트너십으로
   바꾸지 말고, 검토/구상/가능성 관계는 확정 실행처럼 쓰지 않습니다.
   계약/공급계약에서는 타깃 피어가 공급자인지 계약 상대방인지 구분합니다.
   타깃 피어의 정확한 역할이 불명확하면 고객/원청/도입 주체로 단정하지 말고
   계약 상대방, 사업 연결성, 계약 범위로 표현합니다.
   타깃 피어가 고객/계약 상대방으로 보이면 공급사 매출 비율은 타깃 피어의 성과가 아니라
   계약 규모 참고 근거입니다. 이를 피어사의 매출 영향, 역량 강화, 공급 역량으로 쓰면 실패입니다.
   "A사가 타깃 피어 B와 계약했다"는 구조에서는 A사의 매출 비율, 성장 가능성,
   시장 입지를 B사의 전략 의미로 쓰면 실패입니다. B사의 의미는 B가 연결된 사업 범위,
   계약 기간, 산업/고객군, B의 프로필 역량과의 접점으로만 설명합니다.
   타깃 피어가 계약 상대방/고객으로 보이는 경우, B사의 전략/프로젝트 추진/참여/제공 역량처럼
   쓰지 말고 B가 어떤 계약 범위·기간·산업 과제와 연결되어 관찰되는지로 씁니다.
4. market_signal 이 "고객 요구 변화", "중요해지고 있습니다" 같은 일반론이면
   고객명, 사업명, 검증 단계, 운영 대상, 관계 유형을 넣어 다시 작성합니다.
   통합 근거에 수요 증가/확대가 직접 없으면 수요 증가를 단정하지 말고 확인된
   수요 신호나 평가 항목으로 낮춰 씁니다.
   고객 기대 상승, 큰 영향, 적용 범위 확대도 직접 근거가 없으면 쓰지 않습니다.
   "초기 단계"도 IntegratedIssue 에 직접 근거가 없으면 쓰지 말고, 확인된 계약 범위나
   사업 연결성으로 낮춥니다.
   수요 증가/확대, 고객 기대 상승, 큰 영향, 적용 범위 확대도 직접 근거가 없으면
   수요 신호, 비교 기준, 확인된 계약 범위로 낮춥니다.
   cluster_fact_intelligence.products_or_services 에 사업/제품명이 있으면 "관련 프로젝트",
   "디지털 전환 흐름"보다 그 명칭을 우선 사용합니다.
5. analysis 는 피어사와 시장 의미만 다룹니다. SK AX 대응 문장은 implication 에만 둡니다.
6. peer_meaning 과 capability_change 는 짧은 결론으로 끝내지 말고,
   "근거 사실 → 피어사의 적용 범위/고객군/운영 방식/제안 메시지 변화"가 보이게 씁니다.
   peer_meaning 은 문장부호로 분리된 2문장 구조를 지킵니다.
   타깃 피어가 공급자가 아닌 계약 상대방이면 "역량 강화"보다 "해당 현대화 과제와 연결",
   "계약 범위/기간이 확인", "금융/산업별 IT 현대화 흐름 관찰"처럼 근거 수준에 맞춰 씁니다.
7. potential_impact 는 2문장으로 씁니다.
   - 1문장: 입력 근거 때문에 고객이 무엇을 더 비교하거나 평가하게 되는지
   - 2문장: 그래서 SK AX의 제안서, PoC, 레퍼런스, 운영 모델 중 무엇이 어떻게 바뀌어야 하는지
8. impact_reason 과 reason 이 같은 문장이면 실패입니다. impact_reason 은 영향도 판단,
   reason 은 사용한 fact/profile/context 근거와 해석 연결을 설명합니다.
9. opportunities, threats, recommended_actions 는 서로 다른 판단 포인트를 다룹니다.
   같은 꼬리 문장이나 같은 template 을 반복하지 않습니다.
   threats 에 경쟁 심화를 쓰려면 직접 비교 근거가 있어야 합니다. 비교 근거가 없으면
   threats 는 빈 배열이어도 됩니다.
10. recommended_actions 는 "강조/강화/재검토" 같은 추상 동사만으로 끝내지 말고,
   산출물 또는 실행 장면에서 무엇을 설명, 분리, 검증, 비교할지까지 씁니다.
   "제안서"는 단독으로 쓰지 말고, 어떤 고객/사업/도입 프로젝트를 위한 제안 산출물인지
   IntegratedIssue 의 사업명/고객군/제품명/관계 수준으로 붙여 씁니다.
   피어사의 특정 프로젝트에 SK AX가 직접 제안서를 준비하라는 문장은 실패입니다.
   유사 고객군/유사 사업에서 SK AX가 바꿀 제안 산출물, PoC 기준, 레퍼런스 구성,
   운영 모델을 씁니다.
   좋은 대응방향은 "피어 신호 → 관련 판단 기준 → SK AX가 재구성할 산출물/운영 방식"이
   2문장 안에서 보여야 합니다. 피어명은 근거 출처로만 쓰고, 대응 대상은 유사 고객군이나
   유사 사업으로 씁니다.
   "준비/강화/경쟁력/차별화"가 남은 대응방향은 실패입니다. 그런 표현은
   재구성할 산출물, 분리해 보여줄 기준, 검증할 조건으로 바꿉니다.
   recommended_actions 안에서는 "강화", "경쟁력", "준비", "차별화" 단어를 사용하지 않습니다.
11. 추상 표현이 구체 앵커 없이 남아 있으면 수정합니다. 구체 앵커는 고객명, 사업명,
   제품/서비스명, 적용 영역, 검증 단계, 관계 수준, 운영 책임, 프로필 역량입니다.
12. "두 가지 사업", "여러 기술", "이러한 융합"처럼 앞 문장을 다시 봐야 하는 압축 표현은
   실제 명칭이나 판단 기준으로 풀어 씁니다.
13. 통합 근거에 직접 없는 수요 증가/확대와 선호 비교 표현은 보수적으로 낮춥니다.
   "수요가 증가"는 "수요 신호가 확인", "더 중시"는 "함께 비교/확인" 수준으로 씁니다.
14. business_line_mapping 은 입력 후보 name 중에서만 고릅니다.
15. sourced_evidence_ids 와 used_fact_ids 는 입력에 존재하는 fact_id 만 사용합니다.
16. 각 주요 필드는 독자가 바로 납득할 수 있는 인과 연결이 있어야 합니다.
   - analysis: 현재 사실 → 피어사/시장 의미
   - peer_implication: 현재 사실 → 피어 프로필 역량/사업영역과의 연결
   - skax_implication: 피어 신호 → 관련 판단 기준 → SK AX 산출물/운영 방식 변화
   이 연결이 보이지 않는 문장은 좋은 문장으로 보지 말고 수정합니다.

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
        "prompt_version": "strategic-insight-v1.39-report-specificity",
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
0. 입력 4의 검증 실패 사유는 hard blocker 입니다. 최종 출력에 같은 위반 표현이 남으면
   valid 로 둘 수 없습니다. 위반 필드의 의미를 유지하되, 근거 수준에 맞는 보수 표현으로
   반드시 다시 씁니다.
   위반된 list 항목이 공급사 재무, 근거 없는 협력, 근거 없는 역량 강화만 말한다면
   억지로 살리지 말고 삭제하세요. 남은 항목이 1개여도 괜찮습니다.
1. 위반 사유가 있는 필드만 고칩니다. 위반이 없는 구체 문장과 fact_id 는 유지합니다.
   수치 위반이 있으면 fact_basis/key_numbers/representative_sources에 없는 수치 표현을
   삭제하거나 "근거에 언급된 규모/기간"처럼 비수치 표현으로 낮춥니다.
   role_interpretation_hints 가 있으면 반드시 따릅니다. 타깃 피어가 고객/계약 상대방으로
   보이는 계약에서는 공급사의 매출 비율을 타깃 피어의 성과로 쓰지 않습니다.
2. 시장 점유율, 선점, 기술적 우위, 혁신성, 경쟁 심화, 매출 기여 같은 표현은
   IntegratedIssue 에 직접 근거가 없으면 제거합니다.
   디지털 전환 가속화, 높은 신뢰, 장기 고객 관계, 안정적 수익원도 IntegratedIssue 에
   직접 근거가 없으면 계약 범위, 사업 연결성, 계약 기간, 검증 대상으로 낮춥니다.
   위반 사유에 특정 표현이 들어 있으면 해당 표현이 남아 있지 않게 삭제하거나
   "계약 범위 확인", "사업 연결성 확인", "평가 기준으로 참고"처럼 보수 표현으로 바꿉니다.
3. 투자/지분 취득, 수주, 선정, PoC, 검토 같은 관계 수준을 IntegratedIssue 와
   동일하게 맞춥니다. 근거에 없는 협업/파트너십으로 바꾸지 않고, 검토/구상은
   확정 실행으로 바꾸지 않습니다.
   계약/공급계약에서는 타깃 피어가 공급자인지 계약 상대방인지 구분합니다.
   타깃 피어의 정확한 역할이 불명확하면 고객/원청/도입 주체로 단정하지 말고
   계약 상대방, 사업 연결성, 계약 범위로 고칩니다.
   타깃 피어가 공급자가 아닌 계약 상대방이면 "제공 역량 강화", "공급 역량 확대"가 아니라
   "해당 사업과 연결된 계약 범위/기간이 확인"되는 수준으로 낮춥니다.
   타깃 피어의 "프로젝트", "전략", "기회 제공"으로 쓰지 말고, 통합 근거에서 확인된
   계약 상대방/사업 연결성/계약 범위/계약 기간과 ProfileContext 의 사업영역 접점으로
   다시 씁니다.
   공급사의 매출 비율이나 성장 가능성은 타깃 피어의 성과가 아닙니다. 필요한 경우
   계약 규모 참고 근거로만 쓰고, 피어 의미와 SK AX 대응은 사업 범위/기간/고객군/프로필 역량으로
   다시 연결합니다.
   analysis.reason 에서도 공급사 매출 비율을 "전략적 중요성" 근거로 쓰지 않습니다.
   계약 규모 참고가 필요하면 "계약 규모가 확인된다"까지만 쓰고, 피어 의미는
   계약 상대방으로 확인된 사업 범위/기간/프로필 접점으로 설명합니다.
   위반 사유에 "협업/파트너십"이 있으면 협력, 협업, 파트너십, 제휴, 공동, 기술적 협력이라는
   단어를 최종 출력에서 제거하고 계약 범위, 계약 기간, 사업 연결성, 과제 연결성으로 바꿉니다.
   위반 사유에 "공급사 재무/성장"이 있으면 공급사 매출 비율·성장 가능성·시장 입지는
   analysis, peer_implication, skax_implication 에서 제거하거나 계약 규모 참고 근거로만 낮춥니다.
   위반 사유에 "역량 강화/경쟁력 강화"가 있으면 타깃 피어의 성과로 쓰지 말고
   계약 상대방으로서 확인된 사업 범위와 관찰 지점으로 낮춥니다.
   위반 사유에 "피어 프로젝트 직접 제안"이 있으면 SK AX가 타깃 피어의 해당 프로젝트에
   직접 제안하는 문장을 삭제하고, 유사 고객군/유사 사업 대응 산출물로 바꿉니다.
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
   각 항목에는 왜 SK AX가 그 산출물을 바꿔야 하는지와 어떤 관련 부분이 중요해지는지가
   함께 들어가야 합니다.
   각 항목은 2문장으로 고칩니다. 첫 문장은 피어 신호와 해당 관련 부분,
   둘째 문장은 SK AX가 바꿀 산출물 또는 운영 방식입니다.
   첫 문장에서 피어명은 근거 출처로만 쓰고, 대응 대상은 유사 고객군/유사 사업으로
   바꿉니다. "타깃 피어 프로젝트에 제안서를 준비"하는 문장은 허용하지 않습니다.
   둘째 문장에는 적용 범위, 운영 책임, 검증 지표, 성과 기준, 비용/안정성,
   데이터 통제, 권한, SLA 중 입력 맥락에 맞는 판단 기준을 반드시 넣습니다.
   "성공 사례 포함", "고객 신뢰 구축", "사례 시연"처럼 보여주는 행위만 있으면
   제안서/PoC/레퍼런스/운영 모델에서 바꿀 평가 기준, 검증 지표, 운영 책임을 추가합니다.
   "강화해야 합니다", "집중해야 합니다", "준비해야 합니다"로 끝내지 말고
   "재구성할 필요가 있습니다", "분리해 제시해야 합니다", "검증 기준으로 넣어야 합니다"처럼
   산출물/운영 방식의 변경이 드러나게 씁니다.
   "경쟁력을 높일 필요", "차별화된 기능 강조", "제안서를 준비"처럼 결과나 행위만 말하면
   실패입니다. 왜 그 대응이 필요한지와 무엇을 재구성할지까지 다시 씁니다.
   위반 사유에 "대응방향이 준비/강화/경쟁력"이 있으면 해당 recommended_actions 는
   반드시 다시 씁니다. 같은 단어를 남긴 채 뒤에 검증 지표만 붙이지 마세요.
   이 경우 recommended_actions 에 "강화", "경쟁력", "준비", "차별화" 단어가 하나라도
   남으면 repair 실패입니다.
   제안서/PoC/레퍼런스/운영 모델 같은 산출물 단어가 있어도 왜 그 산출물을 바꿔야
   하는지 없으면 보완합니다.
   "제안서"만 단독으로 쓰면 어떤 고객/사업/도입 프로젝트의 제안 산출물인지
   IntegratedIssue 의 사업명/고객군/제품명/관계 수준으로 보완합니다.
   타깃 피어 이름과 "제안서 준비", "고객에게 제시"가 붙어 있으면 피어사 대상 영업처럼
   읽히므로, 유사 금융/제조/공공 고객군 등 입력 산업 맥락의 대응 산출물로 바꿉니다.
   products_or_services 에 사업명이 있으면 recommended_actions 에서도 그 사업명을 사용해
   어떤 제안 산출물/PoC/레퍼런스를 재구성할지 씁니다.
13. evidence id 와 business_line_mapping 은 입력 후보 안에서만 유지합니다.
14. 전체 결과를 "입력 근거", "관련 시장", "관련 적용 범위" 같은 일반 문장으로
   덮어쓰지 마세요. 기존 결과에 있던 고객명/사업명/검증 단계/협업 대상을 보존합니다.
15. 문장 전체 뒤에 "이 확인됩니다", "이 직접 근거입니다"를 붙이지 마세요.
   사실 문장은 자연스럽게 명사절로 바꾸거나, 해당 사실이 고객 평가 기준으로
   어떻게 연결되는지 설명합니다.

## 출력
StrategicInsightAgent 최종 schema 그대로 출력합니다.
"""


ACTION_REPAIR_SYSTEM_PROMPT = """\
당신은 SK AX 대응방향만 보수적으로 다시 쓰는 전략 action repair agent 입니다.
새 사실을 만들지 말고, 입력 근거와 기존 why_important / potential_impact 에서
도출되는 recommended_actions 만 JSON 으로 출력합니다.
"""


ACTION_REPAIR_USER_PROMPT_TEMPLATE = """\
## 입력 1 — IntegratedIssue
{integrated_issue_json}

## 입력 2 — ProfileContext
{profile_json}

## 입력 3 — 현재 skax_implication
{skax_json}

## 작성 규칙
1. recommended_actions 만 1~3개 작성합니다. 사업 범위, 계약 기간, 규모, 고객군 중
   2개 이상이 IntegratedIssue 에 있으면 서로 다른 산출물 기준으로 2개까지 작성합니다.
2. 각 action 은 2문장입니다.
   - 1문장: 피어 신호가 유사 고객군/유사 사업에서 어떤 판단 기준을 만들 수 있는지
   - 2문장: SK AX가 어떤 산출물/운영 방식을 어떻게 재구성해야 하는지
3. "강화", "경쟁력", "준비", "차별화" 단어를 쓰지 않습니다.
4. 타깃 피어의 특정 프로젝트에 SK AX가 직접 제안하는 문장으로 쓰지 않습니다.
   유사 고객군/유사 사업 대응으로 씁니다.
5. 사업명, 고객군, 계약 범위, 기간, 프로필 역량 중 입력에 있는 표현만 씁니다.
6. "고객 평가 기준 변화"라고만 쓰지 말고, 유사 고객군이 무엇을 비교/확인하게 되는지
   구체 기준을 써야 합니다.
7. products_or_services 에 사업명이 있으면 action 안에서도 그 사업명을 사용합니다.

## 출력
{{
  "recommended_actions": ["string"]
}}
"""


COUNTERPARTY_REPAIR_SYSTEM_PROMPT = """\
당신은 계약 상대방/고객 슬롯으로 등장한 피어사를 보수적으로 해석하는
전략 인사이트 repair agent 입니다. 현재 사건의 사실은 IntegratedIssue 에서만
사용하고, ProfileContext 는 기존 사업영역/역량 배경으로만 연결합니다.
JSON 외 텍스트를 출력하지 마세요.
"""


COUNTERPARTY_REPAIR_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## ProfileContext
{profile_json}

## business_line_mapping 후보
{business_lines_json}

## 기존 결과
{result_json}

## 다시 작성할 때 반드시 지킬 것
1. 타깃 피어는 공급자/수행사로 단정하지 않습니다. 계약 상대방, 사업 연결성,
   계약 범위, 계약 기간이 확인된 피어로만 설명합니다.
2. 공급사의 매출 비율과 성장 가능성은 타깃 피어의 성과나 역량 변화로 쓰지 않습니다.
   필요하면 계약 규모 참고 근거로만 씁니다.
3. products_or_services 의 사업명/제품명이 있으면 넓은 "디지털 전환 흐름"보다
   그 명칭을 우선 사용합니다.
4. "가속화", "경쟁 심화", "초기 단계", "기회를 제공", "고객 기대를 높임",
   "전략과 직접 연결", "프로젝트를 추진/참여/수행"은 IntegratedIssue 에 직접
   근거가 없으면 쓰지 않습니다.
5. analysis.reason 은 fact_id 와 profile 의 어떤 배경을 연결했는지 설명합니다.
   impact_reason 과 같은 문장을 반복하지 않습니다.
6. recommended_actions 는 1~2개입니다. 각 항목은 2문장으로 씁니다.
   첫 문장은 피어 신호가 유사 고객군/유사 사업에서 어떤 관련 기준을 만들 수 있는지,
   둘째 문장은 SK AX가 제안 산출물/PoC/레퍼런스/운영 모델 중 무엇을 어떻게
   재구성할 필요가 있는지 씁니다.
7. 타깃 피어의 특정 프로젝트에 SK AX가 직접 제안한다는 식으로 쓰지 않습니다.

## 출력 schema
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
            role_mode_instructions=_role_mode_instructions(integrated_issue),
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
            violations = _quality_gate_violations(
                result,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            if violations:
                guarded = _minimal_quality_guard(
                    result,
                    integrated_issue=integrated_issue,
                )
                remaining = _quality_gate_violations(
                    guarded,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                )
                if remaining:
                    return _mark_quality_gate_failed(guarded, remaining)
                return guarded
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
        current = result
        current_violations = violations
        try:
            for attempt in range(3):
                prompt = REPAIR_USER_PROMPT_TEMPLATE.format(
                    integrated_issue_json=_json_dumps(
                        _integrated_issue_for_prompt(integrated_issue)
                    ),
                    profile_json=_json_dumps(_profile_for_prompt(profile_context)),
                    business_lines_json=_json_dumps(
                        _business_line_candidate_details(profile_context)
                    ),
                    violations_json=_json_dumps(current_violations),
                    result_json=_json_dumps(current),
                )
                content = self._invoke_llm(
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    bundle_id=bundle_id,
                    phase=f"quality_repair_{attempt + 1}",
                )
                current = _parse_and_normalize(
                    content,
                    integrated_issue=integrated_issue,
                    classification=classification,
                    cluster_metadata={},
                    profile_context=profile_context,
                    analysis_context=analysis_context,
                    model=self.model,
                )
                current_violations = _quality_gate_violations(
                    current,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                )
                if not current_violations:
                    return current
            if _main_company_is_customer_or_buyer(integrated_issue):
                current = self._repair_counterparty_role_result(
                    current,
                    integrated_issue=integrated_issue,
                    classification=classification,
                    profile_context=profile_context,
                    analysis_context=analysis_context,
                    bundle_id=bundle_id,
                )
                current_violations = _quality_gate_violations(
                    current,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                )
                if not current_violations:
                    return current
            guarded = _minimal_quality_guard(current, integrated_issue=integrated_issue)
            if not _string_list(
                ((guarded.get("implication") or {}).get("skax_implication") or {}).get(
                    "recommended_actions"
                ),
                max_items=3,
            ):
                guarded = self._repair_missing_recommended_actions(
                    guarded,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    bundle_id=bundle_id,
                )
            final_remaining = _quality_gate_violations(
                guarded,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            if final_remaining:
                return _mark_quality_gate_failed(guarded, final_remaining)
            return guarded
        except Exception as exc:  # noqa: BLE001 - fail closed instead of passing risky copy.
            log.warning(
                "StrategicInsightAgent quality repair failed | bundle=%s error=%s",
                bundle_id,
                exc,
            )
            guarded = _minimal_quality_guard(
                result,
                integrated_issue=integrated_issue,
            )
            final_remaining = _quality_gate_violations(
                guarded,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            if final_remaining:
                return _mark_quality_gate_failed(guarded, final_remaining)
            return guarded

    def _repair_counterparty_role_result(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
    ) -> dict[str, Any]:
        prompt = COUNTERPARTY_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(_profile_for_prompt(profile_context)),
            business_lines_json=_json_dumps(_business_line_candidate_details(profile_context)),
            result_json=_json_dumps(result),
        )
        content = self._invoke_llm(
            system_prompt=COUNTERPARTY_REPAIR_SYSTEM_PROMPT,
            user_prompt=prompt,
            bundle_id=bundle_id,
            phase="quality_repair_counterparty_role",
        )
        return _parse_and_normalize(
            content,
            integrated_issue=integrated_issue,
            classification=classification,
            cluster_metadata={},
            profile_context=profile_context,
            analysis_context=analysis_context,
            model=self.model,
        )

    def _repair_missing_recommended_actions(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        profile_context: dict[str, Any],
        bundle_id: str,
    ) -> dict[str, Any]:
        out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        implication = out.get("implication") or {}
        skax = implication.get("skax_implication") or {}
        prompt = ACTION_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(_profile_for_prompt(profile_context)),
            skax_json=_json_dumps(skax),
        )
        content = self._invoke_llm(
            system_prompt=ACTION_REPAIR_SYSTEM_PROMPT,
            user_prompt=prompt,
            bundle_id=bundle_id,
            phase="quality_repair_actions",
        )
        data = _json_dict(_parse_json_loose(content))
        actions = [
            action
            for index, action in enumerate(
                _string_list(data.get("recommended_actions"), max_items=3), start=1
            )
            if not _recommended_action_quality_violation(
                action,
                label=f"skax_implication.recommended_actions[{index}]",
            )
        ]
        if actions:
            skax["recommended_actions"] = actions
            implication["skax_implication"] = skax
            out["implication"] = implication
        return out

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
    return normalized


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


def _fact_basis_from_evidence_refs(rows: Sequence[Any]) -> list[dict[str, Any]]:
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


def _consolidated_facts_from_evidence_refs(rows: Sequence[Any]) -> list[dict[str, Any]]:
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
        "cluster_fact_intelligence": _cluster_fact_intelligence_for_prompt(
            integrated_issue.get("cluster_fact_intelligence") or {}
        ),
        "role_interpretation_hints": _role_interpretation_hints(integrated_issue),
        "missing_or_uncertain_points": integrated_issue.get("missing_or_uncertain_points", []),
        "confidence": integrated_issue.get("confidence", 0.0),
    }


def _cluster_fact_intelligence_for_prompt(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "activity_types": _string_list(value.get("activity_types"), max_items=10),
        "customers_or_industries": _string_list(value.get("customers_or_industries"), max_items=10),
        "products_or_services": _string_list(value.get("products_or_services"), max_items=10),
        "numbers_and_dates": _string_list(value.get("numbers_and_dates"), max_items=10),
        "unique_facts": [
            {
                "fact": str(item.get("fact") or "").strip(),
                "activity_types": _string_list(item.get("activity_types"), max_items=5),
                "customers_or_industries": _string_list(
                    item.get("customers_or_industries"), max_items=5
                ),
                "products_or_services": _string_list(item.get("products_or_services"), max_items=5),
                "summary_role": str(item.get("summary_role") or "").strip(),
                "fact_type": str(item.get("fact_type") or "").strip(),
            }
            for item in (value.get("unique_facts") or [])[:8]
            if isinstance(item, dict)
        ],
        "uncertain_facts": [
            fact_text
            for item in (value.get("uncertain_facts") or [])[:5]
            if (fact_text := _fact_like_text(item))
        ],
    }


def _fact_like_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("fact") or value.get("summary") or "").strip()
    return str(value or "").strip()


def _role_interpretation_hints(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    main_company = str(integrated_issue.get("main_company") or "").strip()
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    activity_types = _string_list(intelligence.get("activity_types"), max_items=10)
    customers_or_industries = _string_list(
        intelligence.get("customers_or_industries"), max_items=20
    )
    products_or_services = _string_list(intelligence.get("products_or_services"), max_items=20)
    main_tokens = _company_token_variants(main_company)
    target_in_customer_slot = any(
        _normalize_entity_token(item) in main_tokens for item in customers_or_industries
    )
    contract_like = any(
        str(activity or "").strip().casefold() in {"contract", "order", "supply_contract"}
        for activity in activity_types
    ) or bool(re.search(r"계약|수주|공급계약", _integrated_grounding_text(integrated_issue)))

    guidance: list[str] = []
    if target_in_customer_slot and contract_like:
        guidance.append(
            "target_peer_appears_as_contract_counterparty_or_customer; "
            "do_not_treat_supplier_revenue_ratio_as_target_peer_performance"
        )
        guidance.append(
            "if target role is unclear, describe business connection/contract scope rather than "
            "supplier capability or procurement ownership"
        )
    elif contract_like:
        guidance.append(
            "contract_like_event; preserve supplier/counterparty role from evidence and avoid "
            "unstated customer/procurement assumptions"
        )

    return {
        "target_company": main_company,
        "activity_types": activity_types,
        "target_in_customers_or_industries": target_in_customer_slot,
        "customers_or_industries": customers_or_industries,
        "products_or_services": products_or_services,
        "guidance": guidance,
    }


def _role_mode_instructions(integrated_issue: dict[str, Any]) -> str:
    if not _main_company_is_customer_or_buyer(integrated_issue):
        return "일반 모드: IntegratedIssue 의 관계 수준을 그대로 보존합니다."
    hints = _role_interpretation_hints(integrated_issue)
    suppliers = _supplier_names_for_target_counterparty(integrated_issue)
    products = _string_list(
        (hints.get("products_or_services") if isinstance(hints, dict) else None),
        max_items=5,
    )
    return "\n".join(
        [
            "계약 상대방/고객 슬롯 모드입니다.",
            "- 타깃 피어가 customers_or_industries 슬롯에 있고 "
            "공급사/계약 체결 주체가 따로 보입니다.",
            f"- 추출된 공급사 후보: {', '.join(suppliers) if suppliers else '없음'}",
            f"- 확인된 사업/제품 후보: {', '.join(products) if products else '없음'}",
            "- analysis 와 peer_implication 에서 타깃 피어를 "
            "프로젝트 추진/참여/제공/공급 주체처럼 쓰지 마세요.",
            "- 타깃 피어는 계약 상대방, 사업 연결성, 계약 범위/기간이 확인된 피어로만 설명하세요.",
            "- 공급사 매출 비율은 요약의 계약 규모 근거일 뿐, "
            "타깃 피어의 역량/성과/전략 근거가 아닙니다.",
            "- SK AX 대응방향은 타깃 피어 프로젝트에 직접 제안하는 문장이 아니라 "
            "유사 고객군/유사 사업의 산출물 재구성으로 쓰세요.",
        ]
    )


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
    integrated_evidence_text = _integrated_grounding_text(integrated_issue)
    grounded_numeric_keys = _grounded_numeric_keys_for_issue(integrated_issue)
    violations: list[str] = []
    for label, value_text in _quality_checked_texts(result):
        if "quality_gate_failed:" in value_text:
            value_text = value_text.split("| quality_gate_failed:", 1)[0].strip()
        for token in _NUMERIC_TOKEN_PATTERN.findall(value_text):
            token_text = str(token or "").strip()
            if token_text and _numeric_token_key(token_text) not in grounded_numeric_keys:
                violations.append(
                    f"{label}: fact_basis/key_numbers/representative_sources에 없는 "
                    f"수치 `{token_text}`를 사용했습니다."
                )
        for pattern in _UNSUPPORTED_CLAIM_PATTERNS:
            if _has_unsupported_pattern(
                value_text,
                pattern,
                evidence_text=integrated_evidence_text,
            ):
                violations.append(f"{label}: 입력 근거 없이 `{pattern}` 계열 표현을 사용했습니다.")
        relation_violation = _relationship_grounding_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            integrated_evidence_text=integrated_evidence_text,
        )
        if relation_violation:
            violations.append(f"{label}: {relation_violation}")
        supplier_role_violation = _supplier_role_overstatement_violation(
            value_text,
            integrated_issue=integrated_issue,
        )
        if supplier_role_violation:
            violations.append(f"{label}: {supplier_role_violation}")
        supplier_financial_focus_violation = _supplier_financial_focus_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if supplier_financial_focus_violation:
            violations.append(f"{label}: {supplier_financial_focus_violation}")
        counterparty_overclaim_violation = _counterparty_capability_overclaim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if counterparty_overclaim_violation:
            violations.append(f"{label}: {counterparty_overclaim_violation}")
        counterparty_role_violation = _counterparty_role_action_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if counterparty_role_violation:
            violations.append(f"{label}: {counterparty_role_violation}")
        action_quality_violation = _recommended_action_quality_violation(value_text, label=label)
        if action_quality_violation:
            violations.append(f"{label}: {action_quality_violation}")

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


def _relationship_grounding_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    integrated_evidence_text: str,
) -> str:
    if not text or not _RELATIONSHIP_PATTERN.search(text):
        return ""
    if _is_follow_up_or_watch_field(label):
        return ""
    if not _has_relationship_grounding(integrated_issue, integrated_evidence_text):
        return "IntegratedIssue 에 없는 협업/파트너십 계열 관계 표현을 사용했습니다."
    if _relationship_only_uncertain(integrated_evidence_text) and not _UNCERTAINTY_PATTERN.search(
        text
    ):
        return "검토/구상/가능성 단계의 관계를 확정 실행처럼 표현했습니다."
    return ""


def _is_follow_up_or_watch_field(label: str) -> bool:
    return label.startswith(("implication.follow_up_questions", "implication.watch_points"))


def _has_relationship_grounding(integrated_issue: dict[str, Any], evidence_text: str) -> bool:
    if _RELATIONSHIP_PATTERN.search(evidence_text):
        return True
    event_type = str(integrated_issue.get("cluster_event_type") or "").strip().casefold()
    if event_type in _RELATIONSHIP_ACTIVITY_TYPES:
        return True
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    activity_types = _string_list(intelligence.get("activity_types"), max_items=20)
    return any(
        str(activity).strip().casefold() in _RELATIONSHIP_ACTIVITY_TYPES
        for activity in activity_types
    )


def _relationship_only_uncertain(text: str) -> bool:
    relation_sentences = [
        sentence for sentence in _split_sentences(text) if _RELATIONSHIP_PATTERN.search(sentence)
    ]
    return bool(relation_sentences) and all(
        _UNCERTAINTY_PATTERN.search(sentence) for sentence in relation_sentences
    )


def _supplier_role_overstatement_violation(
    text: str,
    *,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not _SUPPLIER_CAPABILITY_PATTERN.search(text):
        return ""
    if not _main_company_is_customer_or_buyer(integrated_issue):
        return ""
    return "타깃 피어가 계약의 고객/도입/조달 주체로 보이는데 공급자 역량처럼 표현했습니다."


def _supplier_financial_focus_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not _main_company_is_customer_or_buyer(integrated_issue):
        return ""
    if label.startswith(("implication.follow_up_questions", "implication.watch_points")):
        return ""
    supplier_names = _supplier_names_for_target_counterparty(integrated_issue)
    if not supplier_names:
        return ""
    supplier_mentioned = any(
        re.search(re.escape(name), text, flags=re.IGNORECASE) for name in supplier_names
    )
    if not supplier_mentioned:
        return ""
    if not re.search(r"매출|성장|시장\s*입지|중요한\s*매출원|성과|시장\s*반응", text):
        return ""
    return (
        "타깃 피어가 계약 상대방으로 보이는데 공급사 재무/성장 논리를 "
        "피어 전략 의미처럼 사용했습니다."
    )


def _counterparty_capability_overclaim_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not _main_company_is_customer_or_buyer(integrated_issue):
        return ""
    if label.startswith(("implication.follow_up_questions", "implication.watch_points")):
        return ""
    overclaim_pattern = (
        r"역량[이가을를\s]*(강화|확장)|"
        r"경쟁력[이가을를\s]*강화|"
        r"입지[가를\s]*강화"
    )
    if not re.search(overclaim_pattern, text):
        return ""
    return (
        "타깃 피어가 계약 상대방으로 보이는 계약을 역량 강화/경쟁력 강화 성과처럼 "
        "단정했습니다. 계약 범위/사업 연결성 수준으로 낮춰야 합니다."
    )


def _counterparty_role_action_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not _main_company_is_customer_or_buyer(integrated_issue):
        return ""
    if label.startswith(("implication.follow_up_questions", "implication.watch_points")):
        return ""

    target_patterns = _target_name_patterns(str(integrated_issue.get("main_company") or ""))
    mentions_target = any(
        re.search(pattern, text, flags=re.IGNORECASE) for pattern in target_patterns
    )
    if not mentions_target:
        return ""

    direct_role_patterns = (
        r"전략(적)?\s*(방향|움직임|일관성|일환|추진|구체화|강화)",
        r"프로젝트[가를은\s]*(추진|참여|제공|공급|수행|구축|운영)",
        r"사업[가를은\s]*(추진|참여|제공|공급|수행|구축|운영)",
        r"(솔루션|서비스|기회)[이가를은\s]*.{0,20}제공",
        r"(제공|공급|수행|구축|운영)\s*역량",
    )
    if any(re.search(pattern, text) for pattern in direct_role_patterns):
        return (
            "타깃 피어가 계약 상대방/고객 슬롯에 있는데 피어의 프로젝트 실행이나 "
            "공급자 행동처럼 썼습니다. 계약 범위/사업 연결성/관찰 지점으로 낮춰야 합니다."
        )

    conservative_role_terms = (
        r"계약\s*상대방|계약\s*범위|계약\s*기간|사업\s*연결|"
        r"과제와\s*연결|연결성|연결|확인|관찰|참고\s*근거"
    )
    if re.search(conservative_role_terms, text):
        return ""

    if label.startswith("skax_implication.recommended_actions") and re.search(
        r"제안서|PoC|레퍼런스|운영\s*모델", text
    ):
        if re.search(r"SK\s*AX|유사\s*고객군|유사\s*사업", text, flags=re.IGNORECASE) and not any(
            re.search(
                pattern + r".{0,24}(에게|대상|상대로|제안|제시|영업)",
                text,
                flags=re.IGNORECASE,
            )
            for pattern in target_patterns
        ):
            return ""
        return (
            "SK AX 대응을 타깃 피어의 특정 프로젝트에 직접 제안하는 것처럼 썼습니다. "
            "유사 고객군/유사 사업 대응 산출물로 바꿔야 합니다."
        )
    return ""


def _recommended_action_quality_violation(text: str, *, label: str) -> str:
    if not text or not label.startswith("skax_implication.recommended_actions"):
        return ""
    vague_action_pattern = (
        r"강화|"
        r"제안서.{0,16}강화|"
        r"PoC.{0,16}강화|"
        r"기준.{0,16}강화|"
        r"경쟁력[을를이가\s]*(유지|높|강화)|"
        r"차별화된\s*기능|"
        r"제안서.{0,16}준비"
    )
    if re.search(vague_action_pattern, text):
        return (
            "대응방향이 준비/강화/경쟁력 같은 추상 표현에 머물렀습니다. "
            "피어 신호, 관련 판단 기준, SK AX가 재구성할 산출물/운영 방식을 함께 써야 합니다."
        )
    return ""


def _supplier_names_for_target_counterparty(integrated_issue: dict[str, Any]) -> list[str]:
    main_company = str(integrated_issue.get("main_company") or "").strip()
    if not main_company:
        return []
    target_patterns = _target_name_patterns(main_company)
    if not target_patterns:
        return []
    evidence_text = _integrated_grounding_text(integrated_issue)
    suppliers: list[str] = []
    for target_pattern in target_patterns:
        patterns = (
            re.compile(
                rf"([A-Za-z가-힣0-9&㈜\.·_-]{{2,30}})(?:가|이|는|은)\s+"
                rf"[^.。!?\n]{{0,50}}?{target_pattern}\s*(?:와|과|하고)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                rf"([A-Za-z가-힣0-9&㈜\.·_-]{{2,30}})\s*(?:와|과)\s*{target_pattern}",
                flags=re.IGNORECASE,
            ),
        )
        for pattern in patterns:
            for match in pattern.finditer(evidence_text):
                supplier = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:·-")
                if supplier and _normalize_entity_token(supplier) not in _company_token_variants(
                    main_company
                ):
                    suppliers.append(supplier)
    return list(dict.fromkeys(suppliers))[:5]


def _target_name_patterns(company_id: str) -> list[str]:
    variants = expand_peer_aliases(company_id)
    normalized_seen: set[str] = set()
    patterns: list[str] = []
    for variant in variants:
        text = str(variant or "").strip()
        normalized = _normalize_entity_token(text)
        if not text or normalized in normalized_seen:
            continue
        normalized_seen.add(normalized)
        escaped = re.escape(text)
        patterns.append(escaped.replace(r"\ ", r"\s*").replace("_", r"[_\s]*"))
    return patterns


def _main_company_is_customer_or_buyer(integrated_issue: dict[str, Any]) -> bool:
    main_company = str(integrated_issue.get("main_company") or "").strip()
    if not main_company:
        return False
    evidence_text = _integrated_grounding_text(integrated_issue)
    if not _SUPPLY_CONTRACT_PATTERN.search(evidence_text):
        return False

    main_tokens = _company_token_variants(main_company)
    for item in _iter_dicts(integrated_issue):
        for key in ("customers_or_industries", "customers", "customer", "clients", "client"):
            if key not in item:
                continue
            for value in _jsonish_list(item.get(key)):
                if _normalize_entity_token(value) in main_tokens:
                    return True
    return False


def _company_token_variants(company: str) -> set[str]:
    raw = str(company or "").strip()
    if not raw:
        return set()
    variants = {
        raw,
        raw.replace("_", " "),
        raw.replace("_", ""),
        raw.upper(),
        raw.replace("_", " ").upper(),
    }
    return {_normalize_entity_token(value) for value in variants if value}


def _normalize_entity_token(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(value or "")).casefold()


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        result.append(value)
        for child in value.values():
            result.extend(_iter_dicts(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_iter_dicts(child))
    return result


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
    intelligence = integrated_issue.get("cluster_fact_intelligence")
    if intelligence:
        parts.append(_json_dumps(_cluster_fact_intelligence_for_prompt(intelligence)))
    if profile_context:
        parts.append(_json_dumps(_profile_for_prompt(profile_context)))
    return "\n".join(parts)


def _grounded_numeric_keys_for_issue(integrated_issue: dict[str, Any]) -> set[str]:
    chunks: list[str] = []
    for key in ("fact_basis", "key_numbers", "representative_sources"):
        value = integrated_issue.get(key)
        if value:
            chunks.append(_json_dumps(value))
    grounded = " | ".join(chunks)
    return {
        key
        for match in _NUMERIC_TOKEN_PATTERN.finditer(grounded)
        if (key := _numeric_token_key(match.group(0)))
    }


def _numeric_token_key(token: str) -> str:
    text = re.sub(r"\s+", "", str(token or "").strip().lower())
    if not text:
        return ""
    unit = ""
    for candidate in ("억원", "억", "조원", "조", "만원", "만", "천만", "백만", "%", "원"):
        if text.endswith(candidate):
            unit = candidate
            text = text[: -len(candidate)]
            break
    if unit == "억원":
        unit = "억"
    elif unit == "조원":
        unit = "조"
    number_text = text.replace(",", "")
    try:
        number = float(number_text)
    except ValueError:
        normalized_number = number_text
    else:
        normalized_number = str(int(number)) if number.is_integer() else f"{number:.6f}".rstrip("0")
    return f"{normalized_number}{unit}"


def _integrated_grounding_text(integrated_issue: dict[str, Any]) -> str:
    return _grounding_text(integrated_issue=integrated_issue, profile_context=None)


def _sentence_count(text: str) -> int:
    return len(_split_sentences(text))


def _split_sentences(text: str) -> list[str]:
    return [item for item in re.split(r"[.!?。]\s*", str(text or "").strip()) if item.strip()]


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
    """Backward-compatible alias for old tests; no longer writes template copy."""
    return _minimal_quality_guard(result, integrated_issue=integrated_issue)


def _minimal_quality_guard(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    """Apply only mechanical safety fixes, never generate strategic copy.

    LLM self-review owns content repair. This guard only removes unsupported
    numeric drift and contract-role overstatement that can be detected safely.
    """
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    analysis = out.get("analysis") or {}
    implication = out.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}

    if _main_company_is_customer_or_buyer(integrated_issue):
        for key in ("analysis_summary", "market_signal", "impact_reason", "reason"):
            if analysis.get(key):
                analysis[key] = _repair_customer_role_overstatement(str(analysis[key]))
        analysis["strategic_meaning"] = [
            _repair_customer_role_overstatement(item)
            for item in _string_list(analysis.get("strategic_meaning"), max_items=3)
        ]
        for key in ("peer_meaning", "capability_change"):
            if peer.get(key):
                peer[key] = _repair_customer_role_overstatement(str(peer[key]))
        for key in ("why_important", "potential_impact"):
            if skax.get(key):
                skax[key] = _repair_customer_role_overstatement(str(skax[key]))

    _repair_result_numeric_grounding(
        analysis=analysis,
        implication=implication,
        integrated_issue=integrated_issue,
    )
    skax["recommended_actions"] = [
        action
        for index, action in enumerate(
            _string_list(skax.get("recommended_actions"), max_items=3), start=1
        )
        if not _recommended_action_quality_violation(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
        )
        and not _counterparty_role_action_violation(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
        )
    ]

    out["analysis"] = analysis
    implication["peer_implication"] = peer
    implication["skax_implication"] = skax
    out["implication"] = implication
    return out


def _repair_result_numeric_grounding(
    *,
    analysis: dict[str, Any],
    implication: dict[str, Any],
    integrated_issue: dict[str, Any],
) -> None:
    for key in ("analysis_summary", "market_signal", "impact_reason", "reason"):
        if analysis.get(key):
            analysis[key] = _remove_ungrounded_numeric_tokens(
                str(analysis[key]),
                integrated_issue=integrated_issue,
            )
    analysis["strategic_meaning"] = [
        _remove_ungrounded_numeric_tokens(item, integrated_issue=integrated_issue)
        for item in _string_list(analysis.get("strategic_meaning"), max_items=3)
    ]

    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    for key in ("peer_meaning", "capability_change"):
        if peer.get(key):
            peer[key] = _remove_ungrounded_numeric_tokens(
                str(peer[key]),
                integrated_issue=integrated_issue,
            )
    for key in ("why_important", "potential_impact"):
        if skax.get(key):
            skax[key] = _remove_ungrounded_numeric_tokens(
                str(skax[key]),
                integrated_issue=integrated_issue,
            )
    for key in ("opportunities", "threats", "recommended_actions"):
        skax[key] = [
            _remove_ungrounded_numeric_tokens(item, integrated_issue=integrated_issue)
            for item in _string_list(skax.get(key), max_items=3)
        ]
    implication["follow_up_questions"] = [
        _remove_ungrounded_numeric_tokens(item, integrated_issue=integrated_issue)
        for item in _string_list(implication.get("follow_up_questions"), max_items=3)
    ]
    implication["watch_points"] = [
        _remove_ungrounded_numeric_tokens(item, integrated_issue=integrated_issue)
        for item in _string_list(implication.get("watch_points"), max_items=3)
    ]


def _remove_ungrounded_numeric_tokens(
    text: str,
    *,
    integrated_issue: dict[str, Any],
) -> str:
    grounded_keys = _grounded_numeric_keys_for_issue(integrated_issue)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0).strip()
        if _numeric_token_key(token) in grounded_keys:
            return token
        return "근거에 언급된 수치"

    out = _NUMERIC_TOKEN_PATTERN.sub(replace, str(text or ""))
    out = re.sub(r"근거에 언급된 수치\s*%?\s*(이상|내외|가량|정도)", "근거에 언급된 규모", out)
    out = re.sub(r"근거에 언급된 수치\s*이상의\s*규모", "근거에 언급된 규모", out)
    out = re.sub(r"근거에 언급된 수치\s*규모", "근거에 언급된 규모", out)
    return out


def _repair_customer_role_overstatement(text: str) -> str:
    sentence = str(text or "").strip()
    replacements = (
        (r"([가-힣A-Za-z0-9&·+_\-\s]+?)의\s*공급\s*역량", r"\1의 계약 연결성과 사업 범위"),
        (r"([가-힣A-Za-z0-9&·+_\-\s]+?)\s*공급\s*역량", r"\1 계약 연결성과 사업 범위"),
        (r"전략적\s*방향과\s*일치", "프로필상 사업영역과 연결"),
        (r"전략과의\s*일관성", "프로필상 사업영역과의 접점"),
        (
            r"프로젝트[가은]\s*유사한\s*고객군과\s*사업\s*영역에서의\s*기회를\s*제공합니다",
            "계약 신호는 유사 고객군과 사업 영역에서 참고할 사업 연결성을 보여줍니다",
        ),
        (r"기회를\s*제공하는\s*것", "참고 근거가 되는 것"),
        (r"기회를\s*제공하는\s*것으로", "참고 근거로"),
        (r"기회를\s*제공할\s*수\s*있습니다", "참고 근거가 될 수 있습니다"),
        (r"기회를\s*제공합니다", "참고 근거가 됩니다"),
        (r"프로젝트에\s*참여하여", "프로젝트와 연결되어"),
        (r"프로젝트에\s*참여", "프로젝트와 연결"),
        (r"사업에\s*참여하여", "사업과 연결되어"),
        (r"사업에\s*참여", "사업과 연결"),
        (r"기여하고\s*있습니다", "연결성을 보여줍니다"),
        (r"공급\s*역량", "계약 연결성과 사업 범위"),
        (r"납품\s*역량", "계약 연결성과 사업 범위"),
        (r"도입[·\s-]*조달\s*주체", "계약 상대방"),
        (r"도입[·\s-]*조달", "계약"),
    )
    for pattern, replacement in replacements:
        sentence = re.sub(pattern, replacement, sentence)
    return sentence


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
