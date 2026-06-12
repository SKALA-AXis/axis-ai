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
_PROMPT_VERSION = "strategic-insight-v1.62-evidence-flow"
_LLM_TEMPERATURE = 0.0
_LLM_MAX_COMPLETION_TOKENS = 2600

_IMPACT_LEVELS = {"high", "medium", "low"}
_RISK_OR_OPPORTUNITY = {"risk", "opportunity", "neutral"}
_EVIDENCE_LABELS = {"sufficient", "moderate", "insufficient"}
# 근거 없이 쓰면 사실 왜곡이 큰 고위험 주장만 최소 차단한다.
# 표현 품질은 아래 구조 게이트와 프롬프트가 담당하고, 문구 blacklist 를 늘리지 않는다.
_UNSUPPORTED_CLAIM_PATTERNS = (
    r"시장\s*점유율\s*확대",
    r"시장\s*점유율[을를\s]*(확보|높|늘)",
    r"점유율[이을가\s]*(확대|상승|증가)",
    r"시장\s*선점",
    r"선점",
    r"기술적\s*우위",
    r"격차",
    r"리더십\s*확보",
    r"매출\s*기여",
    r"큰\s*영향",
    r"수요.{0,20}지속적.{0,20}(증가|확대)",
    r"수요[가를은\s]*(증가|확대)",
    r"수요[가를은\s]*(반영|있)",
    r"수요.{0,20}촉진",
    r"지속적.{0,20}중요",
    r"고객.{0,12}기대[가를을\s]*(높|상승)",
    r"시장\s*점유율\s*감소",
    r"점유율[이을가\s]*(감소|하락|축소)",
    r"경쟁\s*심화",
    r"앞서\s*나갈",
    r"경쟁력[을를이가\s]*(강화|높|제고)",
    r"기술적\s*역량[을를이가\s]*(강화|검증|입증)",
    r"역량[을를이가\s]*(강화)",
    r"역량[을를이가\s]*(검증|입증)",
    r"공공\s*및\s*민간.{0,20}(확대|활용)",
    r"활용\s*확대",
    r"기술적\s*신뢰성",
    r"차별화",
    r"효율성[을를\s]*평가",
    r"효율성[을를\s]*입증",
    r"고객.{0,10}신뢰[를을\s]*(확보|구축)",
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
_SUPPLY_CONTRACT_PATTERN = re.compile(r"공급\s*계약|공급계약|납품|구매|조달|계약\s*체결|계약")
_NUMERIC_TOKEN_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|usd|krw)?",
    re.IGNORECASE,
)
_GENERIC_INSIGHT_PATTERN = re.compile(
    r"디지털\s*전환을\s*지원|"
    r"업무\s*효율성\s*향상|"
    r"업무\s*효율성[을를]?\s*(높|향상|개선)|"
    r"기술\s*발전\s*동향|"
    r"사업\s*전략에\s*반영|"
    r"적용할\s*수\s*있는\s*부분을\s*탐색|"
    r"도입\s*가능성[을를]?\s*탐색|"
    r"적용\s*가능한\s*요소|"
    r"기술\s*개발\s*및\s*적용\s*가능성|"
    r"지속적으로\s*추적|"
    r"영향[을를]?\s*추적|"
    r"역량[을를이가\s]*강화|"
    r"중요성[이을가\s]*부각|"
    r"혁신적인\s*솔루션|"
    r"혁신\s*서비스",
    re.IGNORECASE,
)


ACTION_REPAIR_SYSTEM_PROMPT = """\
당신은 SK AX 관점의 비교 인사이트만 다시 쓰는 repair agent입니다.
새 사실을 만들지 말고 recommended_actions 만 JSON 으로 출력합니다.
각 action 은 실행계획이 아니라 SK AX가 유사 동향을 해석할 때 사용할 비교 인사이트입니다.
현재 사건의 대상 사업/서비스/시스템, 피어 신호, 비교해야 할 축,
후속 확인 데이터가 함께 보여야 합니다. SK AX 프로필 접점이 없으면 특정 역량과
연결하지 말고 사건 기반 비교 인사이트로 낮춥니다.
"""


ACTION_REPAIR_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## IssueFrame
{issue_frame_json}

## ComparableContext
{comparable_context_json}

## ProfileContext
{profile_json}

## 현재 skax_implication
{skax_json}

## recommended_actions 작성 규칙
recommended_actions 는 실행방안이 아니라 SK AX 관점의 비교 인사이트입니다.

1. recommended_actions 는 가능하면 2~3개입니다. 근거가 부족할 때만 1개로 줄입니다.
2. 각 action 은 2~3문장입니다:
   현재 사건에서 확인된 사업 장면 → 해당 기업이 이 장면에서 보여준 방식 →
   SK AX와 비교해야 할 사업 범위 또는 운영 역할 → 그 비교가 필요한 이유 →
   후속으로 확인할 데이터.
3. 고객에게 제안하는 문장이 아니라 SK AX 내부 직원/임원이 보는 비교 인사이트 문장입니다.
4. SK AX 프로필과 연결되지 않으면 특정 역량/사업영역 대응으로 쓰지 말고,
   현재 사건 근거 기반의 비교 인사이트로 낮춥니다.
5. 특정 기술명·사업영역·솔루션명은 현재 입력 컨텍스트에 근거가 있을 때만 씁니다.
   근거 출처는 IntegratedIssue, classification, 관련 peer/skax ProfileContext,
   현재 이슈와 매칭된 AnalysisContext 입니다.
   근거가 약하면 현재 사건의 대상 사업·서비스·시스템·수치·일정 같은
   더 상위의 안전한 표현으로 낮춥니다.
6. business_line_mapping 이름을 솔루션명처럼 그대로 쓰지 않습니다.
   예: business line 이 "클라우드&AI"여도 현재 사건과 직접 연결되지 않으면
   "클라우드 및 AI 솔루션"이라고 쓰지 않습니다.
7. "성공 사례"는 ProfileContext 에 실제 사례 근거가 있을 때만 씁니다.
8. 근거 없는 강화, 경쟁력, 차별화, 고객 신뢰 같은 결과 표현으로 끝내지 않습니다.
9. "모니터링", "추적", "전략 반영", "부분 탐색"만으로 끝내지 말고,
   무엇을 왜 봐야 하는지 현재 사건의 적용 업무·서비스·수치·관계로 설명합니다.
10. "대응 가능한지 내부적으로 구분"처럼 추상적으로 끝내지 말고,
   어떤 역량이 어떤 사업 장면과 비교되는지 설명합니다.
11. 각 action 은 "왜 이 비교가 필요한지"를 문장 안에 포함해야 합니다.
   예: 이 기준이 사업화 가능성, 운영 책임, 성과 검증, 역할 분담 중
   무엇을 판단하게 해주는지 설명합니다.
12. "같은 기준으로 축적", "비교해야 합니다", "모니터링해야 합니다"로 끝내지 않습니다.
   현재 사건의 어떤 사실이 그 기준을 만들었는지, 그 기준이 SK AX의 어떤 판단을
   바꾸게 하는지까지 씁니다.

## 출력
{{
  "recommended_actions": ["string"]
}}
"""


REPORT_COPY_REPAIR_SYSTEM_PROMPT = """\
당신은 임원 보고용 문장을 다듬는 repair agent입니다.
새 사실을 만들지 말고 추상 문장을 근거와 해석이 보이는 문장으로 바꿉니다.
JSON 외 텍스트를 출력하지 마세요.
"""


REPORT_COPY_REPAIR_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## IssueFrame
{issue_frame_json}

## ComparableContext
{comparable_context_json}

## ProfileContext
{profile_json}

## business_line_mapping 후보
{business_lines_json}

## 수정 대상 결과
{result_json}

## 수정 대상 위반
{violations_json}

## 규칙
1. schema 는 유지하고 문장만 고칩니다.
2. analysis 는 현재 사실 → IssueFrame 의 business_object/numbers/target_scope →
   ComparableContext 의 유사 사업·흐름 → 이번 사건에서 확인된 의미를 보여야 합니다.
3. peer_implication 은 현재 사실 → 유사 사업/흐름 → 피어 프로필 접점 또는 접점 부족 한계를
   보여야 합니다.
4. skax_implication 은 피어 신호 → 유사 사업/흐름 → 이번 사건에서 새로 확인된 차이 →
   후속으로 볼 지표를 보여야 합니다.
5. 타깃 피어가 계약 상대방/고객 슬롯이면 공급자처럼 쓰지 않습니다.
6. 프로필 또는 recent context 접점이 없으면 프로필 기반 결론처럼 쓰지 말고
   confidence/evidence_label 을 낮춥니다.
7. 입력에 없는 수치, 고객명, 제품명, 회사명은 추가하지 않습니다.
8. "디지털 전환 지원", "업무 효율성 향상", "역량 강화" 같은 넓은 표현은
   현재 사건의 적용 업무·시스템·서비스·수치·관계와 함께 설명할 때만 사용합니다.
9. "기회", "역량 강화", "표준 제시", "긍정적 영향"처럼 강한 해석은
   현재 사건 사실과 프로필/비교 맥락의 연결이 문장 안에 보일 때만 유지합니다.
   연결이 약하면 어떤 근거가 부족한지와 후속 확인 항목을 밝혀 낮춥니다.

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

## IssueFrame
{issue_frame_json}

## ComparableContext
{comparable_context_json}

## ProfileContext
{profile_json}

## business_line_mapping 후보
{business_lines_json}

## 기존 결과
{result_json}

## 규칙
1. 타깃 피어를 공급자/수행사로 단정하지 않습니다.
   계약 상대방, 사업 범위, 기간, 대상 시스템으로 설명합니다.
2. 공급사 매출 비율은 타깃 피어의 성과나 역량 변화가 아니라 계약 규모 참고 근거입니다.
3. peer_meaning 은 2문장입니다: 현재 계약 사실, 피어 프로필 접점이 드러내는 유사 사업 비교 기준.
4. capability_change 는 공급 역량 강화가 아니라 확인된 사업 범위/대상 시스템/프로필 접점으로 씁니다.
   ProfileContext 에 관련 사업영역/역량이 없으면 모델 일반 지식으로 채우지 말고
   사건 기반 해석으로 낮춥니다.
5. recommended_actions 는 SK AX 관점 체크포인트입니다. 유사 동향에서 비교할 축과
   후속으로 확인할 데이터를 함께 씁니다.
6. 중요성, 연결성, 평가 기준 변화 같은 추상 표현으로 끝내지 않습니다.
7. 이 모드에서는 문장 주어를 가능한 "이번 계약", "해당 사업", "확인된 계약 범위"처럼
   사건/사업명으로 둡니다. 타깃 피어 이름을 주어로 두고 참여·추진·제공·수행·확장한다고
   쓰면 실패입니다.
8. 안전한 구조:
   - analysis: 계약 사실 → 사업명에 드러난 대상 시스템/전환 성격 → 시장 신호
   - peer_meaning: 계약 사실. 타깃 피어는 계약 상대방으로 확인되며, 관련 프로필 사업영역이
     어떤 유사 사업 비교 기준과 접점을 갖는지 설명
   - capability_change: 역량 강화가 아니라 확인된 사업 범위/대상 시스템/기간이 피어 프로필과
     어떤 접점을 갖는지 설명
   - recommended_actions: 유사 동향에서 비교할 축 → 후속 확인 데이터 → SK AX 내부 점검 항목
9. 공개 문장 다듬기는 기존 논리와 구체 명사를 유지한 채 내부 용어만 바꿉니다.
   "통합 결과"는 "기사에서는", "피어 프로필"은 "해당 기업의 기존 사업 흐름",
   "자사 프로필"은 "SK AX의 관련 사업/역량", "linkage/접점"은
   "연결 지점/이어지는 흐름/비교할 지점"으로 바꿉니다.

## 출력
StrategicInsightAgent 최종 schema 그대로 출력합니다.
"""


SYSTEM_PROMPT = """\
당신은 임원 보고용 전략 인사이트를 작성하는 Agent입니다.

데이터 역할:
- IntegratedIssue: 현재 사건의 유일한 사실 근거입니다.
- ProfileContext.peer_profiles: 피어사의 기존 사업영역/역량 배경입니다.
- ProfileContext.skax_profile: SK AX의 기존 사업영역/역량 배경입니다.
- financial_profile_context: 현재 이슈가 재무/IR/투자/공급계약/실적과 직접 관련될 때만 쓰는
  별도 배경입니다. 현재 사건 숫자는 IntegratedIssue.key_numbers 를 우선합니다.
- AnalysisContext: 최근 피어 이벤트와 섹터 흐름 보조 근거입니다.

원칙:
1. 수치, 날짜, 회사명, 고객명, 사업명, 제품명은 IntegratedIssue 근거에 있는 것만 씁니다.
2. ProfileContext 와 AnalysisContext 는 현재 사건의 새 사실이 아니라 해석 배경입니다.
   모델의 일반 지식이나 회사 이미지로 비어 있는 프로필을 채우지 않습니다.
3. 계약/수주/선정/투자/검토 같은 관계 수준을 보존합니다. 역할이 불명확하면
   고객, 원청, 운영 책임자, 도입 주체로 단정하지 않습니다.
4. analysis 는 피어/시장 의미, peer_implication 은 피어 프로필 기반 시사점,
   skax_implication 은 SK AX 관점의 관찰·비교·추적 체크포인트입니다.
5. 좋은 결과는 짧은 결론이 아니라
   "근거 사실 → 왜 그렇게 해석되는지 → 어떤 대응이 필요한지"가 보입니다.
6. JSON 외 텍스트를 출력하지 마세요.
"""


USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## IssueFrame
{issue_frame_json}

## classification
{classification_json}

## input_bundle metadata
{bundle_json}

## ProfileContext
{profile_json}

## AnalysisContext
{context_json}

## ComparableContext
{comparable_context_json}

## Context availability
{context_availability_json}

## SK AX business_line_mapping 후보
{business_lines_json}

## 역할 해석 모드
{role_mode_instructions}

## 생성 순서
1. Fact grounding: 확정 사실, 관계 수준, 수치/날짜, evidence_ids 를 먼저 확인합니다.
2. 사실 기반 해석: 계약 규모, 기간, 고객명, 사업명은 1차 해석 재료로만 쓰고
   이 단계의 결론을 최종 시사점으로 끝내지 않습니다. 아래 peer_signal 까지 연결합니다.
3. peer_role_in_issue: 피어가 공급자/수행사/운영자/고객/계약 상대방 중 무엇으로만
   확인되는지 정합니다. 불명확하면 더 약한 표현을 씁니다.
4. related_peer_profile_context: 현재 사건의 사업명, 대상 시스템, 고객군, 섹터와 맞는
   피어 프로필 사업영역만 고릅니다. 맞지 않는 프로필 조각은 쓰지 않습니다.
5. 사업 맥락 기반 시사점 생성(peer_signal):
   현재 사건의 IssueFrame 을 먼저 봅니다. ComparableContext.matched_cases 가 있으면
   현재 사건과 matched_case 의 공통점과 차이점을 1개 이상 연결합니다.
   ProfileContext 는 matched_cases 보다 우선하지 않습니다.
   관련 피어 ProfileContext 가 현재 사건과 직접 맞으면 기존 사업영역/역량과 현재 사건의
   접점을 설명합니다. 맞지 않으면 프로필 기반 결론처럼 쓰지 않습니다.
   matched_cases 도 없으면 현재 사건 안의 수치/역할/범위만으로 보수적으로 해석합니다.
   피어 프로필에 identity 필드만 있으면 사업영역/역량명을 추측하지 말고 프로필 접점 부족으로 둡니다.
6. SK AX 관점 체크포인트 생성(skax_checkpoint):
   현재 사건의 사업 구조 + 해당 기업의 기존 사업 흐름과의 연결 +
   SK AX와 비교되는 지점 + 왜 중요한지 + 후속으로 확인할 데이터를 정합니다.
   SK AX profile_context 의 관련 사업영역/역량이 있으면 그 접점을 쓰되,
   없으면 특정 역량 대응으로 쓰지 않고 현재 사건 기반 관찰 체크포인트로 낮춥니다.
   고객에게 제안하는 실행계획이 아니라 SK AX 내부 직원/임원이 보는 비교 인사이트로 씁니다.
   "대응 가능한지 내부적으로 구분"처럼 추상적으로 쓰지 말고,
   어떤 역량이 어떤 사업 장면과 비교되는지 설명합니다.
7. final output: 위 중간 판단은 출력하지 말고, schema 필드에 자연어로 반영합니다.

## 시사점 작성 규칙
시사점은 단순 체크리스트가 아니라 사업 비교 인사이트로 작성합니다.
1. 현재 사건의 사업 구조: 기사에서 확인된 사업명, 적용 업무, 시스템, 인프라, 수치,
   참여 주체를 사용합니다.
2. 해당 기업의 기존 사업 흐름과 연결: ProfileContext 또는 ComparableContext에 근거가
   있을 때만 연결합니다. 단순히 "기존 사업 흐름과 연결된다"고 쓰지 말고,
   어떤 역량/사업이 이번 사건의 어떤 적용 장면으로 이어지는지 설명합니다.
3. SK AX와 비교되는 지점: SK AX가 바로 실행해야 할 일을 쓰지 말고,
   어떤 사업 범위, 운영 역할, 데이터 확보 방식, 성과 지표, 책임 구조를
   비교해야 하는지 씁니다.
4. 왜 중요한지: 그 기준이 사업화 가능성, 운영 책임, 성과 검증, 역할 분담 중
   무엇을 판단하게 해주는지 설명합니다.
5. "기존 사업 맥락이 적용 장면과 만난다", "관찰 지점이다"처럼 끝내지 않습니다.
   기존 사업 맥락이 무엇이고, 현재 사건의 어떤 사실 때문에 연결된다고 보는지,
   왜 강한 결론이 아니라 관찰 신호로 낮추는지까지 씁니다.
6. 시사점 한 문장 안에는 가능하면 세 요소가 함께 보여야 합니다:
   현재 사건의 구체 사실, 해당 기업의 기존 사업/역량 근거, 그 둘을 연결해
   해석할 수 있는 범위 또는 한계.
금지: "단순 발표 여부보다 A, B, C를 봐야 한다" 같은 체크리스트형 문장,
"자사 관련 사업이 대응 가능한지 내부적으로 구분" 같은 추상 문장.

## 최종 공개 문장 다듬기 규칙
- 기존 문장의 논리와 구체 명사는 최대한 유지합니다.
- 내부 시스템 용어만 사용자에게 보이는 표현으로 바꿉니다.
- 문장을 짧게 요약하거나 체크리스트로 압축하지 않습니다.
- "통합 결과"는 "기사에서는" 또는 문장 삭제로 처리합니다.
- "피어 프로필"은 "해당 기업의 기존 사업 흐름"으로 바꿉니다.
- "자사 프로필"은 "SK AX의 관련 사업/역량"으로 바꿉니다.
- "피어 쪽"은 해당 기업명 또는 기사에 나온 주체명으로 바꿉니다.
- "linkage", "접점"은 "연결 지점", "이어지는 흐름", "비교할 지점"으로 바꿉니다.
- 단, 원문에 없던 새 사업명이나 역량명은 추가하지 않습니다.

## 필드 기준
- analysis_summary: 현재 사건과 피어/시장 의미를 1~2문장으로 씁니다.
- strategic_meaning: 2~3개. 사실 반복이 아니라 "사실이 의미하는 피어/시장 변화"를 씁니다.
  "영역 확장", "긍정적인 영향"처럼 방향만 말하지 말고, 현재 사건에서 확인된
  대상 시스템/인프라 구성/운영 구조/추진 방식/비교 기준 중 무엇이 바뀌는지 씁니다.
- market_signal: 한 사건으로 수요 증가를 단정하지 말고 현재 사건에서 확인된 수요 신호,
  적용 범위, 비교 기준 변화를 씁니다.
- 시장 범위/표현 강도: 원문이 국내 정부 사업이면 국가 단위/공공 대형 인프라 수준으로 씁니다.
  글로벌, 공공+민간, 전 산업, 사업 확장, 긍정적 영향은 현재 사건 또는 관련 프로필 근거가
  그 범위를 뒷받침할 때만 씁니다. 근거가 약하면 관찰 신호/연결 사례/후속 활용 가능성으로 낮춥니다.
- peer_meaning: 2문장 이상 가능. 현재 사실 → 피어 역할 → 관련 프로필/최근 흐름 접점을 설명합니다.
  관련 피어 프로필이 있으면 프로필의 business_area/core_capability/recent_direction 중
  현재 사건과 맞는 표현을 최소 1개 이상 자연어로 연결합니다.
  단순히 "입지 강화", "영역 확장"으로 끝내지 말고, 관련 프로필의 구체 사업영역/역량명과
  현재 사건의 대상 사업·인프라·고객군이 어떻게 만나는지 씁니다.
  "관찰 지점"이라고 낮출 때는 왜 단정하지 않는지도 함께 씁니다.
  "기존 사업 흐름"이라고 쓰면 그 흐름의 이름/근거와 현재 사건의 적용 업무·시스템·서비스 중
  무엇이 이어지는지까지 씁니다.
- capability_change: 직접 확인되는 사업 범위, 고객군, 대상 시스템, 적용 영역만 씁니다.
  "역량 강화/영역 확장 예상"으로 끝내지 말고, 어떤 고객군·대상 시스템·운영 범위·추진 구조와
  연결되는지 적습니다.
  프로필 근거가 있으면 "기존 역량이 이번 사건에서 어떤 적용 장면/운영 범위/고객군과
  만나는지"를 설명하고, 프로필 근거가 없으면 역량 변화로 단정하지 않습니다.
- why_important: 피어/시장 신호가 SK AX의 어떤 사업영역/역량과 비교되는지 씁니다.
- potential_impact: 피어 신호가 SK AX 내부에서 어떤 비교/관찰 항목을 만들게 하는지,
  어떤 후속 데이터가 있어야 판단 가능한지 2~3문장으로 씁니다.
- recommended_actions: 1~3개. 각 항목은 길어도 됩니다.
  현재 피어 신호 → 비교해야 할 축 → 왜 그 축이 필요한지 → 후속 확인 데이터 →
  SK AX 내부 점검 관점을 연결합니다.
  모호한 "대응 가능 범위 구분"으로 끝내지 말고, 그 범위가 사업화 가능성, 운영 책임,
  성과 검증, 역할 분담 중 무엇을 판단하게 하는지 설명합니다.
- business_line_mapping: 입력 후보 name 중 실제 관련 있는 항목만 0~3개 선택합니다.
- sourced_evidence_ids / used_fact_ids: 입력에 존재하는 fact_id 만 사용합니다.

## 금지
- 근거 없는 기술적 우위, 선점, 격차, 경쟁 심화, 점유율 확대, 성과 예측
- 계약 상대방을 공급자/수행사로 바꾸는 표현
- ProfileContext 에 없는 사업영역/역량명을 모델 일반 지식으로 생성
- SK AX 프로필이 없는데 SK AX의 특정 역량/사업영역과 연결하는 표현
- 현재 사건 신호, 비교 축, 후속 확인 데이터가 없는 체크포인트
- 출력 schema 예시 문구 복사

## 출력
아래 JSON schema 를 그대로 지켜 출력합니다. 설명 텍스트나 markdown 은 출력하지 마세요.
{{
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
    "confidence": 0.0,
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
      "business_line_mapping": ["후보 중 실제 관련 있는 name"]
    }},
    "follow_up_questions": ["string"],
    "watch_points": ["string"],
    "confidence": 0.0,
    "evidence_label": "sufficient|moderate|insufficient",
    "provenance": {{
      "generator": "StrategicInsightAgent",
      "prompt_version": "{prompt_version}",
      "model": "{model}",
      "used_fact_ids": ["입력에 존재하는 fact_id"],
      "used_context_layers": ["실제로 사용한 context layer명"],
      "run_at": "ISO-8601 timestamp"
    }}
  }}
}}
"""


REVIEW_SYSTEM_PROMPT = """\
당신은 StrategicInsightAgent 결과를 점검하는 전략 QA reviewer입니다.
새 사실을 만들지 말고, 입력 근거와 프로필만 사용해 논리 공백을 고칩니다.
복구할 수 없으면 invalid 로 낮춥니다. JSON 외 텍스트를 출력하지 마세요.
"""


REVIEW_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## IssueFrame
{issue_frame_json}

## classification
{classification_json}

## ProfileContext
{profile_json}

## ComparableContext
{comparable_context_json}

## AnalysisContext
{context_json}

## Context availability
{context_availability_json}

## business_line_mapping 후보
{business_lines_json}

## 1차 결과
{result_json}

## 리뷰 기준
1. 수치/날짜/회사명/고객명/사업명은 IntegratedIssue 근거 안에 있어야 합니다.
2. 시사점이 요약 반복이면 고칩니다. 현재 사실, ComparableContext 의 유사 사업/흐름,
   피어 프로필 접점 또는 접점 부족 한계가 보여야 합니다.
   profile_context 나 recent context 가 없거나 현재 사건과 맞는 접점이 없으면
   프로필 기반 결론처럼 쓰지 말고 confidence/evidence_label 을 낮춥니다.
   피어 프로필이 identity 필드뿐이면 모델 일반 지식으로 사업영역/역량명을 만들지 않습니다.
   관련 피어 프로필이 있으면 기존 사업영역/역량 → 현재 사건 접점 → 사업적 의미가
   보여야 합니다. 이 연결이 없으면 요약 반복으로 보고 고칩니다.
   "영역 확장", "긍정적 영향"처럼 방향만 말하는 문장은 현재 사건에서 확인된
   대상 시스템, 인프라 구성, 운영 구조, 추진 방식, 비교 기준으로 구체화합니다.
3. 피어가 계약 상대방/고객 슬롯이면 피어가 제공·수행·지원·운영했다고 쓰지 않습니다.
   계약 상대방으로 확인된 사업 범위, 계약 기간, 대상 시스템, 프로필 사업영역 접점으로 낮춥니다.
4. peer_meaning 은 2문장입니다. 현재 사실과 피어 프로필 접점이 어떤 유사 사업 비교 기준을
   보여주는지까지 설명해야 합니다.
5. recommended_actions 는 현재 피어/시장 신호, 비슷한 사업/흐름, 이번 사건에서 새로 확인된 차이,
   후속 확인 데이터가 연결되어야 합니다. SK AX 프로필이 없으면 특정 역량 대응은 금지하지만,
   동향 관찰 체크포인트는 유지합니다.
6. recommended_actions 는 실행계획이 아니라 SK AX 관점 체크포인트입니다.
   현재 사건 신호, 비교 축, 후속 확인 데이터, 내부 점검 관점이 보여야 합니다.
7. 기술명+솔루션 표현은 IntegratedIssue에 해당 기술명이 직접 있을 때만 씁니다.
   business_line 후보만 보고 "클라우드 및 AI 솔루션"처럼 솔루션명을 만들지 않습니다.
8. 성공 사례, 구축 경험, 운영 역량은 ProfileContext에 실제 근거가 있을 때만 씁니다.
   근거가 없으면 현재 사건에서 확인된 역할, 수치, 일정, 적용 업무 수준으로 낮춥니다.
9. 시장 범위와 표현 강도는 근거 범위를 넘지 않습니다.
   글로벌/해외, 공공+민간 양쪽, 전 산업, 사업영역 확장, 입지 강화,
   긍정적 영향 같은 표현은 IntegratedIssue 또는 관련 프로필에 그 범위를
   뒷받침하는 근거가 있을 때만 씁니다. 근거가 약하면 관찰 신호,
   연결 사례, 검증 계기, 후속 활용 가능성으로 낮춥니다.
10. 근거 없는 우위/선점/점유율/경쟁 심화/성과 예측은 제거합니다.
11. 공개 문장 다듬기는 기존 논리와 구체 명사를 유지한 채 내부 용어만 바꿉니다.
   "통합 결과"는 "기사에서는", "피어 프로필"은 "해당 기업의 기존 사업 흐름",
   "자사 프로필"은 "SK AX의 관련 사업/역량", "linkage/접점"은
   "연결 지점/이어지는 흐름/비교할 지점"으로 바꿉니다.
12. "관찰 지점", "이어지는 흐름", "비교해야 합니다", "모니터링해야 합니다"처럼
   결론만 있는 문장은 고칩니다. 현재 사건의 구체 사실, 연결되는 기존 사업/역량 근거,
   왜 이 기준이 필요한지가 함께 보이게 씁니다.

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

## IssueFrame
{issue_frame_json}

## ComparableContext
{comparable_context_json}

## ProfileContext
{profile_json}

## business_line_mapping 후보
{business_lines_json}

## 검증 실패 사유
{violations_json}

## 수정 대상 결과
{result_json}

## repair 기준
1. 없는 수치, 없는 관계, 근거 없는 역할 단정을 제거합니다.
2. 피어가 계약 상대방/고객 슬롯이면 제공·수행·지원·운영 같은 공급자 행동을 제거하고,
   계약 상대방으로 확인된 사업 범위, 기간, 대상 시스템, 프로필 사업영역 접점으로 고칩니다.
3. 공급사 매출 비율은 계약 규모 참고 근거로만 쓰고 피어사의 성과로 쓰지 않습니다.
4. 시사점은 현재 사건과 ComparableContext, 피어 프로필 접점으로 고칩니다.
   접점이 없으면 사건 기반 해석으로 낮춥니다.
   ProfileContext 에 없는 피어 사업영역/역량명은 제거합니다.
5. SK AX 체크포인트는 skax_profile/business_line 후보와 연결합니다.
6. recommended_actions 는 현재 사건 신호, 비슷한 사업/흐름, 이번 사건에서 새로 확인된 차이,
   후속 확인 데이터가 보이게 고칩니다.
7. 기술명+솔루션 표현은 IntegratedIssue에 해당 기술명이 직접 있을 때만 씁니다.
   business_line 후보만으로 "클라우드 및 AI 솔루션" 같은 표현을 만들지 않습니다.
8. 성공 사례, 구축 경험, 운영 역량은 ProfileContext에 실제 사례 근거가 있을 때만 씁니다.
9. 공개 문장 다듬기는 기존 논리와 구체 명사를 유지한 채 내부 용어만 바꿉니다.
   "통합 결과"는 "기사에서는", "피어 프로필"은 "해당 기업의 기존 사업 흐름",
   "자사 프로필"은 "SK AX의 관련 사업/역량", "linkage/접점"은
   "연결 지점/이어지는 흐름/비교할 지점"으로 바꿉니다.
10. 검증 실패가 "근거 흐름 부족"이면 문장을 짧게 만들지 말고,
   현재 사건의 구체 사실 → 프로필/비교 근거 → 해석 범위 또는 한계 → 후속 확인 데이터
   순서가 보이도록 보강합니다.

## 출력
StrategicInsightAgent 최종 schema 그대로 JSON 으로 출력합니다.
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
        profile_relevance_text = _profile_relevance_hint_text(
            integrated_issue=integrated_issue,
            classification=classification,
            bundle=bundle_dict,
        )
        include_financial_profile_context = _should_include_financial_profile_context(
            integrated_issue=integrated_issue,
            classification=classification,
            bundle=bundle_dict,
        )
        context_for_model = _analysis_context_for_model(
            context_dict,
            integrated_issue=integrated_issue,
            relevance_hint_text=profile_relevance_text,
            include_financial_context=include_financial_profile_context,
        )
        issue_frame = _issue_frame_for_prompt(integrated_issue)
        comparable_context = _comparable_context_for_prompt(
            integrated_issue=integrated_issue,
            issue_frame=issue_frame,
            analysis_context=context_for_model,
            profile_context=profile_dict,
        )

        prompt = USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            issue_frame_json=_json_dumps(issue_frame),
            comparable_context_json=_json_dumps(comparable_context),
            classification_json=_json_dumps(_classification_for_prompt(classification)),
            bundle_json=_json_dumps(_bundle_for_prompt(bundle_dict, cluster_metadata)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_dict,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                )
            ),
            context_json=_json_dumps(_analysis_context_for_prompt(context_for_model)),
            context_availability_json=_json_dumps(
                _context_availability_for_prompt(
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                    analysis_context=context_for_model,
                )
            ),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_dict,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
            role_mode_instructions=_role_mode_instructions(integrated_issue),
            prompt_version=_PROMPT_VERSION,
            model=self.model,
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
                analysis_context=context_for_model,
                model=self.model,
            )
            if not self.enable_self_review:
                return self._finalize_quality_gate(
                    result,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                )
            reviewed = self._review_and_revise(
                result,
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_dict,
                analysis_context=context_for_model,
                bundle_id=str(
                    bundle_dict.get("bundle_id")
                    or integrated_issue.get("bundle_id")
                    or cluster_metadata.get("bundle_id")
                    or ""
                ),
                profile_relevance_text=profile_relevance_text,
                include_financial_profile_context=include_financial_profile_context,
            )
            if "quality_gate_failed" in _json_dumps(reviewed):
                reviewed_violations = _quality_gate_violations(
                    reviewed,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                )
                if _quality_failure_is_mechanical_repair_candidate(
                    reviewed_violations
                ) or _quality_failure_can_fallback_to_issue_frame(
                    reviewed_violations,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                ):
                    guarded = _minimal_quality_guard(
                        reviewed,
                        integrated_issue=integrated_issue,
                        profile_context=profile_dict,
                    )
                    remaining = _quality_gate_violations(
                        guarded,
                        integrated_issue=integrated_issue,
                        profile_context=profile_dict,
                    )
                    if not remaining:
                        return _attach_sentence_grounding(
                            _restore_valid_flags_if_structurally_safe(guarded),
                            integrated_issue=integrated_issue,
                            profile_context=profile_dict,
                        )
                return _attach_sentence_grounding(
                    reviewed,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                )
            return self._finalize_quality_gate(
                reviewed,
                integrated_issue=integrated_issue,
                profile_context=profile_dict,
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

    def _finalize_quality_gate(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        profile_context: dict[str, Any],
    ) -> dict[str, Any]:
        violations = _quality_gate_violations(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if not violations:
            guarded = _minimal_quality_guard(
                result,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            remaining = _quality_gate_violations(
                guarded,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            if remaining:
                return _attach_sentence_grounding(
                    _mark_quality_gate_failed(guarded, remaining),
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                )
            return _attach_sentence_grounding(
                _restore_valid_flags_if_structurally_safe(guarded),
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
        guarded = _minimal_quality_guard(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        remaining = _quality_gate_violations(
            guarded,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if remaining:
            return _attach_sentence_grounding(
                _mark_quality_gate_failed(guarded, remaining),
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
        return _attach_sentence_grounding(
            _restore_valid_flags_if_structurally_safe(guarded),
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )

    def _review_and_revise(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
    ) -> dict[str, Any]:
        prompt = REVIEW_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            issue_frame_json=_json_dumps(_issue_frame_for_prompt(integrated_issue)),
            comparable_context_json=_json_dumps(
                _comparable_context_for_prompt(
                    integrated_issue=integrated_issue,
                    issue_frame=_issue_frame_for_prompt(integrated_issue),
                    analysis_context=analysis_context,
                    profile_context=profile_context,
                )
            ),
            classification_json=_json_dumps(_classification_for_prompt(classification)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                )
            ),
            context_json=_json_dumps(_analysis_context_for_prompt(analysis_context)),
            context_availability_json=_json_dumps(
                _context_availability_for_prompt(
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    analysis_context=analysis_context,
                )
            ),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
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
                profile_relevance_text=profile_relevance_text,
                include_financial_profile_context=include_financial_profile_context,
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
                return _mark_quality_gate_failed(result, violations)
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
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
    ) -> dict[str, Any]:
        current = result
        current_violations = violations
        try:
            for attempt in range(3):
                prompt = REPAIR_USER_PROMPT_TEMPLATE.format(
                    integrated_issue_json=_json_dumps(
                        _integrated_issue_for_prompt(integrated_issue)
                    ),
                    issue_frame_json=_json_dumps(_issue_frame_for_prompt(integrated_issue)),
                    comparable_context_json=_json_dumps(
                        _comparable_context_for_prompt(
                            integrated_issue=integrated_issue,
                            issue_frame=_issue_frame_for_prompt(integrated_issue),
                            analysis_context=analysis_context,
                            profile_context=profile_context,
                        )
                    ),
                    profile_json=_json_dumps(
                        _profile_for_prompt(
                            profile_context,
                            integrated_issue=integrated_issue,
                            relevance_hint_text=profile_relevance_text,
                            include_financial_context=include_financial_profile_context,
                        )
                    ),
                    business_lines_json=_json_dumps(
                        _business_line_candidate_details(
                            profile_context,
                            integrated_issue=integrated_issue,
                            relevance_hint_text=profile_relevance_text,
                        )
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
                    profile_relevance_text=profile_relevance_text,
                    include_financial_profile_context=include_financial_profile_context,
                )
                current_violations = _quality_gate_violations(
                    current,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                )
                if not current_violations:
                    return current
            if current_violations:
                current = self._repair_report_copy_result(
                    current,
                    violations=current_violations,
                    integrated_issue=integrated_issue,
                    classification=classification,
                    profile_context=profile_context,
                    analysis_context=analysis_context,
                    bundle_id=bundle_id,
                    profile_relevance_text=profile_relevance_text,
                    include_financial_profile_context=include_financial_profile_context,
                )
                current_violations = _quality_gate_violations(
                    current,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                )
                if not current_violations:
                    return current
            guarded = _minimal_quality_guard(
                current,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            if (
                len(
                    _string_list(
                        ((guarded.get("implication") or {}).get("skax_implication") or {}).get(
                            "recommended_actions"
                        ),
                        max_items=3,
                    )
                )
                < 2
            ):
                guarded = self._repair_missing_recommended_actions(
                    guarded,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    bundle_id=bundle_id,
                    profile_relevance_text=profile_relevance_text,
                )
                guarded = _minimal_quality_guard(
                    guarded,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                )
            guarded = _ensure_safe_recommended_actions(
                guarded,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
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
            final_remaining = _quality_gate_violations(
                result,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            return _mark_quality_gate_failed(
                result,
                final_remaining
                or [
                    "StrategicInsightAgent quality repair failed; "
                    "근거 기반 repair를 완료하지 못해 fail-closed 처리합니다."
                ],
            )

    def _repair_counterparty_role_result(
        self,
        result: dict[str, Any],
        *,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
    ) -> dict[str, Any]:
        prompt = COUNTERPARTY_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            issue_frame_json=_json_dumps(_issue_frame_for_prompt(integrated_issue)),
            comparable_context_json=_json_dumps(
                _comparable_context_for_prompt(
                    integrated_issue=integrated_issue,
                    issue_frame=_issue_frame_for_prompt(integrated_issue),
                    analysis_context=analysis_context,
                    profile_context=profile_context,
                )
            ),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                )
            ),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
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

    def _repair_report_copy_result(
        self,
        result: dict[str, Any],
        *,
        violations: list[str],
        integrated_issue: dict[str, Any],
        classification: dict[str, Any],
        profile_context: dict[str, Any],
        analysis_context: dict[str, Any],
        bundle_id: str,
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
    ) -> dict[str, Any]:
        prompt = REPORT_COPY_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            issue_frame_json=_json_dumps(_issue_frame_for_prompt(integrated_issue)),
            comparable_context_json=_json_dumps(
                _comparable_context_for_prompt(
                    integrated_issue=integrated_issue,
                    issue_frame=_issue_frame_for_prompt(integrated_issue),
                    analysis_context=analysis_context,
                    profile_context=profile_context,
                )
            ),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                )
            ),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
            result_json=_json_dumps(result),
            violations_json=_json_dumps(violations),
        )
        content = self._invoke_llm(
            system_prompt=REPORT_COPY_REPAIR_SYSTEM_PROMPT,
            user_prompt=prompt,
            bundle_id=bundle_id,
            phase="quality_repair_report_copy",
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
        profile_relevance_text: str = "",
    ) -> dict[str, Any]:
        out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        implication = out.get("implication") or {}
        skax = implication.get("skax_implication") or {}
        prompt = ACTION_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            issue_frame_json=_json_dumps(_issue_frame_for_prompt(integrated_issue)),
            comparable_context_json=_json_dumps(
                _comparable_context_for_prompt(
                    integrated_issue=integrated_issue,
                    issue_frame=_issue_frame_for_prompt(integrated_issue),
                    analysis_context={},
                    profile_context=profile_context,
                )
            ),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                )
            ),
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
                integrated_issue=integrated_issue,
                profile_context=profile_context,
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
        result = {
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
        return _attach_sentence_grounding(
            result,
            integrated_issue=integrated_issue,
            profile_context=_profile_to_dict(profile_context),
        )


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
        "sentence_grounding": {
            "schema_version": "sentence-grounding-v1",
            "generator": "StrategicInsightAgent",
            "entries": [],
            "summary": {
                "entry_count": 0,
                "fact_grounded_count": 0,
                "profile_grounded_count": 0,
                "ungrounded_paths": [],
            },
        },
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


def _issue_frame_for_prompt(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    """Structured reading frame extracted only from IntegratedIssue evidence.

    This is intentionally conservative: it does not decide strategy. It gives the
    LLM a small, evidence-scoped map of who/what/action/scope/numbers so the
    generated implication does not drift into generic proposal copy.
    """
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    intelligence = intelligence if isinstance(intelligence, dict) else {}
    activity_types = _string_list(intelligence.get("activity_types"), max_items=8)
    products_or_services = _unique_texts(
        [
            *_string_list(intelligence.get("products_or_services"), max_items=12),
            *_products_from_facts(integrated_issue),
        ],
        max_items=12,
    )
    customers_or_industries = _unique_texts(
        [
            *_string_list(intelligence.get("customers_or_industries"), max_items=12),
            *_customers_from_facts(integrated_issue),
        ],
        max_items=12,
    )
    main_fact = _primary_issue_fact(integrated_issue)
    focus_text = _issue_focus_text(integrated_issue)
    frame_kind = _issue_frame_kind(integrated_issue)
    evidence_ids = sorted(_known_fact_ids(integrated_issue))
    action = _issue_frame_action(
        integrated_issue,
        activity_types=activity_types,
        evidence_ids=evidence_ids,
        frame_kind=frame_kind,
    )
    business_object = _issue_frame_business_object(
        integrated_issue,
        products_or_services=products_or_services,
        evidence_ids=evidence_ids,
        frame_kind=frame_kind,
    )
    target_scope = _issue_frame_target_scope(
        integrated_issue,
        products_or_services=products_or_services,
        customers_or_industries=customers_or_industries,
        frame_kind=frame_kind,
        focus_text=focus_text,
    )
    return {
        "schema_version": "issue-frame-v1",
        "main_fact": main_fact,
        "frame_kind": frame_kind,
        "entities": _issue_frame_entities(
            integrated_issue,
            customers_or_industries=customers_or_industries,
            evidence_ids=evidence_ids,
        ),
        "action_or_event": action,
        "business_object": business_object,
        "target_scope": target_scope,
        "numbers": _issue_frame_numbers(integrated_issue, evidence_ids=evidence_ids),
        "profile_matching_terms": _issue_terms_for_profile_matching(
            integrated_issue,
            products_or_services=products_or_services,
            customers_or_industries=customers_or_industries,
        ),
        "unsupported_inferences": [
            "IssueFrame에 없는 회사 역할은 현재 사건 역할로 단정하지 않습니다.",
            "ProfileContext의 사업영역은 현재 사건과 연결될 때만 배경으로 사용합니다.",
            "수치·기간·성과·인과는 evidence_id가 있는 근거 범위에서만 사용합니다.",
        ],
    }


def _comparable_context_for_prompt(
    *,
    integrated_issue: dict[str, Any],
    issue_frame: dict[str, Any],
    analysis_context: dict[str, Any],
    profile_context: dict[str, Any],
) -> dict[str, Any]:
    """Find nearby business/flow context from already available context.

    This does not invent a similar business. It only exposes compact candidates
    whose text overlaps with the current issue terms, so the LLM can connect
    current facts to comparable flows when the surrounding context actually has
    a match.
    """
    query_terms = _build_comparable_query_terms(issue_frame)
    candidates: list[dict[str, Any]] = []
    for layer_name, layer_value in (analysis_context or {}).items():
        candidates.extend(
            _comparable_candidates_from_value(
                layer_value,
                query_terms=query_terms,
                layer_name=str(layer_name),
                case_type=_comparable_case_type_for_layer(str(layer_name)),
            )
        )
    candidates.extend(
        _comparable_candidates_from_value(
            _profile_for_prompt(profile_context, integrated_issue=integrated_issue),
            query_terms=query_terms,
            layer_name="profile_context",
            case_type="profile_case",
        )
    )
    ranked = _rank_comparable_candidates(
        candidates,
        query_terms=query_terms,
        issue_frame=issue_frame,
    )
    return {
        "schema_version": "comparable-context-v1",
        "query_terms": query_terms,
        "matched_cases": ranked[:5],
        "has_comparable_context": bool(ranked),
        "guidance": [
            "matched_cases가 있을 때만 유사 사업/비슷한 흐름으로 연결합니다.",
            "matched_cases가 없으면 유사 사업을 상상하지 않습니다.",
            "연결은 현재 사건과 matched_case의 공통 business_object/target_scope/"
            "metric_type 기준으로만 합니다.",
        ],
    }


def _build_comparable_query_terms(issue_frame: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    business_object = issue_frame.get("business_object") or {}
    terms.extend(_string_list(business_object.get("raw_terms"), max_items=12))
    if business_object.get("raw_name"):
        terms.append(str(business_object.get("raw_name")))
    elif business_object.get("name"):
        terms.append(str(business_object.get("name")))

    target_scope = issue_frame.get("target_scope") or {}
    for key in (
        "target_work",
        "target_system",
        "target_customer_or_industry",
        "geography_or_market",
    ):
        terms.extend(_string_list(target_scope.get(key), max_items=8))

    action = issue_frame.get("action_or_event") or {}
    terms.extend(_string_list(action.get("activity_types"), max_items=8))
    for key in ("verb", "status"):
        if action.get(key):
            terms.append(str(action.get(key)))

    for item in _jsonish_list(issue_frame.get("numbers")):
        if not isinstance(item, dict):
            continue
        for key in ("metric", "meaning"):
            if item.get(key):
                terms.append(str(item.get(key)))
    return _unique_texts(terms, max_items=24)


def _comparable_candidates_from_value(
    value: Any,
    *,
    query_terms: list[str],
    layer_name: str,
    case_type: str,
) -> list[dict[str, Any]]:
    query_tokens = _terms_to_content_tokens(query_terms)
    if not query_tokens:
        return []
    candidates: list[dict[str, Any]] = []
    for path, item in _iter_context_dicts(value, root=layer_name, limit=120):
        text = _comparable_item_text(item)
        if not text:
            continue
        item_tokens = _content_tokens(text)
        matched_tokens = sorted(query_tokens & item_tokens)
        if not matched_tokens:
            continue
        score = len(matched_tokens)
        phrase_hits = [
            term
            for term in query_terms
            if len(term) >= 3 and re.search(re.escape(term), text, flags=re.IGNORECASE)
        ]
        score += min(4, len(phrase_hits))
        if score <= 0:
            continue
        title = _comparable_item_title(item, fallback=path)
        candidates.append(
            {
                "case_title": title[:120],
                "case_type": case_type,
                "layer": layer_name,
                "path": path,
                "matched_terms": _unique_texts([*phrase_hits, *matched_tokens], max_items=8),
                "shared_context": _shared_context_phrase(matched_tokens, phrase_hits),
                "source_ref": _comparable_item_source_ref(item),
                "_score": score,
                "_text": text[:500],
            }
        )
    return candidates


def _iter_context_dicts(value: Any, *, root: str, limit: int) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    def walk(node: Any, path: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, dict):
            out.append((path, node))
            for key, child in node.items():
                if key in {"evidence_payload", "raw_payload", "embedding", "vector"}:
                    continue
                walk(child, f"{path}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node[:30]):
                walk(child, f"{path}[{index}]")

    walk(value, root)
    return out


def _comparable_item_text(item: dict[str, Any]) -> str:
    keys = (
        "title",
        "headline",
        "summary",
        "analysis_summary",
        "market_signal",
        "peer_meaning",
        "capability_change",
        "reason",
        "one_liner",
        "company_summary",
        "name",
        "summary_text",
        "event_summary",
        "recent_direction",
        "overall_change",
        "text",
        "content",
    )
    parts: list[str] = []
    for key in keys:
        value = item.get(key)
        if value in ({}, [], "", None):
            continue
        if isinstance(value, (dict, list)):
            parts.append(_json_dumps(_compact_value(value)))
        else:
            parts.append(str(value))
    for key in ("business_areas", "core_capabilities", "strategic_focus", "source_refs"):
        value = item.get(key)
        if value not in ({}, [], "", None):
            parts.append(_json_dumps(_compact_value(value)))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _comparable_item_title(item: dict[str, Any], *, fallback: str) -> str:
    for key in ("title", "headline", "name", "one_liner", "summary"):
        value = str(item.get(key) or "").strip()
        if value:
            return re.sub(r"\s+", " ", value)
    return fallback


def _comparable_item_source_ref(item: dict[str, Any]) -> str:
    for key in ("card_id", "id", "source_ref", "source_id", "event_id", "article_id"):
        value = item.get(key)
        if value not in ("", None):
            return str(value)
    refs = item.get("source_refs")
    if isinstance(refs, list) and refs:
        return str(refs[0])
    return ""


def _terms_to_content_tokens(terms: list[str]) -> set[str]:
    tokens: set[str] = set()
    for term in terms:
        tokens.update(_content_tokens(term))
    return {token for token in tokens if len(token) >= 2}


def _shared_context_phrase(matched_tokens: list[str], phrase_hits: list[str]) -> str:
    terms = _unique_texts([*phrase_hits, *matched_tokens], max_items=5)
    if not terms:
        return "현재 사건과 일부 용어가 겹칩니다."
    return "현재 사건과 공통으로 확인되는 용어: " + ", ".join(terms)


def _rank_comparable_candidates(
    candidates: list[dict[str, Any]],
    *,
    query_terms: list[str],
    issue_frame: dict[str, Any],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (int(item.get("_score") or 0), len(str(item.get("_text") or ""))),
        reverse=True,
    )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted_candidates:
        key = re.sub(r"\s+", " ", str(item.get("case_title") or "")).casefold()
        if not key or key in seen:
            continue
        clean = {
            "case_title": item.get("case_title"),
            "case_type": item.get("case_type"),
            "matched_terms": item.get("matched_terms") or [],
            "shared_context": item.get("shared_context") or "",
            "difference_from_current_issue": _current_issue_difference_hint(
                issue_frame,
                query_terms=query_terms,
            ),
            "source_ref": item.get("source_ref") or item.get("path") or "",
        }
        deduped.append(clean)
        seen.add(key)
        if len(deduped) >= 8:
            break
    return deduped


def _current_issue_difference_hint(issue_frame: dict[str, Any], *, query_terms: list[str]) -> str:
    numbers = [
        " ".join(
            part
            for part in (
                str(item.get("metric") or "").strip(),
                str(item.get("value") or "").strip(),
                str(item.get("unit") or "").strip(),
            )
            if part
        )
        for item in _jsonish_list(issue_frame.get("numbers"))
        if isinstance(item, dict)
    ]
    action = issue_frame.get("action_or_event") or {}
    status = str(action.get("status") or "").strip()
    business_object = issue_frame.get("business_object") or {}
    raw_name = str(business_object.get("raw_name") or business_object.get("name") or "").strip()
    parts = []
    if raw_name:
        parts.append(raw_name)
    if status and status != "unclear":
        parts.append(f"status={status}")
    if numbers:
        parts.append("numbers=" + ", ".join(numbers[:3]))
    if not parts and query_terms:
        parts.append("issue_terms=" + ", ".join(query_terms[:3]))
    return "이번 사건에서 별도로 확인된 조건: " + "; ".join(parts[:3])


def _comparable_case_type_for_layer(layer_name: str) -> str:
    name = str(layer_name or "").casefold()
    if "similar" in name or "card" in name:
        return "past_card"
    if "peer" in name or "timeline" in name:
        return "peer_event"
    if "sector" in name or "pulse" in name:
        return "sector_pulse"
    if "profile" in name:
        return "profile_case"
    return "context_case"


def _issue_focus_text(integrated_issue: dict[str, Any]) -> str:
    chunks = [
        str(integrated_issue.get(key) or "").strip()
        for key in ("headline", "one_line_summary", "main_event", "main_issue")
        if str(integrated_issue.get(key) or "").strip()
    ]
    primary = _primary_issue_fact(integrated_issue)
    if primary:
        chunks.append(primary)
    return " ".join(dict.fromkeys(chunks))


def _issue_frame_kind(integrated_issue: dict[str, Any]) -> str:
    event_type = str(integrated_issue.get("cluster_event_type") or "").casefold()
    focus_text = _issue_focus_text(integrated_issue)
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    activity_text = " ".join(
        _string_list(
            intelligence.get("activity_types") if isinstance(intelligence, dict) else None,
            max_items=12,
        )
    ).casefold()
    event_activity_text = f"{event_type} {activity_text}"
    if event_type in {"earnings", "performance", "financial_result", "financial"} or re.search(
        r"매출|영업이익|순이익|영업이익률|실적|수익성|전년\s*동기|분기",
        focus_text,
    ):
        return "performance"
    if re.search(
        r"selection|selected|build|construction|operation|선정|확정|구축",
        event_activity_text,
    ):
        return "selection_or_build"
    if event_type in {"partnership", "mou", "collaboration"} or re.search(
        r"협약|MOU|협력|공동",
        focus_text,
        flags=re.IGNORECASE,
    ):
        return "partnership"
    if event_type in {"launch", "release"} or re.search(
        r"출시|공개|선보|서비스\s*개시|플랫폼",
        focus_text,
    ):
        return "launch_or_service"
    if re.search(r"계약|수주|전환|현대화|시스템", focus_text):
        return "contract_or_transition"
    return "general"


def _issue_frame_entities(
    integrated_issue: dict[str, Any],
    *,
    customers_or_industries: list[str],
    evidence_ids: list[str],
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    main_company = str(integrated_issue.get("main_company") or "").strip()
    if main_company:
        role = (
            "counterparty_or_customer"
            if _main_company_is_customer_or_buyer(integrated_issue)
            else "main_company_in_issue"
        )
        entities.append(
            {
                "name": main_company,
                "role_in_article": role,
                "evidence_id": evidence_ids[0] if evidence_ids else "",
            }
        )
    for company_id in _jsonish_list(integrated_issue.get("mentioned_peer_companies"))[:5]:
        name = str(company_id or "").strip()
        if name and name != main_company:
            entities.append(
                {
                    "name": name,
                    "role_in_article": "mentioned_peer",
                    "evidence_id": evidence_ids[0] if evidence_ids else "",
                }
            )
    for item in customers_or_industries[:5]:
        if item and item not in {entity["name"] for entity in entities}:
            entities.append(
                {
                    "name": item,
                    "role_in_article": "customer_or_industry_from_evidence",
                    "evidence_id": evidence_ids[0] if evidence_ids else "",
                }
            )
    return entities[:10]


def _issue_frame_action(
    integrated_issue: dict[str, Any],
    *,
    activity_types: list[str],
    evidence_ids: list[str],
    frame_kind: str = "",
) -> dict[str, Any]:
    evidence_text = _integrated_grounding_text(integrated_issue)
    focus_text = _issue_focus_text(integrated_issue)
    if frame_kind == "performance":
        verb = "performance"
        status = "reported"
        return {
            "verb": verb,
            "status": status,
            "activity_types": activity_types,
            "evidence_id": evidence_ids[0] if evidence_ids else "",
        }
    verb = activity_types[0] if activity_types else _fallback_action_label(evidence_text)
    status = "unclear"
    if re.search(r"선정|확정|체결|수주|완료|공개|출시|개시", focus_text or evidence_text):
        status = "confirmed"
    elif re.search(r"계획|예정|검토|추진|모색", focus_text or evidence_text):
        status = "planned_or_under_review"
    return {
        "verb": verb or "unclear",
        "status": status,
        "activity_types": activity_types,
        "evidence_id": evidence_ids[0] if evidence_ids else "",
    }


def _issue_frame_business_object(
    integrated_issue: dict[str, Any],
    *,
    products_or_services: list[str],
    evidence_ids: list[str],
    frame_kind: str = "",
) -> dict[str, Any]:
    focus_text = _issue_focus_text(integrated_issue)
    evidence_text = focus_text or _integrated_grounding_text(integrated_issue)
    if frame_kind == "performance":
        return {
            "name": "전사 실적·수익성 지표",
            "type_from_evidence": "performance",
            "raw_name": "전사 실적·수익성 지표",
            "raw_terms": _issue_object_raw_terms("전사 실적·수익성 지표", evidence_text),
            "object_nature_hints": ["performance_like"],
            "evidence_id": evidence_ids[0] if evidence_ids else "",
        }
    name = (
        _issue_subject_phrase(integrated_issue)
        or str(integrated_issue.get("headline") or "").strip()
    )
    if not name and products_or_services:
        name = products_or_services[0]
    type_from_evidence = "business_or_service"
    if re.search(r"GPU|컴퓨팅|센터|인프라|데이터\s*센터|서버", evidence_text, re.IGNORECASE):
        type_from_evidence = "infrastructure"
    elif re.search(r"플랫폼|서비스|솔루션", evidence_text):
        type_from_evidence = "platform_or_service"
    elif re.search(r"시스템|웹단말|코어뱅킹|전환|현대화", evidence_text):
        type_from_evidence = "target_system_transition"
    elif re.search(r"협약|계약|MOU|수주", evidence_text, re.IGNORECASE):
        type_from_evidence = "contract_or_agreement"
    elif re.search(r"매출|영업이익|수익|실적", evidence_text):
        type_from_evidence = "performance"
    return {
        "name": name,
        "type_from_evidence": type_from_evidence,
        "raw_name": name,
        "raw_terms": _issue_object_raw_terms(name, evidence_text),
        "object_nature_hints": _issue_object_nature_hints(
            name,
            evidence_text,
            type_from_evidence=type_from_evidence,
        ),
        "evidence_id": evidence_ids[0] if evidence_ids else "",
    }


def _issue_object_raw_terms(name: str, evidence_text: str) -> list[str]:
    """Keep observed object terms as evidence hints, not as hard classifications."""
    terms: list[str] = []
    terms.extend(re.findall(r"[가-힣A-Za-z0-9&·+_-]{2,}", str(name or "")))
    quoted_terms = _quoted_entity_terms(evidence_text)
    terms.extend(quoted_terms)
    for match in re.finditer(
        r"[가-힣A-Za-z0-9&·+_-]{2,30}(?:\s+[가-힣A-Za-z0-9&·+_-]{2,30}){0,4}"
        r"(?:사업|시스템|서비스|플랫폼|센터|인프라|계약|협약|전환|현대화|실적)",
        str(evidence_text or ""),
        flags=re.IGNORECASE,
    ):
        terms.append(match.group(0))
    return _unique_texts(terms, max_items=12)


def _issue_object_nature_hints(
    name: str,
    evidence_text: str,
    *,
    type_from_evidence: str,
) -> list[str]:
    """Return broad hints so the prompt sees analysis lenses, not fixed templates."""
    value = f"{name} {evidence_text}"
    hints: list[str] = []
    broad_patterns = (
        ("performance_like", r"매출|영업이익|순이익|수익성|실적"),
        ("relationship_like", r"협약|MOU|협력|공동|파트너"),
        ("contract_like", r"계약|수주|공급|체결"),
        ("build_or_operation_like", r"선정|구축|운영|확보|개시|완료"),
        ("system_or_process_change_like", r"전환|현대화|개편|고도화|자동화"),
        ("service_or_platform_like", r"서비스|플랫폼|솔루션|공개|출시"),
    )
    for hint, pattern in broad_patterns:
        if re.search(pattern, value, flags=re.IGNORECASE):
            hints.append(hint)
    if type_from_evidence and type_from_evidence not in hints:
        hints.append(f"{type_from_evidence}_hint")
    return hints[:5]


def _issue_frame_target_scope(
    integrated_issue: dict[str, Any],
    *,
    products_or_services: list[str],
    customers_or_industries: list[str],
    frame_kind: str = "",
    focus_text: str = "",
) -> dict[str, Any]:
    evidence_text = focus_text or _integrated_grounding_text(integrated_issue)
    if frame_kind == "performance":
        return {
            "target_work": ["매출", "이익", "수익성 지표"],
            "target_system": [],
            "target_customer_or_industry": [],
            "geography_or_market": _market_scope_terms(evidence_text),
        }
    return {
        "target_work": _target_work_terms(evidence_text),
        "target_system": _unique_texts(
            [
                term
                for term in products_or_services
                if re.search(r"시스템|센터|인프라|플랫폼|서비스|단말|로봇|GPU", term, re.IGNORECASE)
            ],
            max_items=8,
        ),
        "target_customer_or_industry": customers_or_industries[:8],
        "geography_or_market": _market_scope_terms(evidence_text),
    }


def _issue_frame_numbers(
    integrated_issue: dict[str, Any],
    *,
    evidence_ids: list[str],
) -> list[dict[str, Any]]:
    numbers: list[dict[str, Any]] = []
    for index, item in enumerate(_jsonish_list(integrated_issue.get("key_numbers"))[:10]):
        if isinstance(item, dict):
            value = str(item.get("value") or item.get("number") or item.get("raw") or "").strip()
            if not value:
                continue
            numbers.append(
                {
                    "metric": str(item.get("metric") or item.get("label") or "").strip(),
                    "value": value,
                    "unit": str(item.get("unit") or "").strip(),
                    "meaning": str(item.get("meaning") or item.get("context") or "").strip(),
                    "evidence_id": str(item.get("fact_id") or item.get("evidence_id") or "").strip()
                    or (evidence_ids[index] if index < len(evidence_ids) else ""),
                }
            )
        else:
            text = str(item or "").strip()
            if text:
                numbers.append(
                    {
                        "metric": "",
                        "value": text,
                        "unit": "",
                        "meaning": "",
                        "evidence_id": evidence_ids[index] if index < len(evidence_ids) else "",
                    }
                )
    if numbers:
        return numbers[:10]
    evidence_text = _integrated_grounding_text(integrated_issue)
    for index, token in enumerate(_NUMERIC_TOKEN_PATTERN.findall(evidence_text)[:8]):
        numbers.append(
            {
                "metric": "",
                "value": str(token).strip(),
                "unit": "",
                "meaning": "raw_numeric_token_from_evidence",
                "evidence_id": evidence_ids[index] if index < len(evidence_ids) else "",
            }
        )
    return numbers


def _issue_terms_for_profile_matching(
    integrated_issue: dict[str, Any],
    *,
    products_or_services: list[str],
    customers_or_industries: list[str],
) -> list[dict[str, Any]]:
    issue_terms = _unique_texts(
        [
            *products_or_services,
            *customers_or_industries,
            *_target_work_terms(_integrated_grounding_text(integrated_issue)),
        ],
        max_items=12,
    )
    return [
        {
            "term": term,
            "source": "current_issue",
            "matching_note": (
                "프로필과 이미 연결됐다는 뜻이 아니라, 프로필 매칭에 사용할 현재 사건 용어입니다."
            ),
        }
        for term in issue_terms[:8]
    ]


def _fallback_action_label(evidence_text: str) -> str:
    value = str(evidence_text or "")
    for label, pattern in (
        ("selection", r"사업자\s*선정|최종\s*선정|참여\s*기업.{0,12}선정"),
        ("contract", r"계약\s*체결|수주|공급\s*계약"),
        ("partnership", r"협약|MOU|공동\s*추진|협력"),
        ("launch", r"정식\s*출시|서비스\s*오픈|개시|공개"),
        ("build_or_operation", r"구축\s*완료|운영\s*시작|구축한다|운영한다"),
        ("performance", r"매출|영업이익|실적"),
    ):
        if re.search(pattern, value, re.IGNORECASE):
            return label
    return "unclear"


def _products_from_facts(integrated_issue: dict[str, Any]) -> list[str]:
    out: list[str] = []
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if not isinstance(intelligence, dict):
        return out
    for group_name in ("common_facts", "unique_facts"):
        for item in _jsonish_list(intelligence.get(group_name)):
            if isinstance(item, dict):
                out.extend(_string_list(item.get("products_or_services"), max_items=8))
    return _unique_texts(out, max_items=12)


def _customers_from_facts(integrated_issue: dict[str, Any]) -> list[str]:
    out: list[str] = []
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if not isinstance(intelligence, dict):
        return out
    for group_name in ("common_facts", "unique_facts"):
        for item in _jsonish_list(intelligence.get(group_name)):
            if isinstance(item, dict):
                out.extend(_string_list(item.get("customers_or_industries"), max_items=8))
    return _unique_texts(out, max_items=12)


def _target_work_terms(text: str) -> list[str]:
    value = str(text or "")
    candidates = re.findall(
        r"[가-힣A-Za-z0-9&·+_-]{2,30}(?:\s+[가-힣A-Za-z0-9&·+_-]{2,30}){0,3}"
        r"(?:전환|현대화|구축|운영|자동화|확보|개시|협약|계약|선정|출시|공개)",
        value,
        flags=re.IGNORECASE,
    )
    return _unique_texts([_clean_phrase(item) for item in candidates], max_items=8)


def _market_scope_terms(text: str) -> list[str]:
    value = str(text or "")
    scopes: list[str] = []
    for term in ("국가", "국내", "공공", "민간", "금융권", "물류", "제조", "글로벌", "해외"):
        if re.search(re.escape(term), value, re.IGNORECASE):
            scopes.append(term)
    return scopes


def _unique_texts(values: list[Any], *, max_items: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_phrase(str(value or ""))
        if not text:
            continue
        key = re.sub(r"\s+", " ", text).casefold()
        if key in seen:
            continue
        out.append(text)
        seen.add(key)
        if len(out) >= max_items:
            break
    return out


def _clean_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip(" .。,:;，"))


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
            "프로젝트 추진/참여/제공/공급/지원/운영/확장 주체처럼 쓰지 마세요.",
            "- 타깃 피어는 계약 상대방, 사업 범위, 계약 범위/기간이 확인된 피어로만 설명하세요.",
            "- 다만 '연결성 확인'으로 끝내지 말고, 확인된 사업/제품명이 어떤 산업 과제나 "
            "유사 사업의 비교 기준을 드러내는지까지 해석하세요.",
            "- 이 모드의 좋은 해석은 '계약 사실 → 사업명에 드러난 대상 업무/시스템과 "
            "전환·검증·운영 성격 → 피어 프로필 사업영역 접점' 순서입니다.",
            "- 문장 주어는 가능한 '이번 계약', '해당 사업', '확인된 계약 범위'처럼 "
            "사건/사업명으로 두세요. 타깃 피어를 주어로 두고 참여·추진·제공·수행·확장한다고 "
            "쓰지 마세요.",
            "- '중요성', '필요성', '관련이 깊습니다', '기회로 작용합니다' 같은 결론형 "
            "표현으로 끝내지 말고, 어떤 범위·기간·전환 성격이 확인됐는지 씁니다.",
            "- 공급사 매출 비율은 요약의 계약 규모 근거일 뿐, "
            "타깃 피어의 역량/성과/전략 근거가 아닙니다.",
            "- SK AX recommended_actions 는 타깃 피어 프로젝트에 직접 제안하는 문장이 아니라 "
            "유사 동향을 내부에서 관찰·비교·추적하는 체크포인트로 쓰세요.",
            "- SK AX recommended_actions 는 '피어 신호 → 비교해야 할 축 → 후속 확인 데이터 → "
            "SK AX 내부 점검 관점' 순서로 2~3문장 작성하세요.",
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


def _profile_for_prompt(
    profile: dict[str, Any],
    *,
    integrated_issue: dict[str, Any] | None = None,
    relevance_hint_text: str = "",
    include_financial_context: bool = False,
) -> dict[str, Any]:
    skax = profile.get("skax_profile") or {}
    peer_profiles = profile.get("peer_profiles") or {}
    relevance_tokens = _issue_relevance_tokens(
        integrated_issue or {},
        extra_text=relevance_hint_text,
    )
    out = {
        "skax_profile": _shrink_profile(skax, relevance_tokens=relevance_tokens),
        "peer_profiles": {
            str(peer_id): _shrink_profile(payload, relevance_tokens=relevance_tokens)
            for peer_id, payload in (
                peer_profiles.items() if isinstance(peer_profiles, dict) else []
            )
        },
        "sector_context": profile.get("sector_context") or {},
    }
    if include_financial_context:
        financial_context = _financial_profile_context_for_prompt(
            skax=skax,
            peer_profiles=peer_profiles,
        )
        if financial_context:
            out["financial_profile_context"] = financial_context
    return out


def _financial_profile_context_for_prompt(
    *,
    skax: Any,
    peer_profiles: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    skax_financial = _compact_financial_profile_context(skax)
    if skax_financial:
        out["skax_profile"] = skax_financial
    if isinstance(peer_profiles, dict):
        peer_out = {
            str(peer_id): compacted
            for peer_id, payload in peer_profiles.items()
            if (compacted := _compact_financial_profile_context(payload))
        }
        if peer_out:
            out["peer_profiles"] = peer_out
    return out


def _compact_financial_profile_context(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    keys = (
        "company_id",
        "company_name",
        "company_name_ko",
        "financial_summary",
        "operational_highlights",
        "investment_roadmap",
        "market_view",
        "validation",
    )
    out: dict[str, Any] = {}
    for key in keys:
        value = profile.get(key)
        if value in ({}, [], "", None):
            continue
        compacted = _compact_value(value)
        if compacted not in ({}, [], "", None):
            out[key] = compacted
    return out


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


def _analysis_context_for_model(
    context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    relevance_hint_text: str = "",
    include_financial_context: bool = False,
) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    relevance_tokens = _issue_relevance_tokens(
        integrated_issue,
        extra_text=relevance_hint_text,
    )
    out = dict(context)
    if not include_financial_context:
        out["financial_trend"] = {}
    for key, limit in (
        ("peer_event_timeline_recent", 8),
        ("sector_pulse_recent", 4),
        ("event_chain_candidates", 5),
        ("similar_cards_rag", 5),
    ):
        out[key] = _relevant_context_items(
            out.get(key),
            relevance_tokens=relevance_tokens,
            max_items=limit,
        )
    return out


def _relevant_context_items(
    value: Any,
    *,
    relevance_tokens: set[str],
    max_items: int,
) -> list[Any]:
    if not isinstance(value, list):
        return []
    if not relevance_tokens:
        return value[:max_items]
    scored: list[tuple[int, int, Any]] = []
    for index, item in enumerate(value):
        score = _profile_relevance_score(item, relevance_tokens)
        if score > 0:
            scored.append((score, -index, item))
    scored.sort(reverse=True)
    return [item for _, _, item in scored[:max_items]]


def _context_availability_for_prompt(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    analysis_context: dict[str, Any],
) -> dict[str, Any]:
    """Expose whether profile/recent context is usable without generating copy."""
    company_ids = _companies_from_integrated_issue(integrated_issue)
    peer_profiles = profile_context.get("peer_profiles") or {}
    matched_peer_profiles: list[dict[str, Any]] = []
    if isinstance(peer_profiles, dict):
        for company_id in company_ids:
            profile = peer_profiles.get(company_id) or {}
            if isinstance(profile, dict):
                linkage = _peer_profile_linkage(
                    profile_context,
                    integrated_issue=integrated_issue,
                    company_id=company_id,
                )
                matched_peer_profiles.append(
                    {
                        "company_id": company_id,
                        "available": _has_profile_context(profile),
                        "profile_fields": _available_profile_fields(profile),
                        "peer_profile_linkage": linkage,
                    }
                )

    skax_profile = profile_context.get("skax_profile") or {}
    recent_layers = _available_analysis_layers(analysis_context)
    return {
        "matched_peer_profiles": matched_peer_profiles,
        "peer_profile_available": any(item["available"] for item in matched_peer_profiles),
        "skax_profile_available": _has_profile_context(skax_profile),
        "skax_profile_fields": _available_profile_fields(skax_profile),
        "recent_context_layers_available": recent_layers,
        "recent_context_available": bool(recent_layers),
        "guidance": [
            (
                "profile_based_implication_requires_current_fact_plus_peer_profile"
                "_plus_recent_context_when_available"
            ),
            (
                "if_relevant_profile_or_recent_context_is_missing_lower_confidence"
                "_instead_of_fabricating_profile_based_claims"
            ),
            (
                "skax_checkpoints_require_current_signal_plus_comparison_axis"
                "_plus_follow_up_data; skax_profile_linkage_is_optional"
                "_and_only_limits_specific_capability_claims"
            ),
        ],
    }


def _has_profile_context(profile: dict[str, Any]) -> bool:
    if not isinstance(profile, dict):
        return False
    return any(
        bool(profile.get(key))
        for key in (
            "one_liner",
            "company_summary",
            "key_products_services",
            "execution_cases",
            "strategic_focus",
            "priority_initiatives",
            "business_areas",
            "core_capabilities",
            "recent_changes",
            "capability_evolution",
            "market_view",
        )
    )


def _available_profile_fields(profile: dict[str, Any]) -> list[str]:
    if not isinstance(profile, dict):
        return []
    keys = (
        "one_liner",
        "company_summary",
        "key_products_services",
        "execution_cases",
        "strategic_focus",
        "priority_initiatives",
        "business_areas",
        "core_capabilities",
        "recent_changes",
        "capability_evolution",
        "market_view",
    )
    return [key for key in keys if profile.get(key)]


def _available_analysis_layers(context: dict[str, Any]) -> list[str]:
    if not isinstance(context, dict):
        return []
    candidate_keys = (
        "peer_event_timeline_recent",
        "sector_pulse_recent",
        "financial_trend",
        "event_chain_candidates",
        "similar_cards_rag",
        "evidence_density_per_peer",
    )
    return [key for key in candidate_keys if bool(context.get(key))]


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


def _should_include_financial_profile_context(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    bundle: dict[str, Any],
) -> bool:
    """Gate profile-level financial/IR context separately from current-event numbers."""
    source_parts = [
        integrated_issue.get("issue_source_type"),
        integrated_issue.get("source_type"),
        bundle.get("source_type"),
        (bundle.get("metadata") or {}).get("source_type") if isinstance(bundle, dict) else None,
    ]
    source_text = " ".join(str(item or "").strip().lower() for item in source_parts)
    if re.search(r"\b(dart|ir|securities_report|securities|financial_report)\b", source_text):
        return True

    event_parts = [
        integrated_issue.get("cluster_event_type"),
        classification.get("event_type"),
        bundle.get("event_type"),
        _json_dumps(classification.get("event_type_scores") or {}),
    ]
    event_text = " ".join(str(item or "").strip().lower() for item in event_parts)
    if re.search(
        r"(실적|재무|매출\s*변화|재무\s*지표|공급\s*계약|공급계약|투자|지분|"
        r"earnings|financial|revenue_change|financial_metric|supply_contract|investment)",
        event_text,
    ):
        return True
    evidence_text = _json_dumps(
        {
            "fact_summary": integrated_issue.get("fact_summary") or [],
            "summary_lines": integrated_issue.get("summary_lines") or [],
            "consolidated_facts": integrated_issue.get("consolidated_facts") or [],
            "key_numbers": integrated_issue.get("key_numbers") or [],
            "cluster_fact_intelligence": integrated_issue.get("cluster_fact_intelligence") or {},
        }
    ).lower()
    if re.search(
        r"(매출|영업\s*이익|순이익|영업\s*이익률|수익성|실적|재무|"
        r"계약\s*금액|공급\s*계약|수주\s*금액|최근\s*매출액|"
        r"revenue|operating\s*profit|net\s*income|margin|earnings|financial)",
        evidence_text,
    ):
        return True
    return False


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


def _business_line_candidate_details(
    profile: dict[str, Any],
    *,
    integrated_issue: dict[str, Any] | None = None,
    relevance_hint_text: str = "",
) -> list[dict[str, Any]]:
    skax = profile.get("skax_profile") or {}
    relevance_tokens = _issue_relevance_tokens(
        integrated_issue or {},
        extra_text=relevance_hint_text,
    )
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for name in _string_list(skax.get("business_lines"), max_items=20):
        if name in seen:
            continue
        candidates.append({"name": name})
        seen.add(name)

    business_areas = skax.get("business_areas") or []
    if relevance_tokens:
        ranked_areas = _rank_relevant_profile_items(
            business_areas,
            relevance_tokens=relevance_tokens,
            max_items=20,
        )
        business_areas = ranked_areas or business_areas
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
        if not _is_follow_up_or_watch_field(label):
            for pattern in _UNSUPPORTED_CLAIM_PATTERNS:
                if _has_unsupported_pattern(
                    value_text,
                    pattern,
                    evidence_text=integrated_evidence_text,
                ):
                    violations.append(
                        f"{label}: 입력 근거 없이 `{pattern}` 계열 표현을 사용했습니다."
                    )
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
        evidence_scoped_claim_violation = _evidence_scoped_business_claim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if evidence_scoped_claim_violation:
            violations.append(f"{label}: {evidence_scoped_claim_violation}")
        scope_expansion_violation = _scope_expansion_guard_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if scope_expansion_violation:
            violations.append(f"{label}: {scope_expansion_violation}")
        action_quality_violation = _recommended_action_quality_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if action_quality_violation:
            violations.append(f"{label}: {action_quality_violation}")
        generic_insight_violation = _generic_insight_quality_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if generic_insight_violation:
            violations.append(f"{label}: {generic_insight_violation}")
        reasoning_flow_violation = _reasoning_flow_quality_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if reasoning_flow_violation:
            violations.append(f"{label}: {reasoning_flow_violation}")
        unsupported_profile_violation = _unsupported_peer_profile_claim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if unsupported_profile_violation:
            violations.append(f"{label}: {unsupported_profile_violation}")
        frame_mismatch_violation = _issue_frame_mismatch_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if frame_mismatch_violation:
            violations.append(f"{label}: {frame_mismatch_violation}")
    return list(dict.fromkeys(violations))


def _quality_failure_is_mechanical_repair_candidate(violations: list[str]) -> bool:
    if not violations:
        return False
    recoverable_markers = (
        "현재 사건과 직접 맞는 피어 프로필 접점",
        "체크포인트에 현재 사건",
        "실적형 이슈",
        "SK AX 내부 관점",
        "비교 축이나 후속 확인 기준",
        "현재 사건의 고유 근거",
        "근거 흐름 부족",
    )
    return all(
        any(marker in violation for marker in recoverable_markers) for violation in violations
    )


def _quality_failure_can_fallback_to_issue_frame(
    violations: list[str],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> bool:
    if not violations or not _profile_context_is_sparse(profile_context):
        return False
    if _issue_frame_kind(integrated_issue) not in {
        "performance",
        "launch_or_service",
        "selection_or_build",
    }:
        return False
    nonrecoverable_markers = (
        "시장 선점",
        "점유율",
        "차별화된 가치",
        "기회를 창출",
        "기술적 역량을 입증",
    )
    joined = " / ".join(violations)
    return not any(marker in joined for marker in nonrecoverable_markers)


def _profile_context_is_sparse(profile_context: dict[str, Any]) -> bool:
    if not isinstance(profile_context, dict) or not profile_context:
        return True
    candidates: list[Any] = [profile_context.get("skax_profile")]
    peer_profiles = profile_context.get("peer_profiles")
    if isinstance(peer_profiles, dict):
        candidates.extend(peer_profiles.values())
    elif isinstance(peer_profiles, list):
        candidates.extend(peer_profiles)
    signal_fields = (
        "business_areas",
        "core_capabilities",
        "key_products_services",
        "execution_cases",
        "strategic_focus",
        "recent_changes",
        "capability_evolution",
        "business_lines",
    )
    for profile in candidates:
        if not isinstance(profile, dict):
            continue
        if any(_has_nonempty_profile_signal(profile.get(field)) for field in signal_fields):
            return False
    return True


def _has_nonempty_profile_signal(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_nonempty_profile_signal(item) for item in value)
    if isinstance(value, dict):
        return any(_has_nonempty_profile_signal(item) for item in value.values())
    return value is not None


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


def _unsupported_claim_pattern_violation(text: str, *, integrated_issue: dict[str, Any]) -> bool:
    evidence_text = _integrated_grounding_text(integrated_issue)
    return any(
        _has_unsupported_pattern(text, pattern, evidence_text=evidence_text)
        for pattern in _UNSUPPORTED_CLAIM_PATTERNS
    )


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


def _issue_frame_mismatch_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or _is_follow_up_or_watch_field(label):
        return ""
    frame_kind = _issue_frame_kind(integrated_issue)
    if frame_kind != "performance":
        return ""
    value = str(text or "")
    focus_text = _issue_focus_text(integrated_issue)
    performance_terms = r"매출|영업이익|순이익|영업이익률|실적|수익성|이익|전사\s*재무"
    if re.search(performance_terms, value):
        return ""
    operational_terms = (
        r"물류|자동화|계약\s*범위|대상\s*시스템|전환\s*범위|업무\s*영향도|"
        r"구축\s*범위|운영\s*책임|일정\s*기준|시스템\s*전환|기술적\s*성과|"
        r"기술\s*도입|기술.{0,8}기여|기술.{0,8}영향"
    )
    if re.search(operational_terms, value) and not re.search(operational_terms, focus_text):
        return (
            "실적형 이슈를 물류/계약/시스템 전환 과제처럼 해석했습니다. "
            "매출·이익·이익률과 사업부 기여도 확인 여부 중심으로 낮춰야 합니다."
        )
    return ""


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
    if not re.search(
        r"성장|시장\s*입지|중요한\s*매출원|성과|시장\s*반응|"
        r"매출\s*(기여|확대|성장|영향)|매출.{0,16}영향",
        text,
    ):
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
    if label.startswith("skax_implication"):
        target_patterns = _target_name_patterns(str(integrated_issue.get("main_company") or ""))
        if not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in target_patterns):
            return ""
    overclaim_pattern = (
        r"역량[이가을를\s]*(강화|확장)|"
        r"경쟁력[이가을를\s]*강화|"
        r"입지[가를\s]*강화|"
        r"역할[이가을를\s]*강화|"
        r"사업\s*영역[이가을를\s]*(확장|확대)|"
        r"사업\s*범위[가를이은을\s]*(확장|확대|넓)|"
        r"제공\s*범위[가를\s]*(확장|확대|넓)|"
        r"운영\s*(안정성|안정화)[이가을를\s]*(확보|강화)|"
        r"시스템\s*전환.{0,20}운영\s*(안정성|안정화)"
    )
    if not re.search(overclaim_pattern, text):
        return ""
    return (
        "타깃 피어가 계약 상대방으로 보이는 계약을 역량 강화/경쟁력 강화 성과처럼 "
        "단정했습니다. 계약 범위/사업영역 접점 수준으로 낮춰야 합니다."
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
        r"프로젝트[가-힣\s]*(추진|참여|제공|공급|수행|구축|운영|지원|확장|확대|통해|기여)",
        r"사업[가-힣\s]*(추진|참여|제공|공급|수행|구축|운영|지원|확장|확대)",
        r"(시스템|서비스|솔루션)[가-힣\s]*(추진|참여|제공|공급|수행|구축|운영|지원|확장|확대|집중|기여)",
        r"(솔루션|서비스|기회)[이가를은\s]*.{0,30}(제공|지원|기여|보여)",
        r"(제공|공급|수행|구축|운영|지원)\s*역량",
        r"디지털\s*전환[을를\s]*지원",
    )
    target_near_direct_role = any(
        re.search(
            rf"({target_pattern}).{{0,32}}({role_pattern})|"
            rf"({role_pattern}).{{0,32}}({target_pattern})",
            text,
            flags=re.IGNORECASE,
        )
        for target_pattern in target_patterns
        for role_pattern in direct_role_patterns
    )
    profile_background_statement = (
        label.startswith(("peer_implication.peer_meaning", "peer_implication.capability_change"))
        and re.search(
            r"(제공하는|보유한)\s*기업|제공\s*역량|기존\s*(사업영역|역량)|프로필\s*접점",
            text,
        )
        and not re.search(r"(프로젝트|사업|계약)[가-힣\s]*(제공|수행|참여|추진|운영)", text)
    )
    if target_near_direct_role and not profile_background_statement:
        return (
            "타깃 피어가 계약 상대방/고객 슬롯에 있는데 피어의 프로젝트 실행이나 "
            "공급자 행동처럼 썼습니다. 계약 범위/사업영역 접점/관찰 지점으로 낮춰야 합니다."
        )

    if label.startswith("peer_implication.capability_change") and re.search(
        r"(프로젝트|사업|계약)[가-힣\s]*(통해|참여|추진|수행|기여|제공|충족|지원)|"
        r"(효율성|안정성)[가-힣\s]*(높|개선|확보)",
        text,
    ):
        return (
            "계약 상대방 피어의 capability_change 를 프로젝트 수행 성과처럼 썼습니다. "
            "확인된 사업 범위/대상 시스템/프로필 접점으로 낮춰야 합니다."
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
        return (
            "SK AX 체크포인트가 외부 실행 산출물 표현에 머물렀습니다. "
            "현재 사건 신호, 비교 축, 후속 확인 데이터, 내부 점검 관점으로 바꿔야 합니다."
        )
    return ""


def _recommended_action_quality_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any] | None = None,
    profile_context: dict[str, Any] | None = None,
) -> str:
    if not text or not label.startswith("skax_implication.recommended_actions"):
        return ""
    evidence_text = _integrated_grounding_text(integrated_issue or {})
    if re.search(r"주가|거래를\s*마쳤|시장\s*반응|투자자\s*반응", text):
        return (
            "체크포인트가 주가/시장 반응을 핵심 근거로 사용했습니다. 현재 사건의 "
            "대상 사업, 역할, 수치, 후속 확인 데이터 중심으로 작성해야 합니다."
        )
    if re.search(r"클라우드|AI|인공지능|에이아이", text, flags=re.IGNORECASE) and not re.search(
        r"클라우드|AI|인공지능|에이아이",
        evidence_text,
        flags=re.IGNORECASE,
    ):
        return (
            "현재 사건 근거에 없는 기술명을 체크포인트에 사용했습니다. 대상 사업·시스템, "
            "역할 범위, 수치, 일정, 후속 확인 데이터 중심으로 낮춰야 합니다."
        )
    if re.search(r"성공|수주에\s*영향|신뢰성", text) and not _profile_has_execution_case(
        profile_context,
        integrated_issue=integrated_issue,
    ):
        return (
            "ProfileContext에 실행 사례 근거가 없는데 성공/수주 영향/신뢰성을 사용했습니다. "
            "현재 사건에서 확인되는 역할·수치·범위와 후속 확인 데이터로 낮춰야 합니다."
        )
    if re.search(r"솔루션|성능.{0,12}(강조|입증|검증|확인)|검증된\s*성능", text):
        return (
            "대응방향이 솔루션/성능 강조 같은 일반 표현에 머물렀습니다. "
            "현재 사건의 대상 사업·시스템, 비교 축, 후속 확인 데이터, 내부 점검 관점으로 "
            "낮춰야 합니다."
        )
    if re.search(r"경쟁력[을를이가\s]*확보|전략에\s*반영|방안[을를]?\s*모색", text):
        return (
            "대응방향이 경쟁력 확보/전략 반영 같은 일반 결론으로 끝났습니다. "
            "현재 사건의 어떤 사실이 어떤 내부 비교 기준을 만들었는지와 "
            "후속으로 확인할 데이터를 함께 써야 합니다."
        )
    off_topic_product_violation = _off_topic_application_product_violation(
        text,
        integrated_issue=integrated_issue or {},
    )
    if off_topic_product_violation:
        return off_topic_product_violation
    evidence_scoped_violation = _evidence_scoped_business_claim_violation(
        text,
        label=label,
        integrated_issue=integrated_issue,
        profile_context=profile_context,
    )
    if evidence_scoped_violation:
        return evidence_scoped_violation
    structural_violation = _recommended_action_checkpoint_violation(
        text,
        integrated_issue=integrated_issue or {},
    )
    if structural_violation:
        return structural_violation
    return ""


def _generic_insight_quality_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or _is_follow_up_or_watch_field(label):
        return ""
    if not _GENERIC_INSIGHT_PATTERN.search(text):
        return ""
    if _text_has_specific_issue_grounding(text, integrated_issue=integrated_issue):
        return ""
    return (
        "문장이 일반 표현에 머물렀고 현재 사건의 적용 업무/시스템/서비스/수치/관계가 "
        "함께 보이지 않습니다. 현재 사건의 고유 근거와 연결해 다시 작성해야 합니다."
    )


def _reasoning_flow_quality_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    """Require evidence flow for interpretive claims without hard-coding domains."""
    if not text or _is_follow_up_or_watch_field(label):
        return ""
    value = str(text or "")
    peer_field = label.startswith(
        (
            "analysis.analysis_summary",
            "analysis.strategic_meaning",
            "analysis.market_signal",
            "analysis.impact_reason",
            "analysis.reason",
            "peer_implication.peer_meaning",
            "peer_implication.capability_change",
        )
    )
    skax_field = label.startswith(
        (
            "skax_implication.why_important",
            "skax_implication.potential_impact",
            "skax_implication.recommended_actions",
        )
    )
    if not (peer_field or skax_field):
        return ""

    has_issue_grounding = _text_has_specific_issue_grounding(
        value,
        integrated_issue=integrated_issue,
    )
    has_reason_marker = _has_reasoning_marker(value)
    if peer_field:
        profile_terms = _profile_terms_for_quality(
            profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
        has_profile_grounding = _text_overlaps_issue_terms(value, profile_terms)
        cautious_when_weak = bool(
            re.search(r"단정하기\s*어렵|후속\s*확인|관찰\s*신호|근거가\s*(약|부족|제한)", value)
        )
        has_metric_effect_support = _has_effect_metric_support(
            value,
            integrated_issue=integrated_issue,
        )
        if _has_unexplained_observation_phrase(value) and not (
            has_issue_grounding
            and (has_profile_grounding or cautious_when_weak)
            and has_reason_marker
        ):
            return (
                "근거 흐름 부족: 기존 사업/관찰 신호 표현을 썼지만 현재 사건의 구체 사실, "
                "연결되는 프로필 근거, 왜 단정하지 않는지 또는 왜 연결되는지가 "
                "함께 보이지 않습니다."
            )
        if _has_strong_interpretive_claim(value) and not (
            has_issue_grounding
            and (has_profile_grounding or has_metric_effect_support)
            and has_reason_marker
        ):
            return (
                "근거 흐름 부족: 기회·역량 강화·효과 같은 강한 해석을 썼지만 현재 사건 사실과 "
                "관련 프로필 근거가 문장 안에서 논리적으로 연결되지 않았습니다."
            )
    if skax_field:
        if _has_vague_skax_comparison(value) and not (
            has_issue_grounding and has_reason_marker and _has_skax_decision_object(value)
        ):
            return (
                "근거 흐름 부족: SK AX 체크포인트가 무엇을 왜 비교해야 하는지 부족합니다. "
                "현재 사건 신호, 비교 축, 그 축이 바꾸는 내부 판단, "
                "후속 확인 데이터를 연결해야 합니다."
            )
    return ""


def _has_unexplained_observation_phrase(text: str) -> bool:
    return bool(
        re.search(
            r"기존\s*사업\s*(흐름|맥락)|관찰\s*(지점|신호)|연결\s*(지점|사례|흐름)|"
            r"이어지는\s*흐름|비교할\s*지점|배경으로\s*확인",
            str(text or ""),
        )
    )


def _has_strong_interpretive_claim(text: str) -> bool:
    return bool(
        re.search(
            r"기회[를가은\s]*(얻|확보|생기)|"
            r"역량[이가을를\s]*(강화|높|확장)|"
            r"표준[을를\s]*(제시|만들)|"
            r"효율성[을를이가\s]*(높|향상|개선)|"
            r"긍정적\s*영향|"
            r"성과[를가\s]*(확인|입증|검증)",
            str(text or ""),
        )
    )


def _has_effect_metric_support(text: str, *, integrated_issue: dict[str, Any]) -> bool:
    """Allow effect/efficiency claims only when the issue itself carries metrics."""
    value = str(text or "")
    if not re.search(r"효율|단축|감소|절감|개선|성과|처리\s*시간|적응\s*기간", value):
        return False
    grounding = _integrated_grounding_text(integrated_issue)
    if not grounding:
        return False
    has_effect_language = bool(
        re.search(r"효율|단축|감소|절감|개선|성과|처리\s*시간|적응\s*기간", grounding)
    )
    has_metric = bool(
        re.search(r"\d[\d,.]*\s*(?:시간|분|개월|주|일|%|퍼센트|배|건|명|억원|원)", grounding)
    )
    return has_effect_language and has_metric


def _has_vague_skax_comparison(text: str) -> bool:
    return bool(
        re.search(
            r"같은\s*기준으로\s*축적|비교해야\s*합니다|구분해야\s*합니다|"
            r"점검해야\s*합니다|모니터링해야\s*합니다|조정할\s*수\s*있습니다|"
            r"대응\s*가능\s*범위",
            str(text or ""),
        )
    )


def _has_reasoning_marker(text: str) -> bool:
    return bool(
        re.search(
            r"때문|근거|보여|의미|따라서|다만|단정|확인되어야|그래야|"
            r"이\s*(기준|구분|정보|데이터)|판단|비교|역할|책임|성과|리스크|"
            r"가능성|범위|왜",
            str(text or ""),
        )
    )


def _has_skax_decision_object(text: str) -> bool:
    return bool(
        re.search(
            r"사업화\s*가능성|운영\s*책임|성과\s*검증|역할\s*분담|책임\s*구조|"
            r"수행\s*범위|보완\s*영역|리스크\s*판단|내부\s*판단|적용\s*업무|"
            r"사용\s*대상|효과\s*수치|대상\s*시스템|후속\s*데이터|관계\s*수준|"
            r"과제\s*전환|후속\s*공시|고객\s*접점|검증\s*환경|운영\s*조건",
            str(text or ""),
        )
    )


def _profile_terms_for_quality(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> list[str]:
    profiles = _profiles_for_quality(
        profile_context,
        integrated_issue=integrated_issue,
        scope=scope,
    )
    terms: list[str] = []
    for profile in profiles:
        for key in (
            "business_areas",
            "core_capabilities",
            "key_products_services",
            "strategic_focus",
            "execution_cases",
            "evidence_digest",
            "profile_linkage_evaluation",
            "skax_response_linkage",
        ):
            terms.extend(_profile_text_terms(profile.get(key), depth=0))
    return _unique_texts(
        [term for term in terms if _profile_quality_term_is_signal(term)], max_items=40
    )


def _profiles_for_quality(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> list[dict[str, Any]]:
    if not isinstance(profile_context, dict):
        return []
    if scope == "skax":
        skax = profile_context.get("skax_profile") or {}
        return [skax] if isinstance(skax, dict) else []
    peer_profiles = profile_context.get("peer_profiles") or {}
    if not isinstance(peer_profiles, dict):
        return []
    companies = _companies_from_integrated_issue(integrated_issue)
    profiles: list[dict[str, Any]] = []
    for company_id in companies:
        profile = peer_profiles.get(company_id)
        if isinstance(profile, dict):
            profiles.append(profile)
    if profiles:
        return profiles
    return [profile for profile in peer_profiles.values() if isinstance(profile, dict)]


def _profile_text_terms(value: Any, *, depth: int) -> list[str]:
    if depth > 4:
        return []
    terms: list[str] = []
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value).strip()
        if text:
            terms.append(text)
            terms.extend(_content_tokens(text))
        return terms
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in {
                "source",
                "source_ref",
                "source_refs",
                "schema_version",
                "generated_at",
                "confidence",
                "status",
            }:
                continue
            terms.extend(_profile_text_terms(item, depth=depth + 1))
        return terms
    if isinstance(value, list):
        for item in value[:12]:
            terms.extend(_profile_text_terms(item, depth=depth + 1))
    return terms


def _profile_quality_term_is_signal(term: str) -> bool:
    value = re.sub(r"\s+", " ", str(term or "").strip())
    if len(value) < 3:
        return False
    if re.fullmatch(r"[A-Za-z]{1,3}", value):
        return False
    lowered = value.casefold()
    if lowered in {"profile", "schema", "version", "summary", "source", "status"}:
        return False
    if value in {"사업", "서비스", "시스템", "플랫폼", "솔루션", "기술", "구축", "운영"}:
        return False
    return True


def _text_has_specific_issue_grounding(
    text: str,
    *,
    integrated_issue: dict[str, Any],
) -> bool:
    value = str(text or "")
    if not value:
        return False
    for token in _NUMERIC_TOKEN_PATTERN.findall(value):
        if _numeric_token_key(token) in _grounded_numeric_keys_for_issue(integrated_issue):
            return True
    specific_terms = _high_specific_issue_terms_for_quality(integrated_issue)
    matched_specific_terms = [
        term for term in specific_terms if _text_overlaps_issue_terms(value, [term])
    ]
    if len(matched_specific_terms) >= 2:
        return True
    if matched_specific_terms and any(len(term) >= 8 for term in matched_specific_terms):
        return True
    return False


def _specific_issue_terms_for_quality(integrated_issue: dict[str, Any]) -> list[str]:
    issue_frame = _issue_frame_for_prompt(integrated_issue)
    terms: list[str] = []
    terms.extend(_checkpoint_issue_terms(issue_frame))
    business_object = issue_frame.get("business_object") or {}
    terms.extend(_string_list(business_object.get("raw_terms"), max_items=20))
    terms.extend(_string_list(issue_frame.get("profile_matching_terms"), max_items=20))

    grounding_text = _integrated_grounding_text(integrated_issue)
    terms.extend(_NUMERIC_TOKEN_PATTERN.findall(grounding_text))
    terms.extend(re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,40})[\"'“”‘’]", grounding_text))
    for token in _content_tokens(grounding_text):
        if _is_specific_issue_token(token):
            terms.append(token)
    return _unique_texts(terms, max_items=40)


def _high_specific_issue_terms_for_quality(integrated_issue: dict[str, Any]) -> list[str]:
    issue_frame = _issue_frame_for_prompt(integrated_issue)
    terms: list[str] = []
    business_object = issue_frame.get("business_object") or {}
    terms.extend(_string_list(business_object.get("raw_terms"), max_items=20))
    target_scope = issue_frame.get("target_scope") or {}
    for key in (
        "target_work",
        "target_system",
        "target_customer_or_industry",
        "geography_or_market",
    ):
        terms.extend(_string_list(target_scope.get(key), max_items=20))

    grounding_text = _integrated_grounding_text(integrated_issue)
    terms.extend(_NUMERIC_TOKEN_PATTERN.findall(grounding_text))
    terms.extend(re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,40})[\"'“”‘’]", grounding_text))
    for token in _content_tokens(grounding_text):
        if _is_high_specific_issue_token(token):
            terms.append(token)
    return _unique_texts([term for term in terms if _clean_issue_axis_phrase(term)], max_items=40)


def _is_high_specific_issue_token(token: str) -> bool:
    value = str(token or "").strip()
    if len(value) < 5:
        return False
    if not _is_specific_issue_token(value):
        return False
    low_specific = {
        "물류센터",
        "업무협약",
        "공통업무",
        "인공지능",
        "클라우드",
        "데이터센터",
        "자동화",
        "컴퓨팅센터",
    }
    return value not in low_specific


def _is_specific_issue_token(token: str) -> bool:
    value = str(token or "").strip()
    if len(value) < 4:
        return False
    lowered = value.casefold()
    low_signal = {
        "서비스",
        "플랫폼",
        "시스템",
        "솔루션",
        "사업",
        "기술",
        "기업",
        "지원",
        "활용",
        "제공",
        "구축",
        "운영",
        "확대",
        "강화",
        "디지털",
        "전환",
        "업무",
        "효율성",
        "전략",
        "동향",
        "피어사",
        "경쟁사",
    }
    if lowered in {"lg", "cns", "sk", "ax", "sds"} or value in low_signal:
        return False
    if re.fullmatch(r"[A-Za-z]{2,4}", value):
        return False
    return True


def _recommended_action_checkpoint_violation(
    text: str,
    *,
    integrated_issue: dict[str, Any],
) -> str:
    value = str(text or "")
    if re.search(r"제안서|PoC|레퍼런스|운영\s*계획|영업\s*대응", value, re.IGNORECASE):
        return (
            "체크포인트가 외부 실행 산출물 표현에 머물렀습니다. SK AX 내부에서 관찰·비교·추적할 "
            "축과 후속 확인 데이터로 바꿔야 합니다."
        )
    if not re.search(r"SK\s*AX|자사|내부", value, re.IGNORECASE):
        return "체크포인트에 SK AX 내부 관점이 부족합니다."

    issue_frame = _issue_frame_for_prompt(integrated_issue)
    issue_terms = _checkpoint_issue_terms(issue_frame)
    if issue_terms and not _text_overlaps_issue_terms(value, issue_terms):
        return "체크포인트에 현재 사건의 대상 사업·서비스·시스템·고객군 신호가 연결되지 않았습니다."
    if not re.search(
        r"비교|관찰|추적|모니터링|후속|확인|점검|구분|조정|축적|범위|역할|수치|일정|"
        r"책임|리스크|조건|구조|변동",
        value,
    ):
        return "체크포인트에 비교 축이나 후속 확인 기준이 부족합니다."
    return ""


def _checkpoint_issue_terms(issue_frame: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    business_object = issue_frame.get("business_object") or {}
    if business_object.get("name"):
        terms.append(str(business_object.get("name")))
    target_scope = issue_frame.get("target_scope") or {}
    for key in (
        "target_work",
        "target_system",
        "target_customer_or_industry",
        "geography_or_market",
    ):
        terms.extend(_string_list(target_scope.get(key), max_items=10))
    for item in _jsonish_list(issue_frame.get("numbers")):
        if isinstance(item, dict):
            terms.extend([str(item.get("metric") or ""), str(item.get("value") or "")])
    return _unique_texts(terms, max_items=20)


def _text_overlaps_issue_terms(text: str, issue_terms: list[str]) -> bool:
    value = str(text or "").casefold()
    for term in issue_terms:
        clean_term = _clean_phrase(term)
        if len(clean_term) < 2:
            continue
        if clean_term.casefold() in value:
            return True
        term_tokens = set(re.findall(r"[가-힣A-Za-z0-9&·+_-]{2,}", clean_term))
        text_tokens = set(re.findall(r"[가-힣A-Za-z0-9&·+_-]{2,}", value))
        if term_tokens and len(term_tokens & text_tokens) >= min(2, len(term_tokens)):
            return True
    return False


def _off_topic_application_product_violation(
    text: str,
    *,
    integrated_issue: dict[str, Any],
) -> str:
    if not text or not integrated_issue:
        return ""
    main_context = _main_issue_context_text(integrated_issue)
    off_topic_terms = _non_main_event_product_terms(integrated_issue)
    for term in off_topic_terms:
        if len(term) < 2:
            continue
        if not re.search(re.escape(term), text, flags=re.IGNORECASE):
            continue
        if re.search(re.escape(term), main_context, flags=re.IGNORECASE):
            continue
        return (
            "현재 클러스터의 핵심 사건이 아닌 부가 적용 사례의 제품/서비스명을 "
            "대응방향에 사용했습니다. 메인 사건의 대상 사업·시스템 기준으로 낮춰야 합니다."
        )
    return ""


def _main_issue_context_text(integrated_issue: dict[str, Any]) -> str:
    parts: list[str] = [
        str(integrated_issue.get(key) or "")
        for key in ("headline", "main_event", "main_issue", "one_line_summary")
    ]
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for item in [
            *(intelligence.get("common_facts") or []),
            *(intelligence.get("unique_facts") or []),
        ]:
            if (
                isinstance(item, dict)
                and _fact_has_summary_role(item, "main_event")
                and not _fact_has_summary_role(item, "application_case")
            ):
                parts.append(str(item.get("fact") or ""))
                parts.extend(
                    str(value or "") for value in _jsonish_list(item.get("products_or_services"))
                )
    parts.extend(_string_list(integrated_issue.get("fact_summary"), max_items=5))
    return re.sub(r"\s+", " ", " ".join(parts))


def _non_main_event_product_terms(integrated_issue: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if not isinstance(intelligence, dict):
        return terms
    for item in [
        *(intelligence.get("common_facts") or []),
        *(intelligence.get("unique_facts") or []),
    ]:
        if not isinstance(item, dict):
            continue
        if _fact_has_summary_role(item, "main_event") and not _fact_has_summary_role(
            item,
            "application_case",
        ):
            continue
        for value in _jsonish_list(item.get("products_or_services")):
            term = re.sub(r"\s+", " ", str(value or "").strip(" ."))
            if term:
                terms.append(term)
        terms.extend(_quoted_entity_terms(str(item.get("fact") or "")))
        for evidence in _jsonish_list(item.get("evidence_texts"))[:3]:
            terms.extend(_quoted_entity_terms(str(evidence or "")))
    return list(dict.fromkeys(terms))


def _quoted_entity_terms(text: str) -> list[str]:
    value = str(text or "")
    if not value:
        return []
    terms: list[str] = []
    for match in re.finditer(r"['‘’\"“”]([^'‘’\"“”]{2,50})['‘’\"“”]", value):
        term = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        if term:
            terms.append(term)
    return terms


def _evidence_scoped_business_claim_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any] | None,
    profile_context: dict[str, Any] | None,
) -> str:
    if not text or not label.startswith("skax_implication."):
        return ""
    text_value = str(text or "")
    if "솔루션" in text_value and not _text_overlaps_issue_terms(
        text_value,
        _checkpoint_issue_terms(_issue_frame_for_prompt(integrated_issue or {})),
    ):
        return (
            "SK AX 영향/체크포인트를 근거가 약한 일반 솔루션 표현으로 썼습니다. "
            "현재 사건에서 확인된 대상 사업·시스템·역할·수치·후속 확인 데이터 중심으로 "
            "낮춰야 합니다."
        )
    solution_scope_violation = _solution_term_scope_violation(
        text_value,
        integrated_issue=integrated_issue,
    )
    if solution_scope_violation:
        return solution_scope_violation
    if re.search(
        r"성공\s*사례|성공\s*레퍼런스|구축\s*경험|운영\s*역량",
        text_value,
    ) and not _profile_has_execution_case(
        profile_context,
        integrated_issue=integrated_issue,
    ):
        return (
            "ProfileContext에 실행/구축 사례 근거가 없는데 성공 사례·구축 경험·운영 역량을 "
            "사용했습니다. 현재 사건의 역할·범위·수치와 후속 확인 데이터로 낮춰야 합니다."
        )
    return ""


def _scope_expansion_guard_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    if not text or _is_follow_up_or_watch_field(label):
        return ""
    value = str(text or "")
    event_text = _integrated_grounding_text(integrated_issue)
    context_text = _grounding_text(
        integrated_issue=integrated_issue,
        profile_context=profile_context,
    )
    event_lower = event_text.casefold()
    context_lower = context_text.casefold()
    performance_metric_observation = _is_performance_metric_observation(
        value,
        integrated_issue=integrated_issue,
    )

    if _has_global_scope(value):
        if _scope_effect_claim(value) and not _has_global_scope(event_lower):
            return (
                "글로벌/해외 범위의 강화·확장·영향 표현을 현재 사건 효과처럼 사용했습니다. "
                "원문에 글로벌/해외 근거가 없으면 국가 단위, 국내, 해당 사업 범위로 낮춰야 합니다."
            )
        if not (_has_global_scope(event_lower) or _has_global_scope(context_lower)):
            return (
                "글로벌/해외 시장 범위를 사용했지만 IntegratedIssue 또는 "
                "관련 프로필 근거가 없습니다. "
                "현재 사건의 실제 시장 범위로 낮춰야 합니다."
            )

    if _has_public_private_scope(value) and not _has_public_private_scope_support(context_lower):
        return (
            "공공과 민간 양쪽으로 범위를 넓혔지만 양쪽 고객군 근거가 모두 확인되지 않습니다. "
            "확인된 고객군 또는 사업 범위로 낮춰야 합니다."
        )

    if _has_all_industry_scope(value) and not _has_all_industry_scope(context_lower):
        return (
            "전 산업/산업 전반 범위를 사용했지만 현재 사건 또는 프로필 근거가 부족합니다. "
            "확인된 산업/고객군 범위로 낮춰야 합니다."
        )

    if _has_status_strength_claim(value) and not _has_status_strength_support(event_lower):
        return (
            "입지 강화·레퍼런스 확보·역량 검증처럼 지위 강화 표현을 썼지만 "
            "사업자 선정, 대형 수주, 공식 협약, 레퍼런스 확보 등 직접 근거가 부족합니다. "
            "관찰 신호나 연결 사례 수준으로 낮춰야 합니다."
        )

    if (
        label.startswith(("analysis.", "peer_implication."))
        and (
            _has_status_strength_claim(value)
            or _has_broad_expansion_claim(value)
            or _has_effectiveness_claim(value)
        )
        and _relevant_profile_linkage_level(
            profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
        in {"high", "medium"}
        and not _has_concrete_profile_term(
            value,
            profile_context=profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
    ):
        return (
            "피어 프로필 기반 강한 해석 표현을 사용했지만 문장 안에 현재 사건과 맞는 "
            "구체 프로필 사업영역/역량명이 보이지 않습니다. 피어의 기존 역량과 현재 사건의 "
            "접점을 명시하거나 관찰 신호 수준으로 낮춰야 합니다."
        )

    if label.startswith(("analysis.", "peer_implication.")) and (
        _has_status_strength_claim(value)
        or _has_broad_expansion_claim(value)
        or _has_attention_growth_claim(value)
    ):
        return (
            "피어 시사점이 입지 강화/영역 확장/관심 반영 같은 추상 표현으로 끝났습니다. "
            "현재 사건의 사실, 관련 피어 프로필 역량, 그 둘의 연결 의미를 명시해야 합니다."
        )

    if _has_broad_expansion_claim(value):
        linkage_level = _relevant_profile_linkage_level(
            profile_context,
            integrated_issue=integrated_issue,
            scope="skax" if label.startswith("skax_implication") else "peer",
        )
        if (
            linkage_level not in {"high", "medium"}
            or not _has_expansion_support(event_lower)
            or _high_signal_issue_overlap_count(value, integrated_issue) < 2
        ):
            return (
                "사업영역/서비스 확장 표현을 사용했지만 현재 사건과 관련 프로필의 연결 또는 "
                "범위 확대 근거가 충분하지 않습니다. 연결 사례, 참여 기반, 레퍼런스 가능성처럼 "
                "강도를 낮춰야 합니다."
            )

    if _has_attention_growth_claim(value) and not re.search(r"관심|주목", event_lower):
        return (
            "관심 증가/주목 같은 시장 반응 표현을 원문 근거 없이 사용했습니다. "
            "확인된 사업, 수요 신호, 비교 기준 변화로 낮춰야 합니다."
        )

    if _has_effectiveness_claim(value) and not performance_metric_observation:
        linkage_level = _relevant_profile_linkage_level(
            profile_context,
            integrated_issue=integrated_issue,
            scope="skax" if label.startswith("skax_implication") else "peer",
        )
        if linkage_level not in {"high", "medium"} or not _has_effect_scope(value):
            return (
                "긍정적 영향·경쟁력 강화·운영 효율성 향상 같은 효과성 표현에 "
                "현재 사건, 관련 프로필 역량, 기대효과 범위가 함께 보이지 않습니다. "
                "관찰 신호, 검증 계기, 레퍼런스 가능성 수준으로 낮춰야 합니다."
            )

    return ""


def _is_performance_metric_observation(text: str, *, integrated_issue: dict[str, Any]) -> bool:
    if _issue_frame_kind(integrated_issue) != "performance":
        return False
    value = str(text or "")
    if re.search(r"입지|경쟁력|사업\s*확장|사업\s*영역\s*확장|긍정적\s*영향", value):
        return False
    return bool(
        re.search(
            r"매출|영업이익|순이익|영업이익률|수익성|실적|이익률|비용\s*구조",
            value,
        )
    )


def _has_global_scope(text: str) -> bool:
    return bool(re.search(r"글로벌|해외|국외|수출|global", str(text or ""), flags=re.IGNORECASE))


def _scope_effect_claim(text: str) -> bool:
    return bool(
        re.search(
            r"강화|확장|확대|영향|기회|성장|진출|입지|레퍼런스|사업\s*영역|서비스",
            str(text or ""),
        )
    )


def _has_public_private_scope(text: str) -> bool:
    value = str(text or "")
    if re.search(r"민관", value):
        return True
    return bool(re.search(r"공공", value) and re.search(r"민간", value))


def _has_public_private_scope_support(text: str) -> bool:
    value = str(text or "")
    if re.search(r"민관", value):
        return True
    return bool(
        re.search(r"공공|정부|국가|공공기관", value) and re.search(r"민간|기업|민간\s*참여", value)
    )


def _has_all_industry_scope(text: str) -> bool:
    return bool(re.search(r"전\s*산업|산업\s*전반|모든\s*산업|전방위", str(text or "")))


def _has_status_strength_claim(text: str) -> bool:
    return bool(
        re.search(
            r"입지[가를은\s]*(강화|확고|확대|확장|높)|"
            r"입지[를을\s]*(강화|확대|확장|높)|"
            r"레퍼런스[가를은\s]*(확보|강화)|"
            r"역량[이가을를\s]*(검증|입증)|"
            r"사업자[로서의\s]*(입지|지위)",
            str(text or ""),
        )
    )


def _has_status_strength_support(text: str) -> bool:
    return bool(
        re.search(
            r"최종\s*선정|사업자\s*선정|민간\s*참여자|공식\s*협약|실시협약|"
            r"주주간\s*계약|대형\s*수주|수주|계약\s*체결|레퍼런스|선정|협약|구축사업",
            str(text or ""),
        )
    )


def _has_broad_expansion_claim(text: str) -> bool:
    return bool(
        re.search(
            r"사업\s*(영역|범위)[이가은을를\s]*(확장|확대|넓)|"
            r"영역[이가은을를\s]*(확장|확대)|"
            r"영역.{0,18}(확장|확대)|"
            r"입지[가를은을\s]*(확장|확대|높)|"
            r"서비스[가를은을\s]*(확장|확대)|"
            r"고객군[이가은을를\s]*(확장|확대)|"
            r"부문[이가은을를에\s]*(확장|확대)",
            str(text or ""),
        )
    )


def _has_attention_growth_claim(text: str) -> bool:
    return bool(
        re.search(
            r"관심[이가은을를\s]*(높|증가|확대|반영)|"
            r"관심.{0,16}반영|"
            r"관심.{0,16}(나타|보여|시사)|"
            r"주목[을를이가\s]*(받|높)",
            str(text or ""),
        )
    )


def _high_signal_issue_overlap_count(text: str, integrated_issue: dict[str, Any]) -> int:
    generic_tokens = {
        "사업",
        "사업을",
        "사업의",
        "부문",
        "영역",
        "영역을",
        "공공",
        "민간",
        "기회",
        "시장",
        "서비스",
        "인프라",
        "고객",
        "유사",
    }
    issue_tokens = _issue_relevance_tokens(integrated_issue)
    text_tokens = _content_tokens(str(text or ""))
    overlap = {
        token
        for token in issue_tokens & text_tokens
        if token not in generic_tokens and len(token) >= 2
    }
    return len(overlap)


def _has_concrete_profile_term(
    text: str,
    *,
    profile_context: dict[str, Any],
    integrated_issue: dict[str, Any],
    scope: str,
) -> bool:
    output_tokens = _content_tokens(str(text or ""))
    profile_terms = _concrete_profile_terms(
        profile_context,
        integrated_issue=integrated_issue,
        scope=scope,
    )
    return bool(output_tokens & profile_terms)


def _concrete_profile_terms(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> set[str]:
    generic_terms = {
        "ai",
        "ax",
        "id",
        "name",
        "사업",
        "시장",
        "서비스",
        "인프라",
        "글로벌",
        "공공",
        "민간",
        "기업",
        "중심",
        "확장",
        "강화",
        "전환",
        "역량",
        "peer",
        "skax",
        "profile",
        "snapshot",
        "version",
        "company",
        "company_id",
        "company_name",
        "company_name_ko",
        "peer_id",
        "schema_version",
        "generated_at",
    }
    prompt_profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue)
    if scope == "skax":
        chunks = [_json_dumps(prompt_profile.get("skax_profile") or {})]
    else:
        peer_profiles = prompt_profile.get("peer_profiles") or {}
        chunks = []
        if isinstance(peer_profiles, dict):
            for company_id in _companies_from_integrated_issue(integrated_issue):
                chunks.append(_json_dumps(peer_profiles.get(company_id) or {}))
    terms = set()
    for chunk in chunks:
        terms.update(_content_tokens(chunk))
    company_identity_terms = _company_identity_terms(integrated_issue)
    return {
        term
        for term in terms
        if term not in generic_terms
        and term not in company_identity_terms
        and len(term) >= 3
        and not re.fullmatch(r"\d+", term)
    }


def _company_identity_terms(integrated_issue: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for company_id in _companies_from_integrated_issue(integrated_issue):
        terms.update(_content_tokens(company_id))
        try:
            aliases = expand_peer_aliases(company_id)
        except Exception:
            aliases = []
        for alias in aliases:
            terms.update(_content_tokens(str(alias or "")))
    return terms


def _has_expansion_support(text: str) -> bool:
    return bool(
        re.search(
            r"선정|수주|계약|협약|구축|참여|추진|확대|확장|신규|진출|전환|도입|센터|인프라",
            str(text or ""),
        )
    )


def _has_effectiveness_claim(text: str) -> bool:
    return bool(
        re.search(
            r"긍정적\s*영향|경쟁력[이가을를\s]*(강화|제고|높)|"
            r"운영\s*효율성[이가을를\s]*(향상|개선|높)|"
            r"수익성[이가을를\s]*(개선|향상)|"
            r"매출[이가을를\s]*(성장|확대|증가)|"
            r"성과[가를은\s]*(확대|개선|향상)",
            str(text or ""),
        )
    )


def _has_effect_scope(text: str) -> bool:
    return bool(
        re.search(
            r"현재|이번|선정|수주|계약|협약|구축|인프라|운영|GPU|데이터센터|"
            r"프로필|기존\s*역량|레퍼런스|검증|범위|기준|고객군|대상\s*시스템",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def _solution_term_scope_violation(
    text: str,
    *,
    integrated_issue: dict[str, Any] | None,
) -> str:
    if not integrated_issue or "솔루션" not in str(text or ""):
        return ""
    evidence_text = _integrated_grounding_text(integrated_issue)
    evidence_terms = _evidence_scope_terms(evidence_text)
    if not evidence_terms:
        return ""
    for match in re.finditer(r"([가-힣A-Za-z0-9&+·/_\s-]{2,56})\s*솔루션", str(text or "")):
        phrase = match.group(1)
        phrase_terms = _evidence_scope_terms(phrase)
        unsupported_terms = [
            term
            for term in phrase_terms
            if _is_claim_scope_term(term)
            and not _scope_term_supported(
                term, evidence_terms=evidence_terms, evidence_text=evidence_text
            )
        ]
        if unsupported_terms:
            return (
                "대응방향의 솔루션명이 현재 IntegratedIssue 근거 범위를 벗어났습니다. "
                f"근거 없는 용어: {', '.join(unsupported_terms[:3])}. "
                "현재 사건의 대상 시스템/전환 범위/검증 기준 중심 표현으로 낮춰야 합니다."
            )
    return ""


def _scope_term_supported(term: str, *, evidence_terms: set[str], evidence_text: str) -> bool:
    if term in evidence_terms or term.upper() in evidence_terms:
        return True
    normalized_evidence = str(evidence_text or "").casefold()
    normalized_term = str(term or "").casefold()
    if normalized_term and normalized_term in normalized_evidence:
        return True
    aliases = {
        "금융": ("금융", "금융권", "금융기관"),
        "IT": ("IT", "아이티"),
        "인프라": ("인프라", "시스템"),
        "AI": ("AI", "인공지능", "에이아이"),
    }
    for alias in aliases.get(term.upper(), aliases.get(term, ())):
        if str(alias).casefold() in normalized_evidence:
            return True
    return False


def _evidence_scope_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·/_-]{1,}", str(text or "")):
        cleaned = token.strip(".,;:()[]{}'\"")
        upper = cleaned.upper()
        if upper in {"AI", "IT", "DX", "AX", "UI", "UX", "SI", "MSP", "ERP", "CRM"}:
            terms.add(upper)
            continue
        normalized = _normalize_content_token(cleaned)
        if normalized and not _is_low_signal_content_token(normalized):
            terms.add(normalized.casefold())
    return terms


def _is_claim_scope_term(term: str) -> bool:
    if term.upper() in {"AI", "DX", "MSP", "ERP", "CRM"}:
        return True
    if term in {
        "sk",
        "ax",
        "고객",
        "고객군",
        "유사",
        "유사한",
        "사업",
        "프로젝트",
        "제안",
        "대상",
        "관련",
    }:
        return False
    return bool(
        re.search(
            r"클라우드|인공지능|블록체인|보안|로봇|팩토리|물류|ERP|CRM|MSP|AI|DX",
            term,
            flags=re.IGNORECASE,
        )
    )


def _profile_has_execution_case(
    profile_context: dict[str, Any] | None,
    *,
    integrated_issue: dict[str, Any] | None,
) -> bool:
    if not isinstance(profile_context, dict):
        return False
    profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue or {})
    for item in _iter_dicts(profile):
        for key, value in item.items():
            if str(key).casefold() in {"execution_cases", "execution_case", "case_studies"}:
                if value not in ({}, [], "", None):
                    return True
    profile_text = _json_dumps(profile)
    return bool(re.search(r"성공\s*사례|구축\s*사례|레퍼런스\s*사례", profile_text))


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
        parts.append(
            _json_dumps(_profile_for_prompt(profile_context, integrated_issue=integrated_issue))
        )
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
    out["analysis"] = analysis
    out["implication"] = implication
    return out


def _restore_valid_flags_if_structurally_safe(result: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    if "quality_gate_failed" in _json_dumps(out):
        return out
    analysis = out.get("analysis") or {}
    implication = out.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}

    if (
        analysis.get("analysis_summary")
        and analysis.get("market_signal")
        and _string_list(analysis.get("strategic_meaning"), max_items=3)
    ):
        analysis["is_valid_analysis"] = True

    if (peer.get("peer_meaning") or skax.get("why_important")) and (
        _string_list(skax.get("recommended_actions"), max_items=3)
        or _string_list(skax.get("opportunities"), max_items=3)
        or skax.get("potential_impact")
    ):
        implication["is_valid_implication"] = True

    out["analysis"] = analysis
    implication["peer_implication"] = peer
    implication["skax_implication"] = skax
    out["implication"] = implication
    return out


def _attach_sentence_grounding(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any] | None,
) -> dict[str, Any]:
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    grounding = _build_sentence_grounding(
        out,
        integrated_issue=integrated_issue,
        profile_context=profile_context or {},
    )
    out["sentence_grounding"] = grounding
    return out


def _critical_ungrounded_paths(grounding: dict[str, Any]) -> list[str]:
    paths = []
    for entry in grounding.get("entries") or []:
        if not isinstance(entry, dict) or not entry.get("needs_review"):
            continue
        path = str(entry.get("path") or "")
        if path.startswith(
            (
                "analysis.",
                "peer_implication.",
                "skax_implication.why_important",
                "skax_implication.potential_impact",
            )
        ):
            paths.append(path)
    return list(dict.fromkeys(paths))


def _build_sentence_grounding(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> dict[str, Any]:
    fact_entries = _fact_texts(integrated_issue)
    profile_entries = _profile_grounding_entries(
        profile_context,
        integrated_issue=integrated_issue,
    )
    entries: list[dict[str, Any]] = []
    for path, target_text, scope in _grounding_target_texts(result):
        entries.extend(
            _grounding_entries_for_text(
                path=path,
                text=target_text,
                scope=scope,
                fact_entries=fact_entries,
                profile_entries=profile_entries,
            )
        )
    ungrounded_paths = [
        item["path"]
        for item in entries
        if item.get("needs_review") and item.get("grounding_type") == "ungrounded"
    ]
    return {
        "schema_version": "sentence-grounding-v1",
        "generator": "StrategicInsightAgent",
        "entries": entries,
        "summary": {
            "entry_count": len(entries),
            "fact_grounded_count": sum(1 for item in entries if item.get("used_fact_ids")),
            "profile_grounded_count": sum(1 for item in entries if item.get("used_profile_fields")),
            "ungrounded_paths": ungrounded_paths,
        },
    }


def _grounding_target_texts(result: dict[str, Any]) -> list[tuple[str, str, str]]:
    analysis = result.get("analysis") or {}
    implication = result.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    targets: list[tuple[str, str, str]] = [
        ("analysis.analysis_summary", str(analysis.get("analysis_summary") or ""), "peer"),
        ("analysis.market_signal", str(analysis.get("market_signal") or ""), "peer"),
        ("analysis.impact_reason", str(analysis.get("impact_reason") or ""), "peer"),
        ("analysis.reason", str(analysis.get("reason") or ""), "peer"),
        ("peer_implication.peer_meaning", str(peer.get("peer_meaning") or ""), "peer"),
        (
            "peer_implication.capability_change",
            str(peer.get("capability_change") or ""),
            "peer",
        ),
        (
            "skax_implication.why_important",
            str(skax.get("why_important") or ""),
            "skax",
        ),
        (
            "skax_implication.potential_impact",
            str(skax.get("potential_impact") or ""),
            "skax",
        ),
    ]
    for index, item in enumerate(_string_list(analysis.get("strategic_meaning"), max_items=3)):
        targets.append((f"analysis.strategic_meaning[{index}]", item, "peer"))
    for field in ("opportunities", "threats", "recommended_actions"):
        for index, item in enumerate(_string_list(skax.get(field), max_items=3)):
            targets.append((f"skax_implication.{field}[{index}]", item, "skax"))
    return [(path, text.strip(), scope) for path, text, scope in targets if text.strip()]


def _grounding_entries_for_text(
    *,
    path: str,
    text: str,
    scope: str,
    fact_entries: list[tuple[str, str]],
    profile_entries: list[dict[str, str]],
) -> list[dict[str, Any]]:
    sentences = _split_sentences(text) or [text]
    output: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        sentence_text = sentence.strip()
        if not sentence_text:
            continue
        used_fact_ids = _matching_fact_ids(sentence_text, fact_entries)
        used_profile_fields = _matching_profile_fields(
            sentence_text,
            profile_entries=profile_entries,
            scope=scope,
        )
        grounding_type = _grounding_type(used_fact_ids, used_profile_fields)
        output.append(
            {
                "path": f"{path}.sentence[{index}]" if len(sentences) > 1 else path,
                "text": sentence_text,
                "used_fact_ids": used_fact_ids,
                "used_profile_fields": used_profile_fields,
                "grounding_type": grounding_type,
                "needs_review": grounding_type == "ungrounded",
            }
        )
    return output


def _matching_fact_ids(text: str, fact_entries: list[tuple[str, str]]) -> list[str]:
    tokens = _content_tokens(text)
    matched: list[str] = []
    for fact_id, fact_text in fact_entries:
        if _fact_is_referenced(fact_text, text, tokens):
            matched.append(fact_id)
        if len(matched) >= 5:
            break
    return matched


def _matching_profile_fields(
    text: str,
    *,
    profile_entries: list[dict[str, str]],
    scope: str,
) -> list[str]:
    tokens = _content_tokens(text)
    matched: list[str] = []
    for entry in profile_entries:
        entry_scope = entry.get("scope") or ""
        if scope == "skax" and entry_scope != "skax":
            continue
        if scope == "peer" and entry_scope == "skax":
            continue
        if not _profile_entry_is_referenced(entry.get("text", ""), text, tokens):
            continue
        path = entry.get("path") or ""
        if path and path not in matched:
            matched.append(path)
        if len(matched) >= 5:
            break
    return matched


def _grounding_type(fact_ids: list[str], profile_fields: list[str]) -> str:
    if fact_ids and profile_fields:
        return "fact+profile"
    if fact_ids:
        return "fact"
    if profile_fields:
        return "profile"
    return "ungrounded"


def _profile_grounding_entries(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> list[dict[str, str]]:
    prompt_profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue)
    entries: list[dict[str, str]] = []
    skax = prompt_profile.get("skax_profile") or {}
    entries.extend(_flatten_profile_grounding_entries(skax, path="skax_profile", scope="skax"))
    peers = prompt_profile.get("peer_profiles") or {}
    if isinstance(peers, dict):
        for peer_id, payload in peers.items():
            entries.extend(
                _flatten_profile_grounding_entries(
                    payload,
                    path=f"peer_profiles.{peer_id}",
                    scope="peer",
                )
            )
    return entries


def _flatten_profile_grounding_entries(
    value: Any,
    *,
    path: str,
    scope: str,
) -> list[dict[str, str]]:
    if value in ({}, [], "", None):
        return []
    if isinstance(value, dict):
        entries: list[dict[str, str]] = []
        combined = _profile_entry_text(value)
        if combined:
            entries.append({"path": path, "scope": scope, "text": combined})
        for key, child in value.items():
            if key in {"company_id", "peer_id", "company_name", "company_name_ko"}:
                continue
            entries.extend(
                _flatten_profile_grounding_entries(child, path=f"{path}.{key}", scope=scope)
            )
        return entries
    if isinstance(value, list):
        entries = []
        for index, child in enumerate(value[:8]):
            entries.extend(
                _flatten_profile_grounding_entries(child, path=f"{path}[{index}]", scope=scope)
            )
        return entries
    text = str(value or "").strip()
    return [{"path": path, "scope": scope, "text": text}] if text else []


def _profile_entry_text(value: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "name",
        "business_area",
        "summary",
        "recent_direction",
        "core_capabilities",
        "capabilities",
        "change_type",
        "overall_change",
    ):
        if key not in value:
            continue
        raw = value.get(key)
        if isinstance(raw, list):
            parts.extend(str(item or "") for item in raw)
        elif isinstance(raw, dict):
            parts.append(_json_dumps(raw))
        else:
            parts.append(str(raw or ""))
    return " ".join(part.strip() for part in parts if part and part.strip())


def _profile_entry_is_referenced(
    entry_text: str,
    output_text: str,
    output_tokens: set[str],
) -> bool:
    tokens = _content_tokens(entry_text)
    if len(tokens & output_tokens) >= 2:
        return True
    for token in tokens:
        if len(token) >= 4 and token in output_text:
            return True
    return False


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
    profile_context: dict[str, Any] | None = None,
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
    frame_kind = _issue_frame_kind(integrated_issue)

    if (
        profile_context
        and _profile_context_is_sparse(profile_context)
        and frame_kind
        in {
            "performance",
            "launch_or_service",
            "selection_or_build",
        }
    ):
        # Sparse profile should limit profile-based claims, not erase a grounded LLM analysis.
        # Fill only truly missing fields so this guard remains mechanical and non-generative.
        for key in ("analysis_summary", "market_signal", "impact_reason", "reason"):
            if not str(analysis.get(key) or "").strip():
                analysis[key] = _event_based_analysis_field(
                    key,
                    integrated_issue=integrated_issue,
                )
        if not _string_list(analysis.get("strategic_meaning"), max_items=3):
            analysis["strategic_meaning"] = _event_based_strategic_meaning_candidates(
                integrated_issue
            )[:3]
        if not str(peer.get("peer_meaning") or "").strip():
            peer["peer_meaning"] = _event_based_peer_meaning(
                integrated_issue=integrated_issue,
                peer=peer,
            )
        if not str(peer.get("capability_change") or "").strip():
            peer["capability_change"] = _event_based_capability_change(integrated_issue)
        if not str(skax.get("why_important") or "").strip():
            skax["why_important"] = _skax_checkpoint_field(
                "why_important",
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
        if not str(skax.get("potential_impact") or "").strip():
            skax["potential_impact"] = _skax_checkpoint_field(
                "potential_impact",
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
        if not _string_list(skax.get("recommended_actions"), max_items=3):
            skax["recommended_actions"] = _event_based_recommended_actions(
                integrated_issue,
                profile_context=profile_context,
            )

    if _main_company_is_customer_or_buyer(integrated_issue):
        for key in ("analysis_summary", "market_signal", "impact_reason", "reason"):
            if analysis.get(key):
                analysis[key] = _repair_customer_role_overstatement(str(analysis[key]))
                if _counterparty_guard_violation(
                    analysis[key],
                    label=f"analysis.{key}",
                    integrated_issue=integrated_issue,
                ) or _hard_quality_violation_for_text(
                    analysis[key],
                    label=f"analysis.{key}",
                    integrated_issue=integrated_issue,
                ):
                    analysis[key] = _event_based_analysis_field(
                        key,
                        integrated_issue=integrated_issue,
                    )
        analysis["strategic_meaning"] = [
            _repair_customer_role_overstatement(item)
            for item in _string_list(analysis.get("strategic_meaning"), max_items=3)
        ]
        analysis["strategic_meaning"] = _safe_event_based_strategic_meanings(
            analysis.get("strategic_meaning"),
            integrated_issue=integrated_issue,
        )
        for key in ("peer_meaning", "capability_change"):
            if peer.get(key):
                peer[key] = _repair_customer_role_overstatement(str(peer[key]))
                if _hard_quality_violation_for_text(
                    peer[key],
                    label=f"peer_implication.{key}",
                    integrated_issue=integrated_issue,
                ):
                    if key == "peer_meaning":
                        peer[key] = _event_based_peer_meaning(
                            integrated_issue=integrated_issue,
                            peer=peer,
                        )
                    else:
                        peer[key] = _event_based_capability_change(integrated_issue)
        for key in ("why_important", "potential_impact"):
            if skax.get(key):
                skax[key] = _repair_customer_role_overstatement(str(skax[key]))
                if _hard_quality_violation_for_text(
                    skax[key],
                    label=f"skax_implication.{key}",
                    integrated_issue=integrated_issue,
                ) or _evidence_scoped_business_claim_violation(
                    skax[key],
                    label=f"skax_implication.{key}",
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                ):
                    skax[key] = ""
        if profile_context and not _has_relevant_peer_profile_context(
            profile_context,
            integrated_issue=integrated_issue,
        ):
            peer["peer_meaning"] = _event_based_peer_meaning(
                integrated_issue=integrated_issue,
                peer=peer,
            )
            peer["capability_change"] = _event_based_capability_change(integrated_issue)

    for key in ("analysis_summary", "market_signal", "impact_reason", "reason"):
        label = f"analysis.{key}"
        if analysis.get(key) and (
            _hard_quality_violation_for_text(
                str(analysis[key]),
                label=label,
                integrated_issue=integrated_issue,
            )
            or _scope_expansion_guard_violation(
                str(analysis[key]),
                label=label,
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
            or _issue_frame_mismatch_violation(
                str(analysis[key]),
                label=label,
                integrated_issue=integrated_issue,
            )
            or _generic_insight_quality_violation(
                str(analysis[key]),
                label=label,
                integrated_issue=integrated_issue,
            )
            or _reasoning_flow_quality_violation(
                str(analysis[key]),
                label=label,
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
        ):
            analysis[key] = _profile_linked_analysis_field(
                key,
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )

    strategic_items = _string_list(analysis.get("strategic_meaning"), max_items=3)
    has_hard_or_scope_strategic_violation = any(
        _hard_quality_violation_for_text(
            item,
            label=f"analysis.strategic_meaning[{index}]",
            integrated_issue=integrated_issue,
        )
        or _scope_expansion_guard_violation(
            item,
            label=f"analysis.strategic_meaning[{index}]",
            integrated_issue=integrated_issue,
            profile_context=profile_context or {},
        )
        or _issue_frame_mismatch_violation(
            item,
            label=f"analysis.strategic_meaning[{index}]",
            integrated_issue=integrated_issue,
        )
        or _generic_insight_quality_violation(
            item,
            label=f"analysis.strategic_meaning[{index}]",
            integrated_issue=integrated_issue,
        )
        or _reasoning_flow_quality_violation(
            item,
            label=f"analysis.strategic_meaning[{index}]",
            integrated_issue=integrated_issue,
            profile_context=profile_context or {},
        )
        for index, item in enumerate(strategic_items, start=1)
    )
    if has_hard_or_scope_strategic_violation:
        analysis["strategic_meaning"] = _profile_linked_strategic_meanings(
            integrated_issue=integrated_issue,
            profile_context=profile_context or {},
        )

    for key in ("peer_meaning", "capability_change"):
        label = f"peer_implication.{key}"
        if peer.get(key) and (
            _hard_quality_violation_for_text(
                str(peer[key]),
                label=label,
                integrated_issue=integrated_issue,
            )
            or _scope_expansion_guard_violation(
                str(peer[key]),
                label=label,
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
            or _issue_frame_mismatch_violation(
                str(peer[key]),
                label=label,
                integrated_issue=integrated_issue,
            )
            or _generic_insight_quality_violation(
                str(peer[key]),
                label=label,
                integrated_issue=integrated_issue,
            )
            or _reasoning_flow_quality_violation(
                str(peer[key]),
                label=label,
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
        ):
            peer[key] = (
                _profile_linked_peer_meaning(
                    integrated_issue=integrated_issue,
                    profile_context=profile_context or {},
                    peer=peer,
                )
                if key == "peer_meaning"
                else _profile_linked_capability_change(
                    integrated_issue=integrated_issue,
                    profile_context=profile_context or {},
                )
            )

    for key in ("why_important", "potential_impact"):
        if skax.get(key) and (
            _hard_quality_violation_for_text(
                str(skax[key]),
                label=f"skax_implication.{key}",
                integrated_issue=integrated_issue,
            )
            or _unsupported_claim_pattern_violation(
                str(skax[key]),
                integrated_issue=integrated_issue,
            )
            or _evidence_scoped_business_claim_violation(
                str(skax[key]),
                label=f"skax_implication.{key}",
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            or _scope_expansion_guard_violation(
                str(skax[key]),
                label=f"skax_implication.{key}",
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
            or _skax_external_execution_copy_violation(str(skax[key]))
            or _issue_frame_mismatch_violation(
                str(skax[key]),
                label=f"skax_implication.{key}",
                integrated_issue=integrated_issue,
            )
            or _generic_insight_quality_violation(
                str(skax[key]),
                label=f"skax_implication.{key}",
                integrated_issue=integrated_issue,
            )
            or _reasoning_flow_quality_violation(
                str(skax[key]),
                label=f"skax_implication.{key}",
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
        ):
            skax[key] = _skax_checkpoint_field(
                key,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )

    _repair_result_numeric_grounding(
        analysis=analysis,
        implication=implication,
        integrated_issue=integrated_issue,
    )
    for field in ("opportunities", "threats"):
        skax[field] = [
            item
            for index, item in enumerate(_string_list(skax.get(field), max_items=3), start=1)
            if not _hard_quality_violation_for_text(
                item,
                label=f"skax_implication.{field}[{index}]",
                integrated_issue=integrated_issue,
            )
            and not _evidence_scoped_business_claim_violation(
                item,
                label=f"skax_implication.{field}[{index}]",
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            and not _scope_expansion_guard_violation(
                item,
                label=f"skax_implication.{field}[{index}]",
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
            and not _unsupported_claim_pattern_violation(
                item,
                integrated_issue=integrated_issue,
            )
            and not _generic_insight_quality_violation(
                item,
                label=f"skax_implication.{field}[{index}]",
                integrated_issue=integrated_issue,
            )
        ]
    skax["recommended_actions"] = [
        action
        for index, action in enumerate(
            _string_list(skax.get("recommended_actions"), max_items=3), start=1
        )
        if not _recommended_action_quality_violation(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        and not _hard_quality_violation_for_text(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
        )
        and not _counterparty_role_action_violation(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
        )
        and not _issue_frame_mismatch_violation(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
        )
        and not _generic_insight_quality_violation(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
        )
        and not _reasoning_flow_quality_violation(
            action,
            label=f"skax_implication.recommended_actions[{index}]",
            integrated_issue=integrated_issue,
            profile_context=profile_context or {},
        )
    ]
    if not _string_list(skax.get("recommended_actions"), max_items=3):
        skax["recommended_actions"] = _event_based_recommended_actions(
            integrated_issue,
            profile_context=profile_context,
        )

    out["analysis"] = analysis
    implication["peer_implication"] = peer
    implication["skax_implication"] = skax
    out["implication"] = implication
    if not _has_required_output_structure(out):
        return _mark_quality_gate_failed(
            out,
            [
                "최소 품질 보정 후 필수 analysis/implication 구조가 남지 않았습니다. "
                "근거 없는 시사점이나 대응방향을 새로 만들지 않고 human_review로 넘깁니다."
            ],
        )
    return out


def _skax_external_execution_copy_violation(text: str) -> bool:
    return bool(
        re.search(
            r"제안서|PoC|레퍼런스|운영\s*계획|제안\s*방식|제안\s*기준|영업\s*대응",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def _skax_checkpoint_field(
    key: str,
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any] | None = None,
) -> str:
    issue_frame = _issue_frame_for_prompt(integrated_issue)
    frame_kind = str(issue_frame.get("frame_kind") or "")
    if frame_kind == "performance":
        if key == "why_important":
            return (
                "피어사의 실적 개선 신호는 SK AX가 유사 피어 동향을 볼 때 외형 성장과 "
                "수익성 개선을 분리해 비교해야 하는 근거입니다."
            )
        return (
            "후속 판단은 사업부별 기여도, 수익성 지속성, 비용 구조 변화가 확인될 때 "
            "가능합니다. SK AX는 특정 사업영역 성과로 단정하지 말고 전사 재무 지표와 "
            "사업부 기여 근거를 나눠 봐야 합니다."
        )
    business_object = issue_frame.get("business_object") or {}
    subject = str(business_object.get("name") or "").strip() or (
        _issue_subject_phrase(integrated_issue) or "현재 동향"
    )
    axes = ", ".join(_decision_axes_from_issue_frame(issue_frame)[:3])
    followups = ", ".join(_checkpoint_followups_from_issue_frame(issue_frame)[:3])
    skax_phrase = _profile_area_phrase(
        profile_context or {},
        integrated_issue=integrated_issue,
        scope="skax",
    )
    skax_context = (
        f"SK AX의 관련 사업/역량 중 {skax_phrase}"
        if skax_phrase
        else "현재 확인 가능한 SK AX 관련 사업/역량"
    )
    if key == "why_important":
        return (
            f"{subject}는 {skax_context}가 어느 지점까지 대응 가능한지 점검하게 하는 "
            f"신호입니다. 기사에서 확인된 비교 축은 {axes}이며, 이 축이 있어야 "
            "단순 관련성보다 실제 내부 대응 범위를 판단할 수 있습니다."
        )
    return (
        f"후속 판단은 {followups}가 확인될 때 가능합니다. SK AX는 {skax_context}를 "
        f"기준으로 {axes} 중 직접 책임질 수 있는 부분과 외부 보완이 필요한 부분을 "
        "나눠 봐야 하며, 그래야 후속 사업에서 운영 책임과 리스크를 현실적으로 판단할 수 있습니다."
    )


def _has_required_output_structure(result: dict[str, Any]) -> bool:
    analysis = result.get("analysis") or {}
    implication = result.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    skax = implication.get("skax_implication") or {}
    has_analysis = bool(
        analysis.get("analysis_summary")
        and analysis.get("market_signal")
        and _string_list(analysis.get("strategic_meaning"), max_items=3)
    )
    has_implication = bool(
        (peer.get("peer_meaning") or peer.get("capability_change") or skax.get("why_important"))
        and (
            skax.get("potential_impact")
            or _string_list(skax.get("recommended_actions"), max_items=3)
            or _string_list(skax.get("opportunities"), max_items=3)
        )
    )
    return has_analysis and has_implication


def _ensure_safe_recommended_actions(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    implication = out.get("implication") or {}
    skax = implication.get("skax_implication") or {}
    safe_actions: list[str] = []
    for index, action in enumerate(
        _string_list(skax.get("recommended_actions"), max_items=3),
        start=1,
    ):
        label = f"skax_implication.recommended_actions[{index}]"
        if (
            _recommended_action_quality_violation(
                action,
                label=label,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
            or _hard_quality_violation_for_text(
                action, label=label, integrated_issue=integrated_issue
            )
            or _counterparty_role_action_violation(
                action,
                label=label,
                integrated_issue=integrated_issue,
            )
            or _generic_insight_quality_violation(
                action,
                label=label,
                integrated_issue=integrated_issue,
            )
        ):
            continue
        safe_actions.append(action)

    if not safe_actions:
        safe_actions = _event_based_recommended_actions(
            integrated_issue,
            profile_context=profile_context,
        )
    skax["recommended_actions"] = safe_actions[:3]
    implication["skax_implication"] = skax
    out["implication"] = implication
    return out


def _event_based_recommended_actions(
    integrated_issue: dict[str, Any],
    *,
    profile_context: dict[str, Any] | None = None,
) -> list[str]:
    issue_frame = _issue_frame_for_prompt(integrated_issue)
    frame_kind = str(issue_frame.get("frame_kind") or "")
    subject = _issue_display_subject(issue_frame, integrated_issue=integrated_issue)
    main_fact = str(issue_frame.get("main_fact") or "").strip()
    axes = _decision_axes_from_issue_frame(issue_frame)
    followups = _checkpoint_followups_from_issue_frame(issue_frame)
    issue_focus = _issue_focus_for_action(issue_frame)
    axis_text = ", ".join(axes[:3])
    followup_text = ", ".join(followups[:4])
    fact_sentence = (
        f"{main_fact.rstrip('.。')}." if main_fact else "현재 사건의 핵심 근거가 확인됐습니다."
    )
    skax_phrase = _profile_area_phrase(
        profile_context or {},
        integrated_issue=integrated_issue,
        scope="skax",
    )
    skax_context = (
        f"SK AX의 {skax_phrase}" if skax_phrase else "현재 확인 가능한 SK AX 관련 사업/역량"
    )
    skax_context_object = f"{skax_context}을 기준으로"
    decision_reason = _decision_reason_from_issue_frame(issue_frame)

    if frame_kind == "performance":
        actions = [
            (
                f"SK AX는 {subject}와 유사한 실적 신호를 볼 때 {skax_context_object} 바로 "
                f"연결해 단정하지 말고, 기사에서 확인된 {axis_text}를 분리해 봐야 합니다. "
                "이렇게 해야 외형 성장, 수익성 변화, 특정 사업 기여도를 혼동하지 않고 "
                "내부 사업 우선순위를 판단할 수 있습니다."
            ),
            (
                f"후속으로는 {followup_text}를 확인해야 합니다. 이 정보가 있어야 "
                "SK AX가 유사 실적 동향을 특정 역량 성과로 과대해석하지 않고, "
                "어떤 사업 지표를 더 봐야 하는지 조정할 수 있습니다."
            ),
        ]
        return actions

    if frame_kind == "launch_or_service":
        return [
            (
                f"SK AX는 {subject}와 유사한 흐름을 볼 때 {skax_context_object} "
                f"{issue_focus}에 실제로 어떻게 닿는지 먼저 봐야 합니다. {fact_sentence} "
                f"따라서 단순 기능 공개 여부보다 {axis_text}를 확인해야 하며, "
                f"그래야 {decision_reason}를 내부적으로 판단할 수 있습니다."
            ),
            (
                f"후속으로는 {followup_text}를 확인해야 합니다. 이 정보가 확인되어야 "
                "SK AX가 같은 유형의 서비스·업무 적용 흐름에서 직접 대응할 영역과 "
                "추가 검증이 필요한 영역을 나눠 조정할 수 있습니다."
            ),
        ]

    if frame_kind == "selection_or_build":
        return [
            (
                f"SK AX는 {subject}와 유사한 흐름을 볼 때 {skax_context_object} "
                f"{issue_focus} 중 어디까지 직접 감당할 수 있는지 확인해야 합니다. "
                f"{fact_sentence} 이 근거 때문에 {axis_text}를 분리해 봐야 하며, "
                f"그래야 {decision_reason}를 현실적으로 판단할 수 있습니다."
            ),
            (
                f"후속으로는 {followup_text}를 확인해야 합니다. 이 정보가 공개되어야 "
                "SK AX가 유사 대형 과제에서 직접 맡을 범위, 파트너 보완이 필요한 범위, "
                "내부 리스크 기준을 후속 상황에 맞게 조정할 수 있습니다."
            ),
        ]

    actions = [
        (
            f"SK AX는 {subject}와 유사한 흐름을 볼 때 {skax_context_object} "
            f"{issue_focus}와 어디에서 겹치는지 먼저 확인해야 합니다. {fact_sentence} "
            f"이 근거 때문에 {axis_text}를 나눠 봐야 하며, 그래야 {decision_reason}를 "
            "내부적으로 판단할 수 있습니다."
        ),
        (
            f"후속으로는 {followup_text}를 확인해야 합니다. 이 정보가 확인되어야 "
            "SK AX가 직접 점검할 영역, 외부 확인이 필요한 영역, 리스크 판단 기준을 "
            "후속 상황에 맞게 조정할 수 있습니다."
        ),
    ]
    return actions[:3]


def _issue_focus_for_action(issue_frame: dict[str, Any]) -> str:
    target_scope = issue_frame.get("target_scope") or {}
    business_object = issue_frame.get("business_object") or {}
    terms: list[str] = []
    for key in ("target_system", "target_work", "target_customer_or_industry"):
        terms.extend(_string_list(target_scope.get(key), max_items=4))
    terms.extend(_string_list(business_object.get("raw_terms"), max_items=4))
    if business_object.get("name"):
        terms.append(str(business_object.get("name")))
    cleaned = _unique_texts(
        [_clean_issue_axis_phrase(term) for term in terms if _clean_issue_axis_phrase(term)],
        max_items=3,
    )
    return ", ".join(cleaned) if cleaned else "현재 사건의 대상 업무·시스템·서비스"


def _decision_axes_from_issue_frame(issue_frame: dict[str, Any]) -> list[str]:
    axes: list[str] = []
    target_scope = issue_frame.get("target_scope") or {}
    business_object = issue_frame.get("business_object") or {}
    action = issue_frame.get("action_or_event") or {}
    numbers = issue_frame.get("numbers") or []

    target_terms: list[str] = []
    for key in ("target_system", "target_work", "target_customer_or_industry"):
        target_terms.extend(_string_list(target_scope.get(key), max_items=4))
    target_terms = _unique_texts(
        [_clean_issue_axis_phrase(term) for term in target_terms if _clean_issue_axis_phrase(term)],
        max_items=4,
    )
    if target_terms:
        axes.append("대상 범위: " + ", ".join(_unique_texts(target_terms, max_items=3)))

    raw_terms = _unique_texts(
        [
            _clean_issue_axis_phrase(term)
            for term in _string_list(business_object.get("raw_terms"), max_items=5)
            if _clean_issue_axis_phrase(term)
        ],
        max_items=5,
    )
    if raw_terms:
        axes.append("사업·서비스 조건: " + ", ".join(_unique_texts(raw_terms, max_items=2)))

    number_terms = []
    for item in numbers:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "").strip()
        value = str(item.get("value") or "").strip()
        unit = str(item.get("unit") or "").strip()
        if not metric or not value:
            continue
        number_terms.append(" ".join(part for part in (metric, f"{value}{unit}") if part))
    if number_terms:
        axes.append("공개 수치: " + ", ".join(_unique_texts(number_terms, max_items=2)))

    action_terms = [
        _clean_issue_axis_phrase(term)
        for term in _string_list(action.get("activity_types"), max_items=3)
    ]
    status = _clean_issue_axis_phrase(action.get("status"))
    relation_terms = [*action_terms, status if status and status != "unclear" else ""]
    relation_terms = _unique_texts(relation_terms, max_items=3)
    if relation_terms:
        axes.append("진행 단계·관계: " + ", ".join(relation_terms))

    if not axes:
        axes.extend(_checkpoint_axes_from_issue_frame(issue_frame)[:3])
    return _unique_texts([axis for axis in axes if axis], max_items=4) or [
        "현재 사건에서 확인된 대상 범위",
        "참여 주체의 역할 범위",
    ]


def _decision_reason_from_issue_frame(issue_frame: dict[str, Any]) -> str:
    frame_kind = str(issue_frame.get("frame_kind") or "")
    action = issue_frame.get("action_or_event") or {}
    status = str(action.get("status") or "").strip()
    if frame_kind == "performance":
        return "외형 성장과 실제 수익성 변화가 같은 방향인지"
    if frame_kind == "launch_or_service":
        return "기능 공개가 실제 업무 적용과 성과 변화로 이어지는지"
    if frame_kind == "selection_or_build":
        return "참여 사실과 실제 구축·운영 책임 범위를 구분할 수 있는지"
    if frame_kind == "contract_or_transition":
        return "계약 사실과 실제 전환 책임·운영 안정화 범위를 구분할 수 있는지"
    if frame_kind == "partnership":
        return "협력 발표와 실제 역할 분담·후속 계약 가능성을 구분할 수 있는지"
    if status == "planned_or_under_review":
        return "검토·계획 단계의 신호를 확정 성과로 과대해석하지 않을 수 있는지"
    return "관련성 있는 동향과 실제 대응 책임을 구분할 수 있는지"


def _issue_display_subject(
    issue_frame: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> str:
    business_object = issue_frame.get("business_object") or {}
    candidates: list[str] = []
    candidates.extend(_string_list(business_object.get("raw_terms"), max_items=10))
    candidates.append(str(business_object.get("name") or ""))
    candidates.append(_issue_subject_phrase(integrated_issue))
    scored: list[tuple[int, str]] = []
    for candidate in candidates:
        text = _clean_issue_axis_phrase(candidate)
        if not text:
            continue
        if len(text) > 45:
            continue
        score = len(text)
        if re.search(r"협약|계약|선정|구축|전환|플랫폼|서비스|센터|인프라|자동화", text):
            score += 20
        if re.search(r"위한\s*업무협약|기념촬영|사진|전무|부사장", text):
            score -= 40
        scored.append((score, text))
    if scored:
        return max(scored, key=lambda item: item[0])[1]
    return "현재 동향"


def _clean_issue_axis_phrase(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .。"))
    if not text:
        return ""
    if text.casefold() in {
        "general_update",
        "reported_fact",
        "core_fact",
        "unique_fact",
        "common_fact",
        "partnership",
        "collaboration",
        "contract",
        "selection",
        "launch",
        "performance",
        "investment",
        "mou",
        "unclear",
    }:
        return ""
    text = re.sub(r"^(이번|해당)\s+", "", text)
    text = re.sub(r"^위한\s+", "", text)
    if len(text) < 2:
        return ""
    if text in {"국가", "민간", "사업", "프로젝트", "분야"}:
        return ""
    if re.fullmatch(r"\d+", text):
        return ""
    if re.search(r"기념촬영|임직원|전무|부사장|사진|이미지", text):
        return ""
    return text


def _checkpoint_axes_from_issue_frame(issue_frame: dict[str, Any]) -> list[str]:
    axes: list[str] = []
    business_object = issue_frame.get("business_object") or {}
    target_scope = issue_frame.get("target_scope") or {}
    for value in _string_list(business_object.get("raw_terms"), max_items=8):
        if cleaned := _clean_issue_axis_phrase(value):
            axes.append(cleaned)
    if business_object.get("name"):
        axes.append(_clean_issue_axis_phrase(business_object.get("name")))
    if target_scope.get("target_work"):
        axes.extend(
            _clean_issue_axis_phrase(item)
            for item in _string_list(target_scope.get("target_work"), max_items=8)
        )
    if target_scope.get("target_system"):
        axes.extend(
            _clean_issue_axis_phrase(item)
            for item in _string_list(target_scope.get("target_system"), max_items=8)
        )
    if target_scope.get("target_customer_or_industry"):
        axes.extend(
            _clean_issue_axis_phrase(item)
            for item in _string_list(target_scope.get("target_customer_or_industry"), max_items=8)
        )
    if issue_frame.get("numbers"):
        number_values = [
            str(item.get("value") or "").strip()
            for item in issue_frame.get("numbers") or []
            if isinstance(item, dict) and str(item.get("value") or "").strip()
        ]
        number_values = [
            value for value in _unique_texts(number_values, max_items=4) if not value == "1"
        ]
        if any(_numeric_value_is_large(value) for value in number_values):
            number_values = [
                value
                for value in number_values
                if _numeric_value_is_large(value) or not _numeric_value_is_small_date_like(value)
            ]
        if number_values:
            axes.append("공개 수치 " + "/".join(number_values))
    if target_scope.get("geography_or_market"):
        axes.extend(
            _clean_issue_axis_phrase(item)
            for item in _string_list(target_scope.get("geography_or_market"), max_items=8)
        )
    return _unique_texts([axis for axis in axes if axis], max_items=5) or [
        "현재 사건의 대상 범위",
        "참여 주체별 역할",
        "후속 확인 데이터",
    ]


def _checkpoint_followups_from_issue_frame(issue_frame: dict[str, Any]) -> list[str]:
    followups: list[str] = []
    action = issue_frame.get("action_or_event") or {}
    target_scope = issue_frame.get("target_scope") or {}
    business_object = issue_frame.get("business_object") or {}
    subject = _clean_issue_axis_phrase(business_object.get("name")) or "현재 사건"
    if action.get("status") and action.get("status") != "unclear":
        followups.append(f"{subject}의 확정 이후 실행 상태")
    if target_scope.get("target_work"):
        followups.extend(
            f"{_clean_issue_axis_phrase(item)}의 후속 범위"
            for item in _string_list(target_scope.get("target_work"), max_items=8)
            if _clean_issue_axis_phrase(item)
        )
    if target_scope.get("target_system"):
        followups.extend(
            f"{_clean_issue_axis_phrase(item)}의 후속 변화"
            for item in _string_list(target_scope.get("target_system"), max_items=8)
            if _clean_issue_axis_phrase(item)
        )
    if issue_frame.get("numbers"):
        followups.append("공개 수치의 후속 변동")
    entities = issue_frame.get("entities") or []
    if len(entities) >= 2:
        followups.append("참여 주체별 역할 범위")
    followups.append("추가 공시·협약·운영 책임 공개 여부")
    return _unique_texts(followups, max_items=5)


def _numeric_value_is_large(value: str) -> bool:
    number = _numeric_value_float(value)
    return number is not None and number > 31


def _numeric_value_is_small_date_like(value: str) -> bool:
    number = _numeric_value_float(value)
    return number is not None and 1 <= number <= 31


def _numeric_value_float(value: str) -> float | None:
    text = re.sub(r"[^0-9.]", "", str(value or ""))
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _has_relevant_peer_profile_context(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> bool:
    prompt_profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue)
    peer_profiles = prompt_profile.get("peer_profiles") or {}
    if not isinstance(peer_profiles, dict):
        return False
    for company_id in _companies_from_integrated_issue(integrated_issue):
        peer = peer_profiles.get(company_id) or {}
        if not isinstance(peer, dict):
            continue
        if any(
            peer.get(key)
            for key in (
                "business_areas",
                "core_capabilities",
                "recent_changes",
                "capability_evolution",
            )
        ):
            return True
    return False


def _peer_profile_linkage(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    company_id: str,
) -> dict[str, Any]:
    prompt_profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue)
    peer_profiles = prompt_profile.get("peer_profiles") or {}
    profile = peer_profiles.get(company_id) if isinstance(peer_profiles, dict) else {}
    if not isinstance(profile, dict) or not profile:
        return {
            "company": company_id,
            "matched_profile_terms": [],
            "linkage_level": "none",
            "reason": "현재 이슈와 비교할 피어 프로필 본문이 없습니다.",
        }
    issue_tokens = _issue_relevance_tokens(integrated_issue)
    profile_tokens = _content_tokens(_json_dumps(profile))
    matched_terms = sorted(issue_tokens & profile_tokens)
    level = _linkage_level_from_match_count(len(matched_terms))
    reason = (
        "현재 사건의 핵심 토큰이 피어 프로필의 사업영역/역량과 연결됩니다."
        if level in {"high", "medium"}
        else "현재 사건과 피어 프로필의 직접 접점이 약하므로 사건 기반 해석을 우선해야 합니다."
    )
    return {
        "company": company_id,
        "matched_profile_terms": matched_terms[:12],
        "linkage_level": level,
        "reason": reason,
    }


def _relevant_profile_linkage_level(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> str:
    prompt_profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue)
    issue_tokens = _issue_relevance_tokens(integrated_issue)
    if not issue_tokens:
        return "none"
    if scope == "skax":
        profile = prompt_profile.get("skax_profile") or {}
        if not isinstance(profile, dict) or not profile:
            return "none"
        return _linkage_level_from_match_count(
            len(issue_tokens & _content_tokens(_json_dumps(profile)))
        )
    peer_profiles = prompt_profile.get("peer_profiles") or {}
    if not isinstance(peer_profiles, dict):
        return "none"
    best = "none"
    for company_id in _companies_from_integrated_issue(integrated_issue):
        profile = peer_profiles.get(company_id) or {}
        if not isinstance(profile, dict) or not profile:
            continue
        level = _linkage_level_from_match_count(
            len(issue_tokens & _content_tokens(_json_dumps(profile)))
        )
        if _linkage_rank(level) > _linkage_rank(best):
            best = level
    return best


def _linkage_level_from_match_count(count: int) -> str:
    if count >= 4:
        return "high"
    if count >= 2:
        return "medium"
    if count >= 1:
        return "low"
    return "none"


def _linkage_rank(level: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(str(level), 0)


def _mentions_profile_based_peer_claim(text: str) -> bool:
    return bool(re.search(r"사업\s*영역|사업영역|역량|프로필|제공|수행|운영|지원", text or ""))


def _unsupported_peer_profile_claim_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str | None:
    if not label.startswith(
        ("peer_implication.peer_meaning", "peer_implication.capability_change")
    ):
        return None
    if _has_relevant_peer_profile_context(profile_context, integrated_issue=integrated_issue):
        return None
    if not _mentions_profile_based_peer_claim(text):
        return None
    if re.search(
        r"부족|확인되지|단정하기\s*어렵|단정할\s*수\s*없|단정하지|아니라|"
        r"사건\s*기반|낮춰|후속.{0,12}확인|해석하는\s*것이\s*안전",
        text or "",
    ):
        return None
    return (
        "현재 사건과 직접 맞는 피어 프로필 접점이 없는데 사업영역/역량 기반 "
        "시사점처럼 썼습니다. 사건 기반 해석으로 낮춰야 합니다."
    )


def _counterparty_guard_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> bool:
    value_text = str(text or "").strip()
    if not value_text:
        return False
    return bool(
        _supplier_role_overstatement_violation(
            value_text,
            integrated_issue=integrated_issue,
        )
        or _supplier_financial_focus_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        or _counterparty_capability_overclaim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        or _counterparty_role_action_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
    )


def _safe_event_based_strategic_meanings(
    values: Any,
    *,
    integrated_issue: dict[str, Any],
) -> list[str]:
    safe_items: list[str] = []
    for index, item in enumerate(_string_list(values, max_items=3), start=1):
        if (
            _counterparty_guard_violation(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
            )
            or _hard_quality_violation_for_text(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
            )
            or _weak_analysis_statement(
                item,
                integrated_issue=integrated_issue,
            )
        ):
            continue
        safe_items.append(item)
    if len(safe_items) >= 2:
        return safe_items[:3]

    for candidate in _event_based_strategic_meaning_candidates(integrated_issue):
        if candidate not in safe_items:
            safe_items.append(candidate)
        if len(safe_items) >= 3:
            break
    return safe_items[:3]


def _event_based_analysis_field(key: str, *, integrated_issue: dict[str, Any]) -> str:
    fact = _primary_issue_fact(integrated_issue)
    if key == "analysis_summary":
        return _event_based_analysis_summary(integrated_issue)
    if key == "market_signal":
        return _event_based_market_signal(integrated_issue)
    if key == "impact_reason":
        return _event_based_impact_reason(integrated_issue)
    if key == "reason":
        return (
            f"{fact} 이 사실을 기준으로 해석하되, 계약 상대방의 수행·운영 역할은 "
            "원문에서 확인되는 범위로만 제한했습니다."
        )
    return fact


def _event_based_analysis_summary(integrated_issue: dict[str, Any]) -> str:
    fact = _primary_issue_fact(integrated_issue)
    target = _main_company_display(integrated_issue)
    frame_kind = _issue_frame_kind(integrated_issue)
    if frame_kind == "performance":
        return f"{fact} 이 사건은 특정 사업 확장보다 전사 실적과 수익성 지표를 관찰하는 카드입니다."
    if _main_company_is_customer_or_buyer(integrated_issue) and target:
        return (
            f"{fact} {target}는 원문상 계약 상대방으로 확인되며, 이 이슈는 계약 "
            "대상 시스템·범위·기간이 구체화된 사건으로 해석하는 것이 안전합니다."
        )
    return fact


def _profile_linked_analysis_field(
    key: str,
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    if key == "analysis_summary":
        return _profile_linked_analysis_summary(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
    if key == "market_signal":
        return _profile_linked_market_signal(integrated_issue)
    if key == "impact_reason":
        return _profile_linked_impact_reason(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
    if key == "reason":
        profile_phrase = _profile_area_phrase(
            profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
        fact = _primary_issue_fact(integrated_issue)
        if profile_phrase:
            return (
                f"{fact} 이 사실을 기준으로 해석했고, 피어 프로필에서는 "
                f"{profile_phrase} 접점만 현재 사건과 연결했습니다."
            )
        return f"{fact} 이 사실을 기준으로 사건 범위 안에서만 해석했습니다."
    return _event_based_analysis_field(key, integrated_issue=integrated_issue)


def _profile_linked_analysis_summary(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    fact = _primary_issue_fact(integrated_issue)
    subject = _issue_subject_phrase(integrated_issue)
    if _issue_frame_kind(integrated_issue) == "performance":
        return _event_based_analysis_summary(integrated_issue)
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if subject and profile_phrase:
        return (
            f"{fact} 이 사건은 피어 프로필의 {profile_phrase} 맥락이 "
            f"{subject}와 연결되는 신호로 해석할 수 있습니다."
        )
    return _event_based_analysis_summary(integrated_issue)


def _profile_linked_market_signal(integrated_issue: dict[str, Any]) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    if _issue_frame_kind(integrated_issue) == "performance":
        return _event_based_market_signal(integrated_issue)
    issue_frame = _issue_frame_for_prompt(integrated_issue)
    axes = ", ".join(_decision_axes_from_issue_frame(issue_frame)[:3])
    if axes:
        return (
            f"{subject or '현재 사건'}에서는 {axes}가 함께 확인됩니다. "
            "따라서 유사 동향은 발표 여부만이 아니라 이 조건들이 실제 수행 단계에서 "
            "어떻게 공개되는지를 기준으로 비교해야 합니다."
        )
    return _event_based_market_signal(integrated_issue)


def _profile_linked_impact_reason(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    if _issue_frame_kind(integrated_issue) == "performance":
        return _event_based_impact_reason(integrated_issue)
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if subject and profile_phrase:
        issue_frame = _issue_frame_for_prompt(integrated_issue)
        axes = ", ".join(_decision_axes_from_issue_frame(issue_frame)[:2])
        return (
            f"{subject}이 확인되면서 해당 기업의 기존 사업 흐름 중 {profile_phrase}가 "
            f"현재 사건의 {axes or '대상 범위와 후속 조건'}와 이어지는지 관찰할 수 있습니다. "
            "다만 이 문장은 역할 확정이 아니라, 어떤 근거가 더 확인되어야 하는지 가르는 "
            "비교 기준으로 해석해야 합니다."
        )
    return _event_based_impact_reason(integrated_issue)


def _profile_linked_strategic_meanings(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> list[str]:
    fact = _primary_issue_fact(integrated_issue)
    subject = _issue_subject_phrase(integrated_issue)
    if _issue_frame_kind(integrated_issue) == "performance":
        return _event_based_strategic_meaning_candidates(integrated_issue)[:3]
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    meanings = [fact]
    if subject and profile_phrase:
        issue_frame = _issue_frame_for_prompt(integrated_issue)
        axes = ", ".join(_decision_axes_from_issue_frame(issue_frame)[:2])
        meanings.append(
            f"해당 기업의 기존 사업 흐름 중 {profile_phrase}와 연결해 보면, 이번 사건은 "
            f"{subject}에서 {axes or '대상 범위와 후속 조건'}가 실제로 어떻게 드러나는지 "
            "확인하게 하는 신호입니다."
        )
    meanings.append(_profile_linked_market_signal(integrated_issue))
    return _normalize_recommended_actions(meanings)[:3]


def _event_based_market_signal(integrated_issue: dict[str, Any]) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    frame_kind = _issue_frame_kind(integrated_issue)
    if frame_kind == "performance":
        return (
            "현재 근거에서는 매출, 이익, 이익률 같은 전사 재무 지표가 확인됩니다. "
            "사업부별 기여도가 공개되지 않으면 특정 사업영역 성과로 연결하지 않는 것이 안전합니다."
        )
    if frame_kind == "launch_or_service":
        return (
            f"{subject or '현재 서비스 신호'}에서 적용 업무와 효과 수치가 함께 확인됩니다. "
            "유사 동향은 기능 공개 여부보다 어떤 업무 시간이 줄었는지, 어떤 사용 대상의 "
            "적응·처리 시간이 바뀌었는지를 비교해야 합니다."
        )
    if frame_kind == "selection_or_build":
        return (
            f"{subject or '현재 인프라 사업'}에서 참여 구조, 자원 규모, 구축 일정이 함께 "
            "제시됐습니다. 유사 대형 인프라 동향은 선정 여부보다 실제 구축·운영 범위와 "
            "후속 일정이 비교 기준이 됩니다."
        )
    duration = _contract_duration_phrase(integrated_issue)
    scale = _contract_scale_phrase(integrated_issue)
    details = " ".join(item for item in (scale, duration) if item)
    if subject and details:
        return f"{subject}이 실제 계약 단위에서 확인됐고, {details}이 함께 제시됐습니다."
    if subject:
        return f"{subject}이 실제 계약 단위에서 확인됐습니다."
    return "현재 사건에서 대상 시스템과 계약 범위가 구체화된 신호가 확인됩니다."


def _event_based_impact_reason(integrated_issue: dict[str, Any]) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    target = _main_company_display(integrated_issue)
    frame_kind = _issue_frame_kind(integrated_issue)
    if frame_kind == "performance":
        return (
            "매출 증가율, 영업이익 증가율, 순이익 증가율, 영업이익률이 함께 제시되어 "
            "피어사의 전사 수익성 흐름을 비교할 근거가 됩니다."
        )
    if frame_kind == "launch_or_service":
        return (
            "플랫폼·서비스 공개 사실과 함께 적용 후 시간 절감 또는 적응 기간 단축 수치가 "
            "제시되어, 유사 AI 업무혁신 동향을 기능명보다 업무 성과 지표로 비교할 근거가 됩니다."
        )
    if frame_kind == "selection_or_build":
        return (
            "사업자 선정, 참여 구조, 인프라 규모, 착공·구축 일정이 함께 제시되어 "
            "대형 인프라 동향을 실행 조건 중심으로 비교할 근거가 됩니다."
        )
    if _main_company_is_customer_or_buyer(integrated_issue) and target and subject:
        return (
            f"{target}의 역할은 계약 상대방으로 확인되는 수준이지만, {subject}의 "
            "계약 범위와 기간이 제시되어 유사 사업에서 비교할 전환 범위와 일정 기준을 "
            "관찰할 수 있습니다."
        )
    if subject:
        return (
            f"{subject}의 계약 범위와 기간이 제시되어 유사 사업의 비교 기준을 관찰할 수 있습니다."
        )
    return "현재 근거에서 계약 범위와 대상 시스템이 확인되어 후속 비교 기준을 관찰할 수 있습니다."


def _event_based_strategic_meaning_candidates(integrated_issue: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    fact = _primary_issue_fact(integrated_issue)
    subject = _issue_subject_phrase(integrated_issue)
    target = _main_company_display(integrated_issue)
    frame_kind = _issue_frame_kind(integrated_issue)
    if fact:
        candidates.append(fact)
    if frame_kind == "performance":
        candidates.append(
            "이번 근거는 특정 사업영역의 확정 성과가 아니라 매출, 이익, 이익률을 분리해 "
            "피어사의 전사 수익성 변화를 관찰해야 하는 실적 신호입니다."
        )
        candidates.append(
            "사업부별 기여도가 원문에 공개되지 않았다면 물류, AI, 클라우드 같은 개별 사업 성과로 "
            "단정하지 말고 후속 IR·실적자료에서 기여 사업과 수익성 지속성을 확인해야 합니다."
        )
        return candidates[:3]
    if frame_kind == "launch_or_service":
        candidates.append(
            f"{subject or '현재 서비스'}는 단순 공개 사실보다 적용 업무와 효과 수치가 함께 "
            "제시된 사례입니다."
        )
        candidates.append(
            "유사 AI 업무혁신 동향은 기능명보다 어떤 업무 시간이 줄었는지, 어떤 사용자군의 "
            "적응 기간이 단축됐는지를 기준으로 비교해야 합니다."
        )
        return candidates[:3]
    if frame_kind == "selection_or_build":
        candidates.append(
            f"{subject or '현재 인프라 사업'}는 선정 사실뿐 아니라 참여 구조, 자원 규모, "
            "구축 일정이 함께 제시된 대형 인프라 실행 신호입니다."
        )
        candidates.append(
            "유사 사업에서는 기술명보다 참여사 역할, 자원·투자 규모, 착공·구축 일정, "
            "운영 책임 공개 여부가 후속 비교 기준이 됩니다."
        )
        return candidates[:3]
    if subject:
        candidates.append(
            f"{subject}이 기사에서 확인된 만큼, 이 이슈는 단순 기능 도입보다 "
            "대상 시스템의 전환 범위, 업무 영향도, 운영 안정성 기준을 함께 봐야 하는 사건입니다."
        )
        candidates.append(
            f"유사 사업에서는 {subject}의 기능 구현 여부만이 아니라 기존 시스템과의 "
            "연계 방식, 전환 일정, 장애 대응 기준까지 비교 기준으로 제시될 수 있습니다."
        )
    scale = _contract_scale_phrase(integrated_issue)
    duration = _contract_duration_phrase(integrated_issue)
    if scale or duration:
        candidates.append(
            " ".join(
                part
                for part in (
                    scale,
                    duration,
                    (
                        "이 함께 확인되어 단기 개선보다 일정 규모의 업무 시스템 "
                        "전환 과제로 해석할 수 있습니다."
                    ),
                )
                if part
            )
        )
    if _main_company_is_customer_or_buyer(integrated_issue) and target:
        candidates.append(
            f"{target}는 계약 상대방으로 확인되지만, 최종 발주자 여부나 수행·운영 책임은 "
            "원문만으로 단정하기 어렵습니다."
        )
    return [item for item in candidates if item]


def _weak_analysis_statement(text: str, *, integrated_issue: dict[str, Any]) -> bool:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return True
    primary = _primary_issue_fact(integrated_issue).rstrip(".")
    if value.rstrip(".") == primary:
        return True
    return bool(
        re.fullmatch(r".{0,40}(중요|변화|관찰|시사)(하|되|되고|된다|고 있다).{0,20}", value)
    )


def _issue_subject_phrase(integrated_issue: dict[str, Any]) -> str:
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        main_event_facts = [
            item
            for item in [
                *(intelligence.get("common_facts") or []),
                *(intelligence.get("unique_facts") or []),
            ]
            if isinstance(item, dict) and _fact_has_summary_role(item, "main_event")
        ]
        for item in main_event_facts:
            for value in _jsonish_list(item.get("products_or_services")):
                text = re.sub(r"\s+", " ", str(value or "").strip(" ."))
                if text and not _is_bad_issue_subject(text):
                    return text
        for item in main_event_facts:
            if subject := _extract_issue_subject_from_text(str(item.get("fact") or "")):
                return subject
        for value in _jsonish_list(intelligence.get("products_or_services")):
            text = re.sub(r"\s+", " ", str(value or "").strip(" ."))
            if text and not _is_bad_issue_subject(text):
                return text
        for item in intelligence.get("common_facts") or []:
            if not isinstance(item, dict):
                continue
            if subject := _extract_issue_subject_from_text(str(item.get("fact") or "")):
                return subject
        for item in intelligence.get("unique_facts") or []:
            if not isinstance(item, dict):
                continue
            for value in _jsonish_list(item.get("products_or_services")):
                text = re.sub(r"\s+", " ", str(value or "").strip(" ."))
                if text and not _is_bad_issue_subject(text):
                    return text
            if subject := _extract_issue_subject_from_text(str(item.get("fact") or "")):
                return subject

    issue_text = " ".join(
        str(integrated_issue.get(key) or "").strip()
        for key in ("main_issue", "main_event", "headline", "one_line_summary")
    )
    if subject := _extract_issue_subject_from_text(issue_text):
        return subject
    return ""


def _fact_has_summary_role(item: dict[str, Any], role_name: str) -> bool:
    roles = {str(role or "") for role in _jsonish_list(item.get("summary_roles"))}
    role = str(item.get("summary_role") or "")
    return role_name in roles or role == role_name


def _is_bad_issue_subject(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    return bool(
        re.search(
            r"팀장|전무|대표|부사장|상무|임원|기자|발표|주제|기념촬영|행사|포럼|세미나",
            value,
        )
    )


def _extract_issue_subject_from_text(text: str) -> str:
    issue_text = re.sub(r"\s+", " ", str(text or "").strip())
    if not issue_text:
        return ""
    match = re.search(
        r"([가-힣A-Za-z0-9&·+\s'‘’\"“”_-]{2,100}?"
        r"(?:센터|시스템|플랫폼|인프라|단말|솔루션|서비스|사업|계약)"
        r"[가-힣A-Za-z0-9&·+\s'‘’\"“”_-]{0,40}?"
        r"(?:전환|현대화|구축|도입|개편|고도화|선정|확정|계약|사업|센터)?)",
        issue_text,
    )
    if match:
        subject = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        subject = re.sub(
            r"^(?:[가-힣A-Za-z0-9&·+\s'‘’\"“”_-]{1,30}?(?:이|가|은|는|와|과)\s+)",
            "",
            subject,
        ).strip(" .")
        if _is_bad_issue_subject(subject):
            return ""
        return subject
    return ""


def _contract_scale_phrase(integrated_issue: dict[str, Any]) -> str:
    numbers = integrated_issue.get("key_numbers") or []
    if isinstance(numbers, list):
        phrases: list[str] = []
        for item in numbers:
            if not isinstance(item, dict):
                continue
            label = str(item.get("metric_label") or item.get("metric_name") or "").strip()
            value = item.get("value")
            unit = str(item.get("unit") or "").strip()
            if value in (None, ""):
                continue
            if re.search(r"계약|금액|매출|비율|규모|amount|revenue|ratio|percent", label, re.I):
                phrases.append(f"{label} {value}{unit}".strip())
        if phrases:
            return ", ".join(phrases[:2])

    evidence = _integrated_grounding_text(integrated_issue)
    matches = _NUMERIC_TOKEN_PATTERN.findall(evidence)
    return ", ".join(
        list(dict.fromkeys(str(match).strip() for match in matches if str(match).strip()))[:2]
    )


def _contract_duration_phrase(integrated_issue: dict[str, Any]) -> str:
    evidence = _integrated_grounding_text(integrated_issue)
    date_matches = re.findall(
        r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|\d{4}[-.]\d{1,2}[-.]\d{1,2}",
        evidence,
    )
    unique_dates = list(dict.fromkeys(re.sub(r"\s+", " ", item).strip() for item in date_matches))
    if len(unique_dates) >= 2:
        return f"계약 기간 {unique_dates[0]}~{unique_dates[1]}"
    return ""


def _main_company_display(integrated_issue: dict[str, Any]) -> str:
    main_company = str(integrated_issue.get("main_company") or "").strip()
    if not main_company:
        return ""
    for alias in expand_peer_aliases(main_company):
        text = str(alias or "").strip()
        if text and not re.fullmatch(r"[a-z0-9_]+", text, flags=re.IGNORECASE):
            return text
    return main_company


def _profile_linked_peer_meaning(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    peer: dict[str, Any],
) -> str:
    peer_name = str(peer.get("company_name_ko") or peer.get("company_id") or "타깃 피어").strip()
    if _issue_frame_kind(integrated_issue) == "performance":
        return _event_based_peer_meaning(integrated_issue=integrated_issue, peer=peer)
    fact = _primary_issue_fact(integrated_issue)
    fact_sentence = fact
    if peer_name and not re.search(re.escape(peer_name), fact_sentence, flags=re.IGNORECASE):
        fact_sentence = f"{peer_name}는 {fact_sentence}"
    subject = _issue_subject_phrase(integrated_issue) or "현재 사건"
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if profile_phrase:
        connection_reason = _issue_profile_connection_reason(
            integrated_issue=integrated_issue,
            profile_phrase=profile_phrase,
        )
        return (
            f"{fact_sentence} 해당 기업의 기존 사업 흐름에서는 {profile_phrase}가 "
            f"확인되고, 이번 사건에서는 {subject}가 확인됩니다. "
            f"{connection_reason} 다만 현재 근거만으로는 해당 기업이 어느 수행 범위와 "
            "운영 책임까지 맡는지 모두 확정하기 어렵기 때문에, 역할 확장이나 성과보다 "
            "기존 사업 흐름이 실제 적용 장면과 이어지는 관찰 신호로 보는 것이 안전합니다."
        )
    return _event_based_peer_meaning(integrated_issue=integrated_issue, peer=peer)


def _profile_linked_capability_change(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    subject = _issue_subject_phrase(integrated_issue) or "현재 사건"
    if _issue_frame_kind(integrated_issue) == "performance":
        return _event_based_capability_change(integrated_issue)
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if profile_phrase:
        connection_reason = _issue_profile_connection_reason(
            integrated_issue=integrated_issue,
            profile_phrase=profile_phrase,
        )
        return (
            f"확인된 변화는 역량 확장 자체가 아니라 {subject}의 대상 업무·서비스·추진 구조가 "
            f"해당 기업의 기존 사업 흐름 중 {profile_phrase}와 비교할 수 있는 형태로 "
            f"드러났다는 점입니다. {connection_reason} 따라서 후속 비교는 해당 기업이 "
            "어떤 적용 범위, 운영 역할, 성과 지표를 실제로 공개하는지에 맞춰야 합니다."
        )
    return _event_based_capability_change(integrated_issue)


def _issue_profile_connection_reason(
    *,
    integrated_issue: dict[str, Any],
    profile_phrase: str,
) -> str:
    issue_terms = [
        *_checkpoint_issue_terms(_issue_frame_for_prompt(integrated_issue)),
        *_high_specific_issue_terms_for_quality(integrated_issue)[:5],
    ]
    clean_terms = [
        _clean_issue_axis_phrase(term) for term in issue_terms if _clean_issue_axis_phrase(term)
    ]
    clean_terms = _unique_texts(clean_terms, max_items=4)
    if clean_terms:
        return (
            f"연결 근거는 현재 사건의 {', '.join(clean_terms[:3])} 같은 구체 항목이 "
            f"{profile_phrase}의 적용 장면을 판단하게 해준다는 점입니다."
        )
    return (
        f"연결 근거는 현재 사건의 사업명·적용 범위가 {profile_phrase}와 비교할 수 있는 "
        "축을 제공한다는 점입니다."
    )


def _profile_area_phrase(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> str:
    names = _relevant_profile_area_names(
        profile_context,
        integrated_issue=integrated_issue,
        scope=scope,
    )
    return "·".join(names[:3])


def _relevant_profile_area_names(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> list[str]:
    prompt_profile = _profile_for_prompt(profile_context, integrated_issue=integrated_issue)
    profiles: list[dict[str, Any]] = []
    if scope == "skax":
        skax = prompt_profile.get("skax_profile") or {}
        if isinstance(skax, dict):
            profiles.append(skax)
    else:
        peer_profiles = prompt_profile.get("peer_profiles") or {}
        if isinstance(peer_profiles, dict):
            for company_id in _companies_from_integrated_issue(integrated_issue):
                profile = peer_profiles.get(company_id) or {}
                if isinstance(profile, dict):
                    profiles.append(profile)
    relevance_tokens = _issue_relevance_tokens(integrated_issue)
    names: list[str] = []
    for profile in profiles:
        business_areas = profile.get("business_areas") or []
        if not isinstance(business_areas, list):
            continue
        ranked = _rank_relevant_profile_items(
            [item for item in business_areas if isinstance(item, dict)],
            relevance_tokens=relevance_tokens,
            max_items=5,
        )
        for area in ranked:
            name = str(area.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
    if names:
        return names
    return _relevant_profile_named_terms(
        profiles,
        relevance_tokens=relevance_tokens,
        integrated_issue=integrated_issue,
        max_items=3,
    )


def _relevant_profile_named_terms(
    profiles: list[dict[str, Any]],
    *,
    relevance_tokens: set[str],
    integrated_issue: dict[str, Any],
    max_items: int,
) -> list[str]:
    company_terms = _company_identity_terms(integrated_issue)
    candidates: list[str] = []
    for profile in profiles:
        for key in (
            "core_capabilities",
            "strategic_focus",
            "priority_initiatives",
            "key_products_services",
            "recent_changes",
        ):
            for value in _jsonish_list(profile.get(key))[:12]:
                if isinstance(value, dict):
                    text = str(value.get("name") or value.get("summary") or "").strip()
                else:
                    text = str(value or "").strip()
                if not text:
                    continue
                tokens = _content_tokens(text)
                if (
                    tokens
                    and tokens - company_terms
                    and (not relevance_tokens or tokens & relevance_tokens)
                ):
                    candidates.append(text)
    return list(dict.fromkeys(candidates))[:max_items]


def _event_based_peer_meaning(
    *,
    integrated_issue: dict[str, Any],
    peer: dict[str, Any],
) -> str:
    peer_name = str(peer.get("company_name_ko") or peer.get("company_id") or "타깃 피어").strip()
    fact = _primary_issue_fact(integrated_issue)
    frame_kind = _issue_frame_kind(integrated_issue)
    if frame_kind == "performance":
        return (
            f"{fact} {peer_name}의 전사 매출·이익 지표가 개선된 실적 신호로 볼 수 있습니다. "
            "다만 현재 근거만으로는 어떤 사업영역이 실적 개선을 견인했는지 단정할 수 없으므로, "
            "특정 기술 성과나 사업 확장보다 재무 성과 관찰 카드로 해석하는 것이 안전합니다."
        )
    if frame_kind == "launch_or_service":
        return (
            f"{fact} {peer_name}의 현재 움직임은 서비스 공개 자체보다 적용 업무와 효과 수치가 "
            "함께 제시됐다는 점에서 관찰할 수 있습니다. 다만 프로필 접점이 충분하지 않으면 "
            "기존 역량 확장으로 단정하지 말고, 업무 적용 사례와 성과 지표가 확인된 사건으로 "
            "낮춰 해석하는 것이 안전합니다."
        )
    if frame_kind == "selection_or_build":
        return (
            f"{fact} {peer_name}의 현재 움직임은 선정 사실뿐 아니라 참여 구조, 자원 규모, "
            "구축 일정이 함께 제시된 대형 인프라 실행 신호로 볼 수 있습니다. 프로필 접점이 "
            "부족하면 역량 강화로 단정하지 말고, 후속 역할·운영 책임 공개를 확인해야 합니다."
        )
    if _main_company_is_customer_or_buyer(integrated_issue):
        return (
            f"{fact} {peer_name}는 원문상 계약 상대방으로 확인되지만, 최종 발주자 "
            "여부나 수행·운영 책임 범위까지는 단정하기 어렵습니다. 따라서 피어사 "
            "관점에서는 역할 확장으로 단정하지 않고, 금융권 핵심 시스템 전환 과제와 "
            "연결된 관찰 신호로 해석하는 것이 안전합니다."
        )
    return (
        f"{fact} 현재 사건과 직접 맞는 피어 프로필 접점이 충분하지 않아, "
        "이 신호는 사건 기반 1차 해석으로 보는 것이 안전합니다."
    )


def _event_based_capability_change(integrated_issue: dict[str, Any]) -> str:
    frame_kind = _issue_frame_kind(integrated_issue)
    if frame_kind == "performance":
        return (
            "확인된 변화는 특정 사업역량의 확장 자체가 아니라 매출 증가율, 영업이익 증가율, "
            "순이익 증가율, 영업이익률 같은 전사 재무 지표가 함께 개선됐다는 점입니다. "
            "사업부별 기여도는 후속 실적자료나 IR에서 별도로 확인해야 합니다."
        )
    if frame_kind == "launch_or_service":
        return (
            "확인된 변화는 역량 확장 자체가 아니라 적용 업무와 효과 수치가 구체화됐다는 점입니다. "
            "플랫폼·서비스가 어떤 업무에 쓰였고 시간이 얼마나 줄었는지 "
            "후속 사례와 함께 봐야 합니다."
        )
    if frame_kind == "selection_or_build":
        return (
            "확인된 변화는 역량 강화 자체가 아니라 선정 이후 참여 구조, 자원 규모, 구축 일정이 "
            "구체화됐다는 점입니다. 실제 수행 범위와 운영 책임은 "
            "후속 협약·공시에서 확인해야 합니다."
        )
    subject = _issue_subject_phrase(integrated_issue) or "확인된 사업"
    duration = _contract_duration_phrase(integrated_issue)
    duration_text = f" {duration}도 함께 확인됩니다." if duration else ""
    return (
        f"확인된 변화는 피어사의 확정된 역할 변화가 아니라 {subject}의 대상 시스템과 "
        f"계약 범위가 구체화된 점입니다.{duration_text} 유사 사업에서는 전환 범위, "
        "업무 영향도, 일정 기준을 함께 비교해야 한다는 신호로 볼 수 있습니다."
    )


def _primary_issue_fact(integrated_issue: dict[str, Any]) -> str:
    for key in ("one_line_summary", "integrated_text", "main_event", "main_issue", "headline"):
        value = str(integrated_issue.get(key) or "").strip()
        if value:
            return _ensure_sentence(value)
    for value in _string_list(integrated_issue.get("fact_summary"), max_items=1):
        if value:
            return _ensure_sentence(value)
    for _, fact_text in _fact_texts(integrated_issue):
        if fact_text:
            return _ensure_sentence(fact_text)
    return "현재 사건에서 확인된 사실이 있습니다."


def _ensure_sentence(text: str) -> str:
    sentence = re.sub(r"\s+", " ", str(text or "").strip())
    if not sentence:
        return ""
    return sentence if sentence.endswith((".", "다.", "요.", "임.")) else f"{sentence}."


def _hard_quality_violation_for_text(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> bool:
    value_text = str(text or "").strip()
    if not value_text:
        return False
    integrated_evidence_text = _integrated_grounding_text(integrated_issue)
    if any(
        _has_unsupported_pattern(value_text, pattern, evidence_text=integrated_evidence_text)
        for pattern in _UNSUPPORTED_CLAIM_PATTERNS
    ):
        return True
    return bool(
        _relationship_grounding_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            integrated_evidence_text=integrated_evidence_text,
        )
        or _supplier_role_overstatement_violation(
            value_text,
            integrated_issue=integrated_issue,
        )
        or _supplier_financial_focus_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        or _counterparty_capability_overclaim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        or _counterparty_role_action_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
    )


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
        (r"([가-힣A-Za-z0-9&·+_\-\s]+?)의\s*공급\s*역량", r"\1의 계약 범위와 사업영역 접점"),
        (r"([가-힣A-Za-z0-9&·+_\-\s]+?)\s*공급\s*역량", r"\1 계약 범위와 사업영역 접점"),
        (r"전략적\s*방향과\s*일치", "프로필상 사업영역과 연결"),
        (r"전략과의\s*일관성", "프로필상 사업영역과의 접점"),
        (
            r"프로젝트[가은]\s*유사한\s*고객군과\s*사업\s*영역에서의\s*기회를\s*제공합니다",
            "계약 신호는 유사 고객군과 사업 영역에서 참고할 사업영역 접점을 보여줍니다",
        ),
        (r"기회를\s*제공하는\s*것", "참고 근거가 되는 것"),
        (r"기회를\s*제공하는\s*것으로", "참고 근거로"),
        (r"기회를\s*제공할\s*수\s*있습니다", "참고 근거가 될 수 있습니다"),
        (r"기회를\s*제공합니다", "참고 근거가 됩니다"),
        (r"프로젝트에\s*참여하여", "프로젝트와 연결되어"),
        (r"프로젝트에\s*참여", "프로젝트와 연결"),
        (r"사업에\s*참여하여", "사업과 연결되어"),
        (r"사업에\s*참여", "사업과 연결"),
        (r"기여하고\s*있습니다", "사업영역 접점을 보여줍니다"),
        (r"공급\s*역량", "계약 범위와 사업영역 접점"),
        (r"납품\s*역량", "계약 범위와 사업영역 접점"),
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
    if token in {
        "계약",
        "계약은",
        "계약을",
        "사업",
        "수주",
        "체결",
        "체결했다",
        "규모",
        "규모로",
        "규모의",
        "기간",
        "기간은",
        "최근",
        "대비",
        "해당",
        "해당하",
        "알려졌다",
        "총액",
        "원이다",
        "일자는",
        "전체",
        "일부",
        "대상",
        "관련",
        "확인",
        "확인된",
    }:
        return True
    return False


def _normalize_content_token(token: str) -> str:
    token = token.strip()
    if len(token) <= 3:
        return token
    return re.sub(r"(으로|에서|에게|과|와|은|는|이|가|을|를|의)$", "", token)


def _issue_relevance_tokens(
    integrated_issue: dict[str, Any],
    *,
    extra_text: str = "",
) -> set[str]:
    if not isinstance(integrated_issue, dict):
        return set()
    parts: list[str] = []
    if extra_text:
        parts.append(extra_text)
    for key in (
        "headline",
        "main_event",
        "main_issue",
        "one_line_summary",
        "integrated_text",
        "cluster_event_type",
    ):
        value = str(integrated_issue.get(key) or "").strip()
        if value:
            parts.append(value)
    parts.extend(str(item or "") for item in integrated_issue.get("fact_summary") or [])
    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for key in ("products_or_services", "customers_or_industries", "activity_types"):
            parts.extend(str(item or "") for item in _jsonish_list(intelligence.get(key)))
        for item in intelligence.get("unique_facts") or []:
            if not isinstance(item, dict):
                continue
            parts.append(str(item.get("fact") or ""))
            for key in ("products_or_services", "customers_or_industries", "activity_types"):
                parts.extend(str(value or "") for value in _jsonish_list(item.get(key)))
    tokens = _content_tokens("\n".join(parts))
    company_tokens: set[str] = set()
    for company_id in _companies_from_integrated_issue(integrated_issue):
        company_tokens.update(_content_tokens(" ".join(expand_peer_aliases(company_id))))
        company_tokens.update(_company_token_variants(company_id))
    for supplier in _supplier_names_for_target_counterparty(integrated_issue):
        company_tokens.update(_content_tokens(supplier))
        company_tokens.add(_normalize_entity_token(supplier))
    filtered = {
        token
        for token in tokens
        if token not in company_tokens and not _is_low_signal_profile_relevance_token(token)
    }
    return _expand_profile_relevance_tokens(filtered)


def _expand_profile_relevance_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    joined = " ".join(tokens)
    if re.search(r"코어\s*뱅킹|뱅킹|은행|증권|보험|결제|카드|토큰증권|스테이블코인", joined):
        expanded.add("금융")
    if re.search(r"물류|창고|배송|로봇|rx", joined, flags=re.IGNORECASE):
        expanded.add("물류")
        expanded.add("로봇")
    if re.search(r"보안|권한|접근|프라이버시|개인정보", joined):
        expanded.add("보안")
    return expanded


def _profile_relevance_hint_text(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    bundle: dict[str, Any],
) -> str:
    parts: list[str] = []
    parts.extend(_string_list(classification.get("sectors"), max_items=10))
    parts.extend(_string_list(classification.get("matched_sectors"), max_items=10))
    sector = str(classification.get("sector") or "").strip()
    if sector:
        parts.append(sector)
    for detail in _jsonish_list(classification.get("matched_sector_details")):
        if isinstance(detail, dict):
            parts.append(str(detail.get("sector_name_ko") or ""))
            parts.append(str(detail.get("keyword") or ""))

    metadata = bundle.get("metadata") or {}
    for detail in _jsonish_list(metadata.get("matched_sector_details")):
        if isinstance(detail, dict):
            parts.append(str(detail.get("sector_name_ko") or ""))
            parts.append(str(detail.get("keyword") or ""))

    representative_id = str(
        integrated_issue.get("representative_id")
        or (bundle.get("metadata") or {}).get("representative_id")
        or ""
    ).strip()
    candidate_items = [
        item for item in _jsonish_list(bundle.get("items")) if isinstance(item, dict)
    ]
    if representative_id:
        selected_items = [
            item
            for item in candidate_items
            if str(item.get("id") or "").strip() == representative_id
        ]
    else:
        selected_items = candidate_items[:1]
    for item in selected_items[:1]:
        if not isinstance(item, dict):
            continue
        parts.append(str(item.get("title") or ""))
        metadata_raw = item.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
        parts.append(str(metadata.get("subtitle") or ""))
        parts.append(str(item.get("content") or "")[:700])

    for key in ("main_event", "main_issue", "one_line_summary"):
        parts.append(str(integrated_issue.get(key) or ""))
    return "\n".join(part for part in parts if part)


def _is_low_signal_profile_relevance_token(token: str) -> bool:
    if re.search(r"\d", token):
        return True
    if re.search(r"(했다|한다고|있다|있다고|됐다|되면|된다|이며|으로)$", token):
        return True
    return token in {
        "공급계약",
        "contract",
        "AI·DX",
        "ax",
        "deal",
        "개발",
        "표준",
        "플랫폼",
        "포함한다",
        "규모이다",
        "itdaily",
        "kr",
        "seungyang",
        "fast-pass",
        "규모다",
        "매출액",
        "계약금",
        "계약금액",
        "금액",
        "원으로",
        "지난",
        "연결기준",
        "총액은",
        "시장",
        "분야",
        "기업",
        "전문기업",
        "역할",
        "직접",
        "진행",
        "수행",
        "이벤트",
        "이번",
        "통해",
        "전환",
        "현대화",
        "기반",
        "기간",
        "사업",
        "프로젝트",
        "주요",
        "추진",
        "제공",
        "확대",
        "시스템",
        "공시를",
        "근거가",
        "기사로",
        "내년",
        "덧붙였다",
        "동종",
        "드러냈다",
        "발주사와",
        "밝혔다",
        "변경을",
        "부가세",
        "사진",
        "수금",
        "실적이",
        "아이티데일리",
        "안내할",
        "예정이라고",
        "없이",
        "이행",
        "입지를",
        "전했다",
        "정정공시",
        "제외한",
        "조건",
        "조건이",
        "중이며",
        "체결하며",
        "체결했다고",
        "최종",
        "판단",
        "피어사",
        "한편",
        "해당한다",
        "핵심",
        "협의를",
        "확정되면",
        "회사는",
    }


def _shrink_profile(profile: Any, *, relevance_tokens: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    relevance_tokens = relevance_tokens or set()
    identity_keys = (
        "company_id",
        "peer_id",
        "company_name",
        "company_name_ko",
        "business_lines",
    )
    scalar_keys = (
        # Broad company summaries often contain multiple business areas and can
        # pull the model toward an unrelated profile branch. Use structured
        # business areas and relevant examples instead.
    )
    relevant_item_keys = (
        # Keep the prompt centered on profile structure. Detailed profile
        # examples can overpower the current IntegratedIssue when the profile
        # snapshot is broad or noisy.
    )
    passthrough_keys = (
        "business_areas",
        "core_capabilities",
        "recent_keywords",
        "capability_evolution",
        "cautions",
    )
    out: dict[str, Any] = {}
    for key in identity_keys:
        if key not in profile:
            continue
        compacted = _compact_value(profile[key])
        if compacted not in ({}, [], "", None):
            out[key] = compacted

    for key in scalar_keys:
        if key not in profile:
            continue
        value = str(profile.get(key) or "").strip()
        if not value:
            continue
        if relevance_tokens and _profile_relevance_score(value, relevance_tokens) <= 0:
            continue
        out[key] = _compact_value(value)

    for key in relevant_item_keys:
        if key not in profile:
            continue
        ranked = _rank_relevant_profile_items(
            profile[key],
            relevance_tokens=relevance_tokens,
            max_items=3,
        )
        if ranked:
            out[key] = ranked

    for key in passthrough_keys:
        if key not in profile:
            continue
        if key == "business_areas":
            compacted = _relevant_business_areas_for_prompt(
                profile[key],
                relevance_tokens=relevance_tokens,
            )
        elif key == "capability_evolution":
            compacted = _compact_capability_evolution_for_prompt(
                profile[key],
                relevance_tokens=relevance_tokens,
            )
        else:
            compacted = _compact_value(profile[key])
        if compacted not in ({}, [], "", None):
            out[key] = compacted
    return out


def _relevant_business_areas_for_prompt(
    value: Any,
    *,
    relevance_tokens: set[str],
) -> list[Any]:
    if not isinstance(value, list):
        return []
    if not relevance_tokens:
        return [_compact_profile_item(item, include_evidence=False) for item in value[:5]]

    ranked = _rank_relevant_profile_items(
        value,
        relevance_tokens=relevance_tokens,
        max_items=5,
    )
    return ranked


def _rank_relevant_profile_items(
    value: Any,
    *,
    relevance_tokens: set[str],
    max_items: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    if not relevance_tokens:
        return [
            compacted
            for item in value[:max_items]
            if (compacted := _compact_profile_item(item)) not in ({}, [], "", None)
        ]

    scored: list[tuple[int, int, Any]] = []
    for index, item in enumerate(value):
        score = _profile_relevance_score(item, relevance_tokens)
        if score > 0:
            scored.append((score, -index, item))
    scored.sort(reverse=True)
    return [
        compacted
        for _, _, item in scored[:max_items]
        if (compacted := _compact_profile_item(item, include_evidence=False))
        not in ({}, [], "", None)
    ]


def _profile_relevance_score(value: Any, relevance_tokens: set[str]) -> int:
    if not relevance_tokens:
        return 0
    item_text = _json_dumps(value) if isinstance(value, dict | list) else str(value)
    item_tokens = _content_tokens(item_text)
    score = 0
    for issue_token in relevance_tokens:
        for item_token in item_tokens:
            if _tokens_semantically_close(issue_token, item_token):
                score += 1
                break
    return score


def _tokens_semantically_close(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) < 3 or len(right) < 3:
        return False
    return left in right or right in left


def _compact_profile_item(value: Any, *, include_evidence: bool = True) -> dict[str, Any] | Any:
    if not isinstance(value, dict):
        return _compact_value(value)
    preferred_keys = (
        "name",
        "business_area",
        "summary",
        "recent_direction",
        "core_capabilities",
        "capabilities",
        "change_type",
        "period",
        "confidence",
        "source_ref",
        "source_refs",
    )
    out: dict[str, Any] = {}
    for key in preferred_keys:
        if key not in value:
            continue
        compacted = _compact_value(value[key])
        if compacted not in ({}, [], "", None):
            out[key] = compacted
    if include_evidence:
        evidence = value.get("evidence_text") or value.get("evidence_texts")
        compacted_evidence = _compact_evidence_value(evidence)
        if compacted_evidence not in ({}, [], "", None):
            out["evidence_hint"] = compacted_evidence
    return out


def _compact_evidence_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:260]
    if isinstance(value, list):
        return [_compact_evidence_value(item) for item in value[:2]]
    if isinstance(value, dict):
        text = str(value.get("text") or value.get("evidence_text") or "").strip()
        return text[:260] if text else _compact_profile_item(value, include_evidence=False)
    return _compact_value(value)


def _compact_capability_evolution_for_prompt(
    value: Any,
    *,
    relevance_tokens: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("period", "watch_points"):
        compacted = _compact_value(value.get(key))
        if compacted not in ({}, [], "", None):
            out[key] = compacted
    changes = _rank_relevant_profile_items(
        value.get("changes"),
        relevance_tokens=relevance_tokens,
        max_items=3,
    )
    if relevance_tokens and not changes:
        return {}
    if changes:
        out["changes"] = changes
    elif not relevance_tokens:
        out["changes"] = _compact_value(value.get("changes") or [])
    overall_change = str(value.get("overall_change") or "").strip()
    if overall_change and (
        not relevance_tokens or _profile_relevance_score(overall_change, relevance_tokens) > 0
    ):
        out["overall_change"] = _compact_value(overall_change)
    return out


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
