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
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
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
_PROMPT_VERSION = "strategic-insight-v1.61-llm-structured-reasoning"
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
    r"격차[가를은\s]*(확대|벌어|커|발생|나타)",
    r"리더십\s*확보",
    r"매출\s*기여",
    r"시장\s*점유율\s*감소",
    r"점유율[이을가\s]*(감소|하락|축소)",
)
_RELATIONSHIP_PATTERN = re.compile(
    r"협업|협력|파트너십|제휴|MOU|얼라이언스|컨소시엄|"
    r"공동\s*(추진|개발|연구|사업|운영|구축|참여|투자|검증)",
    re.IGNORECASE,
)
_RELATIONSHIP_ACTIVITY_TYPES = {"partnership", "collaboration", "alliance", "joint", "mou"}
_UNCERTAINTY_PATTERN = re.compile(r"검토|가능성|구상|계획|예정|모색|논의|추진\s*(중|예정|계획)")
_SUPPLIER_CAPABILITY_PATTERN = re.compile(
    r"(공급|납품)\s*역량|공급\s*계약.{0,30}(제공|수행)\s*역량"
)
_SUPPLY_CONTRACT_PATTERN = re.compile(r"공급\s*계약|공급계약|납품|구매|조달|계약\s*체결|계약")
_NUMERIC_TOKEN_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|usd|krw)?",
    re.IGNORECASE,
)

UNCERTAIN_ACTIVITY_MARKERS = (
    "검토",
    "가능성",
    "구상",
    "계획",
    "예정",
    "모색",
    "논의",
)
GENERIC_BUSINESS_CATEGORIES: dict[str, tuple[str, ...]] = {
    "generic_entity": ("기업", "회사", "고객"),
    "generic_event": ("이번", "해당", "관련"),
    "generic_object": ("사업", "프로젝트", "분야"),
    "generic_relation": ("기반", "통해", "중심"),
}
_INTERNAL_CHECKPOINT_GROUPS: dict[str, str] = {
    "scope": r"범위|대상\s*업무|대상\s*시스템|적용\s*범위|고객군|유사\s*사업",
    "ownership": r"책임|역할\s*분담|운영\s*구조|수행\s*주체|관리\s*주체",
    "validation": r"검증|성능|용량|평가\s*기준|확인\s*기준|전환\s*조건|안착\s*조건",
    "risk": r"리스크|위험|장애\s*대응|보안|권한|운영\s*조건|처리\s*기준",
    "strategy": r"사업\s*기회|역량\s*공백|영업\s*전략|보완|모니터링|후속\s*확인",
}
_SKAX_ACTION_VERB_GROUPS: dict[str, str] = {
    "diagnose": r"점검|확인|비교|분석|검토",
    "define": r"정의|기준화|명시|구체화|항목화",
    "design": r"구조화|구성|설계|재구성|분리|구분",
    "operate": r"관리|추적|측정|반영|모니터링",
}
_DOMAIN_ALIASES: dict[str, set[str]] = {
    "금융": {"금융", "금융권", "은행", "보험", "증권", "카드", "코어뱅킹"},
    "제조": {"제조", "공장", "생산", "스마트팩토리", "팩토리"},
    "공공": {"공공", "정부", "지자체", "공공기관", "국가"},
    "물류": {"물류", "배송", "창고", "풀필먼트"},
    "유통": {"유통", "리테일", "커머스", "이커머스"},
    "통신": {"통신", "텔코", "네트워크", "5G"},
    "의료": {"의료", "병원", "헬스케어", "바이오"},
    "헬스케어": {"헬스케어", "의료", "병원", "건강관리"},
    "인프라": {"인프라", "데이터센터", "컴퓨팅센터", "GPU", "서버", "클러스터", "스토리지"},
}
OVERCLAIM_PATTERNS: dict[str, tuple[str, ...]] = {
    "counterparty": (
        r"신규\s*사업",
        r"사업\s*(영역|범위)?\s*(확장|확대)",
        r"영역\s*(확장|확대)",
        r"입지\s*강화",
        r"역량\s*강화",
        r"경쟁력\s*강화",
        r"레퍼런스\s*확보",
    ),
    "new_signal": (
        r"확정\s*(성과|사업|진출|확장)",
        r"입증",
        r"역량\s*강화",
        r"성과[가를은\s]*(입증|확대|개선|창출)",
        r"경쟁력\s*강화",
        r"입지\s*강화",
        r"사업\s*(영역|범위)?\s*(확장|확대)",
    ),
}
ACTION_REPAIR_SYSTEM_PROMPT = """\
당신은 SK AX 대응방향만 다시 쓰는 repair agent입니다.
새 사실을 만들지 말고 recommended_actions 만 JSON 으로 출력합니다.
recommended_actions 전체 묶음은 피어 신호, 유사 고객군/유사 사업,
피어사 사업군과 SK AX 사업군의 겹침/차이, 내부 점검 기준,
보완할 사업/역량/운영/영업 전략, 후속 모니터링 항목을 보여야 합니다.
각 항목은 명사구가 아니라 "SK AX는 ..." 흐름의 완성된 실행 판단 문장이어야 합니다.
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
2. 현재 사건의 대상 사업/시스템/서비스/인프라에 맞춰 SK AX 내부 점검 기준을 직접 씁니다.
3. skax_implication_mode 가 generic_monitoring_action 이면 특정 기술명/사업영역명을
   새로 쓰지 않습니다.
4. 유사 고객군/유사 사업 관점은 유지하되, 외부 고객에게 보여줄 문장이 아니라
   SK AX 내부 전략 점검 문장으로 씁니다.
5. recommended_actions 전체 묶음은 피어 신호 → 유사 고객군/유사 사업 →
   피어사 사업군/역량과 SK AX 사업군/역량의 겹침/차이 → 내부 비교 기준 →
   보완할 사업/역량/운영/영업 전략 → 후속 모니터링할 경쟁사 움직임이
   드러나야 합니다. 단, 각 action 문장에는 현재 사건 신호, SK AX 내부 점검 기준,
   보완 방향, 후속 모니터링 중 최소 2개 이상을 자연스럽게 포함합니다.
6. 넓은 실행 장면명으로 끝내지 말고, SK AX가 점검할 범위, 책임,
   일정 조건, 검증 기준, 리스크, 운영 조건, 후속 모니터링 항목을 구조화합니다.
7. "역량 강화", "경쟁력 강화", "기회 확대", "전략적 방향성 제시" 같은 명사구로
   끝내지 않습니다. 무엇을 기준으로 비교하고 무엇을 내부적으로 바꿀지 동사까지 씁니다.
8. 2~3개 항목을 출력하되, 각 항목은 하나의 완성된 문장입니다.

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
1. schema 는 유지하고 문장만 고칩니다.
2. analysis 는 현재 사실 → 대상 업무/시스템의 전환·검증·운영 성격 → 시장 신호를 보여야 합니다.
3. peer_implication 은 현재 사실 → 피어 역할 → 피어 프로필 접점 → 사업적 의미를 보여야 합니다.
4. skax_implication 은 피어 신호 → 유사 고객군/유사 사업 →
   피어사 사업군/역량과 SK AX 사업군/역량의 겹침/차이 →
   SK AX의 대응 가능 범위와 역량 공백 →
   보완할 사업/역량/운영/영업 전략 →
   후속 모니터링 항목을 보여야 합니다.
5. 타깃 피어가 계약 상대방/고객 슬롯이면 공급자처럼 쓰지 않습니다.
6. 프로필 또는 recent context 접점이 없으면 프로필 기반 결론처럼 쓰지 말고
   confidence/evidence_label 을 낮춥니다.
7. 입력에 없는 수치, 고객명, 제품명, 회사명은 추가하지 않습니다.

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
4. capability_change 는 공급 역량 강화가 아니라 확인된 사업 범위/대상 시스템/프로필 접점으로 씁니다.
   ProfileContext 에 관련 사업영역/역량이 없으면 모델 일반 지식으로 채우지 말고
   사건 기반 해석으로 낮춥니다.
5. recommended_actions 는 유사 고객군/유사 사업에서 보는 SK AX 내부 전략 점검입니다.
   피어사 사업군/역량과 SK AX 사업군/역량의 겹침과 차이, 대응 가능 범위,
   역량 공백, 운영 구조, 영업 전략, 후속 모니터링 항목을 함께 씁니다.
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
   - recommended_actions: 피어 신호 → 유사 고객군/유사 사업 → 피어사 사업군/역량과
     SK AX 사업군/역량의 겹침/차이 → 내부 점검 기준 → 보완할 사업/역량/운영/영업 전략 →
     후속 모니터링

## 출력
StrategicInsightAgent 최종 schema 그대로 출력합니다.
"""


SYSTEM_PROMPT = """\
당신은 임원 보고용 전략 인사이트를 작성하는 Agent입니다.

데이터 역할:
- IntegratedIssue와 StrategicEvidencePack은 현재 사건의 유일한 사실 근거입니다.
- ProfileContext는 피어사와 SK AX의 기존 사업영역/역량 배경입니다.
- AnalysisContext는 현재 사건과 직접 연결될 때만 보조 맥락으로 사용합니다.
- Machine linkage hints와 Response artifact guidance는 참고용 안전 힌트이며 최종 판단이 아닙니다.

작업 원칙:
1. 먼저 현재 사건의 확정 사실, 관계 수준, 피어 역할을 구조화합니다.
2. 그 다음 현재 사건과 피어 프로필의 사업영역/역량 접점을 판단합니다.
3. 접점이 충분하면 프로필 기반 시사점, 약하면 사건 기반 관찰 신호로 낮춥니다.
4. SK AX 대응은 피어 신호와 SK AX 프로필 접점이 있을 때만 구체화합니다.
5. 대응방향은 현재 상태 → 왜 바꿔야 하는가 → 무엇을 바꿔야 하는가 →
   SK AX가 어떤 전략 판단을 해야 하는가 흐름으로 씁니다.
6. 입력에 없는 사실, 회사 일반 지식, 근거 없는 기술명/수치/제품명은 만들지 않습니다.
7. JSON 외 텍스트를 출력하지 마세요.
"""


USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## StrategicEvidencePack
{strategic_evidence_json}

## classification
{classification_json}

## input_bundle metadata
{bundle_json}

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

## SK AX business_line_mapping 후보
{business_lines_json}

## 역할 해석 모드
{role_mode_instructions}

## 생성 순서
1. 현재 사건 이해:
   StrategicEvidencePack의 fact_basis/evidence_texts를 읽고 확정 사실, 수치, 날짜,
   피어 역할, 대상 사업/시스템/서비스/인프라, 고객군/시장 범위를 구조화합니다.
   역할이 불명확하면 unclear 또는 contract_counterparty처럼 보수적으로 둡니다.
2. 피어 프로필 연결 판단:
   ProfileContext.peer_profiles에서 현재 사건과 직접 연결되는 사업영역/역량만 찾습니다.
   왜 연결되는지 profile_source_ref 또는 프로필 필드 근거와 함께 설명합니다.
   연결 근거가 없으면 프로필 기반 결론을 만들지 않습니다.
3. 시사점 작성:
   현재 사실을 반복하지 말고, 피어 역할과 프로필 접점을 통해 사업적 의미를 씁니다.
   입지 강화/영역 확장/경쟁력 강화/성과 입증은 근거가 충분할 때만 사용합니다.
   근거가 부족하면 관찰 신호, 비교 기준 구체화, 후속 확인 필요로 낮춥니다.
4. SK AX 대응 연결 판단:
   SK AX 프로필과 현재 피어 신호가 어떤 사업/역량/운영/영업 항목에서 비교되는지 구조화합니다.
   연결 근거가 부족하면 특정 기술명/성공 사례를 만들지 않고 generic monitoring 수준으로 둡니다.
5. SK AX 대응방향 작성:
   유사 고객군/유사 사업 관점은 유지하되, 외부 고객 제안 문장이 아니라
   SK AX 내부 전략 판단 문장으로 씁니다.
   피어 신호가 속한 사업/고객군에서 피어사 사업군·역량과 SK AX 사업군·역량이
   어디서 겹치고 어디서 달라지는지, SK AX의 대응 가능 범위와 역량 공백,
   운영·영업 전략, 후속 모니터링 항목을 현재 사건에 맞게 씁니다.
   넓은 실행 장면명을 기본값처럼 반복하지 말고, 내부에서 확인할 기준을 씁니다.
6. 최종 검증:
   모든 강한 주장에 fact_id 또는 profile_source_ref가 있는지 확인합니다.
   근거가 없으면 문장 강도를 낮추고, 같은 의미가 여러 필드에 반복되면 압축합니다.

## 작성 기준
- 시사점은 현재 사실 → 피어 역할 → 연결된 프로필 근거 → 사업적 의미 순서가 보여야 합니다.
- 대응방향은 피어/시장 신호 → 유사 고객군/유사 사업 → 피어사 사업군/역량과
  SK AX 사업군/역량의 겹침/차이 → 대응 가능 범위/역량 공백 → 보완할
  사업/역량/운영/영업 전략 → 후속 모니터링 순서가 보여야 합니다.
- Machine linkage hints의 business_novelty_status가 not_new_business_counterparty_role이면
  신규 사업/사업 확장/입지 강화로 쓰지 않습니다.
- Machine linkage hints의 business_novelty_status가 new_or_untracked_business_signal이면
  확정 성과가 아니라 프로필 기준 미포착 관찰 신호로 씁니다.
- 수치/날짜/회사명/사업명/fact_id는 입력에 있는 것만 사용합니다.

## 출력
아래 JSON schema 를 그대로 지켜 출력합니다. 설명 텍스트나 markdown 은 출력하지 마세요.
모든 자연어 문자열 값은 한국어로 작성하세요.
company_id, fact_id, enum 값, business_line_mapping 후보명처럼 입력에서 정해진 식별자만
원문 값을 유지합니다. analysis_summary, strategic_meaning, market_signal, impact_reason,
peer_meaning, capability_change, why_important, potential_impact, recommended_actions,
follow_up_questions, watch_points 는 반드시 한국어 문장이어야 합니다.
{{
  "issue_understanding": {{
    "confirmed_facts": ["근거 기반 확정 사실"],
    "main_actor": "string",
    "peer_role_in_issue": "enum",
    "role_confidence": 0.0,
    "activity_nature": "enum",
    "target_business_or_system": ["string"],
    "customer_or_market_scope": ["string"],
    "confirmed_numbers_or_dates": ["string"],
    "uncertain_points": ["string"],
    "evidence_ids": ["입력에 존재하는 fact_id"]
  }},
  "profile_linkage": {{
    "peer_company": "string",
    "profile_evidence_available": true,
    "matched_profile_areas": [
      {{
        "profile_area_name": "string",
        "profile_capability": "string",
        "why_relevant_to_issue": "string",
        "profile_source_ref": "string"
      }}
    ],
    "linkage_level": "high|medium|low|none",
    "business_novelty_status": "enum",
    "allowed_interpretation_strength": "enum",
    "reason": "string"
  }},
  "skax_response_linkage": {{
    "skax_profile_evidence_available": true,
    "matched_skax_areas": [
      {{
        "profile_area_name": "string",
        "why_relevant_to_issue": "string",
        "profile_source_ref": "string"
      }}
    ],
    "response_mode": "profile_based_action|cautious_action|generic_monitoring_action",
    "response_focus": ["string"],
    "internal_checkpoints": ["SK AX가 내부적으로 점검해야 할 기준"],
    "recommended_focus": ["보완해야 할 사업/역량/운영/영업 전략"],
    "monitoring_points": ["후속 확인할 피어사 신호"],
    "reason": "string"
  }},
  "claim_strength": "strong|moderate|cautious",
  "grounding_summary": {{
    "used_fact_ids": ["입력에 존재하는 fact_id"],
    "used_profile_refs": ["string"],
    "ungrounded_claims_removed": ["string"]
  }},
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
5. recommended_actions는 유사 고객군/유사 사업 관점을 유지하되,
   외부 고객 제안 문장이 아니라 SK AX 내부 점검/보완 문장이어야 합니다.
   피어 신호, 피어사와 SK AX 사업군/역량의 겹침/차이, 대응 가능 범위,
   역량 공백, 운영·영업 전략, 후속 모니터링 항목이 보여야 합니다.
   명사구로 끝나는 대응방향은 실패입니다. 현재 사건 신호와 내부 점검 기준,
   SK AX가 보완하거나 모니터링할 방식을 포함한 완성 문장으로 고칩니다.
6. 같은 의미가 여러 필드에 반복되면 시사점/대응방향 두 묶음으로 압축합니다.
7. 복구할 수 없으면 confidence/evidence_label을 낮추거나 invalid로 둡니다.

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
4. recommended_actions는 현재 사건의 유사 고객군/유사 사업에서 피어사 사업군과
   SK AX 사업군이 겹치거나 달라지는 지점, 대응 가능 범위, 역량 공백, 운영·영업 전략,
   후속 모니터링 등 SK AX가 내부적으로 점검할 기준을 포함합니다.
   각 항목은 현재 사건의 대상 사업/시스템/서비스/인프라 중 하나와 연결되고,
   범위·책임·일정·검증·리스크·운영 조건·모니터링 중 하나 이상의 기준을 포함해야 합니다.
   "강화", "확대", "방향성"만 남는 명사구는 완성 문장으로 바꿉니다.
5. 필드별 제목을 여러 개 붙인 듯한 반복 문장은 줄이고, 시사점/대응방향 두 묶음만 남깁니다.
6. 새 사실을 만들지 말고 실패 필드만 최소 수정합니다.

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
        profile_linkage_evaluation = _build_profile_linkage_evaluation(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_context=profile_dict,
            relevance_hint_text=profile_relevance_text,
        )
        action_artifact_plan = _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        context_for_model = _analysis_context_for_model(
            context_dict,
            integrated_issue=integrated_issue,
            relevance_hint_text=profile_relevance_text,
            include_financial_context=include_financial_profile_context,
        )

        prompt = USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            strategic_evidence_json=_json_dumps(
                _strategic_evidence_pack_for_prompt(
                    integrated_issue=integrated_issue,
                    bundle=bundle_dict,
                )
            ),
            classification_json=_json_dumps(_classification_for_prompt(classification)),
            bundle_json=_json_dumps(_bundle_for_prompt(bundle_dict, cluster_metadata)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_dict,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            context_json=_json_dumps(_analysis_context_for_prompt(context_for_model)),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
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
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
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
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            reviewed = _with_fallback_linkage_payloads(
                reviewed,
                profile_linkage_evaluation=profile_linkage_evaluation,
                integrated_issue=integrated_issue,
                action_artifact_plan=action_artifact_plan,
            )
            if "quality_gate_failed" in _json_dumps(reviewed):
                fallback = _two_section_fact_based_fallback(
                    reviewed,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                    model=self.model,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                fallback_violations = _quality_gate_violations(
                    fallback,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                if not fallback_violations:
                    return _attach_sentence_grounding(
                        fallback,
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
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
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
        from langchain_openai import ChatOpenAI  # lazy: transformers 체인 회피

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
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification={},
                profile_context=profile_context,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification={},
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        result = _with_fallback_linkage_payloads(
            result,
            profile_linkage_evaluation=profile_linkage_evaluation,
            integrated_issue=integrated_issue,
            action_artifact_plan=action_artifact_plan,
        )
        violations = _quality_gate_violations(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
            action_artifact_plan=action_artifact_plan,
        )
        if not violations:
            return _attach_sentence_grounding(
                _restore_valid_flags_if_structurally_safe(result),
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            )
        fallback = _two_section_fact_based_fallback(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            model=self.model,
            profile_linkage_evaluation=profile_linkage_evaluation,
            action_artifact_plan=action_artifact_plan,
        )
        fallback_violations = _quality_gate_violations(
            fallback,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
            action_artifact_plan=action_artifact_plan,
        )
        if not fallback_violations:
            return _attach_sentence_grounding(
                fallback,
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
            profile_linkage_evaluation=profile_linkage_evaluation,
            action_artifact_plan=action_artifact_plan,
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
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                relevance_hint_text=profile_relevance_text,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        prompt = REVIEW_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            classification_json=_json_dumps(_classification_for_prompt(classification)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            context_json=_json_dumps(_analysis_context_for_prompt(analysis_context)),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
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
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
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
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
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
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
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
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = result
        current_violations = violations
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                relevance_hint_text=profile_relevance_text,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        try:
            for attempt in range(3):
                prompt = REPAIR_USER_PROMPT_TEMPLATE.format(
                    integrated_issue_json=_json_dumps(
                        _integrated_issue_for_prompt(integrated_issue)
                    ),
                    profile_json=_json_dumps(
                        _profile_for_prompt(
                            profile_context,
                            integrated_issue=integrated_issue,
                            relevance_hint_text=profile_relevance_text,
                            include_financial_context=include_financial_profile_context,
                            profile_linkage_evaluation=profile_linkage_evaluation,
                        )
                    ),
                    profile_linkage_json=_json_dumps(profile_linkage_evaluation),
                    action_artifact_plan_json=_json_dumps(action_artifact_plan),
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
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
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
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                current_violations = _quality_gate_violations(
                    current,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
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
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                current_violations = _quality_gate_violations(
                    current,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
                if not current_violations:
                    return current
            fallback = _two_section_fact_based_fallback(
                current,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                model=self.model,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            fallback_violations = _quality_gate_violations(
                fallback,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
            )
            if not fallback_violations:
                return fallback
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
                    action_artifact_plan=action_artifact_plan,
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
                action_artifact_plan=action_artifact_plan,
            )
            if not _string_list(
                ((guarded.get("implication") or {}).get("skax_implication") or {}).get(
                    "recommended_actions"
                ),
                max_items=3,
            ):
                guarded = _two_section_fact_based_fallback(
                    guarded,
                    integrated_issue=integrated_issue,
                    profile_context=profile_context,
                    model=self.model,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                    action_artifact_plan=action_artifact_plan,
                )
            final_remaining = _quality_gate_violations(
                guarded,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
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
                profile_context=profile_context,
            )
            final_remaining = _quality_gate_violations(
                guarded,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                profile_linkage_evaluation=profile_linkage_evaluation,
                action_artifact_plan=action_artifact_plan,
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
        profile_relevance_text: str = "",
        include_financial_profile_context: bool = False,
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                relevance_hint_text=profile_relevance_text,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        prompt = COUNTERPARTY_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
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
        profile_linkage_evaluation: dict[str, Any] | None = None,
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_linkage_evaluation = (
            profile_linkage_evaluation
            or _build_profile_linkage_evaluation(
                integrated_issue=integrated_issue,
                classification=classification,
                profile_context=profile_context,
                relevance_hint_text=profile_relevance_text,
            )
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        prompt = REPORT_COPY_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    include_financial_context=include_financial_profile_context,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
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
        action_artifact_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        implication = out.get("implication") or {}
        skax = implication.get("skax_implication") or {}
        profile_linkage_evaluation = _build_profile_linkage_evaluation(
            integrated_issue=integrated_issue,
            classification={},
            profile_context=profile_context,
            relevance_hint_text=profile_relevance_text,
        )
        action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification={},
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        prompt = ACTION_REPAIR_USER_PROMPT_TEMPLATE.format(
            integrated_issue_json=_json_dumps(_integrated_issue_for_prompt(integrated_issue)),
            profile_json=_json_dumps(
                _profile_for_prompt(
                    profile_context,
                    integrated_issue=integrated_issue,
                    relevance_hint_text=profile_relevance_text,
                    profile_linkage_evaluation=profile_linkage_evaluation,
                )
            ),
            profile_linkage_json=_json_dumps(profile_linkage_evaluation),
            action_artifact_plan_json=_json_dumps(action_artifact_plan),
            business_lines_json=_json_dumps(
                _business_line_candidate_details(
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
            if not _repair_action_violation(
                action,
                label=f"skax_implication.recommended_actions[{index}]",
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                action_artifact_plan=action_artifact_plan,
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
        profile_dict = _profile_to_dict(profile_context)
        profile_relevance_text = _profile_relevance_hint_text(
            integrated_issue=integrated_issue,
            classification=classification,
            bundle=_bundle_to_dict(input_bundle),
        )
        profile_linkage_evaluation = _build_profile_linkage_evaluation(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_context=profile_dict,
            relevance_hint_text=profile_relevance_text,
        )
        action_artifact_plan = _action_artifact_plan_for_prompt(
            integrated_issue=integrated_issue,
            classification=classification,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
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
                profile_context=profile_dict,
                analysis_context={},
                model=self.model,
            ),
        }
        result = _with_fallback_linkage_payloads(
            result,
            profile_linkage_evaluation=profile_linkage_evaluation,
            integrated_issue=integrated_issue,
            action_artifact_plan=action_artifact_plan,
        )
        return _attach_sentence_grounding(
            result,
            integrated_issue=integrated_issue,
            profile_context=profile_dict,
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
        "issue_understanding": _normalize_issue_understanding(
            data.get("issue_understanding"),
            integrated_issue=integrated_issue,
        ),
        "profile_linkage": _normalize_llm_profile_linkage(data.get("profile_linkage")),
        "skax_response_linkage": _normalize_skax_response_linkage(
            data.get("skax_response_linkage")
        ),
        "claim_strength": _choice(
            data.get("claim_strength"),
            {"strong", "moderate", "cautious"},
            "cautious",
        ),
        "grounding_summary": _normalize_grounding_summary(
            data.get("grounding_summary"),
            integrated_issue=integrated_issue,
        ),
        "analysis": analysis,
        "implication": implication,
    }


def _normalize_issue_understanding(
    value: Any,
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    data = _json_dict(value)
    known_fact_ids = _known_fact_ids(integrated_issue)
    return {
        "confirmed_facts": _string_list(data.get("confirmed_facts"), max_items=8),
        "main_actor": str(data.get("main_actor") or integrated_issue.get("main_company") or ""),
        "peer_role_in_issue": _choice(
            data.get("peer_role_in_issue"),
            {
                "provider",
                "builder",
                "operator",
                "selected_party",
                "contract_counterparty",
                "customer_or_buyer",
                "partner",
                "investor",
                "unclear",
            },
            "unclear",
        ),
        "role_confidence": _clamp_float(data.get("role_confidence"), 0.0),
        "activity_nature": _choice(
            data.get("activity_nature"),
            {
                "selection",
                "contract",
                "launch",
                "investment",
                "partnership",
                "operation",
                "system_transition",
                "infrastructure_build",
                "business_update",
                "unclear",
            },
            "unclear",
        ),
        "target_business_or_system": _string_list(
            data.get("target_business_or_system"),
            max_items=8,
        ),
        "customer_or_market_scope": _string_list(
            data.get("customer_or_market_scope"),
            max_items=8,
        ),
        "confirmed_numbers_or_dates": _string_list(
            data.get("confirmed_numbers_or_dates"),
            max_items=8,
        ),
        "uncertain_points": _string_list(data.get("uncertain_points"), max_items=8),
        "evidence_ids": [
            fact_id
            for fact_id in _string_list(data.get("evidence_ids"), max_items=12)
            if fact_id in known_fact_ids
        ],
    }


def _normalize_llm_profile_linkage(value: Any) -> dict[str, Any]:
    data = _json_dict(value)
    return {
        "peer_company": str(data.get("peer_company") or ""),
        "profile_evidence_available": bool(data.get("profile_evidence_available")),
        "matched_profile_areas": [
            {
                "business_line": str(item.get("business_line") or "").strip(),
                "business_area": str(item.get("business_area") or "").strip(),
                "profile_area_name": str(item.get("profile_area_name") or "").strip(),
                "profile_capability": str(item.get("profile_capability") or "").strip(),
                "matched_capabilities": _string_list(
                    item.get("matched_capabilities")
                    or item.get("core_capabilities")
                    or item.get("capabilities"),
                    max_items=8,
                ),
                "matched_products_or_services": _string_list(
                    item.get("matched_products_or_services")
                    or item.get("products_or_services")
                    or item.get("key_products_services"),
                    max_items=8,
                ),
                "matched_issue_terms": _string_list(
                    item.get("matched_issue_terms") or item.get("matched_terms"),
                    max_items=12,
                ),
                "evidence_text": str(item.get("evidence_text") or "").strip(),
                "why_relevant_to_issue": str(item.get("why_relevant_to_issue") or "").strip(),
                "profile_source_ref": str(
                    item.get("profile_source_ref") or item.get("source_ref") or ""
                ).strip(),
                "specificity_level": _choice(
                    item.get("specificity_level"),
                    {
                        "product_or_service",
                        "core_capability",
                        "business_area",
                        "business_line",
                        "profile_context",
                    },
                    "profile_context",
                ),
            }
            for item in _jsonish_list(data.get("matched_profile_areas"))[:5]
            if isinstance(item, dict)
        ],
        "linkage_level": _choice(
            data.get("linkage_level"),
            {"high", "medium", "low", "none"},
            "none",
        ),
        "business_novelty_status": _choice(
            data.get("business_novelty_status"),
            {
                "existing_profile_business_linked",
                "weak_profile_linkage",
                "new_or_untracked_business_signal",
                "role_sensitive_untracked_signal",
                "profile_insufficient_cannot_judge_novelty",
                "uncertain_not_enough_to_call_new_business",
                "not_new_business_counterparty_role",
            },
            "profile_insufficient_cannot_judge_novelty",
        ),
        "allowed_interpretation_strength": _choice(
            data.get("allowed_interpretation_strength"),
            {
                "profile_based",
                "cautious_profile_based",
                "event_based",
                "observation_only",
            },
            "observation_only",
        ),
        "reason": str(data.get("reason") or "").strip(),
    }


def _normalize_skax_response_linkage(value: Any) -> dict[str, Any]:
    data = _json_dict(value)
    return {
        "skax_profile_evidence_available": bool(data.get("skax_profile_evidence_available")),
        "matched_skax_areas": [
            {
                "business_line": str(item.get("business_line") or "").strip(),
                "business_area": str(item.get("business_area") or "").strip(),
                "profile_area_name": str(item.get("profile_area_name") or "").strip(),
                "matched_capabilities": _string_list(
                    item.get("matched_capabilities")
                    or item.get("core_capabilities")
                    or item.get("capabilities"),
                    max_items=8,
                ),
                "matched_products_or_services": _string_list(
                    item.get("matched_products_or_services")
                    or item.get("products_or_services")
                    or item.get("key_products_services"),
                    max_items=8,
                ),
                "matched_issue_terms": _string_list(
                    item.get("matched_issue_terms") or item.get("matched_terms"),
                    max_items=12,
                ),
                "evidence_text": str(item.get("evidence_text") or "").strip(),
                "why_relevant_to_issue": str(item.get("why_relevant_to_issue") or "").strip(),
                "profile_source_ref": str(
                    item.get("profile_source_ref") or item.get("source_ref") or ""
                ).strip(),
                "specificity_level": _choice(
                    item.get("specificity_level"),
                    {
                        "product_or_service",
                        "core_capability",
                        "business_area",
                        "business_line",
                        "profile_context",
                    },
                    "profile_context",
                ),
            }
            for item in _jsonish_list(data.get("matched_skax_areas"))[:5]
            if isinstance(item, dict)
        ],
        "response_mode": _choice(
            data.get("response_mode"),
            {"profile_based_action", "cautious_action", "generic_monitoring_action"},
            "generic_monitoring_action",
        ),
        "response_focus": _string_list(data.get("response_focus"), max_items=8),
        "internal_checkpoints": _string_list(data.get("internal_checkpoints"), max_items=8),
        "recommended_focus": _string_list(data.get("recommended_focus"), max_items=8),
        "monitoring_points": _string_list(data.get("monitoring_points"), max_items=8),
        "reason": str(data.get("reason") or "").strip(),
    }


def _normalize_grounding_summary(
    value: Any,
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    data = _json_dict(value)
    known_fact_ids = _known_fact_ids(integrated_issue)
    return {
        "used_fact_ids": [
            fact_id
            for fact_id in _string_list(data.get("used_fact_ids"), max_items=12)
            if fact_id in known_fact_ids
        ],
        "used_profile_refs": _string_list(data.get("used_profile_refs"), max_items=12),
        "ungrounded_claims_removed": _string_list(
            data.get("ungrounded_claims_removed"),
            max_items=12,
        ),
    }


def _ensure_reasoning_debug_fields(
    result: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    """Keep reasoning/debug fields stable on LLM, repair, invalid, and fallback paths."""
    out = dict(result or {})
    out["issue_understanding"] = _normalize_issue_understanding(
        out.get("issue_understanding"),
        integrated_issue=integrated_issue,
    )
    out["profile_linkage"] = _normalize_llm_profile_linkage(out.get("profile_linkage"))
    out["skax_response_linkage"] = _normalize_skax_response_linkage(
        out.get("skax_response_linkage")
    )
    out["claim_strength"] = _choice(
        out.get("claim_strength"),
        {"strong", "moderate", "cautious"},
        "cautious",
    )
    out["grounding_summary"] = _normalize_grounding_summary(
        out.get("grounding_summary"),
        integrated_issue=integrated_issue,
    )
    return out


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
    result = {
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
    return _ensure_reasoning_debug_fields(result, integrated_issue=integrated_issue)


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


def _strategic_evidence_pack_for_prompt(
    *,
    integrated_issue: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Compact article-derived evidence for strategic implication generation.

    IntegratedIssue remains the only fact source. This pack simply separates the
    article evidence that the integration step already selected from display
    summaries, so the strategic agent does not infer from card copy alone.
    """

    fact_basis = []
    for item in _jsonish_list(integrated_issue.get("fact_basis"))[:12]:
        if not isinstance(item, dict):
            continue
        evidence_texts = [
            str(text or "").strip()
            for text in _jsonish_list(item.get("evidence_texts"))[:3]
            if str(text or "").strip()
        ]
        evidence_text = str(item.get("evidence_text") or "").strip()
        if evidence_text and evidence_text not in evidence_texts:
            evidence_texts.append(evidence_text)
        fact_text = str(item.get("fact") or "").strip()
        fact_basis.append(
            {
                "fact": fact_text,
                "fact_ids": _string_list(item.get("fact_ids"), max_items=5),
                "source_article_ids": _int_list(item.get("source_article_ids"))[:5],
                "evidence_type": str(item.get("evidence_type") or "").strip(),
                "evidence_texts": evidence_texts[:3],
            }
        )

    consolidated_facts = []
    for item in _jsonish_list(integrated_issue.get("consolidated_facts"))[:12]:
        fact_text = _fact_like_text(item)
        if not fact_text:
            continue
        row: dict[str, Any] = {"fact": fact_text}
        if isinstance(item, dict):
            row["fact_id"] = str(item.get("fact_id") or "").strip()
            row["source_article_ids"] = _int_list(item.get("source_article_ids"))[:5]
        consolidated_facts.append(row)

    representative_sources = []
    for item in _jsonish_list(integrated_issue.get("representative_sources"))[:10]:
        if not isinstance(item, dict):
            continue
        representative_sources.append(
            {
                "article_id": item.get("article_id") or item.get("id"),
                "title": str(item.get("title") or "").strip(),
                "publisher": str(item.get("publisher") or "").strip(),
                "source_name": str(item.get("source_name") or "").strip(),
                "published_at": str(item.get("published_at") or "").strip(),
            }
        )

    bundle_evidence_snippets = []
    for item in _jsonish_list(bundle.get("evidence_snippets"))[:12]:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("evidence_text") or "").strip()
            if not text:
                continue
            bundle_evidence_snippets.append(
                {
                    "text": text,
                    "source_article_ids": _int_list(item.get("source_article_ids"))[:5],
                    "fact_ids": _string_list(item.get("fact_ids"), max_items=5),
                }
            )
        else:
            text = str(item or "").strip()
            if text:
                bundle_evidence_snippets.append({"text": text})

    return {
        "purpose": (
            "Use this article-derived pack before display/card summaries when deriving "
            "strategic implications."
        ),
        "current_event": {
            "headline": integrated_issue.get("headline", ""),
            "main_event": integrated_issue.get("main_event", ""),
            "main_issue": integrated_issue.get("main_issue", ""),
            "one_line_summary": integrated_issue.get("one_line_summary", ""),
            "integrated_text": integrated_issue.get("integrated_text", ""),
        },
        "fact_basis": fact_basis,
        "consolidated_facts": consolidated_facts,
        "representative_sources": representative_sources,
        "bundle_evidence_snippets": bundle_evidence_snippets,
        "cluster_fact_intelligence": _cluster_fact_intelligence_for_prompt(
            integrated_issue.get("cluster_fact_intelligence") or {}
        ),
        "missing_or_uncertain_points": integrated_issue.get("missing_or_uncertain_points", []),
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
            "- SK AX 대응방향은 유사 고객군/유사 사업 관점을 유지하되, "
            "외부 고객 제안 문장이 아니라 SK AX 내부 전략 점검으로 쓰세요.",
            "- SK AX 대응방향은 '피어 신호 → 유사 고객군/유사 사업 → 피어사 사업군/역량과 "
            "SK AX 사업군/역량의 겹침/차이 → 대응 가능 범위와 역량 공백 → "
            "보완할 사업/역량/운영/영업 전략 → 후속 모니터링' 순서로 2~3문장 작성하세요.",
            "- 대응방향은 넓은 실행 장면명이 아니라 적용 범위, 역할 분담, 운영 책임, "
            "일정 조건, 장애 대응 기준, 보안·권한 기준, 전환 리스크, 성능/용량 검증 기준, "
            "후속 모니터링 항목처럼 내부적으로 확인할 기준을 중심으로 쓰세요.",
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
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    skax = profile.get("skax_profile") or {}
    peer_profiles = profile.get("peer_profiles") or {}
    relevance_tokens = _issue_relevance_tokens(
        integrated_issue or {},
        extra_text=relevance_hint_text,
    )
    out = {
        "skax_profile": _shrink_profile(
            skax,
            relevance_tokens=relevance_tokens,
            profile_linkage=_profile_linkage_for_company(
                profile_linkage_evaluation,
                company_id="sk_ax",
                scope="skax",
            ),
        ),
        "peer_profiles": {
            str(peer_id): _shrink_profile(
                payload,
                relevance_tokens=relevance_tokens,
                profile_linkage=_profile_linkage_for_company(
                    profile_linkage_evaluation,
                    company_id=str(peer_id),
                    scope="peer",
                ),
            )
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


def _profile_linkage_for_company(
    evaluation: dict[str, Any] | None,
    *,
    company_id: str,
    scope: str,
) -> dict[str, Any]:
    if not isinstance(evaluation, dict):
        return {}
    if scope == "skax":
        linkage = evaluation.get("skax_linkage") or {}
        return linkage if isinstance(linkage, dict) else {}
    for linkage in _jsonish_list(evaluation.get("peer_linkages")):
        if not isinstance(linkage, dict):
            continue
        if str(linkage.get("company_id") or "") == str(company_id):
            return linkage
    return {}


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
                "skax_actions_require_current_signal_plus_skax_profile_plus_internal"
                "_strategy_checkpoint_and_follow_up_monitoring"
            ),
        ],
    }


def _build_profile_linkage_evaluation(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    profile_context: dict[str, Any],
    relevance_hint_text: str = "",
) -> dict[str, Any]:
    issue_context = _profile_linkage_issue_context(
        integrated_issue=integrated_issue,
        classification=classification,
        relevance_hint_text=relevance_hint_text,
    )
    issue_terms = _issue_relevance_tokens(
        integrated_issue,
        extra_text=" ".join(
            item
            for item in (
                relevance_hint_text,
                " ".join(_string_list(issue_context.get("structured_terms"), max_items=40)),
                str(classification.get("sector") or ""),
                " ".join(_string_list(classification.get("sectors"), max_items=8)),
                str(classification.get("event_type") or ""),
            )
            if item
        ),
    )
    role_mode = _peer_role_mode_for_linkage(
        integrated_issue,
        classification=classification,
    )
    peer_profiles = profile_context.get("peer_profiles") or {}
    peer_linkages: list[dict[str, Any]] = []
    for company_id in _companies_from_integrated_issue(integrated_issue):
        profile = peer_profiles.get(company_id) if isinstance(peer_profiles, dict) else {}
        peer_linkages.append(
            _evaluate_single_profile_linkage(
                company_id=company_id,
                profile=profile if isinstance(profile, dict) else {},
                issue_terms=issue_terms,
                issue_context=issue_context,
                integrated_issue=integrated_issue,
                scope="peer",
                role_mode=role_mode,
            )
        )

    skax_linkage = _evaluate_single_profile_linkage(
        company_id="sk_ax",
        profile=profile_context.get("skax_profile") or {},
        issue_terms=issue_terms,
        issue_context=issue_context,
        integrated_issue=integrated_issue,
        scope="skax",
        role_mode="skax_response",
    )
    return {
        "issue_terms": sorted(issue_terms)[:40],
        "peer_role_mode": role_mode,
        "peer_linkages": peer_linkages,
        "skax_linkage": skax_linkage,
        "guidance": _profile_linkage_guidance(peer_linkages, skax_linkage, role_mode),
    }


def _peer_role_mode_for_linkage(
    integrated_issue: dict[str, Any],
    *,
    classification: dict[str, Any] | None = None,
) -> str:
    if _main_company_is_customer_or_buyer(integrated_issue):
        return "counterparty_or_customer"
    structured_role = _role_mode_from_structured_activity(
        _activity_types_from_issue(integrated_issue, classification=classification)
    )
    if structured_role:
        return structured_role
    return _role_mode_from_evidence_fallback(
        _integrated_grounding_text(integrated_issue),
        integrated_issue=integrated_issue,
    )


def _profile_linkage_issue_context(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    relevance_hint_text: str = "",
) -> dict[str, Any]:
    signals = _extract_issue_structured_signals(
        integrated_issue=integrated_issue,
        classification=classification,
    )
    structured_terms: list[str] = []
    for key in (
        "activity_types",
        "products_or_services",
        "customers_or_industries",
        "target_systems",
        "structured_terms",
    ):
        structured_terms.extend(_string_list(signals.get(key), max_items=50))
    structured_terms.extend(_string_list(classification.get("sectors"), max_items=10))
    structured_terms.append(str(classification.get("sector") or ""))
    structured_terms.append(relevance_hint_text)
    evidence_texts: list[str] = []
    evidence_texts.append(_integrated_grounding_text(integrated_issue))
    for fact_id, fact_text in _fact_texts(integrated_issue):
        del fact_id
        evidence_texts.append(fact_text)
    source_noise_terms = _source_noise_terms_from_issue(integrated_issue)
    return {
        "integrated_issue": integrated_issue,
        "classification": classification,
        "activity_types": _string_list(signals.get("activity_types"), max_items=30),
        "products_or_services": _string_list(signals.get("products_or_services"), max_items=30),
        "customers_or_industries": _string_list(
            signals.get("customers_or_industries"),
            max_items=30,
        ),
        "target_systems": _string_list(signals.get("target_systems"), max_items=30),
        "structured_terms": [
            item
            for item in dict.fromkeys(
                re.sub(r"\s+", " ", str(term or "")).strip() for term in structured_terms
            )
            if item
        ][:80],
        "evidence_text": "\n".join(text for text in evidence_texts if text),
        "source_noise_terms": source_noise_terms,
    }


def _source_noise_terms_from_issue(integrated_issue: dict[str, Any]) -> set[str]:
    """Collect publisher/domain tokens that should not drive profile relevance."""
    noise_terms: set[str] = set()

    def add_from_text(value: Any) -> None:
        for term in _raw_normalized_terms(str(value or "")):
            if len(term) > 2:
                noise_terms.add(term.casefold())

    def add_from_url(value: Any) -> None:
        url = str(value or "").strip()
        if not url:
            return
        host_match = re.search(r"^(?:https?://)?([^/?#]+)", url, flags=re.IGNORECASE)
        host = (host_match.group(1) if host_match else url).removeprefix("www.")
        for part in re.split(r"[.\-_/]+", host):
            if len(part) > 2:
                noise_terms.add(_normalize_content_token(part).casefold())

    for source in _jsonish_list(integrated_issue.get("representative_sources")):
        if not isinstance(source, dict):
            continue
        for key in ("source_name", "publisher", "provider", "media", "domain"):
            add_from_text(source.get(key))
        add_from_url(source.get("url"))
    for source_ref in _jsonish_list(integrated_issue.get("source_refs")):
        if isinstance(source_ref, dict):
            for key in ("source_name", "publisher", "provider", "media", "domain"):
                add_from_text(source_ref.get(key))
            add_from_url(source_ref.get("url"))
        else:
            add_from_text(source_ref)
    return {term for term in noise_terms if term}


def _raw_normalized_terms(text: str) -> list[str]:
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·_-]{1,}", text or "")
    terms: list[str] = []
    for token in raw_tokens:
        normalized = _normalize_content_token(token)
        if normalized:
            terms.append(normalized)
    return terms


def _extract_ranked_terms(
    text: str,
    source_role: str,
    issue_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    issue_context = issue_context or {}
    ranked_by_term: dict[str, dict[str, Any]] = {}
    for term in _raw_normalized_terms(text):
        normalized = term.casefold() if term.isascii() else term
        if not normalized:
            continue
        weight = _term_signal_weight(normalized, source_role, issue_context)
        if _looks_like_source_noise(normalized, issue_context):
            term_type = "source_noise"
        elif _looks_like_korean_function_word_or_ending(normalized):
            term_type = "function_word_or_ending"
        elif _appears_in_structured_issue_fields(normalized, issue_context):
            term_type = "structured_issue_term"
        elif _appears_repeatedly_in_evidence(normalized, issue_context):
            term_type = "repeated_evidence_term"
        elif _is_generic_business_term(normalized):
            term_type = "generic_business"
        elif re.search(r"[A-Za-z]", normalized) and re.search(r"[0-9&+._-]|[A-Z]", term):
            term_type = "technical_or_named_term"
        else:
            term_type = "domain_term"
        current = ranked_by_term.get(normalized)
        payload = {
            "term": term,
            "normalized": normalized,
            "source_role": source_role,
            "term_type": term_type,
            "weight": round(weight, 3),
        }
        if current is None or float(current.get("weight") or 0.0) < weight:
            ranked_by_term[normalized] = payload
    return sorted(
        ranked_by_term.values(),
        key=lambda item: (float(item.get("weight") or 0.0), len(str(item.get("normalized") or ""))),
        reverse=True,
    )


def _term_signal_weight(
    term: str,
    source_role: str,
    issue_context: dict[str, Any] | None = None,
) -> float:
    issue_context = issue_context or {}
    normalized = str(term or "").strip()
    if not normalized or len(normalized) <= 1:
        return 0.0
    if _looks_like_source_noise(normalized, issue_context):
        return 0.0
    if re.fullmatch(r"\d+(?:[.,]\d+)*", normalized):
        return 0.0
    if _looks_like_korean_function_word_or_ending(normalized):
        return 0.0

    weight = 0.45
    is_generic = _is_generic_business_term(normalized)
    if is_generic:
        return 0.15
    if source_role in {"profile_business_area", "profile_capability", "profile_recent_change"}:
        weight = max(weight, 0.55)
    elif source_role in {"issue_fact", "structured_issue"}:
        weight = max(weight, 0.6)
    elif source_role == "source_metadata":
        weight = 0.0

    if _appears_repeatedly_in_evidence(normalized, issue_context):
        weight = max(weight, 0.8)
    if _appears_in_structured_issue_fields(normalized, issue_context):
        weight = max(weight, 1.2)
    if re.search(r"[A-Za-z]", normalized) and re.search(r"[0-9&+._-]|[A-Z]", term):
        weight = max(weight, 0.7)
    return weight


def _looks_like_source_noise(
    term: str,
    issue_context: dict[str, Any] | None = None,
) -> bool:
    issue_context = issue_context or {}
    normalized = str(term or "").strip().casefold()
    if not normalized:
        return True
    source_noise_terms = {
        str(item or "").casefold()
        for item in _jsonish_list(issue_context.get("source_noise_terms"))
    }
    return normalized in source_noise_terms


def _looks_like_korean_function_word_or_ending(term: str) -> bool:
    value = str(term or "").strip()
    if not value:
        return True
    if re.fullmatch(r"(이번|해당|관련|통해|위해|대한|따라서|그리고|또는)", value):
        return True
    return bool(
        re.search(
            r"(했다|한다|합니다|있다|있습니다|됐다|됩니다|이며|이다|"
            r"라고|다고|하며|하고|되는|된다|되며|으로서)$",
            value,
        )
    )


def _is_generic_business_term(term: str) -> bool:
    normalized = str(term or "").strip()
    if not normalized:
        return True
    generic_terms = {item for terms in GENERIC_BUSINESS_CATEGORIES.values() for item in terms}
    return normalized in generic_terms


def _appears_in_structured_issue_fields(
    term: str,
    issue_context: dict[str, Any] | None = None,
) -> bool:
    issue_context = issue_context or {}
    normalized = str(term or "").strip()
    if not normalized:
        return False
    structured_values: list[str] = []
    for key in (
        "products_or_services",
        "customers_or_industries",
        "activity_types",
        "target_systems",
        "structured_terms",
    ):
        structured_values.extend(_string_list(issue_context.get(key), max_items=80))
    for value in structured_values:
        for candidate in _raw_normalized_terms(value):
            if _tokens_semantically_close(normalized, candidate):
                return True
    return False


def _appears_repeatedly_in_evidence(
    term: str,
    issue_context: dict[str, Any] | None = None,
) -> bool:
    issue_context = issue_context or {}
    normalized = str(term or "").strip()
    evidence_text = str(issue_context.get("evidence_text") or "")
    if not normalized or not evidence_text:
        return False
    if len(normalized) < 3:
        return False
    return len(re.findall(re.escape(normalized), evidence_text, flags=re.IGNORECASE)) >= 2


def _semantic_fingerprint_for_text(
    text: str,
    issue_context: dict[str, Any] | None = None,
) -> set[str]:
    return {
        str(term.get("normalized") or "")
        for term in _extract_ranked_terms(text, "output_text", issue_context)
        if float(term.get("weight") or 0.0) >= 0.45
        and not _is_generic_business_term(str(term.get("normalized") or ""))
    }


def _activity_types_from_issue(
    integrated_issue: dict[str, Any],
    *,
    classification: dict[str, Any] | None = None,
) -> list[str]:
    values: list[str] = []
    classification = classification or {}

    def add(value: Any) -> None:
        if isinstance(value, str):
            stripped = re.sub(r"\s+", " ", value).strip()
            if stripped:
                values.append(stripped)
            return
        if isinstance(value, dict):
            for nested_key in (
                "activity_type",
                "activity_types",
                "event_type",
                "signal_type",
                "fact_type",
                "type",
            ):
                if nested_key in value:
                    add(value.get(nested_key))
            return
        if isinstance(value, list | tuple | set):
            for item in value:
                add(item)

    add(integrated_issue.get("cluster_event_type"))
    add(integrated_issue.get("event_type"))
    add(classification.get("event_type"))
    add(classification.get("activity_types"))
    add(classification.get("activities"))

    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        add(intelligence.get("activity_types"))
        add(intelligence.get("event_type"))
        for key in ("common_facts", "unique_facts", "key_facts"):
            for fact in _jsonish_list(intelligence.get(key)):
                add(fact)

    for key in ("consolidated_facts", "business_signals"):
        for item in _jsonish_list(integrated_issue.get(key)):
            add(item)

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized[:40]


def _role_mode_from_structured_activity(activity_types: list[str]) -> str:
    if not activity_types:
        return ""
    candidates = _normalize_activity_candidates(activity_types)
    if not candidates:
        return ""
    text = " ".join(candidates)

    if _structured_activity_matches_selected_role(candidates):
        return "selected_operator_or_builder"
    if _structured_activity_matches_active_role(candidates):
        return "active_provider_or_operator"
    if _structured_activity_matches_contract_role(candidates):
        return "contract_related"
    if _structured_activity_matches_partnership_role(candidates):
        return "partnership_governance"
    if _has_uncertain_role(text):
        return "unclear"
    return ""


def _normalize_activity_candidates(activity_types: list[str]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in activity_types:
        text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
        if not text or text in seen:
            continue
        seen.add(text)
        candidates.append(text)
    return candidates


def _structured_activity_matches_selected_role(candidates: list[str]) -> bool:
    exact = {
        "selection",
        "selected",
        "award",
        "operator_selected",
        "사업자 선정",
        "최종 선정",
        "민간 참여자",
        "참여 기업",
    }
    return _activity_candidates_have(candidates, exact=exact)


def _structured_activity_matches_active_role(candidates: list[str]) -> bool:
    exact = {
        "launch",
        "release",
        "service_open",
        "go_live",
        "operation_start",
        "build_completed",
        "deployment_completed",
        "정식 출시",
        "서비스 오픈",
        "서비스 개시",
        "운영 시작",
        "운영 개시",
        "구축 완료",
        "도입 완료",
        "개발 완료",
    }
    return _activity_candidates_have(candidates, exact=exact)


def _structured_activity_matches_contract_role(candidates: list[str]) -> bool:
    exact = {
        "contract",
        "supply_contract",
        "procurement",
        "purchase",
        "order",
        "계약",
        "계약 체결",
        "수주",
        "공급계약",
        "납품 계약",
        "구매 계약",
        "조달 계약",
    }
    return _activity_candidates_have(candidates, exact=exact)


def _structured_activity_matches_partnership_role(candidates: list[str]) -> bool:
    exact = {
        "partnership",
        "collaboration",
        "alliance",
        "mou",
        "consortium",
        "spc",
        "협약",
        "협력",
        "제휴",
        "컨소시엄",
        "실시협약",
        "주주간계약",
        "주주간 계약",
        "공동 추진",
    }
    return _activity_candidates_have(candidates, exact=exact)


def _activity_candidates_have(candidates: list[str], *, exact: set[str]) -> bool:
    exact_normalized = {item.casefold() for item in exact}
    for candidate in candidates:
        if candidate.casefold() in exact_normalized:
            return True
    return False


def _has_uncertain_role(text: str) -> bool:
    return any(marker in text for marker in UNCERTAIN_ACTIVITY_MARKERS)


def _role_mode_from_evidence_fallback(
    evidence_text: str,
    *,
    integrated_issue: dict[str, Any] | None = None,
) -> str:
    text = str(evidence_text or "")
    selected_pattern = (
        r"최종\s*선정|사업자(?:로)?\s*선정|민간\s*참여자(?:로)?\s*(확정|선정)|"
        r"구축\s*사업자(?:로)?\s*(확정|선정)|운영\s*사업자(?:로)?\s*(확정|선정)|"
        r"우선협상(?:대상자)?(?:로)?\s*선정"
    )
    if re.search(selected_pattern, text) and _main_company_near_role_pattern(
        integrated_issue,
        text,
        selected_pattern,
    ):
        return "selected_operator_or_builder"
    if re.search(
        r"실시협약\s*체결|주주간\s*계약\s*체결|SPC\s*설립|컨소시엄\s*(구성|참여)|"
        r"공동\s*추진\s*협약|MOU\s*체결",
        text,
        flags=re.IGNORECASE,
    ):
        return "partnership_governance"
    active_pattern = (
        r"정식\s*출시|서비스\s*(오픈|개시|출시)|구축\s*완료|운영\s*(시작|개시)|"
        r"도입\s*완료|개발\s*완료|제공\s*시작"
    )
    if re.search(active_pattern, text) and _main_company_near_role_pattern(
        integrated_issue,
        text,
        active_pattern,
    ):
        return "active_provider_or_operator"
    if re.search(
        r"계약\s*체결|공급계약\s*체결|수주(?:했다|했다고|계약)|납품\s*계약|"
        r"구매\s*계약|조달\s*계약",
        text,
        flags=re.IGNORECASE,
    ):
        return "contract_related"
    return "unclear"


def _main_company_near_role_pattern(
    integrated_issue: dict[str, Any] | None,
    text: str,
    role_pattern: str,
    *,
    window: int = 56,
) -> bool:
    issue = integrated_issue or {}
    main_company = str(issue.get("main_company") or "").strip()
    if not main_company:
        return True
    role_matches = list(re.finditer(role_pattern, text, flags=re.IGNORECASE))
    if not role_matches:
        return False
    company_matches: list[re.Match[str]] = []
    for pattern in _target_name_patterns(main_company):
        company_matches.extend(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not company_matches:
        return False
    for role_match in role_matches:
        role_start = role_match.start()
        role_end = role_match.end()
        for company_match in company_matches:
            if (
                abs(company_match.start() - role_start) <= window
                or abs(company_match.end() - role_end) <= window
                or 0 <= role_start - company_match.end() <= window
                or 0 <= company_match.start() - role_end <= window
            ):
                return True
    return False


def _evaluate_single_profile_linkage(
    *,
    company_id: str,
    profile: dict[str, Any],
    issue_terms: set[str],
    issue_context: dict[str, Any],
    integrated_issue: dict[str, Any],
    scope: str,
    role_mode: str,
) -> dict[str, Any]:
    if not isinstance(profile, dict) or not profile:
        novelty_status = _business_novelty_status(
            linkage_level="none",
            role_mode=role_mode,
            profile_available=False,
            scope=scope,
        )
        return {
            "company_id": company_id,
            "scope": scope,
            "matched_business_areas": [],
            "matched_capabilities": [],
            "matched_source_refs": [],
            "matched_terms": [],
            "linkage_level": "none",
            "business_novelty_status": novelty_status,
            "implication_mode": _implication_mode_from_linkage(
                linkage_level="none",
                novelty_status=novelty_status,
                role_mode=role_mode,
                scope=scope,
            ),
            "reason": "비교 가능한 프로필 본문이 없습니다.",
        }

    scored_entries: list[tuple[float, set[str], dict[str, Any]]] = []
    for entry in _profile_entries_for_linkage(profile):
        score, matched_terms = _score_profile_entry_against_issue(
            entry=entry,
            issue_terms=issue_terms,
            issue_context=issue_context,
        )
        if score > 0:
            scored_entries.append((score, matched_terms, entry))
    scored_entries.sort(key=lambda item: item[0], reverse=True)

    matched_business_areas: list[dict[str, Any]] = []
    matched_capabilities: list[str] = []
    matched_source_refs: list[Any] = []
    matched_terms_all: list[str] = []
    for score, matched_terms, entry in scored_entries[:5]:
        profile_item = _matched_profile_item_from_entry(
            entry=entry,
            matched_terms=matched_terms,
            score=score,
        )
        if profile_item:
            matched_business_areas.append(profile_item)
        for cap in _string_list(entry.get("capabilities"), max_items=8):
            if cap not in matched_capabilities:
                matched_capabilities.append(cap)
        if entry.get("entry_type") == "capability" and entry.get("name"):
            cap_name = str(entry.get("name") or "").strip()
            if cap_name and cap_name not in matched_capabilities:
                matched_capabilities.append(cap_name)
        matched_source_refs.extend(entry.get("source_refs") or [])
        matched_terms_all.extend(matched_terms)

    linkage_level = _linkage_level_from_profile_score(scored_entries)
    novelty_status = _business_novelty_status(
        linkage_level=linkage_level,
        role_mode=role_mode,
        profile_available=True,
        scope=scope,
    )
    implication_mode = _implication_mode_from_linkage(
        linkage_level=linkage_level,
        novelty_status=novelty_status,
        role_mode=role_mode,
        scope=scope,
    )
    return {
        "company_id": company_id,
        "scope": scope,
        "matched_business_areas": matched_business_areas[:3],
        "matched_capabilities": list(dict.fromkeys(matched_capabilities))[:5],
        "matched_source_refs": _compact_value(matched_source_refs[:5]),
        "matched_terms": sorted(set(matched_terms_all))[:12],
        "linkage_level": linkage_level,
        "business_novelty_status": novelty_status,
        "implication_mode": implication_mode,
        "reason": _profile_linkage_reason(
            company_id=company_id,
            linkage_level=linkage_level,
            novelty_status=novelty_status,
            matched_business_areas=matched_business_areas,
            matched_capabilities=matched_capabilities,
            role_mode=role_mode,
            scope=scope,
        ),
    }


def _matched_profile_item_from_entry(
    *,
    entry: dict[str, Any],
    matched_terms: set[str],
    score: float,
) -> dict[str, Any]:
    name = str(entry.get("name") or "").strip()
    business_area = str(entry.get("business_area") or "").strip()
    business_line = str(entry.get("business_line") or "").strip()
    capabilities = _string_list(entry.get("capabilities"), max_items=8)
    products_or_services = _string_list(entry.get("products_or_services"), max_items=8)
    evidence_texts = _string_list(entry.get("evidence_texts"), max_items=3)
    source_refs = entry.get("source_refs") or []
    if not any([name, business_area, business_line, capabilities, products_or_services]):
        return {}
    return {
        "entry_type": str(entry.get("entry_type") or "").strip(),
        "name": name,
        "business_line": business_line,
        "business_area": business_area
        or (name if entry.get("entry_type") == "business_area" else ""),
        "summary": str(entry.get("summary") or "").strip(),
        "recent_direction": str(entry.get("recent_direction") or "").strip(),
        "matched_capabilities": capabilities,
        "matched_products_or_services": products_or_services,
        "matched_terms": sorted(matched_terms),
        "matched_issue_terms": sorted(matched_terms),
        "evidence_text": evidence_texts[0] if evidence_texts else "",
        "evidence_texts": evidence_texts,
        "source_refs": source_refs,
        "source_ref": source_refs[0] if isinstance(source_refs, list) and source_refs else "",
        "specificity_level": _profile_entry_specificity_level(entry),
        "confidence": min(1.0, round(score / 10, 3)),
    }


def _profile_entry_specificity_level(entry: dict[str, Any]) -> str:
    if _string_list(entry.get("products_or_services"), max_items=1):
        return "product_or_service"
    if _string_list(entry.get("capabilities"), max_items=1):
        return "core_capability"
    if entry.get("entry_type") == "business_area" or entry.get("business_area"):
        return "business_area"
    if entry.get("business_line"):
        return "business_line"
    return "profile_context"


def _business_novelty_status(
    *,
    linkage_level: str,
    role_mode: str,
    profile_available: bool,
    scope: str,
) -> str:
    if scope != "peer":
        return "not_applicable"
    if role_mode == "counterparty_or_customer":
        return "not_new_business_counterparty_role"
    if not profile_available:
        return "profile_insufficient_cannot_judge_novelty"
    if linkage_level in {"high", "medium"}:
        return "existing_profile_business_linked"
    if linkage_level == "low":
        return "weak_profile_linkage"
    if role_mode in {"selected_operator_or_builder", "active_provider_or_operator"}:
        return "new_or_untracked_business_signal"
    if role_mode in {"contract_related", "partnership_governance"}:
        return "role_sensitive_untracked_signal"
    return "uncertain_not_enough_to_call_new_business"


def _implication_mode_from_linkage(
    *,
    linkage_level: str,
    novelty_status: str,
    role_mode: str,
    scope: str,
) -> str:
    if scope == "peer":
        if role_mode == "counterparty_or_customer":
            return "conservative_counterparty"
        if novelty_status == "existing_profile_business_linked":
            return "profile_based"
        if novelty_status == "new_or_untracked_business_signal":
            return "new_business_signal"
        if novelty_status in {
            "weak_profile_linkage",
            "role_sensitive_untracked_signal",
            "profile_insufficient_cannot_judge_novelty",
        }:
            return "event_based"
        return "conservative_observation"
    if scope == "skax":
        if linkage_level in {"high", "medium"}:
            return "profile_based_action"
        if linkage_level == "low":
            return "cautious_action"
        return "generic_monitoring_action"
    return "conservative_observation"


def _profile_entries_for_linkage(profile: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for area in _jsonish_list(profile.get("business_areas")):
        if not isinstance(area, dict):
            continue
        name = str(area.get("name") or area.get("business_area") or "").strip()
        summary = str(area.get("summary") or "").strip()
        recent_direction = str(area.get("recent_direction") or "").strip()
        capabilities = _string_list(
            area.get("core_capabilities") or area.get("capabilities"),
            max_items=8,
        )
        products_or_services = _string_list(
            area.get("products_or_services")
            or area.get("key_products_services")
            or area.get("services"),
            max_items=8,
        )
        evidence_texts = _string_list(area.get("evidence_texts"), max_items=3)
        if area.get("evidence_text"):
            evidence_texts.append(str(area.get("evidence_text") or ""))
        source_refs = area.get("source_refs") or area.get("source_ref") or []
        entries.append(
            {
                "entry_type": "business_area",
                "business_line": str(area.get("business_line") or area.get("line") or "").strip(),
                "business_area": name,
                "name": name,
                "summary": summary,
                "recent_direction": recent_direction,
                "capabilities": capabilities,
                "products_or_services": products_or_services,
                "evidence_texts": evidence_texts,
                "source_refs": source_refs,
                "relevance_text": " ".join(
                    [
                        name,
                        summary,
                        recent_direction,
                        " ".join(capabilities),
                        " ".join(products_or_services),
                    ]
                ),
                "text": " ".join(
                    [
                        name,
                        summary,
                        recent_direction,
                        " ".join(capabilities),
                        " ".join(products_or_services),
                        " ".join(evidence_texts),
                    ]
                ),
            }
        )
        for cap in capabilities:
            entries.append(
                {
                    "entry_type": "capability",
                    "business_line": str(
                        area.get("business_line") or area.get("line") or ""
                    ).strip(),
                    "business_area": name,
                    "name": cap,
                    "capabilities": [cap],
                    "products_or_services": products_or_services,
                    "evidence_texts": evidence_texts,
                    "source_refs": source_refs,
                    "relevance_text": " ".join([cap, name, " ".join(products_or_services)]),
                    "text": " ".join([cap, name, " ".join(products_or_services)]),
                }
            )

    for product in _jsonish_list(profile.get("key_products_services")):
        if isinstance(product, dict):
            name = str(
                product.get("name")
                or product.get("title")
                or product.get("summary")
                or product.get("service_name")
                or ""
            ).strip()
            text = _json_dumps(_compact_profile_item(product))
            source_refs = product.get("source_refs") or product.get("source_ref") or []
            evidence_texts = _string_list(product.get("evidence_texts"), max_items=3)
            if product.get("evidence_text"):
                evidence_texts.append(str(product.get("evidence_text") or ""))
        else:
            name = str(product or "").strip()
            text = name
            source_refs = []
            evidence_texts = []
        if name:
            entries.append(
                {
                    "entry_type": "product_or_service",
                    "business_line": "",
                    "business_area": "",
                    "name": name,
                    "capabilities": [],
                    "products_or_services": [name],
                    "evidence_texts": evidence_texts,
                    "source_refs": source_refs,
                    "relevance_text": name,
                    "text": text,
                }
            )

    for cap in _jsonish_list(profile.get("core_capabilities")):
        if isinstance(cap, dict):
            name = str(cap.get("name") or cap.get("summary") or "").strip()
            text = _json_dumps(cap)
            source_refs = cap.get("source_refs") or cap.get("source_ref") or []
            evidence_texts = _string_list(cap.get("evidence_texts"), max_items=3)
            if cap.get("evidence_text"):
                evidence_texts.append(str(cap.get("evidence_text") or ""))
        else:
            name = str(cap or "").strip()
            text = name
            source_refs = []
            evidence_texts = []
        if name:
            entries.append(
                {
                    "entry_type": "capability",
                    "business_line": "",
                    "business_area": "",
                    "name": name,
                    "capabilities": [name],
                    "products_or_services": [],
                    "evidence_texts": evidence_texts,
                    "source_refs": source_refs,
                    "relevance_text": name,
                    "text": text,
                }
            )

    for key in ("capability_evolution", "recent_changes", "execution_cases"):
        for item in _jsonish_list(profile.get(key)):
            if isinstance(item, dict):
                text = _json_dumps(_compact_profile_item(item))
                source_refs = item.get("source_refs") or item.get("source_ref") or []
                name = str(
                    item.get("name") or item.get("summary") or item.get("title") or ""
                ).strip()
                products_or_services = _string_list(
                    item.get("products_or_services")
                    or item.get("key_products_services")
                    or item.get("services"),
                    max_items=8,
                )
                capabilities = _string_list(
                    item.get("core_capabilities") or item.get("capabilities"),
                    max_items=8,
                )
                evidence_texts = _string_list(item.get("evidence_texts"), max_items=3)
                if item.get("evidence_text"):
                    evidence_texts.append(str(item.get("evidence_text") or ""))
            else:
                text = str(item or "")
                source_refs = []
                name = text[:40]
                products_or_services = []
                capabilities = []
                evidence_texts = []
            if text.strip():
                entries.append(
                    {
                        "entry_type": key,
                        "business_line": "",
                        "business_area": "",
                        "name": name,
                        "capabilities": capabilities,
                        "products_or_services": products_or_services,
                        "evidence_texts": evidence_texts,
                        "source_refs": source_refs,
                        "relevance_text": " ".join(
                            [
                                name,
                                " ".join(capabilities),
                                " ".join(products_or_services),
                            ]
                        ),
                        "text": text,
                    }
                )
    return entries


def _score_profile_entry_against_issue(
    *,
    entry: dict[str, Any],
    issue_terms: set[str],
    issue_context: dict[str, Any],
) -> tuple[float, set[str]]:
    structured_score, structured_matches = _structured_field_overlap_score(
        entry,
        issue_context,
    )
    weighted_score, weighted_matches = _weighted_term_overlap_score(
        entry,
        issue_terms=issue_terms,
        issue_context=issue_context,
    )
    evidence_score, evidence_matches = _evidence_repeat_score(entry, issue_context)
    score = structured_score * 0.6 + weighted_score * 0.3 + evidence_score * 0.1
    matched_terms = structured_matches | weighted_matches | evidence_matches
    if (
        entry.get("entry_type") in {"capability_evolution", "recent_changes", "execution_cases"}
        and not _string_list(entry.get("products_or_services"), max_items=1)
        and not _string_list(entry.get("capabilities"), max_items=1)
    ):
        score = min(score, 1.0)
    if entry.get("entry_type") == "business_area" and structured_score >= 3:
        score += 2
    if entry.get("entry_type") == "capability" and structured_score >= 2:
        score += 1
    if entry.get("source_refs"):
        score += 0.5
    return score, matched_terms


def _structured_field_overlap_score(
    entry: dict[str, Any],
    issue_context: dict[str, Any],
) -> tuple[float, set[str]]:
    entry_text = _profile_entry_relevance_text(entry)
    entry_terms = {
        term["normalized"]
        for term in _extract_ranked_terms(entry_text, "profile_business_area", issue_context)
        if float(term.get("weight") or 0.0) >= 0.2
    }
    structured_groups = (
        ("products_or_services", 4.0),
        ("customers_or_industries", 3.0),
        ("activity_types", 2.0),
        ("target_systems", 3.0),
    )
    score = 0.0
    matches: set[str] = set()
    for key, weight in structured_groups:
        for raw_term in _string_list(issue_context.get(key), max_items=40):
            for structured_term in _extract_ranked_terms(
                raw_term,
                "issue_fact",
                issue_context,
            ):
                normalized = str(structured_term.get("normalized") or "")
                if not normalized:
                    continue
                if float(structured_term.get("weight") or 0.0) < 0.5:
                    continue
                if any(
                    _tokens_semantically_close(normalized, entry_term) for entry_term in entry_terms
                ):
                    score += weight
                    matches.add(normalized)
                    break
    return score, matches


def _weighted_term_overlap_score(
    entry: dict[str, Any],
    *,
    issue_terms: set[str],
    issue_context: dict[str, Any],
) -> tuple[float, set[str]]:
    entry_terms = _extract_ranked_terms(
        _profile_entry_relevance_text(entry),
        "profile_business_area",
        issue_context,
    )
    issue_ranked_terms = [
        term
        for term in _extract_ranked_terms(
            " ".join(sorted(issue_terms)),
            "issue_fact",
            issue_context,
        )
        if float(term.get("weight") or 0.0) > 0.0
    ]
    score = 0.0
    matches: set[str] = set()
    for issue_term in issue_ranked_terms:
        issue_normalized = str(issue_term.get("normalized") or "")
        issue_weight = float(issue_term.get("weight") or 0.0)
        if not issue_normalized:
            continue
        for entry_term in entry_terms:
            entry_normalized = str(entry_term.get("normalized") or "")
            entry_weight = float(entry_term.get("weight") or 0.0)
            if not entry_normalized or min(issue_weight, entry_weight) <= 0.0:
                continue
            if _tokens_semantically_close(issue_normalized, entry_normalized):
                score += min(issue_weight, entry_weight)
                matches.add(issue_normalized)
                break
    return score, matches


def _evidence_repeat_score(
    entry: dict[str, Any],
    issue_context: dict[str, Any],
) -> tuple[float, set[str]]:
    entry_terms = {
        str(term.get("normalized") or "")
        for term in _extract_ranked_terms(
            str(entry.get("text") or ""),
            "profile_business_area",
            issue_context,
        )
        if float(term.get("weight") or 0.0) >= 0.5
    }
    repeated = {
        term
        for term in entry_terms
        if term and _appears_repeatedly_in_evidence(term, issue_context)
    }
    return float(len(repeated)), repeated


def _profile_entry_relevance_text(entry: dict[str, Any]) -> str:
    text = str(entry.get("relevance_text") or "").strip()
    return text if text else str(entry.get("text") or "")


def _linkage_level_from_profile_score(
    scored_entries: list[tuple[float, set[str], dict[str, Any]]],
) -> str:
    if not scored_entries:
        return "none"
    top_entries = scored_entries[:5]
    total_score = sum(score for score, _, _ in top_entries)
    matched_terms: set[str] = set()
    for _, terms, _ in top_entries:
        matched_terms.update(terms)
    if total_score >= 8 and len(matched_terms) >= 3:
        return "high"
    if total_score >= 4 and len(matched_terms) >= 2:
        return "medium"
    if total_score >= 1.5:
        return "low"
    return "none"


def _profile_linkage_reason(
    *,
    company_id: str,
    linkage_level: str,
    novelty_status: str,
    matched_business_areas: list[dict[str, Any]],
    matched_capabilities: list[str],
    role_mode: str,
    scope: str,
) -> str:
    if linkage_level in {"high", "medium"}:
        names = [
            str(item.get("name") or "").strip()
            for item in matched_business_areas[:2]
            if isinstance(item, dict) and item.get("name")
        ]
        caps = matched_capabilities[:3]
        joined = ", ".join([*names, *caps]) or "관련 프로필 항목"
        return (
            f"{company_id}의 프로필 중 {joined} 항목이 현재 사건의 핵심 대상/산업/기술 "
            "토큰과 겹칩니다. 이는 기존 사업영역과의 해석 접점일 뿐, "
            "성과·확장·우위의 근거는 아닙니다."
        )
    if scope == "peer" and novelty_status == "not_new_business_counterparty_role":
        return "현재 피어는 계약 상대방/고객 슬롯으로 보이므로 사업 확장으로 단정하지 않습니다."
    if scope == "peer" and novelty_status == "new_or_untracked_business_signal":
        return (
            "현재 사건에서는 직접 사업/운영 정황이 있으나 "
            "프로필에서 강한 기존 사업 연결은 약합니다."
        )
    if scope == "peer" and novelty_status == "profile_insufficient_cannot_judge_novelty":
        return (
            "프로필 본문이 부족해 신규 사업 여부를 판단할 수 없습니다. 신규 사업 확정이 "
            "아니라 후속 프로필 보강이 필요한 관찰 신호로 다뤄야 합니다."
        )
    if role_mode == "unclear":
        return "현재 사건의 피어 역할이 불명확하므로 보수적 관찰 신호로 해석해야 합니다."
    return "현재 사건과 프로필의 직접 접점이 약하므로 사건 기반 해석을 우선합니다."


def _profile_linkage_guidance(
    peer_linkages: list[dict[str, Any]],
    skax_linkage: dict[str, Any],
    role_mode: str,
) -> list[str]:
    guidance = [f"peer_role_mode={role_mode}"]
    for linkage in peer_linkages:
        company_id = linkage.get("company_id")
        guidance.append(
            f"{company_id}: implication_mode={linkage.get('implication_mode')}, "
            f"linkage_level={linkage.get('linkage_level')}, "
            f"business_novelty_status={linkage.get('business_novelty_status')}"
        )
    guidance.append(
        "sk_ax: implication_mode="
        f"{skax_linkage.get('implication_mode')}, linkage_level={skax_linkage.get('linkage_level')}"
    )
    return guidance


def _action_artifact_plan_for_prompt(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any],
) -> dict[str, Any]:
    signals = _extract_issue_structured_signals(
        integrated_issue=integrated_issue,
        classification=classification,
    )
    return {
        "artifact_generation_mode": "llm_dynamic",
        "current_issue_signals": {
            "event_type": signals.get("event_type"),
            "activity_types": _string_list(signals.get("activity_types"), max_items=20),
            "products_or_services": _string_list(
                signals.get("products_or_services"),
                max_items=12,
            ),
            "customers_or_industries": _string_list(
                signals.get("customers_or_industries"),
                max_items=12,
            ),
            "target_systems": _string_list(signals.get("target_systems"), max_items=12),
            "structured_terms": _string_list(signals.get("structured_terms"), max_items=30),
            "evidence_terms": _issue_evidence_terms_for_action_plan(integrated_issue),
        },
        "skax_implication_mode": (
            (profile_linkage_evaluation.get("skax_linkage") or {}).get("implication_mode")
            if isinstance(profile_linkage_evaluation, dict)
            else ""
        ),
        "artifact_policy": [
            (
                "현재 사건에서 확인된 대상 사업/시스템/서비스/인프라를 기준으로 "
                "내부 점검 항목을 만든다."
            ),
            (
                "유사 고객군/유사 사업에서 피어사 사업군·역량과 SK AX 사업군·역량이 "
                "겹치는 지점과 달라지는 지점을 비교한다."
            ),
            (
                "겹침/차이는 ProfileContext 또는 business_line_mapping 후보에 있는 항목으로만 "
                "작성하고, 없는 사업군/역량명을 새로 만들지 않는다."
            ),
            (
                "SK AX의 대응 가능 범위, 역량 공백, 운영 구조, 영업 전략, "
                "후속 경쟁사 모니터링 항목을 중심으로 구조화한다."
            ),
            (
                "범위, 책임, 일정 조건, 검증 기준, 리스크, 운영 조건, 후속 모니터링 중 "
                "현재 사건에 맞는 항목을 쓴다."
            ),
            (
                "외부 고객 제안 문장이 아니라 SK AX 내부에서 비교하고 보완할 기준을 "
                "문장 안에 포함한다."
            ),
            "현재 사건이나 SK AX 프로필 근거가 없는 기술명/솔루션명/성공 사례는 쓰지 않는다.",
        ],
        "guidance": (
            "이 객체는 고정 taxonomy가 아니라 LLM이 내부 전략 점검 기준을 만들기 위한 정책입니다."
        ),
    }


def _extract_issue_structured_signals(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    activity_types = _activity_types_from_issue(
        integrated_issue,
        classification=classification,
    )
    products_or_services: list[str] = []
    customers_or_industries: list[str] = []
    target_systems: list[str] = []

    def add(target: list[str], value: Any) -> None:
        for item in _string_values_from_any(value):
            if item not in target:
                target.append(item)

    add(products_or_services, classification.get("products_or_services"))
    add(customers_or_industries, classification.get("customers_or_industries"))
    add(target_systems, classification.get("target_systems"))
    for key in ("products_or_services", "customers_or_industries", "target_systems"):
        value = integrated_issue.get(key)
        if key == "products_or_services":
            add(products_or_services, value)
        elif key == "customers_or_industries":
            add(customers_or_industries, value)
        else:
            add(target_systems, value)

    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        add(products_or_services, intelligence.get("products_or_services"))
        add(customers_or_industries, intelligence.get("customers_or_industries"))
        add(target_systems, intelligence.get("target_systems"))

    structured_terms = _issue_structured_terms(
        integrated_issue=integrated_issue,
        classification=classification,
    )
    event_type = str(
        classification.get("event_type") or integrated_issue.get("cluster_event_type") or ""
    ).strip()
    return {
        "event_type": event_type,
        "activity_types": activity_types,
        "products_or_services": products_or_services,
        "customers_or_industries": customers_or_industries,
        "target_systems": target_systems,
        "structured_terms": structured_terms,
        "fallback_text": _integrated_grounding_text(integrated_issue),
    }


def _issue_evidence_terms_for_action_plan(integrated_issue: dict[str, Any]) -> list[str]:
    if not isinstance(integrated_issue, dict):
        return []
    parts = [
        str(integrated_issue.get("headline") or ""),
        str(integrated_issue.get("main_event") or ""),
        str(integrated_issue.get("main_issue") or ""),
        str(integrated_issue.get("one_line_summary") or ""),
    ]
    parts.extend(str(item or "") for item in integrated_issue.get("fact_summary") or [])
    for _, fact_text in _fact_texts(integrated_issue):
        parts.append(fact_text)
    ranked = _extract_ranked_terms(
        "\n".join(part for part in parts if part),
        "issue_evidence",
        {
            "structured_terms": _issue_structured_terms(
                integrated_issue=integrated_issue,
                classification={},
            )
        },
    )
    terms: list[str] = []
    seen: set[str] = set()
    for item in ranked:
        if str(item.get("term_type") or "") in {"source_noise", "function_word_or_ending"}:
            continue
        normalized = str(item.get("normalized") or item.get("term") or "").strip()
        if not normalized or _is_low_signal_content_token(normalized):
            continue
        if normalized in seen:
            continue
        terms.append(normalized)
        seen.add(normalized)
        if len(terms) >= 40:
            break
    return terms


def _string_values_from_any(value: Any) -> list[str]:
    values: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, str):
            stripped = re.sub(r"\s+", " ", item).strip()
            if stripped:
                values.append(stripped)
            return
        if isinstance(item, dict):
            for key in (
                "name",
                "value",
                "summary",
                "title",
                "product",
                "service",
                "industry",
                "target",
                "system",
            ):
                if key in item:
                    collect(item.get(key))
            return
        if isinstance(item, list | tuple | set):
            for nested in item:
                collect(nested)

    collect(value)
    return list(dict.fromkeys(values))


def _issue_structured_terms(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
) -> list[str]:
    terms: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            stripped = re.sub(r"\s+", " ", value).strip()
            if stripped:
                terms.append(stripped)
            return
        if isinstance(value, list | tuple | set):
            for item in value:
                add(item)
            return
        if isinstance(value, dict):
            for key in (
                "activity_type",
                "activity_types",
                "event_type",
                "products_or_services",
                "customers_or_industries",
                "target_systems",
                "business_area",
                "sector",
                "sectors",
                "signal_type",
            ):
                if key in value:
                    add(value.get(key))

    add(_activity_types_from_issue(integrated_issue, classification=classification))
    add(classification.get("sector"))
    add(classification.get("sectors"))
    add(classification.get("products_or_services"))
    add(classification.get("customers_or_industries"))
    for key in ("products_or_services", "customers_or_industries", "target_systems"):
        add(integrated_issue.get(key))

    intelligence = integrated_issue.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for key in (
            "products_or_services",
            "customers_or_industries",
            "target_systems",
            "common_facts",
            "unique_facts",
        ):
            add(intelligence.get(key))

    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return deduped


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
    profile_linkage_evaluation: dict[str, Any] | None = None,
    action_artifact_plan: dict[str, Any] | None = None,
) -> list[str]:
    profile_linkage_evaluation = profile_linkage_evaluation or _build_profile_linkage_evaluation(
        integrated_issue=integrated_issue,
        classification={},
        profile_context=profile_context,
    )
    action_artifact_plan = action_artifact_plan or _action_artifact_plan_for_prompt(
        integrated_issue=integrated_issue,
        classification={},
        profile_linkage_evaluation=profile_linkage_evaluation,
    )
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
        novelty_violation = _business_novelty_overclaim_violation(
            value_text,
            label=label,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if novelty_violation:
            violations.append(f"{label}: {novelty_violation}")
        artifact_plan_violation = _action_artifact_plan_violation(
            value_text,
            label=label,
            action_artifact_plan=action_artifact_plan,
        )
        if artifact_plan_violation:
            violations.append(f"{label}: {artifact_plan_violation}")
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
            profile_linkage_evaluation=profile_linkage_evaluation,
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
        unsupported_profile_violation = _unsupported_peer_profile_claim_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        if unsupported_profile_violation:
            violations.append(f"{label}: {unsupported_profile_violation}")
        unsupported_skax_term_violation = _unsupported_skax_profile_term_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            profile_linkage_evaluation=profile_linkage_evaluation,
        )
        if unsupported_skax_term_violation:
            violations.append(f"{label}: {unsupported_skax_term_violation}")
        unsupported_domain_violation = _unsupported_domain_term_violation(
            value_text,
            label=label,
            integrated_issue=integrated_issue,
        )
        if unsupported_domain_violation:
            violations.append(f"{label}: {unsupported_domain_violation}")
    repetition_violation = _two_section_repetition_violation(result)
    if repetition_violation:
        violations.append(repetition_violation)
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


def _business_novelty_overclaim_violation(
    text: str,
    *,
    label: str,
    profile_linkage_evaluation: dict[str, Any],
) -> str:
    if not text or _is_follow_up_or_watch_field(label):
        return ""
    if not label.startswith(("analysis.", "peer_implication.")):
        return ""
    peer_linkages = [
        linkage
        for linkage in _jsonish_list(profile_linkage_evaluation.get("peer_linkages"))
        if isinstance(linkage, dict)
    ]
    if not peer_linkages:
        return ""
    cautious_terms = re.compile(r"관찰|신호|가능성|후속\s*확인|단정하기\s*어렵|미포착")
    for linkage in peer_linkages:
        novelty = str(linkage.get("business_novelty_status") or "")
        if novelty == "not_new_business_counterparty_role" and _matches_any_pattern(
            text,
            OVERCLAIM_PATTERNS["counterparty"],
        ):
            return (
                "계약 상대방/고객 슬롯인 피어를 신규 사업·입지 강화·역량 강화처럼 과대해석했습니다."
            )
        if novelty in {
            "new_or_untracked_business_signal",
            "profile_insufficient_cannot_judge_novelty",
        } and _matches_any_pattern(text, OVERCLAIM_PATTERNS["new_signal"]):
            if not cautious_terms.search(text):
                return (
                    "프로필에 강하게 포착되지 않은 사업 신호를 확정 성과나 역량 강화처럼 "
                    "단정했습니다. 관찰 신호/후속 확인 수준으로 낮춰야 합니다."
                )
    return ""


def _matches_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _action_artifact_plan_violation(
    text: str,
    *,
    label: str,
    action_artifact_plan: dict[str, Any],
) -> str:
    if not text or not label.startswith("skax_implication.recommended_actions"):
        return ""
    external_phrase = _skax_external_customer_facing_violation(text)
    if external_phrase:
        return external_phrase
    issue_terms = _action_plan_issue_terms(action_artifact_plan)
    if issue_terms and not _action_text_has_issue_signal(text, issue_terms):
        return "대응방향에 현재 사건의 대상 사업/시스템/서비스/고객군 신호가 연결되지 않았습니다."
    if not _action_text_has_internal_strategy_checkpoint(text):
        return "대응방향에 SK AX가 내부적으로 점검할 기준이 부족합니다."
    if not _action_text_has_skax_change(text):
        return "대응방향에 SK AX가 보완하거나 점검할 구체 방식이 부족합니다."
    return ""


def _action_plan_issue_terms(action_artifact_plan: dict[str, Any]) -> set[str]:
    primary_terms = _action_plan_terms_for_keys(
        action_artifact_plan,
        keys=(
            "products_or_services",
            "target_systems",
            "customers_or_industries",
            "evidence_terms",
        ),
    )
    if primary_terms:
        return primary_terms
    return _action_plan_terms_for_keys(
        action_artifact_plan,
        keys=(
            "structured_terms",
            "activity_types",
            "event_type",
        ),
    )


def _action_plan_terms_for_keys(
    action_artifact_plan: dict[str, Any],
    *,
    keys: Sequence[str],
) -> set[str]:
    signals = _json_dict(action_artifact_plan.get("current_issue_signals"))
    terms: set[str] = set()
    for key in keys:
        values = (
            _string_list(signals.get(key), max_items=50)
            if key != "event_type"
            else [str(signals.get(key) or "")]
        )
        for value in values:
            for token in _raw_normalized_terms(value):
                normalized = token.casefold() if token.isascii() else token
                if normalized and not _is_low_signal_content_token(normalized):
                    if _is_generic_business_term(normalized):
                        continue
                    if len(normalized) < 2:
                        continue
                    terms.add(normalized)
    return terms


def _action_text_has_issue_signal(text: str, issue_terms: set[str]) -> bool:
    output_terms = _content_tokens(text)
    if output_terms & issue_terms:
        return True
    return any(
        _tokens_semantically_close(output_term, issue_term)
        for output_term in output_terms
        for issue_term in issue_terms
    )


def _skax_external_customer_facing_violation(text: str) -> str:
    value = str(text or "")
    if re.search(
        r"고객(이|은|에게|한테).{0,24}(확인|비교|평가|판단|볼 수|보여|제시|설명)",
        value,
    ) or re.search(r"고객\s*제안|고객\s*확인\s*기준|고객이\s*확인", value):
        return (
            "대응방향이 외부 고객 제안/확인 문장처럼 작성되었습니다. "
            "유사 고객군/유사 사업 관점은 유지하되, 외부 고객 제안 문장이 아니라 "
            "SK AX 내부에서 경쟁사 사업군과 자사 사업군의 겹침/차이, 대응 가능 범위, "
            "역량 공백, 운영·영업 전략, 후속 모니터링 항목을 점검하는 문장으로 "
            "바꿔야 합니다."
        )
    return ""


def _action_text_has_internal_strategy_checkpoint(text: str) -> bool:
    value = str(text or "")
    matched_groups = [
        group for group, pattern in _INTERNAL_CHECKPOINT_GROUPS.items() if re.search(pattern, value)
    ]
    return len(matched_groups) >= 2


def _action_text_has_skax_change(text: str) -> bool:
    value = str(text or "")
    return any(re.search(pattern, value) for pattern in _SKAX_ACTION_VERB_GROUPS.values())


def _unsupported_skax_profile_term_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
) -> str:
    if not label.startswith("skax_implication."):
        return ""
    if _is_follow_up_or_watch_field(label):
        return ""
    issue_tokens = _issue_relevance_tokens(integrated_issue)
    output_tokens = _content_tokens(text)
    skax_terms = _concrete_profile_terms(
        profile_context,
        integrated_issue=integrated_issue,
        scope="skax",
    )
    used_skax_terms = sorted(output_tokens & skax_terms)
    if not used_skax_terms:
        return ""
    linkage_level = _relevant_profile_linkage_level_from_evaluation(
        profile_linkage_evaluation,
        scope="skax",
    ) or _relevant_profile_linkage_level(
        profile_context,
        integrated_issue=integrated_issue,
        scope="skax",
    )
    if linkage_level in {"high", "medium"}:
        return ""
    unsupported = sorted(set(used_skax_terms) - issue_tokens)
    if not unsupported:
        return ""
    return (
        "SK AX 프로필 연결이 약한 상태에서 구체 SK AX 사업/역량 용어를 사용했습니다: "
        f"{', '.join(unsupported[:3])}. 대응 방향은 현재 사건의 적용 범위/대상 업무/"
        "검증 기준으로 낮춰야 합니다."
    )


def _unsupported_domain_term_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
) -> str:
    if not label.startswith("skax_implication."):
        return ""
    if _is_follow_up_or_watch_field(label):
        return ""
    evidence_text = _integrated_grounding_text(integrated_issue)
    evidence_tokens = _content_tokens(evidence_text)
    output_tokens = _content_tokens(text)
    unsupported = sorted(
        domain
        for domain in output_tokens & set(_DOMAIN_ALIASES)
        if not _domain_supported_by_evidence(
            domain,
            evidence_tokens=evidence_tokens,
            evidence_text=evidence_text,
        )
    )
    if not unsupported:
        return ""
    return (
        "현재 사건 근거에 없는 산업/도메인 용어를 SK AX 대응 방향에 사용했습니다: "
        f"{', '.join(unsupported[:3])}. 현재 사건의 대상 업무/적용 범위/"
        "검증 기준으로 낮춰야 합니다."
    )


def _domain_supported_by_evidence(
    domain: str,
    *,
    evidence_tokens: set[str],
    evidence_text: str,
) -> bool:
    aliases = _DOMAIN_ALIASES.get(domain, {domain})
    evidence_lower = str(evidence_text or "").casefold()
    return any(
        alias.casefold() in evidence_lower or alias.casefold() in evidence_tokens
        for alias in aliases
    )


def _two_section_repetition_violation(result: dict[str, Any]) -> str:
    analysis = result.get("analysis") or {}
    implication = result.get("implication") or {}
    peer = implication.get("peer_implication") or {}
    insight_texts = [
        str(analysis.get("analysis_summary") or ""),
        *[str(item or "") for item in _string_list(analysis.get("strategic_meaning"), max_items=3)],
        str(analysis.get("market_signal") or ""),
        str(peer.get("peer_meaning") or ""),
        str(peer.get("capability_change") or ""),
    ]
    normalized: list[set[str]] = []
    labels: list[str] = []
    for index, insight_text in enumerate(insight_texts):
        tokens = _high_signal_tokens_for_repetition(insight_text)
        if len(tokens) < 4:
            continue
        normalized.append(tokens)
        labels.append(f"시사점 필드 {index + 1}")
    repeated_pairs: list[tuple[str, str]] = []
    for left_index, left_tokens in enumerate(normalized):
        for right_index in range(left_index + 1, len(normalized)):
            right_tokens = normalized[right_index]
            overlap = len(left_tokens & right_tokens)
            smaller = max(1, min(len(left_tokens), len(right_tokens)))
            if overlap / smaller >= 0.75:
                repeated_pairs.append((labels[left_index], labels[right_index]))
    if len(repeated_pairs) >= 2:
        first, second = repeated_pairs[0]
        return (
            f"{first}와 {second} 등 여러 시사점 필드가 같은 의미를 반복합니다. "
            "analysis 와 peer_implication 은 별도 노출 섹션이 아니라 하나의 "
            "시사점 묶음으로 압축해야 합니다."
        )
    direction_terms = ("가속화", "확장", "확대")
    repeated_direction_count = sum(
        1 for text in insight_texts if any(term in text for term in direction_terms)
    )
    if repeated_direction_count >= 3:
        return (
            "시사점 필드 여러 곳에서 가속화/확장/확대 같은 방향성 표현을 반복합니다. "
            "카드 요약을 반복하지 말고 적용 범위, 대상 업무, 검증 기준으로 나눠 써야 합니다."
        )
    return ""


def _high_signal_tokens_for_repetition(text: str) -> set[str]:
    return {token for token in _semantic_fingerprint_for_text(text) if len(token) >= 3}


def _has_unsupported_pattern(text: str, pattern: str, *, evidence_text: str) -> bool:
    if not re.search(pattern, text):
        return False
    if re.search(r"점검|비교|확인|모니터링|여부|기준", text):
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

    current_event_terms = r"이번|해당|계약|사업|프로젝트|과제|수주|협약"
    direct_role_patterns = (
        rf"(?:{current_event_terms}).{{0,28}}(?:제공|공급|수행|구축|운영|지원|추진|참여|기여)",
        rf"(?:제공|공급|수행|구축|운영|지원|추진|참여|기여).{{0,28}}(?:{current_event_terms})",
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
    profile_background_statement = re.search(
        r"(제공하는|보유한)\s*기업|기존\s*(사업영역|역량)|프로필\s*접점|프로필상",
        text,
    ) and not re.search(
        rf"(?:{current_event_terms}).{{0,28}}(?:제공|공급|수행|구축|운영|지원|추진|참여|기여)",
        text,
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

    if label.startswith("skax_implication.recommended_actions") and any(
        re.search(
            pattern + r".{0,24}(에게|대상|상대로|제안|제시|영업)",
            text,
            flags=re.IGNORECASE,
        )
        for pattern in target_patterns
    ):
        if re.search(r"SK\s*AX|유사\s*고객군|유사\s*사업", text, flags=re.IGNORECASE):
            return ""
        return (
            "SK AX 대응을 타깃 피어의 특정 프로젝트에 직접 제안하는 것처럼 썼습니다. "
            "유사 고객군/유사 사업 관점은 유지하되, 외부 고객 제안 문장이 아니라 "
            "SK AX 내부에서 경쟁사 사업군과 자사 사업군의 겹침/차이, 대응 가능 범위, "
            "역량 공백, 운영·영업 전략, 후속 모니터링 항목을 점검하는 문장으로 "
            "바꿔야 합니다."
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
            "대응방향이 주가/시장 반응을 실행 근거로 사용했습니다. 전략 대응은 현재 사건의 "
            "사업 범위, 운영 조건, 검증 기준, 프로필 접점 중심으로 작성해야 합니다."
        )
    if re.search(r"클라우드|AI|인공지능|에이아이", text, flags=re.IGNORECASE):
        evidence_has_tech = re.search(
            r"클라우드|AI|인공지능|에이아이",
            evidence_text,
            flags=re.IGNORECASE,
        )
        profile_support = _has_concrete_profile_term(
            text,
            profile_context=profile_context or {},
            integrated_issue=integrated_issue or {},
            scope="skax",
        )
        if not evidence_has_tech and not profile_support:
            return (
                "현재 사건 근거 또는 SK AX 프로필 접점 없이 기술명을 대응방향에 사용했습니다. "
                "대상 시스템, 전환 범위, 업무 영향도, 운영 책임, 검증 기준 중심으로 낮춰야 합니다."
            )
    if re.search(r"성공|수주에\s*영향|신뢰성", text) and not _profile_has_execution_case(
        profile_context,
        integrated_issue=integrated_issue,
    ):
        return (
            "ProfileContext에 실행 사례 근거가 없는데 성공/수주 영향/신뢰성을 사용했습니다. "
            "SK AX가 내부적으로 점검할 검증 기준과 운영 조건 중심으로 낮춰야 합니다."
        )
    if "솔루션" in text and not _solution_term_supported_by_evidence(
        text,
        integrated_issue=integrated_issue or {},
        profile_context=profile_context or {},
    ):
        return (
            "대응방향이 근거 없는 솔루션 표현에 머물렀습니다. "
            "현재 사건의 전환 범위, 업무 영향도, 운영 책임, 검증 기준처럼 "
            "SK AX가 내부적으로 점검할 기준으로 낮춰야 합니다."
        )
    if re.search(r"성능.{0,12}(강조|입증)|검증된\s*성능", text):
        return (
            "대응방향이 성능 강조 같은 일반 표현에 머물렀습니다. "
            "현재 사건의 전환 범위, 업무 영향도, 운영 책임, 성능/용량 검증 기준처럼 "
            "SK AX가 내부적으로 점검할 기준으로 낮춰야 합니다."
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
    return ""


def _solution_term_supported_by_evidence(
    text: str,
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> bool:
    del text
    evidence_text = _integrated_grounding_text(integrated_issue)
    if "솔루션" in evidence_text:
        return True
    profile_terms = _concrete_profile_terms(
        profile_context,
        integrated_issue=integrated_issue,
        scope="skax",
    ) | _concrete_profile_terms(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    return "솔루션" in profile_terms


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
    if "솔루션" in text_value and not (
        _action_text_has_internal_strategy_checkpoint(text_value)
        or _high_signal_issue_overlap_count(text_value, integrated_issue or {}) >= 2
    ):
        return (
            "SK AX 영향/대응을 일반 솔루션 표현으로 썼습니다. 현재 사건에서 확인된 "
            "전환 범위, 업무 영향도, 운영 책임, 검증 기준 중심으로 낮춰야 합니다."
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
            "사용했습니다. SK AX가 내부적으로 점검할 범위, 책임, 검증 기준, 운영 조건 중심으로 "
            "낮춰야 합니다."
        )
    return ""


def _scope_expansion_guard_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_evaluation: dict[str, Any] | None = None,
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

    if _has_status_strength_claim(value) and not _has_status_strength_event_support(event_lower):
        return (
            "지위 강화나 역량 검증처럼 강한 표현을 썼지만 "
            "선정, 수주, 공식 계약, 실행 근거 등 직접 근거가 부족합니다. "
            "관찰 신호나 연결 사례 수준으로 낮춰야 합니다."
        )

    if (
        label.startswith(("analysis.", "peer_implication."))
        and (
            _has_status_strength_claim(value)
            or _has_broad_expansion_claim(value)
            or _has_effectiveness_claim(value)
        )
        and (
            _relevant_profile_linkage_level_from_evaluation(
                profile_linkage_evaluation,
                scope="peer",
            )
            or _relevant_profile_linkage_level(
                profile_context,
                integrated_issue=integrated_issue,
                scope="peer",
            )
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
        peer_linkage_level = _relevant_profile_linkage_level_from_evaluation(
            profile_linkage_evaluation,
            scope="peer",
        ) or _relevant_profile_linkage_level(
            profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
        has_issue_connection = _high_signal_issue_overlap_count(value, integrated_issue) >= 2
        has_profile_connection = _has_concrete_profile_term(
            value,
            profile_context=profile_context,
            integrated_issue=integrated_issue,
            scope="peer",
        )
        if (
            label == "analysis.impact_reason"
            and has_issue_connection
            and _has_status_strength_event_support(event_lower)
        ):
            return ""
        attention_supported = not _has_attention_growth_claim(value) or re.search(
            r"관심|주목",
            event_lower,
        )
        if not (
            has_issue_connection
            and has_profile_connection
            and peer_linkage_level in {"high", "medium"}
            and attention_supported
        ):
            return (
                "피어 시사점이 입지 강화/영역 확장/관심 반영 같은 강한 표현을 사용했지만 "
                "현재 사건의 구체 사실과 피어 프로필 접점이 함께 보이지 않습니다. "
                "사실-프로필-사업적 의미가 연결되도록 쓰거나 관찰 신호 수준으로 낮춰야 합니다."
            )

    if _has_broad_expansion_claim(value):
        scope = "skax" if label.startswith("skax_implication") else "peer"
        linkage_level = _relevant_profile_linkage_level_from_evaluation(
            profile_linkage_evaluation,
            scope=scope,
        ) or _relevant_profile_linkage_level(
            profile_context,
            integrated_issue=integrated_issue,
            scope=scope,
        )
        if (
            linkage_level not in {"high", "medium"}
            or not _has_expansion_support(event_lower)
            or _high_signal_issue_overlap_count(value, integrated_issue) < 2
        ):
            return (
                "사업영역/서비스 확장 표현을 사용했지만 현재 사건과 관련 프로필의 연결 또는 "
                "범위 확대 근거가 충분하지 않습니다. 연결 사례나 참여 기반처럼 "
                "강도를 낮춰야 합니다."
            )

    if _has_attention_growth_claim(value) and not re.search(r"관심|주목", event_lower):
        return (
            "관심 증가/주목 같은 시장 반응 표현을 원문 근거 없이 사용했습니다. "
            "확인된 사업, 수요 신호, 비교 기준 변화로 낮춰야 합니다."
        )

    if _has_effectiveness_claim(value):
        scope = "skax" if label.startswith("skax_implication") else "peer"
        linkage_level = _relevant_profile_linkage_level_from_evaluation(
            profile_linkage_evaluation,
            scope=scope,
        ) or _relevant_profile_linkage_level(
            profile_context,
            integrated_issue=integrated_issue,
            scope=scope,
        )
        if linkage_level not in {"high", "medium"} or not _has_effect_scope(value):
            return (
                "긍정적 영향·경쟁력 강화·운영 효율성 향상 같은 효과성 표현에 "
                "현재 사건, 관련 프로필 역량, 기대효과 범위가 함께 보이지 않습니다. "
                "관찰 신호나 검증 계기 수준으로 낮춰야 합니다."
            )

    return ""


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


def _has_status_strength_event_support(text: str) -> bool:
    return bool(
        re.search(
            r"최종\s*선정|사업자\s*선정|우선협상|민간\s*참여자|"
            r"공식\s*협약|실시협약|주주간\s*계약|"
            r"대형\s*수주|공급계약\s*체결|계약\s*체결|레퍼런스",
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
    return bool(re.search(r"관심|주목", str(text or "")))


def _high_signal_issue_overlap_count(text: str, integrated_issue: dict[str, Any]) -> int:
    issue_tokens = _issue_relevance_tokens(integrated_issue)
    text_tokens = _semantic_fingerprint_for_text(str(text or ""))
    overlap = {
        token
        for token in issue_tokens & text_tokens
        if len(token) >= 2 and not _is_generic_business_term(token)
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
    issue_context = _profile_linkage_issue_context(
        integrated_issue=integrated_issue,
        classification={},
    )
    if scope == "skax":
        profiles = [profile_context.get("skax_profile") or {}]
    else:
        peer_profiles = profile_context.get("peer_profiles") or {}
        profiles = []
        if isinstance(peer_profiles, dict):
            for company_id in _companies_from_integrated_issue(integrated_issue):
                profile = peer_profiles.get(company_id) or {}
                if isinstance(profile, dict):
                    profiles.append(profile)
    terms: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        for entry in _profile_entries_for_linkage(profile):
            for term in _extract_ranked_terms(
                str(entry.get("text") or ""),
                "profile_business_area",
                issue_context,
            ):
                normalized = str(term.get("normalized") or "")
                if float(term.get("weight") or 0.0) >= 0.5:
                    terms.add(normalized)
    company_identity_terms = _company_identity_terms(integrated_issue)
    return {
        term
        for term in terms
        if term not in company_identity_terms
        and not _looks_like_source_noise(term, issue_context)
        and not _looks_like_korean_function_word_or_ending(term)
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
    out = _scrub_failed_output(json.loads(json.dumps(result, ensure_ascii=False, default=str)))
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


def _scrub_failed_output(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _scrub_failed_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [
            cleaned for item in value if (cleaned := _scrub_failed_output(item)) not in ("", [], {})
        ]
    if isinstance(value, str):
        return "" if _contains_high_risk_unsupported_claim(value) else value
    return value


def _contains_high_risk_unsupported_claim(text: str) -> bool:
    return any(re.search(pattern, str(text or "")) for pattern in _UNSUPPORTED_CLAIM_PATTERNS)


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
    out = _ensure_reasoning_debug_fields(
        json.loads(json.dumps(result, ensure_ascii=False, default=str)),
        integrated_issue=integrated_issue,
    )
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
                    analysis[key] = ""
        analysis["strategic_meaning"] = [
            _repair_customer_role_overstatement(item)
            for item in _string_list(analysis.get("strategic_meaning"), max_items=3)
        ]
        analysis["strategic_meaning"] = [
            item
            for index, item in enumerate(analysis["strategic_meaning"], start=1)
            if not _hard_quality_violation_for_text(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
            )
        ]
        for key in ("peer_meaning", "capability_change"):
            if peer.get(key):
                peer[key] = _repair_customer_role_overstatement(str(peer[key]))
                if _hard_quality_violation_for_text(
                    peer[key],
                    label=f"peer_implication.{key}",
                    integrated_issue=integrated_issue,
                ):
                    peer[key] = ""
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
            peer["peer_meaning"] = ""
            peer["capability_change"] = ""

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
        ):
            analysis[key] = ""

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
        for index, item in enumerate(strategic_items, start=1)
    )
    if has_hard_or_scope_strategic_violation:
        analysis["strategic_meaning"] = [
            item
            for index, item in enumerate(strategic_items, start=1)
            if not _hard_quality_violation_for_text(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
            )
            and not _scope_expansion_guard_violation(
                item,
                label=f"analysis.strategic_meaning[{index}]",
                integrated_issue=integrated_issue,
                profile_context=profile_context or {},
            )
        ]

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
        ):
            peer[key] = ""

    for key in ("why_important", "potential_impact"):
        if skax.get(key) and (
            _hard_quality_violation_for_text(
                str(skax[key]),
                label=f"skax_implication.{key}",
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
        ):
            skax[key] = ""

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
    ]

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
    action_artifact_plan: dict[str, Any] | None = None,
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
        if _repair_action_violation(
            action,
            label=label,
            integrated_issue=integrated_issue,
            profile_context=profile_context or {},
            action_artifact_plan=action_artifact_plan or {},
        ):
            continue
        safe_actions.append(action)

    skax["recommended_actions"] = safe_actions[:3]
    implication["skax_implication"] = skax
    out["implication"] = implication
    return out


def _repair_action_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    action_artifact_plan: dict[str, Any] | None = None,
) -> str:
    violation = _recommended_action_quality_violation(
        text,
        label=label,
        integrated_issue=integrated_issue,
        profile_context=profile_context,
    )
    if violation:
        return violation
    if _hard_quality_violation_for_text(
        text,
        label=label,
        integrated_issue=integrated_issue,
    ):
        return "대응방향이 현재 사건의 근거 범위를 벗어난 표현을 포함했습니다."
    return _counterparty_role_action_violation(
        text,
        label=label,
        integrated_issue=integrated_issue,
    ) or _action_artifact_plan_violation(
        text,
        label=label,
        action_artifact_plan=action_artifact_plan or {},
    )


def _event_based_recommended_actions(
    integrated_issue: dict[str, Any],
    *,
    action_artifact_plan: dict[str, Any] | None = None,
) -> list[str]:
    """Deprecated: final action copy must come from LLM repair, not templates."""
    del integrated_issue
    del action_artifact_plan
    return []


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
    evaluation = _build_profile_linkage_evaluation(
        integrated_issue=integrated_issue,
        classification={},
        profile_context=profile_context,
    )
    linkage = _profile_linkage_for_company(
        evaluation,
        company_id=company_id,
        scope="peer",
    )
    return {
        "company": company_id,
        "matched_profile_terms": _string_list(linkage.get("matched_terms"), max_items=12),
        "linkage_level": str(linkage.get("linkage_level") or "none"),
        "business_novelty_status": str(linkage.get("business_novelty_status") or ""),
        "implication_mode": str(linkage.get("implication_mode") or ""),
        "reason": str(linkage.get("reason") or "현재 이슈와 비교할 피어 프로필 본문이 없습니다."),
    }


def _relevant_profile_linkage_level(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    scope: str,
) -> str:
    evaluation = _build_profile_linkage_evaluation(
        integrated_issue=integrated_issue,
        classification={},
        profile_context=profile_context,
    )
    if scope == "skax":
        linkage = evaluation.get("skax_linkage") if isinstance(evaluation, dict) else {}
        return str((linkage or {}).get("linkage_level") or "none")
    best = "none"
    for linkage in _jsonish_list(evaluation.get("peer_linkages")):
        if not isinstance(linkage, dict):
            continue
        level = str(linkage.get("linkage_level") or "none")
        if _linkage_rank(level) > _linkage_rank(best):
            best = level
    return best


def _relevant_profile_linkage_level_from_evaluation(
    profile_linkage_evaluation: dict[str, Any] | None,
    *,
    scope: str,
) -> str:
    evaluation = profile_linkage_evaluation or {}
    if not isinstance(evaluation, dict):
        return ""
    if scope == "skax":
        linkage = evaluation.get("skax_linkage") or {}
        if not isinstance(linkage, dict):
            return ""
        return str(linkage.get("linkage_level") or "")
    best = ""
    for linkage in _jsonish_list(evaluation.get("peer_linkages")):
        if not isinstance(linkage, dict):
            continue
        level = str(linkage.get("linkage_level") or "")
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
    if re.search(r"부족|확인되지|단정하기\s*어렵|사건\s*기반|낮춰", text or ""):
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
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if subject and profile_phrase:
        return (
            f"{fact} 이 사건은 피어 프로필의 {profile_phrase} 맥락이 "
            f"{_with_particle(subject, '과', '와')} 연결되는 신호로 해석할 수 있습니다."
        )
    return _event_based_analysis_summary(integrated_issue)


def _profile_linked_market_signal(integrated_issue: dict[str, Any]) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    if re.search(
        r"구축|센터|인프라|컴퓨팅|데이터\s*센터|GPU|반도체|서버|SPC|특수목적법인",
        _integrated_grounding_text(integrated_issue),
        flags=re.IGNORECASE,
    ):
        return (
            f"{subject or '현재 사건'}에서 구축 범위, 인프라 구성, 단계별 추진 일정이 "
            "함께 제시되어 대규모 인프라 사업의 비교 기준이 구체화되고 있습니다."
        )
    return _event_based_market_signal(integrated_issue)


def _profile_linked_impact_reason(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    subject = _issue_subject_phrase(integrated_issue)
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if subject and profile_phrase:
        return (
            f"{subject}이 확인되면서 피어 프로필의 {profile_phrase} 역량이 "
            "현재 사건의 구축 범위와 추진 구조에 연결되는지 관찰할 수 있습니다."
        )
    return _event_based_impact_reason(integrated_issue)


def _profile_linked_strategic_meanings(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> list[str]:
    fact = _primary_issue_fact(integrated_issue)
    subject = _issue_subject_phrase(integrated_issue)
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    meanings = [fact]
    if subject and profile_phrase:
        meanings.append(
            f"피어 프로필의 {profile_phrase} 맥락과 연결하면, 이번 사건은 "
            f"{subject}에서 필요한 구축 범위와 운영 구조를 확인하는 신호입니다."
        )
    meanings.append(_profile_linked_market_signal(integrated_issue))
    return _normalize_recommended_actions(meanings)[:3]


def _event_based_market_signal(integrated_issue: dict[str, Any]) -> str:
    subject = _issue_subject_phrase(integrated_issue)
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
    if fact:
        candidates.append(fact)
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
    issue_text = " ".join(
        str(integrated_issue.get(key) or "").strip()
        for key in ("main_event", "main_issue", "headline", "one_line_summary")
    )
    if subject := _extract_issue_subject_from_text(issue_text):
        return subject

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
                if text:
                    return text
        for item in main_event_facts:
            if subject := _extract_issue_subject_from_text(str(item.get("fact") or "")):
                return subject
        for value in _jsonish_list(intelligence.get("products_or_services")):
            text = re.sub(r"\s+", " ", str(value or "").strip(" ."))
            if text:
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
                if text:
                    return text
            if subject := _extract_issue_subject_from_text(str(item.get("fact") or "")):
                return subject

    return ""


def _fact_has_summary_role(item: dict[str, Any], role_name: str) -> bool:
    roles = {str(role or "") for role in _jsonish_list(item.get("summary_roles"))}
    role = str(item.get("summary_role") or "")
    return role_name in roles or role == role_name


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
        return (
            f"{fact_sentence} 피어 프로필에서는 {profile_phrase}가 "
            f"{_with_particle(subject, '과', '와')} 연결되는 배경으로 확인됩니다. "
            "따라서 이 신호는 역할 확장이나 "
            "성과를 단정하기보다, 해당 피어의 기존 사업 맥락이 현재 대형 과제와 만나는 "
            "관찰 지점으로 해석하는 것이 안전합니다."
        )
    return _event_based_peer_meaning(integrated_issue=integrated_issue, peer=peer)


def _profile_linked_capability_change(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    subject = _issue_subject_phrase(integrated_issue) or "현재 사건"
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if profile_phrase:
        return (
            f"확인된 변화는 역량 확장 자체가 아니라 {subject}의 구축 범위와 추진 구조가 "
            f"피어 프로필의 {profile_phrase} 맥락과 연결된다는 점입니다. 유사 사업에서는 "
            "구축 범위, 운영 체계, 단계별 일정이 함께 비교될 수 있습니다."
        )
    return _event_based_capability_change(integrated_issue)


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
    limit = 1 if scope == "skax" else 2
    return "·".join(names[:limit])


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
    return {
        str(term.get("normalized") or "")
        for term in _extract_ranked_terms(text, "output_text")
        if float(term.get("weight") or 0.0) > 0.0
    }


def _is_low_signal_content_token(token: str) -> bool:
    return _term_signal_weight(token, "output_text") <= 0.0


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
    return _term_signal_weight(token, "issue_fact") <= 0.0


def _shrink_profile(
    profile: Any,
    *,
    relevance_tokens: set[str] | None = None,
    profile_linkage: dict[str, Any] | None = None,
) -> dict[str, Any]:
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

    if isinstance(profile_linkage, dict) and profile_linkage:
        linkage_meta = {
            key: profile_linkage.get(key)
            for key in (
                "linkage_level",
                "business_novelty_status",
                "implication_mode",
                "reason",
                "matched_terms",
            )
            if profile_linkage.get(key) not in ({}, [], "", None)
        }
        if linkage_meta:
            out["machine_profile_linkage_hint"] = _compact_value(linkage_meta)
        matched_areas = _jsonish_list(profile_linkage.get("matched_business_areas"))
        if matched_areas:
            out["machine_matched_business_areas"] = [
                _compact_profile_item(item, include_evidence=False)
                for item in matched_areas[:3]
                if isinstance(item, dict)
            ]
        matched_caps = _string_list(profile_linkage.get("matched_capabilities"), max_items=5)
        if matched_caps:
            out["machine_matched_capabilities"] = matched_caps

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
        if key in out:
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
    issue_context = {
        "structured_terms": sorted(relevance_tokens),
        "source_noise_terms": set(),
        "evidence_text": "",
    }
    item_terms = _extract_ranked_terms(item_text, "profile_business_area", issue_context)
    score = 0.0
    for issue_token in relevance_tokens:
        issue_weight = _term_signal_weight(issue_token, "issue_fact", issue_context)
        if issue_weight <= 0.0:
            continue
        for item_term in item_terms:
            item_token = str(item_term.get("normalized") or "")
            item_weight = float(item_term.get("weight") or 0.0)
            if item_weight <= 0.0:
                continue
            if _tokens_semantically_close(issue_token, item_token):
                score += min(issue_weight, item_weight)
                break
    return int(round(score * 10))


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


def _two_section_fact_based_fallback(
    original: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    model: str,
    profile_linkage_evaluation: dict[str, Any] | None = None,
    action_artifact_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative, evidence-scoped fallback when LLM repair overclaims.

    This is intentionally generic: it uses only the current IntegratedIssue,
    profile linkage level, and action plan signals. It does not encode a specific
    article, company, or sector outcome.
    """

    fact_ids = list(_known_fact_ids(integrated_issue))[:5]
    profile_linkage_evaluation = profile_linkage_evaluation or {}
    action_artifact_plan = action_artifact_plan or {}
    peer_linkage_level = _relevant_profile_linkage_level_from_evaluation(
        profile_linkage_evaluation,
        scope="peer",
    )
    skax_linkage_level = _relevant_profile_linkage_level_from_evaluation(
        profile_linkage_evaluation,
        scope="skax",
    )
    has_peer_profile_link = peer_linkage_level in {"high", "medium"}
    peer = {
        "company_id": str(integrated_issue.get("main_company") or ""),
        "company_name_ko": _main_company_display(integrated_issue),
        "sourced_evidence_ids": fact_ids,
    }
    if has_peer_profile_link:
        analysis_summary = _profile_linked_analysis_summary(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        strategic_meaning = [
            _profile_linked_market_signal(integrated_issue),
            _profile_linked_impact_reason(
                integrated_issue=integrated_issue,
                profile_context=profile_context,
            ),
        ]
        market_signal = _profile_linked_market_signal(integrated_issue)
        impact_reason = _profile_linked_impact_reason(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        reason = _profile_linked_analysis_field(
            "reason",
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
        peer_meaning = _profile_linked_peer_meaning(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
            peer=peer,
        )
        capability_change = _profile_linked_capability_change(
            integrated_issue=integrated_issue,
            profile_context=profile_context,
        )
    else:
        analysis_summary = _event_based_analysis_summary(integrated_issue)
        primary_fact = _primary_issue_fact(integrated_issue)
        strategic_meaning = [
            item
            for item in _event_based_strategic_meaning_candidates(integrated_issue)
            if item != primary_fact
        ][:2]
        market_signal = _event_based_market_signal(integrated_issue)
        impact_reason = _event_based_impact_reason(integrated_issue)
        reason = _event_based_analysis_field("reason", integrated_issue=integrated_issue)
        peer_meaning = _event_based_peer_meaning(integrated_issue=integrated_issue, peer=peer)
        capability_change = _event_based_capability_change(integrated_issue)

    peer_meaning = _fallback_peer_meaning_without_summary_repeat(
        peer_meaning,
        integrated_issue=integrated_issue,
        profile_context=profile_context,
        has_peer_profile_link=has_peer_profile_link,
    )
    skax = _fallback_skax_implication(
        integrated_issue=integrated_issue,
        profile_context=profile_context,
        profile_linkage_level=skax_linkage_level,
        action_artifact_plan=action_artifact_plan,
    )
    profile_linkage_payload = _fallback_profile_linkage_payload(
        profile_linkage_evaluation,
        integrated_issue=integrated_issue,
    )
    skax_response_linkage_payload = _fallback_skax_response_linkage_payload(
        profile_linkage_evaluation,
        integrated_issue=integrated_issue,
        action_artifact_plan=action_artifact_plan,
    )
    result = {
        "is_valid_strategic_insight": True,
        "profile_linkage": profile_linkage_payload,
        "skax_response_linkage": skax_response_linkage_payload,
        "claim_strength": (
            "moderate"
            if (
                profile_linkage_payload.get("linkage_level") in {"high", "medium"}
                or skax_response_linkage_payload.get("response_mode") == "profile_based_action"
            )
            else "cautious"
        ),
        "grounding_summary": {
            "used_fact_ids": fact_ids,
            "used_profile_refs": _fallback_used_profile_refs(profile_linkage_evaluation),
            "ungrounded_claims_removed": [],
        },
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": analysis_summary,
            "strategic_meaning": _normalize_recommended_actions(strategic_meaning)[:3],
            "market_signal": market_signal,
            "impact_level": (original.get("analysis") or {}).get("impact_level") or "low",
            "impact_reason": impact_reason,
            "risk_or_opportunity": _choice(
                (original.get("analysis") or {}).get("risk_or_opportunity"),
                _RISK_OR_OPPORTUNITY,
                "neutral",
            ),
            "confidence": 0.65 if has_peer_profile_link else 0.55,
            "reason": reason,
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": peer["company_id"],
                "company_name_ko": peer["company_name_ko"],
                "peer_meaning": peer_meaning,
                "capability_change": capability_change,
                "sourced_evidence_ids": fact_ids,
            },
            "skax_implication": skax,
            "follow_up_questions": [],
            "watch_points": _fallback_watch_points(integrated_issue),
            "confidence": 0.62 if skax_linkage_level in {"high", "medium"} else 0.52,
            "evidence_label": "moderate" if has_peer_profile_link else "insufficient",
            "provenance": {
                "generator": "StrategicInsightAgent",
                "prompt_version": _PROMPT_VERSION,
                "model": model,
                "used_fact_ids": fact_ids,
                "used_context_layers": [
                    "integrated_issue_fact_fallback",
                    "profile_linkage_fallback",
                    "action_plan_fallback",
                ],
                "run_at": datetime.now(UTC).isoformat(),
            },
        },
    }
    return result


def _fallback_profile_linkage_payload(
    profile_linkage_evaluation: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> dict[str, Any]:
    company_id = (_companies_from_integrated_issue(integrated_issue) or [""])[0]
    linkage = _profile_linkage_for_company(
        profile_linkage_evaluation,
        company_id=company_id,
        scope="peer",
    )
    matched_areas = _fallback_matched_profile_areas(linkage)
    linkage_level = _choice(
        linkage.get("linkage_level"),
        {"high", "medium", "low", "none"},
        "none",
    )
    implication_mode = str(linkage.get("implication_mode") or "")
    if implication_mode == "profile_based":
        interpretation_strength = "profile_based"
    elif implication_mode == "new_business_signal":
        interpretation_strength = "event_based"
    elif linkage_level in {"high", "medium"}:
        interpretation_strength = "cautious_profile_based"
    else:
        interpretation_strength = "observation_only"
    return _normalize_llm_profile_linkage(
        {
            "peer_company": company_id or _main_company_display(integrated_issue),
            "profile_evidence_available": bool(matched_areas)
            or linkage_level in {"high", "medium"},
            "matched_profile_areas": matched_areas,
            "linkage_level": linkage_level,
            "business_novelty_status": linkage.get("business_novelty_status"),
            "allowed_interpretation_strength": interpretation_strength,
            "reason": str(linkage.get("reason") or "").strip(),
        }
    )


def _with_fallback_linkage_payloads(
    result: dict[str, Any],
    *,
    profile_linkage_evaluation: dict[str, Any],
    integrated_issue: dict[str, Any],
    action_artifact_plan: dict[str, Any],
) -> dict[str, Any]:
    out = dict(result or {})
    normalized_profile_linkage = _normalize_llm_profile_linkage(out.get("profile_linkage"))
    if not normalized_profile_linkage.get(
        "profile_evidence_available"
    ) and not normalized_profile_linkage.get("matched_profile_areas"):
        normalized_profile_linkage = _fallback_profile_linkage_payload(
            profile_linkage_evaluation,
            integrated_issue=integrated_issue,
        )
    normalized_skax_linkage = _normalize_skax_response_linkage(out.get("skax_response_linkage"))
    if not normalized_skax_linkage.get(
        "skax_profile_evidence_available"
    ) and not normalized_skax_linkage.get("matched_skax_areas"):
        normalized_skax_linkage = _fallback_skax_response_linkage_payload(
            profile_linkage_evaluation,
            integrated_issue=integrated_issue,
            action_artifact_plan=action_artifact_plan,
        )
    out["profile_linkage"] = normalized_profile_linkage
    out["skax_response_linkage"] = normalized_skax_linkage
    if not out.get("claim_strength"):
        out["claim_strength"] = (
            "moderate"
            if (
                normalized_profile_linkage.get("linkage_level") in {"high", "medium"}
                or normalized_skax_linkage.get("response_mode") == "profile_based_action"
            )
            else "cautious"
        )
    grounding_summary = _normalize_grounding_summary(
        out.get("grounding_summary"),
        integrated_issue=integrated_issue,
    )
    if not grounding_summary.get("used_profile_refs"):
        grounding_summary["used_profile_refs"] = _fallback_used_profile_refs(
            profile_linkage_evaluation
        )
    out["grounding_summary"] = grounding_summary
    return out


def _fallback_skax_response_linkage_payload(
    profile_linkage_evaluation: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    action_artifact_plan: dict[str, Any],
) -> dict[str, Any]:
    linkage = _profile_linkage_for_company(
        profile_linkage_evaluation,
        company_id="sk_ax",
        scope="skax",
    )
    matched_areas = _fallback_matched_skax_areas(linkage)
    linkage_level = _choice(
        linkage.get("linkage_level"),
        {"high", "medium", "low", "none"},
        "none",
    )
    response_mode = str(linkage.get("implication_mode") or "")
    if response_mode not in {
        "profile_based_action",
        "cautious_action",
        "generic_monitoring_action",
    }:
        if linkage_level in {"high", "medium"}:
            response_mode = "profile_based_action"
        elif linkage_level == "low":
            response_mode = "cautious_action"
        else:
            response_mode = "generic_monitoring_action"
    if not matched_areas and response_mode == "profile_based_action":
        response_mode = "cautious_action"
    issue_terms = sorted(_action_plan_issue_terms(action_artifact_plan))[:8]
    focus_terms = issue_terms or [_issue_subject_phrase(integrated_issue)]
    focus_terms = [term for term in focus_terms if term]
    return _normalize_skax_response_linkage(
        {
            "skax_profile_evidence_available": bool(matched_areas),
            "matched_skax_areas": matched_areas,
            "response_mode": response_mode,
            "response_focus": focus_terms[:5],
            "internal_checkpoints": _fallback_internal_checkpoints(
                focus_terms=focus_terms,
                action_artifact_plan=action_artifact_plan,
            ),
            "recommended_focus": _fallback_recommended_focus(
                focus_terms=focus_terms,
                linkage=linkage,
            ),
            "monitoring_points": _fallback_watch_points(integrated_issue),
            "reason": str(linkage.get("reason") or "").strip(),
        }
    )


def _fallback_matched_profile_areas(linkage: dict[str, Any]) -> list[dict[str, Any]]:
    areas = _jsonish_list(linkage.get("matched_business_areas"))
    capabilities = _string_list(linkage.get("matched_capabilities"), max_items=5)
    out: list[dict[str, Any]] = []
    for area in areas[:5]:
        if not isinstance(area, dict):
            continue
        name = str(area.get("name") or area.get("profile_area_name") or "").strip()
        business_line = str(area.get("business_line") or "").strip()
        raw_business_area = str(area.get("business_area") or "").strip()
        raw_specificity = str(area.get("specificity_level") or "").strip()
        business_area = raw_business_area or (name if raw_specificity == "business_area" else "")
        matched_capabilities = (
            _string_list(
                area.get("matched_capabilities") or area.get("capabilities"),
                max_items=8,
            )
            or capabilities[:3]
        )
        matched_products = _string_list(
            area.get("matched_products_or_services") or area.get("products_or_services"),
            max_items=8,
        )
        profile_capability = ", ".join(matched_capabilities[:3])
        reason = _fallback_area_reason(area, linkage)
        source_ref = _fallback_first_source_ref(area, linkage)
        specificity_level = _choice(
            area.get("specificity_level"),
            {
                "product_or_service",
                "core_capability",
                "business_area",
                "business_line",
                "profile_context",
            },
            (
                "product_or_service"
                if matched_products
                else "core_capability"
                if matched_capabilities
                else "business_area"
                if business_area
                else "business_line"
                if business_line
                else "profile_context"
            ),
        )
        if name or profile_capability or reason or source_ref:
            out.append(
                {
                    "business_line": business_line,
                    "business_area": business_area,
                    "profile_area_name": name,
                    "profile_capability": profile_capability,
                    "matched_capabilities": matched_capabilities,
                    "matched_products_or_services": matched_products,
                    "matched_issue_terms": _string_list(
                        area.get("matched_issue_terms") or area.get("matched_terms"),
                        max_items=12,
                    ),
                    "evidence_text": str(area.get("evidence_text") or "").strip(),
                    "why_relevant_to_issue": reason,
                    "profile_source_ref": source_ref,
                    "specificity_level": specificity_level,
                }
            )
    if not out:
        for capability in capabilities[:3]:
            out.append(
                {
                    "business_line": "",
                    "business_area": "",
                    "profile_area_name": capability,
                    "profile_capability": capability,
                    "matched_capabilities": [capability],
                    "matched_products_or_services": [],
                    "matched_issue_terms": _string_list(
                        linkage.get("matched_terms"),
                        max_items=12,
                    ),
                    "evidence_text": "",
                    "why_relevant_to_issue": str(linkage.get("reason") or "").strip(),
                    "profile_source_ref": _fallback_first_source_ref({}, linkage),
                    "specificity_level": "core_capability",
                }
            )
    return out[:5]


def _fallback_matched_skax_areas(linkage: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "business_line": item["business_line"],
            "business_area": item["business_area"],
            "profile_area_name": item["profile_area_name"],
            "matched_capabilities": item["matched_capabilities"],
            "matched_products_or_services": item["matched_products_or_services"],
            "matched_issue_terms": item["matched_issue_terms"],
            "evidence_text": item["evidence_text"],
            "why_relevant_to_issue": item["why_relevant_to_issue"],
            "profile_source_ref": item["profile_source_ref"],
            "specificity_level": item["specificity_level"],
        }
        for item in _fallback_matched_profile_areas(linkage)
    ]


def _fallback_area_reason(area: dict[str, Any], linkage: dict[str, Any]) -> str:
    matched_terms = _string_list(area.get("matched_terms"), max_items=6)
    if matched_terms:
        return "현재 이슈의 " + ", ".join(matched_terms[:4]) + " 신호와 연결됩니다."
    return str(linkage.get("reason") or "").strip()


def _fallback_first_source_ref(area: dict[str, Any], linkage: dict[str, Any]) -> str:
    refs = _jsonish_list(area.get("source_refs")) or _jsonish_list(
        linkage.get("matched_source_refs")
    )
    if not refs:
        return ""
    first = refs[0]
    if isinstance(first, dict):
        return str(first.get("source_ref") or first.get("id") or first.get("url") or "").strip()
    return str(first).strip()


def _fallback_internal_checkpoints(
    *,
    focus_terms: list[str],
    action_artifact_plan: dict[str, Any],
) -> list[str]:
    checkpoint = _checkpoint_hint_from_action_plan(action_artifact_plan)
    out = []
    for term in focus_terms[:3]:
        out.append(f"{term} 관련 {checkpoint}")
    if not out:
        out.append(checkpoint)
    return out


def _fallback_recommended_focus(
    *,
    focus_terms: list[str],
    linkage: dict[str, Any],
) -> list[str]:
    level = _choice(linkage.get("linkage_level"), {"high", "medium", "low", "none"}, "none")
    if focus_terms and level in {"high", "medium"}:
        return [f"{term}와 연결된 직접 수행 범위와 보완 필요 영역" for term in focus_terms[:3]]
    if focus_terms:
        return [f"{term} 관련 후속 근거 확인과 보수적 대응 범위 점검" for term in focus_terms[:3]]
    return ["후속 근거 확인과 대응 범위 점검"]


def _fallback_used_profile_refs(profile_linkage_evaluation: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for linkage in [
        *_jsonish_list(profile_linkage_evaluation.get("peer_linkages")),
        profile_linkage_evaluation.get("skax_linkage"),
    ]:
        if not isinstance(linkage, dict):
            continue
        refs.extend(_string_list(linkage.get("matched_source_refs"), max_items=5))
        for area in _jsonish_list(linkage.get("matched_business_areas")):
            if isinstance(area, dict):
                refs.extend(_string_list(area.get("source_refs"), max_items=5))
    return list(dict.fromkeys(refs))[:12]


def _fallback_peer_meaning_without_summary_repeat(
    current: str,
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    has_peer_profile_link: bool,
) -> str:
    subject = _issue_subject_phrase(integrated_issue) or "현재 사건"
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    if has_peer_profile_link and profile_phrase:
        return (
            f"피어 프로필에서는 {profile_phrase}가 "
            f"{_with_particle(subject, '과', '와')} 연결되는 배경으로 확인됩니다. "
            "따라서 이 신호는 역할 확장이나 성과를 단정하기보다, 기존 사업 맥락이 "
            "현재 대형 과제와 만나는 관찰 지점으로 해석하는 것이 안전합니다."
        )
    if current and _primary_issue_fact(integrated_issue).rstrip(".") not in current:
        return current
    return (
        f"{subject}은 피어사의 확정된 역할 변화보다 현재 사건의 대상 사업, "
        "추진 범위, 후속 확인 기준이 구체화된 관찰 신호로 보는 것이 안전합니다."
    )


def _fallback_skax_implication(
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    profile_linkage_level: str,
    action_artifact_plan: dict[str, Any],
) -> dict[str, Any]:
    subject = _issue_subject_phrase(integrated_issue) or _primary_issue_fact(integrated_issue)
    peer_profile = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    skax_profile = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="skax",
    )
    profile_comparison = _profile_comparison_phrase(
        peer_profile=peer_profile,
        skax_profile=skax_profile,
        linkage_level=profile_linkage_level,
    )
    actions = _fallback_internal_actions(
        subject=subject,
        profile_comparison=profile_comparison,
        action_artifact_plan=action_artifact_plan,
    )
    return {
        "why_important": (
            f"{_with_particle(subject, '은', '는')} 유사 고객군/유사 사업에서 "
            "피어 신호와 SK AX의 대응 가능 범위를 "
            "함께 비교해야 하는 사건입니다."
        ),
        "potential_impact": (
            f"유사 사업에서는 {subject}의 구축 범위, 운영 책임, 일정 조건, 검증 기준이 "
            f"함께 비교될 수 있으므로 SK AX는 {profile_comparison}을 기준으로 내부 "
            "대응 범위와 보완 항목을 점검해야 합니다."
        ),
        "opportunities": [],
        "threats": [],
        "recommended_actions": actions,
        "business_line_mapping": _safe_business_line_mapping(
            profile_context,
            integrated_issue=integrated_issue,
        ),
    }


def _profile_comparison_phrase(
    *,
    peer_profile: str,
    skax_profile: str,
    linkage_level: str,
) -> str:
    if linkage_level in {"high", "medium"} and peer_profile and skax_profile:
        return f"피어의 {peer_profile} 접점과 SK AX의 {skax_profile} 접점"
    if skax_profile:
        return f"SK AX의 {skax_profile} 접점"
    if peer_profile:
        return f"피어의 {peer_profile} 접점과 SK AX의 유사 사업 대응 범위"
    return "현재 사건에서 확인된 대상 사업과 SK AX의 유사 사업 대응 범위"


def _fallback_internal_actions(
    *,
    subject: str,
    profile_comparison: str,
    action_artifact_plan: dict[str, Any],
) -> list[str]:
    issue_term = _compact_issue_term(subject)
    checkpoint_hint = _checkpoint_hint_from_action_plan(action_artifact_plan)
    return [
        (
            f"SK AX는 {_with_particle(issue_term, '과', '와')} 유사한 사업에서 "
            f"{profile_comparison}을 비교하고, "
            "직접 수행할 범위와 외부 보완이 필요한 범위를 운영 책임 기준으로 점검해야 합니다."
        ),
        (
            f"SK AX는 {issue_term} 대응 시 "
            f"{_with_particle(checkpoint_hint, '을', '를')} 내부 확인 기준으로 구조화하고, "
            "후속 사업자 선정·협약·서비스 개시 신호를 모니터링해야 합니다."
        ),
    ]


def _compact_issue_term(subject: str) -> str:
    text = re.sub(r"\s+", " ", str(subject or "").strip(" ."))
    return text if len(text) <= 80 else f"{text[:77].rstrip()}..."


def _with_particle(text: str, consonant_particle: str, vowel_particle: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    return f"{value}{consonant_particle if _has_final_consonant(value[-1]) else vowel_particle}"


def _has_final_consonant(char: str) -> bool:
    value = str(char or "")[:1]
    if not value:
        return False
    code = ord(value)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def _checkpoint_hint_from_action_plan(action_artifact_plan: dict[str, Any]) -> str:
    issue_terms = _action_plan_issue_terms(action_artifact_plan)
    if any(re.search(r"GPU|컴퓨팅|인프라|센터|서버|클러스터", term, re.I) for term in issue_terms):
        return "구축 범위, 용량 기준, 장애 대응, 보안·권한 통제"
    if any(re.search(r"시스템|전환|현대화|단말|플랫폼", term, re.I) for term in issue_terms):
        return "전환 범위, 업무 영향도, 일정 조건, 장애 대응"
    return "범위, 책임, 일정 조건, 검증 기준, 리스크"


def _fallback_watch_points(integrated_issue: dict[str, Any]) -> list[str]:
    subject = _issue_subject_phrase(integrated_issue) or "현재 사업"
    return [
        f"{subject}의 후속 협약, 구축 완료, 서비스 개시 일정이 구체화되는지 확인합니다.",
        "피어사의 수행 범위, 운영 책임, 추가 참여 구조가 원문 근거로 확인되는지 모니터링합니다.",
    ]


def _safe_business_line_mapping(
    profile_context: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
) -> list[str]:
    issue_tokens = _issue_relevance_tokens(integrated_issue)
    candidates = _business_line_candidate_details(
        profile_context,
        integrated_issue=integrated_issue,
    )
    selected = []
    for item in candidates:
        name = str(item.get("name") or "").strip()
        if name and (_content_tokens(name) & issue_tokens):
            selected.append(name)
    return selected[:2]


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
