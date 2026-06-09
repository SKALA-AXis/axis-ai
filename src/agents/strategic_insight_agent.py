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
_PROMPT_VERSION = "strategic-insight-v1.58-scope-impact-guard"
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


ACTION_REPAIR_SYSTEM_PROMPT = """\
당신은 SK AX 대응방향만 다시 쓰는 repair agent입니다.
새 사실을 만들지 말고 recommended_actions 만 JSON 으로 출력합니다.
각 action 은 피어 신호, SK AX 프로필 접점, 바꿀 문서·표·계획·검증 방식,
고객이 확인할 기준을 함께 보여야 합니다.
"""


ACTION_REPAIR_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## ProfileContext
{profile_json}

## 현재 skax_implication
{skax_json}

## 규칙
1. recommended_actions 는 가능하면 2~3개입니다. 근거가 부족할 때만 1개로 줄입니다.
   서로 다른 실행 장면(제안서, PoC, 레퍼런스, 운영 전환 계획)을 나눠 씁니다.
2. 각 action 은 2~3문장입니다:
   피어 신호 → 유사 고객군/유사 사업에서 비교할 기준 → SK AX가 바꿀 문서·표·계획·검증 방식 →
   바꾼 뒤 고객이 확인할 수 있는 기준.
3. SK AX 프로필과 연결되지 않으면 일반론을 만들지 말고 action 수를 줄입니다.
4. 타깃 피어의 특정 프로젝트에 직접 제안하는 문장으로 쓰지 말고
   유사 고객군/유사 사업 대응으로 씁니다.
5. "제안 산출물"이라고 쓰지 말고 전환 범위표, 업무 영향도 정리, 운영 전환 계획,
   PoC 검증표, 레퍼런스 비교표 등 실제 바뀌는 이름을 씁니다.
6. 특정 기술명·사업영역·솔루션명은 현재 입력 컨텍스트에 근거가 있을 때만 씁니다.
   근거 출처는 IntegratedIssue, classification, 관련 peer/skax ProfileContext,
   현재 이슈와 매칭된 AnalysisContext 입니다.
   근거가 약하면 현재 사건의 대상 시스템·전환 범위·업무 영향도·운영 안정성·검증 기준처럼
   더 상위의 안전한 표현으로 낮춥니다.
7. business_line_mapping 이름을 솔루션명처럼 그대로 쓰지 않습니다.
   예: business line 이 "클라우드&AI"여도 현재 사건과 직접 연결되지 않으면
   "클라우드 및 AI 솔루션"이라고 쓰지 않습니다.
8. "성공 사례"는 ProfileContext 에 실제 사례 근거가 있을 때만 씁니다.
   근거가 없으면 "레퍼런스 자료"나 "레퍼런스 비교표"라고 씁니다.
9. 근거 없는 강화, 경쟁력, 차별화, 고객 신뢰 같은 결과 표현으로 끝내지 않습니다.

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

## business_line_mapping 후보
{business_lines_json}

## 수정 대상 결과
{result_json}

## 수정 대상 위반
{violations_json}

## 규칙
1. schema 는 유지하고 문장만 고칩니다.
2. analysis 는 현재 사실 → 대상 업무/시스템의 전환·검증·운영 성격 → 시장 신호를 보여야 합니다.
3. peer_implication 은 현재 사실 → 피어 프로필 접점 → 유사 사업에서 관찰할 기준을 보여야 합니다.
4. skax_implication 은 피어 신호 → 유사 고객군/사업의 비교 기준 →
   SK AX가 바꿀 문서·표·계획·검증 방식을 보여야 합니다.
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
5. recommended_actions 는 유사 고객군/유사 사업 대응입니다. SK AX가 바꿀 문서·표·계획·검증 방식과
   고객이 확인할 기준을 함께 씁니다.
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
   - recommended_actions: 유사 고객군/유사 사업에서 비교할 기준 → SK AX가 바꿀
     전환 범위표/업무 영향도 정리/운영 전환 계획/PoC 검증표/레퍼런스 비교표

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
4. 최종 사용자에게 보여줄 논리는 "시사점"과 "SK AX 대응 방향" 두 묶음뿐입니다.
   analysis 와 peer_implication 은 같은 "시사점" 묶음을 구성하는 내부 필드이고,
   skax_implication 은 "SK AX 대응 방향" 묶음을 구성하는 내부 필드입니다.
5. 좋은 결과는 짧은 결론이 아니라
   "근거 사실 → 왜 그렇게 해석되는지 → 어떤 대응이 필요한지"가 보입니다.
6. JSON 외 텍스트를 출력하지 마세요.
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

## Context availability
{context_availability_json}

## SK AX business_line_mapping 후보
{business_lines_json}

## 역할 해석 모드
{role_mode_instructions}

## 생성 순서
1. Fact grounding: 확정 사실, 관계 수준, 수치/날짜, evidence_ids 를 먼저 확인합니다.
   IntegratedIssue 의 표시 요약보다 StrategicEvidencePack 의 fact_basis/evidence_texts/
   representative_sources 를 우선합니다. 카드뉴스용 짧은 요약만 보고 시사점을 만들지 않습니다.
2. 사실 기반 해석: 계약 규모, 기간, 고객명, 사업명은 1차 해석 재료로만 쓰고
   이 단계의 결론을 최종 시사점으로 끝내지 않습니다. 아래 peer_signal 까지 연결합니다.
3. peer_role_in_issue: 피어가 공급자/수행사/운영자/고객/계약 상대방 중 무엇으로만
   확인되는지 정합니다. 불명확하면 더 약한 표현을 씁니다.
4. related_peer_profile_context: 현재 사건의 사업명, 대상 시스템, 고객군, 섹터와 맞는
   피어 프로필 사업영역만 고릅니다. 맞지 않는 프로필 조각은 쓰지 않습니다.
5. 피어 프로필 기반 시사점 생성(peer_signal):
   confirmed_facts + peer_role_in_issue + related_peer_profile_context +
   최근 peer/sector context 를 연결해 피어 관점 시사점을 만듭니다.
   시사점에는 피어사의 기존 사업영역/역량과 현재 사건의 접점이 보여야 합니다.
   반드시 "피어사가 원래 어떤 역량/사업영역을 갖고 있었는지 → 이번 사건이 그 역량과
   어떻게 연결되는지 → 그래서 피어사 관점의 사업적 의미가 무엇인지" 순서로 판단합니다.
   profile/recent context 접점이 없으면 사건 기반 1차 해석으로 낮춰 confidence 를 낮춥니다.
   profile_context 나 recent context 가 없거나 현재 사건과 맞지 않으면
   프로필 기반으로 쓰지 않습니다.
   피어 프로필에 identity 필드만 있으면 사업영역/역량명을 추측하지 말고 프로필 접점 부족으로 둡니다.
6. SK AX 프로필 기반 대응 생성(skax_fit_gap):
   peer_signal + SK AX 관련 프로필 역량 + SK AX가 보완할 산출물/운영 방식 차이를 정합니다.
   SK AX profile_context 의 관련 사업영역/역량과 연결되지 않은 대응은 일반론입니다.
   SK AX 프로필의 사업영역명이 현재 사건의 고객군/대상 업무/기술 기능과 직접 맞지 않으면
   그 사업영역명을 문장에 쓰지 말고, 더 상위의 기업용 AI 도입/운영 검증/전사 확산 대응으로 씁니다.
   대응에는 제안서, PoC, 레퍼런스 비교표, 운영 계획 중 입력 맥락에 맞는 산출물을 씁니다.
   현재 상태 → 왜 바꿔야 하는가 → 무엇을 바꿔야 하는가 → 바꾸면 무엇이 달라지는가
   이 흐름으로 씁니다.
7. final output: 위 중간 판단은 출력하지 말고, schema 필드에 자연어로 반영합니다.
   단, 작성 관점은 반드시 두 묶음입니다:
   - 시사점: analysis_summary, strategic_meaning, market_signal, peer_meaning,
     capability_change 가 서로 중복 없이 하나의 논지로 이어져야 합니다.
   - SK AX 대응 방향: why_important, potential_impact, recommended_actions 가
     하나의 대응 논리로 이어져야 합니다.

## 필드 기준
- analysis_summary: "시사점"의 대표 문장입니다. 현재 사건 요약을 반복하지 말고
  피어/시장 의미를 1문장으로 씁니다. 이 문장은 사용자가 보는 "시사점" 첫 문장처럼
  읽혀야 합니다.
- strategic_meaning: 2~3개. 사실 반복이 아니라 "사실이 의미하는 피어/시장 변화"를 씁니다.
  "영역 확장", "긍정적인 영향"처럼 방향만 말하지 말고, 현재 사건에서 확인된
  대상 시스템/인프라 구성/운영 구조/추진 방식/비교 기준 중 무엇이 바뀌는지 씁니다.
  기사 제목이나 요약에 "가속화", "확장"이 있더라도 그대로 반복하지 말고
  "그룹 계열사 전반 적용", "외부 기업 고객 대상 맞춤형 AI 구축 서비스 제공 계획",
  "개발·문서 처리·협업 업무 적용"처럼 근거 문장의 구체 대상으로 풀어 씁니다.
- market_signal: 한 사건으로 수요 증가를 단정하지 말고 현재 사건에서 확인된 수요 신호,
  적용 범위, 비교 기준 변화를 씁니다. analysis_summary/strategic_meaning 과 같은 말을
  반복하지 말고, 시사점에 추가되는 관찰 기준 1개만 씁니다.
- 시장 범위/표현 강도: 원문이 국내 정부 사업이면 국가 단위/공공 대형 인프라 수준으로 씁니다.
  글로벌, 공공+민간, 전 산업, 사업 확장, 긍정적 영향은 현재 사건 또는 관련 프로필 근거가
  그 범위를 뒷받침할 때만 씁니다. 근거가 약하면 관찰 신호/연결 사례/레퍼런스 가능성으로 낮춥니다.
- peer_meaning: 별도 노출 섹션이 아니라 "시사점" 묶음의 내부 근거입니다.
  2문장 이상 가능. 현재 사실 → 피어 역할 → 관련 프로필/최근 흐름 접점을 설명합니다.
  관련 피어 프로필이 있으면 프로필의 business_area/core_capability/recent_direction 중
  현재 사건과 맞는 표현을 최소 1개 이상 자연어로 연결합니다.
  단순히 "입지 강화", "영역 확장"으로 끝내지 말고, 관련 프로필의 구체 사업영역/역량명과
  현재 사건의 대상 사업·인프라·고객군이 어떻게 만나는지 씁니다.
- capability_change: 별도 노출 섹션이 아니라 "시사점" 묶음의 내부 근거입니다.
  직접 확인되는 사업 범위, 고객군, 대상 시스템, 적용 영역만 씁니다.
  "역량 강화/영역 확장 예상"으로 끝내지 말고, 어떤 고객군·대상 시스템·운영 범위·추진 구조와
  연결되는지 적습니다.
  프로필 근거가 있으면 "기존 역량이 이번 사건에서 어떤 적용 장면/운영 범위/고객군과
  만나는지"를 설명하고, 프로필 근거가 없으면 역량 변화로 단정하지 않습니다.
- why_important: "SK AX 대응 방향"의 도입 문장입니다. 피어/시장 신호가 SK AX의 어떤
  사업영역/역량과 비교되는지 씁니다. 현재 사건이 금융/공공/제조/인프라 사건이 아니면
  SK AX 프로필에 해당 단어가 있더라도 쓰지 않습니다.
- potential_impact: 고객이 무엇을 비교하게 되는지, 기존 설명으로 무엇이 부족한지,
  SK AX가 어떤 구조로 바꾸면 무엇을 확인시킬 수 있는지 2~3문장으로 씁니다.
- recommended_actions: 1~3개. 각 항목은 길어도 됩니다.
  현재 피어 신호 → 해당 사업/운영 의미 → SK AX가 바꿀 구체 문서·표·계획·검증 방식 →
  바꾼 뒤 고객이 확인할 수 있는 기준을 연결합니다.
  각 action 은 StrategicEvidencePack 에서 확인되는 적용 범위, 대상 업무, 기술/서비스 기능,
  고객 확장 신호 중 최소 1개를 포함해야 합니다. 근거가 없으면 상위 표현으로 낮춥니다.
- business_line_mapping: 입력 후보 name 중 실제 관련 있는 항목만 0~3개 선택합니다.
- sourced_evidence_ids / used_fact_ids: 입력에 존재하는 fact_id 만 사용합니다.

## 금지
- 시사점 요약, 전략적 의미, 시장 신호, 피어 시사점, SK AX 관점처럼
  사용자 노출용 섹션을 여러 개로 나누는 문체
- analysis/peer_implication/skax_implication 필드명을 사용자 노출 제목처럼 해석하는 문체
- 현재 사건 근거에 없는 SK AX 사업영역을 끌어오는 문장.
  예: 현재 사건이 금융/인프라가 아닌데 "금융 인프라 혁신"이라고 쓰는 것
- 같은 의미를 analysis_summary, strategic_meaning, market_signal, peer_meaning,
  capability_change 에 반복하는 문장
- 근거 없는 기술적 우위, 선점, 격차, 경쟁 심화, 점유율 확대, 성과 예측
- 근거 없는 경쟁력 강화, 경쟁력에 긍정적인 영향, 효율성 향상, 가속화,
  수요 증가, 외부 확장 같은 효과성/방향성 결론
- "명확한 기회", "중요한 단계", "사업 확장 가능성", "시장 확장",
  "AI 전환 가속화"처럼 평가만 있고 비교 기준이 없는 문장
- 계약 상대방을 공급자/수행사로 바꾸는 표현
- ProfileContext 에 없는 사업영역/역량명을 모델 일반 지식으로 생성
- StrategicEvidencePack 에 없는 기술 기능, 계약 범위, 고객 확장 계획을 새로 생성
- SK AX 프로필과 연결되지 않은 대응방향
- 현재 상태, 변경 이유, 변경 내용, 기대효과가 없는 대응방향
- 출력 schema 예시 문구 복사

## 출력
아래 JSON schema 를 그대로 지켜 출력합니다. 설명 텍스트나 markdown 은 출력하지 마세요.
모든 자연어 문자열 값은 한국어로 작성하세요.
company_id, fact_id, enum 값, business_line_mapping 후보명처럼 입력에서 정해진 식별자만
원문 값을 유지합니다. analysis_summary, strategic_meaning, market_signal, impact_reason,
peer_meaning, capability_change, why_important, potential_impact, recommended_actions,
follow_up_questions, watch_points 는 반드시 한국어 문장이어야 합니다.
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

## classification
{classification_json}

## ProfileContext
{profile_json}

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
2. 사용자에게 보여줄 최종 논리는 "시사점"과 "SK AX 대응 방향" 두 묶음입니다.
   analysis 와 peer_implication 은 같은 시사점 묶음으로 읽혀야 하고,
   skax_implication 은 대응 방향 묶음으로 읽혀야 합니다.
   시사점 요약/전략적 의미/시장 신호/피어 시사점/SK AX 관점처럼
   노출 섹션을 여러 개로 쪼갠 듯한 반복 문장은 고칩니다.
3. 시사점이 요약 반복이면 고칩니다. 현재 사실, 피어 프로필 접점,
   최근 peer/sector context 접점이 보여야 합니다.
   profile_context 나 recent context 가 없거나 현재 사건과 맞는 접점이 없으면
   프로필 기반 결론처럼 쓰지 말고 confidence/evidence_label 을 낮춥니다.
   피어 프로필이 identity 필드뿐이면 모델 일반 지식으로 사업영역/역량명을 만들지 않습니다.
   관련 피어 프로필이 있으면 기존 사업영역/역량 → 현재 사건 접점 → 사업적 의미가
   보여야 합니다. 이 연결이 없으면 요약 반복으로 보고 고칩니다.
   "영역 확장", "긍정적 영향"처럼 방향만 말하는 문장은 현재 사건에서 확인된
   대상 시스템, 인프라 구성, 운영 구조, 추진 방식, 비교 기준으로 구체화합니다.
4. 피어가 계약 상대방/고객 슬롯이면 피어가 제공·수행·지원·운영했다고 쓰지 않습니다.
   계약 상대방으로 확인된 사업 범위, 계약 기간, 대상 시스템, 프로필 사업영역 접점으로 낮춥니다.
5. peer_meaning 은 2문장입니다. 현재 사실과 피어 프로필 접점이 어떤 유사 사업 비교 기준을
   보여주는지까지 설명해야 합니다.
6. 대응방향은 현재 피어/시장 신호, SK AX 프로필 또는 business_line 후보,
   SK AX가 바꿀 제안서/PoC/레퍼런스/운영 모델 구조가 모두 연결되어야 합니다.
   SK AX 프로필과 연결되지 않은 대응방향은 수정하거나 invalid 로 낮춥니다.
   현재 사건이 금융/인프라/공공/제조 등 특정 고객군·대상 업무와 직접 관련되지 않으면
   SK AX 프로필의 해당 사업영역명을 문장에 끌어오지 않습니다.
7. recommended_actions 는 현재 상태, 변경 이유, 변경 내용, 기대효과를 포함해야 합니다.
   "제안 산출물"이라고 쓰지 말고, 어떤 문서·표·계획·검증 방식인지 적습니다.
8. 기술명+솔루션 표현은 IntegratedIssue에 해당 기술명이 직접 있을 때만 씁니다.
   business_line 후보만 보고 "클라우드 및 AI 솔루션"처럼 솔루션명을 만들지 않습니다.
9. 성공 사례, 구축 경험, 운영 역량은 ProfileContext에 실제 근거가 있을 때만 씁니다.
   근거가 없으면 레퍼런스 자료, 비교표, 운영 전환 기준처럼 산출물 표현으로 낮춥니다.
10. 시장 범위와 표현 강도는 근거 범위를 넘지 않습니다.
   글로벌/해외, 공공+민간 양쪽, 전 산업, 사업영역 확장, 입지 강화,
   긍정적 영향 같은 표현은 IntegratedIssue 또는 관련 프로필에 그 범위를
   뒷받침하는 근거가 있을 때만 씁니다. 근거가 약하면 관찰 신호,
   연결 사례, 검증 계기, 레퍼런스 가능성으로 낮춥니다.
11. 근거 없는 우위/선점/점유율/경쟁 심화/성과 예측은 제거합니다.
12. 같은 의미가 analysis_summary, strategic_meaning, market_signal, peer_meaning,
    capability_change 에 반복되면 하나의 시사점 논리로 압축합니다.

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

## business_line_mapping 후보
{business_lines_json}

## 검증 실패 사유
{violations_json}

## 수정 대상 결과
{result_json}

## repair 기준
1. 없는 수치, 없는 관계, 근거 없는 역할 단정을 제거합니다.
2. analysis 와 peer_implication 은 같은 "시사점" 묶음, skax_implication 은
   "SK AX 대응 방향" 묶음입니다. 필드별 제목을 여러 개 붙인 듯한 반복 문장은 줄이고,
   두 묶음의 논리만 남깁니다.
3. 피어가 계약 상대방/고객 슬롯이면 제공·수행·지원·운영 같은 공급자 행동을 제거하고,
   계약 상대방으로 확인된 사업 범위, 기간, 대상 시스템, 프로필 사업영역 접점으로 고칩니다.
4. 공급사 매출 비율은 계약 규모 참고 근거로만 쓰고 피어사의 성과로 쓰지 않습니다.
5. 시사점은 현재 사건과 피어 프로필 접점으로 고칩니다. 접점이 없으면 사건 기반 해석으로 낮춥니다.
   ProfileContext 에 없는 피어 사업영역/역량명은 제거합니다.
6. SK AX 대응은 skax_profile/business_line 후보와 연결합니다.
   단, 현재 사건과 직접 맞지 않는 SK AX 사업영역명은 문장에 쓰지 않습니다.
7. recommended_actions 는 현재 상태, 변경 이유, 변경 내용, 기대효과가 보이게 고칩니다.
   "제안 산출물"이라고 쓰지 말고, 제안서의 전환 범위표, 업무 영향도 정리,
   운영 전환 계획, PoC 검증표, 레퍼런스 비교표처럼 실제 바뀌는 문서·표·계획·검증 방식을 씁니다.
8. 기술명+솔루션 표현은 IntegratedIssue에 해당 기술명이 직접 있을 때만 씁니다.
   business_line 후보만으로 "클라우드 및 AI 솔루션" 같은 표현을 만들지 않습니다.
9. 성공 사례, 구축 경험, 운영 역량은 ProfileContext에 실제 사례 근거가 있을 때만 씁니다.
10. 같은 의미가 여러 필드에 반복되면 analysis_summary 는 시사점 대표 문장,
    strategic_meaning/market_signal/peer_meaning/capability_change 는 서로 다른 근거·비교 기준,
    skax_implication 은 대응 방향으로 정리합니다.
11. 검증 실패 사유에 산업/도메인 용어 위반이 있으면 해당 단어를 제거합니다.
    예: 현재 사건 근거에 인프라가 없으면 "인프라 구성", "대규모 인프라 사업",
    "금융 인프라 혁신"을 쓰지 말고 "전사 적용 범위", "업무 적용 범위",
    "운영 책임", "보안·권한 검증 기준"으로 낮춥니다.
12. 반복 위반이 있으면 analysis_summary/strategic_meaning/market_signal/peer_meaning/
    capability_change 전체를 합쳐 2~3개의 서로 다른 시사점으로만 정리합니다.

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
                fallback = _two_section_fact_based_fallback(
                    reviewed,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
                    model=self.model,
                )
                fallback_violations = _quality_gate_violations(
                    fallback,
                    integrated_issue=integrated_issue,
                    profile_context=profile_dict,
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
        )
        fallback_violations = _quality_gate_violations(
            fallback,
            integrated_issue=integrated_issue,
            profile_context=profile_context,
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
            fallback = _two_section_fact_based_fallback(
                current,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
                model=self.model,
            )
            fallback_violations = _quality_gate_violations(
                fallback,
                integrated_issue=integrated_issue,
                profile_context=profile_context,
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
            guarded = _minimal_quality_guard(
                result,
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
            "- SK AX 대응방향은 타깃 피어 프로젝트에 직접 제안하는 문장이 아니라 "
            "유사 고객군/유사 사업의 산출물 재구성으로 쓰세요.",
            "- SK AX 대응방향은 '유사 사업에서 고객이 비교할 기준 → SK AX가 바꿀 "
            "구체 문서·표·계획·검증 방식' 순서로 2~3문장 작성하세요.",
            "- '제안 산출물'이라는 말은 쓰지 말고, 전환 범위표, 업무 영향도 정리, "
            "운영 전환 계획, PoC 검증표, 레퍼런스 비교표처럼 실제 바뀌는 이름을 쓰세요.",
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
                "skax_actions_require_current_signal_plus_skax_profile_plus_specific"
                "_proposal_poc_reference_or_operating_model_change"
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


def _unsupported_skax_profile_term_violation(
    text: str,
    *,
    label: str,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
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
    unsupported = sorted((output_tokens & skax_terms) - issue_tokens)
    if not unsupported:
        return ""
    return (
        "현재 사건 근거와 직접 맞지 않는 SK AX 프로필 용어를 사용했습니다: "
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
    domain_terms = {
        "금융",
        "제조",
        "공공",
        "물류",
        "유통",
        "통신",
        "의료",
        "헬스케어",
        "인프라",
    }
    evidence_tokens = _content_tokens(_integrated_grounding_text(integrated_issue))
    output_tokens = _content_tokens(text)
    unsupported = sorted((output_tokens & domain_terms) - evidence_tokens)
    if not unsupported:
        return ""
    return (
        "현재 사건 근거에 없는 산업/도메인 용어를 SK AX 대응 방향에 사용했습니다: "
        f"{', '.join(unsupported[:3])}. 현재 사건의 대상 업무/적용 범위/"
        "검증 기준으로 낮춰야 합니다."
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
    generic = {
        "이번",
        "계약",
        "통해",
        "기반",
        "계획",
        "추진",
        "제공",
        "사업",
        "시장",
        "고객",
        "외부",
        "그룹",
        "계열사",
        "전반",
        "혁신",
        "전환",
        "기반으로",
        "있습니다",
        "합니다",
    }
    return {token for token in _content_tokens(text) if token not in generic and len(token) >= 3}


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
    if re.search(r"클라우드|AI|인공지능|에이아이", text, flags=re.IGNORECASE) and not re.search(
        r"클라우드|AI|인공지능|에이아이",
        evidence_text,
        flags=re.IGNORECASE,
    ):
        return (
            "현재 사건 근거에 없는 기술명을 대응방향에 사용했습니다. 대상 시스템, 전환 범위, "
            "업무 영향도, 운영 전환 계획 중심으로 낮춰야 합니다."
        )
    if re.search(r"성공|수주에\s*영향|신뢰성", text) and not _profile_has_execution_case(
        profile_context,
        integrated_issue=integrated_issue,
    ):
        return (
            "ProfileContext에 실행 사례 근거가 없는데 성공/수주 영향/신뢰성을 사용했습니다. "
            "레퍼런스 자료/비교표처럼 검증 가능한 산출물 표현으로 낮춰야 합니다."
        )
    if re.search(r"솔루션|성능.{0,12}(강조|입증|검증|확인)|검증된\s*성능", text):
        return (
            "대응방향이 솔루션/성능 강조 같은 일반 표현에 머물렀습니다. "
            "현재 사건의 전환 범위, 업무 영향도, 운영 전환 계획, PoC 검증표처럼 "
            "고객이 확인할 산출물 기준으로 낮춰야 합니다."
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
    vague_action_pattern = (
        r"강화|"
        r"제안서.{0,16}강화|"
        r"PoC.{0,16}강화|"
        r"기준.{0,16}강화|"
        r"경쟁력[을를이가\s]*(유지|높|강화)|"
        r"차별화된\s*기능|"
        r"제안서.{0,16}준비"
        r"|역량.{0,20}(명확히|보여|강조)"
        r"|방안\s*마련"
        r"|비교\s*기준을\s*제공"
        r"|구체적인\s*가치"
        r"|성능\s*기준을\s*충족"
        r"|고객.{0,10}기대"
        r"|효율성[을를\s]*입증"
        r"|고객.{0,10}신뢰[를을\s]*(확보|구축)"
        r"|사업\s*확장\s*가능성"
        r"|가능성[을를\s]*모색"
        r"|분석하여"
    )
    if re.search(vague_action_pattern, text):
        return (
            "대응방향이 준비/강화/경쟁력 같은 추상 표현에 머물렀습니다. "
            "피어 신호, 관련 판단 기준, SK AX가 재구성할 산출물/운영 방식을 함께 써야 합니다."
        )
    return ""


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
    if "솔루션" in text_value and not re.search(
        r"제안서|PoC|검증표|전환\s*범위|운영\s*전환|업무\s*영향도|레퍼런스",
        text_value,
    ):
        return (
            "SK AX 영향/대응을 일반 솔루션 표현으로 썼습니다. 현재 사건에서 확인된 "
            "전환 범위, 업무 영향도, 운영 전환 계획, 검증 기준 중심으로 낮춰야 합니다."
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
            "사용했습니다. 레퍼런스 자료/비교표/운영 전환 기준처럼 검증 가능한 표현으로 "
            "낮춰야 합니다."
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

    if _has_effectiveness_claim(value):
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
        "AI",
        "ax",
        "AX",
        "ax는",
        "AX는",
        "id",
        "name",
        "있습니다",
        "합니다",
        "됩니다",
        "위해",
        "통해",
        "따라서",
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
        "제안서",
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
        ):
            continue
        safe_actions.append(action)

    if len(safe_actions) < 2:
        safe_actions = _event_based_recommended_actions(integrated_issue)
    skax["recommended_actions"] = safe_actions[:3]
    implication["skax_implication"] = skax
    out["implication"] = implication
    return out


def _event_based_recommended_actions(integrated_issue: dict[str, Any]) -> list[str]:
    subject = _issue_subject_phrase(integrated_issue) or "확인된 핵심 사업 과제"
    subject_text = f"{subject} {_integrated_grounding_text(integrated_issue)}"
    if re.search(
        r"구축|센터|인프라|컴퓨팅|데이터\s*센터|GPU|반도체|서버|SPC|특수목적법인",
        subject_text,
        flags=re.IGNORECASE,
    ):
        return [
            (
                f"유사 사업에서는 {subject}의 구축 범위와 인프라 구성, 단계별 도입 일정이 "
                "먼저 비교될 수 있습니다. SK AX는 제안서에서 제공 범위, 운영 책임, "
                "확장 계획을 분리해 제시해야 합니다. 그래야 고객이 단순 구축 가능성보다 "
                "실제 운영 조건과 사업 범위를 판단할 수 있습니다."
            ),
            (
                f"PoC나 사전 검증은 {subject}의 기능 설명에 그치지 않고 운영 단계에서 "
                "확인할 기준을 정의해야 합니다. 처리 용량, 운영 안정성, 장애 대응, "
                "보안·권한 통제 조건을 검증표로 제시하면 고객이 구축 이후의 운영 가능성을 "
                "더 구체적으로 비교할 수 있습니다."
            ),
            (
                "레퍼런스 자료는 구축 사실 나열보다 유사 사업에서 어떤 구축 범위, 운영 체계, "
                "확장 일정을 관리했는지 보여주는 구조로 재구성할 필요가 있습니다. 이를 통해 "
                "SK AX는 대규모 인프라 사업을 안정적으로 설계·운영할 수 있는 파트너라는 점을 "
                "설명할 수 있습니다."
            ),
        ]
    if re.search(r"전환|현대화|개편|마이그레이션|웹단말", subject_text, flags=re.IGNORECASE):
        return [
            (
                f"유사 사업에서는 {subject}의 대상 시스템과 계약 범위가 먼저 비교될 수 있습니다. "
                "SK AX는 제안서에서 기존 시스템 전환 범위, 업무 영향도, 일정 조건을 분리해 "
                "제시해야 합니다. 그래야 고객이 단순 기능 설명이 아니라 실제 전환 리스크와 "
                "적용 범위를 판단할 수 있습니다."
            ),
            (
                f"PoC는 {subject}의 기능 시연에 그치지 않고 전환 과정에서 확인할 기준을 "
                "정의해야 합니다. 기존 시스템 연계 방식, 장애 대응 기준, 운영 안정화 조건을 "
                "검증표로 제시하면 고객이 전환 가능성을 더 구체적으로 비교할 수 있습니다."
            ),
            (
                "레퍼런스 자료는 구축 사실 나열보다 유사 사업에서 어떤 전환 범위와 운영 전환 "
                "계획을 관리했는지 보여주는 구조로 재구성할 필요가 있습니다. 이를 통해 SK AX는 "
                "핵심 업무 시스템 전환을 안정적으로 설계·관리할 수 있는 파트너라는 점을 "
                "설명할 수 있습니다."
            ),
        ]
    return [
        (
            f"유사 사업에서는 {subject}의 적용 범위와 고객 확인 기준이 먼저 비교될 수 있습니다. "
            "SK AX는 제안서에서 적용 범위, 역할 분담, 일정 조건을 분리해 제시해야 합니다. "
            "그래야 고객이 단순 기능 설명이 아니라 실제 수행 범위를 판단할 수 있습니다."
        ),
        (
            f"PoC는 {subject}의 기능 시연에 그치지 않고 고객이 확인할 기준을 정의해야 합니다. "
            "적용 조건, 운영 기준, 장애 대응 기준을 검증표로 제시하면 고객이 실제 도입 가능성을 "
            "더 구체적으로 비교할 수 있습니다."
        ),
        (
            "레퍼런스 자료는 수행 사실 나열보다 유사 사업에서 어떤 적용 범위와 운영 기준을 "
            "관리했는지 보여주는 구조로 재구성할 필요가 있습니다. 이를 통해 SK AX는 사업 적용과 "
            "운영 안착을 함께 관리할 수 있는 파트너라는 점을 설명할 수 있습니다."
        ),
    ]


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
            f"{subject}와 연결되는 신호로 해석할 수 있습니다."
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
            f"{subject}와 연결되는 배경으로 확인됩니다. 따라서 이 신호는 역할 확장이나 "
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


def _two_section_fact_based_fallback(
    original: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    profile_context: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Build a conservative result from IntegratedIssue facts only."""

    company = _main_company_display(integrated_issue) or "피어사"
    subject = _issue_subject_phrase(integrated_issue) or _primary_issue_fact(integrated_issue)
    subject = subject or "현재 사건"
    evidence_text = _integrated_grounding_text(integrated_issue)
    scope = _fact_based_scope_phrase(evidence_text)
    work_units = _fact_based_work_unit_phrase(evidence_text)
    external_plan = _fact_based_external_customer_phrase(evidence_text)
    profile_phrase = _profile_area_phrase(
        profile_context,
        integrated_issue=integrated_issue,
        scope="peer",
    )
    fact_ids = list(_known_fact_ids(integrated_issue))[:5]
    scope_for_sentence = scope or "기업 고객에게 설명할 적용 범위"
    work_units_for_sentence = work_units or "문서 처리·개발 지원·협업"
    analysis_summary = (
        f"{company}의 이번 사건은 {subject}를 단순 도입 소식이 아니라 "
        f"{scope_for_sentence}를 확인하는 근거로 볼 수 있습니다."
    )

    strategic_meaning = [
        (
            "기사 근거에서 확인되는 적용 범위는 "
            f"{scope_for_sentence}이며, "
            f"업무 단위로 {work_units_for_sentence}가 함께 제시됩니다."
        )
    ]
    if external_plan:
        strategic_meaning.append(
            f"{external_plan}이 확인되므로, 기업용 AI 도입 제안에서는 내부 적용 경험을 "
            "외부 고객이 확인할 수 있는 운영 기준으로 바꾸는지가 중요합니다."
        )
    else:
        strategic_meaning.append(
            "따라서 이 사건은 모델 성능 자체보다 적용 범위, 업무 연결 방식, 운영 검증 기준을 "
            "함께 설명해야 한다는 신호입니다."
        )

    market_signal = (
        "고객은 기업용 AI 도입에서 모델명보다 실제 업무 적용 범위, 내부 시스템 연계 방식, "
        "운영 중 확인할 기준을 함께 비교하게 됩니다."
    )
    peer_meaning = (
        f"{company}는 {subject}를 통해 {scope_for_sentence}와 "
        f"업무 단위의 {work_units_for_sentence}를 제시했습니다."
    )
    if profile_phrase:
        peer_meaning += f" 이는 피어 프로필의 {profile_phrase} 맥락과 연결됩니다."
    capability_change = (
        "확인된 변화는 역량 우위가 아니라 기업용 AI를 어느 조직 범위와 업무 단위에 적용할지, "
        "그리고 그 경험을 외부 고객 설명 근거로 전환할지에 있습니다."
    )
    why_important = (
        "SK AX 대응 방향은 피어사의 도입 사실을 따라가는 것이 아니라, 고객이 확인할 "
        "전사 적용 범위와 운영 검증 기준을 더 구체적으로 제시하는 데 맞춰야 합니다."
    )
    potential_impact = (
        "고객은 AI 도입 제안에서 모델 기능 설명만으로는 실제 적용 가능성을 판단하기 어렵습니다. "
        "따라서 SK AX는 업무별 적용 범위, 내부 시스템 연계 방식, 운영 책임을 한 번에 "
        "비교할 수 있게 제안 구조를 바꿔야 합니다."
    )
    recommended_actions = [
        (
            "제안서에는 모델 기능 비교와 별도로 전사 적용 범위표를 둡니다. "
            "부서·계열사·업무 단위별 적용 대상을 나누면 고객은 도입 범위와 운영 책임을 "
            "한눈에 확인할 수 있습니다."
        ),
        (
            "PoC 검증표는 답변 품질 중심이 아니라 업무 단위별 검증으로 바꿉니다. "
            f"{work_units or '문서 처리, 개발 지원, 협업'} 같은 적용 장면별 검증 기준과 "
            "내부 시스템 연계 조건을 함께 확인하게 해야 합니다."
        ),
        (
            "레퍼런스 자료는 도입 사실 나열보다 내부 적용 경험을 외부 고객 제안에 어떻게 "
            "전환했는지 보여주는 구조로 정리합니다. 고객은 이를 통해 실제 운영 전환 가능성과 "
            "도입 후 확인할 기준을 비교할 수 있습니다."
        ),
    ]
    result = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": analysis_summary,
            "strategic_meaning": strategic_meaning[:2],
            "market_signal": market_signal,
            "impact_level": (original.get("analysis") or {}).get("impact_level") or "medium",
            "impact_reason": (
                "현재 근거에서 조직 적용 범위와 업무 적용 장면이 확인되어, 고객 제안의 "
                "비교 기준을 구체화할 수 있습니다."
            ),
            "risk_or_opportunity": "opportunity",
            "confidence": 0.72,
            "reason": (
                "LLM repair 실패 후 IntegratedIssue 근거만 사용해 "
                "두 섹션 문안으로 복구했습니다."
            ),
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": str(integrated_issue.get("main_company") or ""),
                "company_name_ko": company,
                "peer_meaning": peer_meaning,
                "capability_change": capability_change,
                "sourced_evidence_ids": fact_ids,
            },
            "skax_implication": {
                "why_important": why_important,
                "potential_impact": potential_impact,
                "opportunities": [],
                "threats": [],
                "recommended_actions": recommended_actions,
                "business_line_mapping": _safe_business_line_mapping(
                    profile_context,
                    integrated_issue=integrated_issue,
                ),
            },
            "follow_up_questions": [
                "실제 적용 대상 조직과 업무 단위가 어디까지인지 확인이 필요합니다.",
                "내부 적용 경험을 외부 고객 제안 근거로 쓰기 위한 운영 기준이 "
                "무엇인지 확인해야 합니다.",
            ],
            "watch_points": [
                "전사 적용 범위와 업무별 활용 기준이 후속 기사나 고객 사례에서 구체화되는지",
                "외부 고객 대상 맞춤형 AI 구축 서비스의 실제 제공 방식이 공개되는지",
            ],
            "confidence": 0.72,
            "evidence_label": "moderate",
            "provenance": {
                "generator": "StrategicInsightAgent",
                "prompt_version": _PROMPT_VERSION,
                "model": model,
                "used_fact_ids": fact_ids,
                "used_context_layers": ["integrated_issue_fact_fallback"],
                "run_at": datetime.now(UTC).isoformat(),
            },
        },
    }
    return _restore_valid_flags_if_structurally_safe(result)


def _fact_based_scope_phrase(evidence_text: str) -> str:
    text = str(evidence_text or "")
    group_scope_pattern = (
        r"(?:[가-힣A-Za-z0-9&·+_-]+\s*)?그룹\s*"
        r"(?:계열사|전\s*계열사|사)\s*전반|그룹\s*전\s*계열사"
    )
    if re.search(group_scope_pattern, text):
        return "그룹 계열사 전반의 적용 범위"
    if re.search(r"전사|임직원", text):
        return "전사 임직원의 업무 적용 범위"
    if re.search(r"기업\s*고객|외부\s*고객", text):
        return "기업 고객 대상 적용 범위"
    return ""


def _fact_based_work_unit_phrase(evidence_text: str) -> str:
    text = str(evidence_text or "")
    units = []
    patterns = [
        ("개발", r"개발|코딩"),
        ("문서 처리", r"문서"),
        ("협업", r"협업"),
        ("내부 시스템 연계", r"내부\s*시스템"),
        ("AI 에이전트 구축", r"AI\s*에이전트|에이전트\s*구축"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE) and label not in units:
            units.append(label)
    return "·".join(units[:4])


def _fact_based_external_customer_phrase(evidence_text: str) -> str:
    text = str(evidence_text or "")
    if re.search(r"외부|다른\s*기업|기업\s*고객", text) and re.search(r"맞춤형|구축|서비스", text):
        return "내부 적용 경험을 바탕으로 외부 기업 고객에게 맞춤형 AI 구축 서비스를 제공할 계획"
    return ""


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
