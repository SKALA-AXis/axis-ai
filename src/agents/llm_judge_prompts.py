"""LLM-as-Judge 프롬프트 4종 (W5-2).

설계: design/01-supervisor-implementation-plan.md §3.6.1.
gpt-4o-mini structured output (json_mode) 으로 `{score: 0-5, reasoning: str}` 강제.
"""

from __future__ import annotations

JUDGE_PROMPT_VERSION = "judge-v1.0"
JUDGE_MODEL = "gpt-4o-mini"

# 단일 시스템 프롬프트 — 모든 4 score 가 같은 형식.
JUDGE_SYSTEM_PROMPT = """\
당신은 SK AX 카드뉴스 품질 평가 AI 입니다. 카드의 implication 본문과 evidence
payload 를 비교해서 평가 기준 별로 0-5 정수 점수와 한국어 한 줄 reasoning 을
제시합니다. 출력은 JSON only.

평가 기준 별 점수 의미 (0=최악, 5=최고):
- 0: 평가 불가 / 본문 비어있음 / 명백한 환각
- 1-2: 출처와 어긋남 또는 generic / actionable 하지 않음
- 3: 부분적으로 일치, 일부 generic
- 4: 대부분 일치, 일부 추가 검증 필요
- 5: 모든 claim 이 evidence 와 일치, 구체적이고 actionable

JSON 형식 (다른 텍스트 금지):
{
  "score": 4,
  "reasoning": "한 줄 한국어 평가 근거"
}
"""

FAITHFULNESS_USER_PROMPT = """\
## 평가 기준 — faithfulness
implication 본문의 모든 claim (수치, 날짜, 회사명, 사실 진술) 이 evidence_payload
또는 integrated_issue.fact_basis 의 내용과 grounded 인지 평가합니다. 출처에 없는
수치 / 인용 / 날짜 가 발견되면 점수를 낮춥니다.

## implication
{implication_json}

## evidence_payload
{evidence_json}

## fact_basis (선택)
{fact_basis_json}
"""

SPECIFICITY_USER_PROMPT = """\
## 평가 기준 — specificity
implication 이 '디지털 전환 가속화', '시장 변화' 같은 generic 문장 위주인지,
회사·사업·기술·사람·날짜 등 구체 정보를 인용한 peer-specific 분석인지 평가합니다.
generic 표현 비율이 높을수록 점수가 낮습니다.

## implication
{implication_json}

## 핵심 회사 / 섹터
{cluster_meta_json}
"""

ACTIONABILITY_USER_PROMPT = """\
## 평가 기준 — actionability
skax_implication.recommended_actions 가 SK AX 의 3개 사업군 (에이전틱AI / 제조AX /
MSP) 영역과 매칭되며, 실제 실행 가능한 단위 task 인지 평가합니다. "검토한다",
"강화한다" 같은 추상적 권고 보다는 "고객사 X 에 PoC 제안서 작성", "MSP 입찰 자격
점검" 같은 구체 task 가 높은 점수를 받습니다.

## skax_implication
{skax_json}

## SK AX business_lines
- 에이전틱AI: 자율 AI 에이전트 기반 엔터프라이즈 솔루션
- 제조AX: 제조업 특화 AI 전환 (스마트팩토리, 공정 최적화)
- MSP: 멀티클라우드 관리 서비스
"""

PEER_RELEVANCE_USER_PROMPT = """\
## 평가 기준 — peer_relevance
peer_implication 과 skax_implication 이 각각 다른 관점 (peer 의 자기 사업 관점 vs
SK AX 의 경쟁/협력/대응 관점) 으로 충분히 차별화되어 있는지 평가합니다. 양쪽이
동일 내용 반복이면 낮은 점수, 분명히 다른 분석이면 높은 점수.

## peer_implication
{peer_json}

## skax_implication
{skax_json}
"""


SCORE_PROMPTS: dict[str, str] = {
    "faithfulness": FAITHFULNESS_USER_PROMPT,
    "specificity_llm": SPECIFICITY_USER_PROMPT,
    "actionability_llm": ACTIONABILITY_USER_PROMPT,
    "peer_relevance": PEER_RELEVANCE_USER_PROMPT,
}


__all__ = [
    "ACTIONABILITY_USER_PROMPT",
    "FAITHFULNESS_USER_PROMPT",
    "JUDGE_MODEL",
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SYSTEM_PROMPT",
    "PEER_RELEVANCE_USER_PROMPT",
    "SCORE_PROMPTS",
    "SPECIFICITY_USER_PROMPT",
]
