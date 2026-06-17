"""MixerAnalysisAgent — linked integrated/analysis/implication result mixer.

design: ``axis-ai/design/30-analysis/mixer-analysis.md``.

본 모듈은 카드 표시 문구를 다시 요약하지 않는다. 카드에 연결된
``integrated_issue`` / ``analysis`` / ``implication`` / ``profile_context`` 여러 건을
입력으로 받아, 단일 이슈로는 보이지 않는 공통 패턴 / 비교 포인트 / 숨은 결론 /
대응방향을 도출한다.

핵심 entry point:

    ``MixerAnalysisAgent().analyze(card_ids, ratios, user_context, analysis_mode)``
    — DB 카드 기반.
    ``MixerAnalysisAgent().analyze_items(items, ratios, user_context, analysis_mode)``
    — 로컬 목업/테스트 기반.

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
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from src.agents.implication_agent import ImplicationAgent
from src.llm import LLMSpec, build_chat_llm
from src.middleware.analysis_ledger import with_ledger_writeback
from src.observability.langfuse_client import tracing_config
from src.services.agent_output_validation import (
    confidence_in_range,
)
from src.services.analysis_units import (
    AnalysisUnit,
    analysis_units_from_cards,
    card_like_from_units,
    confidence_penalty_for_flags,
    load_analysis_units_by_card_ids,
    load_analysis_units_by_integrated_issue_ids,
    quality_flags_for_units,
    source_integrated_issue_ids,
)
from src.services.llm_env import (
    ensure_llm_env_loaded,
    is_missing_llm_credentials_error,
    missing_llm_credentials_message,
)
from src.shared.json_helpers import json_dict as _json_dict

log = logging.getLogger(__name__)

_QUICK_LLM_MODEL = os.getenv("MIXER_QUICK_LLM_MODEL", "gpt-4o")
_DEEP_LLM_MODEL = os.getenv("MIXER_DEEP_LLM_MODEL", "gpt-5.5")
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
# 믹서 실행 단계 — SSE progress 용. 에이전트가 실제로 넘는 단계 경계만 emit 한다
# (prepare: 카드/이슈 로드, analyze: 메인 LLM, synthesize: 대응방향 LLM, finalize: 추론 정리).
ProgressFn = Callable[[str, str, int, int], None]
_PROGRESS_STAGES: dict[str, str] = {
    "prepare": "선택한 카드와 통합 이슈를 불러오는 중",
    "analyze": "카드들의 공통 패턴·비교 포인트·숨은 결론을 분석하는 중",
    "synthesize": "SK AX 관점의 대응 방향을 도출하는 중",
    "finalize": "추론 흐름과 근거 카드를 정리하는 중",
}
_PROGRESS_ORDER: list[str] = ["prepare", "analyze", "synthesize", "finalize"]
_PROGRESS_TOTAL = len(_PROGRESS_ORDER)


def clip_string(value: Any, max_length: int, *, suffix: str = "") -> str:
    """Keep mixer text within display bounds without cutting a sentence mid-way."""
    del suffix
    if not isinstance(value, str):
        return ""
    text = _clean_mixer_sentence_text(value)
    if len(text) <= max_length:
        return text
    return _clip_to_complete_sentence(text, max_length)


def clip_final_one_liner(value: Any, *, max_length: int = _MIXER_FINAL_ONE_LINER_MAX) -> str:
    return clip_string(value, max_length)


def clip_implication(value: Any, *, max_length: int = _MIXER_IMPLICATION_MAX) -> str:
    return clip_string(value, max_length)


def _clean_mixer_sentence_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("…", "").replace("...", "").replace("..", ".").replace(" .", ".").strip()


def _sentence_base(value: str) -> str:
    return value.strip().rstrip(".!?。").strip()


def _looks_incomplete_display_sentence(value: str) -> bool:
    base = _sentence_base(value)
    if not base:
        return True
    return any(base.endswith(ending) for ending in _INCOMPLETE_KOREAN_ENDINGS)


def _to_polite_display_sentence(value: str) -> str:
    base = _sentence_base(value)
    if not base:
        return ""
    if _looks_incomplete_display_sentence(base):
        return ""
    for informal, polite in _POLITE_ENDING_REPLACEMENTS:
        if base.endswith(informal):
            base = f"{base[: -len(informal)]}{polite}"
            break
    if not re.search(r"(습니다|합니다|됩니다|입니다|니다|요)$", base):
        return ""
    return f"{base}."


def _is_complete_display_sentence(value: str) -> bool:
    normalized = _to_polite_display_sentence(value)
    if not normalized:
        return False
    base = _sentence_base(normalized)
    return bool(re.search(r"(습니다|합니다|됩니다|입니다|니다|요|다)$", base))


def _complete_sentence_parts(value: str) -> list[str]:
    text = _clean_mixer_sentence_text(value)
    if not text:
        return []
    matches = re.findall(r"[^.!?。]+[.!?。]", text)
    if not matches and _is_complete_display_sentence(text):
        matches = [text]
    return [normalized for part in matches if (normalized := _to_polite_display_sentence(part))]


def _clip_to_complete_sentence(value: str, max_length: int) -> str:
    parts = _complete_sentence_parts(value)
    if not parts:
        return _clean_mixer_sentence_text(value).strip()
    kept: list[str] = []
    for part in parts:
        candidate = " ".join([*kept, part]).strip()
        if len(candidate) > max_length:
            break
        kept.append(part)
    if kept:
        return " ".join(kept).strip()
    return parts[0]


def _normalize_mixer_display_sentence(value: object, max_length: int) -> str:
    text = _clean_mixer_sentence_text(value)
    if not text:
        return ""
    parts = _complete_sentence_parts(text)
    if parts:
        text = " ".join(parts)
    if len(text) > max_length:
        text = _clip_to_complete_sentence(text, max_length)
    return _to_polite_display_sentence(text)


def _emit_progress(progress: "ProgressFn | None", stage: str) -> None:
    """단계 경계에서 progress 콜백 호출 (None 이면 no-op, 예외는 분석을 막지 않음)."""
    if progress is None:
        return
    try:
        index = _PROGRESS_ORDER.index(stage)
    except ValueError:
        index = 0
    try:
        progress(stage, _PROGRESS_STAGES.get(stage, stage), index, _PROGRESS_TOTAL)
    except Exception:
        log.debug("mixer progress 콜백 실패 (무시)", exc_info=True)


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

_llms: dict[str, ChatOpenAI] = {}


def _model_for_mode(analysis_mode: object = "quick") -> str:
    if _normalize_analysis_mode(analysis_mode) == "deep":
        return _DEEP_LLM_MODEL
    return _QUICK_LLM_MODEL


def _llm_max_completion_tokens(model: str) -> int:
    default = "9000" if str(model).startswith("gpt-5") else "3000"
    return int(os.getenv("MIXER_MAX_COMPLETION_TOKENS", default))


def _get_llm(analysis_mode: object = "quick") -> ChatOpenAI:
    model = _model_for_mode(analysis_mode)
    if model not in _llms:
        ensure_llm_env_loaded()
        # gpt-5 reasoning_effort 분기·json_object 래핑은 공용 팩토리가 처리.
        # 모델별 _llms 캐시는 그대로 유지(quick/deep 분리).
        _llms[model] = build_chat_llm(
            LLMSpec(
                model=model,
                temperature=0.15,
                max_tokens=_llm_max_completion_tokens(model),
                json_object=True,
                reasoning_effort=os.getenv("MIXER_DEEP_REASONING_EFFORT", "medium"),
            )
        )
    return _llms[model]


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _normalize_analysis_mode(value: object) -> str:
    return "deep" if str(value or "").strip().lower() == "deep" else "quick"


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
- 각 axis가 본문 근거에서 무엇을 확인하게 하는 분석 질문인지 analysis_prompt에 쓰세요.
- interpretation에는 현재 카드 묶음의 본문 근거를 통해 그 축을 어떻게 읽어야 하는지 답하세요.
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


class MixerAnalysisAgent:
    """3-phase per_card / cross_card / synthesis CoT 카드 분석 agent."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    @with_ledger_writeback("MixerAnalysisAgent")
    async def analyze(
        self,
        card_ids: list[str] | None = None,
        integrated_issue_ids: list[str] | None = None,
        ratios: dict | None = None,
        user_context: str | None = None,
        analysis_mode: str = "quick",
        user_id: str | None = None,
        progress: "ProgressFn | None" = None,
    ) -> dict:
        """N 카드 선택 → 저장된 분석 payload 기반 6축 radar + cross-issue 분석.

        Args:
            card_ids: 프론트에서 선택한 카드 id (호환 입력, 2 ≤ N ≤ 20 권장).
            integrated_issue_ids: canonical integrated_issues.id 입력. card_ids보다 우선.
            ratios: peer / industry / keyword 가중치 (frontend slider 결과).
            user_context: 사용자 자유 입력.
            analysis_mode: "quick" 은 메인 믹스 분석만 실행, "deep" 은 품질 보강과
                mix-level ImplicationAgent 보강까지 실행.

        Returns:
            MixerAnalysisOutput dict — design §5 schema.
        """
        requested_card_ids = _dedupe_keep_order(
            [str(card_id).strip() for card_id in card_ids or [] if str(card_id).strip()]
        )
        requested_integrated_issue_ids = _dedupe_keep_order(
            [
                str(issue_id).strip()
                for issue_id in integrated_issue_ids or []
                if str(issue_id).strip()
            ]
        )
        requested_ids = (
            requested_integrated_issue_ids if requested_integrated_issue_ids else requested_card_ids
        )
        if len(requested_ids) < _MIN_CARDS:
            return _error_response(
                "분석 단위 부족",
                f"mixer 는 최소 {_MIN_CARDS}개 분석 단위 필요 (받음={len(requested_ids)})",
                requested_ids,
                integrated_issue_ids=requested_integrated_issue_ids,
            )

        if len(requested_ids) > _MAX_CARDS:
            log.warning(
                "Mixer | 분석 단위가 너무 많음 — 상위 %d개로 truncate (받음=%d)",
                _MAX_CARDS,
                len(requested_ids),
            )
            if requested_integrated_issue_ids:
                requested_integrated_issue_ids = requested_integrated_issue_ids[:_MAX_CARDS]
            else:
                requested_card_ids = requested_card_ids[:_MAX_CARDS]
            requested_ids = (
                requested_integrated_issue_ids
                if requested_integrated_issue_ids
                else requested_card_ids
            )

        _emit_progress(progress, "prepare")
        if requested_integrated_issue_ids:
            if user_id is not None:
                analysis_units = load_analysis_units_by_integrated_issue_ids(
                    requested_integrated_issue_ids,
                    user_id=user_id,
                )
            else:
                analysis_units = load_analysis_units_by_integrated_issue_ids(
                    requested_integrated_issue_ids
                )
        else:
            if user_id is not None:
                analysis_units = load_analysis_units_by_card_ids(
                    requested_card_ids, user_id=user_id
                )
            else:
                analysis_units = load_analysis_units_by_card_ids(requested_card_ids)
        if len(analysis_units) < _MIN_CARDS:
            return _error_response(
                "분석 단위 조회 실패",
                f"canonical AnalysisUnit 조회 결과 부족 (받음={len(analysis_units)})",
                requested_ids,
                integrated_issue_ids=requested_integrated_issue_ids,
            )

        cards = card_like_from_units(analysis_units)
        source_anchor_ids = [str(card.get("id")) for card in cards if card.get("id")]
        return await self._analyze_cards(
            cards=cards,
            requested_card_ids=source_anchor_ids,
            requested_integrated_issue_ids=requested_integrated_issue_ids,
            analysis_units=analysis_units,
            ratios=ratios,
            user_context=user_context,
            analysis_mode=analysis_mode,
            progress=progress,
        )

    async def analyze_items(
        self,
        items: list[dict],
        ratios: dict | None = None,
        user_context: str | None = None,
        analysis_mode: str = "quick",
        progress: "ProgressFn | None" = None,
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
        analysis_units = analysis_units_from_cards(cards)
        cards = card_like_from_units(analysis_units)
        card_ids = [str(card.get("id")) for card in cards if card.get("id")]

        return await self._analyze_cards(
            cards=cards,
            requested_card_ids=card_ids,
            requested_integrated_issue_ids=[],
            analysis_units=analysis_units,
            ratios=ratios,
            user_context=user_context,
            analysis_mode=analysis_mode,
            progress=progress,
        )

    async def _analyze_cards(
        self,
        *,
        cards: list[dict],
        requested_card_ids: list[str],
        requested_integrated_issue_ids: list[str],
        analysis_units: list[AnalysisUnit],
        ratios: dict | None,
        user_context: str | None,
        analysis_mode: str,
        progress: "ProgressFn | None" = None,
    ) -> dict:
        normalized_mode = _normalize_analysis_mode(analysis_mode)
        # 6축 radar 미리 계산 — LLM input 으로 anchor 제공 (v2)
        radar = _compute_radar(cards)
        prompt = (
            _MIXER_PROMPT.replace("{context}", _format_analysis_units(cards))
            .replace("{radar_text}", _format_radar(radar))
            .replace("{ratios_text}", _format_ratios(ratios))
            .replace("{user_context}", (user_context or "").strip() or "*없음*")
        )

        _emit_progress(progress, "analyze")
        try:
            response = _get_llm(normalized_mode).invoke(
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
            detail = (
                missing_llm_credentials_message() if is_missing_llm_credentials_error(e) else str(e)
            )
            return _error_response(
                "LLM 호출 실패",
                detail,
                requested_card_ids,
                integrated_issue_ids=requested_integrated_issue_ids,
            )

        result = _parse_and_validate(content, cards, requested_card_ids)
        sentence_quality_issues = _mixer_sentence_quality_issues(result)
        if normalized_mode == "deep" or sentence_quality_issues:
            if sentence_quality_issues:
                log.info(
                    "Mixer sentence quality repair requested | issues=%s",
                    sentence_quality_issues,
                )
            result = _repair_mixer_result_quality(
                result=result,
                cards=cards,
                requested_card_ids=requested_card_ids,
            )
        source_issue_ids = source_integrated_issue_ids(analysis_units)
        quality_flags = quality_flags_for_units(analysis_units)
        if source_issue_ids:
            result["source_integrated_issue_ids"] = source_issue_ids
        else:
            result.setdefault("source_integrated_issue_ids", [])
        if quality_flags:
            result["quality_flags"] = quality_flags
            penalty = confidence_penalty_for_flags(quality_flags)
            result["confidence"] = round(
                max(0.0, confidence_in_range(result.get("confidence", 0.0)) - penalty),
                2,
            )
        _emit_progress(progress, "synthesize")
        if normalized_mode == "deep":
            result["action_details"] = _augment_deep_action_details(result)
            actions = _recommended_actions_from_details(result, limit=5)
            mix_implication = _generate_mix_level_implication(result=result, cards=cards)
            if mix_implication:
                result["mix_implication"] = mix_implication
                implication_actions = _recommended_actions_from_implication(
                    mix_implication, result=result, limit=5
                )
                actions = _dedupe_keep_order([*actions, *implication_actions])
                if not actions:
                    actions = _recommended_actions_from_basis(result, limit=5)
            if actions:
                result["recommended_actions"] = actions[:5]
                result["sk_ax_implication"] = clip_implication(" ".join(actions[:5]))
        else:
            actions = _recommended_actions_from_details(result) or _recommended_actions_from_basis(
                result
            )
            if actions:
                result["recommended_actions"] = actions[:3]
                result["sk_ax_implication"] = clip_implication(" ".join(actions[:3]))
        result = _normalize_mixer_result_display_sentences(result)
        result["radar_axes"] = _merge_radar_axis_interpretations(
            radar, result.get("radar_axis_interpretations")
        )
        result["mix_id"] = _new_mix_id()
        result.setdefault("provenance", {}).update(
            {
                "llm_model": _model_for_mode(normalized_mode),
                "prompt_version": _PROMPT_VERSION,
                "analysis_mode": normalized_mode,
                "analysis_quality": "fast" if normalized_mode == "quick" else "detailed",
                "source_card_ids": [c["id"] for c in cards],
                "requested_integrated_issue_ids": requested_integrated_issue_ids,
                "source_integrated_issue_ids": source_issue_ids,
                "quality_flags": quality_flags,
                "ratios": ratios or {},
                "analysis_basis": (
                    "integrated_issues.id -> integrated_issue+analysis+implication+"
                    "classification+validation"
                ),
            }
        )
        # 추론 흐름(trail) / 단계별 CoT(steps) / 후속 질문 — repair 이후 최종 blocks 기반으로
        # 결정적 구성 (추가 LLM 호출 없음). card_ids / integrated_issue_ids 양 경로 모두 채워짐.
        _emit_progress(progress, "finalize")
        result["analysis_depth"] = _build_analysis_depth(result, cards, normalized_mode)
        result["deep_dive_sections"] = (
            _build_deep_dive_sections(result, cards) if normalized_mode == "deep" else []
        )
        result["reasoning_trail"] = _build_reasoning_trail(result, cards)
        result["reasoning_steps"] = _build_reasoning_steps(result, cards)
        result["follow_up_checks"] = _build_follow_up_checks(result, cards)
        result["follow_up_questions"] = [
            str(item.get("question"))
            for item in _json_list(result["follow_up_checks"])
            if isinstance(item, dict) and str(item.get("question") or "").strip()
        ][:3]
        result["warning"] = _warning_for(result)
        return result


# ──────────────────────────────────────────────────────────────────────────
# Deterministic radar (no LLM) — design §6.1
# ──────────────────────────────────────────────────────────────────────────


def _avg(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def _cards_for_event(cards: list[dict], event_types: set[str]) -> list[dict]:
    return [card for card in cards if (card.get("event_type") or "") in event_types]


def _score_for_event(cards: list[dict], event_types: set[str]) -> float:
    return _avg([_card_score(c) for c in _cards_for_event(cards, event_types)])


def _cards_for_sector(cards: list[dict], sectors: set[str]) -> list[dict]:
    return [card for card in cards if _card_sector(card) in sectors]


def _score_for_sector(cards: list[dict], sectors: set[str]) -> float:
    return _avg([_card_score(c) for c in _cards_for_sector(cards, sectors)])


def _radar_axis(
    *,
    axis: str,
    score: float,
    explanation: str,
    calculation: str,
    matched_cards: list[dict],
    total_count: int,
) -> dict:
    rounded_score = round(score, 3)
    return {
        "axis": axis,
        "score": rounded_score,
        "explanation": explanation,
        "calculation": calculation,
        "meaning": _radar_score_meaning(rounded_score),
        "support_count": len(matched_cards),
        "total_count": total_count,
        "matched_card_ids": [
            str(card.get("id")) for card in matched_cards if str(card.get("id") or "").strip()
        ],
    }


def _radar_score_meaning(score: float) -> str:
    if score >= 0.7:
        return "선택한 카드 묶음에서 강한 판단 신호로 볼 수 있습니다."
    if score >= 0.35:
        return "일부 카드가 해당 축을 지지하므로 보조 판단 신호로 봅니다."
    if score > 0:
        return "근거는 있으나 선택 묶음 전체를 대표할 정도는 아닙니다."
    return "이번 선택 묶음에서는 이 축을 직접 지지하는 카드가 확인되지 않았습니다."


def _card_score(card: dict) -> float:
    impl = _dict_or_empty(card.get("implication"))
    sector_meta = _dict_or_empty(impl.get("sector_meta"))
    exposure = impl.get("exposure_score") or sector_meta.get("exposure_score")
    if exposure is None:
        exposure = card.get("importance_score")
    try:
        return float(exposure or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _card_sector(card: dict) -> str:
    impl = _dict_or_empty(card.get("implication"))
    sector_meta = _dict_or_empty(impl.get("sector_meta"))
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
    total_count = len(cards)
    strategic_cards = _cards_for_event(cards, {"ma", "new_biz"})
    tech_cards = _cards_for_sector(cards, {"ax", "ai_tech", "infra"})
    partnership_cards = _cards_for_event(cards, {"partnership"})
    regulation_cards = _cards_for_event(cards, {"regulation"})
    talent_cards = _cards_for_event(cards, {"personnel"})
    peer_cards = [card for card in cards if card.get("peer_id")]
    axes = {
        "peer_strategic_shift": _radar_axis(
            axis="peer_strategic_shift",
            score=_score_for_event(cards, {"ma", "new_biz"}),
            explanation="M&A·신사업처럼 전략 전환 성격으로 분류된 카드의 exposure 평균입니다.",
            calculation="event_type이 ma 또는 new_biz인 카드만 골라 exposure_score를 평균했습니다.",
            matched_cards=strategic_cards,
            total_count=total_count,
        ),
        "tech_investment": _radar_axis(
            axis="tech_investment",
            score=_score_for_sector(cards, {"ax", "ai_tech", "infra"}),
            explanation=(
                "AX·AI 기술·인프라 섹터 카드가 기술 투자 판단을 얼마나 지지하는지 본 값입니다."
            ),
            calculation="sector가 ax, ai_tech, infra인 카드의 exposure_score를 평균했습니다.",
            matched_cards=tech_cards,
            total_count=total_count,
        ),
        "market_position": _radar_axis(
            axis="market_position",
            score=_peer_diversity_score(cards),
            explanation=(
                "선택 묶음이 특정 peer 한 곳의 소식인지, "
                "여러 peer에 걸친 시장 신호인지 보는 값입니다."
            ),
            calculation="서로 다른 peer 수를 국내 주요 peer 4개 기준으로 나눠 정규화했습니다.",
            matched_cards=peer_cards,
            total_count=total_count,
        ),
        "partnership_momentum": _radar_axis(
            axis="partnership_momentum",
            score=_score_for_event(cards, {"partnership"}),
            explanation="제휴·협력 이벤트가 선택 묶음의 핵심 동력인지 보는 값입니다.",
            calculation="event_type이 partnership인 카드의 exposure_score를 평균했습니다.",
            matched_cards=partnership_cards,
            total_count=total_count,
        ),
        "regulatory_risk": _radar_axis(
            axis="regulatory_risk",
            score=_score_for_event(cards, {"regulation"}),
            explanation="규제·정책 이벤트가 의사결정 리스크로 작동하는지 보는 값입니다.",
            calculation="event_type이 regulation인 카드의 exposure_score를 평균했습니다.",
            matched_cards=regulation_cards,
            total_count=total_count,
        ),
        "talent_movement": _radar_axis(
            axis="talent_movement",
            score=_score_for_event(cards, {"personnel"}),
            explanation="조직·인력 변화가 선택 묶음의 실행 역량 신호인지 보는 값입니다.",
            calculation="event_type이 personnel인 카드의 exposure_score를 평균했습니다.",
            matched_cards=talent_cards,
            total_count=total_count,
        ),
    }
    return [axes[axis] for axis in _RADAR_AXIS_ORDER]


def _radar_axis_analysis_prompt(axis: str) -> str:
    return _RADAR_AXIS_PROMPTS.get(axis, "본문 근거가 이 신호 축을 어떻게 지지하거나 약화하는가?")


def _fallback_radar_prompted_interpretation(axis: dict) -> str:
    axis_id = str(axis.get("axis") or "")
    label = _RADAR_AXIS_LABELS.get(axis_id, axis_id or "해당 축")
    score = float(axis.get("score") or 0.0)
    support_count = int(axis.get("support_count") or 0)
    total_count = int(axis.get("total_count") or 0)
    explanation = str(axis.get("explanation") or "").strip()
    meaning = str(axis.get("meaning") or "").strip()
    if support_count <= 0:
        return (
            f"{label}은 이번 카드 묶음에서 직접 근거가 약한 축입니다. "
            "따라서 핵심 결론보다는 공백 신호로 보고, 관련 후속 카드가 반복되는지 확인해야 합니다."
        )
    return clip_string(
        (
            f"{label}은 {total_count}개 중 {support_count}개 카드가 지지하고 "
            f"점수는 {score:.2f}입니다. "
            f"{explanation} {meaning}"
        ).strip(),
        360,
    )


def _merge_radar_axis_interpretations(radar: list[dict], interpretations: object) -> list[dict]:
    by_axis: dict[str, dict[str, object]] = {}
    for item in _json_list(interpretations):
        if not isinstance(item, dict):
            continue
        axis_key = str(item.get("axis") or "").strip()
        if axis_key in _RADAR_AXIS_ORDER:
            by_axis[axis_key] = item

    enriched: list[dict] = []
    for radar_axis in radar:
        axis_id = str(radar_axis.get("axis") or "").strip()
        interpretation = by_axis.get(axis_id, {})
        next_axis = dict(radar_axis)
        next_axis["analysis_prompt"] = clip_string(
            str(
                interpretation.get("analysis_prompt")
                or interpretation.get("prompt")
                or _radar_axis_analysis_prompt(axis_id)
            ).strip(),
            220,
        )
        next_axis["prompted_interpretation"] = clip_string(
            str(
                interpretation.get("interpretation")
                or interpretation.get("prompted_interpretation")
                or interpretation.get("answer")
                or _fallback_radar_prompted_interpretation(radar_axis)
            ).strip(),
            420,
        )
        enriched.append(next_axis)
    return enriched


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
    for axis in radar:
        score = float(axis.get("score") or 0.0)
        bar = "▰" * int(score * 10) + "▱" * (10 - int(score * 10))
        label = _RADAR_AXIS_LABELS.get(axis["axis"], axis["axis"])
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
    return card_like_from_units(load_analysis_units_by_card_ids(card_ids))


def _cards_from_linked_result_items(items: list[dict]) -> list[dict]:
    """런타임 linked result 입력을 Mixer 가 쓰는 card-like dict 로 정규화한다."""
    cards: list[dict] = []
    for index, item in enumerate(items or [], start=1):
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id") or item.get("id") or "").strip()
        if not card_id:
            log.warning(
                "MixerAnalysisAgent linked result item without card_id skipped | index=%s",
                index,
            )
            continue
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
        evidence_refs = evidence.get("evidence_refs") or []
        financial_refs = evidence.get("financial_refs") or []
        raw_ids = c.get("source_raw_article_ids") or []
        integrated_issue_id = (
            c.get("integrated_issue_id")
            or evidence.get("integrated_issue_id")
            or linked_results.get("integrated_issue_id")
        )
        quality_flags = c.get("quality_flags") or evidence.get("quality_flags") or []
        skax_payload = _compact_json(skax_impl) or _compact_json(implication_result)
        block = "\n".join(
            [
                f"[{c['id']}] {c.get('title', '')}",
                f"- Integrated issue id: {integrated_issue_id or '*없음*'}",
                f"- Peer: {c.get('peer_id') or c.get('company') or ''}",
                f"- Sector: {sector}",
                f"- Event type: {c.get('event_type', '')}",
                f"- Exposure: {exposure_band} ({_card_score(c):.2f})",
                f"- Source raw article ids: {raw_ids}",
                f"- Quality flags: {_compact_json(quality_flags) or '*없음*'}",
                f"- 통합 이슈: {_compact_json(_compact_integrated_issue(integrated_issue))}",
                f"- 전략 분석: {_compact_json(_compact_analysis_result(analysis_result))}",
                f"- Peer 분석: {_compact_json(peer_impl)}",
                f"- SK AX 시사점/대응: {skax_payload}",
                f"- Sector/signals: {_compact_json(sector_meta)}",
                f"- Evidence refs: {_compact_json(evidence_refs[:5])}",
                f"- 재무/수치 근거: {_compact_json(financial_refs)}",
                f"- 출처: {_compact_json(evidence_links[:5])}",
                f"- 표시 요약(최하위 보조): {summary}",
            ]
        )
        blocks.append(block)
    return "\n".join(blocks)


def _format_cards(cards: list[dict]) -> str:
    """Backward-compatible alias for older tests/imports."""
    return _format_analysis_units(cards)


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
        if not isinstance(item, int | float | str | bytes | bytearray):
            continue
        try:
            number = int(item)
        except ValueError:
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


def _valid_card_refs(value: object, allowed_card_ids: set[str]) -> list[str]:
    refs = [str(item) for item in _json_list(value) if str(item) in allowed_card_ids]
    result: list[str] = []
    for ref in refs:
        if ref not in result:
            result.append(ref)
    return result


def _valid_connections(value: object, allowed_card_ids: set[str]) -> list[dict]:
    connections: list[dict] = []
    allowed_labels = {"cause", "effect", "similar", "contrast", "reinforce"}
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
        if connection.get("label") not in allowed_labels:
            connection["label"] = "similar"
        connections.append(connection)
    return connections[:20]


def _valid_cross_card_findings(value: object, allowed_card_ids: set[str]) -> list[dict]:
    findings: list[dict] = []
    allowed_pattern_types = {
        "convergent_strategy",
        "divergent_strategy",
        "gap_in_market",
        "acceleration_signal",
        "timing_mismatch",
        "market_baseline",
    }
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        refs = _valid_card_refs(item.get("evidence_card_ids"), allowed_card_ids)
        if len(refs) < 2:
            continue
        finding = dict(item)
        finding["evidence_card_ids"] = refs
        finding["finding"] = _normalize_mixer_display_sentence(finding.get("finding", ""), 180)
        if finding.get("pattern_type") not in allowed_pattern_types:
            finding["pattern_type"] = "convergent_strategy"
        if finding["finding"]:
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
        "finding": _normalize_mixer_display_sentence(value.get("finding", ""), 260),
        "rationale": _normalize_mixer_display_sentence(value.get("rationale", ""), 320),
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
        action = _normalize_mixer_display_sentence(
            item.get("action") or item.get("text") or "",
            420,
        )
        if not action:
            continue
        details.append(
            {
                "action": action,
                "why": _normalize_mixer_display_sentence(
                    item.get("why") or item.get("rationale") or "",
                    360,
                ),
                "use_case": clip_string(item.get("use_case") or "", 80),
                "evidence": evidence[:3],
                "evidence_card_ids": refs[:6],
            }
        )
    return details[:5]


def _recommended_actions_from_details(result: dict, *, limit: int = 3) -> list[str]:
    return _dedupe_keep_order(
        [
            clip_implication(item.get("action"))
            for item in _json_list(result.get("action_details"))
            if isinstance(item, dict) and item.get("action")
        ]
    )[:limit]


def _needs_action_detail_fallback(details: list[dict]) -> bool:
    if len(details) < 2:
        return True
    return any(
        not item.get("why")
        or len(_json_list(item.get("evidence_card_ids"))) < 2
        or _is_generic_action_text(str(item.get("action") or ""))
        for item in details
    )


def _is_generic_action_text(value: str) -> bool:
    text_value = str(value or "").strip()
    if not text_value:
        return True
    generic_patterns = (
        "다음 메시지를 배치",
        "다음 비교 축을 표로 추가",
        "후속 모니터링 항목을 다음 반복 신호",
        "제안서 첫 장",
        "기능 소개",
        "운영 검증표",
        "PoC 설계",
        "후속 모니터링",
        "뉴스 재요약",
        "제안 우선순위",
        "검증표",
        "이 프로그램",
        "대시보드",
        "화면",
        "강화한다",
        "검토한다",
        "모니터링한다",
    )
    if any(pattern in text_value for pattern in generic_patterns):
        return True
    concrete_markers = (
        "사업 라인",
        "고객군",
        "우선순위",
        "오퍼링",
        "파트너십",
        "투자",
        "조직",
        "책임 조직",
        "영업",
        "리스크",
        "거버넌스",
        "상품화",
        "패키지",
        "시장 대응",
        "의사결정",
        "자원 배분",
        "가격",
        "계약",
        "레퍼런스",
        "수익화",
        "규제",
    )
    return not any(marker in text_value for marker in concrete_markers)


def _fallback_action_details_from_result(result: dict) -> list[dict]:
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    action_basis = _json_list(result.get("recommended_action_basis"))
    common_evidence = _json_list(common.get("evidence"))
    comparison_evidence = _json_list(comparison.get("evidence"))
    hidden_evidence = _json_list(hidden.get("evidence"))
    common_refs = _json_list(common.get("evidence_card_ids"))
    comparison_refs = _json_list(comparison.get("evidence_card_ids"))
    hidden_refs = _json_list(hidden.get("evidence_card_ids"))

    return [
        {
            "action": clip_implication(
                "SK AX는 임원 의사결정에서 우선 공략 고객군과 책임 조직을 먼저 정한다. "
                f"{_brief_action_basis(hidden.get('finding') or action_basis[:1])}"
                " 이 판단을 기준으로 금융과 공공/교육 등 입력에서 확인된 고객군별 "
                "사업 우선순위를 나누고, "
                "각 고객군의 오퍼링 책임 조직과 리스크 승인 권한을 지정한다."
            ),
            "why": clip_string(
                hidden.get("rationale")
                or (
                    "여러 이슈를 함께 볼 때 단일 기능보다 "
                    "조합된 판단 기준이 더 중요하게 드러나기 때문이다."
                ),
                260,
            ),
            "use_case": "사업 우선순위",
            "evidence": hidden_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": _dedupe_keep_order(
                [str(item) for item in (hidden_refs or common_refs)]
            )[:6],
        },
        {
            "action": clip_implication(
                "SK AX는 사업 라인별 오퍼링 패키지를 같은 이름의 AX 상품으로 묶지 말고 "
                "고객 의사결정 기준에 맞춰 분리 상품화한다. "
                f"{_brief_action_basis(comparison.get('finding') or action_basis[1:2])}"
                " 이 차이를 기준으로 금융권은 규제·정산·보안 거버넌스 패키지, "
                "공공/교육은 데이터 비학습·권한 통제·운영 레퍼런스 패키지처럼 "
                "영업 우선순위와 가격/계약 조건을 다르게 둔다."
            ),
            "why": clip_string(
                comparison.get("rationale")
                or "같은 흐름 안에서도 이슈마다 앞세우는 적용 장면과 성과 기준이 다르기 때문이다.",
                260,
            ),
            "use_case": "오퍼링/상품화",
            "evidence": comparison_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": _dedupe_keep_order(
                [str(item) for item in (comparison_refs or common_refs)]
            )[:6],
        },
        {
            "action": clip_implication(
                "투자 규모, 수주 전환, 규제 일정, 운영 KPI 중 어떤 지표가 확인될 때 "
                "자원 배분을 바꿀지 의사결정 기준을 먼저 고정한다. "
                f"{_brief_action_basis(common.get('finding') or action_basis[2:3])}"
                " 이 반복 신호와 연결되는 지표가 확인되면 고객군별 전담 인력, "
                "파트너십 후보, 레퍼런스 확보 예산의 우선순위를 조정한다."
            ),
            "why": clip_string(
                common.get("rationale")
                or (
                    "여러 이슈에서 반복되는 움직임은 단순 관찰 대상이 아니라 "
                    "사업 자원 배분 기준으로 반영할 필요가 있기 때문이다."
                ),
                260,
            ),
            "use_case": "파트너십/시장 대응",
            "evidence": common_evidence[:3],
            "evidence_card_ids": _dedupe_keep_order([str(item) for item in common_refs])[:6],
        },
    ]


def _augment_deep_action_details(result: dict) -> list[dict]:
    """정확 분석 전용: LLM 액션을 보존하되 실행 판단 축을 최대 5개로 보강."""
    existing = [
        dict(item) for item in _json_list(result.get("action_details")) if isinstance(item, dict)
    ]
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    action_basis = _json_list(result.get("recommended_action_basis"))
    common_refs = _json_list(common.get("evidence_card_ids"))
    comparison_refs = _json_list(comparison.get("evidence_card_ids"))
    hidden_refs = _json_list(hidden.get("evidence_card_ids"))
    all_refs = _dedupe_keep_order(
        [
            *[str(ref) for ref in common_refs],
            *[str(ref) for ref in comparison_refs],
            *[str(ref) for ref in hidden_refs],
            *[str(ref) for ref in _json_list(result.get("sources_used"))],
        ]
    )

    def refs(*groups: list[object]) -> list[str]:
        flattened: list[str] = []
        for group in groups:
            flattened.extend(str(item) for item in group if str(item))
        return _dedupe_keep_order(flattened or all_refs)[:6]

    common_evidence = _json_list(common.get("evidence"))
    comparison_evidence = _json_list(comparison.get("evidence"))
    hidden_evidence = _json_list(hidden.get("evidence"))
    candidates = [
        {
            "action": clip_implication(
                "SK AX는 입력 근거에서 성과 기준이 가장 분명한 고객군을 우선 공략군으로 정하고 "
                "해당 고객군별 영업 책임 조직과 리스크 승인 권한을 먼저 배정한다. "
                f"{_brief_action_basis(hidden.get('finding') or action_basis[:1])}"
            ),
            "why": clip_string(
                hidden.get("rationale")
                or "여러 이슈를 함께 볼 때 고객군 선택 기준이 대응 방향의 출발점이 되기 때문이다.",
                260,
            ),
            "use_case": "고객군/영업전략",
            "evidence": hidden_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": refs(hidden_refs, common_refs),
        },
        {
            "action": clip_implication(
                "SK AX는 하나의 AX 메시지로 묶기보다 공통 패턴이 가리키는 성과 기준별로 "
                "오퍼링 패키지와 가격·계약 조건을 분리해 상품화한다. "
                f"{_brief_action_basis(common.get('finding') or action_basis[1:2])}"
            ),
            "why": clip_string(
                common.get("rationale")
                or (
                    "반복 신호가 단순 기술 관심이 아니라 구매 판단 기준으로 "
                    "연결될 수 있기 때문이다."
                ),
                260,
            ),
            "use_case": "오퍼링/상품화",
            "evidence": common_evidence[:3],
            "evidence_card_ids": refs(common_refs),
        },
        {
            "action": clip_implication(
                "SK AX는 카드별 접근 차이가 큰 영역은 동일한 투자 우선순위로 보지 말고 "
                "수주 전환 가능성, 운영 KPI, 적용 범위 중 확인된 지표를 자원 배분 게이트로 둔다. "
                f"{_brief_action_basis(comparison.get('finding') or action_basis[2:3])}"
            ),
            "why": clip_string(
                comparison.get("rationale")
                or "같은 흐름 안에서도 실제 적용 장면과 성과 기준이 다르게 나타나기 때문이다.",
                260,
            ),
            "use_case": "자원 배분",
            "evidence": comparison_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": refs(comparison_refs, common_refs),
        },
        {
            "action": clip_implication(
                "SK AX는 자체 역량만으로 레퍼런스 확보가 느린 영역을 따로 분리하고 "
                "적용 현장, 운영 데이터, 고객 접점을 보완할 파트너십 후보를 우선 검증한다. "
                f"{_brief_action_basis(comparison.get('rationale') or common.get('rationale'))}"
            ),
            "why": clip_string(
                "반복 신호가 실제 사업 전환으로 이어지려면 기술 보유보다 적용 현장과 "
                "고객 설득 근거가 먼저 확보되어야 합니다.",
                260,
            ),
            "use_case": "파트너십/시장 대응",
            "evidence": comparison_evidence[:3] or hidden_evidence[:3],
            "evidence_card_ids": refs(comparison_refs, hidden_refs),
        },
        {
            "action": clip_implication(
                "SK AX는 전망·추정·계획 성격의 근거와 확정 사실을 분리해 "
                "확정 투자, 영업 메시지, 리스크 공지를 서로 다른 의사결정 게이트로 관리한다. "
                f"{_brief_action_basis(hidden.get('rationale') or action_basis[3:4])}"
            ),
            "why": clip_string(
                "정확 분석에서는 반복 신호의 방향뿐 아니라 근거의 확정 수준까지 구분해야 "
                "과잉 투자와 과소 대응을 줄일 수 있기 때문입니다.",
                260,
            ),
            "use_case": "리스크/거버넌스",
            "evidence": hidden_evidence[:3] or common_evidence[:3],
            "evidence_card_ids": refs(hidden_refs, common_refs),
        },
    ]
    merged: list[dict] = []
    seen_actions: set[str] = set()
    for item in [*existing, *candidates]:
        action = clip_implication(item.get("action") if isinstance(item, dict) else "")
        if not action or action in seen_actions:
            continue
        seen_actions.add(action)
        merged.append(
            {
                "action": action,
                "why": clip_string(item.get("why") or "", 260),
                "use_case": clip_string(item.get("use_case") or "실행 전략", 80),
                "evidence": _json_list(item.get("evidence"))[:3],
                "evidence_card_ids": refs(_json_list(item.get("evidence_card_ids"))),
            }
        )
    return merged[:5]


def _brief_action_basis(value: object) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if str(item).strip())
    text_value = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text_value:
        return "입력 근거에서 확인된 핵심 변화"
    brief = clip_string(text_value, 90).strip()
    if not brief.endswith((".", "!", "?", "…")):
        brief = f"{brief}."
    return brief


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
    allowed_phases = {"per_card", "cross_card", "synthesis"}
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        updated["inputs_used"] = _valid_card_refs(item.get("inputs_used"), allowed_card_ids)
        if updated.get("phase") not in allowed_phases:
            updated["phase"] = "synthesis"
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

    data["mix_insight"] = _normalize_mixer_display_sentence(
        data.get("mix_insight") or data.get("insight", ""),
        _MIXER_FINAL_ONE_LINER_MAX,
    )
    data["common_pattern"] = _normalize_mix_block(data.get("common_pattern"), allowed_card_ids)
    data["comparison_point"] = _normalize_mix_block(data.get("comparison_point"), allowed_card_ids)
    data["hidden_conclusion"] = _normalize_mix_block(
        data.get("hidden_conclusion"), allowed_card_ids
    )
    data["recommended_actions"] = [
        action
        for item in _json_list(data.get("recommended_actions"))
        if (action := _normalize_mixer_display_sentence(item, 420))
    ][:3]
    data["action_details"] = _normalize_action_details(data.get("action_details"), allowed_card_ids)
    fallback_actions_applied = False
    if _needs_action_detail_fallback(data["action_details"]):
        data["action_details"] = _fallback_action_details_from_result(data)
        fallback_actions_applied = True
    detail_actions = _recommended_actions_from_details(data)
    if (
        fallback_actions_applied
        or not data["recommended_actions"]
        or any(_is_generic_action_text(action) for action in data["recommended_actions"])
    ):
        data["recommended_actions"] = detail_actions or data["recommended_actions"]
    data["confidence"] = confidence_in_range(data.get("confidence", 0.0))
    data["sources_used"] = _valid_card_refs(
        data.get("sources_used") or [c["id"] for c in cards],
        allowed_card_ids,
    )
    if not data["sources_used"]:
        data["sources_used"] = [c["id"] for c in cards]

    # Backward-compatible fields for the existing API/UI.
    data["insight"] = data["mix_insight"]
    data["final_one_liner"] = _normalize_mixer_display_sentence(
        data["hidden_conclusion"].get("finding") or data["mix_insight"],
        _MIXER_FINAL_ONE_LINER_MAX,
    )
    data["sk_ax_implication"] = _normalize_mixer_display_sentence(
        " ".join(data["recommended_actions"]),
        _MIXER_IMPLICATION_MAX,
    )
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

    data = _normalize_mixer_result_display_sentences(data)
    data["langfuse_trace_id"] = _get_langfuse_trace_id()
    data["warning"] = _warning_for(data)
    return data


def _normalize_mixer_result_display_sentences(result: dict) -> dict:
    result["mix_insight"] = _normalize_mixer_display_sentence(
        result.get("mix_insight") or result.get("insight", ""),
        _MIXER_FINAL_ONE_LINER_MAX,
    )
    result["insight"] = result["mix_insight"]
    for block_key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = result.get(block_key)
        if not isinstance(block, dict):
            continue
        block["finding"] = _normalize_mixer_display_sentence(block.get("finding", ""), 260)
        block["rationale"] = _normalize_mixer_display_sentence(
            block.get("rationale", ""),
            320,
        )
    actions = [
        action
        for item in _json_list(result.get("recommended_actions"))
        if (action := _normalize_mixer_display_sentence(item, 420))
    ]
    result["recommended_actions"] = _dedupe_keep_order(actions)
    for detail in _json_list(result.get("action_details")):
        if not isinstance(detail, dict):
            continue
        detail["action"] = _normalize_mixer_display_sentence(detail.get("action", ""), 420)
        detail["why"] = _normalize_mixer_display_sentence(detail.get("why", ""), 360)
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    result["final_one_liner"] = _normalize_mixer_display_sentence(
        result.get("final_one_liner") or hidden.get("finding") or result["mix_insight"],
        _MIXER_FINAL_ONE_LINER_MAX,
    )
    result["sk_ax_implication"] = _normalize_mixer_display_sentence(
        result.get("sk_ax_implication") or " ".join(result["recommended_actions"]),
        _MIXER_IMPLICATION_MAX,
    )
    result["bullet_signals"] = [
        finding
        for finding in (
            _dict_or_empty(result.get("common_pattern")).get("finding"),
            _dict_or_empty(result.get("comparison_point")).get("finding"),
            _dict_or_empty(result.get("hidden_conclusion")).get("finding"),
        )
        if finding
    ]
    return result


def _mixer_sentence_quality_issues(result: dict) -> list[str]:
    checks: list[tuple[str, object]] = [
        ("mix_insight", result.get("mix_insight")),
        ("final_one_liner", result.get("final_one_liner")),
        ("sk_ax_implication", result.get("sk_ax_implication")),
    ]
    for block_key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = result.get(block_key)
        if not isinstance(block, dict):
            checks.append((block_key, ""))
            continue
        checks.append((f"{block_key}.finding", block.get("finding")))
        checks.append((f"{block_key}.rationale", block.get("rationale")))
    for index, detail in enumerate(_json_list(result.get("action_details"))):
        if not isinstance(detail, dict):
            continue
        checks.append((f"action_details.{index}.action", detail.get("action")))
        checks.append((f"action_details.{index}.why", detail.get("why")))

    issues: list[str] = []
    for label, raw_text in checks:
        text = str(raw_text or "").strip()
        if not text:
            issues.append(f"{label}:empty")
            continue
        if "…" in text or "..." in text:
            issues.append(f"{label}:ellipsis")
            continue
        if not _is_complete_display_sentence(text):
            issues.append(f"{label}:incomplete")
    return issues[:12]


def _card_one_liner(card: dict) -> str:
    """카드 1장의 핵심을 한 문장으로 추출 — per_card 추론 단계 입력."""
    linked = _linked_results_from_card(card)
    analysis = _component(linked, "analysis")
    integrated = _component(linked, "integrated_issue")
    candidates: list[object] = [
        analysis.get("strategic_meaning"),
        analysis.get("market_signal"),
        integrated.get("main_issue"),
    ]
    summary_lines = card.get("summary_lines") or []
    if isinstance(summary_lines, str):
        try:
            summary_lines = json.loads(summary_lines)
        except json.JSONDecodeError:
            summary_lines = [summary_lines]
    if isinstance(summary_lines, list):
        candidates.extend(summary_lines)
    candidates.append(card.get("title"))
    for cand in candidates:
        if cand and str(cand).strip():
            return clip_string(str(cand).strip(), 160)
    return ""


def _build_reasoning_trail(result: dict, cards: list[dict]) -> list[dict]:
    """blocks → per-step 요약 trail (결정적). LLM 호출 없이 근거 기반으로 구성."""
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    mix_insight = result.get("mix_insight") or result.get("insight") or ""
    all_ids = [str(c["id"]) for c in cards if c.get("id")]
    trail: list[dict] = []

    def add(label: str, one_liner: object, refs: list[str]) -> None:
        text = str(one_liner or "").strip()
        if not text:
            return
        trail.append(
            {
                "seq": len(trail) + 1,
                "label": label,
                "one_liner": clip_string(text, 160),
                "evidence_refs": refs or all_ids,
                "langfuse_observation_id": None,
            }
        )

    add("공통 패턴", common.get("finding"), _json_list(common.get("evidence_card_ids")))
    add("비교 포인트", comparison.get("finding"), _json_list(comparison.get("evidence_card_ids")))
    add("숨은 결론", hidden.get("finding"), _json_list(hidden.get("evidence_card_ids")))
    add("믹스 인사이트", mix_insight, all_ids)
    return trail


def _build_reasoning_steps(result: dict, cards: list[dict]) -> list[dict]:
    """per_card → cross_card → synthesis 3-phase CoT (결정적, 근거 기반)."""
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    mix_insight = result.get("mix_insight") or result.get("insight") or ""
    all_ids = [str(c["id"]) for c in cards if c.get("id")]
    confidence = round(confidence_in_range(result.get("confidence", 0.0)), 2)
    steps: list[dict] = []

    def push(phase: str, question: str, inputs_used: list[str], answer: str, concl: str) -> None:
        if not str(answer or "").strip():
            return
        steps.append(
            {
                "step_idx": len(steps),
                "phase": phase,
                "question": question,
                "inputs_used": inputs_used or all_ids,
                "answer": clip_string(str(answer).strip(), 220),
                "intermediate_conclusion": clip_string(str(concl or "").strip(), 220),
                "confidence": confidence,
                "langfuse_observation_id": None,
            }
        )

    # per_card — 카드별 핵심 관찰
    for c in cards:
        cid = str(c.get("id") or "")
        if not cid:
            continue
        one_liner = _card_one_liner(c)
        if not one_liner:
            continue
        peer = str(c.get("company") or c.get("peer_id") or "").strip()
        push(
            "per_card",
            f"[{peer or cid}] 이 카드는 무엇을 말하는가?",
            [cid],
            one_liner,
            "",
        )

    # cross_card — 공통 흐름 + 차이
    cross_answer = " / ".join(
        str(x) for x in [common.get("finding"), comparison.get("finding")] if str(x or "").strip()
    )
    cross_concl = " ".join(
        str(x)
        for x in [common.get("rationale"), comparison.get("rationale")]
        if str(x or "").strip()
    )
    cross_inputs = _dedupe_keep_order(
        [
            *_json_list(common.get("evidence_card_ids")),
            *_json_list(comparison.get("evidence_card_ids")),
        ]
    )
    push(
        "cross_card",
        "여러 카드를 함께 보면 어떤 공통 흐름과 차이가 보이는가?",
        cross_inputs,
        cross_answer,
        cross_concl,
    )

    # synthesis — 숨은 결론 → 믹스 인사이트
    push(
        "synthesis",
        "함께 봐야 보이는 판단 기준의 변화는 무엇인가?",
        _json_list(hidden.get("evidence_card_ids")),
        hidden.get("finding") or mix_insight,
        hidden.get("rationale") or "",
    )
    return steps


def _build_follow_up_questions(result: dict, cards: list[dict]) -> list[str]:
    """Backward-compatible question list derived from structured follow-up checks."""
    return [
        str(item.get("question"))
        for item in _build_follow_up_checks(result, cards)
        if isinstance(item, dict) and str(item.get("question") or "").strip()
    ][:3]


def _build_follow_up_checks(result: dict, cards: list[dict]) -> list[dict]:
    """다음 분석 질문과 현재 근거만으로 답할 수 있는 초안을 함께 생성."""
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
    common = _dict_or_empty(result.get("common_pattern"))
    peers = _dedupe_keep_order(
        [
            str(c.get("company") or c.get("peer_id") or "").strip()
            for c in cards
            if str(c.get("company") or c.get("peer_id") or "").strip()
        ]
    )
    peer_phrase = "와 ".join(peers[:3]) if peers else "선택한 Peer"
    checks: list[dict] = []
    if str(hidden.get("finding") or "").strip():
        hidden_refs = _json_list(hidden.get("evidence_card_ids"))
        hidden_answer = (
            f"현재 근거만 보면 이 신호는 {len(hidden_refs) or len(cards)}개 카드에서 연결되는 "
            "반복 판단 기준으로 보는 편이 타당합니다. "
            f"{str(hidden.get('rationale') or hidden.get('finding') or '').strip()}"
        )
        checks.append(
            {
                "question": (
                    f"‘{clip_string(str(hidden['finding']).strip(), 70)}’ 신호가 "
                    "일회성 이벤트인지 반복 신호인지 확인할 후속 근거는 무엇인가?"
                ),
                "answer": clip_string(hidden_answer, 260),
                "purpose": (
                    "숨은 결론이 단일 카드 해석이 아니라 반복되는 시장 판단 기준인지 "
                    "검증하기 위한 확인 포인트입니다."
                ),
                "evidence_refs": hidden_refs,
            }
        )
    if str(comparison.get("finding") or "").strip():
        comparison_answer = (
            f"현재 답은 {peer_phrase}가 같은 흐름 안에서도 서로 다른 성과 기준이나 적용 맥락을 "
            "앞세운다는 점입니다. "
            f"{str(comparison.get('rationale') or comparison.get('finding') or '').strip()}"
        )
        checks.append(
            {
                "question": (
                    f"{peer_phrase}의 서로 다른 접근 중 어느 고객군·업무 맥락에 "
                    "먼저 적용할 수 있는 차이인가?"
                ),
                "answer": clip_string(comparison_answer, 260),
                "purpose": (
                    "비교 포인트가 단순 회사별 차이가 아니라 고객군 선택이나 오퍼링 "
                    "우선순위로 이어질 수 있는지 판단하기 위한 질문입니다."
                ),
                "evidence_refs": _json_list(comparison.get("evidence_card_ids")),
            }
        )
    if _json_list(result.get("recommended_action_basis")) or _json_list(
        result.get("recommended_actions")
    ):
        action_text = "; ".join(
            clip_implication(str(action))
            for action in _json_list(result.get("recommended_actions"))[:2]
            if str(action or "").strip()
        )
        basis_text = "; ".join(
            clip_implication(str(basis))
            for basis in _json_list(result.get("recommended_action_basis"))[:2]
            if str(basis or "").strip()
        )
        action_focus = action_text or basis_text or "추천 액션의 우선순위를 실행 조건별로 나누는 것"
        action_answer = (
            f"현재 답은 {action_focus}입니다. "
            "실행 전에는 고객군, 적용 업무, 수익화 수치, 리스크 게이트를 "
            "분리해 우선순위를 정해야 합니다."
        )
        checks.append(
            {
                "question": (
                    "대응 방향을 실행 판단으로 바꾸려면 어떤 수치·고객·리스크 "
                    "조건이 추가로 필요한가?"
                ),
                "answer": clip_string(action_answer, 280),
                "purpose": (
                    "권고가 선언으로 끝나지 않고 투자, 파트너십, 리스크 게이트 결정으로 "
                    "이어지려면 부족한 근거를 분리해야 합니다."
                ),
                "evidence_refs": _dedupe_keep_order(
                    [
                        *_json_list(common.get("evidence_card_ids")),
                        *_json_list(comparison.get("evidence_card_ids")),
                        *_json_list(hidden.get("evidence_card_ids")),
                    ]
                ),
            }
        )
    seen: set[str] = set()
    result_checks: list[dict] = []
    for check in checks:
        question = str(check.get("question") or "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        result_checks.append(check)
    return result_checks[:3]


def _build_analysis_depth(result: dict, cards: list[dict], analysis_mode: str) -> dict:
    del result
    card_count = len(cards)
    if analysis_mode == "deep":
        return {
            "mode": "deep",
            "label": "정확 분석",
            "summary": (
                f"선택 카드 {card_count}장을 최신 고정밀 모델로 1차 믹스 분석한 뒤 문장 품질 보강, "
                "믹스 단위 시사점 보강, 레이더 축 해석 정교화, "
                "단계별 근거 재구성을 추가로 수행했습니다."
            ),
            "included_steps": [
                f"{_model_for_mode('deep')} 기반 LLM 1차 믹스 분석",
                "공통 패턴·비교 포인트·숨은 결론 품질 보강",
                "믹스 단위 ImplicationAgent 보강",
                "6축 레이더별 본문 해석 질문과 답변 정교화",
                "고객군·오퍼링·자원 배분·파트너십·리스크 게이트별 실행안 보강",
                "근거 카드와 실행 조건 상세 정리",
            ],
            "omitted_steps": [],
        }
    return {
        "mode": "quick",
        "label": "빠른 실행",
        "summary": (
            f"선택 카드 {card_count}장의 핵심 연결만 빠르게 산출했습니다. "
            "정확 분석보다 짧게 끝나도록 고정밀 모델 재검증과 추가 품질 보강 호출은 생략합니다."
        ),
        "included_steps": [
            f"{_model_for_mode('quick')} 기반 LLM 1차 믹스 분석",
            "기본 근거 연결",
            "기본 대응 방향 정리",
        ],
        "omitted_steps": [
            f"{_model_for_mode('deep')} 기반 고정밀 재검증",
            "문장 품질 보강",
            "믹스 단위 ImplicationAgent 보강",
            "6축 레이더별 본문 해석 정교화",
        ],
    }


def _build_deep_dive_sections(result: dict, cards: list[dict]) -> list[dict]:
    card_lookup = {str(card.get("id") or ""): card for card in cards}
    sections: list[dict] = []

    def evidence_refs(block: dict) -> list[str]:
        refs = [str(item) for item in _json_list(block.get("evidence_card_ids")) if str(item)]
        if refs:
            return _dedupe_keep_order(refs)
        return _dedupe_keep_order(
            [
                str(item.get("card_id") or "")
                for item in _json_list(block.get("evidence"))
                if isinstance(item, dict) and str(item.get("card_id") or "")
            ]
        )

    def evidence_summary(refs: list[str]) -> str:
        lines: list[str] = []
        for ref in refs[:5]:
            card = card_lookup.get(ref, {})
            title = str(card.get("title") or ref).strip()
            company = str(card.get("company") or card.get("peer_id") or "").strip()
            prefix = f"{company}: " if company else ""
            lines.append(f"{prefix}{title}")
        return "\n".join(lines)

    for key, title in (
        ("common_pattern", "공통 패턴 상세 검증"),
        ("comparison_point", "비교 포인트 상세 검증"),
        ("hidden_conclusion", "숨은 결론 상세 검증"),
    ):
        block = _dict_or_empty(result.get(key))
        finding = str(block.get("finding") or "").strip()
        rationale = str(block.get("rationale") or "").strip()
        refs = evidence_refs(block)
        if not (finding or rationale or refs):
            continue
        sections.append(
            {
                "title": title,
                "summary": finding,
                "details": [
                    {"label": "핵심 판단", "text": finding, "evidence_refs": refs},
                    {"label": "판단 근거", "text": rationale, "evidence_refs": refs},
                    {"label": "참조 카드", "text": evidence_summary(refs), "evidence_refs": refs},
                ],
            }
        )

    action_details = [
        item for item in _json_list(result.get("action_details")) if isinstance(item, dict)
    ]
    detail_items: list[dict] = []
    for item in action_details[:5]:
        refs = [str(ref) for ref in _json_list(item.get("evidence_card_ids")) if str(ref)]
        action = str(item.get("action") or "").strip()
        why = str(item.get("why") or "").strip()
        use_case = str(item.get("use_case") or "").strip()
        text = "\n".join(
            part
            for part in (
                f"실행안: {action}" if action else "",
                f"판단 이유: {why}" if why else "",
                f"적용 맥락: {use_case}" if use_case else "",
            )
            if part
        )
        if text:
            detail_items.append(
                {"label": use_case or "실행 조건", "text": text, "evidence_refs": refs}
            )
    if detail_items:
        sections.append(
            {
                "title": "대응 방향 상세",
                "summary": "추천 액션을 실행 조건, 적용 맥락, 참조 근거 기준으로 분해했습니다.",
                "details": detail_items,
            }
        )

    return sections[:4]


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
        response = _get_llm("deep").invoke(
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
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
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
    common = _dict_or_empty(result.get("common_pattern"))
    comparison = _dict_or_empty(result.get("comparison_point"))
    hidden = _dict_or_empty(result.get("hidden_conclusion"))
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


def _recommended_actions_from_implication(
    implication: dict, *, result: dict, limit: int = 3
) -> list[str]:
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
    return grounded[:limit]


def _recommended_actions_from_basis(result: dict, *, limit: int = 3) -> list[str]:
    actions: list[str] = []
    for item in _json_list(result.get("recommended_action_basis")):
        text_value = str(item or "").strip()
        if not text_value:
            continue
        if not text_value.endswith(("다.", "요.", ".")):
            text_value = f"{text_value}."
        actions.append(clip_implication(text_value))
    return _dedupe_keep_order(actions)[:limit]


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
    provenance_value = data.get("provenance")
    provenance = provenance_value if isinstance(provenance_value, dict) else {}
    quality_flags = _json_list(data.get("quality_flags") or provenance.get("quality_flags"))
    if confidence < 0.6 or quality_flags:
        warnings.append(
            "일부 카드의 통합 분석 연결이 제한되어 확인 가능한 카드 요약과 "
            "연결 근거를 중심으로 산출했습니다."
        )
    weak_blocks: list[str] = []
    for key in ("common_pattern", "comparison_point", "hidden_conclusion"):
        block = _dict_or_empty(data.get(key))
        if not block.get("finding") or len(_json_list(block.get("evidence_card_ids"))) < 2:
            weak_blocks.append(key)
    if weak_blocks:
        warnings.append(
            "일부 해석 단계는 근거 카드 연결이 적어 결과 화면의 참조 근거를 함께 확인해야 합니다."
        )
    if not data.get("recommended_actions"):
        warnings.append("대응 방향 생성 결과가 비어 있어 원문 근거 확인이 필요합니다.")
    return " ".join(_dedupe_keep_order(warnings)) if warnings else None


def _error_response(
    short_reason: str,
    detail: str,
    card_ids: list[str],
    confidence: float = 0.0,
    integrated_issue_ids: list[str] | None = None,
) -> dict:
    log.warning("Mixer error | %s | detail=%s | ids=%s", short_reason, detail, card_ids)
    issue_ids = integrated_issue_ids or []
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
        "source_integrated_issue_ids": issue_ids,
        "peer_ids": [],
        "langfuse_trace_id": None,
        "warning": f"{short_reason} — {detail}",
        "provenance": {
            "llm_model": _LLM_MODEL,
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": card_ids,
            "source_integrated_issue_ids": issue_ids,
            "quality_flags": [],
            "error": short_reason,
        },
    }


class MixerAgent(MixerAnalysisAgent):
    """Architecture-facing name for the 2단계 mixer agent."""
