"""ChatOpenAI 생성 단일 출처.

각 호출처가 제각각 반복하던 구성(temperature·max_completion_tokens·json_object
래핑·gpt-5 reasoning_effort 분기)을 LLMSpec 한 곳으로 모은다. 동작 불변을 위해
호출처별 값(model·온도·캡)은 전부 파라미터로 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


def is_reasoning_model(model: str) -> bool:
    """gpt-5 계열(추론 모델) 여부.

    추론 모델은 reasoning 토큰이 max_completion_tokens 안에서 소비되고
    reasoning_effort 파라미터를 받는다 (mixer/today_insight/briefing 공통 분기).
    """
    return str(model).startswith("gpt-5")


@dataclass(frozen=True)
class LLMSpec:
    """LLM 한 개의 생성 사양.

    Attributes:
        model: 모델 이름 (예: "gpt-4o", "gpt-5.5", "gpt-4o-mini").
        temperature: 샘플링 온도.
        max_tokens: 비추론 모델(gpt-4o 등)의 출력 토큰 캡.
        max_tokens_reasoning: gpt-5 계열 캡. None 이면 max_tokens 사용.
            (추론 토큰 headroom 때문에 보통 더 크게 잡는다.)
        json_object: True 면 response_format=json_object 강제.
        reasoning_effort: gpt-5 계열에만 적용 ("low"|"medium"|"high").
            비추론 모델에는 전달하지 않는다.
    """

    model: str
    temperature: float = 0.2
    max_tokens: int = 2000
    max_tokens_reasoning: int | None = None
    json_object: bool = False
    reasoning_effort: str = "low"

    def resolved_max_tokens(self) -> int:
        if is_reasoning_model(self.model) and self.max_tokens_reasoning is not None:
            return self.max_tokens_reasoning
        return self.max_tokens


def build_chat_llm(spec: LLMSpec) -> ChatOpenAI:
    """LLMSpec 으로 ChatOpenAI 를 생성한다 (캐싱 없음 — 호출처가 캐싱 담당).

    gpt-5 계열이면 reasoning_effort 를 자동 적용하고, 그 외에는 전달하지 않는다.
    """
    from langchain_openai import ChatOpenAI  # lazy: transformers 풀체인 임포트 회피

    kwargs: dict = {
        "model": spec.model,
        "temperature": spec.temperature,
        "max_completion_tokens": spec.resolved_max_tokens(),
    }
    if spec.json_object:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    if is_reasoning_model(spec.model):
        kwargs["reasoning_effort"] = spec.reasoning_effort
    return ChatOpenAI(**kwargs)
