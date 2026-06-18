# 작성일: 2026-06-14
# 작성자: 최종민
# 변경이력:
#   2026-06-14 최종민 — 공용 LLM 클라이언트 팩토리 패키지 신설
"""공용 LLM 클라이언트 팩토리 (계층화·재사용 — refactoring-architecture §2.3 R1).

axis-ai 전반에 _get_llm 이 19곳 중복돼 있고, 특히 gpt-5 계열의
reasoning_effort 분기와 json_object 래핑이 mixer/today_insight/briefing 등에
복붙돼 있다. 이 패키지는 그 구성 로직을 단일 출처로 모은다.

설계 원칙:
- **순수 빌더**: 전역 캐시를 두지 않는다. 각 호출처는 기존 `_llm` 싱글톤/캐시를
  그대로 유지하고, ChatOpenAI 생성 부분만 build_chat_llm 으로 치환한다
  (동작 불변 + 점진 이행).
- **leaf 레이어**: langchain_openai 외 axis 내부 모듈에 의존하지 않는다.
"""

from src.llm.factory import LLMSpec, build_chat_llm, is_reasoning_model

__all__ = ["LLMSpec", "build_chat_llm", "is_reasoning_model"]
