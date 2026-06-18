# 작성일: 2026-06-02
# 작성자: 박지원
# 변경이력:
#   2026-06-02 박지원 — 분석 파이프라인 에이전트 재구성 및 뉴스 통합 플로우 개선
#   2026-06-18 박지원 — LLM judge sidecar 제거에 맞춰 evaluator 패키지 설명 정리
"""Rule-based evaluator package.

Import submodules directly (e.g. ``from src.evaluators.evaluator import Evaluator``).
Avoid eager imports here so pipeline users do not pull optional dependencies at
module import time.
"""
