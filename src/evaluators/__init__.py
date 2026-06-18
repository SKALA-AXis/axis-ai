# 작성일: 2026-06-02
# 작성자: 박지원
# 변경이력:
#   2026-06-02 박지원 — 분석 파이프라인 에이전트 재구성 및 뉴스 통합 플로우 개선
#   2026-06-09 최종민 — evaluators eager import 제거 및 cron 이미지 CI 스모크 추가
"""Rule-based evaluator package.

Import submodules directly (e.g. ``from src.evaluators.evaluator import Evaluator``).
Avoid eager imports here — axis-ai-cron imports ``llm_judge_prompts`` only and must
not pull ``src.analysis`` (not shipped in the slim cron image).
"""
