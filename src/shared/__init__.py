# 작성일: 2026-06-14
# 작성자: 최종민
# 변경이력:
#   2026-06-14 최종민 — shared 패키지 추가, JSON 헬퍼를 단일 출처로 통합
"""교차 모듈 공용 유틸 (refactoring-architecture §2.3 R2).

agents/services/preprocessing 등 여러 모듈에 중복된 순수 헬퍼의 단일 출처.
leaf 레이어 — axis 내부 상위 모듈에 의존하지 않는다.
"""
