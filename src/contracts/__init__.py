"""레이어 중립 공용 계약 (Pydantic 모델).

api 와 agents 가 함께 쓰는 요청/응답 스키마의 정주소. agents 가 src.api 를
임포트하던 순환 의존(2-A1)을 끊기 위해 도입 — src/api/*_schemas.py 는
기존 임포트 호환용 re-export shim 으로 유지된다.
"""
