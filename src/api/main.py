# 작성일: 2026-04-21
# 작성자: 최종민
# 변경이력:
#   2026-04-21 최종민 — uvicorn 진입점 main.py 추가
from src.api.router import app  # noqa: F401  # uvicorn src.api.main:app 진입점
