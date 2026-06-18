# 작성일: 2026-05-21
# 작성자: 최종민
# 변경이력:
#   2026-05-21 최종민 — CronJob 엔트리포인트용 scripts/ 패키지 마커 추가 (Docker 이미지 포함 대응)
"""Operational entrypoint scripts for axis-ai CronJobs and one-off jobs.

`python -m scripts.<name>` 형식으로 실행되는 모듈을 정상 인식시키기 위한 패키지 마커.
일반 `python scripts/<name>.py` 실행에는 영향 없음.
"""
