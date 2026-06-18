# 작성일: 2026-04-27
# 작성자: 최종민
# 변경이력:
#   2026-04-27 최종민 — v3 전환(Evidence Chain)으로 약한 신호 감지 에이전트 추가
"""약한 신호 감지 에이전트 — 채용공고 패턴 분석"""

import logging

log = logging.getLogger(__name__)


class WeakSignalAgent:
    def detect(self, peer_id: str) -> list[dict]:
        """채용공고 급증·신규 직군 패턴으로 약한 신호를 감지한다."""
        log.info("약한 신호 감지 | peer_id=%s", peer_id)
        # TODO: 채용공고 4주 이동평균 대비 이번 주 3배 초과 감지
        return []
