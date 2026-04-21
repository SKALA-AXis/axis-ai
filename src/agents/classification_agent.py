"""중요도 분류 에이전트 — GPT-4o 호출"""
import logging
import time

log = logging.getLogger(__name__)


class ClassificationAgent:
    def classify(self, cluster_id: int, articles: list[dict]) -> dict:
        """중요도 분류 (긴급/주목/참고).

        Returns:
            {'importance': str, 'importance_score': float, 'event_type': str}
        """
        start = time.time()
        log.info("중요도 분류 시작 | cluster_id=%d", cluster_id)
        # TODO: GPT-4o 호출로 5개 축 가중치 계산
        result = {"importance": "reference", "importance_score": 0.3, "event_type": "tech"}
        elapsed = int((time.time() - start) * 1000)
        log.info(
            "중요도 분류 완료 | cluster_id=%d importance=%s elapsed=%dms",
            cluster_id,
            result["importance"],
            elapsed,
        )
        return result
