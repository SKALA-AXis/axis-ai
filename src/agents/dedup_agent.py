"""중복 제거 및 클러스터링 에이전트"""

import logging

log = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.90


class DeduplicationAgent:
    def deduplicate(self, article_ids: list[int]) -> dict:
        """코사인 유사도 기반 중복 제거 및 클러스터링.

        Args:
            article_ids: PostgreSQL article ID 목록.

        Returns:
            {cluster_id: [article_ids]} 형태의 클러스터 맵.
        """
        log.info("중복 제거 | articles=%d", len(article_ids))
        # TODO: BGE-M3 임베딩 후 코사인 유사도 >= 0.90 클러스터링
        return {i: [aid] for i, aid in enumerate(article_ids)}
