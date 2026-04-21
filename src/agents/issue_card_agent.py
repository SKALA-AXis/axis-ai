"""이슈 카드 생성 에이전트 — GPT-4o 호출"""
import logging

log = logging.getLogger(__name__)


class IssueCardAgent:
    def generate(self, cluster: dict) -> dict:
        """이슈 카드를 생성한다.

        Args:
            cluster: 클러스터 정보 (대표 기사 포함).

        Returns:
            IssueCard dict.
        """
        log.info("이슈 카드 생성 | cluster_id=%s", cluster.get("cluster_id"))
        # TODO: GPT-4o로 제목, 3줄 요약 생성
        return {}
