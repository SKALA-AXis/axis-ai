"""시사점 생성 에이전트 — GPT-4o 호출"""
import logging

log = logging.getLogger(__name__)


class ImplicationAgent:
    def generate(self, issue_card: dict) -> dict:
        """SK AX 관점의 시사점 초안을 생성한다."""
        log.info("시사점 생성 | card_id=%s", issue_card.get("id"))
        # TODO: GPT-4o temperature=0.7 호출
        return {
            "why_important": "",
            "potential_impact": "",
            "follow_up_questions": [],
            "suggested_actions": [],
            "confidence": 0.0,
        }
