"""알림 에이전트 — Slack Webhook 발송"""
import logging
import os

import httpx

log = logging.getLogger(__name__)


class NotificationAgent:
    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

    def send(self, text: str) -> bool:
        """Slack으로 메시지를 발송한다."""
        if not self.webhook_url:
            log.warning("Slack webhook URL 미설정")
            return False
        try:
            resp = httpx.post(self.webhook_url, json={"text": text}, timeout=5)
            resp.raise_for_status()
            return True
        except Exception as e:
            log.error("Slack 발송 실패: %s", e)
            return False
