"""신뢰도 분류 에이전트"""
import logging

log = logging.getLogger(__name__)

CREDIBILITY_TIERS = {
    "tier1": ["news.naver.com", "yonhap.co.kr"],
    "tier2": ["zdnet.co.kr", "etnews.com", "itchosun.com"],
    "tier3": ["samsungsds.com", "lgcns.com"],
}


class CredibilityAgent:
    def classify(self, url: str) -> str:
        """출처 URL 기반 신뢰도 분류.

        Returns:
            'High' | 'Medium' | 'Low' | 'Unverified'
        """
        domain = url.split("/")[2] if "//" in url else url
        for tier, domains in CREDIBILITY_TIERS.items():
            if any(d in domain for d in domains):
                return "High" if tier in ("tier1", "tier3") else "Medium"
        return "Unverified"
