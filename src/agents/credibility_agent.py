"""신뢰도 분류 에이전트"""

import logging
from urllib.parse import urlparse

log = logging.getLogger(__name__)

DISCLOSURE_DOMAINS = [
    "dart.fss.or.kr",
]

OFFICIAL_DOMAINS = [
    "samsungsds.com",
    "lgcns.com",
    "skax.co.kr",
    "skcc.co.kr",
    "poscodx.com",
    "hyundai-autoever.com",
]

RECRUITMENT_DOMAINS = [
    "saramin.co.kr",
    "jobkorea.co.kr",
    "wanted.co.kr",
    "jumpit.co.kr",
    "programmers.co.kr",
    "rocketpunch.com",
    "linkedin.com",
    "greenhouse.io",
    "lever.co",
]

MAJOR_MEDIA_DOMAINS = [
    "yna.co.kr",
    "yonhapnewstv.co.kr",
    "hankyung.com",
    "mk.co.kr",
    "sedaily.com",
    "fnnews.com",
    "newsis.com",
    "newspim.com",
]

TECH_MEDIA_DOMAINS = [
    "etnews.com",
    "zdnet.co.kr",
    "ddaily.co.kr",
    "digitaltoday.co.kr",
    "it.chosun.com",
    "bloter.net",
    "inews24.com",
    "ciokorea.com",
]

PLATFORM_DOMAINS = [
    "news.naver.com",
    "n.news.naver.com",
    "news.google.com",
]


class CredibilityAgent:
    def classify(self, url: str, source_name: str | None = None) -> str:
        return self.classify_detail(url, source_name)["credibility"]

    def classify_detail(self, url: str, source_name: str | None = None) -> dict[str, str]:
        domain = self._extract_domain(url)
        normalized_source = (source_name or "").lower()

        if self._is_disclosure(domain, normalized_source):
            return {
                "credibility": "High",
                "source_type": "disclosure",
                "domain": domain,
                "reason": "official_disclosure",
            }

        if self._is_official(domain, normalized_source):
            return {
                "credibility": "High",
                "source_type": "official",
                "domain": domain,
                "reason": "company_official_source",
            }

        if self._is_recruitment(domain, normalized_source):
            return {
                "credibility": "Medium",
                "source_type": "recruitment",
                "domain": domain,
                "reason": "recruitment_signal",
            }

        if self._matches_domain(domain, MAJOR_MEDIA_DOMAINS):
            return {
                "credibility": "High",
                "source_type": "major_media",
                "domain": domain,
                "reason": "major_media_source",
            }

        if self._matches_domain(domain, TECH_MEDIA_DOMAINS):
            return {
                "credibility": "Medium",
                "source_type": "tech_media",
                "domain": domain,
                "reason": "tech_media_source",
            }

        if self._matches_domain(domain, PLATFORM_DOMAINS):
            return {
                "credibility": "Medium",
                "source_type": "platform",
                "domain": domain,
                "reason": "news_platform_not_original_source",
            }

        return {
            "credibility": "Unverified",
            "source_type": "unknown",
            "domain": domain,
            "reason": "unregistered_source",
        }

    def _is_disclosure(self, domain: str, source_name: str) -> bool:
        return (
            self._matches_domain(domain, DISCLOSURE_DOMAINS)
            or "dart" in source_name
            or "공시" in source_name
        )

    def _is_official(self, domain: str, source_name: str) -> bool:
        return (
            self._matches_domain(domain, OFFICIAL_DOMAINS)
            or "official" in source_name
            or "homepage" in source_name
            or "newsroom" in source_name
            or "홈페이지" in source_name
            or "뉴스룸" in source_name
        )

    def _is_recruitment(self, domain: str, source_name: str) -> bool:
        return (
            self._matches_domain(domain, RECRUITMENT_DOMAINS)
            or "job" in source_name
            or "career" in source_name
            or "recruit" in source_name
            or "채용" in source_name
        )

    def _extract_domain(self, url: str) -> str:
        parsed = urlparse(url or "")
        domain = parsed.netloc or parsed.path
        return domain.lower().replace("www.", "")

    def _matches_domain(self, domain: str, candidates: list[str]) -> bool:
        return any(
            domain == candidate or domain.endswith(f".{candidate}")
            for candidate in candidates
        )