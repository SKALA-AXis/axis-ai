"""크롤러 에이전트 - 뉴스·공시·채용공고 원천 수집"""

import logging

from src.crawler.base_crawler import BaseCrawler, RawArticle
from src.crawler.dart_crawler import DartCrawler
from src.crawler.job_crawler import JobCrawler
from src.crawler.naver_crawler import NaverNewsCrawler
from src.crawler.rss_crawler import RssCrawler

log = logging.getLogger(__name__)

DEFAULT_TOPICS = [
    "AI",
    "인공지능",
    "생성형 AI",
    "LLM",
    "AI 에이전트",
    "에이전틱 AI",
    "AX",
    "AI 전환",
    "디지털 전환",
    "DX",
    "클라우드",
    "Cloud",
    "데이터센터",
    "GPU",
    "IT서비스",
    "IT 서비스",
    "IT 운영",
    "SI",
    "시스템 통합",
    "자동화",
    "보안",
    "금융 인프라",
    "제조 AX",
]

PEER_PROFILES = {
    "samsung_sds": {
        "aliases": ["삼성SDS", "삼성에스디에스"],
        "dart_names": ["삼성에스디에스"],
        "topics": DEFAULT_TOPICS,
    },
    "lg_cns": {
        "aliases": ["LG CNS", "엘지씨엔에스"],
        "dart_names": ["엘지씨엔에스"],
        "topics": DEFAULT_TOPICS,
    },
    "sk_ax": {
        "aliases": ["SK AX", "SK C&C", "SK㈜ C&C", "SK주식회사 C&C", "에스케이씨앤씨"],
        "dart_names": ["에스케이"],
        "topics": DEFAULT_TOPICS,
    },
    "posco_dx": {
        "aliases": ["포스코DX", "포스코 디엑스", "POSCO DX", "포스코ICT", "POSCO ICT"],
        "dart_names": ["포스코DX", "포스코디엑스"],
        "topics": DEFAULT_TOPICS,
    },
}


class CrawlerAgent:
    def __init__(
        self,
        peer_id: str,
        topics: list[str] | None = None,
        corp_code: str | None = None,
        recent_days: int = 1,
    ):
        self.peer_id = peer_id
        profile = PEER_PROFILES.get(peer_id, {"aliases": [peer_id], "topics": DEFAULT_TOPICS})
        self.aliases = profile["aliases"]
        self.dart_names = profile.get("dart_names", [])
        self.topics = topics or profile["topics"]
        self.corp_code = corp_code
        self.recent_days = recent_days

    async def collect(self) -> list[RawArticle]:
        """크롤러 원천 결과를 최대한 그대로 수집한다."""
        log.info(
            "크롤링 시작 | peer_id=%s aliases=%s topics=%s recent_days=%d",
            self.peer_id,
            self.aliases,
            self.topics,
            self.recent_days,
        )

        crawlers: list[BaseCrawler] = [
            RssCrawler(peer_id=self.peer_id, aliases=self.aliases),
            NaverNewsCrawler(peer_id=self.peer_id, aliases=self.aliases),
            JobCrawler(peer_id=self.peer_id),
            DartCrawler(
                peer_id=self.peer_id,
                corp_code=self.corp_code,
                corp_names=self.dart_names,
            ),
        ]

        articles: list[RawArticle] = []

        for crawler in crawlers:
            try:
                crawler_articles = await crawler.crawl()
                articles.extend(crawler_articles)
                log.info(
                    "크롤러 실행 완료 | peer_id=%s crawler=%s articles=%d",
                    self.peer_id,
                    crawler.__class__.__name__,
                    len(crawler_articles),
                )
            except Exception as e:
                log.error(
                    "크롤러 실행 실패 | peer_id=%s crawler=%s error=%s",
                    self.peer_id,
                    crawler.__class__.__name__,
                    e,
                )

        return articles

    async def run(self) -> list[RawArticle]:
        """하위 호환용 실행 함수.

        관련성 필터링은 RelevanceAgent에서 수행한다.
        """
        return await self.collect()


def get_peer_profile(peer_id: str) -> dict:
    return PEER_PROFILES.get(peer_id, {"aliases": [peer_id], "topics": DEFAULT_TOPICS})


def get_other_peer_aliases(peer_id: str) -> list[str]:
    aliases: list[str] = []

    for other_peer_id, profile in PEER_PROFILES.items():
        if other_peer_id != peer_id:
            aliases.extend(profile["aliases"])

    return aliases