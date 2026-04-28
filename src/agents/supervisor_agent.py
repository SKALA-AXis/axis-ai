"""수집 파이프라인을 오케스트레이션하는 Supervisor Agent."""

from __future__ import annotations

import logging
from typing import TypedDict

from src.agents.classification_agent import ClassificationAgent
from src.agents.crawler_agent import CrawlerAgent
from src.agents.credibility_agent import CredibilityAgent
from src.agents.dedup_agent import DeduplicationAgent
from src.agents.implication_agent import ImplicationAgent
from src.agents.issue_card_agent import IssueCardAgent
from src.agents.relevance_agent import RelevanceAgent
from src.agents.validation_agent import ValidationAgent
from src.crawler.base_crawler import RawArticle

log = logging.getLogger(__name__)


class SupervisorResult(TypedDict):
    raw_articles: list[RawArticle]
    relevant_articles: list[RawArticle]
    credible_articles: list[RawArticle]
    cluster_map: dict[int, list[int]]
    classified_clusters: list[dict]
    issue_cards: list[dict]
    implications: list[dict]
    validation_results: list[dict]


class SupervisorAgent:
    def __init__(
        self,
        recent_days: int = 1,
        crawler_cls: type[CrawlerAgent] = CrawlerAgent,
        relevance_agent: RelevanceAgent | None = None,
        credibility_agent: CredibilityAgent | None = None,
        dedup_agent: DeduplicationAgent | None = None,
        classification_agent: ClassificationAgent | None = None,
        issue_card_agent: IssueCardAgent | None = None,
        implication_agent: ImplicationAgent | None = None,
        validation_agent: ValidationAgent | None = None,
    ):
        self.recent_days = recent_days
        self.crawler_cls = crawler_cls
        self.relevance_agent = relevance_agent or RelevanceAgent(recent_days=recent_days)
        self.credibility_agent = credibility_agent or CredibilityAgent()
        self.dedup_agent = dedup_agent or DeduplicationAgent()
        self.classification_agent = classification_agent or ClassificationAgent()
        self.issue_card_agent = issue_card_agent or IssueCardAgent()
        self.implication_agent = implication_agent or ImplicationAgent()
        self.validation_agent = validation_agent or ValidationAgent()

    async def run(self, peer_ids: list[str]) -> SupervisorResult:
        log.info("Supervisor 수집 시작 | peer_ids=%s", peer_ids)

        raw_articles = await self._crawl(peer_ids)
        relevant_articles = self._filter_relevant(raw_articles)
        credible_articles = self._filter_credible(relevant_articles)
        cluster_map = self._deduplicate(credible_articles)
        classified_clusters = self._classify(cluster_map, credible_articles)
        issue_cards = self._make_issue_cards(classified_clusters)
        implications = self._make_implications(issue_cards)
        validation_results = self._validate(issue_cards, implications)

        return {
            "raw_articles": raw_articles,
            "relevant_articles": relevant_articles,
            "credible_articles": credible_articles,
            "cluster_map": cluster_map,
            "classified_clusters": classified_clusters,
            "issue_cards": issue_cards,
            "implications": implications,
            "validation_results": validation_results,
        }

    async def _crawl(self, peer_ids: list[str]) -> list[RawArticle]:
        articles: list[RawArticle] = []

        for peer_id in peer_ids:
            crawler = self.crawler_cls(peer_id=peer_id, recent_days=self.recent_days)
            articles.extend(await crawler.collect())

        return articles

    def _filter_relevant(self, articles: list[RawArticle]) -> list[RawArticle]:
        result = self.relevance_agent.run(articles)

        if isinstance(result, tuple):
            relevant_articles, _rejected_articles = result
            return relevant_articles

        return result

    def _filter_credible(self, articles: list[RawArticle]) -> list[RawArticle]:
        return [
            article
            for article in articles
            if self.credibility_agent.classify(article.url) in ("High", "Medium")
        ]

    def _deduplicate(self, articles: list[RawArticle]) -> dict[int, list[int]]:
        article_ids = list(range(len(articles)))
        return self.dedup_agent.deduplicate(article_ids)

    def _classify(
        self,
        cluster_map: dict[int, list[int]],
        articles: list[RawArticle],
    ) -> list[dict]:
        classified_clusters = []

        for cluster_id, article_ids in cluster_map.items():
            cluster_articles = [articles[article_id].__dict__ for article_id in article_ids]
            classification = self.classification_agent.classify(cluster_id, cluster_articles)

            classified_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "article_ids": article_ids,
                    "classification": classification,
                    "articles": cluster_articles,
                }
            )

        return classified_clusters

    def _make_issue_cards(self, classified_clusters: list[dict]) -> list[dict]:
        return [self.issue_card_agent.generate(cluster) for cluster in classified_clusters]

    def _make_implications(self, issue_cards: list[dict]) -> list[dict]:
        return [self.implication_agent.generate(card) for card in issue_cards]

    def _validate(self, issue_cards: list[dict], implications: list[dict]) -> list[dict]:
        return [
            self.validation_agent.validate(issue_card, implication)
            for issue_card, implication in zip(issue_cards, implications)
        ]