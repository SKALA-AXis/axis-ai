"""중복 제거 및 이슈 클러스터링 에이전트"""

from __future__ import annotations

import logging
from datetime import datetime
from difflib import SequenceMatcher
from math import sqrt
from typing import Protocol

from src.crawler.article_filter import issue_signature, similarity_key
from src.crawler.base_crawler import RawArticle

log = logging.getLogger(__name__)

TITLE_DUPLICATE_THRESHOLD = 0.95
RULE_TEXT_SIMILARITY_THRESHOLD = 0.90
EMBEDDING_SIMILARITY_THRESHOLD = 0.88


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class DeduplicationAgent:
    def __init__(
        self,
        embedder: Embedder | None = None,
        use_embedding: bool = True,
        embedding_threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
    ):
        self.embedder = embedder
        self.use_embedding = use_embedding
        self.embedding_threshold = embedding_threshold

    def run(self, articles: list[RawArticle]) -> list[dict]:
        log.info("중복 제거 시작 | articles=%d", len(articles))

        unique_articles = self._remove_exact_url_duplicates(articles)

        if self.use_embedding and self.embedder is not None:
            clusters = self._cluster_by_embedding(unique_articles)
            cluster_method = "embedding"
        else:
            clusters = self._cluster_by_rule(unique_articles)
            cluster_method = "rule"

        result = self._build_cluster_output(
            clusters=clusters,
            cluster_method=cluster_method,
        )

        log.info(
            "중복 제거 완료 | input=%d unique_url=%d clusters=%d method=%s",
            len(articles),
            len(unique_articles),
            len(result),
            cluster_method,
        )

        return result

    def deduplicate(self, articles: list[RawArticle]) -> list[dict]:
        return self.run(articles)

    def _remove_exact_url_duplicates(self, articles: list[RawArticle]) -> list[RawArticle]:
        seen_urls: set[str] = set()
        unique_articles: list[RawArticle] = []

        for article in articles:
            url = getattr(article, "url", "")

            if not url:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(url)
            unique_articles.append(article)

        return unique_articles

    def _cluster_by_embedding(self, articles: list[RawArticle]) -> list[list[RawArticle]]:
        if not articles:
            return []

        texts = [self._embedding_text(article) for article in articles]
        embeddings = self.embedder.encode(texts)

        clusters: list[list[RawArticle]] = []
        cluster_embeddings: list[list[float]] = []

        for article, embedding in zip(articles, embeddings):
            matched_index = self._find_embedding_cluster(
                embedding=embedding,
                cluster_embeddings=cluster_embeddings,
            )

            if matched_index is None:
                clusters.append([article])
                cluster_embeddings.append(embedding)
            else:
                clusters[matched_index].append(article)
                cluster_embeddings[matched_index] = self._average_embedding(
                    cluster_embeddings[matched_index],
                    embedding,
                )

        return clusters

    def _find_embedding_cluster(
        self,
        embedding: list[float],
        cluster_embeddings: list[list[float]],
    ) -> int | None:
        best_index = None
        best_score = 0.0

        for index, cluster_embedding in enumerate(cluster_embeddings):
            score = self._cosine_similarity(embedding, cluster_embedding)

            if score > best_score:
                best_score = score
                best_index = index

        if best_score >= self.embedding_threshold:
            return best_index

        return None

    def _cluster_by_rule(self, articles: list[RawArticle]) -> list[list[RawArticle]]:
        clusters: list[list[RawArticle]] = []
        cluster_keys: list[str] = []
        cluster_title_keys: list[str] = []
        cluster_issue_signatures: list[set[str]] = []

        for article in articles:
            article_key = similarity_key(article.title, article.content)
            title_key = similarity_key(article.title, "")
            article_signature = issue_signature(article_key)

            matched_index = self._find_rule_cluster(
                article_key=article_key,
                title_key=title_key,
                article_signature=article_signature,
                cluster_keys=cluster_keys,
                cluster_title_keys=cluster_title_keys,
                cluster_issue_signatures=cluster_issue_signatures,
            )

            if matched_index is None:
                clusters.append([article])
                cluster_keys.append(article_key)
                cluster_title_keys.append(title_key)
                cluster_issue_signatures.append(article_signature)
            else:
                clusters[matched_index].append(article)

        return clusters

    def _find_rule_cluster(
        self,
        article_key: str,
        title_key: str,
        article_signature: set[str],
        cluster_keys: list[str],
        cluster_title_keys: list[str],
        cluster_issue_signatures: list[set[str]],
    ) -> int | None:
        for index, cluster_key in enumerate(cluster_keys):
            title_similarity = SequenceMatcher(
                None,
                title_key,
                cluster_title_keys[index],
            ).ratio()

            if title_similarity >= TITLE_DUPLICATE_THRESHOLD:
                return index

            text_similarity = SequenceMatcher(
                None,
                article_key,
                cluster_key,
            ).ratio()

            if text_similarity >= RULE_TEXT_SIMILARITY_THRESHOLD:
                return index

            shared_issues = article_signature & cluster_issue_signatures[index]
            if len(shared_issues) >= 2:
                return index

        return None

    def _build_cluster_output(
        self,
        clusters: list[list[RawArticle]],
        cluster_method: str,
    ) -> list[dict]:
        result = []

        for cluster_id, cluster in enumerate(clusters):
            representative_article = self._select_representative(cluster)
            related_articles = [
                article
                for article in cluster
                if article is not representative_article
            ]

            issue_keywords = sorted(
                issue_signature(
                    similarity_key(
                        representative_article.title,
                        representative_article.content,
                    )
                )
            )

            result.append(
                {
                    "cluster_id": cluster_id,
                    "representative_article": representative_article,
                    "related_articles": related_articles,
                    "articles": cluster,
                    "cluster_size": len(cluster),
                    "issue_keywords": issue_keywords,
                    "representative_score": self._representative_score(representative_article),
                    "cluster_method": cluster_method,
                }
            )

        return result

    def _select_representative(self, cluster: list[RawArticle]) -> RawArticle:
        return max(cluster, key=self._representative_score)

    def _representative_score(self, article: RawArticle) -> float:
        return (
            self._source_score(article) * 3.0
            + self._content_score(article) * 2.0
            + self._recency_score(article)
            + self._title_quality_score(article)
        )

    def _source_score(self, article: RawArticle) -> float:
        url = (getattr(article, "url", "") or "").lower()
        source_name = (getattr(article, "source_name", "") or "").lower()

        if "dart" in source_name or "dart.fss.or.kr" in url:
            return 3.0

        if any(
            domain in url
            for domain in [
                "samsungsds.com",
                "lgcns.com",
                "skax",
                "skcc",
                "poscodx.com",
                "hyundai-autoever.com",
            ]
        ):
            return 3.0

        if any(
            domain in url
            for domain in [
                "yna.co.kr",
                "yonhap",
                "hankyung.com",
                "mk.co.kr",
                "sedaily.com",
                "fnnews.com",
                "newsis.com",
                "newspim.com",
            ]
        ):
            return 2.5

        if any(
            domain in url
            for domain in [
                "zdnet.co.kr",
                "etnews.com",
                "ddaily.co.kr",
                "digitaltoday.co.kr",
                "it.chosun.com",
                "bloter.net",
                "inews24.com",
                "ciokorea.com",
            ]
        ):
            return 2.0

        if "news.naver.com" in url or "n.news.naver.com" in url:
            return 1.8

        if "job" in source_name or "career" in source_name or "recruit" in source_name:
            return 1.5

        return 1.0

    def _content_score(self, article: RawArticle) -> float:
        content = getattr(article, "content", "") or ""
        return min(len(content), 1000) / 1000

    def _recency_score(self, article: RawArticle) -> float:
        published_at = getattr(article, "published_at", None)

        if published_at is None:
            return 0.0

        now = datetime.now(published_at.tzinfo)
        elapsed_seconds = max((now - published_at).total_seconds(), 0)

        return max(0.0, 1.0 - elapsed_seconds / 86400)

    def _title_quality_score(self, article: RawArticle) -> float:
        title = (getattr(article, "title", "") or "").lower()

        penalty_words = [
            "광고",
            "홍보",
            "이벤트",
            "혜택",
            "특가",
        ]

        bonus_words = [
            "단독",
            "속보",
            "공식",
            "발표",
            "수주",
            "협력",
            "투자",
            "출시",
            "계약",
            "인수",
            "파트너십",
        ]

        score = 1.0

        if any(word in title for word in penalty_words):
            score -= 0.4

        if any(word in title for word in bonus_words):
            score += 0.2

        return max(score, 0.0)

    def _embedding_text(self, article: RawArticle) -> str:
        title = getattr(article, "title", "") or ""
        content = getattr(article, "content", "") or ""
        source_name = getattr(article, "source_name", "") or ""

        return f"제목: {title}\n요약: {content[:1000]}\n출처: {source_name}"

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot / (left_norm * right_norm)

    def _average_embedding(self, left: list[float], right: list[float]) -> list[float]:
        return [(a + b) / 2 for a, b in zip(left, right)]