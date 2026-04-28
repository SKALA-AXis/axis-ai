"""Gate 3: BGE-M3 임베딩 기반 코사인 유사도 클러스터링 에이전트."""

import logging
from typing import Any

import numpy as np

from src.db.article_store import get_articles_by_ids, update_cluster

log = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.83  # v3 §2.5 ②: 0.85 → 0.83 (재보도 0.80~0.84 구간 흡수)
EMBED_BATCH_SIZE = 32


class DeduplicationAgent:
    """credible_ids → BGE-M3 임베딩 → 코사인 유사도 ≥ 0.85 클러스터링 → 대표 기사 선정."""

    def deduplicate(self, credible_ids: list[int]) -> tuple[dict[int, list[int]], list[int]]:
        """Gate 3 중복 제거 및 클러스터링.

        Args:
            credible_ids: Gate 2 통과한 raw_articles ID 목록.

        Returns:
            (cluster_map, representative_ids)
            cluster_map: {cluster_id: [article_ids]}
        """
        if not credible_ids:
            return {}, []

        articles = get_articles_by_ids(credible_ids)
        if not articles:
            return {}, []

        embeddings = _embed(articles)
        cluster_map = _cluster(articles, embeddings, DEDUP_THRESHOLD)
        representative_ids = _select_representatives(cluster_map, articles)

        _persist(cluster_map, representative_ids)

        skipped = len(credible_ids) - len(representative_ids)
        log.info(
            "Gate 3 클러스터링 완료 | total=%d clusters=%d reps=%d dupes=%d",
            len(credible_ids),
            len(cluster_map),
            len(representative_ids),
            skipped,
        )
        return cluster_map, representative_ids


# ── 임베딩 ────────────────────────────────────────────────────


def _embed(articles: list[dict[str, Any]]) -> np.ndarray:
    """BGE-M3 dense 벡터 배치 임베딩. 실패 시 OpenAI fallback.

    같은 사건을 다른 언론사가 다르게 서술하는 재보도를 잡기 위해
    제목 가중치를 높인다 (제목 3회 반복 + 본문 256자) — v3 §2.5 ②.
    """
    texts = [
        f"{a['title']}. {a['title']}. {a['title']}. {(a['content'] or '')[:256]}" for a in articles
    ]
    try:
        return _embed_bge(texts)
    except Exception as e:
        log.warning("BGE-M3 임베딩 실패, OpenAI fallback | error=%s", e)
        return _embed_openai(texts)


def _embed_bge(texts: list[str]) -> np.ndarray:
    from src.rag.embedder import get_embedder

    model = get_embedder()
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        result = model.encode(batch, return_dense=True, return_sparse=False)
        all_vecs.append(result["dense_vecs"])
    vecs = np.vstack(all_vecs)
    # L2 정규화 (cosine similarity = dot product after normalization)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def _embed_openai(texts: list[str]) -> np.ndarray:
    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    all_vecs = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        all_vecs.extend([d.embedding for d in resp.data])
    vecs = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


# ── 클러스터링 ─────────────────────────────────────────────────


def _cluster(
    articles: list[dict[str, Any]],
    embeddings: np.ndarray,
    threshold: float,
) -> dict[int, list[int]]:
    """Union-Find 기반 그리디 클러스터링."""
    n = len(articles)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    # 유사도 행렬 계산 (정규화된 벡터이므로 dot product = cosine)
    sim_matrix = embeddings @ embeddings.T
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                union(i, j)

    # 클러스터 맵 구성 (article index → article ID)
    groups: dict[int, list[int]] = {}
    for idx, art in enumerate(articles):
        root = find(idx)
        groups.setdefault(root, []).append(art["id"])

    return {cluster_id: ids for cluster_id, ids in enumerate(groups.values())}


def _select_representatives(
    cluster_map: dict[int, list[int]],
    articles: list[dict[str, Any]],
) -> list[int]:
    """각 클러스터에서 credibility_score 최고 기사를 대표로 선정한다."""
    id_to_article = {a["id"]: a for a in articles}
    rep_ids = []
    for ids in cluster_map.values():
        best = max(ids, key=lambda aid: id_to_article[aid].get("credibility_score") or 0.0)
        rep_ids.append(best)
    return rep_ids


def _persist(cluster_map: dict[int, list[int]], representative_ids: list[int]) -> None:
    """클러스터 정보를 DB에 저장한다."""
    rep_set = set(representative_ids)
    for cluster_id, ids in cluster_map.items():
        for article_id in ids:
            update_cluster(article_id, cluster_id, is_representative=(article_id in rep_set))
