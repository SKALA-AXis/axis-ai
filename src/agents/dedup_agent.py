"""Gate 3: BGE-M3 임베딩 기반 유사 기사 클러스터링 에이전트.

RelevanceAgent를 통과한 기사들을 대상으로 유사 기사 클러스터를 만든다.
title/content 임베딩 유사도로 같은 이슈를 묶는다.
원본 기사는 삭제하지 않고 raw_articles에 cluster_id와 is_representative만 저장한다.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.config.companies import COMPANY_ALIASES
from src.db.article_store import get_articles_by_ids, update_cluster

log = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.83
EMBED_BATCH_SIZE = 32

_CANONICAL_ISSUE_TERMS = {
    "physicalworks": (
        "physicalworks",
        "피지컬웍스",
        "physical works",
        "피지컬 웍스",
    ),
    "national_ai_computing_center": (
        "국가ai컴퓨팅센터",
        "국가 ai 컴퓨팅 센터",
        "국가인공지능컴퓨팅센터",
        "ai컴퓨팅센터",
    ),
}


class DeduplicationAgent:
    """article_ids → BGE-M3 임베딩 → 코사인 유사도 ≥ 0.83 클러스터링 → 대표 기사 선정."""

    def deduplicate(self, article_ids: list[int]) -> tuple[dict[int, list[int]], list[int]]:
        """Gate 3 유사 기사 클러스터링.

        Args:
            article_ids: RelevanceAgent를 통과한 raw_articles ID 목록.

        Returns:
            (cluster_map, representative_ids)
            cluster_map: {cluster_id: [article_ids]}
        """
        if not article_ids:
            return {}, []

        articles = get_articles_by_ids(article_ids)
        if not articles:
            return {}, []

        embeddings = _embed(articles)

        cluster_map = _cluster(
            articles=articles,
            embeddings=embeddings,
            threshold=DEDUP_THRESHOLD,
        )

        representative_ids = _select_representatives(
            cluster_map=cluster_map,
            articles=articles,
            embeddings=embeddings,
        )

        _persist(cluster_map, representative_ids)

        duplicate_count = len(article_ids) - len(representative_ids)
        log.info(
            "Gate 3 클러스터링 완료 | total=%d clusters=%d reps=%d dupes=%d",
            len(article_ids),
            len(cluster_map),
            len(representative_ids),
            duplicate_count,
        )

        return cluster_map, representative_ids


def deduplicate_articles(
    articles: list[dict[str, Any]],
    id_key: str = "preprocess_id",
) -> tuple[dict[int, list[int]], list[int]]:
    """JSON article 목록을 Gate 3 클러스터링 규칙으로 묶는다.

    DB 저장 없이 `DeduplicationAgent`의 embedding/대표 선정 로직을 재사용한다.
    임베딩 모델을 사용할 수 없는 로컬 환경에서는 제목 기반 클러스터링으로
    graceful fallback 한다.
    """

    if not articles:
        return {}, []

    normalized = [_normalize_local_article(article, id_key) for article in articles]

    try:
        embeddings = _embed(normalized, allow_openai_fallback=False)
        cluster_map = _cluster(
            articles=normalized,
            embeddings=embeddings,
            threshold=DEDUP_THRESHOLD,
        )
        representative_ids = _select_representatives(
            cluster_map=cluster_map,
            articles=normalized,
            embeddings=embeddings,
        )
        return cluster_map, representative_ids
    except Exception as e:
        log.warning("로컬 전처리 dedup 임베딩 실패, 제목 기반 fallback | error=%s", e)
        return _deduplicate_by_title(normalized)


def _normalize_local_article(article: dict[str, Any], id_key: str) -> dict[str, Any]:
    item = dict(article)
    item["id"] = int(item.get(id_key) or item.get("id") or 0)
    return item


def _deduplicate_by_title(articles: list[dict[str, Any]]) -> tuple[dict[int, list[int]], list[int]]:
    groups: dict[str, list[int]] = {}
    by_id = {int(article["id"]): article for article in articles}

    for article in articles:
        key = _fallback_dedup_key(article)
        groups.setdefault(key, []).append(int(article["id"]))

    cluster_map = {cluster_id: ids for cluster_id, ids in enumerate(groups.values())}
    representative_ids = [
        max(ids, key=lambda article_id: _representative_score_fallback(by_id[article_id]))
        for ids in cluster_map.values()
    ]
    return cluster_map, representative_ids


def _fallback_dedup_key(article: dict[str, Any]) -> str:
    issue_key = _issue_dedup_key(article)
    if issue_key:
        return issue_key

    title = " ".join(str(article.get("title") or "").lower().split())
    if title:
        return title
    return str(article.get("url_hash") or article.get("url") or article.get("id"))


def _representative_score_fallback(article: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(article.get("relevance_score") or 0.0),
        float(article.get("credibility_score") or 0.0),
        len(article.get("content") or ""),
    )


def _embed(
    articles: list[dict[str, Any]],
    allow_openai_fallback: bool = True,
) -> np.ndarray:
    """BGE-M3 dense 벡터 배치 임베딩.

    DB 파이프라인에서는 OpenAI fallback을 허용한다. 로컬 JSON 전처리에서는
    비용/네트워크 호출을 피하기 위해 fallback을 끄고 제목 기반 dedup으로 넘어간다.
    """
    texts = [_build_embedding_text(article) for article in articles]

    try:
        return _embed_bge(texts)
    except Exception as e:
        if not allow_openai_fallback:
            raise
        log.warning("BGE-M3 임베딩 실패, OpenAI fallback | error=%s", e)
        return _embed_openai(texts)


def _build_embedding_text(article: dict[str, Any]) -> str:
    """유사 기사 판단용 임베딩 입력 텍스트를 만든다."""
    title = article.get("title") or ""
    content = article.get("content") or ""

    return f"{title}. {title}. {title}. {content[:256]}"


def _embed_bge(texts: list[str]) -> np.ndarray:
    from src.rag.embedder import get_embedder

    model = get_embedder()
    all_vecs = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        result = model.encode(batch, return_dense=True, return_sparse=False)
        all_vecs.append(result["dense_vecs"])

    vecs = np.vstack(all_vecs)
    return _normalize_vectors(vecs)


def _embed_openai(texts: list[str]) -> np.ndarray:
    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    all_vecs = []

    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_vecs.extend([d.embedding for d in resp.data])

    vecs = np.array(all_vecs, dtype=np.float32)
    return _normalize_vectors(vecs)


def _normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


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

    sim_matrix = embeddings @ embeddings.T

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold or _same_issue(articles[i], articles[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}

    for idx, article in enumerate(articles):
        root = find(idx)
        groups.setdefault(root, []).append(article["id"])

    # TODO: 운영 환경에서는 batch마다 0부터 시작하는 local cluster_id 대신
    # article_clusters 테이블 또는 batch_id 기반 cluster_key를 사용하는 방식 검토.
    return {cluster_id: ids for cluster_id, ids in enumerate(groups.values())}


def _same_issue(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_key = _issue_dedup_key(left)
    if left_key and left_key == _issue_dedup_key(right):
        return True

    return _same_company_customer_business_issue(left, right)


def _issue_dedup_key(article: dict[str, Any]) -> str | None:
    companies = _company_key(article)
    if not companies:
        return None

    text = _compact_text(
        " ".join(
            [
                str(article.get("title") or ""),
                str(article.get("content") or "")[:1200],
            ]
        )
    )

    for issue, aliases in _CANONICAL_ISSUE_TERMS.items():
        if any(_compact_text(alias) in text for alias in aliases):
            return f"{','.join(companies)}::{issue}"

    quoted = _quoted_product_key(article)
    if quoted:
        return f"{','.join(companies)}::quoted::{quoted}"

    return None


def _company_key(article: dict[str, Any]) -> list[str]:
    value = article.get("matched_companies") or article.get("company") or []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return sorted(str(item) for item in parsed if item)
        except json.JSONDecodeError:
            pass
        return [stripped]
    if isinstance(value, (list, tuple)):
        return sorted(str(item) for item in value if item)
    return []


def _quoted_product_key(article: dict[str, Any]) -> str | None:
    title = str(article.get("title") or "")
    quoted_terms = re.findall(r"['‘’\"“”「」](.{2,40}?)['‘’\"“”「」]", title)
    for term in quoted_terms:
        compact = _compact_text(term)
        if len(compact) >= 4 and not compact.isdigit():
            return compact
    return None


_CUSTOMER_ALIASES = {
    "한국전력": ("한국전력", "한전", "kepco"),
    "현대차그룹": ("현대차그룹", "현대자동차그룹", "현대차"),
    "삼성전자": ("삼성전자",),
    "LG전자": ("LG전자",),
    "SK그룹": ("SK그룹",),
}
_BUSINESS_ISSUE_TERMS = (
    "차세대",
    "영업배전",
    "전력관리",
    "시스템",
    "isp",
    "컨설팅",
    "구축",
    "수주",
    "재설계",
    "전환",
    "ai",
    "ax",
    "클라우드",
    "플랫폼",
    "로봇",
    "휴머노이드",
    "보안",
)
_MIN_SHARED_BUSINESS_TERMS = 2
_MIN_BUSINESS_TERMS_PER_ARTICLE = 2


def _same_company_customer_business_issue(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _company_key(left) != _company_key(right):
        return False

    left_text = _issue_text(left)
    right_text = _issue_text(right)

    if not _shared_customer(left_text, right_text):
        return False

    left_terms = _business_terms(left_text)
    right_terms = _business_terms(right_text)
    if len(left_terms & right_terms) >= _MIN_SHARED_BUSINESS_TERMS:
        return True

    return (
        bool(left_terms & right_terms)
        and len(left_terms) >= _MIN_BUSINESS_TERMS_PER_ARTICLE
        and len(right_terms) >= _MIN_BUSINESS_TERMS_PER_ARTICLE
    )


def _issue_text(article: dict[str, Any]) -> str:
    return _compact_text(
        " ".join(
            [
                str(article.get("title") or ""),
                str(article.get("content") or "")[:1200],
            ]
        )
    )


def _shared_customer(left_text: str, right_text: str) -> bool:
    for aliases in _CUSTOMER_ALIASES.values():
        compact_aliases = [_compact_text(alias) for alias in aliases]
        if any(alias in left_text for alias in compact_aliases) and any(
            alias in right_text for alias in compact_aliases
        ):
            return True
    return False


def _business_terms(text: str) -> set[str]:
    return {
        term for term in (_compact_text(term) for term in _BUSINESS_ISSUE_TERMS) if term in text
    }


def _compact_text(value: str) -> str:
    return re.sub(r"[\s·'‘’\"“”「」()\[\]{}:：,._\-…]+", "", value.lower())


def _select_representatives(
    cluster_map: dict[int, list[int]],
    articles: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> list[int]:
    """각 클러스터에서 대표 기사를 선정한다.

    대표 기사 기준:
    1. RelevanceAgent가 계산한 relevance_score
    2. 클러스터 중심성
    3. 본문 품질
    4. 최신성
    """
    id_to_article = {article["id"]: article for article in articles}
    id_to_index = {article["id"]: idx for idx, article in enumerate(articles)}
    sim_matrix = embeddings @ embeddings.T

    representative_ids: list[int] = []

    for article_ids in cluster_map.values():
        best = max(
            article_ids,
            key=lambda article_id: _representative_score(
                article=id_to_article[article_id],
                cluster_article_ids=article_ids,
                id_to_index=id_to_index,
                sim_matrix=sim_matrix,
            ),
        )
        representative_ids.append(best)

    return representative_ids


def _representative_score(
    article: dict[str, Any],
    cluster_article_ids: list[int],
    id_to_index: dict[int, int],
    sim_matrix: np.ndarray,
) -> float:
    relevance_score = float(article.get("relevance_score") or 0.0)

    centrality = _cluster_centrality(
        article_id=article["id"],
        cluster_article_ids=cluster_article_ids,
        id_to_index=id_to_index,
        sim_matrix=sim_matrix,
    )

    content_quality = _content_quality_score(article)
    recency = _recency_score(article.get("published_at"))

    company_presence = _company_presence_score(article)

    return (
        0.35 * relevance_score
        + 0.30 * centrality
        + 0.20 * company_presence
        + 0.10 * content_quality
        + 0.05 * recency
    )


def _company_presence_score(article: dict[str, Any]) -> float:
    companies = _company_key(article)
    if not companies:
        return 0.0

    title = _compact_text(str(article.get("title") or ""))
    lead = _compact_text(str(article.get("content") or "")[:800])

    aliases = [
        _compact_text(alias)
        for company_id in companies
        for alias in COMPANY_ALIASES.get(company_id, [company_id])
    ]

    if any(alias and alias in title for alias in aliases):
        return 1.0

    if any(alias and alias in lead for alias in aliases):
        return 0.7

    return 0.0


def _cluster_centrality(
    article_id: int,
    cluster_article_ids: list[int],
    id_to_index: dict[int, int],
    sim_matrix: np.ndarray,
) -> float:
    if len(cluster_article_ids) <= 1:
        return 1.0

    idx = id_to_index[article_id]
    other_indices = [
        id_to_index[other_id] for other_id in cluster_article_ids if other_id != article_id
    ]

    if not other_indices:
        return 1.0

    return float(np.mean(sim_matrix[idx, other_indices]))


def _content_quality_score(article: dict[str, Any]) -> float:
    content = (article.get("content") or "").strip()
    length = len(content)

    if length <= 0:
        return 0.0

    length_score = min(length / 1500, 1.0)

    if length < 200:
        length_score *= 0.5

    return float(length_score)


def _recency_score(value: Any) -> float:
    published_at = _parse_datetime(value)

    if published_at is None:
        return 0.5

    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    now = datetime.now(published_at.tzinfo)
    age_days = max((now - published_at).days, 0)

    return max(0.0, 1.0 - age_days / 30)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    return None


def _persist(cluster_map: dict[int, list[int]], representative_ids: list[int]) -> None:
    """클러스터 정보를 DB에 저장한다."""
    representative_set = set(representative_ids)

    for cluster_id, ids in cluster_map.items():
        for article_id in ids:
            update_cluster(
                article_id,
                cluster_id,
                is_representative=article_id in representative_set,
            )
