"""Gate 3: BGE-M3 임베딩 기반 유사 기사 클러스터링 전처리.

RelevanceEvaluator를 통과한 기사들을 대상으로 유사 기사 클러스터를 만든다.
title/content 임베딩 유사도로 같은 이슈를 묶는다.
원본 기사는 삭제하지 않고 raw_articles에 cluster_id와 is_representative만 저장한다.
"""

import json
import logging
import os
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from src.config.companies import COMPANY_ALIASES
from src.config.global_companies import GLOBAL_COMPANY_ALIASES
from src.config.openai_policy import openai_calls_enabled
from src.db.article_store import (
    get_articles_by_ids,
    list_existing_news_cluster_candidates,
    merge_card_news_sources_for_cluster,
    update_cluster,
)

log = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.80
# BGE-M3 fp16 + attention O(len²) → 큰 batch 가 OOM 의 주범.
# 32 → 8 로 낮춰 spike 메모리 4x↓. encode loop 횟수만 늘어남 (CPU 직렬이라 latency
# 영향 미미). 환경변수 DEDUP_EMBED_BATCH_SIZE 로 override 가능.
EMBED_BATCH_SIZE = int(os.getenv("DEDUP_EMBED_BATCH_SIZE", "8"))
_HIGH_CONFIDENCE_SIMILARITY = 0.88
_EXISTING_CLUSTER_THRESHOLD = float(os.getenv("DEDUP_EXISTING_CLUSTER_THRESHOLD", "0.84"))
_EXISTING_CLUSTER_MIN_SIMILARITY = float(os.getenv("DEDUP_EXISTING_CLUSTER_MIN_SIMILARITY", "0.70"))
_EXISTING_CLUSTER_LOOKBACK_HOURS = int(
    os.getenv("DEDUP_EXISTING_CLUSTER_LOOKBACK_HOURS", str(24 * 7))
)
_EXISTING_CLUSTER_CANDIDATE_LIMIT = int(os.getenv("DEDUP_EXISTING_CLUSTER_CANDIDATE_LIMIT", "200"))
_MAX_CLUSTER_PUBLISHED_GAP_DAYS = int(os.getenv("DEDUP_MAX_CLUSTER_PUBLISHED_GAP_DAYS", "5"))
_MAX_BRIDGE_TOPIC_TERMS = 1
_MIN_RELATED_TERM_LENGTH = 6
_TERM_NGRAM_SIMILARITY = 0.40
_CLUSTER_LLM_JUDGE_ENABLED = os.getenv("DEDUP_CLUSTER_LLM_JUDGE_ENABLED", "true").lower() == "true"
_CLUSTER_LLM_MAX_CALLS = int(os.getenv("DEDUP_CLUSTER_LLM_MAX_CALLS", "30"))
_CLUSTER_LLM_MODEL = os.getenv("DEDUP_CLUSTER_LLM_MODEL", "gpt-4o-mini")
_CLUSTER_LLM_CONTENT_CHARS = int(os.getenv("DEDUP_CLUSTER_LLM_CONTENT_CHARS", "280"))
_ALL_COMPANY_ALIASES = {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}
_EVENT_BUCKET_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("investment_deal", ("투자", "지분", "인수", "두나무", "m&a", "ma")),
    (
        "market_reaction",
        (
            "주가",
            "상한가",
            "급등",
            "특징주",
            "거래량",
            "거래대금",
            "매수세",
            "증시키워드",
            "관심종목",
            "목표주가",
            "폭등",
            "하락",
            "랠리",
        ),
    ),
    (
        "ax_strategy",
        (
            "ax",
            "rx",
            "제조ax",
            "제조 ax",
            "제조rx",
            "제조 rx",
            "제조특화",
            "제조 특화",
            "제조플랫폼",
            "제조 플랫폼",
            "ai자율공장",
            "자율공장",
            "스마트팩토리",
            "생성형ai",
            "ai전환",
            "피지컬ai",
            "피지컬 ai",
            "자율용접",
            "자율 용접",
            "용접로봇",
            "용접 로봇",
            "로봇파운데이션",
            "로봇 파운데이션",
            "로봇뇌",
            "로봇 뇌",
            "로봇의두뇌",
            "로봇의 두뇌",
            "로봇두뇌",
            "로봇 두뇌",
            "범용로봇두뇌",
            "범용 로봇 두뇌",
            "로봇ai",
            "로봇 ai",
            "ai로봇",
            "ai 로봇",
            "로봇ai브레인",
            "로봇 ai브레인",
            "로봇 ai 브레인",
            "산업용로봇제어모델",
            "산업용 로봇 제어 모델",
            "피지컬ai맞손",
            "피지컬 ai 맞손",
            "피지컬ai지능화",
            "피지컬 ai 지능화",
            "범용로봇ai",
            "범용 로봇 ai",
            "로봇지능화",
            "로봇 지능화",
            "산업용로봇제어모델",
            "산업용 로봇 제어 모델",
            "ai두뇌",
            "ai 두뇌",
            "ai모델",
            "ai 모델",
        ),
    ),
    (
        "contract_deal",
        (
            "수주",
            "계약",
            "공급계약",
            "사업수주",
            "사업자선정",
            "우선협상",
            "업무협약",
            "구축한다",
            "협력",
            "협업",
            "맞손",
            "실증",
            "검증",
            "도입",
            "poc",
            "mou",
        ),
    ),
    ("cloud_infra", ("클라우드", "데이터센터", "gpu", "gpuass", "gpu서비스", "인프라")),
    ("security", ("보안", "침해", "해킹", "취약점")),
    ("industry_theme", ("si주", "si株", "it서비스업종", "테마", "업종전반", "관련업종")),
)

# 단독(singleton) 클러스터 처리 정책.
#   - bridge risk(제목에 주제어 없음 + 본문 주제어 다수) 가 의심되어도 cluster 를 유지한다.
#   - 이전: ambiguous singleton 은 drop → cycle 마다 clusters=0 으로 떨어져
#     classify 이후 (card_news / evidence / vector_index) 가 entirely skip 되는 문제.
#   - 이 토글은 production 의 cycle 단위 cluster 형성률을 회복하기 위한 것.
#   - default true (singleton 유지). 과거 동작 복원이 필요하면 env 로 false.
_KEEP_AMBIGUOUS_SINGLETONS = os.getenv("DEDUP_KEEP_AMBIGUOUS_SINGLETONS", "true").lower() == "true"

_CANONICAL_ISSUE_TERMS: Mapping[str, tuple[str, ...]] = {}
_TITLE_CONCEPT_TERMS: Mapping[str, tuple[str, ...]] = {
    "lg_cns_agentic_aind": (
        "aind",
        "에이전틱ai개발플랫폼",
        "에이전틱 ai 개발 플랫폼",
        "에이전틱개발플랫폼",
        "에이전틱 개발 플랫폼",
        "ai개발플랫폼",
        "ai 개발 플랫폼",
        "기업용개발플랫폼",
        "기업용 개발 플랫폼",
        "기업시스템개발",
        "기업 시스템 개발",
        "기업시스템구축",
        "기업 시스템 구축",
        "대규모it시스템",
        "대규모 it 시스템",
        "대규모시스템구축",
        "대규모 시스템 구축",
        "데브온",
        "바이브코딩",
        "바이브 코딩",
        "코볼",
        "계좌이체",
    ),
    "gpu_ai_highway": (
        "ai고속도로",
        "ai 고속도로",
        "gpu9704",
        "gpu 9704",
        "9704장",
        "2조원대gpu",
        "2조원대 gpu",
        "2조800억",
        "2조 800억",
        "2조규모gpu",
        "2조 규모 gpu",
        "gpu사업자",
        "gpu 사업자",
        "엘리스그룹",
        "엘리스 그룹",
        "네이버클라우드",
        "네이버 클라우드",
    ),
    "lg_nvidia_physical_ai": (
        "젠슨황",
        "젠슨 황",
        "엔비디아",
        "nvidia",
        "lg엔비디아",
        "lg엔 비디아",
        "한국ai동맹",
        "한국 ai 동맹",
        "서울누빈젠슨황",
        "서울 누빈 젠슨 황",
        "광폭행보",
        "광폭 행보",
        "피지컬ai동맹",
        "피지컬 ai 동맹",
        "피지컬ai협력",
        "피지컬 ai 협력",
        "ai인프라까지맞손",
        "ai 인프라까지 맞손",
        "로봇동맹",
        "로봇 동맹",
        "전방위동맹",
        "전방위 동맹",
    ),
    "hyundai_autoever_robotics_challenge": (
        "현대오토에버",
        "한국과학창의재단",
        "로보틱스챌린지",
        "로보틱스 챌린지",
        "청소년로보틱스",
        "청소년 로보틱스",
    ),
    "saemaul_inspection_system": (
        "새마을금고",
        "검사종합시스템",
        "검사 종합 시스템",
        "이상징후탐지",
        "이상징후 탐지",
    ),
    "autonomous_welding_robot": (
        "자율용접",
        "자율 용접",
        "용접로봇",
        "용접 로봇",
        "ai두뇌",
        "ai 두뇌",
    ),
    "robot_foundation_model": (
        "로봇파운데이션",
        "로봇 파운데이션",
        "산업현장용",
        "산업현장로봇",
        "산업 현장 로봇",
        "로봇뇌",
        "로봇 뇌",
        "로봇두뇌",
        "로봇 두뇌",
        "로봇의두뇌",
        "로봇의 두뇌",
        "범용로봇두뇌",
        "범용 로봇 두뇌",
        "범용로봇ai",
        "범용 로봇 ai",
        "로봇ai",
        "로봇 ai",
        "ai로봇",
        "ai 로봇",
        "로봇ai브레인",
        "로봇 ai브레인",
        "로봇 ai 브레인",
        "ai로봇개발",
        "ai 로봇 개발",
        "로봇ai개발",
        "로봇 ai 개발",
        "산업용로봇제어모델",
        "산업용 로봇 제어 모델",
        "로봇지능개발",
        "로봇 지능 개발",
        "피지컬ai맞손",
        "피지컬 ai 맞손",
        "피지컬ai지능화",
        "피지컬 ai 지능화",
        "로봇지능화",
        "로봇 지능화",
    ),
    "logistics_robotics": (
        "물류센터",
        "물류 센터",
        "물류자동화",
        "물류 자동화",
        "스마트물류",
        "스마트 물류",
        "휴머노이드",
        "피지컬웍스",
        "로봇직원",
        "로봇 직원",
        "로봇피킹",
        "로봇 피킹",
        "로봇도입",
        "로봇 도입",
    ),
    "physicalworks_rx_platform": (
        "피지컬웍스",
        "피지컬 웍스",
        "rx플랫폼",
        "rx 플랫폼",
        "로봇학습",
        "로봇 학습",
        "로봇운영",
        "로봇 운영",
        "로봇플랫폼",
        "로봇 플랫폼",
        "로봇통합",
        "로봇 통합",
        "자율협업",
        "자율 협업",
        "이기종협업",
        "이기종 협업",
    ),
    "manufacturing_ax_market": (
        "제조ax",
        "제조 ax",
        "제조rx",
        "제조 rx",
        "제조기업",
        "제조 기업",
        "제조시장",
        "제조 시장",
        "제조업",
        "제조특화",
        "제조 특화",
        "제조플랫폼",
        "제조 플랫폼",
        "스마트팩토리",
        "스마트 팩토리",
        "공장지능화",
        "공장 지능화",
        "공장전환",
        "공장 전환",
        "ai스마트팩토리",
        "ai 스마트팩토리",
    ),
    "smart_infra_lidar": (
        "스마트인프라",
        "스마트 인프라",
        "스마트시티",
        "스마트 시티",
        "라이다",
        "lidar",
        "에스오에스랩",
        "soslab",
        "북미스마트인프라",
        "북미 스마트 인프라",
    ),
    "openai_enterprise_ai": (
        "오픈ai",
        "오픈 ai",
        "openai",
        "챗gpt",
        "chatgpt",
        "엔터프라이즈ai",
        "엔터프라이즈 ai",
        "기업용ai",
        "기업용 ai",
        "생성형ai",
        "생성형 ai",
    ),
    "national_ai_computing_center": (
        "국가ai컴퓨팅센터",
        "국가 ai 컴퓨팅센터",
        "국가ai컴퓨팅 센터",
        "ai컴퓨팅센터",
        "ai 컴퓨팅센터",
        "ai고속도로",
        "ai 고속도로",
        "gpu1.5만장",
        "gpu 1.5만장",
    ),
    "security_token_platform": (
        "토큰증권",
        "sto",
        "예탁결제원",
        "예탁원",
        "플랫폼구축",
        "플랫폼 구축",
    ),
    "jensen_huang_visit": (
        "젠슨황",
        "젠슨 황",
        "방한",
        "유퀴즈",
    ),
}
_cluster_llm_calls = 0
_cluster_llm_cache: dict[tuple[str, str, str], bool | None] = {}
_cluster_llm_approved_pairs: set[frozenset[int]] = set()


class ArticleDeduplicator:
    """article_ids → BGE-M3 임베딩 → 코사인 유사도 ≥ 0.80 클러스터링 → 대표 기사 선정."""

    def deduplicate(self, article_ids: list[int]) -> tuple[dict[int, list[int]], list[int]]:
        """Gate 3 유사 기사 클러스터링.

        Args:
            article_ids: RelevanceEvaluator를 통과한 raw_articles ID 목록.

        Returns:
            (cluster_map, representative_ids)
            cluster_map: {representative_article_id: [article_ids]}
        """
        if not article_ids:
            return {}, []

        articles = get_articles_by_ids(article_ids)
        if not articles:
            return {}, []

        cluster_map = _cluster_rule_first(
            articles=articles,
            threshold=DEDUP_THRESHOLD,
        )

        embeddings = _embed(articles)

        representative_ids = _select_representatives(
            cluster_map=cluster_map,
            articles=articles,
            embeddings=embeddings,
        )
        cluster_map, representative_ids, existing_matches = _merge_with_existing_clusters(
            cluster_map=cluster_map,
            representative_ids=representative_ids,
            articles=articles,
            embeddings=embeddings,
        )

        _persist(cluster_map, representative_ids)
        _persist_existing_card_updates(existing_matches)

        duplicate_count = len(article_ids) - len(representative_ids)
        log.info(
            "Gate 3 클러스터링 완료 | total=%d clusters=%d reps=%d dupes=%d existing_matches=%d",
            len(article_ids),
            len(cluster_map),
            len(representative_ids),
            duplicate_count,
            len(existing_matches),
        )

        return cluster_map, representative_ids


def deduplicate_articles(
    articles: list[dict[str, Any]],
    id_key: str = "preprocess_id",
) -> tuple[dict[int, list[int]], list[int]]:
    """JSON article 목록을 Gate 3 클러스터링 규칙으로 묶는다.

    DB 저장 없이 `ArticleDeduplicator`의 embedding/대표 선정 로직을 재사용한다.
    임베딩 모델을 사용할 수 없는 로컬 환경에서는 제목 기반 클러스터링으로
    graceful fallback 한다.
    """

    if not articles:
        return {}, []

    normalized = [_normalize_local_article(article, id_key) for article in articles]

    try:
        cluster_map = _cluster_rule_first(
            articles=normalized,
            threshold=DEDUP_THRESHOLD,
            allow_openai_fallback=False,
        )
        embeddings = _embed(normalized, allow_openai_fallback=False)
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


def _representative_score_fallback(article: dict[str, Any]) -> tuple[float, str, str, int]:
    return (
        float(article.get("relevance_score") or 0.0),
        str(article.get("published_at") or ""),
        str(article.get("collected_at") or ""),
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
        if not allow_openai_fallback or not openai_calls_enabled():
            raise
        log.warning("BGE-M3 임베딩 실패, OpenAI fallback | error=%s", e)
        return _embed_openai(texts)


def _build_embedding_text(article: dict[str, Any]) -> str:
    """제목 중심 임베딩 입력을 만든다.

    클러스터링의 목적은 같은 사건을 묶는 것이다. 본문 전체를 넣으면 업계 배경,
    관련 종목, 이전 사례까지 같이 임베딩되어 서로 다른 사건이 붙기 쉬워서
    제목과 짧은 lead, 제목 기반 entity만 사용한다.
    """
    title = _clean_space(str(article.get("title") or ""))
    lead = _content_text(article)[:360]
    entities = _title_issue_entities(article)

    entity_lines = [
        _entity_line("companies", entities["companies"]),
        _entity_line("sectors", entities["sectors"]),
        _entity_line("issues", entities["canonical_issues"]),
        _entity_line("products", entities["quoted_terms"]),
        _entity_line("proper_terms", entities["proper_terms"]),
        _entity_line("numbers", entities["numbers"]),
    ]

    return "\n".join(
        part
        for part in [
            f"title: {title}",
            f"event_bucket: {_event_bucket(article)}",
            f"lead: {lead}",
            *entity_lines,
        ]
        if part.strip()
    )


def _entity_line(label: str, values: list[str]) -> str:
    if not values:
        return ""
    return f"{label}: {', '.join(values)}"


def _content_text(article: dict[str, Any]) -> str:
    return _clean_space(str(article.get("content") or ""))


def _clean_space(value: str) -> str:
    return " ".join(value.split())


def _embed_bge(texts: list[str]) -> np.ndarray:
    from src.rag.embedder import EMBED_MAX_LENGTH, get_embedder

    model = get_embedder()
    all_vecs = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        # max_length 명시로 attention matrix 메모리 spike 차단 (BGE-M3 default 8192).
        # raw_articles 실측: max_length=2048 통과율 85.5%, 평균 truncate 183 tokens
        # (article 후반부 noise 위주라 dedup 시그널 영향 미미).
        result = model.encode(
            batch,
            return_dense=True,
            return_sparse=False,
            max_length=EMBED_MAX_LENGTH,
        )
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


def _cluster_rule_first(
    *,
    articles: list[dict[str, Any]],
    threshold: float,
    allow_openai_fallback: bool = True,
) -> dict[int, list[int]]:
    """Rule-first clustering.

    Articles must pass deterministic company/event-signature grouping before
    BGE similarity is allowed to merge them. This prevents one broad high-sim
    article from bridging unrelated same-company news into a mega-cluster.
    """
    final_values: list[list[int]] = []
    for group_articles in _rule_prefilter_groups(articles):
        if len(group_articles) == 1:
            final_values.append([int(group_articles[0]["id"])])
            continue
        embeddings = _embed(group_articles, allow_openai_fallback=allow_openai_fallback)
        local_map = _cluster(
            articles=group_articles,
            embeddings=embeddings,
            threshold=threshold,
        )
        final_values.extend(local_map.values())
    return {cluster_id: ids for cluster_id, ids in enumerate(final_values)}


def _rule_prefilter_groups(articles: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for article in articles:
        groups.setdefault(_rule_prefilter_key(article), []).append(article)
    return list(groups.values())


def _rule_prefilter_key(article: dict[str, Any]) -> str:
    event_key = _event_prefilter_key(article)
    if _uses_cross_day_prefilter(article):
        return event_key
    return f"{_published_day(article)}::{event_key}"


def _cluster(
    articles: list[dict[str, Any]],
    embeddings: np.ndarray,
    threshold: float,
) -> dict[int, list[int]]:
    """Union-Find 기반 그리디 클러스터링."""
    global _cluster_llm_approved_pairs
    _cluster_llm_approved_pairs = set()

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
            similarity = float(sim_matrix[i, j])
            if _should_merge_articles(articles[i], articles[j], similarity, threshold):
                union(i, j)

    groups: dict[int, list[int]] = {}

    for idx, article in enumerate(articles):
        root = find(idx)
        groups.setdefault(root, []).append(article["id"])

    id_to_article = {int(article["id"]): article for article in articles}
    cluster_values = [
        ids
        for ids in _split_cluster_values_by_event_bucket(groups.values(), id_to_article)
        if not _support_only_singleton(ids, id_to_article)
    ]

    return {cluster_id: ids for cluster_id, ids in enumerate(cluster_values)}


def _with_representative_cluster_ids(
    cluster_map: dict[int, list[int]],
    representative_ids: list[int],
) -> dict[int, list[int]]:
    """DB 파이프라인에서는 run-local index 대신 대표 기사 ID를 cluster_id로 사용한다."""
    return {
        representative_id: article_ids
        for representative_id, article_ids in zip(representative_ids, cluster_map.values())
    }


def _merge_with_existing_clusters(
    *,
    cluster_map: dict[int, list[int]],
    representative_ids: list[int],
    articles: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> tuple[dict[int, list[int]], list[int], dict[int, list[int]]]:
    """이번 실행 클러스터를 최근 기존 클러스터에 붙인다.

    반환값의 cluster_map은 최종 cluster_id 기준이다. 기존 클러스터에 매칭된 경우
    기존 cluster_id/대표기사를 유지하고, 신규 기사만 기존 카드 sources에 병합한다.
    """
    if not cluster_map:
        return {}, [], {}

    id_to_article = {int(article["id"]): article for article in articles}
    id_to_index = {int(article["id"]): idx for idx, article in enumerate(articles)}
    company_keys = sorted(
        {company for article in articles for company in _company_key(article) if company}
    )

    candidates = list_existing_news_cluster_candidates(
        company_keys=company_keys,
        exclude_article_ids=list(id_to_article),
        lookback_hours=_EXISTING_CLUSTER_LOOKBACK_HOURS,
        published_window=_existing_cluster_candidate_window(articles),
        limit=_EXISTING_CLUSTER_CANDIDATE_LIMIT,
    )
    if not candidates:
        return (
            _with_representative_cluster_ids(cluster_map, representative_ids),
            representative_ids,
            {},
        )

    candidate_embeddings = _embed(candidates)
    rep_by_local_cluster = dict(zip(cluster_map, representative_ids))
    final_cluster_map: dict[int, list[int]] = {}
    final_representatives: list[int] = []
    existing_matches: dict[int, list[int]] = {}

    for local_cluster_id, article_ids in cluster_map.items():
        rep_id = rep_by_local_cluster[local_cluster_id]
        rep_article = id_to_article[rep_id]
        rep_embedding = embeddings[id_to_index[rep_id]]
        match = _best_existing_cluster_match(
            rep_article=rep_article,
            rep_embedding=rep_embedding,
            candidates=candidates,
            candidate_embeddings=candidate_embeddings,
        )

        if match is None:
            final_cluster_map.setdefault(rep_id, []).extend(article_ids)
            final_representatives.append(rep_id)
            continue

        existing_cluster_id = int(match["cluster_id"])
        existing_rep_id = int(match["representative_id"])
        existing_article_ids = [
            int(article_id)
            for article_id in (match.get("article_ids") or [])
            if int(article_id) not in id_to_article
        ]
        merged_ids = final_cluster_map.setdefault(existing_cluster_id, [])
        for article_id in [existing_rep_id, *existing_article_ids, *article_ids]:
            if article_id not in merged_ids:
                merged_ids.append(article_id)
        if existing_rep_id not in final_representatives:
            final_representatives.append(existing_rep_id)
        existing_matches.setdefault(existing_cluster_id, []).extend(article_ids)

        log.info(
            "기존 클러스터 매칭 | new_rep=%s existing_cluster=%s existing_rep=%s similarity=%.3f",
            rep_id,
            existing_cluster_id,
            existing_rep_id,
            float(match["similarity"]),
        )

    deduped_matches = {
        cluster_id: sorted(set(article_ids)) for cluster_id, article_ids in existing_matches.items()
    }
    return final_cluster_map, final_representatives, deduped_matches


def _existing_cluster_candidate_window(
    articles: list[dict[str, Any]],
) -> tuple[str, str] | None:
    published_values = [
        parsed
        for article in articles
        if (parsed := _parse_datetime(article.get("published_at") or article.get("collected_at")))
        is not None
    ]
    if not published_values:
        return None

    normalized = [
        value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        for value in published_values
    ]
    start = min(normalized) - timedelta(days=_MAX_CLUSTER_PUBLISHED_GAP_DAYS)
    end = max(normalized) + timedelta(days=_MAX_CLUSTER_PUBLISHED_GAP_DAYS + 1)
    return start.isoformat(), end.isoformat()


def _best_existing_cluster_match(
    *,
    rep_article: dict[str, Any],
    rep_embedding: np.ndarray,
    candidates: list[dict[str, Any]],
    candidate_embeddings: np.ndarray,
) -> dict[str, Any] | None:
    similarities = candidate_embeddings @ rep_embedding
    best: dict[str, Any] | None = None
    best_similarity = -1.0

    for idx, candidate in enumerate(candidates):
        similarity = float(similarities[idx])
        if similarity < best_similarity:
            continue
        if similarity < _EXISTING_CLUSTER_MIN_SIMILARITY:
            continue
        if not _should_merge_articles(
            rep_article,
            candidate,
            similarity,
            _EXISTING_CLUSTER_THRESHOLD,
        ):
            continue
        best = candidate
        best_similarity = similarity

    if best is None:
        return None

    return {**best, "similarity": best_similarity}


def _should_merge_articles(
    left: dict[str, Any],
    right: dict[str, Any],
    similarity: float,
    threshold: float,
) -> bool:
    same_cross_company_title_issue = _same_cross_company_title_issue(left, right)
    if not _same_company_context(left, right) and not same_cross_company_title_issue:
        return False

    if not _within_cluster_time_window(left, right):
        return False

    if not _event_buckets_compatible(left, right):
        llm_decision = _cluster_llm_same_event(left, right, similarity, "event_bucket_conflict")
        if llm_decision is not None:
            return llm_decision
        return False

    if _same_issue(left, right):
        return True

    if _same_company_signature_or_concept(left, right):
        return True

    if not _event_signatures_compatible(left, right):
        if _same_company_title_fallback(left, right, similarity, threshold):
            return True
        llm_decision = _cluster_llm_same_event(left, right, similarity, "event_signature_conflict")
        if llm_decision is not None:
            return llm_decision
        return False

    if same_cross_company_title_issue:
        return similarity >= threshold

    if _same_company_title_fallback(left, right, similarity, threshold):
        return True

    if _has_weak_bridge_risk(left, right):
        return False

    if _has_topic_conflict(left, right):
        llm_decision = _cluster_llm_same_event(left, right, similarity, "topic_conflict")
        if llm_decision is not None:
            return llm_decision
        return similarity >= _HIGH_CONFIDENCE_SIMILARITY

    if similarity < threshold:
        return False

    return True


def _split_cluster_values_by_event_bucket(
    cluster_values: Any,
    id_to_article: dict[int, dict[str, Any]],
) -> list[list[int]]:
    """Prevent union-find bridge chains from creating mixed event mega-clusters."""
    split_values: list[list[int]] = []
    for article_ids in cluster_values:
        split_values.extend(_split_cluster_value_by_event_key(article_ids, id_to_article))
    return split_values


def _split_cluster_value_by_event_key(
    article_ids: list[int],
    id_to_article: dict[int, dict[str, Any]],
) -> list[list[int]]:
    if len(article_ids) <= 1:
        return [article_ids]

    ids = [int(article_id) for article_id in article_ids]
    parent = {article_id: article_id for article_id in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i, left_id in enumerate(ids):
        left_key = _event_split_key(id_to_article.get(left_id, {}))
        for right_id in ids[i + 1 :]:
            right_key = _event_split_key(id_to_article.get(right_id, {}))
            approved_pair = frozenset({left_id, right_id}) in _cluster_llm_approved_pairs
            if left_key == right_key or approved_pair:
                union(left_id, right_id)

    groups: dict[int, list[int]] = {}
    for article_id in ids:
        groups.setdefault(find(article_id), []).append(article_id)
    return list(groups.values())


def _event_buckets_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_bucket = _event_bucket(left)
    right_bucket = _event_bucket(right)
    if left_bucket == "general" or right_bucket == "general":
        return True
    return left_bucket == right_bucket


def _event_signatures_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_signature = _event_signature(left)
    right_signature = _event_signature(right)
    if left_signature.endswith(":general") or right_signature.endswith(":general"):
        return True
    left_bucket = _event_bucket(left)
    right_bucket = _event_bucket(right)
    if left_bucket == right_bucket == "ax_strategy":
        return left_signature == right_signature
    if left_bucket == right_bucket == "contract_deal":
        return left_signature == right_signature
    if left_bucket == right_bucket and left_bucket not in {
        "market_reaction",
        "investment_deal",
    }:
        return True
    return left_signature == right_signature


def _event_prefilter_key(article: dict[str, Any]) -> str:
    bucket = _event_bucket(article)
    if bucket == "contract_deal" or (
        bucket == "market_reaction" and _has_contract_markers(_issue_text(article))
    ):
        return "deal_or_market_contract"
    if bucket == "market_reaction":
        return _event_signature(article)
    if bucket == "investment_deal":
        return _event_signature(article)
    return bucket


def _event_split_key(article: dict[str, Any]) -> str:
    bucket = _event_bucket(article)
    if bucket == "contract_deal":
        return _event_signature(article)
    if bucket == "market_reaction" and _has_contract_markers(_issue_text(article)):
        return "deal_or_market_contract"
    if bucket == "market_reaction":
        return _event_signature(article)
    if bucket in {"investment_deal", "ax_strategy", "cloud_infra"}:
        return _event_signature(article)
    return bucket


def _uses_cross_day_prefilter(article: dict[str, Any]) -> bool:
    bucket = _event_bucket(article)
    if bucket in {"contract_deal", "investment_deal", "ax_strategy", "cloud_infra"}:
        return True
    return bucket == "market_reaction" and _has_contract_markers(_issue_text(article))


def _event_signature(article: dict[str, Any]) -> str:
    """Fine-grained deterministic issue key used before BGE similarity."""
    bucket = _event_bucket(article)
    title = _compact_text(str(article.get("title") or ""))
    text = _issue_text(article)

    if bucket == "market_reaction":
        return f"market_reaction:{_published_day(article)}"
    if "두나무" in text:
        return "investment_deal:dunamu"
    if bucket == "contract_deal":
        contract_key = _contract_issue_key(article)
        if contract_key:
            return f"contract_deal:{contract_key}"
    if "ax서밋" in text or "axsummit" in text:
        return "ax_strategy:ax_summit"
    if "자율공장" in text:
        return "ax_strategy:ai_factory"
    if "인더스트리데이" in text:
        return "ax_strategy:industry_day"
    title_concepts = set(_title_concepts(article))
    if "lg_cns_agentic_aind" in title_concepts:
        return "ax_strategy:lg_cns_agentic_aind"
    if "gpu_ai_highway" in title_concepts:
        return "cloud_infra:gpu_ai_highway"
    if "lg_nvidia_physical_ai" in title_concepts:
        return "ax_strategy:lg_nvidia_physical_ai"
    if "hyundai_autoever_robotics_challenge" in title_concepts:
        return "ax_strategy:hyundai_autoever_robotics_challenge"
    if "saemaul_inspection_system" in title_concepts:
        return "contract_deal:saemaul_inspection_system"
    if "jensen_huang_visit" in title_concepts:
        return "ax_strategy:jensen_huang_nc_meeting"
    if "autonomous_welding_robot" in title_concepts:
        return "ax_strategy:autonomous_welding_robot"
    if "robot_foundation_model" in title_concepts:
        return "ax_strategy:robot_foundation_model"
    if "physicalworks_rx_platform" in title_concepts:
        return "ax_strategy:physicalworks_rx_platform"
    if "manufacturing_ax_market" in title_concepts:
        return "ax_strategy:manufacturing_ax_market"
    if "smart_infra_lidar" in title_concepts:
        return f"{bucket}:smart_infra_lidar"
    if "openai_enterprise_ai" in title_concepts:
        return "ax_strategy:openai_enterprise_ai"
    if "national_ai_computing_center" in title_concepts:
        return "cloud_infra:national_ai_computing_center"
    if "데이터센터" in text or "ai인프라" in text:
        return "cloud_infra:ai_datacenter"
    if "si주" in text or "it서비스업종" in text or "테마주" in text:
        return "industry_theme:si_theme"

    title_terms = sorted(_title_topic_terms(article))
    if title_terms:
        return f"{bucket}:{title_terms[0]}"
    if title:
        return f"{bucket}:title:{title[:24]}"
    return f"{bucket}:general"


def _published_day(article: dict[str, Any]) -> str:
    published_at = _parse_datetime(article.get("published_at") or article.get("collected_at"))
    if published_at is None:
        return "unknown_day"
    return published_at.date().isoformat()


def _event_bucket(article: dict[str, Any]) -> str:
    """Coarse event bucket for card-news-level clustering.

    Embedding similarity is good at grouping same-company/theme articles, but it
    can over-merge different event layers such as market reaction, investment,
    and AX strategy. Title receives priority because it usually encodes the
    article's actual news angle.
    """
    title = _compact_text(str(article.get("title") or ""))
    title_bucket = _event_bucket_from_text(title)
    if title_bucket != "general":
        return title_bucket
    return _event_bucket_from_text(_issue_text(article))


def _event_bucket_from_text(text: str) -> str:
    if not text:
        return "general"
    for bucket, markers in _EVENT_BUCKET_PATTERNS:
        if any(_compact_text(marker) in text for marker in markers):
            return bucket
    return "general"


def _has_contract_markers(compact_text: str) -> bool:
    return any(
        marker in compact_text
        for marker in (
            "계약",
            "공급계약",
            "수주",
            "사업수주",
            "사업자선정",
            "업무협약",
            "mou",
        )
    )


def _cluster_llm_same_event(
    left: dict[str, Any],
    right: dict[str, Any],
    similarity: float,
    reason: str,
) -> bool | None:
    """Use LLM only for ambiguous same-event clustering decisions."""
    if not _should_consult_cluster_llm(left, right, similarity, reason):
        return None

    decision = _invoke_cluster_llm_judge(left, right, similarity, reason)
    if decision is True:
        _remember_cluster_llm_approval(left, right)
    return decision


def _should_consult_cluster_llm(
    left: dict[str, Any],
    right: dict[str, Any],
    similarity: float,
    reason: str,
) -> bool:
    if not _CLUSTER_LLM_JUDGE_ENABLED or not openai_calls_enabled():
        return False
    if _CLUSTER_LLM_MAX_CALLS <= 0:
        return False
    if reason == "event_signature_conflict":
        return False
    if similarity < 0.72:
        return False
    if not _within_cluster_time_window(left, right):
        return False
    return _same_company_context(left, right) or _same_cross_company_title_issue(left, right)


def _invoke_cluster_llm_judge(
    left: dict[str, Any],
    right: dict[str, Any],
    similarity: float,
    reason: str,
) -> bool | None:
    global _cluster_llm_calls

    cache_key = _cluster_llm_cache_key(left, right, reason)
    if cache_key in _cluster_llm_cache:
        return _cluster_llm_cache[cache_key]
    if _cluster_llm_calls >= _CLUSTER_LLM_MAX_CALLS:
        log.info("cluster LLM judge cap reached | cap=%d reason=%s", _CLUSTER_LLM_MAX_CALLS, reason)
        _cluster_llm_cache[cache_key] = None
        return None

    _cluster_llm_calls += 1
    payload = {
        "instruction": (
            "Decide whether the two Korean news articles describe the same underlying business "
            "event and should be in one card-news cluster. Respond as JSON only."
        ),
        "criteria": [
            "same_event=true when one article is a market reaction to the same contract/deal/news.",
            (
                "same_event=false when they are only broad themes, background mentions, "
                "or different deals."
            ),
            "Ignore minor amount wording differences if the business event is the same.",
        ],
        "reason": reason,
        "embedding_similarity": round(similarity, 4),
        "left": _cluster_llm_article_payload(left),
        "right": _cluster_llm_article_payload(right),
        "required_json_schema": {
            "same_event": "boolean",
            "confidence": "number between 0 and 1",
            "reason": "short Korean explanation",
        },
    }

    try:
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model=_CLUSTER_LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict Korean news clustering judge. Return JSON only.",
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        confidence = _safe_float(parsed.get("confidence"), 0.0)
        decision = bool(parsed.get("same_event")) and confidence >= 0.7
        _cluster_llm_cache[cache_key] = decision
        log.info(
            "cluster LLM judge | decision=%s confidence=%.2f reason=%s",
            decision,
            confidence,
            parsed.get("reason", ""),
        )
        return decision
    except Exception as e:
        log.warning("cluster LLM judge failed, using rule fallback | error=%s", e)
        _cluster_llm_cache[cache_key] = None
        return None


def _cluster_llm_article_payload(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": article.get("id"),
        "title": str(article.get("title") or "")[:240],
        "published_at": str(article.get("published_at") or article.get("collected_at") or ""),
        "companies": _company_key(article),
        "sectors": _sector_key(article),
        "event_bucket": _event_bucket(article),
        "event_signature": _event_signature(article),
    }


def _cluster_llm_cache_key(
    left: dict[str, Any],
    right: dict[str, Any],
    reason: str,
) -> tuple[str, str, str]:
    left_key = str(left.get("id") or left.get("title") or "")
    right_key = str(right.get("id") or right.get("title") or "")
    first, second = sorted((left_key, right_key))
    return first, second, reason


def _remember_cluster_llm_approval(left: dict[str, Any], right: dict[str, Any]) -> None:
    left_raw_id = left.get("id")
    right_raw_id = right.get("id")
    if left_raw_id is None or right_raw_id is None:
        return
    try:
        left_id = int(left_raw_id)
        right_id = int(right_raw_id)
    except (TypeError, ValueError):
        return
    _cluster_llm_approved_pairs.add(frozenset({left_id, right_id}))


def _within_cluster_time_window(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """오래 떨어진 반복 주제가 같은 클러스터로 묶이지 않도록 시간 간격을 제한한다."""
    left_dt = _parse_datetime(left.get("published_at") or left.get("collected_at"))
    right_dt = _parse_datetime(right.get("published_at") or right.get("collected_at"))
    if left_dt is None or right_dt is None:
        return True

    if left_dt.tzinfo is None:
        left_dt = left_dt.replace(tzinfo=timezone.utc)
    if right_dt.tzinfo is None:
        right_dt = right_dt.replace(tzinfo=timezone.utc)

    gap_days = abs((left_dt - right_dt).days)
    return gap_days <= _MAX_CLUSTER_PUBLISHED_GAP_DAYS


def _same_issue(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_key = _issue_dedup_key(left)
    if left_key and left_key == _issue_dedup_key(right):
        return True

    return _same_company_business_issue(left, right)


def _same_company_context(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_companies = set(_company_key(left))
    right_companies = set(_company_key(right))
    if not left_companies or not right_companies:
        return True
    return bool(left_companies & right_companies)


def _same_cross_company_title_issue(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """수집 타겟 company가 달라도 제목상 같은 사건이면 비교를 허용한다."""
    left_bucket = _event_bucket(left)
    right_bucket = _event_bucket(right)
    if left_bucket != right_bucket or left_bucket in {"market_reaction", "industry_theme"}:
        return False

    left_terms = _title_topic_terms(left)
    right_terms = _title_topic_terms(right)
    if left_terms and right_terms and _topic_sets_related(left_terms, right_terms):
        return True

    if set(_title_concepts(left)) & set(_title_concepts(right)):
        return True

    return False


def _same_company_title_fallback(
    left: dict[str, Any],
    right: dict[str, Any],
    similarity: float,
    threshold: float,
) -> bool:
    if similarity < min(threshold, 0.68):
        return False
    if not _same_company_context(left, right):
        return False

    left_bucket = _event_bucket(left)
    right_bucket = _event_bucket(right)
    if left_bucket != right_bucket or left_bucket in {"market_reaction", "industry_theme"}:
        return False

    left_signature = _event_signature(left)
    right_signature = _event_signature(right)
    left_specific = (
        not left_signature.endswith(":general")
        and ":title:" not in left_signature
        and ":proper:" not in left_signature
    )
    right_specific = (
        not right_signature.endswith(":general")
        and ":title:" not in right_signature
        and ":proper:" not in right_signature
    )
    if left_specific and right_specific:
        return False

    left_terms = _title_topic_terms(left)
    right_terms = _title_topic_terms(right)
    if left_terms and right_terms and _topic_sets_related(left_terms, right_terms):
        return True

    if set(_title_concepts(left)) & set(_title_concepts(right)):
        return True

    return _title_tokens_related(left, right)


def _same_company_signature_or_concept(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not _same_company_context(left, right):
        return False

    if _has_specific_ax_signature_conflict(left, right):
        return False

    left_signature = _event_signature(left)
    right_signature = _event_signature(right)
    if (
        left_signature == right_signature
        and not left_signature.endswith(":general")
        and ":title:" not in left_signature
        and ":proper:" not in left_signature
    ):
        return True

    strong_concepts = {
        "security_token_platform",
        "national_ai_computing_center",
        "openai_enterprise_ai",
        "physicalworks_rx_platform",
        "robot_foundation_model",
        "logistics_robotics",
        "manufacturing_ax_market",
        "smart_infra_lidar",
    }
    shared_concepts = set(_title_concepts(left)) & set(_title_concepts(right))
    return bool(shared_concepts & strong_concepts)


def _has_weak_bridge_risk(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _has_ambiguous_support_shape(left) or _has_ambiguous_support_shape(right)


def _has_ambiguous_support_shape(article: dict[str, Any]) -> bool:
    """제목은 특정 이슈를 못 잡고 본문 초반에 여러 이슈가 섞인 기사."""
    if _title_topic_terms(article):
        return False
    return len(_full_topic_terms(article)) > _MAX_BRIDGE_TOPIC_TERMS


def _title_tokens_related(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_tokens = _title_event_tokens(left)
    right_tokens = _title_event_tokens(right)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False

    shared = left_tokens & right_tokens
    high_signal_tokens = {
        "skala",
        "토큰증권",
        "예탁결제원",
        "두나무",
        "openai",
        "chatgpt",
        "피지컬웍스",
    }
    if shared & high_signal_tokens:
        return True

    if len(shared) < 2:
        return False

    jaccard = len(shared) / len(left_tokens | right_tokens)
    coverage = len(shared) / min(len(left_tokens), len(right_tokens))
    return jaccard >= 0.35 or coverage >= 0.55


def _title_event_tokens(article: dict[str, Any]) -> set[str]:
    title = str(article.get("title") or "").lower()
    tokens = {
        _normalize_title_token(token)
        for token in re.findall(r"[가-힣A-Za-z0-9]+", title)
        if token.strip()
    }
    tokens = {token for token in tokens if _useful_title_token(token)}

    compact_title = _compact_text(title)
    for marker in (
        "두나무",
        "오픈ai",
        "openai",
        "챗gpt",
        "피지컬웍스",
        "토큰증권",
        "토큰증권플랫폼",
        "스마트팩토리",
        "스마트인프라",
        "예탁결제원",
        "예탁원",
        "로봇파운데이션",
        "로봇브레인",
        "로봇두뇌",
        "로봇지능",
        "지분투자",
        "지분인수",
    ):
        compact_marker = _compact_text(marker)
        if compact_marker in compact_title:
            tokens.add(compact_marker)

    return tokens


def _normalize_title_token(token: str) -> str:
    compact = _compact_text(token.lower())
    aliases = {
        "엔씨": "nc",
        "엔씨ai": "ncai",
        "nc": "nc",
        "ncai": "ncai",
        "포스코dx": "poscodx",
        "poscodx": "poscodx",
        "삼성에스디에스": "samsungsds",
        "삼성sds": "samsungsds",
        "lgcns": "lgcns",
        "lg씨엔에스": "lgcns",
        "오픈ai": "openai",
        "챗gpt": "chatgpt",
        "인공지능": "ai",
        "피지컬ai": "physicalai",
        "스칼라": "skala",
        "skala": "skala",
        "예탁원": "예탁결제원",
        "예탁결제원": "예탁결제원",
        "sto": "토큰증권",
    }
    return aliases.get(compact, compact)


def _useful_title_token(token: str) -> bool:
    if len(token) < 2 or token.isdigit():
        return False
    stopwords = {
        "단독",
        "종합",
        "속보",
        "현장",
        "포토",
        "영상",
        "이슈",
        "특징주",
        "관련주",
        "상승",
        "하락",
        "급등",
        "급락",
        "강세",
        "약세",
        "공개",
        "추진",
        "개발",
        "협력",
        "협업",
        "맞손",
        "체결",
        "공동",
        "나서",
        "한다",
        "위해",
        "기술",
        "시장",
        "사업",
    }
    return token not in stopwords


def _has_topic_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_title_terms = _title_topic_terms(left)
    right_title_terms = _title_topic_terms(right)
    if not left_title_terms or not right_title_terms:
        return False

    if _topic_sets_related(left_title_terms, right_title_terms):
        return False

    left_full_terms = _full_topic_terms(left)
    right_full_terms = _full_topic_terms(right)
    if _topic_sets_related(left_full_terms, right_title_terms) or _topic_sets_related(
        right_full_terms,
        left_title_terms,
    ):
        return False

    return True


def _support_only_singleton(
    article_ids: list[int],
    id_to_article: dict[int, dict[str, Any]],
) -> bool:
    if _KEEP_AMBIGUOUS_SINGLETONS:
        # singleton 은 항상 cluster 로 유지. bridge risk 는 multi-article merge 시점에서만 의미.
        return False

    if len(article_ids) != 1:
        return False

    article = id_to_article.get(article_ids[0], {})
    return (
        not _title_topic_terms(article)
        and len(_full_topic_terms(article)) > _MAX_BRIDGE_TOPIC_TERMS
    )


def _issue_dedup_key(article: dict[str, Any]) -> str | None:
    entities = _issue_entities(article)
    companies = entities["companies"]
    if not companies:
        return None

    for issue in entities["canonical_issues"]:
        return f"{','.join(companies)}::{issue}"

    for quoted in entities["quoted_terms"]:
        return f"{','.join(companies)}::quoted::{quoted}"

    return None


def _issue_entities(article: dict[str, Any]) -> dict[str, list[str]]:
    """클러스터링에 쓰는 가벼운 엔티티를 제목 중심으로 추출한다."""
    title_text = _title_text(article)
    return {
        "companies": _company_key(article),
        "sectors": _sector_key(article),
        "canonical_issues": _matched_alias_keys(_CANONICAL_ISSUE_TERMS, title_text),
        "quoted_terms": _normalize_quoted_product_terms(
            _quoted_terms_from_text(str(article.get("title") or ""))
        ),
        "proper_terms": _proper_terms_from_text(str(article.get("title") or "")),
        "numbers": _number_terms_from_text(str(article.get("title") or "")),
        "concepts": _concepts_from_text(title_text),
    }


def _title_issue_entities(article: dict[str, Any]) -> dict[str, list[str]]:
    """임베딩용 제목 기반 엔티티.

    본문 기반 엔티티는 업계 배경과 관련 종목까지 끌어와 cluster bridge를 만들 수
    있으므로 임베딩 입력에는 제목에서 드러난 사건 신호만 넣는다.
    """
    title = str(article.get("title") or "")
    title_text = _compact_text(title)
    return {
        "companies": _company_key(article),
        "sectors": _sector_key(article),
        "canonical_issues": _matched_alias_keys(_CANONICAL_ISSUE_TERMS, title_text),
        "quoted_terms": _normalize_quoted_product_terms(_quoted_terms_from_text(title)),
        "proper_terms": _proper_terms_from_text(title),
        "numbers": _number_terms_from_text(title),
        "concepts": _concepts_from_text(title_text),
    }


def _company_key(article: dict[str, Any]) -> list[str]:
    value = article.get("matched_companies") or article.get("company") or []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return _canonical_company_keys(parsed)
        except json.JSONDecodeError:
            pass
        return _canonical_company_keys([stripped])
    if isinstance(value, (list, tuple)):
        return _canonical_company_keys(value)
    return []


def _canonical_company_keys(values: list[Any] | tuple[Any, ...]) -> list[str]:
    company_keys: set[str] = set()
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        company_keys.add(_canonical_company_key(raw))
    return sorted(company_keys)


def _canonical_company_key(value: str) -> str:
    value_compact = _compact_text(value)
    for company_id, aliases in _ALL_COMPANY_ALIASES.items():
        if value == company_id or value_compact == _compact_text(company_id):
            return company_id
        if any(value_compact == _compact_text(alias) for alias in aliases):
            return company_id
    return value


def _sector_key(article: dict[str, Any]) -> list[str]:
    sectors = _list_value(article.get("matched_sectors"))
    return sorted({sector for sector in sectors if sector and sector != "other"})


def _list_value(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass
        return [stripped]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return []


def _matched_alias_keys(alias_map: Mapping[str, tuple[str, ...]], compact_text: str) -> list[str]:
    matched: list[str] = []
    for key, aliases in alias_map.items():
        compact_aliases = [_compact_text(alias) for alias in aliases]
        if any(alias and alias in compact_text for alias in compact_aliases):
            matched.append(key)
    return matched


def _quoted_product_terms(article: dict[str, Any]) -> list[str]:
    title = str(article.get("title") or "")
    content = str(article.get("content") or "")
    quoted_terms = _quoted_terms_from_text(f"{title} {content}")
    return _normalize_quoted_product_terms(quoted_terms)


def _quoted_terms_from_text(text: str) -> list[str]:
    return re.findall(r"['‘’\"“”「」](.{2,40}?)['‘’\"“”「」]", text)


def _normalize_named_terms(raw_terms: list[str], min_length: int = 5) -> list[str]:
    terms: list[str] = []
    for term in raw_terms:
        compact = _compact_text(term)
        if len(compact) >= min_length and not compact.isdigit() and compact not in terms:
            terms.append(compact)
    return terms


def _normalize_quoted_product_terms(raw_terms: list[str]) -> list[str]:
    terms = _normalize_named_terms(raw_terms, min_length=4)
    for term in raw_terms:
        compact = _compact_text(term)
        if _is_short_mixed_script_name(term) and compact not in terms and not compact.isdigit():
            terms.append(compact)
    return terms


def _is_short_mixed_script_name(term: str) -> bool:
    compact = _compact_text(term)
    if len(compact) < 3:
        return False
    return bool(re.search(r"[가-힣]", term) and re.search(r"[A-Za-z]", term))


def _proper_terms(article: dict[str, Any]) -> list[str]:
    title = str(article.get("title") or "")
    content = str(article.get("content") or "")
    return _proper_terms_from_text(f"{title} {content}")


def _title_concepts(article: dict[str, Any]) -> list[str]:
    return _concepts_from_text(_compact_text(str(article.get("title") or "")))


def _concepts_from_text(compact_text: str) -> list[str]:
    concepts: list[str] = []
    for concept, markers in _TITLE_CONCEPT_TERMS.items():
        if any(_compact_text(marker) in compact_text for marker in markers):
            concepts.append(concept)
    return concepts


def _proper_terms_from_text(text: str) -> list[str]:
    terms: list[str] = []
    patterns = (
        r"[가-힣A-Za-z0-9]+(?:\s*[가-힣A-Za-z0-9]+){0,4}\s*(?:클라우드|센터|플랫폼|시스템|솔루션|사업|컨소시엄|서비스|기술|프로젝트|반도체|칩|로봇|모델|엔진|두뇌|계약|공급|웹단말|전환)",
    )
    for pattern in patterns:
        for term in re.findall(pattern, text, flags=re.IGNORECASE):
            compact = _compact_text(term)
            if len(compact) >= 5 and compact not in terms:
                terms.append(compact)
    return terms


def _title_topic_terms(article: dict[str, Any]) -> set[str]:
    return _topic_terms_from_text(str(article.get("title") or ""))


def _full_topic_terms(article: dict[str, Any]) -> set[str]:
    title = str(article.get("title") or "")
    content = str(article.get("content") or "")
    return _topic_terms_from_text(f"{title} {content}")


def _topic_terms_from_text(text: str) -> set[str]:
    terms: set[str] = set()

    for term in _proper_terms_from_text(text):
        if not _is_weak_named_topic(term):
            terms.add(f"proper:{term}")

    return terms


def _is_weak_named_topic(term: str) -> bool:
    weak_terms = {"ai", "ax", "dx"}
    return term in weak_terms or len(term) < 4


def _topic_sets_related(left: set[str], right: set[str]) -> bool:
    for left_term in left:
        for right_term in right:
            if _terms_related(left_term, right_term):
                return True
    return False


def _terms_related(left: str, right: str) -> bool:
    left_value = _topic_value(left)
    right_value = _topic_value(right)
    if not left_value or not right_value:
        return False
    return (
        left_value == right_value
        or len(left_value) >= _MIN_RELATED_TERM_LENGTH
        and left_value in right_value
        or len(right_value) >= _MIN_RELATED_TERM_LENGTH
        and right_value in left_value
        or _ngram_similarity(left_value, right_value) >= _TERM_NGRAM_SIMILARITY
    )


def _topic_value(term: str) -> str:
    return _normalize_topic_value(term.split(":", 1)[-1])


def _normalize_topic_value(value: str) -> str:
    normalized = re.sub(r"\d[\d,]*(?:조|억|만|천|%|장|gw|원|년)?", "", value)
    for marker in (
        "서비스전환",
        "전환프로젝트",
        "프로젝트착수",
        "고객서비스",
        "구축사업",
        "운영사업",
        "공급계약",
        "사업수주",
        "시장공략",
        "공략가속",
    ):
        normalized = normalized.replace(marker, "")
    return normalized or value


def _ngram_similarity(left: str, right: str, n: int = 3) -> float:
    if len(left) < _MIN_RELATED_TERM_LENGTH or len(right) < _MIN_RELATED_TERM_LENGTH:
        return 0.0

    left_grams = _char_ngrams(left, n)
    right_grams = _char_ngrams(right, n)
    if not left_grams or not right_grams:
        return 0.0

    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _char_ngrams(value: str, n: int) -> set[str]:
    if len(value) <= n:
        return {value}
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def _number_terms(article: dict[str, Any]) -> list[str]:
    text = f"{article.get('title') or ''} {article.get('content') or ''}"
    return _number_terms_from_text(text)


def _number_terms_from_text(text: str) -> list[str]:
    numbers: list[str] = []
    for term in re.findall(r"\d[\d,]*(?:조|억|만|천|%|장|gw|원|년)?", text, flags=re.IGNORECASE):
        compact = _normalize_number_term(_compact_text(term))
        if len(compact) >= 2 and compact not in numbers:
            numbers.append(compact)
    return numbers


def _normalize_number_term(value: str) -> str:
    if value.endswith("억원"):
        return value[:-1]
    return value


def _contract_issue_key(article: dict[str, Any]) -> str | None:
    title_text = _title_text(article)
    fallback_text = _title_with_short_lead_text(article)
    amount_match = re.search(r"\d+(?:\.\d+)?(?:억|억원|원|만|천)", title_text)
    amount = _normalize_number_term(amount_match.group(0)) if amount_match else ""
    concepts = _event_terms(article, include_lead=False)
    for concept in sorted(concepts):
        return concept
    domain_key = _contract_domain_key(title_text) or _contract_domain_key(fallback_text)
    if domain_key:
        return domain_key
    terms = sorted(_title_topic_terms(article))
    term = _topic_value(terms[0]) if terms else ""
    if term:
        return term
    if amount:
        return amount
    return None


def _contract_domain_key(compact_text: str) -> str | None:
    if "인증중고차" in compact_text or "cpo" in compact_text:
        return "certified_used_car_platform"
    if "코어뱅킹" in compact_text or "웹단말" in compact_text:
        return "core_banking_web_terminal"
    return None


def _same_company_business_issue(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not _company_key(left) or _company_key(left) != _company_key(right):
        return False

    if _has_specific_ax_signature_conflict(left, right):
        return False

    left_entities = _issue_entities(left)
    right_entities = _issue_entities(right)

    if _shared(left_entities["canonical_issues"], right_entities["canonical_issues"]):
        return True

    if _shared(left_entities["quoted_terms"], right_entities["quoted_terms"]):
        return True

    if _shared(left_entities["concepts"], right_entities["concepts"]):
        return True

    if _same_operational_event(left, right):
        return True

    if _event_bucket(left) == _event_bucket(right) == "contract_deal":
        if set(left_entities["numbers"]) & set(right_entities["numbers"]):
            return True
        if _terms_have_relation(left_entities["proper_terms"], right_entities["proper_terms"]):
            return True

    shared_proper_terms = _terms_have_relation(
        left_entities["proper_terms"],
        right_entities["proper_terms"],
    )
    shared_numbers = set(left_entities["numbers"]) & set(right_entities["numbers"])
    if shared_proper_terms and (
        shared_numbers or _shared(left_entities["sectors"], right_entities["sectors"])
    ):
        return True

    return False


def _has_specific_ax_signature_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _event_bucket(left) != "ax_strategy" or _event_bucket(right) != "ax_strategy":
        return False
    left_signature = _event_signature(left)
    right_signature = _event_signature(right)
    if left_signature == right_signature:
        return False
    return (
        not left_signature.endswith(":general")
        and not right_signature.endswith(":general")
        and ":title:" not in left_signature
        and ":title:" not in right_signature
    )


def _same_operational_event(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _event_bucket(left) == _event_bucket(right) == "ax_strategy":
        left_signature = _event_signature(left)
        right_signature = _event_signature(right)
        if (
            not left_signature.endswith(":general")
            and not right_signature.endswith(":general")
            and left_signature != right_signature
        ):
            return False

    left_terms = _event_terms(left, include_lead=False)
    right_terms = _event_terms(right, include_lead=False)
    shared_terms = left_terms & right_terms
    if not shared_terms:
        return False

    left_text = _title_text(left)
    right_text = _title_text(right)
    if not (_has_operational_action(left_text) and _has_operational_action(right_text)):
        return False

    if len(shared_terms) >= 2:
        return True

    strong_terms = {
        "logistics_robotics",
        "physicalworks_rx_platform",
        "manufacturing_ax_market",
        "smart_infra_lidar",
        "openai_enterprise_ai",
        "robot_foundation_model",
        "national_ai_computing_center",
        "security_token_platform",
    }
    return bool(shared_terms & strong_terms)


def _event_terms(article: dict[str, Any], *, include_lead: bool = False) -> set[str]:
    text = _title_with_short_lead_text(article) if include_lead else _title_text(article)
    terms = set(_concepts_from_text(text))
    if "물류" in text and ("로봇" in text or "휴머노이드" in text or "피지컬웍스" in text):
        terms.add("logistics_robotics")
    if "피지컬웍스" in text or (
        ("rx" in text or "로봇" in text) and ("학습" in text or "운영" in text or "플랫폼" in text)
    ):
        terms.add("physicalworks_rx_platform")
    if ("제조" in text or "공장" in text or "스마트팩토리" in text) and (
        "ax" in text or "rx" in text or "ai" in text or "지능화" in text or "플랫폼" in text
    ):
        terms.add("manufacturing_ax_market")
    if ("스마트인프라" in text or "스마트시티" in text or "라이다" in text) and (
        "북미" in text or "에스오에스랩" in text or "lgcns" in text
    ):
        terms.add("smart_infra_lidar")
    if ("오픈ai" in text or "openai" in text or "챗gpt" in text) and (
        "skax" in text or "엔터프라이즈" in text or "기업용" in text or "생성형" in text
    ):
        terms.add("openai_enterprise_ai")
    if ("국가" in text and "ai컴퓨팅" in text) or "ai고속도로" in text:
        terms.add("national_ai_computing_center")
    if ("토큰증권" in text or "sto" in text) and ("예탁결제원" in text or "예탁원" in text):
        terms.add("security_token_platform")
    return terms


def _has_operational_action(compact_text: str) -> bool:
    return any(
        marker in compact_text
        for marker in (
            "계약",
            "수주",
            "협력",
            "협업",
            "협약",
            "업무협약",
            "맞손",
            "도입",
            "실증",
            "검증",
            "poc",
            "추진",
            "착수",
            "개발",
            "구축",
            "공급",
            "적용",
            "공략",
            "확대",
            "정조준",
            "지원",
            "강화",
            "진출",
        )
    )


def _issue_text(article: dict[str, Any]) -> str:
    return _compact_text(
        " ".join(
            [
                str(article.get("title") or ""),
                str(article.get("content") or ""),
            ]
        )
    )


def _title_text(article: dict[str, Any]) -> str:
    return _compact_text(str(article.get("title") or ""))


def _title_with_short_lead_text(article: dict[str, Any]) -> str:
    lead = _clean_space(_content_text(article))[:_CLUSTER_LLM_CONTENT_CHARS]
    return _compact_text(f"{article.get('title') or ''} {lead}")


def _shared(left: list[str], right: list[str]) -> bool:
    return bool(set(left) & set(right))


def _terms_have_relation(left: list[str], right: list[str]) -> bool:
    return any(_terms_related(left_term, right_term) for left_term in left for right_term in right)


def _compact_text(value: str) -> str:
    return re.sub(r"[\s·'‘’\"“”「」()\[\]{}:：,._\-…]+", "", value.lower())


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _select_representatives(
    cluster_map: dict[int, list[int]],
    articles: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> list[int]:
    """각 클러스터에서 대표 기사를 선정한다.

    대표 기사 기준:
    1. RelevanceEvaluator가 계산한 relevance_score
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
    content = _compact_text(str(article.get("content") or ""))

    aliases = [
        _compact_text(alias)
        for company_id in companies
        for alias in _ALL_COMPANY_ALIASES.get(company_id, [company_id])
    ]

    if any(alias and alias in title for alias in aliases):
        return 1.0

    if any(alias and alias in content for alias in aliases):
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


def _persist_existing_card_updates(existing_matches: dict[int, list[int]]) -> None:
    """기존 클러스터에 붙은 신규 기사를 card_news sources에 반영한다."""
    for cluster_id, article_ids in existing_matches.items():
        merge_card_news_sources_for_cluster(cluster_id, article_ids)
