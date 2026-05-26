# HybridSearchService — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `HybridSearchService` |
| **Supervisor** | UserQuery (Search + Dialogue 공통) |
| **상태** | ✅ stub 구현 (`src/rag/hybrid_search.py`) — 실제 Qdrant query 호출은 P6 |
| **Trigger** | User 검색박스 입력 + FloatingAiChat |

## 2. 책임

**한 줄**: 자연어 쿼리 → BGE-M3 Dense + Sparse 임베딩 → Qdrant `axis_main` 컬렉션에서 RRF (Reciprocal Rank Fusion) 으로 top-50 정확 검색.

**구체적**:

1. 쿼리 임베딩 (BGE-M3 원샷)
2. Qdrant Dense prefetch (top 50, cosine)
3. Qdrant Sparse prefetch (top 50, dot product)
4. RRF fusion — score(d) = 1/(60+rank_dense) + 1/(60+rank_sparse)
5. 중복 제거 후 top 20 메타데이터 + payload 리턴 (RerankService 입력)
6. 메타데이터 필터 (peer_id / event_type / sector / 날짜) 지원

## 3. 책임 NOT

- 결과 재정렬 — RerankService (다음 단계)
- 답변 생성 — AnswerService
- 키워드 자동완성 — SearchSuggestService

## 4. 입력 스펙

```python
class HybridSearchInput(TypedDict):
    query: str
    filters: dict | None       # {peer_id, event_type, sector, since, until}
    top_k: int                  # 기본 50 (prefetch), final 20
```

## 5. 출력 스펙

```python
class SearchHit(TypedDict):
    card_news_id: str
    title: str
    summary: str
    peer_id: str
    sector: str
    event_type: str
    exposure_score: float
    published_at: str
    cluster_id: int
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None

class HybridSearchOutput(TypedDict):
    hits: list[SearchHit]
    total: int
    query: str
```

frontend `POST /api/search` 응답.

## 6. 알고리즘

```python
RRF_K = 60

def hybrid_search(query: str, filters=None, top_k=50):
    # 1. 임베딩
    emb = BGE_M3.encode([query])[0]   # dense + sparse

    # 2. Qdrant Dense prefetch
    dense_results = qdrant.search(
        collection_name="axis_main",
        query_vector=("dense", emb["dense"]),
        query_filter=build_filter(filters),
        limit=top_k,
    )

    # 3. Qdrant Sparse prefetch
    sparse_results = qdrant.search(
        collection_name="axis_main",
        query_vector=("sparse", SparseVector(emb["sparse"]["indices"], emb["sparse"]["values"])),
        query_filter=build_filter(filters),
        limit=top_k,
    )

    # 4. RRF fusion
    rrf_scores = {}
    for rank, r in enumerate(dense_results):
        rrf_scores[r.id] = rrf_scores.get(r.id, 0) + 1.0 / (RRF_K + rank)
    for rank, r in enumerate(sparse_results):
        rrf_scores[r.id] = rrf_scores.get(r.id, 0) + 1.0 / (RRF_K + rank)

    # 5. Top 20
    all_results = {r.id: r for r in dense_results + sparse_results}
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    hits = []
    for point_id, score in fused:
        r = all_results[point_id]
        hits.append({
            "card_news_id": r.payload.get("card_news_id"),
            "title": r.payload["title"],
            "summary": r.payload["summary"],
            "peer_id": r.payload["company"],
            ...
            "rrf_score": score,
            "dense_rank": next((i for i,d in enumerate(dense_results) if d.id == point_id), None),
            "sparse_rank": next((i for i,s in enumerate(sparse_results) if s.id == point_id), None),
        })

    return {"hits": hits, "total": len(rrf_scores), "query": query}
```

### Fallback (Qdrant unavailable)

- BGE-M3 embedding 실패 → BM25 PostgreSQL full-text 검색 fallback (3초 timeout)
- 결과 0건 → 날짜 범위 2배 확장 후 재시도

## 7. LLM 모델 + token 예산

- **LLM 미사용** — BGE-M3 임베딩 + Qdrant
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| BGE-M3 timeout 3초 | BM25 fallback |
| Qdrant connection error | BM25 fallback + log |
| Filter 가 너무 narrow → 0건 | window 2배 확장 후 재시도 |
| 쿼리 빈문자 | 400 BadRequest |

## 9. 외부 의존성

- **Model**: BGE-M3
- **Qdrant**: `axis_main` 컬렉션
- **PG**: BM25 fallback (full-text index on `card_news.title + summary_lines`)

## 10. State 흐름

UserQuery state — SearchSupervisor 의 LangGraph 첫 노드.

```python
class UserQueryState(TypedDict):
    query: str
    filters: dict | None
    hits: list[SearchHit]        # HybridSearch output
    reranked: list[SearchHit]    # Rerank output
    answer: str | None           # Answer output
    sc_iter: int                  # SC 카운트 (gen-search)
    confidence: float
```

## 11. Provenance + Confidence

- **Provenance**: response.metadata.search_engine='hybrid_rrf_v1', model='bge-m3-v1.5'
- **Confidence**: rrf_score (top result 가 ≥ 0.05 이면 OK, 아니면 weak match)

## 12. 테스트 시나리오

| Unit | "삼성SDS AX" 쿼리 | hits 에 samsung_sds 카드 top |
| Unit | filters={peer_id:'lg_cns'} | hits 모두 lg_cns |
| Unit | 0건 결과 | window 확장 후 재시도 → 1+ 건 |
| Edge | Qdrant down | BM25 fallback 동작 |
| Edge | 빈 쿼리 | 400 |

## 13. 모니터링

- KPI:
  - 평균 latency ≤ 200ms
  - Hit@5 ≥ 0.80 (4주차 목표) / 0.90 (5주차)
  - MRR ≥ 0.65 / 0.75
- token: ₩0

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/rag/hybrid_search.py`
- Qdrant client wrapper: `src/db/qdrant_client.py`

### Changelog

- **v1 (2026-04-W3)** — stub
- **v2 (P6 활성)** — RRF + filter + fallback
