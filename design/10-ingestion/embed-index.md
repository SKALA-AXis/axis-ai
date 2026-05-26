# EmbedIndexService — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `EmbedIndexService` (rag/embedder + rag/vector_index 통합 view) |
| **Supervisor** | Ingestion + Search 양쪽 사용 |
| **LangGraph node** | `vector_index` (#8) |
| **상태** | ✅ 구현 — `src/rag/embedder.py` (BGE-M3 wrapper) + `src/rag/vector_index.py` (Qdrant upsert) |
| **Trigger** | EvidenceBuilder 통과 카드 (pass=true) 마다 |

## 2. 책임

**한 줄**: evidence pass 통과한 card_news 의 (title + summary) 를 BGE-M3 로 Dense + Sparse 임베딩 후 Qdrant `axis_main` 컬렉션에 upsert.

**구체적**:

1. BGE-M3 원샷 임베딩 — Dense (1024d) + Sparse 동시 생성
2. payload 구성 — rdb_id, peer_id, event_type, sector, exposure_score, exposure_band, cluster_id, title, summary, card_news_id (legacy: issue_card_id), credibility_score, published_at
3. Qdrant upsert (point_id = UUID)
4. `update_classification(article_id, importance, importance_score, qdrant_vector_id=point_id)` 호출하여 raw_articles 의 `qdrant_vector_id` 컬럼 갱신
5. `axis_main` 90일 TTL · `axis_history` 365일 TTL (W6+ 활성)

## 3. 책임 NOT

- 임베딩 모델 학습 — BGE-M3 사용 only (BAAI 사전 학습)
- 검색 자체 — HybridSearchService (Search supervisor)
- payload 의 원문 저장 — 메타데이터만 (원문은 PostgreSQL)

## 4. 입력 스펙

```python
class EmbedIndexInput(TypedDict):
    card_news: list[CardNewsRow]
    evidence_results: list[EvidenceResult]   # pass=true 만 필터
```

## 5. 출력 스펙

```python
class EmbedIndexOutput(TypedDict):
    indexed_vector_ids: list[str]   # Qdrant point_id (UUID) 목록
```

`raw_articles.qdrant_vector_id` 컬럼 갱신 (대표 기사 row).

Qdrant point 구조:
```python
{
    "id": uuid4(),
    "vector": {
        "dense": [1024 float],
        "sparse": {"indices": [...], "values": [...]}
    },
    "payload": {
        "rdb_id": int,                # raw_articles.id (대표 기사)
        "card_news_id": str,          # card_news.id (legacy: issue_card_id; v1.1 design 에서 정정)
        "company": str,               # peer_id
        "event_type": str,
        "sector": str,
        "exposure_score": float,
        "exposure_band": str,
        "credibility_score": float,
        "published_at": int,          # Unix ts
        "cluster_id": int,
        "source_name": str,
        "title": str,                 # ≤ 500 chars
        "summary": str,               # 3줄 summary join
    }
}
```

## 6. 알고리즘

### 6.1 BGE-M3 임베딩 (산식)

```python
from FlagEmbedding import BGEM3FlagModel

class EmbeddingService:
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def encode(self, texts: list[str]) -> list[dict]:
        result = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [{
            "dense": result["dense_vecs"][i].tolist(),
            "sparse": {
                "indices": list(result["lexical_weights"][i].keys()),
                "values": list(result["lexical_weights"][i].values()),
            }
        } for i in range(len(texts))]
```

### 6.2 Qdrant upsert

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SparseVector

def index_card(card, embedding, payload):
    point = PointStruct(
        id=str(uuid4()),
        vector={"dense": embedding["dense"], "sparse": SparseVector(
            indices=embedding["sparse"]["indices"],
            values=embedding["sparse"]["values"],
        )},
        payload=payload,
    )
    qdrant.upsert(collection_name="axis_main", points=[point])
    return point.id
```

### 6.3 TTL / Pruning

```python
# nightly (cron 03:00) - axis_main 90일 초과 → 삭제
qdrant.delete(
    collection_name="axis_main",
    points_selector=FilterSelector(filter=Filter(
        must=[FieldCondition(key="published_at", range=Range(lt=now - 90d))]
    ))
)
# axis_main 에서 삭제된 row 중 1년 미만은 axis_history 로 archive
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — BGE-M3 + Qdrant (둘 다 in-cluster)
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| BGE-M3 모델 로드 실패 (HuggingFace 차단) | Pod startup 시 fail-fast |
| Qdrant connection error | retry 3회 (exponential backoff) → 실패 시 `errors` append, raw 보존 |
| sparse vector indices/values 길이 mismatch | log + skip 해당 카드 |
| payload size > Qdrant 한도 (4MB) | title/summary trim |

## 9. 외부 의존성

- **Model**: BGE-M3 (BAAI/bge-m3) — Pod 캐시 필요
- **lib**: `FlagEmbedding>=1.2`, `qdrant-client>=1.9`
- **DB**: Qdrant `axis_main` 컬렉션 (HTTP `qdrant:6333` ClusterIP)
- **DB**: `raw_articles.qdrant_vector_id` 컬럼 UPDATE

## 10. State 흐름 (LangGraph)

**소비**: `card_news`, `evidence_results`
**생산**: `indexed_vector_ids`

`@_logged_step("vector_index", "card_news", "indexed_vector_ids")` decorator.

## 11. Provenance + Confidence

- **Provenance**: Qdrant payload 자체가 provenance (rdb_id 가 raw_articles 로 reverse lookup)
- **Confidence**: 임베딩 자체에는 confidence 없음. Search 단계에서 cosine score 가 confidence proxy.

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | card 1건 → embed | dense.shape=(1024,), sparse indices≥10 |
| Unit | card 1건 → Qdrant upsert | get_point 가 정확한 payload 리턴 |
| Unit | evidence_pass=false 카드 | skip (indexed_vector_ids 에 없음) |
| Integration | 5 card pass 통과 → 5 point | Qdrant `axis_main` count +5 |
| Edge | title 빈 string | embedding 정상 (summary 만으로) |

## 13. 모니터링

- **pipeline_logs.step**: `vector_index`
- **KPI**:
  - 임베딩 latency ≤ 100ms/card
  - Qdrant upsert 성공률 ≥ 99%
  - axis_main 컬렉션 size 모니터 (3개월 hot 목표)
- **token 예산**: 해당 없음

## 14. 구현 메모 + Changelog

### 의존 lib

```toml
FlagEmbedding = ">=1.2"
qdrant-client = ">=1.9"
torch = ">=2.0"
```

### 핵심 파일

- `src/rag/embedder.py` — BGE-M3 wrapper (singleton)
- `src/rag/vector_index.py` — `index_card()` 함수, Qdrant upsert
- LangGraph 호출: `src/pipeline/ingestion_graph.py` `vector_index_node`

### Changelog

- **v1 (2026-04-W3)** — BGE-M3 Dense + Sparse + Qdrant 도입
- **v2 (2026-05-12)** — V9 rename 후 payload key `issue_card_id` → `card_news_id` 갱신
- **v3 (제안)** — axis_history 컬렉션 분리 (W7+ 약신호 분석용)
