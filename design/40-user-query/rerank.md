# RerankAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `RerankAgent` |
| **Supervisor** | UserQuery |
| **상태** | ✅ stub 구현 (`src/rag/reranker.py`) — P6 활성 |
| **Trigger** | HybridSearch 후 |

## 2. 책임

**한 줄**: HybridSearch 의 top 20 결과를 BGE-reranker-v2-m3 cross-encoder 로 재정렬하여 top 10.

**구체적**:

1. (query, doc) 쌍 모두에 cross-encoder forward pass (BGE-reranker-v2-m3)
2. relevance score 산출 → 재정렬
3. top 10 만 반환 (downstream AnswerAgent 또는 frontend hits 표시)

## 3. 책임 NOT

- 검색 자체 — HybridSearchAgent (이전)
- 답변 — AnswerAgent
- LLM 호출 — Cross-encoder 모델 (transformers) 만, LLM 아님

## 4. 입력 스펙

```python
class RerankInput(TypedDict):
    query: str
    hits: list[SearchHit]    # HybridSearch output (20개)
    top_k: int                # 기본 10
```

## 5. 출력 스펙

```python
class RerankedHit(SearchHit):
    rerank_score: float       # cross-encoder 점수

class RerankOutput(TypedDict):
    hits: list[RerankedHit]   # 재정렬된 top 10
    query: str
```

## 6. 알고리즘

```python
from FlagEmbedding import FlagReranker

class RerankAgent:
    def __init__(self):
        self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    def rerank(self, query: str, hits: list[SearchHit], top_k=10):
        pairs = [[query, hit.title + " " + hit.summary] for hit in hits]
        scores = self.reranker.compute_score(pairs, normalize=True)  # 0~1

        reranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return {
            "hits": [{**hit.dict(), "rerank_score": score} for hit, score in reranked[:top_k]],
            "query": query,
        }
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — Cross-encoder (BGE-reranker-v2-m3, MIT)
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| Reranker 모델 로드 실패 | Pod startup fail-fast |
| hits 빈 list | empty 그대로 |
| Cross-encoder OOM (20 pairs 가 너무 크면) | batch 10 씩 분할 |

## 9. 외부 의존성

- **Model**: BGE-reranker-v2-m3 (~580MB)
- **lib**: `FlagEmbedding>=1.2`, `torch`

## 10. State 흐름

UserQueryState 의 `reranked` 필드.

## 11. Provenance + Confidence

- **Provenance**: response.metadata.reranker='bge-reranker-v2-m3-v1.1'
- **Confidence**: rerank_score top result (≥ 0.7 이면 strong, < 0.3 weak)

## 12. 테스트 시나리오

| Unit | 20 hits → rerank | top 10 + rerank_score 채워짐 |
| Unit | hits 빈 list | [] 그대로 |
| Edge | query 빈문자 | 동작 (모든 score 비슷) |

## 13. 모니터링

- KPI: latency ≤ 500ms (20 pairs)
- token: ₩0

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/rag/reranker.py`

### Changelog

- **v1 (2026-04-W3)** — stub
- **v2 (P6 활성)** — FlagReranker + batch + normalize
