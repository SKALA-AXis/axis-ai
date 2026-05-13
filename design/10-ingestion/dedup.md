# DedupAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `DedupAgent` |
| **Supervisor** | Ingestion |
| **LangGraph node** | `dedup` (#4) |
| **상태** | ✅ 구현 — `src/agents/dedup_agent.py` |
| **Trigger** | RelevanceAgent → `relevant_ids` 입력 시 |

## 2. 책임

**한 줄**: relevant raw_articles 를 BGE-M3 임베딩 후 코사인 유사도 ≥ 0.90 으로 클러스터링 + 대표 기사 선정.

**구체적**:

1. relevant_ids 각각의 (title + content[:500]) 을 BGE-M3 Dense 로 임베딩
2. 24시간 window 내 cosine ≥ 0.90 인 쌍 → 같은 cluster
3. cluster_id 부여 (BIGSERIAL 자체)
4. cluster 별 대표 기사 선정 — credibility_score DESC + published_at DESC
5. `raw_articles.cluster_id`, `raw_articles.is_representative` 업데이트
6. Gate 3 (중복 제거) 통과 — 대표 1건 만 다음 노드로

## 3. 책임 NOT

- 분류 (event/sector) — ClassificationAgent (다음 노드)
- 임베딩 자체는 BGE-M3 — 본 agent 의 핵심 의존성
- Qdrant 인덱싱 — `vector_index` 노드 (#8, EmbedIndexAgent)

## 4. 입력 스펙

```python
class DedupInput(TypedDict):
    relevant_ids: list[int]
```

## 5. 출력 스펙

```python
class DedupOutput(TypedDict):
    cluster_map: dict[int, list[int]]      # {cluster_id: [article_ids]}
    representative_ids: list[int]           # cluster 마다 1건
```

`raw_articles` 컬럼 갱신:

- `cluster_id: bigint` (FK to itself or generated)
- `is_representative: bool`
- `processing_status: 'CLUSTERED_REP' | 'CLUSTERED_DUPE'`

## 6. 알고리즘

```python
THRESHOLD = 0.90
WINDOW_HOURS = 24

def cluster_articles(article_ids):
    articles = fetch_articles(article_ids)
    embeddings = bge_m3.encode_dense([
        f"{a.title} {a.content[:500]}" for a in articles
    ])  # shape: (N, 1024)

    # 24h window grouping (chronological)
    clusters: dict[int, list[int]] = {}
    embeddings_in_cluster: dict[int, np.ndarray] = {}

    for i, article in enumerate(articles):
        emb = embeddings[i]
        # 기존 cluster 중 유사도 ≥ 0.90 인 첫 번째에 join
        matched_cluster = None
        for cluster_id, cluster_emb_mean in embeddings_in_cluster.items():
            similarity = cosine(emb, cluster_emb_mean)
            if similarity >= THRESHOLD:
                matched_cluster = cluster_id
                break
        if matched_cluster is None:
            matched_cluster = next_cluster_id()
            clusters[matched_cluster] = []
            embeddings_in_cluster[matched_cluster] = emb
        else:
            # cluster mean 업데이트 (running average)
            count = len(clusters[matched_cluster])
            embeddings_in_cluster[matched_cluster] = (
                embeddings_in_cluster[matched_cluster] * count + emb
            ) / (count + 1)
        clusters[matched_cluster].append(article.id)

    # 대표 기사 선정
    representatives = {}
    for cluster_id, ids in clusters.items():
        rep_id = max(ids, key=lambda aid:
            (articles_by_id[aid].credibility_score, articles_by_id[aid].published_at))
        representatives[cluster_id] = rep_id

    return clusters, representatives
```

### 클러스터 크기 효과

- 평균 클러스터 크기 3~5 (peer 마다 같은 이벤트가 여러 매체에 보도)
- 큰 클러스터 (≥ 7) — 핫 이벤트 (exposure_score 의 cluster_size 입력)

## 7. LLM 모델 + token 예산

- **LLM 미사용** — BGE-M3 임베딩만
- 토큰 예산: ₩0
- **모델 비용**: BGE-M3 = 로컬 (FlagEmbedding), GPU/CPU 추론 — 비용 ₩0 (전기료 무시)

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| BGE-M3 모델 로드 실패 | Pod startup 시 fail-fast. retry 안 함. (cache 가 mount 안 됐을 가능성) |
| 임베딩 shape mismatch | 본 article skip + `errors` append |
| 단일 row (no neighbors) | 자기 자신만 cluster (is_representative=true) |
| 24h window 외 동일 이벤트 | 별도 cluster (intentional — 시간 격차 큰 보도는 별개 신호) |

## 9. 외부 의존성

- **lib**: `FlagEmbedding>=1.2`, `numpy`, `torch` (BGE-M3 backend)
- **Model**: BGE-M3 (BAAI/bge-m3, MIT) — Pod 에 사전 캐시 필요 (`~/.cache/huggingface/`)
- **DB**: `raw_articles` (UPDATE cluster_id · is_representative · processing_status)

## 10. State 흐름 (LangGraph)

**소비**: `relevant_ids`
**생산**: `cluster_map`, `representative_ids`

## 11. Provenance + Confidence

- **Provenance**: `raw_articles.metadata.dedup_model_version='bge-m3-v1.5'`, `dedup_threshold=0.90`
- **Confidence**: 클러스터 평균 유사도 (모니터링용, downstream 미사용)

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | 동일 제목 5건 | 1 cluster size=5, rep=credibility 최고 |
| Unit | 완전 다른 제목 5건 | 5 cluster |
| Unit | 24h 외 동일 제목 | 2 cluster (시간 격차) |
| Edge | 빈 content | 임베딩 skip → 자기 cluster only |
| Integration | 100건 relevant → dedup | cluster 수 60~80 (70% 가정) |

## 13. 모니터링

- **pipeline_logs.step**: `dedup`
- **KPI**:
  - 평균 클러스터 크기 3~5
  - 대표 기사 선정 정확도 (수동 sampling) ≥ 90%
  - BGE-M3 임베딩 latency ≤ 50ms/article
- **token 예산**: 해당 없음

## 14. 구현 메모 + Changelog

### 의존 lib

```toml
FlagEmbedding = ">=1.2"
numpy = ">=1.26"
torch = ">=2.0"
```

### 핵심 파일

- `src/agents/dedup_agent.py`
- 임베딩 wrapper: `src/rag/embedder.py` (BGE-M3 공유 인스턴스)

### Changelog

- **v1 (2026-04-W1)** — 단순 cosine threshold
- **v2 (2026-04-W3)** — running average cluster centroid + 24h window
- **v3 (현재)** — mypy strict 후 `Mapping[str, tuple[str, ...]]` 으로 invariance 회피
