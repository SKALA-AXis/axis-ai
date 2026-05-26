# KeywordGraphBuilder — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `KeywordGraphBuilder` |
| **Supervisor** | Enrichment |
| **상태** | 🟡 부분 (frontend mock data 있음, P6 우선) |
| **Trigger** | nightly 02:30 (KeywordExtraction 이후) |

## 2. 책임

**한 줄**: KeywordExtraction 결과 + card_news 의 co-occurrence 로 frontend 2D/3D 키워드 네트워크 그래프 (nodes + edges + 중요도) 생성.

**구체적**:

1. KeywordExtraction 의 Top 100 키워드 → 노드 후보
2. 같은 card_news 에 동시 등장하는 키워드 쌍 → 엣지
3. PMI (Pointwise Mutual Information) 로 엣지 가중치 계산
4. 노드 size = TF-IDF score, 색상 = peer / sector 분포
5. (옵션) Force-directed pre-layout coordinate 사전 계산 (frontend 부담 ↓)
6. `enrichment_cache` 에 저장 (TTL 24h)

## 3. 책임 NOT

- frontend 의 실 force simulation 렌더링 — Three.js 가 담당 (본 builder 는 데이터만)
- 사용자 클릭 → 관련 카드 조회 — backend `/api/keyword-graph/{nodeId}/cards` 가 별도 처리

## 4. 입력 스펙

```python
class KeywordGraphInput(TypedDict):
    companies: list[str] | None    # peer ids 필터 (None = 전체)
    sectors: list[str] | None
    keyword_query: str | None      # frontend 검색박스에서 입력 시 highlight 용
```

## 5. 출력 스펙

```python
class GraphNode(TypedDict):
    id: str                # keyword
    label: str
    size: float            # TF-IDF 기반
    category: str          # primary peer or sector
    color: str             # hex
    peer_distribution: dict
    sector_distribution: dict
    card_count: int

class GraphEdge(TypedDict):
    source: str            # keyword id
    target: str
    weight: float          # PMI score
    co_occurrence_count: int

class KeywordGraphOutput(TypedDict):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    categories: list[dict]   # color legend
    cached_at: datetime
```

frontend `GET /api/keyword-graph` 응답.

## 6. 알고리즘

### 6.1 PMI 산출 (산식)

```python
def pmi(token_a, token_b, documents):
    """PMI = log(P(a,b) / (P(a) * P(b)))"""
    N = len(documents)
    count_a = sum(1 for doc in documents if token_a in doc)
    count_b = sum(1 for doc in documents if token_b in doc)
    count_ab = sum(1 for doc in documents if token_a in doc and token_b in doc)
    if count_ab == 0:
        return 0.0
    p_a = count_a / N
    p_b = count_b / N
    p_ab = count_ab / N
    return math.log(p_ab / (p_a * p_b))
```

### 6.2 Graph build

```python
def build_graph(keywords_top_100, documents):
    nodes = [{
        "id": kw["name"],
        "label": kw["name"],
        "size": kw["tf_idf_score"],
        "category": determine_category(kw),  # primary peer or sector
        "color": CATEGORY_COLORS[determine_category(kw)],
        "peer_distribution": kw["peer_distribution"],
        "sector_distribution": kw["sector_distribution"],
        "card_count": kw["frequency"],
    } for kw in keywords_top_100]

    edges = []
    for i, kw_a in enumerate(keywords_top_100):
        for kw_b in keywords_top_100[i+1:]:
            pmi_score = pmi(kw_a["name"], kw_b["name"], documents)
            if pmi_score > 0.5:  # threshold
                co_occ = sum(1 for doc in documents
                            if kw_a["name"] in doc and kw_b["name"] in doc)
                edges.append({
                    "source": kw_a["name"],
                    "target": kw_b["name"],
                    "weight": pmi_score,
                    "co_occurrence_count": co_occ,
                })

    return nodes, edges
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — 산식 only (PMI + 컬러 매핑)
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| keyword 0건 (Ingestion fail) | empty nodes/edges |
| edge 수 > 5000 (그래프 너무 dense) | top edges weight 기준 5000 만 유지 |
| 같은 keyword 가 여러 카테고리 | primary 1개 선택 (가장 빈도 높은 peer/sector) |

## 9. 외부 의존성

- **DB**: KeywordExtraction 결과 + `card_news` (co-occurrence 계산)
- **lib**: 없음 (산식)
- **cache**: `enrichment_cache` (TTL 24h)

## 10. State 흐름

EnrichmentState 의 `keyword_graph: dict` 필드 채움.

## 11. Provenance + Confidence

- **Provenance**: `enrichment_cache.metadata.graph_version`, `pmi_threshold`
- **Confidence**: edge.weight (PMI) 가 confidence proxy

## 12. 테스트 시나리오

| Unit | 100 documents, "AX" + "생성형" co-occur 80% | edge "AX"-"생성형" 존재, PMI ≥ 1.0 |
| Unit | 단일 keyword (edge 없음) | edges=[] |
| Edge | PMI > 5000 edge | top 5000 만 |

## 13. 모니터링

- KPI: 그래프 생성 latency ≤ 60초 nightly
- 노드 수 평균 80~100, 엣지 1000~3000

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/services/keyword_graph_builder.py` (신규 P6)
- frontend 의 mock `graphNodes` / `graphEdges` 대체 대상

### Changelog

- **v1 (제안, P6)** — PMI + TF-IDF + 카테고리 매핑
