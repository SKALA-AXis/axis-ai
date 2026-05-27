# SearchSuggestService — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `SearchSuggestService` |
| **Supervisor** | Enrichment (on-demand, cache + 산식) |
| **상태** | 🟡 backend fixture 존재 (`GET /api/search/suggestions`), axis-ai 신규 |
| **Trigger** | User 검색박스 입력 (debounced 300ms) |

## 2. 책임

**한 줄**: 사용자 검색 prefix + 인기 키워드 + 임베딩 유사도 hybrid 로 top-K 자동완성 추천.

**구체적**:

1. trigram 기반 prefix 매칭 (산식) — 최근 30일 키워드 + peer 이름 + sector 이름 + 카드 제목
2. popularity boost — KeywordExtraction 의 frequency score 가중
3. (옵션) embed-based — 입력 prefix → BGE-M3 임베딩 → Qdrant 유사도 검색 (3+ 글자 이상 시)
4. 결과 5~10개 (UI 카드뉴스 / Peer / Keyword 3 카테고리 분기)

## 3. 책임 NOT

- 실제 검색 결과 — HybridSearchService (UserQuery supervisor)
- 검색 history 학습 — 개인화 (out of scope v1)

## 4. 입력 스펙

```python
class SearchSuggestInput(TypedDict):
    prefix: str
    max_results: int        # 기본 8
    include_embed: bool     # 3+ 글자 이상 시 True
```

## 5. 출력 스펙

```python
class SuggestItem(TypedDict):
    text: str
    category: Literal["keyword","peer","card_title","sector"]
    score: float
    related_count: int      # 매칭 카드 수

class SearchSuggestOutput(TypedDict):
    items: list[SuggestItem]
    prefix: str
```

frontend `GET /api/search/suggestions?q=...` 응답.

## 6. 알고리즘

```python
def suggest(prefix: str, max_results=8, include_embed=False):
    items = []

    # Stage 1: trigram prefix 매칭 (산식)
    keyword_matches = trigram_search(KEYWORDS_TABLE, prefix, top=10)
    peer_matches = [p for p in PEERS if p.startswith(prefix.lower())]
    card_title_matches = trigram_search(CARD_TITLES, prefix, top=5)
    sector_matches = [s for s in SECTORS if s.startswith(prefix.lower())]

    # Stage 2: popularity boost (KeywordExtraction frequency)
    for km in keyword_matches:
        items.append({
            "text": km.name,
            "category": "keyword",
            "score": km.frequency * 0.7,  # popularity boost
            "related_count": km.frequency,
        })

    # Stage 3 (옵션): embed-based (3+ 글자)
    if include_embed and len(prefix) >= 3:
        embedded = BGE_M3.encode(prefix)
        qdrant_results = qdrant.search(collection="axis_main", vector=embedded.dense, limit=5)
        for r in qdrant_results:
            items.append({
                "text": r.payload["title"][:80],
                "category": "card_title",
                "score": r.score * 100,
                "related_count": 1,
            })

    # Top-K 정렬
    items = sorted(items, key=lambda x: x["score"], reverse=True)[:max_results]
    return {"items": items, "prefix": prefix}
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — trigram + BGE-M3 + Qdrant
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| prefix 빈문자 | 인기 키워드 top 8 리턴 |
| prefix < 2 글자 | trigram only (embed skip) |
| Qdrant connection error | trigram only fallback |

## 9. 외부 의존성

- **DB**: PostgreSQL trigram index 권장 (`pg_trgm` extension) on `card_news.title`, KeywordExtraction cache
- **lib**: 없음
- **Qdrant**: optional embedding search

## 10. State 흐름

stateless. on-demand 호출.

## 11. Provenance + Confidence

- **Provenance**: 없음 (인터랙티브 UI)
- **Confidence**: items[*].score 가 ranking

## 12. 테스트 시나리오

| Unit | prefix="삼" | "삼성SDS" peer match 우선 |
| Unit | prefix="AX" | keyword "AX" top |
| Edge | prefix="" | 인기 top 8 |
| Edge | prefix > 100 chars | truncate to 50 |

## 13. 모니터링

- KPI: latency ≤ 50ms (UX 부담)
- token 예산: ₩0

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/search_suggest_agent.py` (신규 P6)
- PostgreSQL `pg_trgm` extension 활성화 V11 migration 필요

### Changelog

- **v1 (제안, P6)** — trigram + popularity boost
- **v2 (제안)** — 개인화 (사용자 검색 history 학습) — out of v1 scope
