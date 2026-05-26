# PeerWordCloudBuilder — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `PeerWordCloudBuilder` |
| **Supervisor** | Enrichment |
| **상태** | 🟡 부분 (frontend mock data 있음, P6 우선) |
| **Trigger** | nightly 03:00 (peer 별 batch) |

## 2. 책임

**한 줄**: 각 peer 의 최근 90일 card_news 키워드 + event_type 묶음 → **카테고리 라벨링 (기술/사업/MOU)** 된 word cloud 생성.

**구체적**:

1. peer 별 최근 90일 card_news 필터링
2. KeywordExtraction 결과 재사용 (peer scope) → Top 50 keyword
3. LLM (gpt-4o-mini) 로 각 키워드 카테고리 라벨링 (기술 / 사업 / MOU / 기타) — 키워드 사전 hit 우선, miss 만 LLM
4. 크기 = frequency, 색상 = 카테고리
5. `enrichment_cache` per peer 에 저장 (TTL 24h)

## 3. 책임 NOT

- 키워드 추출 자체 — KeywordExtractionService 결과 재사용
- 그래프 노드/엣지 — KeywordGraphBuilder (별도)
- 전체 (peer 무관) 키워드 — KeywordExtractionService 의 전체 모드

## 4. 입력 스펙

```python
class PeerWordCloudInput(TypedDict):
    peer_id: str
    window_days: int   # 기본 90
```

## 5. 출력 스펙

```python
class PeerWordCloudOutput(TypedDict):
    peer_id: str
    words: list[dict]   # [{name, frequency, category, color, related_card_ids}]
    categories: list[dict]   # color legend
    cached_at: datetime
```

frontend (신규 spec) `GET /api/peers/{peerId}/wordcloud` 응답.

## 6. 알고리즘

### 6.1 카테고리 사전 (산식)

```python
CATEGORY_KEYWORDS = {
    "기술": ["AI", "AX", "클라우드", "보안", "데이터센터", "GPU", "RAG", "LLM", "kubernetes", ...],
    "사업": ["진출", "확장", "수주", "계약", "사업화", "출범", "런칭", ...],
    "MOU": ["MOU", "협약", "파트너십", "제휴", "협력", "공동개발", ...],
    "기타": []  # default
}

CATEGORY_COLORS = {
    "기술": "#3B82F6",   # blue
    "사업": "#10B981",   # green
    "MOU": "#F59E0B",    # amber
    "기타": "#6B7280",   # gray
}
```

### 6.2 LLM fallback (miss 시)

```python
PROMPT = """
다음 키워드들을 카테고리로 분류해라. 정확히 하나 선택:
기술 | 사업 | MOU | 기타

키워드 리스트:
- "에이전틱 AI"
- "GenSpark"

JSON 반환: {"keywords": [{"name": "에이전틱 AI", "category": "기술"}, ...]}
"""
```

### 6.3 build

```python
def build_wordcloud(peer_id, window_days=90):
    keywords = KeywordExtractionService().extract(window_days, peer_id=peer_id)
    top_50 = keywords[:50]

    # Stage 1: 키워드 사전 매칭
    unmatched = []
    for kw in top_50:
        for cat, kws in CATEGORY_KEYWORDS.items():
            if any(c.lower() in kw["name"].lower() for c in kws):
                kw["category"] = cat
                break
        else:
            unmatched.append(kw)

    # Stage 2: LLM fallback (10~20개 typical)
    if unmatched:
        llm_result = llm_classify_categories([kw["name"] for kw in unmatched])
        for kw in unmatched:
            kw["category"] = next((r["category"] for r in llm_result["keywords"]
                                  if r["name"] == kw["name"]), "기타")

    return {
        "peer_id": peer_id,
        "words": [{
            "name": kw["name"],
            "frequency": kw["frequency"],
            "category": kw["category"],
            "color": CATEGORY_COLORS[kw["category"]],
            "related_card_ids": fetch_related_cards(peer_id, kw["name"], limit=5),
        } for kw in top_50],
        "categories": [{"name": c, "color": CATEGORY_COLORS[c]} for c in CATEGORY_KEYWORDS],
        "cached_at": datetime.utcnow(),
    }
```

## 7. LLM 모델 + token 예산

- **모델**: gpt-4o-mini (카테고리 라벨링 만, miss 케이스 fallback)
- 토큰/호출: ~500 (20 키워드 batch)
- 일일 호출: ~4 (peer 4사 × 1회 nightly)
- 일일 비용: **~₩20**

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| LLM JSON parse 실패 | unmatched 키워드 모두 "기타" |
| peer 의 card 0건 | empty words[] |
| 카테고리 사전 hit 100% | LLM 호출 0회 (best case) |

## 9. 외부 의존성

- **DB**: `card_news` (READ), `enrichment_cache` (UPSERT)
- **외부 API**: OpenAI gpt-4o-mini
- **lib**: 없음
- **sub**: KeywordExtractionService 재사용

## 10. State 흐름

EnrichmentState 의 `peer_wordclouds: dict[str, list[dict]]` 필드 채움.

## 11. Provenance + Confidence

- **Provenance**: `enrichment_cache.metadata.wordcloud_version, llm_model='gpt-4o-mini', prompt_version='v1'`
- **Confidence**: 사전 hit = 1.0, LLM = 0.7 (zero-shot 보수적)

## 12. 테스트 시나리오

| Unit | "AX" 키워드 | category="기술" (사전 hit) |
| Unit | "에이전틱" 키워드 (사전 miss) | LLM 호출 + "기술" 분류 |
| Edge | peer 의 card 0건 | words=[] |
| Edge | LLM 응답 invalid | unmatched 전부 "기타" |

## 13. 모니터링

- KPI: 사전 hit 비율 ≥ 70%
- token 예산: ₩20/일

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/peer_word_cloud_agent.py` (신규 P6)
- 카테고리 사전: `src/config/wordcloud_categories.py`

### Changelog

- **v1 (제안, P6)** — 사전 매칭 + LLM fallback
