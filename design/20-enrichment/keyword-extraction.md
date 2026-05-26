# KeywordExtractionService — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `KeywordExtractionService` |
| **Supervisor** | Enrichment |
| **상태** | 🟡 부분 (산식 구현 가능, P6 우선) |
| **Trigger** | Ingestion 후 매시 + nightly 02:00 batch |

## 2. 책임

**한 줄**: 최근 30일 `card_news` + `raw_articles.title/summary_lines` 에서 KR-TF-IDF 산식으로 키워드 빈도 + 중요도 산출.

**구체적**:

1. 한글 형태소 분석 (간이 — 명사 추출 only) — kiwipiepy 또는 단순 regex
2. 도메인 stopword 제거 (예: "기업", "회사", "발표")
3. TF-IDF — 문서 단위 = card_news.id, 30일 window
4. peer × sector × keyword × count 분포 산출
5. Top-K 키워드 (전체 / peer 별 / sector 별)
6. `enrichment_cache` 테이블에 저장 (TTL 6h)

## 3. 책임 NOT

- 키워드 카테고리 라벨링 (기술/사업/MOU) — PeerWordCloudBuilder (LLM)
- 그래프 노드/엣지 구성 — KeywordGraphBuilder
- 사용자 입력 prefix 매칭 — SearchSuggestService

## 4. 입력 스펙

```python
class KeywordExtractionInput(TypedDict):
    window_days: int        # 기본 30
    peer_id: str | None     # None 이면 전체
    sector: str | None      # None 이면 전체
```

## 5. 출력 스펙

```python
class KeywordExtractionOutput(TypedDict):
    keywords: list[dict]    # [{name, frequency, tf_idf_score, peer_distribution, sector_distribution}]
    total_documents: int
    window_days: int
    cached_at: datetime
```

frontend `GET /api/dashboard/summary` 의 `keywordSeries` 필드 채움.

## 6. 알고리즘

```python
from kiwipiepy import Kiwi   # 한글 형태소
import numpy as np

STOPWORDS = {"기업", "회사", "발표", "관련", "예정", "있다", "통해", ...}

class KeywordExtractionService:
    def __init__(self):
        self.kiwi = Kiwi()

    def extract(self, window_days=30, peer_id=None, sector=None):
        cards = fetch_card_news(window_days, peer_id, sector)
        documents = []
        for card in cards:
            text = card.title + " " + " ".join(card.summary_lines)
            tokens = [t.form for t in self.kiwi.tokenize(text)
                      if t.tag.startswith("NN") and t.form not in STOPWORDS and len(t.form) >= 2]
            documents.append(tokens)

        # TF-IDF
        vocab = set(token for doc in documents for token in doc)
        N = len(documents)
        idf = {token: math.log(N / sum(1 for doc in documents if token in doc))
               for token in vocab}

        keywords = []
        for token in vocab:
            tf = sum(doc.count(token) for doc in documents)
            tf_idf = tf * idf[token]
            keywords.append({"name": token, "frequency": tf, "tf_idf_score": tf_idf})

        # Top 200
        return sorted(keywords, key=lambda k: k["tf_idf_score"], reverse=True)[:200]
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — 산식 only
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| card_news 0건 | empty list 리턴 |
| Kiwi 모델 로드 실패 | regex fallback (간이 명사 추출) |
| stopword list 미정 | 기본 list 만 — admin 페이지에서 추가 가능 (W6+) |

## 9. 외부 의존성

- **DB**: `card_news`, `raw_articles`, `enrichment_cache` (UPSERT)
- **lib**: `kiwipiepy>=0.18`
- **외부 API**: 없음

## 10. State 흐름

EnrichmentSupervisor 의 LangGraph state (간소 — 1 노드 그래프):

```python
class EnrichmentState(TypedDict):
    window_days: int
    keywords: list[dict]
    keyword_graph: dict
    peer_wordclouds: dict[str, list[dict]]
    top_insights: list[str]
    trend_metrics: dict
```

## 11. Provenance + Confidence

- **Provenance**: `enrichment_cache.metadata.extraction_version='v1', tokenizer='kiwipiepy-0.18'`
- **Confidence**: TF-IDF score 가 ranking confidence proxy

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | card 30건, "AX" 50회 등장 | keywords[0].name='AX', frequency=50 |
| Unit | stopword "기업" 포함 본문 | extracted 키워드에 "기업" 없음 |
| Edge | card 0건 | keywords=[] |
| Edge | 영문 only card | extracted 키워드 영문 명사 만 |

## 13. 모니터링

- **KPI**: 추출 latency ≤ 30초 (30일 window, ~1000 cards 기준)
- **token 예산**: ₩0

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/services/keyword_extraction_service.py` (신규 P6)
- 의존 lib: `kiwipiepy>=0.18`

### Changelog

- **v1 (제안, P6)** — KR-TF-IDF + Kiwi 형태소
- **v2 (제안)** — N-gram (2-gram) 지원 — "생성형 AI" 같은 복합어 보존
