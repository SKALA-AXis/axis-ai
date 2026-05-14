# RelevanceAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `RelevanceAgent` |
| **Supervisor** | Ingestion |
| **LangGraph node** | `preprocess_route` (#3) |
| **상태** | ✅ 구현 — `src/agents/relevance_agent.py` |
| **Trigger** | CredibilityAgent 통과 row 마다 |

## 2. 책임

**한 줄**: credible 한 raw_article 이 peer 모니터링과 관련 있는지 zero-shot LLM 분류 + 키워드 매칭 으로 라우팅.

**구체적**:

1. **키워드 매칭** (산식, 1차 필터) — peer 별 키워드 사전 + sector 키워드 사전 hit 검사
2. **LLM 분류** (gpt-4o-mini, hit 못한 경우 fallback) — relevant / irrelevant / edge 3-class
3. **routing 결과 마킹** — `relevance_label`, `relevance_score`, `matched_companies`, `matched_sectors`
4. **다음 노드 입력 분리**:
   - news/RSS → DedupAgent (기본 path)
   - DART 공시 → official_document_ids
   - IR PDF → parsed_document_ids
   - 산업 보고서 → industry_document_ids
   - 채용공고 → structured_signal_ids (WeakSignal input)

## 3. 책임 NOT

- 클러스터링 — DedupAgent
- 분류 (event/sector) — ClassificationAgent
- 신뢰도 — CredibilityAgent (이전 노드)

## 4. 입력 스펙

```python
class RelevanceInput(TypedDict):
    credible_ids: list[int]
```

내부적으로 `raw_articles.content`, `title`, `company`, `source_type` 조회.

## 5. 출력 스펙

```python
class RelevanceOutput(TypedDict):
    relevant_ids: list[int]                  # news 흐름 (DedupAgent 로)
    official_document_ids: list[int]         # DART
    parsed_document_ids: list[int]           # IR PDF
    industry_document_ids: list[int]         # BCG / McKinsey
    structured_signal_ids: list[int]         # 채용공고 (WeakSignal)
    skipped_preprocess_ids: list[int]        # 'irrelevant'
```

`raw_articles` 컬럼 갱신:

- `relevance_score: float` (0.0~1.0)
- `relevance_label: str` (`'relevant' | 'edge' | 'irrelevant'`)
- `relevance_reason: text` (LLM 응답의 explanation)
- `matched_companies: jsonb` (peer ids 배열)
- `matched_sectors: jsonb` (sector ids 배열)
- `processing_status: 'CLASSIFIED_PRE'` (다음 dedup 으로)

## 6. 알고리즘

### Stage 1: 키워드 매칭 (산식, fast path)

```python
PEER_KEYWORDS = {
    "samsung_sds": ["삼성SDS", "삼성에스디에스", "Samsung SDS"],
    "lg_cns": ["LG CNS", "엘지CNS", "LG씨엔에스"],
    "hyundai_autoever": ["현대오토에버", "Hyundai AutoEver"],
    "posco_dx": ["포스코DX", "포스코디엑스", "POSCO DX"],
    "sk_ax": ["SK AX", "SK에이엑스", "SK주식회사 C&C"],  # 자사
}

SECTOR_KEYWORDS = {
    "ax": ["AX", "AI 전환", "디지털 전환", "생성형 AI", "agentic"],
    "security": ["보안", "사이버", "정보보호", "zero trust", "XDR"],
    "infra": ["인프라", "클라우드", "데이터센터", "GPU", "kubernetes"],
    "deal": ["수주", "MOU", "협약", "인수합병", "투자"],
}

def keyword_match(article) -> tuple[list[str], list[str]]:
    text = (article.title + " " + article.content[:2000]).lower()
    peers = [pid for pid, kws in PEER_KEYWORDS.items() if any(kw.lower() in text for kw in kws)]
    sectors = [sid for sid, kws in SECTOR_KEYWORDS.items() if any(kw.lower() in text for kw in kws)]
    return peers, sectors
```

### Stage 2: LLM 분류 (gpt-4o-mini, hit miss 시 또는 edge case)

```python
PROMPT = """다음 기사가 SK AX 사업전략팀이 모니터링해야 하는 peer (삼성SDS / LG CNS / 현대오토에버 / 포스코DX) 의 전략 동향과 관련 있는지 판단하라.

기사 제목: {title}
기사 본문 (200자): {content_preview}

다음 JSON 만 반환:
{{
  "label": "relevant" | "edge" | "irrelevant",
  "score": 0.0~1.0,
  "matched_companies": ["peer_id"],
  "matched_sectors": ["sector_id"],
  "reason": "한 줄 사유"
}}"""

# 보호: 정확한 LLM 비용 추적 + token budget 차단
```

### Stage 3: Routing

```python
def route(article, label, score, peers, sectors):
    if label == "irrelevant" or score < 0.3:
        return "skipped_preprocess_ids"
    if article.source_type == "dart":
        return "official_document_ids"
    if article.source_type == "ir" or article.source_type == "securities_report":
        return "parsed_document_ids"
    if article.source_type in ("trend_report", "search_trend"):
        return "industry_document_ids"
    if article.source_type == "job":
        return "structured_signal_ids"
    return "relevant_ids"   # 기본 (news)
```

### Stage 4: Prompt audit — 02-prompt-design-checklist 17 요소

Relevance 는 fallback LLM 만 사용 (키워드 hit 시 LLM skip). 필수 1, 2, 4, 7, 14.

| # | 요소 | 충족 위치 | 비고 |
|---|---|---|---|
| **1** | 역할 정의 | LLM prompt 도입부 ← 보강 필요 | "당신은 SK AX 사업전략팀의 모니터링 관련성 판정관. 4 peer × 5 sector 기준 외 기사 = irrelevant." 추가 |
| **2** | 추적 대상 기업 | peer 키워드 사전 + companies enum 4 peer | OK |
| **3** | 추적 범위 | sector 키워드 사전 5종 | OK |
| **4** | 출처 우선순위 | (Credibility 가 이미 통과한 row 만 input) | 이전 노드에서 보장 |
| **7** | 단순 뉴스 요약 금지 | 분류는 enum 3-class (relevant/edge/irrelevant) | 자유형 분류 X |
| **14** | 출력 형식 | RelevanceOutput TypedDict + relevance_label enum | 필수 |

→ **6/6 필수 충족** (1 보강 후). Edge case → ClassificationAgent 가 receive 후 정밀 분류.

## 7. LLM 모델 + token 예산

- **모델**: gpt-4o-mini (zero-shot classification 충분)
- 토큰/호출: ~500 (prompt 300 + 본문 preview 200 / output 50)
- **일일 호출수**: ~500 (crawl 전체 중 키워드 매칭 miss 인 비율 가정)
- **일일 비용**: ~₩100

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| LLM 응답 JSON parse 실패 | retry 1회 → 실패 시 `label='edge'` + `score=0.5` (보수적) |
| LLM 타임아웃 (3초) | 키워드 매칭 결과만 사용 |
| 키워드 매칭도 miss | `label='irrelevant'` (skip) |
| matched_companies 가 sk_ax 자사 only | `relevant_ids` 에서 제외 (자사 모니터링 안 함, raw 아카이브만) |

## 9. 외부 의존성

- **DB**: `raw_articles` (UPDATE relevance_*) · `pipeline_logs`
- **외부 API**: OpenAI gpt-4o-mini
- **Config**: `src/config/companies.py` · `src/agents/sector_keywords.py`

## 10. State 흐름 (LangGraph)

**소비**: `credible_ids`
**생산**: `relevant_ids` + 4개 별도 routing list + `skipped_preprocess_ids`

## 11. Provenance + Confidence

- **Provenance**: `raw_articles.metadata.relevance_llm_model`, `relevance_prompt_version`
- **Confidence**: `relevance_score` 자체. ClassificationAgent 의 exposure_score 계산에 입력으로 사용

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | "삼성SDS AX 플랫폼 발표" | matched_companies=['samsung_sds'], matched_sectors=['ax'], relevant |
| Unit | "오늘의 날씨" | irrelevant, score < 0.3 |
| Unit | source_type='dart' + relevant | official_document_ids 로 라우팅 |
| Unit | source_type='job' + 키워드 hit | structured_signal_ids 로 라우팅 (WeakSignal input) |
| Edge | matched_companies=['sk_ax'] only | skipped (자사 제외) |
| Edge | LLM 타임아웃 | 키워드 매칭만으로 결정 |

## 13. 모니터링

- **pipeline_logs.step**: `preprocess_route`
- **KPI**:
  - relevant 비율 30~50% (credible 중)
  - LLM fallback 호출 비율 ≤ 20% (대부분 키워드 매칭으로 해결)
  - irrelevant false negative (sampling 검증) ≤ 5%
- **token 예산**: ₩100/일

## 14. 구현 메모 + Changelog

### 의존 lib

```toml
openai = ">=1.30"   # gpt-4o-mini
```

### 핵심 파일

- `src/agents/relevance_agent.py`
- 키워드 사전: `src/config/companies.py`, `src/agents/sector_keywords.py`

### Changelog

- **v1 (2026-04-W2)** — 키워드 매칭만
- **v2 (2026-04-W3)** — LLM zero-shot fallback 추가
- **v3 (현재)** — source_type 별 routing 분기
