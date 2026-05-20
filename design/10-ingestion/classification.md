# ClassificationAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `ClassificationAgent` |
| **Supervisor** | Ingestion |
| **LangGraph node** | `classify` (#5) |
| **상태** | ✅ 구현 — `src/agents/classification_agent.py` |
| **Trigger** | DedupAgent → cluster 마다 1회 (대표 기사 기준) |

## 2. 책임

**한 줄**: cluster 의 대표 기사를 `event_type` (6종) + `sector` (5종) 로 분류하고 **결정적 산식** 으로 `exposure_score` (0~1) 계산.

**구체적**:

1. **결정적 산식 우선** — exposure_score = 0.40·cluster_size_norm + 0.30·credibility_max + 0.20·peer_mention_rate + 0.10·tier1_diversity (LLM 없음)
2. **event_type 분류** — 키워드 매칭 → LLM gpt-4o-mini fallback (6종 중 1)
3. **sector 분류** — 키워드 매칭 → LLM fallback (5종 중 1)
4. **exposure_band** 매핑 — high (≥0.65) / medium (0.40~0.65) / low (<0.40)
5. `update_classification()` 호출하여 raw_articles.importance_level/score 갱신

## 3. 책임 NOT

- card 생성 (title/summary) — CardNewsAgent (다음 노드)
- evidence 부착 — EvidenceAgent
- 자유형 importance 분류 (v1 의 urgent/notable/reference) — 폐기 (v3 부터 exposure_band 단일)

## 4. 입력 스펙

```python
class ClassificationInput(TypedDict):
    cluster_map: dict[int, list[int]]
    representative_ids: list[int]
```

내부 fetch:
- `raw_articles` 의 대표 기사 (title + content[:1500])
- `raw_articles.credibility_score` (cluster 내 max 계산용)

## 5. 출력 스펙

```python
class ClassifiedCluster(TypedDict):
    cluster_id: int
    representative_id: int
    company: list[str]                 # raw_articles.company (jsonb)
    event_type: Literal["partnership","ma","personnel","tech","regulation","new_biz"]
    sector: Literal["ax","security","infra","deal","other"]
    sectors: list[str]                  # 멀티 sector 가능
    exposure_score: float               # 0~1
    exposure_band: Literal["high","medium","low"]
    signals: dict                       # {cluster_size, credibility_max, peer_mention_rate, tier1_diversity}
    importance_score: float             # exposure_score 와 동일 (v1 호환)
    importance: Literal["urgent","notable","reference"]  # deprecated, exposure_band 대체

class ClassificationOutput(TypedDict):
    classified_clusters: list[ClassifiedCluster]
```

`raw_articles` 컬럼 갱신: `importance_level`, `importance_score`, `processing_status='CLASSIFIED'`, `qdrant_vector_id` (Qdrant 후속 인덱싱 시 채워짐).

## 6. 알고리즘

### 6.1 exposure_score (결정적 산식, ADR-0011)

```python
def exposure_score(cluster, articles_in_cluster):
    cluster_size = len(articles_in_cluster)
    cluster_size_norm = min(cluster_size / WEEKLY_MAX_CLUSTER_SIZE, 1.0)

    credibility_max = max(a.credibility_score for a in articles_in_cluster)

    peer_mentions = sum(1 for a in articles_in_cluster
                        if any(peer_id in a.matched_companies for peer_id in MONITORED_PEERS))
    peer_mention_rate = peer_mentions / cluster_size

    tier1_sources = {a.source_name for a in articles_in_cluster
                     if SOURCE_BASE_SCORE.get(a.source_name, 0) >= 0.80}
    tier1_diversity = min(len(tier1_sources) / 5, 1.0)

    return round(
        0.40 * cluster_size_norm +
        0.30 * credibility_max +
        0.20 * peer_mention_rate +
        0.10 * tier1_diversity,
        4
    )
```

### 6.2 event_type / sector 분류

```python
EVENT_KEYWORDS = {
    "partnership": ["MOU", "협약", "파트너십", "제휴"],
    "ma": ["인수", "합병", "M&A", "지분 인수", "IPO"],
    "personnel": ["인사", "임원", "사장", "조직개편"],
    "tech": ["발표", "출시", "공개", "기술", "플랫폼"],
    "regulation": ["규제", "법안", "정책", "공정위", "방통위"],
    "new_biz": ["진출", "신사업", "사업 확장", "출범"],
}

def classify_event(article):
    text = (article.title + " " + article.content[:1000]).lower()
    for evt, kws in EVENT_KEYWORDS.items():
        if any(kw.lower() in text for kw in kws):
            return evt
    # fallback to LLM
    return llm_classify_event(article)
```

LLM fallback prompt:

~~~text
# SK AX 사업전략팀 분류 전문가

당신은 SK AX 사업전략팀의 분류 전문가입니다.
**키워드 매칭 실패 기사** 1건을 **6 event_type enum** 중 정확히 하나로 분류합니다.

## 입력 데이터
- **제목**: {title}
- **본문 (preview)**: {content_preview}

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **enum 만**: 출력은 정확히 `partnership` / `ma` / `personnel` / `tech` / `regulation` / `new_biz` 6개 중 하나
- **자유형 분류 금지**: 새 카테고리 발명 X

### 일반 규칙 (17 요소 매핑)
1. **(#3 추적 범위)** 6 event_type enum 외 출력 시 응답 무효
2. **(#7 단순 요약 금지)** 분류만 출력, 본문 요약 X
3. **(#14 출력 형식)** strict JSON, 단일 필드

## 출력 형식 (strict JSON)

```json
{"event_type": "partnership"}
```
~~~

### 6.3 sector 와 sectors[]

- `sector` — primary (1개)
- `sectors[]` — 가능 후보 (멀티)
- Frontend 가 `sector` 1개 표시, BriefingService 가 sector-grouped 시 primary 우선

### 6.4 Prompt audit — 02-prompt-design-checklist 17 요소

Classification 은 fallback LLM 만 사용 (키워드 hit 시 LLM skip). 필수 1/2/3/4/6/7/14, 권장 9.

| # | 요소 | 충족 위치 | 비고 |
|---|---|---|---|
| **1** | 역할 정의 | LLM prompt 도입부 ← 보강 필요 | 현재: "이 기사가 다음 6 event…" → "당신은 SK AX 사업전략팀의 분류 전문가. 6 event_type 중 정확히 하나만 반환." 로 추가 |
| **2** | 추적 대상 기업 | input `company` field + companies enum (`src/config/companies.py`) | 4 peer + sk_ax_self enum 사용 |
| **3** | 추적 범위 | 6 event × 5 sector enum (Literal) | TypedDict 강제 |
| **4** | 출처 우선순위 | tier1_diversity 산식 (가중치 0.10) + credibility_max (0.30) | 분류 단계에서 출처 가중 적용 |
| **5** | 분석 기간 | (해당 없음 — 단일 기사) | — |
| **6** | 최신성 검증 | raw_articles.published_at_kst 사용 (분류 출력에 carry 안 함 ← cluster id 로 carry) | OK |
| **7** | 단순 뉴스 요약 금지 | event_type taxonomy 강제 (자유형 분류 불가) | 6 enum |
| **9** | 변화 감지 기준 | exposure_band high/medium/low (0.65/0.40 cutoff) | 산식 |
| **12** | 공식 vs 추정 구분 | provenance.classification_version='v3.0' + signals dict | 산식 입력값 모두 보존 |
| **14** | 출력 형식 | ClassifiedCluster TypedDict | 필수 충족 |

→ **9/9 필수 충족**. PDF 반영 후 보강: prompt 도입부에 역할 정의 1문장 추가.

## 7. LLM 모델 + token 예산

- **모델**: gpt-4o-mini (fallback only — 키워드 매칭 hit 시 skip)
- **호출수**: ~30 (cluster 중 키워드 miss 비율 약 40%)
- **토큰/호출**: ~1,500 (in 1,200 + out 300)
- **일일 비용**: ~₩300

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| LLM JSON parse 실패 | event_type='tech', sector='other' fallback |
| LLM 타임아웃 (3초) | 키워드 매칭 결과만 사용 |
| 키워드 매칭도 miss | event_type='tech' (가장 흔함), sector='other' |
| credibility_score null | 0.5 fallback (산식 보존) |
| peer_mention_rate = 0 | sk_ax 자사 mention 만 — exposure_score 가중치 ↓ |

## 9. 외부 의존성

- **DB**: `raw_articles` (READ + UPDATE)
- **외부 API**: OpenAI gpt-4o-mini (fallback)
- **Config**: `src/agents/sector_keywords.py`, `src/config/companies.py`

## 10. State 흐름 (LangGraph)

**소비**: `cluster_map`, `representative_ids`, `credible_ids` (cluster_size 산정 용)
**생산**: `classified_clusters: list[ClassifiedCluster]`

## 11. Provenance + Confidence

- **Provenance**: `card_news.implication.evidence_chain.provenance.classification_version='v3.0'`
- **Confidence**: 분류 자체에는 confidence 없음 — exposure_score 가 ranking confidence proxy

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | cluster_size=10, credibility_max=0.95, peer_mention=100% | exposure_score ≥ 0.75 (high band) |
| Unit | "삼성SDS LG CNS MOU 체결" | event_type='partnership', sectors=['deal','ax'] |
| Unit | 키워드 매칭 hit | LLM 호출 X |
| Unit | 키워드 miss | LLM 호출 + JSON parse |
| Edge | LLM 응답 invalid JSON | fallback 적용 |

## 13. 모니터링

- **pipeline_logs.step**: `classify`
- **KPI**:
  - LLM fallback 호출 비율 ≤ 40%
  - exposure_band 분포: high 15% / medium 50% / low 35% 정도
  - event_type 분포 (sampling 검증) — 'tech' 50% 이하 (너무 많으면 키워드 사전 보강)
- **token 예산**: ₩300/일

## 14. 구현 메모 + Changelog

### 의존 lib

```toml
openai = ">=1.30"
```

### 핵심 파일

- `src/agents/classification_agent.py`
- 키워드 사전: `src/agents/sector_keywords.py`
- `update_classification()` — `src/db/article_store.py` (signature: article_id, importance, importance_score, qdrant_vector_id)

### Changelog

- **v1 (2026-04-W1)** — LLM 5-축 (긴급/주목/참고) 분류
- **v2 (2026-04-W2)** — sector + event_type 도입
- **v3 (2026-04-W3)** — **결정적 산식 (4-component)** 으로 exposure_score 전환. LLM 분류 (importance) 폐기, exposure_band 단일
- **v3.1 (현재)** — sectors[] 멀티 후보 + sk_ax 자사 분기
