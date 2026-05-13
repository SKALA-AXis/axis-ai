# InsightCascadeAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `InsightCascadeAgent` |
| **Supervisor** | Analysis |
| **상태** | 🟡 backend fixture (`POST /api/insights/generate`, `GET /api/insights/latest`), axis-ai 신규 |
| **Trigger** | User POST request, on-demand |

## 2. 책임

**한 줄**: Top 6 card_news ids 를 입력받아 **4단계 Cause → Change → Impact → Response** chain-of-thought 인사이트 자동 작성 + confidence 산출.

**구체적**:

1. 6 카드 → context 구성 (title + summary + sources + sector)
2. **Cause** — "왜 이런 변화가 일어났는가?" (3~5 bullet)
3. **Change** — "구체적으로 어떤 변화인가?" (3~5 bullet)
4. **Impact** — "SK AX 에 어떤 영향?" (3~5 bullet)
5. **Response** — "SK AX 의 대응 방안?" (3~5 bullet)
6. confidence — 카드 trust_score 평균 + LLM 자체 확신도 가중
7. 결과를 `analysis_cache` 에 7일 TTL 저장 (입력 card_id set hash 키)

## 3. 책임 NOT

- 카드 선정 — 사용자가 frontend 에서 Top 6 선택 (또는 자동 Top 6 = DerivedMetrics 의 top5+1)
- 출처 추적 — EvidenceAgent 가 사전에 부착 (재사용)
- 시각화 — frontend Insight 페이지가 4-tab 렌더

## 4. 입력 스펙

```python
class InsightCascadeInput(TypedDict):
    card_ids: list[str]          # 6 권장, 4~10 허용
    context: dict | None         # frontend 가 추가 컨텍스트 제공 (예: 사용자 관심사)
```

## 5. 출력 스펙

```python
class InsightCascadeOutput(TypedDict):
    cause: list[str]
    change: list[str]
    impact: list[str]
    response: list[str]
    confidence: float            # 0.0~1.0
    sources: list[dict]          # 사용된 카드 id 와 매핑
    provenance: dict
    warning: str | None          # confidence < 0.6 시 표시
```

frontend `POST /api/insights/generate` 응답.

## 6. 알고리즘

### 6.1 Context 구성

```python
def build_context(card_ids):
    cards = fetch_cards(card_ids)
    context_parts = []
    for c in cards:
        context_parts.append(f"""
[{c.id}] {c.title}
- Peer: {c.peer_id}
- Sector: {c.sector}
- Exposure: {c.exposure_band} ({c.exposure_score:.2f})
- 요약: {' / '.join(c.summary_lines)}
- 시사점: {c.implication.get('why_important','')}
""")
    return "\n".join(context_parts)
```

### 6.2 LLM Prompt (gpt-4o, chain-of-thought)

```text
당신은 SK AX 사업전략팀의 인텔리전스 분석가다. 다음 카드 뉴스 {N}개를 보고 4단계 인사이트를 도출하라.

[카드 뉴스]
{context}

[4단계 분석]
다음 순서로 사고하라. 각 단계는 3~5 bullet 로 간결하게.

1. **Cause (원인)**: 이 카드들이 발생한 배경 / 시장 환경 / Peer 의 전략적 motivation 은?
2. **Change (변화)**: Peer 가 실제로 어떤 행동/투자/제품을 했는가? (사실 위주)
3. **Impact (영향)**: SK AX 의 현재 사업·고객·경쟁 환경에 어떤 영향?
4. **Response (대응)**: SK AX 가 취할 수 있는 구체적 액션 (사업/기술/조직 방향)?

**제약**:
- 각 단계 답변은 [카드 ID] 로 근거 인용 (예: "[CN-20260513-001]")
- 출처에 없는 수치/이름 만들지 마라 (환각 금지)
- 불확실하면 "추정" 표시

[JSON 출력]
{
  "cause": ["...", "..."],
  "change": ["...", "..."],
  "impact": ["...", "..."],
  "response": ["...", "..."],
  "confidence": 0.0~1.0,
  "sources_used": ["CN-..." card_ids],
  "uncertainties": ["..."]
}
```

### 6.3 Confidence 산출

```python
def compute_confidence(llm_self_confidence, card_trust_avg):
    return 0.5 * llm_self_confidence + 0.5 * card_trust_avg / 100.0
```

### 6.4 Cache 키

```python
import hashlib
def cache_key(card_ids):
    sorted_ids = sorted(card_ids)
    return hashlib.sha256(",".join(sorted_ids).encode()).hexdigest()[:16]
```

## 7. LLM 모델 + token 예산

- **모델**: gpt-4o (reasoning 깊이 필요)
- 토큰/호출: ~5,000 (in 3,000 + out 2,000)
- 일일 호출: ~10 (사용자 액션 가정)
- **일일 비용**: ~₩650

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| JSON parse 실패 | retry 1회 → 실패 시 503 응답 |
| 카드 1개 이하 | 400 BadRequest |
| LLM 응답에 "카드 ID 인용" 빠짐 | warning 부착, confidence -0.2 |
| confidence < 0.6 | warning 필드에 "근거 불충분 — 다른 카드 조합 권장" |
| `uncertainties` 비어있지 않음 | UI 에 표시 |

## 9. 외부 의존성

- **DB**: `card_news` (READ), `evidence_chain` (READ), `analysis_cache` (UPSERT)
- **외부 API**: OpenAI gpt-4o
- **lib**: `openai>=1.30`

## 10. State 흐름

AnalysisSupervisor 의 state (간소 — on-demand 단일 호출):

```python
class AnalysisState(TypedDict):
    request_type: Literal["insight","mixer","peer_compare","link_verify"]
    card_ids: list[str]
    result: dict
    confidence: float
```

## 11. Provenance + Confidence

- **Provenance**: ProvenanceTrackerAgent 가 자동 부착
  ```json
  {"raw_article_ids": [...], "llm_model": "gpt-4o", "prompt_version": "insight-v1.0",
   "run_at": "...", "agent": "InsightCascadeAgent", "card_ids_input": [...]}
  ```
- **Confidence**: §6.3 산식

## 12. 테스트 시나리오

| Unit | 6 카드 (모두 sector='ax') | cause/change/impact/response 각 3+ bullets, confidence ≥ 0.6 |
| Unit | 카드 1개 | 400 BadRequest |
| Unit | confidence < 0.6 | warning 부착 |
| Edge | LLM 환각 (없는 회사명) | uncertainties 에 mark |
| Integration | 동일 card_id set 두 번 호출 | 두 번째는 cache hit (LLM 호출 0) |

## 13. 모니터링

- KPI:
  - 평균 confidence ≥ 0.70
  - cache hit ratio ≥ 30% (사용 패턴 따라)
  - 평균 latency ≤ 8초
- token 예산: ₩650/일

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/insight_cascade_agent.py` (신규 P7)
- backend wiring: `axis-backend/InsightController` 의 fixture → axis-ai 위임으로 교체

### Changelog

- **v1 (제안, P7)** — 4-step CoT + cache + confidence
