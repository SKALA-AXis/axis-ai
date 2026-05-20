# ConfidenceScoreMiddleware — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `ConfidenceScoreMiddleware` (cross-cutting decorator + computation rule book) |
| **Supervisor** | (none — 모든 agent 에 주입) |
| **상태** | 🟡 부분 — 개별 agent (NewsSummary / NewsAnalysis / CardNews / Answer) 가 자체 confidence 부여 중. 표준화 + propagation 누락 |
| **Trigger** | 모든 agent 의 출력 시점 |

## 2. 책임

**한 줄**: 모든 AI 출력에 0~1 사이 `confidence` 점수를 부여 + supervisor state 를 따라 propagate + UI / API 응답에 surface.

**구체적**:

1. **계산** — agent 유형별 deterministic rule book (아래 §6) 에 따라 confidence 산출
2. **합성** — 다단계 agent (CardComposer 3-phase / Answer SC 3-iter) 의 sub-confidence 를 한 점수로 reduce
3. **저장** — `card_news.implication.confidence` / `evidence_chain.provenance.confidence_components`
4. **표면화** — API 응답 (`CardNewsResponse.confidence`) + UI threshold 처리 (`< 0.6` → 경고 badge / `< 0.4` → "검토 필요" 라벨)

## 3. 책임 NOT

- **사실 검증** — EvidenceAgent 의 `pass / missing` (out-of-evidence 추적). Confidence 는 *추정의 신뢰도*, evidence 는 *추적 가능성*. 두 축이 다름.
- **품질 등급** — exposure_score (영향력 산식) 와 분리. Confidence ≠ exposure.
- **A/B 비교용 score** — MLflow 의 model evaluation metric 은 별도 (precision / recall).

## 4. 입력 스펙

```python
class ConfidenceContext(TypedDict):
    agent_name: str                    # "CardNewsAgent" 등
    phase: str | None                  # 다단계 agent 의 phase (예: "Summarize")
    inputs: dict                       # 산출에 사용한 입력 (raw_article_ids, cluster_size 등)
    raw_output: dict                   # LLM 또는 산식의 원시 출력
    llm_self_confidence: float | None  # LLM 이 JSON 에 자체 보고한 값
    sc_agreement: float | None         # Self-Consistency 일치율 (Answer agent)
    error_count: int                   # retry 횟수
```

## 5. 출력 스펙

```python
class ConfidenceResult(TypedDict):
    score: float                       # 0.0 ~ 1.0 최종 점수
    band: Literal["high", "medium", "low", "review"]   # ≥0.8 / 0.6-0.8 / 0.4-0.6 / <0.4
    components: dict                   # 점수 계산에 기여한 항목 (audit 용)
    # 예: {"llm_self": 0.85, "evidence_coverage": 0.9, "sc_agreement": 0.67, "weight": [0.4,0.3,0.3]}
    threshold_action: Literal["pass", "warn_ui", "needs_review"]
```

`confidence_components` 는 `evidence_chain.provenance` 안에 함께 저장 (audit 가능).

## 6. 알고리즘 — Agent 유형별 산식

### 6.1 Deterministic agent (산식 only)

```python
# Crawler / Parser / Credibility / Dedup / EmbedIndex / DerivedMetrics
confidence = 1.0  # 산식 = 결정적, 입력 검증만 통과하면 100%
```

예외:
- Parser: `parse_strategy_fallback=True` 면 0.7 (best-effort generic strategy 사용)
- Dedup: cluster_size=1 (단독 기사) 면 0.6 (cluster 정보 부족)

### 6.2 LLM 단일 호출 agent

```python
# Classification / IRParser (PDF OCR) / KeywordExtractionAgent (LLM 모드)
def compute(ctx):
    llm_self = ctx.raw_output.get("confidence", 0.7)
    retry_penalty = max(0, 1.0 - 0.1 * ctx.error_count)
    return min(llm_self, retry_penalty)
```

### 6.3 다단계 agent (CardComposer 3-phase)

```python
# Summarize / Analyze / Compose
def compute_card_composer(phase_scores):
    """
    Phase 1 (Summarize): is_valid_summary → 1.0 if true else 0.3
    Phase 2 (Analyze):   LLM self-report (보통 0.6~0.9)
    Phase 3 (Compose):   deterministic = 1.0
    """
    w = [0.3, 0.6, 0.1]  # Analyze 가 가장 중요
    return sum(w[i] * s for i, s in enumerate(phase_scores))
```

### 6.4 Self-Consistency agent (Answer)

```python
# AnswerAgent — SC ×3 + judge
def compute_answer(sc_agreement, judge_score):
    # sc_agreement (0~1) 는 가장 직접적 indicator
    # judge_score = 0 ~ 1 (LLM judge 의 일치 점수)
    return min(sc_agreement, judge_score)
```

### 6.5 Routing agent (ChatOrchestrator)

```python
# Intent 분류 confidence × sub-agent confidence
def compute_chat(intent_conf, sub_result):
    return min(intent_conf, sub_result.get("confidence", 1.0))
```

### 6.6 Anomaly agent (WeakSignal)

```python
# z-score 기반 → 정규화
def compute_anomaly(z):
    return min(abs(z) / 3.0, 1.0)
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — pure 산식 middleware
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| Agent 가 confidence 미출력 | 보수적 default 0.5 + 로그 경고 |
| 계산 결과 [0, 1] 범위 밖 | clamp(score, 0.0, 1.0) |
| llm_self_confidence > 0.95 + sc_agreement < 0.5 | hallucination 의심 → 0.4 강제 (LLM over-confident bias) |
| Phase 결과 dict 가 누락 | 누락 phase 는 0.5 로 가정 |

## 9. 외부 의존성

- 없음 (pure Python)
- 다만 evidence_chain.provenance 저장 시 ProvenanceTracker 와 연동

## 10. State 흐름

```python
# 모든 agent 의 출력 후 wrapper 가 confidence 부착
@with_confidence  # decorator
def card_composer_node(state):
    cards = compose(state["classified_clusters"])
    return {**state, "card_news": cards}  # 각 card 의 implication.confidence 자동 채워짐
```

State 의 `card_news[i].implication.confidence` 채워짐. supervisor 가 별도 reduce 안 함 (per-card 점수).

## 11. Provenance + Confidence

- 본 middleware 자체가 confidence 의 owner. 다른 agent 는 *입력만* 제공.
- `evidence_chain.provenance.confidence_components` 에 계산 내역 저장 (audit + debug 용).

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | Crawler (deterministic) | score=1.0 |
| Unit | CardComposer Phase 1 valid=true, Phase 2 LLM=0.8, Phase 3 det | score = 0.3·1.0+0.6·0.8+0.1·1.0 = 0.88 |
| Unit | Answer SC agreement=0.5 | score=0.5, band=low |
| Unit | Anomaly z=4.0 | score=1.0 (clamped) |
| Edge | LLM self=0.99, SC=0.4 | score=0.4 (over-confident bias detected) |
| Edge | agent 가 confidence 안 줌 | 0.5 default + 로그 경고 |
| Integration | CardComposer → confidence < 0.6 → BriefingService 가 skip | end-to-end 확인 |

## 13. 모니터링

- KPI:
  - 평균 confidence (card_news) ≥ 0.75
  - `band=review` 비율 ≤ 5%
  - `band=low` 비율 ≤ 15%
- Grafana panel: `histogram_quantile(card_news.confidence, [0.5, 0.9, 0.99])`
- Alert: 일 평균 confidence < 0.6 → Slack (deprecated, 이메일) 경고

## 14. 구현 메모 + Changelog

### 핵심 파일 (P9 신설)

- `src/middleware/confidence.py` (신규) — `@with_confidence` decorator + rule book
- 영향 받는 agent: 전체 (decorator 일괄 적용)
- DB column: `card_news.implication.confidence` (이미 jsonb 에 존재) + `evidence_chain.provenance.confidence_components` (jsonb 확장)
- 신규 마이그레이션: 불필요 (jsonb 확장)

### 의존 agent 가 변경할 점

- 각 LLM agent 의 prompt 에 `"confidence": 0.0~1.0` JSON 필드 의무화 (이미 NewsAnalysis / IssueCard 는 있음)
- Multi-phase agent 는 phase 별 confidence 도 반환 (CardComposer Phase 1 의 `is_valid_summary` 가 그 예)

### Changelog

- **v1 (제안, P9)** — 표준 confidence rule book + decorator 통합. 현재는 각 agent 가 임의 산출.
