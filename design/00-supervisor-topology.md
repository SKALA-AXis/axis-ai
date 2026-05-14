# Supervisor Topology — 5+1 Supervisor 전체 흐름

> **버전**: v1.1 (2026-05-13) · **상위 문서**: `axis-infra/docs/AI_AGENT_DESIGN.md` v1.1 §2
>
> 본 문서는 v1.1 의 7-supervisor 카탈로그를 design 단계에서 **5 + 1 supervisor** 로 통합 한 권장안 (v1.1 §11 Open question 의 후속). 발표 후 P9 cleanup PR 에 ADR-0009 로 정착.
>
> **`+1`** = Briefing sub-supervisor (user-triggered async, P7+) — design v1 작성 시 누락되었던 `POST /api/briefings/generate` 의 owner. Delivery 의 daily SES 메일과 별개 (60-briefing/briefing-generation.md 참조).
>
> 전체 endpoint contract → `axis-infra/docs/API_SURFACE.md` (FE↔BE↔axis-ai 3 시스템 매핑이라 인프라 레포 SoT).

## 1. 5+1 Supervisor 도식

```text
                         AxisRouterSupervisor (entry, router-only — no design file)
                                    │
        ┌───────────────┬───────────┼───────────┬──────────────┬──────────────┐
        ▼               ▼           ▼           ▼              ▼              ▼
   Ingestion       Enrichment   Analysis    UserQuery     WeakSignal     Briefing
        │               │           │           │              │              │
        ▼               ▼           ▼           ▼              ▼              ▼
  매시 정각        매시 후속      user POST    user 검색      월 09:00       user POST
  (외부 fetch)    + nightly      request     + chat         (W7+)         (async)
        │               │           │           │              │              │
        └───────────────┴───────────┴───────────┴──────────────┴──────────────┘
                                    │
                                    ▼
                       Cross-cutting middleware (decorator, axis-ai):
                          ConfidenceScoreMiddleware
                          ProvenanceTrackerMiddleware
                          TokenBudgetMiddleware
                       (외부 spec — axis-infra/docs/):
                          AUDIT_LOG.md           ← BE Spring AOP + V14
                          OBSERVABILITY_LANGFUSE.md ← Helm + LangChain handler
```

**AxisRouterSupervisor** — LangGraph entry dispatcher. trigger 종류 / endpoint path 기반으로 6개 sub-supervisor 그래프 중 하나를 invoke. *agent 가 아니라 router 만 담당하므로 별도 design file 없음.*

**Delivery** — axis-ai agent 미사용 (BE Java 직빌드). Spring `@Scheduled.sendDailyBriefing()` → `BriefingService.generateAndSend()` → `SesMailService` (IRSA) 로 동작. supervisor 카탈로그에서 제거. (단 daily 이메일과 별개로 user-triggered briefing 문서 생성은 **Briefing supervisor** 가 담당)

## 2. Supervisor 별 책임 매트릭스

| Supervisor | Trigger | 시간 예산 | LLM 호출 (cycle) | 핵심 책임 | Agent count |
|---|---|---|---|---|---|
| **Ingestion** | Spring @Scheduled 매시 정각 | 30초/cycle | ~80 (relevance + classify + card_news + summary + analysis) | 외부 fetch → 정규화 → 신뢰도/관련성 → dedup → 분류 → card_news INSERT → evidence_chain → Qdrant index | 8 nodes + 3 sub |
| **Enrichment** | Ingestion 후 + nightly 02:00 | 60초/cycle | ~4 (WordCloud 카테고리 라벨링만) | 카드 → 키워드 derivative + 그래프 + word cloud + search suggest + 집계 메트릭 | 5 |
| **Analysis** | User POST request (axios) + nightly 04:30 (global-trends) | 10초 timeout (15초 global) | 3~10 per req | Insight 4단계 (CoT) / Mixer 신호 (CoT) / Peer 비교 + Forecast (CoT) / Global trends (CoT) / Link verify | 5 |
| **UserQuery** | User search box + FloatingAiChat | 10초 / 5~15초 | 1~3 (SC iter) + 2~5 (chat turn) | Hybrid Search + Rerank + Generative Answer + 대화 orchestration | 4 |
| **WeakSignal** | Spring @Scheduled 월 09:00 | 60초/cycle | ~5 | 채용/특허/MOU 패턴 매칭 + 이상 탐지 + 사용자 rule routing | 1 (통합) |
| **Briefing** | User POST `/api/briefings/generate` (axios → BE → axis-ai async) | 30초 평균, 60초 max | ~11 (10 section + 1 exec summary) | 기간/peer/sector 필터 → 카드 종합 → BriefingReport 5-phase | 1 |
| **(Delivery)** | Spring @Scheduled 08:30 KST MON-FRI | 5초 | 0 | (axis-ai 비관여 — BE Java + SES IRSA) | 0 (BE only) |

### Agent 합산 (혼동 방지)

axis-ai/design/ 범위 — **LangGraph agent + axis-ai 미들웨어만**:

- **Top-level agents (cycle/request 마다 LLM 결정 발생)**: 8 + 5 + 5 + 4 + 1 + 1 = **24**
- **Sub-agents (Ingestion 내부 utility)**: financial-linker / ir-parser / embed-index = **3**
- **Middleware (axis-ai decorator)**: 3 (confidence / provenance / token-budget)
- **합계 axis-ai/design/**: 24 + 3 + 3 = **30** (+ 구조 doc 3 = **33 file**)
- v1.1 통합 design 의 "35-agent" 와 차이는 §4 의 통합 표 참조 (5 묶음).
- **2026-05-14 PDF 반영** — Analysis 에 GlobalTrendsAgent 추가, 구조 doc 에 `02-prompt-design-checklist.md` 추가

axis-infra/docs/ 범위 — **인프라 + 크로스시스템 spec**:

- API_SURFACE.md (FE↔BE↔axis-ai contract)
- AUDIT_LOG.md (BE Spring AOP + V14)
- OBSERVABILITY_LANGFUSE.md (Helm + 3-tier 분담)
- admin/user_events.md (V15)
- admin/feedback.md (V16)
- admin/cost_reconciliation.md (V17+V18)
- admin/metrics_exporter.md (Prometheus + Grafana)
- (기존) admin_page.md (전체 대시보드 spec)

= **7 + 1 file**. axis-ai 의 LangGraph 그래프는 본 디렉토리, 그 위 인프라 / BE 책임은 axis-infra/docs/ 가 SoT.

## 3. Supervisor 분리 기준

본 5-supervisor 안은 다음 4 차원으로 분리:

| 차원 | 값 | 영향 |
|---|---|---|
| **Trigger 주기** | 매시 / nightly / on-demand / 주 1회 | 다른 주기 = 다른 supervisor (state 격리) |
| **Consumer** | 시스템 자체 / user / scheduler / external email | 다른 consumer = 다른 supervisor |
| **시간 예산** | 30초 / 60초 / 10초 / 15초 | 다른 budget = 다른 timeout 정책 |
| **LLM density** | 80 calls / 4 / 3~10 / 2~5 / 5 | 비용 추적 단위 |

**왜 Search + Dialogue 통합 (UserQuery)**:
- 같은 trigger (user request) + 같은 consumer (user) + 같은 시간 예산 (~10초)
- Chat 은 본질적으로 Search wrapper (Intent 파악 → Search 위임 → 대화 응답)
- 통합 시 Search 의 HybridSearchAgent/AnswerAgent 를 Chat 이 직접 재사용 (state 공유 용이)

**왜 Delivery 제외**:
- axis-ai 측 agent 0개 (`delivery_graph.py` 는 dead code, 호출자 없음)
- 실 발송 = Spring `BriefingService.java` + AWS SES V2 SDK + IRSA (2026-05-12 검증 완료)
- supervisor 카탈로그는 axis-ai 의 agent 만 다룸

**왜 WeakSignal 분리 유지 (Enrichment 와 통합 안 함)**:
- 같은 batch 이지만 cadence 다름 (매시 vs 주 1회) → 다른 cron + 다른 timeout
- LLM 의존성 다름 (Enrichment 거의 0 vs WeakSignal anomaly 해석에 LLM)
- 운영 alert 가 외부 (사용자 메일) 로 나감 → 격리 권장

## 4. 통합 vs 분산 — v1.1 → v1 본 design 차이

v1.1 통합 design 의 7-supervisor / 35-agent 안은 **feature space 카탈로그** 였다 (모든 기능을 카탈로그화). 본 design 디렉토리의 **5-supervisor / 22-agent** 안은 **구현 단위** 다 (실 코드 합쳐도 되는 것은 합침).

### 통합 후 agent (v1.1 → 본 design)

| v1.1 agents | 통합 후 (본 design) | 위치 |
|---|---|---|
| IssueCardAgent + NewsSummaryAgent + NewsAnalysisAgent | **CardComposerAgent** (3-phase internal) | 10-ingestion/card-composer.md |
| ParserAgent + ParserQualityAgent + DartParserAgent | **ParserAgent (strategy)** (source 별 strategy + quality check 포함) | 10-ingestion/parser.md |
| IntentRouterAgent + ConversationAgent | **ChatOrchestratorAgent** | 40-user-query/chat-orchestrator.md |
| PatternDetectAgent + AnomalyDetectionAgent + AlertRoutingAgent | **WeakSignalAgent** (3-phase) | 50-weak-signal/weak-signal.md |
| TopInsightSelectorAgent + TrendAggregatorAgent + MonitoringOverviewAgent | **DerivedMetricsAgent** (3 mode) | 20-enrichment/derived-metrics.md |

### 분리 유지 (책임 다름)

- KeywordExtractionAgent · KeywordGraphBuilderAgent · PeerWordCloudAgent · SearchSuggestAgent — 모두 키워드 derivative 지만 **output schema 와 caching key 가 달라** 분리 (5 files)
- Insight / Mixer / PeerComparison / LinkVerification — output schema 완전 다름 (4 files)
- Hybrid Search / Rerank / Answer — 각각 LangGraph node + 다른 sub-task (3 files)

## 5. State 모델

### LangGraph TypedDict (Ingestion 기준)

```python
class IngestionState(TypedDict):
    company: list[str]                          # 처리 대상 peer ids
    trigger_type: str                           # 'scheduled' | 'manual'
    collected_since: str | None
    crawl_run_id: str | None

    # 단계별 누적 ids (이전 노드의 출력 → 다음 노드의 입력)
    raw_article_ids: list[int]
    credible_ids: list[int]
    relevant_ids: list[int]
    official_document_ids: list[int]
    parsed_document_ids: list[int]
    industry_document_ids: list[int]
    structured_signal_ids: list[int]
    skipped_preprocess_ids: list[int]

    # 분류·카드 단계
    cluster_map: dict                           # {cluster_id: [article_ids]}
    representative_ids: list[int]
    classified_clusters: list[dict]
    card_news: list[dict]                       # V9 후 — 구 issue_cards
    evidence_results: list[dict]
    indexed_vector_ids: list[str]

    # 에러 + 인적 검토
    errors: Annotated[list[str], operator.add]
    human_review_flags: list[int]
```

### State 변환 규칙

1. **불변 추가 only** — 노드는 새 key 를 추가만 함. 기존 key 의 값을 mutate 하지 않음.
2. **list 는 append-only** — `operator.add` annotation 으로 merge (LangGraph 기본).
3. **에러 격리** — 한 노드의 exception 은 `errors` 에 append, 다음 노드는 자기 입력만 보고 동작.

## 6. Cross-cutting middleware 주입

```python
# 각 agent 의 __call__ 또는 @decorator 로 wrapping
@with_provenance       # → evidence_chain.provenance 자동 부착
@with_token_budget     # → 호출 전 budget check, 초과 시 fallback
@with_confidence       # → 출력에 confidence 점수 자동 부착
@with_audit            # → pipeline_logs 자동 기록
def some_agent(state):
    ...
```

각 cross-cutting agent 의 상세는 [90-cross-cutting/](90-cross-cutting/).

## 7. Optimal Path 결정 흐름

```text
User 액션 / 시스템 trigger
    │
    ▼
AxisRouterSupervisor (LangGraph entry)
    │
    ├─ trigger == 'scheduled_hourly'   → Ingestion 그래프
    ├─ trigger == 'scheduled_nightly'  → Enrichment 그래프
    ├─ trigger == 'scheduled_monday'   → WeakSignal 그래프 (W7+)
    ├─ POST /api/insights/generate     → Analysis.InsightCascade
    ├─ POST /api/mixer                 → Analysis.MixerAnalysis
    ├─ GET /api/monitoring/{id}/strategy → Analysis.PeerComparison
    ├─ POST /api/cards/{id}/verify-link → Analysis.LinkVerification
    ├─ POST /api/search                → UserQuery.HybridSearch + Rerank
    ├─ POST /gen-search                → UserQuery.HybridSearch + Rerank + Answer
    ├─ POST /api/assistant/chat        → UserQuery.ChatOrchestrator → (Search 위임)
    └─ background nightly              → Enrichment 그래프 일부
```

## 8. 후속 작업

- 본 디렉토리 의 각 agent file 확정 후 → v1.1 통합 design (`axis-infra/docs/AI_AGENT_DESIGN.md`) 의 §3 카탈로그 항목과 1:1 매핑 표 추가
- ADR-0009 (Supervisor 분리 + 통합 원칙) — 본 5-supervisor 안을 spec 화
- 발표 후 (W9 cleanup) v1.1 통합 design 을 v2 로 압축하여 본 디렉토리 와 일관성
