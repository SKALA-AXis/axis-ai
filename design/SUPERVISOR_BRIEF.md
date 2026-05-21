# 1단계 데이터 분석 Supervisor — 팀 공유본

> 상세: [`01-supervisor-implementation-plan.md`](01-supervisor-implementation-plan.md) (v3.2.1)
>
> ## 배포 순서 (Hard dependency, 외부 리뷰 R-6 명시)
>
> 다음 순서를 반드시 지켜야 정상 동작:
>
> 1. **axis-backend V32_5 + V33 마이그레이션 적용** — `peer_companies.profile_snapshot` /
>    `card_news.card_schema_version` / `evaluation_payload` 컬럼, `peer_event_timeline` /
>    `peer_financial_trend` VIEW, `sector_pulse` MV 가 만들어진다.
> 2. **axis-ai image 빌드/ECR push** — `scripts/refresh_peer_profile_snapshots.py` /
>    `refresh_capability_evolution.py` / `evaluate_recent_cards.py` 가 image 안에 포함.
> 3. **axis-infra CronJob image tag 갱신 + apply** — CronJob 4종 (profile-refresh /
>    capability-evolution / sector-pulse / card-evaluator) 활성.
> 4. **운영 첫 1주는 W5-1 threshold 비활성** — 카드 차단/플래그 0건, 측정만. 7일 후
>    percentile-based calibration 으로 threshold 자동 산출 (Phase 2).
>
> ## Retry / Tracing 현재 상태 (외부 리뷰 R-7 명시)
>
> * LangGraph `RetryPolicy / with_retry` 정식 적용: **별도 PR** (W3-4 trace + retry 묶음).
> * 현재는 `_logged_step` 데코레이터의 try/except 가 노드 예외를 잡아 `state.errors[]` 에
>   누적하고, ImplicationAgent 가 내부에서 LLM 실패 시 heuristic generator 로 자동 fallback.
> * Langfuse tracing 도 ImplicationAgent / CapabilityEvolutionAgent 만 부분 적용 — 다른
>   supervisor 노드 (issue_integrate / strategic_analyze) 는 별도 PR 에서 trace metadata 부착.
> 작성: 2026-05-20 / 갱신: 2026-05-21 · Critical path **~40h** / 총 **~85h** / **5-5.5주** (W5 Evaluation Layer 포함)

---

## 1. 만들 에이전트 (총 13종, 신규 9종)

| # | Agent | 위치 | 호출 시점 | LLM |
|---|---|---|---|---|
| 1 | **DataAnalysisSupervisorAgent** (재설계) | `pipeline/supervisor_graph.py` | cluster 마다 | ❌ (조율) |
| 2 | IssueIntegrationAgent (유지) | `agents/issue_integration_agent.py` | Supervisor 노드 | ✅ |
| 3 | AnalysisAgent (유지) | `agents/analysis_agent.py` | Supervisor 노드 | ✅ |
| 4 | ProfileAgent (2-tier 분리) | `agents/profile_agent.py` | Supervisor 노드 | ❌ (CronJob 분리) |
| 5 | **ImplicationAgent v4.0** (신규) | `agents/implication_agent.py` | Supervisor 노드 | ✅ |
| 6 | CardNewsAgent (수정 + 이관) | `agents/card_news_agent.py` (호출은 supervisor `card_writer` 노드) | Supervisor 노드 (W2-1 작업 5) | ✅ |
| 7 | **AnalysisContextBuilder** (신규) | `services/analysis_context_builder.py` | Supervisor 노드 | ❌ |
| 8 | **CapabilityEvolutionAgent** (신규) | `agents/context/capability_evolution_agent.py` | 월1회 CronJob | ✅ |
| 9 | **SectorPulseAggregator** (신규) | `agents/context/sector_pulse_aggregator.py` | 주1회 CronJob | ❌ |
| 10 | **EventChainDiscoveryAgent** (옵션) | `agents/context/event_chain_discovery_agent.py` | 매일 CronJob | ✅ |
| 11 | **ProfileSnapshotAgent** (신규) | `scripts/refresh_peer_profile_snapshots.py` | 주1회 CronJob | ✅ |
| 12 | **EvaluatorAgent** (W5-1 신규, rule-based 5 metric) | `agents/evaluator_agent.py` + `validate` 노드 확장 | Supervisor 노드 (in-graph) | ❌ |
| 13 | **CardEvaluatorSidecar** (W5-2 신규, LLM-as-Judge 4 score) | `scripts/evaluate_recent_cards.py` | 5분 주기 CronJob (sidecar) | ✅ gpt-4o-mini |

---

## 2. 구조 — 2-Layer Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Layer A — Data Pipeline (axis-cron-ingestion-* / ingestion_graph)   │
│   crawl → dedup → classify → 정제·매칭                              │
│                          ↓                                          │
│            DB 저장 (raw_articles + raw_article_*)                   │
└─────────────────────────────────────────────────────────────────────┘
                          ↓ (cluster 단위 trigger)
┌─────────────────────────────────────────────────────────────────────┐
│ Layer B — Analysis Supervisor Graph (이 계획서의 작업 범위)         │
│                                                                     │
│   ① context assemble                                                │
│      profile_context  →  build_analysis_context (DB+Qdrant, no LLM) │
│                          ↓                                          │
│   ② LLM reasoning                                                   │
│      issue_integrate (LLM) → strategic_analyze (LLM)                │
│                                       ↓                             │
│                          implication (LLM, v5.0)                    │
│                                       ↓                             │
│   ③ validation + rule-based eval (W5-1)                             │
│      validate (단정/수치 차단 + 5 metric 계산) ── fail ─→ human_review│
│                ↓ pass / low_quality_score → human_review_flags      │
│   ④ package generation                                              │
│      assemble → AnalysisPackage                                     │
│             → card_writer (CardNewsAgent.write_card)                │
│                          ↓                                          │
│         card_news WRITE (v2 schema + evaluation_payload.rule_based) │
└─────────────────────────────────────────────────────────────────────┘
                          ↑ DB read           ↓ 5분 후 평가
                  ┌─────────────────────────┐  ┌─────────────────────────┐
                  │ Context Layer CronJobs  │  │ Evaluation Sidecar (W5) │
                  │  주1회: ProfileSnapshot │  │  5분: CardEvaluator     │
                  │  주1회: SectorPulse MV  │  │       (LLM-as-Judge,    │
                  │  월1회: CapabilityEvol. │  │        gpt-4o-mini)     │
                  │  매일 : EventChain(opt) │  │  매일: RegressionCheck  │
                  └─────────────────────────┘  │       (옵션, W5-3)      │
                                               └─────────────────────────┘
                                                  ↓ UPDATE card_news
                                          evaluation_payload.llm_judge
```

> **As-Is design debt (v3.1.3 해소)**: 현재 코드에서는 카드 생성 (`CardNewsAgent.write_card`) 이 `ingestion_graph.card_news_node` 에서 호출됨 — Layer 위반. **W2-1 작업 5 에서 `card_writer` 노드로 supervisor 의 마지막 노드에 통합** (W2-1 시간 +2~3h, 본 계획서 범위 내). 카드뉴스는 분석/시사점/대응의 직렬화 결과 = Layer B 의 산출물.

---

## 3. Agent I/O (DB 테이블·컬럼 기준)

### 3.1 Supervisor 흐름 내 (cluster-time)

| Agent | Input | Output |
|---|---|---|
| **ProfileAgent** (2-tier) | **READ**: `peer_companies.profile_snapshot` JSONB 컬럼 (Tier A) + `raw_article_business_signals` 최근 30일 top-3 + `raw_article_financial_metrics` 최근 분기 (Tier B) | `ProfileContext` (메모리) |
| **AnalysisContextBuilder** ⭐신규 | **READ**: `peer_event_timeline` VIEW (90일) + `peer_companies.peer_plus_payload['capability_evolution']` + `sector_pulse` MV (4주) + `peer_financial_trend` VIEW (8분기) + `card_news.evidence_payload.financial_refs` + Qdrant `axis_main` (top-3) | `AnalysisContext` (메모리, ≤4k token) |
| **IssueIntegrationAgent** | `AnalysisInputBundle` (cluster 의 raw_articles) | `IntegratedIssue` (메모리, consolidated_facts / key_numbers / fact_basis) |
| **AnalysisAgent** | `IntegratedIssue` + `ProfileContext` | `AnalysisResult` (메모리, strategic_meaning / impact_level / risk_or_opportunity) |
| **ImplicationAgent v4.0** ⭐신규 | `Bundle` + `IntegratedIssue` + `AnalysisResult` + `ProfileContext` + `AnalysisContext` | `ImplicationResult` (메모리, peer_implication / skax_implication / follow_up / confidence) |
| **Validate 노드 + EvaluatorAgent** ⭐신규 (W2-3 + W5-1) | 전체 SupervisorState (+ rolling 7d confidence) | `ValidationReport` (pass/fail + violations + **rule-based 5 metric**) |
| **CardNewsAgent** (via `card_writer` 노드) | `AnalysisPackage` (모든 결과) + ValidationReport | **WRITE**: `card_news` 행 (v2 schema + `evaluation_payload.rule_based`) |

### 3.2 Context Layer CronJobs (배치) + Evaluation Sidecar (W5)

| Agent | 주기 | Input | Output |
|---|---|---|---|
| **ProfileSnapshotAgent** | 주1회 (월 03:00) | **READ**: `raw_articles` (sk_ax_site / official / DART) + `raw_article_business_signals` | **WRITE**: `peer_companies.profile_snapshot` JSONB 컬럼 (+ version / generated_at) |
| **CapabilityEvolutionAgent** ⭐신규 | 월1회 (1일 03:00) | **READ**: `raw_article_business_signals` 4분기 top-5/group | **WRITE**: `peer_companies.peer_plus_payload['capability_evolution']` JSONB |
| **SectorPulseAggregator** ⭐신규 | 주1회 (월 02:00) | **READ**: `card_news` (180일) | **WRITE**: `sector_pulse` MATERIALIZED VIEW REFRESH |
| **EventChainDiscoveryAgent** (옵션) | 매일 (02:00) | **READ**: `card_news` 14일 + Qdrant 임베딩 | **WRITE**: `card_news.evidence_payload['related_card_ids']` JSONB |
| **CardEvaluatorSidecar** ⭐신규 (W5-2) | **5분 주기** | **READ**: `card_news WHERE card_schema_version='v2' AND NOT (evaluation_payload ? 'llm_judge')` (최근 24h, batch 20, cost cap 적용) | **WRITE**: `card_news.evaluation_payload['llm_judge']` JSONB (4 score + reasoning) |
| **EvalRegressionCheck** ⭐신규 (W5-3, **Phase 2, 운영 14일 후**) | 매일 (02:30) | **READ**: 최근 14일 `evaluation_payload` 평균 | Slack/email 알림 (delta < -15%) |

---

## 4. DB 변경 — V33 마이그레이션 단 하나

> **신규 테이블 0개**. 컬럼 1 + VIEW 3 + MATERIALIZED VIEW 1 + 인덱스 3.

```sql
-- V33__context_engineering.sql

-- (1) card_news v2 식별 컬럼
ALTER TABLE card_news ADD COLUMN card_schema_version VARCHAR(10) DEFAULT 'v1';

-- (2) 인덱스 3개 (hot path)
CREATE INDEX idx_card_news_peer_created ON card_news(peer_company_id, created_at DESC);
CREATE INDEX idx_business_signals_peer_area_period ON raw_article_business_signals(peer_id, business_area, period_year DESC, period_quarter DESC NULLS LAST);
CREATE INDEX idx_financial_metrics_peer_metric_period ON raw_article_financial_metrics(peer_id, metric_name, period_year DESC, period_quarter DESC NULLS LAST);

-- (3) VIEW 3개 (실측 dry-run PASS)
CREATE VIEW peer_event_timeline AS ...;          -- peer 별 90일 카드 타임라인
CREATE VIEW general_event_timeline AS ...;       -- peer NULL 카드 (31%) 흡수
CREATE VIEW peer_financial_trend AS ...;         -- metric_name 정규화 (net_income/당기순이익/순이익 → net_income)

-- (4) MATERIALIZED VIEW 1개 (주1회 REFRESH)
CREATE MATERIALIZED VIEW sector_pulse AS ...;    -- sector×week 단위 집계 (Phase 1: count+intensity, Phase 2: z-score 는 4주 후)
```

### 기존 테이블 활용

| 데이터 | 저장 위치 |
|---|---|
| Profile snapshot | `peer_companies.profile_snapshot` JSONB **별도 컬럼** (v3.2.1 정정 — `peer_plus_payload` 안이 아님) |
| Capability evolution narrative | `peer_companies.peer_plus_payload['capability_evolution']` JSONB |
| Event chain (옵션) | `card_news.evidence_payload['related_card_ids']` JSONB |
| Snapshot history | `legacy_records` (V30 archive 재사용) |
| **W5 평가 metrics + LLM-judge scores** | `card_news.evaluation_payload['rule_based']` + `['llm_judge']` JSONB **(v3.2.1 정정 — W4-1 V33 마이그레이션에서 ALTER TABLE ADD COLUMN 으로 신설)** |

---

## 5. 카드뉴스 v2 — 요약 + 시사점 + 대응 3섹션

> **`card_news` 컬럼 변경 거의 없음** — `card_schema_version` 1개 추가만. 모든 새 내용은 기존 `implication` JSONB 안.

| 섹션 | 컬럼 / JSONB key |
|---|---|
| **A. 요약** (3줄) | `summary_lines TEXT[]` |
| **B. 시사점** | `implication.skax_implication.why_important` + `.potential_impact` + `.peer_implication.peer_meaning` |
| **C. 대응** (verb-first 최대 3개) | `implication.skax_implication.recommended_actions` |
| 기회 / 위협 (보조) | `implication.skax_implication.opportunities` / `threats` |
| 후속 모니터링 | `implication.follow_up_questions` / `watch_points` |
| 신뢰도 | `implication.confidence` + `evidence_label` |
| 근거 | `evidence_payload` (source_links / financial_refs / mbb_refs / provenance) |

기존 191 카드 (v1) 는 graceful fallback — frontend 가 `card_schema_version` 으로 분기.

---

## 6. 5 단계 21 작업 (5-5.5주)

| 단계 | 작업 | 시간 |
|---|---|---|
| **W1** (1주차) | ImplicationAgent v4.0 / schema 정합화 / metadata typed / card v2 lint | 12h |
| **W2** (2주차) | Supervisor → LangGraph (+ CardNews 이관) / Profile 2-tier / Validate 노드 / CardNews 정리 | 22-25h |
| **W3** (3주차) | design docs / tests / Langfuse / cleanup | 12h |
| **W4** (4주차) | **data hygiene → V33 → ContextBuilder → CapabilityEvol → SectorPulse → prompt v5.0** | 22h |
| **W5** (5주차) | **Evaluation Layer — rule-based eval (W5-1) + LLM-as-Judge sidecar (W5-2)** | **10-14h** |
| W4-6 (옵션) | EventChainDiscovery — 4주 운영 후 결정 | +6h |
| W5-3 (옵션) | Regression Detection CronJob — W5-2 운영 1주 후 | +2-3h |

---

## 7. 비용·품질 영향

| 지표 | 현재 | W4 완료 | **W5 완료** |
|---|---|---|---|
| Stage 1 LLM 호출 / cluster | 4 | 5 (+implication) | 5 (W5-1 LLM X) |
| 일일 LLM 비용 (Stage 1 + CronJobs) | $1.17 | $1.87 (한도의 25%) | **$3.37 (한도의 46%, +$1.5/일 sidecar)** |
| Cluster 처리 시간 (p50) | 30초 | 33초 | 33초 (W5-1 rule-based <50ms) |
| LLM 실패 시 cluster 손실률 | 100% | ~10% | ~10% (sidecar 실패가 cluster 영향 X) |
| `evidence_label="sufficient"` 비율 | — | ≥ 75% | ≥ 75% |
| 출처 없는 수치 carryover | 검증 X | 0 (validate 차단) | 0 + `evidence_claim_ratio` 측정 |
| **출력 품질 정량화** | ❌ | 부분 (process metrics) | **✅ rule-based 5 + LLM-judge 4 score 카드별 기록** |
| **회귀 감지** | ❌ | golden test hash 만 | **✅ daily drift 모니터링 (W5-3)** |

---

## 8. 핵심 결정 6가지

1. **신규 DB 테이블 0개** — 기존 JSONB + VIEW + MV 로 모두 흡수
2. **Cluster-time LLM 호출 0건 증가** — context 합성은 DB query 만
3. **Supervisor → LangGraph 전환** — retry / 노드별 logging / human_review 라우팅
4. **카드 = 요약 + 시사점 + 대응 3섹션** — `implication` JSONB key 표준화로 해결
5. **W4-6 (EventChain) 보류** — 4주 운영 측정 후 결정
6. **`card_news.cluster_id` 사용 금지** — ephemeral seq. `source_raw_article_ids` 또는 `(peer, event_type, date)` 사용
7. **Evaluation Layer Hybrid placement** — rule-based 5 metric 은 in-graph (`validate` 확장, LLM X, latency <50ms, 즉시 회귀 차단), LLM-as-Judge 4 score 는 sidecar (5분 주기 CronJob, cluster critical path 영향 0). 두 결과 모두 `card_news.evaluation_payload` JSONB 에 누적
8. **W5 Phase 분리 (v3.2.1)** — Phase 1 (즉시): rule-based 4 metric + LLM-judge 4 score, threshold 비활성 (측정만). Phase 2 (운영 14일 후): `regression_drift` + W5-3 알림 + Threshold percentile calibration + Claude 10% cross-check sampling. **사용자 결정 — "결과가 너무 적게 나오거나 많이 제한되지 않게"**
9. **W5 schema 정정 (v3.2.1)** — 실측 결과 `card_news.evaluation_payload` 컬럼 미존재 확인. V33 (W4-1) 에 `ALTER TABLE ADD COLUMN evaluation_payload JSONB DEFAULT '{}'::jsonb` + partial index 추가. 기존 v1 카드 191건은 sidecar `WHERE card_schema_version = 'v2'` guard 로 자동 제외.

---

## 9. Layer B 카드 생성 단 (`card_writer` 노드) 의 정합화 — W2-1 작업 5 흡수

> ingestion (Layer A) 회귀가 아닌 **Supervisor 의 `card_writer` 노드** (As-Is `ingestion_graph.card_news_node`, To-Be `card_writer` in `supervisor_graph.py`) 의 정합화 작업. **W2-1 작업 5 (CardNews 이관) 에 흡수** — 노드 이관 시 아래 항목들을 함께 처리.

- 카드 dedup 강화 (같은 `(peer, event_type, DATE(created_at))` 키에 카드 4-6건 발생) — `card_writer` 진입 시 dedup 키 검사
- `card_news.source_raw_article_ids` 빈 15% 카드 — `card_writer` 가 source 비어있으면 hard fail
- `peer_company_id` FK + `primary_keyword_category` 두 시점 회귀 원인 추적 (5/14~5/20 git log) — 별도 P1, W4-0 작업 5 의 백필 SQL 이 즉시 복구

## 10. W4-0 즉시 백필 (이 계획서 범위 내)

- **`card_news.peer_company_id` FK NULL** (5/15 부분 → 5/16~ 100%, 누적 60건+) — `card_news.company` 와 `peer_companies.id` 100% 매칭 확인됨 → `UPDATE card_news SET peer_company_id = company WHERE peer_company_id IS NULL AND company IN (SELECT id FROM peer_companies)` 1쿼리로 복구
- **`card_news.primary_keyword_category` NULL** (5/20 부터 11건) — `raw_articles.matched_sectors` 정상 → `UPDATE … SET primary_keyword_category = (raw_articles.matched_sectors->>0 FROM source_raw_article_ids …)`
- **회귀 원인 추적** (별도 P1) — axis-ai `CardNewsAgent` 후처리의 5/14~5/20 git 변경 이력 점검 (두 시점 회귀이므로 2개 커밋 식별)
