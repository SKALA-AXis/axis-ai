# 1단계 데이터 분석 Supervisor — 구현 계획서 (v3.2)

> 작성: 2026-05-20 KST · 갱신: 2026-05-21 v3.2 (Evaluation & Observability Layer W5 신설)
> 대상: `axis-ai/src/agents/analysis_supervisor_agent.py` 및 산하 4 child + CardNews + Context layer + Evaluator
> 기준 설계: [`design/00-supervisor-topology.md`](00-supervisor-topology.md)
> 교차 검증: [`docs/AGENT_ARCHITECTURE_VERIFICATION.md`](../docs/AGENT_ARCHITECTURE_VERIFICATION.md)
> 실측 검증: `kubectl exec postgres-...` 으로 cluster DB 직접 쿼리 (2026-05-20 KST). 발견 8 critical issue 는 §2.4 참고.

---

## 0. 한 줄 요약

수집·전처리·매칭이 끝난 데이터를 받아 `DataAnalysisSupervisorAgent` 가 4 child agent 를 조율해 `AnalysisPackage` 를 만들고 `CardNewsAgent` 가 카드로 재가공하는 **wire 는 동작 중**. 그러나 (a) 시사점이 LLM 미사용 heuristic 이라 카드 가치의 핵심이 약하고, (b) Profile context 가 cluster-time 에 빈약하며, (c) Supervisor 가 plain Python 순차 호출이라 retry·관측·검증이 모두 부재. **추가로 (d) 시사점·대응 추론에 필요한 과거 누적 맥락 (뉴스 14k, business_signals 27k, financial_metrics 3.4k, IR/DART) 이 cluster-time 에 활용되지 않고 있으며**, (e) 카드뉴스 형식이 `요약` 중심이라 `요약+시사점+대응` 세 섹션을 명확히 분리하는 schema 정합화가 필요. 본 계획서는 (a)(b)(c) 를 **LangGraph 기반 검증 가능한 Supervisor 로 재설계**, (d) 를 **4-Layer Context Model + AnalysisContextBuilder + 운영 CronJob 2~3종** 으로 해결, (e) 를 **카드뉴스 v2 schema (`summary_lines` + `implication.skax_implication.recommended_actions` 명시화)** 로 마무리. **핵심: 신규 DB 테이블은 거의 불필요 (기존 `card_news` / `raw_article_business_signals` / `raw_article_financial_metrics` / `peer_companies.peer_plus_payload` JSONB / `legacy_records` 로 모두 흡수 가능). 추가는 VIEW 2 + MATERIALIZED VIEW 1 + 인덱스 3개 + V34 (옵션) `event_chain_links` 1 테이블.**

**v3.1 핵심 변경 (실측 기반)**: §2.4 에 8 critical issue 추가 (cluster_id 의미 불일치 / source_raw_article_ids 누락 15% / metric_name 정규화 누락 / matched_companies 표기 혼재 등). §3.4 의 ContextBuilder query 를 **`source_raw_article_ids[]` 기반 join + peer_id alias 정규화 + metric_name canonicalization + chunked LLM input** 으로 재설계. W4 단계 앞에 **W4-0 (data hygiene precondition)** 신설.

---

## 0.1 시스템 2-Layer Architecture

이 계획서가 다루는 범위와 책임 경계를 먼저 명확히 한다.

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Layer A — Data Pipeline   (이 계획서 범위 밖, axis-cron-ingestion-*) │
│   crawl → dedup → classify → 정제·매칭                              │
│                          ↓                                          │
│            DB 저장 (raw_articles + raw_article_*)                   │
│   책임: 원천 데이터를 클러스터 단위로 정제·매칭하여 DB 에 적재.     │
│         분석·시사점 추론은 일체 하지 않음.                          │
└─────────────────────────────────────────────────────────────────────┘
                          ↓ (cluster trigger)
┌─────────────────────────────────────────────────────────────────────┐
│ Layer B — Analysis Supervisor Graph  (이 계획서의 작업 범위)        │
│   ① context assemble  ProfileContext + AnalysisContext (DB+Qdrant)  │
│        ↓                                                            │
│   ② LLM reasoning     IssueIntegration → StrategicAnalysis →        │
│                       Implication                                   │
│        ↓                                                            │
│   ③ validation        Evidence chain / numeric / 단정 표현 검사     │
│        ↓                                                            │
│   ④ package generation AnalysisPackage → CardNewsAgent →            │
│                        card_news WRITE (v2)                         │
│   책임: Layer A 가 적재한 데이터를 받아 cluster 단위 시사점·대응을  │
│         추론하고 카드뉴스 v2 로 직렬화.                             │
└─────────────────────────────────────────────────────────────────────┘
                          ↑ DB read (cluster-time)
                  ┌─────────────────────────┐
                  │ Context Layer CronJobs  │
                  │  주1회 ProfileSnapshot  │
                  │  주1회 SectorPulse MV   │
                  │  월1회 CapabilityEvol.  │
                  │  매일  EventChain(opt)  │
                  └─────────────────────────┘
```

> **As-Is 의 design debt (v3.1.3 에서 해소 예정)**: 현재 코드에서는 카드 생성 (`CardNewsAgent.write_card`) 이 `ingestion_graph.card_news_node` (Layer A 그래프) 안에서 호출됨. 카드뉴스는 분석/시사점/대응의 직렬화 결과 (= Layer B 의 산출물) 이므로 의미상 Layer 위반. **W2-1 작업 5 에서 이관 — `card_writer` 노드로 supervisor 의 마지막에 통합 (본 계획서 범위 내, W2-1 시간 +2~3h)**. 본 계획서가 "카드 dedup 강화 / source 강제 검증 / FK 회귀 추적" 을 "Layer B 카드 생성 단" 으로 분류하는 이유.
>
> **용어 정정 (v3.1.2)**: 본 문서에서 더 이상 "ingestion-side" 라는 표현으로 카드 생성 후속 작업을 가리키지 않는다. ingestion 은 Layer A (수집·전처리·DB 적재) 만 의미. 카드 생성 단 / 후처리 단 / Layer B 카드 단 등을 사용한다.

---

## 1. 현재 진행 현황

### 1.1 컴포넌트 매트릭스

| # | 컴포넌트 | 파일 | LoC | LLM | DB | 상태 |
|---|---|---|---|---|---|---|
| 1 | `DataAnalysisSupervisorAgent` | `agents/analysis_supervisor_agent.py` | 241 | ❌ 조율 | via children | 🟢 wire / 🔴 retry 없음 |
| 2 | `IssueIntegrationAgent` | `agents/issue_integration_agent.py` | 610 | via summarizer | `raw_articles` read | 🟢 |
| 2a | `SourceSummarizer` (engine) | `analysis/summarizer.py` | 2,630 | ✅ 2 invoke | `raw_articles` | 🟢 `summary-v4.0` |
| 3 | `AnalysisAgent` (= `StrategicAnalyzer`) | `agents/analysis_agent.py` + `analysis/analyzer.py` | 343 | ✅ 1 invoke | in-memory | 🟢 `analysis-v3.0` |
| 4 | `ProfileAgent` | `agents/profile_agent.py` | 4,056 | ✅ 7+ spot | `peer_companies`, `raw_article_*` | 🟢 / 🟡 cluster-time 빈약 |
| 4a | `SKAXProfileLoader` | `services/skax_profile_context_loader.py` | 1,587 | ✅ | `raw_articles` (sk_ax_site) | 🟢 |
| 5 | `ImplicationAgent` (현재) | `agents/implication_agent.py` + `analysis/implication.py` | 167 | ❌ heuristic | in-memory | 🔴 **LLM 미사용** |
| 5a | (legacy) LLM 시사점 | `agents/_deprecated/implication_agent.py` | 136 | ✅ | in-memory | 🟠 입력 계약 다름, 부분 참고만 |
| 6 | `CardNewsAgent` | `agents/card_news_agent.py` | 1,409 | ✅ 5 invoke | `card_news` WRITE | 🟢 / 🟡 implication 이중 처리 |
| 7 | `AnalysisPipelineRunner` | `pipeline/analysis_pipeline.py` | 479 | — | `raw_article_*` read | 🟢 3 진입점 |
| 8 | `ingestion_graph.card_news_node` | `pipeline/ingestion_graph.py` | (전체 322) | via runner | `card_news` WRITE | 🟢 `_GPT_WORKERS=5` 병렬 |
| 9 | `AnalysisContextBuilder` (W4 신규) | `services/analysis_context_builder.py` | (예상 ~350) | ❌ DB + Qdrant only | timeline / capability / sector_pulse / financial / RAG read | ⚪ 계획 |
| 10 | `CapabilityEvolutionAgent` (W4 신규) | `agents/context/capability_evolution_agent.py` | (예상 ~250) | ✅ 월1회 | `raw_article_business_signals` read, `peer_companies.peer_plus_payload` write | ⚪ 계획 |
| 11 | `SectorPulseAggregator` (W4 신규) | `agents/context/sector_pulse_aggregator.py` | (예상 ~50) | ❌ MATERIALIZED VIEW REFRESH | `sector_pulse` MV refresh | ⚪ 계획 |
| 12 | `EventChainDiscoveryAgent` (W4-6 옵션) | `agents/context/event_chain_discovery_agent.py` | (예상 ~300) | ✅ 매일 | `card_news` read, `evidence_payload['related_card_ids']` write | ⚪ 보류 |
| 13 | `EvaluatorAgent` (W5-1 신규) | `agents/evaluator_agent.py` + `validate` 노드 확장 | (예상 ~250) | ❌ rule-based | in-memory (validate 결과) + `card_news.evaluation_payload['rule_based']` write | ⚪ 계획 |
| 14 | `CardEvaluatorSidecar` (W5-2 신규) | `scripts/evaluate_recent_cards.py` (CronJob, 5분 주기) | (예상 ~350) | ✅ gpt-4o-mini | `card_news` read (미평가 카드), `card_news.evaluation_payload['llm_judge']` write | ⚪ 계획 |

### 1.2 cluster postgres 실측 (2026-05-20, v3.1 갱신)

| 지표 | 값 | 시뮬레이션에서 발견된 문제 |
|---|---|---|
| `raw_articles` | 15,118 행 (cluster_id distinct 671, max 39,337) | matched_companies 가 string array (객체 X), peer_id 표기 혼재 |
| `card_news` | 191 행 (peer 매칭 131, 60건 peer NULL) | cluster_id distinct **21개만 (max 90,001)** — ephemeral seq |
| `raw_article_business_signals` | 27,377 행 (peer 별 3,715~8,235) | `period_quarter` NULL **7.4%** (2,021건) |
| `raw_article_financial_metrics` | 3,449 행 (peer 별 218~1,006) | `period_quarter` NULL **17.8%** (615건), metric_name 정규화 누락 |
| `peer_companies` | 5 행 (samsung_sds, lg_cns, sk_ax, posco_dx, hyundai_autoever) | `peer_plus_payload`, `financial_history`, `job_posting_history` **모두 비어있음** |
| `legacy_records` | 95,321 행 | V30 archive — `source_table='peer_companies'` snapshot 저장소로 활용 가능 |
| Qdrant `axis_main` | 162 points | hot window 90일 — 충분 |
| Stage 1 GPT-4o 비용 | ≈ **$1.17/일** (15 clusters × $0.078) | — |

→ data flow + LLM + DB + Qdrant end-to-end 동작. 단 `card_news` 의 실 운영 시작이 **2026-05-14** 이후이므로 카드 기반 시계열 (sector_pulse, event_chain) 은 **2주치 baseline 만 존재** — 4주 이상 누적 시 의미 가짐.

---

## 2. 미흡한 점 — 종합 진단

### 🔴 P0 — production 가치에 직접 영향

| ID | 약점 | 영향 |
|---|---|---|
| **P0-1** | `ImplicationAgent` 가 heuristic only (LLM 미사용) | 카드 183건의 SK AX 시사점이 generic 템플릿. 서비스 핵심 가치 손실 |
| **P0-2** | `ProfileAgent` 가 cluster-time 에 lightweight context (static dict) 만 반환 | peer 별 차별화 input 부재 → 시사점 깊이 한계 |
| **P0-3** | Supervisor 가 LLM 한 단계 실패 시 cluster 전체 카드 손실 | 시간당 카드 손실, rate limit / OpenAI 5xx 빈도에 직접 노출 |

### 🟠 P1 — 구조적 약점

| ID | 약점 | 영향 |
|---|---|---|
| **P1-1** | Supervisor 가 LangGraph state machine 이 아니라 plain Python 순차 호출 | 노드별 elapsed_ms · token 추적 불가, retry/fallback 라우팅 어려움 |
| **P1-2** | `AnalysisResult` / `ImplicationResult` dataclass ↔ 실 LLM 출력 키 mismatch (schema drift) | 타입 안전 무력, 신규 팀원 혼란 |
| **P1-3** | Supervisor 레벨 quality gate 부재 (출처 수치 검증, 단정 표현 검출, evidence_chain 무결성) | hallucination 카드가 production 으로 빠져나감 |
| **P1-4** | CardNewsAgent 가 implication 을 2번 처리 (analysis 기반 + ImplicationAgent 결과) | 같은 의미 중복 LLM 가능, 출력 일관성 약함 |

### 🟡 P2 — 품질 / 운영성

| ID | 약점 | 영향 |
|---|---|---|
| **P2-1** | `design/30-analysis/` 에 supervisor 자체 design doc 부재 | 신규 합류자가 옛 도큐 참고하다 혼란 |
| **P2-2** | Supervisor / IssueIntegration / Analysis / Profile / Implication 단위 테스트 0건 | 프롬프트 회귀 잡기 어려움 |
| **P2-3** | Langfuse tracing 일부 누락 (Analyzer 외) | cluster 단위 trace 불완전 |
| **P2-4** | `_deprecated/implication_agent.py` 잔존 — 동일 클래스명 충돌 위험 | import 사고 가능 |
| **P2-5** | `AnalysisInputBundle.metadata` 가 free-form `dict[str, Any]` | 누락 시 KeyError, 계약 없음 |

### 🔵 P3 — Context Engineering (W4 신설)

| ID | 약점 | 영향 |
|---|---|---|
| **P3-1** | cluster-time 에 **과거 누적 데이터 (뉴스 14k / signals 27k / metrics 3.4k / 카드 183)** 가 시사점 추론 input 으로 전혀 활용되지 않음 | "이 이슈가 작년 X 발표의 연속선상" 같은 종방향 추론 불가, 시사점이 항상 단발성 |
| **P3-2** | peer 별 capability 변화의 시계열 narrative 가 어디에도 저장 안 됨 (raw signals 만 있고 LLM 합성 narrative 없음) | "삼성SDS 가 최근 6개월 클라우드 → AI 전환 가속" 같은 추세 진술 매번 LLM 재합성 비용 |
| **P3-3** | 섹터 단위 주간 흐름 (sector pulse) 집계가 없음 | "이번 주 SI 섹터 인사 이벤트 5건, 평균보다 +250%" 같은 비교 진술 불가 |
| **P3-4** | 카드 간 인과/연쇄 관계 (event chain) 가 끊겨 있음 — 같은 cluster_id 만 묶임 | "이 사건은 3월 25일 발표의 후속 액션" 같은 시퀀스 추론 불가 |
| **P3-5** | 현재 카드 schema 가 `요약 중심` (`summary_lines` 만 강조). `시사점`·`대응` 이 `implication` JSONB 안에 묻혀 있어 frontend 가 일관되게 3섹션 분리 불가 | 사용자가 카드를 봐도 "그래서 SK AX 가 뭘 해야 하는지" 가 한눈에 안 보임 |

→ P3 는 P0~P2 와 독립적으로 진행 가능 (W4 단계). 단 **W1-1 의 `ImplicationResult` 가 W4 의 풍부한 `AnalysisContext` 를 입력으로 받으면 prompt 가 `implication-v4.0` → `implication-v5.0` 으로 자연스럽게 진화** (W4 완료 후 prompt-only 변경).

### 2.4 v3.1 — 실측 데이터 시뮬레이션 발견 issue (W4-0 선결 조건)

> `kubectl exec postgres-...` 으로 cluster DB 직접 쿼리해서 v3.0 계획서의 W4 design 가정이 실 데이터와 맞는지 점검. **8개 critical issue 발견** — W4 본 작업 전에 선결 또는 query 재설계 필요.

#### 🔴 P3-CRIT — Data identity / integrity (W4-0 선결)

| ID | 발견 (실측) | W4 영향 | 대응 |
|---|---|---|---|
| **P3-CRIT-1** | `card_news.cluster_id` 가 stable cluster identifier 가 아닌 **ephemeral sequence**. distinct 21개, max 90,001. 같은 cluster_id=0 에 38건 카드, cluster_id=1 에 27건. raw_articles 의 cluster_id (distinct 671, max 39,337) 와 의미 다름. | event_chain join 불가, AnalysisContext 의 `cluster_id` 기반 precedent 조회 무효 | **`source_raw_article_ids[]` 또는 `(peer_company_id, event_type, DATE(created_at))` 복합키로 join key 재정의** |
| **P3-CRIT-2** | `card_news.source_raw_article_ids` 가 **15% 카드 (28건) 에서 빈 array**. 1개 source 161건, 2개 1건, 3개 1건 → 사실상 단일 source 도미넌트. | raw_articles 까지 provenance chain 끊김, evidence_chain 무결성 위반 | **W4-2 의 ContextBuilder query 가 source_raw_article_ids 빈 경우 graceful fallback (sources JSONB 활용) + 카드 생성 단계에서 source 강제 검증 (별도 P0 작업)** |
| **P3-CRIT-3** | **같은 (cluster_id, date, peer) 조합에 카드 4-6건** 다수 — 예: cluster_id=0, 2026-05-14, samsung_sds → 6 카드. 같은 이슈 (오픈AI 파트너십, AI 컴퓨팅센터) 가 여러 카드로 분리 생성. | event_chain candidate 가 "precedent" 가 아닌 "duplicate" 가 됨, ImplicationAgent 의 `precedent_link` 가 같은 이슈 자기참조 | **W4-6 (EventChainDiscovery) 도입 보류 + 카드 dedup 강화 (Layer B 카드 생성 단 — 현재 `ingestion_graph.card_news_node`, To-Be `CardNewsAgent`) 가 더 시급**. ingestion (Layer A) 책임이 아님 — As-Is 코드 위치만 ingestion_graph 안일 뿐 의미상 Supervisor 영역. |
| **P3-CRIT-4** | **이번 주 (5/18 이후) 신규 카드 36건 전부 `peer_company_id` FK NULL + 5/20 부터 `primary_keyword_category` 도 NULL (11건)**. 단 `card_news.company` 는 100% 채워짐 (`lg_cns`/`samsung_sds`/`hyundai_autoever` — 모두 `peer_companies.id` 매칭), `raw_articles.matched_companies` / `matched_sectors` 도 100% 정상. **즉 ingestion 식별은 정상, 카드 후처리 단의 FK 연결 + sector 분류만 회귀**. 회귀 시점 2개: ① 5/15 부분 → 5/16 완전 (FK), ② 5/20 신규 (sector) | sector_pulse `peer_event_count=0`, peer-level context 빈약 — 단 **데이터 손실 X 이므로 백필 1쿼리로 즉시 복구 가능** | **W4-0 첫 마이그레이션에서 즉시 백필**: `UPDATE card_news SET peer_company_id = company WHERE peer_company_id IS NULL AND company IN (SELECT id FROM peer_companies)` + `UPDATE card_news SET primary_keyword_category = (raw_articles.matched_sectors->>0 …)`. **회귀 원인 추적은 axis-ai `CardNewsAgent` (또는 후처리) 의 5/14~5/20 변경 이력 — 별도 P1 ticket** (ingestion 회귀 아님). 백필 후 `general_event_timeline` fallback 불필요. |

#### 🟠 P3-DATA — Data schema / sparsity (W4 query 재설계)

| ID | 발견 (실측) | W4 영향 | 대응 |
|---|---|---|---|
| **P3-DATA-1** | `business_signals.period_quarter` NULL **7.4%** (2,021건), `financial_metrics.period_quarter` NULL **17.8%** (615건) | (peer, business_area, year, quarter) 그룹화 시 다수 raw 행이 누락 | **`COALESCE(period_quarter::text, 'unknown')` fallback bucket + `period` 텍스트 컬럼을 보조 키로 활용** |
| **P3-DATA-2** | `financial_metrics.metric_name` 정규화 누락 — 동일 metric 이 `net_income`(22) / `당기순이익`(22) / `순이익`(17) 으로 분산. | financial_trend 시계열 합산 시 데이터가 분산 (sum 이 부정확) | **W4-1 의 `peer_financial_trend` VIEW 에 `metric_name_canonical` CASE 매핑 추가** (§3.4.3 갱신) |
| **P3-DATA-3** | `raw_articles.matched_companies` 가 **string array (객체 X)** — `["samsung_sds"]` 형식. peer_id 표기 혼재: 정규ID (`samsung_sds`), 한글 (`삼성SDS`), 영문 (`Microsoft`), SK 계열사 (`SK텔레콤`) | `_load_recent_business_signals(peer_id=...)` 등 query 가 단일 ID 만 매칭하면 한글/영문 row 누락 | **`PEER_ID_ALIASES` dict + `peer_id IN (alias_list)` 쿼리 표준 패턴 도입** (§3.4.5 갱신) |
| **P3-DATA-4** | DART/IR row (source_type='dart' 100건, 'ir' 67건) 의 `matched_companies` **빈배열** — financial_metrics 의 peer_id 는 있어도 raw_articles 와 join 불가 | provenance chain 의 raw_article 까지 추적 단계에서 끊김 | **W4-2 의 provenance chain query 가 `raw_article_financial_metrics.raw_article_id` 직접 join (matched_companies 경유 X)** |
| **P3-DATA-5** | **카드 실 운영 시작 2026-05-14** → sector_pulse / event_chain 의 baseline 이 **2주치만**. 정상적 z-score / anomaly 측정은 4주 이상 누적 필요. | sector_pulse 의 anomaly trigger 조건이 실 데이터에서 의미 없음 | **`sector_pulse` MATERIALIZED VIEW 의 anomaly z-score 컬럼은 4주 baseline 충족 후 활성화** (W4-4 의 두 phase 분리) |
| **P3-DATA-6** | LG CNS 의 financial_metrics 만 218건 (다른 peer 의 1/3 수준), business_signals 도 3,900건으로 최소 | LG CNS context layer 빈약, evidence_label="insufficient" 빈도 ↑ | **AnalysisContext 의 `evidence_density_per_peer` 계산 후 ImplicationAgent prompt 에 명시 → peer-level confidence 차등화** |
| **P3-DATA-7** | peer 당 4분기 business_signals raw 텍스트 합 ≈ **170k tokens** (peer 당 ~2,000 signal × avg 258 chars). gpt-4o 128k window 초과 위험. | CapabilityEvolutionAgent (W4-3) 의 단순 prompt assembly 가 hard fail | **SQL 단계에서 (peer × business_area × period × signal_type) GROUP BY top-N (confidence DESC LIMIT 5) 후 LLM 전달 — chunked input** (§3.4.6 갱신) |
| **P3-DATA-8** | 191 카드 중 **31% (60건) 가 `peer_company_id` NULL** — 일반 sector 뉴스 다수. | sector_pulse 의 `active_peers` 가 빈 array, ImplicationAgent 의 peer 별 enrichment 불가 | **sector_pulse VIEW 의 `active_peers` 가 빈 배열 시 `general` sector category 로 라벨링 + ImplicationAgent 가 peer-agnostic mode 로 전환** |

#### 🟡 P3-LOG — 추론 정당성 / 가치 평가 (계획서 보강)

| ID | 발견 | 대응 |
|---|---|---|
| **P3-LOG-1** | 현재 `card_news.implication` JSONB 가 heuristic 메타데이터만 (`sector`, `signals`, `exposure_band`). 자연어 시사점/대응 0건. 191 카드 모두 v1 | W4 완료 후 **191 카드 v2 backfill** 결정 (§3.5.4 옵션 B, $4.2 + 4h) 권장 |
| **P3-LOG-2** | 기존 `evidence_payload.financial_refs.narrative` 가 이미 양질의 텍스트 (예: "2025Q4 매출 3.54조원 (QoQ +4.3%) / YoY -2.9%") | **cluster-time financial_trend 재계산 불필요** — `card_news` 의 기존 `evidence_payload` JSONB 를 그대로 재사용 (W4-2 의 financial layer query 비용 ↓) |
| **P3-LOG-3** | event_chain candidates 가 1일 이내 매칭 (실 데이터 2주만, peer+sector 공유로 자동 매칭) → "precedent" 보다 "duplicate" 가 더 가능성 큼 | **W4-6 도입 보류 + 카드 dedup 강화 (Layer B 카드 생성 단 — As-Is `ingestion_graph.card_news_node`) 가 우선** |
| **P3-LOG-4** | 단순 keyword overlap (3 keys: peer+sector+event_type) 만으로는 precedent 식별 약함 — 모든 동일 peer 의 동 event_type 카드가 자동 매칭 | **W4-6 도입 시 Qdrant embedding 유사도 ≥ 0.75 + 시간차 ≥ 7일 조건 명시 (단순 keyword overlap X)** |

---

## 3. 제안 아키텍처 (수정안)

기존 design (`design/00-supervisor-topology.md`) 의 컴포넌트는 유지하되 **6가지 구조적 수정**:

1. **Supervisor 를 LangGraph StateGraph 로 재구성** — `ingestion_graph` 와 동일한 패턴 (state-based, 노드별 logging, retry decorator).
2. **ProfileContext 를 2-tier 분리** — Static snapshot (CronJob 주1회 LLM 합성) + Recent enrichment (cluster-time DB query). cluster-time LLM 호출 추가 없음.
3. **Validate 노드 신설** — output guardrail 분리. Evidence Chain 무결성 / 출처 수치 / 단정 표현 자동 검사.
4. **Context layer 신설 (W4)** — 4-Layer Context Model (Raw / Derived Timeseries / Retrieval Index / Active Context) + `AnalysisContextBuilder` (cluster-time, LLM X) + 운영 CronJob 2~3종. **신규 DB 테이블은 사실상 0개** — 기존 JSONB + VIEW + MATERIALIZED VIEW + `legacy_records` 로 모두 흡수.
5. **카드뉴스 v2 schema (요약+시사점+대응 3섹션)** — `card_news` 컬럼 추가 없이 `implication` JSONB key namespace 만 V34 에서 정합화. frontend 는 A(`summary_lines`) / B(`implication.skax_implication.why_important+potential_impact`) / C(`implication.skax_implication.recommended_actions`) 로 3섹션 분리 표시.
6. **Evaluation & Observability Layer 신설 (W5)** — **Hybrid 배치**: rule-based 5 metric 은 `validate` 노드 확장 (in-graph, LLM X, latency 0), LLM-as-Judge 4 score 는 별도 sidecar CronJob (`axis-cron-card-evaluator`, 5분 주기, gpt-4o-mini). 두 결과 모두 `card_news.evaluation_payload` JSONB 에 누적 → regression detection 자동화 가능.

### 3.1 LangGraph Supervisor StateGraph

```text
                  ┌────────────────────────────────────────────────┐
                  │  SupervisorState                                │
                  │  - input_bundle: AnalysisInputBundle            │
                  │  - profile_context: ProfileContext              │
                  │  - analysis_context: AnalysisContext   ← W4    │
                  │  - integrated_issue: IntegratedIssue            │
                  │  - analysis: AnalysisResult                     │
                  │  - implication: ImplicationResult               │
                  │  - validation: ValidationReport                 │
                  │  - errors: list[NodeError]                      │
                  │  - human_review_flags: list[str]                │
                  └────────────────────────────────────────────────┘

   [build_input_bundle]           # supervisor 외부 (analysis_pipeline / analysis_delivery)
          ↓
   [issue_integrate]              # ① IssueIntegrationAgent (LLM) — main_company 확정
          ↓                       #   (외부 리뷰 R-1 2026-05-21: 흐름 앞으로 이동)
   [profile_context]              # ② build_profile_context_v2 — main_company 기준 snapshot+enrichment, NO LLM
          ↓
   [build_analysis_context]       # ③ AnalysisContextBuilder — NO LLM (DB+Qdrant), main_company 기준 4-Layer
          ↓
   [strategic_analyze]            # ④ AnalysisAgent (LLM, peer 관점 only)
          ↓
   [implication]                  # ⑤ ImplicationAgent v4.0/v5.0 (LLM, SK AX 관점) ← P0-1 / P3-1 fix
          ↓
   [validate]                     # ⑥ 출처 수치 / 단정 표현 / evidence 무결성 + W5-1 metric ← P1-3 fix
       ↓             ↓
   pass=true      pass=false
       ↓             ↓
   [assemble]    [human_review]   # human_review_flags 추가 후 종료
       ↓             ↓
   [card_writer]   (END)          # ⑧ CardNewsAgent.write_card 호출 + save_card_news v2 INSERT (R-2)
       ↓
     (END) → card_news WRITE (v2 schema: peer_company_id / primary_keyword_category /
                                source_raw_article_ids / evidence_payload /
                                card_schema_version='v2' / evaluation_payload['rule_based'])
```

→ **카드뉴스는 분석/시사점/대응의 직렬화 결과** — 즉 Supervisor 의 최종 산출물이며, ingestion (Layer A) 단에서 만들지 않는다. As-Is 의 `ingestion_graph.card_news_node` 는 W2-1 작업 5 에서 제거되어 `card_writer` 노드로 통합된다.

**Retry 정책** — **목표 (별도 PR), 현재 미적용** (외부 리뷰 R-7 명시):

| 노드 | 목표 max_attempts | 목표 backoff | 실패 시 | **현재 상태** |
|---|---|---|---|---|
| `issue_integrate` | 2 | exponential 1s, 2s | `is_valid_summary=false` → 카드 skip | 미적용. `_logged_step` 의 try/except 가 예외를 `state.errors[]` 에 누적 |
| `profile_context` | 1 | — | static-only fallback | 미적용. v2→legacy `build_context` fallback 만 |
| `build_analysis_context` | 1 | — | empty AnalysisContext fallback (no LLM 이므로 retry 불필요) | DB query try/except 로 layer 별 graceful 빈 결과 |
| `strategic_analyze` | 2 | exponential 1s, 2s | `is_valid_analysis=false` → implication skip | 미적용. 빈 결과 반환 후 다음 노드의 valid 검사 |
| `implication` | 2 | exponential 1s, 2s | heuristic fallback (현 `ImplicationGenerator`) | **이미 적용** — ImplicationAgent 내부 try/except 가 LLM 실패를 잡아 heuristic 으로 fallback |
| `validate` | 1 | — | hard fail (skip 카드) | 적용 |
| `card_writer` | 2 | exponential 1s, 2s | DB transient 실패 시 retry, 2회 실패 시 errors 기록 후 hard fail | 미적용. save_card_news 의 try/except 가 한 번 잡음 |

→ **목표**: LangGraph `node.with_retry(RetryPolicy(...))` 로 위 정책을 정식 적용. **별도 PR (W3-4 trace + retry 묶음)** 에서 도입 예정.
→ **현재 부분 실패 흡수**: implication LLM 만 실패해도 analysis 단계까지의 결과는 보존됨 (heuristic fallback). 다른 노드는 retry 없이 한 번만 시도하고 실패 시 다음 노드의 valid 검사로 라우팅됨. `card_writer` 만 실패하면 `AnalysisPackage` 는 메모리에 남아 호출자가 후속 재시도 가능.

### 3.2 ProfileContext 2-tier 분리

```text
┌──────────────────────────────────────────────────────────────┐
│ Tier A — Static Profile Snapshot (주 1회 LLM 합성)            │
│ • CronJob: axis-cron-profile-refresh (월요일 03:00 KST)       │
│ • 입력: DART + IR + 공식 newsroom + 최근 6개월 뉴스           │
│ • 출력: peer_companies.profile_snapshot JSONB (V33 Flyway)    │
│ • TTL: 7일 + manual refresh CLI                              │
│ • LLM: gpt-4o, 회사당 ~$0.15 (5 회사 × 주 1회 = $0.75/주)     │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Tier B — Recent Signal Enrichment (cluster-time, DB only)    │
│ • profile_agent.build_context_v2()                            │
│ • DB query 만 (LLM 호출 없음, < 50ms)                          │
│ • 최근 30일 business_signals top-3 (peer 당)                  │
│ • 최근 분기 financial_metrics (DART)                          │
│ • sector_id 별 키워드 가중치 (sectors.py static)              │
└──────────────────────────────────────────────────────────────┘
                          ↓
                 ProfileContext 합성 (dataclass)
                          ↓
                 ImplicationAgent input
```

→ cluster-time LLM 추가 호출 0건, ImplicationAgent input 풍부도 ↑.

### 3.3 새 ImplicationAgent (`implication-v4.0` → W4 이후 `implication-v5.0`)

#### 입력 (W1 시점 4종, W4 이후 5종)
- `AnalysisInputBundle` (bundle_id, source_type, companies, fact 추출 결과)
- `IntegratedIssue` (consolidated_facts, key_numbers, business_signals, fact_basis)
- `AnalysisResult` (strategic_meaning, market_signal, impact_level, risk_or_opportunity)
- `ProfileContext` (skax_profile, peer_profiles[peer_id], sector_context)
- **`AnalysisContext` (W4 신설)** — 과거 누적 맥락 6 layer (§3.4 참고): `peer_event_timeline_recent` / `capability_evolution` / `sector_pulse_recent` / `financial_trend` / `event_chain_candidates` / `similar_cards_rag`. **token budget ≤ 4,000** 으로 압축됨.

#### 프롬프트 설계 (P.C.R.O 프레임워크)
- **P**ersona: SK AX 전략기획팀의 시사점 분석 AI
- **C**ontext: 5종 input + SK AX 핵심 사업 (에이전틱AI / 제조AX / MSP)
- **R**ule:
  - 출처에 없는 수치 (금액·%·날짜·인원수) 생성 금지
  - "반드시", "확실히", "분명히" 등 단정 표현 사용 금지
  - `confidence < 0.6` 일 때 `evidence_label="insufficient"` 자동
  - opportunities/threats/actions 는 SK AX `business_lines` 중 매칭되는 사업군에만 작성
  - peer 별 차별화: peer_profiles[peer_id].recent_capability_change 가 있으면 명시 인용
  - **W4 이후 (v5.0)**: `AnalysisContext.event_chain_candidates` 에 선행 사건이 있으면 `peer_implication.precedent_link` 에 `card_id` 명시; `capability_evolution` 에 narrative 가 있으면 `capability_change` 에 인용; `sector_pulse_recent` 의 anomaly (z-score ≥ 1.5) 가 있으면 `skax_implication.why_important` 에 반영
  - 모든 시계열 진술은 `valid_from`/`source_card_ids` 등 provenance 인용 필수
- **O**utput: JSON only

#### 출력 schema
```python
@dataclass(slots=True)
class ImplicationResult:
    is_valid_implication: bool
    implication_scope: Literal["peer_and_skax"]
    peer_implication: PeerImplication      # 아래 정의
    skax_implication: SkaxImplication      # 아래 정의
    follow_up_questions: list[str]         # 정확히 3개
    watch_points: list[str]                # 최대 3개
    confidence: float                      # 0.0~1.0
    evidence_label: Literal["sufficient", "moderate", "insufficient"]
    provenance: ImplicationProvenance

@dataclass(slots=True)
class PeerImplication:
    company_id: str
    company_name_ko: str
    peer_meaning: str                      # peer 관점 1-2문장
    capability_change: str | None          # snapshot vs 현재 이슈의 capability gap
    precedent_link: PrecedentLink | None   # W4 이후 — event_chain_candidates 의 inheritance
    sourced_evidence_ids: list[str]        # fact_basis 의 fact_id list

@dataclass(slots=True)
class PrecedentLink:                       # W4 신설
    card_id: str                            # 선행 카드의 card_news.id
    relation: Literal["follow_up", "reaction", "echo", "contradiction"]
    days_since: int
    rationale: str

@dataclass(slots=True)
class SkaxImplication:
    why_important: str                     # 왜 SK AX 에 중요한가
    potential_impact: str                  # SK AX 사업에 미칠 영향
    opportunities: list[str]               # 최대 3 (SK AX business_lines 와 매칭)
    threats: list[str]                     # 최대 3
    recommended_actions: list[str]         # 최대 3 (verb-first)
    business_line_mapping: list[str]       # ["에이전틱AI", "MSP"] 등

@dataclass(slots=True)
class ImplicationProvenance:
    generator: str = "ImplicationAgent"
    prompt_version: str = "implication-v4.0"
    model: str = "gpt-4o"
    bundle_id: str
    used_peer_profile_keys: list[str]
    used_skax_profile_keys: list[str]
    used_fact_ids: list[str]
    run_at: str
```

#### Post-validation 단계 (validate 노드에서)
- 출력 텍스트 안의 모든 숫자/% 를 regex 추출 (`\d+[.,]?\d*\s*[%원조억]?`).
- `IntegratedIssue.fact_basis[].evidence_text` 와 `key_numbers[].value` 모두 모은 set 에 포함되는지 검사.
- 미포함이면 → `validation.numeric_violations` 에 기록 + `evidence_label="insufficient"` 강제.
- 단정 표현 정규식 (`(반드시|확실(히|하)|분명(히|하)|틀림없)`) 매칭 시 → `certainty_warning=True` 플래그.
- 위반 1건이라도 있으면 `validation.pass=false` → 카드 skip + `human_review_flags` 에 추가.

---

### 3.4 W4 — 4-Layer Context Model + AnalysisContextBuilder

> **답해야 할 질문**: "DB 에 쌓인 모든 뉴스/DART/IR 을 어떻게 cluster-time 추론의 context 로 손실 없이 활용할 것인가?"
>
> **결론**: 4-Layer 구조로 raw → derived → retrievable → active context 의 명시적 분리. **신규 DB 테이블은 사실상 0개** — 기존 schema 로 90% 이상 흡수.

#### 3.4.1 4-Layer 모델 개관

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1 — Raw (immutable)                                           │
│   raw_articles (14k)  raw_article_business_signals (27k)            │
│   raw_article_financial_metrics (3.4k)  raw_article_parse_results   │
│   market_price_ohlcv  crawl_runs / crawl_run_articles               │
│   → 변경 불가. 모든 derived 의 provenance root.                       │
└─────────────────────────────────────────────────────────────────────┘
                          ↓ (집계·합성)
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 2 — Derived Timeseries (re-generatable)                       │
│   A. peer_event_timeline      ← VIEW (card_news 위에)                │
│   B. capability_evolution     ← peer_companies.peer_plus_payload     │
│                                   ['capability_evolution']  JSONB    │
│                                  (월1회 CronJob 갱신, LLM 합성)        │
│   C. sector_pulse             ← MATERIALIZED VIEW (card_news 집계)   │
│   D. financial_trend          ← VIEW (raw_article_financial_metrics  │
│                                   GROUP BY peer_id, metric, period)  │
│   E. event_chain_links        ← (옵션 V34) 신규 테이블 또는           │
│                                   card_news.evidence_payload         │
│                                   ['related_card_ids'] JSONB         │
│   F. profile_snapshot_history ← legacy_records 재사용 (V30 generic)  │
│   → valid_from / valid_to 또는 created_at 로 time-travel 가능        │
└─────────────────────────────────────────────────────────────────────┘
                          ↓ (인덱싱)
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 3 — Retrieval Index                                           │
│   • Qdrant axis_main          (162 points, 90일 hot window)          │
│   • Qdrant axis_history       (옵션 — 90일 초과 카드 cold storage)    │
│   • PostgreSQL GIN indexes    (card_news.keywords, .keyword_categories│
│                                  raw_articles.matched_companies)     │
│   • 새로 추가할 인덱스 3개 (§3.4.4)                                   │
└─────────────────────────────────────────────────────────────────────┘
                          ↓ (cluster-time 조합)
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 4 — Active Analysis Context                                   │
│   AnalysisContext dataclass — token budget ≤ 4,000                   │
│   ImplicationAgent prompt 의 추가 input.                              │
│   AnalysisContextBuilder 가 cluster-time 에 dynamic 합성 (LLM X).    │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 데이터 흐름과 책임

| Layer | 갱신 주체 | 갱신 주기 | LLM | 저장 위치 |
|---|---|---|---|---|
| Layer 1 Raw | `ingestion_graph` | 매시 crawl | (extract 시) | `raw_articles*`, `crawl_runs` |
| 2-A timeline | (자동) | card_news INSERT 와 동기 | ❌ | **`peer_event_timeline` VIEW** (신규) |
| 2-B capability | `CapabilityEvolutionAgent` | 월 1회 (1일 03:00) | ✅ gpt-4o | **`peer_companies.peer_plus_payload['capability_evolution']`** JSONB |
| 2-C sector pulse | `SectorPulseAggregator` | 주 1회 (월 02:00) | ❌ | **`sector_pulse` MATERIALIZED VIEW** (신규) |
| 2-D financial trend | (자동) | metric INSERT 시 | ❌ | **`peer_financial_trend` VIEW** (신규) |
| 2-E event chain (옵션) | `EventChainDiscoveryAgent` | 매일 02:00 (선택) | ✅ gpt-4o | `card_news.evidence_payload['related_card_ids']` JSONB <br/> 또는 V34 `event_chain_links` (table) |
| 2-F snapshot history | `axis-cron-profile-refresh` (W2-2) | 주 1회 | ✅ (W2-2) | **`legacy_records`** (`source_table='peer_companies'`) — 신규 테이블 X |
| Layer 3 인덱스 | Flyway / Qdrant | 정의 시 | ❌ | (DDL) |
| Layer 4 active context | `AnalysisContextBuilder` | 매 cluster (Supervisor 노드) | ❌ | **메모리 — `AnalysisContext` dataclass** |

→ **cluster-time LLM 추가 0건**. context 합성은 SQL + Qdrant retrieve 만.

#### 3.4.3 기존 스키마로 모두 가능한가 — 점검 결과

처음 제안했던 5개 신규 테이블을 기존 schema 와 1:1 매핑해 보면:

| 처음 제안 | 기존 데이터 충분도 | 최종 결정 |
|---|---|---|
| `peer_event_timeline` | `card_news.peer_company_id` + `cluster_id` + `created_at` + `event_type` + `importance_score` + `keyword_categories` + `source_raw_article_ids` ✓ **모두 있음** | **VIEW 만 추가** (테이블 X) |
| `peer_capability_evolution` | `raw_article_business_signals` 가 `peer_id` + `business_area` + `signal_type` + `period_year/quarter` + `summary` + `evidence_text` + `confidence` ✓ **27k 행 raw 충분**. 단 LLM-합성 narrative 만 별도 저장 필요 | **`peer_companies.peer_plus_payload['capability_evolution']` JSONB** (테이블 X) |
| `sector_pulse` | `card_news.primary_keyword_category` + `event_type` + `peer_company_id` + `importance_score` + `created_at` ✓ **집계 가능** | **MATERIALIZED VIEW** (테이블 X) |
| `event_chain_links` | 기존 어디에도 인과 link 정보 없음. **유일한 신규 정보** | **(옵션) V34 신규 테이블 또는 `card_news.evidence_payload['related_card_ids']` JSONB**. MVP 는 JSONB 로 시작 |
| `peer_profile_snapshot_history` | `legacy_records` 가 정확히 row-level archive 목적의 generic 테이블 (V30 부터 존재) | **`legacy_records` 재사용** (테이블 X) |

→ **결론: V33 마이그레이션은 (a) `peer_companies.peer_plus_payload` JSONB key namespace 명시화 (DDL 불필요, 문서·코드 수준) + (b) 인덱스 3개만 추가**.

```sql
-- axis-backend/src/main/resources/db/migration/V33__context_engineering_indexes.sql
-- (또는 axis-infra/db/schema.sql 통합)
-- v3.1: P3-CRIT-1/2 (cluster_id 의미 불일치) 와 P3-DATA-2 (metric_name 정규화) 반영

-- 2-A timeline: peer 별 시간순 카드 조회 hot path (cluster_id 제외)
CREATE INDEX IF NOT EXISTS idx_card_news_peer_created
    ON card_news(peer_company_id, created_at DESC)
    WHERE peer_company_id IS NOT NULL;

-- 2-B capability narrative source: peer × business_area × period 그룹화
CREATE INDEX IF NOT EXISTS idx_business_signals_peer_area_period
    ON raw_article_business_signals(peer_id, business_area, period_year DESC, period_quarter DESC NULLS LAST);

-- 2-D financial trend: peer × metric × period 시계열
CREATE INDEX IF NOT EXISTS idx_financial_metrics_peer_metric_period
    ON raw_article_financial_metrics(peer_id, metric_name, period_year DESC, period_quarter DESC NULLS LAST);

-- 2-A VIEW (P3-CRIT-1: cluster_id 컬럼 노출 하지 않음 — ephemeral 이므로 join key 로 사용 금지)
CREATE OR REPLACE VIEW peer_event_timeline AS
SELECT
    cn.peer_company_id        AS company_id,
    DATE(cn.created_at)       AS event_date,
    cn.id                     AS card_id,
    cn.event_type,
    cn.primary_keyword_category AS sector,
    cn.title                  AS headline,
    cn.summary_lines,
    cn.keyword_categories     AS capability_areas,
    cn.importance,
    cn.importance_score,
    cn.source_raw_article_ids AS evidence_article_ids,  -- P3-CRIT-2: 빈 array 가능
    cn.card_schema_version
FROM card_news cn
WHERE cn.peer_company_id IS NOT NULL
  AND cn.created_at >= NOW() - INTERVAL '180 days';

-- 2-A-supplement: general (peer 없음) timeline VIEW (P3-DATA-8)
CREATE OR REPLACE VIEW general_event_timeline AS
SELECT
    DATE(cn.created_at)       AS event_date,
    cn.id                     AS card_id,
    cn.event_type,
    cn.primary_keyword_category AS sector,
    cn.title                  AS headline,
    cn.summary_lines,
    cn.importance_score
FROM card_news cn
WHERE cn.peer_company_id IS NULL
  AND cn.primary_keyword_category IS NOT NULL
  AND cn.created_at >= NOW() - INTERVAL '180 days';

-- 2-C MATERIALIZED VIEW (주 1회 REFRESH) — P3-DATA-5/8 반영
-- Phase 1 (W4-4): event_count / intensity 만. z-score / anomaly 는 4주 baseline 후 활성화.
CREATE MATERIALIZED VIEW IF NOT EXISTS sector_pulse AS
SELECT
    COALESCE(cn.primary_keyword_category, 'general')                AS sector,
    DATE_TRUNC('week', cn.created_at)::DATE                         AS week_start,
    COUNT(*)                                                        AS event_count,
    COUNT(*) FILTER (WHERE cn.peer_company_id IS NOT NULL)          AS peer_event_count,
    COUNT(*) FILTER (WHERE cn.peer_company_id IS NULL)              AS general_event_count,
    AVG(cn.importance_score)                                        AS intensity_avg,
    jsonb_object_agg(
        DISTINCT cn.event_type,
        (SELECT COUNT(*) FROM card_news cn2
         WHERE cn2.event_type = cn.event_type
           AND DATE_TRUNC('week', cn2.created_at) = DATE_TRUNC('week', cn.created_at)
           AND COALESCE(cn2.primary_keyword_category, 'general') = COALESCE(cn.primary_keyword_category, 'general'))
    )                                                               AS event_type_distribution,
    array_remove(array_agg(DISTINCT cn.peer_company_id), NULL)      AS active_peers,
    (array_agg(cn.id ORDER BY cn.importance_score DESC NULLS LAST)
        FILTER (WHERE cn.importance_score >= 0.7))[1:5]            AS notable_card_ids
FROM card_news cn
WHERE cn.created_at >= NOW() - INTERVAL '180 days'
GROUP BY 1, 2;

CREATE UNIQUE INDEX IF NOT EXISTS uq_sector_pulse_sector_week
    ON sector_pulse(sector, week_start);

-- 2-D VIEW with metric_name canonicalization (P3-DATA-2)
CREATE OR REPLACE VIEW peer_financial_trend AS
SELECT
    rfm.peer_id                AS company_id,
    -- metric_name 정규화: 동일 metric 의 다른 표기 통합
    CASE
        WHEN rfm.metric_name IN ('net_income', '당기순이익', '순이익') THEN 'net_income'
        WHEN rfm.metric_name IN ('revenue_total', '매출', '매출액') THEN 'revenue_total'
        WHEN rfm.metric_name IN ('operating_profit', '영업이익') THEN 'operating_profit'
        WHEN rfm.metric_name IN ('operating_margin', '영업이익률') THEN 'operating_margin'
        WHEN rfm.metric_name IN ('gross_profit', '매출총이익') THEN 'gross_profit'
        ELSE rfm.metric_name
    END                        AS metric_name_canonical,
    rfm.metric_name            AS metric_name_raw,    -- 원본 보존
    rfm.metric_label,
    rfm.business_area,
    rfm.period_year,
    COALESCE(rfm.period_quarter::text, 'annual') AS period_quarter_safe,  -- P3-DATA-1: NULL fallback
    rfm.period_quarter,
    rfm.period,
    rfm.value_numeric,
    rfm.value_krwbn,
    rfm.unit,
    rfm.currency,
    rfm.confidence,
    rfm.raw_article_id        AS evidence_article_id
FROM raw_article_financial_metrics rfm
WHERE rfm.peer_id IS NOT NULL;
```

**`peer_companies` snapshot 저장 위치 표준** (v3.2.1 정정 — 외부 리뷰 R-5):

* **Tier A profile snapshot** → `peer_companies.profile_snapshot` JSONB **별도 컬럼**
  (`profile_snapshot_version` + `profile_snapshot_generated_at` 메타와 함께). V33 에서
  컬럼 신설. JSONB key 예시:
  ```jsonc
  {
    "version": "profile-v5",
    "generated_at": "2026-05-18T03:00:00+09:00",
    "narrative": "...",
    "core_capabilities": [...],
    "recent_keywords": [...]
  }
  ```

* **W4 derived data** (capability_evolution / snapshot_archive_ref) → `peer_companies.peer_plus_payload`
  JSONB key namespace 표준:
  ```jsonc
  {
    "capability_evolution": {            // W4 (Layer 2-B) — 월1회 갱신
      "version": "capability-v1",
      "generated_at": "2026-05-01T03:00:00+09:00",
      "windows": [
        {
          "period": "2025Q3-2026Q1",
          "business_area": "Cloud",
          "narrative": "MSP 매출 성장 가속, 인력 +15%",
          "evidence_signal_ids": ["...", "..."],
          "delta_intensity": 0.78,
          "confidence": 0.72
        }
      ]
    },
    "snapshot_archive_ref": {            // V30 legacy_records 참조 메타
      "last_archived_at": "2026-05-18T03:00:00+09:00",
      "archive_count": 3
    }
  }
  ```

→ **profile_snapshot 은 별도 컬럼 (`peer_companies.profile_snapshot`) 만 인정.**
이전 문서에서 `peer_plus_payload['profile_snapshot']` 표현이 혼재했지만 v3.2.1 부터는
**별도 컬럼 (인덱싱 가능 + version/generated_at 메타 분리)** 로 통일.

→ **V33 = 컬럼 5 (profile_snapshot/_version/_generated_at + card_schema_version + evaluation_payload)
   + 인덱스 5 + VIEW 2 + MATERIALIZED VIEW 1**. 신규 테이블 0.
→ (옵션) **V34 = `event_chain_links` 1 테이블** — MVP 보류, P3-4 측정값으로 결정.

#### 3.4.4 (옵션) V34 — `event_chain_links` (도입 보류)

만약 `card_news.evidence_payload['related_card_ids']` JSONB 만으로 부족하다고 W4 종료 후 판단되면:

```sql
-- V34__add_event_chain_links.sql (조건부 도입)
CREATE TABLE IF NOT EXISTS event_chain_links (
    id BIGSERIAL PRIMARY KEY,
    source_card_id VARCHAR(50) NOT NULL REFERENCES card_news(id) ON DELETE CASCADE,
    target_card_id VARCHAR(50) NOT NULL REFERENCES card_news(id) ON DELETE CASCADE,
    link_type VARCHAR(30) NOT NULL,  -- follow_up | reaction | echo | contradiction
    confidence NUMERIC(3,2) NOT NULL,
    rationale TEXT,
    discovered_by VARCHAR(40) NOT NULL DEFAULT 'EventChainDiscoveryAgent',
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_event_chain_links UNIQUE (source_card_id, target_card_id, link_type),
    CONSTRAINT chk_event_chain_links_confidence CHECK (confidence BETWEEN 0 AND 1),
    CONSTRAINT chk_event_chain_links_type CHECK (
        link_type IN ('follow_up', 'reaction', 'echo', 'contradiction')
    )
);
CREATE INDEX idx_event_chain_links_source ON event_chain_links(source_card_id);
CREATE INDEX idx_event_chain_links_target ON event_chain_links(target_card_id);
```

→ MVP 는 JSONB. 4주 운영 후 link 수 ≥ 200 / 주 면 별도 테이블로 승격.

#### 3.4.5 AnalysisContextBuilder — cluster-time 합성 (v3.1 강화)

**v3.1 추가 (시뮬레이션 기반)**:
- (P3-DATA-3) `PEER_ID_ALIASES` 정규화 — `matched_companies` 의 한글/영문/축약 ID 처리
- (P3-CRIT-1) cluster identity 재정의 — `cluster_id` 사용 금지, `source_raw_article_ids` 또는 `(peer, event_type, date)` 사용
- (P3-DATA-7) chunked SQL pre-aggregation — LLM 직접 전달 X
- (P3-LOG-2) `evidence_payload.financial_refs.narrative` 재사용 (cluster-time 재계산 X)
- (P3-DATA-6) `evidence_density_per_peer` 계산해서 ImplicationAgent confidence 조정 신호

```python
# src/services/analysis_context_builder.py (신규)

# (P3-DATA-3) peer_id 표기 정규화 — raw_articles.matched_companies 는 string array
PEER_ID_ALIASES: dict[str, list[str]] = {
    "samsung_sds":      ["samsung_sds", "삼성SDS", "삼성에스디에스", "samsung sds", "Samsung SDS"],
    "lg_cns":           ["lg_cns", "LG CNS", "LG cns", "엘지씨엔에스"],
    "sk_ax":            ["sk_ax", "SK AX", "SK ax", "에스케이에이엑스"],
    "posco_dx":         ["posco_dx", "포스코DX", "POSCO DX", "포스코디엑스"],
    "hyundai_autoever": ["hyundai_autoever", "현대오토에버", "Hyundai AutoEver"],
}

def _expand_peer_aliases(peer_id: str) -> list[str]:
    return PEER_ID_ALIASES.get(peer_id, [peer_id])

@dataclass(slots=True)
class AnalysisContext:
    peer_event_timeline_recent: list[TimelineEntry]   # 90일, peer 당 ≤ 5
    capability_evolution: dict[str, CapabilityWindow] # peer_id → 최신 1개 narrative
    sector_pulse_recent: list[SectorPulseRow]         # 직전 4주 sector 별
    financial_trend: dict[str, FinancialSeries]       # peer × top 3 metric, 8 분기
    event_chain_candidates: list[PrecedentCandidate]  # 후보 ≤ 5
    similar_cards_rag: list[RetrievedCard]            # Qdrant top-3 by cosine
    evidence_density_per_peer: dict[str, EvidenceDensity]  # P3-DATA-6 — peer 별 풍부도
    token_budget_used: int                             # ≤ 4,000
    provenance: ContextProvenance                      # 모든 source id

@dataclass(slots=True)
class EvidenceDensity:
    signal_count_4q: int       # 4분기 signal 개수
    metric_count_4q: int       # 4분기 metric 개수
    timeline_count_90d: int    # 90일 카드 개수
    density_label: Literal["rich", "moderate", "sparse"]  # → ImplicationAgent confidence 조정

class AnalysisContextBuilder:
    """LLM 호출 없음. DB query + Qdrant retrieve 만."""

    def __init__(self, db, qdrant, *, token_budget: int = 4000) -> None:
        self._db = db
        self._qdrant = qdrant
        self._budget = token_budget

    def build(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        profile_context: ProfileContext,
    ) -> AnalysisContext:
        peers = _peer_ids_from(input_bundle)         # 정규 ID 만 반환
        sectors = input_bundle.metadata.classification.sectors

        # P3-CRIT-1: cluster_id 가 아닌 (peer, event_type, date) 또는 source_raw_article_ids 로 join
        timeline = self._db.query_timeline(peers=peers, days=90, limit_per_peer=5)

        # P3-DATA-3: alias expand 해서 raw_articles.matched_companies 매칭
        capability = self._db.query_capability_evolution(peers=peers)  # peer_plus_payload JSONB read

        sector_pulse = self._db.query_sector_pulse(sectors=sectors, weeks=4)

        # P3-LOG-2: cluster-time 재계산 대신 기존 card_news.evidence_payload.financial_refs 우선 활용
        financial = self._db.query_financial_trend(peers=peers, quarters=8, top_metrics=3,
                                                    metric_name_canonical=True)  # P3-DATA-2

        # P3-CRIT-3: precedent 식별 시 (peer, event_type, date) 만으로는 약함 → embedding 거리 ≥ 0.25
        precedents = self._qdrant.find_precedents_by_embedding(
            bundle=input_bundle,
            min_days_since=7,        # P3-LOG-4: 7일 이내는 duplicate 가능성 — 제외
            min_cosine=0.75,
            top_k=5,
        )

        similar = self._qdrant.search_by_bundle(input_bundle, top_k=3)

        # P3-DATA-6: peer 별 density 측정 → ImplicationAgent prompt 에 sparsity 명시
        density = self._compute_evidence_density(peers, timeline, capability, financial)

        ctx = AnalysisContext(...)
        return _compress_to_budget(ctx, self._budget)
```

**시뮬레이션 기반 token budget 실측**:
- timeline row 1개 ≈ **150 token** (실측 564-780 bytes/row)
- peer 당 5 row × 5 peer = 25 row → **3,750 token** (budget 의 94%)
- → **timeline 을 peer 당 3 row 로 제한** (compression 1순위)
- capability_evolution narrative peer 당 **~200 token** (압축됨)
- sector_pulse row 4주 × 5 sector = 20 row × **80 token** = 1,600 token
- → 전체 합산 시 budget 초과. compression 필요.

**Compression 우선순위** (v3.1 갱신):
1. RAG 결과 → 2개로 축소 (-300 token)
2. financial trend → 4 분기로 축소 (-400 token)
3. sector pulse → 2주로 축소 (-800 token)
4. event_chain_candidates → 3건으로 축소 (-200 token)
5. timeline → peer 당 **3건** 으로 축소 (-1,500 token)
6. capability narrative → peer 당 **150자** truncate (-500 token)

**최악 케이스 (모든 데이터 풍부)**: 6,250 token → 압축 후 **3,800 token** (~95% utilization)
**최선 케이스 (sparse, LG CNS 1개 peer)**: 1,200 token

→ 평균 **2,200-3,500 token** 으로 안정. gpt-4o prompt 전체 (system + user + context) 가 25k token 이내 보장.

#### 3.4.6 운영 Agent — 3종 CronJob (v3.1 — chunked input 강제)

**P3-DATA-7 발견 핵심**: peer 당 4분기 business_signals = 평균 2,000 raw rows × 평균 258 chars ≈ **170k token / peer** — gpt-4o 128k window 초과. 단순 prompt assembly 는 hard fail.

→ **CapabilityEvolutionAgent 의 SQL pre-aggregation 단계가 필수**.

```python
# scripts/refresh_capability_evolution.py — SQL pre-aggregation
PRE_AGG_SQL = """
WITH ranked_signals AS (
    SELECT
        peer_id,
        business_area,
        period_year,
        COALESCE(period_quarter::text, 'annual') AS pq,   -- P3-DATA-1: NULL fallback
        signal_type,
        sentiment,
        summary,
        evidence_text,
        confidence,
        raw_article_id,
        ROW_NUMBER() OVER (
            PARTITION BY peer_id, business_area, period_year, period_quarter, signal_type
            ORDER BY confidence DESC NULLS LAST
        ) AS rn
    FROM raw_article_business_signals
    WHERE peer_id = %(peer_id)s
      AND period_year >= %(start_year)s
)
SELECT business_area, period_year, pq, signal_type, sentiment,
       summary, evidence_text, confidence, raw_article_id
FROM ranked_signals
WHERE rn <= 5      -- 그룹 당 top-5 confidence
ORDER BY period_year DESC, business_area, signal_type;
"""

# 결과 row 수: peer 당 약 ~150-300 rows (4분기 × 20 area × 13 type × 5 = 5,200 → top-5 압축 후)
# 텍스트 합: peer 당 ~30k-60k chars ≈ 10k-20k token → gpt-4o 안전 영역
```

```yaml
# axis-infra/k8s/base/cronjob-capability-evolution.yaml
schedule: "0 3 1 * *"      # 매월 1일 03:00 KST
command: ["python", "scripts/refresh_capability_evolution.py"]
# CapabilityEvolutionAgent — 4분기 business_signals → chunked LLM 합성
# peer_plus_payload['capability_evolution'] 갱신
# LLM 호출: 5 peer × 1회/월 ≈ $0.75/월 (chunked 후 token usage 안전)

# axis-infra/k8s/base/cronjob-sector-pulse.yaml — Phase 분리 (P3-DATA-5)
schedule: "0 2 * * 1"       # 매주 월 02:00 KST
command: ["psql", "-c", "REFRESH MATERIALIZED VIEW CONCURRENTLY sector_pulse"]
# SectorPulseAggregator — REFRESH 만, LLM 없음
# Phase 1 (W4-4): event_count / intensity / event_type_distribution
# Phase 2 (W4 운영 4주 후 활성화): z-score / anomaly trigger (4주 baseline 충족 시)

# axis-infra/k8s/base/cronjob-event-chain.yaml (옵션, 4주 후 결정)
schedule: "0 2 * * *"       # 매일 02:00 KST
command: ["python", "scripts/discover_event_chains.py"]
# EventChainDiscoveryAgent — Qdrant embedding 유사도 + 시간차 ≥ 7일 기반
# (P3-LOG-3, P3-LOG-4): 단순 keyword overlap X
# LLM 호출: 일일 candidates 평균 5건 × $0.05 → ~$0.25/일
```

운영 agent 위치: **`src/agents/context/`** (신규 디렉토리)

```text
src/agents/context/
    capability_evolution_agent.py
    sector_pulse_aggregator.py
    event_chain_discovery_agent.py    # (옵션)
    peer_id_aliases.py                 # P3-DATA-3: PEER_ID_ALIASES 중앙 관리
```

→ Supervisor 와 같은 1단계 agent 가 아님. **batch / context layer agent**. design doc 에 명확히 구분.

#### 3.4.7 Provenance Chain (시계열 진술의 추적성)

derived layer 의 모든 진술은 raw_article 까지 추적 가능해야 함.

```text
implication.peer_implication.capability_change
    ↓ references
peer_companies.peer_plus_payload['capability_evolution'].windows[i].evidence_signal_ids
    ↓ references
raw_article_business_signals.id
    ↓ references
raw_article_business_signals.raw_article_id
    ↓ references
raw_articles.id  (immutable, url, published_at)
```

`ImplicationResult.provenance` 에 모든 hop 의 id 가 인용되므로, **누구든 cluster 처리 1년 뒤에도 시사점의 근거를 단일 SQL 로 재구성 가능**.

#### 3.4.8 Time-Travel (과거 시점 복원)

- **Raw Layer**: `published_at` / `created_at` 기준 그대로 immutable.
- **Layer 2-B capability**: `peer_plus_payload['capability_evolution'].windows[]` 가 `period` 와 `generated_at` 보존 — append-only 로 저장 (덮어쓰지 않음).
- **Layer 2-C sector_pulse**: MATERIALIZED VIEW 에 `week_start` 자체가 시점 인덱스.
- **Layer 2-F snapshot history**: `legacy_records.archived_at` 으로 과거 snapshot 복원.

→ "2025-12-15 시점에 cluster_id=42 가 만들어졌다면 capability narrative 는 어땠을까?" 류 질문은:

```sql
SELECT windows
FROM peer_companies, jsonb_array_elements(peer_plus_payload->'capability_evolution'->'windows') windows
WHERE id = 'samsung_sds'
  AND (windows->>'generated_at')::timestamptz <= '2025-12-15'
ORDER BY (windows->>'generated_at')::timestamptz DESC
LIMIT 1;
```

→ 가능. append-only JSONB 가 핵심.

---

### 3.5 카드뉴스 v2 schema — 요약 + 시사점 + 대응 3섹션

> **답해야 할 질문**: "현재 카드뉴스는 요약만 들어가 있는데, `요약+시사점+대응` 세 섹션으로 명확히 분리되려면 schema 가 어떻게 달라져야 하는가?"
>
> **결론**: **`card_news` 컬럼 추가 0~1개**. 기존 `summary_lines`, `implication` JSONB 가 이미 다 담을 수 있음. 핵심은 **`implication` JSONB 의 key schema 정합화 + frontend 의 3섹션 분리 표시**.

#### 3.5.1 기존 `card_news` 컬럼 → v2 매핑

| v2 섹션 | 사용자 의미 | 기존 컬럼 / JSONB key | 결정 |
|---|---|---|---|
| **A. 요약** | "무슨 일이 있었나" 3줄 | `summary_lines TEXT[]` | ✓ 그대로 |
| **B. 시사점** | "그래서 의미가 뭔가" 1-2 문단 | `implication.skax_implication.why_important` <br/> + `implication.skax_implication.potential_impact` <br/> + `implication.peer_implication.peer_meaning` | ✓ JSONB 안 (W1-1 새 schema) |
| **C. 대응** | "SK AX 가 뭘 해야 하나" verb-first 최대 3개 | `implication.skax_implication.recommended_actions` | ✓ JSONB 안 (W1-1 새 schema) |
| (보조) 기회/위협 | "기회와 위협 각각" | `implication.skax_implication.opportunities` / `threats` | ✓ JSONB 안 |
| (보조) 후속 모니터링 | "지속 관찰할 것" | `implication.follow_up_questions` / `watch_points` | ✓ JSONB 안 |
| (보조) 정밀도 신호 | "이 카드를 얼마나 믿을 수 있나" | `implication.confidence` + `evidence_label` | ✓ JSONB 안 |
| Evidence Chain | "근거 4종" | `evidence_payload` JSONB | ✓ 그대로 |
| Provenance | "어디서 만들어졌나" | `implication.provenance` + `evidence_payload.provenance` | ✓ JSONB 안 |

→ **신규 컬럼 추가 0개**. 단 (a) `implication` JSONB 의 schema 정합화, (b) `card_news` row 가 v1/v2 어느 schema 인지 식별할 수 있어야 함.

#### 3.5.2 V34 (또는 W1-1 의 일부) — `implication` JSONB key 정합화

```sql
-- axis-backend/src/main/resources/db/migration/V34__card_news_v2_schema_lint.sql
-- 또는 V33 에 통합 — W1-1 의 ImplicationResult v4.0 schema 가 LLM 출력으로 안정화 된 후

-- 1) schema version 표기용 generated column (선택)
ALTER TABLE card_news
    ADD COLUMN IF NOT EXISTS card_schema_version VARCHAR(10) NOT NULL DEFAULT 'v1';

-- 2) v2 식별 (implication.provenance.prompt_version 으로도 가능, 컬럼은 편의)
CREATE INDEX IF NOT EXISTS idx_card_news_schema_version
    ON card_news(card_schema_version, created_at DESC);

-- 3) v2 카드의 implication JSONB key 필수 보장 — CHECK (선택, soft 강제)
ALTER TABLE card_news
    ADD CONSTRAINT chk_card_news_v2_implication_keys
    CHECK (
        card_schema_version <> 'v2' OR (
            implication ? 'skax_implication'
            AND implication->'skax_implication' ? 'why_important'
            AND implication->'skax_implication' ? 'recommended_actions'
        )
    ) NOT VALID;     -- 기존 183 행 회귀 차단; 신규만 강제
```

→ **컬럼 1개 (`card_schema_version`) + 인덱스 1개 + CHECK 1개**. 신규 테이블 X.

#### 3.5.3 `implication` JSONB 표준 schema (v2)

```jsonc
{
  "is_valid_implication": true,
  "implication_scope": "peer_and_skax",
  "peer_implication": {
    "company_id": "samsung_sds",
    "company_name_ko": "삼성에스디에스",
    "peer_meaning": "...",
    "capability_change": "...",
    "precedent_link": {
      "card_id": "card_20260415_samsung_sds_42",
      "relation": "follow_up",
      "days_since": 35,
      "rationale": "..."
    },
    "sourced_evidence_ids": ["fact_001", "fact_002"]
  },
  "skax_implication": {
    "why_important": "...",
    "potential_impact": "...",
    "opportunities": ["...", "..."],
    "threats": ["..."],
    "recommended_actions": [
      "에이전틱AI 협업 모델 PoC 제안서 작성",
      "MSP 입찰 사전 자격 점검",
      "고객사 임원 인터뷰 통해 수요 검증"
    ],
    "business_line_mapping": ["에이전틱AI", "MSP"]
  },
  "follow_up_questions": ["...", "...", "..."],
  "watch_points": ["...", "..."],
  "confidence": 0.74,
  "evidence_label": "sufficient",
  "provenance": {
    "generator": "ImplicationAgent",
    "prompt_version": "implication-v5.0",
    "model": "gpt-4o",
    "bundle_id": "...",
    "used_peer_profile_keys": ["..."],
    "used_skax_profile_keys": ["..."],
    "used_fact_ids": ["..."],
    "used_context_layers": [          // W4 이후
      "peer_event_timeline_recent",
      "capability_evolution",
      "sector_pulse_recent",
      "similar_cards_rag"
    ],
    "run_at": "2026-05-22T03:21:14+09:00"
  }
}
```

#### 3.5.4 기존 183 카드 호환 전략

| 옵션 | 작업량 | 결과 |
|---|---|---|
| **A. graceful fallback** (권장) | 0 | frontend 가 `card_schema_version != 'v2'` 인 카드는 `implication.skax_implication.what_it_means` 등 v1 key 를 시도하다 fallback 으로 `legacy_payload` 표시 |
| B. 백필 (LLM 재실행) | 183 × $0.022 = $4.0 + 4h 작업 | v2 schema 로 완전 통일 |
| C. 일괄 삭제 후 재생성 | 1h + LLM 비용 | 가장 깔끔하나 historical 사라짐 |

→ **A 권장** + 4주 후 측정으로 B/C 결정.

#### 3.5.5 Frontend 표시 명세 (백엔드 contract 변경 동반)

`openapi.yaml` 의 `CardNewsResponse` schema 에 다음 필드 명시:

```yaml
CardNewsResponse:
  type: object
  properties:
    id: { type: string }
    title: { type: string }
    summary_lines:           # 섹션 A — 요약
      type: array
      items: { type: string }
      minItems: 3
      maxItems: 3
    implication_summary:     # 섹션 B — 시사점 (조립 응답)
      type: object
      properties:
        why_important: { type: string }
        potential_impact: { type: string }
        peer_meaning: { type: string, nullable: true }
        precedent_card_id: { type: string, nullable: true }
    recommended_actions:     # 섹션 C — 대응 (조립 응답)
      type: array
      items: { type: string }
      minItems: 0
      maxItems: 3
    opportunities: ...
    threats: ...
    evidence_chain: ...
    confidence: ...
    evidence_label: ...
    card_schema_version: { type: string, enum: [v1, v2] }
```

→ `axis-backend` `CardController` 에서 `card_news.implication` JSONB 를 위 schema 로 명시 매핑 (조립). frontend 는 단일 contract 만 보면 됨.

→ **openapi 변경은 axis-infra/api/ 영역, axis-backend 와 조율 필요** (W3-5 운영 절차 문서에 PR 시퀀스 명시).

### 3.6 Evaluation & Observability Layer (W5)

> 출력 품질을 정량 측정. **Hybrid 배치** — rule-based 는 in-graph, LLM-as-Judge 는 sidecar. 두 결과 모두 `card_news.evaluation_payload` JSONB 에 누적.

#### 3.6.1 측정 지표 정의

**Rule-based 5 metric (in-graph, W5-1)** — `validate` 노드 확장. LLM X, latency 추가 < 50ms. **Phase 1: 4 metric (즉시) / Phase 2: regression_drift + W5-3 알림 (운영 14일 후)**.

| Metric | Phase | 정의 | 계산 |
|---|---|---|---|
| `context_hit_ratio` | 1 | implication 출력이 **실제 제공된** AnalysisContext layer 중 인용한 비율 | `len(provenance.used_context_layers) / max(1, available_layer_count)` <br/>※ v3.2.1 정정: 분모는 6 고정이 아닌 `AnalysisContext.available_layer_count` (신규 peer 의 `capability_evolution` 부재 등을 unfair penalty 없이 흡수) |
| `evidence_claim_ratio` | 1 | implication 의 numeric/date claim 중 evidence_payload 에 grounded 된 비율 | regex (한국어 수치/날짜 패턴) 로 출력의 claim 추출 → evidence 의 `financial_refs` / `source_links` / `mbb_refs` 와 substring + 정규화 매칭. 분모는 추출된 총 claim 수 |
| `specificity_score` | 1 | peer/business_area 명시 정도 (0~1) | 0.5 × (peer_name unique occurrence 가 1 이상이면 1, 아니면 비례) + 0.5 × (output 의 business_area 가 classification 의 sectors 와 매칭되는 비율) |
| `actionability_score` | 1 | recommended_actions 의 **한국어 verb-suffix pattern** + 구체성 (0~1) | (각 action 의 어미가 `(한다\|할 것\|검토\|착수\|수립\|구축\|확보\|강화\|점검\|모니터링\|도입\|평가\|분석\|추진\|개시\|확장)` 패턴 매칭 비율) × (action 본문에 회사명/시점/숫자 중 1개 이상 등장 비율) <br/>※ v3.2.1 정정: 한국어 어순상 "verb-first" 불가능 → 어미·동사 어휘 사전 매칭 |
| `regression_drift` | **2** (운영 14일 후) | 7일 평균 confidence 대비 변동 | `(this_cluster.confidence - rolling_7d_avg) / rolling_7d_avg`. Baseline 7일 미만이면 `null` (sector_pulse 와 동일 Phase 분리) |

→ `validate` 노드가 score 계산 후 `state.validation.metrics` 에 저장. `card_writer` 가 `evaluation_payload['rule_based']` 로 INSERT.

→ **Threshold 는 lenient 시작 (v3.2.1 정책)**: 운영 1주차는 카드 차단/플래그 0 — 점수 측정만. **1주 누적 후 percentile-based calibration** (예: `actionability_score` 의 bottom 10% 를 threshold 로). 그 전까지 `human_review_flags` 추가 X. 이는 사용자 결정사항 — "결과가 너무 적게 나오거나 많이 제한되지는 않게" 의 충족.

**LLM-as-Judge 4 score (sidecar, W5-2)** — `axis-cron-card-evaluator` (5분 주기, gpt-4o-mini).

| Score | 정의 | Prompt |
|---|---|---|
| `faithfulness` (0~5) | implication 의 모든 claim 이 evidence 와 일치하는가 | "다음 카드의 implication 본문과 evidence_payload 를 비교해 각 claim 의 근거 여부를 평가" |
| `specificity_llm` (0~5) | generic vs peer-specific 판정 (rule-based 와 보완) | "이 카드가 '디지털 전환 가속화' 같은 generic 문장 위주인지, 회사/사업 특정 분석인지 평가" |
| `actionability_llm` (0~5) | recommended_actions 의 실행 가능성 | "이 actions 가 SK AX 의 사업 영역과 매칭되며 실행 가능한지 평가" |
| `peer_relevance` (0~5) | implication 의 peer 관점 vs SK AX 관점 균형 | "peer_implication 과 skax_implication 이 각각 충분한 차별화를 가지는지 평가" |

→ 평가 결과 4 score + 한 줄 reasoning 을 `evaluation_payload['llm_judge']` JSONB 로 저장. `evaluated_at`, `evaluator_model_version` 메타 동반.

→ 비용: gpt-4o-mini × 1 call/카드 ≈ **$0.05/카드**. 일일 cluster 30개 ≈ +$1.5/일 (전체 한도의 5%).

#### 3.6.2 데이터 스키마 — `evaluation_payload` JSONB

```json
{
  "rule_based": {
    "context_hit_ratio": 0.83,
    "evidence_claim_ratio": 0.91,
    "specificity_score": 0.72,
    "actionability_score": 0.65,
    "regression_drift": -0.04,
    "evaluated_at": "2026-05-21T10:01:23+09:00",
    "evaluator_version": "rule-v1.0"
  },
  "llm_judge": {
    "faithfulness": 4.5,
    "specificity_llm": 4.0,
    "actionability_llm": 3.5,
    "peer_relevance": 4.2,
    "reasoning": "Evidence 와 일치하나 actions 가 일부 generic. peer 차별화 양호.",
    "evaluated_at": "2026-05-21T10:05:14+09:00",
    "evaluator_model_version": "gpt-4o-mini-2024-07"
  }
}
```

→ **`evaluation_payload` JSONB 컬럼 신설 필요** (v3.2.1 정정 — 실측 검증 결과 기존 `card_news` 에 미존재). V33 마이그레이션 (§W4-1) 에 `ALTER TABLE card_news ADD COLUMN evaluation_payload JSONB DEFAULT '{}'::jsonb` 추가. 기존 `validation_pass` (boolean) + `validation_sc_score` (double) 컬럼은 그대로 유지 (legacy validate 결과). 신규 9 score 는 모두 `evaluation_payload` JSONB 안에서 namespace 분리.

#### 3.6.3 회귀 감지 (W5-3, 옵션)

매일 02:30 CronJob 이 최근 7일 vs 직전 7일의 다음 평균 비교:

```sql
SELECT
    period,
    AVG((evaluation_payload->'rule_based'->>'actionability_score')::float) AS act_score,
    AVG((evaluation_payload->'llm_judge'->>'faithfulness')::float)         AS faith_score
FROM card_news
WHERE created_at >= NOW() - INTERVAL '14 days'
GROUP BY DATE_TRUNC('week', created_at);
```

→ `act_score` 또는 `faith_score` 가 직전 주 대비 -10% 이상 감소 시 Slack/email 알림.

#### 3.6.4 Sidecar 운영 흐름

```text
[Supervisor card_writer] → card_news INSERT (evaluation_payload='{}'::jsonb)
                                  ↓
                          (5분 후, sidecar 트리거)
                                  ↓
┌─────────────────────────────────────────────────────┐
│ axis-cron-card-evaluator (CronJob, */5 * * * *)     │
│   1. SELECT id, implication, evidence_payload       │
│      FROM card_news                                  │
│      WHERE card_schema_version = 'v2'   ← v3.2.1   │
│        AND NOT (evaluation_payload ? 'llm_judge')   │
│        AND created_at >= NOW() - INTERVAL '24 h'    │
│      ORDER BY created_at ASC                        │
│      LIMIT LEAST(20, daily_budget_remaining())      │
│   2. LLM-as-Judge prompt 호출 (gpt-4o-mini)         │
│   3. UPDATE card_news                                │
│      SET evaluation_payload =                        │
│          evaluation_payload || jsonb_build_object(  │
│            'llm_judge', $score_obj)                 │
│      WHERE id = $id;                                │
└─────────────────────────────────────────────────────┘
```

→ Idempotent (`NOT (... ? 'llm_judge')` 가드). cluster 처리에 영향 0. 실패 시 다음 주기에 재시도.

→ **v3.2.1 정정**:
  - `WHERE card_schema_version = 'v2'` guard 로 기존 v1 카드 191건 자동 skip (heuristic 메타데이터만 있어 평가 불가). 사용자 결정 — "요약만 담고 있는 이전 버전은 제외".
  - `evaluation_payload ? 'llm_judge'` JSONB key existence 연산 (`IS NULL` 대신) — 컬럼 default 가 `'{}'::jsonb` 이므로 NULL 비교 부적합.
  - `daily_budget_remaining()` — daily LLM 비용 cap (default $5/일 ≈ 100 카드) 의 잔여량. 도달 시 batch skip + Slack 알림 (hard limit 아님, 다음날 자동 재개).
  - `ORDER BY created_at ASC` — 오래된 미평가 카드 우선 (burst 시 fairness).

#### 3.6.5 Self-evaluation bias mitigation (v3.2.1 신설)

> gpt-4o 가 만든 카드를 gpt-4o-mini 가 평가 — model family 가 같아 self-favor 위험.

**Phase 1 (W5-2 도입 즉시)**: gpt-4o-mini 단독 평가. 비용 최소화.

**Phase 2 (운영 1주 후)**: 매주 무작위 sampling 10% 의 카드를 Claude 3.5 Sonnet (또는 다른 family) 로 cross-check 평가.
- `evaluation_payload['llm_judge_crosscheck']` JSONB 에 별도 저장 (gpt-4o-mini 와 비교 가능)
- 두 평가자의 `faithfulness` score 가 1.0 이상 일관되게 다르면 evaluator prompt 재검토 ticket
- 추가 비용: 10% × $0.10/카드 (Claude 가 더 비쌈) ≈ +$0.30/일

→ Bias 자체를 0 으로 만들 수 없으나, **gap 을 측정 가능하게** 만들어 calibration 가능.

#### 3.6.6 Threshold calibration / Burst / Cost cap 정책 (v3.2.1 신설)

> 운영 첫 1~2주는 빡빡한 기준이 결과를 과도하게 제한하지 않도록 lenient 시작.

| 정책 | 정의 |
|---|---|
| **Threshold calibration** | 운영 7일까지 모든 카드는 `human_review_flags` 추가 X (측정만). 7일 후 `actionability_score` / `evidence_claim_ratio` / `specificity_score` 의 **bottom 10% percentile** 을 자동 threshold 로 산출. percentile 산출 SQL 은 weekly CronJob 으로 갱신. |
| **Hard block 절대 X** | W5 의 score 어느 것도 카드 INSERT 차단 X. 차단은 기존 validate 의 numeric/단정표현 검사로 한정 (v3.2 §9 결정 3항). |
| **Sidecar burst** | 일평균 카드 10~30건 가정. backfill 등으로 미평가 카드 ≥ 100건 누적 시: ① Slack 알림, ② batch limit 을 20 → 40 으로 동적 증가 (5분 주기는 유지), ③ 1시간 내 해소되지 않으면 paging. |
| **Cost cap (soft)** | 일일 LLM 비용 $5/일 도달 시 sidecar batch skip + Slack 알림. 다음 KST 00:00 에 자동 재개. **Hard kill 아님** — backlog 가 다음날로 이연될 뿐. 비상 시 manual reset 가능. |
| **Backlog 회복 보장** | 비용 cap 으로 skip 된 카드는 `evaluation_payload['llm_judge']` 부재로 다음 주기에 자동 평가 대상이 됨 — 데이터 손실 0. |

---

## 4. 작업 계획 — 4 단계 18 작업

### Critical Path

```
W1-1 (ImplicationAgent v4.0)  ──┐
W1-2 (schema 정합화)            ─┼─→ W2-1 (LangGraph 화) ──→ W2-3 (Validate 노드) ──→ W3-3 (golden test)
W1-3 (typed metadata)           ─┘                                        ↑
W1-4 (card_news v2 schema lint)                                            │
                                  W2-2 (Profile 2-tier) ─────────────────┘
                                  W2-4 (CardNewsAgent 이중 처리 제거)
W3-1 (design docs)
W3-2 (_deprecated 정리)
W3-4 (Langfuse 전수)
W3-5 (단위 테스트)

W4 (Context Engineering) — W2 완료 후 독립 진행 가능
    W4-1 (V33 인덱스+VIEW+MV)
    W4-2 (AnalysisContextBuilder)
    W4-3 (CapabilityEvolutionAgent CronJob)
    W4-4 (SectorPulse REFRESH CronJob)
    W4-5 (ImplicationAgent prompt v4.0 → v5.0; AnalysisContext 입력 추가)
    W4-6 (옵션) EventChainDiscoveryAgent — 4주 운영 측정 후 도입 결정
```

---

### 단계 W1 — 1주차 (foundations, 약 16h)

#### W1-1. `ImplicationAgent` v2.0 — LLM 기반 재구현 ★ P0-1

**원칙**:
- deprecated 코드 부활 X. 입력 계약·출력 schema 모두 새로.
- 입력 4종 활용 (Bundle + IntegratedIssue + AnalysisResult + ProfileContext).
- 출력은 §3.3 의 `ImplicationResult` dataclass.
- 실패 시 → `analysis/implication.py:ImplicationGenerator` (현 heuristic) 가 fallback 으로 동작.

**파일 변경**:
```text
src/agents/implication_agent.py            (재작성, 16 → ~250 LoC)
src/analysis/implication.py                (heuristic fallback 으로 유지, public API 안 깸)
src/analysis/models.py                     (ImplicationResult / PeerImplication / SkaxImplication 신규)
src/analysis/prompts/implication_v4.py     (신규 — 프롬프트 분리)
src/services/agent_output_validation.py    (regex 검증 헬퍼 추가)
tests/test_implication_agent.py            (신규 — golden file 포함)
tests/golden/implication_v4/               (신규 디렉토리)
```

**code skeleton**:
```python
class ImplicationAgent:
    prompt_version = "implication-v4.0"
    _LLM_MODEL = "gpt-4o"

    def __init__(self, *, fallback: ImplicationGenerator | None = None) -> None:
        self._fallback = fallback or ImplicationGenerator()
        self._llm: ChatOpenAI | None = None

    def generate(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        integrated_issue: dict[str, Any],
        analysis: dict[str, Any],
        profile_context: ProfileContext,
    ) -> ImplicationResult:
        if not _has_minimum_inputs(integrated_issue, analysis):
            return self._fallback_result(reason="insufficient_inputs", ...)
        try:
            prompt = _render_prompt(...)
            raw = self._invoke_llm(prompt)
            result = _parse_and_normalize(raw, ...)
            return _attach_provenance(result, ...)
        except LLMError as exc:
            log.warning("ImplicationAgent LLM failure → fallback | %s", exc)
            return self._fallback_result(reason=str(exc), ...)
```

#### W1-2. AnalysisPackage schema 정합화 ★ P1-2

- `analysis/models.py` 의 `AnalysisResult` / `IntegratedIssue` / `ImplicationResult` dataclass 를 **실 LLM 출력에 맞춤**.
- `_normalize_analysis_result()` / IssueIntegrationAgent 내 `_tag_integrated_issue()` 가 dataclass 직접 반환.
- Backward compat alias 유지 (`SummaryResult = LegacySummaryResult`).

**파일 변경**:
```text
src/analysis/models.py                      (AnalysisResult / IntegratedIssue 재정의)
src/analysis/analyzer.py                    (_normalize_analysis_result → dataclass)
src/agents/issue_integration_agent.py       (_tag_integrated_issue → dataclass)
src/agents/analysis_supervisor_agent.py     (typed read)
```

#### W1-3. `AnalysisInputBundle.metadata` typed ★ P2-5

```python
@dataclass(slots=True)
class AnalysisInputMetadata:
    representative_id: int
    cluster_article_ids: list[int]
    classification: ClassificationPayload  # dict → typed
    created_at: str
    profile_snapshot_version: str | None = None
    source_run_id: str | None = None       # crawl_run UUID
```

**파일 변경**:
```text
src/analysis/models.py
src/agents/issue_integration_agent.py:analysis_input_bundle_from_articles
```

#### W1-4. `card_news` v2 schema lint ★ P3-5 / P3-LOG-1

- `card_news` 에 `card_schema_version VARCHAR(10) DEFAULT 'v1'` 컬럼 추가.
- `CardNewsAgent.generate_from_analysis_package()` 이 새 카드는 `'v2'` 로 INSERT 하도록 변경.
- `implication` JSONB 에 W1-1 의 `ImplicationResult` v4.0 schema 가 정확히 들어가도록 직렬화.
- 기존 191 행은 `'v1'` 로 유지 — graceful fallback (§3.5.4 옵션 A).

**파일 변경**:
```text
[V33 통합 — DDL 자체는 W4-1 의 V33__context_engineering.sql 에 포함. W1-4 는 axis-ai 측 코드 변경만]
src/agents/card_news_agent.py                (INSERT 시 card_schema_version='v2')
src/db/article_store.py                      (INSERT statement 갱신)
```

**V33 파일 단일화 정책**: `V33__context_engineering.sql` (W4-1) 이 모든 W4-관련 DDL + W1-4 의 `card_schema_version` 컬럼을 함께 포함. W1-4 의 axis-ai 코드 변경은 V33 머지 후에만 효력 발생. 즉 **W1-4 는 W4-1 의 V33 머지에 의존** (Critical Path 표 갱신됨).

→ openapi 변경 (`axis-infra/api/openapi.yaml` 의 `CardNewsResponse`) 은 axis-backend lead 와 PR 동기. **이 계획서는 axis-ai 변경만 책임**; openapi PR sequence 는 W3-5 운영 문서에 명시.

---

### 단계 W2 — 2주차 (graph + profile + validation, 약 20h)

#### W2-1. Supervisor 를 LangGraph StateGraph 로 ★ P1-1

```python
# src/pipeline/supervisor_graph.py (신규)
# v3.1: build_analysis_context 노드 + analysis_context state field (§3.1 다이어그램과 정합)
from langgraph.graph import END, StateGraph

class SupervisorState(TypedDict):
    input_bundle: AnalysisInputBundle
    profile_context: ProfileContext | None
    analysis_context: AnalysisContext | None      # W4 신설 — §3.4.5
    integrated_issue: dict[str, Any] | None
    analysis: dict[str, Any] | None
    implication: ImplicationResult | None
    validation: ValidationReport | None
    analysis_package: AnalysisPackage | None      # W2-1 작업 4 (assemble 결과)
    card_news_id: str | None                      # W2-1 작업 5 (card_writer 결과)
    errors: Annotated[list[NodeError], operator.add]
    human_review_flags: list[str]

def _build_supervisor_graph() -> CompiledStateGraph:
    g = StateGraph(SupervisorState)
    g.add_node("profile_context", _profile_context_node)
    g.add_node("build_analysis_context", _build_analysis_context_node)  # W4 신설 — LLM X
    g.add_node("issue_integrate", _issue_integrate_node)
    g.add_node("strategic_analyze", _analyze_node)
    g.add_node("implication", _implication_node)
    g.add_node("validate", _validate_node)
    g.add_node("assemble", _assemble_node)
    g.add_node("card_writer", _card_writer_node)            # W2-1 작업 5 신설 — CardNewsAgent 호출
    g.add_node("human_review", _human_review_node)

    g.set_entry_point("profile_context")
    g.add_edge("profile_context", "build_analysis_context")    # W4 신설
    g.add_edge("build_analysis_context", "issue_integrate")    # W4 신설
    g.add_edge("issue_integrate", "strategic_analyze")
    g.add_edge("strategic_analyze", "implication")
    g.add_edge("implication", "validate")
    g.add_conditional_edges("validate", _route_after_validate, {
        "pass": "assemble",
        "fail": "human_review",
    })
    g.add_edge("assemble", "card_writer")                       # W2-1 작업 5 — 카드는 분석의 결과물
    g.add_edge("card_writer", END)
    g.add_edge("human_review", END)
    return g.compile()

def _card_writer_node(state: SupervisorState) -> SupervisorState:
    """As-Is 의 ingestion_graph.card_news_node 를 이관. CardNewsAgent.write_card 호출."""
    pkg = state["analysis_package"]
    card_id = CardNewsAgent().write_card(pkg)  # card_news WRITE (v2 schema)
    return {**state, "card_news_id": card_id}

# W2 시점에 build_analysis_context 는 stub (W4-2 에서 본격 구현):
def _build_analysis_context_node(state: SupervisorState) -> SupervisorState:
    # W2 시점 — stub. W4-2 에서 AnalysisContextBuilder 로 교체.
    return {**state, "analysis_context": None}

# DataAnalysisSupervisorAgent 는 thin wrapper 로 축소
class DataAnalysisSupervisorAgent:
    def analyze_input_bundle(self, *, input_bundle, ...) -> AnalysisPackage:
        state = self._graph.invoke({"input_bundle": input_bundle, ...})
        return _state_to_package(state)
```

각 노드에 `@_logged_step` 데코레이터 (ingestion_graph 와 동일 패턴) → `pipeline_logs` 에 elapsed_ms/in/out count 자동 기록.

**작업 5 (v3.1.3 신설) — CardNewsAgent 를 Supervisor 의 last 노드로 이관**

> §0.1 의 Layer 분리 약속을 이행. 카드뉴스는 분석/시사점/대응의 직렬화 결과 (= Supervisor 의 산출물) 이므로 `ingestion_graph` 가 아닌 `supervisor_graph` 의 마지막 노드에서 작성.

- `ingestion_graph.card_news_node` 제거 (이관) — `pipeline/ingestion_graph.py` 에서 `card_news_node` 호출 제거. ingestion_graph 의 종료 시점은 `raw_articles` + cluster classification 까지로 한정 (Layer A 책임).
- `pipeline/analysis_pipeline.py` 의 `AnalysisPipelineRunner` 가 cluster 마다 supervisor.invoke 후 `card_news_id` 까지 반환받음 (호출자는 `state['card_news_id']` 로 카드 ID 접근).
- 병렬화 (`_GPT_WORKERS=5`) 는 supervisor invocation 단위로 옮김 — 같은 효과, cluster 단위 격리 (현재는 카드 단위 격리).
- v1 카드 호환: `card_schema_version` 컬럼이 `'v2'` 로 기록되도록 보장 (W1-4 와 정합).

**파일 변경**:
```text
src/pipeline/supervisor_graph.py            (신규, ~450 LoC)  # card_writer 노드 포함 → +50 LoC
src/agents/analysis_supervisor_agent.py     (~241 → ~80 LoC, graph 호출 wrapper)
src/pipeline/analysis_pipeline.py           (Supervisor 호출부만 변경, 외부 계약 동일)
src/pipeline/ingestion_graph.py             (card_news_node 제거, ingestion 의 마지막은 classification)
src/agents/card_news_agent.py               (호출 인터페이스 그대로, 호출 위치만 supervisor 노드로 이동)
src/db/article_store.py                     (save_pipeline_log 호출 추가)
```

#### W2-2. ProfileAgent 2-tier 분리 ★ P0-2

**Tier A — Static snapshot**:

```sql
-- axis-backend/src/main/resources/db/migration/V33__add_peer_profile_snapshot.sql
ALTER TABLE peer_companies
    ADD COLUMN profile_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN profile_snapshot_version VARCHAR(50),
    ADD COLUMN profile_snapshot_generated_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_peer_companies_snapshot_version
    ON peer_companies(profile_snapshot_version);
```

```python
# scripts/refresh_peer_profile_snapshots.py (신규)
def main() -> None:
    for company_id in PEER_COMPANY_IDS + SELF_COMPANY_IDS:
        snapshot = ProfileAgent().build_profile(company_id)
        save_peer_profile_snapshot(
            company_id=company_id,
            payload=snapshot,
            version="profile-v5",
        )
```

```yaml
# axis-infra/k8s/base/cronjob-profile-refresh.yaml (신규)
apiVersion: batch/v1
kind: CronJob
metadata:
  name: axis-cron-profile-refresh
spec:
  schedule: "0 3 * * 1"          # 월요일 03:00 KST
  timeZone: "Asia/Seoul"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: axis-ai-sa
          containers:
            - name: refresh
              image: amdp-registry.skala-ai.com/skala26a-ai3/axis-ai:latest
              command: ["python", "scripts/refresh_peer_profile_snapshots.py"]
```

**Tier B — Recent enrichment**:

```python
# src/agents/profile_agent.py 에 추가
def build_context_v2(
    self,
    *,
    companies: list[str],
    sectors: list[str] | None = None,
    event_type: str | None = None,
    lookback_days: int = 30,
) -> ProfileContext:
    skax_profile = _load_snapshot("sk_ax")             # DB only
    peer_profiles = {
        peer_id: _load_snapshot(peer_id)
        for peer_id in companies if peer_id not in SELF_COMPANY_IDS
    }
    # Tier B enrichment — DB query only, no LLM
    for peer_id, profile in peer_profiles.items():
        profile["recent_signals"] = _load_recent_business_signals(
            company_id=peer_id, days=lookback_days, limit=3,
        )
        profile["recent_financial"] = _load_latest_financial_metrics(
            company_id=peer_id,
        )
    return ProfileContext(
        skax_profile=skax_profile,
        peer_profiles=peer_profiles,
        sector_context=_sector_context(sectors),
    )
```

**파일 변경**:
```text
axis-backend/src/main/resources/db/migration/V33__add_peer_profile_snapshot.sql
src/agents/profile_agent.py                  (build_context_v2 추가)
src/db/article_store.py                      (save/load_peer_profile_snapshot)
scripts/refresh_peer_profile_snapshots.py    (신규)
axis-infra/k8s/base/cronjob-profile-refresh.yaml  (신규)
axis-infra/k8s/base/kustomization.yaml       (cronjob 추가)
```

#### W2-3. Validate 노드 — Quality Gate ★ P1-3

```python
@dataclass(slots=True)
class ValidationReport:
    pass_: bool
    numeric_violations: list[NumericViolation]     # 출처에 없는 수치
    certainty_warnings: list[str]                   # 단정 표현
    evidence_chain_warnings: list[str]              # source_links / provenance 누락
    implication_confidence_warning: bool
    integrated_issue_valid: bool
    analysis_valid: bool

def _validate_node(state: SupervisorState) -> SupervisorState:
    report = ValidationReport(
        integrated_issue_valid=state["integrated_issue"].get("is_valid_summary", False),
        analysis_valid=state["analysis"].get("is_valid_analysis", False),
        numeric_violations=_check_numeric_violations(
            implication=state["implication"],
            fact_basis=state["integrated_issue"].get("fact_basis", []),
            key_numbers=state["integrated_issue"].get("key_numbers", []),
        ),
        certainty_warnings=_check_certainty(state["implication"]),
        evidence_chain_warnings=_check_evidence_chain(state),
        implication_confidence_warning=state["implication"].confidence < 0.4,
        pass_=False,
    )
    report.pass_ = (
        report.integrated_issue_valid
        and report.analysis_valid
        and not report.numeric_violations
        and not report.evidence_chain_warnings
    )
    flags = list(state.get("human_review_flags") or [])
    if not report.pass_:
        flags.append(state["input_bundle"].bundle_id)
    return {**state, "validation": report, "human_review_flags": flags}
```

**파일 변경**:
```text
src/services/agent_output_validation.py     (확장)
src/pipeline/supervisor_graph.py            (_validate_node)
src/analysis/models.py                      (ValidationReport)
```

#### W2-4. CardNewsAgent implication 이중 처리 제거 ★ P1-4

- `_implication()` (analysis 기반 fallback) → ImplicationAgent 가 fail 했을 때만 호출
- `_implication_from_result()` → 정상 path, ImplicationResult 의 `skax_implication.why_important` + `potential_impact` 합성
- frontend_implication 매핑도 새 schema 기반

```python
# src/agents/card_news_agent.py:generate_from_analysis_package
implication_result = package.get("implication") or {}
if isinstance(implication_result, ImplicationResult) and implication_result.is_valid_implication:
    card["implication"] = _format_implication(implication_result)
    card["frontend_implication"] = _format_frontend_implication(implication_result)
else:
    card["implication"] = _implication_from_analysis(package.get("analysis") or {})
    card["frontend_implication"] = _frontend_from_analysis(package.get("analysis") or {})
```

**파일 변경**:
```text
src/agents/card_news_agent.py               (분기 명확화)
```

---

### 단계 W3 — 3주차 (docs / tests / cleanup, 약 12h)

#### W3-1. design docs 신설 ★ P2-1

| 파일 | 내용 |
|---|---|
| `design/30-analysis/00-supervisor.md` | StateGraph 노드 명세, retry 정책, validate gate |
| `design/30-analysis/01-issue-integration.md` | 뉴스/문서형 분기, fact_basis 추출 룰, key_numbers |
| `design/30-analysis/02-strategic-analysis.md` | `analysis-v3.0` 프롬프트, impact_level 기준 |
| `design/30-analysis/03-profile-context.md` | 2-tier (snapshot CronJob + recent enrichment) |
| `design/30-analysis/04-implication.md` | `implication-v4.0` 프롬프트, evidence_label, validation |

#### W3-2. `_deprecated/` 정리 ★ P2-4

- `agents/_deprecated/implication_agent.py` 의 P.C.R.O 프롬프트 요소를 W1-1 새 구현에 흡수.
- 정리: `_deprecated/` → `_archived/` rename + README 추가.
- 동일 클래스명 `ImplicationAgent` 충돌 차단 (`_archived` 는 `__init__.py` 에서 re-export 안 함).

#### W3-3. Golden file 테스트 + 단위 테스트 ★ P2-2 (W1-3 의 후속)

```text
tests/test_supervisor_graph.py              (신규)
tests/test_issue_integration_agent.py       (신규)
tests/test_strategic_analyzer.py            (신규)
tests/test_implication_agent.py             (W1-1 에서 시작, 보강)
tests/test_profile_agent_context.py         (신규)
tests/golden/
    supervisor/
        samsung_sds_partnership.json
        lg_cns_earnings.json
        hyundai_autoever_personnel.json
    implication_v4/
        with_capability_change.json
        without_capability_change.json
        evidence_insufficient.json
```

LLM 은 `pytest-mock` + recorded response (LangChain `FakeChatModel`) 로 deterministic.

#### W3-4. Langfuse tracing 전수 ★ P2-3

- Supervisor graph 의 모든 노드에 `tracing_config(agent=node_name, phase=..., prompt_version=...)` 적용
- session_id = `bundle_id`, trace_id = `cluster_id` → cluster 단위 trace 1개로 묶임

#### W3-5. 운영 절차 문서화

- README 의 "단독 실행 스크립트" 섹션에 `refresh_peer_profile_snapshots.py` 추가
- `CronJob` 동작 모니터링 가이드 + manual trigger 절차
- LLM 비용 분기별 예상치 표
- **W4 이후**: `refresh_capability_evolution.py`, `refresh_sector_pulse.sh`, (옵션) `discover_event_chains.py` 운영 절차
- **카드뉴스 v2 마이그레이션 시퀀스** (§3.5.5 backend openapi PR 순서) 문서화

---

### 단계 W4 — 4주차 (Context Engineering, 약 22h, W2 완료 후 시작 가능)

> **목적**: 시사점·대응 추론 시 cluster 자체 정보 + ProfileContext 만으로는 도달 못 하는 **종방향 (시계열) / 횡방향 (섹터·이벤트체인) 맥락** 을 cluster-time 에 4,000 token 이내 active context 로 합성·주입.
>
> **핵심 원칙**: cluster-time LLM 추가 호출 0건. 비용은 운영 CronJob (월1회 + 주1회) 으로 분산.
>
> **v3.1 추가**: W4-0 (data hygiene precondition) 신설. 실측 발견 8 issue 중 W4 본 작업 전 선결 필요한 항목 처리.

#### W4-0. Data hygiene precondition — schema 정합화 ★ P3-CRIT / P3-DATA-2/3 (3-4h)

> 시뮬레이션 시 발견된 schema 정합성 이슈를 W4 본 작업 전에 확정. **테이블 변경 없음**. Python 측 데이터 모델 / 상수 / utility 만 추가.

**작업 1. `src/services/peer_id_aliases.py` 신규** — peer_id 표기 정규화 중앙화 (P3-DATA-3)
```python
PEER_ID_ALIASES: dict[str, list[str]] = { ... }  # §3.4.5 참고
def expand_peer_aliases(peer_id: str) -> list[str]: ...
def normalize_to_canonical_id(any_id: str) -> str | None: ...
```

**작업 2. `src/services/metric_canonical.py` 신규** — financial_metrics.metric_name 정규화 (P3-DATA-2)
```python
METRIC_CANONICAL: dict[str, str] = {
    "net_income": "net_income", "당기순이익": "net_income", "순이익": "net_income",
    "revenue_total": "revenue_total", "매출": "revenue_total", "매출액": "revenue_total",
    # ... §3.4.3 의 CASE 매핑과 동일
}
def canonicalize_metric(metric_name: str) -> str: ...
```

**작업 3. `src/services/cluster_identity.py` 신규** — card 의 stable identity 추출 (P3-CRIT-1/2)
```python
@dataclass(slots=True)
class CardStableIdentity:
    card_id: str                        # 항상 사용 가능
    source_raw_article_ids: list[int]   # 빈 list 가능 (15% 카드)
    peer_event_date_key: tuple[str | None, str, str]  # (peer, event_type, date) — fallback
    
def stable_card_join_key(card: dict) -> CardStableIdentity: ...
```

**작업 4. `src/agents/context/_data_quality_checks.py` 신규** — W4-2/W4-3 의 graceful fallback 로직
```python
def is_signal_well_grouped(signal: dict) -> bool:
    """period_quarter NULL 인 경우 ('unknown') 그룹화 fallback 가능 여부 판단."""
    return signal.get("period_year") is not None  # year 만 있어도 그룹화 가능

def is_card_provenance_traceable(card: dict) -> bool:
    """source_raw_article_ids 가 비어있어도 sources JSONB 또는 evidence_payload 로 추적 가능?"""
    if card.get("source_raw_article_ids"):
        return True
    if (card.get("sources") or [])  or (card.get("evidence_payload", {}).get("source_links")):
        return True
    return False
```

**작업 5. `V32_5__card_news_backfill.sql` 신규** — P3-CRIT-4 즉시 복구 (≤ 1분 실행)

> `card_news.company` 와 `raw_articles.matched_companies/matched_sectors` 는 모두 정상이지만, CardNewsAgent 후처리의 두 시점 회귀 (5/15 FK, 5/20 sector) 로 `peer_company_id` / `primary_keyword_category` 가 누적 60건+ NULL. **데이터는 다 있으므로 join 만 다시 채우면 됨.**

```sql
-- axis-backend/src/main/resources/db/migration/V32_5__card_news_backfill.sql
BEGIN;

-- (a) peer_company_id FK 백필 — card_news.company 와 peer_companies.id 100% 매칭 확인됨
UPDATE card_news cn
SET peer_company_id = cn.company,
    updated_at = NOW()
WHERE cn.peer_company_id IS NULL
  AND cn.company IS NOT NULL
  AND EXISTS (SELECT 1 FROM peer_companies pc WHERE pc.id = cn.company);

-- (b) primary_keyword_category 백필 — raw_articles.matched_sectors 첫 원소
UPDATE card_news cn
SET primary_keyword_category = sub.sector,
    updated_at = NOW()
FROM (
    SELECT cn2.id AS card_id,
           (SELECT ra.matched_sectors->>0
              FROM raw_articles ra
             WHERE ra.id = ANY(cn2.source_raw_article_ids)
               AND jsonb_array_length(ra.matched_sectors) > 0
             LIMIT 1) AS sector
      FROM card_news cn2
     WHERE cn2.primary_keyword_category IS NULL
       AND cn2.source_raw_article_ids IS NOT NULL
       AND array_length(cn2.source_raw_article_ids, 1) > 0
) sub
WHERE cn.id = sub.card_id AND sub.sector IS NOT NULL;

-- (c) 검증 — 백필 후 NULL 비율
SELECT 'after_backfill' AS phase,
       COUNT(*) AS total,
       COUNT(*) FILTER (WHERE peer_company_id IS NULL) AS fk_null_remaining,
       COUNT(*) FILTER (WHERE primary_keyword_category IS NULL) AS sector_null_remaining
FROM card_news
WHERE created_at >= '2026-05-15';

COMMIT;
```

> **회귀 원인 추적은 별도 P1 ticket** — axis-ai `CardNewsAgent` (또는 후처리 단계) 의 5/14~5/20 변경 이력 점검. 두 시점 회귀이므로 2개 커밋 식별 필요. 추적 완료 전까지 백필 SQL 은 매일 CronJob 으로 1회 idempotent 재실행 (`WHERE … IS NULL` 가드 덕에 안전).

**파일 변경**:
```text
src/services/peer_id_aliases.py             (신규)
src/services/metric_canonical.py            (신규)
src/services/cluster_identity.py            (신규)
src/agents/context/_data_quality_checks.py  (신규)
tests/test_data_hygiene.py                  (신규 — 모든 utility 의 알려진 케이스 verification)
axis-backend/db/migration/V32_5__card_news_backfill.sql  (신규 — P3-CRIT-4 즉시 복구)
```

→ 별도 PR 로 머지. W4-1 ~ W4-5 가 이 모듈을 import. V32_5 는 V33 보다 먼저 적용해 W4-1 의 VIEW/MV 가 백필된 행 위에서 계산되도록 보장.

#### W4-1. V33 마이그레이션 — 인덱스 + VIEW + MATERIALIZED VIEW ★ P3-1/P3-3

> 신규 테이블 0개. **컬럼 2 + 인덱스 4 + VIEW 2 + MATERIALIZED VIEW 1** 만 (v3.2.1: evaluation_payload 컬럼 + 인덱스 1 추가).

```sql
-- axis-backend/src/main/resources/db/migration/V33__context_engineering.sql
-- (W1-4 의 card_schema_version 컬럼 + W5-1 의 evaluation_payload 컬럼 통합)

-- (a) card_news v2 schema 컬럼 (W1-4 통합)
ALTER TABLE card_news
    ADD COLUMN IF NOT EXISTS card_schema_version VARCHAR(10) NOT NULL DEFAULT 'v1';
CREATE INDEX IF NOT EXISTS idx_card_news_schema_version
    ON card_news(card_schema_version, created_at DESC);

-- (a-2) W5 evaluation_payload 컬럼 (v3.2.1 — 실측 검증 후 신설 결정)
-- 실측 (2026-05-21): card_news 에 validation_pass(bool), validation_sc_score(double) 만 존재.
-- 9 신규 score (rule-based 5 + llm-judge 4) 를 namespace 로 분리하여 저장하기 위해 신규 JSONB 컬럼.
ALTER TABLE card_news
    ADD COLUMN IF NOT EXISTS evaluation_payload JSONB NOT NULL DEFAULT '{}'::jsonb;
-- sidecar 의 미평가 카드 SELECT 가 partial index 로 빠르게 동작 (운영 카드만 평가 대상)
CREATE INDEX IF NOT EXISTS idx_card_news_unjudged_v2
    ON card_news(created_at DESC)
    WHERE card_schema_version = 'v2'
      AND (evaluation_payload ? 'llm_judge') = false;
COMMENT ON COLUMN card_news.evaluation_payload IS
    'W5 evaluation results. Keys: rule_based (W5-1, 5 metric, in-graph), llm_judge (W5-2, 4 score, sidecar). v1 카드는 항상 빈 JSONB.';

-- (b) Layer 2 인덱스 3개 (§3.4.3)
CREATE INDEX IF NOT EXISTS idx_card_news_peer_created
    ON card_news(peer_company_id, created_at DESC)
    WHERE peer_company_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_business_signals_peer_area_period
    ON raw_article_business_signals(peer_id, business_area, period_year DESC, period_quarter DESC);
CREATE INDEX IF NOT EXISTS idx_financial_metrics_peer_metric_period
    ON raw_article_financial_metrics(peer_id, metric_name, period_year DESC, period_quarter DESC);

-- (c) Layer 2-A timeline VIEW
CREATE OR REPLACE VIEW peer_event_timeline AS ... ;  -- §3.4.3

-- (d) Layer 2-C sector_pulse MATERIALIZED VIEW
CREATE MATERIALIZED VIEW IF NOT EXISTS sector_pulse AS ... ;  -- §3.4.3
CREATE UNIQUE INDEX IF NOT EXISTS uq_sector_pulse_sector_week
    ON sector_pulse(sector, week_start);

-- (e) Layer 2-D financial trend VIEW
CREATE OR REPLACE VIEW peer_financial_trend AS ... ;  -- §3.4.3

-- (f) peer_companies.peer_plus_payload JSONB key namespace 코멘트
COMMENT ON COLUMN peer_companies.peer_plus_payload IS
    'Namespaces: profile_snapshot (W2-2 weekly), capability_evolution (W4 monthly), snapshot_archive_ref. See design/01-supervisor-implementation-plan.md §3.4.3.';
```

**파일 변경**:
```text
axis-backend/src/main/resources/db/migration/V33__context_engineering.sql   (신규)
axis-infra/db/schema.sql                                                     (DDL 반영)
```

#### W4-2. `AnalysisContextBuilder` (cluster-time, LLM 없음) ★ P3-1

```text
src/services/analysis_context_builder.py        (신규 ~350 LoC)
src/analysis/models.py                          (AnalysisContext / TimelineEntry / CapabilityWindow / SectorPulseRow / FinancialSeries / PrecedentCandidate / RetrievedCard / ContextProvenance)
src/db/article_store.py                         (query_timeline / query_capability_evolution / query_sector_pulse / query_financial_trend / query_precedents 추가)
src/pipeline/supervisor_graph.py                (_build_analysis_context_node)
tests/test_analysis_context_builder.py          (신규)
tests/golden/context/
    high_density_peer.json
    sparse_peer.json
    budget_overflow.json
```

token budget 검증: 각 layer 별 token 추정값 측정 → `_compress_to_budget` 의 우선순위가 일관되게 동작.

#### W4-3. `CapabilityEvolutionAgent` (월 1회 CronJob, LLM ✅) ★ P3-2

```python
# src/agents/context/capability_evolution_agent.py (신규)
class CapabilityEvolutionAgent:
    """월 1회. peer 별 raw_article_business_signals 4분기 분 → LLM 합성 narrative."""

    prompt_version = "capability-v1.0"
    _LLM_MODEL = "gpt-4o"

    def run(self, *, peer_id: str, lookback_quarters: int = 4) -> dict:
        signals = _load_business_signals(peer_id, quarters=lookback_quarters)
        if len(signals) < 5:
            return {"skipped": True, "reason": "insufficient_signals"}
        narrative = self._invoke_llm(_render_prompt(signals))
        return {
            "version": "capability-v1",
            "generated_at": now_iso(),
            "windows": [...],   # business_area 별 narrative
        }

    def persist(self, peer_id: str, result: dict) -> None:
        # peer_companies.peer_plus_payload['capability_evolution'] = result
        _upsert_peer_plus_payload(peer_id, "capability_evolution", result)
```

```yaml
# axis-infra/k8s/base/cronjob-capability-evolution.yaml (신규)
schedule: "0 3 1 * *"
command: ["python", "scripts/refresh_capability_evolution.py"]
```

**파일 변경**:
```text
src/agents/context/capability_evolution_agent.py        (신규)
src/agents/context/__init__.py                          (신규)
src/analysis/prompts/capability_v1.py                   (신규)
scripts/refresh_capability_evolution.py                 (신규)
axis-infra/k8s/base/cronjob-capability-evolution.yaml   (신규)
axis-infra/k8s/base/kustomization.yaml                  (cronjob 추가)
tests/test_capability_evolution_agent.py                (신규, mocked LLM)
```

LLM 비용: 5 peer × 1회/월 × ~$0.15 = **$0.75/월** = $0.025/일.

#### W4-4. `SectorPulseAggregator` REFRESH CronJob (LLM 없음) ★ P3-3

```yaml
# axis-infra/k8s/base/cronjob-sector-pulse.yaml (신규)
schedule: "0 2 * * 1"     # 매주 월 02:00 KST
command:
  - psql
  - -h
  - $(POSTGRES_HOST)
  - -c
  - "REFRESH MATERIALIZED VIEW CONCURRENTLY sector_pulse;"
```

`peer_event_timeline` 과 `peer_financial_trend` 는 VIEW 라 REFRESH 불필요.

**파일 변경**:
```text
axis-infra/k8s/base/cronjob-sector-pulse.yaml          (신규)
axis-infra/k8s/base/kustomization.yaml                 (cronjob 추가)
```

LLM 비용: $0.

#### W4-5. ImplicationAgent prompt v4.0 → v5.0 (AnalysisContext 입력 추가) ★ P3-1

- W1-1 의 ImplicationAgent 가 `AnalysisContext` 도 입력으로 받도록 signature 확장.
- 프롬프트에 §3.3 의 W4 이후 룰 (precedent_link, capability_change 인용, sector pulse anomaly 반영) 추가.
- `ImplicationProvenance.used_context_layers` 필드 활성화.

**파일 변경**:
```text
src/agents/implication_agent.py                  (generate(..., analysis_context) 추가)
src/analysis/prompts/implication_v5.py           (신규 — v4.0 base + context block)
src/pipeline/supervisor_graph.py                 (_implication_node 가 state['analysis_context'] 전달)
tests/test_implication_agent_v5.py               (신규)
tests/golden/implication_v5/
    with_precedent.json
    with_capability_change.json
    sector_anomaly.json
    no_context_fallback.json     # AnalysisContext 비어있을 때 v4.0 동작 확인
```

#### W4-6. (옵션) `EventChainDiscoveryAgent` — 4주 운영 측정 후 결정 ★ P3-4

W4-1 ~ W4-5 운영 4주 후 측정:

| 지표 | 임계값 | → 결정 |
|---|---|---|
| `precedent_link` 인용 빈도 | ≥ 30% 카드 | (충분) JSONB 유지 |
| `precedent_link` 인용 빈도 | < 10% 카드 | (불충분) EventChainDiscoveryAgent + V34 `event_chain_links` 도입 |

→ MVP 에서는 `card_news.evidence_payload['related_card_ids']` JSONB 에 candidate 만 저장. ImplicationAgent 가 link 종류 판단.

LLM 비용 (도입 시): 약 **$0.30/일** = $9/월.

---

### W5 — Evaluation & Observability Layer (W4 직후 또는 W4 와 병렬, 12-18h)

> 출력 품질 정량화. Hybrid 배치: rule-based 는 in-graph (`validate` 노드 확장), LLM-as-Judge 는 sidecar CronJob.

#### W5-1. `validate` 노드 확장 — Rule-based 5 metric ★ Outcome Quality (4-6h)

> 기존 `validate` 노드 (W2-3, 단정표현·수치 차단) 에 5 metric 계산 로직 추가. LLM 호출 없음.

**작업 1. `agents/evaluator_agent.py` 신규** — 5 metric 계산기

```python
@dataclass(slots=True)
class EvaluationMetrics:
    context_hit_ratio: float        # used_context_layers / 6
    evidence_claim_ratio: float     # grounded numeric/date claims / total
    specificity_score: float        # peer/business_area 명시 정도
    actionability_score: float      # verb-first + 구체성
    regression_drift: float | None  # rolling 7d 대비 confidence delta (None 가능)
    
class EvaluatorAgent:
    def evaluate(
        self,
        implication: ImplicationResult,
        evidence_payload: dict,
        analysis_context: AnalysisContext | None,
        rolling_avg: float | None,
    ) -> EvaluationMetrics: ...
```

**작업 2. `pipeline/supervisor_graph.py` — `validate` 노드 확장**

기존 validate 후 `EvaluatorAgent.evaluate()` 호출. **v3.2.1: Phase 1 (4 metric, threshold X) / Phase 2 (drift + threshold 활성화, 운영 14일 후)**.

```python
def _validate_node(state: SupervisorState) -> SupervisorState:
    report = _hard_validate(...)
    if report.passed:
        # Phase 1: 항상 4 metric 계산. drift 는 baseline 7일 미만이면 None.
        report.metrics = EvaluatorAgent().evaluate(
            implication=state["implication"],
            evidence_payload=state["analysis_package"].evidence_payload,
            analysis_context=state["analysis_context"],
            rolling_avg=_load_rolling_confidence_7d_or_none(),
            available_layer_count=_count_available_layers(state["analysis_context"]),  # v3.2.1
        )
        # Phase 2 (운영 14일 후 활성화): calibrated threshold 로 flag.
        # Phase 1 에서는 threshold=None 이므로 flag 추가 절대 X (사용자 결정 — lenient 시작).
        thresholds = _load_calibrated_thresholds()  # weekly CronJob 이 갱신, 부재 시 None
        if thresholds is not None:
            for metric_name, lower in thresholds.items():
                if getattr(report.metrics, metric_name, None) is not None \
                        and getattr(report.metrics, metric_name) < lower:
                    state["human_review_flags"].append(f"low_{metric_name}")
    return {**state, "validation": report}
```

**작업 3. `card_writer` 노드에서 INSERT 시 `evaluation_payload['rule_based']` 채움**

**파일 변경**:
```text
src/agents/evaluator_agent.py               (신규, ~250 LoC)
src/pipeline/supervisor_graph.py            (_validate_node 확장, _card_writer_node INSERT 갱신)
tests/test_evaluator_agent.py               (신규 — 5 metric 각각의 golden 케이스)
```

#### W5-2. `axis-cron-card-evaluator` Sidecar — LLM-as-Judge ★ Outcome Quality (6-8h)

> Cluster 처리 critical path 와 독립. Frontend 사용자에게 가시화되는 카드 latency 영향 0.

**작업 1. `scripts/evaluate_recent_cards.py` 신규** — Sidecar 메인

```python
def main() -> None:
    rows = _select_unjudged_cards(limit=20, hours=24)  # evaluation_payload->'llm_judge' IS NULL
    for row in rows:
        judgment = _llm_judge(
            implication=row["implication"],
            evidence_payload=row["evidence_payload"],
        )  # gpt-4o-mini, structured output
        _update_card_evaluation(row["id"], judgment)
```

**작업 2. `agents/llm_judge_prompts.py` 신규** — 프롬프트 4종

각 score 별 prompt 분리. structured output (`json_mode`) 으로 `{score: 0-5, reasoning: str}` 강제.

**작업 3. `axis-infra/k8s/cronjobs/axis-cron-card-evaluator.yaml` 신규**

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: axis-cron-card-evaluator
spec:
  schedule: "*/5 * * * *"   # 5분 주기
  concurrencyPolicy: Forbid  # 중복 실행 방지
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: evaluator
              image: registry/axis-ai:latest
              command: ["python", "-m", "scripts.evaluate_recent_cards"]
              env:
                - name: EVAL_BATCH_LIMIT
                  value: "20"
              resources:
                requests: { cpu: "100m", memory: "256Mi" }
                limits:   { cpu: "500m", memory: "1Gi" }
          restartPolicy: OnFailure
```

**파일 변경**:
```text
scripts/evaluate_recent_cards.py            (신규, ~350 LoC)
src/agents/llm_judge_prompts.py             (신규, ~150 LoC — 4 score prompts)
axis-infra/k8s/cronjobs/axis-cron-card-evaluator.yaml  (신규)
tests/test_llm_judge.py                     (신규 — mock LLM, golden 3 카드)
```

#### W5-3. (옵션, Phase 2) Regression Detection — Daily CronJob ★ 운영 자동화 (2-3h)

> **운영 14일 후 활성화** (v3.2.1 Phase 분리). 매일 02:30 KST 에 최근 7d vs 직전 7d 의 평균 점수 비교, **-15% 이상** 감소 시 Slack/email 알림 (v3.2.1: 초기 threshold 를 -10% → -15% 로 완화 — 5/14 같은 backfill day 가 false-alarm 일으키지 않도록).

```sql
-- daily regression check (CronJob 내부)
WITH weekly AS (
  SELECT DATE_TRUNC('week', created_at) AS wk,
         AVG((evaluation_payload->'rule_based'->>'actionability_score')::float) AS act,
         AVG((evaluation_payload->'llm_judge'->>'faithfulness')::float)         AS faith,
         AVG((evaluation_payload->'llm_judge'->>'specificity_llm')::float)      AS spec
  FROM card_news
  WHERE created_at >= NOW() - INTERVAL '14 days'
  GROUP BY 1
)
SELECT
    (curr.act - prev.act)     / NULLIF(prev.act, 0)     AS act_delta,
    (curr.faith - prev.faith) / NULLIF(prev.faith, 0)   AS faith_delta,
    (curr.spec - prev.spec)   / NULLIF(prev.spec, 0)    AS spec_delta
FROM (SELECT * FROM weekly ORDER BY wk DESC LIMIT 1) curr
CROSS JOIN (SELECT * FROM weekly ORDER BY wk DESC OFFSET 1 LIMIT 1) prev;
-- delta < -0.10 → Slack 알림
```

**파일 변경**:
```text
scripts/eval_regression_check.py            (신규, ~120 LoC)
axis-infra/k8s/cronjobs/axis-cron-eval-regression.yaml  (신규)
```

→ W5-3 은 **W5-2 운영 1주 후** 도입 권장 (baseline 데이터 누적 필요).

---

## 5. 의존성 & 일정

| ID | 작업 | 시간 | 의존 | 담당 (제안) |
|---|---|---|---|---|
| W1-1 | ImplicationAgent v4.0 + 프롬프트 | 6-8h | — | AI Lead |
| W1-2 | AnalysisPackage schema 정합화 | 2-3h | — | AI Lead |
| W1-3 | AnalysisInputMetadata typed | 1-2h | — | AI Eng |
| W1-4 | card_news v2 schema lint + CardNewsAgent INSERT 변경 | 1-2h | **W4-1 V33 머지 필수** | AI Eng |
| W2-1 | Supervisor LangGraph 화 + CardNews 이관 (작업 5) | 8-11h | W1-1, W1-2 | AI Lead |
| W2-2 | Profile 2-tier + V33 + CronJob | 6-8h | (V33 Flyway 머지) | AI Eng B |
| W2-3 | Validate 노드 | 3-4h | W2-1 | AI Eng A |
| W2-4 | CardNewsAgent 정리 | 1-2h | W1-1 | AI Lead |
| W3-1 | design docs 5종 | 4-6h | W2 완료 | PM |
| W3-2 | `_deprecated/` → `_archived/` | 30m | W1-1 | any |
| W3-3 | Golden + 단위 테스트 보강 | 3-4h | W2 완료 | AI Eng A |
| W3-4 | Langfuse 전수 | 2-3h | W2-1 | AI Eng A |
| W3-5 | 운영 절차 문서화 | 1-2h | W2-2 | PM |
| W4-0 | Data hygiene precondition (peer alias / metric canonical / cluster identity / DQ checks) | 3-4h | — | AI Eng B |
| W4-1 | V33 인덱스+VIEW+MV (+ W1-4 card_schema_version 통합) | 2-3h | W4-0 | AI Eng B |
| W4-2 | AnalysisContextBuilder | 6-8h | W4-0, W4-1, W2-1 | AI Lead |
| W4-3 | CapabilityEvolutionAgent + CronJob (chunked input) | 4-5h | W4-0, W4-1 | AI Eng B |
| W4-4 | SectorPulse REFRESH CronJob (Phase 1 only) | 1h | W4-1 | AI Eng B |
| W4-5 | ImplicationAgent prompt v5.0 (AnalysisContext 입력) | 3-4h | W4-2 | AI Lead |
| W4-6 | (옵션) EventChainDiscoveryAgent — embedding 기반 | 5-6h | W4 운영 4주 측정 후 결정 | (보류) |
| **W5-1** | **`validate` 노드 확장 — Rule-based 4 metric (Phase 1) + calibration script** | **5-7h** | **W2-3, W4-5** | **AI Eng A** |
| **W5-2** | **`axis-cron-card-evaluator` Sidecar — LLM-as-Judge 4 score + cost cap + v2 guard** | **7-9h** | **W4-5, V33 evaluation_payload 컬럼** | **AI Lead** |
| **W5-3** | **(옵션, Phase 2) Regression Detection CronJob + drift metric 활성화 + cross-check sampling** | **3-4h** | **W5-2 운영 14일 후** | **AI Eng B** |

**총 예상**: ~88h (W1~W3 50h + W4 21h + W5 12-16h + W4-6/W5-3 옵션 9h). 실 작업 1주 ~16h 가정 → **5-5.5주**.
Critical path: W1-1 → W2-1 → W2-3 → W4-0 → W4-2 → W4-5 → **W5-1** → W3-3 ≈ **39-41h**.

**병렬화 가능 경로**:
- W4-0 (data hygiene) 은 W1-1/W1-2 와 병렬 시작 가능 (의존 없음).
- W4-1 (V33) 은 W4-0 직후 가능.
- W4-3 (CapabilityEvolution CronJob) 은 W4-1 만 끝나면 W2 완료 전에도 가능.
- W4-4 (SectorPulse REFRESH) 는 W4-1 직후 바로.
- **W5-2 (sidecar) 는 W5-1 과 무관하게 W4-5 직후 시작 가능** — 둘 다 카드의 implication 출력만 입력으로 받음.

---

## 6. 영향 예측

### 6.1 비용

**W4-0 (data hygiene) 은 비용 영향 0** — Python utility 만, LLM 호출 없음.

| 항목 | 현재 | W1 완료 | W2 완료 | W3 완료 | W4 완료 (W4-0~5) | W5 완료 (W5-1~2) | W4-6 도입 시 |
|---|---|---|---|---|---|---|---|
| Cluster 당 LLM 호출 (Stage 1) | 4 | 5 (+implication) | 5 (retry 평균 1.05회) | 5 | 5 (context는 LLM X) | 5 (validate 확장 LLM X) | 5 |
| Cluster 당 비용 (GPT-4o) | $0.078 | $0.100 | $0.105 | $0.105 | $0.115 (prompt 길이 ↑) | $0.115 | $0.115 |
| 일일 Stage 1 (15 clusters) | $1.17 | $1.50 | $1.58 | $1.58 | $1.73 | $1.73 | $1.73 |
| Profile snapshot CronJob (주1회) | — | — | $0.11/일 | $0.11/일 | $0.11/일 | $0.11/일 | $0.11/일 |
| CapabilityEvolution (월1회) | — | — | — | — | $0.025/일 | $0.025/일 | $0.025/일 |
| SectorPulse REFRESH | — | — | — | — | $0 | $0 | $0 |
| **CardEvaluatorSidecar (W5-2)** | — | — | — | — | — | **$1.50/일** (30 카드 × $0.05, gpt-4o-mini) | $1.50/일 |
| EventChainDiscovery (옵션, 매일) | — | — | — | — | — | — | $0.30/일 |
| **일일 총합** | **$1.17** | **$1.50** | **$1.69** | **$1.69** | **$1.87** | **$3.37** | **$3.67** |

`INFRASTRUCTURE_PLAN.md` 한도 ₩10,000/일 (~$7.4/일) 대비:
- W4 완료: **25% 사용**
- W5 완료: **46% 사용** (sidecar 평가 비용 +$1.5/일)
- W4-6 도입 시: **50% 사용**

여유 있음. **W5-1 (rule-based) 은 비용 영향 0** — LLM 호출 없음.

### 6.2 품질·운영 지표

| 지표 | 현재 (5/20) | W1 후 | W2 후 | W3 후 | W4 후 | **W5 후** |
|---|---|---|---|---|---|---|
| 시사점 LLM 사용 | ❌ heuristic | ✅ gpt-4o | ✅ + 풍부한 context | ✅ + tracing | ✅ + 시계열 context | ✅ + outcome eval |
| Cluster 처리 시간 (p50) | ~30초 | ~35초 | ~32초 (retry 효율) | ~32초 | ~33초 (+ context build ~1초) | ~33초 (W5-1 rule-based 추가 <50ms) |
| LLM 실패 시 cluster 손실률 | 100% | 100% | ~10% (retry + fallback) | ~10% | ~10% | ~10% |
| `evidence_label="sufficient"` 비율 | — | 측정 가능 | 목표 ≥ 60% | 목표 ≥ 70% | 목표 ≥ 75% | 목표 ≥ 75% |
| 출처 없는 수치 carryover | (검증 X) | (검증 X) | 0 (validate 차단) | 0 | 0 | 0 (+ evidence_claim_ratio 측정) |
| 단위 테스트 커버리지 (1단계) | 0% | 30% | 50% | ≥ 70% | ≥ 75% | ≥ 78% |
| design doc 정합도 | 부분 | 부분 | 부분 | 100% | 100% (+ context layer doc) | 100% (+ eval layer doc) |
| `human_review` 라우팅 | (없음) | (없음) | 활성 | + Frontend 노출 후속 | + frontend | + quality score 기반 자동 flag |
| 카드뉴스 3섹션 분리 (요약/시사점/대응) | ❌ summary only | 🟡 implication JSONB 안에 있음 | 🟡 v1 | ✅ v2 schema 정합 | ✅ + precedent 인용 | ✅ + 품질 score 동반 |
| `recommended_actions` 비어있지 않은 비율 | (없음) | 측정 가능 | 목표 ≥ 80% | 목표 ≥ 85% | 목표 ≥ 90% | 목표 ≥ 90% |
| `precedent_link` 인용 비율 (W4 신지표) | — | — | — | — | 측정 시작, 목표 ≥ 30% | 측정 + drift 추적 |
| `used_context_layers` 평균 개수 | — | — | — | — | 측정 시작, 목표 ≥ 3 | 측정 + ROI 추적 |
| **W5 신규 4 rule-based metric (Phase 1, 즉시)** | | | | | | |
| `context_hit_ratio` 평균 (분모=available_layer_count) | — | — | — | — | — | **첫 1주 측정만** → 1주 후 calibrated target |
| `evidence_claim_ratio` 평균 | — | — | — | — | — | **첫 1주 측정만** → 1주 후 calibrated target |
| `specificity_score` 평균 | — | — | — | — | — | **첫 1주 측정만** → 1주 후 calibrated target |
| `actionability_score` 평균 (한국어 verb-suffix pattern) | — | — | — | — | — | **첫 1주 측정만** → 1주 후 calibrated target |
| **W5 신규 metric (Phase 2, 운영 14일 후)** | | | | | | |
| `regression_drift` week-over-week | — | — | — | — | — | 목표 \|delta\| < 0.15 |
| **W5 신규 4 LLM-judge score (0-5)** | | | | | | |
| `faithfulness` 평균 | — | — | — | — | — | 목표 ≥ 4.0 |
| `specificity_llm` 평균 | — | — | — | — | — | 목표 ≥ 3.5 |
| `actionability_llm` 평균 | — | — | — | — | — | 목표 ≥ 3.5 |
| `peer_relevance` 평균 | — | — | — | — | — | 목표 ≥ 3.8 |

### 6.3 위험 요소 & 완화

| 리스크 | 가능성 | 완화 |
|---|---|---|
| 새 ImplicationAgent 가 hallucination 더 함 | 중 | validate 노드의 regex 검증 + golden test |
| LangGraph 마이그레이션 시 회귀 | 중 | W1-3 golden 으로 cluster sample 5개 회귀 잡음 |
| Profile snapshot CronJob 실패 → context 비어있음 | 낮 | snapshot 부재 시 build_context_v2 가 static-only fallback |
| 기존 카드 183건 새 schema 와 안 맞음 | 중 | `card_schema_version='v1'` 유지 + frontend graceful fallback (§3.5.4 옵션 A) |
| LLM rate limit 증가 (cluster 당 +1 호출) | 낮 | OpenAI 5,000 RPM 한도 대비 일일 75 calls → 2% 사용 |
| V33 Flyway 가 cluster DB 에 silent migrate | 중 | local Mode B 에서만 검증 후 PR (axis-backend README 정책 준수) |
| W4 context block 이 prompt 길이 폭증 → cost ↑ | 중 | `AnalysisContextBuilder._compress_to_budget` 가 4,000 token 강제 |
| MATERIALIZED VIEW `sector_pulse` 누락 시 ImplicationAgent context 비어있음 | 낮 | builder 가 `try/except` 로 layer 별 graceful — 1개 layer 실패가 전체 중단 안 됨 |
| CapabilityEvolutionAgent LLM 출력이 raw signal 인용 안 함 | 중 | 프롬프트가 `evidence_signal_ids` 출력 강제 + validate 단계에서 빈 list 시 retry 1회 |
| 카드뉴스 frontend 가 v1/v2 mixed 표시 | 낮 | openapi `CardNewsResponse.card_schema_version` 명시 + frontend conditional render |
| **v3.1 추가** — `card_news.cluster_id` 의 ephemeral 성질로 인한 잘못된 precedent 추출 | 높 | W4-0 `cluster_identity.py` 에서 `source_raw_article_ids` 또는 `(peer, event_type, date)` 만 사용. cluster_id 사용 금지 |
| `source_raw_article_ids` 빈 15% 카드의 provenance chain 끊김 | 중 | `_data_quality_checks.is_card_provenance_traceable` 로 fallback (sources / evidence_payload.source_links 활용) + 카드 생성 단 (Layer B `CardNewsAgent`, As-Is `ingestion_graph.card_news_node`) 에서 source 강제 검증 — W2-1 LangGraph 재구성에 흡수 |
| `period_quarter` NULL 25% 합 (signals 7.4% + metrics 17.8%) 으로 시계열 그룹화 손실 | 중 | `COALESCE(period_quarter::text, 'annual')` fallback + period 텍스트 보조 키 |
| `matched_companies` peer_id 표기 불일치로 query 누락 | 높 | W4-0 `peer_id_aliases.py` 가 모든 query 의 single source of truth. 신규 peer 추가 시 PR 강제 |
| **CapabilityEvolution LLM input 170k token 초과** (peer 당 4분기 raw text) | 높 | W4-3 의 SQL pre-aggregation (그룹별 top-5 confidence) — 평균 10-20k token 으로 압축 |
| sector_pulse z-score 가 2주 baseline 으로 무의미 | 중 | Phase 1 (count + intensity 만) → 4주 후 Phase 2 (anomaly) 분리 도입 |
| event_chain 이 cluster duplicate 와 구분 안 됨 | 높 | W4-6 옵션 보류. 도입 시 Qdrant cosine ≥ 0.75 + 시간차 ≥ 7일 강제 |
| 191 기존 v1 카드의 v2 backfill 비용 | 낮 | $4.2 + 4h. graceful fallback 으로 시작, W4 종료 후 결정 |
| `peer_companies.peer_plus_payload` 가 모든 peer 에 비어있음 | 낮 | W2-2 profile_snapshot CronJob 이 채움. 첫 CronJob 실행 전엔 static-only fallback |
| **v3.1 갱신** — 이번 주 신규 카드 `peer_company_id` FK NULL (P3-CRIT-4) | 낮 | **데이터 손실 X — `card_news.company` 와 `raw_articles.matched_companies` 모두 정상**. W4-0 첫 마이그레이션에서 백필 SQL 2개로 즉시 복구. 회귀 추적은 axis-ai `CardNewsAgent` 후처리의 5/14~5/20 변경 이력 (별도 P1 ticket). |
| V33 SQL dry-run 시 cluster 데이터 무결성 | 낮 | **이미 dry-run 통과 (§10.1)** — `BEGIN; ... ROLLBACK;` 으로 prod schema 에 직접 시뮬레이션 검증 완료 |

---

## 7. 검증 기준 (Done Definition)

각 단계 종료 시 아래 모두 통과해야 PR 머지.

### W1 Done
- [ ] `pytest tests/test_implication_agent.py -v` 전부 PASS
- [ ] Golden file 3종 (`with_capability_change`, `without_capability_change`, `evidence_insufficient`) 동일 hash
- [ ] `analysis/models.py` 의 dataclass 가 실 LLM 출력 키와 1:1 매칭 (`mypy` strict pass)
- [ ] `mypy src/agents/implication_agent.py` 0 errors
- [ ] cluster 1개 dry-run (`run_pipeline_once.py --cluster <id>`) 에서 `implication.provenance.prompt_version == "implication-v4.0"`

### W2 Done
- [ ] Supervisor graph cluster sample 5개 e2e (mock LLM) PASS — 마지막 노드까지 (`card_news_id` non-null) 도달
- [ ] `kubectl get cronjob axis-cron-profile-refresh -n skala3-finalproj-class3-team13` 존재 + 1회 manual trigger 성공
- [ ] V33 migration 이 Mode B (local docker) 에서 PASS, cluster prod profile 에서 자동 적용 검증
- [ ] `peer_companies.profile_snapshot` 5 행이 모두 non-empty (4 peer + sk_ax)
- [ ] Validate 노드가 일부러 만든 단정 표현 카드 1건 차단 확인 (`human_review_flags` 추가됨)
- [ ] `pipeline_logs` 에 supervisor 노드 7개 (profile_context / build_analysis_context / issue_integrate / strategic_analyze / implication / validate / **card_writer**) 모두 elapsed_ms 기록됨
- [ ] **W2-1 작업 5**: `ingestion_graph.card_news_node` 가 제거됨 (또는 thin shim 만 잔존), cluster sample 카드 INSERT 가 supervisor 의 `card_writer` 노드에서 일어남 — Langfuse trace 또는 `pipeline_logs.step='card_writer'` 로 확인

### W3 Done
- [ ] `design/30-analysis/{00..04}.md` 5종 존재 + `00-supervisor-topology.md` 와 cross-link
- [ ] `_deprecated/` 디렉토리 부재 (또는 `_archived/` 로 rename + import-block)
- [ ] `tests/golden/supervisor/` 3 cluster 모두 동일 output 재현
- [ ] Langfuse UI 에서 cluster trace 1개에 7 노드 (W4 의 `build_analysis_context` 포함) 모두 보임 (session_id = bundle_id)
- [ ] `evidence_label="sufficient"` 비율 측정 SQL 추가 + 첫 측정값 기록

### W4 Done
- [ ] **W4-0**: `peer_id_aliases.py` / `metric_canonical.py` / `cluster_identity.py` / `_data_quality_checks.py` 4 모듈 + `tests/test_data_hygiene.py` 모두 mypy strict PASS
- [ ] **W4-0**: 5 peer 정규 ID 각각에 대해 `expand_peer_aliases()` 가 ≥ 3 alias 반환 (자기 자신 포함)
- [ ] **W4-0**: `canonicalize_metric()` 이 `net_income`, `당기순이익`, `순이익` 모두 `"net_income"` 으로 매핑
- [ ] **W4-0**: V32_5 백필 SQL 적용 후 `card_news WHERE created_at >= '2026-05-15'` 의 `peer_company_id IS NULL` = 0, `primary_keyword_category IS NULL` = 0 (검증 SQL §10.1 step 5-2 와 동일)
- [ ] V33 마이그레이션이 Mode B (local docker) 에서 PASS, cluster 에 적용 검증
- [ ] `peer_event_timeline` VIEW + `general_event_timeline` VIEW + `peer_financial_trend` VIEW + `sector_pulse` MATERIALIZED VIEW 존재
- [ ] `peer_financial_trend.metric_name_canonical` 컬럼에서 동일 metric 의 표기 다른 row 들이 합산됨 (실측: net_income 22+22+17 → 동일 그룹)
- [ ] `card_news.card_schema_version` 컬럼 존재, 신규 INSERT 가 `'v2'` 로 기록
- [ ] `AnalysisContextBuilder.build()` 가 4,000 token 이내 보장 (golden 3종 PASS: high_density / sparse / budget_overflow)
- [ ] `AnalysisContextBuilder.evidence_density_per_peer` 가 LG CNS 에 대해 `"sparse"` 라벨 반환 (실측 빈약 확인)
- [ ] Supervisor LangGraph 에 `build_analysis_context` 노드 wired, pipeline_logs 에 elapsed_ms 기록 (p95 ≤ 2초)
- [ ] `axis-cron-capability-evolution` CronJob 존재 + 1회 manual trigger 성공, 5 peer 의 `peer_plus_payload['capability_evolution']` non-empty
- [ ] CapabilityEvolutionAgent 의 LLM input token count ≤ 25k / call (chunked pre-aggregation 효과 검증)
- [ ] `axis-cron-sector-pulse` CronJob 존재 + REFRESH 1회 성공, `sector_pulse` 0행 이상. anomaly 컬럼은 Phase 2 (4주 후) 활성화 명시
- [ ] ImplicationAgent prompt v5.0 적용, `implication.provenance.used_context_layers` 비어있지 않음 (`≥ 3`)
- [ ] cluster sample 5개에서 `precedent_link` 또는 `capability_change` 인용 ≥ 2건
- [ ] `evidence_density_per_peer` 가 `"sparse"` 인 peer 에 대해 ImplicationAgent 가 `evidence_label="moderate"` 또는 `"insufficient"` 로 자동 강등
- [ ] (옵션) W4-6 도입 여부 결정 (4주 운영 측정 SQL + 의사결정 기록 doc)

### W5 Done

**W5 Phase 1 — Rule-based 4 metric + Sidecar 4 score (즉시 도입)**

- [ ] **V33 마이그레이션 (W4-1 통합)**: `card_news.evaluation_payload JSONB DEFAULT '{}'::jsonb` 컬럼 + `idx_card_news_unjudged_v2` partial index 존재
- [ ] **W5-1**: `agents/evaluator_agent.py` 의 4 metric (context_hit_ratio / evidence_claim_ratio / specificity_score / actionability_score) 각각 golden 케이스 PASS — mypy strict
- [ ] **W5-1**: `context_hit_ratio` 분모가 `available_layer_count` 인지 unit test 검증 (신규 peer 케이스에서 6 고정 페널티 없음)
- [ ] **W5-1**: `actionability_score` 의 한국어 verb-suffix 사전 (`한다 / 할 것 / 검토 / 착수 / ...`) 매칭 unit test PASS — "디지털 전환을 가속화한다" 가 verb-match, "디지털 전환 가속화" 는 no-match
- [ ] **W5-1**: `validate` 노드가 metric 계산 후 `state.validation.metrics` 에 저장, `card_writer` 가 `evaluation_payload['rule_based']` 로 INSERT — cluster sample 5개 모두 `rule_based` non-null
- [ ] **W5-1**: 운영 첫 7일 동안 `human_review_flags` 추가 0건 (lenient threshold 정책 검증 — 사용자 결정 "결과가 너무 적게/제한되지 않게")
- [ ] **W5-1**: validate 노드 elapsed_ms p95 ≤ 100ms
- [ ] **W5-2**: `axis-cron-card-evaluator` CronJob 존재 + 1회 manual trigger 성공, 평가 대상 카드 ≥ 5건의 `evaluation_payload['llm_judge']` non-null
- [ ] **W5-2**: 4 score (faithfulness / specificity_llm / actionability_llm / peer_relevance) 모두 0-5 범위 + reasoning non-empty
- [ ] **W5-2**: SELECT 가 `card_schema_version = 'v2'` AND `NOT (evaluation_payload ? 'llm_judge')` 가드로 v1 카드 191건 자동 skip 검증 (sidecar 로그에서 v1 평가 시도 0건)
- [ ] **W5-2**: LLM-as-Judge 가 의도적 hallucination 카드 (수치 조작) 에 대해 `faithfulness ≤ 2.0` 부여 (golden test)
- [ ] **W5-2**: sidecar 1회 실행이 cluster 처리 시간에 영향 X — `pipeline_logs.step='supervisor_graph'` 와 별개 step 으로 기록
- [ ] **W5-2**: sidecar 실패가 카드 가시성에 영향 X — `NOT (evaluation_payload ? 'llm_judge')` 카드도 frontend 에 정상 표시 (graceful fallback)
- [ ] **W5-2**: 일일 LLM 비용 (sidecar) ≤ $1.50 — cost cap $5/일 도달 시 Slack 알림 검증 (의도적 backfill 100건 → cap trigger 시뮬레이션)
- [ ] **W5-2**: Sidecar burst 대응 — 미평가 카드 ≥ 100건 시뮬레이션 시 batch limit 동적 증가 + Slack 알림 검증

**W5 Phase 2 — Calibration + drift + cross-check (운영 14일 후)**

- [ ] **Threshold calibration**: 운영 1주 데이터로 4 metric 각각의 bottom-10% percentile 자동 산출, weekly CronJob 으로 갱신 — 산출값이 NULL 이거나 비현실적 (예: `actionability_score < 0.05`) 이면 fallback (전역 floor 적용)
- [ ] **W5-3 (옵션, Phase 2)**: regression CronJob 적용 후 daily SQL 1회 실행, delta 0 (baseline 동일주 비교) 확인
- [ ] **W5-3 (옵션)**: 일부러 회귀 시뮬레이션 (`actionability_score` 평균 -20% 데이터 INSERT) → Slack/email 알림 트리거 검증
- [ ] **W5-3 (옵션, Phase 2)**: `regression_drift` metric 활성화 — 운영 14일 미만 카드는 `null`, 이후 카드만 실 값 검증
- [ ] **Self-eval bias (Phase 2, §3.6.5)**: 주간 10% sampling cross-check (Claude) 도입, 두 evaluator 의 `faithfulness` gap < 1.0 인지 weekly report 생성

---

## 8. 사후 모니터링 (post-W3/W4)

| 지표 | 측정 SQL / Source | 목표 |
|---|---|---|
| `evidence_label="sufficient"` 비율 | `SELECT COUNT(*) FILTER (WHERE implication->>'evidence_label'='sufficient')::float / COUNT(*) FROM card_news WHERE card_schema_version='v2'` | ≥ 70% (W3) / ≥ 75% (W4) |
| Cluster 처리 시간 (p50/p95) | `pipeline_logs` step='supervisor_graph' | p95 ≤ 45s |
| `numeric_violations` 발생률 | `pipeline_logs` step='validate' WHERE input_count > 0 AND output_count = 0 | ≤ 5% / 일 |
| LLM 비용 일일 합계 | Langfuse cost dashboard | ≤ ₩2,500/일 |
| Profile snapshot freshness | `peer_companies.peer_plus_payload->'profile_snapshot'->>'generated_at'` | 모든 회사 ≤ 14일 |
| Capability evolution freshness (W4) | `peer_companies.peer_plus_payload->'capability_evolution'->>'generated_at'` | 모든 회사 ≤ 45일 |
| Sector pulse freshness (W4) | `(SELECT MAX(week_start) FROM sector_pulse)` | ≥ NOW() - 14일 |
| Context build time (W4) | `pipeline_logs` step='build_analysis_context' | p95 ≤ 2초 |
| `used_context_layers` 평균 (W4) | `AVG(jsonb_array_length(implication->'provenance'->'used_context_layers'))` | ≥ 3 |
| `precedent_link` 인용 비율 (W4) | `COUNT(*) FILTER (WHERE implication->'peer_implication'->'precedent_link' IS NOT NULL) / COUNT(*)` | ≥ 30% |
| `recommended_actions` 비어있지 않은 비율 | `COUNT(*) FILTER (WHERE jsonb_array_length(implication->'skax_implication'->'recommended_actions') > 0) / COUNT(*)` | ≥ 90% |
| 카드뉴스 v2 비율 | `COUNT(*) FILTER (WHERE card_schema_version='v2') / COUNT(*)` | W4 후 신규 카드 100% |
| **W5 rule-based** `context_hit_ratio` 평균 | `AVG((evaluation_payload->'rule_based'->>'context_hit_ratio')::float)` | ≥ 0.5 |
| **W5 rule-based** `evidence_claim_ratio` 평균 | `AVG((evaluation_payload->'rule_based'->>'evidence_claim_ratio')::float)` | ≥ 0.85 |
| **W5 rule-based** `actionability_score` 평균 | `AVG((evaluation_payload->'rule_based'->>'actionability_score')::float)` | ≥ 0.65 |
| **W5 LLM-judge** `faithfulness` 평균 | `AVG((evaluation_payload->'llm_judge'->>'faithfulness')::float)` | ≥ 4.0 (0~5) |
| **W5 LLM-judge** `peer_relevance` 평균 | `AVG((evaluation_payload->'llm_judge'->>'peer_relevance')::float)` | ≥ 3.8 (0~5) |
| **W5 회귀 drift** week-over-week | §3.6.3 의 SQL | \|delta\| < 0.10 |

---

## 9. 결정 사항 (v3.1 / v3.2 에서 확정)

| 항목 | 결정 |
|---|---|
| `_deprecated/implication_agent.py` 부활 여부 | **부활 안 함**. P.C.R.O 프롬프트 패턴만 참고, 입력 계약은 새로 (5종 input). |
| LangGraph 도입 | **도입**. ingestion_graph 와 일관성, retry / logging / human_review 라우팅 필수. |
| ProfileAgent cluster-time LLM | **호출 안 함**. CronJob 으로 분리 (Tier A) + DB-only enrichment (Tier B). |
| ImplicationAgent fallback | **현 heuristic `ImplicationGenerator` 유지**. LLM 실패 시 graceful degrade. |
| AnalysisPackage 외부 API | **불변**. Supervisor 내부만 리팩토링, `CardNewsAgent.generate_from_analysis_package()` 진입점 동일. |
| V33 Flyway 의 정체성 | **인덱스 3 + VIEW 3 + MATERIALIZED VIEW 1 + `card_news.card_schema_version` 컬럼 1 + W2-2 의 `peer_companies.peer_plus_payload` JSONB key namespace 표준화**. `profile_snapshot` 별도 컬럼 X → JSONB 내부. `peer_financial_trend` VIEW 에 metric_name_canonical 매핑 포함. |
| Context engineering — 신규 테이블 수 | **0개 (MVP)**. timeline / financial_trend / general_event_timeline 은 VIEW, `sector_pulse` 는 MATERIALIZED VIEW, `capability_evolution` 은 `peer_plus_payload` JSONB, snapshot history 는 `legacy_records` 재사용. |
| `event_chain_links` (V34) | **MVP 보류**. `card_news.evidence_payload['related_card_ids']` JSONB 로 시작 → W4 운영 4주 측정 후 도입 결정. **도입 시 Qdrant cosine ≥ 0.75 + 시간차 ≥ 7일 강제** (단순 keyword overlap X — 실측 시 자기참조 위험). |
| Cluster-time LLM 호출 증가 | **0건**. W4 의 `AnalysisContextBuilder` 는 DB query + Qdrant retrieve 만. 운영 LLM 비용은 월1회 CronJob (capability) + (옵션) 매일 (event chain) 으로 분산. |
| 카드뉴스 schema 변경 | **컬럼 1개 (`card_schema_version`) + 인덱스 1개 + CHECK 1개** (모두 V33 통합). `implication` JSONB key namespace 만 표준화. 기존 191 카드는 `'v1'` 유지 + frontend graceful fallback. |
| 카드뉴스 3섹션 (요약/시사점/대응) 표시 | **frontend / openapi 책임**. axis-backend `CardController` 가 `card_news.implication` JSONB → `CardNewsResponse.recommended_actions` / `implication_summary` 로 명시 매핑 (`openapi.yaml` 변경 PR 별도). |
| W4 시작 시점 | **W2 완료 후 W3 와 병렬**. W4-0/W4-1 은 W1 시작 직후 동시 진행 가능. |
| **v3.1 신설** — `card_news.cluster_id` 사용 정책 | **사용 금지** (P3-CRIT-1). ephemeral sequence 임을 design doc 명시. Cluster identity 는 `source_raw_article_ids` 또는 `(peer_company_id, event_type, DATE(created_at))` 복합키. |
| **v3.1 신설** — peer_id alias 정규화 | **W4-0 `peer_id_aliases.py` single source of truth**. 모든 query 가 `expand_peer_aliases()` 경유. 신규 peer 추가 시 PR review 강제. |
| **v3.1 신설** — financial_metrics 정규화 | **W4-0 `metric_canonical.py` + W4-1 VIEW 의 CASE 매핑 이중화**. 동일 metric 의 한/영/축약 표기 통합. |
| **v3.1 신설** — sector_pulse z-score 활성화 | **Phase 분리** (P3-DATA-5). W4-4 Phase 1 (count + intensity), W4 운영 4주 후 Phase 2 (anomaly trigger). |
| **v3.1 신설** — CapabilityEvolution LLM input | **chunked SQL pre-aggregation 필수** (P3-DATA-7). 그룹 별 top-5 confidence 만 LLM 전달. 평균 10-20k token / call. |
| **v3.1 신설** — sparse peer (LG CNS) 의 시사점 | **`evidence_density_per_peer` 신호** → ImplicationAgent prompt 가 peer-level confidence 차등 부여. sparse → 자동 `evidence_label="moderate"` 강등. |
| **v3.1 신설** — `evidence_payload.financial_refs.narrative` 재사용 | cluster-time financial_trend 재계산 대신 기존 narrative 우선 활용 (P3-LOG-2). VIEW 는 fallback. |
| **v3.1 신설** — 191 v1 카드 backfill | **graceful fallback 으로 시작** (옵션 A). W4 운영 안정화 + frontend v2 표시 검증 후 옵션 B (백필 $4.2 + 4h) 결정. |
| **v3.1 신설** — 카드 dedup 부족 | **Layer B 카드 생성 단** 의 작업으로 분리 (As-Is `ingestion_graph.card_news_node`, To-Be `CardNewsAgent`). W4-6 도입보다 dedup 강화가 시급 (실측: 같은 (peer, event_type, date) 에 카드 4-6건). ingestion (Layer A) 책임 아님. |
| **v3.2 신설** — Evaluation & Observability Layer 도입 | **W5 신설 (12-18h)**. 출력 품질 정량화. **Hybrid placement**: rule-based 5 metric (W5-1) 은 in-graph `validate` 노드 확장 (LLM X, latency <50ms, 회귀 즉시 차단), LLM-as-Judge 4 score (W5-2) 는 sidecar CronJob (`axis-cron-card-evaluator`, 5분 주기, gpt-4o-mini, +$1.5/일). 두 결과 모두 `card_news.evaluation_payload` JSONB 에 누적 → §3.6.3 의 regression detection CronJob (W5-3, 옵션) 자동화 가능. |
| **v3.2 신설** — Evaluator critical path 영향 | **In-graph rule-based 만 supervisor critical path 에 포함** (validate 노드 확장, +50ms 미만). LLM-as-Judge sidecar 는 critical path 밖 (5분 지연 후 평가) — cluster 처리 속도 / 사용자 가시 latency 에 영향 0. Sidecar 실패가 카드 가시성에 영향 X (frontend graceful fallback). |
| **v3.2 신설** — Hard-block vs soft-flag | **W5-1 의 5 metric 은 `human_review_flags` 만 추가** (hard fail X). 카드는 항상 INSERT 되되 품질 신호가 동반. Hard fail 은 기존 validate 노드의 numeric/단정표현 차단으로 한정. → 평가 자체로 인한 카드 손실 0. |
| **v3.2.1 신설** — `evaluation_payload` 컬럼 신설 (실측 후 정정) | 실측 검증 (2026-05-21) 결과 `card_news` 에 `evaluation_payload` 컬럼 미존재 확인 — v3.2 의 "V20 부터 존재" 기재 오류. V33 (W4-1) 에 `ALTER TABLE card_news ADD COLUMN evaluation_payload JSONB DEFAULT '{}'::jsonb` + partial index 추가. 기존 `validation_pass` / `validation_sc_score` 컬럼은 legacy 로 유지. |
| **v3.2.1 신설** — W5 Phase 분리 | **Phase 1 (즉시): rule-based 4 metric + LLM-judge 4 score** (운영 첫 1주는 threshold 비활성, 측정만). **Phase 2 (운영 14일 후): `regression_drift` + W5-3 알림 + Threshold calibration + Self-eval cross-check**. 사용자 결정 — "결과가 너무 적게 나오거나 많이 제한되지 않게" 의 충족. |
| **v3.2.1 신설** — Threshold calibration | **운영 1주 후 percentile-based 자동 산출** (bottom 10%). 초기 1주는 모든 카드 통과. 산출 SQL 은 weekly CronJob 으로 자동 갱신. 임의 hard-coded threshold X. |
| **v3.2.1 신설** — v1 카드 (191건) 평가 제외 | sidecar SQL `WHERE card_schema_version = 'v2'` guard 로 자동 skip. 사용자 결정 — "요약만 담고 있는 이전 버전의 카드 뉴스라면 제외해도 될 거 같아". v1 backfill 옵션 도입 X. |
| **v3.2.1 신설** — Rule-based metric 정정 | (1) `context_hit_ratio` 분모를 6 고정 → `available_layer_count` (신규 peer 의 unfair penalty 제거). (2) `actionability_score` 의 "verb-first" → 한국어 verb-suffix 사전 매칭 (`한다 / 할 것 / 검토 / 착수 / ...`). |
| **v3.2.1 신설** — Self-evaluation bias 대응 | Phase 1 은 gpt-4o-mini 단독. **Phase 2 (운영 1주 후) 부터 매주 10% sampling 을 Claude 3.5 Sonnet 로 cross-check** (+$0.30/일). 두 evaluator 의 `faithfulness` gap > 1.0 일관시 evaluator prompt 재검토 ticket. |
| **v3.2.1 신설** — Cost cap 정책 | **Daily $5/일 soft cap** (≈100 카드). 도달 시 sidecar batch skip + Slack 알림, 다음 KST 00:00 자동 재개. Hard kill 아님 — backlog 다음날 이연. `daily_budget_remaining()` 함수가 batch limit 동적 조정. |
| **v3.1 갱신** — 이번 주 (5/18 이후) 신규 카드 `peer_company_id` FK NULL | **W4-0 첫 마이그레이션에서 백필** (`UPDATE card_news SET peer_company_id = company`, `… SET primary_keyword_category = matched_sectors->>0`). 데이터 손실 X — `card_news.company` / `raw_articles.matched_companies` 모두 정상. ingestion 회귀 아닌 axis-ai `CardNewsAgent` 후처리 회귀 (5/15 FK, 5/20 sector 두 시점). 원인 추적은 별도 P1 ticket. |

---

## 10. 참고

- 설계 문서: [`design/00-supervisor-topology.md`](00-supervisor-topology.md)
- 검증 리포트: [`docs/AGENT_ARCHITECTURE_VERIFICATION.md`](../docs/AGENT_ARCHITECTURE_VERIFICATION.md)
- 데이터 흐름: [`design/README.md`](README.md)
- Evidence Chain 설계: [`design/10-ingestion/evidence.md`](10-ingestion/evidence.md)
- 시스템 아키텍처: `axis-infra/docs/SYSTEM_ARCHITECTURE.md`
- DB 스키마: `axis-infra/db/schema.sql`
- ADR 참고:
  - `axis-infra/docs/adr/0004-pipeline-separation.md` (파이프라인 분리 원칙)
  - `axis-infra/docs/adr/0005-two-storage-design.md` (PostgreSQL + Qdrant 이중 저장소)

### 10.1 v3.1 의 실측 검증 절차 (재현 가능성)

본 v3.1 의 9 critical issue (P3-CRIT 4 + P3-DATA 8 + P3-LOG 4) 는 다음 절차로 재현 가능:

```bash
# 1. 클러스터 postgres 접속 (kubectl context: arn:aws:eks:ap-northeast-2:881490135253:cluster/skala-2025)
NS=skala3-finalproj-class3-team13
POD=$(kubectl get pods -n $NS -l app=postgres -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $NS $POD -- psql -U axuser -d axis -c "SELECT COUNT(*) FROM card_news;"

# 2. P3-CRIT-1 재현 (cluster_id 의미 불일치)
kubectl exec -n $NS $POD -- psql -U axuser -d axis -c "
  SELECT 'card_news' AS t, COUNT(DISTINCT cluster_id), MAX(cluster_id) FROM card_news
  UNION ALL SELECT 'raw_articles', COUNT(DISTINCT cluster_id), MAX(cluster_id) FROM raw_articles;"
# 예상: card_news distinct ≤ 30, raw_articles distinct ≥ 500

# 3. P3-DATA-2 재현 (metric_name 정규화 누락)
kubectl exec -n $NS $POD -- psql -U axuser -d axis -c "
  SELECT metric_name, metric_label, COUNT(*) FROM raw_article_financial_metrics
  WHERE metric_name IN ('net_income') OR metric_label IN ('당기순이익','순이익')
  GROUP BY 1,2;"

# 4. P3-DATA-7 재현 (CapabilityEvolution token 초과 위험)
kubectl exec -n $NS $POD -- psql -U axuser -d axis -c "
  SELECT peer_id,
         COUNT(*) AS n,
         SUM(LENGTH(summary) + COALESCE(LENGTH(evidence_text), 0)) AS total_chars
  FROM raw_article_business_signals
  WHERE period_year >= 2025
  GROUP BY peer_id ORDER BY total_chars DESC;"
# 예상: peer 당 ~500k chars ≈ 170k token

# 5. P3-CRIT-4 재현 (CardNewsAgent 후처리 회귀 — FK + sector, 두 시점)
kubectl exec -n $NS $POD -- psql -U axuser -d axis -c "
  SELECT DATE(created_at) AS d, company,
         COUNT(*) AS total,
         COUNT(*) FILTER (WHERE peer_company_id IS NOT NULL)        AS fk_filled,
         COUNT(*) FILTER (WHERE primary_keyword_category IS NOT NULL) AS sector_filled
  FROM card_news
  WHERE company IN ('lg_cns','samsung_sds','hyundai_autoever')
  GROUP BY 1,2 ORDER BY 1 DESC LIMIT 14;"
# 예상: 5/14 100/100 → 5/15 부분 → 5/16~ FK 0% (1차 회귀)
#       5/19 까지 sector 100% → 5/20 sector 0% (2차 회귀)
# 확인: company 자체는 100% 정상 → ingestion 식별 OK, 카드 후처리만 회귀

# 5-2. 백필 가능성 검증 (W4-0 첫 마이그레이션 dry-run)
kubectl exec -n $NS $POD -- psql -U axuser -d axis -c "
  SELECT cn.company, pc.id IS NOT NULL AS in_peer_companies, COUNT(*) AS n
  FROM card_news cn LEFT JOIN peer_companies pc ON pc.id = cn.company
  WHERE cn.peer_company_id IS NULL GROUP BY 1,2;"
# 예상: in_peer_companies=t 인 행이 NULL 카드 전부 → UPDATE 1쿼리로 복구
```

### 10.2 V33 SQL dry-run 결과 (2026-05-20 검증)

`BEGIN; ... ROLLBACK;` 으로 prod schema 에 직접 시뮬레이션. 모든 단계 PASS:

| 단계 | 결과 |
|---|---|
| `ALTER TABLE card_news ADD COLUMN card_schema_version` | ✅ PASS |
| `CREATE VIEW peer_event_timeline` | ✅ PASS — 131 rows (peer 매칭 카드만) |
| `CREATE MATERIALIZED VIEW sector_pulse` | ✅ PASS — 11 rows (5/11 주 5 sectors + 5/18 주 6 sectors) |
| `CREATE VIEW peer_financial_trend` (canonical 매핑) | ✅ PASS — `samsung_sds` 의 `net_income` canonical 그룹화 검증 |
| `COALESCE(period_quarter, 'annual')` fallback | ✅ PASS — samsung_sds 2025: Q1(46) + Q2(53) + Q3(49) + Q4(60) + annual(19) = 227 |

→ V33 마이그레이션은 prod 데이터 위에서 즉시 실행 가능. 부수효과 없음.

### 10.3 W5 (Evaluation Layer) 사전 검증 결과 (2026-05-21)

cluster DB 직접 쿼리로 W5 의 schema 가정과 baseline 충족 여부 점검. **6개 발견 → 모두 v3.2.1 패치로 반영**.

| 발견 | 실측 | 패치 위치 |
|---|---|---|
| `card_news.evaluation_payload` 컬럼 미존재 | `information_schema.columns` 조회 결과 `validation_pass` (bool), `validation_sc_score` (double) 만 존재 | §3.6.2 schema 정정 + W4-1 V33 SQL 에 `ALTER TABLE ADD COLUMN evaluation_payload JSONB` |
| `regression_drift` baseline 부족 | 카드 운영 정상 day 6일 (5/15~5/20) < 7일 | §3.6.1 Phase 분리 (drift 는 Phase 2, 운영 14일 후 활성화) |
| v1 카드 mix (191건 모두 v1, summary 중심) | `card_schema_version` 컬럼 자체 미존재 + heuristic `implication` 메타데이터만 | §3.6.4 sidecar SQL 에 `WHERE card_schema_version = 'v2'` guard |
| 한국어 verb-first 적용 불가 | "디지털 전환 가속화" 는 명사형 종결, verb-first 불가능 | §3.6.1 actionability 한국어 verb-suffix 사전 패턴으로 변경 |
| `context_hit_ratio` 분모 6 고정의 unfair penalty | 신규 peer 는 `capability_evolution` 부재 → 자동 5/6 이하 | §3.6.1 분모를 `available_layer_count` 로 변경 |
| LLM-as-Judge family bias 위험 | gpt-4o 생성 → gpt-4o-mini 평가, 같은 family | §3.6.5 신설 — Phase 2 부터 Claude 10% cross-check |

**cluster DB 운영 상태 (2026-05-21):**
- `raw_articles`: 16,211 행 / 운영 9일치 (published 15일치 분포)
- `card_news`: 191 행 / 운영 정상 6일 (5/14 의 127건 backfill 제외 시 일평균 ~11건)
- `raw_article_business_signals`: 27,377 행 / 5/19~5/20 backfill 이지만 period_year 2021~2026 분포 — W4-3 capability evolution 가능
- `raw_article_financial_metrics`: 3,449 행 / multi-year — W4-1 trend VIEW 가능
- **W4 전체 진행 가능**. W5 도 즉시 도입 가능 (단 drift 는 Phase 2).

---

*작성: 2026-05-20 KST · **v3.2.1** · Critical path 39-41h · 총 ~88h · 일일 비용 +$3.37 (W5 LLM-judge sidecar 포함, 한도의 46%)*
*v3.0 변경점: §3.4 / §3.5 / W1-4 / W4 단계 신설 / §6.1·6.2 / §7 W4 Done / §9 결정사항 6항 갱신*
*v3.1 변경점 (실측 기반): §2.4 9 critical issue (P3-CRIT 4 + P3-DATA 8 + P3-LOG 4) / W4-0 (data hygiene precondition) 신설 / §3.4.3 metric_name_canonical 매핑 + general_event_timeline VIEW / §3.4.5 PEER_ID_ALIASES + cluster identity 재정의 + token budget 실측 / §3.4.6 chunked LLM input + sector_pulse Phase 분리 / §6.3 신규 리스크 13항 / §9 결정사항 v3.1 10항 추가*
*v3.1 검증 완료 (2026-05-20): §10.1 9 issue 모두 cluster DB 에서 재현 검증 / §10.2 V33 SQL 5단계 dry-run 모두 PASS / 내부 일관성 점검 (W1-4 ↔ W4-1 V33 단일화, SupervisorState 의 analysis_context 정합, build_analysis_context 노드 정합) 완료*
*v3.1.1 정정 (2026-05-20 21:00) — P3-CRIT-4 재진단: `card_news.company` 와 `raw_articles.matched_companies/matched_sectors` 모두 정상임을 확인 (회귀 범위 좁아짐). ingestion 회귀가 아닌 axis-ai `CardNewsAgent` 후처리의 두 시점 회귀 (5/15 FK, 5/20 sector). 데이터 손실 X → W4-0 작업 5 (`V32_5__card_news_backfill.sql`) 로 즉시 복구 가능, 별도 P0 (ingestion-side) 항목에서 제외 / 회귀 원인 추적은 별도 P1 ticket.*
*v3.1.2 정정 (2026-05-20 21:05) — Layer 경계 명확화: §0.1 신설하여 시스템을 Layer A (Data Pipeline — ingestion = crawl/dedup/classify/정제/DB 적재까지만) 와 Layer B (Analysis Supervisor Graph — context assemble → LLM reasoning → validation → package generation) 의 2-layer 로 명시. 본 계획서의 모든 작업은 Layer B. 카드 dedup / source 강제 검증 / FK 회귀 등은 "ingestion-side" 가 아닌 "Layer B 카드 생성 단" (As-Is `ingestion_graph.card_news_node`, To-Be `CardNewsAgent` 이관) 으로 재분류 — §2.4 P3-CRIT-3, §2.4 P3-LOG-3, §6.3 리스크, §9 결정 4군데 표기 정정. SUPERVISOR_BRIEF §2 그림에 DB 경계 추가 및 카드 생성을 Layer B 의 ④ package generation 으로 이동.*
*v3.1.3 정정 (2026-05-20 21:15) — As-Is debt 해소: §0.1 의 "W2-1 시 이관 예정" promise 를 실행 가능한 task 로 전환. **W2-1 작업 5 신설** — `ingestion_graph.card_news_node` 제거 + `card_writer` 노드를 supervisor 의 last 노드로 신설 (assemble → card_writer → END). §3.1 다이어그램에서 `(외부) CardNewsAgent` 제거, retry 정책에 `card_writer` 추가. W2-1 시간 6-8h → 8-11h (+2~3h). W2 Done 에 card_writer wired 검증 항목 + supervisor 노드 7개 elapsed_ms 항목. SUPERVISOR_BRIEF §2 그림 / §9 갱신. 카드뉴스가 "분석/시사점/대응의 결과물 = Supervisor 의 산출물" 이라는 사용자 의도가 코드 위치에서도 일치.*
*v3.2 추가 (2026-05-21 10:00) — **Evaluation & Observability Layer (W5) 신설**: 출력 품질 정량화. §3 6번 조항 (Evaluation Layer) + §3.6 (5 rule-based metric + 4 LLM-judge score 정의, evaluation_payload JSONB schema, sidecar 운영 흐름) + W5-1~3 작업 단락 + §5 일정 (W5 12-18h, critical path +4h, 총 ~85h / 5-5.5주) + §6.1 비용 (+$1.5/일, 일일 총합 $3.37, 한도의 46%) + §6.2 지표 (5+4 신규 metric 목표) + §7 W5 Done + §8 사후 모니터링 SQL 6개 + §9 결정 3항 (Hybrid placement / critical path 영향 / soft-flag 정책) 추가. SUPERVISOR_BRIEF agent 11→13, §2 그림에 evaluator sidecar 박스 추가, §3.2 CardEvaluatorSidecar+RegressionCheck, §4 evaluation_payload 활용, §6 5단계, §7 W5 컬럼, §8 결정 7항 추가.*
*v3.2.1 정정 (2026-05-21 11:00) — **W5 검증 후 6 issue 패치** (사용자 결정 반영): (P5-CRIT-1) `evaluation_payload` 컬럼 실측 미존재 확인 → V33 (W4-1) 에 `ALTER TABLE ADD COLUMN evaluation_payload JSONB DEFAULT '{}'::jsonb` + partial index 추가, §3.6.2 정정. (P5-DATA-1) `regression_drift` baseline 6일치 부족 → Phase 분리 (Phase 1 즉시 4 metric / Phase 2 운영 14일 후 drift+W5-3). (P5-DATA-2) v1 카드 191건 평가 제외 → sidecar SQL `WHERE card_schema_version = 'v2'` guard. (P5-LOG-1) `actionability_score` 한국어 verb-suffix 사전 패턴으로 변경. (P5-LOG-2) `context_hit_ratio` 분모를 `available_layer_count` 로 변경 (unfair penalty 제거). (P5-LOG-3) §3.6.5 신설 — Phase 2 부터 Claude 10% cross-check sampling. §3.6.6 신설 — Threshold lenient 시작 (1주 후 percentile calibration), Sidecar burst 대응, Cost cap $5/일 soft 정책. §5 W5 시간 +1~2h (총 ~88h, critical path 39-41h). §7 W5 Done Phase 1/2 분리. §9 결정 표 6항 추가. §10.3 신설 (W5 사전 검증 결과). SUPERVISOR_BRIEF 정합성 갱신.*
*다음 갱신 권고: W4-0 완료 후 (실 데이터에서 alias / metric / cluster identity 통합 효과 측정) · W1 완료 후 (`evidence_label` 첫 측정값 + LLM 비용 실측 반영) · W4 운영 4주 후 (W4-6 도입 결정 + sector_pulse Phase 2 활성화)*
