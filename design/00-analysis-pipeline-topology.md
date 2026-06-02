# Analysis Pipeline Topology

> 기준: 1단계 데이터 **분석 Pipeline** / 2단계 데이터 활용 Orchestrator
> v3.2.2 갱신 (2026-05-21) — 외부 리뷰 R-rename 반영. "Supervisor Topology" 에서 변경.

## 명칭 (2026-05-21 R-rename)

본 문서의 "Analysis Pipeline" 은 **LangGraph 기반 고정 순서 DAG**. 노드 순서가
정적으로 정의되어 있고, 단 하나의 동적 분기는 `validate` 의 `pass / fail` 라우팅이다.
LLM 이 다음 worker 를 동적으로 선택하는 *multi-agent supervisor pattern* 이 **아니다**.

| 기존 명칭 (backward-compat) | 권장 명칭 (외부 리뷰 R-rename) |
|---|---|
| `AnalysisGraphRunner` | `AnalysisGraphRunner` |
| `SupervisorState` | `AnalysisFlowState` |
| `SupervisorDeps` | `AnalysisFlowDeps` |
| `build_supervisor_graph()` | `build_analysis_flow_graph()` |
| `default_supervisor_graph()` | `default_analysis_flow_graph()` |
| `run_supervisor()` | `run_analysis_flow()` |
| `Supervisor Topology` (문서) | `Analysis Pipeline Topology` |
| `01-supervisor-implementation-plan.md` | `01-analysis-pipeline-implementation-plan.md` |
| `SUPERVISOR_BRIEF.md` | `ANALYSIS_PIPELINE_BRIEF.md` |

1 cluster 처리가 정해진 절차 (요약 → 분석 → 시사점 → 검증 → 카드) 라서 dynamic
routing 가치가 낮고, deterministic pipeline 이 운영 예측·debugging·비용 안정성 측면에서
더 적합하다.

## 1단계: 데이터 수집·정제·통합·분석·시사점·카드뉴스

```text
[Layer A — Data Pipeline (axis-cron-ingestion-*)]
  수집 → 원문 저장 → 전처리·적합성 판단 → 기업·섹터·이벤트 매칭
  → raw_articles / raw_article_business_signals / raw_article_financial_metrics 저장

                            ↓ (cluster / document / period trigger)

[Layer B-0 — Context Memory CronJob (정적·시계열 맥락 갱신)]
  axis-cron-profile-refresh        (분기1회, gpt-4o) → peer_companies.profile_snapshot
  axis-cron-capability-evolution   (월1회, gpt-4o)   → peer_plus_payload['capability_evolution']
  axis-cron-sector-pulse           (주1회, psql)     → sector_pulse MV REFRESH

                            ↓

[Layer B — Analysis Supervisor Graph (cluster-time, src/pipeline/supervisor_graph.py)]
  ① issue_integrate         (LLM gpt-4o)  IntegratedIssue 생성
  ② profile_context         (DB only)     main_company 확정 후 ProfileContext 합성
  ③ build_analysis_context  (DB+Qdrant)   AnalysisContext (4-Layer, ≤4,000 token)
  ④ strategic_insight       (LLM gpt-4o)  AnalysisResult + ImplicationResult
  ⑤ validate                (rule-based)  numeric/certainty/evidence + Evaluator
       pass → ⑥ assemble → ⑦ card_writer → card_news INSERT (v2 schema)
       fail → human_review (flag only, 카드 생성 X)

                            ↓

[Layer B+1 — Evaluation Sidecar (5분 주기, gpt-4o-mini)]
  axis-cron-card-evaluator → card_news.evaluation_payload['llm_judge']
```

### 순서 변경 근거 (외부 리뷰 2026-05-21 반영)

이전 버전은 `profile_context → build_analysis_context → issue_integrate → ...` 였으나
다음 이유로 `issue_integrate → profile_context → build_analysis_context → ...` 로 정정:

1. **잘못 매칭된 cluster (`is_valid_summary=False`) 가 ① 에서 즉시 차단** → profile /
   context build 의 DB query 비용 절약.
2. **`IntegratedIssue.main_company` 가 확정된 후 context 조회** → ProfileContext /
   AnalysisContext 가 cluster 의 실제 주체에 맞게 build 됨.

## 2단계: 저장 데이터 활용

```text
Raw / 정제 데이터 저장소
+
카드뉴스 / 시사점 저장소
+
ProfileContext
→ DataUsageOrchestrator
   → MixerAgent
   → ITTrendAgent
   → ReportAgent
   → InsightAgent
   → ChatbotAgent
   → KeywordGraphBuilder
```

### ITTrendAgent 입력 구분

- 글로벌 회사별 뉴스룸은 카드뉴스 생성 대상이다. Microsoft, AWS, Google, NVIDIA,
  OpenAI 같은 회사별 뉴스룸은 일반 이슈처럼
  `IntegratedIssue → StrategicInsightAgent → CardNewsComposer` 흐름을 탄다.
- SPRi / BCG 자료는 카드뉴스 생성 대상이 아니다. `ITTrendAgent`가 이 자료와 과거
  `TrendContext`를 함께 보고 글로벌·산업 흐름을 갱신한다.
- 글로벌 회사별 뉴스룸의 `IntegratedIssue`와 `AnalysisResult`도 `ITTrendAgent`가
  실행 신호로 참고해 `TrendContext`를 갱신한다.
- `ITTrendAgent` 출력은 카드뉴스가 아니라 `TrendContext`다.
- `StrategicAnalyzer`는 이슈 분석 시 필요하면 `TrendContext`를 참고할 수 있다.
- 카드뉴스 화면 결과나 글로벌 뉴스룸 원문을 `TrendContext` 입력으로 직접 쓰지는 않는다.

## AnalysisGraphRunner

`AnalysisGraphRunner`는 1단계 분석 흐름을 조율하는 analysis flow runner다.
직접 통합, 분석, 시사점을 모두 수행하는 Agent가 아니라 고정 DAG 노드들의 실행 순서와
데이터 전달을 관리한다.

주요 책임:
- Raw / 정제 데이터 저장소에서 분석 대상 데이터를 조회한다.
- 뉴스의 경우 `raw_articles.cluster_id` 기준으로 클러스터 전체 기사를 조회한다.
- DART/IR/리포트의 경우 `raw_articles.id` 또는 `document_group_id` 기준으로
  `raw_article_parse_results`, `raw_article_financial_metrics`,
  `raw_article_business_signals` 등을 함께 조회한다.
- 조회한 데이터를 `AnalysisInputBundle`로 구성한다.
- 하위 Agent를 조율한다.
- 최종 결과를 `AnalysisPackage`로 묶어 `CardNewsComposer`에 전달한다.

내부 흐름 (v3.2.1, LangGraph StateGraph 9-node):

```text
AnalysisInputBundle
→ ① issue_integrate          IntegrationAgent → IntegratedIssue
→ ② profile_context          ProfileContextLoader.load → ProfileContext (Tier A snapshot + Tier B enrichment)
→ ③ build_analysis_context   AnalysisContextBuilder → AnalysisContext (6 layer, ≤4,000 token)
→ ④ strategic_insight        StrategicInsightAgent → AnalysisResult + ImplicationResult
→ ⑤ validate                 _hard_validate + Evaluator → ValidationReport
   ├ pass → ⑥ assemble → ⑦ card_writer → save_card_news (v2 schema) → END
   └ fail → human_review (flag only) → END
```

각 노드는 `_logged_step` 데코레이터로 `pipeline_logs.step='supervisor.<node_name>'` 에
elapsed_ms 기록. 부분 실패는 다음과 같이 흡수:

- ② ProfileContextLoader fail → legacy `ProfileAgent.build_context` fallback
- ③ DB unavailable → 빈 `AnalysisContext` (StrategicInsightAgent 내부 implication fallback 사용)
- ④ LLM fail → 내부 fallback (`is_valid_implication=true` 단순 출력)

LangGraph `RetryPolicy / with_retry` 정식 도입은 별도 PR (`design/01-analysis-pipeline-implementation-plan.md`
의 §3.1 retry 표는 미구현 — 현재는 `_logged_step` try/except + ImplicationAgent fallback 만).

## IntegrationAgent

기존 요약 Agent의 역할을 대체하는 이슈 통합 Agent이다.

역할:
- 원문/클러스터/문서/파싱 결과를 하나의 통합 이슈로 정리한다.
- 여러 기사 또는 문서에서 반복되는 사실을 합친다.
- 일부 원문에만 있는 중요한 정보는 보조 fact로 보존한다.
- 수치, 사업 신호, 리스크, 불확실성을 구조화한다.
- 분석 가능한 하나의 이슈 글을 만든다.

출력:

```json
{
  "main_issue": "",
  "integrated_text": "",
  "consolidated_facts": [],
  "key_numbers": [],
  "business_signals": [],
  "representative_sources": [],
  "missing_or_uncertain_points": []
}
```

주의:
- 카드뉴스용 3줄 요약을 만들지 않는다.
- 시사점이나 대응 방향을 만들지 않는다.

## StrategicAnalyzer

`IntegratedIssue`를 기반으로 전략적 의미를 분석한다.

입력:
- `IntegratedIssue`
- company / sector / event_type metadata
- 필요 시 `TrendContext`

출력:

```json
{
  "strategic_moves": [],
  "market_signals": [],
  "competitive_meaning": "",
  "risk_factors": []
}
```

주의:
- 원문/클러스터/문서 전체를 다시 읽지 않는다.
- 원문 기반 fact 통합은 IntegrationAgent 책임이다.
- StrategicAnalyzer는 IntegratedIssue 안의 `integrated_text`, `consolidated_facts`,
  `key_numbers`, `business_signals`, `fact_basis`를 근거로 해석한다.
- `TrendContext`가 있으면 글로벌/산업 배경으로만 참고하며, IntegratedIssue에 없는
  사실을 새로 만들지 않는다.

## ProfileAgent

**핵심 책임: RDB 백필 데이터를 기반으로 회사별 전략 프로필을 생성하는 합성 Agent.**

ProfileAgent 는 단순 context provider 가 아니라 **원천 데이터 (DART / IR / 공식 newsroom
/ 누적 뉴스 / business_signals / financial_metrics) 를 LLM 으로 합성하여 회사별 전략
프로필을 만드는 합성 책임자** 이다. 외부 리뷰 2026-05-21 명시.

### 두 단계로 분리되어 운영됨 (W2-2 2-tier)

| 단계 | 시점 | 책임 | 출력 |
|---|---|---|---|
| **Tier A (snapshot 생성)** | 분기 1회 CronJob (`axis-cron-profile-refresh`) | RDB 의 6개월치 뉴스 + DART + IR + 공식 newsroom 을 회사별로 종합 → LLM (gpt-4o, `profile-v5` prompt) 으로 **회사 방향성 / 주요 사업 / 전략 변화 / 역량 평가** narrative 합성 | `peer_companies.profile_snapshot` JSONB (별도 컬럼) |
| **Tier B (runtime loader, ProfileContextLoader)** | cluster-time (Analysis Flow ② 노드) | Tier A snapshot 을 그대로 load + 최근 30일 business_signals top-3 + 최근 분기 financial_metrics 보강 (DB query only, LLM X) | `ProfileContext` 메모리 dataclass |

### 출력의 두 관점 (Peer 와 SK AX 분리)

- **`ProfileContext.peer_profiles[peer_id]`** — StrategicAnalyzer 의 입력. peer 의 전략·역량·
  사업 방향 자체를 해석하는 데 사용.
- **`ProfileContext.skax_profile`** — ImplicationAgent 의 입력. SK AX 관점에서 기회/위협/
  대응 방향을 도출하는 데 사용.

### 주의

- 전처리 단계의 기업·섹터·이벤트 **매칭** 과 다르다. 매칭은 raw data 에 라벨을 붙이는
  기능. ProfileAgent 는 그 위에 **회사 전략 합성 narrative** 를 만든다.
- Tier A 가 비어있는 환경 (CronJob 미실행) 에서 Tier B 는 빈 profile_snapshot 위에
  recent enrichment 만 붙인다 — graceful degradation.

## ImplicationAgent

분석 결과와 프로필 context를 결합해 SK AX 관점의 시사점을 도출한다.

입력:
- `AnalysisResult`
- `ProfileContext`
- `IntegratedIssue`
- 필요 시 `AnalysisInputBundle` / sources

출력:

```json
{
  "implication": "",
  "opportunities": [],
  "threats": [],
  "recommended_actions": [],
  "follow_up_questions": []
}
```

## CardNewsComposer

`AnalysisPackage`를 사용자에게 보여주기 좋은 카드뉴스/API 응답 형태로 재가공한다.

역할:
- 카드 제목 생성
- 카드뉴스용 3줄 요약 생성
- 핵심 포인트 생성
- 시사점 문장 재가공
- sources / validation 구성
- 기존 저장 구조에 맞춘 저장 요청

주의:
- 원문 통합을 수행하지 않는다.
- 전략 분석을 수행하지 않는다.
- 카드뉴스용 3줄 요약은 여기서 생성한다.

## DataUsageOrchestrator

2단계 활용 흐름을 조율하는 Supervisor이다.

AnalysisGraphRunner와의 차이:
- `AnalysisGraphRunner`는 하나의 이슈/클러스터/문서를 분석해 카드뉴스를 만드는 1단계 Supervisor이다.
- `DataUsageOrchestrator`는 이미 저장된 결과를 활용해 리포트, 인사이트, 챗봇, 키워드 그래프를 만드는 2단계 Supervisor이다.

지원 Agent:
- `MixerAgent`
- `ITTrendAgent`
- `ReportAgent`
- `InsightAgent`
- `ChatbotAgent`
- `KeywordGraphBuilder`

## 내부 DTO 원칙

`src/analysis/models.py`에 있는 모델은 DB schema가 아니다.

허용 내부 DTO:
- `AnalysisInputBundle`
- `IntegratedIssue`
- `AnalysisResult`
- `ProfileContext`
- `ImplicationResult`
- `AnalysisPackage`
- `CardNews`

금지:
- DB schema 임의 변경
- `src/schemas.py` 임의 변경
- repository/save 로직 임의 변경
- 별도 근거 패키지 저장 테이블 추가
