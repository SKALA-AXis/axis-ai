# AXIS AI — Agent Design Plans

> **위치**: `axis-ai/design/`
> **작성**: 2026-05-13 · **버전**: v1
> **선행 문서**: `axis-infra/docs/AI_AGENT_DESIGN.md` v1.1 (전체 supervisor / agent / cost / priority 통합 설계)
>
> 본 디렉토리는 **agent 별 개별 design plan** 을 담는다. 각 agent 의 책임 · 입출력 schema · 알고리즘 · LLM prompt · 에러 처리 · 외부 의존성 · State 흐름 · Provenance · 테스트 · 모니터링 · 구현 위치 를 깊게 기술.

## 디렉토리 구조 (axis-ai LangGraph agent 설계만)

```text
design/
├── README.md                          ← 본 파일 (index)
├── 00-supervisor-topology.md          ← 5+1 supervisor 전체 흐름
├── 02-prompt-design-checklist.md      ← PDF 2026-05-14 17 요소 + CoT 표준 (모든 LLM agent 의 prompt audit 기준)
│
├── 10-ingestion/                      ← @Scheduled 매시간 batch
│   ├── crawler.md                     ← #1 외부 fetch
│   ├── parser.md                      ← #2 HTML/PDF/JSON 정규화 (strategy)
│   ├── credibility.md                 ← #3 출처 신뢰도
│   ├── relevance.md                   ← #4 관련성 라우팅
│   ├── dedup.md                       ← #5 BGE-M3 클러스터
│   ├── classification.md              ← #6 event/sector + exposure_score
│   ├── card-composer.md               ← #7 IssueCardAgent + NewsSummary + NewsAnalysis 통합
│   ├── evidence.md                    ← #8 evidence_chain 4종
│   ├── financial-linker.md            ← Evidence sub (재무 segment)
│   ├── ir-parser.md                   ← Evidence sub (IR PDF)
│   └── embed-index.md                 ← vector_index 노드
│
├── 20-enrichment/                     ← Ingestion 후 + nightly batch
│   ├── keyword-extraction.md          ← KR-TF-IDF base
│   ├── keyword-graph-builder.md       ← Graph nodes/edges
│   ├── peer-word-cloud.md             ← Peer 별 카테고리 라벨링
│   ├── search-suggest.md              ← 자동완성 추천
│   └── derived-metrics.md             ← TopInsight + Trend + MonitoringOverview + PeerOverview (4-mode)
│
├── 25-knowledge-curation/             ← (신규 2026-05-14) peer narrative 압축 + Analysis Ledger
│   ├── README.md                      ← 5 계층 (L0~L4) + Analysis Ledger 아키텍처
│   ├── compaction-agent.md            ← weekly / monthly / quarterly LLM 압축 (3-mode 통합)
│   ├── context-pack-builder.md        ← L0~L4 stack 을 PeerContextPack 으로 조립 (산식)
│   └── analysis-ledger.md             ← 분석 결과 carry-over (산식, K1 우선 도입 권장)
│
├── 30-analysis/                       ← user POST request, on-demand
│   ├── insight-cascade.md             ← 4단계 Cause→Change→Impact→Response + CoT
│   ├── mixer-analysis.md              ← 2~20 카드 cross-card 신호 + CoT
│   ├── peer-comparison.md             ← SK AX vs Peer 차별 + Forecast (1Q/6M/1Y) + CoT
│   ├── global-trends.md               ← (신규 2026-05-14) 글로벌 6사 트렌드 → SK AX 영향 매트릭스 + Forecast + CoT
│   └── link-verification.md           ← URL liveness + diff
│
├── 40-user-query/                     ← user 검색 + chat
│   ├── hybrid-search.md               ← Dense + Sparse RRF
│   ├── rerank.md                      ← BGE-reranker-v2-m3
│   ├── answer.md                      ← Generative answer + SC
│   └── chat-orchestrator.md           ← Intent + Conversation 통합
│
├── 50-weak-signal/                    ← @Scheduled 월 09:00, W7+
│   └── weak-signal.md                 ← Pattern + Anomaly + AlertRouting 통합
│
├── 60-briefing/                       ← user-triggered async briefing 문서 생성
│   └── briefing-generation.md         ← BriefingReport 5-phase (`/api/briefings/generate`)
│
└── 90-cross-cutting/                  ← 모든 axis-ai agent 에 주입 (middleware)
    ├── confidence-score.md            ← 0~1 점수 산출/전파 표준
    ├── db-relations.md                ← raw_articles 중심 FK/매핑 테이블 표준 (V20/V21)
    ├── provenance-tracker.md          ← llm_model+prompt_version+git_sha+langfuse_trace_id 자동 부착
    └── token-budget.md                ← daily envelope + circuit breaker
```

## 인프라 / 크로스시스템 spec (별도 위치)

axis-ai agent 설계 *외부* 의 spec 은 `axis-infra/docs/` 가 owner:

| 문서 | 위치 | 책임 |
|---|---|---|
| API Surface | `axis-infra/docs/API_SURFACE.md` | FE ↔ BE ↔ axis-ai 의 전체 endpoint 매핑 |
| Audit Log | `axis-infra/docs/AUDIT_LOG.md` | BE Spring AOP @Auditable + audit_logs V14 schema |
| Observability (Langfuse) | `axis-infra/docs/OBSERVABILITY_LANGFUSE.md` | Self-host Helm + LangChain integration + 3-tier 분담 |
| User Events | `axis-infra/docs/admin/user_events.md` | FE tracker SDK + BE INSERT + user_events V15 |
| Feedback | `axis-infra/docs/admin/feedback.md` | BE write + Langfuse score + feedback V16 |
| Cost Reconciliation | `axis-infra/docs/admin/cost_reconciliation.md` | Spring @Scheduled ETL + V17/V18 |
| Metrics Exporter | `axis-infra/docs/admin/metrics_exporter.md` | Prometheus + Grafana 5 폴더 dashboard |

(axis-ai 의 cross-cutting 미들웨어 3종 — confidence / provenance / token-budget — 은 axis-ai agent 코드를 wrapping 하는 decorator 이므로 본 디렉토리에 유지.)

## 카운트

| 그룹 | 파일 | 비고 |
|---|---|---|
| Ingestion | 11 | 8 top-level node + 3 sub-agent |
| Enrichment | 5 | 모두 P6 우선순위 |
| **KnowledgeCuration** *(신규 2026-05-14)* | **4** | **README + 3 agent (compaction / context-pack / analysis-ledger)** |
| Analysis | 5 | P7 + global-trends (신규 2026-05-14) |
| UserQuery | 4 | Search 3 + Dialogue 1 통합 |
| WeakSignal | 1 | 3-phase 통합 (W7+) |
| Briefing | 1 | async user-triggered (P7+, V12 briefing_reports) |
| Cross-cutting (axis-ai 미들웨어만 + DB 관계) | 4 | confidence / db-relations / provenance / token-budget |
| 구조 / 통합 | 3 | README + topology + prompt-design-checklist (신규 2026-05-14) |
| **합계 (axis-ai/design/)** | **38** | LangGraph agent 설계 + axis-ai decorator + DB 관계 표준 + PDF 17-요소 + 5 계층 knowledge |
| (참고) axis-infra/docs 이동분 | +7 | API_SURFACE + AUDIT_LOG + OBSERVABILITY_LANGFUSE + admin/ × 4 |

### Agent 카운트 (파일 ≠ agent)

- **Top-level agents (production-tier)**: 8 (ingestion) + 5 (enrichment) + **3 (knowledge-curation)** + 5 (analysis, +global-trends) + 4 (userquery) + 1 (weak-signal) + 1 (briefing) = **27 agent**
- **Sub-agents (ingestion 내부)**: financial-linker, ir-parser, embed-index = 3
- **Middleware (cross-cutting)**: 3 (agent 아님, decorator)
- 총 30 agent file + 3 middleware file = **33 design 대상 + 3 구조 doc + 1 sub-supervisor README (25-knowledge-curation/) = 37 file**

## 설계 원칙

본 design 디렉토리의 모든 agent 는 다음 원칙을 따른다 (v1.1 통합 design 의 §5 와 동일):

1. **Single Responsibility** — 한 agent = 한 책임 + 한 출력 schema. 책임이 두 개면 두 agent.
2. **Function-named** — 이름은 *무엇을 만드는지* 기준. DB 테이블 명에 따라 가지 않음 (예: `IssueCardAgent` 가 `card_news` 테이블에 INSERT — 클래스명은 유지).
3. **Deterministic-first** — LLM 없이 산식으로 가능하면 산식. exposure_score · credibility · keyword extraction · dedup 등.
4. **State 일방향** — agent 끼리 직접 호출 X. supervisor 의 state (LangGraph TypedDict) 만 공유.
5. **Provenance 필수** — 모든 출력에 `evidence_chain.provenance` 부착 (llm_model · prompt_version · run_at · raw_article_ids · git_sha).
6. **Confidence 표면화** — `confidence < 0.6` 은 UI 경고 (frontend 가 그렇게 구현).
7. **Caching aggressive** — LLM 호출은 cache 우선. 같은 입력 (card_id set hash 등) 에 같은 결과면 LLM 재호출 X.

## 파일 작성 템플릿

각 agent 디자인 파일은 다음 14 section 을 포함:

1. **메타** — 이름 / supervisor / 상태 / owner / version / file path
2. **책임 (Single Responsibility)** — 한 줄 + 구체적
3. **책임 NOT** — out of scope 명시
4. **입력 스펙** — TypedDict / Pydantic
5. **출력 스펙** — schema + post-conditions
6. **알고리즘** — 산식 또는 LLM prompt + chain-of-thought
7. **LLM 모델 + token 예산** — 모델 / token / 비용
8. **에러 처리** — timeout / retry / fallback / DLQ
9. **외부 의존성** — DB / API / model
10. **State 흐름** — input keys → output keys (LangGraph)
11. **Provenance + Confidence** — 어떤 필드 채우는지 + 점수 공식
12. **테스트 시나리오** — unit / integration / edge
13. **모니터링** — pipeline_logs · KPI · token 예산
14. **구현 메모 + Changelog**

## 갱신 정책

- 새 agent 추가 시 본 README 의 디렉토리 구조 + 카운트 표 + 해당 그룹 디렉토리에 신규 파일
- agent 책임 변경 시 해당 agent 파일의 §2 책임 + Changelog 갱신
- LLM 모델 / token 예산 변경 시 모든 영향받는 agent 파일의 §7 갱신
- `axis-infra/docs/AI_AGENT_DESIGN.md` 통합 design 과 동기화 — 본 디렉토리가 SoT 가 아니라 상세 design 보강
- **LLM agent 추가 시** — `02-prompt-design-checklist.md` 17 요소 audit table 을 §6 에 추가 (필수)

## Changelog

- **2026-05-13** — v1 초기 설계 (8 ingestion + 5 enrichment + 4 analysis + 4 userquery + 1 weak + 1 briefing + 3 cross-cutting + 2 구조)
- **2026-05-14** — PDF 사업전략팀 추가 질의 회신 반영
  - 신규: `02-prompt-design-checklist.md` (17 요소 + CoT 표준), `30-analysis/global-trends.md` (글로벌 6사 → SK AX 영향)
  - 갱신: mixer-analysis / insight-cascade / peer-comparison (CoT + final_one_liner + 17 audit) · chat-orchestrator (deep_dive + forecast intent) · derived-metrics (peer_overview mode) · classification / card-composer / relevance / evidence (§6 audit table)
- **2026-05-14 (후속)** — 3-tier observability (trail / steps / langfuse_trace_id) 표준 + provenance-tracker LangfuseTraceLinker
- **2026-05-14 (후속2)** — KnowledgeCuration supervisor 신설 (5+1 → 6+1)
  - 신규 디렉토리 `25-knowledge-curation/`: README + compaction-agent + context-pack-builder + analysis-ledger
  - peer 단위 narrative 의 5 계층 (L0 raw → L1 daily → L2 weekly → L3 monthly → L4 quarterly canon) + Analysis Ledger 의 carry-over
  - 분석 4 agent 입력에 `_context_packs` 필드 추가 (in-process auto-fetch + cold_start fallback)
  - 초기 제안 Flyway 슬롯은 후속 정정 필요: 실제 적용 기준은 V19 (`analysis_ledger`), V20/V21 (`raw_articles` 중심 DB 관계 정비). daily/weekly/monthly/quarterly canon 은 다음 빈 슬롯에서 재배치
  - 도입 단계 K1 (ledger) → K2 (weekly) → K3 (context pack) → K4 (monthly) → K5 (quarterly) → K6 (Qdrant axis_knowledge)
- **2026-05-15** — DB 관계 표준 추가
  - 신규: `90-cross-cutting/db-relations.md`
  - backend Flyway `V20`/`V21` 기준으로 `raw_articles` 중심 FK/매핑 테이블, legacy 컬럼 유지 정책, application writer 전환 순서 문서화
