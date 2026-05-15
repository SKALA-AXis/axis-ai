# KnowledgeCuration Supervisor — Sub-system Overview

> **위치**: `axis-ai/design/25-knowledge-curation/`
> **신설**: 2026-05-14 · **버전**: v1
> **소속**: 6+1 supervisor 중 **#6 (신규)** — 5+1 → 6+1 확장
> **상위 문서**: `00-supervisor-topology.md` v2, `02-prompt-design-checklist.md` v2

## 1. 왜 필요한가 — 현재 구조의 한계

현재 분석 agent (Mixer / Insight / PeerComparison / GlobalTrends / Briefing) 는 매 호출마다 **카드 raw → Qdrant retrieval → LLM** 으로 시간 축 컨텍스트를 *재발견* 한다. 결과적으로:

| 한계 | 영향 |
|---|---|
| 맥락 단편화 | "삼성SDS Q1 매출 YoY -3%" 가 매 분석마다 카드/DART 에서 재 retrieval → 재 추론 |
| 누적 학습 없음 | 어제 Insight 의 결론 ("Aggressive Expansion") 을 오늘 Mixer 가 모름 |
| 추론 깊이 한계 | 카드 6건만 보고 추론. 지난 6개월 strategy 변천은 시야 밖 |
| 일관성 흔들림 | 분석마다 retrieval 결과가 RRF 점수 / 시점 차이로 미세 변동 → 동일 peer 동일 시점 다른 결론 가능 |
| 재계산 비용 | 동일 fact 가 매 LLM call 마다 다시 처리됨 |

**해결**: 시간 축으로 압축된 *materialized knowledge layer* — periodic compaction + analysis ledger 도입.

## 2. 핵심 idea — 5 계층 + Analysis Ledger

```
┌─────────────────────────────────────────────────────────────────────────┐
│  L0  Raw 카드 / DART 공시        PG raw_articles + card_news + Qdrant   │ 매시
│  L1  Daily snapshot              peer_daily_snapshot                    │ 매일 23:55 (산식)
│  L2  Weekly digest               peer_weekly_digest                     │ 일요일 23:55 (LLM)
│  L3  Monthly profile             peer_monthly_profile                   │ 매월 1일 (LLM)
│  L4  Quarterly canon             peer_quarterly_canon                   │ 분기 직후 (LLM + DART)
│  ──                                                                    │
│  Analysis Ledger                 analysis_ledger                        │ 분석 종료 시 INSERT
│  Canonical facts                 peer_canonical_facts                   │ 분기 admin review
└─────────────────────────────────────────────────────────────────────────┘
```

위 → 아래로 *압축율 ↑, 시간 폭 ↑*. 분석 agent 는 raw 가 아닌 이 정제된 stack 을 `PeerContextPack` 로 받는다.

## 3. Supervisor 등록 — 6+1 supervisor

| # | Supervisor | Trigger | LLM (cycle) | 책임 |
|---|---|---|---|---|
| 1 | Ingestion | 매시 정각 | ~80 | raw fetch → card_news + evidence |
| 2 | Enrichment | 매시 후 + nightly 02:00 | ~4 | 카드 derivative (키워드/그래프/word cloud/metrics) |
| **3** | **KnowledgeCuration** *(신규)* | daily 23:55 / 일요일 / 매월 1일 / 분기 직후 | ~10 (weekly+monthly+quarterly) | **peer 단위 narrative 압축 + ledger + context pack** |
| 4 | Analysis | user request | 3~10 | Insight / Mixer / Peer / Global / Link verify |
| 5 | UserQuery | user request | 1~5 | Search + Rerank + Answer + Chat |
| 6 | WeakSignal | 월 09:00 | ~5 | 패턴 + 이상 + 라우팅 |
| 7 | Briefing | user POST | ~11 | 5-phase 문서 생성 |

### 왜 별도 supervisor (Enrichment 의 일부 아님)?

| 차원 | Enrichment | KnowledgeCuration |
|---|---|---|
| **단위** | 카드 (article) | peer (4 + 6 글로벌) |
| **결과물** | 키워드 / 그래프 / metrics — *derivative* | narrative + strategy + KPI 시계열 — *knowledge* |
| **cadence** | nightly 02:00 단일 | daily / weekly / monthly / quarterly 4종 |
| **LLM 의존** | ~₩4/cycle (word cloud 라벨링) | ~₩120/일 (compaction) |
| **consumer** | frontend dashboard | 분석 agent prompt |
| **격리 가치** | — | trigger 다양 + LLM density 다름 + 분석 agent 직접 dependency |

## 4. 디렉토리 구조

```
25-knowledge-curation/
├── README.md                     ← 본 파일 (개요 + supervisor + 5 계층 + dataflow)
├── compaction-agent.md           ← weekly / monthly / quarterly LLM 압축 (3-mode 통합)
├── context-pack-builder.md       ← L0~L4 stack 을 PeerContextPack 로 조립 (산식 + 산식)
└── analysis-ledger.md            ← 분석 결과 carry-over (INSERT, 산식 only)
```

3 agent + 1 README. 모든 agent 는 3-tier observability 표준 (02-prompt-design-checklist §4) 준수.

## 5. End-to-end dataflow

### 5.1 압축 흐름 (위로 갈수록 시간 축 ↑)

```
매시 ingestion → L0 raw → (peer 별 dirty mark)
                           ↓
매일 23:55       L0 그날 카드 ────► L1 daily snapshot
                                    (산식: top-5 + keywords + KPI hash)
                           ↓
일요일 23:55     L0 7일 + prev L2 ─► L2 weekly digest
                                     (LLM: narrative + delta_vs_prev + carry_forward)
                           ↓
매월 1일         4×L2 + DART ──────► L3 monthly profile
                                     (LLM: strategy_label + KPI 추세)
                           ↓
분기 후          3×L3 + DART 공시 ─► L4 quarterly canon
                                     (LLM + DART 정합: canonical_kpi + strategy 확정)
```

### 5.2 분석 consume 흐름

```
분석 호출 (Mixer / Insight / Peer / Global / Briefing)
         ↓
ContextPackBuilder.assemble(peer_id, snapshot_at)   ← PG read 산식, LLM X
         ↓
PeerContextPack {
    recent_cards (L0 7일 top-5)
    weekly_digest (L2 가장 최근)
    monthly_profile (L3 이번 달)
    quarterly_canon (L4 가장 최근 분기)
    kpi_timeseries (8 분기)
    canonical_facts (불변)
    analysis_ledger_top5 (confidence ≥ 0.7)
}
         ↓
분석 LLM 이 context_pack + 새 카드를 *함께* 보고 추론
         ↓
final_one_liner + confidence → AnalysisLedger.insert()
         ↓
다음 분석에 carry-over
```

### 5.3 갱신 / dirty handling

| 시나리오 | 동작 |
|---|---|
| Ingestion 매시 마감 | peer 별 `peer_dirty_flags.cards_changed_at` 갱신 |
| exposure_band=high 카드 진입 | `peer_dirty_flags.urgent=true` → 다음 hour 에 L2 partial recompute 우선순위 ↑ |
| L2 갱신 | dirty flag clear + L3 dirty mark |
| L4 갱신 | DART 공시 fetch 성공 시 trigger (분기 평균 50일 후) |

## 6. 신규 DB 테이블 (Flyway slot 정정)

기존 제안은 V19+ 를 knowledge-curation 전용 슬롯으로 가정했지만, 실제 backend Flyway 기준은 V19=`analysis_ledger`, V20/V21=`raw_articles` 중심 DB 관계 정비다. L1~L4 compaction 테이블은 다음 빈 migration slot 에서 재배치한다.

| Version | 테이블 | 용도 |
|---|---|---|
| V19 | `analysis_ledger` | 분석 결과 carry-over (BIGSERIAL) |
| V21 | `analysis_ledger_card_news` | ledger ↔ card_news FK mapping |
| V21 | `analysis_ledger_peer_companies` | ledger ↔ peer_companies FK mapping |
| TBD | `peer_daily_snapshot` | L1 (PK: peer_id + snapshot_date) |
| TBD | `peer_dirty_flags` | dirty tracking (PK: peer_id) |
| TBD | `peer_weekly_digest` | L2 (PK: peer_id + week_iso) |
| TBD | `peer_monthly_profile` | L3 (PK: peer_id + year_month) |
| TBD | `peer_quarterly_canon` | L4 (PK: peer_id + fiscal_quarter) |
| TBD | `peer_canonical_facts` | 불변 fact (PK: peer_id + fact_key) |

자세한 schema 는 각 agent design 파일의 §9 외부 의존성 절 참조.

## 7. Qdrant collection 확장

| Collection | 내용 | 용도 |
|---|---|---|
| `axis_main` (기존) | L0 카드 dense+sparse | 키워드 유사도 — 매 분석 시 |
| `axis_history` (기존) | L0 카드 1년치 sparse | weak signal |
| **`axis_knowledge`** *(신규)* | L2/L3/L4 narrative dense+sparse | "지난 1년 중 'partnership' 빈도 강했던 시기" 시간 축 검색 |

embedding 비용: digest/profile/canon 은 raw 카드 대비 훨씬 적은 row 수 (10 peer × 52주 + 12월 + 4분기 = 720 row/년) — embedding 부담 무시 가능.

## 8. 비용 / 성능 영향 (10 peer = 4 국내 + 6 글로벌)

### Compaction LLM (신규 비용)

| Compaction | 호출/주기 | 토큰/호출 | 호출 당 비용 | 일평균 |
|---|---|---|---|---|
| Weekly | 10 × 주 1회 | 10K in + 1.5K out | ~₩50 | **~₩70/일** |
| Monthly | 10 × 월 1회 | 8K in + 3K out | ~₩100 | **~₩30/일** |
| Quarterly | 10 × 분기 1회 | 15K in + 5K out | ~₩200 | **~₩20/일** |
| **합계 (compaction)** | | | | **~₩120/일** |

### 분석 agent 영향 (단일 호출 기준)

| 항목 | Before | After | Delta |
|---|---|---|---|
| Input token | 카드 raw 10건 ≈ 10K | context_pack 12K + 새 카드 5건 5K = 17K | +7K |
| Input 비용 | ~₩30 | ~₩50 | +₩20 |
| Retrieval 호출 | 2~3 Qdrant + 5 PG | 1 PG (pack fetch) + 1 Qdrant | **-50%** |
| Latency | ~8초 | ~6초 (retrieval 절감) | **-25%** |
| 출력 품질 | 카드 기반 추론 | 시간 축 narrative + 새 카드 | **+** |

### Net 영향 (일 가중 평균)

- 신규 compaction LLM: **+₩120/일**
- 분석 input token 증가: **+₩60/일** (3 supervisor × 10 호출 가정)
- Retrieval 호출 감소: **-₩30/일** (Qdrant + PG 부하 분담)
- **Net: +₩150/일 (~+10% LLM 예산)** 으로 정확도 + 일관성 + 시간 축 추론 깊이 + 분석 일관성 확보

## 9. 도입 단계 (incremental, 각 단계가 독립 가치)

| Phase | 범위 | LLM | 가치 | 권장 PR |
|---|---|---|---|---|
| **K1** | `analysis_ledger` + `AnalysisLedger.insert()` middleware | 0 | 분석 결과가 다음 분석에 보임 — 가장 빠른 win | 별도 PR |
| **K2** | L2 weekly digest + Compaction (weekly only) | weekly | 한 주 narrative 압축 → 분석 input 일부 | 별도 PR |
| **K3** | ContextPackBuilder + 분석 4 agent 의 input spec 갱신 | 0 | 분석 agent 가 pack 받기 시작 | + K2 |
| **K4** | L3 monthly profile | monthly | 월 단위 strategy 변천 추적 | 별도 PR |
| **K5** | L4 quarterly canon + DART 정합 | quarterly | PDF §4 forecast 정확도 ↑ | 별도 PR |
| **K6** | `axis_knowledge` Qdrant collection | 0 (embed only) | 시간 축 유사도 검색 | 별도 PR |

각 Phase 가 독립적으로 production-ready. K1+K2+K3 까지 도입하면 *충분히 똑똑한* 1차 버전.

## 10. 위험 + 완화

| 위험 | 영향 | 완화 |
|---|---|---|
| **Compaction 환각** — digest 가 원본에 없는 사실 추가 | 잘못된 결론 carry-over | `source_card_ids` 강제 + admin sampling QA + checklist 11/12 (정량 + 출처 prefix) |
| **Stale digest** — 큰 사건 직후 갱신 안 됨 | 분석 결과 시차 | exposure_band=high 카드 진입 시 즉시 dirty + 다음 hour 에 L2 partial recompute 우선순위 ↑ |
| **Cold start** — pack 없는 peer | 분석 fallback 필요 | pack=None 이면 기존 retrieval 방식으로 graceful fallback (분석 agent 가 `context_pack: Optional[]` 처리) |
| **Confabulation 누적** — digest → ledger → 다음 digest 가 잘못된 ledger 인용 | 시스템 drift | confidence ≥ 0.7 threshold + 매 분기 admin canon review (L4 가 ledger 보다 강한 ground truth) |
| **Token 폭증** | 비용 ↑ | pack size cap (12K tokens) + 오래된 ledger 자동 폐기 (90일) |
| **DART 분기 공시 지연** | L4 갱신 지연 | quarterly 는 lazy trigger (실제 DART 공시 fetched 시점) — 분기 종료 후 평균 50일 |
| **Multi-peer 분석 (Mixer)** | pack 여러 개 합치면 token 폭증 | 카드의 peer_id 별로 pack 5~6개 필요 시 *recent_cards + ledger_top3 only* 줄임형 pack 옵션 |
| **Compaction prompt drift** | 시간 흐름에 따라 narrative 양식이 바뀜 | `prompt_version` provenance 추적 (90-cross-cutting/provenance-tracker §6.3) + 분기마다 prompt diff review |

## 11. PDF 17 요소와의 정합

본 supervisor 는 모든 17 요소를 *더 강하게* 지원:

| # | 요소 | KnowledgeCuration 의 기여 |
|---|---|---|
| 5 | 분석 기간 | context pack 의 `snapshot_at_kst` + `kpi_timeseries` 가 명시적 시간 축 carry |
| 6 | 최신성 검증 | pack_version + dirty flag + last_recompute_at 추적 |
| 7 | 단순 뉴스 요약 금지 | weekly digest 가 *delta_vs_prev* 강제 → 변화 패턴 추출 |
| 8 | 회사별 비교 기준 | monthly_profile.kpi_anchored + strategy_label enum 통일 |
| 9 | 변화 감지 기준 | weekly digest 의 ±5/10/30% band 분류 (PeerComparison 과 동일 enum) |
| 11 | 정량 수치 우선 | digest/profile/canon 의 quantitative_anchors 필드 강제 |
| 12 | 공식 vs 추정 구분 | quarterly_canon 이 [DART 공식] / weekly_digest 가 [기사 인용/자체 추정] 마킹 |
| 16 | 리스크 분석 | analysis_ledger 의 superseded_by 로 과거 결론 retraction tracking |
| 17 | 반복 추적 구조 | ledger 자체가 *연속 분석의 trail* — PDF §6 꼬리 물기 의 system-level 구현 |

## 12. State 모델 (LangGraph)

```python
class KnowledgeCurationState(TypedDict):
    trigger: Literal["daily", "weekly", "monthly", "quarterly"]
    peer_ids: list[str]            # 처리할 peer (보통 4 + 6 = 10)
    snapshot_at_kst: str           # KST ISO8601
    fiscal_anchor: str | None      # quarterly 만 사용 ("2026-1Q")

    # 산식 출력
    daily_snapshots: list[dict]    # L1
    dirty_flags: list[dict]        # peer_dirty_flags 갱신 결과

    # LLM 출력
    weekly_digests: list[dict]     # L2 (LLM)
    monthly_profiles: list[dict]   # L3 (LLM)
    quarterly_canons: list[dict]   # L4 (LLM)

    # Cross-cutting
    errors: list[str]
    metrics: dict                  # tokens/cost/latency per horizon
```

## 13. 외부 endpoint (Spring @Scheduled trigger)

axis-ai FastAPI 신규:

```text
POST /knowledge/daily-snapshot       Spring cron "0 55 23 * * *"
POST /knowledge/weekly-digest        Spring cron "0 55 23 ? * SUN"
POST /knowledge/monthly-profile      Spring cron "0 0 1 1 * ?"
POST /knowledge/quarterly-canon      Spring cron "0 0 2 1 1,4,7,10 ?" (분기 시작 + 50일 delay 는 별도)
POST /knowledge/recompute            on-demand (admin) — peer_id + horizon
GET  /knowledge/context-pack         on-demand (다른 agent / admin) — peer_id
```

backend `KnowledgeService.java` 가 Spring scheduler 에서 4 endpoint 호출. 결과는 PG 에 직접 INSERT (axis-ai 가 self-managed).

## 14. Changelog

- **v1 (2026-05-14)** — 신설. PDF 사업전략팀 추가 질의 회신 후 사용자의 후속 요청 ("핵심 컨텍스트를 통합 관리 + 지속 갱신 + 효율 메모리 사용") 반영. 5 계층 + Analysis Ledger 아키텍처.
