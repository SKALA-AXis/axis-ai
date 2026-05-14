# CompactionAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `CompactionAgent` (3-mode: weekly / monthly / quarterly 통합) |
| **Supervisor** | KnowledgeCuration |
| **상태** | 🆕 신규 (Phase K2/K4/K5 — 2026-05-14) |
| **Trigger** | Spring @Scheduled (3 cron entry — `00-supervisor-topology.md` §6 endpoint 표 참조) |

## 2. 책임

**한 줄**: peer 의 시간 축 narrative 를 LLM 으로 압축 — weekly digest / monthly profile / quarterly canon 3 horizon 을 동일 prompt 골격 + 다른 input mode 로 통합 생성.

**구체적 (3-mode)**:

1. **weekly** — 최근 7일 카드 ~30건 + 이전 weekly digest → 1.5K 토큰 narrative (delta_vs_prev 강조)
2. **monthly** — 직전 4 weekly digest + 그 달의 DART 공시 (있으면) → 3K 토큰 profile (strategy_label 갱신 + KPI 시계열)
3. **quarterly** — 직전 3 monthly + DART 분기 공시 → 5K 토큰 canon (DART 정합 quantitative_anchor + strategy 확정)

전 3 mode 공통: **반드시 이전 같은 horizon 의 결과물과 비교** → 단순 요약 X / **변화 추적** 강제 (PDF §9 = checklist 9).

## 3. 책임 NOT

- L0 raw 카드 / DART 원본 → Ingestion 이 담당
- L1 daily snapshot (산식) → ContextPackBuilder 의 sub method
- pack 조립 → ContextPackBuilder
- 분석 결과 carry-over → AnalysisLedger
- DART 공시 자체 fetch / IR PDF 파싱 → financial-linker / ir-parser (10-ingestion)

## 4. 입력 스펙

```python
class CompactionInput(TypedDict):
    peer_id: str
    horizon: Literal["weekly", "monthly", "quarterly"]

    # 시간 범위 (KST 절대 기준, checklist 5/6)
    since_kst: str                 # ISO8601 (분석 기간 시작)
    until_kst: str                 # ISO8601 (분석 기간 끝)
    fiscal_anchor: str | None      # quarterly 만 ("2026-1Q")

    # 입력 데이터 (horizon 에 따라 source 다름)
    cards: list[CardSummary]            # weekly 만 — L0 카드 ~30건
    prev_weekly_digests: list[WeeklyDigest]    # monthly 만 — 4개
    prev_monthly_profiles: list[MonthlyProfile] # quarterly 만 — 3개
    dart_filings: list[DartRef]         # monthly / quarterly 만
    prev_same_horizon: dict | None      # 이전 동일 horizon 결과 (delta 계산용)

    # SK AX 관점 carry (checklist 13)
    sk_ax_business_lines: list[str]     # ["ai_managed", "cloud_msp", "security", ...]
```

`CardSummary` = {id, title, summary_lines, event_type, sector, exposure_band, published_at_kst, source_links, financial_link}.

## 5. 출력 스펙

```python
class QuantitativeAnchor(TypedDict):
    """정량 수치 + 출처 marker (checklist 11/12)"""
    metric: str                      # "revenue_krwbn" / "card_count" / "partnership_count"
    value: float | int
    period: str                      # "2026-04-15 ~ 2026-05-13"
    delta_vs_prev_pct: float | None  # 이전 horizon 동일 metric 대비
    source_marker: str               # "[DART 2026-1Q]" / "[card_count: CN-...]" / "[자체 추정 v1]"

class WeeklyDigest(TypedDict):
    """L2"""
    peer_id: str
    week_iso: str                    # "2026-W19"
    since_kst: str
    until_kst: str

    narrative: str                   # 3~5 문단 (≤ 1,500 토큰)
    quantitative_anchors: list[QuantitativeAnchor]   # 5~10
    delta_vs_prev: list[str]         # 이전 주 대비 *변화점* 3~5개 (checklist 9 band 적용)
    carry_forward_keyfacts: list[str]    # 다음 weekly 빌드 시 보존할 fact 5~10개
    strategy_label_inferred: str     # 5종 enum (이 주 강한 신호 기준)
    confidence: float
    source_card_ids: list[str]       # 환각 방지 — 모든 anchor 가 카드에 trace
    reasoning_trail: list[ReasoningTrailItem]   # 3-tier observability Tier 1 (3~4 step)
    reasoning_steps: list[CoTStep]              # Tier 2 (optional)
    langfuse_trace_id: str | None               # Tier 3
    final_one_liner: str             # ≤ 100자

class MonthlyProfile(TypedDict):
    """L3"""
    peer_id: str
    year_month: str                  # "2026-04"
    since_kst: str
    until_kst: str

    narrative: str                   # 5~7 문단 (≤ 3,000 토큰)
    strategy_label: str              # 5종 enum (월 평균)
    strategy_trajectory: list[str]   # ["W17=Defensive", "W18=Aggressive", ...] — 변천 trace
    quantitative_anchors: list[QuantitativeAnchor]
    kpi_anchored: dict               # DART 분기 + 자체 추정 통합 KPI (혼합 marker)
    delta_vs_prev: list[str]
    carry_forward_keyfacts: list[str]
    confidence: float
    source_weekly_digest_ids: list[str]
    source_dart_refs: list[str]
    reasoning_trail: list[ReasoningTrailItem]
    reasoning_steps: list[CoTStep]
    langfuse_trace_id: str | None
    final_one_liner: str

class QuarterlyCanon(TypedDict):
    """L4 — 가장 강한 ground truth (DART 정합 후)"""
    peer_id: str
    fiscal_quarter: str              # "2026-1Q"
    since_kst: str
    until_kst: str
    dart_filed_at_kst: str           # DART 공시 fetched 시점

    narrative: str                   # 7~10 문단 (≤ 5,000 토큰)
    strategy_label_confirmed: str    # DART 정합 후 확정 (월별 inferred 보다 강한 권위)
    canonical_kpi: dict              # DART 출처만 — revenue / op_income / op_margin / captive / headcount / rd
    quantitative_anchors: list[QuantitativeAnchor]
    delta_vs_prev_quarter: list[str]
    business_segment_breakdown: list[dict]   # DART 사업 segment 별 매출 / 영업이익
    forward_signals: list[str]       # 다음 분기 시그널 (PeerComparison forecast 의 input 후보)
    carry_forward_keyfacts: list[str]
    confidence: float
    source_monthly_profile_ids: list[str]
    source_dart_rcept_nos: list[str] # DART rcept_no 직접 trace
    reasoning_trail: list[ReasoningTrailItem]
    reasoning_steps: list[CoTStep]
    langfuse_trace_id: str | None
    final_one_liner: str

class CompactionOutput(TypedDict):
    horizon: Literal["weekly", "monthly", "quarterly"]
    result: WeeklyDigest | MonthlyProfile | QuarterlyCanon
```

## 6. 알고리즘

### 6.1 공통 prompt 골격 (3-mode 통합)

```text
당신은 SK AX 사업전략팀의 peer monitoring narrative 정제 분석가다.
본 task 는 단순 요약이 아니라 *변화 추적* — 이전 동일 horizon 결과 대비 어떤 신호가
새롭게 출현/강화/약화 됐는지를 명시적으로 비교하라.

[Peer]
{peer_id} ({peer_canonical_facts.full_name})

[Horizon]
{horizon} | {since_kst} ~ {until_kst} (KST)
{fiscal_anchor 가 있으면: "회계 기준: " + fiscal_anchor}

[이전 동일 horizon 결과] (delta 비교 ground truth)
{prev_same_horizon.narrative 또는 "(이전 없음 — cold start)"}
이전 carry_forward_keyfacts: {prev_same_horizon.carry_forward_keyfacts}

[입력 데이터]
{horizon == "weekly" 일 때: 카드 ~30건 (id + title + summary + sector + event_type + exposure)}
{horizon == "monthly" 일 때: 4 weekly digest narrative + DART 공시 (있으면)}
{horizon == "quarterly" 일 때: 3 monthly profile narrative + DART 분기 공시 (rcept_no + 요약)}

[작성 규칙 — 02-prompt-design-checklist 17 요소 적용]

7. (단순 요약 금지) "기사 X 개" 표현 X — 패턴 / event_type / 변화 명시
8. (회사별 비교 기준) KPI 는 enum (revenue / op_income / op_margin / captive_ratio / headcount / rd_ratio)
9. (변화 감지) ±5% normal / >10% 유의 / >30% 급변 — delta_vs_prev 의 explicit 분류
10. (수익화 관점) SK AX 사업 line ({sk_ax_business_lines}) 별 긍정/중립/부정 — narrative 마지막 문단
11. (정량 우선) "성장 추세" 금지 → "QoQ +12.3%" / "card_count 28 (prev 21, +33%)"
12. (출처 prefix) 모든 anchor.source_marker — [DART 2026-1Q] / [card_count: CN-...] / [자체 추정 v1]
13. (전략 시사점, SK AX 화자) "삼성SDS가 X" 가 아니라 "삼성SDS X 는 SK AX 의 ___ 라인에 ___ 영향"
14. (출력 형식) JSON schema 명시
15. (우선순위) delta_vs_prev top 3~5 만 — 모든 변화 X
16. (리스크) carry_forward_keyfacts 에 "본 narrative 가 ___ 가정에 의존" 1~2개
17. (반복 추적) carry_forward_keyfacts 가 자동으로 다음 horizon 의 prev 가 됨

[3-tier observability 표준 — 02-prompt-design-checklist §4]
reasoning_trail (Tier 1, 사용자 default): 3~4 step (label ≤ 12자 + one_liner ≤ 80자).
  권장 label: "이전 대비" / "주요 변화" / "정량 검증" / "결론"
reasoning_steps (Tier 2): 5~8 step (per-cards/per-source 별 분석 → cross-source synthesis)
langfuse_trace_id: 런타임 매핑 (null 로 출력)

[환각 방지]
모든 quantitative_anchor 의 source_marker 가 입력 데이터에 trace 가능해야 함.
DART 출처 anchor 는 dart_rcept_no 명시 (예: "[DART 2026-1Q: rcept_no=20260415000123]").
입력에 없는 수치/이름 추가 금지. 불확실은 [자체 추정] prefix.

[JSON 출력]
{
  "narrative": "...",
  "quantitative_anchors": [{"metric":"...", "value":..., "period":"...", "delta_vs_prev_pct":..., "source_marker":"..."}],
  "delta_vs_prev": ["..."],
  "carry_forward_keyfacts": ["..."],
  "strategy_label_inferred": "Aggressive Expansion | Defensive Hold | Tech Pivot | Customer Lock-in | Cost Leadership",
  "final_one_liner": "≤ 100자",
  "confidence": 0.0~1.0,
  "reasoning_trail": [...],
  "reasoning_steps": [...],
  // horizon 별 추가 필드 (monthly: kpi_anchored / quarterly: canonical_kpi + business_segment_breakdown 등)
}
```

### 6.2 Horizon 별 input mode

```python
async def compact(input: CompactionInput) -> CompactionOutput:
    horizon = input["horizon"]
    prev = fetch_prev_same_horizon(input["peer_id"], horizon)

    if horizon == "weekly":
        input_data = format_cards(input["cards"])
    elif horizon == "monthly":
        input_data = format_weekly_digests(input["prev_weekly_digests"])
        input_data += format_dart(input["dart_filings"])
    elif horizon == "quarterly":
        input_data = format_monthly_profiles(input["prev_monthly_profiles"])
        input_data += format_dart_quarterly(input["dart_filings"])

    prompt = render_prompt(input, prev, input_data)
    resp = await llm_call_gpt4o(prompt)
    result = parse_json(resp)
    result = link_langfuse_trace(result)   # 90-cross-cutting/provenance-tracker §6.4
    upsert_to_pg(result, horizon)
    embed_and_index_to_qdrant(result)      # axis_knowledge collection
    clear_dirty_flag(input["peer_id"], horizon)
    return {"horizon": horizon, "result": result}
```

### 6.3 Qdrant indexing

L2/L3/L4 narrative 는 `axis_knowledge` collection 에 indexing:

```python
def embed_and_index(result, horizon):
    # narrative + carry_forward_keyfacts 를 합쳐 BGE-M3 dense+sparse
    text = result["narrative"] + " ".join(result["carry_forward_keyfacts"])
    vec = bge_m3.embed(text)
    payload = {
        "peer_id": result["peer_id"],
        "horizon": horizon,
        "period": result.get("week_iso") or result.get("year_month") or result.get("fiscal_quarter"),
        "strategy_label": result.get("strategy_label_confirmed") or result.get("strategy_label") or result.get("strategy_label_inferred"),
        "confidence": result["confidence"],
        "source_card_ids": result.get("source_card_ids", []),
        "pg_table": _table_for(horizon),
        "pg_pk": _pk_for(horizon, result),
    }
    qdrant.upsert("axis_knowledge", id=_hash(payload), vector=vec, payload=payload)
```

### 6.4 Prompt audit — 02-prompt-design-checklist 17 요소

(weekly / monthly / quarterly 동일 골격 사용 — 17 요소 모두 적용)

| # | 요소 | 충족 위치 |
|---|---|---|
| **1** | 역할 정의 | "SK AX 사업전략팀의 peer monitoring narrative 정제 분석가" |
| **2** | 추적 대상 기업 | input.peer_id (4 국내 + 6 글로벌 + sk_ax_self enum) |
| **3** | 추적 범위 | card 의 sector + event_type carry (cards 의 metadata) |
| **4** | 출처 우선순위 | DART > card source_links credibility > 자체 추정 — source_marker 강제 |
| **5** | 분석 기간 | since_kst / until_kst / fiscal_anchor 절대 기준 |
| **6** | 최신성 검증 | KST timestamp 명시 — "최근" 같은 상대 표현 금지 |
| **7** | 단순 뉴스 요약 금지 | delta_vs_prev 강제 (변화 추적) |
| **8** | 회사별 비교 기준 | kpi_anchored / canonical_kpi 의 enum |
| **9** | 변화 감지 기준 | delta_vs_prev_pct 의 ±5/10/30% band |
| **10** | 수익화 관점 | narrative 마지막 문단 — SK AX 사업 line 별 긍정/중립/부정 |
| **11** | 정량 우선 | quantitative_anchors 필드 강제 |
| **12** | 공식 vs 추정 | source_marker prefix 의 [DART] / [card] / [자체 추정] |
| **13** | 전략 시사점 | narrative 의 SK AX 화자 pattern |
| **14** | 출력 형식 | WeeklyDigest / MonthlyProfile / QuarterlyCanon TypedDict |
| **15** | 우선순위 | delta_vs_prev top 3~5 강제 |
| **16** | 리스크 분석 | carry_forward_keyfacts 의 가정 명시 |
| **17** | 반복 추적 구조 | carry_forward → 다음 horizon prev — 시스템 레벨 |
| **CoT** | reasoning_trail (Tier 1) + reasoning_steps (Tier 2) + trace_id (Tier 3) | 02-prompt-design-checklist §4 |
| **결론** | final_one_liner ≤ 100자 | strict |

→ **17/17 + CoT + final_one_liner** 모두 충족.

### 6.5 Dirty trigger 우선순위

```python
def select_peers_for_recompute(horizon):
    """exposure_band=high 카드 진입한 peer 를 우선 처리."""
    rows = pg.query("""
        SELECT peer_id, urgent, cards_changed_at
        FROM peer_dirty_flags
        WHERE _horizon_due(?, last_recompute_at) = true
        ORDER BY urgent DESC, cards_changed_at DESC
    """, horizon)
    return [r["peer_id"] for r in rows]
```

`urgent=true` 인 peer 는 batch 우선 처리. cron 이 horizon 끝나도 다 못 돌면 다음 cron 에서 backlog.

## 7. LLM 모델 + token 예산

| Mode | 모델 | input | output | 호출/주기 | 호출 비용 | 일평균 |
|---|---|---|---|---|---|---|
| weekly | gpt-4o | ~10K | ~1.5K | 10 peer × 주 1 | ~₩50 | **~₩70/일** |
| monthly | gpt-4o | ~8K | ~3K | 10 peer × 월 1 | ~₩100 | **~₩30/일** |
| quarterly | gpt-4o | ~15K | ~5K | 10 peer × 분기 1 | ~₩200 | **~₩20/일** |
| **합계** | | | | | | **~₩120/일** |

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| 입력 카드 0건 (weekly) | "cold week" mode — narrative 에 "관망" 명시 + delta_vs_prev=[] + confidence ≤ 0.5 |
| prev_same_horizon=None (cold start) | "cold start" mode — delta_vs_prev=[] + carry_forward_keyfacts 만 새로 빌드 |
| DART 공시 결측 (quarterly) | warn + canonical_kpi=None + fallback to monthly aggregate |
| LLM JSON parse 실패 | retry 1회 → 실패 시 stub (narrative="(생성 실패)" + confidence=0.0) + dirty flag 유지 |
| Source 검증 실패 (anchor 의 source_marker 가 입력에 trace 안 됨) | 해당 anchor 자동 drop + warning 로그 |
| token 한도 초과 | input cards prune (exposure_score top-20 만), 그래도 초과 시 narrative ↓ (1K → 1K 유지) |
| Qdrant indexing 실패 | PG 는 INSERT OK, Qdrant retry 1회 → 실패 시 next cron 재시도 |

## 9. 외부 의존성

- **DB (PG)**: `card_news` (READ), `peer_financials` (READ), `peer_weekly_digest` / `peer_monthly_profile` / `peer_quarterly_canon` (UPSERT), `peer_dirty_flags` (UPDATE)
- **Vector DB (Qdrant)**: `axis_knowledge` (신규 collection — UPSERT)
- **외부 API**: OpenAI gpt-4o
- **Sibling agents**: financial-linker (DART read), ir-parser (PDF parse)

### Flyway V19~V21 schema (요약)

```sql
-- V20
CREATE TABLE peer_weekly_digest (
    peer_id        VARCHAR(50)  NOT NULL,
    week_iso       VARCHAR(10)  NOT NULL,   -- "2026-W19"
    since_kst      DATE         NOT NULL,
    until_kst      DATE         NOT NULL,
    narrative      TEXT         NOT NULL,
    payload        JSONB        NOT NULL,   -- 전체 WeeklyDigest schema
    strategy_label VARCHAR(30),
    confidence     REAL         NOT NULL,
    created_at     TIMESTAMPTZ  DEFAULT now(),
    PRIMARY KEY (peer_id, week_iso)
);
CREATE INDEX idx_wdigest_peer_recent ON peer_weekly_digest (peer_id, week_iso DESC);

CREATE TABLE peer_monthly_profile (
    peer_id      VARCHAR(50) NOT NULL,
    year_month   VARCHAR(7)  NOT NULL,    -- "2026-04"
    payload      JSONB NOT NULL,
    strategy_label VARCHAR(30),
    confidence   REAL NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (peer_id, year_month)
);

-- V21
CREATE TABLE peer_quarterly_canon (
    peer_id        VARCHAR(50) NOT NULL,
    fiscal_quarter VARCHAR(10) NOT NULL,   -- "2026-1Q"
    payload        JSONB NOT NULL,
    strategy_label_confirmed VARCHAR(30),
    canonical_kpi  JSONB NOT NULL,
    confidence     REAL NOT NULL,
    dart_filed_at_kst TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (peer_id, fiscal_quarter)
);

CREATE TABLE peer_canonical_facts (
    peer_id   VARCHAR(50) NOT NULL,
    fact_key  VARCHAR(50) NOT NULL,        -- "hq_location" / "parent_company" / "biz_focus"
    fact_value TEXT NOT NULL,
    source_marker TEXT NOT NULL,
    last_reviewed_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (peer_id, fact_key)
);
```

## 10. State 흐름

KnowledgeCurationState 의 horizon 별 sub-flow (README §12 참조). 매 cron 호출이 1 super-step:

```python
def compaction_node(state: KnowledgeCurationState):
    horizon = state["trigger"]
    peers_to_recompute = select_peers_for_recompute(horizon)
    results = []
    for peer_id in peers_to_recompute:
        try:
            inp = build_compaction_input(peer_id, horizon, state)
            out = compact(inp)
            results.append(out)
        except Exception as e:
            state["errors"].append(f"{peer_id}/{horizon}: {e}")
    return {
        **state,
        f"{horizon}_digests" if horizon == "weekly" else f"{horizon}_profiles" if horizon == "monthly" else f"{horizon}_canons": results,
    }
```

## 11. Provenance + Confidence

- **Provenance**: 90-cross-cutting/provenance-tracker §6.2 decorator + §6.4 LangfuseTraceLinker
  - `agent="CompactionAgent"`, `agent_method="weekly|monthly|quarterly"`
  - `prompt_version="compact-w-v1.0"` / `"compact-m-v1.0"` / `"compact-q-v1.0"` — 각 horizon 별 독립 버전
  - `source_card_ids` / `source_dart_refs` / `source_weekly_digest_ids` etc
- **Confidence**:
  - weekly: avg(card credibility) × 0.5 + LLM self-confidence × 0.5
  - monthly: avg(weekly confidence) × 0.4 + DART presence (0/1) × 0.3 + LLM self × 0.3
  - quarterly: DART presence × 0.5 + avg(monthly confidence) × 0.3 + LLM self × 0.2

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit (weekly) | 카드 30건 + prev digest 있음 | delta_vs_prev ≥ 3, carry_forward ≤ 10, narrative ≤ 1.5K tokens |
| Unit (weekly cold) | 카드 30건 + prev=None | delta_vs_prev=[], carry_forward 만 build, confidence 정상 |
| Unit (monthly) | 4 weekly + DART 공시 1 | strategy_label confirmed, canonical_kpi 의 source_marker=[DART] |
| Unit (quarterly) | 3 monthly + DART rcept_no | canonical_kpi 100% [DART] prefix, business_segment_breakdown 채워짐 |
| Integration | weekly → monthly → quarterly 3 turn 연속 | carry_forward chain 이 strategy 변천 trace 가능 |
| Edge | DART 결측 (quarterly) | warn + canonical_kpi=None + fallback monthly aggregate |
| Edge | LLM 환각 (anchor.source_marker 의 출처가 입력에 없음) | 해당 anchor drop + warning 로그 |
| Edge | exposure_band=high 카드 진입 후 dirty | next cron 에서 그 peer 우선 처리 |

## 13. 모니터링

- **pipeline_logs.step**: `compact_weekly` / `compact_monthly` / `compact_quarterly`
- **KPI**:
  - weekly 호출 성공률 ≥ 95%
  - delta_vs_prev 평균 길이 3~5 (너무 짧으면 narrative 부실)
  - carry_forward_keyfacts 평균 5~10
  - confidence 평균 ≥ 0.65 (weekly) / ≥ 0.70 (monthly) / ≥ 0.80 (quarterly)
  - anchor source_marker [DART] 비율 ≥ 80% (quarterly)
  - 환각 drop 발생 빈도 ≤ 5%
- **Token budget**: ~₩120/일 (compaction 만)
- **Langfuse dashboard**: 3 prompt_version 별 token / latency / confidence trend

## 14. 구현 메모 + Changelog

### 핵심 파일

- `src/agents/compaction_agent.py` (신규 — 3-mode 진입점 `compact(input)`)
- `src/agents/compaction_prompts.py` (3 horizon prompt template — version 별 분리)
- `src/db/knowledge_curation.py` (PG INSERT/UPSERT helper)
- `src/rag/knowledge_indexer.py` (axis_knowledge Qdrant indexer)

### Changelog

- **v1 (2026-05-14)** — 신설. 5 계층 아키텍처의 LLM compaction 단일 진입점. 3 horizon (weekly/monthly/quarterly) 통합.
