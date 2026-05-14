# ContextPackBuilder — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `ContextPackBuilder` |
| **Supervisor** | KnowledgeCuration |
| **상태** | 🆕 신규 (Phase K1/K3 — 2026-05-14) |
| **Trigger** | (1) 매일 23:55 `/knowledge/daily-snapshot` Spring cron — L1 산출. (2) 분석 호출 직전 in-process — pack 조립 |

## 2. 책임

**한 줄**: L0~L4 + canonical facts + analysis ledger 를 **단일 `PeerContextPack` 객체** 로 조립하여 분석 agent prompt 에 박는다. LLM 미사용 — 전부 PG read 산식.

**구체적 (2-mode)**:

1. **`daily_snapshot()`** — 매일 23:55 cron: 그날 카드를 산식으로 압축하여 `peer_daily_snapshot` 에 INSERT (L1). 가장 짧은 압축 단계 — top-5 card_ids + keywords + KPI delta hash.
2. **`assemble(peer_id, snapshot_at_kst)`** — 분석 agent 가 in-process 호출. PG read 만으로 PeerContextPack 객체 즉시 build (~50ms). LLM 미사용.

## 3. 책임 NOT

- LLM 압축 (L2/L3/L4) — CompactionAgent
- 분석 결과 carry-over INSERT — AnalysisLedger
- 카드 retrieval (Qdrant 검색) — 분석 agent 가 별도 호출
- Canonical facts 갱신 — admin manual / 분기 review (산식 X)

## 4. 입력 스펙

### 4.1 daily_snapshot mode

```python
class DailySnapshotInput(TypedDict):
    target_date_kst: str        # ISO8601 (보통 cron 호출일)
    peer_ids: list[str] | None  # None 이면 4 국내 + 6 글로벌 전체
```

### 4.2 assemble mode

```python
class AssembleInput(TypedDict):
    peer_id: str
    snapshot_at_kst: str          # 보통 분석 호출 시점 (UTC 가능, 내부에서 KST 변환)
    horizon_overrides: dict | None
    # {
    #   "recent_cards_days": int (default 7),
    #   "ledger_top_n": int (default 5),
    #   "ledger_min_confidence": float (default 0.7),
    #   "include_canonical_facts": bool (default True),
    #   "kpi_quarters": int (default 8 = 2년),
    # }
    slim_mode: bool               # True 면 분석 input 의 token 부담 줄임 (Mixer 다중 peer 등)
```

## 5. 출력 스펙

### 5.1 daily_snapshot 출력 (DB INSERT)

```python
class DailySnapshot(TypedDict):
    """L1 — 산식 only, LLM X. 압축율 가장 낮음 (그날 카드 ~20건 → 10 row JSON)."""
    peer_id: str
    snapshot_date: str             # "2026-05-14"
    card_count: int
    top5_card_ids: list[str]       # exposure_score top-5
    keywords_top10: list[str]      # KeywordExtractionAgent 결과 reuse (enrichment_cache)
    event_distribution: dict       # {partnership:2, ma:0, tech:5, ...}
    sector_distribution: dict
    avg_exposure_score: float
    kpi_delta_hash: str            # DART 분기 갱신 시 변경 — pack assembly 시 cache invalidation 용
    created_at: str
```

### 5.2 assemble 출력 (in-memory)

```python
class CardSummary(TypedDict):
    """L0 카드의 요약 (raw 본문 X — token 부담 감소)"""
    id: str
    title: str
    summary_lines: list[str]
    event_type: str
    sector: str
    exposure_band: str
    published_at_kst: str
    source_tier: int               # 1/2/3

class KPIPoint(TypedDict):
    fiscal_period: str             # "2026-1Q"
    revenue_krwbn: float
    op_income_krwbn: float
    op_margin_pct: float
    captive_ratio_pct: float | None
    headcount: int | None
    rd_ratio_pct: float | None
    source_marker: str             # "[DART rcept_no=...]" 강제

class CanonicalFact(TypedDict):
    fact_key: str                  # "hq_location" / "parent_company" / "biz_focus" / "founded_year"
    fact_value: str
    source_marker: str
    last_reviewed_at: str

class LedgerEntry(TypedDict):
    analysis_id: str
    analysis_type: str             # "insight" / "mixer" / "peer" / "global"
    conclusion_one_liner: str
    strategy_label: str | None
    confidence: float
    created_at: str
    source_card_ids: list[str]

class PeerContextPack(TypedDict):
    """분석 agent prompt 에 박히는 통합 컨텍스트 객체. LLM 미사용 — PG read 산식.

    Token 예산 (slim_mode=False 기본):
      recent_cards 5건 ≈ 2,000
      weekly_digest ≈ 1,500
      monthly_profile ≈ 3,000
      quarterly_canon ≈ 5,000
      kpi_timeseries 8 quarter ≈ 500
      canonical_facts ≈ 200
      analysis_ledger_top5 ≈ 500
      ─────────────────────────────
      합계 ≈ 12,700 tokens
    """
    peer_id: str
    snapshot_at_kst: str
    pack_version: int              # peer 별 monotonic — dirty 시 +1
    pack_built_at_kst: str

    # 시간 축 layer (위 → 아래 압축율 ↑ 시간 폭 ↑)
    recent_cards: list[CardSummary]            # L0 (analysis_period.recent_cards_days 일 — default 7일 top-5)
    daily_snapshots_last7: list[DailySnapshot] # L1 (slim_mode 면 omit)
    weekly_digest: WeeklyDigest | None         # L2 — 가장 최근 주 (있으면)
    monthly_profile: MonthlyProfile | None     # L3 — 이번 달 (있으면)
    quarterly_canon: QuarterlyCanon | None     # L4 — 가장 최근 분기 (있으면)

    # KPI 시계열 (8 분기 = 2년) — DART 출처
    kpi_timeseries: list[KPIPoint]

    # 누적 학습
    canonical_facts: list[CanonicalFact]       # 본사 / 모회사 / 사업 영역 (불변)
    analysis_ledger_top5: list[LedgerEntry]    # confidence >= 0.7, 최근 5건

    # 메타
    coverage: dict
    # {
    #   "has_weekly": bool, "has_monthly": bool, "has_quarterly": bool,
    #   "kpi_quarters_available": int,
    #   "ledger_entries_count": int,
    #   "cold_start": bool   # 모든 L2~L4 결측이면 True → 분석 agent fallback 모드
    # }
    cached: bool                   # True 면 동일 pack_version 재사용
```

### 5.3 slim_mode

`slim_mode=True` 일 때 (Mixer 가 5+ peer 동시 분석할 때):

- `recent_cards`: 3건 만 (token 절반)
- `daily_snapshots_last7`: omit
- `weekly_digest`: narrative 만, anchors/delta 생략
- `monthly_profile`: strategy_label + kpi_anchored 만, narrative 생략
- `quarterly_canon`: canonical_kpi + strategy_label_confirmed 만
- `kpi_timeseries`: 최근 4 quarter (1년) 만
- `analysis_ledger_top5` → top 3

총 token: ~5,000 (기본 12.7K 의 ~40%).

## 6. 알고리즘

### 6.1 daily_snapshot (산식)

```python
def daily_snapshot(input: DailySnapshotInput) -> list[DailySnapshot]:
    target = parse_kst_date(input["target_date_kst"])
    peers = input["peer_ids"] or MONITORED_PEERS

    results = []
    for peer_id in peers:
        cards = fetch_peer_cards(peer_id, date=target)
        if not cards:
            continue
        snapshot = {
            "peer_id": peer_id,
            "snapshot_date": target.isoformat(),
            "card_count": len(cards),
            "top5_card_ids": [c.id for c in sorted(cards, key=lambda c: c.exposure_score, reverse=True)[:5]],
            "keywords_top10": fetch_keywords_top10(peer_id, target),    # enrichment_cache reuse
            "event_distribution": Counter(c.event_type for c in cards),
            "sector_distribution": Counter(c.sector for c in cards),
            "avg_exposure_score": np.mean([c.exposure_score for c in cards]),
            "kpi_delta_hash": _hash_latest_dart(peer_id),
        }
        upsert(snapshot)
        # peer_dirty_flags 갱신 (urgent if exposure>=high present)
        if any(c.exposure_band == "high" for c in cards):
            mark_urgent(peer_id)
        else:
            mark_dirty(peer_id)
        results.append(snapshot)
    return results
```

### 6.2 assemble (산식, in-process)

```python
def assemble(input: AssembleInput) -> PeerContextPack:
    peer_id = input["peer_id"]
    snapshot_at = parse_kst(input["snapshot_at_kst"])
    overrides = input.get("horizon_overrides") or {}
    slim = input.get("slim_mode", False)

    # 1. Cache check — pack_version 일치 + last_dirty < snapshot_at 이면 cache hit
    cached = _check_cache(peer_id, snapshot_at, slim)
    if cached:
        return cached

    # 2. L0~L4 PG read (병렬 가능)
    recent_days = overrides.get("recent_cards_days", 7)
    recent_cards = fetch_recent_cards(peer_id, days=recent_days, top_n=5 if not slim else 3)
    weekly = fetch_latest_weekly_digest(peer_id, before=snapshot_at)
    monthly = fetch_latest_monthly_profile(peer_id, before=snapshot_at)
    quarterly = fetch_latest_quarterly_canon(peer_id, before=snapshot_at)

    # 3. KPI 시계열
    kpi_q = overrides.get("kpi_quarters", 8 if not slim else 4)
    kpi_series = fetch_kpi_timeseries(peer_id, quarters=kpi_q)

    # 4. Canonical facts (불변)
    facts = fetch_canonical_facts(peer_id) if not slim else []

    # 5. Analysis ledger (carry-over)
    top_n = overrides.get("ledger_top_n", 5 if not slim else 3)
    min_conf = overrides.get("ledger_min_confidence", 0.7)
    ledger = fetch_ledger_top_n(peer_id, top_n=top_n, min_confidence=min_conf)

    # 6. Coverage
    coverage = {
        "has_weekly": weekly is not None,
        "has_monthly": monthly is not None,
        "has_quarterly": quarterly is not None,
        "kpi_quarters_available": len(kpi_series),
        "ledger_entries_count": len(ledger),
        "cold_start": all(x is None for x in [weekly, monthly, quarterly]),
    }

    # 7. Slim transformations
    if slim:
        weekly = _slim_weekly(weekly)
        monthly = _slim_monthly(monthly)
        quarterly = _slim_quarterly(quarterly)

    pack = {
        "peer_id": peer_id,
        "snapshot_at_kst": snapshot_at.isoformat(),
        "pack_version": _latest_pack_version(peer_id),
        "pack_built_at_kst": now_kst().isoformat(),
        "recent_cards": recent_cards,
        "daily_snapshots_last7": [] if slim else fetch_daily_snapshots(peer_id, last=7),
        "weekly_digest": weekly,
        "monthly_profile": monthly,
        "quarterly_canon": quarterly,
        "kpi_timeseries": kpi_series,
        "canonical_facts": facts,
        "analysis_ledger_top5": ledger,
        "coverage": coverage,
        "cached": False,
    }
    _store_cache(pack)
    return pack
```

### 6.3 Cache key

```python
def _pack_cache_key(peer_id, snapshot_at, slim):
    last_dirty = fetch_last_dirty_at(peer_id)
    return f"pack:{peer_id}:{last_dirty.isoformat()}:slim={slim}"
```

In-process LRU (max 200 keys) — cache TTL 무관, dirty 갱신 시 자동 invalidate.

### 6.4 Cold start fallback

`coverage.cold_start=True` 면 분석 agent 가 *기존 retrieval 방식* 으로 fallback:

```python
# 분석 agent 의 input 처리
pack = context_pack_builder.assemble(peer_id=..., slim_mode=...)
if pack["coverage"]["cold_start"]:
    # 기존 retrieval (Qdrant 직접) 우선, pack 은 recent_cards 만 사용
    cards = await hybrid_search(query, peer_id=peer_id, top_n=10)
    context = format_cards_for_prompt(cards)
else:
    context = format_pack_for_prompt(pack)
```

분석 agent 의 prompt rendering 함수가 `pack.coverage` 보고 자동 분기.

### 6.5 Prompt audit — 02-prompt-design-checklist 17 요소

ContextPackBuilder 는 LLM 미사용 (산식 only) — agent 자체의 prompt audit 은 N/A. 다만 출력 schema 가 17 요소를 *consumer* (분석 agent prompt) 가 충족하도록 carry:

| # | 요소 | pack 의 carry 위치 |
|---|---|---|
| 2 | 추적 대상 기업 | peer_id + canonical_facts.parent_company |
| 5 | 분석 기간 | snapshot_at_kst + recent_cards.published_at_kst + kpi_timeseries.fiscal_period |
| 6 | 최신성 검증 | pack_built_at_kst + coverage.has_* + pack_version |
| 8 | 회사별 비교 기준 | kpi_timeseries enum (revenue/op_income/op_margin/captive/headcount/rd) |
| 11 | 정량 우선 | kpi_timeseries 의 float + quantitative_anchors carry |
| 12 | 공식 vs 추정 | source_marker prefix (kpi 는 [DART], digest 는 [card] 또는 [자체 추정]) |
| 17 | 반복 추적 구조 | analysis_ledger_top5 — 과거 결론 carry |

→ 분석 agent 가 pack 받으면 자동으로 5/6/8/11/12/17 강화. checklist §5 의 mapping 이 *시스템 레벨* 로 격상.

## 7. LLM 모델 + token 예산

- **LLM 미사용** — pure 산식
- 비용: ₩0
- Latency: assemble ~50ms (PG read 7~8 query, 병렬 시 ~20ms), daily_snapshot 전체 batch ~10초 (10 peer)

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| daily cron — peer 의 그날 카드 0건 | snapshot row INSERT 안 함 (NULL row 회피) |
| assemble — PG query 1개 timeout | 해당 layer만 None, 다른 layer 정상 채움 (coverage 에 반영) |
| assemble — 모든 layer 결측 (cold start) | coverage.cold_start=True, 분석 agent fallback path |
| kpi_timeseries — DART 데이터 결측 | 결측 quarter 는 None, source_marker="[결측]" |
| canonical_facts 결측 | facts=[], 분석 agent 는 peer_id 만으로 추론 |
| ledger 결측 (cold start) | ledger=[], analysis_ledger_top5=[] |

## 9. 외부 의존성

- **DB (PG)**:
  - READ: `card_news`, `peer_financials`, `peer_weekly_digest`, `peer_monthly_profile`, `peer_quarterly_canon`, `peer_canonical_facts`, `peer_daily_snapshot`, `peer_dirty_flags`, `analysis_ledger`, `enrichment_cache` (keywords)
  - WRITE: `peer_daily_snapshot` (UPSERT), `peer_dirty_flags` (UPDATE — urgent/dirty mark)
- **외부 API**: 없음
- **In-process cache**: LRU (max 200, no TTL — dirty-based invalidation)

### Flyway V19 schema

```sql
-- V19
CREATE TABLE peer_daily_snapshot (
    peer_id        VARCHAR(50) NOT NULL,
    snapshot_date  DATE        NOT NULL,
    payload        JSONB       NOT NULL,
    card_count     INT         NOT NULL,
    avg_exposure_score REAL    NOT NULL,
    kpi_delta_hash VARCHAR(64),
    created_at     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (peer_id, snapshot_date)
);
CREATE INDEX idx_dsnapshot_peer_recent ON peer_daily_snapshot (peer_id, snapshot_date DESC);

CREATE TABLE peer_dirty_flags (
    peer_id           VARCHAR(50) PRIMARY KEY,
    urgent            BOOLEAN     NOT NULL DEFAULT FALSE,
    cards_changed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_weekly_at    TIMESTAMPTZ,
    last_monthly_at   TIMESTAMPTZ,
    last_quarterly_at TIMESTAMPTZ
);
```

## 10. State 흐름

- daily_snapshot cron: KnowledgeCurationState 의 `daily_snapshots` 필드 채움
- assemble: in-process — analysis agent 의 input dict 에 직접 박힘 (state 거치지 않음)

## 11. Provenance + Confidence

- **Provenance**: LLM 미사용이라 provenance 단순 — `pack.pack_built_at_kst` + `pack.pack_version` 만
- **Confidence**: pack 자체에는 confidence 없음. consumer (분석 agent) 가 `coverage` 와 component confidence (weekly.confidence 등) 를 가중 합쳐 최종 confidence 산출

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit (daily) | 그날 카드 15건 + urgent 1건 | top5_card_ids=5, urgent flag=True |
| Unit (assemble) | weekly/monthly 있음, quarterly 결측 | coverage.has_quarterly=False, narrative omit |
| Unit (assemble cold) | 모든 layer 결측 | coverage.cold_start=True, recent_cards 만 채워짐 |
| Unit (slim) | slim_mode=True | total token ≤ 5,000 |
| Unit (cache) | 동일 peer + 동일 snapshot_at 2회 | 2번째 cached=True, latency ~5ms |
| Unit (cache invalidate) | dirty flag 갱신 후 호출 | cached=False, 새로 fetch |
| Edge | KPI 8 quarter 결측 (cold peer) | kpi_quarters_available=0, source_marker=[결측] |
| Integration | 분석 agent (Insight) 가 assemble 호출 후 LLM | LLM input token ~17K (raw 10K → pack 12K + cards 5K) |

## 13. 모니터링

- **pipeline_logs.step**: `daily_snapshot` (cron) / `assemble_pack` (in-process)
- **KPI**:
  - daily snapshot 호출 성공률 ≥ 99% (산식이라 거의 안 깨짐)
  - assemble latency p95 ≤ 100ms (cache miss), p95 ≤ 10ms (cache hit)
  - cache hit rate ≥ 60% (분석 호출 빈도 vs dirty 갱신 빈도 비)
  - coverage.cold_start 비율 ≤ 30% (K2~K5 도입 진척도 indicator)
  - peer_dirty_flags 의 last_*_at 이 cron 주기 보다 늦으면 alert
- **Token budget**: ₩0 (LLM 미사용)

## 14. 구현 메모 + Changelog

### 핵심 파일

- `src/agents/context_pack_builder.py` (신규 — `daily_snapshot()` + `assemble()` 2 entry point)
- `src/db/peer_daily_snapshot.py` (READ + UPSERT)
- `src/db/peer_dirty_flags.py` (READ + UPDATE)
- 분석 agent (`mixer_analysis_agent.py` 등) 의 `_build_context()` 가 본 클래스 호출

### 사용 패턴 (분석 agent 입장)

```python
from src.agents.context_pack_builder import ContextPackBuilder

class InsightCascadeAgent:
    def __init__(self):
        self.pack_builder = ContextPackBuilder()

    async def generate(self, card_ids: list[str]):
        cards = fetch_cards(card_ids)
        peer_ids = {c.peer_id for c in cards}

        # peer 별 pack 조립
        slim = len(peer_ids) > 3
        packs = {
            pid: self.pack_builder.assemble(peer_id=pid, snapshot_at_kst=now_kst().isoformat(), slim_mode=slim)
            for pid in peer_ids
        }

        prompt = render_prompt(cards=cards, packs=packs, slim=slim)
        result = await llm_call(prompt)
        AnalysisLedger.insert(result)   # 다음 분석에 carry-over
        return result
```

### Changelog

- **v1 (2026-05-14)** — 신설. daily_snapshot + assemble 2-mode. slim_mode 도입 (Mixer 다중 peer 대응).
