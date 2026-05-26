# DerivedMetricsService — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `DerivedMetricsService` (4-mode: TopInsight / Trend / MonitoringOverview / PeerOverview 통합) |
| **Supervisor** | Enrichment |
| **상태** | 🟡 부분 (산식 only — P6) |
| **Trigger** | Ingestion 후 매시 후속 + nightly 04:00 |

## 2. 책임

**한 줄**: card_news 집계 산식으로 Home dashboard + Monitoring overview + 트렌드 메트릭을 한 번에 산출.

**구체적 (4-mode)**:

1. **Top Insight Selection** — 오늘 카드 중 Top 5 (exposure_score desc + recency boost)
2. **Trend Aggregation** — 전주 대비 증감 %, 새 카드 수, 핫 키워드
3. **Monitoring Overview** — 4 peer × sector × event_type 매트릭스 + 비교 표
4. **Peer Overview** *(PDF 2026-05-14 §3 추가 — Peer+ Overview)* — 화면 설명 + 기준 시점 + 4 peer 통합 비교 view (KPI snapshot + 최근 카드 요약 + DerivedMetrics 의 sector mix). frontend `PeerOverviewPage` 의 단일 데이터 소스.

## 3. 책임 NOT

- 키워드 추출 — KeywordExtractionService
- LLM 분석 — Analysis supervisor
- 시계열 차트 데이터 (예: 일별 카드 수) — 본 service 가 7일 daily count 만 생성 (실 차트는 frontend 가 렌더)

## 4. 입력 스펙

```python
class DerivedMetricsInput(TypedDict):
    mode: Literal["top_insight", "trend", "monitoring_overview", "peer_overview", "all"]
    peer_id: str | None
    window_days: int | None    # trend/monitoring 용

    # PDF 2026-05-14 §3, checklist 5/6 — 기준 시점 명시화 (peer_overview 필수)
    analysis_period: dict | None
    # {
    #   "since": "2026-04-15",      # KST 절대 날짜
    #   "until": "2026-05-14",
    #   "label": "최근 30일 (2026-04-15 ~ 2026-05-14, KST)",
    #   "fiscal_anchor": "2026-1Q", # DART 기준 분기 (peer_overview 의 재무 KPI 정합용)
    # }
```

## 5. 출력 스펙

```python
class TopInsightOutput(TypedDict):
    top_insight_id: str            # 최상위 카드 id (Home 카루셀)
    top5_card_ids: list[str]

class TrendOutput(TypedDict):
    today_card_count: int
    delta_pct_vs_last_week: float
    hot_keywords: list[dict]       # 전주 대비 frequency 급증 키워드 top 5
    daily_counts: list[dict]       # 최근 7일 일별 카드 수

class MonitoringOverviewOutput(TypedDict):
    peers: list[dict]              # [{peer_id, card_count, top_sector, avg_exposure_score}]
    comparison_matrix: list[dict]  # cross-peer × sector × event_type
    last_updated: datetime

class PeerKPISnapshot(TypedDict):
    """PDF §3 — peer_overview 의 통합 KPI. checklist 8 (회사별 비교 기준) 적용."""
    peer_id: str
    revenue_krw_bn: float           # 매출 (분기 — fiscal_anchor 기준)
    operating_profit_krw_bn: float  # 영업이익
    operating_margin_pct: float     # 영업이익률
    captive_ratio_pct: float | None # Captive 비중 (그룹사 매출 / 전체)
    headcount: int | None
    rd_ratio_pct: float | None      # R&D 매출 비중
    source_marker: str              # "[DART 2026-1Q]" / "[자체 추정 v3]" — checklist 12

class PeerOverviewItem(TypedDict):
    peer_id: str
    kpi: PeerKPISnapshot
    recent_card_count: int                # analysis_period 내 카드 수
    top_sectors: list[str]                # sector 분포 top 2
    top_events: list[str]                 # event_type 분포 top 2
    one_line_status: str                  # "최근 30일 partnership 3건 + DART 매출 QoQ +5%" (≤ 80자)

class PeerOverviewOutput(TypedDict):
    """PDF 2026-05-14 §3 — Peer+ Overview. frontend PeerOverviewPage 단일 소스.
    "화면 설명 + 기준 시점 + 통합 비교 view" 의 직접 응답."""
    screen_description: str               # 화면 상단 안내 (≤ 200자)
    analysis_period: dict                 # input 의 analysis_period echo (UI 노출용)
    peers: list[PeerOverviewItem]         # 4 peer 정렬: revenue desc
    comparison_axes: list[str]            # ["매출", "영업이익률", "Captive 비중", "headcount", "최근 동향 건수"]
    last_updated: datetime

class DerivedMetricsOutput(TypedDict):
    top_insight: TopInsightOutput | None
    trend: TrendOutput | None
    monitoring_overview: MonitoringOverviewOutput | None
    peer_overview: PeerOverviewOutput | None
```

frontend endpoints:
- `GET /api/dashboard/summary` (top_insight + trend 합쳐서)
- `GET /api/monitoring`, `/monitoring/overview`, `/monitoring/comparison` (monitoring_overview)
- `GET /api/peer/overview` *(신규 — PDF §3)* — peer_overview 단독

## 6. 알고리즘

### 6.1 Top Insight Selection (산식)

```python
def top_insight(today_cards):
    # recency boost — 6시간 이내 카드 가중치 ×1.2
    now = datetime.utcnow()
    def score(card):
        recency_boost = 1.2 if (now - card.created_at).total_seconds() < 6*3600 else 1.0
        return card.exposure_score * recency_boost

    sorted_cards = sorted(today_cards, key=score, reverse=True)
    return {
        "top_insight_id": sorted_cards[0].id if sorted_cards else None,
        "top5_card_ids": [c.id for c in sorted_cards[:5]],
    }
```

### 6.2 Trend Aggregation (산식)

```python
def trend(window_days=7):
    today = date.today()
    last_week_start = today - timedelta(days=14)
    this_week_start = today - timedelta(days=7)

    cards_this = fetch_cards(since=this_week_start, until=today)
    cards_last = fetch_cards(since=last_week_start, until=this_week_start)

    delta_pct = (len(cards_this) - len(cards_last)) / max(len(cards_last), 1) * 100

    # hot keywords — frequency 급증
    keywords_this = KeywordExtractionService().extract(window_days=7)
    keywords_last = KeywordExtractionService().extract(window_days=14)  # caching needed
    # ... compute delta per keyword

    daily = [{
        "date": (today - timedelta(days=i)).isoformat(),
        "card_count": fetch_card_count(date=today - timedelta(days=i)),
    } for i in range(window_days, 0, -1)]

    return {
        "today_card_count": fetch_card_count(date=today),
        "delta_pct_vs_last_week": delta_pct,
        "hot_keywords": top_hot_keywords[:5],
        "daily_counts": daily,
    }
```

### 6.3 Monitoring Overview (산식)

```python
def monitoring_overview():
    peers = []
    for peer_id in MONITORED_PEERS:
        cards = fetch_peer_cards(peer_id, since=last_30_days)
        peers.append({
            "peer_id": peer_id,
            "card_count": len(cards),
            "top_sector": Counter([c.sector for c in cards]).most_common(1)[0][0],
            "avg_exposure_score": np.mean([c.exposure_score for c in cards]),
        })

    # cross matrix: peer × sector × event_type → count
    matrix = []
    for peer in MONITORED_PEERS:
        for sector in SECTORS:
            for evt in EVENT_TYPES:
                count = count_cards(peer_id=peer, sector=sector, event_type=evt, since=30d)
                if count > 0:
                    matrix.append({"peer_id": peer, "sector": sector, "event_type": evt, "count": count})

    return {
        "peers": peers,
        "comparison_matrix": matrix,
        "last_updated": datetime.utcnow(),
    }
```

### 6.4 Peer Overview (산식 — PDF 2026-05-14 §3 신규)

```python
def peer_overview(analysis_period):
    """4 peer 통합 KPI snapshot + 최근 동향 합본.
    LLM 미사용 — DART (peer_financials) + card_news (집계) 조합.
    """
    fiscal_anchor = analysis_period["fiscal_anchor"]   # 예: "2026-1Q"
    since = analysis_period["since"]
    until = analysis_period["until"]

    peers = []
    for peer_id in MONITORED_PEERS:  # 4 peer
        # KPI — DART peer_financials (source_marker prefix 강제)
        fin = fetch_peer_financials(peer_id, fiscal=fiscal_anchor)
        kpi = {
            "peer_id": peer_id,
            "revenue_krw_bn": fin.revenue / 1e9,
            "operating_profit_krw_bn": fin.operating_profit / 1e9,
            "operating_margin_pct": fin.operating_margin_pct,
            "captive_ratio_pct": fin.captive_ratio_pct,    # 추정치면 source_marker=[자체 추정]
            "headcount": fin.headcount,
            "rd_ratio_pct": fin.rd_ratio_pct,
            "source_marker": fin.source_marker,
        }
        # 최근 동향 (analysis_period 내)
        cards = fetch_peer_cards(peer_id, since=since, until=until)
        top_sectors = [s for s, _ in Counter(c.sector for c in cards).most_common(2)]
        top_events = [e for e, _ in Counter(c.event_type for c in cards).most_common(2)]

        # one_line_status — 산식 기반 (LLM X)
        top_evt = top_events[0] if top_events else "관망"
        delta = revenue_delta_qoq(peer_id, fiscal_anchor)  # ±x.x%
        one_line = f"최근 {len(cards)}건 (주 {top_evt}) + 매출 QoQ {delta:+.1f}%"

        peers.append({"peer_id": peer_id, "kpi": kpi,
                      "recent_card_count": len(cards),
                      "top_sectors": top_sectors, "top_events": top_events,
                      "one_line_status": one_line[:80]})

    return {
        "screen_description": (
            f"4 peer 통합 비교 — {analysis_period['label']} 기준. "
            "DART 분기 재무 + 최근 동향 카드 집계. 추정치는 [자체 추정] 표시."
        ),
        "analysis_period": analysis_period,
        "peers": sorted(peers, key=lambda p: p["kpi"]["revenue_krw_bn"], reverse=True),
        "comparison_axes": ["매출", "영업이익률", "Captive 비중", "headcount", "최근 동향 건수"],
        "last_updated": datetime.utcnow(),
    }
```

> **checklist 매핑** — peer_overview 는 LLM 미사용이지만 출력 schema 가 17 요소를 직접 충족:
> 2 (추적 대상 4 peer) / 5 (analysis_period) / 6 (KST + fiscal_anchor) / 8 (KPI enum) /
> 11 (정량 수치) / 12 (source_marker prefix) / 14 (TypedDict).

## 7. LLM 모델 + token 예산

- **LLM 미사용** — 산식 only
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| 오늘 card 0건 | top_insight_id=None |
| 전주 데이터 0건 (cold start) | delta_pct=0.0 |
| DB query 실패 | retry 1회 → cache 결과 (어제 값) 리턴 |

## 9. 외부 의존성

- **DB**: `card_news` (READ + 집계), `enrichment_cache` (UPSERT)
- **sub**: KeywordExtractionService (hot keywords)

## 10. State 흐름

EnrichmentState 의 `top_insights`, `trend_metrics`, `monitoring_overview`, `peer_overview` 필드.

## 11. Provenance + Confidence

- **Provenance**: cache.metadata.computed_at
- **Confidence**: deterministic, 1.0

## 12. 테스트 시나리오

| Unit | 오늘 5 card | top_insight 가 exposure 최고 카드 |
| Unit | 6시간 내 신규 + 오래된 카드 같은 exposure | 신규가 우선 (recency boost) |
| Unit | 전주 100 / 이번주 130 | delta_pct = +30% |
| Unit | peer_overview, fiscal_anchor=2026-1Q | 4 peer revenue desc 정렬 + source_marker 검증 |
| Unit | peer_overview, peer_financials 결측 | source_marker=`[추정 없음]`, kpi 필드 None |
| Edge | DB 0 row | top_insight_id=None, delta=0% |

## 13. 모니터링

- KPI: 산출 latency ≤ 5초 (every 매시)
- token 예산: ₩0

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/derived_metrics_agent.py` (신규 P6)
- mode 별 method 분리 — `top_insight()`, `trend()`, `monitoring_overview()`, `peer_overview()` 또는 `compute(mode=...)` 단일 진입점

### Changelog

- **v1 (제안, P6)** — 3-mode 통합 산식 (TopInsight + Trend + Monitoring 묶음)
- **v2 (2026-05-14)** — PDF 사업전략팀 추가 질의 회신 §3 반영
  - 4번째 mode `peer_overview` 추가 (DART 분기 KPI + 최근 동향 합본)
  - 입력에 `analysis_period` (since/until/label/fiscal_anchor) 필드 — checklist 5/6
  - 출력 schema 에 `PeerKPISnapshot.source_marker` (DART vs 추정 구분) — checklist 12
  - 신규 frontend endpoint `GET /api/peer/overview` 명시
