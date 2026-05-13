# DerivedMetricsAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `DerivedMetricsAgent` (3-mode: TopInsight / Trend / MonitoringOverview 통합) |
| **Supervisor** | Enrichment |
| **상태** | 🟡 부분 (산식 only — P6) |
| **Trigger** | Ingestion 후 매시 후속 + nightly 04:00 |

## 2. 책임

**한 줄**: card_news 집계 산식으로 Home dashboard + Monitoring overview + 트렌드 메트릭을 한 번에 산출.

**구체적 (3-mode)**:

1. **Top Insight Selection** — 오늘 카드 중 Top 5 (exposure_score desc + recency boost)
2. **Trend Aggregation** — 전주 대비 증감 %, 새 카드 수, 핫 키워드
3. **Monitoring Overview** — 4 peer × sector × event_type 매트릭스 + 비교 표

## 3. 책임 NOT

- 키워드 추출 — KeywordExtractionAgent
- LLM 분석 — Analysis supervisor
- 시계열 차트 데이터 (예: 일별 카드 수) — 본 agent 가 7일 daily count 만 생성 (실 차트는 frontend 가 렌더)

## 4. 입력 스펙

```python
class DerivedMetricsInput(TypedDict):
    mode: Literal["top_insight", "trend", "monitoring_overview", "all"]
    peer_id: str | None
    window_days: int | None    # trend/monitoring 용
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

class DerivedMetricsOutput(TypedDict):
    top_insight: TopInsightOutput | None
    trend: TrendOutput | None
    monitoring_overview: MonitoringOverviewOutput | None
```

frontend endpoints:
- `GET /api/dashboard/summary` (top_insight + trend 합쳐서)
- `GET /api/monitoring`, `/monitoring/overview`, `/monitoring/comparison` (monitoring_overview)

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
    keywords_this = KeywordExtractionAgent().extract(window_days=7)
    keywords_last = KeywordExtractionAgent().extract(window_days=14)  # caching needed
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
- **sub**: KeywordExtractionAgent (hot keywords)

## 10. State 흐름

EnrichmentState 의 `top_insights`, `trend_metrics`, `monitoring_overview` 필드.

## 11. Provenance + Confidence

- **Provenance**: cache.metadata.computed_at
- **Confidence**: deterministic, 1.0

## 12. 테스트 시나리오

| Unit | 오늘 5 card | top_insight 가 exposure 최고 카드 |
| Unit | 6시간 내 신규 + 오래된 카드 같은 exposure | 신규가 우선 (recency boost) |
| Unit | 전주 100 / 이번주 130 | delta_pct = +30% |
| Edge | DB 0 row | top_insight_id=None, delta=0% |

## 13. 모니터링

- KPI: 산출 latency ≤ 5초 (every 매시)
- token 예산: ₩0

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/derived_metrics_agent.py` (신규 P6)
- mode 별 method 분리 — `top_insight()`, `trend()`, `monitoring_overview()` 또는 `compute(mode=...)` 단일 진입점

### Changelog

- **v1 (제안, P6)** — 3-mode 통합 산식 (TopInsight + Trend + Monitoring 묶음)
