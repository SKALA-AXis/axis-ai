# BriefingGenerationAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `BriefingGenerationAgent` (async, user-triggered) |
| **Supervisor** | Briefing (신규 sub-supervisor; 단일 agent 라 supervisor==agent) |
| **상태** | 🔴 신규 — backend `POST /api/briefings/generate` 가 fixture (202 Accepted 만). 실 generation 미구현. `/pipeline/delivery` 와는 별개 (daily 메일 vs user 요청 briefing 문서) |
| **Trigger** | user POST `/api/briefings/generate` → BE 가 axis-ai 호출 → 비동기 background task → poll `/api/briefings/{id}/status` |

## 2. 책임

**한 줄**: 사용자가 지정한 기간 / 카드 / peer / sector 의 card_news 들을 종합 분석하여 frontend Briefing 화면이 표시할 `BriefingReport` (executive_summary / immediate_trends / watch_trends / sections) 를 생성.

**구체적 (5-phase async)**:

1. **Card 선택** — 입력 필터 (period / peer_ids / sectors / card_ids) 로 card_news 검색. 비어있으면 status=`empty` fail.
2. **Trend 분류** — exposure_score + recency 로 카드 분류:
   - `immediate_trends` — exposure_band=high + recency ≤ 7일
   - `watch_trends` — exposure_band=medium 또는 recency ≤ 30일
3. **Section 조립** — peer × event_type 매트릭스로 그룹화 → 각 그룹 별 대표 카드 + LLM 시사점 synthesize
4. **Executive summary** — 전체 카드 종합 → 3~5 문단 (LLM gpt-4o)
5. **저장 + 상태 갱신** — `briefing_reports` 테이블 INSERT (status='completed') + 비용 / token 기록

## 3. 책임 NOT

- **Daily scheduled 이메일** — `BriefingService.generateAndSend()` (BE) + axis-ai `/pipeline/delivery` 의 책임. 본 agent 는 *user-triggered 화면용 briefing 문서*.
- **카드 자체 생성** — CardNewsAgent (1단계 카드뉴스 생성).
- **PDF/Word export** — 별도 export service (W7+, 미구현).
- **이메일 발송** — `BriefingService` (Java SES) — frontend 가 share link 제공.

## 4. 입력 스펙

```python
class BriefingGenerateInput(TypedDict):
    briefing_type: Literal["daily", "weekly", "custom"]
    date_from: str | None            # YYYY-MM-DD (briefing_type=custom 필수)
    date_to: str | None
    card_ids: list[str] | None       # 명시한 카드만
    peer_ids: list[str] | None       # ['samsung_sds', 'lg_cns']
    sectors: list[str] | None        # ['ax', 'security']
    title: str | None                # 사용자 지정 제목
    requested_by_user_id: int        # audit
```

## 5. 출력 스펙

```python
class TrendItem(TypedDict):
    title: str
    summary: str                     # 1~2 문장
    confidence: float
    card_count: int
    representative_card_id: str
    sk_ax_perspective: str           # SK AX 관점 한 문장

class ReasoningTrailItem(TypedDict):
    """02-prompt-design-checklist.md §4 Tier 1 — 사용자 default.
    Briefing 은 section 별 mini-trail (3 step) — section 안의 LLM 호출 1회당 trail 1세트."""
    seq: int
    label: str                       # ≤ 12자 ("카드 선정" / "공통 신호" / "SK AX 영향")
    one_liner: str                   # ≤ 80자
    evidence_refs: list[str]
    langfuse_observation_id: str | None

class SectionBlock(TypedDict):
    section_id: str                  # 'peer:samsung_sds' / 'event:partnership'
    title: str
    summary: str
    card_ids: list[str]
    implication: str
    reasoning_trail: list[ReasoningTrailItem]   # Tier 1 — section 별 3 step

class BriefingReport(TypedDict):
    id: str                          # 'BR-YYYYMMDD-NNN'
    title: str
    briefing_type: str
    date_from: str
    date_to: str
    executive_summary: str           # 3~5 문단
    executive_reasoning_trail: list[ReasoningTrailItem]   # Tier 1 — top-level 3~5 step
    immediate_trends: list[TrendItem]
    watch_trends: list[TrendItem]
    sections: list[SectionBlock]
    sources: list[dict]
    status: Literal["queued", "running", "completed", "failed"]
    progress: float
    error_message: str | None
    confidence: float
    langfuse_trace_id: str | None    # Tier 3 — admin deep link (briefing 전체 trace)
    provenance: dict                 # ProvenanceTrackerMiddleware
    created_at: str
    completed_at: str | None
```

신규 DB 테이블 `briefing_reports` (V12 migration):

```sql
CREATE TABLE briefing_reports (
    id VARCHAR(40) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    briefing_type VARCHAR(20) NOT NULL,
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    requested_by_user_id BIGINT,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    progress NUMERIC(3,2) DEFAULT 0.0,
    payload JSONB,                              -- BriefingReport 전체
    error_message TEXT,
    confidence NUMERIC(3,2),
    provenance JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX idx_briefing_status ON briefing_reports(status, created_at DESC);
CREATE INDEX idx_briefing_user ON briefing_reports(requested_by_user_id, created_at DESC);
```

관계 정규화 테이블 (V21):

```sql
briefing_reports.id
  ├── briefing_report_cards.briefing_report_id -> card_news.id
  ├── briefing_report_articles.briefing_report_id -> raw_articles.id
  ├── briefing_recipients.briefing_report_id -> recipients.id
  └── briefing_history.briefing_report_id

briefing_history.id
  └── briefing_history_cards.briefing_history_id -> card_news.id
```

## 6. 알고리즘

### 6.1 비동기 실행

```python
# axis-ai POST /briefings/generate (신규 endpoint)
@router.post("/briefings/generate", status_code=202)
async def generate_briefing(payload: BriefingGenerateInput, bg: BackgroundTasks):
    briefing_id = generate_briefing_id()
    save_briefing_row(briefing_id, status="queued", payload_meta=payload)
    bg.add_task(_run_briefing, briefing_id, payload)
    return {"briefing_id": briefing_id, "status": "queued"}

async def _run_briefing(briefing_id: str, payload: BriefingGenerateInput):
    try:
        update_status(briefing_id, "running", progress=0.05)
        cards = fetch_cards(payload)                              # phase 1
        if not cards:
            update_status(briefing_id, "failed", error="empty card set"); return

        update_status(briefing_id, "running", progress=0.20)
        immediate, watch = classify_trends(cards)                 # phase 2

        update_status(briefing_id, "running", progress=0.45)
        sections = assemble_sections(cards)                       # phase 3 (LLM)

        update_status(briefing_id, "running", progress=0.75)
        exec_summary = generate_executive_summary(cards, immediate, watch)  # phase 4 (LLM)

        update_status(briefing_id, "running", progress=0.95)
        report = build_report(briefing_id, payload, cards, immediate, watch, sections, exec_summary)
        save_briefing_payload(briefing_id, report, status="completed", progress=1.0)
    except Exception as e:
        log.exception("briefing %s failed", briefing_id)
        update_status(briefing_id, "failed", error=str(e)[:500])
```

### 6.2 Trend 분류 (산식)

```python
def classify_trends(cards):
    immediate, watch = [], []
    for c in cards:
        recency_days = (date.today() - c.published_at.date()).days
        if c.exposure_band == "high" and recency_days <= 7:
            immediate.append(c)
        elif c.exposure_band in ("high", "medium") and recency_days <= 30:
            watch.append(c)
    # 동일 cluster 카드 grouping
    immediate = group_by_cluster(immediate, max_items=5)
    watch     = group_by_cluster(watch, max_items=8)
    return immediate, watch
```

### 6.3 Section assembly (LLM gpt-4o-mini, ~10 section)

~~~text
# SK AX 브리핑 섹션 작성가

당신은 SK AX 사업전략팀의 브리핑 섹션 작성가입니다.
**{peer or sector} 의 최근 {N}건 카드** 의 핵심 흐름을 1~2 문단으로 정리합니다.

## 입력 데이터
- **섹션 대상**: {peer or sector}
- **카드 목록**: {card_summaries}

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **카드 trace**: implication 의 주장이 카드 본문에 trace 가능
- **SK AX 화자**: implication 은 `"SK AX 관점에서 어떤 의미인지"` 1~2 문장

### 일반 규칙 (17 요소 매핑)
1. **(#7 단순 요약 금지)** 카드 N건 나열 X → 핵심 흐름 + 패턴
2. **(#10 수익화 관점)** implication 에 SK AX 영향 (긍정/중립/부정) 함의
3. **(#13 SK AX 화자)** 절대 규칙 참조
4. **(#14 출력 형식)** strict JSON

## 3-Tier Observability 출력

### Tier 1 — reasoning_trail (사용자 default, 섹션 별)
정확히 **3 step** ("카드 선정" / "공통 신호" / "SK AX 영향"). label ≤ 12자, one_liner ≤ 80자.

## 출력 형식 (strict JSON)

```json
{
  "summary": "1~2 문단",
  "implication": "SK AX 관점에서 어떤 의미 1~2 문장",
  "representative_card_id": "CN-...",
  "reasoning_trail": [
    {"seq": 1, "label": "카드 선정", "one_liner": "...", "evidence_refs": ["CN-..."], "langfuse_observation_id": null}
  ]
}
```
~~~

### 6.4 Executive summary (LLM gpt-4o, 1회)

~~~text
# SK AX Executive Summary 작성가

당신은 SK AX 사업전략팀의 브리핑 executive summary 작성가입니다.
**{date_from} ~ {date_to} 기간** 의 peer 모니터링 결과를 3~5 문단 한국어로 정리합니다.

## 입력 데이터

### 집계
- **총 카드**: {N}
- **즉각 트렌드**: {M}
- **관찰 트렌드**: {K}
- **주요 peer**: {peer_list}

### 즉각 트렌드 요약
{immediate_summaries}

### 관찰 트렌드 요약
{watch_summaries}

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **카드 trace**: 모든 주장에 `[CN-...]` 카드 id 인용 강제
- **분량**: 3~5 문단 한국어

### 일반 규칙 (17 요소 매핑)
1. **(#5 분석 기간)** 첫 문단에 {date_from}~{date_to} 명시
2. **(#7 단순 요약 금지)** peer/sector 별 시사점 패턴
3. **(#10 수익화 관점)** 마지막 문단 권고에 SK AX 영향 함의
4. **(#13 SK AX 화자)** 모든 시사점이 SK AX 관점

### 문단 구조 (순서 고정)
1. **첫 문단**: 기간 개요 + 가장 중요한 변화
2. **두 번째 문단부터**: peer / sector 별 시사점
3. **마지막 문단**: SK AX 가 주목해야 할 권고 1~3 항

## 3-Tier Observability 출력

### Tier 1 — executive_reasoning_trail (사용자 default)
정확히 **3~5 step** ("기간 개요" / "주요 변화" / "peer 시사점" / "권고"). label ≤ 12자.

## 출력 형식 (strict JSON)

```json
{
  "executive_summary": "3~5 문단 (Markdown 가능)",
  "executive_reasoning_trail": [
    {"seq": 1, "label": "기간 개요", "one_liner": "...", "evidence_refs": ["CN-..."], "langfuse_observation_id": null}
  ]
}
```
~~~

### 6.5 Polling endpoint

```python
@router.get("/briefings/{briefing_id}/status")
async def get_status(briefing_id: str):
    row = fetch_briefing(briefing_id)
    return {
        "briefing_id": briefing_id,
        "status": row.status,
        "progress": row.progress,
        "error_message": row.error_message,
    }

@router.get("/briefings/{briefing_id}")
async def get_briefing(briefing_id: str):
    row = fetch_briefing(briefing_id)
    if row.status != "completed":
        raise HTTPException(409, f"not ready (status={row.status})")
    return row.payload
```

## 7. LLM 모델 + token 예산

| Phase | 모델 | 토큰/호출 | 호출 수/briefing | 비용/briefing |
|---|---|---|---|---|
| Section assembly | gpt-4o-mini | ~2,500 | ~10 | ~₩400 |
| Executive summary | gpt-4o | ~6,000 | 1 | ~₩200 |
| **합계** | | | | **~₩600/briefing** |

- 일일 expected briefing 수: ~3 (사용자 manual 요청 + 매주 1회 자동) → 일 ₩1,800
- 일일 expected: 가벼움. TokenBudgetMiddleware 의 ₩5,000 envelope 충분히 수용

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| Cards 0건 | status='failed', error_message="조건에 부합하는 카드 없음" |
| LLM section 한 개 실패 | 해당 section 만 stub (`summary='(생성 실패)'`), 다른 section 진행 |
| Executive summary 실패 | retry 1회 → 실패 시 immediate/watch 요약 list 만 + status='completed_partial' |
| BackgroundTasks 자체 fail (pod crash) | status='running' 인 row 가 30분 이상 머물면 cleanup cron 이 'failed' 전환 |
| Token budget 초과 | 진행 중 phase 까지 stub 으로 마무리 + status='completed_partial' |
| Duplicate request (같은 user, 같은 input, 1분 이내) | 동일 briefing_id 반환 (idempotency) |

## 9. 외부 의존성

- **DB**: card_news (READ), `briefing_reports` (V12 INSERT/UPDATE)
- **DB**: `briefing_report_cards`, `briefing_report_articles`, `briefing_recipients`, `briefing_history_cards` (V21 관계 정규화)
- **외부 API**: OpenAI gpt-4o + gpt-4o-mini
- **Middleware**: ProvenanceTracker, TokenBudget, AuditLog (`@audit("briefing_generate", "briefing_report")`)

## 10. State 흐름

LangGraph 가 아닌 *async function chain* (BackgroundTasks). state 없음. 진행 상황은 briefing_reports 테이블의 status / progress 가 SSOT.

## 11. Provenance + Confidence

- **Provenance**: `briefing_reports.provenance = { llm_model_section, llm_model_summary, prompt_versions, run_at, source_card_ids, git_sha, agent }` + V21 mapping tables
- **Confidence**: average(section.confidence) × executive_summary.confidence × (cards_used / cards_requested ratio)

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | classify_trends (5 high recent + 10 medium) | immediate=5, watch=10 |
| Unit | section assembly (peer=samsung_sds, 3 cards) | summary 비어있지 않음 + representative_card_id 유효 |
| Unit | executive summary (10 cards) | 3~5 문단 + 모든 카드 id 인용 |
| Integration | POST /briefings/generate → poll → GET 완료 | 5초 이내 progress > 0, ≤ 60초 내 completed |
| Edge | cards 0건 | failed + error_message |
| Edge | LLM section 1개 fail | completed_partial + 9 sections + 1 stub |
| Edge | 같은 user 30초 내 동일 요청 | idempotency: 같은 briefing_id |
| Edge | pod crash 중간 | cleanup cron 30분 후 failed 전환 |

## 13. 모니터링

- KPI:
  - 평균 생성 시간 ≤ 30초
  - 성공률 ≥ 95%
  - 평균 confidence ≥ 0.70
  - 일일 briefing 수 ≤ 10 (overload 방지)
- pipeline_logs.step: `briefing_generate`
- token: ~₩1,800/일 (3 briefing 기준)
- alert: 평균 latency > 60초 또는 fail 비율 > 10% → 이메일

## 14. 구현 메모 + Changelog

### 핵심 파일 (신규 P7+)

- `src/agents/briefing_generation_agent.py` (신규) — 5-phase orchestrator
- `src/api/router.py` 의 `/briefings/generate`, `/briefings/{id}/status`, `/briefings/{id}` 신규 endpoint
- `src/db/briefing_reports.py` (신규) — DAO

### 신규 마이그레이션

- **V12** — `briefing_reports` 테이블 (위 §5 schema)
- **V21** — `briefing_report_cards`, `briefing_report_articles`, `briefing_recipients`, `briefing_history_cards`, `briefing_history.briefing_report_id`

### Backend 연동

- `BriefingController.generateBriefing()` — fixture 제거 → `AiClientService.generateBriefing(payload, user_id)` → axis-ai `POST /briefings/generate`
- `BriefingController.getBriefingGenerationStatus()` — fixture 제거 → `AiClientService.getBriefingStatus(id)` → axis-ai `GET /briefings/{id}/status`
- `BriefingController.getBriefingById()` — fixture 제거 → DB SELECT briefing_reports OR axis-ai `GET /briefings/{id}`

### vs `/pipeline/delivery` 차이점

| 항목 | `/pipeline/delivery` (기존) | `/briefings/generate` (신규) |
|---|---|---|
| Trigger | Spring @Scheduled 08:30 KST | user POST 요청 |
| 동기/비동기 | 동기 (Spring 가 결과 받아 SES 발송) | 비동기 (background task + polling) |
| 출력 | `{subject, html, text, recipients}` | `BriefingReport` (executive_summary, trends, sections) |
| 저장 | briefing_history (이메일 발송 이력) | briefing_reports (생성된 문서 본체) |
| 소비자 | SES Email | Frontend Briefing View |

### Changelog

- **v1 (제안, P7+)** — 5-phase async 신설 + V12 + endpoint 3종
- **v1.1 (2026-05-15)** — V21 관계 정규화 반영: briefing report ↔ card/article/recipient, daily history ↔ card mapping
