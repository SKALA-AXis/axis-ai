# TokenBudgetMiddleware — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `TokenBudgetMiddleware` (cross-cutting decorator + budget ledger) |
| **Supervisor** | (none — 모든 LLM 호출 wrapper) |
| **상태** | 🟡 부분 — `pipeline_logs.llm_tokens_used` 컬럼 존재 (`src/db/article_store.py:393-400` 의 `_INSERT_PIPELINE_LOG`). 그러나 budget enforcement / aggregation 없음. /api/admin/usage 는 fixture |
| **Trigger** | 모든 LLM 호출 직전 (pre-check) + 직후 (record) |

## 2. 책임

**한 줄**: 모든 LLM 호출의 token 사용량을 측정·집계하고 일/주/월 예산 envelope 을 enforce (soft warn / hard fail) 하여 cost runaway 차단. *Prompt / response 본문 trace 는 Langfuse 가 담당* — 본 middleware 는 빌링 / 쿼터 / 차단 책임만.

**구체적**:

1. **Pre-call estimate** — tiktoken 으로 input token 수 추정 → budget 잔여 확인
2. **Soft warn** — daily 80% 사용 → 로그 + 이메일 1회 알림 (운영자)
3. **Hard fail** — daily 100% 초과 → LLM 호출 차단 + agent 가 graceful degradation (stub 응답 또는 cache hit)
4. **Override** — admin `/api/admin/usage/limits` 로 envelope 상향 가능
5. **Record (slim)** — 호출 후 actual usage 를 `usage_logs` 테이블 (token 수 + ₩ + `langfuse_trace_id` pointer) + `pipeline_logs.llm_tokens_used` 에 기록. **prompt/response 본문은 저장 X → Langfuse drill-down**
6. **Report** — `/api/admin/usage` 가 일/주/월 집계 + 비용 (₩) 환산 + per-user / per-agent 집계 제공
7. **(NOT) trace 시각화** — Langfuse UI 가 담당 (spec: `axis-infra/docs/OBSERVABILITY_LANGFUSE.md`)

## 3. 책임 NOT

- 외부 API rate-limit (OpenAI 429) — `openai` SDK 의 retry/backoff 가 담당
- Embedding token (BGE-M3) — 로컬 GPU/CPU, 비용 없음 → 추적 대상 외
- Reranker (BGE-reranker) — 로컬 모델 → 추적 대상 외

## 4. 입력 스펙

```python
class BudgetCheckInput(TypedDict):
    agent: str                       # "CardNewsComposer.generate"
    model: str                       # "gpt-4o" | "gpt-4o-mini"
    estimated_input_tokens: int
    max_output_tokens: int           # 호출자가 요청한 max
```

```python
class BudgetRecordInput(TypedDict):
    agent: str
    model: str
    input_tokens: int                # response.usage.prompt_tokens
    output_tokens: int               # response.usage.completion_tokens
    elapsed_ms: int
    success: bool
```

## 5. 출력 스펙

```python
class BudgetCheckResult(TypedDict):
    allowed: bool
    remaining_today: int             # 잔여 token (해당 model)
    warning_band: Literal["ok", "warn80", "warn95", "exceeded"]
    estimated_cost_krw: float        # 이 호출의 예상 비용
```

신규 DB 테이블 `usage_logs` (V13 migration) — **slim 형태**. Prompt / response 본문은 Langfuse (self-host) 가 보존, 본 테이블은 billing 집계 + per-user quota 만:

```sql
CREATE TABLE usage_logs (
    id BIGSERIAL PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    occurred_date DATE NOT NULL,                -- 일자 집계 인덱스
    user_id BIGINT,                              -- chat / briefing 의 호출자 (nullable: scheduled)
    actor_kind VARCHAR(20) NOT NULL,            -- user / scheduler / admin
    agent VARCHAR(120) NOT NULL,                -- "CardNewsComposer.generate"
    model VARCHAR(60) NOT NULL,
    input_tokens INT NOT NULL,
    output_tokens INT NOT NULL,
    elapsed_ms INT,
    cost_krw NUMERIC(12,4) NOT NULL,
    success BOOLEAN NOT NULL,
    error_class VARCHAR(80),                    -- "RateLimitError" 등 (전체 메시지는 Langfuse)
    langfuse_trace_id VARCHAR(80),              -- Langfuse 의 root trace 링크 (drill-down)
    langfuse_observation_id VARCHAR(80),        -- 해당 LLM 호출의 span id
    pipeline_log_id BIGINT REFERENCES pipeline_logs(id),
    metadata JSONB                              -- {prompt_version, request_id}
);
CREATE INDEX idx_usage_logs_date  ON usage_logs(occurred_date, model);
CREATE INDEX idx_usage_logs_agent ON usage_logs(agent, occurred_date);
CREATE INDEX idx_usage_logs_user  ON usage_logs(user_id, occurred_date) WHERE user_id IS NOT NULL;
```

본문 (prompt / response) 은 본 테이블 에 저장 안 함 — Langfuse 가 SoT (30일 보존). `langfuse_trace_id` 로 admin UI 가 클릭 한 번에 trace tree 진입.

`/api/admin/usage` 응답 (변경 없음):

```python
class UsageStats(TypedDict):
    period: dict                     # {date_from, date_to}
    token_usage: dict                # {gpt-4o: {input, output}, gpt-4o-mini: {input, output}}
    cost_krw: float
    api_calls: int
    budget_used_pct: float           # daily 기준
    by_agent: list[dict]             # top 10 agent 별 break-down
```

## 6. 알고리즘

### 6.1 비용 환산 (2026-05 기준 ₩ 환율 1,350/$)

```python
PRICE_USD_PER_1M = {
    # OpenAI 2026 가격 (변경 시 update)
    "gpt-4o":        {"in": 2.50, "out": 10.00},
    "gpt-4o-mini":   {"in": 0.15, "out":  0.60},
}
KRW_PER_USD = 1350

def calc_cost_krw(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICE_USD_PER_1M[model]
    usd = (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000
    return round(usd * KRW_PER_USD, 4)
```

### 6.2 일일 budget 기본값

```python
DEFAULT_DAILY_BUDGET_KRW = 5000.0     # ₩5,000/일 (~$3.7)
# CLAUDE.md "LLM 비용 예측" 항목의 ~$2.15/일 + 50% 버퍼

# Per-model envelope (optional override 가능)
DEFAULT_MODEL_BUDGET_KRW = {
    "gpt-4o":      3500.0,             # 70% (Answer, IssueCard, NewsAnalysis)
    "gpt-4o-mini": 1500.0,             # 30% (Classification, IntentRouter, etc.)
}
```

### 6.3 Pre-check + record

```python
def with_budget(agent: str, model: str):
    def deco(func):
        @wraps(func)
        async def wrapper(*args, prompt: str = "", max_tokens: int = 1000, **kwargs):
            est_in = num_tokens(prompt, model)
            check = await budget_check(agent, model, est_in, max_tokens)
            if not check["allowed"]:
                log.warning("budget exceeded | agent=%s model=%s", agent, model)
                # graceful degradation
                return _fallback_stub(agent, reason="budget_exceeded")
            if check["warning_band"] == "warn80":
                _notify_warn80_once_per_day(agent, model)

            t0 = time.time()
            try:
                response = await func(*args, prompt=prompt, max_tokens=max_tokens, **kwargs)
                actual_in = response.usage.prompt_tokens
                actual_out = response.usage.completion_tokens
                await budget_record(agent, model, actual_in, actual_out,
                                    int((time.time()-t0)*1000), success=True)
                return response
            except Exception as e:
                await budget_record(agent, model, est_in, 0,
                                    int((time.time()-t0)*1000), success=False)
                raise
        return wrapper
    return deco
```

### 6.4 Aggregation query (admin /api/admin/usage)

```sql
SELECT
    model,
    SUM(input_tokens)  AS input_tokens,
    SUM(output_tokens) AS output_tokens,
    SUM(cost_krw)      AS cost_krw,
    COUNT(*)           AS api_calls
FROM usage_logs
WHERE occurred_date BETWEEN :from AND :to
  AND success = true
GROUP BY model;

-- top 10 agent
SELECT agent, SUM(cost_krw) AS cost_krw
FROM usage_logs
WHERE occurred_date BETWEEN :from AND :to
GROUP BY agent
ORDER BY cost_krw DESC LIMIT 10;
```

### 6.5 Sync vs Async write (admin_page §3 호환)

**현재 결정 (v1)**: `usage_logs` INSERT 는 **sync** — agent 의 LLM 호출 직후 같은 Python task 내에서 PG 적재. 일 호출 ~50건 수준에선 latency 영향 ≤ 20ms.

**P9+ 확장 트리거**: 일 호출 > 500 또는 burst > 50 calls/min 발생 시 → Redis Stream 도입:

```python
# P9+ 안 (현재 미적용)
async def budget_record_async(payload):
    await redis.xadd("axis:usage_logs", payload)  # non-blocking
# Worker (별도 pod) 가 stream 소비 → batch INSERT
```

admin_page §3 의 "동기 INSERT 금지" 원칙은 burst 발생 가능한 production scale 의 권장사항. 9주 일정 + 학생 프로젝트 규모에선 sync 가 단순 + 운영 부담 작음 → v1 = sync 채택.

## 7. LLM 모델 + token 예산

- 본 middleware 는 *budget enforcer* 이지 token 소비자 아님 → ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| tiktoken estimate 실패 | conservative 2× 추정 (보수적 차단) |
| usage_logs INSERT 실패 | 호출은 정상 계속, log 에만 기록 (cost runaway 안 함) |
| OpenAI response.usage 누락 | est_in 으로 기록 + `metadata.token_source='estimated'` 마크 |
| budget 초과 + admin 미응답 | 새 호출 모두 stub 반환 + Grafana alert |
| 환율 변경 | `KRW_PER_USD` 상수 PR 로 update |

## 9. 외부 의존성

- **lib**: `tiktoken>=0.7`, `langfuse>=2.50` (trace_id 획득용)
- **DB**: `usage_logs` (V13 신규), `pipeline_logs` (이미 존재, `llm_tokens_used` 컬럼)
- **외부 API**: 없음 (OpenAI response 의 usage 만 활용)
- **연동 middleware**: Langfuse CallbackHandler (spec: `axis-infra/docs/OBSERVABILITY_LANGFUSE.md`) — 호출 후 `handler.get_trace_id()` 로 `usage_logs.langfuse_trace_id` 채움

## 10. State 흐름

state 에는 영향 없음 — middleware 는 LLM client wrapper. Agent 의 결과 자체는 그대로 전달.

다만 graceful degradation 시 `state["errors"].append("budget_exceeded:{agent}")` 추가.

## 11. Provenance + Confidence

- 본 middleware 의 활동은 `usage_logs` 에만 기록.
- Pipeline_logs.llm_tokens_used 가 agent 별 합산 표시 (audit 가능).
- 호출 단위 정확한 token 추적은 usage_logs row id 를 evidence_chain.provenance.usage_log_ids 에 nested 저장 (optional).

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | `calc_cost_krw('gpt-4o', 1000, 500)` | (1000·2.50 + 500·10.00)/1e6 · 1350 = ₩10.13 |
| Unit | `num_tokens` (tiktoken) | 한글 200자 ≈ ~150 tokens 추정 |
| Unit | budget_check (잔여 1500, 요청 추정 1800) | allowed=false |
| Unit | budget_check (잔여 1000, 요청 200) | allowed=true, warning=warn80 |
| Integration | 100 호출 → 일일 합계 < 예산 | usage_logs aggregation 정확 |
| Edge | OpenAI response.usage 누락 | est_in 으로 기록 + metadata 마킹 |
| Edge | budget 초과 시 graceful stub | agent 가 fallback 응답 반환 |

## 13. 모니터링

- KPI:
  - 일 평균 비용 ≤ ₩5,000
  - budget warn80 발생 횟수 (월 ≤ 5회 expected)
  - graceful_degradation 호출 횟수 (월 ≤ 1회, 1회 초과 시 envelope 재산정)
- Grafana panel:
  - 일별 cost_krw stacked by model
  - per-agent top 10
  - degradation count
- Alert: cost_krw[일] > ₩7,000 → 즉시 이메일

## 14. 구현 메모 + Changelog

### 핵심 파일 (신규 P9)

- `src/middleware/token_budget.py` — decorator + 산식 + DB I/O
- `src/db/usage_logs.py` — usage_logs DAO
- `axis-backend/.../UsageService.java` — `/api/admin/usage` 실제 집계 (현재 fixture)

### 신규 마이그레이션

- **V13** — `usage_logs` 테이블 (위 §5 schema)

### Backend 연동

- `AdminController.adminGetUsage()` → fixture 제거 → `UsageService.getUsageStats(from, to)` 실 구현
- `AdminController.adminSetUsageLimits()` → fixture 제거 → `BudgetConfigService.updateLimits()` 실 구현

### Changelog

- **v1 (2026-04-W3)** — pipeline_logs.llm_tokens_used 컬럼만 추가 (record only)
- **v2 (제안, P9)** — V13 usage_logs + budget enforcer + admin endpoint 활성
