# AnalysisLedger — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `AnalysisLedger` (write-through 미들웨어) |
| **Supervisor** | KnowledgeCuration |
| **상태** | 🆕 신규 (Phase K1 — 2026-05-14, **가장 우선 도입 권장**) |
| **Trigger** | 분석 agent (Mixer / Insight / PeerComparison / GlobalTrends / Briefing) 종료 직후 — `with_provenance` decorator 와 같은 layer |

## 2. 책임

**한 줄**: 분석 agent 가 결론을 도출하면 즉시 `analysis_ledger` 테이블에 INSERT — 다음 분석 호출 시 ContextPackBuilder 가 carry-over.

**구체적**:

1. 분석 종료 직후 `final_one_liner` + `confidence` + `source_card_ids` + `strategy_label` (있으면) 을 INSERT
2. 동일 peer + 동일 analysis_type 의 이전 entry 가 같은 conclusion 이고 confidence 같으면 skip (dedup)
3. 이전 entry 의 결론이 *반대 방향* 으로 뒤집히면 `superseded_by` 채움 (retraction tracking)
4. confidence < threshold (default 0.7) 인 결과는 INSERT 는 하되 `included_in_pack=false` 로 마킹 → ContextPackBuilder 가 fetch 시 제외

## 3. 책임 NOT

- 분석 결과 본문 저장 — 기존 `analysis_cache` (TTL 7일) 가 담당. ledger 는 *one_liner + meta* 만
- ContextPack assembly — ContextPackBuilder
- 분석 LLM 호출 자체 — 분석 agent 가 담당
- ledger 의 admin review UI — frontend admin 영역

## 4. 입력 스펙

```python
class LedgerInsertInput(TypedDict):
    analysis_type: Literal["insight", "mixer", "peer", "global", "briefing"]
    analysis_id: str                   # 원본 analysis_cache row id 참조
    peer_ids: list[str]                # 분석에 등장한 peer 들 (1~다 — Mixer 다중 가능)
    conclusion_one_liner: str          # ≤ 100자 — 분석 output 의 final_one_liner 그대로
    strategy_label: str | None         # peer 분석 한정 — 5종 enum
    confidence: float                  # 0.0~1.0
    source_card_ids: list[str]         # 분석에 사용된 카드 id (환각 검증용)
    sk_ax_implication: str | None      # 분석 output 의 sk_ax_implication
    langfuse_trace_id: str | None      # 3-tier observability Tier 3 — admin drill-down
    prompt_version: str                # 분석 agent 의 prompt_version (drift tracking)
    git_sha: str
```

## 5. 출력 스펙

### INSERT 결과

```python
class LedgerInsertResult(TypedDict):
    ledger_id: int                     # BIGSERIAL
    inserted: bool                     # False 면 dedup skip
    superseded_ledger_ids: list[int]   # 본 INSERT 가 뒤집은 이전 entry id 들
    included_in_pack: bool             # confidence >= threshold
```

### Fetch (ContextPackBuilder 가 호출)

```python
class LedgerEntry(TypedDict):
    """context-pack-builder.md 의 동일 schema"""
    analysis_id: str
    analysis_type: str
    conclusion_one_liner: str
    strategy_label: str | None
    confidence: float
    created_at: str
    source_card_ids: list[str]
    superseded_by: int | None
```

## 6. 알고리즘

### 6.1 INSERT 로직

```python
def insert(input: LedgerInsertInput) -> LedgerInsertResult:
    threshold = LEDGER_MIN_CONFIDENCE      # default 0.7
    included = input["confidence"] >= threshold

    # 1. Dedup check — 같은 peer + analysis_type + conclusion 이 직전 24h 에 있으면 skip
    for peer_id in input["peer_ids"]:
        recent = fetch_recent_ledger(peer_id, analysis_type=input["analysis_type"], within_hours=24)
        if any(_text_sim(r.conclusion_one_liner, input["conclusion_one_liner"]) >= 0.92
               and abs(r.confidence - input["confidence"]) < 0.05
               for r in recent):
            return {"ledger_id": -1, "inserted": False, "superseded_ledger_ids": [], "included_in_pack": False}

    # 2. Supersede detection — 같은 peer 의 이전 결론과 *반대 방향* 이면 supersede
    superseded = []
    for peer_id in input["peer_ids"]:
        prev_active = fetch_active_ledger(peer_id, analysis_type=input["analysis_type"], limit=5)
        for prev in prev_active:
            if _is_contradictory(prev, input):
                superseded.append(prev.id)

    # 3. INSERT
    ledger_id = pg.execute("""
        INSERT INTO analysis_ledger (
            analysis_type, analysis_id, peer_ids, conclusion_one_liner,
            strategy_label, confidence, source_card_ids, sk_ax_implication,
            langfuse_trace_id, prompt_version, git_sha, included_in_pack
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
    """, ...).fetchone()[0]

    # 4. Mark superseded
    if superseded:
        pg.execute("UPDATE analysis_ledger SET superseded_by=? WHERE id = ANY(?)", ledger_id, superseded)

    return {
        "ledger_id": ledger_id,
        "inserted": True,
        "superseded_ledger_ids": superseded,
        "included_in_pack": included,
    }
```

### 6.2 Supersede 판정 (`_is_contradictory`)

```python
def _is_contradictory(prev: LedgerEntry, current: LedgerInsertInput) -> bool:
    """이전 결론과 현재 결론이 반대 방향이면 supersede.
    검출 신호 (any 1+ True 시):
      1. strategy_label 가 5 enum 중 *반대 axis* (Aggressive ↔ Defensive Hold / Tech Pivot ↔ Cost Leadership)
      2. sk_ax_implication 의 긍정/중립/부정 분류가 반대 (긍정 → 부정)
      3. conclusion_one_liner 의 negation 키워드 detect ("축소" vs "확대" / "철수" vs "진출")
    """
    if prev.strategy_label and current.get("strategy_label"):
        if (prev.strategy_label, current["strategy_label"]) in CONTRADICTORY_PAIRS:
            return True
    if _polarity(prev.sk_ax_implication) != _polarity(current.get("sk_ax_implication","")):
        return True
    if _contains_negation_swap(prev.conclusion_one_liner, current["conclusion_one_liner"]):
        return True
    return False

CONTRADICTORY_PAIRS = {
    ("Aggressive Expansion", "Defensive Hold"),
    ("Defensive Hold", "Aggressive Expansion"),
    ("Tech Pivot", "Cost Leadership"),
    ("Cost Leadership", "Tech Pivot"),
}
```

자동 검출이 보수적 — false positive 줄임. 모호하면 supersede 안 함 (둘 다 active). admin review 가 분기 1회 ground truth 확정.

### 6.3 Fetch 로직 (ContextPackBuilder 가 호출)

```python
def fetch_ledger_top_n(peer_id, top_n=5, min_confidence=0.7) -> list[LedgerEntry]:
    """confidence >= threshold, not superseded, 최근 90일."""
    return pg.query("""
        SELECT analysis_id, analysis_type, conclusion_one_liner, strategy_label,
               confidence, created_at, source_card_ids, superseded_by
        FROM analysis_ledger
        WHERE ? = ANY(peer_ids)
          AND confidence >= ?
          AND superseded_by IS NULL
          AND included_in_pack = TRUE
          AND created_at > now() - interval '90 days'
        ORDER BY created_at DESC
        LIMIT ?
    """, peer_id, min_confidence, top_n)
```

90일 cutoff — quarterly canon 이 그보다 강한 ground truth 라 ledger 가 너무 길어지면 노이즈. quarterly 갱신 후 분기 ledger 는 자동 weight ↓ (월 단위 retention).

### 6.4 Retention / cleanup

```python
def cleanup_old_ledger(retention_days=90):
    """quarterly canon 출시 후 분기 ledger 는 자동 weight ↓."""
    pg.execute("""
        DELETE FROM analysis_ledger
        WHERE created_at < now() - interval '? days'
          AND analysis_type IN ('insight', 'mixer', 'peer')
    """, retention_days)
    # briefing / global 은 보관 (1년)
    pg.execute("""
        DELETE FROM analysis_ledger
        WHERE created_at < now() - interval '365 days'
          AND analysis_type IN ('briefing', 'global')
    """)
```

nightly cron `03:00 KST` 에 실행.

### 6.5 Integration 패턴 — 분석 agent 의 `with_provenance` decorator 와 통합

```python
from functools import wraps
from src.middleware.analysis_ledger import AnalysisLedger

def with_ledger_writeback(agent_class: str):
    def deco(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            if not isinstance(result, dict):
                return result
            try:
                AnalysisLedger.insert({
                    "analysis_type": _map_agent_to_type(agent_class),
                    "analysis_id": result.get("id") or result.get("mix_id") or result.get("insight_id") or _generate_id(),
                    "peer_ids": _extract_peers(result, kwargs),
                    "conclusion_one_liner": result.get("final_one_liner", "")[:100],
                    "strategy_label": result.get("strategy_label"),
                    "confidence": result.get("confidence", 0.0),
                    "source_card_ids": result.get("sources_used") or result.get("source_card_ids") or [],
                    "sk_ax_implication": result.get("sk_ax_implication"),
                    "langfuse_trace_id": result.get("langfuse_trace_id"),
                    "prompt_version": kwargs.get("_prompt_version", "unversioned"),
                    "git_sha": _GIT_SHA_CACHE,
                })
            except Exception as e:
                logger.warning(f"AnalysisLedger insert failed: {e}")  # 비핵심 — fail-soft
            return result
        return wrapper
    return deco

# 사용
@with_provenance("MixerAnalysisAgent", "analyze")
@with_ledger_writeback("MixerAnalysisAgent")
async def analyze(card_ids, **kwargs): ...
```

`with_ledger_writeback` 은 `with_provenance` 의 *밖에서* wrapping (decorator outermost) — 분석이 성공한 결과만 ledger 에 들어가게.

### 6.6 Prompt audit — N/A

`AnalysisLedger` 는 LLM 미사용 (DB write-through 미들웨어). prompt audit 표 X.

다만 17 요소의 **§17 반복 추적 구조** 의 *시스템 레벨* 구현체 — chat-orchestrator 의 follow_up_suggestions 가 *single turn* 의 trail 이라면, ledger 는 *분석 history* 의 trail.

## 7. LLM 모델 + token 예산

- **LLM 미사용** — pure 미들웨어
- 비용: ₩0
- Latency: INSERT ~5ms, supersede check 추가 ~10ms

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| `insert()` 실패 (PG down 등) | warn log + 분석 결과는 그대로 return (fail-soft) — ledger 누락 1회는 시스템 영향 미미 |
| Dedup 판정 모호 | INSERT 진행 (보수적) |
| `_is_contradictory` 판정 모호 | supersede 안 함 (false positive 회피) |
| confidence=None (분석 agent 가 confidence 안 채움) | confidence=0.0 fallback + included_in_pack=False |
| peer_ids 비어있음 | INSERT 진행하되 fetch 시 peer 기반 검색 불가 — global / cross-peer mixer 만 해당 |

## 9. 외부 의존성

- **DB (PG)**: `analysis_ledger` (INSERT / SELECT / UPDATE — V19)
- **DB (PG)**: `analysis_ledger_card_news`, `analysis_ledger_peer_companies` (V21 관계 정규화. JSONB 배열은 legacy/compat 유지)
- **외부 API**: 없음
- **Cross-agent**: 분석 4 agent + Briefing 의 decorator 통합

### Flyway V19 schema + V21 mappings

```sql
-- V19
CREATE TABLE analysis_ledger (
    id                   BIGSERIAL PRIMARY KEY,
    analysis_type        VARCHAR(20)  NOT NULL,   -- insight | mixer | peer | global | briefing
    analysis_id          VARCHAR(100) NOT NULL,
    peer_ids             JSONB        NOT NULL,
    conclusion_one_liner TEXT         NOT NULL,
    strategy_label       VARCHAR(30),
    confidence           REAL         NOT NULL,
    source_card_ids      JSONB        NOT NULL,
    sk_ax_implication    TEXT,
    langfuse_trace_id    VARCHAR(64),
    prompt_version       VARCHAR(20),
    git_sha              VARCHAR(12),
    included_in_pack     BOOLEAN      NOT NULL DEFAULT TRUE,
    superseded_by        BIGINT       REFERENCES analysis_ledger(id),
    created_at           TIMESTAMPTZ  DEFAULT now()
);

-- ContextPackBuilder 의 fetch_ledger_top_n hot path
CREATE INDEX idx_ledger_peer_active ON analysis_ledger
    USING gin (peer_ids jsonb_path_ops);

CREATE INDEX idx_ledger_active_recent ON analysis_ledger (created_at DESC)
    WHERE superseded_by IS NULL AND included_in_pack = TRUE;

CREATE INDEX idx_ledger_supersede ON analysis_ledger (superseded_by)
    WHERE superseded_by IS NOT NULL;

-- V21
analysis_ledger.id
  ├── analysis_ledger_card_news.analysis_ledger_id -> card_news.id
  └── analysis_ledger_peer_companies.analysis_ledger_id -> peer_companies.id
```

## 10. State 흐름

분석 agent 의 LangGraph 그래프 내부 — state 거치지 않고 decorator 가 in-process INSERT. 별도 supervisor state 불필요.

## 11. Provenance + Confidence

- **Provenance**: ledger 자체가 provenance 의 일종. `analysis_id` + `langfuse_trace_id` + `prompt_version` + `git_sha` 모두 trace. V21 이후 `source_card_ids`/`peer_ids` 는 mapping table 에도 동시 저장.
- **Confidence**: 분석 agent 가 제공한 값 그대로 carry. ledger 가 별도 confidence 계산 X.

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit (basic INSERT) | Mixer 결과 confidence=0.85 | ledger row 1 INSERT, included_in_pack=True |
| Unit (low confidence) | Insight 결과 confidence=0.55 | INSERT but included_in_pack=False |
| Unit (dedup) | 24h 내 동일 conclusion 재호출 | inserted=False |
| Unit (supersede) | Peer "Aggressive Expansion" → 다음 분석 "Defensive Hold" | 이전 row.superseded_by 채워짐 |
| Unit (multi peer) | Mixer 가 peer=[samsung, lg, hyundai] | peer_ids=[3] JSONB 저장 + fetch 시 각 peer 별 검색 가능 |
| Unit (fetch) | peer_id 로 top 5 조회 | confidence >= 0.7, not superseded, 90일 내, DESC |
| Edge | `insert()` 가 PG timeout | 분석 결과는 정상 return + warn log |
| Edge | 90일 지난 entry retention | nightly cleanup 후 삭제 (briefing/global 제외) |
| Integration | Insight → Mixer 연속 호출 | Mixer 의 pack 에 Insight 의 ledger entry 포함 |

## 13. 모니터링

- **pipeline_logs.step**: `ledger_writeback`
- **KPI**:
  - INSERT 성공률 ≥ 99.5% (PG 단순 INSERT)
  - dedup 비율 5~15% (정상 — 같은 peer 재분석 보통 발생)
  - supersede 비율 1~5% (너무 높으면 분석 일관성 의심 — admin review)
  - included_in_pack 비율 ≥ 70% (confidence threshold 기준)
  - retention 후 active row 수 ≤ 5,000 (peer 10 × 분석 10 × 90일 ≈ 9,000 — gin index OK)
- **Token budget**: ₩0
- **Admin alert**: 분기 1회 supersede chain 길이 ≥ 3 인 peer 면 알림 (strategy 가 자꾸 뒤집힘 = 분석 품질 의심)

## 14. 구현 메모 + Changelog

### 핵심 파일

- `src/middleware/analysis_ledger.py` (신규 — `AnalysisLedger.insert()` + decorator `with_ledger_writeback`)
- `src/db/analysis_ledger.py` (PG INSERT / SELECT / UPDATE helper)

### 분석 agent 측 변경 (decorator 추가만)

- `mixer_analysis_agent.py`: `@with_ledger_writeback("MixerAnalysisAgent")`
- `insight_cascade_agent.py`: `@with_ledger_writeback("InsightCascadeAgent")`
- `peer_comparison_agent.py`: `@with_ledger_writeback("PeerComparisonAgent")`
- `global_trends_agent.py`: `@with_ledger_writeback("GlobalTrendsAgent")`
- `briefing_generation_agent.py`: `@with_ledger_writeback("BriefingGenerationAgent")` (briefing 전체 1 ledger entry — section 별 X)

### Phase K1 — 단독 도입 가능

본 agent + V19 Flyway 만으로 K1 완성. V21 mapping 은 FK 기반 조회/정합성 검증을 위한 보강이다. K2/K3 (compaction / context pack) 없이도 *분석 결과가 다음 분석에 보이는 효과* 달성. K1 → K2/K3 → K4/K5 점진 진행.

### Changelog

- **v1 (2026-05-14)** — 신설. 5 계층 아키텍처의 carry-over backbone. Phase K1 우선 도입 권장.
- **v1.1 (2026-05-15)** — 실제 Flyway 번호 정정: `analysis_ledger` 는 V19, FK 정규화 mapping 은 V21. `source_card_ids`/`peer_ids` JSONB 유지 + mapping table 동시 저장 정책 추가.
