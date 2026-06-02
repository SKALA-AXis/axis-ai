# Global IT Trends + Peer Alignment — Design Plan

> **상태 (2026-05-26)**: 🟡 **schema/스켈레톤 존재, 실제 LLM 로직 미구현**.
> 본 문서는 사용자 요청 ("글로벌 IT 트렌드 추출 + AX/peer 동향이 같은 결로 가는지 비교") 을
> **기존 자산 최대 재활용** 으로 충족하는 설계 계획이다.
> 새 DB 테이블·새 agent 모두 **불필요** — 모두 in-place 확장.

---

## 0. TL;DR

| 항목 | 답 |
|---|---|
| 새 agent 만들어야 하나? | **❌ 만들지 않는다.** 기존 `ITTrendAgent` 의 `generate()` 만 채운다. |
| 새 DB 테이블 만들어야 하나? | **❌ 만들지 않는다.** `global_industry_trends` (DDL 완비, V29) 그대로 사용. ⚠ `analysis_ledger` 는 V30 line 694 에서 DROP — 직접 INSERT 경로 (§7). |
| Peer alignment 결과를 어디에 담나? | `global_industry_trends.payload` JSONB 의 `peer_alignment` 키 (1행 × keyword 안에 4 peer + SK AX 모두 포함). |
| 다른 agent 와의 통합 방법 | `analyzer.py` 가 이미 `cluster_metadata["trend_context"]` 를 prompt 에 반영 — 채워주는 **wire-up 만** 추가. |
| 새 API endpoint 필요한가? | `POST /global/trends/run` schema 가 `src/api/global_trends_schemas 2.py` 에 이미 정의. 파일명 정리 + router 등록만 필요. |
| 새 cronjob 필요한가? | 1 개 — `axis-cron-global-trend` (매일 새벽 02:30 KST). |

> ⚠ **검증 단계에서 발견된 2가지 위험** (§16, §17 에서 상세) — 본 설계는 이를 반영함:
> 1. **`analysis_ledger` 테이블은 V30 line 694 에서 DROP 됨.** axis-ai 의 `@with_ledger_writeback` 데코레이터는 silent fail 상태. V30 의 `analysis_ledger → global_industry_trends` 복제 SQL 은 **마이그레이션 1회성** 이며 이후로는 동작 안 함. → ITTrendAgent 는 `global_industry_trends` 에 **직접 upsert** 해야 한다 (§7).
> 2. **`card_news` 에 SK AX 카드가 0건.** 4 peer 만 카드 보유. → Phase 3 의 SK AX alignment 입력은 `raw_articles` (SK AX Site 155건 + Newsroom 12건 + dart 26건 + ir_pdf 14건) 에서 직접 추출해야 한다.

---

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `ITTrendAgent` (확장) — 별칭 `GlobalTrendsAgent` |
| **단계** | 2단계 (저장 데이터 활용) — `DataUsageOrchestrator` 의 `it_trend` task |
| **위치** | `src/agents/it_trend_agent.py` (in-place 확장) |
| **Supervisor** | `DataUsageOrchestrator` ([data_usage_orchestrator.py:14](../../src/pipeline/data_usage_orchestrator.py)) |
| **결과 저장** | `global_industry_trends` 에 **직접 upsert** ([§7](#7-저장-흐름)). axis-ai 표준 `@with_ledger_writeback` 는 `analysis_ledger` 가 V30 line 694 에서 DROP 되어 사용 불가. |
| **호출 빈도** | 매일 1회 (cronjob) + on-demand (POST /global/trends/run) |
| **트리거** | `axis-cron-global-trend` cronjob (KST 02:30 / UTC 17:30) |

---

## 2. 책임 정의

### 한 줄

> SPRi/BCG 리서치 + 글로벌 6사 newsroom 카드뉴스 + 과거 TrendContext 를 입력으로,
> 글로벌 IT 트렌드 라인을 키워드 단위로 추출하고, 각 keyword 별로 **(SK AX + 4 peer) 가 그 흐름에
> alignment / lagging / missing / diverging 중 어디인지** 를 함께 산출한다.

### 책임 (DO)

1. **Snapshot**: 글로벌 6사 newsroom 의 최근 N일 카드뉴스를 회사별로 묶어 top_themes / headlines 추출
2. **Trend Detection**: SPRi/BCG + 6사 newsroom 에서 반복되는 keyword 를 detection
   (frequency_delta_pct, intensity weak/moderate/strong, leading_companies)
3. **Peer Alignment**: 각 trend keyword 마다 SK AX + 4 peer (Samsung SDS, LG CNS, Hyundai AutoEver,
   POSCO DX) 의 최근 카드뉴스 mention / activity 분석 → `aligned | lagging | missing | diverging`
4. **Impact Mapping** (LLM, P3): trend × SK AX business line 매트릭스 (direction/magnitude)
5. **Forecast** (LLM, P4): 1Q/6M/1Y horizon 별 narrative + recommended_response
6. **Synthesis** (LLM, P5): final_one_liner + sk_ax_implication
7. **Persistence**: `global_industry_trends` 에 **직접 upsert** (1 keyword = 1 row, `source_analysis_id` per-keyword 유니크). ⚠ `analysis_ledger` 경로는 V30 에서 DROP — 사용 불가.
8. **TrendContext upcast**: 결과를 `TrendContext` DTO 로도 반환해서 `AnalysisAgent` 가 prompt context 로 사용

### 책임 NOT

- 원문 크롤링 (`crawler/sources/spri.py`, `bcg.py`, `global_newsroom.py` 가 담당)
- IntegratedIssue / AnalysisResult / ImplicationResult 생성 (Track A pipeline 이 담당)
- 카드뉴스 생성 (CardNewsAgent 담당, 글로벌 newsroom 도 카드 만든 후 그 결과를 input 으로만 사용)
- Peer ↔ SK AX 양자 비교 (별도 `PeerComparisonAgent` design 으로 분리 — `design/30-analysis 2/peer-comparison.md` 참조)
- 새 DB 테이블 / 새 SQL migration
- 새 별도 alignment agent

---

## 3. 현재 자산 인벤토리 (재활용 매핑)

### 3.1 코드 — 그대로 사용

| 자산 | 위치 | 재활용 방식 |
|---|---|---|
| `ITTrendAgent` 클래스 / `build_input` | [src/agents/it_trend_agent.py:41-142](../../src/agents/it_trend_agent.py) | 그대로 골격 사용. `generate()` 안만 5-phase 로 확장. ⚠ **`build_input` 의 `_split_trend_inputs` 는 글로벌 6사 raw 를 drop** 함 (§4.1 참조) — 우회 경로 채택. |
| `ITTrendInput` DTO | 동 파일 lines 25-38 | 그대로. `trend_items` 채우는 방식을 `build_input` 우회 (§4.1) 로 변경. |
| `TrendContext` DTO | [src/analysis/models.py:521-542](../../src/analysis/models.py) | 그대로 반환. `signals[]` 에 `peer_alignment` 들어감. |
| `_TREND_SOURCE_NAMES = {"spri", "bcg"}` | it_trend_agent.py:20 | 그대로 (research source 분류용). |
| `_GLOBAL_NEWSROOM_SOURCE_TYPES = {"global_newsroom","company_newsroom"}` | it_trend_agent.py:21 | ⚠ **실제 DB 의 raw_articles.source_type 은 `'official'`** (e.g. `nvidia_official`, `meta_official`). 즉 현재 상수는 raw 매칭 안 됨. **확장 또는 우회**. §4.1 참조. |
| `DataUsageOrchestrator.run` 의 `it_trend` 라우팅 | [data_usage_orchestrator.py:58-67](../../src/pipeline/data_usage_orchestrator.py) | 그대로. peer_alignment 도 같은 task 안에서 자동 처리. |
| `analyzer.py` prompt 의 `trend_context` 변수 | [src/analysis/analyzer.py:41,60,238,242](../../src/analysis/analyzer.py) | 그대로. ITTrendAgent 결과만 `cluster_metadata["trend_context"]` 로 wire-up. |
| `analysis_flow_graph.py` 의 `bundle.metadata["trend_context"]` 읽기 | [src/pipeline/analysis_flow_graph.py:635](../../src/pipeline/analysis_flow_graph.py) | 그대로. |
| ~~`analysis_ledger` writeback decorator~~ | `src/middleware/analysis_ledger.py` (`@with_ledger_writeback`) | ⚠ **사용 불가** — `analysis_ledger` 테이블이 V30 line 694 에서 `DROP TABLE`. 데코레이터는 silent fail 상태. ITTrendAgent 는 `global_industry_trends` 에 **직접 INSERT** (§7 참조). |
| `GlobalTrendsRequest/Response` schema | **source-of-truth = [`axis-infra/api/openapi.yaml` `#/components/schemas/GlobalTrendsRequest` / `GlobalTrendsResult`](../../../axis-infra/api/openapi.yaml)** (git tracked). | ⚠ axis-ai 의 `src/api/global_trends_schemas.py` (정본) 은 **존재하지 않음**. ` 2` suffix Finder 중복본은 untracked (`?? src/api/global_trends_schemas 2.py`) 라 다른 팀원 clone 에는 아예 없음. S1 은 **openapi.yaml 의 schema 정의를 보고 Pydantic class 신규 작성** (단순 rename/복제 아님). |
| `MixerAnalysisAgent` 의 6축 radar 산식 (`_compute_radar`) | [src/agents/mixer_analysis_agent.py:660-691](../../src/agents/mixer_analysis_agent.py) | **패턴만 차용** — peer alignment 점수도 결정적 산식 + LLM 보완 hybrid. |
| `MixerAnalysisAgent._fetch_cards` | mixer_analysis_agent.py:739-776 | **패턴만 차용** — peer 카드 fetch SQL 그대로 사용. |
| `MixerAnalysisAgent._format_analysis_units` | mixer_analysis_agent.py:870-923 | **패턴만 차용** — LLM prompt 용 카드 압축 형식. |

### 3.2 DB — 그대로 사용 (변경 없음)

| 테이블 / 뷰 | 역할 | 사용 방법 |
|---|---|---|
| `raw_articles` | Snapshot Phase 1 + SK AX alignment input | (a) global 6사: `source_name IN ('spri','bcg','nvidia_official','meta_official','amazon_official','microsoft_official','google_official','apple_newsroom')`<br>(b) **SK AX (5/26 확인 — card_news 에 sk_ax 0건이므로 raw 직접 사용)**: `source_name IN ('SK AX Site','SK AX Newsroom','dart','ir_pdf') OR company::text ILIKE '%sk_ax%'` |
| `card_news` | Alignment Phase 3 input (peer 4사 활동) | `peer_company_id IN ('samsung_sds','lg_cns','posco_dx','hyundai_autoever')` AND `created_at >= now() - interval '30 days'`<br>⚠ **`sk_ax` 는 0건** — SK AX 는 raw_articles 에서 별도 fetch |
| `peer_companies` | Peer 목록 source (정적) | `SELECT id, name FROM peer_companies` — 5/26 기준: `sk_ax, samsung_sds, lg_cns, posco_dx, hyundai_autoever` 5개 |
| `peer_event_timeline` (VIEW) | Alignment 의 시간 차 분석 (`recency_gap_days`) | `card_news` 위 derived view — 마찬가지로 SK AX 는 비어있을 가능성 |
| ~~`analysis_ledger`~~ | ~~결과 저장 (axis-ai 표준)~~ | ⚠ **V30 line 694 에서 DROP** — 사용 불가 |
| `global_industry_trends` | **결과 저장 (primary)** | ⚠ V30 자동 복제 path 는 안 작동 (analysis_ledger DROP). ITTrendAgent 가 **직접 INSERT** (V30 의 `uq_global_industry_trends_source_analysis_id` UNIQUE 사용). |

#### `global_industry_trends` 컬럼 매핑 — Peer Alignment 결과 어디에 담나

```sql
-- 1 keyword 당 1 row. ITTrendAgent 가 직접 upsert (analysis_ledger 경로는 V30 DROP).

trend_date       -- 분석 일자 (UTC date)
industry         -- e.g. "ai_infrastructure"
region           -- 'global' (고정) / 'apac' 등
keyword          -- e.g. "agentic_ai" (trend keyword 1개)
keyword_category -- e.g. "ai_tech" (card_news.primary_keyword_category 와 같은 분류)
title            -- trend headline (LLM Synthesis 산출)
summary          -- trend 1~2 문장 요약 (LLM Synthesis 산출)
mention_count    -- 글로벌 6사 newsroom + SPRi/BCG 에서 등장 횟수 (결정적 산식)
impact_score     -- 0~100 (frequency × recency × source_tier 결정적 산식)
confidence       -- 0~1 (LLM)
related_peer_ids -- aligned 인 peer 만 ["samsung_sds","lg_cns"] (직접 컬럼 사용)
related_card_ids -- peer alignment evidence card ids ["CN-..."] (직접 컬럼 사용)
source_raw_article_ids -- SPRi/BCG/newsroom 원천 article id
sk_ax_implication -- SK AX 한 줄 (LLM Synthesis 산출, 직접 컬럼 사용)
payload          -- 자유 JSONB ⚠ alignment 풀 결과 + impact_matrix + forecasts
```

#### `payload` JSONB 안의 구조 (Peer Alignment + Impact + Forecast 통합)

```jsonc
{
  // (a) per-peer alignment (5 peer × 1 keyword)
  "peer_alignment": [
    {
      "peer_id": "sk_ax",
      "alignment_type": "aligned",   // aligned | lagging | missing | diverging
      "alignment_score": 0.72,       // -1 ~ 1
      "peer_mention_count": 4,
      "global_mention_count": 18,
      "recency_gap_days": 7,         // 글로벌 평균 vs peer 최근 활동의 시간 차
      "evidence_card_ids": ["CN-20260520-003", "CN-20260518-011"],
      "strategic_note": "..."        // LLM 1줄 요약
    },
    { "peer_id": "samsung_sds", "alignment_type": "aligned", ... },
    { "peer_id": "lg_cns", "alignment_type": "lagging", ... },
    { "peer_id": "hyundai_autoever", "alignment_type": "missing", ... },
    { "peer_id": "posco_dx", "alignment_type": "diverging", ... }
  ],

  // (b) impact matrix — trend × SK AX business line (Phase 3, LLM)
  "impact_matrix": [
    {
      "sk_ax_line": "cloud_ax",
      "direction": "positive",     // positive | neutral | negative
      "magnitude": "high",         // low | medium | high
      "channel": "manufacturing AI agent demand",
      "quant_hint": "관련 매출 +10~20% 예상",
      "source_marker": "global_6_co_avg"
    }
  ],

  // (c) forecast (Phase 4, LLM)
  "forecasts": [
    {
      "horizon": "1Q",
      "scenario": "baseline",
      "narrative": "...",
      "sk_ax_impact": "...",
      "drivers": ["agent infra cost ↓", "..."],
      "risk_level": "medium",
      "recommended_response": "..."
    }
  ],

  // (d) phase별 reasoning trail (auditability, Langfuse 연동)
  "reasoning_steps": [
    { "step_idx": 1, "phase": "snapshot", "question": "...", "answer": "...", "confidence": 0.8 }
  ],

  // (e) provenance
  "git_sha": "...",
  "prompt_version": "global-trends-v1.0",
  "langfuse_trace_id": "...",
  "leading_companies": ["nvidia", "microsoft"]
}
```

**왜 새 테이블 안 만드나** — `payload` JSONB 가 free-form 이고, alignment 는 **trend keyword 종속** 이라 별도 PK 가 필요 없음. 새 테이블 (`peer_trend_alignment`) 을 만들면 두 곳에 row 가 분산되어 read 성능 이득 없이 V30 의 minimal product schema 정책 ("collapse legacy tables into minimal product schema") 만 거스른다.

### 3.3 Design 문서 — 이미 존재 (참조만)

| 문서 | 상태 | 본 문서와의 관계 |
|---|---|---|
| `design/30-analysis/it-trend.md` | 구조 설계만 | 본 문서가 **상위 superset** — it-trend 의 책임을 그대로 포함하고 alignment + impact + forecast 확장 |
| `design/30-analysis 2/peer-comparison.md` | 🟡 미구현 (axis-ai 신규) | **별개 agent** — peer 1명 vs SK AX 양자 비교 (5-strategy label). 본 문서는 N peer 가 동일 trend 라인을 따라가는지의 alignment. prompt 구조 일부 차용 가능 |
| `design/30-analysis/mixer-analysis.md` | ✅ 구현 완료 | radar 산식 + linked-result fetch 패턴 차용 |
| `design/30-analysis/link-verification.md` | 참고 | trend 와 card 의 연결 검증 시 활용 |

### 3.4 미구현 / 누락 자산

| 자산 | 현재 상태 | 작업 |
|---|---|---|
| `ITTrendAgent.generate()` 본문 | 빈 결과 (`trend_summary=""`, `validation.pass=False`) | 5-phase 로 채움 |
| `db/article_store.py` 의 trend reader | `get_articles_by_ids` 만 있음 | `fetch_global_trend_inputs(window_days, sources)` + `fetch_peer_cards_for_alignment(peer_ids, window_days, keyword)` 추가 |
| `api/router.py` 의 `/global/trends/run` endpoint | 등록 안 됨 | router 등록 |
| `api/global_trends_schemas.py` (정본) | **존재하지 않음**. ` 2` suffix Finder 중복본도 `git status` 상 untracked — 다른 팀원 clone 에 없음. | **신규 작성**. source-of-truth 는 `axis-infra/api/openapi.yaml` 의 `GlobalTrendsRequest` / `GlobalTrendsResult` components (이미 정의됨, git tracked). 거기서 type/field 를 그대로 Pydantic 으로 옮긴다. |
| `axis-cron-global-trend` cronjob | 없음 | `axis-infra/k8s/base/cronjob-global-trend.yaml` 추가 |
| backend `GET /api/global-trends?from=...` read API | 없음 | (선택) — Spring Controller 1개 (read-only) |

---

## 4. 입력 스펙

`ITTrendInput` DTO 그대로 사용. `trend_items` 에 다음 4가지 카테고리가 함께 들어감.

```python
@dataclass
class ITTrendInput:
    trend_items: list[dict]              # SPRi/BCG + 글로벌 newsroom raw or integrated_issue
    period: str | None                   # e.g. "2026-W21" or "2026-05-20~2026-05-26"
    source_groups: list[str]             # ["spri", "bcg", "nvidia", "meta", ...]
    previous_trend_context: dict | None  # 직전 주기 TrendContext (delta 계산)
    reference_issue_results: list[dict]  # 글로벌 newsroom 의 IntegratedIssue/AnalysisResult
    metadata: dict                       # peer_alignment 옵션 등
```

### 4.1 ⚠ `build_input()` 의 source filter 문제와 우회 경로

`ITTrendAgent.build_input()` 은 내부적으로 `_split_trend_inputs()` 를 호출 ([it_trend_agent.py:145-158](../../src/agents/it_trend_agent.py)) → 다음 두 조건만 통과시킴:

```python
# it_trend_agent.py:161-169
def _is_spri_bcg_trend_item(item):
    return source_type == "trend_report" and source_name in {"spri", "bcg"}

def _is_global_newsroom_reference_candidate(item):
    return source_type in {"global_newsroom", "company_newsroom"}
```

그러나 **실제 raw_articles 의 `source_type` 은 `'official'`** (nvidia_official, microsoft_official, meta_official, amazon_official, google_official, apple_newsroom). DB 검증 결과 (`SELECT source_type, source_name FROM raw_articles GROUP BY ...`):

```text
source_type        | source_name        | n
-------------------+--------------------+----
official           | amazon_official    | 476
official           | meta_official      | 226
official           | nvidia_official    | 219
official           | microsoft_official | 129
official           | google_official    |  45
official           | apple_newsroom     |  14
trend_report       | bcg                |  53
trend_report       | spri               |  40
```

→ `_split_trend_inputs()` 가 글로벌 6사 raw (1,109건) 를 **`unsupported_items` 로 흘려보냄** — Phase 1/2 의 핵심 신호 소실.

**선택지**:

| 옵션 | 설명 | 평가 |
|---|---|---|
| **A. `_split_trend_inputs` 확장** | `_GLOBAL_NEWSROOM_SOURCE_TYPES` 에 `'official'` 추가 + `source_name` whitelist 확인 추가 | ⚠ `source_type='official'` 은 글로벌이 아닌 다른 자사 sources (e.g. `LG CNS Press`, `SK AX Newsroom`, `POSCO DX NewsRoom`) 와 충돌. 단순 type 추가로는 글로벌만 분리 안 됨. |
| **B. fetcher 가 `source_type` 을 정규화해서 주입** | `fetch_global_trend_inputs()` 가 raw row 를 dict 로 만들 때 `source_type` 을 `'global_newsroom'` 으로 덮어씀 | ✅ build_input 코드 변경 최소. fetcher 1 곳만 책임. |
| **C. build_input 우회** | `ITTrendAgent.generate()` 가 raw fetched items 를 직접 `ITTrendInput(trend_items=...)` 로 생성하고 `build_input` 호출 안 함 | ✅ 가장 단순. 하지만 `build_input` 의 normalize 로직 (source_groups dedup, metadata 부착) 을 직접 재구현해야 함. |

**본 설계는 옵션 B 채택** — fetcher 책임 단일화.

```python
# db/article_store.py 신규 (작업 S2)
def fetch_global_trend_inputs(window_days: int = 30) -> list[dict[str, Any]]:
    """글로벌 6사 + SPRi/BCG raw_articles 를 build_input 호환 dict 로 정규화 반환.
    
    raw_articles.source_type 이 'official' 인 글로벌 6사 row 는 ITTrendAgent 의
    _split_trend_inputs() 에서 unsupported_items 로 떨어지므로, 여기서 source_type 을
    'global_newsroom' 으로 덮어써서 정규화한다 (옵션 B).
    """
    GLOBAL_NEWSROOMS = {
        "nvidia_official", "microsoft_official", "google_official",
        "amazon_official", "meta_official", "apple_newsroom",
    }
    RESEARCH = {"spri", "bcg"}
    
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT id, source_name, source_type, title, content, url,
                   published_at, collected_at, metadata, company
            FROM raw_articles
            WHERE source_name = ANY(:names)
              AND collected_at >= NOW() - make_interval(days => :days)
            ORDER BY collected_at DESC
        """), {"names": list(GLOBAL_NEWSROOMS | RESEARCH), "days": window_days}).mappings().all()
    
    items: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["source_id"] = d["id"]
        if d["source_name"] in GLOBAL_NEWSROOMS:
            d["source_type"] = "global_newsroom"  # ← 정규화 핵심
        elif d["source_name"] in RESEARCH:
            d["source_type"] = "trend_report"
            d["publisher"] = d["source_name"]  # SPRi/BCG 매칭 helper 가 publisher 도 확인
        items.append(d)
    return items
```

### 입력 source 분류 (현 raw_articles 기준 — 5/26 측정)

| 카테고리 | source_name | 건수 | 역할 |
|---|---|---|---|
| **A. 리서치** | `spri`, `bcg` | 93 | Phase 2 Trend Detection 의 1차 신호 (weight=1.0) |
| **B. 글로벌 6사 newsroom** | `nvidia_official`, `microsoft_official`, `google_official`, `amazon_official`, `meta_official`, `apple_newsroom` | 1,109 | Phase 1 Snapshot 의 회사별 묶음 (weight=0.7) |
| **C. AX/Peer 자사** | `SK AX Site`, `SK AX Newsroom`, `LG CNS Press`, `Samsung SDS Press Release`, `POSCO DX NewsRoom`, `Hyundai AutoEver News` | 431 | Phase 3 Peer Alignment 의 자사 신호 |
| **D. AX/Peer 외부 시각** | `naver_news` (필터: `importance_score >= 0.5`), `naver_research`, `dart`, `ir_pdf` | 일부 | Phase 3 Peer Alignment 의 외부 시각 보조 (over-noisy 라 카드뉴스 단위로만 사용) |

### `metadata` 옵션

```jsonc
{
  "window_days": 30,
  "include_peer_alignment": true,         // default true
  "peer_company_ids": ["sk_ax", "samsung_sds", "lg_cns", "posco_dx", "hyundai_autoever"],
  "sk_ax_business_lines": ["cloud_ax", "manufacturing_ax", "data_platform", "smart_factory"],
  "min_mention_count": 3,                 // trend 후보 keyword 컷오프
  "max_trend_count": 8                    // 출력 keyword 수 cap (LLM 비용 보호)
}
```

---

## 5. 출력 스펙

### 5.1 In-process (DataUsageOrchestrator → caller)

`GlobalTrendsResponse` ([global_trends_schemas 2.py:98-122](../../src/api/global_trends_schemas%202.py)) 그대로.

핵심 필드 mapping:

| Response 필드 | 본 design 의 phase | 저장 위치 |
|---|---|---|
| `snapshots` | Phase 1 (deterministic) | `payload.snapshots` |
| `trend_detections` | Phase 2 (deterministic + LLM 보완) | `payload.trend_detections` (= keyword 별 mention/intensity/leading_companies) |
| `impact_matrix` | Phase 3-LLM | `payload.impact_matrix` |
| `forecasts` | Phase 4-LLM | `payload.forecasts` |
| `final_one_liner` | Phase 5-LLM | `global_industry_trends.title` (per row) |
| `sk_ax_implication` | Phase 5-LLM | `global_industry_trends.sk_ax_implication` (per row) |
| `reasoning_steps` | 전 phase | `payload.reasoning_steps` |
| `confidence` | LLM | `global_industry_trends.confidence` |

### 5.2 부산물 — TrendContext (AnalysisAgent 용)

ITTrendAgent.generate 의 반환값은 그대로 `TrendContext` shape. wire-up 의 **구체적 inject 지점은 2곳**:

#### (a) Reader 신규 — `db/article_store.py` (process-level TTL 캐시 포함)

> ⚠ **호출 빈도 주의**: `analysis_input_bundle_from_articles()` / `analysis_input_bundle_from_bundle()` 가 cluster 분석마다 호출되므로 hot path. 같은 SELECT 반복 방지 필요. **process-level TTL 캐시 60초** 적용.

```python
import time
import threading
from typing import Any

# trend 는 cronjob 으로 일 1회만 갱신되므로 60초 캐시는 정합성 손실 거의 없음.
_TREND_CACHE_TTL_SEC = 60
_trend_cache: dict[int, tuple[float, dict[str, Any]]] = {}   # key = within_days
_trend_cache_lock = threading.Lock()


def fetch_latest_trend_context(within_days: int = 7) -> dict[str, Any]:
    """global_industry_trends 의 최근 N일 row 를 TrendContext shape 로 aggregate.
    
    AnalysisAgent prompt 에 들어갈 글로벌 배경 정보. trend 가 없으면 빈 dict 반환
    (analyzer.py 가 already-empty-safe). process-level TTL 캐시 60초.
    """
    now = time.monotonic()
    with _trend_cache_lock:
        cached = _trend_cache.get(within_days)
        if cached and (now - cached[0]) < _TREND_CACHE_TTL_SEC:
            return cached[1]
    
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT keyword, summary, payload, trend_date, source_analysis_id
            FROM global_industry_trends
            WHERE trend_date >= CURRENT_DATE - make_interval(days => :days)
            ORDER BY impact_score DESC NULLS LAST, trend_date DESC
            LIMIT 10
        """), {"days": within_days}).mappings().all()
    if not rows:
        result: dict[str, Any] = {}
    else:
        result = {
            "period": f"last_{within_days}d",
            "trend_summary": " / ".join(r["summary"] or "" for r in rows[:3] if r["summary"]),
            "trend_lines": [r["summary"] for r in rows if r["summary"]],
            "signals": [
                {
                    "signal": r["keyword"],
                    "intensity": (r["payload"] or {}).get("intensity"),
                    "leading_companies": (r["payload"] or {}).get("leading_companies", []),
                    "source_ids": [r["source_analysis_id"]] if r["source_analysis_id"] else [],
                }
                for r in rows
            ],
            "source_groups": ["global_industry_trends"],
            "sources": [{"source_id": r["source_analysis_id"], "title": r["keyword"]} for r in rows],
            "reference_issue_ids": [],
            "updated_at": rows[0]["trend_date"].isoformat(),
            "validation": {"pass": True},
            "metadata": {"row_count": len(rows)},
        }
    
    with _trend_cache_lock:
        _trend_cache[within_days] = (now, result)
    return result


def invalidate_trend_context_cache() -> None:
    """ITTrendAgent 가 cron 으로 새 trend 를 upsert 한 직후 호출하면 즉시 반영."""
    with _trend_cache_lock:
        _trend_cache.clear()
```

#### (b) Inject 지점 — `src/agents/integration_agent.py` (`analysis_input_bundle_from_articles`)

`AnalysisInputBundle.metadata` 가 만들어지는 **정확한 line 154 의 dict 리터럴** 에 `"trend_context"` 추가:

```python
# src/agents/integration_agent.py:143-160 (변경 후)
return AnalysisInputBundle(
    bundle_id=bundle_id,
    cluster_id=str(cluster_id) if cluster_id is not None else None,
    source_type=source_type,
    companies=companies,
    sectors=sectors,
    event_type=str(classification.get("event_type") or "") or None,
    items=articles,
    facts=facts,
    evidence_snippets=_evidence_snippets(articles, facts),
    sources=sources,
    metadata={
        "representative_id": representative_id,
        "cluster_article_ids": cluster_article_ids or _item_ids(articles),
        "classification": classification,
        "created_at": datetime.now(UTC).isoformat(),
        # ▼ 신규 1줄 — 작업 S5 의 핵심 변경
        "trend_context": fetch_latest_trend_context(within_days=7),
    },
)
```

같은 패턴으로 `analysis_input_bundle_from_bundle()` (line 98-111) 도 `metadata={"collected_at": ..., "trend_context": fetch_latest_trend_context()}` 로 수정.

이렇게 하면:

1. `IntegrationAgent` 가 `AnalysisInputBundle` 만들 때 자동으로 `metadata["trend_context"]` 채워짐.
2. `analysis_flow_graph.py:631-653` 의 `_cluster_metadata(bundle, profile_context)` 가 그 `trend_context` 를 자동으로 cluster_metadata 에 포함.
3. `analyzer.py:227-238` 의 `_cluster_metadata_for_prompt(cluster_metadata)` 가 prompt 변수로 자동 추출.
4. `_PEER_NEWS_ANALYSIS_PROMPT` (analyzer.py:36-81) 가 `trend_context` 변수를 받아 자연어로 흘려보냄.

**fetch_latest_trend_context() 호출 빈도 + 캐시 정책**:

| 항목 | 값 |
|---|---|
| 호출 hot path | `analysis_input_bundle_from_articles()` (cluster 분석 시), `analysis_input_bundle_from_bundle()` |
| 호출 빈도 | 약 시간당 60~100회 (batch cluster 분석 시 burst 가능) |
| 캐시 정책 | **process-level TTL 60초** (in-memory dict, `threading.Lock`) |
| 캐시 키 | `within_days` (대부분 7) |
| 정합성 | trend 는 cronjob 으로 일 1회만 갱신 → 60초 stale 은 허용 범위 |
| Invalidation | `ITTrendAgent.generate()` 가 upsert 후 `invalidate_trend_context_cache()` 직접 호출 (cron path 즉시 반영) |
| Fallback (cache miss) | DB index `idx_global_industry_trends_date` 활용한 fast SELECT (~5ms) |
| 미래 옵션 | 다중 process / 다중 pod 환경에선 Redis cache (TTL 60s) 로 격상 |

---

## 6. 5-Phase Pipeline

각 Phase 는 `ITTrendAgent.generate()` 내부의 메서드. LLM 호출은 phase 3/4/5 만 (총 3회).

### Phase 1 — Snapshot (deterministic, no LLM)

```python
def _phase1_snapshot(self, raw_articles_global: list[dict]) -> list[GlobalSnapshot]:
    """글로벌 6사별 카드 카운트 + top_themes (top 5 keyword) + 최근 headline."""
    by_company = group_by(raw_articles_global, key="source_name")
    snapshots = []
    for company, rows in by_company.items():
        keywords = flatten_keywords(rows)
        snapshots.append(GlobalSnapshot(
            company_id=company.lower().replace("_official", "").replace("_newsroom", ""),
            card_count=len(rows),
            top_themes=top_k(keywords, k=5),
            headline_announcements=top_recent_titles(rows, k=3),
            source_marker=f"raw_articles WHERE source_name={company}",
        ))
    return snapshots
```

### Phase 2 — Trend Detection (deterministic, no LLM)

```python
def _phase2_trends(
    self,
    snapshots: list[GlobalSnapshot],
    research_rows: list[dict],   # spri/bcg
    previous: dict | None,
) -> list[TrendDetection]:
    """반복 keyword 추출 + frequency_delta_pct + intensity 분류."""
    keyword_counts = aggregate_keyword_mentions(snapshots, research_rows)
    previous_counts = previous.get("keyword_counts", {}) if previous else {}
    detections = []
    for kw, n in keyword_counts.items():
        if n < min_mention_count:
            continue
        delta = ((n - previous_counts.get(kw, 0)) / max(previous_counts.get(kw, 1), 1)) * 100
        intensity = classify_intensity(n, delta)  # weak < 5건 / moderate 5-15 / strong 15+
        detections.append(TrendDetection(
            theme=kw,
            frequency_delta_pct=delta,
            intensity=intensity,
            leading_companies=top_companies_by_keyword(snapshots, kw, k=3),
            evidence_card_ids=[],  # snapshot 은 raw_articles 기반 — card_id 없음
        ))
    return sorted(detections, key=lambda d: d.frequency_delta_pct, reverse=True)[:max_trend_count]
```

### Phase 3 — Peer Alignment (deterministic 점수 + LLM 1회 strategic_note)

**이게 사용자가 원하는 "AX/peer 동향이 글로벌 트렌드와 같은 결로 가는지" 비교 부분.**

> ⚠ **데이터 비대칭 처리**: 검증 결과 `card_news.peer_company_id='sk_ax'` 가 **0건**.
> 4 peer (samsung_sds 87, lg_cns 69, hyundai_autoever 23, posco_dx 14) 는 카드로,
> **SK AX 는 raw_articles 에서 직접** fetch 한다 (SK AX Site 155 + Newsroom 12 + dart 26 + ir_pdf 14).
> Helper 분기 처리.

```python
def _phase3_peer_alignment(
    self,
    detections: list[TrendDetection],
    peer_company_ids: list[str],
    window_days: int = 30,
) -> dict[str, list[PeerAlignment]]:
    """trend keyword × peer 매트릭스. 결정적 점수 + LLM 한 번 호출로 strategic_note."""
    result: dict[str, list[PeerAlignment]] = {}
    for det in detections:
        per_peer = []
        for peer_id in peer_company_ids:
            # SK AX 는 card_news 0건이라 raw_articles 에서 직접 fetch
            if peer_id == "sk_ax":
                peer_cards = fetch_sk_ax_raw_for_alignment(
                    window_days=window_days,
                    keyword=det.theme,
                )  # raw_articles WHERE source_name IN ('SK AX Site','SK AX Newsroom','dart','ir_pdf')
            else:
                peer_cards = fetch_peer_cards_for_alignment(
                    peer_id=peer_id,
                    window_days=window_days,
                    keyword=det.theme,
                )  # card_news WHERE peer_company_id=peer AND (keyword OR primary_keyword_category match)
            
            # 결정적 산식
            peer_n = len(peer_cards)
            global_n = sum(snapshot.theme_count(det.theme) for snapshot in snapshots)
            score = compute_alignment_score(peer_n, global_n, peer_cards_recency)
            alignment_type = classify_alignment(score, peer_n)
            #   aligned : score >= 0.6
            #   lagging : 0.2 <= score < 0.6
            #   diverging : -0.2 <= score < 0.2 (있긴 한데 방향이 다름)
            #   missing : score < -0.2 OR peer_n == 0
            
            recency_gap = days_since_latest(peer_cards) - days_since_latest_global(det.theme)
            
            per_peer.append({
                "peer_id": peer_id,
                "alignment_type": alignment_type,
                "alignment_score": round(score, 3),
                "peer_mention_count": peer_n,
                "global_mention_count": global_n,
                "recency_gap_days": recency_gap,
                "evidence_card_ids": [c["id"] for c in peer_cards[:5]],
                "strategic_note": "",  # LLM 으로 batch 1회 채움
            })
        result[det.theme] = per_peer
    
    # LLM 호출 1회 — 모든 (theme × peer) strategic_note 를 한 번에 채움 (batch)
    result = _llm_fill_strategic_notes(result, detections)
    return result
```

### Phase 4 — Impact Mapping (LLM 1회)

```python
def _phase4_impact(
    self,
    detections: list[TrendDetection],
    sk_ax_business_lines: list[str],
) -> list[SKAXImpactCell]:
    """trend × sk_ax_line 매트릭스. LLM 1회 호출."""
    prompt = build_impact_prompt(detections, sk_ax_business_lines, SK_AX_PROFILE)
    response = llm.invoke(prompt, response_format="json")
    return parse_impact_cells(response)
```

### Phase 5 — Forecast + Synthesis (LLM 1회)

```python
def _phase5_forecast_synthesis(
    self,
    detections: list[TrendDetection],
    alignment: dict,
    impact_matrix: list[SKAXImpactCell],
) -> tuple[list[GlobalForecast], str, str]:
    """horizon 별 narrative + final_one_liner + sk_ax_implication."""
    prompt = build_synthesis_prompt(detections, alignment, impact_matrix)
    response = llm.invoke(prompt, response_format="json")
    return (
        parse_forecasts(response),
        response["final_one_liner"],
        response["sk_ax_implication"],
    )
```

**LLM 호출 총 3회** (P3 strategic_notes batch 1회 + P4 impact 1회 + P5 synthesis 1회). 일 비용 ~₩600~1,000 추정 (gpt-4o, in 2k + out 1k × 3).

---

## 7. 저장 흐름 — `global_industry_trends` 직접 INSERT (V30 이후 변경)

> ⚠ **변경 사유**: 초기 가정은 axis-ai 표준 `@with_ledger_writeback` 데코레이터 사용이었으나,
> V30 마이그레이션 line 694 에서 `DROP TABLE IF EXISTS analysis_ledger;` 가 적용되어
> ledger 테이블이 더 이상 존재하지 않음. axis-ai 의 데코레이터 코드는 silent fail 상태.
> 따라서 **`global_industry_trends` 에 직접 INSERT** 한다. UNIQUE 제약은 V29 의
> `uq_global_industry_trends_daily_keyword` (idempotent upsert) + V29 의
> `uq_global_industry_trends_source_analysis_id` (per-batch audit trace) **두 가지 모두** 활용.

```python
import hashlib
import re

# global_industry_trends.source_analysis_id 는 VARCHAR(100) (DB 확인 완료).
# batch_id ("global-YYYYMMDD-HHMMSS") = 22자 + "-" + suffix → 합계 ≤ 100자 보장 필요.
# 따라서 suffix 는 **최대 77자**, hard cap 100자 적용.

_BATCH_ID_LEN = len("global-20260526-023012")   # 22
_MAX_SOURCE_ANALYSIS_ID = 100                    # V29 schema.sql:457
_MAX_SUFFIX_LEN = _MAX_SOURCE_ANALYSIS_ID - _BATCH_ID_LEN - 1   # 77

def _slugify(keyword: str, max_len: int = 40) -> str:
    """source_analysis_id 의 human-readable 부분.
    
    한글/영문/숫자만 남기고 공백을 dash 로. max_len 으로 cap.
    e.g.  "Agentic AI"  → "agentic-ai"
          "에이전트 AI"  → "에이전트-ai"
          "GPU 부족"     → "gpu-부족"
    """
    text = (keyword or "").strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^0-9a-z가-힣\-]+", "", text)
    return (text[:max_len] or "unknown").rstrip("-")

def _make_source_analysis_id(batch_id: str, idx: int, keyword: str) -> str:
    """`global-YYYYMMDD-HHMMSS-NNN-<slug>` 형식.
    
    - idx (3자리 zero-pad) — 같은 batch 내 keyword 순번. slug 충돌 시 unique 보장.
    - short slug — human-readable. max 40자 cap.
    - hard 100자 cap — 한글 character count 가 예상보다 크거나 idx 가 3자리 초과 시 안전.
    
    충돌 방어: idx 가 unique 보장하므로 slug 가 비거나 같아도 OK.
    예: "global-20260526-023012-001-agentic-ai"  (37자)
        "global-20260526-023012-002-에이전트-ai" (한글 포함, 35자)
    """
    suffix = f"{idx:03d}-{_slugify(keyword, max_len=40)}"
    candidate = f"{batch_id}-{suffix}"
    if len(candidate) <= _MAX_SOURCE_ANALYSIS_ID:
        return candidate
    # safety net: keyword 가 비정상적으로 길면 sha1 8자로 fallback
    short_hash = hashlib.sha1(keyword.encode("utf-8")).hexdigest()[:8]
    return f"{batch_id}-{idx:03d}-{short_hash}"[:_MAX_SOURCE_ANALYSIS_ID]
```

```python
class ITTrendAgent:
    # 데코레이터 없음 — 직접 db/article_store.py 의 upsert 함수 호출
    def generate(self, trend_input: ITTrendInput) -> dict[str, Any]:
        snapshots = self._phase1_snapshot(...)
        detections = self._phase2_trends(snapshots, ..., trend_input.previous_trend_context)
        alignment = self._phase3_peer_alignment(detections, ...)
        impact = self._phase4_impact(detections, ...)
        forecasts, final, skax_impl = self._phase5_forecast_synthesis(detections, alignment, impact)
        
        # 1 row per trend keyword. 직접 global_industry_trends 에 upsert.
        # ⚠ source_analysis_id 는 **keyword 별로 유니크**하게 만들어야 한다
        #   (V29 `uq_global_industry_trends_source_analysis_id` partial UNIQUE 제약).
        #   같은 batch 의 8 keyword 가 동일 analysis_id 를 공유하면 2번째 row 부터 conflict.
        batch_id = f"global-{utc_now_compact()}"   # e.g. global-20260526-023012
        rows = self._build_persistence_rows(
            detections, alignment, impact, forecasts, final, skax_impl,
            batch_id=batch_id,   # 각 row 는 batch_id 안에서 keyword slug 추가
        )
        upsert_global_industry_trends(rows)   # db/article_store.py 에 신규 함수 추가
        
        return {
            "analysis_id": batch_id,
            "analysis_type": "global",
            "rows": [
                # idx 는 enumerate 로 부여 (batch 내 keyword 순번)
                # for idx, det in enumerate(detections, start=1): ...
                {
                    # ⚠ per-keyword UNIQUE — batch_id + idx + keyword slug 조합.
                    #   - V29 partial UNIQUE: uq_..._source_analysis_id WHERE NOT NULL.
                    #   - V29 컬럼 길이: VARCHAR(100). _make_source_analysis_id 가 hard cap.
                    #   - idx 가 slug 충돌 방어 (한글 keyword 가 slugify 후 같아져도 unique).
                    "source_analysis_id": _make_source_analysis_id(batch_id, idx, det.theme),
                    "trend_date": utc_today(),
                    "industry": classify_industry(det.theme),
                    "region": "global",
                    "keyword": det.theme,
                    "keyword_category": classify_category(det.theme),
                    "title": final_per_keyword[det.theme],
                    "summary": summary_per_keyword[det.theme],
                    "mention_count": global_mention_per_keyword[det.theme],
                    "impact_score": impact_score_per_keyword[det.theme],
                    "confidence": confidence_per_keyword[det.theme],
                    "related_peer_ids": [p["peer_id"] for p in alignment[det.theme] if p["alignment_type"] == "aligned"],
                    "related_card_ids": [cid for p in alignment[det.theme] for cid in p["evidence_card_ids"]],
                    "source_raw_article_ids": evidence_raw_ids_per_keyword[det.theme],
                    "sk_ax_implication": skax_impl_per_keyword[det.theme],
                    "payload": {
                        "peer_alignment": alignment[det.theme],
                        "impact_matrix": [c.dict() for c in impact if c.trend_theme == det.theme],
                        "forecasts": [f.dict() for f in forecasts],
                        "leading_companies": det.leading_companies,
                        "intensity": det.intensity,
                        "frequency_delta_pct": det.frequency_delta_pct,
                        "git_sha": current_git_sha(),
                        "prompt_version": "global-trends-v1.0",
                    },
                }
                for det in detections
            ],
            "trend_context": trend_context_dto.to_dict(),  # TrendContext shape, AnalysisAgent 가 사용
            "snapshots": [s.dict() for s in snapshots],
            "trend_detections": [d.dict() for d in detections],
            "impact_matrix": [c.dict() for c in impact],
            "forecasts": [f.dict() for f in forecasts],
            "final_one_liner": final,
            "sk_ax_implication": skax_impl,
            "confidence": avg_confidence,
            "validation": {"pass": True, "phase_completion": [1, 2, 3, 4, 5]},
        }
```

### `db/article_store.py` 에 신규 upsert 함수 (필수 추가)

```python
_GLOBAL_TREND_UPSERT_SQL = text("""
INSERT INTO global_industry_trends (
    source_analysis_id, trend_date, industry, region, keyword, keyword_category,
    title, summary, mention_count, impact_score, confidence,
    related_peer_ids, related_card_ids, source_raw_article_ids,
    sk_ax_implication, payload
) VALUES (
    :source_analysis_id, :trend_date, :industry, :region, :keyword, :keyword_category,
    :title, :summary, :mention_count, :impact_score, :confidence,
    :related_peer_ids, :related_card_ids, :source_raw_article_ids,
    :sk_ax_implication, CAST(:payload AS JSONB)
)
ON CONFLICT (trend_date, industry, region, keyword)
DO UPDATE SET
    title = EXCLUDED.title,
    summary = EXCLUDED.summary,
    mention_count = EXCLUDED.mention_count,
    impact_score = EXCLUDED.impact_score,
    confidence = EXCLUDED.confidence,
    related_peer_ids = EXCLUDED.related_peer_ids,
    related_card_ids = EXCLUDED.related_card_ids,
    source_raw_article_ids = EXCLUDED.source_raw_article_ids,
    sk_ax_implication = EXCLUDED.sk_ax_implication,
    payload = EXCLUDED.payload,
    updated_at = NOW()
""")

def upsert_global_industry_trends(rows: list[dict]) -> int:
    """trend_date × industry × region × keyword UNIQUE 로 idempotent upsert.
    
    같은 날 cronjob 이 두 번 돌아도 안전. source_analysis_id 는 첫 INSERT 시점 보존.
    """
    if not rows:
        return 0
    with SessionLocal() as db:
        db.execute(_GLOBAL_TREND_UPSERT_SQL, rows)
        db.commit()
    return len(rows)
```

**핵심**: V29 의 `uq_global_industry_trends_daily_keyword UNIQUE(trend_date, industry, region, keyword)` 제약 활용. cronjob 재시도 / on-demand 호출 모두 idempotent.

> ⚠ **두 UNIQUE 제약의 상호작용** — `global_industry_trends` 에는 **두 개의 UNIQUE 제약** 이 존재:
>
> 1. `uq_global_industry_trends_daily_keyword` — `UNIQUE(trend_date, industry, region, keyword)` (전체)
> 2. `uq_global_industry_trends_source_analysis_id` — `UNIQUE(source_analysis_id) WHERE source_analysis_id IS NOT NULL` (partial)
>
> `ON CONFLICT (trend_date, industry, region, keyword) DO UPDATE` 만으로는 **(2) 의 conflict 가 처리되지 않음**. 따라서:
>
> - 같은 batch 의 8 keyword 가 동일 `source_analysis_id` 를 공유하면 **2번째 row INSERT 부터 conflict**.
> - **추가 제약**: `source_analysis_id` 는 V29 schema 에서 `VARCHAR(100)` (확인됨).
>   `batch_id` (22자) + `-` + slug 만으론 slug 가 80자 가까이면 **101~103자로 길이 초과 → INSERT 실패**.
> - 해결책 A (채택): `source_analysis_id = f"{batch_id}-{idx:03d}-{slug(keyword)[:40]}"` 형식.
>   - `idx` (3자리 zero-pad, batch 내 keyword 순번) — slug 충돌 시에도 unique 보장.
>   - slug 40자 cap + 전체 100자 hard cap (safety net: keyword 가 비정상 길면 sha1 8자 fallback).
>   - audit/replay 용이 (idx + slug 로 사람이 읽기 가능).
> - 해결책 B (대안, 미채택): `source_analysis_id = NULL` 로 두고 (1) UNIQUE 만 의존. 단 batch trace 가 어려워짐.
>
> 본 설계는 `_make_source_analysis_id(batch_id, idx, keyword)` helper 로 일원화 (§7 하단 코드).

⚠ V30 의 자동 복제 SQL (lines 474-509) 은 **마이그레이션 1회 실행 후 사실상 dead** — `analysis_ledger` 가 같은 V30 line 694 에서 DROP 되었기 때문. 향후 ledger 부활 결정시 양쪽 path 가 충돌하지 않게 `source_analysis_id` 네이밍 컨벤션을 분리 (`global-<timestamp>-<keyword-slug>`).

---

## 8. 다른 Agent 와의 통합 — wire-up

### 8.1 AnalysisAgent (StrategicAnalyzer) — inject 지점 명시

| 컴포넌트 | line | 동작 |
|---|---|---|
| (신규) `db/article_store.py` `fetch_latest_trend_context()` | 신규 | `global_industry_trends` 최근 N일 → TrendContext shape dict |
| (수정 1줄) `src/agents/integration_agent.py:143-160` | 154 | `metadata={...}` dict 에 `"trend_context": fetch_latest_trend_context(7)` 추가 |
| (수정 1줄) `src/agents/integration_agent.py:98-111` | 110 | 동일하게 metadata dict 에 `"trend_context"` 추가 |
| (변경 없음) `src/pipeline/analysis_flow_graph.py:631-653` `_cluster_metadata` | 635 | 이미 `bundle.metadata.get("trend_context")` 를 읽음 |
| (변경 없음) `src/analysis/analyzer.py:227-238` `_cluster_metadata_for_prompt` | 238 | 이미 prompt 변수로 추출 |
| (변경 없음) `src/analysis/analyzer.py:36-81` `_PEER_NEWS_ANALYSIS_PROMPT` | 41/60 | 이미 prompt 안에 `trend_context` 변수 슬롯 |

→ **신규 함수 1개 + 기존 파일 1개에 2줄 수정** 이 wire-up 의 전부. 변경 risk surface 매우 작음.

### 8.2 ImplicationAgent — 자연 흐름

`AnalysisResult.strategic_meaning` 이 trend_context 로 풍부해지므로 ImplicationResult 의 `skax_implication.opportunities/threats` 도 자동으로 trend-aware 가 됨. **변경 코드 0**.

### 8.3 CardNewsAgent — 자연 흐름

`ImplicationResult` 가 풍부해지므로 카드 implication 텍스트도 trend-aware. **변경 코드 0**.

### 8.4 MixerAgent — 자연 흐름

`card_news.evidence_payload` 안에 ImplicationResult 가 들어가는데, 그게 trend-aware 가 됨. Mixer 가 cross-card 분석할 때도 자연 흐름. **변경 코드 0**.

### 8.5 ReportAgent / BriefingService — 추가 hook

```python
# axis-backend BriefingService 또는 axis-ai ReportAgent
# 일일 브리핑 이메일 body 에 trend top 3 추가:

global_trends_recent = jdbcTemplate.query("""
    SELECT keyword, title, sk_ax_implication, payload->'peer_alignment' AS alignment
    FROM global_industry_trends
    WHERE trend_date = CURRENT_DATE
    ORDER BY impact_score DESC LIMIT 3
""", ...);
```

(선택 작업)

---

## 9. API + Trigger

### 9.1 REST endpoint (axis-ai)

```
POST /global/trends/run
  body: GlobalTrendsRequest
  resp: GlobalTrendsResponse
```

schema 는 `src/api/global_trends_schemas.py` ( ` 2` 정리 후 ) 그대로. router 등록만 추가:

```python
# src/api/router.py
from src.agents.it_trend_agent import ITTrendAgent
from src.api.global_trends_schemas import GlobalTrendsRequest, GlobalTrendsResponse

@app.post("/global/trends/run", response_model=GlobalTrendsResponse)
async def run_global_trends(req: GlobalTrendsRequest):
    agent = ITTrendAgent()
    trend_input = agent.build_input(
        items=fetch_global_trend_inputs(window_days=req.window_days),
        period=f"last_{req.window_days}d",
        metadata=req.model_dump(),
    )
    return agent.generate(trend_input)
```

### 9.2 Read endpoint (axis-backend, 선택)

```
GET /api/global-trends?from=YYYY-MM-DD&to=YYYY-MM-DD&limit=10
```

`global_industry_trends` 단순 SELECT (V30 view 가 채워주므로 즉시 read 가능).

### 9.3 Cronjob

`axis-infra/k8s/base/cronjob-global-trend.yaml` (신규, in-place 작성):

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: axis-cron-global-trend
  namespace: skala3-finalproj-class3-team13
spec:
  # KST 02:30 = UTC 17:30 (전날). axis-pg-dump (UTC 18:00) 보다 30분 일찍 끝나야 backup 일관성.
  schedule: "30 17 * * *"
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      backoffLimit: 1
      activeDeadlineSeconds: 1800   # 30분 timeout
      template:
        spec:
          containers:
          - name: curl
            image: curlimages/curl:latest
            command:
              - sh
              - -c
              - |
                curl -fsS -X POST http://axis-ai:8000/global/trends/run \
                  -H "Content-Type: application/json" \
                  -d '{"window_days": 7}'
```

---

## 10. Validation Quality Gate

기존 `ValidationReport` ([analysis/models.py:685-732](../../src/analysis/models.py)) 패턴 차용. ITTrendAgent 산출물에 적용할 hard / soft gate:

| Gate | Type | 조건 | 실패 시 |
|---|---|---|---|
| Phase 1 snapshot empty | **hard** | `sum(card_count) == 0` | `global_industry_trends` upsert skip + alert (slack/langfuse warning) |
| Phase 2 detections empty | **hard** | `len(detections) == 0` | `global_industry_trends` upsert skip + alert |
| Per-keyword evidence | **hard** | `mention_count < min_mention_count` | 해당 keyword drop |
| Peer alignment all missing | **soft** | 4 peer 모두 `alignment_type=missing` | warning + 카드 우선 추천 |
| LLM confidence | **soft** | `confidence < 0.5` | warning + reviewer flag |
| Recency | **soft** | `max(card.created_at) > 14d ago` | warning |

---

## 11. 비용 + 모니터링

| 항목 | 값 |
|---|---|
| LLM 호출 / 일 | 3회 (P3 strategic_notes batch + P4 impact + P5 synthesis) |
| Token / 호출 | ~3,000 (in 2k + out 1k) |
| 일일 LLM 비용 | ~₩600~1,000 |
| keyword 출력 / 일 | ≤ 8 (max_trend_count) |
| ledger row insert / 일 | ≤ 8 |
| global_industry_trends row growth | ~8/일 = 240/월 |
| Langfuse trace | 매 phase 별 1 trace + 통합 trace |
| 핵심 KPI | (a) `confidence >= 0.6` 비율 ≥ 80%, (b) peer_alignment 의 `evidence_card_ids` 비어있는 row < 20%, (c) Phase 1 빈 응답 0건 |

---

## 12. 구현 단계별 작업 (가벼움 순)

| Step | 작업 | 소요 |
|---|---|---|
| **S1** | `api/global_trends_schemas.py` **신규 작성**. source-of-truth = `axis-infra/api/openapi.yaml` 의 `GlobalTrendsRequest` / `GlobalTrendsResult` components (이미 정의됨, git tracked). Finder 중복본 (`global_trends_schemas 2.py`) 은 untracked 라 의존 금지. `router.py:11-13` 패턴으로 import 추가 후 `/global/trends/run` endpoint 등록 (openapi.yaml line 2933 의 operationId `runGlobalTrends`). | 1.5h |
| **S2** | `db/article_store.py` 에 reader 2개 추가 (`fetch_global_trend_inputs`, `fetch_peer_cards_for_alignment`) | 1~2h |
| **S3** | `ITTrendAgent.generate()` Phase 1~5 구현 (LLM prompt 3개 작성) | 6~8h |
| **S4** | `ITTrendAgent.generate()` 끝에서 `upsert_global_industry_trends(rows)` 직접 호출 (⚠ `@with_ledger_writeback` 데코레이터는 **적용하지 않는다** — V30 line 694 에서 `analysis_ledger` DROP) | 30분 |
| **S5** | (a) `db/article_store.py` 에 `fetch_latest_trend_context()` + `invalidate_trend_context_cache()` 신규 추가 (**process-level TTL 60초 캐시 포함** — hot path, `threading.Lock` 사용), (b) `src/agents/integration_agent.py` 의 metadata dict 2곳 (L98-111, L143-160) 에 `"trend_context": fetch_latest_trend_context(7)` 1줄씩 추가, (c) `ITTrendAgent.generate()` 끝에 `invalidate_trend_context_cache()` 호출 추가 (cron path 즉시 반영). `analysis_flow_graph` 자체는 이미 read hook 보유라 수정 불필요. | 1.5h |
| **S6** | `api/router.py` 에 `POST /global/trends/run` 등록 | 30분 |
| **S7** | `axis-infra/k8s/base/cronjob-global-trend.yaml` 추가 + kustomize overlay 연결 | 30분 |
| **S8** | 검증: `kubectl exec` 로 한 번 호출 → `global_industry_trends` row 생성 확인 → `payload.peer_alignment` 5 peer 모두 들어가는지 확인 | 1h |
| **S9** | (선택) backend `GET /api/global-trends` read API + frontend 표시 | 2~3h |

**합계 약 12~16h** (1명 기준 1.5~2일).

신규 파일 단 2개 (`db/article_store.py` 안 함수만 추가 / cronjob yaml 1개), **새 테이블·새 agent·새 model class 모두 0**.

---

## 13. 위험 + 완화

| 위험 | 영향 | 완화 |
|---|---|---|
| **DART OOM 패치 전에 시작하면 axis-ai 또 OOM** | 데이터 비어버림 | DART 패치 (별도 진행 중) **후** 착수 |
| **글로벌 6사 newsroom 카드 부족** | snapshots 빈약 | (a) `naver_research` 글로벌 부분 보완 (b) min_mention_count 동적 조정 |
| **`previous_trend_context` 부재 (cold start)** | frequency_delta_pct = 0 | 첫 7일은 delta 무시, 절대값만 사용 |
| **LLM JSON parse 실패** | row 0건 | Mixer 의 `_parse_and_validate` 패턴 차용 (retry 1회 → empty graceful) |
| **peer_alignment 의 LLM strategic_note batch 가 너무 큼** | token > 4k | keyword × peer batch 를 4개 단위로 split |
| **`global_industry_trends` UNIQUE(source_analysis_id) 위반** + VARCHAR(100) 길이 초과 (analysis_ledger 아님 — V30 DROP) | INSERT skip | `_make_source_analysis_id(batch_id, idx, keyword)` helper — `idx:03d` 순번으로 unique 보장 + slug 40자 cap + 100자 hard cap + sha1 8자 fallback (§7 하단 코드) |
| **`naver_news` noise** | alignment 오탐 | `importance_score >= 0.5` filter 적용 (이미 카드뉴스 단계에서 적용됨) |

---

## 14. 결론

| 질문 | 답 |
|---|---|
| 글로벌 IT 트렌드 추출 agent 가 이미 있나? | **있다** — `ITTrendAgent`. 단 `generate()` 가 빈 스켈레톤. |
| AX/peer 동향이 글로벌 트렌드와 같은 결로 가는지 비교 기능 있나? | **부분만**. schema (`SKAXImpactCell`, `TrendDetection.leading_companies`) 와 V30 의 `related_peer_ids` 컬럼은 있지만 채우는 로직 0. |
| 새 agent 필요한가? | **아니오.** `ITTrendAgent` 안에 Phase 3 (peer_alignment) 통합. |
| 새 DB 테이블 필요한가? | **아니오.** `global_industry_trends.payload` JSONB + `related_peer_ids` 직접 컬럼. |
| 다른 agent 와 충돌? | 없음 — `analyzer.py` 가 이미 `trend_context` hook 보유. wire-up 만. |
| 가장 작은 변경 폭? | Step S1~S8 (12~16h). |

---

## 15. 다른 agent / 스키마와의 cross-check 검증 결과

설계 초안 작성 직후 **실제 DB / 코드 / 마이그레이션 대조 검증** 결과:

### 15.1 통과 (no change)

| 항목 | 검증 내용 | 결과 |
|---|---|---|
| `peer_companies` 테이블 | 5 peer (sk_ax + 4) 존재 확인 | ✅ |
| 글로벌 6사 raw_articles 30일 | amazon 476 / meta 226 / nvidia 219 / msft 129 / google 45 / apple 14 = **1,109건** | ✅ Phase 1/2 input 충분 |
| SPRi/BCG raw_articles | spri 40 + bcg 53 = 93건 | ✅ Phase 2 보강 가능 |
| 4 peer card_news 30일 (samsung_sds 87, lg_cns 69, hyundai_autoever 23, posco_dx 14) | 모두 ≥10건 | ✅ Phase 3 카드 매칭 가능 |
| `card_news.primary_keyword_category` | ax 92 / infra 44 / security 15 / deal 5 / other 23 분포 | ✅ keyword 매칭 가능 |
| `analyzer.py` 의 `trend_context` hook | line 41/60/238/242 에서 prompt 변수로 받음 | ✅ wire-up 만 |
| `analysis_flow_graph.py` 의 `bundle.metadata["trend_context"]` | line 635 에서 읽기 | ✅ wire-up 만 |
| `global_industry_trends` 테이블 (V29) | 컬럼 / 인덱스 / UNIQUE 모두 설계와 일치 | ✅ |
| `MixerAnalysisAgent` 의 carding/comparison 패턴 | LLM 호출 / repair / radar 산식 | ✅ Phase 3/5 prompt 의 시발점으로 사용 |
| Flyway migration 적용 상태 | V29 / V30 모두 success (`flyway_schema_history`) | ✅ |

### 15.2 위험 발견 (설계 수정 완료)

#### 자체 검증 (1차)

| # | 위험 | 검증 증거 | 설계 반영 |
|---|---|---|---|
| **R1** | `analysis_ledger` 테이블 존재 안 함 | V30 line 694 `DROP TABLE IF EXISTS analysis_ledger;` + DB 에서 `SELECT FROM analysis_ledger` 실패 | §7 을 **직접 INSERT** 로 변경. `db/article_store.py` 에 `upsert_global_industry_trends()` 신규 추가. |
| **R2** | `card_news.peer_company_id='sk_ax'` 가 0건 | `SELECT COUNT(*) FROM card_news WHERE peer_company_id='sk_ax'` = 0. SK AX raw 는 풍부 (Site 155 + Newsroom 12 + dart 26 + ir_pdf 14) | §6 Phase 3 코드에 SK AX 분기 추가 (`fetch_sk_ax_raw_for_alignment`). §3 의 reader 함수 목록도 갱신. |
| **R3** | `axis-ai/src/middleware/analysis_ledger.py` 가 dead code | 코드는 ledger INSERT 시도하지만 silent fail | 본 설계 범위 밖. 추후 별도 cleanup PR 권장 — `MixerAnalysisAgent.@with_ledger_writeback` 등 다른 사용처 모두 동일 영향. |

#### 외부 LLM Review (2차, 2026-05-26)

| # | 위험 | 검증 증거 | 설계 반영 |
|---|---|---|---|
| **R4** (H1) | 같은 batch 의 8 keyword 가 동일 `source_analysis_id` 를 공유하면 V29 의 `uq_global_industry_trends_source_analysis_id` partial UNIQUE 와 충돌 — 2번째 INSERT 부터 fail | `\d global_industry_trends` 로 `CREATE UNIQUE INDEX ... WHERE source_analysis_id IS NOT NULL` 확인 | §7 의 `source_analysis_id` 를 **`f"{batch_id}-{slug(keyword)}"` per-keyword** 로 변경. §7 박스에 두 UNIQUE 제약의 상호작용 명시. |
| **R5** (H2) | §12 S4 가 `@with_ledger_writeback` 데코레이터 적용을 지시 — §7 의 "직접 INSERT" 정책과 모순 | 문서 자체 cross-read | §12 S4 의 작업 항목을 "직접 `upsert_global_industry_trends()` 호출, **데코레이터는 적용하지 않는다**" 로 변경. |
| **R6** (M1) | `build_input()` → `_split_trend_inputs()` 가 글로벌 6사 raw 를 unsupported 로 drop. 실제 raw_articles.source_type='official' 인데 상수는 `{global_newsroom, company_newsroom}` 만 통과 | [it_trend_agent.py:21,145,161](../../src/agents/it_trend_agent.py) + DB 의 source_type 분포 | §4.1 신설 — 3 옵션 비교 후 **옵션 B (fetcher 가 source_type 정규화)** 채택. `fetch_global_trend_inputs()` 코드 명시. §3.1 의 `_GLOBAL_NEWSROOM_SOURCE_TYPES` 줄에 ⚠ 표시. |
| **R7** (M2) | `api/global_trends_schemas.py` 정본 자체가 없음 — Finder 중복본만 있고 router.py 미등록. 다른 schema 는 정본+중복본 모두 있는데 global_trends 만 정본 누락 | `ls src/api/global_trends*` + `head -25 router.py` | §3.1, §3.4, §12 S1 모두 "rename" → "신규 정본 생성 + router 등록" 으로 정정. |
| **R8** (M3) | trend_context wire-up 의 구체적 inject 지점이 추상적 ("analysis_flow_graph 에서 채우면 됨") — 실제로 bundle 을 생성하는 곳 (integration_agent.py:143-160) 의 metadata dict 가 미명시 | `grep AnalysisInputBundle\(` → integration_agent.py 의 두 build 함수 발견 | §5.2 + §8.1 + §12 S5 모두 **integration_agent.py L98-111 + L143-160 의 metadata dict 에 1줄씩 추가** 로 구체화. fetch_latest_trend_context() 코드 명시. |

#### 외부 LLM Review (3차, 2026-05-26)

| # | 위험 | 검증 증거 | 설계 반영 |
|---|---|---|---|
| **R9** (H1) | §0 TL;DR / §1 메타 / §2 책임 DO 7번 / §3 코드 주석 / line 186 표현 등 **상단 5곳이 여전히 "analysis_ledger → 자동 복제" 라는 dead path 를 정답처럼 기술** — 구현자가 상단만 보고 따라가면 silent fail 코드 작성 | 본 문서 자체 cross-read | 5곳 모두 "global_industry_trends 직접 upsert" 로 정정. §15.4 의 일관성 체크리스트도 다시 통과. |
| **R10** (H2) | `_make_source_analysis_id` 의 batch_id (22자) + `-` + slug 80자 = **103자** → `VARCHAR(100)` 제약 초과 → INSERT 실패. slug 충돌 (한글 slugify 후 같아질 수 있음) 도 미해결 | DB 의 `character_maximum_length=100` 확인 + 길이 계산 | `_make_source_analysis_id(batch_id, idx, keyword)` 으로 변경 — **`idx:03d` (batch 내 순번)** 으로 unique 보장 + slug 40자 cap + 100자 hard cap + sha1 8자 fallback (safety net). |
| **R11** (M1) | `global_trends_schemas 2.py` 는 macOS Finder 중복본이고 **git untracked** (`?? src/api/global_trends_schemas 2.py`) — 다른 팀원 clone 에는 아예 없음. S1 의 "복제" 가 실제로는 불가능 | `git ls-files` + `git status --short` 로 untracked 확인 | source-of-truth 를 **`axis-infra/api/openapi.yaml` 의 `GlobalTrendsRequest` / `GlobalTrendsResult` components** (git tracked, line 2933/6846/6874) 로 변경. §3.1 + §3.4 + §12 S1 모두 정정. |
| **R12** (M2) | `fetch_latest_trend_context()` 가 `integration_agent.py` 의 cluster 분석 hot path 에서 매번 DB SELECT — burst 시 같은 SELECT 반복 | 본 design 의 §5.2 inject 지점이 hot 함을 자체 인정 | reader 에 **process-level TTL 60초 캐시** 추가 (in-memory dict + `threading.Lock`). `invalidate_trend_context_cache()` helper 도 추가해서 ITTrendAgent.generate() 가 upsert 후 직접 invalidate (cron path 즉시 반영). §12 S5 작업 항목에도 cache 포함 명시. |
| **R13** (Low) | §10 Quality Gate 표의 failure action 이 아직 `"ledger 저장 skip, alert"` — operational runbook 성격이라 혼동 유발 | 본 문서 자체 cross-read | `"global_industry_trends upsert skip + alert (slack/langfuse warning)"` 로 정정. |

### 15.3 충돌 없음 — 통합 검증

| 다른 agent / 컴포넌트 | 본 설계의 영향 |
|---|---|
| `MixerAnalysisAgent` | 영향 없음. `card_news.evidence_payload` 가 풍부해지면 mixer 도 자동으로 trend-aware. |
| `InsightAgent` (insight_cascade) | 영향 없음. card 단위로 작동. |
| `ImplicationAgent` | 자연 흐름 — `AnalysisResult.strategic_meaning` 이 trend_context 로 풍부해지면 `SkaxImplication.opportunities/threats` 도 자동 풍부화. |
| `CardNewsAgent` | 자연 흐름 — implication 텍스트가 trend-aware. |
| `ReportAgent` / `BriefingService` | 선택적 — 일일 브리핑에 trend top 3 표시 (추후 작업). |
| `PeerComparisonAgent` (design 만 존재) | **별개 agent**. peer 1명 ↔ SK AX 양자 비교 (5-strategy label). 본 설계 (N peer 가 동일 trend 라인 따라가는지의 alignment) 와는 차원이 다름. 동시 운영 가능. |
| `KeywordGraphAgent` | 영향 없음. |
| `ChatbotAgent` | 자연 흐름 — global_industry_trends 가 chat context 의 추가 source 가 될 수 있음 (추후 RAG 확장). |
| `analysis_flow_graph` 의 Track A (news/dart/ir 분석) | 영향 없음. `bundle.metadata["trend_context"]` 가 채워질 뿐. 기존 흐름 유지. |

### 15.4 비교 — 본 설계 vs 기존 `peer-comparison.md`

| 측면 | `PeerComparisonAgent` (design only) | 본 설계 (`ITTrendAgent.Phase 3`) |
|---|---|---|
| 비교 단위 | peer 1명 vs SK AX 양자 | N peer 가 동일 global trend 라인을 어떻게 따르는지 |
| Trigger | User Peer+ 페이지 select | 자동 cronjob + on-demand |
| 출력 | `strategy_label` (5종 중 1) + differentiators + strengths/weaknesses | per-trend × per-peer alignment matrix |
| 캐시 | peer_id × week (7일 TTL) | trend_date × keyword UNIQUE |
| 호출 빈도 | 사용자 클릭 시 | 일 1회 + on-demand |
| **상호 보완** | 본 설계가 만든 trend keyword 가 `PeerComparisonAgent` 의 `focus_sector` / context 로 흘러갈 수 있음 | 본 설계의 `payload.peer_alignment` 에서 lagging/missing 인 peer 가 발견되면 → `PeerComparisonAgent` 로 drill-down |

---

## 16. 발견된 추가 위험 (R1) — `analysis_ledger` DROP 의 후폭풍

V30 line 694 가 `analysis_ledger` 를 DROP 함. 그러나 axis-ai 코드는 여전히 ledger 사용 가정:

| 영향 받는 코드 | 현재 상태 | 영향 |
|---|---|---|
| `src/middleware/analysis_ledger.py` 의 `AnalysisLedger.insert()` | silent fail (테이블 없음) | mixer/insight/global/briefing 모두 INSERT 시도하다가 fail-soft warning log 만 남음 |
| `@with_ledger_writeback` 데코레이터 (mixer / insight / 등) | dead code | INSERT 실패해도 분석 결과 자체는 반환됨 — 기능적으론 OK 인데 로그가 더러워짐 |
| `ContextPackBuilder` 의 carry-over | ledger 가 없으니 carry-over 0 건 | 분석 간 누적 context 가 깨짐 |
| 본 설계의 Phase 2 `previous_trend_context` | ledger 가 없으니 직전 주기 ledger lookup 불가 | **alternative**: `global_industry_trends` 자체에서 `trend_date < CURRENT_DATE` 의 가장 최근 row 를 직전으로 사용 |

**본 설계는 위 영향을 회피** — 직접 `global_industry_trends` 에 INSERT 하고, `previous_trend_context` 도 `global_industry_trends` self-read 로 처리.

**ledger 부활 vs 영구 폐기 결정** — 본 설계 범위 밖. 팀 결정 필요 (별도 사안).

---

## 17. 발견된 추가 위험 (R2) — `card_news` 에 `sk_ax` 가 0건

검증 결과:

```text
SELECT peer_company_id, COUNT(*) FROM card_news
WHERE peer_company_id IS NOT NULL GROUP BY 1;
 peer_company_id  | n_total | n_30d 
------------------+---------+-------
 samsung_sds      |      87 |    87
 lg_cns           |      69 |    69
 hyundai_autoever |      23 |    23
 posco_dx         |      14 |    14
(4 rows)   -- sk_ax 없음

SELECT company, peer_company_id FROM card_news 
WHERE company IN ('sk_ax','SK AX') OR peer_company_id='sk_ax';
 (0 rows)
```

**원인 추정**: 카드뉴스 분류 파이프라인이 SK AX 자사 raw_articles 를 `company=sk_ax` 카드로 만들지 않고 있다. SK AX Site / Newsroom 은 raw_articles 에 있지만 (각 155 / 12 건), `card_composer` 가 자사 카드를 별도 처리하지 않는 듯.

**본 설계의 회피**:

- Phase 3 SK AX alignment 는 **raw_articles 에서 직접 fetch**.
- helper: `fetch_sk_ax_raw_for_alignment(window_days, keyword)` 신규 추가 (`db/article_store.py`).
- SK AX evidence 는 `evidence_card_ids` 가 아닌 `evidence_raw_article_ids` 별도 필드로 alignment payload 에 저장.

```jsonc
{
  "peer_id": "sk_ax",
  "alignment_type": "lagging",
  "alignment_score": 0.35,
  "peer_mention_count": 6,    // raw_articles 매칭 건수
  "global_mention_count": 18,
  "recency_gap_days": 14,
  "evidence_card_ids": [],                     // 카드 없음
  "evidence_raw_article_ids": [12345, 67890],  // raw_articles 직접
  "evidence_source": "raw_articles",           // 분기 표시
  "strategic_note": "..."
}
```

**별도 후속 작업 권장** — 카드뉴스 파이프라인이 SK AX 자사 카드를 만들도록 수정. 이건 본 설계 범위 밖이지만 발견된 시스템 위험이므로 팀에 공유 필요.

---

## 18. 최종 체크리스트

| # | 항목 | 상태 |
|---|---|---|
| ✅ | 글로벌 IT 트렌드 추출 agent 가 있는지 확인 | `ITTrendAgent` 존재. `generate()` 스켈레톤. |
| ✅ | DB 스키마 변경 필요성 재검토 | **변경 불필요** — `global_industry_trends.payload` JSONB + V29 컬럼으로 충분 |
| ✅ | 기존 코드 재활용 매핑 | §3.1 에서 10+ 자산 재활용 명시 |
| ✅ | 기존 모델 (TrendContext / IntegratedIssue / AnalysisResult / ImplicationResult) 재활용 | §3.1 + §5.2 wire-up |
| ✅ | 5-Phase pipeline 정의 | §6 |
| ✅ | 저장 흐름 | §7 (직접 INSERT, R1 반영) |
| ✅ | 다른 agent 와의 통합 | §8 + §15.3 |
| ✅ | API + cron | §9 |
| ✅ | Validation gate | §10 |
| ✅ | 비용 + 모니터링 | §11 |
| ✅ | 단계별 작업 | §12 |
| ✅ | 위험 + 완화 | §13 |
| ✅ | DB / 코드 / migration cross-check 검증 완료 | §15.1 (10 항목 pass) |
| ✅ | 1차 자체 검증 위험 (R1~R3) 모두 설계에 반영 | §15.2 자체 검증 부분 (§3 / §6 / §7 수정 완료) |
| ✅ | 2차 외부 LLM review 위험 (R4~R8) 모두 설계에 반영 | §15.2 외부 review 부분 (§4.1 / §5.2 / §7 / §8.1 / §12 수정 완료) |
| ✅ | 충돌 없는지 — 다른 9개 agent 영향 분석 | §15.3 |
| ⚠ | 후속 별도 사안 (본 설계 범위 밖) | (a) analysis_ledger 부활/폐기 결정, (b) card_composer 가 SK AX 자사 카드 만들도록 수정 |
