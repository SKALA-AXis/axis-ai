# PeerComparisonAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `PeerComparisonAgent` |
| **Supervisor** | Analysis |
| **상태** | 🟡 backend fixture (`GET /api/monitoring/{peerId}/strategy`, `GET /api/monitoring/comparison`), axis-ai 신규 |
| **Trigger** | User Peer+ 페이지에서 peer 선택 시 |

## 2. 책임

**한 줄**: 선택 peer 의 최근 카드 + IR + 키워드 vs **SK AX (자사)** 의 카드/포지션을 비교하여 **차별 시사점 + 전략 라벨 + 향후 forecast (1Q / 반기 / 1년)** 자동 추출.

**구체적 (4-phase, PDF 2026-05-14 §4 직접 대응)**:

1. **Phase 1 — Current state** (산식 + LLM): peer 의 최근 30일 card_news + 재무 IR pack + word cloud 종합. SK AX 자사 동일 영역 카드 비교.
2. **Phase 2 — Trend** (산식): QoQ / YoY 매출·영업이익·R&D 추세 + 카드 빈도 변화 (Tech vs Partnership vs M&A 비중). 변화 감지 기준 (PDF §9): ±5% normal / >10% 유의 / >30% 급변.
3. **Phase 3 — Forecast** (LLM, **PDF §4 신규**): 분기 재무 + 최근 동향 → **1Q 후 / 반기 후 / 1년 후** 의 3 시나리오 (낙관 / 기준 / 비관) + driver 명시 + 가정의 신뢰도.
4. **Phase 4 — Strategic implication + CoT** (LLM): SK AX 가 어떤 행동 / 어떤 대응 / 어떤 협력 — `final_one_liner` 강제.

PDF 직접 인용 (§4): *"Peer+의 경우 분기별로 다트 데이터를 참고하여 실적을 정리해준다고 한다면, 차라리, 그 실적과 최근 기사와 동향을 파악하여 향후 전망을 보여주는 형태 (1분기 후, 반기 후, 1년 후 등) 로 Insight 를 담아주면 좋을 것 같습니다."*

전략 라벨 (LLM, 5종 중 1) — "Aggressive Expansion" / "Defensive Hold" / "Tech Pivot" / "Customer Lock-in" / "Cost Leadership".

결과 cache (peer_id × week 키, 7일 TTL).

## 3. 책임 NOT

- IR 데이터 추출 — FinancialLinkerAgent + IRParserAgent
- 워드클라우드 — PeerWordCloudAgent
- 시각화 — frontend Peer+ 페이지

## 4. 입력 스펙

```python
class PeerComparisonInput(TypedDict):
    # 25-knowledge-curation Phase K3+ 도입 후 — agent 가 in-process 채움.
    # Peer 분석은 단일 peer 이므로 slim_mode=False (full pack 사용 권장).
    _context_pack: "PeerContextPack | None"

    # 기존 필드
    peer_id: str
    window_days: int            # 기본 30
    focus_sector: str | None    # 옵션 (특정 sector 만 비교)
```

## 5. 출력 스펙

```python
class TrendDelta(TypedDict):
    """Phase 2 — 정량 변화 지표."""
    metric: str                              # "revenue_krwbn" / "op_income_krwbn" / "op_margin_pct" / ...
    qoq_pct: float | None                    # 전기 대비
    yoy_pct: float | None                    # 전년 동기 대비
    band: Literal["normal","유의","급변"]    # PDF §9 임계값 (±5% / >10% / >30%)
    direction: Literal["up","down","flat"]
    source: str                              # "[공식 DART 2026-1Q]" 등

class Forecast(TypedDict):
    """Phase 3 — PDF §4 직접 대응. 1Q / 반기 / 1년 후 시나리오."""
    horizon: Literal["1Q","6M","1Y"]
    scenario: Literal["optimistic","baseline","pessimistic"]
    summary: str                             # 1 문장 (≤ 100자)
    drivers: list[str]                       # 2~4 driver
    quantitative_estimate: str | None        # "매출 +12~18% YoY" 같은 정량 범위 (가능 시)
    risk_assumptions: list[str]              # 본 시나리오가 틀릴 조건
    confidence: float                        # 0~1

class ReasoningTrailItem(TypedDict):
    """02-prompt-design-checklist.md §4 Tier 1 — 사용자 default 노출.
    Peer 의 4 phase (Current/Trend/Forecast/Strategic) 자연 매핑 → trail 4 step 권장."""
    seq: int
    label: str                          # ≤ 12자 ("현재 포지션" / "추세 비교" / "전망" / "SK AX 대응" / "결론")
    one_liner: str                      # ≤ 80자, 정량 수치 1개 우선 (QoQ +X% 등)
    evidence_refs: list[str]            # card_id / DART id / IR ref
    langfuse_observation_id: str | None

class CoTStep(TypedDict):
    """Tier 2 — 상세 ("더 자세히" 패널). 02-prompt-design-checklist §4 표준."""
    step_idx: int
    phase: Literal["current","trend","forecast","strategic"]
    question: str
    inputs_used: list[str]
    answer: str
    intermediate_conclusion: str
    confidence: float
    langfuse_observation_id: str | None

class PeerComparisonOutput(TypedDict):
    peer_id: str
    # Phase 1 — Current
    strategy_label: str                # "Aggressive Expansion" 등
    differentiators: list[dict]
    strengths_of_peer: list[str]
    weaknesses_of_peer: list[str]
    collaboration_potential: list[str]
    # Phase 2 — Trend
    trend_deltas: list[TrendDelta]
    # Phase 3 — Forecast (PDF §4)
    forecasts: list[Forecast]
    # Phase 4 — Synthesis
    sk_ax_implication: str             # PDF §13 — 1~2 문장 (긍정/중립/부정)
    final_one_liner: str               # PDF §5 — ≤ 100자
    follow_up_questions: list[str]     # PDF §17 — 2~3개
    # Meta — 3-tier observability
    reasoning_trail: list[ReasoningTrailItem]   # Tier 1 — 사용자 default (4~5)
    reasoning_steps: list[CoTStep]              # Tier 2 — 상세 (4~8)
    langfuse_trace_id: str | None               # Tier 3 — admin deep link
    confidence: float
    provenance: dict
    sources: list[str]
    analysis_period: dict              # PDF §6
```

frontend `GET /api/monitoring/{peerId}/strategy` 및 `/comparison` 응답.

## 6. 알고리즘

### 6.1 컨텍스트 수집

```python
def build_context(peer_id, window_days, focus_sector):
    peer_cards = fetch_cards(peer_id=peer_id, since=window_days, sector=focus_sector)
    peer_ir = FinancialLinkerAgent().fetch_segment_data(peer_id)
    peer_wordcloud = PeerWordCloudAgent().get(peer_id)
    skax_cards = fetch_cards(peer_id="sk_ax", since=window_days, sector=focus_sector)  # 자사 raw 만
    return {
        "peer": {"cards": peer_cards, "ir": peer_ir, "wordcloud": peer_wordcloud},
        "skax": {"cards": skax_cards, "ir": SK_AX_IR_BASELINE},
    }
```

### 6.2 Phase 2 — Trend deltas (산식, LLM 없음)

QoQ / YoY 추세는 재무 segment 데이터에서 deterministic 계산:

```python
def compute_trend_deltas(peer_ir):
    deltas = []
    for metric in ("revenue_krwbn", "op_income_krwbn", "op_margin_pct", "rd_investment_krwbn"):
        latest = peer_ir.series[-1][metric]
        prev_q = peer_ir.series[-2][metric] if len(peer_ir.series) >= 2 else None
        prev_y = peer_ir.series[-5][metric] if len(peer_ir.series) >= 5 else None
        qoq = (latest - prev_q) / prev_q * 100 if prev_q else None
        yoy = (latest - prev_y) / prev_y * 100 if prev_y else None
        band = _band(qoq, yoy)  # ±5% normal / >10% 유의 / >30% 급변 (PDF §9)
        deltas.append({
            "metric": metric,
            "qoq_pct": qoq,
            "yoy_pct": yoy,
            "band": band,
            "direction": _direction(qoq or yoy),
            "source": f"[공식 DART {peer_ir.latest_period}]",
        })
    return deltas
```

### 6.3 Phase 3 — Forecast (LLM, PDF §4 직접 대응)

분기 재무 추세 + 최근 동향 카드 → 3 horizon × 3 scenario = **9 forecast** (또는 핵심 3~5 select).

### 6.4 LLM Prompt (gpt-4o, 4-phase CoT)

~~~text
# SK AX 경쟁사 분석가

당신은 SK AX 사업전략팀의 경쟁사 분석가입니다.
**{peer_id} 의 현재 / 추세 / 향후 전망** 을 SK AX 관점에서 분석합니다.
추론 과정을 단계별로 명시적으로 노출합니다 (PDF §1).

## 입력 데이터

### 분석 기간 (KST 절대 기준)
- **since**: {since}
- **until**: {until}
- **fiscal_anchor**: {fiscal_anchor}

### {peer_id} 데이터
- **최근 30일 카드**: {peer_cards_summary}
- **재무 IR (분기 5기)**: {peer_ir_pack}
- **트렌드 deltas (산식)**: {trend_deltas}

### SK AX 비교 베이스라인
- **자사 카드**: {skax_cards_summary}
- **자사 베이스라인**: {skax_baseline}

### Peer Context Pack (Phase K3+)
{context_pack 또는 "*cold start*"}

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **출처 prefix**: 모든 정량 수치 앞에 `[공식 DART {fiscal_anchor}]` / `[기사 인용 CN-...]` / `[자체 추정]`
- **화자 고정**: 모든 phase 결론이 `"SK AX 의 ___"` pattern (peer 의 행동을 SK AX 영향으로 환산)
- **환각 금지**: trend_deltas 가 산식 출력이므로 임의 수치 추가 금지 — 입력 carry only

### 일반 규칙 (17 요소 매핑)
1. **(#1 역할)** SK AX 경쟁사 분석가 관점만
2. **(#2 추적 대상)** 4 국내 peer / 6 글로벌 (cross-comparison 가능)
3. **(#5 분석 기간)** `analysis_period` 절대 기준 (`"2026-04-15 ~ 2026-05-14 KST"`)
4. **(#6 최신성)** published_at 의 KST 변환 + 분기 boundary
5. **(#8 회사별 비교 기준)** 매출 / 영업이익 / 영업이익률 / R&D / Captive 비중 enum
6. **(#9 변화 감지)** ±5% normal / >10% 유의 / >30% 급변 — band 명시
7. **(#10 수익화 관점)** sk_ax_implication 의 긍정/중립/부정
8. **(#11 정량 우선)** trend_deltas + forecasts.quantitative_estimate
9. **(#13 SK AX 화자)** 위 절대 규칙 pattern
10. **(#16 리스크)** forecasts.risk_assumptions 필수
11. **(#17 반복 추적)** follow_up_questions[]

## 추론 단계 (Chain of Thought)

### Phase 1 — Current state
- **자기 질문**: `"현재 {peer_id} 의 포지션과 SK AX 와의 차별점은?"`
- **입력**: peer_cards + skax_cards + context_pack.canonical_facts
- **출력**: strategy_label (5종 중 1), differentiators, strengths/weaknesses, collaboration_potential

### Phase 2 — Trend interpretation
- **자기 질문**: `"trend_deltas 의 의미는? 어떤 사업 방향 전환?"`
- **입력**: trend_deltas + peer_ir_pack
- **출력**: 정량 (QoQ +X%) 후 정성 해석 (PDF §11 적용)

### Phase 3 — Forecast (PDF §4 직접 대응)
- **자기 질문**: `"1Q 후 / 반기 후 / 1년 후 {peer_id} 의 시나리오는?"`
- **입력**: Phase 1 + 2 + DART 추세
- **출력**: 3 horizon × 3 scenario (낙관/기준/비관) — 9개 (또는 baseline 만 3개)
- **필수 필드**: drivers + risk_assumptions (#16) + quantitative_estimate (가능 시)

### Phase 4 — Strategic implication + final
- **자기 질문**: `"SK AX 는 어떤 행동? 우선순위 1개?"`
- **출력**: sk_ax_implication (1~2 문장, 긍정/중립/부정) + final_one_liner (≤ 100자)

## 3-Tier Observability 출력

### Tier 1 — reasoning_trail (사용자 default)
정확히 **4 step**. 각 phase 1 step.

권장 label sequence:
1. **"현재 포지션"**
2. **"추세 비교"** (one_liner 예시: `"매출 QoQ +12% / 영업이익률 -2pp [DART]"`)
3. **"전망"**
4. **"SK AX 대응"**

### Tier 2 — reasoning_steps (상세)
phase=current / trend / forecast (3 step — 3 horizon 별) / strategic = **총 6 step**.

### Tier 3 — langfuse_trace_id
`null` 로 출력. `LangfuseTraceLinker` 미들웨어가 자동 매핑.

## 출력 형식 (strict JSON)

```json
{
  "strategy_label": "Aggressive Expansion | Defensive Hold | Tech Pivot | Customer Lock-in | Cost Leadership",
  "differentiators": [{"aspect": "...", "peer_position": "...", "skax_position": "...", "opportunity": "..."}],
  "strengths_of_peer": ["..."],
  "weaknesses_of_peer": ["..."],
  "collaboration_potential": ["..."],
  "trend_deltas": [],
  "forecasts": [
    {
      "horizon": "1Q|6M|1Y",
      "scenario": "optimistic|baseline|pessimistic",
      "summary": "...",
      "drivers": ["...", "..."],
      "quantitative_estimate": "매출 +12~18% YoY",
      "risk_assumptions": ["..."],
      "confidence": 0.0
    }
  ],
  "sk_ax_implication": "1~2 문장. 긍정/중립/부정 명시.",
  "final_one_liner": "≤ 100자, SK AX 관점",
  "follow_up_questions": ["...", "..."],
  "reasoning_trail": [
    {"seq": 1, "label": "현재 포지션", "one_liner": "...", "evidence_refs": ["CN-..."], "langfuse_observation_id": null}
  ],
  "reasoning_steps": [
    {"step_idx": 0, "phase": "current", "question": "...", "inputs_used": [],
     "answer": "...", "intermediate_conclusion": "...", "confidence": 0.0, "langfuse_observation_id": null}
  ],
  "confidence": 0.0,
  "sources_used": ["CN-...", "DART:rcept-..."]
}
```
~~~

### 6.4.1 Prompt 양식 audit — 02-prompt-design-checklist §6

| 양식 항목 | 충족 |
|---|---|
| `#` agent role 1개만 | ✅ |
| `##` 5 major section | ✅ |
| `###` sub-section heading | ✅ |
| 절대 규칙 + bold label | ✅ |
| numbered rule + (#N) inline | ✅ |
| JSON code fence | ✅ |
| `[Brackets]` 폐기 | ✅ |

### 6.5 17 요소 prompt audit table

| # | 요소 | 충족 | 위치 |
|---|---|---|---|
| 1 | 역할 정의 | ✅ | "SK AX 경쟁사 분석가" |
| 2 | 추적 대상 | ✅ | peer_id + SK AX |
| 5 | 분석 기간 | ✅ | analysis_period strict |
| 6 | 최신성 | ✅ | KST + 분기 boundary |
| 8 | 회사별 비교 기준 | ✅ | 매출/영업이익/마진/R&D/Captive |
| 9 | 변화 감지 기준 | ✅ | ±5/10/30% band |
| 10 | 수익화 관점 | ✅ | sk_ax_implication 긍정/중립/부정 |
| 11 | 정량 우선 | ✅ | trend_deltas + forecasts |
| 12 | 공식 vs 추정 | ✅ | source prefix 강제 |
| 13 | 전략 시사점 | ✅ | "SK AX ___" 화자 |
| 14 | 출력 형식 | ✅ | strict JSON schema |
| 16 | 리스크 분석 | ✅ | forecasts.risk_assumptions |
| 17 | 반복 추적 | ✅ | follow_up_questions |
| 3 | 추적 범위 | ⚪ | event_type taxonomy 는 카드가 가지고 옴 |
| 4 | 정보 출처 우선순위 | 🟡 | Tier1 (DART/IR) > Tier2 명시 권장 |
| 7 | 단순 요약 금지 | ✅ | 4-phase pattern |
| 15 | 우선순위 | 🟡 | sk_ax_implication 의 "우선 1 행동" 명시 권장 |

## 7. LLM 모델 + token 예산

- **모델**: gpt-4o
- 토큰/호출: ~3,000 (in 2,000 + out 1,000)
- 일일 호출: ~4 (peer 4사 각 1회 + on-demand 추가)
- **일일 비용**: ~₩200

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| peer cards 0건 | 404 또는 "최근 데이터 없음" stub |
| skax cards 0건 | 자사 baseline static 데이터 사용 |
| LLM JSON parse fail | retry 1회 → stub |
| confidence < 0.5 | warning "데이터 부족 — window_days 늘리세요" |

## 9. 외부 의존성

- **DB**: `card_news`, `peer_financials`, `analysis_cache`
- **외부 API**: OpenAI gpt-4o
- **sub**: FinancialLinkerAgent, PeerWordCloudAgent

## 10. State 흐름

AnalysisState `request_type='peer_compare'` 라우팅.

## 11. Provenance + Confidence

- **Provenance**: 자동
- **Confidence**: LLM 자체 + 데이터량 가중

## 12. 테스트 시나리오

| Unit | peer='samsung_sds', focus='ax' | strategy_label 1개 + differentiators 3+ |
| Unit | peer='unknown_peer' | 400 |
| Edge | peer 카드 5건만 (sparse) | confidence < 0.5 + warning |

## 13. 모니터링

- KPI: 평균 confidence ≥ 0.65
- token: ₩200/일

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/peer_comparison_agent.py` (신규 P7)
- SK AX 자사 baseline: `data/sk_ax_baseline.json` (manual seed)

### Changelog

- **v1 (제안, P7)** — LLM 비교 + 5-strategy 라벨
- **v2 (2026-05-14, 사업전략팀 추가 질의 회신 §4 반영)** — Forecast phase 추가
  (1Q / 6M / 1Y × 낙관/기준/비관), Trend deltas (산식 자동), CoT
  reasoning_steps, final_one_liner / sk_ax_implication / follow_up_questions /
  risk_assumptions / analysis_period. PDF §4 직접 인용 충실 + §5/§6/§8/§9/
  §11/§13/§16/§17 적용. 17 요소 prompt audit (14/17 충족, 3 권장).
