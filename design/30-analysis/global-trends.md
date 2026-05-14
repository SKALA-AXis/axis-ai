# GlobalTrendsAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `GlobalTrendsAgent` |
| **Supervisor** | Analysis |
| **상태** | 🆕 신규 — PDF 2026-05-14 §4 (글로벌 동향 모듈) 직접 대응. backend 미구현. |
| **Trigger** | nightly 04:30 (배치) + on-demand `POST /api/global/trends/run` |

## 2. 책임

**한 줄**: 글로벌 6사 (NVIDIA / Apple / Microsoft / Google / Amazon / Meta) 의 최근 동향을 → SK AX 의 국내 IT 서비스 사업 (AI/AX/Cloud/Infra/Security) 에 어떤 영향을 미치는지 자동 분석 + 1Q/반기/1년 시나리오까지.

**구체적 (4-phase + CoT — Peer+ 4-phase 와 동일 패턴, 단 화자가 SK AX, 비교군이 글로벌)**:

1. **Phase 1 — Global Snapshot** (산식): 6사 × 최근 30일 (analysis_period) 의 카드 수 + 주요 sector mix + 발화 (announcements/keynote).
2. **Phase 2 — Trend Detection** (산식): 새로 등장한 키워드 / 기술 / 파트너십 패턴 — ±50% frequency band (checklist 9).
3. **Phase 3 — SK AX Impact Mapping** (LLM): 글로벌 트렌드 × SK AX 사업 영역 (AI 매니지드 / Cloud MSP / Security / Smart Factory) 매트릭스 — 긍정/중립/부정.
4. **Phase 4 — Forecast + Strategic Response** (LLM, **PDF §4 직접 대응**): 1Q 후 / 반기 후 / 1년 후 SK AX 가 받을 영향 시나리오 (낙관/기준/비관) + 권장 대응 + `final_one_liner`.

> **왜 PeerComparison 과 분리하나?** Peer (국내 4사) 는 *경쟁* / *벤치마크* 관점 — KPI 비교가 핵심. Global (해외 6사) 는 *기술/규제/생태계* shock 의 source 로 **간접 영향 추정** 이 핵심. 출처 (Tier1: 공식 keynote/SEC) + 매트릭스 (영향 채널) 구조가 다름.

## 3. 책임 NOT

- 글로벌 6사 자체의 매출/주가 예측 → 본 agent 영역 아님 (외부 데이터)
- 단순 영문 기사 번역/요약 → NewsSummaryAgent (영어 모드) 가 담당
- 출처 검증 → EvidenceAgent

## 4. 입력 스펙

```python
class GlobalTrendsInput(TypedDict):
    company_ids: list[str] | None
    # ["nvidia", "apple", "microsoft", "google", "amazon", "meta"] — None 이면 6사 전부
    focus_themes: list[str] | None
    # ["agentic_ai", "custom_silicon", "ai_infra", "data_center", ...] — None 이면 모든 theme

    # checklist 5/6 — 절대 기준 (필수)
    analysis_period: dict
    # { "since": "2026-04-15", "until": "2026-05-14",
    #   "label": "최근 30일 (KST)", "fiscal_anchor": "2026-1Q" }

    # checklist 10 — 수익화 관점 명시
    sk_ax_business_lines: list[str]
    # ["ai_managed", "cloud_msp", "security", "smart_factory", "data_platform"]
    # 영향 매트릭스의 컬럼 축으로 사용
```

## 5. 출력 스펙

```python
class GlobalSnapshot(TypedDict):
    company_id: str
    card_count: int                       # analysis_period 내 카드 수
    top_themes: list[str]                 # top 3 theme
    headline_announcements: list[dict]    # [{title, source, source_tier, url, published_at_kst}]
    source_marker: str                    # checklist 12 — "[공식 keynote]" / "[SEC 10-Q]" / "[Tier2 기사]"

class TrendDetection(TypedDict):
    theme: str                            # "agentic_ai" / "custom_silicon" / ...
    frequency_delta_pct: float            # 전기 대비 (4주 평균) — ±50% 시 strong
    intensity: Literal["weak", "moderate", "strong"]  # checklist 9 band
    leading_companies: list[str]          # 이 theme 을 주도하는 글로벌 회사
    evidence_card_ids: list[str]          # 근거 카드 id

class SKAXImpactCell(TypedDict):
    """매트릭스 1 cell — (global_trend × sk_ax_business_line)"""
    trend_theme: str
    sk_ax_line: str                       # 예: "ai_managed"
    direction: Literal["positive", "neutral", "negative"]    # checklist 10
    magnitude: Literal["low", "medium", "high"]
    channel: str                          # 영향 경로 (≤ 100자)
    quant_hint: str | None                # 가능 시 정량 (예: "GPU 가격 +30% → margin -2pp")
    source_marker: str                    # checklist 12

class GlobalForecast(TypedDict):
    horizon: Literal["1Q", "6M", "1Y"]
    scenario: Literal["optimistic", "baseline", "pessimistic"]
    narrative: str                        # ≤ 300자
    sk_ax_impact: str                     # ≤ 200자 — SK AX 의 sector 별 영향
    drivers: list[str]                    # 주요 가정 (3~5)
    risk_level: Literal["low", "medium", "high"]   # checklist 16
    recommended_response: str             # ≤ 200자 — SK AX 권장 대응

class CoTStep(TypedDict):
    """02-prompt-design-checklist §4 표준."""
    step_idx: int
    phase: Literal["snapshot", "trend_detect", "impact_map", "forecast", "synthesis"]
    question: str
    inputs_used: list[str]
    answer: str
    intermediate_conclusion: str
    confidence: float

class GlobalTrendsOutput(TypedDict):
    analysis_period: dict                 # input echo (UI 노출)
    snapshots: list[GlobalSnapshot]       # 6사
    trend_detections: list[TrendDetection]  # 5~10 theme
    impact_matrix: list[SKAXImpactCell]   # |trends| × |sk_ax_lines|
    forecasts: list[GlobalForecast]       # 3 horizon × 3 scenario = 9 (또는 baseline 만 3)
    final_one_liner: str                  # checklist 18 — SK AX 관점 한 줄 결론 (≤ 100자)
    sk_ax_implication: str                # 1~2 문장 — 국내 IT 서비스사 관점 (긍정/중립/부정 명시)
    follow_up_questions: list[str]        # checklist 17 — 2~3 후속 질문
    reasoning_steps: list[CoTStep]
    risk_assumptions: list[str]           # checklist 16
    confidence: float
    provenance: dict
```

frontend endpoint: `GET /api/global/trends` (캐시 24h) + `POST /api/global/trends/run` (배치 트리거).

## 6. 알고리즘

### 6.1 Context 구성

```python
def build_context(input: GlobalTrendsInput):
    companies = input["company_ids"] or GLOBAL_COMPANIES
    since, until = input["analysis_period"]["since"], input["analysis_period"]["until"]

    blocks = []
    for cid in companies:
        cards = fetch_global_cards(cid, since=since, until=until)
        # source tier 분포 (Tier1 = 공식 keynote/SEC / Tier2 = Bloomberg, Reuters, FT / Tier3 = blog)
        tier_counts = Counter(c.source_tier for c in cards)
        blocks.append({
            "company_id": cid,
            "cards": cards,
            "tier_distribution": dict(tier_counts),
        })
    return blocks
```

### 6.2 Trend Detection 산식

```python
def detect_trends(blocks, prev_blocks):
    """analysis_period 의 theme frequency vs 직전 동일 기간 비교."""
    cur = Counter()
    prev = Counter()
    for b in blocks:
        for c in b["cards"]:
            cur.update(c.themes)   # 카드 enrichment 단계의 themes[] 활용
    for b in prev_blocks:
        for c in b["cards"]:
            prev.update(c.themes)

    detections = []
    for theme, cur_n in cur.most_common():
        prev_n = max(prev.get(theme, 0), 1)
        delta = (cur_n - prev_n) / prev_n * 100
        if abs(delta) < 20:
            continue
        intensity = "strong" if abs(delta) >= 50 else ("moderate" if abs(delta) >= 30 else "weak")
        detections.append({
            "theme": theme,
            "frequency_delta_pct": round(delta, 1),
            "intensity": intensity,
            "leading_companies": _leaders_for(theme, blocks),
            "evidence_card_ids": _cards_for(theme, blocks),
        })
    return detections[:10]
```

### 6.3 LLM Prompt (gpt-4o, 4-phase CoT)

```text
당신은 SK AX 사업전략팀의 글로벌 트렌드 분석 전문가입니다.
글로벌 빅테크 6사 (NVIDIA / Apple / Microsoft / Google / Amazon / Meta) 의 최근 동향이
SK AX 의 국내 IT 서비스 사업 ({sk_ax_business_lines}) 에 어떤 영향을 미치는지를
4-phase 로 추론해야 합니다.

분석 기간: {analysis_period.label}  ← 절대 기준, "최근" 같은 모호 표현 금지
회계 기준: {analysis_period.fiscal_anchor}

[Phase 1 — Snapshot] 산식 결과:
{snapshots_json}

[Phase 2 — Trend Detection] 산식 결과:
{trend_detections_json}

[Phase 3 — Impact Mapping] (당신이 채울 것)
각 trend × sk_ax_line cell 에 대해:
  - direction: positive / neutral / negative
  - magnitude: low / medium / high
  - channel: 영향이 흐르는 경로 (구체적, ≤ 100자)
  - quant_hint: 가능 시 정량 수치 (예: "AI GPU 가격 +30%")
  - source_marker: [공식 keynote] / [SEC 10-Q] / [Tier2 기사] / [자체 추정]

[Phase 4 — Forecast] (당신이 채울 것)
1Q 후 / 반기 후 / 1년 후 각각 baseline 시나리오 1개씩 (총 3개) — 분량 통제.
optimistic / pessimistic 은 baseline 가 risk_level=high 인 경우만 추가 생성.

각 forecast 는:
  - narrative (≤ 300자)
  - sk_ax_impact (≤ 200자, 사업 line 별 영향)
  - drivers (3~5개 가정)
  - risk_level: low / medium / high
  - recommended_response (SK AX 권장 대응, ≤ 200자)

[Synthesis]
  - final_one_liner: 한 문장 ≤ 100자, 모호함 금지
  - sk_ax_implication: 1~2 문장, 국내 IT 서비스사 관점 (긍정/중립/부정 명시)
  - follow_up_questions: 다음 분석 시 추가로 봐야 할 항목 2~3개
  - risk_assumptions: 본 분석이 틀릴 가정 2~3개

[CoT 강제]
reasoning_steps[] 에 각 phase 별 1+ step 으로 question / inputs_used / answer /
intermediate_conclusion / confidence 명시. 사용자가 "global agent 가 어떻게 추론했는가" 를
expandable panel 에서 단계 별로 확인할 수 있도록.

[정량 우선] 가능한 모든 곳에 수치 + source_marker prefix. "성장 추세" → 금지 / "AI 인프라
지출 YoY +35% [Microsoft FY25 Q3]" → 권장.

[화자]
"NVIDIA 가 X 했다" 가 아니라 "NVIDIA 의 X 는 SK AX 의 ai_managed 라인에 ___ 영향" — 화자 = SK AX.

JSON 만 출력. schema:
{GlobalTrendsOutput JSON Schema 압축}
```

### 6.4 Prompt audit — 02-prompt-design-checklist 17 요소

GlobalTrendsAgent 는 **모든 17 요소 + CoT + final_one_liner 충족 의무** (PDF 4페이지 직접 대응 + checklist §5 명시).

| # | 요소 | 충족 위치 | 비고 |
|---|---|---|---|
| **1** | 역할 정의 | prompt 도입부 | "SK AX 사업전략팀의 글로벌 트렌드 분석 전문가" |
| **2** | 추적 대상 기업 | input.company_ids (6사 enum) | NVIDIA / Apple / MS / Google / Amazon / Meta |
| **3** | 추적 범위 | input.focus_themes + sk_ax_business_lines | theme taxonomy + SK AX 5 line |
| **4** | 출처 우선순위 | snapshots[].source_marker + source_tier | Tier1 keynote/SEC > Tier2 Bloomberg/Reuters > Tier3 blog |
| **5** | 분석 기간 | input.analysis_period (since/until/label/fiscal_anchor) | 절대 기준 |
| **6** | 최신성 검증 | snapshots[].headline_announcements[].published_at_kst | KST 변환 강제 |
| **7** | 단순 뉴스 요약 금지 | 4-phase + theme detection | "기사 N개" 금지 → trend × impact pattern |
| **8** | 회사별 비교 기준 | snapshots[].card_count + top_themes + tier_distribution | 6사 동일 KPI |
| **9** | 변화 감지 기준 | trend_detections[].intensity (±20/30/50%) | checklist 9 band |
| **10** | 수익화 관점 | impact_matrix[].direction (positive/neutral/negative) | SK AX 매출/마진 영향 명시 |
| **11** | 정량 수치 우선 | impact_matrix[].quant_hint + forecasts[].drivers | "+30%" / "-2pp" 표기 |
| **12** | 공식 vs 추정 구분 | source_marker prefix 강제 ([공식 keynote] vs [자체 추정]) | 모든 정량 출력 |
| **13** | 전략적 시사점 | sk_ax_implication 필드 + impact_matrix 화자 = SK AX | "_의 X 는 SK AX 의 Y 라인에 ___" |
| **14** | 출력 형식 | GlobalTrendsOutput TypedDict | 필수 |
| **15** | 우선순위 판단 | trend_detections top 10 + forecasts baseline 만 기본 (high risk 만 3 scenario) | 모호한 "중요" 금지 |
| **16** | 리스크 분석 | forecasts[].risk_level + risk_assumptions[] | 가정 + 대응 명시 |
| **17** | 반복 추적 구조 | follow_up_questions[] 2~3개 | "다음 분석 시 봐야 할 항목" |
| **CoT** | reasoning_steps | 5 phase × 1+ step | snapshot/trend_detect/impact_map/forecast/synthesis |
| **결론** | final_one_liner | 한 문장 ≤ 100자 | 모호함 금지 |

→ **17/17 직접 충족 + CoT + final_one_liner** (PDF 직접 요구 모두 만족)

## 7. LLM 모델 + token 예산

| Phase | 모델 | input | output | 호출/일 | 비용 |
|---|---|---|---|---|---|
| Snapshot/Trend 산식 | (LLM 미사용) | — | — | — | ₩0 |
| Impact Map + Forecast + Synthesis | gpt-4o | ~6,000 | ~3,500 | 1 (nightly) + on-demand ~3 | ₩900 |
| **합계** | | | | | **~₩900/일** |

> 캐시 24h. on-demand 호출은 input hash 키로 분기 캐싱.

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| 글로벌 카드 < 5건 (cold start) | "데이터 부족" warning + snapshots 만 반환 (forecast 생략) |
| theme detection 0건 | trend_detections=[], impact_matrix=[], forecast 는 generic baseline 만 |
| LLM JSON 파싱 실패 | 1회 재시도, 그래도 실패 시 reasoning_steps 만 반환 (final_one_liner=None) |
| confidence < 0.5 | warning="추정 신뢰도 낮음" 표시 |

## 9. 외부 의존성

- **DB**: `global_company_cards` (READ — 6사 카드), `analysis_cache` (UPSERT, TTL 24h)
- **sub-agents**: 없음 (산식 + 단일 LLM call)
- **외부 API**: OpenAI gpt-4o

## 10. State 흐름

AnalysisState 의 `global_trends` 필드. Briefing supervisor 가 daily 브리핑에 final_one_liner 인용.

## 11. Provenance + Confidence

- **Provenance**: `provenance.{llm_model, prompt_version, run_at_kst, evidence_version, source_card_ids}`
- **Confidence**: min(snapshot tier1 비율, LLM 자체 평가, trend detection coverage) — 0.0~1.0

## 12. 테스트 시나리오

| Unit | NVIDIA Blackwell 발표 + Microsoft GPT-5 announcements | trend "agentic_ai" intensity=strong, impact_matrix.ai_managed.direction=positive |
| Unit | Apple WWDC + Google I/O 동주 | trend "on_device_ai" detect, sk_ax_implication 명시 |
| Unit | 6사 모두 0 카드 (cold start) | warning + snapshots only |
| Unit | risk_level=high 인 baseline | optimistic/pessimistic 시나리오 추가 생성 |
| Edge | analysis_period 결측 | 400 BadRequest |
| Edge | confidence < 0.5 | warning 표시 + UI 강조 |

## 13. 모니터링

- KPI:
  - trend detection precision (sampling) ≥ 70%
  - impact_matrix coverage ≥ 60% (cell 채워진 비율)
  - final_one_liner 길이 ≤ 100자 (100%)
  - 평균 confidence ≥ 0.65
- 일일 token: ~₩900

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/global_trends_agent.py` (신규)
- 종속: `global_company_cards` 테이블 (V11+ migration 필요 — Classification 결과의 company_id enum 확장)
- frontend Peer+ 페이지에 "Global" 탭 추가 → `GET /api/global/trends`

### Changelog

- **v1 (2026-05-14)** — PDF 사업전략팀 추가 질의 회신 §4 직접 대응 신설. 17 요소 + CoT + final_one_liner 모두 충족.
