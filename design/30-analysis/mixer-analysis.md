# MixerAnalysisAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `MixerAnalysisAgent` |
| **Supervisor** | Analysis |
| **상태** | 🟡 backend fixture (`POST /api/mixer`, `GET /api/mixer/options`, `POST /api/mixer/{id}/share`), axis-ai 신규 |
| **Trigger** | User Mixer 페이지에서 "Generate" 클릭 |

## 2. 책임

**한 줄**: 2~20 card_news 조합 + 입력 비율 (peer/industry/keyword) 로부터 **cross-card 신호** + **6축 radar 분포** + **연결 관계 (인과/유사)** 추출 + **CoT reasoning 가시화** + **최종 한 줄 결론** (PDF 2026-05-14 §5 직접 대응).

**구체적 (3-phase CoT, single-LLM-call with explicit reasoning_steps)**:

1. **카드 조합 정렬** (산식) — exposure_score / event_type / sector 다양성 우선
2. **6축 radar score** (산식) — Peer Strategic Shift / Tech Investment / Market Position / Partnership Momentum / Regulatory Risk / Talent Movement
3. **3-phase CoT LLM** — single gpt-4o call 의 prompt 가 step-by-step thinking 을 명시적으로 `reasoning_steps[]` JSON 으로 출력:
   - Phase 1: 카드별 핵심 신호 추출 (per-card mini-summary)
   - Phase 2: cross-card 연결 (pair-wise relation 추론)
   - Phase 3: 종합 결론 + SK AX 시사점 + **최종 한 줄 결론**

> PDF 피드백 §5 직접 대응 — "에이전트들 간 협업이 어떻게 이뤄졌는지, 구축한 아키텍쳐의 일부가 UI 에 녹여지면". `reasoning_steps[]` 가 frontend 의 expandable panel 로 렌더되어 "추론 과정 보기" 클릭 시 step 별 input → question → answer → conclusion 체인 표시.

## 3. 책임 NOT

- 카드 선택 자체 — frontend Mixer UI 가 사용자에게 위임
- Insight 4-step 분석 — InsightCascadeAgent (별도)
- 단일 카드 분석 — CardComposerAgent

## 4. 입력 스펙

```python
class MixerAnalysisInput(TypedDict):
    card_ids: list[str]        # 2~20
    ratios: dict                # {peer: {samsung_sds: 0.4, ...}, industry: {...}, keyword: [...]}
    user_context: str | None    # 추가 컨텍스트 (자유 입력)
```

## 5. 출력 스펙

```python
class RadarAxis(TypedDict):
    axis: Literal["peer_strategic_shift","tech_investment","market_position",
                  "partnership_momentum","regulatory_risk","talent_movement"]
    score: float    # 0~1
    explanation: str

class Connection(TypedDict):
    source_card_id: str
    target_card_id: str
    label: Literal["cause","effect","similar","contrast","reinforce"]
    weight: float

class ReasoningTrailItem(TypedDict):
    """02-prompt-design-checklist.md §4 Tier 1 — 사용자 default 노출.
    raw reasoning_steps 가 5~8 step 이라도 trail 은 3~5 step 으로 압축."""
    seq: int                            # 1부터
    label: str                          # ≤ 12자 ("카드 비교" / "패턴 발견" / "재무 검증" / "결론")
    one_liner: str                      # ≤ 80자, 가능하면 정량 수치 1개 포함
    evidence_refs: list[str]            # card_id / DART id / IR ref (UI 클릭 → 원본)
    langfuse_observation_id: str | None # 런타임에 매핑 (admin only)

class CoTStep(TypedDict):
    """Tier 2 — 상세 ("더 자세히" 접힌 패널). PDF 2026-05-14 §5 대응."""
    step_idx: int                       # 0부터
    phase: str                          # "per_card" | "cross_card" | "synthesis"
    question: str                       # ≤ 200자
    inputs_used: list[str]              # card_id list
    answer: str                         # LLM raw response (≤ 500자)
    intermediate_conclusion: str        # ≤ 150자
    confidence: float
    langfuse_observation_id: str | None # 이 step 에 대응하는 Langfuse span id

class MixerAnalysisOutput(TypedDict):
    mix_id: str
    insight: str               # 1문장 종합 (현행 유지)
    final_one_liner: str       # PDF §5 — SK AX 관점 한 줄 결론 (≤ 100자, 모호함 금지)
    sk_ax_implication: str     # PDF §13 — 국내 IT 서비스사 (SK AX) 관점 1~2 문장
    bullet_signals: list[str]  # 3 핵심 신호
    radar_axes: list[RadarAxis]
    connections: list[Connection]
    reasoning_trail: list[ReasoningTrailItem]   # Tier 1 — 사용자 default (3~5)
    reasoning_steps: list[CoTStep]              # Tier 2 — 상세 (5~8)
    langfuse_trace_id: str | None               # Tier 3 — admin deep link
    follow_up_questions: list[str]   # PDF §17 — 다음 분석 제안 2~3개
    confidence: float
    sources_used: list[str]
    provenance: dict
```

frontend `POST /api/mixer` 응답. **3-tier observability** — UI default 는 `reasoning_trail` (3~5 step, "추론 흐름" 패널), 클릭 시 `reasoning_steps` expand, admin 만 `langfuse_trace_id` 딥링크 노출.

## 6. 알고리즘

### 6.1 6축 radar score (산식)

```python
RADAR_AXES = {
    "peer_strategic_shift": lambda cards: avg([c.exposure_score for c in cards if c.event_type in ("ma","new_biz")]),
    "tech_investment": lambda cards: avg([c.exposure_score for c in cards if c.sector in ("ax","infra")]),
    "market_position": lambda cards: peer_diversity_score(cards),  # 다양한 peer 일수록 높음
    "partnership_momentum": lambda cards: avg([c.exposure_score for c in cards if c.event_type == "partnership"]),
    "regulatory_risk": lambda cards: avg([c.exposure_score for c in cards if c.event_type == "regulation"]),
    "talent_movement": lambda cards: avg([c.exposure_score for c in cards if c.event_type == "personnel"]),
}
```

### 6.2 LLM Prompt — 3-phase CoT (gpt-4o, single call with explicit reasoning)

PDF 2026-05-14 §5 직접 대응. 단일 LLM call 안에서 step-by-step thinking 을 explicit 하게 출력 (비용 ↓ + UI 노출 ↑).

```text
당신은 SK AX 사업전략팀의 멀티 카드 분석 전문가다. 본 task 는 단일 답 생성이 아니라
*추론 과정 자체를 명시적으로 보여주는 것* — 분석가가 어떻게 결론에 도달했는지 UI 가
사용자에게 노출한다. 따라서 각 단계의 question / inputs / answer / intermediate_conclusion
을 정확히 채워라.

[입력]
- 카드 N건 (peer / event / sector / exposure 다양): {cards_summary}
- 사용자 비율 (Peer / Industry / Keyword): {ratios}
- 사용자 컨텍스트: {user_context}

[추론 단계]

Phase 1 — per_card (각 카드 1 step):
  step.question = "이 카드 (CN-XXX) 의 핵심 신호는?"
  step.inputs_used = ["CN-XXX"]
  step.answer = 카드의 event_type + 핵심 사실 + 정량 수치 (있으면 "[공식 DART]" 또는 "[기사 인용]" prefix)
  step.intermediate_conclusion = "카드 X = {one line signal}"

Phase 2 — cross_card (의미 있는 pair 별 1 step, 최대 8 step):
  step.question = "CN-A 와 CN-B 는 어떻게 연결되는가?"
  step.inputs_used = ["CN-A", "CN-B"]
  step.answer = 두 카드의 비교 + 인과 / 유사 / 대조 / 강화 관계 추론
  step.intermediate_conclusion = "{label}: {evidence}"
  → connections[] 항목과 1:1 대응

Phase 3 — synthesis (반드시 마지막 step):
  step.question = "위 단계의 결론을 종합하면 SK AX 가 주목해야 하는 단일 주제는?"
  step.inputs_used = 모든 card_id
  step.answer = 종합 reasoning (≤ 500자)
  step.intermediate_conclusion = 최종 한 줄 결론 (← final_one_liner 와 동일해야 함)

[작성 규칙 — 02-prompt-design-checklist.md 의 17 요소 적용]
1. (역할) SK AX 사업전략팀 관점만 사용. 일반 분석가 X.
2. (추적 대상) 카드의 peer 가 4 + 6 글로벌 + SK AX 자체에 한정.
7. (단순 요약 금지) "기사 X 개 요약" X — event_type / 변화 / 시사점 패턴.
10. (수익화 관점) "SK AX 매출/마진 영향 = 긍정/중립/부정" 명시.
11. (정량 수치 우선) "성장 추세" X → "QoQ +12.3%" 형태.
12. (공식 vs 추정 구분) [공식 DART] / [기사 인용] / [자체 추정] prefix.
13. (전략 시사점) "삼성SDS 가 X" X → "삼성SDS X 는 SK AX Y 에 ___ 영향".
15. (우선순위) "다음 3 신호 중 가장 영향 큰 것 1개 선택" 강제.
17. (반복 추적) follow_up_questions[] 2~3개 — 다음 분석 위한 질문.

[Tier 1 — reasoning_trail 압축 narrative (사용자 default)]
raw reasoning_steps 가 5~8 step 이어도 trail 은 **정확히 3~5 step** 으로 압축하라.
탐색/시도/hedging 표현 금지 ("~ 인 듯하다" / "고민했으나" X). 핵심 결정만 trail.
각 trail step:
- seq: 1, 2, 3, ...
- label: ≤ 12자 명사구 ("카드 비교" / "패턴 발견" / "재무 검증" / "결론")
- one_liner: ≤ 80자 한국어 단문, 가능하면 정량 수치 1개 ("QoQ +12%")
- evidence_refs: 참조한 card_id 목록
- langfuse_observation_id: null (런타임에 매핑됨)

목표: 사용자가 trail 만 보고도 "왜 이 결론에 왔는가" 명확.

[출력 — strict JSON]
{
  "insight": "1문장 종합 (≤ 50자)",
  "final_one_liner": "최종 한 줄 결론, SK AX 관점, 모호함 금지 (≤ 100자)",
  "sk_ax_implication": "국내 IT 서비스사 (SK AX) 관점 1~2 문장. '긍정/중립/부정' 명시",
  "bullet_signals": ["신호 1", "신호 2", "신호 3"],
  "connections": [
    {"source_card_id": "CN-...", "target_card_id": "CN-...", "label": "cause|effect|similar|contrast|reinforce", "weight": 0.0~1.0}
  ],
  "reasoning_trail": [
    {"seq": 1, "label": "카드 비교", "one_liner": "...", "evidence_refs": ["CN-..."], "langfuse_observation_id": null}
    // 정확히 3~5 item
  ],
  "reasoning_steps": [
    {
      "step_idx": 0,
      "phase": "per_card|cross_card|synthesis",
      "question": "...",
      "inputs_used": ["CN-..."],
      "answer": "...",
      "intermediate_conclusion": "...",
      "confidence": 0.0~1.0,
      "langfuse_observation_id": null
    }
    // ... 5~8 step
  ],
  "follow_up_questions": ["...", "...", "..."],
  "confidence": 0.0~1.0
}
```

### 6.2.1 langfuse_trace_id / observation_id 매핑

LLM call 직후 axis-ai 의 `LangfuseTraceLinker` middleware 가:

```python
def link_trace(output_dict: dict) -> dict:
    from langfuse import get_client
    client = get_client()
    output_dict["langfuse_trace_id"] = client.get_current_trace_id()
    # reasoning_steps[].langfuse_observation_id 는 step 별 sub-span 이 있으면 매핑
    # mixer 는 single call 이라 모든 step 의 observation_id = generation_id 동일
    gen_id = client.get_current_observation_id()
    for step in output_dict.get("reasoning_steps", []):
        step["langfuse_observation_id"] = gen_id
    # reasoning_trail 도 동일 (single call 기반)
    for trail_step in output_dict.get("reasoning_trail", []):
        trail_step["langfuse_observation_id"] = gen_id
    return output_dict
```

Multi-call agent (Briefing 의 section 별 LLM call) 는 각 section 의 generation_id 가 달라지므로 step 별 매핑이 더 풍부.

### 6.3 17 요소 prompt audit table

| # | 요소 | 충족 | 위치 |
|---|---|---|---|
| 1 | 역할 정의 | ✅ | "SK AX 사업전략팀의 멀티 카드 분석 전문가" |
| 2 | 추적 대상 | ✅ | "4 peer + 6 글로벌 + SK AX 자체" |
| 7 | 단순 요약 금지 | ✅ | event_type / 변화 / 시사점 패턴 |
| 10 | 수익화 관점 | ✅ | sk_ax_implication 의 긍정/중립/부정 |
| 11 | 정량 우선 | ✅ | "QoQ +X%" 형태 강제 |
| 12 | 공식 vs 추정 | ✅ | prefix 강제 |
| 13 | 전략 시사점 | ✅ | "SK AX 의 ___ 에 영향" pattern |
| 14 | 출력 형식 | ✅ | strict JSON schema |
| 15 | 우선순위 | ✅ | "다음 3 중 가장 영향" 명시 |
| 17 | 반복 추적 | ✅ | follow_up_questions |
| 5, 6, 8, 9 | (적용 권장이나 mixer 는 cross-card 분석이라 시계열 KPI 직접 X) | ⚪ | per-card 카드 본문이 가지고 옴 |
| 16 | 리스크 분석 | 🟡 | sk_ax_implication 에 "단 X 가정이 틀리면" 포함 권장 |

## 7. LLM 모델 + token 예산

- **모델**: gpt-4o (cross-card reasoning)
- 토큰/호출: ~5,000 (in 3,500 + out 1,500)
- 일일 호출: ~5
- **일일 비용**: ~₩325

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| card_ids < 2 | 400 BadRequest |
| card_ids > 20 | 400 (frontend 가 이미 제한, 보호용) |
| LLM JSON parse 실패 | retry 1회 → 실패 시 stub (insight="(생성 실패)", confidence=0.0) |
| connections 가 empty (single peer only 등) | OK — UI 가 "단일 흐름" 표시 |
| confidence < 0.5 | warning "조합이 산만함 — 카드 줄여보세요" |

## 9. 외부 의존성

- **DB**: `card_news` (READ), `analysis_cache` (UPSERT, key = card_ids hash)
- **외부 API**: OpenAI gpt-4o

## 10. State 흐름

AnalysisState 에서 `request_type='mixer'` 로 라우팅.

## 11. Provenance + Confidence

- **Provenance**: 자동 (ProvenanceTrackerAgent)
- **Confidence**: LLM self-confidence × card trust 평균

## 12. 테스트 시나리오

| Unit | 5 카드 (peer 다양) | insight 1문장 + connections ≥ 3 |
| Unit | 1 카드 | 400 BadRequest |
| Unit | 21 카드 | 400 |
| Edge | 모두 같은 cluster_id | connections label="similar" 다수 |

## 13. 모니터링

- KPI:
  - 평균 confidence ≥ 0.65
  - cache hit ratio ≥ 20%
  - 평균 latency ≤ 10초
- token: ₩325/일

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/mixer_analysis_agent.py` (신규 P7)

### Changelog

- **v1 (제안, P7)** — LLM cross-card + 6축 산식 + connection 라벨링
- **v2 (2026-05-14, 사업전략팀 추가 질의 회신 반영)** — CoT `reasoning_steps[]` 가시화,
  `final_one_liner` / `sk_ax_implication` / `follow_up_questions` 추가, 17 요소
  prompt audit. PDF §1 / §5 / §13 / §15 / §17 직접 대응. 비용 영향 ≤ 10%
  (single LLM call 내 explicit reasoning, out token ~500 증가).
