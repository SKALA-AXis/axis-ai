# InsightCascadeAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `InsightCascadeAgent` |
| **Supervisor** | DataUsageOrchestrator |
| **상태** | 🟡 backend fixture (`POST /api/insights/generate`, `GET /api/insights/latest`), axis-ai 신규 |
| **Trigger** | 2단계 데이터 활용 요청: User POST request, on-demand |

## 2. 책임

**한 줄**: Top 6 card_news ids 를 입력받아 **4단계 Cause → Change → Impact → Response** chain-of-thought 인사이트 자동 작성 + confidence 산출.

**구체적**:

1. 6 카드 → context 구성 (title + summary + sources + sector)
2. **Cause** — "왜 이런 변화가 일어났는가?" (3~5 bullet)
3. **Change** — "구체적으로 어떤 변화인가?" (3~5 bullet)
4. **Impact** — "SK AX 에 어떤 영향?" (3~5 bullet)
5. **Response** — "SK AX 의 대응 방안?" (3~5 bullet)
6. confidence — 카드 trust_score 평균 + LLM 자체 확신도 가중
7. 결과를 `analysis_cache` 에 7일 TTL 저장 (입력 card_id set hash 키)

## 3. 책임 NOT

- 카드 선정 — 사용자가 frontend 에서 Top 6 선택 (또는 자동 Top 6 = DerivedMetrics 의 top5+1)
- 출처 추적 — EvidenceBuilder 가 사전에 부착 (재사용)
- 시각화 — frontend Insight 페이지가 4-tab 렌더

## 4. 입력 스펙

```python
class InsightCascadeInput(TypedDict):
    card_ids: list[str]          # 6 권장, 4~10 허용
    context: dict | None         # frontend 가 추가 컨텍스트 제공 (예: 사용자 관심사)

    # 25-knowledge-curation 도입 후 추가 (Phase K3+)
    # 분석 agent 가 in-process ContextPackBuilder.assemble() 호출하여 자체 채움.
    # frontend / caller 는 채울 필요 없음 — agent 가 card_ids 의 peer_id 별 pack 자동 fetch.
    # cold_start (모든 pack layer 결측) 면 None 처리 + fallback path (기존 retrieval)
    _context_packs: dict[str, "PeerContextPack"] | None  # {peer_id: PeerContextPack}
```

## 5. 출력 스펙

```python
class ReasoningTrailItem(TypedDict):
    """02-prompt-design-checklist.md §4 Tier 1 — 사용자 default 노출.
    Insight 의 4 phase × 3~5 bullet = 15+ raw → trail 은 4~5 step 으로 압축."""
    seq: int
    label: str                          # ≤ 12자 ("배경 진단" / "Peer 행동" / "SK AX 영향" / "권장 대응" / "결론")
    one_liner: str                      # ≤ 80자, 가능 시 정량 1개
    evidence_refs: list[str]
    langfuse_observation_id: str | None

class CoTStep(TypedDict):
    """Tier 2 — 상세 ("더 자세히" 패널). Insight 4 phase + synthesis × 1+ step.
    PDF §1 / §5 직접 대응. confabulation 위험 있음 — admin 이 Tier 3 으로 검증."""
    step_idx: int
    phase: Literal["cause", "change", "impact", "response", "synthesis"]
    question: str
    inputs_used: list[str]              # card_id
    answer: str
    intermediate_conclusion: str
    confidence: float
    langfuse_observation_id: str | None

class InsightCascadeOutput(TypedDict):
    cause: list[str]
    change: list[str]
    impact: list[str]
    response: list[str]
    final_one_liner: str                # PDF 2026-05-14 §5 — SK AX 관점 한 줄 결론 (≤ 100자)
    sk_ax_implication: str              # 국내 IT 서비스사 관점 1~2 문장 (긍정/중립/부정 명시)
    reasoning_trail: list[ReasoningTrailItem]   # Tier 1 — 사용자 default (4~5)
    reasoning_steps: list[CoTStep]              # Tier 2 — 상세 (5~10)
    langfuse_trace_id: str | None               # Tier 3 — admin deep link
    follow_up_questions: list[str]      # PDF §17 — 다음 분석 제안 2~3개
    risk_assumptions: list[str]         # PDF §16 — 본 인사이트가 틀릴 가능성 / 가정
    confidence: float
    sources: list[dict]
    provenance: dict
    warning: str | None
```

frontend `POST /api/insights/generate` 응답.

## 6. 알고리즘

### 6.1 Context 구성

```python
def build_context(card_ids):
    cards = fetch_cards(card_ids)
    context_parts = []
    for c in cards:
        context_parts.append(f"""
[{c.id}] {c.title}
- Peer: {c.peer_id}
- Sector: {c.sector}
- Exposure: {c.exposure_band} ({c.exposure_score:.2f})
- 요약: {' / '.join(c.summary_lines)}
- 시사점: {c.implication.get('why_important','')}
""")
    return "\n".join(context_parts)
```

### 6.2 LLM Prompt (gpt-4o, 4-step CoT with explicit reasoning_steps)

PDF 2026-05-14 §1 / §5 직접 대응. 4단계 인사이트가 *그 자체로* CoT 흐름이지만,
사용자가 "agent 들 간 협업이 어떻게 이뤄졌는지" 확인하려면 단계 별 *추론 흔적*
이 별도 필요. single LLM call 안에서 4 phase 의 question / inputs / answer /
intermediate_conclusion 을 explicit 하게 출력.

~~~text
# SK AX 인텔리전스 분석가

당신은 SK AX 사업전략팀의 인텔리전스 분석가입니다.
본 task 는 **단순 답 생성이 아닌 *추론 과정의 명시적 노출*** — 사용자가 어떻게 결론에
도달했는지 UI 가 단계별로 보여줍니다.

## 입력 데이터

### 카드 뉴스
{context}

### Peer Context Packs (Phase K3+)
{context_packs_rendered 또는 "*cold start — pack 없음, recent_cards 만 사용*"}

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **카드 ID 근거**: 모든 bullet 에 `[CN-...]` 카드 ID 로 출처 인용 강제
- **환각 금지**: 출처에 없는 수치/이름 추가 시 즉시 `[자체 추정]` prefix
- **정량 보강**: 정성 표현 뒤에 정량 수치 (예: `"급성장 (QoQ +18.4%)"`)

### 일반 규칙 (17 요소 매핑)
1. **(#1 역할)** SK AX 사업전략팀 분석가 관점만
2. **(#2 추적 대상)** 카드 안의 4 국내 + 6 글로벌 + SK AX 자체에 한정
3. **(#4 출처 우선순위)** Tier1 (DART/IR) > Tier2 (대형 미디어) > Tier3 (Naver/RSS)
4. **(#7 단순 요약 금지)** event_type / 변화 / 시사점 패턴
5. **(#10 수익화 관점)** Impact bullet 에 `긍정/중립/부정` 명시
6. **(#11 정량 우선)** Change bullet 에 수치 / 날짜 / 제품명 우선
7. **(#12 출처 prefix)** `[공식 DART]` / `[기사 인용]` / `[자체 추정]`
8. **(#13 SK AX 화자)** Impact + Response 의 `"SK AX 의 ___ 에 영향"` pattern
9. **(#15 우선순위)** Response 3 액션 중 가장 영향 큰 1개를 `priority=1`
10. **(#16 리스크)** `risk_assumptions[]` 에 1~3개 명시
11. **(#17 반복 추적)** `follow_up_questions[]` 2~3개

## 추론 단계 (Chain of Thought)

### Phase 1 — Cause
- **자기 질문**: `"이 카드들이 발생한 배경 / 시장 환경 / Peer 의 전략적 motivation 은?"`
- **입력**: 카드 N건 + context_pack 의 weekly_digest / monthly_profile
- **출력**: cause bullet 3~5

### Phase 2 — Change
- **자기 질문**: `"Peer 가 실제로 어떤 행동/투자/제품을 했는가? (사실 위주)"`
- **입력**: 카드 본문 + DART refs
- **출력**: change bullet 3~5 (수치 / 날짜 / 제품명 우선)

### Phase 3 — Impact
- **자기 질문**: `"SK AX 의 사업·고객·경쟁 환경에 어떤 영향? (긍정/중립/부정 명시)"`
- **입력**: Phase 1 + 2 + context_pack.sk_ax_canonical_facts
- **출력**: impact bullet 3~5 (각각 prefix `긍정:` / `중립:` / `부정:`)

### Phase 4 — Response
- **자기 질문**: `"SK AX 가 취할 구체적 액션? 3 중 최고 1개?"`
- **입력**: Phase 3 + 자산/조직 컨텍스트
- **출력**: response 3개 (`priority` 1/2/3 부여)

### Phase 5 — Synthesis (final)
- **자기 질문**: `"위 4단계 종합 → 한 줄 결론 + 본 분석이 틀릴 가정?"`
- **출력**: `final_one_liner` + `risk_assumptions`

## 3-Tier Observability 출력

### Tier 1 — reasoning_trail (사용자 default)
4 phase × 3~5 bullet = 15+ step 의 raw 를 **정확히 4~5 step** 으로 압축.

권장 label sequence:
1. **"배경 진단"**
2. **"Peer 행동"**
3. **"SK AX 영향"**
4. **"권장 대응"**
5. **"결론"**

각 trail step: seq + label (≤ 12자) + one_liner (≤ 80자) + evidence_refs + langfuse_observation_id=null.
탐색/시도/hedging 금지.

### Tier 2 — reasoning_steps (상세)
phase=cause / change / impact / response / synthesis 각 1+ step. 총 5~10 step.

### Tier 3 — langfuse_trace_id
`null` 로 출력. `LangfuseTraceLinker` 미들웨어가 자동 매핑.

## 출력 형식 (strict JSON)

```json
{
  "cause": ["[CN-...] ...", "..."],
  "change": ["[공식 DART] [CN-...] ...", "..."],
  "impact": ["긍정: [CN-...] ...", "중립: ...", "부정: ..."],
  "response": [
    {"action": "...", "priority": 1, "rationale": "[CN-...]"},
    {"action": "...", "priority": 2, "rationale": "..."},
    {"action": "...", "priority": 3, "rationale": "..."}
  ],
  "final_one_liner": "≤ 100자, SK AX 관점, 모호 X",
  "sk_ax_implication": "1~2 문장. 긍정/중립/부정 명시.",
  "reasoning_trail": [
    {"seq": 1, "label": "배경 진단", "one_liner": "...", "evidence_refs": ["CN-..."], "langfuse_observation_id": null},
    {"seq": 2, "label": "Peer 행동", "one_liner": "...", "evidence_refs": ["CN-..."], "langfuse_observation_id": null},
    {"seq": 3, "label": "SK AX 영향", "one_liner": "...", "evidence_refs": ["CN-..."], "langfuse_observation_id": null},
    {"seq": 4, "label": "권장 대응", "one_liner": "...", "evidence_refs": ["CN-..."], "langfuse_observation_id": null},
    {"seq": 5, "label": "결론", "one_liner": "...", "evidence_refs": [], "langfuse_observation_id": null}
  ],
  "reasoning_steps": [
    {"step_idx": 0, "phase": "cause", "question": "...", "inputs_used": ["CN-..."],
     "answer": "...", "intermediate_conclusion": "...", "confidence": 0.0, "langfuse_observation_id": null}
  ],
  "follow_up_questions": ["...", "...", "..."],
  "risk_assumptions": ["본 인사이트가 ___ 가정에 의존. 그 가정이 틀리면 ___"],
  "confidence": 0.0,
  "sources_used": ["CN-..."],
  "uncertainties": ["..."]
}
```
~~~

### 6.2.1 Prompt 양식 audit — 02-prompt-design-checklist §6

| 양식 항목 | 충족 |
|---|---|
| `#` agent role 1개만 | ✅ |
| `##` 5 major section | ✅ |
| `###` sub-section heading | ✅ |
| 절대 규칙 + bold label | ✅ |
| numbered rule + (#N) inline | ✅ |
| JSON code fence | ✅ |
| `[Brackets]` 폐기 | ✅ |

> Single LLM call 이라 모든 trail/step 의 `langfuse_observation_id` 는 런타임에 동일 generation_id 로 매핑됨. `LangfuseTraceLinker` middleware (mixer §6.2.1 참조) 가 자동 처리.

### 6.3 17 요소 prompt audit table

| # | 요소 | 충족 | 위치 |
|---|---|---|---|
| 1 | 역할 정의 | ✅ | "SK AX 사업전략팀 인텔리전스 분석가" |
| 2 | 추적 대상 | ✅ | 4 peer + 6 글로벌 + SK AX |
| 4 | 정보 출처 우선순위 | ✅ | Tier1 > Tier2 > Tier3 명시 |
| 7 | 단순 요약 금지 | ✅ | 4-phase pattern + event 분류 |
| 10 | 수익화 관점 | ✅ | Impact bullet 의 긍정/중립/부정 |
| 11 | 정량 우선 | ✅ | Change 의 수치 / 날짜 / 제품명 |
| 12 | 공식 vs 추정 | ✅ | prefix 강제 |
| 13 | 전략 시사점 | ✅ | Impact / Response 의 "SK AX ___" pattern |
| 14 | 출력 형식 | ✅ | strict JSON schema |
| 15 | 우선순위 | ✅ | Response 의 priority=1 강제 |
| 16 | 리스크 분석 | ✅ | `risk_assumptions[]` 필수 |
| 17 | 반복 추적 | ✅ | `follow_up_questions[]` 필수 |
| 5, 6 | 분석 기간 / 최신성 | 🟡 | 카드의 published_at 가 가지고 옴 — prompt 에 명시 권장 |
| 8, 9 | 회사별 비교 / 변화 감지 | ⚪ | Insight 는 cross-card 추론 — 별도 카드 직접 비교 X |

### 6.3 Confidence 산출

```python
def compute_confidence(llm_self_confidence, card_trust_avg):
    return 0.5 * llm_self_confidence + 0.5 * card_trust_avg / 100.0
```

### 6.4 Cache 키

```python
import hashlib
def cache_key(card_ids):
    sorted_ids = sorted(card_ids)
    return hashlib.sha256(",".join(sorted_ids).encode()).hexdigest()[:16]
```

## 7. LLM 모델 + token 예산

- **모델**: gpt-4o (reasoning 깊이 필요)
- 토큰/호출: ~5,000 (in 3,000 + out 2,000)
- 일일 호출: ~10 (사용자 액션 가정)
- **일일 비용**: ~₩650

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| JSON parse 실패 | retry 1회 → 실패 시 503 응답 |
| 카드 1개 이하 | 400 BadRequest |
| LLM 응답에 "카드 ID 인용" 빠짐 | warning 부착, confidence -0.2 |
| confidence < 0.6 | warning 필드에 "근거 불충분 — 다른 카드 조합 권장" |
| `uncertainties` 비어있지 않음 | UI 에 표시 |

## 9. 외부 의존성

- **DB**: `card_news` (READ), `evidence_chain` (READ), `analysis_cache` (UPSERT)
- **외부 API**: OpenAI gpt-4o
- **lib**: `openai>=1.30`

## 10. State 흐름

AnalysisSupervisor 의 state (간소 — on-demand 단일 호출):

```python
class AnalysisState(TypedDict):
    request_type: Literal["insight","mixer","it_trend","link_verify"]
    card_ids: list[str]
    result: dict
    confidence: float
```

## 11. Provenance + Confidence

- **Provenance**: ProvenanceTrackerAgent 가 자동 부착
  ```json
  {"raw_article_ids": [...], "llm_model": "gpt-4o", "prompt_version": "insight-v1.0",
   "run_at": "...", "agent": "InsightCascadeAgent", "card_ids_input": [...]}
  ```
- **Confidence**: §6.3 산식

## 12. 테스트 시나리오

| Unit | 6 카드 (모두 sector='ax') | cause/change/impact/response 각 3+ bullets, confidence ≥ 0.6 |
| Unit | 카드 1개 | 400 BadRequest |
| Unit | confidence < 0.6 | warning 부착 |
| Edge | LLM 환각 (없는 회사명) | uncertainties 에 mark |
| Integration | 동일 card_id set 두 번 호출 | 두 번째는 cache hit (LLM 호출 0) |

## 13. 모니터링

- KPI:
  - 평균 confidence ≥ 0.70
  - cache hit ratio ≥ 30% (사용 패턴 따라)
  - 평균 latency ≤ 8초
- token 예산: ₩650/일

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/insight_cascade_agent.py` (신규 P7)
- backend wiring: `axis-backend/InsightController` 의 fixture → axis-ai 위임으로 교체

### Changelog

- **v1 (제안, P7)** — 4-step CoT + cache + confidence
- **v2 (2026-05-14, 사업전략팀 추가 질의 회신 반영)** — `reasoning_steps[]` 가시화,
  `final_one_liner` / `sk_ax_implication` / `follow_up_questions` /
  `risk_assumptions` 추가, Response 의 priority=1 강제, 17 요소 prompt audit
  (12/17 충족). PDF §1 / §5 / §10 / §13 / §15 / §16 / §17 직접 대응.
