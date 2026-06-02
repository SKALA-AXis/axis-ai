# Prompt Design Checklist — 사업전략팀 추가 질의 회신 (2026-05-14) 반영

> **버전**: v1 (2026-05-14) · **상위 문서**: `axis-infra/docs/13팀_사업전략팀 추가 질의 회신.docx.pdf`
>
> SK AX 사업전략팀이 1차 데모 (frontend designing → develop) 검토 후 보낸 추가 피드백을 우리 LLM agent prompt 설계에 직접 반영한 17-요소 체크리스트. 모든 LLM-driven agent (Classification / IssueIntegration / Analysis / CardNews / Insight / Mixer / ITTrend / Chat) 의 §6 algorithm / prompt 설계 시 본 checklist 17 항 모두 답할 수 있어야 함.

## 1. 배경 — PDF 피드백 핵심 6

| # | 피드백 | 우리 design 영향 |
|---|---|---|
| 1 | **멀티 agent 협업 과정 가시화** — 단일 agent 열거 X, 흐름 + 오케스트레이션 + CoT 루프 UI 노출 | 30-analysis/* + 60-briefing/* 의 output schema 에 `reasoning_steps[]` 추가 |
| 2 | **출처 + 논리 + 근거 명확화** — 1차원 Sheet 도 원본 출처 + 가공 로직 + 합리적 추론 | evidence_chain (이미 적용) + UI 측 메타 표시 |
| 3 | **Peer+ Overview** — 화면 설명 + 기준 시점 + 통합 비교 view | 20-enrichment/derived-metrics.md 에 `peer_overview` mode 추가 |
| 4 | **기업·섹터 흐름 비교** — 저장된 카드/분석 결과를 기업별·기간별로 비교 | Mixer / Insight 흐름에서 처리 |
| 5 | **Mixer CoT 노출 + 최종 한 줄 결론** | 30-analysis/mixer-analysis.md output schema |
| 6 | **꼬리 물기 + 채팅형 인사이트** — 사용자 대화 깊이 확장 + 지식모델 / 지침 주입 | 40-user-query/chat-orchestrator.md 의 `deep_dive` mode |

## 2. 17 요소 체크리스트

각 LLM agent prompt 가 충족해야 하는 항목. 농도 (필수 / 권장 / 옵션) 는 agent 유형에 따라 다름.

| # | 요소 | 의미 | Agent 별 적용 |
|---|---|---|---|
| **1** | **역할 정의 (Role)** | "당신은 SK AX 사업전략팀의 ___" 으로 시작. 일반 도우미가 아닌 도메인 전문가로 정의 | **필수 — 모든 agent** |
| **2** | **추적 대상 기업** | 4 Peer (samsung_sds / lg_cns / hyundai_autoever / posco_dx) + 6 글로벌 (nvidia / apple / microsoft / google / amazon / meta) + SK AX 자체 | **필수 — Classification / IssueIntegration / Analysis / ITTrend** |
| **3** | **추적 범위** | 6 event_type taxonomy (partnership / ma / personnel / tech / regulation / new_biz) × 5 sector (ax / security / infra / deal / other) | **필수 — Classification** |
| **4** | **정보 출처 우선순위** | Tier1 (DART / IR / 공식 뉴스룸) > Tier2 (대형 미디어 — 한경/매경) > Tier3 (Naver/RSS/Bloter 등) | **필수 — Evidence / NewsAnalysis** |
| **5** | **분석 기간** | window_days 명시 (기본 7일 / 30일 / 90일). 비교 시 동일 기간 적용 | **필수 — Mixer / DerivedMetrics / WeakSignal** |
| **6** | **최신성 검증** | published_at 의 KST 변환 + 분기 / 반기 / 1년 boundary 명시. "최신 정보" 라는 표현 X — "2026-1Q 기준" 같이 절대 기준 | **필수 — 모든 시계열 agent** |
| **7** | **단순 뉴스 요약 금지 (동향 분류 체계)** | "기사 N개 요약" 이 아닌 "이벤트 X 가 발생 → 시사점 Y" 패턴. event_type taxonomy 기반 분류 강제 | **필수 — NewsSummary / NewsAnalysis / Insight** |
| **8** | **회사별 비교 기준** | peer 별 KPI 명시 — 매출 / 영업이익 / 영업이익률 / Captive 비중 / 인력 / R&D 비중. 시계열은 QoQ / YoY 표기 | **필수 — Mixer / DerivedMetrics** |
| **9** | **변화 감지 기준** | 임계값 명시 — 매출 ±5% 가 normal / >10% 가 유의 / >30% 가 급변. 텍스트는 동일 키워드 빈도 ±50% 이상 | **필수 — WeakSignal / Mixer** |
| **10** | **수익화 관점** | "이 변화가 SK AX 매출 또는 마진에 어떻게 영향?" 항상 명시 (긍정 / 중립 / 부정) | **권장 — Insight / Mixer** |
| **11** | **정량 수치 우선** | "성장 추세" → "QoQ +12.3%" / "흑자전환" → "영업이익 -120억 → +340억". 정성 표현 후 (정량) 보강 강제 | **필수 — 재무 연계 agent** |
| **12** | **공식 수치 vs 추정치 구분** | `[공식 DART 2026-1Q]` / `[자체 추정 v3 산식]` / `[기사 인용]` 같이 출처 prefix 강제 | **필수 — 모든 정량 출력** |
| **13** | **전략적 시사점 (국내 IT서비스사 관점)** | "삼성SDS 가 X 했다" 가 아니라 "삼성SDS 의 X 는 SK AX 의 Y 사업에 ___ 영향" — 화자 = SK AX | **필수 — Insight / Mixer / ITTrend** |
| **14** | **출력 형식** | JSON schema 명시 + 필수 / 옵션 필드 / enum 값 / 길이 제약. 항상 검증 가능 | **필수 — 모든 agent** |
| **15** | **우선순위 판단 기준** | "다음 3 항목 중 가장 영향 큰 것 1개 선택: A/B/C". 모호한 "중요한 것" X | **권장 — Insight / Mixer** |
| **16** | **리스크 분석** | "이 가정이 틀릴 가능성: ___ (낮음/중간/높음)" + "그 경우 대응: ___" | **권장 — Forecast / Insight** |
| **17** | **반복 추적 구조** | "다음 분석 시 추가로 확인할 항목: ___" 항상 후속 actions 제공 (꼬리 물기) | **필수 — Insight / Mixer / Chat** |

## 3. 추가 — 모든 agent 의 "최종 한 줄 결론"

PDF §3, §5 의 직접 요구: **모든 LLM output 의 마지막에 `final_one_liner` (한 문장 ≤ 100자) 강제**.

```python
class AnyLLMOutput(TypedDict):
    ...  # agent 별 schema
    final_one_liner: str   # 최대 100자. SK AX 관점의 한 줄 결론. 모호함 금지.
```

예시:
- 좋음: `"삼성SDS 의 Palantir 파트너십은 SK AX 의 manufacturing AX 입찰 경쟁을 강화한다."`
- 나쁨: `"의미 있는 변화로 보인다."` ← 모호

## 4. CoT (Chain of Thought) — 3-tier observability 표준

PDF §1, §5 의 "에이전트 간 소통 과정 가시화" 요구. 단, 사용자 대화형 (Mixer / Chat) 외에는 모두 **batch** 결과를 보여주는 형태이므로 raw LLM 추론 trace 를 그대로 노출하면 노이즈가 크다 (수많은 시도 / 탐색 token / 자기 반복). 따라서 **post-hoc 정제된 narrative** 가 default 이고, 상세 추적은 옵션이어야 한다.

본 design 은 **3-tier observability** 로 표준화:

### Tier 1 — `reasoning_trail[]` (사용자 default 노출)

분석 결과 페이지 "추론 흐름" 패널에 표시될 **압축 narrative**. 동일 LLM call 내에서 self-summarize 강제 (별도 LLM call 아님 — 추가 비용 ~200 token / ~₩50).

```python
class ReasoningTrailItem(TypedDict):
    seq: int                            # 1부터
    label: str                          # 짧은 단계명, ≤ 12자 ("카드 비교" / "재무 매칭" / "결론")
    one_liner: str                      # ≤ 80자 한국어 단문, 핵심만 + 가능 시 정량 수치 1개 포함
    evidence_refs: list[str]            # card_id / DART id / 외부 근거 id (UI 클릭 시 원본 노출)
    langfuse_observation_id: str | None # 이 step 에 대응하는 Langfuse span/generation id (admin only)
```

- **분량**: 3~5 step 강제. raw CoT 가 15 step 이라도 trail 은 4~5 로 압축
- **언어**: 한국어 단문, 추측 / 탐색 / "고민했으나" 같은 hedging 금지
- **정량 우선** (checklist 11): 가능하면 step 의 one_liner 에 수치 1개 (`"QoQ +12%"`)
- **PDF §5 직접 대응**: 최종 한 줄 결론 (`final_one_liner`) 의 *어떻게* 부분을 채우는 layer

### Tier 2 — `reasoning_steps[]` ("더 자세히" 접힌 패널)

기존 CoT — 단계별 question / inputs / answer / conclusion / confidence. trail 이 압축본이라면 steps 는 원본 narrative.

```python
class CoTStep(TypedDict):
    step_idx: int                       # 0부터
    agent: str                          # 어느 sub-agent 또는 phase
    phase: str | None                   # ex: "per_card" / "cross_card" / "synthesis"
    question: str                       # ≤ 200자
    inputs_used: list[str]              # card_id / DART id / 외부 근거 id
    answer: str                         # ≤ 500자
    intermediate_conclusion: str        # ≤ 150자
    confidence: float                   # 0.0~1.0
    langfuse_observation_id: str | None # 이 step 에 대응하는 Langfuse span id
```

- 사용자가 trail 의 step 한 줄을 클릭 → 동일 seq 에 매핑된 steps 가 expand
- LLM 의 self-narration 이라 confabulation 위험 존재 → admin 이 Tier 3 으로 검증

### Tier 3 — `langfuse_trace_id` (admin only deep link)

LangChain CallbackHandler 가 자동 캡처한 *실제 실행 trace* — 원본 prompt + response + token + latency + cost. **사용자에게는 노출 X**.

```python
class LangfuseTraceRef(TypedDict):
    trace_id: str                       # Langfuse Cloud 의 trace id
    project: str                        # "axis-ai"
    deep_link: str                      # https://cloud.langfuse.com/project/{project}/traces/{trace_id}
```

- frontend admin panel 만 `LangfuseTraceRef` surface
- 일반 사용자 응답은 trace_id 미포함 (또는 비공개 prefix)
- multi-agent orchestration (Briefing 10 section, Insight → Mixer 위임 등) 의 *진짜* 협업 흐름은 Langfuse span tree 가 ground truth

### 3-tier 간 책임 분담 (요약)

| 질문 | 어디서 답하나 |
|---|---|
| "이 시사점이 어떻게 나왔어?" (사용자) | **Tier 1** trail 3~5 step |
| "step 3 의 근거가 뭐야?" (사용자, 클릭) | **Tier 2** steps[seq=3] expand |
| "agent A 가 어떤 prompt 로 LLM 호출했고 token 얼마 썼어?" (admin) | **Tier 3** Langfuse 원본 |
| "이번 분석이 실제로 어떤 카드들을 봤어?" (검증) | Tier 2 inputs_used vs Tier 3 prompt 대조 |

### `AnyLLMOutput` 표준 schema

multi-step LLM agent (Insight / Mixer / ITTrend / Briefing) 의 output 은 다음 3 필드 모두 포함:

```python
class AnyLLMOutput(TypedDict):
    ...  # agent 별 schema
    reasoning_trail: list[ReasoningTrailItem]   # Tier 1 — 사용자 default (3~5)
    reasoning_steps: list[CoTStep]              # Tier 2 — 상세 ("더 자세히")
    langfuse_trace_id: str | None               # Tier 3 — admin deep link
    final_one_liner: str                        # ≤ 100자, SK AX 관점
```

### Tier 1 생성 prompt 조각 (재사용 표준)

모든 multi-step agent prompt 끝에 다음 instruction 박음:

```text
[reasoning_trail 출력]
사용자 UI 에 표시할 추론 흐름을 정확히 3~5 step 으로 작성하라.
raw reasoning_steps 가 더 길어도 압축할 것. 핵심 결정만 trail.

각 trail step:
- seq: 1, 2, 3, ...
- label: ≤ 12자 명사구 ("카드 비교" / "재무 검증" / "결론")
- one_liner: ≤ 80자 한국어 단문. 가능하면 정량 수치 1개 포함.
               탐색/시도/hedging 표현 금지 ("~ 인 것 같다" X).
- evidence_refs: 이 step 이 참조한 card_id / DART id / IR ref 목록
- langfuse_observation_id: null (런타임에 자동 매핑됨)

목표: 사용자가 trail 만 읽고도 "왜 이 결론에 왔는가" 가 명확히 보이도록.
```

### 적용 우선순위 (가치 큰 순)

| Agent | trail 가치 | 비고 |
|---|---|---|
| **MixerAnalysis** | HIGH | PDF §5 직접 요구 (CoT 노출 + 한 줄 결론) |
| **InsightCascade** | HIGH | 4 phase × 3~5 bullet = 15+ raw → 4~5 trail |
| **ITTrend** | HIGH | SPRi/BCG 자료 + 글로벌 뉴스룸 통합/분석 결과로 TrendContext 생성·갱신 |
| **BriefingGeneration** | MEDIUM | section 별 mini-trail (10 section × 3 step) 권장 |
| **ChatOrchestrator** | LOW | 실시간 turn 이라 streaming 으로 충분, trail 보다 follow_up_suggestions 가 효과적 |
| Ingestion agents | NONE | 사용자 노출 안 함 |

## 5. 17 요소 ↔ agent 매핑 표

| Agent | 필수 (Mandatory) | 권장 (Recommended) | 3-tier observability |
|---|---|---|---|
| **ClassificationService** | 1, 2, 3, 4, 6, 7, 14 | 9 | trace_id only |
| **IntegrationAgent** | 1, 2, 4, 6, 7, 11, 12, 14 | 16 | trace_id only |
| **AnalysisAgent** | 1, 2, 4, 6, 7, 10, 11, 13, 14 | 15, 17 | trace_id only |
| **CardNewsAgent** | 1, 2, 3, 4, 6, 7, 11, 12, 13, 14 | 15, 17 | trace_id only |
| **EvidenceBuilder** | 1, 4, 11, 12, 14 | — | trace_id only |
| **InsightCascadeAgent** | 1, 2, 7, 10, 13, 14, 15, 16, 17 + CoT | 5, 8 | **trail + steps + trace_id** |
| **MixerAnalysisAgent** | 1, 2, 7, 10, 13, 14, 15, 17 + CoT | 5, 8, 16 | **trail + steps + trace_id** |
| **ITTrendAgent** | 1, 2, 5, 6, 9, 11, 13, 14, 16 + CoT | 10, 17 | **trail + steps + trace_id** |
| **AnswerService** | 1, 4, 7, 11, 12, 14 | 13, 17 | trace_id only |
| **ChatOrchestratorAgent** | 1, 14, 17 (deep_dive) | — | trace_id only (streaming) |
| **WeakSignalAgent** | 1, 2, 5, 6, 9, 11, 14 | 13, 16 | trace_id only |
| **BriefingGenerationService** | 1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 17 + CoT | 15, 16 | **trail (section 별) + steps + trace_id** |

> 사용자 UI default 노출은 trail. steps 는 "더 자세히". trace_id 는 admin only deep link.
> trace_id only 인 agent 는 사용자에 직접 결과 노출되지 않거나 (ingestion) 결과가 stream 형태 (chat) 이므로 trail 가치 낮음.

## 6. Prompt audit 진행 plan

각 agent 의 `§6. 알고리즘` 의 prompt template 을 본 checklist 로 자가 검증:

```
[Agent X 의 prompt audit]
- 1 (역할 정의): ✅ "당신은 SK AX 전략기획팀의 ..."
- 7 (단순 뉴스 요약 금지): ✅ event_type taxonomy + sector 강제
- 17 (반복 추적 구조): ❌ follow_up_questions 없음 → 추가
- ...
```

audit 결과는 각 agent 파일의 §6 (또는 §14 Changelog) 에 audit table 으로 표시.

## 6. Prompt 작성 양식 표준 (2026-05-14 추가)

### 6.1 왜 양식 표준이 필요한가

LLM (특히 gpt-4o) 은 입력 prompt 의 **구조적 신호** 에 강하게 반응한다. 같은 내용이라도:

- `[입력]` bracket 형식 → 약한 boundary 신호 (LLM 의 attention 가중치 낮음)
- `## 입력` markdown heading → 명확한 boundary + section recall ↑
- `**핵심 규칙**: ...` bold → 절대 규칙 시각적 + attention 강화
- ```` ```json ... ``` ```` code fence → strict format following ↑

따라서 본 design 의 모든 LLM prompt 는 **markdown 계층 + bold + structured list + code fence** 조합으로 작성한다. `[Brackets]` style 은 deprecated.

### 6.2 표준 양식

모든 LLM agent 의 prompt 는 다음 5 section 구조로:

```text
# {Role}

{1~2 문장 역할 정의}

## 입력 데이터

### {input subsection 1}
{content with {placeholders}}

### {input subsection 2}
...

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **{rule label}**: 설명
- **{rule label}**: 설명

### 일반 규칙 (17 요소 매핑)
1. (#N) 규칙 설명
2. (#N) 규칙 설명

## 추론 단계

### Phase 1 — {phase name}
- **자기 질문**: "..."
- **입력**: ...
- **출력**: ...
- **intermediate_conclusion**: ...

### Phase 2 — {phase name}
...

## 3-Tier Observability 출력

### Tier 1 — reasoning_trail (사용자 default)
정확히 {N}~{M} step. label ≤ 12자, one_liner ≤ 80자. 탐색/시도/hedging 금지.

### Tier 2 — reasoning_steps (상세)
{N}~{M} step. question / inputs_used / answer / intermediate_conclusion / confidence.

## 출력 형식 (strict JSON)

```json
{
  "field": "..."
}
```
```

### 6.3 heading hierarchy 의미

| Level | 용도 | 개수 (권장) |
|---|---|---|
| `#` | **Agent role** — prompt 최상위, 1개만 | 1 |
| `##` | **Major section** — 입력 / 규칙 / 추론 / 출력 등 | 4~6 |
| `###` | **Sub-section** — Phase 별 / 규칙 카테고리 / Tier 등 | 8~15 |
| `####` | (사용 안 함) | 0 |

`#` 가 너무 많으면 LLM 이 *어떤 게 진짜 중요한지* 혼란. 1 prompt = 1 `#` 원칙.

### 6.4 강조 표현 우선순위 (위로 갈수록 강함)

1. **`### 절대 규칙 (위반 시 응답 무효)`** + bold label — 최강 신호
2. **bold label**: numbered list 안의 `**규칙명**: 설명`
3. **bold inline**: 본문 안의 `**핵심 단어**`
4. *italic*: 부가 설명 (드물게)
5. plain text

> 너무 많은 bold 는 신호 약화 — *진짜 critical* 한 것만. prompt 당 bold 5~10 회 권장.

### 6.5 List 사용 가이드

| List 종류 | 용도 | 예시 |
|---|---|---|
| numbered (1./2./3.) | **순차** 규칙 / phase / 17 요소 매핑 | "1. (#7) 단순 요약 금지" |
| bullet (-) | **항목 나열** (순서 무관) | "- **DART 출처**: [DART rcept_no=...]" |
| nested bullet | sub-detail | `- **규칙**: 설명\n    - 예외 1\n    - 예외 2` |

### 6.6 Code fence 사용

| 용도 | fence |
|---|---|
| **JSON output schema** | ` ```json ` |
| **Python sub-code** (예: 전처리 산식) | ` ```python ` |
| **자유형 예시** | ` ```text ` |
| **placeholder block** (input value rendering) | 들여쓰기 4 spaces 또는 plain text |

JSON schema 는 *반드시* ` ```json ` fence — LLM 의 JSON-mode 신호 강화.

### 6.7 Placeholder 표기

| 표기 | 의미 |
|---|---|
| `{var_name}` | runtime 에 채워질 값 (Python f-string 호환) |
| `{var.field}` | nested object field |
| `{var:format}` | format spec (예: `{score:.2f}`) |
| `<...>` | (사용 안 함 — XML tag 와 혼동) |
| `[...]` | (placeholder 로 사용 안 함 — markdown link / 출처 marker 와 혼동) |

### 6.8 17 요소 inline 매핑

prompt 의 numbered rule 에 17 요소 번호 prefix:

```text
### 일반 규칙 (17 요소 매핑)
1. **(#7 단순 요약 금지)** "기사 N개" 표현 금지 — event_type / 변화 / 시사점 패턴
2. **(#9 변화 감지)** ±5% normal / >10% 유의 / >30% 급변 분류
3. **(#11 정량 우선)** "성장 추세" 금지 → "QoQ +12.3%"
4. **(#12 공식 vs 추정)** 모든 정량에 `[DART 2026-1Q]` / `[card: CN-...]` / `[자체 추정 v1]` prefix
5. **(#13 SK AX 화자)** "삼성SDS 가 X" 금지 → "삼성SDS X 는 SK AX 의 ___ 라인에 ___ 영향"
```

각 agent 의 §6.4 prompt audit table 의 17 요소 매핑이 prompt 자체에 *명시적으로* 들어가야 LLM 이 규칙을 정확히 따른다.

### 6.9 Anti-pattern (deprecated 양식)

| ❌ Bad | ✅ Good | 이유 |
|---|---|---|
| `[입력]` `[규칙]` `[출력]` | `## 입력` `## 규칙` `## 출력` | bracket = 약한 boundary |
| `Phase 1 — per_card:` | `### Phase 1 — per_card` | heading 으로 명시 |
| `중요: 환각 금지` | `### 절대 규칙` + `- **환각 금지**: ...` | label + bold |
| numbering 없는 자유 텍스트 규칙 | numbered list `1. (#N) ...` | LLM 의 enumeration recall |
| JSON 예시를 plain text 로 | ` ```json ... ``` ` | JSON-mode 신호 |
| `{변수명}` 와 `<var>` 혼용 | 일관 `{var_name}` | parsing 일관성 |

### 6.10 양식 적용 audit (2026-05-14 신규)

각 agent 의 §6.x LLM prompt 가 본 표준을 따르는지 audit. table 형식:

| 양식 항목 | 충족 |
|---|---|
| `#` agent role 1개만 | ✅ / 🟡 / ❌ |
| `##` 4~6 major section | ✅ / 🟡 / ❌ |
| `###` sub-section + heading 명시 | ✅ / 🟡 / ❌ |
| 절대 규칙 + bold label | ✅ / 🟡 / ❌ |
| numbered rule + (#N) inline | ✅ / 🟡 / ❌ |
| JSON code fence | ✅ / 🟡 / ❌ |
| `[Brackets]` 폐기 | ✅ / 🟡 / ❌ |

audit 결과는 agent 파일의 §6.x 마지막에 추가.

## 7. 갱신 우선순위 (PDF 직접 영향 큰 순)

| # | 갱신 대상 | 영향 |
|---|---|---|
| 1 | `30-analysis/mixer-analysis.md` | CoT + 17 audit + final_one_liner |
| 2 | `30-analysis/insight-cascade.md` | CoT + 17 audit + 4단계 명확화 |
| 3 | `30-analysis/peer-comparison.md` | forecast phase 추가 + CoT + 17 audit |
| 4 | `40-user-query/chat-orchestrator.md` | deep_dive mode + 꼬리 물기 |
| 5 | `20-enrichment/derived-metrics.md` | peer_overview mode + 기준 시점 metadata |
| 6 | **`30-analysis/global-trends.md`** (신규) | 17 요소 전부 적용 — PDF 4 페이지의 예시 직접 대응 |
| 7 | 기존 ingestion agent (Classification / NewsSummary / NewsAnalysis / IssueCard) | §6 prompt audit table 추가 |

## 8. Changelog

- **v1 (2026-05-14)** — PDF 추가 질의 회신 반영 신설
