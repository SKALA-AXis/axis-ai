# Prompt Design Checklist — 사업전략팀 추가 질의 회신 (2026-05-14) 반영

> **버전**: v1 (2026-05-14) · **상위 문서**: `axis-infra/docs/13팀_사업전략팀 추가 질의 회신.docx.pdf`
>
> SK AX 사업전략팀이 1차 데모 (frontend designing → develop) 검토 후 보낸 추가 피드백을 우리 LLM agent prompt 설계에 직접 반영한 17-요소 체크리스트. 모든 LLM-driven agent (Classification / NewsSummary / NewsAnalysis / IssueCard / Insight / Mixer / PeerComparison / GlobalTrends / Chat) 의 §6 algorithm / prompt 설계 시 본 checklist 17 항 모두 답할 수 있어야 함.

## 1. 배경 — PDF 피드백 핵심 6

| # | 피드백 | 우리 design 영향 |
|---|---|---|
| 1 | **멀티 agent 협업 과정 가시화** — 단일 agent 열거 X, 흐름 + 오케스트레이션 + CoT 루프 UI 노출 | 30-analysis/* + 60-briefing/* 의 output schema 에 `reasoning_steps[]` 추가 |
| 2 | **출처 + 논리 + 근거 명확화** — 1차원 Sheet 도 원본 출처 + 가공 로직 + 합리적 추론 | evidence_chain (이미 적용) + UI 측 메타 표시 |
| 3 | **Peer+ Overview** — 화면 설명 + 기준 시점 + 통합 비교 view | 20-enrichment/derived-metrics.md 에 `peer_overview` mode 추가 |
| 4 | **PeerComparison forecast** — DART 분기 + 동향 → 1Q / 반기 / 1년 후 시나리오 | 30-analysis/peer-comparison.md 에 forecast phase 추가 |
| 5 | **Mixer CoT 노출 + 최종 한 줄 결론** | 30-analysis/mixer-analysis.md output schema |
| 6 | **꼬리 물기 + 채팅형 인사이트** — 사용자 대화 깊이 확장 + 지식모델 / 지침 주입 | 40-user-query/chat-orchestrator.md 의 `deep_dive` mode |

## 2. 17 요소 체크리스트

각 LLM agent prompt 가 충족해야 하는 항목. 농도 (필수 / 권장 / 옵션) 는 agent 유형에 따라 다름.

| # | 요소 | 의미 | Agent 별 적용 |
|---|---|---|---|
| **1** | **역할 정의 (Role)** | "당신은 SK AX 사업전략팀의 ___" 으로 시작. 일반 도우미가 아닌 도메인 전문가로 정의 | **필수 — 모든 agent** |
| **2** | **추적 대상 기업** | 4 Peer (samsung_sds / lg_cns / hyundai_autoever / posco_dx) + 6 글로벌 (nvidia / apple / microsoft / google / amazon / meta) + SK AX 자체 | **필수 — Classification / NewsAnalysis / PeerComparison / GlobalTrends** |
| **3** | **추적 범위** | 6 event_type taxonomy (partnership / ma / personnel / tech / regulation / new_biz) × 5 sector (ax / security / infra / deal / other) | **필수 — Classification** |
| **4** | **정보 출처 우선순위** | Tier1 (DART / IR / 공식 뉴스룸) > Tier2 (대형 미디어 — 한경/매경) > Tier3 (Naver/RSS/Bloter 등) | **필수 — Evidence / NewsAnalysis** |
| **5** | **분석 기간** | window_days 명시 (기본 7일 / 30일 / 90일). 비교 시 동일 기간 적용 | **필수 — PeerComparison / DerivedMetrics / WeakSignal** |
| **6** | **최신성 검증** | published_at 의 KST 변환 + 분기 / 반기 / 1년 boundary 명시. "최신 정보" 라는 표현 X — "2026-1Q 기준" 같이 절대 기준 | **필수 — 모든 시계열 agent** |
| **7** | **단순 뉴스 요약 금지 (동향 분류 체계)** | "기사 N개 요약" 이 아닌 "이벤트 X 가 발생 → 시사점 Y" 패턴. event_type taxonomy 기반 분류 강제 | **필수 — NewsSummary / NewsAnalysis / Insight** |
| **8** | **회사별 비교 기준** | peer 별 KPI 명시 — 매출 / 영업이익 / 영업이익률 / Captive 비중 / 인력 / R&D 비중. 시계열은 QoQ / YoY 표기 | **필수 — PeerComparison / DerivedMetrics** |
| **9** | **변화 감지 기준** | 임계값 명시 — 매출 ±5% 가 normal / >10% 가 유의 / >30% 가 급변. 텍스트는 동일 키워드 빈도 ±50% 이상 | **필수 — WeakSignal / PeerComparison** |
| **10** | **수익화 관점** | "이 변화가 SK AX 매출 또는 마진에 어떻게 영향?" 항상 명시 (긍정 / 중립 / 부정) | **권장 — Insight / Mixer / PeerComparison** |
| **11** | **정량 수치 우선** | "성장 추세" → "QoQ +12.3%" / "흑자전환" → "영업이익 -120억 → +340억". 정성 표현 후 (정량) 보강 강제 | **필수 — 재무 연계 agent** |
| **12** | **공식 수치 vs 추정치 구분** | `[공식 DART 2026-1Q]` / `[자체 추정 v3 산식]` / `[기사 인용]` 같이 출처 prefix 강제 | **필수 — 모든 정량 출력** |
| **13** | **전략적 시사점 (국내 IT서비스사 관점)** | "삼성SDS 가 X 했다" 가 아니라 "삼성SDS 의 X 는 SK AX 의 Y 사업에 ___ 영향" — 화자 = SK AX | **필수 — Insight / Mixer / PeerComparison / GlobalTrends** |
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

## 4. CoT (Chain of Thought) reasoning_steps 표준

PDF §1, §5 의 "에이전트 간 소통 과정 가시화" 요구. multi-step LLM agent (Insight / Mixer / PeerComparison / Briefing) 의 output schema 에 추가:

```python
class CoTStep(TypedDict):
    step_idx: int                       # 0부터
    agent: str                          # 어느 sub-agent 또는 phase
    question: str                       # 이 step 의 자기 질문 (≤ 200자)
    inputs_used: list[str]              # card_id / DART id / 외부 근거 id
    answer: str                         # LLM 의 raw response (≤ 500자)
    intermediate_conclusion: str        # 다음 step 으로 넘어갈 핵심 요약 (≤ 150자)
    confidence: float                   # 0.0 ~ 1.0
```

frontend 가 expandable panel ("추론 과정 보기") 로 렌더. 사용자가 클릭 시 step 별 input → question → answer → conclusion 체인 표시.

## 5. 17 요소 ↔ agent 매핑 표

| Agent | 필수 (Mandatory) | 권장 (Recommended) |
|---|---|---|
| **ClassificationAgent** | 1, 2, 3, 4, 6, 7, 14 | 9 |
| **PeerNewsSummaryAgent** | 1, 2, 4, 6, 7, 11, 12, 14 | 16 |
| **PeerNewsAnalysisAgent** | 1, 2, 4, 6, 7, 10, 11, 13, 14 | 15, 17 |
| **IssueCardAgent** | 1, 2, 3, 4, 6, 7, 11, 12, 13, 14 | 15, 17 |
| **EvidenceAgent** | 1, 4, 11, 12, 14 | — |
| **InsightCascadeAgent** | 1, 2, 7, 10, 13, 14, 15, 16, 17 + CoT | 5, 8 |
| **MixerAnalysisAgent** | 1, 2, 7, 10, 13, 14, 15, 17 + CoT | 5, 8, 16 |
| **PeerComparisonAgent** | 1, 2, 5, 6, 8, 9, 11, 12, 13, 14, 16 + CoT | 10, 17 |
| **GlobalTrendsAgent** *(신규)* | **17 요소 모두** + CoT + final_one_liner | — |
| **AnswerAgent** | 1, 4, 7, 11, 12, 14 | 13, 17 |
| **ChatOrchestratorAgent** | 1, 14, 17 (deep_dive) | — |
| **WeakSignalAgent** | 1, 2, 5, 6, 9, 11, 14 | 13, 16 |
| **BriefingGenerationAgent** | 1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 17 + CoT | 15, 16 |

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
