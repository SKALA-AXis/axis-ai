# ChatbotAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `ChatbotAgent` (Intent + Conversation + RAG 응답 통합) |
| **Supervisor** | DataUsageOrchestrator |
| **상태** | 🟡 backend fixture (`POST /api/assistant/chat`), axis-ai 신규 |
| **Trigger** | 2단계 데이터 활용 요청: frontend FloatingAiChat 위젯 메시지 |

## 2. 책임

**한 줄**: 저장된 Raw/정제 데이터, 카드뉴스, 시사점, ProfileContext를 기반으로 사용자 질문에 답한다.

**구체적 (3-phase orchestration)**:

1. **Intent 추출** (LLM mini, zero-shot) — search / summary / insight / mixer / it_trend / smalltalk + entity (peer/sector/date)
2. **Sub-component 호출** — intent 별 위임:
   - search → HybridSearchService + AnswerService
   - insight → InsightCascadeAgent
   - mixer → MixerAnalysisAgent
   - 기업/섹터 비교 → MixerAnalysisAgent 또는 InsightCascadeAgent
   - it_trend → ITTrendAgent
   - smalltalk → 직접 LLM 응답
3. **응답 생성** (LLM mini) — sub-component 결과 + 대화 톤 + follow-up 제안

## 3. 책임 NOT

- 실제 검색/분석 — sub-agent 들이 담당 (orchestrator 는 routing)
- 영구 history 저장 — `chat_sessions` 테이블 (cross-cutting Audit)

## 4. 입력 스펙

```python
class ChatTurnInput(TypedDict):
    message: str
    session_id: str           # 세션 식별 (frontend localStorage)
    history: list[dict]       # 마지막 10 turn (frontend 가 전달)

    # PDF 2026-05-14 §6 — 사용자 대화로 깊이 확장 (꼬리 물기)
    deep_dive_context: dict | None
    # {
    #   "parent_turn_idx": int,                 # 이 turn 이 어느 이전 turn 의 follow-up 인지
    #   "topic_anchor": str,                    # 이어서 깊이 파는 주제 (예: "삼성SDS Palantir 파트너십")
    #   "lens": Literal["technical","financial","competitive","regulatory","customer"] | None,
    #     # 사용자가 어떤 관점으로 깊이 파고 싶은지 (UI 버튼 또는 자유 선택)
    #   "iteration": int,                       # 같은 주제 몇 번째 deep dive 인지 (1~5)
    # }

    # PDF §6 — "에이전트에서도 지침이나, 지식모델을 추가하므로서 사용자가 원하는 인사이트"
    user_guidance: dict | None
    # {
    #   "persona": str | None,                  # "스타트업 관점" / "재무팀 관점" / "기술팀 관점"
    #   "depth": Literal["headline","detail","deep"] | None,  # 응답 깊이
    #   "knowledge_modules": list[str] | None,  # 사용자가 활성화한 추가 지식 모듈 ID
    # }
```

## 5. 출력 스펙

```python
class FollowUpSuggestion(TypedDict):
    """PDF 2026-05-14 §6 — "꼬리 물기" 구조화. 단순 string 이 아닌 typed suggestion
    으로 frontend 가 deep_dive_context 자동 구성 가능."""
    label: str                       # UI 버튼 라벨 (≤ 30자)
    intent: str                      # 클릭 시 trigger 할 intent
    deep_dive: bool                  # True 면 같은 topic 의 추가 깊이
    topic_anchor: str | None         # deep_dive 시 carry-over 할 주제
    lens: str | None                 # "technical" / "financial" / ... (옵션)

class ChatTurnOutput(TypedDict):
    reply: str                       # 대화체 응답 (인용 포함, [CN-...] 강제)
    intent: str                      # 분류된 intent
    entities: dict                   # {peer_id, sector, date_range, keywords}
    sources: list[dict]              # 사용된 카드 ids + urls
    follow_up_suggestions: list[FollowUpSuggestion]   # typed (PDF §6 — 꼬리 물기 구조화)
    final_one_liner: str | None      # smalltalk 외 모든 intent — SK AX 관점 한 줄 결론 (≤ 100자)
    sk_ax_implication: str | None    # 전략 시사점 (Insight / Mixer / Peer 위임 시)
    deep_dive_depth: int             # 현 turn 의 iteration (1 = 첫 질문, >1 = 깊이 확장)
    reasoning_steps: list[dict] | None   # sub-agent 가 CoT 반환하면 그대로 surface
    confidence: float
    session_id: str
    provenance: dict
```

frontend `POST /api/assistant/chat` 응답.

## 6. 알고리즘

### 6.1 Intent Router Prompt (gpt-4o-mini)

~~~text
# SK AX 사업전략팀 도우미 — Intent Router

당신은 SK AX 사업전략팀의 대화형 도우미입니다.
**사용자 메시지의 intent 와 entity 를 분류** 합니다.

## 입력 데이터
- **메시지**: {message}
- **이전 대화 (최근 3 turn)**: {history_short}
- **deep_dive_context**: {deep_dive_context 또는 "null"}

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **intent enum 만**: 9개 enum 외 X
- **entity trace**: peer_ids / sectors / keywords 는 메시지 또는 history 에 등장한 값만

### 일반 규칙 (17 요소 매핑)
1. **(#2 추적 대상)** peer_ids 는 4 국내 + 6 글로벌 enum 만
2. **(#3 추적 범위)** sectors 는 5 enum 만
3. **(#5 분석 기간)** date_range 의 since/until 은 절대 날짜 (YYYY-MM-DD)
4. **(#14 출력 형식)** strict JSON

## 가능한 intent

| intent | 예시 |
|---|---|
| **search** | `"삼성SDS AX 최근 동향"` |
| **summary** | `"오늘 핵심 변화 알려줘"` |
| **insight** | `"이 카드들로 인사이트 만들어줘"` |
| **mixer** | `"카드 조합 분석"` |
| **mixer** | `"삼성SDS vs LG CNS"` |
| **it_trend** | `"최근 글로벌 IT 흐름과 Peer사 동향 연결해줘"` |
| **deep_dive** | 이전 turn 주제 깊이 확장 (PDF §6 꼬리 물기), `deep_dive_context` 필수 |
| **alternative_view** | `"재무 관점에서 다시 봐줘"` (persona 변경) |
| **smalltalk** | 일반 대화 |

## 출력 형식 (strict JSON)

```json
{
  "intent": "search",
  "entities": {
    "peer_ids": [],
    "sectors": [],
    "date_range": {"since": "YYYY-MM-DD", "until": "YYYY-MM-DD"},
    "keywords": [],
    "card_ids": []
  },
  "confidence": 0.0
}
```
~~~

### 6.2 Routing

```python
async def orchestrate(message, session_id, history, deep_dive_context=None, user_guidance=None):
    intent_result = await llm_intent_router.classify(
        message, history, deep_dive_context=deep_dive_context
    )
    entities = intent_result["entities"]
    intent = intent_result["intent"]

    # Sub-component 위임
    sub_result = None
    if intent == "search":
        hits = await HybridSearchService().search(message, filters=build_filter(entities))
        reranked = await RerankService().rerank(message, hits.hits)
        sub_result = await AnswerService().answer(message, reranked.hits)
    elif intent == "insight":
        cards = entities.get("card_ids") or top_today_cards(6)
        sub_result = await InsightCascadeAgent().generate(cards, user_guidance=user_guidance)
    elif intent == "mixer":
        sub_result = await MixerAnalysisAgent().analyze(
            entities["card_ids"], ratios=default_ratios(), user_guidance=user_guidance,
        )
    elif intent == "it_trend":
        sub_result = await ITTrendAgent().generate(
            build_it_trend_input(entities, user_guidance=user_guidance)
        )
    elif intent == "deep_dive":
        # PDF §6 — 같은 topic 으로 깊이 확장. parent turn 의 sub_result 를 context 로 carry-over.
        sub_result = await deep_dive_handler(
            message=message,
            deep_dive_context=deep_dive_context,
            history=history,
            user_guidance=user_guidance,
        )
    elif intent == "alternative_view":
        # PDF §6 — 같은 데이터, 다른 persona (재무팀 / 기술팀 / 스타트업)
        parent_sub = _get_parent_sub_result(deep_dive_context, history)
        sub_result = await llm_reframe(
            parent_sub, persona=user_guidance.get("persona"),
        )
    else:  # smalltalk / summary
        sub_result = await llm_smalltalk(message, history)

    # 응답 생성 (대화체 변환)
    reply = await llm_chat_compose(sub_result, intent_result, message, user_guidance=user_guidance)

    # follow-up 제안 (PDF §6 꼬리 물기 — 구조화된 typed FollowUpSuggestion)
    follow_ups = generate_typed_follow_ups(intent_result, sub_result, deep_dive_context)

    # 세션 history 저장
    save_chat_turn(session_id, message, reply, intent_result, sub_result, deep_dive_context)

    return {
        "reply": reply,
        "intent": intent,
        "entities": entities,
        "sources": sub_result.get("sources", []),
        "follow_up_suggestions": follow_ups,
        "final_one_liner": sub_result.get("final_one_liner"),
        "sk_ax_implication": sub_result.get("sk_ax_implication"),
        "deep_dive_depth": (deep_dive_context.get("iteration", 0) + 1) if deep_dive_context else 1,
        "reasoning_steps": sub_result.get("reasoning_steps"),
        "confidence": min(intent_result["confidence"], sub_result.get("confidence", 1.0)),
        "session_id": session_id,
        "provenance": build_provenance(),
    }


async def deep_dive_handler(message, deep_dive_context, history, user_guidance):
    """PDF §6 — 사용자가 follow_up_suggestion 클릭 → 같은 topic 으로 반복 sub-agent invoke.

    iteration ≤ 5 까지 허용 (cost 통제 + UI fatigue 방지).
    각 iteration 은 parent turn 의 result 를 context 로 받아 *더 깊은 1 질문*만 답함.
    """
    if deep_dive_context["iteration"] > 5:
        return {"reply": "이 주제는 깊이 충분히 다뤘습니다. 새 주제는 어떠세요?",
                "follow_up_suggestions": _suggest_new_topics(history)}

    parent_sub = _get_parent_sub_result(deep_dive_context, history)
    topic = deep_dive_context["topic_anchor"]
    lens = deep_dive_context.get("lens")

    # parent_sub 의 reasoning_steps 에서 가장 confidence 낮은 step 또는 user_question 의 lens
    # 에 해당하는 sub-agent 를 다시 호출. 예:
    #   lens=financial → Mixer/Insight 의 재무 관점 재분석
    #   lens=technical → Mixer 의 tech_investment axis 재분석
    #   lens=competitive → Insight 의 Impact + Response 만 재생성
    return await _route_by_lens(topic, lens, parent_sub, message)


def generate_typed_follow_ups(intent_result, sub_result, deep_dive_context):
    """PDF §6 — 꼬리 물기 buttons. parent intent + result 기반."""
    intent = intent_result["intent"]
    suggestions = []

    if intent in ("insight", "mixer", "it_trend"):
        # 5 lens 모두 button 으로 노출 — 사용자 자유 선택
        topic = _extract_topic(sub_result)
        for lens in ("technical", "financial", "competitive", "regulatory", "customer"):
            suggestions.append({
                "label": f"{lens} 관점에서 깊이 분석",
                "intent": "deep_dive",
                "deep_dive": True,
                "topic_anchor": topic,
                "lens": lens,
            })
        # follow_up_questions 도 button 으로 (sub_result 가 제공)
        for q in sub_result.get("follow_up_questions", [])[:3]:
            suggestions.append({
                "label": q[:30], "intent": "deep_dive", "deep_dive": True,
                "topic_anchor": q, "lens": None,
            })

    return suggestions[:6]
```

### 6.3 Prompt audit — 02-prompt-design-checklist 17 요소

(ChatOrchestrator 필수: 1, 14, 17 / 위임된 sub-agent 가 나머지 충족)

| # | 요소 | 충족 위치 | 비고 |
|---|---|---|---|
| 1 | 역할 정의 | Intent Router prompt 도입부 + Compose prompt | "당신은 SK AX 사업전략팀 도우미" |
| 2 | 추적 대상 기업 | entity extraction (peer_ids enum) | Classification 과 동일 enum 재사용 |
| 3 | 추적 범위 | entity extraction (sectors enum) | 6 event × 5 sector taxonomy |
| 4 | 출처 우선순위 | sub-agent (Answer/Insight 등) 가 carry | reply 에 [CN-…] 인용 강제 |
| 5 | 분석 기간 | entity.date_range (since/until 절대 기준) | 상대 표현 ("최근") 금지 → 절대 변환 |
| 6 | 최신성 검증 | sub-agent carry — KST timestamp | smalltalk 외 모든 intent |
| 7 | 단순 뉴스 요약 금지 | 위임 sub-agent (Summary 도 NewsAnalysis 호출) | smalltalk 만 예외 |
| 8 | 회사별 비교 기준 | mixer → MixerAnalysis 위임 | KPI enum 동일 |
| 9 | 변화 감지 기준 | mixer / it_trend 결과에서 변화 시그널 표기 | ±5/10/30% band |
| 10 | 수익화 관점 | sub_result.sk_ax_implication surface | 응답에 명시 포함 |
| 11 | 정량 수치 우선 | sub-agent (Mixer / Insight / ITTrend) carry | reply 작성 시 prefer |
| 12 | 공식 vs 추정 구분 | sub-agent carry — 그대로 reply 에 surface | `[DART 2026-1Q]` prefix |
| 13 | 전략적 시사점 | sk_ax_implication 필드 명시 출력 | 모든 intent (smalltalk 제외) |
| **14** | **출력 형식** | **ChatTurnOutput TypedDict 명시** | **필수** |
| 15 | 우선순위 판단 | sub-agent (Insight / Mixer) carry | follow_up 도 top-3 만 |
| 16 | 리스크 분석 | Insight / ITTrend risk signal | deep_dive lens=regulatory 시 노출 |
| **17** | **반복 추적 구조** | **`follow_up_suggestions` (typed) + deep_dive_context carry-over + deep_dive_depth** | **필수 — PDF §6 꼬리 물기의 핵심** |

orchestrator 자체는 routing 이므로 sub-component 의 17 요소 충족을 reply 에 누락 없이 surface 하는 것이 책임. 위 표는 통과한다고 가정한 sub-component 결과를 어떻게 reply 에 노출하는지의 mapping.

### 6.4 Context Window 관리

- history 최근 10 turn 만 prompt 에 포함
- 그 이상은 LLM 요약 후 압축 ("이전 5턴은 ... 주제였음")
- `deep_dive_context.iteration > 5` 면 새 주제 제안 모드 (cost + UX 보호)

## 7. LLM 모델 + token 예산

| Phase | 모델 | token | 일일 호출 | 비용 |
|---|---|---|---|---|
| Intent Router | gpt-4o-mini | ~500 | ~50 | ₩30 |
| Chat compose | gpt-4o-mini | ~1,500 | ~50 | ₩200 |
| smalltalk (직접) | gpt-4o-mini | ~1,000 | ~10 | ₩30 |
| Sub-component 호출 | 별도 카운트 | | | |
| **합계 (orchestrator 만)** | | | | **~₩260/일** |

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| Intent classification fail | intent='search' fallback (보수적) |
| Sub-component fail | "분석 일시 불가, 다시 시도" reply |
| session_id null | 신규 session 생성 + 응답에 포함 |
| history > 10 turn | 요약 압축 |

## 9. 외부 의존성

- **외부 API**: OpenAI gpt-4o-mini
- **DB**: `chat_sessions` (UPSERT — 신규 **V10** migration; 현재 master V9 다음)
- **Sub-components**: HybridSearchService, RerankService, AnswerService, InsightCascade, MixerAnalysis, ITTrend

## 10. State 흐름

DialogueSupervisor (UserQuery 의 sub-supervisor) state:

```python
class DialogueState(TypedDict):
    session_id: str
    message: str
    history: list[dict]
    intent: str
    entities: dict
    sub_result: dict
    reply: str
    confidence: float
```

## 11. Provenance + Confidence

- **Provenance**: chat_sessions.metadata 에 turn 별 sub-agent 호출 chain 기록
- **Confidence**: min(intent_confidence, sub_agent_confidence)

## 12. 테스트 시나리오

| Unit | "삼성SDS AX 최근 동향" | intent='search', peer_ids=['samsung_sds'], sectors=['ax'] |
| Unit | "오늘 핵심 변화" | intent='summary' |
| Unit | "이 카드들로 인사이트" + card_ids | intent='insight' |
| Unit | "안녕" | intent='smalltalk' |
| Unit | "글로벌 IT 흐름과 Peer사 동향 연결해줘" | intent='it_trend' → ITTrendAgent |
| Unit | follow_up_suggestion 클릭 (lens=financial) | intent='deep_dive', deep_dive_context.iteration=2 |
| Unit | "재무 관점에서 다시" | intent='alternative_view', persona='재무팀' |
| Edge | 빈 메시지 | 400 BadRequest |
| Edge | history > 100 turn | 요약 적용 |
| Edge | deep_dive_context.iteration=6 | "이미 깊이 다뤘다" + 새 주제 제안 |

## 13. 모니터링

- KPI:
  - intent 분류 정확도 (sampling) ≥ 85%
  - 평균 turn latency ≤ 12초 (sub-agent 포함)
  - 평균 confidence ≥ 0.70
- token (orchestrator만): ₩260/일

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/chat_orchestrator_agent.py` (신규 P8)
- backend: `AssistantController.sendAssistantChatMessage` 의 fixture → axis-ai 위임

### Changelog

- **v1 (제안, P8)** — Intent + Routing + Compose 통합 단일 agent
- **v2 (2026-05-14)** — PDF 사업전략팀 추가 질의 회신 반영
  - 새 intent 3종: `forecast`, `deep_dive`, `alternative_view` (PDF §4, §6)
  - 입력에 `deep_dive_context`, `user_guidance` 추가 (꼬리 물기 + persona/지식 모듈)
  - 출력에 `final_one_liner`, `sk_ax_implication`, `deep_dive_depth`, `reasoning_steps` 추가
  - `follow_up_suggestions` 를 string → typed `FollowUpSuggestion` 으로 구조화 (5 lens 버튼)
  - §6.3 17-요소 audit table 추가 (3/17 직접 필수, 나머지는 sub-agent carry 정책)
