# ChatOrchestratorAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `ChatOrchestratorAgent` (Intent + Conversation 통합) |
| **Supervisor** | UserQuery (Dialogue sub) |
| **상태** | 🟡 backend fixture (`POST /api/assistant/chat`), axis-ai 신규 |
| **Trigger** | frontend FloatingAiChat 위젯 메시지 |

## 2. 책임

**한 줄**: FloatingAiChat 의 user message → intent 분기 → sub-agent 호출 (Search/Insight/Mixer/Peer) → 대화 응답 생성 + history 보존.

**구체적 (3-phase orchestration)**:

1. **Intent 추출** (LLM mini, zero-shot) — search / summary / insight / mixer / peer_compare / smalltalk + entity (peer/sector/date)
2. **Sub-agent 호출** — intent 별 위임:
   - search → HybridSearchAgent + AnswerAgent
   - insight → InsightCascadeAgent
   - mixer → MixerAnalysisAgent
   - peer_compare → PeerComparisonAgent
   - smalltalk → 직접 LLM 응답
3. **응답 생성** (LLM mini) — sub-agent 결과 + 대화 톤 + follow-up 제안

## 3. 책임 NOT

- 실제 검색/분석 — sub-agent 들이 담당 (orchestrator 는 routing)
- 영구 history 저장 — `chat_sessions` 테이블 (cross-cutting Audit)

## 4. 입력 스펙

```python
class ChatTurnInput(TypedDict):
    message: str
    session_id: str           # 세션 식별 (frontend localStorage)
    history: list[dict]       # 마지막 10 turn (frontend 가 전달)
```

## 5. 출력 스펙

```python
class ChatTurnOutput(TypedDict):
    reply: str                 # 대화체 응답 (인용 포함)
    intent: str                # 분류된 intent
    entities: dict             # {peer_id, sector, date_range, keywords}
    sources: list[dict]        # 사용된 카드 ids + urls
    follow_up_suggestions: list[str]   # 다음 질문 제안 (UI 버튼)
    confidence: float
    session_id: str
    provenance: dict
```

frontend `POST /api/assistant/chat` 응답.

## 6. 알고리즘

### 6.1 Intent Router Prompt (gpt-4o-mini)

```text
다음 사용자 메시지의 intent 와 추출 가능한 entity 를 분류하라.

메시지: {message}
이전 대화 (최근 3 turn): {history_short}

가능한 intent:
- search: 키워드/주제 검색 (예: "삼성SDS AX 최근 동향")
- summary: 특정 카드/기간 요약 (예: "오늘 핵심 변화 알려줘")
- insight: 4단계 인사이트 (예: "이 카드들로 인사이트 만들어줘")
- mixer: 카드 조합 분석
- peer_compare: peer 비교 (예: "삼성SDS vs LG CNS")
- smalltalk: 일반 대화

JSON 출력:
{
  "intent": "...",
  "entities": {
    "peer_ids": [],
    "sectors": [],
    "date_range": {"since": "YYYY-MM-DD", "until": "YYYY-MM-DD"},
    "keywords": [],
    "card_ids": []   # 사용자가 명시한 카드 id
  },
  "confidence": 0.0~1.0
}
```

### 6.2 Routing

```python
async def orchestrate(message, session_id, history):
    intent_result = await llm_intent_router.classify(message, history)
    entities = intent_result["entities"]

    # Sub-agent 위임
    sub_result = None
    if intent_result["intent"] == "search":
        hits = await HybridSearchAgent().search(message, filters=build_filter(entities))
        reranked = await RerankAgent().rerank(message, hits.hits)
        sub_result = await AnswerAgent().answer(message, reranked.hits)
    elif intent_result["intent"] == "insight":
        cards = entities.get("card_ids") or top_today_cards(6)
        sub_result = await InsightCascadeAgent().generate(cards)
    elif intent_result["intent"] == "mixer":
        sub_result = await MixerAnalysisAgent().analyze(entities["card_ids"], ratios=default_ratios())
    elif intent_result["intent"] == "peer_compare":
        sub_result = await PeerComparisonAgent().compare(entities["peer_ids"][0])
    else:  # smalltalk / summary
        sub_result = await llm_smalltalk(message, history)

    # 응답 생성 (대화체 변환)
    reply = await llm_chat_compose(sub_result, intent_result, message)

    # follow-up 제안
    follow_ups = generate_follow_ups(intent_result, sub_result)

    # 세션 history 저장
    save_chat_turn(session_id, message, reply, intent_result, sub_result)

    return {
        "reply": reply,
        "intent": intent_result["intent"],
        "entities": entities,
        "sources": sub_result.get("sources", []),
        "follow_up_suggestions": follow_ups,
        "confidence": min(intent_result["confidence"], sub_result.get("confidence", 1.0)),
        "session_id": session_id,
        "provenance": build_provenance(),
    }
```

### 6.3 Context Window 관리

- history 최근 10 turn 만 prompt 에 포함
- 그 이상은 LLM 요약 후 압축 ("이전 5턴은 ... 주제였음")

## 7. LLM 모델 + token 예산

| Phase | 모델 | token | 일일 호출 | 비용 |
|---|---|---|---|---|
| Intent Router | gpt-4o-mini | ~500 | ~50 | ₩30 |
| Chat compose | gpt-4o-mini | ~1,500 | ~50 | ₩200 |
| smalltalk (직접) | gpt-4o-mini | ~1,000 | ~10 | ₩30 |
| Sub-agent 호출 | 별도 카운트 | | | |
| **합계 (orchestrator 만)** | | | | **~₩260/일** |

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| Intent classification fail | intent='search' fallback (보수적) |
| Sub-agent fail | "분석 일시 불가, 다시 시도" reply |
| session_id null | 신규 session 생성 + 응답에 포함 |
| history > 10 turn | 요약 압축 |

## 9. 외부 의존성

- **외부 API**: OpenAI gpt-4o-mini
- **DB**: `chat_sessions` (UPSERT — 신규 **V10** migration; 현재 master V9 다음)
- **Sub-agents**: HybridSearch, Rerank, Answer, InsightCascade, MixerAnalysis, PeerComparison

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
| Edge | 빈 메시지 | 400 BadRequest |
| Edge | history > 100 turn | 요약 적용 |

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
