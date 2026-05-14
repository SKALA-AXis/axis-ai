# AnswerAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `AnswerAgent` |
| **Supervisor** | UserQuery |
| **상태** | ✅ stub (`/gen-search` handler), P6 활성 |
| **Trigger** | `POST /gen-search` (Generative Search) |

## 2. 책임

**한 줄**: 재정렬된 top 10 카드 + 쿼리 → LLM (gpt-4o) generative answer + 자가 일관성 (SC ×3) 검증.

**구체적**:

1. top 10 hits → context 구성
2. LLM 호출 ×3 (temperature 0.3, 동일 prompt) — Self-Consistency
3. 3 응답의 핵심 주장 LLM 비교 (judge prompt)
4. 일치율 ≥ 2/3 → Pass + 최빈 응답 선택
5. 일치율 < 2/3 → Fail → "근거 부족" stub
6. 출처 카드 ID 인라인 인용 강제

## 3. 책임 NOT

- 검색 — HybridSearchAgent + RerankAgent (이전)
- 카드 ranking — exposure_score 별도 산식

## 4. 입력 스펙

```python
class AnswerInput(TypedDict):
    query: str
    hits: list[RerankedHit]   # top 10
    sc_iterations: int         # 기본 3 (SC)
```

## 5. 출력 스펙

```python
class AnswerOutput(TypedDict):
    answer: str               # 자연어 답변 (인용 포함)
    sources: list[dict]       # 사용된 card_ids + url
    sc_passed: bool           # 일치율 ≥ 2/3
    sc_score: float           # 일치율 0.0~1.0
    confidence: float
    provenance: dict
    warning: str | None       # sc_passed=false 시
```

frontend `POST /gen-search` 응답.

## 6. 알고리즘

### 6.1 LLM Prompt (gpt-4o, temperature 0.3)

~~~text
# SK AX 지능형 검색 도우미

당신은 SK AX 사업전략팀의 지능형 검색 도우미입니다.
**참고 카드 뉴스 (top 10) 만 근거로** 질문에 답합니다.

## 입력 데이터
- **질문**: {query}
- **참고 카드 (top 10)**: {hits_summary}

## 작성 규칙

### 절대 규칙 (위반 시 응답 무효)
- **카드 trace**: 카드에 명시된 사실만 사용 — 가정/추측 금지
- **인용 강제**: 답변 안에 `[CN-...]` 형식으로 카드 id 인용
- **fallback 명시**: 카드들로 답할 수 없으면 `"제공된 자료로는 답변 곤란"` 출력

### 일반 규칙 (17 요소 매핑)
1. **(#4 출처 우선순위)** 카드의 source_links carry — Tier1 (DART/IR) 우선 인용
2. **(#7 단순 요약 금지)** 질문에 답 — 카드 요약 X
3. **(#11 정량 우선)** 핵심 수치/날짜는 카드 원문 직접 인용
4. **(#12 출처 prefix)** 카드 raw 의 `[DART]` / `[기사 인용]` carry
5. **(#14 출력 형식)** strict JSON

## 출력 형식 (strict JSON)

```json
{
  "answer": "...",
  "key_facts": ["...", "..."],
  "sources_used": ["CN-...", "CN-..."]
}
```
~~~

### 6.2 Self-Consistency (SC) — 3회 호출

```python
def gen_answer_sc(query, hits, sc_iter=3):
    responses = []
    for _ in range(sc_iter):
        r = llm.generate(prompt(query, hits), temperature=0.3)
        responses.append(json.loads(r))

    # judge: 3 응답의 key_facts 일치 여부
    judge_prompt = build_judge_prompt(responses)
    judge_result = llm.generate(judge_prompt, temperature=0.0)
    agreement = parse_agreement(judge_result)  # 0.0~1.0

    if agreement >= 0.67:  # 2/3
        # 최빈 응답 (key_facts 가장 자주 등장) 선택
        chosen = mode(responses)
        return {
            "answer": chosen["answer"],
            "sources": [{"id": cid, "url": find_url(cid)} for cid in chosen["sources_used"]],
            "sc_passed": True,
            "sc_score": agreement,
            "confidence": min(agreement, 1.0),
        }
    else:
        return {
            "answer": "제공된 자료로 일관된 답변 도출 불가. 추가 카드 또는 쿼리 재구성 필요.",
            "sources": [],
            "sc_passed": False,
            "sc_score": agreement,
            "confidence": 0.0,
            "warning": "SC validation failed (agreement < 0.67)",
        }
```

## 7. LLM 모델 + token 예산

- **모델**: gpt-4o (3× SC + 1× judge = 4 calls)
- 토큰/호출: ~5,000
- 일일 호출: ~20 generative search × 4 = 80
- **일일 비용**: ~₩400

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| LLM JSON parse fail (1회) | 해당 iter skip, 나머지로 SC 진행 |
| 3회 모두 fail | sc_passed=false |
| LLM timeout 5초 | retry 1회 → 실패 시 fallback "검색 결과만 표시" |
| hits 0건 | "검색 결과 없음" stub |

## 9. 외부 의존성

- **외부 API**: OpenAI gpt-4o
- **DB**: card_news (read for url)

## 10. State 흐름

UserQueryState 의 `answer`, `sc_iter`, `confidence`.

## 11. Provenance + Confidence

- **Provenance**: response.metadata.llm_model='gpt-4o', sc_iter, prompt_version
- **Confidence**: sc_score 직접

## 12. 테스트 시나리오

| Unit | 10 hits, 쿼리 "삼성SDS AX" | answer 포함 [CN-...] 인용, sc_score ≥ 0.67 |
| Unit | hits 0건 | "검색 결과 없음" |
| Unit | LLM 3회 모두 다른 답변 (SC fail) | sc_passed=false, warning 부착 |
| Edge | hits 가 query 무관 (off-topic) | answer "관련 정보 없음" |

## 13. 모니터링

- KPI:
  - 평균 sc_score ≥ 0.70
  - sc_pass 비율 ≥ 80%
  - 평균 latency ≤ 8초 (4 calls × 2초)
- token: ₩400/일

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/answer_agent.py` (신규 P6, `/gen-search` handler 분리)
- 현재 router.py 의 `gen_search` handler 가 stub — 본 agent 로 위임

### Changelog

- **v1 (제안, P6)** — SC ×3 + judge agreement
- **v2 (제안)** — RAG-Fusion (query rephrase × 3) 추가 검토
