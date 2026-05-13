# PeerComparisonAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `PeerComparisonAgent` |
| **Supervisor** | Analysis |
| **상태** | 🟡 backend fixture (`GET /api/monitoring/{peerId}/strategy`, `GET /api/monitoring/comparison`), axis-ai 신규 |
| **Trigger** | User Peer+ 페이지에서 peer 선택 시 |

## 2. 책임

**한 줄**: 선택 peer 의 최근 카드 + IR + 키워드 vs **SK AX (자사)** 의 카드/포지션을 비교하여 **차별 시사점 3~5건** + 전략 라벨 자동 추출.

**구체적**:

1. peer 의 최근 30일 card_news + 재무 IR pack 수집
2. SK AX (sk_ax 자사) 의 동일 영역 카드 + 컨텍스트 수집
3. LLM 비교 분석 — 차별점 / 강점 / 약점 / 협력 가능성
4. 전략 라벨 (LLM, 5종 중 1) — "Aggressive Expansion" / "Defensive Hold" / "Tech Pivot" / "Customer Lock-in" / "Cost Leadership"
5. 결과 cache (peer_id × week 키, 7일 TTL)

## 3. 책임 NOT

- IR 데이터 추출 — FinancialLinkerAgent + IRParserAgent
- 워드클라우드 — PeerWordCloudAgent
- 시각화 — frontend Peer+ 페이지

## 4. 입력 스펙

```python
class PeerComparisonInput(TypedDict):
    peer_id: str
    window_days: int            # 기본 30
    focus_sector: str | None    # 옵션 (특정 sector 만 비교)
```

## 5. 출력 스펙

```python
class PeerComparisonOutput(TypedDict):
    peer_id: str
    strategy_label: str                # "Aggressive Expansion" 등
    differentiators: list[dict]        # 3~5건 [{aspect, peer_position, skax_position, opportunity}]
    strengths_of_peer: list[str]
    weaknesses_of_peer: list[str]
    collaboration_potential: list[str]  # 협력 가능 영역
    confidence: float
    provenance: dict
    sources: list[str]                  # 사용된 card_ids
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

### 6.2 LLM Prompt (gpt-4o, 구조화 출력)

```text
당신은 SK AX 사업전략팀의 경쟁사 분석가다. {peer_id} 와 SK AX 의 최근 30일 동향을 비교하라.

[{peer_id} 동향 — N건]
{peer_cards_summary}
[{peer_id} 재무 segment (분기)]
{peer_ir}

[SK AX 동향 (자사)]
{skax_cards_summary}

[분석 요청]
1. **전략 라벨**: peer 의 전략 방향을 5종 중 하나로 분류
   - Aggressive Expansion | Defensive Hold | Tech Pivot | Customer Lock-in | Cost Leadership
2. **차별점 3~5건**: 동일 영역에서 peer 와 SK AX 의 포지션 차이
3. **Peer 의 강점**: 본받을 / 위협 받을 영역
4. **Peer 의 약점**: SK AX 가 공략할 수 있는 영역
5. **협력 가능성**: 경쟁이 아닌 협력 영역

JSON 출력:
{
  "strategy_label": "Aggressive Expansion",
  "differentiators": [
    {"aspect": "AI 플랫폼 차별점",
     "peer_position": "...",
     "skax_position": "...",
     "opportunity": "..."}
  ],
  "strengths_of_peer": ["..."],
  "weaknesses_of_peer": ["..."],
  "collaboration_potential": ["..."],
  "confidence": 0.0~1.0,
  "sources_used": ["CN-..."]
}
```

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
