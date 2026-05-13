# MixerAnalysisAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `MixerAnalysisAgent` |
| **Supervisor** | Analysis |
| **상태** | 🟡 backend fixture (`POST /api/mixer`, `GET /api/mixer/options`, `POST /api/mixer/{id}/share`), axis-ai 신규 |
| **Trigger** | User Mixer 페이지에서 "Generate" 클릭 |

## 2. 책임

**한 줄**: 2~20 card_news 조합 + 입력 비율 (peer/industry/keyword) 로부터 **cross-card 신호** + **6축 radar 분포** + **연결 관계 (인과/유사)** 추출.

**구체적**:

1. 카드 조합 정렬 — exposure_score / event_type / sector 다양성 우선
2. 반복 신호 추출 (LLM) — 2개 이상 카드에 공통 주제/키워드 식별
3. 연결 관계 (LLM) — 카드 쌍 간 cause/effect, similarity, contrast 라벨링
4. 6축 radar score 산식 — Peer Strategic Shift / Tech Investment / Market Position / Partnership Momentum / Regulatory Risk / Talent Movement
5. 종합 인사이트 1문장 + bullet 3개

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

class MixerAnalysisOutput(TypedDict):
    mix_id: str
    insight: str               # 1문장 종합
    bullet_signals: list[str]  # 3 핵심 신호
    radar_axes: list[RadarAxis]
    connections: list[Connection]
    confidence: float
    sources_used: list[str]
    provenance: dict
```

frontend `POST /api/mixer` 응답.

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

### 6.2 LLM Prompt (gpt-4o)

```text
다음 카드 뉴스 {N}개를 함께 보고 분석하라.

[카드 목록]
{cards_summary}

[사용자 입력 비율]
- Peer: {ratios.peer}
- Industry: {ratios.industry}
- Keyword: {ratios.keyword}

[분석 요청]
1. **종합 인사이트** (1문장, 50자 이내): N건 카드를 관통하는 단일 주제
2. **핵심 신호 (bullet 3)**: 반복 등장하는 패턴 / 공통점
3. **연결 관계**: 카드 쌍 간 관계 라벨링 (cause, effect, similar, contrast, reinforce). 최대 10 쌍.

JSON 반환:
{
  "insight": "...",
  "bullet_signals": ["...", "...", "..."],
  "connections": [
    {"source_card_id": "CN-...", "target_card_id": "CN-...", "label": "cause", "weight": 0.0~1.0}
  ],
  "confidence": 0.0~1.0
}
```

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
