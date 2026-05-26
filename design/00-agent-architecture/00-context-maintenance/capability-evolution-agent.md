# CapabilityEvolutionAgent

## Status

- layer: Context Maintenance Layer
- type: Agent
- code: `src/agents/context/capability_evolution_agent.py`
- status: implemented / needs contract stabilization

## Purpose

피어사별 capability 변화성을 추적한다. 최근 business signal을 시계열로 묶어,
어떤 사업 영역과 역량이 강화/약화/전환되고 있는지 context narrative로 만든다.

## Call Input

- `peer_id`
- lookback period, default 4 quarters
- optional `business_area`

## DB Read

- `raw_article_business_signals`
- optional `raw_articles` for source metadata

## Return Output

```json
{
  "peer_id": "samsung_sds",
  "windows": [
    {
      "period": "2026Q1",
      "business_area": "ai_ax",
      "narrative": "AI AX 관련 사업 신호가 증가했다.",
      "delta_intensity": 0.42,
      "confidence": 0.76,
      "evidence_signal_ids": []
    }
  ]
}
```

## DB Write

- `peer_companies.peer_plus_payload["capability_evolution"]`

## Does

- `peer_id x business_area x period x signal_type` 단위로 신호를 압축한다.
- confidence 높은 evidence 중심으로 LLM 입력을 제한한다.
- AnalysisContextBuilder가 바로 사용할 수 있는 compact window를 만든다.

## Does Not

- SK AX 대응 전략을 만들지 않는다.
- 개별 카드의 시사점을 만들지 않는다.
- raw signal row를 수정하지 않는다.

## Failure Policy

- 실패 시 기존 `capability_evolution` payload를 유지한다.
- 신규 peer에 대해 실패하면 빈 windows와 error metadata를 반환한다.

## Open Questions

- refresh cadence: weekly vs monthly.
- `business_area` taxonomy를 `config/sectors.py`와 어떻게 맞출지 결정 필요.
