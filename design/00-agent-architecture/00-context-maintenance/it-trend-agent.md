# ITTrendAgent

## Status

- layer: Context Maintenance Layer
- type: Agent
- code: `src/agents/it_trend_agent.py`
- status: implemented / storage target open

## Purpose

SPRi, BCG, Gartner, 시장 리포트, 산업 브리핑, 글로벌 회사별 뉴스룸 분석 결과를 바탕으로
글로벌 IT 트렌드와 산업 흐름을 `TrendContext`로 정리한다.

## Call Input

- trend report raw articles
- global newsroom `IntegratedIssue`
- global newsroom `AnalysisResult`
- optional previous TrendContext
- optional reference issue results

## DB Read

- `raw_articles`
- `raw_article_parse_results`
- `card_news` or analysis package payloads for global newsroom reuse
- previous trend storage, pending decision

## Return Output

```json
{
  "period": "2026-W22",
  "trend_summary": "",
  "trend_lines": [],
  "signals": [],
  "source_groups": [],
  "sources": [],
  "reference_issue_ids": []
}
```

## DB Write

Target pending:

- preferred: `global_industry_trends`
- fallback: existing JSONB/legacy storage

## Does

- 글로벌/산업 자료에서 반복되는 IT trend를 추출한다.
- AnalysisAgent가 참고할 배경 context를 만든다.
- 해외 뉴스룸 이슈 분석 결과를 trend source로 재활용한다.

## Does Not

- peer 개별 이슈 카드뉴스를 만들지 않는다.
- SK AX 대응 action을 만들지 않는다.
- source article 자체의 relevance/dedup/classification을 수정하지 않는다.

## Failure Policy

- 이전 TrendContext가 있으면 유지한다.
- 이전 context가 없으면 empty TrendContext를 반환하고 해당 주기 trend를 skip한다.

## Open Questions

- `global_industry_trends` table schema.
- Gartner 등 유료/제한 자료의 저장/인용 정책.
- trend context를 모든 분석에 넣을지, source family가 trend/global일 때만 넣을지.
